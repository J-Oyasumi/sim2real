"""Real-robot depth source publisher for the carry policy.

Publishes downscaled depth frames to the `depth_image` ZMQ port that
`sim2real.rl_policy.observations.depth.warp_depth_camera` subscribes to.

Modes:
  - realsense: capture from an Intel RealSense depth camera (D435/D455).
  - dummy:     emit a deterministic depth pattern (testing without hardware).
  - file:      replay a saved .npy depth stream (shape (N, H, W) float meters).

The publisher resizes to (H, W) matching the policy yaml (default 60x106) and
sends meters as float32 via `DepthImageMessage`.

Calibration note: extrinsics (camera_poses.front_depth in the yaml — body_link
+ offset + rotation) are physical mounting constraints. Mount the camera
matching the yaml's offset/rotation; this publisher does not transform the
image.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
import tyro
import zmq
from loguru import logger
from rich import print

from sim2real.utils.common import PORTS, DepthImageMessage


@dataclass
class Args:
    """Depth publisher."""

    source: Literal["realsense", "dummy", "file"] = "realsense"

    # Target output resolution (must match policy yaml's width/height).
    width: int = 106
    height: int = 60

    # Publish endpoint
    bind: str = "*"
    port: int = PORTS["depth_image"]

    # Publish rate (Hz). Policy is 50Hz; 30Hz is fine — obs side reuses
    # last frame between updates.
    rate_hz: float = 30.0

    # RealSense-specific
    rs_width: int = 848
    rs_height: int = 480
    rs_fps: int = 30
    # Depth clipping at capture time (meters); should match policy yaml.
    rs_min_depth: float = 0.1
    rs_max_depth: float = 2.0

    # file-mode: path to .npy with shape (N, H, W), meters
    file_path: Optional[str] = None

    # Debug: show the published frame
    show: bool = False


def _make_socket(args: Args) -> zmq.Socket:
    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.PUB)
    sock.setsockopt(zmq.SNDHWM, 1)
    sock.setsockopt(zmq.LINGER, 0)
    endpoint = f"tcp://{args.bind}:{args.port}"
    sock.bind(endpoint)
    print(f"[green]depth_publisher[/green]: bound {endpoint}")
    return sock


def _resize_depth(depth_m: np.ndarray, out_w: int, out_h: int) -> np.ndarray:
    """Resize depth (meters) to (out_h, out_w) using nearest neighbour.

    Nearest avoids smearing across discontinuities (which would corrupt
    occlusion edges that the policy relies on).
    """
    import cv2  # type: ignore[import-untyped]

    if depth_m.shape == (out_h, out_w):
        return depth_m.astype(np.float32, copy=False)
    return cv2.resize(
        depth_m.astype(np.float32, copy=False),
        (out_w, out_h),
        interpolation=cv2.INTER_NEAREST,
    )


def _show(depth_m: np.ndarray, mn: float, mx: float) -> None:
    import cv2  # type: ignore[import-untyped]

    vis = np.clip((depth_m - mn) / max(mx - mn, 1e-6) * 255.0, 0, 255).astype(np.uint8)
    cv2.imshow("depth_publisher", vis)
    cv2.waitKey(1)


# ---------------------------------------------------------------------------
# Source implementations
# ---------------------------------------------------------------------------


def _run_realsense(args: Args, sock: zmq.Socket) -> None:
    try:
        import pyrealsense2 as rs  # type: ignore[import-untyped]
    except ImportError:
        logger.error(
            "pyrealsense2 is not installed. "
            "Install it with: pip install pyrealsense2"
        )
        sys.exit(1)

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(
        rs.stream.depth, args.rs_width, args.rs_height, rs.format.z16, args.rs_fps
    )
    profile = pipeline.start(config)

    # Read scale (m per unit, typically 0.001 for D4xx).
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = float(depth_sensor.get_depth_scale())
    print(f"[cyan]realsense[/cyan]: stream {args.rs_width}x{args.rs_height}@{args.rs_fps} "
          f"depth_scale={depth_scale:.6f} m/unit")

    # Match policy yaml clipping
    clip_min, clip_max = args.rs_min_depth, args.rs_max_depth

    period = 1.0 / args.rate_hz
    next_t = time.perf_counter()
    n_sent = 0
    last_log = time.perf_counter()

    try:
        while True:
            now = time.perf_counter()
            if now < next_t:
                time.sleep(max(0.0, next_t - now))
            next_t += period

            frames = pipeline.wait_for_frames(timeout_ms=1000)
            depth_frame = frames.get_depth_frame()
            if not depth_frame:
                continue

            depth_u16 = np.asanyarray(depth_frame.get_data())
            depth_m = depth_u16.astype(np.float32) * depth_scale
            # Treat 0 as invalid → clip to max
            depth_m[depth_m <= 0.0] = clip_max
            depth_m = np.clip(depth_m, clip_min, clip_max)
            depth_small = _resize_depth(depth_m, args.width, args.height)

            msg = DepthImageMessage(depth_small)
            sock.send(msg.to_bytes())
            n_sent += 1

            if args.show:
                _show(depth_small, clip_min, clip_max)

            if now - last_log > 2.0:
                fps = n_sent / (now - last_log)
                lo = float(depth_small.min())
                hi = float(depth_small.max())
                print(f"[dim]realsense: {fps:.1f} Hz, range=[{lo:.2f}, {hi:.2f}] m[/dim]")
                n_sent = 0
                last_log = now
    finally:
        pipeline.stop()


def _run_dummy(args: Args, sock: zmq.Socket) -> None:
    """Emit a synthetic depth pattern for protocol testing."""
    period = 1.0 / args.rate_hz
    # Pattern: horizontal gradient 0.1 m → 2.0 m, with a moving stripe at 1.0 m
    base = np.tile(
        np.linspace(0.1, 2.0, args.width, dtype=np.float32)[None, :],
        (args.height, 1),
    )
    next_t = time.perf_counter()
    t0 = next_t
    n_sent = 0
    last_log = t0
    print(f"[yellow]dummy[/yellow]: publishing synthetic {args.width}x{args.height} @ {args.rate_hz} Hz")

    while True:
        now = time.perf_counter()
        if now < next_t:
            time.sleep(max(0.0, next_t - now))
        next_t += period

        frame = base.copy()
        # Add a moving vertical stripe at depth 1.0 m
        x = int((now - t0) * 20.0) % args.width
        frame[:, max(0, x - 3): x + 3] = 1.0

        msg = DepthImageMessage(frame)
        sock.send(msg.to_bytes())
        n_sent += 1

        if args.show:
            _show(frame, 0.1, 2.0)

        if now - last_log > 2.0:
            print(f"[dim]dummy: {n_sent / (now - last_log):.1f} Hz[/dim]")
            n_sent = 0
            last_log = now


def _run_file(args: Args, sock: zmq.Socket) -> None:
    if args.file_path is None:
        raise ValueError("--file_path is required for source=file")
    stream = np.load(args.file_path)
    if stream.ndim != 3:
        raise ValueError(f"Expected file shape (N, H, W), got {stream.shape}")
    print(f"[cyan]file[/cyan]: loaded {stream.shape} from {args.file_path}")

    period = 1.0 / args.rate_hz
    next_t = time.perf_counter()
    idx = 0
    while True:
        now = time.perf_counter()
        if now < next_t:
            time.sleep(max(0.0, next_t - now))
        next_t += period

        frame = stream[idx % len(stream)].astype(np.float32, copy=False)
        frame = _resize_depth(frame, args.width, args.height)
        sock.send(DepthImageMessage(frame).to_bytes())
        idx += 1

        if args.show:
            _show(frame, 0.1, 2.0)


def main(args: Args) -> None:
    sock = _make_socket(args)
    print(f"[green]depth_publisher[/green]: source={args.source}, "
          f"out=({args.height}x{args.width}), rate={args.rate_hz} Hz")
    try:
        if args.source == "realsense":
            _run_realsense(args, sock)
        elif args.source == "dummy":
            _run_dummy(args, sock)
        elif args.source == "file":
            _run_file(args, sock)
        else:
            raise ValueError(f"Unsupported source: {args.source}")
    except KeyboardInterrupt:
        print("[yellow]depth_publisher[/yellow]: interrupted")
    finally:
        sock.close(0)


if __name__ == "__main__":
    main(tyro.cli(Args))
