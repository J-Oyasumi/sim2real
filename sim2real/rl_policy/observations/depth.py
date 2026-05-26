"""Depth camera observation via ZMQ subscriber.

Receives depth frames published by an external process (sim depth renderer
or a real camera capture process) and exposes them to the policy with the
shape the ONNX model expects: (1, 1, H, W).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import zmq
from loguru import logger
from rich import print

from sim2real.rl_policy.observations.base import Observation
from sim2real.utils.common import PORTS, DepthImageMessage


class warp_depth_camera(Observation):
    """Subscribe to depth frames via ZMQ.

    Matches the ONNX model's "depth" input tensor (shape (1, 1, H, W)).
    All training-only kwargs (camera_poses, dynamic_meshes, scene_meshes,
    noise_std, episodic_noise_range, delay_range, fov_randomization, ...) are
    accepted and ignored — the publisher is responsible for delivering frames
    in the correct frame, intrinsics, and noise distribution.
    """

    def __init__(
        self,
        width: int = 106,
        height: int = 60,
        min_depth: float = 0.1,
        max_depth: float | None = None,
        far_clip: float = 2.0,
        show: bool = False,
        host: str = "127.0.0.1",
        port: int | None = None,
        normalization: dict | None = None,
        # Training-only kwargs we accept and ignore
        camera_poses: Any = None,
        horizontal_fov: float = 87.0,
        noise_std: float = 0.0,
        episodic_noise_range: Any = None,
        delay_range: Any = None,
        dynamic_meshes: Any = None,
        scene_meshes: Any = None,
        fov_randomization: Any = None,
        calculate_depth: Any = None,
        euler_frame_rot_deg: Any = None,
        **unused,
    ) -> None:
        super().__init__(**unused)
        self.width = int(width)
        self.height = int(height)
        self.min_depth = float(min_depth)
        self.max_depth = float(max_depth) if max_depth is not None else float(far_clip)
        self.show = bool(show)
        self._frame_count = 0

        # ONNX expects (1, 1, H, W). Preallocate buffer.
        self.depth_buf = np.zeros((1, 1, self.height, self.width), dtype=np.float32)

        self.normalization = normalization  # set in carry.py if applicable
        self._normalize_fn = self._build_normalizer(normalization)

        # ZMQ subscriber
        depth_port = int(port) if port is not None else PORTS["depth_image"]
        self.zmq_context = zmq.Context.instance()
        self.depth_socket = self.zmq_context.socket(zmq.SUB)
        self.depth_socket.setsockopt(zmq.SUBSCRIBE, b"")
        self.depth_socket.setsockopt(zmq.CONFLATE, 1)
        self.depth_socket.setsockopt(zmq.RCVTIMEO, 1000)  # 1s for the first frame
        self.depth_socket.connect(f"tcp://{host}:{depth_port}")
        print(f"[cyan]warp_depth_camera[/cyan]: subscribing tcp://{host}:{depth_port}")

        # Block on first frame so the first compute() has valid data.
        print(f"[yellow]warp_depth_camera[/yellow]: waiting for first depth frame...")
        data = self.depth_socket.recv()
        msg = DepthImageMessage.from_bytes(data)
        if msg.image.shape != (self.height, self.width):
            raise ValueError(
                f"Received depth shape {msg.image.shape} does not match expected "
                f"({self.height}, {self.width})"
            )
        self._apply_frame(msg.image)
        logger.info(
            "warp_depth_camera ready: shape={}, range=[{:.3f}, {:.3f}]",
            msg.image.shape,
            float(msg.image.min()),
            float(msg.image.max()),
        )

        # Subsequent reads are non-blocking; reuse last frame if none arrived.
        self.depth_socket.setsockopt(zmq.RCVTIMEO, 0)

    def _build_normalizer(self, cfg: dict | None):
        if cfg is None:
            return None
        scheme = str(cfg.get("scheme", "")).lower().strip()
        if scheme == "inverse_minmax":
            d_min = float(cfg.get("depth_min", self.min_depth))
            d_max = float(cfg.get("depth_max", self.max_depth))
            inv_min = 1.0 / d_min
            inv_max = 1.0 / d_max
            denom = inv_min - inv_max

            def _normalize(d: np.ndarray) -> np.ndarray:
                # Clip first to avoid 1/0 or out-of-range values
                d_clipped = np.clip(d, d_min, d_max)
                return (1.0 / d_clipped - inv_max) / denom

            return _normalize
        raise ValueError(f"Unsupported depth_normalization scheme: {scheme}")

    def _apply_frame(self, raw: np.ndarray) -> None:
        clipped = np.clip(raw, self.min_depth, self.max_depth).astype(np.float32, copy=False)
        if self._normalize_fn is not None:
            clipped = self._normalize_fn(clipped).astype(np.float32, copy=False)
        self.depth_buf[0, 0] = clipped

    def update(self, data) -> None:
        # Drain to latest frame (CONFLATE keeps only the newest, but the call
        # is still cheap and explicit). zmq.Again means no new frame this
        # tick — reuse the last buffer instead of blocking the policy.
        try:
            payload = self.depth_socket.recv(flags=zmq.DONTWAIT)
        except zmq.Again:
            return

        msg = DepthImageMessage.from_bytes(payload)
        if msg.image.shape != (self.height, self.width):
            logger.warning(
                "Depth frame shape {} != expected ({}, {}), dropping",
                msg.image.shape,
                self.height,
                self.width,
            )
            return
        self._apply_frame(msg.image)
        self._frame_count += 1

        if self.show:
            self._render_vis()

    def _render_vis(self) -> None:
        # Throttle vis to ~25 fps to keep cv2 from stealing policy CPU.
        if self._frame_count % 2 != 0:
            return
        import cv2  # type: ignore[import-untyped]

        depth_m = self.depth_buf[0, 0]
        lo, hi = self.min_depth, self.max_depth
        # Inverse map so close objects are bright (matches intuition)
        gray = np.clip(1.0 - (depth_m - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
        gray_u8 = (gray * 255).astype(np.uint8)
        colored = cv2.applyColorMap(gray_u8, cv2.COLORMAP_TURBO)

        # Upscale 5x with nearest so individual policy pixels stay crisp
        big = cv2.resize(
            colored,
            (self.width * 5, self.height * 5),
            interpolation=cv2.INTER_NEAREST,
        )

        # Overlay range + per-frame stats
        cur_lo = float(depth_m.min())
        cur_hi = float(depth_m.max())
        cur_mean = float(depth_m.mean())
        text = (
            f"clip=[{lo:.2f}, {hi:.2f}]m  "
            f"frame=[{cur_lo:.2f}, {cur_hi:.2f}]m  "
            f"mean={cur_mean:.2f}m  "
            f"#{self._frame_count}"
        )
        cv2.putText(big, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(big, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.imshow("policy.depth", big)
        cv2.waitKey(1)

    def compute(self) -> np.ndarray:
        return self.depth_buf
