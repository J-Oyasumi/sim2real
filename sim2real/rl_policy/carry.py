"""Depth-based carry policy entry point.

Targets policies trained without reference motion (no HDMI tracking inputs)
and with a depth observation alongside proprioception. Compatible with both
the new policy_config format (joint_names_simulation, body_names_simulation)
and the legacy format that ships with the hdmi-tag checkpoints
(isaac_joint_names, observation without `_target_`).
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

import numpy as np
import tyro
from loguru import logger
from rich import print

from sim2real.rl_policy.base_policy import BasePolicy, BasePolicyArgs


def _legacy_yaml_to_new(policy_config: dict) -> dict:
    """In-memory upgrade of legacy hdmi-tag yaml to the new framework schema.

    Mutations:
      - isaac_joint_names         → joint_names_simulation
      - body_names_simulation     ← []  (no body-targeted obs in carry policies)

    Depth normalization note:
      The exported carry ONNX bakes in the full depth pipeline:
        NaN/±Inf safety → Clip[0.1, 2.0] → (1/d - 1/max) / (1/min - 1/max)
      So the obs feeds RAW meters and lets ONNX do the rest. We deliberately
      do NOT inject `depth_normalization` into the obs config — that would
      double-normalize and shift the input distribution off-policy.
    """
    cfg = deepcopy(policy_config)

    if "joint_names_simulation" not in cfg and "isaac_joint_names" in cfg:
        cfg["joint_names_simulation"] = list(cfg["isaac_joint_names"])
        logger.info("yaml: aliased isaac_joint_names → joint_names_simulation")

    cfg.setdefault("body_names_simulation", [])
    return cfg


class Carry(BasePolicy):
    """Depth-based carry policy runner (no reference motion)."""

    args: "CarryArgs"

    def prepare_policy_config(self, policy_config):
        policy_config = super().prepare_policy_config(policy_config)
        policy_config = _legacy_yaml_to_new(policy_config)
        # No `motion` block → StateProcessor uses motion_backend="none"

        # CLI override: --show_depth flips the depth obs's `show` flag so the
        # policy renders the live depth view at inference time without
        # touching the checkpoint yaml.
        if self.args.show_depth:
            depth_group = policy_config.get("observation", {}).get("depth", {})
            for obs_name, obs_kwargs in depth_group.items():
                if isinstance(obs_kwargs, dict):
                    obs_kwargs["show"] = True
                    logger.info("yaml: --show_depth → observation.depth.{}.show=True", obs_name)
        return policy_config


@dataclass
class CarryArgs(BasePolicyArgs):
    """Depth-based carry policy."""
    show_depth: bool = False  # open a cv2 window with the live depth obs


if __name__ == "__main__":
    args = tyro.cli(CarryArgs)
    print(f"[green]Carry policy starting[/green]: robot={args.robot}, "
          f"backend={args.inference_backend}, controller={args.controller}")
    policy = Carry(args=args)
    policy.run()
