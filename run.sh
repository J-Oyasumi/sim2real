#!/bin/bash
# Deploy a depth-based carry policy on the real robot.
#
# Usage:
#   ./run.sh <policy_config_yaml> [extra args passed to carry.py...]
#
# Example:
#   ./run.sh checkpoints/carry_policy/mlp/8v7f8e95/policy-8v7f8e95-final.yaml
#   ./run.sh <yaml> --inference_backend onnx-cpu --controller joystick
#
# Defaults: --robot g1 --inference_backend tensorrt --controller keyboard
#           --rl_rate 50 --show_depth
# (override any of them by passing the same flag in the extra args.)
#
# This script runs the policy process only. In separate terminals also start:
#   uv run scripts/real_bridge.py --robot g1 --interface eth0 --rate 100
#   uv run scripts/depth_publisher.py --source realsense

set -euo pipefail

if [ $# -lt 1 ]; then
  echo "usage: $0 <policy_config_yaml> [extra carry.py args...]" >&2
  exit 1
fi

policy_cfg=$1
shift

if [ ! -f "$policy_cfg" ]; then
  echo "policy config not found: $policy_cfg" >&2
  exit 1
fi

uv run sim2real/rl_policy/carry.py \
  --policy_config "$policy_cfg" \
  --robot g1 \
  --inference_backend tensorrt \
  --controller keyboard \
  --rl_rate 50 \
  --show_depth \
  "$@"
