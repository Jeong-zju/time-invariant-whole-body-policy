#!/usr/bin/env bash
set -euo pipefail

PROJECT=/workspace/lp-act-v1/robocasa-lp-act-v1
DATA=/workspace/lp-act-v1/robocasa/datasets/v1.0/pretrain/composite/LineUpCondiments/20250731/lerobot
STATS=/workspace/lp-act-v1/outputs/robocasa_lineup_base_only_stats_v1/stats.json
CHECKPOINT=/workspace/lp-act-v1/outputs/robocasa_act_lineup_b32_s50000_seed0_v2_cache/checkpoint_0010000.pt
cd "$PROJECT"
export PYTHONPATH="$PROJECT/src:/workspace/lp-act-v1/lerobot/src:/workspace/lp-act-v1/robocasa"
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

exec /venv/main/bin/python -m robocasa_lp_act.rollout \
  --mode standard --checkpoint "$CHECKPOINT" --stats "$STATS" \
  --data-root "$DATA" --episode 0 --max-steps 1 \
  --video /workspace/lp-act-v1/outputs/robocasa_rollout_smoke_act_10k_ep0_v1.mp4 \
  --video-skip 1 --device cuda:0
