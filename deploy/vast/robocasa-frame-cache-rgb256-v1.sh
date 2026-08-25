#!/usr/bin/env bash
set -euo pipefail

PROJECT=/workspace/lp-act-v1/robocasa-lp-act-v1
DATA=/workspace/lp-act-v1/robocasa/datasets/v1.0/pretrain/composite/LineUpCondiments/20250731/lerobot
CACHE=/workspace/lp-act-v1/cache/lineup_condiments_rgb256_v1.uint8
cd "$PROJECT"
export PYTHONPATH="$PROJECT/src:/workspace/lp-act-v1/lerobot/src"
export PYTHONUNBUFFERED=1

exec /venv/main/bin/python -m robocasa_lp_act.prepare_frame_cache \
  --data-root "$DATA" --output "$CACHE" --workers 16
