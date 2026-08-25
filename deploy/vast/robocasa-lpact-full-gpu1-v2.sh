#!/usr/bin/env bash
set -euo pipefail

PROJECT=/workspace/lp-act-v1/robocasa-lp-act-v1
DATA=/workspace/lp-act-v1/robocasa/datasets/v1.0/pretrain/composite/LineUpCondiments/20250731/lerobot
STATS=/workspace/lp-act-v1/outputs/robocasa_lineup_base_only_stats_v1
CACHE=/workspace/lp-act-v1/cache/lineup_condiments_rgb256_v1.uint8
RESUME=/workspace/lp-act-v1/outputs/robocasa_lpact_base_only_lineup_b32_s50000_seed0_v1/checkpoint_0005000.pt
cd "$PROJECT"
export PYTHONPATH="$PROJECT/src:/workspace/lp-act-v1/lerobot/src"
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=1

exec /venv/main/bin/python -m robocasa_lp_act.train \
  --mode lp --data-root "$DATA" --frame-cache "$CACHE" \
  --stats "$STATS/stats.json" --metric-config "$STATS/metric_config.json" \
  --resume-checkpoint "$RESUME" \
  --output-dir /workspace/lp-act-v1/outputs/robocasa_lpact_base_only_lineup_b32_s50000_seed0_v2_cache \
  --device cuda:0 --steps 50000 --batch-size 32 --num-workers 8 \
  --log-every 20 --checkpoint-every 5000 --validate-every 1000 --validation-samples 256
