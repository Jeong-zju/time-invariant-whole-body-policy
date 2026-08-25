#!/usr/bin/env bash
set -euo pipefail

PROJECT=/workspace/lp-act-v1/robocasa-lp-act-v1
DATA=/workspace/lp-act-v1/robocasa/datasets/v1.0/pretrain/composite/LineUpCondiments/20250731/lerobot
STATS=/workspace/lp-act-v1/outputs/robocasa_lineup_base_only_stats_v1
cd "$PROJECT"
export PYTHONPATH="$PROJECT/src:/workspace/lp-act-v1/lerobot/src"
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0

exec /venv/main/bin/python -m robocasa_lp_act.train \
  --mode standard --data-root "$DATA" \
  --stats "$STATS/stats.json" --metric-config "$STATS/metric_config.json" \
  --output-dir /workspace/lp-act-v1/outputs/robocasa_act_lineup_tiny_b16_s300_seed0_v3 \
  --device cuda:0 --steps 300 --batch-size 16 --tiny-samples 16 \
  --num-workers 0 --log-every 20 --checkpoint-every 300 --validate-every 300
