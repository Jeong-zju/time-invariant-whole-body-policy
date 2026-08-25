#!/usr/bin/env bash
set -euo pipefail

PROJECT=/workspace/lp-act-v1/robocasa-lp-act-v1
DATA=/workspace/lp-act-v1/robocasa/datasets/v1.0/pretrain/composite/LineUpCondiments/20250731/lerobot
OUT=/workspace/lp-act-v1/outputs/robocasa_lineup_base_only_stats_v1
mkdir -p "$OUT"
cd "$PROJECT"
export PYTHONPATH="$PROJECT/src:/workspace/lp-act-v1/lerobot/src"
export PYTHONUNBUFFERED=1

exec /venv/main/bin/python -m robocasa_lp_act.stats \
  --data-root "$DATA" \
  --output "$OUT/stats.json" \
  --metric-output "$OUT/metric_config.json"
