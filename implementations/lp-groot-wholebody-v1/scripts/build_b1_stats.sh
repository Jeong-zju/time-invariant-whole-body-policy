#!/usr/bin/env bash
set -euo pipefail

GROOT=${GROOT:-/workspace/grootn16}
IMPL=${IMPL:-/workspace/time-invariant-whole-body-policy/implementations/lp-groot-wholebody-v1}
RUN_ROOT=${RUN_ROOT:-/workspace/lpwb-run}
export PYTHONPATH="$IMPL/src:$GROOT"

rm -f "$RUN_ROOT/B1_LABEL_STATS_COMPLETE"
"$GROOT/.venv/bin/python" "$IMPL/scripts/build_label_stats.py" \
  --method b1 \
  --dataset "$RUN_ROOT/data/robocasa365/NavigateKitchen/lerobot" \
  --dataset "$RUN_ROOT/data/robocasa365/PickPlaceCounterToStove/lerobot" \
  --dataset "$RUN_ROOT/data/robocasa365/DeliverStraw/lerobot" \
  --output-dir "$RUN_ROOT/label_stats_b1" \
  --seed 20260818 \
  --samples-per-episode 128
touch "$RUN_ROOT/B1_LABEL_STATS_COMPLETE"
