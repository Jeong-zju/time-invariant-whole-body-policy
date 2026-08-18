#!/usr/bin/env bash
set -euo pipefail

GROOT=${GROOT:-/workspace/grootn16}
IMPL=${IMPL:-/workspace/time-invariant-whole-body-policy/implementations/lp-groot-wholebody-v1}
RUN_ROOT=${RUN_ROOT:-/workspace/lpwb-run}
DATA_ROOT="$RUN_ROOT/data/robocasa365"
DATASETS=(
  "$DATA_ROOT/NavigateKitchen/lerobot"
  "$DATA_ROOT/PickPlaceCounterToStove/lerobot"
  "$DATA_ROOT/DeliverStraw/lerobot"
)
DATA_ARGS=()
for dataset in "${DATASETS[@]}"; do DATA_ARGS+=(--dataset "$dataset"); done

export PYTHONPATH="$IMPL/src:$IMPL/gr00t_patch:$GROOT"
mkdir -p "$RUN_ROOT/label_stats" "$RUN_ROOT/validation"
"$GROOT/.venv/bin/python" "$IMPL/scripts/build_label_stats.py" \
  "${DATA_ARGS[@]}" \
  --output-dir "$RUN_ROOT/label_stats" \
  --seed 20260818 \
  --train-fraction 0.9 \
  --samples-per-episode 32
"$GROOT/.venv/bin/python" "$IMPL/scripts/validate_labels.py" \
  "${DATA_ARGS[@]}" \
  --output-dir "$RUN_ROOT/validation" \
  --seed 20260818
cd "$IMPL"
"$GROOT/.venv/bin/pytest" -q tests
touch "$RUN_ROOT/LABEL_GATES_COMPLETE"
