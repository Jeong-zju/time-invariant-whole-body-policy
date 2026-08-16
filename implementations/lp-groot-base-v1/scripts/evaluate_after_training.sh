#!/usr/bin/env bash
set -euo pipefail

PROJECT=/workspace/gr00t-n16-stackbowls-v1/lp-groot-base-v1
GROOT=/workspace/gr00t-n16-stackbowls-v1/Isaac-GR00T
PYTHON="$GROOT/.venv/bin/python"
DATASET=/workspace/gr00t-n16-stackbowls-v1/datasets/robocasa365-NavigateKitchen-v3
TRAIN_ROOT="$PROJECT/outputs/lp_groot_base_v1_full_2gpu_20260815/lp_groot_base_v1_full_2gpu_20260815"
EVAL_ROOT="$PROJECT/outputs/lp_groot_base_v1_auto_eval_20260815"

mkdir -p "$EVAL_ROOT"
export PYTHONPATH="$PROJECT/src:$GROOT"
export LP_LABEL_CACHE="$PROJECT/outputs/navigatekitchen_lp_cache_v1"

cd "$GROOT"
"$PYTHON" "$PROJECT/scripts/validate_oracle.py" \
  --dataset-root "$DATASET" \
  --output-dir "$EVAL_ROOT/offline_oracle"

"$PYTHON" "$PROJECT/scripts/calibrate_base_controller.py" \
  --dataset-root "$DATASET" \
  --output "$EVAL_ROOT/base_controller_calibration.json"

LATEST_CHECKPOINT="$(find "$TRAIN_ROOT" -maxdepth 1 -type d -name 'checkpoint-*' -print | sort -V | tail -n 1)"
if [[ -z "$LATEST_CHECKPOINT" ]]; then
  echo "No checkpoint found below $TRAIN_ROOT" >&2
  exit 1
fi

"$PYTHON" "$PROJECT/scripts/evaluate_lp_robocasa.py" \
  --model-path "$LATEST_CHECKPOINT" \
  --output-dir "$EVAL_ROOT" \
  --calibration "$EVAL_ROOT/base_controller_calibration.json" \
  --episodes 10 \
  --seeds 0 1 2 3 4 5 6 7 8 9

touch "$EVAL_ROOT/EVALUATION_COMPLETE"
