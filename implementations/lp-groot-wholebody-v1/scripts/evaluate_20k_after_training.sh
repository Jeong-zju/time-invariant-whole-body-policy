#!/usr/bin/env bash
set -euo pipefail

GROOT=${GROOT:-/workspace/grootn16}
IMPL=${IMPL:-/workspace/time-invariant-whole-body-policy/implementations/lp-groot-wholebody-v1}
RUN_ROOT=${RUN_ROOT:-/workspace/lpwb-run}
EVAL_DIR="$RUN_ROOT/evaluation_20k_seed20260818"
B0_DIR="$RUN_ROOT/outputs/b0_groot_command_seed20260818_20k_bs16_20260818"
B1_DIR="$RUN_ROOT/outputs/b1_pose_time_seed20260818_20k_bs16_20260818"
B2_DIR="$RUN_ROOT/outputs/b2_path_time_seed20260818_20k_bs16_20260818"
DATA_ARGS=(
  --dataset "$RUN_ROOT/data/robocasa365/NavigateKitchen/lerobot"
  --dataset "$RUN_ROOT/data/robocasa365/PickPlaceCounterToStove/lerobot"
  --dataset "$RUN_ROOT/data/robocasa365/DeliverStraw/lerobot"
)

export PYTHONPATH="$IMPL/src:$IMPL/gr00t_patch:$GROOT"
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libnccl.so.2.30.7${LD_PRELOAD:+:$LD_PRELOAD}
mkdir -p "$EVAL_DIR" "$RUN_ROOT/logs"

while [[ ! -f "$RUN_ROOT/TRAINING_20K_PIPELINE_COMPLETE" ]]; do
  monitor_status=$(supervisorctl status lpwb-20k-monitor 2>/dev/null || true)
  if grep -Eq 'FATAL|BACKOFF|UNKNOWN' <<<"$monitor_status"; then
    echo "20k monitor failed before training completion: $monitor_status" >&2
    exit 1
  fi
  sleep 60
done

for checkpoint in \
  "$B0_DIR/checkpoint-20000" \
  "$B1_DIR/checkpoint-20000" \
  "$B2_DIR/checkpoint-20000"; do
  if [[ ! -f "$checkpoint/model.safetensors.index.json" ]]; then
    echo "inference checkpoint is incomplete: $checkpoint" >&2
    exit 1
  fi
done

CALIBRATION="$EVAL_DIR/execution_calibration.json"
if [[ ! -f "$CALIBRATION" ]]; then
  "$GROOT/.venv/bin/python" "$IMPL/scripts/calibrate_execution.py" \
    "${DATA_ARGS[@]}" \
    --output "$CALIBRATION" \
    --seed 20260818 \
    --train-fraction 0.9 \
    --max-episodes-per-task 64 \
    --max-lag 4 \
    >"$RUN_ROOT/logs/calibration-20k.log" \
    2>"$RUN_ROOT/logs/calibration-20k.err.log"
fi

run_heldout() {
  local method=$1
  local gpu=$2
  local checkpoint=$3
  LPWB_METHOD="$method" CUDA_VISIBLE_DEVICES="$gpu" \
    "$GROOT/.venv/bin/python" "$IMPL/scripts/evaluate_heldout.py" \
      --method "$method" \
      --checkpoint "$checkpoint" \
      "${DATA_ARGS[@]}" \
      --output-dir "$EVAL_DIR" \
      --seed 20260818 \
      --samples-per-task 30 \
      >"$RUN_ROOT/logs/eval-${method}-20k.log" \
      2>"$RUN_ROOT/logs/eval-${method}-20k.err.log"
}

run_heldout b0 0 "$B0_DIR/checkpoint-20000" &
B0_PID=$!
run_heldout b1 1 "$B1_DIR/checkpoint-20000" &
B1_PID=$!
run_heldout b2 2 "$B2_DIR/checkpoint-20000" &
B2_PID=$!

status=0
wait "$B0_PID" || status=1
wait "$B1_PID" || status=1
wait "$B2_PID" || status=1
if [[ "$status" != 0 ]]; then
  echo "one or more 20k held-out evaluations failed" >&2
  exit 1
fi

for report in "$EVAL_DIR"/{b0,b1,b2}_heldout.json; do
  test -s "$report"
done
touch "$EVAL_DIR/OPEN_LOOP_EVALUATION_COMPLETE"
