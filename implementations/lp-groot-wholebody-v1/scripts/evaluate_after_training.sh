#!/usr/bin/env bash
set -euo pipefail

GROOT=${GROOT:-/workspace/grootn16}
IMPL=${IMPL:-/workspace/time-invariant-whole-body-policy/implementations/lp-groot-wholebody-v1}
RUN_ROOT=${RUN_ROOT:-/workspace/lpwb-run}
DATA_ROOT="$RUN_ROOT/data/robocasa365"
DATA_ARGS=(
  --dataset "$DATA_ROOT/NavigateKitchen/lerobot"
  --dataset "$DATA_ROOT/PickPlaceCounterToStove/lerobot"
  --dataset "$DATA_ROOT/DeliverStraw/lerobot"
)
export PYTHONPATH="$IMPL/src:$IMPL/gr00t_patch:$GROOT"
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libnccl.so.2.30.7${LD_PRELOAD:+:$LD_PRELOAD}

while [[ ! -f "$RUN_ROOT/outputs/b0_groot_command_seed20260818/TRAINING_COMPLETE" || \
         ! -f "$RUN_ROOT/outputs/b2_path_time_seed20260818/TRAINING_COMPLETE" ]]; do
  if supervisorctl status lpwb-b0 | grep -Eq 'EXITED|FATAL|BACKOFF'; then
    echo "B0 exited before TRAINING_COMPLETE" >&2
    exit 1
  fi
  if supervisorctl status lpwb-b2 | grep -Eq 'EXITED|FATAL|BACKOFF'; then
    echo "B2 exited before TRAINING_COMPLETE" >&2
    exit 1
  fi
  sleep 60
done

mkdir -p "$RUN_ROOT/evaluation"
CUDA_VISIBLE_DEVICES=0 "$GROOT/.venv/bin/python" "$IMPL/scripts/evaluate_heldout.py" \
  --method b0 \
  --checkpoint "$RUN_ROOT/outputs/b0_groot_command_seed20260818/checkpoint-3000" \
  "${DATA_ARGS[@]}" \
  --output-dir "$RUN_ROOT/evaluation" \
  --samples-per-task 10 >"$RUN_ROOT/logs/eval-b0.log" 2>"$RUN_ROOT/logs/eval-b0.err.log" &
B0_PID=$!
CUDA_VISIBLE_DEVICES=2 "$GROOT/.venv/bin/python" "$IMPL/scripts/evaluate_heldout.py" \
  --method b2 \
  --checkpoint "$RUN_ROOT/outputs/b2_path_time_seed20260818/checkpoint-3000" \
  "${DATA_ARGS[@]}" \
  --output-dir "$RUN_ROOT/evaluation" \
  --samples-per-task 10 >"$RUN_ROOT/logs/eval-b2.log" 2>"$RUN_ROOT/logs/eval-b2.err.log" &
B2_PID=$!
wait "$B0_PID"
wait "$B2_PID"
touch "$RUN_ROOT/EVALUATION_COMPLETE"
