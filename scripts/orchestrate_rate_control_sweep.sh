#!/usr/bin/env bash
set -euo pipefail

export RUN_ROOT=${RUN_ROOT:-/workspace/act-path-pointer-rate-sweep-navigate-kitchen-20260824-v1}
SOURCE=/workspace/act-base-only-navigate-kitchen-scratch-20260823/source/lp-act-v1
mkdir -p "$RUN_ROOT/logs"

run_wave() {
  local pids=()
  while [ "$#" -gt 0 ]; do
    method=$1; rate=$2; gpu=$3; port=$4; shift 4
    "$SOURCE/scripts/run_navigate_rate_control_condition.sh" "$method" "$rate" "$gpu" "$port" \
      >"$RUN_ROOT/logs/${method}-rate${rate}.log" \
      2>"$RUN_ROOT/logs/${method}-rate${rate}.err.log" &
    pids+=("$!")
  done
  rc=0
  for pid in "${pids[@]}"; do wait "$pid" || rc=1; done
  return "$rc"
}

run_wave act 0.5 0 5630 point 0.5 1 5631 act 1.0 2 5632 point 1.0 3 5633
run_wave act 1.5 0 5634 point 1.5 1 5635

export PYTHONPATH="/workspace/grootn17:$SOURCE/src:/workspace/act-navigate-kitchen-scratch-20260821/source/lerobot/src"
/workspace/grootn16/.venv/bin/python -m robocasa_act_navigate.analyze_rate_control_sweep \
  --root "$RUN_ROOT" >"$RUN_ROOT/logs/analysis.log" 2>"$RUN_ROOT/logs/analysis.err.log"
echo RATE_CONTROL_PIPELINE_COMPLETE
