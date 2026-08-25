#!/usr/bin/env bash
set -euo pipefail

ROOT=/workspace/act-exact-output-sweep-navigate-kitchen-20260824-v8

condition_dir() {
  local method=$1 hz=$2
  echo "$ROOT/evaluation/${method}_request${hz}hz_control50hz_checkpoint50000_seed20260818_count30"
}

wait_stage() {
  local hz=$1
  while true; do
    local terminal=0
    for method in act geometric; do
      local dir service state
      dir=$(condition_dir "$method" "$hz")
      service="exact-output-${method}${hz}"
      if [ -f "$dir/EVALUATION_COMPLETE" ]; then
        terminal=$((terminal + 1))
        continue
      fi
      state=$(supervisorctl status "$service" 2>/dev/null | awk '{print $2}' || true)
      if [ -f "$dir/EVALUATION_FAILED" ] && [ "$state" != RUNNING ] && [ "$state" != STARTING ]; then
        terminal=$((terminal + 1))
      fi
    done
    if [ "$terminal" -eq 2 ]; then return 0; fi
    sleep 30
  done
}

start_stage() {
  local hz=$1
  supervisorctl start "exact-output-act${hz}" "exact-output-geometric${hz}" || true
  wait_stage "$hz"
}

wait_stage 20
start_stage 10
start_stage 50

failed=0
for hz in 10 20 50; do
  for method in act geometric; do
    dir=$(condition_dir "$method" "$hz")
    if [ ! -f "$dir/EVALUATION_COMPLETE" ]; then failed=1; fi
  done
done
if [ "$failed" -eq 0 ]; then
  touch "$ROOT/SWEEP_COMPLETE"
  echo SWEEP_COMPLETE
else
  touch "$ROOT/SWEEP_FAILED"
  echo SWEEP_FAILED
fi
