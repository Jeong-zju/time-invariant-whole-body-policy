#!/usr/bin/env bash
set -euo pipefail

RUN_ROOT=${RUN_ROOT:-/workspace/lpwb-run}
while true; do
  now=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  b0_status=$(supervisorctl status lpwb-b0 || true)
  b2_status=$(supervisorctl status lpwb-b2 || true)
  b0_ckpt=$(find "$RUN_ROOT/outputs/b0_groot_command_seed20260818" -maxdepth 1 -type d -name 'checkpoint-*' -printf '%f\n' 2>/dev/null | sort -V | tail -n 1)
  b2_ckpt=$(find "$RUN_ROOT/outputs/b2_path_time_seed20260818" -maxdepth 1 -type d -name 'checkpoint-*' -printf '%f\n' 2>/dev/null | sort -V | tail -n 1)
  echo "$now | $b0_status | latest=${b0_ckpt:-none} | $b2_status | latest=${b2_ckpt:-none}"
  if grep -Eqi 'Traceback|CUDA out of memory|NCCL.*(error|failed)|(^|[^a-z])nan([^a-z]|$)' \
      "$RUN_ROOT/logs/b0.err.log" "$RUN_ROOT/logs/b2.err.log"; then
    echo "FATAL_PATTERN_DETECTED"
    exit 1
  fi
  if [[ -f "$RUN_ROOT/EVALUATION_COMPLETE" ]]; then
    echo "PIPELINE_COMPLETE"
    exit 0
  fi
  sleep 60
done
