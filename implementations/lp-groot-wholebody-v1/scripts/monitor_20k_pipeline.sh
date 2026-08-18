#!/usr/bin/env bash
set -euo pipefail

RUN_ROOT=${RUN_ROOT:-/workspace/lpwb-run}
B0_DIR="$RUN_ROOT/outputs/b0_groot_command_seed20260818_20k_bs16_20260818"
B1_DIR="$RUN_ROOT/outputs/b1_pose_time_seed20260818_20k_bs16_20260818"
B2_DIR="$RUN_ROOT/outputs/b2_path_time_seed20260818_20k_bs16_20260818"
LOG_DIR="$RUN_ROOT/logs"
mkdir -p "$RUN_ROOT/monitor_20k"

status_or_unknown() {
  supervisorctl status "$1" 2>/dev/null || true
}

latest_checkpoint() {
  { find "$1" -maxdepth 1 -type d -name 'checkpoint-*' -printf '%f\n' 2>/dev/null || true; } \
    | sort -V | tail -n 1
}

mark_milestone() {
  local method=$1
  local directory=$2
  local step=$3
  if [[ -d "$directory/checkpoint-$step" ]]; then
    touch "$RUN_ROOT/monitor_20k/${method}_checkpoint_${step}_ready"
  fi
}

while true; do
  now=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  b0_status=$(status_or_unknown lpwb-b0-20k)
  b1_status=$(status_or_unknown lpwb-b1-20k)
  b2_status=$(status_or_unknown lpwb-b2-20k)
  b0_ckpt=$(latest_checkpoint "$B0_DIR")
  b1_ckpt=$(latest_checkpoint "$B1_DIR")
  b2_ckpt=$(latest_checkpoint "$B2_DIR")
  gpu=$(nvidia-smi --query-gpu=index,memory.used,utilization.gpu \
    --format=csv,noheader,nounits | tr '\n' ';')
  echo "$now | $b0_status latest=${b0_ckpt:-none} | $b1_status latest=${b1_ckpt:-none} | $b2_status latest=${b2_ckpt:-none} | gpu=$gpu"

  for method in b0 b1 b2; do
    log="$LOG_DIR/${method}-20k.log"
    err="$LOG_DIR/${method}-20k.err.log"
    if grep -Eqi 'Traceback|CUDA out of memory|NCCL.*(error|failed)|(^|[^a-z])nan([^a-z]|$)' \
      "$log" "$err" 2>/dev/null; then
      echo "FATAL_PATTERN_DETECTED method=$method"
      exit 1
    fi
  done

  mark_milestone b0 "$B0_DIR" 10000
  mark_milestone b0 "$B0_DIR" 20000
  mark_milestone b1 "$B1_DIR" 10000
  mark_milestone b1 "$B1_DIR" 20000
  mark_milestone b2 "$B2_DIR" 10000
  mark_milestone b2 "$B2_DIR" 20000

  if [[ -f "$B0_DIR/TRAINING_COMPLETE" && -f "$B1_DIR/TRAINING_COMPLETE" ]]; then
    if [[ ! -f "$B2_DIR/TRAINING_COMPLETE" ]] && ! grep -q RUNNING <<<"$b2_status"; then
      echo "$now | starting lpwb-b2-20k on all four GPUs"
      supervisorctl start lpwb-b2-20k
    fi
  fi

  if [[ -f "$B0_DIR/TRAINING_COMPLETE" && -f "$B1_DIR/TRAINING_COMPLETE" && -f "$B2_DIR/TRAINING_COMPLETE" ]]; then
    touch "$RUN_ROOT/TRAINING_20K_PIPELINE_COMPLETE"
    echo "$now | TRAINING_20K_PIPELINE_COMPLETE"
    exit 0
  fi
  sleep 60
done
