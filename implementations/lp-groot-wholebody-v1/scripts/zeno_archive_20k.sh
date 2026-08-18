#!/usr/bin/env bash
set -euo pipefail

VAST_HOST=${VAST_HOST:-81.183.231.113}
VAST_PORT=${VAST_PORT:-44685}
REMOTE_RUN_ROOT=${REMOTE_RUN_ROOT:-/workspace/lpwb-run}
ARCHIVE_ROOT=${ARCHIVE_ROOT:-/media/zeno-rp/BAF88BBFF88B7881/lpwb-runs/20260818/20k}
SSH=(ssh -o BatchMode=yes -o ConnectTimeout=15 -p "$VAST_PORT" "root@$VAST_HOST")
RSYNC_SHELL="ssh -o BatchMode=yes -o ConnectTimeout=15 -p $VAST_PORT"
mkdir -p "$ARCHIVE_ROOT/checkpoints" "$ARCHIVE_ROOT/logs"

remote_directory() {
  case "$1" in
    b0) echo "$REMOTE_RUN_ROOT/outputs/b0_groot_command_seed20260818_20k_bs16_20260818" ;;
    b1) echo "$REMOTE_RUN_ROOT/outputs/b1_pose_time_seed20260818_20k_bs16_20260818" ;;
    b2) echo "$REMOTE_RUN_ROOT/outputs/b2_path_time_seed20260818_20k_bs16_20260818" ;;
    *) return 1 ;;
  esac
}

archive_checkpoint() {
  local method=$1
  local step=$2
  local remote_dir
  remote_dir=$(remote_directory "$method")
  local source="$remote_dir/checkpoint-$step"
  local destination="$ARCHIVE_ROOT/checkpoints/$method/checkpoint-$step"
  if [[ -f "$destination/ARCHIVE_COMPLETE" ]]; then
    return 0
  fi
  if ! "${SSH[@]}" test -d "$source"; then
    return 0
  fi
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) archive $method checkpoint-$step"
  mkdir -p "$destination"
  rsync -az --partial --exclude='global_step*' \
    -e "$RSYNC_SHELL" "root@$VAST_HOST:$source/" "$destination/"
  test -s "$destination/config.json"
  test -s "$destination/model.safetensors.index.json"
  test -s "$destination/model-00001-of-00002.safetensors"
  test -s "$destination/model-00002-of-00002.safetensors"
  touch "$destination/ARCHIVE_COMPLETE"
}

while true; do
  if ! "${SSH[@]}" true; then
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) Vast unreachable; retrying"
    sleep 60
    continue
  fi
  for method in b0 b1 b2; do
    archive_checkpoint "$method" 10000
    archive_checkpoint "$method" 20000
  done
  rsync -az --partial -e "$RSYNC_SHELL" \
    "root@$VAST_HOST:$REMOTE_RUN_ROOT/logs/" "$ARCHIVE_ROOT/logs/"
  if "${SSH[@]}" test -f "$REMOTE_RUN_ROOT/TRAINING_20K_PIPELINE_COMPLETE"; then
    touch "$ARCHIVE_ROOT/TRAINING_20K_PIPELINE_COMPLETE"
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) archive pipeline complete"
    exit 0
  fi
  sleep 60
done
