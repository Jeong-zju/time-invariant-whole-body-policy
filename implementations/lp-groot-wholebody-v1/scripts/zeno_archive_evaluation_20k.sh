#!/usr/bin/env bash
set -euo pipefail

VAST_HOST=${VAST_HOST:-81.183.231.113}
VAST_PORT=${VAST_PORT:-44685}
REMOTE_RUN_ROOT=${REMOTE_RUN_ROOT:-/workspace/lpwb-run}
ARCHIVE_ROOT=${ARCHIVE_ROOT:-/media/zeno-rp/BAF88BBFF88B7881/lpwb-runs/20260818/20k}
SSH=(ssh -o BatchMode=yes -o ConnectTimeout=15 -p "$VAST_PORT" "root@$VAST_HOST")
RSYNC_SHELL="ssh -o BatchMode=yes -o ConnectTimeout=15 -p $VAST_PORT"
mkdir -p \
  "$ARCHIVE_ROOT/logs" \
  "$ARCHIVE_ROOT/robocasa365_setup" \
  "$ARCHIVE_ROOT/evaluation_20k_seed20260818" \
  "$ARCHIVE_ROOT/closed_loop_20k_seed20260818"

sync_if_remote_directory_exists() {
  local remote=$1
  local local_directory=$2
  if "${SSH[@]}" test -d "$remote"; then
    rsync -az --partial -e "$RSYNC_SHELL" \
      "root@$VAST_HOST:$remote/" "$local_directory/"
  fi
}

while true; do
  if ! "${SSH[@]}" true; then
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) Vast unreachable; retrying"
    sleep 60
    continue
  fi

  sync_if_remote_directory_exists \
    "$REMOTE_RUN_ROOT/robocasa365_setup" \
    "$ARCHIVE_ROOT/robocasa365_setup"
  sync_if_remote_directory_exists \
    "$REMOTE_RUN_ROOT/evaluation_20k_seed20260818" \
    "$ARCHIVE_ROOT/evaluation_20k_seed20260818"
  sync_if_remote_directory_exists \
    "$REMOTE_RUN_ROOT/closed_loop_20k_seed20260818" \
    "$ARCHIVE_ROOT/closed_loop_20k_seed20260818"
  rsync -az --partial -e "$RSYNC_SHELL" \
    "root@$VAST_HOST:$REMOTE_RUN_ROOT/logs/" "$ARCHIVE_ROOT/logs/"

  if "${SSH[@]}" test -f \
    "$REMOTE_RUN_ROOT/closed_loop_20k_seed20260818/CLOSED_LOOP_MATRIX_COMPLETE"; then
    summary="$ARCHIVE_ROOT/closed_loop_20k_seed20260818/summary.json"
    test -s "$summary"
    python3 - "$summary" <<'PY'
import json
import sys

with open(sys.argv[1]) as file:
    summary = json.load(file)
assert summary["completed_total_episodes"] == 270, summary
assert summary["expected_total_episodes"] == 270, summary
PY
    video_count=$(find "$ARCHIVE_ROOT/closed_loop_20k_seed20260818" -type f -name '*.mp4' | wc -l)
    if (( video_count < 270 )); then
      echo "only $video_count archived videos; retrying" >&2
      sleep 60
      continue
    fi
    touch "$ARCHIVE_ROOT/EVALUATION_20K_ARCHIVE_COMPLETE"
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) evaluation archive complete"
    exit 0
  fi
  sleep 60
done
