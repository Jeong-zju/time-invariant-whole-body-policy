#!/usr/bin/env bash
set -euo pipefail

VAST_HOST=${VAST_HOST:-81.183.231.113}
VAST_PORT=${VAST_PORT:-44685}
REMOTE_ROOT=${REMOTE_ROOT:-/workspace/lpwb-run/closed_loop_nav20_b2_checkpoint14000_seed20260818_20260819}
ARCHIVE_ROOT=${ARCHIVE_ROOT:-/media/zeno-rp/BAF88BBFF88B7881/lpwb-runs/20260819/nav20_b2_checkpoint14000_seed20260818}
SSH=(ssh -o BatchMode=yes -o ConnectTimeout=15 -p "$VAST_PORT" "root@$VAST_HOST")
RSYNC_SHELL="ssh -o BatchMode=yes -o ConnectTimeout=15 -p $VAST_PORT"
mkdir -p "$ARCHIVE_ROOT"

while true; do
  if ! "${SSH[@]}" true; then
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) Vast unreachable; retrying"
    sleep 60
    continue
  fi
  if ! "${SSH[@]}" test -f "$REMOTE_ROOT/B2_NAV20_COMPLETE"; then
    sleep 60
    continue
  fi

  rsync -az --partial -e "$RSYNC_SHELL" \
    "root@$VAST_HOST:$REMOTE_ROOT/" "$ARCHIVE_ROOT/"
  summary="$ARCHIVE_ROOT/threeway_summary.json"
  test -s "$summary"
  python3 - "$summary" <<'PY'
import json
import sys

with open(sys.argv[1]) as file:
    summary = json.load(file)
assert summary["completed_total_episodes"] == 60, summary
assert summary["expected_total_episodes"] == 60, summary
assert summary["methods"]["b2"]["completed"] == 20, summary
assert summary["methods"]["b2"]["video_files"] == 20, summary
assert summary["b2_checkpoint"]["checkpoint_step"] == 14000, summary
PY
  video_count=$(find "$ARCHIVE_ROOT" -type f -name '*.mp4' | wc -l)
  if (( video_count != 20 )); then
    echo "expected exactly 20 archived B2 videos, got $video_count" >&2
    exit 1
  fi
  touch "$ARCHIVE_ROOT/ARCHIVE_COMPLETE"
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) B2 checkpoint-14000 archive complete"
  exit 0
done
