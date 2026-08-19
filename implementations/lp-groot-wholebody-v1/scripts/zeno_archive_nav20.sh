#!/usr/bin/env bash
set -euo pipefail

VAST_HOST=${VAST_HOST:-81.183.231.113}
VAST_PORT=${VAST_PORT:-44685}
REMOTE_ROOT=${REMOTE_ROOT:-/workspace/lpwb-run/closed_loop_nav20_b0_b1_seed20260818_20260819}
ARCHIVE_ROOT=${ARCHIVE_ROOT:-/media/zeno-rp/BAF88BBFF88B7881/lpwb-runs/20260819/nav20_b0_b1_seed20260818}
SSH=(ssh -o BatchMode=yes -o ConnectTimeout=15 -p "$VAST_PORT" "root@$VAST_HOST")
RSYNC_SHELL="ssh -o BatchMode=yes -o ConnectTimeout=15 -p $VAST_PORT"
mkdir -p "$ARCHIVE_ROOT"

while true; do
  if ! "${SSH[@]}" true; then
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) Vast unreachable; retrying"
    sleep 60
    continue
  fi
  if "${SSH[@]}" test -d "$REMOTE_ROOT"; then
    rsync -az --partial -e "$RSYNC_SHELL" \
      "root@$VAST_HOST:$REMOTE_ROOT/" "$ARCHIVE_ROOT/"
  fi

  if "${SSH[@]}" test -f "$REMOTE_ROOT/B0_B1_NAV20_COMPLETE"; then
    summary="$ARCHIVE_ROOT/summary.json"
    test -s "$summary"
    python3 - "$summary" <<'PY'
import json
import sys

with open(sys.argv[1]) as file:
    summary = json.load(file)
assert summary["completed_total_episodes"] == 40, summary
assert summary["expected_total_episodes"] == 40, summary
assert set(summary["methods"]) == {"b0", "b1"}, summary
assert all(summary["methods"][method]["completed"] == 20 for method in ("b0", "b1"))
assert sum(summary["methods"][method]["video_files"] for method in ("b0", "b1")) >= 40
PY
    touch "$ARCHIVE_ROOT/ARCHIVE_COMPLETE"
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) NavigateKitchen pair archive complete"
    exit 0
  fi
  sleep 60
done
