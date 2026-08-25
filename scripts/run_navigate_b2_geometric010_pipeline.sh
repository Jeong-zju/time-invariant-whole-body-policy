#!/usr/bin/env bash
set -euo pipefail

export RUN_ROOT=/workspace/act-b2-geometric010-wholebody-navigate-kitchen-scratch-20260821
SOURCE=/workspace/lp-act-v1-geometric010

test -f "$SOURCE/scripts/run_navigate_b2_ratefree_50k.sh"
test -f "$SOURCE/scripts/run_navigate_b2_closed_loop_30.sh"

mkdir -p "$RUN_ROOT/source"
rm -rf "$RUN_ROOT/source/lp-act-v1"
cp -a "$SOURCE" "$RUN_ROOT/source/lp-act-v1"

bash "$RUN_ROOT/source/lp-act-v1/scripts/run_navigate_b2_ratefree_50k.sh"

export OUT="$RUN_ROOT/evaluation/closed_loop_geometric010_wholebody_checkpoint50000_seed20260818_count30"
export PREFLIGHT="$RUN_ROOT/evaluation/preflight_geometric010_wholebody_seed20260817_horizon8"
export PORT=5566
bash "$RUN_ROOT/source/lp-act-v1/scripts/run_navigate_b2_closed_loop_30.sh"

touch "$RUN_ROOT/EXPERIMENT_COMPLETE"
echo '{"event":"EXPERIMENT_COMPLETE","representation":"B2_Path_RateFree_Geometric0p10_WholeBody"}'
