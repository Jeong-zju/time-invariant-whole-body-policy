#!/usr/bin/env bash
set -euo pipefail

export LPWB_MAX_EPISODES=${LPWB_MAX_EPISODES:-2}
export LPWB_SAMPLES_PER_EPISODE=${LPWB_SAMPLES_PER_EPISODE:-8}
export LPWB_MAX_STEPS=${LPWB_MAX_STEPS:-20}
export LPWB_DATALOADER_WORKERS=${LPWB_DATALOADER_WORKERS:-1}
export LPWB_RUN_SUFFIX=${LPWB_RUN_SUFFIX:-_tiny}

CUDA_VISIBLE_DEVICES=0,1 MASTER_PORT=29644 bash "$(dirname "$0")/train_b0_2gpu.sh" &
B0_PID=$!
CUDA_VISIBLE_DEVICES=2,3 MASTER_PORT=29646 bash "$(dirname "$0")/train_b2_2gpu.sh" &
B2_PID=$!
wait "$B0_PID"
wait "$B2_PID"
touch /workspace/lpwb-run/TINY_PAIR_COMPLETE
