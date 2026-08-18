#!/usr/bin/env bash
set -euo pipefail

GROOT=${GROOT:-/workspace/grootn16}
IMPL=${IMPL:-/workspace/time-invariant-whole-body-policy/implementations/lp-groot-wholebody-v1}
RUN_ROOT=${RUN_ROOT:-/workspace/lpwb-run}
CHECKPOINT="$RUN_ROOT/checkpoints/grootn16-robocasa365/checkpoint-120000"
OUTPUT_ROOT="$RUN_ROOT/outputs"
RUN_NAME="b0_groot_command_seed20260818${LPWB_RUN_SUFFIX:-}"
RUN_DIR="$OUTPUT_ROOT/$RUN_NAME"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
export PYTHONPATH="$IMPL/src:$IMPL/gr00t_patch:$GROOT"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# The image's PyTorch wheel bundles NCCL 2.26, which stalls during communicator
# initialization with this CUDA 13.3 driver. The host-provided NCCL 2.30.7 passed
# the 100 MB two-rank broadcast gate on GPU pairs 0/1 and 2/3.
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libnccl.so.2.30.7${LD_PRELOAD:+:$LD_PRELOAD}
export LPWB_METHOD=b0
export LPWB_SPLIT=train
export LPWB_TRAIN_FRACTION=0.9
export LPWB_SAMPLES_PER_EPISODE=${LPWB_SAMPLES_PER_EPISODE:-128}
export LPWB_MAX_EPISODES=${LPWB_MAX_EPISODES:--1}
mkdir -p "$RUN_DIR"

cd "$GROOT"
"$GROOT/.venv/bin/torchrun" --nproc_per_node=2 --master_port=${MASTER_PORT:-29640} \
  "$IMPL/scripts/launch_finetune_lpwb.py" \
  --method b0 \
  --base-model-path "$CHECKPOINT" \
  --dataset-path "$RUN_ROOT/data/robocasa365/NavigateKitchen/lerobot" \
  --dataset-path "$RUN_ROOT/data/robocasa365/PickPlaceCounterToStove/lerobot" \
  --dataset-path "$RUN_ROOT/data/robocasa365/DeliverStraw/lerobot" \
  --modality-config-path "$IMPL/configs/robocasa_lpwb_config.py" \
  --num-gpus 2 \
  --output-dir "$OUTPUT_ROOT" \
  --experiment-name "$RUN_NAME" \
  --save-steps 500 \
  --save-total-limit 4 \
  --max-steps ${LPWB_MAX_STEPS:-3000} \
  --global-batch-size 16 \
  --gradient-accumulation-steps 8 \
  --dataloader-num-workers ${LPWB_DATALOADER_WORKERS:-2} \
  --learning-rate 1e-4 \
  --warmup-ratio 0.05 \
  --weight-decay 1e-5 \
  --shard-size 256 \
  --num-shards-per-epoch 100000
touch "$RUN_DIR/TRAINING_COMPLETE"
