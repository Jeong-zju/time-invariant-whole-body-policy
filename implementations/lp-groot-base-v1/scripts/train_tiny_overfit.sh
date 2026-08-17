#!/usr/bin/env bash
set -euo pipefail

PROJECT=/workspace/gr00t-n16-stackbowls-v1/lp-groot-base-v1
GROOT=/workspace/gr00t-n16-stackbowls-v1/Isaac-GR00T
export CUDA_VISIBLE_DEVICES=0
export LP_LABEL_CACHE="$PROJECT/outputs/navigatekitchen_lp_cache_v1"
export LP_MAX_EPISODES=1
export LP_MAX_STEPS=16
export LP_STATIC_FRACTION_CAP=0.2
export PYTHONPATH="$PROJECT/src:$GROOT"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd "$GROOT"
exec .venv/bin/python gr00t/experiment/launch_finetune.py \
  --base-model-path /workspace/gr00t-n16-stackbowls-v1/checkpoints/grootn16_robocasa365_multitask_learning/checkpoint-120000 \
  --dataset-path /workspace/gr00t-n16-stackbowls-v1/datasets/robocasa365-NavigateKitchen-v3 \
  --embodiment-tag NEW_EMBODIMENT \
  --modality-config-path "$PROJECT/configs/lp_base_modality.py" \
  --num-gpus 1 \
  --output-dir "$PROJECT/outputs/lp_groot_base_v1_tiny_gpu0_v4" \
  --experiment-name lp_groot_base_v1_tiny_gpu0_v4 \
  --no-tune-projector \
  --optim adafactor \
  --save-steps 100 \
  --save-total-limit 2 \
  --max-steps 200 \
  --global-batch-size 16 \
  --gradient-accumulation-steps 16 \
  --dataloader-num-workers 2 \
  --learning-rate 1e-4 \
  --warmup-ratio 0.05 \
  --weight-decay 1e-5 \
  --shard-size 16 \
  --episode-sampling-rate 1.0 \
  --num-shards-per-epoch 10000
