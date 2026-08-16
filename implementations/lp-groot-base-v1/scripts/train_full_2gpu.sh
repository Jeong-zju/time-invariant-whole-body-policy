#!/usr/bin/env bash
set -euo pipefail

PROJECT=/workspace/gr00t-n16-stackbowls-v1/lp-groot-base-v1
GROOT=/workspace/gr00t-n16-stackbowls-v1/Isaac-GR00T
export CUDA_VISIBLE_DEVICES=0,1
export LP_LABEL_CACHE="$PROJECT/outputs/navigatekitchen_lp_cache_v1"
export LP_MAX_EPISODES=-1
export LP_MAX_STEPS=-1
export LP_STATIC_FRACTION_CAP=0.2
export PYTHONPATH="$PROJECT/src:$GROOT"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# This Vast host's default intra-node NCCL transports trigger an illegal GPU
# memory access even in a minimal 100 MB broadcast. Socket transport is slower
# but passed the same two-rank smoke test on both RTX 5090s.
export NCCL_P2P_DISABLE=1
export NCCL_SHM_DISABLE=1
export NCCL_IB_DISABLE=1

cd "$GROOT"
.venv/bin/torchrun --nproc_per_node=2 --master_port=29631 gr00t/experiment/launch_finetune.py \
  --base-model-path /workspace/gr00t-n16-stackbowls-v1/checkpoints/grootn16_robocasa365_multitask_learning/checkpoint-120000 \
  --dataset-path /workspace/gr00t-n16-stackbowls-v1/datasets/robocasa365-NavigateKitchen-v3 \
  --embodiment-tag NEW_EMBODIMENT \
  --modality-config-path "$PROJECT/configs/lp_base_modality.py" \
  --num-gpus 2 \
  --output-dir "$PROJECT/outputs/lp_groot_base_v1_full_2gpu_20260815" \
  --experiment-name lp_groot_base_v1_full_2gpu_20260815 \
  --no-tune-projector \
  --optim adafactor \
  --save-steps 500 \
  --save-total-limit 4 \
  --max-steps 10000 \
  --global-batch-size 16 \
  --gradient-accumulation-steps 8 \
  --dataloader-num-workers 2 \
  --learning-rate 1e-4 \
  --warmup-ratio 0.05 \
  --weight-decay 1e-5 \
  --shard-size 1024 \
  --episode-sampling-rate 1.0 \
  --num-shards-per-epoch 100000

touch "$PROJECT/outputs/lp_groot_base_v1_full_2gpu_20260815/TRAINING_COMPLETE"
"$PROJECT/scripts/evaluate_after_training.sh"
