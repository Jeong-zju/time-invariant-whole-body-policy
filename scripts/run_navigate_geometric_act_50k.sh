#!/usr/bin/env bash
set -euo pipefail

RUN_ROOT=${RUN_ROOT:-/workspace/act-geometric-navigate-kitchen-scratch-20260823}
BASE_RUN=/workspace/act-base-only-navigate-kitchen-scratch-20260823
DATA_ROOT=/workspace/lpwb-navigate-kitchen-ratefree-20260820/data/NavigateKitchen/lerobot
SPLIT=/workspace/lpwb-navigate-kitchen-ratefree-20260820/manifests/navigate_kitchen_seed20260820.json
FRAME_CACHE=/workspace/act-navigate-kitchen-scratch-20260821/cache/navigate_kitchen_rgb256_v1.uint8
PATH_CACHE="$RUN_ROOT/config/geometric_labels_k32_epsxy0p01_epsyaw0p02.float32"
STATS="$RUN_ROOT/config/train_stats_geometric.json"
SOURCE_ROOT="$BASE_RUN/source"
LEROBOT_ROOT=/workspace/act-navigate-kitchen-scratch-20260821/source/lerobot
TORCHRUN=/workspace/grootn16/.venv/bin/torchrun
export PYTHONPATH="$SOURCE_ROOT/lp-act-v1/src:$LEROBOT_ROOT/src"
export OMP_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false

test -f "$RUN_ROOT/GEOMETRIC_TRAINING_GATES_PASSED"
test -f "$PATH_CACHE"
test -f "$STATS"
test ! -e "$RUN_ROOT/train-50000"

CUDA_VISIBLE_DEVICES=0,1,2,3 "$TORCHRUN" --standalone --nproc_per_node=4 \
  -m robocasa_act_navigate.train_ddp \
  --geometric-path \
  --path-cache "$PATH_CACHE" \
  --data-root "$DATA_ROOT" \
  --split "$SPLIT" \
  --frame-cache "$FRAME_CACHE" \
  --stats "$STATS" \
  --output-dir "$RUN_ROOT/train-50000" \
  --steps 50000 \
  --batch-size-per-gpu 32 \
  --num-workers 6 \
  --learning-rate 1e-5 \
  --checkpoint-every 5000 \
  --log-every 20 | tee "$RUN_ROOT/logs/train-50000.log"

touch "$RUN_ROOT/TRAINING_COMPLETE"
echo '{"event":"GEOMETRIC_ACT_50K_COMPLETE"}'
