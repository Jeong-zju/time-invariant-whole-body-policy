#!/usr/bin/env bash
set -euo pipefail

RUN_ROOT=/workspace/act-base-only-navigate-kitchen-scratch-20260823
DATA_ROOT=/workspace/lpwb-navigate-kitchen-ratefree-20260820/data/NavigateKitchen/lerobot
SPLIT=/workspace/lpwb-navigate-kitchen-ratefree-20260820/manifests/navigate_kitchen_seed20260820.json
SHARED_CACHE=/workspace/act-navigate-kitchen-scratch-20260821/cache/navigate_kitchen_rgb256_v1.uint8
PYTHON=/workspace/grootn16/.venv/bin/python
TORCHRUN=/workspace/grootn16/.venv/bin/torchrun
SOURCE_ROOT=/workspace/act-base-only-navigate-kitchen-scratch-20260823/source
LEROBOT_ROOT=/workspace/act-navigate-kitchen-scratch-20260821/source/lerobot
export PYTHONPATH="$SOURCE_ROOT/lp-act-v1/src:$LEROBOT_ROOT/src"
export OMP_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false

mkdir -p "$RUN_ROOT/logs" "$RUN_ROOT/config"
cp "$SPLIT" "$RUN_ROOT/config/navigate_kitchen_seed20260820.json"

echo '{"event":"BASE_ONLY_PREPARATION_STARTED"}'
"$PYTHON" -m robocasa_act_navigate.prepare \
  --base-only \
  --data-root "$DATA_ROOT" \
  --split "$SPLIT" \
  --cache "$SHARED_CACHE" \
  --stats "$RUN_ROOT/config/train_stats_base_only.json" \
  --workers 16

echo '{"event":"BASE_ONLY_TINY_OVERFIT_STARTED"}'
CUDA_VISIBLE_DEVICES=0 "$TORCHRUN" --standalone --nproc_per_node=1 \
  -m robocasa_act_navigate.train_ddp \
  --base-only \
  --data-root "$DATA_ROOT" \
  --split "$SPLIT" \
  --frame-cache "$SHARED_CACHE" \
  --stats "$RUN_ROOT/config/train_stats_base_only.json" \
  --output-dir "$RUN_ROOT/tiny-overfit" \
  --steps 500 \
  --batch-size-per-gpu 8 \
  --num-workers 0 \
  --tiny-samples 32 \
  --checkpoint-every 500 \
  --log-every 20 | tee "$RUN_ROOT/logs/tiny-overfit.log"

"$PYTHON" - "$RUN_ROOT/logs/tiny-overfit.log" <<'PY'
import json
import sys
from pathlib import Path

records = []
for line in Path(sys.argv[1]).read_text().splitlines():
    try:
        row = json.loads(line)
    except json.JSONDecodeError:
        continue
    if "l1_loss" in row and "step" in row:
        records.append(row)
if len(records) < 10:
    raise SystemExit("tiny gate failed: insufficient loss records")
first = sum(float(row["l1_loss"]) for row in records[:5]) / 5
last = sum(float(row["l1_loss"]) for row in records[-5:]) / 5
if not last < first * 0.85:
    raise SystemExit(f"tiny gate failed: l1 first={first}, last={last}")
print(json.dumps({"event": "BASE_ONLY_TINY_OVERFIT_PASSED", "first_l1": first, "last_l1": last}))
PY

echo '{"event":"BASE_ONLY_FORMAL_TRAINING_STARTED"}'
CUDA_VISIBLE_DEVICES=0,1,2,3 "$TORCHRUN" --standalone --nproc_per_node=4 \
  -m robocasa_act_navigate.train_ddp \
  --base-only \
  --data-root "$DATA_ROOT" \
  --split "$SPLIT" \
  --frame-cache "$SHARED_CACHE" \
  --stats "$RUN_ROOT/config/train_stats_base_only.json" \
  --output-dir "$RUN_ROOT/train-50000" \
  --steps 50000 \
  --batch-size-per-gpu 32 \
  --num-workers 6 \
  --checkpoint-every 5000 \
  --log-every 20 | tee "$RUN_ROOT/logs/train-50000.log"

touch "$RUN_ROOT/TRAINING_COMPLETE"
echo '{"event":"BASE_ONLY_PIPELINE_COMPLETE"}'
