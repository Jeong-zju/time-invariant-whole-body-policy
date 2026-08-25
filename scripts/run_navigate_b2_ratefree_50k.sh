#!/usr/bin/env bash
set -euo pipefail

RUN_ROOT=${RUN_ROOT:-/workspace/act-b2-ratefree-navigate-kitchen-scratch-20260821}
B0_ROOT=/workspace/act-navigate-kitchen-scratch-20260821
DATA_ROOT=/workspace/lpwb-navigate-kitchen-ratefree-20260820/data/NavigateKitchen/lerobot
SPLIT=/workspace/lpwb-navigate-kitchen-ratefree-20260820/manifests/navigate_kitchen_seed20260820.json
PYTHON=/workspace/grootn16/.venv/bin/python
TORCHRUN=/workspace/grootn16/.venv/bin/torchrun
SOURCE_ROOT="$RUN_ROOT/source"
export PYTHONPATH="$SOURCE_ROOT/lp-act-v1/src:$B0_ROOT/source/lerobot/src"
export OMP_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false

mkdir -p "$RUN_ROOT/logs" "$RUN_ROOT/config"
cp "$SPLIT" "$RUN_ROOT/config/navigate_kitchen_seed20260820.json"

echo '{"event":"B2_SPATIAL_SWEEP_STARTED","anchors":8,"candidates_m":[0.01,0.015,0.02,0.025,0.03,0.04]}'
"$PYTHON" -m robocasa_act_navigate.sweep_b2_spatial_step \
  --data-root "$DATA_ROOT" --split "$SPLIT" \
  --calibration /workspace/lpwb-navigate-kitchen-ratefree-20260820/audit/execution_calibration.json \
  --output "$RUN_ROOT/config/spatial-sweep.json" --samples 2048 \
  --steps-m 0.01,0.015,0.02,0.025,0.03,0.04 \
  2>&1 | tee "$RUN_ROOT/logs/spatial-sweep.log"

SPATIAL_STEP="$($PYTHON - "$RUN_ROOT/config/spatial-sweep.json" <<'PY'
import json,sys
value=json.load(open(sys.argv[1]))['selected']
assert int(value['num_anchors']) == 8
print(value['spatial_step_m'])
PY
)"
echo "{\"event\":\"B2_SPATIAL_SWEEP_SELECTED\",\"spatial_step_m\":$SPATIAL_STEP,\"anchors\":8}"

echo '{"event":"B2_LABEL_VALIDATION_STARTED"}'
"$PYTHON" -m robocasa_act_navigate.prepare_b2 \
  --data-root "$DATA_ROOT" \
  --split "$SPLIT" \
  --output-dir "$RUN_ROOT/config/b2-labels" \
  --spatial-step-m "$SPATIAL_STEP" --num-anchors 8 \
  2>&1 | tee "$RUN_ROOT/logs/label-validation.log"

echo '{"event":"B2_TINY_OVERFIT_STARTED","world_size":4,"global_batch_size":32}'
CUDA_VISIBLE_DEVICES=0,1,2,3 "$TORCHRUN" --standalone --nproc_per_node=4 \
  -m robocasa_act_navigate.train_b2_ddp \
  --data-root "$DATA_ROOT" \
  --split "$SPLIT" \
  --frame-cache "$B0_ROOT/cache/navigate_kitchen_rgb256_v1.uint8" \
  --stats "$RUN_ROOT/config/b2-labels/train_stats.json" \
  --metric "$RUN_ROOT/config/b2-labels/metric.json" \
  --output-dir "$RUN_ROOT/tiny-overfit" \
  --steps 500 \
  --batch-size-per-gpu 8 \
  --num-workers 0 \
  --tiny-samples 32 \
  --checkpoint-every 500 \
  --log-every 20 2>&1 | tee "$RUN_ROOT/logs/tiny-overfit.log"

"$PYTHON" - "$RUN_ROOT/logs/tiny-overfit.log" <<'PY'
import json, sys
from pathlib import Path
records = []
for line in Path(sys.argv[1]).read_text().splitlines():
    try: row = json.loads(line)
    except json.JSONDecodeError: continue
    if "l1_loss" in row and "step" in row: records.append(row)
if len(records) < 10: raise SystemExit("tiny gate failed: insufficient records")
first = sum(float(row["l1_loss"]) for row in records[:5]) / 5
last = sum(float(row["l1_loss"]) for row in records[-5:]) / 5
if not last < first * 0.85: raise SystemExit(f"tiny gate failed: first={first}, last={last}")
print(json.dumps({"event":"B2_TINY_OVERFIT_PASSED","first_l1":first,"last_l1":last,"reduction":1-last/first}))
PY

echo '{"event":"B2_FORMAL_TRAINING_STARTED","world_size":4,"batch_size_per_gpu":32,"global_batch_size":128,"steps":50000}'
CUDA_VISIBLE_DEVICES=0,1,2,3 "$TORCHRUN" --standalone --nproc_per_node=4 \
  -m robocasa_act_navigate.train_b2_ddp \
  --data-root "$DATA_ROOT" \
  --split "$SPLIT" \
  --frame-cache "$B0_ROOT/cache/navigate_kitchen_rgb256_v1.uint8" \
  --stats "$RUN_ROOT/config/b2-labels/train_stats.json" \
  --metric "$RUN_ROOT/config/b2-labels/metric.json" \
  --output-dir "$RUN_ROOT/train-50000" \
  --steps 50000 \
  --batch-size-per-gpu 32 \
  --num-workers 6 \
  --checkpoint-every 5000 \
  --log-every 20 2>&1 | tee "$RUN_ROOT/logs/train-50000.log"

touch "$RUN_ROOT/PIPELINE_COMPLETE"
echo '{"event":"B2_PIPELINE_COMPLETE"}'
