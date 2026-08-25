#!/usr/bin/env bash
set -euo pipefail

RUN_ROOT=${RUN_ROOT:-/workspace/act-geometric-navigate-kitchen-scratch-20260823}
BASE_RUN=/workspace/act-base-only-navigate-kitchen-scratch-20260823
DATA_ROOT=/workspace/lpwb-navigate-kitchen-ratefree-20260820/data/NavigateKitchen/lerobot
SPLIT=/workspace/lpwb-navigate-kitchen-ratefree-20260820/manifests/navigate_kitchen_seed20260820.json
FRAME_CACHE=/workspace/act-navigate-kitchen-scratch-20260821/cache/navigate_kitchen_rgb256_v1.uint8
SOURCE_ROOT="$BASE_RUN/source"
LEROBOT_ROOT=/workspace/act-navigate-kitchen-scratch-20260821/source/lerobot
PYTHON=/workspace/grootn16/.venv/bin/python
TORCHRUN=/workspace/grootn16/.venv/bin/torchrun
export PYTHONPATH="$SOURCE_ROOT/lp-act-v1/src:$LEROBOT_ROOT/src"
export OMP_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false

mkdir -p "$RUN_ROOT/config" "$RUN_ROOT/logs"

"$PYTHON" -m robocasa_act_navigate.prepare_geometric \
  --data-root "$DATA_ROOT" \
  --split "$SPLIT" \
  --cache "$RUN_ROOT/config/geometric_labels_k32_epsxy0p01_epsyaw0p02.float32" \
  --stats "$RUN_ROOT/config/train_stats_geometric.json" \
  --report "$RUN_ROOT/config/label_cache_report.json" \
  --translation-tolerance-m 0.01 \
  --yaw-tolerance-rad 0.02

test ! -e "$RUN_ROOT/offline-validation.tmp"
test ! -e "$RUN_ROOT/offline-validation"
"$PYTHON" -m robocasa_act_navigate.validate_geometric_labels \
  --data-root "$DATA_ROOT" \
  --split "$SPLIT" \
  --cache "$RUN_ROOT/config/geometric_labels_k32_epsxy0p01_epsyaw0p02.float32" \
  --output-dir "$RUN_ROOT/offline-validation.tmp" \
  --translation-tolerance-m 0.01 \
  --yaw-tolerance-rad 0.02 \
  --samples 2048
mv "$RUN_ROOT/offline-validation.tmp" "$RUN_ROOT/offline-validation"

test ! -e "$RUN_ROOT/tiny-overfit.tmp"
test ! -e "$RUN_ROOT/tiny-overfit"
CUDA_VISIBLE_DEVICES=2 "$TORCHRUN" --standalone --nproc_per_node=1 \
  -m robocasa_act_navigate.train_ddp \
  --geometric-path \
  --path-cache "$RUN_ROOT/config/geometric_labels_k32_epsxy0p01_epsyaw0p02.float32" \
  --data-root "$DATA_ROOT" \
  --split "$SPLIT" \
  --frame-cache "$FRAME_CACHE" \
  --stats "$RUN_ROOT/config/train_stats_geometric.json" \
  --output-dir "$RUN_ROOT/tiny-overfit.tmp" \
  --steps 1000 \
  --batch-size-per-gpu 8 \
  --num-workers 0 \
  --tiny-samples 32 \
  --checkpoint-every 1000 \
  --log-every 20 | tee "$RUN_ROOT/logs/tiny-overfit.log"
mv "$RUN_ROOT/tiny-overfit.tmp" "$RUN_ROOT/tiny-overfit"

"$PYTHON" - "$RUN_ROOT/logs/tiny-overfit.log" "$RUN_ROOT/config/tiny_overfit_report.json" <<'PY'
import json
import sys
from pathlib import Path
records=[]
for line in Path(sys.argv[1]).read_text().splitlines():
    try: row=json.loads(line)
    except json.JSONDecodeError: continue
    if "step" in row and "l1_loss" in row: records.append(row)
if len(records) < 20: raise SystemExit("insufficient tiny-overfit records")
first=sum(float(x["l1_loss"]) for x in records[:5])/5
last=sum(float(x["l1_loss"]) for x in records[-5:])/5
ratio=last/first
if not ratio < 0.25: raise SystemExit(f"tiny overfit failed: first={first} last={last} ratio={ratio}")
report={"event":"GEOMETRIC_TINY_OVERFIT_PASSED","first_l1":first,"last_l1":last,"ratio":ratio,"samples":32,"steps":1000}
Path(sys.argv[2]).write_text(json.dumps(report,indent=2)+"\n")
print(json.dumps(report))
PY

touch "$RUN_ROOT/GEOMETRIC_TRAINING_GATES_PASSED"
echo '{"event":"GEOMETRIC_TRAINING_GATES_PASSED"}'
