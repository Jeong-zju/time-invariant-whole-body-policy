#!/usr/bin/env bash
set -euo pipefail

RUN_ROOT=/workspace/act-b2-se2delta-geometryloss-wholebody-v4-navigate-kitchen-scratch-20260822
SOURCE=/workspace/lp-act-v1-b2v4
B0_ROOT=/workspace/act-navigate-kitchen-scratch-20260821
DATA_ROOT=/workspace/lpwb-navigate-kitchen-ratefree-20260820/data/NavigateKitchen/lerobot
SPLIT=/workspace/lpwb-navigate-kitchen-ratefree-20260820/manifests/navigate_kitchen_seed20260820.json
CALIBRATION=/workspace/lpwb-navigate-kitchen-ratefree-20260820/audit/execution_calibration.json
PYTHON=/workspace/grootn16/.venv/bin/python
TORCHRUN=/workspace/grootn16/.venv/bin/torchrun

mkdir -p "$RUN_ROOT/source" "$RUN_ROOT/logs" "$RUN_ROOT/config"
rm -rf "$RUN_ROOT/source/lp-act-v1"
cp -a "$SOURCE" "$RUN_ROOT/source/lp-act-v1"
cp "$SPLIT" "$RUN_ROOT/config/navigate_kitchen_seed20260820.json"
export PYTHONPATH="$RUN_ROOT/source/lp-act-v1/src:$B0_ROOT/source/lerobot/src"
export OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false

echo '{"event":"B2_V4_SPATIAL_SWEEP_STARTED"}'
"$PYTHON" -m robocasa_act_navigate.sweep_b2_spatial_step \
  --data-root "$DATA_ROOT" --split "$SPLIT" --calibration "$CALIBRATION" \
  --output "$RUN_ROOT/config/spatial-sweep.json" --samples 2048 \
  --steps-m 0.01,0.015,0.02,0.025,0.03,0.04 \
  2>&1 | tee "$RUN_ROOT/logs/spatial-sweep.log"
SPATIAL_STEP="$($PYTHON - "$RUN_ROOT/config/spatial-sweep.json" <<'PY'
import json,sys
v=json.load(open(sys.argv[1]))['selected']; assert int(v['num_anchors'])==8
print(v['spatial_step_m'])
PY
)"

echo '{"event":"B2_V4_LABEL_VALIDATION_STARTED"}'
"$PYTHON" -m robocasa_act_navigate.prepare_b2 \
  --data-root "$DATA_ROOT" --split "$SPLIT" --output-dir "$RUN_ROOT/config/b2-labels" \
  --spatial-step-m "$SPATIAL_STEP" --num-anchors 8 \
  2>&1 | tee "$RUN_ROOT/logs/label-validation.log"

echo '{"event":"B2_V4_TINY_OVERFIT_STARTED","world_size":4,"global_batch_size":32}'
CUDA_VISIBLE_DEVICES=0,1,2,3 "$TORCHRUN" --standalone --nproc_per_node=4 \
  -m robocasa_act_navigate.train_b2_ddp \
  --data-root "$DATA_ROOT" --split "$SPLIT" \
  --frame-cache "$B0_ROOT/cache/navigate_kitchen_rgb256_v1.uint8" \
  --stats "$RUN_ROOT/config/b2-labels/train_stats.json" --metric "$RUN_ROOT/config/b2-labels/metric.json" \
  --output-dir "$RUN_ROOT/tiny-overfit" --steps 500 --batch-size-per-gpu 8 \
  --num-workers 0 --tiny-samples 32 --checkpoint-every 500 --log-every 20 \
  2>&1 | tee "$RUN_ROOT/logs/tiny-overfit.log"
"$PYTHON" - "$RUN_ROOT/logs/tiny-overfit.log" <<'PY'
import json,sys
rows=[]
for line in open(sys.argv[1]):
 try: r=json.loads(line)
 except: continue
 if 'l1_loss' in r and 'step' in r: rows.append(r)
assert len(rows)>=10
first=sum(r['l1_loss'] for r in rows[:5])/5; last=sum(r['l1_loss'] for r in rows[-5:])/5
assert last < first*.80, (first,last)
print(json.dumps({'event':'B2_V4_TINY_OVERFIT_PASSED','first_l1':first,'last_l1':last,'reduction':1-last/first}))
PY

echo '{"event":"B2_V4_STAGE1_10K_STARTED","global_batch_size":128}'
CUDA_VISIBLE_DEVICES=0,1,2,3 "$TORCHRUN" --standalone --nproc_per_node=4 \
  -m robocasa_act_navigate.train_b2_ddp \
  --data-root "$DATA_ROOT" --split "$SPLIT" \
  --frame-cache "$B0_ROOT/cache/navigate_kitchen_rgb256_v1.uint8" \
  --stats "$RUN_ROOT/config/b2-labels/train_stats.json" --metric "$RUN_ROOT/config/b2-labels/metric.json" \
  --output-dir "$RUN_ROOT/train-stage1-10000" --steps 10000 --batch-size-per-gpu 32 \
  --num-workers 6 --checkpoint-every 5000 --log-every 20 \
  2>&1 | tee "$RUN_ROOT/logs/train-stage1-10000.log"

CUDA_VISIBLE_DEVICES=0 "$PYTHON" -m robocasa_act_navigate.diagnose_b2_checkpoint \
  --data-root "$DATA_ROOT" --split "$SPLIT" \
  --frame-cache "$B0_ROOT/cache/navigate_kitchen_rgb256_v1.uint8" \
  --metric "$RUN_ROOT/config/b2-labels/metric.json" \
  --checkpoint "$RUN_ROOT/train-stage1-10000/checkpoint-10000-model.pt" \
  --samples 256 --batch-size 32 --device cuda --output "$RUN_ROOT/stage1-10000-audit.json" \
  >"$RUN_ROOT/logs/stage1-10000-audit.log" 2>"$RUN_ROOT/logs/stage1-10000-audit.err.log"

"$PYTHON" - "$RUN_ROOT/stage1-10000-audit.json" <<'PY'
import json,sys
r=json.load(open(sys.argv[1]))
checks={
 'endpoint_translation_lt_v3_platform':r['endpoint_translation_error_m']['mean'] < .065,
 'endpoint_yaw_lt_v3_platform':r['endpoint_yaw_error_rad']['mean'] < .12,
 'first_reverse_lt_v3_platform':r['first_anchor_reverse_fraction'] < .03,
 'stop_accuracy':r['stop_accuracy_all'] > .90,
}
print(json.dumps({'event':'B2_V4_STAGE1_GATE','checks':checks,'report':r}))
if not all(checks.values()): raise SystemExit('V4 10k did not beat V3 platform; 40k continuation blocked')
PY
touch "$RUN_ROOT/STAGE1_10K_GATE_PASSED"

echo '{"event":"B2_V4_STAGE2_40K_CONTINUATION_STARTED","target_total_step":50000}'
CUDA_VISIBLE_DEVICES=0,1,2,3 "$TORCHRUN" --standalone --nproc_per_node=4 \
  -m robocasa_act_navigate.train_b2_ddp \
  --data-root "$DATA_ROOT" --split "$SPLIT" \
  --frame-cache "$B0_ROOT/cache/navigate_kitchen_rgb256_v1.uint8" \
  --stats "$RUN_ROOT/config/b2-labels/train_stats.json" --metric "$RUN_ROOT/config/b2-labels/metric.json" \
  --output-dir "$RUN_ROOT/train-stage2-total50000" --steps 40000 --batch-size-per-gpu 32 \
  --num-workers 6 --checkpoint-every 5000 --log-every 20 \
  --resume-from "$RUN_ROOT/train-stage1-10000/resume-latest.pt" \
  2>&1 | tee "$RUN_ROOT/logs/train-stage2-total50000.log"

CUDA_VISIBLE_DEVICES=0 "$PYTHON" -m robocasa_act_navigate.diagnose_b2_checkpoint \
  --data-root "$DATA_ROOT" --split "$SPLIT" \
  --frame-cache "$B0_ROOT/cache/navigate_kitchen_rgb256_v1.uint8" \
  --metric "$RUN_ROOT/config/b2-labels/metric.json" \
  --checkpoint "$RUN_ROOT/train-stage2-total50000/checkpoint-50000-model.pt" \
  --samples 512 --batch-size 32 --device cuda --output "$RUN_ROOT/posttrain-offline-audit.json" \
  >"$RUN_ROOT/logs/posttrain-offline-audit.log" 2>"$RUN_ROOT/logs/posttrain-offline-audit.err.log"

"$PYTHON" - "$RUN_ROOT/posttrain-offline-audit.json" <<'PY'
import json,sys
r=json.load(open(sys.argv[1]))
checks={
 'endpoint_translation_mean_lt_0p05':r['endpoint_translation_error_m']['mean'] < .05,
 'endpoint_yaw_mean_lt_0p10':r['endpoint_yaw_error_rad']['mean'] < .10,
 'first_reverse_lt_0p02':r['first_anchor_reverse_fraction'] < .02,
 'stop_accuracy_gt_0p90':r['stop_accuracy_all'] > .90,
}
print(json.dumps({'event':'B2_V4_POSTTRAIN_GATE','checks':checks,'report':r}))
if not all(checks.values()): raise SystemExit('V4 final offline gate failed; closed loop blocked')
PY
touch "$RUN_ROOT/POSTTRAIN_OFFLINE_GATE_PASSED"
echo '{"event":"B2_V4_TRAINING_COMPLETE","closed_loop_requires_next_stage":true}'
