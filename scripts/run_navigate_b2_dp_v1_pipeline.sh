#!/usr/bin/env bash
set -euo pipefail

RUN_ROOT=/workspace/dp-b2-se2delta-ratefree-wholebody-v1-navigate-kitchen-scratch-20260822
SOURCE=/workspace/lp-act-v1-b2dp
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

echo '{"event":"B2_DP_V1_SPATIAL_SWEEP_STARTED"}'
"$PYTHON" -m robocasa_act_navigate.sweep_b2_spatial_step \
  --data-root "$DATA_ROOT" --split "$SPLIT" --calibration "$CALIBRATION" \
  --output "$RUN_ROOT/config/spatial-sweep.json" --samples 2048 \
  --steps-m 0.01,0.015,0.02,0.025,0.03,0.04 \
  2>&1 | tee "$RUN_ROOT/logs/spatial-sweep.log"
SPATIAL_STEP="$("$PYTHON" - "$RUN_ROOT/config/spatial-sweep.json" <<'PY'
import json,sys
v=json.load(open(sys.argv[1]))['selected']; assert int(v['num_anchors'])==8
print(v['spatial_step_m'])
PY
)"

echo '{"event":"B2_DP_V1_LABEL_VALIDATION_STARTED"}'
"$PYTHON" -m robocasa_act_navigate.prepare_b2 \
  --data-root "$DATA_ROOT" --split "$SPLIT" --output-dir "$RUN_ROOT/config/b2-labels" \
  --spatial-step-m "$SPATIAL_STEP" --num-anchors 8 \
  2>&1 | tee "$RUN_ROOT/logs/label-validation.log"

echo '{"event":"B2_DP_V1_TINY_OVERFIT_STARTED","world_size":4,"global_batch_size":32}'
CUDA_VISIBLE_DEVICES=0,1,2,3 "$TORCHRUN" --standalone --nproc_per_node=4 \
  -m robocasa_act_navigate.train_b2_dp_ddp \
  --data-root "$DATA_ROOT" --split "$SPLIT" \
  --frame-cache "$B0_ROOT/cache/navigate_kitchen_rgb256_v1.uint8" \
  --stats "$RUN_ROOT/config/b2-labels/train_stats.json" --metric "$RUN_ROOT/config/b2-labels/metric.json" \
  --output-dir "$RUN_ROOT/tiny-overfit" --steps 1000 --schedule-total-steps 1000 \
  --batch-size-per-gpu 8 --num-workers 0 --tiny-samples 32 --checkpoint-every 1000 --log-every 20 \
  2>&1 | tee "$RUN_ROOT/logs/tiny-overfit.log"

CUDA_VISIBLE_DEVICES=0 "$PYTHON" -m robocasa_act_navigate.diagnose_b2_dp_checkpoint \
  --data-root "$DATA_ROOT" --split "$SPLIT" --split-key train \
  --selected-indices "$RUN_ROOT/tiny-overfit/selected-indices.json" \
  --frame-cache "$B0_ROOT/cache/navigate_kitchen_rgb256_v1.uint8" \
  --metric "$RUN_ROOT/config/b2-labels/metric.json" \
  --checkpoint "$RUN_ROOT/tiny-overfit/checkpoint-01000-model.pt" \
  --batch-size 8 --num-inference-steps 20 --device cuda \
  --output "$RUN_ROOT/tiny-overfit-audit.json" \
  >"$RUN_ROOT/logs/tiny-overfit-audit.log" 2>"$RUN_ROOT/logs/tiny-overfit-audit.err.log"

"$PYTHON" - "$RUN_ROOT/logs/tiny-overfit.log" "$RUN_ROOT/tiny-overfit-audit.json" <<'PY'
import json,sys
rows=[]
for line in open(sys.argv[1]):
 try: r=json.loads(line)
 except: continue
 if 'loss' in r and 'step' in r: rows.append(r)
assert len(rows)>=10
first=sum(r['loss'] for r in rows[:5])/5; last=sum(r['loss'] for r in rows[-5:])/5
r=json.load(open(sys.argv[2]))
checks={
 'noise_loss_reduction_gt_0p15':last < first*.85,
 'train_endpoint_translation_lt_0p06':r['endpoint_translation_error_m']['mean'] < .06,
 'train_endpoint_yaw_lt_0p12':r['endpoint_yaw_error_rad']['mean'] < .12,
 'train_first_reverse_lt_0p05':r['first_anchor_reverse_fraction'] < .05,
 'train_stop_accuracy_gt_0p85':r['stop_accuracy_all'] > .85,
}
print(json.dumps({'event':'B2_DP_V1_TINY_GATE','checks':checks,'first_noise_loss':first,'last_noise_loss':last,'audit':r}))
if not all(checks.values()): raise SystemExit('B2 DP tiny overfit physical gate failed')
PY
touch "$RUN_ROOT/TINY_OVERFIT_GATE_PASSED"

echo '{"event":"B2_DP_V1_GLOBAL_BATCH_128_MEMORY_SMOKE_STARTED"}'
CUDA_VISIBLE_DEVICES=0,1,2,3 "$TORCHRUN" --standalone --nproc_per_node=4 \
  -m robocasa_act_navigate.train_b2_dp_ddp \
  --data-root "$DATA_ROOT" --split "$SPLIT" \
  --frame-cache "$B0_ROOT/cache/navigate_kitchen_rgb256_v1.uint8" \
  --stats "$RUN_ROOT/config/b2-labels/train_stats.json" --metric "$RUN_ROOT/config/b2-labels/metric.json" \
  --output-dir "$RUN_ROOT/memory-smoke-b128" --steps 2 --schedule-total-steps 2 \
  --batch-size-per-gpu 32 --num-workers 0 --tiny-samples 128 --log-every 1 --no-save \
  2>&1 | tee "$RUN_ROOT/logs/memory-smoke-b128.log"
touch "$RUN_ROOT/GLOBAL_BATCH_128_MEMORY_SMOKE_PASSED"

echo '{"event":"B2_DP_V1_STAGE1_10K_STARTED","global_batch_size":128}'
CUDA_VISIBLE_DEVICES=0,1,2,3 "$TORCHRUN" --standalone --nproc_per_node=4 \
  -m robocasa_act_navigate.train_b2_dp_ddp \
  --data-root "$DATA_ROOT" --split "$SPLIT" \
  --frame-cache "$B0_ROOT/cache/navigate_kitchen_rgb256_v1.uint8" \
  --stats "$RUN_ROOT/config/b2-labels/train_stats.json" --metric "$RUN_ROOT/config/b2-labels/metric.json" \
  --output-dir "$RUN_ROOT/train-stage1-10000" --steps 10000 --schedule-total-steps 50000 \
  --batch-size-per-gpu 32 --num-workers 6 --checkpoint-every 5000 --log-every 20 \
  2>&1 | tee "$RUN_ROOT/logs/train-stage1-10000.log"

CUDA_VISIBLE_DEVICES=0 "$PYTHON" -m robocasa_act_navigate.diagnose_b2_dp_checkpoint \
  --data-root "$DATA_ROOT" --split "$SPLIT" --split-key val \
  --frame-cache "$B0_ROOT/cache/navigate_kitchen_rgb256_v1.uint8" \
  --metric "$RUN_ROOT/config/b2-labels/metric.json" \
  --checkpoint "$RUN_ROOT/train-stage1-10000/checkpoint-10000-model.pt" \
  --samples 256 --batch-size 16 --device cuda --output "$RUN_ROOT/stage1-10000-audit.json" \
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
print(json.dumps({'event':'B2_DP_V1_STAGE1_GATE','checks':checks,'report':r}))
if not all(checks.values()): raise SystemExit('B2 DP 10k did not beat V3 platform; 40k continuation blocked')
PY
touch "$RUN_ROOT/STAGE1_10K_GATE_PASSED"

echo '{"event":"B2_DP_V1_STAGE2_40K_CONTINUATION_STARTED","target_total_step":50000}'
CUDA_VISIBLE_DEVICES=0,1,2,3 "$TORCHRUN" --standalone --nproc_per_node=4 \
  -m robocasa_act_navigate.train_b2_dp_ddp \
  --data-root "$DATA_ROOT" --split "$SPLIT" \
  --frame-cache "$B0_ROOT/cache/navigate_kitchen_rgb256_v1.uint8" \
  --stats "$RUN_ROOT/config/b2-labels/train_stats.json" --metric "$RUN_ROOT/config/b2-labels/metric.json" \
  --output-dir "$RUN_ROOT/train-stage2-total50000" --steps 40000 --schedule-total-steps 50000 \
  --batch-size-per-gpu 32 --num-workers 6 --checkpoint-every 5000 --log-every 20 \
  --resume-from "$RUN_ROOT/train-stage1-10000/resume-latest.pt" \
  2>&1 | tee "$RUN_ROOT/logs/train-stage2-total50000.log"

CUDA_VISIBLE_DEVICES=0 "$PYTHON" -m robocasa_act_navigate.diagnose_b2_dp_checkpoint \
  --data-root "$DATA_ROOT" --split "$SPLIT" --split-key val \
  --frame-cache "$B0_ROOT/cache/navigate_kitchen_rgb256_v1.uint8" \
  --metric "$RUN_ROOT/config/b2-labels/metric.json" \
  --checkpoint "$RUN_ROOT/train-stage2-total50000/checkpoint-50000-model.pt" \
  --samples 512 --batch-size 16 --device cuda --output "$RUN_ROOT/posttrain-offline-audit.json" \
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
print(json.dumps({'event':'B2_DP_V1_POSTTRAIN_GATE','checks':checks,'report':r}))
if not all(checks.values()): raise SystemExit('B2 DP final offline gate failed; closed loop blocked')
PY
touch "$RUN_ROOT/POSTTRAIN_OFFLINE_GATE_PASSED"
echo '{"event":"B2_DP_V1_TRAINING_COMPLETE","closed_loop_requires_next_stage":true}'
