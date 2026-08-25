#!/usr/bin/env bash
set -euo pipefail

export RUN_ROOT=/workspace/act-b2-spatialstep-stop-wholebody-v3-navigate-kitchen-scratch-20260821
SOURCE=/workspace/lp-act-v1-b2v3

mkdir -p "$RUN_ROOT/source" "$RUN_ROOT/logs"
rm -rf "$RUN_ROOT/source/lp-act-v1"
cp -a "$SOURCE" "$RUN_ROOT/source/lp-act-v1"

bash "$RUN_ROOT/source/lp-act-v1/scripts/run_navigate_b2_ratefree_50k.sh"

PYTHONPATH="$RUN_ROOT/source/lp-act-v1/src:/workspace/act-navigate-kitchen-scratch-20260821/source/lerobot/src" \
/workspace/grootn16/.venv/bin/python -m robocasa_act_navigate.diagnose_b2_checkpoint \
  --data-root /workspace/lpwb-navigate-kitchen-ratefree-20260820/data/NavigateKitchen/lerobot \
  --split "$RUN_ROOT/config/navigate_kitchen_seed20260820.json" \
  --frame-cache /workspace/act-navigate-kitchen-scratch-20260821/cache/navigate_kitchen_rgb256_v1.uint8 \
  --metric "$RUN_ROOT/config/b2-labels/metric.json" \
  --checkpoint "$RUN_ROOT/train-50000/checkpoint-50000-model.pt" \
  --samples 512 --batch-size 32 --device cuda \
  --output "$RUN_ROOT/posttrain-offline-audit.json" \
  >"$RUN_ROOT/logs/posttrain-offline-audit.log" 2>"$RUN_ROOT/logs/posttrain-offline-audit.err.log"

/workspace/grootn16/.venv/bin/python - "$RUN_ROOT/posttrain-offline-audit.json" <<'PY'
import json,sys
r=json.load(open(sys.argv[1]))
checks={
 'endpoint_translation_mean_lt_0p05': r['endpoint_translation_error_m']['mean'] < .05,
 'endpoint_yaw_mean_lt_0p10': r['endpoint_yaw_error_rad']['mean'] < .10,
 'first_reverse_lt_0p02': r['first_anchor_reverse_fraction'] < .02,
 'stop_accuracy_gt_0p90': r['stop_accuracy_all'] > .90,
}
print(json.dumps({'event':'B2_POSTTRAIN_OFFLINE_GATE','checks':checks,'report':r}))
if not all(checks.values()): raise SystemExit('posttrain offline gate failed; closed loop blocked')
PY

touch "$RUN_ROOT/POSTTRAIN_OFFLINE_GATE_PASSED"
echo '{"event":"B2_V3_TRAINING_AND_OFFLINE_GATE_COMPLETE","closed_loop_requires_separate_authorized_stage":true}'
