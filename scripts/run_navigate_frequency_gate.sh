#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 2 ]; then
  echo "usage: $0 CONTROL_HZ GPU" >&2
  exit 2
fi
HZ=$1
GPU=$2
case "$HZ" in 10|20|50) ;; *) echo "CONTROL_HZ must be 10, 20, or 50" >&2; exit 2;; esac

RUN_ROOT=${RUN_ROOT:-/workspace/act-frequency-sweep-navigate-kitchen-20260823}
SOURCE=/workspace/act-base-only-navigate-kitchen-scratch-20260823/source/lp-act-v1
OUT="$RUN_ROOT/controller-gates/${HZ}hz"
CALIBRATION="$OUT/base-rate-calibration-${HZ}hz.json"
VALIDATION="$OUT/follower-validation-${HZ}hz.json"
WARMUP_STEPS=$(((HZ + 3) / 6))
RECORD_STEPS=$((HZ / 2))
MAX_STEPS=$((HZ * 5))

mkdir -p "$OUT/logs"
if [ -f "$OUT/GATE_COMPLETE" ]; then
  echo "{\"event\":\"FREQUENCY_GATE_ALREADY_COMPLETE\",\"control_hz\":$HZ}"
  exit 0
fi

export PYTHONPATH="/workspace/grootn17:$SOURCE/src"
export MUJOCO_GL=egl
export MUJOCO_EGL_DEVICE_ID="$GPU"
export PYOPENGL_PLATFORM=egl
cd /workspace/grootn17

CUDA_VISIBLE_DEVICES="$GPU" /workspace/robocasa365-official/.venv/bin/python \
  -m robocasa_act_navigate.calibrate_base_30hz \
  --output "$CALIBRATION" \
  --control-hz "$HZ" \
  --split target \
  --seed 20260823 \
  --warmup-steps "$WARMUP_STEPS" \
  --record-steps "$RECORD_STEPS" \
  >"$OUT/logs/calibration.log" 2>"$OUT/logs/calibration.err.log"

CUDA_VISIBLE_DEVICES="$GPU" /workspace/robocasa365-official/.venv/bin/python \
  -m robocasa_act_navigate.validate_follower_30hz \
  --calibration "$CALIBRATION" \
  --output "$VALIDATION" \
  --seed 20260823 \
  --control-hz "$HZ" \
  --max-steps "$MAX_STEPS" \
  >"$OUT/logs/validation.log" 2>"$OUT/logs/validation.err.log"

/workspace/grootn16/.venv/bin/python - "$CALIBRATION" "$VALIDATION" "$HZ" <<'PY'
import json, math, sys
calibration=json.load(open(sys.argv[1]))
validation=json.load(open(sys.argv[2]))
expected=float(sys.argv[3])
assert abs(float(calibration['base_rate']['fitted_control_hz'])-expected)<1e-9
assert abs(float(calibration['actual_control_timestep_s'])-1/expected)<1e-9
assert abs(float(validation['actual_control_hz'])-expected)<1e-9
assert bool(validation['all_pass'])
assert len(validation['trials'])==4
PY
touch "$OUT/GATE_COMPLETE"
echo "{\"event\":\"FREQUENCY_GATE_COMPLETE\",\"control_hz\":$HZ,\"calibration\":\"$CALIBRATION\",\"validation\":\"$VALIDATION\"}"
