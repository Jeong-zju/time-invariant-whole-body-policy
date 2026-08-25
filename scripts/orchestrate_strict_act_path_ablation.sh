#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/workspace/act-path-pointer-causal-ablation-navigate-kitchen-20260825-v1}
SOURCE=/workspace/act-base-only-navigate-kitchen-scratch-20260823/source/lp-act-v1
ACT=/workspace/act-base-only-navigate-kitchen-scratch-20260823/train-50000/checkpoint-50000-model.pt
POINT=/workspace/act-geometric-continuousyaw-navigate-kitchen-scratch-20260823/train-50000/checkpoint-50000-model.pt
TASKS=/workspace/lpwb-navigate-kitchen-ratefree-20260820/data/NavigateKitchen/lerobot/meta/tasks.jsonl
CALIBRATION=/workspace/act-frequency-sweep-navigate-kitchen-20260823/controller-gates/50hz/base-rate-calibration-50hz.json
mkdir -p "$ROOT/logs" "$ROOT/seeds"
export PYTHONPATH="/workspace/grootn17:$SOURCE/src:/workspace/act-navigate-kitchen-scratch-20260821/source/lerobot/src"

run_seed() {
  seed=$1; gpu=$2
  out="$ROOT/seeds/seed_$seed"
  if [ -f "$out/STRICT_ACT_PATH_ABLATION_COMPLETE" ]; then return 0; fi
  CUDA_VISIBLE_DEVICES="$gpu" MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID="$gpu" PYOPENGL_PLATFORM=egl \
    /workspace/robocasa365-official/.venv/bin/python \
      -m robocasa_act_navigate.strict_act_path_ablation \
      --act-checkpoint "$ACT" --point-checkpoint "$POINT" \
      --tasks "$TASKS" --calibration "$CALIBRATION" \
      --output "$out" --seed "$seed" --device cpu --control-hz 50 --steps 250 \
      >"$ROOT/logs/seed-$seed.log" 2>"$ROOT/logs/seed-$seed.err.log"
}

run_wave() {
  pids=()
  while [ "$#" -gt 0 ]; do run_seed "$1" "$2" & pids+=("$!"); shift 2; done
  rc=0; for pid in "${pids[@]}"; do wait "$pid" || rc=1; done; return "$rc"
}
run_wave 20260818 0 20260819 1 20260820 2 20260821 3
run_wave 20260822 0 20260823 1 20260824 2 20260825 3
run_wave 20260826 0 20260827 1

/workspace/grootn16/.venv/bin/python \
  -m robocasa_act_navigate.analyze_strict_act_path_ablation \
  --root "$ROOT" --num-seeds 10 \
  >"$ROOT/logs/analysis.log" 2>"$ROOT/logs/analysis.err.log"
echo STRICT_ACT_PATH_ABLATION_PIPELINE_COMPLETE
