#!/usr/bin/env bash
set -euo pipefail

cd /workspace/behavior-2026/BEHAVIOR-1K
export PYTHONPATH=/workspace/behavior-2026/BEHAVIOR-1K/OmniGibson
export CUDA_VISIBLE_DEVICES=0
export LD_LIBRARY_PATH=/venv/behavior2026/lib/python3.11/site-packages/nvidia/npp/lib:/venv/behavior2026/lib/python3.11/site-packages/nvidia/cuda_nvrtc/lib:${LD_LIBRARY_PATH:-}

exec /workspace/behavior-2026/miniforge3/bin/conda run --no-capture-output -n behavior2026 \
    python -m omnigibson.eval.eval \
    --task-name turning_on_radio \
    --mode public_test \
    --host 127.0.0.1 \
    --port 8000 \
    --instance-indices 0 \
    --num-rollouts 1 \
    --max-steps 5 \
    --output-dir /workspace/behavior-2026/lp-act-v1/outputs/act_eval_turning_radio_instance0_smoke_v1
