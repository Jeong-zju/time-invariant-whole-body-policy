#!/usr/bin/env bash
set -euo pipefail

cd /workspace/behavior-2026/lp-act-v1
export PYTHONPATH=/workspace/behavior-2026/lp-act-v1/src:/workspace/behavior-2026/BEHAVIOR-1K/OmniGibson
export CUDA_VISIBLE_DEVICES=0
export LD_LIBRARY_PATH=/venv/behavior2026/lib/python3.11/site-packages/nvidia/npp/lib:/venv/behavior2026/lib/python3.11/site-packages/nvidia/cuda_nvrtc/lib:${LD_LIBRARY_PATH:-}

exec /workspace/behavior-2026/miniforge3/bin/conda run --no-capture-output -n behavior2026 \
    python -m lp_act.serve_eval \
    --checkpoint /workspace/behavior-2026/lp-act-v1/outputs/full_act_turning_radio_b32_seed0_v7/checkpoint_best.pt \
    --device cuda:0 \
    --execute-prefix 6 \
    --host 127.0.0.1 \
    --port 8000
