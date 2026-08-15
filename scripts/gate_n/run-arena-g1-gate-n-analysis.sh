#!/usr/bin/env bash

set -euo pipefail

project_root="${ARENA_G1_ROOT:-/workspace/time-invariant-whole-body-policy}"
python_bin="${ARENA_G1_PYTHON:-${project_root}/venvs/arena/bin/python}"
dataset_root="${ARENA_G1_DATASET:-${project_root}/datasets/arena_g1/arena_g1_loco_manipulation_dataset_generated/lerobot}"
output_dir="${ARENA_G1_GATE_N_OUTPUT:-${project_root}/artifacts/validation/arena-g1-gate-n/offline-diagnostics-v0}"

test -x "${python_bin}"
test -f "${dataset_root}/meta/info.json"
mkdir -p "${output_dir}" "${project_root}/logs/gate_n"

export PYTHONUNBUFFERED=1

exec "${python_bin}" "${project_root}/scripts/gate_n/analyze_arena_gate_n.py" \
    --dataset-root "${dataset_root}" \
    --output-dir "${output_dir}" \
    --sample-stride 5 \
    --max-matched-pairs 3000
