#!/usr/bin/env bash

set -euo pipefail

project_root="${ARENA_G1_ROOT:-/workspace/time-invariant-whole-body-policy}"
python_bin="${ARENA_G1_PYTHON:-${project_root}/venvs/arena/bin/python}"
checkpoint_root="${ARENA_G1_M1_CHECKPOINT_ROOT:-${project_root}/checkpoints/arena_g1/phase_1_m1}"
train_output="${ARENA_G1_M1_TRAIN_OUTPUT:-${checkpoint_root}/full_r32_47468}"
base_model="${ARENA_G1_BASE_MODEL:-/dev/shm/arena_g1_checkpoint-20000}"
merged_model="${ARENA_G1_M1_MERGED_MODEL:-${checkpoint_root}/merged}"
dataset="${ARENA_G1_M1_DATASET:-${project_root}/datasets/arena_g1/arena_g1_phase_1_m1_v1/lerobot}"
expected_steps="${ARENA_G1_EXPECTED_STEPS:-47468}"

"${python_bin}" "${project_root}/scripts/research_phase_1/finalize_m1_training.py" \
    --train-output "${train_output}" \
    --expected-steps "${expected_steps}" \
    --dataset-manifest "${project_root}/artifacts/phase_1_m1/dataset-manifest.json" \
    --dataset-stats "${dataset}/meta/stats.json" \
    --checkpoint-root "${checkpoint_root}"

if [[ -e "${merged_model}" && -n "$(find "${merged_model}" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]; then
    printf 'Refusing to overwrite merged checkpoint: %s\n' "${merged_model}" >&2
    exit 1
fi

cd "${project_root}/repos/IsaacLab-Arena/submodules/Isaac-GR00T"
"${python_bin}" "${project_root}/scripts/arena_g1/merge_arena_g1_lora.py" \
    --base-model "${base_model}" \
    --adapter "${train_output}" \
    --output "${merged_model}"
