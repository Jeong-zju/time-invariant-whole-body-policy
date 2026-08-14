#!/usr/bin/env bash

set -euo pipefail

project_root="${ARENA_G1_ROOT:-/workspace/time-invariant-whole-body-policy}"
python_bin="${ARENA_G1_PYTHON:-${project_root}/venvs/arena/bin/python}"
hf_bin="${ARENA_G1_HF_BIN:-${project_root}/venvs/arena/bin/hf}"
dataset_root="${ARENA_G1_DATA_ROOT:-${project_root}/datasets/arena_g1}"
lerobot_parent="${dataset_root}/arena_g1_loco_manipulation_dataset_generated"
model_root="${ARENA_G1_BASE_MODEL:-/dev/shm/arena_g1_checkpoint-20000}"
dataset_revision="97f7b5a4135e4e6a206d394c9b6ec0253f1558a7"
model_revision="629479fedb1cf97c2f11ddc49eed951c5b750139"

if [[ ! -x "${hf_bin}" ]]; then
    printf 'Hugging Face CLI not found: %s\n' "${hf_bin}" >&2
    exit 1
fi

mkdir -p "${dataset_root}" "${lerobot_parent}" "${model_root}"

"${hf_bin}" download \
    nvidia/Arena-G1-Loco-Manipulation-Task \
    arena_g1_loco_manipulation_dataset_generated_small.hdf5 \
    --repo-type dataset \
    --revision "${dataset_revision}" \
    --local-dir "${dataset_root}"

"${hf_bin}" download \
    nvidia/Arena-G1-Loco-Manipulation-Task \
    --include 'lerobot/*' \
    --repo-type dataset \
    --revision "${dataset_revision}" \
    --local-dir "${lerobot_parent}"

"${hf_bin}" download \
    nvidia/GN1x-Tuned-Arena-G1-Loco-Manipulation \
    config.json \
    model-00001-of-00002.safetensors \
    model-00002-of-00002.safetensors \
    model.safetensors.index.json \
    experiment_cfg/metadata.json \
    --revision "${model_revision}" \
    --local-dir "${model_root}"

"${python_bin}" "$(dirname "$0")/verify_arena_g1.py" \
    --project-root "${project_root}" \
    --dataset-root "${dataset_root}" \
    --model-root "${model_root}" \
    --output "${project_root}/artifacts/validation/arena-g1-bm0/install-manifest.json"
