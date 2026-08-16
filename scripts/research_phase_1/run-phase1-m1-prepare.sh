#!/usr/bin/env bash

set -euo pipefail

project_root="${ARENA_G1_ROOT:-/workspace/time-invariant-whole-body-policy}"
python_bin="${ARENA_G1_PYTHON:-${project_root}/venvs/arena/bin/python}"
hf_bin="${ARENA_G1_HF_BIN:-${project_root}/venvs/arena/bin/hf}"
dataset_root="${ARENA_G1_DATA_ROOT:-${project_root}/datasets/arena_g1}"
aligned_source_root="${ARENA_G1_ALIGNED_SOURCE_ROOT:-${dataset_root}/phase1_hdf5_aligned}"
source_lerobot="${ARENA_G1_SOURCE_LEROBOT:-${aligned_source_root}/arena_g1_loco_manipulation_dataset_generated/lerobot}"
full_hdf5="${ARENA_G1_FULL_HDF5:-${dataset_root}/arena_g1_loco_manipulation_dataset_generated.hdf5}"
derived_lerobot="${ARENA_G1_M1_DATASET:-${dataset_root}/arena_g1_phase_1_m1_v1/lerobot}"
artifact_root="${ARENA_G1_M1_ARTIFACT_ROOT:-${project_root}/artifacts/phase_1_m1}"
dataset_revision="97f7b5a4135e4e6a206d394c9b6ec0253f1558a7"

test -x "${python_bin}"
test -x "${hf_bin}"
if [[ ! -f "${full_hdf5}" ]]; then
    "${hf_bin}" download \
        nvidia/Arena-G1-Loco-Manipulation-Task \
        arena_g1_loco_manipulation_dataset_generated.hdf5 \
        --repo-type dataset \
        --revision "${dataset_revision}" \
        --local-dir "${dataset_root}"
fi

export PYTHONPATH="${project_root}/src:${project_root}${PYTHONPATH:+:${PYTHONPATH}}"
if [[ ! -f "${source_lerobot}/meta/info.json" ]]; then
    "${python_bin}" "${project_root}/scripts/research_phase_1/convert_m1_hdf5_source.py" \
        --hdf5 "${full_hdf5}" \
        --output-root "${aligned_source_root}"
fi

"${python_bin}" "${project_root}/scripts/research_phase_1/prepare_m1_dataset.py" \
    --hdf5 "${full_hdf5}" \
    --source-lerobot "${source_lerobot}" \
    --output-lerobot "${derived_lerobot}" \
    --artifact-root "${artifact_root}"

"${python_bin}" "${project_root}/scripts/research_phase_1/inspect_m1_dataset.py" \
    --dataset "${derived_lerobot}" \
    --target "${artifact_root}/targets/episode-000.npz" \
    --output "${artifact_root}/dataset-smoke.json"
