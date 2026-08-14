#!/usr/bin/env bash

set -euo pipefail

project_root="${ARENA_G1_ROOT:-/workspace/time-invariant-whole-body-policy}"
arena_root="${ARENA_G1_REPO:-${project_root}/repos/IsaacLab-Arena}"
python_bin="${ARENA_G1_PYTHON:-${project_root}/venvs/arena/bin/python}"
dataset_file="${ARENA_G1_REPLAY_DATASET:-${project_root}/datasets/arena_g1/arena_g1_loco_manipulation_dataset_generated_small.hdf5}"
run_log="${ARENA_G1_REPLAY_LOG:-${project_root}/logs/arena_g1_bm0/expert-replay.log}"

test -f "${dataset_file}"
mkdir -p "$(dirname "${run_log}")"

export ACCEPT_EULA=Y
export PRIVACY_CONSENT=Y
export OMNI_KIT_ACCEPT_EULA=YES
export PYTHONUNBUFFERED=1

cd "${arena_root}"

"${python_bin}" isaaclab_arena/scripts/replay_demos.py \
    --headless \
    --device cpu \
    --enable_cameras \
    --dataset_file "${dataset_file}" \
    galileo_g1_locomanip_pick_and_place \
    --object brown_box \
    --embodiment g1_wbc_pink 2>&1 | tee "${run_log}"
