#!/usr/bin/env bash

set -euo pipefail

project_root="${ARENA_G1_ROOT:-/workspace/time-invariant-whole-body-policy}"
arena_root="${ARENA_G1_REPO:-${project_root}/repos/IsaacLab-Arena}"
python_bin="${ARENA_G1_PYTHON:-${project_root}/venvs/arena/bin/python}"
policy_config="${ARENA_G1_POLICY_CONFIG:-${project_root}/configs/arena-g1-gr00t-closedloop.yaml}"
num_steps="${ARENA_G1_NUM_STEPS:-1200}"
seed="${ARENA_G1_SEED:-0}"
run_log="${ARENA_G1_RUN_LOG:-${project_root}/logs/arena_g1_bm0/closed-loop-seed-${seed}.log}"

mkdir -p "$(dirname "${run_log}")"

export ACCEPT_EULA=Y
export PRIVACY_CONSENT=Y
export OMNI_KIT_ACCEPT_EULA=YES
export NO_ALBUMENTATIONS_UPDATE=1
export PYTHONUNBUFFERED=1

cd "${arena_root}"

"${python_bin}" isaaclab_arena/examples/policy_runner.py \
    --headless \
    --device cpu \
    --policy_device cuda \
    --seed "${seed}" \
    --policy_type gr00t_closedloop \
    --policy_config_yaml_path "${policy_config}" \
    --num_steps "${num_steps}" \
    --enable_cameras \
    galileo_g1_locomanip_pick_and_place \
    --object brown_box \
    --embodiment g1_wbc_joint 2>&1 | tee "${run_log}"
