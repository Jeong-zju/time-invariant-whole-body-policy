#!/usr/bin/env bash

set -euo pipefail

project_root="${ARENA_G1_ROOT:-/workspace/time-invariant-whole-body-policy}"
arena_root="${ARENA_G1_REPO:-${project_root}/repos/IsaacLab-Arena}"
python_bin="${ARENA_G1_PYTHON:-${project_root}/venvs/arena/bin/python}"
policy_config="${ARENA_G1_POLICY_CONFIG:-${project_root}/configs/arena-g1-gr00t-closedloop.yaml}"
reference_root="${GATE_N_REFERENCE_ROOT:-${project_root}/artifacts/validation/arena-g1-gate-n/frequency-v1}"
output_root="${GATE_N_REPEAT_OUTPUT_ROOT:-${project_root}/artifacts/validation/arena-g1-gate-n/frequency-v1-repeatability}"
num_steps="${GATE_N_NUM_STEPS:-1200}"
seed_values="${GATE_N_SEEDS:-0 1 2 3 4 5 6 7 8 9}"
script_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export ACCEPT_EULA=Y
export PRIVACY_CONSENT=Y
export OMNI_KIT_ACCEPT_EULA=YES
export NO_ALBUMENTATIONS_UPDATE=1
export PYTHONUNBUFFERED=1

for seed in ${seed_values}; do
    run_dir="${output_root}/repeat-16/seed-$(printf '%03d' "${seed}")"
    report_path="${run_dir}/rollout.json"
    telemetry_path="${run_dir}/rollout.npz"
    reference_telemetry="${reference_root}/replan-16/seed-$(printf '%03d' "${seed}")/rollout.npz"
    mkdir -p "${run_dir}"
    if [[ -s "${report_path}" && -s "${telemetry_path}" ]]; then
        echo "Gate N repeatability: keeping completed seed=${seed}"
        continue
    fi
    if [[ ! -s "${reference_telemetry}" ]]; then
        echo "Missing reference telemetry: ${reference_telemetry}" >&2
        exit 1
    fi
    echo "Gate N repeatability: running 16-step duplicate seed=${seed}"
    cd "${arena_root}"
    "${python_bin}" "${script_root}/run_frequency_rollout.py" \
        --headless \
        --device cpu \
        --policy_device cuda \
        --seed "${seed}" \
        --policy_type gr00t_closedloop \
        --policy_config_yaml_path "${policy_config}" \
        --num_steps "${num_steps}" \
        --replan_steps 16 \
        --first_chunk_reference "${reference_telemetry}" \
        --telemetry_output "${report_path}" \
        --enable_cameras \
        galileo_g1_locomanip_pick_and_place \
        --object brown_box \
        --embodiment g1_wbc_joint 2>&1 | tee "${run_dir}/rollout.log"
done
