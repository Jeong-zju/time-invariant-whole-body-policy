#!/usr/bin/env bash

set -euo pipefail

project_root="${ARENA_G1_ROOT:-/workspace/time-invariant-whole-body-policy}"
arena_root="${ARENA_G1_REPO:-${project_root}/repos/IsaacLab-Arena}"
python_bin="${ARENA_G1_PYTHON:-${project_root}/venvs/arena/bin/python}"
policy_config="${ARENA_G1_POLICY_CONFIG:-${project_root}/configs/arena-g1-gr00t-closedloop.yaml}"
protocol_config="${GATE_N_CONSUMER_CONFIG:-${project_root}/configs/arena-g1-gate-n-consumers-v1.yaml}"
raw_reference_root="${GATE_N_RAW_REFERENCE_ROOT:-${project_root}/artifacts/validation/arena-g1-gate-n/frequency-v1}"
output_root="${GATE_N_CONSUMER_OUTPUT_ROOT:-${project_root}/artifacts/validation/arena-g1-gate-n/consumer-v1}"
num_steps="${GATE_N_NUM_STEPS:-1200}"
method_values="${GATE_N_CONSUMER_METHODS:-se2_waypoint}"
replan_steps_values="${GATE_N_REPLAN_STEPS:-16 8 4}"
seed_values="${GATE_N_SEEDS:-0 1 2 3 4 5 6 7 8 9}"
script_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

mkdir -p "${output_root}"

export ACCEPT_EULA=Y
export PRIVACY_CONSENT=Y
export OMNI_KIT_ACCEPT_EULA=YES
export NO_ALBUMENTATIONS_UPDATE=1
export PYTHONUNBUFFERED=1

for method in ${method_values}; do
    for seed in ${seed_values}; do
        raw_first_chunk="${raw_reference_root}/replan-16/seed-$(printf '%03d' "${seed}")/rollout.npz"
        if [[ ! -s "${raw_first_chunk}" ]]; then
            echo "Missing raw first-chunk reference: ${raw_first_chunk}" >&2
            exit 1
        fi
        for replan_steps in ${replan_steps_values}; do
            run_dir="${output_root}/${method}/replan-${replan_steps}/seed-$(printf '%03d' "${seed}")"
            report_path="${run_dir}/rollout.json"
            telemetry_path="${run_dir}/rollout.npz"
            log_path="${run_dir}/rollout.log"
            mkdir -p "${run_dir}"
            if [[ -s "${report_path}" && -s "${telemetry_path}" ]]; then
                echo "Gate N consumer: keeping method=${method} replan=${replan_steps} seed=${seed}"
                continue
            fi
            echo "Gate N consumer: running method=${method} replan=${replan_steps} seed=${seed}"
            cd "${arena_root}"
            "${python_bin}" "${script_root}/run_frequency_rollout.py" \
                --headless \
                --device cpu \
                --policy_device cuda \
                --seed "${seed}" \
                --policy_type gr00t_closedloop \
                --policy_config_yaml_path "${policy_config}" \
                --num_steps "${num_steps}" \
                --replan_steps "${replan_steps}" \
                --consumer "${method}" \
                --protocol_id arena-g1-gate-n-consumers-v1 \
                --first_chunk_reference "${raw_first_chunk}" \
                --telemetry_output "${report_path}" \
                --enable_cameras \
                galileo_g1_locomanip_pick_and_place \
                --object brown_box \
                --embodiment g1_wbc_joint 2>&1 | tee "${log_path}"
        done
    done
done

if [[ "${GATE_N_SKIP_FINALIZE:-0}" != "1" ]]; then
    for method in ${method_values}; do
        cd "${project_root}"
        "${python_bin}" "${script_root}/summarize_frequency_gate.py" \
            --input-root "${output_root}/${method}" \
            --config "${protocol_config}" \
            --output "${output_root}/${method}/report.json"
    done
fi
