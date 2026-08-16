#!/usr/bin/env bash

set -euo pipefail

project_root="${ARENA_G1_ROOT:-/workspace/time-invariant-whole-body-policy}"
arena_root="${ARENA_G1_REPO:-${project_root}/repos/IsaacLab-Arena}"
python_bin="${ARENA_G1_PYTHON:-${project_root}/venvs/arena/bin/python}"
protocol_config="${GATE_N_CONSUMER_CONFIG:-${project_root}/configs/arena-g1-gate-n-consumers-v1.yaml}"
method="${GATE_N_POLICY_METHOD:?Set GATE_N_POLICY_METHOD}"
policy_config="${GATE_N_POLICY_CONFIG:?Set GATE_N_POLICY_CONFIG}"
output_root="${GATE_N_POLICY_OUTPUT_ROOT:-${project_root}/artifacts/validation/arena-g1-gate-n/consumer-v1/${method}}"
num_steps="${GATE_N_NUM_STEPS:-1200}"
replan_steps_values="${GATE_N_REPLAN_STEPS:-16 8 4}"
seed_values="${GATE_N_SEEDS:-0 1 2 3 4 5 6 7 8 9}"
require_first_chunk="${GATE_N_REQUIRE_FIRST_CHUNK:-1}"
reference_wait_seconds="${GATE_N_REFERENCE_WAIT_SECONDS:-900}"
script_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

mkdir -p "${output_root}"
export ACCEPT_EULA=Y
export PRIVACY_CONSENT=Y
export OMNI_KIT_ACCEPT_EULA=YES
export NO_ALBUMENTATIONS_UPDATE=1
export PYTHONUNBUFFERED=1

for seed in ${seed_values}; do
    for replan_steps in ${replan_steps_values}; do
        run_dir="${output_root}/replan-${replan_steps}/seed-$(printf '%03d' "${seed}")"
        report_path="${run_dir}/rollout.json"
        telemetry_path="${run_dir}/rollout.npz"
        log_path="${run_dir}/rollout.log"
        mkdir -p "${run_dir}"
        if [[ -s "${report_path}" && -s "${telemetry_path}" ]]; then
            echo "Gate N policy: keeping method=${method} replan=${replan_steps} seed=${seed}"
            continue
        fi

        first_chunk_args=()
        if [[ "${require_first_chunk}" == "1" && "${replan_steps}" != "16" ]]; then
            reference_telemetry="${output_root}/replan-16/seed-$(printf '%03d' "${seed}")/rollout.npz"
            waited=0
            while [[ ! -s "${reference_telemetry}" && "${waited}" -lt "${reference_wait_seconds}" ]]; do
                echo "Gate N policy: waiting for method=${method} seed=${seed} reference (${waited}s)"
                sleep 5
                waited=$((waited + 5))
            done
            if [[ ! -s "${reference_telemetry}" ]]; then
                echo "Timed out waiting for ${reference_telemetry}" >&2
                exit 1
            fi
            first_chunk_args=(--first_chunk_reference "${reference_telemetry}")
        fi

        echo "Gate N policy: running method=${method} replan=${replan_steps} seed=${seed}"
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
            --consumer raw \
            --protocol_id arena-g1-gate-n-consumers-v1 \
            --telemetry_output "${report_path}" \
            "${first_chunk_args[@]}" \
            --enable_cameras \
            galileo_g1_locomanip_pick_and_place \
            --object brown_box \
            --embodiment g1_wbc_joint 2>&1 | tee "${log_path}"
    done
done

if [[ "${GATE_N_SKIP_FINALIZE:-0}" != "1" ]]; then
    cd "${project_root}"
    "${python_bin}" "${script_root}/summarize_frequency_gate.py" \
        --input-root "${output_root}" \
        --config "${protocol_config}" \
        --output "${output_root}/report.json"
fi
