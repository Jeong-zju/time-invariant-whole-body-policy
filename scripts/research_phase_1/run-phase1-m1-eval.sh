#!/usr/bin/env bash

set -euo pipefail

project_root="${ARENA_G1_ROOT:-/workspace/time-invariant-whole-body-policy}"
arena_root="${ARENA_G1_REPO:-${project_root}/repos/IsaacLab-Arena}"
python_bin="${ARENA_G1_PYTHON:-${project_root}/venvs/arena/bin/python}"
policy_config="${M1_POLICY_CONFIG:-${project_root}/configs/arena-g1-phase-1-m1-closedloop.yaml}"
protocol_config="${M1_PROTOCOL_CONFIG:-${project_root}/configs/arena-g1-phase-1-m1-evaluation.yaml}"
output_root="${M1_EVAL_OUTPUT_ROOT:-${project_root}/artifacts/validation/arena-g1-phase-1-m1}"
num_steps="${M1_EVAL_NUM_STEPS:-1200}"
seed_values="${M1_EVAL_SEEDS:-0 1 2 3 4 5 6 7 8 9}"
suite="${M1_EVAL_SUITE:-all}"
skip_finalize="${M1_EVAL_SKIP_FINALIZE:-0}"
script_root="${project_root}/scripts/gate_n"

export ACCEPT_EULA=Y
export PRIVACY_CONSENT=Y
export OMNI_KIT_ACCEPT_EULA=YES
export NO_ALBUMENTATIONS_UPDATE=1
export PYTHONUNBUFFERED=1

run_one() {
    local seed="$1"
    local condition="$2"
    local schedule_flag="$3"
    local schedule_value="$4"
    local reference_npz="${5:-}"
    local run_dir="${output_root}/${condition}/seed-$(printf '%03d' "${seed}")"
    local report_path="${run_dir}/rollout.json"
    mkdir -p "${run_dir}"
    if [[ -s "${report_path}" && -s "${run_dir}/rollout.npz" ]]; then
        echo "Phase 1 M1 eval: keeping ${condition} seed=${seed}"
        return
    fi
    local reference_args=()
    if [[ -n "${reference_npz}" ]]; then
        reference_args=(--first_chunk_reference "${reference_npz}")
    fi
    echo "Phase 1 M1 eval: running ${condition} seed=${seed}"
    cd "${arena_root}"
    "${python_bin}" "${script_root}/run_frequency_rollout.py" \
        --headless \
        --device cpu \
        --policy_device cuda \
        --seed "${seed}" \
        --policy_type gr00t_closedloop \
        --policy_config_yaml_path "${policy_config}" \
        --num_steps "${num_steps}" \
        "${schedule_flag}" ${schedule_value} \
        --consumer phase1_m1 \
        --protocol_id arena-g1-phase-1-m1-v1 \
        "${reference_args[@]}" \
        --telemetry_output "${report_path}" \
        --enable_cameras \
        galileo_g1_locomanip_pick_and_place \
        --object brown_box \
        --embodiment g1_wbc_joint 2>&1 | tee "${run_dir}/rollout.log"
}

mkdir -p "${output_root}"

if [[ "${suite}" == "all" || "${suite}" == "standard" || "${suite}" == "smoke" ]]; then
    for seed in ${seed_values}; do
        run_one "${seed}" fixed-3.125hz --replan_steps 16
        reference="${output_root}/fixed-3.125hz/seed-$(printf '%03d' "${seed}")/rollout.npz"
        if [[ "${suite}" != "smoke" ]]; then
            run_one "${seed}" fixed-6.25hz --replan_steps 8 "${reference}"
            run_one "${seed}" fixed-12.5hz --replan_steps 4 "${reference}"
        fi
    done
fi

if [[ "${suite}" == "all" || "${suite}" == "extended" ]]; then
    for seed in ${seed_values}; do
        reference="${output_root}/fixed-3.125hz/seed-$(printf '%03d' "${seed}")/rollout.npz"
        if [[ ! -s "${reference}" ]]; then
            echo "Missing M1 reference first chunk: ${reference}" >&2
            exit 1
        fi
        for hz in 10 15 20 30; do
            run_one "${seed}" "fixed-${hz}hz" --replan_frequency_hz "${hz}" "${reference}"
        done
        run_one "${seed}" jitter-10-30hz --jitter_frequency_hz "10 30" "${reference}"
    done
fi

if [[ "${skip_finalize}" != "1" && ( "${suite}" == "standard" || "${suite}" == "all" || "${suite}" == "finalize" ) ]]; then
    temporary_root="$(mktemp -d "${output_root}/.standard-summary.XXXXXX")"
    trap 'rm -rf -- "${temporary_root}"' EXIT
    ln -s "${output_root}/fixed-3.125hz" "${temporary_root}/replan-16"
    ln -s "${output_root}/fixed-6.25hz" "${temporary_root}/replan-8"
    ln -s "${output_root}/fixed-12.5hz" "${temporary_root}/replan-4"
    cd "${project_root}"
    "${python_bin}" "${script_root}/summarize_frequency_gate.py" \
        --input-root "${temporary_root}" \
        --config "${protocol_config}" \
        --output "${output_root}/report.json"
fi

if [[ "${skip_finalize}" != "1" && ( "${suite}" == "all" || "${suite}" == "finalize" ) ]]; then
    "${python_bin}" "${project_root}/scripts/research_phase_1/evaluate_phase1_m1.py" \
        --standard-report "${output_root}/report.json" \
        --matched-lora-report "${project_root}/artifacts/validation/arena-g1-gate-n/consumer-v1/matched_unconditioned_lora/report.json" \
        --config "${protocol_config}" \
        --rollout-root "${output_root}" \
        --dataset-manifest "${project_root}/artifacts/phase_1_m1/dataset-manifest.json" \
        --training-manifest "${project_root}/checkpoints/arena_g1/phase_1_m1/training-manifest.json" \
        --output "${output_root}/gate-evaluation.json"
fi
