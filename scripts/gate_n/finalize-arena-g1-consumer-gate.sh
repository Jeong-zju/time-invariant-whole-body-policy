#!/usr/bin/env bash

set -euo pipefail

project_root="${ARENA_G1_ROOT:-/workspace/time-invariant-whole-body-policy}"
python_bin="${ARENA_G1_PYTHON:-${project_root}/venvs/arena/bin/python}"
script_root="${project_root}/scripts/gate_n"
protocol_config="${project_root}/configs/arena-g1-gate-n-consumers-v1.yaml"
output_root="${project_root}/artifacts/validation/arena-g1-gate-n/consumer-v1"
official_raw_root="${project_root}/artifacts/validation/arena-g1-gate-n/frequency-v1"
train_program="arena_g1_gate_n_replan_dt_train"
train_output="${project_root}/checkpoints/arena_g1/gate_n_replan_dt_lora_r32_2ep"
merged_model="/dev/shm/arena_g1_replan_dt_lora_merged"
base_model="/dev/shm/arena_g1_checkpoint-20000"
merge_script="${project_root}/artifacts/deployment/arena_g1/merge_arena_g1_lora.py"

wait_for_exit() {
    local program="$1"
    while true; do
        local line state
        # supervisorctl returns a non-zero shell status for a normally exited
        # one-shot program.  Inspect its reported state instead of letting
        # `set -e` abort the finalizer before the EXITED branch is reached.
        line="$(supervisorctl status "${program}" || true)"
        state="$(awk '{print $2}' <<<"${line}")"
        echo "Gate N finalizer: ${line}"
        case "${state}" in
            EXITED) return 0 ;;
            RUNNING|STARTING|STOPPING) sleep 30 ;;
            *) echo "Gate N finalizer: unexpected ${program} state ${state}" >&2; return 1 ;;
        esac
    done
}

require_rollouts() {
    local method="$1"
    local count
    count="$(find "${output_root}/${method}" -path '*/seed-*/rollout.json' -type f | wc -l)"
    if [[ "${count}" != "30" ]]; then
        echo "Gate N finalizer: ${method} has ${count}/30 rollout reports" >&2
        return 1
    fi
}

run_pair() {
    local low="$1"
    local high="$2"
    supervisorctl start "${low}"
    supervisorctl start "${high}"
    wait_for_exit "${low}"
    wait_for_exit "${high}"
}

mkdir -p "${output_root}"
wait_for_exit "${train_program}"

"${python_bin}" - "${train_output}/trainer_state.json" <<'PY'
import json, sys
path = sys.argv[1]
payload = json.load(open(path, encoding="utf-8"))
if int(payload.get("global_step", -1)) != 47468:
    raise SystemExit(f"expected global_step 47468, got {payload.get('global_step')}")
PY
test -s "${train_output}/adapter_model.safetensors"

if [[ ! -s "${merged_model}/merge_manifest.json" ]]; then
    test ! -e "${merged_model}"
    cd "${project_root}/repos/IsaacLab-Arena/submodules/Isaac-GR00T"
    "${python_bin}" "${merge_script}" \
        --base-model "${base_model}" \
        --adapter "${train_output}" \
        --output "${merged_model}"
fi

run_pair arena_g1_gate_n_matched_lora_low arena_g1_gate_n_matched_lora_high
require_rollouts matched_unconditioned_lora
"${python_bin}" "${script_root}/summarize_frequency_gate.py" \
    --input-root "${output_root}/matched_unconditioned_lora" \
    --config "${protocol_config}" \
    --output "${output_root}/matched_unconditioned_lora/report.json"

run_pair arena_g1_gate_n_replan_dt_low arena_g1_gate_n_replan_dt_high
require_rollouts replan_dt_conditioned_policy
"${python_bin}" "${script_root}/summarize_frequency_gate.py" \
    --input-root "${output_root}/replan_dt_conditioned_policy" \
    --config "${protocol_config}" \
    --output "${output_root}/replan_dt_conditioned_policy/report.json"

run_pair arena_g1_gate_n_se2_low arena_g1_gate_n_se2_high
require_rollouts se2_waypoint
"${python_bin}" "${script_root}/summarize_frequency_gate.py" \
    --input-root "${output_root}/se2_waypoint" \
    --config "${protocol_config}" \
    --output "${output_root}/se2_waypoint/report.json"

"${python_bin}" "${script_root}/evaluate_consumer_gate.py" \
    --official-raw-report "${official_raw_root}/report.json" \
    --matched-lora-report "${output_root}/matched_unconditioned_lora/report.json" \
    --dt-report "${output_root}/replan_dt_conditioned_policy/report.json" \
    --se2-report "${output_root}/se2_waypoint/report.json" \
    --config "${protocol_config}" \
    --output "${output_root}/gate-evaluation.json"

echo "Gate N finalizer: complete ${output_root}/gate-evaluation.json"
