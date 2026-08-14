#!/usr/bin/env bash

set -euo pipefail

project_root="${ARENA_G1_ROOT:-/workspace/time-invariant-whole-body-policy}"
python_bin="${ARENA_G1_PYTHON:-${project_root}/venvs/arena/bin/python}"
train_program="${ARENA_G1_TRAIN_PROGRAM:-arena_g1_train}"
train_output="${ARENA_G1_TRAIN_OUTPUT:-${project_root}/checkpoints/arena_g1/bm0_lora_r32_2ep}"
expected_steps="${ARENA_G1_EXPECTED_STEPS:-47468}"
base_model="${ARENA_G1_BASE_MODEL:-/dev/shm/arena_g1_checkpoint-20000}"
merged_model="${ARENA_G1_MERGED_MODEL:-/dev/shm/arena_g1_lora_merged}"
policy_config="${ARENA_G1_POLICY_CONFIG:-${project_root}/configs/arena-g1-gr00t-lora-closedloop.yaml}"
seed="${ARENA_G1_SEED:-0}"
run_log="${ARENA_G1_RUN_LOG:-${project_root}/logs/arena_g1_bm0/closed-loop-lora-2ep-seed-${seed}.log}"
summary="${ARENA_G1_SUMMARY:-${project_root}/artifacts/validation/arena-g1-bm0/lora-2ep-seed-${seed}.json}"
script_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

while supervisorctl status "${train_program}" | grep -Eq 'RUNNING|STARTING'; do
    sleep 30
done

train_status="$(supervisorctl status "${train_program}" || true)"
printf '%s\n' "${train_status}"
if ! grep -q 'EXITED' <<<"${train_status}"; then
    printf 'Training did not exit normally; refusing to merge.\n' >&2
    exit 1
fi

"${python_bin}" - "${train_output}" "${expected_steps}" <<'PY'
import json
import sys
from pathlib import Path

output = Path(sys.argv[1])
expected = int(sys.argv[2])
state_path = output / "trainer_state.json"
adapter_path = output / "adapter_model.safetensors"
if not state_path.is_file() or not adapter_path.is_file():
    raise SystemExit(f"Incomplete training output: {output}")
step = int(json.loads(state_path.read_text())["global_step"])
if step != expected:
    raise SystemExit(f"Expected global_step={expected}, got {step}")
print(f"Verified completed training at global_step={step}", flush=True)
PY

if [[ -e "${merged_model}" || -L "${merged_model}" ]]; then
    printf 'Refusing to overwrite merged model path: %s\n' "${merged_model}" >&2
    exit 1
fi

cd "${project_root}/repos/IsaacLab-Arena/submodules/Isaac-GR00T"
"${python_bin}" "${script_root}/merge_arena_g1_lora.py" \
    --base-model "${base_model}" \
    --adapter "${train_output}" \
    --output "${merged_model}"

ARENA_G1_POLICY_CONFIG="${policy_config}" \
ARENA_G1_SEED="${seed}" \
ARENA_G1_NUM_STEPS=1200 \
ARENA_G1_RUN_LOG="${run_log}" \
    "${script_root}/run-arena-g1-eval.sh"

"${python_bin}" "${script_root}/summarize_arena_g1.py" \
    --log "${run_log}" \
    --output "${summary}" \
    --checkpoint "${merged_model}" \
    --run-id "lora-r32-2ep-seed-${seed}" \
    --seed "${seed}" \
    --require-nonzero
