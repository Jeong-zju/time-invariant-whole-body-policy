#!/usr/bin/env bash

set -euo pipefail

project_root="${ARENA_G1_ROOT:-/workspace/time-invariant-whole-body-policy}"
python_bin="${ARENA_G1_PYTHON:-${project_root}/venvs/arena/bin/python}"
script_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
policy_config="${ARENA_G1_POLICY_CONFIG:-${project_root}/configs/arena-g1-gr00t-closedloop.yaml}"
suite_id="${ARENA_G1_SUITE_ID:-dev-seeds-0-9}"
seeds_string="${ARENA_G1_SEEDS:-0 1 2 3 4 5 6 7 8 9}"
output_root="${ARENA_G1_SUITE_OUTPUT:-${project_root}/artifacts/validation/arena-g1-bm0/${suite_id}}"
resume="${ARENA_G1_SUITE_RESUME:-1}"

checkpoint="$("${python_bin}" - "${policy_config}" <<'PY'
import sys
import yaml

with open(sys.argv[1], encoding="utf-8") as handle:
    print(yaml.safe_load(handle)["model_path"])
PY
)"

mkdir -p "${output_root}/runs" "${project_root}/logs/arena_g1_bm0/${suite_id}"
read -r -a seeds <<<"${seeds_string}"
result_files=()

for seed in "${seeds[@]}"; do
    run_log="${project_root}/logs/arena_g1_bm0/${suite_id}/seed-${seed}.log"
    result="${output_root}/runs/seed-${seed}.json"
    if [[ "${resume}" == "1" && -s "${result}" ]]; then
        "${python_bin}" - "${result}" "${checkpoint}" "${seed}" <<'PY'
import json
import sys

result_path, checkpoint, seed = sys.argv[1:]
result = json.load(open(result_path, encoding="utf-8"))
if (
    result.get("protocol_id") != "arena-g1-box-pick-place-v0"
    or result.get("checkpoint") != checkpoint
    or result.get("seed") != int(seed)
):
    raise SystemExit(
        f"Refusing stale suite result {result_path}: "
        f"checkpoint={result.get('checkpoint')!r}, seed={result.get('seed')!r}"
    )
PY
        result_files+=("${result}")
        continue
    fi
    ARENA_G1_POLICY_CONFIG="${policy_config}" \
    ARENA_G1_SEED="${seed}" \
    ARENA_G1_NUM_STEPS=1200 \
    ARENA_G1_RUN_LOG="${run_log}" \
        "${script_root}/run-arena-g1-eval.sh"
    "${python_bin}" "${script_root}/summarize_arena_g1.py" \
        --log "${run_log}" \
        --output "${result}" \
        --checkpoint "${checkpoint}" \
        --run-id "${suite_id}-seed-${seed}" \
        --seed "${seed}"
    result_files+=("${result}")
done

"${python_bin}" "${script_root}/aggregate_arena_g1.py" \
    --inputs "${result_files[@]}" \
    --output "${output_root}/summary.json" \
    --suite-id "${suite_id}"
