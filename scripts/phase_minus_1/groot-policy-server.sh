#!/usr/bin/env bash

set -euo pipefail
set -o pipefail

phase_root="${PHASE_MINUS_ONE_ROOT:-/workspace/time-invariant-whole-body-policy}"
groot_root="${phase_root}/repos/Isaac-GR00T"
checkpoint="${phase_root}/checkpoints/turning_on_radio_GR00T-checkpoint-150000"
log_file="${phase_root}/logs/groot-policy-server-supervisor.log"

export HF_HOME="${HF_HOME:-/workspace/.hf_home}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${phase_root}/cache}"

cd "${groot_root}"
.venv/bin/python scripts/b1k/serve_b1k.py \
  --model-path "${checkpoint}" \
  --modality-config-path examples/b1k/r1pro.py \
  --embodiment-tag NEW_EMBODIMENT \
  --host 127.0.0.1 \
  --port 8000 2>&1 | tee -a "${log_file}"
