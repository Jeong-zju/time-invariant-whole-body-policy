#!/usr/bin/env bash
set -euo pipefail

GROOT=${GROOT:-/workspace/grootn16}
IMPL=${IMPL:-/workspace/time-invariant-whole-body-policy/implementations/lp-groot-wholebody-v1}
RUN_ROOT=${RUN_ROOT:-/workspace/lpwb-run}
ROBOCASA365_PYTHON=${ROBOCASA365_PYTHON:-/workspace/robocasa365-official/.venv/bin/python}
CHECKPOINT="$RUN_ROOT/outputs/b2_path_time_seed20260818_20k_bs16_20260818/checkpoint-14000"
OUTPUT_ROOT="$RUN_ROOT/closed_loop_nav20_b2_checkpoint14000_seed20260818_20260819"
PAIR_ROOT="$RUN_ROOT/closed_loop_nav20_b0_b1_seed20260818_20260819"
CALIBRATION="$RUN_ROOT/closed_loop_20260818/execution_calibration.json"
MODALITY_CONFIG="$IMPL/configs/robocasa_lpwb_config.py"
MIN_PREFLIGHT_FREE_MIB=${MIN_PREFLIGHT_FREE_MIB:-10240}
MIN_SERVER_FREE_MIB=${MIN_SERVER_FREE_MIB:-2048}
MIN_RUNTIME_FREE_MIB=${MIN_RUNTIME_FREE_MIB:-768}

export PYTHONPATH="$IMPL/src:$IMPL/gr00t_patch:$GROOT"
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libnccl.so.2.30.7${LD_PRELOAD:+:$LD_PRELOAD}
mkdir -p "$OUTPUT_ROOT/logs"

if [[ -f "$OUTPUT_ROOT/B2_NAV20_COMPLETE" ]]; then
  echo "B2 checkpoint-14000 NavigateKitchen evaluation already complete"
  exit 0
fi
if [[ ! -f "$RUN_ROOT/robocasa365_setup/ROBOCASA365_SETUP_COMPLETE" ]]; then
  echo "RoboCasa365 setup completion marker is missing" >&2
  exit 1
fi
if [[ ! -f "$PAIR_ROOT/B0_B1_NAV20_COMPLETE" ]]; then
  echo "B0/B1 NavigateKitchen completion marker is missing" >&2
  exit 1
fi
for path in \
  "$CHECKPOINT/config.json" \
  "$CHECKPOINT/model.safetensors.index.json" \
  "$CHECKPOINT/model-00001-of-00002.safetensors" \
  "$CHECKPOINT/model-00002-of-00002.safetensors" \
  "$CALIBRATION"; do
  if [[ ! -s "$path" ]]; then
    echo "required artifact is missing or empty: $path" >&2
    exit 1
  fi
done

"$GROOT/.venv/bin/python" - "$CHECKPOINT" "$OUTPUT_ROOT/checkpoint_provenance.json" <<'PY'
import json
import sys
from pathlib import Path

checkpoint = Path(sys.argv[1])
files = {}
for path in sorted(checkpoint.iterdir()):
    if path.is_file():
        stat = path.stat()
        files[path.name] = {"size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}
result = {
    "checkpoint": str(checkpoint),
    "checkpoint_step": 14000,
    "save_only_model": True,
    "files": files,
}
Path(sys.argv[2]).write_text(json.dumps(result, indent=2))
PY

gpu_free_mib() {
  nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | sed -n "$(( $1 + 1 ))p"
}
record_gpu_state() {
  {
    echo "# $(date -u +%Y-%m-%dT%H:%M:%SZ) $1"
    nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu \
      --format=csv,noheader,nounits
  } >>"$OUTPUT_ROOT/gpu_states.log"
}

record_gpu_state preflight
for gpu in 0 1; do
  free_mib=$(gpu_free_mib "$gpu")
  if (( free_mib < MIN_PREFLIGHT_FREE_MIB )); then
    echo "GPU $gpu has only ${free_mib} MiB free" >&2
    touch "$OUTPUT_ROOT/ABORTED_PREFLIGHT_MEMORY"
    exit 1
  fi
done

SERVER_PID=
CLIENT_PID=
MONITOR_PID=
cleanup() {
  for pid in "$MONITOR_PID" "$CLIENT_PID" "$SERVER_PID"; do
    [[ -n "$pid" ]] && kill "$pid" 2>/dev/null || true
  done
  for pid in "$MONITOR_PID" "$CLIENT_PID" "$SERVER_PID"; do
    [[ -n "$pid" ]] && wait "$pid" 2>/dev/null || true
  done
}
trap cleanup EXIT INT TERM

env LPWB_METHOD=b2 CUDA_VISIBLE_DEVICES=0 \
  nice -n 10 "$GROOT/.venv/bin/python" "$IMPL/scripts/run_closed_loop_server.py" \
    --method b2 \
    --checkpoint "$CHECKPOINT" \
    --modality-config-path "$MODALITY_CONFIG" \
    --calibration "$CALIBRATION" \
    --device cuda \
    --host 127.0.0.1 \
    --port 5662 \
    --control-dt 0.05 \
    >"$OUTPUT_ROOT/logs/server-b2.log" \
    2>"$OUTPUT_ROOT/logs/server-b2.err.log" &
SERVER_PID=$!

for _ in $(seq 1 120); do
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "B2 policy server exited before becoming ready" >&2
    exit 1
  fi
  if "$GROOT/.venv/bin/python" - <<'PY'
from gr00t.policy.server_client import PolicyClient

client = PolicyClient(host="127.0.0.1", port=5662, timeout_ms=5000, strict=False)
raise SystemExit(0 if client.ping() else 1)
PY
  then
    break
  fi
  sleep 5
done
if ! "$GROOT/.venv/bin/python" - <<'PY'
from gr00t.policy.server_client import PolicyClient

client = PolicyClient(host="127.0.0.1", port=5662, timeout_ms=5000, strict=False)
raise SystemExit(0 if client.ping() else 1)
PY
then
  echo "B2 policy server did not become ready" >&2
  exit 1
fi
record_gpu_state server_ready
server_free=$(gpu_free_mib 0)
if (( server_free < MIN_SERVER_FREE_MIB )); then
  echo "B2 server left only ${server_free} MiB free on GPU 0" >&2
  touch "$OUTPUT_ROOT/ABORTED_SERVER_MEMORY"
  exit 1
fi

env MUJOCO_GL=egl PYOPENGL_PLATFORM=egl CUDA_VISIBLE_DEVICES=1 \
  nice -n 10 "$ROBOCASA365_PYTHON" "$IMPL/scripts/run_closed_loop_eval.py" \
    --method b2 \
    --policy-host 127.0.0.1 \
    --policy-port 5662 \
    --task 'NavigateKitchen::robocasa/NavigateKitchen::20260818::20::450' \
    --output-dir "$OUTPUT_ROOT" \
    --batch-size 1 \
    --n-action-steps 8 \
    --steps-per-render 4 \
    --split target \
    >"$OUTPUT_ROOT/logs/client-b2.log" \
    2>"$OUTPUT_ROOT/logs/client-b2.err.log" &
CLIENT_PID=$!

PARENT_PID=$$
monitor_memory() {
  local low_samples=0
  while kill -0 "$CLIENT_PID" 2>/dev/null; do
    record_gpu_state runtime
    local minimum=999999
    local gpu free_mib
    for gpu in 0 1 2 3; do
      free_mib=$(gpu_free_mib "$gpu")
      (( free_mib < minimum )) && minimum=$free_mib
    done
    if (( minimum < MIN_RUNTIME_FREE_MIB )); then
      low_samples=$((low_samples + 1))
    else
      low_samples=0
    fi
    if (( low_samples >= 2 )); then
      touch "$OUTPUT_ROOT/ABORTED_RUNTIME_MEMORY"
      kill -TERM "$CLIENT_PID" "$PARENT_PID" 2>/dev/null || true
      return 1
    fi
    sleep 30
  done
}
monitor_memory &
MONITOR_PID=$!

wait "$CLIENT_PID"
CLIENT_PID=
kill "$MONITOR_PID" 2>/dev/null || true
wait "$MONITOR_PID" 2>/dev/null || true
MONITOR_PID=
record_gpu_state client_finished

"$GROOT/.venv/bin/python" "$IMPL/scripts/summarize_nav20_threeway.py" \
  --pair-root "$PAIR_ROOT" \
  --b2-root "$OUTPUT_ROOT" \
  --seed-start 20260818 \
  --count 20 \
  --checkpoint-provenance "$OUTPUT_ROOT/checkpoint_provenance.json" \
  --output "$OUTPUT_ROOT/threeway_summary.json" \
  >"$OUTPUT_ROOT/logs/summary.log" \
  2>"$OUTPUT_ROOT/logs/summary.err.log"
touch "$OUTPUT_ROOT/B2_NAV20_COMPLETE"
