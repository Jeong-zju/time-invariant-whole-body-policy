#!/usr/bin/env bash
set -euo pipefail

GROOT=${GROOT:-/workspace/grootn16}
IMPL=${IMPL:-/workspace/time-invariant-whole-body-policy/implementations/lp-groot-wholebody-v1}
RUN_ROOT=${RUN_ROOT:-/workspace/lpwb-run}
ROBOCASA365_PYTHON=${ROBOCASA365_PYTHON:-/workspace/robocasa365-official/.venv/bin/python}
OUTPUT_ROOT=${OUTPUT_ROOT:-$RUN_ROOT/closed_loop_nav20_b0_b1_seed20260818_20260819}
CALIBRATION=${CALIBRATION:-$RUN_ROOT/closed_loop_20260818/execution_calibration.json}
MODALITY_CONFIG="$IMPL/configs/robocasa_lpwb_config.py"
B0_CHECKPOINT="$RUN_ROOT/outputs/b0_groot_command_seed20260818_20k_bs16_20260818/checkpoint-20000"
B1_CHECKPOINT="$RUN_ROOT/outputs/b1_pose_time_seed20260818_20k_bs16_20260818/checkpoint-20000"
MIN_PREFLIGHT_FREE_MIB=${MIN_PREFLIGHT_FREE_MIB:-10240}
MIN_SERVER_FREE_MIB=${MIN_SERVER_FREE_MIB:-2048}
MIN_RUNTIME_FREE_MIB=${MIN_RUNTIME_FREE_MIB:-768}

export PYTHONPATH="$IMPL/src:$IMPL/gr00t_patch:$GROOT"
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libnccl.so.2.30.7${LD_PRELOAD:+:$LD_PRELOAD}
mkdir -p "$OUTPUT_ROOT/logs"

if [[ -f "$OUTPUT_ROOT/B0_B1_NAV20_COMPLETE" ]]; then
  echo "paired NavigateKitchen evaluation already complete"
  exit 0
fi

if [[ ! -f "$RUN_ROOT/robocasa365_setup/ROBOCASA365_SETUP_COMPLETE" ]]; then
  echo "RoboCasa365 setup completion marker is missing" >&2
  exit 1
fi

for path in \
  "$B0_CHECKPOINT/config.json" \
  "$B0_CHECKPOINT/model.safetensors.index.json" \
  "$B1_CHECKPOINT/config.json" \
  "$B1_CHECKPOINT/model.safetensors.index.json" \
  "$CALIBRATION"; do
  if [[ ! -s "$path" ]]; then
    echo "required artifact is missing or empty: $path" >&2
    exit 1
  fi
done

gpu_free_mib() {
  local gpu=$1
  nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | sed -n "$((gpu + 1))p"
}

record_gpu_state() {
  local phase=$1
  {
    echo "# $(date -u +%Y-%m-%dT%H:%M:%SZ) $phase"
    nvidia-smi \
      --query-gpu=index,memory.used,memory.free,utilization.gpu \
      --format=csv,noheader,nounits
  } >>"$OUTPUT_ROOT/gpu_states.log"
}

record_gpu_state preflight
for gpu in 0 1 2 3; do
  free_mib=$(gpu_free_mib "$gpu")
  if (( free_mib < MIN_PREFLIGHT_FREE_MIB )); then
    echo "GPU $gpu has only ${free_mib} MiB free; need ${MIN_PREFLIGHT_FREE_MIB} MiB" >&2
    touch "$OUTPUT_ROOT/ABORTED_PREFLIGHT_MEMORY"
    exit 1
  fi
done

SERVER_PIDS=()
CLIENT_PIDS=()
MONITOR_PID=
cleanup() {
  if [[ -n "$MONITOR_PID" ]]; then
    kill "$MONITOR_PID" 2>/dev/null || true
    wait "$MONITOR_PID" 2>/dev/null || true
  fi
  for pid in "${CLIENT_PIDS[@]}" "${SERVER_PIDS[@]}"; do
    [[ -n "$pid" ]] && kill "$pid" 2>/dev/null || true
  done
  for pid in "${CLIENT_PIDS[@]}" "${SERVER_PIDS[@]}"; do
    [[ -n "$pid" ]] && wait "$pid" 2>/dev/null || true
  done
}
trap cleanup EXIT INT TERM

start_server() {
  local method=$1
  local gpu=$2
  local port=$3
  local checkpoint=$4
  local extra=()
  if [[ "$method" == b1 ]]; then
    extra=(--calibration "$CALIBRATION")
  fi
  env LPWB_METHOD="$method" CUDA_VISIBLE_DEVICES="$gpu" \
    nice -n 10 "$GROOT/.venv/bin/python" "$IMPL/scripts/run_closed_loop_server.py" \
      --method "$method" \
      --checkpoint "$checkpoint" \
      --modality-config-path "$MODALITY_CONFIG" \
      "${extra[@]}" \
      --device cuda \
      --host 127.0.0.1 \
      --port "$port" \
      --control-dt 0.05 \
      >"$OUTPUT_ROOT/logs/server-${method}.log" \
      2>"$OUTPUT_ROOT/logs/server-${method}.err.log" &
  SERVER_PIDS+=("$!")

  for _ in $(seq 1 120); do
    if ! kill -0 "$!" 2>/dev/null; then
      echo "$method policy server exited before becoming ready" >&2
      return 1
    fi
    if "$GROOT/.venv/bin/python" - "$port" <<'PY'
import sys
from gr00t.policy.server_client import PolicyClient

client = PolicyClient(
    host="127.0.0.1", port=int(sys.argv[1]), timeout_ms=5000, strict=False
)
raise SystemExit(0 if client.ping() else 1)
PY
    then
      free_mib=$(gpu_free_mib "$gpu")
      record_gpu_state "${method}_server_ready"
      if (( free_mib < MIN_SERVER_FREE_MIB )); then
        echo "$method server left only ${free_mib} MiB free on GPU $gpu" >&2
        touch "$OUTPUT_ROOT/ABORTED_SERVER_MEMORY"
        return 1
      fi
      return 0
    fi
    sleep 5
  done
  echo "$method policy server on port $port did not become ready" >&2
  return 1
}

start_server b0 0 5660 "$B0_CHECKPOINT"
start_server b1 2 5661 "$B1_CHECKPOINT"

run_client() {
  local method=$1
  local gpu=$2
  local port=$3
  if [[ -f "$OUTPUT_ROOT/$method/CLOSED_LOOP_COMPLETE" ]]; then
    return 0
  fi
  env MUJOCO_GL=egl PYOPENGL_PLATFORM=egl CUDA_VISIBLE_DEVICES="$gpu" \
    nice -n 10 "$ROBOCASA365_PYTHON" "$IMPL/scripts/run_closed_loop_eval.py" \
      --method "$method" \
      --policy-host 127.0.0.1 \
      --policy-port "$port" \
      --task 'NavigateKitchen::robocasa/NavigateKitchen::20260818::20::450' \
      --output-dir "$OUTPUT_ROOT/$method" \
      --batch-size 1 \
      --n-action-steps 8 \
      --steps-per-render 4 \
      --split target \
      >"$OUTPUT_ROOT/logs/client-${method}.log" \
      2>"$OUTPUT_ROOT/logs/client-${method}.err.log"
}

run_client b0 1 5660 &
B0_CLIENT_PID=$!
CLIENT_PIDS+=("$B0_CLIENT_PID")
run_client b1 3 5661 &
B1_CLIENT_PID=$!
CLIENT_PIDS+=("$B1_CLIENT_PID")

PARENT_PID=$$
monitor_runtime_memory() {
  local low_samples=0
  while kill -0 "$B0_CLIENT_PID" 2>/dev/null || kill -0 "$B1_CLIENT_PID" 2>/dev/null; do
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
      echo "two consecutive low-memory samples; stopping evaluation only" \
        >>"$OUTPUT_ROOT/logs/runtime-memory-abort.log"
      touch "$OUTPUT_ROOT/ABORTED_RUNTIME_MEMORY"
      kill -TERM "$B0_CLIENT_PID" "$B1_CLIENT_PID" 2>/dev/null || true
      kill -TERM "$PARENT_PID" 2>/dev/null || true
      return 1
    fi
    sleep 30
  done
}
monitor_runtime_memory &
MONITOR_PID=$!

status=0
wait "$B0_CLIENT_PID" || status=1
wait "$B1_CLIENT_PID" || status=1
kill "$MONITOR_PID" 2>/dev/null || true
wait "$MONITOR_PID" 2>/dev/null || true
MONITOR_PID=
record_gpu_state clients_finished
if [[ "$status" != 0 ]]; then
  echo "one or more NavigateKitchen clients failed" >&2
  exit 1
fi

"$GROOT/.venv/bin/python" "$IMPL/scripts/summarize_closed_loop_pair.py" \
  --root "$OUTPUT_ROOT" \
  --task NavigateKitchen \
  --gym-id robocasa/NavigateKitchen \
  --seed-start 20260818 \
  --count 20 \
  --output "$OUTPUT_ROOT/summary.json" \
  >"$OUTPUT_ROOT/logs/summary.log" \
  2>"$OUTPUT_ROOT/logs/summary.err.log"
touch "$OUTPUT_ROOT/B0_B1_NAV20_COMPLETE"
