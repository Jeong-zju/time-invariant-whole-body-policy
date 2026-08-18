#!/usr/bin/env bash
set -euo pipefail

GROOT=${GROOT:-/workspace/grootn16}
IMPL=${IMPL:-/workspace/time-invariant-whole-body-policy/implementations/lp-groot-wholebody-v1}
RUN_ROOT=${RUN_ROOT:-/workspace/lpwb-run}
ROBOCASA365_PYTHON=${ROBOCASA365_PYTHON:-/workspace/robocasa365-official/.venv/bin/python}
SETUP_DIR="$RUN_ROOT/robocasa365_setup"
EVAL_DIR="$RUN_ROOT/evaluation_20k_seed20260818"
OUTPUT_ROOT="$RUN_ROOT/closed_loop_20k_seed20260818"
CALIBRATION="$EVAL_DIR/execution_calibration.json"
MODALITY_CONFIG="$IMPL/configs/robocasa_lpwb_config.py"
B0_CHECKPOINT="$RUN_ROOT/outputs/b0_groot_command_seed20260818_20k_bs16_20260818/checkpoint-20000"
B1_CHECKPOINT="$RUN_ROOT/outputs/b1_pose_time_seed20260818_20k_bs16_20260818/checkpoint-20000"
B2_CHECKPOINT="$RUN_ROOT/outputs/b2_path_time_seed20260818_20k_bs16_20260818/checkpoint-20000"

export PYTHONPATH="$IMPL/src:$IMPL/gr00t_patch:$GROOT"
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libnccl.so.2.30.7${LD_PRELOAD:+:$LD_PRELOAD}
mkdir -p "$OUTPUT_ROOT" "$RUN_ROOT/logs"

while [[ ! -f "$SETUP_DIR/ROBOCASA365_SETUP_COMPLETE" || \
         ! -f "$EVAL_DIR/OPEN_LOOP_EVALUATION_COMPLETE" ]]; do
  sleep 60
done

if [[ ! -f "$SETUP_DIR/ROBOCASA365_ENVIRONMENTS_VALIDATED" ]]; then
  MUJOCO_GL=egl PYOPENGL_PLATFORM=egl CUDA_VISIBLE_DEVICES=3 \
    "$ROBOCASA365_PYTHON" "$IMPL/scripts/validate_robocasa365_envs.py" \
      --output "$SETUP_DIR/environments.json" \
      --seed 20260818 \
      >"$RUN_ROOT/logs/robocasa365-validation.log" \
      2>"$RUN_ROOT/logs/robocasa365-validation.err.log"
fi

SERVER_PIDS=()
cleanup() {
  for pid in "${SERVER_PIDS[@]}"; do
    kill "$pid" 2>/dev/null || true
  done
  for pid in "${SERVER_PIDS[@]}"; do
    wait "$pid" 2>/dev/null || true
  done
}
trap cleanup EXIT INT TERM

start_server() {
  local method=$1
  local gpu=$2
  local port=$3
  local checkpoint=$4
  local extra=()
  if [[ "$method" != b0 ]]; then
    extra=(--calibration "$CALIBRATION")
  fi
  LPWB_METHOD="$method" CUDA_VISIBLE_DEVICES="$gpu" \
    "$GROOT/.venv/bin/python" "$IMPL/scripts/run_closed_loop_server.py" \
      --method "$method" \
      --checkpoint "$checkpoint" \
      --modality-config-path "$MODALITY_CONFIG" \
      "${extra[@]}" \
      --device cuda \
      --host 127.0.0.1 \
      --port "$port" \
      --control-dt 0.05 \
      >"$RUN_ROOT/logs/server-${method}-20k.log" \
      2>"$RUN_ROOT/logs/server-${method}-20k.err.log" &
  SERVER_PIDS+=("$!")
}

start_server b0 0 5650 "$B0_CHECKPOINT"
start_server b1 1 5651 "$B1_CHECKPOINT"
start_server b2 2 5652 "$B2_CHECKPOINT"

for port in 5650 5651 5652; do
  "$GROOT/.venv/bin/python" - "$port" <<'PY'
import sys
import time
from gr00t.policy.server_client import PolicyClient

port = int(sys.argv[1])
for _ in range(120):
    client = PolicyClient(host="127.0.0.1", port=port, timeout_ms=5000, strict=False)
    if client.ping():
        break
    time.sleep(5)
else:
    raise RuntimeError(f"policy server on port {port} did not become ready")
PY
done

run_client() {
  local method=$1
  local port=$2
  if [[ -f "$OUTPUT_ROOT/$method/CLOSED_LOOP_COMPLETE" ]]; then
    return 0
  fi
  MUJOCO_GL=egl PYOPENGL_PLATFORM=egl CUDA_VISIBLE_DEVICES=3 \
    "$ROBOCASA365_PYTHON" "$IMPL/scripts/run_closed_loop_eval.py" \
      --method "$method" \
      --policy-host 127.0.0.1 \
      --policy-port "$port" \
      --task 'NavigateKitchen::robocasa/NavigateKitchen::20260818::30::450' \
      --task 'PickPlaceCounterToStove::robocasa/PickPlaceCounterToStove::20260818::30::600' \
      --task 'DeliverStraw::robocasa/DeliverStraw::20260818::30::2550' \
      --output-dir "$OUTPUT_ROOT/$method" \
      --batch-size 3 \
      --n-action-steps 8 \
      --steps-per-render 4 \
      --split target \
      >"$RUN_ROOT/logs/client-${method}-20k.log" \
      2>"$RUN_ROOT/logs/client-${method}-20k.err.log"
}

run_client b0 5650 &
B0_CLIENT_PID=$!
run_client b1 5651 &
B1_CLIENT_PID=$!
run_client b2 5652 &
B2_CLIENT_PID=$!

status=0
wait "$B0_CLIENT_PID" || status=1
wait "$B1_CLIENT_PID" || status=1
wait "$B2_CLIENT_PID" || status=1
if [[ "$status" != 0 ]]; then
  echo "one or more closed-loop clients failed" >&2
  exit 1
fi

"$GROOT/.venv/bin/python" "$IMPL/scripts/summarize_closed_loop_matrix.py" \
  --root "$OUTPUT_ROOT" \
  --count 30 \
  --seed-start 20260818 \
  --output "$OUTPUT_ROOT/summary.json" \
  >"$RUN_ROOT/logs/closed-loop-summary-20k.log" \
  2>"$RUN_ROOT/logs/closed-loop-summary-20k.err.log"
touch "$OUTPUT_ROOT/CLOSED_LOOP_MATRIX_COMPLETE"
