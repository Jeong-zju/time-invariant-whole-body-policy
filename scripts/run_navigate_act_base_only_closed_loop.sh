#!/usr/bin/env bash
set -euo pipefail

RUN_ROOT=/workspace/act-base-only-navigate-kitchen-scratch-20260823
SOURCE_ROOT="$RUN_ROOT/source"
LEROBOT_ROOT=/workspace/act-navigate-kitchen-scratch-20260821/source/lerobot
CHECKPOINT="$RUN_ROOT/train-50000/checkpoint-50000-model.pt"
TASKS=/workspace/lpwb-navigate-kitchen-ratefree-20260820/data/NavigateKitchen/lerobot/meta/tasks.jsonl
OUT="$RUN_ROOT/evaluation/closed_loop_checkpoint50000_seed20260818_count30"
SERVER_PORT=5561
SERVER_PID=""

mkdir -p "$OUT/logs" "$OUT/videos/NavigateKitchen"
if [ -f "$OUT/EVALUATION_COMPLETE" ]; then
  echo '{"event":"EVALUATION_ALREADY_COMPLETE"}'
  exit 0
fi

echo '{"event":"WAITING_FOR_BASE_ONLY_TRAINING"}'
while [ ! -f "$RUN_ROOT/train-50000/TRAINING_COMPLETE" ] || [ ! -f "$CHECKPOINT" ]; do
  if ! supervisorctl status act-base-only-navigate-50k | grep -q RUNNING; then
    echo 'TRAINING_SERVICE_EXITED_WITHOUT_COMPLETE_ARTIFACTS' >&2
    exit 1
  fi
  sleep 30
done
echo '{"event":"BASE_ONLY_TRAINING_ARTIFACTS_READY"}'

cleanup() {
  if [ -n "$SERVER_PID" ] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
}
on_exit() {
  rc=$?
  cleanup
  if [ "$rc" -ne 0 ]; then
    touch "$OUT/EVALUATION_FAILED"
    printf 'EVALUATION_FAILED rc=%s %s\n' "$rc" "$(date -u +%FT%TZ)" >&2
  fi
}
trap on_exit EXIT

export PYTHONPATH="/workspace/grootn17:$SOURCE_ROOT/lp-act-v1/src:$LEROBOT_ROOT/src"
export TOKENIZERS_PARALLELISM=false

CUDA_VISIBLE_DEVICES=0 /workspace/grootn16/.venv/bin/python \
  -m robocasa_act_navigate.eval_server \
  --checkpoint "$CHECKPOINT" \
  --tasks "$TASKS" \
  --device cuda \
  --host 127.0.0.1 \
  --port "$SERVER_PORT" \
  >"$OUT/logs/server.log" 2>"$OUT/logs/server.err.log" &
SERVER_PID=$!

ready=0
for _ in $(seq 1 120); do
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo MODEL_SERVER_EXITED_BEFORE_READY >&2
    exit 1
  fi
  if /workspace/grootn16/.venv/bin/python - "$SERVER_PORT" <<'PY'
import socket
import sys
with socket.create_connection(("127.0.0.1", int(sys.argv[1])), timeout=1):
    pass
PY
  then
    ready=1
    break
  fi
  sleep 2
done
test "$ready" -eq 1
echo '{"event":"BASE_ONLY_MODEL_SERVER_READY"}'

RESULTS_JSONL="$OUT/results.jsonl"
touch "$RESULTS_JSONL"
for seed in $(seq 20260818 20260847); do
  if grep -q "\"seed\":$seed," "$RESULTS_JSONL"; then
    continue
  fi
  seed_dir="$OUT/videos/NavigateKitchen/seed_$seed"
  client_log="$OUT/logs/client-seed-$seed.log"
  client_err="$OUT/logs/client-seed-$seed.err.log"
  mkdir -p "$seed_dir"
  cd /workspace/grootn17
  CUDA_VISIBLE_DEVICES=1 \
  MUJOCO_GL=egl \
  MUJOCO_EGL_DEVICE_ID=1 \
  PYOPENGL_PLATFORM=egl \
  PYTHONPATH=/workspace/grootn17 \
  /workspace/robocasa365-official/.venv/bin/python \
    gr00t/eval/rollout_policy.py \
    --n-episodes 1 \
    --n-envs 1 \
    --policy-client-host 127.0.0.1 \
    --policy-client-port "$SERVER_PORT" \
    --max-episode-steps 450 \
    --env-name robocasa365_panda_omron/NavigateKitchen_PandaOmron_Env \
    --n-action-steps 8 \
    --video-dir "$seed_dir" \
    --seed "$seed" \
    --robocasa-split target \
    >"$client_log" 2>"$client_err"
  success="$(/workspace/grootn16/.venv/bin/python - "$client_log" <<'PY'
import re
import sys
text = open(sys.argv[1], encoding="utf-8", errors="replace").read()
matches = re.findall(r"success rate:\s*([0-9.eE+-]+)", text)
if not matches:
    raise SystemExit("missing success rate")
print("true" if float(matches[-1]) >= 0.5 else "false")
PY
)"
  printf '{"seed":%s,"success":%s}\n' "$seed" "$success" >>"$RESULTS_JSONL"
  printf 'EPISODE_COMPLETE seed=%s success=%s %s\n' "$seed" "$success" "$(date -u +%FT%TZ)"
done

/workspace/grootn16/.venv/bin/python - "$RESULTS_JSONL" "$OUT" "$CHECKPOINT" <<'PY'
import json
from pathlib import Path
import sys
jsonl, out, checkpoint = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
records = [json.loads(line) for line in jsonl.read_text().splitlines() if line.strip()]
records = sorted({int(row["seed"]): row for row in records}.values(), key=lambda row: int(row["seed"]))
expected = list(range(20260818, 20260848))
assert [int(row["seed"]) for row in records] == expected
successes = sum(bool(row["success"]) for row in records)
videos = sorted(str(path.relative_to(out)) for path in (out / "videos").rglob("*.mp4") if path.stat().st_size > 0)
assert len(videos) >= 30, f"expected 30 non-empty videos, got {len(videos)}"
payload = {
  "task": "NavigateKitchen",
  "model": "ACT base-only scratch",
  "learned_action": ["vx", "vy", "omega"],
  "fixed_native_hold": {
    "torso": 0.0,
    "control_mode": 1.0,
    "end_effector_position": [0.0, 0.0, 0.0],
    "end_effector_rotation": [0.0, 0.0, 0.0],
    "gripper_close": -1.0,
  },
  "checkpoint": checkpoint,
  "num_episodes": 30,
  "num_successes": successes,
  "success_rate": successes / 30,
  "episodes": records,
  "seed_start": expected[0],
  "seed_end": expected[-1],
  "max_episode_steps": 450,
  "n_action_steps": 8,
  "nominal_hz": 20,
  "robocasa_split": "target",
  "video_count": len(videos),
}
(out / "results.json").write_text(json.dumps(payload, indent=2) + "\n")
(out / "summary.json").write_text(json.dumps({key: value for key, value in payload.items() if key != "episodes"}, indent=2) + "\n")
PY
touch "$OUT/EVALUATION_COMPLETE"
echo '{"event":"EVALUATION_COMPLETE"}'
