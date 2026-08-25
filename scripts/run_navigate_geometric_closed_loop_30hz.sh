#!/usr/bin/env bash
set -euo pipefail

RUN_ROOT=${RUN_ROOT:-/workspace/act-geometric-navigate-kitchen-scratch-20260823}
BASE_RUN=/workspace/act-base-only-navigate-kitchen-scratch-20260823
CHECKPOINT="$RUN_ROOT/train-50000/checkpoint-50000-model.pt"
TASKS=/workspace/lpwb-navigate-kitchen-ratefree-20260820/data/NavigateKitchen/lerobot/meta/tasks.jsonl
CALIBRATION="$BASE_RUN/config/base-rate-calibration-30hz.json"
OUT="$RUN_ROOT/evaluation/closed_loop_30hz_checkpoint50000_seed20260818_count30"
PREFLIGHT="$RUN_ROOT/evaluation/preflight_30hz_seed20260817_horizon12"
PORT=5563
SERVER_PID=""

mkdir -p "$OUT/logs" "$OUT/videos/NavigateKitchen" "$PREFLIGHT"
if [ -f "$OUT/EVALUATION_COMPLETE" ]; then
  echo '{"event":"EVALUATION_ALREADY_COMPLETE"}'
  exit 0
fi
while [ ! -f "$RUN_ROOT/TRAINING_COMPLETE" ] || [ ! -f "$CHECKPOINT" ]; do
  if ! supervisorctl status "${TRAIN_SERVICE:-act-geometric-navigate-50k}" | grep -q RUNNING; then
    echo 'TRAINING_SERVICE_EXITED_WITHOUT_COMPLETE_ARTIFACTS' >&2
    exit 1
  fi
  sleep 30
done

cleanup() {
  if [ -n "$SERVER_PID" ] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
}
on_exit() {
  rc=$?
  cleanup
  if [ "$rc" -ne 0 ]; then touch "$OUT/EVALUATION_FAILED"; fi
}
trap on_exit EXIT

export PYTHONPATH="/workspace/grootn17:$BASE_RUN/source/lp-act-v1/src:/workspace/act-navigate-kitchen-scratch-20260821/source/lerobot/src"
export TOKENIZERS_PARALLELISM=false

CUDA_VISIBLE_DEVICES=0 /workspace/grootn16/.venv/bin/python \
  -m robocasa_act_navigate.geometric_eval_server \
  --checkpoint "$CHECKPOINT" \
  --tasks "$TASKS" \
  --calibration "$CALIBRATION" \
  --device cuda \
  --host 127.0.0.1 \
  --port "$PORT" \
  --control-hz 30 \
  --replan-ticks 12 \
  --trace "$OUT/logs/follower-trace.jsonl" \
  >"$OUT/logs/server.log" 2>"$OUT/logs/server.err.log" &
SERVER_PID=$!

ready=0
for _ in $(seq 1 120); do
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo MODEL_SERVER_EXITED_BEFORE_READY >&2
    exit 1
  fi
  if /workspace/grootn16/.venv/bin/python - "$PORT" <<'PY'
import socket,sys
with socket.create_connection(("127.0.0.1",int(sys.argv[1])),timeout=1): pass
PY
  then ready=1; break; fi
  sleep 2
done
test "$ready" -eq 1

if [ ! -f "$PREFLIGHT/PREFLIGHT_PASSED" ]; then
  mkdir -p "$PREFLIGHT/videos"
  cd /workspace/grootn17
  CUDA_VISIBLE_DEVICES=1 MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=1 PYOPENGL_PLATFORM=egl \
  PYTHONPATH="/workspace/grootn17:$BASE_RUN/source/lp-act-v1/src" \
  /workspace/robocasa365-official/.venv/bin/python \
    -m robocasa_act_navigate.rollout_policy_30hz \
    --policy-client-host 127.0.0.1 \
    --policy-client-port "$PORT" \
    --max-episode-steps 12 \
    --video-dir "$PREFLIGHT/videos" \
    --seed 20260817 \
    --robocasa-split target \
    >"$PREFLIGHT/client.log" 2>"$PREFLIGHT/client.err.log"
  grep -q ENVIRONMENT_30HZ_VERIFIED "$PREFLIGHT/client.log"
  test -n "$(find "$PREFLIGHT/videos" -type f -name '*.mp4' -size +0c -print -quit)"
  touch "$PREFLIGHT/PREFLIGHT_PASSED"
fi
echo '{"event":"GEOMETRIC_30HZ_PREFLIGHT_PASSED","counted_episodes":0}'

RESULTS="$OUT/results.jsonl"
touch "$RESULTS"
for seed in $(seq 20260818 20260847); do
  if grep -q "\"seed\":$seed," "$RESULTS"; then continue; fi
  seed_dir="$OUT/videos/NavigateKitchen/seed_$seed"
  mkdir -p "$seed_dir"
  log="$OUT/logs/client-seed-$seed.log"
  err="$OUT/logs/client-seed-$seed.err.log"
  cd /workspace/grootn17
  CUDA_VISIBLE_DEVICES=1 MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=1 PYOPENGL_PLATFORM=egl \
  PYTHONPATH="/workspace/grootn17:$BASE_RUN/source/lp-act-v1/src" \
  /workspace/robocasa365-official/.venv/bin/python \
    -m robocasa_act_navigate.rollout_policy_30hz \
    --policy-client-host 127.0.0.1 \
    --policy-client-port "$PORT" \
    --max-episode-steps 675 \
    --video-dir "$seed_dir" \
    --seed "$seed" \
    --robocasa-split target >"$log" 2>"$err"
  success="$(/workspace/grootn16/.venv/bin/python - "$log" <<'PY'
import re,sys
text=open(sys.argv[1],encoding='utf-8',errors='replace').read()
found=re.findall(r'success rate:\s*([0-9.eE+-]+)',text)
if not found: raise SystemExit('missing success rate')
print('true' if float(found[-1]) >= .5 else 'false')
PY
)"
  video="$(find "$seed_dir" -type f -name '*.mp4' -size +0c -print -quit)"
  test -n "$video"
  printf '{"seed":%s,"success":%s,"video":"%s"}\n' "$seed" "$success" "$video" >>"$RESULTS"
  printf 'EPISODE_COMPLETE seed=%s success=%s\n' "$seed" "$success"
done

/workspace/grootn16/.venv/bin/python - "$RESULTS" "$OUT" "$CHECKPOINT" <<'PY'
import json,sys
from pathlib import Path
jsonl,out,checkpoint=Path(sys.argv[1]),Path(sys.argv[2]),sys.argv[3]
records=[json.loads(line) for line in jsonl.read_text().splitlines() if line.strip()]
records=sorted({int(row['seed']):row for row in records}.values(),key=lambda row:int(row['seed']))
expected=list(range(20260818,20260848))
assert [int(row['seed']) for row in records] == expected
videos=[p for p in (out/'videos').rglob('*.mp4') if p.stat().st_size>0]
assert len(videos) >= 30
successes=sum(bool(row['success']) for row in records)
payload={
 'task':'NavigateKitchen',
 'model':'ACT fixed-token common-origin measured-SE2 path scratch',
 'checkpoint':checkpoint,
 'num_episodes':30,
 'num_successes':successes,
 'success_rate':successes/30,
 'episodes':records,
 'seed_start':expected[0],
 'seed_end':expected[-1],
 'source_label_hz':20,
 'controller_hz':30,
 'controller_timestep_s':1/30,
 'model_replan_ticks':12,
 'model_replan_period_s':12/30,
 'max_episode_steps':675,
 'max_episode_seconds':675/30,
 'baseline_matched_seconds':True,
 'robocasa_split':'target',
 'n_action_steps':1,
 'measured_pose_feedback_every_tick':True,
 'predicts_time_duration_velocity_rate':False,
 'video_count':len(videos),
}
(out/'results.json').write_text(json.dumps(payload,indent=2)+'\n')
(out/'summary.json').write_text(json.dumps({k:v for k,v in payload.items() if k!='episodes'},indent=2)+'\n')
PY
touch "$OUT/EVALUATION_COMPLETE"
echo '{"event":"EVALUATION_COMPLETE"}'
