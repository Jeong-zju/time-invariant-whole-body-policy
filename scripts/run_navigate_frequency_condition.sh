#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 4 ]; then
  echo "usage: $0 METHOD CONTROL_HZ GPU PORT" >&2
  exit 2
fi
METHOD=$1
HZ=$2
GPU=$3
PORT=$4
case "$METHOD" in baseline_native|baseline|geometric) ;; *) echo "METHOD must be baseline_native, baseline, or geometric" >&2; exit 2;; esac
case "$HZ" in 10|20|50) ;; *) echo "CONTROL_HZ must be 10, 20, or 50" >&2; exit 2;; esac

RUN_ROOT=${RUN_ROOT:-/workspace/act-frequency-sweep-navigate-kitchen-20260823}
SOURCE=/workspace/act-base-only-navigate-kitchen-scratch-20260823/source/lp-act-v1
BASE_CHECKPOINT=/workspace/act-base-only-navigate-kitchen-scratch-20260823/train-50000/checkpoint-50000-model.pt
GEOMETRIC_CHECKPOINT=/workspace/act-geometric-continuousyaw-navigate-kitchen-scratch-20260823/train-50000/checkpoint-50000-model.pt
TASKS=/workspace/lpwb-navigate-kitchen-ratefree-20260820/data/NavigateKitchen/lerobot/meta/tasks.jsonl
CALIBRATION="$RUN_ROOT/controller-gates/${HZ}hz/base-rate-calibration-${HZ}hz.json"
OUT="$RUN_ROOT/evaluation/${METHOD}_${HZ}hz_checkpoint50000_seed20260818_count30"
PREFLIGHT="$RUN_ROOT/preflight/${METHOD}_${HZ}hz_seed20260817"
if [ "$METHOD" = baseline_native ]; then
  REPLAN_TICKS=8
  CLIENT_ACTION_STEPS=8
else
  REPLAN_TICKS=$((HZ * 2 / 5))
  CLIENT_ACTION_STEPS=1
fi
MAX_EPISODE_STEPS=$((HZ * 45 / 2))
SERVER_PID=""

mkdir -p "$OUT/logs" "$OUT/videos/NavigateKitchen" "$PREFLIGHT/videos"
if [ -f "$OUT/EVALUATION_COMPLETE" ]; then
  echo "{\"event\":\"EVALUATION_ALREADY_COMPLETE\",\"method\":\"$METHOD\",\"control_hz\":$HZ}"
  exit 0
fi
if [ "$METHOD" = geometric ]; then
  test -f "$RUN_ROOT/controller-gates/${HZ}hz/GATE_COMPLETE"
  test -f "$CALIBRATION"
  CHECKPOINT=$GEOMETRIC_CHECKPOINT
else
  CHECKPOINT=$BASE_CHECKPOINT
fi
test -f "$CHECKPOINT"

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

export PYTHONPATH="/workspace/grootn17:$SOURCE/src:/workspace/act-navigate-kitchen-scratch-20260821/source/lerobot/src"
export TOKENIZERS_PARALLELISM=false
if [ "$METHOD" = baseline ]; then
  CUDA_VISIBLE_DEVICES="$GPU" /workspace/grootn16/.venv/bin/python \
    -m robocasa_act_navigate.frequency_eval_server \
    --checkpoint "$CHECKPOINT" \
    --tasks "$TASKS" \
    --device cuda \
    --host 127.0.0.1 \
    --port "$PORT" \
    --source-action-hz 20 \
    --control-hz "$HZ" \
    --replan-period-s 0.4 \
    --trace "$OUT/logs/action-trace.jsonl" \
    >"$OUT/logs/server.log" 2>"$OUT/logs/server.err.log" &
elif [ "$METHOD" = baseline_native ]; then
  CUDA_VISIBLE_DEVICES="$GPU" /workspace/grootn16/.venv/bin/python \
    -m robocasa_act_navigate.eval_server \
    --checkpoint "$CHECKPOINT" \
    --tasks "$TASKS" \
    --device cuda \
    --host 127.0.0.1 \
    --port "$PORT" \
    >"$OUT/logs/server.log" 2>"$OUT/logs/server.err.log" &
else
  CUDA_VISIBLE_DEVICES="$GPU" /workspace/grootn16/.venv/bin/python \
    -m robocasa_act_navigate.geometric_eval_server \
    --checkpoint "$CHECKPOINT" \
    --tasks "$TASKS" \
    --calibration "$CALIBRATION" \
    --device cuda \
    --host 127.0.0.1 \
    --port "$PORT" \
    --control-hz "$HZ" \
    --replan-ticks "$REPLAN_TICKS" \
    --trace "$OUT/logs/follower-trace.jsonl" \
    >"$OUT/logs/server.log" 2>"$OUT/logs/server.err.log" &
fi
SERVER_PID=$!

ready=0
for _ in $(seq 1 120); do
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo MODEL_SERVER_EXITED_BEFORE_READY >&2
    exit 1
  fi
  if /workspace/grootn16/.venv/bin/python -c \
    "import socket; s=socket.create_connection(('127.0.0.1',$PORT),1); s.close()" \
    2>/dev/null; then
    ready=1
    break
  fi
  sleep 2
done
test "$ready" -eq 1

if [ ! -f "$PREFLIGHT/PREFLIGHT_PASSED" ]; then
  cd /workspace/grootn17
  CUDA_VISIBLE_DEVICES="$GPU" MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID="$GPU" PYOPENGL_PLATFORM=egl \
    /workspace/robocasa365-official/.venv/bin/python \
      -m robocasa_act_navigate.rollout_policy_at_hz \
      --policy-client-host 127.0.0.1 \
      --policy-client-port "$PORT" \
      --control-hz "$HZ" \
      --max-episode-steps "$REPLAN_TICKS" \
      --n-action-steps "$CLIENT_ACTION_STEPS" \
      --video-dir "$PREFLIGHT/videos" \
      --seed 20260817 \
      --robocasa-split target \
      >"$PREFLIGHT/client.log" 2>"$PREFLIGHT/client.err.log"
  grep -q ENVIRONMENT_FREQUENCY_VERIFIED "$PREFLIGHT/client.log"
  test -n "$(find "$PREFLIGHT/videos" -type f -name '*.mp4' -size +0c -print -quit)"
  touch "$PREFLIGHT/PREFLIGHT_PASSED"
fi

RESULTS="$OUT/results.jsonl"
touch "$RESULTS"
for seed in $(seq 20260818 20260847); do
  if grep -q "\"seed\":$seed," "$RESULTS"; then continue; fi
  seed_dir="$OUT/videos/NavigateKitchen/seed_$seed"
  mkdir -p "$seed_dir"
  log="$OUT/logs/client-seed-$seed.log"
  err="$OUT/logs/client-seed-$seed.err.log"
  cd /workspace/grootn17
  CUDA_VISIBLE_DEVICES="$GPU" MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID="$GPU" PYOPENGL_PLATFORM=egl \
    /workspace/robocasa365-official/.venv/bin/python \
      -m robocasa_act_navigate.rollout_policy_at_hz \
      --policy-client-host 127.0.0.1 \
      --policy-client-port "$PORT" \
      --control-hz "$HZ" \
      --max-episode-steps "$MAX_EPISODE_STEPS" \
      --n-action-steps "$CLIENT_ACTION_STEPS" \
      --video-dir "$seed_dir" \
      --seed "$seed" \
      --robocasa-split target \
      >"$log" 2>"$err"
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
  printf 'EPISODE_COMPLETE method=%s hz=%s seed=%s success=%s\n' "$METHOD" "$HZ" "$seed" "$success"
done

/workspace/grootn16/.venv/bin/python - "$RESULTS" "$OUT" "$CHECKPOINT" "$METHOD" "$HZ" "$REPLAN_TICKS" "$MAX_EPISODE_STEPS" <<'PY'
import json,sys
from pathlib import Path
jsonl,out,checkpoint,method=Path(sys.argv[1]),Path(sys.argv[2]),sys.argv[3],sys.argv[4]
hz,replan_ticks,max_steps=map(int,sys.argv[5:8])
records=[json.loads(line) for line in jsonl.read_text().splitlines() if line.strip()]
records=sorted({int(row['seed']):row for row in records}.values(),key=lambda row:int(row['seed']))
expected=list(range(20260818,20260848))
assert [int(row['seed']) for row in records] == expected
videos=[p for p in (out/'videos').rglob('*.mp4') if p.stat().st_size>0]
assert len(videos) >= 30
successes=sum(bool(row['success']) for row in records)
payload={
 'task':'NavigateKitchen',
 'method':method,
 'checkpoint':checkpoint,
 'num_episodes':30,
 'num_successes':successes,
 'success_rate':successes/30,
 'episodes':records,
 'seed_start':expected[0],
 'seed_end':expected[-1],
 'source_action_or_label_hz':20,
 'controller_hz':hz,
 'controller_timestep_s':1/hz,
 'model_replan_ticks':replan_ticks,
 'model_replan_period_s':replan_ticks/hz,
 'max_episode_steps':max_steps,
 'max_episode_seconds':max_steps/hz,
 'robocasa_split':'target',
 'n_action_steps':8 if method=='baseline_native' else 1,
 'base_only':True,
 'fixed_upper_body':True,
 'baseline_time_resampling': (
   'overlap_weighted_zoh_integral_preserving' if method=='baseline'
   else 'none_token_index_equals_control_tick' if method=='baseline_native'
   else None
 ),
 'geometric_predicts_time_duration_velocity_rate': False if method=='geometric' else None,
 'measured_pose_feedback_every_tick': True if method=='geometric' else False,
 'video_count':len(videos),
}
(out/'results.json').write_text(json.dumps(payload,indent=2)+'\n')
(out/'summary.json').write_text(json.dumps({k:v for k,v in payload.items() if k!='episodes'},indent=2)+'\n')
PY
touch "$OUT/EVALUATION_COMPLETE"
echo "{\"event\":\"EVALUATION_COMPLETE\",\"method\":\"$METHOD\",\"control_hz\":$HZ}"
