#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 4 ]; then
  echo "usage: $0 METHOD RATE_SCALE GPU PORT" >&2
  exit 2
fi
METHOD=$1
RATE_SCALE=$2
GPU=$3
PORT=$4
case "$METHOD" in act|point) ;; *) echo "METHOD must be act or point" >&2; exit 2;; esac
case "$RATE_SCALE" in 0.5|1.0|1.5) ;; *) echo "RATE_SCALE must be 0.5, 1.0, or 1.5" >&2; exit 2;; esac

RUN_ROOT=${RUN_ROOT:-/workspace/act-path-pointer-rate-sweep-navigate-kitchen-20260824-v1}
SOURCE=/workspace/act-base-only-navigate-kitchen-scratch-20260823/source/lp-act-v1
ACT_CHECKPOINT=/workspace/act-base-only-navigate-kitchen-scratch-20260823/train-50000/checkpoint-50000-model.pt
POINT_CHECKPOINT=/workspace/act-geometric-continuousyaw-navigate-kitchen-scratch-20260823/train-50000/checkpoint-50000-model.pt
TASKS=/workspace/lpwb-navigate-kitchen-ratefree-20260820/data/NavigateKitchen/lerobot/meta/tasks.jsonl
CALIBRATION=/workspace/act-frequency-sweep-navigate-kitchen-20260823/controller-gates/50hz/base-rate-calibration-50hz.json
RATE_TAG=${RATE_SCALE/./p}
OUT="$RUN_ROOT/evaluation/${METHOD}_rate${RATE_TAG}_seed20260818_count10"
PREFLIGHT="$RUN_ROOT/preflight/${METHOD}_rate${RATE_TAG}_seed20260817"
TRACE="$OUT/logs/action-trace.jsonl"
RESULTS="$OUT/results.jsonl"
CONTROL_HZ=50
MAX_EPISODE_STEPS=150
SERVER_PID=""

mkdir -p "$OUT/logs" "$OUT/videos" "$PREFLIGHT/videos"
if [ -f "$OUT/EVALUATION_COMPLETE" ]; then
  echo "RATE_CONDITION_ALREADY_COMPLETE method=$METHOD rate=$RATE_SCALE"
  exit 0
fi
test -f "$ACT_CHECKPOINT"
test -f "$POINT_CHECKPOINT"
test -f "$CALIBRATION"

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
if [ "$METHOD" = act ]; then
  CHECKPOINT=$ACT_CHECKPOINT
  EXTRA_ARGS=(--source-hz 20 --source-duration-s 0.4)
else
  CHECKPOINT=$POINT_CHECKPOINT
  EXTRA_ARGS=(--calibration "$CALIBRATION" --nominal-progress-rate-mps 0.25 --segment-extent-m 0.20)
fi

CUDA_VISIBLE_DEVICES="$GPU" /workspace/grootn16/.venv/bin/python \
  -m robocasa_act_navigate.rate_control_eval_server \
  --method "$METHOD" \
  --checkpoint "$CHECKPOINT" \
  --tasks "$TASKS" \
  --device cuda \
  --host 127.0.0.1 \
  --port "$PORT" \
  --rate-scale "$RATE_SCALE" \
  --control-hz "$CONTROL_HZ" \
  --trace "$TRACE" \
  "${EXTRA_ARGS[@]}" \
  >"$OUT/logs/server.log" 2>"$OUT/logs/server.err.log" &
SERVER_PID=$!

ready=0
for _ in $(seq 1 120); do
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo MODEL_SERVER_EXITED_BEFORE_READY >&2
    exit 1
  fi
  if /workspace/grootn16/.venv/bin/python -c \
    "import socket; s=socket.create_connection(('127.0.0.1',$PORT),1); s.close()" 2>/dev/null; then
    ready=1
    break
  fi
  sleep 2
done
test "$ready" -eq 1

run_trial() {
  local seed=$1
  local steps=$2
  local video_dir=$3
  local log=$4
  local err=$5
  mkdir -p "$video_dir"
  cd /workspace/grootn17
  CUDA_VISIBLE_DEVICES="$GPU" MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID="$GPU" PYOPENGL_PLATFORM=egl \
    /workspace/robocasa365-official/.venv/bin/python \
      -m robocasa_act_navigate.rollout_policy_at_hz \
      --policy-client-host 127.0.0.1 \
      --policy-client-port "$PORT" \
      --control-hz "$CONTROL_HZ" \
      --max-episode-steps "$steps" \
      --n-action-steps 1 \
      --video-dir "$video_dir" \
      --seed "$seed" \
      --robocasa-split target \
      --no-terminate-on-success \
      >"$log" 2>"$err"
}

if [ ! -f "$PREFLIGHT/PREFLIGHT_PASSED" ]; then
  run_trial 20260817 20 "$PREFLIGHT/videos" "$PREFLIGHT/client.log" "$PREFLIGHT/client.err.log"
  grep -q ENVIRONMENT_FREQUENCY_VERIFIED "$PREFLIGHT/client.log"
  test -n "$(find "$PREFLIGHT/videos" -type f -name '*.mp4' -size +0c -print -quit)"
  touch "$PREFLIGHT/PREFLIGHT_PASSED"
fi

touch "$RESULTS"
for seed in $(seq 20260818 20260827); do
  if grep -q "\"seed\":$seed," "$RESULTS"; then continue; fi
  seed_dir="$OUT/videos/seed_$seed"
  log="$OUT/logs/client-seed-$seed.log"
  err="$OUT/logs/client-seed-$seed.err.log"
  run_trial "$seed" "$MAX_EPISODE_STEPS" "$seed_dir" "$log" "$err"
  success="$(/workspace/grootn16/.venv/bin/python - "$log" <<'PY'
import re,sys
text=open(sys.argv[1],encoding='utf-8',errors='replace').read()
found=re.findall(r'success rate:\s*([0-9.eE+-]+)',text)
if not found: raise SystemExit('missing success rate')
print('true' if float(found[-1]) >= .5 else 'false')
PY
)"
  episode_id="$(/workspace/grootn16/.venv/bin/python - "$TRACE" <<'PY'
import json,sys
rows=[json.loads(x) for x in open(sys.argv[1]) if x.strip()]
print(max(int(x['episode_id']) for x in rows))
PY
)"
  action_count="$(/workspace/grootn16/.venv/bin/python - "$TRACE" "$episode_id" <<'PY'
import json,sys
print(sum(1 for x in open(sys.argv[1]) if x.strip() and (lambda r:int(r['episode_id'])==int(sys.argv[2]) and r['event']=='ACTION')(json.loads(x))))
PY
)"
  test "$action_count" -eq "$MAX_EPISODE_STEPS"
  video="$(find "$seed_dir" -type f -name '*.mp4' -size +0c -print -quit)"
  test -n "$video"
  printf '{"seed":%s,"episode_id":%s,"success":%s,"video":"%s","action_count":%s}\n' \
    "$seed" "$episode_id" "$success" "$video" "$action_count" >>"$RESULTS"
  echo "RATE_EPISODE_COMPLETE method=$METHOD rate=$RATE_SCALE seed=$seed episode_id=$episode_id"
done

/workspace/grootn16/.venv/bin/python - "$RESULTS" "$OUT" "$METHOD" "$RATE_SCALE" "$CHECKPOINT" <<'PY'
import json,sys
from pathlib import Path
results,out,method,rate,checkpoint=Path(sys.argv[1]),Path(sys.argv[2]),sys.argv[3],float(sys.argv[4]),sys.argv[5]
rows=[json.loads(x) for x in results.read_text().splitlines() if x.strip()]
rows=sorted({int(x['seed']):x for x in rows}.values(), key=lambda x:int(x['seed']))
assert [int(x['seed']) for x in rows] == list(range(20260818,20260828))
assert all(int(x['action_count']) == 150 for x in rows)
videos=[p for p in (out/'videos').rglob('*.mp4') if p.stat().st_size > 0]
assert len(videos) >= 10
summary={
  'task':'NavigateKitchen', 'method':method, 'rate_scale':rate,
  'checkpoint':checkpoint, 'control_hz':50, 'fixed_single_prediction':True,
  'num_trials':10, 'seed_start':20260818, 'seed_end':20260827,
  'max_episode_steps':150, 'max_episode_seconds':3.0,
  'terminate_on_success':False, 'task_successes_diagnostic_only':sum(bool(x['success']) for x in rows),
  'video_count':len(videos),
  'act_source_hz':20 if method=='act' else None,
  'act_source_prefix_s':0.4 if method=='act' else None,
  'point_nominal_progress_rate_mps':0.25 if method=='point' else None,
  'point_segment_extent_m':0.20 if method=='point' else None,
}
(out/'results.json').write_text(json.dumps({'summary':summary,'trials':rows},indent=2)+'\n')
(out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
PY
touch "$OUT/EVALUATION_COMPLETE"
echo "RATE_CONDITION_COMPLETE method=$METHOD rate=$RATE_SCALE"
