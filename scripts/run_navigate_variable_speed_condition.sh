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
case "$METHOD" in
  act_native|act_retime|act_calibrated_pointer|learned_point_pointer) ;;
  *) echo "invalid METHOD: $METHOD" >&2; exit 2 ;;
esac
case "$RATE_SCALE" in 0.5|1.0|1.5) ;; *) echo "invalid RATE_SCALE" >&2; exit 2 ;; esac

RUN_ROOT=${RUN_ROOT:-/workspace/act-variable-speed-closed-loop-navigate-kitchen-20260825-v1}
SOURCE=/workspace/act-base-only-navigate-kitchen-scratch-20260823/source/lp-act-v1
ACT_CHECKPOINT=/workspace/act-base-only-navigate-kitchen-scratch-20260823/train-50000/checkpoint-50000-model.pt
POINT_CHECKPOINT=/workspace/act-geometric-continuousyaw-navigate-kitchen-scratch-20260823/train-50000/checkpoint-50000-model.pt
TASKS=/workspace/lpwb-navigate-kitchen-ratefree-20260820/data/NavigateKitchen/lerobot/meta/tasks.jsonl
CALIBRATION=/workspace/act-frequency-sweep-navigate-kitchen-20260823/controller-gates/50hz/base-rate-calibration-50hz.json
RATE_TAG=${RATE_SCALE/./p}
OUT="$RUN_ROOT/evaluation/${METHOD}_rate${RATE_TAG}_seed20260818_count30_fixed22p5s"
PREFLIGHT="$RUN_ROOT/preflight/${METHOD}_rate${RATE_TAG}_seed20260817"
TRACE="$OUT/logs/action-trace.jsonl"
RESULTS="$OUT/results.jsonl"
CONTROL_HZ=50
MAX_EPISODE_STEPS=1125
REPLAN_SECONDS=0.4
SERVER_PID=""

mkdir -p "$OUT/logs" "$OUT/videos" "$PREFLIGHT/videos"
if [ -f "$OUT/EVALUATION_COMPLETE" ]; then
  echo "VARIABLE_SPEED_CONDITION_ALREADY_COMPLETE method=$METHOD rate=$RATE_SCALE"
  exit 0
fi
rm -f "$OUT/EVALUATION_FAILED"
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
if [ "$METHOD" = learned_point_pointer ]; then
  CHECKPOINT=$POINT_CHECKPOINT
else
  CHECKPOINT=$ACT_CHECKPOINT
fi
EXTRA_ARGS=()
case "$METHOD" in
  act_calibrated_pointer|learned_point_pointer)
    EXTRA_ARGS+=(--calibration "$CALIBRATION")
    ;;
esac

CUDA_VISIBLE_DEVICES="$GPU" /workspace/grootn16/.venv/bin/python \
  -m robocasa_act_navigate.variable_speed_eval_server \
  --method "$METHOD" \
  --checkpoint "$CHECKPOINT" \
  --tasks "$TASKS" \
  --device cuda \
  --host 127.0.0.1 \
  --port "$PORT" \
  --rate-scale "$RATE_SCALE" \
  --control-hz "$CONTROL_HZ" \
  --source-hz 20 \
  --replan-seconds "$REPLAN_SECONDS" \
  --point-nominal-duration-s 1.6 \
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
  local terminate=$3
  local video_dir=$4
  local log=$5
  local err=$6
  local terminate_arg=--terminate-on-success
  if [ "$terminate" = false ]; then terminate_arg=--no-terminate-on-success; fi
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
      "$terminate_arg" \
      >"$log" 2>"$err"
}

if [ ! -f "$PREFLIGHT/PREFLIGHT_PASSED" ]; then
  before=$(wc -l <"$TRACE")
  run_trial 20260817 20 false "$PREFLIGHT/videos" "$PREFLIGHT/client.log" "$PREFLIGHT/client.err.log"
  grep -q ENVIRONMENT_FREQUENCY_VERIFIED "$PREFLIGHT/client.log"
  test -n "$(find "$PREFLIGHT/videos" -type f -name '*.mp4' -size +0c -print -quit)"
  after=$(wc -l <"$TRACE")
  test "$after" -gt "$before"
  touch "$PREFLIGHT/PREFLIGHT_PASSED"
fi

touch "$RESULTS"
for seed in $(seq 20260818 20260847); do
  if grep -q "\"seed\":$seed," "$RESULTS"; then continue; fi
  seed_dir="$OUT/videos/seed_$seed"
  log="$OUT/logs/client-seed-$seed.log"
  err="$OUT/logs/client-seed-$seed.err.log"
  before=$(wc -l <"$TRACE")
  run_trial "$seed" "$MAX_EPISODE_STEPS" true "$seed_dir" "$log" "$err"
  after=$(wc -l <"$TRACE")
  action_count=$(/workspace/grootn16/.venv/bin/python - "$TRACE" "$before" <<'PY'
import json,sys
path,start=sys.argv[1],int(sys.argv[2])
rows=open(path,encoding='utf-8').read().splitlines()[start:]
print(sum(json.loads(row).get('event') == 'ACTION' for row in rows if row.strip()))
PY
)
  test "$action_count" -gt 0
  success=$(/workspace/grootn16/.venv/bin/python - "$log" <<'PY'
import re,sys
text=open(sys.argv[1],encoding='utf-8',errors='replace').read()
found=re.findall(r'success rate:\s*([0-9.eE+-]+)',text)
if not found: raise SystemExit('missing success rate')
print('true' if float(found[-1]) >= .5 else 'false')
PY
)
  video=$(find "$seed_dir" -type f -name '*.mp4' -size +0c -print -quit)
  test -n "$video"
  printf '{"seed":%s,"success":%s,"video":"%s","action_count":%s,"trace_start_line":%s,"trace_end_line":%s}\n' \
    "$seed" "$success" "$video" "$action_count" "$((before + 1))" "$after" >>"$RESULTS"
  echo "VARIABLE_SPEED_EPISODE_COMPLETE method=$METHOD rate=$RATE_SCALE seed=$seed success=$success"
done

/workspace/grootn16/.venv/bin/python - "$RESULTS" "$OUT" "$METHOD" "$RATE_SCALE" "$CHECKPOINT" <<'PY'
import json,sys
from pathlib import Path
results,out,method,rate,checkpoint=Path(sys.argv[1]),Path(sys.argv[2]),sys.argv[3],float(sys.argv[4]),sys.argv[5]
rows=[json.loads(x) for x in results.read_text().splitlines() if x.strip()]
rows=sorted({int(x['seed']):x for x in rows}.values(), key=lambda x:int(x['seed']))
assert [int(x['seed']) for x in rows] == list(range(20260818,20260848))
videos=[p for p in (out/'videos').rglob('*.mp4') if p.stat().st_size > 0]
assert len(videos) >= 30
successes=sum(bool(x['success']) for x in rows)
summary={
  'task':'NavigateKitchen','method':method,'rate_scale':rate,'checkpoint':checkpoint,
  'control_hz':50,'model_replan_seconds':0.4,'model_replan_hz':2.5,
  'num_trials':30,'seed_start':20260818,'seed_end':20260847,
  'max_episode_steps':1125,'max_episode_seconds':22.5,'terminate_on_success':True,
  'successes':successes,'success_rate':successes/30.0,'video_count':len(videos),
  'rate_changes_progress_inside_each_chunk':True,
  'observation_and_replanning_schedule_matched_across_rates':True,
  'fixed_wall_clock_horizon':True,
}
(out/'results.json').write_text(json.dumps({'summary':summary,'trials':rows},indent=2)+'\n')
(out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
PY
touch "$OUT/EVALUATION_COMPLETE"
echo "VARIABLE_SPEED_CONDITION_COMPLETE method=$METHOD rate=$RATE_SCALE"
