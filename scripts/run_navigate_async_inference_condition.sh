#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 4 ]; then
  echo "usage: $0 METHOD REQUEST_HZ GPU PORT" >&2
  exit 2
fi
METHOD=$1
REQUEST_HZ=$2
GPU=$3
PORT=$4
SIM_GPU=${SIM_GPU:-$GPU}
EXACT_OUTPUT=${EXACT_OUTPUT:-0}
EXACT_WARMUP_INFERENCES=${EXACT_WARMUP_INFERENCES:-10}
EXACT_PIPELINE_WORKERS=${EXACT_PIPELINE_WORKERS:-3}
EXACT_OUTPUT_DELAY_S=${EXACT_OUTPUT_DELAY_S:-0.10}
case "$METHOD" in act|geometric) ;; *) echo "METHOD must be act or geometric" >&2; exit 2;; esac
case "$REQUEST_HZ" in 10|20|50) ;; *) echo "REQUEST_HZ must be 10, 20, or 50" >&2; exit 2;; esac

CONTROL_HZ=50
RUN_ROOT=${RUN_ROOT:-/workspace/act-async-inference-sweep-navigate-kitchen-20260824-v2}
SOURCE=/workspace/act-base-only-navigate-kitchen-scratch-20260823/source/lp-act-v1
BASE_CHECKPOINT=/workspace/act-base-only-navigate-kitchen-scratch-20260823/train-50000/checkpoint-50000-model.pt
GEOMETRIC_CHECKPOINT=/workspace/act-geometric-continuousyaw-navigate-kitchen-scratch-20260823/train-50000/checkpoint-50000-model.pt
TASKS=/workspace/lpwb-navigate-kitchen-ratefree-20260820/data/NavigateKitchen/lerobot/meta/tasks.jsonl
CALIBRATION=/workspace/act-frequency-sweep-navigate-kitchen-20260823/controller-gates/50hz/base-rate-calibration-50hz.json
OUT="$RUN_ROOT/evaluation/${METHOD}_request${REQUEST_HZ}hz_control50hz_checkpoint50000_seed20260818_count30"
PREFLIGHT="$RUN_ROOT/preflight/${METHOD}_request${REQUEST_HZ}hz_control50hz_seed20260817"
TRACE="$OUT/logs/action-trace.jsonl"
MAX_EPISODE_STEPS=1125
PREFLIGHT_STEPS=250
SERVER_PID=""

mkdir -p "$OUT/logs" "$OUT/videos/NavigateKitchen" "$PREFLIGHT/videos"
if [ -f "$OUT/EVALUATION_COMPLETE" ]; then
  echo "{\"event\":\"EVALUATION_ALREADY_COMPLETE\",\"method\":\"$METHOD\",\"request_hz\":$REQUEST_HZ}"
  exit 0
fi
if [ "$METHOD" = geometric ]; then
  CHECKPOINT=$GEOMETRIC_CHECKPOINT
  test -f /workspace/act-frequency-sweep-navigate-kitchen-20260823/controller-gates/50hz/GATE_COMPLETE
  test -f "$CALIBRATION"
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
server_args=(
  -m robocasa_act_navigate.async_eval_server
  --method "$METHOD"
  --checkpoint "$CHECKPOINT"
  --tasks "$TASKS"
  --device cuda
  --host 127.0.0.1
  --port "$PORT"
  --control-hz "$CONTROL_HZ"
  --request-hz "$REQUEST_HZ"
  --source-action-hz 20
  --trace "$TRACE"
)
if [ "$METHOD" = geometric ]; then server_args+=(--calibration "$CALIBRATION"); fi
if [ "$EXACT_OUTPUT" = 1 ]; then server_args+=(--exact-output); fi
if [ "$EXACT_OUTPUT" = 1 ]; then server_args+=(--exact-warmup-inferences "$EXACT_WARMUP_INFERENCES"); fi
if [ "$EXACT_OUTPUT" = 1 ]; then server_args+=(--exact-pipeline-workers "$EXACT_PIPELINE_WORKERS"); fi
if [ "$EXACT_OUTPUT" = 1 ]; then server_args+=(--exact-output-delay-s "$EXACT_OUTPUT_DELAY_S"); fi
CUDA_VISIBLE_DEVICES="$GPU" /workspace/grootn16/.venv/bin/python "${server_args[@]}" \
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

run_client() {
  local seed=$1 steps=$2 video_dir=$3 log=$4 err=$5
  cd /workspace/grootn17
  CUDA_VISIBLE_DEVICES="$SIM_GPU" MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID="$SIM_GPU" PYOPENGL_PLATFORM=egl \
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
      >"$log" 2>"$err"
}

if [ ! -f "$PREFLIGHT/PREFLIGHT_PASSED" ]; then
  before=$(wc -l < "$TRACE")
  run_client 20260817 "$PREFLIGHT_STEPS" "$PREFLIGHT/videos" "$PREFLIGHT/client.log" "$PREFLIGHT/client.err.log"
  after=$(wc -l < "$TRACE")
  grep -q ENVIRONMENT_FREQUENCY_VERIFIED "$PREFLIGHT/client.log"
  test -n "$(find "$PREFLIGHT/videos" -type f -name '*.mp4' -size +0c -print -quit)"
  /workspace/grootn16/.venv/bin/python - "$TRACE" "$before" "$after" "$REQUEST_HZ" "$EXACT_OUTPUT" "$PREFLIGHT/preflight.json" <<'PY'
import json,sys
from pathlib import Path
trace,before,after,request,exact,out=Path(sys.argv[1]),int(sys.argv[2]),int(sys.argv[3]),int(sys.argv[4]),bool(int(sys.argv[5])),Path(sys.argv[6])
rows=[json.loads(x) for x in trace.read_text().splitlines()[before:after] if x.strip()]
actions=[x for x in rows if x.get('event')=='ACTION']
assert len(actions)>=20
assert any(x.get('request_accepted') for x in actions)
assert any(x.get('inference_completed') for x in actions)
assert not any(x.get('chunk_exhausted') for x in actions)
released=[x for x in actions if x.get('output_released')]
releases=len(released)
if exact:
    release_times=[float(x['output_release_time_s']) for x in released]
    assert releases>=20,releases
    assert all(abs((b-a)-1/request)<1e-9 for a,b in zip(release_times,release_times[1:])),release_times
    assert not any(x.get('request_dropped') for x in actions)
    assert not any(x.get('event')=='OUTPUT_DEADLINE_MISS' for x in rows)
payload={'event':'ASYNC_PREFLIGHT_PASSED','action_ticks':len(actions),'request_hz':request,
 'accepted':sum(bool(x.get('request_accepted')) for x in actions),
 'completed':sum(bool(x.get('inference_completed')) for x in actions),
 'dropped':sum(bool(x.get('request_dropped')) for x in actions),
 'exact_output':exact,'output_releases':releases}
out.write_text(json.dumps(payload,indent=2)+'\n')
print(json.dumps(payload))
PY
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
  before=$(wc -l < "$TRACE")
  run_client "$seed" "$MAX_EPISODE_STEPS" "$seed_dir" "$log" "$err"
  after=$(wc -l < "$TRACE")
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
  printf '{"seed":%s,"success":%s,"video":"%s","trace_line_start":%s,"trace_line_end":%s}\n' \
    "$seed" "$success" "$video" "$((before+1))" "$after" >>"$RESULTS"
  printf 'EPISODE_COMPLETE method=%s request_hz=%s seed=%s success=%s\n' "$METHOD" "$REQUEST_HZ" "$seed" "$success"
done

/workspace/grootn16/.venv/bin/python - "$RESULTS" "$TRACE" "$OUT" "$CHECKPOINT" "$METHOD" "$REQUEST_HZ" "$EXACT_OUTPUT" <<'PY'
import json,sys
from pathlib import Path
import numpy as np
jsonl,trace_path,out,checkpoint,method=Path(sys.argv[1]),Path(sys.argv[2]),Path(sys.argv[3]),sys.argv[4],sys.argv[5]
request_hz=int(sys.argv[6]); exact=bool(int(sys.argv[7])); control_hz=50
records=[json.loads(line) for line in jsonl.read_text().splitlines() if line.strip()]
records=sorted({int(row['seed']):row for row in records}.values(),key=lambda row:int(row['seed']))
expected_seeds=list(range(20260818,20260848))
assert [int(row['seed']) for row in records] == expected_seeds
videos=[p for p in (out/'videos').rglob('*.mp4') if p.stat().st_size>0]
assert len(videos)>=30
all_trace=[json.loads(line) for line in trace_path.read_text().splitlines() if line.strip()]
actions=[]
episode_action_counts=[]
for record in records:
    start=int(record['trace_line_start'])-1; end=int(record['trace_line_end'])
    selected=[x for x in all_trace[start:end] if x.get('event')=='ACTION']
    actions.extend(selected); episode_action_counts.append(len(selected))
assert actions
lat=np.asarray([float(x['completed_latency_s']) for x in actions if x.get('inference_completed')],dtype=float)
assert len(lat)>0 and np.isfinite(lat).all()
pipeline_lat=np.asarray([float(x['completed_pipeline_latency_s']) for x in actions if x.get('inference_completed')],dtype=float) if exact else np.asarray([],dtype=float)
accepted=sum(bool(x.get('request_accepted')) for x in actions)
completed=sum(bool(x.get('inference_completed')) for x in actions)
dropped=sum(bool(x.get('request_dropped')) for x in actions)
due=sum(bool(x.get('request_due')) for x in actions)
releases=sum(bool(x.get('output_released')) for x in actions)
if exact:
    for record in records:
        start=int(record['trace_line_start'])-1; end=int(record['trace_line_end'])
        selected=[x for x in all_trace[start:end] if x.get('event')=='ACTION' and x.get('output_released')]
        release_times=[float(x['output_release_time_s']) for x in selected]
        assert len(release_times)>=20,len(release_times)
        assert all(abs((b-a)-1/request_hz)<1e-9 for a,b in zip(release_times,release_times[1:])),release_times
    assert dropped==0
sim_seconds=len(actions)/control_hz
wall_seconds=sum(max(float(x.get('control_interval_wall_s',0.0)),1/control_hz) for x in actions[1:])
successes=sum(bool(row['success']) for row in records)
payload={
 'task':'NavigateKitchen','method':method,'checkpoint':checkpoint,
 'num_episodes':30,'num_successes':successes,'success_rate':successes/30,'episodes':records,
 'seed_start':expected_seeds[0],'seed_end':expected_seeds[-1],'source_training_hz':20,
 'controller_hz':control_hz,'controller_timestep_s':1/control_hz,
 'requested_inference_hz':request_hz,'single_flight':not exact,'drop_on_busy':not exact,'queue_depth':0 if not exact else None,
 'exact_model_output_frequency':exact,
 'exact_output_pipeline_workers':3 if exact else None,'exact_output_delay_s':0.10 if exact else None,
 'inference_busy_clock':'simulated_50hz_control_clock_with_measured_wall_latency',
 'max_episode_steps':1125,'max_episode_seconds':22.5,'robocasa_split':'target','n_action_steps':1,
 'base_only':True,'fixed_upper_body':True,
 'act_execution_semantics':'capture-time-aligned_20hz_zoh' if method=='act' else None,
 'geometric_capture_pose_anchoring':True if method=='geometric' else None,
 'geometric_predicts_time_duration_velocity_rate':False if method=='geometric' else None,
 'measured_pose_feedback_every_tick':True if method=='geometric' else False,
 'action_ticks':len(actions),'simulated_control_seconds':sim_seconds,'observed_wall_seconds_lower_bound':wall_seconds,
 'request_due_count':due,'request_accepted_count':accepted,'request_dropped_count':dropped,
 'inference_completed_count':completed,'request_drop_fraction':dropped/due if due else 0.0,
 'model_output_release_count':releases,
 'model_output_releases_per_control_second':releases/sim_seconds,
 'completed_inference_per_control_second':completed/sim_seconds,
 'completed_inference_per_observed_wall_second_upper_bound':completed/wall_seconds if wall_seconds else 0.0,
 'inference_latency_p50_s':float(np.quantile(lat,.5)),'inference_latency_p95_s':float(np.quantile(lat,.95)),
 'inference_latency_max_s':float(lat.max()),
 'pipeline_latency_p50_s':float(np.quantile(pipeline_lat,.5)) if exact else None,
 'pipeline_latency_p95_s':float(np.quantile(pipeline_lat,.95)) if exact else None,
 'pipeline_latency_max_s':float(pipeline_lat.max()) if exact else None,
 'inference_virtual_latency_ticks_p50':float(np.quantile([x['completed_virtual_latency_ticks'] for x in actions if x.get('inference_completed')],.5)),
 'inference_virtual_latency_ticks_p95':float(np.quantile([x['completed_virtual_latency_ticks'] for x in actions if x.get('inference_completed')],.95)),
 'control_deadline_miss_count':sum(bool(x.get('deadline_missed')) for x in actions),
 'control_deadline_miss_fraction':sum(bool(x.get('deadline_missed')) for x in actions)/len(actions),
 'plan_age_ticks_p50':float(np.quantile([x['plan_age_ticks'] for x in actions],.5)),
 'plan_age_ticks_p95':float(np.quantile([x['plan_age_ticks'] for x in actions],.95)),
 'chunk_exhaustion_count':sum(bool(x.get('chunk_exhausted')) for x in actions),
 'video_count':len(videos),
}
assert payload['chunk_exhaustion_count']==0
(out/'results.json').write_text(json.dumps(payload,indent=2)+'\n')
(out/'summary.json').write_text(json.dumps({k:v for k,v in payload.items() if k!='episodes'},indent=2)+'\n')
PY
touch "$OUT/EVALUATION_COMPLETE"
echo "{\"event\":\"EVALUATION_COMPLETE\",\"method\":\"$METHOD\",\"request_hz\":$REQUEST_HZ}"
