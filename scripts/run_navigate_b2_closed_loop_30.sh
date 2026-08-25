#!/usr/bin/env bash
set -euo pipefail

RUN_ROOT=${RUN_ROOT:-/workspace/act-b2-ratefree-navigate-kitchen-scratch-20260821}
B0_ROOT=/workspace/act-navigate-kitchen-scratch-20260821
CHECKPOINT="$RUN_ROOT/train-50000/checkpoint-50000-model.pt"
TASKS=/workspace/lpwb-navigate-kitchen-ratefree-20260820/data/NavigateKitchen/lerobot/meta/tasks.jsonl
CALIBRATION=/workspace/lpwb-navigate-kitchen-ratefree-20260820/audit/execution_calibration.json
OUT=${OUT:-$RUN_ROOT/evaluation/closed_loop_feedback_checkpoint50000_seed20260818_count30}
PREFLIGHT=${PREFLIGHT:-$RUN_ROOT/evaluation/preflight_feedback_seed20260817_horizon8}
PORT=${PORT:-5562}
SERVER_PID=""

mkdir -p "$OUT/logs" "$OUT/videos/NavigateKitchen" "$PREFLIGHT"
cleanup() { if [ -n "$SERVER_PID" ] && kill -0 "$SERVER_PID" 2>/dev/null; then kill "$SERVER_PID" 2>/dev/null || true; wait "$SERVER_PID" 2>/dev/null || true; fi; }
on_exit() { rc=$?; cleanup; if [ "$rc" -ne 0 ]; then touch "$OUT/EVALUATION_FAILED"; fi; }
trap on_exit EXIT

test -f "$RUN_ROOT/train-50000/TRAINING_COMPLETE"
test -f "$CHECKPOINT"
test -f "$CALIBRATION"
export PYTHONPATH="/workspace/grootn17:$RUN_ROOT/source/lp-act-v1/src:$B0_ROOT/source/lerobot/src"
export TOKENIZERS_PARALLELISM=false

CUDA_VISIBLE_DEVICES=0 /workspace/grootn16/.venv/bin/python -m robocasa_act_navigate.b2_eval_server \
  --checkpoint "$CHECKPOINT" --tasks "$TASKS" --calibration "$CALIBRATION" \
  --device cuda --host 127.0.0.1 --port "$PORT" --replan-steps 8 \
  >"$OUT/logs/server.log" 2>"$OUT/logs/server.err.log" &
SERVER_PID=$!
ready=0
for _ in $(seq 1 120); do
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then echo MODEL_SERVER_EXITED_BEFORE_READY >&2; exit 1; fi
  if /workspace/grootn16/.venv/bin/python - "$PORT" <<'PY'
import socket,sys
with socket.create_connection(("127.0.0.1",int(sys.argv[1])),timeout=1): pass
PY
  then ready=1; break; fi
  sleep 2
done
test "$ready" -eq 1
echo '{"event":"B2_FEEDBACK_SERVER_READY"}'

# Non-scoring eight-step schema/action/video preflight.
if [ ! -f "$PREFLIGHT/PREFLIGHT_PASSED" ]; then
  mkdir -p "$PREFLIGHT/videos"
  cd /workspace/grootn17
  CUDA_VISIBLE_DEVICES=1 MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=1 PYOPENGL_PLATFORM=egl PYTHONPATH=/workspace/grootn17 \
  /workspace/robocasa365-official/.venv/bin/python gr00t/eval/rollout_policy.py \
    --n-episodes 1 --n-envs 1 --policy-client-host 127.0.0.1 --policy-client-port "$PORT" \
    --max-episode-steps 8 --env-name robocasa365_panda_omron/NavigateKitchen_PandaOmron_Env \
    --n-action-steps 1 --video-dir "$PREFLIGHT/videos" --seed 20260817 --robocasa-split target \
    >"$PREFLIGHT/client.log" 2>"$PREFLIGHT/client.err.log"
  test -n "$(find "$PREFLIGHT/videos" -type f -name '*.mp4' -size +0c -print -quit)"
  touch "$PREFLIGHT/PREFLIGHT_PASSED"
fi
echo '{"event":"B2_PREFLIGHT_PASSED","counted_episodes":0}'

RESULTS="$OUT/results.jsonl"; touch "$RESULTS"
for seed in $(seq 20260818 20260847); do
  if grep -q "\"seed\":$seed," "$RESULTS"; then continue; fi
  seed_dir="$OUT/videos/NavigateKitchen/seed_$seed"; mkdir -p "$seed_dir"
  log="$OUT/logs/client-seed-$seed.log"; err="$OUT/logs/client-seed-$seed.err.log"
  cd /workspace/grootn17
  CUDA_VISIBLE_DEVICES=1 MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=1 PYOPENGL_PLATFORM=egl PYTHONPATH=/workspace/grootn17 \
  /workspace/robocasa365-official/.venv/bin/python gr00t/eval/rollout_policy.py \
    --n-episodes 1 --n-envs 1 --policy-client-host 127.0.0.1 --policy-client-port "$PORT" \
    --max-episode-steps 450 --env-name robocasa365_panda_omron/NavigateKitchen_PandaOmron_Env \
    --n-action-steps 1 --video-dir "$seed_dir" --seed "$seed" --robocasa-split target >"$log" 2>"$err"
  success="$(/workspace/grootn16/.venv/bin/python - "$log" <<'PY'
import re,sys
text=open(sys.argv[1],encoding='utf-8',errors='replace').read(); found=re.findall(r'success rate:\s*([0-9.eE+-]+)',text)
if not found: raise SystemExit('missing success rate')
print('true' if float(found[-1]) >= .5 else 'false')
PY
)"
  video="$(find "$seed_dir" -type f -name '*.mp4' -size +0c -print -quit)"; test -n "$video"
  printf '{"seed":%s,"success":%s,"video":"%s"}\n' "$seed" "$success" "$video" >>"$RESULTS"
  printf 'EPISODE_COMPLETE seed=%s success=%s\n' "$seed" "$success"
done

/workspace/grootn16/.venv/bin/python - "$RESULTS" "$OUT" "$CHECKPOINT" <<'PY'
import json,sys
from pathlib import Path
jsonl,out,checkpoint=Path(sys.argv[1]),Path(sys.argv[2]),sys.argv[3]
records=[json.loads(x) for x in jsonl.read_text().splitlines() if x.strip()]
records=sorted({int(r['seed']):r for r in records}.values(),key=lambda r:int(r['seed']))
expected=list(range(20260818,20260848)); assert [int(r['seed']) for r in records]==expected
videos=[p for p in (out/'videos').rglob('*.mp4') if p.stat().st_size>0]; assert len(videos)>=30
successes=sum(bool(r['success']) for r in records)
payload={'task':'NavigateKitchen','model':'ACT-B2 Path-RateFree Geometric0p10 WholeBody scratch','checkpoint':checkpoint,
 'num_episodes':30,'num_successes':successes,'success_rate':successes/30,'episodes':records,
 'seed_start':expected[0],'seed_end':expected[-1],'max_episode_steps':450,'robocasa_split':'target',
 'controller_feedback_steps':1,'model_replan_steps':8,'predicted_time_duration_velocity_rate':False,
 'label_uses_timestamp_or_frame_horizon':False,'path_extent_m_equivalent':0.10,'whole_body':True,
 'video_count':len(videos),'comparison_note':'B0 result used execute-prefix 8; strict matched-prefix comparison requires B0 feedback-step-1 rerun.'}
(out/'results.json').write_text(json.dumps(payload,indent=2)+'\n')
(out/'summary.json').write_text(json.dumps({k:v for k,v in payload.items() if k!='episodes'},indent=2)+'\n')
PY
touch "$OUT/EVALUATION_COMPLETE"
echo '{"event":"EVALUATION_COMPLETE"}'
