#!/usr/bin/env bash
set -euo pipefail

RUN_ROOT=${RUN_ROOT:-/workspace/act-variable-speed-closed-loop-navigate-kitchen-20260825-v1}
SOURCE=/workspace/act-base-only-navigate-kitchen-scratch-20260823/source/lp-act-v1
mkdir -p "$RUN_ROOT/logs"
export RUN_ROOT

run_rate_wave() {
  rate=$1
  case "$rate" in
    0.5) port0=5660 ;;
    1.0) port0=5670 ;;
    1.5) port0=5680 ;;
    *) exit 2 ;;
  esac
  methods=(act_native act_retime act_calibrated_pointer learned_point_pointer)
  pids=()
  for gpu in 0 1 2 3; do
    method=${methods[$gpu]}
    port=$((port0 + gpu))
    /bin/bash "$SOURCE/scripts/run_navigate_variable_speed_condition.sh" \
      "$method" "$rate" "$gpu" "$port" \
      >"$RUN_ROOT/logs/${method}-rate${rate}.log" \
      2>"$RUN_ROOT/logs/${method}-rate${rate}.err.log" &
    pids+=("$!")
  done
  rc=0
  for pid in "${pids[@]}"; do wait "$pid" || rc=1; done
  if [ "$rc" -ne 0 ]; then
    echo "VARIABLE_SPEED_RATE_WAVE_FAILED rate=$rate" >&2
    exit 1
  fi
  echo "VARIABLE_SPEED_RATE_WAVE_COMPLETE rate=$rate"
}

run_rate_wave 0.5
run_rate_wave 1.0
run_rate_wave 1.5

/workspace/grootn16/.venv/bin/python - "$RUN_ROOT" <<'PY'
import json,sys
from pathlib import Path
root=Path(sys.argv[1])
methods=['act_native','act_retime','act_calibrated_pointer','learned_point_pointer']
rates=[0.5,1.0,1.5]
rows=[]
for method in methods:
    for rate in rates:
        tag=str(rate).replace('.','p')
        path=root/'evaluation'/f'{method}_rate{tag}_seed20260818_count30_fixed22p5s'/'summary.json'
        value=json.loads(path.read_text())
        rows.append(value)
assert len(rows)==12 and all(x['num_trials']==30 and x['video_count']>=30 for x in rows)
(root/'summary.json').write_text(json.dumps({'protocol':{
    'task':'NavigateKitchen','methods':methods,'rate_scales':rates,'control_hz':50,
    'model_replan_seconds':0.4,'fixed_wall_clock_horizon_seconds':22.5,
    'fixed_seeds':[20260818,20260847],
},'conditions':rows},indent=2)+'\n')
PY
touch "$RUN_ROOT/EVALUATION_COMPLETE"
echo VARIABLE_SPEED_CLOSED_LOOP_PIPELINE_COMPLETE
