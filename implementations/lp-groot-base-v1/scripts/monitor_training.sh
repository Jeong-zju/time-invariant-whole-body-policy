#!/usr/bin/env bash
set -uo pipefail

PROJECT=/workspace/gr00t-n16-stackbowls-v1/lp-groot-base-v1
OUT="$PROJECT/outputs/lp_groot_base_v1_full_2gpu_20260815"
TRAIN_MARKER="$OUT/TRAINING_COMPLETE"
EVAL_MARKER="$PROJECT/outputs/lp_groot_base_v1_auto_eval_20260815/EVALUATION_COMPLETE"
last_step=-1
last_change_epoch="$(date +%s)"

while true; do
  now_epoch="$(date +%s)"
  now="$(date -Is)"
  phase=TRAIN
  [[ -f "$TRAIN_MARKER" ]] && phase=EVAL
  [[ -f "$EVAL_MARKER" ]] && phase=COMPLETE
  service_name=lp_groot_base_v1_full_2gpu_20260815
  [[ "$phase" == EVAL ]] && service_name=lp_groot_base_v1_eval_20260816
  service_status="$(supervisorctl status "$service_name" 2>&1)"
  progress="$(tail -c 20000 "$OUT.stderr.log" 2>/dev/null | tr '\r' '\n' | grep -E '[0-9]+%.*[0-9]+/10000' | tail -n 1 || true)"
  step="$(printf '%s\n' "$progress" | sed -nE 's/.*\|[[:space:]]*([0-9]+)\/10000.*/\1/p')"
  loss="$(grep "{.loss." "$OUT.stdout.log" 2>/dev/null | tail -n 1 || true)"
  checkpoint="$(find "$OUT" -maxdepth 4 -type d -name 'checkpoint-*' -printf '%f\n' 2>/dev/null | sort -V | tail -n 1)"
  gpu="$(nvidia-smi --query-gpu=index,utilization.gpu,memory.used,power.draw --format=csv,noheader 2>&1 | tr '\n' ';')"

  if [[ -n "$step" && "$step" != "$last_step" ]]; then
    last_step="$step"
    last_change_epoch="$now_epoch"
  fi
  stalled_seconds=$((now_epoch - last_change_epoch))

  printf '%s phase=%s step=%s stalled_s=%s checkpoint=%s gpu="%s" loss="%s" status="%s"\n' \
    "$now" "$phase" "${step:-unknown}" "$stalled_seconds" "${checkpoint:-none}" "$gpu" "$loss" "$service_status"

  if [[ "$phase" == COMPLETE ]]; then
    exit 0
  fi
  if [[ "$stalled_seconds" -ge 1200 && "$phase" == TRAIN ]]; then
    printf '%s ALERT training step has not advanced for %s seconds\n' "$now" "$stalled_seconds" >&2
  fi
  if [[ "$service_status" == *"EXITED"* || "$service_status" == *"FATAL"* || "$service_status" == *"no such process"* ]]; then
    printf '%s ALERT %s exited before %s completed\n' "$now" "$service_name" "$phase" >&2
    exit 2
  fi
  sleep 60
done
