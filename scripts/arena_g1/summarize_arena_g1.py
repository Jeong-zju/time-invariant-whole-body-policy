#!/usr/bin/env python3
"""Extract an Arena closed-loop result into a machine-readable manifest."""

from __future__ import annotations

import argparse
import ast
import json
import re
from datetime import datetime, timezone
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--require-nonzero", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    matches = re.findall(r"Metrics:\s*(\{[^\n]+\})", args.log.read_text(encoding="utf-8"))
    if not matches:
        raise RuntimeError(f"No Metrics record found in {args.log}")
    evaluator_metrics = ast.literal_eval(matches[-1])
    rollout_matches = re.findall(r"Rollout:\s*(\{[^\n]+\})", args.log.read_text(encoding="utf-8"))
    rollout = ast.literal_eval(rollout_matches[-1]) if rollout_matches else {}
    recorded_episodes = int(evaluator_metrics.get("num_episodes", 0))
    if recorded_episodes > 1:
        raise RuntimeError(f"Expected at most one recorded episode, got {recorded_episodes}")
    metrics = dict(evaluator_metrics)
    metrics["num_episodes"] = 1
    if recorded_episodes == 0:
        metrics["success_rate"] = 0.0
    budget_exhausted = bool(rollout.get("budget_exhausted", recorded_episodes == 0))
    result = {
        "schema_version": 1,
        "protocol_id": "arena-g1-box-pick-place-v0",
        "run_id": args.run_id,
        "benchmark": "NVIDIA Isaac Lab Arena G1 Loco-Manipulation",
        "task": "Box Pick-and-Place (brown box to blue bin)",
        "model_family": "GR00T N1.5",
        "checkpoint": args.checkpoint,
        "seed": args.seed,
        "num_steps": 1200,
        "steps_executed": int(rollout.get("steps_executed", 1200)),
        "budget_exhausted": budget_exhausted,
        "metrics": metrics,
        "evaluator_metrics": evaluator_metrics,
        "source_log": str(args.log),
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    if args.require_nonzero and float(metrics.get("success_rate", 0.0)) <= 0.0:
        raise SystemExit("Closed-loop success rate is zero")


if __name__ == "__main__":
    main()
