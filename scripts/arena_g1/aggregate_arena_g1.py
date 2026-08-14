#!/usr/bin/env python3
"""Aggregate per-run Arena metrics with a Wilson confidence interval."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--suite-id", required=True)
    return parser.parse_args()


def wilson(successes: int, total: int, z: float = 1.959963984540054) -> list[float]:
    if total == 0:
        return [0.0, 0.0]
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    radius = z * math.sqrt(proportion * (1 - proportion) / total + z * z / (4 * total * total)) / denominator
    return [max(0.0, center - radius), min(1.0, center + radius)]


def main() -> None:
    args = parse_args()
    runs = [json.loads(path.read_text(encoding="utf-8")) for path in args.inputs]
    if not runs:
        raise SystemExit("No Arena result files were provided")
    expected_protocol = "arena-g1-box-pick-place-v0"
    checkpoints = {run.get("checkpoint") for run in runs}
    seeds = [int(run["seed"]) for run in runs]
    if any(run.get("protocol_id") != expected_protocol for run in runs):
        raise SystemExit("Refusing to aggregate results from another protocol")
    if len(checkpoints) != 1 or not all(isinstance(value, str) and value for value in checkpoints):
        raise SystemExit(f"Refusing to mix checkpoints: {sorted(map(str, checkpoints))}")
    if len(seeds) != len(set(seeds)):
        raise SystemExit(f"Duplicate seeds in suite: {seeds}")
    for run in runs:
        if int(run.get("num_steps", 0)) != 1200:
            raise SystemExit(f"Unexpected step budget in run {run.get('run_id')}")
        episodes = int(run["metrics"]["num_episodes"])
        success_rate = float(run["metrics"]["success_rate"])
        if episodes <= 0 or not 0.0 <= success_rate <= 1.0:
            raise SystemExit(f"Invalid metrics in run {run.get('run_id')}: {run['metrics']}")
    total_episodes = sum(int(run["metrics"]["num_episodes"]) for run in runs)
    successes = sum(
        round(float(run["metrics"]["success_rate"]) * int(run["metrics"]["num_episodes"])) for run in runs
    )
    result = {
        "schema_version": 1,
        "protocol_id": expected_protocol,
        "suite_id": args.suite_id,
        "benchmark": "NVIDIA Isaac Lab Arena G1 Loco-Manipulation",
        "task": "Box Pick-and-Place (brown box to blue bin)",
        "model_family": "GR00T N1.5",
        "num_runs": len(runs),
        "checkpoint": next(iter(checkpoints)),
        "seeds": seeds,
        "total_episodes": total_episodes,
        "successes": successes,
        "success_rate": successes / total_episodes if total_episodes else 0.0,
        "success_rate_wilson_95": wilson(successes, total_episodes),
        "runs": [str(path) for path in args.inputs],
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
