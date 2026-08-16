#!/usr/bin/env python3
"""Build one episode of Phase 1 targets from a small, explicit NPZ contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from whole_body_policy import build_multi_horizon_targets, save_target_batch  # noqa: E402


def parse_query_times(raw: str) -> np.ndarray:
    values = np.asarray([float(item) for item in raw.split(",") if item.strip()], dtype=np.float64)
    if len(values) == 0:
        raise argparse.ArgumentTypeError("at least one comma-separated query time is required")
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument(
        "--source",
        choices=("measured_odometry", "command_integration_proxy"),
        required=True,
    )
    parser.add_argument(
        "--query-times-s",
        type=parse_query_times,
        default=parse_query_times(",".join(f"{index * 0.02:.2f}" for index in range(1, 17))),
    )
    args = parser.parse_args()

    with np.load(args.input, allow_pickle=False) as data:
        common = {
            "timestamps_s": data["timestamps_s"],
            "upper_body_position": data["upper_body_position"],
            "base_height": data["base_height"],
            "query_times_s": args.query_times_s,
            "source": args.source,
        }
        if args.source == "measured_odometry":
            common["base_pose_se2"] = data["base_pose_se2"]
        else:
            common["base_twist_body"] = data["base_twist_body"]
        batch = build_multi_horizon_targets(**common)

    save_target_batch(args.output, batch)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "schema_version": 1,
        "protocol_id": "arena-g1-phase-1-m1-v1",
        "input": str(args.input),
        "output": str(args.output),
        "source": batch.source,
        "samples": batch.samples,
        "query_times_s": batch.query_times_s.tolist(),
        "target_shape": list(batch.targets.shape),
        "terminal_anchors_dropped": True,
    }
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
