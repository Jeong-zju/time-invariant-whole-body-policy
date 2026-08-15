#!/usr/bin/env python3
"""Gate Arena command-integration targets against measured replay odometry."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from whole_body_policy import compare_base_proxy_to_odometry  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--horizon-s", type=float, default=0.32)
    parser.add_argument("--max-xy-p95-m", type=float, default=0.05)
    parser.add_argument("--max-yaw-p95-rad", type=float, default=0.08)
    args = parser.parse_args()

    with np.load(args.input, allow_pickle=False) as data:
        metrics = compare_base_proxy_to_odometry(
            timestamps_s=data["timestamps_s"],
            body_twist=data["base_twist_body"],
            measured_base_pose_se2=data["base_pose_se2"],
            horizon_s=args.horizon_s,
        )
    passed = bool(
        metrics["xy_p95_m"] <= args.max_xy_p95_m
        and metrics["yaw_p95_rad"] <= args.max_yaw_p95_rad
    )
    report = {
        "schema_version": 1,
        "protocol_id": "arena-g1-phase-1-m1-v1",
        "input": str(args.input),
        "metrics": metrics,
        "thresholds": {
            "maximum_xy_p95_m": args.max_xy_p95_m,
            "maximum_yaw_p95_rad": args.max_yaw_p95_rad,
        },
        "proxy_data_gate_passed": passed,
        "decision": (
            "allow_proxy_labeled_m1_smoke"
            if passed
            else "capture_measured_root_odometry_and_rebuild_targets"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
