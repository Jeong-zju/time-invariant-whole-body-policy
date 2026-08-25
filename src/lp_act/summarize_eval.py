"""Aggregate official BEHAVIOR evaluator JSON files from matched rollouts."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-results", type=int, default=10)
    args = parser.parse_args()

    paths = sorted(path for directory in args.input_dir for path in (directory / "json").glob("*.json"))
    if len(paths) != args.expected_results:
        raise ValueError(f"Expected {args.expected_results} result JSONs, found {len(paths)}: {paths}")

    rows = []
    for path in paths:
        with path.open() as handle:
            result = json.load(handle)
        rows.append(
            {
                "task": result["task"],
                "instance_id": int(result["instance_id"]),
                "rollout_id": int(result["rollout_id"]),
                "steps": int(result["steps"]),
                "success": bool(result["success"]),
                "q_score": float(result["q_score"]["final"]),
                "base_distance": float(result["agent_distance"]["base"]),
                "left_eef_displacement": float(result["agent_distance"]["left"]),
                "right_eef_displacement": float(result["agent_distance"]["right"]),
                "simulator_time": float(result["time"]["simulator_time"]),
                "source_json": str(path),
            }
        )
    rows.sort(key=lambda row: row["instance_id"])

    q_scores = np.asarray([row["q_score"] for row in rows], dtype=np.float64)
    summary = {
        "num_rollouts": len(rows),
        "num_success": int(sum(row["success"] for row in rows)),
        "success_rate": float(np.mean([row["success"] for row in rows])),
        "num_positive_q": int((q_scores > 0.0).sum()),
        "mean_q_score": float(q_scores.mean()),
        "median_q_score": float(np.median(q_scores)),
        "mean_steps": float(np.mean([row["steps"] for row in rows])),
        "mean_base_distance": float(np.mean([row["base_distance"] for row in rows])),
        "mean_left_eef_displacement": float(np.mean([row["left_eef_displacement"] for row in rows])),
        "mean_right_eef_displacement": float(np.mean([row["right_eef_displacement"] for row in rows])),
        "instances": [row["instance_id"] for row in rows],
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "rollouts.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (args.output_dir / "summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
