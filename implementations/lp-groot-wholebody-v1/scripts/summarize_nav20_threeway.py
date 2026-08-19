#!/usr/bin/env python3
"""Validate and summarize the fixed-seed B0/B1/B2 NavigateKitchen comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


METHODS = ("b0", "b1", "b2")
TASK = "NavigateKitchen"
GYM_ID = "robocasa/NavigateKitchen"


def load_method(
    method: str, method_root: Path, expected_seeds: list[int]
) -> tuple[dict, dict[int, bool]]:
    if not (method_root / "CLOSED_LOOP_COMPLETE").is_file():
        raise RuntimeError(f"{method}: CLOSED_LOOP_COMPLETE is missing")
    report_path = method_root / "results.json"
    with report_path.open() as file:
        report = json.load(file)
    if report.get("method") != method:
        raise ValueError(f"{method}: report method is {report.get('method')!r}")
    if report.get("split") != "target":
        raise ValueError(f"{method}: expected target split")
    if report.get("n_action_steps") != 8:
        raise ValueError(f"{method}: expected n_action_steps=8")

    episodes = report.get("episodes", [])
    if len(episodes) != len(expected_seeds):
        raise ValueError(
            f"{method}: expected {len(expected_seeds)} episodes, got {len(episodes)}"
        )
    outcomes: dict[int, bool] = {}
    video_count = 0
    elapsed_seconds = 0.0
    for episode in episodes:
        if episode.get("task") != TASK or episode.get("gym_id") != GYM_ID:
            raise ValueError(f"{method}: unexpected task record {episode}")
        seed = int(episode["seed"])
        if seed in outcomes:
            raise ValueError(f"{method}: duplicate seed {seed}")
        success = episode.get("success")
        if not isinstance(success, bool):
            raise TypeError(f"{method}: seed {seed} has no boolean success result")
        videos = [Path(path) for path in episode.get("videos", [])]
        if not videos:
            raise ValueError(f"{method}: seed {seed} has no video")
        for video in videos:
            if not video.is_file() or video.stat().st_size == 0:
                raise ValueError(f"{method}: missing or empty video {video}")
        video_count += len(videos)
        elapsed_seconds += float(episode.get("elapsed_seconds", 0.0))
        outcomes[seed] = success
    if sorted(outcomes) != expected_seeds:
        raise ValueError(f"{method}: fixed seeds do not match")

    successes = sum(outcomes.values())
    return (
        {
            "completed": len(episodes),
            "successes": successes,
            "success_rate": successes / len(episodes),
            "video_files": video_count,
            "reported_elapsed_seconds_sum": elapsed_seconds,
            "results_path": str(report_path),
        },
        outcomes,
    )


def paired_counts(
    left: dict[int, bool], right: dict[int, bool], left_name: str, right_name: str
) -> dict[str, int]:
    counts = {
        "both_success": 0,
        f"{left_name}_only": 0,
        f"{right_name}_only": 0,
        "neither_success": 0,
    }
    for seed in sorted(left):
        if left[seed] and right[seed]:
            counts["both_success"] += 1
        elif left[seed]:
            counts[f"{left_name}_only"] += 1
        elif right[seed]:
            counts[f"{right_name}_only"] += 1
        else:
            counts["neither_success"] += 1
    return counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair-root", required=True)
    parser.add_argument("--b2-root", required=True)
    parser.add_argument("--seed-start", type=int, required=True)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--checkpoint-provenance", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    expected_seeds = list(range(args.seed_start, args.seed_start + args.count))
    pair_root, b2_root = Path(args.pair_root), Path(args.b2_root)
    roots = {"b0": pair_root / "b0", "b1": pair_root / "b1", "b2": b2_root}

    summaries: dict[str, dict] = {}
    outcomes: dict[str, dict[int, bool]] = {}
    for method in METHODS:
        summaries[method], outcomes[method] = load_method(
            method, roots[method], expected_seeds
        )
    with Path(args.checkpoint_provenance).open() as file:
        checkpoint = json.load(file)

    result = {
        "schema_version": 1,
        "comparison": "B0_vs_B1_vs_B2_fixed_seed_closed_loop",
        "task": TASK,
        "gym_id": GYM_ID,
        "split": "target",
        "seed_start": args.seed_start,
        "seed_end_inclusive": expected_seeds[-1],
        "episodes_per_method": args.count,
        "expected_total_episodes": len(METHODS) * args.count,
        "completed_total_episodes": sum(item["completed"] for item in summaries.values()),
        "methods": summaries,
        "b2_checkpoint": checkpoint,
        "paired_outcomes": {
            "b0_vs_b1": paired_counts(outcomes["b0"], outcomes["b1"], "b0", "b1"),
            "b0_vs_b2": paired_counts(outcomes["b0"], outcomes["b2"], "b0", "b2"),
            "b1_vs_b2": paired_counts(outcomes["b1"], outcomes["b2"], "b1", "b2"),
        },
        "success_rate_deltas": {
            "b2_minus_b0": summaries["b2"]["success_rate"] - summaries["b0"]["success_rate"],
            "b2_minus_b1": summaries["b2"]["success_rate"] - summaries["b1"]["success_rate"],
        },
        "per_seed": [
            {"seed": seed, **{method: outcomes[method][seed] for method in METHODS}}
            for seed in expected_seeds
        ],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as file:
        json.dump(result, file, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
