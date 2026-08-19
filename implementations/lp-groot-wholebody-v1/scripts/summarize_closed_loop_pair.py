#!/usr/bin/env python3
"""Validate and summarize a paired fixed-seed closed-loop comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


METHODS = ("b0", "b1")


def _load_method(
    root: Path,
    method: str,
    task: str,
    gym_id: str,
    expected_seeds: list[int],
) -> tuple[dict, dict[int, bool]]:
    method_root = root / method
    if not (method_root / "CLOSED_LOOP_COMPLETE").is_file():
        raise RuntimeError(f"{method}: CLOSED_LOOP_COMPLETE is missing")
    report_path = method_root / "results.json"
    with report_path.open() as file:
        report = json.load(file)
    if report.get("method") != method:
        raise ValueError(f"{method}: report method is {report.get('method')!r}")

    episodes = report.get("episodes", [])
    if len(episodes) != len(expected_seeds):
        raise ValueError(
            f"{method}: expected {len(expected_seeds)} episodes, got {len(episodes)}"
        )
    by_seed: dict[int, bool] = {}
    video_count = 0
    elapsed_seconds = 0.0
    for episode in episodes:
        if episode.get("task") != task or episode.get("gym_id") != gym_id:
            raise ValueError(f"{method}: unexpected task record {episode}")
        seed = int(episode["seed"])
        if seed in by_seed:
            raise ValueError(f"{method}: duplicate seed {seed}")
        if not isinstance(episode.get("success"), bool):
            raise TypeError(f"{method}: seed {seed} has no boolean success result")
        videos = [Path(path) for path in episode.get("videos", [])]
        if not videos:
            raise ValueError(f"{method}: seed {seed} has no recorded video")
        for video in videos:
            if not video.is_file() or video.stat().st_size == 0:
                raise ValueError(f"{method}: missing or empty video {video}")
        video_count += len(videos)
        elapsed_seconds += float(episode.get("elapsed_seconds", 0.0))
        by_seed[seed] = episode["success"]

    if sorted(by_seed) != expected_seeds:
        raise ValueError(
            f"{method}: seeds do not match fixed range {expected_seeds[0]}..{expected_seeds[-1]}"
        )
    successes = sum(by_seed.values())
    return (
        {
            "completed": len(episodes),
            "successes": successes,
            "success_rate": successes / len(episodes),
            "video_files": video_count,
            "reported_elapsed_seconds_sum": elapsed_seconds,
            "results_path": str(report_path),
        },
        by_seed,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--gym-id", required=True)
    parser.add_argument("--seed-start", type=int, required=True)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.count <= 0:
        raise ValueError("count must be positive")

    root = Path(args.root)
    expected_seeds = list(range(args.seed_start, args.seed_start + args.count))
    methods: dict[str, dict] = {}
    outcomes: dict[str, dict[int, bool]] = {}
    for method in METHODS:
        methods[method], outcomes[method] = _load_method(
            root, method, args.task, args.gym_id, expected_seeds
        )

    paired = {
        "both_success": 0,
        "b0_only": 0,
        "b1_only": 0,
        "neither_success": 0,
    }
    for seed in expected_seeds:
        b0, b1 = outcomes["b0"][seed], outcomes["b1"][seed]
        if b0 and b1:
            paired["both_success"] += 1
        elif b0:
            paired["b0_only"] += 1
        elif b1:
            paired["b1_only"] += 1
        else:
            paired["neither_success"] += 1

    result = {
        "schema_version": 1,
        "comparison": "B0_vs_B1_fixed_seed_closed_loop",
        "task": args.task,
        "gym_id": args.gym_id,
        "split": "target",
        "seed_start": args.seed_start,
        "seed_end_inclusive": expected_seeds[-1],
        "episodes_per_method": args.count,
        "expected_total_episodes": 2 * args.count,
        "completed_total_episodes": sum(item["completed"] for item in methods.values()),
        "methods": methods,
        "paired_outcomes": paired,
        "b1_minus_b0_success_rate": (
            methods["b1"]["success_rate"] - methods["b0"]["success_rate"]
        ),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as file:
        json.dump(result, file, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
