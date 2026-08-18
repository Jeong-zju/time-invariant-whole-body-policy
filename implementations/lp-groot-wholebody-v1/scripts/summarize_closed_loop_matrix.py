#!/usr/bin/env python3
"""Validate completeness and summarize the matched 3x3 closed-loop matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


METHODS = ("b0", "b1", "b2")
TASKS = ("NavigateKitchen", "PickPlaceCounterToStove", "DeliverStraw")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--count", type=int, default=30)
    parser.add_argument("--seed-start", type=int, default=20260818)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    root = Path(args.root)
    expected_seeds = set(range(args.seed_start, args.seed_start + args.count))
    summary = {
        "schema_version": 1,
        "seed_start": args.seed_start,
        "episodes_per_task_method": args.count,
        "expected_total_episodes": len(METHODS) * len(TASKS) * args.count,
        "methods": {},
    }
    all_video_paths: list[Path] = []
    total = 0
    for method in METHODS:
        path = root / method / "results.json"
        with path.open() as file:
            report = json.load(file)
        episodes = report["episodes"]
        method_summary = {}
        for task in TASKS:
            selected = [item for item in episodes if item["task"] == task]
            seeds = [int(item["seed"]) for item in selected]
            if len(selected) != args.count or set(seeds) != expected_seeds:
                raise RuntimeError(
                    f"{method}/{task} is incomplete: count={len(selected)}, "
                    f"seeds={sorted(set(seeds))}"
                )
            if len(seeds) != len(set(seeds)):
                raise RuntimeError(f"{method}/{task} contains duplicate seeds")
            video_paths = [Path(video) for item in selected for video in item["videos"]]
            if len(video_paths) < len(selected):
                raise RuntimeError(
                    f"{method}/{task} has only {len(video_paths)} videos for "
                    f"{len(selected)} episodes"
                )
            missing_videos = [str(video) for video in video_paths if not video.is_file()]
            empty_videos = [str(video) for video in video_paths if video.is_file() and video.stat().st_size == 0]
            if missing_videos or empty_videos:
                raise RuntimeError(
                    f"{method}/{task} video audit failed: "
                    f"missing={missing_videos[:3]}, empty={empty_videos[:3]}"
                )
            successes = sum(bool(item["success"]) for item in selected)
            method_summary[task] = {
                "completed": len(selected),
                "successes": successes,
                "success_rate": successes / len(selected),
                "videos": len(video_paths),
            }
            all_video_paths.extend(video_paths)
            total += len(selected)
        method_successes = sum(item["successes"] for item in method_summary.values())
        method_summary["aggregate"] = {
            "completed": len(TASKS) * args.count,
            "successes": method_successes,
            "success_rate": method_successes / (len(TASKS) * args.count),
        }
        summary["methods"][method] = method_summary

    if total != summary["expected_total_episodes"]:
        raise RuntimeError(f"matrix total mismatch: {total}")
    summary["completed_total_episodes"] = total
    summary["video_files"] = len(all_video_paths)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as file:
        json.dump(summary, file, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
