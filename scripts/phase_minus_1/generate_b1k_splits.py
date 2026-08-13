#!/usr/bin/env python3
"""Generate deterministic episode-level train/validation splits for B1K chunks."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import pyarrow.parquet as pq


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tasks", type=int, nargs="+", default=[0, 17, 57])
    parser.add_argument("--seed", type=int, default=20260813)
    parser.add_argument("--validation-episodes", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = {
        "strategy": "episode_level_random_split",
        "seed": args.seed,
        "validation_episodes_per_task": args.validation_episodes,
        "tasks": {},
    }

    for task_id in args.tasks:
        chunk = f"chunk-{task_id:03d}"
        metadata_path = args.data_root / "meta" / "episodes" / chunk / "file-000.parquet"
        rows = pq.read_table(
            metadata_path,
            columns=["episode_index", "task_index", "demo_index_within_task"],
        ).to_pylist()
        if not rows:
            raise ValueError(f"no episodes in {metadata_path}")
        if {row["task_index"] for row in rows} != {task_id}:
            raise ValueError(f"task index mismatch in {metadata_path}")

        episode_ids = sorted(int(row["episode_index"]) for row in rows)
        task_rng = random.Random(args.seed + task_id)
        task_rng.shuffle(episode_ids)
        validation = sorted(episode_ids[: args.validation_episodes])
        train = sorted(episode_ids[args.validation_episodes :])
        if set(train) & set(validation):
            raise AssertionError(f"split overlap for task {task_id}")
        if sorted(train + validation) != sorted(int(row["episode_index"]) for row in rows):
            raise AssertionError(f"split does not cover task {task_id}")

        result["tasks"][str(task_id)] = {
            "chunk": chunk,
            "train_episode_indices": train,
            "validation_episode_indices": validation,
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "status": "pass",
        "output": str(args.output),
        "tasks": {
            task_id: {
                "train": len(split["train_episode_indices"]),
                "validation": len(split["validation_episode_indices"]),
            }
            for task_id, split in result["tasks"].items()
        },
    }, indent=2))


if __name__ == "__main__":
    main()
