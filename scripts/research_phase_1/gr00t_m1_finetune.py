#!/usr/bin/env python3
"""Run frozen GR00T fine-tuning while dropping invalid terminal anchors.

Isaac-GR00T's stock ``LeRobotSingleDataset`` pads out-of-range absolute actions
with the last row.  Phase 1 forbids that behavior: an anchor is trainable only
when all 16 future states through 320 ms exist.  We patch only the sample-index
enumeration and execute the frozen upstream training script unchanged.
"""

from __future__ import annotations

import os
from pathlib import Path
import runpy

from gr00t.data.dataset import LeRobotSingleDataset


def _phase1_valid_steps(self: LeRobotSingleDataset) -> list[tuple[int, int]]:
    action_config = self.modality_configs.get("action")
    if action_config is None or not action_config.delta_indices:
        raise ValueError("Phase 1 requires a non-empty action delta-index list")
    maximum_future_index = max(action_config.delta_indices)
    if maximum_future_index <= 0:
        raise ValueError("Phase 1 action indices must query future states")

    episode_limit = int(os.environ.get("PHASE1_TRAIN_EPISODE_LIMIT", "0"))
    anchor_limit = int(os.environ.get("PHASE1_MAX_ANCHORS_PER_EPISODE", "0"))
    allowed_ids = set(self.trajectory_ids[:episode_limit]) if episode_limit > 0 else None

    steps: list[tuple[int, int]] = []
    for trajectory_id, trajectory_length in zip(self.trajectory_ids, self.trajectory_lengths):
        if allowed_ids is not None and trajectory_id not in allowed_ids:
            continue
        valid_count = max(0, int(trajectory_length) - maximum_future_index)
        if anchor_limit > 0:
            valid_count = min(valid_count, anchor_limit)
        steps.extend((int(trajectory_id), base_index) for base_index in range(valid_count))
    if not steps:
        raise ValueError("Phase 1 filtering removed every training anchor")
    print(
        "Phase 1 terminal-anchor filter: "
        f"{len(steps)} valid anchors, maximum_future_index={maximum_future_index}, "
        f"episode_limit={episode_limit or 'all'}, anchor_limit={anchor_limit or 'all'}",
        flush=True,
    )
    return steps


def main() -> None:
    LeRobotSingleDataset._get_all_steps = _phase1_valid_steps
    project_root = Path(__file__).resolve().parents[2]
    upstream = (
        project_root
        / "repos"
        / "IsaacLab-Arena"
        / "submodules"
        / "Isaac-GR00T"
        / "scripts"
        / "gr00t_finetune.py"
    )
    if not upstream.is_file():
        raise FileNotFoundError(f"Frozen GR00T entrypoint is missing: {upstream}")
    runpy.run_path(str(upstream), run_name="__main__")


if __name__ == "__main__":
    main()
