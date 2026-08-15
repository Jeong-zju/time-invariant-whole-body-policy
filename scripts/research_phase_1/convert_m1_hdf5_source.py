#!/usr/bin/env python3
"""Convert the frozen Phase 1 HDF5 into an exactly aligned LeRobot source.

The dataset repository's pre-converted LeRobot artifact is not frame-aligned
with the HDF5 at the same frozen revision.  Phase 1 needs root telemetry that
only exists in the HDF5, so silently joining those two releases would corrupt
the supervision.  This wrapper invokes Arena's frozen official converter and
then proves that every generated episode has the expected HDF5 trajectory and
frame count before the measured-root enrichment step is allowed to run.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import h5py


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARENA_ROOT = PROJECT_ROOT / "repos" / "IsaacLab-Arena"
sys.path.insert(0, str(ARENA_ROOT))

from isaaclab_arena_gr00t.config.dataset_config import Gr00tDatasetConfig  # noqa: E402
from isaaclab_arena_gr00t.data_utils.convert_hdf5_to_lerobot import (  # noqa: E402
    convert_hdf5_to_lerobot,
)


def _validate(hdf5_path: Path, lerobot_path: Path) -> None:
    episodes = [
        json.loads(line)
        for line in (lerobot_path / "meta" / "episodes.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    with h5py.File(hdf5_path, "r") as hdf5_file:
        trajectory_ids = list(hdf5_file["data"].keys())
        if len(episodes) != len(trajectory_ids):
            raise ValueError(
                f"converted episodes {len(episodes)} != HDF5 trajectories {len(trajectory_ids)}"
            )
        for episode, trajectory_id in zip(episodes, trajectory_ids, strict=True):
            expected = len(hdf5_file["data"][trajectory_id]["obs/robot_pos"]) - 1
            if int(episode["length"]) != expected:
                raise ValueError(
                    f"episode {episode['episode_index']} ({trajectory_id}): "
                    f"converted length {episode['length']} != expected {expected}"
                )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hdf5", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    if not args.hdf5.is_file():
        raise FileNotFoundError(args.hdf5)
    args.output_root.mkdir(parents=True, exist_ok=True)
    linked_hdf5 = args.output_root / args.hdf5.name
    if linked_hdf5.exists() and not linked_hdf5.samefile(args.hdf5):
        raise FileExistsError(f"refusing to replace unrelated input link: {linked_hdf5}")
    if not linked_hdf5.exists():
        linked_hdf5.symlink_to(args.hdf5.resolve())

    output = args.output_root / args.hdf5.stem / "lerobot"
    if output.exists():
        raise FileExistsError(f"refusing to overwrite aligned source: {output}")

    config = Gr00tDatasetConfig(
        data_root=args.output_root,
        language_instruction=(
            "Pick up the brown box from the shelf, and place it into the blue bin "
            "on the table located at the right of the shelf."
        ),
        task_index=2,
        hdf5_name=args.hdf5.name,
        state_name_sim="robot_joint_pos",
        left_eef_pos_name_sim="left_eef_pos",
        left_eef_quat_name_sim="left_eef_quat",
        right_eef_pos_name_sim="right_eef_pos",
        right_eef_quat_name_sim="right_eef_quat",
        teleop_base_height_command_name_sim="base_height_cmd",
        teleop_navigate_command_name_sim="navigate_cmd",
        teleop_torso_orientation_rpy_command_name_sim="torso_orientation_rpy_cmd",
        action_name_sim="processed_actions",
        pov_cam_name_sim="robot_head_cam_rgb",
        state_name_lerobot="observation.state",
        action_name_lerobot="action",
        video_name_lerobot="observation.images.ego_view",
        task_description_lerobot="annotation.human.task_description",
        chunks_size=1000,
        fps=50,
        robot_type="unitree_g1",
    )
    convert_hdf5_to_lerobot(config)
    _validate(args.hdf5, output)
    print(f"validated aligned Phase 1 source: {output}", flush=True)


if __name__ == "__main__":
    main()
