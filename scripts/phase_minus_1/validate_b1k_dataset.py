#!/usr/bin/env python3
"""Validate one downloaded BEHAVIOR 2026 task chunk and plot episode 0.

The validator intentionally works on the released low-dimensional parquet files;
videos and OmniGibson are not required. Timestamps are checked per episode because
LeRobot data timestamps reset at each episode, while packed video timestamps do not.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq


EXPECTED_REVISION = "4f50b44796641a4d526a19d9aeadc8aa51e2f2c2"
EXPECTED_FPS = 30
EXPECTED_STATE_DIM = 61
EXPECTED_ACTION_DIM = 23


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--chunk", type=int, default=0)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset-revision", default=EXPECTED_REVISION)
    return parser.parse_args()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def integrate_body_twist(timestamps: np.ndarray, twist: np.ndarray) -> np.ndarray:
    """Integrate [vx, vy, wz] as a body-frame SE(2) twist."""

    poses = np.zeros((len(timestamps), 3), dtype=np.float64)
    for index in range(len(timestamps) - 1):
        dt = float(timestamps[index + 1] - timestamps[index])
        vx, vy, wz = (float(value) for value in twist[index])
        delta_yaw = wz * dt
        if abs(wz) < 1e-8:
            delta_x_body = vx * dt
            delta_y_body = vy * dt
        else:
            a = math.sin(delta_yaw) / wz
            b = (1.0 - math.cos(delta_yaw)) / wz
            delta_x_body = a * vx - b * vy
            delta_y_body = b * vx + a * vy

        yaw = poses[index, 2]
        cos_yaw, sin_yaw = math.cos(yaw), math.sin(yaw)
        poses[index + 1, 0] = poses[index, 0] + cos_yaw * delta_x_body - sin_yaw * delta_y_body
        poses[index + 1, 1] = poses[index, 1] + sin_yaw * delta_x_body + cos_yaw * delta_y_body
        poses[index + 1, 2] = math.atan2(
            math.sin(yaw + delta_yaw), math.cos(yaw + delta_yaw)
        )
    return poses


def validate_list_lengths(table, column: str, expected: int) -> None:
    lengths = pc.list_value_length(table[column]).combine_chunks().to_numpy()
    require(np.all(lengths == expected), f"{column} contains a row that is not {expected}-D")


def plot_episode(
    output_path: Path,
    timestamps: np.ndarray,
    state: np.ndarray,
    action: np.ndarray,
    poses: np.ndarray,
) -> None:
    elapsed = timestamps - timestamps[0]
    figure, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)

    axes[0, 0].plot(poses[:, 0], poses[:, 1])
    axes[0, 0].scatter(poses[0, 0], poses[0, 1], label="start", marker="o")
    axes[0, 0].scatter(poses[-1, 0], poses[-1, 1], label="end", marker="x")
    axes[0, 0].set_title("Integrated current-relative base path")
    axes[0, 0].set_xlabel("x [m]")
    axes[0, 0].set_ylabel("y [m]")
    axes[0, 0].axis("equal")
    axes[0, 0].legend()

    for index, name in enumerate(("vx", "vy", "wz")):
        axes[0, 1].plot(elapsed, state[:, index], label=f"measured {name}")
        axes[0, 1].plot(elapsed, action[:, index], alpha=0.65, label=f"command {name}")
    axes[0, 1].set_title("Base measured velocity and command")
    axes[0, 1].set_xlabel("episode time [s]")
    axes[0, 1].legend(ncol=2, fontsize=8)

    for index in range(3, 10):
        axes[1, 0].plot(elapsed, state[:, index], linewidth=0.8)
    axes[1, 0].set_title("Left arm measured qpos: state[3:10]")
    axes[1, 0].set_xlabel("episode time [s]")
    axes[1, 0].set_ylabel("joint position")

    for index in range(28, 35):
        axes[1, 1].plot(elapsed, state[:, index], linewidth=0.8)
    axes[1, 1].set_title("Right arm measured qpos: state[28:35]")
    axes[1, 1].set_xlabel("episode time [s]")
    axes[1, 1].set_ylabel("joint position")

    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    chunk_name = f"chunk-{args.chunk:03d}"
    info_path = args.data_root / "meta" / "info.json"
    episode_meta_path = args.data_root / "meta" / "episodes" / chunk_name / "file-000.parquet"
    data_dir = args.data_root / "data" / chunk_name
    args.output_dir.mkdir(parents=True, exist_ok=True)

    info = json.loads(info_path.read_text())
    require(info["codebase_version"] == "v3.0", "expected LeRobot v3.0")
    require(info["fps"] == EXPECTED_FPS, f"expected fps={EXPECTED_FPS}")
    require(info["robot_type"] == "R1Pro", "expected R1Pro")
    require(info["features"]["observation.state"]["shape"] == [EXPECTED_STATE_DIM], "bad state dim")
    require(info["features"]["action"]["shape"] == [EXPECTED_ACTION_DIM], "bad action dim")

    episode_meta = pq.read_table(episode_meta_path)
    episode_rows = episode_meta.to_pylist()
    expected_episodes = len(episode_rows)
    require(expected_episodes > 0, "episode metadata is empty")
    require({row["task_index"] for row in episode_rows} == {args.chunk}, "task/chunk mapping differs")
    require(
        [row["demo_index_within_task"] for row in episode_rows] == list(range(expected_episodes)),
        "demo indices are not contiguous",
    )

    data_paths = sorted(data_dir.glob("*.parquet"))
    require(data_paths, f"no parquet files below {data_dir}")
    tables = []
    row_count = 0
    for data_path in data_paths:
        table = pq.read_table(
            data_path,
            columns=[
                "action",
                "observation.state",
                "timestamp",
                "frame_index",
                "episode_index",
                "task_index",
            ],
        )
        validate_list_lengths(table, "action", EXPECTED_ACTION_DIM)
        validate_list_lengths(table, "observation.state", EXPECTED_STATE_DIM)
        row_count += table.num_rows
        tables.append(table)

    all_episode_indices = np.concatenate(
        [table["episode_index"].combine_chunks().to_numpy() for table in tables]
    )
    all_frame_indices = np.concatenate(
        [table["frame_index"].combine_chunks().to_numpy() for table in tables]
    )
    all_timestamps = np.concatenate(
        [table["timestamp"].combine_chunks().to_numpy() for table in tables]
    )
    all_task_indices = np.concatenate(
        [table["task_index"].combine_chunks().to_numpy() for table in tables]
    )

    expected_episode_ids = {row["episode_index"] for row in episode_rows}
    require(
        set(np.unique(all_episode_indices)) == expected_episode_ids,
        "data episode ids differ from episode metadata",
    )
    require(set(np.unique(all_task_indices)) == {args.chunk}, "data contains another task index")
    require(row_count == sum(row["length"] for row in episode_rows), "episode lengths do not sum to rows")

    nominal_dt = 1.0 / EXPECTED_FPS
    max_dt_error = 0.0
    for episode in sorted(expected_episode_ids):
        mask = all_episode_indices == episode
        frame_indices = all_frame_indices[mask]
        timestamps = all_timestamps[mask]
        require(len(timestamps) > 1, f"episode {episode} has fewer than two frames")
        require(np.array_equal(frame_indices, np.arange(len(frame_indices))), f"bad frames in episode {episode}")
        require(np.all(np.diff(timestamps) > 0.0), f"non-monotonic timestamp in episode {episode}")
        max_dt_error = max(max_dt_error, float(np.max(np.abs(np.diff(timestamps) - nominal_dt))))
    require(max_dt_error < 1e-4, f"timestamp step differs from 30 Hz by {max_dt_error}")

    selected_tables = []
    for table in tables:
        selected = table.filter(pc.equal(table["episode_index"], args.episode))
        if selected.num_rows:
            selected_tables.append(selected)
    require(selected_tables, f"episode {args.episode} not found")

    timestamps = np.concatenate(
        [table["timestamp"].combine_chunks().to_numpy() for table in selected_tables]
    ).astype(np.float64)
    state = np.vstack(
        [table["observation.state"].combine_chunks().to_pylist() for table in selected_tables]
    ).astype(np.float64)
    action = np.vstack(
        [table["action"].combine_chunks().to_pylist() for table in selected_tables]
    ).astype(np.float64)
    require(np.isfinite(state).all(), "selected episode state contains NaN/Inf")
    require(np.isfinite(action).all(), "selected episode action contains NaN/Inf")

    poses = integrate_body_twist(timestamps, state[:, :3])
    figure_path = args.output_dir / f"{chunk_name}-episode-{args.episode:06d}.png"
    plot_episode(figure_path, timestamps, state, action, poses)

    report = {
        "status": "pass",
        "dataset_revision": args.dataset_revision,
        "chunk": args.chunk,
        "task": episode_rows[0]["tasks"][0],
        "episodes": expected_episodes,
        "rows": row_count,
        "fps": info["fps"],
        "state_dim": EXPECTED_STATE_DIM,
        "action_dim": EXPECTED_ACTION_DIM,
        "max_timestamp_step_error_seconds": max_dt_error,
        "visualized_episode": args.episode,
        "visualized_episode_frames": len(timestamps),
        "visualized_episode_duration_seconds": float(timestamps[-1] - timestamps[0]),
        "integrated_final_pose_xy_yaw": poses[-1].tolist(),
        "figure": str(figure_path),
    }
    report_path = args.output_dir / f"{chunk_name}-validation.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
