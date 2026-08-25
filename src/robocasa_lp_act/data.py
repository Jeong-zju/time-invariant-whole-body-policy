"""Read-only trajectory access for RoboCasa LineUpCondiments LeRobot data."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from lp_act.se2 import wrap_angle

from .schema import ACTION_DIM, BASE_POSITION_STATE, BASE_QUATERNION_STATE, STATE_DIM, validate_shapes


def _list_column_to_numpy(column: pa.ChunkedArray, width: int) -> np.ndarray:
    array = column.combine_chunks()
    if not pa.types.is_list(array.type) and not pa.types.is_fixed_size_list(array.type):
        raise TypeError(f"Expected list column, got {array.type}")
    return np.asarray(array.values.to_numpy(zero_copy_only=False), dtype=np.float64).reshape(len(array), width)


def quaternion_xyzw_to_yaw(quaternion: np.ndarray) -> np.ndarray:
    quaternion = np.asarray(quaternion, dtype=np.float64)
    x, y, z, w = np.moveaxis(quaternion, -1, 0)
    return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


@dataclass(frozen=True)
class RoboCasaData:
    actions: np.ndarray
    states: np.ndarray
    timestamps: np.ndarray
    frame_indices: np.ndarray
    episode_indices: np.ndarray
    global_indices: np.ndarray
    base_world_poses: np.ndarray
    episode_bounds: dict[int, tuple[int, int]]

    def validate(self) -> None:
        validate_shapes(self.actions, self.states)
        size = self.actions.shape[0]
        for value in (
            self.states,
            self.timestamps,
            self.frame_indices,
            self.episode_indices,
            self.global_indices,
            self.base_world_poses,
        ):
            if value.shape[0] != size:
                raise ValueError("Trajectory arrays do not share the same first dimension")
        if self.base_world_poses.shape != (size, 3):
            raise ValueError(f"Expected base poses (N,3), got {self.base_world_poses.shape}")
        if not np.isfinite(self.actions).all() or not np.isfinite(self.states).all():
            raise ValueError("Non-finite action/state value found")
        for episode, (start, end) in self.episode_bounds.items():
            if start < 0 or end <= start or end > size:
                raise ValueError(f"Invalid episode bounds for {episode}: {(start, end)}")
            if not np.all(self.episode_indices[start:end] == episode):
                raise ValueError(f"Episode {episode} is not contiguous")
            dt = np.diff(self.timestamps[start:end])
            if np.any(dt <= 0.0):
                raise ValueError(f"Episode {episode} has non-positive timestamp increments")


def load_robocasa_data(root: str | Path) -> RoboCasaData:
    root = Path(root)
    paths = sorted((root / "data").glob("chunk-*/*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No parquet files found under {root / 'data'}")

    columns = ["action", "observation.state", "timestamp", "frame_index", "episode_index", "index"]
    actions: list[np.ndarray] = []
    states: list[np.ndarray] = []
    timestamps: list[np.ndarray] = []
    frame_indices: list[np.ndarray] = []
    episode_indices: list[np.ndarray] = []
    global_indices: list[np.ndarray] = []
    for path in paths:
        table = pq.read_table(path, columns=columns)
        actions.append(_list_column_to_numpy(table["action"], ACTION_DIM))
        states.append(_list_column_to_numpy(table["observation.state"], STATE_DIM))
        timestamps.append(table["timestamp"].combine_chunks().to_numpy(zero_copy_only=False).astype(np.float64))
        frame_indices.append(table["frame_index"].combine_chunks().to_numpy(zero_copy_only=False).astype(np.int64))
        episode_indices.append(table["episode_index"].combine_chunks().to_numpy(zero_copy_only=False).astype(np.int64))
        global_indices.append(table["index"].combine_chunks().to_numpy(zero_copy_only=False).astype(np.int64))

    action_array = np.concatenate(actions)
    state_array = np.concatenate(states)
    timestamp_array = np.concatenate(timestamps)
    frame_array = np.concatenate(frame_indices)
    episode_array = np.concatenate(episode_indices)
    global_array = np.concatenate(global_indices)
    order = np.argsort(global_array, kind="stable")
    action_array = action_array[order]
    state_array = state_array[order]
    timestamp_array = timestamp_array[order]
    frame_array = frame_array[order]
    episode_array = episode_array[order]
    global_array = global_array[order]

    episode_bounds: dict[int, tuple[int, int]] = {}
    for episode in sorted(np.unique(episode_array).tolist()):
        locations = np.flatnonzero(episode_array == episode)
        if locations.size == 0 or locations[-1] - locations[0] + 1 != locations.size:
            raise ValueError(f"Episode {episode} is not contiguous")
        episode_bounds[int(episode)] = (int(locations[0]), int(locations[-1] + 1))

    poses = np.empty((state_array.shape[0], 3), dtype=np.float64)
    poses[:, :2] = state_array[:, BASE_POSITION_STATE][:, :2]
    for start, end in episode_bounds.values():
        yaw = np.unwrap(quaternion_xyzw_to_yaw(state_array[start:end, BASE_QUATERNION_STATE]))
        poses[start:end, 2] = yaw
    poses[:, 2] = wrap_angle(poses[:, 2])

    result = RoboCasaData(
        actions=action_array,
        states=state_array,
        timestamps=timestamp_array,
        frame_indices=frame_array,
        episode_indices=episode_array,
        global_indices=global_array,
        base_world_poses=poses,
        episode_bounds=episode_bounds,
    )
    result.validate()
    return result


def split_episodes(episode_bounds: dict[int, tuple[int, int]]) -> tuple[list[int], list[int]]:
    """Deterministic 80/20 split: every fifth episode is validation."""

    episodes = sorted(episode_bounds)
    validation = [episode for episode in episodes if episode % 5 == 0]
    training = [episode for episode in episodes if episode % 5 != 0]
    return training, validation
