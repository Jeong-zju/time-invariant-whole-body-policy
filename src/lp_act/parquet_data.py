"""Fast read-only access to the downloaded `turning_on_radio` parquet shards."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .r1pro import ACTION_DIM, STATE_DIM, validate_shapes


def _list_column_to_numpy(column: pa.ChunkedArray, width: int, dtype: np.dtype) -> np.ndarray:
    array = column.combine_chunks()
    if not pa.types.is_list(array.type) and not pa.types.is_fixed_size_list(array.type):
        raise TypeError(f"Expected list column, got {array.type}")
    values = array.values.to_numpy(zero_copy_only=False)
    result = np.asarray(values, dtype=dtype).reshape(len(array), width)
    return result


@dataclass(frozen=True)
class TurningData:
    actions: np.ndarray
    states: np.ndarray
    timestamps: np.ndarray
    frame_indices: np.ndarray
    episode_indices: np.ndarray
    global_indices: np.ndarray
    episode_bounds: dict[int, tuple[int, int]]

    def validate(self) -> None:
        validate_shapes(self.actions, self.states)
        size = self.actions.shape[0]
        arrays = (
            self.states,
            self.timestamps,
            self.frame_indices,
            self.episode_indices,
            self.global_indices,
        )
        if any(array.shape[0] != size for array in arrays):
            raise ValueError("TurningData arrays do not share the same first dimension")
        if not np.isfinite(self.actions).all() or not np.isfinite(self.states).all():
            raise ValueError("Non-finite action/state value found")
        for episode, (start, end) in self.episode_bounds.items():
            if start < 0 or end <= start or end > size:
                raise ValueError(f"Invalid bounds for episode {episode}: {(start, end)}")
            if not np.all(self.episode_indices[start:end] == episode):
                raise ValueError(f"Episode {episode} is not contiguous")


def load_turning_data(root: str | Path) -> TurningData:
    """Load task chunk 000 without decoding videos."""

    root = Path(root)
    data_paths = sorted((root / "data" / "chunk-000").glob("*.parquet"))
    episode_paths = sorted((root / "meta" / "episodes" / "chunk-000").glob("*.parquet"))
    if not data_paths or not episode_paths:
        raise FileNotFoundError(f"Missing turning_on_radio parquet shards under {root}")

    actions: list[np.ndarray] = []
    states: list[np.ndarray] = []
    timestamps: list[np.ndarray] = []
    frame_indices: list[np.ndarray] = []
    episode_indices: list[np.ndarray] = []
    global_indices: list[np.ndarray] = []

    columns = ["action", "observation.state", "timestamp", "frame_index", "episode_index", "index"]
    for path in data_paths:
        table = pq.read_table(path, columns=columns)
        actions.append(_list_column_to_numpy(table["action"], ACTION_DIM, np.float32))
        states.append(_list_column_to_numpy(table["observation.state"], STATE_DIM, np.float32))
        timestamps.append(table["timestamp"].combine_chunks().to_numpy(zero_copy_only=False).astype(np.float64))
        frame_indices.append(table["frame_index"].combine_chunks().to_numpy(zero_copy_only=False).astype(np.int64))
        episode_indices.append(
            table["episode_index"].combine_chunks().to_numpy(zero_copy_only=False).astype(np.int64)
        )
        global_indices.append(table["index"].combine_chunks().to_numpy(zero_copy_only=False).astype(np.int64))

    action_array = np.concatenate(actions)
    state_array = np.concatenate(states)
    timestamp_array = np.concatenate(timestamps)
    frame_index_array = np.concatenate(frame_indices)
    episode_index_array = np.concatenate(episode_indices)
    global_index_array = np.concatenate(global_indices)

    order = np.argsort(global_index_array, kind="stable")
    action_array = action_array[order]
    state_array = state_array[order]
    timestamp_array = timestamp_array[order]
    frame_index_array = frame_index_array[order]
    episode_index_array = episode_index_array[order]
    global_index_array = global_index_array[order]

    metadata = pa.concat_tables([pq.read_table(path) for path in episode_paths])
    episode_bounds: dict[int, tuple[int, int]] = {}
    global_to_local = {int(index): local for local, index in enumerate(global_index_array)}
    for row in metadata.to_pylist():
        episode = int(row["episode_index"])
        global_start = int(row["dataset_from_index"])
        global_end = int(row["dataset_to_index"])
        if global_start not in global_to_local or global_end - 1 not in global_to_local:
            raise ValueError(f"Episode {episode} points outside downloaded chunk 000")
        episode_bounds[episode] = (global_to_local[global_start], global_to_local[global_end - 1] + 1)

    result = TurningData(
        actions=action_array,
        states=state_array,
        timestamps=timestamp_array,
        frame_indices=frame_index_array,
        episode_indices=episode_index_array,
        global_indices=global_index_array,
        episode_bounds=episode_bounds,
    )
    result.validate()
    return result


def split_episodes(episode_bounds: dict[int, tuple[int, int]]) -> tuple[list[int], list[int]]:
    episodes = sorted(episode_bounds)
    validation = [episode for episode in episodes if episode % 5 == 0]
    training = [episode for episode in episodes if episode % 5 != 0]
    return training, validation
