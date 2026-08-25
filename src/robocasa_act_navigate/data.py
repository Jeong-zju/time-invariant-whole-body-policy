"""Read and validate the native NavigateKitchen LeRobot trajectories."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .schema import ACTION_DIM, NUM_TASKS, ROBOT_STATE_DIM


def _list_to_numpy(column: pa.ChunkedArray, width: int) -> np.ndarray:
    array = column.combine_chunks()
    if not (pa.types.is_list(array.type) or pa.types.is_fixed_size_list(array.type)):
        raise TypeError(f"expected list column, got {array.type}")
    return np.asarray(array.values.to_numpy(zero_copy_only=False), dtype=np.float32).reshape(len(array), width)


@dataclass(frozen=True)
class NavigateData:
    actions: np.ndarray
    states: np.ndarray
    task_indices: np.ndarray
    timestamps: np.ndarray
    frame_indices: np.ndarray
    episode_indices: np.ndarray
    global_indices: np.ndarray
    episode_bounds: dict[int, tuple[int, int]]

    def augmented_state(self, index: int) -> np.ndarray:
        task = int(self.task_indices[index])
        one_hot = np.zeros(NUM_TASKS, dtype=np.float32)
        one_hot[task] = 1.0
        return np.concatenate((self.states[index], one_hot))


def load_data(root: str | Path) -> NavigateData:
    root = Path(root)
    paths = sorted((root / "data").glob("chunk-*/*.parquet"))
    if not paths:
        raise FileNotFoundError(f"no parquet files under {root / 'data'}")
    columns = ["action", "observation.state", "task_index", "timestamp", "frame_index", "episode_index", "index"]
    collected: dict[str, list[np.ndarray]] = {name: [] for name in columns}
    for path in paths:
        table = pq.read_table(path, columns=columns)
        collected["action"].append(_list_to_numpy(table["action"], ACTION_DIM))
        collected["observation.state"].append(_list_to_numpy(table["observation.state"], ROBOT_STATE_DIM))
        for name, dtype in (
            ("task_index", np.int64),
            ("timestamp", np.float64),
            ("frame_index", np.int64),
            ("episode_index", np.int64),
            ("index", np.int64),
        ):
            collected[name].append(table[name].combine_chunks().to_numpy(zero_copy_only=False).astype(dtype))
    values = {name: np.concatenate(parts) for name, parts in collected.items()}
    order = np.argsort(values["index"], kind="stable")
    values = {name: value[order] for name, value in values.items()}
    if not np.array_equal(values["index"], np.arange(len(order))):
        raise ValueError("global frame indices are not contiguous from zero")
    if not np.isfinite(values["action"]).all() or not np.isfinite(values["observation.state"]).all():
        raise ValueError("non-finite state/action value")
    if values["task_index"].min() < 0 or values["task_index"].max() >= NUM_TASKS:
        raise ValueError("task_index outside verified range 0..12")
    bounds: dict[int, tuple[int, int]] = {}
    for episode in sorted(np.unique(values["episode_index"]).tolist()):
        positions = np.flatnonzero(values["episode_index"] == episode)
        if positions[-1] - positions[0] + 1 != len(positions):
            raise ValueError(f"episode {episode} is not contiguous")
        start, end = int(positions[0]), int(positions[-1] + 1)
        if np.any(np.diff(values["timestamp"][start:end]) <= 0):
            raise ValueError(f"episode {episode} timestamps are not increasing")
        bounds[int(episode)] = (start, end)
    return NavigateData(
        actions=values["action"],
        states=values["observation.state"],
        task_indices=values["task_index"],
        timestamps=values["timestamp"],
        frame_indices=values["frame_index"],
        episode_indices=values["episode_index"],
        global_indices=values["index"],
        episode_bounds=bounds,
    )


def load_split(path: str | Path) -> dict[str, list[int]]:
    payload = json.loads(Path(path).read_text())
    splits = {name: [int(value) for value in payload["splits"][name]] for name in ("train", "val", "test")}
    flat = [value for values in splits.values() for value in values]
    if len(flat) != len(set(flat)) or sorted(flat) != list(range(int(payload["total_episodes"]))):
        raise ValueError("split manifest does not partition all episodes exactly once")
    return splits
