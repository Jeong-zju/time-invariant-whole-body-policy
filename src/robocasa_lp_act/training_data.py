"""Fair paired datasets for RoboCasa baseline ACT and base-only LP-ACT."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .data import RoboCasaData, load_robocasa_data
from .labels import LabelConfig, MetricConfig, build_base_path_label
from .schema import CAMERA_KEYS, NONBASE_ACTION, NUM_QUERIES
from .schema import IMAGE_SIZE
from .stats import time_action_chunk


class LineUpCondimentsDataset(Dataset):
    def __init__(
        self,
        data_root: str | Path,
        episodes: list[int],
        mode: str,
        *,
        metric: MetricConfig | None = None,
        trajectory: RoboCasaData | None = None,
        video_backend: str = "pyav",
        frame_cache: str | Path | None = None,
    ) -> None:
        if mode not in {"standard", "lp"}:
            raise ValueError("mode must be standard or lp")
        if mode == "lp" and metric is None:
            raise ValueError("LP mode requires metric")
        self.mode = mode
        self.metric = metric
        self.trajectory = trajectory or load_robocasa_data(data_root)
        self.label_config = LabelConfig()
        # LeRobot v0.3.3 indexes subset episode bounds with the original episode
        # id, which fails for non-contiguous train splits. Load all local episodes
        # and apply the split through frame indices in this wrapper.
        self.indices = np.flatnonzero(np.isin(self.trajectory.episode_indices, np.asarray(episodes)))
        self.frame_cache_path = Path(frame_cache) if frame_cache is not None else None
        self._frame_cache = None
        self.dataset = None
        if self.frame_cache_path is None:
            from lerobot.datasets.lerobot_dataset import LeRobotDataset

            self.dataset = LeRobotDataset(
                "robocasa-lineup-condiments-v1.0",
                root=Path(data_root),
                episodes=None,
                delta_timestamps={"action": [index / 20.0 for index in range(NUM_QUERIES)]},
                tolerance_s=1e-3,
                video_backend=video_backend,
            )
        elif not self.frame_cache_path.is_file():
            raise FileNotFoundError(self.frame_cache_path)

    def __len__(self) -> int:
        return int(self.indices.size)

    def _cached_images(self, global_index: int) -> dict[str, torch.Tensor]:
        if self._frame_cache is None:
            shape = (self.trajectory.actions.shape[0], len(CAMERA_KEYS), *IMAGE_SIZE, 3)
            self._frame_cache = np.memmap(self.frame_cache_path, mode="r", dtype=np.uint8, shape=shape)
        raw = np.array(self._frame_cache[global_index], copy=True)
        tensor = torch.from_numpy(raw).permute(0, 3, 1, 2).to(dtype=torch.float32).div_(255.0)
        return {key: tensor[camera_index] for camera_index, key in enumerate(CAMERA_KEYS)}

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        global_index = int(self.indices[index])
        if self.frame_cache_path is None:
            item = self.dataset[global_index]
            batch = {key: item[key].to(dtype=torch.float32) for key in CAMERA_KEYS}
            state = item["observation.state"].to(dtype=torch.float32)
            time_actions = item["action"].to(dtype=torch.float32)
            is_pad = item["action_is_pad"].bool()
            item_global_index = int(item["index"])
        else:
            batch = self._cached_images(global_index)
            state = torch.from_numpy(self.trajectory.states[global_index].astype(np.float32))
            actions, padding = time_action_chunk(self.trajectory, global_index)
            time_actions = torch.from_numpy(actions.astype(np.float32))
            is_pad = torch.from_numpy(padding)
            item_global_index = int(self.trajectory.global_indices[global_index])
        batch["observation.state"] = state
        if self.mode == "standard":
            target = time_actions
        else:
            local_index = int(np.searchsorted(self.trajectory.global_indices, item_global_index))
            if local_index >= self.trajectory.global_indices.size or int(self.trajectory.global_indices[local_index]) != item_global_index:
                raise IndexError(f"Global frame {item_global_index} was not found")
            label = build_base_path_label(self.trajectory, local_index, self.metric, self.label_config)
            nonbase = time_actions[:, NONBASE_ACTION].numpy()
            target = torch.from_numpy(label.hybrid_target(nonbase))
        batch["action"] = target
        batch["action_is_pad"] = is_pad
        return batch
