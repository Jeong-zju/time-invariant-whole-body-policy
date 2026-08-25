"""Cached-image dataset producing fixed-extent Path-RateFree ACT targets."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .b2_labels import RateFreeMetric, build_rate_free_label
from .data import NavigateData
from .schema import CAMERA_KEYS, IMAGE_SIZE


class NavigateB2Dataset(Dataset):
    def __init__(self, data: NavigateData, episodes: list[int], frame_cache: str | Path, metric: RateFreeMetric) -> None:
        self.data = data
        self.metric = metric
        self.indices = np.flatnonzero(np.isin(data.episode_indices, np.asarray(episodes, dtype=np.int64)))
        self.frame_cache_path = Path(frame_cache)
        if not self.frame_cache_path.is_file():
            raise FileNotFoundError(self.frame_cache_path)
        self._cache: np.memmap | None = None

    def __len__(self) -> int:
        return int(self.indices.size)

    def _images(self, index: int) -> dict[str, torch.Tensor]:
        if self._cache is None:
            shape = (self.data.actions.shape[0], len(CAMERA_KEYS), *IMAGE_SIZE, 3)
            self._cache = np.memmap(self.frame_cache_path, mode="r", dtype=np.uint8, shape=shape)
        raw = np.array(self._cache[index], copy=True)
        images = torch.from_numpy(raw).permute(0, 3, 1, 2).float().div_(255.0)
        return {key: images[camera] for camera, key in enumerate(CAMERA_KEYS)}

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        index = int(self.indices[item])
        batch = self._images(index)
        label = build_rate_free_label(self.data, index, self.metric)
        batch["observation.state"] = torch.from_numpy(self.data.augmented_state(index))
        batch["action"] = torch.from_numpy(label.target)
        # Every query is supervised: after the truthful terminal anchor the
        # pose/action is held and channel 11 carries STOP=-1.
        batch["action_is_pad"] = torch.zeros(self.metric.num_anchors, dtype=torch.bool)
        weight = np.ones_like(label.target, dtype=np.float32)
        valid = label.valid_count
        weight[:valid, :3] *= np.linspace(1.0, 3.0, valid, dtype=np.float32)[:, None]
        if valid < self.metric.num_anchors:
            weight[valid:, :11] *= 0.25
        weight[:, 11] = 2.0
        batch["action_loss_weight"] = torch.from_numpy(weight)
        return batch
