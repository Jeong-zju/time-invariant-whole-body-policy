"""Cached-image dataset producing native 32-step ACT action chunks."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .data import NavigateData
from .schema import CAMERA_KEYS, CHUNK_SIZE, IMAGE_SIZE


def action_chunk(data: NavigateData, index: int) -> tuple[np.ndarray, np.ndarray]:
    episode = int(data.episode_indices[index])
    _, end = data.episode_bounds[episode]
    available = min(CHUNK_SIZE, end - index)
    actions = np.empty((CHUNK_SIZE, data.actions.shape[1]), dtype=np.float32)
    actions[:available] = data.actions[index : index + available]
    actions[available:] = data.actions[end - 1]
    padding = np.arange(CHUNK_SIZE) >= available
    return actions, padding


class NavigateACTDataset(Dataset):
    def __init__(self, data: NavigateData, episodes: list[int], frame_cache: str | Path) -> None:
        self.data = data
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
        actions, padding = action_chunk(self.data, index)
        batch["observation.state"] = torch.from_numpy(self.data.augmented_state(index))
        batch["action"] = torch.from_numpy(actions)
        batch["action_is_pad"] = torch.from_numpy(padding)
        return batch


class NavigateBaseOnlyACTDataset(NavigateACTDataset):
    """Native time-indexed ACT supervision restricted to chassis commands.

    The first three verified action channels are normalized ``(vx, vy, omega)``.
    Torso and upper-body channels are deliberately absent from the learning
    target and are filled by the evaluation adapter with fixed hold commands.
    """

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        batch = super().__getitem__(item)
        batch["action"] = batch["action"][..., :3]
        return batch


class NavigateGeometricACTDataset(NavigateACTDataset):
    """ACT supervision over fixed-token common-origin measured SE(2) paths."""

    def __init__(
        self,
        data: NavigateData,
        episodes: list[int],
        frame_cache: str | Path,
        path_cache: str | Path,
    ) -> None:
        super().__init__(data, episodes, frame_cache)
        self.path_cache_path = Path(path_cache)
        if not self.path_cache_path.is_file():
            raise FileNotFoundError(self.path_cache_path)
        self._path_cache: np.memmap | None = None

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        index = int(self.indices[item])
        batch = self._images(index)
        if self._path_cache is None:
            self._path_cache = np.memmap(
                self.path_cache_path,
                mode="r",
                dtype=np.float32,
                shape=(len(self.data.actions), CHUNK_SIZE, 3),
            )
        target = np.array(self._path_cache[index], copy=True)
        if not np.isfinite(target).all():
            raise ValueError(f"non-finite geometric target at global frame {index}")
        batch["observation.state"] = torch.from_numpy(self.data.augmented_state(index))
        batch["action"] = torch.from_numpy(target)
        batch["action_is_pad"] = torch.zeros(CHUNK_SIZE, dtype=torch.bool)
        return batch
