"""BEHAVIOR turning_on_radio datasets for standard ACT and LP-ACT V1."""

from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from .labels import LabelConfig, MetricConfig, build_path_label
from .parquet_data import TurningData, load_turning_data
from .stats import FeatureStats


CAMERA_KEYS = (
    "observation.rgb.left_realsense_link_camera_0",
    "observation.rgb.right_realsense_link_camera_0",
    "observation.rgb.zed_link_camera_0",
)
IMAGE_SIZE = (224, 224)
VIDEO_BACKEND = os.environ.get("LP_ACT_VIDEO_BACKEND", "torchcodec")
VIDEO_TOLERANCE_S = 1e-3
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)[:, None, None]
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)[:, None, None]
LOGGER = logging.getLogger(__name__)


class TurningACTDataset(Dataset):
    def __init__(
        self,
        data_root: str | Path,
        episodes: list[int],
        mode: str,
        state_stats: FeatureStats,
        target_stats: FeatureStats,
        metric: MetricConfig | None = None,
        video_backend: str | None = None,
    ) -> None:
        if mode not in {"standard", "lp"}:
            raise ValueError("mode must be 'standard' or 'lp'")
        if mode == "lp" and metric is None:
            raise ValueError("LP mode requires a metric config")
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        self.mode = mode
        self.state_stats = state_stats
        self.target_stats = target_stats
        self.metric = metric
        self.label_config = LabelConfig(extent_mode="fixed")
        delta_timestamps = {"action": [index / 30.0 for index in range(32)]} if mode == "standard" else None
        self.video_backend = video_backend or VIDEO_BACKEND
        self._dataset_kwargs = {
            "repo_id": "behavior-2026-partial",
            "root": Path(data_root),
            "episodes": episodes,
            "delta_timestamps": delta_timestamps,
            "tolerance_s": VIDEO_TOLERANCE_S,
        }
        self.dataset = LeRobotDataset(
            "behavior-2026-partial",
            root=Path(data_root),
            episodes=episodes,
            delta_timestamps=delta_timestamps,
            tolerance_s=VIDEO_TOLERANCE_S,
            video_backend=self.video_backend,
        )
        self._pyav_fallback = None
        self.trajectory: TurningData | None = load_turning_data(data_root) if mode == "lp" else None

    def __len__(self) -> int:
        return len(self.dataset)

    def _image(self, image: torch.Tensor) -> torch.Tensor:
        if tuple(image.shape[-2:]) != IMAGE_SIZE:
            image = F.interpolate(
                image.unsqueeze(0), size=IMAGE_SIZE, mode="bilinear", align_corners=False, antialias=True
            ).squeeze(0)
        return (image - IMAGENET_MEAN) / IMAGENET_STD

    def _load_item(self, index: int):
        try:
            return self.dataset[index]
        except Exception as error:
            if self.video_backend != "torchcodec":
                raise
            from lerobot.datasets.video_utils import FrameTimestampError

            recoverable_packet_error = isinstance(error, RuntimeError) and (
                "Could not push packet to decoder" in str(error)
                or "Invalid data found when processing input" in str(error)
            )
            if not isinstance(error, FrameTimestampError) and not recoverable_packet_error:
                raise
            if self._pyav_fallback is None:
                from lerobot.datasets.lerobot_dataset import LeRobotDataset

                self._pyav_fallback = LeRobotDataset(
                    **self._dataset_kwargs,
                    video_backend="pyav",
                )
            LOGGER.warning(
                "TorchCodec failed for dataset index %d (%s); retrying the same sample with PyAV",
                index,
                type(error).__name__,
            )
            return self._pyav_fallback[index]

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        item = self._load_item(index)
        batch = {key: self._image(item[key]) for key in CAMERA_KEYS}
        state = item["observation.state"].numpy()
        batch["observation.state"] = torch.from_numpy(self.state_stats.normalize(state).astype(np.float32))

        if self.mode == "standard":
            target = item["action"].numpy()
            is_pad = item["action_is_pad"].bool()
        else:
            assert self.trajectory is not None and self.metric is not None
            global_index = int(item["index"])
            local_index = int(np.searchsorted(self.trajectory.global_indices, global_index))
            if (
                local_index >= self.trajectory.global_indices.size
                or int(self.trajectory.global_indices[local_index]) != global_index
            ):
                raise IndexError(f"Global frame {global_index} is not in turning_on_radio trajectory data")
            label = build_path_label(self.trajectory, local_index, self.metric, self.label_config)
            target = label.target
            is_pad = torch.from_numpy(label.is_pad.copy())

        batch["action"] = torch.from_numpy(self.target_stats.normalize(target).astype(np.float32))
        batch["action_is_pad"] = is_pad
        return batch
