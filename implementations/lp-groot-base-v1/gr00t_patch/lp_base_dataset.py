"""GR00T dataset adapter for LP-GR00T-Base V1 and compact LeRobot v3 data."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
from gr00t.data.dataset.sharded_single_step_dataset import ShardedSingleStepDataset
from gr00t.data.types import MessageType, VLAStepData
from gr00t.utils.video_utils import get_frames_by_indices


class LPBaseEpisodeLoader(LeRobotEpisodeLoader):
    """Load v3 episodes and cached 32x4 labels without integrating commands."""

    def _load_metadata(self) -> None:
        meta_dir = self.dataset_path / "meta"
        with (meta_dir / "info.json").open() as file:
            self.info_meta = json.load(file)
        if self.info_meta.get("codebase_version") != "v3.0":
            raise ValueError("LP v1 adapter currently expects compact LeRobot v3.0")
        episode_paths = sorted((meta_dir / "episodes").glob("chunk-*/*.parquet"))
        episode_frame = pd.concat([pd.read_parquet(path) for path in episode_paths], ignore_index=True)
        self._episode_frame = episode_frame
        self.episodes_metadata = episode_frame.to_dict("records")
        self.tasks_map = {}
        self.modality_meta = {
            "video": {
                "res256_image_side_0": {"original_key": "observation.images.robot0_agentview_left"},
                "res256_image_side_1": {"original_key": "observation.images.robot0_agentview_right"},
                "res256_image_wrist_0": {"original_key": "observation.images.robot0_eye_in_hand"},
            },
            "state": {
                "base_position": {"start": 0, "end": 3, "original_key": "observation.state"},
                "base_rotation": {"start": 3, "end": 7, "original_key": "observation.state"},
            },
            "action": {
                "lp_base_path": {"start": 0, "end": 4, "original_key": "lp_base_path"},
            },
        }
        with (meta_dir / "stats.json").open() as file:
            self.stats = json.load(file)
        cache = Path(os.environ["LP_LABEL_CACHE"])
        with (cache / "manifest.json").open() as file:
            self._lp_manifest = json.load(file)
        self._lp_cache = cache
        self.feature_config = self.info_meta["features"]
        self.data_path_pattern = self.info_meta["data_path"]
        self.video_path_pattern = self.info_meta["video_path"]
        self.mask_path_pattern = None
        self.chunk_size = int(self.info_meta["chunks_size"])
        self.fps = int(self.info_meta["fps"])
        data_paths = sorted((self.dataset_path / "data").glob("chunk-*/*.parquet"))
        self._v3_table = pd.concat([pd.read_parquet(path) for path in data_paths], ignore_index=True)

    def _episode_row(self, episode_index: int) -> pd.Series:
        rows = self._episode_frame[self._episode_frame["episode_index"] == episode_index]
        if len(rows) != 1:
            raise KeyError(f"Expected one metadata row for episode {episode_index}")
        return rows.iloc[0]

    def _load_parquet_data(self, episode_index: int) -> pd.DataFrame:
        row = self._episode_row(episode_index)
        start, stop = int(row["dataset_from_index"]), int(row["dataset_to_index"])
        original = self._v3_table.iloc[start:stop].reset_index(drop=True)
        state = original["observation.state"]
        labels = np.load(self._lp_cache / f"episode_{episode_index:06d}.npz")
        if len(labels["action"]) != len(original):
            raise ValueError(f"LP label length mismatch for episode {episode_index}")
        loaded = pd.DataFrame()
        loaded["state.base_position"] = state.map(lambda value: np.asarray(value)[:3])
        loaded["state.base_rotation"] = state.map(lambda value: np.asarray(value)[3:7])
        loaded["action.lp_base_path"] = list(labels["action"])
        loaded["lp_base_valid"] = list(labels["valid"])
        return loaded

    def _load_video_data(self, episode_index: int, indices: np.ndarray) -> dict[str, np.ndarray]:
        row = self._episode_row(episode_index)
        global_indices = int(row["dataset_from_index"]) + np.asarray(indices)
        result = {}
        for key in self.modality_configs["video"].modality_keys:
            original_key = self.modality_meta["video"][key]["original_key"]
            chunk = int(row[f"videos/{original_key}/chunk_index"])
            file_index = int(row[f"videos/{original_key}/file_index"])
            path = self.dataset_path / self.video_path_pattern.format(
                video_key=original_key, chunk_index=chunk, file_index=file_index
            )
            result[key] = get_frames_by_indices(
                str(path), global_indices, video_backend=self.video_backend,
                video_backend_kwargs=self.video_backend_kwargs or {},
            )
        return result

    def get_dataset_statistics(self) -> dict:
        source = self.stats["observation.state"]
        def sliced(start: int, stop: int) -> dict:
            return {name: np.asarray(values)[start:stop].tolist() for name, values in source.items()}
        return {
            "state": {"base_position": sliced(0, 3), "base_rotation": sliced(3, 7)},
            "action": {"lp_base_path": self._lp_manifest["action_stats"]},
        }


class LPBaseSingleStepDataset(ShardedSingleStepDataset):
    """Use one cached path-time chunk at the current observation and mask padding."""

    def __init__(self, *args, **kwargs):
        # Reproduce the parent initialization so its sharder uses our loader.
        from gr00t.data.interfaces import ShardedDataset
        dataset_path = kwargs["dataset_path"]
        ShardedDataset.__init__(self, dataset_path)
        self.embodiment_tag = kwargs["embodiment_tag"]
        self.modality_configs = kwargs["modality_configs"]
        self.video_backend = kwargs.get("video_backend", "torchcodec")
        self.video_backend_kwargs = kwargs.get("video_backend_kwargs")
        self.shard_size = kwargs.get("shard_size", 1024)
        self.episode_sampling_rate = kwargs.get("episode_sampling_rate", 0.1)
        self.seed = kwargs.get("seed", 42)
        self.allow_padding = True
        self.processor = None
        self.rng = np.random.default_rng(self.seed)
        self.action_horizon = 32
        self.episode_loader = LPBaseEpisodeLoader(
            dataset_path=dataset_path, modality_configs=self.modality_configs,
            video_backend=self.video_backend, video_backend_kwargs=self.video_backend_kwargs,
        )
        self.shard_dataset()

    def get_effective_episode_length(self, episode_index: int) -> int:
        return self.episode_loader.get_episode_length(episode_index)

    def shard_dataset(self):
        max_episodes = int(os.environ.get("LP_MAX_EPISODES", "-1"))
        max_steps = int(os.environ.get("LP_MAX_STEPS", "-1"))
        static_cap = float(os.environ.get("LP_STATIC_FRACTION_CAP", "0.2"))
        episode_indices = np.arange(len(self.episode_loader))
        if max_episodes > 0:
            episode_indices = episode_indices[:max_episodes]
        candidates = []
        static = []
        threshold = self.episode_loader._lp_manifest["label_config"]["path_extent"] / 32.0
        for ep in episode_indices:
            cached = np.load(self.episode_loader._lp_cache / f"episode_{ep:06d}.npz")
            moving_idx = np.flatnonzero(cached["attained_sigma"] >= threshold)
            static_idx = np.flatnonzero(cached["attained_sigma"] < threshold)
            candidates.extend((int(ep), int(i)) for i in moving_idx)
            static.extend((int(ep), int(i)) for i in static_idx)
        self.rng.shuffle(static)
        allowed_static = int(len(candidates) * static_cap / max(1.0 - static_cap, 1e-8))
        candidates.extend(static[:allowed_static])
        self.rng.shuffle(candidates)
        if max_steps > 0:
            candidates = candidates[:max_steps]
        if not candidates:
            raise ValueError("No LP training samples selected")
        # Keep episode chunks together. The v3 videos are concatenated files and
        # loading one whole episode once per shard is much cheaper than repeatedly
        # seeking hundreds of episodes for globally shuffled individual frames.
        grouped = {}
        for ep, step in candidates:
            grouped.setdefault(ep, []).append(step)
        blocks = []
        for ep, steps in grouped.items():
            steps = np.asarray(steps)
            self.rng.shuffle(steps)
            blocks.extend((ep, part) for part in np.array_split(steps, max(1, int(np.ceil(len(steps) / self.shard_size)))) if len(part))
        self.rng.shuffle(blocks)
        self.sharded_episodes = []
        current, current_size = [], 0
        for ep, steps in blocks:
            if current and current_size + len(steps) > self.shard_size:
                self.sharded_episodes.append(current)
                current, current_size = [], 0
            current.append((ep, steps))
            current_size += len(steps)
        if current:
            self.sharded_episodes.append(current)
        num_shards = len(self.sharded_episodes)
        self.shard_lengths = np.asarray([sum(len(steps) for _, steps in shard) for shard in self.sharded_episodes])
        print(f"LP selected {len(candidates)} samples in {num_shards} shards; static cap={static_cap}")

    def get_datapoint(self, episode_data: pd.DataFrame, step_index: int) -> dict:
        assert self.processor is not None
        content = VLAStepData(
            images={key: [episode_data[f"video.{key}"].iloc[step_index]] for key in self.modality_configs["video"].modality_keys},
            states={key: np.vstack([episode_data[f"state.{key}"].iloc[step_index]]).astype(np.float32) for key in self.modality_configs["state"].modality_keys},
            actions={"lp_base_path": np.asarray(episode_data["action.lp_base_path"].iloc[step_index], dtype=np.float32)},
            text=episode_data["language.task"].iloc[step_index],
            embodiment=self.embodiment_tag,
            metadata={"action_valid_mask": np.asarray(episode_data["lp_base_valid"].iloc[step_index], dtype=np.float32)},
        )
        return self.processor([{"type": MessageType.EPISODE_STEP.value, "content": content}])
