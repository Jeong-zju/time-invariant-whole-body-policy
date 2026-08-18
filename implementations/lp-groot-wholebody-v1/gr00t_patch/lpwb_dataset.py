"""Episode-balanced GR00T dataset adapters for matched B0/B1/B2 training."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
from gr00t.data.dataset.sharded_single_step_dataset import (
    ShardedSingleStepDataset,
    extract_step_data,
)
from gr00t.data.interfaces import ShardedDataset
from gr00t.data.types import MessageType, VLAStepData
from lpwb.labels import LabelConfig, build_path_time_label, build_pose_time_label


def _selected_episode_indices(num_episodes: int, seed: int) -> np.ndarray:
    fraction = float(os.environ.get("LPWB_TRAIN_FRACTION", "0.9"))
    split = os.environ.get("LPWB_SPLIT", "train")
    if not 0.0 < fraction < 1.0:
        raise ValueError("LPWB_TRAIN_FRACTION must lie strictly between zero and one")
    order = np.random.default_rng(seed).permutation(num_episodes)
    boundary = int(np.floor(fraction * num_episodes))
    if split == "train":
        selected = order[:boundary]
    elif split == "val":
        selected = order[boundary:]
    elif split == "all":
        selected = order
    else:
        raise ValueError("LPWB_SPLIT must be train, val, or all")
    selected = np.sort(selected)
    maximum = int(os.environ.get("LPWB_MAX_EPISODES", "-1"))
    if maximum > 0:
        selected = selected[:maximum]
    return selected


class EpisodeBalancedMixin:
    """Task mixing is handled by GR00T; this balances episodes and their frames."""

    def shard_dataset(self) -> None:
        samples_per_episode = int(os.environ.get("LPWB_SAMPLES_PER_EPISODE", "128"))
        if samples_per_episode <= 0:
            raise ValueError("LPWB_SAMPLES_PER_EPISODE must be positive")
        episode_indices = _selected_episode_indices(len(self.episode_loader), self.seed)
        episode_samples: list[tuple[int, np.ndarray]] = []
        for episode_index in episode_indices:
            length = self.get_effective_episode_length(int(episode_index))
            if length <= 0:
                continue
            steps = self.rng.choice(
                length, size=samples_per_episode, replace=length < samples_per_episode
            )
            self.rng.shuffle(steps)
            episode_samples.append((int(episode_index), np.asarray(steps, dtype=np.int64)))
        if not episode_samples:
            raise ValueError(f"no samples selected below {self.dataset_path}")

        # Keep each episode's sampled frames together. The episode loader decodes
        # its videos in one pass, so globally shuffled individual frames would
        # repeatedly decode hundreds of entire videos per shard.
        blocks: list[tuple[int, np.ndarray]] = []
        for episode_index, steps in episode_samples:
            parts = np.array_split(steps, max(1, int(np.ceil(len(steps) / self.shard_size))))
            blocks.extend((episode_index, part) for part in parts if len(part))
        self.rng.shuffle(blocks)
        self.sharded_episodes = []
        self.shard_lengths = []
        current: list[tuple[int, np.ndarray]] = []
        current_size = 0
        for episode_index, steps in blocks:
            if current and current_size + len(steps) > self.shard_size:
                self.sharded_episodes.append(current)
                self.shard_lengths.append(current_size)
                current, current_size = [], 0
            current.append((episode_index, steps))
            current_size += len(steps)
        if current:
            self.sharded_episodes.append(current)
            self.shard_lengths.append(current_size)
        self.shard_lengths = np.asarray(self.shard_lengths, dtype=np.int64)
        candidate_count = int(sum(len(steps) for _, steps in episode_samples))
        print(
            f"LPWB split={os.environ.get('LPWB_SPLIT', 'train')} selected "
            f"{len(episode_indices)} episodes, {candidate_count} samples, "
            f"{len(self.sharded_episodes)} shards "
            f"from {self.dataset_path}"
        )


class SparseVideoShardMixin:
    """Decode only observation frames selected for a shard, not entire episodes."""

    def get_shard(self, idx: int) -> list:
        datapoints = []
        for episode_index, step_indices in self.sharded_episodes[idx]:
            episode_meta = self.episode_loader.episodes_metadata[episode_index]
            episode_id = int(episode_meta["episode_index"])
            episode_data = self.episode_loader._load_parquet_data(episode_id)
            actual_length = min(len(episode_data), int(episode_meta["length"]))
            episode_data = episode_data.iloc[:actual_length].copy()
            unique_steps = np.unique(np.asarray(step_indices, dtype=np.int64))
            video_data = self.episode_loader._load_video_data(episode_id, unique_steps)
            for key, frames in video_data.items():
                column = np.empty(actual_length, dtype=object)
                column[:] = None
                for step, frame in zip(unique_steps, frames):
                    column[int(step)] = frame
                episode_data[f"video.{key}"] = column
            for step_index in step_indices:
                datapoints.append(self.get_datapoint(episode_data, int(step_index)))
        return datapoints


class B0EpisodeBalancedDataset(
    SparseVideoShardMixin, EpisodeBalancedMixin, ShardedSingleStepDataset
):
    """Native 32-command target with matched episode/frame sampling."""

    def get_effective_episode_length(self, episode_index: int) -> int:
        # Match B2's set of valid t0 indices exactly.
        return max(0, self.episode_loader.get_episode_length(episode_index) - 32)


class PathTimeEpisodeLoader(LeRobotEpisodeLoader):
    """Add timestamps needed to derive B2 labels from measured state."""

    def _load_parquet_data(self, episode_index: int) -> pd.DataFrame:
        loaded = super()._load_parquet_data(episode_index)
        chunk_idx = episode_index // self.chunk_size
        path = self.dataset_path / self.data_path_pattern.format(
            episode_chunk=chunk_idx, episode_index=episode_index
        )
        raw = pd.read_parquet(path, columns=["timestamp"])
        loaded["lpwb.timestamp"] = raw["timestamp"].to_numpy()
        return loaded


class B2PathTimeDataset(SparseVideoShardMixin, EpisodeBalancedMixin, ShardedSingleStepDataset):
    """Replace the native command chunk with measured 32-anchor Path-Time labels."""

    label_builder = staticmethod(build_path_time_label)
    stats_environment_variable = "LPWB_B2_STATS_DIR"

    def __init__(self, *args, **kwargs):
        dataset_path = kwargs["dataset_path"]
        ShardedDataset.__init__(self, dataset_path)
        self.embodiment_tag = kwargs["embodiment_tag"]
        self.modality_configs = kwargs["modality_configs"]
        self.video_backend = kwargs.get("video_backend", "torchcodec")
        self.video_backend_kwargs = kwargs.get("video_backend_kwargs")
        self.shard_size = kwargs.get("shard_size", 1024)
        self.episode_sampling_rate = kwargs.get("episode_sampling_rate", 0.1)
        self.seed = kwargs.get("seed", 42)
        self.allow_padding = False
        self.processor = None
        self.rng = np.random.default_rng(self.seed)
        self.action_horizon = 32
        self.label_config = LabelConfig(num_segments=32)
        self.episode_loader = PathTimeEpisodeLoader(
            dataset_path=dataset_path,
            modality_configs=self.modality_configs,
            video_backend=self.video_backend,
            video_backend_kwargs=self.video_backend_kwargs,
        )
        self.shard_dataset()

    def get_effective_episode_length(self, episode_index: int) -> int:
        # A B2 sample needs state[t0:t0+33], whereas B0 needs 32 commands.
        return max(0, self.episode_loader.get_episode_length(episode_index) - 32)

    def get_datapoint(self, episode_data: pd.DataFrame, step_index: int) -> dict:
        assert self.processor is not None
        state_keys = self.modality_configs["state"].modality_keys
        action_keys = self.modality_configs["action"].modality_keys
        states_by_key = {
            key: np.vstack(episode_data[f"state.{key}"].iloc[step_index : step_index + 33])
            for key in state_keys
        }
        raw_action_by_key = {
            key: np.vstack(episode_data[f"action.{key}"].iloc[step_index : step_index + 32])
            for key in action_keys
        }
        raw_state = np.column_stack(
            [
                states_by_key["base_position"],
                states_by_key["base_rotation"],
                states_by_key["end_effector_position_relative"],
                states_by_key["end_effector_rotation_relative"],
                states_by_key["gripper_qpos"],
            ]
        )
        raw_action = np.column_stack(
            [
                raw_action_by_key["base_motion"],
                raw_action_by_key["control_mode"],
                raw_action_by_key["end_effector_position"],
                raw_action_by_key["end_effector_rotation"],
                raw_action_by_key["gripper_close"],
            ]
        )
        timestamps = episode_data["lpwb.timestamp"].iloc[step_index : step_index + 33].to_numpy()
        label = self.label_builder(raw_state, raw_action, timestamps, self.label_config)
        content = VLAStepData(
            images={
                key: [episode_data[f"video.{key}"].iloc[step_index]]
                for key in self.modality_configs["video"].modality_keys
            },
            states={key: states_by_key[key][0:1].astype(np.float32) for key in state_keys},
            actions=label.action_dict(),
            text=episode_data["language.annotation.human.task_description"].iloc[step_index],
            embodiment=self.embodiment_tag,
        )
        return self.processor([{"type": MessageType.EPISODE_STEP.value, "content": content}])

    def get_dataset_statistics(self) -> dict:
        statistics = self.episode_loader.get_dataset_statistics()
        stats_root = Path(os.environ[self.stats_environment_variable])
        stats_path = stats_root / f"{Path(self.dataset_path).parent.name}.json"
        if not stats_path.exists():
            raise FileNotFoundError(
                f"{os.environ.get('LPWB_METHOD', 'pose').upper()} label stats missing: "
                f"{stats_path}; run scripts/build_label_stats.py"
            )
        with stats_path.open() as file:
            statistics["action"] = json.load(file)["action"]
        statistics.pop("relative_action", None)
        return statistics


class B1PoseTimeDataset(B2PathTimeDataset):
    """Frame-aligned measured pose anchors with the original segment durations."""

    label_builder = staticmethod(build_pose_time_label)
    stats_environment_variable = "LPWB_B1_STATS_DIR"


def selected_dataset_class():
    method = os.environ.get("LPWB_METHOD", "b0").lower()
    if method == "b0":
        return B0EpisodeBalancedDataset
    if method == "b1":
        return B1PoseTimeDataset
    if method == "b2":
        return B2PathTimeDataset
    raise ValueError("LPWB_METHOD must be b0, b1, or b2")
