#!/usr/bin/env python3
from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

import numpy as np

from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS
from gr00t.data.dataset.lp_base_dataset import LPBaseSingleStepDataset
from gr00t.data.embodiment_tags import EmbodimentTag


def main() -> None:
    project = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project / "configs"))
    importlib.import_module("lp_base_modality")
    os.environ.setdefault("LP_MAX_EPISODES", "2")
    os.environ.setdefault("LP_MAX_STEPS", "8")
    dataset = LPBaseSingleStepDataset(
        dataset_path=os.environ["LP_DATASET_ROOT"],
        embodiment_tag=EmbodimentTag.NEW_EMBODIMENT,
        modality_configs=MODALITY_CONFIGS[EmbodimentTag.NEW_EMBODIMENT.value],
        shard_size=8,
        episode_sampling_rate=1.0,
        seed=7,
        allow_padding=True,
    )
    frame = dataset.episode_loader[0]
    action = frame["action.lp_base_path"].iloc[0]
    valid = frame["lp_base_valid"].iloc[0]
    assert action.shape == (32, 4)
    assert valid.shape == (32,)
    assert np.all(np.isfinite(action))
    assert 1 <= valid.sum() <= 32
    for key in MODALITY_CONFIGS[EmbodimentTag.NEW_EMBODIMENT.value]["video"].modality_keys:
        assert frame[f"video.{key}"].iloc[0].shape[-2:] == (256, 256) or frame[f"video.{key}"].iloc[0].shape[:2] == (256, 256)
    stats = dataset.get_dataset_statistics()
    assert len(stats["action"]["lp_base_path"]["mean"]) == 4
    print({
        "selected_steps": int(dataset.shard_lengths.sum()),
        "episode_frames": len(frame),
        "action_shape": action.shape,
        "valid_anchors": int(valid.sum()),
        "video_shapes": {key: frame[f"video.{key}"].iloc[0].shape for key in MODALITY_CONFIGS[EmbodimentTag.NEW_EMBODIMENT.value]["video"].modality_keys},
    })


if __name__ == "__main__":
    main()
