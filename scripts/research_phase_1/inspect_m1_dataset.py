#!/usr/bin/env python3
"""Validate the derived M1 dataset against saved measured target artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from gr00t.data.dataset import LeRobotSingleDataset  # noqa: E402
from gr00t.data.schema import EmbodimentTag  # noqa: E402
from scripts.research_phase_1.gr00t_m1_finetune import _phase1_valid_steps  # noqa: E402
from whole_body_policy import load_target_batch  # noqa: E402
from whole_body_policy.groot_m1_data_config import (  # noqa: E402
    ANCHOR_BASE_POSE_KEY,
    BASE_TARGET_KEY,
    HEIGHT_TARGET_KEY,
    UPPER_TARGET_KEY,
    AnchorRelativeSE2Transform,
    UnitreeG1Phase1M1DataConfig,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--skip-transformed-sample", action="store_true")
    args = parser.parse_args()

    LeRobotSingleDataset._get_all_steps = _phase1_valid_steps
    config = UnitreeG1Phase1M1DataConfig()
    dataset = LeRobotSingleDataset(
        dataset_path=args.dataset,
        modality_configs=config.modality_config(),
        transforms=config.transform(),
        embodiment_tag=EmbodimentTag.NEW_EMBODIMENT,
        video_backend="decord",
    )
    raw = dataset.get_step_data(trajectory_id=0, base_index=0)
    transformed_se2 = AnchorRelativeSE2Transform(
        apply_to=[ANCHOR_BASE_POSE_KEY, BASE_TARGET_KEY]
    )(
        {
            ANCHOR_BASE_POSE_KEY: raw[ANCHOR_BASE_POSE_KEY].copy(),
            BASE_TARGET_KEY: raw[BASE_TARGET_KEY].copy(),
        }
    )[BASE_TARGET_KEY]
    target = load_target_batch(args.target)
    expected = target.targets[0]

    errors = {
        "upper_max_abs": float(np.max(np.abs(raw[UPPER_TARGET_KEY] - expected[:, :28]))),
        "height_max_abs": float(np.max(np.abs(raw[HEIGHT_TARGET_KEY] - expected[:, 28:29]))),
        "base_se2_max_abs": float(np.max(np.abs(transformed_se2 - expected[:, 29:32]))),
    }
    if any(value > 2e-5 for value in errors.values()):
        raise ValueError(f"derived loader target mismatch: {errors}")

    transformed_shapes = None
    if not args.skip_transformed_sample:
        sample = dataset[0]
        transformed_shapes = {
            key: list(value.shape) if hasattr(value, "shape") else str(type(value))
            for key, value in sample.items()
        }
        if "action" not in sample or list(sample["action"].shape) != [16, 32]:
            raise ValueError(f"unexpected transformed action shape: {transformed_shapes}")

    report = {
        "schema_version": 1,
        "protocol_id": "arena-g1-phase-1-m1-v1",
        "dataset": str(args.dataset),
        "valid_anchor_count": len(dataset),
        "first_target_errors": errors,
        "raw_action_shapes": {
            UPPER_TARGET_KEY: list(raw[UPPER_TARGET_KEY].shape),
            HEIGHT_TARGET_KEY: list(raw[HEIGHT_TARGET_KEY].shape),
            BASE_TARGET_KEY: list(raw[BASE_TARGET_KEY].shape),
        },
        "transformed_shapes": transformed_shapes,
        "status": "pass",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
