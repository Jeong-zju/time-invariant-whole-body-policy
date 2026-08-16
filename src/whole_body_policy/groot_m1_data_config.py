"""GR00T data configuration for the Phase 1 M1 action representation.

The derived LeRobot parquet stores measured base poses in world coordinates so
that one row can be shared by every action anchor.  This transform converts the
16 future poses to one anchor-relative SE(2) plan *before* normalization, then
removes the world-frame anchor so it is never exposed to the model.

This module intentionally depends on Isaac-GR00T and is imported only inside the
frozen Arena training environment.  The representation and tracker primitives
remain dependency-light in :mod:`whole_body_policy.phase1`.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from gr00t.data.dataset import ModalityConfig
from gr00t.data.transform.base import ComposedModalityTransform, ModalityTransform
from gr00t.data.transform.concat import ConcatTransform
from gr00t.data.transform.state_action import StateActionToTensor, StateActionTransform
from gr00t.data.transform.video import VideoColorJitter, VideoCrop, VideoResize, VideoToNumpy, VideoToTensor
from gr00t.experiment.data_config import BaseDataConfig
from gr00t.model.transforms import GR00TTransform

from .se2 import relative


ANCHOR_BASE_POSE_KEY = "state.phase1_anchor_base_pose_se2"
UPPER_TARGET_KEY = "action.phase1_upper_body_position"
HEIGHT_TARGET_KEY = "action.phase1_base_height"
BASE_TARGET_KEY = "action.phase1_base_relative_se2"


class AnchorRelativeSE2Transform(ModalityTransform):
    """Replace future world poses with poses relative to the current anchor."""

    anchor_key: str = ANCHOR_BASE_POSE_KEY
    target_key: str = BASE_TARGET_KEY

    def apply(self, data: dict[str, Any]) -> dict[str, Any]:
        anchor_present = self.anchor_key in data
        target_present = self.target_key in data
        # Inference contains neither loader-only state anchors nor action
        # targets.  A partial pair, however, is a corrupt training sample.
        if not anchor_present and not target_present:
            return data
        if not anchor_present or not target_present:
            missing = [key for key in (self.anchor_key, self.target_key) if key not in data]
            raise KeyError(f"Phase 1 anchor transform is missing keys: {missing}")

        anchor = np.asarray(data[self.anchor_key], dtype=np.float64)
        targets = np.asarray(data[self.target_key], dtype=np.float64)
        if anchor.shape != (1, 3):
            raise ValueError(f"{self.anchor_key} must have shape (1, 3), got {anchor.shape}")
        if targets.ndim != 2 or targets.shape[1] != 3:
            raise ValueError(f"{self.target_key} must have shape (H, 3), got {targets.shape}")

        data[self.target_key] = np.stack(
            [relative(anchor[0], target) for target in targets], axis=0
        ).astype(np.float32)
        # The world-frame value is a loader-only anchor.  It must not reach the
        # state normalizer or the model input.
        del data[self.anchor_key]
        return data


class UnitreeG1Phase1M1DataConfig(BaseDataConfig):
    """Frozen ``16 x 32`` GR00T configuration for Phase 1 M1."""

    video_keys = ["video.ego_view"]
    state_keys = [
        "state.left_arm",
        "state.right_arm",
        "state.left_hand",
        "state.right_hand",
        "state.waist",
    ]
    action_keys = [UPPER_TARGET_KEY, HEIGHT_TARGET_KEY, BASE_TARGET_KEY]
    observation_indices = [0]
    # Row h is the measured state at anchor + (h + 1) * 20 ms.  The custom
    # training entrypoint removes terminal anchors that cannot supply index 16.
    action_indices = list(range(1, 17))

    def modality_config(self) -> dict[str, ModalityConfig]:
        return {
            "video": ModalityConfig(
                delta_indices=self.observation_indices,
                modality_keys=self.video_keys,
            ),
            "state": ModalityConfig(
                delta_indices=self.observation_indices,
                modality_keys=[*self.state_keys, ANCHOR_BASE_POSE_KEY],
            ),
            "action": ModalityConfig(
                delta_indices=self.action_indices,
                modality_keys=self.action_keys,
            ),
            "language": ModalityConfig(
                delta_indices=self.observation_indices,
                modality_keys=["annotation.human.task_description"],
            ),
        }

    def transform(self) -> ModalityTransform:
        transforms: list[ModalityTransform] = [
            AnchorRelativeSE2Transform(apply_to=[ANCHOR_BASE_POSE_KEY, BASE_TARGET_KEY]),
            VideoToTensor(apply_to=self.video_keys),
            VideoCrop(apply_to=self.video_keys, scale=0.95),
            VideoResize(
                apply_to=self.video_keys,
                height=224,
                width=224,
                interpolation="linear",
            ),
            VideoColorJitter(
                apply_to=self.video_keys,
                brightness=0.3,
                contrast=0.4,
                saturation=0.5,
                hue=0.08,
            ),
            VideoToNumpy(apply_to=self.video_keys),
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(
                apply_to=self.state_keys,
                normalization_modes={key: "min_max" for key in self.state_keys},
            ),
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={key: "min_max" for key in self.action_keys},
            ),
            ConcatTransform(
                video_concat_order=self.video_keys,
                state_concat_order=self.state_keys,
                action_concat_order=self.action_keys,
            ),
            GR00TTransform(
                state_horizon=len(self.observation_indices),
                action_horizon=len(self.action_indices),
                max_state_dim=64,
                max_action_dim=32,
            ),
        ]
        return ComposedModalityTransform(transforms=transforms)
