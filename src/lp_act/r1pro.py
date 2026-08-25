"""Authoritative R1Pro slices used by the BEHAVIOR 2026 demonstrations."""

from __future__ import annotations

import numpy as np

FPS = 30
ACTION_DIM = 23
STATE_DIM = 61
LP_ACTION_DIM = 24

BASE_ACTION = slice(0, 3)
TRUNK_ACTION = slice(3, 7)
LEFT_ARM_ACTION = slice(7, 14)
LEFT_GRIPPER_ACTION = 14
RIGHT_ARM_ACTION = slice(15, 22)
RIGHT_GRIPPER_ACTION = 22
NONBASE_ACTION = slice(3, 23)

BASE_QVEL_STATE = slice(0, 3)
LEFT_ARM_QPOS_STATE = slice(3, 10)
RIGHT_ARM_QPOS_STATE = slice(28, 35)
TRUNK_QPOS_STATE = slice(53, 57)

# Absolute joint-position targets participating in the whole-body path metric.
CONTINUOUS_ACTION_INDICES = np.r_[3:14, 15:22]

# Indices after removing action[0:3].
NONBASE_CONTINUOUS_INDICES = np.r_[0:11, 12:19]
NONBASE_GRIPPER_INDICES = np.asarray([11, 19], dtype=np.int64)


def measured_joint_positions(state: np.ndarray) -> np.ndarray:
    """Return measured trunk + left arm + right arm qpos in action-target order."""

    state = np.asarray(state)
    return np.concatenate(
        [
            state[..., TRUNK_QPOS_STATE],
            state[..., LEFT_ARM_QPOS_STATE],
            state[..., RIGHT_ARM_QPOS_STATE],
        ],
        axis=-1,
    )


def current_nonbase_target(state: np.ndarray, action: np.ndarray) -> np.ndarray:
    """Build the chunk-origin non-base configuration in action[3:23] order.

    Joint entries use measured qpos. The dataset has two finger positions per
    gripper but the controller target is one scalar, so grippers retain the
    current demonstration command convention.
    """

    state = np.asarray(state)
    action = np.asarray(action)
    return np.concatenate(
        [
            state[..., TRUNK_QPOS_STATE],
            state[..., LEFT_ARM_QPOS_STATE],
            action[..., LEFT_GRIPPER_ACTION : LEFT_GRIPPER_ACTION + 1],
            state[..., RIGHT_ARM_QPOS_STATE],
            action[..., RIGHT_GRIPPER_ACTION : RIGHT_GRIPPER_ACTION + 1],
        ],
        axis=-1,
    )


def validate_shapes(actions: np.ndarray, states: np.ndarray) -> None:
    if actions.ndim != 2 or actions.shape[1] != ACTION_DIM:
        raise ValueError(f"Expected actions with shape (N, {ACTION_DIM}), got {actions.shape}")
    if states.ndim != 2 or states.shape[1] != STATE_DIM:
        raise ValueError(f"Expected states with shape (N, {STATE_DIM}), got {states.shape}")
