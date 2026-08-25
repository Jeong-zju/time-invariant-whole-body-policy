"""Verified RoboCasa v1.0 LineUpCondiments schema constants."""

from __future__ import annotations

import numpy as np

FPS = 20
CONTROL_DT = 1.0 / FPS
NUM_QUERIES = 32
EXECUTION_STEPS = 8
STATE_DIM = 16
ACTION_DIM = 12

# action[0:3] is the mobile chassis command. action[3] is torso motion,
# action[4] is control mode, and action[5:12] is arm/gripper control.
BASE_ACTION = slice(0, 3)
NONBASE_ACTION = slice(3, 12)
NONBASE_DIM = 9

# LP query = local base pose (x, y, yaw), log segment duration, and the
# ordinary time-indexed non-base action for the same ACT query.
LP_BASE_POSE = slice(0, 3)
LP_LOG_DURATION = 3
LP_NONBASE_ACTION = slice(4, 13)
LP_ACTION_DIM = 13

BASE_POSITION_STATE = slice(0, 3)
BASE_QUATERNION_STATE = slice(3, 7)  # xyzw, verified from dataset and converter

CAMERA_KEYS = (
    "observation.images.robot0_eye_in_hand",
    "observation.images.robot0_agentview_left",
    "observation.images.robot0_agentview_right",
)
IMAGE_SIZE = (256, 256)


def validate_shapes(actions: np.ndarray, states: np.ndarray) -> None:
    if actions.ndim != 2 or actions.shape[1] != ACTION_DIM:
        raise ValueError(f"Expected actions (N,{ACTION_DIM}), got {actions.shape}")
    if states.ndim != 2 or states.shape[1] != STATE_DIM:
        raise ValueError(f"Expected states (N,{STATE_DIM}), got {states.shape}")
