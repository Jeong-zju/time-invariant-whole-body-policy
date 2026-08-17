"""GR00T simulation wrapper for LP-GR00T-Base V1."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from gr00t.policy.policy import PolicyWrapper
from lp_groot_base.execution import ControllerCalibration, path_to_base_commands


class LPBaseExecutionPolicy(PolicyWrapper):
    """Decode LP predictions while commanding only the mobile base."""

    def __init__(self, policy, calibration_path: str, num_control_steps: int = 8):
        super().__init__(policy, strict=False)
        report = json.loads(Path(calibration_path).read_text())
        self.calibration = ControllerCalibration(
            command_to_twist=np.asarray(report["command_to_twist"], dtype=np.float64),
            command_offset=np.asarray(report["command_offset"], dtype=np.float64),
            control_dt=float(report["control_dt_median"]),
        )
        self.num_control_steps = num_control_steps

    def check_observation(self, observation: dict[str, Any]) -> None:
        return None

    def check_action(self, action: dict[str, Any]) -> None:
        return None

    def _get_action(self, observation, options=None):
        prediction, model_info = self.policy.get_action(observation, options)
        paths = np.asarray(prediction["action.lp_base_path"], dtype=np.float32)
        batch = paths.shape[0]
        base = np.zeros((batch, self.num_control_steps, 4), dtype=np.float32)
        desired = np.zeros((batch, self.num_control_steps, 3), dtype=np.float32)
        for index in range(batch):
            commands, desired_poses = path_to_base_commands(
                paths[index], self.calibration, self.num_control_steps
            )
            base[index, :, :3] = commands
            desired[index] = desired_poses
        zeros3 = np.zeros((batch, self.num_control_steps, 3), dtype=np.float32)
        zeros1 = np.zeros((batch, self.num_control_steps, 1), dtype=np.float32)
        action = {
            "action.end_effector_position": zeros3.copy(),
            "action.end_effector_rotation": zeros3.copy(),
            "action.gripper_close": zeros1.copy(),
            "action.base_motion": base,
            # RoboCasa's HybridMobileBase maps values >= 0.5 to base mode.
            # The LP labels and controller calibration use the same mode gate.
            "action.control_mode": np.ones_like(zeros1),
        }
        info = dict(model_info)
        info["lp_desired_local_pose"] = desired
        info["lp_normalized_base_command"] = base[:, :, :3].copy()
        return action, info
