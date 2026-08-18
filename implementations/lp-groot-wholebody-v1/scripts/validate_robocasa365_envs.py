#!/usr/bin/env python3
"""Construct the three exact RoboCasa365 target environments and audit I/O."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import robocasa
import robosuite


TASKS = (
    ("NavigateKitchen", "robocasa/NavigateKitchen", 450),
    ("PickPlaceCounterToStove", "robocasa/PickPlaceCounterToStove", 600),
    ("DeliverStraw", "robocasa/DeliverStraw", 2550),
)

EXPECTED_OBSERVATIONS = {
    "state.base_position",
    "state.base_rotation",
    "state.end_effector_position_relative",
    "state.end_effector_rotation_relative",
    "state.gripper_qpos",
    "video.robot0_eye_in_hand",
    "video.robot0_agentview_left",
    "video.robot0_agentview_right",
    "annotation.human.task_description",
}

EXPECTED_ACTIONS = {
    "action.base_motion",
    "action.control_mode",
    "action.end_effector_position",
    "action.end_effector_rotation",
    "action.gripper_close",
}


def zero_action(space: spaces.Space) -> Any:
    if isinstance(space, spaces.Dict):
        return {key: zero_action(value) for key, value in space.items()}
    if isinstance(space, spaces.Box):
        return np.zeros(space.shape, dtype=space.dtype)
    if isinstance(space, spaces.Discrete):
        return 0
    raise TypeError(f"unsupported action space: {space!r}")


def value_schema(value: Any) -> dict[str, Any]:
    if isinstance(value, str):
        return {"type": "str", "value": value}
    array = np.asarray(value)
    return {
        "type": "array",
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "finite": bool(np.all(np.isfinite(array))) if array.dtype.kind != "O" else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=20260818)
    args = parser.parse_args()

    report: dict[str, Any] = {
        "schema_version": 1,
        "robocasa_version": getattr(robocasa, "__version__", "unknown"),
        "robosuite_version": getattr(robosuite, "__version__", "unknown"),
        "split": "target",
        "seed": args.seed,
        "tasks": [],
    }
    for task_name, gym_id, official_horizon in TASKS:
        env = gym.make(gym_id, enable_render=False, split="target")
        try:
            observation, reset_info = env.reset(seed=args.seed)
            missing_observations = sorted(EXPECTED_OBSERVATIONS - set(observation))
            missing_actions = sorted(EXPECTED_ACTIONS - set(env.action_space.spaces))
            if missing_observations or missing_actions:
                raise RuntimeError(
                    f"{task_name} schema mismatch: missing observations="
                    f"{missing_observations}, missing actions={missing_actions}"
                )
            next_observation, reward, terminated, truncated, step_info = env.step(
                zero_action(env.action_space)
            )
            nonfinite = [
                key
                for key, value in next_observation.items()
                if not isinstance(value, str)
                and np.asarray(value).dtype.kind != "O"
                and not np.all(np.isfinite(np.asarray(value)))
            ]
            if nonfinite:
                raise RuntimeError(f"{task_name} produced non-finite observations: {nonfinite}")
            report["tasks"].append(
                {
                    "task": task_name,
                    "gym_id": gym_id,
                    "official_horizon": official_horizon,
                    "observation": {
                        key: value_schema(value) for key, value in observation.items()
                    },
                    "action_space_keys": sorted(env.action_space.spaces),
                    "reset_info": {
                        "keys": sorted(reset_info),
                        "success": bool(reset_info.get("success", False)),
                    },
                    "zero_step": {
                        "reward": float(reward),
                        "terminated": bool(terminated),
                        "truncated": bool(truncated),
                        "success": bool(step_info.get("success", False)),
                    },
                }
            )
        finally:
            env.close()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as file:
        json.dump(report, file, indent=2)
    output.with_name("ROBOCASA365_ENVIRONMENTS_VALIDATED").touch()
    print(json.dumps({"validated": [item[0] for item in TASKS]}, indent=2))


if __name__ == "__main__":
    main()
