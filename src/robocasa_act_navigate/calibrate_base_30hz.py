"""Fit RoboCasa normalized base commands to measured twist at a requested Hz."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import gymnasium as gym
import numpy as np

from .b2_labels import yaw_from_xyzw
from .b2_se2 import between, log
from .geometric_follower import BaseRateCalibration


def native_hold_action(base_command: np.ndarray) -> dict[str, np.ndarray | int]:
    base = np.zeros(4, dtype=np.float32)
    base[:3] = np.asarray(base_command, dtype=np.float32)
    return {
        "action.base_motion": base,
        "action.control_mode": 1,
        "action.end_effector_position": np.zeros(3, dtype=np.float32),
        "action.end_effector_rotation": np.zeros(3, dtype=np.float32),
        "action.gripper_close": -1,
    }


def observation_pose(observation: dict) -> np.ndarray:
    position = np.asarray(observation["state.base_position"], dtype=np.float64)
    quaternion = np.asarray(observation["state.base_rotation"], dtype=np.float64)
    yaw = float(yaw_from_xyzw(quaternion.reshape(1, 4))[0])
    return np.asarray([position[0], position[1], yaw])


def fit_rate_calibration(commands: np.ndarray, measured_twists: np.ndarray, control_hz: float) -> tuple[BaseRateCalibration, dict]:
    commands = np.asarray(commands, dtype=np.float64)
    measured = np.asarray(measured_twists, dtype=np.float64)
    if commands.ndim != 2 or commands.shape[1] != 3 or measured.shape != commands.shape:
        raise ValueError("commands and measured_twists must both be (N,3)")
    design = np.column_stack((commands, np.ones(len(commands), dtype=np.float64)))
    coefficients, _, _, _ = np.linalg.lstsq(design, measured, rcond=None)
    calibration = BaseRateCalibration(coefficients[:3], coefficients[3], control_hz)
    prediction = commands @ calibration.matrix + calibration.bias
    residual = prediction - measured
    rmse = np.sqrt(np.mean(np.square(residual), axis=0))
    centered = measured - measured.mean(axis=0)
    denominator = np.sum(np.square(centered), axis=0)
    r_squared = 1.0 - np.sum(np.square(residual), axis=0) / np.maximum(denominator, 1e-12)
    return calibration, {
        "num_samples": int(len(commands)),
        "rmse_body_twist_per_second": rmse.tolist(),
        "r_squared": r_squared.tolist(),
        "max_abs_residual": np.max(np.abs(residual), axis=0).tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--control-hz", type=float, default=30.0)
    parser.add_argument("--split", default="target")
    parser.add_argument("--seed", type=int, default=20260823)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--record-steps", type=int, default=15)
    args = parser.parse_args()

    import gr00t.eval.sim.robocasa365.gymnasium_groot  # noqa: F401

    environment_name = "robocasa365_panda_omron/NavigateKitchen_PandaOmron_Env"
    env = gym.make(
        environment_name,
        enable_render=False,
        split=args.split,
        control_freq=args.control_hz,
    )
    actual_control_hz = float(env.unwrapped.control_freq)
    actual_control_timestep = float(env.unwrapped.control_timestep)
    if abs(actual_control_hz - args.control_hz) > 1e-9:
        raise RuntimeError(f"requested {args.control_hz} Hz but simulator reports {actual_control_hz} Hz")

    command_rows: list[np.ndarray] = []
    twist_rows: list[np.ndarray] = []
    pulse_commands = []
    for axis in range(3):
        for amplitude in (-0.5, -0.25, 0.25, 0.5):
            command = np.zeros(3, dtype=np.float64)
            command[axis] = amplitude
            pulse_commands.append(command)
    try:
        for pulse_index, command in enumerate(pulse_commands):
            observation, _ = env.reset(seed=args.seed + pulse_index)
            for _ in range(3):
                observation, _, _, _, _ = env.step(native_hold_action(np.zeros(3)))
            previous = observation_pose(observation)
            for step in range(args.warmup_steps + args.record_steps):
                observation, _, _, _, _ = env.step(native_hold_action(command))
                current = observation_pose(observation)
                measured_twist = log(between(previous, current)) * actual_control_hz
                if step >= args.warmup_steps:
                    command_rows.append(command.copy())
                    twist_rows.append(measured_twist)
                previous = current
    finally:
        env.close()

    commands = np.asarray(command_rows, dtype=np.float64)
    measured_twists = np.asarray(twist_rows, dtype=np.float64)
    calibration, diagnostics = fit_rate_calibration(commands, measured_twists, actual_control_hz)
    payload = {
        "task": "NavigateKitchen",
        "environment": environment_name,
        "split": args.split,
        "base_rate": calibration.to_dict(),
        "actual_control_timestep_s": actual_control_timestep,
        "native_hold": {
            "torso": 0.0,
            "control_mode": 1,
            "end_effector_position": [0.0, 0.0, 0.0],
            "end_effector_rotation": [0.0, 0.0, 0.0],
            "gripper_close": -1.0,
        },
        "pulse_amplitudes": [-0.5, -0.25, 0.25, 0.5],
        "warmup_steps": args.warmup_steps,
        "record_steps_per_pulse": args.record_steps,
        "diagnostics": diagnostics,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(
        json.dumps(
            {
                "event": "BASE_RATE_CALIBRATION_COMPLETE",
                "control_hz": actual_control_hz,
                "output": str(args.output),
                **diagnostics,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
