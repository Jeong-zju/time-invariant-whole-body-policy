"""Run measured-pose tracking trials at a requested RoboCasa control rate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import gymnasium as gym
import numpy as np

from .adaptive_path import wrap_angle
from .calibrate_base_30hz import native_hold_action
from .geometric_follower import BaseRateCalibration, MeasuredPosePathFollower


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260823)
    parser.add_argument("--control-hz", type=float, default=30.0)
    parser.add_argument("--max-steps", type=int, default=150)
    args = parser.parse_args()

    import gr00t.eval.sim.robocasa365.gymnasium_groot  # noqa: F401

    calibration = BaseRateCalibration.load(args.calibration)
    if abs(calibration.fitted_control_hz - args.control_hz) > 1e-9:
        raise ValueError(
            f"calibration fitted at {calibration.fitted_control_hz} Hz, "
            f"requested {args.control_hz} Hz"
        )
    environment_name = "robocasa365_panda_omron/NavigateKitchen_PandaOmron_Env"
    env = gym.make(
        environment_name,
        enable_render=False,
        split="target",
        control_freq=args.control_hz,
    )
    actual_hz = float(env.unwrapped.control_freq)
    if abs(actual_hz - args.control_hz) > 1e-9:
        raise RuntimeError(f"requested {args.control_hz} Hz but simulator reports {actual_hz}")

    trials = {
        # The robot starts facing its source fixture.  Negative local x moves
        # away from that fixture and avoids turning this controller test into
        # an obstacle-collision test.
        "backward_0p10m": np.asarray([[-0.05, 0.0, 0.0], [-0.10, 0.0, 0.0]]),
        "lateral_0p10m": np.asarray([[0.0, 0.05, 0.0], [0.0, 0.10, 0.0]]),
        "rotate_0p20rad": np.asarray([[0.0, 0.0, 0.10], [0.0, 0.0, 0.20]]),
        "backward_curve_0p10m": np.asarray([[-0.05, 0.00, -0.05], [-0.10, 0.03, -0.15]]),
    }
    records = []
    try:
        for trial_index, (name, path) in enumerate(trials.items()):
            observation, _ = env.reset(seed=args.seed)
            for _ in range(3):
                observation, _, _, _, _ = env.step(native_hold_action(np.zeros(3)))
            follower = MeasuredPosePathFollower(
                calibration,
                control_hz=args.control_hz,
                lookahead_m=0.04,
                max_translation_speed_mps=0.20,
                max_yaw_rate_radps=0.60,
                # Preserve the original 30 Hz value (0.10/tick) as a
                # frequency-independent slew rate of 3 normalized units/s.
                max_command_delta_per_tick=3.0 / args.control_hz,
                goal_position_tolerance_m=0.01,
                goal_yaw_tolerance_rad=0.02,
            )
            follower.set_plan(
                path,
                observation["state.base_position"],
                observation["state.base_rotation"],
            )
            target = follower.world_path[-1].copy()
            trace = []
            done = False
            for step in range(1, args.max_steps + 1):
                command, info = follower.command(
                    observation["state.base_position"],
                    observation["state.base_rotation"],
                )
                trace.append(
                    {
                        "step": step,
                        "command": command[:3].astype(float).tolist(),
                        "progress_m": float(info["progress_m"]),
                        "endpoint_position_error_m": float(info.get("endpoint_position_error_m", np.nan)),
                        "endpoint_yaw_error_rad": float(info.get("endpoint_yaw_error_rad", np.nan)),
                    }
                )
                if bool(info["done"]):
                    done = True
                    break
                observation, _, _, _, _ = env.step(native_hold_action(command[:3]))
            final_position = np.asarray(observation["state.base_position"], dtype=np.float64)[:2]
            from .b2_labels import yaw_from_xyzw

            final_yaw = float(yaw_from_xyzw(np.asarray(observation["state.base_rotation"]).reshape(1, 4))[0])
            position_error = float(np.linalg.norm(target[:2] - final_position))
            yaw_error = abs(float(wrap_angle(target[2] - final_yaw)))
            passed = position_error <= 0.02 and yaw_error <= 0.04
            records.append(
                {
                    "trial": name,
                    "steps": len(trace),
                    "follower_done": done,
                    "position_error_m": position_error,
                    "yaw_error_rad": yaw_error,
                    "passed_0p02m_0p04rad": passed,
                    "target_world_pose": target.tolist(),
                    "final_world_pose": [float(final_position[0]), float(final_position[1]), final_yaw],
                    "trace": trace,
                }
            )
            print(json.dumps({"event": "TRACKING_TRIAL_COMPLETE", **{k: v for k, v in records[-1].items() if k != "trace"}}), flush=True)
    finally:
        env.close()

    payload = {
        "task": "NavigateKitchen",
        "environment": environment_name,
        "actual_control_hz": actual_hz,
        "calibration": str(args.calibration),
        "position_gate_m": 0.02,
        "yaw_gate_rad": 0.04,
        "all_pass": all(bool(record["passed_0p02m_0p04rad"]) for record in records),
        "trials": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(
        json.dumps(
            {
                "event": "FOLLOWER_FREQUENCY_VALIDATION_COMPLETE",
                "control_hz": actual_hz,
                "all_pass": payload["all_pass"],
                "output": str(args.output),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
