"""Strict same-ACT-output causal ablation for command, path, and pointer execution."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import gymnasium as gym
import numpy as np

from .b2_labels import yaw_from_xyzw
from .b2_se2 import between
from .eval_server import LANGUAGE_SOURCES, NavigateACTPolicy
from .geometric_eval_server import NavigateGeometricPolicy
from .geometric_follower import BaseRateCalibration, MeasuredPosePathFollower
from .rate_control_eval_server import (
    array_sha256,
    calibrated_command_path,
    geometric_prefix,
    pointer_command,
    retimed_zoh_command,
)
from .strict_rate_replay import (
    _batch_observation,
    _input_hash,
    _native_action,
    _restore,
)


RATES = (0.5, 1.0, 1.5)
METHODS = (
    "act_native",
    "act_oracle",
    "act_calibrated_pointer",
    "act_measured_pointer_oracle",
    "learned_point_pointer",
)


def _writer(path: Path, frame: np.ndarray, fps: float) -> cv2.VideoWriter:
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frame.shape[:2]
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        raise RuntimeError(f"failed to open video writer: {path}")
    return writer


def _pose(observation: dict) -> np.ndarray:
    position = np.asarray(observation["state.base_position"], dtype=np.float64)
    rotation = np.asarray(observation["state.base_rotation"], dtype=np.float64)
    yaw = float(yaw_from_xyzw(rotation.reshape(1, 4))[0])
    return np.asarray([position[0], position[1], yaw])


def _measured_act_path(
    env,
    wrapper,
    xml: str,
    state: np.ndarray,
    act_chunk: np.ndarray,
    *,
    control_hz: float,
    source_hz: float,
    source_duration_s: float,
) -> np.ndarray:
    observation = _restore(wrapper, xml, state)
    world = [_pose(observation)]
    ticks = int(round(source_duration_s * control_hz))
    for tick in range(ticks):
        command3, _ = retimed_zoh_command(
            act_chunk,
            tick / control_hz,
            (tick + 1) / control_hz,
            rate_scale=1.0,
            source_hz=source_hz,
            source_duration_s=source_duration_s,
            compensate_amplitude=False,
        )
        base = np.concatenate((command3, np.zeros(1, dtype=np.float32)))
        observation, _, _, _, _ = env.step(_native_action(base))
        world.append(_pose(observation))
    world = np.asarray(world, dtype=np.float64)
    local = between(world[0], world[1:])
    local[:, 2] = np.unwrap(local[:, 2])
    return local.astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--act-checkpoint", type=Path, required=True)
    parser.add_argument("--point-checkpoint", type=Path, required=True)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--control-hz", type=int, default=50)
    parser.add_argument("--steps", type=int, default=250)
    parser.add_argument("--act-source-hz", type=float, default=20.0)
    parser.add_argument("--act-source-duration-s", type=float, default=1.6)
    parser.add_argument("--point-segment-extent-m", type=float, default=0.40)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    import gr00t.eval.sim.robocasa365.gymnasium_groot  # noqa: F401

    env = gym.make(
        "robocasa365_panda_omron/NavigateKitchen_PandaOmron_Env",
        enable_render=True,
        split="target",
        control_freq=args.control_hz,
        seed=args.seed,
    )
    wrapper = env.unwrapped
    observation, _ = env.reset(seed=args.seed)
    if abs(float(wrapper.control_freq) - args.control_hz) > 1e-12:
        raise RuntimeError("simulator frequency mismatch")
    capture_hash = _input_hash(observation)
    raw_env = wrapper.env
    xml = raw_env.sim.model.get_xml()
    state = raw_env.sim.get_state().flatten().copy()
    language = next(
        str(observation[key]) for key in LANGUAGE_SOURCES if key in observation
    )

    act = NavigateACTPolicy(args.act_checkpoint, args.tasks, args.device)
    act.reset()
    native, _ = act._get_action(_batch_observation(observation), None)
    act_chunk = np.asarray(
        native["action.base_motion"][0, :, :3], dtype=np.float32
    ).copy()
    act_hash = array_sha256(act_chunk)
    calibration = BaseRateCalibration.load(args.calibration)
    act_calibrated_path = calibrated_command_path(
        act_chunk, calibration, source_hz=args.act_source_hz
    )
    act_measured_path = _measured_act_path(
        env,
        wrapper,
        xml,
        state,
        act_chunk,
        control_hz=args.control_hz,
        source_hz=args.act_source_hz,
        source_duration_s=args.act_source_duration_s,
    )

    # Point is predicted from the exact original observation, not from the
    # temporary measured-oracle rollout above.
    observation = _restore(wrapper, xml, state)
    point = NavigateGeometricPolicy(
        args.point_checkpoint,
        args.tasks,
        args.calibration,
        args.device,
        control_hz=args.control_hz,
        replan_ticks=1,
        trace=None,
    )
    point.reset()
    point_full = point._predict(_batch_observation(observation))
    point_segment = geometric_prefix(
        point_full, args.point_segment_extent_m, yaw_radius_m=0.25
    )

    plans = {
        "act_calibrated_pointer": act_calibrated_path,
        "act_measured_pointer_oracle": act_measured_path,
        "learned_point_pointer": point_segment,
    }
    hashes = {
        "act_native": act_hash,
        "act_oracle": act_hash,
        **{method: array_sha256(path) for method, path in plans.items()},
    }
    np.save(args.output / "act_chunk.npy", act_chunk)
    np.save(args.output / "act_calibrated_path.npy", act_calibrated_path)
    np.save(args.output / "act_measured_path.npy", act_measured_path)
    np.save(args.output / "learned_point_path.npy", point_segment)

    summary = {
        "seed": args.seed,
        "language": language,
        "control_hz": args.control_hz,
        "steps": args.steps,
        "capture_input_sha256": capture_hash,
        "act_model_output_sha256": act_hash,
        "learned_point_model_output_sha256": array_sha256(point_full),
        "execution_plan_sha256": hashes,
        "act_source_hz": args.act_source_hz,
        "act_source_duration_s": args.act_source_duration_s,
        "calibration_semantics": calibration.to_dict()["semantics"],
        "act_calibrated_path_is_ground_truth": False,
        "act_measured_path_is_deployable": False,
        "conditions": [],
    }
    for method in METHODS:
        for rate in RATES:
            restored = _restore(wrapper, xml, state)
            restored_state = raw_env.sim.get_state().flatten().copy()
            state_error = float(np.max(np.abs(restored_state - state)))
            if state_error > 1e-12:
                raise RuntimeError(
                    f"strict state mismatch seed={args.seed} method={method} "
                    f"rate={rate}: {state_error}"
                )
            tag = str(rate).replace(".", "p")
            directory = args.output / f"{method}_rate{tag}"
            trace_path = directory / "trace.jsonl"
            video_path = directory / "rollout.mp4"
            writer = _writer(video_path, wrapper.render(), args.control_hz)
            follower = None
            nominal_progress_rate = None
            if method in plans:
                follower = MeasuredPosePathFollower(
                    calibration,
                    control_hz=args.control_hz,
                    max_command_delta_per_tick=4.5 / args.control_hz,
                )
                follower.set_plan(
                    plans[method],
                    restored["state.base_position"],
                    restored["state.base_rotation"],
                )
                nominal_progress_rate = (
                    float(follower.cumulative_progress[-1])
                    / args.act_source_duration_s
                )
            rows = []
            for tick in range(args.steps):
                position = np.asarray(restored["state.base_position"])
                rotation = np.asarray(restored["state.base_rotation"])
                if method in ("act_native", "act_oracle"):
                    command3, details = retimed_zoh_command(
                        act_chunk,
                        tick / args.control_hz,
                        (tick + 1) / args.control_hz,
                        rate_scale=rate,
                        source_hz=args.act_source_hz,
                        source_duration_s=args.act_source_duration_s,
                        compensate_amplitude=method == "act_oracle",
                    )
                    base = np.concatenate(
                        (command3, np.zeros(1, dtype=np.float32))
                    )
                    executor = {"retime": details}
                else:
                    assert follower is not None and nominal_progress_rate is not None
                    reference = (
                        (tick + 1)
                        / args.control_hz
                        * nominal_progress_rate
                        * rate
                    )
                    base, details = pointer_command(
                        follower, position, rotation, reference
                    )
                    executor = {
                        "pointer": details,
                        "nominal_progress_rate_mps": nominal_progress_rate,
                    }
                rows.append(
                    {
                        "event": "ACTION",
                        "seed": args.seed,
                        "method": method,
                        "rate_scale": rate,
                        "tick": tick,
                        "model_input_sha256": capture_hash,
                        "source_act_model_output_sha256": (
                            act_hash if method != "learned_point_pointer" else None
                        ),
                        "execution_plan_sha256": hashes[method],
                        "restored_state_max_abs_error": state_error,
                        "position": position.reshape(-1)[:2].tolist(),
                        "yaw": float(yaw_from_xyzw(rotation.reshape(1, 4))[0]),
                        "base_command": np.asarray(base).tolist(),
                        **executor,
                    }
                )
                restored, _, _, _, _ = env.step(_native_action(base))
                writer.write(np.ascontiguousarray(wrapper.render()[..., ::-1]))
            writer.release()
            trace_path.write_text(
                "".join(
                    json.dumps(row, separators=(",", ":")) + "\n"
                    for row in rows
                )
            )
            if len(rows) != args.steps or video_path.stat().st_size == 0:
                raise RuntimeError("causal ablation artifact mismatch")
            condition_summary = {
                "method": method,
                "rate_scale": rate,
                "model_input_sha256": capture_hash,
                "execution_plan_sha256": hashes[method],
                "restored_state_max_abs_error": state_error,
                "trace": str(trace_path),
                "video": str(video_path),
            }
            (directory / "summary.json").write_text(
                json.dumps(condition_summary, indent=2) + "\n"
            )
            summary["conditions"].append(condition_summary)
    env.close()
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (args.output / "STRICT_ACT_PATH_ABLATION_COMPLETE").write_text("\n")
    print(json.dumps({"event": "STRICT_ACT_PATH_ABLATION_COMPLETE", "seed": args.seed}))


if __name__ == "__main__":
    main()
