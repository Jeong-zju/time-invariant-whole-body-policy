"""Run all ACT/Point execution rates from one exactly replayed simulator state."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import gymnasium as gym
import numpy as np

from .b2_labels import yaw_from_xyzw
from .eval_server import LANGUAGE_SOURCES, STATE_SOURCES, VIDEO_SOURCES, NavigateACTPolicy
from .geometric_eval_server import NavigateGeometricPolicy
from .geometric_follower import BaseRateCalibration, MeasuredPosePathFollower
from .rate_control_eval_server import (
    array_sha256,
    geometric_prefix,
    pointer_command,
    retimed_zoh_command,
)


RATES = (0.5, 1.0, 1.5)


def _batch_observation(observation: dict) -> dict:
    result = {}
    for key in set(VIDEO_SOURCES.values()) | set(STATE_SOURCES):
        result[key] = np.asarray(observation[key])[None, None]
    for key in LANGUAGE_SOURCES:
        if key in observation:
            result[key] = [observation[key]]
    return result


def _input_hash(observation: dict) -> str:
    digest = hashlib.sha256()
    for key in sorted(set(VIDEO_SOURCES.values()) | set(STATE_SOURCES)):
        value = np.ascontiguousarray(np.asarray(observation[key]))
        digest.update(key.encode())
        digest.update(str(value.dtype).encode())
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.tobytes())
    language = next(str(observation[key]) for key in LANGUAGE_SOURCES if key in observation)
    digest.update(language.encode())
    return digest.hexdigest()


def _native_action(base: np.ndarray) -> dict[str, np.ndarray | float]:
    return {
        "action.base_motion": np.asarray(base, dtype=np.float32),
        "action.control_mode": 1.0,
        "action.end_effector_position": np.zeros(3, dtype=np.float32),
        "action.end_effector_rotation": np.zeros(3, dtype=np.float32),
        "action.gripper_close": -1.0,
    }


def _restore(wrapper, xml: str, state: np.ndarray) -> dict:
    raw_env = wrapper.env
    raw_env.reset_from_xml_string(xml)
    raw_env.sim.set_state_from_flattened(state)
    raw_env.sim.forward()
    raw = raw_env._get_observations(force_update=True)
    return wrapper._get_groot_observation(raw)


def _video_writer(path: Path, frame: np.ndarray, fps: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frame.shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"failed to open video writer: {path}")
    return writer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--act-checkpoint", type=Path, required=True)
    parser.add_argument("--point-checkpoint", type=Path, required=True)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--control-hz", type=int, default=50)
    parser.add_argument("--steps", type=int, default=250)
    parser.add_argument("--act-source-duration-s", type=float, default=1.6)
    parser.add_argument("--point-segment-extent-m", type=float, default=0.40)
    parser.add_argument("--point-nominal-progress-rate-mps", type=float, default=0.25)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    # Importing registers the upstream environment namespace.
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
    language = next(str(observation[key]) for key in LANGUAGE_SOURCES if key in observation)

    act = NavigateACTPolicy(args.act_checkpoint, args.tasks, args.device)
    act.reset()
    native, _ = act._get_action(_batch_observation(observation), None)
    act_chunk = np.asarray(native["action.base_motion"][0, :, :3], dtype=np.float32).copy()
    act_hash = array_sha256(act_chunk)

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
        point_full,
        args.point_segment_extent_m,
        yaw_radius_m=0.25,
    )
    point_hash = array_sha256(point_full)
    point_segment_hash = array_sha256(point_segment)
    calibration = BaseRateCalibration.load(args.calibration)

    summary = {
        "seed": args.seed,
        "language": language,
        "control_hz": args.control_hz,
        "steps": args.steps,
        "capture_input_sha256": capture_hash,
        "act_output_sha256": act_hash,
        "point_output_sha256": point_hash,
        "point_segment_sha256": point_segment_hash,
        "act_source_duration_s": args.act_source_duration_s,
        "point_segment_extent_m": args.point_segment_extent_m,
        "point_nominal_progress_rate_mps": args.point_nominal_progress_rate_mps,
        "act_chunk_mean_abs": float(np.mean(np.abs(act_chunk))),
        "act_chunk_max_abs": float(np.max(np.abs(act_chunk))),
        "conditions": [],
    }
    np.save(args.output / "act_chunk.npy", act_chunk)
    np.save(args.output / "point_segment.npy", point_segment)
    for method in ("act_native", "act_oracle", "point"):
        for rate in RATES:
            restored = _restore(wrapper, xml, state)
            restored_hash = _input_hash(restored)
            restored_state = raw_env.sim.get_state().flatten().copy()
            state_max_abs_error = float(np.max(np.abs(restored_state - state)))
            if state_max_abs_error > 1e-12:
                raise RuntimeError(
                    f"strict replay state mismatch: seed={args.seed} method={method} "
                    f"rate={rate} max_abs={state_max_abs_error}"
                )
            condition = f"{method}_rate{str(rate).replace('.', 'p')}"
            trace_path = args.output / condition / "trace.jsonl"
            video_path = args.output / condition / "rollout.mp4"
            trace_path.parent.mkdir(parents=True, exist_ok=True)
            frame = wrapper.render()
            writer = _video_writer(video_path, frame, args.control_hz)
            follower = None
            if method == "point":
                follower = MeasuredPosePathFollower(
                    calibration,
                    control_hz=args.control_hz,
                    max_command_delta_per_tick=4.5 / args.control_hz,
                )
                follower.set_plan(
                    point_segment,
                    restored["state.base_position"],
                    restored["state.base_rotation"],
                )
            rows = []
            task_success_any = False
            for tick in range(args.steps):
                position = np.asarray(restored["state.base_position"])
                quaternion = np.asarray(restored["state.base_rotation"])
                if method in ("act_native", "act_oracle"):
                    command3, info = retimed_zoh_command(
                        act_chunk,
                        tick / args.control_hz,
                        (tick + 1) / args.control_hz,
                        rate_scale=rate,
                        source_hz=20.0,
                        source_duration_s=args.act_source_duration_s,
                        compensate_amplitude=method == "act_oracle",
                    )
                    base = np.concatenate((command3, np.zeros(1, dtype=np.float32)))
                    details = {"retime": info}
                    output_hash = act_hash
                else:
                    assert follower is not None
                    reference = (
                        (tick + 1)
                        / args.control_hz
                        * args.point_nominal_progress_rate_mps
                        * rate
                    )
                    base, info = pointer_command(follower, position, quaternion, reference)
                    details = {"pointer": info}
                    output_hash = point_hash
                row = {
                    "event": "ACTION",
                    "seed": args.seed,
                    "method": method,
                    "rate_scale": rate,
                    "control_hz": args.control_hz,
                    "tick": tick,
                    "input_sha256": capture_hash,
                    "restored_observation_sha256_diagnostic": restored_hash,
                    "restored_state_max_abs_error": state_max_abs_error,
                    "output_sha256": output_hash,
                    "position": position.reshape(-1)[:2].tolist(),
                    "yaw": float(yaw_from_xyzw(quaternion.reshape(1, 4))[0]),
                    "base_command": np.asarray(base).tolist(),
                    **details,
                }
                rows.append(row)
                restored, _, _, _, info_env = env.step(_native_action(base))
                task_success_any = task_success_any or bool(info_env.get("success", False))
                writer.write(np.ascontiguousarray(wrapper.render()[..., ::-1]))
            writer.release()
            trace_path.write_text("".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows))
            if len(rows) != args.steps or not video_path.is_file() or video_path.stat().st_size == 0:
                raise RuntimeError("strict replay artifact mismatch")
            condition_summary = {
                "method": method,
                "rate_scale": rate,
                "input_sha256": restored_hash,
                "model_input_sha256": capture_hash,
                "restored_state_max_abs_error": state_max_abs_error,
                "output_sha256": output_hash,
                "trace": str(trace_path),
                "video": str(video_path),
                "task_success_any_diagnostic_only": task_success_any,
            }
            (trace_path.parent / "summary.json").write_text(json.dumps(condition_summary, indent=2) + "\n")
            summary["conditions"].append(condition_summary)
    env.close()
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (args.output / "STRICT_REPLAY_COMPLETE").write_text("\n")
    print(json.dumps({"event": "STRICT_REPLAY_COMPLETE", "seed": args.seed}))


if __name__ == "__main__":
    main()
