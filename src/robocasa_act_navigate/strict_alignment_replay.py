"""Exact-state stale-plan alignment gate for native ACT and path pointers.

The model predicts once from the reset observation.  A locally consistent old
plan advances the simulator while that prediction is assumed to be in flight.
When it arrives after 0.4 or 0.8 seconds, native ACT restarts the stale chunk at
token zero, while the geometric executor keeps the original world path and
recovers its pointer by projecting the measured pose.  Each condition restores
the exact same MuJoCo state and uses byte-identical model outputs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import gymnasium as gym
import numpy as np

from .eval_server import NavigateACTPolicy
from .geometric_eval_server import NavigateGeometricPolicy
from .geometric_follower import BaseRateCalibration, MeasuredPosePathFollower
from .rate_control_eval_server import (
    array_sha256,
    geometric_prefix,
    pointer_command,
    retimed_zoh_command,
)
from .strict_rate_replay import (
    LANGUAGE_SOURCES,
    _batch_observation,
    _input_hash,
    _native_action,
    _restore,
)
from .b2_labels import yaw_from_xyzw


DELAYS_S = (0.4, 0.8)


def _writer(path: Path, frame: np.ndarray, fps: float) -> cv2.VideoWriter:
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frame.shape[:2]
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
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
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--control-hz", type=int, default=50)
    parser.add_argument("--steps", type=int, default=250)
    parser.add_argument("--act-source-duration-s", type=float, default=1.6)
    parser.add_argument("--point-segment-extent-m", type=float, default=0.40)
    parser.add_argument("--point-progress-rate-mps", type=float, default=0.25)
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
    capture_hash = _input_hash(observation)
    raw_env = wrapper.env
    xml = raw_env.sim.model.get_xml()
    state = raw_env.sim.get_state().flatten().copy()
    capture_position = np.asarray(observation["state.base_position"]).copy()
    capture_rotation = np.asarray(observation["state.base_rotation"]).copy()
    language = next(
        str(observation[key]) for key in LANGUAGE_SOURCES if key in observation
    )

    act = NavigateACTPolicy(args.act_checkpoint, args.tasks, args.device)
    act.reset()
    native, _ = act._get_action(_batch_observation(observation), None)
    act_chunk = np.asarray(native["action.base_motion"][0, :, :3], dtype=np.float32)

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
    point_segment = geometric_prefix(point_full, args.point_segment_extent_m)
    calibration = BaseRateCalibration.load(args.calibration)
    act_hash = array_sha256(act_chunk)
    point_hash = array_sha256(point_full)

    conditions = [
        ("act", 0.0, "reference"),
        ("point", 0.0, "reference"),
    ]
    for delay in DELAYS_S:
        conditions.extend(
            [
                ("act", delay, "stale_restart"),
                ("point", delay, "projected_pointer"),
            ]
        )

    summary = {
        "seed": args.seed,
        "language": language,
        "control_hz": args.control_hz,
        "steps": args.steps,
        "capture_input_sha256": capture_hash,
        "act_output_sha256": act_hash,
        "point_output_sha256": point_hash,
        "delays_s": list(DELAYS_S),
        "conditions": [],
    }
    np.save(args.output / "act_chunk.npy", act_chunk)
    np.save(args.output / "point_segment.npy", point_segment)

    dt = 1.0 / args.control_hz
    for method, delay_s, adapter in conditions:
        restored = _restore(wrapper, xml, state)
        restored_state = raw_env.sim.get_state().flatten().copy()
        state_error = float(np.max(np.abs(restored_state - state)))
        if state_error > 1e-12:
            raise RuntimeError(f"strict state replay mismatch: {state_error}")
        delay_ticks = int(round(delay_s * args.control_hz))
        delay_tag = str(delay_s).replace(".", "p")
        condition = f"{method}_{adapter}_delay{delay_tag}"
        directory = args.output / condition
        trace_path = directory / "trace.jsonl"
        video_path = directory / "rollout.mp4"
        writer = _writer(video_path, wrapper.render(), args.control_hz)
        follower = None
        pointer = 0.0
        last_command = np.zeros(3, dtype=np.float64)
        if method == "point":
            follower = MeasuredPosePathFollower(
                calibration,
                control_hz=args.control_hz,
                max_command_delta_per_tick=4.5 / args.control_hz,
            )
            follower.set_plan(point_segment, capture_position, capture_rotation)
        rows = []
        for tick in range(args.steps):
            position = np.asarray(restored["state.base_position"])
            rotation = np.asarray(restored["state.base_rotation"])
            alignment_event = False
            if method == "act":
                source_tick = tick
                if adapter == "stale_restart" and tick >= delay_ticks:
                    source_tick = tick - delay_ticks
                    alignment_event = tick == delay_ticks
                command3, details = retimed_zoh_command(
                    act_chunk,
                    source_tick * dt,
                    (source_tick + 1) * dt,
                    rate_scale=1.0,
                    source_hz=20.0,
                    source_duration_s=args.act_source_duration_s,
                    compensate_amplitude=False,
                )
                base = np.concatenate((command3, np.zeros(1, dtype=np.float32)))
                output_hash = act_hash
                executor = {"source_tick": source_tick, "retime": details}
            else:
                assert follower is not None
                if adapter == "projected_pointer" and tick == delay_ticks:
                    last_command = follower.last_command.copy()
                    follower = MeasuredPosePathFollower(
                        calibration,
                        control_hz=args.control_hz,
                        max_command_delta_per_tick=4.5 / args.control_hz,
                    )
                    # The stale relative path remains anchored at the pose from
                    # which it was predicted.  Re-anchoring at the arrival pose
                    # would silently change the model output's meaning.
                    follower.set_plan(point_segment, capture_position, capture_rotation)
                    follower.last_command = last_command
                    pointer = follower._project_progress(
                        np.asarray(position).reshape(-1)[:2],
                        follower._world_yaw(rotation),
                    )
                    alignment_event = True
                pointer += args.point_progress_rate_mps * dt
                base, details = pointer_command(follower, position, rotation, pointer)
                output_hash = point_hash
                executor = {"pointer": details}
            rows.append(
                {
                    "event": "ACTION",
                    "seed": args.seed,
                    "method": method,
                    "adapter": adapter,
                    "delay_s": delay_s,
                    "tick": tick,
                    "alignment_event": alignment_event,
                    "model_input_sha256": capture_hash,
                    "output_sha256": output_hash,
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
            "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows)
        )
        if len(rows) != args.steps or video_path.stat().st_size == 0:
            raise RuntimeError("alignment replay artifact mismatch")
        condition_summary = {
            "method": method,
            "adapter": adapter,
            "delay_s": delay_s,
            "model_input_sha256": capture_hash,
            "output_sha256": output_hash,
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
    (args.output / "STRICT_ALIGNMENT_REPLAY_COMPLETE").write_text("\n")
    print(json.dumps({"event": "STRICT_ALIGNMENT_REPLAY_COMPLETE", "seed": args.seed}))


if __name__ == "__main__":
    main()
