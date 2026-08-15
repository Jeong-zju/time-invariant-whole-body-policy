#!/usr/bin/env python3
"""Run one Arena rollout with a chosen closed-loop replanning interval.

The simulator action/control clock remains unchanged.  ``replan_steps`` only
controls how many rows of each GR00T action chunk are consumed before a fresh
chunk is requested.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import time
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch
import tqdm

from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
from isaaclab_arena.examples.example_environments.cli import get_arena_builder_from_cli
from isaaclab_arena.examples.policy_runner_cli import create_policy, setup_policy_argument_parser
from isaaclab_arena.utils.isaaclab_utils.simulation_app import SimulationAppContext

from action_consumers import make_consumer
from frequency_metrics import nested_observation_sha256, summarize_trajectory, yaw_from_wxyz


PROTOCOL_ID = "arena-g1-gate-n-frequency-v1"


def _numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _first_env(value: Any) -> np.ndarray:
    array = _numpy(value)
    return np.asarray(array[0], dtype=np.float64)


def _latest_first_env_observation(value: Any, feature_dim: int) -> np.ndarray:
    array = _first_env(value)
    if array.ndim == 1:
        if array.shape[0] != feature_dim:
            raise ValueError(f"expected feature dimension {feature_dim}, got {array.shape}")
    elif array.ndim == 2:
        if array.shape[-1] == feature_dim:
            array = array[-1]
        elif array.shape[0] == feature_dim:
            array = array[:, -1]
        else:
            raise ValueError(f"cannot find feature dimension {feature_dim} in observation shape {array.shape}")
    else:
        raise ValueError(f"expected a feature vector with optional history, got shape {array.shape}")
    return np.asarray(array, dtype=np.float64)


def _first_env_pose_matrix(value: Any) -> np.ndarray:
    array = _first_env(value)
    if array.shape != (4, 4):
        raise ValueError(f"expected a 4x4 pose matrix, got {array.shape}")
    return np.asarray(array, dtype=np.float64)


def _jsonable(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return value.item()
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _seed_first_policy_inference(seed: int) -> None:
    """Make the paired conditions start from the same stochastic policy draw."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _register_gate_n_data_configs() -> None:
    """Make the local delta-t config visible to Arena's policy factory."""
    from arena_g1_replan_dt_data_config import UnitreeG1SimWBCReplanDtDataConfig
    from gr00t.experiment.data_config import DATA_CONFIG_MAP

    DATA_CONFIG_MAP["unitree_g1_sim_wbc_replan_dt"] = UnitreeG1SimWBCReplanDtDataConfig()


def main() -> None:
    args_parser = get_isaaclab_arena_cli_parser()
    args_parser.add_argument("--telemetry_output", type=Path, required=True)
    args_parser.add_argument("--replan_steps", type=int, required=True)
    args_parser.add_argument("--first_chunk_reference", type=Path)
    args_parser.add_argument(
        "--consumer",
        choices=("raw", "se2_waypoint"),
        default="raw",
    )
    args_parser.add_argument("--protocol_id", default=PROTOCOL_ID)
    args_cli, _ = args_parser.parse_known_args()

    with SimulationAppContext(args_cli):
        args_parser = setup_policy_argument_parser(args_parser)
        args_cli = args_parser.parse_args()
        if args_cli.replan_steps <= 0:
            raise ValueError("--replan_steps must be positive")

        arena_builder = get_arena_builder_from_cli(args_cli)
        env_name, env_cfg = arena_builder.build_registered()
        # Arena's default recorder filename is shared by every process.  Use a
        # per-rollout location so frequency conditions can run concurrently
        # without HDF5 file-lock collisions.
        env_cfg.recorders.dataset_export_dir_path = str(args_cli.telemetry_output.parent)
        env_cfg.recorders.dataset_filename = "arena-metrics"
        env = gym.make(env_name, cfg=env_cfg).unwrapped

        if args_cli.seed is not None:
            env.seed(args_cli.seed)
            torch.manual_seed(args_cli.seed)
            np.random.seed(args_cli.seed)
            random.seed(args_cli.seed)

        obs, _ = env.reset()
        control_dt_s = float(env.unwrapped.step_dt)
        os.environ["ARENA_G1_REPLAN_DT_S"] = str(args_cli.replan_steps * control_dt_s)
        _register_gate_n_data_configs()
        policy, step_budget = create_policy(args_cli)
        if not hasattr(policy, "action_chunk_length") or not hasattr(policy, "current_action_chunk"):
            raise TypeError("frequency gate requires the GR00T closed-loop chunk policy")
        action_horizon = int(policy.current_action_chunk.shape[1])
        if args_cli.replan_steps > action_horizon:
            raise ValueError(f"replan_steps={args_cli.replan_steps} exceeds action_horizon={action_horizon}")
        policy.action_chunk_length = int(args_cli.replan_steps)
        reference_first_chunk: np.ndarray | None = None
        if args_cli.first_chunk_reference is not None:
            with np.load(args_cli.first_chunk_reference) as reference_data:
                reference_first_chunk = np.asarray(reference_data["first_action_chunk"], dtype=np.float32)
            expected_shape = tuple(policy.current_action_chunk.shape[1:])
            if reference_first_chunk.shape != expected_shape:
                raise ValueError(
                    f"reference first chunk shape {reference_first_chunk.shape} does not match {expected_shape}"
                )

        unwrapped = env.unwrapped
        robot = unwrapped.scene["robot"]
        action_consumer = make_consumer(args_cli.consumer, control_dt_s)

        root_position: list[np.ndarray] = [_first_env(robot.data.root_pos_w)]
        root_quaternion: list[np.ndarray] = [_first_env(robot.data.root_quat_w)]
        root_linear_velocity: list[np.ndarray] = [_first_env(robot.data.root_lin_vel_w)]
        root_angular_velocity: list[np.ndarray] = [_first_env(robot.data.root_ang_vel_w)]
        navigate_command: list[np.ndarray] = []
        policy_navigate_command: list[np.ndarray] = []
        consumer_active_plans: list[int] = []
        consumer_position_error: list[float] = []
        consumer_yaw_error: list[float] = []
        joint_position: list[np.ndarray] = [
            _latest_first_env_observation(obs["policy"]["robot_joint_pos"], 43)
        ]
        left_wrist_pose: list[np.ndarray] = [
            _first_env_pose_matrix(obs["policy"]["left_wrist_pose_pelvis_frame"])
        ]
        right_wrist_pose: list[np.ndarray] = [
            _first_env_pose_matrix(obs["policy"]["right_wrist_pose_pelvis_frame"])
        ]
        chunk_index: list[int] = []
        new_chunk: list[bool] = []
        inference_wall_s: list[float] = []

        steps_executed = 0
        terminated_once = False
        truncated_once = False
        first_chunk: np.ndarray | None = None
        generated_first_chunk: np.ndarray | None = None
        first_policy_observation_sha256: str | None = None

        for step in tqdm.tqdm(range(step_budget)):
            requires_chunk = bool(policy.env_requires_new_action_chunk[0].item())
            selected_chunk_index = 0 if requires_chunk else int(policy.current_action_index[0].item())
            if requires_chunk and first_chunk is None:
                # Environment creation, reset, and model loading consume global
                # RNG state.  Reset immediately before the first policy call so
                # a paired seed has the same initial stochastic policy draw no
                # matter which replanning interval will be used later.
                # Hash the exact GR00T input contract, including the resized
                # head-camera image, proprioception, and language instruction.
                first_policy_observation_sha256 = nested_observation_sha256(
                    policy.get_observations(obs, policy.policy_config.pov_cam_name_sim)
                )
                _seed_first_policy_inference(int(args_cli.seed))
            started = time.perf_counter()
            with torch.inference_mode():
                actions = policy.get_action(env, obs)
            action_wall_s = time.perf_counter() - started
            if requires_chunk and first_chunk is None:
                generated_first_chunk = _numpy(policy.current_action_chunk[0]).astype(np.float32, copy=True)
                if reference_first_chunk is not None:
                    # CUDA inference is not bitwise reproducible across fresh
                    # simulator processes on this stack.  Execute the reference
                    # condition's first complete chunk so the paired trajectories
                    # branch only when their replanning schedules first differ.
                    policy.current_action_chunk[0] = torch.as_tensor(
                        reference_first_chunk,
                        dtype=policy.current_action_chunk.dtype,
                        device=policy.current_action_chunk.device,
                    )
                    actions = policy.current_action_chunk[:, 0].clone()
                first_chunk = _numpy(policy.current_action_chunk[0]).astype(np.float32, copy=True)

            raw_action_array = _first_env(actions)
            if raw_action_array.shape != (50,):
                raise ValueError(f"expected Arena G1 simulator action shape (50,), got {raw_action_array.shape}")
            root_position_before = _first_env(robot.data.root_pos_w)
            root_quaternion_before = _first_env(robot.data.root_quat_w)
            root_se2_before = np.asarray(
                [
                    root_position_before[0],
                    root_position_before[1],
                    yaw_from_wxyz(root_quaternion_before),
                ],
                dtype=np.float64,
            )
            consumer_diagnostic: dict[str, float | int] = {"active_plans": 1}
            if action_consumer is not None:
                if requires_chunk:
                    action_consumer.add_chunk(
                        _numpy(policy.current_action_chunk[0]),
                        start_step=step,
                        root_se2_w=root_se2_before,
                    )
                consumed_navigate, consumer_diagnostic = action_consumer.command(
                    control_step=step,
                    root_se2_w=root_se2_before,
                    fallback_navigate_command=raw_action_array[43:46],
                )
                actions = actions.clone()
                actions[0, 43:46] = torch.as_tensor(
                    consumed_navigate,
                    dtype=actions.dtype,
                    device=actions.device,
                )

            with torch.inference_mode():
                obs, _, terminated, truncated, _ = env.step(actions)

            action_array = _first_env(actions)
            if action_array.shape != (50,):
                raise ValueError(f"expected Arena G1 simulator action shape (50,), got {action_array.shape}")

            root_position.append(_first_env(robot.data.root_pos_w))
            root_quaternion.append(_first_env(robot.data.root_quat_w))
            root_linear_velocity.append(_first_env(robot.data.root_lin_vel_w))
            root_angular_velocity.append(_first_env(robot.data.root_ang_vel_w))
            navigate_command.append(action_array[43:46])
            policy_navigate_command.append(raw_action_array[43:46])
            consumer_active_plans.append(int(consumer_diagnostic["active_plans"]))
            consumer_position_error.append(float(consumer_diagnostic.get("position_error_m", 0.0)))
            consumer_yaw_error.append(float(consumer_diagnostic.get("yaw_error_rad", 0.0)))
            joint_position.append(_latest_first_env_observation(obs["policy"]["robot_joint_pos"], 43))
            left_wrist_pose.append(_first_env_pose_matrix(obs["policy"]["left_wrist_pose_pelvis_frame"]))
            right_wrist_pose.append(_first_env_pose_matrix(obs["policy"]["right_wrist_pose_pelvis_frame"]))
            chunk_index.append(selected_chunk_index)
            new_chunk.append(requires_chunk)
            inference_wall_s.append(action_wall_s if requires_chunk else 0.0)

            steps_executed = step + 1
            terminated_once = bool(terminated.any().item())
            truncated_once = bool(truncated.any().item())
            if terminated_once or truncated_once:
                break

        from isaaclab_arena.metrics.metrics import compute_metrics

        metrics = compute_metrics(env)
        if terminated_once or truncated_once:
            # ManagerBasedRLEnv auto-resets before returning from the terminal
            # step.  Drop that reset state and its unmatched command rather than
            # misreporting the initial pose as the terminal pose.
            root_position.pop()
            root_quaternion.pop()
            root_linear_velocity.pop()
            root_angular_velocity.pop()
            joint_position.pop()
            left_wrist_pose.pop()
            right_wrist_pose.pop()
            navigate_command.pop()
            policy_navigate_command.pop()
            consumer_active_plans.pop()
            consumer_position_error.pop()
            consumer_yaw_error.pop()
            chunk_index.pop()
            new_chunk.pop()
            inference_wall_s.pop()
        arrays = {
            "root_position_w": np.asarray(root_position, dtype=np.float64),
            "root_quaternion_wxyz": np.asarray(root_quaternion, dtype=np.float64),
            "root_linear_velocity_w": np.asarray(root_linear_velocity, dtype=np.float64),
            "root_angular_velocity_w": np.asarray(root_angular_velocity, dtype=np.float64),
            "navigate_command": np.asarray(navigate_command, dtype=np.float64),
            "policy_navigate_command": np.asarray(policy_navigate_command, dtype=np.float64),
            "consumer_active_plans": np.asarray(consumer_active_plans, dtype=np.int32),
            "consumer_position_error_m": np.asarray(consumer_position_error, dtype=np.float64),
            "consumer_yaw_error_rad": np.asarray(consumer_yaw_error, dtype=np.float64),
            "joint_position": np.asarray(joint_position, dtype=np.float64),
            "left_wrist_pose_pelvis": np.asarray(left_wrist_pose, dtype=np.float64),
            "right_wrist_pose_pelvis": np.asarray(right_wrist_pose, dtype=np.float64),
            "chunk_index": np.asarray(chunk_index, dtype=np.int32),
            "new_chunk": np.asarray(new_chunk, dtype=np.bool_),
            "inference_wall_s": np.asarray(inference_wall_s, dtype=np.float64),
        }
        if first_chunk is None:
            raise RuntimeError("policy did not produce an action chunk")
        if generated_first_chunk is None:
            raise RuntimeError("policy did not generate an initial action chunk")
        if first_policy_observation_sha256 is None:
            raise RuntimeError("policy did not receive an initial observation")

        output_json = args_cli.telemetry_output
        output_npz = output_json.with_suffix(".npz")
        output_json.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output_npz, first_action_chunk=first_chunk, **arrays)

        metric_payload = _jsonable(metrics)
        success_value = metric_payload.get("success_rate", 0.0)
        if isinstance(success_value, list):
            success_value = success_value[0] if success_value else 0.0
        success = bool(float(success_value) > 0.0)

        trajectory_summary = summarize_trajectory(
            arrays["root_position_w"],
            arrays["root_quaternion_wxyz"],
            arrays["root_linear_velocity_w"],
            arrays["navigate_command"],
            arrays["left_wrist_pose_pelvis"],
            arrays["right_wrist_pose_pelvis"],
            control_dt_s,
        )
        report = {
            "schema_version": 1,
            "protocol_id": str(args_cli.protocol_id),
            "consumer_method": str(args_cli.consumer),
            "seed": int(args_cli.seed),
            "replan_steps": int(args_cli.replan_steps),
            "action_horizon": action_horizon,
            "control_dt_s": control_dt_s,
            "control_frequency_hz": 1.0 / control_dt_s,
            "nominal_replan_frequency_hz": 1.0 / (control_dt_s * args_cli.replan_steps),
            "policy_replan_delta_t_s": float(args_cli.replan_steps * control_dt_s),
            "rollout": {
                "steps_executed": steps_executed,
                "step_budget": int(step_budget),
                "terminated": terminated_once,
                "truncated": truncated_once,
                "budget_exhausted": not (terminated_once or truncated_once),
                "success": success,
            },
            "metrics": metric_payload,
            "inference": {
                "calls": int(arrays["new_chunk"].sum()),
                "wall_time_total_s": float(arrays["inference_wall_s"].sum()),
                "wall_time_median_s": float(np.median(arrays["inference_wall_s"][arrays["new_chunk"]])),
            },
            "consumer": {
                "method": str(args_cli.consumer),
                "changes_only_base_navigation_dimensions": bool(args_cli.consumer != "raw"),
                "active_plan_count_median": float(np.median(arrays["consumer_active_plans"])),
                "active_plan_count_max": int(np.max(arrays["consumer_active_plans"])),
                "position_error_m_median": float(np.median(arrays["consumer_position_error_m"])),
                "yaw_error_rad_median": float(np.median(arrays["consumer_yaw_error_rad"])),
            },
            "first_action_chunk_sha256": hashlib.sha256(first_chunk.tobytes()).hexdigest(),
            "generated_first_action_chunk_sha256": hashlib.sha256(
                generated_first_chunk.tobytes()
            ).hexdigest(),
            "first_policy_observation_sha256": first_policy_observation_sha256,
            "first_policy_inference_rng_seed": int(args_cli.seed),
            "first_action_chunk_control": (
                "forced_from_reference_condition"
                if args_cli.first_chunk_reference is not None
                else "generated_reference_condition"
            ),
            "first_action_chunk_reference": (
                str(args_cli.first_chunk_reference) if args_cli.first_chunk_reference is not None else None
            ),
            "telemetry_npz": output_npz.name,
            "arena_metrics_hdf5": "arena-metrics.hdf5",
            "trajectory": trajectory_summary,
        }
        output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"GateNTelemetry: {json.dumps(report, ensure_ascii=False)}")
        env.close()


if __name__ == "__main__":
    main()
