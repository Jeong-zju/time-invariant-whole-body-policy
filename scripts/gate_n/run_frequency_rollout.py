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
import sys
import time
import types
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch
import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
from isaaclab_arena.examples.example_environments.cli import get_arena_builder_from_cli
from isaaclab_arena.examples.policy_runner_cli import create_policy, setup_policy_argument_parser
from isaaclab_arena.utils.isaaclab_utils.simulation_app import SimulationAppContext

from action_consumers import make_consumer
from frequency_metrics import nested_observation_sha256, summarize_trajectory, yaw_from_wxyz
from replan_schedule import ReplanSchedule
from whole_body_policy import (
    ArenaM1PlanExecutor,
    decode_m1_policy_output_to_simulator_chunk,
    ordered_upper_sim_indices,
)


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
    from whole_body_policy.groot_m1_data_config import UnitreeG1Phase1M1DataConfig

    DATA_CONFIG_MAP["unitree_g1_phase1_m1"] = UnitreeG1Phase1M1DataConfig()


def _install_m1_chunk_decoder(policy: Any, upper_sim_indices: np.ndarray) -> None:
    """Replace Arena's legacy action-key decoder for the M1 output contract."""

    def get_m1_action_chunk(self: Any, observation: dict[str, Any], camera_name: str) -> torch.Tensor:
        policy_observations = self.get_observations(observation, camera_name)
        decoded = self.policy.get_action(policy_observations)
        chunk = decode_m1_policy_output_to_simulator_chunk(decoded, upper_sim_indices)
        return torch.as_tensor(chunk, dtype=torch.float32, device=self.device)

    policy.get_action_chunk = types.MethodType(get_m1_action_chunk, policy)


def main() -> None:
    args_parser = get_isaaclab_arena_cli_parser()
    args_parser.add_argument("--telemetry_output", type=Path, required=True)
    schedule_group = args_parser.add_mutually_exclusive_group(required=True)
    schedule_group.add_argument("--replan_steps", type=int)
    schedule_group.add_argument("--replan_frequency_hz", type=float)
    schedule_group.add_argument(
        "--jitter_frequency_hz",
        type=float,
        nargs=2,
        metavar=("MIN_HZ", "MAX_HZ"),
    )
    args_parser.add_argument("--first_chunk_reference", type=Path)
    args_parser.add_argument(
        "--consumer",
        choices=("raw", "se2_waypoint", "phase1_m1"),
        default="raw",
    )
    args_parser.add_argument("--protocol_id", default=PROTOCOL_ID)
    args_cli, _ = args_parser.parse_known_args()

    with SimulationAppContext(args_cli):
        args_parser = setup_policy_argument_parser(args_parser)
        args_cli = args_parser.parse_args()
        if args_cli.replan_steps is not None and args_cli.replan_steps <= 0:
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
        initial_replan_steps = args_cli.replan_steps
        if initial_replan_steps is None:
            requested_hz = (
                args_cli.replan_frequency_hz
                if args_cli.replan_frequency_hz is not None
                else args_cli.jitter_frequency_hz[1]
            )
            initial_replan_steps = max(1, int(round(1.0 / (requested_hz * control_dt_s))))
        os.environ["ARENA_G1_REPLAN_DT_S"] = str(initial_replan_steps * control_dt_s)
        _register_gate_n_data_configs()
        policy, step_budget = create_policy(args_cli)
        if not hasattr(policy, "action_chunk_length") or not hasattr(policy, "current_action_chunk"):
            raise TypeError("frequency gate requires the GR00T closed-loop chunk policy")
        action_horizon = int(policy.current_action_chunk.shape[1])
        scheduler = ReplanSchedule(
            control_dt_s=control_dt_s,
            fixed_steps=args_cli.replan_steps,
            fixed_frequency_hz=args_cli.replan_frequency_hz,
            jitter_frequency_hz=(
                tuple(args_cli.jitter_frequency_hz)
                if args_cli.jitter_frequency_hz is not None
                else None
            ),
            seed=int(args_cli.seed),
        )
        if initial_replan_steps > action_horizon:
            raise ValueError(
                f"initial replan interval={initial_replan_steps} exceeds action_horizon={action_horizon}"
            )
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
        action_consumer = (
            None if args_cli.consumer == "phase1_m1" else make_consumer(args_cli.consumer, control_dt_s)
        )
        m1_executor = None
        if args_cli.consumer == "phase1_m1":
            m1_upper_indices = ordered_upper_sim_indices(
                policy.policy_joints_config,
                policy.robot_action_joints_config,
            )
            _install_m1_chunk_decoder(policy, m1_upper_indices)
            m1_executor = ArenaM1PlanExecutor(
                upper_sim_indices=m1_upper_indices,
                query_times_s=np.arange(1, 17, dtype=np.float64) * control_dt_s,
                kp_xy_per_s=1.0,
                kp_yaw_per_s=1.0,
                max_abs_base_twist=(0.5, 0.5, 0.5),
            )

        root_position: list[np.ndarray] = [_first_env(robot.data.root_pos_w)]
        root_quaternion: list[np.ndarray] = [_first_env(robot.data.root_quat_w)]
        root_linear_velocity: list[np.ndarray] = [_first_env(robot.data.root_lin_vel_w)]
        root_angular_velocity: list[np.ndarray] = [_first_env(robot.data.root_ang_vel_w)]
        navigate_command: list[np.ndarray] = []
        policy_navigate_command: list[np.ndarray] = []
        consumer_active_plans: list[int] = []
        consumer_position_error: list[float] = []
        consumer_yaw_error: list[float] = []
        plan_age_s: list[float] = []
        query_interval_index: list[int] = []
        clamped_to_horizon: list[bool] = []
        replan_position_jump_m: list[float] = []
        replan_yaw_jump_rad: list[float] = []
        replan_upper_position_jump_norm: list[float] = []
        replan_base_velocity_jump_norm: list[float] = []
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
        scheduled_replan_steps: list[int] = []

        steps_executed = 0
        terminated_once = False
        truncated_once = False
        first_chunk: np.ndarray | None = None
        generated_first_chunk: np.ndarray | None = None
        first_policy_observation_sha256: str | None = None
        active_replan_steps = int(initial_replan_steps)
        # Arena does not expose ``base_height_cmd`` as its own observation
        # group.  Track the command actually sent to the WBC instead.  The
        # first plan has no previous command, so anchor it at the first
        # decoded height target; later replans anchor at the last executed
        # command and therefore remain continuous.
        last_executed_base_height_command: float | None = None

        for step in tqdm.tqdm(range(step_budget)):
            requires_chunk = bool(policy.env_requires_new_action_chunk[0].item())
            if requires_chunk:
                active_replan_steps = scheduler.next_steps()
                if active_replan_steps > action_horizon:
                    raise ValueError(
                        f"scheduled replan interval={active_replan_steps} exceeds action_horizon={action_horizon}"
                    )
                policy.action_chunk_length = active_replan_steps
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
            raw_action_array = _first_env(actions)
            if raw_action_array.shape != (50,):
                raise ValueError(f"expected Arena G1 simulator action shape (50,), got {raw_action_array.shape}")
            consumer_diagnostic: dict[str, float | int] = {"active_plans": 1}
            activation_diagnostic = None
            if m1_executor is not None:
                if requires_chunk:
                    current_joint_position = _latest_first_env_observation(
                        obs["policy"]["robot_joint_pos"], 43
                    )
                    current_base_height = (
                        float(raw_action_array[46])
                        if last_executed_base_height_command is None
                        else last_executed_base_height_command
                    )
                    activation_diagnostic = m1_executor.activate(
                        _numpy(policy.current_action_chunk[0]),
                        measured_sim_joint_position=current_joint_position,
                        measured_base_pose_se2_w=root_se2_before,
                        current_base_height_command=float(current_base_height),
                        activation_monotonic_s=time.monotonic(),
                    )
                executed_action, consumer_diagnostic = m1_executor.command(
                    raw_action_array,
                    measured_base_pose_se2_w=root_se2_before,
                    monotonic_s=time.monotonic(),
                )
                actions = actions.clone()
                actions[0] = torch.as_tensor(
                    executed_action,
                    dtype=actions.dtype,
                    device=actions.device,
                )
                last_executed_base_height_command = float(executed_action[46])
            elif action_consumer is not None:
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
            plan_age_s.append(float(consumer_diagnostic.get("plan_age_s", 0.0)))
            query_interval_index.append(int(consumer_diagnostic.get("query_interval_index", 0)))
            clamped_to_horizon.append(bool(consumer_diagnostic.get("clamped_to_horizon", False)))
            replan_position_jump_m.append(
                float(activation_diagnostic.position_jump_m) if activation_diagnostic is not None else 0.0
            )
            replan_yaw_jump_rad.append(
                float(activation_diagnostic.yaw_jump_rad) if activation_diagnostic is not None else 0.0
            )
            replan_upper_position_jump_norm.append(
                float(activation_diagnostic.upper_position_jump_norm)
                if activation_diagnostic is not None
                else 0.0
            )
            replan_base_velocity_jump_norm.append(
                float(activation_diagnostic.base_velocity_jump_norm)
                if activation_diagnostic is not None
                else 0.0
            )
            joint_position.append(_latest_first_env_observation(obs["policy"]["robot_joint_pos"], 43))
            left_wrist_pose.append(_first_env_pose_matrix(obs["policy"]["left_wrist_pose_pelvis_frame"]))
            right_wrist_pose.append(_first_env_pose_matrix(obs["policy"]["right_wrist_pose_pelvis_frame"]))
            chunk_index.append(selected_chunk_index)
            new_chunk.append(requires_chunk)
            inference_wall_s.append(action_wall_s if requires_chunk else 0.0)
            scheduled_replan_steps.append(active_replan_steps)

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
            plan_age_s.pop()
            query_interval_index.pop()
            clamped_to_horizon.pop()
            replan_position_jump_m.pop()
            replan_yaw_jump_rad.pop()
            replan_upper_position_jump_norm.pop()
            replan_base_velocity_jump_norm.pop()
            chunk_index.pop()
            new_chunk.pop()
            inference_wall_s.pop()
            scheduled_replan_steps.pop()
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
            "plan_age_s": np.asarray(plan_age_s, dtype=np.float64),
            "query_interval_index": np.asarray(query_interval_index, dtype=np.int32),
            "clamped_to_horizon": np.asarray(clamped_to_horizon, dtype=np.bool_),
            "replan_position_jump_m": np.asarray(replan_position_jump_m, dtype=np.float64),
            "replan_yaw_jump_rad": np.asarray(replan_yaw_jump_rad, dtype=np.float64),
            "replan_upper_position_jump_norm": np.asarray(
                replan_upper_position_jump_norm, dtype=np.float64
            ),
            "replan_base_velocity_jump_norm": np.asarray(
                replan_base_velocity_jump_norm, dtype=np.float64
            ),
            "joint_position": np.asarray(joint_position, dtype=np.float64),
            "left_wrist_pose_pelvis": np.asarray(left_wrist_pose, dtype=np.float64),
            "right_wrist_pose_pelvis": np.asarray(right_wrist_pose, dtype=np.float64),
            "chunk_index": np.asarray(chunk_index, dtype=np.int32),
            "new_chunk": np.asarray(new_chunk, dtype=np.bool_),
            "inference_wall_s": np.asarray(inference_wall_s, dtype=np.float64),
            "scheduled_replan_steps": np.asarray(scheduled_replan_steps, dtype=np.int32),
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
        activation_steps = np.flatnonzero(arrays["new_chunk"])
        if len(activation_steps) >= 2:
            realized_replan_frequency_hz = float(
                (len(activation_steps) - 1)
                / ((activation_steps[-1] - activation_steps[0]) * control_dt_s)
            )
        else:
            realized_replan_frequency_hz = 0.0
        report = {
            "schema_version": 1,
            "protocol_id": str(args_cli.protocol_id),
            "consumer_method": str(args_cli.consumer),
            "seed": int(args_cli.seed),
            "replan_steps": int(args_cli.replan_steps) if args_cli.replan_steps is not None else None,
            "action_horizon": action_horizon,
            "control_dt_s": control_dt_s,
            "control_frequency_hz": 1.0 / control_dt_s,
            "nominal_replan_frequency_hz": (
                1.0 / (control_dt_s * args_cli.replan_steps)
                if args_cli.replan_steps is not None
                else args_cli.replan_frequency_hz
            ),
            "policy_replan_delta_t_s": (
                float(args_cli.replan_steps * control_dt_s)
                if args_cli.replan_steps is not None
                else None
            ),
            "replan_schedule": {
                **scheduler.description(),
                "realized_calls": int(arrays["new_chunk"].sum()),
                "realized_frequency_hz": realized_replan_frequency_hz,
                "realized_interval_steps_min": int(arrays["scheduled_replan_steps"].min()),
                "realized_interval_steps_max": int(arrays["scheduled_replan_steps"].max()),
            },
            "rollout": {
                "steps_executed": steps_executed,
                "step_budget": int(step_budget),
                "terminated": terminated_once,
                "truncated": truncated_once,
                "budget_exhausted": not (terminated_once or truncated_once),
                "success": success,
            },
            "metrics": metric_payload,
            "safety": {
                "telemetry_all_finite": bool(
                    all(np.all(np.isfinite(value)) for value in arrays.values())
                ),
                "base_twist_abs_max": [
                    float(value) for value in np.max(np.abs(arrays["navigate_command"]), axis=0)
                ],
                "base_twist_cap": [0.5, 0.5, 0.5],
                "base_twist_cap_violations": int(
                    np.count_nonzero(np.abs(arrays["navigate_command"]) > 0.500001)
                ),
                "unsuccessful_early_termination": bool(terminated_once and not success),
            },
            "inference": {
                "calls": int(arrays["new_chunk"].sum()),
                "wall_time_total_s": float(arrays["inference_wall_s"].sum()),
                "wall_time_median_s": float(np.median(arrays["inference_wall_s"][arrays["new_chunk"]])),
            },
            "consumer": {
                "method": str(args_cli.consumer),
                "changes_only_base_navigation_dimensions": bool(args_cli.consumer == "se2_waypoint"),
                "native_m1_whole_body_tracker": bool(args_cli.consumer == "phase1_m1"),
                "active_plan_count_median": float(np.median(arrays["consumer_active_plans"])),
                "active_plan_count_max": int(np.max(arrays["consumer_active_plans"])),
                "position_error_m_median": float(np.median(arrays["consumer_position_error_m"])),
                "yaw_error_rad_median": float(np.median(arrays["consumer_yaw_error_rad"])),
                "plan_age_s_median": float(np.median(arrays["plan_age_s"])),
                "plan_age_s_max": float(np.max(arrays["plan_age_s"])),
                "clamped_to_horizon_steps": int(arrays["clamped_to_horizon"].sum()),
                "replan_position_jump_m_max": float(np.max(arrays["replan_position_jump_m"])),
                "replan_velocity_jump_norm_max": float(
                    np.max(arrays["replan_base_velocity_jump_norm"])
                ),
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
