"""Arena G1 bridge for executing native Phase 1 M1 plans.

Arena's closed-loop policy expands a 32-D GR00T action into 50 simulator
values: 43 joint slots, three navigation values, one base-height command and
three torso-RPY commands.  In M1 the navigation slots temporarily carry an
anchor-relative SE(2) pose.  This bridge extracts the full M1 plan, tracks it
in real elapsed time, and writes executable velocity commands back into the
50-D tensor while leaving the lower-body/WBC contract unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .phase1 import ACTION_DIM, Phase1ExecutionAdapter, WholeBodyPlan


SIM_JOINT_DIM = 43
SIM_NAVIGATE_SLICE = slice(43, 46)
SIM_BASE_HEIGHT_INDEX = 46
SIM_ACTION_DIM = 50
M1_UPPER_OUTPUT_KEY = "action.phase1_upper_body_position"
M1_HEIGHT_OUTPUT_KEY = "action.phase1_base_height"
M1_BASE_OUTPUT_KEY = "action.phase1_base_relative_se2"


def ordered_upper_sim_indices(
    policy_joint_groups: dict[str, Sequence[str]],
    simulator_joint_indices: dict[str, int],
) -> np.ndarray:
    """Return simulator indices in the frozen M1 28-D upper-body order."""
    names: list[str] = []
    for group in ("left_arm", "right_arm", "left_hand", "right_hand"):
        if group not in policy_joint_groups:
            raise KeyError(f"missing policy joint group {group!r}")
        names.extend(policy_joint_groups[group])
    if len(names) != 28 or len(set(names)) != 28:
        raise ValueError(f"expected 28 unique upper-body joints, got {len(names)}")
    missing = [name for name in names if name not in simulator_joint_indices]
    if missing:
        raise KeyError(f"upper-body joints missing from simulator mapping: {missing}")
    indices = np.asarray([simulator_joint_indices[name] for name in names], dtype=np.int64)
    if np.any(indices < 0) or np.any(indices >= SIM_JOINT_DIM):
        raise ValueError(f"invalid simulator upper-body indices: {indices.tolist()}")
    return indices


def decode_m1_policy_output_to_simulator_chunk(
    policy_output: dict[str, np.ndarray],
    upper_sim_indices: np.ndarray,
) -> np.ndarray:
    """Map decoded M1 keys into Arena's 50-D chunk without changing semantics."""
    required = (M1_UPPER_OUTPUT_KEY, M1_HEIGHT_OUTPUT_KEY, M1_BASE_OUTPUT_KEY)
    missing = [key for key in required if key not in policy_output]
    if missing:
        raise KeyError(f"M1 policy output is missing keys: {missing}")
    upper = np.asarray(policy_output[M1_UPPER_OUTPUT_KEY], dtype=np.float32)
    height = np.asarray(policy_output[M1_HEIGHT_OUTPUT_KEY], dtype=np.float32)
    base = np.asarray(policy_output[M1_BASE_OUTPUT_KEY], dtype=np.float32)
    if upper.ndim != 3 or upper.shape[-1] != 28:
        raise ValueError(f"M1 upper output must have shape (B, H, 28), got {upper.shape}")
    if height.shape != (*upper.shape[:2], 1):
        raise ValueError(f"M1 height output must have shape (B, H, 1), got {height.shape}")
    if base.shape != (*upper.shape[:2], 3):
        raise ValueError(f"M1 base output must have shape (B, H, 3), got {base.shape}")
    indices = np.asarray(upper_sim_indices, dtype=np.int64)
    if indices.shape != (28,) or len(np.unique(indices)) != 28:
        raise ValueError("upper_sim_indices must contain 28 unique entries")
    chunk = np.zeros((*upper.shape[:2], SIM_ACTION_DIM), dtype=np.float32)
    chunk[..., indices] = upper
    chunk[..., SIM_NAVIGATE_SLICE] = base
    chunk[..., SIM_BASE_HEIGHT_INDEX] = height[..., 0]
    return chunk


@dataclass(frozen=True)
class PlanActivationDiagnostics:
    plan_id: int
    position_jump_m: float
    yaw_jump_rad: float
    upper_position_jump_norm: float
    base_velocity_jump_norm: float


class ArenaM1PlanExecutor:
    """Track one latest-wins M1 plan and emit Arena-compatible actions."""

    def __init__(
        self,
        *,
        upper_sim_indices: np.ndarray,
        query_times_s: np.ndarray,
        kp_xy_per_s: float = 1.0,
        kp_yaw_per_s: float = 1.0,
        max_abs_base_twist: tuple[float, float, float] = (0.5, 0.5, 0.5),
    ) -> None:
        indices = np.asarray(upper_sim_indices, dtype=np.int64)
        if indices.shape != (28,) or len(np.unique(indices)) != 28:
            raise ValueError("upper_sim_indices must contain 28 unique entries")
        if np.any(indices < 0) or np.any(indices >= SIM_JOINT_DIM):
            raise ValueError("upper_sim_indices contains an out-of-range joint index")
        query = np.asarray(query_times_s, dtype=np.float64)
        if query.ndim != 1 or len(query) == 0:
            raise ValueError("query_times_s must be a non-empty vector")
        self.upper_sim_indices = indices
        self.query_times_s = query
        self.tracker = Phase1ExecutionAdapter(
            kp_xy_per_s=kp_xy_per_s,
            kp_yaw_per_s=kp_yaw_per_s,
            max_abs_base_twist=max_abs_base_twist,
        )
        self.plan: WholeBodyPlan | None = None
        self.activation_monotonic_s: float | None = None
        self.plan_id = -1
        self._last_command = np.zeros(ACTION_DIM, dtype=np.float64)

    def _model_targets(self, simulator_chunk: np.ndarray) -> np.ndarray:
        chunk = np.asarray(simulator_chunk, dtype=np.float64)
        expected = (len(self.query_times_s), SIM_ACTION_DIM)
        if chunk.shape != expected:
            raise ValueError(f"simulator_chunk must have shape {expected}, got {chunk.shape}")
        if not np.all(np.isfinite(chunk)):
            raise ValueError("simulator_chunk contains non-finite values")
        targets = np.empty((len(chunk), ACTION_DIM), dtype=np.float64)
        targets[:, :28] = chunk[:, self.upper_sim_indices]
        targets[:, 28] = chunk[:, SIM_BASE_HEIGHT_INDEX]
        targets[:, 29:32] = chunk[:, SIM_NAVIGATE_SLICE]
        return targets

    def activate(
        self,
        simulator_chunk: np.ndarray,
        *,
        measured_sim_joint_position: np.ndarray,
        measured_base_pose_se2_w: np.ndarray,
        current_base_height_command: float,
        activation_monotonic_s: float,
    ) -> PlanActivationDiagnostics:
        joints = np.asarray(measured_sim_joint_position, dtype=np.float64)
        root = np.asarray(measured_base_pose_se2_w, dtype=np.float64)
        if joints.shape != (SIM_JOINT_DIM,):
            raise ValueError(f"measured_sim_joint_position must have shape (43,), got {joints.shape}")
        if root.shape != (3,):
            raise ValueError(f"measured_base_pose_se2_w must have shape (3,), got {root.shape}")
        if not np.isfinite(current_base_height_command):
            raise ValueError("current_base_height_command must be finite")
        activation = float(activation_monotonic_s)
        if not np.isfinite(activation):
            raise ValueError("activation_monotonic_s must be finite")

        old_ref = None
        if self.plan is not None and self.activation_monotonic_s is not None:
            old_ref = self.plan.sample(max(0.0, activation - self.activation_monotonic_s))
        new_plan = WholeBodyPlan(
            query_times_s=self.query_times_s.copy(),
            targets=self._model_targets(simulator_chunk),
            anchor_base_pose_se2_w=root.copy(),
            anchor_upper_body_position=joints[self.upper_sim_indices].copy(),
            anchor_base_height=float(current_base_height_command),
        )
        new_ref = new_plan.sample(0.0)
        self.plan = new_plan
        self.activation_monotonic_s = activation
        self.plan_id += 1

        if old_ref is None:
            position_jump = yaw_jump = upper_jump = velocity_jump = 0.0
        else:
            position_jump = float(np.linalg.norm(new_ref.base_pose_se2_w[:2] - old_ref.base_pose_se2_w[:2]))
            yaw_jump = float(
                abs((new_ref.base_pose_se2_w[2] - old_ref.base_pose_se2_w[2] + np.pi) % (2 * np.pi) - np.pi)
            )
            upper_jump = float(np.linalg.norm(new_ref.upper_body_position - old_ref.upper_body_position))
            velocity_jump = float(np.linalg.norm(new_ref.base_twist_body_ff - old_ref.base_twist_body_ff))
        return PlanActivationDiagnostics(
            plan_id=self.plan_id,
            position_jump_m=position_jump,
            yaw_jump_rad=yaw_jump,
            upper_position_jump_norm=upper_jump,
            base_velocity_jump_norm=velocity_jump,
        )

    def command(
        self,
        simulator_action_template: np.ndarray,
        *,
        measured_base_pose_se2_w: np.ndarray,
        monotonic_s: float,
    ) -> tuple[np.ndarray, dict[str, float | bool | int]]:
        if self.plan is None or self.activation_monotonic_s is None:
            raise RuntimeError("no M1 plan has been activated")
        template = np.asarray(simulator_action_template, dtype=np.float64)
        if template.shape != (SIM_ACTION_DIM,):
            raise ValueError(f"simulator_action_template must have shape (50,), got {template.shape}")
        elapsed = max(0.0, float(monotonic_s) - self.activation_monotonic_s)
        command32, diagnostic = self.tracker.command(
            self.plan,
            elapsed_s=elapsed,
            measured_base_pose_se2_w=measured_base_pose_se2_w,
        )
        output = template.copy()
        output[self.upper_sim_indices] = command32[:28]
        output[SIM_NAVIGATE_SLICE] = command32[29:32]
        output[SIM_BASE_HEIGHT_INDEX] = command32[28]
        interval = int(np.searchsorted(self.query_times_s, elapsed, side="right"))
        diagnostic = {
            **diagnostic,
            "plan_id": self.plan_id,
            "plan_age_s": elapsed,
            "query_interval_index": min(interval, len(self.query_times_s)),
            "executed_vx": float(command32[29]),
            "executed_vy": float(command32[30]),
            "executed_yaw_rate": float(command32[31]),
        }
        self._last_command = command32
        return output, diagnostic
