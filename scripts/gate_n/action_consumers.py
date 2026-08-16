"""Non-learned SE(2) base-action consumer for Gate N.

The SE(2) consumer uses a timestamped plan bank, integrates each
predicted base twist into a world-frame pose reference and tracks that reference
with feed-forward plus proportional feedback.  Upper-body position targets are
never changed.  The timestamped bank is an implementation detail of waypoint
tracking; Gate N's delta-t baseline conditions the policy itself.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


def wrap_to_pi(angle: float | np.ndarray) -> float | np.ndarray:
    """Wrap radians to ``[-pi, pi)``."""
    return (np.asarray(angle) + math.pi) % (2.0 * math.pi) - math.pi


@dataclass(frozen=True)
class TimestampedPlan:
    start_step: int
    navigate_command: np.ndarray
    se2_reference_w: np.ndarray | None = None


class TimestampedPlanBank:
    """Fuse overlapping base predictions at equal physical times.

    A plan's weight decays with its physical age rather than its list index, so
    the rule itself is independent of how frequently the policy is called.
    """

    def __init__(self, control_dt_s: float, decay_time_s: float = 0.32) -> None:
        if control_dt_s <= 0.0:
            raise ValueError("control_dt_s must be positive")
        if decay_time_s <= 0.0:
            raise ValueError("decay_time_s must be positive")
        self.control_dt_s = float(control_dt_s)
        self.decay_time_s = float(decay_time_s)
        self._plans: list[TimestampedPlan] = []

    @property
    def active_plan_count(self) -> int:
        return len(self._plans)

    def add_chunk(self, action_chunk: np.ndarray, start_step: int, root_se2_w: np.ndarray) -> None:
        del root_se2_w
        chunk = np.asarray(action_chunk, dtype=np.float64)
        if chunk.ndim != 2 or chunk.shape[1] < 46:
            raise ValueError(f"expected an [H, >=46] simulator action chunk, got {chunk.shape}")
        self._plans.append(
            TimestampedPlan(
                start_step=int(start_step),
                navigate_command=chunk[:, 43:46].copy(),
            )
        )

    def _aligned(self, control_step: int) -> tuple[list[TimestampedPlan], np.ndarray, np.ndarray]:
        step = int(control_step)
        self._plans = [
            plan
            for plan in self._plans
            if plan.start_step <= step < plan.start_step + len(plan.navigate_command)
        ]
        if not self._plans:
            raise RuntimeError(f"no timestamped plan covers control step {step}")
        offsets = np.asarray([step - plan.start_step for plan in self._plans], dtype=np.int64)
        ages_s = offsets.astype(np.float64) * self.control_dt_s
        weights = np.exp(-ages_s / self.decay_time_s)
        weights /= weights.sum()
        return self._plans, offsets, weights

    def command(
        self,
        control_step: int,
        root_se2_w: np.ndarray,
        fallback_navigate_command: np.ndarray,
    ) -> tuple[np.ndarray, dict[str, float | int]]:
        del root_se2_w, fallback_navigate_command
        plans, offsets, weights = self._aligned(control_step)
        commands = np.stack(
            [plan.navigate_command[offset] for plan, offset in zip(plans, offsets, strict=True)]
        )
        command = np.average(commands, axis=0, weights=weights)
        return command, {
            "active_plans": len(plans),
            "oldest_plan_age_s": float(offsets.max() * self.control_dt_s),
            "newest_weight": float(weights[-1]),
        }


class SE2WaypointConsumer(TimestampedPlanBank):
    """Track timestamp-aligned, current-relative SE(2) references."""

    def __init__(
        self,
        control_dt_s: float,
        decay_time_s: float = 0.32,
        kp_xy_per_s: float = 1.0,
        kp_yaw_per_s: float = 1.0,
        max_abs_command: tuple[float, float, float] = (0.5, 0.5, 0.5),
    ) -> None:
        super().__init__(control_dt_s=control_dt_s, decay_time_s=decay_time_s)
        if kp_xy_per_s < 0.0 or kp_yaw_per_s < 0.0:
            raise ValueError("tracker gains must be non-negative")
        limits = np.asarray(max_abs_command, dtype=np.float64)
        if limits.shape != (3,) or np.any(limits <= 0.0):
            raise ValueError("max_abs_command must contain three positive values")
        self.kp_xy_per_s = float(kp_xy_per_s)
        self.kp_yaw_per_s = float(kp_yaw_per_s)
        self.max_abs_command = limits

    def add_chunk(self, action_chunk: np.ndarray, start_step: int, root_se2_w: np.ndarray) -> None:
        chunk = np.asarray(action_chunk, dtype=np.float64)
        if chunk.ndim != 2 or chunk.shape[1] < 46:
            raise ValueError(f"expected an [H, >=46] simulator action chunk, got {chunk.shape}")
        navigate = chunk[:, 43:46].copy()
        pose = np.asarray(root_se2_w, dtype=np.float64).copy()
        if pose.shape != (3,):
            raise ValueError(f"expected root_se2_w shape (3,), got {pose.shape}")

        # references[k] is the desired pose at the start of slot k.  The
        # corresponding twist is feed-forward for the interval [k, k + 1].
        references = np.empty((len(navigate), 3), dtype=np.float64)
        for index, (vx, vy, yaw_rate) in enumerate(navigate):
            references[index] = pose
            c, s = math.cos(pose[2]), math.sin(pose[2])
            pose[0] += (c * vx - s * vy) * self.control_dt_s
            pose[1] += (s * vx + c * vy) * self.control_dt_s
            pose[2] = float(wrap_to_pi(pose[2] + yaw_rate * self.control_dt_s))

        self._plans.append(
            TimestampedPlan(
                start_step=int(start_step),
                navigate_command=navigate,
                se2_reference_w=references,
            )
        )

    def command(
        self,
        control_step: int,
        root_se2_w: np.ndarray,
        fallback_navigate_command: np.ndarray,
    ) -> tuple[np.ndarray, dict[str, float | int]]:
        del fallback_navigate_command
        plans, offsets, weights = self._aligned(control_step)
        references = np.stack(
            [plan.se2_reference_w[offset] for plan, offset in zip(plans, offsets, strict=True)]
        )
        feedforward = np.stack(
            [plan.navigate_command[offset] for plan, offset in zip(plans, offsets, strict=True)]
        )
        target_xy = np.average(references[:, :2], axis=0, weights=weights)
        target_yaw = math.atan2(
            float(np.sum(weights * np.sin(references[:, 2]))),
            float(np.sum(weights * np.cos(references[:, 2]))),
        )
        ff = np.average(feedforward, axis=0, weights=weights)

        current = np.asarray(root_se2_w, dtype=np.float64)
        delta_w = target_xy - current[:2]
        c, s = math.cos(current[2]), math.sin(current[2])
        delta_body = np.asarray(
            [c * delta_w[0] + s * delta_w[1], -s * delta_w[0] + c * delta_w[1]],
            dtype=np.float64,
        )
        yaw_error = float(wrap_to_pi(target_yaw - current[2]))
        command = ff.copy()
        command[:2] += self.kp_xy_per_s * delta_body
        command[2] += self.kp_yaw_per_s * yaw_error
        command = np.clip(command, -self.max_abs_command, self.max_abs_command)
        return command, {
            "active_plans": len(plans),
            "oldest_plan_age_s": float(offsets.max() * self.control_dt_s),
            "newest_weight": float(weights[-1]),
            "position_error_m": float(np.linalg.norm(delta_w)),
            "yaw_error_rad": float(abs(yaw_error)),
        }


def make_consumer(method: str, control_dt_s: float) -> SE2WaypointConsumer | None:
    if method == "raw":
        return None
    if method == "se2_waypoint":
        return SE2WaypointConsumer(control_dt_s=control_dt_s)
    raise ValueError(f"unknown consumer method: {method}")
