from __future__ import annotations

import math
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from whole_body_policy import (  # noqa: E402
    Phase1ExecutionAdapter,
    WholeBodyPlan,
    build_multi_horizon_targets,
    compare_base_proxy_to_odometry,
    compose,
    exp,
    integrate_body_twist,
    interpolate,
    inverse,
    load_target_batch,
    log,
    relative,
    save_target_batch,
    wrap_angle,
)


class SE2Test(unittest.TestCase):
    def test_compose_inverse_round_trip(self) -> None:
        pose = np.asarray([1.3, -0.7, 2.9])
        np.testing.assert_allclose(compose(inverse(pose), pose), np.zeros(3), atol=1e-12)

    def test_exp_log_round_trip(self) -> None:
        for tangent in (
            np.asarray([0.2, -0.3, 0.0]),
            np.asarray([0.2, -0.3, 0.8]),
            np.asarray([-0.1, 0.4, -2.8]),
        ):
            np.testing.assert_allclose(log(exp(tangent)), tangent, atol=1e-12)

    def test_relative_uses_origin_body_frame(self) -> None:
        origin = np.asarray([1.0, 2.0, math.pi / 2.0])
        target = np.asarray([1.0, 3.0, math.pi / 2.0])
        np.testing.assert_allclose(relative(origin, target), [1.0, 0.0, 0.0], atol=1e-12)

    def test_interpolation_crosses_pi_by_short_arc(self) -> None:
        start = np.asarray([0.0, 0.0, math.pi - 0.1])
        end = np.asarray([0.0, 0.0, -math.pi + 0.1])
        midpoint = interpolate(start, end, 0.5)
        self.assertAlmostEqual(abs(float(wrap_angle(midpoint[2]))), math.pi)


class TargetBuilderTest(unittest.TestCase):
    def setUp(self) -> None:
        self.timestamps = np.arange(0.0, 0.42, 0.02)
        self.upper = np.repeat(self.timestamps[:, None], 28, axis=1)
        self.height = 0.75 + self.timestamps
        self.twist = np.zeros((len(self.timestamps), 3))
        self.twist[:, 0] = 1.0

    def test_proxy_targets_use_real_query_seconds(self) -> None:
        batch = build_multi_horizon_targets(
            timestamps_s=self.timestamps,
            upper_body_position=self.upper,
            base_height=self.height,
            query_times_s=np.asarray([0.02, 0.10, 0.32]),
            base_twist_body=self.twist,
            source="command_integration_proxy",
        )
        self.assertEqual(batch.targets.shape, (5, 3, 32))
        np.testing.assert_allclose(batch.targets[0, :, 0], [0.02, 0.10, 0.32])
        np.testing.assert_allclose(batch.targets[0, :, 29], [0.02, 0.10, 0.32], atol=1e-12)
        self.assertEqual(batch.source, "command_integration_proxy")

    def test_last_anchors_are_removed_instead_of_padded(self) -> None:
        batch = build_multi_horizon_targets(
            timestamps_s=self.timestamps,
            upper_body_position=self.upper,
            base_height=self.height,
            query_times_s=np.asarray([0.10]),
            base_twist_body=self.twist,
            source="command_integration_proxy",
        )
        self.assertEqual(batch.anchor_indices[-1], len(self.timestamps) - 6)

    def test_measured_odometry_interpolates_yaw_on_group(self) -> None:
        pose = np.zeros((len(self.timestamps), 3))
        pose[:, 2] = np.linspace(math.pi - 0.2, math.pi + 0.2, len(self.timestamps))
        pose[:, 2] = wrap_angle(pose[:, 2])
        batch = build_multi_horizon_targets(
            timestamps_s=self.timestamps,
            upper_body_position=self.upper,
            base_height=self.height,
            query_times_s=np.asarray([0.03]),
            base_pose_se2=pose,
            source="measured_odometry",
        )
        self.assertAlmostEqual(batch.targets[0, 0, 31], 0.03, places=12)

    def test_source_is_explicit_and_exclusive(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires only"):
            build_multi_horizon_targets(
                timestamps_s=self.timestamps,
                upper_body_position=self.upper,
                base_height=self.height,
                query_times_s=np.asarray([0.02]),
                base_pose_se2=np.zeros((len(self.timestamps), 3)),
                base_twist_body=self.twist,
                source="measured_odometry",
            )

    def test_npz_round_trip_preserves_provenance(self) -> None:
        batch = build_multi_horizon_targets(
            timestamps_s=self.timestamps,
            upper_body_position=self.upper,
            base_height=self.height,
            query_times_s=np.asarray([0.02, 0.04]),
            base_twist_body=self.twist,
            source="command_integration_proxy",
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "targets.npz"
            save_target_batch(path, batch)
            restored = load_target_batch(path)
        np.testing.assert_allclose(restored.targets, batch.targets, atol=1e-7)
        self.assertEqual(restored.source, batch.source)

    def test_proxy_gate_is_zero_for_matching_odometry(self) -> None:
        measured = integrate_body_twist(self.timestamps, self.twist)
        report = compare_base_proxy_to_odometry(
            timestamps_s=self.timestamps,
            body_twist=self.twist,
            measured_base_pose_se2=measured,
            horizon_s=0.32,
        )
        self.assertGreater(report["windows"], 0)
        self.assertAlmostEqual(report["xy_p95_m"], 0.0)


class PlanAndTrackerTest(unittest.TestCase):
    def make_plan(self) -> WholeBodyPlan:
        targets = np.zeros((2, 32), dtype=np.float64)
        targets[0, :28] = 0.1
        targets[1, :28] = 0.2
        targets[:, 28] = [0.8, 0.9]
        targets[:, 29] = [0.1, 0.2]
        return WholeBodyPlan(
            query_times_s=np.asarray([0.1, 0.2]),
            targets=targets,
            anchor_base_pose_se2_w=np.asarray([1.0, 2.0, 0.0]),
            anchor_upper_body_position=np.zeros(28),
            anchor_base_height=0.7,
        )

    def test_plan_queries_wall_clock_and_derivative(self) -> None:
        reference = self.make_plan().sample(0.05)
        np.testing.assert_allclose(reference.upper_body_position, 0.05)
        np.testing.assert_allclose(reference.base_pose_se2_w, [1.05, 2.0, 0.0], atol=1e-12)
        np.testing.assert_allclose(reference.base_twist_body_ff, [1.0, 0.0, 0.0], atol=1e-12)
        self.assertAlmostEqual(reference.base_height, 0.75)

    def test_plan_holds_final_pose_with_zero_feedforward(self) -> None:
        reference = self.make_plan().sample(1.0)
        np.testing.assert_allclose(reference.base_pose_se2_w, [1.2, 2.0, 0.0], atol=1e-12)
        np.testing.assert_allclose(reference.base_twist_body_ff, np.zeros(3))
        self.assertTrue(reference.clamped_to_horizon)

    def test_tracker_feedback_is_body_frame_and_clipped(self) -> None:
        plan = self.make_plan()
        adapter = Phase1ExecutionAdapter(
            kp_xy_per_s=1.0,
            kp_yaw_per_s=1.0,
            max_abs_base_twist=(0.5, 0.5, 0.5),
        )
        command, diagnostic = adapter.command(
            plan,
            elapsed_s=0.05,
            measured_base_pose_se2_w=np.asarray([1.0, 2.0, 0.0]),
        )
        self.assertEqual(command.shape, (32,))
        self.assertAlmostEqual(command[29], 0.5)
        self.assertAlmostEqual(diagnostic["position_error_m"], 0.05)

    def test_tracker_rotates_reference_feedforward_into_measured_body(self) -> None:
        plan = self.make_plan()
        adapter = Phase1ExecutionAdapter(
            kp_xy_per_s=0.0,
            kp_yaw_per_s=0.0,
            max_abs_base_twist=(2.0, 2.0, 2.0),
        )
        command, _ = adapter.command(
            plan,
            elapsed_s=0.05,
            measured_base_pose_se2_w=np.asarray([1.05, 2.0, -math.pi / 2.0]),
        )
        np.testing.assert_allclose(command[29:31], [0.0, 1.0], atol=1e-12)


if __name__ == "__main__":
    unittest.main()
