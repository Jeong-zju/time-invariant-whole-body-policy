from __future__ import annotations

import math
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "gate_n"))

from replan_schedule import ReplanSchedule  # noqa: E402

from whole_body_policy import (  # noqa: E402
    ArenaM1PlanExecutor,
    Phase1ExecutionAdapter,
    WholeBodyPlan,
    build_multi_horizon_targets,
    compare_base_proxy_to_odometry,
    compose,
    decode_m1_policy_output_to_simulator_chunk,
    exp,
    integrate_body_twist,
    interpolate,
    inverse,
    load_target_batch,
    log,
    relative,
    save_target_batch,
    ordered_upper_sim_indices,
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


class ReplanScheduleTest(unittest.TestCase):
    def test_non_integer_fixed_frequency_has_no_long_term_tick_drift(self) -> None:
        schedule = ReplanSchedule(control_dt_s=0.02, fixed_frequency_hz=15.0)
        intervals = [schedule.next_steps() for _ in range(300)]
        self.assertEqual(sum(intervals), 1000)
        self.assertTrue(set(intervals).issubset({3, 4}))

    def test_jitter_is_seeded_and_stays_inside_quantized_range(self) -> None:
        left = ReplanSchedule(control_dt_s=0.02, jitter_frequency_hz=(10.0, 30.0), seed=7)
        right = ReplanSchedule(control_dt_s=0.02, jitter_frequency_hz=(10.0, 30.0), seed=7)
        left_intervals = [left.next_steps() for _ in range(100)]
        right_intervals = [right.next_steps() for _ in range(100)]
        self.assertEqual(left_intervals, right_intervals)
        self.assertTrue(set(left_intervals).issubset({1, 2, 3, 4, 5}))


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


class ArenaM1PlanExecutorTest(unittest.TestCase):
    def make_executor(self) -> ArenaM1PlanExecutor:
        return ArenaM1PlanExecutor(
            upper_sim_indices=np.arange(11, 39),
            query_times_s=np.asarray([0.1, 0.2]),
            max_abs_base_twist=(2.0, 2.0, 2.0),
        )

    def test_ordered_upper_indices_follows_m1_group_order(self) -> None:
        groups = {
            "left_arm": ["la"],
            "right_arm": ["ra"],
            "left_hand": [f"lh{i}" for i in range(13)],
            "right_hand": [f"rh{i}" for i in range(13)],
        }
        names = ["la", "ra", *groups["left_hand"], *groups["right_hand"]]
        mapping = {name: index for index, name in enumerate(names)}
        np.testing.assert_array_equal(ordered_upper_sim_indices(groups, mapping), np.arange(28))

    def test_m1_decoder_maps_new_keys_without_legacy_action_names(self) -> None:
        output = {
            "action.phase1_upper_body_position": np.full((1, 16, 28), 0.2),
            "action.phase1_base_height": np.full((1, 16, 1), 0.75),
            "action.phase1_base_relative_se2": np.full((1, 16, 3), 0.1),
        }
        chunk = decode_m1_policy_output_to_simulator_chunk(output, np.arange(11, 39))
        self.assertEqual(chunk.shape, (1, 16, 50))
        np.testing.assert_allclose(chunk[..., 11:39], 0.2)
        np.testing.assert_allclose(chunk[..., 43:46], 0.1)
        np.testing.assert_allclose(chunk[..., 46], 0.75)
        np.testing.assert_allclose(chunk[..., :11], 0.0)

    def test_executor_reanchors_and_rewrites_only_m1_slots(self) -> None:
        executor = self.make_executor()
        chunk = np.zeros((2, 50), dtype=np.float64)
        chunk[0, 11:39] = 0.1
        chunk[1, 11:39] = 0.2
        chunk[:, 46] = [0.8, 0.9]
        chunk[:, 43] = [0.1, 0.2]
        activation = executor.activate(
            chunk,
            measured_sim_joint_position=np.zeros(43),
            measured_base_pose_se2_w=np.asarray([1.0, 2.0, 0.0]),
            current_base_height_command=0.7,
            activation_monotonic_s=10.0,
        )
        self.assertEqual(activation.plan_id, 0)
        template = np.arange(50, dtype=np.float64)
        action, diagnostic = executor.command(
            template,
            measured_base_pose_se2_w=np.asarray([1.0, 2.0, 0.0]),
            monotonic_s=10.05,
        )
        np.testing.assert_allclose(action[11:39], 0.05)
        self.assertAlmostEqual(action[46], 0.75)
        self.assertAlmostEqual(action[43], 1.05)
        np.testing.assert_allclose(action[[0, 1, 2, 47, 48, 49]], template[[0, 1, 2, 47, 48, 49]])
        self.assertAlmostEqual(diagnostic["plan_age_s"], 0.05)

    def test_executor_holds_pose_after_horizon_with_zero_feedforward(self) -> None:
        executor = self.make_executor()
        chunk = np.zeros((2, 50), dtype=np.float64)
        chunk[:, 43] = [0.1, 0.2]
        executor.activate(
            chunk,
            measured_sim_joint_position=np.zeros(43),
            measured_base_pose_se2_w=np.zeros(3),
            current_base_height_command=0.7,
            activation_monotonic_s=1.0,
        )
        action, diagnostic = executor.command(
            np.zeros(50),
            measured_base_pose_se2_w=np.asarray([0.2, 0.0, 0.0]),
            monotonic_s=2.0,
        )
        np.testing.assert_allclose(action[43:46], np.zeros(3), atol=1e-12)
        self.assertTrue(diagnostic["clamped_to_horizon"])


if __name__ == "__main__":
    unittest.main()
