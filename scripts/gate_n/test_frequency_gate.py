from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from evaluate_frequency_gate import evaluate
from evaluate_frequency_gate_final import evaluate_final
from frequency_metrics import (
    integrate_body_twist,
    nested_observation_sha256,
    summarize_trajectory,
    yaw_from_wxyz,
)
from summarize_frequency_gate import aggregate
from summarize_frequency_repeatability import summarize as summarize_repeatability


def quaternion_wxyz(yaw: float) -> np.ndarray:
    return np.asarray([math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)])


class FrequencyMetricsTest(unittest.TestCase):
    def test_nested_observation_hash_is_key_order_invariant(self) -> None:
        first = {"image": np.arange(6, dtype=np.uint8).reshape(2, 3), "state": np.asarray([1.0])}
        second = {"state": np.asarray([1.0]), "image": np.arange(6, dtype=np.uint8).reshape(2, 3)}
        self.assertEqual(nested_observation_sha256(first), nested_observation_sha256(second))

    def test_integrate_straight_body_velocity(self) -> None:
        commands = np.tile(np.asarray([[1.0, 0.0, 0.0]]), (10, 1))
        pose = integrate_body_twist(commands, np.zeros(10), 0.1)
        np.testing.assert_allclose(pose[-1], [1.0, 0.0, 0.0], atol=1e-12)

    def test_yaw_round_trip(self) -> None:
        self.assertAlmostEqual(yaw_from_wxyz(quaternion_wxyz(0.7)), 0.7)

    def test_summary_uses_initial_state(self) -> None:
        commands = np.tile(np.asarray([[1.0, 0.0, 0.0]]), (10, 1))
        root_position = np.zeros((11, 3))
        root_position[:, 0] = np.arange(11) * 0.1
        root_quaternion = np.tile(quaternion_wxyz(0.0), (11, 1))
        root_linear_velocity = np.zeros((11, 3))
        root_linear_velocity[1:, 0] = 1.0
        wrist = np.zeros((11, 7))
        summary = summarize_trajectory(
            root_position,
            root_quaternion,
            root_linear_velocity,
            commands,
            wrist,
            wrist,
            0.1,
        )
        self.assertAlmostEqual(summary["root_net_displacement_xy_m"], 1.0)
        self.assertAlmostEqual(summary["command_vs_measured_endpoint_error_m"], 0.0)
        self.assertAlmostEqual(summary["command_vs_measured_linear_velocity_rms_mps"], 0.0)

    def test_summary_reads_translation_from_pose_matrix(self) -> None:
        commands = np.zeros((2, 3))
        root_position = np.zeros((3, 3))
        root_quaternion = np.tile(quaternion_wxyz(0.0), (3, 1))
        root_linear_velocity = np.zeros((3, 3))
        wrist = np.tile(np.eye(4), (3, 1, 1))
        wrist[:, 0, 3] = [0.0, 0.5, 1.0]
        summary = summarize_trajectory(
            root_position,
            root_quaternion,
            root_linear_velocity,
            commands,
            wrist,
            wrist,
            0.02,
        )
        self.assertGreater(summary["base_arm_progress_phase_mae"], 0.0)


class FrequencyAggregateTest(unittest.TestCase):
    def _write_run(self, root: Path, replan_steps: int, scale: float) -> None:
        directory = root / f"replan-{replan_steps}" / "seed-000"
        directory.mkdir(parents=True)
        position = np.zeros((10, 3))
        position[:9, 0] = np.arange(9) * 0.1
        position[9, 0] = 0.8 + scale
        quaternion = np.tile(quaternion_wxyz(0.0), (10, 1))
        zeros3 = np.zeros((10, 3))
        joints = np.zeros((10, 43))
        wrists = np.tile(np.eye(4), (10, 1, 1))
        np.savez_compressed(
            directory / "rollout.npz",
            root_position_w=position,
            root_quaternion_wxyz=quaternion,
            root_linear_velocity_w=zeros3,
            root_angular_velocity_w=zeros3,
            joint_position=joints,
            left_wrist_pose_pelvis=wrists,
            right_wrist_pose_pelvis=wrists,
            navigate_command=np.zeros((9, 3)),
        )
        report = {
            "replan_steps": replan_steps,
            "seed": 0,
            "telemetry_npz": "rollout.npz",
            "first_action_chunk_sha256": "same",
            "first_policy_observation_sha256": "same-observation",
            "rollout": {"success": replan_steps == 16, "steps_executed": 3},
            "trajectory": {
                "root_path_length_xy_m": 3.0 * scale,
                "command_vs_measured_endpoint_error_m": scale,
                "base_arm_progress_phase_mae": 0.1 * scale,
            },
            "inference": {"wall_time_median_s": 0.2},
        }
        (directory / "rollout.json").write_text(json.dumps(report), encoding="utf-8")

    def test_aggregate_pairs_identical_first_chunk(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_run(root, 16, 0.1)
            self._write_run(root, 8, 0.2)
            config = {
                "protocol_id": "test",
                "closed_loop_diagnostic": {
                    "replan_steps": [16, 8],
                    "seeds": [0],
                    "reference_replan_steps": 16,
                },
            }
            report = aggregate(root, config)
            comparison = report["paired_against_reference"]["8"]
            self.assertEqual(report["completed_runs"], 2)
            self.assertTrue(comparison["all_first_action_chunks_identical"])
            self.assertTrue(comparison["all_first_policy_observations_identical"])
            self.assertTrue(comparison["all_prebranch_physical_telemetry_identical"])
            self.assertGreater(comparison["root_xy_trajectory_rms_m"]["median"], 0.0)

    def test_same_frequency_repeatability_control_passes_on_identical_runs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            reference = root / "reference"
            repeat = root / "repeat"
            self._write_run(reference, 16, 0.1)
            self._write_run(repeat, 16, 0.1)
            (repeat / "replan-16").rename(repeat / "repeat-16")
            config = {
                "protocol_id": "test",
                "repeatability_audit": {
                    "seeds": [0],
                    "maximum_tolerable_same_frequency_noise": {
                        "median_root_xy_trajectory_rms_m": 0.1,
                        "median_root_yaw_trajectory_rms_rad": 0.15,
                        "absolute_success_rate_change": 0.2,
                    },
                },
            }
            report = summarize_repeatability(reference, repeat, config)
            self.assertTrue(report["valid"])
            self.assertTrue(report["repeatability_control_passed"])


class FrequencyEvaluationTest(unittest.TestCase):
    def test_final_gate_rejects_positive_screen_with_material_repeat_noise(self) -> None:
        primary = {
            "protocol_id": "test",
            "frequency_problem_observed": True,
            "decision": "continue",
        }
        repeatability = {
            "repeatability_control_passed": False,
            "decision": "do_not_attribute",
        }
        result = evaluate_final(primary, repeatability)
        self.assertFalse(result["final_gate_n_passed"])
        self.assertEqual(result["decision"], "stabilize_or_model_same_frequency_variance_before_gate_c")

    def test_gate_passes_on_preregistered_xy_effect(self) -> None:
        report = {
            "protocol_id": "test",
            "missing_runs": [],
            "paired_against_reference": {
                "8": {
                    "pairs": 10,
                    "all_first_action_chunks_identical": True,
                    "all_first_policy_observations_identical": True,
                    "all_prebranch_physical_telemetry_identical": True,
                    "success_rate_change": -0.1,
                    "root_xy_trajectory_rms_m": {"median": 0.2},
                    "root_yaw_trajectory_rms_rad": {"median": 0.02},
                }
            },
        }
        config = {
            "protocol_id": "test",
            "gate_thresholds": {
                "minimum_complete_seed_pairs_per_condition": 10,
                "minimum_median_root_xy_trajectory_rms_m": 0.1,
                "minimum_median_root_yaw_trajectory_rms_rad": 0.15,
                "minimum_absolute_success_rate_change": 0.2,
                "pass_action": "continue",
                "failure_action": "stop",
            },
        }
        result = evaluate(report, config)
        self.assertTrue(result["frequency_problem_observed"])
        self.assertEqual(result["decision"], "continue")

    def test_gate_marks_bad_pairing_invalid_not_negative(self) -> None:
        report = {
            "protocol_id": "test",
            "missing_runs": [],
            "paired_against_reference": {
                "8": {
                    "pairs": 10,
                    "all_first_action_chunks_identical": False,
                    "all_first_policy_observations_identical": True,
                    "all_prebranch_physical_telemetry_identical": True,
                    "success_rate_change": 0.0,
                    "root_xy_trajectory_rms_m": {"median": 0.2},
                    "root_yaw_trajectory_rms_rad": {"median": 0.02},
                }
            },
        }
        config = {
            "protocol_id": "test",
            "gate_thresholds": {
                "minimum_complete_seed_pairs_per_condition": 10,
                "minimum_median_root_xy_trajectory_rms_m": 0.1,
                "minimum_median_root_yaw_trajectory_rms_rad": 0.15,
                "minimum_absolute_success_rate_change": 0.2,
                "pass_action": "continue",
                "failure_action": "stop",
                "incomplete_action": "resume",
                "invalid_action": "rerun",
            },
        }
        result = evaluate(report, config)
        self.assertFalse(result["frequency_problem_observed"])
        self.assertEqual(result["decision"], "rerun")


if __name__ == "__main__":
    unittest.main()
