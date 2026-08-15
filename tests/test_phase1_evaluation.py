from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import sys
import unittest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "research_phase_1"))

from evaluate_phase1_m1 import evaluate  # noqa: E402


class Phase1EvaluationTest(unittest.TestCase):
    def setUp(self) -> None:
        pair = {
            "pairs": 10,
            "all_first_action_chunks_identical": True,
            "all_prebranch_physical_telemetry_identical": True,
            "root_xy_trajectory_rms_m": {"median": 0.05},
            "root_yaw_trajectory_rms_rad": {"median": 0.04},
            "success_rate_change": -0.1,
        }
        self.standard = {
            "conditions": {"16": {"success_rate": 0.9}},
            "paired_against_reference": {"8": pair, "4": deepcopy(pair)},
        }
        self.matched = {"conditions": {"16": {"success_rate": 1.0}}}
        self.config = {
            "protocol_id": "arena-g1-phase-1-m1-v1",
            "frozen_variables": {"optimizer_steps": 47468},
            "validity": {"minimum_complete_seed_pairs_per_frequency": 10},
            "resistance_thresholds": {
                "maximum_median_root_xy_trajectory_rms_m": 0.1,
                "maximum_median_root_yaw_trajectory_rms_rad": 0.15,
                "maximum_absolute_success_rate_change": 0.2,
                "maximum_default_frequency_success_drop_vs_matched_lora": 0.1,
            },
        }
        condition = {"completed_runs": 10, "safety_checks_passed": True}
        self.extended = {str(index): deepcopy(condition) for index in range(8)}
        self.dataset = {
            "target_source": "measured_odometry",
            "target_source_counts": {"measured_odometry": 84289},
        }
        self.training = {"global_step": 47468, "upstream": {"groot": "revision"}}

    def test_all_frozen_gates_pass(self) -> None:
        result = evaluate(
            self.standard,
            self.matched,
            self.config,
            self.extended,
            self.dataset,
            self.training,
        )
        self.assertTrue(result["m1_passed"])
        self.assertEqual(result["decision"], "stop_and_report_action_representation_as_sufficient")

    def test_threshold_is_strict_for_frequency_effect(self) -> None:
        self.standard["paired_against_reference"]["4"]["root_xy_trajectory_rms_m"][
            "median"
        ] = 0.1
        result = evaluate(
            self.standard,
            self.matched,
            self.config,
            self.extended,
            self.dataset,
            self.training,
        )
        self.assertFalse(result["m1_passed"])


if __name__ == "__main__":
    unittest.main()
