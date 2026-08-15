#!/usr/bin/env python3

import unittest

from evaluate_gate_n_report import evaluate


class GateNEvaluationTest(unittest.TestCase):
    def test_failed_ambiguity_and_turning_stop_training(self) -> None:
        summary = lambda value: {"median": value}
        report = {
            "protocol_id": "report",
            "matching": {"selected_pairs_after_future_filter": 3000},
            "label_alignment": {
                "same_direction_speed_contrast_pairs": {
                    "pairs": 92,
                    "progress_median_reduction_vs_time": 0.08,
                    "fraction_progress_better_than_time": 0.81,
                }
            },
            "resampling": {
                "methods": {
                    "time_uniform_3x": {
                        "geometry_reconstruction_rms": summary(0.02),
                        "high_acceleration_coverage": summary(1.0),
                        "turning_coverage": summary(1.0),
                        "base_motion_coverage": summary(1.0),
                    },
                    "whole_body_progress": {
                        "geometry_reconstruction_rms": summary(0.018),
                        "high_acceleration_coverage": summary(1.0),
                        "turning_coverage": summary(0.88),
                        "base_motion_coverage": summary(0.99),
                    },
                    "isr_budget_matched": {
                        "geometry_reconstruction_rms": summary(0.019),
                        "high_acceleration_coverage": summary(1.0),
                        "turning_coverage": summary(0.92),
                        "base_motion_coverage": summary(0.92),
                    },
                    "isr_paper_default": {"retention_ratio": summary(1.0)},
                }
            },
        }
        config = {
            "protocol_id": "gate",
            "offline_diagnostic": {
                "sample_budget": {"compare_isr_at_dataset_global_median_retention_ratio": 1 / 3}
            },
            "offline_gate": {
                "minimum_matched_pairs": 1000,
                "minimum_same_direction_speed_contrast_pairs": 200,
                "minimum_progress_median_disagreement_reduction_vs_time": 0.2,
                "minimum_fraction_pairs_where_progress_beats_time": 0.6,
                "maximum_critical_event_coverage_drop_vs_time_uniform": 0.05,
                "maximum_geometry_reconstruction_rms_ratio_vs_time_uniform": 1.25,
                "failure_action": "stop",
            },
        }
        result = evaluate(report, config)
        self.assertFalse(result["offline_gate_passed"])
        self.assertEqual(result["decision"], "stop")
        self.assertFalse(result["n1_label_ambiguity"]["passed"])
        self.assertFalse(result["n2_resampling"]["passed"])


if __name__ == "__main__":
    unittest.main()
