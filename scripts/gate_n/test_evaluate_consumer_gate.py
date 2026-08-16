import unittest

from evaluate_consumer_gate import evaluate


def report(xy: float, success_rates: tuple[float, float, float], identical: bool = True) -> dict:
    r16, r8, r4 = success_rates
    pair = lambda rate: {
        "pairs": 10,
        "all_first_action_chunks_identical": identical,
        "all_prebranch_physical_telemetry_identical": identical,
        "root_xy_trajectory_rms_m": {"median": xy},
        "root_yaw_trajectory_rms_rad": {"median": 0.01},
        "success_rate_change": rate - r16,
    }
    return {
        "conditions": {"16": {"success_rate": r16}},
        "paired_against_reference": {"8": pair(r8), "4": pair(r4)},
    }


CONFIG = {
    "protocol_id": "test",
    "validity": {"minimum_complete_seed_pairs_per_frequency": 10},
    "resistance_thresholds": {
        "maximum_median_root_xy_trajectory_rms_m": 0.10,
        "maximum_median_root_yaw_trajectory_rms_rad": 0.15,
        "maximum_absolute_success_rate_change": 0.20,
        "maximum_default_frequency_success_drop_vs_raw": 0.10,
    },
}


class ConsumerGateEvaluationTest(unittest.TestCase):
    def test_research_proceeds_only_when_all_simple_routes_fail(self) -> None:
        raw = report(0.5, (0.8, 0.5, 0.4))
        matched = report(0.4, (0.8, 0.5, 0.4))
        dt = report(0.3, (0.8, 0.6, 0.5), identical=False)
        se2 = report(0.2, (0.8, 0.6, 0.5))
        result = evaluate(raw, matched, dt, se2, CONFIG)
        self.assertTrue(result["gate_n_a_passed"])
        self.assertTrue(result["gate_n_b_new_policy_necessity_passed"])
        self.assertTrue(result["mobile_vla_policy_research_warranted"])
        self.assertTrue(result["gate_n_passed"])

    def test_dt_policy_can_stop_new_model_claim_without_forced_prefix(self) -> None:
        raw = report(0.5, (0.8, 0.5, 0.4))
        matched = report(0.4, (0.8, 0.5, 0.4))
        dt = report(0.05, (0.8, 0.8, 0.7), identical=False)
        se2 = report(0.2, (0.8, 0.6, 0.5))
        result = evaluate(raw, matched, dt, se2, CONFIG)
        self.assertTrue(result["gate_n_a_passed"])
        self.assertFalse(result["gate_n_b_new_policy_necessity_passed"])
        self.assertFalse(result["mobile_vla_policy_research_warranted"])
        self.assertIn("replan_dt_conditioned_policy", result["sufficient_methods"])

    def test_no_raw_frequency_problem_stops_before_necessity_claim(self) -> None:
        raw = report(0.05, (0.8, 0.8, 0.8))
        failing = report(0.4, (0.8, 0.5, 0.4))
        result = evaluate(raw, failing, failing, failing, CONFIG)
        self.assertFalse(result["gate_n_a_passed"])
        self.assertTrue(result["all_simple_baselines_failed"])
        self.assertFalse(result["gate_n_b_new_policy_necessity_passed"])
        self.assertEqual(result["decision"], "stop_no_practical_frequency_problem_observed")


if __name__ == "__main__":
    unittest.main()
