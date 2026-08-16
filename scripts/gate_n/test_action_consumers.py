import math
import unittest

import numpy as np

from action_consumers import SE2WaypointConsumer, TimestampedPlanBank, wrap_to_pi


def chunk(vx: float, length: int = 16) -> np.ndarray:
    value = np.zeros((length, 50), dtype=np.float64)
    value[:, 43] = vx
    return value


class TimestampedPlanBankTest(unittest.TestCase):
    def test_single_plan_matches_raw_velocity(self) -> None:
        consumer = TimestampedPlanBank(control_dt_s=0.02)
        value = chunk(0.2)
        consumer.add_chunk(value, start_step=0, root_se2_w=np.zeros(3))
        command, diagnostic = consumer.command(5, np.zeros(3), np.zeros(3))
        np.testing.assert_allclose(command, [0.2, 0.0, 0.0])
        self.assertEqual(diagnostic["active_plans"], 1)

    def test_overlapping_plans_are_aligned_by_absolute_step(self) -> None:
        consumer = TimestampedPlanBank(control_dt_s=0.02, decay_time_s=0.32)
        consumer.add_chunk(chunk(0.1), start_step=0, root_se2_w=np.zeros(3))
        consumer.add_chunk(chunk(0.3), start_step=4, root_se2_w=np.zeros(3))
        command, diagnostic = consumer.command(4, np.zeros(3), np.zeros(3))
        old_weight = math.exp(-0.08 / 0.32)
        expected = (old_weight * 0.1 + 0.3) / (old_weight + 1.0)
        self.assertAlmostEqual(command[0], expected)
        self.assertEqual(diagnostic["active_plans"], 2)

    def test_expired_plans_are_removed(self) -> None:
        consumer = TimestampedPlanBank(control_dt_s=0.02)
        consumer.add_chunk(chunk(0.1, length=4), start_step=0, root_se2_w=np.zeros(3))
        consumer.add_chunk(chunk(0.2, length=4), start_step=4, root_se2_w=np.zeros(3))
        command, diagnostic = consumer.command(4, np.zeros(3), np.zeros(3))
        np.testing.assert_allclose(command, [0.2, 0.0, 0.0])
        self.assertEqual(diagnostic["active_plans"], 1)


class SE2WaypointConsumerTest(unittest.TestCase):
    def test_zero_tracking_error_returns_feedforward(self) -> None:
        consumer = SE2WaypointConsumer(control_dt_s=0.02)
        consumer.add_chunk(chunk(0.2), start_step=0, root_se2_w=np.asarray([1.0, 2.0, 0.0]))
        command, diagnostic = consumer.command(
            0,
            root_se2_w=np.asarray([1.0, 2.0, 0.0]),
            fallback_navigate_command=np.zeros(3),
        )
        np.testing.assert_allclose(command, [0.2, 0.0, 0.0])
        self.assertAlmostEqual(diagnostic["position_error_m"], 0.0)

    def test_position_error_is_converted_to_body_frame(self) -> None:
        consumer = SE2WaypointConsumer(control_dt_s=0.02, kp_xy_per_s=1.0)
        consumer.add_chunk(chunk(0.0), start_step=0, root_se2_w=np.asarray([1.0, 0.0, 0.0]))
        command, _ = consumer.command(
            0,
            root_se2_w=np.asarray([0.0, 0.0, math.pi / 2.0]),
            fallback_navigate_command=np.zeros(3),
        )
        np.testing.assert_allclose(command[:2], [0.0, -0.5], atol=1e-12)

    def test_yaw_wrap_uses_short_direction(self) -> None:
        error = float(wrap_to_pi((-math.pi + 0.1) - (math.pi - 0.1)))
        self.assertAlmostEqual(error, 0.2)


if __name__ == "__main__":
    unittest.main()
