#!/usr/bin/env python3

import unittest

import numpy as np

from analyze_arena_gate_n import (
    equidistant_indices,
    integrate_base,
    isr_indices,
    sample_by_coordinate,
    uniform_time_indices,
)


class GateNAnalysisTest(unittest.TestCase):
    def test_integrate_base_straight(self) -> None:
        command = np.tile(np.array([[1.0, 0.0, 0.0]]), (4, 1))
        timestamp = np.arange(4) * 0.1
        pose = integrate_base(command, timestamp)
        np.testing.assert_allclose(pose[:, 0], [0.0, 0.1, 0.2, 0.3], atol=1e-12)
        np.testing.assert_allclose(pose[:, 1:], 0.0, atol=1e-12)

    def test_time_indices_include_endpoint(self) -> None:
        np.testing.assert_array_equal(uniform_time_indices(8, 3), [0, 3, 6, 7])

    def test_equidistant_indices_constant_path_is_safe(self) -> None:
        indices = equidistant_indices(np.zeros(10), 4)
        self.assertEqual(indices[0], 0)
        self.assertEqual(indices[-1], 9)
        self.assertTrue(np.all(np.diff(indices) > 0))

    def test_progress_interpolation_removes_speed_parameterization(self) -> None:
        slow_coordinate = np.linspace(0.0, 1.0, 101)
        fast_coordinate = np.linspace(0.0, 1.0, 51)
        slow_values = np.stack([slow_coordinate, slow_coordinate**2], axis=1)
        fast_values = np.stack([fast_coordinate, fast_coordinate**2], axis=1)
        targets = np.linspace(0.1, 0.9, 9)
        slow = sample_by_coordinate(slow_values, slow_coordinate, targets)
        fast = sample_by_coordinate(fast_values, fast_coordinate, targets)
        np.testing.assert_allclose(slow, fast, atol=1e-12)

    def test_isr_indices_are_monotone_and_keep_endpoints(self) -> None:
        timestamp = np.arange(21) * 0.02
        positions = np.stack([np.linspace(0.0, 0.2, 21), np.zeros(21), np.zeros(21)], axis=1)
        indices = isr_indices(
            positions, timestamp, target_distance=0.05, lambda_vel=1.0, lambda_acc=0.0
        )
        self.assertEqual(indices[0], 0)
        self.assertEqual(indices[-1], 20)
        self.assertTrue(np.all(np.diff(indices) > 0))
        self.assertGreaterEqual(len(indices), 4)
        self.assertLessEqual(len(indices), 6)


if __name__ == "__main__":
    unittest.main()
