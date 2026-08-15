import math
import unittest

from replan_dt import encode_replan_dt


class ReplanDeltaTEncodingTest(unittest.TestCase):
    def test_three_gate_intervals_map_to_symmetric_values(self) -> None:
        self.assertAlmostEqual(encode_replan_dt(0.08), -1.0)
        self.assertAlmostEqual(encode_replan_dt(0.16), 0.0)
        self.assertAlmostEqual(encode_replan_dt(0.32), 1.0)

    def test_nonpositive_interval_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            encode_replan_dt(0.0)

    def test_log_encoding_is_continuous(self) -> None:
        self.assertAlmostEqual(encode_replan_dt(0.16 * math.sqrt(2.0)), 0.5)
