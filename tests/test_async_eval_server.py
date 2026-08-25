import numpy as np

from robocasa_act_navigate.async_eval_server import FractionalRequestScheduler
from robocasa_act_navigate.frequency_eval_server import interval_average_zoh


def test_fractional_scheduler_emits_exact_rates_on_50hz_grid():
    for request_hz in (10, 20, 50):
        scheduler = FractionalRequestScheduler(request_hz, 50)
        assert sum(scheduler.step() for _ in range(50)) == request_hz


def test_time_aligned_act_skips_stale_prefix_without_changing_time_semantics():
    chunk = np.arange(32, dtype=np.float64)[:, None]
    # A prediction captured four 50 Hz ticks ago starts at 80 ms.  The exact
    # 80--100 ms interval lies wholly inside the second native 20 Hz token.
    actual = interval_average_zoh(chunk, 0.08, 0.10, source_hz=20.0)
    np.testing.assert_allclose(actual, [1.0])
