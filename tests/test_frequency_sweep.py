import numpy as np

from robocasa_act_navigate.frequency_eval_server import interval_average_zoh


def test_20hz_resampling_is_identity_for_first_replan_window():
    chunk = np.arange(32 * 3, dtype=np.float64).reshape(32, 3)
    actual = np.stack(
        [interval_average_zoh(chunk, k / 20, (k + 1) / 20, source_hz=20) for k in range(8)]
    )
    np.testing.assert_allclose(actual, chunk[:8], atol=1e-12)


def test_10hz_resampling_averages_each_pair_of_20hz_commands():
    chunk = np.arange(32, dtype=np.float64)[:, None]
    actual = np.stack(
        [interval_average_zoh(chunk, k / 10, (k + 1) / 10, source_hz=20) for k in range(4)]
    )
    np.testing.assert_allclose(actual[:, 0], [0.5, 2.5, 4.5, 6.5], atol=1e-12)


def test_50hz_resampling_preserves_20hz_command_integral():
    rng = np.random.default_rng(20260823)
    chunk = rng.normal(size=(32, 3))
    target = np.stack(
        [interval_average_zoh(chunk, k / 50, (k + 1) / 50, source_hz=20) for k in range(20)]
    )
    np.testing.assert_allclose(target.sum(axis=0) / 50, chunk[:8].sum(axis=0) / 20, atol=1e-12)


def test_constant_velocity_is_frequency_invariant():
    chunk = np.repeat(np.asarray([[0.2, -0.1, 0.3]]), 32, axis=0)
    for hz in (10, 20, 50):
        target = np.stack(
            [
                interval_average_zoh(chunk, k / hz, (k + 1) / hz, source_hz=20)
                for k in range(round(0.4 * hz))
            ]
        )
        expected = np.repeat(chunk[:1], len(target), axis=0)
        np.testing.assert_allclose(target, expected, atol=1e-12)
