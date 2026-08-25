import numpy as np

from robocasa_act_navigate.rate_control_eval_server import (
    calibrated_command_path,
    geometric_prefix,
    retimed_zoh_command,
)
from robocasa_act_navigate.geometric_follower import BaseRateCalibration


def test_act_retiming_preserves_selected_velocity_integral_without_saturation():
    chunk = np.linspace(-0.2, 0.2, 32 * 3, dtype=np.float64).reshape(32, 3)
    expected = chunk[:8].sum(axis=0) / 20.0
    for rate in (0.5, 1.0, 1.5):
        dt = 1.0 / 50.0
        count = int(np.ceil((0.4 / rate) / dt)) + 1
        commands = []
        for tick in range(count):
            command, _ = retimed_zoh_command(
                chunk,
                tick * dt,
                (tick + 1) * dt,
                rate_scale=rate,
                source_hz=20.0,
                source_duration_s=0.4,
            )
            commands.append(command)
        np.testing.assert_allclose(np.sum(commands, axis=0) * dt, expected, atol=1e-7)


def test_act_retiming_reports_and_clips_infeasible_speed():
    chunk = np.ones((32, 3), dtype=np.float64)
    command, info = retimed_zoh_command(
        chunk,
        0.0,
        0.02,
        rate_scale=1.5,
        source_hz=20.0,
        source_duration_s=0.4,
    )
    np.testing.assert_allclose(command, 1.0)
    assert info["unclipped_saturation_fraction"] == 1.0


def test_native_act_playback_does_not_silently_compensate_amplitude():
    chunk = np.full((32, 3), 0.2, dtype=np.float64)
    dt = 1.0 / 50.0
    source_integral = chunk.sum(axis=0) / 20.0
    for rate in (0.5, 1.0, 1.5):
        count = int(np.ceil((1.6 / rate) / dt)) + 1
        commands = []
        for tick in range(count):
            command, info = retimed_zoh_command(
                chunk,
                tick * dt,
                (tick + 1) * dt,
                rate_scale=rate,
                source_hz=20.0,
                source_duration_s=1.6,
                compensate_amplitude=False,
            )
            commands.append(command)
            assert info.get("compensate_amplitude", False) is False
        np.testing.assert_allclose(
            np.sum(commands, axis=0) * dt,
            source_integral / rate,
            atol=2e-6,
        )


def test_geometric_prefix_interpolates_exact_metric_extent():
    path = np.asarray([[0.1, 0.0, 0.0], [0.3, 0.0, 0.0]], dtype=np.float64)
    prefix = geometric_prefix(path, 0.2)
    np.testing.assert_allclose(prefix, [[0.1, 0.0, 0.0], [0.2, 0.0, 0.0]], atol=1e-7)


def test_geometric_prefix_keeps_short_truthful_path():
    path = np.asarray([[0.05, 0.0, 0.1]], dtype=np.float64)
    prefix = geometric_prefix(path, 0.2)
    np.testing.assert_allclose(prefix, path, atol=1e-7)


def test_calibrated_command_path_integrates_body_twist_without_calling_it_truth():
    calibration = BaseRateCalibration(
        np.eye(3, dtype=np.float64), np.zeros(3, dtype=np.float64), 20.0
    )
    chunk = np.asarray([[0.2, 0.0, 0.0], [0.2, 0.0, 0.0]], dtype=np.float64)
    path = calibrated_command_path(chunk, calibration, source_hz=20.0)
    np.testing.assert_allclose(path, [[0.01, 0.0, 0.0], [0.02, 0.0, 0.0]])


def test_calibrated_command_path_applies_calibration_bias():
    calibration = BaseRateCalibration(
        np.eye(3, dtype=np.float64), np.asarray([0.1, 0.0, 0.0]), 20.0
    )
    path = calibrated_command_path(np.zeros((2, 3)), calibration, source_hz=20.0)
    np.testing.assert_allclose(path[:, 0], [0.005, 0.01])
