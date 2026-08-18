import numpy as np

from lpwb.execution import ExecutionCalibration, LinearMap, decode_path_time_commands


def calibration() -> ExecutionCalibration:
    return ExecutionCalibration(
        base=LinearMap(np.diag([0.05, 0.05, 0.10]), np.zeros(3)),
        eef_position=LinearMap(np.diag([0.01, 0.01, 0.01]), np.zeros(3)),
        eef_rotation=LinearMap(np.diag([0.10, 0.10, 0.10]), np.zeros(3)),
    )


def prediction() -> dict[str, np.ndarray]:
    count = 4
    return {
        "base_motion": np.column_stack(
            [
                0.025 * np.arange(1, count + 1),
                np.zeros(count),
                np.zeros(count),
                np.log(np.full(count, 0.05)),
            ]
        ),
        "end_effector_position": np.column_stack(
            [0.005 * np.arange(1, count + 1), np.zeros((count, 2))]
        ),
        "end_effector_rotation": np.column_stack(
            [np.zeros((count, 2)), 0.05 * np.arange(1, count + 1)]
        ),
        "gripper_close": np.array([[-1.0], [-1.0], [1.0], [1.0]]),
        "control_mode": -np.ones((count, 1)),
    }


def test_decode_uniform_path_to_constant_native_commands():
    commands, diagnostics = decode_path_time_commands(
        prediction(), calibration(), control_dt=0.05
    )
    np.testing.assert_allclose(commands["base_motion"][:, 0], 0.5, atol=1e-6)
    np.testing.assert_allclose(commands["base_motion"][:, 2], 0.0, atol=1e-6)
    np.testing.assert_allclose(commands["base_motion"][:, 3], 0.0)
    np.testing.assert_allclose(
        commands["end_effector_position"][:, 0], 0.5, atol=1e-6
    )
    np.testing.assert_allclose(
        commands["end_effector_rotation"][:, 2], 0.5, atol=1e-6
    )
    np.testing.assert_array_equal(
        commands["gripper_close"][:, 0], [-1.0, -1.0, 1.0, 1.0]
    )
    np.testing.assert_allclose(diagnostics["predicted_duration_sum_s"], 0.2)


def test_pure_base_rotation_uses_yaw_command():
    value = prediction()
    value["base_motion"][:, 0] = 0.0
    value["base_motion"][:, 2] = 0.05 * np.arange(1, 5)
    commands, _ = decode_path_time_commands(value, calibration())
    np.testing.assert_allclose(commands["base_motion"][:, :2], 0.0, atol=1e-6)
    np.testing.assert_allclose(commands["base_motion"][:, 2], 0.5, atol=1e-6)


def test_duration_clipping_and_command_clipping_are_finite():
    value = prediction()
    value["base_motion"][:, 3] = [100.0, -100.0, 100.0, -100.0]
    value["base_motion"][:, 0] = 100.0
    commands, diagnostics = decode_path_time_commands(value, calibration())
    assert all(np.all(np.isfinite(group)) for group in commands.values())
    assert np.max(np.abs(commands["base_motion"])) <= 1.0
    assert diagnostics["base_command_saturation_fraction"] > 0.0


def test_lag_compensation_leads_desired_increments():
    value = prediction()
    lagged = ExecutionCalibration(
        base=calibration().base,
        eef_position=LinearMap(np.eye(3), np.zeros(3), lag_steps=1),
        eef_rotation=calibration().eef_rotation,
    )
    commands, _ = decode_path_time_commands(value, lagged)
    np.testing.assert_allclose(commands["end_effector_position"][:, 0], 0.005)
