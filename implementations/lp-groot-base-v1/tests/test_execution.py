import numpy as np

from lp_groot_base.execution import (
    ControllerCalibration,
    path_to_base_commands,
    reconstruct_prediction,
)


def test_reconstruct_prediction_respects_arrival_time():
    path = np.array([[1.0, 0.0, 0.0, np.log(2.0)]], dtype=np.float32)
    pose = reconstruct_prediction(path, np.array([0.5, 1.0, 2.0]))
    np.testing.assert_allclose(pose[:, 0], [0.25, 0.5, 1.0], atol=1e-6)


def test_identity_calibration_maps_path_velocity_to_command():
    calibration = ControllerCalibration(np.eye(3), np.zeros(3), control_dt=0.05)
    path = np.array([[0.1, 0.0, 0.0, np.log(0.1)]], dtype=np.float32)
    command, desired = path_to_base_commands(path, calibration, num_control_steps=2)
    np.testing.assert_allclose(command[:, 0], [1.0, 1.0], atol=1e-5)
    np.testing.assert_allclose(desired[:, 0], [0.05, 0.1], atol=1e-5)
