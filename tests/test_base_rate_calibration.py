import numpy as np

from robocasa_act_navigate.calibrate_base_30hz import fit_rate_calibration


def test_rate_calibration_recovers_matrix_and_bias():
    rng = np.random.default_rng(20260823)
    commands = rng.uniform(-0.8, 0.8, size=(500, 3))
    matrix = np.asarray([[0.6, 0.01, 0.0], [-0.02, 0.62, 0.01], [0.0, 0.02, 1.2]])
    bias = np.asarray([0.001, -0.002, 0.003])
    measured = commands @ matrix + bias
    calibration, diagnostics = fit_rate_calibration(commands, measured, 30.0)
    np.testing.assert_allclose(calibration.matrix, matrix, atol=1e-12)
    np.testing.assert_allclose(calibration.bias, bias, atol=1e-12)
    np.testing.assert_allclose(diagnostics["rmse_body_twist_per_second"], 0.0, atol=1e-12)
