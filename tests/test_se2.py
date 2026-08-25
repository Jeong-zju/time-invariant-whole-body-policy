import numpy as np

from lp_act.se2 import between, compose, exp, integrate_body_velocity, inverse, log


def test_exp_log_round_trip() -> None:
    rng = np.random.default_rng(0)
    twists = rng.normal(scale=[0.3, 0.3, 0.8], size=(1000, 3))
    recovered = log(exp(twists))
    np.testing.assert_allclose(recovered, twists, atol=1e-9)


def test_compose_inverse_identity() -> None:
    poses = np.asarray([[1.2, -0.3, 0.7], [-2.0, 1.0, -1.2]])
    np.testing.assert_allclose(compose(poses, inverse(poses)), np.zeros_like(poses), atol=1e-9)
    np.testing.assert_allclose(between(poses, poses), np.zeros_like(poses), atol=1e-9)


def test_positive_axis_and_rotation_signs() -> None:
    dt = np.asarray([1.0])
    np.testing.assert_allclose(integrate_body_velocity(np.asarray([[1.0, 0.0, 0.0]]), dt)[-1], [1, 0, 0])
    np.testing.assert_allclose(integrate_body_velocity(np.asarray([[0.0, 1.0, 0.0]]), dt)[-1], [0, 1, 0])
    np.testing.assert_allclose(
        integrate_body_velocity(np.asarray([[0.0, 0.0, 0.5]]), dt)[-1], [0, 0, 0.5]
    )


def test_curved_body_motion() -> None:
    pose = integrate_body_velocity(np.asarray([[1.0, 0.0, np.pi / 2]]), np.asarray([1.0]))[-1]
    np.testing.assert_allclose(pose, [2 / np.pi, 2 / np.pi, np.pi / 2], atol=1e-9)
