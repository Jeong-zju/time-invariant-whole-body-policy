import numpy as np

from lp_groot_base import se2


def test_exp_log_roundtrip() -> None:
    twists = np.array(
        [
            [0.2, 0.0, 0.0],
            [0.0, 0.2, 0.0],
            [0.0, 0.0, 0.4],
            [0.3, -0.1, 0.5],
            [1e-9, -2e-9, 1e-10],
        ]
    )
    np.testing.assert_allclose(se2.log(se2.exp(twists)), twists, atol=1e-8)


def test_between_uses_common_origin() -> None:
    origin = np.array([1.0, 2.0, np.pi / 2])
    target = np.array([1.0, 3.0, np.pi / 2])
    np.testing.assert_allclose(se2.between(origin, target), [1.0, 0.0, 0.0], atol=1e-8)


def test_quaternion_is_xyzw() -> None:
    quat = np.array([0.0, 0.0, np.sin(np.pi / 4), np.cos(np.pi / 4)])
    np.testing.assert_allclose(se2.quat_xyzw_to_yaw(quat), np.pi / 2, atol=1e-8)

