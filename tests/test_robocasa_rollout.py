import numpy as np

from robocasa_lp_act.rollout import lerobot_to_env_action


def test_lerobot_to_env_action_order() -> None:
    action = np.arange(12, dtype=np.float64)
    expected = np.asarray([5, 6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4], dtype=np.float64)
    np.testing.assert_allclose(lerobot_to_env_action(action / 20.0), expected / 20.0)
