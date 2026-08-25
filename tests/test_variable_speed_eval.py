import numpy as np
import pytest

from robocasa_act_navigate.variable_speed_eval_server import nominal_path_progress_rate


def test_nominal_path_progress_rate_translation() -> None:
    path = np.asarray([[0.1, 0.0, 0.0], [0.3, 0.0, 0.0]], dtype=np.float32)
    assert nominal_path_progress_rate(path, 1.5) == pytest.approx(0.2)


def test_nominal_path_progress_rate_includes_yaw_metric() -> None:
    path = np.asarray([[0.0, 0.0, 1.0]], dtype=np.float32)
    assert nominal_path_progress_rate(path, 0.5) == pytest.approx(0.5)


@pytest.mark.parametrize(
    "path,duration",
    [
        (np.zeros((3, 2), dtype=np.float32), 1.0),
        (np.zeros((0, 3), dtype=np.float32), 1.0),
        (np.zeros((3, 3), dtype=np.float32), 0.0),
    ],
)
def test_nominal_path_progress_rate_rejects_invalid_inputs(path, duration) -> None:
    with pytest.raises(ValueError):
        nominal_path_progress_rate(path, duration)
