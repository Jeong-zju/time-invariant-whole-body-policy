import numpy as np

from lp_act.execution import decode_lp_chunk


def test_decode_straight_path_to_constant_velocity() -> None:
    target = np.zeros((32, 24), dtype=np.float32)
    target[:, 0] = np.arange(1, 33) * 0.01
    target[:, 3:23] = np.arange(20, dtype=np.float32)
    target[:, 23] = np.log(1.0 / 30.0)
    origin = np.arange(20, dtype=np.float32)
    decoded = decode_lp_chunk(target, origin)
    assert decoded.actions.shape == (32, 23)
    np.testing.assert_allclose(decoded.actions[:, 0], 0.3, atol=1e-6)
    np.testing.assert_allclose(decoded.actions[:, 1:3], 0.0, atol=1e-6)
    np.testing.assert_allclose(decoded.actions[:, 3:23], np.broadcast_to(origin, (32, 20)), atol=1e-6)


def test_decode_clips_invalid_durations() -> None:
    target = np.zeros((2, 24), dtype=np.float32)
    target[:, 0] = [0.01, 0.02]
    target[:, 23] = [100.0, -100.0]
    decoded = decode_lp_chunk(target, np.zeros(20), max_total_duration=0.2)
    assert decoded.actions.shape[0] == 6
    assert np.isfinite(decoded.actions).all()
