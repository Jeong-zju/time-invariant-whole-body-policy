import numpy as np

from lp_groot_base.labels import LabelConfig, build_path_time_label, reconstruct_at_times


def _quat_from_yaw(yaw: np.ndarray) -> np.ndarray:
    q = np.zeros((len(yaw), 4), dtype=np.float64)
    q[:, 2] = np.sin(yaw / 2.0)
    q[:, 3] = np.cos(yaw / 2.0)
    return q


def test_complete_straight_path_reconstructs() -> None:
    times = np.arange(0.0, 4.05, 0.05)
    positions = np.zeros((len(times), 3))
    positions[:, 0] = 0.2 * times
    yaw = np.zeros_like(times)
    cfg = LabelConfig(l_xy=0.05, l_yaw=0.15, path_extent=8.0, max_window_seconds=4.0)
    label = build_path_time_label(positions, _quat_from_yaw(yaw), times, 0, cfg)
    assert label.complete
    assert np.all(label.valid == 1)
    query = np.arange(0.0, label.endpoint_time + 1e-9, 0.05)
    recon = reconstruct_at_times(label, query)
    np.testing.assert_allclose(recon[:, 0], 0.2 * query, atol=1e-6)
    np.testing.assert_allclose(recon[:, 1:], 0.0, atol=1e-6)


def test_incomplete_window_has_truthful_terminal_and_mask() -> None:
    times = np.arange(0.0, 0.55, 0.05)
    positions = np.zeros((len(times), 3))
    positions[:, 0] = 0.02 * np.arange(len(times))
    yaw = np.zeros_like(times)
    cfg = LabelConfig(l_xy=0.05, l_yaw=0.15, path_extent=20.0, max_window_seconds=4.0)
    label = build_path_time_label(positions, _quat_from_yaw(yaw), times, 0, cfg)
    assert not label.complete
    valid_count = int(label.valid.sum())
    assert 1 <= valid_count < cfg.num_anchors
    np.testing.assert_allclose(label.poses[valid_count - 1, 0], positions[-1, 0], atol=1e-7)
    np.testing.assert_allclose(label.durations[:valid_count].sum(), times[-1], atol=1e-6)
    assert np.all(label.valid[valid_count:] == 0)


def test_static_window_has_one_truthful_anchor() -> None:
    times = np.arange(0.0, 1.05, 0.05)
    positions = np.zeros((len(times), 3))
    yaw = np.zeros_like(times)
    cfg = LabelConfig(path_extent=10.0, max_window_seconds=1.0)
    label = build_path_time_label(positions, _quat_from_yaw(yaw), times, 0, cfg)
    assert label.valid.sum() == 1
    np.testing.assert_allclose(label.poses, 0.0, atol=1e-9)
    np.testing.assert_allclose(label.durations[0, 0], 1.0, atol=1e-6)

