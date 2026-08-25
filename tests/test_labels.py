import numpy as np

from lp_act.labels import LabelConfig, MetricConfig, build_path_label
from lp_act.parquet_data import TurningData
from lp_act.reconstruction import oracle_metrics, reconstruct_at_times


def _synthetic_data(static: bool = False) -> TurningData:
    count = 181
    timestamps = np.arange(count, dtype=np.float64) / 30.0
    actions = np.zeros((count, 23), dtype=np.float32)
    states = np.zeros((count, 61), dtype=np.float32)
    if not static:
        states[:, 0] = 0.3
        actions[:, 0] = 0.3
        actions[:, 7] = np.linspace(0.0, 0.5, count)
    actions[:, 14] = 1.0
    actions[:, 22] = 1.0
    return TurningData(
        actions=actions,
        states=states,
        timestamps=timestamps,
        frame_indices=np.arange(count),
        episode_indices=np.zeros(count, dtype=np.int64),
        global_indices=np.arange(count),
        episode_bounds={0: (0, count)},
    )


def _metric() -> MetricConfig:
    return MetricConfig(
        xy_scale=0.01,
        theta_scale=0.01,
        joint_scales=tuple([0.01] * 18),
        joint_weight=1 / 18,
        path_length=10.0,
    )


def test_straight_path_label_and_reconstruction() -> None:
    data = _synthetic_data()
    label = build_path_label(data, 0, _metric(), LabelConfig(extent_mode="attained"))
    assert label.target.shape == (32, 24)
    assert not label.is_pad.any()
    reconstruction = reconstruct_at_times(label, label.raw_times)
    np.testing.assert_allclose(reconstruction.base_poses, label.raw_base_poses, atol=1e-8)
    metrics = oracle_metrics(label, data.states[: label.end_index, :3], data.actions[: label.end_index, :3])
    assert metrics["base_translation_rmse_m"] < 1e-8
    assert metrics["base_twist_measured_rmse"] < 5e-8


def test_static_window_is_finite() -> None:
    data = _synthetic_data(static=True)
    label = build_path_label(data, 0, _metric(), LabelConfig())
    assert label.static
    assert np.isfinite(label.target).all()
    assert np.all(label.target[:, :3] == 0.0)
    assert not label.is_pad.any()


def test_fixed_extent_records_padding() -> None:
    data = _synthetic_data()
    metric = MetricConfig(
        xy_scale=0.01,
        theta_scale=0.01,
        joint_scales=tuple([0.01] * 18),
        joint_weight=1 / 18,
        path_length=1000.0,
    )
    label = build_path_label(data, 170, metric, LabelConfig(extent_mode="fixed"))
    assert label.is_pad.any()
    assert label.valid_count >= 1
    assert not label.is_pad[label.valid_count - 1]
    np.testing.assert_allclose(label.anchor_times[label.valid_count - 1], label.raw_times[-1])
    reconstruction = reconstruct_at_times(label, label.raw_times)
    np.testing.assert_allclose(reconstruction.base_poses, label.raw_base_poses, atol=1e-8)


def test_fixed_extent_clips_overshooting_raw_interval() -> None:
    data = _synthetic_data()
    metric = MetricConfig(
        xy_scale=0.01,
        theta_scale=0.01,
        joint_scales=tuple([0.01] * 18),
        joint_weight=1 / 18,
        path_length=10.05,
    )
    label = build_path_label(data, 0, metric, LabelConfig(extent_mode="fixed"))
    assert label.reached_path_limit
    np.testing.assert_allclose(label.attained_length, metric.path_length)
    np.testing.assert_allclose(label.anchor_times[-1], label.raw_times[-1])
    reconstruction = reconstruct_at_times(label, label.raw_times)
    np.testing.assert_allclose(reconstruction.base_poses, label.raw_base_poses, atol=1e-8)
