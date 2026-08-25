import numpy as np

from robocasa_lp_act.data import RoboCasaData
from robocasa_lp_act.execution import ControllerCalibration, decode_lp_chunk
from robocasa_lp_act.labels import LabelConfig, MetricConfig, build_base_path_label
from robocasa_lp_act.schema import NONBASE_ACTION


def _data(*, moving: bool = True, arm_scale: float = 1.0) -> RoboCasaData:
    count = 101
    timestamps = np.arange(count, dtype=np.float64) / 20.0
    actions = np.zeros((count, 12), dtype=np.float64)
    actions[:, 5] = arm_scale * np.sin(np.arange(count) / 3.0)
    states = np.zeros((count, 16), dtype=np.float64)
    states[:, 6] = 1.0
    poses = np.zeros((count, 3), dtype=np.float64)
    if moving:
        poses[:, 0] = np.arange(count) * 0.01
        states[:, 0] = poses[:, 0]
        actions[:, 0] = 0.2
    return RoboCasaData(
        actions=actions,
        states=states,
        timestamps=timestamps,
        frame_indices=np.arange(count),
        episode_indices=np.zeros(count, dtype=np.int64),
        global_indices=np.arange(count),
        base_world_poses=poses,
        episode_bounds={0: (0, count)},
    )


def _metric() -> MetricConfig:
    return MetricConfig(xy_scale=0.01, theta_scale=0.05, path_length=32.0)


def test_arm_motion_cannot_change_base_phase() -> None:
    low = build_base_path_label(_data(arm_scale=0.0), 0, _metric())
    high = build_base_path_label(_data(arm_scale=100.0), 0, _metric())
    np.testing.assert_allclose(low.base_anchors, high.base_anchors)
    np.testing.assert_allclose(low.anchor_times, high.anchor_times)
    np.testing.assert_allclose(low.anchor_path_positions, high.anchor_path_positions)


def test_static_base_remains_static_despite_arm_motion() -> None:
    data = _data(moving=False, arm_scale=100.0)
    label = build_base_path_label(data, 0, _metric(), LabelConfig())
    assert label.static
    np.testing.assert_allclose(label.base_anchors, 0.0)
    assert np.all(np.diff(label.anchor_times) > 0.0)


def test_decoder_reconstructs_pose_and_preserves_time_indexed_upper() -> None:
    data = _data()
    label = build_base_path_label(data, 0, _metric())
    upper = data.actions[:32, NONBASE_ACTION]
    target = label.hybrid_target(upper)
    decoded = decode_lp_chunk(
        target,
        horizon_steps=32,
        calibration=ControllerCalibration(vx_per_action=1.0, vy_per_action=1.0, yaw_rate_per_action=1.0),
    )
    np.testing.assert_allclose(decoded.boundary_base_poses[:, 0], np.arange(33) * 0.01, atol=1e-8)
    np.testing.assert_allclose(decoded.boundary_base_poses[:, 1:], 0.0, atol=1e-8)
    np.testing.assert_allclose(decoded.actions[:, 3:12], upper, atol=1e-8)
