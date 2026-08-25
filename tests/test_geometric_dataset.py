import numpy as np

from robocasa_act_navigate.adaptive_path import (
    build_fixed_token_path_target,
    world_poses_to_common_origin,
)


def test_world_poses_all_share_one_rotated_origin():
    poses = np.asarray(
        [
            [1.0, 2.0, np.pi / 2],
            [1.0, 3.0, np.pi / 2],
            [0.0, 3.0, np.pi],
        ]
    )
    local = world_poses_to_common_origin(poses)
    np.testing.assert_allclose(local[0], 0.0, atol=1e-12)
    np.testing.assert_allclose(local[1], [1.0, 0.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(local[2], [1.0, 1.0, np.pi / 2], atol=1e-12)


def test_fixed_tokens_preserve_common_origin_absolute_poses():
    path = np.column_stack((np.linspace(0.0, 0.4, 101), np.zeros(101), np.zeros(101)))
    target = build_fixed_token_path_target(
        path,
        num_tokens=32,
        translation_tolerance_m=0.01,
        yaw_tolerance_rad=0.02,
    )
    assert target.anchors.shape == (32, 3)
    assert target.terminal
    np.testing.assert_allclose(target.anchors[-1], path[-1], atol=1e-12)
    assert np.all(np.diff(target.anchors[:, 0]) >= 0.0)


def test_fixed_tokens_do_not_create_plus_minus_pi_discontinuity():
    yaw = np.linspace(0.0, 4.0, 101)
    path = np.column_stack((np.zeros(101), np.zeros(101), yaw))
    target = build_fixed_token_path_target(
        path,
        num_tokens=32,
        translation_tolerance_m=0.01,
        yaw_tolerance_rad=0.02,
    )
    assert target.anchors[-1, 2] == 4.0
    assert np.max(np.abs(np.diff(target.anchors[:, 2]))) < np.pi
