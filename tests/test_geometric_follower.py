import numpy as np

from robocasa_act_navigate.geometric_follower import BaseRateCalibration, MeasuredPosePathFollower


IDENTITY_QUATERNION_XYZW = np.array([0.0, 0.0, 0.0, 1.0])


def test_plan_explicitly_starts_at_current_common_origin():
    follower = MeasuredPosePathFollower(BaseRateCalibration.identity())
    local = np.asarray([[0.1, 0.0, 0.0], [0.2, 0.0, 0.0]])
    follower.set_plan(local, np.array([2.0, -1.0, 0.0]), IDENTITY_QUATERNION_XYZW)
    np.testing.assert_allclose(follower.world_path[0], [2.0, -1.0, 0.0])
    np.testing.assert_allclose(follower.world_path[1:, :2], [[2.1, -1.0], [2.2, -1.0]])


def test_world_translation_does_not_change_body_command():
    local = np.asarray([[0.1, 0.0, 0.0], [0.2, 0.0, 0.0]])
    commands = []
    for origin in (np.array([0.0, 0.0, 0.0]), np.array([3.0, -2.0, 0.0])):
        follower = MeasuredPosePathFollower(BaseRateCalibration.identity())
        follower.set_plan(local, origin, IDENTITY_QUATERNION_XYZW)
        command, _ = follower.command(origin, IDENTITY_QUATERNION_XYZW)
        commands.append(command)
    np.testing.assert_allclose(commands[0], commands[1], atol=1e-7)
    assert commands[0][0] > 0.0


def test_command_rate_limit_is_applied_at_30hz():
    follower = MeasuredPosePathFollower(
        BaseRateCalibration.identity(),
        control_hz=30.0,
        max_command_delta_per_tick=0.1,
    )
    follower.set_plan(np.asarray([[1.0, 0.0, 0.0]]), np.zeros(3), IDENTITY_QUATERNION_XYZW)
    first, _ = follower.command(np.zeros(3), IDENTITY_QUATERNION_XYZW)
    second, _ = follower.command(np.zeros(3), IDENTITY_QUATERNION_XYZW)
    np.testing.assert_allclose(first[:3], [0.1, 0.0, 0.0], atol=1e-7)
    np.testing.assert_allclose(second[:3], [0.2, 0.0, 0.0], atol=1e-7)


def test_goal_stops_without_predicting_time_or_rate():
    follower = MeasuredPosePathFollower(BaseRateCalibration.identity())
    follower.set_plan(np.asarray([[0.2, 0.0, 0.0]]), np.zeros(3), IDENTITY_QUATERNION_XYZW)
    command, info = follower.command(np.array([0.2, 0.0, 0.0]), IDENTITY_QUATERNION_XYZW)
    np.testing.assert_allclose(command, 0.0)
    assert info["done"] is True


def test_replan_can_preserve_command_continuity():
    follower = MeasuredPosePathFollower(
        BaseRateCalibration.identity(),
        max_command_delta_per_tick=0.1,
    )
    path = np.asarray([[1.0, 0.0, 0.0]])
    follower.set_plan(path, np.zeros(3), IDENTITY_QUATERNION_XYZW)
    first, _ = follower.command(np.zeros(3), IDENTITY_QUATERNION_XYZW)
    follower.set_plan(
        path,
        np.zeros(3),
        IDENTITY_QUATERNION_XYZW,
        preserve_last_command=True,
    )
    second, _ = follower.command(np.zeros(3), IDENTITY_QUATERNION_XYZW)
    np.testing.assert_allclose(first[:3], [0.1, 0.0, 0.0], atol=1e-7)
    np.testing.assert_allclose(second[:3], [0.2, 0.0, 0.0], atol=1e-7)
