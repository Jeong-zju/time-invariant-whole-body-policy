import numpy as np

from lpwb.geometry import (
    local_base_pose,
    quaternion_between,
    quaternion_to_rotvec,
    rotvec_to_quaternion,
)
from lpwb.labels import (
    LabelConfig,
    build_path_time_label,
    build_pose_time_label,
    event_signature,
)


def make_chunk():
    state = np.zeros((33, 16), dtype=np.float64)
    state[:, 3:7] = [0.0, 0.0, 0.0, 1.0]
    state[:, 10:14] = [0.0, 0.0, 0.0, 1.0]
    action = np.zeros((32, 12), dtype=np.float64)
    action[:, 4] = 1.0
    action[:, 11] = -1.0
    timestamp = np.arange(33, dtype=np.float64) / 20.0
    return state, action, timestamp


def test_quaternion_rotvec_round_trip():
    vector = np.array([0.2, -0.4, 0.1])
    recovered = quaternion_to_rotvec(rotvec_to_quaternion(vector))
    np.testing.assert_allclose(recovered, vector, atol=1e-10)


def test_stationary_chunk_has_truthful_duration():
    state, action, timestamp = make_chunk()
    label = build_path_time_label(state, action, timestamp)
    assert np.isclose(label.durations.sum(), 1.6)
    np.testing.assert_allclose(label.base_local, 0.0)
    np.testing.assert_allclose(label.eef_position_delta, 0.0)


def test_base_label_uses_measured_pose_not_command_integral():
    state, action, timestamp = make_chunk()
    action[:, :4] = 1.0
    label = build_path_time_label(state, action, timestamp)
    np.testing.assert_allclose(label.base_local, 0.0)


def test_eef_a_b_a_is_not_dropped():
    state, action, timestamp = make_chunk()
    state[:, 7] = np.sin(np.linspace(0.0, np.pi, 33))
    label = build_path_time_label(state, action, timestamp)
    assert label.eef_position_delta[:, 0].max() > 0.99
    assert abs(label.eef_position_delta[-1, 0]) < 1e-6


def test_discrete_event_signature_is_preserved():
    state, action, timestamp = make_chunk()
    state[:, 0] = np.linspace(0.0, 1.0, 33) ** 3
    action[8:16, 11] = 1.0
    action[16:, 11] = -1.0
    action[20:, 4] = -1.0
    label = build_path_time_label(state, action, timestamp)
    assert event_signature(label.gripper) == event_signature(action[:, 11])
    assert event_signature(label.control_mode) == event_signature(action[:, 4])


def test_same_geometry_different_timing_changes_time_not_path():
    state_a, action_a, time_a = make_chunk()
    state_b, action_b, time_b = make_chunk()
    progress_a = np.linspace(0.0, 1.0, 33)
    progress_b = np.linspace(0.0, 1.0, 33) ** 2
    state_a[:, 0] = progress_a
    state_b[:, 0] = progress_b
    label_a = build_path_time_label(state_a, action_a, time_a)
    label_b = build_path_time_label(state_b, action_b, time_b)
    np.testing.assert_allclose(label_a.base_local, label_b.base_local, atol=1e-6)
    assert not np.allclose(label_a.durations, label_b.durations, atol=1e-3)


def test_pose_time_is_exactly_frame_aligned():
    state, action, timestamp = make_chunk()
    progress = np.linspace(0.0, 1.0, 33) ** 2
    state[:, 0] = progress
    state[:, 7] = np.sin(np.linspace(0.0, 2.0 * np.pi, 33))
    label = build_pose_time_label(state, action, timestamp)
    np.testing.assert_allclose(
        label.base_local,
        local_base_pose(state[:, 0:3], state[:, 3:7])[1:],
        atol=1e-7,
    )
    np.testing.assert_allclose(
        label.eef_position_delta,
        state[1:, 7:10] - state[0, 7:10],
        atol=1e-7,
    )
    np.testing.assert_allclose(label.durations, 0.05, atol=1e-7)
    np.testing.assert_array_equal(label.source_indices, np.arange(32))


def test_pose_time_never_integrates_commands():
    state, action, timestamp = make_chunk()
    action[:, :4] = 1.0
    action[:, 5:11] = 1.0
    label = build_pose_time_label(state, action, timestamp)
    np.testing.assert_allclose(label.base_local, 0.0)
    np.testing.assert_allclose(label.eef_position_delta, 0.0)
    np.testing.assert_allclose(label.eef_rotation_delta, 0.0)


def test_pose_time_preserves_every_discrete_frame():
    state, action, timestamp = make_chunk()
    action[3:7, 11] = 1.0
    action[7:10, 11] = -1.0
    action[10:12, 11] = 1.0
    action[5:9, 4] = -1.0
    label = build_pose_time_label(state, action, timestamp)
    np.testing.assert_array_equal(label.gripper, action[:, 11])
    np.testing.assert_array_equal(label.control_mode, action[:, 4])
