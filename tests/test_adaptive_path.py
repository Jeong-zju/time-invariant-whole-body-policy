import numpy as np

from robocasa_act_navigate.adaptive_path import (
    build_adaptive_path_target,
    build_fixed_token_path_target,
    geometric_rdp_indices,
    piecewise_reconstruction_error,
)


def test_straight_path_is_frequency_independent():
    coarse_progress = np.array([0.0, 0.01, 0.07, 0.4, 0.42, 1.0])
    dense_progress = np.linspace(0.0, 1.0, 101) ** 2
    coarse = np.column_stack((coarse_progress, np.zeros_like(coarse_progress), 0.2 * coarse_progress))
    dense = np.column_stack((dense_progress, np.zeros_like(dense_progress), 0.2 * dense_progress))
    np.testing.assert_array_equal(geometric_rdp_indices(coarse, 0.01, 0.02), [0, len(coarse) - 1])
    np.testing.assert_array_equal(geometric_rdp_indices(dense, 0.01, 0.02), [0, len(dense) - 1])


def test_rdp_reconstruction_respects_physical_bounds():
    progress = np.linspace(0.0, 1.0, 201)
    path = np.column_stack((progress, 0.12 * np.sin(2.0 * np.pi * progress), 0.4 * progress))
    keep = geometric_rdp_indices(path, 0.01, 0.02)
    translation, yaw = piecewise_reconstruction_error(path, keep, 0.01, 0.02)
    assert translation <= 0.01 + 1e-12
    assert yaw <= 0.02 + 1e-12


def test_adaptive_target_uses_common_origin_and_complexity_budget():
    progress = np.linspace(0.0, 1.0, 201)
    path = np.column_stack((progress, 0.2 * np.sin(4.0 * np.pi * progress), 0.5 * progress))
    target = build_adaptive_path_target(
        path,
        max_anchors=4,
        translation_tolerance_m=0.005,
        yaw_tolerance_rad=0.01,
    )
    assert target.valid_count == 4
    assert not target.terminal
    assert target.covered_source_index < len(path) - 1
    assert target.max_translation_error_m <= 0.005 + 1e-12
    assert target.max_yaw_error_rad <= 0.01 + 1e-12
    np.testing.assert_allclose(target.anchors[: target.valid_count], path[target.source_indices[: target.valid_count]])


def test_same_continuous_curve_is_stable_at_10_20_30_40hz():
    dense_count = 1200
    progress = np.linspace(0.0, 1.0, dense_count + 1)
    dense_path = np.column_stack(
        (
            progress,
            0.08 * np.sin(np.pi * progress),
            0.25 * np.sin(0.5 * np.pi * progress),
        )
    )
    retained_counts = []
    for frequency in (10, 20, 30, 40):
        sampled_indices = np.arange(0, dense_count + 1, dense_count // frequency)
        keep = geometric_rdp_indices(dense_path[sampled_indices], 0.01, 0.02)
        translation, yaw = piecewise_reconstruction_error(
            dense_path,
            sampled_indices[keep],
            0.01,
            0.02,
        )
        retained_counts.append(len(keep))
        assert translation <= 0.01
        assert yaw <= 0.02
    assert retained_counts == [5, 5, 5, 5]


def test_fixed_token_target_preserves_knots_without_padding_or_stop():
    progress = np.linspace(0.0, 1.0, 201)
    path = np.column_stack((progress, 0.12 * np.sin(2.0 * np.pi * progress), 0.4 * progress))
    target = build_fixed_token_path_target(
        path,
        num_tokens=32,
        translation_tolerance_m=0.01,
        yaw_tolerance_rad=0.02,
    )
    assert target.anchors.shape == (32, 3)
    assert target.terminal
    assert target.covered_source_index == len(path) - 1
    assert target.max_translation_error_m <= 0.01 + 1e-12
    assert target.max_yaw_error_rad <= 0.02 + 1e-12
    for source_index in target.retained_source_indices:
        distance = np.linalg.norm(target.anchors - path[source_index], axis=1)
        assert float(distance.min()) <= 1e-12
