"""Frequency-independent, loss-bounded compression of measured SE(2) paths."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def wrap_angle(value: np.ndarray | float) -> np.ndarray:
    return (np.asarray(value) + np.pi) % (2.0 * np.pi) - np.pi


def _unwrapped(path: np.ndarray) -> np.ndarray:
    value = np.asarray(path, dtype=np.float64).copy()
    if value.ndim != 2 or value.shape[1] != 3 or len(value) == 0:
        raise ValueError(f"expected nonempty SE(2) path (N,3), got {value.shape}")
    if not np.isfinite(value).all():
        raise ValueError("path contains non-finite values")
    value[:, 2] = np.unwrap(value[:, 2])
    return value


def world_poses_to_common_origin(world_poses: np.ndarray, start: int = 0) -> np.ndarray:
    """Express every future measured pose in the one current-base frame.

    The returned row ``j`` is ``T_start^-1 T_j``.  It is deliberately not an
    incremental pose between adjacent samples, and the first row is identity.
    """

    poses = _unwrapped(world_poses)
    if start < 0 or start >= len(poses):
        raise IndexError(start)
    future = poses[start:]
    origin = poses[start]
    delta = future[:, :2] - origin[:2]
    cosine = np.cos(origin[2])
    sine = np.sin(origin[2])
    result = np.column_stack(
        (
            cosine * delta[:, 0] + sine * delta[:, 1],
            -sine * delta[:, 0] + cosine * delta[:, 1],
            wrap_angle(future[:, 2] - origin[2]),
        )
    )
    result[0] = 0.0
    return result


def _segment_errors(
    path: np.ndarray,
    left: int,
    right: int,
    translation_tolerance_m: float,
    yaw_tolerance_rad: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return geometric point-to-chord translation and yaw errors."""

    if right < left:
        raise ValueError("right endpoint precedes left endpoint")
    points = path[left : right + 1]
    if len(points) <= 2:
        return np.zeros(len(points)), np.zeros(len(points))
    yaw_radius_m = translation_tolerance_m / yaw_tolerance_rad
    start = path[left].copy()
    finish = path[right].copy()
    start[2] *= yaw_radius_m
    finish[2] *= yaw_radius_m
    embedded = points.copy()
    embedded[:, 2] *= yaw_radius_m
    chord = finish - start
    denominator = float(np.dot(chord, chord))
    if denominator <= 1e-18:
        alpha = np.zeros(len(points), dtype=np.float64)
    else:
        alpha = np.clip((embedded - start) @ chord / denominator, 0.0, 1.0)
    interpolated_xy = path[left, :2] + alpha[:, None] * (path[right, :2] - path[left, :2])
    interpolated_yaw = path[left, 2] + alpha * (path[right, 2] - path[left, 2])
    translation = np.linalg.norm(points[:, :2] - interpolated_xy, axis=-1)
    yaw = np.abs(points[:, 2] - interpolated_yaw)
    return translation, yaw


def geometric_rdp_indices(
    path: np.ndarray,
    translation_tolerance_m: float,
    yaw_tolerance_rad: float,
) -> np.ndarray:
    """Select SE(2) knots using only geometry and physical error bounds.

    No timestamp, frame index, speed, duration, or sampling frequency enters
    the selection rule.
    """

    if translation_tolerance_m <= 0.0 or yaw_tolerance_rad <= 0.0:
        raise ValueError("physical tolerances must be positive")
    value = _unwrapped(path)
    if len(value) == 1:
        return np.array([0], dtype=np.int64)
    keep = {0, len(value) - 1}
    stack = [(0, len(value) - 1)]
    while stack:
        left, right = stack.pop()
        if right <= left + 1:
            continue
        translation, yaw = _segment_errors(
            value,
            left,
            right,
            translation_tolerance_m,
            yaw_tolerance_rad,
        )
        normalized = np.maximum(
            translation / translation_tolerance_m,
            yaw / yaw_tolerance_rad,
        )
        normalized[[0, -1]] = 0.0
        split = int(np.argmax(normalized))
        if normalized[split] > 1.0:
            absolute = left + split
            keep.add(absolute)
            stack.extend(((left, absolute), (absolute, right)))
    return np.asarray(sorted(keep), dtype=np.int64)


def piecewise_reconstruction_error(
    path: np.ndarray,
    keep_indices: np.ndarray,
    translation_tolerance_m: float,
    yaw_tolerance_rad: float,
) -> tuple[float, float]:
    """Measure maximum raw-path error against the retained geometric curve."""

    value = _unwrapped(path)
    keep = np.asarray(keep_indices, dtype=np.int64)
    if keep.ndim != 1 or len(keep) == 0 or keep[0] != 0 or keep[-1] != len(value) - 1:
        raise ValueError("keep_indices must be sorted and cover both path endpoints")
    if np.any(np.diff(keep) <= 0):
        raise ValueError("keep_indices must be strictly increasing")
    maximum_translation = 0.0
    maximum_yaw = 0.0
    for left, right in zip(keep[:-1], keep[1:]):
        translation, yaw = _segment_errors(
            value,
            int(left),
            int(right),
            translation_tolerance_m,
            yaw_tolerance_rad,
        )
        maximum_translation = max(maximum_translation, float(translation.max(initial=0.0)))
        maximum_yaw = max(maximum_yaw, float(yaw.max(initial=0.0)))
    return maximum_translation, maximum_yaw


@dataclass(frozen=True)
class AdaptivePathTarget:
    """Fixed-capacity target whose covered horizon is set by path complexity."""

    anchors: np.ndarray
    is_pad: np.ndarray
    source_indices: np.ndarray
    terminal: bool
    covered_source_index: int
    max_translation_error_m: float
    max_yaw_error_rad: float

    @property
    def valid_count(self) -> int:
        return int((~self.is_pad).sum())


@dataclass(frozen=True)
class FixedTokenPathTarget:
    """Exactly K geometric poses, with no time target, padding, or STOP token."""

    anchors: np.ndarray
    terminal: bool
    covered_source_index: int
    retained_source_indices: np.ndarray
    max_translation_error_m: float
    max_yaw_error_rad: float


def densify_geometric_knots(
    knots_with_origin: np.ndarray,
    *,
    num_future_tokens: int,
    yaw_radius_m: float,
) -> np.ndarray:
    """Preserve every knot and bisect longest SE(2) chords until K tokens exist."""

    # These are sparse indices from an already-unwrapped source curve.  Calling
    # np.unwrap again after sparsification is incorrect: two valid retained
    # knots may differ by more than pi because intermediate turns were removed.
    value = np.asarray(knots_with_origin, dtype=np.float64).copy()
    if value.ndim != 2 or value.shape[1] != 3 or len(value) == 0:
        raise ValueError(f"expected nonempty SE(2) knots (N,3), got {value.shape}")
    if not np.isfinite(value).all():
        raise ValueError("knots contain non-finite values")
    if num_future_tokens <= 0 or yaw_radius_m <= 0.0:
        raise ValueError("token count and yaw radius must be positive")
    if len(value) > num_future_tokens + 1:
        raise ValueError("more retained knots than the fixed token budget")
    vertices = [pose.copy() for pose in value]
    while len(vertices) < num_future_tokens + 1:
        if len(vertices) == 1:
            vertices.append(vertices[0].copy())
            continue
        array = np.asarray(vertices)
        metric = array.copy()
        metric[:, 2] *= yaw_radius_m
        lengths = np.linalg.norm(np.diff(metric, axis=0), axis=-1)
        segment = int(np.argmax(lengths))
        midpoint = 0.5 * (array[segment] + array[segment + 1])
        vertices.insert(segment + 1, midpoint)
    # Keep yaw continuously unwrapped along the measured path.  Wrapping every
    # token independently creates artificial +/-pi jumps and makes ordinary
    # regression penalize equivalent orientations by almost 2*pi.
    return np.asarray(vertices[1:], dtype=np.float64)


def build_fixed_token_path_target(
    common_origin_path: np.ndarray,
    *,
    num_tokens: int,
    translation_tolerance_m: float,
    yaw_tolerance_rad: float,
) -> FixedTokenPathTarget:
    """Build a fixed ACT target while retaining loss-bounded adaptive geometry.

    The first ``num_tokens`` future RDP knots determine the adaptive horizon.
    If fewer are needed, longest geometric chords are bisected until exactly K
    output poses exist.  Every retained RDP knot remains in the final path, so
    densification cannot weaken the reconstruction-error guarantee.
    """

    if num_tokens <= 0:
        raise ValueError("num_tokens must be positive")
    value = _unwrapped(common_origin_path)
    keep = geometric_rdp_indices(value, translation_tolerance_m, yaw_tolerance_rad)
    retained = keep[1 : num_tokens + 1]
    if len(retained):
        covered = int(retained[-1])
        reconstruction_keep = np.concatenate((np.asarray([0], dtype=np.int64), retained))
        translation_error, yaw_error = piecewise_reconstruction_error(
            value[: covered + 1],
            reconstruction_keep,
            translation_tolerance_m,
            yaw_tolerance_rad,
        )
        vertices = value[reconstruction_keep]
    else:
        covered = 0
        translation_error = 0.0
        yaw_error = 0.0
        vertices = value[:1]
    anchors = densify_geometric_knots(
        vertices,
        num_future_tokens=num_tokens,
        yaw_radius_m=translation_tolerance_m / yaw_tolerance_rad,
    )
    return FixedTokenPathTarget(
        anchors=anchors,
        terminal=bool(len(keep) - 1 <= num_tokens),
        covered_source_index=covered,
        retained_source_indices=retained,
        max_translation_error_m=translation_error,
        max_yaw_error_rad=yaw_error,
    )


def build_adaptive_path_target(
    common_origin_path: np.ndarray,
    *,
    max_anchors: int,
    translation_tolerance_m: float,
    yaw_tolerance_rad: float,
) -> AdaptivePathTarget:
    """Compress the future path until the fixed geometric-token budget is full.

    The current identity pose is implicit and consumes no output token.  If the
    complete future route needs more than ``max_anchors`` retained knots, the
    target ends at the last knot that fits.  Thus the physical horizon adapts
    to curvature and path complexity instead of time or distance.
    """

    if max_anchors <= 0:
        raise ValueError("max_anchors must be positive")
    value = _unwrapped(common_origin_path)
    keep = geometric_rdp_indices(value, translation_tolerance_m, yaw_tolerance_rad)
    selected = keep[1 : max_anchors + 1]
    anchors = np.zeros((max_anchors, 3), dtype=np.float64)
    is_pad = np.ones(max_anchors, dtype=bool)
    source_indices = np.full(max_anchors, -1, dtype=np.int64)
    if len(selected):
        anchors[: len(selected)] = value[selected]
        anchors[: len(selected), 2] = wrap_angle(anchors[: len(selected), 2])
        anchors[len(selected) :] = anchors[len(selected) - 1]
        is_pad[: len(selected)] = False
        source_indices[: len(selected)] = selected
        covered = int(selected[-1])
        covered_keep = np.concatenate((np.array([0], dtype=np.int64), selected))
        translation_error, yaw_error = piecewise_reconstruction_error(
            value[: covered + 1],
            covered_keep,
            translation_tolerance_m,
            yaw_tolerance_rad,
        )
    else:
        covered = 0
        translation_error = 0.0
        yaw_error = 0.0
    return AdaptivePathTarget(
        anchors=anchors,
        is_pad=is_pad,
        source_indices=source_indices,
        terminal=bool(len(keep) - 1 <= max_anchors),
        covered_source_index=covered,
        max_translation_error_m=translation_error,
        max_yaw_error_rad=yaw_error,
    )
