"""Small, dependency-free SE(2) utilities used by Phase 1.

Poses are stored as ``[..., (x, y, yaw)]``.  Tangent vectors use the same
layout and express translation in the local/body frame.  The implementation is
NumPy-only so the data gate and controller tests can run without Isaac Sim.
"""

from __future__ import annotations

import math

import numpy as np


_EPS = 1e-9


def _as_pose(value: np.ndarray | list[float] | tuple[float, ...], name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape[-1:] != (3,):
        raise ValueError(f"{name} must end in dimension 3, got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values")
    return array


def wrap_angle(angle: float | np.ndarray) -> float | np.ndarray:
    """Wrap radians to ``[-pi, pi)`` while preserving scalar inputs."""
    wrapped = (np.asarray(angle, dtype=np.float64) + math.pi) % (2.0 * math.pi) - math.pi
    if np.ndim(angle) == 0:
        return float(wrapped)
    return wrapped


def compose(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Compose two SE(2) poses, broadcasting leading dimensions."""
    left = _as_pose(left, "left")
    right = _as_pose(right, "right")
    left, right = np.broadcast_arrays(left, right)
    c = np.cos(left[..., 2])
    s = np.sin(left[..., 2])
    result = np.empty_like(left, dtype=np.float64)
    result[..., 0] = left[..., 0] + c * right[..., 0] - s * right[..., 1]
    result[..., 1] = left[..., 1] + s * right[..., 0] + c * right[..., 1]
    result[..., 2] = wrap_angle(left[..., 2] + right[..., 2])
    return result


def inverse(pose: np.ndarray) -> np.ndarray:
    """Return the group inverse of an SE(2) pose."""
    pose = _as_pose(pose, "pose")
    c = np.cos(pose[..., 2])
    s = np.sin(pose[..., 2])
    result = np.empty_like(pose, dtype=np.float64)
    result[..., 0] = -c * pose[..., 0] - s * pose[..., 1]
    result[..., 1] = s * pose[..., 0] - c * pose[..., 1]
    result[..., 2] = wrap_angle(-pose[..., 2])
    return result


def relative(origin: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Express ``target`` in ``origin``'s local frame."""
    return compose(inverse(origin), target)


def exp(tangent: np.ndarray) -> np.ndarray:
    """SE(2) exponential map for body-frame tangent coordinates."""
    tangent = _as_pose(tangent, "tangent")
    vx = tangent[..., 0]
    vy = tangent[..., 1]
    omega = tangent[..., 2]
    small = np.abs(omega) < _EPS
    a = np.empty_like(omega)
    b = np.empty_like(omega)
    a[small] = 1.0 - omega[small] ** 2 / 6.0
    b[small] = omega[small] / 2.0 - omega[small] ** 3 / 24.0
    a[~small] = np.sin(omega[~small]) / omega[~small]
    b[~small] = (1.0 - np.cos(omega[~small])) / omega[~small]
    result = np.empty_like(tangent, dtype=np.float64)
    result[..., 0] = a * vx - b * vy
    result[..., 1] = b * vx + a * vy
    result[..., 2] = wrap_angle(omega)
    return result


def log(pose: np.ndarray) -> np.ndarray:
    """SE(2) logarithm map returning body-frame tangent coordinates."""
    pose = _as_pose(pose, "pose")
    x = pose[..., 0]
    y = pose[..., 1]
    omega = np.asarray(wrap_angle(pose[..., 2]), dtype=np.float64)
    small = np.abs(omega) < _EPS
    a = np.empty_like(omega)
    b = np.empty_like(omega)
    a[small] = 1.0 - omega[small] ** 2 / 6.0
    b[small] = omega[small] / 2.0 - omega[small] ** 3 / 24.0
    a[~small] = np.sin(omega[~small]) / omega[~small]
    b[~small] = (1.0 - np.cos(omega[~small])) / omega[~small]
    determinant = a * a + b * b
    if np.any(determinant < _EPS):
        raise ValueError("SE(2) logarithm is ill-conditioned for the supplied pose")
    result = np.empty_like(pose, dtype=np.float64)
    result[..., 0] = (a * x + b * y) / determinant
    result[..., 1] = (-b * x + a * y) / determinant
    result[..., 2] = omega
    return result


def interpolate(start: np.ndarray, end: np.ndarray, fraction: float | np.ndarray) -> np.ndarray:
    """Interpolate along the SE(2) geodesic from ``start`` to ``end``."""
    start = _as_pose(start, "start")
    end = _as_pose(end, "end")
    fraction = np.asarray(fraction, dtype=np.float64)
    if np.any((fraction < 0.0) | (fraction > 1.0)):
        raise ValueError("fraction must lie in [0, 1]")
    delta = log(relative(start, end))
    return compose(start, exp(delta * np.expand_dims(fraction, axis=-1)))
