"""Small, dependency-free SE(2) helpers.

Poses are ``[x, y, yaw]`` and twists are body-frame ``[vx, vy, omega]``.
Quaternions use RoboSuite's verified ``[x, y, z, w]`` convention.
"""

from __future__ import annotations

import numpy as np


def wrap_angle(angle: np.ndarray | float) -> np.ndarray | float:
    """Wrap angles to [-pi, pi)."""
    return (np.asarray(angle) + np.pi) % (2.0 * np.pi) - np.pi


def quat_xyzw_to_yaw(quat: np.ndarray) -> np.ndarray:
    """Extract world-frame yaw from ``[..., 4]`` xyzw quaternions."""
    quat = np.asarray(quat, dtype=np.float64)
    if quat.shape[-1] != 4:
        raise ValueError(f"Expected xyzw quaternion with last dim 4, got {quat.shape}")
    x, y, z, w = np.moveaxis(quat, -1, 0)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return np.arctan2(siny_cosp, cosy_cosp)


def compose(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compose poses: result transforms through ``a`` and then local pose ``b``."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    c = np.cos(a[..., 2])
    s = np.sin(a[..., 2])
    x = a[..., 0] + c * b[..., 0] - s * b[..., 1]
    y = a[..., 1] + s * b[..., 0] + c * b[..., 1]
    yaw = wrap_angle(a[..., 2] + b[..., 2])
    return np.stack([x, y, yaw], axis=-1)


def inverse(pose: np.ndarray) -> np.ndarray:
    """Invert one or more SE(2) poses."""
    pose = np.asarray(pose, dtype=np.float64)
    c = np.cos(pose[..., 2])
    s = np.sin(pose[..., 2])
    x = -c * pose[..., 0] - s * pose[..., 1]
    y = s * pose[..., 0] - c * pose[..., 1]
    return np.stack([x, y, wrap_angle(-pose[..., 2])], axis=-1)


def between(origin: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Express ``target`` in the coordinate frame of ``origin``."""
    return compose(inverse(origin), target)


def exp(twist: np.ndarray) -> np.ndarray:
    """SE(2) exponential for body-frame exponential coordinates."""
    twist = np.asarray(twist, dtype=np.float64)
    vx, vy, omega = np.moveaxis(twist, -1, 0)
    small = np.abs(omega) < 1e-8
    safe_omega = np.where(small, 1.0, omega)
    a = np.where(small, 1.0 - omega * omega / 6.0, np.sin(omega) / safe_omega)
    b = np.where(
        small,
        omega / 2.0 - omega**3 / 24.0,
        (1.0 - np.cos(omega)) / safe_omega,
    )
    x = a * vx - b * vy
    y = b * vx + a * vy
    return np.stack([x, y, wrap_angle(omega)], axis=-1)


def log(pose: np.ndarray) -> np.ndarray:
    """SE(2) logarithm returning body-frame exponential coordinates."""
    pose = np.asarray(pose, dtype=np.float64)
    x, y, omega = np.moveaxis(pose, -1, 0)
    small = np.abs(omega) < 1e-8
    safe_omega = np.where(small, 1.0, omega)
    a = np.where(small, 1.0 - omega * omega / 6.0, np.sin(omega) / safe_omega)
    b = np.where(
        small,
        omega / 2.0 - omega**3 / 24.0,
        (1.0 - np.cos(omega)) / safe_omega,
    )
    det = np.maximum(a * a + b * b, 1e-12)
    vx = (a * x + b * y) / det
    vy = (-b * x + a * y) / det
    return np.stack([vx, vy, omega], axis=-1)


def interpolate(a: np.ndarray, b: np.ndarray, alpha: float) -> np.ndarray:
    """Geodesically interpolate from pose ``a`` to ``b``."""
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")
    return compose(a, exp(alpha * log(between(a, b))))


def world_xy_quat_to_se2(position: np.ndarray, quat_xyzw: np.ndarray) -> np.ndarray:
    """Convert measured world position/quaternion arrays to planar poses."""
    position = np.asarray(position, dtype=np.float64)
    if position.shape[-1] < 2:
        raise ValueError(f"Expected position with at least xy, got {position.shape}")
    yaw = quat_xyzw_to_yaw(quat_xyzw)
    return np.stack([position[..., 0], position[..., 1], yaw], axis=-1)

