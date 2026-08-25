"""Numerically stable SE(2) operations using poses ``[x, y, yaw]``."""

from __future__ import annotations

import numpy as np


def wrap_angle(angle: np.ndarray | float) -> np.ndarray:
    angle = np.asarray(angle)
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def _ab(theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    theta = np.asarray(theta, dtype=np.float64)
    small = np.abs(theta) < 1e-6
    theta2 = theta * theta
    a = np.empty_like(theta)
    b = np.empty_like(theta)
    np.divide(np.sin(theta), theta, out=a, where=~small)
    np.divide(1.0 - np.cos(theta), theta, out=b, where=~small)
    a = np.where(small, 1.0 - theta2 / 6.0 + theta2 * theta2 / 120.0, a)
    b = np.where(small, theta / 2.0 - theta * theta2 / 24.0 + theta * theta2 * theta2 / 720.0, b)
    return a, b


def exp(twist: np.ndarray) -> np.ndarray:
    """Map integrated body twist ``[rho_x, rho_y, yaw]`` to an SE(2) pose."""

    twist = np.asarray(twist, dtype=np.float64)
    if twist.shape[-1] != 3:
        raise ValueError(f"Expected (..., 3) twist, got {twist.shape}")
    rho_x, rho_y, theta = np.moveaxis(twist, -1, 0)
    a, b = _ab(theta)
    x = a * rho_x - b * rho_y
    y = b * rho_x + a * rho_y
    return np.stack([x, y, wrap_angle(theta)], axis=-1)


def log(pose: np.ndarray) -> np.ndarray:
    """Map an SE(2) pose to integrated body-twist coordinates."""

    pose = np.asarray(pose, dtype=np.float64)
    if pose.shape[-1] != 3:
        raise ValueError(f"Expected (..., 3) pose, got {pose.shape}")
    x, y, theta = np.moveaxis(pose, -1, 0)
    theta = wrap_angle(theta)
    a, b = _ab(theta)
    denom = np.maximum(a * a + b * b, np.finfo(np.float64).eps)
    rho_x = (a * x + b * y) / denom
    rho_y = (-b * x + a * y) / denom
    return np.stack([rho_x, rho_y, theta], axis=-1)


def compose(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Compose poses as ``T_left @ T_right``."""

    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    c = np.cos(left[..., 2])
    s = np.sin(left[..., 2])
    x = left[..., 0] + c * right[..., 0] - s * right[..., 1]
    y = left[..., 1] + s * right[..., 0] + c * right[..., 1]
    yaw = wrap_angle(left[..., 2] + right[..., 2])
    return np.stack([x, y, yaw], axis=-1)


def inverse(pose: np.ndarray) -> np.ndarray:
    pose = np.asarray(pose, dtype=np.float64)
    c = np.cos(pose[..., 2])
    s = np.sin(pose[..., 2])
    x = -c * pose[..., 0] - s * pose[..., 1]
    y = s * pose[..., 0] - c * pose[..., 1]
    return np.stack([x, y, wrap_angle(-pose[..., 2])], axis=-1)


def between(origin: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Return ``T_origin^-1 @ T_target``."""

    return compose(inverse(origin), target)


def interpolate(start: np.ndarray, end: np.ndarray, alpha: np.ndarray | float) -> np.ndarray:
    """Interpolate on SE(2): ``start @ Exp(alpha * Log(start^-1 end))``."""

    alpha = np.asarray(alpha, dtype=np.float64)
    delta = log(between(start, end))
    return compose(start, exp(delta * np.expand_dims(alpha, axis=-1)))


def integrate_body_velocity(velocity: np.ndarray, dt: np.ndarray) -> np.ndarray:
    """Integrate body-frame velocities into poses relative to a common origin."""

    velocity = np.asarray(velocity, dtype=np.float64)
    dt = np.asarray(dt, dtype=np.float64)
    if velocity.shape[-1] != 3 or velocity.shape[0] != dt.shape[0]:
        raise ValueError(f"Expected velocity (N,3) and dt (N,), got {velocity.shape}, {dt.shape}")
    poses = np.zeros((velocity.shape[0] + 1, 3), dtype=np.float64)
    for index, (vel, step_dt) in enumerate(zip(velocity, dt, strict=True)):
        poses[index + 1] = compose(poses[index], exp(vel * step_dt))
    return poses
