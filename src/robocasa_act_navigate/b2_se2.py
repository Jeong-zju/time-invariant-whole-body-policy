"""Dependency-free SE(2) helpers for B2 labels (poses are x, y, yaw)."""

from __future__ import annotations

import numpy as np


def wrap_angle(angle):
    return (np.asarray(angle) + np.pi) % (2.0 * np.pi) - np.pi


def _ab(theta):
    theta = np.asarray(theta, dtype=np.float64)
    small = np.abs(theta) < 1e-6
    theta2 = theta * theta
    a = np.empty_like(theta); b = np.empty_like(theta)
    np.divide(np.sin(theta), theta, out=a, where=~small)
    np.divide(1.0 - np.cos(theta), theta, out=b, where=~small)
    a = np.where(small, 1.0 - theta2 / 6.0 + theta2 * theta2 / 120.0, a)
    b = np.where(small, theta / 2.0 - theta * theta2 / 24.0 + theta * theta2 * theta2 / 720.0, b)
    return a, b


def exp(twist):
    twist = np.asarray(twist, dtype=np.float64)
    x, y, theta = np.moveaxis(twist, -1, 0)
    a, b = _ab(theta)
    return np.stack((a * x - b * y, b * x + a * y, wrap_angle(theta)), axis=-1)


def log(pose):
    pose = np.asarray(pose, dtype=np.float64)
    x, y, theta = np.moveaxis(pose, -1, 0); theta = wrap_angle(theta)
    a, b = _ab(theta); denominator = np.maximum(a * a + b * b, np.finfo(np.float64).eps)
    return np.stack(((a * x + b * y) / denominator, (-b * x + a * y) / denominator, theta), axis=-1)


def compose(left, right):
    left = np.asarray(left, dtype=np.float64); right = np.asarray(right, dtype=np.float64)
    c, s = np.cos(left[..., 2]), np.sin(left[..., 2])
    return np.stack((left[..., 0] + c * right[..., 0] - s * right[..., 1],
                     left[..., 1] + s * right[..., 0] + c * right[..., 1],
                     wrap_angle(left[..., 2] + right[..., 2])), axis=-1)


def inverse(pose):
    pose = np.asarray(pose, dtype=np.float64); c, s = np.cos(pose[..., 2]), np.sin(pose[..., 2])
    return np.stack((-c * pose[..., 0] - s * pose[..., 1], s * pose[..., 0] - c * pose[..., 1], wrap_angle(-pose[..., 2])), axis=-1)


def between(origin, target):
    return compose(inverse(origin), target)


def interpolate(start, end, alpha):
    delta = log(between(start, end))
    return compose(start, exp(delta * np.expand_dims(np.asarray(alpha, dtype=np.float64), axis=-1)))
