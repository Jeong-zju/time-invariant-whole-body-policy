"""Small, dependency-free SE(2) and quaternion helpers (quaternion order xyzw)."""

from __future__ import annotations

import numpy as np


def wrap_angle(angle: np.ndarray | float) -> np.ndarray:
    return (np.asarray(angle) + np.pi) % (2.0 * np.pi) - np.pi


def normalize_quaternion(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    if np.any(norm < 1e-12):
        raise ValueError("zero-norm quaternion")
    return q / norm


def quaternion_conjugate(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64).copy()
    q[..., :3] *= -1.0
    return q


def quaternion_multiply(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    lv, lw = left[..., :3], left[..., 3:4]
    rv, rw = right[..., :3], right[..., 3:4]
    xyz = lw * rv + rw * lv + np.cross(lv, rv)
    w = lw * rw - np.sum(lv * rv, axis=-1, keepdims=True)
    return normalize_quaternion(np.concatenate([xyz, w], axis=-1))


def quaternion_between(reference: np.ndarray, target: np.ndarray) -> np.ndarray:
    return quaternion_multiply(quaternion_conjugate(normalize_quaternion(reference)), target)


def quaternion_to_rotvec(q: np.ndarray) -> np.ndarray:
    q = normalize_quaternion(q)
    q = np.where((q[..., 3:4] < 0.0), -q, q)
    xyz = q[..., :3]
    sin_half = np.linalg.norm(xyz, axis=-1, keepdims=True)
    angle = 2.0 * np.arctan2(sin_half, np.clip(q[..., 3:4], -1.0, 1.0))
    scale = np.where(sin_half > 1e-10, angle / np.maximum(sin_half, 1e-12), 2.0)
    return xyz * scale


def rotvec_to_quaternion(rotvec: np.ndarray) -> np.ndarray:
    rotvec = np.asarray(rotvec, dtype=np.float64)
    angle = np.linalg.norm(rotvec, axis=-1, keepdims=True)
    half = 0.5 * angle
    scale = np.where(angle > 1e-10, np.sin(half) / np.maximum(angle, 1e-12), 0.5)
    return normalize_quaternion(np.concatenate([rotvec * scale, np.cos(half)], axis=-1))


def quaternion_slerp(q0: np.ndarray, q1: np.ndarray, alpha: float) -> np.ndarray:
    q0 = normalize_quaternion(q0)
    q1 = normalize_quaternion(q1)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    dot = float(np.clip(dot, -1.0, 1.0))
    if dot > 0.9995:
        return normalize_quaternion((1.0 - alpha) * q0 + alpha * q1)
    theta = np.arccos(dot)
    return normalize_quaternion(
        np.sin((1.0 - alpha) * theta) / np.sin(theta) * q0
        + np.sin(alpha * theta) / np.sin(theta) * q1
    )


def yaw_from_quaternion(q: np.ndarray) -> np.ndarray:
    q = normalize_quaternion(q)
    x, y, z, w = np.moveaxis(q, -1, 0)
    return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def local_base_pose(position: np.ndarray, quaternion_xyzw: np.ndarray) -> np.ndarray:
    """Return each planar base pose expressed in the first pose frame."""
    position = np.asarray(position, dtype=np.float64)
    yaw = yaw_from_quaternion(quaternion_xyzw)
    delta = position[:, :2] - position[0, :2]
    c, s = np.cos(yaw[0]), np.sin(yaw[0])
    rotation_world_to_local = np.array([[c, s], [-s, c]])
    local_xy = delta @ rotation_world_to_local.T
    local_yaw = wrap_angle(yaw - yaw[0])
    return np.column_stack([local_xy, local_yaw])


def interpolate_base(p0: np.ndarray, p1: np.ndarray, alpha: float) -> np.ndarray:
    result = (1.0 - alpha) * np.asarray(p0, dtype=np.float64) + alpha * np.asarray(
        p1, dtype=np.float64
    )
    result[2] = p0[2] + alpha * float(wrap_angle(p1[2] - p0[2]))
    return result
