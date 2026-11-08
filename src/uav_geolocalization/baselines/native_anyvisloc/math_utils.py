"""
Math utilities for the native AnyVisLoc baseline adapter.

Quaternion operations and Euler-angle helpers used by geometry_solver.
Adapted from src/utils/math_utils.py.
"""

from __future__ import annotations
import numpy as np
from scipy.spatial.transform import Rotation


def quaternion_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Multiply two quaternions [w, x, y, z]."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])


def quaternion_normalize(q: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(q)
    return q / n if n > 1e-10 else np.array([1., 0., 0., 0.])


def euler_to_quaternion(roll: float, pitch: float, yaw: float,
                        order: str = "ZYX") -> np.ndarray:
    """Return quaternion [w, x, y, z] from Euler angles (radians, ZYX default)."""
    rot = Rotation.from_euler(order.lower(), [yaw, pitch, roll])
    q = rot.as_quat()          # scipy → [x, y, z, w]
    return np.array([q[3], q[0], q[1], q[2]])


def rotation_matrix_to_quaternion(R: np.ndarray) -> np.ndarray:
    rot = Rotation.from_matrix(R)
    q = rot.as_quat()          # [x, y, z, w]
    return np.array([q[3], q[0], q[1], q[2]])
