"""
Math helpers for quaternion/euler conversion.
"""

from __future__ import annotations

import numpy as np


def quaternion_to_euler(q_wxyz: np.ndarray) -> np.ndarray:
    """
    Convert quaternion [w, x, y, z] to Euler [roll, pitch, yaw] radians.
    """
    q = np.asarray(q_wxyz, dtype=np.float64)
    w, x, y, z = q
    roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1.0, 1.0))
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return np.array([roll, pitch, yaw], dtype=np.float64)
