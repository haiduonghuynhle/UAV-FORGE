"""
Core data contracts for HD4 runtime modules.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np


@dataclass
class Pose:
    """6-DoF pose in local frame with quaternion orientation [w, x, y, z]."""
    timestamp: float
    position: np.ndarray
    orientation: np.ndarray
    covariance: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        self.position = np.asarray(self.position, dtype=np.float64)
        self.orientation = np.asarray(self.orientation, dtype=np.float64)
        qn = np.linalg.norm(self.orientation)
        if qn > 1e-10:
            self.orientation = self.orientation / qn
        if self.covariance is not None:
            self.covariance = np.asarray(self.covariance, dtype=np.float64)

    @property
    def euler_angles(self) -> np.ndarray:
        """Return Euler [roll, pitch, yaw] in radians (ZYX)."""
        w, x, y, z = self.orientation
        roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
        pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1.0, 1.0))
        yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        return np.array([roll, pitch, yaw], dtype=np.float64)


@dataclass
class MatchResult:
    """Feature matching result between query and reference image."""
    query_keypoints: np.ndarray
    ref_keypoints: np.ndarray
    confidence: np.ndarray
    homography: Optional[np.ndarray] = None
    inlier_mask: Optional[np.ndarray] = None
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self) -> None:
        self.query_keypoints = np.asarray(self.query_keypoints, dtype=np.float64)
        self.ref_keypoints = np.asarray(self.ref_keypoints, dtype=np.float64)
        self.confidence = np.asarray(self.confidence, dtype=np.float64)
        if self.homography is not None:
            self.homography = np.asarray(self.homography, dtype=np.float64)
        if self.inlier_mask is not None:
            self.inlier_mask = np.asarray(self.inlier_mask, dtype=bool)
        if self.metadata is None:
            self.metadata = {}

    @property
    def num_matches(self) -> int:
        return int(len(self.query_keypoints))

    @property
    def num_inliers(self) -> int:
        if self.inlier_mask is None:
            return self.num_matches
        return int(np.sum(self.inlier_mask))

    def get_inlier_matches(self) -> Tuple[np.ndarray, np.ndarray]:
        if self.inlier_mask is None:
            return self.query_keypoints, self.ref_keypoints
        return self.query_keypoints[self.inlier_mask], self.ref_keypoints[self.inlier_mask]


@dataclass
class CameraIntrinsics:
    """Camera intrinsic parameters."""
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int

    @property
    def K(self) -> np.ndarray:
        return np.array(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
