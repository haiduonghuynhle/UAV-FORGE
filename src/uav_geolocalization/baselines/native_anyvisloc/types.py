"""
Core data types for the native AnyVisLoc baseline adapter.

Lightweight versions of Pose, MatchResult, CameraIntrinsics needed by the
combined Benchmark + src pipeline.  Taken from src/core/types.py and
trimmed to what is actually used by the native AnyVisLoc adapter.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, Any
import numpy as np
from enum import Enum


class CoordinateFrame(Enum):
    NED  = "ned"
    ENU  = "enu"
    ECEF = "ecef"
    BODY = "body"


# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class Pose:
    """6-DoF pose: position (m) + orientation (quaternion [w,x,y,z])."""
    timestamp:   float
    position:    np.ndarray   # (3,)
    orientation: np.ndarray   # (4,) [w,x,y,z]
    frame:       CoordinateFrame = CoordinateFrame.NED
    covariance:  Optional[np.ndarray] = None  # (6,6)

    def __post_init__(self):
        self.position    = np.asarray(self.position,    dtype=np.float64)
        self.orientation = np.asarray(self.orientation, dtype=np.float64)
        n = np.linalg.norm(self.orientation)
        if n > 1e-10:
            self.orientation /= n
        if self.covariance is not None:
            self.covariance = np.asarray(self.covariance, dtype=np.float64)

    # ── shortcuts ────────────────────────────────────────────────────────────
    @property
    def x(self) -> float: return float(self.position[0])
    @property
    def y(self) -> float: return float(self.position[1])
    @property
    def z(self) -> float: return float(self.position[2])

    @property
    def rotation_matrix(self) -> np.ndarray:
        w, x, y, z = self.orientation
        return np.array([
            [1-2*(y**2+z**2), 2*(x*y-w*z),   2*(x*z+w*y)],
            [2*(x*y+w*z),   1-2*(x**2+z**2),  2*(y*z-w*x)],
            [2*(x*z-w*y),   2*(y*z+w*x),  1-2*(x**2+y**2)],
        ])

    @property
    def euler_angles(self) -> np.ndarray:
        """Return [roll, pitch, yaw] in radians (ZYX convention)."""
        w, x, y, z = self.orientation
        roll  = np.arctan2(2*(w*x+y*z), 1-2*(x**2+y**2))
        sinp  = np.clip(2*(w*y-z*x), -1, 1)
        pitch = np.arcsin(sinp)
        yaw   = np.arctan2(2*(w*z+x*y), 1-2*(y**2+z**2))
        return np.array([roll, pitch, yaw])

    @classmethod
    def from_euler(cls, timestamp: float, position: np.ndarray,
                   euler: np.ndarray,
                   frame: CoordinateFrame = CoordinateFrame.NED) -> "Pose":
        roll, pitch, yaw = euler
        cy, sy = np.cos(yaw*.5),   np.sin(yaw*.5)
        cp, sp = np.cos(pitch*.5), np.sin(pitch*.5)
        cr, sr = np.cos(roll*.5),  np.sin(roll*.5)
        q = np.array([
            cr*cp*cy + sr*sp*sy,
            sr*cp*cy - cr*sp*sy,
            cr*sp*cy + sr*cp*sy,
            cr*cp*sy - sr*sp*cy,
        ])
        return cls(timestamp, position, q, frame)


# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class MatchResult:
    """Feature matching result between query and reference images."""
    query_keypoints: np.ndarray   # (N,2) – pixel coords in query
    ref_keypoints:   np.ndarray   # (N,2) – pixel coords in reference
    confidence:      np.ndarray   # (N,)
    homography:      Optional[np.ndarray] = None  # (3,3)
    inlier_mask:     Optional[np.ndarray] = None  # (N,) bool
    metadata:        Optional[Dict[str, Any]] = None

    def __post_init__(self):
        self.query_keypoints = np.asarray(self.query_keypoints, dtype=np.float64)
        self.ref_keypoints   = np.asarray(self.ref_keypoints,   dtype=np.float64)
        self.confidence      = np.asarray(self.confidence,      dtype=np.float64)
        if self.homography  is not None:
            self.homography  = np.asarray(self.homography,  dtype=np.float64)
        if self.inlier_mask is not None:
            self.inlier_mask = np.asarray(self.inlier_mask, dtype=bool)

    @property
    def num_matches(self) -> int:
        return len(self.query_keypoints)

    @property
    def num_inliers(self) -> int:
        if self.inlier_mask is None:
            return self.num_matches
        return int(np.sum(self.inlier_mask))

    @property
    def inlier_ratio(self) -> float:
        return self.num_inliers / self.num_matches if self.num_matches > 0 else 0.0

    def get_inlier_matches(self) -> Tuple[np.ndarray, np.ndarray]:
        if self.inlier_mask is None:
            return self.query_keypoints, self.ref_keypoints
        return self.query_keypoints[self.inlier_mask], self.ref_keypoints[self.inlier_mask]


# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class CameraIntrinsics:
    """Camera intrinsic parameters."""
    fx: float
    fy: float
    cx: float
    cy: float
    width:  int
    height: int
    distortion: np.ndarray = field(default_factory=lambda: np.zeros(5))

    def __post_init__(self):
        self.distortion = np.asarray(self.distortion, dtype=np.float64)

    @property
    def K(self) -> np.ndarray:
        return np.array([[self.fx, 0, self.cx],
                         [0, self.fy, self.cy],
                         [0, 0, 1]], dtype=np.float64)

    @property
    def K_inv(self) -> np.ndarray:
        return np.linalg.inv(self.K)


# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class GeoReference:
    """Geographic reference used for local ↔ geodetic conversions."""
    origin_lat: float
    origin_lon: float
    origin_alt: float = 0.0
    gsd:        float = 0.1  # metres per pixel

    def __post_init__(self):
        self._lat_rad = np.radians(self.origin_lat)
        R = 6378137.0
        self._m_per_deg_lat = R * np.pi / 180
        self._m_per_deg_lon = R * np.cos(self._lat_rad) * np.pi / 180
