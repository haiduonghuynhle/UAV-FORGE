"""
Homography-Based Yaw & Height Solver.

Combines the best of src/modules/vision/geometry_solver.py:
  • Yaw  via cv2.decomposeHomographyMat (heading angle from rotation matrix)
  • Altitude via GSD scale factor or homography determinant
"""

from __future__ import annotations
import logging
import numpy as np
import cv2
from typing import Optional, Tuple

from core.types import Pose, MatchResult, CameraIntrinsics
# Note: keep this module independent from native AnyVisLoc adapter math utilities.
# We'll just define the quaternion helpers here.

logger = logging.getLogger(__name__)

def euler_to_quaternion(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Convert Euler angles (rad) to quaternion [w, x, y, z]."""
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy

    return np.array([w, x, y, z], dtype=np.float64)

def _build_homography(query_pts: np.ndarray,
                      ref_pts: np.ndarray,
                      ransac_threshold: float = 5.0
                      ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    if len(query_pts) < 4 or len(ref_pts) < 4:
        return None, None
    H, mask = cv2.findHomography(
        query_pts.astype(np.float32),
        ref_pts.astype(np.float32),
        cv2.USAC_MAGSAC,  # Prefer USAC_MAGSAC for stability in dev/HD4
        ransacReprojThreshold=ransac_threshold,
    )
    if H is None:
        return None, None
    return H.astype(np.float64), (mask.ravel().astype(bool) if mask is not None else None)


class HomographyYawHeightSolver:
    """
    Solve yaw (heading) and altitude from feature matches using homography decomposition.
    Position is estimated by projecting the query image centre through the homography.
    """

    def __init__(self,
                 gsd: float = 0.3,
                 min_inliers: int = 8,
                 altitude_source: str = "estimate"):
        self._gsd             = gsd
        self._min_inliers     = min_inliers
        self._altitude_source = altitude_source
        self._last_heading_deg: Optional[float] = None

    def solve(self,
              match_result: MatchResult,
              reference_pose: Pose,
              camera: CameraIntrinsics,
              altitude_hint: Optional[float] = None,
              rotation_correction_rad: float = 0.0,
              query_size:  Optional[Tuple[int, int]] = None,
              ref_size:    Optional[Tuple[int, int]] = None,
              gsd:         Optional[float] = None,
              gsd_e:       Optional[float] = None,
              **kwargs) -> Optional[Pose]:
        
        if match_result.num_inliers < self._min_inliers:
            logger.debug("Not enough inliers (%d < %d)", match_result.num_inliers, self._min_inliers)
            return None

        # Allow passing GSD directly from the frame metadata (like in geometry_solver.py)
        gsd_to_use = gsd if gsd is not None else self._gsd

        H = match_result.homography
        if H is None:
            q_pts, r_pts = match_result.get_inlier_matches()
            H, _ = _build_homography(q_pts, r_pts)
            if H is None:
                return None
        H = np.asarray(H, dtype=np.float64)

        if query_size is not None:
            qh, qw = query_size[1], query_size[0]
        else:
            qw, qh = camera.width, camera.height

        if ref_size is not None:
            rh, rw = ref_size[1], ref_size[0]
        else:
            rw, rh = camera.width, camera.height

        q_ctr  = np.array([qw / 2.0, qh / 2.0, 1.0])
        r_mapped = H @ q_ctr
        if abs(r_mapped[2]) < 1e-8:
            return None
        r_mapped = r_mapped[:2] / r_mapped[2]

        dx_px = r_mapped[0] - rw / 2.0
        dy_px = r_mapped[1] - rh / 2.0

        # gsd is the north metres per pixel. On a square-degree raster the east
        # metres per pixel is smaller by cos(latitude), so the caller passes it
        # separately. Datasets on a metric raster pass nothing and keep the
        # isotropic expression.
        east_m  =  dx_px * (gsd_e if gsd_e is not None else gsd_to_use)
        north_m = -dy_px * gsd_to_use

        pos = reference_pose.position.copy()
        
        # Convert offset to deg like the current dev/HD4 solver
        lat_ref = pos[0]
        meters_per_deg_lat = 111320.0
        meters_per_deg_lon = 111320.0 * np.cos(np.radians(lat_ref))
        if meters_per_deg_lon < 1.0:
            meters_per_deg_lon = 1.0

        pos[0] += north_m / meters_per_deg_lat
        pos[1] += east_m / meters_per_deg_lon

        if self._altitude_source == "hint" and altitude_hint is not None:
            alt = float(altitude_hint)
        elif self._altitude_source == "fixed":
            alt = 100.0
        else:
            q_inliers, r_inliers = match_result.get_inlier_matches()
            alt = self._estimate_altitude(camera, gsd_to_use, q_inliers, r_inliers)
        
        # In dev/HD4, pos[2] is positive altitude usually, let's keep it consistent
        pos[2] = alt

        heading_rad = rotation_correction_rad
        try:
            sx    = qw / camera.width  if camera.width  > 0 else 1.0
            sy    = qh / camera.height if camera.height > 0 else 1.0
            K_scl = np.array([
                [camera.fx * sx,  0,              camera.cx * sx],
                [0,               camera.fy * sy, camera.cy * sy],
                [0,               0,              1             ],
            ])
            n_sol, rotations, translations, normals = cv2.decomposeHomographyMat(H, K_scl)
            if n_sol > 0:
                best = self._best_solution(rotations, normals, translations)
                heading_rad = rotation_correction_rad + self._yaw_from_R(rotations[best])
        except Exception as exc:
            logger.debug("decomposeHomographyMat failed: %s", exc)

        heading_rad = heading_rad % (2 * np.pi)
        self._last_heading_deg = float(np.degrees(heading_rad))

        # In dev/HD4, orientation is represented as [w, x, y, z] but typically only rotation around Z.
        # We can use the simple qw, qz if pitch=0 roll=0
        qw_yaw = np.cos(heading_rad / 2.0)
        qz_yaw = np.sin(heading_rad / 2.0)
        orientation = np.array([qw_yaw, 0.0, 0.0, qz_yaw], dtype=np.float64)

        return Pose(
            timestamp   = reference_pose.timestamp,
            position    = pos,
            orientation = orientation,
        )

    @staticmethod
    def _scale_from_correspondences(q_pts: np.ndarray,
                                    r_pts: np.ndarray,
                                    min_dist_px: float = 5.0,
                                    max_pairs: int = 500) -> Optional[float]:
        n = len(q_pts)
        if n < 2:
            return None

        if n > max_pairs:
            idx = np.random.choice(n, max_pairs, replace=False)
            q_pts = q_pts[idx]
            r_pts = r_pts[idx]
            n = max_pairs

        dq = np.linalg.norm(q_pts[:, None, :] - q_pts[None, :, :], axis=-1)
        dr = np.linalg.norm(r_pts[:, None, :] - r_pts[None, :, :], axis=-1)

        upper = np.triu_indices(n, k=1)
        dq_flat = dq[upper]
        dr_flat = dr[upper]

        valid = dq_flat > min_dist_px
        if valid.sum() < 3:
            return None

        sigma = dr_flat[valid] / dq_flat[valid]

        q1, q3 = np.percentile(sigma, [25, 75])
        iqr = q3 - q1
        inliers = sigma[(sigma >= q1 - 1.5 * iqr) & (sigma <= q3 + 1.5 * iqr)]
        if len(inliers) < 3:
            inliers = sigma

        return float(np.median(inliers))

    def _estimate_altitude(self,
                           camera: CameraIntrinsics,
                           gsd_sat: float,
                           q_pts: Optional[np.ndarray] = None,
                           r_pts: Optional[np.ndarray] = None) -> float:
        focal_px = (camera.fx + camera.fy) / 2.0
        base_alt = gsd_sat * focal_px

        sigma = None
        if q_pts is not None and r_pts is not None and len(q_pts) >= 2:
            sigma = self._scale_from_correspondences(q_pts, r_pts)

        if sigma is not None and 0.05 < sigma < 20.0:
            alt = sigma * base_alt
        else:
            alt = base_alt

        return float(np.clip(alt, 10.0, 1000.0))

    @staticmethod
    def _yaw_from_R(R: np.ndarray) -> float:
        if abs(R[2, 0]) > 0.998:
            return float(np.arctan2(-R[0, 1], R[1, 1]))
        return float(np.arctan2(R[1, 0], R[0, 0]))

    @staticmethod
    def _best_solution(rotations, normals, translations) -> int:
        best_idx   = 0
        best_score = -np.inf
        # Assuming Z is positive up in ENU, normals pointing down into camera
        up_vec     = np.array([0.0, 0.0, -1.0])
        for i, (R, n, t) in enumerate(zip(rotations, normals, translations)):
            score = float(np.dot(n.ravel(), up_vec))
            if score > best_score:
                best_score = score
                best_idx   = i
        return best_idx
