"""
Homography-based yaw and height solver for the native AnyVisLoc adapter.

Combines the best of src/modules/vision/geometry_solver.py:
  • Yaw  via cv2.decomposeHomographyMat (heading angle from rotation matrix)
  • Altitude via GSD scale factor or homography determinant

Standalone: only depends on native AnyVisLoc adapter types and math utilities.
"""

from __future__ import annotations
import logging
import numpy as np
import cv2
from typing import Optional, Tuple

from .types     import Pose, MatchResult, CameraIntrinsics
from .math_utils import euler_to_quaternion, rotation_matrix_to_quaternion

logger = logging.getLogger(__name__)


def _build_homography(query_pts: np.ndarray,
                      ref_pts: np.ndarray,
                      ransac_threshold: float = 5.0
                      ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Estimate homography from corresponding pixel coordinates.

    Returns
    -------
    H    : (3,3) float64 or None
    mask : (N,) bool or None
    """
    if len(query_pts) < 4 or len(ref_pts) < 4:
        return None, None
    H, mask = cv2.findHomography(
        query_pts.astype(np.float32),
        ref_pts.astype(np.float32),
        cv2.RANSAC,
        ransacReprojThreshold=ransac_threshold,
    )
    if H is None:
        return None, None
    return H.astype(np.float64), (mask.ravel().astype(bool) if mask is not None else None)


class HomographyYawHeightSolver:
    """
    Solve yaw (heading) and altitude from feature matches using homography decomposition.

    Position is estimated by projecting the query image centre through the
    homography onto the reference tile, then converting the pixel offset to
    metric using GSD.

    Yaw is extracted with ``cv2.decomposeHomographyMat``.

    Altitude is estimated from:
      1. The GSD formula:  alt = gsd × focal_length_pixels  (primary)
      2. Scale factor from det(H)                           (secondary correction)

    Parameters
    ----------
    gsd : float
        Ground Sampling Distance of the *reference* tile in metres/pixel.
    min_inliers : int
        Minimum RANSAC inliers required for a valid solution.
    altitude_source : {"estimate", "hint", "fixed"}
        "estimate"  – derive from homography + GSD  (default)
        "hint"      – use ``altitude_hint`` argument
        "fixed"     – always return 100 m
    """

    def __init__(self,
                 gsd: float = 0.1,
                 min_inliers: int = 8,
                 altitude_source: str = "estimate"):
        self._gsd             = gsd
        self._min_inliers     = min_inliers
        self._altitude_source = altitude_source
        self._last_heading_deg: Optional[float] = None

    # ──────────────────────────────────────────────────────────────────────────
    def solve(self,
              match_result: MatchResult,
              reference_pose: Pose,
              camera: CameraIntrinsics,
              altitude_hint: Optional[float] = None,
              rotation_correction_rad: float = 0.0,
              query_size:  Optional[Tuple[int, int]] = None,
              ref_size:    Optional[Tuple[int, int]] = None) -> Optional[Pose]:
        """
        Compute position + heading from ``match_result``.

        Parameters
        ----------
        match_result : MatchResult
            Contains keypoints (query + ref) and optionally a pre-computed
            homography. If ``match_result.homography`` is None a new one is
            estimated via RANSAC.
        reference_pose : Pose
            Pose of the reference tile centre in the local frame (NED).
        camera : CameraIntrinsics
            Camera intrinsics of the query (UAV) image.
        altitude_hint : float, optional
            Known/estimated altitude, used when ``altitude_source="hint"``.
        rotation_correction_rad : float
            Heading correction from prior alignment stage (radians).
        query_size : (h, w), optional
            Query image dimensions; inferred from keypoints if not given.
        ref_size : (h, w), optional
            Reference image dimensions; inferred from keypoints if not given.

        Returns
        -------
        Pose or None
        """
        # ── 1. Validate ───────────────────────────────────────────────────────
        if match_result.num_inliers < self._min_inliers:
            logger.debug("Not enough inliers (%d < %d)",
                         match_result.num_inliers, self._min_inliers)
            return None

        # ── 2. Homography ─────────────────────────────────────────────────────
        H = match_result.homography
        if H is None:
            q_pts, r_pts = match_result.get_inlier_matches()
            H, _ = _build_homography(q_pts, r_pts)
            if H is None:
                return None
        H = np.asarray(H, dtype=np.float64)

        # ── 3. Image dimensions ───────────────────────────────────────────────
        if query_size is not None:
            qh, qw = query_size
        else:
            qkp = match_result.query_keypoints
            qw  = int(qkp[:, 0].max() * 1.15) + 1
            qh  = int(qkp[:, 1].max() * 1.15) + 1

        if ref_size is not None:
            rh, rw = ref_size
        else:
            rkp = match_result.ref_keypoints
            rw  = int(rkp[:, 0].max() * 1.15) + 1
            rh  = int(rkp[:, 1].max() * 1.15) + 1

        # ── 4. Position via centre-mapping ────────────────────────────────────
        # Project query image centre → reference pixel
        q_ctr  = np.array([qw / 2.0, qh / 2.0, 1.0])
        r_mapped = H @ q_ctr
        if abs(r_mapped[2]) < 1e-8:
            return None
        r_mapped = r_mapped[:2] / r_mapped[2]

        dx_px = r_mapped[0] - rw / 2.0   # positive → East
        dy_px = r_mapped[1] - rh / 2.0   # positive → South (image Y)

        east_m  =  dx_px * self._gsd
        north_m = -dy_px * self._gsd   # flip: image-Y is south

        p_ref   = reference_pose.position.copy()
        pos     = p_ref.copy()
        pos[0] += north_m   # NED: X = North
        pos[1] += east_m    # NED: Y = East

        # ── 5. Altitude ───────────────────────────────────────────────────────
        if self._altitude_source == "hint" and altitude_hint is not None:
            alt = float(altitude_hint)
        elif self._altitude_source == "fixed":
            alt = 100.0
        else:
            # Use inlier correspondences for a physically correct scale estimate
            q_inliers, r_inliers = match_result.get_inlier_matches()
            alt = self._estimate_altitude(camera, q_inliers, r_inliers)
        pos[2] = alt    # Positive AGL (consistent with Pose convention in NED frame)

        # ── 6. Heading via decomposeHomographyMat ────────────────────────────
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

        # ── 7. Orientation quaternion ─────────────────────────────────────────
        orientation = euler_to_quaternion(0.0, np.radians(-90.0), heading_rad)

        return Pose(
            timestamp   = reference_pose.timestamp,
            position    = pos,
            orientation = orientation,
            frame       = reference_pose.frame,
        )

    # ──────────────────────────────────────────────────────────────────────────
    def solve_from_points(self,
                          query_pts: np.ndarray,
                          ref_pts:   np.ndarray,
                          reference_pose: Pose,
                          camera: CameraIntrinsics,
                          altitude_hint:   Optional[float] = None,
                          rotation_correction_rad: float = 0.0,
                          query_size:  Optional[Tuple[int, int]] = None,
                          ref_size:    Optional[Tuple[int, int]] = None,
                          ransac_threshold: float = 5.0,
                          min_inliers: Optional[int] = None) -> Optional[Pose]:
        """
        Convenience wrapper: build a MatchResult from raw point arrays, then call solve().

        Parameters
        ----------
        query_pts : (N,2) matching pixel coordinates in the query image
        ref_pts   : (N,2) matching pixel coordinates in the reference image
        (remaining parameters are the same as solve())

        Returns
        -------
        Pose or None
        """
        if len(query_pts) < 4:
            return None

        H, mask = _build_homography(query_pts, ref_pts, ransac_threshold)
        if H is None:
            return None

        n_inliers = int(mask.sum()) if mask is not None else len(query_pts)
        eff_min   = min_inliers if min_inliers is not None else self._min_inliers
        if n_inliers < eff_min:
            return None

        from .types import MatchResult
        mr = MatchResult(
            query_keypoints = np.array(query_pts, dtype=np.float64),
            ref_keypoints   = np.array(ref_pts,   dtype=np.float64),
            confidence      = np.ones(len(query_pts)),
            homography      = H,
            inlier_mask     = mask,
        )
        return self.solve(
            match_result            = mr,
            reference_pose          = reference_pose,
            camera                  = camera,
            altitude_hint           = altitude_hint,
            rotation_correction_rad = rotation_correction_rad,
            query_size              = query_size,
            ref_size                = ref_size,
        )

    # ──────────────────────────────────────────────────────────────────────────
    #  Helpers
    # ──────────────────────────────────────────────────────────────────────────

    def get_last_heading(self) -> Optional[float]:
        """Heading in degrees [0, 360) from the most recent solve() call."""
        return self._last_heading_deg

    # ── private ───────────────────────────────────────────────────────────────

    @staticmethod
    def _scale_from_correspondences(q_pts: np.ndarray,
                                    r_pts: np.ndarray,
                                    min_dist_px: float = 5.0,
                                    max_pairs: int = 500) -> Optional[float]:
        """
        Estimate the pixel-scale ratio σ = dist_r / dist_q from inlier pairs.

        Physical meaning
        ----------------
        H maps query (UAV) → reference (satellite).  For any pair of matched
        features at ground distance d apart:
            d = dist_q * GSD_UAV  =  dist_r * GSD_sat
          → GSD_UAV = (dist_r / dist_q) * GSD_sat  =  σ * GSD_sat
          → altitude = GSD_UAV * focal_px  =  σ * GSD_sat * focal_px

        A UAV with better resolution than the satellite has σ < 1, giving an
        altitude below the naive ``GSD_sat * focal_px`` base estimate.

        Parameters
        ----------
        q_pts, r_pts : (M,2) inlier pixel coordinates
        min_dist_px  : ignore pairs closer than this many pixels (in UAV image)
                       to avoid dividing by near-zero distances.
        max_pairs    : cap on number of pairs sampled to bound O(n²) cost.

        Returns
        -------
        Median σ, or None if too few usable pairs.
        """
        n = len(q_pts)
        if n < 2:
            return None

        # If n is large, sub-sample to keep computation fast
        if n > max_pairs:
            idx = np.random.choice(n, max_pairs, replace=False)
            q_pts = q_pts[idx]
            r_pts = r_pts[idx]
            n = max_pairs

        # Vectorised pairwise distances using broadcasting
        dq = np.linalg.norm(q_pts[:, None, :] - q_pts[None, :, :], axis=-1)  # (n,n)
        dr = np.linalg.norm(r_pts[:, None, :] - r_pts[None, :, :], axis=-1)  # (n,n)

        # Upper-triangle only (avoid duplicates & diagonal)
        upper = np.triu_indices(n, k=1)
        dq_flat = dq[upper]
        dr_flat = dr[upper]

        # Keep only pairs where UAV distance is large enough (numerical stability)
        valid = dq_flat > min_dist_px
        if valid.sum() < 3:
            return None

        sigma = dr_flat[valid] / dq_flat[valid]

        # Reject obvious outliers (Tukey fences)
        q1, q3 = np.percentile(sigma, [25, 75])
        iqr = q3 - q1
        inliers = sigma[(sigma >= q1 - 1.5 * iqr) & (sigma <= q3 + 1.5 * iqr)]
        if len(inliers) < 3:
            inliers = sigma  # fallback: use all

        return float(np.median(inliers))

    def _estimate_altitude(self,
                           camera: CameraIntrinsics,
                           q_pts: Optional[np.ndarray] = None,
                           r_pts: Optional[np.ndarray] = None) -> float:
        """
        Estimate altitude (metres) using the GSD-focal-length identity.

        Formula
        -------
        altitude = σ × GSD_sat × focal_px

        where σ = median(dist_r / dist_q) over inlier correspondence pairs
        (see ``_scale_from_correspondences``).  When correspondences are not
        available, σ defaults to 1.0 (returns the base estimate GSD_sat × f_px).

        This is physically correct and avoids the biased ``det(H)^0.5`` proxy
        which conflates scale with perspective warp terms.
        """
        focal_px = (camera.fx + camera.fy) / 2.0
        base_alt = self._gsd * focal_px   # altitude where GSD_UAV == GSD_sat

        sigma = None
        if q_pts is not None and r_pts is not None and len(q_pts) >= 2:
            sigma = self._scale_from_correspondences(q_pts, r_pts)

        if sigma is not None and 0.05 < sigma < 20.0:
            alt = sigma * base_alt        # correct direction: multiply
        else:
            alt = base_alt                # fallback: no scale correction

        return float(np.clip(alt, 10.0, 1000.0))

    @staticmethod
    def _yaw_from_R(R: np.ndarray) -> float:
        """Extract yaw from rotation matrix using ZYX Euler convention."""
        if abs(R[2, 0]) > 0.998:          # gimbal-lock
            return float(np.arctan2(-R[0, 1], R[1, 1]))
        return float(np.arctan2(R[1, 0], R[0, 0]))

    @staticmethod
    def _best_solution(rotations, normals, translations) -> int:
        """
        Pick the homography decomposition consistent with a nadir-down view.
        Prefer the solution whose normal points most upward (toward camera)
        and whose translation has a positive Z component (camera above plane).
        """
        best_idx   = 0
        best_score = -np.inf
        up_vec     = np.array([0.0, 0.0, -1.0])   # NED: -Z is "up"
        for i, (R, n, t) in enumerate(zip(rotations, normals, translations)):
            # Normal should point toward camera (roughly -Z in scene)
            score = float(np.dot(n.ravel(), up_vec))
            if score > best_score:
                best_score = score
                best_idx   = i
        return best_idx


# ─────────────────────────────────────────────────────────────────────────────
# Convenience stand-alone function
# ─────────────────────────────────────────────────────────────────────────────

def extract_yaw_and_height(
    query_pts:   np.ndarray,
    ref_pts:     np.ndarray,
    camera_K:    np.ndarray,
    img_width:   int,
    img_height:  int,
    ref_cut_W:   int,
    ref_cut_H:   int,
    gsd:         float,
    rotation_correction_rad: float = 0.0,
    ransac_threshold: float = 5.0,
    min_inliers: int = 8,
) -> Tuple[Optional[float], Optional[float], int]:
    """
    Standalone helper – no MatchResult / CameraIntrinsics needed.

    Parameters
    ----------
    query_pts : (N,2) float  UAV image pixel coordinates
    ref_pts   : (N,2) float  Reference tile pixel coordinates
    camera_K  : (3,3) float  Camera intrinsic matrix
    img_width, img_height : int  Query image dimensions
    ref_cut_W, ref_cut_H  : int  Reference tile dimensions (pixels)
    gsd       : float  Reference tile GSD in m/px
    rotation_correction_rad : float  Prior alignment angle
    ransac_threshold : float  RANSAC reprojection threshold (pixels)
    min_inliers : int

    Returns
    -------
    yaw_deg   : float or None   Heading in degrees [0, 360)
    alt_m     : float or None   Altitude in metres
    n_inliers : int             Number of RANSAC inliers used
    """
    if len(query_pts) < 4:
        return None, None, 0

    H, mask = _build_homography(query_pts, ref_pts, ransac_threshold)
    if H is None:
        return None, None, 0

    n_inliers = int(mask.sum()) if mask is not None else len(query_pts)
    if n_inliers < min_inliers:
        return None, None, n_inliers

    # ── Heading ───────────────────────────────────────────────────────────────
    heading_rad = rotation_correction_rad
    try:
        n_sol, rotations, translations, normals = cv2.decomposeHomographyMat(H, camera_K)
        if n_sol > 0:
            best = HomographyYawHeightSolver._best_solution(rotations, normals, translations)
            heading_rad = (rotation_correction_rad +
                           HomographyYawHeightSolver._yaw_from_R(rotations[best]))
    except Exception:
        pass
    heading_rad = heading_rad % (2 * np.pi)
    yaw_deg = float(np.degrees(heading_rad))

    # ── Altitude ──────────────────────────────────────────────────────────────
    # Use inlier correspondences for a physically correct, unbiased scale.
    # σ = median(dist_r / dist_q) gives GSD_UAV = σ * GSD_sat
    # → altitude = σ * GSD_sat * focal_px  (multiply, not divide)
    focal_px = (camera_K[0, 0] + camera_K[1, 1]) / 2.0
    base_alt = gsd * focal_px

    inlier_q = query_pts[mask] if mask is not None else query_pts
    inlier_r = ref_pts[mask]   if mask is not None else ref_pts

    sigma = HomographyYawHeightSolver._scale_from_correspondences(inlier_q, inlier_r)
    if sigma is not None and 0.05 < sigma < 20.0:
        alt_m = sigma * base_alt
    else:
        alt_m = base_alt

    alt_m = float(np.clip(alt_m, 10.0, 1000.0))

    return yaw_deg, alt_m, n_inliers
