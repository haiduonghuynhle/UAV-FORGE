"""
Homography-based geometry solver for HD4 runtime.
"""

from __future__ import annotations

import logging

import numpy as np

from core.types import CameraIntrinsics, MatchResult, Pose
from .dsm_map import BaseDSMMap

logger = logging.getLogger(__name__)


class HomographyGeometrySolver:
    """Estimate local pose from homography and GSD."""

    def __init__(self, gsd: float = 0.3, min_inliers: int = 8):
        self._gsd = gsd
        self._min_inliers = min_inliers

    def solve(
        self,
        match_result: MatchResult,
        reference_pose: Pose,
        camera: CameraIntrinsics,
        query_size=None,
        altitude_hint=None,
        gsd=None,
        ref_size=None,
        gsd_e=None,
        **_kwargs,
    ):
        import cv2

        if match_result.num_inliers < self._min_inliers:
            logger.debug(
                "Geometry skip: num_inliers %s < min_inliers %s",
                match_result.num_inliers,
                self._min_inliers,
            )
            return None

        H = match_result.homography
        if H is None:
            H, mask = cv2.findHomography(
                match_result.query_keypoints.astype(np.float32),
                match_result.ref_keypoints.astype(np.float32),
                cv2.USAC_MAGSAC,
                20.0,
            )
            if H is None:
                logger.debug("findHomography returned None")
                return None
            if mask is not None:
                match_result.inlier_mask = mask.ravel().astype(bool)

        q_w = query_size[0] if query_size is not None else camera.width
        q_h = query_size[1] if query_size is not None else camera.height
        q_ctr = np.array([q_w / 2.0, q_h / 2.0, 1.0], dtype=np.float64)
        r_mapped = H @ q_ctr
        if abs(r_mapped[2]) < 1e-8:
            logger.debug("Homography degenerate: r_mapped[2]=%s", r_mapped[2])
            return None
        r_mapped = r_mapped[:2] / r_mapped[2]

        r_w = ref_size[0] if ref_size is not None else camera.width
        r_h = ref_size[1] if ref_size is not None else camera.height

        dx_px = r_mapped[0] - (r_w / 2.0)
        dy_px = r_mapped[1] - (r_h / 2.0)
        
        gsd_to_use = gsd if gsd is not None else self._gsd
        # See homography_solver: on a square-degree raster the east scale is
        # gsd * cos(latitude), supplied by the caller.
        east_m = dx_px * (gsd_e if gsd_e is not None else gsd_to_use)
        north_m = -dy_px * gsd_to_use

        pos = reference_pose.position.copy()
        # Convert meter offsets to degree offsets (WGS84 approximation)
        # 1 degree latitude ≈ 111,320 m
        # 1 degree longitude ≈ 111,320 m × cos(latitude)
        lat_ref = pos[0]
        meters_per_deg_lat = 111320.0
        meters_per_deg_lon = 111320.0 * np.cos(np.radians(lat_ref))
        if meters_per_deg_lon < 1.0:
            meters_per_deg_lon = 1.0  # safety for poles
        
        pos[0] += north_m / meters_per_deg_lat
        pos[1] += east_m / meters_per_deg_lon
        pos[2] = float(altitude_hint if altitude_hint is not None else reference_pose.position[2])

        yaw = self._yaw_from_homography(H, camera.K, match_result)

        qw = np.cos(yaw / 2.0)
        qz = np.sin(yaw / 2.0)
        orientation = np.array([qw, 0.0, 0.0, qz], dtype=np.float64)
        return Pose(timestamp=reference_pose.timestamp, position=pos, orientation=orientation)

    @staticmethod
    def _yaw_from_homography(H: np.ndarray, K: np.ndarray, match_result: MatchResult) -> float:
        """
        Recover planar yaw from H using cv2.decomposeHomographyMat.
        Prefer OpenCV's visible-refpoint filter to disambiguate the four solutions.
        """
        import cv2

        K64 = np.asarray(K, dtype=np.float64)
        H64 = np.asarray(H, dtype=np.float64)
        try:
            n_sol, rotations, _translations, normals = cv2.decomposeHomographyMat(H64, K64)
        except Exception as exc:
            logger.debug("decomposeHomographyMat failed: %s", exc)
            return 0.0

        if n_sol <= 0:
            return 0.0

        q_pts = np.asarray(match_result.query_keypoints, dtype=np.float32)
        r_pts = np.asarray(match_result.ref_keypoints, dtype=np.float32)
        if match_result.inlier_mask is not None:
            m = match_result.inlier_mask.astype(bool)
            q_pts = q_pts[m]
            r_pts = r_pts[m]
        if len(q_pts) >= 4 and hasattr(cv2, "filterHomographyDecompByVisibleRefpoints"):
            q_cv = q_pts.reshape(-1, 1, 2)
            r_cv = r_pts.reshape(-1, 1, 2)
            rots = [rotations[i] for i in range(n_sol)]
            nv = [normals[i] for i in range(n_sol)]
            try:
                sols = cv2.filterHomographyDecompByVisibleRefpoints(rots, nv, q_cv, r_cv)
                sol_arr = np.asarray(sols).ravel()
                if sol_arr.size > 0 and sol_arr[0] is not None:
                    idx = int(sol_arr[0])
                    if 0 <= idx < len(rotations):
                        R = rotations[idx]
                    else:
                        R = None
                    if R is None:
                        return 0.0
                    return float(np.arctan2(R[1, 0], R[0, 0]))
            except cv2.error as exc:
                logger.debug("filterHomographyDecompByVisibleRefpoints: %s", exc)

        best_yaw = 0.0
        best_score = -1.0
        for i in range(n_sol):
            R = np.asarray(rotations[i], dtype=np.float64)
            det = float(np.linalg.det(R))
            if det < 0.1:
                continue
            tr = float(np.trace(R))
            yaw_i = float(np.arctan2(R[1, 0], R[0, 0]))
            score = det * (1.0 + 0.01 * tr)
            if score > best_score:
                best_score = score
                best_yaw = yaw_i
        return best_yaw


class PnPGeometrySolver:
    """Planar PnP solver using reference pixel-to-metric projection."""

    def __init__(self, gsd: float = 0.3, min_inliers: int = 8):
        self._gsd = gsd
        self._min_inliers = min_inliers
        self._fallback = HomographyGeometrySolver(gsd=gsd, min_inliers=min_inliers)

    @staticmethod
    def _meters_per_degree(lat_deg: float) -> tuple[float, float]:
        m_per_deg_lat = 111320.0
        m_per_deg_lon = 111320.0 * np.cos(np.radians(lat_deg))
        return m_per_deg_lat, float(max(m_per_deg_lon, 1.0))

    def solve(
        self,
        match_result: MatchResult,
        reference_pose: Pose,
        camera: CameraIntrinsics,
        query_size=None,
        altitude_hint=None,
        gsd=None,
        ref_size=None,
        gsd_e=None,
        **_kwargs,
    ):
        import cv2

        q_pts, r_pts = match_result.get_inlier_matches()
        if len(q_pts) < self._min_inliers:
            return self._fallback.solve(
                match_result=match_result,
                reference_pose=reference_pose,
                camera=camera,
                query_size=query_size,
                altitude_hint=altitude_hint,
                gsd=gsd,
                ref_size=ref_size,
                gsd_e=gsd_e,
            )

        gsd_to_use = float(gsd if gsd is not None else self._gsd)
        rw = ref_size[0] if ref_size is not None else camera.width
        rh = ref_size[1] if ref_size is not None else camera.height
        cx_ref = rw / 2.0
        cy_ref = rh / 2.0

        # Build planar object points in local ENU-like frame (z=0 map plane).
        # On a square-degree raster the east scale is gsd * cos(latitude); see
        # homography_solver. Metric rasters pass no gsd_e and stay isotropic.
        gsd_east = gsd_e if gsd_e is not None else gsd_to_use
        obj_pts = np.zeros((len(r_pts), 3), dtype=np.float64)
        obj_pts[:, 0] = (r_pts[:, 0] - cx_ref) * gsd_east     # east
        obj_pts[:, 1] = -(r_pts[:, 1] - cy_ref) * gsd_to_use  # north
        img_pts = q_pts.astype(np.float64)

        ok, rvec, tvec, inliers = cv2.solvePnPRansac(
            objectPoints=obj_pts.astype(np.float32),
            imagePoints=img_pts.astype(np.float32),
            cameraMatrix=camera.K.astype(np.float64),
            distCoeffs=None,
            reprojectionError=6.0,
            confidence=0.999,
            iterationsCount=200,
            flags=cv2.SOLVEPNP_EPNP,
        )
        if not ok:
            return self._fallback.solve(
                match_result=match_result,
                reference_pose=reference_pose,
                camera=camera,
                query_size=query_size,
                altitude_hint=altitude_hint,
                gsd=gsd,
                ref_size=ref_size,
                gsd_e=gsd_e,
            )

        R, _ = cv2.Rodrigues(rvec)
        C = -R.T @ tvec.reshape(3, 1)  # camera center in map plane frame
        east_m, north_m = float(C[0, 0]), float(C[1, 0])
        lat_ref, lon_ref = float(reference_pose.position[0]), float(reference_pose.position[1])
        meters_per_deg_lat, meters_per_deg_lon = self._meters_per_degree(lat_ref)
        lat = lat_ref + (north_m / meters_per_deg_lat)
        lon = lon_ref + (east_m / meters_per_deg_lon)
        alt = float(altitude_hint if altitude_hint is not None else reference_pose.position[2])
        # solvePnP returns R_world->cam, so the ROWS of R are the camera axes in
        # ENU. Row 1 is image-down, hence -R[1,:] is image-up, which is the
        # heading direction. atan2(R[1,0], R[0,0]) is the yaw of world->camera,
        # that is, the negative of the vehicle heading.
        yaw = float(np.arctan2(-R[1, 0], -R[1, 1]))
        qw = np.cos(yaw / 2.0)
        qz = np.sin(yaw / 2.0)
        pose = Pose(
            timestamp=reference_pose.timestamp,
            position=np.array([lat, lon, alt], dtype=np.float64),
            orientation=np.array([qw, 0.0, 0.0, qz], dtype=np.float64),
        )
        return pose


class DSMBackedPnPGeometrySolver:
    """DSM-aware 2D-3D PnP pose head.

    This solver is intentionally fail-closed when no DSM map is available. It is
    the experimental 6-DoF path; homography remains a separate diagnostic head.
    """

    def __init__(
        self,
        dsm_map: BaseDSMMap | None = None,
        min_inliers: int = 8,
        reprojection_error_px: float = 6.0,
        strict_dsm: bool = True,
    ):
        self._dsm_map = dsm_map
        self._min_inliers = int(min_inliers)
        self._reprojection_error_px = float(reprojection_error_px)
        self._strict_dsm = bool(strict_dsm)
        self.last_diagnostics = {
            "status": "not_run",
            "dsm_available": bool(getattr(dsm_map, "available", False)),
        }

    @staticmethod
    def _quaternion_from_rotation(R: np.ndarray) -> np.ndarray:
        R = np.asarray(R, dtype=np.float64)
        tr = float(np.trace(R))
        if tr > 0.0:
            s = np.sqrt(tr + 1.0) * 2.0
            qw = 0.25 * s
            qx = (R[2, 1] - R[1, 2]) / s
            qy = (R[0, 2] - R[2, 0]) / s
            qz = (R[1, 0] - R[0, 1]) / s
        else:
            idx = int(np.argmax(np.diag(R)))
            if idx == 0:
                s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
                qw = (R[2, 1] - R[1, 2]) / s
                qx = 0.25 * s
                qy = (R[0, 1] + R[1, 0]) / s
                qz = (R[0, 2] + R[2, 0]) / s
            elif idx == 1:
                s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
                qw = (R[0, 2] - R[2, 0]) / s
                qx = (R[0, 1] + R[1, 0]) / s
                qy = 0.25 * s
                qz = (R[1, 2] + R[2, 1]) / s
            else:
                s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
                qw = (R[1, 0] - R[0, 1]) / s
                qx = (R[0, 2] + R[2, 0]) / s
                qy = (R[1, 2] + R[2, 1]) / s
                qz = 0.25 * s
        return np.array([qw, qx, qy, qz], dtype=np.float64)

    def solve(
        self,
        match_result: MatchResult,
        reference_pose: Pose,
        camera: CameraIntrinsics,
        query_size=None,
        altitude_hint=None,
        gsd=None,
        ref_size=None,
        tile_meta=None,
        **_kwargs,
    ):
        import cv2

        self.last_diagnostics = {
            "status": "not_run",
            "dsm_available": bool(getattr(self._dsm_map, "available", False)),
            "pnp_inliers": 0,
            "pnp_reprojection_error_px": float("nan"),
        }
        if self._dsm_map is None or not bool(getattr(self._dsm_map, "available", False)):
            self.last_diagnostics["status"] = "dsm_unavailable"
            return None

        q_pts, r_pts = match_result.get_inlier_matches()
        if len(q_pts) < self._min_inliers:
            self.last_diagnostics["status"] = "insufficient_matches"
            return None

        obj_points = []
        img_points = []
        for q, r in zip(q_pts, r_pts):
            p3d = self._dsm_map.pixel_to_enu(float(r[0]), float(r[1]), tile_meta=tile_meta)
            if p3d is None or not p3d.valid:
                continue
            arr = p3d.as_array()
            if np.isfinite(arr).all():
                obj_points.append(arr)
                img_points.append(np.asarray(q, dtype=np.float64))

        if len(obj_points) < self._min_inliers:
            self.last_diagnostics["status"] = "insufficient_valid_dsm_points"
            self.last_diagnostics["valid_dsm_points"] = int(len(obj_points))
            return None

        obj = np.asarray(obj_points, dtype=np.float64)
        img = np.asarray(img_points, dtype=np.float64)
        ok, rvec, tvec, inliers = cv2.solvePnPRansac(
            objectPoints=obj.astype(np.float32),
            imagePoints=img.astype(np.float32),
            cameraMatrix=camera.K.astype(np.float64),
            distCoeffs=None,
            reprojectionError=self._reprojection_error_px,
            confidence=0.999,
            iterationsCount=500,
            flags=cv2.SOLVEPNP_EPNP,
        )
        if not ok or inliers is None or len(inliers) < self._min_inliers:
            self.last_diagnostics["status"] = "pnp_failed"
            self.last_diagnostics["pnp_inliers"] = 0 if inliers is None else int(len(inliers))
            return None

        inlier_idx = np.asarray(inliers, dtype=np.int32).reshape(-1)
        if len(inlier_idx) >= 6:
            try:
                cv2.solvePnPRefineLM(
                    objectPoints=obj[inlier_idx].astype(np.float32),
                    imagePoints=img[inlier_idx].astype(np.float32),
                    cameraMatrix=camera.K.astype(np.float64),
                    distCoeffs=None,
                    rvec=rvec,
                    tvec=tvec,
                )
            except Exception as exc:
                logger.debug("solvePnPRefineLM failed: %s", exc)

        R, _ = cv2.Rodrigues(rvec)
        center = (-R.T @ tvec.reshape(3, 1)).reshape(3)
        projected, _ = cv2.projectPoints(obj[inlier_idx], rvec, tvec, camera.K.astype(np.float64), None)
        residuals = np.linalg.norm(projected.reshape(-1, 2) - img[inlier_idx], axis=1)
        reproj = float(np.mean(residuals)) if len(residuals) else float("nan")

        lat, lon = self._dsm_map.enu_to_latlon(
            float(center[0]),
            float(center[1]),
            float(reference_pose.position[0]),
            float(reference_pose.position[1]),
        )
        # R is R_world->cam, so its ZYX yaw is the yaw of world->camera, which is
        # the negative of the vehicle heading. The ROWS of R are the camera axes
        # in ENU, so -R[1,:] (image-up) gives the heading direction.
        #
        # A = R @ NADIR is the camera attitude relative to a nadir, north-up
        # camera. That is exactly the Ry(pitch) Rx(roll) Rz(-yaw) which ipm.py
        # inverts, so roll and pitch come out in the convention the runner
        # differences against ground truth. Do not publish a yaw-only quaternion
        # here: the runner reads euler_angles[0] and [1] into pred_roll and
        # pred_pitch and into rotation_geodesic_error_deg, so zeroing them would
        # report a constant instead of a measurement for a 6-DoF head.
        NADIR = np.diag([1.0, -1.0, -1.0])
        A = R @ NADIR
        yaw_rad = float(np.arctan2(-R[1, 0], -R[1, 1]))
        roll_rad = float(-np.arcsin(np.clip(A[1, 2], -1.0, 1.0)))
        _b0 = np.array([np.cos(yaw_rad), np.sin(yaw_rad), 0.0])
        pitch_rad = float(np.arctan2(-float(A[2] @ _b0), float(A[0] @ _b0)))
        _cy, _sy = np.cos(yaw_rad / 2.0), np.sin(yaw_rad / 2.0)
        _cp, _sp = np.cos(pitch_rad / 2.0), np.sin(pitch_rad / 2.0)
        _cr, _sr = np.cos(roll_rad / 2.0), np.sin(roll_rad / 2.0)
        quat = np.array(
            [
                _cr * _cp * _cy + _sr * _sp * _sy,
                _sr * _cp * _cy - _cr * _sp * _sy,
                _cr * _sp * _cy + _sr * _cp * _sy,
                _cr * _cp * _sy - _sr * _sp * _cy,
            ],
            dtype=np.float64,
        )
        pose = Pose(
            timestamp=reference_pose.timestamp,
            position=np.array([lat, lon, float(center[2])], dtype=np.float64),
            orientation=quat,
        )
        euler_deg = np.degrees(pose.euler_angles)
        self.last_diagnostics = {
            "status": "ok",
            "dsm_available": True,
            "valid_dsm_points": int(len(obj_points)),
            "pnp_inliers": int(len(inlier_idx)),
            "pnp_reprojection_error_px": reproj,
            "pnp_camera_east_m": float(center[0]),
            "pnp_camera_north_m": float(center[1]),
            "pnp_camera_z_m": float(center[2]),
            "pnp_roll_deg": float(euler_deg[0]),
            "pnp_pitch_deg": float(euler_deg[1]),
            "pnp_yaw_deg": float(euler_deg[2]),
        }
        return pose


def create_geometry_solver(name: str, gsd: float = 0.3, **kwargs):
    """Factory for geometry solver backend."""
    if name == "pnp":
        return PnPGeometrySolver(
            gsd=gsd,
            min_inliers=int(kwargs.get("min_inliers", 8)),
        )
    elif name == "dsm_pnp":
        return DSMBackedPnPGeometrySolver(
            dsm_map=kwargs.get("dsm_map"),
            min_inliers=int(kwargs.get("min_inliers", 8)),
            reprojection_error_px=float(kwargs.get("reprojection_error_px", 6.0)),
            strict_dsm=bool(kwargs.get("strict_dsm", True)),
        )
    elif name == "homography_scale":
        from .homography_solver import HomographyYawHeightSolver
        # altitude_source="hint" by default: the scale-from-correspondences
        # estimator is uncorrelated with truth on VPAIR (r = -0.07 over 2,201
        # accepted frames) because a single planar homography cannot separate
        # altitude from map scale. The protocol supplies altitude as telemetry,
        # which is what the wrapper pipelines consume (2.2 m error vs 333 m).
        return HomographyYawHeightSolver(
            gsd=gsd,
            altitude_source=str(kwargs.get("altitude_source", "hint")),
        )
    return HomographyGeometrySolver(gsd=gsd)
