"""
Analytical Ray-Plane Intersection for 3D Point Recovery.

RESEARCH CONTRIBUTION (Proposal §3.3, Eq. 4):
Recovers 3D world coordinates from 2D pixel matches by intersecting
camera rays with the ground plane, using IMU attitude for plane orientation
and GSD-derived scale for metric scale recovery.

Key advantages:
- O(N) complexity for N matches (vs O(N²) for dense depth)
- No neural network — fully deterministic
- Bounded by GSD accuracy
- Provides metric 3D points for PnP or factor graph optimization

Algorithm:
    For each matched pixel p = (u, v):
    1. Ray direction in camera: d_cam = K⁻¹ · [u, v, 1]ᵀ
    2. Ray in world frame: d_world = R_imu · d_cam
    3. Intersect with ground plane (Z = Z_ground):
       t = (Z_ground - C_z) / d_world_z
       X_world = C + t · d_world
    4. Convert to metric using scale factor s from GSD ratio

Reference:
    - Ma et al., "An Invitation to 3-D Vision" (Springer, 2004)
    - Proposal §3.3: Metric Scale Recovery
"""

import numpy as np
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class RayPlaneIntersector:
    """
    Compute 3D world points from 2D pixel matches via ray-plane intersection.
    
    Assumes a locally planar ground surface. The ground plane height
    is determined either from:
    - IMU altitude (barometric) 
    - GSD-based scale estimation
    - Fixed altitude hint
    """
    
    def __init__(self,
                 camera_matrix: np.ndarray = None,
                 default_altitude: float = 100.0,
                 gsd: float = 0.3):
        """
        Initialize ray-plane intersector.
        
        Args:
            camera_matrix: 3x3 camera intrinsic matrix K
            default_altitude: Default camera altitude above ground (meters)
            gsd: Ground Sampling Distance of reference map (meters/pixel)
        """
        self._K = camera_matrix
        self._K_inv = np.linalg.inv(camera_matrix) if camera_matrix is not None else None
        self._default_alt = default_altitude
        self._gsd = gsd
    
    def set_camera(self, K: np.ndarray):
        """Set camera intrinsic matrix."""
        self._K = np.asarray(K, dtype=np.float64)
        self._K_inv = np.linalg.inv(self._K)
    
    def compute_ground_plane_height(self,
                                     R_imu: np.ndarray,
                                     altitude: float = None,
                                     scale_s: float = None) -> float:
        """
        Compute ground plane height in camera frame.
        
        Project convention: altitude is positive AGL (upward), and ground
        plane for local reconstruction is Z = 0.
        
        If scale_s is provided (from similarity transform), altitude
        is recovered as: altitude = f * GSD / scale_s
        
        Args:
            R_imu: 3x3 rotation matrix from IMU (body-to-world)
            altitude: Known altitude above ground (meters)
            scale_s: Scale ratio from similarity transform
            
        Returns:
            Ground plane height Z_ground in world frame
        """
        if altitude is not None:
            return 0.0
        
        if scale_s is not None and scale_s > 0:
            # Recover altitude from scale and GSD
            # scale_s = GSD_query / GSD_ref
            # GSD_query = altitude * pixel_size / focal_length
            # altitude = GSD_ref * scale_s * focal_length / pixel_size
            # Simplified: altitude ≈ GSD_ref * focal_length (pixels) for scale_s ≈ 1
            if self._K is not None:
                f = (self._K[0, 0] + self._K[1, 1]) / 2.0
                altitude = self._gsd * f / scale_s
                altitude = np.clip(altitude, 10.0, 500.0)
                return 0.0
        
        return 0.0
    
    def intersect_single(self,
                          pixel: np.ndarray,
                          R_cam_to_world: np.ndarray,
                          camera_center: np.ndarray,
                          z_ground: float = 0.0) -> Optional[np.ndarray]:
        """
        Intersect a single camera ray with the ground plane.
        
        Args:
            pixel: 2D pixel coordinate [u, v]
            R_cam_to_world: 3x3 rotation from camera to world frame
            camera_center: Camera center position in world frame [X, Y, Z]
            z_ground: Ground plane height in world frame (typically 0)
            
        Returns:
            3D world point [X, Y, Z] or None if ray is parallel to plane
        """
        if self._K_inv is None:
            raise ValueError("Camera matrix not set")
        
        # Ray direction in camera frame
        p_hom = np.array([pixel[0], pixel[1], 1.0])
        d_cam = self._K_inv @ p_hom
        d_cam = d_cam / np.linalg.norm(d_cam)
        
        # Ray direction in world frame
        d_world = R_cam_to_world @ d_cam
        
        # Intersect with ground plane Z = z_ground
        # C + t * d = [X, Y, z_ground]
        # t = (z_ground - C_z) / d_z
        if abs(d_world[2]) < 1e-8:
            # Ray parallel to ground plane — no intersection
            return None
        
        t = (z_ground - camera_center[2]) / d_world[2]
        
        if t < 0:
            # Intersection behind camera
            return None
        
        world_pt = camera_center + t * d_world
        return world_pt
    
    def intersect_batch(self,
                         pixels: np.ndarray,
                         R_cam_to_world: np.ndarray,
                         camera_center: np.ndarray,
                         z_ground: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Vectorized ray-plane intersection for multiple pixels.
        
        Args:
            pixels: Nx2 array of pixel coordinates [u, v]
            R_cam_to_world: 3x3 rotation from camera to world
            camera_center: Camera center in world frame [X, Y, Z]
            z_ground: Ground plane height
            
        Returns:
            Tuple of:
                - world_points: Nx3 array of 3D world points (NaN for invalid)
                - valid_mask: N boolean array (True where intersection is valid)
        """
        if self._K_inv is None:
            raise ValueError("Camera matrix not set")
        
        N = len(pixels)
        
        # Homogeneous pixel coordinates: Nx3
        ones = np.ones((N, 1))
        p_hom = np.hstack([pixels, ones])  # Nx3
        
        # Ray directions in camera frame: Nx3
        d_cam = (self._K_inv @ p_hom.T).T  # Nx3
        norms = np.linalg.norm(d_cam, axis=1, keepdims=True)
        d_cam = d_cam / (norms + 1e-12)
        
        # Ray directions in world frame: Nx3
        d_world = (R_cam_to_world @ d_cam.T).T  # Nx3
        
        # Intersection parameters: t = (z_ground - C_z) / d_z
        d_z = d_world[:, 2]
        
        # Valid rays: not parallel to ground, intersection in front of camera
        valid_dz = np.abs(d_z) > 1e-8
        t = np.full(N, np.nan)
        t[valid_dz] = (z_ground - camera_center[2]) / d_z[valid_dz]
        
        valid_t = t > 0
        valid_mask = valid_dz & valid_t
        
        # Compute world points
        world_points = np.full((N, 3), np.nan)
        if valid_mask.any():
            t_valid = t[valid_mask, np.newaxis]  # Mx1
            d_valid = d_world[valid_mask]  # Mx3
            world_points[valid_mask] = camera_center + t_valid * d_valid
        
        logger.debug(
            f"Ray-plane intersection: {valid_mask.sum()}/{N} valid "
            f"({100*valid_mask.sum()/N:.0f}%)"
        )
        
        return world_points, valid_mask
    
    def pixel_to_metric_offset(self,
                                query_pixels: np.ndarray,
                                ref_pixels: np.ndarray,
                                R_imu: np.ndarray = None,
                                altitude: float = None) -> np.ndarray:
        """
        Convert pixel-to-pixel correspondences to metric 3D offsets.
        
        For each match pair (q_i, r_i), computes the metric displacement
        on the ground plane using the reference map GSD.
        
        Simplified formula (for nadir-looking camera):
            ΔX_meters = Δpx * GSD
        
        Args:
            query_pixels: Nx2 query keypoints
            ref_pixels: Nx2 reference keypoints (in geo-referenced map)
            R_imu: Optional 3x3 IMU rotation (for oblique compensation)
            altitude: Camera altitude for scale correction
            
        Returns:
            Nx3 metric offsets [dN, dE, dAlt] in NED frame
        """
        N = len(query_pixels)
        
        # Pixel displacements in reference image
        dx_px = ref_pixels[:, 0] - query_pixels[:, 0]  # positive = East
        dy_px = ref_pixels[:, 1] - query_pixels[:, 1]  # positive = South (image Y)
        
        # Convert to metric using GSD
        east_m = dx_px * self._gsd
        north_m = -dy_px * self._gsd  # Flip Y to get North
        
        # Altitude offset (from scale if available)
        alt_m = np.zeros(N)
        
        offsets = np.column_stack([north_m, east_m, alt_m])  # NED
        
        return offsets
    
    def estimate_altitude_from_scale(self,
                                      query_kpts: np.ndarray,
                                      ref_kpts: np.ndarray,
                                      homography: np.ndarray = None) -> float:
        """
        Estimate camera altitude from homography scale.
        
        If the reference map GSD is known, the UAV altitude can be
        recovered from the scale change in the homography:
        
            s = sqrt(|det(H)|)
            altitude = GSD_ref * f / s
        
        Args:
            query_kpts: Nx2 query keypoints
            ref_kpts: Nx2 reference keypoints
            homography: 3x3 homography matrix (if available)
            
        Returns:
            Estimated altitude in meters
        """
        if homography is not None:
            # Method 1: From homography determinant
            det_H = np.linalg.det(homography)
            scale = np.abs(det_H) ** 0.5
            
            if self._K is not None and scale > 0.01:
                f = (self._K[0, 0] + self._K[1, 1]) / 2.0
                altitude = self._gsd * f / scale
                return float(np.clip(altitude, 10.0, 500.0))
        
        # Method 2: From correspondence distances
        if len(query_kpts) >= 3 and len(ref_kpts) >= 3:
            # Compute scale from average displacement ratio
            q_dists = np.linalg.norm(
                query_kpts[1:] - query_kpts[:-1], axis=1
            )
            r_dists = np.linalg.norm(
                ref_kpts[1:] - ref_kpts[:-1], axis=1
            )
            
            valid = (q_dists > 1.0) & (r_dists > 1.0)
            if valid.sum() > 0:
                scale = np.median(r_dists[valid] / q_dists[valid])
                if self._K is not None and scale > 0.01:
                    f = (self._K[0, 0] + self._K[1, 1]) / 2.0
                    altitude = self._gsd * f / scale
                    return float(np.clip(altitude, 10.0, 500.0))
        
        return self._default_alt
