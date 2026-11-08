"""
IMU-Driven Inverse Perspective Mapping (IPM) Module.

RESEARCH CONTRIBUTION (Proposal §2.2):
Transforms oblique UAV images into pseudo-nadir views using IMU attitude
priors (roll, pitch) to resolve the perspective distortion that degrades
feature matching between UAV imagery and nadir satellite reference maps.

Key advantages over dense depth estimation:
- O(1) analytical computation per pixel (no neural network)
- Deterministic output (no mode collapse or depth hallucination)
- Leverages existing IMU data from flight controller
- Compatible with ground-plane assumption for flat terrain

Algorithm:
    1. Construct R_body_to_nadir from IMU attitude (roll, pitch)
    2. Build IPM homography: H_ipm = K · R_correction · K_inv
    3. Apply horizon culling to remove sky pixels above horizon line
    4. Warp query image using H_ipm to produce pseudo-nadir view

Reference:
    - "Inverse Perspective Mapping" (Bertozzi & Broggi, IEEE IV 1998)
    - Proposal: "Flatten-then-Match" paradigm §2.2
"""

import numpy as np
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class InversePerspectiveMapper:
    """
    IMU-Driven Inverse Perspective Mapping for UAV images.
    
    Transforms oblique UAV imagery to pseudo-nadir view using
    analytical homography computed from IMU attitude priors.
    
    This enables heading-invariant feature matching against
    nadir satellite reference maps without dense depth networks.
    """
    
    def __init__(self,
                 camera_matrix: np.ndarray = None,
                 pitch_threshold_deg: float = 15.0,
                 roll_threshold_deg: float = 15.0,
                 yaw_threshold_deg: float = 15.0,
                 output_scale: float = 1.0,
                 horizon_margin_px: int = 50):
        """
        Initialize IPM module.

        Args:
            camera_matrix: 3x3 camera intrinsic matrix K.
                           If None, will be set later via set_camera().
            pitch_threshold_deg: Minimum pitch angle (from nadir) to apply IPM.
                                 Below this, image is already near-nadir.
            roll_threshold_deg: Minimum roll angle to apply IPM.
            yaw_threshold_deg: Minimum yaw angle to apply IPM.
            output_scale: Scale factor for output image (1.0 = same size).
            horizon_margin_px: Extra margin above horizon line to crop.
        """
        self._K = camera_matrix
        self._K_inv = np.linalg.inv(camera_matrix) if camera_matrix is not None else None
        self._pitch_threshold = np.radians(pitch_threshold_deg)
        self._roll_threshold = np.radians(roll_threshold_deg)
        self._yaw_threshold = np.radians(yaw_threshold_deg)
        self._output_scale = output_scale
        self._horizon_margin = horizon_margin_px
        
        # Cache last computed values for debugging
        self._last_H_ipm = None
        self._last_horizon_row = None
        self._last_yaw_applied_rad = 0.0
    
    def set_camera(self, K: np.ndarray):
        """Set camera intrinsic matrix."""
        self._K = np.asarray(K, dtype=np.float64)
        self._K_inv = np.linalg.inv(self._K)
    
    @staticmethod
    def rotation_x(angle_rad: float) -> np.ndarray:
        """Rotation matrix about X-axis (roll)."""
        c, s = np.cos(angle_rad), np.sin(angle_rad)
        return np.array([
            [1,  0, 0],
            [0,  c, -s],
            [0,  s,  c]
        ])
    
    @staticmethod
    def rotation_y(angle_rad: float) -> np.ndarray:
        """Rotation matrix about Y-axis (pitch)."""
        c, s = np.cos(angle_rad), np.sin(angle_rad)
        return np.array([
            [ c, 0, s],
            [ 0, 1, 0],
            [-s, 0, c]
        ])
    
    @staticmethod
    def rotation_z(angle_rad: float) -> np.ndarray:
        """Rotation matrix about Z-axis (yaw)."""
        c, s = np.cos(angle_rad), np.sin(angle_rad)
        return np.array([
            [c, -s, 0],
            [s,  c, 0],
            [0,  0, 1]
        ])
    
    def compute_ipm_homography(self,
                                pitch_rad: float,
                                roll_rad: float,
                                yaw_rad: float = 0.0) -> np.ndarray:
        """
        Compute IPM homography from IMU attitude.
        
        The homography transforms oblique image pixels to pseudo-nadir
        coordinates by "undoing" the camera tilt.
        
        For a camera looking down with pitch θ from nadir:
        - R_correction rotates the viewing direction to nadir
        - H_ipm = K · R_correction · K⁻¹
        
        Args:
            pitch_rad: Pitch angle in radians (from nadir, 0 = looking down)
            roll_rad: Roll angle in radians
            yaw_rad: Yaw angle in radians (optional, for heading correction)
            
        Returns:
            3x3 IPM homography matrix
            
        Raises:
            ValueError: If camera matrix not set
        """
        if self._K is None:
            raise ValueError("Camera matrix not set. Call set_camera() first.")
        
        # Build rotation that corrects the camera tilt to nadir
        # R_correction = R_yaw(yaw) @ R_roll(-roll) @ R_pitch(-pitch)
        # This "undoes" the yaw, roll and pitch to simulate nadir, North-up view
        R_correction = self.rotation_z(yaw_rad) @ self.rotation_x(-roll_rad) @ self.rotation_y(-pitch_rad)
        
        # IPM homography: H = K · R_correction · K⁻¹
        H_ipm = self._K @ R_correction @ self._K_inv
        
        # Normalize so H[2,2] = 1
        H_ipm = H_ipm / H_ipm[2, 2]
        
        self._last_H_ipm = H_ipm
        
        logger.debug(
            f"IPM homography computed: pitch={np.degrees(pitch_rad):.1f}° "
            f"roll={np.degrees(roll_rad):.1f}° yaw={np.degrees(yaw_rad):.1f}°"
        )
        
        return H_ipm
    
    def compute_horizon_line(self,
                              pitch_rad: float,
                              image_height: int) -> int:
        """
        Compute horizon line row in image coordinates.
        
        Everything above this line is sky and should be culled.
        
        For a camera with pitch θ from nadir:
            horizon_row = cy - fy * tan(π/2 - |pitch|)
        
        Args:
            pitch_rad: Pitch angle from nadir (radians)
            image_height: Image height in pixels
            
        Returns:
            Row index of horizon line (0-indexed from top)
        """
        if self._K is None:
            return 0
        
        fy = self._K[1, 1]
        cy = self._K[1, 2]
        
        # Angle from horizontal to optical axis
        # pitch_from_nadir = |pitch_rad|, so angle from horizontal = π/2 - |pitch|
        angle_from_horizontal = np.pi / 2 - abs(pitch_rad)
        
        if angle_from_horizontal <= 0:
            # Camera pointing at or above horizon — cull entire image
            return image_height
        
        # Horizon is where the ground plane intersects the image
        # In camera coords, horizon is at angle (π/2 - pitch) from optical axis
        horizon_row = int(cy - fy * np.tan(angle_from_horizontal))
        
        # Apply margin
        horizon_row = max(0, horizon_row - self._horizon_margin)
        
        self._last_horizon_row = horizon_row
        
        return horizon_row
    
    def apply_ipm(self,
                   image: np.ndarray,
                   pitch_rad: float,
                   roll_rad: float,
                   yaw_rad: float = 0.0,
                   crop_horizon: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply IPM transformation to UAV image.
        
        Full pipeline:
        1. Compute horizon line and optionally crop sky
        2. Build IPM homography from IMU attitude
        3. Warp image to pseudo-nadir view
        
        Args:
            image: Input UAV image (H, W, C) or (H, W)
            pitch_rad: Pitch from nadir in radians
            roll_rad: Roll in radians
            yaw_rad: Yaw in radians (optional)
            crop_horizon: Whether to crop pixels above horizon
            
        Returns:
            Tuple of (pseudo_nadir_image, H_ipm)
        """
        try:
            import cv2
        except ImportError:
            raise ImportError("OpenCV required: pip install opencv-python")
        
        h, w = image.shape[:2]
        
        # Per-axis near-nadir clamping. An axis below its threshold is treated as
        # exactly nadir: warping with sub-threshold (noise-dominated) pitch/roll
        # displaces the center ray by ~h*tan(noise) and turns attitude noise into
        # position error at altitude scale. Yaw-only rotation is in-plane about
        # the optical axis and cannot displace the center ray, so it is safe to
        # keep for North-alignment.
        if abs(pitch_rad) < self._pitch_threshold:
            pitch_rad = 0.0
        if abs(roll_rad) < self._roll_threshold:
            roll_rad = 0.0
        if abs(yaw_rad) < self._yaw_threshold:
            yaw_rad = 0.0
        # Record the yaw the warp actually removes. The homography measured
        # afterwards is relative to this rotated frame, so the pose solver must
        # add it back; without that the reported heading collapses to ~0.
        self._last_yaw_applied_rad = float(yaw_rad)
        if pitch_rad == 0.0 and roll_rad == 0.0 and yaw_rad == 0.0:
            logger.debug("Attitude below thresholds on all axes — skipping IPM")
            return image.copy(), np.eye(3)
        
        # Step 1: Horizon culling
        work_image = image.copy()
        if crop_horizon:
            horizon_row = self.compute_horizon_line(pitch_rad, h)
            if 0 < horizon_row < h:
                # Zero out sky region
                work_image[:horizon_row, :] = 0
                logger.debug(f"Horizon culling: rows 0-{horizon_row} (sky)")
        
        # Step 2: Compute IPM homography
        H_ipm = self.compute_ipm_homography(pitch_rad, roll_rad, yaw_rad)
        
        # Step 3: Warp image
        out_h = int(h * self._output_scale)
        out_w = int(w * self._output_scale)
        
        pseudo_nadir = cv2.warpPerspective(
            work_image, H_ipm, (out_w, out_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0
        )
        
        logger.debug(f"IPM applied: {w}×{h} → {out_w}×{out_h}")
        
        return pseudo_nadir, H_ipm
    
    def apply_ipm_from_metadata(self,
                                 image: np.ndarray,
                                 metadata: dict,
                                 crop_horizon: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply IPM using dataset metadata (pitch, roll, yaw).
        
        Convenience wrapper that extracts attitude from metadata dict.
        Supports various metadata key conventions.
        
        Args:
            image: Input UAV image
            metadata: Dict with attitude angles (degrees or radians)
            crop_horizon: Whether to crop horizon
            
        Returns:
            Tuple of (pseudo_nadir_image, H_ipm)
        """
        # Extract pitch (various key names)
        pitch_deg = metadata.get('pitch', metadata.get('gimbal_pitch',
                    metadata.get('pitch_deg', metadata.get('tilt', 0.0))))
        roll_deg = metadata.get('roll', metadata.get('gimbal_roll',
                   metadata.get('roll_deg', 0.0)))
        yaw_deg = metadata.get('yaw', metadata.get('heading',
                  metadata.get('yaw_deg', 0.0)))
        
        pitch_rad = np.radians(pitch_deg)
        roll_rad = np.radians(roll_deg)
        yaw_rad = np.radians(yaw_deg)

        # Convert pitch to our convention: 0 = nadir, 90 = horizon.
        # An explicit 'pitch_convention' from the dataset adapter takes priority;
        # the heuristic below cannot distinguish aircraft-convention small tilts
        # (VPAIR: 0 = nadir) from UAVD4L angles-from-horizontal in 0..45.
        convention = str(metadata.get('pitch_convention', '')).lower()
        if convention == 'gimbal_neg90_nadir':
            # Gimbal: -90 = nadir, 0 = horizon (AVL)
            pitch_from_nadir = np.radians(90.0 - abs(pitch_deg))
        elif convention == 'aircraft_nadir_zero':
            # Aircraft/body: 0 = nadir camera, small values = slight tilt (VPAIR)
            pitch_from_nadir = abs(pitch_rad)
        elif pitch_deg <= -45:
            # Heuristic: gimbal convention (-90 = nadir)
            pitch_from_nadir = np.radians(90.0 - abs(pitch_deg))
        elif abs(pitch_deg) < 45:
            # Heuristic: near-nadir aircraft convention (0 = nadir)
            pitch_from_nadir = abs(pitch_rad)
        elif 45 <= pitch_deg < 90:
            # Heuristic: UAVD4L convention (angle from horizontal)
            pitch_from_nadir = np.radians(90.0 - pitch_deg)
        else:
            pitch_from_nadir = abs(pitch_rad)
        
        return self.apply_ipm(image, pitch_from_nadir, roll_rad, yaw_rad, crop_horizon)
    
    @property
    def last_homography(self) -> Optional[np.ndarray]:
        """Get the last computed IPM homography."""
        return self._last_H_ipm
    
    @property
    def last_horizon_row(self) -> Optional[int]:
        """Get the last computed horizon row."""
        return self._last_horizon_row

    @property
    def last_yaw_applied_rad(self) -> float:
        """Yaw removed by the last warp; add back to recover absolute heading."""
        return float(getattr(self, "_last_yaw_applied_rad", 0.0))


class IPMFactory:
    """Factory for creating IPM module with common camera configurations."""
    
    # Common drone camera presets
    PRESETS = {
        'dji_mini3': {
            'fx': 2782.0, 'fy': 2782.0,
            'cx': 2016.0, 'cy': 1512.0,
            'width': 4032, 'height': 3024,
        },
        'dji_mavic3': {
            'fx': 3550.0, 'fy': 3550.0,
            'cx': 2736.0, 'cy': 1824.0,
            'width': 5472, 'height': 3648,
        },
        'generic_720p': {
            'fx': 1000.0, 'fy': 1000.0,
            'cx': 640.0, 'cy': 360.0,
            'width': 1280, 'height': 720,
        },
        'vpair': {
            # VPAir dataset camera calibration
            'fx': 614.2, 'fy': 614.2,
            'cx': 320.0, 'cy': 240.0,
            'width': 640, 'height': 480,
        },
    }
    
    @classmethod
    def create(cls,
               camera_preset: str = None,
               camera_matrix: np.ndarray = None,
               **kwargs) -> InversePerspectiveMapper:
        """
        Create IPM module.
        
        Args:
            camera_preset: Name of camera preset ('dji_mini3', 'generic_720p', etc.)
            camera_matrix: Custom 3x3 camera intrinsic matrix
            **kwargs: Additional arguments for InversePerspectiveMapper
            
        Returns:
            Configured InversePerspectiveMapper
        """
        if camera_matrix is not None:
            K = np.asarray(camera_matrix, dtype=np.float64)
        elif camera_preset and camera_preset in cls.PRESETS:
            preset = cls.PRESETS[camera_preset]
            K = np.array([
                [preset['fx'], 0, preset['cx']],
                [0, preset['fy'], preset['cy']],
                [0, 0, 1]
            ], dtype=np.float64)
        else:
            # Default generic camera
            K = np.array([
                [1000, 0, 640],
                [0, 1000, 360],
                [0, 0, 1]
            ], dtype=np.float64)
        
        return InversePerspectiveMapper(camera_matrix=K, **kwargs)
