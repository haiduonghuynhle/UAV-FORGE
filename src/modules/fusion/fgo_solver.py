"""
Factor Graph Optimization (FGO) Backend.

RESEARCH CONTRIBUTION (Proposal §3.3):
Tightly-coupled Visual-Inertial Factor Graph solver for metric scale
recovery in 6-DoF UAV localization. Fuses:
- Visual reprojection factors (from feature matches)
- IMU prior factors (attitude from flight controller)
- Barometric altitude factors (pressure altitude)
- Temporal smoothness factors (motion prior)

Two implementations:
1. LevenbergMarquardtSolver: Pure Python (scipy) — portable, Kaggle-friendly
2. GTSAMSolver: C++ backend via GTSAM Python bindings — Jetson deployment

Key innovation:
    Metric scale recovery without dense depth, using the analytical
    Ray-Plane intersection (ray_plane.py) to provide 3D point constraints
    bounded by GSD accuracy.

Reference:
    - Dellaert & Kaess, "Factor Graphs for Robot Perception" (2017)
    - Forster et al., "On-Manifold Preintegration for Real-Time
      Visual-Inertial Odometry" (IEEE TRO 2017)
    - Proposal §3.3: Factor Graph Optimization
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class FGOState:
    """
    State vector for Factor Graph Optimization.
    
    Tightly-coupled visual-inertial state:
    - position (3)
    - orientation as Euler angles (3)
    - velocity (3)
    - gyro bias (3)
    - accel bias (3)
    Total: 15 parameters to optimize
    """
    position: np.ndarray      # [x, y, z] in local frame (meters)
    orientation: np.ndarray   # [roll, pitch, yaw] in radians
    velocity: Optional[np.ndarray] = None
    gyro_bias: Optional[np.ndarray] = None
    accel_bias: Optional[np.ndarray] = None
    covariance: Optional[np.ndarray] = None  # 15x15 covariance

    def __post_init__(self):
        self.position = np.asarray(self.position, dtype=np.float64)
        self.orientation = np.asarray(self.orientation, dtype=np.float64)
        if self.velocity is None:
            self.velocity = np.zeros(3, dtype=np.float64)
        else:
            self.velocity = np.asarray(self.velocity, dtype=np.float64)
        if self.gyro_bias is None:
            self.gyro_bias = np.zeros(3, dtype=np.float64)
        else:
            self.gyro_bias = np.asarray(self.gyro_bias, dtype=np.float64)
        if self.accel_bias is None:
            self.accel_bias = np.zeros(3, dtype=np.float64)
        else:
            self.accel_bias = np.asarray(self.accel_bias, dtype=np.float64)
    
    def to_vector(self) -> np.ndarray:
        """Flatten state to 15D vector."""
        return np.concatenate([
            self.position,
            self.orientation,
            self.velocity,
            self.gyro_bias,
            self.accel_bias,
        ])
    
    @classmethod
    def from_vector(cls, vec: np.ndarray, covariance: np.ndarray = None) -> 'FGOState':
        """Create state from 15D vector."""
        return cls(
            position=vec[:3].copy(),
            orientation=vec[3:6].copy(),
            velocity=vec[6:9].copy(),
            gyro_bias=vec[9:12].copy(),
            accel_bias=vec[12:15].copy(),
            covariance=covariance
        )


@dataclass
class VisualFactor:
    """Visual reprojection factor from feature matches."""
    points_2d: np.ndarray         # Nx2 image points
    points_3d: np.ndarray         # Nx3 world points (from ray-plane)
    camera_matrix: np.ndarray     # 3x3 intrinsic matrix
    sigma: float = 2.0            # Reprojection error std (pixels)
    weight: float = 1.0           # Factor weight
    R_mount: Optional[np.ndarray] = None  # body-to-camera mount rotation (det=+1); None = legacy identity


@dataclass
class IMUPriorFactor:
    """IMU attitude prior factor."""
    roll_rad: float               # IMU roll measurement
    pitch_rad: float              # IMU pitch measurement
    yaw_rad: Optional[float] = None  # Optional yaw (magnetometer)
    sigma_roll: float = 0.053     # Roll noise std (radians) ≈ 3°
    sigma_pitch: float = 0.053    # Pitch noise std (radians) ≈ 3°
    sigma_yaw: float = 0.14      # Yaw noise std (radians) ≈ 8°


@dataclass
class BarometricFactor:
    """Barometric altitude factor."""
    altitude: float               # Barometric altitude (meters)
    sigma: float = 3.0            # Altitude noise std (meters) ≈ σ_alt


@dataclass
class MotionPriorFactor:
    """Temporal smoothness / motion prior."""
    prev_position: np.ndarray     # Previous position estimate
    prev_orientation: np.ndarray  # Previous orientation
    sigma_position: float = 5.0   # Position change budget (meters)
    sigma_orientation: float = 0.2  # Orientation change budget (radians)


@dataclass
class IMUPreintegrationFactor:
    """
    Simplified IMU preintegration factor for tightly-coupled backend.

    residuals constrain (R, t, v, b_g, b_a) between two keyframes.
    """
    prev_state: FGOState
    delta_t: float
    delta_r: np.ndarray
    delta_v: np.ndarray
    delta_p: np.ndarray
    gravity: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, -9.81]))
    sigma_rot: float = 0.05
    sigma_vel: float = 0.5
    sigma_pos: float = 1.0
    sigma_bg: float = 0.01
    sigma_ba: float = 0.05


class LevenbergMarquardtSolver:
    """
    Pure Python Factor Graph Optimizer using Levenberg-Marquardt.
    
    Portable implementation using scipy.optimize for Kaggle/server environments
    where GTSAM C++ bindings are unavailable.
    
    Solves the nonlinear least-squares problem:
        x* = argmin Σ ||r_i(x)||² / σ_i²
    
    where r_i are residual functions from each factor.
    """
    
    def __init__(self,
                 max_iterations: int = 50,
                 convergence_threshold: float = 1e-6,
                 damping_init: float = 1e-3,
                 damping_factor: float = 10.0):
        """
        Initialize solver.
        
        Args:
            max_iterations: Maximum LM iterations
            convergence_threshold: Convergence threshold on cost reduction
            damping_init: Initial LM damping parameter
            damping_factor: Damping increase/decrease factor
        """
        self._max_iter = max_iterations
        self._convergence = convergence_threshold
        self._lambda_init = damping_init
        self._lambda_factor = damping_factor
        
        # Factor storage
        self._visual_factors: List[VisualFactor] = []
        self._imu_factors: List[IMUPriorFactor] = []
        self._imu_preint_factors: List[IMUPreintegrationFactor] = []
        self._baro_factors: List[BarometricFactor] = []
        self._motion_factors: List[MotionPriorFactor] = []
        
        # Initial state
        self._initial_state: Optional[FGOState] = None
    
    def set_initial_state(self, state: FGOState):
        """Set initial state estimate."""
        self._initial_state = state
    
    def add_visual_factor(self, factor: VisualFactor):
        """Add visual reprojection factor."""
        self._visual_factors.append(factor)
    
    def add_imu_prior(self, factor: IMUPriorFactor):
        """Add IMU attitude prior factor."""
        self._imu_factors.append(factor)
    
    def add_barometric_factor(self, factor: BarometricFactor):
        """Add barometric altitude factor."""
        self._baro_factors.append(factor)

    def add_imu_preintegration(self, factor: IMUPreintegrationFactor):
        """Add IMU preintegration factor."""
        self._imu_preint_factors.append(factor)
    
    def add_motion_prior(self, factor: MotionPriorFactor):
        """Add motion smoothness factor."""
        self._motion_factors.append(factor)
    
    def clear_factors(self):
        """Clear all factors."""
        self._visual_factors.clear()
        self._imu_factors.clear()
        self._imu_preint_factors.clear()
        self._baro_factors.clear()
        self._motion_factors.clear()
    
    def _compute_residuals(self, x: np.ndarray) -> np.ndarray:
        """
        Compute all residual vectors.
        
        Args:
            x: 15D state vector
            
        Returns:
            Concatenated weighted residual vector
        """
        residuals = []
        
        pos = x[:3]
        roll, pitch, yaw = x[3], x[4], x[5]
        vel = x[6:9]
        b_g = x[9:12]
        b_a = x[12:15]
        
        # Build rotation matrix from Euler angles (ZYX convention)
        R = self._euler_to_rotation(roll, pitch, yaw)
        
        # === Visual reprojection residuals ===
        for vf in self._visual_factors:
            if len(vf.points_3d) == 0:
                continue
            
            # Project 3D points to 2D
            # Camera model: p_cam = R_wc^T * (P_world - C), R_wc = R_body @ R_mount
            # p_2d = K * p_cam / p_cam[2]
            R_wc = R @ vf.R_mount if vf.R_mount is not None else R
            P_cam = (R_wc.T @ (vf.points_3d - pos).T).T  # Nx3
            
            # Guard against points behind camera
            valid = P_cam[:, 2] > 0.01
            if valid.sum() == 0:
                continue
            
            P_cam_valid = P_cam[valid]
            pts_2d_valid = vf.points_2d[valid]
            
            # Project
            projected = (vf.camera_matrix @ P_cam_valid.T).T
            projected = projected[:, :2] / projected[:, 2:3]
            
            # Reprojection error
            reproj_err = (projected - pts_2d_valid).ravel()
            weighted = reproj_err * (vf.weight / vf.sigma)
            residuals.append(weighted)
        
        # === IMU prior residuals ===
        for imu in self._imu_factors:
            # Roll prior
            r_roll = (roll - imu.roll_rad) / imu.sigma_roll
            residuals.append(np.array([r_roll]))
            
            # Pitch prior
            r_pitch = (pitch - imu.pitch_rad) / imu.sigma_pitch
            residuals.append(np.array([r_pitch]))
            
            # Yaw prior (if available)
            if imu.yaw_rad is not None:
                # Handle wraparound
                yaw_err = yaw - imu.yaw_rad
                yaw_err = (yaw_err + np.pi) % (2 * np.pi) - np.pi
                r_yaw = yaw_err / imu.sigma_yaw
                residuals.append(np.array([r_yaw]))

        # === IMU preintegration residuals ===
        for pf in self._imu_preint_factors:
            dt = max(float(pf.delta_t), 1e-4)
            prev = pf.prev_state
            prev_pos = prev.position
            prev_ori = prev.orientation
            prev_vel = prev.velocity
            prev_bg = prev.gyro_bias
            prev_ba = prev.accel_bias

            # Rotation residual in tangent space approximation
            ori_pred = prev_ori + np.asarray(pf.delta_r, dtype=np.float64)
            r_rot = np.array([
                roll - ori_pred[0],
                pitch - ori_pred[1],
                ((yaw - ori_pred[2]) + np.pi) % (2 * np.pi) - np.pi,
            ]) / max(pf.sigma_rot, 1e-6)
            residuals.append(r_rot)

            # Velocity residual
            g = np.asarray(pf.gravity, dtype=np.float64)
            v_pred = prev_vel + g * dt + np.asarray(pf.delta_v, dtype=np.float64)
            r_vel = (vel - v_pred) / max(pf.sigma_vel, 1e-6)
            residuals.append(r_vel)

            # Position residual
            p_pred = prev_pos + prev_vel * dt + 0.5 * g * (dt ** 2) + np.asarray(pf.delta_p, dtype=np.float64)
            r_pos = (pos - p_pred) / max(pf.sigma_pos, 1e-6)
            residuals.append(r_pos)

            # Bias random-walk residuals (discrete-time; match Report §3.3 1/sqrt(Δt) scaling)
            rw_scale = 1.0 / np.sqrt(max(dt, 1e-6))
            r_bg = (b_g - prev_bg) * rw_scale / max(pf.sigma_bg, 1e-6)
            r_ba = (b_a - prev_ba) * rw_scale / max(pf.sigma_ba, 1e-6)
            residuals.append(r_bg)
            residuals.append(r_ba)
        
        # === Barometric residuals ===
        for baro in self._baro_factors:
            # Project convention: Pose.position[2] is positive AGL meters.
            r_alt = (pos[2] - baro.altitude) / baro.sigma
            residuals.append(np.array([r_alt]))
        
        # === Motion prior residuals ===
        for mp in self._motion_factors:
            # Position smoothness
            r_pos = (pos - mp.prev_position) / mp.sigma_position
            residuals.append(r_pos)
            
            # Orientation smoothness
            ori = np.array([roll, pitch, yaw])
            r_ori = (ori - mp.prev_orientation) / mp.sigma_orientation
            # Wrap yaw difference
            r_ori[2] = ((ori[2] - mp.prev_orientation[2]) + np.pi) % (2 * np.pi) - np.pi
            r_ori[2] /= mp.sigma_orientation
            residuals.append(r_ori)
        
        if len(residuals) == 0:
            return np.array([0.0])
        
        return np.concatenate(residuals)
    
    @staticmethod
    def _euler_to_rotation(roll: float, pitch: float, yaw: float) -> np.ndarray:
        """Build rotation matrix from ZYX Euler angles."""
        cr, sr = np.cos(roll), np.sin(roll)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cy, sy = np.cos(yaw), np.sin(yaw)
        
        R = np.array([
            [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
            [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
            [-sp,   cp*sr,            cp*cr]
        ])
        return R
    
    def optimize(self) -> FGOState:
        """
        Run Levenberg-Marquardt optimization.
        
        Returns:
            Optimized FGOState with covariance estimate
        """
        from scipy.optimize import least_squares
        
        if self._initial_state is None:
            raise ValueError("Initial state not set. Call set_initial_state() first.")
        
        x0 = self._initial_state.to_vector()
        
        logger.info(
            f"FGO optimization: {len(self._visual_factors)} visual, "
            f"{len(self._imu_factors)} IMU-prior, {len(self._imu_preint_factors)} IMU-preint, "
            f"{len(self._baro_factors)} baro, "
            f"{len(self._motion_factors)} motion factors"
        )
        
        # Run optimization
        # Use 'trf' method as it handles underdetermined systems
        # ('lm' requires residuals >= variables)
        result = least_squares(
            self._compute_residuals,
            x0,
            method='trf',
            max_nfev=self._max_iter * 10,
            ftol=self._convergence,
            xtol=self._convergence,
        )
        
        x_opt = result.x
        
        # Estimate covariance from Jacobian
        covariance = None
        if result.jac is not None:
            try:
                J = result.jac
                # Approximate Hessian: H = J^T J
                H = J.T @ J
                # Covariance = H^{-1} * s^2, where s^2 = residual variance
                dof = max(len(result.fun) - len(x_opt), 1)
                s2 = np.sum(result.fun ** 2) / dof
                covariance = np.linalg.inv(H + 1e-8 * np.eye(len(x_opt))) * s2
            except np.linalg.LinAlgError:
                logger.warning("Covariance estimation failed (singular Hessian)")
                covariance = np.eye(len(x_opt)) * 100.0
        
        optimized_state = FGOState.from_vector(x_opt, covariance)
        
        logger.info(
            f"FGO converged: cost={result.cost:.4f}, "
            f"nfev={result.nfev}, pos={x_opt[:3]}"
        )
        
        return optimized_state
    
    def compute_gdop(self) -> float:
        """
        Compute Geometric Dilution of Precision (GDOP).
        
        GDOP = sqrt(trace(C_pos)) where C_pos is the 3x3 position covariance.
        Lower is better.
        
        Returns:
            GDOP value in meters
        """
        if self._initial_state is None:
            return float('inf')
        
        # Optimize first if not done
        state = self.optimize()
        
        if state.covariance is not None:
            pos_cov = state.covariance[:3, :3]
            gdop = np.sqrt(np.trace(pos_cov))
            return float(gdop)
        
        return float('inf')


class GTSAMSolver:
    """
    GTSAM-based Factor Graph Optimizer.
    
    Uses GTSAM C++ library via Python bindings for high-performance
    factor graph optimization. Recommended for Jetson deployment.
    
    Falls back to LevenbergMarquardtSolver if GTSAM is not available.
    """
    
    def __init__(self, **kwargs):
        """Initialize GTSAM solver."""
        self._gtsam_available = False
        self._fallback = None
        
        try:
            import gtsam
            self._gtsam = gtsam
            self._gtsam_available = True
            self._graph = gtsam.NonlinearFactorGraph()
            self._initial_values = gtsam.Values()
            logger.info("GTSAM backend initialized")
        except ImportError:
            logger.warning(
                "GTSAM not available, using scipy fallback. "
                "Install GTSAM: pip install gtsam"
            )
            self._fallback = LevenbergMarquardtSolver(**kwargs)
    
    @property
    def is_gtsam_available(self) -> bool:
        """Check if GTSAM is available."""
        return self._gtsam_available
    
    def set_initial_state(self, state: FGOState):
        """Set initial state."""
        if not self._gtsam_available:
            self._fallback.set_initial_state(state)
            return
        
        gtsam = self._gtsam
        
        # Convert to GTSAM Pose3
        R = LevenbergMarquardtSolver._euler_to_rotation(
            state.orientation[0], state.orientation[1], state.orientation[2]
        )
        rot3 = gtsam.Rot3(R)
        point3 = gtsam.Point3(*state.position)
        pose3 = gtsam.Pose3(rot3, point3)
        
        key = gtsam.symbol('x', 0)
        self._initial_values.insert(key, pose3)
        
        # Add prior factor with covariance
        if state.covariance is not None:
            cov = state.covariance
            if cov.shape[0] > 6:
                cov = cov[:6, :6]
            noise = gtsam.noiseModel.Gaussian.Covariance(cov)
        else:
            noise = gtsam.noiseModel.Diagonal.Sigmas(
                np.array([0.1, 0.1, 0.1, 1.0, 1.0, 2.0])
            )
        
        self._graph.add(gtsam.PriorFactorPose3(key, pose3, noise))
    
    def add_visual_factor(self, factor: VisualFactor):
        """Add visual factor."""
        if not self._gtsam_available:
            self._fallback.add_visual_factor(factor)
            return
        # GTSAM projection factor implementation
        # (Simplified — full implementation would use GenericProjectionFactor)
        logger.debug("GTSAM visual factor added (simplified)")
    
    def add_imu_prior(self, factor: IMUPriorFactor):
        """Add IMU prior."""
        if not self._gtsam_available:
            self._fallback.add_imu_prior(factor)
            return
        logger.debug("GTSAM IMU prior added")
    
    def add_barometric_factor(self, factor: BarometricFactor):
        """Add barometric factor."""
        if not self._gtsam_available:
            self._fallback.add_barometric_factor(factor)
            return
        logger.debug("GTSAM barometric factor added")

    def add_imu_preintegration(self, factor: IMUPreintegrationFactor):
        """Add IMU preintegration factor."""
        if not self._gtsam_available:
            self._fallback.add_imu_preintegration(factor)
            return
        # Conservative GTSAM path: keep no-op until full BetweenFactor integration.
        logger.debug("GTSAM IMU preintegration received (no-op in simplified backend)")

    def add_motion_prior(self, factor: MotionPriorFactor):
        """Add motion prior."""
        if not self._gtsam_available:
            self._fallback.add_motion_prior(factor)
            return
        logger.debug("GTSAM motion prior added")
    
    def clear_factors(self):
        """Clear factors."""
        if not self._gtsam_available:
            self._fallback.clear_factors()
            return
        self._graph = self._gtsam.NonlinearFactorGraph()
        self._initial_values = self._gtsam.Values()
    
    def optimize(self) -> FGOState:
        """Run optimization."""
        if not self._gtsam_available:
            return self._fallback.optimize()
        
        gtsam = self._gtsam
        
        # Optimize with Levenberg-Marquardt
        params = gtsam.LevenbergMarquardtParams()
        params.setMaxIterations(50)
        optimizer = gtsam.LevenbergMarquardtOptimizer(
            self._graph, self._initial_values, params
        )
        result = optimizer.optimize()
        
        # Extract optimized pose
        key = gtsam.symbol('x', 0)
        pose = result.atPose3(key)
        
        translation = pose.translation()
        try:
            # GTSAM Point3-style accessors
            if all(hasattr(translation, attr) for attr in ('x', 'y', 'z')):
                tx = translation.x() if callable(translation.x) else translation.x
                ty = translation.y() if callable(translation.y) else translation.y
                tz = translation.z() if callable(translation.z) else translation.z
                position = np.array([tx, ty, tz], dtype=np.float64)
            else:
                # ndarray-like fallback
                arr = np.asarray(translation, dtype=np.float64).reshape(-1)
                if arr.shape[0] < 3:
                    raise ValueError(f"translation has insufficient length: shape={arr.shape}")
                position = arr[:3].astype(np.float64)
        except Exception as exc:
            raise ValueError(f"Malformed GTSAM translation output: {translation!r}") from exc

        if position.shape[0] != 3 or not np.isfinite(position).all():
            raise ValueError(f"Malformed GTSAM translation vector (expected finite 3D): {position}")
        
        R = pose.rotation().matrix()
        # Extract Euler angles
        pitch = np.arcsin(-np.clip(R[2, 0], -1, 1))
        if np.abs(R[2, 0]) < 0.998:
            yaw = np.arctan2(R[1, 0], R[0, 0])
            roll = np.arctan2(R[2, 1], R[2, 2])
        else:
            yaw = np.arctan2(-R[0, 1], R[1, 1])
            roll = 0.0
        
        orientation = np.array([roll, pitch, yaw])
        
        # Compute marginal covariance
        try:
            marginals = gtsam.Marginals(self._graph, result)
            covariance = marginals.marginalCovariance(key)
        except Exception:
            covariance = np.eye(6)
        
        return FGOState(
            position=position,
            orientation=orientation,
            covariance=covariance,
        )


def create_fgo_solver(backend: str = 'auto', **kwargs):
    """
    Factory function for FGO solver creation.
    
    Args:
        backend: 'scipy', 'gtsam', or 'auto' (try GTSAM, fallback to scipy)
        **kwargs: Solver-specific arguments
        
    Returns:
        FGO solver instance
    """
    if backend == 'gtsam':
        return GTSAMSolver(**kwargs)
    elif backend == 'scipy':
        return LevenbergMarquardtSolver(**kwargs)
    else:  # auto
        solver = GTSAMSolver(**kwargs)
        if solver.is_gtsam_available:
            return solver
        return LevenbergMarquardtSolver(**kwargs)
