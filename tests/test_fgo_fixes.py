"""Regression tests for the FGO fixes (branch copy code_fgo_fix).

Covers:
  A. mount-aware visual factor produces positive-depth points and near-zero
     residuals at the true state (and documents that the legacy identity
     convention silently kills the factor);
  C. barometric + vision-altitude factors genuinely fuse (z between the two);
  D. clear_factors empties every factor list.
Fix B (step-relative guard) is validated by the 30-frame smoke run, since the
guard lives inside the frame-processing method, not a unit-testable function.
"""
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from modules.fusion.fgo_solver import (  # noqa: E402
    BarometricFactor,
    FGOState,
    LevenbergMarquardtSolver,
    VisualFactor,
)

R_MOUNT = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])
K = np.array([[800.0, 0.0, 400.0], [0.0, 800.0, 300.0], [0.0, 0.0, 1.0]])
ALT = 100.0


def _nadir_scene():
    """Synthetic nadir scene: camera at (0,0,ALT) looking straight down."""
    cam_c = np.array([0.0, 0.0, ALT])
    pix = np.array([[400.0, 300.0], [200.0, 150.0], [600.0, 450.0],
                    [100.0, 500.0], [700.0, 100.0], [400.0, 500.0]])
    pts3d = []
    for u, v in pix:
        d_cam = np.linalg.inv(K) @ np.array([u, v, 1.0])
        d_world = R_MOUNT @ d_cam
        t = (0.0 - cam_c[2]) / d_world[2]
        pts3d.append(cam_c + t * d_world)
    return cam_c, pix, np.asarray(pts3d)


def _state_vec(pos, euler=(0.0, 0.0, 0.0)):
    return np.concatenate([np.asarray(pos, float), np.asarray(euler, float), np.zeros(9)])


def test_fix_a_mounted_factor_active_and_consistent():
    cam_c, pix, pts3d = _nadir_scene()
    assert np.allclose(pts3d[:, 2], 0.0)

    solver = LevenbergMarquardtSolver()
    solver.set_initial_state(FGOState(position=cam_c, orientation=np.zeros(3)))
    solver.add_visual_factor(VisualFactor(points_2d=pix, points_3d=pts3d,
                                          camera_matrix=K, R_mount=R_MOUNT))
    res = solver._compute_residuals(_state_vec(cam_c))
    # 6 points x 2 residual components each => the factor is ACTIVE
    assert res.size >= 12, f'visual factor inactive, residual size {res.size}'
    assert np.max(np.abs(res[:12])) < 1e-6, f'nonzero residual at truth: {res[:12]}'


def test_fix_a_documents_legacy_defect():
    cam_c, pix, pts3d = _nadir_scene()
    solver = LevenbergMarquardtSolver()
    solver.set_initial_state(FGOState(position=cam_c, orientation=np.zeros(3)))
    solver.add_visual_factor(VisualFactor(points_2d=pix, points_3d=pts3d,
                                          camera_matrix=K, R_mount=None))
    res = solver._compute_residuals(_state_vec(cam_c))
    # Legacy identity convention: all points behind the camera, factor mute.
    assert res.size < 12, 'legacy convention unexpectedly produced residuals'


def test_fix_a_recovers_position_offset():
    cam_c, pix, pts3d = _nadir_scene()
    solver = LevenbergMarquardtSolver()
    solver.set_initial_state(FGOState(position=cam_c + np.array([5.0, -4.0, 3.0]),
                                      orientation=np.zeros(3)))
    solver.add_visual_factor(VisualFactor(points_2d=pix, points_3d=pts3d,
                                          camera_matrix=K, R_mount=R_MOUNT))
    opt = solver.optimize()
    err = np.linalg.norm(np.asarray(opt.position) - cam_c)
    assert err < 0.5, f'visual-only optimization did not converge to truth: {err:.3f} m'


def test_fix_c_baro_and_vision_altitude_fuse():
    solver = LevenbergMarquardtSolver()
    solver.set_initial_state(FGOState(position=np.array([0.0, 0.0, 120.0]),
                                      orientation=np.zeros(3)))
    solver.add_barometric_factor(BarometricFactor(altitude=100.0, sigma=3.0))
    solver.add_barometric_factor(BarometricFactor(altitude=120.0, sigma=20.0))
    opt = solver.optimize()
    z = float(np.asarray(opt.position)[2])
    assert 100.0 < z < 120.0, f'z={z} not between the two altitude factors'
    assert abs(z - 100.0) < abs(z - 120.0), f'z={z} not weighted toward the tighter factor'


def test_fix_d_clear_factors_empties_everything():
    cam_c, pix, pts3d = _nadir_scene()
    solver = LevenbergMarquardtSolver()
    solver.add_visual_factor(VisualFactor(points_2d=pix, points_3d=pts3d,
                                          camera_matrix=K, R_mount=R_MOUNT))
    solver.add_barometric_factor(BarometricFactor(altitude=100.0, sigma=3.0))
    solver.clear_factors()
    assert not solver._visual_factors and not solver._baro_factors


if __name__ == '__main__':
    fails = 0
    for name, fn in sorted({k: v for k, v in globals().items() if k.startswith('test_')}.items()):
        try:
            fn()
            print(f'PASS {name}')
        except AssertionError as e:
            fails += 1
            print(f'FAIL {name}: {e}')
    sys.exit(1 if fails else 0)
