import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


class DSMGeometryContractsTest(unittest.TestCase):
    def test_array_dsm_map_converts_tile_pixel_to_enu(self):
        from modules.vision.dsm_map import ArrayDSMMap

        dsm = np.arange(100, dtype=np.float32).reshape(10, 10)
        dsm_map = ArrayDSMMap(
            dsm=dsm,
            gsd_m=2.0,
            origin_lat=10.0,
            origin_lon=106.0,
            origin_col=0.0,
            origin_row=0.0,
        )

        point = dsm_map.pixel_to_enu(1.5, 2.0, tile_meta={"col_start": 3, "row_start": 4})

        self.assertIsNotNone(point)
        self.assertAlmostEqual(point.east_m, 9.0)
        self.assertAlmostEqual(point.north_m, -12.0)
        self.assertAlmostEqual(point.up_m, float(dsm[6, 4]))

    def test_dsm_pnp_solver_fails_closed_without_dsm(self):
        from core.types import CameraIntrinsics, MatchResult, Pose
        from modules.vision.geometry_solver import create_geometry_solver

        solver = create_geometry_solver("dsm_pnp", strict_dsm=True)
        match = MatchResult(
            query_keypoints=np.zeros((8, 2), dtype=np.float64),
            ref_keypoints=np.zeros((8, 2), dtype=np.float64),
            confidence=np.ones(8, dtype=np.float64),
        )
        pose = Pose(0.0, np.array([10.0, 106.0, 0.0]), np.array([1.0, 0.0, 0.0, 0.0]))
        camera = CameraIntrinsics(800.0, 800.0, 320.0, 240.0, 640, 480)

        self.assertIsNone(solver.solve(match, pose, camera))
        self.assertEqual(solver.last_diagnostics.get("status"), "dsm_unavailable")

    def test_dsm_pnp_solver_recovers_synthetic_camera_center(self):
        import cv2
        from core.types import CameraIntrinsics, MatchResult, Pose
        from modules.vision.dsm_map import ArrayDSMMap
        from modules.vision.geometry_solver import create_geometry_solver

        dsm = np.zeros((80, 80), dtype=np.float32)
        for row in range(dsm.shape[0]):
            for col in range(dsm.shape[1]):
                dsm[row, col] = 0.02 * col + 0.03 * row
        dsm_map = ArrayDSMMap(
            dsm=dsm,
            gsd_m=1.0,
            origin_lat=10.0,
            origin_lon=106.0,
            origin_col=40.0,
            origin_row=40.0,
        )

        ref_pts = np.array(
            [[20, 20], [30, 18], [45, 16], [55, 24], [22, 35], [35, 44], [50, 42], [60, 55], [28, 58], [44, 60]],
            dtype=np.float64,
        )
        obj = np.array(
            [[dsm_map.pixel_to_enu(u, v).as_array() for u, v in ref_pts]], dtype=np.float64
        ).reshape(-1, 3)

        camera = CameraIntrinsics(700.0, 700.0, 320.0, 240.0, 640, 480)
        rvec = np.array([[0.03], [-0.02], [0.01]], dtype=np.float64)
        tvec = np.array([[0.4], [-0.2], [35.0]], dtype=np.float64)
        img_pts, _ = cv2.projectPoints(obj, rvec, tvec, camera.K, None)
        img_pts = img_pts.reshape(-1, 2)

        match = MatchResult(
            query_keypoints=img_pts,
            ref_keypoints=ref_pts,
            confidence=np.ones(len(ref_pts), dtype=np.float64),
        )
        solver = create_geometry_solver("dsm_pnp", dsm_map=dsm_map, strict_dsm=True, min_inliers=6)
        ref_pose = Pose(0.0, np.array([10.0, 106.0, 0.0]), np.array([1.0, 0.0, 0.0, 0.0]))

        solved = solver.solve(match, ref_pose, camera)

        self.assertIsNotNone(solved)
        self.assertGreaterEqual(solver.last_diagnostics.get("pnp_inliers", 0), 6)
        self.assertLess(solver.last_diagnostics.get("pnp_reprojection_error_px", 999.0), 1.0)


if __name__ == "__main__":
    unittest.main()
