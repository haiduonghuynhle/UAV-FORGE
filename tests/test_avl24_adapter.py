"""Contracts for the AnyVisLoc 2026 (NPZ) adapter.

The release ships scene-local metres with no geodetic anchor, so the adapter
synthesizes lat/lon. These tests pin the two properties the rest of the stack
depends on: the synthetic frame is invertible under the runner's own distance
metric, and corpus anchors carry the tile offsets the DSM pose head consumes.
"""

import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


MAP_RES = 0.5          # m/px
MAP_W, MAP_H = 400, 300
DSM_RES = 2.0          # m/px, deliberately coarser than the map
SCENE = "Scene_07"
SCENE_ID = 7


def _build_scene(root: Path) -> Path:
    """Write a minimal but structurally faithful AVL24 scene."""
    import cv2

    scene_dir = root / SCENE
    scene_dir.mkdir(parents=True)

    ref = np.random.default_rng(0).integers(0, 255, (MAP_H, MAP_W, 3), dtype=np.uint8)
    cv2.imwrite(str(scene_dir / "aerial_map.png"), ref)

    dsm_h = int(MAP_H * MAP_RES / DSM_RES)
    dsm_w = int(MAP_W * MAP_RES / DSM_RES)
    dsm = np.arange(dsm_h * dsm_w, dtype=np.float32).reshape(dsm_h, dsm_w)
    np.save(scene_dir / "aerial_dsm.npy", dsm)

    with open(scene_dir / f"L{SCENE_ID:02d}_reference.json", "w", encoding="utf-8") as f:
        json.dump({
            "modes": {
                "aerial": {
                    "map_path": "aerial_map.png",
                    "dsm_path": "aerial_dsm.npy",
                    "map_resolution": [MAP_RES, MAP_RES],
                    "dsm_resolution": [DSM_RES, DSM_RES],
                    "map_origin_local": [0.0, 0.0],
                    "dsm_origin_local": [0.0, 0.0],
                }
            }
        }, f)

    K = np.array([[800.0, 0.0, 320.0], [0.0, 800.0, 240.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    for i, (x, y, z, pitch) in enumerate(
        [(50.0, 40.0, 120.0, 0.0), (90.0, 70.0, 150.0, 35.0)], start=1
    ):
        np.savez(
            scene_dir / f"L{SCENE_ID:02d}_{i:04d}.npz",
            image=np.zeros((480, 640, 3), dtype=np.uint8),
            image_size=np.array([480, 640], dtype=np.int32),
            K=K,
            dist=np.zeros(5, dtype=np.float32),
            pose_c2w=np.eye(4, dtype=np.float32),
            pose_w2c=np.eye(4, dtype=np.float32),
            xyz=np.array([x, y, z], dtype=np.float32),
            euler_deg=np.array([0.0, pitch, 15.0], dtype=np.float32),
            sample_id=f"L{SCENE_ID:02d}_{i:04d}",
            scene_id=SCENE_ID,
        )
    return scene_dir


class AVL24AdapterTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="avl24_"))
        _build_scene(cls.tmp)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def _dataset(self, **kwargs):
        from data.dataset_adapters import AVL24Dataset

        ds = AVL24Dataset(self.tmp, scene=SCENE, tile_size_px=64, **kwargs)
        ds.load()
        return ds

    def test_load_maps_npz_fields_to_frame_contract(self):
        ds = self._dataset()
        self.assertEqual(len(ds._entries), 2)

        frame = ds[0]
        for key in ("image_name", "query_image", "gt_pose", "camera", "gsd", "reference_image"):
            self.assertIn(key, frame)

        self.assertEqual(frame["image_name"], f"L{SCENE_ID:02d}_0001")
        self.assertEqual(frame["query_image"].shape, (480, 640, 3))
        self.assertEqual(frame["query_image"].dtype, np.uint8)
        self.assertAlmostEqual(frame["camera"]["fx"], 800.0)
        self.assertEqual(frame["camera"]["width"], 640)
        self.assertAlmostEqual(frame["gsd"], MAP_RES)

        gt = frame["gt_pose"]
        # Upstream geometry (ground_dist = z/|sin(90+pitch)|, shift = z*tan(pitch))
        # is only self-consistent when pitch == 0 at nadir.
        self.assertEqual(gt["pitch_convention"], "aircraft_nadir_zero")
        self.assertAlmostEqual(gt["alt"], 120.0)
        self.assertAlmostEqual(gt["pitch"], 0.0)
        self.assertAlmostEqual(gt["yaw"], 15.0)

    def test_synthetic_frame_is_invertible_under_runner_metric(self):
        """Local metres must survive the lat/lon round trip the runner measures in."""
        from uav_geolocalization.cli.benchmark_runner_impl import _approx_latlon_distance_m

        ds = self._dataset()
        lat_a, lon_a = ds._local_to_latlon(50.0, 40.0)
        lat_b, lon_b = ds._local_to_latlon(90.0, 70.0)

        expected = float(np.hypot(90.0 - 50.0, 70.0 - 40.0))
        measured = _approx_latlon_distance_m(lat_a, lon_a, lat_b, lon_b)
        self.assertAlmostEqual(measured, expected, places=3)

        x, y = ds._latlon_to_local(lat_a, lon_a)
        self.assertAlmostEqual(x, 50.0, places=6)
        self.assertAlmostEqual(y, 40.0, places=6)

    def test_pixel_latlon_round_trip(self):
        ds = self._dataset()
        for col, row in ((0, 0), (137, 91), (MAP_W - 1, MAP_H - 1)):
            lat, lon = ds._pixel_to_latlon(col, row)
            self.assertEqual(ds._latlon_to_pixel(lat, lon), (col, row))

    def test_scenes_do_not_collide_in_the_synthetic_frame(self):
        from uav_geolocalization.cli.benchmark_runner_impl import _approx_latlon_distance_m
        from data.dataset_adapters import AVL24Dataset

        ds = self._dataset()
        other = AVL24Dataset(self.tmp, scene=SCENE, tile_size_px=64)
        other._scene_id = SCENE_ID + 1  # simulate a sibling scene's origin
        other._ref_meta = ds._load_ref_meta()

        lat_a, lon_a = ds._local_to_latlon(0.0, 0.0)
        lat_b, lon_b = other._local_to_latlon(0.0, 0.0)
        self.assertGreater(_approx_latlon_distance_m(lat_a, lon_a, lat_b, lon_b), 100_000.0)

    def test_corpus_anchors_carry_tile_offsets(self):
        ds = self._dataset()
        entries = list(ds.iter_reference_corpus())
        self.assertGreater(len(entries), 1)

        for entry in entries:
            anchor = entry["anchor"]
            self.assertEqual(entry["ref_image"].shape, (64, 64, 3))
            # The DSM pose head reads these to place a tile pixel on the map grid.
            for key in ("row_start", "col_start", "tile_size_px", "lat", "lon"):
                self.assertIn(key, anchor)

        first = entries[0]["anchor"]
        self.assertEqual((first["row_start"], first["col_start"]), (0, 0))
        lat, lon = ds._pixel_to_latlon(32, 32)
        self.assertAlmostEqual(first["lat"], lat, places=9)
        self.assertAlmostEqual(first["lon"], lon, places=9)

    def test_metric_tile_size_tracks_scene_resolution(self):
        ds = self._dataset()
        ds._forced_tile_size_px = None

        ds._tile_size_m = 400.0
        self.assertEqual(ds.tile_size_px, int(400.0 / MAP_RES))

        # Clamped so a very fine scene map cannot request a degenerate tile.
        ds._tile_size_m = 10.0
        self.assertEqual(ds.tile_size_px, 256)

    def test_dsm_bridge_resolves_tile_pixels_on_the_coarser_grid(self):
        ds = self._dataset()
        dsm_map = ds.build_dsm_map()
        self.assertIsNotNone(dsm_map)
        self.assertTrue(dsm_map.available)

        # Map pixel (col=20, row=12) -> local (10 m, 6 m) -> DSM cell (5, 3).
        point = dsm_map.pixel_to_enu(4.0, 2.0, tile_meta={"col_start": 16, "row_start": 10})
        self.assertIsNotNone(point)
        self.assertAlmostEqual(point.east_m, 10.0)
        self.assertAlmostEqual(point.north_m, -6.0)

        dsm = np.load(self.tmp / SCENE / "aerial_dsm.npy")
        self.assertAlmostEqual(point.up_m, float(dsm[3, 5]))

        # ENU is absolute in the scene frame, so the tile anchor passed by the
        # solver must not shift the result.
        lat, lon = dsm_map.enu_to_latlon(point.east_m, point.north_m, 12.34, 56.78)
        self.assertEqual((lat, lon), ds._local_to_latlon(10.0, 6.0))

    def test_out_of_bounds_dsm_lookup_fails_closed(self):
        ds = self._dataset()
        dsm_map = ds.build_dsm_map()
        self.assertIsNone(
            dsm_map.pixel_to_enu(0.0, 0.0, tile_meta={"col_start": 10_000, "row_start": 10_000})
        )

    def test_reference_tile_is_centred_on_the_query(self):
        ds = self._dataset()
        frame = ds[0]
        tile = frame["reference_image"]
        self.assertEqual(tile.shape, (64, 64, 3))

        import cv2
        ref = cv2.cvtColor(cv2.imread(str(self.tmp / SCENE / "aerial_map.png")), cv2.COLOR_BGR2RGB)
        col, row = ds._latlon_to_pixel(frame["gt_pose"]["lat"], frame["gt_pose"]["lon"])
        np.testing.assert_array_equal(tile, ref[row - 32:row + 32, col - 32:col + 32])

    def test_missing_reference_json_is_loud(self):
        from data.dataset_adapters import AVL24Dataset

        empty = Path(tempfile.mkdtemp(prefix="avl24_empty_"))
        try:
            (empty / "Scene_01").mkdir()
            with self.assertRaises(FileNotFoundError):
                AVL24Dataset(empty, scene="Scene_01").load()
        finally:
            shutil.rmtree(empty, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
