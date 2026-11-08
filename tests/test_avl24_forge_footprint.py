"""FORGE on AnyVisLoc 2026 uses the camera's real ground footprint.

Oblique frames look far ahead of the aircraft, so the footprint is a box that
is not centred on the ground-truth position. A tile can overlap that box while
sitting farther from the camera than any symmetric half-extent would allow.
"""
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data.dataset_adapters import AVL24Dataset  # noqa: E402
from uav_geolocalization.cli import benchmark_runner_impl as impl  # noqa: E402

M = impl.AVL24_M_PER_DEG
K = np.array([[500.0, 0, 500.0], [0, 500.0, 375.0], [0, 0, 1.0]])


def _rot_x(deg):
    a = math.radians(deg)
    return np.array([[1, 0, 0], [0, math.cos(a), -math.sin(a)], [0, math.sin(a), math.cos(a)]])


def _ds(ground=0.0):
    d = AVL24Dataset.__new__(AVL24Dataset)
    d._ground_height = lambda x, y: ground
    return d


def _sample(pitch_deg, z=100.0):
    P = np.eye(4); P[:3, :3] = _rot_x(pitch_deg); P[:3, 3] = [0.0, 0.0, z]
    return {"pose_c2w": P, "xyz": np.array([0.0, 0.0, z]), "K": K, "width": 1000, "height": 750}


def test_nadir_footprint_is_centred():
    xmin, xmax, ymin, ymax = _ds()._forge_footprint_box(_sample(0.0))
    assert abs(xmin + xmax) < 1e-6 and abs(ymin + ymax) < 1e-6
    assert abs(xmax - 100.0) < 1e-6          # 100 m AGL * (500 px / 500 px focal)


def test_oblique_footprint_is_shifted_and_long():
    # Pitch 40 deg, vertical half-FOV 36.9 deg: the near edge looks 3.1 deg
    # ahead of nadir and the far edge 76.9 deg, so the whole footprint lies
    # ahead of the camera, from about 5 m to about 430 m.
    xmin, xmax, ymin, ymax = _ds()._forge_footprint_box(_sample(40.0))
    near, far = sorted((abs(ymin), abs(ymax)))
    assert math.copysign(1, ymin) == math.copysign(1, ymax)   # one side only
    assert 3.0 < near < 8.0 and 400.0 < far < 460.0
    assert abs((ymin + ymax) / 2.0) > 150.0                   # box centre far from the camera


def test_ray_above_horizon_makes_footprint_unbounded():
    assert _ds()._forge_footprint_box(_sample(80.0)) is None


def _cand(ref_lat, box, ref_half=128.0):
    return {"anchor": {"lat": ref_lat, "lon": 0.0}, "query_box_latlon": box, "ref_half_m": ref_half}


def test_box_test_sees_overlap_a_symmetric_test_would_miss():
    # Footprint stretches 20..500 m ahead (north) of the camera at lat 0.
    box = [20.0 / M, 500.0 / M, -60.0 / M, 60.0 / M]
    far_tile = _cand(560.0 / M, box)          # 560 - 128 = 432 m: still under the footprint
    assert not impl._forge_candidate_excludes_gt(far_tile, "avl24", 560.0, 150.0)
    clear_tile = _cand(-200.0 / M, box)       # behind the camera, footprint starts at +20 m
    assert impl._forge_candidate_excludes_gt(clear_tile, "avl24", 200.0, 150.0)


def test_unbounded_footprint_never_counts_as_a_fault():
    assert not impl._forge_candidate_excludes_gt(_cand(5000.0 / M, None), "avl24", 5000.0, 150.0)
