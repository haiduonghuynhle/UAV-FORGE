"""East and north metres per pixel differ on a square-degree raster.

The UAV-VisLoc satellite maps are EPSG:4326 with square-DEGREE pixels, so one
pixel spans the same angle on both axes but not the same distance: north stays
near 0.2986 m/px while east is that value times cos(latitude), which is 0.229
to 0.273 m/px across the dataset's 24.7 to 40.3 degrees. Multiplying the x
offset by the north value inflated every east component by 1.10x to 1.31x.

The latitude that makes the correction exact belongs to the retrieved
reference tile, not to the query, so the adapter publishes only the axis
convention and the runner supplies the number. Two guards below pin that split,
because deriving the latitude in the adapter is correct only when the query and
the reference come from the same location, which cross-flight retrieval breaks.
"""
from __future__ import annotations

import inspect

import numpy as np
import pytest

from core.types import CameraIntrinsics, MatchResult, Pose
from modules.vision.geometry_solver import HomographyGeometrySolver, PnPGeometrySolver


def make_match(dx_px: float, dy_px: float, size: int = 512) -> MatchResult:
    """A pure translation homography that moves the query centre by (dx, dy)."""
    H = np.array([[1.0, 0.0, dx_px], [0.0, 1.0, dy_px], [0.0, 0.0, 1.0]])
    n = 20
    q = np.random.default_rng(0).uniform(0, size, size=(n, 2))
    return MatchResult(
        query_keypoints=q,
        ref_keypoints=q + np.array([dx_px, dy_px]),
        confidence=np.ones(n),
        homography=H,
        inlier_mask=np.ones(n, dtype=bool),
    )


def solve_offset(gsd: float, gsd_e: float | None) -> tuple[float, float]:
    """Return (east_m, north_m) the solver recovers for a 100 px diagonal shift."""
    solver = HomographyGeometrySolver(gsd=gsd, min_inliers=4)
    size = 512
    ref = Pose(timestamp=0.0, position=np.array([30.0, 120.0, 100.0]),
               orientation=np.array([1.0, 0.0, 0.0, 0.0]))
    cam = CameraIntrinsics(fx=1000, fy=1000, cx=size / 2, cy=size / 2, width=size, height=size)
    pose = solver.solve(
        match_result=make_match(100.0, 100.0, size),
        reference_pose=ref, camera=cam,
        query_size=(size, size), ref_size=(size, size),
        altitude_hint=100.0, gsd=gsd, gsd_e=gsd_e,
    )
    assert pose is not None
    m_per_deg_lat = 111_320.0
    m_per_deg_lon = m_per_deg_lat * np.cos(np.radians(ref.position[0]))
    north_m = (pose.position[0] - ref.position[0]) * m_per_deg_lat
    east_m = (pose.position[1] - ref.position[1]) * m_per_deg_lon
    return float(east_m), float(north_m)


def test_east_uses_gsd_e_north_uses_gsd() -> None:
    gsd, lat = 0.3, np.radians(30.0)
    gsd_e = gsd * float(np.cos(lat))
    east_m, north_m = solve_offset(gsd, gsd_e)
    assert north_m == pytest.approx(-100.0 * gsd, abs=1e-3)
    assert east_m == pytest.approx(100.0 * gsd_e, abs=1e-3)
    # the whole point: the axes must not agree
    assert abs(east_m) < abs(north_m)


def test_isotropic_path_is_unchanged_when_no_gsd_e_is_given() -> None:
    """AVL and VPAIR pass nothing and must keep today's numbers exactly."""
    gsd = 0.42
    east_m, north_m = solve_offset(gsd, None)
    assert east_m == pytest.approx(100.0 * gsd, abs=1e-3)
    assert north_m == pytest.approx(-100.0 * gsd, abs=1e-3)


def test_gsd_e_is_appended_not_inserted() -> None:
    """ref_size keeps its positional slot, so no caller silently shifts meaning."""
    for cls in (HomographyGeometrySolver, PnPGeometrySolver):
        names = list(inspect.signature(cls.solve).parameters)
        assert names.index("ref_size") < names.index("gsd_e"), cls.__name__


def test_adapter_publishes_the_axis_convention_not_a_latitude() -> None:
    """No latitude may leave the adapter.

    Deriving it there from the query's own map is exact only when the retrieved
    reference is from the same location. Under cross-flight retrieval a
    24.7 N query against a 40.3 N reference would take a 19 percent east error,
    which is larger than the defect being fixed.
    """
    from data.dataset_adapters import UAVVisLocDataset

    src = inspect.getsource(UAVVisLocDataset._get_frame)
    assert "'gsd_axis': 'square_degree'" in src
    assert "gsd_e" not in src, "the adapter must not compute the east scale itself"


def test_runner_takes_the_latitude_from_the_reference_tile() -> None:
    from pathlib import Path

    src = (Path(__file__).resolve().parents[1] / "src" / "uav_geolocalization"
           / "cli" / "benchmark_runner_impl.py").read_text(encoding="utf-8")
    assert "ref_pose.position[0]" in src.split("gsd_e=(")[1][:400], \
        "gsd_e must be built from the reference anchor latitude"
    assert "frame.get('gsd_axis') == 'square_degree'" in src
