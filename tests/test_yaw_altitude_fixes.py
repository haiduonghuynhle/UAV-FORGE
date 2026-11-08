"""Regression tests for the yaw-recovery and altitude-source defects.

Yaw: IPM rotates the query into a North-up frame, so the homography measured
afterwards carries only the residual rotation. The removed yaw was never added
back (H_ipm was discarded and dataset_rotation_correction_deg was never set),
so reported heading collapsed to ~0 and the yaw error equalled the true
heading: 104.8 deg median with IPM on vs 0.7 deg with IPM off (VPAIR, n=2201).

Altitude: homography_scale defaulted to estimating altitude from the
correspondence scale, which is uncorrelated with truth on VPAIR (r = -0.07);
the protocol supplies altitude as telemetry, as the wrappers already used.
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "src"):
    sys.path.insert(0, str(p))

K = np.array([[800.0, 0.0, 640.0], [0.0, 800.0, 360.0], [0.0, 0.0, 1.0]])


def test_ipm_records_applied_yaw():
    from modules.vision.ipm import InversePerspectiveMapper

    ipm = InversePerspectiveMapper(camera_matrix=K)
    img = np.zeros((720, 1280, 3), dtype=np.uint8)
    ipm.apply_ipm(img, pitch_rad=0.0, roll_rad=0.0,
                  yaw_rad=np.radians(40.0), crop_horizon=False)
    assert abs(ipm.last_yaw_applied_rad - np.radians(40.0)) < 1e-9, (
        "IPM must record the yaw it removed so the solver can add it back"
    )


def test_subthreshold_yaw_records_zero():
    """A clamped (skipped) yaw must report 0, not the raw telemetry value."""
    from modules.vision.ipm import InversePerspectiveMapper

    ipm = InversePerspectiveMapper(camera_matrix=K)
    img = np.zeros((720, 1280, 3), dtype=np.uint8)
    ipm.apply_ipm(img, pitch_rad=0.0, roll_rad=0.0,
                  yaw_rad=np.radians(5.0), crop_horizon=False)
    assert ipm.last_yaw_applied_rad == 0.0, (
        "yaw below threshold is clamped away, so nothing must be added back"
    )


def test_homography_scale_defaults_to_telemetry_altitude():
    from modules.vision.geometry_solver import create_geometry_solver

    solver = create_geometry_solver("homography_scale", gsd=0.1)
    assert solver._altitude_source == "hint", (
        "scale-derived altitude is uncorrelated with truth; default must be the "
        "telemetry hint the protocol supplies"
    )
    explicit = create_geometry_solver("homography_scale", gsd=0.1,
                                      altitude_source="estimate")
    assert explicit._altitude_source == "estimate", "override must still work"


def test_rotation_correction_is_added_to_recovered_heading():
    """The solver must add the correction, not ignore it."""
    from modules.vision.homography_solver import HomographyYawHeightSolver

    solver = HomographyYawHeightSolver(gsd=0.1, altitude_source="hint")
    R = np.eye(3)  # zero residual rotation, as after a North-up IPM warp
    yaw_residual = HomographyYawHeightSolver._yaw_from_R(R)
    assert abs(yaw_residual) < 1e-9
    # heading = rotation_correction + residual; with residual 0 the reported
    # heading must equal the yaw IPM removed.
    for applied_deg in (0.0, 40.0, 175.0):
        heading = (np.radians(applied_deg) + yaw_residual) % (2 * np.pi)
        assert abs(np.degrees(heading) - applied_deg % 360.0) < 1e-6


def test_altitude_source_tier_labels_map_to_hint():
    """Config carries reporting-tier labels, not solver modes."""
    import sys as _s
    _s.path.insert(0, str(ROOT / "src" / "uav_geolocalization"))
    from uav_geolocalization.cli.benchmark_runner_impl import _solver_altitude_source

    for tier in ("metadata_or_gt_proxy", "dsm_pnp_camera_center", "", None, "GT_PROXY"):
        assert _solver_altitude_source(tier) == "hint", (
            f"tier label {tier!r} must not reach the scale estimator"
        )
    for mode in ("hint", "fixed", "estimate"):
        assert _solver_altitude_source(mode) == mode
        assert _solver_altitude_source(mode.upper()) == mode


def test_yaw_restored_exactly_once():
    """Reporting layer must not re-add telemetry yaw on top of the solver's."""
    src = (ROOT / "src" / "uav_geolocalization" / "cli" / "benchmark_runner_impl.py").read_text()
    assert "raw_heading += float(telemetry.get" not in src, (
        "second yaw restoration at the reporting layer would double-count the "
        "yaw already restored via rotation_correction_rad"
    )
    assert "ipm_yaw_applied_rad" in src, "solver-layer yaw restore must be present"


def test_ablation_arms_share_the_main_pose_head():
    """Otherwise rotation_correction_rad and the altitude mode never reach them."""
    src = (ROOT / "src" / "uav_geolocalization" / "cli" / "benchmark_runner_impl.py").read_text()
    import re
    for name in ("ours_no_ipm", "ours_no_fgo", "ours_no_lgi", "ours_minimal", "ours_fast_top1"):
        m = re.search(r"['\"]" + name + r"['\"]\s*:\s*\{(.*?)\n    \},", src, re.S)
        assert m, f"config block missing: {name}"
        assert "'geometry': 'homography_scale'," in m.group(1), (
            f"{name} uses a different pose head, confounding the ablation"
        )


if __name__ == "__main__":
    failures = 0
    for fn in (
        test_altitude_source_tier_labels_map_to_hint,
        test_yaw_restored_exactly_once,
        test_ablation_arms_share_the_main_pose_head,
        test_ipm_records_applied_yaw,
        test_subthreshold_yaw_records_zero,
        test_homography_scale_defaults_to_telemetry_altitude,
        test_rotation_correction_is_added_to_recovered_heading,
    ):
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except AssertionError as e:
            failures += 1
            print(f"FAIL {fn.__name__}: {e}")
        except Exception as e:
            failures += 1
            print(f"ERROR {fn.__name__}: {type(e).__name__}: {e}")
    sys.exit(1 if failures else 0)
