"""Regression tests for the round-7 diagnostic fixes.

Covers:
  1. IPM per-axis near-nadir clamping: sub-threshold pitch/roll must be zeroed
     so the warp is yaw-only and the principal ray is not displaced
     (position error ~= h*tan(noise) otherwise).
  2. Rerank margin guard: a rank>0 candidate may displace the descriptor top-1
     only if it beats the top-1 verification score by an absolute margin.
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "src", ROOT / "src" / "uav_geolocalization"):
    sys.path.insert(0, str(p))


def _make_ipm():
    from modules.vision.ipm import InversePerspectiveMapper

    K = np.array([[800.0, 0.0, 640.0],
                  [0.0, 800.0, 360.0],
                  [0.0, 0.0, 1.0]])
    return InversePerspectiveMapper(camera_matrix=K), K


def test_ipm_subthreshold_pitch_roll_is_yaw_only():
    ipm, K = _make_ipm()
    img = np.zeros((720, 1280, 3), dtype=np.uint8)
    # pitch/roll below the 15 deg threshold (pure telemetry noise regime),
    # yaw well above it -> warp must be pure in-plane yaw rotation.
    out, H = ipm.apply_ipm(
        img,
        pitch_rad=np.radians(3.0),
        roll_rad=np.radians(2.0),
        yaw_rad=np.radians(40.0),
        crop_horizon=False,
    )
    c = np.array([K[0, 2], K[1, 2], 1.0])
    c_mapped = H @ c
    c_mapped = c_mapped[:2] / c_mapped[2]
    # A rotation about the optical axis keeps the principal point fixed.
    assert np.linalg.norm(c_mapped - c[:2]) < 1e-6, (
        f"principal point moved by {np.linalg.norm(c_mapped - c[:2]):.3f}px: "
        "sub-threshold pitch/roll leaked into the warp"
    )


def test_ipm_all_axes_subthreshold_skips():
    ipm, _ = _make_ipm()
    img = np.zeros((720, 1280, 3), dtype=np.uint8)
    out, H = ipm.apply_ipm(
        img,
        pitch_rad=np.radians(3.0),
        roll_rad=np.radians(2.0),
        yaw_rad=np.radians(5.0),
        crop_horizon=False,
    )
    assert np.allclose(H, np.eye(3)), "all-subthreshold attitude must skip IPM"


def test_ipm_oblique_still_warps():
    ipm, K = _make_ipm()
    img = np.zeros((720, 1280, 3), dtype=np.uint8)
    out, H = ipm.apply_ipm(
        img,
        pitch_rad=np.radians(30.0),
        roll_rad=0.0,
        yaw_rad=0.0,
        crop_horizon=False,
    )
    assert not np.allclose(H, np.eye(3)), "genuinely oblique frame must be warped"


def test_rerank_margin_guard():
    from uav_geolocalization.cli.benchmark_runner_impl import BenchmarkRunner

    guard = BenchmarkRunner._challenger_passes_margin
    # rank 0 always passes
    assert guard(0, 10.0, None, 50.0)
    # margin disabled -> legacy behavior, any challenger passes
    assert guard(2, 10.0, 100.0, 0.0)
    # challenger below top1+margin is rejected
    assert not guard(2, 120.0, 100.0, 50.0)
    # challenger clearing the margin passes
    assert guard(2, 151.0, 100.0, 50.0)
    # negative top-1 scores behave monotonically
    assert not guard(1, -10.0, -30.0, 50.0)
    assert guard(1, 25.0, -30.0, 50.0)


if __name__ == "__main__":
    failures = 0
    for fn in [
        test_ipm_subthreshold_pitch_roll_is_yaw_only,
        test_ipm_all_axes_subthreshold_skips,
        test_ipm_oblique_still_warps,
        test_rerank_margin_guard,
    ]:
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
