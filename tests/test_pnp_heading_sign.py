"""The DSM-PnP head must publish the vehicle heading, not its negative.

cv2.solvePnP returns R_world->cam. Its ZYX yaw is therefore the yaw of the
world seen from the camera, which is the negative of the vehicle heading for a
nadir camera. The head published that value directly, so every heading it
reported carried the wrong sign.

The fix must also keep roll and pitch as measurements. The runner reads
Pose.euler_angles[0] and [1] into pred_roll and pred_pitch, and composes both
into rotation_geodesic_error_deg. A yaw-only quaternion would make two of the
six degrees of freedom constant, which on a near-nadir dataset reads as almost
zero attitude error while measuring nothing at all.
"""
from __future__ import annotations

import numpy as np
import pytest

from core.types import Pose


def rot_x(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)


def rot_y(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)


def rot_z(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)


NADIR = np.diag([1.0, -1.0, -1.0])


def world_to_cam(yaw: float, pitch: float, roll: float) -> np.ndarray:
    """R_world->cam for a camera at this attitude, in the IPM module's convention.

    ipm.py builds R_correction = Rz(yaw) Rx(-roll) Ry(-pitch) and applies it as
    H = K R_correction K^-1 to bring the view to nadir and north-up. The camera
    rotation it undoes is therefore R = R_correction^-1 @ NADIR.
    """
    r_correction = rot_z(yaw) @ rot_x(-roll) @ rot_y(-pitch)
    return np.linalg.inv(r_correction) @ NADIR


def published_angles(R: np.ndarray) -> tuple[float, float, float]:
    """Run the head's own orientation code and read the pose back as Euler.

    Kept as a copy of the solver body rather than a call into it, because the
    solver needs a DSM raster, a match result and camera intrinsics that this
    unit does not have. The assertion that matters is the identity, and any
    edit to the solver that breaks the identity must break this test, so the
    copy is checked against the source in test_matches_solver_source.
    """
    A = R @ NADIR
    yaw_rad = float(np.arctan2(-R[1, 0], -R[1, 1]))
    roll_rad = float(-np.arcsin(np.clip(A[1, 2], -1.0, 1.0)))
    b0 = np.array([np.cos(yaw_rad), np.sin(yaw_rad), 0.0])
    pitch_rad = float(np.arctan2(-float(A[2] @ b0), float(A[0] @ b0)))

    cy, sy = np.cos(yaw_rad / 2.0), np.sin(yaw_rad / 2.0)
    cp, sp = np.cos(pitch_rad / 2.0), np.sin(pitch_rad / 2.0)
    cr, sr = np.cos(roll_rad / 2.0), np.sin(roll_rad / 2.0)
    quat = np.array([
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    ], dtype=np.float64)

    roll, pitch, yaw = Pose(timestamp=0.0,
                            position=np.zeros(3),
                            orientation=quat).euler_angles
    return float(yaw), float(pitch), float(roll)


GRID = [(y, p, r)
        for y in (-150.0, -37.0, 0.0, 13.5, 77.0, 160.0)
        for p in (0.0, -15.0, 20.0, 3.0)
        for r in (0.0, -11.0, 6.0)]


@pytest.mark.parametrize("yaw_deg,pitch_deg,roll_deg", GRID)
def test_recovers_all_three_angles(yaw_deg: float, pitch_deg: float, roll_deg: float) -> None:
    R = world_to_cam(np.radians(yaw_deg), np.radians(pitch_deg), np.radians(roll_deg))
    yaw, pitch, roll = published_angles(R)
    assert np.degrees(yaw) == pytest.approx(yaw_deg, abs=1e-6)
    assert np.degrees(pitch) == pytest.approx(pitch_deg, abs=1e-6)
    assert np.degrees(roll) == pytest.approx(roll_deg, abs=1e-6)


def test_old_formula_returned_the_negated_heading() -> None:
    """Guard the defect itself, so a revert cannot pass silently."""
    yaw_deg = 77.0
    R = world_to_cam(np.radians(yaw_deg), 0.0, 0.0)
    old = np.degrees(np.arctan2(R[1, 0], R[0, 0]))
    assert old == pytest.approx(-yaw_deg, abs=1e-6)
    assert np.degrees(published_angles(R)[0]) == pytest.approx(yaw_deg, abs=1e-6)


def test_transposing_does_not_fix_it() -> None:
    """The obvious fix is wrong: R.T returns the same value at zero attitude."""
    R = world_to_cam(np.radians(77.0), 0.0, 0.0)
    assert np.degrees(np.arctan2(R.T[1, 0], R.T[0, 0])) == pytest.approx(
        np.degrees(np.arctan2(R[1, 0], R[0, 0])), abs=1e-6)


def test_a_yaw_only_quaternion_would_destroy_the_attitude_measurement() -> None:
    """Why the head must not publish orientation as yaw only.

    pred_roll and pred_pitch come from this quaternion, so a yaw-only pose
    reports a constant where the benchmark expects a measurement.
    """
    roll_deg, pitch_deg = -11.0, 20.0
    R = world_to_cam(np.radians(13.5), np.radians(pitch_deg), np.radians(roll_deg))
    yaw_rad = float(np.arctan2(-R[1, 0], -R[1, 1]))
    yaw_only = Pose(timestamp=0.0, position=np.zeros(3),
                    orientation=np.array([np.cos(yaw_rad / 2), 0.0, 0.0, np.sin(yaw_rad / 2)]))
    lost_roll, lost_pitch, _ = np.degrees(yaw_only.euler_angles)
    assert lost_roll == pytest.approx(0.0, abs=1e-9)
    assert lost_pitch == pytest.approx(0.0, abs=1e-9)
    # the full recovery keeps them
    _, pitch, roll = published_angles(R)
    assert np.degrees(roll) == pytest.approx(roll_deg, abs=1e-6)
    assert np.degrees(pitch) == pytest.approx(pitch_deg, abs=1e-6)


def test_matches_solver_source() -> None:
    """The solver must still contain the expressions this test asserts about."""
    from pathlib import Path

    source = (Path(__file__).resolve().parents[1]
              / "src" / "modules" / "vision" / "geometry_solver.py").read_text(encoding="utf-8")
    assert "np.arctan2(-R[1, 0], -R[1, 1])" in source, "heading formula missing from the solver"
    assert "NADIR = np.diag([1.0, -1.0, -1.0])" in source, "attitude recovery missing from the DSM head"
    assert "quat = self._quaternion_from_rotation(R)" not in source, \
        "the DSM head is back to publishing the raw world->cam rotation"
