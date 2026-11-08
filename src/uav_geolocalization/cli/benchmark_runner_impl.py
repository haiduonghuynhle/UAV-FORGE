#!/usr/bin/env python3
"""
Unified Multi-Dataset Benchmark Runner.

Runs the full localization pipeline on multiple datasets and produces
paper-ready metrics, comparison tables, and CDF plots.

Usage:
    # Run on single dataset
    python benchmark_runner.py --dataset vpair --pipeline ours --output results/vpair/

    # Run on all 3 main datasets
    python benchmark_runner.py --dataset all --pipeline ours --output results/

    # Quick debug run (small subset)
    python benchmark_runner.py --dataset vpair --pipeline ours --max-samples 10 --output results/debug/

    # Ablation study (all proposed + E2E RoMa baseline; oracle baselines unless --ablation-skip-oracle)
    python benchmark_runner.py --ablation --dataset vpair --output results/ablation/
    python benchmark_runner.py --ablation --dataset all --strict-components --strict-end2end --output results/full_ablation/

    # Compare methods
    python benchmark_runner.py --compare results/vpair/ours/ results/vpair/baseline/ --output results/compare/

Supported pipelines:
    - "ours":            UltraVPR + IPM + LightGlue + FGO
    - "ours_no_ipm":     UltraVPR + LightGlue + Homography (ablation)
    - "ours_no_fgo":     UltraVPR + IPM + LightGlue + Homography (ablation)
    - "ours_cascade":    UltraVPR Top-K + LightGlue + selective RoMa + FGO
    - "roma_every_frame": Oracle reference + RoMa every frame
"""

import argparse
import csv
import sys
import json
import hashlib
import time
import logging
import os
import re
import platform
import shutil
import subprocess
import importlib
from pathlib import Path
from typing import Callable, Iterable, List, Dict, Any, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass
from types import SimpleNamespace
import numpy as np
try:
    import fcntl
except ImportError:  # pragma: no cover - non-POSIX fallback
    fcntl = None

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"

# Setup path for repo root and src/ so legacy imports remain compatible.
for _path in (str(REPO_ROOT), str(SRC_ROOT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)
_overlay_src = os.environ.get("HD4_PATCH_SRC")
if _overlay_src:
    sys.path.insert(0, _overlay_src)

try:
    from evaluation.metrics import NpEncoder
except ImportError:
    # Fallback encoder if metrics module unavailable during early init
    import json as _json
    class NpEncoder(_json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (int,)):
                return int(obj)
            if hasattr(obj, 'item'):   # numpy scalar
                return obj.item()
            if hasattr(obj, 'tolist'):  # numpy array
                return obj.tolist()
            return super().default(obj)

# (sys.path already configured above for NpEncoder import)

logger = logging.getLogger(__name__)

FGO_REJECT_NON_FINITE_POS = "non_finite_pos"
FGO_REJECT_STEP_XY = "step_xy"
FGO_REJECT_STEP_ALT = "step_alt"
FGO_REJECT_XY_EXPLODED = "xy_exploded"
FGO_REJECT_RUNTIME_EXCEPTION = "runtime_exception"
FGO_BACKEND_POLICY_PREFER_SCIPY = "prefer_scipy_with_fallback"
RETRIEVAL_DB_CACHE_SCHEMA_VERSION = 2
FGO_GUARD_REJECT_CODES = {
    FGO_REJECT_NON_FINITE_POS,
    FGO_REJECT_STEP_XY,
    FGO_REJECT_STEP_ALT,
    FGO_REJECT_XY_EXPLODED,
}

def _safe_int(value: Any, default: int, min_value: int | None = None) -> int:
    """Best-effort integer coercion with safe fallback."""
    try:
        out = int(value)
    except (TypeError, ValueError):
        out = int(default)
    if min_value is not None:
        out = max(int(min_value), out)
    return out

_SOLVER_ALTITUDE_MODES = ("hint", "fixed", "estimate")


def _pitch_from_nadir_deg(pitch_deg: float, convention: str) -> float:
    """Normalize a dataset-native pitch to degrees-from-nadir (aircraft frame).

    Mirrors the IPM convention handling: AVL gimbal reports -90 at nadir,
    photogrammetry Phi and aircraft pitch report ~0 at nadir. Feeding the raw
    gimbal value into the FGO ray orientation pointed the rays horizontally
    and killed or corrupted the visual factor on AVL/UAV-VisLoc.
    """
    conv = (convention or "").lower()
    if conv == "gimbal_neg90_nadir":
        return 90.0 - abs(pitch_deg)
    if conv == "aircraft_nadir_zero":
        return pitch_deg
    # Heuristics, identical to InversePerspectiveMapper.apply_ipm_from_metadata
    if pitch_deg <= -45:
        return 90.0 - abs(pitch_deg)
    if abs(pitch_deg) < 45:
        return pitch_deg
    if 45 <= pitch_deg < 90:
        return 90.0 - pitch_deg
    return pitch_deg


def _solver_altitude_source(config_value: Any) -> str:
    """Map the config's altitude field to a solver mode.

    Pipeline configs carry reporting-tier labels such as 'metadata_or_gt_proxy'
    or 'dsm_pnp_camera_center'. Passed through verbatim these match no solver
    mode and fall through to the scale-derived estimator, whose output is
    uncorrelated with truth (r = -0.07 on VPAIR) and produced E_alt around
    300 m on the main rows against 2.4 m on rows that used the telemetry hint.
    Anything that is not an explicit solver mode means "use the telemetry
    altitude the protocol supplies".
    """
    value = str(config_value or "").strip().lower()
    return value if value in _SOLVER_ALTITUDE_MODES else "hint"


def _safe_float(value: Any, default: float, min_value: float | None = None) -> float:
    """Best-effort float coercion with finite guard."""
    try:
        out = float(value)
    except (TypeError, ValueError):
        out = float(default)
    if not np.isfinite(out):
        out = float(default)
    if min_value is not None:
        out = max(float(min_value), out)
    return out

def _json_number(value: Any) -> Any:
    """Normalize non-finite values for deterministic JSON output."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) else None


def _finite_result_float(row: Dict[str, Any], key: str) -> float | None:
    try:
        value = float(row.get(key))
    except (TypeError, ValueError):
        return None
    return value if np.isfinite(value) else None


def _sanitize_caves_provenance_export(row: Dict[str, Any]) -> None:
    """Omit adversarial-only offset metadata from predictions.csv (Tier-1 CAVES provenance)."""
    geometry_anchor = _finite_result_float(row, "geometry_anchor_offset_m")
    adversarial_offset = _finite_result_float(row, "adversarial_offset_m")
    if geometry_anchor is None and adversarial_offset is not None:
        row["adversarial_offset_m"] = ""


def evaluate_final_pose_acceptance(result: Dict[str, Any], cfg: Dict[str, Any]) -> tuple[bool, str]:
    """Final output gate independent from FGO guard diagnostics.

    Missing diagnostics are treated as non-blocking so native/wrapper rows do
    not get rejected by a gate they cannot populate.
    """
    if not bool(cfg.get("final_pose_gate_enabled", False)):
        return True, ""

    anchor_offset_m = _finite_result_float(result, "geometry_anchor_offset_m")
    max_anchor_offset_m = _finite_result_float(cfg, "final_pose_max_geometry_anchor_offset_m")
    if anchor_offset_m is not None and max_anchor_offset_m is not None and anchor_offset_m > max_anchor_offset_m:
        return False, "final_pose_anchor_offset_reject"

    shape_confidence = _finite_result_float(result, "shape_confidence")
    min_shape_conf = _finite_result_float(cfg, "final_pose_min_shape_conf")
    if shape_confidence is not None and min_shape_conf is not None and shape_confidence < min_shape_conf:
        return False, "final_pose_shape_reject"

    reproj_residual_px = _finite_result_float(result, "reproj_residual_px")
    max_reproj_residual_px = _finite_result_float(cfg, "final_pose_max_reproj_residual_px")
    if (
        reproj_residual_px is not None
        and max_reproj_residual_px is not None
        and reproj_residual_px > max_reproj_residual_px
    ):
        return False, "final_pose_reproj_reject"

    # Visual heading must agree with the IMU/telemetry yaw prior (sigma ~2 deg).
    # A large residual means the homography locked onto a rotated look-alike
    # (e.g. 180-deg ambiguous river banks) - position is then untrustworthy no
    # matter how many inliers agree with each other. Off unless configured.
    max_yaw_residual = _finite_result_float(cfg, "final_pose_max_yaw_prior_residual_deg")
    if max_yaw_residual is not None:
        pred_heading = _finite_result_float(result, "pred_heading")
        prior_yaw = _finite_result_float(result, "prior_yaw_deg")
        if pred_heading is not None and prior_yaw is not None:
            residual = abs(_angle_diff_deg(pred_heading, prior_yaw))
            if np.isfinite(residual) and residual > max_yaw_residual:
                return False, "final_pose_yaw_prior_reject"

    return True, ""


def count_final_pose_gate_rejects(results: Iterable[Dict[str, Any]]) -> int:
    return int(sum(1 for row in results if bool(row.get("final_pose_gate_reject", False)) or row.get("failure_reason") == "final_pose_gate_reject"))


def count_fgo_guard_rejects(results: Iterable[Dict[str, Any]]) -> int:
    return int(sum(1 for row in results if row.get("fgo_reject_code") in FGO_GUARD_REJECT_CODES))


def _angle_diff_deg(pred_deg: Any, gt_deg: Any) -> float:
    """Smallest signed angular difference in degrees."""
    try:
        pred = float(pred_deg)
        gt = float(gt_deg)
    except (TypeError, ValueError):
        return float("nan")
    if not np.isfinite(pred) or not np.isfinite(gt):
        return float("nan")
    return float((pred - gt + 180.0) % 360.0 - 180.0)


def _euler_zyx_to_rotation_matrix(roll_deg: float, pitch_deg: float, yaw_deg: float) -> np.ndarray:
    roll, pitch, yaw = np.radians([roll_deg, pitch_deg, yaw_deg])
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    return rz @ ry @ rx


def _rotation_geodesic_error_deg(
    pred_roll_deg: Any,
    pred_pitch_deg: Any,
    pred_yaw_deg: Any,
    gt_roll_deg: Any,
    gt_pitch_deg: Any,
    gt_yaw_deg: Any,
) -> float:
    vals = [pred_roll_deg, pred_pitch_deg, pred_yaw_deg, gt_roll_deg, gt_pitch_deg, gt_yaw_deg]
    try:
        vals_f = [float(v) for v in vals]
    except (TypeError, ValueError):
        return float("nan")
    if not np.isfinite(vals_f).all():
        return float("nan")
    r_pred = _euler_zyx_to_rotation_matrix(vals_f[0], vals_f[1], vals_f[2])
    r_gt = _euler_zyx_to_rotation_matrix(vals_f[3], vals_f[4], vals_f[5])
    cos_theta = (float(np.trace(r_pred.T @ r_gt)) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0))))


PREDICTIONS_CSV_COLUMNS = [
    "image_name",
    "gt_lat",
    "gt_lon",
    "gt_alt",
    "gt_heading",
    "gt_pitch",
    "gt_roll",
    "pred_lat",
    "pred_lon",
    "pred_alt",
    "pred_heading",
    "pred_roll",
    "pred_pitch",
    "yaw_error_deg",
    "pitch_error_deg",
    "roll_error_deg",
    "rotation_geodesic_error_deg",
    "pose_success",
    "accepted_for_metrics",
    "success",
    "failure_reason",
    "retrieval_anchor_error_m",
    "raw_retrieval_anchor_error_m",
    "geodetic_anchor",
    "raw_retrieval_anchor",
    "retrieval_candidate_index",
    "raw_retrieval_candidate_index",
    "raw_retrieval_topk_indices",
    "selected_reference_name",
    "raw_top1_reference_name",
    "gt_tile_name",
    "tile_identity_correct",
    "raw_top1_tile_identity_correct",
    "query_flight_id",
    "selected_ref_flight_id",
    "raw_top1_ref_flight_id",
    "raw_topk_ref_flights",
    "retrieval_protocol_violation",
    "candidate_rank_selected",
    "candidate_quality_score",
    "num_inliers",
    "num_matches",
    "match_confidence_mean",
    "shape_confidence",
    "reproj_residual_px",
    "reference_preconditioning_attempted",
    "reference_preconditioning_accepted",
    "reference_preconditioning_reason",
    "reference_preconditioning_warp_valid",
    "reference_preconditioning_inliers_before",
    "reference_preconditioning_inliers_after",
    "reference_preconditioning_reproj_before_px",
    "reference_preconditioning_reproj_after_px",
    "reference_preconditioning_shape_before",
    "reference_preconditioning_shape_after",
    "reference_preconditioning_score_before",
    "reference_preconditioning_score_after",
    "pnp_inliers",
    "pnp_reprojection_error_px",
    "dsm_available",
    "dsm_status",
    "homography_valid",
    "geometry_anchor_offset_m",
    "visual_pred_lat",
    "visual_pred_lon",
    "visual_pred_alt",
    "telemetry_prior_policy",
    "native_prior_uses_exact_gt",
    "native_anyvisloc_schema_ok",
    "native_yaw_height_status",
    "E_alt_available",
    "phase1_source",
    "retrieval_verification_status",
    "bottleneck_label",
    "timing_retrieval_ms",
    "timing_matching_ms",
    "timing_geometry_ms",
    "timing_fgo_ms",
    "total_time_ms",
    "adversarial_mode",
    "adversarial_condition",
    "adversarial_offset_m",
    "adversarial_candidate_policy",
    "adversarial_candidate_rank",
    "adversarial_candidate_pool_size",
    "adversarial_eligible_wrong_count",
    "adversarial_selection_reason",
    "adversarial_failure_reason",
    "adversarial_forced_reference_name",
    "adversarial_original_reference_name",
    "adversarial_forced_distance_to_gt_m",
    "is_adversarial",
    "adversarial_assigned_attack",
    "adversarial_attacked",
    "adversarial_inject_rate",
    "is_wrong_reference",
    "retrieval_correct_threshold_m",
    "localization_success_threshold_m",
    "retrieval_rank",
    "localization_is_success",
    "caves_reject",
    "false_reject",
    "bad_pose_label",
    "unsafe_accept",
    "unsafe_accept_severity_m",
    "pre_fgo_error_m",
    "post_fgo_error_m",
    "inlier_ratio",
    "pnp_inlier_ratio",
    "retrieval_score_top1",
    "retrieval_score_top2",
    "retrieval_score_margin",
    "caves_risk_score",
    "caves_confidence",
    "caves_decision",
    "obs_gate_score",
    "obs_gate_reason",
    "caves_evidence_count",
]


def _csv_scalar(value: Any) -> Any:
    """Convert nested/non-finite values into stable CSV-friendly scalars."""
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, cls=NpEncoder, ensure_ascii=True, sort_keys=True)
    try:
        if value is None:
            return ""
        if isinstance(value, float) and not np.isfinite(value):
            return ""
        if hasattr(value, "item"):
            return _csv_scalar(value.item())
    except Exception:
        pass
    return value


def write_predictions_csv(path: Path, results: List[Dict[str, Any]]) -> None:
    """Write the required per-frame benchmark predictions.csv artifact."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=PREDICTIONS_CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in results:
            timings = row.get("timings", {}) if isinstance(row.get("timings"), dict) else {}
            out = dict(row)
            out.setdefault("pose_success", bool(row.get("success", False)))
            out.setdefault("accepted_for_metrics", bool(row.get("success", False)))
            _apply_reliability_schema_defaults(out)
            _sanitize_caves_provenance_export(out)
            out["timing_retrieval_ms"] = timings.get("retrieval_ms", row.get("retrieval_time_ms", ""))
            out["timing_matching_ms"] = timings.get("matching_ms", row.get("matching_time_ms", ""))
            out["timing_geometry_ms"] = timings.get("geometry_ms", row.get("geometry_time_ms", ""))
            out["timing_fgo_ms"] = timings.get("fgo_ms", row.get("fgo_time_ms", ""))
            writer.writerow({col: _csv_scalar(out.get(col, "")) for col in PREDICTIONS_CSV_COLUMNS})


def write_metrics_summary_json(
    path: Path,
    *,
    dataset: str,
    pipeline: str,
    metrics: Dict[str, Any],
    paper_metrics: Dict[str, Any],
    results: List[Dict[str, Any]],
    manifest: Dict[str, Any],
    step_metrics: Optional[Dict[str, Any]] = None,
) -> None:
    """Write stable machine-readable summary for every benchmark method."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "dataset": dataset,
        "pipeline": pipeline,
        "canonical_pipeline": manifest.get("canonical_pipeline", pipeline),
        "method_family": manifest.get("method_family", ""),
        "native_baseline": bool(manifest.get("native_baseline", False)),
        "oracle_mode": bool(manifest.get("oracle_mode", False)),
        "deprecated_alias": bool(manifest.get("deprecated_alias", False)),
        "comparison_tier": manifest.get("comparison_tier", ""),
        "tile_source": manifest.get("tile_source", ""),
        "attitude_source": manifest.get("attitude_source", ""),
        "altitude_source": manifest.get("altitude_source", ""),
        "pose_head": manifest.get("pose_head", ""),
        "is_native": bool(manifest.get("is_native", False)),
        "is_wrapper": bool(manifest.get("is_wrapper", False)),
        "is_hybrid": bool(manifest.get("is_hybrid", False)),
        "uses_oracle": bool(manifest.get("uses_oracle", False)),
        "oracle_modes": manifest.get("oracle_modes", []),
        "benchmark_family": manifest.get("benchmark_family", ""),
        "adversarial_mode": manifest.get("adversarial_mode", "none"),
        "adversarial_condition": manifest.get("adversarial_condition", "clean"),
        "adversarial_candidate_policy": manifest.get("adversarial_candidate_policy", ""),
        "adversarial_candidate_rank": manifest.get("adversarial_candidate_rank"),
        "adversarial_seed": manifest.get("adversarial_seed"),
        "adversarial_official_metrics": bool(manifest.get("adversarial_official_metrics", False)),
        "adversarial_diagnostic": bool(manifest.get("adversarial_diagnostic", False)),
        "official_metric_eligible": bool(manifest.get("official_metric_eligible", manifest.get("run_integrity_for_official_metrics", True))),
        "run_integrity_for_official_metrics": bool(
            manifest.get("run_integrity_for_official_metrics", True)
        ),
        "eval_mode": manifest.get("eval_mode"),
        "strict_components": bool(manifest.get("strict_components", False)),
        "strict_end2end": bool(manifest.get("strict_end2end", False)),
        "retrieval_only": bool(manifest.get("retrieval_only", False)),
        "uav_visloc_protocol": manifest.get("uav_visloc_protocol"),
        "uav_tile_size_px": manifest.get("uav_tile_size_px"),
        "uav_tile_step_px": manifest.get("uav_tile_step_px"),
        "retrieval_top_k": manifest.get("retrieval_top_k"),
        "use_reference_preconditioning": bool(manifest.get("use_reference_preconditioning", False)),
        "reference_preconditioning_phase": manifest.get("reference_preconditioning_phase"),
        "reference_preconditioning_acceptance_policy": manifest.get("reference_preconditioning_acceptance_policy", {}),
        "num_reference_preconditioning_attempted": int(paper_metrics.get("num_reference_preconditioning_attempted", 0) or 0),
        "num_reference_preconditioning_accepted": int(paper_metrics.get("num_reference_preconditioning_accepted", 0) or 0),
        "num_reference_preconditioning_rejected": int(paper_metrics.get("num_reference_preconditioning_rejected", 0) or 0),
        "reference_preconditioning_accept_rate": paper_metrics.get("reference_preconditioning_accept_rate"),
        "final_pose_gate_enabled": manifest.get("final_pose_gate_enabled"),
        "final_pose_min_shape_conf": manifest.get("final_pose_min_shape_conf"),
        "final_pose_max_reproj_residual_px": manifest.get("final_pose_max_reproj_residual_px"),
        "final_pose_max_geometry_anchor_offset_m": manifest.get("final_pose_max_geometry_anchor_offset_m"),
        "num_adversarial_samples": int(paper_metrics.get("num_adversarial_samples", manifest.get("run_stats", {}).get("num_adversarial_samples", 0)) or 0),
        "num_wrong_reference": int(paper_metrics.get("num_wrong_reference", manifest.get("run_stats", {}).get("num_wrong_reference", 0)) or 0),
        "num_bad_pose": int(paper_metrics.get("num_bad_pose", manifest.get("run_stats", {}).get("num_bad_pose", 0)) or 0),
        "num_unsafe_accept": int(paper_metrics.get("num_unsafe_accept", manifest.get("run_stats", {}).get("num_unsafe_accept", 0)) or 0),
        "unsafe_accept_rate": paper_metrics.get("unsafe_accept_rate"),
        "num_false_reject": int(paper_metrics.get("num_false_reject", 0) or 0),
        "false_rejection_rate": paper_metrics.get("false_rejection_rate"),
        "num_caves_reject": int(paper_metrics.get("num_caves_reject", 0) or 0),
        "rejection_rate": paper_metrics.get("rejection_rate"),
        "caves_rejection_rate_denominator": paper_metrics.get("caves_rejection_rate_denominator"),
        "mean_unsafe_accept_severity_m": paper_metrics.get("mean_unsafe_accept_severity_m"),
        "median_unsafe_accept_severity_m": paper_metrics.get("median_unsafe_accept_severity_m"),
        "p90_unsafe_accept_severity_m": paper_metrics.get("p90_unsafe_accept_severity_m"),
        "max_unsafe_accept_severity_m": paper_metrics.get("max_unsafe_accept_severity_m"),
        "num_adversarial_fail_closed": int(paper_metrics.get("num_adversarial_fail_closed", manifest.get("run_stats", {}).get("num_adversarial_fail_closed", 0)) or 0),
        "num_adversarial_unsupported_condition": int(paper_metrics.get("num_adversarial_unsupported_condition", manifest.get("run_stats", {}).get("num_adversarial_unsupported_condition", 0)) or 0),
        "num_adversarial_metadata_unavailable": int(paper_metrics.get("num_adversarial_metadata_unavailable", manifest.get("run_stats", {}).get("num_adversarial_metadata_unavailable", 0)) or 0),
        "num_adversarial_policy_force_rank_k": int(paper_metrics.get("num_adversarial_policy_force_rank_k", manifest.get("run_stats", {}).get("num_adversarial_policy_force_rank_k", 0)) or 0),
        "num_adversarial_policy_nearest_wrong": int(paper_metrics.get("num_adversarial_policy_nearest_wrong", manifest.get("run_stats", {}).get("num_adversarial_policy_nearest_wrong", 0)) or 0),
        "num_adversarial_policy_far_wrong": int(paper_metrics.get("num_adversarial_policy_far_wrong", manifest.get("run_stats", {}).get("num_adversarial_policy_far_wrong", 0)) or 0),
        "num_adversarial_policy_visually_similar_wrong": int(paper_metrics.get("num_adversarial_policy_visually_similar_wrong", manifest.get("run_stats", {}).get("num_adversarial_policy_visually_similar_wrong", 0)) or 0),
        "num_adversarial_policy_cross_flight": int(paper_metrics.get("num_adversarial_policy_cross_flight", manifest.get("run_stats", {}).get("num_adversarial_policy_cross_flight", 0)) or 0),
        "telemetry_prior_policy": manifest.get("telemetry_prior_policy"),
        "native_prior_uses_exact_gt": bool(manifest.get("native_prior_uses_exact_gt", False)),
        "native_anyvisloc_schema_ok": bool(manifest.get("native_anyvisloc_schema_ok", True)),
        "native_yaw_height_status": manifest.get("native_yaw_height_status"),
        "retrieval_db_cache_enabled": bool(manifest.get("retrieval_db_cache_enabled", False)),
        "retrieval_db_cache_hit": bool(manifest.get("retrieval_db_cache_hit", False)),
        "retrieval_db_cache_key": str(manifest.get("retrieval_db_cache_key", "")),
        "retrieval_db_cache_rebuild_reason": str(manifest.get("retrieval_db_cache_rebuild_reason", "")),
        "retrieval_protocol_violation_count": int(
            manifest.get("retrieval_protocol_violation_count", 0)
        ),
        "retrieval_protocol_filtered_candidates_count": int(
            manifest.get("retrieval_protocol_filtered_candidates_count", 0)
        ),
        "uav_selected_query_locations": manifest.get("uav_selected_query_locations", []),
        "uav_selected_sample_signature": manifest.get("uav_selected_sample_signature", {}),
        "uav_corpus_ref_flight_hash": str(manifest.get("uav_corpus_ref_flight_hash", "")),
        "native_anyvisloc_cache_requested": bool(manifest.get("native_anyvisloc_cache_requested", False)),
        "native_anyvisloc_cache_enabled": bool(manifest.get("native_anyvisloc_cache_enabled", False)),
        "native_anyvisloc_cache_key": str(manifest.get("native_anyvisloc_cache_key", "")),
        "native_anyvisloc_cache_hit": bool(manifest.get("native_anyvisloc_cache_hit", False)),
        "native_anyvisloc_cache_rebuild_reason": str(
            manifest.get("native_anyvisloc_cache_rebuild_reason", "")
        ),
        "num_frames": int(len(results)),
        "num_pose_success": int(sum(1 for r in results if bool(r.get("pose_success", r.get("success", False))))),
        "num_accepted_for_metrics": int(
            sum(1 for r in results if bool(r.get("accepted_for_metrics", r.get("success", False))))
        ),
        "metrics": metrics,
        "paper_metrics": paper_metrics,
        "step_metrics": step_metrics or {},
        "failure_reason_counts": {},
        "manifest": manifest,
    }
    for row in results:
        reason = str(row.get("failure_reason", "") or "success")
        payload["failure_reason_counts"][reason] = int(payload["failure_reason_counts"].get(reason, 0)) + 1
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, cls=NpEncoder)
    # Every run also ships results.xlsx (user requirement: all experiments
    # save Excel like reference/summary_benchmarking.xlsx). Best-effort:
    # skipped when predictions.csv is absent or openpyxl is missing.
    if os.environ.get("FORGE_NO_XLSX", "") != "1":
        try:
            import sys as _sys
            _rep = str(Path(__file__).resolve().parents[3] / "scripts" / "reporting")
            if _rep not in _sys.path:
                _sys.path.insert(0, _rep)
            from export_run_xlsx import export_run  # type: ignore
            if (path.parent / "predictions.csv").exists():
                export_run(path.parent)
        except Exception as exc:  # noqa: BLE001
            logger.warning("results.xlsx export skipped: %s", exc)


def is_official_metrics_payload(
    manifest: Optional[Dict[str, Any]] = None,
    paper_metrics: Optional[Dict[str, Any]] = None,
    *,
    require_strict_when_present: bool = True,
) -> bool:
    """Return whether a run is eligible for official strict E2E reporting."""
    manifest = manifest or {}
    paper_metrics = paper_metrics or {}

    def _value(key: str, default: Any = None) -> Any:
        if key in manifest:
            return manifest.get(key)
        if key in paper_metrics:
            return paper_metrics.get(key)
        return default

    eval_mode = _value("eval_mode")
    if eval_mode is not None and str(eval_mode) != "end_to_end":
        return False

    if require_strict_when_present:
        if manifest.get("strict_components") is False:
            return False
        if manifest.get("strict_end2end") is False:
            return False

    if bool(_value("oracle_mode", False)):
        return False
    if bool(_value("uses_oracle", False)):
        return False
    if str(_value("tile_source", "")) in {"oracle_tile", "oracle_pairing"}:
        return False
    comparison_tier = str(_value("comparison_tier", "strict_e2e"))
    if comparison_tier not in {"strict_e2e", "wrapper_e2e"}:
        return False
    if bool(_value("is_native", False)) or bool(_value("native_baseline", False)):
        return False
    if bool(_value("is_hybrid", False)):
        return False
    oracle_modes = _value("oracle_modes", [])
    if isinstance(oracle_modes, list) and len(oracle_modes) > 0:
        return False
    if bool(_value("deprecated_alias", False)):
        return False
    if _is_adversarial_payload(manifest) or _is_adversarial_payload(paper_metrics):
        return False
    if bool(_value("official_metric_eligible", True)) is False:
        return False
    if bool(_value("retrieval_only", False)):
        return False
    if bool(_value("native_prior_uses_exact_gt", False)):
        return False
    if _value("native_anyvisloc_schema_ok", True) is False:
        return False
    if _value("run_integrity_for_official_metrics", True) is False:
        return False
    if bool(_value("end2end_oracle_fallback_contaminated", False)):
        return False
    if _safe_int(_value("retrieval_protocol_violation_count", 0), default=0) > 0:
        return False
    # Note: retrieval_protocol_filtered_candidates_count > 0 means the
    # intra_flight filter DID ITS JOB; it is not an integrity failure.
    if bool(_value("retrieval_protocol_violation", False)):
        return False
    return True


def _flight_id_from_ref_name(ref_name: str) -> str:
    """Resolve UAV-VisLoc flight/location id from a corpus reference name."""
    parts = str(ref_name or "").replace("\\", "/").split("/")
    return parts[0] if parts else ""


def resolve_uav_visloc_protocol_fields(
    row: Dict[str, Any],
    *,
    protocol: str,
    reference_names: Optional[Dict[int, str]] = None,
) -> Dict[str, Any]:
    """Resolve per-frame UAV-VisLoc protocol diagnostics."""
    protocol = str(protocol or "intra_flight")
    reference_names = reference_names or {}
    query_flight = (
        str(row.get("query_flight_id", "") or "")
        or str(row.get("flight_id", "") or "")
        or _flight_id_from_ref_name(str(row.get("image_name", "")))
    )
    selected_ref_name = str(row.get("selected_reference_name", "") or "")
    raw_top1_name = str(row.get("raw_top1_reference_name", "") or "")
    selected_ref_flight = _flight_id_from_ref_name(selected_ref_name)
    raw_top1_ref_flight = _flight_id_from_ref_name(raw_top1_name)

    raw_topk_ref_flights: List[str] = []
    raw_topk_indices = row.get("raw_retrieval_topk_indices", [])
    if isinstance(raw_topk_indices, list):
        for raw_idx in raw_topk_indices:
            idx = _safe_int(raw_idx, default=-1)
            name = str(reference_names.get(idx, ""))
            if not name and idx == _safe_int(row.get("raw_retrieval_candidate_index", -1), default=-1):
                name = raw_top1_name
            flight = _flight_id_from_ref_name(name)
            if flight:
                raw_topk_ref_flights.append(flight)

    violation = bool(row.get("retrieval_protocol_violation", False))
    if protocol == "intra_flight" and query_flight:
        # A violation is USING a cross-flight reference. Judging the raw
        # pre-filter top-K flagged frames whose cross-flight candidates were
        # correctly discarded (3,996/3,996 flags were false positives; the
        # selected reference matched the query flight on every one).
        if selected_ref_flight:
            violation = violation or (selected_ref_flight != query_flight)

    return {
        "query_flight_id": query_flight,
        "selected_ref_flight_id": selected_ref_flight,
        "raw_top1_ref_flight_id": raw_top1_ref_flight,
        "raw_topk_ref_flights": raw_topk_ref_flights,
        "retrieval_protocol_violation": bool(violation),
    }


def build_retrieval_cache_key_payload(
    *,
    schema_version: int,
    dataset_name: str,
    retrieval_backend: str,
    model_signature_hash: str,
    preprocess_signature_hash: str,
    corpus_hash: str,
    corpus_size: int,
    uav_visloc_protocol: Optional[str] = None,
    selected_query_locations: Optional[List[str]] = None,
    selected_sample_signature: Optional[Dict[str, Any]] = None,
    corpus_ref_flight_hash: Optional[str] = None,
    uav_tile_size_px: Optional[int] = None,
    uav_tile_step_px: Optional[int] = None,
) -> Dict[str, Any]:
    """Build retrieval cache key payload with dataset/protocol identity."""
    payload: Dict[str, Any] = {
        "schema_version": int(schema_version),
        "dataset_name": str(dataset_name),
        "retrieval_backend": str(retrieval_backend),
        "model_signature_hash": str(model_signature_hash),
        "preprocess_signature_hash": str(preprocess_signature_hash),
        "corpus_hash": str(corpus_hash),
        "corpus_size": int(corpus_size),
    }
    if str(dataset_name) == "uav_visloc":
        payload.update({
            "uav_visloc_protocol": str(uav_visloc_protocol or "intra_flight"),
            "selected_query_locations": sorted(str(v) for v in (selected_query_locations or [])),
            "selected_sample_signature": selected_sample_signature or {},
            "corpus_ref_flight_hash": str(corpus_ref_flight_hash or ""),
            "uav_tile_size_px": _safe_int(uav_tile_size_px, default=2048, min_value=1),
            "uav_tile_step_px": _safe_int(uav_tile_step_px, default=512, min_value=1),
            # retrieval_top_k intentionally NOT in the key: the encoded DB does
            # not depend on k (ranking truncation is query-time), and keying on
            # it forced a full re-encode per k-sweep cell.
        })
    return payload


def cleanup_stale_retrieval_cache(
    cache_dir: Path,
    current_cache_key: str,
    dataset_name: str,
    retrieval_backend: str,
) -> Dict[str, Any]:
    """Remove stale cache entries for a dataset/backend combination.

    Keeps only the entry matching *current_cache_key*. Other entries under
    the same ``cache_dir / dataset_name / retrieval_backend /`` are deleted.
    Locked entries (another process building them) are skipped.

    Returns ``{"removed": [...], "kept": str, "skipped": [...]}``.
    """
    backend_dir = cache_dir / str(dataset_name) / str(retrieval_backend)
    removed: List[str] = []
    skipped: List[str] = []

    if not backend_dir.is_dir():
        return {"removed": removed, "kept": current_cache_key, "skipped": skipped}

    for entry in sorted(backend_dir.iterdir()):
        if not entry.is_dir():
            continue
        if entry.name == current_cache_key:
            continue
        lock_path = entry / ".build.lock"
        try:
            if fcntl is not None:
                with open(lock_path, "w", encoding="utf-8") as lf:
                    fcntl.flock(lf.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    shutil.rmtree(entry, ignore_errors=True)
                    fcntl.flock(lf.fileno(), fcntl.LOCK_UN)
            else:
                shutil.rmtree(entry, ignore_errors=True)
            removed.append(entry.name[:12])
            logger.info(
                "Removed stale cache %s/%s/%s",
                dataset_name, retrieval_backend, entry.name[:12],
            )
        except (BlockingIOError, OSError):
            skipped.append(entry.name[:12])
            logger.info(
                "Skipped stale cache %s (locked by another process)",
                entry.name[:12],
            )

    return {"removed": removed, "kept": current_cache_key, "skipped": skipped}


def audit_uav_visloc_reference_anchors(
    references: List[Dict[str, Any]],
    *,
    bboxes_by_flight: Optional[Dict[str, Dict[str, float]]] = None,
    query_rows: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Audit UAV-VisLoc reference anchors and query/map protocol consistency."""
    bboxes_by_flight = bboxes_by_flight or {}
    query_rows = query_rows or []
    zero_anchor_count = 0
    out_of_bbox_count = 0
    anchor_groups: Dict[tuple, List[str]] = {}
    anchor_items: List[tuple[str, float, float]] = []
    total = 0

    def _bbox_contains(bbox: Dict[str, Any], lat: float, lon: float) -> bool:
        min_lat = _safe_float(bbox.get("min_lat", bbox.get("lt_lat", np.nan)), default=float("nan"))
        max_lat = _safe_float(bbox.get("max_lat", bbox.get("rb_lat", np.nan)), default=float("nan"))
        min_lon = _safe_float(bbox.get("min_lon", bbox.get("lt_lon", np.nan)), default=float("nan"))
        max_lon = _safe_float(bbox.get("max_lon", bbox.get("rb_lon", np.nan)), default=float("nan"))
        lo_lat, hi_lat = sorted([min_lat, max_lat])
        lo_lon, hi_lon = sorted([min_lon, max_lon])
        return bool(
            np.isfinite(lat)
            and np.isfinite(lon)
            and np.isfinite(lo_lat)
            and np.isfinite(hi_lat)
            and np.isfinite(lo_lon)
            and np.isfinite(hi_lon)
            and lo_lat <= lat <= hi_lat
            and lo_lon <= lon <= hi_lon
        )

    def _row_flight(row: Dict[str, Any]) -> str:
        metadata = row.get("metadata", {}) if isinstance(row.get("metadata"), dict) else {}
        return (
            str(row.get("flight_id", "") or "")
            or str(metadata.get("location", "") or "")
            or _flight_id_from_ref_name(str(row.get("image_name", row.get("ref_name", ""))))
        )

    def _row_lat_lon(row: Dict[str, Any]) -> tuple[float, float]:
        pose = row.get("gt_pose", {}) if isinstance(row.get("gt_pose"), dict) else {}
        lat = _safe_float(row.get("gt_lat", row.get("lat", pose.get("lat", np.nan))), default=float("nan"))
        lon = _safe_float(row.get("gt_lon", row.get("lon", pose.get("lon", np.nan))), default=float("nan"))
        return lat, lon

    def _summary(values: List[float]) -> Dict[str, Any]:
        vals = np.asarray([float(v) for v in values if np.isfinite(float(v))], dtype=np.float64)
        if len(vals) == 0:
            return {"count": 0, "p50": None, "p90": None, "mean": None, "max": None}
        return {
            "count": int(len(vals)),
            "p50": _json_number(float(np.percentile(vals, 50))),
            "p90": _json_number(float(np.percentile(vals, 90))),
            "mean": _json_number(float(np.mean(vals))),
            "max": _json_number(float(np.max(vals))),
        }

    for ref in references:
        total += 1
        anchor = ref.get("anchor", {}) or {}
        lat = _safe_float(anchor.get("lat", np.nan), default=float("nan"))
        lon = _safe_float(anchor.get("lon", np.nan), default=float("nan"))
        ref_name = str(ref.get("ref_name", ""))
        flight = str(ref.get("flight_id", "")) or _flight_id_from_ref_name(ref_name)
        if not np.isfinite(lat) or not np.isfinite(lon) or (abs(lat) < 1e-6 and abs(lon) < 1e-6):
            zero_anchor_count += 1
        anchor_groups.setdefault((round(lat, 8), round(lon, 8)), []).append(ref_name)
        if np.isfinite(lat) and np.isfinite(lon):
            anchor_items.append((flight, float(lat), float(lon)))

        bbox = bboxes_by_flight.get(flight)
        if bbox:
            if np.isfinite(lat) and np.isfinite(lon) and not _bbox_contains(bbox, lat, lon):
                out_of_bbox_count += 1

    query_bbox_checked_count = 0
    query_out_of_bbox_count = 0
    query_out_of_bbox_examples: List[Dict[str, Any]] = []
    nearest_tile_floor_values: List[float] = []
    for row in query_rows:
        flight = _row_flight(row)
        q_lat, q_lon = _row_lat_lon(row)
        bbox = bboxes_by_flight.get(flight)
        if bbox and np.isfinite(q_lat) and np.isfinite(q_lon):
            query_bbox_checked_count += 1
            if not _bbox_contains(bbox, q_lat, q_lon):
                query_out_of_bbox_count += 1
                if len(query_out_of_bbox_examples) < 10:
                    query_out_of_bbox_examples.append({
                        "image_name": str(row.get("image_name", "")),
                        "flight_id": flight,
                        "lat": float(q_lat),
                        "lon": float(q_lon),
                    })
        if np.isfinite(q_lat) and np.isfinite(q_lon) and anchor_items:
            same_flight = [(lat, lon) for item_flight, lat, lon in anchor_items if item_flight == flight]
            candidates = same_flight if same_flight else [(lat, lon) for _item_flight, lat, lon in anchor_items]
            nearest_tile_floor_values.append(
                min(_approx_latlon_distance_m(q_lat, q_lon, lat, lon) for lat, lon in candidates)
            )

    duplicate_groups = {
        f"{lat:.8f},{lon:.8f}": names
        for (lat, lon), names in anchor_groups.items()
        if len(names) > 1
    }
    passed = bool(
        total > 0
        and zero_anchor_count == 0
        and out_of_bbox_count == 0
        and query_out_of_bbox_count == 0
        and len(duplicate_groups) == 0
    )
    return {
        "dataset": "uav_visloc",
        "passed": passed,
        "total_references": int(total),
        "zero_anchor_count": int(zero_anchor_count),
        "duplicate_anchor_groups": int(len(duplicate_groups)),
        "duplicate_anchor_examples": dict(list(duplicate_groups.items())[:10]),
        "out_of_bbox_count": int(out_of_bbox_count),
        "query_bbox_checked_count": int(query_bbox_checked_count),
        "query_out_of_bbox_count": int(query_out_of_bbox_count),
        "query_out_of_bbox_examples": query_out_of_bbox_examples,
        "nearest_tile_oracle_floor_m_summary": _summary(nearest_tile_floor_values),
    }


def _approx_latlon_distance_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Approximate WGS84 local distance for diagnostics."""
    m_per_deg_lat = 111_320.0
    mean_lat = np.radians((lat1 + lat2) * 0.5)
    m_per_deg_lon = 111_320.0 * np.cos(mean_lat)
    dlat_m = (lat1 - lat2) * m_per_deg_lat
    dlon_m = (lon1 - lon2) * m_per_deg_lon
    return float(np.hypot(dlat_m, dlon_m))


def compute_retrieval_diagnostics(
    results: List[Dict[str, Any]],
    reference_anchors: Dict[int, Dict[str, float]],
    *,
    dataset_name: str | None = None,
    thresholds_m: List[float] | None = None,
    topk_values: List[int] | None = None,
) -> Dict[str, Any]:
    """Compute dataset-agnostic retrieval diagnostics for raw TopK and selected anchors."""
    thresholds_m = thresholds_m or [20.0, 50.0, 150.0, 300.0]
    topk_values = sorted({1, *[int(k) for k in (topk_values or [5, 10, 20, 50])]})
    anchor_items = [
        (idx, float(a.get("lat", np.nan)), float(a.get("lon", np.nan)))
        for idx, a in (reference_anchors or {}).items()
        if np.isfinite(float(a.get("lat", np.nan))) and np.isfinite(float(a.get("lon", np.nan)))
    ]

    def _anchor_error(anchor: Any, row: Dict[str, Any]) -> float:
        if not isinstance(anchor, dict):
            return float("nan")
        lat = _safe_float(anchor.get("lat", np.nan), default=float("nan"))
        lon = _safe_float(anchor.get("lon", np.nan), default=float("nan"))
        gt_lat = _safe_float(row.get("gt_lat", np.nan), default=float("nan"))
        gt_lon = _safe_float(row.get("gt_lon", np.nan), default=float("nan"))
        if not all(np.isfinite(v) for v in [lat, lon, gt_lat, gt_lon]):
            return float("nan")
        return _approx_latlon_distance_m(lat, lon, gt_lat, gt_lon)

    def _topk_error(row: Dict[str, Any], k: int) -> float:
        gt_lat = _safe_float(row.get("gt_lat", np.nan), default=float("nan"))
        gt_lon = _safe_float(row.get("gt_lon", np.nan), default=float("nan"))
        if not np.isfinite(gt_lat) or not np.isfinite(gt_lon):
            return float("nan")
        indices = row.get("raw_retrieval_topk_indices", [])
        if not isinstance(indices, list):
            return float("nan")
        best = float("inf")
        for idx_raw in indices[:k]:
            idx = _safe_int(idx_raw, default=-1)
            anchor = reference_anchors.get(idx)
            if not isinstance(anchor, dict):
                continue
            err = _anchor_error(anchor, row)
            if np.isfinite(err):
                best = min(best, err)
        return best if np.isfinite(best) else float("nan")

    def _oracle_floor(row: Dict[str, Any]) -> float:
        gt_lat = _safe_float(row.get("gt_lat", np.nan), default=float("nan"))
        gt_lon = _safe_float(row.get("gt_lon", np.nan), default=float("nan"))
        if not np.isfinite(gt_lat) or not np.isfinite(gt_lon) or not anchor_items:
            return float("nan")
        best = min(
            _approx_latlon_distance_m(lat, lon, gt_lat, gt_lon)
            for _idx, lat, lon in anchor_items
        )
        return float(best)

    rows = []
    previous_selected_anchor: Dict[str, Any] | None = None
    for row in results:
        flight = str(row.get("flight_id", "")) or _flight_id_from_ref_name(str(row.get("image_name", "")))
        selected_err = _anchor_error(row.get("geodetic_anchor"), row)
        raw_top1_err = _anchor_error(row.get("raw_retrieval_anchor"), row)
        if not np.isfinite(raw_top1_err):
            raw_idx = _safe_int(row.get("raw_retrieval_candidate_index", -1), default=-1)
            raw_top1_err = _anchor_error(reference_anchors.get(raw_idx), row)
        selected_anchor = row.get("geodetic_anchor")
        anchor_jump_m = (
            _anchor_error(previous_selected_anchor, {"gt_lat": selected_anchor.get("lat"), "gt_lon": selected_anchor.get("lon")})
            if isinstance(previous_selected_anchor, dict) and isinstance(selected_anchor, dict)
            else float("nan")
        )
        if isinstance(selected_anchor, dict):
            previous_selected_anchor = selected_anchor
        out_row = {
            "flight_id": flight or "unknown",
            "selected_error_m": selected_err,
            "raw_top1_error_m": raw_top1_err,
            "nearest_tile_oracle_floor_m": _oracle_floor(row),
            "anchor_jump_m": anchor_jump_m,
        }
        for k in topk_values:
            out_row[f"raw_top{k}_error_m"] = _topk_error(row, int(k))
        rows.append(out_row)

    def _summarize(subrows: List[Dict[str, Any]]) -> Dict[str, Any]:
        out: Dict[str, Any] = {"num_frames": int(len(subrows))}
        error_keys = (
            ["selected_error_m", "raw_top1_error_m"]
            + [f"raw_top{int(k)}_error_m" for k in topk_values]
            + ["nearest_tile_oracle_floor_m", "anchor_jump_m"]
        )
        for key in error_keys:
            vals = np.array([float(r[key]) for r in subrows if np.isfinite(float(r[key]))], dtype=np.float64)
            out[f"{key}_p50"] = _json_number(float(np.percentile(vals, 50))) if len(vals) else None
            out[f"{key}_p90"] = _json_number(float(np.percentile(vals, 90))) if len(vals) else None
            out[f"{key}_mean"] = _json_number(float(np.mean(vals))) if len(vals) else None
            for threshold in thresholds_m:
                out[f"{key}_hit_at_{int(threshold)}m"] = (
                    _json_number(float(np.mean(vals <= float(threshold)))) if len(vals) else None
                )
        for threshold in thresholds_m:
            selected = out.get(f"selected_error_m_hit_at_{int(threshold)}m")
            raw_top50 = out.get(f"raw_top50_error_m_hit_at_{int(threshold)}m")
            if selected is not None and raw_top50 is not None:
                out[f"selected_vs_raw_gap_at_{int(threshold)}m"] = _json_number(float(raw_top50) - float(selected))
            else:
                out[f"selected_vs_raw_gap_at_{int(threshold)}m"] = None
        return out

    by_flight: Dict[str, Any] = {}
    for flight in sorted({r["flight_id"] for r in rows}):
        by_flight[flight] = _summarize([r for r in rows if r["flight_id"] == flight])
    return {
        "schema_version": 1,
        "dataset": dataset_name,
        "thresholds_m": thresholds_m,
        "topk_values": [int(k) for k in topk_values],
        "num_reference_anchors": int(len(anchor_items)),
        "global": _summarize(rows),
        "by_flight": by_flight,
        "rows": rows,
    }


def compute_uav_visloc_retrieval_diagnostics(
    results: List[Dict[str, Any]],
    reference_anchors: Dict[int, Dict[str, float]],
    *,
    thresholds_m: List[float] | None = None,
    topk_values: List[int] | None = None,
) -> Dict[str, Any]:
    """Backward-compatible UAV-VisLoc diagnostics artifact."""
    return compute_retrieval_diagnostics(
        results,
        reference_anchors,
        dataset_name="uav_visloc",
        thresholds_m=thresholds_m,
        topk_values=topk_values,
    )


def _finite_float_list(values: List[Any]) -> List[float]:
    out: List[float] = []
    for value in values:
        try:
            v = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(v):
            out.append(v)
    return out


def _timing_stats_ms(values: List[Any]) -> Dict[str, Any]:
    vals = np.asarray(_finite_float_list(values), dtype=np.float64)
    if len(vals) == 0:
        return {
            "timing_ms_count": 0,
            "timing_ms_total": None,
            "timing_ms_mean": None,
            "timing_ms_p50": None,
            "timing_ms_p90": None,
            "timing_ms_max": None,
        }
    return {
        "timing_ms_count": int(len(vals)),
        "timing_ms_total": _json_number(float(np.sum(vals))),
        "timing_ms_mean": _json_number(float(np.mean(vals))),
        "timing_ms_p50": _json_number(float(np.percentile(vals, 50))),
        "timing_ms_p90": _json_number(float(np.percentile(vals, 90))),
        "timing_ms_max": _json_number(float(np.max(vals))),
    }


def compute_step_metrics_summary(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate per-step runtime and health metrics for benchmark audit logs."""
    num_frames = int(len(results))

    def _timing(row: Dict[str, Any], key: str) -> float:
        timings = row.get("timings", {})
        if not isinstance(timings, dict):
            return float("nan")
        return _safe_float(timings.get(key, np.nan), default=float("nan"))

    def _has_finite_anchor(row: Dict[str, Any]) -> bool:
        anchor = row.get("geodetic_anchor")
        if not isinstance(anchor, dict):
            return False
        lat = _safe_float(anchor.get("lat", np.nan), default=float("nan"))
        lon = _safe_float(anchor.get("lon", np.nan), default=float("nan"))
        return bool(np.isfinite(lat) and np.isfinite(lon))

    pose_success = [bool(r.get("pose_success", r.get("success", False))) for r in results]
    accepted = [bool(r.get("accepted_for_metrics", r.get("success", False))) for r in results]
    retrieval_success = [_has_finite_anchor(r) for r in results]
    matching_success = [
        _safe_int(r.get("num_matches", 0), default=0, min_value=0) > 0
        or _safe_int(r.get("num_inliers", 0), default=0, min_value=0) > 0
        for r in results
    ]
    geometry_success = pose_success
    fgo_attempted = [
        ("fgo_applied" in r)
        or np.isfinite(_timing(r, "fgo_ms")) and _timing(r, "fgo_ms") > 0
        for r in results
    ]
    fgo_success = [bool(r.get("fgo_applied", False)) for r in results]

    def _rate(flags: List[bool], denom: int | None = None) -> Any:
        d = int(num_frames if denom is None else denom)
        return _json_number(float(sum(flags) / d)) if d > 0 else None

    def _step(
        name: str,
        timing_key: str,
        success_flags: List[bool],
        attempted_flags: Optional[List[bool]] = None,
    ) -> Dict[str, Any]:
        attempted_flags = attempted_flags if attempted_flags is not None else [True] * num_frames
        attempted_count = int(sum(bool(v) for v in attempted_flags))
        success_count = int(sum(bool(s) and bool(a) for s, a in zip(success_flags, attempted_flags)))
        failure_count = max(0, attempted_count - success_count)
        payload = {
            "name": name,
            "attempted": attempted_count,
            "success": success_count,
            "failed": failure_count,
            "success_rate": _rate([bool(s) and bool(a) for s, a in zip(success_flags, attempted_flags)], attempted_count),
        }
        payload.update(_timing_stats_ms([_timing(r, timing_key) for r in results]))
        return payload

    failure_reason_counts: Dict[str, int] = {}
    bottleneck_counts: Dict[str, int] = {}
    for row in results:
        if not bool(row.get("accepted_for_metrics", row.get("success", False))):
            reason = str(row.get("failure_reason", "") or "unknown_failure")
            failure_reason_counts[reason] = int(failure_reason_counts.get(reason, 0)) + 1
        bottleneck = str(row.get("bottleneck_label", "") or "unknown")
        bottleneck_counts[bottleneck] = int(bottleneck_counts.get(bottleneck, 0)) + 1

    inlier_values = _finite_float_list([r.get("num_inliers", np.nan) for r in results])
    shape_values = _finite_float_list([r.get("shape_confidence", np.nan) for r in results])
    summary = {
        "schema_version": 1,
        "num_frames": num_frames,
        "overall": {
            "pose_success_count": int(sum(pose_success)),
            "accepted_for_metrics_count": int(sum(accepted)),
            "pose_success_rate": _rate(pose_success),
            "accepted_for_metrics_rate": _rate(accepted),
            "mean_inliers": _json_number(float(np.mean(inlier_values))) if inlier_values else None,
            "p50_inliers": _json_number(float(np.percentile(inlier_values, 50))) if inlier_values else None,
            "mean_shape_confidence": _json_number(float(np.mean(shape_values))) if shape_values else None,
        },
        "steps": {
            "retrieval": _step("retrieval", "retrieval_ms", retrieval_success),
            "matching": _step("matching", "matching_ms", matching_success),
            "geometry": _step("geometry", "geometry_ms", geometry_success),
            "fgo": _step("fgo", "fgo_ms", fgo_success, fgo_attempted),
            "total": {
                "name": "total",
                "attempted": num_frames,
                "success": int(sum(accepted)),
                "failed": max(0, num_frames - int(sum(accepted))),
                "success_rate": _rate(accepted),
                **_timing_stats_ms([r.get("total_time_ms", _timing(r, "total_ms")) for r in results]),
            },
        },
        "failure_reason_counts": failure_reason_counts,
        "bottleneck_counts": bottleneck_counts,
    }
    return summary


def log_step_metrics_summary(summary: Dict[str, Any], *, dataset: str, pipeline: str) -> None:
    """Emit concise but detailed phase-level audit logs."""
    overall = summary.get("overall", {})
    logger.info(
        "STEP_AUDIT dataset=%s pipeline=%s frames=%s pose_success=%s/%s (%.3f) accepted=%s/%s (%.3f)",
        dataset,
        pipeline,
        summary.get("num_frames", 0),
        overall.get("pose_success_count", 0),
        summary.get("num_frames", 0),
        float(overall.get("pose_success_rate") or 0.0),
        overall.get("accepted_for_metrics_count", 0),
        summary.get("num_frames", 0),
        float(overall.get("accepted_for_metrics_rate") or 0.0),
    )
    for step_name, step in (summary.get("steps") or {}).items():
        logger.info(
            "STEP_AUDIT phase=%s attempted=%s success=%s failed=%s success_rate=%s "
            "time_ms(mean=%s,p50=%s,p90=%s,max=%s,total=%s)",
            step_name,
            step.get("attempted"),
            step.get("success"),
            step.get("failed"),
            step.get("success_rate"),
            step.get("timing_ms_mean"),
            step.get("timing_ms_p50"),
            step.get("timing_ms_p90"),
            step.get("timing_ms_max"),
            step.get("timing_ms_total"),
        )
    logger.info("STEP_AUDIT failures=%s", summary.get("failure_reason_counts", {}))
    logger.info("STEP_AUDIT bottlenecks=%s", summary.get("bottleneck_counts", {}))


def select_native_anyvisloc_candidate_by_pnp_inliers(
    candidates: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Native AnyVisLoc Top-N rerank: PnP inliers first, descriptor rank as tie-breaker."""
    if not candidates:
        return None

    def _score(item: Dict[str, Any]) -> tuple:
        inliers = _safe_int(item.get("pnp_inliers", item.get("num_inliers", 0)), default=0, min_value=0)
        rank = _safe_int(item.get("retrieval_rank", 10**9), default=10**9, min_value=0)
        retrieval_score = _safe_float(item.get("retrieval_score", 0.0), default=0.0)
        return (inliers, -rank, retrieval_score)

    return max(candidates, key=_score)


def check_ortholoc_preflight(
    *,
    importer: Callable[[str], Any] = importlib.import_module,
    repo_root: Path = REPO_ROOT,
    dataset_root: Optional[Path] = None,
) -> Dict[str, Any]:
    """Fail-closed OrthoLoC dependency/import preflight."""
    reference_path = Path(repo_root) / "reference" / "ortholoc"
    scripts_path = reference_path / "scripts"
    required = ["poselib", "rasterio", "appdirs"]
    missing = []
    for module_name in required:
        try:
            importer(module_name)
        except Exception:
            missing.append(module_name)
    if dataset_root is None:
        env_root = os.environ.get("ORTHOLOC_DATA") or os.environ.get("ORTHOLOC_DATA_ROOT")
        dataset_root = Path(env_root) if env_root else None

    def _has_named_asset(root: Optional[Path], patterns: List[str]) -> bool:
        if root is None or not root.exists():
            return False
        for pattern in patterns:
            try:
                next(root.rglob(pattern))
                return True
            except StopIteration:
                continue
            except Exception:
                return False
        return False

    reference_exists = bool(reference_path.exists())
    scripts_exist = bool((scripts_path / "run_localization.py").exists() and (scripts_path / "run_benchmark.py").exists())
    dop_assets_present = _has_named_asset(dataset_root, ["*DOP*", "*dop*"])
    dsm_assets_present = _has_named_asset(dataset_root, ["*DSM*", "*dsm*"])
    # DOP/DSM extraction needs the import chain above; without it the wiring
    # cannot run, so report it un-wired (fail-closed contract).
    dataset_extraction_wired = not missing
    return {
        "ready": bool(reference_exists and scripts_exist and not missing and dataset_extraction_wired),
        "reference_path": str(reference_path),
        "reference_exists": reference_exists,
        "scripts_path": str(scripts_path),
        "scripts_exist": scripts_exist,
        "dataset_root": str(dataset_root) if dataset_root is not None else "",
        "dataset_root_exists": bool(dataset_root is not None and dataset_root.exists()),
        "dop_assets_present": dop_assets_present,
        "dsm_assets_present": dsm_assets_present,
        "dataset_extraction_wired": dataset_extraction_wired,
        "numeric_runtime_enabled": False,
        "missing_dependencies": missing,
        "required_dependencies": required,
    }


def _env_flag(name: str, default: bool) -> bool:
    """Parse boolean env flags with common truthy/falsy forms."""
    raw = os.environ.get(name)
    if raw is None:
        return bool(default)
    return str(raw).strip().lower() not in {"0", "false", "no", "off", ""}


def _stable_json_hash(payload: Any) -> str:
    """Deterministic SHA256 hash for JSON-serializable payloads."""
    raw = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        cls=NpEncoder,
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _ablation_table_float(value: Any, default: float = float("nan")) -> float:
    """Coerce table metric values (including JSON ``None`` from _json_number) to float."""
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


# =============================================================================
# SIMULATED UAV TELEMETRY (Realistic Sensor Noise)
# =============================================================================
# In real UAV deployment, attitude (pitch/roll/yaw) comes from IMU, altitude
# from barometer, and coarse position from GNSS. These sensors have inherent
# noise + bias drift. To avoid GT leakage, we simulate realistic telemetry by
# adding noise to GT values before any pipeline consumes them as priors.
# Noise parameters are based on consumer/industrial-grade MEMS IMUs (Pixhawk,
# DJI Mavic/Matrice class) under typical flight vibration conditions.

# Noise standard deviations (aggressive-but-realistic, "worst-case consumer")
_SIGMA_ALT_M = 3.0        # Barometer: ±0.5-2m typical + downwash/wind → σ=3m
_SIGMA_PITCH_DEG = 3.0    # IMU pitch after Kalman fusion: ~1-2° RMS + vibration → σ=3°
_SIGMA_ROLL_DEG = 3.0     # IMU roll (same as pitch)
_SIGMA_YAW_DEG = 8.0      # Magnetometer: 5-15° drift from motor EMI → σ=8°
_SIGMA_GNSS_M = 5.0       # GNSS horizontal fix noise used for native GPS/search priors

# Bias drift rate per frame (simulates slow sensor drift)
_BIAS_DRIFT_ALT = 0.02    # ~±1m drift over 1000 frames
_BIAS_DRIFT_PITCH = 0.005 # ~±0.5° drift over 1000 frames
_BIAS_DRIFT_ROLL = 0.005
_BIAS_DRIFT_YAW = 0.015   # ~±1.5° drift over 1000 frames


def _precompute_bias_drift(total_frames: int, seed: int = 12345) -> list:
    """Precompute deterministic bias drift trajectory for all frames."""
    import random
    rng = random.Random(seed)
    drift = []
    alt, pitch, roll, yaw = 0.0, 0.0, 0.0, 0.0
    for _ in range(total_frames):
        alt += rng.gauss(0, _BIAS_DRIFT_ALT)
        pitch += rng.gauss(0, _BIAS_DRIFT_PITCH)
        roll += rng.gauss(0, _BIAS_DRIFT_ROLL)
        yaw += rng.gauss(0, _BIAS_DRIFT_YAW)
        drift.append((alt, pitch, roll, yaw))
    return drift


def get_simulated_telemetry(
    gt_pose: dict,
    frame_index: int,
    bias_drift: list | None = None,
) -> dict:
    """
    Simulate UAV telemetry from GT pose with realistic sensor noise.

    This replaces direct GT access for IPM, FGO, and geometry altitude hints.
    Ensures no GT leakage while modelling real sensor characteristics.

    Args:
        gt_pose: Ground-truth pose dict with keys like 'alt', 'pitch', 'roll', 'yaw'.
        frame_index: Current frame index (used as deterministic seed).
        bias_drift: Precomputed bias drift array from _precompute_bias_drift().

    Returns:
        Telemetry dict with noisy sensor values.
    """
    import random
    import copy

    telemetry = copy.deepcopy(gt_pose)
    rng = random.Random(42 + frame_index)  # Deterministic per-frame jitter

    # Gaussian jitter (zero-mean, per-frame)
    jitter_alt = rng.gauss(0, _SIGMA_ALT_M)
    jitter_pitch = rng.gauss(0, _SIGMA_PITCH_DEG)
    jitter_roll = rng.gauss(0, _SIGMA_ROLL_DEG)
    jitter_yaw = rng.gauss(0, _SIGMA_YAW_DEG)
    jitter_north_m = rng.gauss(0, _SIGMA_GNSS_M)
    jitter_east_m = rng.gauss(0, _SIGMA_GNSS_M)

    # Bias drift (correlated across frames)
    if bias_drift is not None and 0 <= frame_index < len(bias_drift):
        b_alt, b_pitch, b_roll, b_yaw = bias_drift[frame_index]
    else:
        b_alt = b_pitch = b_roll = b_yaw = 0.0

    if 'alt' in telemetry:
        telemetry['alt'] = float(telemetry['alt']) + jitter_alt + b_alt
    if 'rel_alt' in telemetry:
        telemetry['rel_alt'] = float(telemetry['rel_alt']) + jitter_alt + b_alt
    if 'pitch' in telemetry:
        telemetry['pitch'] = float(telemetry['pitch']) + jitter_pitch + b_pitch
    if 'roll' in telemetry:
        telemetry['roll'] = float(telemetry['roll']) + jitter_roll + b_roll
    if 'yaw' in telemetry:
        telemetry['yaw'] = float(telemetry['yaw']) + jitter_yaw + b_yaw
    if 'lat' in telemetry and 'lon' in telemetry:
        lat0 = _safe_float(telemetry.get('lat'), default=float("nan"))
        lon0 = _safe_float(telemetry.get('lon'), default=float("nan"))
        if np.isfinite(lat0) and np.isfinite(lon0):
            lat_scale = 111320.0
            lon_scale = max(1e-6, abs(111320.0 * np.cos(np.radians(lat0))))
            telemetry['lat'] = lat0 + jitter_north_m / lat_scale
            telemetry['lon'] = lon0 + jitter_east_m / lon_scale

    return telemetry


# =============================================================================
# PIPELINE CONFIGURATIONS
# =============================================================================

PIPELINE_CONFIGS = {
    'ours': {
        'name': 'Ours (Full Pipeline)',
        'retrieval': 'ultravpr',
        'retrieval_top_k': 5,
        'use_ipm': True,
        'alignment': 'coarse_to_fine',
        'matcher': 'aliked_lightglue',
        'geometry': 'homography_scale',
        'use_fgo': True,
        'use_lgi_filter': True,
        'enable_roma_fallback': True,
        'description': 'UltraVPR + IPM + CoarseToFine + LightGlue + RoMa fallback + LGI + PnP + FGO',
        # Thresholds relaxed for cross-view matching parity with ours_cascade.
        'min_inliers_fast': 15,
        'min_shape_conf': 0.20,
        'max_reproj_residual_px': 10.0,
        'roma_budget_rate': 0.70,
    },
    'ours_no_ipm': {
        'name': 'Ours (No IPM)',
        'retrieval': 'ultravpr',
        'retrieval_top_k': 5,
        'use_ipm': False,
        'alignment': 'coarse_to_fine',
        'matcher': 'aliked_lightglue',
        'geometry': 'homography_scale',
        'use_fgo': True,
        'use_lgi_filter': True,
        'description': 'UltraVPR + CoarseToFine + LightGlue + LGI + FGO',
    },
    'ours_no_fgo': {
        'name': 'Ours (No FGO)',
        'retrieval': 'ultravpr',
        'retrieval_top_k': 5,
        'use_ipm': True,
        'alignment': 'coarse_to_fine',
        'matcher': 'aliked_lightglue',
        'geometry': 'homography_scale',
        'use_fgo': False,
        'use_lgi_filter': True,
        'description': 'UltraVPR + IPM + CoarseToFine + LightGlue + LGI + Homography',
    },
    'ours_no_lgi': {
        'name': 'Ours (No LGI)',
        'retrieval': 'ultravpr',
        'retrieval_top_k': 5,
        'use_ipm': True,
        'alignment': 'coarse_to_fine',
        'matcher': 'aliked_lightglue',
        'geometry': 'homography_scale',
        'use_fgo': True,
        'use_lgi_filter': False,
        'description': 'UltraVPR + IPM + CoarseToFine + LightGlue + FGO (no LGI)',
    },
    'ours_minimal': {
        'name': 'Ours (Minimal)',
        'retrieval': 'ultravpr',
        'retrieval_top_k': 5,
        'use_ipm': False,
        'alignment': 'none',
        'matcher': 'aliked_lightglue',
        'geometry': 'homography_scale',
        'use_fgo': False,
        'use_lgi_filter': False,
        'description': 'UltraVPR + LightGlue + Homography (baseline ablation)',
    },
    'ours_fast_top1': {
        'name': 'Ours Fast Top1 (Speed)',
        'retrieval': 'ultravpr',
        'retrieval_top_k': 1,
        'retrieval_verify_top_k': False,
        'retrieval_input_size': (300, 400),
        'use_ipm': True,
        'alignment': 'none',
        'matcher': 'aliked_lightglue',
        'matcher_max_keypoints': 1024,
        'matcher_resize_max': 768,
        'geometry': 'homography_scale',
        'use_fgo': False,
        'use_lgi_filter': False,
        'description': 'UltraVPR Top1 + capped ALIKED-LightGlue + Homography',
    },
    'oracle_fast_lightglue': {
        'name': 'Oracle Fast LightGlue (Diagnostic)',
        'retrieval': 'oracle_reference',
        'use_ipm': True,
        'alignment': 'none',
        'matcher': 'aliked_lightglue',
        'matcher_max_keypoints': 1024,
        'matcher_resize_max': 768,
        'geometry': 'homography',
        'use_fgo': False,
        'use_lgi_filter': False,
        'description': 'Oracle reference + capped ALIKED-LightGlue + Homography',
    },
    'roma_every_frame_e2e': {
        'name': 'Deprecated Alias: RoMa Every Frame Wrapper E2E',
        'retrieval': 'ultravpr',
        'retrieval_top_k': 1,
        'retrieval_verify_top_k': False,
        'retrieval_input_size': (300, 400),
        'use_ipm': False,
        'alignment': 'none',
        'matcher': 'roma',
        'geometry': 'homography',
        'use_fgo': False,
        'use_lgi_filter': False,
        'method_family': 'roma_every_frame_wrapper',
        'comparison_tier': 'wrapper_e2e',
        'tile_source': 'retrieval_top1',
        'attitude_source': 'metadata_or_gt_proxy',
        'altitude_source': 'metadata_or_gt_proxy',
        'pose_head': 'hd4_homography_wrapper',
        'is_native': False,
        'is_wrapper': True,
        'is_hybrid': False,
        'uses_oracle': False,
        'native_baseline': False,
        'oracle_mode': False,
        'deprecated_alias': True,
        'canonical_pipeline': 'roma_every_frame_wrapper_e2e',
        'description': 'Deprecated alias of roma_every_frame_wrapper_e2e.',
    },
    'roma_every_frame_wrapper_e2e': {
        'name': 'RoMa Every Frame Wrapper E2E',
        'retrieval': 'ultravpr',
        'retrieval_top_k': 1,
        'retrieval_verify_top_k': False,
        'retrieval_input_size': (300, 400),
        'use_ipm': False,
        'alignment': 'none',
        'matcher': 'roma',
        'geometry': 'homography',
        'use_fgo': False,
        'use_lgi_filter': False,
        'method_family': 'roma_every_frame_wrapper',
        'comparison_tier': 'wrapper_e2e',
        'tile_source': 'retrieval_top1',
        'attitude_source': 'metadata_or_gt_proxy',
        'altitude_source': 'metadata_or_gt_proxy',
        'pose_head': 'hd4_homography_wrapper',
        'is_native': False,
        'is_wrapper': True,
        'is_hybrid': False,
        'uses_oracle': False,
        'native_baseline': False,
        'oracle_mode': False,
        'deprecated_alias': False,
        'canonical_pipeline': 'roma_every_frame_wrapper_e2e',
        'description': 'UltraVPR Top1 + RoMa every frame + HD4 homography wrapper; not native OrthoLoC.',
    },
    'roma_every_frame': {
        'name': 'RoMa Every Frame (Oracle Reference)',
        'retrieval': 'oracle_reference',
        'use_ipm': False,
        'alignment': 'none',
        'matcher': 'roma',
        'geometry': 'pnp',
        'use_fgo': False,
        'use_lgi_filter': False,
        'description': 'Oracle reference + RoMa + PnP/Homography',
    },
    # Backward-compatibility alias for old scripts (identical to roma_every_frame).
    'baseline_roma': {
        'name': 'RoMa Every Frame (Oracle Reference)',
        'retrieval': 'oracle_reference',
        'use_ipm': False,
        'alignment': 'none',
        'matcher': 'roma',
        'geometry': 'pnp',
        'use_fgo': False,
        'use_lgi_filter': False,
        'description': 'Deprecated alias of roma_every_frame',
    },
    'ours_cascade': {
        'name': 'Ours (Top-K + Selective RoMa Cascade)',
        'retrieval': 'ultravpr',
        'retrieval_top_k': 5,
        'use_ipm': True,
        'alignment': 'coarse_to_fine',
        'matcher': 'aliked_lightglue',
        'geometry': 'homography_scale',
        'use_fgo': True,
        'use_lgi_filter': True,
        'enable_roma_fallback': True,
        'roma_fallback_min_inliers': 24,
        'roma_fallback_min_confidence': 0.20,
        # Cascade thresholds: escalate to RoMa whenever LightGlue quality is low.
        # roma_budget_rate=1.0 means no budget cap — always invoke RoMa on hard frames.
        # min_inliers_fast=15: LightGlue needs >=15 inliers to skip RoMa escalation.
        # max_reproj_residual_px=10.0: cross-view matching has higher residuals than same-view;
        #   using 6.0 was incorrectly gating out high-inlier AVL frames.
        'min_inliers_fast': 15,
        'min_shape_conf': 0.20,
        'max_reproj_residual_px': 10.0,
        'final_pose_gate_enabled': True,
        'final_pose_max_geometry_anchor_offset_m': 400.0,
        'final_pose_min_shape_conf': 0.05,
        'final_pose_max_reproj_residual_px': 15.0,
        'roma_budget_rate': 1.0,
        'description': 'UltraVPR Top-K + LightGlue verifier + selective RoMa + FGO',
    },
    'ours_cascade_homography': {
        'name': 'Deprecated Alias: Ours UltraVPR Cascade 6-DoF E2E',
        'retrieval': 'ultravpr',
        'retrieval_top_k': 5,
        'use_ipm': True,
        'alignment': 'coarse_to_fine',
        'matcher': 'aliked_lightglue',
        'geometry': 'homography_scale',
        'use_fgo': True,
        'use_lgi_filter': True,
        'enable_roma_fallback': True,
        'roma_fallback_min_inliers': 24,
        'roma_fallback_min_confidence': 0.20,
        'min_inliers_fast': 15,
        'min_shape_conf': 0.20,
        'max_reproj_residual_px': 10.0,
        'final_pose_gate_enabled': True,
        'final_pose_max_geometry_anchor_offset_m': 400.0,
        'final_pose_min_shape_conf': 0.05,
        'final_pose_max_reproj_residual_px': 15.0,
        'roma_budget_rate': 1.0,
        'method_family': 'ours_cascade_6dof',
        'comparison_tier': 'strict_e2e',
        'tile_source': 'retrieval_topk_rerank',
        'attitude_source': 'metadata_or_gt_proxy',
        'altitude_source': 'metadata_or_gt_proxy',
        'pose_head': 'hd4_homography_scale_fgo',
        'is_native': False,
        'is_wrapper': False,
        'is_hybrid': False,
        'uses_oracle': False,
        'native_baseline': False,
        'oracle_mode': False,
        'deprecated_alias': True,
        'canonical_pipeline': 'ours_ultravpr_cascade_6dof_e2e',
        'description': 'Deprecated alias of ours_ultravpr_cascade_6dof_e2e.',
    },
    'ours_ultravpr_cascade_6dof_e2e': {
        'name': 'Ours UltraVPR Cascade 6-DoF E2E',
        'retrieval': 'ultravpr',
        'retrieval_top_k': 5,
        'use_ipm': True,
        'alignment': 'coarse_to_fine',
        'matcher': 'aliked_lightglue',
        'geometry': 'homography_scale',
        'use_fgo': True,
        'use_lgi_filter': True,
        'enable_roma_fallback': True,
        'roma_fallback_min_inliers': 24,
        'roma_fallback_min_confidence': 0.20,
        'min_inliers_fast': 15,
        'min_shape_conf': 0.20,
        'max_reproj_residual_px': 10.0,
        'final_pose_gate_enabled': True,
        'final_pose_max_geometry_anchor_offset_m': 400.0,
        'final_pose_min_shape_conf': 0.05,
        'final_pose_max_reproj_residual_px': 15.0,
        'roma_budget_rate': 1.0,
        'method_family': 'ours_cascade_6dof',
        'comparison_tier': 'strict_e2e',
        'tile_source': 'retrieval_topk_rerank',
        'attitude_source': 'metadata_or_gt_proxy',
        'altitude_source': 'metadata_or_gt_proxy',
        'pose_head': 'hd4_homography_scale_fgo',
        'is_native': False,
        'is_wrapper': False,
        'is_hybrid': False,
        'uses_oracle': False,
        'native_baseline': False,
        'oracle_mode': False,
        'deprecated_alias': False,
        'canonical_pipeline': 'ours_ultravpr_cascade_6dof_e2e',
        'description': 'UltraVPR Top-K + verifier + ALIKED/LightGlue + RoMa selective + LGI + HomographyScale + FGO.',
    },
    'retrieval_megaloc': {
        'name': 'MegaLoc Retrieval-Only (zero-shot)',
        'retrieval': 'megaloc',
        'retrieval_top_k': 5,
        'use_ipm': False,
        'alignment': 'none',
        'matcher': 'none',
        'geometry': 'none',
        'use_fgo': False,
        'use_lgi_filter': False,
        'enable_roma_fallback': False,
        'final_pose_gate_enabled': False,
        'method_family': 'retrieval_only_baseline',
        'comparison_tier': 'retrieval_only',
        'tile_source': 'retrieval_top1',
        'attitude_source': 'none',
        'altitude_source': 'none',
        'pose_head': 'none',
        'is_native': False,
        'is_wrapper': False,
        'is_hybrid': False,
        'uses_oracle': False,
        'native_baseline': True,
        'oracle_mode': False,
        'deprecated_alias': False,
        'canonical_pipeline': 'retrieval_megaloc',
        'description': 'MegaLoc (torch.hub gmberton/MegaLoc) zero-shot retrieval-only baseline; run with --retrieval-only.',
    },
    'ours_cascade_camp_homography': {
        'name': 'Deprecated Alias: Ours CAMP Cascade 6-DoF E2E',
        'retrieval': 'camp',
        'retrieval_top_k': 5,
        'use_ipm': True,
        'alignment': 'coarse_to_fine',
        'matcher': 'aliked_lightglue',
        'geometry': 'homography_scale',
        'use_fgo': True,
        'use_lgi_filter': True,
        'enable_roma_fallback': True,
        'roma_fallback_min_inliers': 24,
        'roma_fallback_min_confidence': 0.20,
        'min_inliers_fast': 15,
        'min_shape_conf': 0.20,
        'max_reproj_residual_px': 10.0,
        'final_pose_gate_enabled': True,
        'final_pose_max_geometry_anchor_offset_m': 400.0,
        'final_pose_min_shape_conf': 0.05,
        'final_pose_max_reproj_residual_px': 15.0,
        'roma_budget_rate': 1.0,
        'method_family': 'ours_cascade_6dof',
        'comparison_tier': 'strict_e2e',
        'tile_source': 'retrieval_topk_rerank',
        'attitude_source': 'metadata_or_gt_proxy',
        'altitude_source': 'metadata_or_gt_proxy',
        'pose_head': 'hd4_homography_scale_fgo',
        'is_native': False,
        'is_wrapper': False,
        'is_hybrid': False,
        'uses_oracle': False,
        'native_baseline': False,
        'oracle_mode': False,
        'deprecated_alias': True,
        'canonical_pipeline': 'ours_camp_cascade_6dof_e2e',
        'description': 'Deprecated alias of ours_camp_cascade_6dof_e2e.',
    },
    'ours_camp_cascade_6dof_e2e': {
        'name': 'Ours CAMP Cascade 6-DoF E2E',
        'retrieval': 'camp',
        'retrieval_top_k': 5,
        'use_ipm': True,
        'alignment': 'coarse_to_fine',
        'matcher': 'aliked_lightglue',
        'geometry': 'homography_scale',
        'use_fgo': True,
        'use_lgi_filter': True,
        'enable_roma_fallback': True,
        'roma_fallback_min_inliers': 24,
        'roma_fallback_min_confidence': 0.20,
        'min_inliers_fast': 15,
        'min_shape_conf': 0.20,
        'max_reproj_residual_px': 10.0,
        'final_pose_gate_enabled': True,
        'final_pose_max_geometry_anchor_offset_m': 400.0,
        'final_pose_min_shape_conf': 0.05,
        'final_pose_max_reproj_residual_px': 15.0,
        'roma_budget_rate': 1.0,
        'method_family': 'ours_cascade_6dof',
        'comparison_tier': 'strict_e2e',
        'tile_source': 'retrieval_topk_rerank',
        'attitude_source': 'metadata_or_gt_proxy',
        'altitude_source': 'metadata_or_gt_proxy',
        'pose_head': 'hd4_homography_scale_fgo',
        'is_native': False,
        'is_wrapper': False,
        'is_hybrid': False,
        'uses_oracle': False,
        'native_baseline': False,
        'oracle_mode': False,
        'deprecated_alias': False,
        'canonical_pipeline': 'ours_camp_cascade_6dof_e2e',
        'description': 'CAMP Top-K + verifier + ALIKED/LightGlue + RoMa selective + LGI + HomographyScale + FGO.',
    },
    'ours_dsm_pnp_6dof_e2e': {
        'name': 'Ours DSM-Aware PnP 6-DoF E2E (Experimental)',
        'retrieval': 'camp',
        'retrieval_top_k': 5,
        'retrieval_verify_top_k': True,
        'official_fail_closed_retrieval_verifier': True,
        'use_ipm': False,
        'alignment': 'none',
        'matcher': 'roma',
        'geometry': 'dsm_pnp',
        'use_fgo': False,
        'use_lgi_filter': False,
        'geometry_min_inliers_after_lgi': 8,
        'max_reproj_residual_px': 8.0,
        'final_pose_gate_enabled': True,
        'final_pose_max_geometry_anchor_offset_m': 400.0,
        'final_pose_min_shape_conf': 0.02,
        'final_pose_max_reproj_residual_px': 12.0,
        'method_family': 'ours_dsm_aware_6dof',
        'comparison_tier': 'experimental_dsm_6dof',
        'tile_source': 'retrieval_topk_render_verify_fail_closed',
        'attitude_source': 'metadata_or_gt_proxy',
        'altitude_source': 'dsm_pnp_camera_center',
        'pose_head': 'dsm_backed_2d3d_pnp',
        'is_native': False,
        'is_wrapper': False,
        'is_hybrid': False,
        'uses_oracle': False,
        'native_baseline': False,
        'oracle_mode': False,
        'deprecated_alias': False,
        'canonical_pipeline': 'ours_dsm_pnp_6dof_e2e',
        'description': 'Experimental DOP+DSM 2D-3D PnP head. Excluded from strict official rows until DSM and orientation metrics are audited.',
    },
    'camp_roma_homography_e2e': {
        'name': 'Deprecated Alias: CAMP+RoMa+Homography Wrapper',
        'retrieval': 'camp',
        'retrieval_top_k': 5,
        'use_ipm': False,
        'alignment': 'none',
        'matcher': 'roma',
        'geometry': 'homography',
        'use_fgo': False,
        'use_lgi_filter': False,
        'method_family': 'camp_roma_homography_wrapper',
        'comparison_tier': 'wrapper_e2e',
        'tile_source': 'retrieval_topk',
        'attitude_source': 'metadata_or_gt_proxy',
        'altitude_source': 'metadata_or_gt_proxy',
        'pose_head': 'hd4_homography_wrapper',
        'is_native': False,
        'is_wrapper': True,
        'is_hybrid': False,
        'uses_oracle': False,
        'native_baseline': False,
        'oracle_mode': False,
        'deprecated_alias': True,
        'canonical_pipeline': 'camp_roma_homography_wrapper_e2e',
        'description': 'Deprecated alias of camp_roma_homography_wrapper_e2e.',
    },
    'camp_roma_homography_wrapper_e2e': {
        'name': 'CAMP+RoMa+Homography Wrapper E2E',
        'retrieval': 'camp',
        'retrieval_top_k': 5,
        'use_ipm': False,
        'alignment': 'none',
        'matcher': 'roma',
        'geometry': 'homography',
        'use_fgo': False,
        'use_lgi_filter': False,
        'method_family': 'camp_roma_homography_wrapper',
        'comparison_tier': 'wrapper_e2e',
        'tile_source': 'retrieval_topk',
        'attitude_source': 'metadata_or_gt_proxy',
        'altitude_source': 'metadata_or_gt_proxy',
        'pose_head': 'hd4_homography_wrapper',
        'is_native': False,
        'is_wrapper': True,
        'is_hybrid': False,
        'uses_oracle': False,
        'native_baseline': False,
        'oracle_mode': False,
        'deprecated_alias': False,
        'canonical_pipeline': 'camp_roma_homography_wrapper_e2e',
        'description': 'CAMP Top-K + RoMa + HD4 homography wrapper; not native AnyVisLoc.',
    },
    'anyvisloc_e2e': {
        'name': 'Deprecated Alias: CAMP+RoMa+Homography Wrapper',
        'retrieval': 'camp',
        'retrieval_top_k': 5,
        'use_ipm': False,
        'alignment': 'none',
        'matcher': 'roma',
        'geometry': 'homography',
        'use_fgo': False,
        'use_lgi_filter': False,
        'method_family': 'camp_roma_homography_wrapper',
        'native_baseline': False,
        'oracle_mode': False,
        'deprecated_alias': True,
        'comparison_tier': 'wrapper_e2e',
        'tile_source': 'retrieval_topk',
        'attitude_source': 'metadata_or_gt_proxy',
        'altitude_source': 'metadata_or_gt_proxy',
        'pose_head': 'hd4_homography_wrapper',
        'is_native': False,
        'is_wrapper': True,
        'is_hybrid': False,
        'uses_oracle': False,
        'canonical_pipeline': 'camp_roma_homography_wrapper_e2e',
        'description': 'Deprecated alias of camp_roma_homography_e2e; not native AnyVisLoc.',
    },
    'ultravpr_roma_homography_e2e': {
        'name': 'Deprecated Alias: UltraVPR+RoMa+Homography Wrapper',
        'retrieval': 'ultravpr',
        'retrieval_top_k': 5,
        'use_ipm': False,
        'alignment': 'none',
        'matcher': 'roma',
        'geometry': 'homography',
        'use_fgo': False,
        'use_lgi_filter': False,
        'method_family': 'ultravpr_roma_homography_wrapper',
        'comparison_tier': 'wrapper_e2e',
        'tile_source': 'retrieval_topk',
        'attitude_source': 'metadata_or_gt_proxy',
        'altitude_source': 'metadata_or_gt_proxy',
        'pose_head': 'hd4_homography_wrapper',
        'is_native': False,
        'is_wrapper': True,
        'is_hybrid': False,
        'uses_oracle': False,
        'native_baseline': False,
        'oracle_mode': False,
        'deprecated_alias': True,
        'canonical_pipeline': 'ultravpr_roma_homography_wrapper_e2e',
        'description': 'Deprecated alias of ultravpr_roma_homography_wrapper_e2e.',
    },
    'ultravpr_roma_homography_wrapper_e2e': {
        'name': 'UltraVPR+RoMa+Homography Wrapper E2E',
        'retrieval': 'ultravpr',
        'retrieval_top_k': 5,
        'use_ipm': False,
        'alignment': 'none',
        'matcher': 'roma',
        'geometry': 'homography',
        'use_fgo': False,
        'use_lgi_filter': False,
        'method_family': 'ultravpr_roma_homography_wrapper',
        'comparison_tier': 'wrapper_e2e',
        'tile_source': 'retrieval_topk',
        'attitude_source': 'metadata_or_gt_proxy',
        'altitude_source': 'metadata_or_gt_proxy',
        'pose_head': 'hd4_homography_wrapper',
        'is_native': False,
        'is_wrapper': True,
        'is_hybrid': False,
        'uses_oracle': False,
        'native_baseline': False,
        'oracle_mode': False,
        'deprecated_alias': False,
        'canonical_pipeline': 'ultravpr_roma_homography_wrapper_e2e',
        'description': 'UltraVPR Top-K + RoMa + HD4 homography wrapper; not native OrthoLoC.',
    },
    'ortholoc_e2e': {
        'name': 'Deprecated Alias: UltraVPR+RoMa+Homography Wrapper',
        'retrieval': 'ultravpr',
        'retrieval_top_k': 5,
        'use_ipm': False,
        'alignment': 'none',
        'matcher': 'roma',
        'geometry': 'homography',
        'use_fgo': False,
        'use_lgi_filter': False,
        'method_family': 'ultravpr_roma_homography_wrapper',
        'native_baseline': False,
        'oracle_mode': False,
        'deprecated_alias': True,
        'comparison_tier': 'wrapper_e2e',
        'tile_source': 'retrieval_topk',
        'attitude_source': 'metadata_or_gt_proxy',
        'altitude_source': 'metadata_or_gt_proxy',
        'pose_head': 'hd4_homography_wrapper',
        'is_native': False,
        'is_wrapper': True,
        'is_hybrid': False,
        'uses_oracle': False,
        'canonical_pipeline': 'ultravpr_roma_homography_wrapper_e2e',
        'description': 'Deprecated alias of ultravpr_roma_homography_e2e; not native OrthoLoC.',
    },
    'native_anyvisloc_e2e': {
        'name': 'Native AnyVisLoc E2E (CAMP+Yaw/Pitch+RoMa+DSM-P3P)',
        'native_backend': 'anyvisloc',
        'retrieval': 'camp',
        'retrieval_top_k': 5,
        'use_ipm': False,
        'alignment': 'native_anyvisloc',
        'matcher': 'roma',
        'geometry': 'dsm_p3p',
        'use_fgo': False,
        'use_lgi_filter': False,
        'method_family': 'native_anyvisloc',
        'comparison_tier': 'native_reproduction',
        'tile_source': 'native_camp_retrieval',
        'attitude_source': 'simulated_telemetry_from_gt',
        'altitude_source': 'simulated_telemetry_from_gt',
        'pose_head': 'native_anyvisloc_dsm_p3p',
        'is_native': True,
        'is_wrapper': False,
        'is_hybrid': False,
        'uses_oracle': False,
        'native_baseline': True,
        'oracle_mode': False,
        'deprecated_alias': False,
        'canonical_pipeline': 'native_anyvisloc_e2e',
        'supported_datasets': ['avl', 'vpair', 'uav_visloc'],
        'description': 'Port of anyvisloc_kaggle_v2: CAMP retrieval, yaw/pitch prior, rotated map, RoMa, DSM-backed solvePnPRansac(P3P), Top-N rerank by PnP inliers. VPair/UAV-VisLoc: adapted with flat DSM.',
    },
    'native_anyvisloc_oracle_tile': {
        'name': 'Native AnyVisLoc Oracle Tile (Diagnostic)',
        'native_backend': 'anyvisloc_oracle_tile',
        'retrieval': 'oracle_reference',
        'retrieval_top_k': 1,
        'use_ipm': False,
        'alignment': 'native_anyvisloc',
        'matcher': 'roma',
        'geometry': 'dsm_p3p',
        'use_fgo': False,
        'use_lgi_filter': False,
        'method_family': 'native_anyvisloc',
        'comparison_tier': 'native_oracle_tile',
        'tile_source': 'oracle_tile',
        'attitude_source': 'simulated_telemetry_from_gt',
        'altitude_source': 'simulated_telemetry_from_gt',
        'pose_head': 'native_anyvisloc_dsm_p3p',
        'is_native': True,
        'is_wrapper': False,
        'is_hybrid': False,
        'uses_oracle': True,
        'oracle_modes': ['oracle_tile'],
        'native_baseline': True,
        'oracle_mode': True,
        'native_oracle_tile': True,
        'deprecated_alias': False,
        'canonical_pipeline': 'native_anyvisloc_oracle_tile',
        'supported_datasets': ['avl'],
        'description': 'Diagnostic native AnyVisLoc backend with a GT-position reference tile supplied; excluded from strict E2E.',
    },
    'native_anyvisloc_oracle_tile_pose_prior': {
        'name': 'Native AnyVisLoc Oracle Tile + Pose Prior (Diagnostic)',
        'native_backend': 'anyvisloc_oracle_tile_pose_prior',
        'retrieval': 'oracle_reference',
        'retrieval_top_k': 1,
        'use_ipm': False,
        'alignment': 'native_anyvisloc',
        'matcher': 'roma',
        'geometry': 'dsm_p3p',
        'use_fgo': False,
        'use_lgi_filter': False,
        'method_family': 'native_anyvisloc',
        'comparison_tier': 'native_oracle_pose_prior',
        'tile_source': 'oracle_tile',
        'attitude_source': 'oracle_pose_prior',
        'altitude_source': 'oracle_altitude',
        'pose_head': 'native_anyvisloc_dsm_p3p',
        'is_native': True,
        'is_wrapper': False,
        'is_hybrid': False,
        'uses_oracle': True,
        'oracle_modes': ['oracle_tile', 'oracle_pose_prior', 'oracle_altitude'],
        'native_baseline': True,
        'oracle_mode': True,
        'native_use_exact_gt_prior': True,
        'native_oracle_tile': True,
        'deprecated_alias': False,
        'canonical_pipeline': 'native_anyvisloc_oracle_tile_pose_prior',
        'supported_datasets': ['avl'],
        'description': 'Strong diagnostic upper bound with GT/reference tile and exact GT yaw/pitch/roll/altitude; excluded from strict E2E.',
    },
    'anyvisloc_frontend_shared_6dof_e2e': {
        'name': 'AnyVisLoc Frontend + Shared 6-DoF Head E2E',
        'native_backend': 'anyvisloc_shared_6dof',
        'retrieval': 'camp',
        'retrieval_top_k': 5,
        'use_ipm': False,
        'alignment': 'native_anyvisloc',
        'matcher': 'roma',
        'geometry': 'shared_6dof_head',
        'use_fgo': True,
        'use_lgi_filter': False,
        'method_family': 'anyvisloc_frontend_shared_6dof',
        'comparison_tier': 'hybrid_fair_6dof',
        'tile_source': 'native_camp_retrieval',
        'attitude_source': 'metadata_or_gt_proxy',
        'altitude_source': 'metadata_or_gt_proxy',
        'pose_head': 'hd4_shared_6dof',
        'is_native': False,
        'is_wrapper': False,
        'is_hybrid': True,
        'uses_oracle': False,
        'native_baseline': False,
        'oracle_mode': False,
        'deprecated_alias': False,
        'canonical_pipeline': 'anyvisloc_frontend_shared_6dof_e2e',
        'supported_datasets': ['avl'],
        'description': 'Hybrid comparison: AnyVisLoc frontend normalized into the shared HD4 6-DoF pose head; fail-closed until adapter is wired.',
    },
    'anyvisloc_oracle_tile_shared_6dof': {
        'name': 'AnyVisLoc Oracle Tile + Shared 6-DoF Head (Diagnostic)',
        'native_backend': 'anyvisloc_oracle_tile_shared_6dof',
        'retrieval': 'oracle_reference',
        'retrieval_top_k': 1,
        'use_ipm': False,
        'alignment': 'native_anyvisloc',
        'matcher': 'roma',
        'geometry': 'shared_6dof_head',
        'use_fgo': True,
        'use_lgi_filter': False,
        'method_family': 'anyvisloc_frontend_shared_6dof',
        'comparison_tier': 'hybrid_fair_6dof',
        'tile_source': 'oracle_tile',
        'attitude_source': 'metadata_or_gt_proxy',
        'altitude_source': 'metadata_or_gt_proxy',
        'pose_head': 'hd4_shared_6dof',
        'is_native': False,
        'is_wrapper': False,
        'is_hybrid': True,
        'uses_oracle': True,
        'oracle_modes': ['oracle_tile'],
        'native_baseline': False,
        'oracle_mode': True,
        'deprecated_alias': False,
        'canonical_pipeline': 'anyvisloc_oracle_tile_shared_6dof',
        'supported_datasets': ['avl'],
        'description': 'Hybrid oracle-tile diagnostic using AnyVisLoc frontend and shared HD4 6-DoF head; not native AnyVisLoc.',
    },
    'ortholoc_2d3d_e2e': {
        'name': 'Native OrthoLoC 2D-3D E2E (Retrieval First)',
        'native_backend': 'ortholoc_2d3d',
        'retrieval': 'ultravpr',
        'retrieval_top_k': 5,
        'use_ipm': False,
        'alignment': 'ortholoc_2d3d',
        'matcher': 'roma',
        'geometry': 'ortholoc_2d3d_pnp',
        'use_fgo': False,
        'use_lgi_filter': False,
        'method_family': 'ortholoc_2d3d',
        'comparison_tier': 'native_reproduction',
        'tile_source': 'native_or_retrieval_first',
        'attitude_source': 'metadata_or_gt_proxy',
        'altitude_source': 'metadata_or_gt_proxy',
        'pose_head': 'native_ortholoc_2d3d_pnp',
        'is_native': True,
        'is_wrapper': False,
        'is_hybrid': False,
        'uses_oracle': False,
        'native_baseline': True,
        'oracle_mode': False,
        'deprecated_alias': False,
        'canonical_pipeline': 'native_ortholoc_2d3d_e2e',
        'description': 'OrthoLoC reference matcher/correspondence/PnP path after retrieval; no GT tile oracle.',
    },
    'native_ortholoc_2d3d_e2e': {
        'name': 'Native OrthoLoC 2D-3D E2E (Retrieval First)',
        'native_backend': 'ortholoc_2d3d',
        'retrieval': 'ultravpr',
        'retrieval_top_k': 5,
        'use_ipm': False,
        'alignment': 'ortholoc_2d3d',
        'matcher': 'roma',
        'geometry': 'ortholoc_2d3d_pnp',
        'use_fgo': False,
        'use_lgi_filter': False,
        'method_family': 'ortholoc_2d3d',
        'comparison_tier': 'native_reproduction',
        'tile_source': 'native_or_retrieval_first',
        'attitude_source': 'metadata_or_gt_proxy',
        'altitude_source': 'metadata_or_gt_proxy',
        'pose_head': 'native_ortholoc_2d3d_pnp',
        'is_native': True,
        'is_wrapper': False,
        'is_hybrid': False,
        'uses_oracle': False,
        'native_baseline': True,
        'oracle_mode': False,
        'deprecated_alias': False,
        'canonical_pipeline': 'native_ortholoc_2d3d_e2e',
        'supported_datasets': ['avl'],
        'description': 'OrthoLoC reference: DOP/DSM crop -> RoMa match -> 2D-3D lift -> PnP via poselib. AVL only (requires DOP + real DSM).',
    },
    'ortholoc_2d3d_oracle': {
        'name': 'Deprecated Alias: Native OrthoLoC Oracle Tile',
        'native_backend': 'ortholoc_2d3d',
        'retrieval': 'oracle_reference',
        'retrieval_top_k': 1,
        'use_ipm': False,
        'alignment': 'ortholoc_2d3d',
        'matcher': 'roma',
        'geometry': 'ortholoc_2d3d_pnp',
        'use_fgo': False,
        'use_lgi_filter': False,
        'method_family': 'ortholoc_2d3d',
        'comparison_tier': 'native_oracle_tile',
        'tile_source': 'oracle_tile',
        'attitude_source': 'metadata_or_gt_proxy',
        'altitude_source': 'metadata_or_gt_proxy',
        'pose_head': 'native_ortholoc_2d3d_pnp',
        'is_native': True,
        'is_wrapper': False,
        'is_hybrid': False,
        'uses_oracle': True,
        'oracle_modes': ['oracle_tile'],
        'native_baseline': True,
        'oracle_mode': True,
        'deprecated_alias': True,
        'canonical_pipeline': 'native_ortholoc_2d3d_oracle_tile',
        'description': 'Deprecated alias of native_ortholoc_2d3d_oracle_tile.',
    },
    'native_ortholoc_2d3d_oracle_tile': {
        'name': 'Native OrthoLoC 2D-3D Oracle Tile (Diagnostic)',
        'native_backend': 'ortholoc_2d3d',
        'retrieval': 'oracle_reference',
        'retrieval_top_k': 1,
        'use_ipm': False,
        'alignment': 'ortholoc_2d3d',
        'matcher': 'roma',
        'geometry': 'ortholoc_2d3d_pnp',
        'use_fgo': False,
        'use_lgi_filter': False,
        'method_family': 'ortholoc_2d3d',
        'comparison_tier': 'native_oracle_tile',
        'tile_source': 'oracle_tile',
        'attitude_source': 'metadata_or_gt_proxy',
        'altitude_source': 'metadata_or_gt_proxy',
        'pose_head': 'native_ortholoc_2d3d_pnp',
        'is_native': True,
        'is_wrapper': False,
        'is_hybrid': False,
        'uses_oracle': True,
        'oracle_modes': ['oracle_tile'],
        'native_baseline': True,
        'oracle_mode': True,
        'deprecated_alias': False,
        'canonical_pipeline': 'native_ortholoc_2d3d_oracle_tile',
        'description': 'Diagnostic upper-bound using dataset-paired reference; fail-closed until DOP/DSM dataset extraction is wired.',
    },
    'native_ortholoc_2d3d_oracle_tile_pose_prior': {
        'name': 'Native OrthoLoC 2D-3D Oracle Tile + Pose Prior (Diagnostic)',
        'native_backend': 'ortholoc_2d3d',
        'retrieval': 'oracle_reference',
        'retrieval_top_k': 1,
        'use_ipm': False,
        'alignment': 'ortholoc_2d3d',
        'matcher': 'roma',
        'geometry': 'ortholoc_2d3d_pnp',
        'use_fgo': False,
        'use_lgi_filter': False,
        'method_family': 'ortholoc_2d3d',
        'comparison_tier': 'native_oracle_pose_prior',
        'tile_source': 'oracle_tile',
        'attitude_source': 'oracle_pose_prior',
        'altitude_source': 'oracle_altitude',
        'pose_head': 'native_ortholoc_2d3d_pnp',
        'is_native': True,
        'is_wrapper': False,
        'is_hybrid': False,
        'uses_oracle': True,
        'oracle_modes': ['oracle_tile', 'oracle_pose_prior', 'oracle_altitude'],
        'native_baseline': True,
        'oracle_mode': True,
        'native_use_exact_gt_prior': True,
        'deprecated_alias': False,
        'canonical_pipeline': 'native_ortholoc_2d3d_oracle_tile_pose_prior',
        'description': 'Strong diagnostic upper bound with GT/reference tile and exact GT pose prior; fail-closed until OrthoLoC wiring is complete.',
    },
    'ortholoc_frontend_shared_6dof_e2e': {
        'name': 'OrthoLoC Frontend + Shared 6-DoF Head E2E',
        'native_backend': 'ortholoc_shared_6dof',
        'retrieval': 'ultravpr',
        'retrieval_top_k': 5,
        'use_ipm': False,
        'alignment': 'ortholoc_2d3d',
        'matcher': 'roma',
        'geometry': 'shared_6dof_head',
        'use_fgo': True,
        'use_lgi_filter': False,
        'method_family': 'ortholoc_frontend_shared_6dof',
        'comparison_tier': 'hybrid_fair_6dof',
        'tile_source': 'native_or_retrieval_first',
        'attitude_source': 'metadata_or_gt_proxy',
        'altitude_source': 'metadata_or_gt_proxy',
        'pose_head': 'hd4_shared_6dof',
        'is_native': False,
        'is_wrapper': False,
        'is_hybrid': True,
        'uses_oracle': False,
        'native_baseline': False,
        'oracle_mode': False,
        'deprecated_alias': False,
        'canonical_pipeline': 'ortholoc_frontend_shared_6dof_e2e',
        'description': 'Hybrid comparison: OrthoLoC frontend normalized into the shared HD4 6-DoF pose head; fail-closed until adapter is wired.',
    },
    'ortholoc_oracle_tile_shared_6dof': {
        'name': 'OrthoLoC Oracle Tile + Shared 6-DoF Head (Diagnostic)',
        'native_backend': 'ortholoc_oracle_tile_shared_6dof',
        'retrieval': 'oracle_reference',
        'retrieval_top_k': 1,
        'use_ipm': False,
        'alignment': 'ortholoc_2d3d',
        'matcher': 'roma',
        'geometry': 'shared_6dof_head',
        'use_fgo': True,
        'use_lgi_filter': False,
        'method_family': 'ortholoc_frontend_shared_6dof',
        'comparison_tier': 'hybrid_fair_6dof',
        'tile_source': 'oracle_tile',
        'attitude_source': 'metadata_or_gt_proxy',
        'altitude_source': 'metadata_or_gt_proxy',
        'pose_head': 'hd4_shared_6dof',
        'is_native': False,
        'is_wrapper': False,
        'is_hybrid': True,
        'uses_oracle': True,
        'oracle_modes': ['oracle_tile'],
        'native_baseline': False,
        'oracle_mode': True,
        'deprecated_alias': False,
        'canonical_pipeline': 'ortholoc_oracle_tile_shared_6dof',
        'description': 'Hybrid oracle-tile diagnostic using OrthoLoC frontend and shared HD4 6-DoF head; not native OrthoLoC.',
    },
}


def resolve_pipeline_metadata(pipeline_name: str) -> Dict[str, Any]:
    """Return canonical labels used by manifests, summaries, and comparison tables."""
    if pipeline_name not in PIPELINE_CONFIGS:
        raise ValueError(f"Unknown pipeline: {pipeline_name}")
    cfg = PIPELINE_CONFIGS[pipeline_name]
    retrieval = str(cfg.get("retrieval", ""))
    oracle_modes = list(cfg.get("oracle_modes", []))
    if bool(cfg.get("oracle_mode", False)) and not oracle_modes:
        if retrieval == "oracle_reference":
            oracle_modes.append("oracle_tile")
        else:
            oracle_modes.append("oracle")
    uses_oracle = bool(
        cfg.get("uses_oracle", False)
        or cfg.get("oracle_mode", False)
        or retrieval == "oracle_reference"
        or oracle_modes
    )
    metadata = {
        "pipeline": pipeline_name,
        "canonical_pipeline": str(cfg.get("canonical_pipeline", pipeline_name)),
        "method_family": str(cfg.get("method_family", pipeline_name)),
        "native_baseline": bool(cfg.get("native_baseline", False)),
        "oracle_mode": bool(cfg.get("oracle_mode", retrieval == "oracle_reference")),
        "deprecated_alias": bool(cfg.get("deprecated_alias", False)),
        "comparison_tier": str(
            cfg.get(
                "comparison_tier",
                "native_oracle_tile" if retrieval == "oracle_reference" else "strict_e2e",
            )
        ),
        "tile_source": str(
            cfg.get("tile_source", "oracle_tile" if retrieval == "oracle_reference" else "retrieval")
        ),
        "attitude_source": str(cfg.get("attitude_source", "metadata_or_gt_proxy")),
        "altitude_source": str(cfg.get("altitude_source", "metadata_or_gt_proxy")),
        "pose_head": str(cfg.get("pose_head", cfg.get("geometry", ""))),
        "is_native": bool(cfg.get("is_native", cfg.get("native_baseline", False))),
        "is_wrapper": bool(cfg.get("is_wrapper", False)),
        "is_hybrid": bool(cfg.get("is_hybrid", False)),
        "uses_oracle": uses_oracle,
        "oracle_modes": oracle_modes,
    }
    return metadata

# Full ablation matrix: proposed variants + E2E RoMa baseline (oracle-style baselines listed separately).
ABLATION_PIPELINES_E2E = [
    'ours',
    'ours_fast_top1',
    'ours_no_ipm',
    'ours_no_fgo',
    'ours_no_lgi',
    'ours_minimal',
    'ours_cascade',
    'ours_ultravpr_cascade_6dof_e2e',
    'ours_camp_cascade_6dof_e2e',
    'roma_every_frame_wrapper_e2e',
    'camp_roma_homography_wrapper_e2e',
    'ultravpr_roma_homography_wrapper_e2e',
    'native_anyvisloc_e2e',
]

# Proposal-aligned E2E sweep (Report/main.tex): primary ours vs ours_cascade, plus
# speed (fast top-1) and RoMa-every-frame under real retrieval. Skips module dropouts.
ABLATION_PIPELINES_E2E_CORE = [
    'ours',
    'ours_fast_top1',
    'ours_cascade',
    'ours_ultravpr_cascade_6dof_e2e',
    'ours_camp_cascade_6dof_e2e',
    'roma_every_frame_wrapper_e2e',
    'camp_roma_homography_wrapper_e2e',
    'ultravpr_roma_homography_wrapper_e2e',
    'native_anyvisloc_e2e',
]

# Upper-bound / diagnostic baselines (require eval_mode=oracle_reference).
# NOTE: baseline_roma is a backward-compatible alias of roma_every_frame (same config);
# do not add both to this list or you duplicate work.
ABLATION_PIPELINES_ORACLE = [
    'roma_every_frame',
    'oracle_fast_lightglue',
    'native_anyvisloc_oracle_tile',
    'native_anyvisloc_oracle_tile_pose_prior',
    'native_ortholoc_2d3d_oracle_tile',
    'native_ortholoc_2d3d_oracle_tile_pose_prior',
]

# Dataset configurations
DATASET_CONFIGS = {
    'vpair': {
        'data_root': os.environ.get('VPAIR_DATA', './data/VPAir'),
        'adapter': 'vpair',
        # Measured reference GSD (three independent estimates ~0.42 m/px);
        # 0.3 under-scaled every native-path offset by ~1.4x.
        'gsd': 0.42,
    },
    'avl': {
        'data_root': os.environ.get('AVL_DATA', './data/AVLData'),
        'adapter': 'avl',
        'gsd': 0.3,
        'kwargs': {'region': 'QZ_Town'},
    },
    'avl24': {
        # AnyVisLoc 2026 public release (24 scenes, sanitized scene-local NPZ).
        # One scene per run; the launcher loops scenes.
        'data_root': os.environ.get('AVL24_DATA', './data/AnyVisLoc'),
        'adapter': 'avl24',
        # Construction-time fallback only. Scene map resolution spans
        # 0.020-0.320 m/px across the release, so the solver reads the real
        # value per frame from the adapter (frame['gsd'] wins in solve()).
        'gsd': 0.1,
        'kwargs': {'scene': 'Scene_01'},
    },
    'uav_visloc': {
        'data_root': os.environ.get('VISLOC_DATA', './data/UAV_VisLoc_dataset'),
        'adapter': 'uav_visloc',
        'gsd': 0.3,
    },
}

# Retrieval Recall@1 threshold per dataset.
# UAV-VisLoc tiles span ~150-300m at GSD 0.3, so a 20m threshold is too tight;
# the "correct" tile center can easily be 50-150m from the query GT.
DATASET_RETRIEVAL_THRESHOLD_M = {
    'vpair': 20.0,
    'avl': 20.0,
    # AVL24 sizes its corpus tiles in metres (default 256 m), so the correct
    # tile centre sits up to a half-tile from the query GT.
    'avl24': 150.0,
    'uav_visloc': 150.0,
}

SUPPORTED_ADVERSARIAL_MODES = {"none", "wrong_reference"}
SUPPORTED_ADVERSARIAL_CONDITIONS = {"clean", "wrong_near", "wrong_far", "visually_similar_wrong", "wrong_semihard"}
SUPPORTED_ADVERSARIAL_CANDIDATE_POLICIES = {"force_rank_k", "nearest_wrong", "far_wrong", "visually_similar_wrong", "wrong_semihard", "cross_flight"}
FORGE_SUPPORTED_DATASETS = {"avl", "vpair", "uav_visloc", "avl24"}
# Reference footprint half-extents (east, north) in metres. A forced reference
# whose footprint still covers the query is NOT a wrong-reference fault: the
# geometry head can legitimately localize from it, so accepting it is correct
# behaviour and counting it as a fault inflates the measured fault rate.
# VPAir reference views: 800x600 px at 0.42 m/px. Tiled datasets: 2048 px tiles.
# Values are the recoverability bound minus FORGE_FOOTPRINT_SLACK_M, measured
# per dataset rather than assumed. AVL tiles a 0.061 m/px orthomosaic at
# 2048 px / 1024 px step (125 m tiles), NOT the 614 m assumed earlier, so the
# old 307 m half-extent left 83% of AVL frames with no eligible wrong reference
# and collapsed the three injection conditions onto the same candidate.
# Reference footprint half-extents (east, north), measured from the rasters the
# pipeline actually tiles. UAV-VisLoc rasters are stored with square pixels in
# DEGREES, so the ground pixel is anisotropic (0.23-0.26 m/px east against
# ~0.30 m/px north) and its tiles are not square.
FORGE_REF_HALF_M = {
    "vpair": (168.0, 126.0),
    "avl": (62.46, 62.46),
    "uav_visloc": (254.0, 305.8),
}
# Query footprint half-extent. AVL publishes intrinsics so this is computed per
# frame from altitude; VPAir queries are the same views as its references;
# UAV-VisLoc publishes none, so an empirical allowance is used.
AVL_QUERY_HALF_PER_M_ALT = 2000.0 / (3.6 / 7.81 * 4000.0)   # 1.0847 m per m altitude
FORGE_QUERY_HALF_M = {
    "vpair": (168.0, 126.0),
    "uav_visloc": (101.0, 101.0),
}
FORGE_FOOTPRINT_SLACK_M = 15.0
# Metres per degree of the AnyVisLoc 2026 synthetic frame (linear, both axes).
AVL24_M_PER_DEG = 111320.0
# Optional per-map reference half-extents (east, north) in metres, keyed by
# the map prefix of the reference name ("02/..." -> "02"). Empty unless the
# config sets forge_ref_half_m_per_map, so the cells of record keep the
# per-dataset constant above. UAV-VisLoc tiles differ by flight (233-278 m
# east), and one constant mislabels a held-out flight.
FORGE_REF_HALF_M_PER_MAP: Dict[str, Tuple[float, float]] = {}
# Optional per-frame query footprint bound (metres, both axes), keyed by
# image name, loaded from forge_query_half_m_table (CSV with columns
# image_name, status, query_half_bound_m). A frame that is missing from a
# loaded table, or whose status is not OK, has an indeterminate footprint
# and can carry no wrong-reference fault. Empty unless configured.
FORGE_QUERY_HALF_M_TABLE: Dict[str, float] = {}
FORGE_QUERY_HALF_M_TABLE_ACTIVE = [False]


def forge_query_half_m(dataset_name: str, gt_pose: Dict[str, Any]) -> Tuple[float, float]:
    """Query footprint half-extents (east, north) for one frame, in metres."""
    if dataset_name == "avl":
        alt = _safe_float((gt_pose or {}).get("alt", np.nan), default=float("nan"))
        if not np.isfinite(alt) or alt <= 0:
            alt = 119.0                     # dataset median, used only if altitude is missing
        h = AVL_QUERY_HALF_PER_M_ALT * float(alt)
        return h, h
    return FORGE_QUERY_HALF_M.get(dataset_name, (0.0, 0.0))


def _forge_candidate_excludes_gt(
    candidate: Dict[str, Any],
    dataset_name: str,
    dist_m: float,
    fallback_threshold_m: float,
) -> bool:
    """True when the query shares no ground with this candidate's reference.

    Only such a reference is a genuine wrong-reference fault: one that still
    overlaps the query is one the geometry head can legitimately localize from,
    so forcing it measures absorption rather than fault rejection. Limits are
    per axis, and the query side is per frame wherever intrinsics exist.
    """
    if "query_box_latlon" in candidate:
        # Box test for datasets that publish the camera's actual ground
        # footprint (AnyVisLoc 2026, oblique views). The footprint is not
        # centred on the camera, so a symmetric half-extent would be wrong.
        box = candidate.get("query_box_latlon")
        if box is None:
            return False                      # unbounded footprint: no fault possible
        anchor = candidate.get("anchor", {}) or {}
        lat = _safe_float(anchor.get("lat", np.nan), default=float("nan"))
        lon = _safe_float(anchor.get("lon", np.nan), default=float("nan"))
        if not (np.isfinite(lat) and np.isfinite(lon)):
            return False
        deg = 1.0 / AVL24_M_PER_DEG
        rh = _safe_float(candidate.get("ref_half_m", np.nan), default=float("nan")) * deg
        sl = FORGE_FOOTPRINT_SLACK_M * deg
        lat_min, lat_max, lon_min, lon_max = (float(v) for v in box)
        return bool(lat - rh > lat_max + sl or lat + rh < lat_min - sl
                    or lon - rh > lon_max + sl or lon + rh < lon_min - sl)
    de = _safe_float(candidate.get("offset_e_m", np.nan), default=float("nan"))
    dn = _safe_float(candidate.get("offset_n_m", np.nan), default=float("nan"))
    if not (np.isfinite(de) and np.isfinite(dn)):
        return bool(dist_m > fallback_threshold_m)
    rh_e, rh_n = FORGE_REF_HALF_M.get(dataset_name, (fallback_threshold_m, fallback_threshold_m))
    _ref = str(candidate.get("reference_name", "") or "")
    _map = _ref.split("/")[0] if "/" in _ref else ""
    if _map and _map in FORGE_REF_HALF_M_PER_MAP:
        rh_e, rh_n = FORGE_REF_HALF_M_PER_MAP[_map]
    try:
        _qraw = (float(candidate.get("query_half_e_m", 0.0)), float(candidate.get("query_half_n_m", 0.0)))
    except (TypeError, ValueError):
        _qraw = (0.0, 0.0)
    if any(np.isinf(v) for v in _qraw):
        return False                          # indeterminate footprint: no fault possible
    qh_e = _safe_float(candidate.get("query_half_e_m", 0.0), default=0.0)
    qh_n = _safe_float(candidate.get("query_half_n_m", 0.0), default=0.0)
    return bool(de > rh_e + qh_e + FORGE_FOOTPRINT_SLACK_M
                or dn > rh_n + qh_n + FORGE_FOOTPRINT_SLACK_M)


ADVERSARIAL_POLICY_BY_CONDITION = {
    "wrong_near": "nearest_wrong",
    "wrong_far": "far_wrong",
    "visually_similar_wrong": "visually_similar_wrong",
    "wrong_semihard": "visually_similar_wrong",
}
ADVERSARIAL_POLICY_COUNTER_KEYS = {
    "force_rank_k": "num_adversarial_policy_force_rank_k",
    "nearest_wrong": "num_adversarial_policy_nearest_wrong",
    "far_wrong": "num_adversarial_policy_far_wrong",
    "visually_similar_wrong": "num_adversarial_policy_visually_similar_wrong",
    "wrong_semihard": "num_adversarial_policy_visually_similar_wrong",
    "cross_flight": "num_adversarial_policy_cross_flight",
}


def _forge_attack_assignment(seed: Optional[int], image_name: str, inject_rate: float) -> bool:
    """Decide whether one frame is assigned to the attacked arm.

    Stateless hash of (seed, image_name) rather than a threaded RNG, so the
    assignment is independent of frame order, resume and sharding. The same
    seed and frame always land in the same arm.
    """
    if inject_rate >= 1.0:
        return True
    if inject_rate <= 0.0:
        return False
    key = "%s|%s" % (0 if seed is None else int(seed), image_name)
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()
    return (int(digest[:8], 16) / float(0xFFFFFFFF)) < inject_rate


def _configure_forge_footprints(cfg: Optional[Dict[str, Any]]) -> None:
    """Set the run's FORGE footprint tables from its config, once per run.

    The eligibility globals are set here and nowhere else. The config
    normaliser runs again for every output row, sometimes with an empty
    config, so it must not reset them.
    """
    cfg = cfg or {}
    FORGE_QUERY_HALF_M_TABLE.clear()
    FORGE_QUERY_HALF_M_TABLE_ACTIVE[0] = False
    table_path = cfg.get("forge_query_half_m_table", None)
    if table_path:
        import csv as _csv
        paths = table_path if isinstance(table_path, list) else [table_path]
        _seen = set()
        for _p in paths:
            with open(str(_p), newline="") as _fh:
                for _row in _csv.DictReader(_fh):
                    _name = str(_row.get("image_name", "")).strip()
                    if not _name or _name in _seen:
                        raise ValueError(f"forge_query_half_m_table {_p}: empty or duplicate image_name {_name!r}")
                    _seen.add(_name)
                    if str(_row.get("status", "")).strip() != "OK":
                        continue
                    _bound = float(_row["query_half_bound_m"])
                    if not (np.isfinite(_bound) and _bound > 0.0):
                        raise ValueError(f"forge_query_half_m_table {_p}: OK row {_name!r} has bound {_bound!r}")
                    FORGE_QUERY_HALF_M_TABLE[_name] = _bound
        if not FORGE_QUERY_HALF_M_TABLE:
            raise ValueError(f"forge_query_half_m_table {paths}: no OK rows")
        FORGE_QUERY_HALF_M_TABLE_ACTIVE[0] = True
        logger.info("FORGE per-frame query footprint table active: %d frames with a "
                    "certified bound (others unattackable) from %s",
                    len(FORGE_QUERY_HALF_M_TABLE), paths)
    per_map = cfg.get("forge_ref_half_m_per_map", None)
    FORGE_REF_HALF_M_PER_MAP.clear()
    if isinstance(per_map, dict):
        for _k, _v in per_map.items():
            FORGE_REF_HALF_M_PER_MAP[str(_k)] = (float(_v[0]), float(_v[1]))


def _normalize_adversarial_config(cfg: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Normalize FORGE/adversarial config from pipeline overrides."""
    cfg = cfg or {}
    mode = str(cfg.get("adversarial_mode", "none") or "none").strip().lower()
    condition_default = "clean" if mode == "none" else "wrong_near"
    condition = str(cfg.get("adversarial_condition", condition_default) or condition_default).strip().lower()
    explicit_policy = cfg.get("adversarial_candidate_policy", None)
    policy_default = ADVERSARIAL_POLICY_BY_CONDITION.get(condition, "force_rank_k")
    policy = str(explicit_policy if explicit_policy not in (None, "") else policy_default).strip().lower()
    if mode not in SUPPORTED_ADVERSARIAL_MODES:
        raise ValueError(f"Unknown adversarial_mode: {mode}")
    if condition not in SUPPORTED_ADVERSARIAL_CONDITIONS:
        raise ValueError(f"Unknown adversarial_condition: {condition}")
    if policy not in SUPPORTED_ADVERSARIAL_CANDIDATE_POLICIES:
        raise ValueError(f"Unknown adversarial_candidate_policy: {policy}")
    rank_raw = cfg.get("adversarial_candidate_rank", None)
    rank = None if rank_raw in (None, "") else _safe_int(rank_raw, default=-1)
    seed_raw = cfg.get("adversarial_seed", None)
    seed = None if seed_raw in (None, "") else _safe_int(seed_raw, default=0)
    rate_raw = cfg.get("adversarial_inject_rate", None)
    inject_rate = 1.0 if rate_raw in (None, "") else _safe_float(rate_raw, default=1.0)
    if not np.isfinite(inject_rate):
        inject_rate = 1.0
    inject_rate = float(min(1.0, max(0.0, inject_rate)))
    enabled = mode == "wrong_reference"
    return {
        "adversarial_mode": mode,
        "adversarial_condition": "clean" if not enabled else condition,
        "adversarial_candidate_policy": "" if not enabled else policy,
        "adversarial_candidate_rank": rank,
        "adversarial_seed": seed,
        "adversarial_inject_rate": inject_rate,
        "is_adversarial_enabled": bool(enabled),
        "adversarial_official_metrics": False,
    }


def _is_adversarial_payload(payload: Optional[Dict[str, Any]]) -> bool:
    payload = payload or {}
    mode = str(payload.get("adversarial_mode", "none") or "none").strip().lower()
    return bool(
        mode != "none"
        or payload.get("benchmark_family") == "forge"
        or bool(payload.get("adversarial_diagnostic", False))
    )


def _final_pose_gate_metadata(cfg: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    cfg = cfg or {}
    return {
        "final_pose_gate_enabled": bool(cfg.get("final_pose_gate_enabled", False)),
        "final_pose_min_shape_conf": cfg.get("final_pose_min_shape_conf"),
        "final_pose_max_reproj_residual_px": cfg.get("final_pose_max_reproj_residual_px"),
        "final_pose_max_geometry_anchor_offset_m": cfg.get("final_pose_max_geometry_anchor_offset_m"),
    }


def _latlon_distance_m_approx(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    m_per_deg_lat = 111_320.0
    mean_lat = np.radians((lat1 + lat2) * 0.5)
    m_per_deg_lon = 111_320.0 * np.cos(mean_lat)
    return float(np.hypot((lat1 - lat2) * m_per_deg_lat, (lon1 - lon2) * m_per_deg_lon))


def _safe_anchor_distance_m(anchor_a: Any, anchor_b: Any) -> float:
    if not isinstance(anchor_a, dict) or not isinstance(anchor_b, dict):
        return float("nan")
    lat1 = _safe_float(anchor_a.get("lat", np.nan), default=float("nan"))
    lon1 = _safe_float(anchor_a.get("lon", np.nan), default=float("nan"))
    lat2 = _safe_float(anchor_b.get("lat", np.nan), default=float("nan"))
    lon2 = _safe_float(anchor_b.get("lon", np.nan), default=float("nan"))
    if not all(np.isfinite(v) for v in [lat1, lon1, lat2, lon2]):
        return float("nan")
    return _latlon_distance_m_approx(lat1, lon1, lat2, lon2)


def _anchor_offset_per_axis_m(anchor_a: Any, anchor_b: Any) -> Tuple[float, float]:
    """East/north offset magnitudes between two anchors, in metres."""
    if not isinstance(anchor_a, dict) or not isinstance(anchor_b, dict):
        return float("nan"), float("nan")
    lat1 = _safe_float(anchor_a.get("lat", np.nan), default=float("nan"))
    lon1 = _safe_float(anchor_a.get("lon", np.nan), default=float("nan"))
    lat2 = _safe_float(anchor_b.get("lat", np.nan), default=float("nan"))
    lon2 = _safe_float(anchor_b.get("lon", np.nan), default=float("nan"))
    if not all(np.isfinite(v) for v in [lat1, lon1, lat2, lon2]):
        return float("nan"), float("nan")
    dn = abs(lat1 - lat2) * 111320.0
    de = abs(lon1 - lon2) * 111320.0 * float(np.cos(np.radians(lat1)))
    return float(de), float(dn)


def _apply_reliability_schema_defaults(row: Dict[str, Any], adversarial_cfg: Optional[Dict[str, Any]] = None) -> None:
    """Populate backward-compatible reliability/FORGE/CAVES columns."""
    cfg = adversarial_cfg or _normalize_adversarial_config({})
    enabled = bool(cfg.get("is_adversarial_enabled", False))
    row.setdefault("adversarial_mode", cfg.get("adversarial_mode", "none"))
    row.setdefault("adversarial_condition", cfg.get("adversarial_condition", "clean"))
    row.setdefault("adversarial_offset_m", float("nan"))
    row.setdefault("adversarial_candidate_policy", cfg.get("adversarial_candidate_policy", ""))
    rank = cfg.get("adversarial_candidate_rank")
    row.setdefault("adversarial_candidate_rank", "" if rank is None else rank)
    row.setdefault("adversarial_candidate_pool_size", 0)
    row.setdefault("adversarial_eligible_wrong_count", 0)
    row.setdefault("adversarial_selection_reason", "")
    row.setdefault("adversarial_failure_reason", "")
    row.setdefault("adversarial_forced_reference_name", "")
    row.setdefault("adversarial_original_reference_name", "")
    row.setdefault("adversarial_forced_distance_to_gt_m", float("nan"))
    row.setdefault("is_adversarial", bool(enabled))
    row.setdefault("adversarial_assigned_attack", bool(enabled))
    row.setdefault("adversarial_attacked", bool(enabled))
    row.setdefault("adversarial_inject_rate", float(cfg.get("adversarial_inject_rate", 1.0)))
    row.setdefault("is_wrong_reference", False)
    row.setdefault("retrieval_correct_threshold_m", float("nan"))
    row.setdefault("localization_success_threshold_m", float("nan"))
    row.setdefault("retrieval_rank", float("nan"))
    row.setdefault("localization_is_success", False)
    row.setdefault("caves_reject", False)
    row.setdefault("false_reject", False)
    row.setdefault("bad_pose_label", False)
    row.setdefault("unsafe_accept", False)
    row.setdefault("unsafe_accept_severity_m", 0.0)
    for key in (
        "inlier_ratio",
        "pnp_inlier_ratio",
        "retrieval_score_top1",
        "retrieval_score_top2",
        "retrieval_score_margin",
        "caves_risk_score",
        "caves_confidence",
        "obs_gate_score",
    ):
        row.setdefault(key, float("nan"))
    row.setdefault("caves_decision", "")
    row.setdefault("obs_gate_reason", "")
    row.setdefault("caves_evidence_count", 0)

    num_matches = _safe_float(row.get("num_matches", np.nan), default=float("nan"))
    num_inliers = _safe_float(row.get("num_inliers", np.nan), default=float("nan"))
    if np.isfinite(num_matches) and num_matches > 0 and np.isfinite(num_inliers):
        row["inlier_ratio"] = float(num_inliers / num_matches)
    pnp_inliers = _safe_float(row.get("pnp_inliers", np.nan), default=float("nan"))
    if np.isfinite(num_matches) and num_matches > 0 and np.isfinite(pnp_inliers):
        row["pnp_inlier_ratio"] = float(pnp_inliers / num_matches)
    top1 = _safe_float(row.get("retrieval_score_top1", row.get("raw_retrieval_similarity_score", np.nan)), default=float("nan"))
    top2 = _safe_float(row.get("retrieval_score_top2", np.nan), default=float("nan"))
    if np.isfinite(top1):
        row["retrieval_score_top1"] = float(top1)
    if np.isfinite(top2):
        row["retrieval_score_top2"] = float(top2)
    if np.isfinite(top1) and np.isfinite(top2):
        row["retrieval_score_margin"] = float(top1 - top2)
    rank_one_based = _safe_float(row.get("retrieval_rank", np.nan), default=float("nan"))
    if not np.isfinite(rank_one_based):
        candidate_rank = _safe_float(row.get("candidate_rank_selected", np.nan), default=float("nan"))
        if np.isfinite(candidate_rank) and candidate_rank >= 0:
            rank_one_based = float(candidate_rank + 1.0)
    if np.isfinite(rank_one_based):
        row["retrieval_rank"] = float(max(1.0, rank_one_based))


def _select_forge_candidate(
    adversarial_cfg: Dict[str, Any],
    candidates: List[Dict[str, Any]],
    dataset_name: str,
) -> Dict[str, Any]:
    """Select a wrong-reference candidate from normalized retrieval records."""
    pool_size = len(candidates or [])

    def _result(**kwargs: Any) -> Dict[str, Any]:
        payload = {"candidate_pool_size": pool_size, "eligible_wrong_count": 0}
        payload.update(kwargs)
        return payload

    if not bool(adversarial_cfg.get("is_adversarial_enabled", False)):
        return _result(ok=False, reason="adversarial_disabled", candidate=None)
    dataset_name = str(dataset_name or "").lower()
    if dataset_name not in FORGE_SUPPORTED_DATASETS:
        return _result(ok=False, reason="adversarial_unsupported_dataset", candidate=None, unsupported_condition=True)
    policy = str(adversarial_cfg.get("adversarial_candidate_policy", "force_rank_k") or "force_rank_k")
    if policy == "cross_flight":
        return _result(ok=False, reason="adversarial_cross_flight_unsupported", candidate=None, unsupported_condition=True)
    if not candidates:
        return _result(ok=False, reason="adversarial_no_candidates", candidate=None, metadata_unavailable=True)

    threshold_m = float(DATASET_RETRIEVAL_THRESHOLD_M.get(dataset_name, 20.0))
    if policy == "force_rank_k":
        public_rank = adversarial_cfg.get("adversarial_candidate_rank")
        if public_rank is None or int(public_rank) <= 0:
            return _result(ok=False, reason="adversarial_rank_out_of_range", candidate=None)
        target_rank = int(public_rank) - 1
        selected = next((c for c in candidates if int(c.get("rank", -1)) == target_rank), None)
        if selected is None:
            return _result(ok=False, reason="adversarial_rank_out_of_range", candidate=None)
        if bool(selected.get("is_original_candidate", False)):
            return _result(ok=False, reason="adversarial_rank_is_original", candidate=None)
        if selected.get("reference_image") is None:
            return _result(ok=False, reason="adversarial_metadata_unavailable", candidate=None, metadata_unavailable=True)
        return _result(ok=True, reason="force_rank_k", candidate=selected, eligible_wrong_count=1)

    # A reference from a different map is not a realistic retrieval fault: on
    # UAV-VisLoc the shortlist can carry tiles from another flight 460 km away,
    # which any check rejects trivially and which would inflate the measured
    # rejection rate. Restrict injection to the map the system actually
    # retrieved from. Datasets whose references are flat (no map prefix) are
    # unaffected.
    origin_map = ""
    for candidate in candidates:
        qm = str(candidate.get("query_map", "") or "")
        if qm:
            origin_map = qm
            break

    wrong_candidates = []
    for candidate in candidates:
        if bool(candidate.get("is_original_candidate", False)):
            continue
        if origin_map:
            cand_name = str(candidate.get("reference_name", ""))
            # Enforce only when the reference corpus itself is partitioned by
            # map (names carry a map prefix). AVL queries are named with a
            # flight-folder prefix while its tiles are flat; requiring a match
            # there excluded every candidate and fail-closed all 487 frames.
            if "/" in cand_name and cand_name.split("/")[0] != origin_map:
                continue
        dist = _safe_float(candidate.get("distance_to_gt_m", np.nan), default=float("nan"))
        if not np.isfinite(dist):
            continue
        if not _forge_candidate_excludes_gt(candidate, dataset_name, dist, threshold_m):
            continue
        if candidate.get("reference_image") is None:
            continue
        wrong_candidates.append(candidate)
    eligible_wrong_count = len(wrong_candidates)
    if not wrong_candidates:
        if not any(np.isfinite(_safe_float(c.get("distance_to_gt_m", np.nan), default=float("nan"))) for c in candidates):
            return _result(ok=False, reason="adversarial_metadata_unavailable", candidate=None, metadata_unavailable=True)
        return _result(ok=False, reason="adversarial_no_eligible_wrong_reference", candidate=None)
    if policy == "nearest_wrong":
        selected = min(wrong_candidates, key=lambda c: float(c.get("distance_to_gt_m")))
        return _result(ok=True, reason="nearest_wrong", candidate=selected, eligible_wrong_count=eligible_wrong_count)
    if policy == "far_wrong":
        selected = max(wrong_candidates, key=lambda c: float(c.get("distance_to_gt_m")))
        return _result(ok=True, reason="far_wrong", candidate=selected, eligible_wrong_count=eligible_wrong_count)
    if policy in {"visually_similar_wrong", "wrong_semihard"}:
        def _semihard_key(candidate: Dict[str, Any]) -> Tuple[int, float, int, int, str]:
            score = _safe_float(candidate.get("similarity_score", np.nan), default=float("nan"))
            score_missing = 0 if np.isfinite(score) else 1
            score_value = float(score) if np.isfinite(score) else float("-inf")
            rank = _safe_int(candidate.get("rank", 10**9), default=10**9)
            candidate_index = _safe_int(candidate.get("candidate_index", 10**9), default=10**9)
            reference_name = str(candidate.get("reference_name", ""))
            return (score_missing, -score_value, rank, candidate_index, reference_name)

        selected = min(wrong_candidates, key=_semihard_key)
        return _result(ok=True, reason="visually_similar_wrong", candidate=selected, eligible_wrong_count=eligible_wrong_count)
    return _result(ok=False, reason="adversarial_unsupported_policy", candidate=None, unsupported_condition=True)


def _finalize_forge_labels(row: Dict[str, Any], dataset_name: str, cfg: Optional[Dict[str, Any]]) -> None:
    """Finalize FORGE bad-pose and unsafe-accept diagnostics without changing success semantics."""
    assert "bad_pose_label" in row, "bad_pose_label must be initialized by _apply_reliability_schema_defaults"
    dataset_name = str(dataset_name or "").lower()
    threshold_m = _safe_float((cfg or {}).get("bad_pose_threshold_m", DATASET_RETRIEVAL_THRESHOLD_M.get(dataset_name, 20.0)), default=DATASET_RETRIEVAL_THRESHOLD_M.get(dataset_name, 20.0), min_value=0.0)
    post_error = _safe_float(row.get("post_fgo_error_m", np.nan), default=float("nan"))
    if np.isfinite(post_error):
        row["bad_pose_label"] = bool(post_error > threshold_m)
    elif bool(row.get("is_adversarial", False)) and bool(row.get("is_wrong_reference", False)) and not bool(row.get("accepted_for_metrics", False)):
        row["bad_pose_label"] = True
    row["unsafe_accept"] = bool(row.get("accepted_for_metrics", row.get("success", False))) and bool(row.get("bad_pose_label", False))
    _apply_reliability_schema_defaults(row, _normalize_adversarial_config(cfg or {}))
    try:
        from uav_geolocalization.benchmark.caves import apply_caves_v0

        reject_threshold = _safe_float((cfg or {}).get("caves_reject_threshold", 0.5), default=0.5, min_value=0.0)
        apply_caves_v0(row, reject_threshold=reject_threshold)
    except Exception as exc:  # keep CAVES diagnostic fail-open; never crash official runtime
        row.setdefault("caves_risk_score", float("nan"))
        row.setdefault("caves_confidence", float("nan"))
        row["caves_decision"] = "abstain_insufficient_evidence"
        row.setdefault("obs_gate_score", float("nan"))
        row["obs_gate_reason"] = f"caves_scoring_exception:{type(exc).__name__}"
    try:
        from uav_geolocalization.benchmark.caves import apply_caves_outcome_labels

        retrieval_threshold_m = _safe_float(
            (cfg or {}).get("retrieval_correct_threshold_m", DATASET_RETRIEVAL_THRESHOLD_M.get(dataset_name, 20.0)),
            default=DATASET_RETRIEVAL_THRESHOLD_M.get(dataset_name, 20.0),
            min_value=0.0,
        )
        apply_caves_outcome_labels(
            row,
            localization_success_threshold_m=threshold_m,
            retrieval_correct_threshold_m=retrieval_threshold_m,
        )
    except Exception as exc:  # labels are diagnostic; never crash benchmark runtime
        row.setdefault("localization_success_threshold_m", threshold_m)
        row.setdefault("retrieval_correct_threshold_m", DATASET_RETRIEVAL_THRESHOLD_M.get(dataset_name, 20.0))
        row.setdefault("localization_is_success", False)
        row.setdefault("caves_reject", False)
        row.setdefault("false_reject", False)
        row.setdefault("unsafe_accept_severity_m", 0.0)
        row["obs_gate_reason"] = row.get("obs_gate_reason") or f"caves_label_exception:{type(exc).__name__}"


@dataclass
class Phase1Output:
    """Phase-1 retrieval contract: only index + anchor + selected reference."""
    candidate_index: int
    geodetic_anchor: Dict[str, float]
    reference_image: Optional[np.ndarray]
    similarity_score: float = 0.0


@dataclass
class Phase2Input:
    """Phase-2 contract: pseudo-nadir query + selected reference tile."""
    pseudo_nadir_query: np.ndarray
    reference_tile: np.ndarray


# =============================================================================
# BENCHMARK RUNNER
# =============================================================================

class BenchmarkRunner:
    """
    Orchestrates multi-dataset benchmark execution.
    
    Configurable pipeline variants for ablation studies
    and comparison against baselines.
    """
    
    def __init__(self,
                 pipeline_name: str = 'ours',
                 output_dir: Path = Path("results"),
                 max_samples: int = None,
                 max_samples_percent: int = None,
                 test_interval: int = 1,
                 device: str = 'cpu',
                 eval_mode: str = 'end_to_end',
                 strict_components: bool = False,
                 strict_end2end: bool = False,
                 use_finetuned: bool = False,
                 pipeline_overrides: Optional[Dict] = None,
                 uav_visloc_protocol: str = "intra_flight",
                 retrieval_only: bool = False,
                 verbose: bool = True,
                 use_ipm_override: Optional[bool] = None):
        """
        Initialize benchmark runner.
        
        Args:
            pipeline_name: Name from PIPELINE_CONFIGS
            output_dir: Directory to save results
            max_samples: Max samples per dataset (None = all)
            max_samples_percent: Percentage of dataset to run (None = all)
            test_interval: Sample every N frames
            device: 'cpu' or 'cuda'
            use_finetuned: Use finetuned model checkpoints
            verbose: Verbose logging
        """
        self._pipeline_name = pipeline_name
        self._output_dir = Path(output_dir)
        self._max_samples = max_samples
        self._max_samples_percent = max_samples_percent
        self._test_interval = test_interval
        self._device = device
        self._eval_mode = eval_mode
        self._strict_components = bool(strict_components)
        self._strict_end2end = bool(strict_end2end)
        self._use_finetuned = use_finetuned
        self._uav_visloc_protocol = str(uav_visloc_protocol or "intra_flight")
        self._retrieval_only = bool(retrieval_only)
        self._verbose = verbose
        if self._eval_mode not in ('oracle_reference', 'end_to_end'):
            raise ValueError(
                f"Unknown eval_mode: {self._eval_mode}. "
                "Use 'oracle_reference' or 'end_to_end'."
            )
        if self._uav_visloc_protocol not in {"intra_flight", "cross_flight", "all_maps"}:
            raise ValueError(
                f"Unknown UAV-VisLoc protocol: {self._uav_visloc_protocol}. "
                "Use intra_flight, cross_flight, or all_maps."
            )
        
        if pipeline_name not in PIPELINE_CONFIGS:
            raise ValueError(
                f"Unknown pipeline: {pipeline_name}. "
                f"Available: {list(PIPELINE_CONFIGS.keys())}"
            )
        
        self._config = dict(PIPELINE_CONFIGS[pipeline_name])
        if pipeline_overrides:
            self._config.update(pipeline_overrides)
        if use_ipm_override is not None:
            logger.info(
                "IPM CLI override: use_ipm=%s (pipeline default was %s)",
                use_ipm_override, self._config.get('use_ipm', False),
            )
            self._config['use_ipm'] = use_ipm_override
        _configure_forge_footprints(self._config)
        self._adversarial_config = _normalize_adversarial_config(self._config)
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._stats = {
            "frame_exceptions": 0,
            "phase1_verify_exceptions": 0,
            "fgo_exceptions": 0,
            "retrieval_oracle_fallback_frames": 0,
            "roma_fallback_frames": 0,
            "reference_preconditioning_attempted": 0,
            "reference_preconditioning_accepted": 0,
            "reference_preconditioning_rejected": 0,
            "processed_frames": 0,
            "fgo_backend_policy": FGO_BACKEND_POLICY_PREFER_SCIPY,
            "effective_fgo_backend": "disabled",
            "fgo_backend_parity_ok": False,
            "fgo_backend_fallback_reason": "not_initialized",
            "retrieval_db_cache_enabled": False,
            "retrieval_db_cache_hit": False,
            "retrieval_db_cache_key": "",
            "retrieval_db_cache_rebuild_reason": "",
            "native_anyvisloc_cache_requested": False,
            "native_anyvisloc_cache_enabled": False,
            "native_anyvisloc_cache_key": "",
            "native_anyvisloc_cache_hit": False,
            "native_anyvisloc_cache_rebuild_reason": "",
            "telemetry_prior_policy": "simulated_uav_telemetry",
            "native_prior_uses_exact_gt": False,
            "native_anyvisloc_schema_ok": True,
            "native_yaw_height_status": "not_applicable",
            "retrieval_protocol_violation_count": 0,
            "retrieval_protocol_filtered_candidates_count": 0,
            "official_fail_closed_retrieval_reject_frames": 0,
            "dsm_available": False,
            "dsm_status": "not_initialized",
            "dsm_strict_fail_count": 0,
            "uav_selected_query_locations": [],
            "uav_selected_sample_signature": {},
            "uav_corpus_ref_flight_hash": "",
            "uav_tile_size_px": _safe_int(self._config.get("uav_tile_size_px", 2048), default=2048, min_value=1),
            "uav_tile_step_px": _safe_int(self._config.get("uav_tile_step_px", 512), default=512, min_value=1),
            "retrieval_top_k": _safe_int(self._config.get("retrieval_top_k", 5), default=5, min_value=1),
            "num_adversarial_samples": 0,
            "num_wrong_reference": 0,
            "num_bad_pose": 0,
            "num_unsafe_accept": 0,
            "num_adversarial_fail_closed": 0,
            "num_adversarial_unsupported_condition": 0,
            "num_adversarial_metadata_unavailable": 0,
            "num_adversarial_policy_force_rank_k": 0,
            "num_adversarial_policy_nearest_wrong": 0,
            "num_adversarial_policy_far_wrong": 0,
            "num_adversarial_policy_visually_similar_wrong": 0,
            "num_adversarial_policy_cross_flight": 0,
        }
    
    def _create_dataset(self, dataset_name: str):
        """Create dataset adapter."""
        from data.dataset_adapters import DatasetFactory
        
        cfg = DATASET_CONFIGS[dataset_name]
        kwargs = dict(cfg.get('kwargs', {}))
        if dataset_name == "uav_visloc":
            kwargs.setdefault("protocol", self._uav_visloc_protocol)
            kwargs.setdefault(
                "tile_size_px",
                _safe_int(self._config.get("uav_tile_size_px", 2048), default=2048, min_value=1),
            )
            kwargs.setdefault(
                "tile_step_px",
                _safe_int(self._config.get("uav_tile_step_px", 512), default=512, min_value=1),
            )
            # Flight subset for the v3 protocol (contiguous per-flight blocks
            # keep FGO's inter-frame motion prior valid, unlike test_interval).
            locations_cfg = self._config.get("uav_visloc_locations")
            if locations_cfg:
                if isinstance(locations_cfg, str):
                    locations_cfg = [v.strip() for v in locations_cfg.split(",") if v.strip()]
                kwargs.setdefault("locations", [str(v) for v in locations_cfg])
        if dataset_name == "avl":
            # Scale-mismatch diagnostic: query ground footprints span ~35-350 m
            # (30-300 m altitude) while the default corpus uses fixed 2048 px
            # (~123 m) tiles; expose the tile size so server sweeps can probe
            # retrieval scale sensitivity.
            kwargs.setdefault(
                "tile_size_px",
                _safe_int(self._config.get("avl_tile_size_px", 2048), default=2048, min_value=128),
            )
        if dataset_name == "avl24":
            scene_cfg = self._config.get("avl24_scene")
            if scene_cfg:
                kwargs["scene"] = str(scene_cfg)
            ref_mode = self._config.get("avl24_reference_mode")
            if ref_mode:
                kwargs["reference_mode"] = str(ref_mode)
            # Metric tiles: scene map resolution varies 16x across the release.
            kwargs.setdefault(
                "tile_size_m",
                _safe_float(self._config.get("avl24_tile_size_m", 256.0), default=256.0),
            )
            tile_px_cfg = self._config.get("avl24_tile_size_px")
            if tile_px_cfg:
                kwargs["tile_size_px"] = _safe_int(tile_px_cfg, default=2048, min_value=128)

        dataset = DatasetFactory.create(
            dataset_name=cfg['adapter'],
            data_root=Path(cfg['data_root']),
            test_interval=self._test_interval,
            max_samples=self._max_samples,
            max_samples_percent=self._max_samples_percent,
            **kwargs
        )
        list_file = self._config.get("query_name_list_file")
        if list_file:
            names = [ln.strip() for ln in Path(str(list_file)).read_text().splitlines()
                     if ln.strip() and not ln.startswith("#")]
            if not names:
                raise RuntimeError(f"query_name_list_file {list_file} lists no query")
            dataset._query_name_allowlist = set(names)
        return dataset

    def _build_native_anyvisloc_pipeline(self, dataset_name: str, cfg: Dict[str, Any]) -> Dict[str, Any]:
        """Build native AnyVisLoc components from the canonical native baseline adapter."""
        supported = set(cfg.get("supported_datasets", ["avl"]))
        if dataset_name not in supported:
            raise RuntimeError(
                f"{self._pipeline_name} supports datasets={sorted(supported)}; got {dataset_name}. "
                "No homography-wrapper fallback is allowed for native baselines."
            )

        try:
            from uav_geolocalization.baselines.native_anyvisloc.config import load_benchmark_config
            from uav_geolocalization.baselines.native_anyvisloc.benchmark_utils import (
                load_config_parameters_new,
                matching_init,
                retrieval_init,
            )
            from uav_geolocalization.baselines.native_anyvisloc.pipeline import (
                BenchmarkPipeline as NativeAnyVisLocPipeline,
            )
        except Exception as exc:
            raise RuntimeError(
                f"native_anyvisloc_e2e import preflight failed: {type(exc).__name__}: {exc}"
            ) from exc

        device = self._device
        if device == "cuda":
            try:
                import torch
                if not torch.cuda.is_available():
                    if self._strict_components:
                        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
                    device = "cpu"
            except Exception as exc:
                if self._strict_components:
                    raise
                logger.warning("native_anyvisloc_e2e CUDA preflight degraded to CPU: %s", exc)
                device = "cpu"

        ref_type = str(cfg.get("ref_type", "HIGH"))
        work_dir = Path(os.environ.get("HD4_NATIVE_ANYVISLOC_WORK_DIR", ".cache/native_anyvisloc_runtime"))
        work_dir = work_dir / self._run_id
        opt = SimpleNamespace(
            pose_priori="yp",
            strategy="Topn_opt",
            Ref_type=ref_type,
            resize_ratio=_safe_float(cfg.get("resize_ratio", 0.2), default=0.2, min_value=0.01),
            save_dir=str(work_dir),
            device=device,
        )
        retrieval_top_k = _safe_int(cfg.get("retrieval_top_k", 5), default=5, min_value=1)

        if dataset_name == "avl":
            return self._build_native_anyvisloc_avl(cfg, opt, ref_type, device, retrieval_top_k,
                                                     load_benchmark_config, load_config_parameters_new,
                                                     matching_init, retrieval_init, NativeAnyVisLocPipeline)
        elif dataset_name == "uav_visloc":
            return self._build_native_anyvisloc_uav_visloc(cfg, opt, ref_type, device, retrieval_top_k,
                                                            load_benchmark_config, matching_init,
                                                            retrieval_init, NativeAnyVisLocPipeline)
        elif dataset_name == "vpair":
            return self._build_native_anyvisloc_vpair(cfg, opt, ref_type, device, retrieval_top_k,
                                                       load_benchmark_config, matching_init,
                                                       retrieval_init, NativeAnyVisLocPipeline)
        else:
            raise RuntimeError(f"native_anyvisloc_e2e: unsupported dataset {dataset_name}")

    def _build_native_anyvisloc_avl(self, cfg, opt, ref_type, device, retrieval_top_k,
                                     load_benchmark_config, load_config_parameters_new,
                                     matching_init, retrieval_init, NativeAnyVisLocPipeline):
        """AVL path: original native AnyVisLoc with georeferenced raster + DSM."""
        dataset_cfg = DATASET_CONFIGS["avl"]
        region = str(dataset_cfg.get("kwargs", {}).get("region", "QZ_Town"))
        data_root = Path(dataset_cfg["data_root"])
        ref_path = data_root / "Reference_map" / region / "result_roi.tif"
        dsm_path = data_root / "Reference_map" / region / "dsm_roi.tif"
        if not ref_path.exists() or not dsm_path.exists():
            raise RuntimeError(
                "native_anyvisloc_e2e preflight failed: missing AVL ref/DSM maps "
                f"(ref={ref_path}, dsm={dsm_path})."
            )

        region_config = load_benchmark_config(region)
        region_config[f"{region}_UAV_PATH"] = str(data_root / "UAV_image" / region) + "/"
        region_config[f"{region}_{ref_type}_REF_PATH"] = str(ref_path)
        region_config[f"{region}_{ref_type}_DSM_PATH"] = str(dsm_path)
        region_config["DEVICE"] = device
        region_config["RETRIEVAL_TOPN"] = retrieval_top_k

        method_dict = {"retrieval_method": "CAMP"}
        method_dict = retrieval_init(method_dict, region_config)
        method_dict["matching_method"] = "Roma"
        method_dict = matching_init(method_dict)
        ref_image, dsm_image, _save_path, ref_resolution = load_config_parameters_new(region_config, opt, region)
        native_pipeline = NativeAnyVisLocPipeline(
            opt=opt, config=region_config, region=region,
            ref_image=ref_image, dsm_image=dsm_image, ref_resolution=ref_resolution,
            method_dict=method_dict, gsd_for_solver=ref_resolution,
            min_inliers_solver=_safe_int(cfg.get("geometry_min_inliers_after_lgi", 8), default=8, min_value=4),
        )
        self._stats["native_anyvisloc_ref_map_signature"] = self._file_signature(ref_path)
        self._stats["native_anyvisloc_dsm_signature"] = self._file_signature(dsm_path)
        self._stats["native_anyvisloc_cache_enabled"] = False
        self._stats["native_anyvisloc_cache_hit"] = False
        self._stats["native_anyvisloc_cache_rebuild_reason"] = "not_implemented"
        return {
            "native_backend": "anyvisloc",
            "native_pipeline": native_pipeline,
            "config": cfg,
            "effective_retrieval_backend": "CAMP(native_anyvisloc)",
            "effective_matcher_backend": "RoMa(native_anyvisloc)",
            "effective_geometry_backend": "DSM_P3P(native_anyvisloc)",
            "retrieval_reference_images": {},
            "retrieval_reference_names": {},
            "retrieval_reference_anchors": {},
        }

    @staticmethod
    def _make_synthetic_region_config(region: str, ref_type: str, utm_zone: int,
                                      northern: bool, origin_x: float, origin_y: float,
                                      gsd: float, device: str, retrieval_top_k: int) -> dict:
        """Generate a synthetic native AnyVisLoc config for non-AVL datasets."""
        import yaml
        hemisphere = "N" if northern else "S"
        base_yaml = Path(__file__).resolve().parents[2] / "baselines" / "native_anyvisloc" / "config" / "config.yaml"
        if base_yaml.exists():
            with open(base_yaml) as f:
                config = yaml.safe_load(f) or {}
        else:
            config = {
                "RETRIEVAL_METHODS": ["CAMP"], "MATCHING_METHODS": ["Roma"],
                "PNP_METHODS": ["P3P"], "USE_DEM": False,
                "RETRIEVAL_COVER": 50, "RETRIEVAL_TOPN": 5,
                "RETRIEVAL_IMG": "/1_Retrieval.png", "RETRIEVAL_IMG_NUM": 5,
                "RETRIEVAL_FEATURE_NORM": True, "SHOW_RETRIEVAL_RESULT": False,
                "BATCH_SIZE": 128,
            }
        config["DEVICE"] = device
        config["RETRIEVAL_TOPN"] = retrieval_top_k
        config["USE_DEM"] = False
        config[f"{region}_UTM_SYSTEM"] = f"{utm_zone}{hemisphere}"
        config[f"{region}_{ref_type}_REF_initialX"] = origin_x
        config[f"{region}_{ref_type}_REF_initialY"] = origin_y
        config[f"{region}_{ref_type}_REF_resolution"] = gsd
        config[f"{region}_{ref_type}_DSM_resolution"] = gsd
        config[f"{region}_{ref_type}_REF_COORDINATE"] = [0, 0]
        config[f"{region}_{ref_type}_DSM_COORDINATE"] = [0, 0]
        return config

    @staticmethod
    def _utm_zone_from_lon(lon: float) -> tuple:
        """Return (zone_number, is_northern) from longitude and assume northern hemisphere."""
        zone = int((lon + 180) / 6) + 1
        return zone, True

    def _build_native_anyvisloc_uav_visloc(self, cfg, opt, ref_type, device, retrieval_top_k,
                                            load_benchmark_config, matching_init,
                                            retrieval_init, NativeAnyVisLocPipeline):
        """UAV-VisLoc path: per-location satellite rasters with flat DSM."""
        import rasterio
        from pyproj import Transformer

        dataset_cfg = DATASET_CONFIGS["uav_visloc"]
        visloc_root = Path(dataset_cfg["data_root"])
        from data.dataset_adapters import _normalize_uav_visloc_root
        visloc_root = _normalize_uav_visloc_root(visloc_root)

        base_yaml = Path(__file__).resolve().parents[2] / "baselines" / "native_anyvisloc" / "config" / "config.yaml"
        import yaml
        if base_yaml.exists():
            with open(base_yaml) as f:
                base_config = yaml.safe_load(f) or {}
        else:
            base_config = {"RETRIEVAL_METHODS": ["CAMP"], "RETRIEVAL_COVER": 50, "RETRIEVAL_TOPN": 5,
                           "RETRIEVAL_IMG": "/1_Retrieval.png", "RETRIEVAL_IMG_NUM": 5,
                           "RETRIEVAL_FEATURE_NORM": True, "SHOW_RETRIEVAL_RESULT": False, "BATCH_SIZE": 128}
        base_config["DEVICE"] = device
        base_config["RETRIEVAL_TOPN"] = retrieval_top_k

        method_dict = {"retrieval_method": "CAMP"}
        method_dict = retrieval_init(method_dict, base_config)
        method_dict["matching_method"] = "Roma"
        method_dict = matching_init(method_dict)

        location_pipelines = {}
        for loc_dir in sorted(visloc_root.iterdir()):
            if not loc_dir.is_dir() or not loc_dir.name.isdigit():
                continue
            sat_dir = loc_dir / "satellite_map"
            sat_paths = sorted(list(loc_dir.glob("satellite*.tif")) + list(loc_dir.glob("satellite*.TIF")))
            if sat_dir.exists():
                sat_paths += sorted(list(sat_dir.glob("satellite*.tif")) + list(sat_dir.glob("satellite*.TIF")))
            if not sat_paths:
                logger.warning("UAV-VisLoc location %s: no satellite GeoTIFF found, skipping", loc_dir.name)
                continue
            sat_path = sat_paths[0]
            try:
                with rasterio.open(str(sat_path)) as src:
                    data = src.read()
                    if data.shape[0] >= 3:
                        ref_image = np.stack([data[2], data[1], data[0]], axis=-1).astype(np.uint8)
                    else:
                        ref_image = np.stack([data[0]] * 3, axis=-1).astype(np.uint8)
                    has_geo = bool(getattr(src, "crs", None))
                    if has_geo and src.transform is not None:
                        crs = src.crs
                        try:
                            epsg = crs.to_epsg()
                        except Exception:
                            epsg = None
                        is_geographic = epsg == 4326 or (
                            crs is not None and getattr(crs, "is_geographic", False)
                        )
                        if is_geographic:
                            center_lon = src.transform.c + abs(src.transform.a) * src.width / 2
                            center_lat = src.transform.f + src.transform.e * src.height / 2
                            utm_zone = int((center_lon + 180) / 6) + 1
                            northern = center_lat >= 0
                            epsg_utm = 32600 + utm_zone if northern else 32700 + utm_zone
                            trf_to_utm = Transformer.from_crs("epsg:4326", f"epsg:{epsg_utm}", always_xy=True)
                            origin_x, origin_y = trf_to_utm.transform(
                                src.transform.c, src.transform.f
                            )
                            br_x, br_y = trf_to_utm.transform(
                                src.transform.c + abs(src.transform.a) * src.width,
                                src.transform.f + src.transform.e * src.height,
                            )
                            gsd = abs(br_x - origin_x) / src.width
                        else:
                            origin_x = src.transform.c
                            origin_y = src.transform.f
                            gsd = abs(src.transform.a)
                            if epsg and 32600 < epsg < 32661:
                                utm_zone = epsg - 32600
                                northern = True
                            elif epsg and 32700 < epsg < 32761:
                                utm_zone = epsg - 32700
                                northern = False
                            else:
                                utm_zone, northern = 50, True
                    else:
                        origin_x, origin_y = 0.0, float(ref_image.shape[0])
                        gsd = 0.3
                        utm_zone, northern = 50, True
            except Exception as exc:
                logger.warning("UAV-VisLoc location %s: failed to load %s: %s", loc_dir.name, sat_path, exc)
                continue

            flat_dsm = np.zeros(ref_image.shape[:2], dtype=np.float32)
            region = f"uavvisloc_{loc_dir.name}"
            region_config = self._make_synthetic_region_config(
                region, ref_type, utm_zone, northern, origin_x, origin_y, gsd, device, retrieval_top_k)
            pipeline = NativeAnyVisLocPipeline(
                opt=opt, config=region_config, region=region,
                ref_image=ref_image, dsm_image=flat_dsm, ref_resolution=gsd,
                method_dict=method_dict, gsd_for_solver=gsd,
                min_inliers_solver=_safe_int(cfg.get("geometry_min_inliers_after_lgi", 6), default=6, min_value=4),
            )
            location_pipelines[loc_dir.name] = pipeline
            logger.info("UAV-VisLoc location %s: loaded satellite %s (%dx%d, GSD=%.4f)",
                        loc_dir.name, sat_path.name, ref_image.shape[1], ref_image.shape[0], gsd)

        if not location_pipelines:
            raise RuntimeError("native_anyvisloc_e2e on uav_visloc: no location satellite maps found")

        self._stats["native_anyvisloc_uav_visloc_locations"] = sorted(location_pipelines.keys())
        self._stats["native_anyvisloc_cache_enabled"] = False
        self._stats["native_anyvisloc_cache_hit"] = False
        self._stats["native_anyvisloc_cache_rebuild_reason"] = "per_location_mode"
        return {
            "native_backend": "anyvisloc",
            "native_pipeline": None,
            "native_location_pipelines": location_pipelines,
            "config": cfg,
            "effective_retrieval_backend": "CAMP(native_anyvisloc_per_location)",
            "effective_matcher_backend": "RoMa(native_anyvisloc)",
            "effective_geometry_backend": "homography_flat_terrain(native_anyvisloc)",
            "retrieval_reference_images": {},
            "retrieval_reference_names": {},
            "retrieval_reference_anchors": {},
        }

    def _build_native_anyvisloc_vpair(self, cfg, opt, ref_type, device, retrieval_top_k,
                                       load_benchmark_config, matching_init,
                                       retrieval_init, NativeAnyVisLocPipeline):
        """VPair path: gallery mode — per-frame reference image, no single raster."""
        dataset_cfg = DATASET_CONFIGS["vpair"]
        data_root = Path(dataset_cfg["data_root"])

        base_yaml = Path(__file__).resolve().parents[2] / "baselines" / "native_anyvisloc" / "config" / "config.yaml"
        import yaml
        if base_yaml.exists():
            with open(base_yaml) as f:
                base_config = yaml.safe_load(f) or {}
        else:
            base_config = {"RETRIEVAL_METHODS": ["CAMP"], "RETRIEVAL_COVER": 50, "RETRIEVAL_TOPN": 5,
                           "RETRIEVAL_IMG": "/1_Retrieval.png", "RETRIEVAL_IMG_NUM": 5,
                           "RETRIEVAL_FEATURE_NORM": True, "SHOW_RETRIEVAL_RESULT": False, "BATCH_SIZE": 128}
        base_config["DEVICE"] = device
        base_config["RETRIEVAL_TOPN"] = retrieval_top_k

        method_dict = {"retrieval_method": "CAMP"}
        method_dict = retrieval_init(method_dict, base_config)
        method_dict["matching_method"] = "Roma"
        method_dict = matching_init(method_dict)

        vpair_gsd = _safe_float(dataset_cfg.get("gsd", 0.3), default=0.3, min_value=0.01)
        region = "vpair_gallery"
        region_config = self._make_synthetic_region_config(
            region, ref_type, 33, True, 0.0, 0.0, vpair_gsd, device, retrieval_top_k)

        self._stats["native_anyvisloc_vpair_gallery_mode"] = True
        self._stats["native_anyvisloc_cache_enabled"] = False
        self._stats["native_anyvisloc_cache_hit"] = False
        self._stats["native_anyvisloc_cache_rebuild_reason"] = "gallery_mode"
        return {
            "native_backend": "anyvisloc",
            "native_pipeline": None,
            "native_gallery_mode": True,
            "native_method_dict": method_dict,
            "native_region_config": region_config,
            "native_opt": opt,
            "native_vpair_gsd": vpair_gsd,
            "native_vpair_data_root": str(data_root),
            "native_NativeAnyVisLocPipeline": NativeAnyVisLocPipeline,
            "config": cfg,
            "effective_retrieval_backend": "CAMP(native_anyvisloc_gallery)",
            "effective_matcher_backend": "RoMa(native_anyvisloc)",
            "effective_geometry_backend": "homography_flat_terrain(native_anyvisloc)",
            "retrieval_reference_images": {},
            "retrieval_reference_names": {},
            "retrieval_reference_anchors": {},
        }

    def _build_native_ortholoc_pipeline(self, dataset_name: str, cfg: Dict[str, Any]) -> Dict[str, Any]:
        """Build native OrthoLoC pipeline: DOP/DSM crop -> RoMa -> 2D-3D lift -> PnP."""
        if dataset_name != "avl":
            raise RuntimeError(
                f"native_ortholoc_2d3d_e2e: only AVL supported (requires DOP + real DSM), got {dataset_name}"
            )
        repo_root = REPO_ROOT
        report = check_ortholoc_preflight(repo_root=repo_root)
        self._stats["ortholoc_preflight"] = report
        if not report["ready"]:
            raise RuntimeError(
                "ortholoc_2d3d native preflight failed; missing dependencies="
                f"{report['missing_dependencies']}, reference_exists={report['reference_exists']}. "
                "Install poselib, rasterio and keep reference/ortholoc importable."
            )
        reference_parent = str(repo_root / "reference")
        if reference_parent not in sys.path:
            sys.path.insert(0, reference_parent)

        import rasterio
        from pyproj import Transformer

        dataset_cfg = DATASET_CONFIGS["avl"]
        region = str(dataset_cfg.get("kwargs", {}).get("region", "QZ_Town"))
        data_root = Path(dataset_cfg["data_root"])
        dop_path = data_root / "Reference_map" / region / "result_roi.tif"
        dsm_path = data_root / "Reference_map" / region / "dsm_roi.tif"
        if not dop_path.exists() or not dsm_path.exists():
            raise RuntimeError(
                f"native_ortholoc: missing DOP/DSM (dop={dop_path}, dsm={dsm_path})"
            )

        logger.info("OrthoLoC: loading DOP %s", dop_path)
        with rasterio.open(str(dop_path)) as src:
            dop_image = src.read().transpose(1, 2, 0)
            dop_transform = src.transform
            dop_crs = src.crs
        if dop_image.ndim != 3 or dop_image.shape[2] < 3:
            raise RuntimeError(f"DOP must be RGB, got shape {dop_image.shape}")
        dop_image = dop_image[:, :, :3]

        logger.info("OrthoLoC: loading DSM elevation %s", dsm_path)
        with rasterio.open(str(dsm_path)) as src:
            dsm_elev = src.read(1).astype(np.float64)
            dsm_transform = src.transform
            dsm_nodata = src.nodata
        if dsm_nodata is not None:
            dsm_elev[dsm_elev == dsm_nodata] = np.nan

        to_latlon = Transformer.from_crs(dop_crs, "EPSG:4326", always_xy=True)
        from_latlon = Transformer.from_crs("EPSG:4326", dop_crs, always_xy=True)

        device = str(cfg.get("device", "cuda"))
        from ortholoc.image_matching.MatcherRoMa import MatcherRoMa
        matcher_roma = MatcherRoMa(device=device)

        self._stats["native_ortholoc_dop_signature"] = self._file_signature(dop_path)
        self._stats["native_ortholoc_dsm_signature"] = self._file_signature(dsm_path)
        self._stats["native_ortholoc_dop_shape"] = list(dop_image.shape)
        self._stats["native_ortholoc_dsm_shape"] = list(dsm_elev.shape)
        logger.info(
            "OrthoLoC pipeline ready: DOP %s  DSM elev %s  CRS %s",
            dop_image.shape, dsm_elev.shape, dop_crs,
        )

        return {
            "native_backend": "ortholoc_2d3d",
            "native_ortholoc": {
                "dop_image": dop_image,
                "dop_transform": dop_transform,
                "dsm_elev": dsm_elev,
                "dsm_transform": dsm_transform,
                "matcher": matcher_roma,
                "to_latlon": to_latlon,
                "from_latlon": from_latlon,
            },
            "config": cfg,
            "effective_retrieval_backend": "telemetry_nearest(native_ortholoc)",
            "effective_matcher_backend": "RoMa(native_ortholoc)",
            "effective_geometry_backend": "OrthoLoC_PnP(poselib)",
            "retrieval_reference_images": {},
            "retrieval_reference_names": {},
            "retrieval_reference_anchors": {},
        }
    
    def _build_pipeline(self, dataset_name: str):
        """Build the localization pipeline from config."""
        cfg = self._config
        cfg = dict(cfg)
        dataset_cfg = DATASET_CONFIGS.get(dataset_name, {})
        gsd = dataset_cfg.get('gsd', 0.1)
        if dataset_name == "vpair":
            cfg.setdefault("retrieval_input_size", (300, 400))
            if cfg.get("enable_roma_fallback", False):
                cfg["min_inliers_fast"] = _safe_int(cfg.get("min_inliers_fast", 24), default=24, min_value=1)
                cfg["min_shape_conf"] = _safe_float(cfg.get("min_shape_conf", 0.30), default=0.30, min_value=0.0)
                cfg["max_reproj_residual_px"] = _safe_float(cfg.get("max_reproj_residual_px", 5.0), default=5.0, min_value=0.1)
                cfg["roma_budget_rate"] = _safe_float(cfg.get("roma_budget_rate", 0.70), default=0.70, min_value=0.0)
            cfg.setdefault("geometry_min_inliers_after_lgi", 8)
            cfg.setdefault("geometry_max_anchor_offset_m", 450.0)
            cfg.setdefault("fgo_max_step_xy_m", 120.0)
            cfg.setdefault("fgo_max_step_alt_m", 80.0)
        elif dataset_name == "avl":
            cfg.setdefault("geometry_min_inliers_after_lgi", 8)
            cfg.setdefault("geometry_max_anchor_offset_m", 350.0)
            cfg.setdefault("fgo_max_step_xy_m", 80.0)
            cfg.setdefault("fgo_max_step_alt_m", 60.0)
        elif dataset_name == "uav_visloc":
            # Cross-dataset Phase-1 parity: same resize/normalization contract as VPAIR.
            cfg.setdefault("retrieval_input_size", (300, 400))
            cfg.setdefault("retrieval_top_k", 10)
            cfg.setdefault("uav_tile_size_px", 2048)
            cfg.setdefault("uav_tile_step_px", 512)
            cfg.setdefault("retrieval_verify_top_k", True)
            cfg.setdefault("geometry_min_inliers_after_lgi", 6)
            cfg.setdefault("geometry_max_anchor_offset_m", 400.0)
            cfg.setdefault("max_accept_position_error_m", 500.0)
            cfg.setdefault("roma_budget_rate", 0.55)
            cfg.setdefault("min_inliers_fast", 15)
            cfg.setdefault("max_reproj_residual_px", 10.0)
            cfg.setdefault("fgo_max_step_xy_m", 150.0)
            cfg.setdefault("fgo_max_step_alt_m", 100.0)

        self._stats["retrieval_top_k"] = _safe_int(cfg.get("retrieval_top_k", 5), default=5, min_value=1)
        if dataset_name == "uav_visloc":
            self._stats["uav_tile_size_px"] = _safe_int(cfg.get("uav_tile_size_px", 2048), default=2048, min_value=1)
            self._stats["uav_tile_step_px"] = _safe_int(cfg.get("uav_tile_step_px", 512), default=512, min_value=1)

        native_backend = str(cfg.get("native_backend", "") or "")
        if native_backend in {
            "anyvisloc_shared_6dof",
            "anyvisloc_oracle_tile_shared_6dof",
            "ortholoc_shared_6dof",
            "ortholoc_oracle_tile_shared_6dof",
        }:
            raise RuntimeError(
                f"{self._pipeline_name} is registered as a hybrid shared-6DoF comparison, but "
                "the normalized frontend-output adapter and shared pose-head handoff are not wired. "
                "Failing closed so this row cannot be mistaken for a native or strict E2E result."
            )
        if native_backend in {"anyvisloc", "anyvisloc_oracle_tile", "anyvisloc_oracle_tile_pose_prior"}:
            return self._build_native_anyvisloc_pipeline(dataset_name, cfg)
        if native_backend == "ortholoc_2d3d":
            return self._build_native_ortholoc_pipeline(dataset_name, cfg)
        
        # Build components lazily
        components = {
            'config': cfg,
            'retrieval': None,
            'retrieval_reference_images': {},
            'retrieval_reference_names': {},
            'local_origin': None,
            'alignment': None,
            'matcher': None,
            'ipm': None,
            'lgi_filter': None,
            'fgo_solver': None,
            'last_fgo_state': None,
            'last_timestamp': None,
            'roma_fallback_matcher': None,
            'fgo_factors_added': 0,
            # ponytail: single-state solver; a >1 window pins the state to stale
            # frame positions on a moving platform (30-70 m lag on AVL). Keep 1
            # unless the solver grows per-frame states.
            'fgo_factor_window': _safe_int(cfg.get('fgo_factor_window', 1), default=1, min_value=1),
            'fgo_use_imu_preintegration': False,
            'fgo_preintegration_backend_parity_logged': False,
            'fgo_backend_policy': FGO_BACKEND_POLICY_PREFER_SCIPY,
            'effective_fgo_backend': 'disabled',
            'fgo_backend_parity_ok': False,
            'fgo_backend_fallback_reason': '',
            'effective_matcher_backend': 'unknown',
            'effective_retrieval_backend': 'none',
        }
        
        # IPM module
        if cfg.get('use_ipm', False) and not self._retrieval_only:
            from modules.vision.ipm import IPMFactory
            components['ipm'] = IPMFactory.create()
            logger.info("IPM module enabled")
        
        # LGI filter
        if cfg.get('use_lgi_filter', False) and not self._retrieval_only:
            from modules.vision.lgi_filter import LGIFilter
            components['lgi_filter'] = LGIFilter()
            logger.info("LGI filter enabled")
        
        # FGO solver
        if cfg.get('use_fgo', False) and not self._retrieval_only:
            # Auto-disable FGO when frames are sub-sampled (non-sequential).
            # FGO relies on inter-frame motion priors; with test_interval > 1
            # the optimization is underdetermined (15 DOF vs ~4 residuals) and
            # empirically degrades single-frame accuracy.
            if self._test_interval > 1:
                logger.warning(
                    "FGO auto-disabled: test_interval=%d > 1, frames are non-sequential. "
                    "FGO requires sequential frames for meaningful motion priors.",
                    self._test_interval,
                )
                cfg['use_fgo'] = False
                components['effective_fgo_backend'] = 'disabled'
                components['fgo_backend_parity_ok'] = False
                components['fgo_backend_fallback_reason'] = (
                    f"disabled_non_sequential_test_interval({self._test_interval})"
                )
            else:
                from modules.fusion.fgo_solver import create_fgo_solver
                scipy_ok = False
                scipy_err = ""
                try:
                    importlib.import_module("scipy.optimize")
                    scipy_ok = True
                except Exception as exc:
                    scipy_err = f"{type(exc).__name__}: {exc}"

                gtsam_ok = False
                gtsam_err = ""
                try:
                    importlib.import_module("gtsam")
                    gtsam_ok = True
                except Exception as exc:
                    gtsam_err = f"{type(exc).__name__}: {exc}"

                selected_backend = "disabled"
                fallback_reason = ""
                if scipy_ok:
                    selected_backend = "scipy"
                elif gtsam_ok:
                    selected_backend = "gtsam"
                    fallback_reason = f"scipy_unavailable({scipy_err})"
                else:
                    cfg['use_fgo'] = False
                    fallback_reason = (
                        f"scipy_unavailable({scipy_err}); gtsam_unavailable({gtsam_err})"
                    )
                    logger.error(
                        "FGO disabled: neither scipy nor gtsam is available. "
                        "Recommended fix: install scipy in benchmark env "
                        "(e.g., `conda install scipy` or `pip install scipy`) and rerun."
                    )

                if cfg.get('use_fgo', False) and selected_backend in {"scipy", "gtsam"}:
                    components['fgo_solver'] = create_fgo_solver(backend=selected_backend)
                    if selected_backend == "gtsam":
                        runtime_ok = bool(getattr(components['fgo_solver'], "is_gtsam_available", False))
                        if not runtime_ok:
                            cfg['use_fgo'] = False
                            components['fgo_solver'] = None
                            selected_backend = "disabled"
                            fallback_reason = (
                                "gtsam_selected_but_runtime_unavailable; "
                                f"scipy_unavailable({scipy_err})"
                            )
                            logger.error(
                                "FGO disabled: gtsam import preflight passed but runtime backend is unavailable. "
                                "Recommended fix: install scipy in benchmark env "
                                "(e.g., `conda install scipy` or `pip install scipy`) and rerun."
                            )
                        else:
                            logger.warning(
                                "FGO backend fallback: scipy unavailable (%s); using gtsam. "
                                "This keeps the run alive but parity is degraded. "
                                "For official thesis metrics, fix scipy and rerun.",
                                scipy_err if scipy_err else "unknown",
                            )
                    elif selected_backend == "scipy":
                        logger.info("FGO backend selected: scipy (benchmark parity mode).")

                components['effective_fgo_backend'] = selected_backend
                components['fgo_backend_parity_ok'] = bool(selected_backend == "scipy")
                components['fgo_backend_fallback_reason'] = fallback_reason
                # Benchmark parity policy: keep IMU preintegration OFF for every backend.
                # GTSAM simplified add_imu_preintegration is currently a no-op, so enabling
                # it only on scipy would create backend-dependent behavior.
                components['fgo_use_imu_preintegration'] = False
                if components['fgo_solver'] is not None:
                    logger.info("FGO solver enabled")
                    logger.warning(
                        "Benchmark parity: IMU preintegration is forcibly disabled for all FGO backends "
                        "(GTSAM simplified add_imu_preintegration is currently no-op)."
                    )

        # Retrieval module (Phase-1): UltraVPR -> candidate index + geodetic anchor.
        # Yaw/pose are intentionally not estimated here
        retrieval_backend = cfg.get('retrieval')
        if retrieval_backend in ['ultravpr', 'camp', 'megaloc']:
            from modules.vision.retrieval import create_retriever
            
            weights_path = None
            onnx_path = None
            
            if retrieval_backend == 'ultravpr':
                model_dir = Path("models") / "ultravpr"
                ckpt_candidates = []
                if self._use_finetuned:
                    ckpt_candidates.extend([
                        model_dir / "checkpoints" / "finetuned" / "model_best.pth.tar",
                        model_dir / "checkpoints" / "finetuned" / "checkpoint.pth.tar",
                    ])
                ckpt_candidates.extend([
                    model_dir / "checkpoints" / "model_best.pth.tar",
                    model_dir / "checkpoints" / "checkpoint.pth.tar",
                ])
                weights_path = next((p for p in ckpt_candidates if p.exists()), None)
                onnx_candidates = [
                    model_dir / "ultravpr_simplified.onnx",
                    model_dir / "ultravpr_original.onnx",
                ]
                onnx_path = next((p for p in onnx_candidates if p.exists()), None)
            elif retrieval_backend == 'camp':
                weights_path = Path("Retrieval_Models") / "CAMP" / "weights" / "weights_0.9446_for_U1652.pth"
            # megaloc: no local weights file; torch.hub downloads and caches.

            ret_input_size = cfg.get('retrieval_input_size')
            if retrieval_backend == 'camp':
                ret_input_size = 384
                
            components['retrieval'] = create_retriever(
                backend=retrieval_backend,
                weights_path=weights_path,
                onnx_path=onnx_path,
                device=self._device,
                top_k=cfg.get('retrieval_top_k', 5),
                verify_top_k=cfg.get('retrieval_verify_top_k', False),
                input_size=ret_input_size
            )
            components['effective_retrieval_backend'] = retrieval_backend
            logger.info(f"Retrieval engine ({retrieval_backend}) enabled (top_k={components['retrieval']._top_k})")
            if (
                self._strict_components
                and retrieval_backend != 'megaloc'  # hub-managed weights, no local file
                and (weights_path is None and onnx_path is None)
            ):
                raise RuntimeError(
                    "strict-components enabled: UltraVPR weights/onnx missing; random fallback is forbidden."
                )
            logger.info(
                "Retrieval module enabled: UltraVPR (finetuned=%s, weights=%s, onnx=%s)",
                self._use_finetuned,
                str(weights_path) if weights_path else "none",
                str(onnx_path) if onnx_path else "none",
            )
            
        if self._retrieval_only:
            logger.info("Retrieval-only mode: skipping matcher, geometry, and RoMa fallback initialization.")
            self._stats["fgo_backend_policy"] = str(
                components.get("fgo_backend_policy", FGO_BACKEND_POLICY_PREFER_SCIPY)
            )
            self._stats["effective_fgo_backend"] = str(
                components.get("effective_fgo_backend", "disabled")
            )
            self._stats["fgo_backend_parity_ok"] = bool(
                components.get("fgo_backend_parity_ok", False)
            )
            self._stats["fgo_backend_fallback_reason"] = str(
                components.get("fgo_backend_fallback_reason", "")
            )
            return components

        # Canonical matcher/geometry modules in src/modules/vision
        from modules.vision.matching import create_matcher
        from modules.vision.geometry_solver import create_geometry_solver
        roma_weights = os.environ.get("ROMA_WEIGHTS_PATH")
        dinov2_weights = os.environ.get("DINOV2_WEIGHTS_PATH")
        matcher_backend = cfg.get('matcher', 'lightglue')
        force_matcher = (os.environ.get("HD4_FORCE_MATCHER") or "").strip().lower()
        if force_matcher:
            logger.warning(
                "HD4_FORCE_MATCHER is set: overriding matcher backend %s -> %s",
                matcher_backend,
                force_matcher,
            )
            matcher_backend = force_matcher
        strict_roma = matcher_backend == 'roma'
        if strict_roma:
            logger.info(
                "RoMa strict mode enabled (weights=%s, dino=%s)",
                roma_weights if roma_weights else "unset",
                dinov2_weights if dinov2_weights else "unset",
            )
        components['matcher'] = create_matcher(
            matcher_backend,
            device=self._device,
            roma_weights_path=roma_weights,
            dinov2_weights_path=dinov2_weights,
            strict_roma=strict_roma,
            max_num_keypoints=_safe_int(cfg.get('matcher_max_keypoints', 2048), default=2048, min_value=128),
            resize_max=_safe_int(cfg.get('matcher_resize_max', 1024), default=1024, min_value=256),
        )
        components['effective_matcher_backend'] = type(components['matcher']).__name__
        if self._strict_components and type(components['matcher']).__name__ == "ORBMatcher":
            raise RuntimeError(
                "strict-components enabled: matcher degraded to ORB fallback."
            )
        components['geometry'] = create_geometry_solver(
            cfg.get('geometry', 'homography'),
            gsd=gsd,
            min_inliers=_safe_int(cfg.get("geometry_min_inliers_after_lgi", 8), default=8, min_value=4),
            reprojection_error_px=_safe_float(cfg.get("max_reproj_residual_px", 6.0), default=6.0, min_value=0.1),
            strict_dsm=bool(cfg.get("strict_dsm", True)),
            altitude_source=_solver_altitude_source(cfg.get("altitude_source")),
        )
        if cfg.get('enable_roma_fallback', False):
            components['roma_fallback_matcher'] = create_matcher(
                'roma',
                device=self._device,
                roma_weights_path=roma_weights,
                dinov2_weights_path=dinov2_weights,
                strict_roma=False,
            )
            if self._strict_components and type(components['roma_fallback_matcher']).__name__ == "ORBMatcher":
                raise RuntimeError(
                    "strict-components enabled: RoMa fallback matcher degraded to ORB."
                )

        # Expose effective FGO backend metadata for manifest/summary reproducibility.
        self._stats["fgo_backend_policy"] = str(
            components.get("fgo_backend_policy", FGO_BACKEND_POLICY_PREFER_SCIPY)
        )
        self._stats["effective_fgo_backend"] = str(
            components.get("effective_fgo_backend", "disabled")
        )
        self._stats["fgo_backend_parity_ok"] = bool(
            components.get("fgo_backend_parity_ok", False)
        )
        self._stats["fgo_backend_fallback_reason"] = str(
            components.get("fgo_backend_fallback_reason", "")
        )
        
        return components

    @staticmethod
    def _resolve_shape_confidence(
        match_result,
        query_image: np.ndarray,
        ref_image: np.ndarray,
    ) -> float:
        """Resolve shape confidence from homography; fallback to inlier ratio proxy."""
        from modules.vision.matching import compute_shape_confidence

        if match_result is None:
            return 0.0
        sc = compute_shape_confidence(
            homography=match_result.homography,
            query_size=(query_image.shape[1], query_image.shape[0]),
            ref_size=(ref_image.shape[1], ref_image.shape[0]),
        )
        if np.isfinite(sc):
            return float(sc)
        n_in = int(match_result.num_inliers)
        n_all = max(1, int(match_result.num_matches))
        return float(np.clip(n_in / max(8.0, float(n_all)), 0.0, 1.0))

    @staticmethod
    def _compute_reprojection_residual_px(match_result) -> float:
        """Mean reprojection residual from inlier correspondences."""
        if match_result is None or match_result.homography is None:
            return float("inf")
        q_pts, r_pts = match_result.get_inlier_matches()
        if q_pts is None or r_pts is None or len(q_pts) < 4:
            return float("inf")
        q_h = np.concatenate([q_pts.astype(np.float64), np.ones((len(q_pts), 1), dtype=np.float64)], axis=1)
        proj = (match_result.homography @ q_h.T).T
        z = proj[:, 2:3]
        valid = np.abs(z[:, 0]) > 1e-9
        if valid.sum() < 4:
            return float("inf")
        proj_xy = proj[valid, :2] / z[valid]
        err = np.linalg.norm(proj_xy - r_pts[valid].astype(np.float64), axis=1)
        return float(np.mean(err)) if len(err) > 0 else float("inf")

    @staticmethod
    def _candidate_quality_score(
        inliers: int,
        shape_conf: float,
        match_conf: float,
        retrieval_sim: float,
        reproj_residual_px: float,
    ) -> float:
        """Unified reliability score for candidate/match quality."""
        sc = shape_conf if np.isfinite(shape_conf) else 0.0
        mr = match_conf if np.isfinite(match_conf) else 0.0
        rs = retrieval_sim if np.isfinite(retrieval_sim) else 0.0
        rr = reproj_residual_px if np.isfinite(reproj_residual_px) else 1e3
        return (
            1.00 * float(inliers) +
            20.0 * float(sc) +
            30.0 * float(mr) +
            15.0 * float(rs) -
            1.50 * float(rr)
        )

    @staticmethod
    def _challenger_passes_margin(rank_idx: int,
                                  candidate_score: float,
                                  top1_score: Optional[float],
                                  margin_score: float) -> bool:
        """Rerank margin guard: a rank>0 candidate may displace the descriptor
        top-1 only if it beats the top-1's verification score by an absolute
        margin. Measured motivation: with margin 0 the verifier replaced 905
        correct top-1 anchors on VPAIR while repairing 192 (51.8% -> 25.5%
        selected-hit). margin_score <= 0 disables the guard (legacy behavior).
        """
        if rank_idx == 0 or margin_score <= 0.0 or top1_score is None:
            return True
        return candidate_score > top1_score + margin_score

    def _should_escalate_to_roma(
        self,
        pipeline: Dict[str, Any],
        num_inliers: int,
        shape_confidence: float,
        match_confidence: float,
        reproj_residual_px: float,
        pitch_deg: float,
    ) -> tuple[bool, str]:
        """Gate whether dense RoMa fallback should be invoked."""
        cfg = pipeline.get("config", {})
        min_inliers_fast = _safe_int(cfg.get("min_inliers_fast", cfg.get("roma_fallback_min_inliers", 24)), default=20, min_value=1)
        min_shape_conf = _safe_float(cfg.get("min_shape_conf", 0.25), default=0.25, min_value=0.0)
        max_reproj = _safe_float(cfg.get("max_reproj_residual_px", 6.0), default=6.0, min_value=0.1)
        roma_budget_rate = _safe_float(cfg.get("roma_budget_rate", 0.35), default=0.35, min_value=0.0)
        processed = max(1, int(self._stats.get("processed_frames", 0)))
        roma_rate = float(self._stats.get("roma_fallback_frames", 0)) / processed
        budget_exhausted = roma_rate >= roma_budget_rate

        if num_inliers < min_inliers_fast:
            return (not budget_exhausted, "low_inliers")
        if np.isfinite(shape_confidence) and shape_confidence < min_shape_conf:
            return (not budget_exhausted, "low_shape_confidence")
        if np.isfinite(reproj_residual_px) and reproj_residual_px > max_reproj:
            return (not budget_exhausted, "high_reprojection_residual")
        if abs(float(pitch_deg)) > 45.0 and match_confidence < 0.25:
            return (not budget_exhausted, "high_pitch_low_confidence")
        return (False, "fast_path_ok")

    @staticmethod
    def _atomic_dump_json(path: Path, payload: Dict[str, Any]) -> None:
        """Atomic JSON write via temp file + replace."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, cls=NpEncoder)
            os.replace(tmp, path)
        finally:
            try:
                if tmp.exists():
                    tmp.unlink()
            except OSError:
                pass

    @staticmethod
    def _file_signature(path_like: Any) -> Dict[str, Any]:
        """File signature for deterministic cache keying."""
        if not path_like:
            return {"path": "", "exists": False, "size": None, "mtime_ns": None}
        p = Path(path_like)
        try:
            rp = p.resolve()
        except Exception:
            rp = p
        sig = {
            "path": str(rp),
            "exists": bool(rp.exists()),
            "size": None,
            "mtime_ns": None,
        }
        if rp.exists():
            try:
                st = rp.stat()
                sig["size"] = int(st.st_size)
                sig["mtime_ns"] = int(st.st_mtime_ns)
            except OSError:
                pass
        return sig

    @classmethod
    def _retrieval_model_signature(cls, retriever) -> Dict[str, Any]:
        """Canonical model signature used for retrieval DB cache identity."""
        model = getattr(retriever, "model", None)
        backend = str(getattr(retriever, "_backend", getattr(model, "backend", "unknown")))
        preprocess_contract = {}
        if model is not None and hasattr(model, "get_preprocess_contract"):
            try:
                preprocess_contract = model.get_preprocess_contract()
            except Exception:
                preprocess_contract = {}
        return {
            "backend": backend,
            "model_class": type(model).__name__ if model is not None else "none",
            "input_hw": list(getattr(model, "_input_hw", [])) if model is not None else [],
            "descriptor_dim": int(getattr(model, "_descriptor_dim", 0)) if model is not None else 0,
            "weights": cls._file_signature(getattr(model, "_weights_path", None)),
            "onnx": cls._file_signature(getattr(model, "_onnx_path", None)),
            "preprocess_contract": preprocess_contract,
        }

    def _prepare_retrieval_database(self, dataset_name: str, dataset, pipeline: Dict) -> None:
        """Build or load retrieval DB with strong cross-run descriptor cache."""
        retriever = pipeline.get('retrieval')
        if retriever is None:
            return
        if retriever.database.size > 0:
            return

        cache_enabled = _env_flag("HD4_RETRIEVAL_DB_CACHE_ENABLE", True)
        cache_fail_closed = _env_flag("HD4_RETRIEVAL_DB_CACHE_FAIL_CLOSED", False)
        cache_dir = Path(os.environ.get("HD4_RETRIEVAL_DB_CACHE_DIR", ".cache/retrieval_db"))
        self._stats["retrieval_db_cache_enabled"] = bool(cache_enabled)
        self._stats["retrieval_db_cache_hit"] = False
        self._stats["retrieval_db_cache_key"] = ""
        self._stats["retrieval_db_cache_rebuild_reason"] = ""

        def _iter_unique_reference_corpus():
            seen = set()
            if hasattr(dataset, "iter_reference_corpus"):
                for ref in dataset.iter_reference_corpus():
                    ref_img = ref.get("ref_image")
                    if ref_img is None:
                        continue
                    ref_name = str(ref.get("ref_name", ""))
                    anchor = ref.get("anchor", {})
                    lat = float(anchor.get("lat", 0.0))
                    lon = float(anchor.get("lon", 0.0))
                    key = (
                        ref_name if ref_name else f"{round(lat, 6)}_{round(lon, 6)}",
                        int(ref_img.shape[0]),
                        int(ref_img.shape[1]),
                    )
                    if key in seen:
                        continue
                    seen.add(key)
                    yield {
                        "ref_name": ref_name,
                        "ref_image": ref_img,
                        "lat": lat,
                        "lon": lon,
                        "anchor": dict(anchor) if isinstance(anchor, dict) else {"lat": lat, "lon": lon},
                    }
            else:
                for frame in dataset:
                    ref_img = frame.get('reference_image')
                    gt = frame.get('gt_pose', {})
                    if ref_img is None:
                        continue
                    lat = float(gt.get('lat', 0.0))
                    lon = float(gt.get('lon', 0.0))
                    ref_name = str(frame.get("image_name", ""))
                    key = (
                        ref_name if ref_name else f"{round(lat, 6)}_{round(lon, 6)}",
                        int(ref_img.shape[0]),
                        int(ref_img.shape[1]),
                    )
                    if key in seen:
                        continue
                    seen.add(key)
                    yield {
                        "ref_name": ref_name,
                        "ref_image": ref_img,
                        "lat": lat,
                        "lon": lon,
                        "anchor": {"lat": lat, "lon": lon},
                    }

        corpus_signature: List[Dict[str, Any]] = []
        def _signature_anchor_meta(anchor: Dict[str, Any]) -> Dict[str, Any]:
            if not isinstance(anchor, dict):
                return {}
            out: Dict[str, Any] = {}
            for key in ("anchor_source", "row_start", "col_start", "tile_size_px", "tile_step_px"):
                if key not in anchor:
                    continue
                value = anchor.get(key)
                if isinstance(value, (int, np.integer)):
                    out[key] = int(value)
                elif isinstance(value, (float, np.floating)):
                    out[key] = round(float(value), 6)
                else:
                    out[key] = str(value)
            return out

        for ref in _iter_unique_reference_corpus():
            ref_name = str(ref.get("ref_name", ""))
            lat = float(ref.get("lat", 0.0))
            lon = float(ref.get("lon", 0.0))
            corpus_signature.append({
                "ref_name": ref_name if ref_name else f"{round(lat, 6)}_{round(lon, 6)}",
                "lat": round(lat, 9),
                "lon": round(lon, 9),
                "anchor_meta": _signature_anchor_meta(ref.get("anchor", {})),
            })

        if not corpus_signature:
            logger.warning("No reference tiles found to build retrieval DB.")
            return

        retrieval_backend = str(getattr(retriever, "_backend", "unknown"))
        model_signature = self._retrieval_model_signature(retriever)
        weights_exists = bool((model_signature.get("weights") or {}).get("exists", False))
        onnx_exists = bool((model_signature.get("onnx") or {}).get("exists", False))
        if retrieval_backend == "ultravpr" and not (weights_exists or onnx_exists):
            cache_enabled = False
            self._stats["retrieval_db_cache_enabled"] = False
            self._stats["retrieval_db_cache_rebuild_reason"] = "cache_disabled_missing_model_artifacts"
            logger.warning(
                "Retrieval DB cross-run cache disabled: UltraVPR weights/onnx unavailable "
                "(random_projection fallback would make cached descriptors non-reusable)."
            )
        preprocess_signature = model_signature.get("preprocess_contract", {})
        model_signature_hash = _stable_json_hash(model_signature)
        preprocess_signature_hash = _stable_json_hash(preprocess_signature)
        corpus_hash = _stable_json_hash(corpus_signature)
        selected_query_locations: List[str] = []
        selected_sample_signature: Dict[str, Any] = {}
        corpus_ref_flight_hash = ""
        uav_tile_size_px = None
        uav_tile_step_px = None
        if str(dataset_name) == "uav_visloc":
            if hasattr(dataset, "selected_query_locations"):
                selected_query_locations = list(dataset.selected_query_locations())
            if hasattr(dataset, "retrieval_tile_config"):
                tile_cfg = dataset.retrieval_tile_config()
                uav_tile_size_px = _safe_int(tile_cfg.get("uav_tile_size_px"), default=2048, min_value=1)
                uav_tile_step_px = _safe_int(tile_cfg.get("uav_tile_step_px"), default=512, min_value=1)
            selected_names: List[str] = []
            try:
                if hasattr(dataset, "selected_entry_indices") and hasattr(dataset, "_entries"):
                    selected_names = [
                        str(dataset._entries[idx].get("image_name", ""))
                        for idx in dataset.selected_entry_indices()
                    ]
            except Exception:
                selected_names = []
            selected_sample_signature = {
                "num_selected": int(len(selected_names)),
                "first": selected_names[:5],
                "last": selected_names[-5:] if selected_names else [],
            }
            corpus_ref_flights = [_flight_id_from_ref_name(str(item.get("ref_name", ""))) for item in corpus_signature]
            corpus_ref_flight_hash = _stable_json_hash({
                "num_refs": len(corpus_ref_flights),
                "flights": sorted(set(f for f in corpus_ref_flights if f)),
                "ordered": corpus_ref_flights,
            })
            self._stats["uav_selected_query_locations"] = selected_query_locations
            self._stats["uav_selected_sample_signature"] = selected_sample_signature
            self._stats["uav_corpus_ref_flight_hash"] = corpus_ref_flight_hash
            self._stats["uav_tile_size_px"] = _safe_int(uav_tile_size_px, default=2048, min_value=1)
            self._stats["uav_tile_step_px"] = _safe_int(uav_tile_step_px, default=512, min_value=1)
        cache_key_payload = build_retrieval_cache_key_payload(
            schema_version=RETRIEVAL_DB_CACHE_SCHEMA_VERSION,
            dataset_name=str(dataset_name),
            retrieval_backend=retrieval_backend,
            model_signature_hash=model_signature_hash,
            preprocess_signature_hash=preprocess_signature_hash,
            corpus_hash=corpus_hash,
            corpus_size=int(len(corpus_signature)),
            uav_visloc_protocol=self._uav_visloc_protocol,
            selected_query_locations=selected_query_locations,
            selected_sample_signature=selected_sample_signature,
            corpus_ref_flight_hash=corpus_ref_flight_hash,
            uav_tile_size_px=uav_tile_size_px,
            uav_tile_step_px=uav_tile_step_px,
        )
        cache_key = _stable_json_hash(cache_key_payload)
        cache_key_short = cache_key[:12]
        self._stats["retrieval_db_cache_key"] = cache_key_short

        cache_entry_dir = cache_dir / str(dataset_name) / retrieval_backend / cache_key
        cache_npz = cache_entry_dir / "retrieval_db.npz"
        cache_meta = cache_entry_dir / "retrieval_db.metadata.json"
        cache_tiles_dir = cache_entry_dir / "tiles"
        lock_override = str(os.environ.get("HD4_RETRIEVAL_DB_LOCK_PATH", "")).strip()
        lock_path = Path(lock_override) if lock_override else (cache_entry_dir / ".build.lock")
        serialize_db = _env_flag("HD4_SERIALIZE_RETRIEVAL_DB", True)

        def _write_images_and_encode(tile_dir: Path) -> tuple[list[str], list[Dict[str, float]], Dict[int, str]]:
            import cv2
            tile_images: List[str] = []
            tile_anchors: List[Dict[str, float]] = []
            ref_names: Dict[int, str] = {}
            if tile_dir.exists():
                shutil.rmtree(tile_dir, ignore_errors=True)
            tile_dir.mkdir(parents=True, exist_ok=True)
            for ref in _iter_unique_reference_corpus():
                ref_img = ref.get("ref_image")
                if ref_img is None:
                    continue
                idx = len(tile_images)
                tile_path = (tile_dir / f"tile_{idx:06d}.png").resolve()
                cv2.imwrite(str(tile_path), cv2.cvtColor(ref_img, cv2.COLOR_RGB2BGR))
                tile_images.append(str(tile_path))
                anchor_payload = dict(ref.get("anchor", {})) if isinstance(ref.get("anchor"), dict) else {}
                anchor_payload["lat"] = float(ref.get("lat", anchor_payload.get("lat", 0.0)))
                anchor_payload["lon"] = float(ref.get("lon", anchor_payload.get("lon", 0.0)))
                tile_anchors.append(anchor_payload)
                name = str(ref.get("ref_name", ""))
                ref_names[idx] = name if name else f"ref_{idx:06d}"
            if not tile_images:
                raise RuntimeError("No reference tiles available after corpus iteration.")
            retriever.encode_database(tile_images, tile_anchors)
            return tile_images, tile_anchors, ref_names

        def _load_cached_reference_maps(meta_payload: Dict[str, Any]) -> tuple[Dict[int, Any], Dict[int, str], Dict[int, Dict[str, float]]]:
            entries = meta_payload.get("ordered_entries")
            if not isinstance(entries, list):
                raise RuntimeError("Cache metadata missing ordered_entries list.")
            ref_images: Dict[int, Any] = {}
            ref_names: Dict[int, str] = {}
            ref_anchors: Dict[int, Dict[str, float]] = {}
            for idx, item in enumerate(entries):
                tile_path = str((item or {}).get("tile_path", ""))
                if not tile_path:
                    raise RuntimeError(f"Cache metadata missing tile_path at idx={idx}.")
                tile_path_resolved = str(Path(tile_path).resolve())
                if not Path(tile_path_resolved).exists():
                    raise RuntimeError(f"Cached tile missing at idx={idx}: {tile_path_resolved}")
                ref_images[idx] = tile_path_resolved
                ref_name = str((item or {}).get("ref_name", ""))
                ref_names[idx] = ref_name if ref_name else f"ref_{idx:06d}"
                anchor_payload = dict((item or {}).get("anchor", {})) if isinstance((item or {}).get("anchor"), dict) else {}
                anchor_payload["lat"] = float((item or {}).get("lat", anchor_payload.get("lat", np.nan)))
                anchor_payload["lon"] = float((item or {}).get("lon", anchor_payload.get("lon", np.nan)))
                ref_anchors[idx] = anchor_payload
            return ref_images, ref_names, ref_anchors

        def _cache_metadata_matches(meta_payload: Dict[str, Any]) -> tuple[bool, str]:
            if int(meta_payload.get("schema_version", -1)) != RETRIEVAL_DB_CACHE_SCHEMA_VERSION:
                return False, "schema_mismatch"
            if str(meta_payload.get("cache_key", "")) != cache_key:
                return False, "key_mismatch"
            if str(meta_payload.get("dataset_name", "")) != str(dataset_name):
                return False, "dataset_mismatch"
            if str(meta_payload.get("retrieval_backend", "")) != retrieval_backend:
                return False, "backend_mismatch"
            if str(meta_payload.get("model_signature_hash", "")) != model_signature_hash:
                return False, "model_signature_mismatch"
            if str(meta_payload.get("preprocess_signature_hash", "")) != preprocess_signature_hash:
                return False, "preprocess_signature_mismatch"
            if str(meta_payload.get("corpus_hash", "")) != corpus_hash:
                return False, "corpus_hash_mismatch"
            if int(meta_payload.get("corpus_size", -1)) != int(len(corpus_signature)):
                return False, "corpus_size_mismatch"
            ordered_refs = meta_payload.get("ordered_refs")
            if not isinstance(ordered_refs, list):
                return False, "ordered_refs_missing"
            if len(ordered_refs) != len(corpus_signature):
                return False, "ordered_refs_len_mismatch"
            for idx, (cached_item, live_item) in enumerate(zip(ordered_refs, corpus_signature)):
                cached_name = str((cached_item or {}).get("ref_name", ""))
                if cached_name != str(live_item.get("ref_name", "")):
                    return False, f"ordered_ref_name_mismatch_idx_{idx}"
                try:
                    cached_lat = float((cached_item or {}).get("lat", np.nan))
                    cached_lon = float((cached_item or {}).get("lon", np.nan))
                except (TypeError, ValueError):
                    return False, f"ordered_ref_anchor_invalid_idx_{idx}"
                if round(cached_lat, 9) != round(float(live_item.get("lat", 0.0)), 9):
                    return False, f"ordered_ref_lat_mismatch_idx_{idx}"
                if round(cached_lon, 9) != round(float(live_item.get("lon", 0.0)), 9):
                    return False, f"ordered_ref_lon_mismatch_idx_{idx}"
            return True, ""

        def _encode_and_optionally_save(enable_cache_write: bool = True) -> tuple[Dict[int, Any], Dict[int, str], Dict[int, Dict[str, float]], str]:
            if cache_enabled and enable_cache_write:
                tile_images, tile_anchors, ref_names = _write_images_and_encode(cache_tiles_dir)
                ordered_entries = []
                ref_images: Dict[int, Any] = {}
                ref_anchors: Dict[int, Dict[str, float]] = {}
                for idx, img_path in enumerate(tile_images):
                    ref_images[idx] = img_path
                    ref_anchors[idx] = {
                        **dict(tile_anchors[idx]),
                        "lat": float(tile_anchors[idx].get("lat", np.nan)),
                        "lon": float(tile_anchors[idx].get("lon", np.nan)),
                    }
                    ordered_entries.append({
                        "ref_name": str(ref_names.get(idx, f"ref_{idx:06d}")),
                        "lat": round(float(tile_anchors[idx].get("lat", 0.0)), 9),
                        "lon": round(float(tile_anchors[idx].get("lon", 0.0)), 9),
                        "anchor": dict(tile_anchors[idx]),
                        "tile_path": str(img_path),
                    })
                cache_entry_dir.mkdir(parents=True, exist_ok=True)
                db_tmp = cache_entry_dir / f"retrieval_db.tmp.{os.getpid()}.npz"
                retriever.database.save(db_tmp)
                os.replace(db_tmp, cache_npz)
                metadata = {
                    "schema_version": RETRIEVAL_DB_CACHE_SCHEMA_VERSION,
                    "created_at": datetime.now().isoformat(),
                    "cache_key": cache_key,
                    "cache_key_short": cache_key_short,
                    "dataset_name": str(dataset_name),
                    "retrieval_backend": retrieval_backend,
                    "model_signature": model_signature,
                    "model_signature_hash": model_signature_hash,
                    "preprocess_signature": preprocess_signature,
                    "preprocess_signature_hash": preprocess_signature_hash,
                    "corpus_hash": corpus_hash,
                    "corpus_size": int(len(corpus_signature)),
                    "ordered_refs": corpus_signature,
                    "ordered_entries": ordered_entries,
                }
                self._atomic_dump_json(cache_meta, metadata)
                # Keep sibling cache entries by default so smoke subsets and the
                # full matrix can coexist; opt into GC with HD4_RETRIEVAL_DB_CACHE_GC=1.
                if _env_flag("HD4_RETRIEVAL_DB_CACHE_GC", False):
                    try:
                        cleanup_result = cleanup_stale_retrieval_cache(
                            cache_dir=cache_dir,
                            current_cache_key=cache_key,
                            dataset_name=str(dataset_name),
                            retrieval_backend=retrieval_backend,
                        )
                        if cleanup_result.get("removed"):
                            logger.info(
                                "Cleaned up %d stale cache entries for %s/%s",
                                len(cleanup_result["removed"]),
                                dataset_name,
                                retrieval_backend,
                            )
                    except Exception as cleanup_exc:
                        logger.warning("Cache cleanup failed (non-fatal): %s", cleanup_exc)
                return ref_images, ref_names, ref_anchors, "cache_rebuilt"

            import uuid
            run_uuid = uuid.uuid4().hex[:8]
            tile_cache_dir = Path(f"/tmp/forge_tile_cache_{self._config['name'].replace(' ', '_')}_{run_uuid}")
            tile_images, tile_anchors, ref_names = _write_images_and_encode(tile_cache_dir)
            ref_images = {idx: p for idx, p in enumerate(tile_images)}
            ref_anchors = {
                idx: {
                    **dict(anchor),
                    "lat": float(anchor.get("lat", np.nan)),
                    "lon": float(anchor.get("lon", np.nan)),
                }
                for idx, anchor in enumerate(tile_anchors)
            }
            return ref_images, ref_names, ref_anchors, "cache_disabled"

        cache_rebuild_reason = ""
        ref_images: Dict[int, Any] = {}
        ref_names: Dict[int, str] = {}
        ref_anchors: Dict[int, Dict[str, float]] = {}

        try:
            if cache_enabled:
                cache_entry_dir.mkdir(parents=True, exist_ok=True)
                lock_path.parent.mkdir(parents=True, exist_ok=True)
                if serialize_db and fcntl is not None:
                    with open(lock_path, "w", encoding="utf-8") as lock_fp:
                        logger.info("Retrieval DB cache lock: waiting on %s", lock_path)
                        fcntl.flock(lock_fp.fileno(), fcntl.LOCK_EX)
                        logger.info("Retrieval DB cache lock: acquired %s", lock_path)
                        try:
                            if cache_npz.exists() and cache_meta.exists():
                                with open(cache_meta, "r", encoding="utf-8") as f:
                                    meta_payload = json.load(f)
                                ok, reason = _cache_metadata_matches(meta_payload)
                                if ok:
                                    retriever.database.load(cache_npz)
                                    ref_images, ref_names, ref_anchors = _load_cached_reference_maps(meta_payload)
                                    if int(retriever.database.size) != int(len(ref_images)):
                                        raise RuntimeError(
                                            f"Descriptor/image count mismatch: db={retriever.database.size}, refs={len(ref_images)}"
                                        )
                                    self._stats["retrieval_db_cache_hit"] = True
                                    self._stats["retrieval_db_cache_rebuild_reason"] = ""
                                    cache_rebuild_reason = ""
                                else:
                                    cache_rebuild_reason = reason
                            else:
                                cache_rebuild_reason = "cache_miss"

                            if not self._stats.get("retrieval_db_cache_hit", False):
                                ref_images, ref_names, ref_anchors, default_reason = _encode_and_optionally_save(enable_cache_write=True)
                                if not cache_rebuild_reason:
                                    cache_rebuild_reason = default_reason
                        finally:
                            fcntl.flock(lock_fp.fileno(), fcntl.LOCK_UN)
                else:
                    if cache_npz.exists() and cache_meta.exists():
                        with open(cache_meta, "r", encoding="utf-8") as f:
                            meta_payload = json.load(f)
                        ok, reason = _cache_metadata_matches(meta_payload)
                        if ok:
                            retriever.database.load(cache_npz)
                            ref_images, ref_names, ref_anchors = _load_cached_reference_maps(meta_payload)
                            if int(retriever.database.size) != int(len(ref_images)):
                                raise RuntimeError(
                                    f"Descriptor/image count mismatch: db={retriever.database.size}, refs={len(ref_images)}"
                                )
                            self._stats["retrieval_db_cache_hit"] = True
                            self._stats["retrieval_db_cache_rebuild_reason"] = ""
                            cache_rebuild_reason = ""
                        else:
                            cache_rebuild_reason = reason
                    else:
                        cache_rebuild_reason = "cache_miss"
                    if not self._stats.get("retrieval_db_cache_hit", False):
                        ref_images, ref_names, ref_anchors, default_reason = _encode_and_optionally_save(enable_cache_write=True)
                        if not cache_rebuild_reason:
                            cache_rebuild_reason = default_reason
            else:
                ref_images, ref_names, ref_anchors, cache_rebuild_reason = _encode_and_optionally_save(enable_cache_write=False)
        except Exception as exc:
            if cache_fail_closed:
                raise RuntimeError(
                    f"Retrieval DB cache failed in fail-closed mode: {type(exc).__name__}: {exc}"
                ) from exc
            logger.warning(
                "Retrieval DB cache error (%s: %s). Falling back to live encode.",
                type(exc).__name__,
                exc,
            )
            try:
                ref_images, ref_names, ref_anchors, fallback_reason = _encode_and_optionally_save(enable_cache_write=bool(cache_enabled))
                cache_rebuild_reason = f"fallback_after_error({type(exc).__name__})::{fallback_reason}"
                self._stats["retrieval_db_cache_hit"] = False
            except Exception as rebuild_exc:
                logger.warning(
                    "Retrieval DB cache rebuild failed (%s: %s). Using no-cache encode fallback.",
                    type(rebuild_exc).__name__,
                    rebuild_exc,
                )
                try:
                    ref_images, ref_names, ref_anchors, fallback_reason = _encode_and_optionally_save(enable_cache_write=False)
                    cache_rebuild_reason = (
                        f"fallback_after_error({type(exc).__name__})::"
                        f"fallback_no_cache_after_rebuild_error({type(rebuild_exc).__name__})::{fallback_reason}"
                    )
                    self._stats["retrieval_db_cache_hit"] = False
                except Exception as fallback_exc:
                    raise RuntimeError(
                        f"Retrieval DB encode fallback failed: {type(fallback_exc).__name__}: {fallback_exc}"
                    ) from fallback_exc

        if self._strict_components and getattr(retriever.model, "backend", None) == "random_projection":
            raise RuntimeError(
                "strict-components enabled: UltraVPR is running random_projection backend."
            )

        model = getattr(retriever, "model", None)
        if model is not None and hasattr(model, "get_preprocess_contract"):
            logger.info(
                "Retrieval preprocess contract (%s): %s",
                dataset_name,
                model.get_preprocess_contract(),
            )

        pipeline['retrieval_reference_images'] = ref_images
        pipeline['retrieval_reference_names'] = ref_names
        pipeline['retrieval_reference_anchors'] = ref_anchors
        if not bool(self._stats.get("retrieval_db_cache_hit", False)):
            existing_reason = str(self._stats.get("retrieval_db_cache_rebuild_reason", ""))
            if existing_reason and str(cache_rebuild_reason) == "cache_disabled":
                pass
            else:
                self._stats["retrieval_db_cache_rebuild_reason"] = str(cache_rebuild_reason)

        logger.info(
            "Prepared retrieval DB: %d tiles (cache_enabled=%s, cache_hit=%s, key=%s, reason=%s).",
            len(ref_images),
            cache_enabled,
            bool(self._stats.get("retrieval_db_cache_hit", False)),
            cache_key_short,
            self._stats.get("retrieval_db_cache_rebuild_reason", ""),
        )

    def _prepare_dsm_map(self, dataset_name: str, dataset: Any, pipeline: Dict[str, Any]) -> None:
        """Attach a DSM map to experimental DSM-aware pose heads when available."""
        cfg = pipeline.get("config", {})
        if str(cfg.get("geometry", "")) != "dsm_pnp":
            self._stats["dsm_available"] = False
            self._stats["dsm_status"] = "not_requested"
            return
        dsm_map = None
        status = "unavailable"
        try:
            # Datasets that carry their own DSM grid (e.g. the AnyVisLoc NPZ
            # release, whose DSM is a .npy on a scene-local grid) build the
            # bridge themselves; RasterDSMMap only knows the GeoTIFF layout.
            if hasattr(dataset, "build_dsm_map"):
                dsm_map = dataset.build_dsm_map()
            if dsm_map is None:
                from modules.vision.dsm_map import RasterDSMMap
                dsm_map = RasterDSMMap.from_dataset(dataset)
            if dsm_map is not None:
                status = str(getattr(dsm_map, "status", "unknown"))
        except Exception as exc:
            status = f"init_failed:{type(exc).__name__}"
            dsm_map = None
        pipeline["dsm_map"] = dsm_map
        if pipeline.get("geometry") is not None and hasattr(pipeline["geometry"], "_dsm_map"):
            pipeline["geometry"]._dsm_map = dsm_map
        available = bool(getattr(dsm_map, "available", False))
        self._stats["dsm_available"] = available
        self._stats["dsm_status"] = status if status else ("available" if available else "unavailable")
        if not available:
            logger.warning(
                "DSM-aware pipeline requested on %s but DSM map is unavailable (%s). "
                "Frames will fail closed; do not claim 6-DoF metrics for this run.",
                dataset_name,
                self._stats["dsm_status"],
            )

    @staticmethod
    def _load_reference_image_if_needed(ref_img: Any) -> Optional[np.ndarray]:
        if ref_img is None:
            return None
        if isinstance(ref_img, str):
            import cv2
            loaded = cv2.imread(ref_img, cv2.IMREAD_COLOR)
            if loaded is None:
                return None
            return cv2.cvtColor(loaded, cv2.COLOR_BGR2RGB)
        return ref_img

    def _build_forge_candidate_records(
        self,
        *,
        raw_retrieval_results: List[Any],
        ref_map: Dict[int, Any],
        ref_name_map: Dict[int, str],
        ref_anchor_map: Dict[int, Dict[str, Any]],
        frame: Dict[str, Any],
        selected_candidate_index: int,
        dataset_name: str = "",
    ) -> List[Dict[str, Any]]:
        gt = frame.get("gt_pose", {}) if isinstance(frame.get("gt_pose", {}), dict) else {}
        gt_anchor = {"lat": gt.get("lat", np.nan), "lon": gt.get("lon", np.nan)}
        candidates: List[Dict[str, Any]] = []
        for rank, rr in enumerate(raw_retrieval_results):
            try:
                candidate_idx = int(rr.candidate_index)
            except (TypeError, ValueError):
                continue
            ref_img = self._load_reference_image_if_needed(ref_map.get(candidate_idx))
            stored_anchor = ref_anchor_map.get(candidate_idx, {})
            anchor_payload = dict(stored_anchor) if isinstance(stored_anchor, dict) else {}
            rr_anchor = getattr(rr, "geodetic_anchor", {}) if isinstance(getattr(rr, "geodetic_anchor", {}), dict) else {}
            anchor_payload.update({
                "lat": _safe_float(rr_anchor.get("lat", anchor_payload.get("lat", np.nan)), default=float("nan")),
                "lon": _safe_float(rr_anchor.get("lon", anchor_payload.get("lon", np.nan)), default=float("nan")),
                "alt": _safe_float(rr_anchor.get("alt", anchor_payload.get("alt", np.nan)), default=float("nan")),
            })
            distance_to_gt_m = _safe_anchor_distance_m(anchor_payload, gt_anchor)
            offset_e_m, offset_n_m = _anchor_offset_per_axis_m(anchor_payload, gt_anchor)
            query_half_e_m, query_half_n_m = forge_query_half_m(str(dataset_name or "").lower(), gt)
            _qname = str(frame.get("image_name", "") or "")
            if FORGE_QUERY_HALF_M_TABLE_ACTIVE[0]:
                _q = FORGE_QUERY_HALF_M_TABLE.get(_qname)
                # Indeterminate footprint: an infinite half-extent makes every
                # candidate overlap, so the frame fails closed as unattackable.
                query_half_e_m = query_half_n_m = float(_q) if _q is not None else float("inf")
            query_map = _qname.split("/")[0] if "/" in _qname else ""
            candidates.append({
                "candidate_index": candidate_idx,
                "rank": int(rank),
                "reference_name": str(ref_name_map.get(candidate_idx, f"ref_{candidate_idx:06d}")),
                "reference_image": ref_img,
                "anchor": anchor_payload,
                "similarity_score": _safe_float(getattr(rr, "similarity_score", np.nan), default=float("nan")),
                "distance_to_gt_m": distance_to_gt_m,
                # Per-axis offsets drive FORGE eligibility: a reference whose
                # footprint still covers the query is not a wrong-reference fault.
                "offset_e_m": offset_e_m,
                "offset_n_m": offset_n_m,
                "query_half_e_m": query_half_e_m,
                "query_half_n_m": query_half_n_m,
                "query_map": query_map,
                "is_original_candidate": int(candidate_idx) == int(selected_candidate_index),
            })
            _meta = frame.get("metadata", {}) if isinstance(frame.get("metadata", {}), dict) else {}
            if "forge_footprint_latlon_box" in _meta:
                candidates[-1]["query_box_latlon"] = _meta.get("forge_footprint_latlon_box")
                candidates[-1]["ref_half_m"] = _safe_float(_meta.get("forge_ref_half_m", np.nan), default=float("nan"))
        return candidates

    def _make_adversarial_fail_closed_output(
        self,
        *,
        base_output: Optional[Phase1Output],
        reason: str,
        selection: Optional[Dict[str, Any]] = None,
    ) -> Phase1Output:
        cfg = self._adversarial_config
        selection = selection or {}
        anchor_payload = dict(base_output.geodetic_anchor) if base_output and isinstance(base_output.geodetic_anchor, dict) else {}
        original_name = str(anchor_payload.get("_selected_reference_name", ""))
        anchor_payload.update({
            "_adversarial_mode": cfg.get("adversarial_mode", "wrong_reference"),
            "_adversarial_condition": cfg.get("adversarial_condition", ""),
            "_adversarial_candidate_policy": cfg.get("adversarial_candidate_policy", ""),
            "_adversarial_candidate_rank": cfg.get("adversarial_candidate_rank"),
            "_adversarial_candidate_pool_size": int(selection.get("candidate_pool_size", 0) or 0),
            "_adversarial_eligible_wrong_count": int(selection.get("eligible_wrong_count", 0) or 0),
            "_adversarial_selection_reason": str(selection.get("reason", "")),
            "_adversarial_forced_reference_name": "",
            "_adversarial_assigned_attack": True,
            "_adversarial_attacked": False,
            "_adversarial_original_reference_name": original_name,
            "_adversarial_forced_distance_to_gt_m": float("nan"),
            "_adversarial_offset_m": float("nan"),
            "_is_adversarial": True,
            "_is_wrong_reference": False,
            "_adversarial_fail_closed": True,
            "_adversarial_failure_reason": reason,
            "_adversarial_unsupported_condition": reason in {"adversarial_cross_flight_unsupported", "adversarial_unsupported_dataset", "adversarial_unsupported_policy"},
            "_adversarial_metadata_unavailable": reason in {"adversarial_metadata_unavailable", "adversarial_no_candidates"},
        })
        return Phase1Output(
            candidate_index=-1,
            geodetic_anchor=anchor_payload,
            reference_image=None,
            similarity_score=0.0,
        )

    def _maybe_apply_forge_reference_selection(
        self,
        *,
        base_output: Phase1Output,
        raw_retrieval_results: List[Any],
        ref_map: Dict[int, Any],
        ref_name_map: Dict[int, str],
        ref_anchor_map: Dict[int, Dict[str, Any]],
        frame: Dict[str, Any],
    ) -> Phase1Output:
        cfg = self._adversarial_config
        if not bool(cfg.get("is_adversarial_enabled", False)):
            return base_output
        frame_meta = frame.get("metadata", {}) if isinstance(frame.get("metadata", {}), dict) else {}
        dataset_name = str(frame_meta.get("dataset", "") or "").lower()
        # Treatment assignment happens before any counter moves, so the attacked
        # counters keep counting attacked frames only.
        inject_rate = float(cfg.get("adversarial_inject_rate", 1.0))
        assigned_attack = _forge_attack_assignment(
            cfg.get("adversarial_seed"),
            str(frame.get("image_name", "") or ""),
            inject_rate,
        )
        if not assigned_attack:
            self._stats["num_adversarial_control_frames"] = int(self._stats.get("num_adversarial_control_frames", 0)) + 1
            if isinstance(base_output.geodetic_anchor, dict):
                base_output.geodetic_anchor.update({
                    "_adversarial_assigned_attack": False,
                    "_adversarial_attacked": False,
                    "_is_adversarial": False,
                })
            return base_output
        policy = str(cfg.get("adversarial_candidate_policy", "force_rank_k") or "force_rank_k")
        self._stats["num_adversarial_samples"] = int(self._stats.get("num_adversarial_samples", 0)) + 1
        policy_counter = ADVERSARIAL_POLICY_COUNTER_KEYS.get(policy)
        if policy_counter:
            self._stats[policy_counter] = int(self._stats.get(policy_counter, 0)) + 1

        candidates = self._build_forge_candidate_records(
            raw_retrieval_results=raw_retrieval_results,
            ref_map=ref_map,
            ref_name_map=ref_name_map,
            ref_anchor_map=ref_anchor_map,
            frame=frame,
            selected_candidate_index=int(base_output.candidate_index),
            dataset_name=dataset_name,
        )
        selection = _select_forge_candidate(cfg, candidates, dataset_name)
        if not bool(selection.get("ok", False)):
            reason = str(selection.get("reason", "adversarial_no_eligible_wrong_reference"))
            self._stats["num_adversarial_fail_closed"] = int(self._stats.get("num_adversarial_fail_closed", 0)) + 1
            if bool(selection.get("unsupported_condition", False)) or "unsupported" in reason:
                self._stats["num_adversarial_unsupported_condition"] = int(self._stats.get("num_adversarial_unsupported_condition", 0)) + 1
            if bool(selection.get("metadata_unavailable", False)) or "metadata" in reason:
                self._stats["num_adversarial_metadata_unavailable"] = int(self._stats.get("num_adversarial_metadata_unavailable", 0)) + 1
            return self._make_adversarial_fail_closed_output(base_output=base_output, reason=reason, selection=selection)

        candidate = selection["candidate"]
        original_anchor = dict(base_output.geodetic_anchor) if isinstance(base_output.geodetic_anchor, dict) else {}
        anchor_payload = dict(candidate.get("anchor") or {})
        for key, value in original_anchor.items():
            if key.startswith("_raw_") or key.startswith("_retrieval_protocol_"):
                anchor_payload[key] = value
        original_name = str(original_anchor.get("_selected_reference_name", ""))
        forced_name = str(candidate.get("reference_name", ""))
        selected_rank = int(candidate.get("rank", -1))
        distance_to_gt_m = _safe_float(candidate.get("distance_to_gt_m", np.nan), default=float("nan"))
        offset_m = _safe_anchor_distance_m(anchor_payload, original_anchor)
        if not np.isfinite(offset_m):
            offset_m = distance_to_gt_m
        threshold_m = float(DATASET_RETRIEVAL_THRESHOLD_M.get(dataset_name, 20.0))
        is_wrong = bool(np.isfinite(distance_to_gt_m) and distance_to_gt_m > threshold_m)
        if is_wrong:
            self._stats["num_wrong_reference"] = int(self._stats.get("num_wrong_reference", 0)) + 1
        anchor_payload.update({
            "_candidate_quality_score": _safe_float(candidate.get("similarity_score", np.nan), default=float("nan")),
            "_candidate_rank_selected": selected_rank,
            "_selected_reference_name": forced_name,
            "_adversarial_mode": cfg.get("adversarial_mode", "wrong_reference"),
            "_adversarial_condition": cfg.get("adversarial_condition", ""),
            "_adversarial_candidate_policy": cfg.get("adversarial_candidate_policy", ""),
            "_adversarial_candidate_rank": cfg.get("adversarial_candidate_rank"),
            "_adversarial_candidate_pool_size": int(selection.get("candidate_pool_size", len(candidates)) or 0),
            "_adversarial_eligible_wrong_count": int(selection.get("eligible_wrong_count", 0) or 0),
            "_adversarial_selection_reason": str(selection.get("reason", "")),
            "_adversarial_forced_reference_name": forced_name,
            "_adversarial_original_reference_name": original_name,
            "_adversarial_forced_distance_to_gt_m": float(distance_to_gt_m) if np.isfinite(distance_to_gt_m) else float("nan"),
            "_adversarial_offset_m": float(offset_m) if np.isfinite(offset_m) else float("nan"),
            "_is_adversarial": True,
            "_adversarial_assigned_attack": True,
            "_adversarial_attacked": True,
            "_is_wrong_reference": is_wrong,
            "_adversarial_fail_closed": False,
            "_adversarial_failure_reason": "",
        })
        return Phase1Output(
            candidate_index=int(candidate["candidate_index"]),
            geodetic_anchor=anchor_payload,
            reference_image=candidate.get("reference_image"),
            similarity_score=_safe_float(candidate.get("similarity_score", np.nan), default=float("nan")),
        )

    def _run_phase1_retrieval(
        self,
        frame: Dict[str, Any],
        pipeline: Dict,
        query_image: Optional[np.ndarray] = None,
        telemetry: Optional[Dict[str, Any]] = None,
    ) -> Optional[Phase1Output]:
        """
        Phase-1 contract.

        Returns only candidate index + geodetic anchor + selected reference tile.
        """
        # Oracle mode: use dataset-paired reference tile and GT anchor.
        def _finite_or_nan(value: Any) -> float:
            try:
                v = float(value)
            except (TypeError, ValueError):
                return float("nan")
            return v if np.isfinite(v) else float("nan")

        if self._eval_mode == 'oracle_reference':
            ref_image = frame.get('reference_image')
            gt = frame.get('gt_pose', {})
            if ref_image is None:
                return None
            return Phase1Output(
                candidate_index=0,
                geodetic_anchor={
                    'lat': float(gt.get('lat', 0.0)),
                    'lon': float(gt.get('lon', 0.0)),
                    'alt': float(gt.get('alt', 0.0)),
                },
                reference_image=ref_image,
                similarity_score=1.0,
            )

        retriever = pipeline.get('retrieval')
        if retriever is not None and retriever.database.size > 0:
            query = query_image if query_image is not None else frame.get('query_image')
            if query is not None:
                frame_meta_pre = frame.get("metadata", {}) if isinstance(frame.get("metadata"), dict) else {}
                if (frame_meta_pre.get("dataset") == "uav_visloc"
                        and self._uav_visloc_protocol == "intra_flight"):
                    # Global index + same-flight answer: over-fetch so the
                    # same-flight list below has full top_k depth. At K=5
                    # straight, 37.3% of frames had zero same-flight
                    # candidates (flight 07 lost every frame).
                    base_k = int(getattr(retriever, "_top_k", 5) or 5)
                    fetched_results = retriever.retrieve_top_k(query, top_k=base_k * 20)
                    retrieval_results = list(fetched_results)
                    # Raw metrics keep pre-filter descriptor-ranking semantics
                    # at the ORIGINAL K, so hit rates and the FORGE shortlist
                    # are unchanged by the over-fetch.
                    raw_retrieval_results = list(fetched_results[:base_k])
                else:
                    retrieval_results = retriever.retrieve_top_k(query)
                    raw_retrieval_results = list(retrieval_results)
                ref_map = pipeline.get('retrieval_reference_images', {})
                ref_name_map = pipeline.get('retrieval_reference_names', {})
                ref_anchor_map = pipeline.get('retrieval_reference_anchors', {})
                frame_meta = frame.get("metadata", {}) if isinstance(frame.get("metadata"), dict) else {}
                query_flight = str(frame_meta.get("location", "")) or _flight_id_from_ref_name(
                    str(frame.get("image_name", ""))
                )
                raw_top1 = raw_retrieval_results[0] if len(raw_retrieval_results) > 0 else None
                raw_topk_indices: List[int] = []
                for rr in raw_retrieval_results:
                    try:
                        raw_topk_indices.append(int(rr.candidate_index))
                    except (TypeError, ValueError):
                        continue

                def _candidate_allowed(candidate_idx: int) -> bool:
                    if frame_meta.get("dataset") != "uav_visloc":
                        return True
                    if self._uav_visloc_protocol != "intra_flight" or not query_flight:
                        return True
                    ref_flight = _flight_id_from_ref_name(str(ref_name_map.get(int(candidate_idx), "")))
                    return bool(ref_flight and ref_flight == query_flight)

                filtered_out = 0
                if frame_meta.get("dataset") == "uav_visloc" and self._uav_visloc_protocol == "intra_flight":
                    filtered_results = []
                    for rr in retrieval_results:
                        try:
                            candidate_idx = int(rr.candidate_index)
                        except (TypeError, ValueError):
                            filtered_out += 1
                            continue
                        if _candidate_allowed(candidate_idx):
                            filtered_results.append(rr)
                        else:
                            filtered_out += 1
                    if filtered_out:
                        self._stats["retrieval_protocol_filtered_candidates_count"] = (
                            int(self._stats.get("retrieval_protocol_filtered_candidates_count", 0)) + filtered_out
                        )
                    base_k = int(getattr(retriever, "_top_k", 5) or 5)
                    retrieval_results = filtered_results[:base_k]

                # Filtering cross-flight candidates is correct enforcement of
                # intra_flight, not a violation; the violation check (using a
                # cross-flight reference) lives in
                # resolve_uav_visloc_protocol_fields on selected_ref_flight.
                protocol_violation = False
                selected_top1 = retrieval_results[0] if len(retrieval_results) > 0 else None
                verify_top_k = bool(pipeline.get("config", {}).get("retrieval_verify_top_k", True))
                rerank_margin_score = _safe_float(
                    pipeline.get("config", {}).get("rerank_margin_score", 0.0),
                    default=0.0,
                )
                top1_verified_score: Optional[float] = None
                best_score = float("-inf")
                best_output = None
                fast_matcher = pipeline.get('matcher')
                best_rank = -1
                for rank_idx, rr in enumerate(retrieval_results):
                    try:
                        candidate_idx = int(rr.candidate_index)
                    except (TypeError, ValueError):
                        continue
                    ref_img = ref_map.get(candidate_idx)
                    if ref_img is None:
                        continue
                    if isinstance(ref_img, str):
                        import cv2
                        ref_img = cv2.imread(ref_img, cv2.IMREAD_COLOR)
                        if ref_img is not None:
                            ref_img = cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB)
                    verify_inliers = 0
                    verify_conf = 0.0
                    verify_shape = np.nan
                    verify_reproj = float("inf")
                    if verify_top_k and fast_matcher is not None:
                        try:
                            mr = fast_matcher.match(query, ref_img)
                            verify_inliers = int(mr.num_inliers)
                            if len(mr.confidence) > 0:
                                verify_conf = float(np.mean(mr.confidence))
                            verify_shape = self._resolve_shape_confidence(
                                match_result=mr,
                                query_image=query,
                                ref_image=ref_img,
                            )
                            verify_reproj = self._compute_reprojection_residual_px(mr)
                        except Exception as exc:
                            self._stats["phase1_verify_exceptions"] += 1
                            if self._verbose:
                                logger.debug("Phase1 verifier exception: %s", exc, exc_info=True)
                            verify_inliers = 0
                            verify_conf = 0.0
                            verify_shape = np.nan
                            verify_reproj = float("inf")
                    if verify_top_k:
                        candidate_score = self._candidate_quality_score(
                            inliers=verify_inliers,
                            shape_conf=verify_shape,
                            match_conf=verify_conf,
                            retrieval_sim=float(rr.similarity_score),
                            reproj_residual_px=verify_reproj,
                        )
                    else:
                        # Fast path: pick by descriptor score only.
                        candidate_score = float(rr.similarity_score)
                    if self._verbose:
                        logger.debug(
                            "TopK candidate idx=%s sim=%.4f verify_inliers=%d verify_conf=%.4f score=%.4f",
                            candidate_idx,
                            float(rr.similarity_score),
                            verify_inliers,
                            verify_conf,
                            candidate_score,
                        )
                    if rank_idx == 0:
                        top1_verified_score = float(candidate_score)
                    if not self._challenger_passes_margin(
                        rank_idx, float(candidate_score), top1_verified_score, rerank_margin_score
                    ):
                        continue
                    if candidate_score > best_score:
                        best_score = candidate_score
                        best_rank = int(rank_idx)
                        telemetry_alt = _finite_or_nan((telemetry or {}).get('alt', np.nan))
                        retrieval_alt = _finite_or_nan(rr.geodetic_anchor.get('alt', np.nan))
                        anchor_alt = retrieval_alt if np.isfinite(retrieval_alt) else telemetry_alt
                        stored_anchor = ref_anchor_map.get(candidate_idx, {})
                        anchor_payload = dict(stored_anchor) if isinstance(stored_anchor, dict) else {}
                        anchor_payload.update({
                            'lat': float(rr.geodetic_anchor.get('lat', anchor_payload.get('lat', 0.0))),
                            'lon': float(rr.geodetic_anchor.get('lon', anchor_payload.get('lon', 0.0))),
                            'alt': anchor_alt,
                        })
                        best_output = Phase1Output(
                            candidate_index=candidate_idx,
                            geodetic_anchor=anchor_payload,
                            reference_image=ref_img,
                            similarity_score=float(rr.similarity_score),
                        )
                if best_output is not None:
                    if isinstance(best_output.geodetic_anchor, dict):
                        raw_scores = [float(rr.similarity_score) for rr in raw_retrieval_results if hasattr(rr, "similarity_score")]
                        best_output.geodetic_anchor["_candidate_quality_score"] = float(best_score)
                        best_output.geodetic_anchor["_candidate_rank_selected"] = 0 if best_rank < 0 else best_rank
                        best_output.geodetic_anchor["_selected_reference_name"] = str(
                            ref_name_map.get(int(best_output.candidate_index), f"ref_{int(best_output.candidate_index):06d}")
                        )
                        best_output.geodetic_anchor["_raw_topk_candidate_indices"] = [int(v) for v in raw_topk_indices]
                        best_output.geodetic_anchor["_raw_topk_similarity_scores"] = raw_scores
                        best_output.geodetic_anchor["_raw_top2_similarity"] = raw_scores[1] if len(raw_scores) > 1 else float("nan")
                        best_output.geodetic_anchor["_retrieval_protocol_violation"] = bool(protocol_violation)
                        best_output.geodetic_anchor["_retrieval_protocol_filtered_candidates_count"] = int(filtered_out)
                        if raw_top1 is not None:
                            best_output.geodetic_anchor["_raw_top1_candidate_index"] = int(raw_top1.candidate_index)
                            best_output.geodetic_anchor["_raw_top1_reference_name"] = str(
                                ref_name_map.get(int(raw_top1.candidate_index), f"ref_{int(raw_top1.candidate_index):06d}")
                            )
                            best_output.geodetic_anchor["_raw_top1_lat"] = float(raw_top1.geodetic_anchor.get("lat", np.nan))
                            best_output.geodetic_anchor["_raw_top1_lon"] = float(raw_top1.geodetic_anchor.get("lon", np.nan))
                            best_output.geodetic_anchor["_raw_top1_similarity"] = float(raw_top1.similarity_score)
                    return self._maybe_apply_forge_reference_selection(
                        base_output=best_output,
                        raw_retrieval_results=raw_retrieval_results,
                        ref_map=ref_map,
                        ref_name_map=ref_name_map,
                        ref_anchor_map=ref_anchor_map,
                        frame=frame,
                    )

                # Descriptor hit but geometric verifier rejected every candidate.
                # Experimental official-fail-closed pipelines must reject here
                # instead of accepting descriptor top-1.
                if bool(pipeline.get("config", {}).get("official_fail_closed_retrieval_verifier", False)):
                    self._stats["official_fail_closed_retrieval_reject_frames"] = (
                        int(self._stats.get("official_fail_closed_retrieval_reject_frames", 0)) + 1
                    )
                    return None

                # Legacy strict E2E behavior: fall back to raw top-1 (not GT oracle pairing).
                if selected_top1 is not None:
                    try:
                        raw_idx = int(selected_top1.candidate_index)
                    except (TypeError, ValueError):
                        raw_idx = -1
                    raw_ref = ref_map.get(raw_idx) if raw_idx >= 0 else None
                    if isinstance(raw_ref, str):
                        import cv2
                        raw_ref = cv2.imread(raw_ref, cv2.IMREAD_COLOR)
                        if raw_ref is not None:
                            raw_ref = cv2.cvtColor(raw_ref, cv2.COLOR_BGR2RGB)
                    if raw_ref is not None:
                        telemetry_alt = _finite_or_nan((telemetry or {}).get('alt', np.nan))
                        retrieval_alt = _finite_or_nan(selected_top1.geodetic_anchor.get('alt', np.nan))
                        anchor_alt = retrieval_alt if np.isfinite(retrieval_alt) else telemetry_alt
                        stored_anchor = ref_anchor_map.get(raw_idx, {})
                        anchor_payload = dict(stored_anchor) if isinstance(stored_anchor, dict) else {}
                        anchor_payload.update({
                            'lat': float(selected_top1.geodetic_anchor.get('lat', anchor_payload.get('lat', 0.0))),
                            'lon': float(selected_top1.geodetic_anchor.get('lon', anchor_payload.get('lon', 0.0))),
                            'alt': anchor_alt,
                        })
                        fallback = Phase1Output(
                            candidate_index=raw_idx,
                            geodetic_anchor=anchor_payload,
                            reference_image=raw_ref,
                            similarity_score=float(selected_top1.similarity_score),
                        )
                        if isinstance(fallback.geodetic_anchor, dict):
                            raw_scores = [float(rr.similarity_score) for rr in raw_retrieval_results if hasattr(rr, "similarity_score")]
                            fallback.geodetic_anchor["_candidate_quality_score"] = float(selected_top1.similarity_score)
                            fallback.geodetic_anchor["_candidate_rank_selected"] = 0
                            fallback.geodetic_anchor["_selected_reference_name"] = str(
                                ref_name_map.get(raw_idx, f"ref_{raw_idx:06d}")
                            )
                            fallback.geodetic_anchor["_raw_topk_candidate_indices"] = [int(v) for v in raw_topk_indices]
                            fallback.geodetic_anchor["_raw_topk_similarity_scores"] = raw_scores
                            fallback.geodetic_anchor["_raw_top2_similarity"] = raw_scores[1] if len(raw_scores) > 1 else float("nan")
                            fallback.geodetic_anchor["_retrieval_protocol_violation"] = bool(protocol_violation)
                            fallback.geodetic_anchor["_retrieval_protocol_filtered_candidates_count"] = int(filtered_out)
                            if raw_top1 is not None:
                                fallback.geodetic_anchor["_raw_top1_candidate_index"] = int(raw_top1.candidate_index)
                                fallback.geodetic_anchor["_raw_top1_reference_name"] = str(
                                    ref_name_map.get(int(raw_top1.candidate_index), f"ref_{int(raw_top1.candidate_index):06d}")
                                )
                                fallback.geodetic_anchor["_raw_top1_lat"] = float(raw_top1.geodetic_anchor.get("lat", np.nan))
                                fallback.geodetic_anchor["_raw_top1_lon"] = float(raw_top1.geodetic_anchor.get("lon", np.nan))
                                fallback.geodetic_anchor["_raw_top1_similarity"] = float(raw_top1.similarity_score)
                            fallback.geodetic_anchor["_retrieval_fallback"] = "raw_top1_after_verify_exhausted"
                        return self._maybe_apply_forge_reference_selection(
                            base_output=fallback,
                            raw_retrieval_results=raw_retrieval_results,
                            ref_map=ref_map,
                            ref_name_map=ref_name_map,
                            ref_anchor_map=ref_anchor_map,
                            frame=frame,
                        )

        # End-to-end fallback only when retrieval backend unavailable.
        if self._strict_end2end:
            raise RuntimeError(
                "strict-end2end enabled: retrieval failed; dataset-paired reference fallback is forbidden."
            )
        self._stats["retrieval_oracle_fallback_frames"] += 1
        ref_image = frame.get('reference_image')
        gt = frame.get('gt_pose', {})
        if ref_image is None:
            return None
        telemetry_alt = _finite_or_nan((telemetry or {}).get('alt', np.nan))
        return Phase1Output(
            candidate_index=0,
            geodetic_anchor={
                'lat': float(gt.get('lat', 0.0)),
                'lon': float(gt.get('lon', 0.0)),
                'alt': telemetry_alt,
            },
            reference_image=ref_image,
        )

    def _resolve_run_output(self, dataset_name: str) -> Path:
        """
        Canonical per-run output layout:
            <output>/<dataset>/<pipeline>/<run_id>/
        """
        run_dir = self._output_dir / dataset_name / self._pipeline_name / self._run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        self._current_run_dir = run_dir
        return run_dir

    def _dump_match_evidence(self, frame: Dict[str, Any], match_result: Any) -> None:
        """One npz per frame: the correspondence set behind the scalar summaries.

        Written only when the config sets dump_match_evidence. Failure to write
        never touches the benchmark result - evidence collection must not be
        able to change the numbers it is evidence for.
        """
        if not bool(self._config.get("dump_match_evidence", False)):
            return
        run_dir = getattr(self, "_current_run_dir", None)
        if run_dir is None or match_result is None:
            return
        try:
            q = np.asarray(getattr(match_result, "query_keypoints", None), dtype=np.float32)
            r = np.asarray(getattr(match_result, "ref_keypoints", None), dtype=np.float32)
            if q is None or r is None or q.size == 0 or q.shape != r.shape:
                return
            conf = np.asarray(getattr(match_result, "confidence", np.ones(len(q))), dtype=np.float32)
            mask = getattr(match_result, "inlier_mask", None)
            mask = (np.asarray(mask, dtype=bool) if mask is not None
                    else np.ones(len(q), dtype=bool))
            H = getattr(match_result, "homography", None)
            if H is not None:
                H = np.asarray(H, dtype=np.float64)
                ones = np.ones((len(q), 1), dtype=np.float64)
                proj = (H @ np.hstack([q.astype(np.float64), ones]).T).T
                denom = proj[:, 2:3]
                denom[np.abs(denom) < 1e-12] = 1e-12
                residual = (proj[:, :2] / denom - r.astype(np.float64)).astype(np.float32)
            else:
                residual = np.full_like(q, np.nan, dtype=np.float32)
            out_dir = run_dir / "match_evidence"
            out_dir.mkdir(parents=True, exist_ok=True)
            stem = str(frame.get("image_name", "unknown")).replace("/", "__")
            np.savez_compressed(
                out_dir / (stem + ".npz"),
                query_xy=q, ref_xy=r, confidence=conf,
                inlier_mask=mask, residual_xy=residual,
                homography=(H if H is not None else np.full((3, 3), np.nan)),
            )
        except Exception:
            return

    def _run_retrieval_only_dataset(self, dataset_name: str, dataset, pipeline: Dict, ds_output: Path) -> Dict[str, Any]:
        """Run Phase-1 retrieval only and write diagnostic artifacts."""
        ds_output = Path(ds_output)
        ds_output.mkdir(parents=True, exist_ok=True)
        retriever = pipeline.get("retrieval")
        if retriever is None:
            raise RuntimeError("retrieval-only mode requires a retrieval backend.")
        if getattr(retriever, "database", None) is None or int(getattr(retriever.database, "size", 0)) <= 0:
            raise RuntimeError("retrieval-only mode requires a prepared retrieval database.")

        top_k = _safe_int(
            (pipeline.get("config") or {}).get("retrieval_top_k", getattr(retriever, "_top_k", 5)),
            default=5,
            min_value=1,
        )
        self._stats["retrieval_top_k"] = int(top_k)
        ref_names = pipeline.get("retrieval_reference_names", {}) or {}
        ref_anchors = pipeline.get("retrieval_reference_anchors", {}) or {}
        results: List[Dict[str, Any]] = []
        start_t = time.time()

        def _anchor_error(anchor: Any, row: Dict[str, Any]) -> float:
            if not isinstance(anchor, dict):
                return float("nan")
            lat = _safe_float(anchor.get("lat", np.nan), default=float("nan"))
            lon = _safe_float(anchor.get("lon", np.nan), default=float("nan"))
            gt_lat = _safe_float(row.get("gt_lat", np.nan), default=float("nan"))
            gt_lon = _safe_float(row.get("gt_lon", np.nan), default=float("nan"))
            if not all(np.isfinite(v) for v in [lat, lon, gt_lat, gt_lon]):
                return float("nan")
            return self._latlon_distance_m(lat, lon, gt_lat, gt_lon)

        with open(ds_output / "frame_diagnostics.jsonl", "w", encoding="utf-8") as diag_fp:
            for frame_index, frame in enumerate(dataset):
                query = frame.get("query_image")
                gt = frame.get("gt_pose", {}) or {}
                frame_meta = frame.get("metadata", {}) if isinstance(frame.get("metadata"), dict) else {}
                image_name = str(frame.get("image_name", ""))
                t0 = time.time()
                retrieval_results = retriever.retrieve_top_k(query, top_k=top_k) if query is not None else []
                retrieval_ms = float(
                    retrieval_results[0].timing_ms
                    if retrieval_results and hasattr(retrieval_results[0], "timing_ms")
                    else (time.time() - t0) * 1000.0
                )
                raw_indices = [
                    _safe_int(getattr(rr, "candidate_index", -1), default=-1)
                    for rr in retrieval_results
                ]
                raw_indices = [idx for idx in raw_indices if idx >= 0]
                raw_top1 = retrieval_results[0] if retrieval_results else None
                raw_idx = _safe_int(getattr(raw_top1, "candidate_index", -1), default=-1) if raw_top1 is not None else -1
                selected_anchor = dict(getattr(raw_top1, "geodetic_anchor", {}) or {}) if raw_top1 is not None else {}
                selected_ref = str(ref_names.get(raw_idx, f"ref_{raw_idx:06d}" if raw_idx >= 0 else ""))
                row = {
                    "image_name": image_name,
                    "frame_index": int(frame_index),
                    "retrieval_only": True,
                    "success": bool(raw_top1 is not None),
                    "pose_success": False,
                    "accepted_for_metrics": False,
                    "failure_reason": "" if raw_top1 is not None else "retrieval_failed",
                    "bottleneck_label": "retrieval_only",
                    "phase1_source": "retrieval_only",
                    "pred_lat": _safe_float(selected_anchor.get("lat", np.nan), default=float("nan")),
                    "pred_lon": _safe_float(selected_anchor.get("lon", np.nan), default=float("nan")),
                    "pred_alt": float("nan"),
                    "pred_heading": float("nan"),
                    "gt_lat": _safe_float(gt.get("lat", np.nan), default=float("nan")),
                    "gt_lon": _safe_float(gt.get("lon", np.nan), default=float("nan")),
                    "gt_alt": _safe_float(gt.get("alt", np.nan), default=float("nan")),
                    "gt_heading": _safe_float(gt.get("yaw", np.nan), default=float("nan")),
                    "gt_pitch": _safe_float(gt.get("pitch", np.nan), default=float("nan")),
                    "geodetic_anchor": selected_anchor,
                    "raw_retrieval_anchor": selected_anchor,
                    "selected_reference_name": selected_ref,
                    "raw_top1_reference_name": selected_ref,
                    "retrieval_candidate_index": int(raw_idx),
                    "raw_retrieval_candidate_index": int(raw_idx),
                    "raw_retrieval_topk_indices": raw_indices,
                    "retrieval_similarity_score": _safe_float(getattr(raw_top1, "similarity_score", np.nan), default=float("nan")) if raw_top1 is not None else float("nan"),
                    "raw_retrieval_similarity_score": _safe_float(getattr(raw_top1, "similarity_score", np.nan), default=float("nan")) if raw_top1 is not None else float("nan"),
                    "timings": {"retrieval_ms": retrieval_ms, "matching_ms": 0.0, "geometry_ms": 0.0, "fgo_ms": 0.0},
                    "total_time_ms": retrieval_ms,
                    "num_matches": 0,
                    "num_inliers": 0,
                    "homography_valid": False,
                    "flight_id": str(frame_meta.get("location", "")) or _flight_id_from_ref_name(image_name),
                }
                row["retrieval_anchor_error_m"] = _anchor_error(row.get("geodetic_anchor"), row)
                row["raw_retrieval_anchor_error_m"] = row["retrieval_anchor_error_m"]
                if dataset_name == "uav_visloc":
                    row.update(resolve_uav_visloc_protocol_fields(
                        row,
                        protocol=self._uav_visloc_protocol,
                        reference_names=ref_names,
                    ))
                    if bool(row.get("retrieval_protocol_violation", False)):
                        self._stats["retrieval_protocol_violation_count"] = (
                            int(self._stats.get("retrieval_protocol_violation_count", 0)) + 1
                        )
                    for threshold_m in (20, 50, 150, 300):
                        selected_hit = int(
                            np.isfinite(row.get("retrieval_anchor_error_m", np.nan))
                            and float(row.get("retrieval_anchor_error_m")) <= float(threshold_m)
                        )
                        row[f"selected_anchor_hit_at_{threshold_m}m"] = selected_hit
                        row[f"raw_top1_anchor_hit_at_{threshold_m}m"] = selected_hit
                    for k in (5, 10, 20, 50):
                        best_err = float("inf")
                        for idx_val in raw_indices[:k]:
                            err = _anchor_error(ref_anchors.get(_safe_int(idx_val, default=-1)), row)
                            if np.isfinite(err):
                                best_err = min(best_err, err)
                        for threshold_m in (20, 50, 150, 300):
                            row[f"raw_top{k}_anchor_hit_at_{threshold_m}m"] = int(best_err <= threshold_m)
                gt_tile = self._find_gt_tile_name(
                    float(row.get("gt_lat", np.nan)),
                    float(row.get("gt_lon", np.nan)),
                    dataset_name,
                    ref_anchors,
                    query_name=str(row.get("image_name", "")),
                )
                row["gt_tile_name"] = gt_tile or ""
                if gt_tile:
                    sel_name = str(row.get("selected_reference_name", ""))
                    raw_name = str(row.get("raw_top1_reference_name", ""))
                    if dataset_name == "vpair":
                        row["tile_identity_correct"] = int(Path(sel_name).stem == gt_tile)
                        row["raw_top1_tile_identity_correct"] = int(Path(raw_name).stem == gt_tile)
                    else:
                        row["tile_identity_correct"] = int(sel_name == gt_tile)
                        row["raw_top1_tile_identity_correct"] = int(raw_name == gt_tile)
                else:
                    row["tile_identity_correct"] = 0
                    row["raw_top1_tile_identity_correct"] = 0
                results.append(row)
                diag_fp.write(json.dumps(row, cls=NpEncoder) + "\n")

        retrieval_diagnostics = compute_retrieval_diagnostics(
            results,
            ref_anchors,
            dataset_name=dataset_name,
        )
        uav_retrieval_diagnostics = retrieval_diagnostics if dataset_name == "uav_visloc" else None
        pipeline_meta = resolve_pipeline_metadata(self._pipeline_name)
        global_diag = (retrieval_diagnostics or {}).get("global", {})
        paper_metrics = {
            **pipeline_meta,
            "retrieval_only": True,
            "run_integrity_for_official_metrics": False,
            "uav_visloc_protocol": self._uav_visloc_protocol if dataset_name == "uav_visloc" else None,
            "uav_tile_size_px": self._stats.get("uav_tile_size_px") if dataset_name == "uav_visloc" else None,
            "uav_tile_step_px": self._stats.get("uav_tile_step_px") if dataset_name == "uav_visloc" else None,
            "retrieval_top_k": int(top_k),
            "num_frames": len(results),
            "num_success": sum(1 for r in results if bool(r.get("success", False))),
            "num_pose_success": 0,
            "num_accepted_for_metrics": 0,
            "M_ME": None,
            "E_ME": None,
            "HR": None,
            # Per-dataset R@1 threshold (20m vpair/avl, 150m uav_visloc);
            # hardcoding 150m here inflated vpair/avl retrieval-only Recall@1.
            "Recall@1": global_diag.get(
                f"selected_error_m_hit_at_{int(DATASET_RETRIEVAL_THRESHOLD_M.get(dataset_name, 20.0))}m"
            ),
            "selected_anchor_hit_at_threshold": global_diag.get(
                f"selected_error_m_hit_at_{int(DATASET_RETRIEVAL_THRESHOLD_M.get(dataset_name, 20.0))}m"
            ),
            "raw_retrieval_recall_at1_threshold": global_diag.get(
                f"raw_top1_error_m_hit_at_{int(DATASET_RETRIEVAL_THRESHOLD_M.get(dataset_name, 20.0))}m"
            ),
            "recall_at_1_threshold_m": float(DATASET_RETRIEVAL_THRESHOLD_M.get(dataset_name, 20.0)),
            "uav_raw_top1_hit_at_150m": global_diag.get("raw_top1_error_m_hit_at_150m"),
            "uav_raw_top5_hit_at_150m": global_diag.get("raw_top5_error_m_hit_at_150m"),
            "uav_raw_top10_hit_at_150m": global_diag.get("raw_top10_error_m_hit_at_150m"),
            "uav_raw_top20_hit_at_150m": global_diag.get("raw_top20_error_m_hit_at_150m"),
            "uav_raw_top50_hit_at_150m": global_diag.get("raw_top50_error_m_hit_at_150m"),
            "uav_nearest_tile_oracle_floor_p50_m": global_diag.get("nearest_tile_oracle_floor_m_p50"),
            "retrieval_db_cache_enabled": bool(self._stats.get("retrieval_db_cache_enabled", False)),
            "retrieval_db_cache_hit": bool(self._stats.get("retrieval_db_cache_hit", False)),
            "retrieval_db_cache_key": str(self._stats.get("retrieval_db_cache_key", "")),
            "retrieval_db_cache_rebuild_reason": str(self._stats.get("retrieval_db_cache_rebuild_reason", "")),
            "retrieval_protocol_violation_count": int(self._stats.get("retrieval_protocol_violation_count", 0)),
            "retrieval_protocol_filtered_candidates_count": int(self._stats.get("retrieval_protocol_filtered_candidates_count", 0)),
        }
        with open(ds_output / "results_detailed.json", "w", encoding="utf-8") as f:
            json.dump({"metrics": {}, "predictions": results}, f, indent=2, cls=NpEncoder)
        with open(ds_output / "paper_metrics_summary.json", "w", encoding="utf-8") as f:
            json.dump(paper_metrics, f, indent=2, cls=NpEncoder)
        with open(ds_output / "retrieval_diagnostics.json", "w", encoding="utf-8") as f:
            json.dump(retrieval_diagnostics, f, indent=2, cls=NpEncoder)
        if uav_retrieval_diagnostics is not None:
            with open(ds_output / "uav_visloc_retrieval_diagnostics.json", "w", encoding="utf-8") as f:
                json.dump(uav_retrieval_diagnostics, f, indent=2, cls=NpEncoder)
        write_predictions_csv(ds_output / "predictions.csv", results)
        manifest = self._build_run_manifest_payload(dataset_name)
        write_metrics_summary_json(
            ds_output / "metrics_summary.json",
            dataset=dataset_name,
            pipeline=self._pipeline_name,
            metrics={},
            paper_metrics=paper_metrics,
            results=results,
            manifest=manifest,
            step_metrics={"steps": {"retrieval": {"success_rate": float(np.mean([bool(r.get("success")) for r in results])) if results else 0.0}}},
        )
        self._write_run_manifest(ds_output, dataset_name, run_completed=True)
        total_time = time.time() - start_t
        return {
            "dataset": dataset_name,
            "pipeline": self._pipeline_name,
            "retrieval_only": True,
            "paper_metrics": paper_metrics,
            "run_manifest": self._build_run_manifest_payload(dataset_name),
            "predictions": results,
            "run_dir": str(ds_output),
            "timing": {
                "total_time_s": total_time,
                "avg_time_ms": (total_time / len(results) * 1000.0) if results else 0.0,
            },
        }

    def _write_uav_visloc_anchor_audit(self, dataset, run_dir: Path) -> None:
        """Run UAV-VisLoc anchor audit before benchmark execution."""
        audit_payload: Dict[str, Any]
        if hasattr(dataset, "audit_reference_corpus_anchors"):
            audit_payload = dataset.audit_reference_corpus_anchors()
        else:
            refs = []
            if hasattr(dataset, "iter_reference_corpus"):
                for idx, ref in enumerate(dataset.iter_reference_corpus()):
                    refs.append(ref)
                    if idx >= 499:
                        break
            audit_payload = audit_uav_visloc_reference_anchors(refs)
        audit_payload["uav_visloc_protocol"] = self._uav_visloc_protocol
        with open(Path(run_dir) / "uav_visloc_anchor_audit.json", "w", encoding="utf-8") as f:
            json.dump(audit_payload, f, indent=2, cls=NpEncoder)
        if not bool(audit_payload.get("passed", False)):
            message = (
                "UAV-VisLoc anchor audit failed; see uav_visloc_anchor_audit.json. "
                f"protocol={self._uav_visloc_protocol}"
            )
            if self._strict_components:
                raise RuntimeError(message)
            logger.warning(message)

    @staticmethod
    def _latlon_distance_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Approximate local WGS84 distance in meters."""
        m_per_deg_lat = 111_320.0
        mean_lat = np.radians((lat1 + lat2) * 0.5)
        m_per_deg_lon = 111_320.0 * np.cos(mean_lat)
        dlat_m = (lat1 - lat2) * m_per_deg_lat
        dlon_m = (lon1 - lon2) * m_per_deg_lon
        return float(np.hypot(dlat_m, dlon_m))

    @staticmethod
    def _vpair_index_from_name(name: str) -> Optional[int]:
        """Extract numeric index from VPAir filename."""
        import re
        if not name:
            return None
        stem = Path(str(name)).stem
        nums = re.findall(r"\d+", stem)
        if not nums:
            return None
        try:
            return int(nums[-1])
        except (TypeError, ValueError):
            return None

    @classmethod
    def _vpair_name_hit(cls, query_name: str, ref_name: str, tolerance: int = 1) -> int:
        """Official-style VPAir retrieval hit: |idx(query)-idx(ref)| <= tolerance."""
        q_idx = cls._vpair_index_from_name(query_name)
        r_idx = cls._vpair_index_from_name(ref_name)
        if q_idx is None or r_idx is None:
            return 0
        return int(abs(q_idx - r_idx) <= int(tolerance))

    @classmethod
    def _find_gt_tile_name(
        cls,
        gt_lat: float,
        gt_lon: float,
        dataset_name: str,
        ref_anchors: Dict[int, Dict[str, Any]],
        query_name: str = "",
    ) -> Optional[str]:
        """Find the GT tile name by nearest-anchor lookup.

        For VPAIR, the GT tile is the same-index file in reference_views.
        For AVL/UAV-VisLoc, the GT tile is whichever reference tile has
        its center anchor closest to the query GT position.
        """
        if not (np.isfinite(gt_lat) and np.isfinite(gt_lon)):
            return None
        if dataset_name == "vpair":
            return Path(str(query_name)).stem if query_name else None
        best_name: Optional[str] = None
        best_dist = float("inf")
        for _idx, anchor in ref_anchors.items():
            if not isinstance(anchor, dict):
                continue
            a_lat = float(anchor.get("lat", float("nan")))
            a_lon = float(anchor.get("lon", float("nan")))
            if not (np.isfinite(a_lat) and np.isfinite(a_lon)):
                continue
            dist = cls._latlon_distance_m(a_lat, a_lon, gt_lat, gt_lon)
            if dist < best_dist:
                best_dist = dist
                best_name = str(anchor.get("ref_name", f"ref_{_idx:06d}"))
        return best_name

    @staticmethod
    def _meters_per_degree(lat_deg: float) -> tuple[float, float]:
        """Return (meters/deg latitude, meters/deg longitude)."""
        m_per_deg_lat = 111_320.0
        m_per_deg_lon = 111_320.0 * np.cos(np.radians(lat_deg))
        return m_per_deg_lat, float(max(m_per_deg_lon, 1.0))

    @classmethod
    def _geodetic_to_local_xy(
        cls,
        lat: float,
        lon: float,
        origin_lat: float,
        origin_lon: float,
    ) -> tuple[float, float]:
        """Convert geodetic coordinates to local ENU-like planar metric frame."""
        m_per_deg_lat, m_per_deg_lon = cls._meters_per_degree(origin_lat)
        x_east = (lon - origin_lon) * m_per_deg_lon
        y_north = (lat - origin_lat) * m_per_deg_lat
        return float(x_east), float(y_north)

    @classmethod
    def _local_xy_to_geodetic(
        cls,
        x_east: float,
        y_north: float,
        origin_lat: float,
        origin_lon: float,
    ) -> tuple[float, float]:
        """Convert local planar metric coordinates back to geodetic."""
        m_per_deg_lat, m_per_deg_lon = cls._meters_per_degree(origin_lat)
        lat = origin_lat + (y_north / m_per_deg_lat)
        lon = origin_lon + (x_east / m_per_deg_lon)
        return float(lat), float(lon)

    @staticmethod
    def _build_visual_factor_from_ray_plane(
        match_result,
        camera_matrix: np.ndarray,
        local_position_m: np.ndarray,
        altitude_m: float,
        attitude_rad=None,
    ):
        """
        Build a VisualFactor using analytical ray-plane intersection.
        Uses local metric frame to keep reprojection residuals numerically stable.
        attitude_rad: optional (roll, pitch, yaw) radians used to orient the rays;
        the same mount rotation is stored on the factor so the solver projection
        stays consistent with the construction.
        """
        from modules.fusion.fgo_solver import VisualFactor
        from modules.vision.ray_plane import RayPlaneIntersector

        if match_result is None:
            return None
        q_pts, _r_pts = match_result.get_inlier_matches()
        if q_pts is None or len(q_pts) < 6:
            return None

        intersector = RayPlaneIntersector(camera_matrix=camera_matrix)
        # Pseudo-nadir camera convention: local +Z is up, camera optical axis points down.
        # Proper rotation (det=+1): x_cam->x_world, y_cam->-y_world, z_cam->-z_world.
        R_mount = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]], dtype=np.float64)
        if attitude_rad is not None:
            r, p_, y_ = (float(attitude_rad[0]), float(attitude_rad[1]), float(attitude_rad[2]))
            cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p_), np.sin(p_), np.cos(y_), np.sin(y_)
            Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=np.float64)
            Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=np.float64)
            Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=np.float64)
            R_att = Rz @ Ry @ Rx  # ZYX, matches the solver's _euler_to_rotation
        else:
            R_att = np.eye(3, dtype=np.float64)
        R_cam_to_world = R_att @ R_mount
        camera_center = np.asarray(local_position_m, dtype=np.float64).copy()
        camera_center[2] = max(float(altitude_m), 1.0)

        world_pts, valid_mask = intersector.intersect_batch(
            pixels=np.asarray(q_pts, dtype=np.float64),
            R_cam_to_world=R_cam_to_world,
            camera_center=camera_center,
            z_ground=0.0,
        )
        if valid_mask.sum() < 6:
            return None

        pts2d = np.asarray(q_pts, dtype=np.float64)[valid_mask]
        pts3d = np.asarray(world_pts, dtype=np.float64)[valid_mask]
        ray_lengths = np.linalg.norm(pts3d - camera_center.reshape(1, 3), axis=1)
        zeta = float(np.median(ray_lengths)) if len(ray_lengths) > 0 else float(max(altitude_m, 1.0))
        depth_ratio = zeta / max(float(altitude_m), 1.0)
        sigma_vis = float(np.clip(1.5 * (1.0 + 0.25 * (depth_ratio ** 2)), 1.0, 12.0))

        return VisualFactor(
            points_2d=pts2d,
            points_3d=pts3d,
            R_mount=R_mount,
            camera_matrix=np.asarray(camera_matrix, dtype=np.float64),
            sigma=sigma_vis,
            weight=1.0,
        )

    def _build_run_manifest_payload(self, dataset_name: str) -> Dict[str, Any]:
        """Build reproducibility manifest payload for this run."""
        def _safe_git(cmd: List[str]) -> str:
            try:
                return subprocess.check_output(
                    cmd,
                    text=True,
                    stderr=subprocess.DEVNULL,
                    timeout=5,
                ).strip()
            except Exception:
                return "unknown"

        pipeline_meta = resolve_pipeline_metadata(self._pipeline_name)
        retrieval_oracle_fallback_frames = int(self._stats.get("retrieval_oracle_fallback_frames", 0))
        end2end_oracle_fallback_contaminated = bool(
            self._eval_mode == "end_to_end" and retrieval_oracle_fallback_frames > 0
        )
        oracle_mode = bool(pipeline_meta.get("oracle_mode", False) or self._eval_mode == "oracle_reference")
        native_prior_uses_exact_gt = bool(self._stats.get("native_prior_uses_exact_gt", False))
        native_anyvisloc_schema_ok = bool(self._stats.get("native_anyvisloc_schema_ok", True))
        retrieval_protocol_violation_count = int(
            self._stats.get("retrieval_protocol_violation_count", 0)
        )
        retrieval_protocol_filtered_candidates_count = int(
            self._stats.get("retrieval_protocol_filtered_candidates_count", 0)
        )
        adversarial_enabled = bool(self._adversarial_config.get("is_adversarial_enabled", False))
        adversarial_diagnostic = bool(adversarial_enabled)
        native_integrity_ok = bool(
            pipeline_meta.get("method_family") != "native_anyvisloc"
            or (not native_prior_uses_exact_gt and native_anyvisloc_schema_ok)
        )
        strict_table_tier = str(pipeline_meta.get("comparison_tier", "strict_e2e")) in {"strict_e2e", "wrapper_e2e"}
        run_integrity_for_official_metrics = bool(
            not end2end_oracle_fallback_contaminated
            and not oracle_mode
            and not bool(pipeline_meta.get("uses_oracle", False))
            and not bool(pipeline_meta.get("deprecated_alias", False))
            and strict_table_tier
            and not bool(getattr(self, "_retrieval_only", False))
            and not adversarial_diagnostic
            and native_integrity_ok
            and retrieval_protocol_violation_count == 0
            and retrieval_protocol_filtered_candidates_count == 0
        )
        manifest = {
            "run_id": self._run_id,
            "timestamp": datetime.now().isoformat(),
            "dataset": dataset_name,
            "pipeline": self._pipeline_name,
            "canonical_pipeline": pipeline_meta["canonical_pipeline"],
            "method_family": pipeline_meta["method_family"],
            "native_baseline": pipeline_meta["native_baseline"],
            "oracle_mode": oracle_mode,
            "deprecated_alias": pipeline_meta["deprecated_alias"],
            "comparison_tier": pipeline_meta["comparison_tier"],
            "tile_source": pipeline_meta["tile_source"],
            "attitude_source": pipeline_meta["attitude_source"],
            "altitude_source": pipeline_meta["altitude_source"],
            "pose_head": pipeline_meta["pose_head"],
            "is_native": pipeline_meta["is_native"],
            "is_wrapper": pipeline_meta["is_wrapper"],
            "is_hybrid": pipeline_meta["is_hybrid"],
            "uses_oracle": bool(pipeline_meta["uses_oracle"] or oracle_mode),
            "oracle_modes": pipeline_meta["oracle_modes"],
            "benchmark_family": "forge" if adversarial_diagnostic else pipeline_meta.get("method_family", ""),
            "adversarial_mode": self._adversarial_config.get("adversarial_mode", "none"),
            "adversarial_condition": self._adversarial_config.get("adversarial_condition", "clean"),
            "adversarial_candidate_policy": self._adversarial_config.get("adversarial_candidate_policy", ""),
            "adversarial_candidate_rank": self._adversarial_config.get("adversarial_candidate_rank"),
            "adversarial_seed": self._adversarial_config.get("adversarial_seed"),
            "adversarial_inject_rate": self._adversarial_config.get("adversarial_inject_rate", 1.0),
            "adversarial_official_metrics": False,
            "adversarial_diagnostic": adversarial_diagnostic,
            "official_metric_eligible": run_integrity_for_official_metrics,
            **_final_pose_gate_metadata(self._config),
            "pipeline_config": self._config,
            "eval_mode": self._eval_mode,
            "strict_components": self._strict_components,
            "strict_end2end": self._strict_end2end,
            "retrieval_only": bool(getattr(self, "_retrieval_only", False)),
            "end2end_oracle_fallback_contaminated": end2end_oracle_fallback_contaminated,
            "run_integrity_for_official_metrics": run_integrity_for_official_metrics,
            "telemetry_prior_policy": str(self._stats.get("telemetry_prior_policy", "simulated_uav_telemetry")),
            "native_prior_uses_exact_gt": native_prior_uses_exact_gt,
            "native_anyvisloc_schema_ok": native_anyvisloc_schema_ok,
            "native_yaw_height_status": str(self._stats.get("native_yaw_height_status", "not_applicable")),
            "official_success_uses_gt_position_error": False,
            "diagnostic_threshold_sweeps": {
                "geometry_min_inliers_after_lgi": [4, 6, 8],
                "max_reproj_residual_px": [6, 10, 15],
                "used_for_official_success": False,
            },
            "max_samples": self._max_samples,
            "max_samples_percent": self._max_samples_percent,
            "test_interval": self._test_interval,
            "device": self._device,
            "use_finetuned": self._use_finetuned,
            "uav_visloc_protocol": self._uav_visloc_protocol,
            "uav_tile_size_px": self._stats.get("uav_tile_size_px"),
            "uav_tile_step_px": self._stats.get("uav_tile_step_px"),
            "retrieval_top_k": self._stats.get("retrieval_top_k"),
            "use_reference_preconditioning": bool(self._config.get("use_reference_preconditioning", self._config.get("enable_adhop_preconditioning", False))),
            "reference_preconditioning_phase": "phase2_selected_reference_only",
            "reference_preconditioning_acceptance_policy": {
                "min_inliers": self._config.get("reference_preconditioning_min_inliers", 8),
                "min_inlier_gain": self._config.get("reference_preconditioning_min_inlier_gain", 2),
                "min_score_gain": self._config.get("reference_preconditioning_min_score_gain", 1.0),
                "max_reproj_px": self._config.get("reference_preconditioning_max_reproj_px", self._config.get("max_reproj_residual_px", 10.0)),
                "shape_tolerance": self._config.get("reference_preconditioning_shape_tolerance", 0.03),
            },
            "fgo_backend_policy": str(self._stats.get("fgo_backend_policy", FGO_BACKEND_POLICY_PREFER_SCIPY)),
            "effective_fgo_backend": str(self._stats.get("effective_fgo_backend", "disabled")),
            "fgo_backend_parity_ok": bool(self._stats.get("fgo_backend_parity_ok", False)),
            "fgo_backend_fallback_reason": str(self._stats.get("fgo_backend_fallback_reason", "")),
            "retrieval_db_cache_enabled": bool(self._stats.get("retrieval_db_cache_enabled", False)),
            "retrieval_db_cache_hit": bool(self._stats.get("retrieval_db_cache_hit", False)),
            "retrieval_db_cache_key": str(self._stats.get("retrieval_db_cache_key", "")),
            "retrieval_db_cache_rebuild_reason": str(self._stats.get("retrieval_db_cache_rebuild_reason", "")),
            "retrieval_protocol_violation_count": retrieval_protocol_violation_count,
            "retrieval_protocol_filtered_candidates_count": retrieval_protocol_filtered_candidates_count,
            "uav_selected_query_locations": list(self._stats.get("uav_selected_query_locations", [])),
            "uav_selected_sample_signature": dict(self._stats.get("uav_selected_sample_signature", {})),
            "uav_corpus_ref_flight_hash": str(self._stats.get("uav_corpus_ref_flight_hash", "")),
            "native_anyvisloc_cache_requested": bool(self._stats.get("native_anyvisloc_cache_requested", False)),
            "native_anyvisloc_cache_enabled": bool(self._stats.get("native_anyvisloc_cache_enabled", False)),
            "native_anyvisloc_cache_key": str(self._stats.get("native_anyvisloc_cache_key", "")),
            "native_anyvisloc_cache_hit": bool(self._stats.get("native_anyvisloc_cache_hit", False)),
            "native_anyvisloc_cache_rebuild_reason": str(self._stats.get("native_anyvisloc_cache_rebuild_reason", "")),
            "env": {
                "VPAIR_DATA": os.environ.get("VPAIR_DATA"),
                "AVL_DATA": os.environ.get("AVL_DATA"),
                "VISLOC_DATA": os.environ.get("VISLOC_DATA"),
            },
            "system": {
                "python": sys.version,
                "platform": platform.platform(),
            },
            "git": {
                "branch": _safe_git(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
                "commit": _safe_git(["git", "rev-parse", "HEAD"]),
            },
            "run_stats": self._stats,
        }
        return manifest

    def _write_run_manifest(
        self,
        run_dir: Path,
        dataset_name: str,
        *,
        run_completed: bool = False,
    ) -> None:
        """Write reproducibility manifest for this run."""
        manifest = self._build_run_manifest_payload(dataset_name)
        manifest["run_completed"] = bool(run_completed)
        with open(run_dir / "run_manifest.json", "w") as f:
            json.dump(manifest, f, indent=2, cls=NpEncoder)

    def _resolve_native_anyvisloc_pipeline(self, frame, pipeline):
        """Resolve the correct native AnyVisLoc pipeline for this frame.

        For AVL: returns the single pre-built pipeline.
        For UAV-VisLoc: selects the per-location pipeline.
        For VPair: builds a temporary pipeline from the per-frame reference image.
        """
        native_pipeline = pipeline.get("native_pipeline")
        if native_pipeline is not None:
            return native_pipeline

        location_pipelines = pipeline.get("native_location_pipelines")
        if location_pipelines:
            metadata = frame.get("metadata", {}) or {}
            location = str(metadata.get("location", ""))
            if location in location_pipelines:
                return location_pipelines[location]
            for loc_key in sorted(location_pipelines.keys()):
                if location.endswith(loc_key) or loc_key.endswith(location):
                    return location_pipelines[loc_key]
            logger.warning("native_anyvisloc: no pipeline for location=%s (available: %s)",
                           location, sorted(location_pipelines.keys()))
            return None

        if pipeline.get("native_gallery_mode"):
            import cv2
            ref_path = frame.get("reference_path")
            if ref_path is None or not Path(str(ref_path)).exists():
                data_root = Path(pipeline.get("native_vpair_data_root", ""))
                ref_dir = data_root / "reference_views"
                image_name = frame.get("image_name", "")
                ref_path = ref_dir / image_name
                if not ref_path.exists():
                    return None

            ref_image = cv2.imread(str(ref_path))
            if ref_image is None:
                return None

            gsd = pipeline.get("native_vpair_gsd", 0.3)
            flat_dsm = np.zeros(ref_image.shape[:2], dtype=np.float32)
            region = "vpair_gallery"
            ref_type = pipeline.get("native_opt", SimpleNamespace(Ref_type="HIGH")).Ref_type
            gt = frame.get("gt_pose", {}) or {}
            gt_lon = float(gt.get("lon", 0.0))
            gt_lat = float(gt.get("lat", 0.0))
            from pyproj import Transformer
            utm_zone, northern = self._utm_zone_from_lon(gt_lon)
            hemisphere = "N" if northern else "S"
            try:
                trf = Transformer.from_crs("epsg:4326", f"epsg:326{utm_zone:02d}", always_xy=True)
                origin_x, origin_y = trf.transform(gt_lon, gt_lat)
            except Exception:
                origin_x, origin_y = 0.0, float(ref_image.shape[0]) * gsd
            origin_x -= (ref_image.shape[1] / 2.0) * gsd
            origin_y += (ref_image.shape[0] / 2.0) * gsd

            region_config = pipeline.get("native_region_config", {}).copy()
            region_config[f"{region}_UTM_SYSTEM"] = f"{utm_zone:02d}{hemisphere}"
            region_config[f"{region}_{ref_type}_REF_initialX"] = origin_x
            region_config[f"{region}_{ref_type}_REF_initialY"] = origin_y
            region_config[f"{region}_{ref_type}_REF_resolution"] = gsd
            region_config[f"{region}_{ref_type}_DSM_resolution"] = gsd
            region_config[f"{region}_{ref_type}_REF_COORDINATE"] = [0, 0]
            region_config[f"{region}_{ref_type}_DSM_COORDINATE"] = [0, 0]

            NativeAnyVisLocPipeline = pipeline.get("native_NativeAnyVisLocPipeline")
            native_opt = pipeline.get("native_opt")
            method_dict = pipeline.get("native_method_dict")
            if NativeAnyVisLocPipeline is None or native_opt is None or method_dict is None:
                return None

            return NativeAnyVisLocPipeline(
                opt=native_opt, config=region_config, region=region,
                ref_image=ref_image, dsm_image=flat_dsm, ref_resolution=gsd,
                method_dict=method_dict, gsd_for_solver=gsd, min_inliers_solver=6,
            )

        return None

    def _process_frame_native_anyvisloc(
        self,
        frame: Dict[str, Any],
        pipeline: Dict[str, Any],
        result: Dict[str, Any],
        start_total: float,
        frame_index: int = 0,
        bias_drift: list | None = None,
    ) -> Dict[str, Any]:
        """Process one frame through native AnyVisLoc, not HD4 homography wrapper."""
        native_pipeline = self._resolve_native_anyvisloc_pipeline(frame, pipeline)
        if native_pipeline is None:
            result["failure_reason"] = "native_anyvisloc_pipeline_missing"
            result["total_time_ms"] = (time.perf_counter() - start_total) * 1000
            return result
        gt = frame.get("gt_pose", {}) or {}
        camera = frame.get("camera", {}) or {}
        metadata = frame.get("metadata", {}) or {}
        cfg = pipeline.get("config", {}) if isinstance(pipeline.get("config", {}), dict) else {}
        native_prior_uses_exact_gt = bool(cfg.get("native_use_exact_gt_prior", False))
        telemetry = dict(gt) if native_prior_uses_exact_gt else get_simulated_telemetry(gt, frame_index, bias_drift)
        telemetry_prior_policy = "exact_gt_pose" if native_prior_uses_exact_gt else "simulated_uav_telemetry"
        self._stats["telemetry_prior_policy"] = telemetry_prior_policy
        self._stats["native_prior_uses_exact_gt"] = bool(
            self._stats.get("native_prior_uses_exact_gt", False) or native_prior_uses_exact_gt
        )
        self._stats["native_yaw_height_status"] = (
            str(self._stats.get("native_yaw_height_status", "not_applicable"))
            if self._stats.get("native_yaw_height_status") not in {"not_applicable", "not_run"}
            else "not_run"
        )

        def _coerce_anchor(value: Any) -> Optional[Dict[str, float]]:
            if not isinstance(value, dict):
                return None
            lat = _safe_float(value.get("lat", value.get("latitude", np.nan)), default=float("nan"))
            lon = _safe_float(value.get("lon", value.get("longitude", np.nan)), default=float("nan"))
            if not np.isfinite(lat) or not np.isfinite(lon):
                return None
            out = {"lat": float(lat), "lon": float(lon)}
            alt = _safe_float(value.get("alt", value.get("height", np.nan)), default=float("nan"))
            if np.isfinite(alt):
                out["alt"] = float(alt)
            return out

        def _native_schema_ok(row: Dict[str, Any]) -> bool:
            selected_anchor = _coerce_anchor(row.get("geodetic_anchor"))
            raw_anchor = _coerce_anchor(row.get("raw_retrieval_anchor"))
            timings = row.get("timings", {})
            has_timings = isinstance(timings, dict) and all(
                key in timings for key in ["retrieval_ms", "matching_ms", "geometry_ms"]
            )
            return bool(
                selected_anchor is not None
                and raw_anchor is not None
                and np.isfinite(row.get("visual_pred_lat", np.nan))
                and np.isfinite(row.get("visual_pred_lon", np.nan))
                and has_timings
            )

        true_pos = {
            "lat": float(telemetry.get("lat", gt.get("lat", 0.0))),
            "lon": float(telemetry.get("lon", gt.get("lon", 0.0))),
            "rel_alt": float(
                telemetry.get("rel_alt", telemetry.get("alt", gt.get("alt", gt.get("rel_alt", 0.0))))
            ),
            "yaw": float(telemetry.get("yaw", gt.get("yaw", 0.0))),
            "pitch": float(telemetry.get("pitch", gt.get("pitch", 0.0))),
            "roll": float(telemetry.get("roll", gt.get("roll", 0.0))),
            "width": int(camera.get("width", 4000)),
            "height": int(camera.get("height", 3000)),
            "cam_size": float(metadata.get("cam_size_mm", metadata.get("cam_size", 7.81))),
            "focal_len": float(metadata.get("focal_len_mm", metadata.get("focal_len", 3.6))),
        }
        # Datasets without sensor mm metadata (VPAIR ships Kalibr pixel
        # intrinsics) must not inherit AVL's 7.81mm/3.6mm defaults: that
        # yields fx≈461 px against the true 750.63 px and rescales every
        # native P3P solution. Derive an equivalent mm pair from the
        # calibrated fx so resolution_size() reproduces the pixel focal.
        if "cam_size_mm" not in metadata and "cam_size" not in metadata:
            fx_px = float(camera.get("fx", 0.0))
            if fx_px > 0:
                w_px = float(true_pos["width"])
                h_px = float(true_pos["height"])
                diag_px = float(np.hypot(w_px, h_px))
                # resolution_size uses: px_size = cam_size/diag; focal_px = focal_len/px_size
                # Choose cam_size = diag_px (px_size=1 unit) and focal_len = fx_px.
                true_pos["cam_size"] = diag_px
                true_pos["focal_len"] = fx_px
            else:
                raise RuntimeError(
                    "native pipeline: no sensor mm metadata and no calibrated fx; "
                    "refusing AVL-default intrinsics for a non-AVL dataset"
                )
        if bool(cfg.get("native_oracle_tile", False)):
            ref_path = frame.get("reference_path")
            true_pos.update(
                {
                    "oracle_tile_enabled": True,
                    "oracle_reference_anchor": {
                        "lat": float(gt.get("lat", true_pos["lat"])),
                        "lon": float(gt.get("lon", true_pos["lon"])),
                        "alt": float(gt.get("alt", gt.get("rel_alt", true_pos["rel_alt"]))),
                    },
                    "oracle_reference_path": str(ref_path) if ref_path is not None else "",
                }
            )
        query_path = str(frame.get("query_path", ""))
        try:
            native_result = native_pipeline.process_frame(query_path, true_pos)
            pred_lat = _safe_float(native_result.get("pred_lat", np.nan), default=float("nan"))
            pred_lon = _safe_float(native_result.get("pred_lon", np.nan), default=float("nan"))
            pred_alt = _safe_float(
                native_result.get("pred_alt", native_result.get("src_alt_m", np.nan)),
                default=float("nan"),
            )
            pred_heading = _safe_float(native_result.get("src_yaw_deg", np.nan), default=float("nan"))
            selected_anchor = _coerce_anchor(
                native_result.get("geodetic_anchor")
                or native_result.get("selected_anchor")
                or native_result.get("selected_geodetic_anchor")
            )
            raw_anchor = _coerce_anchor(
                native_result.get("raw_retrieval_anchor")
                or native_result.get("raw_top1_anchor")
                or native_result.get("raw_geodetic_anchor")
            )
            yaw_height_status = str(native_result.get("native_yaw_height_status", "") or "")
            if not yaw_height_status:
                yaw_height_status = "ok" if np.isfinite(pred_alt) and np.isfinite(pred_heading) else "failed"
            pose_success = bool(
                native_result.get("success", False)
                and np.isfinite(pred_lat)
                and np.isfinite(pred_lon)
            )
            result["post_fgo_error_m"] = (
                _latlon_distance_m_approx(
                    float(gt.get("lat", float("nan"))),
                    float(gt.get("lon", float("nan"))),
                    float(pred_lat),
                    float(pred_lon),
                )
                if pose_success else float("nan")
            )
            result.update(
                {
                    "phase1_source": "native_anyvisloc_camp_topn",
                    "retrieval_verification_status": "native_pnp_rerank",
                    "matcher_backend_effective": "RoMa(native_anyvisloc)",
                    "geometry_backend_effective": "DSM_P3P(native_anyvisloc)",
                    "native_rerank_policy": "max_pnp_inliers_then_retrieval_rank",
                    "telemetry_prior_policy": telemetry_prior_policy,
                    "native_prior_uses_exact_gt": native_prior_uses_exact_gt,
                    "native_yaw_height_status": yaw_height_status,
                    "E_alt_available": bool(np.isfinite(pred_alt)),
                    "pred_lat": pred_lat,
                    "pred_lon": pred_lon,
                    "pred_alt": pred_alt,
                    "pred_heading": pred_heading,
                    "visual_pred_lat": pred_lat,
                    "visual_pred_lon": pred_lon,
                    "visual_pred_alt": pred_alt,
                    "geodetic_anchor": selected_anchor,
                    "raw_retrieval_anchor": raw_anchor,
                    "homography_valid": bool(yaw_height_status == "ok"),
                    "success": pose_success,
                    "pose_success": pose_success,
                    "accepted_for_metrics": pose_success,
                    "num_inliers": _safe_int(
                        native_result.get("n_inliers", native_result.get("n_inliers_yaw", 0)),
                        default=0,
                        min_value=0,
                    ),
                    "num_matches": _safe_int(native_result.get("num_matches", 0), default=0, min_value=0),
                    "failure_reason": "" if pose_success else str(native_result.get("error", "native_anyvisloc_failed")),
                    "bottleneck_label": "healthy" if pose_success else "native_anyvisloc_failed",
                    "retrieval_candidate_index": _safe_int(
                        native_result.get("retrieval_candidate_index", native_result.get("selected_candidate_index", -1)),
                        default=-1,
                    ),
                    "raw_retrieval_candidate_index": _safe_int(
                        native_result.get("raw_retrieval_candidate_index", -1),
                        default=-1,
                    ),
                    "raw_retrieval_topk_indices": native_result.get("raw_retrieval_topk_indices", []),
                    "candidate_rank_selected": _safe_int(native_result.get("selected_candidate_rank", -1), default=-1),
                    "candidate_quality_score": _safe_float(native_result.get("n_inliers", 0), default=0.0),
                    "selected_reference_name": str(native_result.get("selected_reference_name", "")),
                    "raw_top1_reference_name": str(native_result.get("raw_top1_reference_name", "")),
                    "timings": {
                        "retrieval_ms": 1000.0 * _safe_float(native_result.get("retrieval_time_s", 0.0), default=0.0),
                        "matching_ms": 1000.0 * _safe_float(native_result.get("match_time_s", 0.0), default=0.0),
                        "geometry_ms": 1000.0 * _safe_float(native_result.get("pnp_time_s", 0.0), default=0.0),
                        "fgo_ms": 0.0,
                    },
                    "native_result": native_result,
                }
            )
            result["native_anyvisloc_schema_ok"] = _native_schema_ok(result)
            self._stats["native_anyvisloc_schema_ok"] = bool(
                self._stats.get("native_anyvisloc_schema_ok", True) and result["native_anyvisloc_schema_ok"]
            )
            previous_yaw_height_status = str(self._stats.get("native_yaw_height_status", "not_run"))
            if yaw_height_status != "ok":
                self._stats["native_yaw_height_status"] = yaw_height_status
            elif previous_yaw_height_status in {"not_applicable", "not_run", "ok"}:
                self._stats["native_yaw_height_status"] = "ok"
            else:
                self._stats["native_yaw_height_status"] = "mixed"
            logger.debug(
                "Native AnyVisLoc frame=%s prior=%s pose_success=%s schema_ok=%s yaw_height=%s "
                "inliers=%s matches=%s timings_ms=%s selected_ref=%s raw_ref=%s",
                frame.get("image_name"),
                telemetry_prior_policy,
                pose_success,
                result["native_anyvisloc_schema_ok"],
                yaw_height_status,
                result.get("num_inliers"),
                result.get("num_matches"),
                result.get("timings"),
                result.get("selected_reference_name"),
                result.get("raw_top1_reference_name"),
            )
        except Exception as exc:
            result["failure_reason"] = f"native_anyvisloc_exception:{type(exc).__name__}: {exc}"
            result["bottleneck_label"] = "native_anyvisloc_exception"
            result["success"] = False
            result["pose_success"] = False
            result["accepted_for_metrics"] = False
            result["pred_lat"] = float("nan")
            result["pred_lon"] = float("nan")
            result["pred_alt"] = float("nan")
            result["pred_heading"] = float("nan")
            result["visual_pred_lat"] = float("nan")
            result["visual_pred_lon"] = float("nan")
            result["visual_pred_alt"] = float("nan")
            result["telemetry_prior_policy"] = telemetry_prior_policy
            result["native_prior_uses_exact_gt"] = native_prior_uses_exact_gt
            result["native_anyvisloc_schema_ok"] = False
            result["native_yaw_height_status"] = "failed"
            result["E_alt_available"] = False
            self._stats["native_anyvisloc_schema_ok"] = False
            self._stats["native_yaw_height_status"] = "failed"
            self._stats["frame_exceptions"] += 1
            logger.error("Native AnyVisLoc frame failed: %s", exc, exc_info=True)

        self._stats["processed_frames"] = int(self._stats.get("processed_frames", 0)) + 1
        result.setdefault("timings", {})
        result["timings"]["total_ms"] = (time.perf_counter() - start_total) * 1000
        result["total_time_ms"] = result["timings"]["total_ms"]
        return result

    def _process_frame_native_ortholoc(
        self,
        frame: Dict[str, Any],
        pipeline: Dict[str, Any],
        result: Dict[str, Any],
        start_total: float,
        frame_index: int = 0,
        bias_drift: list | None = None,
    ) -> Dict[str, Any]:
        """Process one frame through native OrthoLoC: telemetry crop -> RoMa -> 2D-3D lift -> PnP."""
        ortholoc_assets = pipeline.get("native_ortholoc")
        if ortholoc_assets is None:
            result["failure_reason"] = "native_ortholoc_assets_missing"
            result["total_time_ms"] = (time.perf_counter() - start_total) * 1000
            return result

        gt = frame.get("gt_pose", {}) or {}
        telemetry = get_simulated_telemetry(gt, frame_index, bias_drift)

        dop_image = ortholoc_assets["dop_image"]
        dop_transform = ortholoc_assets["dop_transform"]
        dsm_elev = ortholoc_assets["dsm_elev"]
        dsm_transform = ortholoc_assets["dsm_transform"]
        matcher = ortholoc_assets["matcher"]
        to_latlon = ortholoc_assets["to_latlon"]
        from_latlon = ortholoc_assets["from_latlon"]

        try:
            import cv2 as _cv2
            import rasterio as _rio
            from scipy.ndimage import map_coordinates as _map_coords

            tel_lat = float(telemetry.get("lat", gt.get("lat", 0.0)))
            tel_lon = float(telemetry.get("lon", gt.get("lon", 0.0)))

            map_x, map_y = from_latlon.transform(tel_lon, tel_lat)
            inv_dop = ~dop_transform
            px_col, px_row = inv_dop * (map_x, map_y)
            px_row, px_col = int(round(px_row)), int(round(px_col))
            # Post-submission probe (23/08): center the DOP crop on a retrieved
            # tile recorded by another run (e.g. the CAMP top-1 of the cascade)
            # instead of the telemetry prior, so the OrthoLoC back end is
            # scored behind a real retrieval, as the VPAIR probe row is.
            center_csv = pipeline.get("config", {}).get("ortholoc_center_tile_csv")
            center_source = "telemetry"
            if center_csv:
                if not hasattr(self, "_ortholoc_center_tiles"):
                    import csv as _csv
                    with open(str(center_csv), newline="", encoding="utf-8") as _fh:
                        self._ortholoc_center_tiles = {
                            r["image_name"]: r["reference_name"] for r in _csv.DictReader(_fh)
                        }
                tile_name = self._ortholoc_center_tiles.get(str(frame.get("image_name", "")), "")
                _m = re.search(r"_r(\d+)_c(\d+)", tile_name)
                if not _m:
                    result["failure_reason"] = "ortholoc_center_tile_missing"
                    result["total_time_ms"] = (time.perf_counter() - start_total) * 1000
                    return result
                tile_px = _safe_int(
                    pipeline.get("config", {}).get("ortholoc_center_tile_px", 2048),
                    default=2048, min_value=64,
                )
                px_row = int(_m.group(1)) + tile_px // 2
                px_col = int(_m.group(2)) + tile_px // 2
                _cx, _cy = _rio.transform.xy(dop_transform, px_row, px_col)
                tel_lon, tel_lat = to_latlon.transform(_cx, _cy)
                center_source = "retrieved_tile"

            crop_size = _safe_int(
                pipeline.get("config", {}).get("ortholoc_crop_size_px", 1024),
                default=1024, min_value=256,
            )
            half = crop_size // 2
            dop_h, dop_w = dop_image.shape[:2]
            r0 = max(0, px_row - half)
            r1 = min(dop_h, px_row + half)
            c0 = max(0, px_col - half)
            c1 = min(dop_w, px_col + half)

            if r1 - r0 < 128 or c1 - c0 < 128:
                result["failure_reason"] = (
                    f"ortholoc_crop_too_small_{r1-r0}x{c1-c0}_at_r{px_row}_c{px_col}"
                )
                result["total_time_ms"] = (time.perf_counter() - start_total) * 1000
                return result

            dop_crop = dop_image[r0:r1, c0:c1].copy()

            crop_h, crop_w = r1 - r0, c1 - c0
            crop_rows, crop_cols = np.meshgrid(
                np.arange(r0, r1), np.arange(c0, c1), indexing="ij",
            )
            xs_flat, ys_flat = _rio.transform.xy(
                dop_transform, crop_rows.ravel(), crop_cols.ravel(),
            )
            x_map = np.asarray(xs_flat, dtype=np.float64).reshape(crop_h, crop_w)
            y_map = np.asarray(ys_flat, dtype=np.float64).reshape(crop_h, crop_w)

            inv_dsm = ~dsm_transform
            dsm_c, dsm_r = inv_dsm * (x_map.ravel(), y_map.ravel())
            dsm_r = np.asarray(dsm_r, dtype=np.float64).reshape(crop_h, crop_w)
            dsm_c = np.asarray(dsm_c, dtype=np.float64).reshape(crop_h, crop_w)
            z_map = _map_coords(dsm_elev, [dsm_r, dsm_c], order=1, mode="nearest")

            dsm_crop = np.stack((x_map, y_map, z_map), axis=-1)

            result["selected_reference_name"] = f"ortholoc_dop_r{r0}_c{c0}_{r1-r0}x{c1-c0}"
            result["raw_top1_reference_name"] = result["selected_reference_name"]
            result["geodetic_anchor"] = {"lat": tel_lat, "lon": tel_lon}
            result["raw_retrieval_anchor"] = {"lat": tel_lat, "lon": tel_lon}
            result["phase1_source"] = (
                "telemetry_crop_ortholoc" if center_source == "telemetry"
                else "retrieved_tile_crop_ortholoc"
            )
            result["retrieval_verification_status"] = "ortholoc_dop_crop"

            query_path = str(frame.get("query_path", ""))
            query_bgr = _cv2.imread(query_path)
            if query_bgr is None:
                query_image = frame.get("query_image")
                if query_image is None:
                    result["failure_reason"] = "query_image_load_failed"
                    result["total_time_ms"] = (time.perf_counter() - start_total) * 1000
                    return result
                query_rgb = (
                    _cv2.cvtColor(query_image, _cv2.COLOR_BGR2RGB)
                    if query_image.ndim == 3 and query_image.shape[2] == 3
                    else query_image
                )
            else:
                query_rgb = _cv2.cvtColor(query_bgr, _cv2.COLOR_BGR2RGB)

            t_match = time.perf_counter()
            correspondences_2d2d = matcher(query_rgb, dop_crop, silent=True)
            match_time_s = time.perf_counter() - t_match

            n_raw = len(correspondences_2d2d)
            result["num_matches"] = n_raw
            result.setdefault("timings", {})
            result["timings"]["matching_ms"] = match_time_s * 1000

            min_corr = 10
            if n_raw < min_corr:
                result["failure_reason"] = f"ortholoc_too_few_correspondences_{n_raw}"
                result["total_time_ms"] = (time.perf_counter() - start_total) * 1000
                return result

            filtered = correspondences_2d2d.take_min_conf(0.5)
            if len(filtered) < min_corr:
                result["failure_reason"] = f"ortholoc_too_few_after_conf_filter_{len(filtered)}"
                result["total_time_ms"] = (time.perf_counter() - start_total) * 1000
                return result

            correspondences_2d3d_raw = filtered.to_2d3d(grid3d_1=dsm_crop)
            n_2d3d = len(correspondences_2d3d_raw)
            if n_2d3d < 6:
                result["failure_reason"] = f"ortholoc_too_few_2d3d_{n_2d3d}"
                result["total_time_ms"] = (time.perf_counter() - start_total) * 1000
                return result

            from ortholoc.correspondences.Correspondences2D3D import Correspondences2D3D
            pts3d_raw = correspondences_2d3d_raw.pts1.copy()
            finite_mask = np.isfinite(pts3d_raw).all(axis=1)
            if finite_mask.sum() < 6:
                result["failure_reason"] = f"ortholoc_too_few_finite_3d_{int(finite_mask.sum())}"
                result["total_time_ms"] = (time.perf_counter() - start_total) * 1000
                return result
            centroid_3d = pts3d_raw[finite_mask].mean(axis=0)
            pts3d_centered = pts3d_raw - centroid_3d
            correspondences_2d3d = Correspondences2D3D(
                pts0=correspondences_2d3d_raw.pts0.copy(),
                pts1=pts3d_centered,
                confidences=correspondences_2d3d_raw.confidences.copy()
                    if correspondences_2d3d_raw.confidences is not None else None,
                is_normalized=correspondences_2d3d_raw.is_normalized,
            )

            t_pnp = time.perf_counter()
            query_h, query_w = query_rgb.shape[:2]
            num_points = _safe_int(
                pipeline.get("config", {}).get("ortholoc_pnp_num_points", 5000),
                default=5000, min_value=100,
            )
            camera = frame.get("camera", {}) or {}
            metadata = frame.get("metadata", {}) or {}
            cam_size_mm = _safe_float(
                metadata.get("cam_size_mm", metadata.get("cam_size", 0)), default=0, min_value=0
            )
            focal_len_mm = _safe_float(
                metadata.get("focal_len_mm", metadata.get("focal_len", 0)), default=0, min_value=0
            )
            intrinsics_matrix = None
            if cam_size_mm > 0 and focal_len_mm > 0:
                px_size = cam_size_mm / np.sqrt(query_w**2 + query_h**2)
                fx = focal_len_mm / px_size
                intrinsics_matrix = np.array([
                    [fx, 0, query_w / 2.0],
                    [0, fx, query_h / 2.0],
                    [0, 0, 1],
                ], dtype=np.float64)
            success_pnp, pose_c2w, intrinsics_out, inlier_mask, reproj_errors = \
                correspondences_2d3d.calibrate(
                    num_points=num_points,
                    width=query_w,
                    height=query_h,
                    intrinsics_matrix=intrinsics_matrix,
                    pnp_mode="poselib",
                )
            pnp_time_s = time.perf_counter() - t_pnp
            result["timings"]["geometry_ms"] = pnp_time_s * 1000
            result["timings"]["retrieval_ms"] = 0.0
            result["timings"]["fgo_ms"] = 0.0

            n_inliers = int(np.sum(inlier_mask)) if inlier_mask is not None else 0
            result["num_inliers"] = n_inliers

            if not success_pnp or pose_c2w is None:
                result["failure_reason"] = "ortholoc_pnp_failed"
                result["total_time_ms"] = (time.perf_counter() - start_total) * 1000
                return result

            translation = pose_c2w[:3, 3].astype(np.float64) + centroid_3d
            pred_x, pred_y, pred_z = float(translation[0]), float(translation[1]), float(translation[2])
            pred_lon_out, pred_lat_out = to_latlon.transform(pred_x, pred_y)

            # PnP already yields the full camera-to-world rotation, so heading
            # is available at no extra cost. It was previously discarded, which
            # left the yaw column blank and read as "cannot estimate heading".
            pred_heading_deg = float("nan")
            try:
                R_c2w = np.asarray(pose_c2w[:3, :3], dtype=np.float64)
                # Heading comes from the image "up" axis (-Y in the OpenCV
                # camera frame), not the optical axis: for a nadir camera the
                # optical axis is vertical and its map-plane projection
                # vanishes, which would leave the 162 nadir AVL frames
                # undefined. The world frame here is the projected map CRS,
                # so component 0 is easting and component 1 is northing.
                up_world = R_c2w @ np.array([0.0, -1.0, 0.0])
                if float(np.hypot(up_world[0], up_world[1])) > 1e-6:
                    pred_heading_deg = float(
                        (np.degrees(np.arctan2(up_world[0], up_world[1])) + 180.0) % 360.0 - 180.0)
            except Exception:
                pred_heading_deg = float("nan")

            # The shared FORGE labeller thresholds post_fgo_error_m. This path has
            # no FGO stage, so without this the column stays NaN and every native
            # row silently reads bad_pose_label=False / unsafe_accept=False.
            result["post_fgo_error_m"] = _latlon_distance_m_approx(
                float(gt.get("lat", float("nan"))),
                float(gt.get("lon", float("nan"))),
                float(pred_lat_out),
                float(pred_lon_out),
            )
            result.update({
                "pred_lat": float(pred_lat_out),
                "pred_lon": float(pred_lon_out),
                "pred_alt": float(pred_z),
                "pred_heading": pred_heading_deg,
                "native_yaw_height_status": "ok" if np.isfinite(pred_heading_deg) else "failed",
                "visual_pred_lat": float(pred_lat_out),
                "visual_pred_lon": float(pred_lon_out),
                "visual_pred_alt": float(pred_z),
                "success": True,
                "pose_success": True,
                "accepted_for_metrics": True,
                "homography_valid": True,
                "failure_reason": "",
                "bottleneck_label": "healthy",
                "matcher_backend_effective": "RoMa(native_ortholoc)",
                "geometry_backend_effective": "OrthoLoC_PnP(poselib)",
                "telemetry_prior_policy": "simulated_uav_telemetry",
            })

            logger.debug(
                "OrthoLoC frame=%s  pred=(%.6f,%.6f)  inliers=%d  corr=%d  2d3d=%d  crop=%dx%d",
                frame.get("image_name"), pred_lat_out, pred_lon_out,
                n_inliers, n_raw, n_2d3d, r1 - r0, c1 - c0,
            )

        except Exception as exc:
            result["failure_reason"] = f"native_ortholoc_exception:{type(exc).__name__}: {exc}"
            result["bottleneck_label"] = "native_ortholoc_exception"
            result["success"] = False
            result["pose_success"] = False
            result["accepted_for_metrics"] = False
            self._stats["frame_exceptions"] = int(self._stats.get("frame_exceptions", 0)) + 1
            logger.warning("OrthoLoC exception for %s: %s", frame.get("image_name"), exc, exc_info=True)

        self._stats["processed_frames"] = int(self._stats.get("processed_frames", 0)) + 1
        result.setdefault("timings", {})
        result["timings"]["total_ms"] = (time.perf_counter() - start_total) * 1000
        result["total_time_ms"] = result["timings"]["total_ms"]
        return result

    def _process_frame(self,
                        frame: Dict[str, Any],
                        pipeline: Dict,
                        frame_index: int = 0,
                        bias_drift: list | None = None) -> Dict[str, Any]:
        """
        Process a single frame through the pipeline.
        
        Returns dict with predictions and timing info.
        """
        result = {
            'image_name': frame['image_name'],
            'success': False,
            'pose_success': False,
            'accepted_for_metrics': False,
            'pred_lat': 0.0, 'pred_lon': 0.0,
            'pred_alt': 0.0, 'pred_heading': 0.0,
            'num_inliers': 0,
            'num_matches': 0,
            'match_confidence_mean': 0.0,
            'shape_confidence': np.nan,
            'homography_valid': False,
            'timings': {},
            'phase1_source': 'unknown',
            'retrieval_verification_status': 'not_run',
            'verification_exception': '',
            'matcher_backend_effective': pipeline.get('effective_matcher_backend', 'unknown'),
            'candidate_rank_selected': -1,
            'candidate_quality_score': float("nan"),
            'reproj_residual_px': float("nan"),
            'gate_decision': 'none',
            'gate_reason': '',
            'visual_pred_lat': np.nan,
            'visual_pred_lon': np.nan,
            'visual_pred_alt': np.nan,
            'raw_retrieval_anchor': None,
            'raw_retrieval_similarity_score': np.nan,
            'raw_retrieval_candidate_index': -1,
            'raw_retrieval_topk_indices': [],
            'raw_retrieval_topk_hits_window1': 0,
            'raw_retrieval_topk_hits_window5': 0,
            'raw_retrieval_topk_hits_window10': 0,
            'selected_reference_name': '',
            'raw_top1_reference_name': '',
            'bottleneck_label': 'unknown',
            'pre_fgo_error_m': float("nan"),
            'post_fgo_error_m': float("nan"),
            'final_pose_gate_reject': False,
            'final_pose_gate_reason': '',
        }
        _apply_reliability_schema_defaults(result, self._adversarial_config)

        def finalize_native(res):
            # Native backends return early, so they bypassed the labeller call
            # at the end of this method and shipped unlabelled FORGE rows.
            meta = frame.get("metadata", {}) if isinstance(frame.get("metadata"), dict) else {}
            _finalize_forge_labels(res, str(meta.get("dataset", "")), self._config)
            # metrics_summary.json reads these counters; without them a native run
            # reports num_unsafe_accept=0 while its own rows say otherwise.
            if bool(res.get("bad_pose_label", False)):
                self._stats["num_bad_pose"] = int(self._stats.get("num_bad_pose", 0)) + 1
            if bool(res.get("unsafe_accept", False)):
                self._stats["num_unsafe_accept"] = int(self._stats.get("num_unsafe_accept", 0)) + 1
            return res

        start_total = time.perf_counter()
        if pipeline.get("native_backend") == "anyvisloc":
            return finalize_native(self._process_frame_native_anyvisloc(
                frame,
                pipeline,
                result,
                start_total,
                frame_index=frame_index,
                bias_drift=bias_drift,
            ))
        if pipeline.get("native_backend") == "ortholoc_2d3d":
            return finalize_native(self._process_frame_native_ortholoc(
                frame,
                pipeline,
                result,
                start_total,
                frame_index=frame_index,
                bias_drift=bias_drift,
            ))

        try:
            query_image = frame.get('query_image')
            raw_query_image = query_image
            ref_image = frame.get('reference_image')
            metadata_gt = frame.get('gt_pose', {})  # GT — used ONLY for evaluation metrics
            # Simulated UAV telemetry with realistic sensor noise (no GT leakage)
            telemetry = get_simulated_telemetry(metadata_gt, frame_index, bias_drift)
            camera = frame.get('camera', {})
            
            if query_image is None:
                result['failure_reason'] = 'no_query_image'
                return result
            
            # === Step 1: IPM (if enabled) ===
            t0 = time.perf_counter()
            ipm_yaw_applied_rad = 0.0
            if pipeline.get('ipm') is not None:
                ipm = pipeline['ipm']
                # Set camera matrix
                K = np.array([
                    [camera.get('fx', 1000), 0, camera.get('cx', 640)],
                    [0, camera.get('fy', 1000), camera.get('cy', 360)],
                    [0, 0, 1]
                ])
                ipm.set_camera(K)
                
                # Apply IPM using simulated IMU attitude (realistic sensor noise)
                query_image, H_ipm = ipm.apply_ipm_from_metadata(
                    query_image, telemetry
                )
                # The warp rotates the query into a North-up frame, so the
                # homography measured downstream carries only the residual
                # rotation. Carry the removed yaw forward so the pose solver can
                # add it back; otherwise the reported heading collapses to ~0
                # and the yaw error equals the true heading.
                ipm_yaw_applied_rad = float(getattr(ipm, "last_yaw_applied_rad", 0.0))
            match_used_raw_query = False  # set True if RoMa (raw-query) match is accepted
            result['timings']['ipm_ms'] = (time.perf_counter() - t0) * 1000
            
            # === Step 2: Retrieval (UltraVPR candidate index + anchor) ===
            t0 = time.perf_counter()
            # IMPORTANT:
            # Retrieval must run on the original perspective query. Running retrieval
            # on IPM-warped imagery hurts descriptor invariance and can collapse recall.
            retrieval_query = raw_query_image if raw_query_image is not None else query_image
            phase1 = self._run_phase1_retrieval(frame, pipeline, query_image=retrieval_query, telemetry=telemetry)
            if phase1 is None:
                result['failure_reason'] = 'no_reference_image'
                result['bottleneck_label'] = 'retrieval_fail'
            else:
                result['phase1_source'] = (
                    'oracle_eval_mode' if self._eval_mode == 'oracle_reference'
                    else (
                        'retrieval_topk_verified'
                        if (pipeline.get('retrieval') is not None and bool(pipeline.get('config', {}).get('retrieval_verify_top_k', True)))
                        else ('retrieval_top1_raw' if pipeline.get('retrieval') is not None else 'retrieval_unavailable_fallback')
                    )
                )
                result['retrieval_verification_status'] = 'ok'
                ref_image = phase1.reference_image
                result['retrieval_candidate_index'] = phase1.candidate_index
                result['geodetic_anchor'] = phase1.geodetic_anchor
                result['retrieval_similarity_score'] = phase1.similarity_score
                anchor_meta = phase1.geodetic_anchor if isinstance(phase1.geodetic_anchor, dict) else {}
                result['candidate_quality_score'] = _safe_float(anchor_meta.get("_candidate_quality_score", np.nan), default=float("nan"))
                result['candidate_rank_selected'] = _safe_int(anchor_meta.get("_candidate_rank_selected", -1), default=-1)
                result['selected_reference_name'] = str(anchor_meta.get("_selected_reference_name", ""))
                raw_lat = _safe_float(anchor_meta.get("_raw_top1_lat", np.nan), default=float("nan"))
                raw_lon = _safe_float(anchor_meta.get("_raw_top1_lon", np.nan), default=float("nan"))
                if np.isfinite(raw_lat) and np.isfinite(raw_lon):
                    result['raw_retrieval_anchor'] = {"lat": raw_lat, "lon": raw_lon}
                result['raw_retrieval_similarity_score'] = _safe_float(anchor_meta.get("_raw_top1_similarity", np.nan), default=float("nan"))
                result['raw_retrieval_candidate_index'] = _safe_int(anchor_meta.get("_raw_top1_candidate_index", -1), default=-1)
                result['raw_top1_reference_name'] = str(anchor_meta.get("_raw_top1_reference_name", ""))
                raw_topk = anchor_meta.get("_raw_topk_candidate_indices", [])
                if isinstance(raw_topk, list):
                    result['raw_retrieval_topk_indices'] = [
                        _safe_int(v, default=-1) for v in raw_topk if _safe_int(v, default=-1) >= 0
                    ]
                result['retrieval_score_top1'] = _safe_float(anchor_meta.get("_raw_top1_similarity", np.nan), default=float("nan"))
                result['retrieval_score_top2'] = _safe_float(anchor_meta.get("_raw_top2_similarity", np.nan), default=float("nan"))
                if np.isfinite(result['retrieval_score_top1']) and np.isfinite(result['retrieval_score_top2']):
                    result['retrieval_score_margin'] = float(result['retrieval_score_top1'] - result['retrieval_score_top2'])
                result.update({
                    "adversarial_mode": str(anchor_meta.get("_adversarial_mode", result.get("adversarial_mode", "none"))),
                    "adversarial_condition": str(anchor_meta.get("_adversarial_condition", result.get("adversarial_condition", "clean"))),
                    "adversarial_offset_m": _safe_float(anchor_meta.get("_adversarial_offset_m", result.get("adversarial_offset_m", np.nan)), default=float("nan")),
                    "adversarial_candidate_policy": str(anchor_meta.get("_adversarial_candidate_policy", result.get("adversarial_candidate_policy", ""))),
                    "adversarial_candidate_rank": anchor_meta.get("_adversarial_candidate_rank", result.get("adversarial_candidate_rank", "")),
                    "adversarial_candidate_pool_size": _safe_int(anchor_meta.get("_adversarial_candidate_pool_size", result.get("adversarial_candidate_pool_size", 0)), default=0),
                    "adversarial_eligible_wrong_count": _safe_int(anchor_meta.get("_adversarial_eligible_wrong_count", result.get("adversarial_eligible_wrong_count", 0)), default=0),
                    "adversarial_selection_reason": str(anchor_meta.get("_adversarial_selection_reason", result.get("adversarial_selection_reason", ""))),
                    "adversarial_failure_reason": str(anchor_meta.get("_adversarial_failure_reason", result.get("adversarial_failure_reason", ""))),
                    "adversarial_forced_reference_name": str(anchor_meta.get("_adversarial_forced_reference_name", result.get("adversarial_forced_reference_name", ""))),
                    "adversarial_original_reference_name": str(anchor_meta.get("_adversarial_original_reference_name", result.get("adversarial_original_reference_name", ""))),
                    "adversarial_forced_distance_to_gt_m": _safe_float(anchor_meta.get("_adversarial_forced_distance_to_gt_m", result.get("adversarial_forced_distance_to_gt_m", np.nan)), default=float("nan")),
                    "is_adversarial": bool(anchor_meta.get("_is_adversarial", result.get("is_adversarial", False))),
                    "adversarial_assigned_attack": bool(anchor_meta.get("_adversarial_assigned_attack", result.get("adversarial_assigned_attack", False))),
                    "adversarial_attacked": bool(anchor_meta.get("_adversarial_attacked", result.get("adversarial_attacked", False))),
                    "is_wrong_reference": bool(anchor_meta.get("_is_wrong_reference", result.get("is_wrong_reference", False))),
                })
                if bool(anchor_meta.get("_adversarial_fail_closed", False)):
                    result['failure_reason'] = str(anchor_meta.get("_adversarial_failure_reason", "adversarial_fail_closed"))
                    result['bottleneck_label'] = 'retrieval_fail'
                    result['retrieval_verification_status'] = 'adversarial_fail_closed'
                    result['pose_success'] = False
                    result['success'] = False
                    result['accepted_for_metrics'] = False
                result['retrieval_protocol_violation'] = bool(
                    anchor_meta.get("_retrieval_protocol_violation", False)
                )
                result['retrieval_protocol_filtered_candidates_count'] = _safe_int(
                    anchor_meta.get("_retrieval_protocol_filtered_candidates_count", 0),
                    default=0,
                )
                if pipeline.get('local_origin') is None:
                    pipeline['local_origin'] = {
                        'lat': float(phase1.geodetic_anchor.get('lat', 0.0)) if phase1 else 0.0,
                        'lon': float(phase1.geodetic_anchor.get('lon', 0.0)) if phase1 else 0.0,
                    }
            result['timings']['retrieval_ms'] = (time.perf_counter() - t0) * 1000
            
            # === Step 3: Feature Matching & LGI Filtering ===
            t0 = time.perf_counter()
            match_result = None
            num_matches = 0
            num_inliers = 0
            
            if ref_image is not None and query_image is not None and pipeline.get('matcher'):
                phase2_in = Phase2Input(
                    pseudo_nadir_query=query_image,
                    reference_tile=ref_image,
                )
                match_result = pipeline['matcher'].match(
                    phase2_in.pseudo_nadir_query,
                    phase2_in.reference_tile,
                )
                # Apply LGI Filter immediately to primary matcher
                if pipeline.get('lgi_filter') is not None and match_result.num_inliers >= 4:
                    match_result = pipeline['lgi_filter'].filter_match_result(match_result)

                num_matches = len(match_result.query_keypoints)
                num_inliers = match_result.num_inliers if match_result.inlier_mask is not None else num_matches

                if len(match_result.confidence) > 0:
                    result['match_confidence_mean'] = float(np.mean(match_result.confidence))
                sc0 = self._resolve_shape_confidence(
                    match_result=match_result,
                    query_image=phase2_in.pseudo_nadir_query,
                    ref_image=phase2_in.reference_tile,
                )
                result['shape_confidence'] = float(sc0)
                if isinstance(match_result.metadata, dict):
                    match_result.metadata['shape_confidence'] = float(sc0)
                result['matching_attempted'] = True

                # Optional AdHoP-style selected-reference preconditioning.
                # Default-off: used only for explicit diagnostic/follow-up runs.
                cfg_pre = pipeline.get("config", {})
                if bool(cfg_pre.get("use_reference_preconditioning", cfg_pre.get("enable_adhop_preconditioning", False))):
                    try:
                        from modules.vision.matching import rematch_with_reference_preconditioning

                        pre_result, pre_diag = rematch_with_reference_preconditioning(
                            pipeline['matcher'],
                            phase2_in.pseudo_nadir_query,
                            phase2_in.reference_tile,
                            match_result,
                            min_inliers=_safe_int(cfg_pre.get("reference_preconditioning_min_inliers", 8), default=8, min_value=4),
                            min_inlier_gain=_safe_int(cfg_pre.get("reference_preconditioning_min_inlier_gain", 2), default=2, min_value=0),
                            min_score_gain=_safe_float(cfg_pre.get("reference_preconditioning_min_score_gain", 1.0), default=1.0),
                            max_reproj_px=_safe_float(cfg_pre.get("reference_preconditioning_max_reproj_px", cfg_pre.get("max_reproj_residual_px", 10.0)), default=10.0, min_value=0.1),
                            shape_tolerance=_safe_float(cfg_pre.get("reference_preconditioning_shape_tolerance", 0.03), default=0.03, min_value=0.0),
                        )
                        self._stats["reference_preconditioning_attempted"] = int(self._stats.get("reference_preconditioning_attempted", 0)) + 1
                        if bool(pre_diag.get("accepted", False)):
                            self._stats["reference_preconditioning_accepted"] = int(self._stats.get("reference_preconditioning_accepted", 0)) + 1
                        else:
                            self._stats["reference_preconditioning_rejected"] = int(self._stats.get("reference_preconditioning_rejected", 0)) + 1
                        result.update({
                            "reference_preconditioning_attempted": True,
                            "reference_preconditioning_accepted": bool(pre_diag.get("accepted", False)),
                            "reference_preconditioning_reason": str(pre_diag.get("reason", "")),
                            "reference_preconditioning_warp_valid": bool(pre_diag.get("warp_valid", False)),
                            "reference_preconditioning_inliers_before": _safe_int(pre_diag.get("inliers_before", 0), default=0),
                            "reference_preconditioning_inliers_after": _safe_int(pre_diag.get("inliers_after", 0), default=0),
                            "reference_preconditioning_reproj_before_px": _safe_float(pre_diag.get("reproj_before_px", np.nan), default=float("nan")),
                            "reference_preconditioning_reproj_after_px": _safe_float(pre_diag.get("reproj_after_px", np.nan), default=float("nan")),
                            "reference_preconditioning_shape_before": _safe_float(pre_diag.get("shape_before", np.nan), default=float("nan")),
                            "reference_preconditioning_shape_after": _safe_float(pre_diag.get("shape_after", np.nan), default=float("nan")),
                            "reference_preconditioning_score_before": _safe_float(pre_diag.get("score_before", np.nan), default=float("nan")),
                            "reference_preconditioning_score_after": _safe_float(pre_diag.get("score_after", np.nan), default=float("nan")),
                        })
                        if pre_result is not match_result:
                            match_result = pre_result
                            if pipeline.get('lgi_filter') is not None and match_result.num_inliers >= 4:
                                match_result = pipeline['lgi_filter'].filter_match_result(match_result)
                            num_matches = len(match_result.query_keypoints)
                            num_inliers = match_result.num_inliers if match_result.inlier_mask is not None else num_matches
                            if len(match_result.confidence) > 0:
                                result['match_confidence_mean'] = float(np.mean(match_result.confidence))
                            sc0 = self._resolve_shape_confidence(
                                match_result=match_result,
                                query_image=phase2_in.pseudo_nadir_query,
                                ref_image=phase2_in.reference_tile,
                            )
                            result['shape_confidence'] = float(sc0)
                            result['reproj_residual_px'] = self._compute_reprojection_residual_px(match_result)
                            result['matcher_backend'] = 'reference_preconditioned'
                    except Exception as exc:
                        result['reference_preconditioning_attempted'] = True
                        result['reference_preconditioning_accepted'] = False
                        result['reference_preconditioning_reason'] = f"exception:{type(exc).__name__}"
                        self._stats["reference_preconditioning_attempted"] = int(self._stats.get("reference_preconditioning_attempted", 0)) + 1
                        self._stats["reference_preconditioning_rejected"] = int(self._stats.get("reference_preconditioning_rejected", 0)) + 1
                        if self._verbose:
                            logger.debug("Reference preconditioning exception: %s", exc, exc_info=True)

                # Selective dense fallback: invoke RoMa only for hard cases.
                if pipeline.get('roma_fallback_matcher') is not None:
                    mean_conf = float(np.mean(match_result.confidence)) if len(match_result.confidence) > 0 else 0.0
                    shape_conf = float(result.get('shape_confidence', np.nan))
                    reproj_residual = self._compute_reprojection_residual_px(match_result)
                    result['reproj_residual_px'] = reproj_residual
                    escalate, reason = self._should_escalate_to_roma(
                        pipeline=pipeline,
                        num_inliers=int(num_inliers),
                        shape_confidence=shape_conf,
                        match_confidence=mean_conf,
                        reproj_residual_px=reproj_residual,
                        pitch_deg=float(telemetry.get("pitch", 0.0)),
                    )
                    result['gate_reason'] = reason
                    if escalate:
                        result['gate_decision'] = 'roma_escalate'
                        # Note: we MUST pass the raw_query_image to RoMa since
                        # it struggles with the large blank areas caused by IPM horizon crop
                        try:
                            roma_result = pipeline['roma_fallback_matcher'].match(
                                raw_query_image if raw_query_image is not None else phase2_in.pseudo_nadir_query,
                                phase2_in.reference_tile,
                            )
                            # Apply LGI Filter immediately to RoMa matcher
                            if pipeline.get('lgi_filter') is not None and roma_result.num_inliers >= 4:
                                roma_result = pipeline['lgi_filter'].filter_match_result(roma_result)

                            # Accept RoMa if it improves on LightGlue.
                            # When LightGlue gave 0 inliers, only require the geometry minimum (4)
                            # rather than 8, to avoid silently discarding valid-but-low-inlier RoMa matches.
                            roma_min_inliers = 4 if num_inliers == 0 else max(num_inliers, 8)
                            if roma_result.num_inliers >= roma_min_inliers:
                                match_result = roma_result
                                # RoMa matched the RAW query (see note above), so
                                # its homography already carries the full heading;
                                # the IPM yaw must NOT be re-added for this frame.
                                match_used_raw_query = True
                                num_matches = len(match_result.query_keypoints)
                                num_inliers = match_result.num_inliers if match_result.inlier_mask is not None else num_matches
                                result['matcher_backend'] = 'roma_fallback'
                                self._stats["roma_fallback_frames"] += 1
                                if len(match_result.confidence) > 0:
                                    result['match_confidence_mean'] = float(np.mean(match_result.confidence))
                                result['shape_confidence'] = self._resolve_shape_confidence(
                                    match_result=match_result,
                                    query_image=phase2_in.pseudo_nadir_query,
                                    ref_image=phase2_in.reference_tile,
                                )
                                # Update reproj_residual_px from RoMa so the geometry gate
                                # in Step 5 uses the accepted matcher's actual quality.
                                result['reproj_residual_px'] = self._compute_reprojection_residual_px(match_result)
                            else:
                                result['matcher_backend'] = 'lightglue'
                        except Exception as exc:
                            # Fail-safe: if RoMa fallback crashes, keep primary matcher result.
                            result['matcher_backend'] = 'lightglue'
                            result['gate_decision'] = 'roma_escalate_exception'
                            result['gate_reason'] = f"roma_exception:{type(exc).__name__}"
                            if self._verbose:
                                logger.debug("RoMa fallback exception: %s", exc, exc_info=True)
                    else:
                        result['gate_decision'] = 'fast_path'
            
            result['timings']['matching_ms'] = (time.perf_counter() - t0) * 1000
            
            # === Step 5: Geometry Solving ===
            t0 = time.perf_counter()
            vision_pose = None
            if num_inliers >= 4 and pipeline.get('geometry') and match_result:
                cfg = pipeline.get("config", {})
                max_reproj = _safe_float(pipeline.get("config", {}).get("max_reproj_residual_px", 6.0), default=6.0, min_value=0.1)
                if np.isfinite(result.get("reproj_residual_px", np.nan)) and float(result["reproj_residual_px"]) > (1.5 * max_reproj):
                    result["failure_reason"] = "geometry_reproj_gate_reject"
                    result["geometry_gate_reason"] = "high_reprojection_residual"
                    result["bottleneck_label"] = "geometry_fail_after_good_retrieval"
                    num_inliers = 0
                    result['timings']['geometry_ms'] = (time.perf_counter() - t0) * 1000
                    vision_pose = None
                else:
                    min_geom_inliers = _safe_int(cfg.get("geometry_min_inliers_after_lgi", 6), default=6, min_value=4)
                    if int(num_inliers) < min_geom_inliers:
                        result["failure_reason"] = "geometry_inlier_gate_reject"
                        result["geometry_gate_reason"] = "low_inliers_after_lgi"
                        result["bottleneck_label"] = "geometry_fail_after_good_retrieval"
                        vision_pose = None
                    else:
                        result['homography_valid'] = True
                        from core.types import Pose, CameraIntrinsics
                        anchor = result.get('geodetic_anchor') or {}
                        ref_pose = Pose(
                            timestamp=0,
                            position=np.array([
                                float(anchor.get('lat', telemetry.get('lat', 0))),
                                float(anchor.get('lon', telemetry.get('lon', 0))),
                                float(anchor.get('alt', telemetry.get('alt', 0))),
                            ]),
                            orientation=np.array([1, 0, 0, 0])
                        )
                        cam_intrinsics = CameraIntrinsics(
                            fx=camera.get('fx', 1000), fy=camera.get('fy', 1000), 
                            cx=camera.get('cx', 640), cy=camera.get('cy', 360),
                            width=camera.get('width', 1280), height=camera.get('height', 720)
                        )
                        
                        self._dump_match_evidence(frame, match_result)
                        vision_pose = pipeline['geometry'].solve(
                            match_result=match_result,
                            reference_pose=ref_pose,
                            camera=cam_intrinsics,
                            altitude_hint=telemetry.get('alt', 100.0),
                            query_size=(query_image.shape[1], query_image.shape[0]),
                            gsd=frame.get('gsd'),
                            # Square-degree rasters: east metres per pixel is
                            # gsd * cos(lat) at the REFERENCE tile's latitude,
                            # which is the same latitude the solver divides by
                            # when it converts the offset back to degrees, so
                            # the cosine cancels exactly. This holds under every
                            # retrieval protocol, including cross-flight, where
                            # the reference comes from another location. AVL and
                            # VPAIR frames carry no 'gsd_axis', so they pass
                            # None and keep the isotropic expression.
                            gsd_e=(
                                float(frame['gsd']) * float(np.cos(np.radians(ref_pose.position[0])))
                                if frame.get('gsd_axis') == 'square_degree' else None
                            ),
                            ref_size=(ref_image.shape[1], ref_image.shape[0]),
                            tile_meta=anchor,
                            rotation_correction_rad=(
                                np.radians(pipeline.get('dataset_rotation_correction_deg', 0.0))
                                + (0.0 if match_used_raw_query else ipm_yaw_applied_rad)
                            )
                        )
                        geom_diag = getattr(pipeline.get('geometry'), "last_diagnostics", None)
                        if isinstance(geom_diag, dict):
                            result["dsm_available"] = bool(geom_diag.get("dsm_available", self._stats.get("dsm_available", False)))
                            result["dsm_status"] = str(geom_diag.get("status", self._stats.get("dsm_status", "")))
                            result["pnp_inliers"] = _safe_int(geom_diag.get("pnp_inliers", 0), default=0, min_value=0)
                            result["pnp_reprojection_error_px"] = _safe_float(
                                geom_diag.get("pnp_reprojection_error_px", np.nan),
                                default=float("nan"),
                            )
                            result["pnp_camera_east_m"] = _safe_float(geom_diag.get("pnp_camera_east_m", np.nan), default=float("nan"))
                            result["pnp_camera_north_m"] = _safe_float(geom_diag.get("pnp_camera_north_m", np.nan), default=float("nan"))
                            result["pnp_camera_z_m"] = _safe_float(geom_diag.get("pnp_camera_z_m", np.nan), default=float("nan"))
                            if result["dsm_status"] in {"dsm_unavailable", "insufficient_valid_dsm_points", "pnp_failed"}:
                                self._stats["dsm_strict_fail_count"] = int(self._stats.get("dsm_strict_fail_count", 0)) + 1
                        if vision_pose is not None:
                            result['visual_pred_lat'] = float(vision_pose.position[0])
                            result['visual_pred_lon'] = float(vision_pose.position[1])
                            result['visual_pred_alt'] = float(vision_pose.position[2])
                            pose_euler_deg = np.degrees(vision_pose.euler_angles)
                            result['visual_pred_roll'] = float(pose_euler_deg[0])
                            result['visual_pred_pitch'] = float(pose_euler_deg[1])
                            result['visual_pred_yaw'] = float(pose_euler_deg[2])
                            anchor_lat = float(anchor.get('lat', telemetry.get('lat', 0.0)))
                            anchor_lon = float(anchor.get('lon', telemetry.get('lon', 0.0)))
                            anchor_offset_m = self._latlon_distance_m(
                                float(result['visual_pred_lat']),
                                float(result['visual_pred_lon']),
                                anchor_lat,
                                anchor_lon,
                            )
                            result["geometry_anchor_offset_m"] = float(anchor_offset_m)
                            # --- Geometry debug fields for GSD/anchor diagnosis ---
                            result['_debug_gsd'] = frame.get('gsd')
                            result['_debug_gsd_fallback'] = float(pipeline['geometry']._gsd) if hasattr(pipeline.get('geometry', object()), '_gsd') else None
                            result['_debug_ref_size'] = (ref_image.shape[1], ref_image.shape[0]) if ref_image is not None else None
                            result['_debug_query_size'] = (query_image.shape[1], query_image.shape[0]) if query_image is not None else None
                            result['_debug_anchor_lat'] = anchor_lat
                            result['_debug_anchor_lon'] = anchor_lon
                            if hasattr(self, '_latlon_distance_m'):
                                result['_debug_gt_lat'] = float(metadata_gt.get('lat', 0.0))
                                result['_debug_gt_lon'] = float(metadata_gt.get('lon', 0.0))
                                result['_debug_anchor_to_gt_m'] = self._latlon_distance_m(anchor_lat, anchor_lon, float(metadata_gt.get('lat', 0.0)), float(metadata_gt.get('lon', 0.0)))
                            max_anchor_offset_m = _safe_float(
                                cfg.get("geometry_max_anchor_offset_m", 400.0),
                                default=400.0,
                                min_value=50.0,
                            )
                            if np.isfinite(anchor_offset_m) and anchor_offset_m > max_anchor_offset_m:
                                result["failure_reason"] = "geometry_anchor_offset_reject"
                                result["geometry_gate_reason"] = "large_visual_anchor_offset"
                                result["bottleneck_label"] = "geometry_fail_after_good_retrieval"
                                vision_pose = None
                            else:
                                if hasattr(self, '_latlon_distance_m'):
                                    result['pre_fgo_error_m'] = self._latlon_distance_m(
                                        float(vision_pose.position[0]),
                                        float(vision_pose.position[1]),
                                        float(metadata_gt.get('lat', 0.0)),
                                        float(metadata_gt.get('lon', 0.0)),
                                    )
                        else:
                            result["geometry_gate_reason"] = "solver_returned_none"
            result['timings']['geometry_ms'] = (time.perf_counter() - t0) * 1000
            
            # === Step 6: FGO update ===
            t0 = time.perf_counter()
            if pipeline.get('fgo_solver') is not None and vision_pose is not None:
                from modules.fusion.fgo_solver import (
                    FGOState,
                    IMUPriorFactor,
                    BarometricFactor,
                    IMUPreintegrationFactor,
                    MotionPriorFactor,
                )
                origin = pipeline.get('local_origin') or {
                    'lat': float(phase1.geodetic_anchor.get('lat', 0.0)) if phase1 else 0.0,
                    'lon': float(phase1.geodetic_anchor.get('lon', 0.0)) if phase1 else 0.0,
                }
                x_local, y_local = self._geodetic_to_local_xy(
                    lat=float(vision_pose.position[0]),
                    lon=float(vision_pose.position[1]),
                    origin_lat=float(origin['lat']),
                    origin_lon=float(origin['lon']),
                )
                initial_state = FGOState(
                    position=np.array([x_local, y_local, float(telemetry.get('alt', 100.0))], dtype=np.float64),
                    orientation=np.array([0.0, 0.0, float(vision_pose.euler_angles[2])], dtype=np.float64),
                )
                pre_fgo_local_pos = np.asarray([x_local, y_local, float(vision_pose.position[2])], dtype=np.float64).copy()
                fgo_state = initial_state
                fgo_success = False
                try:
                    if _safe_int(pipeline.get('fgo_factors_added', 0), default=0, min_value=0) == 0:
                        pipeline['fgo_solver'].clear_factors()
                    elif _safe_int(pipeline.get('fgo_factors_added', 0), default=0, min_value=0) >= _safe_int(pipeline.get('fgo_factor_window', 1), default=1, min_value=1):
                        pipeline['fgo_solver'].clear_factors()
                        pipeline['fgo_factors_added'] = 0
                    pipeline['fgo_solver'].set_initial_state(initial_state)
                    camera_matrix = np.array([
                        [camera.get('fx', 1000), 0, camera.get('cx', 640)],
                        [0, camera.get('fy', 1000), camera.get('cy', 360)],
                        [0, 0, 1],
                    ], dtype=np.float64)
                    visual_factor = self._build_visual_factor_from_ray_plane(
                        match_result=match_result,
                        camera_matrix=camera_matrix,
                        local_position_m=pre_fgo_local_pos,
                        altitude_m=float(telemetry.get('alt', 100.0)),
                        attitude_rad=np.radians([
                            float(telemetry.get('roll', 0.0)),
                            _pitch_from_nadir_deg(
                                float(telemetry.get('pitch', 0.0)),
                                str(metadata_gt.get('pitch_convention', '')),
                            ),
                            float(telemetry.get('yaw', 0.0)),
                        ]),
                    )
                    shape_conf = float(result.get('shape_confidence')) if np.isfinite(result.get('shape_confidence', np.nan)) else np.nan
                    cfg = pipeline.get("config", {})
                    fgo_min_shape_conf = _safe_float(
                        cfg.get("fgo_min_shape_conf", 0.05),
                        default=0.05,
                        min_value=0.0,
                    )
                    max_reproj = _safe_float(cfg.get("max_reproj_residual_px", 6.0), default=6.0, min_value=0.1)
                    reproj_ok = (
                        np.isfinite(result.get("reproj_residual_px", np.nan))
                        and float(result["reproj_residual_px"]) <= (1.5 * max_reproj)
                    )
                    shape_gate_ok = np.isfinite(shape_conf) and shape_conf >= fgo_min_shape_conf
                    robust_visual_ok = shape_gate_ok or (num_inliers >= 20 and reproj_ok)
                    if visual_factor is not None and num_inliers >= 8 and robust_visual_ok:
                        pipeline['fgo_solver'].add_visual_factor(visual_factor)
                        result['visual_factor_points'] = int(len(visual_factor.points_3d))
                        result['visual_sigma'] = float(visual_factor.sigma)
                    else:
                        result['visual_factor_points'] = 0
                    # State pitch is nadir-relative (matches the visual factor's
                    # attitude_rad above); feeding raw gimbal pitch (-90 at
                    # nadir on AVL) here made this 3-sigma prior fight the
                    # visual factor and blow up the FGO cost.
                    _prior_pitch_deg = _pitch_from_nadir_deg(
                        float(telemetry.get('pitch', 0.0)),
                        str(metadata_gt.get('pitch_convention', '')),
                    )
                    pipeline['fgo_solver'].add_imu_prior(
                        IMUPriorFactor(
                            roll_rad=np.radians(telemetry.get('roll', 0.0)),
                            pitch_rad=np.radians(_prior_pitch_deg),
                            yaw_rad=np.radians(telemetry.get('yaw', 0.0)),
                            sigma_roll=0.053,
                            sigma_pitch=0.053,
                            sigma_yaw=0.14,
                        )
                    )
                    pipeline['fgo_solver'].add_barometric_factor(
                        BarometricFactor(
                            altitude=float(telemetry.get('alt', 100.0)),
                            sigma=3.0,
                        )
                    )
                    if pipeline.get('last_fgo_state') is not None:
                        dt = 1.0
                        if pipeline.get('last_timestamp') is not None:
                            dt = max(1e-3, float(frame.get('timestamp', pipeline['last_timestamp'] + 1.0)) - float(pipeline['last_timestamp']))
                        if bool(pipeline.get('fgo_use_imu_preintegration', False)):
                            pipeline['fgo_solver'].add_imu_preintegration(
                                IMUPreintegrationFactor(
                                    prev_state=pipeline['last_fgo_state'],
                                    delta_t=dt,
                                    delta_r=np.array([
                                        np.radians(telemetry.get('roll', 0.0)),
                                        np.radians(_prior_pitch_deg),
                                        np.radians(telemetry.get('yaw', 0.0)),
                                    ]) - pipeline['last_fgo_state'].orientation,
                                    sigma_rot=0.05,
                                    sigma_pos=1.0,
                                    sigma_vel=0.5,
                                    delta_v=np.zeros(3),
                                    delta_p=np.zeros(3),
                                    gravity=np.zeros(3)
                                )
                            )
                        elif not bool(pipeline.get('fgo_preintegration_backend_parity_logged', False)):
                            logger.info(
                                "FGO IMU preintegration skipped (backend parity mode enabled in benchmark)."
                            )
                            pipeline['fgo_preintegration_backend_parity_logged'] = True
                        # Motion-prior tightness must scale with inter-frame
                        # displacement: 8 m suits VPAIR/AVL (~10-20 m/frame) but
                        # drags UAV-VisLoc poses (~75 m/frame) toward the
                        # previous state. Dataset cells override via config.
                        pipeline['fgo_solver'].add_motion_prior(
                            MotionPriorFactor(
                                prev_position=np.asarray(pipeline['last_fgo_state'].position, dtype=np.float64),
                                prev_orientation=np.asarray(pipeline['last_fgo_state'].orientation, dtype=np.float64),
                                sigma_position=_safe_float(
                                    cfg.get("fgo_motion_prior_sigma_pos_m", 8.0),
                                    default=8.0,
                                    min_value=0.1,
                                ),
                                sigma_orientation=0.3,
                            )
                        )
                    # Count this frame's factors BEFORE the guards: counting
                    # only adopted frames left the counter stuck after one
                    # rejection, so the window never cleared, factors from the
                    # whole run piled into one state (cost grew monotonically)
                    # and the step guards then rejected FGO forever.
                    pipeline['fgo_factors_added'] = _safe_int(pipeline.get('fgo_factors_added', 0), default=0, min_value=0) + 1
                    optimized = pipeline['fgo_solver'].optimize()
                    opt_pos = np.asarray(getattr(optimized, "position", np.zeros(3)), dtype=np.float64).reshape(-1)
                    if opt_pos.shape[0] < 3 or not np.isfinite(opt_pos[:3]).all():
                        raise ValueError(FGO_REJECT_NON_FINITE_POS)
                    # Absolute-XY guard removed: opt_pos is in the local frame
                    # pinned to the FIRST frame's anchor, so its norm measures
                    # distance flown, not divergence; with a 2 km bound it
                    # rejected FGO on ~98% of VPAIR frames. The per-step guards
                    # below are the actual divergence test.
                    fgo_step_xy_m = float(np.linalg.norm(opt_pos[:2] - pre_fgo_local_pos[:2]))
                    fgo_step_alt_m = float(abs(opt_pos[2] - pre_fgo_local_pos[2]))
                    result["fgo_step_xy_m"] = fgo_step_xy_m
                    result["fgo_step_alt_m"] = fgo_step_alt_m
                    fgo_max_step_xy_m = _safe_float(
                        cfg.get("fgo_max_step_xy_m", 100.0),
                        default=100.0,
                        min_value=1.0,
                    )
                    fgo_max_step_alt_m = _safe_float(
                        cfg.get("fgo_max_step_alt_m", 70.0),
                        default=70.0,
                        min_value=1.0,
                    )
                    if fgo_step_xy_m > fgo_max_step_xy_m:
                        raise ValueError(FGO_REJECT_STEP_XY)
                    if fgo_step_alt_m > fgo_max_step_alt_m:
                        raise ValueError(FGO_REJECT_STEP_ALT)
                    pred_lat, pred_lon = self._local_xy_to_geodetic(
                        x_east=float(opt_pos[0]),
                        y_north=float(opt_pos[1]),
                        origin_lat=float(origin['lat']),
                        origin_lon=float(origin['lon']),
                    )
                    vision_pose.position[0] = pred_lat
                    vision_pose.position[1] = pred_lon
                    vision_pose.position[2] = float(opt_pos[2])
                    result["fgo_applied"] = True
                    result["fgo_reject_code"] = None
                    result["fgo_guard_reject"] = False
                    fgo_success = True

                    fgo_state = optimized
                    result['fgo_cov_trace'] = float(np.trace(optimized.covariance)) if optimized.covariance is not None else None
                except ValueError as exc:
                    self._stats["fgo_exceptions"] += 1
                    reject_code = str(exc)
                    if reject_code not in FGO_GUARD_REJECT_CODES:
                        reject_code = FGO_REJECT_RUNTIME_EXCEPTION
                    result['fgo_warning'] = f"FGO rejected by guard: {reject_code}"
                    result["fgo_applied"] = False
                    result["fgo_reject_code"] = reject_code
                    result["fgo_guard_reject"] = reject_code in FGO_GUARD_REJECT_CODES
                except Exception as exc:
                    self._stats["fgo_exceptions"] += 1
                    warning_text = str(exc)
                    result['fgo_warning'] = warning_text
                    result["fgo_applied"] = False
                    result["fgo_reject_code"] = FGO_REJECT_RUNTIME_EXCEPTION
                    result["fgo_guard_reject"] = False
                # Keep the motion prior anchored to the PREVIOUS frame even
                # when FGO is rejected: fgo_state falls back to this frame's
                # visual-pose initial_state, so the prior never goes stale.
                if fgo_state is not None:
                    pipeline['last_fgo_state'] = fgo_state
                pipeline['last_timestamp'] = float(frame.get('timestamp', result.get('timings', {}).get('total_ms', 0.0)))
            result['timings']['fgo_ms'] = (time.perf_counter() - t0) * 1000
            
            # Record prediction
            if vision_pose is not None:
                result['pred_lat'] = vision_pose.position[0]
                result['pred_lon'] = vision_pose.position[1]
                result['pred_alt'] = vision_pose.position[2]
                
                try:
                    pred_euler_deg = np.degrees(vision_pose.euler_angles)
                    result['pred_roll'] = float(pred_euler_deg[0])
                    result['pred_pitch'] = float(pred_euler_deg[1])
                    # The yaw removed by the IPM warp is restored inside the pose
                    # solver via rotation_correction_rad, using the yaw actually
                    # applied after per-axis clamping. Adding raw telemetry yaw
                    # here as well would double-count it, and would be wrong on
                    # frames where the warp clamped yaw to zero.
                    raw_heading = float(pred_euler_deg[2])
                    result['pred_heading'] = (raw_heading + 180.0) % 360.0 - 180.0
                except Exception:
                    result['pred_roll'] = float('nan')
                    result['pred_pitch'] = float('nan')
                    result['pred_heading'] = telemetry.get('yaw', 0)
                result['yaw_error_deg'] = abs(_angle_diff_deg(result.get('pred_heading'), metadata_gt.get('yaw', np.nan)))
                # pred_pitch is nadir-relative (solver frame); GT pitch is
                # dataset-native (AVL gimbal reports -90 at nadir), so
                # normalize GT to the same frame before differencing.
                _gt_pitch_nadir = _pitch_from_nadir_deg(
                    float(metadata_gt.get('pitch', np.nan)),
                    str(metadata_gt.get('pitch_convention', '')),
                )
                result['pitch_error_deg'] = abs(_angle_diff_deg(result.get('pred_pitch'), _gt_pitch_nadir))
                result['roll_error_deg'] = abs(_angle_diff_deg(result.get('pred_roll'), metadata_gt.get('roll', np.nan)))
                result['rotation_geodesic_error_deg'] = _rotation_geodesic_error_deg(
                    result.get('pred_roll'),
                    result.get('pred_pitch'),
                    result.get('pred_heading'),
                    metadata_gt.get('roll', np.nan),
                    _gt_pitch_nadir,
                    metadata_gt.get('yaw', np.nan),
                )
                
                result['success'] = True
                result['pose_success'] = True
                result['accepted_for_metrics'] = True
                if hasattr(self, '_latlon_distance_m'):
                    result['post_fgo_error_m'] = self._latlon_distance_m(
                        result['pred_lat'],
                        result['pred_lon'],
                        float(metadata_gt.get('lat', 0.0)),
                        float(metadata_gt.get('lon', 0.0)),
                    )
                
                if np.isfinite(result.get('pre_fgo_error_m', np.nan)) and np.isfinite(result.get('post_fgo_error_m', np.nan)):
                    if float(result['post_fgo_error_m']) > float(result['pre_fgo_error_m']) + 1e-6:
                        result['bottleneck_label'] = 'fusion_degradation'
                    elif result.get('bottleneck_label', 'unknown') == 'unknown':
                        result['bottleneck_label'] = 'healthy'
                elif result.get('bottleneck_label', 'unknown') == 'unknown':
                    result['bottleneck_label'] = 'healthy'

                result['prior_yaw_deg'] = _safe_float(telemetry.get('yaw', np.nan), default=float('nan'))
                final_pose_accepted, final_pose_reason = evaluate_final_pose_acceptance(
                    result,
                    pipeline.get("config", {}),
                )
                result["final_pose_gate_reject"] = not final_pose_accepted
                result["final_pose_gate_reason"] = final_pose_reason
                if not final_pose_accepted:
                    result["failure_reason"] = "final_pose_gate_reject"
                    result["success"] = False
                    result["pose_success"] = False
                    result["accepted_for_metrics"] = False
                    if result.get("bottleneck_label", "unknown") != "retrieval_fail":
                        result["bottleneck_label"] = "geometry_fail_after_good_retrieval"
            else:
                result['pred_lat'] = float('nan')
                result['pred_lon'] = float('nan')
                result['pred_alt'] = float('nan')
                result['pred_heading'] = float('nan')
                result['pred_roll'] = float('nan')
                result['pred_pitch'] = float('nan')
                result['yaw_error_deg'] = float('nan')
                result['pitch_error_deg'] = float('nan')
                result['roll_error_deg'] = float('nan')
                result['rotation_geodesic_error_deg'] = float('nan')
                result['success'] = False
                result['pose_success'] = False
                result['accepted_for_metrics'] = False
                if not result.get('failure_reason'):
                    result['failure_reason'] = 'geometry_failed'
                if result.get('bottleneck_label', 'unknown') == 'unknown':
                    result['bottleneck_label'] = 'geometry_fail_after_good_retrieval'
                
            result['num_inliers'] = num_inliers
            result['num_matches'] = num_matches
            
        except Exception as e:
            logger.debug(f"Frame processing failed: {e}", exc_info=True)
            if not result['success']:
                result['pred_lat'] = float('nan')
                result['pred_lon'] = float('nan')
                result['pred_alt'] = float('nan')
                result['pred_heading'] = float('nan')
            result['failure_reason'] = str(e)
            # Strict retrieval failures must be countable: previously these
            # frames kept bottleneck_label='unknown', so num_retrieval_fail
            # reported 0 against 2,526 real failures on UAV-VisLoc.
            if "retrieval failed" in str(e) and result.get('bottleneck_label', 'unknown') == 'unknown':
                result['bottleneck_label'] = 'retrieval_fail'
            self._stats["frame_exceptions"] += 1
            logger.error(
                "Frame %s failed: %s",
                frame.get('image_name', '?'),
                e,
                exc_info=True,
            )
        
        frame_meta = frame.get("metadata", {}) if isinstance(frame.get("metadata"), dict) else {}
        _finalize_forge_labels(result, str(frame_meta.get("dataset", "")), self._config)
        if bool(result.get("bad_pose_label", False)):
            self._stats["num_bad_pose"] = int(self._stats.get("num_bad_pose", 0)) + 1
        if bool(result.get("unsafe_accept", False)):
            self._stats["num_unsafe_accept"] = int(self._stats.get("num_unsafe_accept", 0)) + 1

        self._stats["processed_frames"] = int(self._stats.get("processed_frames", 0)) + 1
        result['timings']['total_ms'] = (time.perf_counter() - start_total) * 1000
        result['total_time_ms'] = result['timings']['total_ms']

        return result
    
    def run_dataset(self, dataset_name: str) -> Dict[str, Any]:
        """
        Run benchmark on a single dataset.
        
        Args:
            dataset_name: Name from DATASET_CONFIGS
            
        Returns:
            Dict with results and metrics
        """
        start_t = time.time()
        logger.info(f"\n{'='*60}")
        logger.info(f"Running {self._config['name']} on {dataset_name}")
        logger.info(f"{'='*60}")
        
        # Create dataset
        dataset = self._create_dataset(dataset_name)

        # Resolve output early to stream per-frame diagnostics and pre-run audits.
        ds_output = self._resolve_run_output(dataset_name)
        if dataset_name == "uav_visloc" and not self._pipeline_name.startswith("native_"):
            self._write_uav_visloc_anchor_audit(dataset, ds_output)
        
        # Build pipeline
        pipeline = self._build_pipeline(dataset_name)
        self._prepare_retrieval_database(dataset_name, dataset, pipeline)
        self._prepare_dsm_map(dataset_name, dataset, pipeline)
        
        self._write_run_manifest(ds_output, dataset_name)
        if self._retrieval_only:
            logger.info(
                "Retrieval-only diagnostics enabled: dataset=%s pipeline=%s top_k=%s",
                dataset_name,
                self._pipeline_name,
                self._stats.get("retrieval_top_k"),
            )
            return self._run_retrieval_only_dataset(dataset_name, dataset, pipeline, ds_output)
        diagnostics_path = ds_output / "frame_diagnostics.jsonl"

        # Process all frames
        results = []
        from tqdm import tqdm
        
        # Precompute deterministic bias drift trajectory for realistic sensor simulation
        bias_drift = _precompute_bias_drift(len(dataset))
        
        with open(diagnostics_path, "w") as diag_fp:
            for frame_index, frame in enumerate(tqdm(dataset, desc=f"{dataset_name}", total=len(dataset))):
                result = self._process_frame(frame, pipeline, frame_index=frame_index, bias_drift=bias_drift)
                result['gt_lat'] = frame['gt_pose'].get('lat', 0)
                result['gt_lon'] = frame['gt_pose'].get('lon', 0)
                result['gt_alt'] = frame['gt_pose'].get('alt', 0)
                result['gt_heading'] = frame['gt_pose'].get('yaw', 0)
                result['gt_pitch'] = frame['gt_pose'].get('pitch', 0)
                result['gt_roll'] = frame['gt_pose'].get('roll', 0)
                frame_meta = frame.get("metadata", {}) if isinstance(frame.get("metadata"), dict) else {}
                if dataset_name == "uav_visloc":
                    result["flight_id"] = str(frame_meta.get("location", "")) or _flight_id_from_ref_name(
                        str(result.get("image_name", ""))
                    )
                    protocol_fields = resolve_uav_visloc_protocol_fields(
                        result,
                        protocol=self._uav_visloc_protocol,
                        reference_names=pipeline.get("retrieval_reference_names", {}),
                    )
                    result.update(protocol_fields)
                    if bool(result.get("retrieval_protocol_violation", False)):
                        self._stats["retrieval_protocol_violation_count"] = (
                            int(self._stats.get("retrieval_protocol_violation_count", 0)) + 1
                        )
                        if self._strict_end2end:
                            result["pose_success"] = False
                            result["success"] = False
                            result["accepted_for_metrics"] = False
                            result["failure_reason"] = (
                                result.get("failure_reason") or "retrieval_protocol_violation"
                            )
                            result["bottleneck_label"] = "retrieval_protocol_violation"
                anchor = result.get('geodetic_anchor')
                if isinstance(anchor, dict):
                    anchor_lat = float(anchor.get('lat', frame['gt_pose'].get('lat', 0.0)))
                    anchor_lon = float(anchor.get('lon', frame['gt_pose'].get('lon', 0.0)))
                    result['geodetic_anchor'] = anchor
                    a_lat = float(anchor.get('lat', np.nan))
                    a_lon = float(anchor.get('lon', np.nan))
                    if np.isfinite(a_lat) and np.isfinite(a_lon):
                        result['retrieval_anchor_error_m'] = self._latlon_distance_m(
                            a_lat,
                            a_lon,
                            float(result['gt_lat']),
                            float(result['gt_lon']),
                        )
                if dataset_name == "uav_visloc":
                    ref_anchors = pipeline.get("retrieval_reference_anchors", {})
                    for threshold_m in (20, 50, 150, 300):
                        result[f"selected_anchor_hit_at_{threshold_m}m"] = int(
                            np.isfinite(result.get("retrieval_anchor_error_m", np.nan))
                            and float(result.get("retrieval_anchor_error_m")) <= float(threshold_m)
                        )
                        result[f"raw_top1_anchor_hit_at_{threshold_m}m"] = 0
                        result[f"raw_top5_anchor_hit_at_{threshold_m}m"] = 0
                        result[f"raw_top10_anchor_hit_at_{threshold_m}m"] = 0
                        result[f"raw_top20_anchor_hit_at_{threshold_m}m"] = 0
                        result[f"raw_top50_anchor_hit_at_{threshold_m}m"] = 0
                    raw_idx = _safe_int(result.get("raw_retrieval_candidate_index", -1), default=-1)
                    raw_anchor = ref_anchors.get(raw_idx)
                    if isinstance(raw_anchor, dict):
                        raw_err = self._latlon_distance_m(
                            float(raw_anchor.get("lat", np.nan)),
                            float(raw_anchor.get("lon", np.nan)),
                            float(result["gt_lat"]),
                            float(result["gt_lon"]),
                        )
                        if np.isfinite(raw_err):
                            result["raw_retrieval_anchor_error_m"] = raw_err
                            for threshold_m in (20, 50, 150, 300):
                                result[f"raw_top1_anchor_hit_at_{threshold_m}m"] = int(raw_err <= threshold_m)
                    raw_topk_indices = result.get("raw_retrieval_topk_indices", [])
                    if isinstance(raw_topk_indices, list):
                        for k in (5, 10, 20, 50):
                            best_err = float("inf")
                            for idx_val in raw_topk_indices[:k]:
                                ref_anchor = ref_anchors.get(_safe_int(idx_val, default=-1))
                                if not isinstance(ref_anchor, dict):
                                    continue
                                err = self._latlon_distance_m(
                                    float(ref_anchor.get("lat", np.nan)),
                                    float(ref_anchor.get("lon", np.nan)),
                                    float(result["gt_lat"]),
                                    float(result["gt_lon"]),
                                )
                                if np.isfinite(err):
                                    best_err = min(best_err, err)
                            for threshold_m in (20, 50, 150, 300):
                                result[f"raw_top{k}_anchor_hit_at_{threshold_m}m"] = int(best_err <= threshold_m)
                if dataset_name == "vpair":
                    qname = str(result.get("image_name", ""))
                    result["vpair_selected_hit_window1"] = self._vpair_name_hit(
                        qname,
                        str(result.get("selected_reference_name", "")),
                        tolerance=1,
                    )
                    topk_names = []
                    for idx_val in result.get("raw_retrieval_topk_indices", []):
                        ridx = _safe_int(idx_val, default=-1)
                        if ridx < 0:
                            continue
                        topk_names.append(
                            str(pipeline.get("retrieval_reference_names", {}).get(ridx, ""))
                        )
                    result["vpair_raw_top1_hit_window1"] = self._vpair_name_hit(
                        qname,
                        str(result.get("raw_top1_reference_name", "")),
                        tolerance=1,
                    )
                    result["raw_retrieval_topk_hits_window1"] = int(
                        any(self._vpair_name_hit(qname, n, tolerance=1) for n in topk_names[:1])
                    )
                    result["raw_retrieval_topk_hits_window5"] = int(
                        any(self._vpair_name_hit(qname, n, tolerance=1) for n in topk_names[:5])
                    )
                    result["raw_retrieval_topk_hits_window10"] = int(
                        any(self._vpair_name_hit(qname, n, tolerance=1) for n in topk_names[:10])
                    )

                ref_anchors_all = pipeline.get("retrieval_reference_anchors", {})
                gt_tile = self._find_gt_tile_name(
                    float(result.get("gt_lat", np.nan)),
                    float(result.get("gt_lon", np.nan)),
                    dataset_name,
                    ref_anchors_all,
                    query_name=str(result.get("image_name", "")),
                )
                result["gt_tile_name"] = gt_tile or ""
                if gt_tile:
                    sel_name = str(result.get("selected_reference_name", ""))
                    raw_name = str(result.get("raw_top1_reference_name", ""))
                    if dataset_name == "vpair":
                        result["tile_identity_correct"] = int(
                            Path(sel_name).stem == gt_tile
                        )
                        result["raw_top1_tile_identity_correct"] = int(
                            Path(raw_name).stem == gt_tile
                        )
                    else:
                        result["tile_identity_correct"] = int(sel_name == gt_tile)
                        result["raw_top1_tile_identity_correct"] = int(raw_name == gt_tile)
                else:
                    result["tile_identity_correct"] = 0
                    result["raw_top1_tile_identity_correct"] = 0

                results.append(result)
                
                # Periodically clean up garbage to prevent CPU OOM on large datasets (>1000 frames)
                import gc, torch
                if len(results) % 50 == 0:
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                diag_row = {
                    "image_name": result.get("image_name"),
                    "success": result.get("success", False),
                    "pose_success": result.get("pose_success", result.get("success", False)),
                    "accepted_for_metrics": result.get("accepted_for_metrics", result.get("success", False)),
                    "failure_reason": result.get("failure_reason", ""),
                    "retrieval_candidate_index": result.get("retrieval_candidate_index"),
                    "retrieval_similarity_score": result.get("retrieval_similarity_score"),
                    "raw_retrieval_similarity_score": result.get("raw_retrieval_similarity_score"),
                    "retrieval_anchor_error_m": result.get("retrieval_anchor_error_m"),
                    "raw_retrieval_anchor_error_m": result.get("raw_retrieval_anchor_error_m"),
                    "geodetic_anchor": result.get("geodetic_anchor"),
                    "raw_retrieval_anchor": result.get("raw_retrieval_anchor"),
                    "raw_retrieval_candidate_index": result.get("raw_retrieval_candidate_index"),
                    "raw_retrieval_topk_indices": result.get("raw_retrieval_topk_indices"),
                    "query_flight_id": result.get("query_flight_id"),
                    "selected_ref_flight_id": result.get("selected_ref_flight_id"),
                    "raw_top1_ref_flight_id": result.get("raw_top1_ref_flight_id"),
                    "raw_topk_ref_flights": result.get("raw_topk_ref_flights"),
                    "retrieval_protocol_violation": result.get("retrieval_protocol_violation", False),
                    "selected_reference_name": result.get("selected_reference_name"),
                    "raw_top1_reference_name": result.get("raw_top1_reference_name"),
                    "vpair_selected_hit_window1": result.get("vpair_selected_hit_window1"),
                    "vpair_raw_top1_hit_window1": result.get("vpair_raw_top1_hit_window1"),
                    "raw_retrieval_topk_hits_window1": result.get("raw_retrieval_topk_hits_window1"),
                    "raw_retrieval_topk_hits_window5": result.get("raw_retrieval_topk_hits_window5"),
                    "raw_retrieval_topk_hits_window10": result.get("raw_retrieval_topk_hits_window10"),
                    "pre_fgo_error_m": result.get("pre_fgo_error_m"),
                    "post_fgo_error_m": result.get("post_fgo_error_m"),
                    "geometry_anchor_offset_m": result.get("geometry_anchor_offset_m"),
                    "final_pose_gate_reject": result.get("final_pose_gate_reject", False),
                    "final_pose_gate_reason": result.get("final_pose_gate_reason", ""),
                    "_debug_gsd": result.get("_debug_gsd"),
                    "_debug_gsd_fallback": result.get("_debug_gsd_fallback"),
                    "_debug_ref_size": result.get("_debug_ref_size"),
                    "_debug_query_size": result.get("_debug_query_size"),
                    "_debug_anchor_lat": result.get("_debug_anchor_lat"),
                    "_debug_anchor_lon": result.get("_debug_anchor_lon"),
                    "_debug_gt_lat": result.get("_debug_gt_lat"),
                    "_debug_gt_lon": result.get("_debug_gt_lon"),
                    "_debug_anchor_to_gt_m": result.get("_debug_anchor_to_gt_m"),
                    "bottleneck_label": result.get("bottleneck_label"),
                    "phase1_source": result.get("phase1_source"),
                    "retrieval_verification_status": result.get("retrieval_verification_status"),
                    "verification_exception": result.get("verification_exception"),
                    "matcher_backend_effective": result.get("matcher_backend_effective"),
                    "candidate_rank_selected": result.get("candidate_rank_selected"),
                    "candidate_quality_score": result.get("candidate_quality_score"),
                    "adversarial_mode": result.get("adversarial_mode"),
                    "adversarial_condition": result.get("adversarial_condition"),
                    "adversarial_candidate_policy": result.get("adversarial_candidate_policy"),
                    "adversarial_candidate_rank": result.get("adversarial_candidate_rank"),
                    "adversarial_candidate_pool_size": result.get("adversarial_candidate_pool_size"),
                    "adversarial_eligible_wrong_count": result.get("adversarial_eligible_wrong_count"),
                    "adversarial_selection_reason": result.get("adversarial_selection_reason"),
                    "adversarial_failure_reason": result.get("adversarial_failure_reason"),
                    "adversarial_forced_reference_name": result.get("adversarial_forced_reference_name"),
                    "adversarial_original_reference_name": result.get("adversarial_original_reference_name"),
                    "adversarial_forced_distance_to_gt_m": result.get("adversarial_forced_distance_to_gt_m"),
                    "adversarial_offset_m": result.get("adversarial_offset_m"),
                    "is_adversarial": result.get("is_adversarial"),
                    "is_wrong_reference": result.get("is_wrong_reference"),
                    "retrieval_correct_threshold_m": result.get("retrieval_correct_threshold_m"),
                    "localization_success_threshold_m": result.get("localization_success_threshold_m"),
                    "retrieval_rank": result.get("retrieval_rank"),
                    "localization_is_success": result.get("localization_is_success"),
                    "caves_reject": result.get("caves_reject"),
                    "false_reject": result.get("false_reject"),
                    "bad_pose_label": result.get("bad_pose_label"),
                    "unsafe_accept": result.get("unsafe_accept"),
                    "unsafe_accept_severity_m": result.get("unsafe_accept_severity_m"),
                    "caves_risk_score": result.get("caves_risk_score"),
                    "caves_confidence": result.get("caves_confidence"),
                    "caves_decision": result.get("caves_decision"),
                    "caves_evidence_count": result.get("caves_evidence_count"),
                    "obs_gate_score": result.get("obs_gate_score"),
                    "obs_gate_reason": result.get("obs_gate_reason"),
                    "reproj_residual_px": result.get("reproj_residual_px"),
                    "gate_decision": result.get("gate_decision"),
                    "gate_reason": result.get("gate_reason"),
                    "num_matches": result.get("num_matches", 0),
                    "num_inliers": result.get("num_inliers", 0),
                    "match_confidence_mean": result.get("match_confidence_mean", 0.0),
                    "shape_confidence": result.get("shape_confidence"),
                    "homography_valid": result.get("homography_valid", False),
                    "timings_ms": result.get("timings", {}),
                    "pred": {
                        "lat": result.get("pred_lat"),
                        "lon": result.get("pred_lon"),
                        "alt": result.get("pred_alt"),
                        "heading": result.get("pred_heading"),
                    },
                    "gt": {
                        "lat": result.get("gt_lat"),
                        "lon": result.get("gt_lon"),
                        "alt": result.get("gt_alt"),
                        "heading": result.get("gt_heading"),
                        "pitch": result.get("gt_pitch"),
                        "roll": result.get("gt_roll"),
                    },
                }
                if dataset_name == "uav_visloc":
                    diag_row["flight_id"] = result.get("flight_id")
                    for threshold_m in (20, 50, 150, 300):
                        diag_row[f"selected_anchor_hit_at_{threshold_m}m"] = result.get(
                            f"selected_anchor_hit_at_{threshold_m}m"
                        )
                        diag_row[f"raw_top1_anchor_hit_at_{threshold_m}m"] = result.get(
                            f"raw_top1_anchor_hit_at_{threshold_m}m"
                        )
                        diag_row[f"raw_top5_anchor_hit_at_{threshold_m}m"] = result.get(
                            f"raw_top5_anchor_hit_at_{threshold_m}m"
                        )
                        diag_row[f"raw_top10_anchor_hit_at_{threshold_m}m"] = result.get(
                            f"raw_top10_anchor_hit_at_{threshold_m}m"
                        )
                        diag_row[f"raw_top20_anchor_hit_at_{threshold_m}m"] = result.get(
                            f"raw_top20_anchor_hit_at_{threshold_m}m"
                        )
                        diag_row[f"raw_top50_anchor_hit_at_{threshold_m}m"] = result.get(
                            f"raw_top50_anchor_hit_at_{threshold_m}m"
                        )
                diag_fp.write(json.dumps(diag_row, cls=NpEncoder) + "\n")
        
        # Compute metrics
        from evaluation.metrics import (
            MetricsCalculator, SinglePrediction,
            ProposalMetricsCalculator, generate_cdf_plot
        )
        
        predictions = []
        proposal_calc = ProposalMetricsCalculator()
        
        for r in results:
            pred = SinglePrediction(
                image_name=r['image_name'],
                pred_lat=r['pred_lat'],
                pred_lon=r['pred_lon'],
                pred_alt=r['pred_alt'],
                pred_heading=r['pred_heading'],
                gt_lat=r['gt_lat'],
                gt_lon=r['gt_lon'],
                gt_alt=r['gt_alt'],
                gt_heading=r['gt_heading'],
                num_inliers=r.get('num_inliers', 0),
                total_time_ms=r.get('total_time_ms', 0),
                retrieval_time_ms=r.get('timings', {}).get('retrieval_ms', 0),
                matching_time_ms=r.get('timings', {}).get('matching_ms', 0),
                geometry_time_ms=r.get('timings', {}).get('geometry_ms', 0),
                success=bool(r.get('accepted_for_metrics', r.get('success', False))),
                failure_reason=r.get('failure_reason', ''),
            )
            predictions.append(pred)
            
            # Add to proposal metrics
            proposal_calc.add_frame(
                raw_position_error=(
                    self._latlon_distance_m(
                        float(r.get('visual_pred_lat')),
                        float(r.get('visual_pred_lon')),
                        float(r['gt_lat']),
                        float(r['gt_lon']),
                    ) if (np.isfinite(r.get('visual_pred_lat', np.nan)) and np.isfinite(r.get('visual_pred_lon', np.nan)) and pred.success)
                    else float('inf')
                ),
                fgo_position_error=pred.position_error_2d_m if pred.success else float('inf'),
                homography_valid=r.get('homography_valid', False),
                pitch_angle_deg=r.get('gt_pitch', 0),
            )
        
        # Standard metrics
        calc = MetricsCalculator(predictions)
        metrics = calc.compute()
        
        # Proposal metrics
        proposal_metrics = proposal_calc.compute()
        fgo_errors = [
            float(p.position_error_2d_m)
            for p in predictions
            if p.success and np.isfinite(p.position_error_2d_m)
        ]
        e_me_median = float(np.median(fgo_errors)) if fgo_errors else float("nan")
        e_me_p90 = float(np.percentile(fgo_errors, 90)) if fgo_errors else float("nan")
        e_me_trimmed = (
            float(np.mean(sorted(fgo_errors)[1:-1]))
            if len(fgo_errors) >= 3
            else (float(np.mean(fgo_errors)) if fgo_errors else float("nan"))
        )
        def _finite_result_values(key: str) -> List[float]:
            vals: List[float] = []
            for row in results:
                if not bool(row.get('accepted_for_metrics', row.get('success', False))):
                    continue
                v = _safe_float(row.get(key, np.nan), default=float("nan"))
                if np.isfinite(v):
                    vals.append(float(v))
            return vals

        yaw_errors = _finite_result_values("yaw_error_deg")
        pitch_errors = _finite_result_values("pitch_error_deg")
        roll_errors = _finite_result_values("roll_error_deg")
        rot_errors = _finite_result_values("rotation_geodesic_error_deg")
        pnp_inliers = _finite_result_values("pnp_inliers")
        pnp_reproj = _finite_result_values("pnp_reprojection_error_px")

        retrieval_threshold_m = DATASET_RETRIEVAL_THRESHOLD_M.get(dataset_name, 20.0)
        retrieval_candidates = []
        retrieval_hits = 0
        raw_retrieval_candidates = []
        raw_retrieval_hits = 0
        for r in results:
            anchor = r.get('geodetic_anchor')
            if not isinstance(anchor, dict):
                continue
            lat = float(anchor.get('lat', np.nan))
            lon = float(anchor.get('lon', np.nan))
            if not np.isfinite(lat) or not np.isfinite(lon):
                continue
            retrieval_candidates.append(r)
            dist_m = self._latlon_distance_m(lat, lon, float(r['gt_lat']), float(r['gt_lon']))
            r['retrieval_anchor_error_m'] = dist_m
            if dist_m <= retrieval_threshold_m:
                retrieval_hits += 1
            raw_anchor = r.get('raw_retrieval_anchor')
            if isinstance(raw_anchor, dict):
                rlat = float(raw_anchor.get('lat', np.nan))
                rlon = float(raw_anchor.get('lon', np.nan))
                if np.isfinite(rlat) and np.isfinite(rlon):
                    raw_retrieval_candidates.append(r)
                    raw_dist = self._latlon_distance_m(rlat, rlon, float(r['gt_lat']), float(r['gt_lon']))
                    r['raw_retrieval_anchor_error_m'] = raw_dist
                    if raw_dist <= retrieval_threshold_m:
                        raw_retrieval_hits += 1
        # Denominator = all evaluated frames: frames without a usable anchor
        # count as misses. The previous len(retrieval_candidates) denominator
        # was method-dependent (4,820 vs 4,240 rows on the same 6,766
        # UAV-VisLoc queries) and inflated recall 1.4-1.6x.
        n_all_frames = len(results)
        recall_at_1 = (
            float(retrieval_hits / n_all_frames)
            if n_all_frames else 0.0
        )
        raw_recall_at_1 = (
            float(raw_retrieval_hits / n_all_frames)
            if n_all_frames else 0.0
        )
        vpair_raw_top1_recall_w1 = float("nan")
        vpair_raw_top5_recall_w1 = float("nan")
        vpair_raw_top10_recall_w1 = float("nan")
        vpair_selected_hit_w1 = float("nan")
        if dataset_name == "vpair" and len(results) > 0:
            vpair_raw_top1_recall_w1 = float(np.mean([
                float(r.get("raw_retrieval_topk_hits_window1", 0)) for r in results
            ]))
            vpair_raw_top5_recall_w1 = float(np.mean([
                float(r.get("raw_retrieval_topk_hits_window5", 0)) for r in results
            ]))
            vpair_raw_top10_recall_w1 = float(np.mean([
                float(r.get("raw_retrieval_topk_hits_window10", 0)) for r in results
            ]))
            vpair_selected_hit_w1 = float(np.mean([
                float(r.get("vpair_selected_hit_window1", 0)) for r in results
            ]))
        retrieval_diagnostics = None
        uav_retrieval_diagnostics = None
        if dataset_name in {"uav_visloc", "vpair"}:
            retrieval_diagnostics = compute_retrieval_diagnostics(
                results,
                pipeline.get("retrieval_reference_anchors", {}),
                dataset_name=dataset_name,
            )
        if dataset_name == "uav_visloc":
            uav_retrieval_diagnostics = retrieval_diagnostics
        step_metrics_summary = compute_step_metrics_summary(results)
        
        # Save results
        calc.save_results(ds_output / 'results.json')
        with open(ds_output / 'results_detailed.json', 'w') as f:
            json.dump({
                'metrics': metrics.to_dict(),
                'predictions': results,
            }, f, indent=2, cls=NpEncoder)
        
        # Save proposal metrics
        with open(ds_output / 'proposal_metrics.json', 'w') as f:
            json.dump(proposal_metrics.to_dict(), f, indent=2, cls=NpEncoder)

        # Paper-facing altitude metrics:
        # - E_alt_gt: post-fusion altitude vs GT altitude
        # - E_alt: proxy for |h_baro - h_ray| using GT altitude as barometric proxy
        alt_errors_gt = [
            abs(r['pred_alt'] - r['gt_alt'])
            for r in results
            if r.get('success', False) and np.isfinite(r.get('pred_alt', np.nan)) and np.isfinite(r.get('gt_alt', np.nan))
        ]
        e_alt_gt = float(np.mean(alt_errors_gt)) if len(alt_errors_gt) > 0 else float("inf")
        ray_alt_values = [
            float(r.get('visual_pred_alt', np.nan))
            for r in results
            if r.get('success', False) and np.isfinite(r.get('visual_pred_alt', np.nan))
        ]
        alt_errors_baro_vs_ray = [
            abs(float(r.get('gt_alt', np.nan)) - float(r.get('visual_pred_alt', np.nan)))
            for r in results
            if r.get('success', False) and np.isfinite(r.get('visual_pred_alt', np.nan)) and np.isfinite(r.get('gt_alt', np.nan))
        ]
        e_alt = float(np.mean(alt_errors_baro_vs_ray)) if len(alt_errors_baro_vs_ray) > 0 else float("inf")
        if len(ray_alt_values) >= 3 and float(np.std(ray_alt_values)) < 1e-3:
            e_alt = float("nan")
        e_alt_available = bool(np.isfinite(e_alt))
        valid_shape_conf = [
            float(r.get('shape_confidence'))
            for r in results
            if np.isfinite(r.get('shape_confidence', np.nan))
        ]
        retrieval_oracle_fallback_frames = int(self._stats.get("retrieval_oracle_fallback_frames", 0))
        end2end_oracle_fallback_contaminated = bool(
            self._eval_mode == "end_to_end" and retrieval_oracle_fallback_frames > 0
        )
        if end2end_oracle_fallback_contaminated:
            logger.warning(
                "End-to-end integrity warning: %d frame(s) used dataset-paired retrieval fallback. "
                "This run is contaminated for official strict E2E thesis metrics.",
                retrieval_oracle_fallback_frames,
            )
        pipeline_meta = resolve_pipeline_metadata(self._pipeline_name)
        native_prior_uses_exact_gt = bool(self._stats.get("native_prior_uses_exact_gt", False))
        native_anyvisloc_schema_ok = bool(self._stats.get("native_anyvisloc_schema_ok", True))
        retrieval_protocol_violation_count = int(
            self._stats.get("retrieval_protocol_violation_count", 0)
        )
        retrieval_protocol_filtered_candidates_count = int(
            self._stats.get("retrieval_protocol_filtered_candidates_count", 0)
        )
        adversarial_enabled = bool(self._adversarial_config.get("is_adversarial_enabled", False))
        adversarial_diagnostic = bool(adversarial_enabled)
        native_integrity_ok = bool(
            pipeline_meta.get("method_family") != "native_anyvisloc"
            or (not native_prior_uses_exact_gt and native_anyvisloc_schema_ok)
        )
        strict_table_tier = str(pipeline_meta.get("comparison_tier", "strict_e2e")) in {"strict_e2e", "wrapper_e2e"}
        num_caves_reject = int(sum(1 for r in results if bool(r.get("caves_reject", False))))
        num_false_reject = int(sum(1 for r in results if bool(r.get("false_reject", False))))
        num_localization_success = int(sum(1 for r in results if bool(r.get("localization_is_success", False))))
        caves_denominator = int(self._stats.get("num_adversarial_samples", 0)) if adversarial_diagnostic else len(results)
        unsafe_accept_severities = [
            float(_safe_float(r.get("unsafe_accept_severity_m", np.nan), default=float("nan")))
            for r in results
            if bool(r.get("unsafe_accept", False))
            and np.isfinite(_safe_float(r.get("unsafe_accept_severity_m", np.nan), default=float("nan")))
        ]
        paper_metrics = {
            **pipeline_meta,
            "M-ME": _json_number(proposal_metrics.m_me),
            "M_ME": _json_number(proposal_metrics.m_me),
            "HR": _json_number(proposal_metrics.hr),
            "E-ME": _json_number(proposal_metrics.e_me),
            "E_ME": _json_number(proposal_metrics.e_me),
            "E-ME_median": _json_number(e_me_median),
            "E_ME_median": _json_number(e_me_median),
            "E-ME_p90": _json_number(e_me_p90),
            "E_ME_p90": _json_number(e_me_p90),
            "p90_error_m": _json_number(e_me_p90),
            "E-ME_trimmed_mean": _json_number(e_me_trimmed),
            "E_ME_trimmed_mean": _json_number(e_me_trimmed),
            "yaw_error_deg_mean": _json_number(float(np.mean(yaw_errors)) if yaw_errors else float("nan")),
            "yaw_error_deg_median": _json_number(float(np.median(yaw_errors)) if yaw_errors else float("nan")),
            "pitch_error_deg_mean": _json_number(float(np.mean(pitch_errors)) if pitch_errors else float("nan")),
            "pitch_error_deg_median": _json_number(float(np.median(pitch_errors)) if pitch_errors else float("nan")),
            "roll_error_deg_mean": _json_number(float(np.mean(roll_errors)) if roll_errors else float("nan")),
            "roll_error_deg_median": _json_number(float(np.median(roll_errors)) if roll_errors else float("nan")),
            "rotation_geodesic_error_deg_mean": _json_number(float(np.mean(rot_errors)) if rot_errors else float("nan")),
            "rotation_geodesic_error_deg_median": _json_number(float(np.median(rot_errors)) if rot_errors else float("nan")),
            "pnp_inlier_count_mean": _json_number(float(np.mean(pnp_inliers)) if pnp_inliers else float("nan")),
            "pnp_reprojection_error_mean": _json_number(float(np.mean(pnp_reproj)) if pnp_reproj else float("nan")),
            "dsm_available": bool(self._stats.get("dsm_available", False)),
            "dsm_status": str(self._stats.get("dsm_status", "")),
            "dsm_strict_fail_count": int(self._stats.get("dsm_strict_fail_count", 0)),
            "E_alt": _json_number(e_alt),
            "E_alt_available": e_alt_available,
            "E_alt_gt": _json_number(e_alt_gt),
            "Recall@1": _json_number(recall_at_1),
            "selected_anchor_hit_at_threshold": _json_number(recall_at_1),
            "raw_retrieval_recall_at1_threshold": _json_number(raw_recall_at_1),
            "Recall@1_20m": _json_number(recall_at_1),
            "selected_anchor_hit_at20m": _json_number(recall_at_1),
            "raw_retrieval_recall_at1_20m": _json_number(raw_recall_at_1),
            "recall_at_1_threshold_m": retrieval_threshold_m,
            "retrieval_only": bool(getattr(self, "_retrieval_only", False)),
            "uav_visloc_protocol": self._uav_visloc_protocol if dataset_name == "uav_visloc" else None,
            "uav_tile_size_px": self._stats.get("uav_tile_size_px") if dataset_name == "uav_visloc" else None,
            "uav_tile_step_px": self._stats.get("uav_tile_step_px") if dataset_name == "uav_visloc" else None,
            "retrieval_top_k": self._stats.get("retrieval_top_k"),
            "benchmark_family": "forge" if adversarial_diagnostic else pipeline_meta.get("method_family", ""),
            "adversarial_mode": self._adversarial_config.get("adversarial_mode", "none"),
            "adversarial_condition": self._adversarial_config.get("adversarial_condition", "clean"),
            "adversarial_candidate_policy": self._adversarial_config.get("adversarial_candidate_policy", ""),
            "adversarial_candidate_rank": self._adversarial_config.get("adversarial_candidate_rank"),
            "adversarial_seed": self._adversarial_config.get("adversarial_seed"),
            "adversarial_inject_rate": self._adversarial_config.get("adversarial_inject_rate", 1.0),
            "adversarial_official_metrics": False,
            "adversarial_diagnostic": adversarial_diagnostic,
            "official_metric_eligible": False if adversarial_diagnostic else True,
            **_final_pose_gate_metadata(self._config),
            "num_adversarial_samples": int(self._stats.get("num_adversarial_samples", 0)),
            "num_wrong_reference": int(self._stats.get("num_wrong_reference", 0)),
            "num_bad_pose": int(self._stats.get("num_bad_pose", 0)),
            "num_unsafe_accept": int(self._stats.get("num_unsafe_accept", 0)),
            "unsafe_accept_rate": (
                float(self._stats.get("num_unsafe_accept", 0)) / max(1, int(self._stats.get("num_adversarial_samples", 0)))
                if adversarial_diagnostic else 0.0
            ),
            "num_false_reject": num_false_reject,
            "false_rejection_rate": float(num_false_reject / max(1, num_localization_success)),
            "num_caves_reject": num_caves_reject,
            "rejection_rate": float(num_caves_reject / max(1, caves_denominator)),
            "caves_rejection_rate_denominator": "adversarial_samples" if adversarial_diagnostic else "frames",
            "mean_unsafe_accept_severity_m": _json_number(float(np.mean(unsafe_accept_severities)) if unsafe_accept_severities else float("nan")),
            "median_unsafe_accept_severity_m": _json_number(float(np.median(unsafe_accept_severities)) if unsafe_accept_severities else float("nan")),
            "p90_unsafe_accept_severity_m": _json_number(float(np.percentile(unsafe_accept_severities, 90)) if unsafe_accept_severities else float("nan")),
            "max_unsafe_accept_severity_m": _json_number(float(np.max(unsafe_accept_severities)) if unsafe_accept_severities else float("nan")),
            "num_adversarial_fail_closed": int(self._stats.get("num_adversarial_fail_closed", 0)),
            "num_adversarial_unsupported_condition": int(self._stats.get("num_adversarial_unsupported_condition", 0)),
            "num_adversarial_metadata_unavailable": int(self._stats.get("num_adversarial_metadata_unavailable", 0)),
            "num_adversarial_policy_force_rank_k": int(self._stats.get("num_adversarial_policy_force_rank_k", 0)),
            "num_adversarial_policy_nearest_wrong": int(self._stats.get("num_adversarial_policy_nearest_wrong", 0)),
            "num_adversarial_policy_far_wrong": int(self._stats.get("num_adversarial_policy_far_wrong", 0)),
            "num_adversarial_policy_visually_similar_wrong": int(self._stats.get("num_adversarial_policy_visually_similar_wrong", 0)),
            "num_adversarial_policy_cross_flight": int(self._stats.get("num_adversarial_policy_cross_flight", 0)),
            "uav_raw_top1_hit_at_150m": (
                (uav_retrieval_diagnostics or {}).get("global", {}).get("raw_top1_error_m_hit_at_150m")
                if dataset_name == "uav_visloc" else None
            ),
            "uav_raw_top5_hit_at_150m": (
                (uav_retrieval_diagnostics or {}).get("global", {}).get("raw_top5_error_m_hit_at_150m")
                if dataset_name == "uav_visloc" else None
            ),
            "uav_raw_top10_hit_at_150m": (
                (uav_retrieval_diagnostics or {}).get("global", {}).get("raw_top10_error_m_hit_at_150m")
                if dataset_name == "uav_visloc" else None
            ),
            "uav_raw_top20_hit_at_150m": (
                (uav_retrieval_diagnostics or {}).get("global", {}).get("raw_top20_error_m_hit_at_150m")
                if dataset_name == "uav_visloc" else None
            ),
            "uav_raw_top50_hit_at_150m": (
                (uav_retrieval_diagnostics or {}).get("global", {}).get("raw_top50_error_m_hit_at_150m")
                if dataset_name == "uav_visloc" else None
            ),
            "uav_nearest_tile_oracle_floor_p50_m": (
                (uav_retrieval_diagnostics or {}).get("global", {}).get("nearest_tile_oracle_floor_m_p50")
                if dataset_name == "uav_visloc" else None
            ),
            "num_retrieval_eval": len(retrieval_candidates),
            "num_raw_retrieval_eval": len(raw_retrieval_candidates),
            "vpair_raw_recall_at1_index_window1": _json_number(vpair_raw_top1_recall_w1),
            "vpair_raw_recall_at5_index_window1": _json_number(vpair_raw_top5_recall_w1),
            "vpair_raw_recall_at10_index_window1": _json_number(vpair_raw_top10_recall_w1),
            "vpair_selected_hit_index_window1": _json_number(vpair_selected_hit_w1),
            "tile_identity_rate": _json_number(
                100.0 * sum(1 for r in results if r.get("tile_identity_correct", 0))
                / max(1, len(results))
            ) if results else 0.0,
            "raw_top1_tile_identity_rate": _json_number(
                100.0 * sum(1 for r in results if r.get("raw_top1_tile_identity_correct", 0))
                / max(1, len(results))
            ) if results else 0.0,
            "mean_inliers": _json_number(float(np.mean([r.get('num_inliers', 0) for r in results])) if results else 0.0),
            "mean_match_confidence": _json_number(float(np.mean([r.get('match_confidence_mean', 0.0) for r in results])) if results else 0.0),
            "mean_shape_confidence": _json_number(float(np.mean(valid_shape_conf)) if len(valid_shape_conf) > 0 else float("nan")),
            "A@5m": _json_number(metrics.accuracy_at_5m),
            "A@10m": _json_number(metrics.accuracy_at_10m),
            "accuracy_at_5m": _json_number(metrics.accuracy_at_5m),
            "accuracy_at_10m": _json_number(metrics.accuracy_at_10m),
            "median_error_m": _json_number(metrics.median_error_m),
            "FPS": _json_number(metrics.fps),
            "frame_exceptions": int(self._stats.get("frame_exceptions", 0)),
            "phase1_verify_exceptions": int(self._stats.get("phase1_verify_exceptions", 0)),
            "official_fail_closed_retrieval_reject_frames": int(self._stats.get("official_fail_closed_retrieval_reject_frames", 0)),
            "fgo_exceptions": int(self._stats.get("fgo_exceptions", 0)),
            "retrieval_oracle_fallback_frames": retrieval_oracle_fallback_frames,
            "roma_fallback_frames": int(self._stats.get("roma_fallback_frames", 0)),
            "num_reference_preconditioning_attempted": int(self._stats.get("reference_preconditioning_attempted", 0)),
            "num_reference_preconditioning_accepted": int(self._stats.get("reference_preconditioning_accepted", 0)),
            "num_reference_preconditioning_rejected": int(self._stats.get("reference_preconditioning_rejected", 0)),
            "reference_preconditioning_accept_rate": (
                float(self._stats.get("reference_preconditioning_accepted", 0)) /
                max(1, int(self._stats.get("reference_preconditioning_attempted", 0)))
            ),
            "use_reference_preconditioning": bool(self._config.get("use_reference_preconditioning", self._config.get("enable_adhop_preconditioning", False))),
            "fgo_backend_policy": str(self._stats.get("fgo_backend_policy", FGO_BACKEND_POLICY_PREFER_SCIPY)),
            "effective_fgo_backend": str(self._stats.get("effective_fgo_backend", "disabled")),
            "fgo_backend_parity_ok": bool(self._stats.get("fgo_backend_parity_ok", False)),
            "fgo_backend_fallback_reason": str(self._stats.get("fgo_backend_fallback_reason", "")),
            "retrieval_db_cache_enabled": bool(self._stats.get("retrieval_db_cache_enabled", False)),
            "retrieval_db_cache_hit": bool(self._stats.get("retrieval_db_cache_hit", False)),
            "retrieval_db_cache_key": str(self._stats.get("retrieval_db_cache_key", "")),
            "retrieval_db_cache_rebuild_reason": str(self._stats.get("retrieval_db_cache_rebuild_reason", "")),
            "retrieval_protocol_violation_count": retrieval_protocol_violation_count,
            "retrieval_protocol_filtered_candidates_count": retrieval_protocol_filtered_candidates_count,
            "uav_selected_query_locations": list(self._stats.get("uav_selected_query_locations", [])),
            "uav_selected_sample_signature": dict(self._stats.get("uav_selected_sample_signature", {})),
            "uav_corpus_ref_flight_hash": str(self._stats.get("uav_corpus_ref_flight_hash", "")),
            "native_anyvisloc_cache_requested": bool(self._stats.get("native_anyvisloc_cache_requested", False)),
            "native_anyvisloc_cache_enabled": bool(self._stats.get("native_anyvisloc_cache_enabled", False)),
            "native_anyvisloc_cache_key": str(self._stats.get("native_anyvisloc_cache_key", "")),
            "native_anyvisloc_cache_hit": bool(self._stats.get("native_anyvisloc_cache_hit", False)),
            "native_anyvisloc_cache_rebuild_reason": str(self._stats.get("native_anyvisloc_cache_rebuild_reason", "")),
            "end2end_oracle_fallback_contaminated": end2end_oracle_fallback_contaminated,
            "telemetry_prior_policy": str(self._stats.get("telemetry_prior_policy", "simulated_uav_telemetry")),
            "native_prior_uses_exact_gt": native_prior_uses_exact_gt,
            "native_anyvisloc_schema_ok": native_anyvisloc_schema_ok,
            "native_yaw_height_status": str(self._stats.get("native_yaw_height_status", "not_applicable")),
            "run_integrity_for_official_metrics": bool(
                not end2end_oracle_fallback_contaminated
                and self._eval_mode != "oracle_reference"
                and not bool(pipeline_meta.get("oracle_mode", False))
                and not bool(pipeline_meta.get("uses_oracle", False))
                and not bool(pipeline_meta.get("deprecated_alias", False))
                and strict_table_tier
                and not bool(getattr(self, "_retrieval_only", False))
                and not adversarial_diagnostic
                and native_integrity_ok
                and retrieval_protocol_violation_count == 0
                # filtered_candidates_count > 0 = correct intra_flight
                # enforcement, not an integrity failure (see manifest site).
            ),
            "official_success_uses_gt_position_error": False,
            "diagnostic_threshold_sweeps": {
                "geometry_min_inliers_after_lgi": [4, 6, 8],
                "max_reproj_residual_px": [6, 10, 15],
                "used_for_official_success": False,
            },
            "num_frames": len(results),
            "num_success": sum(1 for r in results if bool(r.get('accepted_for_metrics', r.get('success', False)))),
            "num_pose_success": sum(1 for r in results if bool(r.get('pose_success', r.get('success', False)))),
            "num_accepted_for_metrics": sum(1 for r in results if bool(r.get('accepted_for_metrics', r.get('success', False)))),
            "num_retrieval_fail": int(sum(1 for r in results if r.get("bottleneck_label") == "retrieval_fail")),
            "num_geometry_fail_after_good_retrieval": int(sum(1 for r in results if r.get("bottleneck_label") == "geometry_fail_after_good_retrieval")),
            "num_fusion_degradation": int(sum(1 for r in results if r.get("bottleneck_label") == "fusion_degradation")),
            "num_healthy": int(sum(1 for r in results if r.get("bottleneck_label") == "healthy")),
            "num_geometry_reject_reproj": int(sum(1 for r in results if r.get("failure_reason") == "geometry_reproj_gate_reject")),
            "num_geometry_reject_inlier": int(sum(1 for r in results if r.get("failure_reason") == "geometry_inlier_gate_reject")),
            "num_geometry_reject_anchor_offset": int(sum(1 for r in results if r.get("failure_reason") == "geometry_anchor_offset_reject")),
            "num_position_error_cap_reject": int(sum(1 for r in results if r.get("failure_reason") == "position_error_cap_reject")),
            "num_pose_rejected_by_final_gate": count_final_pose_gate_rejects(results),
            "num_fgo_rejected_by_guard": count_fgo_guard_rejects(results),
            "num_fgo_runtime_exceptions": int(sum(1 for r in results if r.get("fgo_reject_code") == FGO_REJECT_RUNTIME_EXCEPTION)),
            "num_fgo_rejected_or_exception_legacy": int(sum(1 for r in results if r.get("fgo_applied") is False and r.get("fgo_warning"))),
            "phase_retrieval_ms_mean": _json_number(float(np.mean([float(r.get("timings", {}).get("retrieval_ms", 0.0)) for r in results])) if results else 0.0),
            "phase_matching_ms_mean": _json_number(float(np.mean([float(r.get("timings", {}).get("matching_ms", 0.0)) for r in results])) if results else 0.0),
            "phase_geometry_ms_mean": _json_number(float(np.mean([float(r.get("timings", {}).get("geometry_ms", 0.0)) for r in results])) if results else 0.0),
            "phase_fgo_ms_mean": _json_number(float(np.mean([float(r.get("timings", {}).get("fgo_ms", 0.0)) for r in results])) if results else 0.0),
            "step_retrieval_success_rate": (step_metrics_summary.get("steps", {}).get("retrieval", {}) or {}).get("success_rate"),
            "step_matching_success_rate": (step_metrics_summary.get("steps", {}).get("matching", {}) or {}).get("success_rate"),
            "step_geometry_success_rate": (step_metrics_summary.get("steps", {}).get("geometry", {}) or {}).get("success_rate"),
            "step_fgo_success_rate": (step_metrics_summary.get("steps", {}).get("fgo", {}) or {}).get("success_rate"),
        }
        paper_metrics["official_metric_eligible"] = bool(paper_metrics.get("run_integrity_for_official_metrics", False))
        with open(ds_output / "paper_metrics_summary.json", "w") as f:
            json.dump(paper_metrics, f, indent=2, cls=NpEncoder)
        with open(ds_output / "step_metrics_summary.json", "w", encoding="utf-8") as f:
            json.dump(step_metrics_summary, f, indent=2, cls=NpEncoder)
        log_step_metrics_summary(step_metrics_summary, dataset=dataset_name, pipeline=self._pipeline_name)
        if uav_retrieval_diagnostics is not None:
            with open(ds_output / "uav_visloc_retrieval_diagnostics.json", "w", encoding="utf-8") as f:
                json.dump(uav_retrieval_diagnostics, f, indent=2, cls=NpEncoder)
        if retrieval_diagnostics is not None:
            with open(ds_output / "retrieval_diagnostics.json", "w", encoding="utf-8") as f:
                json.dump(retrieval_diagnostics, f, indent=2, cls=NpEncoder)

        write_predictions_csv(ds_output / "predictions.csv", results)
        write_metrics_summary_json(
            ds_output / "metrics_summary.json",
            dataset=dataset_name,
            pipeline=self._pipeline_name,
            metrics=metrics.to_dict(),
            paper_metrics=paper_metrics,
            results=results,
            manifest=self._build_run_manifest_payload(dataset_name),
            step_metrics=step_metrics_summary,
        )
        self._write_run_manifest(ds_output, dataset_name, run_completed=True)
        
        # Generate CDF plot
        if len(metrics.position_errors) > 0:
            generate_cdf_plot(
                errors=[metrics.position_errors],
                output_path=ds_output / 'cdf_position.png',
                labels=[self._config['name']],
                title=f"Position Error CDF — {dataset_name}",
                thresholds=[5, 10, 20, 50],
            )
        
        # Print summary
        logger.info(f"\n{metrics}")
        e_alt_str = f"{e_alt:.2f}m" if np.isfinite(e_alt) else "n/a"
        unit = "m"
        logger.info(f"Proposal: M-ME={paper_metrics['M_ME']}{unit}, HR={paper_metrics['HR']}%, E-ME={paper_metrics['E_ME']}{unit}, E_alt={paper_metrics.get('E_alt', 'n/a')}, E_alt_gt={paper_metrics.get('E_alt_gt', 'n/a')}{unit}, Recall@1={paper_metrics['selected_anchor_hit_at_threshold']}% (<= {retrieval_threshold_m:.0f}m)")
        
        total_time = time.time() - start_t
        logger.info(f"Dataset {dataset_name} complete in {total_time:.2f}s")
        return {
            'dataset': dataset_name,
            'pipeline': self._pipeline_name,
            'metrics': metrics.to_dict(),
            'proposal_metrics': proposal_metrics.to_dict(),
            'paper_metrics': paper_metrics,
            'run_manifest': self._build_run_manifest_payload(dataset_name),
            'predictions': results,
            'timing': {
                'total_time_s': total_time,
                'avg_time_ms': (total_time / len(results) * 1000) if results else 0
            }
        }
    
    def run_all(self, datasets: List[str] = None) -> Dict[str, Any]:
        """
        Run benchmark on multiple datasets.
        
        Args:
            datasets: List of dataset names (None = all configured)
            
        Returns:
            Combined results dict
        """
        if datasets is None:
            datasets = list(DATASET_CONFIGS.keys())
        
        all_results = {}
        for ds in datasets:
            if ds not in DATASET_CONFIGS:
                logger.warning(f"Unknown dataset: {ds}, skipping")
                continue
            
            try:
                result = self.run_dataset(ds)
                all_results[ds] = result
            except Exception as e:
                logger.error(f"Dataset {ds} failed: {e}")
                all_results[ds] = {'error': str(e)}
        
        # Save combined results
        with open(self._output_dir / 'combined_results.json', 'w') as f:
            json.dump(all_results, f, indent=2, cls=NpEncoder)
        
        return all_results


def build_ablation_paper_rows_for_dataset(
    all_results: Dict[str, Any],
    ds: str,
    pipes: List[str],
    *,
    official_only: bool = True,
) -> List[Dict[str, Any]]:
    """Build paper-table rows from ablation results with official-row filtering."""
    rows_out: List[Dict[str, Any]] = []
    bucket = all_results.get(ds, {})
    for pipe in pipes:
        result = bucket.get(pipe, {})
        if 'error' in result:
            continue
        pm = result.get('paper_metrics', {})
        manifest = result.get('run_manifest', {})
        if official_only and not is_official_metrics_payload(manifest, pm):
            continue
        rows_out.append({
            "method": PIPELINE_CONFIGS[pipe]['name'],
            "M-ME": _ablation_table_float(pm.get("M-ME", float("nan"))),
            "HR": _ablation_table_float(pm.get("HR", float("nan"))),
            "E-ME": _ablation_table_float(pm.get("E-ME", float("nan"))),
            "E_alt": _ablation_table_float(pm.get("E_alt", float("nan"))),
            "Recall@1": _ablation_table_float(pm.get("Recall@1", float("nan"))),
            "A@10m": _ablation_table_float(pm.get("A@10m", pm.get("accuracy_at_10m", float("nan")))),
            "Inliers": _ablation_table_float(pm.get("mean_inliers", float("nan"))),
            "ShapeConf": _ablation_table_float(pm.get("mean_shape_confidence", float("nan"))),
        })
    return rows_out


def run_ablation(output_dir: Path,
                  max_samples: int = None,
                  max_samples_percent: int = None,
                  device: str = 'cpu',
                  eval_mode: str = 'end_to_end',
                  strict_components: bool = False,
                  strict_end2end: bool = False,
                  include_oracle_baselines: bool = True,
                  skip_module_dropouts: bool = False,
                  dataset: str = 'all',
                  uav_visloc_protocol: str = "intra_flight"):
    """
    Run ablation study across pipeline variants (proposed + E2E baselines).

    When ``dataset=='all'``, runs every dataset in ``DATASET_CONFIGS``. Run outputs
    use the flat matrix layout ``<output>/<dataset>/<pipeline>/<run_id>/`` so
    ``build_comparison_artifacts`` can aggregate rows.

    Oracle diagnostic baselines (``roma_every_frame``, ``oracle_fast_lightglue``)
    run in a second pass with ``eval_mode='oracle_reference'`` and
    ``strict_end2end=False`` unless ``include_oracle_baselines`` is False.

    Args:
        output_dir: Matrix root for all runs
        max_samples: Max samples per run (None = all)
        max_samples_percent: Percentage of dataset to run
        device: 'cpu' or 'cuda'
        eval_mode: Evaluation contract for the primary (E2E) pipeline block
        strict_components: Fail if components degrade to forbidden fallbacks
        strict_end2end: For E2E block only; oracle pass disables this flag
        include_oracle_baselines: Also run oracle-reference baselines
        skip_module_dropouts: If True, run only ``ABLATION_PIPELINES_E2E_CORE`` (faster;
            omits ours_no_ipm / ours_no_fgo / ours_no_lgi / ours_minimal).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if dataset == 'all':
        datasets = list(DATASET_CONFIGS.keys())
    else:
        datasets = [dataset]

    e2e_pipelines = ABLATION_PIPELINES_E2E_CORE if skip_module_dropouts else ABLATION_PIPELINES_E2E
    e2e_table_title = (
        "Paper Ablation Table (E2E core — proposal-aligned)"
        if skip_module_dropouts
        else "Paper Ablation Table (E2E + module dropouts)"
    )

    # Nested: ds -> pipe -> result; single-dataset JSON stays flat for backward compatibility.
    all_results: Dict[str, Any] = {}

    def _cleanup_torch_runtime() -> None:
        """Best-effort CUDA cleanup between sequential ablation runs."""
        import gc

        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                try:
                    torch.cuda.ipc_collect()
                except Exception:
                    pass
        except Exception:
            pass
    for ds in datasets:
        if ds not in DATASET_CONFIGS:
            logger.warning("Unknown dataset %s, skipping", ds)
            continue
        all_results[ds] = {}

        for pipe in e2e_pipelines:
            logger.info("\n%s", "#" * 60)
            logger.info("ABLATION E2E [%s]: %s", ds, PIPELINE_CONFIGS[pipe]['name'])
            logger.info("%s", "#" * 60)

            runner = BenchmarkRunner(
                pipeline_name=pipe,
                output_dir=output_dir,
                max_samples=max_samples,
                max_samples_percent=max_samples_percent,
                device=device,
                eval_mode=eval_mode,
                strict_components=strict_components,
                strict_end2end=strict_end2end,
                uav_visloc_protocol=uav_visloc_protocol,
            )
            try:
                all_results[ds][pipe] = runner.run_dataset(ds)
            except Exception as e:
                logger.error("Ablation %s / %s failed: %s", ds, pipe, e, exc_info=True)
                all_results[ds][pipe] = {'error': str(e)}
            finally:
                _cleanup_torch_runtime()

        if include_oracle_baselines:
            for pipe in ABLATION_PIPELINES_ORACLE:
                logger.info("\n%s", "#" * 60)
                logger.info("ABLATION ORACLE [%s]: %s", ds, PIPELINE_CONFIGS[pipe]['name'])
                logger.info("%s", "#" * 60)
                runner = BenchmarkRunner(
                    pipeline_name=pipe,
                    output_dir=output_dir,
                    max_samples=max_samples,
                    max_samples_percent=max_samples_percent,
                    device=device,
                    eval_mode='oracle_reference',
                    strict_components=strict_components,
                    strict_end2end=False,
                    uav_visloc_protocol=uav_visloc_protocol,
                )
                try:
                    all_results[ds][pipe] = runner.run_dataset(ds)
                except Exception as e:
                    logger.error("Ablation oracle %s / %s failed: %s", ds, pipe, e, exc_info=True)
                    all_results[ds][pipe] = {'error': str(e)}
                finally:
                    _cleanup_torch_runtime()

    from evaluation.metrics import generate_latex_table, LocalizationMetrics

    def _paper_rows_for_dataset(ds: str, pipes: List[str], *, official_only: bool = True) -> List[Dict[str, Any]]:
        return build_ablation_paper_rows_for_dataset(
            all_results,
            ds,
            pipes,
            official_only=official_only,
        )

    def _write_paper_table(md_path: Path, tex_path: Path, ds: str, paper_rows: List[Dict[str, Any]], title: str) -> None:
        if not paper_rows:
            return
        with open(md_path, "w") as f:
            f.write(f"# {title}\n\n")
            f.write(f"Dataset: `{ds}`\n\n")
            f.write("| Method | M-ME (m) | HR | E-ME (m) | E_alt (m) | Recall@1 | A@10m | Mean Inliers | Mean ShapeConf |\n")
            f.write("|---|---:|---:|---:|---:|---:|---:|---:|---:|\n")
            for row in paper_rows:
                f.write(
                    f"| {row['method']} | {row['M-ME']:.2f} | {row['HR']:.3f} | "
                    f"{row['E-ME']:.2f} | {row['E_alt']:.2f} | {row['Recall@1']:.3f} | "
                    f"{row['A@10m']:.3f} | {row['Inliers']:.2f} | {row['ShapeConf']:.3f} |\n"
                )
        with open(tex_path, "w") as f:
            f.write("% Auto-generated paper ablation table\n")
            f.write("\\begin{tabular}{lrrrrrrrr}\n")
            f.write("\\toprule\n")
            f.write("Method & M-ME (m) & HR & E-ME (m) & E_{alt} (m) & Recall@1 & A@10m & Mean Inliers & Mean ShapeConf \\\\\n")
            f.write("\\midrule\n")
            for row in paper_rows:
                method = row["method"].replace("_", "\\_")
                f.write(
                    f"{method} & {row['M-ME']:.2f} & {row['HR']:.3f} | "
                    f"{row['E-ME']:.2f} & {row['E_alt']:.2f} & {row['Recall@1']:.3f} | "
                    f"{row['A@10m']:.3f} & {row['Inliers']:.2f} & {row['ShapeConf']:.3f} \\\\\n"
                )
            f.write("\\bottomrule\n")
            f.write("\\end{tabular}\n")

    for ds in datasets:
        if ds not in all_results:
            continue
        metrics_dict = {}
        for pipe, result in all_results[ds].items():
            if 'error' not in result:
                m = result['metrics']
                metrics_dict[PIPELINE_CONFIGS[pipe]['name']] = LocalizationMetrics(**{
                    k: v for k, v in m.items()
                    if not isinstance(v, (list, np.ndarray))
                })
        if not metrics_dict:
            continue

        suffix = f"_{ds}" if len(datasets) > 1 else ""
        generate_latex_table(metrics_dict, output_path=output_dir / f'ablation_table{suffix}.tex')
        with open(output_dir / f'ablation_table{suffix}.md', 'w') as f:
            f.write("# Ablation Study Results\n\n")
            f.write(f"Dataset: {ds}\n\n")
            methods = list(metrics_dict.keys())
            f.write("| Metric | " + " | ".join(methods) + " |\n")
            f.write("|" + "---|" * (len(methods) + 1) + "\n")
            for label, field in [
                ('A@5m', 'accuracy_at_5m'),
                ('A@10m', 'accuracy_at_10m'),
                ('Mean Error', 'mean_error_m'),
                ('Median Error', 'median_error_m'),
                ('RMSE', 'rmse_m'),
                ('Heading MAE', 'heading_mae_deg'),
                ('Success %', 'success_rate'),
                ('FPS', 'fps'),
            ]:
                values = [f"{getattr(metrics_dict[m], field, 0):.2f}" for m in methods]
                f.write(f"| {label} | " + " | ".join(values) + " |\n")

        e2e_rows = _paper_rows_for_dataset(ds, e2e_pipelines, official_only=True)
        if len(datasets) == 1:
            e2e_md = output_dir / "ablation_paper_table.md"
            e2e_tex = output_dir / "ablation_paper_table.tex"
        else:
            e2e_md = output_dir / f"ablation_paper_table_e2e{suffix}.md"
            e2e_tex = output_dir / f"ablation_paper_table_e2e{suffix}.tex"
        _write_paper_table(
            e2e_md,
            e2e_tex,
            ds,
            e2e_rows,
            e2e_table_title,
        )
        if include_oracle_baselines:
            oracle_rows = _paper_rows_for_dataset(ds, ABLATION_PIPELINES_ORACLE, official_only=False)
            _write_paper_table(
                output_dir / f"ablation_paper_table_oracle{suffix}.md",
                output_dir / f"ablation_paper_table_oracle{suffix}.tex",
                ds,
                oracle_rows,
                "Paper Ablation Table (Oracle baselines)",
            )

    serializable = all_results[datasets[0]] if len(datasets) == 1 else all_results
    with open(output_dir / 'ablation_results.json', 'w') as f:
        json.dump(serializable, f, indent=2, cls=NpEncoder)

    logger.info("\nAblation complete. Results in %s", output_dir)
    if len(datasets) == 1:
        return all_results[datasets[0]]
    return all_results


def compare_results(result_dirs: List[Path], output_dir: Path):
    """Compare results from multiple experiments."""
    from evaluation.metrics import (
        generate_cdf_plot, generate_latex_table, LocalizationMetrics
    )
    
    all_metrics = {}
    all_errors = []
    labels = []
    
    for rdir in result_dirs:
        rdir = Path(rdir)
        result_file = rdir / 'results.json'
        if not result_file.exists():
            logger.warning(f"No results.json in {rdir}")
            continue
        
        with open(result_file, 'r') as f:
            data = json.load(f)
        
        method_name = rdir.name
        m = data['metrics']
        
        # Reconstruct metrics
        metrics = LocalizationMetrics(**{
            k: v for k, v in m.items()
            if not isinstance(v, (list, np.ndarray))
        })
        all_metrics[method_name] = metrics
        
        # Collect errors for CDF
        if 'predictions' in data:
            errors = np.array([
                p['position_error_2d_m']
                for p in data['predictions']
                if p.get('success', True) and np.isfinite(p.get('position_error_2d_m', float('inf')))
            ])
            all_errors.append(errors)
            labels.append(method_name)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate comparison CDF
    if all_errors:
        generate_cdf_plot(
            errors=all_errors,
            output_path=output_dir / 'comparison_cdf.png',
            labels=labels,
            title="Position Error CDF — Method Comparison",
            thresholds=[5, 10, 20, 50],
        )
    
    # Generate LaTeX table
    if all_metrics:
        generate_latex_table(all_metrics, output_dir / 'comparison_table.tex')
    
    logger.info(f"Comparison complete. Results in {output_dir}")


def build_comparison_artifacts(matrix_root: Path, eval_mode: str = "end_to_end") -> Dict[str, Any]:
    """
    Build comparison_metrics.{csv,json} and comparison_summary.md from detailed run outputs.
    """
    import csv
    matrix_root = Path(matrix_root)
    detailed_files = list(matrix_root.glob("**/results_detailed.json"))
    rows: List[Dict[str, Any]] = []
    tier_rows_for_export: List[Dict[str, Any]] = []
    reportable_non_strict_tiers = {
        "experimental_dsm_6dof",
        "native_reproduction",
        "native_oracle_tile",
        "native_oracle_pose_prior",
    }
    for fp in detailed_files:
        rel = fp.relative_to(matrix_root).parts
        # Layout A (matrix): <run_name>/<dataset>/<pipeline>/<run_id>/results_detailed.json  (5 parts)
        # Layout B (flat):   <dataset>/<pipeline>/<run_id>/results_detailed.json           (4 parts)
        if len(rel) == 5:
            run_name, dataset, pipeline, run_id = rel[0], rel[1], rel[2], rel[3]
            if str(run_name).lower() in {"smoke", "debug", "preflight"}:
                continue
            manifest_path = matrix_root / run_name / dataset / pipeline / run_id / "run_manifest.json"
        elif len(rel) == 4:
            dataset, pipeline, run_id = rel[0], rel[1], rel[2]
            run_name = "flat_matrix"
            manifest_path = fp.parent / "run_manifest.json"
        else:
            continue
        run_eval_mode = None
        run_strict_components = None
        run_strict_end2end = None
        manifest_payload: Dict[str, Any] = {}
        if manifest_path.exists():
            try:
                with open(manifest_path, "r") as mf:
                    manifest_payload = json.load(mf)
                    run_eval_mode = manifest_payload.get("eval_mode")
                    run_strict_components = bool(manifest_payload.get("strict_components", False))
                    run_strict_end2end = bool(manifest_payload.get("strict_end2end", False))
            except Exception:
                run_eval_mode = None
                run_strict_components = None
                run_strict_end2end = None
        try:
            registry_meta = resolve_pipeline_metadata(pipeline)
        except Exception:
            registry_meta = {}
        if registry_meta:
            for key, value in registry_meta.items():
                manifest_payload.setdefault(key, value)
        if run_eval_mode is not None and run_eval_mode != eval_mode:
            continue
        if run_eval_mode is None and len(rel) == 5 and eval_mode not in run_name:
            continue
        if run_strict_components is False:
            continue
        # Strict end-to-end flag applies only to end_to_end runs; oracle_reference
        # diagnostics intentionally omit strict_end2end.
        if run_eval_mode != "oracle_reference" and run_strict_end2end is False:
            continue
        with open(fp, "r") as f:
            payload = json.load(f)
        preds = payload.get("predictions", [])
        if not preds:
            continue
        pm_path = fp.parent / "paper_metrics_summary.json"
        paper_metrics: Dict[str, Any] = {}
        if pm_path.exists():
            try:
                with open(pm_path, "r") as pf:
                    paper_metrics = json.load(pf)
            except Exception:
                paper_metrics = {}
        strict_row_eligible = True
        if eval_mode == "end_to_end":
            strict_row_eligible = is_official_metrics_payload(manifest_payload, paper_metrics)
            tier_name = str(
                manifest_payload.get(
                    "comparison_tier",
                    paper_metrics.get("comparison_tier", registry_meta.get("comparison_tier", "")),
                )
            )
            reportable_non_strict = (
                tier_name in reportable_non_strict_tiers
                and not bool(manifest_payload.get("deprecated_alias", paper_metrics.get("deprecated_alias", False)))
                and not bool(manifest_payload.get("is_hybrid", paper_metrics.get("is_hybrid", False)))
                and not bool(manifest_payload.get("retrieval_only", paper_metrics.get("retrieval_only", False)))
                and manifest_payload.get("native_anyvisloc_schema_ok", paper_metrics.get("native_anyvisloc_schema_ok", True)) is not False
            )
            if not strict_row_eligible and not reportable_non_strict:
                continue
        n = len(preds)
        succ = [p for p in preds if p.get("accepted_for_metrics", p.get("success", False))]
        times = [float(p.get("total_time_ms", 0.0)) for p in preds if p.get("total_time_ms") is not None]
        fps = (1000.0 / (sum(times) / len(times))) if len(times) > 0 and sum(times) > 0 else 0.0
        errors = []
        for p in succ:
            try:
                errors.append(
                    float(
                        BenchmarkRunner._latlon_distance_m(
                            float(p.get("pred_lat", 0.0)),
                            float(p.get("pred_lon", 0.0)),
                            float(p.get("gt_lat", 0.0)),
                            float(p.get("gt_lon", 0.0)),
                        )
                    )
                )
            except Exception:
                pass
        if len(errors) == 0:
            mean_err = float("inf")
            median_err = float("inf")
            rmse = float("inf")
        else:
            err_arr = np.asarray(errors, dtype=np.float64)
            mean_err = float(np.mean(err_arr))
            median_err = float(np.median(err_arr))
            rmse = float(np.sqrt(np.mean(np.square(err_arr))))
        inliers = float(np.mean([float(p.get("num_inliers", 0.0)) for p in preds])) if preds else 0.0
        roma_rate = float(sum(1 for p in preds if "roma" in str(p.get("matcher_backend", "")))) / max(1, n)
        selected_hits = 0
        selected_total = 0
        raw_hits = 0
        raw_total = 0
        retrieval_threshold_m = DATASET_RETRIEVAL_THRESHOLD_M.get(dataset, 20.0)
        for p in preds:
            a = p.get("geodetic_anchor")
            if isinstance(a, dict) and "lat" in a and "lon" in a:
                selected_total += 1
                d = BenchmarkRunner._latlon_distance_m(
                    float(a.get("lat", 0.0)),
                    float(a.get("lon", 0.0)),
                    float(p.get("gt_lat", 0.0)),
                    float(p.get("gt_lon", 0.0)),
                )
                if d <= retrieval_threshold_m:
                    selected_hits += 1
            ra = p.get("raw_retrieval_anchor")
            if isinstance(ra, dict) and "lat" in ra and "lon" in ra:
                raw_total += 1
                dr = BenchmarkRunner._latlon_distance_m(
                    float(ra.get("lat", 0.0)),
                    float(ra.get("lon", 0.0)),
                    float(p.get("gt_lat", 0.0)),
                    float(p.get("gt_lon", 0.0)),
                )
                if dr <= retrieval_threshold_m:
                    raw_hits += 1
        def _pm_float(key: str, default: float = float("nan")) -> float:
            v = paper_metrics.get(key)
            if v is None:
                return default
            try:
                return float(v)
            except (TypeError, ValueError):
                return default

        def _pm_int(key: str, default: int = 0) -> int:
            v = paper_metrics.get(key)
            if v is None:
                return default
            try:
                return int(v)
            except (TypeError, ValueError):
                return default

        raw_recall_at_1_threshold = float(raw_hits) / max(1, raw_total)
        selected_hit_at_threshold = float(selected_hits) / max(1, selected_total)
        row: Dict[str, Any] = {
            "dataset": dataset,
            "pipeline": pipeline,
            "canonical_pipeline": str(
                manifest_payload.get(
                    "canonical_pipeline",
                    paper_metrics.get("canonical_pipeline", registry_meta.get("canonical_pipeline", pipeline)),
                )
            ),
            "comparison_tier": str(
                manifest_payload.get(
                    "comparison_tier",
                    paper_metrics.get("comparison_tier", registry_meta.get("comparison_tier", "strict_e2e")),
                )
            ),
            "method_family": str(
                manifest_payload.get(
                    "method_family",
                    paper_metrics.get("method_family", registry_meta.get("method_family", pipeline)),
                )
            ),
            "tile_source": str(
                manifest_payload.get(
                    "tile_source",
                    paper_metrics.get("tile_source", registry_meta.get("tile_source", "")),
                )
            ),
            "attitude_source": str(
                manifest_payload.get(
                    "attitude_source",
                    paper_metrics.get("attitude_source", registry_meta.get("attitude_source", "")),
                )
            ),
            "altitude_source": str(
                manifest_payload.get(
                    "altitude_source",
                    paper_metrics.get("altitude_source", registry_meta.get("altitude_source", "")),
                )
            ),
            "pose_head": str(
                manifest_payload.get(
                    "pose_head",
                    paper_metrics.get("pose_head", registry_meta.get("pose_head", "")),
                )
            ),
            "is_native": bool(
                manifest_payload.get("is_native", paper_metrics.get("is_native", registry_meta.get("is_native", False)))
            ),
            "is_wrapper": bool(
                manifest_payload.get(
                    "is_wrapper",
                    paper_metrics.get("is_wrapper", registry_meta.get("is_wrapper", False)),
                )
            ),
            "is_hybrid": bool(
                manifest_payload.get("is_hybrid", paper_metrics.get("is_hybrid", registry_meta.get("is_hybrid", False)))
            ),
            "uses_oracle": bool(
                manifest_payload.get(
                    "uses_oracle",
                    paper_metrics.get("uses_oracle", registry_meta.get("uses_oracle", False)),
                )
            ),
            "oracle_modes": manifest_payload.get(
                "oracle_modes",
                paper_metrics.get("oracle_modes", registry_meta.get("oracle_modes", [])),
            ),
            "eval_mode": eval_mode,
            "fps": fps,
            "mean_error_m": mean_err,
            "median_error_m": median_err,
            "rmse_m": rmse,
            "p90_error_m": _pm_float("p90_error_m"),
            "success_rate": float(len(succ)) / max(1, n),
            "recall_at_1_threshold_m": retrieval_threshold_m,
            "raw_retrieval_recall_at1_threshold": raw_recall_at_1_threshold,
            "selected_anchor_hit_at_threshold": selected_hit_at_threshold,
            "raw_retrieval_recall_at1_20m": raw_recall_at_1_threshold,
            "selected_anchor_hit_at20m": selected_hit_at_threshold,
            "mean_inliers": inliers,
            "roma_invocation_rate": roma_rate,
            "failed_frames": int(n - len(succ)),
            "run_name": run_name,
            "M_ME": _pm_float("M_ME"),
            "E_ME": _pm_float("E_ME"),
            "HR": _pm_float("HR"),
            "yaw_error_deg_mean": _pm_float("yaw_error_deg_mean"),
            "yaw_error_deg_median": _pm_float("yaw_error_deg_median"),
            "pitch_error_deg_mean": _pm_float("pitch_error_deg_mean"),
            "pitch_error_deg_median": _pm_float("pitch_error_deg_median"),
            "roll_error_deg_mean": _pm_float("roll_error_deg_mean"),
            "roll_error_deg_median": _pm_float("roll_error_deg_median"),
            "rotation_geodesic_error_deg_mean": _pm_float("rotation_geodesic_error_deg_mean"),
            "rotation_geodesic_error_deg_median": _pm_float("rotation_geodesic_error_deg_median"),
            "pnp_inlier_count_mean": _pm_float("pnp_inlier_count_mean"),
            "pnp_reprojection_error_mean": _pm_float("pnp_reprojection_error_mean"),
            "dsm_available": bool(paper_metrics.get("dsm_available", manifest_payload.get("dsm_available", False))),
            "dsm_status": str(paper_metrics.get("dsm_status", manifest_payload.get("dsm_status", ""))),
            "dsm_strict_fail_count": _pm_int("dsm_strict_fail_count"),
            "num_retrieval_fail": _pm_int("num_retrieval_fail"),
            "num_geometry_fail_after_good_retrieval": _pm_int("num_geometry_fail_after_good_retrieval"),
            "num_fusion_degradation": _pm_int("num_fusion_degradation"),
            "num_pose_rejected_by_final_gate": _pm_int("num_pose_rejected_by_final_gate"),
            "num_fgo_rejected_by_guard": _pm_int("num_fgo_rejected_by_guard"),
            "num_healthy": _pm_int("num_healthy"),
            "phase_retrieval_ms_mean": _pm_float("phase_retrieval_ms_mean"),
            "phase_matching_ms_mean": _pm_float("phase_matching_ms_mean"),
            "phase_geometry_ms_mean": _pm_float("phase_geometry_ms_mean"),
            "phase_fgo_ms_mean": _pm_float("phase_fgo_ms_mean"),
            "vpair_raw_recall_at1_index_window1": _pm_float("vpair_raw_recall_at1_index_window1"),
            "vpair_selected_hit_index_window1": _pm_float("vpair_selected_hit_index_window1"),
        }
        tier_rows_for_export.append(row)
        if strict_row_eligible:
            rows.append(row)
    rows = sorted(rows, key=lambda r: (r["dataset"], r["pipeline"]))
    tier_rows_for_export = sorted(tier_rows_for_export, key=lambda r: (r["comparison_tier"], r["dataset"], r["pipeline"]))
    out_json = matrix_root / "comparison_metrics.json"
    out_csv = matrix_root / "comparison_metrics.csv"
    out_md = matrix_root / "comparison_summary.md"
    with open(out_json, "w") as f:
        json.dump(rows, f, indent=2, cls=NpEncoder)
    csv_fields = [
        "dataset", "pipeline", "canonical_pipeline", "comparison_tier", "method_family",
        "tile_source", "attitude_source", "altitude_source", "pose_head",
        "is_native", "is_wrapper", "is_hybrid", "uses_oracle", "oracle_modes",
        "eval_mode", "fps", "mean_error_m", "median_error_m", "rmse_m",
        "p90_error_m",
        "success_rate", "recall_at_1_threshold_m", "raw_retrieval_recall_at1_threshold", "selected_anchor_hit_at_threshold",
        "raw_retrieval_recall_at1_20m", "selected_anchor_hit_at20m",
        "mean_inliers", "roma_invocation_rate", "failed_frames", "run_name",
        "M_ME", "E_ME", "HR",
        "yaw_error_deg_mean", "yaw_error_deg_median",
        "pitch_error_deg_mean", "pitch_error_deg_median",
        "roll_error_deg_mean", "roll_error_deg_median",
        "rotation_geodesic_error_deg_mean", "rotation_geodesic_error_deg_median",
        "pnp_inlier_count_mean", "pnp_reprojection_error_mean",
        "dsm_available", "dsm_status", "dsm_strict_fail_count",
        "num_retrieval_fail", "num_geometry_fail_after_good_retrieval", "num_fusion_degradation",
        "num_pose_rejected_by_final_gate", "num_fgo_rejected_by_guard", "num_healthy",
        "phase_retrieval_ms_mean", "phase_matching_ms_mean", "phase_geometry_ms_mean", "phase_fgo_ms_mean",
        "vpair_raw_recall_at1_index_window1", "vpair_selected_hit_index_window1",
    ]
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    tier_outputs: Dict[str, Dict[str, str]] = {}
    for tier in sorted({str(r.get("comparison_tier", "unknown")) for r in tier_rows_for_export}):
        safe_tier = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in tier)
        tier_rows = [r for r in tier_rows_for_export if str(r.get("comparison_tier", "unknown")) == tier]
        tier_json = matrix_root / f"comparison_{safe_tier}.json"
        tier_csv = matrix_root / f"comparison_{safe_tier}.csv"
        tier_md = matrix_root / f"comparison_{safe_tier}.md"
        with open(tier_json, "w") as f:
            json.dump(tier_rows, f, indent=2, cls=NpEncoder)
        with open(tier_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=csv_fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(tier_rows)
        with open(tier_md, "w") as f:
            f.write(f"# Comparison Tier: {tier}\n\n")
            f.write("| dataset | pipeline | canonical | tile_source | pose_head | mean_error_m | E_ME | FPS | oracle |\n")
            f.write("|---|---|---|---|---|---:|---:|---:|---|\n")
            for r in tier_rows:
                f.write(
                    f"| {r['dataset']} | {r['pipeline']} | {r['canonical_pipeline']} | "
                    f"{r['tile_source']} | {r['pose_head']} | {r['mean_error_m']:.3f} | "
                    f"{r['E_ME']:.3f} | {r['fps']:.3f} | {r['uses_oracle']} |\n"
                )
        tier_outputs[tier] = {
            "json": str(tier_json),
            "csv": str(tier_csv),
            "md": str(tier_md),
        }
    with open(out_md, "w") as f:
        f.write("# Comparison Summary (Strict End-to-End)\n\n")
        f.write("## Core metrics\n\n")
        f.write("| dataset | pipeline | fps | mean_error_m | median_error_m | rmse_m | success_rate | recall_threshold_m | raw_recall@1(thr) | selected_hit(thr) | mean_inliers | roma_rate | failed |\n")
        f.write("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for r in rows:
            f.write(
                f"| {r['dataset']} | {r['pipeline']} | {r['fps']:.3f} | {r['mean_error_m']:.3f} | "
                f"{r['median_error_m']:.3f} | {r['rmse_m']:.3f} | {r['success_rate']:.3f} | "
                f"{r['recall_at_1_threshold_m']:.1f} | {r['raw_retrieval_recall_at1_threshold']:.3f} | {r['selected_anchor_hit_at_threshold']:.3f} | "
                f"{r['mean_inliers']:.2f} | {r['roma_invocation_rate']:.3f} | "
                f"{r['failed_frames']} |\n"
            )
        f.write("\n## Paper metrics & bottleneck attribution\n\n")
        f.write("| dataset | pipeline | M_ME | E_ME | HR | retr_fail | geom_fail | fusion_deg | healthy | "
                "ms_ret | ms_match | ms_geom | ms_fgo | vpair_idx@1 | vpair_sel_w1 |\n")
        f.write("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for r in rows:
            f.write(
                f"| {r['dataset']} | {r['pipeline']} | "
                f"{r['M_ME']:.3f} | {r['E_ME']:.3f} | {r['HR']:.3f} | "
                f"{r['num_retrieval_fail']} | {r['num_geometry_fail_after_good_retrieval']} | "
                f"{r['num_fusion_degradation']} | {r['num_healthy']} | "
                f"{r['phase_retrieval_ms_mean']:.2f} | {r['phase_matching_ms_mean']:.2f} | "
                f"{r['phase_geometry_ms_mean']:.2f} | {r['phase_fgo_ms_mean']:.2f} | "
                f"{r['vpair_raw_recall_at1_index_window1']:.3f} | {r['vpair_selected_hit_index_window1']:.3f} |\n"
            )
    logger.info("Comparison artifacts written to %s", matrix_root)
    return {"rows": rows, "json": str(out_json), "csv": str(out_csv), "md": str(out_md), "tiers": tier_outputs}


# =============================================================================
# MAIN
# =============================================================================

def main():
    import cv2
    cv2.setRNGSeed(0)
    parser = argparse.ArgumentParser(
        description='UAV Geo-Localization Benchmark Runner',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python benchmark_runner.py --dataset vpair --pipeline ours --output results/vpair/
  python benchmark_runner.py --dataset all --pipeline ours --output results/
  python benchmark_runner.py --dataset vpair --max-samples 10 --output results/debug/
  python benchmark_runner.py --ablation --dataset all --output results/full_ablation/
  python benchmark_runner.py --ablation --dataset all --max-samples 50 --ablation-skip-module-dropouts --output results/ablation_core_n50/
  python benchmark_runner.py --ablation --dataset vpair --ablation-skip-oracle --output results/ablation_e2e_only/
  python benchmark_runner.py --compare results/vpair/ours results/vpair/baseline --output results/compare/
        """
    )
    
    parser.add_argument('--dataset', type=str, default='vpair',
                        choices=list(DATASET_CONFIGS.keys()) + ['all'],
                        help='Dataset to run on')
    parser.add_argument('--pipeline', type=str, default='ours',
                        choices=list(PIPELINE_CONFIGS.keys()),
                        help='Pipeline variant to use')
    parser.add_argument('--output', type=str, default='results',
                        help='Output directory')
    parser.add_argument('--max-samples', type=int, default=None,
                        help='Max samples per dataset (for quick testing)')
    parser.add_argument('--max-samples-percent', type=int, default=None,
                        help='Percentage of dataset to run')
    parser.add_argument('--test-interval', type=int, default=1,
                        help='Sample every N-th frame')
    parser.add_argument('--device', type=str, default='cpu',
                        choices=['cpu', 'cuda'],
                        help='Compute device')
    parser.add_argument('--eval-mode', type=str, default='end_to_end',
                        choices=['oracle_reference', 'end_to_end'],
                        help='Evaluation contract: oracle paired reference or end-to-end retrieval')
    parser.add_argument('--strict-components', action='store_true',
                        help='Fail run if retrieval/matcher silently degrades to fallback backends')
    parser.add_argument('--strict-end2end', action='store_true',
                        help='In end_to_end mode, fail if dataset-paired oracle fallback is used')
    parser.add_argument('--finetuned', action='store_true',
                        help='Use finetuned model checkpoints')
    parser.add_argument('--ablation', action='store_true',
                        help='Run full ablation (proposed + roma_every_frame_e2e; oracle baselines unless skipped)')
    parser.add_argument('--ablation-skip-oracle', action='store_true',
                        help='With --ablation: skip oracle_reference baselines (roma_every_frame, oracle_fast_lightglue)')
    parser.add_argument('--ablation-skip-module-dropouts', action='store_true',
                        help='With --ablation: skip module ablations (ours_no_ipm, ours_no_fgo, ours_no_lgi, ours_minimal); '
                             'run only proposal-aligned E2E core (ours, ours_fast_top1, ours_cascade, roma_every_frame_e2e)')
    parser.add_argument('--compare', nargs='+', type=str,
                        help='Compare result directories')
    parser.add_argument('--build-comparison-root', type=str, default=None,
                        help='Build consolidated comparison artifacts from matrix root directory')
    parser.add_argument('--verbose', action='store_true',
                        help='Verbose logging')
    parser.add_argument(
        '--config-json',
        type=str,
        default=None,
        help='JSON object merged into pipeline config (UAV sweep / debug)',
    )
    parser.add_argument(
        '--uav-visloc-protocol',
        type=str,
        default='intra_flight',
        choices=['intra_flight', 'cross_flight', 'all_maps'],
        help='UAV-VisLoc retrieval protocol for reference corpus construction',
    )
    parser.add_argument(
        '--retrieval-only',
        action='store_true',
        help='Run retrieval-only diagnostics and skip matcher/geometry/FGO (diagnostic, non-official).',
    )
    ipm_group = parser.add_mutually_exclusive_group()
    ipm_group.add_argument(
        '--use-ipm', dest='use_ipm', action='store_true', default=None,
        help='Force-enable IPM regardless of pipeline config',
    )
    ipm_group.add_argument(
        '--no-ipm', dest='use_ipm', action='store_false',
        help='Force-disable IPM regardless of pipeline config',
    )
    parser.add_argument(
        '--clear-cache',
        action='store_true',
        help='Delete ALL retrieval DB cache entries and exit',
    )

    args = parser.parse_args()
    
    # Setup logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        datefmt='%H:%M:%S',
    )
    
    # Mode: Clear cache
    if args.clear_cache:
        cache_dir = Path(os.environ.get("HD4_RETRIEVAL_DB_CACHE_DIR", ".cache/retrieval_db"))
        if cache_dir.exists():
            entry_count = sum(1 for _ in cache_dir.rglob("retrieval_db.npz"))
            shutil.rmtree(cache_dir)
            logger.info("Cleared %d cache entries from %s", entry_count, cache_dir)
        else:
            logger.info("No cache directory at %s — nothing to clear.", cache_dir)
        return

    output_dir = Path(args.output)

    # Mode: Compare
    if args.compare:
        compare_results([Path(d) for d in args.compare], output_dir)
        return
    if args.build_comparison_root:
        build_comparison_artifacts(Path(args.build_comparison_root), eval_mode=args.eval_mode)
        return
    
    # Mode: Ablation
    if args.ablation:
        run_ablation(
            output_dir=output_dir,
            max_samples=args.max_samples,
            max_samples_percent=args.max_samples_percent,
            device=args.device,
            eval_mode=args.eval_mode,
            strict_components=args.strict_components,
            strict_end2end=args.strict_end2end,
            include_oracle_baselines=not args.ablation_skip_oracle,
            skip_module_dropouts=args.ablation_skip_module_dropouts,
            dataset=args.dataset,
            uav_visloc_protocol=args.uav_visloc_protocol,
        )
        return
    
    # Mode: Single run
    pipeline_overrides = None
    if args.config_json:
        pipeline_overrides = json.loads(args.config_json)
        if not isinstance(pipeline_overrides, dict):
            raise ValueError("--config-json must be a JSON object")
    runner = BenchmarkRunner(
        pipeline_name=args.pipeline,
        output_dir=output_dir,
        max_samples=args.max_samples,
        max_samples_percent=args.max_samples_percent,
        test_interval=args.test_interval,
        device=args.device,
        eval_mode=args.eval_mode,
        strict_components=args.strict_components,
        strict_end2end=args.strict_end2end,
        use_finetuned=args.finetuned,
        pipeline_overrides=pipeline_overrides,
        uav_visloc_protocol=args.uav_visloc_protocol,
        retrieval_only=args.retrieval_only,
        verbose=args.verbose,
        use_ipm_override=args.use_ipm,
    )
    
    if args.dataset == 'all':
        runner.run_all()
    else:
        runner.run_dataset(args.dataset)


if __name__ == '__main__':
    main()
