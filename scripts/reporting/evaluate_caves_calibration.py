#!/usr/bin/env python3
"""Evaluate CAVES false-localization risk calibration from benchmark artifacts."""

from __future__ import annotations

import argparse
import csv
import datetime
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np

try:
    from sklearn.linear_model import LogisticRegression

    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

from evaluation.metrics import NpEncoder
from uav_geolocalization.benchmark.caves import (  # noqa: E402
    apply_caves_outcome_labels,
    apply_caves_v0,
    bootstrap_auroc_ci,
    bootstrap_metric_ci,
    describe_caves_v0_components,
    expected_calibration_error,
    fit_isotonic_calibrator,
    fit_temperature_calibrator,
    risk_coverage_curve,
    rows_to_labels_scores,
    score_row_for_feature_set,
    summarize_caves_metrics,
)

DEFAULT_FEATURE_SETS = [
    "random_baseline",
    "gate_only",
    "inlier_ratio_only",
    "retrieval_margin_only",
    "retrieval_margin_plus_rank",
    "geometry_only",
    "full_caves",
    "full_without_rank",
    "lr_baseline",
]
RISK_COVERAGE_FEATURE_SETS = ["full_caves", "gate_only", "geometry_only"]
SPLIT_IDENTITY_KEY_FIELDS = ["dataset", "pipeline", "image_name", "artifact_run_dir"]
ECE_SENSITIVITY_BINS = [8, 10, 12, 15]
LAST_POSE_CANDIDATE_FIELDS = [
    "pose_jump_m",
    "last_pose_distance_m",
    "distance_to_last_pose_m",
    "distance_from_previous_pose_m",
    "previous_pose_distance_m",
    "prev_pose_distance_m",
]

SPLIT_BALANCE_COLUMNS = [
    "dataset",
    "pipeline",
    "adversarial_condition",
    "adversarial_candidate_policy",
    "artifact_dataset",
    "artifact_pipeline",
]
DATASET_RETRIEVAL_THRESHOLDS_M = {
    "avl": 20.0,
    "vpair": 20.0,
    "uav_visloc": 150.0,
}

COVERAGE_BAND = (0.10, 0.30)
ACCEPTED_BAD_CAP = 0.20
# Pilot run: 430/1200 pre-CAVES unsafe-accept rate (AVL+VPAIR, n=100/cell).
BASELINE_UNSAFE_ACCEPT = 0.358
LOW_COVERAGE_FLOOR = 0.05
MAX_ACCEPTED_BAD_CALIB = 0.25
MIN_COVERAGE_CALIB = 0.70

LR_FEATURE_FIELDS = [
    "inlier_ratio",
    "num_inliers",
    "reproj_residual_px",
    "shape_confidence",
    "retrieval_score_margin",
    "retrieval_rank",
    "geometry_anchor_offset_m",
    "final_pose_gate_reject",
    "retrieval_protocol_violation",
]


def _extract_lr_feature_vector(row: Dict[str, Any]) -> List[float]:
    """Extract LR feature vector from a prediction row (NaN → 0 imputation)."""
    vec: List[float] = []
    for field in LR_FEATURE_FIELDS:
        raw = row.get(field)
        if field == "retrieval_rank" and (raw is None or raw == ""):
            raw = row.get("candidate_rank_selected")
        if field == "inlier_ratio" and (raw is None or raw == ""):
            nm = _safe_float(row.get("num_matches"))
            ni = _safe_float(row.get("num_inliers"))
            if nm > 0 and np.isfinite(nm) and np.isfinite(ni):
                raw = ni / nm
        if field in ("final_pose_gate_reject", "retrieval_protocol_violation"):
            vec.append(1.0 if _truthy(raw) else 0.0)
        else:
            val = float(raw) if raw is not None and raw != "" else float("nan")
            vec.append(0.0 if not np.isfinite(val) else val)
    return vec


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return False


def _safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        if value is None or value == "":
            return default
        out = float(value)
        return out if out == out and abs(out) != float("inf") else default
    except (TypeError, ValueError):
        return default


def _both_classes(labels: Sequence[int]) -> bool:
    vals = {int(label) for label in labels}
    return 0 in vals and 1 in vals


def _has_value(row: Dict[str, Any], field: str) -> bool:
    value = row.get(field)
    return value is not None and str(value).strip() != ""


def _is_forge_diagnostic_row(row: Dict[str, Any]) -> bool:
    """Return True for rows that carry explicit wrong-reference/FORGE provenance."""
    if _truthy(row.get("is_adversarial")) or _truthy(row.get("is_wrong_reference")):
        return True
    if _has_value(row, "adversarial_condition") or _has_value(row, "adversarial_candidate_policy"):
        return True
    path_blob = " ".join(
        str(row.get(field, "") or "").lower()
        for field in ("artifact_predictions_csv", "artifact_run_dir")
    )
    return "forge" in path_blob


def _row_unsafe_accept_with_semantics(row: Dict[str, Any]) -> Tuple[bool | None, str]:
    """Resolve pre-CAVES unsafe acceptance without using CAVES accept/reject decisions."""
    if _has_value(row, "unsafe_accept"):
        return _truthy(row.get("unsafe_accept")), "unsafe_accept"
    if not _has_value(row, "bad_pose_label"):
        return None, "missing_bad_pose_label"
    if _has_value(row, "accepted_for_metrics"):
        accepted = _truthy(row.get("accepted_for_metrics"))
        return bool(accepted and _truthy(row.get("bad_pose_label"))), "accepted_for_metrics_and_bad_pose_label"
    if _has_value(row, "success"):
        accepted = _truthy(row.get("success"))
        return bool(accepted and _truthy(row.get("bad_pose_label"))), "success_and_bad_pose_label"
    return None, "missing_acceptance_semantics"


def resolve_baseline_accepted_bad_rate(
    rows: Sequence[Dict[str, Any]],
    *,
    override: float | None = None,
) -> Dict[str, Any]:
    """Resolve FORGE-baseline ABR from pre-CAVES acceptance semantics."""
    if override is not None:
        value = float(override)
        if not (0.0 <= value <= 1.0):
            raise ValueError("--baseline-accepted-bad-rate must be in [0, 1]")
        return {
            "baseline_accepted_bad_rate": value,
            "baseline_source": "cli_override",
            "eligible_forge_rows": 0,
            "usable_acceptance_rows": 0,
            "unsafe_accept_rows": 0,
            "nan_fallback_bad_pose_rows": 0,
            "nan_fallback_bad_pose_rate": 0.0,
            "acceptance_semantics": "cli_override",
        }

    eligible = [row for row in rows if _is_forge_diagnostic_row(row)]
    usable_rows = 0
    unsafe_rows = 0
    nan_fallback_bad_pose_rows = 0
    semantics_used: Dict[str, int] = {}
    for row in eligible:
        unsafe_accept, semantic = _row_unsafe_accept_with_semantics(row)
        semantics_used[semantic] = semantics_used.get(semantic, 0) + 1
        if unsafe_accept is None:
            continue
        usable_rows += 1
        if unsafe_accept:
            unsafe_rows += 1
        has_finite_error = (
            _safe_float(row.get("post_fgo_error_m")) == _safe_float(row.get("post_fgo_error_m"))
            or _safe_float(row.get("position_error_m")) == _safe_float(row.get("position_error_m"))
        )
        if _truthy(row.get("bad_pose_label")) and not has_finite_error:
            nan_fallback_bad_pose_rows += 1

    if usable_rows == 0:
        raise ValueError(
            "No FORGE rows with usable acceptance semantics; pass --baseline-accepted-bad-rate "
            "or provide unsafe_accept/accepted_for_metrics/success fields."
        )

    return {
        "baseline_accepted_bad_rate": float(unsafe_rows) / float(usable_rows),
        "baseline_source": "computed_from_forge_rows",
        "eligible_forge_rows": len(eligible),
        "usable_acceptance_rows": usable_rows,
        "unsafe_accept_rows": unsafe_rows,
        "nan_fallback_bad_pose_rows": nan_fallback_bad_pose_rows,
        "nan_fallback_bad_pose_rate": float(nan_fallback_bad_pose_rows) / float(usable_rows),
        "acceptance_semantics": semantics_used,
    }


def _infer_dataset_from_path(path: Path) -> str:
    for part in reversed(path.parts):
        key = part.strip().lower()
        if key in DATASET_RETRIEVAL_THRESHOLDS_M:
            return key
    return ""


def _infer_pipeline_from_path(path: Path, dataset_name: str) -> str:
    parts = list(path.parts)
    if dataset_name and dataset_name in [p.lower() for p in parts]:
        for idx, part in enumerate(parts):
            if part.lower() == dataset_name and idx + 1 < len(parts):
                return parts[idx + 1]
    if len(parts) >= 3:
        return parts[-3]
    return ""


def _dataset_threshold(row: Dict[str, Any], *, default_threshold_m: float) -> float:
    dataset = str(row.get("dataset") or row.get("artifact_dataset") or "").strip().lower()
    return float(DATASET_RETRIEVAL_THRESHOLDS_M.get(dataset, default_threshold_m))


def discover_prediction_csvs(roots: Sequence[Path]) -> List[Path]:
    out: List[Path] = []
    for root in roots:
        if root.is_file() and root.name == "predictions.csv":
            out.append(root)
            continue
        if root.exists():
            out.extend(sorted(root.rglob("predictions.csv")))
    return sorted(set(out))


def normalize_caves_row(
    row: Dict[str, Any],
    *,
    default_threshold_m: float,
    reject_threshold: float,
) -> None:
    """Backfill CAVES score and outcome labels for old prediction artifacts."""
    if row.get("caves_risk_score") in (None, "") or row.get("caves_decision") in (None, ""):
        apply_caves_v0(row, reject_threshold=reject_threshold)

    rank = _safe_float(row.get("retrieval_rank"))
    if not rank == rank:
        candidate_rank = _safe_float(row.get("candidate_rank_selected"))
        if candidate_rank == candidate_rank and candidate_rank >= 0:
            row["retrieval_rank"] = float(candidate_rank + 1.0)

    loc_threshold = _safe_float(row.get("localization_success_threshold_m"), default=float("nan"))
    if not loc_threshold == loc_threshold:
        loc_threshold = _dataset_threshold(row, default_threshold_m=default_threshold_m)
    retrieval_threshold = _safe_float(row.get("retrieval_correct_threshold_m"), default=float("nan"))
    if not retrieval_threshold == retrieval_threshold:
        retrieval_threshold = _dataset_threshold(row, default_threshold_m=default_threshold_m)

    if row.get("localization_is_success") in (None, "") or row.get("unsafe_accept_severity_m") in (None, ""):
        apply_caves_outcome_labels(
            row,
            localization_success_threshold_m=loc_threshold,
            retrieval_correct_threshold_m=retrieval_threshold,
        )


def read_prediction_rows(
    paths: Sequence[Path],
    *,
    default_threshold_m: float = 20.0,
    reject_threshold: float = 0.5,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for path in paths:
        try:
            artifact_dataset = _infer_dataset_from_path(path)
            artifact_pipeline = _infer_pipeline_from_path(path, artifact_dataset)
            with path.open(newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    row = dict(row)
                    row["artifact_predictions_csv"] = str(path)
                    row.setdefault("artifact_run_dir", str(path.parent))
                    row.setdefault("artifact_dataset", artifact_dataset)
                    row.setdefault("artifact_pipeline", artifact_pipeline)
                    row.setdefault("dataset", artifact_dataset)
                    row.setdefault("pipeline", artifact_pipeline)
                    normalize_caves_row(
                        row,
                        default_threshold_m=default_threshold_m,
                        reject_threshold=reject_threshold,
                    )
                    rows.append(row)
        except Exception as exc:
            rows.append({
                "artifact_predictions_csv": str(path),
                "artifact_read_error": f"{type(exc).__name__}: {exc}",
            })
    return rows


def random_baseline_score(row: Dict[str, Any], *, seed: int) -> float:
    key = "|".join([
        str(seed),
        str(row.get("dataset", "")),
        str(row.get("pipeline", "")),
        str(row.get("image_name", "")),
        str(row.get("artifact_run_dir", "")),
        str(row.get("candidate_rank_selected", "")),
    ])
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()
    return int(digest[:12], 16) / float(0xFFFFFFFFFFFF)


def rows_to_report_labels_scores(
    rows: Sequence[Dict[str, Any]],
    *,
    feature_set: str,
    random_seed: int,
) -> Tuple[List[int], List[float]]:
    labels, scores, _ = rows_to_report_labels_scores_group_keys(
        rows,
        feature_set=feature_set,
        random_seed=random_seed,
    )
    return labels, scores


def rows_to_report_labels_scores_group_keys(
    rows: Sequence[Dict[str, Any]],
    *,
    feature_set: str,
    random_seed: int,
) -> Tuple[List[int], List[float], List[str]]:
    labels: List[int] = []
    scores: List[float] = []
    groups: List[str] = []
    for original_idx, row in enumerate(rows):
        if row.get("bad_pose_label") in (None, ""):
            continue
        if feature_set == "random_baseline":
            score = random_baseline_score(row, seed=random_seed)
        else:
            score = score_row_for_feature_set(row, feature_set=feature_set)
        if not (score == score and abs(score) != float("inf")):
            continue
        labels.append(1 if _truthy(row.get("bad_pose_label")) else 0)
        scores.append(float(score))
        groups.append(group_key(row, fallback_index=original_idx))
    return labels, scores, groups


def _has_finite_field(row: Dict[str, Any], field: str) -> bool:
    value = _safe_float(row.get(field))
    return value == value


def compute_anchor_provenance_audit_rows(rows: Sequence[Dict[str, Any]], *, split_name: str) -> List[Dict[str, Any]]:
    by_key: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = {}
    for row in rows:
        dataset = str(row.get("dataset") or row.get("artifact_dataset") or "unknown").strip() or "unknown"
        pipeline = str(row.get("pipeline") or row.get("artifact_pipeline") or "unknown").strip() or "unknown"
        condition = str(row.get("adversarial_condition") or "unknown").strip() or "unknown"
        by_key.setdefault((dataset, pipeline, condition), []).append(row)

    out: List[Dict[str, Any]] = []
    for dataset, pipeline, condition in sorted(by_key):
        group = by_key[(dataset, pipeline, condition)]
        n_rows = len(group)
        n_geometry = sum(1 for row in group if _has_finite_field(row, "geometry_anchor_offset_m"))
        n_adversarial = sum(1 for row in group if _has_finite_field(row, "adversarial_offset_m"))
        n_only_adversarial = sum(
            1 for row in group
            if not _has_finite_field(row, "geometry_anchor_offset_m") and _has_finite_field(row, "adversarial_offset_m")
        )
        n_both = sum(
            1 for row in group
            if _has_finite_field(row, "geometry_anchor_offset_m") and _has_finite_field(row, "adversarial_offset_m")
        )
        n_neither = sum(
            1 for row in group
            if not _has_finite_field(row, "geometry_anchor_offset_m") and not _has_finite_field(row, "adversarial_offset_m")
        )
        gate_status = "PASS"
        gate_reason = "geometry_anchor_available_or_component_omitted"
        if n_only_adversarial > 0:
            gate_status = "FAIL"
            gate_reason = "rows_have_only_adversarial_offset_metadata"
        out.append({
            "split": split_name,
            "dataset": dataset,
            "pipeline": pipeline,
            "condition": condition,
            "n_rows": n_rows,
            "n_geometry_anchor_present": n_geometry,
            "n_geometry_anchor_missing": n_rows - n_geometry,
            "geometry_anchor_missing_rate": (n_rows - n_geometry) / n_rows if n_rows else float("nan"),
            "n_adversarial_offset_present": n_adversarial,
            "n_rows_with_only_adversarial_offset": n_only_adversarial,
            "n_rows_with_both_geometry_and_adversarial": n_both,
            "n_rows_with_neither": n_neither,
            "gate_status": gate_status,
            "gate_reason": gate_reason,
        })
    return out


def compute_split_balance(
    valid_rows: Sequence[Dict[str, Any]],
    *,
    fraction_delta_threshold: float = 0.20,
) -> List[Dict[str, Any]]:
    present_columns = [
        col for col in SPLIT_BALANCE_COLUMNS
        if any(str(row.get(col, "")).strip() for row in valid_rows)
    ]
    out: List[Dict[str, Any]] = []
    for column in present_columns:
        values = sorted({
            str(row.get(column, "")).strip()
            for row in valid_rows
            if str(row.get(column, "")).strip()
        })
        for value in values:
            matching = [row for row in valid_rows if str(row.get(column, "")).strip() == value]
            total = len(matching)
            calib = sum(1 for row in matching if row.get("caves_split") == "calibration")
            heldout = sum(1 for row in matching if row.get("caves_split") == "heldout")
            calib_frac = calib / total if total > 0 else float("nan")
            heldout_frac = heldout / total if total > 0 else float("nan")
            max_abs_delta = abs(calib_frac - heldout_frac) if total > 0 else float("nan")
            warnings: List[str] = []
            if total < 5:
                warnings.append("low_support_total_lt_5")
            if calib == 0:
                warnings.append("missing_from_calibration")
            if heldout == 0:
                warnings.append("missing_from_heldout")
            if total > 0 and max_abs_delta > fraction_delta_threshold:
                warnings.append("split_fraction_delta_gt_0.20")
            out.append({
                "condition_column": column,
                "condition_value": value,
                "total_count": total,
                "calibration_count": calib,
                "heldout_count": heldout,
                "calibration_fraction": calib_frac,
                "heldout_fraction": heldout_frac,
                "max_abs_fraction_delta": max_abs_delta,
                "warning": ";".join(warnings),
            })
    return out


def compute_risk_coverage_rows(
    rows: Sequence[Dict[str, Any]],
    *,
    feature_sets: Sequence[str],
    split: str,
    condition: str,
    random_seed: int,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for feature_set in feature_sets:
        labels, scores = rows_to_report_labels_scores(rows, feature_set=feature_set, random_seed=random_seed)
        for curve_row in risk_coverage_curve(labels, scores):
            out.append({
                "feature_set": feature_set,
                "score_type": "raw",
                "split": split,
                "condition": condition,
                **curve_row,
            })
    return out


def select_operating_points(
    sweep_rows: Sequence[Dict[str, Any]],
    *,
    baseline_accepted_bad_rate: float = BASELINE_UNSAFE_ACCEPT,
) -> List[Dict[str, Any]]:
    finite = [
        row for row in sweep_rows
        if _safe_float(row.get("coverage")) == _safe_float(row.get("coverage"))
        and _safe_float(row.get("accepted_bad_rate")) == _safe_float(row.get("accepted_bad_rate"))
    ]
    out: List[Dict[str, Any]] = []
    zero_bad = [row for row in finite if _safe_float(row.get("accepted_bad_rate")) <= 0.0]
    if zero_bad:
        best = max(zero_bad, key=lambda row: _safe_float(row.get("coverage"), 0.0))
        out.append({"name": "safety_first", **best})
    cov_lo, cov_hi = COVERAGE_BAND
    medium = [
        row for row in finite
        if cov_lo <= _safe_float(row.get("coverage"), -1.0) <= cov_hi
        and _safe_float(row.get("accepted_bad_rate"), 1.0) < min(ACCEPTED_BAD_CAP, baseline_accepted_bad_rate)
    ]
    if medium:
        best = min(medium, key=lambda row: (_safe_float(row.get("accepted_bad_rate"), 1.0), -_safe_float(row.get("coverage"), 0.0)))
        out.append({"name": "medium_risk", **best})
    else:
        out.append({
            "name": "medium_risk_unavailable",
            "threshold": float("nan"),
            "coverage": float("nan"),
            "accepted_bad_rate": float("nan"),
            "safe_reject_rate": float("nan"),
            "false_reject_rate": float("nan"),
            "reason": f"No threshold found with coverage in [{cov_lo}, {cov_hi}] and accepted_bad_rate < min({ACCEPTED_BAD_CAP}, baseline).",
        })
    return out


def select_illustrative_curve_points(
    curve_rows: Sequence[Dict[str, Any]],
    *,
    baseline_accepted_bad_rate: float = BASELINE_UNSAFE_ACCEPT,
) -> List[Dict[str, Any]]:
    """Select four descriptive points on the held-out risk-coverage curve.

    These are empirical observations, not prespecified operating targets.
    """
    finite = [
        row for row in curve_rows
        if _safe_float(row.get("coverage")) == _safe_float(row.get("coverage"))
        and _safe_float(row.get("risk")) == _safe_float(row.get("risk"))
    ]
    if not finite:
        return []
    sorted_rows = sorted(finite, key=lambda r: _safe_float(r.get("coverage"), 0.0))
    out: List[Dict[str, Any]] = []

    conservative = [
        r for r in sorted_rows
        if _safe_float(r.get("coverage"), 0.0) >= LOW_COVERAGE_FLOOR
        and _safe_float(r.get("risk"), 1.0) <= baseline_accepted_bad_rate
    ]
    if conservative:
        out.append({"label": "conservative_low_coverage", **conservative[0]})

    cov_lo, cov_hi = COVERAGE_BAND
    medium_band = [
        r for r in sorted_rows
        if cov_lo <= _safe_float(r.get("coverage"), -1.0) <= cov_hi
    ]
    if medium_band:
        best = min(medium_band, key=lambda r: _safe_float(r.get("risk"), 1.0))
        out.append({"label": "closest_infeasible_medium", **best})

    baseline_match = min(
        sorted_rows,
        key=lambda r: abs(_safe_float(r.get("risk"), 1.0) - baseline_accepted_bad_rate),
    )
    out.append({"label": "baseline_unsafe_rate", **baseline_match})

    out.append({"label": "accept_all_endpoint", **sorted_rows[-1]})

    return out


def assess_last_pose_gate_feasibility(rows: Sequence[Dict[str, Any]], *, min_rows: int = 30) -> Dict[str, Any]:
    fields_detected = [
        field for field in LAST_POSE_CANDIDATE_FIELDS
        if any(_safe_float(row.get(field)) == _safe_float(row.get(field)) for row in rows)
    ]
    rows_with_required_fields = 0
    if fields_detected:
        rows_with_required_fields = sum(
            1 for row in rows
            if any(_safe_float(row.get(field)) == _safe_float(row.get(field)) for field in fields_detected)
        )
    if not fields_detected or rows_with_required_fields < min_rows:
        return {
            "status": "insufficient_data",
            "fields_detected": fields_detected,
            "rows_with_required_fields": rows_with_required_fields,
            "minimum_required_rows": min_rows,
            "decision": "do_not_implement_runtime_gate",
            "reason": "Existing artifacts do not expose standardized previous-pose/distance/order fields at sufficient support.",
        }
    return {
        "status": "supported_diagnostic_only",
        "fields_detected": fields_detected,
        "rows_with_required_fields": rows_with_required_fields,
        "minimum_required_rows": min_rows,
        "decision": "evaluate_as_diagnostic_ablation_only",
        "reason": "Previous-pose distance fields exist with sufficient support, but this remains diagnostic and complementary to single-frame FORGE/CAVES.",
    }


def compute_per_condition_risk_coverage_rows(
    rows: Sequence[Dict[str, Any]],
    *,
    feature_sets: Sequence[str],
    random_seed: int,
) -> List[Dict[str, Any]]:
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        condition = str(row.get("adversarial_condition") or "unknown").strip() or "unknown"
        groups.setdefault(condition, []).append(row)
    out: List[Dict[str, Any]] = []
    for condition in sorted(groups):
        out.extend(compute_risk_coverage_rows(
            groups[condition],
            feature_sets=feature_sets,
            split="heldout",
            condition=condition,
            random_seed=random_seed,
        ))
    return out


def compute_per_condition_metrics(
    rows: Sequence[Dict[str, Any]],
    *,
    feature_set: str,
    threshold: float,
    random_seed: int,
    score_type: str = "raw",
) -> List[Dict[str, Any]]:
    """Compute held-out selective-localization metrics per adversarial_condition."""
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        condition = str(row.get("adversarial_condition") or "unknown").strip() or "unknown"
        groups.setdefault(condition, []).append(row)

    out: List[Dict[str, Any]] = []
    for condition in sorted(groups):
        group_rows = groups[condition]
        labels, scores = rows_to_report_labels_scores(
            group_rows,
            feature_set=feature_set,
            random_seed=random_seed,
        )
        n_total = len(labels)
        if n_total == 0:
            continue
        n_bad = int(sum(labels))
        base_bad_rate = float(n_bad / n_total)
        metrics = summarize_caves_metrics(labels, scores, threshold=threshold)
        out.append({
            "feature_set": feature_set,
            "score_type": score_type,
            "condition": condition,
            "n_total": n_total,
            "n_bad": n_bad,
            "base_bad_rate": base_bad_rate,
            "coverage": metrics.get("coverage"),
            "accepted_bad_rate": metrics.get("accepted_bad_rate"),
            "safe_reject_rate": metrics.get("safe_reject_rate"),
            "false_reject_rate": metrics.get("false_reject_rate"),
        })
    return out


@dataclass(frozen=True)
class ThresholdSelection:
    threshold: float
    safety_gate_infeasible: bool
    sweep_rows: Tuple[Dict[str, Any], ...]


def _threshold_metrics_at(
    pairs: Sequence[tuple[int, float]],
    threshold: float,
) -> Dict[str, float]:
    accepted = [(label, score) for label, score in pairs if score < threshold]
    coverage = len(accepted) / len(pairs)
    if not accepted:
        return {
            "coverage": float(coverage),
            "accepted_bad_rate": float("nan"),
            "false_reject_rate": float("nan"),
            "safe_reject_rate": float("nan"),
        }
    accepted_bad_rate = sum(label for label, _ in accepted) / len(accepted)
    false_reject_good = sum(1 for label, score in pairs if label == 0 and score >= threshold)
    good_total = sum(1 for label, _ in pairs if label == 0)
    rejected_bad = sum(1 for label, score in pairs if label == 1 and score >= threshold)
    bad_total = sum(1 for label, _ in pairs if label == 1)
    return {
        "coverage": float(coverage),
        "accepted_bad_rate": float(accepted_bad_rate),
        "false_reject_rate": float(false_reject_good / max(1, good_total)),
        "safe_reject_rate": float(rejected_bad / bad_total) if bad_total > 0 else float("nan"),
    }


def _build_threshold_sweep(
    pairs: Sequence[tuple[int, float]],
    *,
    min_coverage: float,
    max_accepted_bad_rate: float | None,
) -> Tuple[List[Dict[str, Any]], List[tuple[float, float, float, float]]]:
    unique_scores = sorted({score for _, score in pairs})
    candidates = [0.0] + unique_scores + [1.0 + 1e-9]
    sweep_rows: List[Dict[str, Any]] = []
    feasible: List[tuple[float, float, float, float]] = []
    for threshold in candidates:
        metrics = _threshold_metrics_at(pairs, threshold)
        coverage = metrics["coverage"]
        accepted_bad_rate = metrics["accepted_bad_rate"]
        feasible_coverage_gate = coverage >= min_coverage
        if max_accepted_bad_rate is None:
            feasible_abr_gate = True
        elif not (accepted_bad_rate == accepted_bad_rate):
            feasible_abr_gate = False
        else:
            feasible_abr_gate = accepted_bad_rate <= max_accepted_bad_rate
        sweep_rows.append({
            "threshold": float(threshold),
            "coverage": coverage,
            "accepted_bad_rate": accepted_bad_rate,
            "false_reject_rate": metrics["false_reject_rate"],
            "safe_reject_rate": metrics["safe_reject_rate"],
            "feasible_coverage_gate": bool(feasible_coverage_gate),
            "feasible_abr_gate": bool(feasible_abr_gate),
        })
        if feasible_coverage_gate and feasible_abr_gate:
            feasible.append((
                accepted_bad_rate,
                metrics["false_reject_rate"],
                -coverage,
                float(threshold),
            ))
    return sweep_rows, feasible


def choose_threshold(
    labels: Sequence[int],
    scores: Sequence[float],
    *,
    min_coverage: float,
    default_threshold: float,
    max_accepted_bad_rate: float | None = None,
    _allow_fallback: bool = True,
) -> ThresholdSelection:
    pairs = [(int(label), float(score)) for label, score in zip(labels, scores) if score == score]
    if not pairs:
        return ThresholdSelection(float(default_threshold), False, ())

    sweep_rows, feasible = _build_threshold_sweep(
        pairs,
        min_coverage=min_coverage,
        max_accepted_bad_rate=max_accepted_bad_rate,
    )
    sweep_tuple = tuple(sweep_rows)

    if feasible:
        best = min(feasible)
        return ThresholdSelection(float(best[3]), False, sweep_tuple)

    if max_accepted_bad_rate is not None and _allow_fallback and min_coverage > 0.0:
        fallback = choose_threshold(
            labels,
            scores,
            min_coverage=0.0,
            default_threshold=default_threshold,
            max_accepted_bad_rate=max_accepted_bad_rate,
            _allow_fallback=False,
        )
        return ThresholdSelection(fallback.threshold, fallback.safety_gate_infeasible, sweep_tuple)

    if max_accepted_bad_rate is not None:
        return ThresholdSelection(float(default_threshold), True, sweep_tuple)

    return ThresholdSelection(float(default_threshold), False, sweep_tuple)


def split_identity_key(row: Dict[str, Any]) -> str:
    return "|".join(str(row.get(field, "")) for field in SPLIT_IDENTITY_KEY_FIELDS)


def split_name(row: Dict[str, Any], *, calibration_percent: int) -> str:
    key = split_identity_key(row)
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) % 100
    return "calibration" if bucket < calibration_percent else "heldout"


def group_key(row: Dict[str, Any], *, fallback_index: int) -> str:
    parts = [
        str(row.get("artifact_run_dir", "")),
        str(row.get("dataset", row.get("artifact_dataset", ""))),
        str(row.get("pipeline", row.get("artifact_pipeline", ""))),
        str(row.get("image_name", "")),
    ]
    key = "|".join(part for part in parts if part)
    return key or f"row:{fallback_index}"


def rows_to_group_keys(rows: Sequence[Dict[str, Any]], scores: Sequence[float]) -> List[str]:
    keys: List[str] = []
    for idx, (row, score) in enumerate(zip(rows, scores)):
        if score == score:
            keys.append(group_key(row, fallback_index=idx))
    return keys


def compute_split_identity_leakage_audit(
    calib_rows: Sequence[Dict[str, Any]],
    heldout_rows: Sequence[Dict[str, Any]],
    *,
    max_examples: int = 10,
) -> Dict[str, Any]:
    calibration_keys = {split_identity_key(row) for row in calib_rows}
    heldout_keys = {split_identity_key(row) for row in heldout_rows}
    overlap = sorted(calibration_keys.intersection(heldout_keys))
    return {
        "split_identity_key_fields": list(SPLIT_IDENTITY_KEY_FIELDS),
        "split_identity_overlap_count": len(overlap),
        "split_identity_overlap_examples": overlap[:max(0, int(max_examples))],
        "split_identity_unique_calibration": len(calibration_keys),
        "split_identity_unique_heldout": len(heldout_keys),
    }


def compute_ece_sensitivity_rows(
    calib_rows: Sequence[Dict[str, Any]],
    heldout_rows: Sequence[Dict[str, Any]],
    *,
    bins: Sequence[int] = ECE_SENSITIVITY_BINS,
    bootstrap_seed: int,
) -> List[Dict[str, Any]]:
    calib_labels, calib_scores = rows_to_report_labels_scores(
        calib_rows,
        feature_set="full_caves",
        random_seed=bootstrap_seed,
    )
    heldout_labels, heldout_scores = rows_to_report_labels_scores(
        heldout_rows,
        feature_set="full_caves",
        random_seed=bootstrap_seed,
    )
    out: List[Dict[str, Any]] = []
    calibration_status = "ok" if len(calib_labels) >= 4 and _both_classes(calib_labels) else "insufficient_calibration_data"
    isotonic_scores: List[float] | None = None
    if calibration_status == "ok":
        isotonic_calibrator = fit_isotonic_calibrator(calib_labels, calib_scores)
        if isotonic_calibrator.thresholds:
            isotonic_scores = isotonic_calibrator.predict(heldout_scores)
        else:
            calibration_status = "insufficient_calibration_data"
    for n_bins in bins:
        out.append({
            "feature_set": "full_caves",
            "split": "heldout",
            "score_type": "raw",
            "n_bins": int(n_bins),
            "ece": expected_calibration_error(heldout_labels, heldout_scores, n_bins=int(n_bins)),
            "bootstrap_seed": int(bootstrap_seed),
            "calibration_status": "not_applicable",
        })
        out.append({
            "feature_set": "full_caves",
            "split": "heldout",
            "score_type": "isotonic_calibrated",
            "n_bins": int(n_bins),
            "ece": expected_calibration_error(heldout_labels, isotonic_scores, n_bins=int(n_bins)) if isotonic_scores is not None else float("nan"),
            "bootstrap_seed": int(bootstrap_seed),
            "calibration_status": calibration_status,
        })
    return out


def compute_per_condition_grouped_ci_rows(
    rows: Sequence[Dict[str, Any]],
    *,
    bootstrap: int,
    bootstrap_seed: int,
    n_bins: int,
) -> List[Dict[str, Any]]:
    by_condition: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        condition = str(row.get("adversarial_condition") or "unknown").strip() or "unknown"
        by_condition.setdefault(condition, []).append(row)
    out: List[Dict[str, Any]] = []
    for condition in sorted(by_condition):
        condition_rows = by_condition[condition]
        labels, scores, groups = rows_to_report_labels_scores_group_keys(
            condition_rows,
            feature_set="full_caves",
            random_seed=bootstrap_seed,
        )
        ci = bootstrap_metric_ci(
            labels,
            scores,
            groups=groups,
            metric="auroc",
            n_bootstrap=bootstrap,
            seed=bootstrap_seed,
            n_bins=n_bins,
        )
        out.append({
            "name": "heldout_full_caves_raw_auroc_by_condition",
            "condition": condition,
            **ci,
            "n": len(labels),
            "n_bad": int(sum(labels)),
        })
    return out


def _metric_row(
    *,
    feature_set: str,
    split: str,
    score_type: str,
    labels: Sequence[int],
    scores: Sequence[float],
    threshold: float,
    n_bins: int,
    bootstrap: int,
    bootstrap_seed: int,
    calibration_status: str = "not_applicable",
    groups: Sequence[str] | None = None,
) -> Dict[str, Any]:
    metrics = summarize_caves_metrics(labels, scores, threshold=threshold, n_bins=n_bins)
    if groups is not None and len(groups) == len(labels):
        gci = bootstrap_metric_ci(
            labels, scores, groups=groups, metric="auroc",
            n_bootstrap=bootstrap, seed=bootstrap_seed, n_bins=n_bins,
        )
        ci = {
            "auroc_ci_low": gci.get("ci_low"),
            "auroc_ci_high": gci.get("ci_high"),
            "auroc_bootstrap_n": gci.get("bootstrap_n"),
            "auroc_bootstrap_valid": gci.get("bootstrap_valid"),
        }
    else:
        ci = bootstrap_auroc_ci(labels, scores, n_bootstrap=bootstrap, seed=bootstrap_seed)
    return {
        "feature_set": feature_set,
        "split": split,
        "score_type": score_type,
        "calibration_status": calibration_status,
        "n": metrics.get("num_samples", 0),
        "num_samples": metrics.get("num_samples", 0),
        "num_bad_pose": metrics.get("num_bad_pose", 0),
        "threshold": metrics.get("threshold", threshold),
        "coverage": metrics.get("coverage"),
        "accepted_bad_rate": metrics.get("accepted_bad_rate"),
        "false_rejection_rate": metrics.get("false_reject_rate"),
        "safe_reject_rate": metrics.get("safe_reject_rate"),
        "forge_true_reject_rate": metrics.get("forge_true_reject_rate"),
        "forge_false_reject_rate": metrics.get("forge_false_reject_rate"),
        "auroc": metrics.get("auroc"),
        "auroc_ci_low": ci.get("auroc_ci_low"),
        "auroc_ci_high": ci.get("auroc_ci_high"),
        "auroc_bootstrap_n": ci.get("auroc_bootstrap_n"),
        "auroc_bootstrap_valid": ci.get("auroc_bootstrap_valid"),
        "auroc_bootstrap_valid_flag": bool(int(ci.get("auroc_bootstrap_valid", 0) or 0) > 0),
        "auprc": metrics.get("auprc"),
        "brier": metrics.get("brier"),
        "ece": metrics.get("ece"),
    }


def _append_calibrated_rows(
    *,
    rows_out: List[Dict[str, Any]],
    feature_summary: Dict[str, Any],
    feature_set: str,
    raw_split_scores: Dict[str, Tuple[List[int], List[float]]],
    calib_labels: Sequence[int],
    calib_scores: Sequence[float],
    calibrated_scores_by_split: Dict[str, List[float]],
    score_type: str,
    threshold_key: str,
    calibration_status: str,
    default_threshold: float,
    min_coverage: float,
    max_accepted_bad_rate: float | None,
    n_bins: int,
    bootstrap: int,
    bootstrap_seed: int,
    groups_by_split: Dict[str, Sequence[str]] | None = None,
) -> None:
    if calibration_status != "ok":
        feature_summary[threshold_key] = float("nan")
        for split, (labels, _scores) in raw_split_scores.items():
            split_groups = (groups_by_split or {}).get(split)
            row = _metric_row(
                feature_set=feature_set,
                split=split,
                score_type=score_type,
                labels=labels,
                scores=[float("nan") for _ in labels],
                threshold=float("nan"),
                n_bins=n_bins,
                bootstrap=bootstrap,
                bootstrap_seed=bootstrap_seed,
                calibration_status=calibration_status,
                groups=split_groups,
            )
            rows_out.append(row)
            feature_summary["splits"][f"{split}_{score_type}"] = row
        return

    calibrated_calib_scores = calibrated_scores_by_split["calibration"]
    calibrated_selection = choose_threshold(
        list(calib_labels),
        calibrated_calib_scores,
        min_coverage=max(0.0, min(1.0, min_coverage)),
        default_threshold=default_threshold,
        max_accepted_bad_rate=max_accepted_bad_rate,
    )
    calibrated_threshold = calibrated_selection.threshold
    feature_summary[threshold_key] = calibrated_threshold
    feature_summary[f"{threshold_key}_safety_gate_infeasible"] = calibrated_selection.safety_gate_infeasible
    for split, (labels, _scores) in raw_split_scores.items():
        calibrated_scores = calibrated_scores_by_split[split]
        split_groups = (groups_by_split or {}).get(split)
        row = _metric_row(
            feature_set=feature_set,
            split=split,
            score_type=score_type,
            labels=labels,
            scores=calibrated_scores,
            threshold=calibrated_threshold,
            n_bins=n_bins,
            bootstrap=bootstrap,
            bootstrap_seed=bootstrap_seed,
            calibration_status=calibration_status,
            groups=split_groups,
        )
        rows_out.append(row)
        feature_summary["splits"][f"{split}_{score_type}"] = row


def _compute_lr_baseline(
    *,
    split_rows: Dict[str, Sequence[Dict[str, Any]]],
    default_threshold: float,
    min_coverage: float,
    max_accepted_bad_rate: float | None,
    n_bins: int,
    bootstrap: int,
    bootstrap_seed: int,
    random_seed: int,
    use_grouped_bootstrap: bool,
) -> Dict[str, Any]:
    """Train LogisticRegression on calibration, score held-out. Returns rows + summary."""
    rows_out: List[Dict[str, Any]] = []
    feature_summary: Dict[str, Any] = {"splits": {}, "lr_baseline_imputation": "zero"}

    if not HAS_SKLEARN:
        for split in ("all", "calibration", "heldout"):
            labels, _, groups = rows_to_report_labels_scores_group_keys(
                split_rows[split], feature_set="full_caves", random_seed=random_seed,
            )
            row = _metric_row(
                feature_set="lr_baseline", split=split, score_type="raw",
                labels=labels, scores=[float("nan")] * len(labels),
                threshold=float("nan"), n_bins=n_bins,
                bootstrap=bootstrap, bootstrap_seed=bootstrap_seed,
                calibration_status="sklearn_unavailable",
            )
            rows_out.append(row)
            feature_summary["splits"][f"{split}_raw"] = row
        return {"rows": rows_out, "summary": feature_summary}

    calib_rows_list = list(split_rows["calibration"])
    calib_labels_full: List[int] = []
    calib_features: List[List[float]] = []
    for row in calib_rows_list:
        if row.get("bad_pose_label") in (None, ""):
            continue
        calib_labels_full.append(1 if _truthy(row.get("bad_pose_label")) else 0)
        calib_features.append(_extract_lr_feature_vector(row))

    if len(calib_labels_full) < 4 or len(set(calib_labels_full)) < 2:
        for split in ("all", "calibration", "heldout"):
            labels, _, _ = rows_to_report_labels_scores_group_keys(
                split_rows[split], feature_set="full_caves", random_seed=random_seed,
            )
            row = _metric_row(
                feature_set="lr_baseline", split=split, score_type="raw",
                labels=labels, scores=[float("nan")] * len(labels),
                threshold=float("nan"), n_bins=n_bins,
                bootstrap=bootstrap, bootstrap_seed=bootstrap_seed,
                calibration_status="insufficient_calibration_data",
            )
            rows_out.append(row)
            feature_summary["splits"][f"{split}_raw"] = row
        return {"rows": rows_out, "summary": feature_summary}

    X_calib = np.array(calib_features, dtype=np.float64)
    y_calib = np.array(calib_labels_full, dtype=np.int32)
    lr = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs")
    lr.fit(X_calib, y_calib)

    for split in ("all", "calibration", "heldout"):
        split_labels: List[int] = []
        split_scores: List[float] = []
        split_groups: List[str] = []
        for idx, row in enumerate(split_rows[split]):
            if row.get("bad_pose_label") in (None, ""):
                continue
            split_labels.append(1 if _truthy(row.get("bad_pose_label")) else 0)
            feat = np.array([_extract_lr_feature_vector(row)], dtype=np.float64)
            prob = float(lr.predict_proba(feat)[0, 1])
            split_scores.append(prob)
            split_groups.append(group_key(row, fallback_index=idx))

        groups_arg = split_groups if (use_grouped_bootstrap and split == "heldout") else None
        threshold_for_lr = default_threshold
        if split == "calibration":
            sel = choose_threshold(
                split_labels, split_scores,
                min_coverage=max(0.0, min(1.0, min_coverage)),
                default_threshold=default_threshold,
                max_accepted_bad_rate=max_accepted_bad_rate,
            )
            threshold_for_lr = sel.threshold
            feature_summary["raw_threshold"] = threshold_for_lr
            feature_summary["safety_gate_infeasible"] = sel.safety_gate_infeasible

        row = _metric_row(
            feature_set="lr_baseline", split=split, score_type="raw",
            labels=split_labels, scores=split_scores,
            threshold=feature_summary.get("raw_threshold", default_threshold),
            n_bins=n_bins, bootstrap=bootstrap, bootstrap_seed=bootstrap_seed,
            groups=groups_arg,
        )
        rows_out.append(row)
        feature_summary["splits"][f"{split}_raw"] = row

    return {"rows": rows_out, "summary": feature_summary}


def compute_feature_set_metrics(
    *,
    valid_rows: Sequence[Dict[str, Any]],
    calib_rows: Sequence[Dict[str, Any]],
    heldout_rows: Sequence[Dict[str, Any]],
    feature_sets: Sequence[str],
    default_threshold: float,
    min_coverage: float,
    n_bins: int,
    bootstrap: int,
    bootstrap_seed: int,
    random_baseline_seed: int | None = None,
    max_accepted_bad_rate: float | None = None,
    use_grouped_bootstrap: bool = True,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rows_out: List[Dict[str, Any]] = []
    summary: Dict[str, Any] = {}
    random_seed = bootstrap_seed if random_baseline_seed is None else int(random_baseline_seed)
    split_rows = {
        "all": valid_rows,
        "calibration": calib_rows,
        "heldout": heldout_rows,
    }
    for feature_set in feature_sets:
        if feature_set == "lr_baseline":
            lr_summary = _compute_lr_baseline(
                split_rows=split_rows,
                default_threshold=default_threshold,
                min_coverage=min_coverage,
                max_accepted_bad_rate=max_accepted_bad_rate,
                n_bins=n_bins,
                bootstrap=bootstrap,
                bootstrap_seed=bootstrap_seed,
                random_seed=random_seed,
                use_grouped_bootstrap=use_grouped_bootstrap,
            )
            rows_out.extend(lr_summary["rows"])
            summary["lr_baseline"] = lr_summary["summary"]
            continue
        calib_labels, calib_scores = rows_to_report_labels_scores(
            calib_rows,
            feature_set=feature_set,
            random_seed=random_seed,
        )
        raw_selection = choose_threshold(
            calib_labels,
            calib_scores,
            min_coverage=max(0.0, min(1.0, min_coverage)),
            default_threshold=default_threshold,
            max_accepted_bad_rate=max_accepted_bad_rate,
        )
        threshold = raw_selection.threshold
        feature_summary: Dict[str, Any] = {
            "raw_threshold": threshold,
            "safety_gate_infeasible": raw_selection.safety_gate_infeasible,
            "splits": {},
        }
        raw_split_scores: Dict[str, Tuple[List[int], List[float]]] = {}
        groups_by_split: Dict[str, Sequence[str]] = {}
        for split, rows_for_split in split_rows.items():
            labels, scores, split_groups = rows_to_report_labels_scores_group_keys(
                rows_for_split,
                feature_set=feature_set,
                random_seed=random_seed,
            )
            raw_split_scores[split] = (labels, scores)
            if use_grouped_bootstrap and split == "heldout":
                groups_by_split[split] = split_groups
            row = _metric_row(
                feature_set=feature_set,
                split=split,
                score_type="raw",
                labels=labels,
                scores=scores,
                threshold=threshold,
                n_bins=n_bins,
                bootstrap=bootstrap,
                bootstrap_seed=bootstrap_seed,
                groups=groups_by_split.get(split),
            )
            rows_out.append(row)
            feature_summary["splits"][f"{split}_raw"] = row

        calibration_status = "ok"
        isotonic_calibrator = None
        temperature_calibrator = None
        if len(calib_labels) < 4 or not _both_classes(calib_labels):
            calibration_status = "insufficient_calibration_data"
        else:
            isotonic_calibrator = fit_isotonic_calibrator(calib_labels, calib_scores)
            temperature_calibrator = fit_temperature_calibrator(calib_labels, calib_scores)
            if not isotonic_calibrator.thresholds:
                calibration_status = "insufficient_calibration_data"
            if not (temperature_calibrator.temperature == temperature_calibrator.temperature):
                calibration_status = "insufficient_calibration_data"

        isotonic_by_split: Dict[str, List[float]] = {}
        temperature_by_split: Dict[str, List[float]] = {}
        if isotonic_calibrator is not None and temperature_calibrator is not None and calibration_status == "ok":
            for split, (_labels, scores) in raw_split_scores.items():
                isotonic_by_split[split] = isotonic_calibrator.predict(scores)
                temperature_by_split[split] = temperature_calibrator.predict(scores)

        _append_calibrated_rows(
            rows_out=rows_out,
            feature_summary=feature_summary,
            feature_set=feature_set,
            raw_split_scores=raw_split_scores,
            calib_labels=calib_labels,
            calib_scores=calib_scores,
            calibrated_scores_by_split=isotonic_by_split,
            score_type="isotonic_calibrated",
            threshold_key="isotonic_threshold",
            calibration_status=calibration_status,
            default_threshold=default_threshold,
            min_coverage=min_coverage,
            max_accepted_bad_rate=max_accepted_bad_rate,
            n_bins=n_bins,
            bootstrap=bootstrap,
            bootstrap_seed=bootstrap_seed,
            groups_by_split=groups_by_split,
        )
        _append_calibrated_rows(
            rows_out=rows_out,
            feature_summary=feature_summary,
            feature_set=feature_set,
            raw_split_scores=raw_split_scores,
            calib_labels=calib_labels,
            calib_scores=calib_scores,
            calibrated_scores_by_split=temperature_by_split,
            score_type="temperature_calibrated",
            threshold_key="temperature_threshold",
            calibration_status=calibration_status,
            default_threshold=default_threshold,
            min_coverage=min_coverage,
            max_accepted_bad_rate=max_accepted_bad_rate,
            n_bins=n_bins,
            bootstrap=bootstrap,
            bootstrap_seed=bootstrap_seed,
            groups_by_split=groups_by_split,
        )
        feature_summary["calibration_status"] = calibration_status
        if temperature_calibrator is not None:
            feature_summary["temperature"] = temperature_calibrator.temperature
        summary[feature_set] = feature_summary
    return rows_out, summary


def write_csv(path: Path, rows: Iterable[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _component_weight_csv_rows(components: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for component in components:
        row = dict(component)
        row["source_fields"] = json.dumps(list(component.get("source_fields", [])))
        row["label_like_fields_excluded"] = json.dumps(list(component.get("label_like_fields_excluded", [])))
        rows.append(row)
    return rows


def maybe_write_risk_coverage_plots(output: Path, rows: Sequence[Dict[str, Any]]) -> str:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        return "skipped_matplotlib_unavailable"
    by_feature: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        if row.get("condition") == "all":
            by_feature.setdefault(str(row.get("feature_set")), []).append(row)
    plt.figure(figsize=(7, 5))
    plotted = False
    for feature_set, feature_rows in sorted(by_feature.items()):
        xs = [_safe_float(row.get("coverage")) for row in feature_rows]
        ys = [_safe_float(row.get("risk")) for row in feature_rows]
        pairs = sorted((x, y) for x, y in zip(xs, ys) if x == x and y == y)
        if not pairs:
            continue
        px, py = zip(*pairs)
        plt.plot(px, py, label=feature_set)
        plotted = True
    if not plotted:
        plt.close()
        return "skipped_no_curve_rows"
    plt.xlabel("Coverage (accepted fraction)")
    plt.ylabel("Accepted bad-pose rate")
    plt.title("FORGE/CAVES risk–coverage trade-off")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output / "forge_caves_risk_coverage.png", dpi=160)
    plt.close()
    return "ok"


def maybe_write_per_condition_risk_coverage_plot(output: Path, rows: Sequence[Dict[str, Any]]) -> str:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        return "skipped_matplotlib_unavailable"
    full_rows = [row for row in rows if row.get("feature_set") == "full_caves"]
    by_condition: Dict[str, List[Dict[str, Any]]] = {}
    for row in full_rows:
        condition = str(row.get("condition") or "")
        if condition and condition != "all":
            by_condition.setdefault(condition, []).append(row)
    plt.figure(figsize=(7, 5))
    plotted = False
    for condition, condition_rows in sorted(by_condition.items()):
        pairs = sorted(
            (_safe_float(row.get("coverage")), _safe_float(row.get("risk")))
            for row in condition_rows
            if _safe_float(row.get("coverage")) == _safe_float(row.get("coverage"))
            and _safe_float(row.get("risk")) == _safe_float(row.get("risk"))
        )
        if not pairs:
            continue
        xs, ys = zip(*pairs)
        plt.plot(xs, ys, label=condition)
        plotted = True
    if not plotted:
        plt.close()
        return "skipped_no_condition_curve_rows"
    plt.xlabel("Coverage (accepted fraction)")
    plt.ylabel("Accepted bad-pose rate")
    plt.title("FORGE/CAVES per-condition risk–coverage")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output / "forge_caves_risk_coverage_by_condition.png", dpi=160)
    plt.close()
    return "ok"


def ablation_interpretation_notes(ablation_rows: Sequence[Dict[str, Any]]) -> List[str]:
    notes: List[str] = []
    for row in ablation_rows:
        if row.get("feature_set") != "inlier_ratio_only":
            continue
        if row.get("split") != "heldout" or row.get("score_type") != "raw":
            continue
        auroc = _safe_float(row.get("auroc"))
        if auroc == auroc and auroc < 0.5:
            notes.append(
                "`inlier_ratio_only` is directionally weak/inverted in this FORGE scope; "
                "wrong-reference failures can retain high inlier ratios after retrieval mismatch, "
                "so inlier ratio alone is not a safe risk proxy."
            )
            break
    return notes


def format_ablation_markdown(ablation_rows: Sequence[Dict[str, Any]]) -> str:
    heldout = [
        row for row in ablation_rows
        if row.get("split") == "heldout"
        and row.get("score_type") in {"raw", "isotonic_calibrated", "temperature_calibrated"}
    ]
    interpretation_notes = ablation_interpretation_notes(ablation_rows)
    lines = [
        "# CAVES Ablation Metrics",
        "",
        "Diagnostic note: CAVES ablations are false-localization risk diagnostics and must not be merged into clean official localization tables without a separately approved protocol.",
        "",
    ]
    if interpretation_notes:
        lines.extend(["Interpretation notes:", *[f"- {note}" for note in interpretation_notes], ""])
    lines.extend([
        "| Feature set | Score type | Status | n | Bad poses | AUROC | AUROC CI | AUPRC | Brier | ECE | Coverage | Accepted bad rate | False rejection rate | Safe reject rate | FORGE true reject | FORGE false reject |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in heldout:
        ci_low = row.get("auroc_ci_low")
        ci_high = row.get("auroc_ci_high")
        ci = f"{ci_low:.3f}–{ci_high:.3f}" if isinstance(ci_low, (int, float)) and isinstance(ci_high, (int, float)) and ci_low == ci_low and ci_high == ci_high else "n/a"
        lines.append(
            "| {feature_set} | {score_type} | {status} | {n} | {bad} | {auroc} | {ci} | {auprc} | {brier} | {ece} | {coverage} | {abr} | {frr} | {srr} | {ftr} | {ffr} |".format(
                feature_set=row.get("feature_set", ""),
                score_type=row.get("score_type", ""),
                status=row.get("calibration_status", ""),
                n=row.get("num_samples", 0),
                bad=row.get("num_bad_pose", 0),
                auroc=_fmt(row.get("auroc")),
                ci=ci,
                auprc=_fmt(row.get("auprc")),
                brier=_fmt(row.get("brier")),
                ece=_fmt(row.get("ece")),
                coverage=_fmt(row.get("coverage")),
                abr=_fmt(row.get("accepted_bad_rate")),
                frr=_fmt(row.get("false_rejection_rate")),
                srr=_fmt(row.get("safe_reject_rate")),
                ftr=_fmt(row.get("forge_true_reject_rate")),
                ffr=_fmt(row.get("forge_false_reject_rate")),
            )
        )
    lines.extend([
        "",
        "Feature-set definitions:",
        "- `random_baseline`: deterministic chance control (reporter layer only).",
        "- `gate_only`: existing final-pose gate reject signal only.",
        "- `inlier_ratio_only`: matching inlier ratio only.",
        "- `retrieval_margin_only`: retrieval score margin only.",
        "- `retrieval_margin_plus_rank`: retrieval margin plus explicit retrieval rank.",
        "- `geometry_only`: CAVES geometry/pose-gate diagnostics without retrieval margin/rank/protocol fields.",
        "- `full_caves`: complete v0 CAVES score.",
        "- `full_without_rank`: complete v0 CAVES score without retrieval rank.",
        "",
    ])
    return "\n".join(lines)


def _fmt(value: Any) -> str:
    val = _safe_float(value)
    return f"{val:.3f}" if val == val else "n/a"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    _ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    parser.add_argument("--roots", nargs="+", required=True, help="Result roots or predictions.csv files to scan")
    parser.add_argument(
        "--output",
        default=f"results/forge_caves_{_ts}",
        help="Output directory (default: results/forge_caves_YYYYMMDD_HHMMSS)",
    )
    parser.add_argument("--default-threshold", type=float, default=0.5)
    parser.add_argument("--default-localization-threshold-m", type=float, default=20.0)
    parser.add_argument("--min-coverage", type=float, default=0.70)
    parser.add_argument("--calibration-percent", type=int, default=50)
    parser.add_argument("--n-bins", type=int, default=10)
    parser.add_argument("--feature-sets", nargs="+", default=DEFAULT_FEATURE_SETS)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    parser.add_argument("--max-accepted-bad-rate", type=float, default=0.25)
    parser.add_argument(
        "--baseline-accepted-bad-rate",
        type=float,
        default=None,
        help="Optional FORGE-baseline ABR override; defaults to computing from FORGE unsafe_accept rows.",
    )
    args = parser.parse_args(argv)

    roots = [Path(p) for p in args.roots]
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    csv_paths = discover_prediction_csvs(roots)
    rows = read_prediction_rows(
        csv_paths,
        default_threshold_m=args.default_localization_threshold_m,
        reject_threshold=args.default_threshold,
    )
    valid_rows = [
        row for row in rows
        if not row.get("artifact_read_error")
        and row.get("bad_pose_label") not in (None, "")
    ]
    for row in valid_rows:
        row["caves_split"] = split_name(row, calibration_percent=max(1, min(99, args.calibration_percent)))

    calib_rows = [row for row in valid_rows if row.get("caves_split") == "calibration"]
    heldout_rows = [row for row in valid_rows if row.get("caves_split") == "heldout"]
    baseline_audit = resolve_baseline_accepted_bad_rate(
        valid_rows,
        override=args.baseline_accepted_bad_rate,
    )
    baseline_accepted_bad_rate = float(baseline_audit["baseline_accepted_bad_rate"])
    anchor_provenance_audit = (
        compute_anchor_provenance_audit_rows(calib_rows, split_name="calibration")
        + compute_anchor_provenance_audit_rows(heldout_rows, split_name="heldout")
    )
    anchor_provenance_gate_status = "PASS" if all(
        row.get("gate_status") == "PASS" for row in anchor_provenance_audit
    ) else "FAIL"
    split_identity_leakage_audit = compute_split_identity_leakage_audit(calib_rows, heldout_rows)
    ece_sensitivity_rows = compute_ece_sensitivity_rows(
        calib_rows,
        heldout_rows,
        bootstrap_seed=int(args.bootstrap_seed),
    )
    per_condition_grouped_ci = compute_per_condition_grouped_ci_rows(
        heldout_rows,
        bootstrap=max(0, int(args.bootstrap)),
        bootstrap_seed=int(args.bootstrap_seed),
        n_bins=int(args.n_bins),
    )
    calib_labels, calib_scores = rows_to_labels_scores(calib_rows, feature_set="full_caves")
    heldout_labels, heldout_scores, heldout_groups = rows_to_report_labels_scores_group_keys(
        heldout_rows,
        feature_set="full_caves",
        random_seed=int(args.bootstrap_seed),
    )
    all_labels, all_scores = rows_to_labels_scores(valid_rows, feature_set="full_caves")

    # Compute grouped-bootstrap AUROC CI for all ablation feature sets so the
    # ablation figure uses apples-to-apples group-level CIs throughout.
    # lr_baseline requires separate LR training; its grouped CI is excluded here.
    _grouped_ci_feature_sets = [fs for fs in DEFAULT_FEATURE_SETS if fs != "lr_baseline"]
    grouped_ci: Dict[str, Any] = {}
    for _fs in _grouped_ci_feature_sets:
        _labels, _scores, _groups = rows_to_report_labels_scores_group_keys(
            heldout_rows,
            feature_set=_fs,
            random_seed=int(args.bootstrap_seed),
        )
        grouped_ci[f"heldout_{_fs}_raw_auroc"] = bootstrap_metric_ci(
            _labels,
            _scores,
            groups=_groups,
            metric="auroc",
            n_bootstrap=args.bootstrap,
            seed=args.bootstrap_seed,
            n_bins=args.n_bins,
        )
    # ECE is reported only for full_caves (calibration-relevant metric).
    grouped_ci["heldout_full_caves_raw_ece"] = bootstrap_metric_ci(
        heldout_labels,
        heldout_scores,
        groups=heldout_groups,
        metric="ece",
        n_bootstrap=args.bootstrap,
        seed=args.bootstrap_seed,
        n_bins=args.n_bins,
    )

    threshold_selection = choose_threshold(
        calib_labels,
        calib_scores,
        min_coverage=max(0.0, min(1.0, args.min_coverage)),
        default_threshold=args.default_threshold,
        max_accepted_bad_rate=args.max_accepted_bad_rate,
    )
    threshold = threshold_selection.threshold
    operating_points = select_operating_points(
        threshold_selection.sweep_rows,
        baseline_accepted_bad_rate=baseline_accepted_bad_rate,
    )

    split_balance_rows = compute_split_balance(valid_rows)
    split_balance_warnings = sum(1 for row in split_balance_rows if row.get("warning"))

    ablation_rows, ablation_summary = compute_feature_set_metrics(
        valid_rows=valid_rows,
        calib_rows=calib_rows,
        heldout_rows=heldout_rows,
        feature_sets=args.feature_sets,
        default_threshold=args.default_threshold,
        min_coverage=args.min_coverage,
        max_accepted_bad_rate=args.max_accepted_bad_rate,
        n_bins=args.n_bins,
        bootstrap=max(0, int(args.bootstrap)),
        bootstrap_seed=int(args.bootstrap_seed),
    )
    interpretation_notes = ablation_interpretation_notes(ablation_rows)

    per_condition_metrics = compute_per_condition_metrics(
        heldout_rows,
        feature_set="full_caves",
        threshold=threshold,
        random_seed=int(args.bootstrap_seed),
        score_type="raw",
    )
    coverage_rows = risk_coverage_curve(heldout_labels, heldout_scores)
    curve_samples = select_illustrative_curve_points(
        coverage_rows,
        baseline_accepted_bad_rate=baseline_accepted_bad_rate,
    )
    risk_coverage_by_feature = compute_risk_coverage_rows(
        heldout_rows,
        feature_sets=RISK_COVERAGE_FEATURE_SETS,
        split="heldout",
        condition="all",
        random_seed=int(args.bootstrap_seed),
    )
    risk_coverage_by_condition = compute_per_condition_risk_coverage_rows(
        heldout_rows,
        feature_sets=RISK_COVERAGE_FEATURE_SETS,
        random_seed=int(args.bootstrap_seed),
    )
    plot_status = maybe_write_risk_coverage_plots(output, risk_coverage_by_feature)
    per_condition_plot_status = maybe_write_per_condition_risk_coverage_plot(output, risk_coverage_by_condition)
    component_weights = describe_caves_v0_components()

    summary = {
        "schema_version": 3,
        "official_metric_eligible": False,
        "no_universal_threshold": True,
        "production_safety_claim": False,
        "diagnostic_scope": "tier1_forge_caves_diagnostic",
        "risk_coverage_feature_sets": RISK_COVERAGE_FEATURE_SETS,
        "component_weights": component_weights,
        "curve_artifacts": [
            "caves_risk_coverage.csv",
            "caves_risk_coverage_by_feature.csv",
            "caves_risk_coverage_by_condition.csv",
            "caves_curve_samples.csv",
        ],
        "num_prediction_csvs": len(csv_paths),
        "num_rows_total": len(rows),
        "num_rows_valid": len(valid_rows),
        "calibration_percent": max(1, min(99, args.calibration_percent)),
        "selected_threshold": threshold,
        "threshold_policy_max_accepted_bad_rate": args.max_accepted_bad_rate,
        "baseline_accepted_bad_rate": baseline_accepted_bad_rate,
        "baseline_accepted_bad_rate_audit": baseline_audit,
        "safety_gate_infeasible": threshold_selection.safety_gate_infeasible,
        "min_coverage": args.min_coverage,
        "bootstrap": int(args.bootstrap),
        "bootstrap_seed": int(args.bootstrap_seed),
        "feature_sets": list(args.feature_sets),
        "metrics_all": summarize_caves_metrics(all_labels, all_scores, threshold=threshold, n_bins=args.n_bins),
        "metrics_calibration": summarize_caves_metrics(calib_labels, calib_scores, threshold=threshold, n_bins=args.n_bins),
        "metrics_heldout": summarize_caves_metrics(heldout_labels, heldout_scores, threshold=threshold, n_bins=args.n_bins),
        "auroc_ci_all": bootstrap_auroc_ci(all_labels, all_scores, n_bootstrap=args.bootstrap, seed=args.bootstrap_seed),
        "auroc_ci_heldout": bootstrap_auroc_ci(heldout_labels, heldout_scores, n_bootstrap=args.bootstrap, seed=args.bootstrap_seed),
        "ablation_metrics": ablation_rows,
        "ablation_summary": ablation_summary,
        "split_balance": split_balance_rows,
        "split_balance_warning_count": split_balance_warnings,
        "split_identity_leakage_audit": split_identity_leakage_audit,
        "anchor_provenance_audit": anchor_provenance_audit,
        "anchor_provenance_gate_status": anchor_provenance_gate_status,
        "ece_sensitivity": ece_sensitivity_rows,
        "per_condition_metrics": per_condition_metrics,
        "per_condition_grouped_ci": per_condition_grouped_ci,
        "grouping_policy": {
            "group_key_fields": ["artifact_run_dir", "dataset", "pipeline", "image_name"],
            "fallback": "row_index",
        },
        "grouped_ci": grouped_ci,
        "operating_points": operating_points,
        "curve_samples": curve_samples,
        "last_pose_gate_decision": assess_last_pose_gate_feasibility(valid_rows),
        "plot_status": plot_status,
        "per_condition_plot_status": per_condition_plot_status,
        "source_roots": [str(root) for root in roots],
    }

    (output / "caves_calibration_metrics.json").write_text(
        json.dumps(summary, indent=2, cls=NpEncoder),
        encoding="utf-8",
    )

    risk_coverage_fieldnames = [
        "feature_set", "score_type", "split", "condition",
        "coverage", "risk", "threshold", "accepted", "total",
    ]
    write_csv(
        output / "caves_risk_coverage.csv",
        coverage_rows,
        ["coverage", "risk", "threshold", "accepted", "total"],
    )
    write_csv(output / "caves_risk_coverage_by_feature.csv", risk_coverage_by_feature, risk_coverage_fieldnames)
    write_csv(output / "caves_risk_coverage_by_condition.csv", risk_coverage_by_condition, risk_coverage_fieldnames)
    write_csv(
        output / "caves_component_weights.csv",
        _component_weight_csv_rows(component_weights),
        ["name", "weight", "source_fields", "formula_note", "label_like_fields_excluded"],
    )
    write_csv(
        output / "caves_thresholds.csv",
        [{
            "split": "calibration",
            "feature_set": "full_caves",
            "score_type": "raw",
            "selected_threshold": threshold,
            "min_coverage": args.min_coverage,
            "max_accepted_bad_rate": args.max_accepted_bad_rate,
            "safety_gate_infeasible": threshold_selection.safety_gate_infeasible,
        }],
        [
            "split",
            "feature_set",
            "score_type",
            "selected_threshold",
            "min_coverage",
            "max_accepted_bad_rate",
            "safety_gate_infeasible",
        ],
    )
    write_csv(
        output / "caves_threshold_sweep.csv",
        threshold_selection.sweep_rows,
        [
            "threshold",
            "coverage",
            "accepted_bad_rate",
            "false_reject_rate",
            "safe_reject_rate",
            "feasible_coverage_gate",
            "feasible_abr_gate",
        ],
    )
    write_csv(
        output / "caves_operating_points.csv",
        operating_points,
        ["name", "threshold", "coverage", "accepted_bad_rate", "false_reject_rate", "safe_reject_rate", "feasible_coverage_gate", "feasible_abr_gate", "reason"],
    )
    write_csv(
        output / "caves_curve_samples.csv",
        curve_samples,
        ["label", "coverage", "risk", "threshold", "accepted", "total"],
    )
    split_balance_fieldnames = [
        "condition_column",
        "condition_value",
        "total_count",
        "calibration_count",
        "heldout_count",
        "calibration_fraction",
        "heldout_fraction",
        "max_abs_fraction_delta",
        "warning",
    ]
    write_csv(output / "caves_split_balance.csv", split_balance_rows, split_balance_fieldnames)
    write_csv(
        output / "caves_anchor_provenance_audit.csv",
        anchor_provenance_audit,
        [
            "split",
            "dataset",
            "pipeline",
            "condition",
            "n_rows",
            "n_geometry_anchor_present",
            "n_geometry_anchor_missing",
            "geometry_anchor_missing_rate",
            "n_adversarial_offset_present",
            "n_rows_with_only_adversarial_offset",
            "n_rows_with_both_geometry_and_adversarial",
            "n_rows_with_neither",
            "gate_status",
            "gate_reason",
        ],
    )
    write_csv(
        output / "caves_grouped_ci.csv",
        [{"name": name, **payload} for name, payload in grouped_ci.items()],
        ["name", "metric", "point", "ci_low", "ci_high", "bootstrap_n", "bootstrap_valid", "grouped", "num_groups"],
    )
    write_csv(
        output / "caves_ece_sensitivity.csv",
        ece_sensitivity_rows,
        ["feature_set", "split", "score_type", "n_bins", "ece", "bootstrap_seed", "calibration_status"],
    )
    write_csv(
        output / "caves_per_condition_grouped_ci.csv",
        per_condition_grouped_ci,
        ["name", "condition", "metric", "point", "ci_low", "ci_high", "bootstrap_n", "bootstrap_valid", "grouped", "num_groups", "n", "n_bad"],
    )
    per_condition_fieldnames = [
        "feature_set",
        "score_type",
        "condition",
        "n_total",
        "n_bad",
        "base_bad_rate",
        "coverage",
        "accepted_bad_rate",
        "safe_reject_rate",
        "false_reject_rate",
    ]
    write_csv(output / "caves_per_condition_metrics.csv", per_condition_metrics, per_condition_fieldnames)
    ablation_fieldnames = [
        "feature_set",
        "split",
        "score_type",
        "calibration_status",
        "n",
        "num_samples",
        "num_bad_pose",
        "threshold",
        "coverage",
        "accepted_bad_rate",
        "false_rejection_rate",
        "safe_reject_rate",
        "forge_true_reject_rate",
        "forge_false_reject_rate",
        "auroc",
        "auroc_ci_low",
        "auroc_ci_high",
        "auroc_bootstrap_n",
        "auroc_bootstrap_valid",
        "auroc_bootstrap_valid_flag",
        "auprc",
        "brier",
        "ece",
    ]
    write_csv(output / "caves_ablation_metrics.csv", ablation_rows, ablation_fieldnames)
    (output / "caves_ablation_metrics.md").write_text(format_ablation_markdown(ablation_rows), encoding="utf-8")

    report = [
        "# CAVES Calibration Report",
        "",
        f"- prediction CSVs: `{len(csv_paths)}`",
        f"- valid labeled rows: `{len(valid_rows)}`",
        f"- selected full-CAVES raw threshold: `{threshold:.4f}`",
        f"- minimum calibration coverage target: `{args.min_coverage:.2f}`",
        f"- maximum accepted bad rate target: `{args.max_accepted_bad_rate:.2f}`",
        f"- FORGE-baseline ABR: `{baseline_accepted_bad_rate:.4f}` "
        f"({baseline_audit.get('baseline_source')})",
        f"- FORGE baseline rows: `{baseline_audit.get('usable_acceptance_rows')}` usable / "
        f"`{baseline_audit.get('eligible_forge_rows')}` eligible; unsafe accepts: "
        f"`{baseline_audit.get('unsafe_accept_rows')}`",
        f"- NaN-fallback bad-pose rows in baseline audit: "
        f"`{baseline_audit.get('nan_fallback_bad_pose_rows')}` "
        f"({_fmt(baseline_audit.get('nan_fallback_bad_pose_rate'))})",
        f"- safety gate infeasible: `{threshold_selection.safety_gate_infeasible}`",
        f"- bootstrap resamples: `{int(args.bootstrap)}`",
        "",
        "## Held-out metrics (full CAVES raw)",
        "",
    ]
    for key, value in summary["metrics_heldout"].items():
        report.append(f"- `{key}`: `{value}`")
    report.extend([
        "",
        "## AUROC confidence interval (held-out full CAVES raw)",
        "",
    ])
    for key, value in summary["auroc_ci_heldout"].items():
        report.append(f"- `{key}`: `{value}`")
    report.extend([
        "",
        "## Per-condition breakdown",
        "",
        "Held-out `full_caves` raw metrics at the globally selected threshold.",
        "",
        "| condition | n_total | n_bad | base_bad_rate | coverage | accepted_bad_rate | safe_reject_rate | false_reject_rate |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for row in per_condition_metrics:
        if row.get("feature_set") != "full_caves" or row.get("score_type") != "raw":
            continue
        report.append(
            f"| {row.get('condition')} | {row.get('n_total')} | {row.get('n_bad')} | "
            f"{_fmt(row.get('base_bad_rate'))} | {_fmt(row.get('coverage'))} | "
            f"{_fmt(row.get('accepted_bad_rate'))} | {_fmt(row.get('safe_reject_rate'))} | "
            f"{_fmt(row.get('false_reject_rate'))} |"
        )
    report.extend([
        "",
        "## Split balance diagnostics",
        "",
        f"- warning count: `{split_balance_warnings}`",
        f"- anchor provenance gate: `{anchor_provenance_gate_status}`",
        "",
    ])
    for row in split_balance_rows:
        report.append(
            f"- `{row.get('condition_column')}`=`{row.get('condition_value')}`: "
            f"total={row.get('total_count')}, calib={row.get('calibration_count')}, "
            f"heldout={row.get('heldout_count')}, warning=`{row.get('warning', '')}`"
        )
    if interpretation_notes:
        report.extend([
            "",
            "## Ablation interpretation notes",
            "",
        ])
        report.extend(f"- {note}" for note in interpretation_notes)
    report.extend([
        "",
        "## Generated ablation artifacts",
        "",
        "- `caves_ablation_metrics.csv`",
        "- `caves_ablation_metrics.md`",
        "- `caves_risk_coverage_by_feature.csv`",
        "- `caves_risk_coverage_by_condition.csv`",
        "- `caves_split_balance.csv`",
        "- `caves_anchor_provenance_audit.csv`",
        "- `caves_grouped_ci.csv`",
        "- `caves_ece_sensitivity.csv`",
        "- `caves_per_condition_metrics.csv`",
        "- `caves_per_condition_grouped_ci.csv`",
        "- `caves_threshold_sweep.csv`",
        "- `caves_operating_points.csv`",
        "- `caves_curve_samples.csv`",
        "",
        "Diagnostic note: CAVES metrics are selective-localization diagnostics and must not be merged into clean official localization tables without a separately approved protocol.",
        "",
    ])
    (output / "caves_calibration_report.md").write_text("\n".join(report), encoding="utf-8")
    print(json.dumps({"output": str(output), "selected_threshold": threshold, "num_rows_valid": len(valid_rows)}, cls=NpEncoder))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
