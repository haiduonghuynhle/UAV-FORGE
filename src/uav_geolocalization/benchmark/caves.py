"""CAVES risk scoring and calibration utilities.

CAVES (Calibrated Visual Evidence Safety) is intentionally lightweight here:
- the online v0 scorer is deterministic and transparent;
- calibration metrics are computed offline from exported prediction rows;
- no official clean metric should be replaced by these diagnostics unless a paper
  protocol explicitly promotes them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

CAVES_ACCEPT = "accept"
CAVES_REJECT = "reject_false_localization"
CAVES_ABSTAIN = "abstain_insufficient_evidence"


@dataclass(frozen=True)
class CavesScore:
    risk_score: float
    confidence: float
    decision: str
    obs_gate_score: float
    obs_gate_reason: str
    evidence_count: int


@dataclass(frozen=True)
class CavesV0ComponentSpec:
    name: str
    weight: float
    source_fields: Tuple[str, ...]
    formula_note: str


CAVES_V0_LABEL_LIKE_FIELDS_EXCLUDED = (
    "post_fgo_error_m",
    "bad_pose_label",
    "unsafe_accept",
    "accepted_for_metrics",
    "localization_is_success",
)

CAVES_V0_FORBIDDEN_SCORE_INPUT_FIELDS = (
    *CAVES_V0_LABEL_LIKE_FIELDS_EXCLUDED,
    "adversarial_offset_m",
)


CAVES_V0_COMPONENT_SPECS = (
    CavesV0ComponentSpec(
        "low_inlier_ratio",
        1.5,
        ("inlier_ratio", "num_matches", "num_inliers"),
        "risk = 1 - clip(inlier_ratio, 0, 0.6) / 0.6; derive inlier_ratio from num_inliers / num_matches when missing",
    ),
    CavesV0ComponentSpec(
        "low_inlier_count",
        1.0,
        ("num_inliers",),
        "risk = 1 - clip(num_inliers, 0, 20) / 20",
    ),
    CavesV0ComponentSpec(
        "high_reprojection_residual",
        1.2,
        ("reproj_residual_px",),
        "risk = clip((reproj_residual_px - 3) / 17, 0, 1)",
    ),
    CavesV0ComponentSpec(
        "low_shape_confidence",
        1.0,
        ("shape_confidence",),
        "risk = 1 - clip(shape_confidence, 0, 1)",
    ),
    CavesV0ComponentSpec(
        "low_retrieval_margin",
        0.7,
        ("retrieval_score_margin",),
        "risk = 1 - clip(retrieval_score_margin, 0, 0.25) / 0.25",
    ),
    CavesV0ComponentSpec(
        "late_retrieval_rank",
        0.4,
        ("retrieval_rank", "candidate_rank_selected"),
        "risk = clip((max(1, rank) - 1) / 4, 0, 1)",
    ),
    CavesV0ComponentSpec(
        "large_anchor_offset",
        1.0,
        ("geometry_anchor_offset_m",),
        "risk = clip((geometry_anchor_offset_m - 20) / 130, 0, 1); omitted when geometry_anchor_offset_m is missing",
    ),
    CavesV0ComponentSpec(
        "final_pose_gate_reject",
        1.5,
        ("final_pose_gate_reject",),
        "risk = 1 when final_pose_gate_reject is truthy",
    ),
    CavesV0ComponentSpec(
        "retrieval_protocol_violation",
        2.0,
        ("retrieval_protocol_violation",),
        "risk = 1 when retrieval_protocol_violation is truthy",
    ),
)
CAVES_V0_COMPONENT_BY_NAME = {spec.name: spec for spec in CAVES_V0_COMPONENT_SPECS}


@dataclass(frozen=True)
class TemperatureCalibrator:
    temperature: float

    def predict(self, scores: Sequence[Any]) -> List[float]:
        out: List[float] = []
        temp = float(self.temperature)
        if not np.isfinite(temp) or temp <= 0.0:
            return [float("nan") for _ in scores]
        for score_raw in scores:
            score = _safe_float(score_raw)
            if not np.isfinite(score):
                out.append(float("nan"))
                continue
            clipped = float(np.clip(score, 1e-6, 1.0 - 1e-6))
            logit = float(np.log(clipped / (1.0 - clipped)))
            calibrated = 1.0 / (1.0 + np.exp(-logit / temp))
            out.append(_clip01(calibrated))
        return out


@dataclass(frozen=True)
class IsotonicCalibrator:
    thresholds: Tuple[float, ...]
    values: Tuple[float, ...]

    def predict(self, scores: Sequence[Any]) -> List[float]:
        out: List[float] = []
        if not self.thresholds:
            return [float("nan") for _ in scores]
        for score_raw in scores:
            score = _safe_float(score_raw)
            if not np.isfinite(score):
                out.append(float("nan"))
                continue
            idx = int(np.searchsorted(np.asarray(self.thresholds), score, side="right") - 1)
            idx = max(0, min(idx, len(self.values) - 1))
            out.append(float(self.values[idx]))
        return out


def _safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        if value is None or value == "":
            return default
        out = float(value)
        return out if np.isfinite(out) else default
    except (TypeError, ValueError):
        return default


def _safe_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return False


def _clip01(value: float) -> float:
    if not np.isfinite(value):
        return float("nan")
    return float(min(1.0, max(0.0, value)))


def _add_component(components: List[tuple[float, float, str]], value: float, weight: float, reason: str) -> None:
    if np.isfinite(value):
        components.append((_clip01(value), float(weight), reason))


def _caves_v0_weight(name: str) -> float:
    return CAVES_V0_COMPONENT_BY_NAME[name].weight


def describe_caves_v0_components() -> List[Dict[str, Any]]:
    """Return auditable, JSON/CSV-serializable CAVES v0 component definitions."""
    return [
        {
            "name": spec.name,
            "weight": spec.weight,
            "source_fields": list(spec.source_fields),
            "formula_note": spec.formula_note,
            "label_like_fields_excluded": list(CAVES_V0_LABEL_LIKE_FIELDS_EXCLUDED),
        }
        for spec in CAVES_V0_COMPONENT_SPECS
    ]


def compute_caves_v0(row: Dict[str, Any], *, reject_threshold: float = 0.5) -> CavesScore:
    """Compute a transparent v0 false-localization risk score from row diagnostics.

    The score deliberately avoids label-like fields such as post-fusion position
    error, `bad_pose_label`, and `unsafe_accept`. It uses only confidence and
    observability diagnostics that are available to the runtime before a paper
    evaluator labels the result.
    """
    components: List[tuple[float, float, str]] = []

    num_matches = _safe_float(row.get("num_matches"))
    num_inliers = _safe_float(row.get("num_inliers"))
    inlier_ratio = _safe_float(row.get("inlier_ratio"))
    if not np.isfinite(inlier_ratio) and np.isfinite(num_matches) and num_matches > 0 and np.isfinite(num_inliers):
        inlier_ratio = float(num_inliers / num_matches)
    _add_component(components, 1.0 - min(max(inlier_ratio, 0.0), 0.6) / 0.6, _caves_v0_weight("low_inlier_ratio"), "low_inlier_ratio")

    if np.isfinite(num_inliers):
        _add_component(components, 1.0 - min(max(num_inliers, 0.0), 20.0) / 20.0, _caves_v0_weight("low_inlier_count"), "low_inlier_count")

    reproj = _safe_float(row.get("reproj_residual_px"))
    _add_component(components, (reproj - 3.0) / 17.0, _caves_v0_weight("high_reprojection_residual"), "high_reprojection_residual")

    shape = _safe_float(row.get("shape_confidence"))
    _add_component(components, 1.0 - min(max(shape, 0.0), 1.0), _caves_v0_weight("low_shape_confidence"), "low_shape_confidence")

    margin = _safe_float(row.get("retrieval_score_margin"))
    _add_component(components, 1.0 - min(max(margin, 0.0), 0.25) / 0.25, _caves_v0_weight("low_retrieval_margin"), "low_retrieval_margin")

    rank = _safe_float(row.get("retrieval_rank", row.get("candidate_rank_selected")))
    if np.isfinite(rank):
        # Ranks are one-based in exported diagnostics; tolerate zero-based internals by clipping.
        rank_one_based = max(1.0, rank)
        _add_component(components, (rank_one_based - 1.0) / 4.0, _caves_v0_weight("late_retrieval_rank"), "late_retrieval_rank")

    anchor_offset = _safe_float(row.get("geometry_anchor_offset_m"))
    _add_component(components, (anchor_offset - 20.0) / 130.0, _caves_v0_weight("large_anchor_offset"), "large_anchor_offset")

    if _safe_bool(row.get("final_pose_gate_reject")):
        _add_component(components, 1.0, _caves_v0_weight("final_pose_gate_reject"), "final_pose_gate_reject")
    if _safe_bool(row.get("retrieval_protocol_violation")):
        _add_component(components, 1.0, _caves_v0_weight("retrieval_protocol_violation"), "retrieval_protocol_violation")

    if not components:
        return CavesScore(
            risk_score=float("nan"),
            confidence=float("nan"),
            decision=CAVES_ABSTAIN,
            obs_gate_score=float("nan"),
            obs_gate_reason="no_caves_evidence",
            evidence_count=0,
        )

    weighted_sum = sum(value * weight for value, weight, _ in components)
    weight_sum = sum(weight for _, weight, _ in components)
    risk = _clip01(weighted_sum / weight_sum)
    confidence = _clip01(1.0 - risk)
    dominant = max(components, key=lambda item: (item[0] * item[1], item[1]))[2]
    decision = CAVES_REJECT if risk >= float(reject_threshold) else CAVES_ACCEPT
    return CavesScore(
        risk_score=risk,
        confidence=confidence,
        decision=decision,
        obs_gate_score=risk,
        obs_gate_reason=dominant,
        evidence_count=len(components),
    )


def apply_caves_v0(row: Dict[str, Any], *, reject_threshold: float = 0.5) -> None:
    """Populate CAVES schema fields in-place without changing success semantics."""
    score = compute_caves_v0(row, reject_threshold=reject_threshold)
    row["caves_risk_score"] = score.risk_score
    row["caves_confidence"] = score.confidence
    row["caves_decision"] = score.decision
    row["obs_gate_score"] = score.obs_gate_score
    row["obs_gate_reason"] = score.obs_gate_reason
    row["caves_evidence_count"] = score.evidence_count


def apply_caves_outcome_labels(
    row: Dict[str, Any],
    *,
    localization_success_threshold_m: float,
    retrieval_correct_threshold_m: float | None = None,
) -> None:
    """Populate row-level safety labels without changing benchmark success semantics."""
    loc_threshold = float(localization_success_threshold_m)
    retrieval_threshold = loc_threshold if retrieval_correct_threshold_m is None else float(retrieval_correct_threshold_m)
    row["localization_success_threshold_m"] = loc_threshold
    row["retrieval_correct_threshold_m"] = retrieval_threshold

    error_m = _safe_float(row.get("post_fgo_error_m", row.get("position_error_m")))
    localization_success = bool(np.isfinite(error_m) and error_m <= loc_threshold)
    caves_reject = str(row.get("caves_decision", "")).strip().lower() == CAVES_REJECT
    accepted = _safe_bool(row.get("accepted_for_metrics", row.get("success")))
    bad_pose = bool(np.isfinite(error_m) and error_m > loc_threshold)

    row["localization_is_success"] = localization_success
    row["caves_reject"] = caves_reject
    row["false_reject"] = bool(caves_reject and localization_success)
    row["bad_pose_label"] = bool(_safe_bool(row.get("bad_pose_label", bad_pose)) or bad_pose)
    row["unsafe_accept"] = bool(accepted and row["bad_pose_label"])
    if row["unsafe_accept"] and np.isfinite(error_m):
        row["unsafe_accept_severity_m"] = float(max(0.0, error_m - loc_threshold))
    else:
        row["unsafe_accept_severity_m"] = 0.0


def _finite_label_score(labels: Sequence[Any], scores: Sequence[Any]) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray([1 if _safe_bool(v) else 0 for v in labels], dtype=float)
    s = np.asarray([_safe_float(v) for v in scores], dtype=float)
    mask = np.isfinite(s)
    return y[mask], s[mask]


def roc_auc_score(labels: Sequence[Any], scores: Sequence[Any]) -> float:
    """Compute binary AUROC with average ranks for ties."""
    y, s = _finite_label_score(labels, scores)
    if y.size == 0 or np.sum(y == 1) == 0 or np.sum(y == 0) == 0:
        return float("nan")
    order = np.argsort(s)
    ranks = np.empty_like(s, dtype=float)
    sorted_scores = s[order]
    n = len(s)
    start = 0
    while start < n:
        end = start + 1
        while end < n and sorted_scores[end] == sorted_scores[start]:
            end += 1
        avg_rank = (start + 1 + end) / 2.0
        ranks[order[start:end]] = avg_rank
        start = end
    pos_ranks = np.sum(ranks[y == 1])
    n_pos = np.sum(y == 1)
    n_neg = np.sum(y == 0)
    return float((pos_ranks - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def average_precision_score(labels: Sequence[Any], scores: Sequence[Any]) -> float:
    y, s = _finite_label_score(labels, scores)
    if y.size == 0 or np.sum(y == 1) == 0:
        return float("nan")
    order = np.argsort(-s)
    y_sorted = y[order]
    tp = 0.0
    precisions: List[float] = []
    for idx, label in enumerate(y_sorted, start=1):
        if label == 1:
            tp += 1.0
            precisions.append(tp / idx)
    return float(np.mean(precisions)) if precisions else float("nan")


def brier_score(labels: Sequence[Any], scores: Sequence[Any]) -> float:
    y, s = _finite_label_score(labels, scores)
    if y.size == 0:
        return float("nan")
    s = np.clip(s, 0.0, 1.0)
    return float(np.mean((s - y) ** 2))


def expected_calibration_error(labels: Sequence[Any], scores: Sequence[Any], *, n_bins: int = 10) -> float:
    y, s = _finite_label_score(labels, scores)
    if y.size == 0:
        return float("nan")
    s = np.clip(s, 0.0, 1.0)
    n_bins = max(1, int(n_bins))
    ece = 0.0
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    for idx in range(n_bins):
        lo = edges[idx]
        hi = edges[idx + 1]
        if idx == n_bins - 1:
            mask = (s >= lo) & (s <= hi)
        else:
            mask = (s >= lo) & (s < hi)
        if not np.any(mask):
            continue
        bin_conf = float(np.mean(s[mask]))
        bin_acc = float(np.mean(y[mask]))
        ece += float(np.mean(mask)) * abs(bin_conf - bin_acc)
    return float(ece)


def risk_coverage_curve(labels: Sequence[Any], scores: Sequence[Any]) -> List[Dict[str, float]]:
    """Return accepted-error risk for prefixes sorted by increasing risk score."""
    y, s = _finite_label_score(labels, scores)
    if y.size == 0:
        return []
    order = np.argsort(s)
    y_sorted = y[order]
    s_sorted = s[order]
    n = len(y_sorted)
    rows: List[Dict[str, float]] = []
    bad_so_far = 0.0
    for idx, label in enumerate(y_sorted, start=1):
        bad_so_far += float(label)
        coverage = idx / n
        rows.append({
            "coverage": float(coverage),
            "risk": float(bad_so_far / idx),
            "threshold": float(s_sorted[idx - 1]),
            "accepted": float(idx),
            "total": float(n),
        })
    return rows


def summarize_caves_metrics(labels: Sequence[Any], scores: Sequence[Any], *, threshold: float = 0.5, n_bins: int = 10) -> Dict[str, float]:
    y, s = _finite_label_score(labels, scores)
    if y.size == 0:
        return {
            "num_samples": 0,
            "num_bad_pose": 0,
            "auroc": float("nan"),
            "auprc": float("nan"),
            "brier": float("nan"),
            "ece": float("nan"),
            "coverage": float("nan"),
            "accepted_bad_rate": float("nan"),
            "false_reject_rate": float("nan"),
            "safe_reject_rate": float("nan"),
            "forge_true_reject_rate": float("nan"),
            "forge_false_reject_rate": float("nan"),
        }
    reject = s >= float(threshold)
    accept = ~reject
    good = y == 0
    bad = y == 1
    accepted_bad = float(np.sum(accept & bad))
    accepted_total = float(np.sum(accept))
    false_rejects = float(np.sum(reject & good))
    rejected_bad = float(np.sum(reject & bad))
    good_total = float(np.sum(good))
    bad_total = float(np.sum(bad))
    return {
        "num_samples": int(y.size),
        "num_bad_pose": int(np.sum(bad)),
        "auroc": roc_auc_score(y, s),
        "auprc": average_precision_score(y, s),
        "brier": brier_score(y, s),
        "ece": expected_calibration_error(y, s, n_bins=n_bins),
        "coverage": float(accepted_total / y.size),
        "accepted_bad_rate": float(accepted_bad / accepted_total) if accepted_total > 0 else float("nan"),
        "false_reject_rate": float(false_rejects / good_total) if good_total > 0 else float("nan"),
        "safe_reject_rate": float(rejected_bad / bad_total) if bad_total > 0 else float("nan"),
        "forge_true_reject_rate": float(rejected_bad / bad_total) if bad_total > 0 else float("nan"),
        "forge_false_reject_rate": float(false_rejects / good_total) if good_total > 0 else float("nan"),
        "threshold": float(threshold),
    }


def _inlier_ratio_risk(row: Dict[str, Any]) -> float:
    num_matches = _safe_float(row.get("num_matches"))
    num_inliers = _safe_float(row.get("num_inliers"))
    ratio = _safe_float(row.get("inlier_ratio"))
    if not np.isfinite(ratio) and np.isfinite(num_matches) and num_matches > 0 and np.isfinite(num_inliers):
        ratio = float(num_inliers / num_matches)
    if not np.isfinite(ratio):
        return float("nan")
    return _clip01(1.0 - min(max(ratio, 0.0), 0.6) / 0.6)


def _retrieval_margin_risk(row: Dict[str, Any]) -> float:
    margin = _safe_float(row.get("retrieval_score_margin"))
    if not np.isfinite(margin):
        return float("nan")
    return _clip01(1.0 - min(max(margin, 0.0), 0.25) / 0.25)


def _retrieval_rank_risk(row: Dict[str, Any]) -> float:
    rank = _safe_float(row.get("retrieval_rank", row.get("candidate_rank_selected")))
    if not np.isfinite(rank):
        return float("nan")
    return _clip01((max(1.0, rank) - 1.0) / 4.0)


def _mean_finite(values: Sequence[float]) -> float:
    finite = [float(v) for v in values if np.isfinite(v)]
    return float(np.mean(finite)) if finite else float("nan")


def score_row_for_feature_set(row: Dict[str, Any], feature_set: str = "full_caves") -> float:
    """Score a row for CAVES ablation feature sets."""
    feature_set = str(feature_set or "full_caves")
    if feature_set == "full_caves":
        return compute_caves_v0(row).risk_score
    if feature_set == "inlier_ratio_only":
        return _inlier_ratio_risk(row)
    if feature_set == "retrieval_margin_only":
        return _retrieval_margin_risk(row)
    if feature_set == "retrieval_margin_plus_rank":
        return _mean_finite([_retrieval_margin_risk(row), _retrieval_rank_risk(row)])
    if feature_set == "geometry_only":
        geom = dict(row)
        for key in (
            "retrieval_score_margin",
            "retrieval_rank",
            "candidate_rank_selected",
            "retrieval_protocol_violation",
        ):
            geom.pop(key, None)
        return compute_caves_v0(geom).risk_score
    if feature_set == "full_without_rank":
        no_rank = dict(row)
        no_rank.pop("retrieval_rank", None)
        no_rank.pop("candidate_rank_selected", None)
        return compute_caves_v0(no_rank).risk_score
    if feature_set == "gate_only":
        gate_reject = _safe_bool(row.get("final_pose_gate_reject")) or str(row.get("failure_reason", "")).strip() == "final_pose_gate_reject"
        return 1.0 if gate_reject else 0.0
    raise ValueError(f"Unknown CAVES feature_set: {feature_set}")


def rows_to_labels_scores(rows: Iterable[Dict[str, Any]], *, feature_set: str = "full_caves") -> tuple[List[int], List[float]]:
    labels: List[int] = []
    scores: List[float] = []
    for row in rows:
        score = score_row_for_feature_set(row, feature_set=feature_set)
        if not np.isfinite(score):
            continue
        labels.append(1 if _safe_bool(row.get("bad_pose_label")) else 0)
        scores.append(float(score))
    return labels, scores


def fit_temperature_calibrator(labels: Sequence[Any], scores: Sequence[Any]) -> TemperatureCalibrator:
    """Fit temperature scaling on score-derived logits (not neural logits)."""
    y, s = _finite_label_score(labels, scores)
    if y.size == 0 or np.sum(y == 1) == 0 or np.sum(y == 0) == 0:
        return TemperatureCalibrator(float("nan"))
    logits = np.log(np.clip(s, 1e-6, 1.0 - 1e-6) / np.clip(1.0 - s, 1e-6, 1.0 - 1e-6))
    best_temp = float("nan")
    best_nll = float("inf")
    for temp in np.geomspace(0.05, 20.0, 200):
        probs = 1.0 / (1.0 + np.exp(-logits / float(temp)))
        probs = np.clip(probs, 1e-6, 1.0 - 1e-6)
        nll = float(-np.mean(y * np.log(probs) + (1.0 - y) * np.log(1.0 - probs)))
        if nll < best_nll:
            best_nll = nll
            best_temp = float(temp)
    return TemperatureCalibrator(best_temp)


def fit_isotonic_calibrator(labels: Sequence[Any], scores: Sequence[Any]) -> IsotonicCalibrator:
    """Fit a dependency-free monotone isotonic calibrator using PAV."""
    y, s = _finite_label_score(labels, scores)
    if y.size == 0:
        return IsotonicCalibrator((), ())
    order = np.argsort(s)
    sorted_s = s[order]
    sorted_y = y[order]
    blocks: List[Dict[str, float]] = []
    for score, label in zip(sorted_s, sorted_y):
        blocks.append({"start": float(score), "end": float(score), "weight": 1.0, "sum": float(label)})
        while len(blocks) >= 2:
            prev = blocks[-2]
            cur = blocks[-1]
            if prev["sum"] / prev["weight"] <= cur["sum"] / cur["weight"]:
                break
            merged = {
                "start": prev["start"],
                "end": cur["end"],
                "weight": prev["weight"] + cur["weight"],
                "sum": prev["sum"] + cur["sum"],
            }
            blocks[-2:] = [merged]
    thresholds = tuple(block["start"] for block in blocks)
    values = tuple(_clip01(block["sum"] / block["weight"]) for block in blocks)
    return IsotonicCalibrator(thresholds=thresholds, values=values)


def _metric_value(labels: np.ndarray, scores: np.ndarray, *, metric: str, n_bins: int) -> float:
    if metric == "auroc":
        return roc_auc_score(labels, scores)
    if metric == "ece":
        return expected_calibration_error(labels, scores, n_bins=n_bins)
    raise ValueError(f"Unknown bootstrap metric: {metric}")


def bootstrap_metric_ci(
    labels: Sequence[Any],
    scores: Sequence[Any],
    *,
    groups: Optional[Sequence[Any]] = None,
    metric: str = "auroc",
    n_bootstrap: int = 1000,
    seed: int = 0,
    ci: float = 0.95,
    n_bins: int = 10,
) -> Dict[str, Any]:
    y, s = _finite_label_score(labels, scores)
    if groups is None:
        group_values = np.asarray([str(i) for i in range(len(y))], dtype=object)
    else:
        raw_groups = list(groups)
        if len(raw_groups) != len(labels):
            raise ValueError("groups must have the same length as labels before finite-score filtering")
        score_arr = np.asarray([_safe_float(v) for v in scores], dtype=float)
        mask = np.isfinite(score_arr)
        group_values = np.asarray([str(v) for v in raw_groups], dtype=object)[mask]
    point = _metric_value(y, s, metric=metric, n_bins=n_bins)
    unique_groups = sorted(set(str(group) for group in group_values))
    if y.size == 0 or not np.isfinite(point) or not unique_groups:
        return {
            "metric": metric,
            "point": float(point),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "bootstrap_n": int(n_bootstrap),
            "bootstrap_valid": 0,
            "grouped": bool(groups is not None),
            "num_groups": int(len(unique_groups)),
        }
    by_group = {group: np.where(group_values == group)[0] for group in unique_groups}
    rng = np.random.default_rng(int(seed))
    vals: List[float] = []
    for _ in range(max(0, int(n_bootstrap))):
        sampled_groups = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        idx = np.concatenate([by_group[str(group)] for group in sampled_groups])
        value = _metric_value(y[idx], s[idx], metric=metric, n_bins=n_bins)
        if np.isfinite(value):
            vals.append(float(value))
    if vals:
        alpha = (1.0 - float(ci)) / 2.0
        low = float(np.quantile(vals, alpha))
        high = float(np.quantile(vals, 1.0 - alpha))
    else:
        low = high = float("nan")
    return {
        "metric": metric,
        "point": float(point),
        "ci_low": low,
        "ci_high": high,
        "bootstrap_n": int(n_bootstrap),
        "bootstrap_valid": int(len(vals)),
        "grouped": bool(groups is not None),
        "num_groups": int(len(unique_groups)),
    }


def bootstrap_auroc_ci(
    labels: Sequence[Any],
    scores: Sequence[Any],
    *,
    n_bootstrap: int = 1000,
    seed: int = 0,
    ci: float = 0.95,
) -> Dict[str, float]:
    """Bootstrap AUROC confidence interval, skipping one-class resamples."""
    y, s = _finite_label_score(labels, scores)
    point = roc_auc_score(y, s)
    if y.size == 0 or not np.isfinite(point):
        return {
            "auroc": point,
            "auroc_ci_low": float("nan"),
            "auroc_ci_high": float("nan"),
            "auroc_bootstrap_n": int(n_bootstrap),
            "auroc_bootstrap_valid": 0,
        }
    rng = np.random.default_rng(int(seed))
    aucs: List[float] = []
    for _ in range(max(0, int(n_bootstrap))):
        idx = rng.integers(0, y.size, size=y.size)
        sample_auc = roc_auc_score(y[idx], s[idx])
        if np.isfinite(sample_auc):
            aucs.append(float(sample_auc))
    if not aucs:
        low = high = float("nan")
    else:
        alpha = (1.0 - float(ci)) / 2.0
        low = float(np.quantile(aucs, alpha))
        high = float(np.quantile(aucs, 1.0 - alpha))
    return {
        "auroc": point,
        "auroc_ci_low": low,
        "auroc_ci_high": high,
        "auroc_bootstrap_n": int(n_bootstrap),
        "auroc_bootstrap_valid": int(len(aucs)),
    }
