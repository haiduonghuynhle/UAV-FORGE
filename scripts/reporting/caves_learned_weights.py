#!/usr/bin/env python3
"""Fit CAVES component weights instead of setting them by hand, and report the gap.

CAVES v0 combines nine risk components under nine weights chosen before any
evaluation. This script asks what those weights would have been had they been
learned, and whether learning them helps.

Protocol: leave-one-dataset-out. Weights are fitted on two datasets and every
number is reported on the third, which never entered the fit. That is the only
honest comparison against a hand-set rule, because a rule fitted and scored on
the same frames always wins.

Population and label match tab_caves_components exactly: accepted frames only,
bad := final position error beyond the dataset radius. Features are the CAVES
source fields. Two of the nine are dead on v3 runs, so they are dropped rather
than silently imputed to zero:
  * final_pose_gate_reject is not exported as a column;
  * retrieval_protocol_violation is constant False.

Features are standardized on the training fold before fitting, so the reported
coefficients are comparable to one another and to the hand-set weights. Without
that, L2 regularization penalizes a feature for its units.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

_REPORTING_ROOT = Path(__file__).resolve().parent
if str(_REPORTING_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPORTING_ROOT))

import evaluate_caves_calibration as eval_caves  # noqa: E402
from uav_geolocalization.benchmark.caves import (  # noqa: E402
    CAVES_V0_COMPONENT_SPECS,
    bootstrap_metric_ci,
    summarize_caves_metrics,
)

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
    from sklearn.metrics import roc_auc_score
except ImportError:  # pragma: no cover
    LogisticRegression = None


def _lr_l2():
    return LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs")


def _lr_l1():
    return LogisticRegression(C=1.0, max_iter=2000, solver="liblinear", penalty="l1")


def _gb():
    # Small and shallow: the training folds hold a few thousand rows, and the
    # point is to test whether the weighted-sum form is the ceiling, not to
    # win by capacity.
    return GradientBoostingClassifier(n_estimators=200, max_depth=3, learning_rate=0.05,
                                      subsample=0.9, random_state=0)


def _mlp():
    # The examiner's charge is that CAVES is "a one-layer network with W set by
    # hand". This is the literal test of that charge: an actual trained network
    # on the same seven scalars. Kept small on purpose - the training folds
    # hold a few thousand rows, and the question is whether capacity helps at
    # all, not how much capacity fits.
    return MLPClassifier(hidden_layer_sizes=(32, 16), activation="relu",
                         solver="adam", alpha=1e-3, max_iter=2000,
                         early_stopping=True, validation_fraction=0.15,
                         n_iter_no_change=25, random_state=0)


def _tabpfn(features=None):
    # Prior-fitted transformer; own preprocessing, no tuning knobs to leak.
    from tabpfn import TabPFNClassifier
    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        device = "cpu"
    return TabPFNClassifier(device=device)


# Monotone directions with a physical justification that also held in the
# per-dataset direction audit. The three components whose assumed direction
# reversed on at least one dataset (inlier ratio, inlier count, reprojection
# residual) are deliberately left unconstrained: forcing an unstable direction
# would build the misspecification into the model instead of fixing it.
MONOTONE_DIRECTION = {
    "shape_confidence": -1,          # weaker shape agreement -> riskier
    "geometry_anchor_offset_m": +1,  # solver landing far from its tile -> riskier
    "retrieval_score_margin": -1,    # more retrieval ambiguity -> riskier
}


def _xgb_mono(features=None):
    from xgboost import XGBClassifier
    constraints = tuple(MONOTONE_DIRECTION.get(f, 0) for f in (features or []))
    return XGBClassifier(
        n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.9,
        monotone_constraints=constraints, random_state=0,
        eval_metric="logloss",
    )


MODEL_FACTORIES = {"lr_l2": _lr_l2, "lr_l1": _lr_l1, "gb": _gb, "mlp": _mlp,
                   "tabpfn": _tabpfn, "xgb_mono": _xgb_mono}

# The seven CAVES inputs that carry information on v3 runs.
CAVES_NO_RANK = False


def trajectory_blocks(rows: List[Dict[str, Any]], target: int = 25) -> List[List[int]]:
    """Contiguous stretches of one flight, as row-index lists.

    VPAIR and AVL publish no flight id at all and UAV-VisLoc publishes three,
    so resampling flights is either impossible or far too coarse. Blocks of
    consecutive frames are the resampling unit that matches how the data was
    actually collected.
    """
    by_flight: Dict[str, List[int]] = {}
    for index, row in enumerate(rows):
        flight = str(row.get("query_flight_id", "") or "")
        if not flight:
            name = str(row.get("image_name", "") or "")
            flight = name.split("/")[0] if "/" in name else ""
        by_flight.setdefault(flight, []).append(index)
    blocks: List[List[int]] = []
    for flight, indices in by_flight.items():
        indices.sort(key=lambda i: str(rows[i].get("image_name", "")))
        per_flight = max(1, target // max(1, len(by_flight)))
        size = max(1, math.ceil(len(indices) / per_flight))
        block: List[int] = []
        for i in indices:
            same_frame = block and (str(rows[i].get("image_name", ""))
                                    == str(rows[block[-1]].get("image_name", "")))
            if len(block) >= size and not same_frame:
                blocks.append(block)
                block = []
            block.append(i)
        if block:
            blocks.append(block)
    return blocks

LIVE_FEATURES = [
    "inlier_ratio",
    "num_inliers",
    "reproj_residual_px",
    "shape_confidence",
    "retrieval_score_margin",
    "retrieval_rank",
    "geometry_anchor_offset_m",
]
DEAD_FEATURES = ["final_pose_gate_reject", "retrieval_protocol_violation"]

# Which hand-set component each feature drives, for the side-by-side table.
FEATURE_TO_COMPONENT = {
    "inlier_ratio": "low_inlier_ratio",
    "num_inliers": "low_inlier_count",
    "reproj_residual_px": "high_reprojection_residual",
    "shape_confidence": "low_shape_confidence",
    "retrieval_score_margin": "low_retrieval_margin",
    "retrieval_rank": "late_retrieval_rank",
    "geometry_anchor_offset_m": "large_anchor_offset",
}

# Sign the hand-set rule assumes: +1 means "larger value means more risk".
HANDSET_RISK_DIRECTION = {
    "inlier_ratio": -1,
    "num_inliers": -1,
    "reproj_residual_px": +1,
    "shape_confidence": -1,
    "retrieval_score_margin": -1,
    "retrieval_rank": +1,
    "geometry_anchor_offset_m": +1,
}


def handset_weights() -> Dict[str, float]:
    return {spec.name: float(spec.weight) for spec in CAVES_V0_COMPONENT_SPECS}


def as_float(value) -> Optional[float]:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def feature_vector(row: Dict[str, Any], features: Sequence[str]) -> Optional[List[float]]:
    """Return the row's features, or None if any is missing.

    Dropping incomplete rows rather than imputing zero keeps a missing signal
    from masquerading as a confident measurement of zero.
    """
    out: List[float] = []
    for field in features:
        raw = row.get(field)
        if field == "retrieval_rank" and raw in (None, ""):
            raw = row.get("candidate_rank_selected")
        if field == "inlier_ratio" and raw in (None, ""):
            inliers = as_float(row.get("num_inliers"))
            matches = as_float(row.get("num_matches"))
            raw = (inliers / matches) if inliers is not None and matches else None
        if isinstance(raw, str) and raw.strip().lower() in {"true", "false"}:
            raw = 1.0 if raw.strip().lower() == "true" else 0.0
        value = as_float(raw)
        if value is None:
            return None
        out.append(value)
    return out


def label_of(row: Dict[str, Any], source: str) -> Optional[int]:
    """1 means the pose is bad.

    `clean` uses whatever the clean run happened to get wrong, so its positives
    are only the faults that occurred by luck. `forge` uses the injected runs,
    where a frame is positive when it was actually given a wrong reference:
    the faults are produced on purpose and in known quantity, which is the
    whole reason the injection exists.
    """
    if source == "clean":
        if row.get("bad_pose_label") in (None, ""):
            return None
        return 1 if eval_caves._truthy(row.get("bad_pose_label")) else 0
    if source == "forge":
        if "adversarial_attacked" not in row:
            return None
        return 1 if eval_caves._truthy(row.get("adversarial_attacked")) else 0
    raise SystemExit(f"unknown label source: {source}")


def load(roots: Sequence[Path], label_source: str = "clean",
         accepted_only: bool = True) -> List[Dict[str, Any]]:
    paths = eval_caves.discover_prediction_csvs([Path(r) for r in roots])
    rows = eval_caves.read_prediction_rows(paths)
    keep: List[Dict[str, Any]] = []
    for row in rows:
        if row.get("artifact_read_error"):
            continue
        label = label_of(row, label_source)
        if label is None:
            continue
        # Acceptance is decided partly from the same geometry the score reads,
        # so filtering on it conditions on a collider: treatment and evidence
        # both drive survival into the sample. Kept as the default only because
        # it is the population tab_caves_components scores; switch it off to
        # test whether a result survives without the conditioning.
        if accepted_only and not eval_caves._truthy(row.get("accepted_for_metrics")):
            continue
        row["_caves_label"] = label
        keep.append(row)
    return keep


def forward_select(
    x: np.ndarray,
    y: np.ndarray,
    names: Sequence[str],
    folds: int = 5,
    seed: int = 0,
    groups: Optional[Sequence[str]] = None,
) -> List[int]:
    """Greedy forward selection scored by inner cross-validation.

    Run entirely inside the training fold. Selecting on the held-out dataset
    would be the same leak as fitting on it, and would make the reported
    numbers a description of the test set rather than a prediction about it.
    """
    remaining = list(range(len(names)))
    chosen: List[int] = []
    best_overall = 0.5
    while remaining:
        scored = []
        for idx in remaining:
            trial = chosen + [idx]
            scores = []
            # Group-aware: one frame evaluated under several injection
            # conditions must not land on both sides of an inner split.
            if groups is not None:
                splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
                split_iter = splitter.split(x[:, trial], y, groups=groups)
            else:
                splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
                split_iter = splitter.split(x[:, trial], y)
            for tr, va in split_iter:
                if len(set(y[tr].tolist())) < 2 or len(set(y[va].tolist())) < 2:
                    continue
                sub = x[:, trial]
                mean, scale = sub[tr].mean(axis=0), sub[tr].std(axis=0)
                scale = np.where(scale == 0, 1.0, scale)
                model = _lr_l2()
                model.fit((sub[tr] - mean) / scale, y[tr])
                probability = model.predict_proba((sub[va] - mean) / scale)[:, 1]
                scores.append(roc_auc_score(y[va], probability))
            scored.append((float(np.mean(scores)) if scores else 0.5, idx))
        scored.sort(reverse=True)
        best, idx = scored[0]
        if best <= best_overall + 1e-4:      # no useful gain; stop
            break
        best_overall = best
        chosen.append(idx)
        remaining.remove(idx)
    return chosen or list(range(len(names)))


def fold(
    train: List[Dict[str, Any]],
    test: List[Dict[str, Any]],
    features: Sequence[str],
    bootstrap: int,
    seed: int,
    models: Sequence[str] = ("lr_l2",),
    select: bool = False,
) -> Dict[str, Any]:
    if LogisticRegression is None:
        raise SystemExit("scikit-learn is required")

    def matrix(rows: List[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
        xs, ys, kept = [], [], []
        for row in rows:
            vec = feature_vector(row, features)
            if vec is None:
                continue
            xs.append(vec)
            ys.append(int(row["_caves_label"]))
            kept.append(row)
        return np.array(xs, dtype=np.float64), np.array(ys, dtype=np.int32), kept

    x_tr, y_tr, rows_tr = matrix(train)
    x_te, y_te, rows_te = matrix(test)
    if len(set(y_tr.tolist())) < 2 or len(set(y_te.tolist())) < 2:
        return {"status": "degenerate", "n_train": int(len(y_tr)), "n_test": int(len(y_te))}

    # Standardize on the training fold only; test rows are transformed with the
    # training statistics, never their own.
    mean = x_tr.mean(axis=0)
    scale = x_tr.std(axis=0)
    scale[scale == 0] = 1.0
    x_tr_s, x_te_s = (x_tr - mean) / scale, (x_te - mean) / scale

    train_groups = [f"{r.get('dataset', '')}|{r.get('image_name', '')}" for r in rows_tr]
    selected = (forward_select(x_tr_s, y_tr, features, seed=seed, groups=train_groups)
                if select else list(range(len(features))))
    selected_names = [features[i] for i in selected]

    if CAVES_NO_RANK:
        # Recorded scores carry the rank component, and the injector never
        # selects the top-ranked candidate, so rank marks the arm rather than
        # the evidence. Rescore without it for a fair comparison.
        from forge_mixed_confusion import caves_risk
        caves_scores = [caves_risk(r, True) for r in rows_te]
    else:
        caves_scores = [as_float(r.get("caves_risk_score")) for r in rows_te]
    keep_idx = [i for i, c in enumerate(caves_scores) if c is not None]
    labels = [int(y_te[i]) for i in keep_idx]
    caves_only = [caves_scores[i] for i in keep_idx]
    # Resampling unit: contiguous within-flight frame blocks, the same rule the
    # paired-delta CIs use. A block keeps serial correlation and the repeats of
    # a frame across injection conditions together (cuts land on frame boundaries), so
    # neither narrows the interval on samples never drawn independently.
    rows_kept = [rows_te[i] for i in keep_idx]
    block_of: Dict[int, str] = {}
    for b, idxs in enumerate(trajectory_blocks(rows_kept)):
        for i in idxs:
            block_of[i] = f"block:{b}"
    groups = [block_of[j] for j in range(len(rows_kept))]

    # The positive class mixes two phenomenologies: frames given a wrong
    # reference on purpose, and control-arm frames that went unsafe on their
    # own. A scorer can catch one and miss the other, and the mixture differs
    # by dataset, so score each positive subtype against the shared negatives.
    def strata_auroc(sc: Sequence[float]) -> Dict[str, Any]:
        neg = [i for i, y in enumerate(labels) if y == 0]
        out: Dict[str, Any] = {}
        for tag, want in (("forced", True), ("control", False)):
            pos = [i for i, y in enumerate(labels)
                   if y == 1 and bool(eval_caves._truthy(
                       rows_kept[i].get("adversarial_attacked"))) == want]
            if not pos:
                continue
            idx = pos + neg
            sub_labels = [labels[i] for i in idx]
            sub_scores = [sc[i] for i in idx]
            ci = bootstrap_metric_ci(sub_labels, sub_scores, metric="auroc",
                                     n_bootstrap=bootstrap, seed=seed,
                                     groups=[groups[i] for i in idx])
            out[tag] = {
                "n_pos": len(pos),
                "auroc": summarize_caves_metrics(sub_labels, sub_scores).get("auroc"),
                "auroc_ci": [ci["ci_low"], ci["ci_high"]],
            }
        return out

    variants: Dict[str, Any] = {}
    coefficients: List[Dict[str, Any]] = []
    for model_name in models:
        columns = selected if model_name.startswith("lr") else list(range(len(features)))
        factory = MODEL_FACTORIES[model_name]
        try:
            estimator = factory(features=[features[i] for i in columns])
        except TypeError:
            estimator = factory()
        estimator.fit(x_tr_s[:, columns], y_tr)
        probability = estimator.predict_proba(x_te_s[:, columns])[:, 1]
        scores = [float(probability[i]) for i in keep_idx]
        summary = summarize_caves_metrics(labels, scores, threshold=0.5)
        interval = bootstrap_metric_ci(labels, scores, metric="auroc",
                                       n_bootstrap=bootstrap, seed=seed, groups=groups)
        variants[model_name] = {
            "auroc": summary.get("auroc"),
            "auroc_ci": [interval["ci_low"], interval["ci_high"]],
            "features_used": [features[i] for i in columns],
            "scores": scores,
            "strata": strata_auroc(scores),
        }
        if model_name == "lr_l2" and not coefficients:
            model = estimator
            coefficient_columns = columns

    caves_summary = summarize_caves_metrics(labels, caves_only, threshold=0.5)
    caves_ci = bootstrap_metric_ci(labels, caves_only, metric="auroc",
                                   n_bootstrap=bootstrap, seed=seed, groups=groups)

    hand = handset_weights()
    for name, coef in zip([features[i] for i in coefficient_columns], model.coef_[0]):
        component = FEATURE_TO_COMPONENT.get(name, "")
        assumed = HANDSET_RISK_DIRECTION.get(name, 0)
        # A standardized coefficient is signed towards higher risk; the hand-set
        # rule assumes a fixed direction per component. Disagreement means the
        # fixed sign is wrong on this data, not merely the magnitude.
        learned_sign = 1 if coef > 0 else (-1 if coef < 0 else 0)
        coefficients.append({
            "feature": name,
            "component": component,
            "handset_weight": hand.get(component),
            "learned_coefficient": float(coef),
            "learned_abs_rank": 0,
            "handset_assumed_direction": assumed,
            "learned_direction": learned_sign,
            "direction_agrees": bool(assumed == learned_sign),
        })
    order = sorted(range(len(coefficients)), key=lambda i: -abs(coefficients[i]["learned_coefficient"]))
    for rank, idx in enumerate(order, start=1):
        coefficients[idx]["learned_abs_rank"] = rank

    return {
        "status": "ok",
        "n_train": int(len(y_tr)),
        "n_test": int(len(labels)),
        "n_bad_test": int(sum(labels)),
        "intercept": float(model.intercept_[0]),
        "coefficients": coefficients,
        "selected_features": selected_names,
        "variants": variants,
        "lr_auroc": variants.get("lr_l2", {}).get("auroc"),
        "lr_auroc_ci": variants.get("lr_l2", {}).get("auroc_ci", [float("nan")] * 2),
        "caves_auroc": caves_summary.get("auroc"),
        "caves_auroc_ci": [caves_ci["ci_low"], caves_ci["ci_high"]],
        "caves_strata": strata_auroc(caves_only),
        # Per-observation scores and labels, so risk-coverage curves reuse this
        # exact fit instead of refitting and risking a different number.
        "labels": labels,
        "caves_scores": caves_only,
        "blocks": groups,
    }


def render(results: Dict[str, Any], features: Sequence[str]) -> str:
    lines: List[str] = []
    lines.append("LEAVE-ONE-DATASET-OUT: weights fitted on the other two, scored here")
    lines.append("")
    header = f"{'held-out dataset':<14}{'n':>6}{'bad':>6}  {'scorer':<10}{'AUROC':>8}{'95% CI':>18}"
    lines.append(header)
    lines.append("-" * len(header))
    for dataset, res in results.items():
        if res.get("status") != "ok":
            lines.append(f"{dataset:<14}degenerate — {res.get('n_test', 0)} usable test rows")
            continue
        cl, ch = res["caves_auroc_ci"]
        lines.append(
            f"{dataset:<14}{res['n_test']:>6}{res['n_bad_test']:>6}  "
            f"{'CAVES v0':<10}{res['caves_auroc']:>8.3f}{f'[{cl:.3f}, {ch:.3f}]':>18}"
        )
        for model_name, variant in res.get("variants", {}).items():
            lo, hi = variant["auroc_ci"]
            lines.append(
                f"{'':<14}{'':>6}{'':>6}  {model_name:<10}{variant['auroc']:>8.3f}"
                f"{f'[{lo:.3f}, {hi:.3f}]':>18}"
            )
        if res.get("selected_features"):
            lines.append(f"{'':<14}selected: {', '.join(res['selected_features'])}")
    lines.append("")
    lines.append("Fitted weights against hand-set weights (standardized coefficients)")
    for dataset, res in results.items():
        if res.get("status") != "ok":
            continue
        lines.append("")
        lines.append(f"  held out: {dataset}   (fitted on the other two)")
        head = f"    {'component':<30}{'hand':>7}{'learned':>10}{'|rank|':>8}{'direction':>12}"
        lines.append(head)
        lines.append("    " + "-" * (len(head) - 4))
        for entry in sorted(res["coefficients"], key=lambda e: e["learned_abs_rank"]):
            hand = entry["handset_weight"]
            lines.append(
                f"    {entry['component']:<30}"
                f"{(f'{hand:.1f}' if hand is not None else '-'):>7}"
                f"{entry['learned_coefficient']:>10.3f}"
                f"{entry['learned_abs_rank']:>8}"
                f"{('agrees' if entry['direction_agrees'] else 'REVERSED'):>12}"
            )
    lines.append("")
    lines.append(f"  dropped as dead on v3 runs: {', '.join(DEAD_FEATURES)}")
    lines.append(
        "  'REVERSED' means the fitted weight points the opposite way to the sign the"
    )
    lines.append(
        "  hand-set rule assumes. Read it as a flag, not as proof: these features are"
    )
    lines.append(
        "  correlated, and a single coefficient's sign in a joint fit can flip for"
    )
    lines.append(
        "  reasons that have nothing to do with the feature's own direction. The"
    )
    lines.append(
        "  univariate per-component AUROC table is the cleaner evidence on direction;"
    )
    lines.append("  this table is evidence on transfer.")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--roots", nargs="+", required=True, help="run directories, any mix of datasets")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--strata-tex", type=Path,
                        help="write macros for the positive-subtype stratification")
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    parser.add_argument("--drop-rank", action="store_true", help="also drop retrieval_rank")
    parser.add_argument("--caves-no-rank", action="store_true",
                        help="rescore the hand-set CAVES baseline without its rank component")
    parser.add_argument("--models", nargs="+", default=["lr_l2"], choices=sorted(MODEL_FACTORIES))
    parser.add_argument("--select-features", action="store_true",
                        help="forward-select inside each training fold")
    parser.add_argument("--all-frames", action="store_true",
                        help="do not restrict to accepted frames (avoids the acceptance collider)")
    parser.add_argument("--drop-features", nargs="*", default=[],
                        help="feature names to exclude, e.g. geometry_anchor_offset_m")
    parser.add_argument("--label-source", default="clean", choices=["clean", "forge"],
                        help="clean: bad_pose_label from a clean run; forge: was this frame injected")
    args = parser.parse_args()

    global CAVES_NO_RANK
    CAVES_NO_RANK = bool(args.caves_no_rank)
    features = [f for f in LIVE_FEATURES
                if not (args.drop_rank and f == "retrieval_rank")
                and f not in set(args.drop_features)]
    rows = load([Path(r) for r in args.roots], label_source=args.label_source,
                accepted_only=not args.all_frames)
    if not rows:
        raise SystemExit("no accepted, labelled rows found under the given roots")

    by_dataset: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        by_dataset.setdefault(str(row.get("dataset", "unknown")), []).append(row)
    if len(by_dataset) < 2:
        raise SystemExit(f"leave-one-dataset-out needs at least two datasets, found {list(by_dataset)}")

    results: Dict[str, Any] = {}
    for held in sorted(by_dataset):
        train = [r for ds, rs in by_dataset.items() if ds != held for r in rs]
        results[held] = fold(train, by_dataset[held], features, args.bootstrap, args.bootstrap_seed,
                             models=args.models, select=args.select_features)

    print(render(results, features))

    if args.strata_tex:
        tex = {"avl": "Avl", "vpair": "Vpair", "uav_visloc": "Visloc"}
        name = {"caves": "Caves", "lr_l2": "Lr", "gb": "Gb", "mlp": "Mlp"}
        lines = ["% Generated by scripts/reporting/caves_learned_weights.py --strata-tex; do not edit.",
                 f"% roots: {' '.join(str(r) for r in args.roots)}",
                 f"% models={','.join(args.models)} drop_rank={args.drop_rank} "
                 f"caves_no_rank={args.caves_no_rank} bootstrap={args.bootstrap}"]
        for ds, t in tex.items():
            f = results.get(ds, {})
            for key, strata in [("caves", f.get("caves_strata", {}))] + \
                    [(m, v.get("strata", {})) for m, v in f.get("variants", {}).items()]:
                for sub, tag in (("forced", "Forced"), ("control", "Control")):
                    w = strata.get(sub)
                    if not w:
                        continue
                    lines.append(f"\\newcommand{{\\st{name[key]}{tag}{t}}}{{{w['auroc']:.3f}}}")
                    lines.append(f"\\newcommand{{\\st{name[key]}{tag}Lo{t}}}{{{w['auroc_ci'][0]:.3f}}}")
                    lines.append(f"\\newcommand{{\\st{name[key]}{tag}Hi{t}}}{{{w['auroc_ci'][1]:.3f}}}")
                    if key == "caves":
                        lines.append(f"\\newcommand{{\\stN{tag}{t}}}{{{w['n_pos']}}}")
        args.strata_tex.write_text("\n".join(lines) + "\n")
    if args.output:
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "caves_learned_weights.json").write_text(
            json.dumps({"features": features, "dropped": DEAD_FEATURES,
                        "label_source": args.label_source, "models": args.models,
                        "feature_selection": bool(args.select_features),
                        "folds": results}, indent=2, default=str)
        )
        print(f"\nwrote {args.output / 'caves_learned_weights.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
