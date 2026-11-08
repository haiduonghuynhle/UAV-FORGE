#!/usr/bin/env python3
"""Diagnostics for the CAVES reliability score: direction, transfer, and paired comparison.

Three questions, none of which the existing per-component table answers:

  DIRECTION   Does each component point the way its hand-written monotone rule
              assumes? A component whose univariate AUROC sits below 0.5 is not
              weak, it is reversed: the rule reads its evidence backwards.

  TRANSFER    When a rule is fitted on one dataset and scored on another, does
              it hold up? A model that scores below 0.5 off-domain has not
              merely lost accuracy, it has flipped.

  COMPARISON  Is scorer B better than scorer A? Both are read on the same
              frames, so their AUROCs are correlated and separate confidence
              intervals cannot be subtracted. The interval has to be built on
              the difference itself, and the resampling unit has to be a
              contiguous stretch of trajectory rather than a frame, because
              neighbouring UAV frames are not independent draws.
"""
from __future__ import annotations

import argparse
import hashlib
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
from caves_learned_weights import (  # noqa: E402
    DEAD_FEATURES,
    FEATURE_TO_COMPONENT,
    HANDSET_RISK_DIRECTION,
    LIVE_FEATURES,
    as_float,
    feature_vector,
    load,
    trajectory_blocks,
)

try:
    from sklearn.linear_model import LogisticRegression
except ImportError:  # pragma: no cover
    LogisticRegression = None


def auroc(labels: Sequence[int], scores: Sequence[float]) -> float:
    """Rank AUROC with ties averaged. 0.5 is chance; below 0.5 means reversed."""
    positives = [s for label, s in zip(labels, scores) if label == 1]
    negatives = [s for label, s in zip(labels, scores) if label == 0]
    if not positives or not negatives:
        return float("nan")
    marked = sorted([(s, 1) for s in positives] + [(s, 0) for s in negatives])
    total = 0.0
    i = 0
    while i < len(marked):
        j = i
        while j < len(marked) and marked[j][0] == marked[i][0]:
            j += 1
        average_rank = (i + j + 1) / 2.0
        total += sum(average_rank for k in range(i, j) if marked[k][1] == 1)
        i = j
    return (total - len(positives) * (len(positives) + 1) / 2.0) / (len(positives) * len(negatives))


def paired_delta_ci(
    labels: Sequence[int],
    scores_a: Sequence[float],
    scores_b: Sequence[float],
    blocks: List[List[int]],
    draws: int = 5000,
    seed: int = 0,
) -> Dict[str, float]:
    """Confidence interval for AUROC(B) - AUROC(A), built on the difference.

    Both scorers are recomputed on each resampled set, so the correlation
    between them is carried through instead of being thrown away by
    subtracting two independently built intervals.
    """
    if len(blocks) < 2:
        return {"delta": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"),
                "draws_valid": 0, "n_blocks": len(blocks)}
    point = auroc(labels, scores_b) - auroc(labels, scores_a)
    deltas: List[float] = []
    for draw in range(draws):
        picked: List[int] = []
        for slot in range(len(blocks)):
            key = f"{seed}|{draw}|{slot}".encode()
            picked.extend(blocks[int(hashlib.sha1(key).hexdigest()[:8], 16) % len(blocks)])
        y = [labels[i] for i in picked]
        if len(set(y)) < 2:
            continue
        delta = auroc(y, [scores_b[i] for i in picked]) - auroc(y, [scores_a[i] for i in picked])
        if delta == delta:
            deltas.append(delta)
    if not deltas:
        return {"delta": point, "ci_low": float("nan"), "ci_high": float("nan"),
                "draws_valid": 0, "n_blocks": len(blocks)}
    deltas.sort()
    return {
        "delta": point,
        "ci_low": deltas[max(0, int(0.025 * len(deltas)) - 1)],
        "ci_high": deltas[min(len(deltas) - 1, int(0.975 * len(deltas)))],
        "draws_valid": len(deltas),
        "n_blocks": len(blocks),
    }


# The monotone rules CAVES applies before weighting. They are the part of the
# hand design that does variable transformation rather than weighting, and a
# learner fed raw values does not have them.
CAVES_TRANSFORMS = {
    "inlier_ratio": lambda v: 1.0 - min(max(v, 0.0), 0.6) / 0.6,
    "num_inliers": lambda v: 1.0 - min(max(v, 0.0), 20.0) / 20.0,
    "reproj_residual_px": lambda v: min(max((v - 3.0) / 17.0, 0.0), 1.0),
    "shape_confidence": lambda v: 1.0 - min(max(v, 0.0), 1.0),
    "retrieval_score_margin": lambda v: 1.0 - min(max(v, 0.0), 0.25) / 0.25,
    "retrieval_rank": lambda v: min(max((max(1.0, v) - 1.0) / 4.0, 0.0), 1.0),
    "geometry_anchor_offset_m": lambda v: min(max((v - 20.0) / 130.0, 0.0), 1.0),
}

TRANSFORM_MODE = {"none": None, "caves": CAVES_TRANSFORMS}


def apply_transform(values: Sequence[float], features: Sequence[str], mode: str) -> List[float]:
    if mode == "none":
        return list(values)
    if mode == "log":
        return [math.log1p(max(0.0, v)) for v in values]
    return [CAVES_TRANSFORMS.get(f, lambda x: x)(v) for f, v in zip(features, values)]


TRANSFORM = {"mode": "none"}


def matrix_of(rows: List[Dict[str, Any]], features: Sequence[str]):
    xs, ys, kept = [], [], []
    for row in rows:
        vec = feature_vector(row, features)
        if vec is None:
            continue
        xs.append(apply_transform(vec, features, TRANSFORM["mode"]))
        ys.append(int(row["_caves_label"]))
        kept.append(row)
    return np.array(xs, dtype=np.float64), np.array(ys, dtype=np.int32), kept


def fit_and_score(train: List[Dict[str, Any]], test: List[Dict[str, Any]],
                  features: Sequence[str]) -> Optional[float]:
    x_tr, y_tr, _ = matrix_of(train, features)
    x_te, y_te, _ = matrix_of(test, features)
    if len(set(y_tr.tolist())) < 2 or len(set(y_te.tolist())) < 2 or len(y_tr) < 20:
        return None
    mean, scale = x_tr.mean(axis=0), x_tr.std(axis=0)
    scale = np.where(scale == 0, 1.0, scale)
    model = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs")
    model.fit((x_tr - mean) / scale, y_tr)
    return auroc(y_te.tolist(), model.predict_proba((x_te - mean) / scale)[:, 1].tolist())


def report_direction(by_dataset: Dict[str, List[Dict[str, Any]]], features: Sequence[str]) -> List[str]:
    lines = ["DIRECTION - univariate AUROC of each component, per dataset",
             "(oriented the way the hand-written rule assumes; below 0.5 means reversed)", ""]
    header = f"{'component':<30}" + "".join(f"{d:>14}" for d in sorted(by_dataset))
    lines += [header, "-" * len(header)]
    for feature in features:
        cells = []
        for dataset in sorted(by_dataset):
            rows = by_dataset[dataset]
            pairs = [(int(r["_caves_label"]), feature_vector(r, [feature])) for r in rows]
            pairs = [(y, v[0]) for y, v in pairs if v is not None]
            if not pairs:
                cells.append("     n/a")
                continue
            sign = HANDSET_RISK_DIRECTION.get(feature, 1)
            value = auroc([p[0] for p in pairs], [sign * p[1] for p in pairs])
            flag = "*" if value == value and value < 0.5 else " "
            cells.append(f"{value:>13.3f}{flag}")
        lines.append(f"{FEATURE_TO_COMPONENT.get(feature, feature):<30}" + "".join(cells))
    lines += ["", "  * below chance once oriented as the rule assumes: the direction is",
              "    empirically reversed on that dataset, not merely uninformative."]
    return lines


def report_transfer(by_dataset: Dict[str, List[Dict[str, Any]]], features: Sequence[str]) -> List[str]:
    lines = ["", "TRANSFER - fit on the row dataset, score on the column dataset",
             "(diagonal is in-domain and optimistic; read the off-diagonal)", ""]
    names = sorted(by_dataset)
    header = f"{'fitted on':<14}" + "".join(f"{d:>14}" for d in names)
    lines += [header, "-" * len(header)]
    for source in names:
        cells = []
        for target in names:
            value = fit_and_score(by_dataset[source], by_dataset[target], features)
            cells.append("           n/a" if value is None else f"{value:>14.3f}")
        lines.append(f"{source:<14}" + "".join(cells))
    return lines


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--roots", nargs="+", required=True)
    parser.add_argument("--label-source", default="clean", choices=["clean", "forge"])
    parser.add_argument("--all-frames", action="store_true")
    parser.add_argument("--compare", nargs=2, metavar=("A_FEATURES", "B_FEATURES"),
                        help="comma-separated feature lists; B is compared against A "
                             "leave-one-dataset-out. Use the literal 'caves' for the "
                             "hand-set score.")
    parser.add_argument("--transform", default="none", choices=["none", "caves", "log"],
                        help="apply CAVES's own monotone rules (or a log) before fitting, so "
                             "learned and hand-set weights are compared on the same footing")
    parser.add_argument("--draws", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--caves-no-rank", action="store_true",
                        help="rescore the hand-set CAVES baseline without its rank component")
    args = parser.parse_args()

    if LogisticRegression is None:
        raise SystemExit("scikit-learn is required")
    TRANSFORM["mode"] = args.transform

    rows = load([Path(r) for r in args.roots], label_source=args.label_source,
                accepted_only=not args.all_frames)
    if not rows:
        raise SystemExit("no labelled rows under the given roots")

    by_dataset: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        by_dataset.setdefault(str(row.get("dataset", "unknown")), []).append(row)

    features = [f for f in LIVE_FEATURES]
    out_lines = report_direction(by_dataset, features)
    out_lines += report_transfer(by_dataset, features)

    payload: Dict[str, Any] = {"label_source": args.label_source,
                               "all_frames": bool(args.all_frames),
                               "dropped": DEAD_FEATURES,
                               "datasets": {d: len(r) for d, r in by_dataset.items()}}

    if args.compare:
        a_spec, b_spec = args.compare
        out_lines += ["", "COMPARISON - AUROC(B) - AUROC(A) on the same frames",
                      f"  A = {a_spec}", f"  B = {b_spec}",
                      "  paired block bootstrap over contiguous within-flight frame blocks", ""]
        header = f"{'held-out dataset':<16}{'n':>7}{'AUROC A':>10}{'AUROC B':>10}{'delta':>9}{'95% CI of delta':>22}"
        out_lines += [header, "-" * len(header)]
        comparisons = {}
        for held in sorted(by_dataset):
            train = [r for d, rs in by_dataset.items() if d != held for r in rs]
            test = by_dataset[held]

            def scores_for(spec: str) -> Tuple[Optional[List[float]], List[Dict[str, Any]]]:
                if spec == "caves":
                    if args.caves_no_rank:
                        from forge_mixed_confusion import caves_risk
                        scored = [(caves_risk(r, True), r) for r in test]
                        kept = [r for s_, r in scored if s_ is not None]
                        return [s_ for s_, r in scored if s_ is not None], kept
                    kept = [r for r in test if as_float(r.get("caves_risk_score")) is not None]
                    return [as_float(r.get("caves_risk_score")) for r in kept], kept
                cols = [f.strip() for f in spec.split(",") if f.strip()]
                x_tr, y_tr, _ = matrix_of(train, cols)
                x_te, _, kept = matrix_of(test, cols)
                if len(set(y_tr.tolist())) < 2 or not len(kept):
                    return None, kept
                mean, scale = x_tr.mean(axis=0), x_tr.std(axis=0)
                scale = np.where(scale == 0, 1.0, scale)
                model = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs")
                model.fit((x_tr - mean) / scale, y_tr)
                return model.predict_proba((x_te - mean) / scale)[:, 1].tolist(), kept

            # Both scorers must be read on the same frames or the difference is
            # not a paired quantity: intersect on the rows both can score.
            needed = set()
            for spec in (a_spec, b_spec):
                if spec != "caves":
                    needed.update(f.strip() for f in spec.split(",") if f.strip())
            common = [r for r in test
                      if (not needed or feature_vector(r, sorted(needed)) is not None)
                      and as_float(r.get("caves_risk_score")) is not None]
            if len(common) < 20:
                out_lines.append(f"{held:<16}{len(common):>7}  too few rows both scorers can read")
                continue
            saved, by_dataset[held] = by_dataset[held], common
            test = common
            a_scores, _ = scores_for(a_spec)
            b_scores, _ = scores_for(b_spec)
            by_dataset[held] = saved
            if a_scores is None or b_scores is None:
                out_lines.append(f"{held:<16}{len(common):>7}  degenerate training fold")
                continue
            labels = [int(r["_caves_label"]) for r in common]
            stats = paired_delta_ci(labels, a_scores, b_scores,
                                    trajectory_blocks(common), draws=args.draws, seed=args.seed)
            a_value, b_value = auroc(labels, a_scores), auroc(labels, b_scores)
            interval = f"[{stats['ci_low']:.3f}, {stats['ci_high']:.3f}]"
            out_lines.append(
                f"{held:<16}{len(common):>7}{a_value:>10.3f}{b_value:>10.3f}"
                f"{stats['delta']:>9.3f}{interval:>22}"
            )
            comparisons[held] = {"auroc_a": a_value, "auroc_b": b_value, **stats}
        out_lines += ["", "  A CI that excludes zero is evidence the two scorers differ.",
                      f"  Blocks per dataset: " +
                      ", ".join(f"{d}={comparisons[d]['n_blocks']}" for d in comparisons)]
        payload["comparison"] = {"a": a_spec, "b": b_spec, "folds": comparisons}

    print("\n".join(out_lines))
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2, default=str))
        print(f"\nwrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
