#!/usr/bin/env python3
"""Confusion matrix for a mixed-rate FORGE run.

Reads one or more run directories produced with adversarial_inject_rate < 1 and
reports, per run:

  Table 1  the acceptance decision the system actually makes
           (accepted_for_metrics) against the arm each frame was assigned to,
           analysed intention-to-treat.
  Table 2  CAVES treated as a post-hoc ranker, reported twice: with the
           retrieval-rank component and without it.

The two tables answer different questions and must not be merged. CAVES is
computed after acceptance and never changes it, so nothing in Table 2 is a
claim about the deployed system.

Both arms come from one run, so a frame appears in exactly one arm and the two
arms are independent groups, not pairs: the comparison is a two-proportion one
and the intervals come from a block bootstrap over contiguous within-flight
segments, because neighbouring UAV frames are correlated and frame-level
intervals overstate precision.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

RANK_FIELDS = ("retrieval_rank", "candidate_rank_selected")


def reference_is_correct(row: Dict[str, str], dataset: str) -> Optional[bool]:
    """Does the reference the system actually used share ground with the query?

    Assignment says which arm a frame was put in; this says what reference it
    ended up with. They differ in both directions: an assigned frame whose
    shortlist held no eligible wrong tile keeps a correct one, and a control
    frame can still retrieve a wrong tile on its own. Reported as a
    stratification of the arms, never as a replacement for them, because it is
    measured after treatment.
    """
    import gen_v3_tables as gen

    try:
        gt_lat, gt_lon = float(row["gt_lat"]), float(row["gt_lon"])
    except (KeyError, ValueError, TypeError):
        return None
    anchor = str(row.get("geodetic_anchor", "") or "")
    if not anchor.startswith("{"):
        return None
    try:
        parsed = json.loads(anchor.replace("'", '"'))
        return bool(gen.selected_reference_can_localize(
            gt_lat, gt_lon, float(parsed["lat"]), float(parsed["lon"]), dataset, row))
    except (ValueError, KeyError, TypeError):
        return None


def truthy(value) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def as_float(value) -> Optional[float]:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def load_rows(run_dir: Path) -> List[Dict[str, str]]:
    paths = sorted(run_dir.rglob("predictions.csv"))
    if not paths:
        raise SystemExit(f"no predictions.csv under {run_dir}")
    if len(paths) > 1:
        # A cell must hold exactly one timestamped run; partial retries are
        # moved aside by the launcher. Concatenating two runs would double
        # count every frame and shrink every interval.
        raise SystemExit(f"{len(paths)} predictions.csv under {run_dir}; pin one run")
    rows: List[Dict[str, str]] = []
    for path in paths:
        with path.open(newline="") as handle:
            rows.extend(csv.DictReader(handle))
    return rows


def dataset_of_run(run_dir: Path, rows: List[Dict[str, str]]) -> str:
    """Dataset key for the footprint predicate, from run_manifest.json.

    predictions.csv carries an empty `dataset` column, and an empty key makes
    the predicate fall back to 15 m half-extents, which labels most correct
    references wrong. Refuse to guess."""
    import json as _json
    for mf in sorted(run_dir.rglob("run_manifest.json")):
        ds = str(_json.loads(mf.read_text()).get("dataset", "") or "").lower()
        if ds:
            break
    else:
        ds = ""
    if not ds:
        for key in ("uav_visloc", "vpair", "avl"):
            if key in run_dir.name.lower():
                ds = key
                break
    if ds not in {"avl", "vpair", "uav_visloc"}:
        raise SystemExit(f"cannot determine dataset for {run_dir}: {ds!r}")
    return ds


def flight_of(row: Dict[str, str]) -> str:
    flight = str(row.get("query_flight_id", "") or "").strip()
    if flight:
        return flight
    name = str(row.get("image_name", "") or "")
    return name.split("/")[0] if "/" in name else ""


def block_keys(rows: List[Dict[str, str]], target_blocks: int = 25) -> Dict[int, str]:
    """Assign each row a contiguous-trajectory block, keyed by row identity.

    VPAIR and AVL carry no flight id at all and UAV-VisLoc carries three, so
    resampling flights is either impossible or far too coarse. Neighbouring
    frames along a route are strongly correlated, which is what a block
    bootstrap is for: blocks are contiguous runs of frames within a flight, and
    the bootstrap resamples blocks rather than frames.
    """
    ordered: Dict[str, List[Dict[str, str]]] = {}
    for row in rows:
        ordered.setdefault(flight_of(row), []).append(row)
    keys: Dict[int, str] = {}
    for flight, group in ordered.items():
        group = sorted(group, key=lambda r: str(r.get("image_name", "")))
        per_flight = max(1, target_blocks // max(1, len(ordered)))
        size = max(1, math.ceil(len(group) / per_flight))
        for position, row in enumerate(group):
            keys[id(row)] = f"{flight}#{position // size}"
    return keys


def wilson(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def auroc(pos: Sequence[float], neg: Sequence[float]) -> float:
    """Rank AUROC with ties averaged."""
    n1, n0 = len(pos), len(neg)
    if n1 == 0 or n0 == 0:
        return float("nan")
    marked = sorted([(v, 1) for v in pos] + [(v, 0) for v in neg])
    total = 0.0
    i = 0
    while i < len(marked):
        j = i
        while j < len(marked) and marked[j][0] == marked[i][0]:
            j += 1
        avg_rank = (i + j + 1) / 2.0
        total += sum(avg_rank for k in range(i, j) if marked[k][1] == 1)
        i = j
    return (total - n1 * (n1 + 1) / 2.0) / (n1 * n0)


def cluster_bootstrap(
    rows: List[Dict[str, str]],
    statistic,
    draws: int = 2000,
    seed: int = 0,
    target_blocks: int = 25,
) -> Tuple[float, float]:
    """Resample contiguous trajectory blocks, not frames.

    `target_blocks` sets how many blocks the rows are cut into, so halving or
    doubling it halves or doubles the block length. It exists for the
    sensitivity check; every reported interval uses the default.
    """
    keys = block_keys(rows, target_blocks=target_blocks)
    by_block: Dict[str, List[Dict[str, str]]] = {}
    for row in rows:
        by_block.setdefault(keys[id(row)], []).append(row)
    blocks = sorted(by_block)
    if len(blocks) < 2:
        return (float("nan"), float("nan"))
    values: List[float] = []
    for draw in range(draws):
        # Deterministic resample: hash the draw index rather than an RNG so the
        # interval is reproducible without carrying seed state around.
        picked: List[Dict[str, str]] = []
        for slot in range(len(blocks)):
            key = f"{seed}|{draw}|{slot}".encode()
            idx = int(hashlib.sha1(key).hexdigest()[:8], 16) % len(blocks)
            picked.extend(by_block[blocks[idx]])
        value = statistic(picked)
        if value == value:
            values.append(value)
    if not values:
        return (float("nan"), float("nan"))
    values.sort()
    lo = values[max(0, int(0.025 * len(values)) - 1)]
    hi = values[min(len(values) - 1, int(0.975 * len(values)))]
    return (lo, hi)


def caves_risk(row: Dict[str, str], drop_rank: bool) -> Optional[float]:
    """Recompute CAVES so the rank component can be removed.

    The injector must skip the originally selected candidate, so retrieval rank
    differs between arms by construction and marks the treatment rather than
    evidencing it. Reporting the score both ways separates the two.
    """
    from uav_geolocalization.benchmark.caves import compute_caves_v0

    payload = dict(row)
    if drop_rank:
        for field in RANK_FIELDS:
            payload[field] = ""
    return as_float(compute_caves_v0(payload).risk_score)


def accept_rate(rows: List[Dict[str, str]]) -> float:
    if not rows:
        return float("nan")
    return sum(1 for r in rows if truthy(r.get("accepted_for_metrics"))) / len(rows)


def analyse(run_dir: Path, draws: int) -> Dict[str, object]:
    rows = load_rows(run_dir)
    if not rows or "adversarial_assigned_attack" not in rows[0]:
        raise SystemExit(
            f"{run_dir}: no adversarial_assigned_attack column; this run predates "
            "mixed-rate injection and has no control arm to compare against"
        )

    attacked = [r for r in rows if truthy(r.get("adversarial_assigned_attack"))]
    control = [r for r in rows if not truthy(r.get("adversarial_assigned_attack"))]

    # Intention-to-treat: a frame assigned to the attacked arm whose shortlist
    # held no eligible wrong reference stays in the denominator. Dropping it
    # would reward the system for being unattackable on exactly the frames
    # where injection was hardest.
    treated = [r for r in attacked if truthy(r.get("adversarial_attacked"))]

    acc_att = sum(1 for r in attacked if truthy(r.get("accepted_for_metrics")))
    acc_ctl = sum(1 for r in control if truthy(r.get("accepted_for_metrics")))

    n_att, n_ctl = len(attacked), len(control)
    tpr = acc_ctl / n_ctl if n_ctl else float("nan")
    leak = acc_att / n_att if n_att else float("nan")

    def diff(sample: List[Dict[str, str]]) -> float:
        a = [r for r in sample if truthy(r.get("adversarial_assigned_attack"))]
        c = [r for r in sample if not truthy(r.get("adversarial_assigned_attack"))]
        if not a or not c:
            return float("nan")
        return accept_rate(c) - accept_rate(a)

    result: Dict[str, object] = {
        "run_dir": str(run_dir),
        "condition": rows[0].get("adversarial_condition", ""),
        "inject_rate": rows[0].get("adversarial_inject_rate", ""),
        "n_total": len(rows),
        "n_attacked_assigned": n_att,
        "n_attacked_treated": len(treated),
        "n_control": n_ctl,
        "control_accepted": acc_ctl,
        "control_rejected": n_ctl - acc_ctl,
        "attacked_accepted": acc_att,
        "attacked_rejected": n_att - acc_att,
        "tpr_control_accept": tpr,
        "tpr_ci": wilson(acc_ctl, n_ctl),
        "leak_attacked_accept": leak,
        "leak_ci": wilson(acc_att, n_att),
        "balanced_accuracy": (tpr + (1 - leak)) / 2 if n_att and n_ctl else float("nan"),
        "accept_rate_difference": tpr - leak if n_att and n_ctl else float("nan"),
        "accept_rate_difference_cluster_ci": cluster_bootstrap(rows, diff, draws=draws),
        "n_blocks": len(set(block_keys(rows).values())),
    }

    dataset = dataset_of_run(run_dir, rows)
    strata: Dict[str, Dict[str, int]] = {}
    for arm_name, arm in (("control", control), ("attacked", attacked)):
        buckets = {"correct_ref_accept": 0, "correct_ref_reject": 0,
                   "wrong_ref_accept": 0, "wrong_ref_reject": 0, "unknown": 0,
                   "fail_closed": 0}
        for row in arm:
            # An assigned frame the injector could not attack has no reference
            # at all (fail-closed, rejected mechanically); classifying it by the
            # reference retrieval would have used misfiles it into a column the
            # gate never judged.
            if arm_name == "attacked" and str(row.get("adversarial_failure_reason") or "").strip():
                buckets["fail_closed"] += 1
                continue
            ok = reference_is_correct(row, dataset)
            if ok is None:
                buckets["unknown"] += 1
                continue
            key = ("correct_ref_" if ok else "wrong_ref_") + (
                "accept" if truthy(row.get("accepted_for_metrics")) else "reject")
            buckets[key] += 1
        strata[arm_name] = buckets
    result["reference_strata"] = strata

    for label, drop in (("caves_auroc_with_rank", False), ("caves_auroc_without_rank", True)):
        pos = [v for v in (caves_risk(r, drop) for r in attacked) if v is not None]
        neg = [v for v in (caves_risk(r, drop) for r in control) if v is not None]
        result[label] = auroc(pos, neg)
    return result


def render(results: List[Dict[str, object]]) -> str:
    lines: List[str] = []
    lines.append("TABLE 1 - the acceptance decision the system actually makes")
    lines.append("(accepted_for_metrics, intention-to-treat, control = correct reference)")
    lines.append("")
    header = (
        f"{'condition':<24}{'rate':>6}{'n_ctl':>7}{'n_att':>7}"
        f"{'ctl acc/rej':>14}{'att acc/rej':>14}{'TPR':>8}{'leak':>8}{'bal-acc':>9}"
    )
    lines.append(header)
    lines.append("-" * len(header))
    for r in results:
        lines.append(
            f"{str(r['condition']):<24}{str(r['inject_rate']):>6}"
            f"{r['n_control']:>7}{r['n_attacked_assigned']:>7}"
            f"{str(r['control_accepted']) + '/' + str(r['control_rejected']):>14}"
            f"{str(r['attacked_accepted']) + '/' + str(r['attacked_rejected']):>14}"
            f"{r['tpr_control_accept']:>8.3f}{r['leak_attacked_accept']:>8.3f}"
            f"{r['balanced_accuracy']:>9.3f}"
        )
    lines.append("")
    for r in results:
        lo, hi = r["accept_rate_difference_cluster_ci"]
        tlo, thi = r["tpr_ci"]
        llo, lhi = r["leak_ci"]
        lines.append(
            f"  {r['condition']}: TPR {r['tpr_control_accept']:.3f} [{tlo:.3f}, {thi:.3f}] "
            f"| leak {r['leak_attacked_accept']:.3f} [{llo:.3f}, {lhi:.3f}] "
            f"| difference {r['accept_rate_difference']:.3f} "
            f"[{lo:.3f}, {hi:.3f}] cluster CI over {r['n_blocks']} trajectory blocks"
        )
        if r["n_attacked_treated"] != r["n_attacked_assigned"]:
            missed = r["n_attacked_assigned"] - r["n_attacked_treated"]
            lines.append(
                f"    {missed} of {r['n_attacked_assigned']} assigned frames had no eligible "
                "wrong reference; the injector fails closed (no reference image), the"
                " pipeline rejects them mechanically, and they stay in the ITT denominator"
            )
    lines.append("")
    lines.append("TABLE 1b - the same frames split by the reference they actually got")
    lines.append("(post-treatment stratification: descriptive, it does not replace Table 1)")
    lines.append("")
    head1b = f"{'condition':<24}{'arm':<10}{'correct ref acc/rej':>22}{'wrong ref acc/rej':>21}{'unknown':>9}"
    lines.append(head1b)
    lines.append("-" * len(head1b))
    for r in results:
        for arm in ("control", "attacked"):
            b = r["reference_strata"][arm]
            lines.append(
                f"{str(r['condition']) if arm == 'control' else '':<24}{arm:<10}"
                f"{str(b['correct_ref_accept']) + '/' + str(b['correct_ref_reject']):>22}"
                f"{str(b['wrong_ref_accept']) + '/' + str(b['wrong_ref_reject']):>21}"
                f"{b['unknown']:>9}"
            )
    lines.append("")
    lines.append("TABLE 2 - CAVES as a post-hoc ranker, NOT the acceptance decision")
    lines.append("(CAVES is computed after acceptance and never changes it)")
    lines.append("")
    header2 = f"{'condition':<24}{'AUROC with rank':>18}{'AUROC without rank':>21}"
    lines.append(header2)
    lines.append("-" * len(header2))
    for r in results:
        lines.append(
            f"{str(r['condition']):<24}{r['caves_auroc_with_rank']:>18.3f}"
            f"{r['caves_auroc_without_rank']:>21.3f}"
        )
    lines.append("")
    lines.append(
        "  Retrieval rank differs between arms by construction: injection must skip the"
    )
    lines.append(
        "  originally selected candidate. The without-rank column is the one that measures"
    )
    lines.append("  evidence rather than the injection procedure.")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dirs", nargs="+", type=Path)
    parser.add_argument("--src", type=Path, default=Path("src"), help="path holding uav_geolocalization")
    parser.add_argument("--draws", type=int, default=2000)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    sys.path.insert(0, str(args.src.resolve()))
    results = [analyse(d, args.draws) for d in args.run_dirs]
    print(render(results))
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(results, indent=2, default=str))
        print(f"\nwrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
