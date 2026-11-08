#!/usr/bin/env python3
"""Emit tables/postsub_numbers.tex: every post-submission number the book cites.

One generator, one macro file, one source of truth. The thesis chapters and the
slide deck must quote the same campaign artifacts, so every quantity below is
computed from the recorded JSON/CSV outputs rather than typed by hand. Runs on
the server (needs the results tree and scripts/reporting on the path).

Blocks:
  A  mixed-protocol confusion summary  (analysis_{vpair,avl,visloc}.json)
  B  condition-overlap shares          (predictions.csv of the nine cells)
  C  leave-one-dataset-out scorers     (caves_learned_mlp/caves_learned_weights.json)
  D  paired AUROC deltas               (paired_delta.json from caves_diagnostics)
  E  direction audit                   (recomputed, oriented univariate AUROC)
  F  counterfactual gate replay        (gate_replay.json)
  G  QC boundary count for the cited UAV-VisLoc FORGE run (forge_cell_qc output)
  H  environment drift: 100%-injection reference cell rerun in the current
     environment vs the cell of record, same frames (ref100_* vs --ref-e1)
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
from datetime import date
from pathlib import Path
from typing import Any, Dict, List

_REPORTING_ROOT = Path(__file__).resolve().parent
if str(_REPORTING_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPORTING_ROOT))

DATASET_KEY = {"vpair": "Vpair", "avl": "Avl", "visloc": "Visloc", "uav_visloc": "Visloc"}
COND_KEY = {"wrong_near": "Near", "wrong_far": "Far", "visually_similar_wrong": "Sim"}


def pct(x: float, digits: int = 1) -> str:
    return f"{100.0 * x:.{digits}f}\\%"


def one_run(cell: Path, name: str = "predictions.csv") -> Path:
    """The single timestamped artifact of a cell; 0 or >1 is a pinning error."""
    hits = sorted(cell.rglob(name))
    if len(hits) != 1:
        raise SystemExit(f"{len(hits)} {name} under {cell}; exactly one run per cell is required")
    return hits[0]


def load_analyses(root: Path) -> Dict[str, List[Dict[str, Any]]]:
    out = {}
    for ds in ("vpair", "avl", "visloc"):
        out[ds] = json.loads((root / f"analysis_{ds}.json").read_text())
    return out


def block_mixed(analyses: Dict[str, List[Dict[str, Any]]]) -> List[str]:
    lines = ["% --- A. mixed-protocol summary (per-protocol strata; control arm is the"]
    lines += ["%     correct-reference column, fail-closed assigned frames excluded and"]
    lines += ["%     counted separately). Same math as gen_mixed_confusion_table.py."]
    bals, leaks = [], []
    fail_closed = 0
    fail_closed_ds: Dict[str, int] = {}
    for ds, entries in analyses.items():
        for e in entries:
            s = e["reference_strata"]
            control, attacked = s["control"], s["attacked"]
            ca, cr = control["correct_ref_accept"], control["correct_ref_reject"]
            wa = control["wrong_ref_accept"] + attacked["wrong_ref_accept"]
            wr = control["wrong_ref_reject"] + attacked["wrong_ref_reject"]
            fail_closed += attacked["fail_closed"]
            fail_closed_ds[ds] = fail_closed_ds.get(ds, 0) + attacked["fail_closed"]
            if not (ca + cr) or not (wa + wr):
                continue
            tpr, leak = ca / (ca + cr), wa / (wa + wr)
            bals.append((tpr + 1 - leak) / 2)
            leaks.append(leak)
            key = DATASET_KEY[ds] + COND_KEY[e["condition"]]
            lines.append(f"\\newcommand{{\\psMixTpr{key}}}{{{pct(tpr)}}}")
            lines.append(f"\\newcommand{{\\psMixLeak{key}}}{{{pct(leak)}}}")
    vp = analyses["vpair"][0]["reference_strata"]
    ca = vp["control"]["correct_ref_accept"]
    cr = vp["control"]["correct_ref_reject"]
    lines += [
        f"\\newcommand{{\\psMixVpairCorrectN}}{{{ca + cr}}}",
        f"\\newcommand{{\\psMixFailClosed}}{{{fail_closed}}}",
        *[f"\\newcommand{{\\psMixFailClosed{DATASET_KEY[ds]}}}{{{v}}}" for ds, v in fail_closed_ds.items()],
        f"\\newcommand{{\\psMixBalLo}}{{{min(bals):.2f}}}",
        f"\\newcommand{{\\psMixBalHi}}{{{max(bals):.2f}}}",
        f"\\newcommand{{\\psMixLeakLo}}{{{pct(min(leaks))}}}",
        f"\\newcommand{{\\psMixLeakHi}}{{{pct(max(leaks))}}}",
        "\\newcommand{\\psMixChance}{0.50}",
    ]
    # Pooled 2x2 per dataset (sum of the three condition cells), for the
    # confusion-matrix figure. Same per-protocol strata as the rows above.
    for ds, entries in analyses.items():
        pooled = {"CA": 0, "CR": 0, "WA": 0, "WR": 0}
        for e in entries:
            s = e["reference_strata"]
            pooled["CA"] += s["control"]["correct_ref_accept"]
            pooled["CR"] += s["control"]["correct_ref_reject"]
            pooled["WA"] += s["control"]["wrong_ref_accept"] + s["attacked"]["wrong_ref_accept"]
            pooled["WR"] += s["control"]["wrong_ref_reject"] + s["attacked"]["wrong_ref_reject"]
        key = DATASET_KEY[ds]
        for cell, value in pooled.items():
            lines.append(f"\\newcommand{{\\psMixPool{key}{cell}}}{{{value}}}")
        if not (pooled["CA"] + pooled["CR"]) or not (pooled["WA"] + pooled["WR"]):
            raise SystemExit(f"pooled 2x2 for {ds} has an empty column: {pooled}")
        tpr = pooled["CA"] / (pooled["CA"] + pooled["CR"])
        leak = pooled["WA"] / (pooled["WA"] + pooled["WR"])
        lines.append(f"\\newcommand{{\\psMixPool{key}Tpr}}{{{pct(tpr)}}}")
        lines.append(f"\\newcommand{{\\psMixPool{key}Leak}}{{{pct(leak)}}}")
    # Injected-arm reading: the wrong column holds only the forced wrong
    # references (unambiguous faults), denominator = treated frames, as in the
    # full-coverage campaign; the correct column is the control arm's
    # predicate-correct references. Natural wrong retrievals of the control arm
    # are reported separately (their label is the predicate's, which UAV-VisLoc
    # refutes).
    for ds, entries in analyses.items():
        key = DATASET_KEY[ds]
        pool = {"CA": 0, "CR": 0, "WA": 0, "WR": 0, "NatWA": 0, "NatWR": 0, "Treated": 0, "Assigned": 0}
        for e in entries:
            s = e["reference_strata"]; ck = DATASET_KEY[ds] + COND_KEY[e["condition"]]
            ca, cr = s["control"]["correct_ref_accept"], s["control"]["correct_ref_reject"]
            wa, wr = s["attacked"]["wrong_ref_accept"], s["attacked"]["wrong_ref_reject"]
            if s["attacked"]["correct_ref_accept"] + s["attacked"]["correct_ref_reject"]:
                raise SystemExit(f"{ds}/{e['condition']}: treated frames with a correct reference")
            if not (wa + wr) or not (ca + cr):
                raise SystemExit(f"{ds}/{e['condition']}: empty injected or control column")
            lines.append(f"\\newcommand{{\\psMixInjAcc{ck}}}{{{wa}}}")
            lines.append(f"\\newcommand{{\\psMixInjN{ck}}}{{{wa + wr}}}")
            lines.append(f"\\newcommand{{\\psMixInjLeak{ck}}}{{{pct(wa / (wa + wr))}}}")
            lines.append(f"\\newcommand{{\\psMixInjBal{ck}}}{{{(ca / (ca + cr) + 1 - wa / (wa + wr)) / 2:.2f}}}")
            lines.append(f"\\newcommand{{\\psMixAssigned{ck}}}{{{e['n_attacked_assigned']}}}")
            for k, v in (("CA", ca), ("CR", cr), ("WA", wa), ("WR", wr),
                         ("NatWA", s["control"]["wrong_ref_accept"]), ("NatWR", s["control"]["wrong_ref_reject"]),
                         ("Treated", wa + wr), ("Assigned", e["n_attacked_assigned"])):
                pool[k] += v
        for k, v in pool.items():
            lines.append(f"\\newcommand{{\\psMixInj{key}{k}}}{{{v}}}")
        tpr = pool["CA"] / (pool["CA"] + pool["CR"]); leak = pool["WA"] / (pool["WA"] + pool["WR"])
        lines.append(f"\\newcommand{{\\psMixInj{key}Tpr}}{{{pct(tpr)}}}")
        lines.append(f"\\newcommand{{\\psMixInj{key}Leak}}{{{pct(leak)}}}")
        lines.append(f"\\newcommand{{\\psMixInj{key}Bal}}{{{(tpr + 1 - leak) / 2:.2f}}}")
        if pool["NatWA"] + pool["NatWR"]:
            lines.append(f"\\newcommand{{\\psMixInj{key}NatWrongAcc}}{{{pct(pool['NatWA'] / (pool['NatWA'] + pool['NatWR']))}}}")
    # ITT reading: accept-rate difference between assigned arms, cluster CI.
    for ds, entries in analyses.items():
        for e in entries:
            key = DATASET_KEY[ds] + COND_KEY[e["condition"]]
            lo, hi = e["accept_rate_difference_cluster_ci"]
            lines.append(
                f"\\newcommand{{\\psIttDiff{key}}}{{{e['accept_rate_difference']:.3f}}}"
            )
            lines.append(f"\\newcommand{{\\psIttDiffLo{key}}}{{{float(lo):.3f}}}")
            lines.append(f"\\newcommand{{\\psIttDiffHi{key}}}{{{float(hi):.3f}}}")
    return lines


def block_overlap(campaign_root: Path) -> List[str]:
    lines = ["% --- B. share of injected frames on which all three conditions forced the"]
    lines += ["%     same tile (identity of adversarial_forced_reference_name)."]
    name_map = {"vpair": "vpair", "avl": "avl", "visloc": "uav_visloc"}
    for short, ds in name_map.items():
        per_cond = {}
        for cond in ("wrong_near", "wrong_far", "visually_similar_wrong"):
            rows = list(csv.DictReader(open(one_run(campaign_root / f"{ds}_{cond}"))))
            per_cond[cond] = {
                r["image_name"]: r["adversarial_forced_reference_name"]
                for r in rows
                if str(r.get("adversarial_attacked", "")).lower() == "true"
            }
        keys = set.intersection(*(set(v) for v in per_cond.values()))
        same = sum(
            1 for k in keys
            if len({per_cond[c][k] for c in per_cond}) == 1
        )
        lines.append(
            f"\\newcommand{{\\psOverlap{DATASET_KEY[short]}}}"
            f"{{{pct(same / len(keys), 0)}}}"
        )
    return lines


def block_lodo(mlp_json: Path) -> List[str]:
    data = json.loads(mlp_json.read_text())
    lines = ["% --- C. leave-one-dataset-out scorers, accepted observations of the mixed"]
    lines += ["%     campaign, label = final error beyond the dataset radius."]
    for fold, res in data["folds"].items():
        key = DATASET_KEY[fold]
        lines.append(f"\\newcommand{{\\psLodoN{key}}}{{{res['n_test']}}}")
        lines.append(f"\\newcommand{{\\psLodoBad{key}}}{{{res['n_bad_test']}}}")
        lines.append(f"\\newcommand{{\\psLodoCaves{key}}}{{{res['caves_auroc']:.3f}}}")
        for model, macro in (("lr_l2", "Lr"), ("gb", "Gb"), ("mlp", "Mlp")):
            lines.append(
                f"\\newcommand{{\\psLodo{macro}{key}}}"
                f"{{{res['variants'][model]['auroc']:.3f}}}"
            )
    return lines


def block_delta(delta_json: Path) -> List[str]:
    data = json.loads(delta_json.read_text())
    lines = ["% --- D. paired AUROC difference (learned logistic minus CAVES v0), moving-"]
    lines += ["%     block bootstrap over contiguous trajectory blocks, all frames."]
    for fold, res in data["comparison"]["folds"].items():
        key = DATASET_KEY[fold]
        lines.append(f"\\newcommand{{\\psDelta{key}}}{{{res['delta']:.3f}}}")
        lines.append(f"\\newcommand{{\\psDeltaLo{key}}}{{{res['ci_low']:.3f}}}")
        lines.append(f"\\newcommand{{\\psDeltaHi{key}}}{{{res['ci_high']:.3f}}}")
    return lines


def block_direction(campaign_root: Path) -> List[str]:
    from caves_learned_weights import HANDSET_RISK_DIRECTION, feature_vector, load
    from caves_diagnostics import auroc

    # The nine mixed cells by name: a glob on *_wrong_near would also pick up
    # the 100%-injection reference cell (ref100_vpair_wrong_near).
    roots = [campaign_root / f"{ds}_{cond}"
             for ds in ("vpair", "avl", "uav_visloc")
             for cond in ("wrong_near", "wrong_far", "visually_similar_wrong")]
    missing = [r for r in roots if not r.exists()]
    if missing:
        raise SystemExit(f"missing mixed cells: {missing}")
    rows = load(roots, label_source="clean", accepted_only=False)
    by: Dict[str, list] = {}
    for r in rows:
        by.setdefault(str(r.get("dataset", "")), []).append(r)
    lines = ["% --- E. direction audit: univariate AUROC oriented the way the hand rule"]
    lines += ["%     assumes; below one half means the assumed direction is reversed."]
    wanted = {
        "inlier_ratio": "InlierRatio",
        "num_inliers": "InlierCount",
        "reproj_residual_px": "Reproj",
        "shape_confidence": "Shape",
        "geometry_anchor_offset_m": "Anchor",
    }
    for feature, macro in wanted.items():
        sign = HANDSET_RISK_DIRECTION.get(feature)
        # The hand rule's assumed risk direction for the raw field: components
        # not in the monotone map keep their published CAVES orientation.
        assumed = {"inlier_ratio": -1, "num_inliers": -1, "reproj_residual_px": +1,
                   "shape_confidence": -1, "geometry_anchor_offset_m": +1}[feature]
        for ds, rs in sorted(by.items()):
            pairs = [(int(r["_caves_label"]), v[0])
                     for r in rs if (v := feature_vector(r, [feature])) is not None]
            value = auroc([p[0] for p in pairs], [assumed * p[1] for p in pairs])
            lines.append(
                f"\\newcommand{{\\psDir{macro}{DATASET_KEY[ds]}}}{{{value:.3f}}}"
            )
    return lines


def block_replay(replay_json: Path) -> List[str]:
    data = json.loads(replay_json.read_text())
    lines = ["% --- F. counterfactual second-stage gate replay, threshold = order statistic"]
    lines += [f"%     of training control-arm good scores at a {data['far_budget']:.0%} "
              "false-alert budget, chosen on the other two datasets."]
    lines += [f"\\newcommand{{\\psReplayBudget}}{{{pct(data['far_budget'], 0)}}}"]
    for fold, res in data["folds"].items():
        key = DATASET_KEY[fold]
        lines.append(f"\\newcommand{{\\psReplayObs{key}}}{{{res['n_scorable']}}}")
        for scorer, tag in (("mlp", "Mlp"), ("caves", "Caves")):
            for arm, atag in (("control", "Ctl"), ("attacked", "Att"), ("all", "All")):
                s = data["folds"][fold]["scorers"][scorer][arm]
                p = f"\\psReplay{tag}{atag}{key}"
                lines += [
                    f"\\newcommand{{{p}UnsafeB}}{{{s['unsafe_before']}}}",
                    f"\\newcommand{{{p}UnsafeA}}{{{s['unsafe_after']}}}",
                    f"\\newcommand{{{p}KmB}}{{{s['km_before']}}}",
                    f"\\newcommand{{{p}KmA}}{{{s['km_after']}}}",
                    f"\\newcommand{{{p}Kept}}{{{s['kept']}}}",
                    f"\\newcommand{{{p}Fr}}{{{s['false_rejects']}}}",
                    f"\\newcommand{{{p}Good}}{{{s['n_good']}}}",
                    f"\\newcommand{{{p}NObs}}{{{s['n_obs']}}}",
                ]
        ctl = data["folds"][fold]["scorers"]["mlp"]["control"]
        kept_pct = ctl["kept"] / ctl["n_obs"] if ctl["n_obs"] else float("nan")
        lines.append(f"\\newcommand{{\\psReplayMlpCtl{key}KeptPct}}{{{pct(kept_pct)}}}")
    return lines


def block_qc(code_root: Path, cell: Path) -> List[str]:
    proc = subprocess.run(
        [sys.executable, str(code_root / "scripts/reporting/forge_cell_qc.py"), str(cell)],
        capture_output=True, text=True, timeout=600,
    )
    text = proc.stdout + proc.stderr
    if not re.search(r"^(PASS|WARN|FAIL) ", text, re.M):
        raise SystemExit(f"forge_cell_qc.py produced no verdict for {cell}:\n{text[-800:]}")
    fail = re.search(r"FAIL I4 (\d+) accepted frames", text)
    warn = re.search(r"WARN I4 (\d+) boundary-tolerance", text)
    lines = ["% --- G. internal recoverability check on the cited UAV-VisLoc FORGE run"]
    lines += [f"%     ({cell}); counts parsed from forge_cell_qc.py output."]
    lines.append(f"\\newcommand{{\\psQcVislocInside}}{{{fail.group(1) if fail else 0}}}")
    lines.append(f"\\newcommand{{\\psQcVislocBoundary}}{{{warn.group(1) if warn else 0}}}")
    return lines


def block_drift(campaign_root: Path, ref_e1: Path) -> List[str]:
    """H. Same 100%-injection cell run in the current environment vs the cell of
    record: accept rate in each and per-frame agreement, on the shared frames."""
    lines = ["% --- H. environment drift on a 100%-injection reference cell",
             f"%     ({campaign_root / 'ref100_vpair_wrong_near'} vs {ref_e1})."]
    cell = campaign_root / "ref100_vpair_wrong_near"
    if not cell.exists() or not ref_e1.exists():
        lines.append("%     (reference cell not present yet; block skipped)")
        return lines
    new = list(csv.DictReader(open(one_run(cell))))
    old = {r["image_name"]: r for r in csv.DictReader(open(one_run(ref_e1)))}

    def injected(r):
        return not str(r.get("adversarial_failure_reason") or "").strip() and \
            str(r.get("is_adversarial", "")).lower() in ("true", "1")

    def acc(r):
        return str(r.get("accepted_for_metrics", "")).lower() == "true"

    # Intention-to-treat over every shared frame of the two 100%-injection runs:
    # a frame whose injection failed closed counts as rejected, so a change in
    # eligibility between environments shows up as drift instead of vanishing.
    pairs = [(r, old[r["image_name"]]) for r in new if r["image_name"] in old]
    if not pairs:
        lines.append("%     (no shared frames; block skipped)")
        return lines
    n = len(pairs)
    lines += [
        f"\\newcommand{{\\psDriftN}}{{{n}}}",
        f"\\newcommand{{\\psDriftFailClosedRecord}}{{{sum(not injected(b) for _, b in pairs)}}}",
        f"\\newcommand{{\\psDriftFailClosedRerun}}{{{sum(not injected(a) for a, _ in pairs)}}}",
        f"\\newcommand{{\\psDriftAcceptRecord}}{{{pct(sum(acc(b) for _, b in pairs) / n)}}}",
        f"\\newcommand{{\\psDriftAcceptRerun}}{{{pct(sum(acc(a) for a, _ in pairs) / n)}}}",
        f"\\newcommand{{\\psDriftAgree}}{{{sum(acc(a) == acc(b) for a, b in pairs)}}}",
        f"\\newcommand{{\\psDriftSameRef}}{{{sum(a.get('selected_reference_name') == b.get('selected_reference_name') for a, b in pairs)}}}",
    ]
    return lines


def check_table(table: Path, analyses: Dict[str, List[Dict[str, Any]]]) -> None:
    """The pooled macros must equal the column sums of the generated table."""
    sums: Dict[str, List[int]] = {}
    current = None
    for line in table.read_text().splitlines():
        if "&" not in line or line.startswith(("%", "\\")):
            continue
        cells = [c.strip() for c in line.rstrip("\\").split("&")]
        if len(cells) < 6 or not cells[2].isdigit():
            continue
        if cells[0]:
            current = cells[0]
        sums.setdefault(current, [0, 0, 0, 0])
        for i in range(4):
            sums[current][i] += int(cells[2 + i])
    name = {"vpair": "VPAIR", "avl": "AVL", "visloc": "UAV-VisLoc"}
    for ds, entries in analyses.items():
        want = [0, 0, 0, 0]
        for e in entries:
            s = e["reference_strata"]
            want[0] += s["control"]["correct_ref_accept"]
            want[1] += s["control"]["correct_ref_reject"]
            want[2] += s["control"]["wrong_ref_accept"] + s["attacked"]["wrong_ref_accept"]
            want[3] += s["control"]["wrong_ref_reject"] + s["attacked"]["wrong_ref_reject"]
        got = sums.get(name[ds])
        if got != want:
            raise SystemExit(f"pooled check failed for {ds}: table sums {got} != analysis {want}")
    print(f"pooled 2x2 macros match the column sums of {table}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path,
                        default=Path("/workspace/forge/results/v3_forge5050"))
    parser.add_argument("--code-root", type=Path, default=Path("/workspace/forge/code"))
    parser.add_argument("--qc-cell", type=Path,
                        default=Path("/workspace/forge/results/v3/forgeb_visloc_wrong_near"))
    parser.add_argument("--ref-e1", type=Path,
                        default=Path("/workspace/forge/results/v3/forgec_vpair_wrong_near/vpair/"
                                     "ours_ultravpr_cascade_6dof_e2e/20260810_155044"),
                        help="100%-injection cell of record for block H")
    parser.add_argument("--analysis-dir", type=Path,
                        help="where analysis_*.json, caves_learned_mlp/, paired_delta.json and "
                             "gate_replay.json live (default: the campaign root)")
    parser.add_argument("--check-table", type=Path,
                        help="generated mixed confusion table; pooled macros must equal its column sums")
    parser.add_argument("--paper", action="store_true",
                        help="header for the paper copy: sources only, no thesis-relative wording")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    adir = args.analysis_dir or args.campaign
    analyses = load_analyses(adir)
    header = [
        "% Auto-generated by gen_book_postsub_macros.py - DO NOT EDIT NUMBERS BY HAND",
        f"% source: {args.campaign} (mixed-rate FORGE campaign, seed-hashed 50/50",
        "%         assignment, one run per cell) + gate_replay.json + LODO scorer",
        "%         audit + forge_cell_qc.py.",
    ]
    if not args.paper:
        header += ["%         Post-submission: none of these numbers appear in the",
                   "%         printed thesis the committee holds."]
    header.append(f"% Values audited {date.today():%d/%m}.")
    body: List[str] = []
    body += block_mixed(analyses)
    body += block_overlap(args.campaign)
    body += block_lodo(adir / "caves_learned_mlp/caves_learned_weights.json")
    body += block_delta(adir / "paired_delta.json")
    body += block_direction(args.campaign)
    body += block_replay(adir / "gate_replay.json")
    body += block_qc(args.code_root, args.qc_cell)
    body += block_drift(args.campaign, args.ref_e1)
    if args.check_table:
        check_table(args.check_table, analyses)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(header + body) + "\n")
    print(f"wrote {args.out} ({len(body)} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
