#!/usr/bin/env python3
"""Does acceptance of a forced wrong reference depend on how wrong it is?

A faultload whose acceptance rate is flat in the fault's magnitude gives one
number for a whole family of faults, which reviewers cannot read as easy or
hard. FORGE records, for every attacked trial, the distance from the forced
reference tile to the ground-truth tile. This script reads that dose against
the recorded accept decision on the full-coverage cells.

The dose reading is the AUROC of the distance against acceptance: one half
means acceptance is independent of how far the forced tile sits, below one
half means the cascade rejects distant tiles more often. Intervals come from
the same contiguous within-flight block bootstrap the paper uses elsewhere.

Run directories must be passed as pinned timestamps, not globs.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from forge_arm_separation_audit import auroc, blocks, fnum, truthy  # noqa: E402

import random  # noqa: E402

DRAWS = 2000
DS_TAG = {"avl": "Avl", "vpair": "Vpair", "uav_visloc": "Visloc"}


def load(run_dir: str):
    """Attacked trials of one cell: dose, accept decision, flight, order."""
    path = os.path.join(run_dir, "predictions.csv")
    if not os.path.isfile(path):
        raise SystemExit(f"{run_dir}: no predictions.csv (pin the run timestamp)")
    with open(path) as fh:
        rows = list(csv.DictReader(fh))
    out = []
    for i, r in enumerate(rows):
        if not truthy(r.get("is_adversarial")):
            continue
        if not str(r.get("adversarial_forced_reference_name") or "").strip():
            continue
        dose = fnum(r.get("adversarial_forced_distance_to_gt_m"))
        if dose is None:
            continue
        out.append({
            "i": i,
            "flight": r.get("flight_id") or r.get("sequence") or r.get("location") or "0",
            "dose": dose,
            "accept": 1 if truthy(r.get("accepted_for_metrics")) else 0,
        })
    return out


def dataset_of(run_dir: str) -> str:
    parts = os.path.normpath(run_dir).split(os.sep)
    for p in parts:
        if p in DS_TAG:
            return p
    raise SystemExit(f"{run_dir}: cannot infer dataset from path")


def ci(rows, seed: int = 7):
    bl = blocks(rows)
    rng = random.Random(seed)
    vals = []
    for _ in range(DRAWS):
        samp = []
        for _ in range(len(bl)):
            samp.extend(bl[rng.randrange(len(bl))])
        a = auroc([(r["dose"], r["accept"]) for r in samp])
        if a is not None:
            vals.append(a)
    if len(vals) < DRAWS // 2:
        return None, None
    vals.sort()
    return vals[int(0.025 * len(vals))], vals[int(0.975 * len(vals)) - 1]


def terciles(rows):
    """Acceptance rate and count in the nearest and farthest dose tercile."""
    s = sorted(rows, key=lambda r: r["dose"])
    k = len(s) // 3
    lo, hi = s[:k], s[len(s) - k:]
    rate = lambda g: 100.0 * sum(r["accept"] for r in g) / len(g)
    return (rate(lo), len(lo), s[0]["dose"], s[k - 1]["dose"],
            rate(hi), len(hi), s[len(s) - k]["dose"], s[-1]["dose"])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="+", required=True, help="pinned run directories")
    ap.add_argument("--out", help="LaTeX macro file")
    ap.add_argument("--table", help="LaTeX table body, one row per condition")
    ap.add_argument("--json-out")
    args = ap.parse_args()

    pooled, sources, per_cell = {}, [], []
    for run in args.roots:
        ds = dataset_of(run)
        rows = load(run)
        if not rows:
            raise SystemExit(f"{run}: no attacked trials with a recorded dose")
        cell = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(run))))
        for r in rows:
            r["flight"] = f"{cell}/{r['flight']}"
        per_cell.append((ds, cell, rows))
        pooled.setdefault(ds, []).extend(rows)
        sources.append(f"% source: {run} ({len(rows)} attacked trials)")

    report = {}
    for ds, rows in sorted(pooled.items()):
        a = auroc([(r["dose"], r["accept"]) for r in rows])
        lo, hi = ci(rows)
        t = terciles(rows)
        report[ds] = {
            "n": len(rows), "n_accept": sum(r["accept"] for r in rows),
            "auroc": a, "lo": lo, "hi": hi,
            "near_rate": t[0], "near_n": t[1], "near_min": t[2], "near_max": t[3],
            "far_rate": t[4], "far_n": t[5], "far_min": t[6], "far_max": t[7],
        }
        print(f"{ds:<12} n={len(rows):>5} acc={report[ds]['n_accept']:>4}  "
              f"AUROC(dose vs accept)={a:.3f} [{lo:.3f}, {hi:.3f}]  "
              f"near tercile {t[0]:.1f}% ({t[2]:.0f}--{t[3]:.0f} m, n={t[1]}), "
              f"far tercile {t[4]:.1f}% ({t[6]:.0f}--{t[7]:.0f} m, n={t[5]})")

    cells = []
    for ds, cell, rows in per_cell:
        a = auroc([(r["dose"], r["accept"]) for r in rows])
        lo, hi = ci(rows)
        t = terciles(rows)
        cells.append({"dataset": ds, "cell": cell, "n": len(rows), "auroc": a,
                      "lo": lo, "hi": hi, "near_rate": t[0], "far_rate": t[4],
                      "near_max": t[3], "far_min": t[6]})
        print(f"  {cell:<38} n={len(rows):>5} AUROC={a:.3f} [{lo:.3f}, {hi:.3f}]"
              f"  near {t[0]:.1f}%  far {t[4]:.1f}%")

    if args.table:
        label = {"wrong_near": "Wrong, near", "wrong_far": "Wrong, far",
                 "visually_similar_wrong": "Visually similar"}
        name = {"avl": "AVL", "vpair": "VPAIR", "uav_visloc": "UAV-VisLoc"}
        lines = ["% Generated by scripts/reporting/forge_dose_response.py"] + sources
        lines += ["\\begin{tabular}{llrrrl}", "\\toprule",
                  "Dataset & Condition & Trials & Near & Far & AUROC [95\\% CI] \\\\",
                  " &  &  & \\multicolumn{2}{c}{accept (\\%)} &  \\\\", "\\midrule"]
        for c in cells:
            cond = next((v for k, v in label.items() if c["cell"].endswith(k)), c["cell"])
            lines.append("%s & %s & %d & %.1f & %.1f & %.3f [%.3f, %.3f] \\\\"
                         % (name[c["dataset"]], cond, c["n"], c["near_rate"],
                            c["far_rate"], c["auroc"], c["lo"], c["hi"]))
        lines += ["\\bottomrule", "\\end{tabular}"]
        with open(args.table, "w") as f:
            f.write("\n".join(lines) + "\n")
        print(f"\nwrote {args.table}")

    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump({"pooled": report, "cells": cells}, f, indent=2)
    if args.out:
        lines = ["% Generated by scripts/reporting/forge_dose_response.py"] + sources
        lines.append(f"% block bootstrap {DRAWS} draws, seed 7")
        for ds, r in report.items():
            k = DS_TAG[ds]
            lines += [
                "\\newcommand{\\dr%s}{%.3f}" % (k, r["auroc"]),
                "\\newcommand{\\drLo%s}{%.3f}" % (k, r["lo"]),
                "\\newcommand{\\drHi%s}{%.3f}" % (k, r["hi"]),
                "\\newcommand{\\drNear%s}{%.1f\\%%}" % (k, r["near_rate"]),
                "\\newcommand{\\drFar%s}{%.1f\\%%}" % (k, r["far_rate"]),
                "\\newcommand{\\drTercN%s}{%d}" % (k, r["near_n"]),
                "\\newcommand{\\drNearMax%s}{%.0f}" % (k, r["near_max"]),
                "\\newcommand{\\drFarMin%s}{%.0f}" % (k, r["far_min"]),
            ]
        with open(args.out, "w") as f:
            f.write("\n".join(lines) + "\n")
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
