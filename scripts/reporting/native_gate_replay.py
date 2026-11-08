#!/usr/bin/env python3
"""Counterfactual acceptance gate on the third-party OrthoLoC FORGE cell.

The OrthoLoC 2D-3D back end publishes no accept decision: on every frame that
reaches PnP it emits a pose, so the acceptance columns of the supplementary
cell are total and carry no information. This script asks what the same frozen
records would say if a minimum-inlier rule sat in front of them, and reports
acceptance and unsafe acceptance per arm with Wilson intervals.

Pure re-analysis. It reads only columns the frozen runs already wrote; it never
re-runs the matcher and needs no GPU.

What the rule is, honestly
--------------------------
The inlier COUNT is upstream's: OrthoLoC's own `Correspondences2D3D.calibrate`
declares a correspondence an inlier at its own default RANSAC reprojection
radius of 5 px (reference/ortholoc/correspondences/Correspondences2D3D.py:52),
and poselib does the counting. The CUTOFF on that count is not upstream's.
OrthoLoC's released path contains no minimum-inlier and no minimum-ratio test:
`utils.pose.run_pnp` in poselib mode reports failure only when fewer than five
points are supplied or poselib returns no pose at all (utils/pose.py:182,
211-212). So this is a community-default rule of the shape standard practice
applies, replayed on someone else's inlier counts -- NOT a gate OrthoLoC ships.
Report it that way. Sweep the threshold and show the curve; do not pick the
value that flatters the result.

Two measurement caveats a reader is owed
----------------------------------------
* `num_matches` is the RAW RoMa sample, and MatcherRoMa samples a fixed 3000
  correspondences per pair (image_matching/MatcherRoMa.py:20). It is close to
  constant, so `inlier_ratio` here is a rescaled `num_inliers` and an
  inlier-ratio rule is the same rule as an inlier-count rule. Nothing records
  how many correspondences actually survived the 0.5 confidence filter and the
  2D-3D lift to reach PnP, so no honest ratio is recoverable from the records.
* The recorded `unsafe_accept` is `accepted_for_metrics AND error > radius`.
  At threshold 0 this replay must reproduce it exactly; main() asserts that, so
  a schema drift fails loudly instead of shifting the numbers quietly.

How the operating point is chosen
---------------------------------
Picking the threshold that flatters the result is the obvious way to cheat
here, so the choice never looks at the forced arm and never looks at ground
truth. The rule is: over the sweep grid, take the strictest setting whose
CONTROL-arm acceptance still reaches --target-coverage. Control acceptance is
a count of published poses, so it needs no labels. Everything the forced arm
says is then read at that frozen setting, and the whole grid is printed beside
it so a reader can see what another choice would have given.

Usage:
  python3 scripts/reporting/native_gate_replay.py \
      --control <run_dir> --wrong <run_dir> \
      --min-inliers 0 25 50 100 200 --max-reproj inf 10 5 3 \
      --target-coverage 0.90 --macros-out tables/native_gate_numbers.tex
  python3 scripts/reporting/native_gate_replay.py --self-test
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd

_REPORTING_ROOT = Path(__file__).resolve().parent
if str(_REPORTING_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPORTING_ROOT))

from gen_native_forge_macros import R_M, _err_m, _load, _oblique_faults, _truthy, wilson  # noqa: E402


def replay_arm(df: pd.DataFrame, min_inliers: int, radius_m: float,
               max_reproj: float = float("inf")) -> dict:
    """Counts under the two-check rule this replay puts in front of the records.

    Accept iff the back end returned a pose, its inlier count reaches
    min_inliers, and its PnP reprojection error stays under max_reproj. Both
    checks are community defaults; neither is a test OrthoLoC ships.

    Frames that never reached PnP carry num_inliers 0 and accepted_for_metrics
    False, so they are refused at every setting, as they already were. A frame
    with no recorded reprojection error passes the second check, so a missing
    column can never tighten the gate silently.
    """
    returned = _truthy(df["accepted_for_metrics"])
    inliers = pd.to_numeric(df["num_inliers"], errors="coerce").fillna(0.0)
    if "pnp_reprojection_error_px" in df.columns:
        reproj = pd.to_numeric(df["pnp_reprojection_error_px"], errors="coerce")
        within_reproj = reproj.isna() | (reproj <= max_reproj)
    else:
        # No residual recorded: the second check cannot fire, and must not.
        within_reproj = pd.Series(True, index=df.index)
    accept = returned & (inliers >= min_inliers) & within_reproj
    err = _err_m(df)
    bad = err > radius_m
    return {
        "n": int(len(df)),
        "returned": int(returned.sum()),
        "accepted": int(accept.sum()),
        "unsafe": int((accept & bad).sum()),
        # A pose the back end returned, landing inside the radius, that the
        # rule throws away: the price of the gate on this arm.
        "false_reject": int((returned & ~accept & err.notna() & ~bad).sum()),
    }


def _fmt(k: int, n: int) -> str:
    if not n:
        return "--"
    lo, hi = wilson(k, n)
    return f"{100.0 * k / n:5.1f}% [{lo:5.1f}, {hi:5.1f}]"



def _grid(min_inliers, max_reproj):
    """Sweep points, strictest first: most inliers, then tightest residual."""
    return [(t, r) for t in sorted(min_inliers, reverse=True)
            for r in sorted(max_reproj)]


def pick_operating_point(ctrl: pd.DataFrame, radius_m: float, min_inliers, max_reproj,
                         target_coverage: float):
    """Strictest grid point whose CONTROL acceptance still reaches the target.

    Uses the control arm only, and counts published poses rather than correct
    ones, so neither the forced arm nor ground truth can influence it.
    """
    n = len(ctrl)
    best = None
    for t, r in _grid(min_inliers, max_reproj):
        coverage = replay_arm(ctrl, t, radius_m, r)["accepted"] / n if n else 0.0
        if coverage >= target_coverage:
            best = (t, r, coverage)
            break
    return best


def _macro_lines(point, ctrl_stats, wrong_stats, n_paired, radius_m, target,
                 control_run="", wrong_run="", max_reproj_used=True) -> list:
    t, r, coverage = point
    reproj = "none" if r == float("inf") else f"{r:g}"

    def pct(k, n):
        return f"{100.0 * k / n:.1f}\\%" if n else "--"

    def ci(k, n):
        lo, hi = wilson(k, n)
        return f"{lo:.1f}\\%", f"{hi:.1f}\\%"

    wa_lo, wa_hi = ci(wrong_stats["accepted"], n_paired)
    out = [
        "% Auto-generated by scripts/reporting/native_gate_replay.py - DO NOT EDIT BY HAND.",
        f"% Counterfactual {'two-check' if max_reproj_used else 'minimum-inlier'} gate"
        " replayed on the frozen third-party records.",
        "% The cutoffs are community defaults, not checks the back end ships.",
        f"% source control: {control_run}",
        f"% source forced:  {wrong_run}",
        f"\\newcommand{{\\ngN}}{{{n_paired}}}",
        f"\\newcommand{{\\ngMinInliers}}{{{t}}}",
        f"\\newcommand{{\\ngMaxReproj}}{{{reproj}}}",
        f"\\newcommand{{\\ngTargetCoverage}}{{{100.0 * target:.0f}\\%}}",
        f"\\newcommand{{\\ngRadius}}{{{radius_m:.0f}}}",
        f"\\newcommand{{\\ngCtrlAccept}}{{{pct(ctrl_stats['accepted'], n_paired)}}}",
        f"\\newcommand{{\\ngCtrlUnsafe}}{{{pct(ctrl_stats['unsafe'], n_paired)}}}",
        f"\\newcommand{{\\ngCtrlFalseRej}}{{{ctrl_stats['false_reject']}}}",
        f"\\newcommand{{\\ngWrongAccept}}{{{pct(wrong_stats['accepted'], n_paired)}}}",
        f"\\newcommand{{\\ngWrongAcceptLo}}{{{wa_lo}}}",
        f"\\newcommand{{\\ngWrongAcceptHi}}{{{wa_hi}}}",
        f"\\newcommand{{\\ngWrongUnsafe}}{{{pct(wrong_stats['unsafe'], n_paired)}}}",
        f"\\newcommand{{\\ngWrongAcceptN}}{{{wrong_stats['accepted']}}}",
        f"\\newcommand{{\\ngWrongUnsafeN}}{{{wrong_stats['unsafe']}}}",
    ]
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--control", required=True, help="control-arm run directory")
    parser.add_argument("--wrong", required=True, help="forced-reference run directory")
    parser.add_argument("--min-inliers", type=int, nargs="+", default=[0, 25, 50, 100, 200],
                        help="inlier cutoffs to sweep; 0 with no residual bound "
                             "reproduces the recorded cell")
    parser.add_argument("--max-reproj", type=float, nargs="+", default=[float("inf"), 10.0, 5.0, 3.0],
                        help="PnP reprojection bounds in px; inf disables the check")
    parser.add_argument("--target-coverage", type=float, default=0.90,
                        help="control-arm acceptance the operating point must keep")
    parser.add_argument("--radius", type=float, default=R_M,
                        help="localization radius defining an unsafe accept (m)")
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--macros-out", type=Path, help="LaTeX macros for the operating point")
    parser.add_argument("--provenance-out", type=Path)
    args = parser.parse_args()

    ctrl, wrong = _load(args.control), _oblique_faults(_load(args.wrong))
    paired = ctrl.index.intersection(wrong.index)
    ctrl, wrong = ctrl.loc[paired], wrong.loc[paired]
    arms = [("control", ctrl), ("wrong", wrong)]

    # A missing residual column makes the second check a no-op, and a silent
    # no-op would turn a two-check result into a one-check result without
    # anyone noticing. Say so, and drop the dimension from the grid.
    usable_reproj = all(
        "pnp_reprojection_error_px" in df.columns
        and pd.to_numeric(df["pnp_reprojection_error_px"], errors="coerce").notna().any()
        for _, df in arms)
    if not usable_reproj:
        bounded = [r for r in args.max_reproj if r != float("inf")]
        if bounded:
            print("WARNING: these runs record no PnP reprojection error.")
            print("         The residual check cannot fire, so the replay is the")
            print("         inlier rule alone. Reporting it as a two-check layer")
            print("         would be false; the sweep below drops that dimension.")
            print()
        args.max_reproj = [float("inf")]

    # The base setting is the recorded cell. If the replay disagrees with the
    # frozen unsafe_accept column there, the schema moved and every swept
    # number below it is meaningless.
    for name, df in arms:
        base = replay_arm(df, 0, args.radius, float("inf"))
        recorded = int(_truthy(df["unsafe_accept"]).sum())
        assert base["unsafe"] == recorded, (
            f"{name} arm: replay at the base setting says {base['unsafe']} unsafe, "
            f"recorded column says {recorded}")

    shape = "TWO-CHECK" if usable_reproj else "MINIMUM-INLIER"
    print(f"COUNTERFACTUAL {shape} GATE on the frozen native OrthoLoC records")
    print(f"paired frames per arm: {len(paired)}   unsafe radius: {args.radius:.0f} m")
    print("The counts and residuals are the back end's own;")
    print("the cutoffs are community defaults, not checks OrthoLoC ships.")
    print()
    header = (f"{'inl':>5}{'reproj':>8}  {'arm':<8}{'returned':>9}{'accepted':>9}"
              f"{'accept rate 95% CI':>24}{'unsafe rate 95% CI':>24}{'false rej':>10}")
    print(header)
    print("-" * len(header))

    payload: dict = {"radius_m": args.radius, "n_paired": int(len(paired)),
                     "target_coverage": args.target_coverage, "sweep": {}}
    for t, r in _grid(args.min_inliers, args.max_reproj):
        key = f"{t}|{r}"
        for name, df in arms:
            stats = replay_arm(df, t, args.radius, r)
            label = "inf" if r == float("inf") else f"{r:g}"
            print(f"{t:>5}{label:>8}  {name:<8}{stats['returned']:>9}{stats['accepted']:>9}"
                  f"{_fmt(stats['accepted'], stats['n']):>24}"
                  f"{_fmt(stats['unsafe'], stats['n']):>24}"
                  f"{stats['false_reject']:>10}")
            payload["sweep"].setdefault(key, {})[name] = stats
        print()

    point = pick_operating_point(ctrl, args.radius, args.min_inliers, args.max_reproj,
                                 args.target_coverage)
    if point is None:
        print(f"NO operating point keeps {100 * args.target_coverage:.0f}% of control poses.")
        print("Report that as the result: a standard gate cannot run at this coverage here.")
        return 1

    t, r, coverage = point
    cs = replay_arm(ctrl, t, args.radius, r)
    ws = replay_arm(wrong, t, args.radius, r)
    label = "none" if r == float("inf") else f"{r:g} px"
    print(f"OPERATING POINT (chosen on the control arm only, target "
          f"{100 * args.target_coverage:.0f}% coverage)")
    print(f"  min inliers {t}, reprojection bound {label}, control coverage {100 * coverage:.1f}%")
    print(f"  control: accepted {cs['accepted']}/{len(paired)}, unsafe {cs['unsafe']}")
    print(f"  forced : accepted {ws['accepted']}/{len(paired)}, unsafe {ws['unsafe']}")
    payload["operating_point"] = {"min_inliers": t, "max_reproj": None if r == float("inf") else r,
                                  "control_coverage": coverage, "control": cs, "wrong": ws}

    if args.macros_out:
        args.macros_out.write_text("\n".join(
            _macro_lines((t, r, coverage), cs, ws, len(paired), args.radius,
                         args.target_coverage, args.control, args.wrong,
                         usable_reproj)) + "\n",
            encoding="utf-8")
        print(f"wrote {args.macros_out}")

    if args.json_out:
        args.json_out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {args.json_out}")

    if args.provenance_out:
        import hashlib
        prov = {"generator": "scripts/reporting/native_gate_replay.py",
                "control_run": str(args.control), "wrong_run": str(args.wrong),
                "avl_footprint_label": os.environ.get("FORGE_AVL_FOOTPRINT", "nadir"),
                "operating_point": payload["operating_point"],
                "sources": []}
        for name, run in (("control", args.control), ("wrong", args.wrong)):
            csv = Path(run) / "predictions.csv"
            prov["sources"].append({"arm": name, "path": str(csv),
                                    "sha256": hashlib.sha256(csv.read_bytes()).hexdigest()})
        args.provenance_out.write_text(json.dumps(prov, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {args.provenance_out}")

    print()
    print("Reading guide: rates use the all-frames denominator of gen_v3_tables.py.")
    print("'false rej' counts poses the back end returned inside the radius that the")
    print("rule discards. Re-analysis of frozen records under a rule we chose the")
    print("shape of; not a gate the third-party system publishes, and not a new run.")
    return 0

def _self_check() -> None:
    """Synthetic rows: the gate must refuse on inliers and price the refusal."""
    df = pd.DataFrame({
        "accepted_for_metrics": ["True", "True", "True", "False"],
        "num_inliers": [300, 40, 500, 0],
        # good, good, unsafe, no pose
        "post_fgo_error_m": [5.0, 8.0, 900.0, float("nan")],
    })
    wide_open = replay_arm(df, 0, 20.0)
    assert wide_open == {"n": 4, "returned": 3, "accepted": 3, "unsafe": 1,
                         "false_reject": 0}, wide_open

    # t=100 refuses the 40-inlier frame, which was a GOOD pose: a false reject,
    # and the unsafe 500-inlier frame survives, which is the honest bad news.
    strict = replay_arm(df, 100, 20.0)
    assert strict == {"n": 4, "returned": 3, "accepted": 2, "unsafe": 1,
                      "false_reject": 1}, strict

    # t above every count refuses everything, including the unsafe accept.
    shut = replay_arm(df, 1000, 20.0)
    assert shut == {"n": 4, "returned": 3, "accepted": 0, "unsafe": 0,
                    "false_reject": 2}, shut

    # A blank inlier count must read as 0, never as "pass".
    blank = pd.DataFrame({"accepted_for_metrics": ["True"], "num_inliers": [""],
                          "post_fgo_error_m": [3.0]})
    assert replay_arm(blank, 1, 20.0)["accepted"] == 0

    lo, hi = wilson(1, 4)
    assert lo < 25.0 < hi, (lo, hi)

    # The residual bound must bite on its own, and a missing residual must not.
    res = pd.DataFrame({
        "accepted_for_metrics": ["True", "True"],
        "num_inliers": [100, 100],
        "pnp_reprojection_error_px": [2.0, 9.0],
        "post_fgo_error_m": [3.0, 900.0],
    })
    assert replay_arm(res, 0, 20.0, float("inf"))["accepted"] == 2
    assert replay_arm(res, 0, 20.0, 5.0)["accepted"] == 1
    assert replay_arm(res, 0, 20.0, 5.0)["unsafe"] == 0

    # The operating point is the strictest grid setting that still clears the
    # control-arm target, and it never consults the forced arm.
    ctrl = pd.DataFrame({
        "accepted_for_metrics": ["True"] * 10,
        "num_inliers": [10, 30, 30, 60, 60, 60, 60, 60, 60, 60],
        "pnp_reprojection_error_px": [1.0] * 10,
        "post_fgo_error_m": [3.0] * 10,
    })
    point = pick_operating_point(ctrl, 20.0, [0, 25, 50], [float("inf"), 5.0], 0.70)
    assert point[0] == 50 and point[1] == 5.0, point
    assert abs(point[2] - 0.7) < 1e-9, point
    # 9 of 10 clear an inlier cutoff of 25, so a 95% target falls back to 0.
    assert pick_operating_point(ctrl, 20.0, [0, 25, 50], [float("inf")], 0.95)[0] == 0
    assert pick_operating_point(ctrl, 20.0, [100], [float("inf")], 0.5) is None

    print("self-check ok")


if __name__ == "__main__":
    if "--self-check" in sys.argv:
        _self_check()
        raise SystemExit(0)
    raise SystemExit(main())
