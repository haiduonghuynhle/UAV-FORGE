#!/usr/bin/env python3
"""Macros for the third-party-gated FORGE cell (native OrthoLoC on AVL).

The acceptance decision on `native_ortholoc_2d3d_e2e` belongs to upstream
OrthoLoC code: its RoMa matcher, its confidence filter, and its PnP call. Our
runner only chooses which DOP crop that stack is shown. FORGE therefore forces
a footprint-disjoint crop and counts what the upstream layer still accepts.

Usage:
  python3 scripts/reporting/gen_native_forge_macros.py \
      --control <run_dir> --wrong <run_dir> --out <paper>/tables/native_forge_numbers.tex
"""
import argparse
import datetime as _dt
import math
import os
from pathlib import Path

import pandas as pd

R_M = 20.0  # AVL localization radius of record
Z = 1.959963984540054  # 95% normal quantile


def wilson(k, n):
    """Wilson score interval, the interval the supplementary tables report."""
    if not n:
        return float("nan"), float("nan")
    ph, z2 = k / n, Z * Z
    c = (ph + z2 / (2 * n)) / (1 + z2 / n)
    h = Z * math.sqrt(ph * (1 - ph) / n + z2 / (4 * n * n)) / (1 + z2 / n)
    return 100.0 * max(c - h, 0.0), 100.0 * min(c + h, 1.0)


def _truthy(series):
    return series.astype(str).str.lower().isin(["true", "1"])


def _load(run_dir):
    csv = Path(run_dir) / "predictions.csv"
    df = pd.read_csv(csv)
    # Frames the injection could not attack carry no forced tile; they are not
    # part of the paired population.
    df = df[df["failure_reason"].astype(str) != "ortholoc_center_tile_missing"]
    return df.set_index("image_name")


def _oblique_faults(df):
    """Under FORGE_AVL_FOOTPRINT=oblique, keep only forced trials whose tile is
    disjoint from the oblique AnyVisLoc camera footprint (gen_v3_tables); the
    control arm then pairs on the same frames. A no-op under the nadir labels."""
    if os.environ.get("FORGE_AVL_FOOTPRINT", "nadir") != "oblique":
        return df
    import gen_v3_tables as G
    keep = [n for n, row in df.iterrows() if G._is_genuine_fault({**row.to_dict(), "image_name": n}, "avl")]
    return df.loc[keep]


def _err_m(df):
    return pd.to_numeric(df["post_fgo_error_m"], errors="coerce")


def _arm_stats(df):
    acc = _truthy(df["accepted_for_metrics"])
    err = _err_m(df)[acc]
    return {
        "n": len(df),
        "accepted": int(acc.sum()),
        "median": float(err.median()) if len(err) else float("nan"),
        "within_r": int((err <= R_M).sum()),
        "unsafe": int(_truthy(df["unsafe_accept"]).sum()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--control", required=True)
    ap.add_argument("--wrong", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    ctrl, wrong = _load(args.control), _oblique_faults(_load(args.wrong))
    paired = ctrl.index.intersection(wrong.index)
    ctrl, wrong = ctrl.loc[paired], wrong.loc[paired]
    c, w = _arm_stats(ctrl), _arm_stats(wrong)

    # Forced offset: distance from the forced crop centre to the GT position.
    offset = pd.to_numeric(wrong.get("retrieval_anchor_error_m"), errors="coerce")
    # Frames the upstream layer accepted inside R although its reference was
    # forced away from the query footprint: these refute the impossibility
    # predicate that AVL geometry was assumed to enforce.
    w_err = _err_m(wrong)
    refute = wrong[_truthy(wrong["accepted_for_metrics"]) & (w_err <= R_M)]
    refute_off = offset.loc[refute.index]
    refute_err = float(w_err.loc[refute.index].iloc[0]) if len(refute) == 1 else float("nan")
    # Paired: the same frame in both arms, beyond R or not.
    beyond_c = _err_m(ctrl) > R_M
    beyond_w = w_err > R_M
    disc_w = int((beyond_w & ~beyond_c).sum())
    disc_c = int((beyond_c & ~beyond_w).sum())

    def pct(k, n):
        return f"{100.0 * k / n:.1f}\\%" if n else "--"

    m = {
        "nfN": f"{len(paired)}",
        "nfAcceptCtrlN": f"{c['accepted']}",
        "nfAcceptWrongN": f"{w['accepted']}",
        "nfAcceptCtrl": pct(c["accepted"], c["n"]),
        "nfAcceptWrong": pct(w["accepted"], w["n"]),
        "nfMedCtrl": f"{c['median']:.1f}",
        "nfMedWrong": f"{w['median']:.1f}",
        "nfWithinRCtrlN": f"{c['within_r']}",
        "nfWithinRWrongN": f"{w['within_r']}",
        # FORGE reports unsafe acceptance over all attacked frames, not over the
        # accepted subset (gen_v3_tables.py uses the same denominator).
        "nfUnsafeCtrl": pct(c["unsafe"], c["n"]),
        "nfUnsafeWrong": pct(w["unsafe"], w["n"]),
        "nfOffsetMin": f"{offset.min():.1f}",
        "nfOffsetMed": f"{offset.median():.1f}",
        "nfRefuteN": f"{len(refute)}",
        "nfRefuteOffsetLo": f"{refute_off.min():.0f}" if len(refute) else "--",
        "nfRefuteOffsetHi": f"{refute_off.max():.0f}" if len(refute) else "--",
        "nfRadius": f"{R_M:.0f}",
        "nfUnsafeCtrlLo": f"{wilson(c['unsafe'], c['n'])[0]:.1f}",
        "nfUnsafeCtrlHi": f"{wilson(c['unsafe'], c['n'])[1]:.1f}",
        "nfUnsafeWrongLo": f"{wilson(w['unsafe'], w['n'])[0]:.1f}",
        "nfUnsafeWrongHi": f"{wilson(w['unsafe'], w['n'])[1]:.1f}",
        # Paired discordance on the same frames: beyond R in one arm only.
        "nfDiscordWrongOnly": f"{int(disc_w)}",
        "nfDiscordCtrlOnly": f"{int(disc_c)}",
        "nfRefuteErr": f"{refute_err:.1f}" if len(refute) == 1 else "--",
    }
    assert all(v == v for v in [c["median"], w["median"]]), "empty accepted set"
    assert c["n"] == w["n"] == len(paired), "arms are not paired"
    # A NaN error compares False against R, so a partially blank column would
    # silently undercount within-R instead of failing loudly.
    for name, arm in (("control", ctrl), ("wrong", wrong)):
        acc = _truthy(arm["accepted_for_metrics"])
        bad = int(_err_m(arm)[acc].isna().sum())
        assert bad == 0, f"{name} arm: {bad} accepted rows without post_fgo_error_m"
    assert int(offset.isna().sum()) == 0, "forced-offset column has blanks"

    stamp = _dt.date.today().isoformat()
    lines = [
        f"% Generated by scripts/reporting/gen_native_forge_macros.py on {stamp}. Do not edit by hand.",
        f"% source: {args.control}",
        f"% source: {args.wrong}",
    ]
    lines += [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in m.items()]
    Path(args.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    for k, v in m.items():
        print(f"{k} = {v}")


if __name__ == "__main__":
    main()
