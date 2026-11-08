#!/usr/bin/env python3
"""Block bootstrap intervals, distinct-trial counts and paired ablation
differences for the FORGE full-coverage campaign (CPU only, no new runs).

Per dataset and condition (plus pooled per dataset):
  1. attacked (genuine-fault) trials n, accepted k, rate; Wilson CI and a
     block bootstrap CI over contiguous within-flight frame blocks;
  2. distinct (image_name, forced reference) pairs across the three conditions
     and the frame-level rate (accepted under >= 1 condition) with a block CI;
  3. paired ablation on the SAME frames: gates-off minus full-gate acceptance,
     paired block bootstrap (resample frames, recompute both arms), and ratio.

Point estimates must reproduce \\forgeStrat*FaultAccept, \\forge{Avl,Vpair,Visloc}*Accept
and \\forgeGatesOff* in v3_numbers.tex; pass --check to verify and report.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import gen_v3_tables as gen
from forge_mixed_confusion import block_keys, cluster_bootstrap, wilson

R = "/workspace/forge/results"
OURS = "ours_ultravpr_cascade_6dof_e2e"
CONDS = {"wrong_near": "Near", "wrong_far": "Far", "visually_similar_wrong": "Sim"}
DS_NAME = {"avl": "Avl", "vpair": "Vpair", "uav_visloc": "Visloc"}

# Cells of record: exactly one timestamped run each (see % source: in v3_numbers.tex).
FULL = {
    "avl": {c: f"{R}/v3/forgec_avl_{c}/avl/{OURS}/{ts}" for c, ts in
            zip(CONDS, ("20260805_195057", "20260805_195151", "20260805_211521"))},
    "vpair": {c: f"{R}/v3/forgec_vpair_{c}/vpair/{OURS}/{ts}" for c, ts in
              zip(CONDS, ("20260810_155044", "20260810_164157", "20260810_173120"))},
    "uav_visloc": {c: f"{R}/v3/forgeb_visloc_{c}/uav_visloc/{OURS}/{ts}" for c, ts in
                   zip(CONDS, ("20260811_044051", "20260811_094249", "20260811_144540"))},
}
# Ablation pairs (full-gate arm, two-checks-removed arm). UAV-VisLoc uses the
# matched ablation_b campaign for BOTH arms; AVL/VPAIR pair the cell of record
# with the gatesoff cell.
ABL = {
    "avl": {c: (FULL["avl"][c], f"{R}/v3/forge_gatesoff_avl_{c}/avl/{OURS}/{ts}") for c, ts in
            zip(CONDS, ("20260814_115125", "20260814_123450", "20260814_131943"))},
    "vpair": {c: (FULL["vpair"][c], f"{R}/v3/forge_gatesoff_vpair_{c}/vpair/{OURS}/{ts}") for c, ts in
              zip(CONDS, ("20260813_152910", "20260813_164019", "20260813_174850"))},
    "uav_visloc": {c: (f"{R}/v3_visloc_ablation_b/fullgate_{c}/uav_visloc/{OURS}/{f}",
                       f"{R}/v3_visloc_ablation_b/gatesoff_{c}/uav_visloc/{OURS}/{g}")
                   for c, f, g in zip(CONDS, ("20260822_195213", "20260822_205150", "20260822_215007"),
                                      ("20260822_224915", "20260822_234850", "20260823_004801"))},
}
DRAWS = 2000
# Block-length rule: cut the rows of each flight into contiguous blocks so the
# campaign yields about TARGET blocks in total. 25 is the reported setting;
# --block-target halves or doubles it for the sensitivity check.
TARGET = 25


def acc(row) -> bool:
    return gen._truthy(row.get("accepted_for_metrics"))


def fault_rows(run_dir: str, ds: str) -> List[Dict[str, str]]:
    rows = [r for r in gen._attacked_rows(run_dir) if gen._is_genuine_fault(r, ds)]
    names = [r["image_name"] for r in rows]
    assert len(names) == len(set(names)), f"duplicate image_name in {run_dir}"
    return rows


def block_info(rows) -> Dict[str, int]:
    sizes = Counter(block_keys(rows, target_blocks=TARGET).values())
    return {"n_blocks": len(sizes), "block_len_rows": max(sizes.values()), "block_len_min": min(sizes.values())}


def rate_stats(rows) -> Dict[str, object]:
    n, k = len(rows), sum(acc(r) for r in rows)
    lo, hi = cluster_bootstrap(rows, lambda s: sum(acc(r) for r in s) / len(s), draws=DRAWS, target_blocks=TARGET)
    return {"n": n, "accepted": k, "rate": k / n, "wilson": wilson(k, n), "block": (lo, hi), **block_info(rows)}


def paired(pairs) -> Dict[str, object]:
    """pairs: pseudo-rows with full/off booleans; resample frame blocks, recompute both arms."""
    def rates(s):
        return sum(p["full"] for p in s) / len(s), sum(p["off"] for p in s) / len(s)
    f, o = rates(pairs)
    diff = cluster_bootstrap(pairs, lambda s: rates(s)[1] - rates(s)[0], draws=DRAWS, target_blocks=TARGET)
    ratio = cluster_bootstrap(pairs, lambda s: (lambda a, b: b / a if a else float("nan"))(*rates(s)), draws=DRAWS, target_blocks=TARGET)
    return {"n_frames": len(pairs), "full_rate": f, "off_rate": o, "diff": o - f, "diff_ci": diff,
            "ratio": o / f if f else float("nan"), "ratio_ci": ratio, **block_info(pairs)}


def pseudo(row, **flags):
    return {"image_name": row["image_name"], "query_flight_id": row.get("query_flight_id", ""), **flags}


def analyse(ds: str) -> Dict[str, object]:
    out: Dict[str, object] = {"cells": {}, "ablation_cells": {}, "conditions": {}, "ablation": {}}
    pooled, common, acc_any, distinct = [], None, set(), set()
    for cond in CONDS:
        rows = fault_rows(FULL[ds][cond], ds)
        out["cells"][cond] = FULL[ds][cond]
        out["conditions"][cond] = rate_stats(rows)
        pooled += rows
        names = {r["image_name"] for r in rows}
        common = names if common is None else common & names
        acc_any |= {r["image_name"] for r in rows if acc(r)}
        distinct |= {(r["image_name"], r.get("adversarial_forced_reference_name", "")) for r in rows}
    out["conditions"]["pooled"] = rate_stats(pooled)
    out["distinct_trials"] = len(distinct)
    # Two conditions can select the same (frame, forced reference) pair. The
    # repeat is not a redundant observation: the fusion carries trajectory state
    # from earlier frames, which the other condition perturbed differently. We
    # keep the first occurrence and count how often the repeats disagree.
    seen: Dict[tuple, Dict[str, str]] = {}
    repeats = disagree = 0
    for r in pooled:
        key = (r["image_name"], r.get("adversarial_forced_reference_name", ""))
        if key in seen:
            repeats += 1
            disagree += acc(seen[key]) != acc(r)
            continue
        seen[key] = r
    assert len(seen) == len(distinct)
    out["distinct"] = {**rate_stats(list(seen.values())),
                       "repeats": repeats, "repeat_disagree": disagree}
    # Frame level: every frame with at least one genuine-fault trial (union),
    # accepted if any of its fault trials is. Where every condition attacks the
    # same frames (all nadir labels) this equals the complete-case set, which
    # is also reported: frames whose three trials are all faults.
    def frame_stats(names):
        firsts: Dict[str, Dict[str, str]] = {}
        for r in pooled:
            firsts.setdefault(r["image_name"], r)
        fr = [pseudo(firsts[n], a=n in acc_any) for n in sorted(names)]
        k = sum(p["a"] for p in fr)
        return {"n": len(fr), "accepted": k, "rate": k / len(fr),
                "block": cluster_bootstrap(fr, lambda s: sum(p["a"] for p in s) / len(s), draws=DRAWS, target_blocks=TARGET),
                **block_info(fr)}
    out["frame"] = frame_stats({r["image_name"] for r in pooled})
    out["frame_common"] = frame_stats(common)
    all_pairs = []
    for cond, (full_dir, off_dir) in ABL[ds].items():
        full, off = fault_rows(full_dir, ds), fault_rows(off_dir, ds)
        out["ablation_cells"][cond] = {"full": full_dir, "off": off_dir,
                                       "full_all": rate_stats(full), "off_all": rate_stats(off)}
        offmap = {r["image_name"]: acc(r) for r in off}
        pairs = [pseudo(r, full=acc(r), off=offmap[r["image_name"]]) for r in full if r["image_name"] in offmap]
        out["ablation"][cond] = paired(pairs)
        all_pairs += pairs
    out["ablation"]["pooled"] = paired(all_pairs)
    return out


def pct(x: float) -> str:
    return f"{100 * x:.1f}\\%"


def macros(res: Dict[str, Dict[str, object]]) -> str:
    L = ["% Auto-generated by scripts/reporting/forge_block_ci.py - DO NOT EDIT NUMBERS BY HAND",
         f"% block bootstrap, {DRAWS} resamples, 2.5/97.5 percentiles; blocks = contiguous frames per flight (forge_mixed_confusion.block_keys)"]
    for ds, r in res.items():
        D = DS_NAME[ds]
        for cond, c in r["conditions"].items():
            C = CONDS.get(cond, "Pooled")
            src = r["cells"].get(cond, "pooled over the three cells above")
            L.append(f"% source: {src}")
            L += [f"\\newcommand{{\\fbcN{D}{C}}}{{{c['n']}}}",
                  f"\\newcommand{{\\fbcAcc{D}{C}}}{{{c['accepted']}}}",
                  f"\\newcommand{{\\fbcRate{D}{C}}}{{{pct(c['rate'])}}}",
                  f"\\newcommand{{\\fbcLo{D}{C}}}{{{pct(c['block'][0])}}}",
                  f"\\newcommand{{\\fbcHi{D}{C}}}{{{pct(c['block'][1])}}}",
                  f"\\newcommand{{\\fbcWilsonLo{D}{C}}}{{{pct(c['wilson'][0])}}}",
                  f"\\newcommand{{\\fbcWilsonHi{D}{C}}}{{{pct(c['wilson'][1])}}}"]
        f = r["frame"]
        fc = r["frame_common"]
        L += [f"\\newcommand{{\\fbcFrameCommonN{D}}}{{{fc['n']}}}",
              f"\\newcommand{{\\fbcFrameCommonRate{D}}}{{{pct(fc['rate'])}}}",
              f"\\newcommand{{\\fbcFrameCommonLo{D}}}{{{pct(fc['block'][0])}}}",
              f"\\newcommand{{\\fbcFrameCommonHi{D}}}{{{pct(fc['block'][1])}}}"]
        d = r["distinct"]
        L += [f"\\newcommand{{\\fbcDistinct{D}}}{{{r['distinct_trials']}}}",
              f"\\newcommand{{\\fbcDistinctAcc{D}}}{{{d['accepted']}}}",
              f"\\newcommand{{\\fbcDistinctRate{D}}}{{{pct(d['rate'])}}}",
              f"\\newcommand{{\\fbcDistinctLo{D}}}{{{pct(d['block'][0])}}}",
              f"\\newcommand{{\\fbcDistinctHi{D}}}{{{pct(d['block'][1])}}}",
              f"\\newcommand{{\\fbcRepeat{D}}}{{{d['repeats']}}}",
              f"\\newcommand{{\\fbcRepeatFlip{D}}}{{{d['repeat_disagree']}}}",
              f"\\newcommand{{\\fbcFrameN{D}}}{{{f['n']}}}",
              f"\\newcommand{{\\fbcFrameAcc{D}}}{{{f['accepted']}}}",
              f"\\newcommand{{\\fbcFrameRate{D}}}{{{pct(f['rate'])}}}",
              f"\\newcommand{{\\fbcFrameLo{D}}}{{{pct(f['block'][0])}}}",
              f"\\newcommand{{\\fbcFrameHi{D}}}{{{pct(f['block'][1])}}}",
              f"\\newcommand{{\\fbcBlockLen{D}}}{{{max(c['block_len_rows'] for k, c in r['conditions'].items() if k != 'pooled')}}}",
              f"\\newcommand{{\\fbcBlocks{D}}}{{{r['conditions']['wrong_near']['n_blocks']}}}"]
        for cond, a in r["ablation"].items():
            C = CONDS.get(cond, "Pooled")
            cell = r["ablation_cells"].get(cond)
            L.append(f"% source: {cell['full']} vs {cell['off']}" if cell else "% source: pooled over the three pairs above")
            L += [f"\\newcommand{{\\fbcAblN{D}{C}}}{{{a['n_frames']}}}",
                  f"\\newcommand{{\\fbcAblFullRate{D}{C}}}{{{pct(a['full_rate'])}}}",
                  f"\\newcommand{{\\fbcAblOffRate{D}{C}}}{{{pct(a['off_rate'])}}}",
                  f"\\newcommand{{\\fbcAblDiff{D}{C}}}{{{pct(a['diff'])}}}",
                  f"\\newcommand{{\\fbcAblDiffLo{D}{C}}}{{{pct(a['diff_ci'][0])}}}",
                  f"\\newcommand{{\\fbcAblDiffHi{D}{C}}}{{{pct(a['diff_ci'][1])}}}",
                  f"\\newcommand{{\\fbcAblRatio{D}{C}}}{{{a['ratio']:.1f}}}",
                  f"\\newcommand{{\\fbcAblRatioLo{D}{C}}}{{{a['ratio_ci'][0]:.1f}}}",
                  f"\\newcommand{{\\fbcAblRatioHi{D}{C}}}{{{a['ratio_ci'][1]:.1f}}}"]
    return "\n".join(L) + "\n"


def check(res, tex: Path) -> List[str]:
    """Compare point estimates against the paper macros; return mismatch lines."""
    have = dict(re.findall(r"\\newcommand\{\\(\w+)\}\{([^}]*)\}", tex.read_text()))
    want: Dict[str, str] = {}
    for ds, r in res.items():
        D = DS_NAME[ds]
        want[f"forgeStrat{D}FaultN"] = str(r["conditions"]["pooled"]["n"])
        want[f"forgeStrat{D}FaultAccept"] = pct(r["conditions"]["pooled"]["rate"])
        want[f"forge{D}FrameAcc"] = str(r["frame"]["accepted"])
        want[f"forge{D}FrameN"] = str(r["frame"]["n"])
        want[f"forge{D}FramePct"] = pct(r["frame"]["rate"])
        for cond, C in CONDS.items():
            want[f"forge{D}{C}Accept"] = pct(r["conditions"][cond]["rate"])
            off = r["ablation_cells"][cond]["off_all"]
            go = {"avl": "forgeGatesOffAvl", "vpair": "forgeGatesOff", "uav_visloc": "forgeAblVislocOff"}[ds]
            want[f"{go}{C}Accept"] = pct(off["rate"])
            if ds == "uav_visloc":
                want[f"forgeAblVislocFull{C}Accept"] = pct(r["ablation_cells"][cond]["full_all"]["rate"])
    seen = [k for k in want if k in have]
    bad = [f"{k}: computed {want[k]} vs paper {have[k]}" for k in seen if have[k] != want[k]]
    return bad + ([f"({len(seen)} macros compared)"] if not bad else [])


def sensitivity(base: Dict[str, Dict[str, object]]) -> str:
    """Re-run every interval with the block length halved and doubled.

    A block bootstrap needs a block length, and no rule picks it from the data.
    So we report how far the endpoints move when the rule is changed by a
    factor of two in either direction, and report the worst case.
    """
    default = TARGET
    alt = {}
    for target in (default * 2, default // 2):   # half and double the block length
        globals()["TARGET"] = target
        alt[target] = {ds: analyse(ds) for ds in FULL}
    globals()["TARGET"] = default

    def endpoints(res):
        """Every interval the paper prints: rates and paired differences."""
        out = {}
        for ds, r in res.items():
            for cond, c in r["conditions"].items():
                out[f"{ds}/{cond}"] = c["block"]
            out[f"{ds}/frame"] = r["frame"]["block"]
            out[f"{ds}/distinct"] = r["distinct"]["block"]
            for cond, c in r["ablation"].items():
                out[f"{ds}/abl/{cond}"] = c["diff_ci"]
        return out

    b = endpoints(base)
    worst = max(abs(v[i] - b[k][i]) * 100.0
                for res in alt.values() for k, v in endpoints(res).items()
                if k in b for i in (0, 1))
    return "\n".join([
        f"% Block-length sensitivity: rule re-run at {default // 2} and {default * 2} target blocks.",
        f"\\newcommand{{\\fbcBlockTarget}}{{{default}}}",
        f"\\newcommand{{\\fbcBlockTargetHalf}}{{{default // 2}}}",
        f"\\newcommand{{\\fbcBlockTargetDouble}}{{{default * 2}}}",
        f"\\newcommand{{\\fbcBlockShiftMax}}{{{worst:.1f}}}",
        ""])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json-out", type=Path, default=Path(f"{R}/v3/forge_block_ci.json"))
    ap.add_argument("--tex-out", type=Path, default=Path(f"{R}/v3/forge_block_ci_numbers.tex"))
    ap.add_argument("--check", type=Path, nargs="*", default=[], help="v3_numbers.tex-style files to compare against")
    ap.add_argument("--block-target", type=int, default=TARGET,
                    help="target number of blocks; halve or double it for the block-length sensitivity check")
    ap.add_argument("--sensitivity", action="store_true",
                    help="also re-run every interval at half and double the block length")
    a = ap.parse_args()
    globals()["TARGET"] = a.block_target
    res = {ds: analyse(ds) for ds in FULL}
    a.json_out.write_text(json.dumps(res, indent=2))
    tex = macros(res)
    if a.sensitivity:
        tex += "\n" + sensitivity(res)
    a.tex_out.write_text(tex)
    for ds, r in res.items():
        p, f, ab = r["conditions"]["pooled"], r["frame"], r["ablation"]["pooled"]
        print(f"{ds:10s} pooled {p['accepted']}/{p['n']} = {pct(p['rate'])} wilson [{pct(p['wilson'][0])}, {pct(p['wilson'][1])}] "
              f"block [{pct(p['block'][0])}, {pct(p['block'][1])}] blocks={p['n_blocks']} len={p['block_len_rows']}")
        print(f"{'':10s} per-cond " + " ".join(f"{CONDS[c]}={pct(r['conditions'][c]['rate'])}[{pct(r['conditions'][c]['block'][0])},{pct(r['conditions'][c]['block'][1])}] len={r['conditions'][c]['block_len_rows']}" for c in CONDS))
        d = r["distinct"]
        print(f"{'':10s} distinct={r['distinct_trials']} {d['accepted']}/{d['n']} = {pct(d['rate'])} block [{pct(d['block'][0])}, {pct(d['block'][1])}]")
        print(f"{'':10s} frame {f['accepted']}/{f['n']} = {pct(f['rate'])} block [{pct(f['block'][0])}, {pct(f['block'][1])}]")
        print(f"{'':10s} ablation pooled n={ab['n_frames']} full={pct(ab['full_rate'])} off={pct(ab['off_rate'])} "
              f"diff={pct(ab['diff'])} [{pct(ab['diff_ci'][0])}, {pct(ab['diff_ci'][1])}] ratio={ab['ratio']:.1f} [{ab['ratio_ci'][0]:.1f}, {ab['ratio_ci'][1]:.1f}]")
    for tex in a.check:
        bad = check(res, tex)
        ok = len(bad) == 1 and bad[0].startswith("(")
        print(f"check {tex}: " + (f"all point estimates match {bad[0]}" if ok else "MISMATCH\n  " + "\n  ".join(bad)))
    print(f"wrote {a.json_out}\nwrote {a.tex_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
