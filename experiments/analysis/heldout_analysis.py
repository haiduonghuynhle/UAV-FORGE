#!/usr/bin/env python3
"""E1/E1b analysis, as pre-registered.

Primary: frame-level acceptance of the three-condition attack suite. A frame
is attacked if at least one condition forced a valid wrong tile onto it
(gen_v3_tables._attacked_rows, the paper's one definition); it counts as
accepted if any attacked condition was accepted. Clusters are maps (flights);
uncertainty is a two-level bootstrap (resample clusters, then contiguous
blocks of frames inside each), with the block length halved and doubled.

Secondary: pooled trial rate per condition, unsafe share of accepted trials,
median error of accepted forced fixes, accepted trials inside R (inconsistent
with the label), and every rate again on the subset of trials whose forced
tile stays disjoint under a conservative footprint envelope.

Usage:
  e1_analysis.py --cond wrong_near=DIR[,DIR] --cond wrong_far=... \
      --cond visually_similar_wrong=... --dataset uav_visloc --radius 150 \
      --env-ref 277.9 305.8 --env-query 120 --out OUT.json [--macros OUT.tex --prefix eOne]
      [--query-table footprint_01.csv,footprint_04.csv,...]

--query-table relabels trials with the validated per-frame query bound (E5b,
amendment 2): a trial is kept only if its frame has an OK bound and the
forced tile stays disjoint under it. The reference half-extent is the per-map
value of --ref-per-map (the one injection used), else --env-ref; the largest
envelope (--env-ref) is also reported as a sensitivity.
"""
import argparse, json, math, random, statistics, sys
from collections import defaultdict
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2] / "scripts" / "reporting"))
import gen_v3_tables as G

def cluster_of(name):
    return name.split("/")[0] if "/" in name else "route"

def err(r):
    try: return float(r["post_fgo_error_m"])
    except (TypeError, ValueError, KeyError): return float("nan")

def disjoint_under(r, ref_half, q_half, slack=15.0):
    a = G._anchor_latlon(r)
    if a is None: return False
    de, dn = G._per_axis_m(float(r["gt_lat"]), float(r["gt_lon"]), a[0], a[1])
    return de > ref_half[0] + q_half + slack or dn > ref_half[1] + q_half + slack

def load_query_table(paths):
    import csv
    q = {}
    for p in paths:
        for row in csv.DictReader(open(p, newline="")):
            if row.get("status") == "OK":
                q[row["image_name"]] = float(row["query_half_bound_m"])
    return q

def frame_table(cells, keep=lambda r: True):
    """frame -> {cond: accepted bool} for attacked trials passing keep()."""
    t = defaultdict(dict)
    for cond, dirs in cells.items():
        for d in dirs:
            for r in G._attacked_rows(d):
                if keep(r):
                    t[r["image_name"]][cond] = G._truthy(r.get("accepted_for_metrics"))
    return t

def union_by_cluster(table):
    by = defaultdict(list)
    for f in sorted(table):
        by[cluster_of(f)].append(1 if any(table[f].values()) else 0)
    return by

def boot(by, block, draws=2000, seed=0):
    rng = random.Random(seed); cl = sorted(by); out = []
    for _ in range(draws):
        num = den = 0
        for c in (rng.choice(cl) for _ in cl):
            x = by[c]; blocks = [x[i:i + block] for i in range(0, len(x), block)]
            for b in (rng.choice(blocks) for _ in blocks):
                num += sum(b); den += len(b)
        out.append(num / den if den else 0.0)
    out.sort(); return out[int(0.025 * draws)], out[int(0.975 * draws) - 1]

def summarise(cells, radius, keep=lambda r: True, block=20):
    table = frame_table(cells, keep); by = union_by_cluster(table)
    n = sum(len(v) for v in by.values()); k = sum(sum(v) for v in by.values())
    if block <= 0:
        # The paper's rule: about 25 blocks per campaign.
        block = max(2, round(n / 25))
    res = {"attacked_frames": n, "accepted_frames": k, "rate": k / n if n else None, "block_len": block,
           "per_cluster": {c: {"n": len(v), "rate": sum(v) / len(v)} for c, v in by.items()}}
    if n:
        res["ci"] = boot(by, block); res["ci_half_block"] = boot(by, max(2, block // 2)); res["ci_double_block"] = boot(by, block * 2)
    trials = defaultdict(lambda: [0, 0, 0, 0]); errs = []
    for cond, dirs in cells.items():
        for d in dirs:
            for r in G._attacked_rows(d):
                if not keep(r): continue
                t = trials[cond]; t[0] += 1
                if G._truthy(r.get("accepted_for_metrics")):
                    t[1] += 1; e = err(r)
                    if e > radius: t[2] += 1; errs.append(e)
                    elif e == e: t[3] += 1
    res["per_condition"] = {c: {"trials": v[0], "accepted": v[1], "unsafe": v[2], "inside_R": v[3],
                                "rate": v[1] / v[0] if v[0] else None} for c, v in trials.items()}
    res["median_unsafe_error_m"] = statistics.median(errs) if errs else None
    return res

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cond", action="append", required=True)
    ap.add_argument("--dataset", required=True); ap.add_argument("--radius", type=float, required=True)
    ap.add_argument("--env-ref", type=float, nargs=2); ap.add_argument("--env-query", type=float)
    ap.add_argument("--query-table", default=None)
    ap.add_argument("--ref-per-map", default=None, help="JSON {map: [east_m, north_m]}")
    ap.add_argument("--query-scales", default="1.0,1.15,1.30",
                    help="relabel with the per-frame bound times each factor (amendment 2)")
    ap.add_argument("--block", type=int, default=20); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.query_table and not a.ref_per_map:
        ap.error("--query-table needs --ref-per-map: the primary relabel must use the "
                 "per-map reference extents the injection used")
    cells = {c.split("=")[0]: c.split("=")[1].split(",") for c in a.cond}
    out = {"primary": summarise(cells, a.radius, block=a.block)}
    if a.env_ref and a.env_query is not None:
        keep = lambda r: disjoint_under(r, a.env_ref, a.env_query)
        out["envelope"] = {"ref_half_m": a.env_ref, "query_half_m": a.env_query,
                           **summarise(cells, a.radius, keep=keep, block=a.block)}
    if a.query_table and a.env_ref:
        qt = load_query_table(a.query_table.split(","))
        per_map = {m: tuple(v) for m, v in json.load(open(a.ref_per_map)).items()} if a.ref_per_map else {}
        missing = sorted({cluster_of(n) for n in qt} - set(per_map))
        if missing:
            raise SystemExit(f"--ref-per-map has no extent for maps {missing}")
        ref_of = lambda r: per_map[cluster_of(r["image_name"])]
        out["per_frame_table"] = {}
        variants = [(f"x{k:.2f}", k, ref_of) for k in (float(x) for x in a.query_scales.split(","))]
        variants.append(("x1.00_max_ref", 1.0, lambda r: tuple(a.env_ref)))
        for key, k, ref in variants:
            keep_t = (lambda k, ref: lambda r: r["image_name"] in qt
                      and disjoint_under(r, ref(r), k * qt[r["image_name"]]))(k, ref)
            out["per_frame_table"][key] = {"ref": "per_map" if ref is ref_of and per_map else list(a.env_ref),
                                           "scale": k, "frames_with_bound": len(qt),
                                           **summarise(cells, a.radius, keep=keep_t, block=a.block)}
    if a.query_table and a.env_ref:
        # Where each attacked trial falls under the certified bound (scale 1).
        split = {}
        for cond, dirs in cells.items():
            c = {k: {"trials": 0, "accepted": 0, "accepted_inside_R": 0}
                 for k in ("no_bound", "bound_overlap", "bound_disjoint")}
            for d in dirs:
                for r in G._attacked_rows(d):
                    n = r["image_name"]
                    k = ("no_bound" if n not in qt else
                         "bound_disjoint" if disjoint_under(r, ref_of(r), qt[n]) else "bound_overlap")
                    acc = G._truthy(r.get("accepted_for_metrics")); e = err(r)
                    c[k]["trials"] += 1; c[k]["accepted"] += int(acc)
                    c[k]["accepted_inside_R"] += int(acc and e == e and e <= a.radius)
            split[cond] = c
        out["per_frame_table_split"] = split
    json.dump(out, open(a.out, "w"), indent=1, default=float)
    p = out["primary"]
    print("PRIMARY frame-level", p["accepted_frames"], "/", p["attacked_frames"], "=", round(100 * p["rate"], 2),
          "% CI", [round(100 * x, 2) for x in p["ci"]], "half", [round(100 * x, 2) for x in p["ci_half_block"]],
          "double", [round(100 * x, 2) for x in p["ci_double_block"]])
    for c, v in p["per_cluster"].items(): print("  cluster", c, v["n"], round(100 * v["rate"], 2))
    for c, v in p["per_condition"].items(): print("  cond", c, v)
    if "envelope" in out:
        e = out["envelope"]; print("ENVELOPE kept frames", e["attacked_frames"], "rate", round(100 * e["rate"], 2) if e["rate"] is not None else None,
                                   "CI", [round(100 * x, 2) for x in e.get("ci", [0, 0])])
    for key, t in out.get("per_frame_table", {}).items():
        print("PER-FRAME TABLE", key, "kept frames", t["attacked_frames"], "rate",
              round(100 * t["rate"], 2) if t["rate"] is not None else None,
              "CI", [round(100 * x, 2) for x in t.get("ci", [0, 0])])
