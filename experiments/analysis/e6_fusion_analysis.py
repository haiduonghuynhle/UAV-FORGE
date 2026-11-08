#!/usr/bin/env python3
"""E6: paired comparisons of injection density and fusion (pre-registered,
amendment 1 of 2026-10-25).

Density: trials forced in the sparse arm (inject rate 0.2) against the same
trials (condition, frame, forced tile) in the dense E1b run. Fusion: E1b
attacked trials with fusion off against E1b. A frame is accepted if any of its
paired trials is; it is unsafe if any accepted trial has error > R. The
difference is paired per frame; its interval resamples contiguous blocks of
frames in name order (VPAIR names follow the flight), about 25 blocks, with
half and double block length as a sensitivity, 2000 resamples.

Runs pair only on identical trials: every sparse or no-fusion trial must exist
in the dense run with the same forced tile, and the no-fusion arm must force
exactly the dense trials. The script stops otherwise; it never pairs on an
intersection. Dense repeats (E2b) give the same paired difference between two
runs of one arm, as a reference for run-to-run variation. Accepted trials with
no finite error count as unknown, never as safe.

Usage: e6_fusion_analysis.py OUT.json --dense N,F,S --sparse N,F,S
           --nofgo N,F,S [--dense-repeat N,F,S ...] --dense-clean DIR
           --nofgo-clean DIR [--radius 20]
"""
import argparse, json, math, random, statistics, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "reporting"))
import gen_v3_tables as G

CONDS = ("wrong_near", "wrong_far", "visually_similar_wrong")


def _err(r, k):
    try:
        e = float(r[k])
    except (KeyError, TypeError, ValueError):
        return float("nan")
    return e if math.isfinite(e) else float("nan")


def arm(spec):
    """(cond, frame) -> trial dict, for the three attack cells of one arm."""
    dirs = spec.split(",")
    if len(dirs) != len(CONDS):
        raise SystemExit(f"expected {len(CONDS)} cell dirs, got {len(dirs)}: {spec}")
    out = {}
    for cond, d in zip(CONDS, dirs):
        for r in G._attacked_rows(d):
            key = (cond, r["image_name"])
            if key in out:
                raise SystemExit(f"duplicate trial {key} in {d}")
            tile = str(r.get("adversarial_forced_reference_name", "") or "")
            if not tile:
                raise SystemExit(f"trial {key} in {d} has no forced tile")
            out[key] = {"tile": tile, "acc": G._truthy(r.get("accepted_for_metrics")),
                        "err": _err(r, "post_fgo_error_m"), "pre": _err(r, "pre_fgo_error_m")}
    return out


def check_pairs(a, b, name, exact):
    """Every trial of a exists in b with the same tile; exact: same trial set."""
    missing = [k for k in a if k not in b]
    tiles = [k for k in a if k in b and a[k]["tile"] != b[k]["tile"]]
    extra = [k for k in b if k not in a] if exact else []
    if missing or tiles or extra:
        raise SystemExit(f"{name}: {len(missing)} trials missing in dense, {len(tiles)} tile "
                         f"mismatches, {len(extra)} dense trials not forced; refusing to pair "
                         f"(first: {(missing + tiles + extra)[:3]})")


def frame_outcomes(trials, keys, radius):
    """frame -> (accepted, unsafe, unknown) over the given trial keys."""
    by = {}
    for k in keys:
        t = trials[k]
        acc, unsafe, unknown = by.get(k[1], (False, False, False))
        if t["acc"]:
            acc = True
            if math.isnan(t["err"]):
                unknown = True
            elif t["err"] > radius:
                unsafe = True
        by[k[1]] = (acc, unsafe, unknown)
    return by


def block_ci(x, y, size, draws=2000, seed=0):
    n = len(x)
    blocks = [list(range(i, min(i + size, n))) for i in range(0, n, size)]
    rng = random.Random(seed); diffs = []
    for _ in range(draws):
        idx = [i for blk in (rng.choice(blocks) for _ in blocks) for i in blk]
        diffs.append((sum(x[i] for i in idx) - sum(y[i] for i in idx)) / len(idx))
    diffs.sort()
    return [diffs[int(0.025 * draws)], diffs[int(0.975 * draws) - 1]]


def paired(a, b, keys, radius):
    fa, fb = frame_outcomes(a, keys, radius), frame_outcomes(b, keys, radius)
    frames = sorted(fa)
    n = len(frames); size = max(1, round(n / 25))
    res = {"frames": n, "trials": len(keys), "block_len": size}
    for j, name in ((0, "accepted"), (1, "unsafe")):
        x = [int(fa[f][j]) for f in frames]; y = [int(fb[f][j]) for f in frames]
        res[name] = {"rate_a": sum(x) / n, "rate_b": sum(y) / n, "diff": (sum(x) - sum(y)) / n,
                     "diff_ci": block_ci(x, y, size),
                     "diff_ci_half_block": block_ci(x, y, max(1, size // 2)),
                     "diff_ci_double_block": block_ci(x, y, 2 * size),
                     "a_only": sum(1 for i in range(n) if x[i] and not y[i]),
                     "b_only": sum(1 for i in range(n) if y[i] and not x[i])}
    res["unknown_error_frames"] = [sum(fa[f][2] for f in frames), sum(fb[f][2] for f in frames)]
    return res


def summary(t, keys, radius):
    acc = [t[k] for k in keys if t[k]["acc"]]
    errs = [x["err"] for x in acc if not math.isnan(x["err"])]
    mis = sum(1 for e in errs if e > radius)
    return {"trials": len(keys), "accepted": len(acc), "misleading": mis,
            "unknown_error": len(acc) - len(errs),
            "misleading_share": mis / len(acc) if acc else None,
            "median_error_m": statistics.median(errs) if errs else None}


def same_trial_errors(dense, nofgo, keys):
    """Fusion's effect on error, on trials where both errors exist.

    Dense pre-fusion vs post-fusion error on the same trial (no selection),
    and the no-fusion error on trials accepted in both arms (descriptive).
    """
    pp = [(dense[k]["pre"], dense[k]["err"]) for k in keys
          if not math.isnan(dense[k]["pre"]) and not math.isnan(dense[k]["err"])]
    both = [(nofgo[k]["err"], dense[k]["err"]) for k in keys if dense[k]["acc"] and nofgo[k]["acc"]
            and not math.isnan(dense[k]["err"]) and not math.isnan(nofgo[k]["err"])]
    d = [post - pre for pre, post in pp]
    return {"dense_pre_post_trials": len(pp),
            "dense_median_post_minus_pre_m": statistics.median(d) if d else None,
            "dense_share_post_below_pre": sum(x < 0 for x in d) / len(d) if d else None,
            "accepted_in_both": len(both),
            "median_nofgo_minus_dense_m": statistics.median([a - b for a, b in both]) if both else None}


def coverage(d):
    rows = G._load_predictions(d)
    names = [r["image_name"] for r in rows]
    if len(names) != len(set(names)):
        raise SystemExit(f"duplicate frames in {d}")
    return {r["image_name"]: G._truthy(r.get("accepted_for_metrics")) for r in rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    for k in ("dense", "sparse", "nofgo"):
        ap.add_argument(f"--{k}", required=True)
    ap.add_argument("--dense-repeat", action="append", default=[])
    ap.add_argument("--dense-clean", required=True); ap.add_argument("--nofgo-clean", required=True)
    ap.add_argument("--radius", type=float, default=20.0)
    a = ap.parse_args()
    R = a.radius
    dense, sparse, nofgo = arm(a.dense), arm(a.sparse), arm(a.nofgo)
    check_pairs(sparse, dense, "sparse", exact=False)
    check_pairs(nofgo, dense, "nofgo", exact=True)
    hk, fk = sorted(sparse), sorted(dense)
    res = {"radius": R,
           "density": {"paired": paired(sparse, dense, hk, R),
                       "sparse": summary(sparse, hk, R), "dense": summary(dense, hk, R)},
           "fusion": {"paired": paired(nofgo, dense, fk, R),
                      "nofgo": summary(nofgo, fk, R), "dense": summary(dense, fk, R),
                      "errors": same_trial_errors(dense, nofgo, fk)},
           "dense_repeats": []}
    for spec in a.dense_repeat:
        rep = arm(spec)
        check_pairs(rep, dense, "dense-repeat", exact=True)
        res["dense_repeats"].append({"dirs": spec, "on_density_trials": paired(rep, dense, hk, R),
                                     "on_fusion_trials": paired(rep, dense, fk, R)})
    cf, cn = coverage(a.dense_clean), coverage(a.nofgo_clean)
    if set(cf) != set(cn):
        raise SystemExit("clean cells cover different frames")
    res["fusion"]["clean_coverage"] = {"frames": len(cf), "fgo": sum(cf.values()) / len(cf),
                                       "nofgo": sum(cn.values()) / len(cn),
                                       "fgo_only": sum(cf[f] and not cn[f] for f in cf),
                                       "nofgo_only": sum(cn[f] and not cf[f] for f in cf)}
    json.dump(res, open(a.out, "w"), indent=1)
    for k in ("density", "fusion"):
        p = res[k]["paired"]
        for m in ("accepted", "unsafe"):
            q = p[m]
            print(k, m, "frames", p["frames"], "rate %.1f%% vs %.1f%%" % (100 * q["rate_a"], 100 * q["rate_b"]),
                  "diff %.1f [%.1f, %.1f] pts" % (100 * q["diff"], 100 * q["diff_ci"][0], 100 * q["diff_ci"][1]),
                  "discordant", q["a_only"], q["b_only"])
    for r in res["dense_repeats"]:
        q = r["on_density_trials"]["accepted"]
        print("dense repeat on density trials: diff %.1f [%.1f, %.1f] pts" % (
            100 * q["diff"], 100 * q["diff_ci"][0], 100 * q["diff_ci"][1]))
    print("fusion errors", res["fusion"]["errors"], "coverage", res["fusion"]["clean_coverage"])


if __name__ == "__main__":
    main()
