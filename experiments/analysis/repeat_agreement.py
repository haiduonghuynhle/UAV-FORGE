#!/usr/bin/env python3
"""E2b: run-to-run agreement of a held-out FORGE campaign (amendment 1 of
2026-10-25).

Each run is three attack cells (wrong_near, wrong_far, visually_similar_wrong)
on the same frames and forced tiles. The population is the one of record
(gen_v3_tables._attacked_rows, as in heldout_analysis.py); --genuine-only
restricts it to _is_genuine_fault trials as a sensitivity. For each run:
frame-level acceptance (a frame is accepted if any of its trials is), the
unsafe-frame rate (any accepted trial with error > R), pooled trial rate and
the misleading share of accepted trials. Accepted trials with no finite error
count as unknown, never as safe. Across runs: range and SD of the rates,
Fleiss' kappa and mean pairwise Jaccard of accepted frames, and the frames
accepted in all runs against any run. Runs pair only if they attacked the same
trials with the same forced tiles; the script stops otherwise. --expect-run0
A/N stops unless the first run reproduces the reported count.

Usage: repeat_agreement.py --dataset vpair|uav_visloc --radius R OUT.json \
           [--expect-run0 A/N] [--genuine-only] NAME=NEAR,FAR,SIM [NAME=...]
"""
import argparse, itertools, json, math, statistics, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "reporting"))
import gen_v3_tables as G

CONDS = ("wrong_near", "wrong_far", "visually_similar_wrong")


def run_cells(dirs, ds, genuine_only):
    """cond -> frame -> (tile, accepted, error)."""
    if len(dirs) != len(CONDS):
        raise SystemExit(f"expected {len(CONDS)} cell dirs, got {len(dirs)}")
    out = {}
    for cond, d in zip(CONDS, dirs):
        rows = {}
        for r in G._attacked_rows(d):
            if genuine_only and not G._is_genuine_fault(r, ds):
                continue
            if r["image_name"] in rows:
                raise SystemExit(f"duplicate frame {r['image_name']} in {d}")
            try:
                e = float(r["post_fgo_error_m"])
            except (KeyError, TypeError, ValueError):
                e = float("nan")
            rows[r["image_name"]] = (r.get("adversarial_forced_reference_name", ""),
                                     G._truthy(r.get("accepted_for_metrics")),
                                     e if math.isfinite(e) else float("nan"))
        out[cond] = rows
    return out


def fleiss(rows):
    n, N = len(rows[0]), len(rows)
    p1 = sum(sum(r) for r in rows) / (N * n)
    Pi = [(sum(r) ** 2 + (n - sum(r)) ** 2 - n) / (n * (n - 1)) for r in rows]
    Pe = p1 ** 2 + (1 - p1) ** 2
    return None if Pe == 1 else (sum(Pi) / N - Pe) / (1 - Pe)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True); ap.add_argument("--radius", type=float, required=True)
    ap.add_argument("--expect-run0"); ap.add_argument("--genuine-only", action="store_true")
    ap.add_argument("out"); ap.add_argument("runs", nargs="+")
    a = ap.parse_args()
    runs = {}
    for spec in a.runs:
        name, dirs = spec.split("=", 1)
        if name in runs:
            raise SystemExit(f"duplicate run name {name}")
        runs[name] = run_cells(dirs.split(","), a.dataset, a.genuine_only)
    names = list(runs); ref = names[0]
    for n in names[1:]:
        for c in CONDS:
            if set(runs[n][c]) != set(runs[ref][c]) or any(runs[n][c][f][0] != runs[ref][c][f][0] for f in runs[ref][c]):
                raise SystemExit(f"{n}/{c}: different attacked frames or forced tiles than {ref}; refusing to pair")
    frames = sorted(set().union(*(runs[ref][c] for c in CONDS)))
    if not frames:
        raise SystemExit("no attacked frames")
    acc, unsafe = {}, {}
    for n in names:
        tr = {f: [runs[n][c][f] for c in CONDS if f in runs[n][c]] for f in frames}
        acc[n] = {f for f in frames if any(t[1] for t in tr[f])}
        unsafe[n] = {f for f in frames if any(t[1] and t[2] > a.radius for t in tr[f])}
    if a.expect_run0:
        k, N = (int(x) for x in a.expect_run0.split("/"))
        if (len(acc[ref]), len(frames)) != (k, N):
            raise SystemExit(f"run 0 gives {len(acc[ref])}/{len(frames)}, expected {k}/{N}")
    res = {"runs": names, "population": "genuine_only" if a.genuine_only else "attacked_rows",
           "attacked_frames": len(frames), "per_run": {}}
    for n in names:
        tr = [v for c in CONDS for v in runs[n][c].values()]
        k = sum(v[1] for v in tr)
        m = sum(1 for v in tr if v[1] and v[2] > a.radius)
        u = sum(1 for v in tr if v[1] and math.isnan(v[2]))
        res["per_run"][n] = {"frame_rate": len(acc[n]) / len(frames), "frames_accepted": len(acc[n]),
                             "unsafe_frame_rate": len(unsafe[n]) / len(frames),
                             "trials": len(tr), "trial_rate": k / len(tr), "accepted_trials": k,
                             "misleading_trials": m, "unknown_error_trials": u,
                             "misleading_share": m / k if k else None}
    fr = [res["per_run"][n]["frame_rate"] for n in names]
    res["frame_rate_range"] = [min(fr), max(fr)]
    res["frame_rate_sd"] = statistics.stdev(fr) if len(fr) > 1 else 0.0
    res["fleiss_kappa"] = fleiss([[int(f in acc[n]) for n in names] for f in frames]) if len(names) > 1 else None
    pairs = list(itertools.combinations(names, 2))
    res["jaccard_mean"] = statistics.mean(len(acc[x] & acc[y]) / len(acc[x] | acc[y]) if acc[x] | acc[y] else 1.0
                                          for x, y in pairs) if pairs else None
    res["accepted_in_all"] = len(set.intersection(*acc.values()))
    res["accepted_in_any"] = len(set.union(*acc.values()))
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "per_run"}),
          {n: round(100 * v["frame_rate"], 2) for n, v in res["per_run"].items()})


if __name__ == "__main__":
    main()
