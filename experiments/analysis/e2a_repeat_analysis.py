#!/usr/bin/env python3
"""E2a: run-to-run variation of the record FORGE cells (AnyVisLoc).

Endpoints frozen in experiments/campaigns/e2_repeat_record/README.md:
(1) between-run range and SD of the frame-level and pooled acceptance rates,
next to the record's block-bootstrap interval; (2) decision agreement, as the
mean pairwise Jaccard of accepted frame sets and Fleiss' kappa over frames
attacked in every run.

Runs are paired only if they attacked the same frames with the same forced
tiles; otherwise the script stops (memory: corpus hash before pairing).

Usage: e2a_repeat_analysis.py RECORD_ROOT REPEAT_ROOT OUT.json
  RECORD_ROOT holds forgec_avl_<cond>/, REPEAT_ROOT holds rep<i>_forgec_avl_<cond>/.
"""
import glob, itertools, json, statistics, sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2] / "scripts" / "reporting"))
import gen_v3_tables as G

CONDS = ("wrong_near", "wrong_far", "visually_similar_wrong")


def run_dir(root, name):
    d = sorted(glob.glob(f"{root}/{name}/*/*/2*"))
    if not d:
        raise SystemExit(f"missing cell {root}/{name}")
    return d[-1]


def load(d):
    """frame -> (forced tile, accepted) over attacked trials."""
    return {r["image_name"]: (r.get("adversarial_forced_reference_name", ""),
                              G._truthy(r.get("accepted_for_metrics")),
                              _err(r))
            for r in G._attacked_rows(d)}


def _err(r):
    try:
        return float(r["post_fgo_error_m"])
    except (KeyError, TypeError, ValueError):
        return float("nan")


def fleiss(rows):
    """Fleiss' kappa for binary ratings; rows = list of per-frame rating lists."""
    n = len(rows[0]); N = len(rows)
    if N == 0:
        return None
    p1 = sum(sum(r) for r in rows) / (N * n); p0 = 1 - p1
    Pi = [(sum(r) ** 2 + (n - sum(r)) ** 2 - n) / (n * (n - 1)) for r in rows]
    Pbar = sum(Pi) / N; Pe = p1 ** 2 + p0 ** 2
    return None if Pe == 1 else (Pbar - Pe) / (1 - Pe)


def jaccard(a, b):
    return len(a & b) / len(a | b) if a | b else 1.0


def main(rec_root, rep_root, out):
    runs = {"record": {c: load(run_dir(rec_root, f"forgec_avl_{c}")) for c in CONDS}}
    for i in (1, 2, 3):
        runs[f"rep{i}"] = {c: load(run_dir(rep_root, f"rep{i}_forgec_avl_{c}")) for c in CONDS}
    names = list(runs)
    res = {"runs": names, "conditions": {}, "frame_level": {}}
    for c in CONDS:
        frames = set(runs["record"][c])
        for n in names[1:]:
            if set(runs[n][c]) != frames or any(runs[n][c][f][0] != runs["record"][c][f][0] for f in frames):
                diff = len(set(runs[n][c]) ^ frames)
                tiles = sum(runs[n][c].get(f, ("",))[0] != runs["record"][c][f][0] for f in frames)
                raise SystemExit(f"{c}: {n} attacked a different population "
                                 f"(frame set diff {diff}, forced tile diff {tiles}); refusing to pair")
        acc = {n: {f for f, (_, a, _) in runs[n][c].items() if a} for n in names}
        rates = {n: len(acc[n]) / len(frames) for n in names}
        res["conditions"][c] = {
            "trials": len(frames),
            "accepted": {n: len(acc[n]) for n in names},
            "rate": rates,
            "rate_sd": statistics.stdev(rates.values()),
            "rate_range": [min(rates.values()), max(rates.values())],
            "jaccard_mean": statistics.mean(jaccard(acc[a], acc[b]) for a, b in itertools.combinations(names, 2)),
            "jaccard_min": min(jaccard(acc[a], acc[b]) for a, b in itertools.combinations(names, 2)),
            "accepted_in_all": len(set.intersection(*acc.values())),
            "accepted_in_any": len(set.union(*acc.values())),
            "fleiss_kappa": fleiss([[int(f in acc[n]) for n in names] for f in sorted(frames)]),
            # Accepted forced fixes inside the success radius (20 m on AnyVisLoc).
            "accepted_inside_R": [{"run": n, "frame": f, "error_m": e}
                                  for n in names for f, (_, a, e) in runs[n][c].items() if a and e <= 20.0],
        }
    # Frame-level union over the three conditions, per run.
    frames = set.union(*(set(runs["record"][c]) for c in CONDS))
    union = {n: {f for f in frames if any(runs[n][c].get(f, ("", False))[1] for c in CONDS)} for n in names}
    fr = {n: len(union[n]) / len(frames) for n in names}
    res["frame_level"] = {
        "attacked_frames": len(frames),
        "accepted": {n: len(union[n]) for n in names},
        "rate": fr, "rate_sd": statistics.stdev(fr.values()),
        "rate_range": [min(fr.values()), max(fr.values())],
        "jaccard_mean": statistics.mean(jaccard(union[a], union[b]) for a, b in itertools.combinations(names, 2)),
        "fleiss_kappa": fleiss([[int(f in union[n]) for n in names] for f in sorted(frames)]),
        "accepted_in_all": len(set.intersection(*union.values())),
        "accepted_in_any": len(set.union(*union.values())),
    }
    json.dump(res, open(out, "w"), indent=1)
    fl = res["frame_level"]
    print("FRAME-LEVEL", fl["accepted"], "rates", {k: round(100 * v, 2) for k, v in fl["rate"].items()},
          "SD %.2f pts" % (100 * fl["rate_sd"]), "Jaccard %.2f" % fl["jaccard_mean"],
          "kappa", None if fl["fleiss_kappa"] is None else round(fl["fleiss_kappa"], 3),
          "all/any", fl["accepted_in_all"], fl["accepted_in_any"])
    for c, v in res["conditions"].items():
        print(c, v["accepted"], "SD %.2f pts" % (100 * v["rate_sd"]), "Jaccard %.2f" % v["jaccard_mean"],
              "kappa", None if v["fleiss_kappa"] is None else round(v["fleiss_kappa"], 3))


if __name__ == "__main__":
    main(*sys.argv[1:4])
