#!/usr/bin/env python3
"""Clean-run coverage of the held-out UAV-VisLoc flights (E1 controls).

For each flight's clean cell: share of frames the pipeline accepts, and
share accepted with error beyond R (150 m), over all frames and over the
frames with a certified footprint bound. Pairs with the attack rate, as the
paper recommends: acceptance under forced wrong references only means
something next to coverage on the same frames.

Usage: heldout_clean_coverage.py RESULTS_ROOT FOOTPRINT_DIR FLIGHTS OUT.json
"""
import csv, glob, json, sys

R_M = 150.0


def main(root, fp_dir, flights, out):
    res = {"flights": {}, "radius_m": R_M}
    tot = {"frames": 0, "accepted": 0, "unsafe": 0, "cert": 0, "cert_accepted": 0, "cert_unsafe": 0}
    for f in flights.split(","):
        d = sorted(glob.glob(f"{root}/f{f}_clean/*/*/2*"))[-1]
        rows = list(csv.DictReader(open(d + "/predictions.csv")))
        ok = {r["image_name"] for r in csv.DictReader(open(f"{fp_dir}/footprint_{f}.csv")) if r["status"] == "OK"}
        acc = [r for r in rows if r["accepted_for_metrics"] == "True"]
        uns = [r for r in acc if r["post_fgo_error_m"] and float(r["post_fgo_error_m"]) > R_M]
        c = {"frames": len(rows), "accepted": len(acc), "unsafe": len(uns), "cert": len(ok),
             "cert_accepted": sum(r["image_name"] in ok for r in acc),
             "cert_unsafe": sum(r["image_name"] in ok for r in uns), "run_dir": d}
        res["flights"][f] = c
        for k in tot:
            tot[k] += c[k]
    res["total"] = tot
    json.dump(res, open(out, "w"), indent=1)
    print(json.dumps(tot))


if __name__ == "__main__":
    main(*sys.argv[1:5])
