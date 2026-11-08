#!/usr/bin/env python3
"""E5b: per-frame camera footprint for UAV-VisLoc, validated, outcome-free.

The ground sampling distance (GSD) of a frame is measured from its motion
against neighbouring frames: a homography between frame t and t+L (L = 1..3)
moves the image centre by d pixels while the ground truth moves D metres, so
GSD = D / d. A pair counts only if it passes every check below, each aimed at
a known bias of this estimate:

  scale     both singular values of A    altitude change, and anisotropic
            within 1 +- 0.03             stretch or shear (a det-only test
                                         passes diag(2, 0.5))
  altitude  |dh| / h <= 0.01 (GT)         same, from the trajectory
  turn      |rotation of A| <= 5 deg      pitch/roll and principal-point offsets
  tilt      |h31|, |h32| <= 2e-4 per     projective terms: tilt or parallax
            full-resolution pixel
  support   >= 60 RANSAC inliers, inlier ratio >= 0.5
  baseline  D >= 30 m                     ratio unstable on short baselines

A pair that passes has the same GSD at both ends to within 3%, so its
estimate counts for both frames (neighbours on either side, up to 3 apart).
Per frame: median GSD over its accepted pairs; dispersion = MAD / median.
A frame is INDETERMINATE if it has fewer than 3 accepted estimates or a
dispersion above 0.10. Otherwise its footprint half-extents are a = GSD*W/2,
b = GSD*H/2, and the heading-free per-axis bound is sqrt(a^2 + b^2), the
largest E or N half-extent the rectangle can take under any heading, raised
by a margin max(0.10, 2 * dispersion). The table never reads a localization
result, so it can define fault labels before a campaign runs.

Usage: e5b_footprint_table.py <dataset_root> <flights,comma> <out_dir> [workers]
"""
import csv, json, math, statistics, sys
from multiprocessing import Pool
from pathlib import Path
import cv2, numpy as np

SCALE = 0.25
LAGS = (1, 2, 3)
CHECK = dict(scale=0.03, alt=0.01, turn_deg=5.0, tilt=2e-4, inliers=60, inlier_ratio=0.5, baseline_m=30.0)
MIN_EST, MAX_DISP = 3, 0.10

def features(path):
    im = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    H0, W0 = im.shape
    im = cv2.resize(im, None, fx=SCALE, fy=SCALE)
    k, d = cv2.SIFT_create(3000).detectAndCompute(im, None)
    return np.float32([p.pt for p in k]), d, W0, H0, im.shape

def pair_gsd(fa, fb, ra, rb):
    pa, da, W0, H0, shp = fa; pb, db = fb[0], fb[1]
    if da is None or db is None or len(pa) < 100 or len(pb) < 100:
        return None, "features"
    m = cv2.BFMatcher().knnMatch(db, da, k=2)
    good = [x for x, y in (q for q in m if len(q) == 2) if x.distance < 0.75 * y.distance]
    if len(good) < CHECK["inliers"]:
        return None, "support"
    src = pb[[g.queryIdx for g in good]]; dst = pa[[g.trainIdx for g in good]]
    H, inl = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
    if H is None:
        return None, "support"
    n_in = int(inl.sum())
    if n_in < CHECK["inliers"] or n_in / len(good) < CHECK["inlier_ratio"]:
        return None, "support"
    H = H / H[2, 2]
    A = H[:2, :2]
    sv = np.linalg.svd(A, compute_uv=False)
    if np.linalg.det(A) <= 0 or max(abs(sv[0] - 1.0), abs(sv[1] - 1.0)) > CHECK["scale"]:
        return None, "scale"
    if abs(math.degrees(math.atan2(A[1, 0] - A[0, 1], A[0, 0] + A[1, 1]))) > CHECK["turn_deg"]:
        return None, "turn"
    # H is fitted on images downscaled by SCALE, so its projective terms are
    # per downscaled pixel; per full-resolution pixel they are SCALE times smaller.
    if max(abs(H[2, 0]), abs(H[2, 1])) * SCALE > CHECK["tilt"]:
        return None, "tilt"
    ha, hb = float(ra["height"]), float(rb["height"])
    if abs(hb - ha) / max(ha, 1.0) > CHECK["alt"]:
        return None, "alt"
    c = np.array([[[shp[1] / 2, shp[0] / 2]]], np.float32)
    ca = cv2.perspectiveTransform(c, H)[0, 0]
    dpx = math.hypot(ca[0] - shp[1] / 2, ca[1] - shp[0] / 2) / SCALE
    kx = 111320.0 * math.cos(math.radians(float(ra["lat"])))
    D = math.hypot((float(rb["lon"]) - float(ra["lon"])) * kx, (float(rb["lat"]) - float(ra["lat"])) * 111320.0)
    if D < CHECK["baseline_m"] or dpx < 20:
        return None, "baseline"
    return D / dpx, "ok"

def flight_job(args):
    root, loc = args
    rows = sorted(csv.DictReader(open(root / loc / f"{loc}.csv")), key=lambda r: r["filename"])
    feats = [features(root / loc / "drone" / r["filename"]) for r in rows]
    est = {i: [] for i in range(len(rows))}
    reasons = {}
    for L in LAGS:
        for i in range(len(rows) - L):
            g, why = pair_gsd(feats[i], feats[i + L], rows[i], rows[i + L])
            reasons[why] = reasons.get(why, 0) + 1
            if g is not None:
                est[i].append(g); est[i + L].append(g)
    W0, H0 = feats[0][2], feats[0][3]
    out = []
    for i, r in enumerate(rows):
        e = est[i]
        name = f"{loc}/{r['filename']}"
        if len(e) < MIN_EST:
            out.append((name, "INDETERMINATE_FEW", None, None, len(e), None)); continue
        med = statistics.median(e)
        disp = statistics.median([abs(x - med) for x in e]) / med
        if disp > MAX_DISP:
            out.append((name, "INDETERMINATE_DISPERSED", med, None, len(e), disp)); continue
        a, b = med * W0 / 2, med * H0 / 2
        q = math.hypot(a, b) * (1.0 + max(0.10, 2 * disp))
        out.append((name, "OK", med, q, len(e), disp))
    return loc, out, reasons

if __name__ == "__main__":
    root, flights, outd = Path(sys.argv[1]), sys.argv[2].split(","), Path(sys.argv[3])
    workers = int(sys.argv[4]) if len(sys.argv) > 4 else 4
    outd.mkdir(parents=True, exist_ok=True)
    summary = {"checks": CHECK, "min_estimates": MIN_EST, "max_dispersion": MAX_DISP, "lags": LAGS, "flights": {}}
    with Pool(workers) as p:
        for loc, out, reasons in p.imap_unordered(flight_job, [(root, f) for f in flights]):
            with open(outd / f"footprint_{loc}.csv", "w", newline="") as fh:
                w = csv.writer(fh); w.writerow(["image_name", "status", "gsd_m_per_px", "query_half_bound_m", "n_estimates", "dispersion"])
                w.writerows(out)
            ok = [r for r in out if r[1] == "OK"]
            qs = sorted(r[3] for r in ok)
            summary["flights"][loc] = {
                "frames": len(out), "ok": len(ok),
                "indeterminate_few": sum(r[1] == "INDETERMINATE_FEW" for r in out),
                "indeterminate_dispersed": sum(r[1] == "INDETERMINATE_DISPERSED" for r in out),
                "bound_median_m": round(qs[len(qs) // 2], 1) if qs else None,
                "bound_min_m": round(qs[0], 1) if qs else None, "bound_max_m": round(qs[-1], 1) if qs else None,
                "pair_reasons": reasons}
            print(loc, json.dumps(summary["flights"][loc]), flush=True)
    json.dump(summary, open(outd / "summary.json", "w"), indent=1)
