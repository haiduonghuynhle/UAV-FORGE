"""S1 audit: do the CAVES signals separate the *arm* (injected vs control) rather
than the *fault* (unsafe vs safe), among accepted observations of the mixed campaign?

If a signal has AUROC far from 0.5 against the arm label, a learned scorer fitted on
this population can score arm membership instead of geometric fault evidence.

Population: accepted observations (accepted_for_metrics true), all arms, pooled over
the three conditions of one dataset -- the same population as the LODO analysis.
Intervals: moving-block bootstrap over contiguous within-flight blocks, ~25 blocks
per dataset, 2000 draws, central 95 percent. Run dirs are pinned, never globbed.
"""
import csv, json, math, os, random, sys
from pathlib import Path

FEATURES = ["retrieval_score_margin", "inlier_ratio", "num_inliers",
            "reproj_residual_px", "shape_confidence", "geometry_anchor_offset_m",
            "retrieval_rank"]
ROOT = "/workspace/forge/results/v3_forge5050b"
CELLS = {
    "AVL":        ["avl_wrong_near", "avl_wrong_far", "avl_visually_similar_wrong"],
    "VPAIR":      ["vpair_wrong_near", "vpair_wrong_far", "vpair_visually_similar_wrong"],
    "UAV-VisLoc": ["uav_visloc_wrong_near", "uav_visloc_wrong_far",
                   "uav_visloc_visually_similar_wrong"],
}
NBLOCK, DRAWS = 25, 2000


def fnum(v):
    try:
        f = float(v)
        return None if math.isnan(f) else f
    except (TypeError, ValueError):
        return None


def truthy(v):
    return str(v).strip().lower() in ("true", "1", "yes")


def pinned_run(cell):
    """Exactly one run timestamp per cell, or fail loudly."""
    base = os.path.join(ROOT, cell)
    runs = []
    for ds in sorted(os.listdir(base)):
        p1 = os.path.join(base, ds)
        if not os.path.isdir(p1):
            continue
        for pipe in sorted(os.listdir(p1)):
            p2 = os.path.join(p1, pipe)
            if not os.path.isdir(p2):
                continue
            for ts in sorted(os.listdir(p2)):
                p3 = os.path.join(p2, ts)
                if os.path.isfile(os.path.join(p3, "predictions.csv")):
                    runs.append(p3)
    if len(runs) != 1:
        raise SystemExit(f"{cell}: expected exactly one run, found {len(runs)}: {runs}")
    return runs[0]


def load(cell):
    d = pinned_run(cell)
    with open(os.path.join(d, "predictions.csv")) as fh:
        rows = list(csv.DictReader(fh))
    diag = {}
    dj = os.path.join(d, "frame_diagnostics.jsonl")
    if os.path.exists(dj):
        for line in open(dj):
            r = json.loads(line)
            key = r.get("image_name") or r.get("frame_name")
            if key:
                diag[key] = r
    out = []
    for i, r in enumerate(rows):
        key = r.get("image_name") or r.get("frame_name") or ""
        dr = diag.get(key, {})
        acc = r.get("accepted_for_metrics", dr.get("accepted_for_metrics"))
        if not truthy(acc):
            continue
        arm = r.get("is_adversarial", dr.get("is_adversarial"))
        err = fnum(r.get("post_fgo_error_m", dr.get("post_fgo_error_m")))
        thr = fnum(r.get("localization_success_threshold_m"))
        out.append({
            "i": i,
            "flight": r.get("flight_id") or r.get("sequence") or r.get("location") or "0",
            "arm": 1 if truthy(arm) else 0,
            "unsafe": None if (err is None or thr is None) else int(err > thr),
            "f": {k: fnum(r.get(k, dr.get(k))) for k in FEATURES},
        })
    return out, d


def auroc(pairs):
    """pairs: list of (score, label). Rank-based, ties averaged."""
    pairs = [(s, y) for s, y in pairs if s is not None and y is not None]
    pos = sum(y for _, y in pairs)
    neg = len(pairs) - pos
    if pos == 0 or neg == 0:
        return None
    order = sorted(range(len(pairs)), key=lambda k: pairs[k][0])
    ranks = [0.0] * len(pairs)
    j = 0
    while j < len(order):
        k = j
        while k + 1 < len(order) and pairs[order[k + 1]][0] == pairs[order[j]][0]:
            k += 1
        avg = (j + k) / 2.0 + 1.0
        for m in range(j, k + 1):
            ranks[order[m]] = avg
        j = k + 1
    rsum = sum(ranks[k] for k in range(len(pairs)) if pairs[k][1] == 1)
    return (rsum - pos * (pos + 1) / 2.0) / (pos * neg)


def blocks(rows):
    """Contiguous within-flight blocks, ~NBLOCK blocks over the dataset."""
    by = {}
    for r in rows:
        by.setdefault(r["flight"], []).append(r)
    out = []
    size = max(1, len(rows) // NBLOCK)
    for _, rs in sorted(by.items()):
        rs.sort(key=lambda r: r["i"])
        for s in range(0, len(rs), size):
            out.append(rs[s:s + size])
    return out


def ci(rows, feat, label, seed=7):
    bl = blocks(rows)
    rng = random.Random(seed)
    vals = []
    for _ in range(DRAWS):
        samp = []
        for _ in range(len(bl)):
            samp.extend(bl[rng.randrange(len(bl))])
        a = auroc([(r["f"][feat], r[label]) for r in samp])
        if a is not None:
            vals.append(a)
    if len(vals) < DRAWS // 2:
        return None, None
    vals.sort()
    return vals[int(0.025 * len(vals))], vals[int(0.975 * len(vals)) - 1]


def main():
    report = {}
    for ds, cells in CELLS.items():
        rows, srcs = [], []
        for c in cells:
            rs, d = load(c)
            rows.extend(rs)
            srcs.append(d.replace(ROOT + "/", ""))
        n_inj = sum(r["arm"] for r in rows)
        n_uns = sum(r["unsafe"] for r in rows if r["unsafe"] is not None)
        print(f"\n=== {ds}: accepted={len(rows)} injected-arm={n_inj} unsafe={n_uns}")
        ent = {"n": len(rows), "n_injected": n_inj, "n_unsafe": n_uns,
               "sources": srcs, "features": {}}
        for f in FEATURES:
            a_arm = auroc([(r["f"][f], r["arm"]) for r in rows])
            a_uns = auroc([(r["f"][f], r["unsafe"]) for r in rows])
            if a_arm is None:
                print(f"  {f:26s} n/a")
                continue
            lo, hi = ci(rows, f, "arm")
            print(f"  {f:26s} arm AUROC={a_arm:.3f} [{lo:.3f},{hi:.3f}]"
                  f"   unsafe AUROC={a_uns:.3f}" if a_uns is not None else
                  f"  {f:26s} arm AUROC={a_arm:.3f} [{lo:.3f},{hi:.3f}]")
            ent["features"][f] = {"arm_auroc": a_arm, "arm_lo": lo, "arm_hi": hi,
                                  "unsafe_auroc": a_uns}
        report[ds] = ent
    with open("/workspace/forge/results/v3_forge5050b/margin_arm_audit.json", "w") as fh:
        json.dump(report, fh, indent=2)
    write_macros(report, "/workspace/forge/results/v3_forge5050b/arm_audit_numbers.tex")
    print("\nwrote margin_arm_audit.json + arm_audit_numbers.tex")


TAG = {"AVL": "Avl", "VPAIR": "Vpair", "UAV-VisLoc": "Visloc"}
EMIT = [("retrieval_score_margin", "Margin"), ("retrieval_rank", "Rank"),
        ("geometry_anchor_offset_m", "Offset"), ("shape_confidence", "Shape")]


def min_detectable_auroc(n_pos, n_neg, z=1.96):
    """Smallest AUROC whose Hanley--McNeil lower bound clears one half.

    A criterion of the form "lower bound above one half" cannot be met below
    this value at the given class counts, however good the scorer is.
    """
    def se(a):
        q1 = a / (2 - a)
        q2 = 2 * a * a / (1 + a)
        return math.sqrt((a * (1 - a) + (n_pos - 1) * (q1 - a * a)
                          + (n_neg - 1) * (q2 - a * a)) / (n_pos * n_neg))
    lo, hi = 0.5, 0.999
    for _ in range(200):
        mid = (lo + hi) / 2
        if mid - z * se(mid) > 0.5:
            hi = mid
        else:
            lo = mid
    return hi


def write_macros(report, path):
    lines = ["% Generated by scripts/reporting/forge_arm_separation_audit.py -- do not edit.",
             "% Univariate AUROC of each signal against the arm label (injected vs control),",
             "% on accepted observations of the mixed campaign. 95% block bootstrap intervals.",
             "% Values far from one half in either direction indicate arm information."]
    for ds in ("AVL", "VPAIR", "UAV-VisLoc"):
        for src in report[ds]["sources"]:
            lines.append("% source: {}".format(src))
    for ds, tag in TAG.items():
        ent = report[ds]
        lines.append("\\newcommand{\\asN%s}{%s}" % (tag, f"{ent['n']:,}".replace(",", "\\,")))
        n_pos = ent["n_unsafe"]
        n_neg = ent["n"] - n_pos
        lines.append("\\newcommand{\\asPos%s}{%d}" % (tag, n_pos))
        lines.append("\\newcommand{\\asMinAuroc%s}{%.2f}"
                     % (tag, min_detectable_auroc(n_pos, n_neg)))
        for feat, fname in EMIT:
            f = ent["features"].get(feat)
            if not f:
                continue
            lines.append("\\newcommand{\\as%sArm%s}{%.3f}" % (fname, tag, f["arm_auroc"]))
            lines.append("\\newcommand{\\as%sArmLo%s}{%.3f}" % (fname, tag, f["arm_lo"]))
            lines.append("\\newcommand{\\as%sArmHi%s}{%.3f}" % (fname, tag, f["arm_hi"]))
    Path(path).write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
