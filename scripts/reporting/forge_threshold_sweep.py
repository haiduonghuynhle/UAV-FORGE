"""Threshold-sensitivity replay for the two configured acceptance checks.

Replays recorded per-frame signals through the acceptance cascade at swept
thresholds for the minimum-inlier check and the attitude cross-check.

The replay reconstructs the simulated yaw prior, which is deterministic given
the frame index (per-frame jitter seeded 42 + index, plus a cumulative bias
drift seeded 12345). It reproduces the recorded accept decision at the
config-of-record thresholds on every frame before any swept cell is reported;
a cell with a single mismatch is dropped rather than swept.

Loosening the inlier check below the recorded value leaves frames that were
cut at the geometry stage with no recorded downstream signals. Those frames
are counted as unresolved and reported as an upper bound, never as accepts.

Run directories must be passed as pinned timestamps, not globs.
"""
import json, os, sys, random, argparse
from collections import Counter

SIGMA = dict(alt=3.0, pitch=3.0, roll=3.0, yaw=8.0)
BD = dict(alt=0.02, pitch=0.005, roll=0.005, yaw=0.015)


def drift_yaw(n, seed=12345):
    rng = random.Random(seed); out = []; a = p = r = y = 0.0
    for _ in range(n):
        a += rng.gauss(0, BD["alt"]); p += rng.gauss(0, BD["pitch"])
        r += rng.gauss(0, BD["roll"]); y += rng.gauss(0, BD["yaw"])
        out.append(y)
    return out


def jitter_yaw(i):
    rng = random.Random(42 + i)
    rng.gauss(0, SIGMA["alt"]); rng.gauss(0, SIGMA["pitch"]); rng.gauss(0, SIGMA["roll"])
    return rng.gauss(0, SIGMA["yaw"])


def angdiff(a, b):
    return (a - b + 180.0) % 360.0 - 180.0


def fnum(d, k):
    try:
        v = float(d.get(k))
    except (TypeError, ValueError):
        return None
    return v if v == v and abs(v) != float("inf") else None


REACHED_GEOM = {"geometry_inlier_gate_reject", "final_pose_gate_reject", ""}


def replay(preds, cfg, t_inl, t_yaw, bias=0.0):
    """Return (decisions, n_unresolved). decisions[i] in {True, False, None}.

    `bias` adds a constant offset, in degrees, to the reconstructed yaw prior,
    standing in for a magnetometer's dominant error. It leaves every other
    recorded signal alone, so it replays the check, not the pipeline."""
    dy = drift_yaw(len(preds))
    t_inl0 = int(cfg["geometry_min_inliers_after_lgi"])
    max_anchor = cfg.get("final_pose_max_geometry_anchor_offset_m")
    min_shape = cfg.get("final_pose_min_shape_conf")
    max_reproj = cfg.get("final_pose_max_reproj_residual_px")
    out, unresolved = [], 0
    for i, p in enumerate(preds):
        fr = p.get("failure_reason") or ""
        inl = fnum(p, "num_inliers")
        # Frames that never reached the inlier gate keep their recorded outcome.
        if fr not in REACHED_GEOM:
            out.append(bool(p.get("accepted_for_metrics")))
            continue
        if inl is None:
            out.append(bool(p.get("accepted_for_metrics")))
            continue
        if inl < t_inl:
            out.append(False)
            continue
        if inl < t_inl0:
            # Loosened below the recorded threshold: this frame was cut at
            # geometry, so no downstream signal exists to replay.
            out.append(None); unresolved += 1
            continue
        ph, gh = fnum(p, "pred_heading"), fnum(p, "gt_heading")
        if ph is None or gh is None:
            out.append(bool(p.get("accepted_for_metrics")))
            continue
        ao = fnum(p, "geometry_anchor_offset_m")
        if ao is not None and max_anchor is not None and ao > float(max_anchor):
            out.append(False); continue
        sc = fnum(p, "shape_confidence")
        if sc is not None and min_shape is not None and sc < float(min_shape):
            out.append(False); continue
        rp = fnum(p, "reproj_residual_px")
        if rp is not None and max_reproj is not None and rp > float(max_reproj):
            out.append(False); continue
        prior = gh + bias + jitter_yaw(i) + dy[i]
        if abs(angdiff(ph, prior)) > t_yaw:
            out.append(False); continue
        out.append(True)
    return out, unresolved


def unsafe(p):
    err = fnum(p, "post_fgo_error_m")
    thr = fnum(p, "localization_success_threshold_m")
    if err is None or thr is None:
        return None
    return err > thr


def load_cell(d):
    preds = json.load(open(os.path.join(d, "results_detailed.json")))["predictions"]
    m = json.load(open(os.path.join(d, "run_manifest.json")))
    cfg = dict(m.get("pipeline_config") or {})
    for k in ("final_pose_min_shape_conf", "final_pose_max_reproj_residual_px",
              "final_pose_max_geometry_anchor_offset_m"):
        if k in m:
            cfg.setdefault(k, m[k])
    return preds, cfg, m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", nargs="+", required=True)
    ap.add_argument("--yaw", nargs="+", type=float, default=[15.0, 30.0, 45.0])
    ap.add_argument("--inliers", nargs="+", type=int, default=[20, 25, 30])
    ap.add_argument("--json-out")
    ap.add_argument("--tex-out")
    ap.add_argument("--table-out")
    a = ap.parse_args()

    report = {"cells": [], "sweep": []}
    for d in a.cells:
        preds, cfg, m = load_cell(d)
        t_inl0 = int(cfg["geometry_min_inliers_after_lgi"])
        t_yaw0 = float(cfg["final_pose_max_yaw_prior_residual_deg"])
        base, _ = replay(preds, cfg, t_inl0, t_yaw0)
        rec = [bool(p.get("accepted_for_metrics")) for p in preds]
        mism = [i for i, (x, y) in enumerate(zip(base, rec)) if x != y]
        name = "/".join(d.rstrip("/").split("/")[-4:-1])
        attacked = [i for i, p in enumerate(preds)
                    if p.get("adversarial_forced_reference_name")]
        if os.environ.get("FORGE_AVL_FOOTPRINT", "nadir") == "oblique" and m.get("dataset") == "avl":
            # Genuine faults under the oblique AnyVisLoc footprint only.
            import gen_v3_tables as G
            faults = {r["image_name"] for r in G._attacked_rows(d) if G._is_genuine_fault(r, "avl")}
            attacked = [i for i in attacked if preds[i].get("image_name") in faults]
        report["cells"].append(dict(
            cell=name, dir=d, n=len(preds), n_attacked=len(attacked),
            baseline_mismatch=len(mism),
            mismatch_reasons=Counter(preds[i].get("failure_reason") or "" for i in mism).most_common(5),
            t_inl0=t_inl0, t_yaw0=t_yaw0,
            dataset=m.get("dataset"), condition=m.get("adversarial_condition"),
            mode=m.get("adversarial_mode")))
        if mism:
            continue
        for ti in a.inliers:
            for ty in a.yaw:
                dec, unres = replay(preds, cfg, ti, ty)
                idx = attacked if attacked else list(range(len(preds)))
                acc = sum(1 for i in idx if dec[i] is True)
                unk = sum(1 for i in idx if dec[i] is None)
                uns = sum(1 for i in idx if dec[i] is True and unsafe(preds[i]) is True)
                report["sweep"].append(dict(
                    cell=name, dataset=m.get("dataset"),
                    condition=m.get("adversarial_condition"),
                    mode=m.get("adversarial_mode"),
                    t_inl=ti, t_yaw=ty, n_denom=len(idx),
                    accepted=acc, unresolved=unk, unsafe_accepted=uns,
                    accept_rate=round(acc / len(idx), 5) if idx else None,
                    accept_rate_upper=round((acc + unk) / len(idx), 5) if idx else None))
    print(json.dumps(report, indent=1))
    if a.json_out:
        json.dump(report, open(a.json_out, "w"), indent=1)
    if a.tex_out:
        write_macros(report, a.tex_out)
    if a.table_out:
        write_table(report, a.table_out)



DS_TAG = {"avl": "Avl", "vpair": "Vpair", "uav_visloc": "Visloc"}
DS_LABEL = {"avl": "AVL", "vpair": "VPAIR", "uav_visloc": "UAV-VisLoc"}


def _aggregate(report):
    """Pool the per-condition cells into one rate per (dataset, threshold pair)."""
    agg = {}
    for s in report["sweep"]:
        k = (s["dataset"], s["t_inl"], s["t_yaw"])
        a = agg.setdefault(k, dict(acc=0, unk=0, n=0, uns=0))
        a["acc"] += s["accepted"]; a["unk"] += s["unresolved"]
        a["n"] += s["n_denom"]; a["uns"] += s["unsafe_accepted"]
    for a in agg.values():
        a["rate"] = 100.0 * a["acc"] / a["n"]
        a["upper"] = 100.0 * (a["acc"] + a["unk"]) / a["n"]
    return agg


def _provenance(report):
    lines = ["% Generated by scripts/reporting/forge_threshold_sweep.py -- do not edit.",
             "% Counterfactual replay of recorded per-frame signals; not a new run."]
    for c in report["cells"]:
        lines.append("% source: {} (baseline replay mismatches: {}/{})".format(
            c["dir"], c["baseline_mismatch"], c["n"]))
    return lines


def write_macros(report, path):
    agg = _aggregate(report)
    rec_inl = int(report["cells"][0]["t_inl0"])
    rec_yaw = float(report["cells"][0]["t_yaw0"])
    exact = sorted({(ti, ty) for (_, ti, ty) in agg if ti >= rec_inl})
    loose = sorted({ti for (_, ti, _) in agg if ti < rec_inl})
    out = _provenance(report)
    out.append("\\newcommand{\\tsGridN}{%d}" % len(exact))
    out.append("\\newcommand{\\tsInlRec}{%d}" % rec_inl)
    out.append("\\newcommand{\\tsYawRec}{%.0f\\textdegree}" % rec_yaw)
    out.append("\\newcommand{\\tsInlHi}{%d}" % max(ti for ti, _ in exact))
    out.append("\\newcommand{\\tsYawLo}{%.0f\\textdegree}" % min(ty for _, ty in exact))
    out.append("\\newcommand{\\tsYawHi}{%.0f\\textdegree}" % max(ty for _, ty in exact))
    for ds, tag in DS_TAG.items():
        cells = [(ti, ty) for (d, ti, ty) in agg if d == ds and ti >= rec_inl]
        if not cells:
            continue
        rates = [agg[(ds, ti, ty)]["rate"] for ti, ty in cells]
        out.append("\\newcommand{\\ts%sLo}{%.1f\\%%}" % (tag, min(rates)))
        out.append("\\newcommand{\\ts%sHi}{%.1f\\%%}" % (tag, max(rates)))
        out.append("\\newcommand{\\ts%sRec}{%.1f\\%%}"
                   % (tag, agg[(ds, rec_inl, rec_yaw)]["rate"]))
        out.append("\\newcommand{\\ts%sTrials}{%s}"
                   % (tag, "{:,}".format(agg[(ds, rec_inl, rec_yaw)]["n"]).replace(",", "\\,")))
        if loose:
            lo = min(loose)
            out.append("\\newcommand{\\ts%sBound}{%.1f\\%%}" % (tag, agg[(ds, lo, rec_yaw)]["upper"]))
            out.append("\\newcommand{\\ts%sUnres}{%d}" % (tag, agg[(ds, lo, rec_yaw)]["unk"]))
    if loose:
        out.append("\\newcommand{\\tsInlLoose}{%d}" % min(loose))
    with open(path, "w") as f:
        f.write("\n".join(out) + "\n")


def write_table(report, path):
    agg = _aggregate(report)
    rec_inl = int(report["cells"][0]["t_inl0"])
    rec_yaw = float(report["cells"][0]["t_yaw0"])
    grid = sorted({(ti, ty) for (_, ti, ty) in agg})
    rows = []
    for ti, ty in grid:
        cs = []
        for ds in ("avl", "vpair", "uav_visloc"):
            a = agg.get((ds, ti, ty))
            if a is None:
                cs.append("--"); continue
            if a["unk"]:
                cs.append("$\\leq$%.1f" % a["upper"])
            else:
                cs.append("%.1f" % a["rate"])
        mark = "$\\star$" if (ti == rec_inl and ty == rec_yaw) else ""
        rows.append("%d & %.0f%s & %s \\\\" % (ti, ty, mark, " & ".join(cs)))
    body = "\n".join(_provenance(report)) + """
\\begin{tabular}{rl rrr}
\\toprule
Min.\\ inliers & Attitude (\\textdegree) & AVL & VPAIR & UAV-VisLoc \\\\
\\midrule
""" + "\n".join(rows) + """
\\bottomrule
\\end{tabular}
"""
    with open(path, "w") as f:
        f.write(body)


if __name__ == "__main__":
    main()
