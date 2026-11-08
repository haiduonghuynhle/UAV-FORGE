#!/usr/bin/env python3
"""Review-round macros from existing per-frame records (CPU only).

Blocks:
  S  falsification sensitivity on UAV-VisLoc: relabel every full-coverage trial with an
     inflated query half-extent; count flips (wrong -> overlapping), split accepted/rejected;
     acceptance and unsafe rates on the surviving trials. Also min/median forced offset
     and its ratio to R for every dataset (power of the test).
  P  mixed-campaign populations: control-arm frames, predicate-wrong share (canonical key),
     clean-run wrong share on the same frames, flights covered.
  D  environment drift on attacked frames: 26/496-style denominators and flips.
  O  operational leak = clean-run wrong-retrieval share x full-coverage conditional acceptance.
  L  LODO bootstrap intervals from caves_learned_weights.json (accepted-only run).
  A  paired delta and direction audit on ACCEPTED observations (paired_delta_accepted.json).
"""
from __future__ import annotations
import argparse, csv, json, math, statistics as st, sys
from pathlib import Path
_R = Path(__file__).resolve().parent
sys.path.insert(0, str(_R))
import gen_v3_tables as gen  # noqa: E402

DS = {"avl": "Avl", "vpair": "Vpair", "uav_visloc": "Visloc"}
COND = {"wrong_near": "Near", "wrong_far": "Far", "visually_similar_wrong": "Sim"}
RECORD = {
    "avl": {"wrong_near": "forgec_avl_wrong_near/avl/ours_ultravpr_cascade_6dof_e2e/20260805_195057",
            "wrong_far": "forgec_avl_wrong_far/avl/ours_ultravpr_cascade_6dof_e2e/20260805_195151",
            "visually_similar_wrong": "forgec_avl_visually_similar_wrong/avl/ours_ultravpr_cascade_6dof_e2e/20260805_211521"},
    "vpair": {"wrong_near": "forgec_vpair_wrong_near/vpair/ours_ultravpr_cascade_6dof_e2e/20260810_155044",
              "wrong_far": "forgec_vpair_wrong_far/vpair/ours_ultravpr_cascade_6dof_e2e/20260810_164157",
              "visually_similar_wrong": "forgec_vpair_visually_similar_wrong/vpair/ours_ultravpr_cascade_6dof_e2e/20260810_173120"},
    "uav_visloc": {"wrong_near": "forgeb_visloc_wrong_near/uav_visloc/ours_ultravpr_cascade_6dof_e2e/20260811_044051",
                   "wrong_far": "forgeb_visloc_wrong_far/uav_visloc/ours_ultravpr_cascade_6dof_e2e/20260811_094249",
                   "visually_similar_wrong": "forgeb_visloc_visually_similar_wrong/uav_visloc/ours_ultravpr_cascade_6dof_e2e/20260811_144540"},
}
CLEAN = {"avl": "avl_ours_ultravpr/avl/ours_ultravpr_cascade_6dof_e2e/20260803_050832",
         "vpair": "vpair_ours_ultravpr/vpair/ours_ultravpr_cascade_6dof_e2e/20260803_042050",
         "uav_visloc": "visloc_ours_ultravpr/uav_visloc/ours_ultravpr_cascade_6dof_e2e/20260810_232532"}

def tf(v): return str(v).strip().lower() in ("true", "1")
def fl(v):
    try: return float(v)
    except (TypeError, ValueError): return float("nan")
def pct(x, d=1): return f"{100*x:.{d}f}\\%"
def rows(p): return list(csv.DictReader(open(p, newline="")))
def anchor(r):
    try: return json.loads(str(r.get("geodetic_anchor", "")).replace("'", '"'))
    except ValueError: return None
def attacked(rs): return [r for r in rs if tf(r.get("is_adversarial")) and not str(r.get("adversarial_failure_reason") or "").strip()]

def wrong_with_half(r, ds, qh_override=None):
    a = anchor(r)
    if a is None: return None
    gt_lat, gt_lon = fl(r["gt_lat"]), fl(r["gt_lon"])
    if qh_override is None:
        # The one label of record (gen_v3_tables), so the oblique AnyVisLoc
        # footprint reaches this script too; identical to the formula below
        # under the nadir labels.
        return not gen.selected_reference_can_localize(gt_lat, gt_lon, float(a["lat"]), float(a["lon"]), ds, r)
    de, dn = gen._per_axis_m(gt_lat, gt_lon, float(a["lat"]), float(a["lon"]))
    rh_e, rh_n = gen.REF_HALF_M[ds]
    qh_e, qh_n = gen.query_half_m(ds, r) if qh_override is None else (qh_override, qh_override)
    return not (de <= rh_e + qh_e + gen.TILE_SLACK_M and dn <= rh_n + qh_n + gen.TILE_SLACK_M)

def block_sensitivity(v3: Path, lines):
    lines.append("% --- S. falsification power / sensitivity (full-coverage cells of record)")
    for ds, cells in RECORD.items():
        R = gen.DEFAULT_RADIUS_M[ds]; offs = []
        for cond, rel in cells.items():
            for r in attacked(rows(v3 / rel / "predictions.csv")):
                d = fl(r.get("adversarial_forced_distance_to_gt_m"))
                if not math.isnan(d): offs.append(d)
        key = DS[ds]
        lines += [f"\\newcommand{{\\rvMinOffset{key}}}{{{min(offs):.0f}}}",
                  f"\\newcommand{{\\rvMedOffset{key}}}{{{st.median(offs):.0f}}}",
                  f"\\newcommand{{\\rvMinOffsetOverR{key}}}{{{min(offs)/R:.1f}}}"]
    # UAV-VisLoc relabelling bracket
    qh0 = gen.query_half_m("uav_visloc", {})[0]
    for tag, qh in (("Base", qh0), ("Mid", 140.0), ("Max", qh0 + 77.8)):
        n = acc = uns = flips = flips_acc = 0
        for cond, rel in RECORD["uav_visloc"].items():
            for r in attacked(rows(v3 / rel / "predictions.csv")):
                w0 = wrong_with_half(r, "uav_visloc"); w1 = wrong_with_half(r, "uav_visloc", qh)
                if w0 is None or w1 is None: continue
                if w0 and not w1:
                    flips += 1; flips_acc += tf(r["accepted_for_metrics"])
                if w1:
                    n += 1; a = tf(r["accepted_for_metrics"]); acc += a
                    uns += a and fl(r["post_fgo_error_m"]) > gen.DEFAULT_RADIUS_M["uav_visloc"]
        lines += [f"\\newcommand{{\\rvVislocQh{tag}}}{{{qh:.0f}}}",
                  f"\\newcommand{{\\rvVislocTrials{tag}}}{{{n}}}",
                  f"\\newcommand{{\\rvVislocAccept{tag}}}{{{pct(acc/n) if n else '--'}}}",
                  f"\\newcommand{{\\rvVislocUnsafe{tag}}}{{{pct(uns/n) if n else '--'}}}",
                  f"\\newcommand{{\\rvVislocFlips{tag}}}{{{flips}}}",
                  f"\\newcommand{{\\rvVislocFlipsAccepted{tag}}}{{{flips_acc}}}"]

def block_populations(v3: Path, mixed: Path, lines):
    lines.append("% --- P. mixed-campaign populations (control arm) and clean-run shares on the same frames")
    for ds, key in DS.items():
        cell = mixed / f"{ds}_wrong_near"
        p = sorted(cell.rglob("predictions.csv"))
        assert len(p) == 1, cell
        rs = rows(p[0]); ctl = [r for r in rs if not tf(r["adversarial_assigned_attack"])]
        wrong = [r for r in ctl if wrong_with_half(r, ds) is True]
        names = {r["image_name"] for r in rs}
        clean = [r for r in rows(v3 / CLEAN[ds] / "predictions.csv") if r["image_name"] in names]
        cwrong = [r for r in clean if wrong_with_half(r, ds) is True]
        flights = sorted({r["image_name"].split("/")[0] for r in rs if "/" in r["image_name"]})
        lines += [f"\\newcommand{{\\rvMixFrames{key}}}{{{len(rs)}}}",
                  f"\\newcommand{{\\rvMixControl{key}}}{{{len(ctl)}}}",
                  f"\\newcommand{{\\rvMixControlWrong{key}}}{{{pct(len(wrong)/len(ctl))}}}",
                  f"\\newcommand{{\\rvCleanWrongSame{key}}}{{{pct(len(cwrong)/len(clean)) if clean else '--'}}}",
                  f"\\newcommand{{\\rvMixFlights{key}}}{{{', '.join(flights) if flights else 'single route'}}}"]

def block_drift(v3: Path, mixed: Path, lines):
    lines.append("% --- D. environment drift on attacked frames (ref100 vs forgec, VPAIR proximal)")
    new = rows(sorted((mixed / "ref100_vpair_wrong_near").rglob("predictions.csv"))[0])
    old = {r["image_name"]: r for r in rows(v3 / RECORD["vpair"]["wrong_near"] / "predictions.csv")}
    pairs = [(a, old[a["image_name"]]) for a in new if a["image_name"] in old]
    att = [(a, b) for a, b in pairs if a in attacked([a]) and b in attacked([b])]
    flips_att = sum(tf(a["accepted_for_metrics"]) != tf(b["accepted_for_metrics"]) for a, b in att)
    flips_all = sum(tf(a["accepted_for_metrics"]) != tf(b["accepted_for_metrics"]) for a, b in pairs)
    lines += [f"\\newcommand{{\\rvDriftAttN}}{{{len(att)}}}",
              f"\\newcommand{{\\rvDriftAttAccRecord}}{{{sum(tf(b['accepted_for_metrics']) for _, b in att)}}}",
              f"\\newcommand{{\\rvDriftAttAccRerun}}{{{sum(tf(a['accepted_for_metrics']) for a, _ in att)}}}",
              f"\\newcommand{{\\rvDriftAttRateRecord}}{{{pct(sum(tf(b['accepted_for_metrics']) for _, b in att)/len(att))}}}",
              f"\\newcommand{{\\rvDriftAttRateRerun}}{{{pct(sum(tf(a['accepted_for_metrics']) for a, _ in att)/len(att))}}}",
              f"\\newcommand{{\\rvDriftFlipsAttacked}}{{{flips_att}}}",
              f"\\newcommand{{\\rvDriftFlipsAll}}{{{flips_all}}}"]

def block_operational(v3: Path, lines):
    lines.append("% --- O. operational leak = clean wrong-retrieval share x full-coverage conditional acceptance")
    for ds, key in DS.items():
        clean = rows(v3 / CLEAN[ds] / "predictions.csv")
        lab = [wrong_with_half(r, ds) for r in clean]; lab = [x for x in lab if x is not None]
        exposure = sum(lab) / len(lab)
        n = acc = 0
        for cond, rel in RECORD[ds].items():
            for r in attacked(rows(v3 / rel / "predictions.csv")):
                if wrong_with_half(r, ds) is True:
                    n += 1; acc += tf(r["accepted_for_metrics"])
        cond_rate = acc / n
        R = gen.DEFAULT_RADIUS_M[ds]
        unsafe_clean = [r for r in clean if tf(r["accepted_for_metrics"]) and fl(r["post_fgo_error_m"]) > R]
        # self-consistent unsafe = unsafe accepted pose whose selected reference the predicate labels CORRECT
        selfc = [r for r in unsafe_clean if wrong_with_half(r, ds) is False]
        leaks = [r for r in unsafe_clean if wrong_with_half(r, ds) is True]
        lines += [f"\\newcommand{{\\rvCleanSelfConsistentShare{key}}}{{{pct(len(selfc)/len(clean), 1)}}}",
                  f"\\newcommand{{\\rvCleanSelfConsistentN{key}}}{{{len(selfc)}}}",
                  f"\\newcommand{{\\rvCleanLeakUnsafeN{key}}}{{{len(leaks)}}}",
                  f"\\newcommand{{\\rvExposure{key}}}{{{pct(exposure)}}}",
                  f"\\newcommand{{\\rvCondAccept{key}}}{{{pct(cond_rate)}}}",
                  f"\\newcommand{{\\rvOpLeak{key}}}{{{pct(exposure*cond_rate, 2)}}}",
                  f"\\newcommand{{\\rvCleanUnsafeShare{key}}}{{{pct(len(unsafe_clean)/len(clean))}}}"]

def block_lodo_ci(lodo_json: Path, lines):
    lines.append("% --- L. leave-one-dataset-out bootstrap intervals (accepted observations)")
    d = json.loads(lodo_json.read_text())
    for fold, res in d["folds"].items():
        key = DS[fold]
        lo, hi = res["caves_auroc_ci"]
        lines += [f"\\newcommand{{\\rvLodoCavesLo{key}}}{{{lo:.2f}}}", f"\\newcommand{{\\rvLodoCavesHi{key}}}{{{hi:.2f}}}"]
        lines += [f"\\newcommand{{\\rvLodoCavesLo{key}Fine}}{{{lo:.3f}}}", f"\\newcommand{{\\rvLodoCavesHi{key}Fine}}{{{hi:.3f}}}"]
        for m, mac in (("lr_l2", "Lr"), ("gb", "Gb"), ("mlp", "Mlp")):
            lo, hi = res["variants"][m]["auroc_ci"]
            lines += [f"\\newcommand{{\\rvLodo{mac}Lo{key}}}{{{lo:.2f}}}", f"\\newcommand{{\\rvLodo{mac}Hi{key}}}{{{hi:.2f}}}"]

def block_accepted_delta(delta_json: Path, lines):
    lines.append("% --- A. paired delta (logistic - hand-set) and direction audit on ACCEPTED observations")
    d = json.loads(delta_json.read_text())
    for fold, res in d["comparison"]["folds"].items():
        key = DS[fold]
        lines += [f"\\newcommand{{\\rvDeltaAcc{key}}}{{{res['delta']:.3f}}}",
                  f"\\newcommand{{\\rvDeltaAccLo{key}}}{{{res['ci_low']:.3f}}}",
                  f"\\newcommand{{\\rvDeltaAccHi{key}}}{{{res['ci_high']:.3f}}}"]


def block_direction_accepted(mixed: Path, lines):
    """Oriented univariate AUROC on ACCEPTED observations (same rule as the all-frames audit)."""
    from caves_learned_weights import HANDSET_RISK_DIRECTION, feature_vector, load
    from caves_diagnostics import auroc
    roots = [mixed / f"{ds}_{c}" for ds in ("vpair", "avl", "uav_visloc")
             for c in ("wrong_near", "wrong_far", "visually_similar_wrong")]
    rs = load(roots, label_source="clean", accepted_only=True)
    by = {}
    for r in rs: by.setdefault(str(r.get("dataset", "")), []).append(r)
    names = {"inlier_ratio": "InlierRatio", "num_inliers": "InlierCount", "reproj_residual_px": "Reproj",
             "shape_confidence": "Shape", "geometry_anchor_offset_m": "Anchor"}
    lines.append("% --- A2. direction audit on ACCEPTED observations (oriented as the hand rule assumes)")
    for feat, mac in names.items():
        sign = HANDSET_RISK_DIRECTION.get(feat, 1.0)
        for fold in ("avl", "uav_visloc", "vpair"):
            xs, ys = [], []
            for r in by.get(fold, []):
                v = feature_vector(r, [feat])
                if v is None or any(math.isnan(x) for x in v): continue
                xs.append(sign * v[0]); ys.append(int(r["_caves_label"]))
            if len(set(ys)) == 2:
                lines.append(f"\\newcommand{{\\rvDirAcc{mac}{DS[fold]}}}{{{auroc(ys, xs):.3f}}}")

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--v3", type=Path, default=Path("/workspace/forge/results/v3"))
    ap.add_argument("--mixed", type=Path, default=Path("/workspace/forge/results/v3_forge5050b"))
    ap.add_argument("--lodo-json", type=Path)
    ap.add_argument("--delta-accepted-json", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    lines = ["% Auto-generated by gen_review_macros.py - DO NOT EDIT NUMBERS BY HAND",
             f"% source: {a.v3} (cells of record) + {a.mixed} (mixed campaign)"]
    block_sensitivity(a.v3, lines); block_populations(a.v3, a.mixed, lines)
    block_drift(a.v3, a.mixed, lines); block_operational(a.v3, lines)
    if a.lodo_json: block_lodo_ci(a.lodo_json, lines)
    if a.delta_accepted_json and a.delta_accepted_json.exists(): block_accepted_delta(a.delta_accepted_json, lines)
    try:
        block_direction_accepted(a.mixed, lines)
    except Exception as e:  # report, do not silently drop the block
        lines.append(f"% A2 direction audit on accepted observations FAILED: {e!r}")
    a.out.write_text("\n".join(lines) + "\n"); print(f"wrote {a.out} ({len(lines)} lines)")

if __name__ == "__main__":
    main()
