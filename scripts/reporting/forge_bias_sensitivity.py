"""How much protection the attitude cross-check loses under a biased yaw prior.

The recorded runs read a zero-mean simulated yaw prior. A real magnetometer's
dominant error is a standing bias, so the check as measured is optimistic. This
script replays the recorded per-frame signals with a constant bias added to the
reconstructed prior, holding the recorded pose estimates fixed.

Two readings per bias:
  fixed threshold  -- the check keeps its recorded bound, so the bias costs
                      coverage: correct-reference frames start failing it.
  re-tuned bound   -- the bound is widened to the smallest value that restores
                      the recorded correct-reference acceptance, and the wrong-
                      reference acceptance at that bound is reported. This is
                      what a deployment would do, and it is where the check
                      loses protection.

This is a counterfactual on recorded decisions, not a new run: a biased prior
would also move the warp and the fusion, which the replay cannot reproduce. It
therefore bounds the check-only mechanism, not the whole pipeline.

Run directories must be passed as pinned timestamps, not globs.
"""
import argparse, json, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from forge_threshold_sweep import load_cell, replay

DS_TAG = {"avl": "Avl", "vpair": "Vpair", "uav_visloc": "Visloc"}
# LaTeX control sequences take letters only, so a bias of 30 tags as ThreeZero.
DIGITS = "Zero One Two Three Four Five Six Seven Eight Nine".split()
DS_LABEL = {"avl": "AVL", "vpair": "VPAIR", "uav_visloc": "UAV-VisLoc"}


def arms(preds):
    """Indices of the correct-reference and wrong-reference frames."""
    ctrl = [i for i, p in enumerate(preds) if p.get("is_wrong_reference") is False]
    atk = [i for i, p in enumerate(preds) if p.get("is_wrong_reference") is True]
    return ctrl, atk


def rate(dec, idx):
    return sum(1 for i in idx if dec[i] is True) / len(idx) if idx else None


def sweep_cell(d, biases, grid):
    preds, cfg, m = load_cell(d)
    t_inl = int(cfg["geometry_min_inliers_after_lgi"])
    t_yaw = float(cfg["final_pose_max_yaw_prior_residual_deg"])
    base, unres = replay(preds, cfg, t_inl, t_yaw)
    rec = [bool(p.get("accepted_for_metrics")) for p in preds]
    mism = sum(1 for x, y in zip(base, rec) if x != y)
    ctrl, atk = arms(preds)
    out = dict(dir=d, dataset=m.get("dataset"), condition=m.get("adversarial_condition"),
               n=len(preds), n_ctrl=len(ctrl), n_atk=len(atk),
               baseline_mismatch=mism, t_yaw0=t_yaw, t_inl0=t_inl, rows=[])
    if mism:
        return out
    for b in biases:
        dec, _ = replay(preds, cfg, t_inl, t_yaw, bias=b)
        row = dict(bias=b,
                   ctrl_acc=sum(1 for i in ctrl if dec[i] is True),
                   atk_acc=sum(1 for i in atk if dec[i] is True))
        # Smallest bound on the grid that restores the recorded control-arm
        # acceptance. Widening the bound only ever accepts more, so the first
        # hit on an ascending grid is the smallest.
        target = sum(1 for i in ctrl if base[i] is True)
        row["tuned_t"] = None
        for t in grid:
            d2, _ = replay(preds, cfg, t_inl, t, bias=b)
            if sum(1 for i in ctrl if d2[i] is True) >= target:
                row["tuned_t"] = t
                row["tuned_ctrl_acc"] = sum(1 for i in ctrl if d2[i] is True)
                row["tuned_atk_acc"] = sum(1 for i in atk if d2[i] is True)
                break
        out["rows"].append(row)
    return out


def pool(cells, biases):
    """Sum the per-condition cells of one dataset into one rate per bias."""
    agg = {}
    for c in cells:
        for r in c["rows"]:
            a = agg.setdefault(r["bias"], dict(nc=0, na=0, ca=0, aa=0, tca=0, taa=0, t=[]))
            a["nc"] += c["n_ctrl"]; a["na"] += c["n_atk"]
            a["ca"] += r["ctrl_acc"]; a["aa"] += r["atk_acc"]
            a["tca"] += r.get("tuned_ctrl_acc", 0); a["taa"] += r.get("tuned_atk_acc", 0)
            a["t"].append(r["tuned_t"])
    for b, a in agg.items():
        a["ctrl_rate"] = 100.0 * a["ca"] / a["nc"]
        a["atk_rate"] = 100.0 * a["aa"] / a["na"]
        a["tuned_ctrl_rate"] = 100.0 * a["tca"] / a["nc"]
        a["tuned_atk_rate"] = 100.0 * a["taa"] / a["na"]
        a["tuned_t_max"] = None if None in a["t"] else max(a["t"])
    return agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", nargs="+", required=True)
    ap.add_argument("--biases", nargs="+", type=float, default=[0, 5, 10, 20, 30])
    ap.add_argument("--grid", nargs="+", type=float,
                    default=[float(t) for t in range(30, 181, 5)])
    ap.add_argument("--json-out")
    ap.add_argument("--out")
    ap.add_argument("--table")
    a = ap.parse_args()

    cells = [sweep_cell(d, a.biases, a.grid) for d in a.cells]
    bad = [c for c in cells if c["baseline_mismatch"]]
    if bad:
        raise SystemExit("baseline replay does not reproduce recorded decisions: "
                         + ", ".join(f'{c["dir"]} ({c["baseline_mismatch"]})' for c in bad))
    by_ds = {}
    for c in cells:
        by_ds.setdefault(c["dataset"], []).append(c)
    pooled = {ds: pool(cs, a.biases) for ds, cs in by_ds.items()}

    for ds, agg in sorted(pooled.items()):
        print(f"== {ds}  control n={by_ds[ds][0]['n_ctrl']}*{len(by_ds[ds])} "
              f"attacked n={by_ds[ds][0]['n_atk']}*{len(by_ds[ds])}")
        for b in a.biases:
            r = agg[b]
            print(f"  bias {b:5.1f} deg | fixed bound"
                  f" control {r['ctrl_rate']:5.1f}% attacked {r['atk_rate']:5.1f}%"
                  f" | re-tuned bound {r['tuned_t_max']} control {r['tuned_ctrl_rate']:5.1f}%"
                  f" attacked {r['tuned_atk_rate']:5.1f}%")

    prov = ["% Generated by scripts/reporting/forge_bias_sensitivity.py -- do not edit.",
            "% Counterfactual replay of recorded per-frame signals; not a new run."]
    prov += [f'% source: {c["dir"]} (baseline replay mismatches: {c["baseline_mismatch"]}/{c["n"]})'
             for c in cells]
    if a.out:
        lines = list(prov) + [f"\\newcommand{{\\bsBiasMax}}{{{max(a.biases):.0f}}}"]
        for ds, agg in sorted(pooled.items()):
            t = DS_TAG[ds]
            for b in a.biases:
                r = agg[b]
                tag = "B" + "".join(DIGITS[int(c)] for c in str(int(b)))
                lines += [f"\\newcommand{{\\bs{k}{tag}{t}}}{{{v}}}" for k, v in (
                    ("Ctrl", f"{r['ctrl_rate']:.1f}\\%"), ("Atk", f"{r['atk_rate']:.1f}\\%"),
                    ("TunedT", f"{r['tuned_t_max']:.0f}" if r["tuned_t_max"] is not None else "n/a"),
                    ("TunedCtrl", f"{r['tuned_ctrl_rate']:.1f}\\%"),
                    ("TunedAtk", f"{r['tuned_atk_rate']:.1f}\\%"))]
            lines.append(f"\\newcommand{{\\bsN{t}Ctrl}}{{{sum(c['n_ctrl'] for c in by_ds[ds])}}}")
            lines.append(f"\\newcommand{{\\bsN{t}Atk}}{{{sum(c['n_atk'] for c in by_ds[ds])}}}")
        open(a.out, "w").write("\n".join(lines) + "\n")
        print("wrote", a.out)
    if a.table:
        rows = [r"\begin{tabular}{lrrrrrr}", r"\toprule",
                r" & Bias & \multicolumn{2}{c}{Recorded bound} & \multicolumn{3}{c}{Re-tuned bound} \\",
                r"\cmidrule(lr){3-4}\cmidrule(lr){5-7}",
                r"Dataset & ($^\circ$) & Correct & Wrong & Bound ($^\circ$) & Correct & Wrong \\",
                r"\midrule"]
        for ds, agg in sorted(pooled.items(), key=lambda kv: DS_LABEL[kv[0]]):
            for b in a.biases:
                r = agg[b]
                rows.append(f"{DS_LABEL[ds]} & {b:.0f} & {r['ctrl_rate']:.1f} & {r['atk_rate']:.1f}"
                            f" & {r['tuned_t_max']:.0f} & {r['tuned_ctrl_rate']:.1f}"
                            f" & {r['tuned_atk_rate']:.1f} \\\\")
        rows += [r"\bottomrule", r"\end{tabular}"]
        open(a.table, "w").write("\n".join(prov + rows) + "\n")
        print("wrote", a.table)
    if a.json_out:
        json.dump({"cells": cells, "pooled": {k: {str(b): v for b, v in a2.items()}
                                              for k, a2 in pooled.items()}},
                  open(a.json_out, "w"), indent=1)


if __name__ == "__main__":
    main()
