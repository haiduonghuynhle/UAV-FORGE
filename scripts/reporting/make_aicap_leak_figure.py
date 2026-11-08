#!/usr/bin/env python3
"""Draw one accepted wrong-reference fix, and the severity of all of them.

The AICAP figure answers a reviewer request: show what an injected wrong
reference looks like in the data, not only as a rate in a table. Two panels:

  (a) one accepted trial, drawn from the recorded prediction row: where the
      aircraft was, which tile the evaluator forced, and where the pipeline
      said the aircraft was. The health signals of that same trial are printed
      beside it, because the point is that they looked normal.
  (b) every accepted wrong-reference trial, as position error against the
      dataset success radius, on a logarithmic axis. By default panel (b)
      reads the same campaign as panel (a); with --heldout-visloc and
      --heldout-vpair it reads the pre-registered held-out campaigns instead
      (every injection cell under each root), and AnyVisLoc, which has no
      held-out campaign, leaves the panel.

Every number the figure or its caption prints is written to a macro file, so
the paper never types one by hand, and a provenance file records the CSV paths
with their SHA-256 digests.

Usage:
    python3 scripts/reporting/make_aicap_leak_figure.py \
        --campaign results_server/v3_forge5050b_mixed \
        --heldout-visloc results_server/v4_2026-10-24/results/v4_e1_heldout_visloc \
        --heldout-vpair results_server/v4_2026-10-24/results/v4_e1b_heldout_vpair \
        --out Report/Paper/aicap-2027
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import statistics
from pathlib import Path

# Cells of the campaign, grouped by dataset. The names are directory names of
# the campaign root, not results: each holds one condition of one dataset.
CELLS = {
    "Avl": ["avl_wrong_near", "avl_wrong_far", "avl_visually_similar_wrong"],
    "Vpair": ["vpair_wrong_near", "vpair_wrong_far", "vpair_visually_similar_wrong"],
    "Visloc": [
        "uav_visloc_wrong_near",
        "uav_visloc_wrong_far",
        "uav_visloc_visually_similar_wrong",
    ],
}
LABEL = {"Avl": "AnyVisLoc", "Vpair": "VPAIR", "Visloc": "UAV-VisLoc"}
# The illustrated case. UAV-VisLoc publishes the flight raster, so the camera
# frame and both map tiles of this trial can be shown as pictures
# (scripts/reporting/make_aicap_imagery.py cuts them).
CASE_CELL = "uav_visloc_wrong_far"
CASE_FRAME = "01/01_0598.JPG"
CASE_KEY = "Visloc"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def read_macros(tables: Path) -> dict[str, str]:
    """macro name -> literal text, over the generated macro files."""
    out: dict[str, str] = {}
    for path in sorted(tables.glob("*.tex")):
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.split("%", 1)[0]
            while "\\newcommand{\\" in line:
                head = line.index("\\newcommand{\\") + len("\\newcommand{\\")
                close = line.index("}", head)
                name = line[head:close]
                rest = line[close + 1 :]
                if not rest.startswith("{"):
                    break
                depth, end = 0, None
                for i, ch in enumerate(rest):
                    if ch == "{":
                        depth += 1
                    elif ch == "}":
                        depth -= 1
                        if depth == 0:
                            end = i
                            break
                if end is None:
                    break
                out[name] = rest[1:end]
                line = rest[end + 1 :]
    return out


def number(text: str) -> float:
    return float(text.replace("\\%", "").replace("{,}", "").replace(",", ""))


def leaks(csv_path: Path) -> list[dict]:
    with csv_path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    return [
        r
        for r in rows
        if r.get("accepted_for_metrics") == "True"
        and r.get("is_wrong_reference") in ("True", "1")
    ]


def local_metres(lat0: float, lon0: float, lat: float, lon: float) -> tuple[float, float]:
    """East, north offset in metres from (lat0, lon0), flat-earth approximation."""
    m_per_deg_lat = 111_320.0
    m_per_deg_lon = m_per_deg_lat * math.cos(math.radians(lat0))
    return (lon - lon0) * m_per_deg_lon, (lat - lat0) * m_per_deg_lat


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path, help="paper directory")
    ap.add_argument("--heldout-visloc", type=Path, help="E1 campaign root (per-flight injection cells)")
    ap.add_argument("--heldout-vpair", type=Path, help="E1b campaign root")
    args = ap.parse_args()
    heldout = {"Visloc": args.heldout_visloc, "Vpair": args.heldout_vpair}
    use_heldout = any(heldout.values())
    if use_heldout and not all(heldout.values()):
        raise SystemExit("pass both --heldout-visloc and --heldout-vpair, or neither")

    tables = args.out / "tables"
    macros = read_macros(tables)
    radius = {k: number(macros[f"protoRadius{k}"]) for k in CELLS}

    per_dataset: dict[str, list[float]] = {}
    sources: list[dict] = []
    panel_keys = ["Vpair", "Visloc"] if use_heldout else ["Avl", "Vpair", "Visloc"]
    for key in panel_keys:
        errs: list[float] = []
        if use_heldout:
            root = heldout[key]
            paths = sorted(p for p in root.rglob("predictions.csv")
                           if "clean" not in str(p.relative_to(root)))
            if not paths:
                raise SystemExit(f"no injection cells under {root}")
            cell_paths = [(str(p.parent.relative_to(root)), p) for p in paths]
        else:
            cell_paths = [(cell, args.campaign / cell / "predictions.csv") for cell in CELLS[key]]
        for cell, path in cell_paths:
            rows = leaks(path)
            if key == "Avl" and os.environ.get("FORGE_AVL_FOOTPRINT", "nadir") == "oblique":
                # Only forced tiles disjoint from the oblique AnyVisLoc footprint.
                import gen_v3_tables as G
                rows = [r for r in rows if G._is_genuine_fault(r, "avl")]
            errs += [float(r["post_fgo_error_m"]) for r in rows if r["post_fgo_error_m"]]
            sources.append(
                {"cell": cell, "path": str(path), "sha256": sha256(path), "accepted_wrong": len(rows)}
            )
        per_dataset[key] = sorted(errs)

    # ---- the illustrated case: the median accepted leak of the case cell ----
    case_rows = leaks(args.campaign / CASE_CELL / "predictions.csv")
    case = next(r for r in case_rows if r["image_name"] == CASE_FRAME)
    anchor = json.loads(case["geodetic_anchor"])
    gt = (float(case["gt_lat"]), float(case["gt_lon"]))
    pred = (float(case["pred_lat"]), float(case["pred_lon"]))
    tile = (float(anchor["lat"]), float(anchor["lon"]))
    pred_e, pred_n = local_metres(gt[0], gt[1], *pred)
    tile_e, tile_n = local_metres(gt[0], gt[1], *tile)
    err_m = float(case["post_fgo_error_m"])
    # Distance from the forced tile to the true position, not to the original
    # selected tile (_adversarial_offset_m); the figure labels it "away".
    offset_m = float(anchor.get("_adversarial_forced_distance_to_gt_m", anchor["_adversarial_offset_m"]))
    sims = anchor["_raw_topk_similarity_scores"]

    # ---- macros ----
    mac = tables / "aicap_fig_numbers.tex"
    lines = [
        "% Auto-generated by scripts/reporting/make_aicap_leak_figure.py - DO NOT EDIT BY HAND.",
        f"% source: {args.campaign}/{CASE_CELL}/predictions.csv, frame {case['image_name']}",
        f"\\newcommand{{\\figCaseFrame}}{{{Path(case['image_name']).stem.replace('_', '-')}}}",
        f"\\newcommand{{\\figCaseOffsetKm}}{{{offset_m/1000:.1f}}}",
        f"\\newcommand{{\\figCaseErrM}}{{{err_m:,.0f}}}".replace(",", "{,}"),
        f"\\newcommand{{\\figCaseErrOverR}}{{{err_m/radius[CASE_KEY]:.0f}}}",
        f"\\newcommand{{\\figCaseInliers}}{{{int(float(case['num_inliers']))}}}",
        f"\\newcommand{{\\figCaseMatches}}{{{int(float(case['num_matches']))}}}",
        f"\\newcommand{{\\figCaseReproj}}{{{float(case['reproj_residual_px']):.1f}}}",
        f"\\newcommand{{\\figCaseShape}}{{{float(case['shape_confidence']):.2f}}}",
        f"\\newcommand{{\\figCaseRank}}{{{int(float(case['candidate_rank_selected']))}}}",
        f"\\newcommand{{\\figCaseSimTop}}{{{max(sims):.2f}}}",
        f"\\newcommand{{\\figCaseSimForced}}{{{float(anchor['_candidate_quality_score']):.2f}}}",
        f"\\newcommand{{\\figCasePoolSize}}{{{int(anchor['_adversarial_candidate_pool_size'])}}}",
    ]
    for key, errs in per_dataset.items():
        lines.append(f"\\newcommand{{\\figLeakN{key}}}{{{len(errs)}}}")
        if errs:
            lines.append(f"\\newcommand{{\\figLeakMin{key}}}{{{errs[0]:,.0f}}}".replace(",", "{,}"))
            lines.append(
                f"\\newcommand{{\\figLeakMed{key}}}{{{statistics.median(errs):,.0f}}}".replace(",", "{,}")
            )
            lines.append(f"\\newcommand{{\\figLeakMax{key}}}{{{errs[-1]:,.0f}}}".replace(",", "{,}"))
    mac.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # ---- panel (b) coordinates, one row per dataset ----
    rows_b = []
    for i, key in enumerate(panel_keys):
        y = len(panel_keys) - i
        # Deterministic vertical jitter (golden-ratio sequence) so that close
        # errors stay visible as separate points; the x value is unchanged.
        pts = " ".join(f"({e:.1f},{y + ((j * 0.618034) % 1.0 - 0.5) * 0.44:.3f})"
                       for j, e in enumerate(per_dataset[key]))
        rows_b.append((key, y, pts, radius[key], statistics.median(per_dataset[key])))
    yticks = ",".join(str(len(panel_keys) - i) for i in range(len(panel_keys)))
    ylabels = ", ".join(LABEL[k] for k in panel_keys)
    subtitle = ("(b) every accepted forced fix of the pre-registered held-out runs"
                if use_heldout else "(b) every accepted wrong-reference fix")
    axis_h = 3.3 if len(panel_keys) == 3 else 2.4
    # Headroom above the top row keeps the median label clear of the frame.
    ymax = len(panel_keys) + 0.75

    # Overview panel: one straight line, markers placed to scale along it.
    xs = [0.0, tile_e, pred_e]
    lo, hi = min(xs), max(xs)
    span = 6.0
    scale = span / (hi - lo) if hi > lo else 1.0
    gt_x = (0.0 - lo) * scale + 1.1
    tile_x = (tile_e - lo) * scale + 1.1
    pub_x = (pred_e - lo) * scale + 1.1
    span = span + 1.6

    fig = args.out / "figures" / "tikz" / "fig_leak.tex"
    out = [
        "% figures/tikz/fig_leak.tex -- figure BODY only (no \\begin{figure}).",
        "% Auto-generated by scripts/reporting/make_aicap_leak_figure.py. Do not edit:",
        "% regenerate it, so that the drawing and its provenance file stay in step.",
        "\\begin{tikzpicture}[font=\\footnotesize, >=stealth, line cap=round]",
        "% ---------------- (a) one accepted wrong-reference fix ----------------",
        "% The three pictures are cut by scripts/reporting/make_aicap_imagery.py",
        "% from the flight raster and the camera frame of this same trial.",
        "\\node[font=\\footnotesize\\bfseries, text=Navy, anchor=west] at (0,2.5)"
        " {(a) one accepted fix, UAV-VisLoc frame \\figCaseFrame};",
        "\\node[anchor=north west, inner sep=0, draw=Slate!50, line width=0.4pt] (imq)"
        " at (0,2.2) {\\includegraphics[width=2.7cm]{case_query.png}};",
        "\\node[anchor=north west, inner sep=0, draw=Slate!50, line width=0.4pt] (imt)"
        " at (3.05,2.2) {\\includegraphics[width=2.7cm]{case_true.png}};",
        "\\node[anchor=north west, inner sep=0, draw=Bad, line width=0.8pt] (imf)"
        " at (6.1,2.2) {\\includegraphics[width=2.7cm]{case_forced.png}};",
        "\\node[anchor=north, font=\\scriptsize, text=Navy, align=center] at (imq.south)"
        " {camera frame};",
        "\\node[anchor=north, font=\\scriptsize, text=Navy, align=center] at (imt.south)"
        " {map at the true position};",
        "\\node[anchor=north, font=\\scriptsize, text=Bad, align=center] at (imf.south)"
        " {forced tile, \\figCaseOffsetKm\\,km away};",
        "\\draw[->, draw=Bad, line width=1pt] ($(imt.east)+(0.02,0)$) -- ($(imf.west)+(-0.02,0)$);",
        "% The trial's own numbers sit beside the pictures, one fact per block.",
        "\\node[anchor=north west, align=left, font=\\scriptsize, text=Slate, draw=Slate!40,"
        " fill=Slate!5, rounded corners=1.5pt, inner sep=4pt, text width=3.3cm,"
        " minimum height=2.7cm] at ($(imf.north east)+(0.3,0)$)"
        " {\\textbf{accepted}: published fix \\figCaseErrM\\,m from the aircraft,"
        " \\figCaseErrOverR{}$\\times$ the success radius\\\\[3pt]"
        "\\textbf{checks passed}: \\figCaseInliers{} inliers of \\figCaseMatches{} matches,"
        " residual \\figCaseReproj\\,px, shape confidence \\figCaseShape\\\\[3pt]"
        "\\textbf{retrieval}: similarity \\figCaseSimForced{} for the forced tile,"
        " \\figCaseSimTop{} for the tile ranked first};",
        "% ---------------- (b) severity of every accepted leak ----------------",
        "\\node[font=\\footnotesize\\bfseries, text=Navy, anchor=west] at (0,-1.25)"
        f" {{{subtitle}}};",
        f"\\begin{{axis}}[at={{(0cm,-1.65cm)}}, anchor=north west, width=12.65cm, height={axis_h}cm,",
        f"  xmode=log, xmin=12, xmax=60000, ymin=0.4, ymax={ymax},",
        "  xlabel={position error of the published fix (m, log scale)},",
        f"  ytick={{{yticks}}}, yticklabels={{{ylabels}}},",
        "  xlabel style={font=\\scriptsize}, tick label style={font=\\scriptsize},",
        "  xtick pos=bottom, ytick pos=left,",
        "  axis line style={draw=Slate!60}, grid=none, scale only axis]",
    ]
    for key, y, pts, r, med in rows_b:
        out.append(
            f"\\addplot[only marks, mark=*, mark size=1.3pt, draw=Bad!80!black, fill=Bad, fill opacity=0.4,"
            f" line width=0.4pt] coordinates {{{pts}}};"
        )
        out.append(
            f"\\addplot[draw=Navy, line width=1.1pt] coordinates {{({med:.1f},{y-0.3}) ({med:.1f},{y+0.3})}};"
        )
        out.append(
            f"\\node[anchor=south, font=\\scriptsize, text=Navy, inner sep=1pt]"
            f" at (axis cs:{med:.1f},{y+0.3}) {{median}};"
        )
        out.append(
            f"\\addplot[draw=Good, line width=0.8pt, dashed]"
            f" coordinates {{({r},{y-0.32}) ({r},{y+0.32})}};"
        )
        out.append(
            f"\\node[anchor=west, font=\\scriptsize, text=Good] at (axis cs:{r*1.15},{y+0.22})"
            f" {{$R=\\protoRadius{key}$\\,m}};"
        )
    out += ["\\end{axis}", "\\end{tikzpicture}"]
    fig.write_text("\n".join(out) + "\n", encoding="utf-8")

    prov = args.out / "figures" / "fig_leak.provenance.json"
    prov.write_text(
        json.dumps(
            {
                "generator": "scripts/reporting/make_aicap_leak_figure.py",
                "campaign_root": str(args.campaign),
                "panel_b_source": ({k: str(v) for k, v in heldout.items()} if use_heldout
                                   else "same campaign as panel (a)"),
                "case_cell": CASE_CELL,
                "case_frame": case["image_name"],
                "case_forced_reference": anchor["_adversarial_forced_reference_name"],
                "case_original_reference": anchor["_adversarial_original_reference_name"],
                "case_offset_m": offset_m,
                "case_error_m": err_m,
                "accepted_wrong_per_dataset": {k: len(v) for k, v in per_dataset.items()},
                "sources": sources,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote {fig}")
    print(f"wrote {mac}")
    print(f"wrote {prov}")


if __name__ == "__main__":
    main()
