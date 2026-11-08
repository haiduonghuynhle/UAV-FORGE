"""Generate v3 thesis tables (LaTeX + markdown preview) from matrix run artifacts.

Usage:
    python3 scripts/reporting/gen_v3_tables.py <results_root> [--out-dir <dir>]

<results_root> is the matrix output root (e.g. /workspace/forge/results/v3): each
cell dir contains <dataset>/<pipeline>/<timestamp>/metrics_summary.json.
The newest timestamped run per (cell, dataset, pipeline) wins.

Every emitted number traces to an artifact: each table row carries a LaTeX
comment with the source run dir. Both mean and median (and trimmed mean) are
emitted so the final pick can be made at review time. Numbers for prose are
emitted as LaTeX macros in v3_numbers.tex; missing cells produce \\todo
placeholder tables instead of failing, so the thesis builds before the
matrix finishes.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# One rank-based AUROC in this repo, not three: the Mann-Whitney statistic with
# half credit for ties, lifted from the committed per-dataset sweep script.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from caves_per_dataset_operating_points import auroc as _rank_auroc  # noqa: E402

# Display names + row order for the localization tables (per advisor notes:
# ONE primary "ours" first, then variants, then natives).
METHOD_LABELS = {
    "ours_ultravpr_cascade_6dof_e2e": "Ours (UltraVPR-Cascade)",
    "ours_camp_cascade_6dof_e2e": "Ours-var (CAMP backbone)",
    "ours_dsm_pnp_6dof_e2e": "Ours-var (DSM-PnP head)",
    "ultravpr_roma_homography_wrapper_e2e": "Naive wrapper (top-1 + RoMa)",
    "ours_no_ipm": "Ours w/o IPM",
    "ours_no_fgo": "Ours w/o FGO",
    "ours_no_lgi": "Ours w/o LGI",
    "ours_minimal": "Ours minimal",
    "ours_fast_top1": "Ours fast top-1",
    "native_anyvisloc_e2e": "AnyVisLoc (reprod.)",
    "native_ortholoc_2d3d_e2e": "OrthoLoC (reprod.)",
    "native_ortholoc_2d3d_oracle_tile": "OrthoLoC (reprod., tile given)",
    "retrieval_megaloc": "MegaLoc (retrieval-only)",
}
METHOD_ORDER = list(METHOD_LABELS)

# R@1 hit threshold per dataset, matched to reference geometry:
# vpair refs are individual views spaced ~20m -> 20m radial;
# avl/uav_visloc refs are 614m tiles (2048px x 0.3m/px) -> 150m ~ "selected
# tile is the containing or an adjacent overlapping tile" (20m would demand a
# tile CENTER within 20m, structurally near-impossible on a 154m step grid).
RETRIEVAL_TABLE_THRESHOLD_M = {"vpair": 20, "avl": 150, "uav_visloc": 150}

# Wrong-retrieval definition per dataset (FORGE motivation stats):
# vpair: radial anchor error > 20m; tiled datasets: GT NOT contained in the
# selected tile footprint (per-axis > half-tile 307m + 15m georef slack).
TILE_SLACK_M = 15.0

# Reference footprint half-extents (east, north) in metres, measured from the
# rasters the pipeline actually tiles -- not from nominal figures.
#   AVL:        0.061 m/px orthomosaic, 2048 px tiles  -> 125 m square.
#   VPAIR:      800x600 px views at ~0.42 m/px         -> 336 x 252 m.
#   UAV-VisLoc: EPSG:4326 rasters have square DEGREE pixels, so the ground
#               pixel is anisotropic: 0.233-0.260 m/px east but 0.297 m/px
#               north on every flight -> 477-531 m east by ~609 m north.
REF_HALF_M = {
    "avl": (62.46, 62.46),
    "vpair": (168.0, 126.0),
    "uav_visloc": (254.0, 305.8),
}

# Query footprint half-extent. Matching needs shared ground, so the query's own
# footprint extends how far a reference can sit and still be usable.
#   AVL publishes intrinsics (3.6 mm focal, 7.81 mm sensor, 4000 px), so this
#   is computed per frame from altitude rather than frozen into a constant.
#   VPAIR queries are the same 800x600 views at the same GSD as its references.
#   UAV-VisLoc publishes no intrinsics; its allowance is empirical (see below).
AVL_QUERY_HALF_PER_M_ALT = 2000.0 / (3.6 / 7.81 * 4000.0)   # 1.0847 m per m of altitude
VISLOC_QUERY_HALF_M = 101.0

# Localization radius is READ FROM THE DATA (localization_success_threshold_m),
# never reused from RETRIEVAL_TABLE_THRESHOLD_M: AVL records 20 m as its
# localization radius while its R@1 retrieval threshold is 150 m, and
# conflating the two silently recalibrated every AVL figure.
DEFAULT_RADIUS_M = {"avl": 20.0, "vpair": 20.0, "uav_visloc": 150.0}


# Results directories whose CAVES AUROC values were overwritten IN PLACE by
# scripts/reporting/apply_caves_auroc_boost.py (it rewrites full_caves to
# geometry_only x 1.05 from hard-coded baselines). No thesis number may be
# sourced from them again, so the refusal is mechanical rather than a memo.
QUARANTINED_RESULT_PREFIXES = ("caves_avl487_vpair1000_", "caves_combined_")


def refuse_quarantined_path(path) -> None:
    """Abort if `path` names a directory rewritten by the AUROC boost script."""
    for part in Path(path).parts:
        if part.startswith(QUARANTINED_RESULT_PREFIXES):
            raise SystemExit(
                f"refusing to read quarantined results dir '{part}' ({path}): its "
                "CAVES AUROC values were overwritten in place by "
                "scripts/reporting/apply_caves_auroc_boost.py "
                "(full_caves := geometry_only x 1.05, from hard-coded baselines), "
                "so they are not regenerable from artifacts. Read results/v3."
            )


def localization_radius_m(rows: List[Dict[str, str]], ds: str) -> float:
    for row in rows:
        try:
            v = float(row["localization_success_threshold_m"])
            if v > 0:
                return v
        except (KeyError, ValueError, TypeError):
            continue
    return DEFAULT_RADIUS_M.get(ds, 20.0)


def query_half_m(ds: str, row: Dict[str, str]) -> tuple:
    """Query footprint half-extents (east, north) for one frame."""
    if ds == "avl":
        try:
            alt = float(row["gt_alt"])
        except (KeyError, ValueError, TypeError):
            alt = 119.0                      # dataset median, only if altitude is missing
        h = AVL_QUERY_HALF_PER_M_ALT * alt
        return h, h
    if ds == "vpair":
        return REF_HALF_M["vpair"]           # queries are the same views as the references
    return VISLOC_QUERY_HALF_M, VISLOC_QUERY_HALF_M


COND_DISPLAY = {"wrong_near": "proximal", "wrong_far": "distal", "visually_similar_wrong": "visually similar", "clean": "clean (control)"}
FORGE_CONDITIONS = ["wrong_near", "wrong_far", "visually_similar_wrong"]

DATASET_DISPLAY = {"avl": "AVL", "vpair": "VPAIR", "uav_visloc": "UAV-VisLoc"}


def tex_escape(s: str) -> str:
    return str(s).replace("_", "\\_")


def pending(name: str) -> str:
    return f"\\todo{{pending: {tex_escape(name)}}}"


def _get(d: Dict[str, Any], *keys, default=None):
    for k in keys:
        v = d.get(k)
        if v is not None:
            return v
    return default


def load_runs(results_root: Path) -> List[Dict[str, Any]]:
    """Newest metrics_summary.json per (cell, dataset, pipeline)."""
    latest: Dict[tuple, Path] = {}
    for p in sorted(results_root.glob("*/*/*/*/metrics_summary.json")):
        manifest_path = p.parent / "run_manifest.json"
        if not manifest_path.exists():
            print(f"[warn] missing {manifest_path}; run excluded", file=sys.stderr)
            continue
        try:
            manifest = json.loads(manifest_path.read_text())
        except Exception as exc:
            print(f"[warn] unreadable {manifest_path}: {exc}", file=sys.stderr)
            continue
        # Legacy manifests have no completion marker. New runners write false
        # at startup and true only after every result artifact.
        if manifest.get("run_completed") is False:
            continue
        cell, dataset, pipeline, ts = p.parts[-5], p.parts[-4], p.parts[-3], p.parts[-2]
        key = (cell, dataset, pipeline)
        if key not in latest or ts > latest[key].parts[-2]:
            latest[key] = p
    runs = []
    for (cell, dataset, pipeline), path in sorted(latest.items()):
        try:
            summary = json.loads(path.read_text())
            manifest = json.loads((path.parent / "run_manifest.json").read_text())
        except Exception as exc:
            print(f"[warn] unreadable {path}: {exc}", file=sys.stderr)
            continue
        runs.append({
            "cell": cell,
            "dataset": dataset,
            "pipeline": pipeline,
            "run_dir": str(path.parent),
            "summary": summary,
            "manifest": manifest,
        })
    return runs


def _alt_median_from_detailed(run_dir: str) -> Optional[float]:
    """Median |altitude error| from results_detailed.json.

    The native AnyVisLoc adapter never populates predictions.csv:pred_alt (it
    pre-initialises the field and only writes src_alt_m), so the per-frame CSV
    path finds nothing even though the estimate exists.
    """
    p = Path(run_dir) / "results_detailed.json"
    if not p.exists():
        return None
    try:
        j = json.loads(p.read_text())
    except Exception:
        return None
    errs = []
    for row in (j.get("predictions") or j.get("results") or []):
        v = (row.get("native_result") or {}).get("alt_error_m")
        try:
            v = abs(float(v))
        except (TypeError, ValueError):
            continue
        if v == v:
            errs.append(v)
    return statistics.median(errs) if errs else None


def _alt_median_from_predictions(run_dir: str) -> Optional[float]:
    """Median absolute altitude error over accepted frames.

    The summary only carries the mean (E_alt_gt); the tables report medians for
    position and yaw, so altitude is recovered per frame to match.
    """
    errs = []
    for row in _load_predictions(run_dir):
        try:
            if str(row.get("accepted_for_metrics", "True")).lower() not in ("true", "1"):
                continue
            errs.append(abs(float(row["pred_alt"]) - float(row["gt_alt"])))
        except (KeyError, ValueError, TypeError):
            continue
    if not errs:
        return None
    return statistics.median(errs)


def _yaw_median_from_predictions(run_dir: str) -> Optional[float]:
    """Median heading error from per-frame records.

    Native wrappers record pred_heading and gt_heading but leave the derived
    yaw_error_deg column empty, so the summary carries no yaw median even
    though the estimate exists. Recover it rather than printing "--", which
    readers take to mean the method cannot estimate heading.

    Accepted frames only, exactly like the altitude helper above and like the
    summary's own yaw median: a table column whose denominator changes from row
    to row is not a column. Without the filter the native rows averaged in
    frames the method itself rejected, including pred_heading=0 placeholders
    whose ground truth sits near +-179 degrees.
    """
    errs = []
    preds = []
    for row in _load_predictions(run_dir):
        if str(row.get("accepted_for_metrics", "True")).lower() not in ("true", "1"):
            continue
        try:
            pred = float(row["pred_heading"])
            gt = float(row["gt_heading"])
        except (KeyError, ValueError, TypeError):
            continue
        preds.append(pred)
        errs.append(abs((pred - gt + 180.0) % 360.0 - 180.0))
    if not errs:
        return None
    # An all-zero pred_heading column is an unset placeholder, not an estimate.
    if all(abs(p) < 1e-9 for p in preds):
        return None
    return statistics.median(errs)


def row_metrics(run: Dict[str, Any]) -> Dict[str, Optional[float]]:
    s = run["summary"]
    m = s.get("metrics", {}) or {}
    pm = s.get("paper_metrics", {}) or {}
    alt_med = (_get(pm, "alt_abs_error_m_median", "E_alt_gt_median")
               or _alt_median_from_predictions(run["run_dir"])
               or _alt_median_from_detailed(run["run_dir"]))
    return {
        "n": _get(s, "num_frames"),
        "accepted": _get(s, "num_accepted_for_metrics"),
        "median_m": _get(pm, "E_ME_median", default=m.get("median_error_m")),
        "mean_m": _get(pm, "E_ME", default=m.get("mean_error_m")),
        "trimmed_m": _get(pm, "E_ME_trimmed_mean"),
        "rmse_m": m.get("rmse_m"),
        "p90_m": _get(pm, "p90_error_m"),
        "a5": m.get("accuracy_at_5m"),
        "a10": m.get("accuracy_at_10m"),
        "yaw_med": (_get(pm, "yaw_error_deg_median")
                    if _get(pm, "yaw_error_deg_median") is not None
                    else _yaw_median_from_predictions(run["run_dir"])),
        "alt_med": alt_med if alt_med is not None else _get(pm, "E_alt_gt", "E_alt"),
        "sr": m.get("success_rate"),
        "fps": m.get("fps"),
        "r1": _get(pm, "Recall@1", "recall_at_1", default=m.get("recall_at_1")),
        "unsafe": _get(s, "num_unsafe_accept"),
        "rejection_rate": _get(s, "rejection_rate"),
    }


def fmt(v, spec=".1f", pct=False):
    if v is None:
        return "--"
    try:
        v = float(v)
    except (TypeError, ValueError):
        return "--"
    if pct:
        return f"{100.0 * v:.1f}\\%"
    return f"{v:{spec}}"


# ---------------------------------------------------------------------------
# predictions.csv helpers (nadir/oblique split, wrong-retrieval motivation)
# ---------------------------------------------------------------------------

def _load_predictions(run_dir: str) -> List[Dict[str, str]]:
    p = Path(run_dir) / "predictions.csv"
    if not p.exists():
        return []
    with open(p) as f:
        return list(csv.DictReader(f))


def _anchor_latlon(row: Dict[str, str]) -> Optional[tuple]:
    """(lat, lon) of the reference the run actually selected, or None.

    None means the frame never got a reference at all -- not that the reference
    was wrong. Two UAV-VisLoc frames are in this state in every cell.
    """
    anch = str(row.get("geodetic_anchor", "") or "")
    if not anch.startswith("{"):
        return None
    try:
        a = json.loads(anch.replace("'", '"'))
        return float(a["lat"]), float(a["lon"])
    except (ValueError, KeyError, TypeError):
        return None


def _truthy(value) -> bool:
    """CSV booleans arrive as strings; treat only explicit truth as true."""
    return str(value).strip().lower() in {"true", "1", "yes"}


def _control_rows(run_dir: str) -> List[Dict[str, str]]:
    """Frames a mixed-rate run assigned to the control arm (correct reference).

    Empty for a run that injected on every frame, which is what every result
    predating mixed-rate injection did.
    """
    out: List[Dict[str, str]] = []
    for row in _load_predictions(run_dir):
        if "adversarial_assigned_attack" not in row:
            continue
        if _truthy(row.get("adversarial_assigned_attack")):
            continue
        out.append(row)
    return out


def _attacked_rows(run_dir: str) -> List[Dict[str, str]]:
    """Frames the injector actually forced a wrong reference onto.

    Two exclusions, and both belong in the denominator decision:
      * a non-empty adversarial_failure_reason means the shortlist held no
        eligible wrong reference, so the cell failed closed and never put that
        frame at risk;
      * a frame with no parsable selected-reference anchor (or no parsable GT)
        never had a reference forced onto it at all.
    Counting either as a safe accept rewards a system for being unattackable
    and narrows the interval on a sample that was never drawn. This is the one
    definition of the injected population: the row table, the stratified table,
    the macros and the paired clean control all read it from here, so they
    cannot disagree about how many frames were attacked.
    """
    out: List[Dict[str, str]] = []
    for row in _load_predictions(run_dir):
        if "adversarial_attacked" in row and not _truthy(row.get("adversarial_attacked")):
            continue
        if str(row.get("adversarial_failure_reason", "") or "").strip():
            continue
        if _anchor_latlon(row) is None:
            continue
        try:
            float(row["gt_lat"]), float(row["gt_lon"])
        except (KeyError, ValueError, TypeError):
            continue
        out.append(row)
    return out


def _per_axis_m(lat1: float, lon1: float, lat2: float, lon2: float) -> tuple:
    dn = (lat1 - lat2) * 111320.0
    de = (lon1 - lon2) * 111320.0 * math.cos(math.radians(lat1))
    return abs(de), abs(dn)


# AnyVisLoc footprint model. "nadir" (the cells of record) treats every frame
# as looking straight down. Most attacked AnyVisLoc frames are oblique (gimbal
# pitch -45 deg), and an oblique camera sees ground far ahead, so "oblique"
# projects the four image corners onto flat ground with the pipeline's
# intrinsics and the frame's gimbal attitude, and tests the tile (+ slack)
# against that quadrilateral. Frames whose top edge reaches the horizon have
# no bounded footprint and are treated as overlapping (never a fault).
AVL_FOOTPRINT = os.environ.get("FORGE_AVL_FOOTPRINT", "nadir")
AVL_ALT_SCALE = float(os.environ.get("FORGE_AVL_ALT_SCALE", "1.0"))
AVL_IMG_W, AVL_IMG_H, AVL_F_PX = 4000.0, 3000.0, 3.6 / 7.81 * 4000.0


def avl_oblique_footprint(alt: float, yaw_deg: float, pitch_deg: float,
                          roll_deg: float = 0.0) -> Optional[List[Tuple[float, float]]]:
    """Ground quadrilateral (east, north) of the image corners, or None if unbounded.

    DJI gimbal attitude: yaw clockwise from north, pitch -90 = nadir. Checked
    on the record AnyVisLoc trajectory: on 300 east-west legs the yaw matches
    the course within 30 deg for 148 pairs read clockwise and 26 read
    counter-clockwise; roll never exceeds 0.2 deg, so its sign is immaterial.
    """
    dep, psi, rl = math.radians(-pitch_deg), math.radians(yaw_deg), math.radians(roll_deg)
    fwd = (math.sin(psi), math.cos(psi), 0.0)
    right = (math.cos(psi), -math.sin(psi), 0.0)
    axis = (math.cos(dep) * fwd[0], math.cos(dep) * fwd[1], -math.sin(dep))
    up = (math.sin(dep) * fwd[0], math.sin(dep) * fwd[1], math.cos(dep))
    r2 = tuple(math.cos(rl) * right[i] + math.sin(rl) * up[i] for i in range(3))
    u2 = tuple(-math.sin(rl) * right[i] + math.cos(rl) * up[i] for i in range(3))
    pts = []
    for x, y in ((-AVL_IMG_W / 2, -AVL_IMG_H / 2), (AVL_IMG_W / 2, -AVL_IMG_H / 2),
                 (AVL_IMG_W / 2, AVL_IMG_H / 2), (-AVL_IMG_W / 2, AVL_IMG_H / 2)):
        d = [axis[i] + (x / AVL_F_PX) * r2[i] - (y / AVL_F_PX) * u2[i] for i in range(3)]
        if d[2] >= -1e-6:
            return None
        t = alt / -d[2]
        pts.append((t * d[0], t * d[1]))
    return pts


def _square_separated(poly: List[Tuple[float, float]], cx: float, cy: float, half: float) -> bool:
    """Separating-axis test: convex polygon vs the square [cx +- half] x [cy +- half]."""
    sq = [(cx - half, cy - half), (cx + half, cy - half), (cx + half, cy + half), (cx - half, cy + half)]
    axes = [(1.0, 0.0), (0.0, 1.0)]
    for i in range(len(poly)):
        (x1, y1), (x2, y2) = poly[i], poly[(i + 1) % len(poly)]
        axes.append((y2 - y1, x1 - x2))
    for ax in axes:
        pa = [p[0] * ax[0] + p[1] * ax[1] for p in poly]
        pb = [p[0] * ax[0] + p[1] * ax[1] for p in sq]
        if max(pa) < min(pb) or max(pb) < min(pa):
            return True
    return False


def _avl_oblique_can_localize(gt_lat, gt_lon, ref_lat, ref_lon, row, extra_margin_frac=0.0) -> bool:
    try:
        alt = float(row["gt_alt"]) * AVL_ALT_SCALE
        att = (float(row["gt_heading"]), float(row["gt_pitch"]), float(row.get("gt_roll") or 0.0))
    except (KeyError, TypeError, ValueError):
        return True                           # no attitude: cannot certify a fault
    if not (math.isfinite(alt) and alt > 0.0 and all(math.isfinite(v) for v in att)):
        return True
    poly = avl_oblique_footprint(alt, *att)
    if poly is None:
        return True
    de = (ref_lon - gt_lon) * 111320.0 * math.cos(math.radians(gt_lat))
    dn = (ref_lat - gt_lat) * 111320.0
    half = (REF_HALF_M["avl"][0] + TILE_SLACK_M) * (1.0 + max(0.0, extra_margin_frac))
    return not _square_separated(poly, de, dn, half)


def selected_reference_can_localize(gt_lat: float, gt_lon: float,
                                    ref_lat: float, ref_lon: float,
                                    ds: str, row: Dict[str, str],
                                    extra_margin_frac: float = 0.0) -> bool:
    """True when reference and query share ground, so a pose is recoverable.

    Overlap, not containment of the query centre: a reference whose footprint
    reaches the query can be matched even when the query centre lies outside
    it. Both footprints are measured per axis, and the query side is per frame
    wherever the dataset publishes intrinsics. AnyVisLoc uses the oblique
    projected footprint when FORGE_AVL_FOOTPRINT=oblique.
    """
    if ds == "avl" and AVL_FOOTPRINT == "oblique":
        return _avl_oblique_can_localize(gt_lat, gt_lon, ref_lat, ref_lon, row, extra_margin_frac)
    de, dn = _per_axis_m(gt_lat, gt_lon, ref_lat, ref_lon)
    rh_e, rh_n = REF_HALF_M.get(ds, (TILE_SLACK_M, TILE_SLACK_M))
    qh_e, qh_n = query_half_m(ds, row)
    m = 1.0 + max(0.0, extra_margin_frac)
    return (de <= (rh_e + qh_e + TILE_SLACK_M) * m
            and dn <= (rh_n + qh_n + TILE_SLACK_M) * m)


def wrong_retrieval_stats(run: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Per-run wrong-retrieval-but-accepted stats (FORGE motivation)."""
    rows = _load_predictions(run["run_dir"])
    if not rows:
        return None
    ds = run["dataset"]
    n = 0
    wrong = []
    offsets = []
    for r in rows:
        try:
            gt_lat, gt_lon = float(r["gt_lat"]), float(r["gt_lon"])
        except (KeyError, ValueError):
            continue
        anch = r.get("geodetic_anchor", "")
        if not anch.startswith("{"):
            continue
        try:
            a = json.loads(anch.replace("'", '"'))
            is_wrong = not selected_reference_can_localize(
                gt_lat, gt_lon, float(a["lat"]), float(a["lon"]), ds, r)
        except (ValueError, KeyError, TypeError):
            continue
        n += 1
        if is_wrong:
            wrong.append(r)
            # How far the selected tile sits from the truth. This is the error
            # the pipeline would have reported had it trusted that tile, so it
            # is defined for every wrong retrieval and not only for the ones
            # that survived the gates. Without it the two datasets that accept
            # none of their wrong retrievals print an empty pair of cells, and
            # the strongest result in the table reads as missing work.
            dn = (float(a["lat"]) - gt_lat) * 111320.0
            de = (float(a["lon"]) - gt_lon) * 111320.0 * math.cos(math.radians(gt_lat))
            offsets.append(math.hypot(dn, de))
    wrong_acc = [r for r in wrong if r.get("accepted_for_metrics") == "True"]
    errs = []
    for r in wrong_acc:
        try:
            errs.append(float(r["post_fgo_error_m"]))
        except (KeyError, ValueError):
            pass
    errs.sort()
    offsets.sort()
    # Offsets of the tiles that actually leaked, kept apart from the offsets of
    # every wrong retrieval. Prose that says "where it leaks, the tiles sit far
    # from truth" is a statement about this subset, and the two populations do
    # not have the same spread: on UAV-VisLoc the all-wrong median is about
    # 1.9 km while a third of the leaking tiles sit under 1 km.
    leak_offsets = []
    for r in wrong_acc:
        try:
            gt_lat, gt_lon = float(r["gt_lat"]), float(r["gt_lon"])
            a = json.loads(str(r.get("geodetic_anchor", "")).replace("'", '"'))
            dn = (float(a["lat"]) - gt_lat) * 111320.0
            de = (float(a["lon"]) - gt_lon) * 111320.0 * math.cos(math.radians(gt_lat))
            leak_offsets.append(math.hypot(dn, de))
        except (ValueError, KeyError, TypeError):
            continue
    leak_offsets.sort()
    return {
        "n": n,
        "wrong": len(wrong),
        "wrong_accepted": len(wrong_acc),
        "err_median": statistics.median(errs) if errs else None,
        "err_max": errs[-1] if errs else None,
        "offset_median": statistics.median(offsets) if offsets else None,
        "offset_max": offsets[-1] if offsets else None,
        "leak_offset_min": leak_offsets[0] if leak_offsets else None,
        "leak_offset_median": statistics.median(leak_offsets) if leak_offsets else None,
        "leak_offset_max": leak_offsets[-1] if leak_offsets else None,
        "leak_err_min": errs[0] if errs else None,
        "leak_under_km": sum(1 for o in leak_offsets if o < 1000.0) or None,
    }


def _frame_error_m(row: Dict[str, str]) -> Optional[float]:
    """Horizontal position error of one frame, in meters.

    post_fgo_error_m is the value of record. The native-baseline adapters leave
    it empty, so the fallback recomputes it from the predicted and the true
    coordinate; on the AnyVisLoc AVL run the fallback reproduces the pooled
    median, mean and A@10 of its own summary to the printed precision.
    """
    try:
        v = float(row["post_fgo_error_m"])
        if v == v:
            return v
    except (KeyError, ValueError, TypeError):
        pass
    try:
        gt_lat, gt_lon = float(row["gt_lat"]), float(row["gt_lon"])
        pr_lat, pr_lon = float(row["pred_lat"]), float(row["pred_lon"])
    except (KeyError, ValueError, TypeError):
        return None
    north, east = _per_axis_m(pr_lat, pr_lon, gt_lat, gt_lon)
    return math.hypot(north, east)


def nadir_oblique_split(run: Dict[str, Any]) -> Optional[Dict[str, Dict[str, Any]]]:
    """AVL: split accepted metrics by gimbal pitch.

    Three bands: nadir (<15deg from nadir), mid (15-40deg), oblique (>=40deg).
    The tables show only nadir and oblique; mid is carried so that a band with
    ZERO accepted frames is a computed fact with an empty numerator rather than
    a silent gap. Every derived cell is guarded on its own denominator, so an
    empty band renders the missing marker and never 0.0.

    The native adapters accept every frame but leave post_fgo_error_m empty, so
    the per-frame error falls back to the great-circle distance between the
    predicted and the true position. That fallback reproduces the pooled native
    row exactly, which is what makes a band-wise reading of a baseline possible
    at all. None when neither source yields a single accepted error.
    """
    rows = _load_predictions(run["run_dir"])
    if not rows:
        return None
    bands = ("nadir", "mid", "oblique")
    out: Dict[str, Dict[str, Any]] = {}
    groups = {k: [] for k in bands}
    yaws = {k: [] for k in bands}
    alts = {k: [] for k in bands}
    counts = {k: 0 for k in bands}
    accepted = {k: 0 for k in bands}
    for r in rows:
        try:
            from_nadir = abs(90.0 - abs(float(r["gt_pitch"])))
        except (KeyError, ValueError):
            continue
        key = "nadir" if from_nadir < 15.0 else ("oblique" if from_nadir >= 40.0 else "mid")
        counts[key] += 1
        if r.get("accepted_for_metrics") == "True":
            accepted[key] += 1
            e = _frame_error_m(r)
            if e is not None:
                groups[key].append(e)
            try:
                y = float(r["yaw_error_deg"])
                if y == y:
                    yaws[key].append(y)
            except (KeyError, ValueError):
                pass
            try:
                alts[key].append(abs(float(r["pred_alt"]) - float(r["gt_alt"])))
            except (KeyError, ValueError, TypeError):
                pass
    if not any(groups.values()):
        return None
    for key, errs in groups.items():
        errs.sort()
        nacc = len(errs)
        ys = sorted(yaws[key])
        out[key] = {
            "n": counts[key],
            # Accepted frames, not frames with a parsable error: a row whose
            # error column is empty must not read as an acceptance of zero.
            "accepted": accepted[key],
            "median_m": statistics.median(errs) if errs else None,
            "mean_m": (sum(errs) / nacc) if errs else None,
            "yaw_med": statistics.median(ys) if ys else None,
            "a5": (sum(1 for e in errs if e <= 5.0) / nacc) if nacc else None,
            "a10": (sum(1 for e in errs if e <= 10.0) / nacc) if nacc else None,
            # The unconditional column divides by the band's frame count, so it
            # needs the hit count itself. Deriving it as a10 * accepted / n is
            # wrong whenever an accepted frame carries no parsable error: a10
            # divides by the frames that have one, accepted counts all of them.
            "a10_hits": sum(1 for e in errs if e <= 10.0),
            "alt_med": (statistics.median(alts[key]) if alts[key] else None),
        }
    return out


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

def _bold_header(header: str) -> str:
    """Bold each header cell with \\bfseries (matches hand-written tables)."""
    body = header.rstrip()
    suffix = ""
    if body.endswith("\\\\"):
        body, suffix = body[:-2].rstrip(), " \\\\"
    def _bold_cell(c: str) -> str:
        c = c.strip()
        if not c:
            return c
        if c.startswith("\\multicolumn"):
            # \multicolumn must open the cell; bold its content instead
            i = c.index("{", c.index("}", c.index("}") + 1)) + 1
            return c[:i] + "\\bfseries " + c[i:]
        return "\\bfseries " + c
    cells = [_bold_cell(c) for c in body.split(" & ")]
    return " & ".join(cells) + suffix


def _acc_pct(accepted, n) -> str:
    """Acceptance as a percentage; n is given in the block header or caption."""
    try:
        a, t = float(accepted), float(n)
        if t <= 0:
            return "--"
        return f"{100.0 * a / t:.1f}\\%"
    except (TypeError, ValueError):
        return "--"


def _table_head(caption: str, label: str, colspec: str, header: str,
                short: Optional[str] = None) -> List[str]:
    cap = f"\\caption[{short}]{{{caption}}}" if short else f"\\caption{{{caption}}}"
    return [
        "\\begin{table}[H]", "\\centering", cap, f"\\label{{{label}}}",
        "\\footnotesize", "\\setlength{\\tabcolsep}{3pt}",
        f"\\begin{{tabular}}{{{colspec}}}", "\\toprule", _bold_header(header), "\\midrule",
    ]


def _table_tail() -> List[str]:
    return ["\\bottomrule", "\\end{tabular}", "\\end{table}"]


def loc_table(runs: List[Dict[str, Any]], dataset: str, caption: str, label: str,
              include_pipelines: Optional[List[str]] = None,
              avl_split: bool = False,
              extra_rows: Optional[List[str]] = None,
              short: Optional[str] = None) -> str:
    rows = [r for r in runs if r["dataset"] == dataset and not r["summary"].get("retrieval_only")
            and not r["summary"].get("adversarial_diagnostic")
            and r["summary"].get("adversarial_mode", "none") == "none"
            and not _is_control_only(r)]
    if include_pipelines is not None:
        rows = [r for r in rows if r["pipeline"] in include_pipelines]
    rows.sort(key=lambda r: METHOD_ORDER.index(r["pipeline"]) if r["pipeline"] in METHOD_ORDER else 99)
    lines = _table_head(
        caption + " Accept: share of evaluated frames the method commits to. A@$N$ conditions on each "
        "method's accepted set; $\\overline{\\mathrm{A@10}}$ is unconditional "
        "(denominator: all frames), directly comparable across methods with different "
        "acceptance. Trimmed means are reported in the retained records.",
        label, "lrrrrrrrr",
        "Method & Accept & Med. (m) & Mean (m) & A@5m & A@10m & "
        "$\\overline{\\mathrm{A@10}}$ & Yaw ($^\\circ$) & FPS \\\\",
        short=short or "Localization results",
    )
    for r in rows:
        met = row_metrics(r)
        # Same rule as the merged table: the caption fixes n for the whole
        # table, so a run scored on fewer frames cannot appear here. Two
        # 30-frame smoke probes left inside the results root once printed a row
        # reading 96.7 percent acceptance at a 1.9 m median beside the real
        # 487-frame row, which a reader has no way to detect as a probe.
        if (met["n"] or 0) < BLOCK_N.get(dataset, 0):
            lines.append(f"% skipped (n={met['n']} below the block's "
                         f"{BLOCK_N.get(dataset)}): {r['run_dir']}")
            continue
        name = _method_label(r)
        a10_all = None
        if met["a10"] is not None and met["accepted"] and met["n"]:
            a10_all = float(met["a10"]) * float(met["accepted"]) / float(met["n"])
        lines.append(f"% source: {r['run_dir']} (n={met['n']}, accepted={met['accepted']})")
        lines.append(
            f"{name} & {_acc_pct(met['accepted'], met['n'])} & {fmt(met['median_m'])} & "
            f"{fmt(met['mean_m'])} & {fmt(met['a5'], pct=True)} & "
            f"{fmt(met['a10'], pct=True)} & {fmt(a10_all, pct=True)} & "
            f"{fmt(met['yaw_med'])} & {fmt(met['fps'], '.2f')} \\\\"
        )
        # Any AVL row that has a per-frame split gets one. Gating this on the
        # primary pipeline left the DSM-PnP head -- the row whose whole point is
        # the oblique regime -- with no oblique cell, so the prose had to type
        # the number by hand.
        if avl_split:
            split = nadir_oblique_split(r)
            if split:
                for key, lbl in (("nadir", "\\quad nadir ($<15^\\circ$)"),
                                 ("oblique", "\\quad oblique ($\\geq 40^\\circ$)")):
                    s = split[key]
                    s_a10_all = None
                    if s["a10"] is not None and s["n"]:
                        s_a10_all = float(s["a10_hits"]) / float(s["n"])
                    lines.append(
                        f"{lbl} & {_acc_pct(s['accepted'], s['n'])} & {fmt(s['median_m'])} & "
                        f"{fmt(s['mean_m'])} & "
                        f"{fmt(s['a5'], pct=True)} & {fmt(s['a10'], pct=True)} & "
                        f"{fmt(s_a10_all, pct=True)} & {fmt(s['yaw_med'])} & "
                        f"$\\dagger$ \\\\"
                    )
    if extra_rows:
        lines.append("\\midrule")
        lines.extend(extra_rows)
    lines += _table_tail()
    return "\n".join(lines)


def ablation_table(runs: List[Dict[str, Any]]) -> str:
    order = ["ours_ultravpr_cascade_6dof_e2e", "ours_no_ipm", "ours_no_fgo",
             "ours_no_lgi", "ours_fast_top1", "ours_minimal",
             "ultravpr_roma_homography_wrapper_e2e"]
    rows = [r for r in runs if r["dataset"] == "vpair" and r["pipeline"] in order
            and r["summary"].get("adversarial_mode", "none") == "none"
            and not r["summary"].get("retrieval_only")
            and not _is_control_only(r)]
    rows.sort(key=lambda r: order.index(r["pipeline"]))
    lines = _table_head(
        "\\textbf{Module contributions on the 800-frame VPAIR subset.} A@10m "
        "conditions on each variant's own accepted set and is not comparable "
        "across rows; $\\overline{\\mathrm{A@10}}$ divides the same hit count by "
        "all evaluated frames and is the comparable column.",
        "tab:v3-ablation", "lrrrrrrr",
        "Variant & Accept & Med. (m) & Mean (m) & A@10m & "
        "$\\overline{\\mathrm{A@10}}$ & Yaw ($^\\circ$) & FPS \\\\",
        short="Ablation on VPAIR: module contributions",
    )
    for r in rows:
        met = row_metrics(r)
        a10_all = None
        if met["a10"] is not None and met["accepted"] and met["n"]:
            a10_all = float(met["a10"]) * float(met["accepted"]) / float(met["n"])
        lines.append(f"% source: {r['run_dir']} (n={met['n']}, accepted={met['accepted']})")
        lines.append(
            f"{_method_label(r)} & {_acc_pct(met['accepted'], met['n'])} & "
            f"{fmt(met['median_m'])} & {fmt(met['mean_m'])} & {fmt(met['a10'], pct=True)} & "
            f"{fmt(a10_all, pct=True)} & "
            f"{fmt(met['yaw_med'])} & {fmt(met['fps'], '.2f')} \\\\"
        )
    lines += _table_tail()
    return "\n".join(lines)


def retrieval_table(runs: List[Dict[str, Any]]) -> str:
    rows = [r for r in runs if r["summary"].get("retrieval_only")]
    rows.sort(key=lambda r: (r["dataset"], r["pipeline"], r["summary"].get("retrieval_top_k", 0)))
    lines = _table_head(
        "\\textbf{Retrieval-only hit rates across the three datasets.} R@1 is "
        "scored within a threshold matched to each dataset's reference geometry. "
        "R@5 and R@10 bound what the top-$K$ verifier can recover from a "
        "shortlist of that size.",
        "tab:v3-retrieval", "llrrrrrr",
        "Dataset & Backbone & k & thr (m) & R@1 & R@5 & R@10 & n \\\\",
        short="Retrieval-only hit rates across the three datasets",
    )
    for r in rows:
        met = row_metrics(r)
        s = r["summary"]
        backbone = "MegaLoc" if r["pipeline"] == "retrieval_megaloc" else "UltraVPR"
        thr = RETRIEVAL_TABLE_THRESHOLD_M.get(r["dataset"], 20)
        r1 = r5 = r10 = None
        diag_path = Path(r["run_dir"]) / "retrieval_diagnostics.json"
        if diag_path.exists():
            try:
                g = json.loads(diag_path.read_text()).get("global", {})
                r1 = g.get(f"selected_error_m_hit_at_{thr}m")
                r5 = g.get(f"raw_top5_error_m_hit_at_{thr}m")
                r10 = g.get(f"raw_top10_error_m_hit_at_{thr}m")
            except Exception:
                r1 = r5 = r10 = None
        if r1 is None:
            r1 = met["r1"] if met["r1"] is not None else met["sr"]
        lines.append(f"% source: {r['run_dir']}")
        lines.append(
            f"{DATASET_DISPLAY.get(r['dataset'], r['dataset'])} & {backbone} & {s.get('retrieval_top_k', '--')} & {thr} & "
            f"{fmt(r1, pct=True)} & {fmt(r5, pct=True)} & {fmt(r10, pct=True)} & "
            f"{met['n'] or '--'} \\\\"
        )
    lines += _table_tail()
    return "\n".join(lines)


def motivation_table(runs: List[Dict[str, Any]]) -> str:
    """Wrong-retrieval-but-accepted stats on clean primary runs (FORGE motivation)."""
    targets = [r for r in runs
               if r["pipeline"] == "ours_ultravpr_cascade_6dof_e2e"
               and r["summary"].get("adversarial_mode", "none") == "none"
               and not r["summary"].get("retrieval_only")
               and not _is_control_only(r)]
    lines = _table_head(
        "\\textbf{Wrong retrievals the pipeline still accepts on clean runs.} "
        "Wrong is Definition~\\ref{def:wrong-reference}. The last two columns "
        "give the distance from the selected tile to the true position, over "
        "every wrong retrieval rather than only the accepted ones, and they "
        "therefore state the position error each dataset was exposed to. Two "
        "UAV-VisLoc frames carry no parsable anchor and are excluded, hence "
        "$n{=}2143$.",
        "tab:v3-forge-motivation", "lrrrrr",
        "Dataset & n & Wrong retr. & \\dots accepted & Tile offset, med. (m) & "
        "Tile offset, max (m) \\\\",
        short="Wrong retrievals accepted on clean runs",
    )
    for r in sorted(targets, key=lambda x: x["dataset"]):
        st = wrong_retrieval_stats(r)
        if st is None:
            continue
        # Same rule as the merged localization table: a development probe scored
        # on a fraction of the subset does not belong beside the full-set row it
        # shares a dataset name with. The run's own frame count is the test, not
        # st["n"], which drops the frames that carry no parsable anchor.
        if (row_metrics(r)["n"] or 0) < BLOCK_N.get(r["dataset"], 0):
            continue
        lines.append(f"% source: {r['run_dir']}")
        frac_w = st["wrong"] / max(1, st["n"])
        frac_a = st["wrong_accepted"] / max(1, st["wrong"])
        lines.append(
            f"{DATASET_DISPLAY.get(r['dataset'], r['dataset'])} & {st['n']} & {st['wrong']} ({100*frac_w:.1f}\\%) & "
            f"{st['wrong_accepted']} ({100*frac_a:.1f}\\%) & {fmt(st['offset_median'], '.0f')} & "
            f"{fmt(st['offset_max'], '.0f')} \\\\"
        )
    lines += _table_tail()
    return "\n".join(lines)


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple:
    """Wilson score 95% CI for a binomial proportion."""
    if n <= 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, center - half), min(1.0, center + half))


def _forge_row(r: Dict[str, Any], system: str, cond: str, thr: float) -> List[str]:
    # Only frames the injector could actually attack belong in the denominator
    # (see _attacked_rows).
    attacked = _attacked_rows(r["run_dir"])
    if attacked:
        n = len(attacked)
        acc = sum(1 for row in attacked if row.get("accepted_for_metrics") == "True")
        unsafe = 0
        severe = 0
        for row in attacked:
            if row.get("accepted_for_metrics") != "True":
                continue
            try:
                e = float(row["post_fgo_error_m"])
            except (KeyError, ValueError, TypeError):
                continue
            if e > thr:
                unsafe += 1
            if e > 2.5 * thr:
                severe += 1
    else:
        met = row_metrics(r)
        acc = met["accepted"] or 0
        n = met["n"] or 0
        unsafe = met["unsafe"] or 0
        severe = _count_severe(r, thr)
    lo, hi = wilson_ci(unsafe, n)
    rej = 1.0 - acc / max(1, n)
    return [
        f"% source: {r['run_dir']}",
        f"{system} & {COND_DISPLAY.get(cond, cond.replace('_', ' '))} & {n} & {acc} ({100*acc/max(1,n):.1f}\\%) & "
        f"{unsafe} ({100*unsafe/max(1,n):.1f}\\%) "
        f"[{100*lo:.1f}, {100*hi:.1f}] & "
        f"{severe if severe is not None else '--'} & {fmt(rej, pct=True)} \\\\",
    ]


def _is_genuine_fault(row: Dict[str, str], ds: str) -> bool:
    """True when the forced reference really excludes the query.

    One definition of a genuine wrong-reference trial. The stratified table and
    the reject-code table both read it from here, so they cannot disagree about
    which trials count as attacks.
    """
    ref_lat, ref_lon = _anchor_latlon(row)
    return not selected_reference_can_localize(
        float(row["gt_lat"]), float(row["gt_lon"]), ref_lat, ref_lon, ds, row)


def _fault_rows(r: Dict[str, Any], ds: str) -> List[Dict[str, str]]:
    """Attacked frames of one injection cell whose forced reference is genuine."""
    return [row for row in _attacked_rows(r["run_dir"]) if _is_genuine_fault(row, ds)]


def _forge_strata(r: Dict[str, Any], ds: str, thr: float) -> Dict[str, List[int]]:
    """Split an injection cell by whether the forced reference really excludes GT.

    A forced reference whose footprint still covers the query is one the
    geometry head can localize from, so accepting it is correct behaviour and
    pooling it with the true faults understates measured fault rejection.
    """
    out = {"fault": [0, 0, 0], "covered": [0, 0, 0]}   # n, accepted, unsafe
    for row in _attacked_rows(r["run_dir"]):
        key = "fault" if _is_genuine_fault(row, ds) else "covered"
        out[key][0] += 1
        if row.get("accepted_for_metrics") == "True":
            out[key][1] += 1
            try:
                if float(row["post_fgo_error_m"]) > thr:
                    out[key][2] += 1
            except (KeyError, ValueError):
                pass
    return out


def forge_stratified_table(runs: List[Dict[str, Any]]) -> str:
    """FORGE outcomes split into genuine faults vs references that still cover GT."""
    lines = _table_head(
        "\\textbf{FORGE injection outcomes on AVL, VPAIR and UAV-VisLoc}, split "
        "by injection condition. Every trial forces a reference that excludes the "
        "query footprint, so every row is a genuine fault. Frames is the count "
        "of attacked frames and Trials counts each of them once per condition, "
        "so the three trials on one frame are not independent. Brackets are "
        "Wilson 95\\% intervals on the accept rate.",
        "tab:v3-forge-stratified", "llrrrr",
        "Dataset & Condition & Frames & Trials & Accepted [CI] & Unsafe \\\\",
        short="FORGE outcomes stratified by genuine fault",
    )
    found = False
    for ds in ("avl", "vpair", "uav_visloc"):
        thr = DEFAULT_RADIUS_M.get(ds, 20.0)
        stem = FORGE_MACRO_CELL_STEM.get(ds, ds)
        pooled = {"fault": [0, 0, 0], "covered": [0, 0, 0]}
        dsname = DATASET_DISPLAY.get(ds, ds)
        body: List[str] = []
        frame_names = set()
        for cond in FORGE_CONDITIONS:
            # Prefer the containment-eligibility cells; fall back to the originals.
            match = _forge_cell(runs, f"forge_{stem}", cond)
            if not match:
                continue
            found = True
            st = _forge_strata(match[-1], ds, thr)
            for k in pooled:
                for i in range(3):
                    pooled[k][i] += st[k][i]
            f = st["fault"]
            frame_names.update(row.get("image_name", "")
                               for row in _fault_rows(match[-1], ds))
            lo, hi = wilson_ci(f[1], f[0])
            body.append(f"% source: {match[-1]['run_dir']}")
            body.append(
                f"{dsname} & {COND_DISPLAY.get(cond, cond.replace('_', ' '))} & "
                f"{f[0]} & {f[0]} & "
                f"{fmt(f[1]/f[0] if f[0] else None, pct=True)} [{100*lo:.1f}, {100*hi:.1f}] & "
                f"{fmt(f[2]/f[0] if f[0] else None, pct=True)} \\\\")
        if not body:
            continue
        lines += body
        f = pooled["fault"]
        frames_n = len(frame_names)
        lo, hi = wilson_ci(f[1], f[0])
        lines.append(
            f"{dsname} & \\quad pooled & {frames_n} & \\textbf{{{f[0]}}} & "
            f"\\textbf{{{fmt(f[1]/f[0] if f[0] else None, pct=True)}}} "
            f"[{100*lo:.1f}, {100*hi:.1f}] & "
            f"{fmt(f[2]/f[0] if f[0] else None, pct=True)} \\\\")
        lines.append("\\midrule")
    if lines[-1] == "\\midrule":
        lines.pop()
    lines += _table_tail()
    tex = "\n".join(lines)
    return tex if found else "% FORGE cells not available yet\n" + tex


def _count_severe(r: Dict[str, Any], thr: float) -> Optional[int]:
    """Accepted frames with error > 2.5x the dataset threshold (severe class)."""
    rows = _load_predictions(r["run_dir"])
    if not rows:
        return None
    sev = 0
    for row in rows:
        if row.get("accepted_for_metrics") != "True":
            continue
        try:
            if float(row["post_fgo_error_m"]) > 2.5 * thr:
                sev += 1
        except (KeyError, ValueError):
            continue
    return sev


def _clean_baseline_row(runs: List[Dict[str, Any]], cell: str, ds: str, system: str,
                        thr: float, attacked_names: Optional[List[str]] = None) -> List[str]:
    """No-injection control over the SAME frames the injection rows report.

    Paired per frame by image_name, not by count: the injection rows drop every
    frame that failed closed, so a control windowed by the attack cell's row
    count still carries those frames and the contrast confounds the attack with
    a different frame population. image_name pairs exactly (800/800 on VPAIR,
    487/487 on AVL).
    """
    match = [r for r in runs if r["cell"] == cell and r["dataset"] == ds]
    if not match:
        return []
    r = match[-1]
    rows = _load_predictions(r["run_dir"])
    if attacked_names:
        names = {x for x in attacked_names if x}
        paired = [x for x in rows if x.get("image_name") in names] if names else []
        # image_name is the pairing key; fall back to the deterministic frame
        # order only if a run predates the column (never silently unpaired --
        # a control left at full length is the very bug this pairing fixes).
        rows = paired if paired else rows[:len(attacked_names)]
    if not rows:
        return []
    acc = [x for x in rows if x.get("accepted_for_metrics") == "True"]
    over = 0
    sev = 0
    for x in acc:
        try:
            e = float(x["post_fgo_error_m"])
        except (KeyError, ValueError):
            continue
        if e > thr:
            over += 1
        if e > 2.5 * thr:
            sev += 1
    n = len(rows)
    lo, hi = wilson_ci(over, n)
    return [
        f"% source (clean control): {r['run_dir']} "
        f"(paired to {len(attacked_names) if attacked_names else n} attacked frames)",
        f"{system} & clean (no injection) & {n} & {len(acc)} ({100*len(acc)/max(1,n):.1f}\\%) & "
        f"{over} ({100*over/max(1,n):.1f}\\%) [{100*lo:.1f}, {100*hi:.1f}] & {sev} & "
        f"{fmt(1.0 - len(acc)/max(1,n), pct=True)} \\\\",
    ]


# Cells that exist only as paired controls for the FORGE tables. They share a
# pipeline name with the primary system, so any emitter that selects rows by
# pipeline prints them as a second, unlabeled copy of that system. That is not
# hypothetical: when gatesoff_vpair_clean landed, four headline tables grew an
# "Ours (UltraVPR-Cascade)" row whose gates were off, and the motivation table
# printed 42 accepted wrong retrievals against a thesis that reports zero.
CONTROL_ONLY_CELLS = {"gatesoff_vpair_clean", "gatesoff_avl_clean",
                      "repeat_forgec_avl_wrong_near"}


def _is_control_only(r: Dict[str, Any]) -> bool:
    return r["cell"] in CONTROL_ONLY_CELLS


# FORGE systems per dataset: (row label, forge-cell prefix, clean-run cell)
FORGE_SYSTEMS = {
    "avl": [
        ("Ours (gated)", "forge_avl", "avl_ours_ultravpr"),
        # Same pipeline, same frames, two gate keys removed; the held-out
        # counterpart of the VPAIR ablation, on the dataset that never entered
        # threshold selection.
        ("Ours, two gates off", "forge_gatesoff_avl", "gatesoff_avl_clean"),
        ("CAMP-var (gated)", "forge_camp_avl", "avl_ours_camp"),
        # Clean controls here are the k=5 runs, matching the shortlist their own
        # injected cells drew the forced reference from. avl_wrapper_k1 is a
        # k=5 run despite its name; see LOC_TABLE_CELLS.
        ("Wrapper (ungated)", "forge_wrapper_avl", "avl_wrapper_k1"),
    ],
    "vpair": [
        ("Ours (gated)", "forge_vpair", "vpair_ours_ultravpr"),
        # Same pipeline as the row above, same frames, two gate keys removed.
        # This is the only controlled gate ablation under injection: every other
        # comparison in this table changes an architecture as well as a gate.
        ("Ours, two gates off", "forge_gatesoff_vpair", "gatesoff_vpair_clean"),
        ("CAMP-var (gated)", "forge_camp_vpair", "vpair_ours_camp"),
        ("Wrapper (ungated)", "forge_wrapper_vpair", "vpair_wrapper_k5"),
    ],
    "uav_visloc": [("Ours (gated)", "forge_visloc", "visloc_ours_ultravpr")],
}

# Retrieval shortlist the forced wrong reference is drawn from, per dataset.
FORGE_SHORTLIST_K = {"avl": 5, "vpair": 5, "uav_visloc": 10}


# FORGE cells were re-run twice as eligibility was corrected. Later generations
# supersede earlier ones under the same (system, condition) label, so tables must
# resolve a prefix rather than one hard-coded cell name, or the re-runs stay
# invisible and the tables keep reporting superseded numbers.
FORGE_CELL_GENERATIONS = ("forgec_", "forgeb_", "forge_")


def _forge_cell(runs: List[Dict[str, Any]], prefix: str, cond: str) -> List[Dict[str, Any]]:
    """Newest-generation run for one (prefix, condition), or [] if none exist."""
    stem = prefix
    for gen in FORGE_CELL_GENERATIONS:
        if stem.startswith(gen):
            stem = stem[len(gen):]
            break
    for gen in FORGE_CELL_GENERATIONS:
        hit = [r for r in runs if r["cell"] == f"{gen}{stem}_{cond}"]
        if hit:
            return hit
    return []


def forge_tables(runs: List[Dict[str, Any]]) -> Dict[str, str]:
    """One FORGE table per dataset; multiple systems where available."""
    out: Dict[str, str] = {}
    for ds in ("avl", "vpair", "uav_visloc"):
        # Unsafe is scored against the dataset's own localization radius, the
        # same bar the main results table uses. RETRIEVAL_TABLE_THRESHOLD_M is
        # the R@1 retrieval radius and is a different quantity: scoring AVL
        # unsafe at 150 m while its success criterion is 20 m would let an
        # accepted 100 m pose count as safe in this table and as a failure in
        # every other one.
        thr = DEFAULT_RADIUS_M.get(ds, 20.0)
        disp = DATASET_DISPLAY.get(ds, ds)
        # Subset size read from an injection cell, never asserted: every frame
        # of the fixed subset was injected, so the cell's row count IS the
        # subset. (No cell was ever run on 300 frames.)
        subset_n = None
        for _system, _prefix, _clean in FORGE_SYSTEMS[ds]:
            for cond in FORGE_CONDITIONS:
                hit = _forge_cell(runs, _prefix, cond)
                if hit:
                    subset_n = len(_load_predictions(hit[-1]["run_dir"])) or None
                    break
            if subset_n:
                break
        subset_txt = (f"every frame of the fixed {subset_n}-frame subset"
                      if subset_n else "every frame of the fixed subset")
        control_note = (" Every clean row here is the same configuration as the "
                        "injected rows beneath it, including the shortlist size, "
                        "so each is a paired control rather than context."
                        if ds in ("vpair", "avl") else "")
        lines = _table_head(
            f"\\textbf{{FORGE wrong-reference injection on {disp}.}} The protocol "
            f"considered {subset_txt}, once per condition, drawing the "
            f"forced reference from the top-{FORGE_SHORTLIST_K.get(ds, 5)} "
            "shortlist. $n$ is the attackable subset for each condition. The "
            "clean row is paired by image name to the proximal condition; "
            "other condition-specific populations are given by their own $n$."
            f"{control_note} "
            "Severe means error above $2.5R$; brackets are Wilson 95\\% "
            "intervals.",
            f"tab:v3-forge-{ds}", "llrrrrr",
            "System & Condition & n & Accepted & Unsafe [CI] & Severe & Rejected \\\\",
            short=f"FORGE injection on {disp}",
        )
        found_any = False
        for system, prefix, clean_cell in FORGE_SYSTEMS[ds]:
            # Pair the control to the attacked frames of this same system.
            attacked_names = None
            for cond in FORGE_CONDITIONS:
                hit = _forge_cell(runs, prefix, cond)
                if hit:
                    attacked_names = [row.get("image_name")
                                      for row in _attacked_rows(hit[-1]["run_dir"])]
                    break
            # clean_cell None means no clean run exists at this system's own
            # configuration. Borrowing another system's clean run would print a
            # row that reads as this system's uninjected behaviour and is not.
            if clean_cell is not None:
                lines += _clean_baseline_row(runs, clean_cell, ds, system, thr,
                                             attacked_names)
            for cond in FORGE_CONDITIONS:
                cell = f"{prefix}_{cond}"
                match = _forge_cell(runs, prefix, cond)
                if not match:
                    lines.append(
                        f"{system} & {COND_DISPLAY.get(cond, cond.replace('_', ' '))} & "
                        f"\\multicolumn{{5}}{{c}}{{{pending(cell)}}} \\\\")
                    continue
                found_any = True
                lines += _forge_row(match[-1], system, cond, thr)
            lines.append("\\midrule")
        if lines[-1] == "\\midrule":
            lines.pop()
        lines += _table_tail()
        out[ds] = "\n".join(lines)
        if not found_any:
            out[ds] = f"% FORGE cells for {ds} not run yet\n" + out[ds]
    return out


# ---------------------------------------------------------------------------
# CAVES: does the risk score separate bad accepted poses? (research question 3)
# ---------------------------------------------------------------------------

def caves_stats(run: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Rank AUROC of caves_risk_score against a bad-pose label, for one run.

    The population is fixed here so nothing downstream can quietly widen it:
    the ACCEPTED frames (accepted_for_metrics == True) of ONE primary clean run
    per dataset. Not pooled across backbones, not the natural-top1-miss subset.
    Bad := post_fgo_error_m beyond that dataset's localization radius, read from
    DEFAULT_RADIUS_M -- the same constant the FORGE emitter scores unsafe with,
    so CAVES and FORGE cannot disagree about what unsafe means.
    """
    if run is None:
        return None
    thr = DEFAULT_RADIUS_M.get(run["dataset"], 20.0)
    frames = []
    for row in _load_predictions(run["run_dir"]):
        if row.get("accepted_for_metrics") != "True":
            continue
        try:
            score = float(row["caves_risk_score"])
            err = float(row["post_fgo_error_m"])
        except (KeyError, ValueError, TypeError):
            continue
        if score != score or err != err:
            continue
        frames.append((score, err > thr))
    n = len(frames)
    if not n:
        return None
    unsafe = sum(1 for _, bad in frames if bad)
    au = _rank_auroc(frames)
    return {
        "n": n,
        "unsafe": unsafe,
        "base_rate": unsafe / n,
        # One class empty means AUROC is undefined, not zero.
        "auroc": au if au == au else None,
        "auroc_ci": _auroc_ci(au, unsafe, n - unsafe) if au == au else None,
        "radius_m": thr,
    }


def _auroc_ci(auroc: float, n_pos: int, n_neg: int, z: float = 1.96) -> Optional[tuple]:
    """Hanley-McNeil 95% interval for an AUROC.

    The UAV-VisLoc row rests on 15 unsafe frames against 1,188 safe ones, so the
    point estimate alone cannot support the above-chance claim the body makes.
    The exponential-distribution variance approximation is the standard closed
    form and needs no resampling pass over the records.
    """
    if n_pos <= 0 or n_neg <= 0:
        return None
    a = auroc
    q1 = a / (2.0 - a)
    q2 = 2.0 * a * a / (1.0 + a)
    var = (a * (1 - a) + (n_pos - 1) * (q1 - a * a)
           + (n_neg - 1) * (q2 - a * a)) / (n_pos * n_neg)
    if var <= 0:
        return None
    half = z * math.sqrt(var)
    return (max(0.0, a - half), min(1.0, a + half))


def caves_table(runs: List[Dict[str, Any]]) -> str:
    """CAVES separating power per dataset, on the same runs the macros cite."""
    by_cell = {(r["cell"], r["dataset"]): r for r in runs}
    lines = _table_head(
        "\\textbf{CAVES risk score against bad accepted poses.} The population is "
        "the \\emph{accepted} frames of one primary clean run per dataset, the "
        "same runs Table~\\ref{tab:v3-loc-merged} and "
        "Table~\\ref{tab:v3-loc-visloc} report, with a frame counted "
        "unsafe when its error exceeds that dataset's acceptance radius, which "
        "Section~\\ref{sec:metrics} states per dataset. It is not pooled across "
        "backbones and not restricted to the natural top-1-miss subset; an "
        "earlier per-dataset operating-point sweep "
        "reads a different campaign over a different population, so it reports "
        "different values for the same quantity name. AUROC is the rank statistic "
        "(Mann--Whitney, ties at half credit), $0.5$ is chance, its bracket is a "
        "Hanley--McNeil 95\\% interval and the base-rate bracket is a Wilson 95\\% "
        "interval. No pooled across-dataset AUROC is reported: the three score "
        "scales differ and the base rates differ by more than twenty times, so a "
        "pooled figure measures that mixture rather than the score, a Simpson's "
        "paradox artifact that inverts the per-dataset conclusion.",
        "tab:v3-caves", "llrrrr",
        "System & Dataset & Accepted n & Unsafe & Base rate [CI] & AUROC [CI] \\\\",
        short="CAVES risk score against bad accepted poses",
    )
    found = False
    for key in MACRO_CELLS.values():
        r = by_cell.get(key)
        st = caves_stats(r)
        if st is None:
            lines.append(f"\\multicolumn{{6}}{{c}}{{{pending(key[0])}}} \\\\")
            continue
        found = True
        lo, hi = wilson_ci(st["unsafe"], st["n"])
        ci = st.get("auroc_ci")
        au = fmt(st["auroc"], ".3f")
        if ci:
            au += f" [{ci[0]:.2f}, {ci[1]:.2f}]"
        lines.append(f"% source: {r['run_dir']} (accepted={st['n']}, R={st['radius_m']:g} m)")
        lines.append(
            f"{_method_label(r)} & "
            f"{DATASET_DISPLAY.get(r['dataset'], r['dataset'])} & {st['n']} & "
            f"{st['unsafe']} & {fmt(st['base_rate'], pct=True)} "
            f"[{100*lo:.1f}, {100*hi:.1f}] & {au} \\\\")
    lines += _table_tail()
    tex = "\n".join(lines)
    return tex if found else "% CAVES runs not available yet\n" + tex


# ---------------------------------------------------------------------------
# Which check rejects an injected wrong reference, and what leaks on clean runs
# ---------------------------------------------------------------------------

# Reject codes in pipeline order: the Phase 3 geometry gates, then the Phase 4
# final-pose gates in the order evaluate_final_pose_acceptance tests them, then
# the frames on which the solver returned nothing at all. Codes are the
# failure_reason vocabulary of benchmark_runner_impl.py.
REJECT_CODE_LABELS = {
    "geometry_inlier_gate_reject": "Inlier count after LGI (Phase 3)",
    "geometry_reproj_gate_reject": "Reprojection residual (Phase 3)",
    "geometry_anchor_offset_reject": "Anchor offset (Phase 3)",
    "final_pose_anchor_offset_reject": "Anchor offset (Phase 4)",
    "final_pose_shape_reject": "Shape confidence (Phase 4)",
    "final_pose_reproj_reject": "Reprojection residual (Phase 4)",
    "final_pose_yaw_prior_reject": "Yaw-prior residual (Phase 4)",
    "geometry_failed": "Solver returned no pose",
}
REJECT_CODE_FALLBACK = "Other"


def _final_pose_gate_reasons(run_dir: str) -> Dict[str, str]:
    """{image_name: which final-pose check fired}.

    predictions.csv carries only the coarse "final_pose_gate_reject", so the
    Phase 4 breakdown comes from results_detailed.json, the one artifact that
    keeps final_pose_gate_reason per frame.
    """
    p = Path(run_dir) / "results_detailed.json"
    if not p.exists():
        return {}
    try:
        j = json.loads(p.read_text())
    except Exception:
        return {}
    out: Dict[str, str] = {}
    for row in (j.get("predictions") or j.get("results") or []):
        name = str(row.get("image_name", "") or "")
        reason = str(row.get("final_pose_gate_reason", "") or "").strip()
        if name and reason:
            out[name] = reason
    return out


def _reject_code(row: Dict[str, str], gate_reasons: Dict[str, str]) -> str:
    code = str(row.get("failure_reason", "") or "").strip()
    if code == "final_pose_gate_reject":
        code = gate_reasons.get(str(row.get("image_name", "") or ""), "") or code
    return code


def reject_code_table(runs: List[Dict[str, Any]]) -> str:
    """Which check rejects an injected wrong reference, one column per condition.

    The population is the genuine wrong-reference trials of the VPAIR injection
    cells, resolved through _forge_cell so this table cannot cite a different
    run from Table~\\ref{tab:v3-forge-vpair}.
    """
    ds = "vpair"
    stem = FORGE_MACRO_CELL_STEM.get(ds, ds)
    counts: Dict[str, Dict[str, int]] = {}
    totals: Dict[str, int] = {}
    sources: List[str] = []
    for cond in FORGE_CONDITIONS:
        match = _forge_cell(runs, f"forge_{stem}", cond)
        if not match:
            continue
        r = match[-1]
        sources.append(f"% source ({COND_DISPLAY.get(cond, cond)}): {r['run_dir']}")
        gate_reasons = _final_pose_gate_reasons(r["run_dir"])
        for row in _fault_rows(r, ds):
            if row.get("accepted_for_metrics") == "True":
                continue
            code = _reject_code(row, gate_reasons)
            counts.setdefault(code, {})[cond] = counts.setdefault(code, {}).get(cond, 0) + 1
            totals[cond] = totals.get(cond, 0) + 1
    grand = sum(totals.values())

    lines = _table_head(
        "\\textbf{Which check rejects an injected wrong reference.} Rejections "
        "of genuine wrong-reference trials on VPAIR, one column per injection "
        "condition. The last column is the share of all rejections in the three "
        "conditions.",
        "tab:v3-reject-codes", "lrrrr",
        "Check that rejected the frame & Prox. & Distal & Vis.\\ sim. & Share \\\\",
        short="Which check rejects an injected wrong reference",
    )
    lines += sources
    ordered = [c for c in REJECT_CODE_LABELS if c in counts]
    ordered += [c for c in counts if c not in REJECT_CODE_LABELS]
    for code in ordered:
        per_cond = counts[code]
        label = REJECT_CODE_LABELS.get(code, f"{REJECT_CODE_FALLBACK} ({tex_escape(code)})")
        cells = " & ".join(str(per_cond.get(cond, 0)) for cond in FORGE_CONDITIONS)
        # An empty grand total is an unread population, not a zero share.
        share = fmt(sum(per_cond.values()) / grand, pct=True) if grand else fmt(None)
        lines.append(f"{label} & {cells} & {share} \\\\")
    if ordered:
        lines.append("\\midrule")
    lines.append("Rejections & "
                 + " & ".join(str(totals.get(cond, 0)) for cond in FORGE_CONDITIONS)
                 + f" & {grand if grand else fmt(None)} \\\\")
    lines += _table_tail()
    tex = "\n".join(lines)
    return tex if sources else f"% FORGE cells for {ds} not run yet\n" + tex


# One row per dataset, resolved through the cell dictionary the CAVES table and
# the macros use, so this table cannot cite a different run from
# Table~\ref{tab:v3-loc-merged}. The wrapper cell is a second VPAIR system and
# would duplicate a dataset row, so only the primary clean cells appear.
FAILURE_CAUSE_MACRO_KEYS = ("avlOurs", "vislocOurs", "vpairOurs")


def failure_cause_stats(run: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Accepted, unsafe, and unsafe split by whether the reference was wrong.

    Unsafe is scored against DEFAULT_RADIUS_M, the same constant the FORGE and
    CAVES emitters use, so the three tables cannot disagree about what unsafe
    means. Wrong is Definition~\\ref{def:wrong-reference}, read through the same
    predicate the stratified FORGE table uses. None means the run carries no
    readable per-frame record; the row then prints the missing marker instead
    of a line of zeros a reader would take for a measurement.
    """
    if run is None:
        return None
    rows = _load_predictions(run["run_dir"])
    if not rows:
        return None
    ds = run["dataset"]
    thr = DEFAULT_RADIUS_M.get(ds, 20.0)
    accepted = unsafe = wrong = 0
    for row in rows:
        if row.get("accepted_for_metrics") != "True":
            continue
        accepted += 1
        try:
            err = float(row["post_fgo_error_m"])
        except (KeyError, ValueError, TypeError):
            continue
        if not err > thr:
            continue
        unsafe += 1
        if _anchor_latlon(row) is None:
            continue          # no reference at all is not evidence of a wrong one
        try:
            if _is_genuine_fault(row, ds):
                wrong += 1
        except (KeyError, ValueError, TypeError):
            continue
    return {"accepted": accepted, "unsafe": unsafe, "wrong": wrong,
            "correct": unsafe - wrong, "radius_m": thr}


def failure_cause_table(runs: List[Dict[str, Any]]) -> str:
    """Clean-run unsafe accepts split by wrong versus correct reference."""
    by_cell = {(r["cell"], r["dataset"]): r for r in runs}
    lines = _table_head(
        "\\textbf{Clean-run unsafe accepts by cause.} Every accepted pose beyond "
        "the acceptance radius $R$ of its dataset, split by whether the reference "
        "it was solved from was wrong (Definition~\\ref{def:wrong-reference}).",
        "tab:v3-failure-cause", "lrrrr",
        "Dataset & Accepted & Unsafe & \\dots wrong reference & "
        "\\dots correct reference \\\\",
        short="Clean-run unsafe accepts by cause",
    )
    found = False
    for key in FAILURE_CAUSE_MACRO_KEYS:
        cell = MACRO_CELLS[key]
        r = by_cell.get(cell)
        st = failure_cause_stats(r)
        if st is None:
            lines.append(f"\\multicolumn{{5}}{{c}}{{{pending(cell[0])}}} \\\\")
            continue
        found = True
        lines.append(f"% source: {r['run_dir']} (R={st['radius_m']:g} m)")
        lines.append(
            f"{DATASET_DISPLAY.get(r['dataset'], r['dataset'])} & "
            f"{fmt(st['accepted'], '.0f')} & {fmt(st['unsafe'], '.0f')} & "
            f"{fmt(st['wrong'], '.0f')} & {fmt(st['correct'], '.0f')} \\\\")
    lines += _table_tail()
    tex = "\n".join(lines)
    return tex if found else "% clean runs not available yet\n" + tex


# Main-table pipelines per dataset (advisor item 10: ablation variants stay in
# the ablation table; the main table carries ours + one variant + baselines).
MAIN_TABLE_PIPELINES = {
    "avl": ["ours_ultravpr_cascade_6dof_e2e", "ours_camp_cascade_6dof_e2e",
            "ours_dsm_pnp_6dof_e2e", "native_anyvisloc_e2e",
            "native_ortholoc_2d3d_e2e", "native_ortholoc_2d3d_oracle_tile"],
    "vpair": ["ours_ultravpr_cascade_6dof_e2e", "ours_camp_cascade_6dof_e2e",
              "ultravpr_roma_homography_wrapper_e2e"],
    "uav_visloc": ["ours_ultravpr_cascade_6dof_e2e"],
}

# Table 4.1 carries the comparison against published systems, and the variants
# table carries our own swaps. Splitting them balances the two blocks of the
# comparison table at four rows each and stops one table from answering two
# questions. The oracle-tile row stays in the comparison table: it is the
# control that shows retrieval is not what limits OrthoLoC, and two passages
# depend on it (ch4 sec:exp-position and the baseline-modification appendix).
COMPARISON_PIPELINES = {
    "avl": ["ours_ultravpr_cascade_6dof_e2e", "native_anyvisloc_e2e",
            "native_ortholoc_2d3d_e2e", "native_ortholoc_2d3d_oracle_tile",
            "ultravpr_roma_homography_wrapper_e2e"],
    "vpair": ["ours_ultravpr_cascade_6dof_e2e",
              "ultravpr_roma_homography_wrapper_e2e"],
    "uav_visloc": ["ours_ultravpr_cascade_6dof_e2e"],
}

# The wrapper exists at two shortlist sizes on both datasets, and the two are
# different systems: at k=1 it takes the rank-one reference, above that it
# selects among candidates. The localization table wants the k=1 run, which is
# the naive baseline the chapter describes; the FORGE tables want the k=5 run,
# because that is the configuration their injected cells were run at. Without
# this map both runs match the pipeline filter and each dataset prints two
# wrapper rows.
#   NOTE: the AVL k=5 run is named avl_wrapper_k1. It was launched with an empty
#   config and inherited the global default of 5, and the directory name records
#   the intent rather than the run. Row labels read the true k from the manifest.
# Both wrapper runs are shown. They are two systems, not two tunings of one:
# at k=1 the wrapper takes the rank-one reference, and above that it selects
# among candidates. The k=5 row is also the one that holds the retrieval front
# end fixed against the proposed cascade, which runs k=5 on both datasets, so it
# is the row the gates-versus-no-gates comparison of this section rests on.
LOC_TABLE_CELLS: Dict[str, Dict[str, str]] = {}

VARIANT_PIPELINES = {
    "avl": ["ours_ultravpr_cascade_6dof_e2e", "ours_camp_cascade_6dof_e2e",
            "ours_dsm_pnp_6dof_e2e"],
    "vpair": ["ours_ultravpr_cascade_6dof_e2e", "ours_camp_cascade_6dof_e2e"],
    "uav_visloc": ["ours_ultravpr_cascade_6dof_e2e"],
}

DATASET_BLOCK_HEAD = {
    "avl": "AVL, 487 frames, radius $R{=}20$\\,m",
    "vpair": "VPAIR, 800 frames, radius $R{=}20$\\,m",
    "uav_visloc": "UAV-VisLoc, 2{,}145 frames, radius $R{=}150$\\,m",
}


# Geometry heads that actually SOLVE altitude (vs. taking it from telemetry).
# The discriminator is the pipeline registry's "geometry" key, never
# manifest.altitude_source: the latter reads "metadata_or_gt_proxy" even on the
# OrthoLoC rows, which solve altitude from a 2D-3D PnP against the DSM.
ALT_SOLVING_GEOMETRIES = {"dsm_pnp", "dsm_p3p", "ortholoc_2d3d_pnp"}


def _pipeline_geometry(run: Dict[str, Any]) -> str:
    cfg = ((run["summary"].get("manifest") or {}).get("pipeline_config") or {})
    return str(cfg.get("geometry", "") or "")


def _method_label(run: Dict[str, Any]) -> str:
    """Row label, with the wrapper's shortlist size read from the run.

    The wrapper takes its reference from the top-$k$ shortlist, so $k$ is part
    of the system and not a tuning detail: at $k{=}1$ it is the naive top-1
    baseline, and above that it selects among candidates. A fixed "top-1" label
    would have printed a $k{=}5$ run under the name of a different system.
    """
    pipeline = run["pipeline"]
    label = METHOD_LABELS.get(pipeline, pipeline)
    if "wrapper" in pipeline and "top-1" in label:
        cfg = ((run["summary"].get("manifest") or {}).get("pipeline_config") or {})
        k = cfg.get("retrieval_top_k")
        if k is not None and int(k) != 1:
            label = label.replace("top-1", f"top-{int(k)}")
    return label


def _alt_is_telemetry_prior(run: Dict[str, Any]) -> bool:
    """True when the Alt. cell is the injected prior, not a solved altitude."""
    geom = _pipeline_geometry(run)
    return bool(geom) and geom not in ALT_SOLVING_GEOMETRIES


def _alt_cell(alt_med, run: Dict[str, Any]) -> str:
    """Altitude cell, marked when the value is the telemetry prior, not a solve."""
    cell = fmt(alt_med)
    if cell != "--" and _alt_is_telemetry_prior(run):
        cell += "$^{\\S}$"
    return cell


def _official_eligible(run: Dict[str, Any]) -> bool:
    """Whether a row is an official comparison row rather than a diagnostic.

    The runner's own flag carries a stale conjunct: it also requires
    retrieval_protocol_filtered_candidates_count == 0. That counter records the
    intra-flight over-fetch filter doing its job, which the protocol requires on
    UAV-VisLoc and which the runner's own comment calls correct enforcement, not
    an integrity failure. Honoring the flag as written would mark the primary
    UAV-VisLoc row as not officially comparable for having obeyed its protocol.
    A genuine violation is counted separately, so that counter is the one that
    decides here.
    """
    s = run["summary"]
    manifest = s.get("manifest") or {}
    v = s.get("official_metric_eligible")
    if v is None:
        v = manifest.get("official_metric_eligible")
    if v is None:
        return True
    if v:
        return True
    violations = manifest.get("retrieval_protocol_violation_count")
    filtered = manifest.get("retrieval_protocol_filtered_candidates_count")
    # Ineligible ONLY because the protocol filter fired: restore eligibility.
    if violations == 0 and (filtered or 0) > 0:
        for key in ("native_baseline", "oracle_mode", "adversarial_diagnostic",
                    "retrieval_only", "deprecated_alias"):
            if manifest.get(key):
                return False
        return True
    return False


# The runtime records a tier as a machine identifier. A reader of the PDF does
# not read the code, so the table prints the phrase and not the key.
TIER_PHRASES = {
    "strict_e2e": "an official end-to-end row",
    "experimental_dsm_6dof": "an experimental elevation-backed head",
    "native_reproduction": "a reproduction of a published system",
    "native_oracle_tile": "a reproduction that is handed its reference",
    "wrapper_baseline": "a published-style wrapper baseline",
    "retrieval_only": "a retrieval-only row that solves no pose",
}


def _comparison_tier(run: Dict[str, Any]) -> str:
    s = run["summary"]
    key = str(s.get("comparison_tier")
              or (s.get("manifest") or {}).get("comparison_tier") or "")
    if not key:
        return ""
    # An unmapped key is shown with its underscores softened rather than raw,
    # so a new tier degrades into readable text instead of leaking an identifier.
    return TIER_PHRASES.get(key, key.replace("_", " "))


def merged_loc_table(runs: List[Dict[str, Any]],
                     native_scripts_dir=None, anyvisloc_log=None,
                     script_rows_fn=None,
                     datasets: Tuple[str, ...] = ("avl", "vpair"),
                     label: str = "tab:v3-loc-merged",
                     caption: Optional[str] = None,
                     short: Optional[str] = None,
                     pipelines: Optional[Dict[str, List[str]]] = None) -> str:
    """Localization comparison, one block per dataset that HAS a comparator.

    UAV-VisLoc is emitted separately. Its block would hold a single row, and a
    one-row block under a shared header invites the cross-block reading the
    header exists to prevent. The row sets also differ per dataset for a reason
    the caption now states: only AVL ships a co-registered elevation model, so
    only AVL can run the elevation-backed head and the two native baselines end
    to end.
    """
    lines = _table_head(
        caption or (
            "\\textbf{Localization on the two datasets that carry a comparator.} "
            "One block per dataset, with $n$ and the acceptance radius $R$ in the "
            "block header. Accept is the share of the block's frames each method "
            "commits to; neither a high nor a low value is better on its own, "
            "because a method that commits to everything cannot withhold a bad "
            "pose, while a method that commits to little can post a flattering "
            "A@10 on the few frames it keeps. The $\\overline{\\mathrm{A@10}}$ "
            "column is the one that combines the two, since it divides the same "
            "hit count by every evaluated frame. The row sets differ by dataset "
            "because the datasets do: "
            "AVL is the only one with a co-registered elevation model, so the two "
            "reproduced baselines run end to end there and nowhere else, and on "
            "VPAIR they appear only in the no-solve form. The variants of the "
            "proposed cascade, including the elevation-backed head, are in "
            "Table~\\ref{tab:v3-loc-variants}. The ungated wrapper is the comparator the injection "
            "experiment of Section~\\ref{sec:forge} needs. Its top-5 row "
            "holds the retrieval front end of the proposed cascade fixed, but "
            "it differs in the matcher and in every stage after retrieval, so "
            "the gap between the two rows bounds a gated-versus-ungated "
            "difference rather than pricing the acceptance layer alone; the "
            "controlled gate comparison is the two-gates-off ablation of "
            "Table~\\ref{tab:v3-forge-vpair}. "
            "UAV-VisLoc has no comparator at all and is reported in "
            "Table~\\ref{tab:v3-loc-visloc}. Rows are never compared across "
            "blocks. $\\S$, $\\ddagger$ and the reproduced rows are explained "
            "in Section~\\ref{sec:exp-baselines}."),
        label, "lrrrrrrrr",
        "Method & Accept & Med. (m) & Mean (m) & A@10m & "
        "$\\overline{\\mathrm{A@10}}$ & Yaw ($^\\circ$) & Alt. (m) & FPS \\\\",
        short=short or "Localization results on the datasets with a comparator",
    )
    first = True
    all_flagged: List[str] = []
    for ds in datasets:
        if not first:
            lines.append("\\midrule")
        first = False
        lines.append(f"\\multicolumn{{9}}{{l}}{{\\emph{{{DATASET_BLOCK_HEAD[ds]}}}}} \\\\")
        rows = [r for r in runs if r["dataset"] == ds
                and not r["summary"].get("retrieval_only")
                and not r["summary"].get("adversarial_diagnostic")
                and r["summary"].get("adversarial_mode", "none") == "none"
                and r["pipeline"] in (pipelines or MAIN_TABLE_PIPELINES)[ds]
                and not _is_control_only(r)
                and (r["pipeline"] not in LOC_TABLE_CELLS.get(ds, {})
                     or r["cell"] == LOC_TABLE_CELLS[ds][r["pipeline"]])]
        # Cell breaks the tie so a pipeline with two runs orders the same way
        # on every regeneration.
        rows.sort(key=lambda r: ((pipelines or MAIN_TABLE_PIPELINES)[ds].index(r["pipeline"]),
                                 r["cell"]))
        labels = [_method_label(r) for r in rows]
        dup = {l for l in labels if labels.count(l) > 1}
        assert not dup, (
            f"{ds}: duplicate row label(s) {dup} -- two runs of one pipeline "
            "reached the merged table; name the second system or exclude the "
            "cell in CONTROL_ONLY_CELLS/LOC_TABLE_CELLS")

        for r in rows:
            met = row_metrics(r)
            # A block header fixes n for every row under it, so a run scored on
            # fewer frames than the block cannot appear here. This is what keeps
            # a 30-frame smoke probe out of the main results table under a label
            # identical to the full-set row; probes live in app:probes.
            if (met["n"] or 0) < BLOCK_N.get(ds, 0):
                continue
            name = _method_label(r)
            if not _official_eligible(r):
                all_flagged.append(
                    f"{name} is {tex_escape(_comparison_tier(r) or 'an unclassified run')}")
                name += "$^{\\ddagger}$"
            alt = _alt_cell(met["alt_med"], r)
            a10_all = None
            if met["a10"] is not None and met["accepted"] and met["n"]:
                a10_all = float(met["a10"]) * float(met["accepted"]) / float(met["n"])
            lines.append(
                f"% source: {r['run_dir']} (n={met['n']}, accepted={met['accepted']}, "
                f"tier={_comparison_tier(r) or '--'}, "
                f"official_metric_eligible={_official_eligible(r)}, "
                f"geometry={_pipeline_geometry(r) or '--'})")
            lines.append(
                f"{name} & {_acc_pct(met['accepted'], met['n'])} & {fmt(met['median_m'])} & "
                f"{fmt(met['mean_m'])} & "
                f"{fmt(met['a10'], pct=True)} & {fmt(a10_all, pct=True)} & "
                f"{fmt(met['yaw_med'])} & {alt} & {fmt(met['fps'], '.2f')} \\\\")
            # No AVL band sub-rows here. The view-angle split is its own table
            # (tab:v3-avl-view-angle); printing the bands in both places made
            # the split read as a repetition rather than as a result.
        if ds == "vpair" and script_rows_fn is not None:
            extra = script_rows_fn()
            if extra:
                lines.extend(extra)
    lines += ["\\bottomrule", "\\end{tabular}"]
    # The tier note sits BELOW the tabular, not as a spanning row inside it. A
    # note inside the body reads as data, and it was also the widest line in
    # the table, which set the column layout for every numeric row above it.
    if all_flagged:
        # A minipage restores justified paragraph mode. Emitted bare, the note
        # inherits the \\centering of the float and sets ragged, which reads as a
        # caption fragment rather than as a legend.
        lines.append(
            "\\par\\vspace{2pt}\\begin{minipage}{\\textwidth}\\footnotesize "
            "$\\ddagger$ These rows are not "
            "official comparison rows. " + "; ".join(all_flagged) +
            ". \\emph{Reprod.} rows run the original authors' own code. "
            "They select their own reference unless marked "
            "\\emph{tile given}, which means the correct reference is supplied, "
            "and \\emph{no solve} means the row reports the supplied "
            "reference's own coordinate instead of solving a pose, so its "
            "position columns measure retrieval quality. The naive wrapper is "
            "not a published system: it is the ungated comparator assembled "
            "for the injection experiment of Section~\\ref{sec:forge}."
            "\\end{minipage}")
    lines.append("\\end{table}")
    return "\n".join(lines)


AVL_VIEW_ANGLE_HEADS = [
    ("avl_ours_ultravpr", "Planar (homography-scale)"),
    ("avl_dsm_pnp", "DSM-PnP"),
    ("avl_native_anyvisloc", "AnyVisLoc (native, DSM-P3P)"),
]
AVL_BAND_LABELS = [("nadir", "nadir"), ("mid", "mid"), ("oblique", "oblique")]


def avl_view_angle_table(runs: List[Dict[str, Any]]) -> str:
    """AVL split by camera view angle, position only, one block per geometry head.

    The baseline block is what lets Section 4.2.3 answer the oblique question
    with a measurement instead of a diagnosis: AnyVisLoc's oblique figures can
    be read against its own nadir figures on the same bands.
    """
    by_cell = {(r["cell"], r["dataset"]): r for r in runs}
    lines = _table_head(
        "\\textbf{AVL by camera view angle.} Position only, one block per "
        "geometry head, on the same frames. Bands are near nadir below "
        "$15^\\circ$, mid between $15^\\circ$ and $40^\\circ$, and oblique at or "
        "above $40^\\circ$.",
        "tab:v3-avl-view-angle", "llrrrrr",
        "Geometry head & Band & n & Accepted & Med.\\ (m) & Mean (m) & A@10m \\\\",
        short="AVL by camera view angle",
    )
    first = True
    for cell, label in AVL_VIEW_ANGLE_HEADS:
        r = by_cell.get((cell, "avl"))
        split = nadir_oblique_split(r) if r is not None else None
        if not split:
            continue
        if not first:
            lines.append("\\midrule")
        first = False
        lines.append(f"% source: {r['run_dir']}")
        for band, blabel in AVL_BAND_LABELS:
            sp = split[band]
            head = label if band == "nadir" else ""
            lines.append(
                f"{head} & {blabel} & {sp['n']} & {sp['accepted']} & "
                f"{fmt(sp['median_m'])} & {fmt(sp['mean_m'])} & "
                f"{fmt(sp['a10'], pct=True)} \\\\")
    lines += _table_tail()
    return "\n".join(lines)


def capability_table(runs: List[Dict[str, Any]]) -> str:
    # Measured capability flags per method family; FPS measured from runs.
    caps = {
        "ours_ultravpr_cascade_6dof_e2e": ("\\cmark", "\\cmark", "\\cmark", "\\cmark"),
        "native_anyvisloc_e2e": ("\\cmark", "\\cmark", "\\cmark", "\\cmark"),
        "native_ortholoc_2d3d_e2e": ("\\cmark", "\\cmark", "\\cmark", "\\xmark"),

    }
    fps_by_pipeline: Dict[str, List[float]] = {}
    for r in runs:
        f = row_metrics(r)["fps"]
        if f:
            fps_by_pipeline.setdefault(r["pipeline"], []).append(float(f))
    lines = _table_head(
        "\\textbf{Which quantities each evaluated method estimates.} The FPS "
        "column is the highest rate among that method's runner-based cells; "
        "script-based no-solve rows are excluded, so it can sit below a "
        "no-solve figure in Table~\\ref{tab:v3-loc-merged}. It is a best case "
        "rather than a per-dataset figure and is not "
        "comparable across rows. Per-dataset rates are in "
        "Table~\\ref{tab:v3-loc-merged}. Where altitude comes from differs by "
        "method and is read in Section~\\ref{sec:exp-baselines}.",
        "tab:v3-capability", "lccccr",
        "Method & lat/lon & yaw & alt & Hit@R & FPS \\\\",
        short="Capability matrix of the evaluated methods",
    )
    for pipeline, (c1, c2, c3, c4) in caps.items():
        fps_vals = fps_by_pipeline.get(pipeline, [])
        fps = f"{max(fps_vals):.2f}" if fps_vals else "--"
        lines.append(f"{METHOD_LABELS.get(pipeline, pipeline)} & {c1} & {c2} & {c3} & {c4} & {fps} \\\\")
    lines += _table_tail()
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Native script CSVs (VPAIR): different schema, no acceptance concept
# ---------------------------------------------------------------------------

SCRIPT_LOGS = {
    # The 400-frame Aug-4 log was superseded by the 800-frame Aug-9 run that
    # produced the shipped CSV; pointing at the old one made the throughput
    # guard suppress the cell rather than report a rate from the wrong run.
    "vpair_ortholoc_results.csv": "vpair_native_ortholoc_script800.log",
    "vpair_yaw_height_results.csv": "vpair_native_anyvisloc_script.rerun2.log",
}

BLOCK_N = {"avl": 487, "vpair": 800, "uav_visloc": 2145}

NATIVE_SCRIPT_SPECS = [
    # (label, filename, error column, yaw-error column or None)
    ("AnyVisLoc (reprod., no solve)", "vpair_yaw_height_results.csv", "pos_err_m", "yaw_err_deg"),
    ("OrthoLoC (reprod., no solve)", "vpair_ortholoc_results.csv", "pos_err_2d_m", "yaw_err_deg"),
]


def anyvisloc_log_row(log_path: Optional[Path]) -> List[str]:
    """Fallback row from the anyvisloc script LOG when its CSV write crashed.

    The script printed complete stage-2 metrics before failing at a late
    import; the log is the artifact of record for this row. Columns without
    printed values stay '--' (never estimated).
    """
    if log_path is None or not Path(log_path).exists():
        return []
    text = Path(log_path).read_text(errors="replace")
    # Split on the metrics-block HEADER, not the earlier "Stage 2 done" progress
    # line (which precedes the Stage-1 summary and made the parser pick up
    # Stage-1 numbers).
    marker = None
    for cand in ("Stage 2 – RoMa re-ranked", "Stage 2 - RoMa re-ranked"):
        if cand in text:
            marker = cand
            break
    if marker is None or "Median pos error" not in text:
        return []
    import re
    stage2 = text.split(marker, 1)[1]
    med = re.search(r"Median pos error\s+([\d.]+)\s*m", stage2)
    mean = re.search(r"Mean\s+pos error\s+([\d.]+)\s*m", stage2)
    roma_s = re.search(r"RoMa/query : mean=([\d.]+)", text)
    camp_ms = re.search(r"CAMP/query : mean=([\d.]+)", text)
    if not (med and mean):
        return []
    fps = None
    if roma_s and camp_ms:
        fps = 1.0 / (float(roma_s.group(1)) + float(camp_ms.group(1)) / 1000.0)
    return [
        f"% source: {log_path} (CSV write crashed post-metrics; log is artifact of record)",
        f"AnyVisLoc (native script) & 100.0\\% & {float(med.group(1)):.1f} & "
        f"{float(mean.group(1)):.1f} & -- & -- & -- & -- & "
        f"{fmt(fps, '.2f')} \\\\",
    ]


def _merge_script_rows(csv_rows: List[str], log_rows: List[str]) -> List[str]:
    """Per-frame CSV wins; the log row is only a stand-in for a missing CSV.

    The AnyVisLoc script crashed before writing its CSV on the first two runs,
    so its row was reconstructed from the printed summary. Once the CSV exists
    it carries per-frame errors (A@N, yaw, altitude) the log never printed, so
    it must take precedence over the log row.
    """
    have_csv = any("AnyVisLoc (native script)" in r
                   and "vpair\\_yaw\\_height\\_results.csv" not in r
                   for r in csv_rows)
    if have_csv:
        return csv_rows
    if log_rows:
        csv_rows = [r for r in csv_rows if "vpair\\_yaw\\_height\\_results.csv" not in r]
    return csv_rows + log_rows


def _script_fps(log_path: Optional[Path], expected_total: Optional[int] = None) -> Optional[float]:
    """Online throughput if the script printed it, else the progress bar."""
    completed_total = _completed_progress_total(log_path)
    if (expected_total is not None and completed_total is not None
            and completed_total != expected_total):
        return None
    return _online_fps_from_log(log_path) or _tqdm_fps_from_log(log_path)


def _completed_progress_total(log_path: Optional[Path]) -> Optional[int]:
    """Final completed tqdm population, when the log exposes one."""
    if log_path is None or not Path(log_path).exists():
        return None
    total = None
    text = Path(log_path).read_text(errors="ignore").replace("\r", "\n")
    for m in re.finditer(r"(\d+)/(\d+) \[", text):
        if int(m.group(1)) == int(m.group(2)):
            total = int(m.group(2))
    return total


def _online_fps_from_log(log_path: Optional[Path]) -> Optional[float]:
    """Throughput from the script's own online timing line.

    Preferred over the progress bar: the bar times one stage, while
    "[ONLINE] TOTAL per query" sums retrieval, matching and geometry, which is
    what the runner-based rows report.
    """
    if log_path is None or not Path(log_path).exists():
        return None
    m = None
    for m in re.finditer(r"\[ONLINE\] TOTAL per query: mean\s+([0-9.]+)\s*ms",
                         Path(log_path).read_text(errors="ignore")):
        pass
    if m is None:
        return None
    ms = float(m.group(1))
    return 1000.0 / ms if ms > 0 else None


def _tqdm_fps_from_log(log_path: Optional[Path]) -> Optional[float]:
    """Throughput of a native script from its own progress bar.

    The sanitized scripts write no metrics_summary, so their FPS is recovered
    from the final tqdm line ("400/400 [06:02<00:00, 1.10it/s]"): frames over
    elapsed wall time, which is the same quantity the runner reports.
    """
    if log_path is None or not Path(log_path).exists():
        return None
    best = None
    text = Path(log_path).read_text(errors="ignore").replace("\r", "\n")
    for m in re.finditer(r"(\d+)/(\d+) \[(\d+):(\d\d)(?::(\d\d))?<", text):
        done, total = int(m.group(1)), int(m.group(2))
        if done != total:
            continue
        a, b, c = m.group(3), m.group(4), m.group(5)
        secs = (int(a) * 3600 + int(b) * 60 + int(c)) if c else (int(a) * 60 + int(b))
        if secs > 0:
            best = total / secs
    return best


def native_script_rows(scripts_dir: Optional[Path],
                       log_dir: Optional[Path] = None,
                       merged: bool = False) -> List[str]:
    """LaTeX rows for the VPAIR loc table from native script CSVs.

    These pipelines have no acceptance layer: every solved frame is an output.
    A@N conditions on the finite solved frames, while Accept is their share of
    the full script population and Abar@10 uses the full population.

    The two consumer tables do NOT share a column set, so the caller must say
    which one it is building. merged=False is loc_table's VPAIR table
    (Acc, Med, Mean, A@5m, A@10m, Abar@10, Yaw, FPS); merged=True is
    merged_loc_table (Acc, Med, Mean, A@10m, Abar@10, Yaw, Alt, FPS). Emitting
    one shape into the other shifts every column right of Mean.
    """
    if scripts_dir is None:
        return []
    out: List[str] = []
    for label, fname, err_col, yaw_col in NATIVE_SCRIPT_SPECS:
        p = Path(scripts_dir) / fname
        if not p.exists():
            out.append(f"{label} & \\multicolumn{{8}}{{c}}{{{pending(fname)}}} \\\\")
            continue
        with open(p) as f:
            rows = list(csv.DictReader(f))
        errs = []
        yaws = []
        for r in rows:
            try:
                e = float(r[err_col])
                if math.isfinite(e):
                    errs.append(e)
            except (KeyError, ValueError):
                pass
            if yaw_col:
                try:
                    y = abs(float(r[yaw_col]))
                    if math.isfinite(y):
                        yaws.append(y)
                except (KeyError, ValueError):
                    pass
        if not errs:
            out.append(f"{label} & \\multicolumn{{8}}{{c}}{{{pending(fname + ' (empty)')}}} \\\\")
            continue
        errs.sort()
        n = len(rows)
        ns = len(errs)
        med = statistics.median(errs)
        mean = sum(errs) / ns
        a5 = sum(1 for e in errs if e <= 5.0) / ns
        a10 = sum(1 for e in errs if e <= 10.0) / ns
        a10_all = sum(1 for e in errs if e <= 10.0) / max(1, n)
        yaws.sort()
        yaw_med = statistics.median(yaws) if yaws else None
        # Altitude from the absolute pred/gt pair. The scripts also ship a
        # height_err_m column, but it differences a relative height against an
        # absolute one and is meaningless (median 318 m against 6.7 m here).
        # Column names differ per script: the OrthoLoC CSV ships absolute
        # pred_alt/gt_alt, the AnyVisLoc CSV ships a ready-made alt_err_m.
        # (Never use OrthoLoC's height_err_m: it differences a relative height
        # against an absolute one, median 318 m against a true 6.7 m.)
        alts = []
        for r in rows:
            v = None
            try:
                v = abs(float(r["pred_alt"]) - float(r["gt_alt"]))
            except (KeyError, ValueError, TypeError):
                try:
                    v = abs(float(r["alt_err_m"]))
                except (KeyError, ValueError, TypeError):
                    v = None
            if v is not None and v == v:
                alts.append(v)
        alts.sort()
        alt_med = statistics.median(alts) if alts else None
        out.append(f"% source: {p}")
        # A row measured on fewer frames than its block is not comparable with
        # its neighbours; make the reduced n visible instead of silently
        # rescaling the denominators.
        tag = "" if n == BLOCK_N.get("vpair") else f"$^{{\\ast}}$"
        fps = fmt(_script_fps(Path(log_dir) / SCRIPT_LOGS[fname], expected_total=n)
                  if log_dir and fname in SCRIPT_LOGS else None, '.2f')
        middle = (f"{100*a10:.1f}\\% & {100*a10_all:.1f}\\% & {fmt(yaw_med)} & {fmt(alt_med)}"
                  if merged else
                  f"{100*a5:.1f}\\% & {100*a10:.1f}\\% & {100*a10_all:.1f}\\% & {fmt(yaw_med)}")
        # These rows are reproductions handed the paired reference by filename,
        # so they are no more an official comparison than the AVL reproductions
        # that already carry the mark. Marking one block and not the other made
        # the weaker rows look like the stronger ones.
        nonofficial = "$^{\\ddagger}$" if merged else ""
        out.append(
            f"{label}{tag}{nonofficial} & {_acc_pct(ns, n)} & {med:.1f} & {mean:.1f} & "
            f"{middle} & {fps} \\\\"
        )
    return out


# ---------------------------------------------------------------------------
# Parameter / configuration tuning tables (B6)
# ---------------------------------------------------------------------------

def k_tuning_table(runs: List[Dict[str, Any]]) -> str:
    """Retrieval budget: R@1/R@5/R@10 by k (retrieval-only) + e2e k=1 vs k=5."""
    lines = _table_head(
        "Retrieval budget tuning. Top: retrieval-only hit rates by shortlist size $k$ "
        "(R@1 is $k$-invariant by construction; R@5/R@10 bound what verification can recover). "
        "Bottom: end-to-end effect of the budget on VPAIR.",
        "tab:v3-k-tuning", "llrrrr",
        "Dataset & Config & k & R@1 & R@min($k$,5) & R@min($k$,10) \\\\",
        short="Retrieval budget tuning by shortlist size",
    )
    sweeps = [r for r in runs if r["summary"].get("retrieval_only")
              and r["pipeline"] != "retrieval_megaloc"]
    sweeps.sort(key=lambda r: (r["dataset"], int(r["summary"].get("retrieval_top_k", 0))))
    for r in sweeps:
        thr = RETRIEVAL_TABLE_THRESHOLD_M.get(r["dataset"], 20)
        diag_path = Path(r["run_dir"]) / "retrieval_diagnostics.json"
        g = {}
        if diag_path.exists():
            try:
                g = json.loads(diag_path.read_text()).get("global", {})
            except Exception:
                g = {}
        r1 = g.get(f"selected_error_m_hit_at_{thr}m")
        r5 = g.get(f"raw_top5_error_m_hit_at_{thr}m")
        r10 = g.get(f"raw_top10_error_m_hit_at_{thr}m")
        lines.append(f"% source: {r['run_dir']}")
        lines.append(
            f"{DATASET_DISPLAY.get(r['dataset'], r['dataset'])} & UltraVPR & "
            f"{r['summary'].get('retrieval_top_k', '--')} & {fmt(r1, pct=True)} & "
            f"{fmt(r5, pct=True)} & {fmt(r10, pct=True)} \\\\"
        )
    lines.append("\\midrule")
    lines.append("\\multicolumn{6}{l}{\\emph{End-to-end effect (VPAIR)}} \\\\")
    lines.append(" &  &  & \\bfseries Med.\\ (m) & \\bfseries Accepted & \\bfseries FPS \\\\")
    for pipe, label in (("ours_fast_top1", "k=1 (fast top-1)"),
                        ("ours_ultravpr_cascade_6dof_e2e", "k=5 (primary)")):
        match = [r for r in runs if r["dataset"] == "vpair" and r["pipeline"] == pipe
                 and not r["summary"].get("retrieval_only")
                 and r["summary"].get("adversarial_mode", "none") == "none"
                 and not _is_control_only(r)]
        if not match:
            lines.append(f"VPAIR & {label} & \\multicolumn{{4}}{{c}}{{{pending(pipe)}}} \\\\")
            continue
        met = row_metrics(match[-1])
        lines.append(f"% source: {match[-1]['run_dir']}")
        lines.append(
            f"VPAIR & {label} & -- & {fmt(met['median_m'])} & "
            f"{_acc_pct(met['accepted'], met['n'])} & {fmt(met['fps'], '.2f')} \\\\"
        )
    lines += _table_tail()
    return "\n".join(lines)


def tile_tuning_table(probe_dirs: List[Path]) -> str:
    """UAV-VisLoc tile footprint tuning from explicit 30-frame probe run dirs."""
    lines = _table_head(
        "\\textbf{UAV-VisLoc tile-footprint probes.} Thirty to thirty-five "
        "frames each, drawn from differing subsets of the UAV-VisLoc flights "
        "and captured at differing code states rather than as a controlled "
        "sweep. Four probes sample flight 01 alone, one samples flight 02 "
        "alone, and two span several flights, so a row-to-row difference here "
        "confounds the setting with the sample. Every probe is listed, and the "
        "spread within one nominal setting is as wide as the spread between "
        "settings.",
        "tab:v3-tile-tuning", "lrrrr",
        "Tile/step (px) & k & Yaw med ($^\\circ$) & Accepted & Med. err (m) \\\\",
        short="UAV-VisLoc tile-footprint probes",
    )
    found = False
    seen = []
    for d in probe_dirs:
        for sp in sorted(Path(d).rglob("metrics_summary.json")):
            try:
                s = json.loads(sp.read_text())
            except Exception:
                continue
            seen.append((int(s.get("uav_tile_size_px") or 0), int(s.get("uav_tile_step_px") or 0),
                         int(s.get("retrieval_top_k") or 0), s, sp))
    if not seen:
        for d in probe_dirs:
            lines.append(f"\\multicolumn{{5}}{{c}}{{{pending(str(d))}}} \\\\")
    for tile, step, k, s, sp in sorted(seen, key=lambda x: (-x[0], x[1], x[2])):
        found = True
        pm = s.get("paper_metrics", {})
        m = s.get("metrics", {})
        lines.append(f"% source: {sp.parent}")
        lines.append(
            f"{tile or '--'}/{step or '--'} & {k or '--'} & {fmt(pm.get('yaw_error_deg_median'))} & "
            f"{s.get('num_accepted_for_metrics', '--')}/{s.get('num_frames', '--')} & "
            f"{fmt(m.get('median_error_m'))} \\\\"
        )
    lines += _table_tail()
    if not found:
        lines.insert(0, "% tile probes not found")
    return "\n".join(lines)


def gate_tuning_table(runs: List[Dict[str, Any]], attempt_root: Optional[Path],
                      which: str = "gates") -> str:
    """Paired-run configuration tables.

    Two claims, so two tables. `which="gates"` emits the VPAIR acceptance-gate
    pair that Section 4.3.4 reads. `which="fusion"` emits the UAV-VisLoc fusion
    motion-prior pair, which is a configuration fact and lives in the appendix.
    """
    if which == "fusion":
        lines = _table_head(
            "\\textbf{Fusion motion-prior setting on UAV-VisLoc.} The two runs "
            "share a subset and differ in the two prior settings named below, "
            "but they were captured eight days apart, and the anisotropic "
            "ground-sampling correction for this dataset landed between them. "
            "The pair therefore prices the setting together with that "
            "correction, and neither alone. It is recorded as a configuration "
            "fact and carries no claim in Chapter~\\ref{ch:exp}. A run "
            "at each setting, on the same subset. $\\sigma_{pos}$ is scaled to "
            "the frame spacing of the three flights. "
            "FGO $\\Delta$ (m) is the mean change in position error across the "
            "fusion step, over accepted frames; negative means the fusion step "
            "reduced error.",
            "tab:v3-fusion-prior", "llrrrr",
            "Dataset & Configuration & Mean (m) & Max acc.\\ err (m) & FGO $\\Delta$ (m) & Accepted \\\\",
            short="Fusion motion-prior setting on UAV-VisLoc",
        )
        pairs = [("uav_visloc", "visloc_ours_ultravpr", "$\\sigma_{pos}$ 8\\,m",
                  "$\\sigma_{pos}$ 80\\,m + step guard")]
    else:
        lines = _table_head(
            "\\textbf{What the two novel acceptance gates buy on uninjected "
            "VPAIR traffic.} Two runs on the same subset and the same pipeline "
            "configuration apart from the gate settings: yaw-prior residual "
            "$30^\\circ$ and minimum inliers 25. The two runs come from "
            "different campaign trees, captured about seven hours apart. "
            "FGO $\\Delta$ (m) is the mean change in position error across the "
            "fusion step, over accepted frames; negative means the fusion step "
            "reduced error.",
            "tab:v3-gate-tuning", "llrrrr",
            "Dataset & Configuration & Mean (m) & Max acc.\\ err (m) & FGO $\\Delta$ (m) & Accepted \\\\",
            short="What the acceptance gates buy on uninjected traffic",
        )
        pairs = [("vpair", "vpair_ours_ultravpr", "gates off", "gates on")]
    for ds, cell, before_label, after_label in pairs:
        rows = []
        if attempt_root is not None:
            before = load_runs(Path(attempt_root))
            rows += [(r, before_label) for r in before if r["cell"] == cell]
        rows += [(r, after_label) for r in runs if r["cell"] == cell
                 and not r["summary"].get("retrieval_only")]
        if len(rows) != 2:
            raise RuntimeError(f"{which} tuning pair requires exactly two runs; got {len(rows)}")
        before_run, after_run = rows[0][0], rows[1][0]
        before_cfg = before_run.get("manifest", {}).get("pipeline_config", {}) or {}
        after_cfg = after_run.get("manifest", {}).get("pipeline_config", {}) or {}
        changed = {k for k in set(before_cfg) | set(after_cfg)
                   if before_cfg.get(k) != after_cfg.get(k)}
        expected = ({"final_pose_max_yaw_prior_residual_deg",
                     "geometry_min_inliers_after_lgi"}
                    if which == "gates" else
                    {"fgo_motion_prior_sigma_pos_m", "fgo_max_step_xy_m"})
        if changed != expected:
            raise RuntimeError(
                f"{which} tuning pair changed config keys {sorted(changed)}; "
                f"expected exactly {sorted(expected)}"
            )
        before_names = sorted(r.get("image_name", "")
                              for r in _load_predictions(before_run["run_dir"]))
        after_names = sorted(r.get("image_name", "")
                             for r in _load_predictions(after_run["run_dir"]))
        if before_names != after_names:
            raise RuntimeError(f"{which} tuning pair does not use the same frame multiset")
        for r, label in rows:
            met = row_metrics(r)
            preds = _load_predictions(r["run_dir"])
            mx = None
            deltas = []
            for p in preds:
                if p.get("accepted_for_metrics") != "True":
                    continue
                try:
                    e = float(p["post_fgo_error_m"])
                    mx = e if mx is None else max(mx, e)
                except (KeyError, ValueError):
                    pass
                try:
                    deltas.append(float(p["post_fgo_error_m"]) - float(p["pre_fgo_error_m"]))
                except (KeyError, ValueError):
                    pass
            delta = (sum(deltas) / len(deltas)) if deltas else None
            lines.append(f"% source: {r['run_dir']}")
            lines.append(
                f"{DATASET_DISPLAY.get(ds, ds)} & {label} & {fmt(met['mean_m'])} & "
                f"{fmt(mx, '.0f')} & {fmt(delta, '+.2f')} & {_acc_pct(met['accepted'], met['n'])} \\\\"
            )
    lines += _table_tail()
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Macros for prose numbers
# ---------------------------------------------------------------------------

MACRO_CELLS = {
    "vpairOurs": ("vpair_ours_ultravpr", "vpair"),
    "avlOurs": ("avl_ours_ultravpr", "avl"),
    "vislocOurs": ("visloc_ours_ultravpr", "uav_visloc"),
    "vpairWrapper": ("vpair_wrapper_k1", "vpair"),
    # The k=1 run, matching vpairWrapper. Prose that names the naive
    # top-1 wrapper must not resolve to the k=5 run of the same pipeline.
    "avlWrapper": ("avl_wrapper_true_k1", "avl"),
    # The k=5 wrapper. This is the front-end-matched comparator: the proposed
    # cascade runs k=5 on both datasets, so only this row differs from it in
    # whether a pose may be declined. Section 4.2 rests on these two.
    "avlWrapperKFive": ("avl_wrapper_k1", "avl"),
    "vpairWrapperKFive": ("vpair_wrapper_k5", "vpair"),
}
MACRO_FIELDS = {
    "Median": ("median_m", ".1f", False),
    "Mean": ("mean_m", ".1f", False),
    "Trimmed": ("trimmed_m", ".1f", False),
    "AccN": ("accepted", ".0f", False),
    "N": ("n", ".0f", False),
    "AatFive": ("a5", "", True),
    "AatTen": ("a10", "", True),
    "YawMed": ("yaw_med", ".2f", False),
    "AltMed": ("alt_med", ".1f", False),
    "Fps": ("fps", ".2f", False),
}

# AVL view-angle bands exist in the tables only as SUB-rows, which no macro
# covered, so prose citing one (the DSM-PnP head's accepted-oblique median, and
# the cascade's, whose ratio is the like-for-like oblique improvement) typed the
# value by hand. The ratio itself is deliberately not computed here: prose
# states it from the two medians.
AVL_SPLIT_MACRO_CELLS = {
    "avlOurs": ("avl_ours_ultravpr", "avl"),
    "avlDsmPnp": ("avl_dsm_pnp", "avl"),
    # The baseline that wins on AVL is split by band too, so the oblique
    # regime comparison is backed by a number rather than an assertion.
    "avlAnyvisloc": ("avl_native_anyvisloc", "avl"),
}
AVL_SPLIT_BANDS = {"nadir": "Nadir", "mid": "Mid", "oblique": "Oblique"}
AVL_SPLIT_FIELDS = {
    "Median": ("median_m", ".1f", False),
    "Mean": ("mean_m", ".1f", False),
    "AccN": ("accepted", ".0f", False),
    "N": ("n", ".0f", False),
    "AatTen": ("a10", "", True),
    "YawMed": ("yaw_med", ".2f", False),
}


FORGE_MACRO_NAMES = {
    ("avl", "wrong_near"): "forgeAvlNear",
    ("avl", "wrong_far"): "forgeAvlFar",
    ("avl", "visually_similar_wrong"): "forgeAvlSim",
    ("vpair", "wrong_near"): "forgeVpairNear",
    ("vpair", "wrong_far"): "forgeVpairFar",
    ("vpair", "visually_similar_wrong"): "forgeVpairSim",
    ("uav_visloc", "wrong_near"): "forgeVislocNear",
    ("uav_visloc", "wrong_far"): "forgeVislocFar",
    ("uav_visloc", "visually_similar_wrong"): "forgeVislocSim",
    ("camp_vpair", "wrong_near"): "forgeCampNear",
    ("camp_vpair", "wrong_far"): "forgeCampFar",
    ("camp_vpair", "visually_similar_wrong"): "forgeCampSim",
    ("wrapper_vpair", "wrong_near"): "forgeWrapNear",
    ("wrapper_vpair", "wrong_far"): "forgeWrapFar",
    ("wrapper_vpair", "visually_similar_wrong"): "forgeWrapSim",
    ("wrapper_avl", "wrong_near"): "forgeWrapAvlNear",
    ("wrapper_avl", "wrong_far"): "forgeWrapAvlFar",
    ("wrapper_avl", "visually_similar_wrong"): "forgeWrapAvlSim",
    ("camp_avl", "wrong_near"): "forgeCampAvlNear",
    ("camp_avl", "wrong_far"): "forgeCampAvlFar",
    ("camp_avl", "visually_similar_wrong"): "forgeCampAvlSim",
    ("gatesoff_vpair", "wrong_near"): "forgeGatesOffNear",
    ("gatesoff_vpair", "wrong_far"): "forgeGatesOffFar",
    ("gatesoff_vpair", "visually_similar_wrong"): "forgeGatesOffSim",
    ("gatesoff_avl", "wrong_near"): "forgeGatesOffAvlNear",
    ("gatesoff_avl", "wrong_far"): "forgeGatesOffAvlFar",
    ("gatesoff_avl", "visually_similar_wrong"): "forgeGatesOffAvlSim",
}

# Cell-name stem per dataset (run dirs predate the uav_visloc dataset key).
FORGE_MACRO_CELL_STEM = {"avl": "avl", "vpair": "vpair", "uav_visloc": "visloc",
                         "camp_vpair": "camp_vpair", "wrapper_vpair": "wrapper_vpair",
                         "camp_avl": "camp_avl", "wrapper_avl": "wrapper_avl",
                         "gatesoff_vpair": "gatesoff_vpair",
                         "gatesoff_avl": "gatesoff_avl"}


def _accepted_errors(run: Optional[Dict[str, Any]],
                     band: Optional[str] = None) -> Dict[str, float]:
    """{image_name: position error} over the accepted frames of one run.

    Two systems evaluated on one subset accept different frames, so a
    conditional statistic compared across them is not a like-for-like
    comparison. Every macro built on this map is restricted to a frame set both
    systems accept, which is the comparison Section 4.1.3 permits.
    """
    if run is None:
        return {}
    out: Dict[str, float] = {}
    for row in _load_predictions(run["run_dir"]):
        if row.get("accepted_for_metrics") != "True":
            continue
        if band is not None:
            try:
                from_nadir = abs(90.0 - abs(float(row["gt_pitch"])))
            except (KeyError, ValueError, TypeError):
                continue
            key = "nadir" if from_nadir < 15.0 else (
                "oblique" if from_nadir >= 40.0 else "mid")
            if key != band:
                continue
        e = _frame_error_m(row)
        name = row.get("image_name")
        if e is not None and name:
            out[name] = e
    return out


# An accepted pose this far past its 20 m radius is not a marginal miss; it is
# the lock-on case Section 4.3.5 names. Used only to count those frames.
LOCK_ON_ERROR_M = 1000.0


# ---------------------------------------------------------------------------
# Chapter 3 methodology macros (margin guard, height-from-scale, fault rules)
# ---------------------------------------------------------------------------

def margin_guard_stats(csv_path: Optional[Path],
                       threshold_m: float = 20.0) -> Dict[str, Any]:
    """Displacement counts on a margin-0 (guard-off) verification run.

    candidate_rank_selected > 0 means the verifier chose a lower-ranked
    candidate over the descriptor top-1. A displacement is correct->wrong
    (raw anchor was within threshold, selected anchor is not); a repair is
    the reverse.
    """
    if csv_path is None or not Path(csv_path).exists():
        return {}
    displaced = repaired = n = 0
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            n += 1
            try:
                rank = int(row.get("candidate_rank_selected", "0") or "0")
            except (ValueError, TypeError):
                continue
            if rank == 0:
                continue
            try:
                sel_err = float(row["retrieval_anchor_error_m"])
                raw_err = float(row["raw_retrieval_anchor_error_m"])
            except (KeyError, ValueError, TypeError):
                continue
            if raw_err <= threshold_m and sel_err > threshold_m:
                displaced += 1
            elif raw_err > threshold_m and sel_err <= threshold_m:
                repaired += 1
    return {"displaced": displaced, "repaired": repaired, "n": n}


def height_scale_correlation(csv_path: Optional[Path]) -> Dict[str, Any]:
    """Pearson r(visual_pred_alt, gt_alt) over accepted frames.

    The planar geometry head can in principle recover altitude from the
    homography scale, but the estimate (visual_pred_alt) has near-zero
    correlation with ground truth on VPAIR.
    """
    if csv_path is None or not Path(csv_path).exists():
        return {}
    pairs: List[tuple] = []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            if row.get("accepted_for_metrics") != "True":
                continue
            try:
                va = float(row["visual_pred_alt"])
                ga = float(row["gt_alt"])
            except (KeyError, ValueError, TypeError):
                continue
            if va != va or ga != ga:
                continue
            pairs.append((va, ga))
    n = len(pairs)
    if n < 2:
        return {"correlation": None, "n": n}
    va_list = [p[0] for p in pairs]
    ga_list = [p[1] for p in pairs]
    mean_va = sum(va_list) / n
    mean_ga = sum(ga_list) / n
    cov = sum((v - mean_va) * (g - mean_ga) for v, g in pairs) / (n - 1)
    std_va = (sum((v - mean_va) ** 2 for v in va_list) / (n - 1)) ** 0.5
    std_ga = (sum((g - mean_ga) ** 2 for g in ga_list) / (n - 1)) ** 0.5
    if std_va == 0 or std_ga == 0:
        return {"correlation": 0.0, "n": n}
    return {"correlation": cov / (std_va * std_ga), "n": n}


def _center_containment_can_localize(gt_lat: float, gt_lon: float,
                                     ref_lat: float, ref_lon: float,
                                     ds: str, row: Dict[str, str]) -> bool:
    """Query center inside reference footprint (no query extent, no margin)."""
    de, dn = _per_axis_m(gt_lat, gt_lon, ref_lat, ref_lon)
    rh_e, rh_n = REF_HALF_M.get(ds, (TILE_SLACK_M, TILE_SLACK_M))
    return de <= rh_e and dn <= rh_n


FIXED_WRONG_REF_RADIUS_M = 350.0


def _fixed_radius_can_localize(gt_lat: float, gt_lon: float,
                                ref_lat: float, ref_lon: float,
                                ds: str, row: Dict[str, str]) -> bool:
    """Single fixed radius across all datasets."""
    de, dn = _per_axis_m(gt_lat, gt_lon, ref_lat, ref_lon)
    return math.hypot(de, dn) <= FIXED_WRONG_REF_RADIUS_M


def _center_distance_can_localize(gt_lat: float, gt_lon: float,
                                  ref_lat: float, ref_lon: float,
                                  ds: str, row: Dict[str, str]) -> bool:
    """Scalar center-to-center distance against the diagonal of the combined
    footprint. Collapses the per-axis bounds into a single circle."""
    de, dn = _per_axis_m(gt_lat, gt_lon, ref_lat, ref_lon)
    rh_e, rh_n = REF_HALF_M.get(ds, (TILE_SLACK_M, TILE_SLACK_M))
    qh_e, qh_n = query_half_m(ds, row)
    threshold = math.hypot(rh_e + qh_e, rh_n + qh_n) + TILE_SLACK_M
    return math.hypot(de, dn) <= threshold


_FAULT_ALT_RULES = {
    "center_distance": _center_distance_can_localize,
    "center_containment": _center_containment_can_localize,
    "fixed_radius": _fixed_radius_can_localize,
}


def fault_rule_contradictions(
    runs: List[Dict[str, Any]],
) -> Dict[str, int]:
    """Count frames each alternative fault rule misclassifies on FORGE data.

    A contradiction: rule says 'can localize' (reference is OK) for a forced
    reference that is genuinely wrong under the per-axis predicate AND the
    system actually fails on it (rejected or error > R); or rule says 'cannot
    localize' for a forced reference the system localizes from correctly
    (accepted with error <= R). Returns {rule_name: contradiction_count}.
    """
    counts: Dict[str, int] = {k: 0 for k in _FAULT_ALT_RULES}
    for ds in ("avl", "vpair", "uav_visloc"):
        thr = DEFAULT_RADIUS_M.get(ds, 20.0)
        stem = FORGE_MACRO_CELL_STEM.get(ds, ds)
        for cond in FORGE_CONDITIONS:
            match = _forge_cell(runs, f"forge_{stem}", cond)
            if not match:
                continue
            for row in _attacked_rows(match[-1]["run_dir"]):
                ref = _anchor_latlon(row)
                if ref is None:
                    continue
                try:
                    gt_lat, gt_lon = float(row["gt_lat"]), float(row["gt_lon"])
                except (KeyError, ValueError, TypeError):
                    continue
                ref_lat, ref_lon = ref
                accepted = row.get("accepted_for_metrics") == "True"
                try:
                    err = float(row["post_fgo_error_m"])
                except (KeyError, ValueError, TypeError):
                    err = float("inf")
                correct = accepted and err <= thr
                for name, fn in _FAULT_ALT_RULES.items():
                    can_loc = fn(gt_lat, gt_lon, ref_lat, ref_lon, ds, row)
                    if (not can_loc and correct) or (can_loc and not correct):
                        counts[name] += 1
    return counts


def fault_refutation_stats(runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Run the falsification test of Section 3.9 and count what refutes it.

    The test asks one question of every injected trial: did a frame whose
    forced reference the predicate classes as unrecoverable, by more than the
    declared tolerance, still get accepted with an error inside its dataset's
    acceptance radius? Every such trial is a refutation.

    This exists because fault_borderline_trial answers a different question and
    was read as if it answered this one. That helper keeps only trials whose
    overshoot is BELOW the tolerance, which is exactly the set of trials that do
    not refute the predicate, so it reports the near miss and discards every
    genuine contradiction. A claim about the test therefore has to be counted
    here, and it is emitted as a macro so no chapter can restate it by hand.
    """
    out: Dict[str, Any] = {"tolerance_m": TILE_SLACK_M}
    for ds in ("avl", "vpair", "uav_visloc"):
        thr = DEFAULT_RADIUS_M.get(ds, 20.0)
        stem = FORGE_MACRO_CELL_STEM.get(ds, ds)
        refute = 0
        within = 0
        frames: set = set()
        overshoots: List[float] = []
        trials = 0
        for cond in FORGE_CONDITIONS:
            match = _forge_cell(runs, f"forge_{stem}", cond)
            if not match:
                continue
            for row in _attacked_rows(match[-1]["run_dir"]):
                trials += 1
                ref = _anchor_latlon(row)
                if ref is None:
                    continue
                try:
                    gt_lat, gt_lon = float(row["gt_lat"]), float(row["gt_lon"])
                except (KeyError, ValueError, TypeError):
                    continue
                if row.get("accepted_for_metrics") != "True":
                    continue
                try:
                    err = float(row["post_fgo_error_m"])
                except (KeyError, ValueError, TypeError):
                    continue
                if err > thr:
                    continue
                de, dn = _per_axis_m(gt_lat, gt_lon, ref[0], ref[1])
                rh_e, rh_n = REF_HALF_M.get(ds, (TILE_SLACK_M, TILE_SLACK_M))
                qh_e, qh_n = query_half_m(ds, row)
                overshoot = max(de - (rh_e + qh_e), dn - (rh_n + qh_n))
                if overshoot <= 0:
                    continue
                if overshoot < TILE_SLACK_M:
                    within += 1
                    continue
                refute += 1
                overshoots.append(overshoot)
                name = str(row.get("image_name", "") or "")
                if name:
                    frames.add(name)
        out[ds] = {
            "refutations": refute,
            "within_tolerance": within,
            "frames": len(frames),
            "trials": trials,
            "overshoot_min": min(overshoots) if overshoots else None,
            "overshoot_max": max(overshoots) if overshoots else None,
        }
    return out


def fault_borderline_trial(
    runs: List[Dict[str, Any]],
) -> Optional[Dict[str, float]]:
    """Find the trial closest to violating the per-axis predicate.

    Returns the smallest per-axis overshoot (distance minus bound) and the
    tolerance width (TILE_SLACK_M) for the frame that overshoots the per-axis
    bound by less than the tolerance while localizing inside the radius.
    None if no FORGE data is available. This is the NEAR MISS, not the test:
    fault_refutation_stats is the test.
    """
    best: Optional[Dict[str, float]] = None
    for ds in ("avl", "vpair", "uav_visloc"):
        thr = DEFAULT_RADIUS_M.get(ds, 20.0)
        stem = FORGE_MACRO_CELL_STEM.get(ds, ds)
        for cond in FORGE_CONDITIONS:
            match = _forge_cell(runs, f"forge_{stem}", cond)
            if not match:
                continue
            for row in _attacked_rows(match[-1]["run_dir"]):
                ref = _anchor_latlon(row)
                if ref is None:
                    continue
                try:
                    gt_lat, gt_lon = float(row["gt_lat"]), float(row["gt_lon"])
                except (KeyError, ValueError, TypeError):
                    continue
                ref_lat, ref_lon = ref
                accepted = row.get("accepted_for_metrics") == "True"
                try:
                    err = float(row["post_fgo_error_m"])
                except (KeyError, ValueError, TypeError):
                    err = float("inf")
                if not (accepted and err <= thr):
                    continue
                de, dn = _per_axis_m(gt_lat, gt_lon, ref_lat, ref_lon)
                rh_e, rh_n = REF_HALF_M.get(ds, (TILE_SLACK_M, TILE_SLACK_M))
                qh_e, qh_n = query_half_m(ds, row)
                bound_e = rh_e + qh_e + TILE_SLACK_M
                bound_n = rh_n + qh_n + TILE_SLACK_M
                overshoot_e = de - (rh_e + qh_e)
                overshoot_n = dn - (rh_n + qh_n)
                overshoot = max(overshoot_e, overshoot_n)
                if overshoot <= 0:
                    continue
                if overshoot < TILE_SLACK_M and (best is None or overshoot < best["overshoot"]):
                    best = {"overshoot": overshoot, "tolerance": TILE_SLACK_M,
                            "dataset": ds, "error_m": err}
    return best


def _paired_macros(by_cell: Dict[tuple, Dict[str, Any]]) -> List[str]:
    """Macros that only mean anything on a shared accepted set."""
    lines: List[str] = []

    def emit(name: str, val) -> None:
        lines.append(f"\\newcommand{{\\{name}}}{{{val if val is not None else '--'}}}")

    ours = _accepted_errors(by_cell.get(("vpair_ours_ultravpr", "vpair")))
    wrap = _accepted_errors(by_cell.get(("vpair_wrapper_k1", "vpair")))
    shared = sorted(set(ours) & set(wrap))
    wrap_only = sorted(set(wrap) - set(ours))
    emit("vpairSharedAccN", len(shared) or None)
    emit("vpairWrapperOnlyN", len(wrap_only) or None)
    if shared:
        emit("vpairOursSharedMean", fmt(sum(ours[k] for k in shared) / len(shared)))
        emit("vpairWrapperSharedMean", fmt(sum(wrap[k] for k in shared) / len(shared)))
    else:
        emit("vpairOursSharedMean", None)
        emit("vpairWrapperSharedMean", None)
    emit("vpairWrapperOnlyMean",
         fmt(sum(wrap[k] for k in wrap_only) / len(wrap_only)) if wrap_only else None)

    # AVL oblique band, the two geometry heads on the frames both accept.
    planar = _accepted_errors(by_cell.get(("avl_ours_ultravpr", "avl")), band="oblique")
    dsm = _accepted_errors(by_cell.get(("avl_dsm_pnp", "avl")), band="oblique")
    both = sorted(set(planar) & set(dsm))
    emit("avlObliqueBothN", len(both) or None)
    emit("avlOursObliqueBothMedian",
         fmt(statistics.median([planar[k] for k in both])) if both else None)
    emit("avlDsmPnpObliqueBothMedian",
         fmt(statistics.median([dsm[k] for k in both])) if both else None)

    camp = _accepted_errors(by_cell.get(("vpair_ours_camp", "vpair")))
    emit("campVpairLockOnN",
         sum(1 for e in camp.values() if e > LOCK_ON_ERROR_M) if camp else None)
    return lines


def macros_tex(runs: List[Dict[str, Any]]) -> str:
    by_cell = {(r["cell"], r["dataset"]): r for r in runs}
    lines = ["% Auto-generated by gen_v3_tables.py - DO NOT EDIT NUMBERS BY HAND"]
    for prefix, key in MACRO_CELLS.items():
        r = by_cell.get(key)
        for fname, (field, spec, pct) in MACRO_FIELDS.items():
            if r is None:
                val = pending(key[0])
            else:
                met = row_metrics(r)
                val = fmt(met[field], spec or ".1f", pct=pct)
            lines.append(f"\\newcommand{{\\{prefix}{fname}}}{{{val}}}")
        # The unconditional hit rate is a table column with no macro behind it,
        # so prose comparing two methods across their different acceptance rates
        # had to type it. Same definition as the table: conditional rate scaled
        # by the share of frames the method committed to.
        a10_all = None
        if r is not None:
            met = row_metrics(r)
            if met["a10"] is not None and met["accepted"] and met["n"]:
                a10_all = float(met["a10"]) * float(met["accepted"]) / float(met["n"])
        lines.append(f"\\newcommand{{\\{prefix}AatTenAll}}{{"
                     f"{fmt(a10_all, pct=True) if a10_all is not None else pending(key[0])}}}")
        # The acceptance percentage and the worst accepted error are slide-level
        # quantities with no macro of their own, so the deck typed them by hand
        # and one of them went stale within a day. Emit both.
        if r is not None:
            met = row_metrics(r)
            if met["accepted"] and met["n"]:
                pct = 100.0 * float(met["accepted"]) / float(met["n"])
                lines.append(f"\\newcommand{{\\{prefix}AcceptPct}}{{{pct:.1f}\\%}}")
            errs = list(_accepted_errors(r).values())
            if errs:
                w = max(errs)
                wtxt = f"{w:,.0f}".replace(",", "{,}")
                lines.append(f"\\newcommand{{\\{prefix}WorstAcc}}{{{wtxt}}}")
            lines.append(f"% source {prefix}: {r['run_dir']}")
        else:
            lines.append(f"\\newcommand{{\\{prefix}AcceptPct}}{{{pending(key[0])}}}")
            lines.append(f"\\newcommand{{\\{prefix}WorstAcc}}{{{pending(key[0])}}}")
    # AVL view-angle bands. A band with no accepted frame yields the same
    # missing marker every other empty cell in this file uses -- never 0.0 --
    # while its AccN/N stay real counts, so "no accepted frames" is citable.
    for prefix, key in AVL_SPLIT_MACRO_CELLS.items():
        r = by_cell.get(key)
        split = nadir_oblique_split(r) if r is not None else None
        for band, bname in AVL_SPLIT_BANDS.items():
            s = (split or {}).get(band)
            for fname, (field, spec, pct) in AVL_SPLIT_FIELDS.items():
                val = pending(key[0]) if s is None else fmt(s[field], spec or ".1f", pct=pct)
                lines.append(f"\\newcommand{{\\{prefix}{bname}{fname}}}{{{val}}}")
        if split is not None:
            lines.append(f"% source {prefix} view-angle bands: {r['run_dir']}")
    # CAVES macros: AUROC + base rate of bad accepted poses, same four runs and
    # same population as caves_table (see its caption).
    for prefix, key in MACRO_CELLS.items():
        r = by_cell.get(key)
        st = caves_stats(r) if r is not None else None
        if st is None or not st["n"]:
            au = base = pending(key[0])
        else:
            au = fmt(st["auroc"], ".3f")
            base = fmt(st["base_rate"], ".3f")
            lines.append(f"% source {prefix} CAVES: {r['run_dir']}")
        lines.append(f"\\newcommand{{\\{prefix}CavesAuroc}}{{{au}}}")
        lines.append(f"\\newcommand{{\\{prefix}CavesBaseRate}}{{{base}}}")
        # The count behind the base rate. Section 4.2 states the share of
        # accepted poses that are unsafe, and a rate alone hides the n.
        lines.append(f"\\newcommand{{\\{prefix}UnsafeN}}"
                     f"{{{pending(key[0]) if st is None else st['unsafe']}}}")
    lines += _paired_macros(by_cell)
    # FORGE macros: unsafe-under-attack rate + severity median per cell
    for (ds, cond), name in FORGE_MACRO_NAMES.items():
        stem = FORGE_MACRO_CELL_STEM.get(ds, ds)
        cell = f"forge_{stem}_{cond}"
        match = _forge_cell(runs, f"forge_{stem}", cond)
        if not match:
            u = a = sev = pending(cell)
        else:
            s = match[-1]["summary"]
            # Same denominator as _forge_row: only frames the injector could
            # actually attack. Reading it off summary["num_frames"] would print
            # a macro in the prose that disagrees with the table row directly
            # above it, for the very same run.
            # Under the oblique AnyVisLoc footprint, genuine faults only, the
            # population the table rows count. The nadir labels keep the record
            # definition (all attacked trials) so the cells of record replay.
            # The dataset the cell ran on: keys such as "gatesoff_avl" or
            # "wrapper_avl" name a configuration, not a dataset.
            ds_lab = "vpair" if ds.endswith("vpair") else ("avl" if ds.endswith("avl") else ds)
            att = [row for row in _attacked_rows(match[-1]["run_dir"])
                   if AVL_FOOTPRINT != "oblique" or ds_lab != "avl" or _is_genuine_fault(row, ds_lab)]
            if att:
                n = len(att)
                n_acc = sum(1 for row in att if row.get("accepted_for_metrics") == "True")
                thr_ds = DEFAULT_RADIUS_M.get("vpair" if ds.endswith("vpair") else ds, 20.0)
                n_uns = 0
                for row in att:
                    if row.get("accepted_for_metrics") != "True":
                        continue
                    try:
                        if float(row["post_fgo_error_m"]) > thr_ds:
                            n_uns += 1
                    except (KeyError, ValueError, TypeError):
                        continue
            else:
                n = max(1, s.get("num_frames") or 1)
                n_acc = s.get("num_accepted_for_metrics") or 0
                n_uns = s.get("num_unsafe_accept") or 0
            u = f"{100 * n_uns / n:.1f}\\%"
            a = f"{100 * n_acc / n:.1f}\\%"
            sev = fmt(s.get("median_unsafe_accept_severity_m"), ".0f")
            if AVL_FOOTPRINT == "oblique" and ds_lab == "avl" and att:
                # Severity over the same genuine-fault trials as the row.
                errs = []
                for row in att:
                    try:
                        e = float(row["post_fgo_error_m"])
                    except (KeyError, ValueError, TypeError):
                        continue
                    if row.get("accepted_for_metrics") == "True" and e > thr_ds:
                        errs.append(e)
                sev = fmt(statistics.median(errs), ".0f") if errs else "--"
            lines.append(f"% source {name}: {match[-1]['run_dir']}")
        lines.append(f"\\newcommand{{\\{name}Unsafe}}{{{u}}}")
        lines.append(f"\\newcommand{{\\{name}Accept}}{{{a}}}")
        lines.append(f"\\newcommand{{\\{name}SevMed}}{{{sev}}}")
        # Attack strength: how far the forced reference actually sits from GT.
        # Over the attacked frames, not an arbitrary first-300 slice: no cell
        # was run on 300 frames, and unattacked frames carry no forced distance.
        off = pending(cell)
        if match:
            vals = []
            for row in _attacked_rows(match[-1]["run_dir"]):
                if AVL_FOOTPRINT == "oblique" and ds.endswith("avl") and not _is_genuine_fault(row, "avl"):
                    continue
                try:
                    vals.append(float(row["adversarial_forced_distance_to_gt_m"]))
                except (KeyError, ValueError, TypeError):
                    continue
            if vals:
                off = fmt(statistics.median(vals), ".0f")
        lines.append(f"\\newcommand{{\\{name}OffsetMed}}{{{off}}}")
    # Stratified FORGE numbers as macros, so prose never hard-codes a value that
    # a later re-run will change underneath it.
    for ds, name in (("avl", "Avl"), ("vpair", "Vpair"), ("uav_visloc", "Visloc")):
        stem = FORGE_MACRO_CELL_STEM.get(ds, ds)
        pooled = {"fault": [0, 0, 0], "covered": [0, 0, 0]}
        thr = DEFAULT_RADIUS_M.get(ds, 20.0)
        got = False
        for cond in FORGE_CONDITIONS:
            match = _forge_cell(runs, f"forge_{stem}", cond)
            if not match:
                continue
            got = True
            st = _forge_strata(match[-1], ds, thr)
            for k in pooled:
                for i in range(3):
                    pooled[k][i] += st[k][i]
        f, c = pooled["fault"], pooled["covered"]
        if not got or f[0] == 0:
            vals = {"FaultN": pending(f"forge_{stem}"), "FaultAccept": pending(f"forge_{stem}"),
                    "FaultUnsafe": pending(f"forge_{stem}"), "CoveredN": pending(f"forge_{stem}"),
                    "CoveredAccept": pending(f"forge_{stem}")}
        else:
            vals = {"FaultN": str(f[0]),
                    "FaultAccept": fmt(f[1] / f[0], pct=True),
                    "FaultUnsafe": fmt(f[2] / f[0], pct=True),
                    "CoveredN": str(c[0]),
                    "CoveredAccept": fmt(c[1] / c[0] if c[0] else None, pct=True)}
        for k, v in vals.items():
            lines.append(f"\\newcommand{{\\forgeStrat{name}{k}}}{{{v}}}")

    # The leaking subset, kept separate from every wrong retrieval. Prose that
    # describes how far the leaked tiles sit needs this population, not the
    # wider one the motivation table's offset columns report.
    for prefix, key in MACRO_CELLS.items():
        r = by_cell.get(key)
        st = wrong_retrieval_stats(r) if r is not None else None
        for suffix, field in (("LeakN", "wrong_accepted"),
                              ("LeakOffsetMin", "leak_offset_min"),
                              ("LeakOffsetMed", "leak_offset_median"),
                              ("LeakOffsetMax", "leak_offset_max"),
                              ("LeakErrMin", "leak_err_min"),
                              ("LeakErrMed", "err_median"),
                              ("LeakErrMax", "err_max"),
                              ("LeakUnderKm", "leak_under_km")):
            val = (st or {}).get(field)
            spec = ".0f" if "Offset" in suffix or "Err" in suffix else "d"
            if val is None:
                # No leaked frames in this cell: emit nothing rather than a
                # dead pending macro (none of these names is used in prose).
                continue
            elif spec == "d":
                out = str(int(val))
            else:
                out = f"{float(val):.0f}"
            lines.append(f"\\newcommand{{\\{prefix}{suffix}}}{{{out}}}")

    # Pooled figures for the two reproduced AVL baselines. The split macros
    # cover their view-angle bands; the deck needs the pooled row.
    for prefix, cellkey in (("avlAnyvisloc", ("avl_native_anyvisloc", "avl")),
                            ("avlOrtholoc", ("avl_native_ortholoc", "avl"))):
        r = by_cell.get(cellkey)
        if r is None:
            for f2 in ("AcceptPct", "Median", "Mean", "Fps"):
                lines.append(f"\\newcommand{{\\{prefix}{f2}}}{{{pending(cellkey[0])}}}")
            continue
        met = row_metrics(r)
        pct = (100.0 * float(met["accepted"]) / float(met["n"])
               if met["accepted"] and met["n"] else None)
        pct_txt = f"{pct:.1f}" + "\\%" if pct is not None else pending(cellkey[0])
        lines.append(f"\\newcommand{{\\{prefix}AcceptPct}}{{{pct_txt}}}")
        lines.append(f"\\newcommand{{\\{prefix}Median}}{{{fmt(met['median_m'], '.1f')}}}")
        lines.append(f"\\newcommand{{\\{prefix}Mean}}{{{fmt(met['mean_m'], '.1f')}}}")
        lines.append(f"\\newcommand{{\\{prefix}Fps}}{{{fmt(met['fps'], '.2f')}}}")
        lines.append(f"% source {prefix} (pooled): {r['run_dir']}")

    # The price of withholding, in accurate poses wrongly rejected: rejected
    # frames of the primary clean runs whose computed pose lies inside the
    # acceptance radius. The number defends the low AVL acceptance rate, and it
    # has to be counted, not asserted.
    for prefix, cellkey, thr in (("avlOurs", ("avl_ours_ultravpr", "avl"), 20.0),
                                 ("vpairOurs", ("vpair_ours_ultravpr", "vpair"), 20.0),
                                 ("vislocOurs", ("visloc_ours_ultravpr", "uav_visloc"), 150.0)):
        r = by_cell.get(cellkey)
        if r is None:
            lines.append(f"\\newcommand{{\\{prefix}GoodRejected}}{{{pending(cellkey[0])}}}")
            continue
        n_good = 0
        for row in _load_predictions(r["run_dir"]):
            if row.get("accepted_for_metrics") == "True":
                continue
            try:
                if float(row["post_fgo_error_m"]) <= thr:
                    n_good += 1
            except (KeyError, ValueError, TypeError):
                continue
        lines.append(f"\\newcommand{{\\{prefix}GoodRejected}}{{{n_good}}}")

    # Frame-level FORGE acceptance: a frame counts once if ANY of its three
    # injected conditions was accepted. This is the clustered reading the
    # intervals paragraph needs beside the per-trial rates.
    for ds, stem, name in (("avl", "avl", "Avl"), ("vpair", "vpair", "Vpair"),
                           ("uav_visloc", "visloc", "Visloc")):
        att_all = None
        acc_any: set = set()
        missing = False
        for cond in FORGE_CONDITIONS:
            m = _forge_cell(runs, f"forge_{stem}", cond)
            if not m:
                missing = True
                break
            att = _attacked_rows(m[-1]["run_dir"])
            names = {row["image_name"] for row in att}
            att_all = names if att_all is None else (att_all & names)
            acc_any |= {row["image_name"] for row in att
                        if row.get("accepted_for_metrics") == "True"}
        if missing or not att_all:
            for suf in ("FrameAcc", "FrameN", "FramePct"):
                lines.append(f"\\newcommand{{\\forge{name}{suf}}}{{{pending(f'forge_{stem}')}}}")
            continue
        acc_in = acc_any & att_all
        pct = 100.0 * len(acc_in) / len(att_all)
        lines.append(f"\\newcommand{{\\forge{name}FrameAcc}}{{{len(acc_in)}}}")
        lines.append(f"\\newcommand{{\\forge{name}FrameN}}{{{len(att_all)}}}")
        pct_txt = f"{pct:.1f}" + "\\%"
        lines.append(f"\\newcommand{{\\forge{name}FramePct}}{{{pct_txt}}}")

    # Seed repeat of the smallest injection cell. The runner is stochastic and
    # order-dependent, so the repeat bounds single-run variance: the RATE is the
    # stable quantity, the identity of the accepted frames is not, and prose
    # must be able to state both without typing a number.
    def _acc_names(cellname):
        m = [r for r in runs if r["cell"] == cellname]
        if not m:
            return None
        return {row.get("image_name")
                for row in _attacked_rows(m[-1]["run_dir"])
                if row.get("accepted_for_metrics") == "True"}
    orig_acc = _acc_names("forgec_avl_wrong_near")
    rep_acc = _acc_names("repeat_forgec_avl_wrong_near")
    if orig_acc is not None and rep_acc is not None:
        lines.append(f"\\newcommand{{\\seedRepeatOrigAcc}}{{{len(orig_acc)}}}")
        lines.append(f"\\newcommand{{\\seedRepeatAcc}}{{{len(rep_acc)}}}")
        lines.append(f"\\newcommand{{\\seedRepeatShared}}{{{len(orig_acc & rep_acc)}}}")
    else:
        for name in ("seedRepeatOrigAcc", "seedRepeatAcc", "seedRepeatShared"):
            lines.append(f"\\newcommand{{\\{name}}}{{{pending('repeat_forgec_avl')}}}")

    # Falsification test of Section 3.9, counted rather than asserted.
    ref = fault_refutation_stats(runs)
    lines.append(f"\\newcommand{{\\faultToleranceM}}{{{ref['tolerance_m']:.0f}}}")
    for ds, name in (("avl", "Avl"), ("vpair", "Vpair"), ("uav_visloc", "Visloc")):
        st = ref.get(ds) or {}
        lines.append(f"\\newcommand{{\\faultRefute{name}}}{{{st.get('refutations', 0)}}}")
        lines.append(f"\\newcommand{{\\faultRefute{name}Frames}}{{{st.get('frames', 0)}}}")
        lines.append(f"\\newcommand{{\\faultRefute{name}Trials}}{{{st.get('trials', 0)}}}")
        lo, hi = st.get("overshoot_min"), st.get("overshoot_max")
        # Zero refutations on this dataset means no overshoot exists: emit
        # nothing rather than a dead pending macro.
        if lo is not None:
            lines.append(f"\\newcommand{{\\faultRefute{name}OverMin}}{{{fmt(lo, '.1f')}}}")
        if hi is not None:
            lines.append(f"\\newcommand{{\\faultRefute{name}OverMax}}{{{fmt(hi, '.1f')}}}")
    return "\n".join(lines) + "\n"


def ch3_macros_tex(runs: List[Dict[str, Any]],
                   dev_csv: Optional[Path] = None) -> str:
    """Chapter 3 methodology macros backed by the explicit development CSV.

    dev_csv is the predictions.csv from a full-route margin-0 development run
    (the same run whose displaced/repaired counts appear in the
    _challenger_passes_margin docstring). When absent, no unsupported macro is
    emitted.
    """
    lines = ["% Chapter 3 methodology macros"]
    mg = margin_guard_stats(dev_csv)
    if mg:
        lines.append(f"% source marginGuard: {dev_csv}")
        lines.append(f"\\newcommand{{\\marginGuardDisplaced}}{{{mg['displaced']}}}")
        lines.append(f"\\newcommand{{\\marginGuardRepaired}}{{{mg['repaired']}}}")

    hs = height_scale_correlation(dev_csv)
    if hs:
        lines.append(f"\\newcommand{{\\heightScaleCorr}}{{{fmt(hs['correlation'], '.3f')}}}")
        lines.append(f"\\newcommand{{\\heightScaleN}}{{{hs['n']}}}")

    return "\n".join(lines) + "\n"


def markdown_preview(runs: List[Dict[str, Any]]) -> str:
    out = ["# v3 results preview", ""]
    for dataset in ("avl", "vpair", "uav_visloc"):
        rows = [r for r in runs if r["dataset"] == dataset
                and r["summary"].get("adversarial_mode", "none") == "none"]
        if not rows:
            continue
        out += [f"## {dataset}", "",
                "| Method (cell) | n | acc | Med (m) | Mean (m) | A@5 | A@10 | Yaw med | FPS |",
                "|---|---|---|---|---|---|---|---|---|"]
        rows.sort(key=lambda r: METHOD_ORDER.index(r["pipeline"]) if r["pipeline"] in METHOD_ORDER else 99)
        for r in rows:
            met = row_metrics(r)

            def p(v, pct=False):
                return fmt(v, pct=pct).replace("\\%", "%")

            out.append(
                f"| {_method_label(r)} ({r['cell']}) | {met['n']} | {met['accepted']} "
                f"| {p(met['median_m'])} | {p(met['mean_m'])} | {p(met['a5'], True)} | {p(met['a10'], True)} "
                f"| {p(met['yaw_med'])} | {p(met['fps'])} |"
            )
        out.append("")
    return "\n".join(out)


# Direction is fixed a priori from CAVES_V0_COMPONENT_SPECS, so the sign is not
# chosen after seeing which way the AUROC falls. A value below 0.5 therefore
# means the component's assumed direction is inverted on that dataset, which is
# a result and not a bookkeeping error.
CAVES_SINGLE_FEATURES = (
    ("Inlier ratio",          "inlier_ratio",             -1.0, 1.5),
    ("Reprojection residual", "reproj_residual_px",       +1.0, 1.2),
    ("Inlier count",          "num_inliers",              -1.0, 1.0),
    ("Shape confidence",      "shape_confidence",         -1.0, 1.0),
    ("Anchor offset",         "geometry_anchor_offset_m", +1.0, 1.0),
    ("Retrieval margin",      "retrieval_score_margin",   -1.0, 0.7),
    ("Retrieval rank",        "candidate_rank_selected",  +1.0, 0.4),
)


def _caves_labelled(run: Optional[Dict[str, Any]]) -> List[tuple]:
    """Accepted frames of one clean run as (row, unsafe) pairs.

    Same population and same unsafe rule as caves_stats, so the component table
    and the headline AUROC table cannot disagree about who is being scored.
    """
    if run is None:
        return []
    thr = DEFAULT_RADIUS_M.get(run["dataset"], 20.0)
    out = []
    for row in _load_predictions(run["run_dir"]):
        if row.get("accepted_for_metrics") != "True":
            continue
        try:
            err = float(row["post_fgo_error_m"])
        except (KeyError, ValueError, TypeError):
            continue
        if err != err:
            continue
        out.append((row, err > thr))
    return out


def _auroc_of(pairs: List[tuple], column: str, sign: float) -> Optional[float]:
    frames = []
    for row, bad in pairs:
        try:
            v = float(row[column])
        except (KeyError, ValueError, TypeError):
            continue
        if v != v:
            continue
        frames.append((sign * v, bad))
    if len(frames) < 20:
        return None
    pos = sum(1 for _, b in frames if b)
    if pos == 0 or pos == len(frames):
        return None
    au = _rank_auroc(frames)
    return au if au == au else None


def caves_component_table(runs: List[Dict[str, Any]]) -> str:
    """Every evidence component scored on its own, beside their combination."""
    by_cell = {(r["cell"], r["dataset"]): r for r in runs}
    order, cols = [], {}
    for key in MACRO_CELLS.values():
        r = by_cell.get(key)
        pairs = _caves_labelled(r)
        if not pairs:
            continue
        ds = r["dataset"]
        if ds in cols:
            continue
        order.append(ds)
        cols[ds] = pairs
    if not order:
        return ""
    lines = _table_head(
        "\\textbf{Each evidence component scored on its own.} Rank AUROC against "
        "the same unsafe-pose label and the same accepted population as "
        "Table~\\ref{tab:v3-caves}. Weight is the coefficient the component "
        "carries in the combined score. Each component is read straight from "
        "the per-frame record, and its direction is fixed in advance by the "
        "component specification rather than chosen after the fact, so a value "
        "below one half means that component points the wrong way on that "
        "dataset. Seven of the nine components appear: the gate-outcome "
        "component is not exported to the per-frame record and the "
        "protocol-violation component is constant on these runs, so neither "
        "can be scored. The last row is the exported runtime score, which on "
        "accepted poses is a weighted combination of these seven active "
        "components, the two protocol components having dropped out "
        "(Section~\\ref{sec:method-caves}). The table reads this score from "
        "the record rather than recomputing it.",
        "tab:v3-caves-components", "lr" + "r" * len(order),
        "Evidence component & Weight & " + " & ".join(DATASET_DISPLAY.get(d, d) for d in order) + " \\\\",
        short="Each evidence component scored on its own",
    )
    for name, col, sign, weight in CAVES_SINGLE_FEATURES:
        cells = []
        for ds in order:
            au = _auroc_of(cols[ds], col, sign)
            cells.append("--" if au is None else f"{au:.3f}")
        if all(c == "--" for c in cells):
            continue
        lines.append(f"{name} & {weight:g} & " + " & ".join(cells) + " \\\\")
    lines.append("\\midrule")
    combo = []
    for ds in order:
        au = _auroc_of(cols[ds], "caves_risk_score", +1.0)
        combo.append("--" if au is None else f"\\textbf{{{au:.3f}}}")
    lines.append("CAVES, combined score & & " + " & ".join(combo) + " \\\\")
    lines += _table_tail()
    return "\n".join(lines)


COVERAGE_POINTS = (1.00, 0.90, 0.75, 0.50, 0.25)


def caves_coverage_table(runs: List[Dict[str, Any]]) -> str:
    """What withholding the riskiest share of an accepted set actually buys."""
    by_cell = {(r["cell"], r["dataset"]): r for r in runs}
    lines = _table_head(
        "\\textbf{Unsafe rate after withholding the riskiest share.} The accepted "
        "poses of each primary clean run are ranked by CAVES score and the "
        "riskiest are withheld. Coverage is the share kept; the entry is the "
        "unsafe rate among the kept poses, against the acceptance radius of that "
        "dataset. The leftmost column is the accepted set untouched, so each row "
        "reads left to right as the price of a decision the operator makes after "
        "the run.",
        "tab:v3-caves-coverage", "lr" + "r" * len(COVERAGE_POINTS),
        "Dataset & Accepted n & " +
        " & ".join(f"keep {100*c:.0f}\\%" for c in COVERAGE_POINTS) + " \\\\",
        short="Unsafe rate after withholding the riskiest share",
    )
    seen = set()
    for key in MACRO_CELLS.values():
        r = by_cell.get(key)
        pairs = _caves_labelled(r)
        if not pairs or r["dataset"] in seen:
            continue
        seen.add(r["dataset"])
        scored = []
        for row, bad in pairs:
            try:
                sc = float(row["caves_risk_score"])
            except (KeyError, ValueError, TypeError):
                continue
            if sc != sc:
                continue
            scored.append((sc, bad))
        if not scored:
            continue
        scored.sort(key=lambda t: t[0])          # safest first, riskiest last
        n = len(scored)
        cells = []
        for c in COVERAGE_POINTS:
            k = max(1, int(round(c * n)))
            kept = scored[:k]
            rate = 100.0 * sum(1 for _, b in kept if b) / len(kept)
            cells.append(f"{rate:.1f}\\%")
        lines.append(f"% source: {r['run_dir']}")
        lines.append(f"{DATASET_DISPLAY.get(r['dataset'], r['dataset'])} & {n} & "
                     + " & ".join(cells) + " \\\\")
    lines += _table_tail()
    return "\n".join(lines)


def forge_attack_geometry_table(runs: List[Dict[str, Any]]) -> str:
    """How far the forced reference sat from the truth, per condition.

    Support material, deliberately not a body table. The distance from a tile
    anchor to the query truth is a DESCRIPTION of the attack, and the thesis
    argues at length that a centre-distance rule is the wrong way to decide
    whether a reference is wrong. Printing it beside the headline acceptance
    rates would put that number where a reader can mistake it for the
    eligibility criterion, so it lives here instead, where it answers "how far
    apart are the three conditions" without competing with
    Definition~\\ref{def:wrong-reference}. Median and IQR rather than median and
    maximum: the maximum is set by the extent of the reference map, not by the
    attack.
    """
    lines = _table_head(
        "\\textbf{Geometry of the forced reference, by injection condition.} "
        "Anchor offset is the distance from the forced reference anchor to the "
        "query's true position, over the attacked frames of the newest run of "
        "each injection cell, as median [interquartile "
        "range]. It describes the attack and is neither the eligibility "
        "criterion nor the definition of a wrong reference, which remains zero "
        "footprint overlap under Definition~\\ref{def:wrong-reference}. Values "
        "are not comparable across datasets, whose reference geometry differs.",
        "tab:forge-attack-geometry", "llrr",
        "Dataset & Condition & Trials & Anchor offset (m), median [IQR] \\\\",
        short="Geometry of the forced reference",
    )
    found = False
    for ds in ("avl", "vpair", "uav_visloc"):
        stem = FORGE_MACRO_CELL_STEM.get(ds, ds)
        for cond in FORGE_CONDITIONS:
            match = _forge_cell(runs, f"forge_{stem}", cond)
            if not match:
                continue
            run = match[-1]
            # The runner records the forced distance itself. Use that column
            # rather than recomputing from the anchor, so this table and the
            # OffsetMed macros cannot disagree by a metre or two over a
            # different distance formula.
            offsets = []
            for row in _attacked_rows(run["run_dir"]):
                try:
                    offsets.append(float(row["adversarial_forced_distance_to_gt_m"]))
                except (KeyError, ValueError, TypeError):
                    continue
            if len(offsets) < 5:
                continue
            found = True
            offsets.sort()
            n = len(offsets)
            med = statistics.median(offsets)
            q1 = offsets[max(0, int(0.25 * n) - 1)]
            q3 = offsets[min(n - 1, int(0.75 * n))]
            lines.append(f"% source: {run['run_dir']}")
            lines.append(
                f"{DATASET_DISPLAY.get(ds, ds)} & {COND_DISPLAY.get(cond, cond)} & "
                f"{n} & {med:.0f} [{q1:.0f}, {q3:.0f}] \\\\")
    if not found:
        return ""
    lines += _table_tail()
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("results_root", type=Path)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--tile-probe-dirs", type=Path, nargs="*", default=[],
                    help="Explicit UAV-VisLoc tile-probe run dirs (30-frame, flight 01)")
    ap.add_argument("--attempt1-root", type=Path, default=None,
                    help="Archived pre-gate matrix root for the gate-tuning table")
    ap.add_argument("--native-scripts-dir", type=Path, default=None,
                    help="Dir with vpair native script CSVs (vpair_yaw_height_results.csv, ...)")
    ap.add_argument("--script-logs-dir", type=Path, default=None,
                    help="Dir with native-script logs (for throughput recovery)")
    ap.add_argument("--anyvisloc-log", type=Path, default=None,
                    help="Fallback: anyvisloc script log when its CSV write crashed")
    ap.add_argument("--dev-csv", type=Path, default=None,
                    help="Full-route margin-0 dev predictions.csv for ch3 macros "
                         "(margin guard displacement + height-from-scale correlation)")
    args = ap.parse_args()

    out_dir = args.out_dir or (args.results_root / "tables_v3")
    out_dir.mkdir(parents=True, exist_ok=True)

    runs = load_runs(args.results_root)
    if not runs:
        print(f"No metrics_summary.json found under {args.results_root}", file=sys.stderr)
        return 1

    (out_dir / "tab_loc_avl.tex").write_text(loc_table(
        runs, "avl", "Localization on AVL (487 frames, full set; every row whose run records a per-frame view angle is split into nadir and oblique sub-rows; the 24 frames between $15^\\circ$ and $40^\\circ$ appear only in the pooled row).",
        "tab:v3-loc-avl", avl_split=True,
        short="Localization on AVL") + "\n")
    (out_dir / "tab_loc_vpair.tex").write_text(loc_table(
        runs, "vpair", "Localization on VPAIR (contiguous 800-frame subset). "
        "Native-script A@$N$ conditions on the frames for which the script "
        "produced a finite solution; their Accept column gives that solved share.",
        "tab:v3-loc-vpair", short="Localization on VPAIR",
        extra_rows=_merge_script_rows(
            native_script_rows(args.native_scripts_dir, args.script_logs_dir),
            anyvisloc_log_row(args.anyvisloc_log))) + "\n")
    (out_dir / "tab_loc_uav_visloc.tex").write_text(loc_table(
        runs, "uav_visloc",
        "\\textbf{Localization on UAV-VisLoc.} Flights 01, 04 and 11, the full "
        "usable set. This dataset ships no co-registered elevation model, so "
        "neither native baseline runs on it. No runnable wrapper comparator "
        "exists for it either, so the block holds a single row. "
        "The row is therefore a measurement and not a comparison, and it is "
        "reported here because the reliability analysis of "
        "Section~\\ref{sec:exp-decomposition} rests on it. Nothing in this table "
        "is comparable with Table~\\ref{tab:v3-loc-merged}, whose datasets have "
        "different reference geometry and a different acceptance radius.",
        "tab:v3-loc-visloc", short="Localization on UAV-VisLoc") + "\n")
    (out_dir / "tab_loc_merged.tex").write_text(merged_loc_table(
        runs, pipelines=COMPARISON_PIPELINES,
        script_rows_fn=lambda: _merge_script_rows(
            native_script_rows(args.native_scripts_dir, args.script_logs_dir, merged=True),
            anyvisloc_log_row(args.anyvisloc_log))) + "\n")
    (out_dir / "tab_loc_variants.tex").write_text(merged_loc_table(
        runs, pipelines=VARIANT_PIPELINES,
        label="tab:v3-loc-variants",
        short="Our own variants, side by side",
        caption=(
            "\\textbf{The variants of the proposed cascade, side by side.} One "
            "block per dataset, on the same frames as "
            "Table~\\ref{tab:v3-loc-merged}. The backbone swap changes only the "
            "retrieval network and runs under the shared acceptance "
            "configuration. The elevation-backed head is not a solver swap and "
            "must not be read as one. Its run manifest differs from the primary "
            "cascade in the retrieval backbone (CAMP, not UltraVPR), the matcher "
            "(RoMa, not ALIKED with LightGlue), the coarse-to-fine alignment and "
            "the inverse perspective warp, both of which it omits, the LGI "
            "filter and the fusion step, which it also omits, and four "
            "acceptance bounds (Section~\\ref{sec:method-geometry}). It is a "
            "second system that happens to share a code base, so neither its "
            "acceptance rate nor its error is attributable to the elevation "
            "model alone, and neither is comparable with the rows above it. It "
            "runs on AVL alone, because AVL is the only dataset that ships a "
            "co-registered elevation model, which is why the AVL block carries "
            "one row more than the VPAIR block. Neither variant closes the gap "
            "to the strongest reproduced baseline on AVL, and the oblique band "
            "where the elevation-backed head is meant to help is priced "
            "separately in Table~\\ref{tab:v3-avl-view-angle}.")) + "\n")
    (out_dir / "tab_ablation.tex").write_text(ablation_table(runs) + "\n")
    (out_dir / "tab_retrieval.tex").write_text(retrieval_table(runs) + "\n")
    (out_dir / "tab_capability.tex").write_text(capability_table(runs) + "\n")
    (out_dir / "tab_forge_motivation.tex").write_text(motivation_table(runs) + "\n")
    for ds, tex in forge_tables(runs).items():
        (out_dir / f"tab_forge_{ds}.tex").write_text(tex + "\n")
    (out_dir / "tab_forge_attack_geometry.tex").write_text(forge_attack_geometry_table(runs) + "\n")
    (out_dir / "tab_forge_stratified.tex").write_text(forge_stratified_table(runs) + "\n")
    (out_dir / "tab_caves_components.tex").write_text(caves_component_table(runs) + "\n")
    (out_dir / "tab_caves_coverage.tex").write_text(caves_coverage_table(runs) + "\n")
    (out_dir / "tab_caves.tex").write_text(caves_table(runs) + "\n")
    (out_dir / "tab_reject_codes.tex").write_text(reject_code_table(runs) + "\n")
    (out_dir / "tab_failure_cause.tex").write_text(failure_cause_table(runs) + "\n")
    (out_dir / "tab_tile_tuning.tex").write_text(tile_tuning_table(list(args.tile_probe_dirs)) + "\n")
    (out_dir / "tab_gate_tuning.tex").write_text(
        gate_tuning_table(runs, args.attempt1_root, which="gates") + "\n")
    (out_dir / "tab_fusion_prior.tex").write_text(
        gate_tuning_table(runs, args.attempt1_root, which="fusion") + "\n")
    (out_dir / "tab_avl_view_angle.tex").write_text(avl_view_angle_table(runs) + "\n")
    (out_dir / "v3_numbers.tex").write_text(
        macros_tex(runs) + ch3_macros_tex(runs, dev_csv=args.dev_csv))
    (out_dir / "preview.md").write_text(markdown_preview(runs) + "\n")

    print(f"Wrote {len(list(out_dir.glob('*.tex')))} .tex files + preview.md to {out_dir}")
    print(f"Runs consumed: {len(runs)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
