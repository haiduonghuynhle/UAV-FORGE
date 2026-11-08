"""Export one benchmark run directory to results.xlsx.

Sheet `per_frame` mirrors reference/summary_benchmarking.xlsx (stem, gt/pred
pose, pos/yaw/alt errors, inliers, timings); sheet `summary` flattens
metrics_summary.json. Usage:

    python scripts/reporting/export_run_xlsx.py <run_dir> [--out results.xlsx]

`run_dir` is the timestamped directory holding predictions.csv.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd

# Canonical schema. Aliases cover ours (predictions.csv), native AnyVisLoc
# (stem/est_yaw_deg/alt_err_m/camp_q_ms) and native OrthoLoC
# (image/yaw_est_deg/height_err_m/pos_err_2d_m) so every run normalizes to
# the same columns; a method that does not measure a field leaves it blank.
PER_FRAME_COLS = [
    ("stem", ["image_name", "query", "stem", "image", "query_image"]),
    ("success", ["success"]),
    ("accepted", ["accepted_for_metrics"]),
    ("gt_lat", ["gt_lat"]), ("gt_lon", ["gt_lon"]), ("gt_alt", ["gt_alt", "gt_alt_m"]),
    ("pred_lat", ["pred_lat"]), ("pred_lon", ["pred_lon"]), ("pred_alt", ["pred_alt"]),
    ("pos_error_m", ["post_fgo_error_m", "position_error_m", "pos_error_m", "pos_err_2d_m", "pos_err_m"]),
    ("pos_error_3d_m", ["pos_err_3d_m"]),
    ("gt_yaw_deg", ["gt_heading", "gt_yaw", "gt_yaw_deg", "yaw_gt_deg"]),
    ("pred_yaw_deg", ["pred_heading", "pred_yaw", "est_yaw_deg", "yaw_est_deg", "src_yaw_deg"]),
    ("yaw_err_deg", ["yaw_error_deg", "yaw_err_deg"]),
    ("alt_err_m", ["alt_error_m", "alt_err_m", "height_err_m"]),
    ("pitch_err_deg", ["pitch_error_deg"]),
    ("n_inliers", ["num_inliers", "n_inliers"]),
    ("retrieval_rank", ["retrieval_rank"]),
    ("top1_correct", ["top1_correct", "tile_identity_correct"]),
    ("retrieval_score", ["retrieval_score", "retrieval_score_top1"]),
    ("reproj_error_px", ["reproj_error_px", "pnp_reprojection_error_px"]),
    ("bottleneck", ["bottleneck_label"]),
    ("retrieval_ms", ["timing_retrieval_ms", "camp_q_ms"]),
    ("matching_ms", ["roma_ms"]),
    ("fgo_ms", ["timing_fgo_ms"]),
    ("total_ms", ["total_online_ms"]),
]


def export_run(run_dir: Path, out_path: Path | None = None) -> Path:
    run_dir = Path(run_dir)
    csv_path = run_dir / "predictions.csv"
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)
    df = pd.read_csv(csv_path)

    frame = pd.DataFrame()
    for out_col, candidates in PER_FRAME_COLS:
        for c in candidates:
            if c in df.columns:
                frame[out_col] = df[c]
                break
    if "alt_err_m" not in frame.columns and {"pred_alt", "gt_alt"} <= set(frame.columns):
        frame["alt_err_m"] = (frame["pred_alt"] - frame["gt_alt"]).abs()

    summary_rows = []
    ms_path = run_dir / "metrics_summary.json"
    if ms_path.exists():
        def _flatten(obj, prefix=""):
            if isinstance(obj, dict):
                for k, v in obj.items():
                    _flatten(v, f"{prefix}{k}." if isinstance(v, dict) else f"{prefix}{k}")
            elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
                if not (isinstance(obj, float) and math.isnan(obj)):
                    summary_rows.append({"metric": prefix, "value": obj})
        _flatten(json.loads(ms_path.read_text()))
    summary_rows.append({"metric": "n_frames", "value": len(df)})
    if "accepted" in frame.columns:
        summary_rows.append({"metric": "n_accepted", "value": int(frame["accepted"].sum())})

    out_path = Path(out_path) if out_path else run_dir / "results.xlsx"
    with pd.ExcelWriter(out_path, engine="openpyxl") as xw:
        frame.to_excel(xw, sheet_name="per_frame", index=False)
        pd.DataFrame(summary_rows).to_excel(xw, sheet_name="summary", index=False)
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    print(export_run(args.run_dir, args.out))


if __name__ == "__main__":
    main()
