#!/usr/bin/env python3
"""Export CDF and failure-decomposition artifacts from benchmark roots.

This script is intentionally lightweight: it parses JSON/CSV artifacts only and
never imports the benchmark runner or model code. It can therefore run on local
machines or on after a long experiment without touching GPU state.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

EARTH_RADIUS_M = 6_371_008.8
DEFAULT_THRESHOLDS_M = (5.0, 10.0, 20.0, 50.0, 150.0, 500.0, 1000.0)
OFFICIAL_TIERS = {"strict_e2e", "wrapper_e2e"}


@dataclass
class RunInfo:
    root: Path
    stage: str
    dataset: str
    pipeline: str
    run_id: str
    tier: str
    official_like: bool
    summary_path: Path
    predictions_path: Optional[Path]
    metrics: Dict[str, object]


def _read_json(path: Path) -> Dict[str, object]:
    try:
        data = json.loads(path.read_text())
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _safe_float(value: object) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out):
        return None
    return out


def _safe_int(value: object, default: int = 0) -> int:
    try:
        if value is None or value == "":
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    return 2.0 * EARTH_RADIUS_M * math.atan2(math.sqrt(a), math.sqrt(max(0.0, 1.0 - a)))


def _infer_parts(summary_path: Path, root: Path, metrics: Dict[str, object]) -> Tuple[str, str, str, str]:
    rel = summary_path.relative_to(root)
    parts = rel.parts
    # Common layout: stage/dataset/pipeline/run/paper_metrics_summary.json.
    if len(parts) >= 5:
        stage, dataset, pipeline, run_id = parts[0], parts[-4], parts[-3], parts[-2]
    elif len(parts) >= 4:
        stage, dataset, pipeline, run_id = ".", parts[-4], parts[-3], parts[-2]
    else:
        stage, dataset, pipeline, run_id = ".", str(metrics.get("dataset") or "unknown"), str(metrics.get("pipeline") or "unknown"), summary_path.parent.name
    dataset = str(metrics.get("dataset") or dataset)
    pipeline = str(metrics.get("pipeline") or pipeline)
    return stage, dataset, pipeline, run_id


def _is_forge_metrics(metrics: Dict[str, object]) -> bool:
    mode = str(metrics.get("adversarial_mode", "none") or "none").strip().lower()
    family = str(metrics.get("benchmark_family", "") or "").strip().lower()
    return bool(
        family == "forge"
        or mode != "none"
        or bool(metrics.get("adversarial_diagnostic", False))
        or metrics.get("official_metric_eligible") is False and mode != "none"
    )


def discover_runs(roots: Sequence[Path]) -> List[RunInfo]:
    runs: List[RunInfo] = []
    seen: set[Path] = set()
    for root in roots:
        root = root.resolve()
        for summary_path in sorted(root.glob("**/paper_metrics_summary.json")):
            summary_path = summary_path.resolve()
            if summary_path in seen:
                continue
            seen.add(summary_path)
            metrics = _read_json(summary_path)
            stage, dataset, pipeline, run_id = _infer_parts(summary_path, root, metrics)
            tier = str(metrics.get("comparison_tier") or "unknown")
            official_like = (
                tier in OFFICIAL_TIERS
                and not bool(metrics.get("uses_oracle", False))
                and not bool(metrics.get("is_native", False))
                and not bool(metrics.get("is_hybrid", False))
                and not bool(metrics.get("retrieval_only", False))
                and not _is_forge_metrics(metrics)
                and _safe_int(metrics.get("retrieval_protocol_violation_count"), 0) == 0
                and bool(metrics.get("run_integrity_for_official_metrics", True))
            )
            predictions_path = summary_path.parent / "predictions.csv"
            runs.append(
                RunInfo(
                    root=root,
                    stage=stage,
                    dataset=dataset,
                    pipeline=pipeline,
                    run_id=run_id,
                    tier=tier,
                    official_like=official_like,
                    summary_path=summary_path,
                    predictions_path=predictions_path if predictions_path.exists() else None,
                    metrics=metrics,
                )
            )
    return runs


def _row_error_m(row: Dict[str, str]) -> Optional[float]:
    for key in ("position_error_m", "horizontal_error_m", "E_ME_m", "error_m", "position_error"):
        val = _safe_float(row.get(key))
        if val is not None:
            return val
    gt_lat = _safe_float(row.get("gt_lat"))
    gt_lon = _safe_float(row.get("gt_lon"))
    pred_lat = _safe_float(row.get("pred_lat"))
    pred_lon = _safe_float(row.get("pred_lon"))
    if None not in (gt_lat, gt_lon, pred_lat, pred_lon):
        return _haversine_m(gt_lat, gt_lon, pred_lat, pred_lon)  # type: ignore[arg-type]
    return None


def _is_truthy(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y", "ok"}


def collect_prediction_rows(runs: Sequence[RunInfo], official_only: bool) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    cdf_rows: List[Dict[str, object]] = []
    failure_rows: List[Dict[str, object]] = []
    for run in runs:
        if official_only and not run.official_like:
            continue
        if run.predictions_path is None:
            failure_rows.append(_failure_summary_from_run(run, "missing_predictions_csv", 1))
            continue
        try:
            with run.predictions_path.open(newline="") as f:
                reader = csv.DictReader(f)
                for idx, row in enumerate(reader):
                    accepted = _is_truthy(row.get("accepted_for_metrics", row.get("success", "")))
                    pose_success = _is_truthy(row.get("pose_success", row.get("success", "")))
                    failure = row.get("failure_reason") or row.get("bottleneck_label") or ("not_accepted" if not accepted else "")
                    error = _row_error_m(row) if accepted else None
                    if error is not None:
                        cdf_rows.append(
                            {
                                "root": str(run.root),
                                "stage": run.stage,
                                "dataset": run.dataset,
                                "pipeline": run.pipeline,
                                "run_id": run.run_id,
                                "tier": run.tier,
                                "official_like": run.official_like,
                                "frame_index": idx,
                                "image_name": row.get("image_name", ""),
                                "position_error_m": error,
                                "accepted_for_metrics": accepted,
                                "pose_success": pose_success,
                                "failure_reason": failure,
                            }
                        )
                    else:
                        reason = failure or ("missing_error_for_accepted_frame" if accepted else "not_accepted")
                        failure_rows.append(_failure_summary_from_run(run, reason, 1))
        except Exception as exc:
            failure_rows.append(_failure_summary_from_run(run, f"predictions_parse_error:{type(exc).__name__}", 1))
    return cdf_rows, failure_rows


def _failure_summary_from_run(run: RunInfo, reason: str, count: int) -> Dict[str, object]:
    return {
        "root": str(run.root),
        "stage": run.stage,
        "dataset": run.dataset,
        "pipeline": run.pipeline,
        "run_id": run.run_id,
        "tier": run.tier,
        "official_like": run.official_like,
        "failure_reason": reason or "unknown",
        "count": count,
    }


def aggregate_failures(frame_failures: Sequence[Dict[str, object]], runs: Sequence[RunInfo], official_only: bool) -> List[Dict[str, object]]:
    counter: Counter[Tuple[str, str, str, str, str, bool, str]] = Counter()
    for row in frame_failures:
        key = (
            str(row["stage"]),
            str(row["dataset"]),
            str(row["pipeline"]),
            str(row["run_id"]),
            str(row["tier"]),
            bool(row["official_like"]),
            str(row["failure_reason"] or "unknown"),
        )
        counter[key] += _safe_int(row.get("count"), 1)

    # Add summary-level frame exception/protocol/final-gate counters even if rows were accepted.
    for run in runs:
        if official_only and not run.official_like:
            continue
        summary_keys = {
            "frame_exception": "frame_exceptions",
            "retrieval_protocol_violation": "retrieval_protocol_violation_count",
            "final_pose_gate_reject": "num_pose_rejected_by_final_gate",
            "fgo_guard_reject": "num_fgo_rejected_by_guard",
            "geometry_reject_reproj": "num_geometry_reject_reproj",
            "geometry_reject_inlier": "num_geometry_reject_inlier",
            "geometry_reject_anchor_offset": "num_geometry_reject_anchor_offset",
            "unsafe_accept": "num_unsafe_accept",
            "adversarial_fail_closed": "num_adversarial_fail_closed",
            "adversarial_metadata_unavailable": "num_adversarial_metadata_unavailable",
        }
        for reason, metric_key in summary_keys.items():
            value = _safe_int(run.metrics.get(metric_key), 0)
            if value > 0:
                counter[(run.stage, run.dataset, run.pipeline, run.run_id, run.tier, run.official_like, reason)] += value

    rows = []
    for (stage, dataset, pipeline, run_id, tier, official_like, reason), count in sorted(counter.items()):
        rows.append(
            {
                "stage": stage,
                "dataset": dataset,
                "pipeline": pipeline,
                "run_id": run_id,
                "tier": tier,
                "official_like": official_like,
                "failure_reason": reason,
                "count": count,
            }
        )
    return rows


def write_csv(path: Path, rows: Sequence[Dict[str, object]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_cdf_points(rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    grouped: Dict[Tuple[str, str, str, str, bool], List[float]] = defaultdict(list)
    for row in rows:
        key = (str(row["stage"]), str(row["dataset"]), str(row["pipeline"]), str(row["tier"]), bool(row["official_like"]))
        error = _safe_float(row.get("position_error_m"))
        if error is not None:
            grouped[key].append(error)
    out: List[Dict[str, object]] = []
    for (stage, dataset, pipeline, tier, official_like), values in sorted(grouped.items()):
        values = sorted(values)
        n = len(values)
        for idx, error in enumerate(values, start=1):
            out.append(
                {
                    "stage": stage,
                    "dataset": dataset,
                    "pipeline": pipeline,
                    "tier": tier,
                    "official_like": official_like,
                    "rank": idx,
                    "n": n,
                    "position_error_m": error,
                    "cdf": idx / n if n else 0.0,
                }
            )
    return out


def percentile(values: Sequence[float], pct: float) -> Optional[float]:
    finite = sorted(v for v in values if math.isfinite(v))
    if not finite:
        return None
    if len(finite) == 1:
        return finite[0]
    rank = (len(finite) - 1) * pct / 100.0
    lo = math.floor(rank)
    hi = math.ceil(rank)
    if lo == hi:
        return finite[int(rank)]
    return finite[lo] * (hi - rank) + finite[hi] * (rank - lo)


def build_threshold_summary(rows: Sequence[Dict[str, object]], thresholds: Sequence[float]) -> List[Dict[str, object]]:
    grouped: Dict[Tuple[str, str, str, str, bool], List[float]] = defaultdict(list)
    for row in rows:
        key = (str(row["stage"]), str(row["dataset"]), str(row["pipeline"]), str(row["tier"]), bool(row["official_like"]))
        error = _safe_float(row.get("position_error_m"))
        if error is not None:
            grouped[key].append(error)
    out: List[Dict[str, object]] = []
    for (stage, dataset, pipeline, tier, official_like), values in sorted(grouped.items()):
        n = len(values)
        base = {
            "stage": stage,
            "dataset": dataset,
            "pipeline": pipeline,
            "tier": tier,
            "official_like": official_like,
            "n": n,
            "mean_m": sum(values) / n if n else "",
            "median_m": percentile(values, 50) or "",
            "p75_m": percentile(values, 75) or "",
            "p90_m": percentile(values, 90) or "",
            "p95_m": percentile(values, 95) or "",
        }
        for threshold in thresholds:
            row = dict(base)
            row["threshold_m"] = threshold
            row["accuracy_at_threshold"] = sum(1 for v in values if v <= threshold) / n if n else ""
            out.append(row)
    return out


def write_summary_md(path: Path, threshold_rows: Sequence[Dict[str, object]], failure_rows: Sequence[Dict[str, object]], roots: Sequence[Path], official_only: bool) -> None:
    lines = [
        "# CDF and Failure-Decomposition Summary",
        "",
        f"- Roots: {', '.join(str(r) for r in roots)}",
        f"- Official-only filter: `{official_only}`",
        "",
        "## CDF / Threshold Metrics",
        "",
        "| Stage | Dataset | Pipeline | Tier | N | Mean m | Median m | P90 m | A@10m | A@50m | A@150m |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    grouped: Dict[Tuple[str, str, str, str, bool], Dict[float, Dict[str, object]]] = defaultdict(dict)
    for row in threshold_rows:
        key = (str(row["stage"]), str(row["dataset"]), str(row["pipeline"]), str(row["tier"]), bool(row["official_like"]))
        grouped[key][_safe_float(row.get("threshold_m")) or -1.0] = row
    for (stage, dataset, pipeline, tier, _official_like), by_threshold in sorted(grouped.items()):
        first = next(iter(by_threshold.values()))
        def fmt(value: object, digits: int = 2) -> str:
            val = _safe_float(value)
            return "--" if val is None else f"{val:.{digits}f}"
        def acc(thr: float) -> str:
            return fmt(by_threshold.get(thr, {}).get("accuracy_at_threshold"), 3)
        lines.append(
            f"| {stage} | {dataset} | {pipeline} | {tier} | {first.get('n')} | "
            f"{fmt(first.get('mean_m'))} | {fmt(first.get('median_m'))} | {fmt(first.get('p90_m'))} | "
            f"{acc(10.0)} | {acc(50.0)} | {acc(150.0)} |"
        )
    lines.extend(["", "## Failure Decomposition", "", "| Stage | Dataset | Pipeline | Reason | Count |", "|---|---|---|---|---:|"])
    for row in failure_rows:
        lines.append(
            f"| {row.get('stage')} | {row.get('dataset')} | {row.get('pipeline')} | {row.get('failure_reason')} | {row.get('count')} |"
        )
    path.write_text("\n".join(lines) + "\n")


def maybe_write_png(cdf_rows: Sequence[Dict[str, object]], output_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        return
    grouped: Dict[Tuple[str, str, str], List[Tuple[float, float]]] = defaultdict(list)
    for row in cdf_rows:
        grouped[(str(row["stage"]), str(row["dataset"]), str(row["pipeline"]))].append(
            (float(row["position_error_m"]), float(row["cdf"]))
        )
    for dataset in sorted({k[1] for k in grouped}):
        plt.figure(figsize=(7, 5))
        plotted = False
        for (stage, ds, pipeline), values in sorted(grouped.items()):
            if ds != dataset:
                continue
            values = sorted(values)
            if not values:
                continue
            xs, ys = zip(*values)
            plt.plot(xs, ys, label=f"{stage}/{pipeline}")
            plotted = True
        if plotted:
            plt.xscale("symlog", linthresh=10)
            plt.xlabel("Horizontal error (m)")
            plt.ylabel("CDF")
            plt.title(f"CDF: {dataset}")
            plt.grid(True, which="both", alpha=0.3)
            plt.legend(fontsize=7)
            plt.tight_layout()
            plt.savefig(output_dir / f"cdf_{dataset}.png", dpi=160)
        plt.close()


def parse_thresholds(text: str) -> List[float]:
    vals = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        val = _safe_float(part)
        if val is None or val < 0:
            raise ValueError(f"invalid threshold: {part}")
        vals.append(val)
    return vals or list(DEFAULT_THRESHOLDS_M)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", action="append", required=True, help="Benchmark root. May be passed multiple times.")
    parser.add_argument("--output-dir", required=True, help="Directory for exported CSV/Markdown/optional PNG artifacts.")
    parser.add_argument("--thresholds", default=",".join(str(v) for v in DEFAULT_THRESHOLDS_M), help="Comma-separated CDF thresholds in meters.")
    parser.add_argument("--all-tiers", action="store_true", help="Include diagnostic/native/experimental rows in addition to official-like strict/wrapper rows.")
    parser.add_argument("--no-plots", action="store_true", help="Skip optional matplotlib PNG output.")
    args = parser.parse_args(argv)

    roots = [Path(p) for p in args.root]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    thresholds = parse_thresholds(args.thresholds)
    official_only = not args.all_tiers

    runs = discover_runs(roots)
    cdf_frame_rows, frame_failure_rows = collect_prediction_rows(runs, official_only=official_only)
    cdf_points = build_cdf_points(cdf_frame_rows)
    threshold_rows = build_threshold_summary(cdf_frame_rows, thresholds)
    failure_rows = aggregate_failures(frame_failure_rows, runs, official_only=official_only)

    write_csv(
        output_dir / "cdf_points.csv",
        cdf_points,
        ["stage", "dataset", "pipeline", "tier", "official_like", "rank", "n", "position_error_m", "cdf"],
    )
    write_csv(
        output_dir / "cdf_thresholds.csv",
        threshold_rows,
        ["stage", "dataset", "pipeline", "tier", "official_like", "n", "mean_m", "median_m", "p75_m", "p90_m", "p95_m", "threshold_m", "accuracy_at_threshold"],
    )
    write_csv(
        output_dir / "failure_decomposition.csv",
        failure_rows,
        ["stage", "dataset", "pipeline", "run_id", "tier", "official_like", "failure_reason", "count"],
    )
    write_summary_md(output_dir / "cdf_summary.md", threshold_rows, failure_rows, roots, official_only)
    # Keep a separate file for readers who only want failure tables.
    write_summary_md(output_dir / "failure_decomposition.md", [], failure_rows, roots, official_only)
    if not args.no_plots:
        maybe_write_png(cdf_points, output_dir)

    manifest = {
        "roots": [str(r) for r in roots],
        "output_dir": str(output_dir),
        "official_only": official_only,
        "num_runs_discovered": len(runs),
        "num_cdf_frames": len(cdf_frame_rows),
        "num_cdf_points": len(cdf_points),
        "num_failure_rows": len(failure_rows),
        "thresholds_m": thresholds,
    }
    (output_dir / "export_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
