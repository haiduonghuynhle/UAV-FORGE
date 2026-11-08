#!/usr/bin/env python3
"""Post-hoc CAVES analysis on natural top-1 retrieval misses from clean official runs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]

from uav_geolocalization.benchmark.caves import (
    rows_to_labels_scores,
    summarize_caves_metrics,
)

# Reuse artifact readers from the CAVES calibration reporter.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_caves_calibration import (  # noqa: E402
    DATASET_RETRIEVAL_THRESHOLDS_M,
    discover_prediction_csvs,
    read_prediction_rows,
)

DEFAULT_PIPELINES = (
    "ours_camp_cascade_6dof_e2e",
    "ours_ultravpr_cascade_6dof_e2e",
)
FORGE_INJECTED_HELDOUT_AUROC = 0.744
VPAIR_GO_NO_GO_MIN_FRAMES = 30


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return False


def _safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        if value is None or value == "":
            return default
        out = float(value)
        return out if out == out and abs(out) != float("inf") else default
    except (TypeError, ValueError):
        return default


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius_m = 6_371_000.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    return 2.0 * radius_m * math.asin(min(1.0, math.sqrt(a)))


def is_clean_row(row: Dict[str, Any]) -> bool:
    for key in ("adversarial_mode", "is_wrong_reference", "is_adversarial"):
        value = str(row.get(key, "")).strip().lower()
        if value in {"1", "true", "yes", "wrong_reference"}:
            return False
        if key == "adversarial_mode" and value not in {"", "none", "clean"}:
            return False
    return True


def dataset_threshold_m(row: Dict[str, Any]) -> float:
    dataset = str(row.get("dataset") or row.get("artifact_dataset") or "").strip().lower()
    return float(DATASET_RETRIEVAL_THRESHOLDS_M.get(dataset, 20.0))


def natural_miss_threshold_source(row: Dict[str, Any]) -> Dict[str, Any]:
    threshold_m = dataset_threshold_m(row)
    hit_col = f"raw_top1_anchor_hit_at_{int(threshold_m)}m"
    if hit_col in row and str(row.get(hit_col, "")).strip() != "":
        return {"threshold_m": threshold_m, "source": hit_col}
    return {"threshold_m": threshold_m, "source": "raw_retrieval_anchor_error_m"}


def natural_top1_miss(row: Dict[str, Any]) -> bool:
    source = natural_miss_threshold_source(row)
    threshold_m = float(source["threshold_m"])
    if str(source["source"]).startswith("raw_top1_anchor_hit_at_"):
        return not _truthy(row.get(str(source["source"])))
    raw_error_m = _safe_float(row.get("raw_retrieval_anchor_error_m"))
    return raw_error_m == raw_error_m and raw_error_m > threshold_m


def enrich_natural_outcome_labels(row: Dict[str, Any]) -> None:
    threshold_m = dataset_threshold_m(row)
    post_error_m = _safe_float(row.get("post_fgo_error_m"))
    if post_error_m != post_error_m:
        glat = _safe_float(row.get("gt_lat"))
        glon = _safe_float(row.get("gt_lon"))
        plat = _safe_float(row.get("pred_lat"))
        plon = _safe_float(row.get("pred_lon"))
        if all(value == value for value in (glat, glon, plat, plon)):
            post_error_m = haversine_m(glat, glon, plat, plon)
            row["post_fgo_error_m"] = post_error_m
    bad_pose = bool(post_error_m == post_error_m and post_error_m > threshold_m)
    row["bad_pose_label"] = bad_pose
    accepted = _truthy(row.get("accepted_for_metrics", row.get("success", False)))
    row["unsafe_accept"] = bool(accepted and bad_pose)
    row["natural_top1_miss"] = natural_top1_miss(row)


def path_matches_pipelines(path: Path, pipelines: Sequence[str]) -> bool:
    if not pipelines:
        return True
    path_text = str(path)
    return any(pipeline in path_text for pipeline in pipelines)


def selected_prediction_csvs(roots: Sequence[Path], pipelines: Sequence[str]) -> List[Path]:
    return [
        path for path in discover_prediction_csvs(roots)
        if path_matches_pipelines(path, pipelines)
    ]


def load_clean_rows(
    roots: Sequence[Path],
    *,
    pipelines: Sequence[str],
    reject_threshold: float,
) -> List[Dict[str, Any]]:
    paths = selected_prediction_csvs(roots, pipelines)
    rows = read_prediction_rows(paths, reject_threshold=reject_threshold)
    out: List[Dict[str, Any]] = []
    for row in rows:
        if row.get("artifact_read_error"):
            continue
        if not is_clean_row(row):
            continue
        enrich_natural_outcome_labels(row)
        out.append(row)
    return out


def subset_metrics(rows: Sequence[Dict[str, Any]], *, feature_set: str = "full_caves", threshold: float = 0.5) -> Dict[str, Any]:
    labels, scores = rows_to_labels_scores(rows, feature_set=feature_set)
    metrics = summarize_caves_metrics(labels, scores, threshold=threshold)
    return {
        "feature_set": feature_set,
        "n_total": len(rows),
        "n_scored": len(labels),
        "n_bad_pose": int(sum(labels)),
        "base_bad_rate": (float(sum(labels)) / len(labels)) if labels else float("nan"),
        "n_unsafe_accept": int(sum(1 for row in rows if _truthy(row.get("unsafe_accept")))),
        "unsafe_accept_rate": (
            float(sum(1 for row in rows if _truthy(row.get("unsafe_accept")))) / len(rows)
            if rows
            else float("nan")
        ),
        **{key: metrics.get(key, float("nan")) for key in ("auroc", "auprc", "coverage", "accepted_bad_rate")},
    }


def comparison_verdict(natural_auroc: float, injected_auroc: float = FORGE_INJECTED_HELDOUT_AUROC) -> str:
    if natural_auroc != natural_auroc:
        return "insufficient_labeled_support"
    delta = natural_auroc - injected_auroc
    if abs(delta) <= 0.03:
        return "comparable_to_injected"
    if delta > 0.03:
        return "natural_easier_than_injected"
    return "natural_harder_than_injected"


def build_summary_rows(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    natural_rows = [row for row in rows if _truthy(row.get("natural_top1_miss"))]
    summary_rows: List[Dict[str, Any]] = []
    for label, subset in (
        ("all_clean", rows),
        ("all_natural_top1_miss", natural_rows),
    ):
        metrics = subset_metrics(subset)
        metrics["subset"] = label
        metrics["verdict"] = ""
        summary_rows.append(metrics)
    for dataset in sorted({str(row.get("dataset", "")).strip().lower() for row in natural_rows if row.get("dataset")}):
        subset = [row for row in natural_rows if str(row.get("dataset", "")).strip().lower() == dataset]
        metrics = subset_metrics(subset)
        metrics["subset"] = f"natural_top1_miss_{dataset}"
        metrics["verdict"] = comparison_verdict(float(metrics.get("auroc", float("nan"))))
        summary_rows.append(metrics)
    return summary_rows


def write_csv(path: Path, rows: Sequence[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _canonical_row(row: Dict[str, Any]) -> Dict[str, Any]:
    keys = [
        "artifact_predictions_csv",
        "dataset",
        "pipeline",
        "image_name",
        "raw_retrieval_anchor_error_m",
        "post_fgo_error_m",
        "accepted_for_metrics",
        "success",
        "bad_pose_label",
        "unsafe_accept",
        "natural_top1_miss",
    ]
    return {key: str(row.get(key, "")) for key in keys}


def input_fingerprint(rows: Sequence[Dict[str, Any]], paths: Sequence[Path]) -> Dict[str, Any]:
    canonical_rows = sorted(
        (_canonical_row(row) for row in rows),
        key=lambda row: json.dumps(row, sort_keys=True, separators=(",", ":")),
    )
    payload = json.dumps(canonical_rows, sort_keys=True, separators=(",", ":"))
    return {
        "algorithm": "sha256_canonical_selected_clean_rows_v1",
        "sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
        "selected_clean_row_count": len(rows),
        "selected_prediction_csv_count": len(paths),
        "selected_prediction_csvs": [str(path) for path in sorted(paths)],
    }


def threshold_sources_by_dataset(rows: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        dataset = str(row.get("dataset") or row.get("artifact_dataset") or "unknown").strip().lower() or "unknown"
        source = natural_miss_threshold_source(row)
        entry = out.setdefault(dataset, {"threshold_m": source["threshold_m"], "sources": []})
        if source["source"] not in entry["sources"]:
            entry["sources"].append(source["source"])
    for entry in out.values():
        entry["sources"] = sorted(entry["sources"])
    return out


def build_metadata(
    *,
    roots: Sequence[Path],
    pipelines: Sequence[str],
    rows: Sequence[Dict[str, Any]],
    paths: Sequence[Path],
    vpair_count: int,
) -> Dict[str, Any]:
    return {
        "source_roots": [str(root) for root in roots],
        "pipeline_filter": list(pipelines),
        "vpair_go_no_go_count": vpair_count,
        "vpair_go_no_go_min_frames": VPAIR_GO_NO_GO_MIN_FRAMES,
        "comparison_reference_auroc": FORGE_INJECTED_HELDOUT_AUROC,
        "threshold_sources": threshold_sources_by_dataset(rows),
        "input_fingerprint": input_fingerprint(rows, paths),
    }


def render_markdown(
    *,
    roots: Sequence[Path],
    pipelines: Sequence[str],
    summary_rows: Sequence[Dict[str, Any]],
    vpair_count: int,
) -> str:
    vpair_row = next((row for row in summary_rows if row.get("subset") == "natural_top1_miss_vpair"), {})
    lines = [
        "# Natural Failure CAVES Summary",
        "",
        "> Diagnostic post-hoc analysis on clean official predictions with organic top-1 retrieval misses.",
        "> Not merged into official clean tables or FORGE injected metrics.",
        "",
        "## Inputs",
        "",
        f"- Roots: `{', '.join(str(root) for root in roots)}`",
        f"- Pipelines: `{', '.join(pipelines) if pipelines else 'all'}`",
        f"- FORGE injected held-out AUROC reference: **{FORGE_INJECTED_HELDOUT_AUROC:.3f}** (`full_caves` raw)",
        "",
        "## Go/no-go",
        "",
        f"- VPAIR natural top-1 miss frames: **{vpair_count}** (gate: n ≥ {VPAIR_GO_NO_GO_MIN_FRAMES})",
        f"- Status: **{'GO' if vpair_count >= VPAIR_GO_NO_GO_MIN_FRAMES else 'COUNTS_ONLY'}**",
        "",
        "## Metrics",
        "",
        "| subset | n_total | n_bad | base_bad_rate | unsafe_accept_rate | AUROC | verdict |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in summary_rows:
        lines.append(
            "| {subset} | {n_total} | {n_bad_pose} | {base_bad_rate:.3f} | {unsafe_accept_rate:.3f} | {auroc:.3f} | {verdict} |".format(
                subset=row.get("subset", ""),
                n_total=row.get("n_total", 0),
                n_bad_pose=row.get("n_bad_pose", 0),
                base_bad_rate=float(row.get("base_bad_rate", float("nan"))),
                unsafe_accept_rate=float(row.get("unsafe_accept_rate", float("nan"))),
                auroc=float(row.get("auroc", float("nan"))),
                verdict=row.get("verdict", ""),
            )
        )
    lines.extend(
        [
            "",
            "## Paper wording",
            "",
        ]
    )
    verdict = str(vpair_row.get("verdict", ""))
    vpair_auroc = float(vpair_row.get("auroc", float("nan")))
    if vpair_count < VPAIR_GO_NO_GO_MIN_FRAMES:
        lines.append(
            "- Report counts only; do not claim AUROC on natural failures until n≥30 on VPAIR."
        )
    elif verdict == "natural_easier_than_injected":
        lines.append(
            f"- On VPAIR clean official runs, organic top-1 misses (n={vpair_count}) yield "
            f"AUROC **{vpair_auroc:.3f}**, higher than the FORGE injected held-out AUROC "
            f"({FORGE_INJECTED_HELDOUT_AUROC:.3f}) in this evaluation; CAVES features remain informative "
            "on this subset beyond deterministic injection."
        )
    elif verdict == "natural_harder_than_injected":
        lines.append(
            f"- On VPAIR, natural top-1 misses (n={vpair_count}) yield AUROC **{vpair_auroc:.3f}**, "
            f"below FORGE injected held-out ({FORGE_INJECTED_HELDOUT_AUROC:.3f}); injected stress may upper-bound "
            "separability relative to organically occurring retrieval slips."
        )
    else:
        lines.append(
            f"- VPAIR natural top-1 misses (n={vpair_count}) show AUROC **{vpair_auroc:.3f}**, "
            f"comparable to FORGE injected held-out ({FORGE_INJECTED_HELDOUT_AUROC:.3f}) within a ±0.03 band."
        )
    lines.append(
        "- The 35.8% FORGE unsafe-accept rate remains a controlled worst-case benchmark statistic, "
        "not a deployment-frequency claim."
    )
    lines.append("")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--roots",
        nargs="+",
        type=Path,
        required=True,
        help="Artifact roots containing clean predictions.csv files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "Report" / "natural_failure_caves",
        help="Directory for CSV + markdown outputs",
    )
    parser.add_argument(
        "--skip-report-copy",
        action="store_true",
        help="Do not also write Report/natural_failure_caves_summary.md (useful for tests).",
    )
    parser.add_argument(
        "--pipelines",
        nargs="*",
        default=list(DEFAULT_PIPELINES),
        help="Restrict to run paths containing these pipeline names (default: ours camp + ultravpr)",
    )
    parser.add_argument("--reject-threshold", type=float, default=0.5)
    args = parser.parse_args(argv)

    paths = selected_prediction_csvs(args.roots, args.pipelines)
    rows = load_clean_rows(args.roots, pipelines=args.pipelines, reject_threshold=args.reject_threshold)
    summary_rows = build_summary_rows(rows)
    vpair_count = sum(
        1 for row in rows
        if _truthy(row.get("natural_top1_miss")) and str(row.get("dataset", "")).strip().lower() == "vpair"
    )

    output_dir = args.output_dir
    fieldnames = [
        "subset",
        "feature_set",
        "n_total",
        "n_scored",
        "n_bad_pose",
        "base_bad_rate",
        "n_unsafe_accept",
        "unsafe_accept_rate",
        "auroc",
        "auprc",
        "coverage",
        "accepted_bad_rate",
        "verdict",
    ]
    write_csv(output_dir / "natural_failure_caves_summary.csv", summary_rows, fieldnames)
    metadata = build_metadata(
        roots=args.roots,
        pipelines=args.pipelines,
        rows=rows,
        paths=paths,
        vpair_count=vpair_count,
    )
    (output_dir / "natural_failure_caves_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    markdown = render_markdown(roots=args.roots, pipelines=args.pipelines, summary_rows=summary_rows, vpair_count=vpair_count)
    (output_dir / "natural_failure_caves_summary.md").write_text(markdown, encoding="utf-8")
    if not args.skip_report_copy:
        also_report = REPO_ROOT / "Report" / "natural_failure_caves_summary.md"
        also_report.parent.mkdir(parents=True, exist_ok=True)
        also_report.write_text(markdown, encoding="utf-8")
    print(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
