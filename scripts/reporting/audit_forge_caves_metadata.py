#!/usr/bin/env python3
"""Audit FORGE/CAVES metadata before promoting diagnostic runs."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

from evaluation.metrics import NpEncoder


REQUIRED_SUMMARY_FLAGS = {
    "benchmark_family": "forge",
    "adversarial_mode": "wrong_reference",
}


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


def _has_finite_field(row: Dict[str, Any], field: str) -> bool:
    value = _safe_float(row.get(field))
    return value == value


def discover_files(roots: Sequence[Path], name: str) -> List[Path]:
    out: List[Path] = []
    for root in roots:
        if root.is_file() and root.name == name:
            out.append(root)
        elif root.exists():
            out.extend(root.rglob(name))
    return sorted(set(out))


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _row_scope_from_path(path: Path) -> Dict[str, str]:
    parts = path.parts
    for idx, part in enumerate(parts):
        if part in {"avl", "vpair", "uav_visloc"} and idx + 2 < len(parts):
            return {
                "dataset": part,
                "condition": parts[idx + 1],
                "pipeline": parts[idx + 2],
            }
    return {"dataset": "unknown", "condition": "unknown", "pipeline": "unknown"}


def _read_prediction_rows(paths: Sequence[Path]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for path in paths:
        scope = _row_scope_from_path(path)
        with path.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                row = dict(row)
                row.setdefault("dataset", row.get("artifact_dataset") or scope["dataset"])
                row.setdefault("pipeline", row.get("artifact_pipeline") or scope["pipeline"])
                row.setdefault("adversarial_condition", scope["condition"])
                row["artifact_predictions_csv"] = str(path)
                rows.append(row)
    return rows


def _summary_errors(path: Path, summary: Dict[str, Any]) -> List[str]:
    rel = str(path)
    errors: List[str] = []
    for key, expected in REQUIRED_SUMMARY_FLAGS.items():
        if summary.get(key) != expected:
            errors.append(f"{rel}: {key}={summary.get(key)!r} expected {expected!r}")
    if summary.get("adversarial_diagnostic") is not True:
        errors.append(f"{rel}: adversarial_diagnostic is not true")
    if summary.get("official_metric_eligible") is not False:
        errors.append(f"{rel}: official_metric_eligible should be false")
    if summary.get("run_integrity_for_official_metrics") is not False:
        errors.append(f"{rel}: run_integrity_for_official_metrics should be false for FORGE")
    if int(summary.get("retrieval_top_k") or 5) > 5:
        errors.append(f"{rel}: retrieval_top_k={summary.get('retrieval_top_k')} > 5")
    if int(summary.get("frame_exceptions") or 0) > 0:
        errors.append(f"{rel}: frame_exceptions={summary.get('frame_exceptions')}")
    return errors


def audit_roots(roots: Sequence[Path]) -> Dict[str, Any]:
    summary_paths = discover_files(roots, "paper_metrics_summary.json")
    prediction_paths = discover_files(roots, "predictions.csv")
    errors: List[str] = []
    warnings: List[str] = []

    if not summary_paths:
        errors.append("no paper_metrics_summary.json files found")
    if not prediction_paths:
        errors.append("no predictions.csv files found")

    summaries = []
    for path in summary_paths:
        try:
            summary = _read_json(path)
        except Exception as exc:
            errors.append(f"{path}: failed to read JSON: {type(exc).__name__}: {exc}")
            continue
        summaries.append({"path": str(path), "summary": summary})
        errors.extend(_summary_errors(path, summary))

    rows = _read_prediction_rows(prediction_paths)
    by_scope: Counter[tuple[str, str, str]] = Counter()
    seen_within_run: set[tuple[str, str]] = set()
    duplicate_within_run = 0
    missing_identity = 0
    n_geometry_anchor_present = 0
    n_adversarial_offset_present = 0
    n_rows_with_only_adversarial_offset = 0
    n_rows_with_both_geometry_and_adversarial = 0
    n_rows_with_neither = 0

    for row in rows:
        dataset = str(row.get("dataset") or row.get("artifact_dataset") or "unknown").strip() or "unknown"
        pipeline = str(row.get("pipeline") or row.get("artifact_pipeline") or "unknown").strip() or "unknown"
        condition = str(row.get("adversarial_condition") or "unknown").strip() or "unknown"
        by_scope[(dataset, pipeline, condition)] += 1

        image_name = str(row.get("image_name") or "").strip()
        if not image_name:
            missing_identity += 1
        identity_key = (str(row.get("artifact_predictions_csv") or ""), image_name)
        if image_name and identity_key in seen_within_run:
            duplicate_within_run += 1
        if image_name:
            seen_within_run.add(identity_key)

        has_geometry = _has_finite_field(row, "geometry_anchor_offset_m")
        has_adversarial = _has_finite_field(row, "adversarial_offset_m")
        n_geometry_anchor_present += has_geometry
        n_adversarial_offset_present += has_adversarial
        n_rows_with_only_adversarial_offset += (not has_geometry and has_adversarial)
        n_rows_with_both_geometry_and_adversarial += (has_geometry and has_adversarial)
        n_rows_with_neither += (not has_geometry and not has_adversarial)

    n_rows = len(rows)
    n_geometry_anchor_missing = n_rows - n_geometry_anchor_present
    if missing_identity:
        errors.append(f"missing image_name identity rows: {missing_identity}")
    if duplicate_within_run:
        errors.append(f"duplicate image_name within prediction run: {duplicate_within_run}")
    if n_rows_with_only_adversarial_offset:
        errors.append(f"rows with only adversarial_offset_m metadata: {n_rows_with_only_adversarial_offset}")
    if n_rows and n_geometry_anchor_missing:
        warnings.append(f"rows missing geometry_anchor_offset_m: {n_geometry_anchor_missing}/{n_rows}")

    status = "PASS"
    if errors:
        status = "FAIL"
    elif warnings:
        status = "WARN"

    rows_by_scope = [
        {
            "dataset": dataset,
            "pipeline": pipeline,
            "condition": condition,
            "n_rows": count,
        }
        for (dataset, pipeline, condition), count in sorted(by_scope.items())
    ]

    return {
        "metadata_audit_status": status,
        "errors": errors,
        "warnings": warnings,
        "num_paper_metrics_summaries": len(summary_paths),
        "num_prediction_csvs": len(prediction_paths),
        "num_prediction_rows": n_rows,
        "rows_by_scope": rows_by_scope,
        "duplicate_image_name_within_run": duplicate_within_run,
        "missing_identity_rows": missing_identity,
        "n_geometry_anchor_present": n_geometry_anchor_present,
        "n_geometry_anchor_missing": n_geometry_anchor_missing,
        "geometry_anchor_missing_rate": n_geometry_anchor_missing / n_rows if n_rows else float("nan"),
        "n_adversarial_offset_present": n_adversarial_offset_present,
        "n_rows_with_only_adversarial_offset": n_rows_with_only_adversarial_offset,
        "n_rows_with_both_geometry_and_adversarial": n_rows_with_both_geometry_and_adversarial,
        "n_rows_with_neither": n_rows_with_neither,
        "summary_paths": [str(path) for path in summary_paths],
        "prediction_paths": [str(path) for path in prediction_paths],
    }


def format_markdown(audit: Dict[str, Any]) -> str:
    lines = [
        "# FORGE/CAVES Metadata Audit",
        "",
        f"- status: `{audit.get('metadata_audit_status')}`",
        f"- prediction CSVs: `{audit.get('num_prediction_csvs')}`",
        f"- prediction rows: `{audit.get('num_prediction_rows')}`",
        f"- paper metric summaries: `{audit.get('num_paper_metrics_summaries')}`",
        f"- geometry anchors present: `{audit.get('n_geometry_anchor_present')}`",
        f"- geometry anchors missing: `{audit.get('n_geometry_anchor_missing')}`",
        f"- adversarial-only offset rows: `{audit.get('n_rows_with_only_adversarial_offset')}`",
        "",
    ]
    if audit.get("errors"):
        lines.extend(["## Errors", ""])
        lines.extend(f"- {error}" for error in audit["errors"])
        lines.append("")
    if audit.get("warnings"):
        lines.extend(["## Warnings", ""])
        lines.extend(f"- {warning}" for warning in audit["warnings"])
        lines.append("")
    lines.extend([
        "## Rows by scope",
        "",
        "| dataset | pipeline | condition | rows |",
        "|---|---|---|---:|",
    ])
    for row in audit.get("rows_by_scope", []):
        lines.append(f"| {row.get('dataset')} | {row.get('pipeline')} | {row.get('condition')} | {row.get('n_rows')} |")
    lines.append("")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", action="append", required=True, help="Result root to audit; repeat for multiple roots")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--strict", action="store_true", help="Exit non-zero when audit status is FAIL")
    args = parser.parse_args(argv)

    audit = audit_roots([Path(root) for root in args.root])
    output_json = Path(args.output_json)
    output_md = Path(args.output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(audit, indent=2, cls=NpEncoder), encoding="utf-8")
    output_md.write_text(format_markdown(audit), encoding="utf-8")
    print(json.dumps({"metadata_audit_status": audit["metadata_audit_status"], "output_json": str(output_json)}, cls=NpEncoder))
    if args.strict and audit["metadata_audit_status"] == "FAIL":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
