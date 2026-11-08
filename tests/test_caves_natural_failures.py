"""Tests for natural-failure CAVES reporter."""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from scripts.reporting import evaluate_caves_natural_failures as natural


class CavesNaturalFailuresTest(unittest.TestCase):
    def test_natural_top1_miss_uses_raw_anchor_error_when_hit_column_missing(self):
        row = {"dataset": "vpair", "raw_retrieval_anchor_error_m": "35.0"}
        self.assertTrue(natural.natural_top1_miss(row))
        self.assertEqual(natural.natural_miss_threshold_source(row)["source"], "raw_retrieval_anchor_error_m")

    def test_natural_top1_miss_uses_threshold_hit_flag_when_present(self):
        row = {
            "dataset": "vpair",
            "raw_top1_anchor_hit_at_20m": "0",
            "raw_retrieval_anchor_error_m": "5.0",
        }
        self.assertTrue(natural.natural_top1_miss(row))
        source = natural.natural_miss_threshold_source(row)
        self.assertEqual(source["source"], "raw_top1_anchor_hit_at_20m")
        self.assertEqual(source["threshold_m"], 20.0)

    def test_natural_top1_miss_false_when_raw_anchor_within_threshold(self):
        row = {"dataset": "vpair", "raw_retrieval_anchor_error_m": "12.0"}
        self.assertFalse(natural.natural_top1_miss(row))

    def test_is_clean_row_rejects_wrong_reference_rows(self):
        row = {"adversarial_mode": "wrong_reference", "is_wrong_reference": "true"}
        self.assertFalse(natural.is_clean_row(row))

    def test_enrich_labels_marks_unsafe_accept(self):
        row = {
            "dataset": "vpair",
            "raw_retrieval_anchor_error_m": "40.0",
            "accepted_for_metrics": "true",
            "gt_lat": "10.0",
            "gt_lon": "106.0",
            "pred_lat": "10.001",
            "pred_lon": "106.5",
            "num_matches": "50",
            "num_inliers": "20",
            "inlier_ratio": "0.4",
            "reproj_residual_px": "4.0",
            "shape_confidence": "0.8",
            "retrieval_score_margin": "0.1",
            "retrieval_rank": "1",
        }
        natural.enrich_natural_outcome_labels(row)
        self.assertTrue(row["natural_top1_miss"])
        self.assertTrue(row["bad_pose_label"])
        self.assertTrue(row["unsafe_accept"])

    def test_input_fingerprint_handles_multiple_rows_deterministically(self):
        rows = [
            {
                "dataset": "vpair",
                "pipeline": "ours_ultravpr_cascade_6dof_e2e",
                "image_name": "b.jpg",
                "raw_retrieval_anchor_error_m": "25.0",
                "natural_top1_miss": True,
            },
            {
                "dataset": "vpair",
                "pipeline": "ours_camp_cascade_6dof_e2e",
                "image_name": "a.jpg",
                "raw_retrieval_anchor_error_m": "5.0",
                "natural_top1_miss": False,
            },
        ]
        paths = [Path("/tmp/b/predictions.csv"), Path("/tmp/a/predictions.csv")]

        first = natural.input_fingerprint(rows, paths)
        second = natural.input_fingerprint(list(reversed(rows)), list(reversed(paths)))

        self.assertEqual(len(first["sha256"]), 64)
        self.assertEqual(first["selected_clean_row_count"], 2)
        self.assertEqual(first["selected_prediction_csv_count"], 2)
        self.assertEqual(first["sha256"], second["sha256"])

    def test_main_writes_summary_markdown(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "official_10percent" / "vpair" / "ours_camp_cascade_6dof_e2e" / "run"
            root.mkdir(parents=True)
            csv_path = root / "predictions.csv"
            with csv_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "image_name",
                        "gt_lat",
                        "gt_lon",
                        "pred_lat",
                        "pred_lon",
                        "accepted_for_metrics",
                        "raw_retrieval_anchor_error_m",
                        "num_matches",
                        "num_inliers",
                        "inlier_ratio",
                        "reproj_residual_px",
                        "shape_confidence",
                        "retrieval_score_margin",
                        "retrieval_rank",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "image_name": "a.jpg",
                        "gt_lat": "10.0",
                        "gt_lon": "106.0",
                        "pred_lat": "10.001",
                        "pred_lon": "106.5",
                        "accepted_for_metrics": "true",
                        "raw_retrieval_anchor_error_m": "40.0",
                        "num_matches": "50",
                        "num_inliers": "20",
                        "inlier_ratio": "0.4",
                        "reproj_residual_px": "4.0",
                        "shape_confidence": "0.8",
                        "retrieval_score_margin": "0.1",
                        "retrieval_rank": "1",
                    }
                )
                writer.writerow(
                    {
                        "image_name": "b.jpg",
                        "gt_lat": "10.0",
                        "gt_lon": "106.0",
                        "pred_lat": "10.0001",
                        "pred_lon": "106.0001",
                        "accepted_for_metrics": "true",
                        "raw_retrieval_anchor_error_m": "5.0",
                        "num_matches": "80",
                        "num_inliers": "60",
                        "inlier_ratio": "0.75",
                        "reproj_residual_px": "2.0",
                        "shape_confidence": "0.9",
                        "retrieval_score_margin": "0.2",
                        "retrieval_rank": "1",
                    }
                )
            out_dir = Path(tmp) / "out"
            rc = natural.main([
                "--roots",
                str(Path(tmp) / "official_10percent"),
                "--output-dir",
                str(out_dir),
                "--skip-report-copy",
            ])
            self.assertEqual(rc, 0)
            self.assertTrue((out_dir / "natural_failure_caves_summary.csv").exists())
            metadata_path = out_dir / "natural_failure_caves_metadata.json"
            self.assertTrue(metadata_path.exists())
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            self.assertEqual(metadata["source_roots"], [str(Path(tmp) / "official_10percent")])
            self.assertEqual(metadata["pipeline_filter"], list(natural.DEFAULT_PIPELINES))
            self.assertEqual(metadata["vpair_go_no_go_count"], 1)
            self.assertEqual(metadata["comparison_reference_auroc"], natural.FORGE_INJECTED_HELDOUT_AUROC)
            self.assertEqual(metadata["input_fingerprint"]["selected_clean_row_count"], 2)
            self.assertEqual(metadata["input_fingerprint"]["selected_prediction_csv_count"], 1)
            self.assertEqual(metadata["threshold_sources"]["vpair"]["threshold_m"], 20.0)
            self.assertIn("raw_retrieval_anchor_error_m", metadata["threshold_sources"]["vpair"]["sources"])
            md = (out_dir / "natural_failure_caves_summary.md").read_text(encoding="utf-8")
            self.assertIn("natural_top1_miss_vpair", md)
            self.assertIn("Diagnostic post-hoc", md)
            self.assertIn("Not merged into official clean tables", md)
            self.assertIn("not a deployment-frequency claim", md)


if __name__ == "__main__":
    unittest.main()
