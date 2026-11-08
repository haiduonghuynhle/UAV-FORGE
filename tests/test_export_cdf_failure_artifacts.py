import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "reporting" / "export_cdf_failure_artifacts.py"


class ExportCdfFailureArtifactsTest(unittest.TestCase):
    def _make_run(self, root: Path) -> Path:
        run = root / "official_10percent" / "avl" / "ours_test" / "run1"
        run.mkdir(parents=True)
        (run / "paper_metrics_summary.json").write_text(json.dumps({
            "dataset": "avl",
            "pipeline": "ours_test",
            "comparison_tier": "strict_e2e",
            "run_integrity_for_official_metrics": True,
            "retrieval_protocol_violation_count": 0,
            "retrieval_top_k": 5,
            "num_pose_rejected_by_final_gate": 1,
        }))
        with (run / "predictions.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "image_name", "gt_lat", "gt_lon", "pred_lat", "pred_lon",
                "accepted_for_metrics", "pose_success", "failure_reason",
            ])
            writer.writeheader()
            writer.writerow({
                "image_name": "a.jpg",
                "gt_lat": "10.0", "gt_lon": "106.0",
                "pred_lat": "10.0", "pred_lon": "106.0",
                "accepted_for_metrics": "true", "pose_success": "true",
                "failure_reason": "",
            })
            writer.writerow({
                "image_name": "b.jpg",
                "gt_lat": "10.0", "gt_lon": "106.0",
                "pred_lat": "", "pred_lon": "",
                "accepted_for_metrics": "false", "pose_success": "false",
                "failure_reason": "geometry_inlier_gate_reject",
            })
        return run

    def test_exporter_writes_cdf_and_failure_outputs(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "root"
            out = Path(td) / "out"
            self._make_run(root)

            subprocess.run(
                [sys.executable, str(SCRIPT), "--root", str(root), "--output-dir", str(out), "--no-plots"],
                check=True,
                cwd=ROOT,
            )

            for name in [
                "cdf_points.csv",
                "cdf_thresholds.csv",
                "cdf_summary.md",
                "failure_decomposition.csv",
                "failure_decomposition.md",
                "export_manifest.json",
            ]:
                self.assertTrue((out / name).exists(), name)
            with (out / "cdf_points.csv").open() as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["dataset"], "avl")
            with (out / "failure_decomposition.csv").open() as f:
                failures = list(csv.DictReader(f))
            reasons = {r["failure_reason"] for r in failures}
            self.assertIn("geometry_inlier_gate_reject", reasons)
            self.assertIn("final_pose_gate_reject", reasons)

    def test_official_only_filter_excludes_experimental_by_default(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "root"
            out = Path(td) / "out"
            run = root / "dsm_diagnostic_avl" / "avl" / "ours_dsm" / "run1"
            run.mkdir(parents=True)
            (run / "paper_metrics_summary.json").write_text(json.dumps({
                "dataset": "avl",
                "pipeline": "ours_dsm",
                "comparison_tier": "experimental_dsm_6dof",
                "run_integrity_for_official_metrics": False,
            }))
            with (run / "predictions.csv").open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["position_error_m", "accepted_for_metrics"])
                writer.writeheader()
                writer.writerow({"position_error_m": "1.0", "accepted_for_metrics": "true"})

            subprocess.run(
                [sys.executable, str(SCRIPT), "--root", str(root), "--output-dir", str(out), "--no-plots"],
                check=True,
                cwd=ROOT,
            )
            with (out / "cdf_points.csv").open() as f:
                self.assertEqual(len(list(csv.DictReader(f))), 0)

            out_all = Path(td) / "out_all"
            subprocess.run(
                [sys.executable, str(SCRIPT), "--root", str(root), "--output-dir", str(out_all), "--all-tiers", "--no-plots"],
                check=True,
                cwd=ROOT,
            )
            with (out_all / "cdf_points.csv").open() as f:
                self.assertEqual(len(list(csv.DictReader(f))), 1)

    def test_forge_rows_are_diagnostic_even_if_tier_looks_strict(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "root"
            run = root / "forge_smoke" / "avl" / "ours_camp" / "run1"
            run.mkdir(parents=True)
            (run / "paper_metrics_summary.json").write_text(json.dumps({
                "dataset": "avl",
                "pipeline": "ours_camp",
                "comparison_tier": "strict_e2e",
                "benchmark_family": "forge",
                "adversarial_mode": "wrong_reference",
                "adversarial_diagnostic": True,
                "official_metric_eligible": False,
                "run_integrity_for_official_metrics": False,
                "num_unsafe_accept": 1,
                "num_adversarial_fail_closed": 2,
            }))
            with (run / "predictions.csv").open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["position_error_m", "accepted_for_metrics", "failure_reason"])
                writer.writeheader()
                writer.writerow({"position_error_m": "5.0", "accepted_for_metrics": "true", "failure_reason": ""})

            out = Path(td) / "out"
            subprocess.run(
                [sys.executable, str(SCRIPT), "--root", str(root), "--output-dir", str(out), "--no-plots"],
                check=True,
                cwd=ROOT,
            )
            with (out / "cdf_points.csv").open() as f:
                self.assertEqual(len(list(csv.DictReader(f))), 0)

            out_all = Path(td) / "out_all"
            subprocess.run(
                [sys.executable, str(SCRIPT), "--root", str(root), "--output-dir", str(out_all), "--all-tiers", "--no-plots"],
                check=True,
                cwd=ROOT,
            )
            with (out_all / "cdf_points.csv").open() as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["official_like"], "False")
            with (out_all / "failure_decomposition.csv").open() as f:
                reasons = {r["failure_reason"] for r in csv.DictReader(f)}
            self.assertIn("unsafe_accept", reasons)
            self.assertIn("adversarial_fail_closed", reasons)


if __name__ == "__main__":
    unittest.main()
