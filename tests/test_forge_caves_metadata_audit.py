import csv
import json
import tempfile
import unittest
from pathlib import Path

from scripts.reporting import audit_forge_caves_metadata as audit


class ForgeCavesMetadataAuditTest(unittest.TestCase):
    def _write_run(self, root: Path, *, summary_overrides=None, rows=None) -> Path:
        run = root / "main_n100" / "avl" / "wrong_near" / "ours_camp_cascade_6dof_e2e" / "avl" / "ours_camp_cascade_6dof_e2e" / "20260612_000000"
        run.mkdir(parents=True)
        summary = {
            "benchmark_family": "forge",
            "adversarial_mode": "wrong_reference",
            "adversarial_diagnostic": True,
            "official_metric_eligible": False,
            "run_integrity_for_official_metrics": False,
            "retrieval_top_k": 5,
            "frame_exceptions": 0,
        }
        if summary_overrides:
            summary.update(summary_overrides)
        (run / "paper_metrics_summary.json").write_text(json.dumps(summary), encoding="utf-8")
        rows = rows if rows is not None else [
            {
                "image_name": "q0",
                "dataset": "avl",
                "pipeline": "ours_camp_cascade_6dof_e2e",
                "adversarial_condition": "wrong_near",
                "geometry_anchor_offset_m": "30",
                "adversarial_offset_m": "",
            },
            {
                "image_name": "q1",
                "dataset": "avl",
                "pipeline": "ours_camp_cascade_6dof_e2e",
                "adversarial_condition": "wrong_near",
                "geometry_anchor_offset_m": "40",
                "adversarial_offset_m": "100",
            },
        ]
        with (run / "predictions.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "image_name",
                    "dataset",
                    "pipeline",
                    "adversarial_condition",
                    "geometry_anchor_offset_m",
                    "adversarial_offset_m",
                ],
            )
            writer.writeheader()
            writer.writerows(rows)
        return run

    def test_audit_passes_clean_forge_metadata_in_strict_mode(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_run(root)
            output_json = root / "audit.json"
            output_md = root / "audit.md"

            rc = audit.main([
                "--root", str(root),
                "--output-json", str(output_json),
                "--output-md", str(output_md),
                "--strict",
            ])

            self.assertEqual(rc, 0)
            payload = json.loads(output_json.read_text())
            self.assertEqual(payload["metadata_audit_status"], "PASS")
            self.assertEqual(payload["num_prediction_csvs"], 1)
            self.assertEqual(payload["num_prediction_rows"], 2)
            self.assertEqual(payload["n_rows_with_only_adversarial_offset"], 0)
            self.assertTrue(output_md.exists())

    def test_audit_warns_on_missing_geometry_anchor_and_strict_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_run(root, rows=[
                {
                    "image_name": "q0",
                    "dataset": "avl",
                    "pipeline": "ours_camp_cascade_6dof_e2e",
                    "adversarial_condition": "wrong_near",
                    "geometry_anchor_offset_m": "",
                    "adversarial_offset_m": "",
                },
            ])
            output_json = root / "audit.json"
            output_md = root / "audit.md"

            rc = audit.main([
                "--root", str(root),
                "--output-json", str(output_json),
                "--output-md", str(output_md),
            ])
            strict_rc = audit.main([
                "--root", str(root),
                "--output-json", str(output_json),
                "--output-md", str(output_md),
                "--strict",
            ])

            self.assertEqual(rc, 0)
            self.assertEqual(strict_rc, 0)
            payload = json.loads(output_json.read_text())
            self.assertEqual(payload["metadata_audit_status"], "WARN")
            self.assertIn("rows missing geometry_anchor_offset_m", payload["warnings"][0])

    def test_audit_fails_on_bad_summary_or_adversarial_only_offset(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_run(
                root,
                summary_overrides={"retrieval_top_k": 10},
                rows=[
                    {
                        "image_name": "q0",
                        "dataset": "avl",
                        "pipeline": "ours_camp_cascade_6dof_e2e",
                        "adversarial_condition": "wrong_near",
                        "geometry_anchor_offset_m": "",
                        "adversarial_offset_m": "100",
                    },
                ],
            )
            output_json = root / "audit.json"
            output_md = root / "audit.md"

            rc = audit.main([
                "--root", str(root),
                "--output-json", str(output_json),
                "--output-md", str(output_md),
                "--strict",
            ])

            self.assertEqual(rc, 1)
            payload = json.loads(output_json.read_text())
            self.assertEqual(payload["metadata_audit_status"], "FAIL")
            self.assertTrue(any("retrieval_top_k" in error for error in payload["errors"]))
            self.assertTrue(any("only adversarial_offset_m" in error for error in payload["errors"]))

    def test_predictions_export_strips_adversarial_only_offset_for_metadata_audit(self):
        from uav_geolocalization.cli import benchmark_runner_impl as br

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = root / "main_n100" / "avl" / "wrong_near" / "ours_camp_cascade_6dof_e2e" / "avl" / "ours_camp_cascade_6dof_e2e" / "20260612_000001"
            run.mkdir(parents=True)
            (run / "paper_metrics_summary.json").write_text(json.dumps({
                "benchmark_family": "forge",
                "adversarial_mode": "wrong_reference",
                "adversarial_diagnostic": True,
                "official_metric_eligible": False,
                "run_integrity_for_official_metrics": False,
                "retrieval_top_k": 5,
                "frame_exceptions": 0,
            }), encoding="utf-8")
            br.write_predictions_csv(run / "predictions.csv", [{
                "image_name": "q0",
                "success": False,
                "accepted_for_metrics": False,
                "geometry_anchor_offset_m": float("nan"),
                "adversarial_offset_m": 120.0,
                "adversarial_mode": "wrong_reference",
                "adversarial_condition": "wrong_near",
            }])
            output_json = root / "audit.json"
            output_md = root / "audit.md"
            rc = audit.main([
                "--root", str(root),
                "--output-json", str(output_json),
                "--output-md", str(output_md),
                "--strict",
            ])
            self.assertEqual(rc, 0)
            payload = json.loads(output_json.read_text())
            self.assertEqual(payload["metadata_audit_status"], "WARN")
            self.assertEqual(payload["n_rows_with_only_adversarial_offset"], 0)


class AVLImageNameUniquenessTest(unittest.TestCase):
    """Regression: AVL subdirectories share DJI filenames (e.g. DJI_0001.JPG).
    image_name must include the parent subdirectory to avoid identity collisions
    in CAVES hash splits."""

    def test_avl_image_name_includes_subdirectory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            metadata_dir = root / "metadata"
            metadata_dir.mkdir()

            sub_a = root / "UAV_image" / "TestRegion" / "SubdirA"
            sub_b = root / "UAV_image" / "TestRegion" / "SubdirB"
            sub_a.mkdir(parents=True)
            sub_b.mkdir(parents=True)
            (sub_a / "DJI_0001.JPG").write_bytes(b"\xff\xd8")
            (sub_b / "DJI_0001.JPG").write_bytes(b"\xff\xd8")

            metadata = [
                {"name": f"./Data/UAV_image/TestRegion/SubdirA/DJI_0001.JPG",
                 "lat": 10.0, "lon": 106.0},
                {"name": f"./Data/UAV_image/TestRegion/SubdirB/DJI_0001.JPG",
                 "lat": 10.1, "lon": 106.1},
            ]
            (metadata_dir / "TestRegion.json").write_text(
                json.dumps(metadata), encoding="utf-8"
            )

            from data.dataset_adapters import AVLDataset
            ds = AVLDataset(root, region="TestRegion")
            ds.load()
            names = [e["image_name"] for e in ds._entries]
            self.assertEqual(len(names), 2, f"Expected 2 entries, got {len(names)}")
            self.assertEqual(len(set(names)), 2,
                             f"Duplicate image_name detected: {names}")
            self.assertIn("SubdirA/DJI_0001.JPG", names)
            self.assertIn("SubdirB/DJI_0001.JPG", names)

    def test_avl_image_name_uses_metadata_path_not_resolved_fallback(self):
        """When metadata references a subdir that doesn't exist on disk,
        the fallback resolver finds the file elsewhere. image_name must
        still use the METADATA subdirectory, not the fallback location."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            metadata_dir = root / "metadata"
            metadata_dir.mkdir()

            sub_a = root / "UAV_image" / "TestRegion" / "SubdirA"
            sub_a.mkdir(parents=True)
            (sub_a / "DJI_0001.JPG").write_bytes(b"\xff\xd8")
            # SubdirB does NOT exist on disk — fallback will find in SubdirA

            metadata = [
                {"name": "./Data/UAV_image/TestRegion/SubdirA/DJI_0001.JPG",
                 "lat": 10.0, "lon": 106.0},
                {"name": "./Data/UAV_image/TestRegion/SubdirB/DJI_0001.JPG",
                 "lat": 10.1, "lon": 106.1},
            ]
            (metadata_dir / "TestRegion.json").write_text(
                json.dumps(metadata), encoding="utf-8"
            )

            from data.dataset_adapters import AVLDataset
            ds = AVLDataset(root, region="TestRegion")
            ds.load()
            names = [e["image_name"] for e in ds._entries]
            self.assertIn("SubdirA/DJI_0001.JPG", names)
            if len(names) == 2:
                self.assertIn("SubdirB/DJI_0001.JPG", names)
                self.assertEqual(len(set(names)), 2,
                                 f"Duplicate image_name from fallback resolution: {names}")


if __name__ == "__main__":
    unittest.main()
