import csv
import json
import sys
import types
import tempfile
import unittest
from types import SimpleNamespace
from pathlib import Path

import numpy as np

import benchmark_runner as br
import run_experiments_parallel as parallel_runner
from data.dataset_adapters import UAVVisLocDataset


def _load_pipeline_affine_helper():
    """Load native AnyVisLoc pipeline.py without importing optional native dependencies."""
    base = "uav_geolocalization.baselines.native_anyvisloc"
    module_names = [
        base,
        f"{base}.benchmark_utils",
        f"{base}.geometry_solver",
        f"{base}.types",
        f"{base}.coord_transforms",
        f"{base}.pipeline",
    ]
    saved = {name: sys.modules.get(name) for name in module_names}
    try:
        pkg = types.ModuleType(base)
        pkg.__path__ = [
            str(
                Path(__file__).resolve().parents[1]
                / "src"
                / "uav_geolocalization"
                / "baselines"
                / "native_anyvisloc"
            )
        ]
        sys.modules[base] = pkg

        bench = types.ModuleType(f"{base}.benchmark_utils")
        for name in [
            "computeCameraMatrix",
            "dumpRotateImage",
            "Match2Pos_all",
            "retrieval_all",
            "pos2error",
            "query_data_from_file",
            "img_name",
            "save_data",
            "resolution_size",
            "view_center",
        ]:
            setattr(bench, name, lambda *args, **kwargs: None)
        sys.modules[f"{base}.benchmark_utils"] = bench

        geom = types.ModuleType(f"{base}.geometry_solver")
        geom.HomographyYawHeightSolver = object
        geom.extract_yaw_and_height = lambda *args, **kwargs: (None, None, 0)
        sys.modules[f"{base}.geometry_solver"] = geom

        type_mod = types.ModuleType(f"{base}.types")
        type_mod.CameraIntrinsics = object
        type_mod.Pose = object
        sys.modules[f"{base}.types"] = type_mod

        coord = types.ModuleType(f"{base}.coord_transforms")
        coord.deg2utm = lambda *args, **kwargs: (0.0, 0.0)
        coord.utm2deg = lambda *args, **kwargs: (0.0, 0.0)
        sys.modules[f"{base}.coord_transforms"] = coord

        import importlib.util

        path = (
            Path(__file__).resolve().parents[1]
            / "src"
            / "uav_geolocalization"
            / "baselines"
            / "native_anyvisloc"
            / "pipeline.py"
        )
        spec = importlib.util.spec_from_file_location(f"{base}.pipeline", path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[f"{base}.pipeline"] = module
        spec.loader.exec_module(module)
        return module.affine_rotation_angle_rad
    finally:
        for name, old in saved.items():
            if old is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old


class BenchmarkRepairContractsTest(unittest.TestCase):
    def _make_mini_uav_visloc_dataset(self, root: Path) -> None:
        import cv2

        ranges = [
            ["mapname", "LT_lat_map", "LT_lon_map", "RB_lat_map", "RB_lon_map"],
            ["satellite01.png", "10.002", "106.000", "10.000", "106.002"],
            ["satellite02.png", "11.002", "107.000", "11.000", "107.002"],
        ]
        with open(root / "satellite_coordinates_range.csv", "w", newline="") as f:
            csv.writer(f).writerows(ranges)

        for loc, lat, lon in [("01", 10.001, 106.001), ("02", 11.001, 107.001)]:
            loc_dir = root / loc
            drone_dir = loc_dir / "drone"
            sat_dir = loc_dir / "satellite_map"
            drone_dir.mkdir(parents=True)
            sat_dir.mkdir(parents=True)
            img = np.full((256, 256, 3), 80 if loc == "01" else 160, dtype=np.uint8)
            cv2.imwrite(str(sat_dir / f"satellite{loc}.png"), img)
            with open(loc_dir / f"{loc}.csv", "w", newline="") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=["filename", "lat", "lon", "height", "Kappa", "Phi1", "Omega"],
                )
                writer.writeheader()
                for idx in range(1, 6):
                    name = f"{loc}_{idx:04d}.JPG"
                    cv2.imwrite(str(drone_dir / name), img)
                    writer.writerow(
                        {
                            "filename": name,
                            "lat": lat,
                            "lon": lon,
                            "height": 80.0,
                            "Kappa": 0.0,
                            "Phi1": 0.0,
                            "Omega": 0.0,
                        }
                    )

    def test_uav_visloc_intra_flight_corpus_uses_selected_query_locations(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._make_mini_uav_visloc_dataset(root)
            dataset = UAVVisLocDataset(root, max_samples=3, protocol="intra_flight")
            dataset.load()

            self.assertEqual(dataset.selected_query_locations(), ["01"])
            ref_names = [str(ref["ref_name"]) for ref in dataset.iter_reference_corpus()]

        self.assertGreater(len(ref_names), 0)
        self.assertTrue(all(name.startswith("01/") for name in ref_names), ref_names[:5])
        self.assertFalse(any(name.startswith("02/") for name in ref_names), ref_names[:5])

    def test_uav_visloc_selected_locations_honor_percent_sampling(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._make_mini_uav_visloc_dataset(root)
            dataset = UAVVisLocDataset(root, max_samples_percent=10, protocol="intra_flight")
            dataset.load()

            self.assertEqual(dataset.selected_query_locations(), ["01"])

    def test_uav_visloc_tile_config_controls_corpus_tile_size_and_step(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._make_mini_uav_visloc_dataset(root)
            dataset = UAVVisLocDataset(
                root,
                max_samples=3,
                protocol="intra_flight",
                tile_size_px=128,
                tile_step_px=64,
            )
            dataset.load()

            refs = list(dataset.iter_reference_corpus())

        self.assertGreater(len(refs), 1)
        self.assertEqual(refs[0]["ref_image"].shape[:2], (128, 128))
        self.assertEqual(refs[0]["anchor"]["tile_size_px"], 128)
        self.assertEqual(refs[0]["anchor"]["tile_step_px"], 64)
        self.assertTrue(any("_r000000_c000064" in str(ref["ref_name"]) for ref in refs))

    def test_uav_visloc_protocol_fields_flag_cross_flight_candidates(self):
        fields = br.resolve_uav_visloc_protocol_fields(
            {
                "image_name": "01/01_0001.JPG",
                "flight_id": "01",
                "selected_reference_name": "03/satellite03_r000000_c000000.png",
                "raw_top1_reference_name": "01/satellite01_r000000_c000000.png",
                "raw_retrieval_topk_indices": [0, 1],
            },
            protocol="intra_flight",
            reference_names={
                0: "01/satellite01_r000000_c000000.png",
                1: "03/satellite03_r000000_c000000.png",
            },
        )

        self.assertEqual(fields["query_flight_id"], "01")
        self.assertEqual(fields["selected_ref_flight_id"], "03")
        self.assertEqual(fields["raw_topk_ref_flights"], ["01", "03"])
        self.assertTrue(fields["retrieval_protocol_violation"])

    def test_uav_visloc_protocol_fields_preserve_phase1_filter_violation(self):
        fields = br.resolve_uav_visloc_protocol_fields(
            {
                "image_name": "01/01_0001.JPG",
                "flight_id": "01",
                "selected_reference_name": "01/satellite01_r000000_c000000.png",
                "raw_top1_reference_name": "01/satellite01_r000001_c000000.png",
                "raw_retrieval_topk_indices": [0],
                "retrieval_protocol_violation": True,
            },
            protocol="intra_flight",
            reference_names={0: "01/satellite01_r000001_c000000.png"},
        )

        self.assertTrue(fields["retrieval_protocol_violation"])

    def test_generic_retrieval_diagnostics_reports_topk_gap_and_anchor_jump(self):
        ref_anchors = {
            0: {"lat": 10.0000, "lon": 106.0000},
            1: {"lat": 10.0009, "lon": 106.0000},
            2: {"lat": 10.0020, "lon": 106.0000},
        }
        rows = [
            {
                "image_name": "000001.png",
                "gt_lat": 10.0000,
                "gt_lon": 106.0000,
                "geodetic_anchor": ref_anchors[2],
                "raw_retrieval_anchor": ref_anchors[2],
                "raw_retrieval_topk_indices": [2, 1, 0],
            },
            {
                "image_name": "000002.png",
                "gt_lat": 10.0009,
                "gt_lon": 106.0000,
                "geodetic_anchor": ref_anchors[1],
                "raw_retrieval_anchor": ref_anchors[2],
                "raw_retrieval_topk_indices": [2, 1, 0],
            },
        ]

        diag = br.compute_retrieval_diagnostics(
            rows,
            ref_anchors,
            dataset_name="vpair",
            thresholds_m=[50, 150, 300],
            topk_values=[1, 5, 10, 20, 50],
        )
        global_diag = diag["global"]

        self.assertEqual(diag["schema_version"], 1)
        self.assertEqual(global_diag["num_frames"], 2)
        self.assertAlmostEqual(global_diag["raw_top50_error_m_hit_at_150m"], 1.0)
        self.assertLess(global_diag["raw_top1_error_m_hit_at_150m"], 1.0)
        self.assertGreater(global_diag["selected_vs_raw_gap_at_150m"], 0.0)
        self.assertIsNotNone(global_diag["anchor_jump_m_p50"])
        self.assertIn("anchor_jump_m", diag["rows"][1])

    def test_official_filter_excludes_retrieval_protocol_violation(self):
        manifest = {
            "eval_mode": "end_to_end",
            "strict_components": True,
            "strict_end2end": True,
            "oracle_mode": False,
            "deprecated_alias": False,
            "run_integrity_for_official_metrics": True,
            "native_prior_uses_exact_gt": False,
            "native_anyvisloc_schema_ok": True,
            "retrieval_protocol_violation_count": 1,
        }

        self.assertFalse(br.is_official_metrics_payload(manifest, {}))
        # filtered_candidates_count > 0 means the intra_flight over-fetch
        # filter did its job (normal on every uav_visloc run); only actual
        # violations disqualify a run from official metrics.
        self.assertTrue(
            br.is_official_metrics_payload(
                {
                    **manifest,
                    "retrieval_protocol_violation_count": 0,
                    "retrieval_protocol_filtered_candidates_count": 1,
                },
                {},
            )
        )

    def test_retrieval_cache_key_payload_includes_uav_protocol_and_selected_flights(self):
        common = {
            "schema_version": br.RETRIEVAL_DB_CACHE_SCHEMA_VERSION,
            "dataset_name": "uav_visloc",
            "retrieval_backend": "ultravpr",
            "model_signature_hash": "model",
            "preprocess_signature_hash": "pre",
            "corpus_hash": "corpus",
            "corpus_size": 10,
            "uav_visloc_protocol": "intra_flight",
            "selected_query_locations": ["01"],
            "selected_sample_signature": {"num_selected": 3, "first": ["01/a.jpg"], "last": ["01/c.jpg"]},
            "corpus_ref_flight_hash": "flight-hash-01",
            "uav_tile_size_px": 512,
            "uav_tile_step_px": 256,
        }
        key_a = br.build_retrieval_cache_key_payload(**common)
        key_b = br.build_retrieval_cache_key_payload(
            **{
                **common,
                "selected_query_locations": ["02"],
                "corpus_ref_flight_hash": "flight-hash-02",
            }
        )

        self.assertEqual(key_a["uav_visloc_protocol"], "intra_flight")
        self.assertNotEqual(br._stable_json_hash(key_a), br._stable_json_hash(key_b))
        key_c = br.build_retrieval_cache_key_payload(**{**common, "uav_tile_size_px": 384})
        self.assertEqual(key_a["uav_tile_size_px"], 512)
        self.assertEqual(key_a["uav_tile_step_px"], 256)
        # retrieval_top_k is deliberately NOT part of the cache identity: the
        # encoded DB does not depend on k, and keying on it forced a full
        # re-encode (plus GC of the shared entry) per k-sweep cell.
        self.assertNotIn("retrieval_top_k", key_a)
        self.assertNotEqual(br._stable_json_hash(key_a), br._stable_json_hash(key_c))

    def test_dsm_tile_metadata_changes_retrieval_corpus_hash(self):
        corpus_a = [{
            "ref_name": "QZ_Town_r000000_c000000.png",
            "lat": 10.0,
            "lon": 106.0,
            "anchor_meta": {
                "anchor_source": "raster_geotransform",
                "row_start": 0,
                "col_start": 0,
                "tile_size_px": 512,
                "tile_step_px": 256,
            },
        }]
        corpus_b = [{
            **corpus_a[0],
            "anchor_meta": {
                **corpus_a[0]["anchor_meta"],
                "row_start": 256,
            },
        }]

        self.assertNotEqual(br._stable_json_hash(corpus_a), br._stable_json_hash(corpus_b))

    def test_final_pose_gate_is_independent_from_fgo_guard(self):
        result = {
            "fgo_reject_code": br.FGO_REJECT_STEP_XY,
            "fgo_guard_reject": True,
            "geometry_anchor_offset_m": 12.0,
            "shape_confidence": 0.4,
            "reproj_residual_px": 2.0,
        }
        accepted, reason = br.evaluate_final_pose_acceptance(
            result,
            {
                "final_pose_gate_enabled": True,
                "final_pose_max_geometry_anchor_offset_m": 50.0,
                "final_pose_min_shape_conf": 0.1,
                "final_pose_max_reproj_residual_px": 8.0,
            },
        )

        self.assertTrue(accepted)
        self.assertEqual(reason, "")

    def test_final_pose_gate_rejects_bad_geometry_offset(self):
        accepted, reason = br.evaluate_final_pose_acceptance(
            {
                "geometry_anchor_offset_m": 120.0,
                "shape_confidence": 0.4,
                "reproj_residual_px": 2.0,
            },
            {
                "final_pose_gate_enabled": True,
                "final_pose_max_geometry_anchor_offset_m": 50.0,
                "final_pose_min_shape_conf": 0.1,
                "final_pose_max_reproj_residual_px": 8.0,
            },
        )

        self.assertFalse(accepted)
        self.assertEqual(reason, "final_pose_anchor_offset_reject")

    def test_paper_metrics_counts_final_pose_gate_separately_from_fgo_guard(self):
        rows = [
            {"failure_reason": "final_pose_gate_reject", "fgo_reject_code": br.FGO_REJECT_STEP_XY},
            {"failure_reason": "", "fgo_reject_code": br.FGO_REJECT_STEP_ALT},
            {"failure_reason": "geometry_anchor_offset_reject", "fgo_reject_code": None},
        ]

        self.assertEqual(br.count_final_pose_gate_rejects(rows), 1)
        self.assertEqual(br.count_fgo_guard_rejects(rows), 2)

    def test_uav_retrieval_diagnostics_include_top20_and_top50(self):
        reference_anchors = {
            idx: {"lat": 10.0 + idx * 0.001, "lon": 106.0}
            for idx in range(60)
        }
        row = {
            "flight_id": "01",
            "image_name": "01/q.jpg",
            "gt_lat": 10.049,
            "gt_lon": 106.0,
            "geodetic_anchor": reference_anchors[0],
            "raw_retrieval_anchor": reference_anchors[0],
            "raw_retrieval_topk_indices": list(range(50)),
        }

        diagnostics = br.compute_uav_visloc_retrieval_diagnostics(
            [row],
            reference_anchors,
        )

        global_metrics = diagnostics["global"]
        self.assertIn("raw_top20_error_m_hit_at_150m", global_metrics)
        self.assertIn("raw_top50_error_m_hit_at_150m", global_metrics)
        self.assertEqual(global_metrics["raw_top20_error_m_hit_at_150m"], 0.0)
        self.assertEqual(global_metrics["raw_top50_error_m_hit_at_150m"], 1.0)

    def test_retrieval_only_runner_writes_diagnostic_artifacts(self):
        class FakeRetriever:
            def __init__(self):
                self._backend = "fake"
                self._top_k = 50
                self.database = SimpleNamespace(size=2)

            def retrieve_top_k(self, query_image, top_k=None):
                return [
                    SimpleNamespace(
                        candidate_index=0,
                        similarity_score=0.9,
                        geodetic_anchor={"lat": 10.0, "lon": 106.0},
                        timing_ms=1.5,
                    ),
                    SimpleNamespace(
                        candidate_index=1,
                        similarity_score=0.8,
                        geodetic_anchor={"lat": 10.001, "lon": 106.0},
                        timing_ms=1.5,
                    ),
                ]

        class FakeDataset:
            def __len__(self):
                return 1

            def __iter__(self):
                yield {
                    "image_name": "01/q.jpg",
                    "query_image": np.zeros((16, 16, 3), dtype=np.uint8),
                    "gt_pose": {"lat": 10.001, "lon": 106.0, "alt": 80.0, "yaw": 0.0, "pitch": 0.0},
                    "metadata": {"dataset": "uav_visloc", "location": "01"},
                }

        with tempfile.TemporaryDirectory() as td:
            runner = br.BenchmarkRunner(
                pipeline_name="ours_cascade_homography",
                output_dir=td,
                eval_mode="end_to_end",
                retrieval_only=True,
                pipeline_overrides={"retrieval_top_k": 50, "uav_tile_size_px": 384, "uav_tile_step_px": 192},
            )
            pipeline = {
                "retrieval": FakeRetriever(),
                "retrieval_reference_names": {0: "01/a.png", 1: "01/b.png"},
                "retrieval_reference_anchors": {
                    0: {"lat": 10.0, "lon": 106.0},
                    1: {"lat": 10.001, "lon": 106.0},
                },
                "config": {"retrieval_top_k": 50},
            }

            result = runner._run_retrieval_only_dataset(
                "uav_visloc",
                FakeDataset(),
                pipeline,
                Path(td) / "uav_visloc" / "ours_cascade_homography" / "run",
            )

            run_dir = Path(result["run_dir"])
            self.assertTrue((run_dir / "predictions.csv").exists())
            self.assertTrue((run_dir / "uav_visloc_retrieval_diagnostics.json").exists())
            self.assertTrue((run_dir / "metrics_summary.json").exists())
            self.assertTrue((run_dir / "run_manifest.json").exists())
            summary = json.loads((run_dir / "metrics_summary.json").read_text())
            self.assertTrue(summary["retrieval_only"])
            self.assertEqual(summary["manifest"]["uav_tile_size_px"], 384)
            self.assertEqual(summary["manifest"]["retrieval_top_k"], 50)


    def test_pipeline_metadata_aliases_and_native_flags(self):
        alias = br.resolve_pipeline_metadata("anyvisloc_e2e")
        self.assertEqual(alias["canonical_pipeline"], "camp_roma_homography_wrapper_e2e")
        self.assertTrue(alias["deprecated_alias"])
        self.assertFalse(alias["native_baseline"])
        self.assertFalse(alias["oracle_mode"])
        self.assertFalse(alias["uses_oracle"])
        self.assertTrue(alias["is_wrapper"])
        self.assertEqual(alias["method_family"], "camp_roma_homography_wrapper")

        old_ortho = br.resolve_pipeline_metadata("ortholoc_e2e")
        self.assertEqual(old_ortho["canonical_pipeline"], "ultravpr_roma_homography_wrapper_e2e")
        self.assertTrue(old_ortho["deprecated_alias"])
        self.assertFalse(old_ortho["native_baseline"])
        self.assertFalse(old_ortho["uses_oracle"])
        self.assertTrue(old_ortho["is_wrapper"])
        self.assertEqual(old_ortho["method_family"], "ultravpr_roma_homography_wrapper")

        native = br.resolve_pipeline_metadata("native_anyvisloc_e2e")
        self.assertEqual(native["canonical_pipeline"], "native_anyvisloc_e2e")
        self.assertTrue(native["native_baseline"])
        self.assertFalse(native["oracle_mode"])
        self.assertFalse(native["uses_oracle"])
        self.assertTrue(native["is_native"])
        self.assertEqual(native["method_family"], "native_anyvisloc")

        oracle = br.resolve_pipeline_metadata("ortholoc_2d3d_oracle")
        self.assertTrue(oracle["native_baseline"])
        self.assertTrue(oracle["oracle_mode"])
        self.assertTrue(oracle["uses_oracle"])
        self.assertIn("oracle_tile", oracle["oracle_modes"])
        self.assertEqual(oracle["method_family"], "ortholoc_2d3d")

    def test_manifest_marks_contaminated_e2e_not_official(self):
        with tempfile.TemporaryDirectory() as td:
            runner = br.BenchmarkRunner(
                pipeline_name="ours",
                output_dir=td,
                eval_mode="end_to_end",
                strict_end2end=False,
                verbose=False,
            )
            runner._stats["retrieval_oracle_fallback_frames"] = 1
            run_dir = Path(td) / "manifest"
            run_dir.mkdir()
            runner._write_run_manifest(run_dir, "vpair")
            payload = json.loads((run_dir / "run_manifest.json").read_text())
            self.assertFalse(payload["run_integrity_for_official_metrics"])
            self.assertTrue(payload["end2end_oracle_fallback_contaminated"])

    def test_manifest_distinguishes_running_from_completed_artifacts(self):
        with tempfile.TemporaryDirectory() as td:
            runner = br.BenchmarkRunner(
                pipeline_name="ours",
                output_dir=td,
                eval_mode="end_to_end",
                verbose=False,
            )
            run_dir = Path(td) / "manifest"
            run_dir.mkdir()

            runner._write_run_manifest(run_dir, "vpair")
            running = json.loads((run_dir / "run_manifest.json").read_text())
            self.assertFalse(running["run_completed"])

            runner._write_run_manifest(run_dir, "vpair", run_completed=True)
            completed = json.loads((run_dir / "run_manifest.json").read_text())
            self.assertTrue(completed["run_completed"])

    def test_standard_prediction_and_summary_artifacts(self):
        with tempfile.TemporaryDirectory() as td:
            run_dir = Path(td)
            rows = [
                {
                    "image_name": "q001.jpg",
                    "gt_lat": 10.0,
                    "gt_lon": 106.0,
                    "gt_alt": 50.0,
                    "pred_lat": 10.00001,
                    "pred_lon": 106.00002,
                    "pred_alt": 51.0,
                    "pose_success": True,
                    "accepted_for_metrics": True,
                    "retrieval_anchor_error_m": 4.2,
                    "raw_retrieval_candidate_index": 0,
                    "retrieval_candidate_index": 1,
                    "num_inliers": 12,
                    "shape_confidence": 0.7,
                    "failure_reason": "",
                    "timings": {"retrieval_ms": 1.5, "matching_ms": 2.5, "geometry_ms": 3.5},
                }
            ]
            br.write_predictions_csv(run_dir / "predictions.csv", rows)
            br.write_metrics_summary_json(
                run_dir / "metrics_summary.json",
                dataset="vpair",
                pipeline="native_anyvisloc_e2e",
                metrics={"success_rate": 1.0},
                paper_metrics={"FPS": 10.0},
                results=rows,
                manifest={
                    "run_integrity_for_official_metrics": True,
                    "uav_visloc_protocol": "intra_flight",
                    "telemetry_prior_policy": "simulated_uav_telemetry",
                    "native_prior_uses_exact_gt": False,
                    "native_anyvisloc_schema_ok": True,
                    "native_anyvisloc_cache_requested": True,
                    "native_anyvisloc_cache_enabled": False,
                    "native_anyvisloc_cache_key": "abc123",
                    "native_anyvisloc_cache_hit": False,
                    "native_anyvisloc_cache_rebuild_reason": "not_implemented",
                },
            )

            with open(run_dir / "predictions.csv", newline="") as f:
                csv_rows = list(csv.DictReader(f))
            self.assertEqual(len(csv_rows), 1)
            self.assertIn("pose_success", csv_rows[0])
            self.assertIn("accepted_for_metrics", csv_rows[0])
            self.assertIn("retrieval_anchor_error_m", csv_rows[0])
            self.assertIn("timing_matching_ms", csv_rows[0])

            summary = json.loads((run_dir / "metrics_summary.json").read_text())
            self.assertEqual(summary["dataset"], "vpair")
            self.assertEqual(summary["pipeline"], "native_anyvisloc_e2e")
            self.assertEqual(summary["num_frames"], 1)
            self.assertEqual(summary["num_pose_success"], 1)
            self.assertEqual(summary["num_accepted_for_metrics"], 1)
            self.assertEqual(summary["uav_visloc_protocol"], "intra_flight")
            self.assertEqual(summary["telemetry_prior_policy"], "simulated_uav_telemetry")
            self.assertFalse(summary["native_prior_uses_exact_gt"])
            self.assertTrue(summary["native_anyvisloc_schema_ok"])
            self.assertTrue(summary["native_anyvisloc_cache_requested"])
            self.assertFalse(summary["native_anyvisloc_cache_enabled"])
            self.assertEqual(summary["native_anyvisloc_cache_key"], "abc123")
            self.assertFalse(summary["native_anyvisloc_cache_hit"])
            self.assertEqual(summary["native_anyvisloc_cache_rebuild_reason"], "not_implemented")

    def test_uav_anchor_audit_flags_zero_duplicate_and_out_of_bbox(self):
        refs = [
            {"ref_name": "01/a.png", "anchor": {"lat": 0.0, "lon": 0.0}},
            {"ref_name": "01/b.png", "anchor": {"lat": 10.0, "lon": 20.0}},
            {"ref_name": "01/c.png", "anchor": {"lat": 10.0, "lon": 20.0}},
            {"ref_name": "01/d.png", "anchor": {"lat": 99.0, "lon": 20.0}},
        ]
        bboxes = {"01": {"min_lat": 9.0, "max_lat": 11.0, "min_lon": 19.0, "max_lon": 21.0}}
        audit = br.audit_uav_visloc_reference_anchors(refs, bboxes_by_flight=bboxes)
        self.assertFalse(audit["passed"])
        self.assertEqual(audit["zero_anchor_count"], 1)
        self.assertGreaterEqual(audit["duplicate_anchor_groups"], 1)
        self.assertEqual(audit["out_of_bbox_count"], 2)

    def test_native_anyvisloc_reranks_topn_by_pnp_inliers(self):
        candidates = [
            {"candidate_id": "top1", "retrieval_rank": 0, "retrieval_score": 0.99, "pnp_inliers": 3, "homography_quality": 100.0},
            {"candidate_id": "top2", "retrieval_rank": 1, "retrieval_score": 0.60, "pnp_inliers": 14, "homography_quality": 1.0},
        ]
        selected = br.select_native_anyvisloc_candidate_by_pnp_inliers(candidates)
        self.assertEqual(selected["candidate_id"], "top2")

    def test_ortholoc_preflight_fails_closed_for_missing_dependency(self):
        missing = {"appdirs"}

        def fake_import(name):
            if name in missing:
                raise ImportError("missing appdirs")
            return object()

        report = br.check_ortholoc_preflight(importer=fake_import, repo_root=Path.cwd())
        self.assertFalse(report["ready"])
        self.assertIn("appdirs", report["missing_dependencies"])
        self.assertIn("reference/ortholoc", report["reference_path"])
        self.assertFalse(report["numeric_runtime_enabled"])
        self.assertFalse(report["dataset_extraction_wired"])
        self.assertIn("dop_assets_present", report)
        self.assertIn("dsm_assets_present", report)

    def test_step_metrics_summary_reports_phase_timings_and_success(self):
        rows = [
            {
                "pose_success": True,
                "accepted_for_metrics": True,
                "failure_reason": "",
                "geodetic_anchor": {"lat": 10.0, "lon": 20.0},
                "num_inliers": 12,
                "homography_valid": True,
                "timings": {"retrieval_ms": 10.0, "matching_ms": 20.0, "geometry_ms": 30.0, "fgo_ms": 0.0},
            },
            {
                "pose_success": False,
                "accepted_for_metrics": False,
                "failure_reason": "geometry_failed",
                "geodetic_anchor": {"lat": 10.0, "lon": 20.0},
                "num_inliers": 2,
                "homography_valid": False,
                "timings": {"retrieval_ms": 12.0, "matching_ms": 22.0, "geometry_ms": 5.0, "fgo_ms": 0.0},
            },
        ]
        summary = br.compute_step_metrics_summary(rows)
        self.assertEqual(summary["num_frames"], 2)
        self.assertAlmostEqual(summary["overall"]["pose_success_rate"], 0.5)
        self.assertAlmostEqual(summary["steps"]["retrieval"]["success_rate"], 1.0)
        self.assertAlmostEqual(summary["steps"]["geometry"]["success_rate"], 0.5)
        self.assertAlmostEqual(summary["steps"]["matching"]["timing_ms_mean"], 21.0)
        self.assertEqual(summary["failure_reason_counts"]["geometry_failed"], 1)

    def test_native_anyvisloc_uses_noisy_telemetry_and_no_gt_yaw_alt_fallback(self):
        class FakeNativeAnyVisLoc:
            def __init__(self):
                self.received_true_pos = None

            def process_frame(self, query_path, true_pos):
                self.received_true_pos = dict(true_pos)
                return {
                    "success": True,
                    "pred_lat": 10.00020,
                    "pred_lon": 106.00030,
                    # Deliberately omit src_alt_m/src_yaw_deg. Native position PnP can
                    # succeed while yaw/height extraction fails; runner must not fill GT.
                    "geodetic_anchor": {"lat": 10.00015, "lon": 106.00025, "alt": 70.0},
                    "raw_retrieval_anchor": {"lat": 10.00010, "lon": 106.00020, "alt": 70.0},
                    "selected_reference_name": "ref_selected.png",
                    "raw_top1_reference_name": "ref_raw.png",
                    "raw_retrieval_candidate_index": 3,
                    "raw_retrieval_topk_indices": [3, 8, 13],
                    "selected_candidate_rank": 1,
                    "n_inliers": 11,
                    "num_matches": 47,
                    "retrieval_time_s": 0.01,
                    "match_time_s": 0.02,
                    "pnp_time_s": 0.03,
                }

        runner = br.BenchmarkRunner(
            pipeline_name="native_anyvisloc_e2e",
            output_dir=tempfile.mkdtemp(),
            eval_mode="end_to_end",
            strict_end2end=True,
            verbose=False,
        )
        native = FakeNativeAnyVisLoc()
        frame = {
            "image_name": "q001.jpg",
            "query_path": "/tmp/q001.jpg",
            "gt_pose": {
                "lat": 10.0,
                "lon": 106.0,
                "alt": 80.0,
                "yaw": 45.0,
                "pitch": -35.0,
                "roll": 2.0,
            },
            "camera": {"width": 4000, "height": 3000},
            "metadata": {"cam_size_mm": 7.81, "focal_len_mm": 3.6},
        }
        bias_drift = br._precompute_bias_drift(10, seed=7)
        result = runner._process_frame(
            frame,
            {"native_backend": "anyvisloc", "native_pipeline": native, "config": {}},
            frame_index=4,
            bias_drift=bias_drift,
        )

        received = native.received_true_pos
        self.assertIsNotNone(received)
        self.assertNotEqual(received["yaw"], frame["gt_pose"]["yaw"])
        self.assertNotEqual(received["pitch"], frame["gt_pose"]["pitch"])
        self.assertNotEqual(received["roll"], frame["gt_pose"]["roll"])
        self.assertNotEqual(received["rel_alt"], frame["gt_pose"]["alt"])
        self.assertNotEqual(received["lat"], frame["gt_pose"]["lat"])
        self.assertNotEqual(received["lon"], frame["gt_pose"]["lon"])

        self.assertTrue(result["pose_success"])
        self.assertTrue(result["native_anyvisloc_schema_ok"])
        self.assertFalse(result["native_prior_uses_exact_gt"])
        self.assertEqual(result["native_yaw_height_status"], "failed")
        self.assertFalse(result["E_alt_available"])
        self.assertTrue(np.isnan(result["pred_alt"]))
        self.assertTrue(np.isnan(result["pred_heading"]))

        self.assertEqual(result["geodetic_anchor"]["lat"], 10.00015)
        self.assertEqual(result["raw_retrieval_anchor"]["lat"], 10.00010)
        self.assertEqual(result["selected_reference_name"], "ref_selected.png")
        self.assertEqual(result["raw_top1_reference_name"], "ref_raw.png")
        self.assertEqual(result["visual_pred_lat"], result["pred_lat"])
        self.assertEqual(result["visual_pred_lon"], result["pred_lon"])
        self.assertTrue(np.isnan(result["visual_pred_alt"]))
        self.assertEqual(result["num_inliers"], 11)
        self.assertEqual(result["num_matches"], 47)
        self.assertEqual(result["timings"]["retrieval_ms"], 10.0)
        self.assertEqual(result["timings"]["matching_ms"], 20.0)
        self.assertEqual(result["timings"]["geometry_ms"], 30.0)

        manifest = runner._build_run_manifest_payload("avl")
        self.assertEqual(manifest["telemetry_prior_policy"], "simulated_uav_telemetry")
        self.assertFalse(manifest["native_prior_uses_exact_gt"])
        self.assertTrue(manifest["native_anyvisloc_schema_ok"])
        self.assertEqual(manifest["native_yaw_height_status"], "failed")

    def test_native_anyvisloc_oracle_tile_passes_override_to_native_pipeline(self):
        class FakeNativeAnyVisLoc:
            def __init__(self):
                self.received_true_pos = None

            def process_frame(self, query_path, true_pos):
                self.received_true_pos = dict(true_pos)
                return {
                    "success": True,
                    "pred_lat": 10.00020,
                    "pred_lon": 106.00030,
                    "pred_alt": 80.0,
                    "src_yaw_deg": 45.0,
                    "geodetic_anchor": true_pos["oracle_reference_anchor"],
                    "raw_retrieval_anchor": true_pos["oracle_reference_anchor"],
                    "selected_reference_name": "oracle_tile",
                    "raw_top1_reference_name": "oracle_tile",
                    "n_inliers": 11,
                    "num_matches": 47,
                    "retrieval_time_s": 0.0,
                    "match_time_s": 0.02,
                    "pnp_time_s": 0.03,
                }

        runner = br.BenchmarkRunner(
            pipeline_name="native_anyvisloc_oracle_tile",
            output_dir=tempfile.mkdtemp(),
            eval_mode="end_to_end",
            strict_end2end=True,
            verbose=False,
        )
        native = FakeNativeAnyVisLoc()
        frame = {
            "image_name": "q001.jpg",
            "query_path": "/tmp/q001.jpg",
            "reference_path": Path("/tmp/ref001.png"),
            "gt_pose": {
                "lat": 10.0,
                "lon": 106.0,
                "alt": 80.0,
                "yaw": 45.0,
                "pitch": -35.0,
                "roll": 2.0,
            },
            "camera": {"width": 4000, "height": 3000},
            "metadata": {"cam_size_mm": 7.81, "focal_len_mm": 3.6},
        }
        result = runner._process_frame(
            frame,
            {
                "native_backend": "anyvisloc",
                "native_pipeline": native,
                "config": {"native_oracle_tile": True},
            },
            frame_index=0,
            bias_drift=br._precompute_bias_drift(2, seed=3),
        )

        self.assertTrue(result["pose_success"])
        self.assertTrue(native.received_true_pos["oracle_tile_enabled"])
        self.assertEqual(native.received_true_pos["oracle_reference_anchor"]["lat"], frame["gt_pose"]["lat"])
        self.assertEqual(native.received_true_pos["oracle_reference_path"], str(frame["reference_path"]))
        manifest = runner._build_run_manifest_payload("avl")
        self.assertTrue(manifest["oracle_mode"])
        self.assertTrue(manifest["uses_oracle"])
        self.assertIn("oracle_tile", manifest["oracle_modes"])
        self.assertFalse(manifest["run_integrity_for_official_metrics"])

    def test_affine_rotation_angle_helper_replaces_missing_cv2_decompose_affine(self):
        affine_rotation_angle_rad = _load_pipeline_affine_helper()
        matrix = np.array(
            [
                [np.cos(np.deg2rad(30.0)), np.sin(np.deg2rad(30.0)), 12.0],
                [-np.sin(np.deg2rad(30.0)), np.cos(np.deg2rad(30.0)), -5.0],
            ],
            dtype=np.float64,
        )
        self.assertAlmostEqual(np.rad2deg(affine_rotation_angle_rad(matrix)), 30.0, places=6)
        self.assertEqual(affine_rotation_angle_rad(None), 0.0)

    def test_parallel_runner_passes_uav_visloc_protocol_to_child_command(self):
        captured = {}

        def fake_run(cmd, stdout=None, stderr=None, text=None, check=None):
            captured["cmd"] = list(cmd)
            return SimpleNamespace(returncode=0)

        args = SimpleNamespace(
            output="results_parallel",
            device="cpu",
            max_samples=3,
            max_samples_percent=None,
            test_interval=None,
            finetuned=False,
            uav_visloc_protocol="intra_flight",
        )
        old_run = parallel_runner.subprocess.run
        try:
            parallel_runner.subprocess.run = fake_run
            with tempfile.TemporaryDirectory() as td:
                parallel_runner.run_experiment("uav_visloc", "ours", args, Path(td))
        finally:
            parallel_runner.subprocess.run = old_run

        self.assertIn("--uav-visloc-protocol", captured["cmd"])
        idx = captured["cmd"].index("--uav-visloc-protocol")
        self.assertEqual(captured["cmd"][idx + 1], "intra_flight")

    def test_comparison_excludes_deprecated_contaminated_and_bad_native_schema_rows(self):
        def write_run(root, dataset, pipeline, run_id, manifest_extra):
            run_dir = Path(root) / dataset / pipeline / run_id
            run_dir.mkdir(parents=True, exist_ok=True)
            manifest = {
                "eval_mode": "end_to_end",
                "strict_components": True,
                "strict_end2end": True,
                "oracle_mode": False,
                "deprecated_alias": False,
                "run_integrity_for_official_metrics": True,
                "native_prior_uses_exact_gt": False,
                "native_anyvisloc_schema_ok": True,
            }
            manifest.update(manifest_extra)
            (run_dir / "run_manifest.json").write_text(json.dumps(manifest, cls=br.NpEncoder))
            paper = {
                "M_ME": 1.0,
                "E_ME": 2.0,
                "HR": 1.0,
                "phase_retrieval_ms_mean": 1.0,
                "phase_matching_ms_mean": 1.0,
                "phase_geometry_ms_mean": 1.0,
                "phase_fgo_ms_mean": 0.0,
            }
            (run_dir / "paper_metrics_summary.json").write_text(json.dumps(paper, cls=br.NpEncoder))
            payload = {
                "predictions": [
                    {
                        "accepted_for_metrics": True,
                        "success": True,
                        "pred_lat": 10.00001,
                        "pred_lon": 106.00001,
                        "gt_lat": 10.0,
                        "gt_lon": 106.0,
                        "total_time_ms": 100.0,
                        "num_inliers": 9,
                        "geodetic_anchor": {"lat": 10.0, "lon": 106.0},
                        "raw_retrieval_anchor": {"lat": 10.0, "lon": 106.0},
                    }
                ]
            }
            (run_dir / "results_detailed.json").write_text(json.dumps(payload, cls=br.NpEncoder))

        with tempfile.TemporaryDirectory() as td:
            write_run(td, "avl", "native_anyvisloc_e2e", "bad_schema", {"native_anyvisloc_schema_ok": False})
            write_run(td, "avl", "native_anyvisloc_e2e", "bad_prior", {"native_prior_uses_exact_gt": True})
            write_run(td, "avl", "native_anyvisloc_e2e", "bad_tier", {"comparison_tier": "strict_e2e"})
            write_run(td, "avl", "anyvisloc_e2e", "deprecated", {"deprecated_alias": True})
            write_run(td, "avl", "ours", "good", {})

            comparison = br.build_comparison_artifacts(Path(td), eval_mode="end_to_end")

        self.assertEqual([row["pipeline"] for row in comparison["rows"]], ["ours"])

    def test_native_reproduction_rows_export_separately_from_strict_rows(self):
        def write_run(root, dataset, pipeline, run_id, manifest_extra):
            run_dir = Path(root) / dataset / pipeline / run_id
            run_dir.mkdir(parents=True, exist_ok=True)
            manifest = {
                "eval_mode": "end_to_end",
                "strict_components": True,
                "strict_end2end": True,
                "oracle_mode": False,
                "uses_oracle": False,
                "deprecated_alias": False,
                "run_integrity_for_official_metrics": True,
                "native_prior_uses_exact_gt": False,
                "native_anyvisloc_schema_ok": True,
                "retrieval_protocol_violation_count": 0,
                "retrieval_protocol_filtered_candidates_count": 0,
            }
            manifest.update(manifest_extra)
            (run_dir / "run_manifest.json").write_text(json.dumps(manifest, cls=br.NpEncoder))
            paper = {
                "M_ME": 1.0,
                "E_ME": 2.0,
                "HR": 1.0,
                "phase_retrieval_ms_mean": 1.0,
                "phase_matching_ms_mean": 1.0,
                "phase_geometry_ms_mean": 1.0,
                "phase_fgo_ms_mean": 0.0,
            }
            (run_dir / "paper_metrics_summary.json").write_text(json.dumps(paper, cls=br.NpEncoder))
            payload = {
                "predictions": [
                    {
                        "accepted_for_metrics": True,
                        "success": True,
                        "pred_lat": 10.00001,
                        "pred_lon": 106.00001,
                        "gt_lat": 10.0,
                        "gt_lon": 106.0,
                        "total_time_ms": 100.0,
                        "num_inliers": 9,
                        "geodetic_anchor": {"lat": 10.0, "lon": 106.0},
                        "raw_retrieval_anchor": {"lat": 10.0, "lon": 106.0},
                    }
                ]
            }
            (run_dir / "results_detailed.json").write_text(json.dumps(payload, cls=br.NpEncoder))

        with tempfile.TemporaryDirectory() as td:
            write_run(td, "avl", "ours", "strict", {"comparison_tier": "strict_e2e"})
            write_run(
                td,
                "avl",
                "native_anyvisloc_e2e",
                "native",
                {
                    "comparison_tier": "native_reproduction",
                    "native_baseline": True,
                    "is_native": True,
                    "method_family": "native_anyvisloc",
                    "tile_source": "native_camp_retrieval",
                    "pose_head": "native_anyvisloc_dsm_p3p",
                },
            )

            comparison = br.build_comparison_artifacts(Path(td), eval_mode="end_to_end")
            native_rows = json.loads(Path(comparison["tiers"]["native_reproduction"]["json"]).read_text())

        self.assertEqual([row["pipeline"] for row in comparison["rows"]], ["ours"])
        self.assertEqual([row["pipeline"] for row in native_rows], ["native_anyvisloc_e2e"])

    def test_comparison_excludes_smoke_run_groups(self):
        def write_run(run_dir: Path):
            run_dir.mkdir(parents=True, exist_ok=True)
            manifest = {
                "eval_mode": "end_to_end",
                "strict_components": True,
                "strict_end2end": True,
                "oracle_mode": False,
                "deprecated_alias": False,
                "run_integrity_for_official_metrics": True,
                "native_prior_uses_exact_gt": False,
                "native_anyvisloc_schema_ok": True,
                "retrieval_protocol_violation_count": 0,
                "retrieval_protocol_filtered_candidates_count": 0,
            }
            (run_dir / "run_manifest.json").write_text(json.dumps(manifest, cls=br.NpEncoder))
            paper = {
                "M_ME": 1.0,
                "E_ME": 2.0,
                "HR": 1.0,
                "phase_retrieval_ms_mean": 1.0,
                "phase_matching_ms_mean": 1.0,
                "phase_geometry_ms_mean": 1.0,
                "phase_fgo_ms_mean": 0.0,
            }
            (run_dir / "paper_metrics_summary.json").write_text(json.dumps(paper, cls=br.NpEncoder))
            payload = {
                "predictions": [
                    {
                        "accepted_for_metrics": True,
                        "success": True,
                        "pred_lat": 10.00001,
                        "pred_lon": 106.00001,
                        "gt_lat": 10.0,
                        "gt_lon": 106.0,
                        "total_time_ms": 100.0,
                        "num_inliers": 9,
                        "geodetic_anchor": {"lat": 10.0, "lon": 106.0},
                        "raw_retrieval_anchor": {"lat": 10.0, "lon": 106.0},
                    }
                ]
            }
            (run_dir / "results_detailed.json").write_text(json.dumps(payload, cls=br.NpEncoder))

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            write_run(root / "smoke" / "avl" / "ours" / "smoke_run")
            write_run(root / "avl" / "ours" / "full_run")

            comparison = br.build_comparison_artifacts(root, eval_mode="end_to_end")

        self.assertEqual(len(comparison["rows"]), 1)
        self.assertEqual(comparison["rows"][0]["run_name"], "flat_matrix")

    def test_official_filter_and_ablation_rows_exclude_non_official_runs(self):
        good_manifest = {
            "eval_mode": "end_to_end",
            "strict_components": True,
            "strict_end2end": True,
            "oracle_mode": False,
            "deprecated_alias": False,
            "run_integrity_for_official_metrics": True,
            "native_prior_uses_exact_gt": False,
            "native_anyvisloc_schema_ok": True,
        }
        good_metrics = {
            "M-ME": 1.0,
            "HR": 0.5,
            "E-ME": 2.0,
            "E_alt": 3.0,
            "Recall@1": 0.25,
            "A@10m": 0.75,
            "mean_inliers": 9.0,
            "mean_shape_confidence": 0.8,
        }
        self.assertTrue(br.is_official_metrics_payload(good_manifest, good_metrics))
        self.assertFalse(br.is_official_metrics_payload({**good_manifest, "oracle_mode": True}, good_metrics))
        self.assertFalse(br.is_official_metrics_payload({**good_manifest, "deprecated_alias": True}, good_metrics))
        self.assertFalse(br.is_official_metrics_payload({**good_manifest, "native_prior_uses_exact_gt": True}, good_metrics))
        self.assertFalse(br.is_official_metrics_payload({**good_manifest, "native_anyvisloc_schema_ok": False}, good_metrics))
        self.assertFalse(br.is_official_metrics_payload({**good_manifest, "run_integrity_for_official_metrics": False}, good_metrics))
        self.assertFalse(br.is_official_metrics_payload({**good_manifest, "retrieval_protocol_violation_count": 1}, good_metrics))
        # Benign over-fetch filtering must NOT disqualify a run (see above).
        self.assertTrue(br.is_official_metrics_payload({**good_manifest, "retrieval_protocol_filtered_candidates_count": 1}, good_metrics))
        self.assertFalse(br.is_official_metrics_payload({**good_manifest, "comparison_tier": "hybrid_fair_6dof"}, good_metrics))
        self.assertFalse(br.is_official_metrics_payload({**good_manifest, "comparison_tier": "native_reproduction"}, good_metrics))
        self.assertFalse(br.is_official_metrics_payload({**good_manifest, "is_native": True}, good_metrics))
        self.assertFalse(br.is_official_metrics_payload({**good_manifest, "native_baseline": True}, good_metrics))
        self.assertFalse(br.is_official_metrics_payload({**good_manifest, "is_hybrid": True}, good_metrics))

        all_results = {
            "avl": {
                "ours": {"paper_metrics": good_metrics, "run_manifest": good_manifest},
                "anyvisloc_e2e": {
                    "paper_metrics": {**good_metrics, "deprecated_alias": True},
                    "run_manifest": {**good_manifest, "deprecated_alias": True},
                },
                "native_anyvisloc_e2e": {
                    "paper_metrics": {**good_metrics, "native_anyvisloc_schema_ok": False},
                    "run_manifest": {**good_manifest, "native_anyvisloc_schema_ok": False},
                },
                "ortholoc_2d3d_oracle": {
                    "paper_metrics": {**good_metrics, "oracle_mode": True},
                    "run_manifest": {**good_manifest, "oracle_mode": True},
                },
            }
        }
        rows = br.build_ablation_paper_rows_for_dataset(
            all_results,
            "avl",
            ["ours", "anyvisloc_e2e", "native_anyvisloc_e2e", "ortholoc_2d3d_oracle"],
        )
        self.assertEqual([row["method"] for row in rows], [br.PIPELINE_CONFIGS["ours"]["name"]])

    def test_uav_anchor_audit_checks_query_bbox_and_nearest_tile_floor(self):
        refs = [
            {"ref_name": "01/a.png", "anchor": {"lat": 10.0, "lon": 106.0}},
            {"ref_name": "01/b.png", "anchor": {"lat": 10.001, "lon": 106.0}},
        ]
        queries = [
            {"image_name": "01/q_ok.jpg", "gt_pose": {"lat": 10.0001, "lon": 106.0}},
            {"image_name": "01/q_bad.jpg", "gt_pose": {"lat": 11.0, "lon": 106.0}},
        ]
        bboxes = {"01": {"min_lat": 9.9, "max_lat": 10.1, "min_lon": 105.9, "max_lon": 106.1}}

        audit = br.audit_uav_visloc_reference_anchors(
            refs,
            bboxes_by_flight=bboxes,
            query_rows=queries,
        )

        self.assertFalse(audit["passed"])
        self.assertEqual(audit["query_bbox_checked_count"], 2)
        self.assertEqual(audit["query_out_of_bbox_count"], 1)
        self.assertGreater(audit["nearest_tile_oracle_floor_m_summary"]["count"], 0)
        self.assertIsNotNone(audit["nearest_tile_oracle_floor_m_summary"]["p50"])

    def test_native_anyvisloc_cache_manifest_is_truthful_when_cache_not_implemented(self):
        runner = br.BenchmarkRunner(
            pipeline_name="native_anyvisloc_e2e",
            output_dir=tempfile.mkdtemp(),
            eval_mode="end_to_end",
            strict_end2end=True,
            verbose=False,
        )
        runner._stats["native_anyvisloc_cache_requested"] = True
        runner._stats["native_anyvisloc_cache_enabled"] = False
        runner._stats["native_anyvisloc_cache_key"] = "abc123"
        runner._stats["native_anyvisloc_cache_hit"] = False
        runner._stats["native_anyvisloc_cache_rebuild_reason"] = "not_implemented"

        manifest = runner._build_run_manifest_payload("avl")

        self.assertTrue(manifest["native_anyvisloc_cache_requested"])
        self.assertFalse(manifest["native_anyvisloc_cache_enabled"])
        self.assertEqual(manifest["native_anyvisloc_cache_key"], "abc123")
        self.assertFalse(manifest["native_anyvisloc_cache_hit"])
        self.assertEqual(manifest["native_anyvisloc_cache_rebuild_reason"], "not_implemented")

    def test_forge_prediction_schema_defaults_are_backward_compatible(self):
        required = {
            "adversarial_mode",
            "adversarial_condition",
            "adversarial_offset_m",
            "adversarial_candidate_policy",
            "adversarial_candidate_rank",
            "adversarial_forced_reference_name",
            "adversarial_original_reference_name",
            "is_adversarial",
            "is_wrong_reference",
            "bad_pose_label",
            "unsafe_accept",
            "inlier_ratio",
            "pnp_inlier_ratio",
            "retrieval_score_top1",
            "retrieval_score_top2",
            "retrieval_score_margin",
            "caves_risk_score",
            "caves_confidence",
            "caves_decision",
            "obs_gate_score",
            "obs_gate_reason",
        }
        self.assertTrue(required.issubset(set(br.PREDICTIONS_CSV_COLUMNS)))
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "predictions.csv"
            br.write_predictions_csv(path, [{"image_name": "a.jpg", "success": True, "num_inliers": 5, "num_matches": 10}])
            with path.open(newline="") as f:
                row = next(csv.DictReader(f))
        self.assertEqual(row["adversarial_mode"], "none")
        self.assertEqual(row["adversarial_condition"], "clean")
        self.assertIn(row["is_adversarial"].lower(), {"false", "0"})
        self.assertAlmostEqual(float(row["inlier_ratio"]), 0.5)

    def test_forge_payloads_are_not_official_metrics_eligible(self):
        clean_manifest = {
            "eval_mode": "end_to_end",
            "strict_components": True,
            "strict_end2end": True,
            "comparison_tier": "strict_e2e",
            "run_integrity_for_official_metrics": True,
            "retrieval_protocol_violation_count": 0,
            "retrieval_protocol_filtered_candidates_count": 0,
        }
        self.assertTrue(br.is_official_metrics_payload(clean_manifest, {}))
        forge_manifest = {
            **clean_manifest,
            "benchmark_family": "forge",
            "adversarial_mode": "wrong_reference",
            "adversarial_diagnostic": True,
            "official_metric_eligible": False,
        }
        self.assertFalse(br.is_official_metrics_payload(forge_manifest, {}))


if __name__ == "__main__":
    unittest.main()
