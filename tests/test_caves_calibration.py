import csv
import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts.reporting import evaluate_caves_calibration as eval_caves
from uav_geolocalization.benchmark import caves
from uav_geolocalization.cli.benchmark_runner_impl import (
    _apply_reliability_schema_defaults,
    _finalize_forge_labels,
)


class CavesCalibrationTest(unittest.TestCase):
    def test_choose_threshold_respects_abr_gate(self):
        labels = [0] * 50 + [1] * 50
        scores = [0.05 + 0.002 * i for i in range(50)] + [0.90 + 0.001 * i for i in range(50)]
        selection = eval_caves.choose_threshold(
            labels,
            scores,
            min_coverage=0.7,
            default_threshold=0.5,
            max_accepted_bad_rate=0.25,
        )
        self.assertFalse(selection.safety_gate_infeasible)
        metrics = eval_caves._threshold_metrics_at(
            list(zip(labels, scores)),
            selection.threshold,
        )
        self.assertLessEqual(metrics["accepted_bad_rate"], 0.25)
        self.assertLess(metrics["coverage"], 0.7)

    def test_choose_threshold_infeasible_marks_flag(self):
        labels = [1] * 10
        scores = [0.1 * (i + 1) for i in range(10)]
        selection = eval_caves.choose_threshold(
            labels,
            scores,
            min_coverage=0.7,
            default_threshold=0.5,
            max_accepted_bad_rate=0.25,
        )
        self.assertTrue(selection.safety_gate_infeasible)
        self.assertEqual(selection.threshold, 0.5)

    def test_safe_bool_handles_string_false_trap(self):
        cases_true = [True, 1, 1.0, "1", "true", "TRUE", "yes", " y "]
        cases_false = [False, 0, 0.0, "0", "false", "FALSE", "no", "", None, "maybe"]
        for value in cases_true:
            self.assertTrue(caves._safe_bool(value), msg=repr(value))
        for value in cases_false:
            self.assertFalse(caves._safe_bool(value), msg=repr(value))

    def test_describe_caves_v0_components_exports_expected_auditable_weights(self):
        components = caves.describe_caves_v0_components()
        by_name = {component["name"]: component for component in components}
        expected_weights = {
            "low_inlier_ratio": 1.5,
            "low_inlier_count": 1.0,
            "high_reprojection_residual": 1.2,
            "low_shape_confidence": 1.0,
            "low_retrieval_margin": 0.7,
            "late_retrieval_rank": 0.4,
            "large_anchor_offset": 1.0,
            "final_pose_gate_reject": 1.5,
            "retrieval_protocol_violation": 2.0,
        }
        self.assertEqual(set(by_name), set(expected_weights))
        for name, weight in expected_weights.items():
            self.assertAlmostEqual(by_name[name]["weight"], weight, msg=name)
            self.assertIsInstance(by_name[name]["source_fields"], (list, tuple))
            self.assertIn("formula_note", by_name[name])
            self.assertIn("label_like_fields_excluded", by_name[name])

    def test_describe_caves_v0_components_excludes_label_like_source_fields(self):
        label_like_fields = {
            "post_fgo_error_m",
            "bad_pose_label",
            "unsafe_accept",
            "accepted_for_metrics",
            "localization_is_success",
        }
        for component in caves.describe_caves_v0_components():
            self.assertFalse(label_like_fields.intersection(component["source_fields"]), msg=component["name"])

    def test_describe_caves_v0_components_excludes_forbidden_score_inputs(self):
        forbidden = set(caves.CAVES_V0_FORBIDDEN_SCORE_INPUT_FIELDS)
        for component in caves.describe_caves_v0_components():
            self.assertFalse(forbidden.intersection(component["source_fields"]), msg=component["name"])

    def test_geometry_only_ignores_all_retrieval_and_protocol_fields(self):
        base = {
            "num_matches": 80,
            "num_inliers": 40,
            "inlier_ratio": 0.5,
            "reproj_residual_px": 4.0,
            "shape_confidence": 0.7,
            "geometry_anchor_offset_m": 35.0,
            "final_pose_gate_reject": False,
        }
        low_retrieval_risk = {
            **base,
            "retrieval_score_margin": 0.25,
            "retrieval_rank": 1,
            "candidate_rank_selected": 1,
            "retrieval_protocol_violation": False,
        }
        high_retrieval_risk = {
            **base,
            "retrieval_score_margin": 0.0,
            "retrieval_rank": 99,
            "candidate_rank_selected": 99,
            "retrieval_protocol_violation": True,
        }

        self.assertEqual(
            caves.score_row_for_feature_set(low_retrieval_risk, "geometry_only"),
            caves.score_row_for_feature_set(high_retrieval_risk, "geometry_only"),
        )

    def test_compute_caves_v0_ignores_adversarial_offset_when_geometry_anchor_missing(self):
        base = {
            "num_matches": 80,
            "num_inliers": 40,
            "inlier_ratio": 0.5,
            "reproj_residual_px": 4.0,
            "shape_confidence": 0.7,
            "retrieval_score_margin": 0.12,
            "retrieval_rank": 2,
            "final_pose_gate_reject": False,
            "retrieval_protocol_violation": False,
        }
        no_anchor = dict(base)
        adversarial_only = {**base, "adversarial_offset_m": 200.0}

        self.assertAlmostEqual(
            caves.compute_caves_v0(no_anchor).risk_score,
            caves.compute_caves_v0(adversarial_only).risk_score,
            places=12,
        )

    def test_compute_caves_v0_prefers_geometry_anchor_and_ignores_adversarial_offset(self):
        base = {
            "num_matches": 80,
            "num_inliers": 40,
            "inlier_ratio": 0.5,
            "reproj_residual_px": 4.0,
            "shape_confidence": 0.7,
            "retrieval_score_margin": 0.12,
            "retrieval_rank": 2,
            "geometry_anchor_offset_m": 35.0,
            "final_pose_gate_reject": False,
            "retrieval_protocol_violation": False,
        }
        low_adv = {**base, "adversarial_offset_m": 0.0}
        high_adv = {**base, "adversarial_offset_m": 1000.0}

        self.assertAlmostEqual(
            caves.compute_caves_v0(low_adv).risk_score,
            caves.compute_caves_v0(high_adv).risk_score,
            places=12,
        )

    def test_score_row_for_feature_set_ignores_label_fields(self):
        runtime_features = {
            "num_matches": 80,
            "num_inliers": 40,
            "inlier_ratio": 0.5,
            "reproj_residual_px": 4.0,
            "shape_confidence": 0.7,
            "retrieval_score_margin": 0.12,
            "retrieval_rank": 2,
            "geometry_anchor_offset_m": 35.0,
            "final_pose_gate_reject": False,
            "caves_risk_score": 0.35,
        }
        labels_a = {
            **runtime_features,
            "localization_is_success": True,
            "bad_pose_label": False,
            "false_reject": False,
            "unsafe_accept": False,
            "post_fgo_error_m": 10.0,
            "accepted_for_metrics": True,
        }
        labels_b = {
            **runtime_features,
            "localization_is_success": False,
            "bad_pose_label": True,
            "false_reject": True,
            "unsafe_accept": True,
            "post_fgo_error_m": 100.0,
            "accepted_for_metrics": False,
        }
        for feature_set in [
            "inlier_ratio_only",
            "retrieval_margin_only",
            "retrieval_margin_plus_rank",
            "geometry_only",
            "full_caves",
            "full_without_rank",
            "gate_only",
        ]:
            self.assertEqual(
                caves.score_row_for_feature_set(labels_a, feature_set),
                caves.score_row_for_feature_set(labels_b, feature_set),
                msg=feature_set,
            )

    def test_gate_only_uses_final_pose_gate_not_labels(self):
        self.assertEqual(caves.score_row_for_feature_set({"final_pose_gate_reject": True}, "gate_only"), 1.0)
        self.assertEqual(caves.score_row_for_feature_set({"final_pose_gate_reject": False}, "gate_only"), 0.0)
        self.assertEqual(
            caves.score_row_for_feature_set({"failure_reason": "final_pose_gate_reject"}, "gate_only"),
            1.0,
        )

    def test_summarize_caves_metrics_reports_forge_rejection_rates(self):
        labels = [0, 0, 1, 1]
        scores = [0.1, 0.8, 0.9, 0.2]
        metrics = caves.summarize_caves_metrics(labels, scores, threshold=0.5)
        self.assertAlmostEqual(metrics["coverage"], 0.5)
        self.assertAlmostEqual(metrics["accepted_bad_rate"], 0.5)
        self.assertAlmostEqual(metrics["false_reject_rate"], 0.5)
        self.assertAlmostEqual(metrics["safe_reject_rate"], 0.5)
        self.assertAlmostEqual(metrics["forge_true_reject_rate"], 0.5)
        self.assertAlmostEqual(metrics["forge_false_reject_rate"], 0.5)

    def test_summarize_caves_metrics_zero_denominator_returns_nan(self):
        all_good = caves.summarize_caves_metrics([0, 0], [0.1, 0.2], threshold=0.5)
        self.assertTrue(math.isnan(all_good["safe_reject_rate"]))
        self.assertTrue(math.isnan(all_good["forge_true_reject_rate"]))
        all_bad = caves.summarize_caves_metrics([1, 1], [0.8, 0.9], threshold=0.5)
        self.assertTrue(math.isnan(all_bad["forge_false_reject_rate"]))

    def test_random_baseline_score_is_deterministic_and_label_invariant(self):
        row = {
            "dataset": "avl",
            "pipeline": "ours",
            "image_name": "q1",
            "artifact_run_dir": "/tmp/run",
            "candidate_rank_selected": 0,
            "bad_pose_label": False,
        }
        score_a = eval_caves.random_baseline_score(row, seed=0)
        score_b = eval_caves.random_baseline_score(row, seed=0)
        self.assertEqual(score_a, score_b)
        row_alt = dict(row)
        row_alt.update({
            "bad_pose_label": True,
            "localization_is_success": False,
            "false_reject": True,
            "unsafe_accept": True,
        })
        self.assertEqual(score_a, eval_caves.random_baseline_score(row_alt, seed=0))

    def test_compute_feature_set_metrics_includes_baselines(self):
        rows = [
            {"bad_pose_label": False, "inlier_ratio": 0.9, "caves_risk_score": 0.1, "final_pose_gate_reject": False},
            {"bad_pose_label": False, "inlier_ratio": 0.8, "caves_risk_score": 0.2, "final_pose_gate_reject": False},
            {"bad_pose_label": True, "inlier_ratio": 0.2, "caves_risk_score": 0.8, "final_pose_gate_reject": True},
            {"bad_pose_label": True, "inlier_ratio": 0.1, "caves_risk_score": 0.9, "final_pose_gate_reject": True},
        ]
        ablation_rows, _summary = eval_caves.compute_feature_set_metrics(
            valid_rows=rows,
            calib_rows=rows,
            heldout_rows=rows,
            feature_sets=["random_baseline", "gate_only"],
            default_threshold=0.5,
            min_coverage=0.5,
            n_bins=5,
            bootstrap=20,
            bootstrap_seed=1,
        )
        feature_sets = {row["feature_set"] for row in ablation_rows}
        self.assertIn("random_baseline", feature_sets)
        self.assertIn("gate_only", feature_sets)

    def test_temperature_calibrator_predictions_are_bounded(self):
        labels = [0, 0, 0, 1, 1, 1]
        scores = [0.05, 0.20, 0.35, 0.55, 0.70, 0.95]
        calibrator = caves.fit_temperature_calibrator(labels, scores)
        calibrated = calibrator.predict(scores)
        self.assertTrue(all(0.0 <= value <= 1.0 for value in calibrated))
        self.assertTrue(math.isfinite(calibrator.temperature))

    def test_temperature_calibrator_handles_constant_scores(self):
        labels = [0, 0, 1, 1]
        scores = [0.5, 0.5, 0.5, 0.5]
        calibrator = caves.fit_temperature_calibrator(labels, scores)
        calibrated = calibrator.predict(scores)
        self.assertEqual(len(calibrated), 4)
        self.assertTrue(all(math.isfinite(value) for value in calibrated))

    def test_reporter_marks_one_class_temperature_as_insufficient(self):
        rows = [
            {"bad_pose_label": False, "caves_risk_score": 0.1},
            {"bad_pose_label": False, "caves_risk_score": 0.2},
        ]
        ablation_rows, summary = eval_caves.compute_feature_set_metrics(
            valid_rows=rows,
            calib_rows=rows,
            heldout_rows=rows,
            feature_sets=["full_caves"],
            default_threshold=0.5,
            min_coverage=0.5,
            n_bins=5,
            bootstrap=10,
            bootstrap_seed=1,
        )
        self.assertEqual(summary["full_caves"]["calibration_status"], "insufficient_calibration_data")
        temp_rows = [row for row in ablation_rows if row["score_type"] == "temperature_calibrated"]
        self.assertTrue(all(row["calibration_status"] == "insufficient_calibration_data" for row in temp_rows))

    def test_split_name_is_deterministic_for_identity_fields(self):
        row = {
            "dataset": "avl",
            "pipeline": "ours",
            "image_name": "q1",
            "artifact_run_dir": "/tmp/run",
        }
        self.assertEqual(
            eval_caves.split_name(row, calibration_percent=50),
            eval_caves.split_name(dict(row), calibration_percent=50),
        )

    def test_group_key_is_stable_for_identity_fields(self):
        row = {
            "artifact_run_dir": "/tmp/run",
            "dataset": "avl",
            "pipeline": "ours",
            "image_name": "q1",
        }
        self.assertEqual(
            eval_caves.group_key(row, fallback_index=0),
            eval_caves.group_key(dict(row), fallback_index=99),
        )

    def test_split_identity_overlap_audit_detects_leakage(self):
        calib = [{"dataset": "avl", "pipeline": "ours", "image_name": "q1", "artifact_run_dir": "/tmp/a"}]
        heldout = [{"dataset": "avl", "pipeline": "ours", "image_name": "q2", "artifact_run_dir": "/tmp/a"}]
        clean = eval_caves.compute_split_identity_leakage_audit(calib, heldout)
        self.assertEqual(clean["split_identity_overlap_count"], 0)
        leaked = eval_caves.compute_split_identity_leakage_audit(calib, calib)
        self.assertGreater(leaked["split_identity_overlap_count"], 0)
        self.assertEqual(leaked["split_identity_key_fields"], eval_caves.SPLIT_IDENTITY_KEY_FIELDS)

    def test_ece_sensitivity_sweep_reports_requested_bins_and_score_types(self):
        calib_rows = [
            {"bad_pose_label": False, "caves_risk_score": 0.05},
            {"bad_pose_label": False, "caves_risk_score": 0.20},
            {"bad_pose_label": True, "caves_risk_score": 0.80},
            {"bad_pose_label": True, "caves_risk_score": 0.95},
        ]
        heldout_rows = [
            {"bad_pose_label": False, "caves_risk_score": 0.10},
            {"bad_pose_label": False, "caves_risk_score": 0.30},
            {"bad_pose_label": True, "caves_risk_score": 0.70},
            {"bad_pose_label": True, "caves_risk_score": 0.90},
        ]
        rows = eval_caves.compute_ece_sensitivity_rows(
            calib_rows,
            heldout_rows,
            bootstrap_seed=7,
        )
        self.assertEqual({row["n_bins"] for row in rows}, {8, 10, 12, 15})
        self.assertEqual({row["score_type"] for row in rows}, {"raw", "isotonic_calibrated"})
        self.assertTrue(all(row["feature_set"] == "full_caves" and row["split"] == "heldout" for row in rows))

    def test_compute_split_balance_reports_warnings(self):
        rows = [
            {"dataset": "avl", "caves_split": "calibration"},
            {"dataset": "avl", "caves_split": "calibration"},
            {"dataset": "vpair", "caves_split": "heldout"},
        ]
        balance = eval_caves.compute_split_balance(rows)
        self.assertTrue(any(row["condition_column"] == "dataset" for row in balance))
        avl = next(row for row in balance if row["condition_value"] == "avl")
        self.assertIn("missing_from_heldout", avl["warning"])
        vpair = next(row for row in balance if row["condition_value"] == "vpair")
        self.assertIn("missing_from_calibration", vpair["warning"])

    def test_per_condition_metrics_correct(self):
        strong_geometry = {
            "num_matches": 100,
            "num_inliers": 80,
            "reproj_residual_px": 2.0,
            "shape_confidence": 0.95,
            "retrieval_score_margin": 0.4,
            "geometry_anchor_offset_m": 5.0,
        }
        weak_geometry = {
            "num_matches": 10,
            "num_inliers": 1,
            "reproj_residual_px": 25.0,
            "shape_confidence": 0.1,
            "retrieval_score_margin": 0.0,
            "geometry_anchor_offset_m": 200.0,
        }
        rows = [
            {
                "adversarial_condition": "wrong_far",
                "bad_pose_label": True,
                **weak_geometry,
            },
            {
                "adversarial_condition": "wrong_far",
                "bad_pose_label": True,
                **strong_geometry,
            },
            {
                "adversarial_condition": "wrong_near",
                "bad_pose_label": False,
                **strong_geometry,
            },
            {
                "adversarial_condition": "wrong_near",
                "bad_pose_label": False,
                **weak_geometry,
            },
        ]
        results = eval_caves.compute_per_condition_metrics(
            rows,
            feature_set="full_caves",
            threshold=0.5,
            random_seed=0,
        )
        wrong_far_row = next(r for r in results if r["condition"] == "wrong_far")
        self.assertAlmostEqual(wrong_far_row["base_bad_rate"], 1.0)
        self.assertAlmostEqual(wrong_far_row["coverage"], 0.5)

    def test_caves_v0_populates_reject_for_weak_geometry(self):
        row = {
            "num_matches": 10,
            "num_inliers": 1,
            "reproj_residual_px": 25.0,
            "shape_confidence": 0.1,
            "retrieval_score_margin": 0.0,
            "geometry_anchor_offset_m": 200.0,
        }
        caves.apply_caves_v0(row, reject_threshold=0.5)
        self.assertGreaterEqual(row["caves_risk_score"], 0.5)
        self.assertEqual(row["caves_decision"], caves.CAVES_REJECT)
        self.assertGreater(row["caves_evidence_count"], 0)

    def test_caves_v0_accepts_strong_geometry(self):
        row = {
            "num_matches": 100,
            "num_inliers": 80,
            "reproj_residual_px": 2.0,
            "shape_confidence": 0.95,
            "retrieval_score_margin": 0.4,
            "geometry_anchor_offset_m": 5.0,
        }
        caves.apply_caves_v0(row, reject_threshold=0.5)
        self.assertLess(row["caves_risk_score"], 0.5)
        self.assertEqual(row["caves_decision"], caves.CAVES_ACCEPT)

    def test_auc_ap_brier_ece_are_computed_without_sklearn(self):
        labels = [0, 0, 1, 1]
        scores = [0.1, 0.2, 0.8, 0.9]
        self.assertAlmostEqual(caves.roc_auc_score(labels, scores), 1.0)
        self.assertAlmostEqual(caves.average_precision_score(labels, scores), 1.0)
        self.assertAlmostEqual(caves.brier_score(labels, scores), (0.01 + 0.04 + 0.04 + 0.01) / 4.0)
        self.assertGreaterEqual(caves.expected_calibration_error(labels, scores, n_bins=2), 0.0)

    def test_risk_coverage_curve_sorts_by_low_risk_first(self):
        labels = [1, 0, 1, 0]
        scores = [0.9, 0.1, 0.8, 0.2]
        curve = caves.risk_coverage_curve(labels, scores)
        self.assertEqual(len(curve), 4)
        self.assertAlmostEqual(curve[0]["coverage"], 0.25)
        self.assertAlmostEqual(curve[0]["risk"], 0.0)
        self.assertAlmostEqual(curve[-1]["risk"], 0.5)

    def test_rows_to_labels_scores_computes_missing_scores(self):
        rows = [
            {"bad_pose_label": False, "num_matches": 100, "num_inliers": 80, "shape_confidence": 0.9},
            {"bad_pose_label": True, "num_matches": 5, "num_inliers": 0, "shape_confidence": 0.1},
        ]
        labels, scores = caves.rows_to_labels_scores(rows)
        self.assertEqual(labels, [0, 1])
        self.assertEqual(len(scores), 2)
        self.assertTrue(all(math.isfinite(score) for score in scores))

    def test_retrieval_rank_feature_increases_risk_for_later_rank(self):
        base = {
            "num_matches": 100,
            "num_inliers": 80,
            "reproj_residual_px": 2.0,
            "shape_confidence": 0.95,
            "retrieval_score_margin": 0.4,
            "geometry_anchor_offset_m": 5.0,
        }
        top1 = dict(base, retrieval_rank=1)
        rank5 = dict(base, retrieval_rank=5)
        self.assertGreater(caves.compute_caves_v0(rank5).risk_score, caves.compute_caves_v0(top1).risk_score)

    def test_outcome_labels_track_false_reject_and_unsafe_severity(self):
        good_rejected = {
            "post_fgo_error_m": 10.0,
            "accepted_for_metrics": False,
            "caves_decision": caves.CAVES_REJECT,
        }
        caves.apply_caves_outcome_labels(good_rejected, localization_success_threshold_m=20.0, retrieval_correct_threshold_m=20.0)
        self.assertTrue(good_rejected["localization_is_success"])
        self.assertTrue(good_rejected["caves_reject"])
        self.assertTrue(good_rejected["false_reject"])
        self.assertFalse(good_rejected["unsafe_accept"])
        self.assertAlmostEqual(good_rejected["unsafe_accept_severity_m"], 0.0)

        bad_accepted = {
            "post_fgo_error_m": 80.0,
            "accepted_for_metrics": True,
            "caves_decision": caves.CAVES_ACCEPT,
        }
        caves.apply_caves_outcome_labels(bad_accepted, localization_success_threshold_m=20.0, retrieval_correct_threshold_m=20.0)
        self.assertFalse(bad_accepted["localization_is_success"])
        self.assertFalse(bad_accepted["false_reject"])
        self.assertTrue(bad_accepted["unsafe_accept"])
        self.assertAlmostEqual(bad_accepted["unsafe_accept_severity_m"], 60.0)

    def test_isotonic_calibrator_is_monotone_and_can_reduce_brier(self):
        labels = [0, 0, 0, 1, 1, 1]
        scores = [0.05, 0.20, 0.35, 0.55, 0.70, 0.95]
        calibrator = caves.fit_isotonic_calibrator(labels, scores)
        calibrated = calibrator.predict(scores)
        self.assertEqual(sorted(calibrated), list(calibrated))
        self.assertLessEqual(caves.brier_score(labels, calibrated), caves.brier_score(labels, scores))

    def test_bootstrap_auroc_ci_contains_point_estimate(self):
        labels = [0, 0, 1, 1, 0, 1]
        scores = [0.1, 0.2, 0.8, 0.9, 0.3, 0.7]
        ci = caves.bootstrap_auroc_ci(labels, scores, n_bootstrap=100, seed=7)
        self.assertLessEqual(ci["auroc_ci_low"], ci["auroc"])
        self.assertGreaterEqual(ci["auroc_ci_high"], ci["auroc"])
        self.assertGreater(ci["auroc_bootstrap_valid"], 0)

    def test_feature_set_scores_support_caves_ablation(self):
        rows = [
            {"bad_pose_label": False, "inlier_ratio": 0.8, "retrieval_score_margin": 0.4, "retrieval_rank": 1},
            {"bad_pose_label": True, "inlier_ratio": 0.1, "retrieval_score_margin": 0.0, "retrieval_rank": 5},
        ]
        labels, scores = caves.rows_to_labels_scores(rows, feature_set="retrieval_margin_plus_rank")
        self.assertEqual(labels, [0, 1])
        self.assertLess(scores[0], scores[1])

    def test_ablation_interpretation_notes_flag_inverted_inlier_ratio(self):
        notes = eval_caves.ablation_interpretation_notes([
            {
                "feature_set": "inlier_ratio_only",
                "split": "heldout",
                "score_type": "raw",
                "auroc": 0.377,
            }
        ])

        self.assertTrue(notes)
        self.assertIn("inlier_ratio_only", notes[0])
        self.assertIn("not a safe risk proxy", notes[0])

    def test_reporter_feature_set_metrics_include_raw_isotonic_and_ci(self):
        rows = [
            {"bad_pose_label": False, "inlier_ratio": 0.9, "retrieval_score_margin": 0.4, "retrieval_rank": 1, "caves_risk_score": 0.05},
            {"bad_pose_label": False, "inlier_ratio": 0.7, "retrieval_score_margin": 0.3, "retrieval_rank": 1, "caves_risk_score": 0.20},
            {"bad_pose_label": False, "inlier_ratio": 0.6, "retrieval_score_margin": 0.2, "retrieval_rank": 2, "caves_risk_score": 0.35},
            {"bad_pose_label": True, "inlier_ratio": 0.2, "retrieval_score_margin": 0.05, "retrieval_rank": 4, "caves_risk_score": 0.75},
            {"bad_pose_label": True, "inlier_ratio": 0.1, "retrieval_score_margin": 0.0, "retrieval_rank": 5, "caves_risk_score": 0.90},
            {"bad_pose_label": True, "inlier_ratio": 0.0, "retrieval_score_margin": 0.0, "retrieval_rank": 5, "caves_risk_score": 0.95},
        ]
        ablation_rows, summary = eval_caves.compute_feature_set_metrics(
            valid_rows=rows,
            calib_rows=rows,
            heldout_rows=rows,
            feature_sets=["inlier_ratio_only", "retrieval_margin_plus_rank", "full_caves"],
            default_threshold=0.5,
            min_coverage=0.5,
            n_bins=5,
            bootstrap=50,
            bootstrap_seed=3,
        )
        self.assertIn("full_caves", summary)
        full_heldout = [
            row for row in ablation_rows
            if row["feature_set"] == "full_caves"
            and row["split"] == "heldout"
            and row["score_type"] == "isotonic_calibrated"
        ]
        self.assertEqual(len(full_heldout), 1)
        self.assertEqual(full_heldout[0]["calibration_status"], "ok")
        self.assertIn("auroc_ci_low", full_heldout[0])
        self.assertGreater(full_heldout[0]["auroc_bootstrap_valid"], 0)

    def test_anchor_provenance_audit_rows_counts_missing_geometry_anchor(self):
        rows = [
            {
                "dataset": "avl",
                "pipeline": "ours",
                "adversarial_condition": "wrong_near",
                "geometry_anchor_offset_m": "30",
                "adversarial_offset_m": "100",
            },
            {
                "dataset": "avl",
                "pipeline": "ours",
                "adversarial_condition": "wrong_near",
                "geometry_anchor_offset_m": "",
                "adversarial_offset_m": "100",
            },
            {
                "dataset": "avl",
                "pipeline": "ours",
                "adversarial_condition": "wrong_near",
                "geometry_anchor_offset_m": "",
                "adversarial_offset_m": "",
            },
        ]

        audit_rows = eval_caves.compute_anchor_provenance_audit_rows(rows, split_name="heldout")

        self.assertEqual(len(audit_rows), 1)
        row = audit_rows[0]
        self.assertEqual(row["split"], "heldout")
        self.assertEqual(row["dataset"], "avl")
        self.assertEqual(row["pipeline"], "ours")
        self.assertEqual(row["condition"], "wrong_near")
        self.assertEqual(row["n_rows"], 3)
        self.assertEqual(row["n_geometry_anchor_present"], 1)
        self.assertEqual(row["n_geometry_anchor_missing"], 2)
        self.assertAlmostEqual(row["geometry_anchor_missing_rate"], 2 / 3)
        self.assertEqual(row["n_adversarial_offset_present"], 2)
        self.assertEqual(row["n_rows_with_only_adversarial_offset"], 1)
        self.assertEqual(row["n_rows_with_both_geometry_and_adversarial"], 1)
        self.assertEqual(row["n_rows_with_neither"], 1)
        self.assertEqual(row["gate_status"], "FAIL")
        self.assertEqual(row["gate_reason"], "rows_have_only_adversarial_offset_metadata")

    def test_anchor_provenance_audit_rows_pass_when_geometry_anchor_present_or_omitted(self):
        rows = [
            {
                "artifact_dataset": "vpair",
                "artifact_pipeline": "ours",
                "adversarial_condition": "wrong_far",
                "geometry_anchor_offset_m": "40",
                "adversarial_offset_m": "100",
            },
            {
                "artifact_dataset": "vpair",
                "artifact_pipeline": "ours",
                "adversarial_condition": "wrong_far",
                "geometry_anchor_offset_m": "",
                "adversarial_offset_m": "",
            },
        ]

        audit_rows = eval_caves.compute_anchor_provenance_audit_rows(rows, split_name="calibration")

        self.assertEqual(len(audit_rows), 1)
        self.assertEqual(audit_rows[0]["gate_status"], "PASS")
        self.assertEqual(audit_rows[0]["n_rows_with_only_adversarial_offset"], 0)

    def test_resolve_baseline_uses_only_forge_rows_with_acceptance_semantics(self):
        rows = [
            {
                "image_name": "forge_bad_accepted",
                "is_adversarial": "1",
                "is_wrong_reference": "1",
                "adversarial_condition": "wrong_near",
                "post_fgo_error_m": "80",
                "accepted_for_metrics": "1",
                "bad_pose_label": "1",
            },
            {
                "image_name": "forge_bad_rejected",
                "is_adversarial": "1",
                "is_wrong_reference": "1",
                "adversarial_condition": "wrong_far",
                "post_fgo_error_m": "90",
                "accepted_for_metrics": "0",
                "bad_pose_label": "1",
            },
            {
                "image_name": "forge_good_accepted",
                "is_adversarial": "1",
                "is_wrong_reference": "1",
                "adversarial_condition": "visually_similar_wrong",
                "post_fgo_error_m": "10",
                "accepted_for_metrics": "1",
                "bad_pose_label": "0",
            },
            {
                "image_name": "clean_bad_accepted_ignored",
                "post_fgo_error_m": "80",
                "accepted_for_metrics": "1",
                "bad_pose_label": "1",
            },
        ]

        audit = eval_caves.resolve_baseline_accepted_bad_rate(rows)

        self.assertEqual(audit["baseline_source"], "computed_from_forge_rows")
        self.assertEqual(audit["eligible_forge_rows"], 3)
        self.assertEqual(audit["usable_acceptance_rows"], 3)
        self.assertEqual(audit["unsafe_accept_rows"], 1)
        self.assertAlmostEqual(audit["baseline_accepted_bad_rate"], 1 / 3)

    def test_resolve_baseline_counts_nan_fallback_bad_pose_labels(self):
        rows = [
            {
                "image_name": "fallback_bad_pose",
                "is_adversarial": "1",
                "is_wrong_reference": "1",
                "adversarial_condition": "wrong_far",
                "post_fgo_error_m": "",
                "position_error_m": "",
                "accepted_for_metrics": "1",
                "bad_pose_label": "1",
            },
            {
                "image_name": "finite_bad_pose",
                "is_adversarial": "1",
                "is_wrong_reference": "1",
                "adversarial_condition": "wrong_far",
                "post_fgo_error_m": "90",
                "accepted_for_metrics": "0",
                "bad_pose_label": "1",
            },
        ]

        audit = eval_caves.resolve_baseline_accepted_bad_rate(rows)

        self.assertEqual(audit["nan_fallback_bad_pose_rows"], 1)
        self.assertAlmostEqual(audit["nan_fallback_bad_pose_rate"], 0.5)

    def test_resolve_baseline_requires_usable_acceptance_semantics(self):
        rows = [
            {
                "image_name": "forge_row_without_acceptance",
                "is_adversarial": "1",
                "is_wrong_reference": "1",
                "adversarial_condition": "wrong_near",
                "post_fgo_error_m": "80",
                "bad_pose_label": "1",
            },
        ]

        with self.assertRaisesRegex(ValueError, "No FORGE rows with usable acceptance semantics"):
            eval_caves.resolve_baseline_accepted_bad_rate(rows)

    def test_resolve_baseline_accepts_explicit_override(self):
        audit = eval_caves.resolve_baseline_accepted_bad_rate([], override=0.506)

        self.assertEqual(audit["baseline_source"], "cli_override")
        self.assertAlmostEqual(audit["baseline_accepted_bad_rate"], 0.506)

    def test_reporter_marks_degenerate_isotonic_as_insufficient(self):
        rows = [
            {"bad_pose_label": False, "inlier_ratio": 0.9, "caves_risk_score": 0.1},
            {"bad_pose_label": False, "inlier_ratio": 0.8, "caves_risk_score": 0.2},
        ]
        ablation_rows, summary = eval_caves.compute_feature_set_metrics(
            valid_rows=rows,
            calib_rows=rows,
            heldout_rows=rows,
            feature_sets=["full_caves"],
            default_threshold=0.5,
            min_coverage=0.5,
            n_bins=5,
            bootstrap=10,
            bootstrap_seed=1,
        )
        self.assertEqual(summary["full_caves"]["calibration_status"], "insufficient_calibration_data")
        iso_rows = [row for row in ablation_rows if row["score_type"] == "isotonic_calibrated"]
        self.assertTrue(all(row["calibration_status"] == "insufficient_calibration_data" for row in iso_rows))

    def test_reporter_main_writes_ablation_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "avl" / "ours" / "run"
            root.mkdir(parents=True)
            pred = root / "predictions.csv"
            with pred.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "image_name",
                        "dataset",
                        "pipeline",
                        "artifact_run_dir",
                        "post_fgo_error_m",
                        "accepted_for_metrics",
                        "candidate_rank_selected",
                        "num_matches",
                        "num_inliers",
                        "reproj_residual_px",
                        "shape_confidence",
                        "retrieval_score_margin",
                    ],
                )
                writer.writeheader()
                image_names = ["q0", "q1", "q6", "q3", "q4", "q5"]
                for idx, (image_name, error) in enumerate(zip(image_names, [5, 10, 15, 80, 90, 120])):
                    writer.writerow({
                        "image_name": image_name,
                        "dataset": "avl",
                        "pipeline": "ours",
                        "artifact_run_dir": "synthetic_reporter_main",
                        "post_fgo_error_m": error,
                        "accepted_for_metrics": error < 100,
                        "candidate_rank_selected": min(idx, 4),
                        "num_matches": 100 - idx * 10,
                        "num_inliers": 80 if error < 20 else 5,
                        "reproj_residual_px": 2.0 if error < 20 else 25.0,
                        "shape_confidence": 0.9 if error < 20 else 0.1,
                        "retrieval_score_margin": 0.3 if error < 20 else 0.0,
                    })
            output = Path(tmp) / "out"
            rc = eval_caves.main([
                "--roots", str(Path(tmp)),
                "--output", str(output),
                "--bootstrap", "10",
                "--bootstrap-seed", "2",
                "--baseline-accepted-bad-rate", "0.5",
            ])
            self.assertEqual(rc, 0)
            self.assertTrue((output / "caves_ablation_metrics.csv").exists())
            self.assertTrue((output / "caves_ablation_metrics.md").exists())
            self.assertTrue((output / "caves_split_balance.csv").exists())
            self.assertTrue((output / "caves_threshold_sweep.csv").exists())
            self.assertTrue((output / "caves_component_weights.csv").exists())
            self.assertTrue((output / "caves_ece_sensitivity.csv").exists())
            self.assertTrue((output / "caves_per_condition_grouped_ci.csv").exists())
            self.assertTrue((output / "caves_anchor_provenance_audit.csv").exists())
            with (output / "caves_component_weights.csv").open(newline="", encoding="utf-8") as f:
                component_rows = list(csv.DictReader(f))
            self.assertTrue(any(row["name"] == "low_inlier_ratio" and row["source_fields"] for row in component_rows))
            with (output / "caves_threshold_sweep.csv").open(newline="", encoding="utf-8") as f:
                sweep_rows = list(csv.DictReader(f))
            self.assertGreaterEqual(len(sweep_rows), 1)
            with (output / "caves_ablation_metrics.csv").open(newline="", encoding="utf-8") as f:
                ablation_rows = list(csv.DictReader(f))
            heldout_raw = [
                row for row in ablation_rows
                if row["split"] == "heldout" and row["score_type"] == "raw"
            ]
            self.assertTrue(heldout_raw)
            for row in heldout_raw:
                self.assertNotEqual(row["auroc_ci_low"], "", msg=row["feature_set"])
                self.assertNotEqual(row["auroc_ci_high"], "", msg=row["feature_set"])
                auroc = float(row["auroc"])
                ci_low = float(row["auroc_ci_low"])
                ci_high = float(row["auroc_ci_high"])
                if row.get("calibration_status") in ("sklearn_unavailable", "insufficient_calibration_data"):
                    continue
                if auroc != auroc:
                    continue
                self.assertLessEqual(ci_low, auroc)
                self.assertLessEqual(auroc, ci_high)
            self.assertEqual(
                {row["feature_set"] for row in heldout_raw},
                set(eval_caves.DEFAULT_FEATURE_SETS),
            )
            summary = json.loads((output / "caves_calibration_metrics.json").read_text())
            self.assertEqual(summary["schema_version"], 3)
            self.assertIn("threshold_policy_max_accepted_bad_rate", summary)
            self.assertIn("safety_gate_infeasible", summary)
            self.assertIn("random_baseline", summary["feature_sets"])
            self.assertIn("gate_only", summary["feature_sets"])
            self.assertIn("ablation_metrics", summary)
            self.assertIn("split_balance", summary)
            self.assertIn("component_weights", summary)
            self.assertIn("split_identity_leakage_audit", summary)
            self.assertIn("ece_sensitivity", summary)
            self.assertIn("per_condition_grouped_ci", summary)
            self.assertIn("anchor_provenance_audit", summary)
            self.assertIn("anchor_provenance_gate_status", summary)
            self.assertTrue(any(component["name"] == "low_inlier_ratio" for component in summary["component_weights"]))
            self.assertTrue(
                any(row.get("score_type") == "temperature_calibrated" for row in summary["ablation_metrics"])
            )

    def test_reporter_writes_per_condition_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "avl" / "ours" / "run"
            root.mkdir(parents=True)
            pred = root / "predictions.csv"
            with pred.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "image_name",
                        "dataset",
                        "pipeline",
                        "artifact_run_dir",
                        "adversarial_condition",
                        "post_fgo_error_m",
                        "accepted_for_metrics",
                        "bad_pose_label",
                        "unsafe_accept",
                        "candidate_rank_selected",
                        "num_matches",
                        "num_inliers",
                        "reproj_residual_px",
                        "shape_confidence",
                        "retrieval_score_margin",
                    ],
                )
                writer.writeheader()
                for idx, (image_name, condition, error) in enumerate([
                    ("q0", "wrong_far", 80),
                    ("q1", "wrong_near", 10),
                    ("q6", "wrong_far", 90),
                    ("q3", "wrong_near", 15),
                    ("q4", "wrong_far", 85),
                    ("q5", "wrong_near", 12),
                ]):
                    bad_pose = error > 20
                    writer.writerow({
                        "image_name": image_name,
                        "dataset": "avl",
                        "pipeline": "ours",
                        "artifact_run_dir": "synthetic_per_condition",
                        "adversarial_condition": condition,
                        "post_fgo_error_m": error,
                        "accepted_for_metrics": error < 100,
                        "bad_pose_label": "1" if bad_pose else "0",
                        "unsafe_accept": "1" if bad_pose and error < 100 else "0",
                        "candidate_rank_selected": min(idx, 4),
                        "num_matches": 100 - idx * 10,
                        "num_inliers": 80 if error < 20 else 5,
                        "reproj_residual_px": 2.0 if error < 20 else 25.0,
                        "shape_confidence": 0.9 if error < 20 else 0.1,
                        "retrieval_score_margin": 0.3 if error < 20 else 0.0,
                    })
            output = Path(tmp) / "out"
            rc = eval_caves.main([
                "--roots", str(Path(tmp)),
                "--output", str(output),
                "--bootstrap", "10",
                "--bootstrap-seed", "2",
            ])
            self.assertEqual(rc, 0)
            self.assertTrue((output / "caves_per_condition_metrics.csv").exists())
            summary = json.loads((output / "caves_calibration_metrics.json").read_text())
            self.assertIn("per_condition_metrics", summary)
            self.assertGreaterEqual(len(summary["per_condition_metrics"]), 1)

    def test_rows_to_report_labels_scores_group_keys_skips_unusable_rows_in_one_pass(self):
        retained_a = {
            "image_name": "retained_a",
            "bad_pose_label": False,
            "num_matches": 100,
            "num_inliers": 80,
            "reproj_residual_px": 2.0,
            "shape_confidence": 0.95,
            "retrieval_score_margin": 0.4,
            "geometry_anchor_offset_m": 5.0,
        }
        retained_b = {
            "image_name": "retained_b",
            "bad_pose_label": True,
            "num_matches": 10,
            "num_inliers": 1,
            "reproj_residual_px": 25.0,
            "shape_confidence": 0.1,
            "retrieval_score_margin": 0.0,
            "geometry_anchor_offset_m": 200.0,
        }
        labels, scores, groups = eval_caves.rows_to_report_labels_scores_group_keys(
            [
                {"image_name": "missing_label", "bad_pose_label": "", "num_matches": 100},
                {"image_name": "nonfinite_score", "bad_pose_label": True},
                retained_a,
                retained_b,
            ],
            feature_set="full_caves",
            random_seed=7,
        )

        self.assertEqual(labels, [0, 1])
        self.assertEqual(groups, ["retained_a", "retained_b"])
        self.assertEqual(len(scores), 2)
        self.assertTrue(all(score == score for score in scores))

    def test_per_condition_grouped_ci_uses_group_keys_from_retained_finite_rows(self):
        captured_groups = []

        def fake_bootstrap_metric_ci(labels, scores, *, groups, **kwargs):
            captured_groups.append(list(groups))
            return {"mean": 0.0, "ci_low": 0.0, "ci_high": 0.0, "n_bootstrap": 0}

        rows = [
            {
                "adversarial_condition": "wrong_far",
                "image_name": "dropped_nonfinite",
                "bad_pose_label": True,
            },
            {
                "adversarial_condition": "wrong_far",
                "image_name": "retained_a",
                "bad_pose_label": False,
                "num_matches": 100,
                "num_inliers": 80,
                "reproj_residual_px": 2.0,
                "shape_confidence": 0.95,
                "retrieval_score_margin": 0.4,
                "geometry_anchor_offset_m": 5.0,
            },
            {
                "adversarial_condition": "wrong_far",
                "image_name": "retained_b",
                "bad_pose_label": True,
                "num_matches": 10,
                "num_inliers": 1,
                "reproj_residual_px": 25.0,
                "shape_confidence": 0.1,
                "retrieval_score_margin": 0.0,
                "geometry_anchor_offset_m": 200.0,
            },
        ]

        with mock.patch.object(eval_caves, "bootstrap_metric_ci", side_effect=fake_bootstrap_metric_ci):
            eval_caves.compute_per_condition_grouped_ci_rows(
                rows,
                bootstrap=5,
                bootstrap_seed=7,
                n_bins=10,
            )

        self.assertEqual(len(captured_groups), 1)
        self.assertEqual(captured_groups[0], ["retained_a", "retained_b"])


class SelectIllustrativeCurvePointsTest(unittest.TestCase):
    def _make_curve(self):
        """Synthetic held-out risk-coverage curve with known properties."""
        return [
            {"coverage": 0.02, "risk": 0.00, "threshold": 0.95, "accepted": 1, "total": 50},
            {"coverage": 0.06, "risk": 0.10, "threshold": 0.80, "accepted": 3, "total": 50},
            {"coverage": 0.15, "risk": 0.25, "threshold": 0.60, "accepted": 8, "total": 50},
            {"coverage": 0.25, "risk": 0.30, "threshold": 0.45, "accepted": 13, "total": 50},
            {"coverage": 0.40, "risk": 0.35, "threshold": 0.30, "accepted": 20, "total": 50},
            {"coverage": 0.60, "risk": 0.38, "threshold": 0.20, "accepted": 30, "total": 50},
            {"coverage": 1.00, "risk": 0.50, "threshold": 0.00, "accepted": 50, "total": 50},
        ]

    def test_returns_four_points(self):
        curve = self._make_curve()
        pts = eval_caves.select_illustrative_curve_points(curve)
        labels = [p["label"] for p in pts]
        self.assertEqual(len(pts), 4)
        self.assertIn("conservative_low_coverage", labels)
        self.assertIn("closest_infeasible_medium", labels)
        self.assertIn("baseline_unsafe_rate", labels)
        self.assertIn("accept_all_endpoint", labels)

    def test_conservative_point_respects_floor_and_baseline(self):
        curve = self._make_curve()
        pts = eval_caves.select_illustrative_curve_points(curve)
        conservative = next(p for p in pts if p["label"] == "conservative_low_coverage")
        self.assertGreaterEqual(conservative["coverage"], eval_caves.LOW_COVERAGE_FLOOR)
        self.assertLessEqual(conservative["risk"], eval_caves.BASELINE_UNSAFE_ACCEPT)

    def test_medium_point_within_coverage_band(self):
        curve = self._make_curve()
        pts = eval_caves.select_illustrative_curve_points(curve)
        medium = next(p for p in pts if p["label"] == "closest_infeasible_medium")
        cov_lo, cov_hi = eval_caves.COVERAGE_BAND
        self.assertGreaterEqual(medium["coverage"], cov_lo)
        self.assertLessEqual(medium["coverage"], cov_hi)

    def test_medium_picks_lowest_risk_in_band(self):
        curve = self._make_curve()
        pts = eval_caves.select_illustrative_curve_points(curve)
        medium = next(p for p in pts if p["label"] == "closest_infeasible_medium")
        self.assertAlmostEqual(medium["coverage"], 0.15)
        self.assertAlmostEqual(medium["risk"], 0.25)

    def test_baseline_point_closest_to_baseline(self):
        curve = self._make_curve()
        pts = eval_caves.select_illustrative_curve_points(curve)
        baseline_pt = next(p for p in pts if p["label"] == "baseline_unsafe_rate")
        baseline = eval_caves.BASELINE_UNSAFE_ACCEPT
        dist = abs(baseline_pt["risk"] - baseline)
        for row in curve:
            self.assertLessEqual(dist, abs(row["risk"] - baseline) + 1e-9)

    def test_accept_all_is_max_coverage(self):
        curve = self._make_curve()
        pts = eval_caves.select_illustrative_curve_points(curve)
        endpoint = next(p for p in pts if p["label"] == "accept_all_endpoint")
        self.assertAlmostEqual(endpoint["coverage"], 1.0)

    def test_empty_curve_returns_empty(self):
        pts = eval_caves.select_illustrative_curve_points([])
        self.assertEqual(pts, [])

    def test_no_medium_band_points_skips_medium(self):
        curve = [
            {"coverage": 0.06, "risk": 0.10, "threshold": 0.80, "accepted": 3, "total": 50},
            {"coverage": 1.00, "risk": 0.50, "threshold": 0.00, "accepted": 50, "total": 50},
        ]
        pts = eval_caves.select_illustrative_curve_points(curve)
        labels = [p["label"] for p in pts]
        self.assertNotIn("closest_infeasible_medium", labels)
        self.assertIn("accept_all_endpoint", labels)


class SelectOperatingPointsTest(unittest.TestCase):
    def test_safety_first_selects_zero_bad(self):
        sweep = [
            {"coverage": 0.01, "accepted_bad_rate": 0.0, "threshold": 0.9,
             "false_reject_rate": 0.99, "safe_reject_rate": 1.0},
            {"coverage": 0.50, "accepted_bad_rate": 0.15, "threshold": 0.5,
             "false_reject_rate": 0.50, "safe_reject_rate": 0.85},
        ]
        pts = eval_caves.select_operating_points(sweep)
        safety = next(p for p in pts if p["name"] == "safety_first")
        self.assertEqual(safety["coverage"], 0.01)

    def test_medium_unavailable_when_no_feasible(self):
        sweep = [
            {"coverage": 0.20, "accepted_bad_rate": 0.25, "threshold": 0.5,
             "false_reject_rate": 0.80, "safe_reject_rate": 0.75},
        ]
        pts = eval_caves.select_operating_points(sweep)
        medium = next(p for p in pts if "medium_risk" in p["name"])
        self.assertIn("unavailable", medium["name"])
        self.assertTrue(math.isnan(medium["threshold"]))


class FinalizeForgeLabelsTest(unittest.TestCase):
    def _make_row(self, **overrides):
        row = {}
        _apply_reliability_schema_defaults(row)
        row.update(overrides)
        return row

    def test_assert_fires_without_prior_initialization(self):
        with self.assertRaises(AssertionError):
            _finalize_forge_labels({}, "avl", {"bad_pose_threshold_m": 20.0})

    def test_bad_pose_when_error_exceeds_threshold(self):
        row = self._make_row(post_fgo_error_m=25.0)
        _finalize_forge_labels(row, "avl", {"bad_pose_threshold_m": 20.0})
        self.assertTrue(row["bad_pose_label"])

    def test_good_pose_when_error_below_threshold(self):
        row = self._make_row(post_fgo_error_m=15.0)
        _finalize_forge_labels(row, "avl", {"bad_pose_threshold_m": 20.0})
        self.assertFalse(row["bad_pose_label"])

    def test_nan_adversarial_not_accepted_is_bad(self):
        row = self._make_row(
            post_fgo_error_m=float("nan"),
            is_adversarial=True,
            is_wrong_reference=True,
            accepted_for_metrics=False,
        )
        _finalize_forge_labels(row, "avl", {"bad_pose_threshold_m": 20.0})
        self.assertTrue(row["bad_pose_label"])

    def test_nan_adversarial_accepted_keeps_default(self):
        row = self._make_row(
            post_fgo_error_m=float("nan"),
            is_adversarial=True,
            is_wrong_reference=True,
            accepted_for_metrics=True,
        )
        _finalize_forge_labels(row, "avl", {"bad_pose_threshold_m": 20.0})
        self.assertFalse(row["bad_pose_label"])

    def test_nan_non_adversarial_keeps_default(self):
        row = self._make_row(
            post_fgo_error_m=float("nan"),
            is_adversarial=False,
            is_wrong_reference=False,
        )
        _finalize_forge_labels(row, "avl", {"bad_pose_threshold_m": 20.0})
        self.assertFalse(row["bad_pose_label"])

    def test_unsafe_accept_set_when_bad_and_accepted(self):
        row = self._make_row(post_fgo_error_m=25.0, accepted_for_metrics=True)
        _finalize_forge_labels(row, "avl", {"bad_pose_threshold_m": 20.0})
        self.assertTrue(row["bad_pose_label"])
        self.assertTrue(row["unsafe_accept"])

    def test_no_unsafe_accept_when_bad_but_rejected(self):
        row = self._make_row(post_fgo_error_m=25.0, accepted_for_metrics=False)
        _finalize_forge_labels(row, "avl", {"bad_pose_threshold_m": 20.0})
        self.assertTrue(row["bad_pose_label"])
        self.assertFalse(row["unsafe_accept"])


class HighImbalanceBootstrapTest(unittest.TestCase):
    def test_auroc_bootstrap_high_imbalance(self):
        labels = [1] * 97 + [0] * 3
        scores = [0.5 + 0.005 * i for i in range(97)] + [0.1, 0.2, 0.3]
        ci = caves.bootstrap_auroc_ci(labels, scores, n_bootstrap=200, seed=42)
        self.assertTrue(math.isfinite(ci["auroc"]))
        self.assertTrue(math.isfinite(ci["auroc_ci_low"]))
        self.assertTrue(math.isfinite(ci["auroc_ci_high"]))
        self.assertLessEqual(ci["auroc_ci_low"], ci["auroc"])
        self.assertGreaterEqual(ci["auroc_ci_high"], ci["auroc"])
        self.assertGreater(ci["auroc_bootstrap_valid"], 0)


if __name__ == "__main__":
    unittest.main()
