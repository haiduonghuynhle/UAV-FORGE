import json
import tempfile
import unittest
from pathlib import Path

import benchmark_runner as br


class ForgeWrongReferenceTest(unittest.TestCase):
    def _candidate(self, idx, rank, dist, original=False, score=None, name=None):
        return {
            "candidate_index": idx,
            "rank": rank,
            "reference_name": name or f"ref_{idx}",
            "reference_image": object(),
            "anchor": {"lat": 10.0, "lon": 106.0},
            "similarity_score": 1.0 - rank * 0.1 if score is None else score,
            "distance_to_gt_m": dist,
            "is_original_candidate": original,
        }

    def test_defaults_are_clean_and_not_adversarial(self):
        cfg = br._normalize_adversarial_config({})
        self.assertEqual(cfg["adversarial_mode"], "none")
        self.assertEqual(cfg["adversarial_condition"], "clean")
        self.assertFalse(cfg["is_adversarial_enabled"])

    def test_force_rank_k_selects_requested_wrong_candidate(self):
        cfg = br._normalize_adversarial_config({
            "adversarial_mode": "wrong_reference",
            "adversarial_condition": "wrong_near",
            "adversarial_candidate_policy": "force_rank_k",
            "adversarial_candidate_rank": 2,
        })
        candidates = [self._candidate(1, 0, 0.0, original=True), self._candidate(2, 1, 25.0)]
        selected = br._select_forge_candidate(cfg, candidates, "avl")
        self.assertTrue(selected["ok"])
        self.assertEqual(selected["candidate"]["candidate_index"], 2)

    def test_force_rank_k_fails_on_original_or_out_of_range(self):
        cfg = br._normalize_adversarial_config({
            "adversarial_mode": "wrong_reference",
            "adversarial_candidate_policy": "force_rank_k",
            "adversarial_candidate_rank": 1,
        })
        candidates = [self._candidate(1, 0, 0.0, original=True)]
        selected = br._select_forge_candidate(cfg, candidates, "avl")
        self.assertFalse(selected["ok"])
        self.assertEqual(selected["reason"], "adversarial_rank_is_original")

        cfg["adversarial_candidate_rank"] = 5
        selected = br._select_forge_candidate(cfg, candidates, "avl")
        self.assertFalse(selected["ok"])
        self.assertEqual(selected["reason"], "adversarial_rank_out_of_range")

    def test_nearest_wrong_requires_metadata_and_selects_nearest_outside_threshold(self):
        cfg = br._normalize_adversarial_config({
            "adversarial_mode": "wrong_reference",
            "adversarial_candidate_policy": "nearest_wrong",
        })
        missing = [self._candidate(1, 0, float("nan"), original=True), self._candidate(2, 1, float("nan"))]
        selected = br._select_forge_candidate(cfg, missing, "avl")
        self.assertFalse(selected["ok"])
        self.assertTrue(selected.get("metadata_unavailable"))

        candidates = [
            self._candidate(1, 0, 0.0, original=True),
            self._candidate(2, 1, 35.0),
            self._candidate(3, 2, 120.0),
        ]
        selected = br._select_forge_candidate(cfg, candidates, "avl")
        self.assertTrue(selected["ok"])
        self.assertEqual(selected["candidate"]["candidate_index"], 2)

    def test_far_wrong_selects_farthest_outside_threshold(self):
        cfg = br._normalize_adversarial_config({
            "adversarial_mode": "wrong_reference",
            "adversarial_candidate_policy": "far_wrong",
        })
        candidates = [
            self._candidate(1, 0, 0.0, original=True),
            self._candidate(2, 1, 35.0),
            self._candidate(3, 2, 120.0),
        ]
        selected = br._select_forge_candidate(cfg, candidates, "vpair")
        self.assertTrue(selected["ok"])
        self.assertEqual(selected["candidate"]["candidate_index"], 3)

    def test_visually_similar_wrong_condition_defaults_to_semihard_policy(self):
        cfg = br._normalize_adversarial_config({
            "adversarial_mode": "wrong_reference",
            "adversarial_condition": "visually_similar_wrong",
        })
        self.assertTrue(cfg["is_adversarial_enabled"])
        self.assertEqual(cfg["adversarial_condition"], "visually_similar_wrong")
        self.assertEqual(cfg["adversarial_candidate_policy"], "visually_similar_wrong")

    def test_wrong_semihard_selects_highest_similarity_wrong_candidate(self):
        cfg = br._normalize_adversarial_config({
            "adversarial_mode": "wrong_reference",
            "adversarial_candidate_policy": "wrong_semihard",
        })
        candidates = [
            self._candidate(1, 0, 0.0, original=True, score=0.99),
            self._candidate(2, 1, 120.0, score=0.40),
            self._candidate(3, 2, 35.0, score=0.92),
        ]
        selected = br._select_forge_candidate(cfg, candidates, "avl")
        self.assertTrue(selected["ok"])
        self.assertEqual(selected["reason"], "visually_similar_wrong")
        self.assertEqual(selected["candidate"]["candidate_index"], 3)
        self.assertEqual(selected["eligible_wrong_count"], 2)
        self.assertEqual(selected["candidate_pool_size"], 3)

    def test_wrong_semihard_ignores_inside_threshold_even_if_more_similar(self):
        cfg = br._normalize_adversarial_config({
            "adversarial_mode": "wrong_reference",
            "adversarial_candidate_policy": "wrong_semihard",
        })
        candidates = [
            self._candidate(1, 0, 0.0, original=True, score=0.99),
            self._candidate(2, 1, 10.0, score=0.98),
            self._candidate(3, 2, 35.0, score=0.60),
        ]
        selected = br._select_forge_candidate(cfg, candidates, "avl")
        self.assertTrue(selected["ok"])
        self.assertEqual(selected["candidate"]["candidate_index"], 3)

    def test_wrong_semihard_tie_break_is_deterministic(self):
        cfg = br._normalize_adversarial_config({
            "adversarial_mode": "wrong_reference",
            "adversarial_candidate_policy": "wrong_semihard",
        })
        candidates = [
            self._candidate(1, 0, 0.0, original=True, score=0.99),
            self._candidate(4, 3, 80.0, score=0.75, name="b_ref"),
            self._candidate(2, 1, 85.0, score=0.75, name="a_ref"),
            self._candidate(3, 2, 90.0, score=0.75, name="c_ref"),
        ]
        selected = br._select_forge_candidate(cfg, candidates, "avl")
        self.assertTrue(selected["ok"])
        self.assertEqual(selected["candidate"]["candidate_index"], 2)

    def test_uav_visloc_uses_150m_wrong_reference_threshold(self):
        cfg = br._normalize_adversarial_config({
            "adversarial_mode": "wrong_reference",
            "adversarial_candidate_policy": "wrong_semihard",
        })
        candidates = [
            self._candidate(1, 0, 0.0, original=True, score=0.99),
            self._candidate(2, 1, 120.0, score=0.95),
            self._candidate(3, 2, 180.0, score=0.70),
        ]
        selected = br._select_forge_candidate(cfg, candidates, "uav_visloc")
        self.assertTrue(selected["ok"])
        self.assertEqual(selected["candidate"]["candidate_index"], 3)

    def test_cross_flight_is_fail_closed_for_phase2(self):
        cfg = br._normalize_adversarial_config({
            "adversarial_mode": "wrong_reference",
            "adversarial_candidate_policy": "cross_flight",
        })
        selected = br._select_forge_candidate(cfg, [self._candidate(2, 1, 100.0)], "avl")
        self.assertFalse(selected["ok"])
        self.assertEqual(selected["reason"], "adversarial_cross_flight_unsupported")
        self.assertTrue(selected.get("unsupported_condition"))

    def test_predictions_schema_contains_caves_outcome_fields(self):
        required = {
            "retrieval_correct_threshold_m",
            "localization_success_threshold_m",
            "retrieval_rank",
            "localization_is_success",
            "caves_reject",
            "false_reject",
            "unsafe_accept_severity_m",
        }
        self.assertTrue(required.issubset(set(br.PREDICTIONS_CSV_COLUMNS)))

    def test_finalize_forge_labels_populates_caves_outcomes_and_rank(self):
        good_rejected = {
            "post_fgo_error_m": 10.0,
            "accepted_for_metrics": False,
            "candidate_rank_selected": 2,
            "num_matches": 10,
            "num_inliers": 1,
            "reproj_residual_px": 25.0,
            "shape_confidence": 0.1,
            "retrieval_score_margin": 0.0,
            "is_adversarial": True,
            "is_wrong_reference": True,
        }
        br._apply_reliability_schema_defaults(good_rejected)
        br._finalize_forge_labels(good_rejected, "avl", {"caves_reject_threshold": 0.5})
        self.assertAlmostEqual(good_rejected["localization_success_threshold_m"], 20.0)
        self.assertAlmostEqual(good_rejected["retrieval_correct_threshold_m"], 20.0)
        self.assertAlmostEqual(good_rejected["retrieval_rank"], 3.0)
        self.assertTrue(good_rejected["localization_is_success"])
        self.assertTrue(good_rejected["caves_reject"])
        self.assertTrue(good_rejected["false_reject"])
        self.assertFalse(good_rejected["unsafe_accept"])

        bad_accepted = {
            "post_fgo_error_m": 80.0,
            "accepted_for_metrics": True,
            "candidate_rank_selected": 0,
            "num_matches": 100,
            "num_inliers": 80,
            "reproj_residual_px": 2.0,
            "shape_confidence": 0.95,
            "retrieval_score_margin": 0.4,
            "is_adversarial": True,
            "is_wrong_reference": True,
        }
        br._apply_reliability_schema_defaults(bad_accepted)
        br._finalize_forge_labels(bad_accepted, "avl", {"caves_reject_threshold": 0.5})
        self.assertFalse(bad_accepted["localization_is_success"])
        self.assertTrue(bad_accepted["unsafe_accept"])
        self.assertAlmostEqual(bad_accepted["unsafe_accept_severity_m"], 60.0)

    def test_finalize_forge_labels_nan_error_adversarial_fallback(self):
        """When post_fgo_error_m is NaN and the frame is an adversarial
        wrong-reference that was rejected, bad_pose_label should be True
        via the heuristic fallback (line 2576-2577)."""
        nan_rejected = {
            "post_fgo_error_m": float("nan"),
            "accepted_for_metrics": False,
            "candidate_rank_selected": 1,
            "num_matches": 5,
            "num_inliers": 0,
            "reproj_residual_px": float("nan"),
            "shape_confidence": 0.0,
            "retrieval_score_margin": 0.0,
            "is_adversarial": True,
            "is_wrong_reference": True,
        }
        br._apply_reliability_schema_defaults(nan_rejected)
        br._finalize_forge_labels(nan_rejected, "avl", {"caves_reject_threshold": 0.5})
        self.assertTrue(nan_rejected["bad_pose_label"],
                        "NaN error + adversarial wrong-reference + rejected → bad_pose_label=True")
        self.assertFalse(nan_rejected["unsafe_accept"],
                         "rejected frame cannot be unsafe_accept")

        nan_accepted = {
            "post_fgo_error_m": float("nan"),
            "accepted_for_metrics": True,
            "candidate_rank_selected": 0,
            "num_matches": 50,
            "num_inliers": 40,
            "reproj_residual_px": 3.0,
            "shape_confidence": 0.9,
            "retrieval_score_margin": 0.3,
            "is_adversarial": True,
            "is_wrong_reference": True,
        }
        br._apply_reliability_schema_defaults(nan_accepted)
        br._finalize_forge_labels(nan_accepted, "avl", {"caves_reject_threshold": 0.5})
        self.assertFalse(nan_accepted["bad_pose_label"],
                         "NaN error + accepted → fallback does NOT fire (accepted_for_metrics=True)")

    def test_metrics_summary_exports_caves_aggregate_fields(self):
        paper_metrics = {
            "num_false_reject": 1,
            "false_rejection_rate": 0.25,
            "num_caves_reject": 2,
            "rejection_rate": 0.5,
            "caves_rejection_rate_denominator": "adversarial_samples",
            "mean_unsafe_accept_severity_m": 42.0,
            "median_unsafe_accept_severity_m": 40.0,
            "p90_unsafe_accept_severity_m": 50.0,
            "max_unsafe_accept_severity_m": 55.0,
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "metrics_summary.json"
            br.write_metrics_summary_json(
                path,
                dataset="avl",
                pipeline="ours_camp_cascade_6dof_e2e",
                metrics={},
                paper_metrics=paper_metrics,
                results=[{"failure_reason": "", "pose_success": True, "accepted_for_metrics": True}],
                manifest={"canonical_pipeline": "ours_camp_cascade_6dof_e2e"},
            )
            payload = json.loads(path.read_text())
        self.assertEqual(payload["num_false_reject"], 1)
        self.assertEqual(payload["num_caves_reject"], 2)
        self.assertAlmostEqual(payload["p90_unsafe_accept_severity_m"], 50.0)


if __name__ == "__main__":
    unittest.main()
