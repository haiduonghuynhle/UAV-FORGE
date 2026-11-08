import sys
import tempfile
import unittest
from pathlib import Path

try:
    import numpy as np
except ModuleNotFoundError:  # Local Python 3.14 may not have scientific deps installed.
    np = None


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


class _FakeMatcher:
    def __init__(self, q_pts, warped_ref_pts):
        self.q_pts = np.asarray(q_pts, dtype=np.float64)
        self.warped_ref_pts = np.asarray(warped_ref_pts, dtype=np.float64)

    def match(self, query_img, ref_img):
        import cv2
        from core.types import MatchResult

        h, mask = cv2.findHomography(
            self.q_pts.astype(np.float32),
            self.warped_ref_pts.astype(np.float32),
            cv2.USAC_MAGSAC,
            3.0,
        )
        return MatchResult(
            self.q_pts,
            self.warped_ref_pts,
            np.ones(len(self.q_pts), dtype=np.float64),
            homography=h,
            inlier_mask=mask.ravel().astype(bool) if mask is not None else None,
            metadata={"matcher": "fake", "shape_confidence": 0.8},
        )


@unittest.skipIf(np is None, "numpy is required for reference preconditioning tests")
class ReferencePreconditioningContractsTest(unittest.TestCase):
    def _first_result(self, h):
        from core.types import MatchResult

        q = np.array(
            [[10, 10], [40, 10], [70, 20], [20, 40], [50, 50], [80, 70]],
            dtype=np.float64,
        )
        ref = q + np.array([10.0, 5.0])
        return MatchResult(
            q[:4],
            ref[:4],
            np.ones(4, dtype=np.float64),
            homography=np.asarray(h, dtype=np.float64),
            inlier_mask=np.ones(4, dtype=bool),
            metadata={"matcher": "first", "shape_confidence": 0.5},
        )

    def test_remaps_second_pass_reference_points_to_original_reference_frame(self):
        from modules.vision.matching import rematch_with_reference_preconditioning

        h = np.array([[1.0, 0.0, 10.0], [0.0, 1.0, 5.0], [0.0, 0.0, 1.0]])
        q = np.array(
            [[10, 10], [40, 10], [70, 20], [20, 40], [50, 50], [80, 70]],
            dtype=np.float64,
        )
        matcher = _FakeMatcher(q, q)  # warped-reference coordinates equal query-frame coordinates.
        img = np.zeros((120, 120, 3), dtype=np.uint8)
        img[10:90, 10:90] = 255

        result, diag = rematch_with_reference_preconditioning(
            matcher,
            img,
            img,
            self._first_result(h),
            min_inliers=4,
            min_inlier_gain=1,
            min_score_gain=0.0,
            max_reproj_px=5.0,
        )

        self.assertTrue(diag["accepted"], diag)
        np.testing.assert_allclose(result.ref_keypoints, q + np.array([10.0, 5.0]), atol=1e-4)
        self.assertEqual(result.num_inliers, len(q))
        self.assertTrue(result.metadata.get("reference_preconditioning"))

    def test_fails_closed_on_singular_first_homography(self):
        from modules.vision.matching import rematch_with_reference_preconditioning

        img = np.ones((80, 80, 3), dtype=np.uint8) * 127
        first = self._first_result(np.zeros((3, 3), dtype=np.float64))
        result, diag = rematch_with_reference_preconditioning(
            _FakeMatcher([[0, 0], [1, 0], [1, 1], [0, 1]], [[0, 0], [1, 0], [1, 1], [0, 1]]),
            img,
            img,
            first,
        )

        self.assertIs(result, first)
        self.assertFalse(diag["accepted"])
        self.assertFalse(diag["warp_valid"])
        self.assertIn("homography", diag["reason"])


if __name__ == "__main__":
    unittest.main()
