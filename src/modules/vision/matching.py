"""
Feature matching backend factory for HD4.

Provides ORBMatcher (fallback) and LightGlueMatcher (SuperPoint + LightGlue).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
import numpy as np

from core.types import MatchResult

logger = logging.getLogger(__name__)


def _canonical_torch_device(device_str: str):
    """Match LightGlue tensor devices: ``cuda`` must resolve to ``cuda:0`` (or current GPU)."""
    import torch

    if device_str == "cuda" and torch.cuda.is_available():
        return torch.device("cuda", torch.cuda.current_device())
    return torch.device(device_str)


def _tensor_devices_match(expected, actual) -> bool:
    import torch

    exp = torch.device(expected)
    act = torch.device(actual)
    if exp.type != act.type:
        return False
    if exp.type == "cpu":
        return True
    ei = 0 if exp.index is None else exp.index
    ai = 0 if act.index is None else act.index
    return ei == ai


def _recursive_to_device(obj, device):
    """Move every torch.Tensor in nested dict/list/tuple structures to ``device``."""
    import torch

    if isinstance(obj, torch.Tensor):
        return obj.to(device, non_blocking=False).contiguous()
    if isinstance(obj, dict):
        return {k: _recursive_to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        seq = [_recursive_to_device(v, device) for v in obj]
        return type(obj)(seq)
    return obj


def _resolve_matching_models_root() -> Path:
    """
    Directory that contains ``Matching_Models/`` (RoMa weights, etc.).

    When code runs from a writable overlay (e.g. ``code_patch``) the weights
    often live in a sibling checkout (``../code``). ``HD4_REPO_ROOT`` /
    ``FORGE_CODE_ROOT`` override this resolution.
    """
    env = (os.environ.get("HD4_REPO_ROOT") or os.environ.get("FORGE_CODE_ROOT") or "").strip()
    if env:
        return Path(env).expanduser().resolve()
    here = Path(__file__).resolve()
    pkg_root = here.parents[3]
    if (pkg_root / "Matching_Models").is_dir():
        return pkg_root
    sibling = pkg_root.parent / "code"
    if (sibling / "Matching_Models").is_dir():
        return sibling.resolve()
    return pkg_root


def _collect_tensor_devices(obj, out: list | None = None) -> list[tuple[str, str]]:
    """Debug helper: (path, device_str) for tensors in nested structures."""
    import torch

    if out is None:
        out = []

    def walk(x, prefix: str) -> None:
        if isinstance(x, torch.Tensor):
            out.append((prefix or "tensor", str(x.device)))
            return
        if isinstance(x, dict):
            for k, v in x.items():
                walk(v, f"{prefix}.{k}" if prefix else str(k))
            return
        if isinstance(x, (list, tuple)):
            for i, v in enumerate(x):
                walk(v, f"{prefix}[{i}]")

    walk(obj, "")
    return out


def compute_shape_confidence(
    homography: np.ndarray | None,
    query_size: tuple[int, int],
    ref_size: tuple[int, int],
) -> float:
    """
    Minimal structural confidence from homography-projected quadrilateral.

    Score uses the conservative rule:
        C_shape = min(aspect_ratio_consistency, area_ratio_consistency)
    where both terms are normalized into [0, 1].
    """
    if homography is None:
        return float("nan")

    q_w, q_h = int(query_size[0]), int(query_size[1])
    r_w, r_h = int(ref_size[0]), int(ref_size[1])
    if q_w <= 1 or q_h <= 1 or r_w <= 1 or r_h <= 1:
        return float("nan")

    try:
        import cv2

        corners = np.array(
            [[[0.0, 0.0], [q_w - 1.0, 0.0], [q_w - 1.0, q_h - 1.0], [0.0, q_h - 1.0]]],
            dtype=np.float32,
        )
        projected = cv2.perspectiveTransform(corners, homography.astype(np.float64))[0]
        if projected.shape != (4, 2) or not np.isfinite(projected).all():
            return float("nan")

        top = np.linalg.norm(projected[1] - projected[0])
        bottom = np.linalg.norm(projected[2] - projected[3])
        left = np.linalg.norm(projected[3] - projected[0])
        right = np.linalg.norm(projected[2] - projected[1])
        width_est = max(1e-6, 0.5 * (top + bottom))
        height_est = max(1e-6, 0.5 * (left + right))
        proj_aspect = width_est / height_est
        ref_aspect = float(r_w) / float(r_h)
        aspect_consistency = min(proj_aspect, ref_aspect) / max(proj_aspect, ref_aspect)

        area_proj = abs(float(cv2.contourArea(projected.astype(np.float32))))
        area_ref = float(r_w * r_h)
        if area_proj <= 1e-6 or area_ref <= 1e-6:
            return float("nan")
        area_ratio = area_proj / area_ref
        area_consistency = min(area_ratio, 1.0 / max(area_ratio, 1e-12))

        score = float(min(aspect_consistency, area_consistency))
        return float(np.clip(score, 0.0, 1.0))
    except Exception:
        return float("nan")


def _match_reprojection_residual_px(match_result: MatchResult) -> float:
    """Mean homography reprojection residual over inlier correspondences."""
    if match_result.homography is None or match_result.num_matches < 4:
        return float("nan")
    try:
        import cv2

        q = match_result.query_keypoints.astype(np.float64).reshape(-1, 1, 2)
        pred = cv2.perspectiveTransform(q, match_result.homography.astype(np.float64)).reshape(-1, 2)
        residual = np.linalg.norm(pred - match_result.ref_keypoints.astype(np.float64), axis=1)
        if match_result.inlier_mask is not None and len(match_result.inlier_mask) == len(residual):
            residual = residual[match_result.inlier_mask]
        if len(residual) == 0:
            return float("nan")
        return float(np.mean(residual))
    except Exception:
        return float("nan")


def _preconditioning_score(inliers: int, reproj_px: float, shape_conf: float) -> float:
    reproj_term = 0.0 if not np.isfinite(reproj_px) else float(reproj_px)
    shape_term = 0.0 if not np.isfinite(shape_conf) else float(shape_conf)
    return float(inliers + 10.0 * shape_term - reproj_term)


def rematch_with_reference_preconditioning(
    matcher,
    query_img: np.ndarray,
    ref_img: np.ndarray,
    first_result: MatchResult,
    *,
    min_inliers: int = 8,
    min_inlier_gain: int = 2,
    min_score_gain: float = 1.0,
    max_reproj_px: float | None = None,
    shape_tolerance: float = 0.03,
) -> tuple[MatchResult, dict]:
    """AdHoP-style reference preconditioning for a selected reference tile.

    The first-pass homography is interpreted as query -> original-reference. The
    reference image is warped back into the query frame, matched again, and the
    second-pass reference keypoints are mapped back to original-reference pixels.
    The helper is fail-closed: invalid warps/rematches return ``first_result``
    together with diagnostics instead of raising frame-level exceptions.
    """
    diag = {
        "attempted": True,
        "accepted": False,
        "rejected": True,
        "reason": "not_evaluated",
        "warp_valid": False,
        "warp_direction": "ref_to_query_then_backmap",
        "inliers_before": int(first_result.num_inliers),
        "inliers_after": 0,
        "matches_before": int(first_result.num_matches),
        "matches_after": 0,
        "reproj_before_px": float(_match_reprojection_residual_px(first_result)),
        "reproj_after_px": float("nan"),
        "shape_before": float(first_result.metadata.get("shape_confidence", float("nan")) if isinstance(first_result.metadata, dict) else float("nan")),
        "shape_after": float("nan"),
        "score_before": float("nan"),
        "score_after": float("nan"),
    }

    h_q_to_ref = first_result.homography
    if h_q_to_ref is None or np.asarray(h_q_to_ref).shape != (3, 3) or not np.isfinite(h_q_to_ref).all():
        diag["reason"] = "invalid_first_homography"
        return first_result, diag
    try:
        import cv2

        h_q_to_ref = np.asarray(h_q_to_ref, dtype=np.float64)
        if abs(float(np.linalg.det(h_q_to_ref))) < 1e-12:
            diag["reason"] = "singular_first_homography"
            return first_result, diag
        h_ref_to_q = np.linalg.inv(h_q_to_ref)
        if not np.isfinite(h_ref_to_q).all():
            diag["reason"] = "singular_first_homography"
            return first_result, diag
        q_h, q_w = query_img.shape[:2]
        ref_warped = cv2.warpPerspective(ref_img, h_ref_to_q, (int(q_w), int(q_h)))
        if ref_warped.size == 0 or float(np.std(ref_warped)) <= 1e-6:
            diag["reason"] = "blank_or_invalid_warp"
            return first_result, diag
        diag["warp_valid"] = True

        second = matcher.match(query_img, ref_warped)
        diag["matches_after"] = int(second.num_matches)
        if second.num_matches < 4:
            diag["reason"] = "insufficient_second_pass_matches"
            return first_result, diag

        ref_warped_pts = second.ref_keypoints.astype(np.float64).reshape(-1, 1, 2)
        ref_original_pts = cv2.perspectiveTransform(ref_warped_pts, h_q_to_ref).reshape(-1, 2)
        q_pts = second.query_keypoints.astype(np.float64)
        finite = np.isfinite(q_pts).all(axis=1) & np.isfinite(ref_original_pts).all(axis=1)
        if not np.any(finite) or int(np.sum(finite)) < 4:
            diag["reason"] = "nonfinite_backmapped_points"
            return first_result, diag
        q_pts = q_pts[finite]
        ref_original_pts = ref_original_pts[finite]
        conf = second.confidence[finite] if len(second.confidence) == len(finite) else np.ones(len(q_pts), dtype=np.float64)

        homography, mask = cv2.findHomography(
            q_pts.astype(np.float32),
            ref_original_pts.astype(np.float32),
            cv2.USAC_MAGSAC,
            float(max_reproj_px) if max_reproj_px and np.isfinite(max_reproj_px) else 10.0,
        )
        inlier_mask = mask.ravel().astype(bool) if mask is not None else None
        shape_after = compute_shape_confidence(
            homography=homography,
            query_size=(query_img.shape[1], query_img.shape[0]),
            ref_size=(ref_img.shape[1], ref_img.shape[0]),
        )
        candidate = MatchResult(
            query_keypoints=q_pts,
            ref_keypoints=ref_original_pts,
            confidence=conf,
            homography=homography,
            inlier_mask=inlier_mask,
            metadata={
                **(second.metadata or {}),
                "matcher": f"{(second.metadata or {}).get('matcher', 'matcher')}_reference_preconditioned",
                "reference_preconditioning": True,
                "shape_confidence": shape_after,
            },
        )
        reproj_after = _match_reprojection_residual_px(candidate)
        inliers_after = int(candidate.num_inliers)
        diag.update({
            "inliers_after": inliers_after,
            "reproj_after_px": float(reproj_after),
            "shape_after": float(shape_after),
        })
        reproj_before = float(diag["reproj_before_px"])
        shape_before = float(diag["shape_before"])
        score_before = _preconditioning_score(int(diag["inliers_before"]), reproj_before, shape_before)
        score_after = _preconditioning_score(inliers_after, reproj_after, shape_after)
        diag["score_before"] = float(score_before)
        diag["score_after"] = float(score_after)

        if homography is None or not np.isfinite(homography).all():
            diag["reason"] = "invalid_second_homography"
            return first_result, diag
        if inliers_after < max(int(min_inliers), int(diag["inliers_before"]) + int(min_inlier_gain)):
            diag["reason"] = "insufficient_inlier_gain"
            return first_result, diag
        if max_reproj_px is not None and np.isfinite(max_reproj_px) and np.isfinite(reproj_after) and reproj_after > float(max_reproj_px):
            diag["reason"] = "high_second_reprojection"
            return first_result, diag
        if np.isfinite(reproj_before) and np.isfinite(reproj_after) and reproj_after > (reproj_before + 1e-6):
            diag["reason"] = "reprojection_regressed"
            return first_result, diag
        if np.isfinite(shape_before) and np.isfinite(shape_after) and shape_after < max(0.0, shape_before - float(shape_tolerance)):
            diag["reason"] = "shape_regressed"
            return first_result, diag
        if score_after < score_before + float(min_score_gain):
            diag["reason"] = "score_gain_too_small"
            return first_result, diag

        diag["accepted"] = True
        diag["rejected"] = False
        diag["reason"] = "accepted"
        candidate.metadata["reference_preconditioning_diagnostics"] = dict(diag)
        return candidate, diag
    except Exception as exc:
        diag["reason"] = f"exception:{type(exc).__name__}"
        logger.debug("Reference preconditioning failed closed: %s", exc, exc_info=True)
        return first_result, diag


class ORBMatcher:
    """Lightweight ORB matcher for benchmark/runtime fallback."""

    def __init__(self, max_features: int = 2000):
        import cv2

        self._cv2 = cv2
        self._orb = cv2.ORB_create(max_features)
        self._bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    def match(self, query_img: np.ndarray, ref_img: np.ndarray) -> MatchResult:
        cv2 = self._cv2
        q_gray = cv2.cvtColor(query_img, cv2.COLOR_RGB2GRAY) if query_img.ndim == 3 else query_img
        r_gray = cv2.cvtColor(ref_img, cv2.COLOR_RGB2GRAY) if ref_img.ndim == 3 else ref_img
        kq, dq = self._orb.detectAndCompute(q_gray, None)
        kr, dr = self._orb.detectAndCompute(r_gray, None)
        if dq is None or dr is None or len(kq) < 8 or len(kr) < 8:
            return MatchResult(np.zeros((0, 2)), np.zeros((0, 2)), np.zeros((0,)))

        knn = self._bf.knnMatch(dq, dr, k=2)
        good = []
        for pair in knn:
            if len(pair) < 2:
                continue
            m, n = pair
            if m.distance < 0.75 * n.distance:
                good.append(m)
        if len(good) < 4:
            return MatchResult(np.zeros((0, 2)), np.zeros((0, 2)), np.zeros((0,)))

        q_pts = np.array([kq[m.queryIdx].pt for m in good], dtype=np.float64)
        r_pts = np.array([kr[m.trainIdx].pt for m in good], dtype=np.float64)
        conf = np.array([1.0 / (1.0 + m.distance) for m in good], dtype=np.float64)

        homography = None
        inlier_mask = None
        if len(q_pts) >= 4:
            homography, mask = cv2.findHomography(
                q_pts.astype(np.float32),
                r_pts.astype(np.float32),
                cv2.RANSAC,
                5.0,
            )
            if mask is not None:
                inlier_mask = mask.ravel().astype(bool)

        shape_conf = compute_shape_confidence(
            homography=homography,
            query_size=(query_img.shape[1], query_img.shape[0]),
            ref_size=(ref_img.shape[1], ref_img.shape[0]),
        )
        return MatchResult(
            query_keypoints=q_pts,
            ref_keypoints=r_pts,
            confidence=conf,
            homography=homography,
            inlier_mask=inlier_mask,
            metadata={"matcher": "orb", "shape_confidence": shape_conf},
        )


class LightGlueMatcher:
    """
    SuperPoint + LightGlue feature matcher.

    Uses SuperPoint for local feature extraction and LightGlue for
    learned feature matching. Significantly outperforms ORB for
    cross-domain matching (UAV ↔ satellite orthomosaic).

    Reference:
        Lindenberger et al., "LightGlue: Local Feature Matching at
        Light Speed", ICCV 2023.
    """

    def __init__(
        self,
        max_num_keypoints: int = 2048,
        device: str = "cpu",
        resize_max: int = 1024,
    ):
        """
        Initialize SuperPoint + LightGlue matcher.

        Args:
            max_num_keypoints: Maximum keypoints per image for SuperPoint.
            device: 'cpu' or 'cuda'.
            resize_max: Resize longest edge to this before extraction
                        (saves GPU memory for large images).
        """
        self._device = device
        self._resize_max = resize_max
        self._extractor = None
        self._matcher = None
        self._torch = None
        self._loaded = False

        self._max_kps = max_num_keypoints
        self._init_models()

    def _init_models(self):
        """Lazy-load torch + LightGlue models."""
        try:
            import os
            os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
            import torch

            # Import only the specific modules we need, bypassing
            # the broken ALIKED → kornia → flash_attn chain
            from lightglue.superpoint import SuperPoint
            from lightglue.lightglue import LightGlue
            from lightglue.utils import rbd

            self._torch = torch
            self._rbd = rbd

            actual_device = self._device
            if actual_device == "cuda" and not torch.cuda.is_available():
                actual_device = "cpu"
                logger.warning("CUDA not available, falling back to CPU for LightGlue")
            self._device = actual_device

            self._torch_device = _canonical_torch_device(self._device)
            self._extractor = SuperPoint(
                max_num_keypoints=self._max_kps
            ).eval().to(self._torch_device)

            self._matcher = LightGlue(
                features="superpoint"
            ).eval().to(self._torch_device)

            self._loaded = True
            logger.info(
                f"LightGlueMatcher initialized: device={self._torch_device}, "
                f"max_kps={self._max_kps}, resize_max={self._resize_max}"
            )
        except ImportError as e:
            logger.error(
                f"Failed to import LightGlue: {e}. "
                "Install with: pip install git+https://github.com/cvg/LightGlue.git"
            )
            self._loaded = False
        except Exception as e:
            logger.error(f"LightGlue init failed: {e}")
            self._loaded = False

    def _preprocess_image(self, img: np.ndarray):
        """
        Convert numpy image to torch tensor for SuperPoint.

        Args:
            img: (H, W, 3) uint8 RGB or (H, W) grayscale

        Returns:
            (1, 1, H', W') float32 torch tensor (grayscale, normalized)
        """
        torch = self._torch
        import cv2

        # Convert to grayscale if needed
        if img.ndim == 3 and img.shape[2] == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            gray = img

        h, w = gray.shape[:2]

        # Resize if needed
        if max(h, w) > self._resize_max:
            scale = self._resize_max / max(h, w)
            new_w, new_h = int(w * scale), int(h * scale)
            gray = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # Convert to float tensor: (1, 1, H, W), range [0, 1]
        tensor = torch.from_numpy(gray).float() / 255.0
        tensor = tensor.unsqueeze(0).unsqueeze(0)  # add batch + channel
        dev = getattr(self, "_torch_device", None)
        if dev is None:
            dev = _canonical_torch_device(self._device)
        return tensor.to(dev)

    def match(self, query_img: np.ndarray, ref_img: np.ndarray) -> MatchResult:
        """
        Match features between query and reference images.

        Args:
            query_img: (H, W, 3) uint8 RGB query image
            ref_img: (H, W, 3) uint8 RGB reference image

        Returns:
            MatchResult with keypoints, confidence, homography, and inlier mask.
        """
        if not self._loaded:
            logger.warning("LightGlue not loaded — falling back to ORB")
            return ORBMatcher().match(query_img, ref_img)

        torch = self._torch

        try:
            dev = getattr(self, "_torch_device", None) or _canonical_torch_device(self._device)

            with torch.no_grad():
                # Pin the current CUDA device so ops that rely on the default stream/device
                # do not mix with CPU tensors (seen as wrapper_CUDA_bmm mismatches).
                if dev.type == "cuda" and torch.cuda.is_available():
                    torch.cuda.set_device(dev.index if dev.index is not None else 0)

                # Ensure full modules stay on the intended device (some third-party paths
                # can leave buffers on CPU across runs or internal forwards).
                if self._extractor is not None:
                    self._extractor.to(dev)
                if self._matcher is not None:
                    self._matcher.to(dev)

                # Preprocess
                img0 = self._preprocess_image(query_img)
                img1 = self._preprocess_image(ref_img)

                # Extract features
                feats0 = self._extractor.extract(img0)
                feats1 = self._extractor.extract(img1)
                feats0 = _recursive_to_device(feats0, dev)
                feats1 = _recursive_to_device(feats1, dev)

                bundle = {"image0": feats0, "image1": feats1}
                bundle = _recursive_to_device(bundle, dev)
                devs0 = _collect_tensor_devices(bundle["image0"])
                devs1 = _collect_tensor_devices(bundle["image1"])
                bad0 = [(p, d) for p, d in devs0 if not _tensor_devices_match(dev, d)]
                bad1 = [(p, d) for p, d in devs1 if not _tensor_devices_match(dev, d)]
                if bad0 or bad1:
                    logger.error(
                        "LightGlue feature device mismatch before matcher: dev=%s image0=%s image1=%s",
                        dev,
                        bad0[:6],
                        bad1[:6],
                    )
                    bundle = _recursive_to_device(bundle, dev)

                # Match
                matches01 = self._matcher(bundle)

                # Remove batch dimension
                feats0, feats1, matches01 = [
                    self._rbd(x) for x in [feats0, feats1, matches01]
                ]

                # Extract matched keypoints
                matches = matches01["matches"]  # (N, 2) index pairs
                scores = matches01.get("scores", None)

                if len(matches) < 4:
                    return MatchResult(
                        np.zeros((0, 2)), np.zeros((0, 2)), np.zeros((0,))
                    )

                # Get matched keypoint coordinates
                kpts0 = feats0["keypoints"]  # (K0, 2)
                kpts1 = feats1["keypoints"]  # (K1, 2)

                q_pts = kpts0[matches[..., 0]].cpu().numpy()  # (N, 2)
                r_pts = kpts1[matches[..., 1]].cpu().numpy()  # (N, 2)

                # Scale coordinates back to original image size if resized
                h0, w0 = query_img.shape[:2]
                h1, w1 = ref_img.shape[:2]
                proc_h0, proc_w0 = img0.shape[2], img0.shape[3]
                proc_h1, proc_w1 = img1.shape[2], img1.shape[3]

                if (proc_h0, proc_w0) != (h0, w0):
                    scale_x0 = w0 / proc_w0
                    scale_y0 = h0 / proc_h0
                    q_pts[:, 0] *= scale_x0
                    q_pts[:, 1] *= scale_y0

                if (proc_h1, proc_w1) != (h1, w1):
                    scale_x1 = w1 / proc_w1
                    scale_y1 = h1 / proc_h1
                    r_pts[:, 0] *= scale_x1
                    r_pts[:, 1] *= scale_y1

                # Confidence scores
                if scores is not None:
                    conf = scores.cpu().numpy().astype(np.float64)
                else:
                    conf = np.ones(len(q_pts), dtype=np.float64)

                # Compute homography with RANSAC
                import cv2

                homography = None
                inlier_mask = None
                if len(q_pts) >= 4:
                    homography, mask = cv2.findHomography(
                        q_pts.astype(np.float32),
                        r_pts.astype(np.float32),
                        cv2.USAC_MAGSAC,
                        20.0,
                    )
                    if mask is not None:
                        inlier_mask = mask.ravel().astype(bool)

                logger.debug(
                    f"LightGlue: {len(q_pts)} matches, "
                    f"{np.sum(inlier_mask) if inlier_mask is not None else len(q_pts)} inliers"
                )

                shape_conf = compute_shape_confidence(
                    homography=homography,
                    query_size=(query_img.shape[1], query_img.shape[0]),
                    ref_size=(ref_img.shape[1], ref_img.shape[0]),
                )
                return MatchResult(
                    query_keypoints=q_pts.astype(np.float64),
                    ref_keypoints=r_pts.astype(np.float64),
                    confidence=conf,
                    homography=homography,
                    inlier_mask=inlier_mask,
                    metadata={
                        "matcher": "lightglue",
                        "feature_backend": getattr(self, "_feature_backend", "superpoint"),
                        "shape_confidence": shape_conf,
                    },
                )

        except Exception as e:
            logger.error(f"LightGlue matching failed: {e}", exc_info=True)
            if "Expected all tensors to be on the same device" in str(e):
                # Let caller decide if/how to retry on backend downgrade.
                raise RuntimeError(str(e)) from e
            return MatchResult(np.zeros((0, 2)), np.zeros((0, 2)), np.zeros((0,)))


class AlikedLightGlueMatcher(LightGlueMatcher):
    """ALIKED + LightGlue matcher with fallback to SuperPoint+LightGlue."""

    def _init_models(self):
        if os.environ.get("HD4_FORCE_SUPERPOINT_LIGHTGLUE", "").strip().lower() in {"1", "true", "yes", "on"}:
            self._feature_backend = "superpoint"
            self._aliked_retry_done = True
            logger.warning(
                "HD4_FORCE_SUPERPOINT_LIGHTGLUE is set; skipping ALIKED and using SuperPoint+LightGlue."
            )
            super()._init_models()
            return
        self._feature_backend = "aliked"
        self._aliked_retry_done = False
        try:
            os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
            import torch
            from lightglue.aliked import ALIKED
            from lightglue.lightglue import LightGlue
            from lightglue.utils import rbd

            self._torch = torch
            self._rbd = rbd

            actual_device = self._device
            if actual_device == "cuda" and not torch.cuda.is_available():
                actual_device = "cpu"
                logger.warning("CUDA not available, falling back to CPU for ALIKED+LightGlue")
            self._device = actual_device
            self._torch_device = _canonical_torch_device(actual_device)

            self._extractor = ALIKED(max_num_keypoints=self._max_kps).eval().to(self._torch_device)
            self._matcher = LightGlue(features="aliked").eval().to(self._torch_device)
            self._loaded = True
            logger.info(
                "AlikedLightGlueMatcher initialized: device=%s, max_kps=%s, resize_max=%s",
                self._torch_device, self._max_kps, self._resize_max
            )
        except Exception as e:
            logger.warning("ALIKED+LightGlue init failed (%s), using SuperPoint+LightGlue", e)
            self._feature_backend = "superpoint"
            super()._init_models()

    def _fallback_to_superpoint(self) -> bool:
        """Switch extractor+matcher to SuperPoint+LightGlue once per instance."""
        if self._aliked_retry_done:
            return False
        self._aliked_retry_done = True
        logger.warning(
            "ALIKED+LightGlue hit CUDA device mismatch; retrying with SuperPoint+LightGlue."
        )
        self._feature_backend = "superpoint"
        super()._init_models()
        return bool(self._loaded)

    def match(self, query_img: np.ndarray, ref_img: np.ndarray) -> MatchResult:
        try:
            return LightGlueMatcher.match(self, query_img, ref_img)
        except RuntimeError as exc:
            msg = str(exc)
            if (
                self._feature_backend == "aliked"
                and "Expected all tensors to be on the same device" in msg
                and self._fallback_to_superpoint()
            ):
                return LightGlueMatcher.match(self, query_img, ref_img)
            raise


class RoMaMatcher:
    """RoMa outdoor matcher with explicit pretrained asset loading."""

    def __init__(
        self,
        device: str = "cuda",
        roma_weights_path: str | None = None,
        dinov2_weights_path: str | None = None,
        sample_num_matches: int = 3000,
        strict_assets: bool = True,
    ):
        self._device = device
        self._sample_num = int(sample_num_matches)
        self._strict_assets = bool(strict_assets)
        self._loaded = False
        self._model = None

        repo_root = _resolve_matching_models_root()
        default_roma = repo_root / "Matching_Models" / "RoMa" / "weights" / "roma_outdoor.pth"
        default_dino = repo_root / "Matching_Models" / "RoMa" / "weights" / "dinov2_vitl14_pretrain.pth"
        self._roma_weights = Path(
            roma_weights_path or os.environ.get("ROMA_WEIGHTS_PATH", str(default_roma))
        )
        self._dino_weights = Path(
            dinov2_weights_path or os.environ.get("DINOV2_WEIGHTS_PATH", str(default_dino))
        )
        self._init_model()

    def _init_model(self):
        import torch

        actual_device = self._device
        if actual_device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("RoMa requested CUDA device but CUDA is not available")
        if actual_device != "cuda":
            logger.warning("RoMa is configured for %s; this can be very slow.", actual_device)

        from Matching_Models.RoMa.roma import roma_outdoor

        have_local_assets = self._roma_weights.exists() and self._dino_weights.exists()
        if not have_local_assets and self._strict_assets:
            raise FileNotFoundError(
                f"RoMa assets missing. Expected ROMA={self._roma_weights}, DINO={self._dino_weights}. "
                "Set ROMA_WEIGHTS_PATH and DINOV2_WEIGHTS_PATH to pretrained files."
            )

        if have_local_assets:
            logger.info("RoMaMatcher using local weights: roma=%s dino=%s", self._roma_weights, self._dino_weights)
            self._model = roma_outdoor(
                device=actual_device,
                weights=str(self._roma_weights),
                dinov2_weights=str(self._dino_weights),
            )
        else:
            logger.warning("RoMa local assets not found; falling back to torch.hub download path.")
            self._model = roma_outdoor(device=actual_device)
        self._device = actual_device
        self._loaded = True

    @staticmethod
    def _to_rgb_uint8(img: np.ndarray) -> np.ndarray:
        import cv2

        if img.ndim == 2:
            return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        if img.dtype != np.uint8:
            arr = np.clip(img, 0, 255).astype(np.uint8)
        else:
            arr = img
        return arr

    def match(self, query_img: np.ndarray, ref_img: np.ndarray) -> MatchResult:
        if not self._loaded or self._model is None:
            raise RuntimeError("RoMaMatcher is not initialized")

        from PIL import Image
        import cv2

        query_rgb = self._to_rgb_uint8(query_img)
        ref_rgb = self._to_rgb_uint8(ref_img)
        h0, w0 = query_rgb.shape[:2]
        h1, w1 = ref_rgb.shape[:2]

        img0_pil = Image.fromarray(query_rgb)
        img1_pil = Image.fromarray(ref_rgb)

        warp, certainty = self._model.match(img0_pil, img1_pil, device=self._device)
        matches, certainty = self._model.sample(warp, certainty, num=self._sample_num)
        kpts0_t, kpts1_t = self._model.to_pixel_coordinates(matches, h0, w0, h1, w1)
        q_pts = kpts0_t.detach().cpu().numpy().astype(np.float64)
        r_pts = kpts1_t.detach().cpu().numpy().astype(np.float64)
        conf = certainty.detach().cpu().numpy().astype(np.float64)

        if len(q_pts) < 4:
            return MatchResult(np.zeros((0, 2)), np.zeros((0, 2)), np.zeros((0,)))

        homography = None
        inlier_mask = None
        homography, mask = cv2.findHomography(
            q_pts.astype(np.float32),
            r_pts.astype(np.float32),
            cv2.USAC_MAGSAC,
            20.0,
        )
        if mask is not None:
            inlier_mask = mask.ravel().astype(bool)
        metadata = {
            "matcher": "roma",
            "roma_sampled_matches": int(len(q_pts)),
            "shape_confidence": compute_shape_confidence(
                homography=homography,
                query_size=(query_img.shape[1], query_img.shape[0]),
                ref_size=(ref_img.shape[1], ref_img.shape[0]),
            ),
        }
        return MatchResult(
            query_keypoints=q_pts,
            ref_keypoints=r_pts,
            confidence=conf,
            homography=homography,
            inlier_mask=inlier_mask,
            metadata=metadata,
        )


def create_matcher(
    name: str,
    device: str = "cpu",
    roma_weights_path: str | None = None,
    dinov2_weights_path: str | None = None,
    strict_roma: bool = False,
    max_num_keypoints: int | None = None,
    resize_max: int | None = None,
):
    """
    Factory for matcher backend.

    Args:
        name: 'orb', 'lightglue', or 'roma'
        device: 'cpu' or 'cuda' (only used for LightGlue)

    Returns:
        Matcher instance with .match(query, ref) → MatchResult
    """
    max_kps = int(max_num_keypoints) if max_num_keypoints is not None else 2048
    resize_cap = int(resize_max) if resize_max is not None else 1024

    if name == "aliked_lightglue":
        matcher = AlikedLightGlueMatcher(
            max_num_keypoints=max_kps,
            device=device,
            resize_max=resize_cap,
        )
        if matcher._loaded:
            return matcher
        logger.warning("ALIKED+LightGlue unavailable — trying SuperPoint+LightGlue before ORB")
        sp_matcher = LightGlueMatcher(
            max_num_keypoints=max_kps,
            device=device,
            resize_max=resize_cap,
        )
        if sp_matcher._loaded:
            return sp_matcher
        logger.warning("SuperPoint+LightGlue unavailable — falling back to ORB")
        return ORBMatcher()

    if name == "lightglue":
        matcher = LightGlueMatcher(
            max_num_keypoints=max_kps,
            device=device,
            resize_max=resize_cap,
        )
        if matcher._loaded:
            return matcher
        logger.warning("LightGlue unavailable — falling back to ORB")
        return ORBMatcher()

    if name == "roma":
        try:
            return RoMaMatcher(
                device=device,
                roma_weights_path=roma_weights_path,
                dinov2_weights_path=dinov2_weights_path,
                strict_assets=True,
            )
        except Exception as exc:
            if strict_roma:
                raise RuntimeError(f"RoMa init failed (strict mode): {exc}") from exc
            logger.warning("RoMa unavailable (%s) — falling back to ORB", exc)
            return ORBMatcher()

    return ORBMatcher()
