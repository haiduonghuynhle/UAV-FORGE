from __future__ import annotations

import os
import time
import numpy as np
from pathlib import Path
from loguru import logger
from typing import Any

from ortholoc.image_matching import Matcher
from ortholoc.correspondences import Correspondences2D2D


class MatcherRoMa(Matcher):
    """RoMa-based matcher implementing the OrthoLoC Matcher interface."""

    def __init__(
        self,
        device: str = "cuda",
        sample_num_matches: int = 3000,
        angles: list[float] | None = None,
    ) -> None:
        if angles is None:
            angles = [0]

        self._sample_num = int(sample_num_matches)
        self._model = None

        t0 = time.time()
        logger.info("MatcherRoMa: loading RoMa outdoor model")

        import sys
        repo_root = Path(__file__).resolve().parents[3]
        roma_parent = str(repo_root / "Matching_Models" / "RoMa")
        if roma_parent not in sys.path:
            sys.path.insert(0, str(repo_root))
            sys.path.insert(0, roma_parent)

        from Matching_Models.RoMa.roma import roma_outdoor

        roma_weights = Path(
            os.environ.get("ROMA_WEIGHTS_PATH", str(repo_root / "Matching_Models" / "RoMa" / "weights" / "roma_outdoor.pth"))
        )
        dino_weights = Path(
            os.environ.get("DINOV2_WEIGHTS_PATH", str(repo_root / "Matching_Models" / "RoMa" / "weights" / "dinov2_vitl14_pretrain.pth"))
        )

        if roma_weights.exists() and dino_weights.exists():
            self._model = roma_outdoor(
                device=device,
                weights=str(roma_weights),
                dinov2_weights=str(dino_weights),
            )
        else:
            self._model = roma_outdoor(device=device)

        logger.info(f"MatcherRoMa: loaded in {time.time() - t0:.1f}s")
        super().__init__(name="roma_outdoor", device=device, angles=angles)

    def __call__(
        self,
        img0: np.ndarray,
        img1: np.ndarray,
        covisible_only: bool = True,
        normalized: bool = True,
        silent: bool = False,
        *args: Any,
        **kwargs: Any,
    ) -> Correspondences2D2D:
        from PIL import Image

        if not silent:
            logger.info("MatcherRoMa: matching...")
        t0 = time.time()

        h0, w0 = img0.shape[:2]
        h1, w1 = img1.shape[:2]

        img0_rgb = self._ensure_rgb_uint8(img0)
        img1_rgb = self._ensure_rgb_uint8(img1)

        img0_pil = Image.fromarray(img0_rgb)
        img1_pil = Image.fromarray(img1_rgb)

        warp, certainty = self._model.match(img0_pil, img1_pil, device=self.device)
        matches, certainty_sampled = self._model.sample(warp, certainty, num=self._sample_num)
        kpts0, kpts1 = self._model.to_pixel_coordinates(matches, h0, w0, h1, w1)

        pts0 = kpts0.detach().cpu().numpy().astype(np.float64)
        pts1 = kpts1.detach().cpu().numpy().astype(np.float64)
        confs = certainty_sampled.detach().cpu().numpy().astype(np.float64).ravel()

        correspondences = self.build_correspondences(
            pts0=pts0,
            pts1=pts1,
            h0=h0,
            w0=w0,
            h1=h1,
            w1=w1,
            confidences=confs,
            covisible_only=covisible_only,
            normalized=normalized,
        )

        if not silent:
            logger.info(f"MatcherRoMa: {len(correspondences)} correspondences in {time.time() - t0:.2f}s")
        return correspondences

    @staticmethod
    def _ensure_rgb_uint8(img: np.ndarray) -> np.ndarray:
        import cv2

        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        if img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
        elif img.shape[2] == 3:
            pass
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)
        return img
