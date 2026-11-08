"""
AnyVisLoc "scene" adapter (format ``anyvisloc_ref_png_npy_json_v1``).

Standalone on purpose: the 2026-08 AnyVisLoc re-release ships a layout that has
nothing in common with the ``AVLData/`` tree :class:`AVLDataset` reads, so this
lives in its own module and the old adapter is left untouched.  Reverting is a
two-line diff (registry entry + DATASET_CONFIGS entry).

Layout (one directory per scene, unzipped from ``Scene_XX.zip``)::

    Scene_05/
      L05_0001.npz ... L05_1118.npz   # image + K + pose, one per UAV frame
      L05_reference.json              # per-mode map metadata
      aerial_map.png    aerial_dsm.npy
      satellite_map.png satellite_dsm.npy

Everything is expressed in a **local metric frame**: no lat/lon, no UTM, no
GeoTIFF.  The benchmark runner is geodetic end to end, so we bolt the local
frame onto a synthetic local tangent plane (``anchor_lat``/``anchor_lon``).
Only relative geometry is meaningful; absolute coordinates are fictitious and
must never be reported as the true site location.

Conventions, measured on Scene_05 (see ``__main__`` self-check):
  * map pixel  ``col = (x - origin_x) / res``, ``row = (y - origin_y) / res``
  * ``euler_deg`` == Euler angles of ``pose_c2w``: (roll, tilt-from-nadir, yaw)
  * world->image is a pure rotation by ``-euler_deg[2]``, no reflection, so
    ``euler_deg[2]`` is already the heading of the image "up" direction, in
    degrees clockwise from local north (= local -y). Measured from consecutive-
    frame optical flow on Scene_05: 12 pairs, residual median 1.5 deg, MAD 1.0.
  * local +x is east-like, local +y is south-like (row grows with y)
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np

from .dataset_adapters import BaseDataset, DatasetFactory

logger = logging.getLogger(__name__)

# Metres per degree on the synthesized tangent plane (WGS84 mean radius).
_M_PER_DEG = 111320.0


class AVLSceneDataset(BaseDataset):
    """AnyVisLoc scene directory (``Scene_XX/``) with npz frames and PNG maps."""

    def __init__(self,
                 data_root: Path,
                 scene: Optional[str] = None,
                 ref_mode: str = "aerial",
                 tile_size_px: int = 2048,
                 anchor_lat: float = 30.0,
                 anchor_lon: float = 120.0,
                 **kwargs):
        """
        Args:
            data_root: either the ``Scene_XX`` directory itself, or a parent
                directory containing it (then pass ``scene``).
            scene: scene directory name, e.g. ``"Scene_05"``.
            ref_mode: ``"aerial"`` (~0.05 m/px) or ``"satellite"`` (~0.14 m/px).
            tile_size_px: reference tile edge, in reference-map pixels.
            anchor_lat/anchor_lon: synthetic tangent-plane origin. Fictitious.
        """
        root = Path(data_root)
        if scene:
            root = root / scene
        super().__init__(root, **kwargs)

        self._scene = self._data_root.name
        self._ref_mode = str(ref_mode).lower()
        self._tile_size_px = max(128, int(tile_size_px))
        self._anchor_lat = float(anchor_lat)
        self._anchor_lon = float(anchor_lon)
        self._lon_scale = _M_PER_DEG * max(1e-6, math.cos(math.radians(self._anchor_lat)))

        self._ref_meta: Dict[str, Any] = {}
        self._ref_res: float = 0.0
        self._ref_origin: Tuple[float, float] = (0.0, 0.0)
        self._ref_image: Optional[np.ndarray] = None
        self._dsm_path: Optional[Path] = None

    @property
    def name(self) -> str:
        return f"AVLScene-{self._scene}-{self._ref_mode}"

    # ------------------------------------------------------------------ load

    def load(self):
        if self._loaded:
            return

        ref_json = self._find_reference_json()
        modes = json.loads(ref_json.read_text()).get("modes", {})
        if self._ref_mode not in modes:
            raise KeyError(
                f"{self.name}: ref_mode {self._ref_mode!r} not in {sorted(modes)}"
            )
        meta = modes[self._ref_mode]
        self._ref_meta = meta
        self._ref_res = float(meta["map_resolution"][0])
        ox, oy = meta["map_origin_local"]
        self._ref_origin = (float(ox), float(oy))
        self._dsm_path = self._resolve(meta.get("dsm_path"), f"{self._ref_mode}_dsm.npy")

        for npz_path in sorted(self._data_root.glob("*.npz")):
            entry = self._entry_from_npz(npz_path)
            if entry is not None:
                self._entries.append(entry)

        if not self._entries:
            raise FileNotFoundError(f"{self.name}: no *.npz frames under {self._data_root}")

        self._loaded = True
        logger.info("%s: loaded %d frames (res=%.4f m/px)",
                    self.name, len(self._entries), self._ref_res)

    def _find_reference_json(self) -> Path:
        matches = sorted(self._data_root.glob("*_reference.json"))
        if not matches:
            raise FileNotFoundError(f"{self.name}: no *_reference.json in {self._data_root}")
        return matches[0]

    def _resolve(self, relpath: Optional[str], fallback: str) -> Optional[Path]:
        for cand in (relpath, fallback):
            if not cand:
                continue
            p = Path(cand)
            p = p if p.is_absolute() else self._data_root / p.name
            if p.exists():
                return p
        return None

    def _entry_from_npz(self, npz_path: Path) -> Optional[Dict[str, Any]]:
        try:
            with np.load(npz_path, allow_pickle=True) as data:
                xyz = np.asarray(data["xyz"], dtype=float).reshape(3)
                euler = np.asarray(data["euler_deg"], dtype=float).reshape(3)
                K = np.asarray(data["K"], dtype=float).reshape(3, 3)
                h, w = (int(v) for v in np.asarray(data["image_size"]).reshape(2))
                downsample = float(data["uav_downsample"]) if "uav_downsample" in data else 1.0
        except Exception as exc:  # pragma: no cover - corrupt archive
            logger.warning("%s: cannot read %s (%s)", self.name, npz_path.name, exc)
            return None

        lat, lon = self.local_to_latlon(xyz[0], xyz[1])
        return {
            "image_name": npz_path.stem,
            "query_path": npz_path,
            "gt_pose": {
                "lat": lat,
                "lon": lon,
                "alt": float(xyz[2]),
                "yaw": float(euler[2]) % 360.0,
                "pitch": float(euler[1]),
                "roll": float(euler[0]),
                "pitch_convention": "aircraft_nadir_zero",
            },
            "camera": {
                "fx": float(K[0, 0]), "fy": float(K[1, 1]),
                "cx": float(K[0, 2]), "cy": float(K[1, 2]),
                "width": w, "height": h,
            },
            "metadata": {
                "scene": self._scene,
                "ref_mode": self._ref_mode,
                "local_xyz": [float(v) for v in xyz],
                "euler_deg": [float(v) for v in euler],
                "uav_downsample": downsample,
                "anchor_lat": self._anchor_lat,
                "anchor_lon": self._anchor_lon,
                "coordinate_frame": "synthetic_tangent_plane",
                "dsm_path": str(self._dsm_path) if self._dsm_path else None,
            },
        }

    # ------------------------------------------------------------ geometry

    def local_to_latlon(self, x: float, y: float) -> Tuple[float, float]:
        """Local metric (x east-like, y south-like) -> synthetic lat/lon."""
        lat = self._anchor_lat - float(y) / _M_PER_DEG
        lon = self._anchor_lon + float(x) / self._lon_scale
        return lat, lon

    def latlon_to_local(self, lat: float, lon: float) -> Tuple[float, float]:
        x = (float(lon) - self._anchor_lon) * self._lon_scale
        y = (self._anchor_lat - float(lat)) * _M_PER_DEG
        return x, y

    def _latlon_to_pixel(self, lat: float, lon: float) -> Tuple[int, int]:
        x, y = self.latlon_to_local(lat, lon)
        ox, oy = self._ref_origin
        return int(round((x - ox) / self._ref_res)), int(round((y - oy) / self._ref_res))

    def _pixel_to_latlon(self, col: float, row: float) -> Tuple[float, float]:
        ox, oy = self._ref_origin
        return self.local_to_latlon(ox + float(col) * self._ref_res,
                                    oy + float(row) * self._ref_res)

    # ------------------------------------------------------------ reference

    def _load_reference_map(self) -> Optional[np.ndarray]:
        if self._ref_image is not None:
            return self._ref_image
        if not self._loaded:
            self.load()

        path = self._resolve(self._ref_meta.get("map_path"), f"{self._ref_mode}_map.png")
        if path is None:
            logger.warning("%s: reference map not found", self.name)
            return None

        import cv2
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            logger.warning("%s: cannot decode %s", self.name, path)
            return None
        self._ref_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return self._ref_image

    def _crop_reference_tile(self, lat: float, lon: float) -> Optional[np.ndarray]:
        ref = self._load_reference_map()
        if ref is None:
            return None

        col, row = self._latlon_to_pixel(lat, lon)
        half = self._tile_size_px // 2
        return _padded_crop(ref, row - half, col - half, self._tile_size_px)

    # ---------------------------------------------------------------- frames

    def _get_frame(self, idx: int) -> Dict[str, Any]:
        entry = self._entries[idx]
        gt = entry["gt_pose"]
        camera = entry["camera"]

        frame = {
            "image_name": entry["image_name"],
            "query_path": entry["query_path"],
            "gt_pose": gt,
            "metadata": entry["metadata"],
            "camera": camera,
            "gsd": self._ref_res,
        }

        try:
            with np.load(entry["query_path"], allow_pickle=True) as data:
                frame["query_image"] = np.asarray(data["image"], dtype=np.uint8)
        except Exception:
            frame["query_image"] = np.zeros(
                (int(camera["height"]), int(camera["width"]), 3), dtype=np.uint8
            )

        frame["reference_image"] = self._crop_reference_tile(gt["lat"], gt["lon"])
        frame["reference_path"] = self._resolve(
            self._ref_meta.get("map_path"), f"{self._ref_mode}_map.png"
        )
        return frame

    def iter_reference_corpus(self) -> Iterator[Dict[str, Any]]:
        ref = self._load_reference_map()
        if ref is None:
            logger.warning("%s: reference map unavailable, retrieval corpus empty.", self.name)
            return

        height, width = ref.shape[:2]
        tile = self._tile_size_px
        step = max(128, tile // 2)  # 50% overlap, mirrors AVLDataset
        emitted = 0

        for row_start in range(0, height, step):
            for col_start in range(0, width, step):
                h = min(height, row_start + tile) - row_start
                w = min(width, col_start + tile) - col_start
                if h < tile // 3 or w < tile // 3:
                    continue

                patch = _padded_crop(ref, row_start, col_start, tile)
                if patch is None:
                    continue

                lat, lon = self._pixel_to_latlon(col_start + w / 2.0, row_start + h / 2.0)
                emitted += 1
                yield {
                    "ref_name": f"{self._scene}_{self._ref_mode}_r{row_start:06d}_c{col_start:06d}.png",
                    "ref_image": patch,
                    "anchor": {
                        "lat": lat,
                        "lon": lon,
                        "anchor_source": "scene_local_frame",
                        "row_start": int(row_start),
                        "col_start": int(col_start),
                        "tile_size_px": int(tile),
                        "tile_step_px": int(step),
                    },
                }

        logger.info("%s: built retrieval corpus with %d tiles (tile=%d, step=%d).",
                    self.name, emitted, tile, step)


def _padded_crop(image: np.ndarray, row: int, col: int, size: int) -> Optional[np.ndarray]:
    """Crop ``size x size`` at (row, col); zero-pad where it leaves the image."""
    height, width = image.shape[:2]
    r0, c0 = max(0, row), max(0, col)
    r1, c1 = min(height, row + size), min(width, col + size)
    if r1 <= r0 or c1 <= c0:
        return None

    out = np.zeros((size, size, 3), dtype=np.uint8)
    out[r0 - row:r1 - row, c0 - col:c1 - col] = image[r0:r1, c0:c1, :3]
    return out


def _self_check(scene_dir: str) -> None:
    """Guard the two conventions that are easy to silently break."""
    ds = AVLSceneDataset(scene_dir, ref_mode="satellite")
    ds.load()
    assert len(ds) > 0

    # 1. lat/lon <-> local <-> pixel round-trips.
    for entry in ds._entries[:50]:
        gt = entry["gt_pose"]
        x, y, _ = entry["metadata"]["local_xyz"]
        rx, ry = ds.latlon_to_local(gt["lat"], gt["lon"])
        assert abs(rx - x) < 0.05 and abs(ry - y) < 0.05, (rx, ry, x, y)
        col, row = ds._latlon_to_pixel(gt["lat"], gt["lon"])
        assert abs(col - (x - ds._ref_origin[0]) / ds._ref_res) < 1.0
        assert abs(row - (y - ds._ref_origin[1]) / ds._ref_res) < 1.0

    # 2. col<->x / row<->y must not be swapped: the swapped mapping puts most
    #    frames outside the map, the correct one keeps most inside.
    ref = ds._load_reference_map()
    assert ref is not None
    height, width = ref.shape[:2]
    ox, oy = ds._ref_origin
    pts = np.array([e["metadata"]["local_xyz"][:2] for e in ds._entries])
    cols, rows = (pts[:, 0] - ox) / ds._ref_res, (pts[:, 1] - oy) / ds._ref_res
    inside = np.mean((cols >= 0) & (cols < width) & (rows >= 0) & (rows < height))
    swapped = np.mean((rows >= 0) & (rows < width) & (cols >= 0) & (cols < height))
    assert inside > swapped, f"col/row mapping looks swapped ({inside:.2f} vs {swapped:.2f})"

    frame = ds[0]
    assert frame["query_image"].ndim == 3
    assert frame["reference_image"].shape == (ds._tile_size_px, ds._tile_size_px, 3)

    # 3. yaw convention: world->image must be a rotation by -euler_deg[2].
    #    Consecutive-frame optical flow gives the motion direction in the image;
    #    it must equal the world motion direction minus the yaw.
    residuals = _flow_yaw_residuals(ds, max_pairs=8)
    assert len(residuals) >= 4, f"too few usable frame pairs ({len(residuals)})"
    med = float(np.median(residuals))
    assert abs(med) < 10.0, f"yaw convention off by {med:.1f} deg"

    print(f"OK: {ds.name}, {len(ds)} frames, {inside:.1%} inside map, "
          f"gsd={ds._ref_res:.4f}, yaw residual {med:+.1f} deg over {len(residuals)} pairs")


def _flow_yaw_residuals(ds: "AVLSceneDataset", max_pairs: int = 8) -> List[float]:
    """Signed degrees between the measured and predicted image motion direction."""
    import cv2

    sift = cv2.SIFT_create(3000)
    matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    out: List[float] = []
    prev = None

    for entry in ds._entries[::7]:
        if len(out) >= max_pairs:
            break
        with np.load(entry["query_path"], allow_pickle=True) as data:
            cur = (np.asarray(data["image"]),
                   np.asarray(entry["metadata"]["local_xyz"], dtype=float),
                   np.asarray(entry["metadata"]["euler_deg"], dtype=float))
        if prev is not None:
            d = cur[1] - prev[1]
            turn = abs(((cur[2][2] - prev[2][2] + 180.0) % 360.0) - 180.0)
            if math.hypot(d[0], d[1]) > 3.0 and abs(prev[2][1]) < 12.0 and turn < 8.0:
                shift = _image_shift(sift, matcher, prev[0], cur[0])
                if shift is not None:
                    alpha = math.degrees(math.atan2(d[1], d[0]))
                    gamma = math.degrees(math.atan2(-shift[1], -shift[0]))
                    out.append(((gamma - alpha + float(prev[2][2])) + 180.0) % 360.0 - 180.0)
        prev = cur
    return out


def _image_shift(sift, matcher, img_a: np.ndarray, img_b: np.ndarray):
    """Translation at the image centre from a -> b, or None if unreliable."""
    import cv2

    a = cv2.cvtColor(img_a, cv2.COLOR_RGB2GRAY)
    b = cv2.cvtColor(img_b, cv2.COLOR_RGB2GRAY)
    k1, d1 = sift.detectAndCompute(a, None)
    k2, d2 = sift.detectAndCompute(b, None)
    if d1 is None or d2 is None:
        return None
    matches = matcher.match(d1, d2)
    if len(matches) < 20:
        return None
    src = np.float32([k1[m.queryIdx].pt for m in matches])
    dst = np.float32([k2[m.trainIdx].pt for m in matches])
    M, mask = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC,
                                          ransacReprojThreshold=3.0)
    if M is None or int(mask.sum()) < 15:
        return None
    if not 0.8 < math.hypot(M[0, 0], M[1, 0]) < 1.25:
        return None
    centre = np.array([a.shape[1] / 2.0, a.shape[0] / 2.0, 1.0])
    return M @ centre - centre[:2]


DatasetFactory.DATASETS.setdefault("avl_scene", AVLSceneDataset)
DatasetFactory.DEFAULT_ROOTS.setdefault("avl_scene", "/workspace/forge/data/AnyVisLoc_scenes")


if __name__ == "__main__":  # PYTHONPATH=src python3 -m data.avl_scene_adapter <Scene_XX dir>
    import sys
    _self_check(sys.argv[1] if len(sys.argv) > 1 else "Scene_05")
