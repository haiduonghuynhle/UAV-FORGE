"""DSM-backed map utilities for 2D-3D UAV localization."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class DSMPoint:
    """Metric ENU point sampled from a reference DSM."""

    east_m: float
    north_m: float
    up_m: float
    valid: bool = True

    def as_array(self) -> np.ndarray:
        return np.array([self.east_m, self.north_m, self.up_m], dtype=np.float64)


class BaseDSMMap:
    """Minimal DSM map contract consumed by the DSM-PnP pose head."""

    available: bool = False
    status: str = "unavailable"

    def pixel_to_enu(self, u: float, v: float, tile_meta: Optional[Dict[str, Any]] = None) -> Optional[DSMPoint]:
        raise NotImplementedError

    def enu_to_latlon(self, east_m: float, north_m: float, reference_lat: float, reference_lon: float) -> Tuple[float, float]:
        lat_scale = 111320.0
        lon_scale = max(1.0, 111320.0 * np.cos(np.radians(float(reference_lat))))
        return (
            float(reference_lat) + float(north_m) / lat_scale,
            float(reference_lon) + float(east_m) / lon_scale,
        )


class ArrayDSMMap(BaseDSMMap):
    """In-memory DSM map for tests and non-georeferenced planar diagnostics."""

    def __init__(
        self,
        dsm: np.ndarray,
        gsd_m: float,
        origin_lat: float,
        origin_lon: float,
        origin_col: float = 0.0,
        origin_row: float = 0.0,
    ) -> None:
        self._dsm = np.asarray(dsm, dtype=np.float32)
        self._gsd_m = float(gsd_m)
        self._origin_lat = float(origin_lat)
        self._origin_lon = float(origin_lon)
        self._origin_col = float(origin_col)
        self._origin_row = float(origin_row)
        self.available = self._dsm.ndim == 2 and self._dsm.size > 0 and self._gsd_m > 0.0
        self.status = "available" if self.available else "invalid_array_dsm"

    def pixel_to_enu(self, u: float, v: float, tile_meta: Optional[Dict[str, Any]] = None) -> Optional[DSMPoint]:
        if not self.available:
            return None
        tile_meta = tile_meta or {}
        col = float(tile_meta.get("col_start", 0.0)) + float(u)
        row = float(tile_meta.get("row_start", 0.0)) + float(v)
        ci = int(round(col))
        ri = int(round(row))
        if ri < 0 or ci < 0 or ri >= self._dsm.shape[0] or ci >= self._dsm.shape[1]:
            return None
        z = float(self._dsm[ri, ci])
        if not np.isfinite(z):
            return None
        return DSMPoint(
            east_m=(col - self._origin_col) * self._gsd_m,
            north_m=-(row - self._origin_row) * self._gsd_m,
            up_m=z,
        )


class LocalGridDSMMap(BaseDSMMap):
    """DSM bridge for scene-local metric datasets (AnyVisLoc 2026 NPZ release).

    The reference map and the DSM sit on independent grids with their own
    resolution and origin, both expressed in scene-local metres, so reference
    pixels are converted through local metres rather than resampled. ENU is
    absolute in the scene frame and ``enu_to_latlon`` closes over the dataset's
    synthetic geodetic origin, mirroring :class:`RasterDSMMap`; the
    ``reference_lat``/``reference_lon`` arguments are ignored on purpose,
    because the caller passes a tile anchor while ENU here is already absolute.
    """

    def __init__(
        self,
        dsm: np.ndarray,
        dsm_resolution,
        dsm_origin_local,
        map_resolution,
        map_origin_local,
        origin_lat: float,
        origin_lon: float,
        m_per_deg: float = 111_320.0,
    ) -> None:
        self._dsm = dsm
        self._dsm_res = np.asarray(dsm_resolution, dtype=np.float64).reshape(-1)[:2]
        self._dsm_origin = np.asarray(dsm_origin_local, dtype=np.float64).reshape(-1)[:2]
        self._map_res = np.asarray(map_resolution, dtype=np.float64).reshape(-1)[:2]
        self._map_origin = np.asarray(map_origin_local, dtype=np.float64).reshape(-1)[:2]
        self._origin_lat = float(origin_lat)
        self._origin_lon = float(origin_lon)
        self._m_per_deg = float(m_per_deg)
        self.available = (
            getattr(dsm, "ndim", 0) == 2
            and getattr(dsm, "size", 0) > 0
            and float(self._map_res[0]) > 0.0
            and float(self._dsm_res[0]) > 0.0
        )
        self.status = "available" if self.available else "invalid_local_grid_dsm"

    def pixel_to_enu(self, u: float, v: float, tile_meta: Optional[Dict[str, Any]] = None) -> Optional[DSMPoint]:
        if not self.available:
            return None
        tile_meta = tile_meta or {}
        col = float(tile_meta.get("col_start", 0.0)) + float(u)
        row = float(tile_meta.get("row_start", 0.0)) + float(v)

        # Reference-map pixel -> scene-local metres (x east, y south).
        x = float(self._map_origin[0]) + col * float(self._map_res[0])
        y = float(self._map_origin[1]) + row * float(self._map_res[1])

        dsm_col = int(round((x - float(self._dsm_origin[0])) / max(float(self._dsm_res[0]), 1e-12)))
        dsm_row = int(round((y - float(self._dsm_origin[1])) / max(float(self._dsm_res[1]), 1e-12)))
        if dsm_row < 0 or dsm_col < 0 or dsm_row >= self._dsm.shape[0] or dsm_col >= self._dsm.shape[1]:
            return None
        z = float(self._dsm[dsm_row, dsm_col])
        if not np.isfinite(z):
            return None
        return DSMPoint(east_m=x, north_m=-y, up_m=z)

    def enu_to_latlon(self, east_m: float, north_m: float, reference_lat: float, reference_lon: float) -> Tuple[float, float]:
        return (
            self._origin_lat + float(north_m) / self._m_per_deg,
            self._origin_lon + float(east_m) / self._m_per_deg,
        )


class RasterDSMMap(BaseDSMMap):
    """GeoTIFF DOP+DSM bridge.

    Reference pixels are interpreted in the orthophoto raster grid; DSM is
    sampled at the corresponding map coordinate. This is intentionally
    fail-closed: if CRS/transform/rasterio are unavailable, ``available`` is
    false and DSM-PnP must reject rather than silently becoming planar PnP.
    """

    def __init__(self, orthophoto_path: Path, dsm_path: Path, origin_lat: float, origin_lon: float) -> None:
        self.orthophoto_path = Path(orthophoto_path)
        self.dsm_path = Path(dsm_path)
        self.origin_lat = float(origin_lat)
        self.origin_lon = float(origin_lon)
        self.available = False
        self.status = "uninitialized"
        self._ref = None
        self._dsm = None
        self._origin_xy: Optional[Tuple[float, float]] = None
        self._to_ref_crs = None
        self._to_wgs84 = None
        self._open()

    @classmethod
    def from_dataset(cls, dataset: Any) -> Optional["RasterDSMMap"]:
        data_root = Path(getattr(dataset, "_data_root", ""))
        region = str(getattr(dataset, "_region", ""))
        if not data_root or not region:
            return None
        ref_dir = data_root / "Reference_map" / region
        orthophoto = ref_dir / "result_roi.tif"
        if not orthophoto.exists():
            orthophoto = ref_dir / "satellite_roi.tif"
        dsm = ref_dir / "dsm_roi.tif"
        if not dsm.exists():
            alt = ref_dir / "QZ_town_DSM30_reproject_resize_roi.tif"
            dsm = alt if alt.exists() else dsm
        if not orthophoto.exists() or not dsm.exists():
            return None

        origin_lat = origin_lon = None
        try:
            if hasattr(dataset, "_entries") and dataset._entries:
                gt = dataset._entries[0].get("gt_pose", {})
                origin_lat = float(gt.get("lat"))
                origin_lon = float(gt.get("lon"))
        except Exception:
            origin_lat = origin_lon = None
        if origin_lat is None or origin_lon is None:
            origin_lat, origin_lon = 0.0, 0.0
        return cls(orthophoto, dsm, origin_lat=origin_lat, origin_lon=origin_lon)

    def _open(self) -> None:
        try:
            import rasterio
            from pyproj import Transformer
        except Exception as exc:
            self.status = f"dependency_unavailable:{type(exc).__name__}"
            return
        try:
            self._ref = rasterio.open(str(self.orthophoto_path))
            self._dsm = rasterio.open(str(self.dsm_path))
            if self._ref.crs is None or self._dsm.crs is None:
                self.status = "missing_crs"
                return
            self._to_ref_crs = Transformer.from_crs("EPSG:4326", str(self._ref.crs), always_xy=True)
            self._to_wgs84 = Transformer.from_crs(str(self._ref.crs), "EPSG:4326", always_xy=True)
            self._origin_xy = self._to_ref_crs.transform(self.origin_lon, self.origin_lat)
            self.available = True
            self.status = "available"
        except Exception as exc:
            self.status = f"open_failed:{type(exc).__name__}"

    def pixel_to_enu(self, u: float, v: float, tile_meta: Optional[Dict[str, Any]] = None) -> Optional[DSMPoint]:
        if not self.available or self._ref is None or self._dsm is None or self._origin_xy is None:
            return None
        tile_meta = tile_meta or {}
        col = float(tile_meta.get("col_start", 0.0)) + float(u)
        row = float(tile_meta.get("row_start", 0.0)) + float(v)
        if row < 0 or col < 0 or row >= self._ref.height or col >= self._ref.width:
            return None
        try:
            import rasterio.transform
            x, y = rasterio.transform.xy(self._ref.transform, row, col, offset="center")
            if str(self._ref.crs) != str(self._dsm.crs):
                from pyproj import Transformer
                tr = Transformer.from_crs(str(self._ref.crs), str(self._dsm.crs), always_xy=True)
                dx, dy = tr.transform(x, y)
            else:
                dx, dy = x, y
            dsm_row, dsm_col = self._dsm.index(dx, dy)
            if dsm_row < 0 or dsm_col < 0 or dsm_row >= self._dsm.height or dsm_col >= self._dsm.width:
                return None
            z = float(self._dsm.read(1, window=((dsm_row, dsm_row + 1), (dsm_col, dsm_col + 1)))[0, 0])
            if not np.isfinite(z):
                return None
            return DSMPoint(east_m=float(x - self._origin_xy[0]), north_m=float(y - self._origin_xy[1]), up_m=z)
        except Exception:
            return None

    def enu_to_latlon(self, east_m: float, north_m: float, reference_lat: float, reference_lon: float) -> Tuple[float, float]:
        if self.available and self._to_wgs84 is not None and self._origin_xy is not None:
            lon, lat = self._to_wgs84.transform(
                self._origin_xy[0] + float(east_m),
                self._origin_xy[1] + float(north_m),
            )
            return float(lat), float(lon)
        return super().enu_to_latlon(east_m, north_m, reference_lat, reference_lon)
