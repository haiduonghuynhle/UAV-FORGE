"""
Coordinate transforms for the native AnyVisLoc baseline adapter.

Combines:
• LocalTangentPlane (NED) from src/utils/coord_transforms.py
• UTM helpers (deg2utm / utm2deg) adapted from Benchmark/utils.py
• latlon_to_meters / meters_to_latlon linear helpers
"""

from __future__ import annotations
import numpy as np
from typing import Tuple
from dataclasses import dataclass
from pyproj import Transformer

# ── WGS84 constants ──────────────────────────────────────────────────────────
WGS84_A  = 6378137.0
WGS84_B  = 6356752.314245
WGS84_E2 = 1 - (WGS84_B ** 2) / (WGS84_A ** 2)


# ============================================================================
#  Basic geodetic ↔ ECEF
# ============================================================================

def geodetic_to_ecef(lat: float, lon: float, alt: float) -> np.ndarray:
    lat_r = np.radians(lat);  lon_r = np.radians(lon)
    N = WGS84_A / np.sqrt(1 - WGS84_E2 * np.sin(lat_r) ** 2)
    return np.array([
        (N + alt) * np.cos(lat_r) * np.cos(lon_r),
        (N + alt) * np.cos(lat_r) * np.sin(lon_r),
        (N * (1 - WGS84_E2) + alt) * np.sin(lat_r),
    ])


def ecef_to_geodetic(ecef: np.ndarray) -> Tuple[float, float, float]:
    X, Y, Z = ecef
    lon = np.arctan2(Y, X)
    p   = np.sqrt(X**2 + Y**2)
    lat = np.arctan2(Z, p * (1 - WGS84_E2))
    for _ in range(10):
        N   = WGS84_A / np.sqrt(1 - WGS84_E2 * np.sin(lat)**2)
        lat2 = np.arctan2(Z + WGS84_E2 * N * np.sin(lat), p)
        if abs(lat2 - lat) < 1e-12:
            break
        lat = lat2
    N   = WGS84_A / np.sqrt(1 - WGS84_E2 * np.sin(lat)**2)
    alt = (p / np.cos(lat) - N) if abs(lat) < np.pi/4 else (Z / np.sin(lat) - N*(1-WGS84_E2))
    return float(np.degrees(lat)), float(np.degrees(lon)), float(alt)


# ============================================================================
#  Local Tangent Plane (NED)
# ============================================================================

@dataclass
class LocalTangentPlane:
    """Local NED tangent plane centred at (origin_lat, origin_lon, origin_alt)."""
    origin_lat: float   # degrees
    origin_lon: float   # degrees
    origin_alt: float = 0.0
    frame: str = "ned"

    def __post_init__(self):
        self._origin_ecef = geodetic_to_ecef(self.origin_lat, self.origin_lon, self.origin_alt)
        lat_r = np.radians(self.origin_lat)
        lon_r = np.radians(self.origin_lon)
        # NED rotation matrix
        self._R = np.array([
            [-np.sin(lat_r)*np.cos(lon_r), -np.sin(lat_r)*np.sin(lon_r),  np.cos(lat_r)],
            [-np.sin(lon_r),                np.cos(lon_r),                 0             ],
            [-np.cos(lat_r)*np.cos(lon_r), -np.cos(lat_r)*np.sin(lon_r), -np.sin(lat_r)],
        ])

    def geodetic_to_local(self, lat: float, lon: float, alt: float) -> np.ndarray:
        ecef  = geodetic_to_ecef(lat, lon, alt)
        delta = ecef - self._origin_ecef
        return self._R @ delta

    def local_to_geodetic(self, local_pos: np.ndarray) -> Tuple[float, float, float]:
        delta_ecef = self._R.T @ local_pos
        return ecef_to_geodetic(self._origin_ecef + delta_ecef)


# ============================================================================
#  UTM helpers (matches Benchmark/utils.py API exactly)
# ============================================================================

def deg2utm(region: str, config: dict, lon: float, lat: float) -> Tuple[float, float]:
    """WGS-84 lat/lon → UTM (x=easting, y=northing) using region UTM zone from config."""
    utm_system = config[f"{region}_UTM_SYSTEM"]   # e.g. "50N"
    zone_num   = int(utm_system[:2])
    epsg_src   = "epsg:4326"
    epsg_dst   = f"epsg:326{utm_system[:2]}"      # e.g. epsg:32650
    trf = Transformer.from_crs(epsg_src, epsg_dst, always_xy=True)
    x, y = trf.transform(lon, lat)
    return x, y


def utm2deg(region: str, config: dict,
            x: float, y: float) -> Tuple[float, float]:
    """UTM (x=easting, y=northing) → WGS-84 lon/lat."""
    utm_system = config[f"{region}_UTM_SYSTEM"]
    epsg_src   = f"epsg:326{utm_system[:2]}"
    epsg_dst   = "epsg:4326"
    trf = Transformer.from_crs(epsg_src, epsg_dst, always_xy=True)
    lon, lat = trf.transform(x, y)
    return lon, lat


# ============================================================================
#  Simple linear helpers (for small areas)
# ============================================================================

def meters_per_degree(latitude: float) -> Tuple[float, float]:
    lat_r = np.radians(latitude)
    R_m = WGS84_A * (1 - WGS84_E2) / (1 - WGS84_E2 * np.sin(lat_r)**2)**1.5
    R_n = WGS84_A / np.sqrt(1 - WGS84_E2 * np.sin(lat_r)**2)
    return R_m * np.pi / 180, R_n * np.cos(lat_r) * np.pi / 180


def latlon_to_meters(lat: float, lon: float,
                     origin_lat: float, origin_lon: float) -> Tuple[float, float]:
    """Return (east_m, north_m) offset from origin."""
    m_lat, m_lon = meters_per_degree(origin_lat)
    return (lon - origin_lon) * m_lon, (lat - origin_lat) * m_lat


def meters_to_latlon(east_m: float, north_m: float,
                     origin_lat: float, origin_lon: float) -> Tuple[float, float]:
    m_lat, m_lon = meters_per_degree(origin_lat)
    return origin_lat + north_m / m_lat, origin_lon + east_m / m_lon
