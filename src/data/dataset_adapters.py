"""
Unified Dataset Adapters for Multi-Dataset Benchmarking.

Normalizes diverse UAV localization datasets (VPAir, UAVD4L, AVL,
UAV-VisLoc, etc.) to a common interface for use by the benchmark
runner and pipeline.

Each adapter outputs frames in a standardized dict format:
    {
        'image_name': str,
        'query_image': np.ndarray (H, W, 3),
        'query_path': Path,
        'gt_pose': {
            'lat': float, 'lon': float, 'alt': float,
            'yaw': float, 'pitch': float, 'roll': float,
        },
        'reference_image': Optional[np.ndarray],
        'reference_path': Optional[Path],
        'metadata': dict,  # dataset-specific extras
        'camera': dict,    # intrinsics
        'gsd': float,      # ground sampling distance (m/px)
    }
"""

import numpy as np
from pathlib import Path
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Iterator, Tuple
import logging
import json
import csv

logger = logging.getLogger(__name__)


def _normalize_uav_visloc_root(root: Path, max_nested: int = 8) -> Path:
    """
    Descend through repeated ``UAV_VisLoc_dataset/`` directory wrappers until
    numeric location folders (``01``, ``02``, ...) are visible.

    Handles common unzip layouts such as::
        data/UAV_VisLoc_dataset/UAV_VisLoc_dataset/01/...
    """
    root = Path(root)
    for _ in range(max_nested + 1):
        try:
            digit_dirs = [
                d.name for d in root.iterdir()
                if d.is_dir() and d.name.isdigit()
            ]
        except OSError:
            return root
        if digit_dirs:
            return root
        nested = root / "UAV_VisLoc_dataset"
        if nested.is_dir() and nested.resolve() != root.resolve():
            root = nested
            continue
        break
    return root


class BaseDataset(ABC):
    """Abstract base class for UAV localization datasets."""
    
    def __init__(self,
                 data_root: Path,
                 test_interval: int = 1,
                 max_samples: int = None,
                 max_samples_percent: int = None):
        """
        Args:
            data_root: Root directory of the dataset
            test_interval: Sample every N-th frame (for quick testing)
            max_samples: Maximum number of samples (None = all)
            max_samples_percent: Percentage of dataset to use (None = all)
        """
        self._data_root = Path(data_root)
        self._test_interval = max(1, test_interval)
        self._max_samples = max_samples
        self._max_samples_percent = max_samples_percent
        self._entries: List[Dict[str, Any]] = []
        self._loaded = False
    
    @abstractmethod
    def load(self):
        """Load dataset metadata."""
        pass
    
    @abstractmethod
    def _get_frame(self, idx: int) -> Dict[str, Any]:
        """Get a single frame by index."""
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Dataset name."""
        pass
    
    @property
    def num_samples(self) -> int:
        """Number of samples after interval/max filtering."""
        if not self._loaded:
            self.load()
        if getattr(self, "_query_name_allowlist", None) is not None:
            return len(self.selected_entry_indices())
        n = len(self._entries[::self._test_interval])
        if self._max_samples_percent:
            n = int(n * (self._max_samples_percent / 100.0))
        if self._max_samples:
            n = min(n, self._max_samples)
        return n
    
    def __len__(self) -> int:
        return self.num_samples

    def selected_entry_indices(self) -> List[int]:
        """Indices selected by test interval and max-sample controls."""
        if not self._loaded:
            self.load()

        indices = list(range(0, len(self._entries), self._test_interval))
        # Optional allowlist of query names, applied before the sample caps.
        # Unset unless the config names a list file, so default selection is
        # unchanged. Used to run a pre-registered, geometrically buffered
        # held-out subset rather than a prefix.
        allow = getattr(self, "_query_name_allowlist", None)
        if allow is not None:
            def _name(entry):
                if isinstance(entry, dict):
                    return str(entry.get("image_name", ""))
                return str(entry)
            indices = [i for i in indices if _name(self._entries[i]) in allow]
        if self._max_samples_percent:
            cutoff = int(len(indices) * (self._max_samples_percent / 100.0))
            indices = indices[:cutoff]
        if self._max_samples:
            indices = indices[:self._max_samples]
        return indices
    
    def __iter__(self) -> Iterator[Dict[str, Any]]:
        if not self._loaded:
            self.load()
        
        for idx in self.selected_entry_indices():
            yield self._get_frame(idx)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if not self._loaded:
            self.load()
        actual_idx = idx * self._test_interval
        return self._get_frame(actual_idx)
    
    def get_ground_truth(self) -> List[Dict[str, Any]]:
        """Get all ground truth as list of dicts."""
        if not self._loaded:
            self.load()
        return [e.get('gt_pose', {}) for e in self._entries[::self._test_interval]]

    def iter_reference_corpus(self) -> Iterator[Dict[str, Any]]:
        """
        Yield retrieval corpus entries as:
            {"ref_name": str, "ref_image": np.ndarray, "anchor": {"lat": float, "lon": float}}
        Default behavior uses available per-frame reference_image.
        """
        for frame in self:
            ref_img = frame.get("reference_image")
            if ref_img is None:
                continue
            gt = frame.get("gt_pose", {})
            yield {
                "ref_name": str(frame.get("image_name", "")),
                "ref_image": ref_img,
                "anchor": {
                    "lat": float(gt.get("lat", 0.0)),
                    "lon": float(gt.get("lon", 0.0)),
                },
            }


class VPAirDataset(BaseDataset):
    """
    VPAir Dataset Adapter.
    
    Structure:
        VPAir/
        ├── queries/          # UAV query images (.jpg)
        ├── reference_views/  # Reference renders (.jpg)
        ├── distractors/      # Optional distractor images
        ├── poses_query.txt   # Native query poses
        ├── poses_reference_view.txt
        ├── poses.csv         # Optional legacy combined pose file
        └── camera_calibration.yaml
    
    Pose CSV format:
        img_path, qw, qx, qy, qz, tx, ty, tz
    """
    
    @property
    def name(self) -> str:
        return "VPAir"

    _COORDINATE_FRAME = "ecef"
    _COORDINATE_AXES = ("x", "y", "z")
    _COORDINATE_UNIT = "m"

    @staticmethod
    def _read_pose_table(path: Path) -> Dict[str, Dict[str, float]]:
        """Read native VPAir or legacy combined CSV pose metadata."""
        poses: Dict[str, Dict[str, float]] = {}
        with open(path, "r", encoding="utf-8-sig", newline="") as pose_file:
            reader = csv.DictReader(pose_file)
            for raw_row in reader:
                row = {
                    str(key).strip().lower(): value
                    for key, value in raw_row.items()
                    if key is not None
                }

                image_path = next(
                    (
                        row.get(key)
                        for key in ("filepath", "img_path", "filename", "image_name", "path")
                        if row.get(key)
                    ),
                    None,
                )
                if image_path is None and reader.fieldnames:
                    image_path = row.get(str(reader.fieldnames[0]).strip().lower())
                if not image_path:
                    continue

                def required_float(*keys: str) -> float:
                    for key in keys:
                        value = row.get(key)
                        if value not in (None, ""):
                            return float(value)
                    raise ValueError(f"Missing {keys[0]} in {path.name}")

                try:
                    lat = required_float("lat", "latitude", "x")
                    lon = required_float("lon", "longitude", "y")
                    alt = required_float("alt", "altitude", "z")
                    roll = float(row.get("roll", 0.0) or 0.0)
                    pitch = float(row.get("pitch", 0.0) or 0.0)
                    yaw = float(row.get("yaw", 0.0) or 0.0)
                except (TypeError, ValueError):
                    continue

                if max(abs(roll), abs(pitch), abs(yaw)) <= np.pi + 0.05:
                    roll, pitch, yaw = [
                        float(value) for value in np.degrees([roll, pitch, yaw])
                    ]

                poses[Path(str(image_path).replace("\\", "/")).name] = {
                    "lat": lat,
                    "lon": lon,
                    "alt": alt,
                    "roll": roll,
                    "pitch": pitch,
                    "yaw": yaw,
                }
        return poses
    
    def load(self):
        """Load VPAir dataset metadata."""
        poses_csv = self._data_root / "poses.csv"
        poses_query = self._data_root / "poses_query.txt"
        poses_path = poses_csv if poses_csv.exists() else poses_query
        if not poses_path.exists():
            raise FileNotFoundError(
                "VPAir pose metadata not found; expected one of: "
                f"{poses_csv}, {poses_query}"
            )
        
        # Load camera calibration
        cal_path = self._data_root / "camera_calibration.yaml"
        self._camera = self._load_camera_calibration(cal_path)
        
        # Load poses
        query_dir = self._data_root / "queries"
        ref_dir = self._data_root / "reference_views"
        
        poses = self._read_pose_table(poses_path)
        self._entries.clear()
        
        # Build entries for query images
        query_images = sorted(query_dir.glob("*.jpg")) + sorted(query_dir.glob("*.png"))
        
        for qpath in query_images:
            # Check just the name (e.g. 00001.png) since poses.csv uses filename explicitly
            file_name = qpath.name
            if file_name not in poses:
                continue
            
            pose = poses[file_name]
            
            # Find corresponding reference
            ref_name = qpath.name  # VPAir: same name in reference_views/
            ref_path = ref_dir / ref_name
            
            self._entries.append({
                'image_name': qpath.name,
                'query_path': qpath,
                'reference_path': ref_path if ref_path.exists() else None,
                'gt_pose': {
                    'lat': pose['lat'],
                    'lon': pose['lon'],
                    'alt': pose['alt'],
                    'yaw': pose['yaw'],
                    'pitch': pose['pitch'],
                    'roll': pose['roll'],
                    # VPAIR attitude is aircraft/body convention: 0 = nadir camera
                    'pitch_convention': 'aircraft_nadir_zero',
                },
                'metadata': {
                    'dataset': 'vpair',
                    'has_depth': True,  # VPAir provides depth maps
                    'coordinate_frame': self._COORDINATE_FRAME,
                    'coordinate_axes': self._COORDINATE_AXES,
                    'coordinate_unit': self._COORDINATE_UNIT,
                },
            })
        
        self._loaded = True
        logger.info(f"VPAir: loaded {len(self._entries)} query images")
    
    def _get_frame(self, idx: int) -> Dict[str, Any]:
        """Get frame by index."""
        entry = self._entries[idx]
        
        frame = {
            'image_name': entry['image_name'],
            'query_path': entry['query_path'],
            'gt_pose': entry['gt_pose'],
            'metadata': entry['metadata'],
            'camera': self._camera,
            # Measured from adjacent reference views; the former 0.3 m/px
            # value under-scaled visual offsets by about 1.4x.
            'gsd': 0.42,
            'coordinate_frame': self._COORDINATE_FRAME,
            'coordinate_axes': self._COORDINATE_AXES,
            'coordinate_unit': self._COORDINATE_UNIT,
        }
        
        # Lazy load images
        try:
            import cv2
            frame['query_image'] = cv2.cvtColor(
                cv2.imread(str(entry['query_path'])), cv2.COLOR_BGR2RGB
            )
            if entry['reference_path'] and entry['reference_path'].exists():
                frame['reference_image'] = cv2.cvtColor(
                    cv2.imread(str(entry['reference_path'])), cv2.COLOR_BGR2RGB
                )
            else:
                frame['reference_image'] = None
        except Exception as e:
            logger.warning(f"Failed to load images for {entry['image_name']}: {e}")
            frame['query_image'] = np.zeros((480, 640, 3), dtype=np.uint8)
            frame['reference_image'] = None
        
        return frame
    
    def _load_camera_calibration(self, cal_path: Path) -> Dict:
        """Load VPAir camera calibration.

        VPAIR ships a Kalibr-style file: cam0.intrinsics = [fx, fy, cx, cy] and
        cam0.resolution = [width, height]. The previous flat cal.get('fx')
        lookup never matched that shape, so every run silently used the
        614.2 px / 640x480 fallback against the true 750.63 px / 800x600:
        the principal point sat ~100 px off the image centre and the IPM warp
        K R K^-1 rotated about the wrong point (0-47 m heading-dependent
        position bias on IPM-enabled frames).
        """
        default_camera = {
            'fx': 614.2, 'fy': 614.2,
            'cx': 320.0, 'cy': 240.0,
            'width': 640, 'height': 480,
        }

        if not cal_path.exists():
            logger.warning("VPAir calibration missing at %s; using fallback intrinsics %s",
                           cal_path, default_camera)
            return default_camera
        try:
            import yaml
            with open(cal_path, 'r') as f:
                cal = yaml.safe_load(f) or {}
            cam = cal.get('cam0', cal)
            intr = cam.get('intrinsics') if isinstance(cam, dict) else None
            if isinstance(intr, (list, tuple)) and len(intr) >= 4:
                fx, fy, cx, cy = (float(v) for v in intr[:4])
                camera = {'fx': fx, 'fy': fy, 'cx': cx, 'cy': cy}
                res = cam.get('resolution')
                if isinstance(res, (list, tuple)) and len(res) >= 2:
                    camera['width'], camera['height'] = int(res[0]), int(res[1])
                else:
                    camera['width'], camera['height'] = 2 * int(cx), 2 * int(cy)
                return camera
            # Flat-key layout (legacy tests / other exports)
            if isinstance(cam, dict) and 'fx' in cam:
                default_camera.update({k: float(cam[k]) for k in ('fx', 'fy', 'cx', 'cy') if k in cam})
                return default_camera
            logger.warning("VPAir calibration at %s has unrecognised structure "
                           "(top-level keys: %s); using fallback intrinsics",
                           cal_path, sorted(cal) if isinstance(cal, dict) else type(cal).__name__)
        except Exception as exc:
            logger.warning("VPAir calibration parse failed (%s: %s); using fallback intrinsics",
                           type(exc).__name__, exc)
        return default_camera

    def iter_reference_corpus(self) -> Iterator[Dict[str, Any]]:
        """
        Build full retrieval corpus from reference_views (independent of query sampling).
        """
        import cv2
        poses_csv = self._data_root / "poses.csv"
        poses_reference = self._data_root / "poses_reference_view.txt"
        poses_query = self._data_root / "poses_query.txt"
        pose_path = next(
            (path for path in (poses_reference, poses_csv, poses_query) if path.exists()),
            None,
        )
        pose_by_name = self._read_pose_table(pose_path) if pose_path is not None else {}
        ref_dir = self._data_root / "reference_views"
        ref_paths = sorted(list(ref_dir.glob("*.jpg")) + list(ref_dir.glob("*.png")))
        for p in ref_paths:
            img_bgr = cv2.imread(str(p))
            if img_bgr is None:
                continue
            pose = pose_by_name.get(p.name, {})
            yield {
                "ref_name": p.name,
                "ref_image": cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB),
                "anchor": {
                    "lat": float(pose.get("lat", 0.0)),
                    "lon": float(pose.get("lon", 0.0)),
                },
            }


class AVLDataset(BaseDataset):
    """
    AVL (Aerial Visual Localization) Dataset Adapter.
    
    Structure (after unzip):
        AVLData/
        ├── UAV_image/
        │   └── QZ_Town/
        │       ├── QZ_SongCity/      # Sub-region with DJI images
        │       ├── QingZhou_2024/
        │       └── Qingzhou_3_2/
        ├── Reference_map/
        │   └── QZ_Town/
        │       ├── result_roi.tif    # Reference orthomosaic
        │       ├── satellite_roi.tif
        │       └── dsm_roi.tif
        └── metadata/
            └── QZ_Town.json  # GPS + attitude metadata (1265 entries)
    
    Metadata JSON format (per entry):
        {
            "name": "./Data/UAV_image/QZ_Town/QZ_SongCity/DJI_0532.JPG",
            "lat": 36.688..., "lon": 118.475...,
            "rel_alt": 119.8, "pitch": -90.0, "yaw": -13.5, "roll": 0.0,
            "cam_size": 7.81, "focal_len": 3.6,
            "width": 4000.0, "height": 3000.0, "fov": 84.0
        }
    
    Note: The 'name' field uses relative paths with a './Data/' prefix
    that maps to the dataset root. We strip this prefix to resolve
    actual image paths.
    """
    
    def __init__(self,
                 data_root: Path,
                 region: str = "QZ_Town",
                 tile_size_px: int = 2048,
                 **kwargs):
        super().__init__(data_root, **kwargs)
        self._region = region
        self._tile_size_px = tile_size_px
        self._ref_raster = None      # Lazy-loaded rasterio dataset
        self._ref_transform = None
        self._ref_crs = None
    
    @property
    def name(self) -> str:
        return f"AVL-{self._region}"
    
    @staticmethod
    def _compute_focal_px(focal_len_mm: float, cam_size_mm: float,
                          image_width_px: float) -> float:
        """
        Convert physical focal length to pixel focal length.
        
        focal_px = focal_mm * image_width_px / sensor_width_mm
        """
        if cam_size_mm <= 0 or focal_len_mm <= 0:
            return 1000.0  # safe default
        return focal_len_mm * image_width_px / cam_size_mm
    
    def _resolve_image_path(self, raw_name: str) -> Path:
        """
        Resolve the image path from the metadata 'name' field.
        
        The metadata uses paths like './Data/UAV_image/QZ_Town/QZ_SongCity/DJI_0532.JPG'.
        We strip the './Data/' prefix and resolve relative to data_root.
        """
        # Strip leading './' or './Data/'
        cleaned = raw_name
        if cleaned.startswith('./Data/'):
            cleaned = cleaned[len('./Data/'):]
        elif cleaned.startswith('./'):
            cleaned = cleaned[2:]
        
        img_path = self._data_root / cleaned
        
        if img_path.exists():
            return img_path
        
        # Fallback: try just the filename, but ONLY inside the sub-region
        # directory named in the metadata path. Searching all sub-regions
        # (pre-2026-08 behavior) paired 102 Qingzhou_3_2 metadata entries with
        # same-named QZ_SongCity images: GT and image from different flights
        # ~800 m apart, i.e. unwinnable frames poisoning every AVL run.
        filename = Path(raw_name).name
        parts = Path(cleaned).parts
        own_subdir = None
        for i, part in enumerate(parts):
            if part == self._region and i + 1 < len(parts) - 1:
                own_subdir = parts[i + 1]
                break
        if own_subdir is not None:
            candidate = self._data_root / "UAV_image" / self._region / own_subdir / filename
            if candidate.exists():
                return candidate

        # Last resort: return the cleaned path (may not exist)
        return img_path
    
    def load(self):
        """Load AVL dataset metadata."""
        metadata_path = self._data_root / "metadata" / f"{self._region}.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata not found: {metadata_path}")
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        found_count = 0
        missing_count = 0
        
        for entry in metadata:
            raw_name = entry.get('name', entry.get('filename', ''))
            img_path = self._resolve_image_path(raw_name)
            
            if img_path.exists():
                found_count += 1
            else:
                missing_count += 1
                continue  # Skip entries with missing images
            
            # Compute camera intrinsics from metadata
            img_w = entry.get('width', 4000.0)
            img_h = entry.get('height', 3000.0)
            focal_px = self._compute_focal_px(
                entry.get('focal_len', 3.6),
                entry.get('cam_size', 7.81),
                img_w,
            )
            
            self._entries.append({
                'image_name': f"{Path(raw_name).parent.name}/{Path(raw_name).name}",
                'query_path': img_path,
                'gt_pose': {
                    'lat': entry.get('lat', 0),
                    'lon': entry.get('lon', 0),
                    'alt': entry.get('rel_alt', entry.get('alt', 0)),
                    'yaw': entry.get('yaw', 0),
                    'pitch': entry.get('pitch', 0),
                    'roll': entry.get('roll', 0),
                    # AVL attitude is gimbal convention: -90 = nadir
                    'pitch_convention': 'gimbal_neg90_nadir',
                },
                'camera': {
                    'fx': focal_px,
                    'fy': focal_px,
                    'cx': img_w / 2.0,
                    'cy': img_h / 2.0,
                    'width': int(img_w),
                    'height': int(img_h),
                },
                'metadata': {
                    'dataset': 'avl',
                    'region': self._region,
                    'abs_alt': entry.get('abs_alt', 0),
                    'fov': entry.get('fov', 84.0),
                    'cam_size_mm': entry.get('cam_size', 7.81),
                    'focal_len_mm': entry.get('focal_len', 3.6),
                },
            })
        
        self._loaded = True
        logger.info(
            f"AVL-{self._region}: loaded {len(self._entries)} entries "
            f"(found={found_count}, missing={missing_count})"
        )
    
    def _load_reference_map(self):
        """Lazy-load the reference orthomosaic GeoTIFF."""
        if self._ref_raster is not None:
            return
        
        ref_path = self._data_root / "Reference_map" / self._region / "result_roi.tif"
        if not ref_path.exists():
            # Try satellite_roi.tif as fallback
            ref_path = self._data_root / "Reference_map" / self._region / "satellite_roi.tif"
        
        if not ref_path.exists():
            logger.warning(f"No reference map found for {self._region}")
            return
        
        try:
            import rasterio
            self._ref_raster = rasterio.open(str(ref_path))
            self._ref_transform = self._ref_raster.transform
            self._ref_crs = self._ref_raster.crs
            logger.info(
                f"Reference map loaded: {ref_path.name} "
                f"({self._ref_raster.width}x{self._ref_raster.height}, "
                f"CRS={self._ref_crs}, GSD={self._ref_raster.res[0]:.4f}m)"
            )
        except ImportError:
            logger.warning("rasterio not installed — cannot load reference map. pip install rasterio")
        except Exception as e:
            logger.warning(f"Failed to load reference map: {e}")
    
    def _latlon_to_pixel(self, lat: float, lon: float) -> Optional[Tuple[int, int]]:
        """
        Convert WGS84 (lat, lon) to pixel coordinates in the reference raster.
        
        Uses pyproj to transform from EPSG:4326 → raster CRS (e.g. EPSG:32650 UTM).
        Returns (col, row) or None if out of bounds.
        """
        if self._ref_raster is None:
            return None
        
        try:
            from pyproj import Transformer
            
            transformer = Transformer.from_crs(
                "EPSG:4326", str(self._ref_crs), always_xy=True
            )
            # pyproj always_xy: input is (lon, lat), output is (easting, northing)
            easting, northing = transformer.transform(lon, lat)
            
            # Convert to pixel coordinates using rasterio's inverse transform
            from rasterio.transform import rowcol
            row, col = rowcol(self._ref_transform, easting, northing)
            
            return int(col), int(row)
        except ImportError:
            logger.warning("pyproj not installed — cannot convert coordinates. pip install pyproj")
            return None
        except Exception as e:
            logger.debug(f"Coordinate transform failed for ({lat}, {lon}): {e}")
            return None

    def _pixel_to_latlon(self, col: int, row: int) -> Optional[Tuple[float, float]]:
        """
        Convert raster pixel coordinates to WGS84 (lat, lon).

        Used to anchor retrieval corpus tiles independent from query GT.
        """
        if self._ref_raster is None or self._ref_transform is None:
            return None

        try:
            from rasterio.transform import xy
            from pyproj import Transformer

            # Get map coordinates at the pixel center.
            x, y = xy(self._ref_transform, row, col, offset="center")
            transformer = Transformer.from_crs(str(self._ref_crs), "EPSG:4326", always_xy=True)
            lon, lat = transformer.transform(x, y)
            return float(lat), float(lon)
        except Exception as e:
            logger.debug("Pixel->latlon transform failed at (col=%s,row=%s): %s", col, row, e)
            return None
    
    def _crop_reference_tile(self, lat: float, lon: float) -> Optional[np.ndarray]:
        """
        Crop a reference tile centered on the given GPS coordinates.
        
        Args:
            lat: WGS84 latitude
            lon: WGS84 longitude
            
        Returns:
            (tile_size, tile_size, 3) uint8 RGB image or None
        """
        self._load_reference_map()
        
        if self._ref_raster is None:
            return None
        
        pixel = self._latlon_to_pixel(lat, lon)
        if pixel is None:
            return None
        
        col, row = pixel
        half = self._tile_size_px // 2
        
        # Compute window bounds (clipped to raster extent)
        col_start = max(0, col - half)
        row_start = max(0, row - half)
        col_end = min(self._ref_raster.width, col + half)
        row_end = min(self._ref_raster.height, row + half)
        
        if col_end <= col_start or row_end <= row_start:
            logger.debug(f"GPS ({lat:.4f}, {lon:.4f}) maps outside reference raster")
            return None
        
        try:
            from rasterio.windows import Window
            window = Window(col_start, row_start, col_end - col_start, row_end - row_start)
            
            # Read RGB bands (assume bands 1,2,3 = R,G,B)
            bands_to_read = min(self._ref_raster.count, 3)
            tile = self._ref_raster.read(
                list(range(1, bands_to_read + 1)),
                window=window
            )
            
            # tile shape: (C, H, W) → transpose to (H, W, C)
            tile = np.transpose(tile, (1, 2, 0))
            
            # Pad if the window was clipped at the raster edges
            if tile.shape[0] != self._tile_size_px or tile.shape[1] != self._tile_size_px:
                padded = np.zeros((self._tile_size_px, self._tile_size_px, 3), dtype=np.uint8)
                # Center the cropped region
                y_off = (self._tile_size_px - tile.shape[0]) // 2
                x_off = (self._tile_size_px - tile.shape[1]) // 2
                padded[y_off:y_off + tile.shape[0], x_off:x_off + tile.shape[1]] = tile
                tile = padded
            
            return tile.astype(np.uint8)
        except Exception as e:
            logger.debug(f"Failed to crop tile at ({lat:.4f}, {lon:.4f}): {e}")
            return None
    
    def _get_frame(self, idx: int) -> Dict[str, Any]:
        entry = self._entries[idx]
        camera = entry.get('camera', {
            'fx': 1000, 'fy': 1000, 'cx': 640, 'cy': 360,
            'width': 1280, 'height': 720,
        })
        gt = entry['gt_pose']
        
        frame = {
            'image_name': entry['image_name'],
            'query_path': entry['query_path'],
            'gt_pose': gt,
            'metadata': entry['metadata'],
            'camera': camera,
            'gsd': self._ref_raster.res[0] if self._ref_raster else 0.06,
        }
        
        try:
            import cv2
            if entry['query_path'].exists():
                frame['query_image'] = cv2.cvtColor(
                    cv2.imread(str(entry['query_path'])), cv2.COLOR_BGR2RGB
                )
            else:
                h, w = camera.get('height', 720), camera.get('width', 1280)
                frame['query_image'] = np.zeros((h, w, 3), dtype=np.uint8)
        except Exception:
            h, w = camera.get('height', 720), camera.get('width', 1280)
            frame['query_image'] = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Crop reference tile from orthomosaic
        frame['reference_image'] = self._crop_reference_tile(
            gt.get('lat', 0), gt.get('lon', 0)
        )
        
        return frame

    def iter_reference_corpus(self) -> Iterator[Dict[str, Any]]:
        """
        Build retrieval corpus from orthomosaic tiles, independent of query GT.
        """
        self._load_reference_map()
        if self._ref_raster is None:
            logger.warning("AVL-%s: reference map unavailable, retrieval corpus empty.", self._region)
            return

        try:
            from rasterio.windows import Window
        except Exception as e:
            logger.warning("AVL-%s: rasterio window import failed: %s", self._region, e)
            return

        tile_size = max(128, int(self._tile_size_px))
        step = max(128, tile_size // 2)  # 50% overlap improves retrieval coverage.

        max_row = max(0, self._ref_raster.height - 1)
        max_col = max(0, self._ref_raster.width - 1)
        emitted = 0

        for row_start in range(0, self._ref_raster.height, step):
            for col_start in range(0, self._ref_raster.width, step):
                row_end = min(self._ref_raster.height, row_start + tile_size)
                col_end = min(self._ref_raster.width, col_start + tile_size)
                h = row_end - row_start
                w = col_end - col_start
                if h <= 0 or w <= 0:
                    continue
                if h < tile_size // 3 or w < tile_size // 3:
                    continue

                try:
                    window = Window(col_start, row_start, w, h)
                    bands_to_read = min(self._ref_raster.count, 3)
                    tile = self._ref_raster.read(
                        list(range(1, bands_to_read + 1)),
                        window=window
                    )
                except Exception:
                    continue

                if tile.size == 0:
                    continue
                tile = np.transpose(tile, (1, 2, 0)).astype(np.uint8)
                if tile.shape[2] == 1:
                    tile = np.repeat(tile, 3, axis=2)
                elif tile.shape[2] == 2:
                    tile = np.concatenate([tile, tile[:, :, :1]], axis=2)

                if tile.shape[0] != tile_size or tile.shape[1] != tile_size:
                    padded = np.zeros((tile_size, tile_size, 3), dtype=np.uint8)
                    y_off = (tile_size - tile.shape[0]) // 2
                    x_off = (tile_size - tile.shape[1]) // 2
                    padded[y_off:y_off + tile.shape[0], x_off:x_off + tile.shape[1]] = tile
                    tile = padded

                center_col = min(max_col, col_start + (w // 2))
                center_row = min(max_row, row_start + (h // 2))
                anchor = self._pixel_to_latlon(center_col, center_row)
                if anchor is None:
                    continue
                lat, lon = anchor

                ref_name = f"{self._region}_r{row_start:06d}_c{col_start:06d}.png"
                emitted += 1
                yield {
                    "ref_name": ref_name,
                    "ref_image": tile,
                    "anchor": {
                        "lat": lat,
                        "lon": lon,
                        "anchor_source": "raster_geotransform",
                        "row_start": int(row_start),
                        "col_start": int(col_start),
                        "tile_size_px": int(tile_size),
                        "tile_step_px": int(step),
                    },
                }

        logger.info(
            "AVL-%s: built retrieval corpus with %d orthomosaic tiles (tile=%d, step=%d).",
            self._region,
            emitted,
            tile_size,
            step,
        )


# Metres per degree used by the runner's equirectangular distance metric
# (benchmark_runner_impl._latlon_distance_m). AVL24 ships sanitized scene-local
# metres with no geodetic anchor, so we synthesize lat/lon with this exact
# constant at lat0=0: cos(lat0)==1, which makes local metres and the runner's
# own distance metric mutually invertible instead of merely close.
_AVL24_M_PER_DEG = 111_320.0


class AVL24Dataset(BaseDataset):
    """
    AnyVisLoc 2026 public release (sanitized NPZ format).

    Structure (one directory per scene)::

        AnyVisLoc/
        └── Scene_01/
            ├── L01_reference.json
            ├── aerial_map.png / satellite_map.png
            ├── aerial_dsm.npy / satellite_dsm.npy
            └── L01_0001.npz ... L01_1846.npz

    ``L??_reference.json`` carries, per reference mode::

        {"modes": {"aerial": {"map_path": ..., "dsm_path": ...,
                              "map_resolution": [rx, ry],
                              "dsm_resolution": [rx, ry],
                              "map_origin_local": [x, y],
                              "dsm_origin_local": [dx, dy]}}}

    where ``dsm_origin_local`` is relative to ``map_origin_local`` and every
    coordinate is in scene-local metres.

    Each sample NPZ carries ``image`` (H, W, 3 uint8 RGB; empty when the frame
    is stored beside the NPZ under ``image_relpath``), ``K``, ``dist``,
    ``pose_c2w``, ``pose_w2c``, ``xyz``, ``euler_deg``, ``sample_id`` and
    ``scene_id``.

    Coordinates: the release is deliberately coordinate-sanitized — the map's
    upper-left corner is the origin of a scene-local metric frame and no
    latitude/longitude is published (see the upstream docs/GROUND_TRUTH.md).
    This adapter synthesizes a per-scene geodetic origin so the lat/lon-based
    benchmark stack runs unchanged; distances are preserved exactly under the
    runner's metric, absolute geography is meaningless by construction.

    Altitude: upstream shifts UAV heights and DSM elevations by the mean scene
    elevation, and explicitly asks that the dataset not be used to benchmark
    height estimation. Report horizontal error only.
    """

    def __init__(self,
                 data_root: Path,
                 scene: str = "Scene_01",
                 reference_mode: str = "aerial",
                 tile_size_m: float = 256.0,
                 tile_size_px: Optional[int] = None,
                 **kwargs):
        super().__init__(data_root, **kwargs)
        self._scene = str(scene)
        self._reference_mode = self._normalize_reference_mode(reference_mode)
        self._tile_size_m = float(tile_size_m)
        self._forced_tile_size_px = int(tile_size_px) if tile_size_px else None
        self._scene_id: Optional[int] = None
        self._ref_meta: Optional[Dict[str, Any]] = None
        self._ref_map: Optional[np.ndarray] = None
        self._dsm: Optional[np.ndarray] = None

    @property
    def name(self) -> str:
        return f"AVL24-{self._scene}-{self._reference_mode}"

    @staticmethod
    def _normalize_reference_mode(mode: str) -> str:
        m = str(mode or "aerial").lower()
        if m in ("aerial", "ortho", "high"):
            return "aerial"
        if m in ("satellite", "sat", "low"):
            return "satellite"
        raise ValueError(
            f"Unsupported reference_mode {mode!r} for AVL24; use 'aerial' or 'satellite'."
        )

    @property
    def _scene_dir(self) -> Path:
        return self._data_root / self._scene

    # -- reference metadata -------------------------------------------------

    def _load_ref_meta(self) -> Dict[str, Any]:
        """Parse ``L??_reference.json``. Isolated so a schema change is local."""
        if self._ref_meta is not None:
            return self._ref_meta

        scene_dir = self._scene_dir
        candidates = sorted(scene_dir.glob("L??_reference.json"))
        if not candidates:
            raise FileNotFoundError(
                f"No L??_reference.json in {scene_dir}. Expected the AnyVisLoc NPZ release layout."
            )
        ref_json = candidates[0]
        with open(ref_json, "r", encoding="utf-8") as f:
            meta = json.load(f)

        modes = meta.get("modes", {})
        if self._reference_mode not in modes:
            raise KeyError(
                f"{ref_json} has no reference mode {self._reference_mode!r}; available={sorted(modes)}"
            )
        mode = modes[self._reference_mode]

        def _vec2(value, field, default=None):
            if value is None:
                if default is None:
                    raise KeyError(f"{ref_json} mode {self._reference_mode!r} missing {field}")
                value = default
            arr = np.asarray(value, dtype=np.float64).reshape(-1)
            if arr.size == 1:
                arr = np.array([arr[0], arr[0]], dtype=np.float64)
            if arr.size < 2:
                raise ValueError(f"{ref_json}: {field} must have 2 components, got {value!r}")
            return arr[:2]

        def _abs_path(value, field):
            if value is None:
                raise KeyError(f"{ref_json} mode {self._reference_mode!r} missing {field}")
            p = Path(str(value))
            return p if p.is_absolute() else ref_json.parent / p

        self._scene_id = int(str(ref_json.name)[1:3])
        map_origin = _vec2(mode.get("map_origin_local"), "map_origin_local", [0.0, 0.0])
        dsm_origin_rel = _vec2(mode.get("dsm_origin_local"), "dsm_origin_local", [0.0, 0.0])
        self._ref_meta = {
            "ref_json": ref_json,
            "map_path": _abs_path(mode.get("map_path"), "map_path"),
            "dsm_path": _abs_path(mode.get("dsm_path"), "dsm_path"),
            "map_resolution": _vec2(mode.get("map_resolution"), "map_resolution"),
            "dsm_resolution": _vec2(mode.get("dsm_resolution"), "dsm_resolution"),
            "map_origin_local": map_origin,
            # Upstream stores the DSM origin relative to the map origin.
            "dsm_origin_local": (map_origin + dsm_origin_rel),
        }
        return self._ref_meta

    @property
    def _origin_lat(self) -> float:
        return 0.0

    @property
    def _origin_lon(self) -> float:
        # Park each scene ~111 km apart in the synthetic frame so a corpus built
        # from several scenes cannot produce spurious near-zero tile distances.
        self._load_ref_meta()
        return float(self._scene_id or 0)

    @property
    def tile_size_px(self) -> int:
        if self._forced_tile_size_px:
            return self._forced_tile_size_px
        meta = self._load_ref_meta()
        res = float(meta["map_resolution"][0])
        # Scene map resolution spans 0.020-0.320 m/px across the 24 scenes, so a
        # fixed pixel tile would cover 41 m in one scene and 655 m in another.
        # Size the tile in metres and convert per scene.
        px = int(round(self._tile_size_m / max(res, 1e-9)))
        return int(np.clip(px, 256, 4096))

    # -- coordinate conversions --------------------------------------------

    def _local_to_latlon(self, x_m: float, y_m: float) -> Tuple[float, float]:
        """Scene-local metres (x east, y south) -> synthetic WGS84."""
        return (
            self._origin_lat - float(y_m) / _AVL24_M_PER_DEG,
            self._origin_lon + float(x_m) / _AVL24_M_PER_DEG,
        )

    def _latlon_to_local(self, lat: float, lon: float) -> Tuple[float, float]:
        return (
            (float(lon) - self._origin_lon) * _AVL24_M_PER_DEG,
            (self._origin_lat - float(lat)) * _AVL24_M_PER_DEG,
        )

    def _local_to_pixel(self, x_m: float, y_m: float) -> Tuple[int, int]:
        meta = self._load_ref_meta()
        origin = meta["map_origin_local"]
        res = meta["map_resolution"]
        col = (float(x_m) - float(origin[0])) / max(float(res[0]), 1e-12)
        row = (float(y_m) - float(origin[1])) / max(float(res[1]), 1e-12)
        return int(round(col)), int(round(row))

    def _pixel_to_local(self, col: float, row: float) -> Tuple[float, float]:
        meta = self._load_ref_meta()
        origin = meta["map_origin_local"]
        res = meta["map_resolution"]
        return (
            float(origin[0]) + float(col) * float(res[0]),
            float(origin[1]) + float(row) * float(res[1]),
        )

    def _latlon_to_pixel(self, lat: float, lon: float) -> Tuple[int, int]:
        x, y = self._latlon_to_local(lat, lon)
        return self._local_to_pixel(x, y)

    def _pixel_to_latlon(self, col: float, row: float) -> Tuple[float, float]:
        x, y = self._pixel_to_local(col, row)
        return self._local_to_latlon(x, y)

    # -- reference arrays ---------------------------------------------------

    def _load_reference_map(self) -> Optional[np.ndarray]:
        if self._ref_map is not None:
            return self._ref_map
        meta = self._load_ref_meta()
        map_path = meta["map_path"]
        if not map_path.exists():
            logger.warning("AVL24 %s: reference map missing: %s", self._scene, map_path)
            return None
        import cv2
        # ponytail: whole map in RAM (upstream does the same). A 0.02 m/px scene
        # map can reach several GB — switch to windowed PIL crops if a scene OOMs.
        img = cv2.imread(str(map_path), cv2.IMREAD_COLOR)
        if img is None:
            logger.warning("AVL24 %s: cannot read reference map %s", self._scene, map_path)
            return None
        self._ref_map = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        logger.info(
            "AVL24 %s: reference map %s loaded (%dx%d, %.3f m/px, tile=%d px)",
            self._scene, map_path.name, self._ref_map.shape[1], self._ref_map.shape[0],
            float(meta["map_resolution"][0]), self.tile_size_px,
        )
        return self._ref_map

    def _load_dsm(self) -> Optional[np.ndarray]:
        if self._dsm is not None:
            return self._dsm
        meta = self._load_ref_meta()
        dsm_path = meta["dsm_path"]
        if not dsm_path.exists():
            logger.warning("AVL24 %s: DSM missing: %s", self._scene, dsm_path)
            return None
        # mmap keeps a coarse-but-large DSM cheap to open.
        dsm = np.load(str(dsm_path), mmap_mode="r")
        if dsm.ndim == 3:
            dsm = dsm[:, :, 0]
        self._dsm = dsm
        return self._dsm

    def build_dsm_map(self):
        """Build the DSM bridge consumed by the DSM-PnP pose head."""
        from modules.vision.dsm_map import LocalGridDSMMap

        meta = self._load_ref_meta()
        dsm = self._load_dsm()
        if dsm is None:
            return None
        return LocalGridDSMMap(
            dsm=dsm,
            dsm_resolution=meta["dsm_resolution"],
            dsm_origin_local=meta["dsm_origin_local"],
            map_resolution=meta["map_resolution"],
            map_origin_local=meta["map_origin_local"],
            origin_lat=self._origin_lat,
            origin_lon=self._origin_lon,
            m_per_deg=_AVL24_M_PER_DEG,
        )

    # -- samples ------------------------------------------------------------

    @staticmethod
    def _npz_str(value) -> str:
        arr = np.asarray(value)
        return str(arr.item()) if arr.shape == () else str(arr)

    def _parse_npz(self, path: Path, with_image: bool) -> Dict[str, Any]:
        """Read one sample NPZ. Isolated so a schema change is local."""
        with np.load(path, allow_pickle=True) as data:
            keys = set(data.files)
            xyz = np.asarray(data["xyz"], dtype=np.float64).reshape(3)
            euler = np.asarray(data["euler_deg"], dtype=np.float64).reshape(3)
            K = np.asarray(data["K"], dtype=np.float64).reshape(3, 3)
            dist = (
                np.asarray(data["dist"], dtype=np.float64).reshape(-1)
                if "dist" in keys else np.zeros(5, dtype=np.float64)
            )
            pose_c2w = (np.asarray(data["pose_c2w"], dtype=np.float64).reshape(4, 4)
                        if "pose_c2w" in keys else None)
            sample_id = self._npz_str(data["sample_id"]) if "sample_id" in keys else path.stem
            scene_id = int(data["scene_id"]) if "scene_id" in keys else (self._scene_id or 0)

            if "image_size" in keys:
                size = np.asarray(data["image_size"]).reshape(-1)
            elif "downsampled_size" in keys:
                size = np.asarray(data["downsampled_size"]).reshape(-1)
            else:
                size = None

            image = None
            image_relpath = self._npz_str(data["image_relpath"]) if "image_relpath" in keys else ""
            if with_image:
                raw = data["image"] if "image" in keys else np.zeros(0)
                image = np.asarray(raw) if np.asarray(raw).size else None

        if size is not None and size.size >= 2:
            height, width = int(size[0]), int(size[1])
        elif image is not None:
            height, width = int(image.shape[0]), int(image.shape[1])
        else:
            height, width = int(round(2.0 * K[1, 2])), int(round(2.0 * K[0, 2]))

        if with_image and image is None and image_relpath:
            import cv2
            img_path = path.parent / image_relpath
            loaded = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            if loaded is None:
                raise FileNotFoundError(f"AVL24 {path.name}: cannot read image_relpath {img_path}")
            image = cv2.cvtColor(loaded, cv2.COLOR_BGR2RGB)

        return {
            "sample_id": sample_id,
            "scene_id": scene_id,
            "xyz": xyz,
            "euler_deg": euler,
            "K": K,
            "pose_c2w": pose_c2w,
            "dist": dist,
            "width": width,
            "height": height,
            "image": image,
        }

    # Rays that leave the camera within this many degrees of the horizon, or
    # that reach the ground farther than this many AGL heights away, make the
    # footprint unbounded for FORGE purposes.
    _FOOTPRINT_MIN_DOWN = 0.10       # cos of the angle to the horizontal plane
    _FOOTPRINT_MAX_RANGE_AGL = 15.0

    def _ground_height(self, x_m: float, y_m: float) -> Optional[float]:
        """DSM height under (x, y), ignoring nodata; None when unavailable."""
        dsm = self._load_dsm()
        if dsm is None:
            return None
        meta = self._load_ref_meta()
        rx, ry = meta["dsm_resolution"]; ox, oy = meta["dsm_origin_local"]
        c, r = int((x_m - ox) / rx), int((y_m - oy) / ry)
        h, w = dsm.shape[:2]
        win = dsm[max(0, r - 3):min(h, r + 4), max(0, c - 3):min(w, c + 4)]
        vals = np.asarray(win, dtype=np.float64).ravel()
        vals = vals[np.isfinite(vals) & (vals > -1000.0) & (vals < 10000.0)]
        return float(np.median(vals)) if vals.size else None

    def _footprint_latlon(self, s: Dict[str, Any]) -> Optional[List[float]]:
        box = self._forge_footprint_box(s)
        if box is None:
            return None
        xmin, xmax, ymin, ymax = box
        lat_a, lon_a = self._local_to_latlon(xmin, ymin)
        lat_b, lon_b = self._local_to_latlon(xmax, ymax)
        return [min(lat_a, lat_b), max(lat_a, lat_b), min(lon_a, lon_b), max(lon_a, lon_b)]

    def _forge_footprint_box(self, s: Dict[str, Any]) -> Optional[Tuple[float, float, float, float]]:
        """Ground footprint of the camera as a local (xmin, xmax, ymin, ymax) box.

        Corner and edge-midpoint rays are intersected with the ground plane at
        the DSM height under the camera. In this release the c2w rotation
        maps the optical axis to +z towards the ground (a nadir frame has
        d_z = +1) while xyz.z is height, so a ray reaches the ground after
        AGL / d_z. Oblique frames look far ahead of the camera, which is why
        the box is not centred on the ground-truth position. Returns None when
        any ray is near the horizon or reaches too far: such a footprint has
        no bound, and the frame cannot carry a wrong-reference fault.
        """
        P = s.get("pose_c2w")
        if P is None:
            return None
        x, y, z = (float(v) for v in s["xyz"])
        ground = self._ground_height(x, y)
        if ground is None:
            return None
        agl = z - ground
        if not np.isfinite(agl) or agl <= 1.0:
            return None
        K = s["K"]; W, H = float(s["width"]), float(s["height"])
        Kinv = np.linalg.inv(K); R = P[:3, :3]
        xs, ys = [], []
        for u, v in ((0, 0), (W / 2, 0), (W, 0), (W, H / 2), (W, H), (W / 2, H), (0, H), (0, H / 2), (W / 2, H / 2)):
            d = R @ (Kinv @ np.array([u, v, 1.0])); d /= np.linalg.norm(d)
            if d[2] < self._FOOTPRINT_MIN_DOWN:
                return None
            t = agl / d[2]
            gx, gy = x + t * d[0], y + t * d[1]
            if np.hypot(gx - x, gy - y) > self._FOOTPRINT_MAX_RANGE_AGL * agl:
                return None
            xs.append(gx); ys.append(gy)
        return min(xs), max(xs), min(ys), max(ys)

    def load(self):
        scene_dir = self._scene_dir
        if not scene_dir.is_dir():
            raise FileNotFoundError(f"AVL24 scene directory not found: {scene_dir}")
        self._load_ref_meta()

        sample_paths = sorted(scene_dir.glob("L??_????.npz"))
        if not sample_paths:
            raise FileNotFoundError(
                f"No L??_????.npz samples in {scene_dir}. Expected the AnyVisLoc NPZ release layout."
            )

        for path in sample_paths:
            s = self._parse_npz(path, with_image=False)
            x, y, z = float(s["xyz"][0]), float(s["xyz"][1]), float(s["xyz"][2])
            lat, lon = self._local_to_latlon(x, y)
            roll, pitch, yaw = (float(v) for v in s["euler_deg"])
            self._entries.append({
                "image_name": s["sample_id"],
                "query_path": path,
                "gt_pose": {
                    "lat": lat,
                    "lon": lon,
                    "alt": z,
                    "yaw": yaw,
                    "pitch": pitch,
                    "roll": roll,
                    # Upstream geometry uses ground_dist = z / |sin(90 deg + pitch)|
                    # and a view-centre shift of z * tan(pitch), both of which are
                    # identities only when pitch == 0 at nadir.
                    "pitch_convention": "aircraft_nadir_zero",
                },
                "camera": {
                    "fx": float(s["K"][0, 0]),
                    "fy": float(s["K"][1, 1]),
                    "cx": float(s["K"][0, 2]),
                    "cy": float(s["K"][1, 2]),
                    "width": int(s["width"]),
                    "height": int(s["height"]),
                },
                "metadata": {
                    "dataset": "avl24",
                    "scene": self._scene,
                    "scene_id": int(s["scene_id"]),
                    "reference_mode": self._reference_mode,
                    "local_x_m": x,
                    "local_y_m": y,
                    "local_z_m": z,
                    "distortion": [float(v) for v in s["dist"]],
                    # Heights are shifted by the scene's mean elevation upstream;
                    # the release is not a height-estimation benchmark.
                    "altitude_is_sanitized": True,
                    # FORGE: camera ground footprint (lat_min, lat_max, lon_min,
                    # lon_max) in the synthetic frame, or None if unbounded;
                    # and the reference tile half-extent in metres.
                    "forge_footprint_latlon_box": self._footprint_latlon(s),
                    "forge_ref_half_m": self._tile_size_m / 2.0,
                },
            })

        self._loaded = True
        logger.info(
            "AVL24 %s (%s): loaded %d samples",
            self._scene, self._reference_mode, len(self._entries),
        )

    def _crop_reference_tile(self, lat: float, lon: float) -> Optional[np.ndarray]:
        ref = self._load_reference_map()
        if ref is None:
            return None

        col, row = self._latlon_to_pixel(lat, lon)
        tile_size = self.tile_size_px
        half = tile_size // 2
        height, width = ref.shape[0], ref.shape[1]

        col_start = max(0, col - half)
        row_start = max(0, row - half)
        col_end = min(width, col + half)
        row_end = min(height, row + half)
        if col_end <= col_start or row_end <= row_start:
            return None

        tile = ref[row_start:row_end, col_start:col_end]
        if tile.shape[0] != tile_size or tile.shape[1] != tile_size:
            padded = np.zeros((tile_size, tile_size, 3), dtype=np.uint8)
            y_off = (tile_size - tile.shape[0]) // 2
            x_off = (tile_size - tile.shape[1]) // 2
            padded[y_off:y_off + tile.shape[0], x_off:x_off + tile.shape[1]] = tile
            tile = padded
        return np.ascontiguousarray(tile, dtype=np.uint8)

    def _get_frame(self, idx: int) -> Dict[str, Any]:
        entry = self._entries[idx]
        meta = self._load_ref_meta()
        gt = entry["gt_pose"]
        s = self._parse_npz(entry["query_path"], with_image=True)

        image = s["image"]
        if image is None:
            image = np.zeros((entry["camera"]["height"], entry["camera"]["width"], 3), dtype=np.uint8)
        elif image.dtype != np.uint8:
            # Upstream stores uint8 RGB; a float array would be normalized to [0,1].
            scale = 255.0 if float(np.nanmax(image)) <= 1.0 else 1.0
            image = np.clip(np.asarray(image, dtype=np.float32) * scale, 0, 255).astype(np.uint8)

        return {
            "image_name": entry["image_name"],
            "query_path": entry["query_path"],
            "query_image": image,
            "gt_pose": gt,
            "metadata": entry["metadata"],
            "camera": entry["camera"],
            "gsd": float(meta["map_resolution"][0]),
            "reference_image": self._crop_reference_tile(gt["lat"], gt["lon"]),
        }

    def iter_reference_corpus(self) -> Iterator[Dict[str, Any]]:
        ref = self._load_reference_map()
        if ref is None:
            logger.warning("AVL24 %s: reference map unavailable, retrieval corpus empty.", self._scene)
            return

        tile_size = self.tile_size_px
        step = max(128, tile_size // 2)  # 50% overlap, matching the AVL adapter.
        height, width = ref.shape[0], ref.shape[1]
        emitted = 0

        for row_start in range(0, height, step):
            for col_start in range(0, width, step):
                row_end = min(height, row_start + tile_size)
                col_end = min(width, col_start + tile_size)
                h = row_end - row_start
                w = col_end - col_start
                if h < tile_size // 3 or w < tile_size // 3:
                    continue

                tile = ref[row_start:row_end, col_start:col_end]
                if tile.shape[0] != tile_size or tile.shape[1] != tile_size:
                    padded = np.zeros((tile_size, tile_size, 3), dtype=np.uint8)
                    y_off = (tile_size - tile.shape[0]) // 2
                    x_off = (tile_size - tile.shape[1]) // 2
                    padded[y_off:y_off + tile.shape[0], x_off:x_off + tile.shape[1]] = tile
                    tile = padded

                lat, lon = self._pixel_to_latlon(
                    min(width - 1, col_start + w // 2),
                    min(height - 1, row_start + h // 2),
                )
                emitted += 1
                yield {
                    "ref_name": f"{self._scene}_r{row_start:06d}_c{col_start:06d}.png",
                    "ref_image": np.ascontiguousarray(tile, dtype=np.uint8),
                    "anchor": {
                        "lat": lat,
                        "lon": lon,
                        "anchor_source": "avl24_local_grid",
                        "row_start": int(row_start),
                        "col_start": int(col_start),
                        "tile_size_px": int(tile_size),
                        "tile_step_px": int(step),
                    },
                }

        logger.info(
            "AVL24 %s: built retrieval corpus with %d tiles (tile=%d px, step=%d px).",
            self._scene, emitted, tile_size, step,
        )


def _anchor_from_map_bbox(
    bbox: Dict[str, float],
    row_start: int,
    col_start: int,
    img_height: int,
    img_width: int,
    tile_h: int,
    tile_w: int,
) -> Tuple[float, float, str]:
    """Interpolate WGS84 anchor at tile center from map corner bbox metadata."""
    lt_lat = float(bbox.get("lt_lat", bbox.get("lat", 0.0)))
    lt_lon = float(bbox.get("lt_lon", bbox.get("lon", 0.0)))
    rb_lat = float(bbox.get("rb_lat", lt_lat))
    rb_lon = float(bbox.get("rb_lon", lt_lon))
    h = max(1, int(img_height))
    w = max(1, int(img_width))
    cy = min(h - 1, int(row_start) + int(tile_h) // 2)
    cx = min(w - 1, int(col_start) + int(tile_w) // 2)
    v = cy / float(h)
    u = cx / float(w)
    lat = lt_lat + (rb_lat - lt_lat) * v
    lon = lt_lon + (rb_lon - lt_lon) * u
    if np.isfinite(lat) and np.isfinite(lon):
        return float(lat), float(lon), "bbox_interpolated"
    return 0.0, 0.0, "invalid"


def _tile_anchor_wgs84_from_raster(
    src: Any,
    row_start: int,
    col_start: int,
    w: int,
    h: int,
    file_fallback: Optional[Dict[str, float]] = None,
) -> Tuple[float, float, str]:
    """
    Compute per-tile WGS84 anchor from raster geotransform (tile center pixel).

    Returns:
        (lat, lon, source) where source is ``geotiff_center``, ``file_fallback``,
        or ``invalid`` when coordinates are not usable.
    """
    file_fallback = file_fallback or {}
    fallback_lat = float(file_fallback.get("lat", 0.0))
    fallback_lon = float(file_fallback.get("lon", 0.0))

    crs = getattr(src, "crs", None)
    if crs is None:
        if abs(fallback_lat) > 1e-6 or abs(fallback_lon) > 1e-6:
            return fallback_lat, fallback_lon, "file_fallback"
        return 0.0, 0.0, "invalid"

    max_row = max(0, int(src.height) - 1)
    max_col = max(0, int(src.width) - 1)
    center_row = min(max_row, int(row_start) + int(h) // 2)
    center_col = min(max_col, int(col_start) + int(w) // 2)
    try:
        x, y = src.xy(center_row, center_col)
        if str(crs).upper() != "EPSG:4326":
            from pyproj import Transformer

            transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
            x, y = transformer.transform(x, y)
        lat = float(y)
        lon = float(x)
        if np.isfinite(lat) and np.isfinite(lon):
            return lat, lon, "geotiff_center"
    except Exception as exc:
        logger.debug("Tile anchor georef failed: %s", exc)

    if abs(fallback_lat) > 1e-6 or abs(fallback_lon) > 1e-6:
        return fallback_lat, fallback_lon, "file_fallback"
    return 0.0, 0.0, "invalid"


class UAVVisLocDataset(BaseDataset):
    """
    UAV-VisLoc Dataset Adapter.
    
    Structure:
        UAV_VisLoc_dataset/
        ├── 01/              # Location 01
        │   ├── drone/       # Query drone images
        │   ├── satellite_map/  # Reference satellite tiles
        │   └── metadata.json
        ├── 02/
        ...
    """
    
    def __init__(self,
                 data_root: Path,
                 locations: List[str] = None,
                 protocol: str = "intra_flight",
                 tile_size_px: int = 2048,
                 tile_step_px: int = None,
                 **kwargs):
        super().__init__(data_root, **kwargs)
        self._locations = locations  # None = all locations
        self._protocol = str(protocol or "intra_flight")
        if self._protocol not in {"intra_flight", "cross_flight", "all_maps"}:
            raise ValueError(
                f"Unknown UAV-VisLoc protocol: {self._protocol}. "
                "Use intra_flight, cross_flight, or all_maps."
            )
        self._tile_size_px = max(64, int(tile_size_px or 768))
        default_step = max(1, self._tile_size_px // 2)
        self._tile_step_px = max(1, int(tile_step_px or default_step))
        self._satellite_ranges = {}
    
    @property
    def name(self) -> str:
        return "UAV-VisLoc"
    
    def load(self):
        """Load UAV-VisLoc dataset metadata."""
        self._data_root = _normalize_uav_visloc_root(self._data_root)
        # Discover locations
        if self._locations is None:
            digit_dirs = [
                d.name for d in self._data_root.iterdir()
                if d.is_dir() and d.name.isdigit()
            ]
            self._locations = sorted(digit_dirs)
        
        for loc in self._locations:
            loc_dir = self._data_root / loc
            if not loc_dir.exists():
                continue
            
            # Look for drone images
            drone_dir = loc_dir / "drone"
            if not drone_dir.exists():
                # Try alternative layouts
                drone_dir = loc_dir / "query"
                if not drone_dir.exists():
                    drone_dir = loc_dir
            
            # Look for metadata (json or csv)
            meta_path_json = loc_dir / "metadata.json"
            meta_path_csv = loc_dir / f"{loc}.csv"
            
            meta_lookup = {}
            if meta_path_json.exists():
                with open(meta_path_json, 'r') as f:
                    metadata = json.load(f)
                if isinstance(metadata, list):
                    for m in metadata:
                        key = m.get('name', m.get('filename', ''))
                        meta_lookup[key] = m
                elif isinstance(metadata, dict) and 'images' in metadata:
                    for m in metadata['images']:
                        key = m.get('name', m.get('filename', ''))
                        meta_lookup[key] = m
            elif meta_path_csv.exists():
                with open(meta_path_csv, 'r') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        try:
                            # Parse generic keys. Expecting num,filename,date,lat,lon,height,Omega,Kappa,Phi1,Phi2
                            key = row.get('filename', '')
                            if not key:
                                continue
                            m = {
                                'lat': float(row.get('lat', 0)),
                                'lon': float(row.get('lon', 0)),
                                'alt': float(row.get('height', 0)),
                                # README_dataset.txt: Omega=pitch, Kappa=roll,
                                # Phi1/Phi2=yaw (Phi1 higher confidence)
                                'yaw': float(row.get('Phi1', row.get('Phi2', 0))),
                                'pitch': float(row.get('Omega', 0)),
                                'roll': float(row.get('Kappa', 0)),
                            }
                            meta_lookup[key] = m
                        except Exception as e:
                            logger.warning(f"Failed to parse CSV row for {loc}: {e}")
            
            # Get drone images
            images = sorted(
                {
                    *drone_dir.glob("*.jpg"),
                    *drone_dir.glob("*.JPG"),
                    *drone_dir.glob("*.png"),
                }
            )
            
            for img_path in images:
                meta = meta_lookup.get(img_path.name, {})
                
                self._entries.append({
                    'image_name': f"{loc}/{img_path.name}",
                    'query_path': img_path,
                    'gt_pose': {
                        'lat': meta.get('latitude', meta.get('lat', 0)),
                        'lon': meta.get('longitude', meta.get('lon', 0)),
                        'alt': meta.get('altitude', meta.get('alt', meta.get('rel_alt', 0))),
                        'yaw': meta.get('yaw', meta.get('heading', 0)),
                        'pitch': meta.get('pitch', 0),
                        'roll': meta.get('roll', 0),
                        # Photogrammetry Omega/Phi/Kappa: near-nadir frames have
                        # Phi ~ 0, i.e. aircraft convention. The previous
                        # 'gimbal_neg90_nadir' made IPM read Phi~0 as 90 deg
                        # from nadir and warp every use_ipm frame by ~90 deg.
                        'pitch_convention': 'aircraft_nadir_zero',
                    },
                    'metadata': {
                        'dataset': 'uav_visloc',
                        'location': loc,
                    },
                })
        
        self._loaded = True
        logger.info(f"UAV-VisLoc: loaded {len(self._entries)} entries from {len(self._locations)} locations")

        ranges_csv = self._data_root / "satellite_coordinates_range.csv"
        if not ranges_csv.exists():
            alt = self._data_root / "satellite_ coordinates_range.csv"
            if alt.exists():
                ranges_csv = alt
        if ranges_csv.exists():
            try:
                with open(ranges_csv, "r") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        key = str(
                            row.get("filename")
                            or row.get("mapname")
                            or row.get("map_name")
                            or ""
                        ).strip()
                        if not key:
                            continue
                        try:
                            if row.get("LT_lat_map") and row.get("RB_lat_map"):
                                lt_lat = float(row.get("LT_lat_map", 0.0))
                                lt_lon = float(row.get("LT_lon_map", 0.0))
                                rb_lat = float(row.get("RB_lat_map", 0.0))
                                rb_lon = float(row.get("RB_lon_map", 0.0))
                                self._satellite_ranges[key] = {
                                    "lt_lat": lt_lat,
                                    "lt_lon": lt_lon,
                                    "rb_lat": rb_lat,
                                    "rb_lon": rb_lon,
                                    "lat": (lt_lat + rb_lat) * 0.5,
                                    "lon": (lt_lon + rb_lon) * 0.5,
                                }
                            else:
                                self._satellite_ranges[key] = {
                                    "lat": float(row.get("latitude", 0.0)),
                                    "lon": float(row.get("longitude", 0.0)),
                                }
                        except (TypeError, ValueError):
                            continue
            except Exception as e:
                logger.warning("Failed parsing satellite_coordinates_range.csv: %s", e)

    def _resolve_satellite_bbox(self, sat_path: Path) -> Dict[str, float]:
        """Resolve map bbox metadata for a satellite raster (exact or prefix match)."""
        if not self._satellite_ranges:
            return {}
        exact = self._satellite_ranges.get(sat_path.name)
        if exact:
            return exact
        stem = sat_path.stem
        if stem in self._satellite_ranges:
            return self._satellite_ranges[stem]
        candidates = [
            (k, v)
            for k, v in self._satellite_ranges.items()
            if stem.startswith(Path(k).stem) or Path(k).stem.startswith(stem.split("_")[0])
        ]
        if not candidates:
            return {}
        candidates.sort(key=lambda kv: len(kv[0]), reverse=True)
        return candidates[0][1]
    
    def _get_frame(self, idx: int) -> Dict[str, Any]:
        entry = self._entries[idx]
        try:
            import cv2
            if entry['query_path'].exists():
                query_image = cv2.cvtColor(
                    cv2.imread(str(entry['query_path'])), cv2.COLOR_BGR2RGB
                )
            else:
                query_image = np.zeros((720, 1280, 3), dtype=np.uint8)
        except Exception:
            query_image = np.zeros((720, 1280, 3), dtype=np.uint8)

        img_h, img_w = query_image.shape[:2]
        # UAV-VisLoc publishes no intrinsics and strips EXIF. The IPM warp
        # rotates about (cx, cy), so the principal point MUST be the real
        # image center; the old 1280x720 placeholder spun 4k-wide queries
        # around a corner point and blanked them. fx is estimated from
        # nadir geometry alt/GSD with the README's ~0.15 m/px drone GSD.
        # ponytail: fx is +/-35% (GSD range 0.1-0.2); only small-angle IPM
        # correction and the PnP fallback consume it - homography-scale
        # positions come from tile georef. Refine from EXIF/paper if found.
        alt = float(entry['gt_pose'].get('alt', 0) or 0)
        fx = alt / 0.15 if alt > 0 else 2700.0
        frame = {
            'image_name': entry['image_name'],
            'query_path': entry['query_path'],
            'gt_pose': entry['gt_pose'],
            'metadata': entry['metadata'],
            'camera': {'fx': fx, 'fy': fx,
                       'cx': img_w / 2.0, 'cy': img_h / 2.0,
                       'width': img_w, 'height': img_h},
            'gsd': 0.3,
            # The satellite rasters are EPSG:4326 with square-DEGREE pixels
            # (about 2.682e-6 deg/px on both axes, on all 11 maps), so the
            # ground pixel is not square: 'gsd' is the north metres per pixel,
            # and the east metres per pixel is gsd * cos(latitude). The latitude
            # that makes this exact belongs to the RETRIEVED REFERENCE tile,
            # which only the solve call site knows, so publish the axis
            # convention here rather than a number. Taking a latitude from the
            # query would also leak ground truth into the scale.
            'gsd_axis': 'square_degree',
            'query_image': query_image,
        }
        frame['reference_image'] = None
        return frame

    def selected_query_locations(self) -> List[str]:
        """UAV-VisLoc locations present in the sampled query subset."""
        if not self._loaded:
            self.load()
        locations = {
            str((self._entries[idx].get("metadata") or {}).get("location", ""))
            for idx in self.selected_entry_indices()
        }
        locations.discard("")
        return sorted(locations)

    def retrieval_tile_config(self) -> Dict[str, int]:
        """Tile-grid identity used for UAV-VisLoc retrieval corpus/cache keys."""
        return {
            "uav_tile_size_px": int(self._tile_size_px),
            "uav_tile_step_px": int(self._tile_step_px),
        }

    def iter_reference_corpus(self) -> Iterator[Dict[str, Any]]:
        """Build retrieval corpus from per-location satellite maps (tiled when possible)."""
        import cv2
        if not self._loaded:
            self.load()
        # UAV-VisLoc default uses 768px tiles at 50% overlap. The larger tile
        # keeps more semantic context for retrieval; smaller diagnostic tiles improve
        # center-distance floor but can hurt descriptor ranking.
        tile_size = int(self._tile_size_px)
        step = int(self._tile_step_px)
        total_corpus_tiles = 0
        active_query_locations = set(self.selected_query_locations())
        if self._protocol == "intra_flight":
            corpus_locations = [loc for loc in self._locations if str(loc) in active_query_locations]
        elif self._protocol == "cross_flight":
            corpus_locations = [loc for loc in self._locations if str(loc) not in active_query_locations]
        else:
            corpus_locations = list(self._locations)
        if not corpus_locations:
            logger.warning(
                "UAV-VisLoc protocol=%s produced an empty reference corpus "
                "(active_query_locations=%s, locations=%s).",
                self._protocol,
                sorted(active_query_locations),
                self._locations,
            )

        for loc in corpus_locations:
            loc_dir = self._data_root / str(loc)
            sat_dir = loc_dir / "satellite_map"
            loc_tiles_emitted = 0
            sat_paths = sorted(list(loc_dir.glob("satellite*.tif")) + list(loc_dir.glob("satellite*.TIF")))
            sat_paths += sorted(list(sat_dir.glob("satellite*.tif")) + list(sat_dir.glob("satellite*.TIF")))
            sat_paths += sorted(list(sat_dir.glob("*.png")) + list(sat_dir.glob("*.jpg")) + list(sat_dir.glob("*.JPG")))
            for p in sat_paths:
                ext = p.suffix.lower()
                emitted_any = False
                if ext in {".tif", ".tiff"}:
                    try:
                        import rasterio
                        from rasterio.windows import Window
                    except Exception:
                        rasterio = None
                    if rasterio is not None:
                        try:
                            with rasterio.open(str(p)) as src:
                                has_geo = bool(getattr(src, "crs", None))
                                for row_start in range(0, int(src.height), step):
                                    for col_start in range(0, int(src.width), step):
                                        row_end = min(int(src.height), row_start + tile_size)
                                        col_end = min(int(src.width), col_start + tile_size)
                                        h = row_end - row_start
                                        w = col_end - col_start
                                        if h < tile_size // 3 or w < tile_size // 3:
                                            continue
                                        window = Window(col_start, row_start, w, h)
                                        bands_to_read = min(int(src.count), 3)
                                        tile = src.read(list(range(1, bands_to_read + 1)), window=window)
                                        if tile.size == 0:
                                            continue
                                        tile = np.transpose(tile, (1, 2, 0)).astype(np.uint8)
                                        if tile.shape[2] == 1:
                                            tile = np.repeat(tile, 3, axis=2)
                                        elif tile.shape[2] == 2:
                                            tile = np.concatenate([tile, tile[:, :, :1]], axis=2)
                                        if tile.shape[0] != tile_size or tile.shape[1] != tile_size:
                                            padded = np.zeros((tile_size, tile_size, 3), dtype=np.uint8)
                                            y_off = (tile_size - tile.shape[0]) // 2
                                            x_off = (tile_size - tile.shape[1]) // 2
                                            padded[y_off:y_off + tile.shape[0], x_off:x_off + tile.shape[1]] = tile
                                            tile = padded
                                        file_fb = self._resolve_satellite_bbox(p)
                                        anchor_lat, anchor_lon, anchor_src = _tile_anchor_wgs84_from_raster(
                                            src,
                                            row_start=row_start,
                                            col_start=col_start,
                                            w=w,
                                            h=h,
                                            file_fallback=file_fb,
                                        )
                                        # Tiled corpus must not duplicate one file-level anchor for every tile.
                                        if anchor_src == "invalid":
                                            continue
                                        if anchor_src == "file_fallback" and has_geo:
                                            continue
                                        yield {
                                            "ref_name": f"{loc}/{p.stem}_r{row_start:06d}_c{col_start:06d}.png",
                                            "ref_image": tile,
                                            "anchor": {
                                                "lat": anchor_lat,
                                                "lon": anchor_lon,
                                            "anchor_source": anchor_src,
                                            "row_start": int(row_start),
                                            "col_start": int(col_start),
                                            "tile_size_px": int(tile_size),
                                            "tile_step_px": int(step),
                                        },
                                    }
                                        emitted_any = True
                                        loc_tiles_emitted += 1
                                        total_corpus_tiles += 1
                        except Exception as e:
                            logger.warning("UAV-VisLoc: failed tiling %s: %s", p, e)
                if emitted_any:
                    continue
                img_bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
                if img_bgr is None:
                    continue
                anchor_meta = self._resolve_satellite_bbox(p)
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                ih, iw = img_rgb.shape[:2]
                if "lt_lat" in anchor_meta and ih > tile_size // 3 and iw > tile_size // 3:
                    for row_start in range(0, ih, step):
                        for col_start in range(0, iw, step):
                            row_end = min(ih, row_start + tile_size)
                            col_end = min(iw, col_start + tile_size)
                            th = row_end - row_start
                            tw = col_end - col_start
                            if th < tile_size // 3 or tw < tile_size // 3:
                                continue
                            tile = img_rgb[row_start:row_end, col_start:col_end]
                            if tile.shape[0] != tile_size or tile.shape[1] != tile_size:
                                padded = np.zeros((tile_size, tile_size, 3), dtype=np.uint8)
                                y_off = (tile_size - tile.shape[0]) // 2
                                x_off = (tile_size - tile.shape[1]) // 2
                                padded[y_off:y_off + tile.shape[0], x_off:x_off + tile.shape[1]] = tile
                                tile = padded
                            lat, lon, src_tag = _anchor_from_map_bbox(
                                anchor_meta, row_start, col_start, ih, iw, th, tw
                            )
                            if src_tag == "invalid":
                                continue
                            yield {
                                "ref_name": f"{loc}/{p.stem}_r{row_start:06d}_c{col_start:06d}.png",
                                "ref_image": tile,
                                "anchor": {
                                    "lat": lat,
                                    "lon": lon,
                                    "anchor_source": src_tag,
                                    "row_start": int(row_start),
                                    "col_start": int(col_start),
                                    "tile_size_px": int(tile_size),
                                    "tile_step_px": int(step),
                                },
                            }
                            loc_tiles_emitted += 1
                            total_corpus_tiles += 1
                    continue
                ref_entry = {
                    "ref_name": f"{loc}/{p.name}",
                    "ref_image": img_rgb,
                    "anchor": {
                        "lat": float(anchor_meta.get("lat", 0.0)),
                        "lon": float(anchor_meta.get("lon", 0.0)),
                        "anchor_source": "file_fallback",
                        "tile_size_px": int(tile_size),
                        "tile_step_px": int(step),
                    },
                }
                loc_tiles_emitted += 1
                total_corpus_tiles += 1
                yield ref_entry
            logger.info(
                "UAV-VisLoc loc=%s: emitted %d corpus tiles (tile_size=%d, step=%d)",
                loc, loc_tiles_emitted, tile_size, step,
            )
        logger.info(
            "UAV-VisLoc: total retrieval corpus = %d tiles across %d locations (protocol=%s).",
            total_corpus_tiles, len(corpus_locations), self._protocol,
        )

    def audit_reference_corpus_anchors(
        self,
        max_tiles_per_map: int = 12,
        neighbor_stride: int = 512,
    ) -> Dict[str, Any]:
        """
        Audit tiled retrieval corpus anchors for spatial consistency.

        Flags duplicate anchors across neighboring tiles and maps with no georef tiles.
        """
        if not self._loaded:
            self.load()

        def _approx_m_per_deg(lat: float, dlat: float, dlon: float) -> float:
            m_lat = dlat * 111_320.0
            m_lon = dlon * 111_320.0 * np.cos(np.radians(lat))
            return float(np.hypot(m_lat, m_lon))

        per_map: Dict[str, Any] = {}
        duplicate_anchor_groups = 0
        geotiff_tiles = 0
        bbox_tiles = 0
        fallback_tiles = 0
        invalid_skipped = 0
        total_tiles = 0
        bboxes_by_location: Dict[str, List[Dict[str, float]]] = {}

        for loc in self._locations:
            loc_dir = self._data_root / str(loc)
            sat_paths = sorted(
                list(loc_dir.glob("satellite*.tif"))
                + list(loc_dir.glob("satellite*.TIF"))
                + list((loc_dir / "satellite_map").glob("satellite*.tif"))
                + list((loc_dir / "satellite_map").glob("satellite*.TIF"))
            )
            for p in sat_paths:
                if p.suffix.lower() not in {".tif", ".tiff"}:
                    continue
                map_key = f"{loc}/{p.name}"
                samples: List[Dict[str, Any]] = []
                anchor_meta = self._resolve_satellite_bbox(p)
                if "lt_lat" in anchor_meta and "lt_lon" in anchor_meta:
                    bboxes_by_location.setdefault(str(loc), []).append(anchor_meta)
                rasterio = None
                try:
                    import rasterio as _rasterio
                    rasterio = _rasterio
                except Exception:
                    rasterio = None

                if rasterio is None and "lt_lat" in anchor_meta:
                    import cv2
                    img_bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
                    if img_bgr is None:
                        per_map[map_key] = {"error": "opencv_read_failed"}
                        continue
                    ih, iw = img_bgr.shape[:2]
                    step = neighbor_stride
                    count = 0
                    for row_start in range(0, ih, step):
                        for col_start in range(0, iw, step):
                            if count >= max_tiles_per_map:
                                break
                            row_end = min(ih, row_start + step)
                            col_end = min(iw, col_start + step)
                            th = row_end - row_start
                            tw = col_end - col_start
                            if th < step // 3 or tw < step // 3:
                                continue
                            lat, lon, src_tag = _anchor_from_map_bbox(
                                anchor_meta, row_start, col_start, ih, iw, th, tw
                            )
                            if src_tag == "invalid":
                                invalid_skipped += 1
                                continue
                            if src_tag == "bbox_interpolated":
                                bbox_tiles += 1
                            else:
                                fallback_tiles += 1
                            samples.append(
                                {
                                    "row_start": int(row_start),
                                    "col_start": int(col_start),
                                    "lat": lat,
                                    "lon": lon,
                                    "anchor_source": src_tag,
                                }
                            )
                            count += 1
                        if count >= max_tiles_per_map:
                            break
                    total_tiles += len(samples)
                    unique_anchors = {
                        (round(s["lat"], 8), round(s["lon"], 8)) for s in samples
                    }
                    if len(samples) >= 2 and len(unique_anchors) == 1:
                        duplicate_anchor_groups += 1
                    neighbor_deltas_m: List[float] = []
                    for i in range(len(samples) - 1):
                        a = samples[i]
                        b = samples[i + 1]
                        neighbor_deltas_m.append(
                            _approx_m_per_deg(
                                (a["lat"] + b["lat"]) * 0.5,
                                b["lat"] - a["lat"],
                                b["lon"] - a["lon"],
                            )
                        )
                    per_map[map_key] = {
                        "has_georef": False,
                        "anchor_mode": "bbox_interpolated",
                        "num_samples": len(samples),
                        "num_unique_anchors": len(unique_anchors),
                        "duplicate_anchor_group": len(samples) >= 2 and len(unique_anchors) == 1,
                        "neighbor_delta_m_mean": float(np.mean(neighbor_deltas_m)) if neighbor_deltas_m else None,
                        "neighbor_delta_m_min": float(np.min(neighbor_deltas_m)) if neighbor_deltas_m else None,
                        "samples": samples[:6],
                    }
                    continue

                if rasterio is None:
                    per_map[map_key] = {"error": "rasterio_unavailable_no_bbox"}
                    continue
                try:
                    with rasterio.open(str(p)) as src:
                        has_geo = bool(getattr(src, "crs", None))
                        step = neighbor_stride
                        count = 0
                        for row_start in range(0, int(src.height), step):
                            for col_start in range(0, int(src.width), step):
                                if count >= max_tiles_per_map:
                                    break
                                row_end = min(int(src.height), row_start + step)
                                col_end = min(int(src.width), col_start + step)
                                h = row_end - row_start
                                w = col_end - col_start
                                if h < step // 3 or w < step // 3:
                                    continue
                                lat, lon, src_tag = _tile_anchor_wgs84_from_raster(
                                    src,
                                    row_start,
                                    col_start,
                                    w,
                                    h,
                                    file_fallback=anchor_meta,
                                )
                                if src_tag == "invalid":
                                    invalid_skipped += 1
                                    continue
                                if src_tag == "geotiff_center":
                                    geotiff_tiles += 1
                                else:
                                    fallback_tiles += 1
                                samples.append(
                                    {
                                        "row_start": int(row_start),
                                        "col_start": int(col_start),
                                        "lat": lat,
                                        "lon": lon,
                                        "anchor_source": src_tag,
                                    }
                                )
                                count += 1
                            if count >= max_tiles_per_map:
                                break
                except Exception as exc:
                    per_map[map_key] = {"error": str(exc)}
                    continue

                total_tiles += len(samples)
                unique_anchors = {
                    (round(s["lat"], 8), round(s["lon"], 8)) for s in samples
                }
                if len(samples) >= 2 and len(unique_anchors) == 1:
                    duplicate_anchor_groups += 1

                neighbor_deltas_m: List[float] = []
                for i in range(len(samples) - 1):
                    a = samples[i]
                    b = samples[i + 1]
                    d_m = _approx_m_per_deg(
                        (a["lat"] + b["lat"]) * 0.5,
                        b["lat"] - a["lat"],
                        b["lon"] - a["lon"],
                    )
                    neighbor_deltas_m.append(d_m)

                per_map[map_key] = {
                    "has_georef": bool(has_geo),
                    "num_samples": len(samples),
                    "num_unique_anchors": len(unique_anchors),
                    "duplicate_anchor_group": len(samples) >= 2 and len(unique_anchors) == 1,
                    "neighbor_delta_m_mean": float(np.mean(neighbor_deltas_m)) if neighbor_deltas_m else None,
                    "neighbor_delta_m_min": float(np.min(neighbor_deltas_m)) if neighbor_deltas_m else None,
                    "samples": samples[:6],
                }

        emitted: List[Dict[str, Any]] = []
        per_map_emitted: Dict[str, List[Dict[str, float]]] = {}
        max_corpus_audit_tiles = 500
        for ref in self.iter_reference_corpus():
            anchor = ref.get("anchor", {}) or {}
            lat = float(anchor.get("lat", 0.0))
            lon = float(anchor.get("lon", 0.0))
            ref_name = str(ref.get("ref_name", ""))
            map_key = ref_name.rsplit("/", 1)[0] if "/" in ref_name else ref_name
            row = {
                "ref_name": ref_name,
                "lat": lat,
                "lon": lon,
                "anchor_source": str(anchor.get("anchor_source", "unknown")),
            }
            emitted.append(row)
            per_map_emitted.setdefault(map_key, []).append({"lat": lat, "lon": lon})
            if len(emitted) >= max_corpus_audit_tiles:
                break

        emitted_dup_groups = 0
        emitted_per_map_summary: Dict[str, Any] = {}
        for map_key, anchors in per_map_emitted.items():
            uniq = {(round(a["lat"], 8), round(a["lon"], 8)) for a in anchors}
            is_dup = len(anchors) >= 2 and len(uniq) == 1
            if is_dup:
                emitted_dup_groups += 1
            emitted_per_map_summary[map_key] = {
                "num_tiles": len(anchors),
                "num_unique_anchors": len(uniq),
                "duplicate_anchor_group": is_dup,
            }

        valid_anchor_sources = {"geotiff_center", "bbox_interpolated"}
        has_spatial_anchors = any(
            s.get("anchor_source") in valid_anchor_sources for s in emitted[:500]
        )
        emitted_anchor_items = []
        for row in emitted:
            lat = float(row.get("lat", np.nan))
            lon = float(row.get("lon", np.nan))
            if np.isfinite(lat) and np.isfinite(lon):
                flight = str(row.get("ref_name", "")).split("/", 1)[0]
                emitted_anchor_items.append((flight, lat, lon))

        def _bbox_contains(bbox: Dict[str, float], lat: float, lon: float) -> bool:
            min_lat, max_lat = sorted([float(bbox.get("lt_lat", 0.0)), float(bbox.get("rb_lat", 0.0))])
            min_lon, max_lon = sorted([float(bbox.get("lt_lon", 0.0)), float(bbox.get("rb_lon", 0.0))])
            return bool(
                np.isfinite(lat)
                and np.isfinite(lon)
                and min_lat <= lat <= max_lat
                and min_lon <= lon <= max_lon
            )

        def _distance_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
            m_lat = (lat1 - lat2) * 111_320.0
            m_lon = (lon1 - lon2) * 111_320.0 * np.cos(np.radians((lat1 + lat2) * 0.5))
            return float(np.hypot(m_lat, m_lon))

        def _summary(values: List[float]) -> Dict[str, Any]:
            vals = np.asarray([float(v) for v in values if np.isfinite(float(v))], dtype=np.float64)
            if len(vals) == 0:
                return {"count": 0, "p50": None, "p90": None, "mean": None, "max": None}
            return {
                "count": int(len(vals)),
                "p50": float(np.percentile(vals, 50)),
                "p90": float(np.percentile(vals, 90)),
                "mean": float(np.mean(vals)),
                "max": float(np.max(vals)),
            }

        selected_indices = list(range(0, len(self._entries), self._test_interval))
        if self._max_samples_percent:
            cutoff = int(len(selected_indices) * (self._max_samples_percent / 100.0))
            selected_indices = selected_indices[:cutoff]
        if self._max_samples:
            selected_indices = selected_indices[:self._max_samples]

        query_bbox_checked_count = 0
        query_out_of_bbox_count = 0
        query_out_of_bbox_examples: List[Dict[str, Any]] = []
        nearest_tile_floor_values: List[float] = []
        for idx in selected_indices:
            entry = self._entries[idx]
            flight = str((entry.get("metadata") or {}).get("location", ""))
            gt = entry.get("gt_pose", {}) or {}
            q_lat = float(gt.get("lat", np.nan))
            q_lon = float(gt.get("lon", np.nan))
            loc_bboxes = bboxes_by_location.get(flight, [])
            if loc_bboxes and np.isfinite(q_lat) and np.isfinite(q_lon):
                query_bbox_checked_count += 1
                if not any(_bbox_contains(bbox, q_lat, q_lon) for bbox in loc_bboxes):
                    query_out_of_bbox_count += 1
                    if len(query_out_of_bbox_examples) < 10:
                        query_out_of_bbox_examples.append({
                            "image_name": str(entry.get("image_name", "")),
                            "flight_id": flight,
                            "lat": q_lat,
                            "lon": q_lon,
                        })
            if emitted_anchor_items and np.isfinite(q_lat) and np.isfinite(q_lon):
                same_flight = [(lat, lon) for item_flight, lat, lon in emitted_anchor_items if item_flight == flight]
                candidates = same_flight if same_flight else [(lat, lon) for _item_flight, lat, lon in emitted_anchor_items]
                nearest_tile_floor_values.append(
                    min(_distance_m(q_lat, q_lon, lat, lon) for lat, lon in candidates)
                )

        passed = (
            duplicate_anchor_groups == 0
            and emitted_dup_groups == 0
            and len(emitted) > 0
            and has_spatial_anchors
            and query_out_of_bbox_count == 0
            and not any(
                abs(s.get("lat", 0.0)) < 1e-6 and abs(s.get("lon", 0.0)) < 1e-6
                for s in emitted[: min(50, len(emitted))]
            )
        )
        return {
            "dataset": "uav_visloc",
            "passed": passed,
            "total_sampled_tiles": total_tiles,
            "geotiff_center_tiles": geotiff_tiles,
            "bbox_interpolated_tiles": bbox_tiles,
            "file_fallback_tiles": fallback_tiles,
            "invalid_skipped": invalid_skipped,
            "duplicate_anchor_groups": duplicate_anchor_groups,
            "per_map": per_map,
            "emitted_corpus_tiles": len(emitted),
            "emitted_duplicate_groups": emitted_dup_groups,
            "emitted_per_map": emitted_per_map_summary,
            "emitted_samples": emitted[:12],
            "query_bbox_checked_count": int(query_bbox_checked_count),
            "query_out_of_bbox_count": int(query_out_of_bbox_count),
            "query_out_of_bbox_examples": query_out_of_bbox_examples,
            "nearest_tile_oracle_floor_m_summary": _summary(nearest_tile_floor_values),
        }


# ============================================================================
# DATASET FACTORY
# ============================================================================

class DatasetFactory:
    """Factory for creating dataset adapters."""
    
    DATASETS = {
        'vpair': VPAirDataset,
        'avl': AVLDataset,
        'avl24': AVL24Dataset,
        'uav_visloc': UAVVisLocDataset,
    }

    # Default data root paths
    DEFAULT_ROOTS = {
        'vpair': './data/VPAir',
        'avl': './data/AVLData',
        'avl24': './data/AnyVisLoc',
        'uav_visloc': './data/UAV_VisLoc_dataset',
    }
    
    @classmethod
    def create(cls,
               dataset_name: str,
               data_root: Path = None,
               test_interval: int = 1,
               max_samples: int = None,
               max_samples_percent: int = None,
               **kwargs) -> BaseDataset:
        """
        Create dataset adapter.
        
        Args:
            dataset_name: 'vpair', 'avl', 'uav_visloc'
            data_root: Override data root path
            test_interval: Sample every N-th frame
            max_samples: Max samples per dataset
            max_samples_percent: Percentage of dataset to use
            **kwargs: Dataset-specific arguments
            
        Returns:
            Configured BaseDataset instance
        """
        if dataset_name not in cls.DATASETS:
            raise ValueError(
                f"Unknown dataset: {dataset_name}. "
                f"Available: {list(cls.DATASETS.keys())}"
            )
        
        if data_root is None:
            data_root = Path(cls.DEFAULT_ROOTS.get(dataset_name, '.'))
        
        dataset_cls = cls.DATASETS[dataset_name]
        return dataset_cls(
            data_root=data_root,
            test_interval=test_interval,
            max_samples=max_samples,
            max_samples_percent=max_samples_percent,
            **kwargs
        )
    
    @classmethod
    def list_available(cls) -> List[str]:
        """List available dataset names."""
        return list(cls.DATASETS.keys())


# --- AnyVisLoc scene format (2026-08 re-release) ------------------------------
# The module self-registers 'avl_scene'; nothing else here changes, so deleting
# this import reverts it. See src/data/avl_scene_adapter.py.
from . import avl_scene_adapter  # noqa: E402,F401
