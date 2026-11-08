# Explicit sanitized source generated from _raw notebook export.
# Notebook bootstrap/magic cells were removed intentionally for local execution.

# ===== Code Cell 4 =====
import os
import math
import csv
from pathlib import Path

import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch

print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", DEVICE)


# ===== Code Cell 5 =====
# ---- Dataset paths (edit as needed) ----
CANDIDATE_ROOTS = globals().get("CANDIDATE_ROOTS", [
    # Server/local override first (e.g. /workspace/forge/data/VPAir/VPAir)
    *([Path(os.environ["VPAIR_DATA"])] if os.environ.get("VPAIR_DATA") else []),
    Path("/kaggle/input/datasets/anonymous/vpair-full"),
    Path("/kaggle/input/vpair-full/VPAIR-full"),
    Path("/kaggle/input/VPAIR-full/VPAIR-full"),
    Path("/kaggle/input/vpair-full"),
    Path("/kaggle/input/VPAIR-full"),
    Path("../vpair_sample"),
])
VPAIR_ROOT = next((p for p in CANDIDATE_ROOTS if p.exists()), CANDIDATE_ROOTS[0])
print(f"Using VPAIR_ROOT: {VPAIR_ROOT}")

QUERY_DIR      = VPAIR_ROOT / "queries"
REF_DIR        = VPAIR_ROOT / "reference_views"
DISTRACTOR_DIR = VPAIR_ROOT / "distractors"
POSE_CSV       = VPAIR_ROOT / "poses.csv"
POSE_QUERY     = VPAIR_ROOT / "poses_query.txt"
POSE_REF       = VPAIR_ROOT / "poses_reference_view.txt"
CALIB_YAML     = VPAIR_ROOT / "camera_calibration.yaml"

MAX_IMAGES = globals().get("MAX_IMAGES", None)  # set None for full
GSD_REF_MPX = globals().get("GSD_REF_MPX", 0.42)  # measured; 0.3 under-scaled offsets 1.4x
FOCAL_PX_Q = globals().get("FOCAL_PX_Q", 550.0)  # fallback if calibration file unavailable
PATCH_MIN = 320
PATCH_MAX = 1400

for p in [QUERY_DIR, REF_DIR, DISTRACTOR_DIR, POSE_CSV, POSE_QUERY, POSE_REF, CALIB_YAML]:
    status = "ok" if p.exists() else "missing"
    print(f"{status:>7}  {p}")


def parse_poses_txt(filepath: Path) -> dict:
    poses = {}
    with open(filepath, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            stem = Path(row["filepath"]).stem
            poses[stem] = {
                "ecef": np.array([float(row["x"]), float(row["y"]), float(row["z"])]),
                "rpy": np.array([float(row["roll"]), float(row["pitch"]), float(row["yaw"])]),
                "filepath": row["filepath"],
            }
    return poses


def _pick_col(row: dict, names: list[str]):
    lower_map = {k.lower(): k for k in row.keys()}
    for n in names:
        if n.lower() in lower_map:
            return row[lower_map[n.lower()]]
    return None


def _to_float_or_none(v):
    try:
        return float(v)
    except Exception:
        return None


def geodetic_to_ecef_simple(lat: float, lon: float, alt: float) -> np.ndarray:
    a = 6378137.0
    b = 6356752.314245
    e2 = 1.0 - (b * b) / (a * a)
    lat_r = np.radians(lat)
    lon_r = np.radians(lon)
    N = a / np.sqrt(1 - e2 * np.sin(lat_r) ** 2)
    return np.array([
        (N + alt) * np.cos(lat_r) * np.cos(lon_r),
        (N + alt) * np.cos(lat_r) * np.sin(lon_r),
        (N * (1 - e2) + alt) * np.sin(lat_r),
    ], dtype=np.float64)


def parse_poses_csv(filepath: Path):
    query_poses, ref_poses = {}, {}
    n_rows = 0
    n_valid = 0

    with open(filepath, newline="") as f:
        reader = csv.DictReader(f)
        print("poses.csv columns:", reader.fieldnames)

        for row in reader:
            n_rows += 1

            path_raw = _pick_col(row, ["filepath", "path", "image", "img", "filename", "file"])
            if path_raw is None:
                continue
            stem = Path(path_raw).stem

            # Try ECEF first
            x = _to_float_or_none(_pick_col(row, ["x", "ecef_x", "tx"]))
            y = _to_float_or_none(_pick_col(row, ["y", "ecef_y", "ty"]))
            z = _to_float_or_none(_pick_col(row, ["z", "ecef_z", "tz"]))

            # Fallback: lat/lon/alt -> ECEF
            if any(v is None for v in [x, y, z]):
                lat = _to_float_or_none(_pick_col(row, ["lat", "latitude"]))
                lon = _to_float_or_none(_pick_col(row, ["lon", "lng", "longitude"]))
                alt = _to_float_or_none(_pick_col(row, ["alt", "altitude", "height", "z"]))
                if None not in [lat, lon]:
                    if alt is None:
                        alt = 0.0
                    ecef = geodetic_to_ecef_simple(lat, lon, alt)
                    x, y, z = float(ecef[0]), float(ecef[1]), float(ecef[2])

            # Try rpy in degrees; if missing keep zeros (so pose can still be parsed)
            r = _to_float_or_none(_pick_col(row, ["roll", "r"]))
            p = _to_float_or_none(_pick_col(row, ["pitch", "p"]))
            yw = _to_float_or_none(_pick_col(row, ["yaw", "y", "heading"]))
            if r is None:
                r = 0.0
            if p is None:
                p = 0.0
            if yw is None:
                yw = 0.0

            if any(v is None for v in [x, y, z]):
                continue

            entry = {
                "ecef": np.array([x, y, z], dtype=np.float64),
                "rpy": np.array([r, p, yw], dtype=np.float64),
                "filepath": path_raw,
            }
            n_valid += 1

            split_val = _pick_col(row, ["split", "set", "type", "subset", "group"])
            split = (split_val or "").lower()
            path_low = path_raw.lower()
            is_query = ("query" in split) or ("queries" in path_low)
            is_ref = ("ref" in split) or ("reference" in split) or ("reference_views" in path_low)

            if is_query and not is_ref:
                query_poses[stem] = entry
            elif is_ref and not is_query:
                ref_poses[stem] = entry
            else:
                query_poses[stem] = entry

    print(f"poses.csv rows={n_rows}, valid_pose_rows={n_valid}")
    return query_poses, ref_poses


def load_focal_from_yaml(yaml_path: Path, fallback: float) -> float:
    """Read the query focal length in pixels from the dataset calibration.

    A silently substituted focal length rescales every PnP solution, so this
    fails loudly instead of falling back. Set ALLOW_FOCAL_FALLBACK=1 only for
    a deliberate sensitivity probe, never for reported numbers.
    """
    allow_fallback = str(os.environ.get("ALLOW_FOCAL_FALLBACK", "")).strip() in ("1", "true", "yes")

    def _fail(reason: str) -> float:
        msg = (f"VPAIR focal length unavailable ({reason}); refusing the {fallback} px "
               f"fallback because it would silently rescale every pose. "
               f"Checked: {yaml_path}")
        if allow_fallback:
            print(f"WARNING: {msg} -- proceeding on ALLOW_FOCAL_FALLBACK", flush=True)
            return fallback
        raise RuntimeError(msg)

    if not yaml_path.exists():
        return _fail("calibration file missing")
    try:
        import yaml
        with open(yaml_path, "r") as f:
            data = yaml.safe_load(f)
        for key in ["fx", "focal_px", "focal_length_px", "focal_length"]:
            if isinstance(data, dict) and key in data:
                return float(data[key])
        if isinstance(data, dict):
            # VPAIR ships a Kalibr-style file: cam0.intrinsics = [fx, fy, cx, cy].
            # The previous lookup never matched this shape, so every run silently
            # used the 550 px fallback against a true fx of 750.63 px (27% off),
            # rescaling every PnP translation by ~1.36x.
            for cam_key in ("cam0", "cam1"):
                cam_block = data.get(cam_key)
                if isinstance(cam_block, dict):
                    intr = cam_block.get("intrinsics")
                    if isinstance(intr, (list, tuple)) and len(intr) >= 1:
                        return float(intr[0])
            cam = data.get("camera") or data.get("intrinsics")
            if isinstance(cam, (list, tuple)) and len(cam) >= 1:
                return float(cam[0])
            if isinstance(cam, dict):
                for key in ["fx", "focal_px", "focal_length_px", "focal_length"]:
                    if key in cam:
                        return float(cam[key])
                K = cam.get("K")
                if isinstance(K, list) and len(K) >= 1:
                    return float(K[0])
        return _fail(f"no focal key found; top-level keys = {sorted(data) if isinstance(data, dict) else type(data).__name__}")
    except RuntimeError:
        raise
    except Exception as exc:
        return _fail(f"parse error: {type(exc).__name__}: {exc}")


def ecef_to_lla(ecef: np.ndarray):
    x, y, z = ecef.astype(np.float64)
    a = 6378137.0
    b = 6356752.314245
    e2 = 1.0 - (b * b) / (a * a)

    lon = np.arctan2(y, x)
    p = np.sqrt(x * x + y * y)
    lat = np.arctan2(z, p * (1.0 - e2))

    for _ in range(8):
        sin_lat = np.sin(lat)
        N = a / np.sqrt(1.0 - e2 * sin_lat * sin_lat)
        lat_new = np.arctan2(z + e2 * N * sin_lat, p)
        if abs(lat_new - lat) < 1e-12:
            lat = lat_new
            break
        lat = lat_new

    sin_lat = np.sin(lat)
    N = a / np.sqrt(1.0 - e2 * sin_lat * sin_lat)
    alt = p / np.cos(lat) - N

    return float(np.degrees(lat)), float(np.degrees(lon)), float(alt)


def build_enu_rotation(origin_ecef: np.ndarray):
    ox, oy, oz = origin_ecef
    a = 6378137.0
    e2 = 6.69437999014e-3
    lon0 = math.atan2(oy, ox)
    p0 = math.sqrt(ox**2 + oy**2)
    lat0 = math.atan2(oz, p0 * (1 - e2))

    for _ in range(5):
        N = a / math.sqrt(1 - e2 * math.sin(lat0)**2)
        lat0 = math.atan2(oz + e2 * N * math.sin(lat0), p0)

    slat, clat = math.sin(lat0), math.cos(lat0)
    slon, clon = math.sin(lon0), math.cos(lon0)

    R = np.array([
        [-slon,        clon,       0.0],
        [-slat*clon,  -slat*slon,  clat],
        [ clat*clon,   clat*slon,  slat],
    ])
    return R


def ecef_to_enu(ecef: np.ndarray, origin_ecef: np.ndarray, R_enu: np.ndarray) -> np.ndarray:
    delta = ecef - origin_ecef
    return R_enu @ delta


def enu_to_ecef(enu: np.ndarray, origin_ecef: np.ndarray, R_enu: np.ndarray) -> np.ndarray:
    return origin_ecef + (R_enu.T @ enu)


def enu_to_lla(enu: np.ndarray, origin_ecef: np.ndarray, R_enu: np.ndarray):
    return ecef_to_lla(enu_to_ecef(enu, origin_ecef, R_enu))


if POSE_CSV.exists():
    query_poses, ref_poses = parse_poses_csv(POSE_CSV)
else:
    query_poses = parse_poses_txt(POSE_QUERY)
    ref_poses = parse_poses_txt(POSE_REF)

if len(ref_poses) == 0 and len(query_poses) > 0:
    ref_poses = {k: v for k, v in query_poses.items()}

if len(query_poses) == 0 and len(ref_poses) == 0:
    raise ValueError(
        "No valid poses parsed from dataset metadata. "
        "Check poses.csv column names and units."
    )

if len(ref_poses) > 0:
    origin_ecef = next(iter(ref_poses.values()))["ecef"].copy()
else:
    origin_ecef = next(iter(query_poses.values()))["ecef"].copy()

R_ENU = build_enu_rotation(origin_ecef)

for d in list(query_poses.values()) + list(ref_poses.values()):
    d["enu"] = ecef_to_enu(d["ecef"], origin_ecef, R_ENU)

FOCAL_PX_Q = load_focal_from_yaml(CALIB_YAML, FOCAL_PX_Q)
print(f"Using focal px: {FOCAL_PX_Q:.2f}")
print(f"Query poses: {len(query_poses)}")
print(f"Ref poses  : {len(ref_poses)}")


# ===== Code Cell 6 =====
def list_images(folder: Path):
    exts = ["*.png", "*.jpg", "*.jpeg", "*.PNG", "*.JPG", "*.JPEG"]
    paths = []
    for ext in exts:
        paths.extend(folder.glob(ext))
    return sorted(set(paths))


ref_image_paths = {p.stem: p for p in list_images(REF_DIR)}
ref_depth_paths = {}
for p in sorted(REF_DIR.glob("depth_*.npy")):
    stem = p.stem.replace("depth_", "")
    ref_depth_paths[stem] = p

# Extra fallback: allow same-stem depth files without "depth_" prefix.
for p in sorted(REF_DIR.glob("*.npy")):
    if p.stem not in ref_depth_paths and not p.stem.startswith("depth_"):
        ref_depth_paths[p.stem] = p

query_image_paths = {p.stem: p for p in list_images(QUERY_DIR)}

query_stems = sorted(query_image_paths.keys())
if MAX_IMAGES is not None:
    query_stems = query_stems[:MAX_IMAGES]

print(f"Queries active : {len(query_stems)}")
print(f"Refs available : {len(ref_image_paths)}")
print(f"Depth maps     : {len(ref_depth_paths)}")


# ===== Code Cell 7 =====
# MatcherRoMa is OrthoLoC's own direct RoMa matcher (same base Matcher.run
# interface). The MatcherIMCUI wrapper needs the heavy `imcui` stack, which
# is not installable in the pinned benchmark env; with MATCHER_NAME="RoMa"
# the two paths run the same underlying model.
from ortholoc.image_matching.MatcherRoMa import MatcherRoMa
from ortholoc.correspondences import Correspondences2D2D

MATCHER_NAME = "RoMa"
MIN_CONF = 0.5
REPROJECTION_ERROR = 5.0

print(f"Loading matcher: {MATCHER_NAME} ...")
matcher = MatcherRoMa(device=DEVICE, angles=[0])
print("Matcher loaded")


def grid3d_from_depth(depth: np.ndarray, gsd: float) -> np.ndarray:
    h, w = depth.shape
    cx, cy = w / 2.0, h / 2.0
    xs = (np.arange(w) - cx) * gsd
    ys = (np.arange(h) - cy) * gsd
    X, Y = np.meshgrid(xs, ys)
    Z = depth.astype(np.float32)
    return np.stack([X, Y, Z], axis=-1).astype(np.float32)


def yaw_from_rotmat(R: np.ndarray) -> float:
    yaw_rad = np.arctan2(R[1, 0], R[0, 0])
    return float(np.degrees(yaw_rad) % 360.0)


def compute_patch_wh_from_query(query_shape, depth, focal_px, gsd_ref, min_px, max_px):
    hq, wq = query_shape[:2]
    valid = depth[np.isfinite(depth) & (depth > 0)]
    if valid.size == 0 or focal_px <= 0 or gsd_ref <= 0:
        px = min_px
    else:
        z_med = float(np.median(valid))
        view_m = (min(hq, wq) * z_med) / float(focal_px)
        px = int(np.ceil(view_m / gsd_ref))
    px = int(np.clip(px, min_px, max_px))
    if px % 2 == 0:
        px += 1
    return px, px


def center_crop_img_depth(img, depth, crop_w, crop_h):
    h, w = img.shape[:2]
    cw = min(int(crop_w), w)
    ch = min(int(crop_h), h)
    x0 = max(0, (w - cw) // 2)
    y0 = max(0, (h - ch) // 2)
    x1 = x0 + cw
    y1 = y0 + ch
    return img[y0:y1, x0:x1], depth[y0:y1, x0:x1]


def resolve_ref_stem(stem: str):
    if stem in ref_image_paths and stem in ref_depth_paths:
        return stem

    q_path = query_poses.get(stem, {}).get("filepath", "")
    if q_path:
        candidate = Path(q_path).stem
        if candidate in ref_image_paths and candidate in ref_depth_paths:
            return candidate
        candidate = Path(q_path).name
        candidate = Path(candidate).stem
        if candidate in ref_image_paths and candidate in ref_depth_paths:
            return candidate

    return None


results = []
stats = {"missing_ref_or_depth": 0, "missing_query_pose": 0}

for stem in tqdm(query_stems, desc="OrthoLoC VPAir"):
    ref_stem = resolve_ref_stem(stem)
    if ref_stem is None:
        stats["missing_ref_or_depth"] += 1
        continue

    if stem not in query_poses:
        stats["missing_query_pose"] += 1
        results.append({"stem": stem, "success": False, "reason": "missing_query_pose"})
        continue

    query = cv2.imread(str(query_image_paths[stem]))
    ref_img_full = cv2.imread(str(ref_image_paths[ref_stem]))
    depth_full = np.load(str(ref_depth_paths[ref_stem]))

    patch_w, patch_h = compute_patch_wh_from_query(
        query.shape,
        depth_full,
        FOCAL_PX_Q,
        GSD_REF_MPX,
        PATCH_MIN,
        PATCH_MAX,
    )
    ref_img, depth = center_crop_img_depth(ref_img_full, depth_full, patch_w, patch_h)

    h0, w0 = query.shape[:2]
    h1, w1 = ref_img.shape[:2]

    camera_K = np.array([
        [FOCAL_PX_Q, 0.0, w0 / 2.0],
        [0.0, FOCAL_PX_Q, h0 / 2.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)

    grid3d = grid3d_from_depth(depth, GSD_REF_MPX)

    q_enu = query_poses[stem]["enu"]
    q_lat, q_lon, q_alt = ecef_to_lla(query_poses[stem]["ecef"])
    r_enu = ref_poses[ref_stem]["enu"] if ref_stem in ref_poses else None

    rec = {
        "stem": stem,
        "ref_stem": ref_stem,
        "gt_lat": float(q_lat),
        "gt_lon": float(q_lon),
        "gt_alt": float(q_alt),
        "gt_e": float(q_enu[0]),
        "gt_n": float(q_enu[1]),
        "gt_u": float(q_enu[2]),
        "pred_lat": None,
        "pred_lon": None,
        "pred_alt": None,
        "pred_e": None,
        "pred_n": None,
        "pred_u": None,
        "yaw_gt_deg": float(np.degrees(query_poses[stem]["rpy"][2]) % 360.0),
        "height_gt_m": float(np.median(depth[np.isfinite(depth)])) if np.isfinite(depth).any() else None,
        "yaw_est_deg": None,
        "height_est_m": None,
        "yaw_err_deg": None,
        "height_err_m": None,
        "pos_err_2d_m": None,
        "pos_err_3d_m": None,
        "reproj_error_px": None,
        "patch_w": int(w1),
        "patch_h": int(h1),
        "n_inliers": 0,
        "success": False,
    }

    try:
        corr_list = matcher.run(
            img0=query,
            img1=ref_img,
            angles=[0],
            normalized=True,
            silent=True,
        )
    except Exception:
        results.append(rec)
        continue

    corr_2d2d = corr_list[0]
    corr_2d2d = corr_2d2d.take_min_conf(min_conf=MIN_CONF, inclusive=False)
    corr_2d2d = corr_2d2d.take_covisible(h0=h0, w0=w0, h1=h1, w1=w1, is_normalized=True)
    if not corr_2d2d.is_valid:
        results.append(rec)
        continue

    corr_2d3d = corr_2d2d.to_2d3d(grid3d_1=grid3d)
    if not corr_2d3d.is_valid:
        results.append(rec)
        continue

    success, pose_c2w_pred, intr_pred, inliers_mask, opt_reproj_errors = corr_2d3d.calibrate(
        num_points=None,
        intrinsics_matrix=camera_K,
        width=w0,
        height=h0,
        reprojection_error=REPROJECTION_ERROR,
        pnp_mode="poselib",
    )

    if success and pose_c2w_pred is not None:
        rec["success"] = True
        rec["n_inliers"] = int(inliers_mask.sum()) if inliers_mask is not None else 0
        rec["yaw_est_deg"] = yaw_from_rotmat(pose_c2w_pred[:3, :3])
        rec["height_est_m"] = float(pose_c2w_pred[2, 3])

        d = abs(rec["yaw_est_deg"] - rec["yaw_gt_deg"]) % 360.0
        rec["yaw_err_deg"] = min(d, 360.0 - d)
        if rec["height_gt_m"] is not None:
            rec["height_err_m"] = abs(rec["height_est_m"] - rec["height_gt_m"])
        if opt_reproj_errors is not None:
            rec["reproj_error_px"] = float(np.nanmedian(opt_reproj_errors))

        if r_enu is not None:
            pred_enu = np.array([
                r_enu[0] + pose_c2w_pred[0, 3],
                r_enu[1] - pose_c2w_pred[1, 3],
                r_enu[2] + pose_c2w_pred[2, 3],
            ], dtype=np.float64)
            rec["pred_e"] = float(pred_enu[0])
            rec["pred_n"] = float(pred_enu[1])
            rec["pred_u"] = float(pred_enu[2])

            p_lat, p_lon, p_alt = enu_to_lla(pred_enu, origin_ecef, R_ENU)
            rec["pred_lat"] = float(p_lat)
            rec["pred_lon"] = float(p_lon)
            rec["pred_alt"] = float(p_alt)

            rec["pos_err_2d_m"] = float(np.linalg.norm(pred_enu[:2] - q_enu[:2]))
            rec["pos_err_3d_m"] = float(np.linalg.norm(pred_enu - q_enu))

    results.append(rec)

print(f"Solved: {sum(r['success'] for r in results)} / {len(results)}")
print("Skip stats:", stats)


# ===== Code Cell 8 =====
import pandas as pd
from pathlib import Path

ACC_THRESHOLDS = list(range(5, 101, 5))

def _print_acc_table(arr: np.ndarray, unit: str):
    if len(arr) == 0:
        print("  Accuracy: unavailable")
        return
    rows = [f"A@{t}{unit}={100*np.mean(arr <= t):.1f}%" for t in ACC_THRESHOLDS]
    for i in range(0, len(rows), 5):
        print("  " + "  ".join(rows[i:i+5]))

if not results:
    print("No results. Check paths and dataset availability.")
else:
    solved = [r for r in results if r["success"]]
    if not solved:
        print("No successful PnP solutions.")
    else:
        yaw_errs = np.array([r["yaw_err_deg"] for r in solved if r["yaw_err_deg"] is not None])
        h_errs = np.array([r["height_err_m"] for r in solved if r["height_err_m"] is not None])
        pos_err_2d = np.array([r["pos_err_2d_m"] for r in solved if r["pos_err_2d_m"] is not None])

        reproj_errs = np.array([r["reproj_error_px"] for r in solved if r["reproj_error_px"] is not None])
        pos_err_3d = np.array([r["pos_err_3d_m"] for r in solved if r["pos_err_3d_m"] is not None])
        patch_ws = np.array([r["patch_w"] for r in solved if r.get("patch_w") is not None])

        print("=" * 72)
        print("  OrthoLoC VPAir Unified Metrics")
        print("=" * 72)
        print(f"Solved: {len(solved)} / {len(results)}")

        print("\n[Position Error 2D]")
        if len(pos_err_2d) > 0:
            print(f"  mean={np.mean(pos_err_2d):.2f} m  median={np.median(pos_err_2d):.2f} m")
        _print_acc_table(pos_err_2d, "m")

        print("\n[Yaw Error]")
        if len(yaw_errs) > 0:
            print(f"  mean={np.mean(yaw_errs):.2f} deg  median={np.median(yaw_errs):.2f} deg")
        _print_acc_table(yaw_errs, "deg")

        print("\n[Height Error]")
        if len(h_errs) > 0:
            print(f"  mean={np.mean(h_errs):.2f} m  median={np.median(h_errs):.2f} m")
        _print_acc_table(h_errs, "m")

        if len(pos_err_3d) > 0:
            print(f"\n[Position Error 3D]  mean={np.mean(pos_err_3d):.2f} m  median={np.median(pos_err_3d):.2f} m")
        if len(reproj_errs) > 0:
            print(f"[Reprojection]  mean={np.mean(reproj_errs):.2f} px  median={np.median(reproj_errs):.2f} px")
        if len(patch_ws) > 0:
            print(f"[Patch width]  mean={np.mean(patch_ws):.1f} px  min={np.min(patch_ws)}  max={np.max(patch_ws)}")

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        axes[0].hist(yaw_errs, bins=30, color="steelblue", edgecolor="white")
        axes[0].set_title("Yaw Error (deg)")
        axes[1].hist(h_errs, bins=30, color="seagreen", edgecolor="white")
        axes[1].set_title("Height Error (m)")
        axes[2].hist(reproj_errs, bins=30, color="teal", edgecolor="white")
        axes[2].set_title("Reprojection Error (px)")
        plt.tight_layout()
        plt.show()

    out_dir = Path(os.environ.get("ORTHOLOC_OUT_DIR", "")) if os.environ.get("ORTHOLOC_OUT_DIR") else (
        Path("/kaggle/working/FORGE-UAV/ortholoc_results")
        if Path("/kaggle/working").exists()
        else Path.cwd() / "ortholoc_results"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "vpair_ortholoc_results.csv"
    pd.DataFrame(results).to_csv(csv_path, index=False)
    print(f"Saved CSV: {csv_path}")
