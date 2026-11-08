"""
Benchmark utilities for the native AnyVisLoc baseline adapter.

Adapts the key functions from Benchmark/utils.py
(Match2Pos_all, retrieval_all, estimate_drone_pose, computeCameraMatrix, …)
to be importable without the Benchmark folder on sys.path.

Retrieval_Models/ and Matching_Models/ are included directly in the
FORGE-UAV repo root – no separate Benchmark clone is required.
"""

from __future__ import annotations
import glob
import sys
import json
import math
import os
import pickle
import shutil
import time
from math import sqrt
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from pyproj import Transformer


# ─────────────────────────────────────────────────────────────────────────────
#  I/O helpers
# ─────────────────────────────────────────────────────────────────────────────

def img_name(sensing_path: str) -> str:
    return os.path.splitext(os.path.basename(sensing_path))[0]


def load_data(save_path: str, *args):
    with open(save_path, "rb") as f:
        data = pickle.load(f)
    return tuple(data[k] for k in args if k in data)


def save_data(filename: str, **kwargs):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "wb") as f:
        pickle.dump(kwargs, f)


def get_jpg_files(folder_path: str) -> List[str]:
    if not os.path.isdir(folder_path):
        return []
    # Match both .JPG (Windows) and .jpg (Linux/Kaggle)
    files = glob.glob(os.path.join(folder_path, "*.JPG")) + \
            glob.glob(os.path.join(folder_path, "*.jpg")) + \
            glob.glob(os.path.join(folder_path, "*.jpeg")) + \
            glob.glob(os.path.join(folder_path, "*.JPEG"))
    return sorted(set(files))


def query_data_from_file(file_name: str, **query) -> List[Dict]:
    if not os.path.exists(file_name):
        return []
    with open(file_name, "r") as f:
        data = json.load(f)
    return [e for e in data if all(e.get(k) == v for k, v in query.items())]


def read_data_from_file(file_name: str) -> List[Dict]:
    if not os.path.exists(file_name):
        return []
    with open(file_name, "r") as f:
        return json.load(f)


def find_values(config: dict, search_string: str):
    for k, v in config.items():
        if search_string in k:
            return v
    return "None"


# ─────────────────────────────────────────────────────────────────────────────
#  Coordinate helpers
# ─────────────────────────────────────────────────────────────────────────────

def deg2utm(region: str, config: dict,
            lon: float, lat: float) -> Tuple[float, float]:
    utm_system = config[f"{region}_UTM_SYSTEM"]
    trf = Transformer.from_crs("epsg:4326", f"epsg:326{utm_system[:2]}", always_xy=True)
    return trf.transform(lon, lat)          # → (x=easting, y=northing)


def utm2deg(region: str, config: dict,
            x: float, y: float) -> Tuple[float, float]:
    utm_system = config[f"{region}_UTM_SYSTEM"]
    trf = Transformer.from_crs(f"epsg:326{utm_system[:2]}", "epsg:4326", always_xy=True)
    lon, lat = trf.transform(x, y)
    return lon, lat


# ─────────────────────────────────────────────────────────────────────────────
#  Camera matrix
# ─────────────────────────────────────────────────────────────────────────────

def computeCameraMatrix(truePos: dict) -> np.ndarray:
    """Build 3×3 camera-intrinsic matrix from metadata dictionary."""
    imW, imH = truePos["width"], truePos["height"]
    cam_size = truePos["cam_size"]
    focal    = truePos["focal_len"]
    px_size  = cam_size / np.sqrt(imW**2 + imH**2)
    fx = focal / px_size
    fy = focal / px_size
    K  = np.array([
        [fx,  0,   imW / 2],
        [0,   fy,  imH / 2],
        [0,   0,   1      ],
    ])
    return K


# ─────────────────────────────────────────────────────────────────────────────
#  Rotation helpers
# ─────────────────────────────────────────────────────────────────────────────

def dumpRotateImage(img: np.ndarray,
                    degree: float) -> Tuple[np.ndarray, np.ndarray]:
    """Rotate image by ``degree`` degrees (expanding canvas) and return (img, matrix)."""
    rad = degree / 180 * np.pi
    h, w = img.shape[:2]
    hN = int(w * abs(np.sin(rad)) + h * abs(np.cos(rad)))
    wN = int(h * abs(np.sin(rad)) + w * abs(np.cos(rad)))
    if max(wN, hN, w, h) > 32000:
        M = cv2.getRotationMatrix2D((w // 2, h // 2), 0, 1)
        return img.copy(), M
    M  = cv2.getRotationMatrix2D((w // 2, h // 2), degree, 1)
    M[0, 2] += (wN - w) // 2
    M[1, 2] += (hN - h) // 2
    rotated = cv2.warpAffine(img, M, (wN, hN), borderValue=(0, 0, 0))
    return rotated, M


def rotvector2rot(rotvec: np.ndarray) -> np.ndarray:
    return cv2.Rodrigues(rotvec)[0]


def rot_to_euler(dcm: np.ndarray, lim=None) -> np.ndarray:
    r11 = -dcm[1][0]; r12 = dcm[1][1]; r21 = dcm[1][2]
    r31 = -dcm[0][2]; r32 = dcm[2][2]
    r11a = dcm[0][1]; r12a = dcm[0][0]
    r1   = np.arctan2(r11, r12)
    r21  = np.clip(r21, -1, 1)
    r2   = np.arcsin(r21)
    r3   = np.arctan2(r31, r32)
    if lim == "zeror3":
        for i in np.where(np.abs(r21) == 1.0)[0]:
            r1[i] = np.arctan2(r11a[i], r12a[i])
            r3[i] = 0
    return np.array([-r1 - np.pi, -r2, r3 + np.pi]) * 180 / np.pi


# ─────────────────────────────────────────────────────────────────────────────
#  View-centre estimation
# ─────────────────────────────────────────────────────────────────────────────

def view_center(region: str, config: dict, truePos: dict,
                initialX: float, initialY: float,
                ref_resolution: float,
                reverseMatRotation: Optional[np.ndarray]) -> Tuple[float, float]:
    pitch  = -truePos["pitch"] / 180 * np.pi
    yaw    = -truePos["yaw"]   / 180 * np.pi
    utm_x, utm_y = deg2utm(region, config, truePos["lon"], truePos["lat"])
    delta_y = truePos["rel_alt"] / np.tan(pitch) * np.cos(yaw)
    delta_x = -truePos["rel_alt"] / np.tan(pitch) * np.sin(yaw)
    UTM_Y_c = utm_y + delta_y
    UTM_X_c = utm_x + delta_x
    x0 = int((UTM_X_c - initialX) / ref_resolution)
    y0 = int((initialY - UTM_Y_c) / ref_resolution)
    if reverseMatRotation is not None:
        coord = np.array([x0, y0, 1]) @ reverseMatRotation.T
        return coord[0], coord[1]
    return x0, y0


# ─────────────────────────────────────────────────────────────────────────────
#  Resolution / size helpers
# ─────────────────────────────────────────────────────────────────────────────

def resolution_size(data: dict, opt) -> Tuple[float, np.ndarray]:
    img_w, img_h = data["width"], data["height"]
    cam_size = data["cam_size"];  focal = data["focal_len"]
    pitch = data["pitch"] if getattr(opt, "pose_priori", "unknown") != "unknown" else -30
    res = (2 * data["rel_alt"] / np.sin(-np.pi * pitch / 180)
           * cam_size / 2 / focal / sqrt(img_w**2 + img_h**2))
    sz = min(img_w, img_h)
    return res, np.array([sz, sz])


def crop_center(image_path: str, width: int, height: int) -> Image.Image:
    img = cv2.imread(image_path)
    oh, ow = img.shape[:2]
    cx, cy = ow // 2, oh // 2
    l = max(0, cx - width  // 2); t = max(0, cy - height // 2)
    r = min(ow, cx + width  // 2); b = min(oh, cy + height // 2)
    return Image.fromarray(img[t:b, l:r])


# ─────────────────────────────────────────────────────────────────────────────
#  PnP pose estimation
# ─────────────────────────────────────────────────────────────────────────────

def estimate_drone_pose(region: str, config: dict,
                        match_points: np.ndarray,
                        K: np.ndarray,
                        UTM_X: np.ndarray, UTM_Y: np.ndarray,
                        DSM: np.ndarray):
    """
    Solve PnP to get drone position in geodetic coordinates.

    Returns
    -------
    BLH         : dict  {B: lat, L: lon, H: alt}
    cam_angle   : ndarray  Euler angles of camera
    n_inliers   : int
    inliers     : list
    """
    pose_3d    = np.column_stack((UTM_X, UTM_Y, DSM)).astype(np.float32)
    match_pts  = match_points.astype(np.float32)
    dist_coeffs = np.zeros(4, dtype=np.float32)

    ok, rvec, tvec, inliers = cv2.solvePnPRansac(
        pose_3d, match_pts, K, dist_coeffs, flags=cv2.SOLVEPNP_P3P
    )
    if inliers is None or not ok:
        return {"B": None, "L": None, "H": None}, None, 0, []

    R   = rotvector2rot(rvec)
    X0  = -R.T @ tvec
    lon, lat = utm2deg(region, config, float(X0[0]), float(X0[1]))
    BLH = {"B": float(lat), "L": float(lon), "H": float(X0[2])}

    Rx90 = np.array([[1,0,0],[0,np.cos(-np.pi/2),np.sin(-np.pi/2)],[0,-np.sin(-np.pi/2),np.cos(-np.pi/2)]])
    cam_angle = rot_to_euler(Rx90 @ R)

    return BLH, cam_angle, len(inliers), inliers


# ─────────────────────────────────────────────────────────────────────────────
#  Error metric
# ─────────────────────────────────────────────────────────────────────────────

def pos2error(truePos: dict,
              BLH_list: List[dict],
              inliers_list: List[int]):
    inliers_arr = np.array(inliers_list)
    max_idx     = int(np.argmax(inliers_arr))
    errors = []
    for BLH in BLH_list:
        if BLH["L"] and BLH["B"]:
            e_lat = (BLH["B"] - truePos["lat"]) * 110000
            e_lon = (BLH["L"] - truePos["lon"]) * 110000 * math.cos(truePos["lat"])
            errors.append(math.sqrt(e_lat**2 + e_lon**2))
        else:
            errors.append(10000.0)
    pred_loc = {"lon": BLH_list[max_idx]["L"], "lat": BLH_list[max_idx]["B"]}
    return pred_loc, errors[max_idx], errors


# ─────────────────────────────────────────────────────────────────────────────
#  Gallery (reference tile) helpers
# ─────────────────────────────────────────────────────────────────────────────

def compute_block_mid_wo_black(image: np.ndarray,
                               block_size: List[int],
                               step_size:  List[int]) -> np.ndarray:
    xs, ys, _ = image.shape
    small = cv2.resize(image, (ys // 10, xs // 10), interpolation=cv2.INTER_NEAREST)
    mids = []
    total_px = block_size[0] * block_size[1] / 100
    for i in range(len(range(0, xs, step_size[0]))):
        for j in range(len(range(0, ys, step_size[1]))):
            sx = min(i * step_size[0], xs - block_size[0] - 1)
            sy = min(j * step_size[1], ys - block_size[1] - 1)
            ex = sx + block_size[0]; ey = sy + block_size[1]
            blk = small[sx//10:ex//10, sy//10:ey//10]
            if np.sum(blk[:, :, 0] > 0) >= total_px / 5 * 2:
                mids.append([(sx+ex)/2, (sy+ey)/2])
    return np.array(mids)


# ─────────────────────────────────────────────────────────────────────────────
#  Config loader
# ─────────────────────────────────────────────────────────────────────────────

def load_config_parameters_new(config: dict, opt, region: str):
    import rasterio
    ref_type = opt.Ref_type

    # Use rasterio to read GeoTIFF files (cv2.imread cannot handle GeoTIFF)
    ref_path = config[f"{region}_{ref_type}_REF_PATH"]
    dsm_path = config[f"{region}_{ref_type}_DSM_PATH"]

    with rasterio.open(ref_path) as src:
        data = src.read()                          # (bands, H, W)
        if data.shape[0] >= 3:
            # Rasterio reads as (C,H,W); take RGB and convert to HWC uint8
            ref_image = np.stack([data[2], data[1], data[0]], axis=-1).astype(np.uint8)
        else:
            ref_image = np.stack([data[0]] * 3, axis=-1).astype(np.uint8)

    with rasterio.open(dsm_path) as src:
        dsm_image = src.read(1).astype(np.float32)   # single-band elevation

    ref_res   = config[f"{region}_{ref_type}_REF_resolution"]
    save_path = opt.save_dir
    return ref_image, dsm_image, save_path, ref_res


# ─────────────────────────────────────────────────────────────────────────────
#  Model init helpers
# ─────────────────────────────────────────────────────────────────────────────

def matching_init(method_dict: dict) -> dict:
    """Initialise the matching model.

    Parameters
    ----------
    method_dict  : dict with key ``'matching_method'``
    """
    if method_dict["matching_method"] == "Roma":
        from Matching_Models.RoMa.demo.Roma_match import Roma_Init
        method_dict["matching_model"] = Roma_Init()
    else:
        raise ValueError(f"Unknown matching method: {method_dict['matching_method']}")
    return method_dict


def retrieval_init(method_dict: dict, config: dict) -> dict:
    """Initialise the retrieval model.

    Parameters
    ----------
    method_dict  : dict with key ``'retrieval_method'``
    config       : merged benchmark config dict
    """
    from Retrieval_Models.multi_model_loader import get_Model
    if method_dict["retrieval_method"] not in config["RETRIEVAL_METHODS"]:
        raise ValueError(f"Unknown retrieval method: {method_dict['retrieval_method']}")
    method_dict["retrieval_model"], method_dict["img_transform"] = \
        get_Model(method_dict["retrieval_method"])
    method_dict["retrieval_model"].to(config["DEVICE"])
    method_dict["retrieval_cover"]       = config["RETRIEVAL_COVER"]
    method_dict["retrieval_topn"]        = config["RETRIEVAL_TOPN"]
    method_dict["retrieval_img_name"]    = config["RETRIEVAL_IMG"]
    method_dict["retrieval_feat_norm"]   = config["RETRIEVAL_FEATURE_NORM"]
    return method_dict


# ─────────────────────────────────────────────────────────────────────────────
#  Feature extraction helper
# ─────────────────────────────────────────────────────────────────────────────

def _extract_features(retrieval_size, config, method_dict, MID, block_size,
                      ref_image, UAV_img, img_transform, retrieval_model):
    from Retrieval_Models.feature_extract import extract_features
    return extract_features(retrieval_size, config, method_dict, MID, block_size,
                            ref_image, UAV_img, img_transform, retrieval_model)


# ─────────────────────────────────────────────────────────────────────────────
#  Core pipeline functions
# ─────────────────────────────────────────────────────────────────────────────

def retrieval_all(ref_image:      np.ndarray,
                  UAV_path:       str,
                  uav_data:       dict,
                  ref_resolution: float,
                  matRotation:    np.ndarray,
                  save_path:      str,
                  opt,
                  region:         str,
                  config:         dict,
                  method_dict:    dict):
    """
    Image-level retrieval: returns ranked reference tile locations.

    Returns
    -------
    order, refLocX, refLocY, PDE_list, cut_H, cut_W, finescale, retrieval_time
    """
    from Retrieval_Models.feature_extract import extract_features

    retrieval_size = 384
    ref_type = opt.Ref_type
    initialX = config[f"{region}_{ref_type}_REF_initialX"]
    initialY = config[f"{region}_{ref_type}_REF_initialY"]

    if getattr(opt, "pose_priori", "unknown") == "yp":
        revMat = cv2.invertAffineTransform(matRotation)
    else:
        revMat = None

    center_x, center_y = view_center(region, config, uav_data,
                                     initialX, initialY, ref_resolution, revMat)

    cover  = method_dict["retrieval_cover"]
    drone_res, drone_size = resolution_size(uav_data, opt)
    finescale = drone_res / ref_resolution
    view_size = drone_size * drone_res

    block_size = [
        math.ceil(view_size[0]/ref_resolution) + (math.ceil(view_size[0]/ref_resolution)%2!=0),
        math.ceil(view_size[1]/ref_resolution) + (math.ceil(view_size[1]/ref_resolution)%2!=0),
    ]
    step_size = [int(block_size[0]*(100-cover)/100), int(block_size[1]*(100-cover)/100)]

    MID = compute_block_mid_wo_black(ref_image, block_size, step_size)
    if len(MID) == 0:
        h, w = ref_image.shape[:2]
        block_size = [min(block_size[0], h), min(block_size[1], w)]
        MID = np.array([[h / 2, w / 2]])
    mids = MID.reshape(-1, 2)

    UAV_img = crop_center(UAV_path, int(drone_size[0]), int(drone_size[1]))
    UAV_img = UAV_img.resize((retrieval_size, retrieval_size))
    UAV_img_t = method_dict["img_transform"](UAV_img)

    t0 = time.time()
    gf, qf = extract_features(retrieval_size, config, method_dict, MID, block_size,
                               ref_image, UAV_img_t, method_dict["img_transform"],
                               method_dict["retrieval_model"])
    t1 = time.time()
    retrieval_time = (t1 - t0) / (len(MID) + 1)

    score = (gf @ qf.unsqueeze(-1)).squeeze().cpu().numpy()
    score = np.atleast_1d(score)
    order = np.argsort(score)[::-1]

    best_start_x, best_start_y, PDE_list = [], [], []
    for i in order:
        mx, my = mids[i]
        best_start_x.append(max(0, int(mx - block_size[0]/2)))
        best_start_y.append(max(0, int(my - block_size[1]/2)))
        d = math.sqrt((center_x - my)**2 + (center_y - mx)**2)
        PDE_list.append(d / block_size[0])

    return order, best_start_x, best_start_y, PDE_list, \
           block_size[1], block_size[0], finescale, retrieval_time


def Match2Pos_all(opt,
                  region:         str,
                  config:         dict,
                  uav_img0:       np.ndarray,
                  finescale:      float,
                  K:              np.ndarray,
                  ref_image:      np.ndarray,
                  dsm_image:      np.ndarray,
                  refLocY_list,
                  refLocX_list,
                  cut_H:          int,
                  cut_W:          int,
                  save_path:      str,
                  method_dict:    dict,
                  matRotation:    np.ndarray):
    """
    Pixel-level matching + PnP pose solving.

    Returns
    -------
    BLH_list, inliers_list, match_time, pnp_time,
    sen_pts_list, ref_pts_list          ← extra outputs for yaw/height solver
    """
    from Matching_Models.RoMa.demo.Roma_match import Roma_match

    if getattr(opt, "pose_priori", "unknown") == "yp":
        reverseMatRotation = cv2.invertAffineTransform(matRotation)
    else:
        reverseMatRotation = None

    resize_ratio = opt.resize_ratio
    ref_type     = opt.Ref_type
    initialX     = config[f"{region}_{ref_type}_REF_initialX"]
    initialY     = config[f"{region}_{ref_type}_REF_initialY"]
    ref_res      = config[f"{region}_{ref_type}_REF_resolution"]
    dsm_res      = config[f"{region}_{ref_type}_DSM_resolution"]
    dsm_coor     = config[f"{region}_{ref_type}_DSM_COORDINATE"]
    ref_coor     = config[f"{region}_{ref_type}_REF_COORDINATE"]
    dsm_ratio    = dsm_res / ref_res
    dsm_coor2    = [dsm_coor[0]*dsm_ratio, dsm_coor[1]*dsm_ratio]
    dsm_offset   = (int(dsm_coor2[0]-ref_coor[0]), int(dsm_coor2[1]-ref_coor[1]))

    if resize_ratio < 1:
        uav_img = cv2.resize(uav_img0, None, fx=resize_ratio, fy=resize_ratio)
    else:
        uav_img = uav_img0

    if not isinstance(refLocY_list, list):
        refLocX_list = [refLocX_list]; refLocY_list = [refLocY_list]

    strategy = getattr(opt, "strategy", "Topn_opt")
    if strategy == "Top1":
        top_n = 1
    elif strategy == "Topn_opt":
        top_n = min(method_dict["retrieval_topn"], len(refLocY_list))
    else:
        top_n = len(refLocY_list)

    BLH_list, inliers_list, time_list = [], [], []
    sen_pts_list, ref_pts_list = [], []

    for index in range(top_n):
        refLocY = refLocY_list[index]; refLocX = refLocX_list[index]
        fineRef = ref_image[refLocX:refLocX+cut_W, refLocY:refLocY+cut_H]
        fx = fy = resize_ratio / finescale
        if fineRef.size == 0 or round(fineRef.shape[0] * fy) < 1 or round(fineRef.shape[1] * fx) < 1:
            BLH_list.append({"B": None, "L": None, "H": None})
            inliers_list.append(0)
            time_list.append([0.0, 0.0])
            sen_pts_list.append([]); ref_pts_list.append([])
            continue
        fineRef = cv2.resize(fineRef, None, fx=fx, fy=fy)

        t_m0 = time.time()
        Sen_pts, Ref_pts = Roma_match(uav_img, fineRef,
                                      method_dict["matching_model"],
                                      save_path,
                                      f"/top{index+1}_ransac.png",
                                      need_ransac=False, show_matches=False)
        t_m1 = time.time()

        if len(Ref_pts) >= 5:
            refCoord = np.array(Ref_pts) / resize_ratio * finescale + \
                       np.array([refLocY, refLocX])
            if reverseMatRotation is not None:
                rc1 = np.hstack([refCoord, np.ones((len(refCoord), 1))])
                refCoord = rc1 @ reverseMatRotation.T
            UTM_X = refCoord[:, 0] * ref_res + initialX
            UTM_Y = initialY - refCoord[:, 1] * ref_res

            dsm_x = (refCoord[:,1]+dsm_offset[1]+1)/dsm_ratio-1
            dsm_y = (refCoord[:,0]+dsm_offset[0]+1)/dsm_ratio-1
            dx1 = np.clip(dsm_x.astype(int), 0, dsm_image.shape[0]-1)
            dy1 = np.clip(dsm_y.astype(int), 0, dsm_image.shape[1]-1)
            DSM = dsm_image[dx1, dy1]

            match_pts = np.array(Sen_pts) / resize_ratio
            BLH, _, inliers, _ = estimate_drone_pose(
                region, config, match_pts, K, UTM_X, UTM_Y, DSM)
        else:
            BLH     = {"B": None, "L": None, "H": None}
            inliers = 0

        t_p1 = time.time()
        time_list.append([t_m1-t_m0, t_p1-t_m1])
        BLH_list.append(BLH); inliers_list.append(inliers)
        sen_pts_list.append(Sen_pts); ref_pts_list.append(Ref_pts)

    match_time = [t[0] for t in time_list]
    pnp_time   = [t[1] for t in time_list]
    return BLH_list, inliers_list, match_time, pnp_time, sen_pts_list, ref_pts_list
