"""
BenchmarkPipeline for the native AnyVisLoc baseline adapter.

High-level pipeline object that wraps
  • Benchmark CAMP retrieval
  • Benchmark RoMa matching + PnP (position)
  • native AnyVisLoc HomographyYawHeightSolver (yaw + altitude)
into a single per-frame call.

Usage
-----
from uav_geolocalization.baselines.native_anyvisloc.pipeline import BenchmarkPipeline

pipe = BenchmarkPipeline(opt, config, region,
                         ref_image, dsm_image, ref_resolution,
                         method_dict)
result = pipe.process_frame(uav_path, truePos)
"""

from __future__ import annotations
import os
import time
import math
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from .benchmark_utils import (
    computeCameraMatrix, dumpRotateImage,
    Match2Pos_all, retrieval_all, pos2error,
    query_data_from_file, img_name,
    save_data,
    resolution_size, view_center,
)
from .geometry_solver import HomographyYawHeightSolver, extract_yaw_and_height
from .types import CameraIntrinsics, Pose
from .coord_transforms import deg2utm, utm2deg


def affine_rotation_angle_rad(matrix: Optional[np.ndarray]) -> float:
    """Return the in-plane rotation angle from a 2x3 OpenCV affine matrix."""
    if matrix is None:
        return 0.0
    mat = np.asarray(matrix, dtype=np.float64)
    if mat.shape[0] < 2 or mat.shape[1] < 2:
        return 0.0
    return float(np.arctan2(mat[0, 1], mat[0, 0]))


def _reference_window_anchor(
    *,
    region: str,
    config: Dict[str, Any],
    ref_type: str,
    ref_resolution: float,
    row_start: float,
    col_start: float,
    row_size: float,
    col_size: float,
    mat_rotation: Optional[np.ndarray],
) -> Optional[Dict[str, float]]:
    """Convert a retrieved reference crop center into a WGS84 anchor."""
    try:
        ref_x = float(col_start) + float(col_size) / 2.0
        ref_y = float(row_start) + float(row_size) / 2.0
        coord = np.array([ref_x, ref_y, 1.0], dtype=np.float64)
        if mat_rotation is not None:
            coord = coord @ cv2.invertAffineTransform(mat_rotation).T
        initial_x = float(config[f"{region}_{ref_type}_REF_initialX"])
        initial_y = float(config[f"{region}_{ref_type}_REF_initialY"])
        utm_x = float(coord[0]) * float(ref_resolution) + initial_x
        utm_y = initial_y - float(coord[1]) * float(ref_resolution)
        lon, lat = utm2deg(region, config, utm_x, utm_y)
        if not np.isfinite(lat) or not np.isfinite(lon):
            return None
        return {"lat": float(lat), "lon": float(lon)}
    except Exception:
        return None


class BenchmarkPipeline:
    """
    End-to-end UAV visual localization pipeline.

    Parameters
    ----------
    opt : object
        Options namespace with attrs:
          pose_priori, strategy, Ref_type, resize_ratio, save_dir
    config : dict
        Merged region + global config dictionary.
    region : str
        Region name (e.g. "QZ_Town").
    ref_image : np.ndarray
        Full 2-D reference map (BGR, uint8).
    dsm_image : np.ndarray
        DSM elevation map (float32).
    ref_resolution : float
        Reference map GSD in m/pixel.
    method_dict : dict
        Initialised retrieval + matching models.
    gsd_for_solver : float, optional
        GSD passed to HomographyYawHeightSolver (defaults to ref_resolution).
    min_inliers_solver : int
        Minimum inliers for HomographyYawHeightSolver.
    """

    def __init__(self,
                 opt,
                 config:          dict,
                 region:          str,
                 ref_image:       np.ndarray,
                 dsm_image:       np.ndarray,
                 ref_resolution:  float,
                 method_dict:     dict,
                 gsd_for_solver:  Optional[float] = None,
                 min_inliers_solver: int = 8):
        self.opt            = opt
        self.config         = config
        self.region         = region
        self.ref_image      = ref_image
        self.dsm_image      = dsm_image
        self.ref_resolution = ref_resolution
        self.method_dict    = method_dict

        gsd = gsd_for_solver if gsd_for_solver is not None else ref_resolution
        self.solver = HomographyYawHeightSolver(
            gsd         = gsd,
            min_inliers = min_inliers_solver,
            altitude_source = "estimate",
        )

    # ──────────────────────────────────────────────────────────────────────────

    def process_frame(self, uav_path: str, truePos: dict) -> Dict[str, Any]:
        """
        Process a single UAV frame.

        Parameters
        ----------
        uav_path : str   Path to the UAV image file.
        truePos  : dict  Ground-truth metadata (lat, lon, yaw, rel_alt, …).

        Returns
        -------
        dict with keys:
          img_path, gt_lat, gt_lon, gt_alt, gt_yaw,
          pred_lat, pred_lon, pred_alt (PnP),
          src_yaw_deg, src_alt_m, n_inliers_yaw,
          pred_error_m (PnP position error),
          yaw_error_deg, alt_error_m,
          retrieval_time_s, match_time_s, pnp_time_s, total_time_s,
          success
        """
        t_total_start = time.time()

        gt_lat  = truePos["lat"];  gt_lon = truePos["lon"]
        gt_alt  = truePos["rel_alt"]
        gt_yaw  = truePos.get("yaw", None)

        result: Dict[str, Any] = dict(
            img_path=uav_path,
            gt_lat=gt_lat, gt_lon=gt_lon, gt_alt=gt_alt, gt_yaw=gt_yaw,
            pred_lat=None, pred_lon=None, pred_alt=None,
            src_yaw_deg=None, src_alt_m=None, n_inliers_yaw=0,
            native_yaw_height_status="not_run",
            homography_valid=False,
            geodetic_anchor=None, raw_retrieval_anchor=None,
            selected_reference_name="", raw_top1_reference_name="",
            retrieval_candidate_index=-1, raw_retrieval_candidate_index=-1,
            pred_error_m=9999.0,
            yaw_error_deg=None, alt_error_m=None,
            retrieval_time_s=0.0, match_time_s=0.0,
            pnp_time_s=0.0, total_time_s=0.0,
            success=False,
        )

        try:
            K         = computeCameraMatrix(truePos)
            uav_image = cv2.imread(uav_path)

            # ── Step 1: Rotate reference map by prior yaw ──────────────────
            ref_map, matRotation = dumpRotateImage(self.ref_image, truePos["yaw"])

            # ── Step 2: Image-level retrieval ──────────────────────────────
            t0 = time.time()
            save_path = os.path.join(self.opt.save_dir, self.region,
                                     os.path.basename(os.path.dirname(uav_path)),
                                     img_name(uav_path))
            os.makedirs(save_path, exist_ok=True)

            if bool(truePos.get("oracle_tile_enabled", False)):
                ref_type = self.opt.Ref_type
                initial_x = self.config[f"{self.region}_{ref_type}_REF_initialX"]
                initial_y = self.config[f"{self.region}_{ref_type}_REF_initialY"]
                rev_mat = cv2.invertAffineTransform(matRotation) if getattr(self.opt, "pose_priori", "unknown") == "yp" else None
                center_col, center_row = view_center(
                    self.region, self.config, truePos, initial_x, initial_y, self.ref_resolution, rev_mat
                )
                drone_res, drone_size = resolution_size(truePos, self.opt)
                fineScale = drone_res / self.ref_resolution
                view_size = drone_size * drone_res
                block_size = [
                    math.ceil(view_size[0] / self.ref_resolution)
                    + (math.ceil(view_size[0] / self.ref_resolution) % 2 != 0),
                    math.ceil(view_size[1] / self.ref_resolution)
                    + (math.ceil(view_size[1] / self.ref_resolution) % 2 != 0),
                ]
                cut_H, cut_W = int(block_size[1]), int(block_size[0])
                max_row_start = max(0, int(ref_map.shape[0]) - cut_W - 1)
                max_col_start = max(0, int(ref_map.shape[1]) - cut_H - 1)
                refLocX = [int(np.clip(center_row - cut_W / 2.0, 0, max_row_start))]
                refLocY = [int(np.clip(center_col - cut_H / 2.0, 0, max_col_start))]
                IR_order = np.array([0], dtype=np.int64)
                PDE_list = [0.0]
                ret_t = 0.0
                result["native_oracle_tile_used"] = True
            else:
                IR_order, refLocX, refLocY, PDE_list, cut_H, cut_W, fineScale, ret_t = \
                    retrieval_all(ref_map, uav_path, truePos,
                                  self.ref_resolution, matRotation,
                                  save_path, self.opt,
                                  self.region, self.config, self.method_dict)
            result["retrieval_time_s"] = ret_t
            topn = int(min(self.method_dict.get("retrieval_topn", len(IR_order)), len(IR_order)))
            result["raw_retrieval_topk_indices"] = [int(i) for i in list(IR_order[:topn])]
            candidate_anchors: List[Optional[Dict[str, float]]] = []
            for rank in range(topn):
                if bool(truePos.get("oracle_tile_enabled", False)) and isinstance(
                    truePos.get("oracle_reference_anchor"), dict
                ):
                    candidate_anchors.append(dict(truePos["oracle_reference_anchor"]))
                else:
                    candidate_anchors.append(
                        _reference_window_anchor(
                            region=self.region,
                            config=self.config,
                            ref_type=self.opt.Ref_type,
                            ref_resolution=self.ref_resolution,
                            row_start=refLocX[rank],
                            col_start=refLocY[rank],
                            row_size=cut_W,
                            col_size=cut_H,
                            mat_rotation=matRotation,
                        )
                    )
            if topn > 0:
                result["raw_retrieval_candidate_index"] = int(IR_order[0])
                result["raw_retrieval_anchor"] = candidate_anchors[0]
                result["raw_top1_reference_name"] = f"native_tile_rank0_idx{int(IR_order[0])}"

            # ── Step 3: Pixel-level matching + PnP ────────────────────────
            (BLH_list, inliers_list,
             match_time, pnp_time,
             sen_pts_list, ref_pts_list) = Match2Pos_all(
                self.opt, self.region, self.config,
                uav_image, fineScale, K,
                ref_map, self.dsm_image,
                refLocY, refLocX, cut_H, cut_W,
                save_path, self.method_dict, matRotation,
            )
            result["match_time_s"] = float(np.sum(match_time))
            result["pnp_time_s"]   = float(np.sum(pnp_time))
            result["pnp_inliers_list"] = [int(v) for v in inliers_list]
            result["native_rerank_policy"] = "max_pnp_inliers_then_retrieval_rank"
            result["selected_candidate_rank"] = int(np.argmax(inliers_list)) if inliers_list else -1
            result["n_inliers"] = int(max(inliers_list)) if inliers_list else 0
            result["num_matches"] = int(sum(len(v) for v in sen_pts_list)) if sen_pts_list else 0
            if 0 <= result["selected_candidate_rank"] < topn:
                selected_rank = int(result["selected_candidate_rank"])
                result["retrieval_candidate_index"] = int(IR_order[selected_rank])
                result["geodetic_anchor"] = candidate_anchors[selected_rank]
                result["selected_reference_name"] = (
                    f"native_tile_rank{selected_rank}_idx{int(IR_order[selected_rank])}"
                )
            result["pnp_candidates"] = [
                {
                    "retrieval_rank": int(rank),
                    "candidate_id": int(IR_order[rank]),
                    "pnp_inliers": int(inliers_list[rank]) if rank < len(inliers_list) else 0,
                    "geodetic_anchor": candidate_anchors[rank],
                }
                for rank in range(topn)
            ]

            pred_loc, pred_error, _ = pos2error(truePos, BLH_list, inliers_list)
            result["pred_error_m"] = pred_error
            if pred_loc["lat"] is not None:
                result["pred_lat"] = pred_loc["lat"]
                result["pred_lon"] = pred_loc["lon"]
                result["success"]  = True

            # ── Step 4: Yaw + Height from HomographyYawHeightSolver ───────
            #  Use the best candidate (most inliers)
            best_cand = int(np.argmax(inliers_list)) if inliers_list else 0
            sen_pts = np.array(sen_pts_list[best_cand]) if sen_pts_list else np.empty((0,2))
            ref_pts = np.array(ref_pts_list[best_cand]) if ref_pts_list else np.empty((0,2))

            ref_type = self.opt.Ref_type
            finescale_used = fineScale
            resize_ratio   = self.opt.resize_ratio
            ref_cut_W, ref_cut_H = cut_W, cut_H

            # Scale pts back to original reference coordinate space
            if len(sen_pts) > 0 and len(ref_pts) > 0:
                try:
                    # ref_pts are in the cropped-resized tile → rescale to tile pixels
                    ref_pts_tile = ref_pts / resize_ratio * finescale_used

                    yaw_deg, alt_m, n_inliers_yaw = extract_yaw_and_height(
                        query_pts   = sen_pts / resize_ratio,
                        ref_pts     = ref_pts_tile,
                        camera_K    = K,
                        img_width   = truePos["width"],
                        img_height  = truePos["height"],
                        ref_cut_W   = ref_cut_W,
                        ref_cut_H   = ref_cut_H,
                        gsd         = self.solver._gsd,
                        rotation_correction_rad = affine_rotation_angle_rad(matRotation),
                        min_inliers = self.solver._min_inliers,
                    )
                    result["src_yaw_deg"]     = yaw_deg
                    result["src_alt_m"]       = alt_m
                    result["n_inliers_yaw"]   = n_inliers_yaw
                    result["native_yaw_height_status"] = (
                        "ok" if yaw_deg is not None and alt_m is not None else "failed"
                    )
                    result["homography_valid"] = result["native_yaw_height_status"] == "ok"

                    if gt_yaw is not None and yaw_deg is not None:
                        d = abs(yaw_deg - gt_yaw) % 360
                        result["yaw_error_deg"] = float(min(d, 360 - d))
                    if alt_m is not None:
                        result["alt_error_m"] = float(abs(alt_m - gt_alt))
                except Exception as exc:
                    result["native_yaw_height_status"] = "failed"
                    result["yaw_height_error"] = str(exc)
            else:
                result["native_yaw_height_status"] = "failed"

        except Exception as exc:
            result["error"] = str(exc)

        result["total_time_s"] = time.time() - t_total_start
        return result

    # ──────────────────────────────────────────────────────────────────────────

    def run_dataset(self,
                    UAV_img_list:   List[str],
                    json_file:      str,
                    save_pkl_root:  str,
                    skip_existing:  bool = True) -> List[Dict[str, Any]]:
        """
        Process a full list of UAV images.

        Parameters
        ----------
        UAV_img_list  : list of image paths
        json_file     : path to the region metadata JSON
        save_pkl_root : root folder for per-frame pkl files
        skip_existing : skip frames that already have a pkl file

        Returns
        -------
        List of per-frame result dicts (same schema as process_frame()).
        """
        from tqdm.auto import tqdm
        all_results = []
        for idx, uav_path in enumerate(tqdm(UAV_img_list, desc=self.region)):
            pkl_path = os.path.join(
                save_pkl_root, self.region,
                os.path.basename(os.path.dirname(uav_path)),
                f"{img_name(uav_path)}.pkl",
            )
            if skip_existing and os.path.exists(pkl_path):
                try:
                    (r,) = load_data(pkl_path, "result")
                    all_results.append(r); continue
                except Exception:
                    pass

            truePos_list = query_data_from_file(
                json_file,
                name=f"{os.path.dirname(uav_path)}/{os.path.basename(uav_path)}",
            )
            if not truePos_list:
                continue
            truePos = truePos_list[0]

            r = self.process_frame(uav_path, truePos)
            all_results.append(r)
            save_data(pkl_path, result=r)

        return all_results

    # ──────────────────────────────────────────────────────────────────────────
    #  Convenience summary
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def summarize(all_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Print and return aggregate metrics from a list of frame results."""
        import math

        pos_errors = [r["pred_error_m"] for r in all_results
                      if r.get("pred_error_m", 9999) < 9000]
        yaw_errors = [r["yaw_error_deg"] for r in all_results
                      if r.get("yaw_error_deg") is not None]
        alt_errors = [r["alt_error_m"]   for r in all_results
                      if r.get("alt_error_m") is not None]

        def _pct(arr, t):
            return 100*sum(e<=t for e in arr)/len(arr) if arr else 0.0

        summary = {
            "n_total":        len(all_results),
            "n_pos_valid":    len(pos_errors),
            "n_yaw_valid":    len(yaw_errors),
            "pos_mean_m":     float(np.mean(pos_errors))   if pos_errors else None,
            "pos_median_m":   float(np.median(pos_errors)) if pos_errors else None,
            "pos_A5m":  _pct(pos_errors,  5),
            "pos_A10m": _pct(pos_errors, 10),
            "pos_A20m": _pct(pos_errors, 20),
            "yaw_mae_deg":    float(np.mean(yaw_errors))   if yaw_errors else None,
            "yaw_median_deg": float(np.median(yaw_errors)) if yaw_errors else None,
            "yaw_A5deg":  _pct(yaw_errors,  5),
            "yaw_A10deg": _pct(yaw_errors, 10),
            "alt_mae_m":      float(np.mean(alt_errors))   if alt_errors else None,
            "alt_median_m":   float(np.median(alt_errors)) if alt_errors else None,
        }

        print("=" * 60)
        print("PIPELINE SUMMARY")
        print(f"  Frames processed : {summary['n_total']}")
        print(f"  PnP success      : {summary['n_pos_valid']}")
        if pos_errors:
            print(f"\n  [Position]")
            print(f"    Mean   : {summary['pos_mean_m']:.2f} m")
            print(f"    Median : {summary['pos_median_m']:.2f} m")
            print(f"    A@5m   : {summary['pos_A5m']:.1f}%")
            print(f"    A@10m  : {summary['pos_A10m']:.1f}%")
            print(f"    A@20m  : {summary['pos_A20m']:.1f}%")
        if yaw_errors:
            print(f"\n  [Yaw – src solver]")
            print(f"    MAE    : {summary['yaw_mae_deg']:.2f}°")
            print(f"    Median : {summary['yaw_median_deg']:.2f}°")
            print(f"    A@5°   : {summary['yaw_A5deg']:.1f}%")
            print(f"    A@10°  : {summary['yaw_A10deg']:.1f}%")
        if alt_errors:
            print(f"\n  [Altitude – src solver]")
            print(f"    MAE    : {summary['alt_mae_m']:.2f} m")
            print(f"    Median : {summary['alt_median_m']:.2f} m")
        print("=" * 60)
        return summary
