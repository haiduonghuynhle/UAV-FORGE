from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from tqdm import tqdm

from ortholoc.dataset import OrthoLoC
from ortholoc.image_matching.MatcherGT import MatcherGT
from ortholoc.image_matching.MatcherIMCUI import MatcherIMCUI, MATCHER_ZOO
from ortholoc import utils

from uav_geolocalization.baselines.native_anyvisloc.geometry_solver import extract_yaw_and_height


def _yaw_from_rotmat(R: np.ndarray) -> float:
    yaw_rad = np.arctan2(R[1, 0], R[0, 0])
    return float(np.degrees(yaw_rad) % 360.0)


def _robust_center_depth(sample: dict, patch_radius: int = 5) -> float:
    depth_map = utils.geometry.point_to_depth_map(
        point_map=sample["point_map"],
        extrinsics=sample["pose_world2query"],
    )
    h, w = depth_map.shape
    cy, cx = h // 2, w // 2
    y0, y1 = max(0, cy - patch_radius), min(h, cy + patch_radius + 1)
    x0, x1 = max(0, cx - patch_radius), min(w, cx + patch_radius + 1)
    patch = depth_map[y0:y1, x0:x1]
    valid = patch[np.isfinite(patch)]
    valid = valid[valid > 0]
    if len(valid) == 0:
        valid_all = depth_map[np.isfinite(depth_map)]
        valid_all = valid_all[valid_all > 0]
        return float(np.nanmedian(valid_all)) if len(valid_all) else float("nan")
    return float(np.median(valid))


def run_yaw_height(
    dataset_dir: str | None,
    output_csv: str,
    matcher_name: str = "RoMa",
    mode: int = 0,
    set_name: str = "test_inPlace",
    limit_size: float | None = None,
    min_conf: float = 0.5,
    device: str = "cuda",
    angles: list[int] | None = None,
) -> None:
    dataset = OrthoLoC(
        dataset_dir=dataset_dir,
        set_name=set_name,
        mode=mode,
        limit_size=limit_size,
        shuffle=False,
        return_tensor=False,
    )

    if matcher_name == "GT":
        matcher = MatcherGT(dataset=dataset)
    else:
        matcher = MatcherIMCUI(name=matcher_name, device=device, angles=angles)

    rows = []
    for i in tqdm(range(len(dataset)), desc="Yaw/Height"):
        sample = dataset[i]
        sample_id = sample["sample_id"]

        h0, w0 = sample["image_query"].shape[:2]
        h1, w1 = sample["image_dop"].shape[:2]
        K = sample["intrinsics_query"].astype(np.float64)

        yaw_gt = _yaw_from_rotmat(sample["pose_query2world"][:3, :3])
        height_gt = _robust_center_depth(sample)

        all_corr = matcher.run(
            img0=sample["image_query"],
            img1=sample["image_dop"],
            normalized=True,
            covisible_only=True,
            silent=True,
            angles=angles,
        )
        if len(all_corr) == 0:
            rows.append({
                "sample_id": sample_id,
                "yaw_gt_deg": yaw_gt,
                "height_gt_m": height_gt,
                "yaw_est_deg": "",
                "height_est_m": "",
                "yaw_err_deg": "",
                "height_err_m": "",
                "n_matches": 0,
                "n_geo_inliers": 0,
            })
            continue

        best_idx = max(range(len(all_corr)), key=lambda idx: len(all_corr[idx].take_min_conf(min_conf)))
        corr = all_corr[best_idx]
        corr = corr.take_min_conf(min_conf=min_conf, inclusive=False)
        corr = corr.take_covisible(w0=w0, h0=h0, w1=w1, h1=h1, is_normalized=True)

        n_matches = len(corr)
        if not corr.is_valid:
            rows.append({
                "sample_id": sample_id,
                "yaw_gt_deg": yaw_gt,
                "height_gt_m": height_gt,
                "yaw_est_deg": "",
                "height_est_m": "",
                "yaw_err_deg": "",
                "height_err_m": "",
                "n_matches": n_matches,
                "n_geo_inliers": 0,
            })
            continue

        corr_dn = corr.denormalized(w0=w0, h0=h0, w1=w1, h1=h1)
        query_pts = corr_dn.pts0.astype(np.float64)
        ref_pts = corr_dn.pts1.astype(np.float64)
        gsd_ref = float(np.mean(np.abs(np.asarray(sample["scale"], dtype=np.float64))))

        yaw_est, height_est, n_geo_inliers = extract_yaw_and_height(
            query_pts=query_pts,
            ref_pts=ref_pts,
            camera_K=K,
            img_width=w0,
            img_height=h0,
            ref_cut_W=w1,
            ref_cut_H=h1,
            gsd=gsd_ref,
        )

        yaw_err = ""
        if yaw_est is not None:
            d = abs(float(yaw_est) - yaw_gt) % 360.0
            yaw_err = min(d, 360.0 - d)

        height_err = ""
        if height_est is not None and np.isfinite(height_gt):
            height_err = abs(float(height_est) - float(height_gt))

        rows.append({
            "sample_id": sample_id,
            "yaw_gt_deg": yaw_gt,
            "height_gt_m": height_gt,
            "yaw_est_deg": "" if yaw_est is None else float(yaw_est),
            "height_est_m": "" if height_est is None else float(height_est),
            "yaw_err_deg": yaw_err,
            "height_err_m": height_err,
            "n_matches": n_matches,
            "n_geo_inliers": int(n_geo_inliers),
        })

    out_path = Path(output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "sample_id",
                "yaw_gt_deg",
                "height_gt_m",
                "yaw_est_deg",
                "height_est_m",
                "yaw_err_deg",
                "height_err_m",
                "n_matches",
                "n_geo_inliers",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    yaw_errs = np.array([r["yaw_err_deg"] for r in rows if r["yaw_err_deg"] != ""], dtype=float)
    h_errs = np.array([r["height_err_m"] for r in rows if r["height_err_m"] != ""], dtype=float)
    print(f"Saved results to: {out_path}")
    print(f"Samples: {len(rows)}")
    if len(yaw_errs):
        print(f"Yaw error   median={np.median(yaw_errs):.2f}°  mean={np.mean(yaw_errs):.2f}°")
    if len(h_errs):
        print(f"Height error median={np.median(h_errs):.2f}m  mean={np.mean(h_errs):.2f}m")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run OrthoLoC yaw/height evaluation with new_src.geometry_solver")
    p.add_argument("--dataset_dir", type=str, default=None, help="Dataset root; if omitted, OrthoLoC default is used")
    p.add_argument("--set_name", type=str, default="test_inPlace", choices=["all", "train", "val", "test_inPlace", "test_outPlace"])
    p.add_argument("--mode", type=int, default=0, choices=[0, 1, 2, 3])
    p.add_argument("--limit_size", type=float, default=None)
    p.add_argument("--matcher", dest="matcher_name", type=str, default="RoMa", choices=["GT"] + list(MATCHER_ZOO.keys()))
    p.add_argument("--min_conf", type=float, default=0.5)
    p.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--angles", nargs="+", type=int, default=None)
    p.add_argument("--output_csv", type=str, default="ortholoc_results/yaw_height_results.csv")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_yaw_height(**vars(args))
