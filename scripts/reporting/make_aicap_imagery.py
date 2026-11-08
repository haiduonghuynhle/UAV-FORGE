#!/usr/bin/env python3
"""Cut the three pictures behind the qualitative panel of the AICAP figure.

A reviewer of an earlier version asked to see what an injected wrong
reference looks like. This script produces that view for one recorded trial:

  case_query.png   the camera frame the UAV recorded
  case_true.png    the map tile over the aircraft's true position
  case_forced.png  the tile the evaluator forced into the pipeline

It reads the trial from the recorded prediction rows, so the pictures and the
numbers in the caption describe the same frame. A provenance file records the
dataset paths, the pixel windows and the digests of the files that were read.

UAV-VisLoc stores one GeoTIFF per flight with the corner coordinates of the
raster, so a position maps to a pixel by linear interpolation between those
corners. Tile names carry their own pixel window, as satellite01_rRRRRRR_cCCCCCC.

Usage:
    python3 scripts/reporting/make_aicap_imagery.py \
        --dataset ~/Downloads/UAV_VisLoc_dataset \
        --campaign results_server/v3_forge5050b_mixed \
        --cell uav_visloc_wrong_far --frame 01/01_0598.JPG \
        --out Report/Paper/aicap-2027
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path

from PIL import Image

Image.MAX_IMAGE_PIXELS = None  # the flight rasters are far over the default guard

TILE_PX = 2048  # tile side used by the evaluator; see \protoVislocTilePx
OUT_PX = 420    # side of each published picture


def sha256(path: Path, limit: int = 64 << 20) -> str:
    """Digest of the file, or of its first `limit` bytes for a large raster."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        h.update(fh.read(limit))
    return h.hexdigest()


def map_bounds(dataset: Path, map_name: str) -> tuple[float, float, float, float]:
    csv_path = next(p for p in dataset.glob("satellite*coordinates_range.csv"))
    with csv_path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row["mapname"] == map_name:
                return (
                    float(row["LT_lat_map"]),
                    float(row["LT_lon_map"]),
                    float(row["RB_lat_map"]),
                    float(row["RB_lon_map"]),
                )
    raise SystemExit(f"{map_name} not listed in {csv_path}")


def square(img: Image.Image) -> Image.Image:
    w, h = img.size
    side = min(w, h)
    left, top = (w - side) // 2, (h - side) // 2
    return img.crop((left, top, left + side, top + side))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, type=Path)
    ap.add_argument("--campaign", required=True, type=Path)
    ap.add_argument("--cell", required=True)
    ap.add_argument("--frame", required=True, help="image_name as the prediction row spells it")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    dataset = args.dataset.expanduser()
    pred_path = args.campaign / args.cell / "predictions.csv"
    with pred_path.open(newline="", encoding="utf-8") as fh:
        row = next(r for r in csv.DictReader(fh) if r["image_name"] == args.frame)
    if row["accepted_for_metrics"] != "True" or row["is_wrong_reference"] not in ("True", "1"):
        raise SystemExit("that frame is not an accepted wrong-reference trial")
    anchor = json.loads(row["geodetic_anchor"])

    flight = args.frame.split("/")[0]
    stem = Path(args.frame).stem
    raster = dataset / flight / f"satellite{flight}.tif"
    lt_lat, lt_lon, rb_lat, rb_lon = map_bounds(dataset, raster.name)

    sat = Image.open(raster)
    width, height = sat.size

    def pixel_of(lat: float, lon: float) -> tuple[float, float]:
        x = (lon - lt_lon) / (rb_lon - lt_lon) * width
        y = (lat - lt_lat) / (rb_lat - lt_lat) * height
        return x, y

    gx, gy = pixel_of(float(row["gt_lat"]), float(row["gt_lon"]))
    true_box = (
        int(gx - TILE_PX / 2),
        int(gy - TILE_PX / 2),
        int(gx + TILE_PX / 2),
        int(gy + TILE_PX / 2),
    )
    forced_name = anchor["_adversarial_forced_reference_name"].split("/")[-1]
    m = re.search(r"_r(\d+)_c(\d+)", forced_name)
    if not m:
        raise SystemExit(f"cannot read a pixel window out of {forced_name}")
    r0, c0 = int(m.group(1)), int(m.group(2))
    forced_box = (c0, r0, c0 + TILE_PX, r0 + TILE_PX)

    out_dir = args.out / "figures"
    drone = dataset / flight / "drone" / f"{stem}.JPG"
    pictures = {
        "case_query.png": square(Image.open(drone)),
        "case_true.png": sat.crop(true_box),
        "case_forced.png": sat.crop(forced_box),
    }
    for name, img in pictures.items():
        img.convert("RGB").resize((OUT_PX, OUT_PX), Image.LANCZOS).save(out_dir / name, optimize=True)

    prov = {
        "generator": "scripts/reporting/make_aicap_imagery.py",
        "dataset_root": str(dataset),
        "predictions": {"path": str(pred_path), "sha256": sha256(pred_path)},
        "frame": args.frame,
        "cell": args.cell,
        "raster": {"path": str(raster), "size_px": [width, height], "sha256_first_64mb": sha256(raster)},
        "map_corners": {"lt": [lt_lat, lt_lon], "rb": [rb_lat, rb_lon]},
        "windows_px": {"true": list(true_box), "forced": list(forced_box), "tile_px": TILE_PX},
        "forced_reference": anchor["_adversarial_forced_reference_name"],
        "retrieved_reference": anchor["_adversarial_original_reference_name"],
        "offset_m": anchor["_adversarial_offset_m"],
        "error_m": float(row["post_fgo_error_m"]),
        "drone_image": str(drone),
    }
    (out_dir / "case_imagery.provenance.json").write_text(json.dumps(prov, indent=2) + "\n", encoding="utf-8")
    for name in pictures:
        print(f"wrote {out_dir / name}")
    print(f"wrote {out_dir / 'case_imagery.provenance.json'}")


if __name__ == "__main__":
    main()
