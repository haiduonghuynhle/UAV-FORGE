import csv, math, glob, json, os, rasterio
root = "/workspace/forge/data/UAV_VisLoc_dataset"
rows = {r["mapname"].strip(): r for r in csv.DictReader(open(glob.glob(root + "/satellite*coordinates_range.csv")[0]))}
out = {}
for loc in sorted(d for d in os.listdir(root) if d.isdigit()):
    tifs = sorted(glob.glob(f"{root}/{loc}/satellite{loc}*.tif"))
    halves = []
    for t in tifs:
        with rasterio.open(t) as ds:
            W, H = ds.width, ds.height
            b = ds.bounds
            if ds.crs is not None and ds.crs.is_geographic:
                lon0, lat0, lon1, lat1 = b.left, b.bottom, b.right, b.top
            else:
                r = rows.get(os.path.basename(t))
                lat0, lat1 = float(r["RB_lat_map"]), float(r["LT_lat_map"])
                lon0, lon1 = float(r["LT_lon_map"]), float(r["RB_lon_map"])
        latm = (lat0 + lat1) / 2
        m_e = (lon1 - lon0) * 111320.0 * math.cos(math.radians(latm)) / W
        m_n = (lat1 - lat0) * 111320.0 / H
        halves.append((round(1024 * m_e, 1), round(1024 * m_n, 1), W, H))
    out[loc] = halves
    print(loc, halves)
json.dump(out, open("/workspace/forge/analysis/out/uav_visloc_tile_half_extent.json", "w"), indent=1)
