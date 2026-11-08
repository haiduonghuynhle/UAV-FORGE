#!/usr/bin/env python3
"""Build the VPAIR held-out query list with a geometric buffer.

Development frames are the first 800 queries in the order the adapter sorts
them (the record's population). A later frame is held out only if its
footprint cannot overlap any development footprint: footprints are bounded by
the circle circumscribing the 336 x 252 m view (radius 210 m, heading-free),
so two footprints can overlap only if their centres are closer than
2 x 210 m plus the 15 m georeferencing slack. Computed from poses alone,
before any held-out run.
"""
import csv, json, math, sys
from pathlib import Path

root = Path(sys.argv[1]); out = Path(sys.argv[2])
rows = list(csv.DictReader(open(root / "poses.csv")))
key = [k for k in rows[0] if k.lower() in ("filename", "file", "image", "name")][0]
lat = [k for k in rows[0] if k.lower().startswith("lat")][0]
lon = [k for k in rows[0] if k.lower().startswith("lon")][0]
rows.sort(key=lambda r: r[key])
R = 2 * math.hypot(168.0, 126.0) + 15.0
lat0 = float(rows[0][lat]); kx = 111320.0 * math.cos(math.radians(lat0)); ky = 111320.0
pts = [((float(r[lon]) * kx), (float(r[lat]) * ky)) for r in rows]
dev, rest = pts[:800], list(range(800, len(rows)))
keep = []
for i in rest:
    x, y = pts[i]
    if min(math.hypot(x - a, y - b) for a, b in dev) > R:
        keep.append(Path(rows[i][key]).name)
out.write_text("# VPAIR held-out queries, geometric buffer %.0f m from frames 1-800\n" % R + "\n".join(keep) + "\n")
print(json.dumps({"total": len(rows), "candidates_801_plus": len(rest), "kept": len(keep), "buffer_m": R}))
