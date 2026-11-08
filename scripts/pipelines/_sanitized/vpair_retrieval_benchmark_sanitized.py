# Explicit sanitized source generated from _raw notebook export.
# Notebook bootstrap/magic cells were removed intentionally for local execution.

# ===== Code Cell 1 =====
import os
import math
import csv
from pathlib import Path

import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from tqdm import tqdm

import torch
import torchvision.transforms as T

print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", DEVICE)

# ===== Code Cell 2 =====
# ── Auto-detect VPAIR root (supports sample + VPAIR-full) ─────────────────────
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

# ── Benchmark config ──────────────────────────────────────────────────────────
TOP_K_VALUES       = [1, 5, 10]
DIST_THRESHOLD_M   = 25.0
USE_DISTRACTORS    = globals().get("USE_DISTRACTORS", True)
BATCH_SIZE         = 32

# ── Subset control ────────────────────────────────────────────────────────────
# Set to None to run full dataset.
MAX_IMAGES = globals().get("MAX_IMAGES", None)

for p in [QUERY_DIR, REF_DIR, DISTRACTOR_DIR, POSE_CSV, POSE_QUERY, POSE_REF]:
    status = "ok" if p.exists() else "missing"
    print(f"{status:>7}  {p}")

print(f"\nQueries     : {len(list(QUERY_DIR.glob('*.png')))}")
print(f"Ref views   : {len(list(REF_DIR.glob('*.png')))}")
print(f"Depth maps  : {len(list(REF_DIR.glob('*.npy')))}")
print(f"Distractors : {len(list(DISTRACTOR_DIR.glob('*.png')))}")
print(f"\nMAX_IMAGES  : {MAX_IMAGES}  ({'full run' if MAX_IMAGES is None else f'testing with {MAX_IMAGES} queries'})")


# ===== Code Cell 3 =====
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

    with open(filepath, newline="") as f:
        reader = csv.DictReader(f)
        print("poses.csv columns:", reader.fieldnames)

        for row in reader:
            path_raw = _pick_col(row, ["filepath", "path", "image", "img", "filename", "file"])
            if path_raw is None:
                continue
            stem = Path(path_raw).stem

            x = _to_float_or_none(_pick_col(row, ["x", "ecef_x", "tx"]))
            y = _to_float_or_none(_pick_col(row, ["y", "ecef_y", "ty"]))
            z = _to_float_or_none(_pick_col(row, ["z", "ecef_z", "tz"]))

            if any(v is None for v in [x, y, z]):
                lat = _to_float_or_none(_pick_col(row, ["lat", "latitude"]))
                lon = _to_float_or_none(_pick_col(row, ["lon", "lng", "longitude"]))
                alt = _to_float_or_none(_pick_col(row, ["alt", "altitude", "height", "z"]))
                if None not in [lat, lon]:
                    if alt is None:
                        alt = 0.0
                    ecef = geodetic_to_ecef_simple(lat, lon, alt)
                    x, y, z = float(ecef[0]), float(ecef[1]), float(ecef[2])

            if any(v is None for v in [x, y, z]):
                continue

            r = _to_float_or_none(_pick_col(row, ["roll", "r"]))
            p = _to_float_or_none(_pick_col(row, ["pitch", "p"]))
            yw = _to_float_or_none(_pick_col(row, ["yaw", "y", "heading"]))
            if r is None:
                r = 0.0
            if p is None:
                p = 0.0
            if yw is None:
                yw = 0.0

            entry = {
                "ecef": np.array([x, y, z], dtype=np.float64),
                "rpy": np.array([r, p, yw], dtype=np.float64),
                "filepath": path_raw,
            }

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

    return query_poses, ref_poses


def ecef_to_enu(ecef: np.ndarray, origin_ecef: np.ndarray) -> np.ndarray:
    a = 6_378_137.0
    e2 = 6.69437999014e-3

    ox, oy, oz = origin_ecef
    lon0 = math.atan2(oy, ox)
    p0 = math.sqrt(ox**2 + oy**2)
    lat0 = math.atan2(oz, p0 * (1 - e2))
    for _ in range(5):
        N = a / math.sqrt(1 - e2 * math.sin(lat0)**2)
        lat0 = math.atan2(oz + e2 * N * math.sin(lat0), p0)

    slat, clat = math.sin(lat0), math.cos(lat0)
    slon, clon = math.sin(lon0), math.cos(lon0)

    R = np.array([
        [-slon,        clon,       0],
        [-slat*clon,  -slat*slon,  clat],
        [ clat*clon,   clat*slon,  slat],
    ])
    delta = ecef - origin_ecef
    return R @ delta


if POSE_CSV.exists():
    query_poses, ref_poses = parse_poses_csv(POSE_CSV)
else:
    query_poses = parse_poses_txt(POSE_QUERY)
    ref_poses = parse_poses_txt(POSE_REF)

if len(ref_poses) == 0 and len(query_poses) > 0:
    ref_poses = {k: v for k, v in query_poses.items()}

if len(query_poses) == 0 and len(ref_poses) == 0:
    raise ValueError("No valid poses parsed from dataset metadata.")

origin_ecef = (next(iter(ref_poses.values())) if ref_poses else next(iter(query_poses.values())))["ecef"].copy()

for d in list(query_poses.values()) + list(ref_poses.values()):
    d["enu"] = ecef_to_enu(d["ecef"], origin_ecef)

print(f"Query poses   : {len(query_poses)}")
print(f"Ref poses     : {len(ref_poses)}")

if query_poses:
    stems = sorted(query_poses.keys())
    sample_stem = stems[0]
    print(f"Sample stem   : '{sample_stem}' -> ECEF {query_poses[sample_stem]['ecef']}")
    print(f"  ENU offset  : {query_poses[sample_stem]['enu'].round(1)} m")

missing = [s for s in query_poses.keys() if s not in ref_poses]
print(f"Queries without matching reference: {len(missing)}")


# ===== Code Cell 4 =====
# Load all reference image paths and depth maps lazily (indexed by stem)
ref_image_paths = {}
ref_depth_paths = {}

for p in sorted(REF_DIR.glob("*.png")):
    ref_image_paths[p.stem] = p

for p in sorted(REF_DIR.glob("depth_*.npy")):
    # depth_00001.npy → stem "00001"
    stem = p.stem.replace("depth_", "")
    ref_depth_paths[stem] = p

print(f"Reference images : {len(ref_image_paths)}")
print(f"Depth maps       : {len(ref_depth_paths)}")

# Quick stats on a sample depth map
sample_stem = sorted(ref_depth_paths.keys())[0]
d = np.load(ref_depth_paths[sample_stem])
print(f"\nDepth map '{sample_stem}': shape={d.shape}  dtype={d.dtype}")
print(f"  min={d.min():.1f} m   max={d.max():.1f} m   mean={d.mean():.1f} m")

# ===== Code Cell 5 =====
def list_images(folder: Path):
    exts = ["*.png", "*.jpg", "*.jpeg", "*.PNG", "*.JPG", "*.JPEG"]
    paths = []
    for ext in exts:
        paths.extend(folder.glob(ext))
    return sorted(set(paths))


query_image_paths = {p.stem: p for p in list_images(QUERY_DIR)}
distractor_image_paths = {p.stem: p for p in list_images(DISTRACTOR_DIR)}

query_stems = sorted(query_image_paths.keys())
if MAX_IMAGES is not None:
    query_stems = query_stems[:MAX_IMAGES]
    print(f"MAX_IMAGES={MAX_IMAGES}: using first {len(query_stems)} queries only")
print(f"Query images  : {len(query_image_paths)} total, {len(query_stems)} active")
print(f"Distractors   : {len(distractor_image_paths)}")
if query_stems:
    print(f"Sample query  : '{query_stems[0]}' -> GT ref '{query_stems[0]}'")

# Build the database: reference views [+ distractors]
ref_image_paths = {p.stem: p for p in list_images(REF_DIR)}
db_stems = sorted(ref_image_paths.keys())
db_paths = [ref_image_paths[s] for s in db_stems]
is_ref = [True] * len(db_stems)

if USE_DISTRACTORS:
    dist_stems = sorted(distractor_image_paths.keys())
    db_stems += dist_stems
    db_paths += [distractor_image_paths[s] for s in dist_stems]
    is_ref += [False] * len(dist_stems)

print(f"\nDatabase size : {len(db_stems)} "
      f"(refs={sum(is_ref)}, distractors={len(db_stems)-sum(is_ref)})")

# Ground-truth index for each query: index in db_stems of its correct ref
gt_db_idx = []
missing_gt = 0
for s in query_stems:
    if s in db_stems:
        gt_db_idx.append(db_stems.index(s))
    else:
        gt_db_idx.append(-1)
        missing_gt += 1

print(f"GT indices prepared: {len(gt_db_idx)} / {len(query_stems)}")
if missing_gt:
    print(f"Queries without exact GT stem in references: {missing_gt}")


# ===== Code Cell 6 =====
SHOW_STEM = "00001"

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Query
q_img = cv2.imread(str(query_image_paths[SHOW_STEM]))
q_img = cv2.cvtColor(q_img, cv2.COLOR_BGR2RGB)
q_pose = query_poses[SHOW_STEM]
axes[0].imshow(q_img)
axes[0].set_title(f"Query {SHOW_STEM}\n"
                  f"roll={np.degrees(q_pose['rpy'][0]):.2f}°  "
                  f"pitch={np.degrees(q_pose['rpy'][1]):.2f}°  "
                  f"yaw={np.degrees(q_pose['rpy'][2]):.2f}°", fontsize=9)
axes[0].axis("off")

# Reference
r_img = cv2.imread(str(ref_image_paths[SHOW_STEM]))
r_img = cv2.cvtColor(r_img, cv2.COLOR_BGR2RGB)
r_pose = ref_poses[SHOW_STEM]
axes[1].imshow(r_img)
axes[1].set_title(f"Reference view {SHOW_STEM}\n"
                  f"ENU offset: {r_pose['enu'].round(1)} m", fontsize=9)
axes[1].axis("off")

# Depth map
depth = np.load(ref_depth_paths[SHOW_STEM])
im = axes[2].imshow(depth, cmap="plasma")
plt.colorbar(im, ax=axes[2], label="Depth (m)")
axes[2].set_title(f"Depth map {SHOW_STEM}\n"
                  f"min={depth.min():.1f} m  max={depth.max():.1f} m", fontsize=9)
axes[2].axis("off")

plt.suptitle("VPAir Sample – Query | Reference View | Depth Map", fontsize=12, y=1.02)
plt.tight_layout()
plt.show()

# Position distance between query and reference (should be ~0 m for same stem)
dist = np.linalg.norm(q_pose["ecef"] - r_pose["ecef"])
print(f"ECEF distance query↔ref ({SHOW_STEM}): {dist:.2f} m")

# ===== Code Cell 7 =====
def recall_at_k(ranked_db_indices: list[list[int]],
                gt_db_idx: list[int],
                k: int) -> float:
    """
    R@k = fraction of queries whose GT index appears in the top-k retrieved.

    Parameters
    ----------
    ranked_db_indices : list of N lists – ranked db indices per query (best first)
    gt_db_idx         : list of N ground-truth db indices
    k                 : top-k cutoff
    """
    correct = sum(
        1 for ranks, gt in zip(ranked_db_indices, gt_db_idx)
        if gt in ranks[:k]
    )
    return correct / len(gt_db_idx)


def pose_errors(ranked_db_indices: list[list[int]],
                query_stems: list[str],
                db_stems: list[str],
                query_poses: dict,
                ref_poses: dict,
                k: int = 1) -> np.ndarray:
    """
    Compute ECEF position error (m) for the top-k retrieved reference.
    Only ref-view stems that exist in ref_poses are considered;
    distractors have no pose so they contribute inf error.
    Queries missing from query_poses also contribute inf error.
    """
    errors = []
    for ranks, qs in zip(ranked_db_indices, query_stems):
        if qs not in query_poses:
            errors.append(float("inf"))
            continue
        top_stem = db_stems[ranks[k-1]]
        if top_stem in ref_poses:
            err = np.linalg.norm(
                query_poses[qs]["ecef"] - ref_poses[top_stem]["ecef"]
            )
        else:
            err = float("inf")
        errors.append(err)
    return np.array(errors)


def compute_recall_table(ranked_db_indices, gt_db_idx, query_stems,
                          db_stems, query_poses, ref_poses,
                          k_values=TOP_K_VALUES):
    """Print recall and median position error table."""
    print(f"{'Metric':<22}{'Value':>10}")
    print("-" * 33)
    for k in k_values:
        r = recall_at_k(ranked_db_indices, gt_db_idx, k)
        errs = pose_errors(ranked_db_indices, query_stems, db_stems,
                           query_poses, ref_poses, k=1)
        finite = errs[np.isfinite(errs)]
        print(f"  Recall @ {k:<3}          {r*100:>8.1f} %")
    med_err = np.median(finite) if len(finite) > 0 else float("nan")
    mean_err = np.mean(finite) if len(finite) > 0 else float("nan")
    print(f"  Median pos error     {med_err:>8.1f} m")
    print(f"  Mean   pos error     {mean_err:>8.1f} m")

print("Metric helpers defined ✅")


# ===== Code Cell 8 =====
import sys, os
import time as _time

# ── Resolve repo root robustly (local Notebook/ subfolder OR Kaggle) ─────────
def _find_repo_root() -> str:
    marker = "Retrieval_Models"
    candidates = [
        os.path.abspath(".."),
        os.path.abspath("."),
        "/kaggle/working/FORGE-UAV",
        "/kaggle/working",
    ]
    for c in candidates:
        if os.path.isdir(os.path.join(c, marker)):
            return c
    raise RuntimeError(
        f"Cannot find repo root (looked for '{marker}' in: {candidates})\n"
        "Set REPO_ROOT manually below."
    )

REPO_ROOT = _find_repo_root()
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

print(f"Repo root      : {REPO_ROOT}")
print(f"Retrieval_Models: {os.path.isdir(os.path.join(REPO_ROOT, 'Retrieval_Models'))} ✅")
print(f"Matching_Models : {os.path.isdir(os.path.join(REPO_ROOT, 'Matching_Models'))} ✅")

from PIL import Image as PILImage
from Matching_Models.RoMa.demo.Roma_match import Roma_Init
from Retrieval_Models.CAMP.get_CAMP import get_CAMP_model

# ── Config ────────────────────────────────────────────────────────────────────
SHORTLIST_K   = 20     # Stage 1: keep top-K candidates per query for RoMa re-rank
ROMA_NUM_KPTS = 1000   # number of sampled matches per RoMa call (speed vs quality)

# ══════════════════════════════════════════════════════════════════════════════
# Stage 1 – CAMP global descriptors
# ConvNeXt-Base trained for cross-view drone↔satellite retrieval (U-1652)
# ══════════════════════════════════════════════════════════════════════════════
print("Loading CAMP model …")
camp_model = get_CAMP_model().to(DEVICE).eval()

CAMP_SIZE = 384   # CAMP input resolution

CAMP_TRANSFORM = T.Compose([
    T.ToPILImage(),
    T.Resize((CAMP_SIZE, CAMP_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std =[0.229, 0.224, 0.225]),
])


def _camp_forward(batch: torch.Tensor) -> torch.Tensor:
    """Run camp_model and always return a (B, D) tensor (handles tuple output)."""
    out = camp_model(batch)
    if isinstance(out, (tuple, list)):
        out = out[0]   # two_view_net returns (feat, ...) – take first element
    return out


@torch.no_grad()
def extract_camp_features(image_paths: list,
                           batch_size: int = 32) -> np.ndarray:
    """Run CAMP (ConvNeXt-Base); returns L2-normalised (N, D) float32 array."""
    feats = []
    for i in tqdm(range(0, len(image_paths), batch_size),
                  desc="CAMP", leave=False):
        imgs = []
        for p in image_paths[i : i + batch_size]:
            img = cv2.imread(str(p))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            imgs.append(CAMP_TRANSFORM(img))
        batch = torch.stack(imgs).to(DEVICE)
        out   = _camp_forward(batch)
        out   = torch.nn.functional.normalize(out, dim=1)
        feats.append(out.cpu().float().numpy())
    return np.concatenate(feats, axis=0)


print(f"Extracting CAMP features for {len(db_paths)} database images …")
_t_camp_db_start = _time.perf_counter()
db_feats_camp = extract_camp_features(db_paths)
_t_camp_db = _time.perf_counter() - _t_camp_db_start

print(f"Extracting CAMP features for {len(query_stems)} query images …")
_t_camp_q_start = _time.perf_counter()
q_feats_camp = extract_camp_features([query_image_paths[s] for s in query_stems])
_t_camp_q = _time.perf_counter() - _t_camp_q_start

print(f"DB feats: {db_feats_camp.shape}   Query feats: {q_feats_camp.shape}")

sim_camp  = q_feats_camp @ db_feats_camp.T
shortlist = np.argsort(-sim_camp, axis=1)[:, :SHORTLIST_K].tolist()
print(f"Stage 1 done – top-{SHORTLIST_K} shortlist per query ✅")
print(f"  CAMP DB  : {_t_camp_db:.2f} s  ({_t_camp_db/len(db_paths)*1000:.1f} ms/img)")
print(f"  CAMP Q   : {_t_camp_q:.2f} s  ({_t_camp_q/len(query_stems)*1000:.1f} ms/img)")

# ══════════════════════════════════════════════════════════════════════════════
# Stage 2 – RoMa dense-matching re-rank
# Also caches pixel-space keypoints for the top-1 match so the geometry
# solver (Cell 11) can estimate yaw and altitude without a second RoMa call.
# ══════════════════════════════════════════════════════════════════════════════
print("\nLoading RoMa model …")
roma_model = Roma_Init()


def _matches_to_pixels(matches_tensor: torch.Tensor,
                       W_q: int, H_q: int,
                       W_r: int, H_r: int):
    """
    Convert RoMa's normalised [-1,1] warp coordinates to pixel coordinates.
    matches_tensor : (N, 4)  [x_q, y_q, x_r, y_r] in [-1, 1]
    Returns (q_pts, r_pts) each (N, 2) float32 in pixel space.
    """
    m   = matches_tensor.float()
    q   = (m[:, :2] + 1.0) / 2.0 * torch.tensor([W_q, H_q], dtype=torch.float32)
    r   = (m[:, 2:] + 1.0) / 2.0 * torch.tensor([W_r, H_r], dtype=torch.float32)
    return q.numpy(), r.numpy()


print(f"\nStage 2 – RoMa re-ranking {len(query_stems)} × {SHORTLIST_K} = "
      f"{len(query_stems) * SHORTLIST_K} pairs …")

ranked_db_indices  = []
top1_match_pts     = {}   # query_stem → (q_pts, r_pts, best_db_idx)
_roma_per_query_t  = []   # per-query RoMa wall time (seconds)
_camp_per_query_t  = []   # per-query CAMP query extraction time (seconds)

_t_roma_start = _time.perf_counter()

for qi, qs in enumerate(tqdm(query_stems, desc="RoMa re-rank")):
    _tq = _time.perf_counter()

    # ── Measure per-query CAMP extraction time ────────────────────────────────
    _t_camp_one_start = _time.perf_counter()
    with torch.no_grad():
        _img_one = cv2.imread(str(query_image_paths[qs]))
        _img_one = cv2.cvtColor(_img_one, cv2.COLOR_BGR2RGB)
        _batch_one = CAMP_TRANSFORM(_img_one).unsqueeze(0).to(DEVICE)
        _feat_one  = _camp_forward(_batch_one)
    _camp_per_query_t.append(_time.perf_counter() - _t_camp_one_start)
    # ── (use pre-extracted features for shortlist, measured above) ────────────

    img_q        = cv2.imread(str(query_image_paths[qs]))
    H_q, W_q     = img_q.shape[:2]
    pil_q        = PILImage.fromarray(cv2.cvtColor(img_q, cv2.COLOR_BGR2RGB))
    cand_indices = shortlist[qi]

    scores       = {}
    kpts_cache   = {}   # db_idx → matches tensor (N,4)

    for db_idx in cand_indices:
        img_r      = cv2.imread(str(db_paths[db_idx]))
        pil_r      = PILImage.fromarray(cv2.cvtColor(img_r, cv2.COLOR_BGR2RGB))
        with torch.no_grad():
            warp, cert      = roma_model.match(pil_q, pil_r, device=DEVICE)
            matches, cert_s = roma_model.sample(warp, cert, num=ROMA_NUM_KPTS)
        scores[db_idx]     = float(cert_s.mean().cpu())
        kpts_cache[db_idx] = matches.cpu()   # save for geometry solver

    reranked_top = sorted(cand_indices, key=lambda i: scores[i], reverse=True)
    remaining    = [i for i in np.argsort(-sim_camp[qi]).tolist()
                    if i not in set(cand_indices)]
    ranked_db_indices.append(reranked_top + remaining)

    # Cache pixel-space keypoints for the top-1 candidate
    best_idx     = reranked_top[0]
    img_r_best   = cv2.imread(str(db_paths[best_idx]))
    H_r, W_r     = img_r_best.shape[:2]
    q_pts, r_pts = _matches_to_pixels(kpts_cache[best_idx], W_q, H_q, W_r, H_r)
    top1_match_pts[qs] = (q_pts, r_pts, best_idx)

    _roma_per_query_t.append(_time.perf_counter() - _tq)

_t_roma_total = _time.perf_counter() - _t_roma_start
_roma_per_query_t  = np.array(_roma_per_query_t)
_camp_per_query_t  = np.array(_camp_per_query_t)

print(f"Stage 2 done ✅  ({len(top1_match_pts)} top-1 match sets cached)")
print(f"  RoMa total : {_t_roma_total:.2f} s")
print(f"  RoMa/query : mean={_roma_per_query_t.mean():.2f} s  "
      f"median={np.median(_roma_per_query_t):.2f} s  "
      f"min={_roma_per_query_t.min():.2f} s  max={_roma_per_query_t.max():.2f} s")
print(f"  CAMP/query : mean={_camp_per_query_t.mean()*1000:.1f} ms  "
      f"median={np.median(_camp_per_query_t)*1000:.1f} ms")

# Store timing for pipeline summary in Cell 12
pipeline_timing = {
    "camp_db_s":          _t_camp_db,
    "camp_q_s":           _t_camp_q,
    "camp_per_query_arr": _camp_per_query_t,
    "roma_total_s":       _t_roma_total,
    "roma_per_query_arr": _roma_per_query_t,
    "n_db":  len(db_paths),
    "n_q":   len(query_stems),
    "shortlist_k": SHORTLIST_K,
}


# ===== Code Cell 9 =====
# Stage 1 ranking (CAMP only) for comparison
stage1_ranked = np.argsort(-sim_camp, axis=1).tolist()

print("=" * 45)
print("  Stage 1 – CAMP global descriptor only")
print("=" * 45)
compute_recall_table(stage1_ranked, gt_db_idx,
                     query_stems, db_stems,
                     query_poses, ref_poses)

print()
print("=" * 45)
print("  Stage 2 – RoMa re-ranked (top-{})".format(SHORTLIST_K))
print("=" * 45)
compute_recall_table(ranked_db_indices, gt_db_idx,
                     query_stems, db_stems,
                     query_poses, ref_poses)

# ── Per-threshold recall curve (Stage 2) ─────────────────────────────────────
top1_errors   = pose_errors(ranked_db_indices, query_stems, db_stems,
                             query_poses, ref_poses, k=1)
finite_errors = top1_errors[np.isfinite(top1_errors)]

thresholds    = np.arange(0, 200, 1)
recall_curve  = [np.mean(finite_errors <= t) for t in thresholds]

# Stage 1 curve for overlay
top1_s1      = pose_errors(stage1_ranked, query_stems, db_stems,
                            query_poses, ref_poses, k=1)
finite_s1    = top1_s1[np.isfinite(top1_s1)]
recall_s1    = [np.mean(finite_s1 <= t) for t in thresholds]

print("\n── Top-1 position error (RoMa re-ranked) ──")
for p in [25, 50, 75, 100]:
    print(f"  Recall @ {p:3d} m : {np.mean(finite_errors <= p)*100:.1f} %")


# ===== Code Cell 10 =====
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# ── 1. Recall@k bar chart – Stage 1 vs Stage 2 ───────────────────────────────
x      = np.arange(len(TOP_K_VALUES))
width  = 0.35
r1     = [recall_at_k(stage1_ranked,     gt_db_idx, k) * 100 for k in TOP_K_VALUES]
r2     = [recall_at_k(ranked_db_indices, gt_db_idx, k) * 100 for k in TOP_K_VALUES]
bars1  = axes[0].bar(x - width/2, r1, width, label="CAMP only",       color="#90CAF9")
bars2  = axes[0].bar(x + width/2, r2, width, label="+ RoMa re-rank",  color="#1565C0")
axes[0].set_xticks(x); axes[0].set_xticklabels([f"R@{k}" for k in TOP_K_VALUES])
axes[0].set_ylim(0, 108); axes[0].set_ylabel("Recall (%)")
axes[0].set_title("Recall @ K")
axes[0].legend()
for bar, val in list(zip(bars1, r1)) + list(zip(bars2, r2)):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                 f"{val:.0f}%", ha="center", va="bottom", fontsize=9)

# ── 2. Position error histogram  ─────────────────────────────────────────────
axes[1].hist(finite_errors, bins=30, color="#1565C0", edgecolor="white",
             alpha=0.75, label="RoMa re-ranked")
axes[1].hist(finite_s1,     bins=30, color="#90CAF9", edgecolor="white",
             alpha=0.55, label="CAMP only")
axes[1].axvline(np.median(finite_errors), color="red",    linestyle="--",
                label=f"Median {np.median(finite_errors):.1f} m (RoMa)")
axes[1].axvline(DIST_THRESHOLD_M,         color="orange", linestyle=":",
                label=f"Threshold {DIST_THRESHOLD_M} m")
axes[1].set_xlabel("Top-1 Position Error (m)")
axes[1].set_ylabel("Count")
axes[1].set_title("Top-1 Position Error Distribution")
axes[1].legend(fontsize=8)

# ── 3. Recall vs distance-threshold curve ────────────────────────────────────
axes[2].plot(thresholds, [r * 100 for r in recall_s1],
             color="#90CAF9", linewidth=2, label="CAMP only (Stage 1)")
axes[2].plot(thresholds, [r * 100 for r in recall_curve],
             color="#1565C0", linewidth=2, label=f"+ RoMa re-rank top-{SHORTLIST_K}")
axes[2].axvline(DIST_THRESHOLD_M, color="orange", linestyle="--",
                alpha=0.7, label=f"Default threshold ({DIST_THRESHOLD_M} m)")
axes[2].set_xlabel("Distance threshold (m)")
axes[2].set_ylabel("Recall (%)")
axes[2].set_title("Recall vs Distance Threshold (Top-1)")
axes[2].legend(fontsize=8); axes[2].grid(alpha=0.3)

plt.suptitle("VPAir Benchmark – CAMP + RoMa Re-ranking", fontsize=13)
plt.tight_layout()
plt.show()

# ── 4. 2-D ENU map ───────────────────────────────────────────────────────────
fig2, ax = plt.subplots(figsize=(9, 7))
q_enu = np.array([query_poses[s]["enu"] for s in query_stems])
r_enu = np.array([ref_poses[s]["enu"]   for s in sorted(ref_poses)])

ax.scatter(r_enu[:, 0], r_enu[:, 1], c="steelblue", s=20, alpha=0.8,
           label="Reference views", zorder=3)
ax.scatter(q_enu[:, 0], q_enu[:, 1], c="tomato",    s=20, alpha=0.8,
           label="Queries", zorder=3)

rng = np.random.default_rng(42)
for i in rng.choice(len(query_stems), min(40, len(query_stems)), replace=False):
    top1_stem = db_stems[ranked_db_indices[i][0]]
    if top1_stem in ref_poses:
        rx, ry = ref_poses[top1_stem]["enu"][:2]
        color  = "green" if ranked_db_indices[i][0] == gt_db_idx[i] else "red"
        ax.plot([q_enu[i, 0], rx], [q_enu[i, 1], ry],
                color=color, alpha=0.5, linewidth=0.9)

import matplotlib.patches as mpatches
handles = [
    mpatches.Patch(color="steelblue", label="References"),
    mpatches.Patch(color="tomato",    label="Queries"),
    mpatches.Patch(color="green",     label="Correct top-1"),
    mpatches.Patch(color="red",       label="Wrong top-1"),
]
ax.legend(handles=handles, fontsize=9)
ax.set_xlabel("East (m)"); ax.set_ylabel("North (m)")
ax.set_title("2-D ENU Map – RoMa Re-ranked Top-1 Retrievals")
ax.set_aspect("equal")
plt.tight_layout()
plt.show()


# ===== Code Cell 11 =====
import csv as _csv
import time as _time
from uav_geolocalization.baselines.native_anyvisloc.geometry_solver import extract_yaw_and_height

# ── Guard: reload depth maps if Cell 9 wasn't run in this session ─────────────
if not ref_depth_paths:
    print("ref_depth_paths is empty - reloading depth maps from disk ...")
    for _p in sorted(REF_DIR.glob("depth_*.npy")):
        ref_depth_paths[_p.stem.replace("depth_", "")] = _p
print(f"Depth maps loaded : {len(ref_depth_paths)}")
if not ref_depth_paths:
    raise RuntimeError(
        f"No depth_*.npy files found under {REF_DIR}.\n"
        "Check that VPAIR_ROOT points to the correct dataset folder."
    )

# ── Only evaluate queries that have a pose entry ──────────────────────────────
posed_stems = [s for s in query_stems if s in query_poses]
print(f"Queries with pose : {len(posed_stems)} / {len(query_stems)}")
if len(posed_stems) < len(query_stems):
    missing_pose = [s for s in query_stems if s not in query_poses]
    print(f"  Stems missing from pose file: {missing_pose}")

# ── Estimate reference GSD from depth maps ───────────────────────────────────
_sample_depth_alts = []
for _s in list(ref_depth_paths.keys())[:20]:
    _d = np.load(str(ref_depth_paths[_s]))
    _sample_depth_alts.append(float(np.median(_d)))
_median_depth_alt = float(np.median(_sample_depth_alts)) if _sample_depth_alts else None

if _median_depth_alt is not None:
    print(f"\nDepth-map median altitude    : {_median_depth_alt:.1f} m")
else:
    raise RuntimeError("depth maps were found but could not be loaded - check file format")

# ── Camera / sensor config ────────────────────────────────────────────────────
Q_W, Q_H   = 800, 600
FOCAL_PX_Q = 550.0

GSD_REF_MPX = _median_depth_alt / FOCAL_PX_Q
print(f"\nGSD_REF_MPX      : {GSD_REF_MPX:.4f} m/px")

camera_K = np.array([
    [FOCAL_PX_Q,  0.,         Q_W / 2.],
    [0.,          FOCAL_PX_Q, Q_H / 2.],
    [0.,          0.,         1.      ],
], dtype=np.float64)

# ── Run geometry solver on every query's top-1 RoMa match ────────────────────
records         = []
per_img_times   = []
geo_stem_order  = []
_cell_start     = _time.perf_counter()

for qs in tqdm(posed_stems, desc="Geo-solve"):
    _t0 = _time.perf_counter()

    if qs not in top1_match_pts:
        per_img_times.append(_time.perf_counter() - _t0)
        geo_stem_order.append(qs)
        records.append({"stem": qs, "est_yaw": None, "est_alt": None,
                        "n_inliers": 0, "proc_time_s": per_img_times[-1]})
        continue

    q_pts, r_pts, best_db_idx = top1_match_pts[qs]
    ref_img  = cv2.imread(str(db_paths[best_db_idx]))
    R_H, R_W = ref_img.shape[:2]

    yaw_deg, alt_m, n_inl = extract_yaw_and_height(
        query_pts  = q_pts,
        ref_pts    = r_pts,
        camera_K   = camera_K,
        img_width  = Q_W,
        img_height = Q_H,
        ref_cut_W  = R_W,
        ref_cut_H  = R_H,
        gsd        = GSD_REF_MPX,
    )

    if yaw_deg is not None and n_inl >= 8:
        best_ref_stem = db_stems[best_db_idx]
        if best_ref_stem in ref_depth_paths:
            _H_raw, _ = cv2.findHomography(
                q_pts.astype(np.float32), r_pts.astype(np.float32),
                cv2.RANSAC, 5.0)
            if _H_raw is not None:
                _q_ctr  = np.array([[[Q_W / 2.0, Q_H / 2.0]]], dtype=np.float32)
                _r_ctr  = cv2.perspectiveTransform(_q_ctr, _H_raw)[0, 0]
                _depth  = np.load(str(ref_depth_paths[best_ref_stem]))
                _vc = int(np.clip(_r_ctr[1], 0, _depth.shape[0] - 1))
                _uc = int(np.clip(_r_ctr[0], 0, _depth.shape[1] - 1))
                alt_m = float(_depth[_vc, _uc])

    t_elapsed = _time.perf_counter() - _t0
    per_img_times.append(t_elapsed)
    geo_stem_order.append(qs)
    records.append({"stem": qs, "est_yaw": yaw_deg, "est_alt": alt_m,
                    "n_inliers": n_inl, "proc_time_s": t_elapsed})

_t_geo_total  = _time.perf_counter() - _cell_start
per_img_times = np.array(per_img_times)

# ── Ground-truth yaw and height ───────────────────────────────────────────────
gt_yaw_deg = {s: float(np.degrees(query_poses[s]["rpy"][2]) % 360) for s in posed_stems}
gt_alt_m = {
    s: float(np.median(np.load(str(ref_depth_paths[s]))))
    if s in ref_depth_paths
    else float(query_poses[s]["enu"][2])
    for s in posed_stems
}

# ── Error computation ─────────────────────────────────────────────────────────
yaw_errs, alt_errs = [], []
for rec in records:
    s = rec["stem"]
    if rec["est_yaw"] is not None:
        d = abs(rec["est_yaw"] - gt_yaw_deg[s]) % 360
        yaw_errs.append(min(d, 360.0 - d))
    if rec["est_alt"] is not None:
        alt_errs.append(abs(rec["est_alt"] - gt_alt_m[s]))

yaw_errs = np.array(yaw_errs)
alt_errs = np.array(alt_errs)
n_solved = sum(1 for r in records if r["est_yaw"] is not None)

# Position errors from retrieval stage (top-1). Recompute if needed.
if "finite_errors" not in globals():
    top1_errors = pose_errors(ranked_db_indices, query_stems, db_stems, query_poses, ref_poses, k=1)
    finite_errors = top1_errors[np.isfinite(top1_errors)]

ACC_THRESHOLDS = list(range(5, 101, 5))

def _print_acc_table(arr: np.ndarray, unit: str):
    if len(arr) == 0:
        print("  Accuracy: unavailable")
        return
    rows = [f"A@{t}{unit}={100*np.mean(arr <= t):.1f}%" for t in ACC_THRESHOLDS]
    for i in range(0, len(rows), 5):
        print("  " + "  ".join(rows[i:i+5]))

print("\n" + "=" * 72)
print("  VPAir Unified Metrics")
print("=" * 72)
print(f"Total queries: {len(query_stems)}  |  Yaw/Height solved: {n_solved}")

print("\n[Position Error Top-1]")
if len(finite_errors) > 0:
    print(f"  mean={np.mean(finite_errors):.1f} m  median={np.median(finite_errors):.1f} m")
_print_acc_table(finite_errors, "m")

print("\n[Yaw Error]")
if len(yaw_errs) > 0:
    print(f"  mean={np.mean(yaw_errs):.1f} deg  median={np.median(yaw_errs):.1f} deg")
_print_acc_table(yaw_errs, "deg")

print("\n[Height Error]")
if len(alt_errs) > 0:
    print(f"  mean={np.mean(alt_errs):.1f} m  median={np.median(alt_errs):.1f} m")
_print_acc_table(alt_errs, "m")

# ══════════════════════════════════════════════════════════════════════════════
# Full pipeline timing summary
# ══════════════════════════════════════════════════════════════════════════════
pt      = pipeline_timing
nq      = pt["n_q"]
ndb     = pt["n_db"]
k       = pt["shortlist_k"]
roma_pq = pt["roma_per_query_arr"]
camp_pq = pt["camp_per_query_arr"]

geo_t_map = {s: t for s, t in zip(geo_stem_order, per_img_times)}
geo_pq    = np.array([geo_t_map.get(s, 0.0) for s in query_stems])

full_per_query = camp_pq + roma_pq + geo_pq

print("\n" + "=" * 60)
print("  Full Pipeline Timing Summary")
print("=" * 60)
print(f"  [ONLINE] TOTAL per query: mean {full_per_query.mean()*1000:.0f} ms  med {np.median(full_per_query)*1000:.0f} ms")

# ── Save per-query results to CSV ─────────────────────────────────────────────
CSV_OUT = Path(REPO_ROOT) / "vpair_yaw_height_results.csv"

with open(CSV_OUT, "w", newline="") as f:
    writer = _csv.DictWriter(f, fieldnames=[
        "stem",
        "gt_yaw_deg", "est_yaw_deg", "yaw_err_deg",
        "gt_alt_m",   "est_alt_m",   "alt_err_m",
        "n_inliers", "pos_err_m",
        "camp_q_ms", "roma_ms", "geo_ms", "total_online_ms",
        "top1_db_stem", "top1_correct",
    ])
    writer.writeheader()

    q_to_top1  = {qs: ranked_db_indices[qi][0] for qi, qs in enumerate(query_stems)}
    # per-frame top-1 position error (same array the summary prints)
    pos_err_map = {qs: top1_errors[qi] for qi, qs in enumerate(query_stems)}
    camp_t_map = {s: t for s, t in zip(query_stems, camp_pq)}
    roma_t_map = {s: t for s, t in zip(query_stems, roma_pq)}

    for rec in records:
        s     = rec["stem"]
        gt_y  = gt_yaw_deg[s]
        gt_a  = gt_alt_m[s]
        est_y = rec["est_yaw"]
        est_a = rec["est_alt"]

        if est_y is not None:
            d     = abs(est_y - gt_y) % 360
            y_err = min(d, 360.0 - d)
        else:
            y_err = ""

        a_err        = abs(est_a - gt_a) if est_a is not None else ""
        top1_idx     = q_to_top1.get(s)
        top1_stem    = db_stems[top1_idx] if top1_idx is not None else ""
        qi_idx       = query_stems.index(s) if s in query_stems else None
        top1_correct = (top1_idx == gt_db_idx[qi_idx]) if qi_idx is not None else ""

        c_ms = camp_t_map.get(s, 0.0) * 1000
        r_ms = roma_t_map.get(s, 0.0) * 1000
        g_ms = geo_t_map.get(s, 0.0)  * 1000

        writer.writerow({
            "stem":             s,
            "gt_yaw_deg":       f"{gt_y:.4f}",
            "est_yaw_deg":      f"{est_y:.4f}" if est_y is not None else "",
            "yaw_err_deg":      f"{y_err:.4f}" if y_err != "" else "",
            "gt_alt_m":         f"{gt_a:.4f}",
            "est_alt_m":        f"{est_a:.4f}" if est_a is not None else "",
            "alt_err_m":        f"{a_err:.4f}" if a_err != "" else "",
            "n_inliers":        rec["n_inliers"],
            "pos_err_m":        (f"{pos_err_map[s]:.4f}" if s in pos_err_map and np.isfinite(pos_err_map[s]) else ""),
            "camp_q_ms":        f"{c_ms:.2f}",
            "roma_ms":          f"{r_ms:.2f}",
            "geo_ms":           f"{g_ms:.2f}",
            "total_online_ms":  f"{c_ms + r_ms + g_ms:.2f}",
            "top1_db_stem":     top1_stem,
            "top1_correct":     top1_correct,
        })

print(f"\nResults saved -> {CSV_OUT}")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].hist(yaw_errs, bins=36, range=(0, 180), color="#1565C0", edgecolor="white", alpha=0.8)
axes[0].set_xlabel("Yaw Error (deg)")
axes[0].set_ylabel("Count")
axes[0].set_title("Yaw Estimation Error")

axes[1].hist(alt_errs, bins=30, color="#2E7D32", edgecolor="white", alpha=0.8)
axes[1].set_xlabel("Height Error (m)")
axes[1].set_ylabel("Count")
axes[1].set_title("Height Estimation Error")
plt.tight_layout()
plt.show()
