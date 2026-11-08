"""Automated sanity gate for FORGE cells — catches the failure classes that
forced re-runs on 04-05/08 before they burn GPU hours.

Each documented failure this gate would have caught, and the invariant that
encodes it:

  I1  zero/starved injection      forgec_avl gen3: 0 of 487 frames injected in
                                  three full cells (map-restriction bug).
  I2  cross-map injection         first forgeb_visloc launch: queries forced to
                                  references 461 km away on another flight.
  I3  condition collapse          forgeb_avl gen1: proximal/distal/visually-
                                  similar medians 369/378/370 m — one candidate
                                  pool, three labels.
  I4  falsification counterexample (WARN; counted, never holds a cell)
                                  classed unrecoverable (189/28/15 cases).
  I5  degenerate candidate pool   every injection drawn from a pool of one.
  I6  summary/prediction drift    accepted counts in metrics_summary must match
                                  the per-frame records the tables are built on.
  I7  dead heading channel        native OrthoLoC rerun emitting pred_heading=0
                                  on every frame while status said "ok".
  I8  gate config drift           mixed 50/50 campaign 18/08 launched without the
                                  gate block of record (yaw cross-check off,
                                  post-LGI inlier minimum 6, unthresholded rerank);
                                  nine cells and a week of analysis voided.

Modes:
  --early   read frame_diagnostics.jsonl of a RUNNING cell (>=150 frames);
            hard invariants only (I1, I2). Exit 1 = abort the run now.
  --final   full predictions.csv; all invariants. Exit 1 = do not let this
            cell into the tables.

WARN never changes the exit code; it is for the morning read of the log.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from gen_v3_tables import selected_reference_can_localize, DEFAULT_RADIUS_M  # noqa: E402

# Minimum acceptable injected fraction per dataset (final mode). Derived from
# the healthy cells: avl 84/487=0.17, vpair 536/800=0.67, uav_visloc 1703/2000=0.85.
MIN_YIELD = {"avl": 0.08, "vpair": 0.30, "uav_visloc": 0.50}
EARLY_MIN_ROWS = 150

# I8 — gate configuration of record, read from run_manifest.json. Nothing in
# I1-I7 looked at the manifest, which is how the 18/08 mixed campaign ran with
# the two proposed checks off and nobody noticed until the numbers disagreed.
GATES_OF_RECORD = {
    "rerank_margin_score": 50.0,
    "final_pose_max_yaw_prior_residual_deg": 30.0,
    "geometry_min_inliers_after_lgi": 25,
}
TWO_GATES = ("final_pose_max_yaw_prior_residual_deg", "geometry_min_inliers_after_lgi")
TOPK_OF_RECORD = {"avl": 5, "vpair": 5, "uav_visloc": 10}
VISLOC_OF_RECORD = {"fgo_max_step_xy_m": 40.0, "fgo_motion_prior_sigma_pos_m": 80.0}
GATED_PIPELINES = ("ours_ultravpr_cascade_6dof_e2e", "ours_camp_cascade_6dof_e2e")
VISLOC_LOCATIONS_OF_RECORD = ["01", "04", "11"]
# Held-out campaigns (E1, E1b): the gates of record on flights or frames that
# threshold selection never saw, with the frozen footprint inputs present.
VISLOC_HELDOUT_FLIGHTS = {"02", "03", "05", "06", "08", "09", "10"}
I8_PROFILES = ("record", "gates-off", "heldout")


def load_manifest(run_dir: Path):
    """None = absent; a dict = parsed. An unreadable or structurally incomplete
    manifest is returned as a dict carrying `_invalid` so I8 can FAIL on it
    instead of treating it as 'not applicable'."""
    p = run_dir / "run_manifest.json"
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text())
    except ValueError:
        return {"_invalid": "run_manifest.json is not valid JSON"}
    if not isinstance(data, dict):
        return {"_invalid": "run_manifest.json is not a JSON object"}
    for key in ("pipeline", "dataset", "pipeline_config"):
        if key not in data:
            return {"_invalid": f"run_manifest.json lacks '{key}'", **data}
    if not isinstance(data.get("pipeline_config"), dict):
        return {"_invalid": "pipeline_config is not an object", **data}
    return data


def _num_eq(got, want) -> bool:
    try:
        return float(got) == float(want)
    except (TypeError, ValueError):
        return False


def _locations(value) -> list:
    if isinstance(value, (list, tuple)):
        return [str(v).strip() for v in value]
    return [v.strip() for v in str(value or "").split(",") if v.strip()]


def check_gate_config(manifest: dict, ds: str, profile: str = "record",
                      dir_dataset: str = ""):
    """I8. Returns (level, message) with level in INFO / PASS / FAIL.

    record    : the three gate keys, top-k and (UAV-VisLoc) flight/FGO keys must
                equal deploy/run_v3_forge.sh.
    gates-off : the two proposed checks must be absent, everything else of record.
    heldout   : gates of record; UAV-VisLoc runs exactly one held-out flight with
                the per-frame footprint table and per-map extents set; VPAIR runs
                from a query name list.
    Wrappers, natives and retrieval-only cells are not gated cascades: INFO.
    """
    if profile not in I8_PROFILES:
        raise ValueError(f"unknown I8 profile {profile!r}")
    pc = manifest.get("pipeline_config") or {}
    pipeline = str(manifest.get("pipeline") or pc.get("name") or "")
    if (pipeline not in GATED_PIPELINES or pc.get("is_wrapper") or pc.get("is_native")
            or manifest.get("retrieval_only")):
        return "INFO", f"I8 not applicable to pipeline {pipeline!r}"
    bad = []
    man_ds = str(manifest.get("dataset") or "")
    if dir_dataset and man_ds and man_ds != dir_dataset:
        bad.append(f"manifest dataset {man_ds!r} != directory dataset {dir_dataset!r}")
    if pc.get("final_pose_gate_enabled") is not True:
        bad.append(f"final_pose_gate_enabled={pc.get('final_pose_gate_enabled')!r} (record True)")
    if profile in ("record", "heldout"):
        for k, v in GATES_OF_RECORD.items():
            if not _num_eq(pc.get(k), v):
                bad.append(f"{k}={pc.get(k)!r} (record {v})")
    else:
        for k in TWO_GATES:
            if pc.get(k) is not None:
                bad.append(f"{k}={pc.get(k)!r} (must be absent under gates-off)")
        if not _num_eq(pc.get("rerank_margin_score"), GATES_OF_RECORD["rerank_margin_score"]):
            bad.append(f"rerank_margin_score={pc.get('rerank_margin_score')!r} (record 50.0)")
    if ds == "vpair" and profile == "heldout" and not pc.get("query_name_list_file"):
        bad.append("query_name_list_file missing (E1b held-out frames)")
    want_k = TOPK_OF_RECORD.get(ds)
    if want_k is not None and not _num_eq(pc.get("retrieval_top_k"), want_k):
        bad.append(f"retrieval_top_k={pc.get('retrieval_top_k')!r} (record {want_k})")
    if ds == "uav_visloc" and profile == "heldout":
        locs = _locations(pc.get("uav_visloc_locations"))
        if len(locs) != 1 or locs[0] not in VISLOC_HELDOUT_FLIGHTS:
            bad.append(f"uav_visloc_locations={pc.get('uav_visloc_locations')!r} (one held-out flight)")
        if not pc.get("forge_query_half_m_table"):
            bad.append("forge_query_half_m_table missing (amendment 2)")
        if not isinstance(pc.get("forge_ref_half_m_per_map"), dict) or not pc.get("forge_ref_half_m_per_map"):
            bad.append("forge_ref_half_m_per_map missing or empty")
        # Same fusion settings as the development flights of record.
        for k, v in VISLOC_OF_RECORD.items():
            if k != "uav_visloc_locations" and not _num_eq(pc.get(k), v):
                bad.append(f"{k}={pc.get(k)!r} (record {v})")
    elif ds == "uav_visloc":
        if _locations(pc.get("uav_visloc_locations")) != VISLOC_LOCATIONS_OF_RECORD:
            bad.append(f"uav_visloc_locations={pc.get('uav_visloc_locations')!r} (record '01,04,11')")
        for k, v in VISLOC_OF_RECORD.items():
            if not _num_eq(pc.get(k), v):
                bad.append(f"{k}={pc.get(k)!r} (record {v})")
    if bad:
        return "FAIL", f"I8 pipeline_config differs from record ({profile}): " + "; ".join(bad)
    return "PASS", f"I8 pipeline_config matches record ({profile})"


def _f(row, key):
    try:
        v = float(row.get(key))
        return v if v == v else None
    except (TypeError, ValueError):
        return None


def _map_of(name: str) -> str:
    name = str(name or "")
    return name.split("/")[0] if "/" in name else ""


def load_rows(run_dir: Path, early: bool):
    if early:
        p = run_dir / "frame_diagnostics.jsonl"
        if not p.exists():
            return None, f"no frame_diagnostics.jsonl under {run_dir}"
        rows = [json.loads(line) for line in p.open() if line.strip()]
        return rows, None
    p = run_dir / "predictions.csv"
    if not p.exists():
        return None, f"no predictions.csv under {run_dir}"
    return list(csv.DictReader(p.open())), None


def dataset_of(run_dir: Path) -> str:
    # .../<cell>/<dataset>/<pipeline>/<timestamp>
    return run_dir.parent.parent.name


def check_cell(run_dir: Path, early: bool = False,
               sibling_offset_medians: dict | None = None,
               profile: str = "record"):
    """Returns (verdict, messages). verdict in PASS/WARN/FAIL."""
    rows, err = load_rows(run_dir, early)
    if rows is None:
        return "FAIL", [err]
    manifest = load_manifest(run_dir)
    dir_ds = dataset_of(run_dir)
    ds = str((manifest or {}).get("dataset") or dir_ds)
    radius = DEFAULT_RADIUS_M.get(ds, 20.0)
    msgs, verdict = [], "PASS"

    def fail(m):
        nonlocal verdict
        verdict = "FAIL"
        msgs.append("FAIL " + m)

    def warn(m):
        nonlocal verdict
        if verdict != "FAIL":
            verdict = "WARN"
        msgs.append("WARN " + m)

    # I8 — gate configuration, checked before any early return so that a
    # 20-frame smoke run already fails on a misconfigured launcher.
    if manifest is None or manifest.get("_invalid"):
        why = manifest.get("_invalid") if manifest else "no run_manifest.json yet"
        if early:
            warn(f"I8 {why}; gate configuration unverified")
        else:
            fail(f"I8 {why}; gate configuration unverifiable")
    else:
        level, m = check_gate_config(manifest, ds, profile, dir_dataset=dir_ds)
        if level == "FAIL":
            fail(m)
        else:
            msgs.append(("info " if level == "INFO" else "pass ") + m)

    inj = [r for r in rows if not str(r.get("adversarial_failure_reason") or "").strip()
           and str(r.get("is_adversarial", "")).lower() in ("true", "1")]
    n = len(rows)

    if early and n < EARLY_MIN_ROWS:
        return verdict, msgs + [f"only {n} rows; too early to judge"]

    # Non-adversarial cells (clean runs, natives) skip the injection invariants;
    # I6/I7 still apply to them.
    is_adversarial_cell = any(
        str(r.get("is_adversarial", "")).lower() in ("true", "1") for r in rows)

    # I1 — injection yield
    if is_adversarial_cell and not inj:
        fail(f"I1 zero injections in {n} frames (forgec_avl signature)")
        return verdict, msgs
    if not early and is_adversarial_cell:
        # A mixed-rate run only attempts injection on its attacked arm, so the
        # floor scales with the rate rather than being lowered by hand.
        inject_rate = 1.0
        for r in rows:
            try:
                candidate = float(r.get("adversarial_inject_rate"))
            except (TypeError, ValueError):
                continue
            if 0.0 < candidate <= 1.0:
                inject_rate = candidate
                break
        min_yield = MIN_YIELD.get(ds, 0.08) * inject_rate
        denom, what = n, "frames"
        table = ((manifest or {}).get("pipeline_config") or {}).get("forge_query_half_m_table")
        if profile == "heldout" and table:
            # E1 attacks only frames with a certified footprint bound (amendment
            # 2), so the yield is judged against those frames, not the flight.
            import csv as _csv
            paths = table if isinstance(table, list) else [table]
            try:
                denom = sum(1 for p in paths for row in _csv.DictReader(open(p, newline=""))
                            if row.get("status") == "OK")
                what = "certified frames"
            except OSError as e:
                fail(f"I1 footprint table unreadable: {e}")
                denom = n
        if denom and len(inj) / denom < min_yield:
            fail(f"I1 injected {len(inj)}/{denom} {what} = {len(inj)/denom:.2f} < floor {min_yield:.3f} "
                 f"for {ds} (rate {inject_rate:g})")

    # I2 — cross-map injection (only meaningful when references carry a map prefix)
    if not is_adversarial_cell:
        inj = []
    forced_maps = [_map_of(r.get("adversarial_forced_reference_name")) for r in inj]
    if any(forced_maps):
        cross = sum(1 for r, fm in zip(inj, forced_maps)
                    if fm and _map_of(r.get("image_name")) and fm != _map_of(r.get("image_name")))
        if cross:
            fail(f"I2 {cross}/{len(inj)} injections cross-map (461 km signature)")

    offs = [_f(r, "adversarial_forced_distance_to_gt_m") for r in inj]
    offs = [o for o in offs if o is not None]
    med_off = statistics.median(offs) if offs else None
    if med_off is not None:
        msgs.append(f"info injected={len(inj)}/{n} offset_med={med_off:.0f}m")
        if med_off > 100_000:
            fail(f"I2b median forced offset {med_off/1000:.0f} km — injecting a different province")

    # I3 — condition collapse (needs sibling cells; caller supplies their medians)
    if sibling_offset_medians and med_off is not None and len(sibling_offset_medians) >= 1:
        near = min([med_off] + list(sibling_offset_medians.values()))
        far = max([med_off] + list(sibling_offset_medians.values()))
        if far > 0 and (far - near) / far < 0.05:
            warn(f"I3 condition medians within 5% ({near:.0f}..{far:.0f} m) — "
                 "conditions may have collapsed onto one candidate")

    if early:
        return verdict, msgs

    # I4 — falsification: an accepted attacked frame localized inside the radius
    # while its forced reference is classed unrecoverable contradicts the
    # definition itself.
    # Violations at the margin measure the bound's tolerance rather than break
    # the definition: frame 04/04_0143.JPG sits 1.4 m (0.3%) beyond the northern
    # bound yet localizes at 65 m. Sub-2% overshoots WARN; anything deeper FAILs.
    hard_viol, margin_viol = 0, 0
    for r in inj:
        if r.get("accepted_for_metrics") != "True":
            continue
        e = _f(r, "post_fgo_error_m")
        if e is None or e > radius:
            continue
        anch = str(r.get("geodetic_anchor") or "")
        gt_lat, gt_lon = _f(r, "gt_lat"), _f(r, "gt_lon")
        if not anch.startswith("{") or gt_lat is None:
            continue
        try:
            a = json.loads(anch.replace("'", '"'))
            can = selected_reference_can_localize(gt_lat, gt_lon,
                                                  float(a["lat"]), float(a["lon"]), ds, r)
            can_with_margin = selected_reference_can_localize(
                gt_lat, gt_lon, float(a["lat"]) ,float(a["lon"]), ds, r,
                extra_margin_frac=0.02)
        except (ValueError, KeyError, TypeError):
            continue
        if not can:
            if can_with_margin:
                margin_viol += 1
            else:
                hard_viol += 1
    if hard_viol:
        # A pose inside R from a disjoint tile breaks the chain prediction
        # (Sec. III-B), but it is an outcome, so it must be counted and
        # reported, never used to hold a cell out of the analysis.
        warn(f"I4 {hard_viol} accepted frames localized inside {radius:.0f}m from a reference "
             "classed unrecoverable — counterexample to the chain prediction; report it")
    if margin_viol:
        warn(f"I4 {margin_viol} boundary-tolerance case(s) within 2% of the recoverability "
             "bound — measures the bound's tolerance; kept as conservative fault in tables")

    # I5 — degenerate candidate pool
    pools = [_f(r, "adversarial_eligible_wrong_count") for r in inj]
    pools = [p for p in pools if p is not None]
    if pools and statistics.median(pools) <= 1:
        warn("I5 median eligible pool is 1 — the three conditions select the same candidate")

    # I6 — summary vs per-frame drift
    summ = run_dir / "metrics_summary.json"
    if summ.exists():
        s = json.loads(summ.read_text())
        acc_pred = sum(1 for r in rows if r.get("accepted_for_metrics") == "True")
        acc_sum = s.get("num_accepted_for_metrics")
        if acc_sum is not None and int(acc_sum) != acc_pred:
            fail(f"I6 summary accepted={acc_sum} but predictions carry {acc_pred}")
    elif not early:
        warn("I6 no metrics_summary.json (run incomplete?)")

    # I7 — dead heading channel on native runs
    stat = {str(r.get("native_yaw_height_status") or "") for r in rows}
    if stat == {"ok"}:
        heads = [_f(r, "pred_heading") for r in rows]
        heads = [h for h in heads if h is not None]
        if heads and all(abs(h) < 1e-9 for h in heads):
            fail("I7 native_yaw_height_status=ok on every frame but pred_heading is all zero")

    return verdict, msgs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path, help="cell dir or timestamped run dir")
    ap.add_argument("--early", action="store_true")
    ap.add_argument("--siblings", nargs="*", type=Path, default=[],
                    help="other condition cells of the same dataset, for I3")
    ap.add_argument("--profile", choices=I8_PROFILES, default="record",
                    help="I8 expectation: 'record' (default), 'gates-off' for the ablation arm, "
                         "'heldout' for E1/E1b")
    args = ap.parse_args()

    d = args.run_dir
    if not (d / "frame_diagnostics.jsonl").exists() and not (d / "predictions.csv").exists():
        stamps = sorted(d.glob("*/*/2*"))
        if not stamps:
            print(f"FAIL no run under {d}")
            sys.exit(1)
        d = stamps[-1]

    sib = {}
    for sp in args.siblings:
        stamps = sorted(sp.glob("*/*/2*")) if not (sp / "predictions.csv").exists() else [sp]
        if not stamps:
            continue
        rows, err = load_rows(stamps[-1], args.early)
        if not rows:
            continue
        so = [_f(r, "adversarial_forced_distance_to_gt_m") for r in rows
              if not str(r.get("adversarial_failure_reason") or "").strip()]
        so = [o for o in so if o is not None]
        if so:
            sib[str(sp)] = statistics.median(so)

    verdict, msgs = check_cell(d, early=args.early, sibling_offset_medians=sib,
                               profile=args.profile)
    print(f"{verdict} {d}")
    for m in msgs:
        print("  " + m)
    sys.exit(1 if verdict == "FAIL" else 0)


if __name__ == "__main__":
    main()
