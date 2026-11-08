"""Each test reconstructs one real failure from 04-05/08 and asserts the QC
gate catches it. These bugs each burned GPU hours because nothing checked."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts" / "reporting"))
from forge_cell_qc import check_cell


RECORD_PC = {
    "avl": {"final_pose_gate_enabled": True, "retrieval_top_k": 5, "rerank_margin_score": 50.0,
            "final_pose_max_yaw_prior_residual_deg": 30.0, "geometry_min_inliers_after_lgi": 25},
    "vpair": {"final_pose_gate_enabled": True, "retrieval_top_k": 5, "rerank_margin_score": 50.0,
              "final_pose_max_yaw_prior_residual_deg": 30.0, "geometry_min_inliers_after_lgi": 25},
    "uav_visloc": {"final_pose_gate_enabled": True, "retrieval_top_k": 10, "rerank_margin_score": 50.0,
                   "final_pose_max_yaw_prior_residual_deg": 30.0, "geometry_min_inliers_after_lgi": 25,
                   "uav_visloc_locations": "01,04,11", "fgo_max_step_xy_m": 40.0,
                   "fgo_motion_prior_sigma_pos_m": 80.0},
}


def _manifest(d, pc, dataset="avl", pipeline="ours_ultravpr_cascade_6dof_e2e"):
    (d / "run_manifest.json").write_text(json.dumps(
        {"dataset": dataset, "pipeline": pipeline, "pipeline_config": pc}))


def _mk(tmp_path, rows, dataset="avl", summary=None, manifest=True):
    d = tmp_path / "cell" / dataset / "pipe" / "20260805_000000"
    d.mkdir(parents=True)
    if manifest:
        _manifest(d, dict(RECORD_PC[dataset]), dataset=dataset)
    if rows and isinstance(rows[0], dict):
        import csv
        keys = sorted({k for r in rows for k in r})
        with open(d / "predictions.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
        with open(d / "frame_diagnostics.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
    if summary is not None:
        (d / "metrics_summary.json").write_text(json.dumps(summary))
    return d


def _frame(i, injected=True, forced="QZ_Town_r005120.png", off=400.0,
           accepted="False", err="", pool=5, img=None):
    return {
        "image_name": img or f"QZ_SongCity/DJI_{i:04d}.JPG",
        "is_adversarial": "True",
        "adversarial_failure_reason": "" if injected else "adversarial_no_eligible_wrong_reference",
        "adversarial_forced_reference_name": forced if injected else "",
        "adversarial_forced_distance_to_gt_m": str(off) if injected else "",
        "adversarial_eligible_wrong_count": str(pool) if injected else "",
        "accepted_for_metrics": accepted,
        "post_fgo_error_m": err,
        "gt_lat": "36.68", "gt_lon": "118.47", "gt_alt": "119.0",
        "geodetic_anchor": "{'lat': 36.684, 'lon': 118.47}",  # ~445m north -> unrecoverable
        "native_yaw_height_status": "",
        "pred_heading": "",
    }


def test_zero_injection_fails(tmp_path):
    """forgec_avl gen3: 487 frames, zero injected, three cells wasted."""
    d = _mk(tmp_path, [_frame(i, injected=False) for i in range(200)])
    verdict, msgs = check_cell(d)
    assert verdict == "FAIL" and any("I1" in m for m in msgs)


def test_cross_map_injection_fails(tmp_path):
    """first forgeb_visloc launch: flight-01 queries forced to flight-04 tiles."""
    rows = [_frame(i, forced="04/satellite04_r036352.png", off=461381.0,
                   img=f"01/01_{i:04d}.JPG") for i in range(200)]
    d = _mk(tmp_path, rows, dataset="uav_visloc")
    verdict, msgs = check_cell(d)
    assert verdict == "FAIL" and any("I2" in m for m in msgs)


def test_healthy_cell_passes(tmp_path):
    rows = ([_frame(i) for i in range(60)]
            + [_frame(i, injected=False) for i in range(60, 200)])
    summary = {"num_accepted_for_metrics": 0}
    d = _mk(tmp_path, rows, summary=summary)
    verdict, msgs = check_cell(d)
    assert verdict == "PASS", msgs


def test_falsification_violation_fails(tmp_path):
    """A frame accepted at 5m error from a reference classed unrecoverable
    means the fault definition is self-contradictory again (189/28/15-case bug)."""
    rows = [_frame(i, accepted="True" if i < 3 else "False",
                   err="5.0" if i < 3 else "") for i in range(200)]
    summary = {"num_accepted_for_metrics": 3}
    d = _mk(tmp_path, rows, summary=summary)
    verdict, msgs = check_cell(d)
    assert verdict == "FAIL" and any("I4" in m for m in msgs)


def test_dead_heading_channel_fails(tmp_path):
    """OrthoLoC rerun signature: status ok everywhere, pred_heading all zero."""
    rows = []
    for i in range(50):
        r = _frame(i, injected=False)
        r["is_adversarial"] = "False"
        r["native_yaw_height_status"] = "ok"
        r["pred_heading"] = "0.0"
        rows.append(r)
    d = _mk(tmp_path, rows, summary={"num_accepted_for_metrics": 0})
    verdict, msgs = check_cell(d)
    assert verdict == "FAIL" and any("I7" in m for m in msgs)
    # And a native cell must never be flagged for injection yield.
    assert not any("I1" in m for m in msgs), msgs


def test_condition_collapse_warns(tmp_path):
    """forgeb_avl gen1: three conditions, medians 369/378/370 — same candidate."""
    rows = [_frame(i, off=370.0) for i in range(200)]
    d = _mk(tmp_path, rows, summary={"num_accepted_for_metrics": 0})
    verdict, msgs = check_cell(d, sibling_offset_medians={"far": 378.0, "sim": 369.0})
    assert verdict in ("WARN", "FAIL") and any("I3" in m for m in msgs)


def test_early_mode_needs_no_summary(tmp_path):
    rows = [_frame(i) for i in range(160)]
    d = _mk(tmp_path, rows)
    verdict, msgs = check_cell(d, early=True)
    assert verdict == "PASS", msgs


def test_margin_violation_warns_not_fails(tmp_path):
    """04/04_0143.JPG case: 1.4 m (0.3%) beyond the bound yet localized.
    Measures the bound's tolerance; must WARN, not FAIL, and must not FAIL
    the whole cell over one boundary frame."""
    rows = [_frame(i) for i in range(200)]
    # anchor ~423 m north on uav_visloc: bound N = 305.8+101+15 = 421.8
    v = _frame(999, accepted="True", err="65.3",
               img="04/04_0143.JPG", forced="04/satellite04_r1.png")
    v["gt_lat"] = "36.0"; v["gt_lon"] = "118.0"
    v["geodetic_anchor"] = "{'lat': %.8f, 'lon': 118.0}" % (36.0 + 423.2/111320.0)
    rows.append(v)
    d = _mk(tmp_path, rows, dataset="uav_visloc",
            summary={"num_accepted_for_metrics": 1})
    verdict, msgs = check_cell(d)
    assert verdict == "WARN", msgs
    assert any("I4" in m and "tolerance" in m for m in msgs), msgs


def test_deep_violation_still_fails(tmp_path):
    """A frame 50% beyond the bound that localizes is a broken definition."""
    rows = [_frame(i) for i in range(200)]
    v = _frame(999, accepted="True", err="65.3",
               img="04/04_0143.JPG", forced="04/satellite04_r1.png")
    v["gt_lat"] = "36.0"; v["gt_lon"] = "118.0"
    v["geodetic_anchor"] = "{'lat': %.8f, 'lon': 118.0}" % (36.0 + 700.0/111320.0)
    rows.append(v)
    d = _mk(tmp_path, rows, dataset="uav_visloc",
            summary={"num_accepted_for_metrics": 1})
    verdict, msgs = check_cell(d)
    assert verdict == "FAIL" and any("I4" in m for m in msgs), msgs


# --- I8: gate configuration of record (22/08 mixed-campaign incident) ---

def _healthy(tmp_path, dataset="avl", n_inj=60):
    rows = ([_frame(i) for i in range(n_inj)]
            + [_frame(i, injected=False) for i in range(n_inj, 200)])
    return _mk(tmp_path, rows, dataset=dataset, summary={"num_accepted_for_metrics": 0})


def test_i8_record_config_passes(tmp_path):
    d = _healthy(tmp_path)
    verdict, msgs = check_cell(d)
    assert verdict == "PASS" and any("I8" in m and "matches" in m for m in msgs), msgs


def test_i8_missing_gate_block_fails(tmp_path):
    """18/08: run_forge5050.sh launched nine cells without the gate block."""
    d = _healthy(tmp_path)
    _manifest(d, {"retrieval_top_k": 5})
    verdict, msgs = check_cell(d)
    assert verdict == "FAIL" and any("I8" in m for m in msgs), msgs


def test_i8_fails_on_short_smoke_before_early_return(tmp_path):
    d = _mk(tmp_path, [_frame(i) for i in range(20)])
    _manifest(d, {"retrieval_top_k": 5})
    verdict, msgs = check_cell(d, early=True)
    assert verdict == "FAIL" and any("I8" in m for m in msgs), msgs


def test_i8_gates_off_profile(tmp_path):
    d = _healthy(tmp_path)
    _manifest(d, {"final_pose_gate_enabled": True, "retrieval_top_k": 5, "rerank_margin_score": 50.0})
    assert check_cell(d, profile="gates-off")[0] == "PASS"
    assert check_cell(d)[0] == "FAIL"  # the same manifest is wrong under 'record'


def test_i8_wrapper_not_applicable(tmp_path):
    d = _healthy(tmp_path)
    _manifest(d, {"retrieval_top_k": 5, "rerank_margin_score": 50.0, "is_wrapper": True},
              pipeline="ultravpr_roma_homography_wrapper_e2e")
    verdict, msgs = check_cell(d)
    assert verdict == "PASS" and any("not applicable" in m for m in msgs), msgs


def test_i8_visloc_record_keys(tmp_path):
    d = _healthy(tmp_path, dataset="uav_visloc", n_inj=120)
    assert check_cell(d)[0] == "PASS"
    wrong = dict(RECORD_PC["avl"], uav_tile_size_px=768)  # 18/08 visloc mixed cells
    _manifest(d, wrong, dataset="uav_visloc")
    verdict, msgs = check_cell(d)
    assert verdict == "FAIL" and any("retrieval_top_k" in m and "uav_visloc_locations" in m
                                      for m in msgs), msgs


def test_i8_missing_manifest_fails_final_warns_early(tmp_path):
    d = _healthy(tmp_path)
    (d / "run_manifest.json").unlink()
    verdict, msgs = check_cell(d)
    assert verdict == "FAIL" and any("I8" in m for m in msgs), msgs
    verdict, msgs = check_cell(d, early=True)
    assert verdict == "WARN" and any("I8" in m for m in msgs), msgs


def test_i8_gate_disabled_fails_despite_thresholds(tmp_path):
    d = _healthy(tmp_path)
    _manifest(d, dict(RECORD_PC["avl"], final_pose_gate_enabled=False))
    verdict, msgs = check_cell(d)
    assert verdict == "FAIL" and any("final_pose_gate_enabled" in m for m in msgs), msgs


def test_i8_visloc_locations_must_match_record(tmp_path):
    d = _healthy(tmp_path, dataset="uav_visloc", n_inj=120)
    _manifest(d, dict(RECORD_PC["uav_visloc"], uav_visloc_locations=["01", "04", "11"]),
              dataset="uav_visloc")
    assert check_cell(d)[0] == "PASS"  # list form of the same flights
    _manifest(d, dict(RECORD_PC["uav_visloc"], uav_visloc_locations="99"), dataset="uav_visloc")
    assert check_cell(d)[0] == "FAIL"


def test_i8_dataset_mismatch_fails(tmp_path):
    d = _healthy(tmp_path)  # directory says avl
    _manifest(d, RECORD_PC["vpair"], dataset="vpair")
    verdict, msgs = check_cell(d)
    assert verdict == "FAIL" and any("directory dataset" in m for m in msgs), msgs


def test_i8_retrieval_only_not_applicable(tmp_path):
    d = _healthy(tmp_path)
    (d / "run_manifest.json").write_text(json.dumps(
        {"dataset": "avl", "pipeline": "ours_ultravpr_cascade_6dof_e2e", "retrieval_only": True,
         "pipeline_config": {"retrieval_top_k": 1}}))
    verdict, msgs = check_cell(d)
    assert verdict == "PASS" and any("not applicable" in m for m in msgs), msgs
