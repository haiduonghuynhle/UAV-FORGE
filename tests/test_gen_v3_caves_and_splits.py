"""gen_v3_tables: real medians, AVL view-angle bands, CAVES table, quarantine.

Each test pins one defect the reporting audit traced to an artifact:
  * errs[n // 2] is the UPPER middle value, not the median, on even n;
  * the AVL nadir/oblique split was gated on the primary pipeline name, so the
    DSM-PnP head -- the row whose whole point is the oblique regime -- had no
    oblique cell and the prose typed the number;
  * a band with zero accepted frames must render the missing marker, not 0.0;
  * the CAVES figures backing research question 3 were prose only;
  * results/caves_avl487_vpair1000_* and results/caves_combined_* were
    rewritten in place by apply_caves_auroc_boost.py and must be unreadable.
"""
import csv
import json
import statistics
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "reporting"))

import gen_v3_tables as gt  # noqa: E402


def _write_run(root, cell, ds, pipe, rows, summary=None, ts="20260101_000000"):
    d = root / cell / ds / pipe / ts
    d.mkdir(parents=True)
    (d / "metrics_summary.json").write_text(json.dumps(summary or {}))
    (d / "run_manifest.json").write_text(json.dumps({"run_completed": True}))
    if rows:
        fields = sorted({k for r in rows for k in r})
        with open(d / "predictions.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)
    return d


def _avl_frame(pitch_from_nadir, err, accepted=True, yaw=1.0, alt_err=0.0):
    return {
        "image_name": f"f{err}_{pitch_from_nadir}",
        "gt_pitch": f"{90.0 - pitch_from_nadir:.4f}",
        "post_fgo_error_m": str(err),
        "accepted_for_metrics": "True" if accepted else "False",
        "yaw_error_deg": str(yaw),
        "gt_alt": "100.0",
        "pred_alt": str(100.0 + alt_err),
    }


# ---------------------------------------------------------------------------
# median
# ---------------------------------------------------------------------------

def test_split_median_is_a_median_on_even_counts(tmp_path):
    """errs[n // 2] returns the upper middle value; four errors expose it."""
    errs = [1.0, 2.0, 3.0, 10.0]              # median 2.5, errs[2] == 3.0
    yaws = [1.0, 2.0, 3.0, 4.0]               # median 2.5, ys[2] == 3.0
    alts = [1.0, 2.0, 3.0, 4.0]               # median 2.5
    rows = [_avl_frame(50.0, e, yaw=y, alt_err=a)
            for e, y, a in zip(errs, yaws, alts)]
    d = _write_run(tmp_path, "avl_dsm_pnp", "avl", "ours_dsm_pnp_6dof_e2e", rows)
    split = gt.nadir_oblique_split({"run_dir": str(d)})

    assert split["oblique"]["accepted"] == 4
    assert split["oblique"]["median_m"] == statistics.median(errs) == 2.5
    assert split["oblique"]["median_m"] != errs[len(errs) // 2]
    assert split["oblique"]["yaw_med"] == 2.5 != yaws[len(yaws) // 2]
    assert split["oblique"]["alt_med"] == 2.5 != alts[len(alts) // 2]


def test_wrong_retrieval_err_median_is_a_median_on_even_counts(tmp_path):
    """VPAIR references cover 336x252 m; 5 km east of GT is unambiguously wrong."""
    errs = [1.0, 2.0, 3.0, 10.0]
    rows = [{
        "image_name": f"v{e}",
        "gt_lat": "50.0", "gt_lon": "8.0",
        "geodetic_anchor": "{'lat': 50.0, 'lon': 8.07}",   # ~5 km east
        "accepted_for_metrics": "True",
        "post_fgo_error_m": str(e),
    } for e in errs]
    d = _write_run(tmp_path, "vpair_ours_ultravpr", "vpair",
                   "ours_ultravpr_cascade_6dof_e2e", rows)
    st = gt.wrong_retrieval_stats({"run_dir": str(d), "dataset": "vpair"})

    assert st["wrong"] == st["wrong_accepted"] == 4
    assert st["err_median"] == 2.5 != errs[len(errs) // 2]
    assert st["err_max"] == 10.0


def test_native_script_median_is_a_median_on_even_counts(tmp_path):
    d = tmp_path / "scripts_dir"
    d.mkdir()
    with open(d / "vpair_yaw_height_results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["stem", "pos_err_m", "yaw_err_deg", "alt_err_m"])
        w.writeheader()
        for e, y, a in zip([1.0, 2.0, 3.0, 10.0], [1.0, 2.0, 3.0, 4.0],
                           [1.0, 2.0, 3.0, 4.0]):
            w.writerow({"stem": "x", "pos_err_m": e, "yaw_err_deg": y, "alt_err_m": a})
    row = [r for r in gt.native_script_rows(d) if not r.startswith("%")][0]
    cells = row.split(" & ")
    assert cells[2] == "2.5"                  # Med. (m): median, not errs[2]=3.0
    assert cells[7] == "2.5"                  # Yaw: median, not yaws[2]=3.0


def test_script_fps_rejects_log_from_different_population(tmp_path):
    log = tmp_path / "native.log"
    log.write_text("run: 400/400 [06:02<00:00, 1.10it/s]\n", encoding="utf-8")

    assert gt._script_fps(log, expected_total=800) is None
    assert gt._script_fps(log, expected_total=400) == pytest.approx(400 / 362)


# ---------------------------------------------------------------------------
# AVL view-angle bands
# ---------------------------------------------------------------------------

def _avl_dsm_pnp_root(tmp_path):
    """AVL DSM-PnP shaped like the real run: mid band has no accepted frame."""
    rows = ([_avl_frame(5.0, 7.0)] + [_avl_frame(5.0, 99.0, accepted=False)] * 3
            + [_avl_frame(25.0, 99.0, accepted=False)] * 4            # mid: 0 accepted
            + [_avl_frame(50.0, e) for e in (10.0, 20.0, 30.0)])
    # num_frames is the AVL block size, so the sub-population guard in the
    # localization tables does not skip this fixture. The guard has its own
    # test below; this fixture is about the view-angle split.
    _write_run(tmp_path, "avl_dsm_pnp", "avl", "ours_dsm_pnp_6dof_e2e", rows,
               summary={"num_frames": gt.BLOCK_N["avl"], "num_accepted_for_metrics": 4,
                        "metrics": {"median_error_m": 20.0}})
    return gt.load_runs(tmp_path)


def test_sub_population_run_never_prints_under_a_full_set_block(tmp_path):
    """A 30-frame probe must not appear beside a full-set row.

    Two smoke probes left inside the results root did exactly that: one printed
    96.7 percent acceptance at a 1.9 m median next to the real 487-frame AVL
    row, which a reader has no way to recognize as a probe.
    """
    rows = [_avl_frame(5.0, 1.9) for _ in range(30)]
    _write_run(tmp_path, "smoke_c3", "avl", "ours_dsm_pnp_6dof_e2e", rows,
               summary={"num_frames": 30, "num_accepted_for_metrics": 29,
                        "metrics": {"median_error_m": 1.9}})
    runs = gt.load_runs(tmp_path)

    for tex in (gt.loc_table(runs, "avl", "cap.", "tab:x", avl_split=True),
                gt.merged_loc_table(runs)):
        assert "1.9" not in tex, "a 30-frame probe reached a full-set table"
    # and the skip is recorded, so a silent drop is distinguishable from a bug
    assert "% skipped (n=30" in gt.loc_table(runs, "avl", "cap.", "tab:x")


def test_view_angle_split_is_not_gated_on_the_primary_pipeline(tmp_path):
    """Every AVL row with a computable split gets one, not just the cascade.

    The split used to be gated on the primary pipeline name, which left the
    DSM-PnP head -- the row whose whole point is the oblique regime -- with no
    oblique cell, so the prose typed the number by hand. The bands now live in
    their own table and in macros; the merged table stays one row per system.
    """
    runs = _avl_dsm_pnp_root(tmp_path)
    single = gt.loc_table(runs, "avl", "cap.", "tab:x", avl_split=True)
    assert "\\quad oblique" in single, "DSM-PnP row still has no oblique sub-row"
    assert "\\quad nadir" in single

    # The dedicated view-angle table carries this run's own oblique median.
    view = gt.avl_view_angle_table(runs)
    oblique = [l for l in view.splitlines() if "oblique" in l and "&" in l
               and not l.lstrip().startswith("\\bfseries")]
    assert oblique, "the view-angle table has no oblique row"
    assert "20.0" in oblique[0]

    # The merged table hands the bands to that table instead of nesting them.
    assert "\\quad oblique" not in gt.merged_loc_table(runs)


def test_no_split_for_rows_without_per_frame_position_error(tmp_path):
    """The native adapters accept every frame but write no post_fgo_error_m.

    Splitting them printed Acc. 0.0\\% sub-rows under a pooled row reading
    100\\%, with every other cell "--". Two guards: the accepted count is the
    accepted count, and an uncomputable split is not emitted at all.
    """
    rows = []
    for pitch in (5.0, 25.0, 50.0):
        r = _avl_frame(pitch, 0.0)
        del r["post_fgo_error_m"]
        rows.append(r)
    _write_run(tmp_path, "avl_native_anyvisloc", "avl", "native_anyvisloc_e2e", rows,
               summary={"num_frames": gt.BLOCK_N["avl"], "num_accepted_for_metrics": 3})
    runs = gt.load_runs(tmp_path)

    assert gt.nadir_oblique_split(runs[0]) is None
    assert "AnyVisLoc (native)" in gt.merged_loc_table(runs)
    # An uncomputable split is emitted nowhere: not as sub-rows, not as a row
    # in the dedicated view-angle table.
    assert "\\quad nadir" not in gt.loc_table(runs, "avl", "cap.", "tab:x", avl_split=True)
    assert "AnyVisLoc" not in gt.avl_view_angle_table(runs)


def test_empty_band_renders_the_missing_marker_not_zero(tmp_path):
    runs = _avl_dsm_pnp_root(tmp_path)
    split = gt.nadir_oblique_split(runs[0])
    assert split["mid"]["n"] == 4 and split["mid"]["accepted"] == 0
    for field in ("median_m", "mean_m", "yaw_med", "a5", "a10", "alt_med"):
        assert split["mid"][field] is None, field

    tex = gt.macros_tex(runs)
    assert "\\newcommand{\\avlDsmPnpMidMedian}{--}" in tex
    assert "\\newcommand{\\avlDsmPnpMidAccN}{0}" in tex      # the count is real
    assert "\\newcommand{\\avlDsmPnpMidN}{4}" in tex


def test_oblique_medians_are_exported_as_macros(tmp_path):
    """The like-for-like ratio must be stateable from two macros, not typed."""
    runs = _avl_dsm_pnp_root(tmp_path)
    _write_run(tmp_path, "avl_ours_ultravpr", "avl",
               "ours_ultravpr_cascade_6dof_e2e",
               [_avl_frame(50.0, e) for e in (100.0, 120.0, 140.0)])
    tex = gt.macros_tex(gt.load_runs(tmp_path))

    assert "\\newcommand{\\avlDsmPnpObliqueMedian}{20.0}" in tex
    assert "\\newcommand{\\avlOursObliqueMedian}{120.0}" in tex
    # The ratio itself is never emitted: prose states it from the two medians.
    assert "Ratio" not in tex and "Improvement" not in tex


def test_unconditional_a10_divides_hits_by_the_band_count(tmp_path):
    """The two denominators in the sub-row must not be mixed.

    A@10m divides by the accepted frames that carry a parsable error. The
    acceptance rate divides by the band's frame count. Deriving the
    unconditional column as A@10m times the acceptance rate multiplies one by
    the other and overstates it whenever an accepted frame has no error value.
    """
    rows = [_avl_frame(50.0, 4.0) for _ in range(5)]           # accepted, under 10 m
    for i in range(5):                                          # accepted, no error
        rows.append({"image_name": f"blank{i}", "gt_pitch": "40.0",
                     "post_fgo_error_m": "", "accepted_for_metrics": "True"})
    for i in range(10):                                         # not accepted
        rows.append({"image_name": f"rej{i}", "gt_pitch": "40.0",
                     "post_fgo_error_m": "", "accepted_for_metrics": "False"})
    _write_run(tmp_path, "avl_dsm_pnp", "avl", "ours_dsm_pnp_6dof_e2e", rows)
    split = gt.nadir_oblique_split(gt.load_runs(tmp_path)[0])

    band = split["oblique"]
    assert band["n"] == 20 and band["accepted"] == 10
    assert band["a10"] == pytest.approx(1.0)      # 5 of the 5 frames with an error
    assert band["a10_hits"] == 5
    # 5 hits out of 20 frames is 25 percent. The old expression gave
    # 1.0 * 10 / 20 = 50 percent.
    assert band["a10_hits"] / band["n"] == pytest.approx(0.25)


# ---------------------------------------------------------------------------
# CAVES
# ---------------------------------------------------------------------------

def _caves_frame(name, score, err, accepted=True):
    return {"image_name": name, "caves_risk_score": str(score),
            "post_fgo_error_m": str(err),
            "accepted_for_metrics": "True" if accepted else "False"}


def test_caves_uses_accepted_frames_only_and_the_forge_radius(tmp_path):
    rows = [
        _caves_frame("g1", 0.10, 1.0), _caves_frame("g2", 0.20, 2.0),
        _caves_frame("b1", 0.80, 100.0), _caves_frame("b2", 0.90, 400.0),
        # Rejected frames must not enter: these would drag AUROC below 1.0.
        _caves_frame("r1", 0.05, 900.0, accepted=False),
        _caves_frame("r2", 0.99, 1.0, accepted=False),
    ]
    _write_run(tmp_path, "vpair_ours_ultravpr", "vpair",
               "ours_ultravpr_cascade_6dof_e2e", rows)
    # 100 m is unsafe at R=20 (VPAIR) and safe at R=150 (UAV-VisLoc): the same
    # frame set must therefore give different base rates per dataset.
    _write_run(tmp_path, "visloc_ours_ultravpr", "uav_visloc",
               "ours_ultravpr_cascade_6dof_e2e", rows)
    runs = gt.load_runs(tmp_path)
    by_cell = {(r["cell"], r["dataset"]): r for r in runs}

    vp = gt.caves_stats(by_cell[("vpair_ours_ultravpr", "vpair")])
    assert vp["n"] == 4 and vp["unsafe"] == 2 and vp["base_rate"] == 0.5
    assert vp["auroc"] == 1.0 and vp["radius_m"] == gt.DEFAULT_RADIUS_M["vpair"]

    vl = gt.caves_stats(by_cell[("visloc_ours_ultravpr", "uav_visloc")])
    assert vl["radius_m"] == gt.DEFAULT_RADIUS_M["uav_visloc"] == 150.0
    assert vl["unsafe"] == 1 and vl["base_rate"] == 0.25


def test_caves_auroc_is_the_lifted_implementation():
    """No third AUROC in the repo: the emitter uses the sweep script's own."""
    import caves_per_dataset_operating_points as ops
    assert gt._rank_auroc is ops.auroc
    # Ties take half credit, and the 3-tuple caller in that script still works.
    assert ops.auroc([(0.5, True), (0.5, False)]) == 0.5
    assert ops.auroc([(0.5, True, "p"), (0.5, False, "p")]) == 0.5


def test_caves_table_and_macros(tmp_path):
    rows = [_caves_frame("g1", 0.1, 1.0), _caves_frame("g2", 0.2, 2.0),
            _caves_frame("b1", 0.8, 100.0), _caves_frame("b2", 0.9, 400.0)]
    _write_run(tmp_path, "vpair_ours_ultravpr", "vpair",
               "ours_ultravpr_cascade_6dof_e2e", rows)
    runs = gt.load_runs(tmp_path)
    tex = gt.caves_table(runs)

    body = [l for l in tex.splitlines() if l.startswith("Ours (UltraVPR-Cascade)")]
    assert len(body) == 1
    # The point estimate and its interval must be in the same unit, or the
    # interval reads as though it never brackets the estimate.
    assert body[0].split(" & ")[1:] == ["VPAIR", "4", "2",
                                        "50.0\\% [15.0, 85.0]", "1.000 \\\\"]
    # One source comment per row, and the caption must name the population and
    # explain why no pooled AUROC is reported.
    assert sum(1 for l in tex.splitlines() if l.startswith("% source:")) == len(body)
    assert "\\emph{accepted} frames of one primary clean run per dataset" in tex
    assert "Table~\\ref{tab:v3-loc-merged} and Table~\\ref{tab:v3-loc-visloc}" in tex
    assert "pooled across backbones" in tex and "top-1-miss subset" in tex
    assert "Simpson" in tex and "forty times" in tex
    # The three missing cells stay placeholders, never fabricated rows.
    assert tex.count("todo{pending:") == 3

    macros = gt.macros_tex(runs)
    assert "\\newcommand{\\vpairOursCavesAuroc}{1.000}" in macros
    assert "\\newcommand{\\vpairOursCavesBaseRate}{0.500}" in macros
    assert "\\newcommand{\\avlOursCavesAuroc}{\\todo{pending: avl\\_ours\\_ultravpr}}" in macros


# ---------------------------------------------------------------------------
# reject codes and clean-run failure cause (folded in from gen_ch4_new_tables)
# ---------------------------------------------------------------------------

def _attacked_frame(name, reason, accepted=False, wrong=True):
    """One VPAIR injection trial. 0.07 deg east is ~5 km, far outside the tile."""
    return {
        "image_name": name,
        "gt_lat": "50.0", "gt_lon": "8.0",
        "geodetic_anchor": "{'lat': 50.0, 'lon': %s}" % ("8.07" if wrong else "8.0"),
        "adversarial_failure_reason": "",
        "accepted_for_metrics": "True" if accepted else "False",
        "failure_reason": reason,
        "post_fgo_error_m": "5000.0" if accepted else "",
    }


def _write_detailed(run_dir, reasons):
    """results_detailed.json is the only artifact carrying final_pose_gate_reason."""
    (run_dir / "results_detailed.json").write_text(json.dumps({
        "predictions": [{"image_name": k, "final_pose_gate_reason": v}
                        for k, v in reasons.items()]}))


def test_reject_code_table_rows_sources_and_phase4_breakdown(tmp_path):
    """The Phase 4 sub-code comes from results_detailed, not from the CSV.

    predictions.csv records only the coarse "final_pose_gate_reject", so a
    generator reading it alone collapses the yaw-prior and shape-confidence
    channels into one unnamed bucket, and the chapter's bound on the
    simulated-telemetry channel loses its evidence.
    """
    rows = [
        _attacked_frame("a", "geometry_inlier_gate_reject"),
        _attacked_frame("b", "geometry_inlier_gate_reject"),
        _attacked_frame("c", "final_pose_gate_reject"),
        _attacked_frame("d", "geometry_inlier_gate_reject", accepted=True),
        # Covered by the forced reference: not a genuine wrong-reference trial.
        _attacked_frame("e", "geometry_inlier_gate_reject", wrong=False),
    ]
    d = _write_run(tmp_path, "forgec_vpair_wrong_near", "vpair",
                   "ours_ultravpr_cascade_6dof_e2e", rows)
    _write_detailed(d, {"c": "final_pose_yaw_prior_reject"})
    tex = gt.reject_code_table(gt.load_runs(tmp_path))
    body = {l.split(" & ")[0]: l for l in tex.splitlines() if " & " in l
            and not l.startswith("\\bfseries")}

    assert body["Inlier count after LGI (Phase 3)"] == \
        "Inlier count after LGI (Phase 3) & 2 & 0 & 0 & 66.7\\% \\\\"
    assert body["Yaw-prior residual (Phase 4)"] == \
        "Yaw-prior residual (Phase 4) & 1 & 0 & 0 & 33.3\\% \\\\"
    assert body["Rejections"] == "Rejections & 3 & 0 & 0 & 3 \\\\"
    # Provenance: the path as given on the command line, one per source run.
    assert f"% source (proximal): {d}" in tex.splitlines()
    assert sum(1 for l in tex.splitlines() if l.startswith("% source")) == 1


def test_reject_code_share_of_no_rejections_is_the_missing_marker(tmp_path):
    """An empty denominator is an unread population, not a zero share."""
    rows = [_attacked_frame("a", "", accepted=True),
            _attacked_frame("b", "", accepted=True)]
    _write_run(tmp_path, "forgec_vpair_wrong_near", "vpair",
               "ours_ultravpr_cascade_6dof_e2e", rows)
    tex = gt.reject_code_table(gt.load_runs(tmp_path))

    assert "Rejections & 0 & 0 & 0 & -- \\\\" in tex
    assert "0.0\\%" not in tex


def _clean_frame(name, err, accepted=True, wrong=False):
    return {
        "image_name": name,
        "gt_lat": "50.0", "gt_lon": "8.0",
        "geodetic_anchor": "{'lat': 50.0, 'lon': %s}" % ("8.07" if wrong else "8.0"),
        "accepted_for_metrics": "True" if accepted else "False",
        "post_fgo_error_m": str(err),
    }


def test_failure_cause_table_splits_unsafe_accepts_by_reference(tmp_path):
    """Unsafe is scored at the shared radius, and wrong is the shared predicate."""
    rows = [
        _clean_frame("safe1", 1.0), _clean_frame("safe2", 2.0),
        _clean_frame("bad_correct", 100.0),
        _clean_frame("bad_wrong", 5000.0, wrong=True),
        _clean_frame("rejected", 900.0, accepted=False),
    ]
    d = _write_run(tmp_path, "vpair_ours_ultravpr", "vpair",
                   "ours_ultravpr_cascade_6dof_e2e", rows)
    runs = gt.load_runs(tmp_path)
    st = gt.failure_cause_stats(runs[0])
    assert st["accepted"] == 4 and st["unsafe"] == 2
    assert st["wrong"] == 1 and st["correct"] == 1
    assert st["radius_m"] == gt.DEFAULT_RADIUS_M["vpair"]

    tex = gt.failure_cause_table(runs)
    assert "VPAIR & 4 & 2 & 1 & 1 \\\\" in tex
    assert f"% source: {d} (R=20 m)" in tex.splitlines()
    assert sum(1 for l in tex.splitlines() if l.startswith("% source:")) == 1
    # The two cells with no run stay placeholders, never fabricated rows.
    assert tex.count("todo{pending:") == 2


def test_failure_cause_missing_run_renders_the_marker_not_zeros(tmp_path):
    """A run with no per-frame record must not print a line of zeros.

    Four zeros read as a measured result: no unsafe accept anywhere, which is
    the opposite of "this run was never read".
    """
    _write_run(tmp_path, "avl_ours_ultravpr", "avl",
               "ours_ultravpr_cascade_6dof_e2e", [])
    runs = gt.load_runs(tmp_path)

    assert gt.failure_cause_stats(runs[0]) is None
    tex = gt.failure_cause_table(runs)
    assert "AVL & 0 & 0 & 0 & 0" not in tex
    assert tex.count("todo{pending:") == 3
    assert tex.startswith("% clean runs not available yet")


# ---------------------------------------------------------------------------
# quarantine
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad", [
    "results/caves_avl487_vpair1000_20260625",
    "/workspace/forge/results/caves_avl487_vpair1000_20260624/caves_risk_coverage.csv",
    "results/caves_combined_avl487_vpair1k_uavvisloc_20260626",
])
def test_quarantined_dirs_are_refused(bad):
    with pytest.raises(SystemExit) as exc:
        gt.refuse_quarantined_path(bad)
    assert "apply_caves_auroc_boost.py" in str(exc.value)


def test_v3_paths_are_not_refused():
    gt.refuse_quarantined_path("/workspace/forge/results/v3/avl_ours_ultravpr")
    gt.refuse_quarantined_path("results/caves_pilot_avl_vpair_n1200")


# ---------------------------------------------------------------------------
# Chapter 3 methodology macros
# ---------------------------------------------------------------------------

def test_margin_guard_stats_counts_displaced_and_repaired(tmp_path):
    """displaced = correct top-1 swapped for wrong; repaired = the reverse."""
    p = tmp_path / "predictions.csv"
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "candidate_rank_selected", "retrieval_anchor_error_m",
            "raw_retrieval_anchor_error_m"])
        w.writeheader()
        # rank 0: kept top-1 (not counted)
        w.writerow({"candidate_rank_selected": "0",
                     "retrieval_anchor_error_m": "5", "raw_retrieval_anchor_error_m": "5"})
        # rank 2: correct -> wrong (displaced)
        w.writerow({"candidate_rank_selected": "2",
                     "retrieval_anchor_error_m": "50", "raw_retrieval_anchor_error_m": "10"})
        # rank 1: wrong -> correct (repaired)
        w.writerow({"candidate_rank_selected": "1",
                     "retrieval_anchor_error_m": "8", "raw_retrieval_anchor_error_m": "100"})
        # rank 3: wrong -> wrong (neither)
        w.writerow({"candidate_rank_selected": "3",
                     "retrieval_anchor_error_m": "99", "raw_retrieval_anchor_error_m": "80"})
    st = gt.margin_guard_stats(p, threshold_m=20.0)
    assert st["displaced"] == 1
    assert st["repaired"] == 1
    assert st["n"] == 4


def test_margin_guard_stats_missing_csv_returns_empty():
    assert gt.margin_guard_stats(None) == {}
    assert gt.margin_guard_stats(Path("/nonexistent")) == {}


def test_height_scale_correlation_near_zero_on_uncorrelated(tmp_path):
    """visual_pred_alt uncorrelated with gt_alt -> |r| near 0."""
    p = tmp_path / "predictions.csv"
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "accepted_for_metrics", "visual_pred_alt", "gt_alt"])
        w.writeheader()
        # Deliberately uncorrelated: gt climbs, visual is flat
        for i in range(20):
            w.writerow({"accepted_for_metrics": "True",
                         "visual_pred_alt": str(100 + (i % 3)),
                         "gt_alt": str(400 + i * 10)})
        # Rejected frames should be excluded
        w.writerow({"accepted_for_metrics": "False",
                     "visual_pred_alt": "9999", "gt_alt": "9999"})
    st = gt.height_scale_correlation(p)
    assert st["n"] == 20
    assert abs(st["correlation"]) < 0.3


def test_height_scale_correlation_perfect(tmp_path):
    """Perfectly correlated data -> r = 1.0."""
    p = tmp_path / "predictions.csv"
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "accepted_for_metrics", "visual_pred_alt", "gt_alt"])
        w.writeheader()
        for i in range(10):
            v = 100 + i * 50
            w.writerow({"accepted_for_metrics": "True",
                         "visual_pred_alt": str(v), "gt_alt": str(v)})
    st = gt.height_scale_correlation(p)
    assert st["n"] == 10
    assert abs(st["correlation"] - 1.0) < 1e-9


def test_fault_rule_contradictions_on_synthetic_forge(tmp_path):
    """The center-containment rule produces a contradiction when a query just
    outside the reference footprint localizes correctly from it.
    """
    # A VPAIR frame: query center is 170 m east of reference center.
    # REF_HALF_M for vpair = (168, 126). So 170 > 168 -> center-containment
    # says "wrong". But per-axis with query half (~168 for vpair) says "OK"
    # because 170 < 168 + 168 + 15 = 351. If the frame localizes correctly,
    # center-containment has a contradiction.
    offset_deg = 170.0 / (111320.0 * 0.6427876)  # ~170 m east at lat 50
    rows = [{
        "image_name": "edge_frame",
        "gt_lat": "50.0", "gt_lon": "8.0",
        "geodetic_anchor": f"{{'lat': 50.0, 'lon': {8.0 + offset_deg}}}",
        "adversarial_failure_reason": "",
        "accepted_for_metrics": "True",
        "post_fgo_error_m": "5.0",
    }]
    _write_run(tmp_path, "forgec_vpair_wrong_near", "vpair",
               "ours_ultravpr_cascade_6dof_e2e", rows)
    runs = gt.load_runs(tmp_path)
    counts = gt.fault_rule_contradictions(runs)
    assert counts["center_containment"] >= 1


def test_fault_borderline_trial_finds_small_overshoot(tmp_path):
    """A frame that overshoots the no-margin bound by less than TILE_SLACK_M
    but localizes correctly is the borderline trial."""
    # For AVL: rh_e = rh_n = 62.46, query half ~ 1.0847 * alt per axis.
    # If alt=100: qh = 108.47. Bound without slack = 62.46 + 108.47 = 170.93.
    # Place reference at 175 m north -> overshoot = 175 - 170.93 = 4.07 m,
    # which is < TILE_SLACK_M (15 m). With slack: bound = 185.93 -> 175 < 185.93,
    # so per-axis predicate says "can localize".
    offset_deg = 175.0 / 111320.0  # ~175 m north
    rows = [{
        "image_name": "border_frame",
        "gt_lat": "50.0", "gt_lon": "8.0",
        "gt_alt": "100.0",
        "geodetic_anchor": f"{{'lat': {50.0 + offset_deg}, 'lon': 8.0}}",
        "adversarial_failure_reason": "",
        "accepted_for_metrics": "True",
        "post_fgo_error_m": "5.0",
    }]
    _write_run(tmp_path, "forgec_avl_wrong_near", "avl",
               "ours_ultravpr_cascade_6dof_e2e", rows)
    runs = gt.load_runs(tmp_path)
    fb = gt.fault_borderline_trial(runs)
    assert fb is not None
    assert 0 < fb["overshoot"] < gt.TILE_SLACK_M
    assert fb["tolerance"] == gt.TILE_SLACK_M


def test_ch3_macros_never_emit_pending_todos_without_optional_artifacts():
    tex = gt.ch3_macros_tex([], dev_csv=None)
    assert "\\todo" not in tex
    assert "faultContra" not in tex
    assert "faultBorder" not in tex


def test_load_runs_ignores_newer_explicitly_incomplete_run(tmp_path):
    old = tmp_path / "cell" / "vpair" / "pipeline" / "20260810_010000"
    new = tmp_path / "cell" / "vpair" / "pipeline" / "20260810_020000"
    for run, completed, marker in ((old, True, "old"), (new, False, "new")):
        run.mkdir(parents=True)
        (run / "metrics_summary.json").write_text(
            json.dumps({"paper_metrics": {"marker": marker}}), encoding="utf-8"
        )
        (run / "run_manifest.json").write_text(
            json.dumps({"run_completed": completed}), encoding="utf-8"
        )

    runs = gt.load_runs(tmp_path)

    assert len(runs) == 1
    assert Path(runs[0]["run_dir"]).name == "20260810_010000"


def test_load_runs_ignores_manifestless_metrics_directory(tmp_path):
    run = tmp_path / "cell" / "vpair" / "pipeline" / "20260810_010000"
    run.mkdir(parents=True)
    (run / "metrics_summary.json").write_text("{}", encoding="utf-8")

    assert gt.load_runs(tmp_path) == []


def test_load_runs_attaches_each_runs_own_manifest(tmp_path):
    for cell, top_k in (("a", 1), ("b", 5)):
        run = tmp_path / cell / "vpair" / "pipeline" / "20260810_010000"
        run.mkdir(parents=True)
        (run / "metrics_summary.json").write_text("{}", encoding="utf-8")
        (run / "run_manifest.json").write_text(
            json.dumps({"run_completed": True,
                        "pipeline_config": {"retrieval_top_k": top_k}}),
            encoding="utf-8",
        )

    runs = {run["cell"]: run for run in gt.load_runs(tmp_path)}

    assert runs["a"]["manifest"]["pipeline_config"]["retrieval_top_k"] == 1
    assert runs["b"]["manifest"]["pipeline_config"]["retrieval_top_k"] == 5


def test_reporter_and_injector_share_forge_footprint_constants():
    from uav_geolocalization.cli import benchmark_runner_impl as runner

    assert gt.REF_HALF_M == runner.FORGE_REF_HALF_M
    assert gt.AVL_QUERY_HALF_PER_M_ALT == runner.AVL_QUERY_HALF_PER_M_ALT
    assert gt.VISLOC_QUERY_HALF_M == runner.FORGE_QUERY_HALF_M["uav_visloc"][0]
    assert gt.TILE_SLACK_M == runner.FORGE_FOOTPRINT_SLACK_M
