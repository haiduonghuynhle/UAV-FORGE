# FORGE — wrong-reference integrity testing for UAV visual position fixes

Source code for FORGE. It contains the gated
pipeline, the injection harness, the pre-registered campaigns with their
dated amendments, and the scripts that turn run outputs into every number
in the paper. Datasets, model weights and per-frame records are not
included: VPAIR is licensed for academic use only, UAV-VisLoc publishes no
licence (citation requested), and the AnyVisLoc licence forbids derived
per-frame records. The harness regenerates the per-frame logs from the
datasets; see below.

## Layout

| Path | Content |
|---|---|
| `benchmark_runner.py`, `src/` | Gated pipeline and FORGE injection (runner: `src/uav_geolocalization/cli/benchmark_runner_impl.py`) |
| `Matching_Models/`, `Retrieval_Models/`, `reference/` | Matcher, retrievers and the third-party OrthoLoC back end (code only) |
| `experiments/lib/common.sh` | One cell runner and the gate block of record |
| `experiments/campaigns/` | E1 (held-out UAV-VisLoc), E1b (held-out VPAIR), E2a (repeats), E2b (held-out repeats), E6 (fusion and injection density): pre-registrations, amendments, launchers, frozen inputs |
| `experiments/analysis/` | Pre-registered analyses (`heldout_analysis.py`, `run_heldout_analysis.sh`), footprint estimator (`e5b_footprint_table.py`), repeats (`e2a_repeat_analysis.py`, `repeat_agreement.py`), fusion and density arms (`e6_fusion_analysis.py`) |
| `scripts/reporting/` | Macro generators for the paper tables (`gen_v3_tables.py`, `forge_block_ci.py`, `make_heldout_macros.py`, ...) and the per-cell QC (`forge_cell_qc.py`) |
| `tests/` | Unit tests |
| `PROVENANCE.md` | Commit dates of the pre-registrations and amendments, first-cell start times |

## Environment

Python 3.10, CUDA 12.4. `requirements-record.txt` pins the versions the
recorded runs used (torch 2.4.0+cu124, numpy 1.24.4, OpenCV 4.7.0.72).
Model weights are the public RoMa and DINOv2 checkpoints; set
`ROMA_WEIGHTS_PATH` and `DINOV2_WEIGHTS_PATH`.

## Data

Public datasets: AnyVisLoc (Ye et al., CVPR 2026 Findings), VPAIR
(Schleiss et al., 2022) and UAV-VisLoc (Xu et al., 2024). Point
`AVL_DATA`, `VPAIR_DATA` and `VISLOC_DATA` at the extracted releases
(defaults: `./data/...`). Launchers expect a workspace root `K`
(default `/workspace/forge`) holding `code/`, `data/` and `results/`.

## Reproducing the paper

1. Unit tests: `PYTHONPATH=src:.:scripts/reporting:tests python -m unittest discover -s tests`.
2. Held-out campaigns, one GPU each, in the pre-registered order:
   `bash experiments/campaigns/e1_heldout_visloc/run.sh` and
   `bash experiments/campaigns/e1b_heldout_vpair/run.sh`. The E1 launcher
   waits for `LABELS_READY`, which marks the frozen footprint tables. Each cell is
   checked by `scripts/reporting/forge_cell_qc.py --profile heldout`.
3. Analyses: `bash experiments/analysis/run_heldout_analysis.sh e1 OUT` and `... e1b OUT`.
   Exploratory follow-ups (E2b repeats, E6 arms) run after them:
   `bash experiments/queue_e2b_e6.sh`, then
   `experiments/analysis/repeat_agreement.py` (run 0 = the E1/E1b cells,
   `--expect-run0 A/N` with the accepted and attacked frame counts that
   your run 0 printed;
   the flag only verifies the run of record and is not used for new runs) and
   `experiments/analysis/e6_fusion_analysis.py`. Both scripts stop unless
   every paired trial has the same frame and forced tile.
4. Paper macros: `scripts/reporting/make_heldout_macros.py` for the
   held-out tables, `make_e2b_e6_macros.py` for the repeats and E6; for the development tables run the generators with
   `FORGE_AVL_FOOTPRINT=oblique` (the projected AnyVisLoc footprint).
   With the variable unset, they reproduce the nadir-label values of
   record.

Runs are not seeded (matching and robust estimation sample internally);
repeats agree on rates, not on individual frames (paper, Sec. V-D).

## Paper macros and their generators

Every `tables/*_numbers.tex` file of the paper is generated; each value
carries a `% source` line. The complete map, macro file to generator in
`scripts/reporting/`:

| Macro file | Generator |
|---|---|
| `heldout_visloc_numbers`, `heldout_vpair_numbers` | `make_heldout_macros.py` |
| `e2a_numbers` | `make_e2a_macros.py` |
| `e2b_e6_numbers` | `make_e2b_e6_macros.py` |
| `footprint_relabel_numbers` | `make_footprint_relabel_macros.py` |
| `abl_points_numbers` | `make_points_macros.py` |
| `v3_numbers` | `gen_v3_tables.py` |
| `v3_numbers_numeric` | `derive_slide_numbers.py` (numeric twins of `v3_numbers`) |
| `forge_block_ci_numbers`, `avl_label_sensitivity` | `forge_block_ci.py` |
| `tradeoff_numbers` | `gen_tradeoff_macros.py` |
| `native_forge_numbers` | `gen_native_forge_macros.py` |
| `native_gate_numbers` | `native_gate_replay.py` |
| `gate_rules_numbers` | `gate_rule_replay.py` |
| `dose_response_numbers` | `forge_dose_response.py` |
| `threshold_sweep_numbers` | `forge_threshold_sweep.py` |
| `bias_sensitivity_numbers` | `forge_bias_sensitivity.py` |
| `arm_audit_numbers` | `forge_arm_separation_audit.py` |
| `review_numbers` | `gen_review_macros.py` |
| `postsub_numbers` | `gen_book_postsub_macros.py`, `gen_mixed_confusion_table.py` |
| `caves_indomain_numbers` | `caves_indomain_cv.py` |
| `lodo_strata_numbers` | `caves_learned_weights.py` |
| `risk_coverage_numbers` | `caves_risk_coverage.py` |
| `weight_sensitivity_numbers` | `caves_weight_sensitivity.py` |
| `visloc_ablation_numbers` | `gen_visloc_ablation_macros.py` |
| `aicap_fig_numbers` + leak figure | `make_aicap_leak_figure.py` (`--heldout-visloc`, `--heldout-vpair` for panel (b)) |
| Fig. 2(a) imagery | `make_aicap_imagery.py` (needs the UAV-VisLoc release) |
| `heldout_alert_numbers` | `make_heldout_alert_macros.py` (held-out forced fixes against fixed alert limits) |
| `yawoff_numbers` | `make_yawoff_macros.py`, after `forge_threshold_sweep.py --inliers 25 --yaw 30 180` (attitude check alone) |
| `clean_bounded_numbers` | `make_clean_bounded_macros.py` (clean run on frames with a measured footprint) |
| `wrapper_pooled_numbers` | `make_wrapper_pooled_macros.py` (AnyVisLoc with `FORGE_AVL_FOOTPRINT=oblique`) |

`merge_macro_values.py` merges regenerated values into an existing macro
file without touching macros from other sources. `protocol_constants.tex`
holds the frozen protocol constants of record; its `% source` lines name
the defining module for each value. Generators read campaign outputs under
`$RESULTS` (the `results/` tree a run writes); none of those outputs ship
in this repository.

## License

Copyright 2026 the FORGE authors. The code in this repository is licensed
under the Apache License 2.0 (`LICENSE`), except the third-party code
listed below, which keeps its own licence. See `THIRD_PARTY_NOTICES.md`.

| Path | Upstream | Licence |
|---|---|---|
| `Matching_Models/RoMa/` | [Parskatt/RoMa](https://github.com/Parskatt/RoMa) | MIT |
| `Matching_Models/RoMa/roma/models/transformer/` (DINOv2 parts) | [facebookresearch/dinov2](https://github.com/facebookresearch/dinov2) | Apache-2.0 |
| `Retrieval_Models/CAMP/` | [Mabel0403/CAMP](https://github.com/Mabel0403/CAMP) | MIT |
| `reference/UltraVPR/` | UltraVPR | MIT |
| `reference/ortholoc/` | [deepscenario/OrthoLoC](https://github.com/deepscenario/OrthoLoC) | CC BY-NC-SA 4.0 (non-commercial) |
