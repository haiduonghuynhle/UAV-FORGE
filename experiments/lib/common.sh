#!/bin/bash
# Shared environment and cell runner for v4 campaigns.
# Every campaign sources this file, so the dataset paths, weights and the gate
# block of record exist in exactly one place (memory: forge-launch-gates-block).
set -u
K=${K:-/workspace/forge}
export K
cd "$K/code"
export VPAIR_DATA=$K/data/VPAir/VPAir
export AVL_DATA=$K/data/AVLData/AVLData
export VISLOC_DATA=$K/data/UAV_VisLoc_dataset
export ROMA_WEIGHTS_PATH=$K/code/Matching_Models/RoMa/weights/roma_outdoor.pth
export DINOV2_WEIGHTS_PATH=$K/code/Matching_Models/RoMa/weights/dinov2_vitl14_pretrain.pth
export HD4_RETRIEVAL_DB_CACHE_GC=0

# Gate block of record. Omitting it silently runs a different method.
GATES='"rerank_margin_score":50.0,"final_pose_max_yaw_prior_residual_deg":30.0,"geometry_min_inliers_after_lgi":25'
VISLOC_RECORD='"uav_visloc_locations":"01,04,11","fgo_max_step_xy_m":40.0,"fgo_motion_prior_sigma_pos_m":80.0'

# Cells that did not finish with a passing QC, for the launcher's exit code.
FAILED_CELLS=""

# run_cell <out_root> <name> <dataset> <max_samples> <config_json>
# One output dir, one log, one status line per cell; a .done marker lets a
# restarted campaign skip finished cells. Exit code of the runner is kept.
run_cell() {
  local root="$1" name="$2" dataset="$3" max_samples="$4" cfg="$5"
  local logs="$root/_logs"; mkdir -p "$logs"
  local status="$root/_STATUS"
  if [ -f "$root/$name/.done" ]; then echo "[$(date +%F' '%T)] SKIP  $name (done)" | tee -a "$status"; return 0; fi
  if [ -f "$root/$name/.qc_check" ]; then echo "[$(date +%F' '%T)] HOLD  $name (QC failed earlier; review before rerun)" | tee -a "$status"; FAILED_CELLS="$FAILED_CELLS $name"; return 3; fi
  echo "[$(date +%F' '%T)] START $name" | tee -a "$status"
  python3 benchmark_runner.py --dataset "$dataset" --pipeline ours_ultravpr_cascade_6dof_e2e \
    --strict-components --strict-end2end --device cuda --max-samples "$max_samples" \
    --config-json "$cfg" --output "$root/$name" > "$logs/$name.log" 2>&1
  local rc=$?
  local run_dir; run_dir=$(ls -td "$root/$name"/*/*/2* 2>/dev/null | head -1)
  local qc="-"
  # A runner that exits 0 without an output dir did not produce a cell.
  if [ $rc -eq 0 ] && [ -z "$run_dir" ]; then rc=4; fi
  # .done only after QC passes: a cell that fails QC is held, not skipped
  # as finished on resume, and its status line and return code say so.
  if [ $rc -eq 0 ] && [ -n "$run_dir" ]; then
    if python3 scripts/reporting/forge_cell_qc.py --profile "${QC_PROFILE:-record}" "$run_dir" >> "$logs/$name.log" 2>&1; then
      qc=PASS; touch "$root/$name/.done"
    else
      qc=CHECK; touch "$root/$name/.qc_check"; rc=3
    fi
  fi
  echo "[$(date +%F' '%T)] DONE  $name rc=$rc qc=$qc dir=$run_dir" | tee -a "$status"
  [ $rc -eq 0 ] || FAILED_CELLS="$FAILED_CELLS $name"
  return $rc
}

# finish <root> <label>: one status line naming every failed cell; exit
# nonzero if any failed, so a queue never reads a partial campaign as complete.
finish() {
  local root="$1" label="$2"
  if [ -n "$FAILED_CELLS" ]; then
    echo "[$(date +%F' '%T)] $label INCOMPLETE failed:$FAILED_CELLS" | tee -a "$root/_STATUS"; exit 1
  fi
  echo "[$(date +%F' '%T)] $label COMPLETE" | tee -a "$root/_STATUS"
}
