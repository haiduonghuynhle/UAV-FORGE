#!/bin/bash
# E2b repeats of the held-out endpoint (see PREREGISTRATION.md).
# Usage: DATASET=vpair|visloc REP=rep1|rep2 bash run.sh
source "$(dirname "$0")/../../lib/common.sh"
QC_PROFILE=heldout
ROOT=$K/results/v4_e2b_heldout_repeats
C=$K/code/experiments/campaigns
DATASET=${DATASET:?}; REP=${REP:?}
for c in wrong_near wrong_far visually_similar_wrong; do
  if [ "$DATASET" = vpair ]; then
    cfg="{\"retrieval_top_k\":5,${GATES},\"query_name_list_file\":\"$C/e1b_heldout_vpair/heldout_query_names.txt\",\"adversarial_mode\":\"wrong_reference\",\"adversarial_condition\":\"${c}\"}"
    run_cell "$ROOT" "vpair_${REP}_${c}" vpair 100000 "$cfg"
  else
    H=$C/e1_heldout_visloc
    cfg="{\"retrieval_top_k\":10,${GATES},\"uav_visloc_locations\":\"09\",\"fgo_max_step_xy_m\":40.0,\"fgo_motion_prior_sigma_pos_m\":80.0,\"forge_ref_half_m_per_map\":$(cat $H/per_map_half_m.json),\"forge_query_half_m_table\":\"$H/footprint/footprint_09.csv\",\"adversarial_mode\":\"wrong_reference\",\"adversarial_condition\":\"${c}\"}"
    run_cell "$ROOT" "visloc_f09_${REP}_${c}" uav_visloc 100000 "$cfg"
  fi
done
finish "$ROOT" "E2b DATASET=$DATASET REP=$REP"
