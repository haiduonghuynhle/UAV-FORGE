#!/bin/bash
# E1b VPAIR within-route held-out (see PREREGISTRATION.md). Runs after E1.
source "$(dirname "$0")/../../lib/common.sh"
QC_PROFILE=heldout   # I8: gates of record on held-out data
ROOT=$K/results/v4_e1b_heldout_vpair
LIST=$K/code/experiments/campaigns/e1b_heldout_vpair/heldout_query_names.txt
while pgrep -f "experiments/campaigns/e[12]_[a-z_]*/ru[n]" >/dev/null; do sleep 120; done
base="{\"retrieval_top_k\":5,${GATES},\"query_name_list_file\":\"${LIST}\""
CONDS=${CONDS:-"clean wrong_near wrong_far visually_similar_wrong"}
for c in $CONDS; do
  if [ "$c" = clean ]; then cfg="${base}}"; else cfg="${base},\"adversarial_mode\":\"wrong_reference\",\"adversarial_condition\":\"${c}\"}"; fi
  run_cell "$ROOT" "vpair_heldout_${c}" vpair 100000 "$cfg"
done
finish "$ROOT" "E1b CONDS=[$CONDS]"
