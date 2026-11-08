#!/bin/bash
# E6 fusion isolation on held-out VPAIR (see PREREGISTRATION.md).
# Usage: ARM=sparse|nofgo bash run.sh
source "$(dirname "$0")/../../lib/common.sh"
QC_PROFILE=heldout
ROOT=$K/results/v4_e6_fusion_isolation
LIST=$K/code/experiments/campaigns/e1b_heldout_vpair/heldout_query_names.txt
ARM=${ARM:?}
base="{\"retrieval_top_k\":5,${GATES},\"query_name_list_file\":\"${LIST}\""
if [ "$ARM" = sparse ]; then extra=",\"adversarial_inject_rate\":0.2,\"adversarial_seed\":0"; CONDS="wrong_near wrong_far visually_similar_wrong"
else extra=",\"use_fgo\":false"; CONDS="clean wrong_near wrong_far visually_similar_wrong"; fi
for c in $CONDS; do
  if [ "$c" = clean ]; then cfg="${base}${extra}}"; else cfg="${base}${extra},\"adversarial_mode\":\"wrong_reference\",\"adversarial_condition\":\"${c}\"}"; fi
  run_cell "$ROOT" "vpair_${ARM}_${c}" vpair 100000 "$cfg"
done
finish "$ROOT" "E6 ARM=$ARM"
