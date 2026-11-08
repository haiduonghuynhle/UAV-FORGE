#!/bin/bash
# E2a launcher. PARALLEL cells share the GPU; 2 fit in a 20 GB MIG slice.
source "$(dirname "$0")/../../lib/common.sh"
ROOT=$K/results/v4_e2_repeat
PARALLEL=${PARALLEL:-2}
jobs_list=()
for rep in rep1 rep2 rep3; do
  for cond in wrong_near wrong_far visually_similar_wrong; do
    jobs_list+=("${rep}_forgec_avl_${cond}|$cond")
  done
done
for item in "${jobs_list[@]}"; do
  name=${item%%|*}; cond=${item##*|}
  while [ "$(jobs -rp | wc -l)" -ge "$PARALLEL" ]; do sleep 20; done
  run_cell "$ROOT" "$name" avl 487 "{\"retrieval_top_k\":5,${GATES},\"adversarial_mode\":\"wrong_reference\",\"adversarial_condition\":\"${cond}\"}" &
  sleep 60
done
wait
# Cells ran in background subshells, so failures are read back from markers.
for item in "${jobs_list[@]}"; do
  name=${item%%|*}; [ -f "$ROOT/$name/.done" ] || FAILED_CELLS="$FAILED_CELLS $name"
done
finish "$ROOT" "E2a"
