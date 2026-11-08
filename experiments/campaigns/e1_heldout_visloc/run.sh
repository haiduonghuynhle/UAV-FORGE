#!/bin/bash
# E1 held-out confirmatory campaign (see PREREGISTRATION.md, amendments 1-2).
# Waits for any running campaign so it never shares the GPU slice.
# Order (amendment 2): flights by number of certified frames, attack
# conditions before the clean controls, so a time-limited run loses the
# least informative cells first.
source "$(dirname "$0")/../../lib/common.sh"
QC_PROFILE=heldout   # I8: gates of record on held-out data
ROOT=$K/results/v4_e1_heldout_visloc
HERE=$K/code/experiments/campaigns/e1_heldout_visloc
PER_MAP=$(cat "$HERE/per_map_half_m.json")
FLIGHTS=${FLIGHTS:-"09 08 03 10 05 02"}   # 06: no certified frame (amendment 2)
ATTACKS=${ATTACKS:-"wrong_near wrong_far visually_similar_wrong"}
PHASES=${PHASES:-"attacks clean"}   # a queue may run one phase at a time
while pgrep -f "experiments/campaigns/e2_repeat_record/ru[n]" >/dev/null; do sleep 120; done
# Labels gate (amendment 2): E1 waits until the validated footprint table exists.
while [ ! -f "$HERE/LABELS_READY" ]; do sleep 300; done
for f in $FLIGHTS; do
  [ -s "$HERE/footprint/footprint_${f}.csv" ] || { echo "missing footprint table for flight $f" >&2; exit 1; }
done
base_cfg() {
  echo "{\"retrieval_top_k\":10,${GATES},\"uav_visloc_locations\":\"$1\",\"fgo_max_step_xy_m\":40.0,\"fgo_motion_prior_sigma_pos_m\":80.0,\"forge_ref_half_m_per_map\":${PER_MAP},\"forge_query_half_m_table\":\"$HERE/footprint/footprint_$1.csv\""
}
case " $PHASES " in *" attacks "*)
for f in $FLIGHTS; do
  for c in $ATTACKS; do
    run_cell "$ROOT" "f${f}_${c}" uav_visloc 100000 "$(base_cfg $f),\"adversarial_mode\":\"wrong_reference\",\"adversarial_condition\":\"${c}\"}"
  done
done ;; esac
case " $PHASES " in *" clean "*)
for f in $FLIGHTS; do
  run_cell "$ROOT" "f${f}_clean" uav_visloc 100000 "$(base_cfg $f)}"
done ;; esac
finish "$ROOT" "E1 PHASES=[$PHASES] FLIGHTS=[$FLIGHTS]"
