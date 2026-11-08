#!/bin/bash
# Run the pre-registered E1 / E1b analysis on every finished cell.
# Usage: run_heldout_analysis.sh e1|e1b OUT_DIR
# Only cells with a .done marker (QC passed) enter. A flight enters E1 only
# when all three attack conditions are done for it, so the frame-level union
# is never computed over a partial condition set.
set -u
K=${K:-/workspace/forge}; C=$K/code; A=$C/experiments/analysis
which=$1; out=$2; mkdir -p "$out"
export PYTHONPATH=$C/scripts/reporting:${PYTHONPATH:-}
cell_dir() { ls -td "$1"/*/*/2* 2>/dev/null | head -1; }
if [ "$which" = e1 ]; then
  R=$K/results/v4_e1_heldout_visloc; H=$C/experiments/campaigns/e1_heldout_visloc
  declare -A D; tables=""; run=""
  for f in 09 08 03 10 05 02; do
    ok=1; for c in wrong_near wrong_far visually_similar_wrong; do [ -f "$R/f${f}_$c/.done" ] || ok=0; done
    [ $ok -eq 1 ] || continue
    run="$run,$f"; tables="$tables,$H/footprint/footprint_$f.csv"
    for c in wrong_near wrong_far visually_similar_wrong; do D[$c]="${D[$c]:-},$(cell_dir $R/f${f}_$c)"; done
  done
  [ -n "$run" ] || { echo "no complete E1 flight yet"; exit 2; }
  python3 $A/heldout_analysis.py \
    --cond wrong_near=${D[wrong_near]#,} --cond wrong_far=${D[wrong_far]#,} \
    --cond visually_similar_wrong=${D[visually_similar_wrong]#,} \
    --dataset uav_visloc --radius 150 --block 0 --env-ref 277.9 305.8 \
    --ref-per-map $H/per_map_half_m.json --query-table ${tables#,} \
    --out $out/e1_analysis.json
  echo "${run#,}" > $out/e1_flights_run.txt
else
  R=$K/results/v4_e1b_heldout_vpair; args=""
  for c in wrong_near wrong_far visually_similar_wrong; do
    [ -f "$R/vpair_heldout_$c/.done" ] || { echo "E1b $c not done"; exit 2; }
    args="$args --cond $c=$(cell_dir $R/vpair_heldout_$c)"
  done
  python3 $A/heldout_analysis.py $args --dataset vpair --radius 20 --block 0 --out $out/e1b_analysis.json
fi
