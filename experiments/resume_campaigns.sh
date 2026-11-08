#!/bin/bash
# Resume the v4 campaign queue after a restart. Order fits a ~3-day GPU
# budget (E1 amendment 2): E2a, then attack cells of the most informative
# E1 flights, then E1b attacks, then clean controls, then the rest of E1.
# Cells with a .done marker are skipped by run_cell; a cell interrupted
# mid-run has no marker, so its partial output is moved aside first.
set -u
# One canonical root for every path: launchers, run_cell and the pgrep
# guard below must spell output paths the same way, or the guard misses a
# live cell and moves it.
[ -e /workspace/forge ] || ln -s /workspace/forge /workspace/forge
export K=/workspace/forge
# Never run two queues: a second one would move the first one's live cells.
exec 9>$K/results/.queue.lock; flock -n 9 || { echo "queue already running" >&2; exit 1; }
for root in $K/results/v4_e2_repeat $K/results/v4_e1_heldout_visloc $K/results/v4_e1b_heldout_vpair; do
  [ -d "$root" ] || continue
  for cell in "$root"/*/; do
    name=$(basename "$cell"); case "$name" in _*) continue;; esac
    [ -f "$cell/.done" ] || [ -f "$cell/.qc_check" ] && continue
    if ! pgrep -f -- "--output $root/$name( |$)" >/dev/null; then
      mkdir -p "$root/_interrupted"; mv "$cell" "$root/_interrupted/${name}_$(date +%Y%m%d_%H%M%S)"
      echo "[$(date +%F' '%T)] MOVED interrupted $name" | tee -a "$root/_STATUS"
    fi
  done
done
cd $K/code
E1=experiments/campaigns/e1_heldout_visloc/run.sh
E1B=experiments/campaigns/e1b_heldout_vpair/run.sh
while pgrep -f "experiments/campaigns/e2_repeat_record/ru[n]" >/dev/null; do sleep 120; done
# Each step runs even if an earlier one had a failed cell, so the GPU never
# idles overnight; failures are listed and the queue exits nonzero.
QFAIL=""
step() { local label="$1"; shift; env "$@" || QFAIL="$QFAIL [$label]"; }
step E2a PARALLEL=1 bash experiments/campaigns/e2_repeat_record/run.sh
step E1-attacks-09 PHASES=attacks FLIGHTS="09" bash $E1
step E1b-attacks CONDS="wrong_near wrong_far visually_similar_wrong" bash $E1B
step E1-attacks-0803 PHASES=attacks FLIGHTS="08 03" bash $E1
step E1-clean-090803 PHASES=clean FLIGHTS="09 08 03" bash $E1
step E1b-clean CONDS=clean bash $E1B
step E1-attacks-100502 PHASES=attacks FLIGHTS="10 05 02" bash $E1
step E1-clean-100502 PHASES=clean FLIGHTS="10 05 02" bash $E1
if [ -n "$QFAIL" ]; then
  echo "[$(date +%F' '%T)] QUEUE DONE with failures:$QFAIL" | tee -a $K/results/_QUEUE_STATUS; exit 1
fi
echo "[$(date +%F' '%T)] QUEUE COMPLETE" | tee -a $K/results/_QUEUE_STATUS
