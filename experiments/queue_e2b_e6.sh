#!/bin/bash
# Queue for E2b and E6 (pre-registered 2026-10-25). One queue at a time.
set -u
export K=${K:-/workspace/forge}
exec 9>$K/results/.queue.lock; flock -n 9 || { echo "queue already running" >&2; exit 1; }
cd $K/code
C=experiments/campaigns
QFAIL=""
step() { local label="$1"; shift; env "$@" || QFAIL="$QFAIL [$label]"; }
step E2b-vpair-rep1  DATASET=vpair  REP=rep1 bash $C/e2b_heldout_repeats/run.sh
step E6-sparse       ARM=sparse          bash $C/e6_fusion_isolation/run.sh
step E6-nofgo        ARM=nofgo           bash $C/e6_fusion_isolation/run.sh
step E2b-visloc-rep1 DATASET=visloc REP=rep1 bash $C/e2b_heldout_repeats/run.sh
step E2b-vpair-rep2  DATASET=vpair  REP=rep2 bash $C/e2b_heldout_repeats/run.sh
step E2b-visloc-rep2 DATASET=visloc REP=rep2 bash $C/e2b_heldout_repeats/run.sh
echo "[$(date +%F' '%T)] QUEUE E2b/E6 DONE${QFAIL:+ with failures:$QFAIL}" | tee -a $K/results/_QUEUE_STATUS
