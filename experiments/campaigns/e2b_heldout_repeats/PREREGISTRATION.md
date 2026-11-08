# E2b: run-to-run variation of the held-out endpoint

Frozen 2026-10-25, before any E2b cell ran.

## Question
How much do the pre-registered held-out rates (E1, E1b) and individual
decisions move when the same frames, forced tiles and configuration run again?
The record code has no seed control, so each repeat samples the run-to-run
variation of the matcher and robust estimator.

## Cells
- VPAIR held-out (E1b population, 1,897 frames): wrong_near, wrong_far,
  visually_similar_wrong; repeats rep1 and rep2.
- UAV-VisLoc held-out flight 09 (E1 population, 449 attacked frames, the
  largest flight): the same three conditions; repeats rep1 and rep2.
Configuration identical to E1/E1b (gate block of record, frozen footprint
table, query list, QC profile heldout). Outputs: results/v4_e2b_heldout_repeats.

## Endpoints
Per dataset, with the original E1/E1b run as run 0:
1. Frame-level rate per run; range and standard deviation across runs.
2. Decision agreement on attacked frames: Fleiss' kappa across runs, mean
   pairwise Jaccard of accepted frames, frames accepted in all runs vs any run.
3. Misleading share of accepted trials per run.
Runs pair only if they attacked the same frames with the same forced tiles.

## Reading
If the rates of repeats stay inside the E1/E1b intervals, the reported
intervals are not made too narrow by ignoring run variation; if not, the
paper reports the spread as an additional uncertainty term.

## Order and budget
Queue order (GPU time to 2026-10-28): VPAIR rep1, E6, UAV-VisLoc rep1,
VPAIR rep2, UAV-VisLoc rep2. Cells not reached are reported as not run.

## Amendment 1 (2026-10-25, while VPAIR rep1 cell 1 was running, unread)
Written after an independent design audit; no E2b output had been read.
1. Population. The analysis uses the population of record
   (gen_v3_tables._attacked_rows, as in heldout_analysis.py). The extra
   genuine-fault filter is a sensitivity only. Run 0 must reproduce the
   reported counts or the analysis stops: VPAIR 106/1,066 frames (E1b);
   UAV-VisLoc flight 09 18/449 frames (E1, that flight only). Both filters
   give the same run-0 counts.
2. Endpoints add the unsafe-frame rate and the misleading share of accepted
   trials; accepted trials with no finite error count as unknown.
3. Reading. The E1/E1b intervals resample frames within one run; they are
   not prediction intervals for a new run. Repeats falling inside them does
   not show that the intervals are wide enough. The paper reports the
   spread across runs (range, SD, kappa, all-vs-any) as a separate
   uncertainty term, with three runs per dataset at most, so the spread is
   a description, not a stable estimate of run variance.
4. Queue order stays as frozen: UAV-VisLoc rep1 before VPAIR rep2, so each
   dataset gets one repeat before either gets two. The queue script is not
   edited while it runs.
