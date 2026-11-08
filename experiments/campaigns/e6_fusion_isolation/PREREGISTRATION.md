# E6: how much of forced acceptance runs through fusion

Frozen 2026-10-25, before any E6 cell ran.

## Question
The final gate decides on the fused fix, and fusion keeps every visual fix,
including refused and forced ones (no rollback). In an E1b attack run every
attacked frame is forced, so a frame's outcome can depend on earlier forced
fixes. E6 separates two parts: (a) history: does a forced frame accept
differently when its neighbours were clean? (b) fusion itself: what happens
to acceptance and error with fusion switched off?

## Population
VPAIR held-out frames (E1b list, 1,897 frames), the three conditions.

## Arms
- E6a sparse injection: adversarial_inject_rate 0.2, adversarial_seed 0
  (hash of the frame name), so about one frame in five is forced and its
  neighbours mostly run clean. Everything else as E1b.
- E6b no fusion: use_fgo false, injection as E1b (rate 1), plus one clean
  cell for coverage. The visual fix is published directly.

## Endpoints
1. History effect: on frames forced in E6a, frame-level acceptance in E6a
   vs the same frames in E1b (paired, contiguous-block bootstrap of the
   difference).
2. Fusion effect: on E1b attacked frames, frame-level acceptance and the
   misleading share in E6b vs E1b (paired); clean coverage E6b vs E1b clean.
3. Median error of accepted forced fixes in each arm.

## Reading
A history effect near zero means the E1b rate is a per-frame quantity; a
large one means it partly measures accumulated contamination. The fusion
arm shows whether fusion pulls forced fixes toward truth (lower error,
possibly more accepted) or away.

## Amendment 1 (2026-10-25, before any E6 cell ran)
Written after an independent design audit. No E6 output existed; the only
E2b output was the first VPAIR repeat cell, still running and unread.
1. Name of the first contrast. The sparse arm still has attacked history
   (one frame in five), and fusion state carries over from initialisation
   and earlier fixes. The contrast is therefore "injection density 0.2 vs
   1.0", not "clean history vs attacked history". A difference near zero
   does not show that the E1b rate is history-free; it bounds how much the
   rate moves with this density change.
2. Pairing. Trials pair on (condition, frame, forced tile). Every sparse
   trial must exist in E1b with the same tile; the no-fusion arm must force
   exactly the E1b trials. The analysis stops on any mismatch and never pairs
   on an intersection. (Checked on the smoke cells: 7/7 and 35/35 identical
   tiles; retrieval reads telemetry altitude only, never fusion state.)
3. Run-to-run reference. Each paired difference mixes the arm effect with
   run-to-run variation. The same paired difference between an E2b dense
   repeat and E1b, on the same trials, is reported beside each E6 contrast.
   It is a reference, not a correction subtracted from the E6 difference.
4. Endpoints. Endpoint 1 and 2 report the frame-level accepted rate (as
   E1b) and, separately, the unsafe-frame rate (any accepted trial with
   error > 20 m) and the misleading share of accepted trials. An accepted
   trial with no finite error counts as unknown, never as safe.
5. Fusion and error. Endpoint 3 stays descriptive: the arms accept different
   trials, so the two medians are not a like-for-like comparison. Added: the
   pre- vs post-fusion error on the same E1b trials, and the error difference
   on trials accepted in both arms.
6. Uncertainty. Blocks follow name order, which is flight order on VPAIR,
   as in E1b; half and double block length are reported. A confidence
   interval that contains zero is not read as "no effect"; no equivalence
   margin is set, so no equivalence claim is made.
7. Status. Both contrasts are exploratory and descriptive; no multiplicity
   adjustment is applied, and the paper does not report them as confirmatory.
8. Clean coverage pairs the two clean cells on the same frames and reports
   the discordant frames.
