# E1: held-out confirmatory FORGE campaign on UAV-VisLoc

Frozen 2026-10-21. Code: record tree + opt-in per-map footprint (commit
"opt-in per-map footprint" on top of the record tree; CODE_IDENTITY.txt).
Nothing below may change after the first E1 cell starts; later changes go in a
dated section with their reason.

## Question
On flights that never informed any threshold or constant, how often does the
gated cascade accept a forced wrong reference?

## Population
UAV-VisLoc flights 02, 03, 05, 06, 08, 09, 10: every frame of each flight.
Flight 07 is excluded a priori: its raster is 3000x170 px, smaller than one
2048-px tile, so retrieval has one whole-raster reference and every frame
fails closed. Flights 01, 04, 11 (development) are not used.

## Frozen configuration
- Pipeline ours_ultravpr_cascade_6dof_e2e, --strict-components --strict-end2end,
  retrieval_top_k 10, the gate block of record, fgo_max_step_xy_m 40,
  fgo_motion_prior_sigma_pos_m 80, uav_visloc_protocol intra_flight.
- One run per flight and condition (single-flight corpus; FGO state does not
  cross flights).
- Reference half-extents per map, measured from each raster before any run
  (tile_half_extent_measured.json; 1024 px x metres per pixel). Query
  half-extent 101 m and slack 15 m stay at the record values.
- Conditions wrong_near, wrong_far, visually_similar_wrong, plus a clean
  control run per flight for coverage.

## Primary endpoint
Frame-level acceptance (at least one forced tile accepted across the three
conditions) over attacked frames, pooled over flights, with a two-level
bootstrap (flights, then contiguous blocks within a flight), 2000 resamples.

## Precision target and expected size
Half-width <= 3 points at a rate near 7%, design effect 2: about 560 attacked
frames. The held-out set has 4,599 frames; at the record's yield it gives
far more than that.

## Secondary endpoints
Pooled trial acceptance per condition; unsafe share of accepted trials;
median error of accepted forced fixes; per-flight rates (descriptive);
trials inconsistent with the label (accepted inside R), reported separately.

## How the result will be read
The development estimate on UAV-VisLoc (frame level 5.3% [3.7, 6.9]) is a
prediction. If the held-out interval overlaps it, the development rate
transfers. If not, the paper reports the held-out rate as primary and the
development rate as optimistic or pessimistic accordingly.

## Amendment 1 (2026-10-21, before any E1 cell started)
Prompted by an external methods review of this design.

- Label sensitivity, conservative envelope. A forced tile counts as a fault in
  the primary analysis under the frozen constants above. A secondary analysis
  keeps only trials whose forced tile stays disjoint after inflating both
  half-extents to a conservative envelope: the reference side by the largest
  per-flight value measured over all 11 flights (277.9 m east, 305.8 m north),
  the query side by the upper 95th percentile of an independent ground
  footprint estimate (consecutive-frame self-calibration against the GT
  trajectory, computed on development flights 01/04/11 only, before E1
  results are read). The share of trials that survive, and the rates on them,
  are reported next to the primary rates. The envelope is computed from
  metadata and imagery, never from localization outcomes.
- Clusters. Per-flight estimates are reported first; the aggregate weights
  frames equally. Uncertainty: two-level bootstrap (flights, then contiguous
  blocks), with block length halved and doubled as a sensitivity check.
- Denominator. Frames with at least one valid forced tile across the three
  conditions; attack coverage over all frames is reported separately.

## Amendment 2 (2026-10-21, before any E1 cell started)
Prompted by the development-only footprint estimate of amendment 1 (E5,
flights 01/04/11). Consecutive-frame self-calibration gives median query
half-extents of 188, 262 and 363 m (long side), against the frozen 101 m.
A 101 m constant would label as faults some forced tiles that still overlap
the camera footprint, so it cannot define the population of a confirmatory
run.

- Query footprint, per frame, fixed before injection. Each frame's ground
  sampling distance comes from its pairs with neighbouring frames up to
  three apart, on either side: the GT displacement divided by the matched
  pixel displacement. A pair is used only if it passes all checks: both
  singular values of the affine part within 3% of 1 (rejects altitude
  change, anisotropic stretch and shear), altitude change <= 1%, heading
  change <= 5 degrees, projective terms <= 2e-4 per full-resolution pixel,
  at least 60 inliers at inlier ratio >= 0.5, and a baseline >= 30 m. A
  passing pair has the same GSD at both ends to within 3%, so it counts
  for both frames. The bound is the half-diagonal of the
  image at the median GSD, inflated by max(10%, twice the relative
  dispersion). It is applied to both axes.
- Indeterminate frames. A frame with fewer than 3 valid pairs, or a relative
  dispersion above 0.10, has no certified bound. It is never attacked, and
  it is reported in attack coverage. It is not counted as a non-acceptance.
- Implementation. The bounds are in footprint/footprint_<flight>.csv
  (experiments/analysis/e5b_footprint_table.py). The runner reads them
  through forge_query_half_m_table. A frame that is missing or not OK gets
  an infinite half-extent, so no tile is eligible for it. The bound decides
  eligibility and therefore tile selection, as the frozen constant did.
- Table summary (frozen by this commit):

  | Flight | Frames | OK | Too few pairs | Dispersed | Bound, median [min, max] m |
  |---|---|---|---|---|---|
  | 02 | 1071 | 20 | 1048 | 3 | 249.2 [223.1, 269.9] |
  | 03 | 768 | 91 | 677 | 0 | 300.5 [260.2, 330.3] |
  | 05 | 473 | 15 | 457 | 1 | 265.3 [230.1, 334.1] |
  | 06 | 344 | 0 | 344 | 0 | none |
  | 08 | 1033 | 304 | 722 | 7 | 358.2 [307.1, 410.6] |
  | 09 | 766 | 452 | 314 | 0 | 375.1 [348.1, 397.7] |
  | 10 | 144 | 7 | 137 | 0 | 185.8 [170.4, 208.9] |

  889 of 4,599 held-out frames carry a certified bound. Flight 06 has
  none, so it cannot be attacked and is not run. The estimand is
  therefore restricted: acceptance of forced tiles disjoint under the measured envelope
  among certified frames. It is not a whole-flight rate, and the paper
  says so.
- Sensitivities. The frozen 101 m definition and the flight-median
  definition are reported as relabels of the E1 trials, with the number
  of trials that change label under each.
  The development cells (flights 01/04/11) are relabelled with the same
  tables (heldout_analysis.py --query-table). The paper reports both labels
  and names the per-frame table as primary.
- Bound scale sensitivity. The primary analysis is repeated on the subset
  of trials that stay disjoint with every bound multiplied by 1.15 and by
  1.30 (heldout_analysis.py --query-scales). Agreement supports, but does
  not validate, the bound.
- A flight-level envelope for uncertified frames (1.15 x the flight's
  largest bound, flights with >= 30 OK frames, height within 0.98-1.02 of
  the OK height range) is exploratory only and is not run in E1.
- Run order and budget. GPU time is limited to about three days. The queue
  (experiments/resume_campaigns.sh) runs, in this order: E1 attacks on
  flights 09, 08, 03; E1b attacks; E1 clean controls on 09, 08, 03; E1b
  clean control; E1 attacks on 10, 05, 02; their clean controls. The
  controls of the first three flights come before the attacks of the last
  three because those hold few certified frames (42 in all). The order uses table
  counts only, never outcomes. Cells not reached are reported as not run,
  with their counts.
- Wording. "Certified" names a table status: the frame passed every check
  above. The bound is an estimate with a margin, not a guarantee, and the
  paper calls such tiles disjoint under the measured envelope.
- Hazardous acceptance (accepted and error > R) over all trials is reported
  next to the primary endpoint. Under this design every trial is disjoint,
  so it equals the unsafe share times the acceptance rate; it is not a new
  primary endpoint.
- E1 starts only after this amendment is committed and LABELS_READY is
  created on the server. Precision is re-checked against the attacked-frame
  count after the first flight: if fewer than 560 attacked frames are
  projected, the shortfall is reported, and no flights are added.

## Amendment 3 (2026-10-22, before any E1 cell started)
Queue order only. The smoke on flight 09 (40 frames, run before E1, read
for eligibility and timing only) attacked 35 of 40 frames and took about
16 s per frame. At that speed the attack cells of flights 09, 08 and 03
take about 34 hours, so E1b attacks (about 6 hours) move up to run right
after flight 09: E1 attacks on 09; E1b attacks; E1 attacks on 08 and 03;
then the controls and the remaining flights as in amendment 2. No
population, endpoint or analysis changes.
