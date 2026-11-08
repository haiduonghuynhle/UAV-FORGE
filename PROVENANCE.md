# Provenance of the pre-registered campaigns

Commit timestamps are git author dates from the authors' working repository
(this release is a cleaned snapshot, so its own history is a single commit).
First-cell start times come from each campaign's status log on the GPU host.
All times are UTC+07:00 unless marked UTC.

| Item | Commit | Committed |
|---|---|---|
| E2a protocol (repeats of the AnyVisLoc campaign of record) | 74a66a9 | 2026-10-21 20:39 |
| E1 pre-registration, held-out UAV-VisLoc | b7b9993 | 2026-10-21 20:39 |
| E1 amendment 1 (conservative envelope) | ccd4323 | 2026-10-21 20:39 |
| E1b pre-registration, held-out VPAIR | 70841e6 | 2026-10-21 20:41 |
| E1 amendment 2 (measured per-frame footprints) | 18b4837 | 2026-10-21 22:51 |
| E1 amendment 2, stricter footprint tables | 994d873 | 2026-10-21 23:40 |
| E1 amendment 3 (run order only) | b2bb279 | 2026-10-22 00:10 |
| E2b and E6 pre-registrations (exploratory) | 017f9a0 | 2026-10-25 13:21 |
| E2b and E6 amendment 1 (pairing checks, reading rules) | 4baefb0 | 2026-10-25 13:55 |

| Campaign | First cell started |
|---|---|
| E1 held-out UAV-VisLoc (flight 09) | 2026-10-21 18:53 UTC (2026-10-22 01:53) |
| E1b held-out VPAIR | 2026-10-22 00:57 UTC (2026-10-22 07:57) |
| E2b held-out repeats (VPAIR rep1) | 2026-10-25 06:21 UTC (2026-10-25 13:21) |
| E6 sparse arm | 2026-10-25 12:36 UTC (2026-10-25 19:36) |

Every E1/E1b amendment therefore precedes the first held-out cell it governs.
The E6 amendment precedes the first E6 cell. The E2b amendment was committed
while the first E2b cell ran and before any E2b output was read; it changes
the analysis only, not the cells. The
frozen footprint tables are in `experiments/campaigns/e1_heldout_visloc/footprint/`
and the frozen VPAIR frame list in `experiments/campaigns/e1b_heldout_vpair/`.
