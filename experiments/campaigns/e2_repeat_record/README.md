# E2a: run-to-run variation of the record FORGE cells (AnyVisLoc)

Frozen 2026-10-21, before any repeat ran.

Question: how much do acceptance rates and individual decisions move when the
record cells are run again with the same code, data and configuration?

- Code: the code of record (bbac198 + server patch + untracked files), tree
  hash in CODE_IDENTITY.txt. No seed control exists in that code, so each
  repeat samples the run-to-run variation the record itself was subject to.
- Environment: record pins (torch 2.4.0+cu124, numpy 1.24.4, OpenCV 4.7.0),
  see /workspace/forge/env_record_v3.txt. A 30-frame clean smoke matched the record
  on 30/30 accept decisions and every error to 0.01 m.
- Cells: forgec_avl_{wrong_near,wrong_far,visually_similar_wrong}, 487 frames,
  k=5, the gate block of record; three repeats each (rep1..rep3).
- Output: /workspace/forge/results/v4_e2_repeat/ (never results/v3).

Primary endpoints, per condition: (1) between-repeat range and SD of the
frame-level and pooled acceptance rates, set against the block-bootstrap
interval of the record; (2) decision agreement: Jaccard of accepted frame
sets and Fleiss' kappa across the record and the three repeats.
