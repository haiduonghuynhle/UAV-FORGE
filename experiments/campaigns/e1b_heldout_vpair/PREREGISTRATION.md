# E1b: within-route held-out FORGE campaign on VPAIR

Frozen 2026-10-21, before any E1b cell ran. Supporting evidence only: VPAIR is
one route, so these frames share sensor, season and terrain type with the
development frames. They are not a new environment.

## Population
Frames 801-2706 of the VPAIR route, minus every frame whose footprint could
overlap a development footprint (frames 1-800): circumscribed footprint radius
210 m (336 x 252 m view), so centres closer than 435 m (2 x 210 + 15 m slack)
are removed. 1,897 of 1,906 frames remain (heldout_query_names.txt, built by
build_heldout_list.py from poses.csv alone).

## Frozen configuration
Record pipeline and gate block, retrieval_top_k 5, footprint constants of
record (VPAIR views are 336 x 252 m on both sides, measured), clean control
plus wrong_near, wrong_far, visually_similar_wrong, one run each. The query
list is passed with the opt-in key query_name_list_file.

## Endpoints
As E1: primary frame-level acceptance over attacked frames (at least one of
the three conditions), contiguous-block bootstrap; secondaries as in E1. The
development estimate (12.7% [8.8, 16.4]) is the prediction.
