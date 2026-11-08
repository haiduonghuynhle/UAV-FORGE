#!/usr/bin/env python3
"""Emit the mixed-rate FORGE confusion table as LaTeX, from the analysis JSON.

One row per dataset and condition, reported per protocol: a frame is counted
under the reference it actually received, not the arm it was assigned to. So a
control frame whose own retrieval returned a wrong tile counts as a wrong
reference, and an assigned frame the injector could not attack is dropped
rather than counted, because the pipeline rejects it without ever consulting
the gate.

Written rather than hand-typed so the deck and the thesis cannot drift from the
run that produced them.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

LABEL = {"wrong_near": "nearby", "wrong_far": "far",
         "visually_similar_wrong": "look-alike"}
PAPER_LABEL = {"wrong_near": "proximal", "wrong_far": "distal",
               "visually_similar_wrong": "appearance-similar"}
DATASET = {"vpair": "VPAIR", "avl": "AVL", "visloc": "UAV-VisLoc"}


def rows_from(path: Path, dataset: str, labels: Dict[str, str] = LABEL) -> List[Dict[str, Any]]:
    out = []
    for entry in json.loads(path.read_text()):
        strata = entry["reference_strata"]
        control, attacked = strata["control"], strata["attacked"]
        # Correct-reference column is the control arm only. An assigned frame
        # whose shortlist held no eligible wrong tile does NOT keep a usable
        # reference: the injector fails closed with no reference image
        # (benchmark_runner_impl.py:4603), so the pipeline rejects it
        # mechanically. That is not the gate judging a good frame, and counting
        # it as a false alarm would punish the gate for a frame it never saw.
        # These exclusions are counted and must be disclosed next to the table.
        correct_accept = control["correct_ref_accept"]
        correct_reject = control["correct_ref_reject"]
        fail_closed = attacked["fail_closed"]
        if attacked["correct_ref_accept"] + attacked["correct_ref_reject"]:
            raise SystemExit("attacked arm holds correct-reference rows: treated frames must carry a wrong reference")
        # A wrong reference is a wrong reference whichever arm produced it:
        # control frames whose own retrieval went wrong count too.
        wrong_accept = control["wrong_ref_accept"] + attacked["wrong_ref_accept"]
        wrong_reject = control["wrong_ref_reject"] + attacked["wrong_ref_reject"]
        n_correct = correct_accept + correct_reject
        n_wrong = wrong_accept + wrong_reject
        if not n_correct or not n_wrong:
            continue
        tpr = correct_accept / n_correct
        leak = wrong_accept / n_wrong
        out.append({
            "dataset": DATASET.get(dataset, dataset),
            "condition": labels.get(entry["condition"], entry["condition"]),
            "correct_accept": correct_accept, "correct_reject": correct_reject,
            "wrong_accept": wrong_accept, "wrong_reject": wrong_reject,
            "tpr": tpr, "leak": leak, "bal": (tpr + 1.0 - leak) / 2.0,
            "fail_closed": fail_closed,
        })
    return out


def render(rows: List[Dict[str, Any]]) -> str:
    lines = [
        r"\begin{tabular}{llrrrrrr}",
        r"\toprule",
        r"& & \multicolumn{2}{c}{\bfseries Correct reference}"
        r" & \multicolumn{2}{c}{\bfseries Wrong reference} & & \\",
        r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}",
        r"\bfseries Dataset & \bfseries Condition & \bfseries acc.\ & \bfseries rej.\ "
        r"& \bfseries acc.\ & \bfseries rej.\ & \bfseries TPR & \bfseries Bal.\ acc. \\",
        r"\midrule",
    ]
    seen = set()
    for row in rows:
        name = row["dataset"] if row["dataset"] not in seen else ""
        seen.add(row["dataset"])
        lines.append(
            f"{name} & {row['condition']} & {row['correct_accept']} & {row['correct_reject']}"
            f" & {row['wrong_accept']} & {row['wrong_reject']}"
            f" & {row['tpr']:.3f} & {row['bal']:.3f} \\\\"
        )
    # Nothing after \end{tabular}: \thesistable puts the file in LR mode, where
    # a trailing paragraph is a fatal error. The excluded-frame count is
    # reported by the caller instead.
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--paper", action="store_true",
                        help="paper labels (proximal/distal/appearance-similar); no thesis-relative wording")
    args = parser.parse_args()

    rows: List[Dict[str, Any]] = []
    for dataset in ("vpair", "avl", "visloc"):
        path = args.analysis_dir / f"analysis_{dataset}.json"
        if not path.exists():
            raise SystemExit(f"missing {path}")
        rows += rows_from(path, dataset, PAPER_LABEL if args.paper else LABEL)

    header = (
        "% Generated by scripts/reporting/gen_mixed_confusion_table.py\n"
        f"% source: {args.analysis_dir}/analysis_{{vpair,avl,visloc}}.json\n"
        "% Mixed-rate FORGE, one run per cell at injection rate 0.5, reported\n"
        "% per protocol (the reference the frame actually received).\n"
        + ("" if args.paper else "% Post-submission: not in the submitted thesis.\n")
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(header + render(rows) + "\n")
    dropped = sum(r["fail_closed"] for r in rows)
    print(f"wrote {args.out} ({len(rows)} rows; "
          f"{dropped} fail-closed frames excluded - report this alongside the table)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
