#!/usr/bin/env python3
"""Patch a committed macro file with the values of a regenerated one.

Used to switch the AnyVisLoc fault label to the oblique footprint
(FORGE_AVL_FOOTPRINT=oblique) without disturbing what the regeneration does
not touch: a macro keeps its committed text when its value is unchanged
(thousands separators included), takes the regenerated text when the value
differs, and is appended when it is new. Macros only the committed file has
(from optional inputs the regeneration did not receive) are kept. A
provenance line names the regenerated source and every macro that changed.

Usage: merge_macro_values.py COMMITTED.tex REGENERATED.tex OUT.tex --note TEXT
"""
import argparse
import re
from pathlib import Path

MACRO = re.compile(r"^\\newcommand\{\\([A-Za-z]+)\}\{(.*)\}\s*$")


def parse(path):
    out = {}
    for line in Path(path).read_text().splitlines():
        m = MACRO.match(line.strip())
        if m:
            out[m.group(1)] = m.group(2)
    return out


def norm(v):
    return v.replace("{,}", "").replace("\\,", "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("committed"); ap.add_argument("regenerated"); ap.add_argument("out")
    ap.add_argument("--note", required=True)
    a = ap.parse_args()
    new = parse(a.regenerated)
    lines, changed, seen = [], [], set()
    for line in Path(a.committed).read_text().splitlines():
        m = MACRO.match(line.strip())
        if not m:
            lines.append(line); continue
        name, val = m.group(1), m.group(2)
        seen.add(name)
        if name in new and norm(new[name]) != norm(val):
            lines.append(f"\\newcommand{{\\{name}}}{{{new[name]}}}")
            changed.append(name)
        else:
            lines.append(line)
    added = [n for n in new if n not in seen]
    # Committed macros the regeneration did not produce (optional inputs it was
    # not given) are kept as committed, and named so that is never silent.
    kept = [n for n in seen if n not in new]
    lines += [f"\\newcommand{{\\{n}}}{{{new[n]}}}" for n in added]
    head = [f"% {a.note}",
            f"% values merged from {a.regenerated} by scripts/reporting/merge_macro_values.py",
            f"% changed: {' '.join(changed) or 'none'}",
            f"% added: {' '.join(added) or 'none'}",
            f"% kept from the committed file, not regenerated: {' '.join(kept) or 'none'}"]
    Path(a.out).write_text("\n".join(head + lines) + "\n")
    print(f"{Path(a.committed).name}: {len(changed)} changed, {len(added)} added, {len(kept)} kept unregenerated")


if __name__ == "__main__":
    main()
