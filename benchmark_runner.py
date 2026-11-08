#!/usr/bin/env python3
"""Compatibility wrapper for the canonical benchmark CLI.

Keep this root entrypoint stable for existing scripts/server jobs. The
implementation lives under the ``uav_geolocalization`` package.
"""
from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent
_SRC_ROOT = _REPO_ROOT / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

import uav_geolocalization.cli.benchmark_runner_impl as _impl  # noqa: E402

for _name in dir(_impl):
    if not (_name.startswith("__") and _name.endswith("__")):
        globals()[_name] = getattr(_impl, _name)


def main() -> None:
    """Run the benchmark CLI."""
    _impl.main()


if __name__ == "__main__":
    main()
