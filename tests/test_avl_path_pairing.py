"""Regression test for the AVL image/metadata pairing defect.

The adapter's filename fallback used to search every sub-region folder, so a
metadata entry for Qingzhou_3_2/DJI_0532.JPG resolved to the same-named
QZ_SongCity image when its own file was missing. That paired one flight's
ground truth with another flight's photo (~800 m apart) on 102 of 589 AVL
frames, making them unwinnable. The fallback must stay inside the entry's own
flight folder.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "src"):
    sys.path.insert(0, str(p))


import pytest


@pytest.fixture
def tmp(tmp_path: Path) -> Path:
    return _make_tree(tmp_path)


def _make_tree(tmp: Path):
    """QZ_SongCity has DJI_0532.JPG; Qingzhou_3_2 does not."""
    song = tmp / "UAV_image" / "QZ_Town" / "QZ_SongCity"
    qing = tmp / "UAV_image" / "QZ_Town" / "Qingzhou_3_2"
    song.mkdir(parents=True)
    qing.mkdir(parents=True)
    (song / "DJI_0532.JPG").write_bytes(b"song")
    (qing / "DJI_9999.JPG").write_bytes(b"qing")
    return tmp


def _adapter(tmp: Path):
    from data.dataset_adapters import AVLDataset

    ds = AVLDataset.__new__(AVLDataset)          # bypass __init__ (needs real data)
    ds._data_root = tmp
    ds._region = "QZ_Town"
    return ds


def test_missing_image_does_not_borrow_from_other_flight(tmp: Path):
    ds = _adapter(tmp)
    got = ds._resolve_image_path("./Data/UAV_image/QZ_Town/Qingzhou_3_2/DJI_0532.JPG")
    borrowed = tmp / "UAV_image" / "QZ_Town" / "QZ_SongCity" / "DJI_0532.JPG"
    assert got != borrowed, (
        "Qingzhou_3_2 metadata resolved to a QZ_SongCity image: the cross-flight "
        "pairing defect is back"
    )
    assert not got.exists(), "a genuinely missing image must stay missing (skipped by load)"


def test_direct_path_still_resolves(tmp: Path):
    ds = _adapter(tmp)
    got = ds._resolve_image_path("./Data/UAV_image/QZ_Town/QZ_SongCity/DJI_0532.JPG")
    assert got.exists() and got.name == "DJI_0532.JPG"
    assert got.parent.name == "QZ_SongCity"


def test_fallback_within_own_flight_still_works(tmp: Path):
    """A path that is wrong only above the flight folder must still resolve."""
    ds = _adapter(tmp)
    got = ds._resolve_image_path("./Data/Other/QZ_Town/Qingzhou_3_2/DJI_9999.JPG")
    assert got.exists(), "fallback inside the entry's own flight folder must still work"
    assert got.parent.name == "Qingzhou_3_2"


if __name__ == "__main__":
    import tempfile

    failures = 0
    for fn in (
        test_missing_image_does_not_borrow_from_other_flight,
        test_direct_path_still_resolves,
        test_fallback_within_own_flight_still_works,
    ):
        with tempfile.TemporaryDirectory() as d:
            try:
                fn(_make_tree(Path(d)))
                print(f"PASS {fn.__name__}")
            except AssertionError as e:
                failures += 1
                print(f"FAIL {fn.__name__}: {e}")
            except Exception as e:
                failures += 1
                print(f"ERROR {fn.__name__}: {type(e).__name__}: {e}")
    sys.exit(1 if failures else 0)
