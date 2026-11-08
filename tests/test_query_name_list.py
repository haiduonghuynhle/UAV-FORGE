"""query_name_list_file selects named queries; without it selection is a prefix."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data.dataset_adapters import BaseDataset  # noqa: E402


class _Toy(BaseDataset):
    name = "toy"
    def load(self):
        self._entries = [{"image_name": f"{i:05d}.png"} for i in range(10)]
        self._loaded = True
    def _get_frame(self, idx):
        return self._entries[idx]


def _toy(**kw):
    d = _Toy.__new__(_Toy)
    d._loaded = False; d._test_interval = 1
    d._max_samples = kw.get("max_samples"); d._max_samples_percent = None
    return d


def test_default_is_unchanged():
    d = _toy(max_samples=3)
    assert d.selected_entry_indices() == [0, 1, 2] and d.num_samples == 3


def test_allowlist_filters_before_caps():
    d = _toy(max_samples=2)
    d._query_name_allowlist = {"00004.png", "00007.png", "00009.png"}
    assert d.selected_entry_indices() == [4, 7]
    assert d.num_samples == 2
