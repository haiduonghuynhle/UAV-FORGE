"""The per-map footprint table is opt-in and leaves the record behaviour alone."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from uav_geolocalization.cli import benchmark_runner_impl as impl  # noqa: E402


def _cand(name, de, dn=0.0, qh=101.0):
    return {"reference_name": name, "offset_e_m": de, "offset_n_m": dn,
            "query_half_e_m": qh, "query_half_n_m": qh}


def test_default_uses_dataset_constant():
    impl._configure_forge_footprints({})
    assert impl.FORGE_REF_HALF_M_PER_MAP == {}
    # 254 + 101 + 15 = 370: 375 m east is a fault under the constant.
    assert impl._forge_candidate_excludes_gt(_cand("05/t.png", 375.0), "uav_visloc", 375.0, 150.0)


def test_per_map_value_overrides_constant():
    impl._configure_forge_footprints({"forge_ref_half_m_per_map": {"05": [277.9, 305.7]}})
    # 277.9 + 101 + 15 = 393.9: the same 375 m tile still overlaps on flight 05.
    assert not impl._forge_candidate_excludes_gt(_cand("05/t.png", 375.0), "uav_visloc", 375.0, 150.0)
    # Another map falls back to the dataset constant.
    assert impl._forge_candidate_excludes_gt(_cand("06/t.png", 375.0), "uav_visloc", 375.0, 150.0)
    impl._configure_forge_footprints({})
    assert impl.FORGE_REF_HALF_M_PER_MAP == {}
