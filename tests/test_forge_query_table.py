"""A per-frame footprint table, when configured, decides FORGE eligibility."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from uav_geolocalization.cli import benchmark_runner_impl as impl  # noqa: E402


def test_table_is_inactive_by_default():
    impl._configure_forge_footprints({})
    assert not impl.FORGE_QUERY_HALF_M_TABLE_ACTIVE[0] and impl.FORGE_QUERY_HALF_M_TABLE == {}


def test_table_loads_only_ok_rows(tmp_path=None):
    import tempfile
    d = Path(tempfile.mkdtemp())
    f = d / "t.csv"
    f.write_text("image_name,status,gsd_m_per_px,query_half_bound_m,n_estimates,dispersion\n"
                 "02/a.JPG,OK,0.1,250.0,6,0.02\n02/b.JPG,INDETERMINATE_FEW,,,1,\n")
    impl._configure_forge_footprints({"forge_query_half_m_table": str(f)})
    assert impl.FORGE_QUERY_HALF_M_TABLE_ACTIVE[0]
    assert impl.FORGE_QUERY_HALF_M_TABLE == {"02/a.JPG": 250.0}
    impl._configure_forge_footprints({})
    assert not impl.FORGE_QUERY_HALF_M_TABLE_ACTIVE[0]


def test_infinite_query_half_is_never_a_fault():
    cand = {"reference_name": "02/t.png", "offset_e_m": 5000.0, "offset_n_m": 5000.0,
            "query_half_e_m": float("inf"), "query_half_n_m": float("inf")}
    assert not impl._forge_candidate_excludes_gt(cand, "uav_visloc", 7000.0, 150.0)


def _write(rows):
    import tempfile
    f = Path(tempfile.mkdtemp()) / "t.csv"
    f.write_text("image_name,status,gsd_m_per_px,query_half_bound_m,n_estimates,dispersion\n" + rows)
    return str(f)


def _rejects(rows):
    try:
        impl._configure_forge_footprints({"forge_query_half_m_table": _write(rows)})
    except ValueError:
        return True
    finally:
        impl._configure_forge_footprints({})
    return False


def test_invalid_tables_are_rejected():
    assert _rejects("02/a.JPG,OK,0.1,-1000,6,0.02\n")            # negative bound
    assert _rejects("02/a.JPG,OK,0.1,0,6,0.02\n")                # zero bound
    assert _rejects("02/a.JPG,OK,0.1,inf,6,0.02\n")              # infinite bound
    assert _rejects("02/a.JPG,OK,0.1,250,6,0.02\n02/a.JPG,INDETERMINATE_FEW,,,1,\n")  # duplicate
    assert _rejects("02/a.JPG,INDETERMINATE_FEW,,,1,\n")          # no OK rows
    assert not _rejects("02/a.JPG,OK,0.1,250,6,0.02\n02/b.JPG,INDETERMINATE_FEW,,,1,\n")


def test_normaliser_never_resets_the_tables():
    # predictions.csv export calls the normaliser with no config; that must
    # not switch eligibility back to the frozen constants.
    f = _write("02/a.JPG,OK,0.1,250,6,0.02\n")
    impl._configure_forge_footprints({"forge_query_half_m_table": f,
                                      "forge_ref_half_m_per_map": {"02": [260.0, 300.0]}})
    impl._normalize_adversarial_config({})
    impl._apply_reliability_schema_defaults({})
    assert impl.FORGE_QUERY_HALF_M_TABLE_ACTIVE[0]
    assert impl.FORGE_QUERY_HALF_M_TABLE == {"02/a.JPG": 250.0}
    assert impl.FORGE_REF_HALF_M_PER_MAP == {"02": (260.0, 300.0)}
    impl._configure_forge_footprints({})
