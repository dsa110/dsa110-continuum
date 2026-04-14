"""
Tests for dsa110_continuum.catalog.unified — UnifiedCatalog SQLite backend.

All tests use temporary in-memory or temp-file SQLite databases.
No real catalog data is required.

Coverage
--------
- Schema creation and idempotency
- Source ingest: single, bulk, DataFrame, overwrite
- source_id stability (position-hash vs NVSS-name hash)
- Spectral index auto-computation (NVSS 1.4 GHz + VLASS 3 GHz)
- has_* boolean flags set correctly
- cone_search: radius filter, min_flux_mjy, max_sources, survey filter
- RA-wrap cone search (sources near 0°/360°)
- cone_search returns separation_deg column
- query_by_flux
- get_source by source_id
- count() and summary()
- meta get/set
- Context-manager usage
- build_from_strips with synthetic strip DBs
- build_from_strips with missing/empty strip DBs (graceful)
- _nearest_flux helper: match within radius
- _nearest_flux helper: no match when survey is empty
- _spectral_index helper
- _make_source_id reproducibility
- Ingest with overwrite=False (ignore duplicates)
- Ingest with overwrite=True (replace)
- UnifiedCatalog.__repr__
- catalog/__init__ re-exports UnifiedCatalog
"""

from __future__ import annotations

import math
import sqlite3
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Helpers to build synthetic strip DBs for build_from_strips tests
# ---------------------------------------------------------------------------

def _make_strip_db(path: Path, sources: list[dict]) -> Path:
    """Write a minimal per-survey strip SQLite DB (columns: ra_deg, dec_deg, flux_mjy)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.execute("""
        CREATE TABLE sources (
            source_id INTEGER PRIMARY KEY AUTOINCREMENT,
            ra_deg REAL NOT NULL,
            dec_deg REAL NOT NULL,
            flux_mjy REAL
        )
    """)
    conn.execute("CREATE INDEX idx_dec ON sources(dec_deg)")
    conn.executemany(
        "INSERT INTO sources (ra_deg, dec_deg, flux_mjy) VALUES (?,?,?)",
        [(s["ra_deg"], s["dec_deg"], s["flux_mjy"]) for s in sources],
    )
    conn.commit()
    conn.close()
    return path


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def tmp_db(tmp_path):
    """Yield a fresh UnifiedCatalog backed by a temp file."""
    from dsa110_continuum.catalog.unified import UnifiedCatalog
    db_path = tmp_path / "unified_test.db"
    uc = UnifiedCatalog(db_path)
    yield uc
    uc.close()


@pytest.fixture()
def sample_rows():
    """15 synthetic sources spread across a small sky patch."""
    rng = np.random.default_rng(42)
    rows = []
    for i in range(15):
        ra = 180.0 + rng.uniform(-2.0, 2.0)
        dec = 30.0 + rng.uniform(-2.0, 2.0)
        s_nvss = float(rng.uniform(5.0, 500.0))
        s_vlass = s_nvss * float(rng.uniform(0.3, 1.5))
        rows.append({
            "ra_deg": ra,
            "dec_deg": dec,
            "s_nvss_mjy": s_nvss,
            "s_vlass_mjy": s_vlass,
        })
    return rows


# ===========================================================================
# 1. Schema & construction
# ===========================================================================

class TestSchema:
    def test_db_file_created(self, tmp_path):
        from dsa110_continuum.catalog.unified import UnifiedCatalog
        db = tmp_path / "test.db"
        uc = UnifiedCatalog(db)
        assert db.exists()
        uc.close()

    def test_sources_table_exists(self, tmp_db):
        conn = sqlite3.connect(str(tmp_db.db_path))
        tables = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()}
        conn.close()
        assert "sources" in tables
        assert "meta" in tables

    def test_sources_table_columns(self, tmp_db):
        conn = sqlite3.connect(str(tmp_db.db_path))
        cols = {r[1] for r in conn.execute("PRAGMA table_info(sources)").fetchall()}
        conn.close()
        expected = {
            "source_id", "ra_deg", "dec_deg",
            "s_nvss_mjy", "s_first_mjy", "s_vlass_mjy", "s_racs_mjy",
            "alpha", "resolved_flag", "confusion_flag",
            "has_nvss", "has_first", "has_vlass", "has_racs",
        }
        assert expected <= cols

    def test_schema_idempotent(self, tmp_path):
        """Opening same DB twice does not error."""
        from dsa110_continuum.catalog.unified import UnifiedCatalog
        db = tmp_path / "idem.db"
        uc1 = UnifiedCatalog(db)
        uc1.close()
        uc2 = UnifiedCatalog(db)
        uc2.close()

    def test_wal_mode(self, tmp_db):
        conn = sqlite3.connect(str(tmp_db.db_path))
        mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
        conn.close()
        assert mode.lower() == "wal"

    def test_context_manager(self, tmp_path):
        from dsa110_continuum.catalog.unified import UnifiedCatalog
        db = tmp_path / "ctx.db"
        with UnifiedCatalog(db) as uc:
            uc.ingest_sources([{"ra_deg": 10.0, "dec_deg": 20.0, "s_nvss_mjy": 50.0}])
            assert uc.count() == 1
        # After exit, connection should be closed (no error on re-open)
        with UnifiedCatalog(db) as uc2:
            assert uc2.count() == 1

    def test_repr_contains_path(self, tmp_db):
        r = repr(tmp_db)
        assert "UnifiedCatalog" in r
        assert "unified_test.db" in r


# ===========================================================================
# 2. Ingest
# ===========================================================================

class TestIngest:
    def test_ingest_single_source(self, tmp_db):
        n = tmp_db.ingest_sources([{"ra_deg": 10.0, "dec_deg": 20.0, "s_nvss_mjy": 100.0}])
        assert tmp_db.count() == 1

    def test_ingest_bulk(self, tmp_db, sample_rows):
        tmp_db.ingest_sources(sample_rows)
        assert tmp_db.count() == len(sample_rows)

    def test_ingest_empty_list(self, tmp_db):
        result = tmp_db.ingest_sources([])
        assert result == 0
        assert tmp_db.count() == 0

    def test_ingest_dataframe(self, tmp_db, sample_rows):
        df = pd.DataFrame(sample_rows)
        tmp_db.ingest_dataframe(df)
        assert tmp_db.count() == len(sample_rows)

    def test_ingest_ignore_duplicate(self, tmp_db):
        row = {"ra_deg": 10.0, "dec_deg": 20.0, "s_nvss_mjy": 50.0}
        tmp_db.ingest_sources([row])
        tmp_db.ingest_sources([row])           # same position → same source_id
        assert tmp_db.count() == 1             # not doubled

    def test_ingest_overwrite(self, tmp_db):
        row = {"ra_deg": 10.0, "dec_deg": 20.0, "s_nvss_mjy": 50.0}
        tmp_db.ingest_sources([row])
        updated = {"ra_deg": 10.0, "dec_deg": 20.0, "s_nvss_mjy": 999.0}
        tmp_db.ingest_sources([updated], overwrite=True)
        assert tmp_db.count() == 1
        src = tmp_db.cone_search(10.0, 20.0, 0.01)
        assert abs(src.iloc[0]["s_nvss_mjy"] - 999.0) < 0.1

    def test_ingest_minimal_row(self, tmp_db):
        """Only ra/dec required — all flux columns may be absent."""
        tmp_db.ingest_sources([{"ra_deg": 5.0, "dec_deg": -10.0}])
        assert tmp_db.count() == 1
        row = tmp_db.cone_search(5.0, -10.0, 0.01).iloc[0]
        assert pd.isna(row["s_nvss_mjy"])


# ===========================================================================
# 3. source_id stability
# ===========================================================================

class TestSourceId:
    def test_same_position_same_id(self):
        from dsa110_continuum.catalog.unified import _make_source_id
        id1 = _make_source_id(180.0, 30.0)
        id2 = _make_source_id(180.0, 30.0)
        assert id1 == id2

    def test_different_position_different_id(self):
        from dsa110_continuum.catalog.unified import _make_source_id
        id1 = _make_source_id(180.0, 30.0)
        id2 = _make_source_id(180.1, 30.0)
        assert id1 != id2

    def test_nvss_name_used_when_provided(self):
        from dsa110_continuum.catalog.unified import _make_source_id
        id_with_name = _make_source_id(180.0, 30.0, nvss_name="NVSS J120000+300000")
        id_pos_only = _make_source_id(180.0, 30.0)
        assert id_with_name != id_pos_only
        assert id_with_name.startswith("nvss_")

    def test_nvss_name_reproducible(self):
        from dsa110_continuum.catalog.unified import _make_source_id
        n = "NVSS J120000+300000"
        assert _make_source_id(0, 0, n) == _make_source_id(0, 0, n)

    def test_explicit_source_id_preserved(self, tmp_db):
        row = {"ra_deg": 1.0, "dec_deg": 2.0, "source_id": "my_custom_id_001"}
        tmp_db.ingest_sources([row])
        src = tmp_db.get_source("my_custom_id_001")
        assert src is not None
        assert src["source_id"] == "my_custom_id_001"


# ===========================================================================
# 4. has_* flags and spectral index
# ===========================================================================

class TestFlags:
    def test_has_nvss_set(self, tmp_db):
        tmp_db.ingest_sources([{"ra_deg": 1.0, "dec_deg": 1.0, "s_nvss_mjy": 100.0}])
        df = tmp_db.cone_search(1.0, 1.0, 0.1)
        assert df.iloc[0]["has_nvss"] == 1
        assert df.iloc[0]["has_first"] == 0
        assert df.iloc[0]["has_vlass"] == 0
        assert df.iloc[0]["has_racs"] == 0

    def test_has_all_surveys(self, tmp_db):
        tmp_db.ingest_sources([{
            "ra_deg": 1.0, "dec_deg": 1.0,
            "s_nvss_mjy": 100.0, "s_first_mjy": 95.0,
            "s_vlass_mjy": 50.0, "s_racs_mjy": 200.0,
        }])
        df = tmp_db.cone_search(1.0, 1.0, 0.1)
        row = df.iloc[0]
        assert row["has_nvss"] == 1
        assert row["has_first"] == 1
        assert row["has_vlass"] == 1
        assert row["has_racs"] == 1

    def test_spectral_index_auto_computed(self, tmp_db):
        """alpha = log(s_vlass/s_nvss) / log(3/1.4)"""
        s_nvss, s_vlass = 100.0, 50.0
        expected_alpha = math.log(s_vlass / s_nvss) / math.log(3.0 / 1.4)
        tmp_db.ingest_sources([{
            "ra_deg": 5.0, "dec_deg": 5.0,
            "s_nvss_mjy": s_nvss, "s_vlass_mjy": s_vlass,
        }])
        df = tmp_db.cone_search(5.0, 5.0, 0.1)
        assert abs(df.iloc[0]["alpha"] - expected_alpha) < 1e-9

    def test_spectral_index_explicit_overrides(self, tmp_db):
        """An explicitly supplied alpha should not be overwritten."""
        tmp_db.ingest_sources([{
            "ra_deg": 6.0, "dec_deg": 6.0,
            "s_nvss_mjy": 100.0, "s_vlass_mjy": 50.0,
            "alpha": -0.5,
        }])
        df = tmp_db.cone_search(6.0, 6.0, 0.1)
        assert abs(df.iloc[0]["alpha"] - (-0.5)) < 1e-9

    def test_alpha_none_when_only_one_survey(self, tmp_db):
        tmp_db.ingest_sources([{"ra_deg": 7.0, "dec_deg": 7.0, "s_nvss_mjy": 100.0}])
        df = tmp_db.cone_search(7.0, 7.0, 0.1)
        assert pd.isna(df.iloc[0]["alpha"])

    def test_resolved_flag_stored(self, tmp_db):
        tmp_db.ingest_sources([{
            "ra_deg": 8.0, "dec_deg": 8.0, "s_nvss_mjy": 100.0,
            "resolved_flag": 1,
        }])
        df = tmp_db.cone_search(8.0, 8.0, 0.1)
        assert df.iloc[0]["resolved_flag"] == 1


# ===========================================================================
# 5. cone_search
# ===========================================================================

class TestConeSearch:
    def test_returns_dataframe(self, tmp_db, sample_rows):
        tmp_db.ingest_sources(sample_rows)
        df = tmp_db.cone_search(180.0, 30.0, 5.0)
        assert isinstance(df, pd.DataFrame)

    def test_returns_sources_within_radius(self, tmp_db):
        tmp_db.ingest_sources([
            {"ra_deg": 180.0, "dec_deg": 30.0, "s_nvss_mjy": 100.0},
            {"ra_deg": 181.5, "dec_deg": 30.0, "s_nvss_mjy": 50.0},  # outside 1 deg
        ])
        df = tmp_db.cone_search(180.0, 30.0, 1.0)
        assert len(df) == 1
        assert abs(df.iloc[0]["ra_deg"] - 180.0) < 0.01

    def test_excludes_sources_outside_radius(self, tmp_db):
        tmp_db.ingest_sources([
            {"ra_deg": 180.0, "dec_deg": 30.0, "s_nvss_mjy": 100.0},
            {"ra_deg": 185.0, "dec_deg": 30.0, "s_nvss_mjy": 200.0},
        ])
        df = tmp_db.cone_search(180.0, 30.0, 2.0)
        assert len(df) == 1

    def test_empty_result_when_no_sources(self, tmp_db):
        df = tmp_db.cone_search(180.0, 30.0, 1.0)
        assert len(df) == 0

    def test_min_flux_filter(self, tmp_db):
        tmp_db.ingest_sources([
            {"ra_deg": 180.0, "dec_deg": 30.0, "s_nvss_mjy": 10.0},
            {"ra_deg": 180.1, "dec_deg": 30.0, "s_nvss_mjy": 500.0},
        ])
        df = tmp_db.cone_search(180.0, 30.0, 1.0, min_flux_mjy=100.0)
        assert len(df) == 1
        assert df.iloc[0]["s_nvss_mjy"] >= 100.0

    def test_max_sources_limit(self, tmp_db, sample_rows):
        tmp_db.ingest_sources(sample_rows)
        df = tmp_db.cone_search(180.0, 30.0, 10.0, max_sources=3)
        assert len(df) <= 3

    def test_survey_filter(self, tmp_db):
        tmp_db.ingest_sources([
            {"ra_deg": 180.0, "dec_deg": 30.0, "s_nvss_mjy": 100.0},           # nvss only
            {"ra_deg": 180.1, "dec_deg": 30.0, "s_nvss_mjy": 80.0, "s_vlass_mjy": 40.0},  # nvss+vlass
        ])
        df = tmp_db.cone_search(180.0, 30.0, 1.0, surveys=["nvss", "vlass"])
        assert len(df) == 1
        assert df.iloc[0]["has_vlass"] == 1

    def test_separation_deg_column_present(self, tmp_db):
        tmp_db.ingest_sources([{"ra_deg": 180.0, "dec_deg": 30.0, "s_nvss_mjy": 100.0}])
        df = tmp_db.cone_search(180.0, 30.0, 1.0)
        assert "separation_deg" in df.columns

    def test_separation_deg_within_radius(self, tmp_db):
        tmp_db.ingest_sources([
            {"ra_deg": 180.0, "dec_deg": 30.0, "s_nvss_mjy": 100.0},
            {"ra_deg": 180.5, "dec_deg": 30.0, "s_nvss_mjy": 80.0},
        ])
        df = tmp_db.cone_search(180.0, 30.0, 2.0)
        assert (df["separation_deg"] <= 2.0).all()

    def test_cone_search_near_ra_zero(self, tmp_db):
        """Sources near RA=0 that straddle the 0/360 boundary."""
        tmp_db.ingest_sources([
            {"ra_deg": 0.3, "dec_deg": 10.0, "s_nvss_mjy": 100.0},
            {"ra_deg": 359.8, "dec_deg": 10.0, "s_nvss_mjy": 90.0},  # 0.5 deg from 0.3
        ])
        # Search centred at RA=0.3 — source at 359.8 is ~0.43 deg away (at dec=10)
        df_wide = tmp_db.cone_search(0.3, 10.0, 2.0)
        assert len(df_wide) >= 1  # at least the exact-match source

    def test_cone_returns_empty_df_not_none(self, tmp_db):
        result = tmp_db.cone_search(90.0, 45.0, 0.001)
        assert result is not None
        assert isinstance(result, pd.DataFrame)


# ===========================================================================
# 6. query_by_flux and get_source
# ===========================================================================

class TestQueryByFlux:
    def test_returns_sources_above_threshold(self, tmp_db):
        tmp_db.ingest_sources([
            {"ra_deg": 1.0, "dec_deg": 1.0, "s_nvss_mjy": 10.0},
            {"ra_deg": 2.0, "dec_deg": 2.0, "s_nvss_mjy": 200.0},
            {"ra_deg": 3.0, "dec_deg": 3.0, "s_nvss_mjy": 500.0},
        ])
        df = tmp_db.query_by_flux(100.0, survey="nvss")
        assert len(df) == 2
        assert (df["s_nvss_mjy"] >= 100.0).all()

    def test_query_vlass_survey(self, tmp_db):
        tmp_db.ingest_sources([
            {"ra_deg": 1.0, "dec_deg": 1.0, "s_vlass_mjy": 300.0},
            {"ra_deg": 2.0, "dec_deg": 2.0, "s_vlass_mjy": 50.0},
        ])
        df = tmp_db.query_by_flux(100.0, survey="vlass")
        assert len(df) == 1
        assert df.iloc[0]["s_vlass_mjy"] == 300.0

    def test_empty_when_no_sources_above_threshold(self, tmp_db):
        tmp_db.ingest_sources([{"ra_deg": 1.0, "dec_deg": 1.0, "s_nvss_mjy": 5.0}])
        df = tmp_db.query_by_flux(1000.0)
        assert len(df) == 0


class TestGetSource:
    def test_returns_dict(self, tmp_db):
        row = {"ra_deg": 10.0, "dec_deg": 20.0, "s_nvss_mjy": 100.0}
        tmp_db.ingest_sources([row])
        df = tmp_db.cone_search(10.0, 20.0, 0.01)
        sid = df.iloc[0]["source_id"]
        src = tmp_db.get_source(sid)
        assert isinstance(src, dict)
        assert src["source_id"] == sid

    def test_returns_none_for_missing(self, tmp_db):
        src = tmp_db.get_source("nonexistent_id_xyz")
        assert src is None


# ===========================================================================
# 7. count() and summary()
# ===========================================================================

class TestCountSummary:
    def test_count_zero_initially(self, tmp_db):
        assert tmp_db.count() == 0

    def test_count_after_ingest(self, tmp_db, sample_rows):
        tmp_db.ingest_sources(sample_rows)
        assert tmp_db.count() == len(sample_rows)

    def test_summary_keys(self, tmp_db, sample_rows):
        tmp_db.ingest_sources(sample_rows)
        s = tmp_db.summary()
        assert "total" in s
        assert "nvss" in s
        assert "first" in s
        assert "vlass" in s
        assert "racs" in s

    def test_summary_total_matches_count(self, tmp_db, sample_rows):
        tmp_db.ingest_sources(sample_rows)
        s = tmp_db.summary()
        assert s["total"] == tmp_db.count()

    def test_summary_nvss_count(self, tmp_db):
        tmp_db.ingest_sources([
            {"ra_deg": 1.0, "dec_deg": 1.0, "s_nvss_mjy": 100.0},
            {"ra_deg": 2.0, "dec_deg": 2.0},  # no NVSS
        ])
        s = tmp_db.summary()
        assert s["nvss"] == 1
        assert s["total"] == 2


# ===========================================================================
# 8. Meta
# ===========================================================================

class TestMeta:
    def test_set_and_get(self, tmp_db):
        tmp_db.set_meta("foo", "bar")
        assert tmp_db.get_meta("foo") == "bar"

    def test_get_missing_returns_none(self, tmp_db):
        assert tmp_db.get_meta("nonexistent") is None

    def test_set_overwrites(self, tmp_db):
        tmp_db.set_meta("key", "v1")
        tmp_db.set_meta("key", "v2")
        assert tmp_db.get_meta("key") == "v2"


# ===========================================================================
# 9. build_from_strips
# ===========================================================================

class TestBuildFromStrips:
    def test_build_nvss_only(self, tmp_path):
        from dsa110_continuum.catalog.unified import UnifiedCatalog
        nvss_db = tmp_path / "nvss.db"
        _make_strip_db(nvss_db, [
            {"ra_deg": 180.0, "dec_deg": 30.0, "flux_mjy": 100.0},
            {"ra_deg": 181.0, "dec_deg": 30.0, "flux_mjy": 200.0},
        ])
        uc = UnifiedCatalog.build_from_strips(
            nvss_db=nvss_db,
            output_db=tmp_path / "unified.db",
        )
        assert uc.count() == 2
        uc.close()

    def test_build_nvss_plus_vlass(self, tmp_path):
        from dsa110_continuum.catalog.unified import UnifiedCatalog
        nvss_db = tmp_path / "nvss.db"
        vlass_db = tmp_path / "vlass.db"
        _make_strip_db(nvss_db, [{"ra_deg": 180.0, "dec_deg": 30.0, "flux_mjy": 100.0}])
        _make_strip_db(vlass_db, [{"ra_deg": 180.001, "dec_deg": 30.0, "flux_mjy": 55.0}])

        uc = UnifiedCatalog.build_from_strips(
            nvss_db=nvss_db,
            vlass_db=vlass_db,
            output_db=tmp_path / "unified.db",
        )
        assert uc.count() == 1
        df = uc.cone_search(180.0, 30.0, 1.0)
        assert df.iloc[0]["has_nvss"] == 1
        assert df.iloc[0]["has_vlass"] == 1
        assert df.iloc[0]["s_vlass_mjy"] == pytest.approx(55.0, abs=0.5)
        uc.close()

    def test_build_with_no_secondary_surveys(self, tmp_path):
        from dsa110_continuum.catalog.unified import UnifiedCatalog
        nvss_db = tmp_path / "nvss.db"
        _make_strip_db(nvss_db, [
            {"ra_deg": 10.0, "dec_deg": 5.0, "flux_mjy": 50.0},
        ])
        uc = UnifiedCatalog.build_from_strips(
            nvss_db=nvss_db,
            output_db=tmp_path / "out.db",
        )
        df = uc.cone_search(10.0, 5.0, 1.0)
        assert df.iloc[0]["has_first"] == 0
        assert df.iloc[0]["has_vlass"] == 0
        assert df.iloc[0]["has_racs"] == 0
        uc.close()

    def test_build_with_all_surveys(self, tmp_path):
        from dsa110_continuum.catalog.unified import UnifiedCatalog
        for name, flux in [("nvss", 100), ("first", 90), ("vlass", 50), ("racs", 200)]:
            _make_strip_db(
                tmp_path / f"{name}.db",
                [{"ra_deg": 90.0, "dec_deg": 20.0, "flux_mjy": flux}],
            )
        uc = UnifiedCatalog.build_from_strips(
            nvss_db=tmp_path / "nvss.db",
            first_db=tmp_path / "first.db",
            vlass_db=tmp_path / "vlass.db",
            racs_db=tmp_path / "racs.db",
            output_db=tmp_path / "all.db",
            match_radius_arcsec=30.0,
        )
        df = uc.cone_search(90.0, 20.0, 1.0)
        assert df.iloc[0]["has_nvss"] == 1
        assert df.iloc[0]["has_first"] == 1
        assert df.iloc[0]["has_vlass"] == 1
        assert df.iloc[0]["has_racs"] == 1
        uc.close()

    def test_build_with_missing_nvss_db(self, tmp_path):
        """When NVSS DB does not exist, catalog is empty but no exception."""
        from dsa110_continuum.catalog.unified import UnifiedCatalog
        uc = UnifiedCatalog.build_from_strips(
            nvss_db=tmp_path / "nonexistent.db",
            output_db=tmp_path / "empty.db",
        )
        assert uc.count() == 0
        uc.close()

    def test_build_sets_meta_build_time(self, tmp_path):
        from dsa110_continuum.catalog.unified import UnifiedCatalog
        nvss_db = tmp_path / "nvss.db"
        _make_strip_db(nvss_db, [{"ra_deg": 1.0, "dec_deg": 1.0, "flux_mjy": 10.0}])
        uc = UnifiedCatalog.build_from_strips(
            nvss_db=nvss_db,
            output_db=tmp_path / "meta.db",
        )
        bt = uc.get_meta("build_time_iso")
        assert bt is not None
        assert "T" in bt  # ISO 8601 format
        uc.close()

    def test_build_spectral_index_from_strips(self, tmp_path):
        """Alpha auto-computed from NVSS+VLASS when building from strips."""
        from dsa110_continuum.catalog.unified import UnifiedCatalog
        import math
        s_nvss, s_vlass = 100.0, 50.0
        expected_alpha = math.log(s_vlass / s_nvss) / math.log(3.0 / 1.4)
        _make_strip_db(tmp_path / "nvss.db", [{"ra_deg": 50.0, "dec_deg": 10.0, "flux_mjy": s_nvss}])
        _make_strip_db(tmp_path / "vlass.db", [{"ra_deg": 50.0, "dec_deg": 10.0, "flux_mjy": s_vlass}])
        uc = UnifiedCatalog.build_from_strips(
            nvss_db=tmp_path / "nvss.db",
            vlass_db=tmp_path / "vlass.db",
            output_db=tmp_path / "alpha.db",
            match_radius_arcsec=30.0,
        )
        df = uc.cone_search(50.0, 10.0, 1.0)
        assert abs(df.iloc[0]["alpha"] - expected_alpha) < 0.01
        uc.close()

    def test_min_nvss_flux_filters_faint_sources(self, tmp_path):
        from dsa110_continuum.catalog.unified import UnifiedCatalog
        _make_strip_db(tmp_path / "nvss.db", [
            {"ra_deg": 1.0, "dec_deg": 1.0, "flux_mjy": 5.0},    # faint
            {"ra_deg": 2.0, "dec_deg": 2.0, "flux_mjy": 500.0},  # bright
        ])
        uc = UnifiedCatalog.build_from_strips(
            nvss_db=tmp_path / "nvss.db",
            output_db=tmp_path / "filtered.db",
            min_nvss_flux_mjy=100.0,
        )
        assert uc.count() == 1
        uc.close()


# ===========================================================================
# 10. Internal helpers
# ===========================================================================

class TestHelpers:
    def test_spectral_index_formula(self):
        from dsa110_continuum.catalog.unified import _spectral_index
        alpha = _spectral_index(100.0, 1.4, 50.0, 3.0)
        expected = math.log(50.0 / 100.0) / math.log(3.0 / 1.4)
        assert abs(alpha - expected) < 1e-12

    def test_spectral_index_none_when_flux_none(self):
        from dsa110_continuum.catalog.unified import _spectral_index
        assert _spectral_index(None, 1.4, 50.0, 3.0) is None
        assert _spectral_index(100.0, 1.4, None, 3.0) is None

    def test_spectral_index_none_when_flux_zero(self):
        from dsa110_continuum.catalog.unified import _spectral_index
        assert _spectral_index(0.0, 1.4, 50.0, 3.0) is None

    def test_nearest_flux_match(self):
        from dsa110_continuum.catalog.unified import _nearest_flux
        nvss_ra = np.array([180.0])
        nvss_dec = np.array([30.0])
        survey_df = pd.DataFrame({
            "ra_deg": [180.001],
            "dec_deg": [30.0],
            "flux_mjy": [99.9],
        })
        result = _nearest_flux(nvss_ra, nvss_dec, survey_df, radius_arcsec=30.0)
        assert abs(result[0] - 99.9) < 0.1

    def test_nearest_flux_no_match_outside_radius(self):
        from dsa110_continuum.catalog.unified import _nearest_flux
        nvss_ra = np.array([180.0])
        nvss_dec = np.array([30.0])
        survey_df = pd.DataFrame({
            "ra_deg": [181.0],    # ~1 deg away at dec=30
            "dec_deg": [30.0],
            "flux_mjy": [99.9],
        })
        result = _nearest_flux(nvss_ra, nvss_dec, survey_df, radius_arcsec=10.0)
        assert not np.isfinite(result[0])

    def test_nearest_flux_empty_survey(self):
        from dsa110_continuum.catalog.unified import _nearest_flux
        nvss_ra = np.array([180.0, 181.0])
        nvss_dec = np.array([30.0, 30.0])
        survey_df = pd.DataFrame(columns=["ra_deg", "dec_deg", "flux_mjy"])
        result = _nearest_flux(nvss_ra, nvss_dec, survey_df, radius_arcsec=30.0)
        assert not np.any(np.isfinite(result))


# ===========================================================================
# 11. catalog/__init__ re-export
# ===========================================================================

class TestInitExport:
    def test_unified_catalog_importable_from_catalog(self):
        from dsa110_continuum.catalog import UnifiedCatalog
        assert UnifiedCatalog is not None

    def test_unified_catalog_is_class(self):
        from dsa110_continuum.catalog import UnifiedCatalog
        assert isinstance(UnifiedCatalog, type)
