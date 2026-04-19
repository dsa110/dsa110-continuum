"""Tests for Stage C: post-discovery cross-match."""
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from astropy.io import fits


# ---------------------------------------------------------------------------
# Synthetic FITS helper
# ---------------------------------------------------------------------------

def _make_aegean_fits(path: str, n: int = 5, snr_override: list | None = None) -> None:
    """Write a minimal Aegean-style FITS binary table."""
    from astropy.table import Table
    t = Table()
    t["source_name"] = [f"J344.{i:04d}+16.0000" for i in range(n)]
    t["ra_deg"]      = np.array([344.0 + i * 0.05 for i in range(n)])
    t["dec_deg"]     = np.full(n, 16.15)
    rms = 0.001
    peak = [rms * (snr_override[i] if snr_override else 10.0) for i in range(n)]
    t["peak_flux_jy"]     = np.array(peak)
    t["peak_flux_err_jy"] = np.full(n, rms * 0.5)
    t["int_flux_jy"]      = np.array(peak) * 1.1
    t["a_arcsec"]         = np.full(n, 36.9)
    t["b_arcsec"]         = np.full(n, 25.5)
    t["pa_deg"]           = np.full(n, 130.75)
    t["local_rms_jy"]     = np.full(n, rms)
    t.write(path, format="fits", overwrite=True)


def _make_master_cone_df(ra_list, dec_list, flux_mjy_list, id_list=None):
    """Return a DataFrame as if returned by cone_search('master', ...)."""
    n = len(ra_list)
    return pd.DataFrame({
        "ra_deg":    ra_list,
        "dec_deg":   dec_list,
        "flux_mjy":  flux_mjy_list,
        "source_id": id_list or [f"MASTER_{i}" for i in range(n)],
    })


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_run_stage_c_empty_catalog():
    """Empty Aegean FITS → ValueError('No sources')."""
    from astropy.table import Table
    with tempfile.NamedTemporaryFile(suffix=".fits", delete=False) as f:
        empty_path = f.name
    with tempfile.TemporaryDirectory() as out_dir:
        try:
            t = Table(names=["source_name","ra_deg","dec_deg","peak_flux_jy",
                             "peak_flux_err_jy","int_flux_jy","a_arcsec","b_arcsec",
                             "pa_deg","local_rms_jy"],
                      dtype=["U64",float,float,float,float,float,float,float,float,float])
            t.write(empty_path, format="fits", overwrite=True)
            from dsa110_continuum.catalog.stage_c import run_stage_c
            with pytest.raises(ValueError, match="No sources"):
                run_stage_c(empty_path, out_dir)
        finally:
            os.unlink(empty_path)


def test_run_stage_c_all_matched():
    """5 sources, master returns 5 nearby matches → all master_matched=True."""
    with tempfile.NamedTemporaryFile(suffix=".fits", delete=False) as f:
        cat_path = f.name
    with tempfile.TemporaryDirectory() as out_dir:
        try:
            _make_aegean_fits(cat_path, n=5)
            # Build master catalog rows at exactly the same positions (0 arcsec separation)
            master_df = _make_master_cone_df(
                ra_list=[344.0 + i * 0.05 for i in range(5)],
                dec_list=[16.15] * 5,
                flux_mjy_list=[100.0 * (i + 1) for i in range(5)],
            )
            with patch("dsa110_continuum.catalog.stage_c._cone_search", return_value=master_df):
                from dsa110_continuum.catalog.stage_c import run_stage_c
                out_path = run_stage_c(cat_path, out_dir,
                                       ra_center=344.1, dec_center=16.15, radius_deg=1.0)
            assert out_path is not None
            assert Path(str(out_path)).exists()
            with fits.open(str(out_path)) as hdul:
                tbl = hdul[1].data
                assert all(tbl["master_matched"]), "All 5 should be master-matched"
        finally:
            os.unlink(cat_path)


def test_run_stage_c_new_source_candidate():
    """Unmatched source with SNR >= 5 → new_source_candidate=True."""
    with tempfile.NamedTemporaryFile(suffix=".fits", delete=False) as f:
        cat_path = f.name
    with tempfile.TemporaryDirectory() as out_dir:
        try:
            # One source with SNR=10 (peak=0.01, rms=0.001)
            _make_aegean_fits(cat_path, n=1, snr_override=[10.0])
            empty_df = pd.DataFrame(columns=["ra_deg","dec_deg","flux_mjy","source_id"])
            # All catalog queries return empty → no match
            with patch("dsa110_continuum.catalog.stage_c._cone_search", return_value=empty_df):
                from dsa110_continuum.catalog.stage_c import run_stage_c
                out_path = run_stage_c(cat_path, out_dir,
                                       ra_center=344.0, dec_center=16.15)
            with fits.open(str(out_path)) as hdul:
                tbl = hdul[1].data
                assert tbl["new_source_candidate"][0], "High-SNR unmatched source should be a candidate"
        finally:
            os.unlink(cat_path)


def test_run_stage_c_low_snr_not_candidate():
    """Unmatched source with SNR < 5 → new_source_candidate=False."""
    with tempfile.NamedTemporaryFile(suffix=".fits", delete=False) as f:
        cat_path = f.name
    with tempfile.TemporaryDirectory() as out_dir:
        try:
            # SNR=3 (below default threshold of 5)
            _make_aegean_fits(cat_path, n=1, snr_override=[3.0])
            empty_df = pd.DataFrame(columns=["ra_deg","dec_deg","flux_mjy","source_id"])
            with patch("dsa110_continuum.catalog.stage_c._cone_search", return_value=empty_df):
                from dsa110_continuum.catalog.stage_c import run_stage_c
                out_path = run_stage_c(cat_path, out_dir,
                                       ra_center=344.0, dec_center=16.15)
            with fits.open(str(out_path)) as hdul:
                tbl = hdul[1].data
                assert not tbl["new_source_candidate"][0], "Low-SNR unmatched source should NOT be a candidate"
        finally:
            os.unlink(cat_path)


def test_output_fits_columns():
    """Output FITS has all 19 required columns."""
    REQUIRED_COLS = [
        "source_name", "ra_deg", "dec_deg", "peak_flux_jy", "snr",
        "master_matched", "master_sep_arcsec", "master_flux_mjy",
        "master_flux_ratio", "master_source_id",
        "nvss_matched", "nvss_sep_arcsec", "nvss_flux_mjy",
        "first_matched", "first_sep_arcsec",
        "racs_matched", "racs_sep_arcsec",
        "any_matched", "new_source_candidate",
    ]
    with tempfile.NamedTemporaryFile(suffix=".fits", delete=False) as f:
        cat_path = f.name
    with tempfile.TemporaryDirectory() as out_dir:
        try:
            _make_aegean_fits(cat_path, n=3)
            empty_df = pd.DataFrame(columns=["ra_deg","dec_deg","flux_mjy","source_id"])
            with patch("dsa110_continuum.catalog.stage_c._cone_search", return_value=empty_df):
                from dsa110_continuum.catalog.stage_c import run_stage_c
                out_path = run_stage_c(cat_path, out_dir,
                                       ra_center=344.0, dec_center=16.15)
            with fits.open(str(out_path)) as hdul:
                cols = [c.name for c in hdul[1].columns]
                for req in REQUIRED_COLS:
                    assert req in cols, f"Missing required column: {req}"
        finally:
            os.unlink(cat_path)


def test_run_stage_c_no_master_fallback_nvss():
    """Master returns empty, NVSS returns 2 matches → nvss_matched=True for those 2."""
    with tempfile.NamedTemporaryFile(suffix=".fits", delete=False) as f:
        cat_path = f.name
    with tempfile.TemporaryDirectory() as out_dir:
        try:
            _make_aegean_fits(cat_path, n=3)
            empty_df = pd.DataFrame(columns=["ra_deg","dec_deg","flux_mjy","source_id"])
            nvss_df  = _make_master_cone_df(
                ra_list=[344.0, 344.05],
                dec_list=[16.15, 16.15],
                flux_mjy_list=[80.0, 90.0],
            )

            def fake_cone(catalog_type, ra, dec, radius):
                if catalog_type == "nvss":
                    return nvss_df
                return empty_df

            with patch("dsa110_continuum.catalog.stage_c._cone_search", side_effect=fake_cone):
                from dsa110_continuum.catalog.stage_c import run_stage_c
                out_path = run_stage_c(cat_path, out_dir,
                                       ra_center=344.1, dec_center=16.15)
            with fits.open(str(out_path)) as hdul:
                tbl = hdul[1].data
                nvss_hits = int(np.sum(tbl["nvss_matched"]))
                assert nvss_hits == 2, f"Expected 2 NVSS hits, got {nvss_hits}"
                assert not any(tbl["master_matched"]), "Master should be unmatched"
        finally:
            os.unlink(cat_path)


def test_output_path_default_stem():
    """Default output path is {catalog_stem}_crossmatched.fits next to input."""
    with tempfile.NamedTemporaryFile(suffix=".fits", delete=False) as f:
        cat_path = f.name
    try:
        _make_aegean_fits(cat_path, n=2)
        empty_df = pd.DataFrame(columns=["ra_deg","dec_deg","flux_mjy","source_id"])
        crossmatched_path = cat_path.replace(".fits", "_crossmatched.fits")
        with patch("dsa110_continuum.catalog.stage_c._cone_search", return_value=empty_df):
            from dsa110_continuum.catalog.stage_c import run_stage_c
            out_path = run_stage_c(cat_path, ra_center=344.0, dec_center=16.15)
        expected_name = Path(cat_path).stem + "_crossmatched.fits"
        assert Path(str(out_path)).name == expected_name
        assert Path(str(out_path)).exists()
    finally:
        os.unlink(cat_path)
        if os.path.exists(crossmatched_path):
            os.unlink(crossmatched_path)


def test_flux_ratio_computed():
    """Flux ratio = peak_flux_jy / (master_flux_mjy / 1000)."""
    with tempfile.NamedTemporaryFile(suffix=".fits", delete=False) as f:
        cat_path = f.name
    with tempfile.TemporaryDirectory() as out_dir:
        try:
            # SNR=10, rms=0.001, so peak=0.01 Jy; master=10 mJy=0.01 Jy → ratio≈1.0
            _make_aegean_fits(cat_path, n=1, snr_override=[10.0])
            master_df = _make_master_cone_df(
                ra_list=[344.0], dec_list=[16.15], flux_mjy_list=[10.0]
            )
            with patch("dsa110_continuum.catalog.stage_c._cone_search", return_value=master_df):
                from dsa110_continuum.catalog.stage_c import run_stage_c
                out_path = run_stage_c(cat_path, out_dir, ra_center=344.0, dec_center=16.15)
            with fits.open(str(out_path)) as hdul:
                tbl = hdul[1].data
                ratio = float(tbl["master_flux_ratio"][0])
                assert 0.5 < ratio < 2.0, f"Flux ratio should be near 1.0, got {ratio}"
        finally:
            os.unlink(cat_path)


def test_astrometry_qa_logged(caplog):
    """Astrometry QA log message appears when >= 3 sources matched."""
    import logging
    with tempfile.NamedTemporaryFile(suffix=".fits", delete=False) as f:
        cat_path = f.name
    with tempfile.TemporaryDirectory() as out_dir:
        try:
            _make_aegean_fits(cat_path, n=5)
            master_df = _make_master_cone_df(
                ra_list=[344.0 + i * 0.05 for i in range(5)],
                dec_list=[16.15] * 5,
                flux_mjy_list=[100.0] * 5,
            )
            with patch("dsa110_continuum.catalog.stage_c._cone_search", return_value=master_df):
                from dsa110_continuum.catalog.stage_c import run_stage_c
                with caplog.at_level(logging.INFO, logger="dsa110_continuum.catalog.stage_c"):
                    run_stage_c(cat_path, out_dir, ra_center=344.1, dec_center=16.15)
            assert any("Astrometry QA" in r.message for r in caplog.records), \
                "Expected astrometry QA log message"
        finally:
            os.unlink(cat_path)


def test_cli_sim_missing_catalog_exits_zero(monkeypatch, capsys):
    """CLI --sim with missing catalog file exits 0 with a warning (not a crash)."""
    import sys
    scripts_dir = str(Path(__file__).resolve().parent.parent / "scripts")
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    # Point sim catalog to a nonexistent path
    monkeypatch.setenv("DSA110_SIM_STAGE_B_CATALOG", "/nonexistent/stage_b_sources.fits")
    try:
        import stage_c_crossmatch
        with pytest.raises(SystemExit) as exc_info:
            stage_c_crossmatch.main(["--sim"])
        assert exc_info.value.code == 0, "Expected exit code 0 for missing sim catalog"
    except ImportError:
        pytest.skip("stage_c_crossmatch.py not yet created — will pass after Task 3")
