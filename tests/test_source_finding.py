# tests/test_source_finding.py
import tempfile
import os
import pytest
from astropy.table import Table

from dsa110_continuum.source_finding.core import (
    SourceCatalogEntry,
    write_catalog,
    write_empty_catalog,
)


def _make_entry(n=0) -> SourceCatalogEntry:
    return SourceCatalogEntry(
        source_name=f"AEG_J344.1234+16.5678_{n}",
        ra_deg=344.1234 + n,
        dec_deg=16.5678,
        peak_flux_jy=0.05 + n * 0.01,
        peak_flux_err_jy=0.002,
        int_flux_jy=0.06 + n * 0.01,
        a_arcsec=36.9,
        b_arcsec=25.5,
        pa_deg=130.75,
        local_rms_jy=0.003,
    )


def test_source_catalog_entry_fields():
    e = _make_entry()
    assert e.source_name.startswith("AEG_")
    assert isinstance(e.ra_deg, float)
    assert isinstance(e.dec_deg, float)
    assert isinstance(e.peak_flux_jy, float)
    assert isinstance(e.peak_flux_err_jy, float)
    assert isinstance(e.int_flux_jy, float)
    assert isinstance(e.a_arcsec, float)
    assert isinstance(e.b_arcsec, float)
    assert isinstance(e.pa_deg, float)
    assert isinstance(e.local_rms_jy, float)


def test_write_catalog_roundtrip():
    entries = [_make_entry(i) for i in range(3)]
    with tempfile.NamedTemporaryFile(suffix=".fits", delete=False) as f:
        path = f.name
    try:
        write_catalog(entries, path)
        t = Table.read(path)
        assert len(t) == 3
        expected_cols = [
            "source_name", "ra_deg", "dec_deg", "peak_flux_jy",
            "peak_flux_err_jy", "int_flux_jy", "a_arcsec", "b_arcsec",
            "pa_deg", "local_rms_jy",
        ]
        for col in expected_cols:
            assert col in t.colnames, f"Missing column: {col}"
        assert abs(t["ra_deg"][1] - 345.1234) < 1e-6
        assert abs(t["peak_flux_jy"][2] - 0.07) < 1e-6
    finally:
        os.unlink(path)


def test_write_empty_catalog_schema():
    with tempfile.NamedTemporaryFile(suffix=".fits", delete=False) as f:
        path = f.name
    try:
        write_empty_catalog(path)
        t = Table.read(path)
        assert len(t) == 0
        expected_cols = [
            "source_name", "ra_deg", "dec_deg", "peak_flux_jy",
            "peak_flux_err_jy", "int_flux_jy", "a_arcsec", "b_arcsec",
            "pa_deg", "local_rms_jy",
        ]
        for col in expected_cols:
            assert col in t.colnames, f"Missing column: {col}"
    finally:
        os.unlink(path)


import sys
import importlib
from unittest.mock import MagicMock


def test_run_bane_skip_existing():
    """If both bkg and rms files already exist, run_bane returns immediately without calling BANE."""
    import os
    # Create a fake mosaic + pre-existing bkg/rms files
    with tempfile.NamedTemporaryFile(suffix=".fits", delete=False) as f:
        mosaic_path = f.name
    stem = mosaic_path.replace(".fits", "")
    bkg_path = stem + "_bkg.fits"
    rms_path = stem + "_rms.fits"
    open(bkg_path, "w").close()
    open(rms_path, "w").close()
    try:
        from dsa110_continuum.source_finding.core import run_bane
        result_bkg, result_rms = run_bane(mosaic_path, skip_existing=True)
        assert result_bkg == bkg_path
        assert result_rms == rms_path
    finally:
        for p in [mosaic_path, bkg_path, rms_path]:
            if os.path.exists(p):
                os.unlink(p)


def test_run_bane_missing_output():
    """RuntimeError raised when BANE mock runs but produces no output files."""
    import os
    with tempfile.NamedTemporaryFile(suffix=".fits", delete=False) as f:
        mosaic_path = f.name
    try:
        # Inject a mock AegeanTools.BANE whose filter_image is a no-op
        mock_bane_mod = MagicMock()
        mock_bane_mod.filter_image = MagicMock()  # does NOT create output files
        mock_at = MagicMock()
        mock_at.BANE = mock_bane_mod
        sys.modules["AegeanTools"] = mock_at
        sys.modules["AegeanTools.BANE"] = mock_bane_mod

        # Reload core so run_bane picks up the injected mock on the next import
        from dsa110_continuum.source_finding import core as sf_core
        importlib.reload(sf_core)

        with pytest.raises(RuntimeError, match="BANE did not produce"):
            sf_core.run_bane(mosaic_path, skip_existing=False)
    finally:
        os.unlink(mosaic_path)
        # Clean up injected mocks
        for k in ["AegeanTools", "AegeanTools.BANE", "AegeanTools.source_finder"]:
            sys.modules.pop(k, None)
        # Reload core back to a clean state
        importlib.reload(sf_core)
