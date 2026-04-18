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
from unittest.mock import MagicMock, patch


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
        # Ensure AegeanTools is not in sys.modules so we can detect if it gets imported
        for k in list(sys.modules.keys()):
            if "AegeanTools" in k:
                del sys.modules[k]
        result_bkg, result_rms = run_bane(mosaic_path, skip_existing=True)
        assert result_bkg == bkg_path
        assert result_rms == rms_path
        # Verify AegeanTools was NOT imported (skip path should not touch it)
        assert not any("AegeanTools" in k for k in sys.modules), \
            "skip_existing path should not import AegeanTools"
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
        # Build mock hierarchy: AegeanTools.BANE.filter_image is a no-op
        mock_bane_mod = MagicMock()
        mock_bane_mod.filter_image = MagicMock()  # does NOT create output files
        mock_at = MagicMock()
        mock_at.BANE = mock_bane_mod

        with patch.dict(sys.modules, {
            "AegeanTools": mock_at,
            "AegeanTools.BANE": mock_bane_mod,
        }):
            # Re-import core inside the patch context so the deferred import resolves to the mock
            import importlib
            from dsa110_continuum.source_finding import core as sf_core
            importlib.reload(sf_core)

            with pytest.raises(RuntimeError, match="BANE did not produce"):
                sf_core.run_bane(mosaic_path, skip_existing=False)
    finally:
        os.unlink(mosaic_path)


# ── check_catalog tests ─────────────────────────────────────────────────────

def test_check_catalog_empty():
    """check_catalog returns False for a zero-row catalog."""
    import os
    with tempfile.NamedTemporaryFile(suffix=".fits", delete=False) as f:
        path = f.name
    try:
        write_empty_catalog(path)
        from dsa110_continuum.source_finding.core import check_catalog
        assert check_catalog(path) is False
    finally:
        os.unlink(path)


def test_check_catalog_non_empty_no_bright():
    """Non-empty catalog with no bright (>1 Jy) sources returns True (warning only)."""
    import os
    entry = _make_entry(0)  # peak_flux_jy = 0.05 Jy — well below 1 Jy
    with tempfile.NamedTemporaryFile(suffix=".fits", delete=False) as f:
        path = f.name
    try:
        write_catalog([entry], path)
        from dsa110_continuum.source_finding.core import check_catalog
        assert check_catalog(path) is True
    finally:
        os.unlink(path)


def test_check_catalog_with_bright_source():
    """Catalog with a >1 Jy source returns True."""
    import os
    bright = SourceCatalogEntry(
        source_name="AEG_J344.0000+16.0000",
        ra_deg=344.0,
        dec_deg=16.0,
        peak_flux_jy=2.5,
        peak_flux_err_jy=0.05,
        int_flux_jy=3.0,
        a_arcsec=36.9,
        b_arcsec=25.5,
        pa_deg=130.0,
        local_rms_jy=0.003,
    )
    with tempfile.NamedTemporaryFile(suffix=".fits", delete=False) as f:
        path = f.name
    try:
        write_catalog([bright], path)
        from dsa110_continuum.source_finding.core import check_catalog
        assert check_catalog(path) is True
    finally:
        os.unlink(path)


# ── run_aegean tests ────────────────────────────────────────────────────────

def _make_aegean_source(ra=344.0, dec=16.0, peak=0.05, rms=0.003):
    src = MagicMock()
    src.ra = ra
    src.dec = dec
    src.peak_flux = peak
    src.err_peak_flux = 0.002
    src.int_flux = peak * 1.1
    src.a = 36.9
    src.b = 25.5
    src.pa = 130.75
    src.local_rms = rms
    return src


def test_run_aegean_import_error():
    """ImportError with install hint when AegeanTools is absent."""
    # Ensure AegeanTools is not available
    for k in list(sys.modules.keys()):
        if "AegeanTools" in k:
            del sys.modules[k]
    import importlib
    from dsa110_continuum.source_finding import core as sf_core
    importlib.reload(sf_core)
    with pytest.raises(ImportError, match="AegeanTools not installed"):
        sf_core.run_aegean("fake.fits", "fake_bkg.fits", "fake_rms.fits")
    # Restore module state
    importlib.reload(sf_core)


def test_run_aegean_returns_empty_list():
    """Aegean finds nothing → returns empty list."""
    import importlib
    mock_sf_instance = MagicMock()
    mock_sf_instance.find_sources_in_image.return_value = []
    mock_sf_cls = MagicMock(return_value=mock_sf_instance)
    mock_sf_mod = MagicMock()
    mock_sf_mod.SourceFinder = mock_sf_cls
    mock_at = MagicMock()
    mock_at.source_finder = mock_sf_mod

    with patch.dict(sys.modules, {
        "AegeanTools": mock_at,
        "AegeanTools.source_finder": mock_sf_mod,
    }):
        from dsa110_continuum.source_finding import core as sf_core
        importlib.reload(sf_core)
        result = sf_core.run_aegean("fake.fits", "fake_bkg.fits", "fake_rms.fits")
        assert result == []


def test_run_aegean_returns_entries():
    """Mock SourceFinder returns 2 sources → list of SourceCatalogEntry with correct values."""
    import importlib
    sources = [
        _make_aegean_source(ra=344.0, dec=16.0, peak=0.05),
        _make_aegean_source(ra=345.0, dec=16.5, peak=2.5),
    ]
    mock_sf_instance = MagicMock()
    mock_sf_instance.find_sources_in_image.return_value = sources
    mock_sf_cls = MagicMock(return_value=mock_sf_instance)
    mock_sf_mod = MagicMock()
    mock_sf_mod.SourceFinder = mock_sf_cls
    mock_at = MagicMock()
    mock_at.source_finder = mock_sf_mod

    with patch.dict(sys.modules, {
        "AegeanTools": mock_at,
        "AegeanTools.source_finder": mock_sf_mod,
    }):
        from dsa110_continuum.source_finding import core as sf_core
        importlib.reload(sf_core)
        result = sf_core.run_aegean("fake.fits", "fake_bkg.fits", "fake_rms.fits")

    assert len(result) == 2
    assert all(isinstance(e, sf_core.SourceCatalogEntry) for e in result)
    assert abs(result[0].ra_deg - 344.0) < 1e-6
    assert abs(result[1].peak_flux_jy - 2.5) < 1e-6
    assert result[0].source_name.startswith("AEG_J")
    assert result[0].int_flux_jy == pytest.approx(0.05 * 1.1, rel=1e-5)


def test_run_aegean_source_missing_local_rms():
    """Sources missing local_rms attribute use 0.0 fallback without crashing."""
    import importlib
    # Build a mock source without a local_rms attribute
    src = MagicMock(spec=[])  # spec=[] means NO attributes are auto-created
    src.ra = 344.0
    src.dec = 16.0
    src.peak_flux = 0.1
    # Intentionally omit: err_peak_flux, int_flux, a, b, pa, local_rms
    # MagicMock with spec=[] will raise AttributeError for any getattr not in spec

    mock_sf_instance = MagicMock()
    mock_sf_instance.find_sources_in_image.return_value = [src]
    mock_sf_cls = MagicMock(return_value=mock_sf_instance)
    mock_sf_mod = MagicMock()
    mock_sf_mod.SourceFinder = mock_sf_cls
    mock_at = MagicMock()
    mock_at.source_finder = mock_sf_mod

    with patch.dict(sys.modules, {
        "AegeanTools": mock_at,
        "AegeanTools.source_finder": mock_sf_mod,
    }):
        from dsa110_continuum.source_finding import core as sf_core
        importlib.reload(sf_core)
        result = sf_core.run_aegean("fake.fits", "fake_bkg.fits", "fake_rms.fits")

    assert len(result) == 1
    entry = result[0]
    assert entry.local_rms_jy == 0.0  # fallback, not a crash
    assert entry.peak_flux_jy == pytest.approx(0.1)
    assert entry.a_arcsec == 0.0      # fallback
    assert entry.b_arcsec == 0.0      # fallback
