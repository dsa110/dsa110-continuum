import sqlite3
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits
from astropy.wcs import WCS


def _make_simple_wcs():
    wcs = WCS(naxis=2)
    wcs.wcs.crpix = [5, 5]
    wcs.wcs.cdelt = [-0.1, 0.1]
    wcs.wcs.crval = [40.0, 16.0]
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    return wcs


def test_sources_in_footprint_excludes_nan_positions():
    from dsa110_continuum.photometry.footprint import sources_in_footprint
    data = np.ones((10, 10))
    data[7:, :] = np.nan   # top 3 rows blanked (high dec = high row index with +cdelt_y)
    valid_mask = np.isfinite(data)
    wcs = _make_simple_wcs()
    # high-dec corner is blanked (maps to row ~9); centre pixel is valid
    ra  = np.array([40.45, 40.0])
    dec = np.array([16.45, 16.0])
    mask = sources_in_footprint(ra, dec, wcs, valid_mask)
    assert mask[0] == False   # blanked pixel
    assert mask[1] == True    # valid pixel


def test_sources_in_footprint_excludes_out_of_bounds():
    from dsa110_continuum.photometry.footprint import sources_in_footprint
    data = np.ones((10, 10))
    valid_mask = np.isfinite(data)
    wcs = _make_simple_wcs()
    ra  = np.array([41.5])   # far outside 10-px image
    dec = np.array([16.0])
    mask = sources_in_footprint(ra, dec, wcs, valid_mask)
    assert mask[0] == False


def test_sources_in_footprint_handles_empty_input():
    from dsa110_continuum.photometry.footprint import sources_in_footprint
    wcs = _make_simple_wcs()
    valid_mask = np.ones((10, 10), dtype=bool)
    result = sources_in_footprint(np.array([]), np.array([]), wcs, valid_mask)
    assert result.shape == (0,)
    assert result.dtype == bool


def test_measure_peak_box_returns_correct_flux():
    from dsa110_continuum.photometry.simple_peak import measure_peak_box
    wcs = _make_simple_wcs()
    data = np.zeros((10, 10))
    # crpix=[5,5] 1-indexed -> pixel (4,4) 0-indexed for crval=(40.0, 16.0)
    data[4, 4] = 0.5
    flux, snr, x, y = measure_peak_box(data, wcs, ra_deg=40.0, dec_deg=16.0,
                                        box_pix=2, rms=0.010)
    assert abs(flux - 0.5) < 0.001
    assert abs(snr - 50.0) < 1.0


def test_measure_peak_box_returns_nan_outside_image():
    from dsa110_continuum.photometry.simple_peak import measure_peak_box
    wcs = _make_simple_wcs()
    data = np.zeros((10, 10))
    flux, snr, x, y = measure_peak_box(data, wcs, ra_deg=99.0, dec_deg=99.0,
                                        box_pix=2, rms=0.010)
    assert not np.isfinite(flux)
    assert not np.isfinite(snr)


# ---------------------------------------------------------------------------
# Integration test for verify_sources.py
# ---------------------------------------------------------------------------

def _make_synthetic_fits(path: Path, source_flux_jy: float = 0.5,
                          noise_jy: float = 0.010) -> None:
    """Write a 50x50 pixel FITS mosaic with 3 bright sources and background noise.

    Sources are planted at pixel offsets from centre so they each have a
    distinct catalog position.  Noise ensures MAD-RMS > 0 and SNR is finite.
    """
    rng = np.random.default_rng(seed=42)
    ny, nx = 50, 50
    data = rng.normal(0.0, noise_jy, (ny, nx)).astype(np.float32)
    # crpix=(25, 25) 1-indexed -> pixel (24, 24) 0-indexed at crval
    # plant 3 sources: centre, +3 pix RA, +3 pix Dec
    for dy, dx in [(0, 0), (3, 0), (0, 3)]:
        data[24 + dy, 24 + dx] = source_flux_jy

    hdr = fits.Header()
    hdr["NAXIS"]  = 2
    hdr["NAXIS1"] = nx
    hdr["NAXIS2"] = ny
    hdr["CTYPE1"] = "RA---TAN"
    hdr["CTYPE2"] = "DEC--TAN"
    hdr["CRPIX1"] = 25.0
    hdr["CRPIX2"] = 25.0
    hdr["CRVAL1"] = 40.0
    hdr["CRVAL2"] = 16.0
    hdr["CDELT1"] = -0.1
    hdr["CDELT2"] =  0.1
    hdr["BUNIT"]  = "Jy/beam"
    fits.writeto(path, data, hdr, overwrite=True)


def _make_synthetic_master_db(path: Path) -> None:
    """Write a minimal master_sources.sqlite3 with 3 continuum sources.

    Positions match the 3 sources planted in ``_make_synthetic_fits``.
    """
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TABLE sources ("
        "source_id INTEGER PRIMARY KEY, ra_deg REAL, dec_deg REAL, flux_jy REAL)"
    )
    # crval=(40.0, 16.0), cdelt=(-0.1, 0.1) -> offsets of +3 pix
    sources = [
        (1, 40.0,       16.0,       0.5),   # centre
        (2, 40.0 + 0.3, 16.0,       0.5),   # RA offset (cdelt=-0.1, so +3px=-0.3deg RA? no)
        (3, 40.0,       16.0 + 0.3, 0.5),   # Dec offset
    ]
    # Note: cdelt1=-0.1 means RA decreases with pixel x. +3 pixels -> RA = 40.0 - 0.3 = 39.7
    sources = [
        (1, 40.0,  16.0,  0.5),
        (2, 39.7,  16.0,  0.5),
        (3, 40.0,  16.3,  0.5),
    ]
    conn.executemany("INSERT INTO sources VALUES (?, ?, ?, ?)", sources)
    conn.commit()
    conn.close()


def _make_empty_atnf_db(path: Path) -> None:
    """Write a minimal atnf_full.sqlite3 with no sources."""
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TABLE sources ("
        "source_id INTEGER PRIMARY KEY, ra_deg REAL, dec_deg REAL, "
        "flux_mjy REAL, name TEXT, period_s REAL, dm REAL)"
    )
    conn.commit()
    conn.close()


def _make_empty_nvss_db(path: Path) -> None:
    """Write a minimal nvss_full.sqlite3 with no sources."""
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TABLE sources ("
        "source_id INTEGER PRIMARY KEY, ra_deg REAL, dec_deg REAL, "
        "flux_mjy REAL, flux_err_mjy REAL)"
    )
    conn.commit()
    conn.close()


def test_verify_sources_integration():
    """End-to-end: synthetic FITS + synthetic DB => VERIFY PASS or WARN."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        fits_path = tmp / "mosaic.fits"
        master_db = tmp / "master_sources.sqlite3"
        nvss_db   = tmp / "nvss_full.sqlite3"
        atnf_db   = tmp / "atnf_full.sqlite3"
        out_csv   = tmp / "verify.csv"

        _make_synthetic_fits(fits_path, source_flux_jy=0.5, noise_jy=0.010)
        _make_synthetic_master_db(master_db)
        _make_empty_nvss_db(nvss_db)
        _make_empty_atnf_db(atnf_db)

        result = subprocess.run(
            [
                sys.executable, "scripts/verify_sources.py",
                "--fits",        str(fits_path),
                "--master-db",   str(master_db),
                "--nvss-db",     str(nvss_db),
                "--atnf-db",     str(atnf_db),
                "--out",         str(out_csv),
                "--min-flux-jy", "0.010",
                "--box-pix",     "3",
            ],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent,
        )
        stdout = result.stdout.strip()
        assert "VERIFY" in stdout, (
            f"No VERIFY line in stdout: {stdout!r}\nstderr: {result.stderr}"
        )
        assert "PASS" in stdout or "WARN" in stdout, (
            f"Expected PASS or WARN, got: {stdout!r}\nstderr: {result.stderr}"
        )
        # Output CSV must have at least one data row
        assert out_csv.exists(), "Output CSV not created"
        rows = out_csv.read_text().splitlines()
        assert len(rows) >= 2, (
            f"CSV has fewer than 2 lines (header + data): {rows}"
        )
