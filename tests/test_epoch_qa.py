"""Tests for dsa110_continuum.photometry.epoch_qa."""
from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from dsa110_continuum.photometry.epoch_qa import (
    QA_RATIO_HIGH,
    QA_RATIO_LOW,
    QA_RMS_LIMIT_MJY,
    EpochQAResult,
    measure_epoch_qa,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_test_fits(
    tmp_path: Path,
    rms_jy: float = 0.0085,
    n_sources: int = 6,
    source_flux_jy: float = 0.5,
    ra_center: float = 45.0,
    dec_center: float = 16.1,
) -> tuple[Path, list[tuple[float, float, float]]]:
    """Write a synthetic 500x500 FITS mosaic with Gaussian noise + point sources.

    Returns (fits_path, source_list) where source_list contains (ra, dec, flux_mjy)
    tuples at the exact sky positions of embedded sources.
    """
    ny, nx = 500, 500
    rng = np.random.default_rng(42)
    data = rng.normal(0, rms_jy, (ny, nx)).astype(np.float32)

    w = WCS(naxis=2)
    w.wcs.crpix = [nx // 2, ny // 2]
    w.wcs.cdelt = [-6.0 / 3600, 6.0 / 3600]
    w.wcs.crval = [ra_center, dec_center]
    w.wcs.ctype = ["RA---SIN", "DEC--SIN"]

    # Embed point sources at known pixel positions
    source_list: list[tuple[float, float, float]] = []
    if n_sources > 0:
        ys = np.linspace(100, 400, n_sources, dtype=int)
        for y in ys:
            x = nx // 2
            data[y, x] += source_flux_jy
            sky = w.pixel_to_world(x, y)
            source_list.append((sky.ra.deg, sky.dec.deg, source_flux_jy * 1000.0))

    hdr = w.to_header()
    hdr["BUNIT"] = "Jy/beam"
    hdu = fits.PrimaryHDU(data=data[np.newaxis, np.newaxis], header=hdr)
    out = tmp_path / "mosaic.fits"
    hdu.writeto(str(out), overwrite=True)
    return out, source_list


def _make_nvss_db(tmp_path: Path, sources: list[tuple[float, float, float]]) -> Path:
    """Write a minimal NVSS SQLite DB with (ra_deg, dec_deg, flux_mjy) rows."""
    db_path = tmp_path / "nvss.sqlite3"
    con = sqlite3.connect(str(db_path))
    con.execute(
        "CREATE TABLE sources ("
        "source_id INTEGER PRIMARY KEY, ra_deg REAL, dec_deg REAL, "
        "flux_mjy REAL, flux_err_mjy REAL, major_axis REAL, "
        "minor_axis REAL, position_angle REAL, catalog_row_id INTEGER)"
    )
    for i, (ra, dec, flux_mjy) in enumerate(sources):
        con.execute(
            "INSERT INTO sources VALUES (?, ?, ?, ?, 0, 0, 0, 0, ?)",
            (i + 1, ra, dec, flux_mjy, i + 1),
        )
    con.commit()
    con.close()
    return db_path


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestNominalCase:
    """Nominal case: all gates pass."""

    def test_all_gates_pass(self, tmp_path):
        fits_path, sources = _make_test_fits(tmp_path, rms_jy=0.0085, n_sources=6)
        nvss_db = _make_nvss_db(tmp_path, sources)
        result = measure_epoch_qa(str(fits_path), str(nvss_db))

        assert isinstance(result, EpochQAResult)
        assert result.ratio_gate == "PASS"
        assert result.rms_gate == "PASS"
        assert result.qa_result == "PASS"
        assert result.n_recovered >= 3
        assert result.mosaic_rms_mjy < QA_RMS_LIMIT_MJY

    def test_median_ratio_near_unity(self, tmp_path):
        fits_path, sources = _make_test_fits(tmp_path, rms_jy=0.0085, n_sources=6)
        nvss_db = _make_nvss_db(tmp_path, sources)
        result = measure_epoch_qa(str(fits_path), str(nvss_db))

        assert QA_RATIO_LOW <= result.median_ratio <= QA_RATIO_HIGH


class TestEmptyReferenceSet:
    """Empty / degenerate catalog inputs."""

    def test_no_catalog_sources_gives_fail(self, tmp_path):
        fits_path, _ = _make_test_fits(tmp_path, rms_jy=0.0085, n_sources=6)
        nvss_db = _make_nvss_db(tmp_path, [])  # empty catalog
        result = measure_epoch_qa(str(fits_path), str(nvss_db))

        assert result.n_catalog == 0
        assert result.n_recovered == 0
        assert result.ratio_gate == "FAIL"  # no detections
        assert result.completeness_gate == "SKIP"  # <5 sources
        assert result.qa_result == "FAIL"

    def test_sources_outside_footprint_not_counted(self, tmp_path):
        fits_path, _ = _make_test_fits(tmp_path, rms_jy=0.0085, n_sources=0)
        # Sources far outside the image footprint
        far_sources = [(180.0, -30.0, 500.0) for _ in range(10)]
        nvss_db = _make_nvss_db(tmp_path, far_sources)
        result = measure_epoch_qa(str(fits_path), str(nvss_db))

        assert result.n_catalog == 0  # none in footprint
        assert result.n_recovered == 0


class TestGateLogic:
    """Individual gate logic."""

    def test_rms_gate_fails_when_noisy(self, tmp_path):
        fits_path, sources = _make_test_fits(tmp_path, rms_jy=0.050, n_sources=6)
        nvss_db = _make_nvss_db(tmp_path, sources)
        result = measure_epoch_qa(str(fits_path), str(nvss_db))

        assert result.rms_gate == "FAIL"
        assert result.mosaic_rms_mjy > QA_RMS_LIMIT_MJY
        assert result.qa_result == "FAIL"

    def test_completeness_skip_with_few_sources(self, tmp_path):
        fits_path, sources = _make_test_fits(tmp_path, rms_jy=0.0085, n_sources=3)
        # Only 3 catalog sources → below QA_MIN_CATALOG_SOURCES (5)
        nvss_db = _make_nvss_db(tmp_path, sources[:3])
        result = measure_epoch_qa(str(fits_path), str(nvss_db))

        assert result.completeness_gate == "SKIP"

    def test_completeness_fails_when_too_few_recovered(self, tmp_path):
        fits_path, _ = _make_test_fits(tmp_path, rms_jy=0.0085, n_sources=0)
        # 8 catalog sources but no embedded sources → 0% completeness
        dummy_sources = [(45.0, 16.1 + i * 0.02, 500.0) for i in range(8)]
        nvss_db = _make_nvss_db(tmp_path, dummy_sources)
        result = measure_epoch_qa(str(fits_path), str(nvss_db))

        assert result.completeness_gate == "FAIL"
        assert result.qa_result == "FAIL"

    def test_ratio_gate_fails_when_scale_wrong(self, tmp_path):
        # Inject very bright sources but catalog says they should be dim
        # → measured/catalog ratio >> 1.2
        fits_path, sources = _make_test_fits(
            tmp_path, rms_jy=0.0085, n_sources=6, source_flux_jy=2.0,
        )
        # Catalog says these are 50 mJy sources, but they measure at 2 Jy
        weak_cat = [(ra, dec, 50.0) for ra, dec, _ in sources]
        nvss_db = _make_nvss_db(tmp_path, weak_cat)
        result = measure_epoch_qa(str(fits_path), str(nvss_db))

        assert result.ratio_gate == "FAIL"
        assert result.median_ratio > QA_RATIO_HIGH


class TestOverallVerdict:
    """Overall verdict logic."""

    def test_pass_requires_all_active_gates_pass(self, tmp_path):
        fits_path, sources = _make_test_fits(tmp_path, rms_jy=0.0085, n_sources=6)
        nvss_db = _make_nvss_db(tmp_path, sources)
        result = measure_epoch_qa(str(fits_path), str(nvss_db))

        if result.ratio_gate == "PASS" and result.rms_gate == "PASS":
            assert result.qa_result == "PASS"

    def test_skip_gate_does_not_cause_fail(self, tmp_path):
        # 3 sources → completeness SKIP, but ratio and RMS should PASS
        fits_path, sources = _make_test_fits(tmp_path, rms_jy=0.0085, n_sources=3)
        nvss_db = _make_nvss_db(tmp_path, sources[:3])
        result = measure_epoch_qa(str(fits_path), str(nvss_db))

        assert result.completeness_gate == "SKIP"
        # With 3 detections, ratio gate needs >= QA_MIN_RATIO_DETECTIONS (3)
        if result.ratio_gate == "PASS" and result.rms_gate == "PASS":
            assert result.qa_result == "PASS"


class TestToDictCSV:
    """to_dict() serialisation."""

    def test_to_dict_has_all_csv_columns(self, tmp_path):
        fits_path, sources = _make_test_fits(tmp_path)
        nvss_db = _make_nvss_db(tmp_path, sources)
        result = measure_epoch_qa(str(fits_path), str(nvss_db))
        d = result.to_dict()

        required = {
            "n_catalog", "n_recovered", "completeness_frac",
            "median_ratio", "ratio_gate", "completeness_gate",
            "rms_gate", "mosaic_rms_mjy", "qa_result",
        }
        assert required.issubset(d.keys())
        assert "ratios" not in d  # excluded from CSV row
