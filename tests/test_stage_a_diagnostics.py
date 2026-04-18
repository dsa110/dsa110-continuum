# tests/test_stage_a_diagnostics.py
import matplotlib
matplotlib.use("Agg")

import pytest
import matplotlib.pyplot as plt

from dsa110_continuum.visualization.config import FigureConfig, PlotStyle


def test_style_context_publication():
    """FigureConfig(PUBLICATION).style_context() applies scienceplots rcParams."""
    config = FigureConfig(style=PlotStyle.PUBLICATION)
    before = plt.rcParams.get("font.size", None)
    with config.style_context():
        inside = plt.rcParams.get("font.size", None)
    assert inside is not None
    after = plt.rcParams.get("font.size", None)
    assert after == before

import csv
import os
import tempfile
import numpy as np
from astropy.io import fits


def _make_minimal_fits(path: str, nx: int = 20, ny: int = 20) -> None:
    """Write a minimal FITS with WCS headers for testing."""
    data = np.random.default_rng(42).standard_normal((ny, nx)).astype(np.float32) * 0.001
    hdu = fits.PrimaryHDU(data)
    h = hdu.header
    h["NAXIS"] = 2
    h["NAXIS1"] = nx
    h["NAXIS2"] = ny
    h["CTYPE1"] = "RA---SIN"
    h["CTYPE2"] = "DEC--SIN"
    h["CRPIX1"] = nx // 2
    h["CRPIX2"] = ny // 2
    h["CRVAL1"] = 344.124
    h["CRVAL2"] = 16.15
    h["CDELT1"] = -20.0 / 3600.0
    h["CDELT2"] = 20.0 / 3600.0
    h["CUNIT1"] = "deg"
    h["CUNIT2"] = "deg"
    fits.writeto(path, data, h, overwrite=True)


def _make_forced_phot_csv(path: str, n: int = 5, use_injected: bool = False) -> None:
    """Write a minimal forced photometry CSV."""
    ref_col = "injected_flux_jy" if use_injected else "catalog_flux_jy"
    fieldnames = ["source_name", "ra_deg", "dec_deg", ref_col,
                  "measured_flux_jy", "flux_err_jy", "flux_ratio", "snr"]
    rows = []
    for i in range(n):
        ref_flux = 0.1 + i * 0.05
        meas_flux = ref_flux * (0.9 + i * 0.02)
        rows.append({
            "source_name": f"J344.{i:04d}+16.0000",
            "ra_deg": 344.0 + i * 0.01,
            "dec_deg": 16.15,
            ref_col: round(ref_flux, 5),
            "measured_flux_jy": round(meas_flux, 5),
            "flux_err_jy": 0.005,
            "flux_ratio": round(meas_flux / ref_flux, 4),
            "snr": round(meas_flux / 0.003, 1),
        })
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def test_plot_flux_scale_sim_mode():
    """plot_flux_scale with injected_flux_jy column writes a PNG file."""
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
        csv_path = f.name
    with tempfile.NamedTemporaryFile(suffix=".fits", delete=False) as f:
        fits_path = f.name
    with tempfile.TemporaryDirectory() as out_dir:
        try:
            _make_forced_phot_csv(csv_path, n=5, use_injected=True)
            _make_minimal_fits(fits_path)
            from dsa110_continuum.visualization.stage_a_diagnostics import plot_flux_scale
            result = plot_flux_scale(csv_path, fits_path, out_dir)
            assert result is not None
            assert os.path.exists(str(result))
            assert str(result).endswith(".png")
        finally:
            os.unlink(csv_path)
            os.unlink(fits_path)


def test_plot_flux_scale_catalog_mode():
    """plot_flux_scale with catalog_flux_jy column writes a PNG file."""
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
        csv_path = f.name
    with tempfile.NamedTemporaryFile(suffix=".fits", delete=False) as f:
        fits_path = f.name
    with tempfile.TemporaryDirectory() as out_dir:
        try:
            _make_forced_phot_csv(csv_path, n=5, use_injected=False)
            _make_minimal_fits(fits_path)
            from dsa110_continuum.visualization.stage_a_diagnostics import plot_flux_scale
            result = plot_flux_scale(csv_path, fits_path, out_dir)
            assert result is not None
            assert os.path.exists(str(result))
        finally:
            os.unlink(csv_path)
            os.unlink(fits_path)


def test_plot_flux_scale_empty_csv():
    """plot_flux_scale raises ValueError for a zero-row CSV."""
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
        csv_path = f.name
        f.write("source_name,ra_deg,dec_deg,catalog_flux_jy,measured_flux_jy,snr\n")
    with tempfile.TemporaryDirectory() as out_dir:
        try:
            from dsa110_continuum.visualization.stage_a_diagnostics import plot_flux_scale
            with pytest.raises(ValueError, match="No sources"):
                plot_flux_scale(csv_path, "/nonexistent/mosaic.fits", out_dir)
        finally:
            os.unlink(csv_path)


def test_plot_source_field_produces_file():
    """plot_source_field writes a PNG file when mosaic is readable."""
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
        csv_path = f.name
    with tempfile.NamedTemporaryFile(suffix=".fits", delete=False) as f:
        fits_path = f.name
    with tempfile.TemporaryDirectory() as out_dir:
        try:
            _make_forced_phot_csv(csv_path, n=5)
            _make_minimal_fits(fits_path)
            from dsa110_continuum.visualization.stage_a_diagnostics import plot_source_field
            result = plot_source_field(csv_path, fits_path, out_dir)
            assert result is not None
            assert os.path.exists(str(result))
            assert str(result).endswith(".png")
        finally:
            os.unlink(csv_path)
            os.unlink(fits_path)


def test_plot_source_field_no_mosaic():
    """plot_source_field returns a path even when mosaic doesn't exist (graceful degradation)."""
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
        csv_path = f.name
    with tempfile.TemporaryDirectory() as out_dir:
        try:
            _make_forced_phot_csv(csv_path, n=3)
            from dsa110_continuum.visualization.stage_a_diagnostics import plot_source_field
            # Missing mosaic should not crash — returns path to written PNG
            result = plot_source_field(csv_path, "/nonexistent/mosaic.fits", out_dir)
            assert result is not None
            assert os.path.exists(str(result))
        finally:
            os.unlink(csv_path)
