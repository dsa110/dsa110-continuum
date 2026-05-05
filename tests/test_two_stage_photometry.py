# tests/test_two_stage_photometry.py
import dataclasses
import math
import numpy as np
import pytest
from pathlib import Path
from unittest.mock import patch

from dsa110_continuum.photometry.two_stage import CoarseAugment, beam_correction_factor, run_coarse_pass, run_two_stage

MOSAIC = Path("pipeline_outputs/step6/step6_mosaic.fits")


def test_coarse_augment_fields():
    aug = CoarseAugment(
        ra_deg=344.1, dec_deg=16.15,
        coarse_peak_jyb=0.12, coarse_snr=8.5, passed_coarse=True,
    )
    assert dataclasses.asdict(aug) == {
        "ra_deg": 344.1,
        "dec_deg": 16.15,
        "coarse_peak_jyb": 0.12,
        "coarse_snr": 8.5,
        "passed_coarse": True,
    }


def test_beam_correction_known_values():
    # BMAJ=36.9", BMIN=25.5", CDELT1=CDELT2=20" (Step 6 mosaic — square pixels)
    mock_hdr = {
        "BMAJ": 36.9 / 3600,
        "BMIN": 25.5 / 3600,
        "CDELT1": -20.0 / 3600,   # RA axis is typically negative
        "CDELT2": 20.0 / 3600,
    }
    with patch("dsa110_continuum.photometry.two_stage.fits.getheader", return_value=mock_hdr):
        factor = beam_correction_factor("dummy.fits")
    bmaj_rad = math.radians(36.9 / 3600)
    bmin_rad = math.radians(25.5 / 3600)
    pixel_area_sr = math.radians(20.0 / 3600) * math.radians(20.0 / 3600)
    expected = (math.pi / (4 * math.log(2))) * bmaj_rad * bmin_rad / pixel_area_sr
    assert abs(factor - expected) / expected < 1e-6


def test_beam_correction_missing_keywords():
    mock_hdr = {}
    with patch("dsa110_continuum.photometry.two_stage.fits.getheader", return_value=mock_hdr):
        factor = beam_correction_factor("dummy.fits")
    assert factor == 1.0


def test_beam_correction_zero_cdelt():
    mock_hdr = {"BMAJ": 36.9 / 3600, "BMIN": 25.5 / 3600, "CDELT1": 20.0 / 3600, "CDELT2": 0.0}
    with patch("dsa110_continuum.photometry.two_stage.fits.getheader", return_value=mock_hdr):
        factor = beam_correction_factor("dummy.fits")
    assert factor == 1.0


def test_coarse_pass_synthetic_fits():
    """run_coarse_pass works on a small synthetic FITS — no real mosaic needed."""
    import tempfile
    import os
    from astropy.io import fits as afits
    from astropy.wcs import WCS as AWCS
    import numpy as np

    # Build a tiny 50×50 FITS with a point source at the centre
    ny, nx = 50, 50
    data = np.zeros((ny, nx), dtype=np.float32)
    data[25, 25] = 0.5  # 0.5 Jy/beam source at centre

    w = AWCS(naxis=2)
    w.wcs.ctype = ["RA---SIN", "DEC--SIN"]
    w.wcs.crval = [180.0, 30.0]
    w.wcs.crpix = [25.0, 25.0]
    w.wcs.cdelt = [-20.0 / 3600, 20.0 / 3600]  # 20 arcsec/pix

    hdr = w.to_header()
    hdr["BMAJ"] = 36.9 / 3600
    hdr["BMIN"] = 25.5 / 3600
    hdr["BPA"] = 130.0
    hdu = afits.PrimaryHDU(data=data, header=hdr)

    with tempfile.NamedTemporaryFile(suffix=".fits", delete=False) as tf:
        fpath = tf.name
    try:
        hdu.writeto(fpath)

        # Source is at the WCS centre — (180.0, 30.0)
        coords = [(180.0, 30.0)]
        results = run_coarse_pass(fpath, coords, global_rms=0.001, snr_coarse_min=3.0)
    finally:
        os.unlink(fpath)

    assert len(results) == 1
    aug = results[0]
    assert np.isfinite(aug.coarse_peak_jyb)
    assert aug.coarse_peak_jyb == pytest.approx(0.5, rel=0.05)
    assert aug.coarse_snr == pytest.approx(500.0, rel=0.05)  # 0.5 / 0.001
    assert aug.passed_coarse is True


def test_two_stage_returns_paired_lists():
    """run_two_stage returns one result and one augment per input coord."""
    import tempfile
    from astropy.io import fits as afits
    from astropy.wcs import WCS as AWCS

    ny, nx = 120, 120
    rng = np.random.default_rng(0)
    data = rng.normal(0, 0.001, (ny, nx)).astype(np.float32)
    data[60, 60] = 0.5

    w = AWCS(naxis=2)
    w.wcs.ctype = ["RA---SIN", "DEC--SIN"]
    w.wcs.crval = [180.0, 30.0]
    w.wcs.crpix = [60.0, 60.0]
    w.wcs.cdelt = [-20.0 / 3600, 20.0 / 3600]
    hdr = w.to_header()
    hdr["BMAJ"] = 36.9 / 3600
    hdr["BMIN"] = 25.5 / 3600

    with tempfile.NamedTemporaryFile(suffix=".fits", delete=False) as f:
        fpath = f.name
    afits.PrimaryHDU(data=data, header=hdr).writeto(fpath, overwrite=True)

    coords = [(180.0, 30.0), (180.0, 30.1)]
    results, augments = run_two_stage(fpath, coords, snr_coarse_min=0.0)
    assert len(results) == len(augments) == 2


def test_fine_pass_skips_failing_coarse():
    """Coord that fails the coarse SNR gate gets a NaN fine result."""
    import tempfile
    from astropy.io import fits as afits
    from astropy.wcs import WCS as AWCS

    ny, nx = 120, 120
    rng = np.random.default_rng(1)
    data = rng.normal(0, 0.001, (ny, nx)).astype(np.float32)
    data[60, 60] = 0.5

    w = AWCS(naxis=2)
    w.wcs.ctype = ["RA---SIN", "DEC--SIN"]
    w.wcs.crval = [180.0, 30.0]
    w.wcs.crpix = [60.0, 60.0]
    w.wcs.cdelt = [-20.0 / 3600, 20.0 / 3600]
    hdr = w.to_header()
    hdr["BMAJ"] = 36.9 / 3600
    hdr["BMIN"] = 25.5 / 3600

    with tempfile.NamedTemporaryFile(suffix=".fits", delete=False) as f:
        fpath = f.name
    afits.PrimaryHDU(data=data, header=hdr).writeto(fpath, overwrite=True)

    # Use enormous rms so the coord at image centre fails the coarse gate
    coords = [(180.0, 30.0)]
    results, augments = run_two_stage(fpath, coords, global_rms=1e6, snr_coarse_min=3.0)
    assert augments[0].passed_coarse is False
    assert not np.isfinite(results[0].peak_jyb)


def test_two_stage_synthetic_fits():
    """run_two_stage: one coord passes coarse, one fails (off-image)."""
    import tempfile
    from astropy.io import fits as afits
    from astropy.wcs import WCS as AWCS

    # Use a large enough image so the annulus (r=30..50) has pixels with noise.
    # Omit BPA so measure_many uses the simple-peak fallback (avoids the
    # weighted-convolution path that requires the optional dsa110_contimg package).
    np.random.seed(0)
    ny, nx = 120, 120
    data = np.random.normal(0, 0.001, (ny, nx)).astype(np.float32)
    data[60, 60] = 0.5  # strong source at centre

    w = AWCS(naxis=2)
    w.wcs.ctype = ["RA---SIN", "DEC--SIN"]
    w.wcs.crval = [180.0, 30.0]
    w.wcs.crpix = [60.0, 60.0]
    w.wcs.cdelt = [-20.0 / 3600, 20.0 / 3600]
    hdr = w.to_header()
    hdr["BMAJ"] = 36.9 / 3600
    hdr["BMIN"] = 25.5 / 3600
    # Intentionally omit BPA so measure_many uses the simple-peak fallback

    with tempfile.NamedTemporaryFile(suffix=".fits", delete=False) as f:
        fpath = f.name
    afits.PrimaryHDU(data=data, header=hdr).writeto(fpath, overwrite=True)

    coords = [
        (180.0, 30.0),    # on-image: should pass coarse
        (180.0, 35.0),    # off-image: should fail coarse (NaN peak)
    ]
    results, augments = run_two_stage(fpath, coords, global_rms=0.001, snr_coarse_min=3.0)

    assert len(results) == len(augments) == 2
    assert augments[0].passed_coarse is True
    assert augments[1].passed_coarse is False
    # Fine-pass result for survivor should have a finite peak
    assert np.isfinite(results[0].peak_jyb)
    # Fine-pass result for coarse failure should be NaN placeholder
    assert not np.isfinite(results[1].peak_jyb)


@pytest.mark.skipif(
    not Path("pipeline_outputs/step6/step6_mosaic.fits").exists(),
    reason="Step 6 mosaic not on disk"
)
def test_beam_correction_ratio_bright_sources():
    """Beam-corrected peak / injected_flux is in physically reasonable range.

    NOTE: The task spec referenced `SkyModel(seed=42)` from
    `dsa110_continuum.simulation.ground_truth`, but that class does not exist.
    The actual API uses `SimulationHarness.make_sky_model()` from
    `dsa110_continuum.simulation.harness` (as used in validate_step6_mosaic.py).
    The attribute access pattern `sky.ra[k].deg`, `sky.stokes[0, 0, k].value`,
    and `sky.Ncomponents` matches the spec and works correctly.
    """
    from dsa110_continuum.simulation.harness import SimulationHarness

    # Mirror the setup from validate_step6_mosaic.py (tile 0, seed=42)
    harness = SimulationHarness(
        n_antennas=117, n_integrations=24, n_sky_sources=5,
        noise_jy=1.0, seed=42, use_real_positions=True,
    )
    harness.pointing_ra_deg = 344.124
    harness.pointing_dec_deg = 16.15
    sky = harness.make_sky_model(fov_deg=3.0)

    # --- Attribute access matches validate_step6_mosaic.py exactly ---
    coords = [(float(sky.ra[k].deg), float(sky.dec[k].deg)) for k in range(sky.Ncomponents)]
    injected_flux = [float(sky.stokes[0, 0, k].value) for k in range(sky.Ncomponents)]
    # -----------------------------------------------------------------

    mosaic_path = "pipeline_outputs/step6/step6_mosaic.fits"
    correction = beam_correction_factor(mosaic_path)
    coarse = run_coarse_pass(mosaic_path, coords, global_rms=None, snr_coarse_min=0.0)

    ratios = []
    for aug, s_inj in zip(coarse, injected_flux):
        if np.isfinite(aug.coarse_peak_jyb) and s_inj > 0:
            ratio = (aug.coarse_peak_jyb * correction) / s_inj
            ratios.append(ratio)

    assert len(ratios) >= 2, f"Need at least 2 finite measurements, got {len(ratios)}"
    median_ratio = float(np.median(ratios))
    # The Step 6 mosaic is a noise-weighted dirty-image mosaic (no CLEAN), so the
    # peak Jy/beam is well below the injected flux.  validate_step6_mosaic.py
    # documents the expected mean ratio as ~0.05 (range 0.00–0.10) for this
    # simulation setup.  The beam_correction_factor falls back to 1.0 because the
    # mosaic header carries no BMAJ/BMIN keywords.
    # We use a generous range of 1e-3 to 0.5 to confirm the coarse pass is
    # returning sensible (non-zero, not astronomically large) flux values.
    assert 1e-3 <= median_ratio <= 0.5, (
        f"Median beam-corrected ratio = {median_ratio:.4f} — expected in [0.001, 0.5] "
        "for a dirty-image mosaic with beam_correction_factor=1.0"
    )


# ── Task 5: CLI integration tests ──────────────────────────────────────────

@pytest.mark.skipif(
    not Path("pipeline_outputs/step6/step6_mosaic.fits").exists(),
    reason="Step 6 mosaic not on disk",
)
def test_cli_simple_peak_sim_produces_csv():
    import csv as _csv, tempfile, subprocess, sys
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
        out_csv = f.name
    result = subprocess.run(
        [sys.executable, "scripts/forced_photometry.py",
         "--mosaic", "pipeline_outputs/step6/step6_mosaic.fits",
         "--method", "simple_peak",
         "--sim",
         "--output", out_csv],
        capture_output=True, text=True,
        cwd="/home/user/workspace/dsa110-continuum",
    )
    assert result.returncode == 0, result.stderr
    with open(out_csv) as f:
        rows = list(_csv.DictReader(f))
    assert len(rows) > 0
    assert "measured_flux_jy" in rows[0]
    assert "snr" in rows[0]
    assert "injected_flux_jy" in rows[0]


@pytest.mark.skipif(
    not Path("pipeline_outputs/step6/step6_mosaic.fits").exists(),
    reason="Step 6 mosaic not on disk",
)
def test_cli_two_stage_sim_produces_coarse_snr_column():
    import csv as _csv, tempfile, subprocess, sys
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
        out_csv = f.name
    result = subprocess.run(
        [sys.executable, "scripts/forced_photometry.py",
         "--mosaic", "pipeline_outputs/step6/step6_mosaic.fits",
         "--method", "two_stage",
         "--sim",
         "--snr-coarse", "0.0",
         "--output", out_csv],
        capture_output=True, text=True,
        cwd="/home/user/workspace/dsa110-continuum",
    )
    assert result.returncode == 0, result.stderr
    with open(out_csv) as f:
        rows = list(_csv.DictReader(f))
    assert len(rows) > 0
    assert "coarse_snr" in rows[0]
    assert "passed_coarse" in rows[0]
