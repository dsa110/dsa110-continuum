# tests/test_two_stage_photometry.py
import math
import numpy as np
import pytest
from pathlib import Path
from unittest.mock import patch

from dsa110_continuum.photometry.two_stage import CoarseAugment, beam_correction_factor, run_coarse_pass

MOSAIC = Path("pipeline_outputs/step6/step6_mosaic.fits")


def test_coarse_augment_fields():
    import dataclasses
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


@pytest.mark.skipif(not MOSAIC.exists(), reason="Step 6 mosaic not on disk")
def test_coarse_pass_returns_finite():
    coords = [(344.124, 16.15), (346.71, 16.15)]
    results = run_coarse_pass(str(MOSAIC), coords, global_rms=None)
    assert len(results) == 2
    for aug in results:
        assert isinstance(aug, CoarseAugment)
        assert np.isfinite(aug.coarse_peak_jyb) or np.isnan(aug.coarse_peak_jyb)


@pytest.mark.skipif(not MOSAIC.exists(), reason="Step 6 mosaic not on disk")
def test_snr_gate_all_pass_with_low_rms():
    coords = [(344.124, 16.15)]
    results = run_coarse_pass(str(MOSAIC), coords, global_rms=1e-6, snr_coarse_min=3.0)
    assert results[0].passed_coarse is True


@pytest.mark.skipif(not MOSAIC.exists(), reason="Step 6 mosaic not on disk")
def test_snr_gate_all_fail_with_high_rms():
    coords = [(344.124, 16.15)]
    results = run_coarse_pass(str(MOSAIC), coords, global_rms=1e6, snr_coarse_min=3.0)
    assert results[0].passed_coarse is False
