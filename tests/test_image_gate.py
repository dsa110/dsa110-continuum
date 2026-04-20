"""Tests for the pre-source-finding image quality gate."""
import os
import tempfile

import numpy as np
import pytest
from astropy.io import fits


def _make_test_fits(path: str, shape: tuple = (100, 100),
                    peak: float = 1.0, rms: float = 0.01,
                    nan_frac: float = 0.0, add_beam: bool = False) -> None:
    """Write a synthetic FITS for image gate tests."""
    rng = np.random.default_rng(42)
    data = rng.standard_normal(shape).astype(np.float32) * rms
    data[shape[0]//2, shape[1]//2] = peak   # inject a bright source
    if nan_frac > 0:
        n_nan = int(nan_frac * data.size)
        idx = rng.choice(data.size, n_nan, replace=False)
        data.flat[idx] = np.nan
    hdu = fits.PrimaryHDU(data)
    if add_beam:
        hdu.header["BMAJ"] = 36.9 / 3600.0
        hdu.header["BMIN"] = 25.5 / 3600.0
        hdu.header["BPA"] = 130.75
    hdu.header["CDELT1"] = -20.0 / 3600.0
    hdu.header["CDELT2"] = 20.0 / 3600.0
    fits.writeto(path, data, hdu.header, overwrite=True)


def test_image_gate_pass():
    """High dynamic range, low RMS ratio → overall PASS."""
    with tempfile.NamedTemporaryFile(suffix=".fits", delete=False) as f:
        path = f.name
    try:
        _make_test_fits(path, peak=1.0, rms=0.005)  # DR=200, RMS ratio ~0.5
        from dsa110_continuum.qa.image_gate import check_image_quality_for_source_finding
        result = check_image_quality_for_source_finding(
            path, integration_time_s=12.88
        )
        assert result.overall in ("PASS", "WARN"), f"Expected PASS or WARN, got {result.overall}"
        assert result.dynamic_range > 50
        assert result.pixel_coverage_frac > 0.9
    finally:
        os.unlink(path)


def test_image_gate_fail_low_dynamic_range():
    """Dynamic range < 30 → FAIL."""
    with tempfile.NamedTemporaryFile(suffix=".fits", delete=False) as f:
        path = f.name
    try:
        # peak=0.05, rms=0.01 → DR=5 → below FAIL threshold of 30
        _make_test_fits(path, peak=0.05, rms=0.01)
        from dsa110_continuum.qa.image_gate import check_image_quality_for_source_finding
        result = check_image_quality_for_source_finding(path, integration_time_s=12.88)
        assert result.dynamic_range_gate == "FAIL"
        assert result.overall == "FAIL"
    finally:
        os.unlink(path)


def test_image_gate_warn_dynamic_range():
    """Dynamic range between 30 and 100 → WARN (not FAIL)."""
    with tempfile.NamedTemporaryFile(suffix=".fits", delete=False) as f:
        path = f.name
    try:
        # peak=0.5, rms=0.01 → DR=50 → between WARN(100) and FAIL(30)
        _make_test_fits(path, peak=0.5, rms=0.01)
        from dsa110_continuum.qa.image_gate import check_image_quality_for_source_finding
        result = check_image_quality_for_source_finding(path, integration_time_s=12.88)
        assert result.dynamic_range_gate == "WARN"
    finally:
        os.unlink(path)


def test_image_gate_fail_pixel_coverage():
    """More than 50% NaN pixels → pixel_coverage_gate FAIL and overall FAIL."""
    with tempfile.NamedTemporaryFile(suffix=".fits", delete=False) as f:
        path = f.name
    try:
        _make_test_fits(path, peak=1.0, rms=0.005, nan_frac=0.6)
        from dsa110_continuum.qa.image_gate import check_image_quality_for_source_finding
        result = check_image_quality_for_source_finding(path, integration_time_s=12.88)
        assert result.pixel_coverage_gate == "FAIL"
        assert result.overall == "FAIL"
    finally:
        os.unlink(path)


def test_image_gate_result_has_all_fields():
    """ImageQAResult has all documented fields."""
    with tempfile.NamedTemporaryFile(suffix=".fits", delete=False) as f:
        path = f.name
    try:
        _make_test_fits(path, peak=1.0, rms=0.005)
        from dsa110_continuum.qa.image_gate import check_image_quality_for_source_finding, ImageQAResult
        result = check_image_quality_for_source_finding(path, integration_time_s=12.88)
        assert isinstance(result, ImageQAResult)
        for field in ("dynamic_range", "dynamic_range_gate", "rms_mjy",
                      "theoretical_rms_mjy", "rms_ratio", "rms_ratio_gate",
                      "pixel_coverage_frac", "pixel_coverage_gate", "overall"):
            assert hasattr(result, field), f"Missing field: {field}"
    finally:
        os.unlink(path)


def test_image_gate_beam_sanity_logged(caplog):
    """FITS with BMAJ/BMIN → beam sanity check runs and logs something."""
    import logging
    with tempfile.NamedTemporaryFile(suffix=".fits", delete=False) as f:
        path = f.name
    try:
        _make_test_fits(path, peak=1.0, rms=0.005, add_beam=True)
        from dsa110_continuum.qa.image_gate import check_image_quality_for_source_finding
        with caplog.at_level(logging.INFO, logger="dsa110_continuum.qa.image_gate"):
            check_image_quality_for_source_finding(path, integration_time_s=12.88)
        # Just assert the function ran and returned a valid result (beam info logged)
        assert any("beam" in r.message.lower() or "dynamic" in r.message.lower()
                   for r in caplog.records), "Expected at least one informational log"
    finally:
        os.unlink(path)
