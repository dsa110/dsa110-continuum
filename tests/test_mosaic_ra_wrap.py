"""
Tests for the mosaic RA-wrap fix and reprojection performance path (Task 9).

Tests cover:
  1. compute_optimal_wcs: non-wrap (normal) cases
  2. compute_optimal_wcs: RA wrap-around near 0°/360° (the bug fix)
  3. compute_optimal_wcs: CRVAL stays in [0, 360)
  4. compute_optimal_wcs: edge cases (single tile, two-tile wrap)
  5. fast_reproject_and_coadd: basic output shape/type
  6. fast_reproject_and_coadd: auto-WCS from reproject.mosaicking
  7. fast_reproject_and_coadd: supplied WCS + shape
  8. fast_reproject_and_coadd: footprint coverage
  9. fast_reproject_and_coadd: bad reproject_function raises
"""
from __future__ import annotations

import numpy as np
import pytest
from astropy.io import fits
from astropy.wcs import WCS

# Import directly from the submodule to avoid the mosaic/__init__.py chain
# which requires dsa110_contimg (H17-only) decorators via jobs.py etc.
import importlib.util, sys
_spec = importlib.util.spec_from_file_location(
    "dsa110_continuum.mosaic.builder",
    "/home/user/workspace/dsa110-continuum/dsa110_continuum/mosaic/builder.py",
)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["dsa110_continuum.mosaic.builder"] = _mod
_spec.loader.exec_module(_mod)

compute_optimal_wcs = _mod.compute_optimal_wcs
fast_reproject_and_coadd = _mod.fast_reproject_and_coadd


# ── FITS HDU helpers ──────────────────────────────────────────────────────────

def _make_hdu(
    ra_center_deg: float,
    dec_center_deg: float,
    nx: int = 64,
    ny: int = 64,
    cdelt_deg: float = 1.0 / 60,  # 1 arcmin/pixel
    value: float = 1.0,
) -> fits.PrimaryHDU:
    """Create a minimal synthetic FITS HDU centred at (ra, dec)."""
    wcs = WCS(naxis=2)
    wcs.wcs.crpix = [nx / 2 + 0.5, ny / 2 + 0.5]
    wcs.wcs.crval = [ra_center_deg, dec_center_deg]
    wcs.wcs.cdelt = [-cdelt_deg, cdelt_deg]
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]

    data = np.full((ny, nx), value, dtype=np.float32)
    hdr = wcs.to_header()
    hdr["NAXIS"] = 2
    hdr["NAXIS1"] = nx
    hdr["NAXIS2"] = ny
    hdr["BMAJ"] = 5 / 3600.0  # 5 arcsec beam — Jy/beam
    hdr["BMIN"] = 5 / 3600.0
    hdr["BPA"] = 0.0
    return fits.PrimaryHDU(data=data, header=hdr)


# ══════════════════════════════════════════════════════════════════════════════
# 1–4. compute_optimal_wcs RA wrap fixes
# ══════════════════════════════════════════════════════════════════════════════

class TestComputeOptimalWCSNoWrap:

    def test_returns_wcs_and_shape(self):
        hdus = [_make_hdu(180.0, 16.0), _make_hdu(185.0, 16.0)]
        wcs, shape = compute_optimal_wcs(hdus)
        assert isinstance(wcs, WCS)
        ny, nx = shape
        assert ny > 0 and nx > 0

    def test_center_ra_close_to_tile_mean(self):
        """CRVAL[0] should be near 182.5° for tiles at 180° and 185°."""
        hdus = [_make_hdu(180.0, 16.0), _make_hdu(185.0, 16.0)]
        wcs, _ = compute_optimal_wcs(hdus)
        crval_ra = wcs.wcs.crval[0]
        # Allow ±5° for pixel-scale rounding
        assert abs(crval_ra - 182.5) < 5.0

    def test_center_dec_close_to_tile_mean(self):
        hdus = [_make_hdu(180.0, 10.0), _make_hdu(180.0, 20.0)]
        wcs, _ = compute_optimal_wcs(hdus)
        crval_dec = wcs.wcs.crval[1]
        assert abs(crval_dec - 15.0) < 3.0

    def test_shape_large_enough_to_cover_tiles(self):
        """Output grid should be large enough to cover both tiles."""
        cdelt = 1.0 / 60  # 1 arcmin/pixel
        hdus = [
            _make_hdu(170.0, 16.0, nx=64, ny=64, cdelt_deg=cdelt),
            _make_hdu(190.0, 16.0, nx=64, ny=64, cdelt_deg=cdelt),
        ]
        wcs, (ny, nx) = compute_optimal_wcs(hdus)
        # The 20-deg RA span needs at least 20*60 = 1200 pixels
        assert nx >= 1000


class TestComputeOptimalWCSWrap:

    def test_wrap_around_360_gives_compact_output(self):
        """Tiles at 355° and 5° should produce a small RA span, not ~350°."""
        hdus = [_make_hdu(355.0, 16.0), _make_hdu(5.0, 16.0)]
        wcs, (ny, nx) = compute_optimal_wcs(hdus)
        # RA span should be ~10° not ~350°.  At 1 arcmin/pixel, 10° = 600 px.
        # 350° would give >20000 px.
        assert nx < 5000, (
            f"RA-wrap bug: output grid is {nx} pixels wide "
            f"(expected <5000 for a ~10° span)"
        )

    def test_wrap_around_crval_in_range(self):
        """CRVAL[0] must be in [0, 360) even after wrap correction."""
        hdus = [_make_hdu(358.0, 16.0), _make_hdu(2.0, 16.0)]
        wcs, _ = compute_optimal_wcs(hdus)
        crval_ra = wcs.wcs.crval[0]
        assert 0.0 <= crval_ra < 360.0, f"CRVAL[0]={crval_ra} out of range"

    def test_wrap_center_near_zero(self):
        """Three tiles at 355°, 0°, 5° should centre near 0° (≡360°)."""
        hdus = [_make_hdu(355.0, 16.0), _make_hdu(0.0, 16.0), _make_hdu(5.0, 16.0)]
        wcs, _ = compute_optimal_wcs(hdus)
        crval_ra = wcs.wcs.crval[0]
        # Centre at 0°, wrapped to [0, 360): should be 0° or 360°→0°
        # Allow ±5° for rounding
        dist_from_zero = min(abs(crval_ra), abs(crval_ra - 360))
        assert dist_from_zero < 5.0, f"Centre RA={crval_ra:.2f}° too far from 0°"

    def test_non_wrap_case_unchanged(self):
        """Non-wrapping tiles (e.g. 10°–30°) should not be affected by the fix."""
        hdus = [_make_hdu(10.0, 16.0), _make_hdu(20.0, 16.0), _make_hdu(30.0, 16.0)]
        wcs, (ny, nx) = compute_optimal_wcs(hdus)
        assert nx < 5000
        crval = wcs.wcs.crval[0]
        assert 0.0 <= crval < 360.0

    def test_wrap_with_dec_variation(self):
        """RA wrap + varying Dec should still produce a valid, compact grid."""
        hdus = [
            _make_hdu(350.0, 10.0),
            _make_hdu(355.0, 16.0),
            _make_hdu(5.0, 16.0),
            _make_hdu(10.0, 20.0),
        ]
        wcs, (ny, nx) = compute_optimal_wcs(hdus)
        assert nx < 10_000
        assert ny > 0
        assert 0.0 <= wcs.wcs.crval[0] < 360.0

    def test_single_tile(self):
        """A single tile should return a valid WCS centred on that tile."""
        hdu = _make_hdu(45.0, 30.0)
        wcs, (ny, nx) = compute_optimal_wcs([hdu])
        assert nx > 0 and ny > 0
        assert 0.0 <= wcs.wcs.crval[0] < 360.0


# ══════════════════════════════════════════════════════════════════════════════
# 5–9. fast_reproject_and_coadd
# ══════════════════════════════════════════════════════════════════════════════

class TestFastReprojectAndCoadd:

    def test_returns_two_arrays(self):
        hdus = [_make_hdu(180.0, 16.0, nx=32, ny=32),
                _make_hdu(182.0, 16.0, nx=32, ny=32)]
        mosaic, footprint = fast_reproject_and_coadd(hdus)
        assert isinstance(mosaic, np.ndarray)
        assert isinstance(footprint, np.ndarray)

    def test_shapes_match(self):
        hdus = [_make_hdu(180.0, 16.0, nx=32, ny=32),
                _make_hdu(182.0, 16.0, nx=32, ny=32)]
        mosaic, footprint = fast_reproject_and_coadd(hdus)
        assert mosaic.shape == footprint.shape

    def test_output_dtype_float32(self):
        hdus = [_make_hdu(180.0, 16.0, nx=32, ny=32)]
        mosaic, footprint = fast_reproject_and_coadd(hdus)
        assert mosaic.dtype == np.float32
        assert footprint.dtype == np.float32

    def test_footprint_in_0_1(self):
        """Footprint values should be in [0, 1]."""
        hdus = [_make_hdu(180.0, 16.0, nx=32, ny=32),
                _make_hdu(182.0, 16.0, nx=32, ny=32)]
        _, fp = fast_reproject_and_coadd(hdus)
        valid = fp[np.isfinite(fp)]
        assert float(valid.min()) >= 0.0
        assert float(valid.max()) <= 1.0 + 1e-5

    def test_coverage_nonzero(self):
        """At least some pixels should have footprint > 0."""
        hdus = [_make_hdu(180.0, 16.0, nx=32, ny=32)]
        _, fp = fast_reproject_and_coadd(hdus)
        assert (fp > 0).any()

    def test_supplied_wcs_and_shape(self):
        """Caller can supply output WCS and shape explicitly."""
        hdu = _make_hdu(180.0, 16.0, nx=32, ny=32)
        # Build a known output WCS
        out_wcs = WCS(naxis=2)
        out_wcs.wcs.crpix = [32, 32]
        out_wcs.wcs.crval = [180.0, 16.0]
        out_wcs.wcs.cdelt = [-1.0 / 60, 1.0 / 60]
        out_wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
        mosaic, fp = fast_reproject_and_coadd(
            [hdu], output_wcs=out_wcs, output_shape=(64, 64)
        )
        assert mosaic.shape == (64, 64)
        assert fp.shape == (64, 64)

    def test_supplied_wcs_without_shape_raises(self):
        hdu = _make_hdu(180.0, 16.0)
        out_wcs = WCS(naxis=2)
        out_wcs.wcs.crpix = [32, 32]
        out_wcs.wcs.crval = [180.0, 16.0]
        out_wcs.wcs.cdelt = [-1.0 / 60, 1.0 / 60]
        out_wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
        with pytest.raises(ValueError, match="output_shape"):
            fast_reproject_and_coadd([hdu], output_wcs=out_wcs)

    def test_bad_reproject_function_raises(self):
        hdus = [_make_hdu(180.0, 16.0, nx=32, ny=32)]
        with pytest.raises(ValueError, match="Unknown reproject_function"):
            fast_reproject_and_coadd(hdus, reproject_function="gibberish")

    def test_wrap_tiles_produce_compact_mosaic(self):
        """Tiles at 355° and 5° should mosaic into a compact result, not 350°-wide."""
        # Use very small tiles (16×16) for speed
        hdus = [_make_hdu(355.0, 16.0, nx=16, ny=16, cdelt_deg=0.1),
                _make_hdu(5.0, 16.0, nx=16, ny=16, cdelt_deg=0.1)]
        mosaic, fp = fast_reproject_and_coadd(hdus)
        # At 0.1 deg/pixel, 10° span = 100 pixels.  350° span = 3500 pixels.
        ny, nx = mosaic.shape
        assert nx < 500, (
            f"RA-wrap mosaic is {nx} pixels wide — expected <500 for ~10° span"
        )

    def test_uniform_sky_mean_value(self):
        """Three tiles all at value=2.0 should produce mosaic ≈ 2.0 in covered pixels."""
        hdus = [
            _make_hdu(178.0, 16.0, nx=24, ny=24, value=2.0, cdelt_deg=0.05),
            _make_hdu(180.0, 16.0, nx=24, ny=24, value=2.0, cdelt_deg=0.05),
            _make_hdu(182.0, 16.0, nx=24, ny=24, value=2.0, cdelt_deg=0.05),
        ]
        mosaic, fp = fast_reproject_and_coadd(hdus)
        covered = mosaic[fp > 0.5]
        if covered.size > 0:
            np.testing.assert_allclose(
                float(np.nanmedian(covered)), 2.0, atol=0.2,
                err_msg="Mean value of covered pixels should be ≈ 2.0"
            )
