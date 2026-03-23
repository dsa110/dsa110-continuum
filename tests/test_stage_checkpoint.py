"""Tests for applycal sentinel and FITS validation (mosaic_day.py).

All tests use tmp_path and mocks — no real CASA/WSClean calls.
"""

import os
import sys
from pathlib import Path
from unittest import mock

import numpy as np
from astropy.io import fits

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from mosaic_day import TileConfig, _applycal_sentinel_path, _fits_is_valid

# ── Sentinel path helpers ────────────────────────────────────────────────────

class TestApplycalSentinelPath:
    def test_format(self):
        assert _applycal_sentinel_path("/data/ms/tile_meridian.ms") == (
            "/data/ms/tile_meridian.ms.applycal_done"
        )

    def test_strips_trailing_slash(self):
        assert _applycal_sentinel_path("/data/ms/tile_meridian.ms/") == (
            "/data/ms/tile_meridian.ms.applycal_done"
        )


# ── FITS validation ──────────────────────────────────────────────────────────

class TestFitsIsValid:
    def test_good_file(self, tmp_path):
        p = str(tmp_path / "good.fits")
        hdu = fits.PrimaryHDU(np.zeros((10, 10), dtype=np.float32))
        hdu.writeto(p, overwrite=True)
        assert _fits_is_valid(p) is True

    def test_missing_file(self):
        assert _fits_is_valid("/nonexistent/path/tile.fits") is False

    def test_truncated_file(self, tmp_path):
        p = str(tmp_path / "truncated.fits")
        with open(p, "wb") as f:
            f.write(b"\x00" * 512)  # random bytes, not valid FITS
        assert _fits_is_valid(p) is False

    def test_empty_data(self, tmp_path):
        p = str(tmp_path / "nodata.fits")
        hdu = fits.PrimaryHDU()  # no data array
        hdu.writeto(p, overwrite=True)
        assert _fits_is_valid(p) is False


# ── Integration tests (mocked process_ms) ────────────────────────────────────

def _make_cfg(tmp_path):
    """Build a TileConfig pointing at tmp_path directories."""
    ms_dir = str(tmp_path / "ms")
    img_dir = str(tmp_path / "images")
    os.makedirs(ms_dir, exist_ok=True)
    os.makedirs(img_dir, exist_ok=True)
    # Create fake cal tables so process_ms doesn't abort early
    bp = os.path.join(ms_dir, "2026-01-25T22:26:05_0~23.b")
    ga = os.path.join(ms_dir, "2026-01-25T22:26:05_0~23.g")
    os.makedirs(bp, exist_ok=True)
    os.makedirs(ga, exist_ok=True)
    return TileConfig(
        date="2026-01-25",
        ms_dir=ms_dir,
        image_dir=img_dir,
        mosaic_out=os.path.join(img_dir, "full_mosaic.fits"),
        products_dir=str(tmp_path / "products"),
        bp_table=bp,
        g_table=ga,
    )


def _make_valid_ms(path):
    """Create a directory that looks like a valid MS to _ms_is_valid()."""
    os.makedirs(path, exist_ok=True)
    Path(os.path.join(path, "table.dat")).touch()
    Path(os.path.join(path, "table.f0")).touch()


def _write_valid_fits(path):
    """Write a minimal valid FITS file."""
    hdu = fits.PrimaryHDU(np.zeros((10, 10), dtype=np.float32))
    hdu.writeto(path, overwrite=True)


class TestProcessMsSentinel:
    """Test sentinel creation/checking in process_ms()."""

    @mock.patch("mosaic_day.image_ms")
    @mock.patch("mosaic_day.apply_to_target")
    @mock.patch("mosaic_day.phaseshift_ms")
    @mock.patch("dsa110_continuum.validation.image_validator.validate_image_quality", return_value=(True, []))
    def test_writes_sentinel_after_applycal(
        self, _mock_qa, mock_phaseshift, mock_applycal, mock_image, tmp_path
    ):
        from mosaic_day import process_ms

        cfg = _make_cfg(tmp_path)
        ms_path = os.path.join(cfg.ms_dir, "2026-01-25T21:17:33.ms")
        _make_valid_ms(ms_path)
        meridian = ms_path.replace(".ms", "_meridian.ms")
        sentinel = _applycal_sentinel_path(meridian)

        # phaseshift_ms creates the meridian MS directory
        def fake_phaseshift(**kwargs):
            _make_valid_ms(kwargs["output_ms"])
        mock_phaseshift.side_effect = fake_phaseshift

        # image_ms creates a valid FITS output
        def fake_image(**kwargs):
            _write_valid_fits(kwargs["imagename"] + "-image.fits")
        mock_image.side_effect = fake_image

        result = process_ms(ms_path, cfg, keep_intermediates=True)
        assert result.ok
        assert os.path.isfile(sentinel), "Sentinel should exist after successful applycal"

    @mock.patch("mosaic_day.image_ms")
    @mock.patch("mosaic_day.apply_to_target")
    @mock.patch("mosaic_day.phaseshift_ms")
    @mock.patch("dsa110_continuum.validation.image_validator.validate_image_quality", return_value=(True, []))
    def test_skips_applycal_when_sentinel_exists(
        self, _mock_qa, mock_phaseshift, mock_applycal, mock_image, tmp_path
    ):
        from mosaic_day import process_ms

        cfg = _make_cfg(tmp_path)
        ms_path = os.path.join(cfg.ms_dir, "2026-01-25T21:17:33.ms")
        _make_valid_ms(ms_path)
        meridian = ms_path.replace(".ms", "_meridian.ms")
        _make_valid_ms(meridian)

        # Pre-create sentinel
        sentinel = _applycal_sentinel_path(meridian)
        Path(sentinel).write_text("applycal completed\n")

        # image_ms creates a valid FITS output
        def fake_image(**kwargs):
            _write_valid_fits(kwargs["imagename"] + "-image.fits")
        mock_image.side_effect = fake_image

        result = process_ms(ms_path, cfg, keep_intermediates=True)
        assert result.ok
        mock_applycal.assert_not_called()

    @mock.patch("mosaic_day.image_ms")
    @mock.patch("mosaic_day.apply_to_target")
    @mock.patch("mosaic_day.phaseshift_ms")
    @mock.patch("dsa110_continuum.validation.image_validator.validate_image_quality", return_value=(True, []))
    def test_removes_partial_fits(
        self, _mock_qa, mock_phaseshift, mock_applycal, mock_image, tmp_path
    ):
        from mosaic_day import process_ms

        cfg = _make_cfg(tmp_path)
        ms_path = os.path.join(cfg.ms_dir, "2026-01-25T21:17:33.ms")
        _make_valid_ms(ms_path)
        meridian = ms_path.replace(".ms", "_meridian.ms")
        _make_valid_ms(meridian)

        # Pre-create sentinel so applycal is skipped
        sentinel = _applycal_sentinel_path(meridian)
        Path(sentinel).write_text("applycal completed\n")

        # Place an invalid (truncated) FITS at the output path
        tag = Path(ms_path).stem
        bad_fits = os.path.join(cfg.image_dir, tag + "-image.fits")
        with open(bad_fits, "wb") as f:
            f.write(b"\x00" * 512)

        # image_ms should be called because the invalid FITS was removed
        def fake_image(**kwargs):
            _write_valid_fits(kwargs["imagename"] + "-image.fits")
        mock_image.side_effect = fake_image

        result = process_ms(ms_path, cfg, keep_intermediates=True)
        assert result.ok
        mock_image.assert_called_once()

    @mock.patch("mosaic_day.image_ms")
    @mock.patch("mosaic_day.apply_to_target", side_effect=RuntimeError("CASA crash"))
    @mock.patch("mosaic_day.phaseshift_ms")
    def test_no_sentinel_after_applycal_failure(
        self, mock_phaseshift, mock_applycal, mock_image, tmp_path
    ):
        from mosaic_day import process_ms

        cfg = _make_cfg(tmp_path)
        ms_path = os.path.join(cfg.ms_dir, "2026-01-25T21:17:33.ms")
        _make_valid_ms(ms_path)
        meridian = ms_path.replace(".ms", "_meridian.ms")

        def fake_phaseshift(**kwargs):
            _make_valid_ms(kwargs["output_ms"])
        mock_phaseshift.side_effect = fake_phaseshift

        result = process_ms(ms_path, cfg, keep_intermediates=True)
        assert not result.ok
        assert result.failed_stage == "applycal"
        sentinel = _applycal_sentinel_path(meridian)
        assert not os.path.isfile(sentinel), "Sentinel must NOT exist after failed applycal"

    @mock.patch("mosaic_day.image_ms")
    @mock.patch("mosaic_day.apply_to_target")
    @mock.patch("mosaic_day.phaseshift_ms")
    @mock.patch("dsa110_continuum.validation.image_validator.validate_image_quality", return_value=(True, []))
    def test_cleanup_removes_sentinel(
        self, _mock_qa, mock_phaseshift, mock_applycal, mock_image, tmp_path
    ):
        from mosaic_day import process_ms

        cfg = _make_cfg(tmp_path)
        ms_path = os.path.join(cfg.ms_dir, "2026-01-25T21:17:33.ms")
        _make_valid_ms(ms_path)
        meridian = ms_path.replace(".ms", "_meridian.ms")

        def fake_phaseshift(**kwargs):
            _make_valid_ms(kwargs["output_ms"])
        mock_phaseshift.side_effect = fake_phaseshift

        def fake_image(**kwargs):
            _write_valid_fits(kwargs["imagename"] + "-image.fits")
        mock_image.side_effect = fake_image

        result = process_ms(ms_path, cfg, keep_intermediates=False)
        assert result.ok
        sentinel = _applycal_sentinel_path(meridian)
        assert not os.path.isfile(sentinel), "Sentinel should be cleaned up with MS"
        assert not os.path.isdir(meridian), "Meridian MS should be cleaned up"
