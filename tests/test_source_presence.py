"""Tests for source-presence QA checks in image_validator."""

import numpy as np
import pytest


def _write_fits(path, data):
    """Write a minimal FITS file with WCS headers."""
    from astropy.io import fits

    hdr = fits.Header()
    hdr["CRPIX1"] = data.shape[1] / 2
    hdr["CRPIX2"] = data.shape[0] / 2
    hdr["CRVAL1"] = 180.0
    hdr["CRVAL2"] = 0.0
    hdr["CDELT1"] = -0.001
    hdr["CDELT2"] = 0.001
    hdr["CTYPE1"] = "RA---SIN"
    hdr["CTYPE2"] = "DEC--SIN"
    hdu = fits.PrimaryHDU(data=data.astype(np.float32), header=hdr)
    hdu.writeto(str(path), overwrite=True)


@pytest.fixture()
def work_dir():
    import shutil
    import tempfile
    from pathlib import Path

    d = tempfile.mkdtemp(prefix="dsa110_source_pres_")
    yield Path(d)
    shutil.rmtree(d, ignore_errors=True)


class TestSourcePresenceChecks:
    """Verify that noise-only images fail the new source-presence QA."""

    def test_pure_noise_fails_asymmetry(self, work_dir):
        """Symmetric noise (|peak/trough| ≈ 1) is rejected."""
        from dsa110_continuum.validation.image_validator import validate_image_quality

        rng = np.random.default_rng(42)
        data = rng.normal(0, 0.01, (200, 200))  # symmetric noise
        fpath = work_dir / "noise.fits"
        _write_fits(fpath, data)

        ok, errors = validate_image_quality(fpath, min_snr=1.0)
        assert not ok, f"Pure noise should fail, got: {errors}"
        assert any("asymmetry" in e for e in errors)

    def test_large_noise_fails_asymmetry(self, work_dir):
        """Large noise image also caught by asymmetry check."""
        from dsa110_continuum.validation.image_validator import validate_image_quality

        rng = np.random.default_rng(42)
        data = rng.normal(0, 0.01, (500, 500))
        fpath = work_dir / "noise_large.fits"
        _write_fits(fpath, data)

        ok, errors = validate_image_quality(fpath, min_snr=1.0)
        assert not ok
        assert any("asymmetry" in e for e in errors)

    def test_real_source_passes(self, work_dir):
        """Image with a bright source and noise background passes."""
        from dsa110_continuum.validation.image_validator import validate_image_quality

        rng = np.random.default_rng(42)
        data = rng.normal(0, 0.001, (200, 200))
        # Add a bright compact source (30x noise)
        data[100, 100] = 0.03
        data[99:102, 99:102] += 0.02  # extended a bit
        fpath = work_dir / "source.fits"
        _write_fits(fpath, data)

        ok, errors = validate_image_quality(fpath, min_snr=3.0)
        assert ok, f"Image with source should pass, got: {errors}"

    def test_multiple_sources_pass(self, work_dir):
        """Image with several real sources passes island balance."""
        from dsa110_continuum.validation.image_validator import validate_image_quality

        rng = np.random.default_rng(42)
        data = rng.normal(0, 0.001, (300, 300))
        # Add 10 sources at various fluxes
        for y, x, flux in [
            (50, 50, 0.05), (100, 200, 0.03), (150, 100, 0.02),
            (200, 250, 0.04), (250, 50, 0.025), (30, 270, 0.015),
            (180, 180, 0.02), (270, 130, 0.035), (80, 160, 0.018),
            (220, 80, 0.022),
        ]:
            data[y - 1:y + 2, x - 1:x + 2] += flux

        fpath = work_dir / "multi_source.fits"
        _write_fits(fpath, data)

        ok, errors = validate_image_quality(fpath, min_snr=3.0)
        assert ok, f"Multi-source image should pass, got: {errors}"

    def test_pb_amplified_noise_fails(self, work_dir):
        """Noise with PB-like amplification at edges still fails."""
        from dsa110_continuum.validation.image_validator import validate_image_quality

        rng = np.random.default_rng(42)
        data = rng.normal(0, 0.01, (200, 200))
        # Simulate PB correction amplifying edges
        y, x = np.mgrid[0:200, 0:200]
        r = np.sqrt((y - 100) ** 2 + (x - 100) ** 2)
        pb = np.clip(1.0 - (r / 100) ** 2, 0.1, 1.0)
        data = data / pb  # PB correction amplifies edge noise

        fpath = work_dir / "pb_noise.fits"
        _write_fits(fpath, data)

        ok, errors = validate_image_quality(fpath, min_snr=1.0)
        assert not ok, f"PB-amplified noise should fail, got: {errors}"

    def test_known_bad_tile_would_fail(self):
        """The actual 2026-03-16 tile (if present) should fail."""
        from pathlib import Path

        from dsa110_continuum.validation.image_validator import validate_image_quality

        fpath = Path(
            "/stage/dsa110-contimg/images/mosaic_2026-03-16/"
            "2026-03-16T04:15:21-image-pb.fits"
        )
        if not fpath.exists():
            pytest.skip("Test tile not on disk")

        ok, errors = validate_image_quality(fpath, min_snr=3.0)
        assert not ok, f"Known noise-only tile should fail, got: {errors}"
        assert any("asymmetry" in e or "island counts" in e for e in errors)

    def test_known_good_tile_passes(self):
        """The actual 2026-01-25 tile (if present) should pass."""
        from pathlib import Path

        from dsa110_continuum.validation.image_validator import validate_image_quality

        fpath = Path(
            "/stage/dsa110-contimg/images/mosaic_2026-01-25/"
            "2026-01-25T14:42:00-image-pb.fits"
        )
        if not fpath.exists():
            pytest.skip("Test tile not on disk")

        ok, errors = validate_image_quality(fpath, min_snr=3.0)
        assert ok, f"Known good tile should pass, got: {errors}"

    def test_high_rms_fails(self, work_dir):
        """Image with very high RMS is rejected."""
        from dsa110_continuum.validation.image_validator import validate_image_quality

        rng = np.random.default_rng(42)
        # RMS ~ 150 mJy >> 100 mJy threshold
        data = rng.normal(0, 0.15, (200, 200))
        # Add a bright source so asymmetry passes
        data[100, 100] = 2.0
        fpath = work_dir / "high_rms.fits"
        _write_fits(fpath, data)

        ok, errors = validate_image_quality(fpath, min_snr=1.0)
        assert not ok
        assert any("MAD RMS too high" in e for e in errors)

    def test_normal_rms_passes(self, work_dir):
        """Image with RMS well below threshold passes RMS gate."""
        from dsa110_continuum.validation.image_validator import validate_image_quality

        rng = np.random.default_rng(42)
        # RMS ~ 10 mJy, well under 100 mJy
        data = rng.normal(0, 0.01, (200, 200))
        data[100, 100] = 0.5  # bright source for asymmetry
        fpath = work_dir / "normal_rms.fits"
        _write_fits(fpath, data)

        ok, errors = validate_image_quality(fpath, min_snr=1.0)
        assert ok, f"Normal-RMS image should pass, got: {errors}"

    def test_known_bad_high_rms_tile(self):
        """Feb 15 tile with extreme RMS (if present) should fail."""
        from pathlib import Path

        from dsa110_continuum.validation.image_validator import validate_image_quality

        fpath = Path(
            "/stage/dsa110-contimg/images/mosaic_2026-02-15/"
            "2026-02-15T00:49:04-image-pb.fits"
        )
        if not fpath.exists():
            pytest.skip("Test tile not on disk")

        ok, errors = validate_image_quality(fpath, min_snr=3.0)
        assert not ok, f"Known high-RMS tile should fail, got: {errors}"
        assert any("MAD RMS" in e or "asymmetry" in e for e in errors)
