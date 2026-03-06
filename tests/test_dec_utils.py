import pytest
from unittest.mock import patch, MagicMock
import numpy as np


def test_read_ms_dec_from_ms(tmp_path):
    """read_ms_dec returns Dec from MS FIELD table."""
    from dsa110_continuum.calibration.dec_utils import read_ms_dec

    mock_table = MagicMock()
    mock_table.__enter__ = lambda s: s
    mock_table.__exit__ = MagicMock(return_value=False)
    # PHASE_DIR shape: (n_fields, 1, 2) in radians — RA=0.5 rad, Dec=0.28 rad ≈ 16.04°
    mock_table.getcol.return_value = np.array([[[0.5, 0.28]]])

    with patch("casacore.tables.table", return_value=mock_table):
        dec = read_ms_dec(str(tmp_path / "test.ms"))

    assert abs(dec - np.degrees(0.28)) < 0.01


def test_read_ms_dec_falls_back_to_fits(tmp_path):
    """read_ms_dec falls back to CRVAL2 when MS read fails."""
    from astropy.io import fits
    from dsa110_continuum.calibration.dec_utils import read_ms_dec

    fits_path = tmp_path / "tile.fits"
    hdu = fits.PrimaryHDU()
    hdu.header["CRVAL1"] = 40.0
    hdu.header["CRVAL2"] = 33.0
    hdu.writeto(fits_path)

    dec = read_ms_dec(str(tmp_path / "nonexistent.ms"), fits_fallback=str(fits_path))
    assert abs(dec - 33.0) < 0.01


def test_read_ms_dec_raises_when_both_fail(tmp_path):
    """read_ms_dec raises RuntimeError when MS and FITS fallback both fail."""
    from dsa110_continuum.calibration.dec_utils import read_ms_dec

    with pytest.raises(RuntimeError, match="Cannot determine Dec"):
        read_ms_dec(str(tmp_path / "nonexistent.ms"))
