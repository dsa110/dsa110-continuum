import numpy as np
import pytest
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
    import numpy as np
    wcs = _make_simple_wcs()
    valid_mask = np.ones((10, 10), dtype=bool)
    result = sources_in_footprint(np.array([]), np.array([]), wcs, valid_mask)
    assert result.shape == (0,)
    assert result.dtype == bool


def test_measure_peak_box_returns_correct_flux():
    from dsa110_continuum.photometry.simple_peak import measure_peak_box
    wcs = _make_simple_wcs()
    data = np.zeros((10, 10))
    # crpix=[5,5] 1-indexed → pixel (4,4) 0-indexed for crval=(40.0, 16.0)
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
