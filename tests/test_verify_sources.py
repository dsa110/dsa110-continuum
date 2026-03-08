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
