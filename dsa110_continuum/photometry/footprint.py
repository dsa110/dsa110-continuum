"""Mosaic loading and sky-footprint filtering utilities."""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.stats import mad_std
from astropy.wcs import WCS

log = logging.getLogger(__name__)


def load_mosaic(fits_path: str | Path) -> tuple[np.ndarray, WCS, float, np.ndarray]:
    """Load a FITS mosaic and return (data, wcs, rms, valid_mask).

    Parameters
    ----------
    fits_path : str or Path
        Path to the FITS file to load.

    Returns
    -------
    data : np.ndarray, shape (ny, nx), float64
        Pixel values in Jy/beam.
    wcs : astropy.wcs.WCS
        Celestial 2-D WCS extracted from the FITS header.
    rms : float
        Global MAD-RMS in Jy/beam (NaN if no finite pixels).
    valid_mask : np.ndarray, bool, shape (ny, nx)
        True where pixels are finite (not NaN / primary-beam-blanked).
    """
    p = Path(fits_path)
    with fits.open(p) as hdul:
        data = hdul[0].data.squeeze().astype(np.float64)
        hdr  = hdul[0].header
    wcs = WCS(hdr).celestial
    valid_mask = np.isfinite(data)
    finite = data[valid_mask]
    if finite.size == 0:
        rms = float("nan")
        log.warning("load_mosaic %s: no finite pixels — RMS is NaN", p.name)
    else:
        rms = float(mad_std(finite))
    log.info(
        "Loaded %s: %dx%d px, RMS=%.4f Jy/beam, valid=%.1f%%",
        p.name, data.shape[1], data.shape[0], rms, 100 * valid_mask.mean(),
    )
    return data, wcs, rms, valid_mask


def sources_in_footprint(
    ra_arr: np.ndarray,
    dec_arr: np.ndarray,
    wcs: WCS,
    valid_mask: np.ndarray,
) -> np.ndarray:
    """Return a boolean array: True where the sky position falls on a valid (non-NaN) pixel.

    Parameters
    ----------
    ra_arr : np.ndarray
        Right ascension of each source in degrees.
    dec_arr : np.ndarray
        Declination of each source in degrees.
    wcs : astropy.wcs.WCS
        Celestial 2-D WCS used to project sky coordinates to pixel coordinates.
    valid_mask : np.ndarray, bool, shape (ny, nx)
        True where the mosaic pixel is finite and within the primary-beam footprint.

    Returns
    -------
    np.ndarray, bool, same length as ra_arr
        True for each source whose projected pixel position lies within the image
        bounds and on a valid (non-NaN) pixel.
    """
    ny, nx = valid_mask.shape
    sky = np.column_stack([ra_arr, dec_arr])
    pix = wcs.all_world2pix(sky, 0)   # shape (N, 2)
    xi  = np.round(pix[:, 0]).astype(int)
    yi  = np.round(pix[:, 1]).astype(int)

    in_bounds = (xi >= 0) & (xi < nx) & (yi >= 0) & (yi < ny)
    result = np.zeros(len(ra_arr), dtype=bool)
    idx = np.where(in_bounds)[0]
    result[idx] = valid_mask[yi[idx], xi[idx]]
    return result
