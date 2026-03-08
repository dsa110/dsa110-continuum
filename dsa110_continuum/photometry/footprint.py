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

    valid_mask is True where pixels are finite (not NaN / primary-beam-blanked).
    rms is the global MAD-RMS in Jy/beam.
    """
    p = Path(fits_path)
    with fits.open(p) as hdul:
        data = hdul[0].data.squeeze().astype(np.float64)
        hdr  = hdul[0].header
    wcs = WCS(hdr).celestial
    valid_mask = np.isfinite(data)
    finite = data[valid_mask]
    rms = 1.4826 * float(np.median(np.abs(finite - np.median(finite))))
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
    """Return a boolean array: True where the sky position falls on a valid (non-NaN) pixel."""
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
