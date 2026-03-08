"""Lightweight peak-in-box forced photometry.

No GPU, no Condon convolution. Returns (peak_flux_jyb, snr, x_pix, y_pix).
Used for catalog verification where a reliable median ratio matters more than
precision. For science-quality flux measurements use
`dsa110_continuum.photometry.forced.measure_forced_peak` instead.
"""
from __future__ import annotations

import numpy as np
from astropy.wcs import WCS


def measure_peak_box(
    data: np.ndarray,
    wcs: WCS,
    ra_deg: float,
    dec_deg: float,
    *,
    box_pix: int = 5,
    rms: float | None = None,
) -> tuple[float, float, float, float]:
    """Measure peak flux in a pixel box centred on a sky position.

    Parameters
    ----------
    data : np.ndarray, shape (ny, nx)
        Image data in Jy/beam.
    wcs : astropy.wcs.WCS
        Celestial 2-D WCS for the image.
    ra_deg : float
        Right ascension of target in degrees.
    dec_deg : float
        Declination of target in degrees.
    box_pix : int, optional
        Half-width of the search box in pixels (default 5, giving an
        11×11 pixel box ≈ 66″ at 6″/px).
    rms : float or None, optional
        Global noise estimate in Jy/beam used to compute SNR.
        If None, SNR is returned as NaN.

    Returns
    -------
    peak_flux_jyb : float
        Maximum finite pixel value within the box in Jy/beam.
        NaN if the position is outside the image or the box contains no
        finite pixels.
    snr : float
        peak_flux_jyb / rms.  NaN when rms is None or the position is
        invalid.
    x_pix : float
        Pixel x-coordinate of the catalog position (0-indexed).
        NaN if the position is outside the image.
    y_pix : float
        Pixel y-coordinate of the catalog position (0-indexed).
        NaN if the position is outside the image.
    """
    nan4 = (float("nan"),) * 4
    ny, nx = data.shape
    xy = wcs.all_world2pix([[ra_deg, dec_deg]], 0)[0]
    xi, yi = int(round(float(xy[0]))), int(round(float(xy[1])))
    if not (0 <= xi < nx and 0 <= yi < ny):
        return nan4
    y1, y2 = max(0, yi - box_pix), min(ny, yi + box_pix + 1)
    x1, x2 = max(0, xi - box_pix), min(nx, xi + box_pix + 1)
    box = data[y1:y2, x1:x2]
    valid = box[np.isfinite(box)]
    if valid.size == 0:
        return nan4
    flux = float(np.nanmax(valid))
    snr  = (flux / rms) if (rms is not None and rms > 0) else float("nan")
    return flux, snr, float(xi), float(yi)
