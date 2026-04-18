"""Two-stage forced photometry: coarse peak-in-box -> Condon fine pass."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from astropy.io import fits
from astropy.stats import mad_std
from astropy.wcs import WCS

from dsa110_continuum.photometry.simple_peak import measure_peak_box


@dataclass
class CoarseAugment:
    """Coarse-pass metadata attached alongside a ForcedPhotometryResult."""
    ra_deg: float
    dec_deg: float
    coarse_peak_jyb: float
    coarse_snr: float
    passed_coarse: bool


def beam_correction_factor(fits_path: str) -> float:
    """Return beam_area_sr / pixel_area_sr from a FITS header.
    Pixel area is |CDELT1| x |CDELT2| in steradians.
    For an unresolved point source: peak_jyb = S_total_Jy / correction_factor.
    So: S_total_Jy approximately peak_jyb * correction_factor.
    Returns 1.0 if beam keywords are absent or zero.
    """
    hdr = fits.getheader(fits_path)
    bmaj_deg = hdr.get("BMAJ")
    bmin_deg = hdr.get("BMIN")
    cdelt1_deg = abs(hdr.get("CDELT1", 0.0))
    cdelt2_deg = abs(hdr.get("CDELT2", 0.0))
    if bmaj_deg is None or bmin_deg is None or bmaj_deg <= 0.0 or bmin_deg <= 0.0 or cdelt1_deg == 0.0 or cdelt2_deg == 0.0:
        return 1.0
    bmaj_rad = math.radians(bmaj_deg)
    bmin_rad = math.radians(bmin_deg)
    beam_area_sr = (math.pi / (4.0 * math.log(2.0))) * bmaj_rad * bmin_rad
    pixel_area_sr = math.radians(cdelt1_deg) * math.radians(cdelt2_deg)
    return beam_area_sr / pixel_area_sr


def run_coarse_pass(
    fits_path: str,
    coords: list[tuple[float, float]],
    *,
    box_pix: int = 5,
    global_rms: float | None = None,
    snr_coarse_min: float = 3.0,
) -> list[CoarseAugment]:
    """Run peak-in-box measurement at each sky position.

    Parameters
    ----------
    fits_path : str
        Path to mosaic FITS file (Jy/beam).
    coords : list of (ra_deg, dec_deg)
        Source positions to measure.
    box_pix : int
        Half-width of the search box in pixels (default 5, giving 11x11 box).
    global_rms : float or None
        Noise estimate in Jy/beam.  If None, estimated from image MAD.
    snr_coarse_min : float
        Sources with coarse_snr >= this value have passed_coarse=True.

    Returns
    -------
    list[CoarseAugment]
        One entry per input coordinate, preserving order.
    """
    with fits.open(fits_path) as hdul:
        data = np.squeeze(np.asarray(hdul[0].data, dtype=float))
        wcs = WCS(hdul[0].header).celestial

    rms = global_rms
    if rms is None:
        finite = data[np.isfinite(data)]
        rms = float(mad_std(finite)) if finite.size > 0 else float("nan")

    results = []
    for ra, dec in coords:
        peak, snr, _, _ = measure_peak_box(data, wcs, ra, dec, box_pix=box_pix, rms=rms)
        if not np.isfinite(snr):
            snr = (peak / rms) if (np.isfinite(peak) and np.isfinite(rms) and rms > 0) else float("nan")
        results.append(CoarseAugment(
            ra_deg=ra,
            dec_deg=dec,
            coarse_peak_jyb=peak,
            coarse_snr=snr,
            passed_coarse=bool(np.isfinite(snr) and snr >= snr_coarse_min),
        ))
    return results
