"""Two-stage forced photometry: coarse peak-in-box → Condon fine pass."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS


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

    For an unresolved point source: peak_jyb = S_total_Jy / correction_factor.
    So: S_total_Jy ≈ peak_jyb * correction_factor.

    Returns 1.0 if beam keywords are absent (no correction applied).
    """
    hdr = fits.getheader(fits_path)
    bmaj_deg = hdr.get("BMAJ")
    bmin_deg = hdr.get("BMIN")
    cdelt2_deg = abs(hdr.get("CDELT2", 0.0))
    if bmaj_deg is None or bmin_deg is None or cdelt2_deg == 0.0:
        return 1.0
    bmaj_rad = math.radians(bmaj_deg)
    bmin_rad = math.radians(bmin_deg)
    pixel_rad = math.radians(cdelt2_deg)
    beam_area_sr = (math.pi / (4.0 * math.log(2.0))) * bmaj_rad * bmin_rad
    pixel_area_sr = pixel_rad ** 2
    return beam_area_sr / pixel_area_sr
