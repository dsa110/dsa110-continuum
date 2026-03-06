"""Utilities for reading the observed declination from a Measurement Set or FITS tile."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)


def read_ms_dec(ms_path: str, fits_fallback: str | None = None) -> float:
    """Return the median observed declination (degrees) for a Measurement Set.

    Reads FIELD::PHASE_DIR (column shape n_fields × 1 × 2, values in radians).
    If the MS cannot be opened, tries *fits_fallback* CRVAL2.

    Parameters
    ----------
    ms_path:
        Path to the Measurement Set directory.
    fits_fallback:
        Optional path to a tile FITS file whose CRVAL2 will be used if the MS
        cannot be opened.

    Raises
    ------
    RuntimeError
        If neither source yields a valid Dec.
    """
    try:
        import casacore.tables as ct
        with ct.table(ms_path + "/FIELD", readonly=True, ack=False) as t:
            phase_dir = t.getcol("PHASE_DIR")   # (n_fields, 1, 2)
        dec_rad = np.median(phase_dir[:, 0, 1])
        return float(np.degrees(dec_rad))
    except Exception as e:
        log.debug("read_ms_dec: MS read failed (%s), trying FITS fallback", e)

    if fits_fallback is not None:
        try:
            from astropy.io import fits as _fits
            with _fits.open(fits_fallback) as hdul:
                crval2 = hdul[0].header.get("CRVAL2")
            if crval2 is not None:
                return float(crval2)
        except Exception as e2:
            log.debug("read_ms_dec: FITS fallback failed (%s)", e2)

    raise RuntimeError(
        f"Cannot determine Dec for {ms_path!r}: MS unreadable and no valid FITS fallback."
    )
