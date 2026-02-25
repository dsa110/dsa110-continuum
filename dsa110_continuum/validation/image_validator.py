"""Image validation utilities."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def validate_image_structure(image_path: Path) -> tuple[bool, list[str]]:
    """Validate FITS image structure.

    Parameters
    ----------
    image_path : str
        Path to FITS image

    Returns
    -------
        tuple
        Tuple of (is_valid, error_messages)
    """
    errors = []

    if not image_path.exists():
        return False, [f"Image does not exist: {image_path}"]

    if not image_path.is_file():
        return False, [f"Image should be file: {image_path}"]

    try:
        from astropy.io import fits

        with fits.open(image_path) as hdul:
            if len(hdul) < 1:
                errors.append("FITS file has no HDUs")
                return False, errors

            header = hdul[0].header
            data = hdul[0].data

            if data is None:
                errors.append("Primary HDU has no data")

            # Check WCS
            required_wcs = ["CRPIX1", "CRPIX2", "CRVAL1", "CRVAL2"]
            for key in required_wcs:
                if key not in header:
                    errors.append(f"Missing WCS keyword: {key}")

    except Exception as e:
        errors.append(f"Error reading FITS: {e}")

    return len(errors) == 0, errors


def validate_image_quality(
    image_path: Path,
    min_snr: float = 5.0,
    max_flagged_fraction: float = 0.5,
) -> tuple[bool, list[str]]:
    """Validate image quality metrics.

    Parameters
    ----------
    image_path : str
        Path to FITS image
    min_snr : float
        Minimum required SNR
    max_flagged_fraction : float
        Maximum allowed NaN fraction

    Returns
    -------
        tuple
        Tuple of (is_valid, error_messages)
    """
    errors = []

    try:
        from astropy.io import fits

        with fits.open(image_path) as hdul:
            data = np.squeeze(hdul[0].data)

        # Check for all zeros
        if np.allclose(data, 0, atol=1e-10):
            errors.append("Image is all zeros")
            return False, errors

        # Check NaN fraction
        nan_fraction = np.mean(np.isnan(data))
        if nan_fraction > max_flagged_fraction:
            errors.append(f"Too many NaN pixels: {nan_fraction:.1%}")

        # Compute SNR
        valid_data = data[np.isfinite(data)]
        if len(valid_data) == 0:
            errors.append("No valid pixels")
            return False, errors

        peak = np.max(valid_data)
        off_source = valid_data[valid_data < 3 * np.median(valid_data)]
        if len(off_source) > 100:
            noise = np.std(off_source)
        else:
            noise = np.std(valid_data)

        if noise > 0:
            snr = peak / noise
            if snr < min_snr:
                errors.append(f"SNR too low: {snr:.1f} < {min_snr}")
        else:
            errors.append("Cannot compute SNR (zero noise)")

    except Exception as e:
        errors.append(f"Error validating image quality: {e}")

    return len(errors) == 0, errors
