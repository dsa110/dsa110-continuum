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
    min_peak_asymmetry: float = 1.2,
    max_mad_rms_mjy: float = 100.0,
    source_sigma: float = 5.0,
) -> tuple[bool, list[str]]:
    """Validate image quality metrics including source-presence checks.

    In addition to basic SNR and NaN-fraction checks, this function detects
    noise-only images that can arise from failed calibration.  Two checks:

    1. **Peak/trough asymmetry**: Real sources make the positive peak
       substantially larger than the negative trough.  A ratio close to 1.0
       indicates pure noise (or severely miscalibrated data where PB
       correction amplifies noise but no coherent source is present).
       Catches: noise-only images, symmetric calibration failures.

    2. **MAD RMS ceiling**: The median-absolute-deviation RMS must be below
       a maximum threshold.  Catches: severe calibration failures that
       produce asymmetric but extremely noisy images (high RMS with
       artifacts that happen to be one-sided).

    Positive/negative island counts are logged for diagnostics but not
    gated on, because single DSA-110 tiles have noise-dominated island
    counts at any sigma threshold.

    Parameters
    ----------
    image_path :
        Path to FITS image.
    min_snr :
        Minimum required peak / off-source-noise ratio.
    max_flagged_fraction :
        Maximum allowed NaN fraction.
    min_peak_asymmetry :
        Minimum |peak / trough| ratio.  Images below this are flagged
        as noise-dominated.
    max_mad_rms_mjy :
        Maximum MAD-estimated RMS in mJy.  Images above this are flagged
        as having severe calibration or RFI issues.
    source_sigma :
        Sigma threshold for island counting (diagnostic).
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

        peak = float(np.max(valid_data))
        trough = float(np.min(valid_data))
        off_source = valid_data[valid_data < 3 * np.median(valid_data)]
        if len(off_source) > 100:
            noise = float(np.std(off_source))
        else:
            noise = float(np.std(valid_data))

        if noise > 0:
            snr = peak / noise
            if snr < min_snr:
                errors.append(f"SNR too low: {snr:.1f} < {min_snr}")
        else:
            errors.append("Cannot compute SNR (zero noise)")

        # ── MAD RMS ceiling ────────────────────────────────────────────
        median = float(np.median(valid_data))
        mad_rms_jy = float(1.4826 * np.median(np.abs(valid_data - median)))
        mad_rms_mjy = mad_rms_jy * 1000.0
        if mad_rms_mjy > max_mad_rms_mjy:
            errors.append(
                f"MAD RMS too high: {mad_rms_mjy:.1f} mJy > {max_mad_rms_mjy:.0f} mJy. "
                f"Possible severe calibration failure or RFI."
            )

        # ── Source-presence checks ────────────────────────────────────
        # Guard: only run if there are enough pixels for meaningful stats
        if len(valid_data) > 1000:
            _check_source_presence(
                data, valid_data, peak, trough,
                min_peak_asymmetry, source_sigma,
                errors,
            )

    except Exception as e:
        errors.append(f"Error validating image quality: {e}")

    return len(errors) == 0, errors


def _check_source_presence(
    data: np.ndarray,
    valid_data: np.ndarray,
    peak: float,
    trough: float,
    min_peak_asymmetry: float,
    source_sigma: float,
    errors: list[str],
) -> None:
    """Detect noise-only images from calibration failure.

    Appends to *errors* in-place if the image appears to contain no real
    sources.  Primary check: peak/trough asymmetry.  Island counts are
    logged for diagnostics but not gated on.
    """
    from scipy import ndimage

    # 1) Peak/trough asymmetry
    abs_trough = abs(trough)
    if abs_trough > 0:
        asymmetry = abs(peak) / abs_trough
    else:
        asymmetry = float("inf")  # no negative values → fine

    if asymmetry < min_peak_asymmetry:
        errors.append(
            f"Source-presence: peak/trough asymmetry too low "
            f"({asymmetry:.2f} < {min_peak_asymmetry}). "
            f"Image may be noise-only (peak={peak * 1000:.1f} mJy, "
            f"trough={trough * 1000:.1f} mJy)."
        )

    # 2) Positive/negative island balance
    median = float(np.median(valid_data))
    mad_rms = float(1.4826 * np.median(np.abs(valid_data - median)))
    if mad_rms <= 0:
        return

    pos_threshold = median + source_sigma * mad_rms
    neg_threshold = median - source_sigma * mad_rms
    finite_mask = np.isfinite(data)

    pos_mask = (data > pos_threshold) & finite_mask
    neg_mask = (data < neg_threshold) & finite_mask

    _, n_pos = ndimage.label(pos_mask)
    _, n_neg = ndimage.label(neg_mask)

    logger.debug(
        "Source-presence: %d positive / %d negative %g-sigma islands",
        n_pos, n_neg, source_sigma,
    )

    # Log the island counts for diagnostics; don't gate on them.
    # Single DSA-110 tiles have hundreds of noise islands that dominate
    # real source counts, making the pos/neg ratio ≈ 1 even for valid
    # images.  The asymmetry check above is the reliable discriminator.
    # Island balance may be useful for deeper mosaics in future.
