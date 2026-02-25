"""Image quality metrics for DSA-110 continuum imaging pipeline."""

from pathlib import Path

import numpy as np

from dsa110_contimg.common.unified_config import settings
from dsa110_contimg.common.utils.fits_utils import get_2d_data_and_wcs
from dsa110_contimg.common.utils.gpu_utils import get_array_module


def _maybe_to_gpu(array: np.ndarray, *, min_elements: int = 1_000_000):
    """Return (array_on_backend, xp, is_gpu) with a size heuristic."""
    xp, is_gpu = get_array_module(
        prefer_gpu=settings.gpu.prefer_gpu,
        min_elements=settings.gpu.min_array_size,
    )
    if is_gpu and array.size >= min_elements:
        return xp.asarray(array), xp, True
    return array, np, False


def get_center_cutout(data: np.ndarray, size: int = 50) -> np.ndarray:
    """Extract a central square cutout from an image array."""
    y, x = data.shape
    cy, cx = y // 2, x // 2
    r = size // 2
    return data[cy - r : cy + r, cx - r : cx + r]


def calculate_psf_correlation(
    dirty_path: str | Path, psf_path: str | Path, cutout_size: int = 64
) -> float:
    """Calculate Pearson correlation between the central regions of a dirty image and its PSF.

    Parameters
    ----------
    dirty_path : str
        Path to the dirty image FITS file.
    psf_path : str
        Path to the PSF image FITS file.
    cutout_size : int, optional
        Size of the central cutout to analyze (default is 64).

    Returns
    -------
        float
        Pearson correlation coefficient.
    """
    dirty_data, _, _ = get_2d_data_and_wcs(dirty_path)
    psf_data, _, _ = get_2d_data_and_wcs(psf_path)

    source_cutout = get_center_cutout(dirty_data, cutout_size)
    psf_cutout = get_center_cutout(psf_data, cutout_size)

    source_backend, xp, is_gpu = _maybe_to_gpu(
        source_cutout, min_elements=settings.gpu.min_array_size
    )
    psf_backend = xp.asarray(psf_cutout) if is_gpu else psf_cutout

    source_peak = xp.max(xp.abs(source_backend))
    psf_peak = xp.max(xp.abs(psf_backend))
    if source_peak == 0 or psf_peak == 0:
        return float("nan")

    source_norm = source_backend / source_peak
    psf_norm = psf_backend / psf_peak

    s_flat = source_norm.ravel()
    p_flat = psf_norm.ravel()

    s_mean = xp.mean(s_flat)
    p_mean = xp.mean(p_flat)
    s_centered = s_flat - s_mean
    p_centered = p_flat - p_mean

    denom = xp.sqrt(xp.mean(s_centered**2)) * xp.sqrt(xp.mean(p_centered**2))
    if denom == 0:
        return float("nan")

    r_val = xp.mean(s_centered * p_centered) / denom
    to_host = xp.asnumpy if hasattr(xp, "asnumpy") else np.asarray
    return float(to_host(r_val))


def calculate_dynamic_range(
    image_path: str | Path,
    peak_flux: float | None = None,
    rms_region: tuple | None = None,
) -> float:
    """Calculate the dynamic range of an image.

        Dynamic Range = Peak Flux / RMS Noise

    Parameters
    ----------
    image_path : str
        Path to FITS image.
    peak_flux : float or None, optional
        Peak flux in Jy/beam. If None, uses max(abs(image)).
    rms_region : tuple or None, optional
        Tuple (y_start, y_end, x_start, x_end) defining noise region.
        If None, uses corners of image.

    Returns
    -------
        float
        Dynamic range (unitless).
    """
    data, _, _ = get_2d_data_and_wcs(image_path)
    data_backend, xp, is_gpu = _maybe_to_gpu(data, min_elements=settings.gpu.min_array_size)

    if peak_flux is None:
        peak_flux_val = xp.max(xp.abs(data_backend))
    else:
        peak_flux_val = peak_flux

    if rms_region is None:
        # Use corner regions (assume source is in center)
        corner_size = min(data.shape) // 10
        corners = [
            data_backend[:corner_size, :corner_size],
            data_backend[:corner_size, -corner_size:],
            data_backend[-corner_size:, :corner_size],
            data_backend[-corner_size:, -corner_size:],
        ]
        noise_data = xp.concatenate([c.ravel() for c in corners])
    else:
        y0, y1, x0, x1 = rms_region
        noise_data = data_backend[y0:y1, x0:x1].ravel()

    rms = xp.std(noise_data)

    if rms == 0:
        return np.inf

    # Move scalars back to host if needed
    peak_flux_host = float(peak_flux_val) if is_gpu else float(peak_flux_val)
    rms_host = float(rms)
    return peak_flux_host / rms_host


def calculate_residual_stats(residual_path: str | Path) -> dict:
    """Calculate statistics for a CLEAN residual image.

    Parameters
    ----------
    residual_path : str
        Path to residual FITS image.

    Returns
    -------
        dict
        Dictionary with:
        - 'mean': Mean residual (Jy/beam)
        - 'std': Standard deviation (Jy/beam)
        - 'max': Maximum residual (Jy/beam)
        - 'min': Minimum residual (Jy/beam)
        - 'rms': RMS of residual (Jy/beam)
        - 'normality_p': p-value from Shapiro-Wilk test (if data < 5000 pixels)
    """
    from scipy.stats import shapiro

    data, _, _ = get_2d_data_and_wcs(residual_path)
    data_backend, xp, is_gpu = _maybe_to_gpu(data, min_elements=settings.gpu.min_array_size)
    flat = data_backend.ravel()

    stats = {
        "mean": float(xp.mean(flat)),
        "std": float(xp.std(flat)),
        "max": float(xp.max(flat)),
        "min": float(xp.min(flat)),
        "rms": float(xp.sqrt(xp.mean(flat**2))),
    }

    # Shapiro-Wilk test requires CPU arrays
    flat_np = xp.asnumpy(flat) if is_gpu else flat  # type: ignore[attr-defined]
    if len(flat_np) < 5000:
        _, p_value = shapiro(flat_np)
        stats["normality_p"] = float(p_value)
    else:
        sample = np.random.choice(flat_np, size=5000, replace=False)
        _, p_value = shapiro(sample)
        stats["normality_p"] = float(p_value)

    return stats
