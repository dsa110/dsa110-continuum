"""
Image-level quality assessment metrics.

Provides QA metrics computed from image data and noise maps,
adopted from VAST Pipeline methodology for radio transient surveys.

Reference: askap-vast/vast-pipeline image/utils.py
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

logger = logging.getLogger(__name__)


@dataclass
class ImageRMSMetrics:
    """RMS statistics from an image's noise map.

    Attributes
    ----------
    rms_median : float
        Median RMS value (Jy/beam)
    rms_min : float
        Minimum RMS value (Jy/beam)
    rms_max : float
        Maximum RMS value (Jy/beam)
    rms_mean : float
        Mean RMS value (Jy/beam)
    rms_std : float
        Standard deviation of RMS values (Jy/beam)
    n_valid_pixels : int
        Number of valid (non-NaN) pixels in noise map
    coverage_fraction : float
        Fraction of image with valid RMS values
    """

    rms_median: float
    rms_min: float
    rms_max: float
    rms_mean: float
    rms_std: float
    n_valid_pixels: int
    coverage_fraction: float

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "rms_median": self.rms_median,
            "rms_min": self.rms_min,
            "rms_max": self.rms_max,
            "rms_mean": self.rms_mean,
            "rms_std": self.rms_std,
            "n_valid_pixels": self.n_valid_pixels,
            "coverage_fraction": self.coverage_fraction,
        }


@dataclass
class ImageQAMetrics:
    """Comprehensive image quality metrics.

    Attributes
    ----------
    image_path : str or Path
        Path to the image FITS file
    rms : ImageRMSMetrics
        RMS statistics from noise map
    dynamic_range : float
        Peak flux divided by median RMS
    peak_flux : float
        Maximum flux in image (Jy/beam)
    theoretical_rms : float or None
        Expected thermal noise (if calculable)
    beam_major_arcsec : float
        Beam major axis in arcseconds
    beam_minor_arcsec : float
        Beam minor axis in arcseconds
    beam_pa_deg : float
        Beam position angle in degrees
    freq_hz : float
        Central frequency in Hz
    n_channels : int
        Number of frequency channels
    warnings : list of str
        List of QA warnings
    """

    image_path: str
    rms: ImageRMSMetrics | None = None
    dynamic_range: float | None = None
    peak_flux: float | None = None
    theoretical_rms: float | None = None
    beam_major_arcsec: float | None = None
    beam_minor_arcsec: float | None = None
    beam_pa_deg: float | None = None
    freq_hz: float | None = None
    n_channels: int = 1
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "image_path": self.image_path,
            "rms": self.rms.to_dict() if self.rms else None,
            "dynamic_range": self.dynamic_range,
            "peak_flux": self.peak_flux,
            "theoretical_rms": self.theoretical_rms,
            "beam_major_arcsec": self.beam_major_arcsec,
            "beam_minor_arcsec": self.beam_minor_arcsec,
            "beam_pa_deg": self.beam_pa_deg,
            "freq_hz": self.freq_hz,
            "n_channels": self.n_channels,
            "warnings": self.warnings,
        }

    @property
    def passed(self) -> bool:
        """Check if image passes basic QA checks."""
        if self.rms is None:
            return False
        if self.dynamic_range is not None and self.dynamic_range < 10:
            return False
        return True


def get_rms_noise_image_values(
    noise_map: np.ndarray,
    mask: np.ndarray | None = None,
) -> ImageRMSMetrics:
    """Extract RMS statistics from a noise map.

        Adopted from VAST Pipeline: image/utils.py::get_rms_noise_image_values()

    Parameters
    ----------
    noise_map : np.ndarray
        2D array of RMS values (noise map from BANE or similar)
    mask : np.ndarray of bool, optional
        Optional boolean mask where True indicates valid pixels to include

    Returns
    -------
        ImageRMSMetrics
        Object containing RMS statistics

    Raises
    ------
        ValueError
        If no valid pixels in noise map
    """
    # Squeeze to 2D if needed (handle FITS 4D arrays)
    data = np.asarray(noise_map).squeeze()

    if mask is not None:
        mask = np.asarray(mask).squeeze()
        if mask.shape != data.shape:
            raise ValueError(f"Mask shape {mask.shape} != data shape {data.shape}")
        valid_mask = mask & np.isfinite(data) & (data > 0)
    else:
        valid_mask = np.isfinite(data) & (data > 0)

    n_valid = int(np.sum(valid_mask))
    total_pixels = data.size

    if n_valid == 0:
        raise ValueError("No valid pixels in noise map")

    valid_data = data[valid_mask]

    return ImageRMSMetrics(
        rms_median=float(np.median(valid_data)),
        rms_min=float(np.min(valid_data)),
        rms_max=float(np.max(valid_data)),
        rms_mean=float(np.mean(valid_data)),
        rms_std=float(np.std(valid_data)),
        n_valid_pixels=n_valid,
        coverage_fraction=n_valid / total_pixels if total_pixels > 0 else 0.0,
    )


def compute_image_qa_metrics(
    image_path: str,
    noise_map_path: str | None = None,
    background_path: str | None = None,
) -> ImageQAMetrics:
    """Compute comprehensive QA metrics for an image.

    Parameters
    ----------
    image_path : str or Path
        Path to the science image FITS file
    noise_map_path : str or None, optional
        Optional path to noise/RMS map (_rms.fits from BANE)
    background_path : str or None, optional
        Optional path to background map (_bkg.fits from BANE)

    Returns
    -------
        ImageQAMetrics
        Object containing all computed metrics
    """
    metrics = ImageQAMetrics(image_path=image_path)

    image_file = Path(image_path)
    if not image_file.exists():
        metrics.warnings.append(f"Image file not found: {image_path}")
        return metrics

    try:
        with fits.open(image_path) as hdul:
            header = hdul[0].header
            data = np.asarray(hdul[0].data).squeeze()

            # Extract beam parameters
            if "BMAJ" in header:
                metrics.beam_major_arcsec = header["BMAJ"] * 3600
            if "BMIN" in header:
                metrics.beam_minor_arcsec = header["BMIN"] * 3600
            if "BPA" in header:
                metrics.beam_pa_deg = header["BPA"]

            # Extract frequency
            if "CRVAL3" in header:
                metrics.freq_hz = header["CRVAL3"]
            elif "RESTFREQ" in header:
                metrics.freq_hz = header["RESTFREQ"]

            # Peak flux
            valid_data = data[np.isfinite(data)]
            if len(valid_data) > 0:
                metrics.peak_flux = float(np.max(np.abs(valid_data)))

    except (OSError, KeyError) as e:
        metrics.warnings.append(f"Error reading image: {e}")
        return metrics

    # Process noise map if provided
    if noise_map_path:
        noise_file = Path(noise_map_path)
        if noise_file.exists():
            try:
                with fits.open(noise_map_path) as hdul:
                    noise_data = np.asarray(hdul[0].data).squeeze()
                    metrics.rms = get_rms_noise_image_values(noise_data)

                    # Calculate dynamic range
                    if metrics.peak_flux and metrics.rms.rms_median > 0:
                        metrics.dynamic_range = metrics.peak_flux / metrics.rms.rms_median

            except (OSError, ValueError) as e:
                metrics.warnings.append(f"Error reading noise map: {e}")
        else:
            metrics.warnings.append(f"Noise map not found: {noise_map_path}")
    else:
        # Try to find noise map automatically (BANE naming convention)
        auto_rms_path = str(image_file).replace(".fits", "_rms.fits")
        if Path(auto_rms_path).exists():
            try:
                with fits.open(auto_rms_path) as hdul:
                    noise_data = np.asarray(hdul[0].data).squeeze()
                    metrics.rms = get_rms_noise_image_values(noise_data)

                    if metrics.peak_flux and metrics.rms.rms_median > 0:
                        metrics.dynamic_range = metrics.peak_flux / metrics.rms.rms_median
            except (OSError, ValueError) as e:
                metrics.warnings.append(f"Error reading auto-detected noise map: {e}")
        else:
            # Estimate RMS from image itself using sigma-clipping
            try:
                from astropy.stats import sigma_clipped_stats

                _, median_bg, std_bg = sigma_clipped_stats(valid_data, sigma=3.0, maxiters=5)
                metrics.rms = ImageRMSMetrics(
                    rms_median=float(std_bg),
                    rms_min=float(std_bg * 0.8),  # Estimate
                    rms_max=float(std_bg * 1.5),  # Estimate
                    rms_mean=float(std_bg),
                    rms_std=0.0,  # Unknown from single estimate
                    n_valid_pixels=len(valid_data),
                    coverage_fraction=len(valid_data) / data.size,
                )
                metrics.warnings.append("RMS estimated from sigma-clipping (no noise map)")

                if metrics.peak_flux and metrics.rms.rms_median > 0:
                    metrics.dynamic_range = metrics.peak_flux / metrics.rms.rms_median

            except ImportError:
                metrics.warnings.append("Could not estimate RMS (astropy.stats unavailable)")

    return metrics


def get_local_rms_at_position(
    noise_map: np.ndarray,
    wcs: WCS,
    ra_deg: float,
    dec_deg: float,
    aperture_pixels: int = 5,
) -> float:
    """Get local RMS at a specific sky position.

        Adopted from VAST Pipeline methodology for source-level RMS.

    Parameters
    ----------
    noise_map : array_like
        2D noise/RMS map array.
    wcs : WCS
        WCS object for coordinate transformation.
    ra_deg : float
        Right ascension (degrees).
    dec_deg : float
        Declination (degrees).
    aperture_pixels : int, optional
        Size of aperture to average (default 5 pixels).

    Returns
    -------
        float
        Local RMS value at position (same units as noise_map).

    Raises
    ------
        ValueError
        If position is outside image or no valid pixels.
    """
    # Convert to pixel coordinates
    try:
        xy = wcs.world_to_pixel_values(ra_deg, dec_deg)
        x_pix, y_pix = int(round(xy[0])), int(round(xy[1]))
    except Exception as e:
        raise ValueError(f"Could not convert coordinates: {e}")

    # Squeeze noise map to 2D
    data = np.asarray(noise_map).squeeze()
    ny, nx = data.shape

    # Check bounds
    if x_pix < 0 or x_pix >= nx or y_pix < 0 or y_pix >= ny:
        raise ValueError(f"Position ({ra_deg}, {dec_deg}) outside image bounds")

    # Extract aperture
    half = aperture_pixels // 2
    x1 = max(0, x_pix - half)
    x2 = min(nx, x_pix + half + 1)
    y1 = max(0, y_pix - half)
    y2 = min(ny, y_pix + half + 1)

    aperture = data[y1:y2, x1:x2]
    valid = aperture[np.isfinite(aperture) & (aperture > 0)]

    if len(valid) == 0:
        # Fall back to single pixel
        val = data[y_pix, x_pix]
        if np.isfinite(val) and val > 0:
            return float(val)
        raise ValueError(f"No valid RMS data at position ({ra_deg}, {dec_deg})")

    return float(np.median(valid))


def check_edge_proximity(
    wcs: WCS,
    ra_deg: float,
    dec_deg: float,
    image_shape: tuple[int, int],
    beam_major_deg: float,
    n_beam_buffer: float = 3.0,
) -> tuple[bool, float]:
    """Check if source is too close to image edge.

        Adopted from VAST Pipeline new source validation.

    Parameters
    ----------
    wcs : WCS
        WCS object for coordinate transformation.
    ra_deg : float
        Right ascension (degrees).
    dec_deg : float
        Declination (degrees).
    image_shape : tuple of int
        Shape of image (ny, nx).
    beam_major_deg : float
        Beam major axis in degrees.
    n_beam_buffer : int, optional
        Number of beams to use as edge buffer (default 3).

    Returns
    -------
        tuple
        Tuple of (is_near_edge, distance_to_edge_beams).
    """
    try:
        xy = wcs.world_to_pixel_values(ra_deg, dec_deg)
        x_pix, y_pix = float(xy[0]), float(xy[1])
    except Exception:
        return True, 0.0

    ny, nx = image_shape

    # Calculate pixel scale (approximate)
    try:
        # Get pixel scale from WCS
        pixel_scale_deg = abs(wcs.wcs.cdelt[0]) if hasattr(wcs.wcs, "cdelt") else 0.001
    except (AttributeError, IndexError):
        pixel_scale_deg = 0.001  # Default ~3.6 arcsec

    # Buffer in pixels
    buffer_pix = n_beam_buffer * beam_major_deg / pixel_scale_deg

    # Distance to nearest edge
    dist_left = x_pix
    dist_right = nx - x_pix
    dist_bottom = y_pix
    dist_top = ny - y_pix

    min_dist_pix = min(dist_left, dist_right, dist_bottom, dist_top)
    min_dist_beams = min_dist_pix * pixel_scale_deg / beam_major_deg

    is_near_edge = min_dist_pix < buffer_pix

    return is_near_edge, min_dist_beams


def check_nan_proximity(
    data: np.ndarray,
    wcs: WCS,
    ra_deg: float,
    dec_deg: float,
    check_radius_pix: int = 10,
) -> tuple[bool, float]:
    """Check proximity to NaN/masked regions.

        Important for validating new source detections near image edges
        or blanked regions.

    Parameters
    ----------
    data : array_like
        2D image data array.
    wcs : WCS
        WCS object for coordinate transformation.
    ra_deg : float
        Right ascension (degrees).
    dec_deg : float
        Declination (degrees).
    check_radius_pix : int
        Radius to check for NaN pixels.

    Returns
    -------
        tuple
        Tuple of (has_nearby_nan, fraction_nan_in_radius).
    """
    try:
        xy = wcs.world_to_pixel_values(ra_deg, dec_deg)
        x_pix, y_pix = int(round(xy[0])), int(round(xy[1]))
    except Exception:
        return True, 1.0

    # Squeeze data to 2D
    arr = np.asarray(data).squeeze()
    ny, nx = arr.shape

    # Extract region
    x1 = max(0, x_pix - check_radius_pix)
    x2 = min(nx, x_pix + check_radius_pix + 1)
    y1 = max(0, y_pix - check_radius_pix)
    y2 = min(ny, y_pix + check_radius_pix + 1)

    region = arr[y1:y2, x1:x2]
    n_total = region.size
    n_nan = np.sum(~np.isfinite(region))

    if n_total == 0:
        return True, 1.0

    nan_fraction = n_nan / n_total
    has_nearby_nan = nan_fraction > 0.1  # >10% NaN is concerning

    return has_nearby_nan, nan_fraction
