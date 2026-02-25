"""
Adaptive channel binning for DSA-110 photometry.

Implements WABIFAT's adaptive binning algorithm: dynamically combines frequency
channels or time intervals to optimize Signal-to-Noise Ratio (SNR) for source
detection and photometry.

Algorithm:
1. Start with narrow bins (initial_width channels/intervals)
2. Iteratively increase bin width until detection (SNR >= target_snr)
3. Track misfits (non-detections) and try combining adjacent ones
4. Return list of detections with optimal binning

This is particularly useful for weak sources that may not be detected in
individual subbands but become detectable when multiple subbands are combined.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Detection:
    """A detection with optimal binning."""

    channels: list[int]  # List of channel/subband indices
    flux_jy: float
    rms_jy: float
    snr: float
    center_freq_mhz: float | None = None  # Central frequency (MHz)
    bin_width: int = 1  # Number of channels/intervals in this bin


@dataclass
class AdaptiveBinningConfig:
    """Configuration for adaptive binning."""

    target_snr: float = 5.0  # Target SNR threshold
    initial_width: int = 1  # Starting bin width
    max_width: int = 16  # Maximum bin width (for DSA-110: 16 subbands)
    min_width: int = 1  # Minimum bin width
    allow_adjacent_misfits: bool = True  # Try combining adjacent misfits


def _find_consecutive_series(channels: list[int]) -> list[list[int]]:
    """Find consecutive series of channel indices.

    Parameters
    ----------
    channels : list
        List of channel indices (may not be consecutive)

    Returns
    -------
        list
        List of consecutive series

    Examples
    --------
        >>> _find_consecutive_series([0, 1, 2, 5, 6, 8])
        [[0, 1, 2], [5, 6], [8]]
    """
    if not channels:
        return []

    sorted_channels = sorted(set(channels))
    if not sorted_channels:
        return []

    series = []
    current_series = [sorted_channels[0]]

    for i in range(1, len(sorted_channels)):
        if sorted_channels[i] == sorted_channels[i - 1] + 1:
            # Consecutive
            current_series.append(sorted_channels[i])
        else:
            # Gap - start new series
            series.append(current_series)
            current_series = [sorted_channels[i]]

    series.append(current_series)
    return series


def _split_into_slices(
    channels: list[int],
    slice_width: int,
) -> list[list[int]]:
    """Split consecutive channels into slices of specified width.

    Parameters
    ----------
    channels : list
        Consecutive channel indices
    slice_width : int
        Width of each slice

    Returns
    -------
        list
        List of slices

    Examples
    --------
        >>> _split_into_slices([0, 1, 2, 3, 4, 5], 2)
        [[0, 1], [2, 3], [4, 5]]
        >>> _split_into_slices([0, 1, 2, 3, 4], 2)
        [[0, 1], [2, 3], [4]]
    """
    slices = []
    for i in range(0, len(channels), slice_width):
        slices.append(channels[i : i + slice_width])
    return slices


def _try_adjacent_misfits(
    misfit_channels: list[int],
    measure_fn: Callable[[list[int]], tuple[float, float, float]],
    target_snr: float,
) -> list[Detection]:
    """Try combining adjacent misfit channels.

    Parameters
    ----------
    misfit_channels : list
        List of channel indices that didn't achieve target SNR
    measure_fn : callable
        Function to measure flux, RMS, SNR for given channels
    target_snr : float
        Target SNR threshold

    Returns
    -------
        list
        List of detections from combined misfits
    """
    if not misfit_channels:
        return []

    detections = []
    consecutive_series = _find_consecutive_series(misfit_channels)

    for series in consecutive_series:
        if len(series) < 2:
            continue  # Need at least 2 channels to combine

        # Try combining all channels in series
        try:
            flux, rms, snr = measure_fn(series)
            if snr >= target_snr:
                detections.append(
                    Detection(
                        channels=series,
                        flux_jy=flux,
                        rms_jy=rms,
                        snr=snr,
                        bin_width=len(series),
                    )
                )
        except (ValueError, RuntimeError):
            # Skip if measurement fails
            continue

    return detections


def adaptive_bin_channels(
    n_channels: int,
    measure_fn: Callable[[list[int]], tuple[float, float, float]],
    config: AdaptiveBinningConfig | None = None,
    *,
    channel_freqs_mhz: list[float] | None = None,
) -> list[Detection]:
    """Adaptive channel binning following WABIFAT algorithm.

        This function implements the core adaptive binning logic:
        1. Start with all channels available
        2. For each bin width (initial_width to max_width):
        a. Find consecutive series of remaining channels
        b. Split each series into slices of current width
        c. Measure each slice
        d. If SNR >= target_snr, record as detection
        e. Otherwise, add channels back to pool for next iteration
        3. Final pass: try combining adjacent misfits

    Parameters
    ----------
    n_channels : int
        Total number of channels/subbands
    measure_fn : callable
        Function that takes a list of channel indices and returns
        (flux_jy, rms_jy, snr). Should raise exception on failure.
    config : object, optional
        Configuration (uses defaults if None)
    channel_freqs_mhz : list, optional
        Optional list of central frequencies for each channel

    Returns
    -------
        list
        List of Detection objects with optimal binning

    Examples
    --------
        >>> def measure(channels):
        ...     # Simulate measurement: combine channels, measure flux
        ...     combined_flux = sum([flux_per_channel[i] for i in channels])
        ...     combined_rms = sqrt(len(channels)) * rms_per_channel
        ...     snr = combined_flux / combined_rms
        ...     return combined_flux, combined_rms, snr
        >>>
        >>> detections = adaptive_bin_channels(
        ...     n_channels=16,
        ...     measure_fn=measure,
        ...     config=AdaptiveBinningConfig(target_snr=5.0, max_width=16),
        ... )
    """
    if config is None:
        config = AdaptiveBinningConfig()

    # Initialize: all channels available
    all_channels = list(range(n_channels))
    detections = []

    # Iterate through bin widths
    for check_width in range(config.initial_width, config.max_width + 1):
        if not all_channels:
            break  # No more channels to process

        new_all_channels = []

        # Find consecutive series
        consecutive_series = _find_consecutive_series(all_channels)

        for series_channels in consecutive_series:
            # Split into slices of check_width
            slices = _split_into_slices(series_channels, check_width)

            for slice_channels in slices:
                try:
                    # Measure photometry for this slice
                    flux, rms, snr = measure_fn(slice_channels)

                    if snr >= config.target_snr:
                        # Detection! Record it
                        center_freq = None
                        if channel_freqs_mhz and slice_channels:
                            # Calculate central frequency
                            freqs = [channel_freqs_mhz[i] for i in slice_channels]
                            center_freq = np.mean(freqs)

                        detections.append(
                            Detection(
                                channels=slice_channels,
                                flux_jy=flux,
                                rms_jy=rms,
                                snr=snr,
                                center_freq_mhz=center_freq,
                                bin_width=len(slice_channels),
                            )
                        )
                    else:
                        # Not detected - add back to pool for next iteration
                        new_all_channels.extend(slice_channels)

                except (ValueError, RuntimeError):
                    # Measurement failed - add back to pool
                    new_all_channels.extend(slice_channels)

        # Update channel pool for next iteration
        all_channels = new_all_channels

    # Final pass: try combining adjacent misfits
    if config.allow_adjacent_misfits and all_channels:
        misfit_detections = _try_adjacent_misfits(
            all_channels,
            measure_fn,
            config.target_snr,
        )
        detections.extend(misfit_detections)

    return detections


def create_measure_fn_from_images(
    image_paths: list[str],
    ra_deg: float,
    dec_deg: float,
    photometry_fn: Callable[[str, float, float], tuple[float, float]],
) -> Callable[[list[int]], tuple[float, float, float]]:
    """Create a measure function from a list of per-channel images.

        This helper creates a measure function that can be used with
        adaptive_bin_channels(). It combines multiple images and measures
        photometry at the specified coordinates.

    Parameters
    ----------
    image_paths : list
        List of image paths (one per channel/subband)
    ra_deg : float
        Right ascension (degrees)
    dec_deg : float
        Declination (degrees)
    photometry_fn : callable
        Function that takes (image_path, ra, dec) and returns
        (flux_jy, rms_jy)

    Returns
    -------
        callable
        Measure function compatible with adaptive_bin_channels()

    Examples
    --------
        >>> from dsa110_contimg.core.photometry.forced import measure_forced_peak
        >>>
        >>> def photometry_fn(image_path, ra, dec):
        ...     result = measure_forced_peak(image_path, ra, dec)
        ...     return result.peak_jyb, result.peak_err_jyb
        >>>
        >>> measure_fn = create_measure_fn_from_images(
        ...     image_paths=['sb0.fits', 'sb1.fits', ..., 'sb15.fits'],
        ...     ra_deg=128.725,
        ...     dec_deg=55.573,
        ...     photometry_fn=photometry_fn,
        ... )
        >>>
        >>> detections = adaptive_bin_channels(
        ...     n_channels=16,
        ...     measure_fn=measure_fn,
        ... )
    """

    def measure_fn(channels: list[int]) -> tuple[float, float, float]:
        """Measure combined flux, RMS, and SNR for given channels."""
        fluxes = []
        rms_values = []

        for channel_idx in channels:
            if channel_idx >= len(image_paths):
                raise ValueError(f"Channel index {channel_idx} out of range")

            image_path = image_paths[channel_idx]
            if not Path(image_path).exists():
                raise FileNotFoundError(f"Image not found: {image_path}")

            flux, rms = photometry_fn(image_path, ra_deg, dec_deg)

            if not (np.isfinite(flux) and np.isfinite(rms) and rms > 0):
                raise ValueError(f"Invalid measurement for channel {channel_idx}")

            fluxes.append(flux)
            rms_values.append(rms)

        # Combine: flux adds, RMS adds in quadrature
        combined_flux = sum(fluxes)
        combined_rms = np.sqrt(sum(r**2 for r in rms_values))

        # Calculate SNR
        snr = combined_flux / combined_rms if combined_rms > 0 else 0.0

        return combined_flux, combined_rms, snr

    return measure_fn
