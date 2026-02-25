"""
Adaptive binning photometry integration for DSA-110 pipeline.

This module integrates adaptive channel binning with the photometry workflow,
allowing automatic detection of weak sources by combining multiple subbands.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from dsa110_contimg.core.imaging.spw_imaging import get_spw_info, image_all_spws
from dsa110_contimg.core.photometry.adaptive_binning import (
    AdaptiveBinningConfig,
    Detection,
    adaptive_bin_channels,
    create_measure_fn_from_images,
)
from dsa110_contimg.core.photometry.forced import measure_forced_peak
from dsa110_contimg.common.utils.runtime_safeguards import (
    log_progress,
    progress_monitor,
)

LOG = logging.getLogger(__name__)


@dataclass
class AdaptivePhotometryResult:
    """Result from adaptive binning photometry."""

    ra_deg: float
    dec_deg: float
    detections: list[Detection]
    n_spws: int
    spw_info: list  # List of SPWInfo objects
    success: bool
    error_message: str | None = None


@progress_monitor(operation_name="Adaptive Photometry", warn_threshold=300.0)
def measure_with_adaptive_binning(
    ms_path: str,
    ra_deg: float,
    dec_deg: float,
    output_dir: Path,
    config: AdaptiveBinningConfig | None = None,
    photometry_fn: Callable[[str, float, float], tuple[float, float]] | None = None,
    max_spws: int | None = None,
    **imaging_kwargs,
) -> AdaptivePhotometryResult:
    """Measure photometry using adaptive channel binning.

        This function:
        1. Images all SPWs individually
        2. Measures photometry on each SPW image
        3. Applies adaptive binning algorithm to find optimal combinations
        4. Returns detections with optimal binning

    Parameters
    ----------
    ms_path : str or Path
        Path to Measurement Set
    ra_deg : float
        Right ascension (degrees)
    dec_deg : float
        Declination (degrees)
    output_dir : str or Path
        Directory for SPW images and results
    config : object, optional
        Adaptive binning configuration (uses defaults if None)
    photometry_fn : callable, optional
        Optional custom photometry function. If None, uses
        measure_forced_peak(). Should take (image_path, ra, dec)
        and return (flux_jy, rms_jy).
        **imaging_kwargs : dict
        Additional arguments passed to image_ms()

    Returns
    -------
        AdaptivePhotometryResult
        Result object with detections

    Examples
    --------
        >>> from pathlib import Path
        >>> result = measure_with_adaptive_binning(
        ...     ms_path="data.ms",
        ...     ra_deg=128.725,
        ...     dec_deg=55.573,
        ...     output_dir=Path("adaptive_results/"),
        ...     imsize=1024,
        ...     quality_tier="standard",
        ... )
        >>> print(f"Found {len(result.detections)} detections")
        >>> for det in result.detections:
        ...     print(f"SPWs {det.channels}: SNR={det.snr:.2f}, Flux={det.flux_jy:.6f} Jy")
    """
    import time

    start_time_sec = time.time()
    LOG.info(f"Starting adaptive photometry at ({ra_deg:.6f}, {dec_deg:.6f})...")

    try:
        # Get SPW information
        spw_info_list = get_spw_info(ms_path)
        n_spws = len(spw_info_list)

        if n_spws == 0:
            return AdaptivePhotometryResult(
                ra_deg=ra_deg,
                dec_deg=dec_deg,
                detections=[],
                n_spws=0,
                spw_info=[],
                success=False,
                error_message="No SPWs found in Measurement Set",
            )

        LOG.info(f"Found {n_spws} SPW(s) in {ms_path}")

        # Limit SPWs if requested
        if max_spws is not None and max_spws < n_spws:
            LOG.info(f"Limiting to first {max_spws} SPW(s) (out of {n_spws})")
            spw_ids_to_image = [info.spw_id for info in spw_info_list[:max_spws]]
            n_spws_used = max_spws
        else:
            spw_ids_to_image = None
            n_spws_used = n_spws

        # Image SPWs
        output_dir.mkdir(parents=True, exist_ok=True)
        spw_images_dir = output_dir / "spw_images"

        LOG.info(f"Imaging {n_spws_used} SPW(s)...")
        # Extract parallel imaging parameters if provided
        parallel = imaging_kwargs.pop("parallel", False)
        max_workers = imaging_kwargs.pop("max_workers", None)
        serialize_ms_access = imaging_kwargs.pop("serialize_ms_access", False)

        spw_image_paths = image_all_spws(
            ms_path=ms_path,
            output_dir=spw_images_dir,
            base_name="spw",
            spw_ids=spw_ids_to_image,
            parallel=parallel,
            max_workers=max_workers,
            serialize_ms_access=serialize_ms_access,
            **imaging_kwargs,
        )

        # Sort by SPW ID and extract paths
        spw_image_paths_sorted = sorted(spw_image_paths, key=lambda x: x[0])
        image_paths = [str(path) for _, path in spw_image_paths_sorted]

        if len(image_paths) != n_spws:
            LOG.warning(
                f"Expected {n_spws} images but got {len(image_paths)}. "
                "Some SPWs may have failed to image."
            )

        # Create photometry function if not provided
        if photometry_fn is None:

            def photometry_fn(image_path: str, ra: float, dec: float) -> tuple[float, float]:
                """Default photometry using measure_forced_peak."""
                result = measure_forced_peak(
                    image_path,
                    ra,
                    dec,
                    box_size_pix=5,
                    annulus_pix=(30, 50),
                )
                # Convert from Jy/beam to Jy (approximate)
                flux_jy = result.peak_jyb
                rms_jy = (
                    result.peak_err_jyb if result.peak_err_jyb is not None else result.local_rms_jy
                )
                # Filter non-finite values from RMS calculation
                if rms_jy is None or not np.isfinite(rms_jy):
                    # Use safe filtering if rms_jy is invalid
                    if hasattr(result, "rms_jy") and result.rms_jy is not None:
                        rms_jy = result.rms_jy if np.isfinite(result.rms_jy) else None
                    rms_jy = 0.001  # Default RMS if not available
                return flux_jy, rms_jy

        # Create measure function for adaptive binning
        measure_fn = create_measure_fn_from_images(
            image_paths=image_paths,
            ra_deg=ra_deg,
            dec_deg=dec_deg,
            photometry_fn=photometry_fn,
        )

        # Get channel frequencies for center frequency calculation
        # Use only the SPWs that were actually imaged
        spw_ids_imaged = [spw_id for spw_id, _ in spw_image_paths]
        spw_info_used = [info for info in spw_info_list if info.spw_id in spw_ids_imaged]
        channel_freqs_mhz = [info.center_freq_mhz for info in spw_info_used]

        # Run adaptive binning
        LOG.info("Running adaptive binning algorithm...")
        detections = adaptive_bin_channels(
            n_channels=len(spw_info_used),
            measure_fn=measure_fn,
            config=config,
            channel_freqs_mhz=channel_freqs_mhz,
        )

        LOG.info(f"Found {len(detections)} detection(s) with adaptive binning")

        log_progress(
            f"Completed adaptive photometry: {len(detections)} detection(s) found",
            start_time_sec,
        )
        return AdaptivePhotometryResult(
            ra_deg=ra_deg,
            dec_deg=dec_deg,
            detections=detections,
            n_spws=len(spw_info_used),
            spw_info=spw_info_used,
            success=True,
        )

    except Exception as e:
        LOG.error(f"Adaptive binning photometry failed: {e}", exc_info=True)
        log_progress(f"Adaptive photometry failed: {e}", start_time_sec)
        return AdaptivePhotometryResult(
            ra_deg=ra_deg,
            dec_deg=dec_deg,
            detections=[],
            n_spws=0,
            spw_info=[],
            success=False,
            error_message=str(e),
        )
