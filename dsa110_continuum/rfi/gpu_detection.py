"""
GPU-accelerated RFI detection using CuPy.

This module implements MAD-based (Median Absolute Deviation) outlier detection
on GPU using CuPy. It processes measurement sets in chunks to handle large
datasets while staying within GPU memory limits.

Note: We use CuPy instead of Numba CUDA due to driver constraints
(Driver 455.23.05 limits PTX version, blocking Numba CUDA compilation).
CuPy's ElementwiseKernel provides similar performance with better compatibility.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from dsa110_contimg.common.utils.gpu_safety import (
    check_system_memory_available,
    initialize_gpu_safety,
    safe_gpu_context,
)

logger = logging.getLogger(__name__)

# Try to import CuPy - graceful fallback if unavailable
try:
    import cupy as cp

    CUPY_AVAILABLE = True
except ImportError:
    cp = None
    CUPY_AVAILABLE = False
    logger.warning("CuPy not available - GPU RFI detection disabled")


@dataclass
class RFIDetectionConfig:
    """Configuration for RFI detection.

    Attributes
    ----------
    threshold : float
        Detection threshold in MAD units (default 5.0)
    gpu_id : int
        GPU device ID to use (default 0)
    chunk_size : int
        Number of visibilities per processing chunk (default 10M)
    apply_flags : bool
        Whether to apply flags to the MS (default True)
    detect_only : bool
        Only detect, don't modify MS (default False)
    time_window : float or None
        Time window for temporal RFI detection in seconds (default None)
    freq_window : int or None
        Frequency window for spectral RFI detection in channels (default None)
    """

    threshold: float = 5.0
    gpu_id: int = 0
    chunk_size: int = 10_000_000
    apply_flags: bool = True
    detect_only: bool = False
    time_window: float | None = None
    freq_window: int | None = None


@dataclass
class RFIDetectionResult:
    """Result of RFI detection.

    Attributes
    ----------
    ms_path : str
        Path to the measurement set
    total_vis : int
        Total number of visibilities
    flagged_vis : int
        Number of newly flagged visibilities
    flag_percent : float
        Percentage of data flagged
    threshold : float
        Detection threshold used (MAD units)
    gpu_id : int
        GPU device used
    processing_time_s : float
        Processing time in seconds
    chunks_processed : int
        Number of chunks processed
    pre_existing_flags : int
        Number of visibilities already flagged
    error : str or None
        Error message if detection failed (None if successful)
    """

    ms_path: str
    total_vis: int = 0
    flagged_vis: int = 0
    flag_percent: float = 0.0
    threshold: float = 5.0
    gpu_id: int = 0
    processing_time_s: float = 0.0
    chunks_processed: int = 0
    pre_existing_flags: int = 0
    error: str | None = None

    @property
    def success(self) -> bool:
        """Check if detection completed successfully."""
        return self.error is None


def _detect_outliers_cupy(
    vis_data: np.ndarray,
    threshold: float,
) -> tuple[np.ndarray, int]:
    """Detect RFI using MAD-based outlier detection on GPU.

        Uses CuPy for GPU acceleration. The algorithm:
        1. Compute amplitude of complex visibilities
        2. Calculate median and MAD (Median Absolute Deviation)
        3. Flag visibilities exceeding threshold * MAD from median

    Parameters
    ----------
    vis_data : array_like
        Complex visibility data (any shape, will be flattened)
    threshold : float
        Detection threshold in MAD units

    Returns
    -------
        tuple
        Tuple of (flags array matching input shape, count of flagged samples)
    """
    if not CUPY_AVAILABLE:
        raise RuntimeError("CuPy not available for GPU RFI detection")

    original_shape = vis_data.shape

    # Transfer to GPU
    vis_gpu = cp.asarray(vis_data.ravel())

    # Compute amplitude
    amplitude = cp.abs(vis_gpu)

    # Compute MAD threshold
    # MAD = median(|x - median(x)|)
    median_amp = cp.median(amplitude)
    mad = cp.median(cp.abs(amplitude - median_amp))

    # Avoid division by zero - use small epsilon if MAD is 0
    if float(mad) < 1e-10:
        logger.warning("MAD is near zero - data may be constant or all flagged")
        mad = cp.float32(1e-10)

    # Threshold value
    thresh_value = median_amp + threshold * mad * 1.4826  # 1.4826 makes MAD comparable to std

    # Detect outliers
    flags_gpu = amplitude > thresh_value

    # Count flagged
    n_flagged = int(cp.sum(flags_gpu))

    # Transfer back to CPU
    flags_cpu = cp.asnumpy(flags_gpu).reshape(original_shape)

    # Cleanup GPU memory
    del vis_gpu, amplitude, flags_gpu
    cp.get_default_memory_pool().free_all_blocks()

    return flags_cpu.astype(bool), n_flagged


def _estimate_ms_shape(ms_path: str) -> tuple[int, int, int, int]:
    """Estimate the shape of data in a measurement set.

    Returns
    -------
        Tuple of (n_rows, n_channels, n_correlations, total_vis)
    """
    try:
        from casatools import table as tb

        t = tb()
        t.open(str(ms_path))

        n_rows = t.nrows()

        # Get shape from first row
        data_shape = t.getcell("DATA", 0).shape
        n_channels = data_shape[0]
        n_corr = data_shape[1]

        t.close()

        total_vis = n_rows * n_channels * n_corr
        return n_rows, n_channels, n_corr, total_vis

    except Exception as e:
        logger.error(f"Failed to read MS shape: {e}")
        raise


def gpu_rfi_detection(
    ms_path: str,
    *,
    config: RFIDetectionConfig | None = None,
    threshold: float | None = None,
    gpu_id: int | None = None,
    chunk_size: int | None = None,
) -> RFIDetectionResult:
    """GPU-accelerated RFI detection for measurement sets.

        Uses MAD-based outlier detection on GPU to identify and optionally
        flag Radio Frequency Interference in visibility data.

    Parameters
    ----------
    ms_path : str
        Path to measurement set
    config : RFIDetectionConfig, optional
        Detection configuration (optional)
    threshold : float, optional
        Detection threshold in MAD units (overrides config)
    gpu_id : int, optional
        GPU device ID (overrides config)
    chunk_size : int, optional
        Visibilities per chunk (overrides config)

    Returns
    -------
        RFIDetectionResult
        Detection statistics

        Example
    -------
        >>> result = gpu_rfi_detection(
        ...     "/data/dsa110/ms/2024-12-03_1234.ms",
        ...     threshold=5.0,
        ...     gpu_id=0
        ... )
        >>> print(f"Flagged {result.flag_percent:.2f}% of data")
    """
    import time

    # Initialize GPU safety if not already done
    initialize_gpu_safety()

    # Build configuration
    if config is None:
        config = RFIDetectionConfig()

    if threshold is not None:
        config.threshold = threshold
    if gpu_id is not None:
        config.gpu_id = gpu_id
    if chunk_size is not None:
        config.chunk_size = chunk_size

    # Initialize result
    result = RFIDetectionResult(
        ms_path=str(ms_path),
        threshold=config.threshold,
        gpu_id=config.gpu_id,
    )

    start_time = time.time()

    # Check if CuPy is available
    if not CUPY_AVAILABLE:
        result.error = "CuPy not available - GPU RFI detection disabled"
        logger.error(result.error)
        return result

    # Validate MS path
    ms_path = Path(ms_path)
    if not ms_path.exists():
        result.error = f"Measurement set not found: {ms_path}"
        logger.error(result.error)
        return result

    try:
        from casatools import table as tb

        # Get MS shape
        n_rows, n_channels, n_corr, total_vis = _estimate_ms_shape(str(ms_path))
        result.total_vis = total_vis

        logger.info(
            f"GPU RFI detection on {ms_path.name}: "
            f"{n_rows:,} rows × {n_channels} channels × {n_corr} corr = {total_vis:,} vis"
        )

        # Estimate memory and check safety
        mem_gb = (total_vis * 16) / (1024**3)  # complex128 = 16 bytes
        is_safe, reason = check_system_memory_available(mem_gb * 2)  # 2x for GPU copy
        if not is_safe:
            logger.warning(f"Memory check: {reason} - using chunked processing")

        # Calculate rows per chunk
        vis_per_row = n_channels * n_corr
        rows_per_chunk = max(1, config.chunk_size // vis_per_row)

        logger.info(f"Processing in chunks of {rows_per_chunk:,} rows")

        # Open MS
        t = tb()
        t.open(str(ms_path), nomodify=config.detect_only)

        total_flagged = 0
        pre_existing = 0
        chunks_processed = 0

        # Use safe GPU context
        with safe_gpu_context(max_gpu_gb=9.0, gpu_id=config.gpu_id):
            # Process in chunks
            for start_row in range(0, n_rows, rows_per_chunk):
                end_row = min(start_row + rows_per_chunk, n_rows)
                chunk_rows = end_row - start_row

                # Read chunk
                data = t.getcol("DATA", startrow=start_row, nrow=chunk_rows)
                existing_flags = t.getcol("FLAG", startrow=start_row, nrow=chunk_rows)

                # Count pre-existing flags
                pre_existing += int(np.sum(existing_flags))

                # Detect RFI on GPU
                new_flags, n_flagged = _detect_outliers_cupy(data, config.threshold)

                total_flagged += n_flagged

                # Apply flags if requested
                if config.apply_flags and not config.detect_only:
                    combined_flags = existing_flags | new_flags
                    t.putcol("FLAG", combined_flags, startrow=start_row, nrow=chunk_rows)

                chunks_processed += 1

                if chunks_processed % 10 == 0:
                    logger.debug(
                        f"Processed {chunks_processed} chunks, {total_flagged:,} flagged so far"
                    )

        t.close()

        # Update result
        result.flagged_vis = total_flagged
        result.flag_percent = (total_flagged / total_vis * 100) if total_vis > 0 else 0.0
        result.chunks_processed = chunks_processed
        result.pre_existing_flags = pre_existing
        result.processing_time_s = time.time() - start_time

        logger.info(
            f"RFI detection complete: {total_flagged:,} / {total_vis:,} "
            f"({result.flag_percent:.2f}%) flagged in {result.processing_time_s:.2f}s"
        )

    except Exception as e:
        result.error = str(e)
        result.processing_time_s = time.time() - start_time
        logger.error(f"RFI detection failed: {e}")

    return result


def cpu_rfi_detection(
    ms_path: str,
    *,
    threshold: float = 5.0,
    chunk_size: int = 1_000_000,
) -> RFIDetectionResult:
    """CPU fallback for RFI detection when GPU is unavailable.

        Uses the same MAD-based algorithm but runs on CPU with NumPy.

    Parameters
    ----------
    ms_path : str
        Path to measurement set
    threshold : float
        Detection threshold in MAD units
    chunk_size : int
        Visibilities per chunk

    Returns
    -------
        RFIDetectionResult
        Detection statistics
    """
    import time

    result = RFIDetectionResult(
        ms_path=str(ms_path),
        threshold=threshold,
        gpu_id=-1,  # -1 indicates CPU
    )

    start_time = time.time()
    ms_path = Path(ms_path)

    if not ms_path.exists():
        result.error = f"Measurement set not found: {ms_path}"
        return result

    try:
        from casatools import table as tb

        n_rows, n_channels, n_corr, total_vis = _estimate_ms_shape(str(ms_path))
        result.total_vis = total_vis

        vis_per_row = n_channels * n_corr
        rows_per_chunk = max(1, chunk_size // vis_per_row)

        t = tb()
        t.open(str(ms_path), nomodify=True)

        total_flagged = 0
        chunks_processed = 0

        for start_row in range(0, n_rows, rows_per_chunk):
            end_row = min(start_row + rows_per_chunk, n_rows)
            chunk_rows = end_row - start_row

            data = t.getcol("DATA", startrow=start_row, nrow=chunk_rows)

            # CPU MAD detection
            amplitude = np.abs(data.ravel())
            median_amp = np.median(amplitude)
            mad = np.median(np.abs(amplitude - median_amp))

            if mad < 1e-10:
                mad = 1e-10

            thresh_value = median_amp + threshold * mad * 1.4826
            flags = amplitude > thresh_value
            total_flagged += int(np.sum(flags))
            chunks_processed += 1

        t.close()

        result.flagged_vis = total_flagged
        result.flag_percent = (total_flagged / total_vis * 100) if total_vis > 0 else 0.0
        result.chunks_processed = chunks_processed
        result.processing_time_s = time.time() - start_time

    except Exception as e:
        result.error = str(e)
        result.processing_time_s = time.time() - start_time

    return result
