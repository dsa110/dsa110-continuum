"""
Downsample UVH5 files for faster testing.

This module provides functionality to downsample UVH5 (HDF5) visibility files
by averaging in time and frequency, producing smaller datasets suitable for
rapid pipeline testing while preserving data integrity.

Averaging strategy:
- Frequency: Average every N channels within each subband (weighted by nsamples)
- Time: Average every N integrations (weighted by nsamples)
- Flags: Conservative propagation (flagged if ANY input sample flagged)
- Weights: Sum of input nsamples
- UVW: Linear average (valid for short time spans)

Usage:
    from dsa110_contimg.core.conversion.downsample_uvh5 import downsample_uvh5

    downsample_uvh5(
        input_path="/path/to/input.hdf5",
        output_path="/path/to/output.hdf5",
        time_factor=4,
        freq_factor=4,
    )
"""

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from scipy import stats

if TYPE_CHECKING:
    from pyuvdata import UVData

logger = logging.getLogger(__name__)


def downsample_uvh5(
    input_path: str | Path,
    output_path: str | Path,
    time_factor: int = 4,
    freq_factor: int = 4,
    overwrite: bool = False,
) -> Path:
    """Downsample a UVH5 file by averaging in time and frequency.

    Parameters
    ----------
    input_path :
        Path to input UVH5 file
    output_path :
        Path for output downsampled UVH5 file
    time_factor :
        Number of time samples to average (must evenly divide Ntimes)
    freq_factor :
        Number of frequency channels to average per subband
    overwrite :
        If True, overwrite existing output file
    input_path : Union[str, Path]
    output_path: Union[str :

    Returns
    -------
        Path to the output file

    Raises
    ------
    FileNotFoundError
        If input file doesn't exist
    FileExistsError
        If output exists and overwrite=False
    ValueError
        If factors don't evenly divide data dimensions

    """
    from pyuvdata import UVData

    input_path = Path(input_path)
    output_path = Path(output_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output file exists: {output_path}. Use overwrite=True.")

    logger.info(f"Downsampling {input_path.name} by {time_factor}x time, {freq_factor}x freq")

    # Read the input UVH5 file
    uv = UVData()
    uv.read(
        str(input_path),
        file_type="uvh5",
        run_check=False,
        run_check_acceptability=False,
        strict_uvw_antpos_check=False,
        check_extra=False,
    )

    # Get original dimensions
    n_blts_orig = uv.Nblts
    n_freqs_orig = uv.Nfreqs
    n_times_orig = uv.Ntimes
    n_bls = uv.Nbls
    n_pols = uv.Npols

    logger.info(
        f"Original: Ntimes={n_times_orig}, Nfreqs={n_freqs_orig}, "
        f"Nbls={n_bls}, Npols={n_pols}, Nblts={n_blts_orig}"
    )

    # Validate factors
    if n_times_orig % time_factor != 0:
        raise ValueError(
            f"time_factor={time_factor} does not evenly divide Ntimes={n_times_orig}. "
            f"Valid factors: {_get_divisors(n_times_orig)}"
        )
    if n_freqs_orig % freq_factor != 0:
        raise ValueError(
            f"freq_factor={freq_factor} does not evenly divide Nfreqs={n_freqs_orig}. "
            f"Valid factors: {_get_divisors(n_freqs_orig)}"
        )

    # Perform frequency averaging
    if freq_factor > 1:
        uv = _average_frequency(uv, freq_factor)

    # Perform time averaging
    if time_factor > 1:
        uv = _average_time(uv, time_factor)

    # Log final dimensions
    logger.info(
        f"Downsampled: Ntimes={uv.Ntimes}, Nfreqs={uv.Nfreqs}, "
        f"Nbls={uv.Nbls}, Npols={uv.Npols}, Nblts={uv.Nblts}"
    )

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Write output
    if output_path.exists():
        output_path.unlink()

    uv.write_uvh5(str(output_path), run_check=False)
    logger.info(f"Wrote downsampled file: {output_path}")

    return output_path


def _average_frequency(uv: "UVData", factor: int) -> "UVData":
    """Average visibility data over frequency channels.

    Uses weighted averaging based on nsamples, with conservative flag propagation.

    Parameters
    ----------
    uv :
        UVData object to average
    factor :
        Number of channels to average

    Returns
    -------
        New UVData object with averaged frequencies

    """
    n_freqs_orig = uv.Nfreqs
    n_freqs_new = n_freqs_orig // factor

    logger.debug(f"Frequency averaging: {n_freqs_orig} -> {n_freqs_new} channels")

    # Get data arrays
    # data_array shape: (Nblts, Nfreqs, Npols) for pyuvdata 3.0+
    # or (Nblts, Nspws, Nfreqs, Npols) for older versions
    data = uv.data_array
    flags = uv.flag_array
    nsamples = uv.nsample_array

    # Handle different pyuvdata versions (with or without Nspws dimension)
    if data.ndim == 4:
        # Old format: (Nblts, Nspws, Nfreqs, Npols)
        n_blts, n_spws, n_freqs, n_pols = data.shape
        has_spws_dim = True
    else:
        # New format: (Nblts, Nfreqs, Npols)
        n_blts, n_freqs, n_pols = data.shape
        n_spws = 1
        has_spws_dim = False

    # Reshape for averaging
    if has_spws_dim:
        # (Nblts, Nspws, Nfreqs, Npols) -> (Nblts, Nspws, Nfreqs_new, factor, Npols)
        data_reshaped = data.reshape(n_blts, n_spws, n_freqs_new, factor, n_pols)
        flags_reshaped = flags.reshape(n_blts, n_spws, n_freqs_new, factor, n_pols)
        nsamples_reshaped = nsamples.reshape(n_blts, n_spws, n_freqs_new, factor, n_pols)
    else:
        # (Nblts, Nfreqs, Npols) -> (Nblts, Nfreqs_new, factor, Npols)
        data_reshaped = data.reshape(n_blts, n_freqs_new, factor, n_pols)
        flags_reshaped = flags.reshape(n_blts, n_freqs_new, factor, n_pols)
        nsamples_reshaped = nsamples.reshape(n_blts, n_freqs_new, factor, n_pols)

    # Weighted average of visibilities
    # Zero out weights where flagged
    weights = np.where(flags_reshaped, 0.0, nsamples_reshaped)
    weight_sum = weights.sum(axis=-2 if has_spws_dim else -2, keepdims=True)
    # Avoid division by zero
    weight_sum = np.where(weight_sum == 0, 1.0, weight_sum)

    # Weighted mean
    data_weighted = data_reshaped * weights
    if has_spws_dim:
        data_avg = data_weighted.sum(axis=3) / weight_sum.squeeze(axis=3)
    else:
        data_avg = data_weighted.sum(axis=2) / weight_sum.squeeze(axis=2)

    # Conservative flag propagation: flagged if ANY input flagged
    if has_spws_dim:
        flags_avg = flags_reshaped.any(axis=3)
        nsamples_avg = nsamples_reshaped.sum(axis=3)
    else:
        flags_avg = flags_reshaped.any(axis=2)
        nsamples_avg = nsamples_reshaped.sum(axis=2)

    # Update frequency array (take center of each averaged bin)
    freq_array = uv.freq_array
    if freq_array.ndim == 2:
        # Old format: (Nspws, Nfreqs)
        freq_reshaped = freq_array.reshape(n_spws, n_freqs_new, factor)
        freq_avg = freq_reshaped.mean(axis=2)
    else:
        # New format: (Nfreqs,)
        freq_reshaped = freq_array.reshape(n_freqs_new, factor)
        freq_avg = freq_reshaped.mean(axis=1)

    # Update channel width
    if hasattr(uv, "channel_width"):
        if np.isscalar(uv.channel_width):
            new_channel_width = uv.channel_width * factor
        elif isinstance(uv.channel_width, np.ndarray):
            if uv.channel_width.ndim == 0:
                new_channel_width = float(uv.channel_width) * factor
            else:
                # Average channel widths and multiply by factor
                new_channel_width = uv.channel_width.reshape(-1, factor).sum(axis=1)
        else:
            new_channel_width = float(uv.channel_width) * factor
    else:
        new_channel_width = None

    # Update flex_spw_id_array if present
    # This array has one entry per frequency channel, so we need to resample it
    # Take every factor'th element (center of each averaging bin)
    if uv.flex_spw_id_array is not None:
        flex_spw_orig = uv.flex_spw_id_array
        if len(flex_spw_orig) == n_freqs_orig:
            # Take center element of each averaging group
            flex_spw_new = flex_spw_orig[factor // 2 :: factor]
            logger.debug(f"Updated flex_spw_id_array: {len(flex_spw_orig)} -> {len(flex_spw_new)}")
        else:
            # Shape mismatch, leave as-is with warning
            logger.warning(
                f"flex_spw_id_array length {len(flex_spw_orig)} != Nfreqs {n_freqs_orig}, "
                "leaving unchanged"
            )
            flex_spw_new = flex_spw_orig
    else:
        flex_spw_new = None

    # Update UVData object
    uv.data_array = data_avg
    uv.flag_array = flags_avg
    uv.nsample_array = nsamples_avg.astype(np.float32)
    uv.freq_array = freq_avg
    uv.Nfreqs = n_freqs_new
    if new_channel_width is not None:
        uv.channel_width = new_channel_width
    if flex_spw_new is not None:
        uv.flex_spw_id_array = flex_spw_new

    return uv


def _average_time(uv: "UVData", factor: int) -> "UVData":
    """Average visibility data over time samples.

    Uses weighted averaging based on nsamples, with conservative flag propagation.
    Linear averaging of UVW coordinates (valid for short time spans).

    Parameters
    ----------
    uv :
        UVData object to average
    factor :
        Number of time samples to average

    Returns
    -------
        New UVData object with averaged times

    """
    n_times_orig = uv.Ntimes
    n_times_new = n_times_orig // factor
    n_bls = uv.Nbls
    n_freqs = uv.Nfreqs
    n_pols = uv.Npols

    logger.debug(f"Time averaging: {n_times_orig} -> {n_times_new} time samples")

    # Get unique times and baselines
    unique_times = np.unique(uv.time_array)
    if len(unique_times) != n_times_orig:
        raise ValueError(
            f"Unexpected time structure: found {len(unique_times)} unique times, "
            f"expected {n_times_orig}"
        )

    # Data is typically in baseline-time order: (baseline varies fast, time varies slow)
    # or time-baseline order. We need to handle both.
    # Check the ordering by looking at ant1/ant2 arrays
    data = uv.data_array
    flags = uv.flag_array
    nsamples = uv.nsample_array
    uvw = uv.uvw_array

    # Handle different pyuvdata versions
    if data.ndim == 4:
        has_spws_dim = True
        n_blts, n_spws, n_freq, n_pol = data.shape
    else:
        has_spws_dim = False
        n_blts, n_freq, n_pol = data.shape
        n_spws = 1

    # Reshape assuming time-major order (time varies slow, baseline varies fast)
    # Shape: (Ntimes, Nbls, ...) for averaging over time axis
    if has_spws_dim:
        data_reshaped = data.reshape(n_times_orig, n_bls, n_spws, n_freqs, n_pols)
        flags_reshaped = flags.reshape(n_times_orig, n_bls, n_spws, n_freqs, n_pols)
        nsamples_reshaped = nsamples.reshape(n_times_orig, n_bls, n_spws, n_freqs, n_pols)
    else:
        data_reshaped = data.reshape(n_times_orig, n_bls, n_freqs, n_pols)
        flags_reshaped = flags.reshape(n_times_orig, n_bls, n_freqs, n_pols)
        nsamples_reshaped = nsamples.reshape(n_times_orig, n_bls, n_freqs, n_pols)

    uvw_reshaped = uvw.reshape(n_times_orig, n_bls, 3)

    # Group time samples for averaging
    # (Ntimes, Nbls, ...) -> (Ntimes_new, factor, Nbls, ...)
    if has_spws_dim:
        data_grouped = data_reshaped.reshape(n_times_new, factor, n_bls, n_spws, n_freqs, n_pols)
        flags_grouped = flags_reshaped.reshape(n_times_new, factor, n_bls, n_spws, n_freqs, n_pols)
        nsamples_grouped = nsamples_reshaped.reshape(
            n_times_new, factor, n_bls, n_spws, n_freqs, n_pols
        )
    else:
        data_grouped = data_reshaped.reshape(n_times_new, factor, n_bls, n_freqs, n_pols)
        flags_grouped = flags_reshaped.reshape(n_times_new, factor, n_bls, n_freqs, n_pols)
        nsamples_grouped = nsamples_reshaped.reshape(n_times_new, factor, n_bls, n_freqs, n_pols)

    uvw_grouped = uvw_reshaped.reshape(n_times_new, factor, n_bls, 3)

    # Weighted average of visibilities
    weights = np.where(flags_grouped, 0.0, nsamples_grouped)
    weight_sum = weights.sum(axis=1, keepdims=True)
    weight_sum = np.where(weight_sum == 0, 1.0, weight_sum)

    data_weighted = data_grouped * weights
    data_avg = data_weighted.sum(axis=1) / weight_sum.squeeze(axis=1)

    # Conservative flag propagation
    flags_avg = flags_grouped.any(axis=1)
    nsamples_avg = nsamples_grouped.sum(axis=1)

    # Linear average of UVW
    uvw_avg = uvw_grouped.mean(axis=1)

    # Flatten back to (Nblts_new, ...)
    n_blts_new = n_times_new * n_bls
    if has_spws_dim:
        data_final = data_avg.reshape(n_blts_new, n_spws, n_freqs, n_pols)
        flags_final = flags_avg.reshape(n_blts_new, n_spws, n_freqs, n_pols)
        nsamples_final = nsamples_avg.reshape(n_blts_new, n_spws, n_freqs, n_pols)
    else:
        data_final = data_avg.reshape(n_blts_new, n_freqs, n_pols)
        flags_final = flags_avg.reshape(n_blts_new, n_freqs, n_pols)
        nsamples_final = nsamples_avg.reshape(n_blts_new, n_freqs, n_pols)

    uvw_final = uvw_avg.reshape(n_blts_new, 3)

    # Average time array (take center of each averaged bin)
    times_reshaped = unique_times.reshape(n_times_new, factor)
    times_avg = times_reshaped.mean(axis=1)

    # Expand back to full blt ordering
    time_array_new = np.repeat(times_avg, n_bls)

    # Update integration time
    int_time_orig = uv.integration_time
    if np.isscalar(int_time_orig) or int_time_orig.size == 1:
        int_time_new = float(int_time_orig) * factor * np.ones(n_blts_new)
    else:
        # Reshape and sum integration times
        int_time_reshaped = int_time_orig.reshape(n_times_orig, n_bls)
        int_time_grouped = int_time_reshaped.reshape(n_times_new, factor, n_bls)
        int_time_avg = int_time_grouped.sum(axis=1)
        int_time_new = int_time_avg.flatten()

    # Update antenna arrays (take first baseline set, repeated for new times)
    ant1_orig = uv.ant_1_array.reshape(n_times_orig, n_bls)
    ant2_orig = uv.ant_2_array.reshape(n_times_orig, n_bls)
    ant1_new = np.tile(ant1_orig[0], n_times_new)
    ant2_new = np.tile(ant2_orig[0], n_times_new)

    # Update baseline array if present
    if hasattr(uv, "baseline_array") and uv.baseline_array is not None:
        bl_orig = uv.baseline_array.reshape(n_times_orig, n_bls)
        bl_new = np.tile(bl_orig[0], n_times_new)
        uv.baseline_array = bl_new

    # Update UVData object
    uv.data_array = data_final
    uv.flag_array = flags_final
    uv.nsample_array = nsamples_final.astype(np.float32)
    uv.uvw_array = uvw_final.astype(np.float64)
    uv.time_array = time_array_new
    uv.integration_time = int_time_new
    uv.ant_1_array = ant1_new
    uv.ant_2_array = ant2_new
    uv.Ntimes = n_times_new
    uv.Nblts = n_blts_new

    # Update LST array if present
    if hasattr(uv, "lst_array") and uv.lst_array is not None:
        lst_orig = uv.lst_array.reshape(n_times_orig, n_bls)
        lst_grouped = lst_orig.reshape(n_times_new, factor, n_bls)
        # Handle LST wraparound at 2π
        lst_avg = stats.circmean(lst_grouped, high=2 * np.pi, low=0, axis=1)
        uv.lst_array = lst_avg.flatten()

    # Update phase center arrays if present (these are per-blt)
    # Take the center value of each time averaging bin
    center_idx = factor // 2
    for attr in [
        "phase_center_app_ra",
        "phase_center_app_dec",
        "phase_center_frame_pa",
        "phase_center_id_array",
    ]:
        if hasattr(uv, attr):
            arr = getattr(uv, attr)
            if arr is not None and len(arr) == n_times_orig * n_bls:
                arr_reshaped = arr.reshape(n_times_orig, n_bls)
                arr_grouped = arr_reshaped.reshape(n_times_new, factor, n_bls)
                # Take center value from each averaging group
                arr_new = arr_grouped[:, center_idx, :].flatten()
                setattr(uv, attr, arr_new)
                logger.debug(f"Updated {attr}: {len(arr)} -> {len(arr_new)}")

    return uv


def _get_divisors(n: int) -> list[int]:
    """Get all divisors of n for helpful error messages.

    Parameters
    ----------
    """
    divisors = []
    for i in range(1, int(n**0.5) + 1):
        if n % i == 0:
            divisors.append(i)
            if i != n // i:
                divisors.append(n // i)
    return sorted(divisors)


def get_downsampling_info(input_path: str | Path) -> dict:
    """Get information about a UVH5 file to help choose downsampling factors.

    Parameters
    ----------
    input_path :
        Path to UVH5 file
    input_path : Union[str, Path]

    Returns
    -------
        Dict with Ntimes, Nfreqs, valid time factors, valid freq factors

    """
    from pyuvdata import UVData

    uv = UVData()
    # Read only metadata (fast)
    uv.read(
        str(input_path),
        file_type="uvh5",
        read_data=False,
        run_check=False,
    )

    return {
        "Ntimes": uv.Ntimes,
        "Nfreqs": uv.Nfreqs,
        "Nbls": uv.Nbls,
        "Npols": uv.Npols,
        "integration_time_sec": float(np.median(uv.integration_time)),
        "channel_width_hz": float(np.abs(np.median(uv.channel_width))),
        "valid_time_factors": _get_divisors(uv.Ntimes),
        "valid_freq_factors": _get_divisors(uv.Nfreqs),
    }
