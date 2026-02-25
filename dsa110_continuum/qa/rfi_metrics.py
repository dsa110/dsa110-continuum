"""
RFI quality metrics for DSA-110 continuum imaging pipeline.

Calculates RFI occupancy statistics from Measurement Sets.
"""

import logging
from pathlib import Path

import numpy as np

from dsa110_contimg.common.utils.casa_init import ensure_casa_path

ensure_casa_path()

try:
    import casacore.tables as casatables

    HAVE_CASACORE = True
except ImportError:
    HAVE_CASACORE = False

logger = logging.getLogger(__name__)


def calculate_rfi_occupancy(ms_path: str | Path) -> dict[str, np.ndarray]:
    """Calculate RFI occupancy statistics from a Measurement Set.

        Computes the percentage of flagged data vs frequency and vs time.

    Parameters
    ----------
    ms_path : str
        Path to the Measurement Set.

    Returns
    -------
        dict
        Dictionary containing:
        - 'freq_occupancy': 1D array of % flagged per channel
        - 'time_occupancy': 1D array of % flagged per timestamp
        - 'frequencies': 1D array of channel frequencies (Hz)
        - 'times': 1D array of timestamps (MJD)
        - 'total_occupancy': Scalar % total flagged
    """
    if not HAVE_CASACORE:
        raise ImportError("casacore.tables is required for RFI metrics")

    ms_path = str(ms_path)
    if not Path(ms_path).exists():
        raise FileNotFoundError(f"Measurement Set not found: {ms_path}")

    logger.info(f"Calculating RFI metrics for {ms_path}")

    with casatables.table(ms_path, ack=False) as tb:
        # Read flags (Row x Chan x Pol)
        # Warning: This reads the entire flag column into memory.
        # For very large MS, this should be chunked.
        flags = tb.getcol("FLAG")

        # Get time axis (averaged over baselines for visualization)
        # Note: MS rows are baseline-time. To get pure time occupancy,
        # we ideally aggregate by time.
        # For simplicity in this metric, we'll just return the raw time column
        # and let the visualizer handle the scatter/binning, or we can unique it here.
        times = tb.getcol("TIME")

    # Get frequencies from SPECTRAL_WINDOW subtable
    with casatables.table(f"{ms_path}/SPECTRAL_WINDOW", ack=False) as sw:
        freqs = sw.getcol("CHAN_FREQ")[0]  # Assuming SPW 0 for now

    # Calculate occupancy vs Frequency (average over Time and Pol)
    # flags shape: (n_row, n_chan, n_pol)
    # Result: (n_chan,)
    freq_occupancy = np.mean(flags, axis=(0, 2)) * 100.0

    # Calculate occupancy vs Time (average over Freq and Pol)
    # Result: (n_row,) - This is per-baseline-time.
    # For a true "Time" occupancy, we might want to average all baselines for a given timestamp.
    # But returning the per-row metric allows seeing if specific baselines are bad.
    # For the waterfall, we usually want (Time, Freq).

    # Let's compute a simplified time occupancy for the 1D plot
    # We will average over all baselines for each unique timestamp later or just return per-row.
    # Returning per-row is safer for general usage.
    time_occupancy = np.mean(flags, axis=(1, 2)) * 100.0

    total_occupancy = np.mean(flags) * 100.0

    return {
        "freq_occupancy": freq_occupancy,
        "time_occupancy": time_occupancy,
        "frequencies": freqs,
        "times": times,
        "total_occupancy": total_occupancy,
        "n_channels": len(freqs),
        "n_rows": len(times),
    }


def get_rfi_waterfall_data(
    ms_path: str | Path, time_bin_seconds: float = 10.0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate Time-Frequency RFI occupancy grid (Waterfall).

        Aggregates flags across baselines to create a (Time, Freq) 2D array.

    Parameters
    ----------
    ms_path : str
        Path to MS.
    time_bin_seconds : int or float
        Binning size for time axis.

    Returns
    -------
        tuple
        Tuple of (waterfall_array, unique_times, freqs).
        waterfall_array shape is (n_times, n_freqs).
    """
    if not HAVE_CASACORE:
        raise ImportError("casacore.tables is required for RFI metrics")

    ms_path = str(ms_path)

    with casatables.table(ms_path, ack=False) as tb:
        times = tb.getcol("TIME")
        flags = tb.getcol("FLAG")  # (Row, Chan, Pol)

    with casatables.table(f"{ms_path}/SPECTRAL_WINDOW", ack=False) as sw:
        freqs = sw.getcol("CHAN_FREQ")[0]

    # Collapse Pol axis -> (Row, Chan)
    # If any pol is flagged, we count it? Or average?
    # Usually RFI affects all pols, but let's average to get a "fraction flagged" 0.0-1.0
    flags_row_chan = np.mean(flags, axis=2)

    # Bin by time
    # Find unique times and inverse indices
    unique_times, inverse_indices = np.unique(times, return_inverse=True)

    n_times = len(unique_times)
    n_chans = len(freqs)

    waterfall = np.zeros((n_times, n_chans))
    np.zeros((n_times, 1))  # Broadcastable

    # Accumulate flags for each time bin
    # This is a simple summation. For huge arrays, this loop might be slow in pure Python,
    # but n_times is usually small (~few hundred for 5 min obs).
    # Optimization: use numpy.add.at

    np.add.at(waterfall, inverse_indices, flags_row_chan)

    # Count how many rows (baselines) contributed to each time
    # We can just count occurrences of each index
    row_counts = np.bincount(inverse_indices, minlength=n_times)

    # Avoid division by zero
    row_counts[row_counts == 0] = 1

    # Normalize
    waterfall = (waterfall / row_counts[:, None]) * 100.0

    return waterfall, unique_times, freqs
