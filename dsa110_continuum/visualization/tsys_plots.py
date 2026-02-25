"""
Autocorrelation amplitude monitoring and visualization utilities.

**IMPORTANT**: This module does NOT calculate actual system temperature (Tsys).
It extracts autocorrelation amplitudes from Measurement Sets, which are
proportional to Tsys but require proper calibration (noise diode measurements)
to convert to actual Tsys in Kelvin.

The autocorrelation amplitudes are useful for relative monitoring:
- Comparing antennas (identifying hardware problems)
- Tracking changes over time (detecting instability)
- Identifying anomalous behavior

However, the absolute values have no physical meaning without proper calibration.

**DSA-110 Expected Tsys**: < 20 K (due to low-noise amplifiers)
Note: This is the actual expected Tsys range, but the values returned by this
module are autocorrelation amplitudes in Jy, NOT calibrated Tsys in Kelvin.

For proper Tsys measurement, use noise diode calibration or other system
calibration methods.

DSA-110 Array:
- 117 antenna indices in data files
- 96 active antennas used in observations

This module provides:
- Autocorrelation amplitude time series plots per antenna
- Summary statistics and heatmaps
- Anomaly detection visualization
- Autocorrelation amplitude vs elevation correlation
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from matplotlib.figure import Figure
    from numpy.typing import NDArray

from dsa110_contimg.core.visualization.config import FigureConfig, PlotStyle

logger = logging.getLogger(__name__)


# DSA-110 system constants
# Note: These are not used for Tsys computation (proper Tsys requires noise diode calibration)
DSA110_BANDWIDTH_HZ = 250e6  # Total bandwidth in Hz (250 MHz)
DSA110_DISH_DIAMETER_M = 4.65  # Dish diameter in meters


def _setup_matplotlib() -> None:
    """Configure matplotlib for headless operation."""
    import matplotlib

    matplotlib.use("Agg")


def extract_tsys_from_ms(
    ms_path: str | Path,
    spw: int = 0,
    time_average: int = 1,
    datacolumn: str = "DATA",
) -> dict[str, NDArray]:
    """Extract autocorrelation amplitudes from a Measurement Set.

        **IMPORTANT**: This function does NOT calculate actual Tsys. It extracts autocorrelation
        amplitudes, which are proportional to system temperature but require proper calibration
        (noise diode measurements) to convert to actual Tsys in Kelvin.

        **Data Column Selection**:
        - **DATA**: Raw visibilities (default). Nominally in Jy but NOT flux-calibrated.
        Use for relative monitoring before calibration.
        - **CORRECTED_DATA**: Calibrated visibilities (after gain/bandpass/flux calibration).
        Use when MS has been flux-calibrated for more accurate Jy values.
        Falls back to DATA if CORRECTED_DATA doesn't exist.

        **Flux Calibration Note**: Accurate Jy-unit values require flux calibration (observing
        a flux calibrator and applying the flux scale). Without flux calibration, values are
        only relative/scaled and should be used for relative monitoring only.

        The values are averaged over polarization and frequency channels. They are useful for:
        - Relative monitoring (comparing antennas, identifying hardware problems)
        - Tracking changes over time (detecting instability)
        - Identifying anomalous behavior

        However, they are NOT:
        - Calibrated Tsys values (requires noise diode calibration)
        - Flux-calibrated Jy values unless datacolumn="CORRECTED_DATA" AND MS has been flux-calibrated

        For proper Tsys measurement, use noise diode calibration or other system calibration methods.

    Parameters
    ----------
    ms_path : str or Path
        Path to the Measurement Set.
    spw : int, optional
        Spectral window to analyze. Default is 0.
    time_average : int, optional
        Number of time samples to average (reduces noise). Default is 1.
    datacolumn : str, optional
        Data column to read from ("DATA" or "CORRECTED_DATA", default: "DATA").
        If "CORRECTED_DATA" is requested but doesn't exist, falls back to "DATA".
        Default is "DATA".

    Returns
    -------
        dict
        Dictionary containing:
        - 'times': Time array (MJD).
        - 'tsys': Autocorrelation amplitude array, shape (n_times, n_antennas) in Jy
        (labeled as 'tsys' for API compatibility, but values are NOT Tsys in K).
        - 'antenna_ids': Array of antenna IDs.
        - 'antenna_names': List of antenna names.
        - 'n_times': Number of time samples.
        - 'n_antennas': Number of antennas with autocorrelations.
        - 'datacolumn_used': Which data column was actually used ("DATA" or "CORRECTED_DATA").

    Raises
    ------
        ValueError
        If no autocorrelation data is found or invalid datacolumn specified.
        RuntimeError
        If MS cannot be opened.

    Examples
    --------
        >>> # Read from raw DATA column (before calibration)
        >>> tsys_data = extract_tsys_from_ms('/path/to/observation.ms', datacolumn="DATA")
        >>> # Read from CORRECTED_DATA (after calibration, if flux-calibrated)
        >>> tsys_data = extract_tsys_from_ms('/path/to/observation.ms', datacolumn="CORRECTED_DATA")
        >>> plot_tsys_time_series(tsys_data['times'], tsys_data['tsys'])
    """
    import casatools

    ms_path = Path(ms_path)
    if not ms_path.exists():
        raise FileNotFoundError(f"MS not found: {ms_path}")

    ms = casatools.ms()
    tb = casatools.table()

    # Validate datacolumn parameter
    if datacolumn not in ("DATA", "CORRECTED_DATA"):
        raise ValueError(f"datacolumn must be 'DATA' or 'CORRECTED_DATA', got '{datacolumn}'")

    try:
        # Open MS and check which columns exist
        ms.open(str(ms_path))
        ms.selectinit(datadescid=spw)

        # Check if requested column exists, fall back to DATA if not
        tb.open(str(ms_path))
        colnames = tb.colnames()
        tb.close()

        actual_column = datacolumn
        if datacolumn == "CORRECTED_DATA" and "CORRECTED_DATA" not in colnames:
            logger.warning(f"CORRECTED_DATA column not found in {ms_path}, falling back to DATA")
            actual_column = "DATA"

        # Get all data first, then filter for autocorrelations
        # Use lowercase for ms.getdata() column names
        column_name = actual_column.lower()
        data = ms.getdata(["antenna1", "antenna2", column_name, "time", "flag"])

        if data["antenna1"] is None or len(data["antenna1"]) == 0:
            raise ValueError(f"No data found in MS: {ms_path}")

        # Filter for autocorrelations (antenna1 == antenna2)
        auto_mask = data["antenna1"] == data["antenna2"]
        n_auto = auto_mask.sum()

        if n_auto == 0:
            raise ValueError(f"No autocorrelation data found in MS: {ms_path}")

        logger.info(f"Found {n_auto} autocorrelation records")

        # Extract autocorrelation data
        auto_ant = data["antenna1"][auto_mask]
        auto_time = data["time"][auto_mask]
        # Use the actual column name (lowercase for ms.getdata() return dict)
        auto_data = data[column_name][:, :, auto_mask]  # (npol, nchan, nauto)
        # Note: flag data available but not currently used - could mask flagged data
        # auto_flag = data['flag'][:, :, auto_mask] if data['flag'] is not None else None

        # Get unique antennas and times
        unique_ants = np.unique(auto_ant)
        unique_times = np.unique(auto_time)
        n_antennas = len(unique_ants)
        n_times = len(unique_times)

        logger.debug(f"Unique antennas: {n_antennas}, unique times: {n_times}")

        # Reshape into (n_times, n_antennas) array
        # Average over polarizations and channels
        # Note: These are autocorrelation amplitudes from the selected data column.
        # - DATA: Nominally in Jy but NOT flux-calibrated unless MS has been flux-calibrated
        # - CORRECTED_DATA: Calibrated visibilities (more likely to be flux-calibrated)
        # They are NOT Tsys values (requires noise diode calibration).
        tsys_array = np.full((n_times, n_antennas), np.nan)

        for i, ant in enumerate(unique_ants):
            ant_mask = auto_ant == ant
            ant_times = auto_time[ant_mask]
            ant_data = auto_data[:, :, ant_mask]  # (npol, nchan, n_ant_times)

            # Compute amplitude (average over pol and channel)
            # Use real part of autocorrelation (should be real for autocorr)
            amp = np.abs(ant_data).mean(axis=(0, 1))  # Average over pol and chan

            # Map to time indices
            for j, t in enumerate(ant_times):
                t_idx = np.searchsorted(unique_times, t)
                if t_idx < n_times:
                    tsys_array[t_idx, i] = amp[j]

        # Return autocorrelation amplitudes directly
        # Units: Nominally Jy (from selected column), flux-calibrated only if:
        #        - Using CORRECTED_DATA AND MS has been flux-calibrated
        # These are NOT Tsys values - proper Tsys requires noise diode calibration
        # The values are useful for relative monitoring but have no absolute calibration
        # unless flux-calibrated

        # Apply time averaging if requested
        if time_average > 1 and n_times > time_average:
            n_avg = n_times // time_average
            tsys_avg = np.zeros((n_avg, n_antennas))
            times_avg = np.zeros(n_avg)

            for i in range(n_avg):
                start = i * time_average
                end = start + time_average
                tsys_avg[i] = np.nanmean(tsys_array[start:end], axis=0)
                times_avg[i] = np.mean(unique_times[start:end])

            tsys_array = tsys_avg
            unique_times = times_avg
            n_times = n_avg

        # Get antenna names from ANTENNA table
        tb.open(str(ms_path / "ANTENNA"))
        all_ant_names = tb.getcol("NAME")
        tb.close()

        antenna_names = [
            all_ant_names[ant] if ant < len(all_ant_names) else f"Ant{ant}" for ant in unique_ants
        ]

        ms.close()

        return {
            "times": unique_times,
            "tsys": tsys_array,
            "antenna_ids": unique_ants,
            "antenna_names": antenna_names,
            "n_times": n_times,
            "n_antennas": n_antennas,
            "spw": spw,
            "ms_path": str(ms_path),
            "datacolumn_used": actual_column,
        }

    except Exception as e:
        logger.error(f"Failed to extract Tsys from {ms_path}: {e}")
        raise
    finally:
        try:
            ms.close()
        except Exception:
            pass


def plot_tsys_time_series(
    times: NDArray,
    tsys: NDArray,
    output: str | Path | None = None,
    config: FigureConfig | None = None,
    title: str = "System Temperature vs Time",
    antenna_names: list[str] | None = None,
    max_antennas: int = 16,
    tsys_warning: float = 50.0,
    tsys_bad: float = 100.0,
    show_median: bool = True,
) -> Figure:
    """Plot system temperature as function of time per antenna.

    Parameters
    ----------
    times :
        Time array (MJD or seconds since start)
    tsys :
        Tsys array, shape (Ntimes, Nantennas) or (Ntimes,) for single antenna
    output :
        Output file path
    config :
        Figure configuration
    title :
        Plot title
    antenna_names :
        List of antenna names
    max_antennas :
        Maximum antennas to display
    tsys_warning :
        Warning threshold (K)
    tsys_bad :
        Bad threshold (K)
    show_median :
        Show median line

    Returns
    -------
        matplotlib Figure object

    """
    _setup_matplotlib()
    import matplotlib.pyplot as plt

    if config is None:
        config = FigureConfig(style=PlotStyle.QUICKLOOK)

    times = np.asarray(times)
    tsys = np.asarray(tsys)

    # Convert time to minutes from start
    if times.max() > 1e5:  # Likely MJD
        t_plot = (times - times.min()) * 24 * 60  # minutes
        t_label = "Time (minutes)"
    else:
        t_plot = times / 60  # assume seconds, convert to minutes
        t_label = "Time (minutes)"

    # Handle different shapes
    if tsys.ndim == 1:
        tsys = tsys.reshape(-1, 1)

    n_times, n_antennas = tsys.shape
    n_plot = min(n_antennas, max_antennas)

    if antenna_names is None:
        antenna_names = [f"Ant {i}" for i in range(n_antennas)]

    fig, ax = plt.subplots(figsize=config.figsize, dpi=config.dpi)

    colors = plt.cm.tab20(np.linspace(0, 1, n_plot))

    for i in range(n_plot):
        ax.plot(
            t_plot, tsys[:, i], "-", color=colors[i], linewidth=1, alpha=0.7, label=antenna_names[i]
        )

    # Threshold lines
    ax.axhline(
        tsys_warning,
        color="orange",
        linestyle="--",
        alpha=0.7,
        linewidth=1.5,
        label=f"Warning ({tsys_warning} K)",
    )
    ax.axhline(
        tsys_bad, color="red", linestyle="--", alpha=0.7, linewidth=1.5, label=f"Bad ({tsys_bad} K)"
    )

    # Median line
    if show_median:
        median_tsys = np.nanmedian(tsys, axis=1)
        ax.plot(t_plot, median_tsys, "k-", linewidth=2, label="Array median")

    ax.set_xlabel(t_label, fontsize=config.effective_label_size)
    ax.set_ylabel("Autocorr Amplitude (Jy)", fontsize=config.effective_label_size)
    ax.set_title(title, fontsize=config.effective_title_size)
    ax.set_ylim(0, max(tsys_bad * 1.5, np.nanmax(tsys) * 1.1))

    # Legend outside if many antennas
    if n_plot > 8:
        ax.legend(
            fontsize=config.effective_tick_size - 2,
            ncol=2,
            loc="upper right",
            bbox_to_anchor=(1.0, 1.0),
        )
    else:
        ax.legend(fontsize=config.effective_tick_size - 1, loc="best")

    ax.grid(True, alpha=0.3)

    fig.tight_layout()

    if output:
        fig.savefig(output, dpi=config.dpi, bbox_inches="tight")
        logger.info(f"Saved Tsys time series: {output}")
        plt.close(fig)

    return fig


def plot_tsys_summary(
    tsys: NDArray,
    output: str | Path | None = None,
    config: FigureConfig | None = None,
    title: str = "System Temperature Summary",
    antenna_names: list[str] | None = None,
    tsys_warning: float = 50.0,
    tsys_bad: float = 100.0,
) -> Figure:
    """Plot Tsys summary statistics per antenna.

    Shows median, interquartile range, and outliers per antenna.

    Parameters
    ----------
    tsys :
        Tsys array, shape (Ntimes, Nantennas)
    output :
        Output file path
    config :
        Figure configuration
    title :
        Plot title
    antenna_names :
        List of antenna names
    tsys_warning :
        Warning threshold (K)
    tsys_bad :
        Bad threshold (K)

    Returns
    -------
        matplotlib Figure object

    """
    _setup_matplotlib()
    import matplotlib.pyplot as plt

    if config is None:
        config = FigureConfig(style=PlotStyle.QUICKLOOK)

    tsys = np.asarray(tsys)
    if tsys.ndim == 1:
        tsys = tsys.reshape(-1, 1)

    n_times, n_antennas = tsys.shape

    if antenna_names is None:
        antenna_names = [f"Ant {i}" for i in range(n_antennas)]

    # Compute statistics per antenna
    medians = np.nanmedian(tsys, axis=0)
    q25 = np.nanpercentile(tsys, 25, axis=0)
    q75 = np.nanpercentile(tsys, 75, axis=0)

    # Color by median value
    colors = []
    for med in medians:
        if med >= tsys_bad:
            colors.append("red")
        elif med >= tsys_warning:
            colors.append("orange")
        else:
            colors.append("green")

    fig, ax = plt.subplots(figsize=(max(config.figsize[0], n_antennas * 0.4), config.figsize[1]))

    x = np.arange(n_antennas)

    # Bar plot with error bars showing IQR
    ax.bar(x, medians, color=colors, alpha=0.7, edgecolor="black", linewidth=0.5)

    # Error bars for IQR
    ax.errorbar(
        x, medians, yerr=[medians - q25, q75 - medians], fmt="none", color="black", capsize=3
    )

    # Threshold lines (note: thresholds are in Jy, not K - these are relative values)
    ax.axhline(
        tsys_warning,
        color="orange",
        linestyle="--",
        alpha=0.7,
        linewidth=1.5,
        label=f"Warning ({tsys_warning} Jy)",
    )
    ax.axhline(
        tsys_bad,
        color="red",
        linestyle="--",
        alpha=0.7,
        linewidth=1.5,
        label=f"Bad ({tsys_bad} Jy)",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(
        antenna_names, rotation=45, ha="right", fontsize=config.effective_tick_size - 2
    )
    ax.set_xlabel("Antenna", fontsize=config.effective_label_size)
    ax.set_ylabel("Autocorr Amplitude (Jy)", fontsize=config.effective_label_size)
    ax.set_title(title, fontsize=config.effective_title_size)
    ax.legend(fontsize=config.effective_tick_size - 1, loc="upper right")
    ax.grid(True, alpha=0.3, axis="y")

    # Statistics annotation
    array_median = np.nanmedian(tsys)
    n_bad = np.sum(medians >= tsys_bad)
    n_warning = np.sum((medians >= tsys_warning) & (medians < tsys_bad))

    stats_text = f"Array median: {array_median:.1f} Jy\nBad antennas: {n_bad}\nWarning: {n_warning}"
    ax.text(
        0.02,
        0.98,
        stats_text,
        transform=ax.transAxes,
        verticalalignment="top",
        fontsize=config.effective_tick_size,
        bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"),
    )

    fig.tight_layout()

    if output:
        fig.savefig(output, dpi=config.dpi, bbox_inches="tight")
        logger.info(f"Saved Tsys summary: {output}")
        plt.close(fig)

    return fig


def plot_tsys_heatmap(
    times: NDArray,
    tsys: NDArray,
    output: str | Path | None = None,
    config: FigureConfig | None = None,
    title: str = "System Temperature Heatmap",
    antenna_names: list[str] | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    cmap: str = "YlOrRd",
) -> Figure:
    """Plot Tsys as a time-antenna heatmap.

    Useful for identifying time intervals or antennas with elevated Tsys.

    Parameters
    ----------
    times :
        Time array
    tsys :
        Tsys array, shape (Ntimes, Nantennas)
    output :
        Output file path
    config :
        Figure configuration
    title :
        Plot title
    antenna_names :
        List of antenna names
    vmin :
        Minimum for colorscale
    vmax :
        Maximum for colorscale
    cmap :
        Colormap

    Returns
    -------
        matplotlib Figure object

    """
    _setup_matplotlib()
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    if config is None:
        config = FigureConfig(style=PlotStyle.QUICKLOOK)

    times = np.asarray(times)
    tsys = np.asarray(tsys)

    if tsys.ndim == 1:
        tsys = tsys.reshape(-1, 1)

    n_times, n_antennas = tsys.shape

    if antenna_names is None:
        antenna_names = [f"Ant {i}" for i in range(n_antennas)]

    # Time axis
    if times.max() > 1e5:
        t_plot = (times - times.min()) * 24 * 60
        t_label = "Time (minutes)"
    else:
        t_plot = times / 60
        t_label = "Time (minutes)"

    # Default colorscale
    if vmin is None:
        vmin = np.nanpercentile(tsys, 5)
    if vmax is None:
        vmax = np.nanpercentile(tsys, 95)

    fig, ax = plt.subplots(figsize=config.figsize, dpi=config.dpi)

    extent = [t_plot[0], t_plot[-1], -0.5, n_antennas - 0.5]

    im = ax.imshow(
        tsys.T,
        aspect="auto",
        origin="lower",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        extent=extent,
    )

    ax.set_yticks(range(n_antennas))
    ax.set_yticklabels(antenna_names, fontsize=config.effective_tick_size - 2)
    ax.set_xlabel(t_label, fontsize=config.effective_label_size)
    ax.set_ylabel("Antenna", fontsize=config.effective_label_size)
    ax.set_title(title, fontsize=config.effective_title_size)

    # Colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("Autocorr Amplitude (Jy)", fontsize=config.effective_label_size)

    fig.tight_layout()

    if output:
        fig.savefig(output, dpi=config.dpi, bbox_inches="tight")
        logger.info(f"Saved Tsys heatmap: {output}")
        plt.close(fig)

    return fig


def plot_tsys_histogram(
    tsys: NDArray,
    output: str | Path | None = None,
    config: FigureConfig | None = None,
    title: str = "System Temperature Distribution",
    bins: int = 50,
    tsys_range: tuple[float, float] = (0, 150),
    tsys_warning: float = 50.0,
    tsys_bad: float = 100.0,
) -> Figure:
    """Plot histogram of all Tsys measurements.

    Parameters
    ----------
    tsys :
        Tsys array (any shape, will be flattened)
    output :
        Output file path
    config :
        Figure configuration
    title :
        Plot title
    bins :
        Number of histogram bins
    tsys_range :
        Range for histogram
    tsys_warning :
        Warning threshold
    tsys_bad :
        Bad threshold

    Returns
    -------
        matplotlib Figure object

    """
    _setup_matplotlib()
    import matplotlib.pyplot as plt

    if config is None:
        config = FigureConfig(style=PlotStyle.QUICKLOOK)

    tsys_flat = np.asarray(tsys).flatten()
    tsys_valid = tsys_flat[np.isfinite(tsys_flat)]

    fig, ax = plt.subplots(figsize=config.figsize, dpi=config.dpi)

    # Handle empty data case
    if len(tsys_valid) == 0:
        ax.text(
            0.5,
            0.5,
            "No valid autocorrelation data\n(all values are NaN or infinite)",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=config.effective_label_size,
            color="red",
        )
        ax.set_xlabel("Autocorr Amplitude (Jy)", fontsize=config.effective_label_size)
        ax.set_ylabel("Count", fontsize=config.effective_label_size)
        ax.set_title(title, fontsize=config.effective_title_size)

        if output:
            fig.savefig(output, dpi=config.dpi, bbox_inches="tight")
            logger.warning(f"Saved empty Tsys histogram (no valid data): {output}")
            plt.close(fig)

        return fig

    ax.hist(
        tsys_valid,
        bins=bins,
        range=tsys_range,
        color="steelblue",
        alpha=0.7,
        edgecolor="black",
        linewidth=0.5,
    )

    # Threshold lines (note: thresholds are in Jy, not K - these are relative values)
    ax.axvline(
        tsys_warning,
        color="orange",
        linestyle="--",
        linewidth=2,
        label=f"Warning ({tsys_warning} Jy)",
    )
    ax.axvline(tsys_bad, color="red", linestyle="--", linewidth=2, label=f"Bad ({tsys_bad} Jy)")

    # Statistics (safe since we checked len > 0 above)
    median = np.median(tsys_valid)
    mean = np.mean(tsys_valid)
    ax.axvline(median, color="green", linestyle="-", linewidth=2, label=f"Median={median:.1f} Jy")

    n_valid = len(tsys_valid)
    frac_warning = np.sum((tsys_valid >= tsys_warning) & (tsys_valid < tsys_bad)) / n_valid * 100
    frac_bad = np.sum(tsys_valid >= tsys_bad) / n_valid * 100

    stats_text = (
        f"N samples: {n_valid:,}\n"
        f"Mean: {mean:.1f} Jy\n"
        f"Median: {median:.1f} Jy\n"
        f"Warning: {frac_warning:.1f}%\n"
        f"Bad: {frac_bad:.1f}%"
    )
    ax.text(
        0.98,
        0.98,
        stats_text,
        transform=ax.transAxes,
        verticalalignment="top",
        horizontalalignment="right",
        fontsize=config.effective_tick_size,
        bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"),
    )

    ax.set_xlabel("Autocorr Amplitude (Jy)", fontsize=config.effective_label_size)
    ax.set_ylabel("Count", fontsize=config.effective_label_size)
    ax.set_title(title, fontsize=config.effective_title_size)
    ax.legend(fontsize=config.effective_tick_size - 1, loc="upper left")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()

    if output:
        fig.savefig(output, dpi=config.dpi, bbox_inches="tight")
        logger.info(f"Saved Tsys histogram: {output}")
        plt.close(fig)

    return fig


def plot_tsys_vs_elevation(
    elevations: NDArray,
    tsys: NDArray,
    output: str | Path | None = None,
    config: FigureConfig | None = None,
    title: str = "System Temperature vs Elevation",
    antenna_names: list[str] | None = None,
    show_model: bool = True,
    t_rx: float = 25.0,
    t_atm_zenith: float = 5.0,
) -> Figure:
    """Plot Tsys as function of elevation.

    Tsys typically increases at low elevation due to atmospheric absorption:
        Tsys(el) ≈ T_rx + T_atm / sin(el)

    Parameters
    ----------
    elevations :
        Elevation array in degrees
    tsys :
        Tsys array (can be 2D: Nsamples x Nantennas)
    output :
        Output file path
    config :
        Figure configuration
    title :
        Plot title
    antenna_names :
        Antenna names for legend
    show_model :
        Show simple atmospheric model
    t_rx :
        Receiver temperature for model (K)
    t_atm_zenith :
        Atmospheric contribution at zenith (K)

    Returns
    -------
        matplotlib Figure object

    """
    _setup_matplotlib()
    import matplotlib.pyplot as plt

    if config is None:
        config = FigureConfig(style=PlotStyle.QUICKLOOK)

    el = np.asarray(elevations).flatten()
    tsys = np.asarray(tsys)

    if tsys.ndim == 1:
        tsys = tsys.reshape(-1, 1)

    n_samples, n_antennas = tsys.shape

    if antenna_names is None:
        antenna_names = [f"Ant {i}" for i in range(n_antennas)]

    fig, ax = plt.subplots(figsize=config.figsize, dpi=config.dpi)

    colors = plt.cm.tab10(np.linspace(0, 1, min(n_antennas, 10)))

    for i in range(min(n_antennas, 10)):
        el_valid = el[: len(tsys[:, i])]
        ax.scatter(el_valid, tsys[:, i], s=5, alpha=0.3, c=[colors[i]], label=antenna_names[i])

    # Simple atmospheric model
    if show_model:
        el_model = np.linspace(10, 90, 100)
        tsys_model = t_rx + t_atm_zenith / np.sin(np.radians(el_model))
        ax.plot(
            el_model,
            tsys_model,
            "k--",
            linewidth=2,
            label=f"Model: T_rx={t_rx}K, T_atm={t_atm_zenith}K",
        )

    ax.set_xlabel("Elevation (degrees)", fontsize=config.effective_label_size)
    ax.set_ylabel("Autocorr Amplitude (Jy)", fontsize=config.effective_label_size)
    ax.set_title(title, fontsize=config.effective_title_size)
    ax.set_xlim(0, 90)
    ax.legend(fontsize=config.effective_tick_size - 2, ncol=2, loc="upper right")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()

    if output:
        fig.savefig(output, dpi=config.dpi, bbox_inches="tight")
        logger.info(f"Saved Tsys vs elevation: {output}")
        plt.close(fig)

    return fig


def detect_tsys_anomalies(
    tsys: NDArray,
    times: NDArray | None = None,
    sigma_threshold: float = 3.0,
) -> dict[str, NDArray]:
    """Detect anomalous Tsys measurements.

    Parameters
    ----------
    tsys :
        Tsys array (Ntimes, Nantennas)
    times :
        Optional time array
    sigma_threshold :
        Number of sigma for anomaly detection

    Returns
    -------
    Dictionary with anomaly information

    - 'mask'
        Boolean mask of anomalies
    - 'antenna_indices'
        Antenna indices with anomalies
    - 'time_indices'
        Time indices with anomalies
    - 'values'
        Anomalous Tsys values

    """
    tsys = np.asarray(tsys)
    if tsys.ndim == 1:
        tsys = tsys.reshape(-1, 1)

    # Compute robust statistics
    median = np.nanmedian(tsys)
    mad = np.nanmedian(np.abs(tsys - median))
    sigma = 1.4826 * mad  # MAD to sigma conversion

    # Find anomalies
    threshold = median + sigma_threshold * sigma
    mask = tsys > threshold

    time_idx, ant_idx = np.where(mask)

    return {
        "mask": mask,
        "antenna_indices": ant_idx,
        "time_indices": time_idx,
        "values": tsys[mask],
        "threshold": threshold,
        "median": median,
        "sigma": sigma,
    }


# Alias for convenience
plot_tsys_elevation = plot_tsys_vs_elevation
