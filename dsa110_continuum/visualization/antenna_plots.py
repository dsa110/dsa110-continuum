"""
Per-antenna diagnostic plotting utilities.

Provides functions for:
- Antenna flagging statistics
- Per-antenna gain/phase time series
- Antenna-based data quality summary
- System temperature monitoring
- Antenna response comparison

Essential for identifying problematic antennas and troubleshooting.
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


def _setup_matplotlib() -> None:
    """Configure matplotlib for headless operation."""
    import matplotlib

    matplotlib.use("Agg")


def plot_antenna_flagging_summary(
    flag_fractions: NDArray,
    antenna_names: list[str] | None = None,
    output: str | Path | None = None,
    config: FigureConfig | None = None,
    title: str = "Antenna Flagging Summary",
    threshold_warning: float = 0.2,
    threshold_bad: float = 0.5,
) -> Figure:
    """Plot bar chart of flagging fraction per antenna.

    Parameters
    ----------
    flag_fractions :
        Array of flagging fractions (0-1) per antenna
    antenna_names :
        List of antenna names/IDs
    output :
        Output file path
    config :
        Figure configuration
    title :
        Plot title
    threshold_warning :
        Fraction above which to mark as warning (yellow)
    threshold_bad :
        Fraction above which to mark as bad (red)

    Returns
    -------
        matplotlib Figure object

    """
    _setup_matplotlib()
    import matplotlib.pyplot as plt

    if config is None:
        config = FigureConfig(style=PlotStyle.QUICKLOOK)

    fracs = np.asarray(flag_fractions)
    n_ants = len(fracs)

    if antenna_names is None:
        antenna_names = [f"Ant {i}" for i in range(n_ants)]

    # Color based on flagging level
    colors = []
    for f in fracs:
        if f >= threshold_bad:
            colors.append("red")
        elif f >= threshold_warning:
            colors.append("orange")
        else:
            colors.append("green")

    fig, ax = plt.subplots(figsize=(max(config.figsize[0], n_ants * 0.3), config.figsize[1]))

    ax.bar(range(n_ants), fracs * 100, color=colors, edgecolor="black", linewidth=0.5)

    ax.set_xticks(range(n_ants))
    ax.set_xticklabels(
        antenna_names, rotation=45, ha="right", fontsize=config.effective_tick_size - 2
    )
    ax.set_xlabel("Antenna", fontsize=config.effective_label_size)
    ax.set_ylabel("Flagged (%)", fontsize=config.effective_label_size)
    ax.set_title(title, fontsize=config.effective_title_size)

    # Add threshold lines
    ax.axhline(
        threshold_warning * 100,
        color="orange",
        linestyle="--",
        alpha=0.7,
        label=f"Warning ({threshold_warning * 100:.0f}%)",
    )
    ax.axhline(
        threshold_bad * 100,
        color="red",
        linestyle="--",
        alpha=0.7,
        label=f"Bad ({threshold_bad * 100:.0f}%)",
    )
    ax.legend(fontsize=config.effective_tick_size, loc="upper right")

    ax.set_ylim(0, max(100, np.max(fracs) * 100 * 1.1))
    ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()

    if output:
        fig.savefig(output, dpi=config.dpi, bbox_inches="tight")
        logger.info(f"Saved antenna flagging summary: {output}")
        plt.close(fig)

    return fig


def plot_antenna_gain_time_series(
    times: NDArray,
    gains: NDArray,
    antenna_indices: NDArray,
    output: str | Path | None = None,
    config: FigureConfig | None = None,
    title: str = "Antenna Gain vs Time",
    antenna_names: list[str] | None = None,
    pol_index: int = 0,
    plot_amplitude: bool = True,
    plot_phase: bool = True,
    max_antennas: int = 16,
) -> Figure:
    """Plot gain amplitude and/or phase vs time for each antenna.

    Parameters
    ----------
    times :
        Time array (MJD or seconds)
    gains :
        Complex gain array (shape: Nrows, Nfreqs, Npols or Nrows, Npols)
    antenna_indices :
        Antenna index for each row
    output :
        Output file path
    config :
        Figure configuration
    title :
        Plot title
    antenna_names :
        List of antenna names
    pol_index :
        Polarization index to plot
    plot_amplitude :
        Plot amplitude panel
    plot_phase :
        Plot phase panel
    max_antennas :
        Maximum number of antennas to show

    Returns
    -------
        matplotlib Figure object

    """
    _setup_matplotlib()
    import matplotlib.pyplot as plt

    if config is None:
        config = FigureConfig(style=PlotStyle.QUICKLOOK)

    times = np.asarray(times)
    gains = np.asarray(gains)
    antenna_indices = np.asarray(antenna_indices)

    # Extract gains for requested polarization
    if gains.ndim == 3:
        # (Nrows, Nfreqs, Npols) - average over frequency
        gain_vals = np.nanmean(gains[:, :, pol_index], axis=1)
    elif gains.ndim == 2:
        # (Nrows, Npols)
        gain_vals = gains[:, pol_index]
    else:
        gain_vals = gains

    # Time in minutes from start
    t_min = (times - times.min()) * 24 * 60 if times.max() > 1000 else times / 60

    unique_ants = np.unique(antenna_indices)[:max_antennas]
    n_ants = len(unique_ants)

    n_panels = int(plot_amplitude) + int(plot_phase)
    if n_panels == 0:
        plot_amplitude = True
        n_panels = 1

    fig, axes = plt.subplots(
        n_panels, 1, figsize=(config.figsize[0], config.figsize[1] * n_panels), sharex=True
    )

    if n_panels == 1:
        axes = [axes]

    colors = plt.cm.tab20(np.linspace(0, 1, n_ants))

    ax_idx = 0

    # Amplitude panel
    if plot_amplitude:
        ax = axes[ax_idx]
        for i, ant in enumerate(unique_ants):
            mask = antenna_indices == ant
            amp = np.abs(gain_vals[mask])
            t = t_min[mask]

            label = (
                antenna_names[ant] if antenna_names and ant < len(antenna_names) else f"Ant {ant}"
            )
            ax.plot(t, amp, ".", markersize=3, alpha=0.7, color=colors[i], label=label)

        ax.set_ylabel("Gain Amplitude", fontsize=config.effective_label_size)
        ax.legend(fontsize=config.effective_tick_size - 2, ncol=4, loc="upper right")
        ax.grid(True, alpha=0.3)
        ax_idx += 1

    # Phase panel
    if plot_phase:
        ax = axes[ax_idx]
        for i, ant in enumerate(unique_ants):
            mask = antenna_indices == ant
            phase = np.angle(gain_vals[mask], deg=True)
            t = t_min[mask]

            label = (
                antenna_names[ant] if antenna_names and ant < len(antenna_names) else f"Ant {ant}"
            )
            ax.plot(t, phase, ".", markersize=3, alpha=0.7, color=colors[i], label=label)

        ax.set_ylabel("Gain Phase (deg)", fontsize=config.effective_label_size)
        ax.set_ylim(-180, 180)
        ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
        if not plot_amplitude:  # Only add legend if not already shown
            ax.legend(fontsize=config.effective_tick_size - 2, ncol=4, loc="upper right")
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time (minutes)", fontsize=config.effective_label_size)
    fig.suptitle(title, fontsize=config.effective_title_size)
    fig.tight_layout()

    if output:
        fig.savefig(output, dpi=config.dpi, bbox_inches="tight")
        logger.info(f"Saved antenna gain time series: {output}")
        plt.close(fig)

    return fig


def plot_antenna_gain_spectrum(
    freqs_ghz: NDArray,
    gains: NDArray,
    antenna_indices: NDArray,
    output: str | Path | None = None,
    config: FigureConfig | None = None,
    title: str = "Antenna Gain vs Frequency",
    antenna_names: list[str] | None = None,
    pol_index: int = 0,
    max_antennas: int = 16,
    time_average: bool = True,
) -> Figure:
    """Plot gain amplitude vs frequency for each antenna (bandpass shape).

    Parameters
    ----------
    freqs_ghz :
        Frequency array in GHz
    gains :
        Complex gain array (shape: Nrows, Nfreqs, Npols)
    antenna_indices :
        Antenna index for each row
    output :
        Output file path
    config :
        Figure configuration
    title :
        Plot title
    antenna_names :
        List of antenna names
    pol_index :
        Polarization index to plot
    max_antennas :
        Maximum number of antennas to show
    time_average :
        Average over time for each antenna

    Returns
    -------
        matplotlib Figure object

    """
    _setup_matplotlib()
    import matplotlib.pyplot as plt

    if config is None:
        config = FigureConfig(style=PlotStyle.QUICKLOOK)

    freqs = np.asarray(freqs_ghz)
    gains = np.asarray(gains)
    antenna_indices = np.asarray(antenna_indices)

    unique_ants = np.unique(antenna_indices)[:max_antennas]
    n_ants = len(unique_ants)

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(config.figsize[0], config.figsize[1] * 2), sharex=True
    )

    colors = plt.cm.tab20(np.linspace(0, 1, n_ants))

    for i, ant in enumerate(unique_ants):
        mask = antenna_indices == ant

        if gains.ndim == 3:
            ant_gains = gains[mask, :, pol_index]  # (Ntimes, Nfreqs)
        elif gains.ndim == 2:
            ant_gains = gains[mask, :]  # (Ntimes, Nfreqs)
        else:
            continue

        if time_average and ant_gains.shape[0] > 1:
            # Average over time
            amp = np.nanmean(np.abs(ant_gains), axis=0)
            phase = np.angle(np.nanmean(ant_gains, axis=0), deg=True)
        else:
            amp = np.abs(ant_gains[0])
            phase = np.angle(ant_gains[0], deg=True)

        label = antenna_names[ant] if antenna_names and ant < len(antenna_names) else f"Ant {ant}"

        ax1.plot(freqs, amp, "-", linewidth=1, alpha=0.7, color=colors[i], label=label)
        ax2.plot(freqs, phase, "-", linewidth=1, alpha=0.7, color=colors[i])

    ax1.set_ylabel("Gain Amplitude", fontsize=config.effective_label_size)
    ax1.legend(fontsize=config.effective_tick_size - 2, ncol=4, loc="upper right")
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel("Frequency (GHz)", fontsize=config.effective_label_size)
    ax2.set_ylabel("Gain Phase (deg)", fontsize=config.effective_label_size)
    ax2.set_ylim(-180, 180)
    ax2.axhline(0, color="gray", linestyle="--", alpha=0.5)
    ax2.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=config.effective_title_size)
    fig.tight_layout()

    if output:
        fig.savefig(output, dpi=config.dpi, bbox_inches="tight")
        logger.info(f"Saved antenna gain spectrum: {output}")
        plt.close(fig)

    return fig


def plot_antenna_statistics_grid(
    statistics: dict,
    output: str | Path | None = None,
    config: FigureConfig | None = None,
    title: str = "Antenna Statistics Summary",
) -> Figure:
    """Plot multi-panel summary of antenna statistics.

    Parameters
    ----------
    statistics :
        Dictionary with keys 'antenna_names', 'flag_fraction',
        'mean_amplitude', 'rms_phase', 'snr' (all arrays of length Nants)
    output :
        Output file path
    config :
        Figure configuration
    title :
        Overall title

    Returns
    -------
        matplotlib Figure object

    """
    _setup_matplotlib()
    import matplotlib.pyplot as plt

    if config is None:
        config = FigureConfig(style=PlotStyle.QUICKLOOK)

    antenna_names = statistics.get(
        "antenna_names", [f"Ant {i}" for i in range(len(statistics.get("flag_fraction", [])))]
    )
    n_ants = len(antenna_names)

    # Determine which panels to show
    panels = []
    if "flag_fraction" in statistics:
        panels.append(("flag_fraction", "Flagged (%)", lambda x: x * 100))
    if "mean_amplitude" in statistics:
        panels.append(("mean_amplitude", "Mean Amplitude", lambda x: x))
    if "rms_phase" in statistics:
        panels.append(("rms_phase", "RMS Phase (deg)", lambda x: x))
    if "snr" in statistics:
        panels.append(("snr", "SNR", lambda x: x))

    n_panels = len(panels)
    if n_panels == 0:
        logger.warning("No statistics to plot")
        return None

    fig, axes = plt.subplots(
        n_panels,
        1,
        figsize=(max(config.figsize[0], n_ants * 0.3), config.figsize[1] * n_panels),
        sharex=True,
    )

    if n_panels == 1:
        axes = [axes]

    x = range(n_ants)

    for ax, (key, ylabel, transform) in zip(axes, panels):
        values = transform(np.asarray(statistics[key]))

        # Color by relative value
        norm_vals = (values - np.nanmin(values)) / (np.nanmax(values) - np.nanmin(values) + 1e-10)

        # For flagging, red is bad (high), for SNR, red is bad (low)
        if key in ["flag_fraction"]:
            colors = plt.cm.RdYlGn_r(norm_vals)
        else:
            colors = plt.cm.RdYlGn(norm_vals)

        ax.bar(x, values, color=colors, edgecolor="black", linewidth=0.5)
        ax.set_ylabel(ylabel, fontsize=config.effective_label_size)
        ax.grid(True, alpha=0.3, axis="y")

        # Annotate outliers
        median = np.nanmedian(values)
        std = np.nanstd(values)
        for i, v in enumerate(values):
            if np.abs(v - median) > 2 * std:
                ax.annotate(
                    antenna_names[i],
                    (i, v),
                    textcoords="offset points",
                    xytext=(0, 5),
                    ha="center",
                    fontsize=config.effective_tick_size - 2,
                    color="red",
                )

    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(
        antenna_names, rotation=45, ha="right", fontsize=config.effective_tick_size - 2
    )
    axes[-1].set_xlabel("Antenna", fontsize=config.effective_label_size)

    fig.suptitle(title, fontsize=config.effective_title_size)
    fig.tight_layout()

    if output:
        fig.savefig(output, dpi=config.dpi, bbox_inches="tight")
        logger.info(f"Saved antenna statistics grid: {output}")
        plt.close(fig)

    return fig


def compute_antenna_statistics_from_ms(
    ms_path: str | Path,
    field_id: int | None = None,
) -> dict:
    """Compute per-antenna statistics from a Measurement Set.

    Parameters
    ----------
    ms_path :
        Path to Measurement Set
    field_id :
        Field ID to analyze (None = all)
    ms_path : Union[str, Path]
    field_id: Optional[int] :
         (Default value = None)

    Returns
    -------
        Dictionary with antenna statistics suitable for plot_antenna_statistics_grid

    """
    try:
        from casacore.tables import table
    except ImportError:
        raise ImportError("casacore required. Install with: pip install python-casacore")

    ms_path = Path(ms_path)
    stats = {}

    # Get antenna names
    with table(str(ms_path / "ANTENNA"), readonly=True, ack=False) as ant_tb:
        stats["antenna_names"] = list(ant_tb.getcol("NAME"))

    n_ants = len(stats["antenna_names"])
    flag_counts = np.zeros(n_ants)
    total_counts = np.zeros(n_ants)
    amp_sums = np.zeros(n_ants)
    amp_counts = np.zeros(n_ants)

    with table(str(ms_path), readonly=True, ack=False) as tb:
        ant1 = tb.getcol("ANTENNA1")
        ant2 = tb.getcol("ANTENNA2")
        flags = tb.getcol("FLAG")

        if "DATA" in tb.colnames():
            data = tb.getcol("DATA")
        elif "CORRECTED_DATA" in tb.colnames():
            data = tb.getcol("CORRECTED_DATA")
        else:
            data = None

        # Aggregate per antenna
        for row in range(len(ant1)):
            a1, a2 = ant1[row], ant2[row]
            row_flags = flags[row]
            n_flagged = np.sum(row_flags)
            n_total = row_flags.size

            for ant in [a1, a2]:
                flag_counts[ant] += n_flagged
                total_counts[ant] += n_total

                if data is not None:
                    row_data = data[row]
                    valid = ~row_flags
                    if np.any(valid):
                        amp_sums[ant] += np.sum(np.abs(row_data[valid]))
                        amp_counts[ant] += np.sum(valid)

    # Compute statistics
    with np.errstate(divide="ignore", invalid="ignore"):
        stats["flag_fraction"] = np.where(total_counts > 0, flag_counts / total_counts, 0)
        stats["mean_amplitude"] = np.where(amp_counts > 0, amp_sums / amp_counts, 0)

    return stats
