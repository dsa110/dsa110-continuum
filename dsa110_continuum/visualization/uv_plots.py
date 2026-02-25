"""
UV coverage and visibility plotting utilities.

Provides functions for:
- UV coverage scatter plots (u vs v)
- UV density maps / histograms
- Baseline length distribution
- Visibility amplitude vs UV distance
- Visibility phase vs time

Essential for assessing data quality and understanding (u,v) plane sampling.
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


def plot_uv_coverage(
    u_lambda: NDArray,
    v_lambda: NDArray,
    output: str | Path | None = None,
    config: FigureConfig | None = None,
    title: str = "UV Coverage",
    show_conjugate: bool = True,
    color_by_time: NDArray | None = None,
    color_by_baseline: NDArray | None = None,
    marker_size: float = 0.5,
    alpha: float = 0.3,
) -> Figure:
    """Plot UV coverage showing (u,v) point distribution.

    Parameters
    ----------
    u_lambda :
        U coordinates in wavelengths (shape: Nblts or Nvis)
    v_lambda :
        V coordinates in wavelengths (shape: Nblts or Nvis)
    output :
        Output file path (None for interactive display)
    config :
        Figure configuration
    title :
        Plot title
    show_conjugate :
        If True, also plot (-u, -v) conjugate points
    color_by_time :
        Optional time array for color-coding points
    color_by_baseline :
        Optional baseline index array for color-coding
    marker_size :
        Size of scatter points
    alpha :
        Transparency of points

    Returns
    -------
        matplotlib Figure object

    """
    _setup_matplotlib()
    import matplotlib.pyplot as plt

    if config is None:
        config = FigureConfig(style=PlotStyle.QUICKLOOK)

    u = np.asarray(u_lambda).flatten()
    v = np.asarray(v_lambda).flatten()

    fig, ax = plt.subplots(figsize=config.figsize, dpi=config.dpi)

    # Determine color mapping
    if color_by_time is not None:
        c = np.asarray(color_by_time).flatten()
        cmap = "viridis"
        cbar_label = "Time (s)"
    elif color_by_baseline is not None:
        c = np.asarray(color_by_baseline).flatten()
        cmap = "tab20"
        cbar_label = "Baseline Index"
    else:
        c = "cyan"
        cmap = None
        cbar_label = None

    # Plot UV points
    scatter = ax.scatter(
        u / 1e3,
        v / 1e3,
        c=c,
        cmap=cmap,
        s=marker_size,
        alpha=alpha,
        edgecolors="none",
    )

    # Plot conjugate points if requested
    if show_conjugate:
        ax.scatter(
            -u / 1e3,
            -v / 1e3,
            c=c if isinstance(c, str) else c,
            cmap=cmap,
            s=marker_size,
            alpha=alpha,
            edgecolors="none",
        )

    # Colorbar if color-coded
    if cbar_label and cmap:
        cbar = fig.colorbar(scatter, ax=ax)
        cbar.set_label(cbar_label, fontsize=config.effective_label_size)

    ax.set_xlabel("u (kλ)", fontsize=config.effective_label_size)
    ax.set_ylabel("v (kλ)", fontsize=config.effective_label_size)
    ax.set_title(title, fontsize=config.effective_title_size)

    # Equal aspect ratio for proper UV representation
    ax.set_aspect("equal")

    # Add grid
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color="gray", linestyle="-", alpha=0.5, linewidth=0.5)
    ax.axvline(0, color="gray", linestyle="-", alpha=0.5, linewidth=0.5)

    fig.tight_layout()

    if output:
        fig.savefig(output, dpi=config.dpi, bbox_inches="tight")
        logger.info(f"Saved UV coverage plot: {output}")
        plt.close(fig)

    return fig


def plot_uv_density(
    u_lambda: NDArray,
    v_lambda: NDArray,
    output: str | Path | None = None,
    config: FigureConfig | None = None,
    title: str = "UV Density",
    bins: int = 100,
    show_colorbar: bool = True,
    log_scale: bool = True,
) -> Figure:
    """Plot UV density as a 2D histogram heatmap.

    Parameters
    ----------
    u_lambda :
        U coordinates in wavelengths
    v_lambda :
        V coordinates in wavelengths
    output :
        Output file path
    config :
        Figure configuration
    title :
        Plot title
    bins :
        Number of bins for 2D histogram
    show_colorbar :
        Show colorbar
    log_scale :
        Use log scale for density

    Returns
    -------
        matplotlib Figure object

    """
    _setup_matplotlib()
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    if config is None:
        config = FigureConfig(style=PlotStyle.QUICKLOOK)

    u = np.asarray(u_lambda).flatten() / 1e3  # Convert to kλ
    v = np.asarray(v_lambda).flatten() / 1e3

    # Include conjugate points
    u_all = np.concatenate([u, -u])
    v_all = np.concatenate([v, -v])

    fig, ax = plt.subplots(figsize=config.figsize, dpi=config.dpi)

    # 2D histogram
    norm = LogNorm() if log_scale else None
    h, xedges, yedges, im = ax.hist2d(
        u_all,
        v_all,
        bins=bins,
        cmap="hot",
        norm=norm,
    )

    if show_colorbar:
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label(
            "Counts (log)" if log_scale else "Counts", fontsize=config.effective_label_size
        )

    ax.set_xlabel("u (kλ)", fontsize=config.effective_label_size)
    ax.set_ylabel("v (kλ)", fontsize=config.effective_label_size)
    ax.set_title(title, fontsize=config.effective_title_size)
    ax.set_aspect("equal")

    fig.tight_layout()

    if output:
        fig.savefig(output, dpi=config.dpi, bbox_inches="tight")
        logger.info(f"Saved UV density plot: {output}")
        plt.close(fig)

    return fig


def plot_baseline_distribution(
    u_lambda: NDArray,
    v_lambda: NDArray,
    output: str | Path | None = None,
    config: FigureConfig | None = None,
    title: str = "Baseline Length Distribution",
    bins: int = 50,
    show_cumulative: bool = True,
) -> Figure:
    """Plot histogram of baseline lengths.

    Parameters
    ----------
    u_lambda :
        U coordinates in wavelengths
    v_lambda :
        V coordinates in wavelengths
    output :
        Output file path
    config :
        Figure configuration
    title :
        Plot title
    bins :
        Number of histogram bins
    show_cumulative :
        Show cumulative distribution overlay

    Returns
    -------
        matplotlib Figure object

    """
    _setup_matplotlib()
    import matplotlib.pyplot as plt

    if config is None:
        config = FigureConfig(style=PlotStyle.QUICKLOOK)

    u = np.asarray(u_lambda).flatten()
    v = np.asarray(v_lambda).flatten()

    # Calculate baseline lengths
    uvdist = np.sqrt(u**2 + v**2) / 1e3  # kλ

    if show_cumulative:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(config.figsize[0] * 2, config.figsize[1]))
    else:
        fig, ax1 = plt.subplots(figsize=config.figsize)

    # Histogram
    ax1.hist(uvdist, bins=bins, color="steelblue", alpha=0.7, edgecolor="black", linewidth=0.5)
    ax1.set_xlabel("Baseline Length (kλ)", fontsize=config.effective_label_size)
    ax1.set_ylabel("Count", fontsize=config.effective_label_size)
    ax1.set_title(title, fontsize=config.effective_title_size)
    ax1.grid(True, alpha=0.3)

    # Statistics annotation
    stats_text = (
        f"Min: {uvdist.min():.1f} kλ\n"
        f"Max: {uvdist.max():.1f} kλ\n"
        f"Median: {np.median(uvdist):.1f} kλ\n"
        f"N: {len(uvdist):,}"
    )
    ax1.text(
        0.95,
        0.95,
        stats_text,
        transform=ax1.transAxes,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"),
        fontsize=config.effective_tick_size,
    )

    # Cumulative distribution
    if show_cumulative:
        sorted_uvdist = np.sort(uvdist)
        cumulative = np.arange(1, len(sorted_uvdist) + 1) / len(sorted_uvdist)
        ax2.plot(sorted_uvdist, cumulative, "b-", linewidth=2)
        ax2.set_xlabel("Baseline Length (kλ)", fontsize=config.effective_label_size)
        ax2.set_ylabel("Cumulative Fraction", fontsize=config.effective_label_size)
        ax2.set_title("Cumulative Distribution", fontsize=config.effective_title_size)
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1)

    fig.tight_layout()

    if output:
        fig.savefig(output, dpi=config.dpi, bbox_inches="tight")
        logger.info(f"Saved baseline distribution plot: {output}")
        plt.close(fig)

    return fig


def plot_visibility_amplitude_vs_uvdist(
    u_lambda: NDArray,
    v_lambda: NDArray,
    visibility: NDArray,
    output: str | Path | None = None,
    config: FigureConfig | None = None,
    title: str = "Visibility Amplitude vs UV Distance",
    show_binned: bool = True,
    n_bins: int = 50,
    pol_index: int = 0,
    freq_index: int | None = None,
) -> Figure:
    """Plot visibility amplitude as function of UV distance.

    Parameters
    ----------
    u_lambda :
        U coordinates in wavelengths (shape: Nblts or Nblts,Nfreqs)
    v_lambda :
        V coordinates in wavelengths
    visibility :
        Complex visibility array (shape: Nblts, Nfreqs, Npols or similar)
    output :
        Output file path
    config :
        Figure configuration
    title :
        Plot title
    show_binned :
        Show binned median with error bars
    n_bins :
        Number of bins for binned plot
    pol_index :
        Polarization index to plot
    freq_index :
        Frequency index to plot (None = average all)

    Returns
    -------
        matplotlib Figure object

    """
    _setup_matplotlib()
    import matplotlib.pyplot as plt

    if config is None:
        config = FigureConfig(style=PlotStyle.QUICKLOOK)

    u = np.asarray(u_lambda).flatten()
    v = np.asarray(v_lambda).flatten()
    vis = np.asarray(visibility)

    # Extract amplitude
    if vis.ndim == 1:
        amp = np.abs(vis)
    elif vis.ndim == 2:
        # (Nblts, Nfreqs) or (Nblts, Npols)
        if freq_index is not None:
            amp = np.abs(vis[:, freq_index])
        else:
            amp = np.nanmean(np.abs(vis), axis=1)
    elif vis.ndim == 3:
        # (Nblts, Nfreqs, Npols)
        if freq_index is not None:
            amp = np.abs(vis[:, freq_index, pol_index])
        else:
            amp = np.nanmean(np.abs(vis[:, :, pol_index]), axis=1)
    else:
        # Higher dimensions - flatten appropriately
        amp = np.nanmean(np.abs(vis.reshape(vis.shape[0], -1)), axis=1)

    # Calculate UV distance
    uvdist = np.sqrt(u**2 + v**2) / 1e3  # kλ

    # Ensure same length
    min_len = min(len(uvdist), len(amp))
    uvdist = uvdist[:min_len]
    amp = amp[:min_len]

    # Filter valid data
    mask = np.isfinite(amp) & np.isfinite(uvdist) & (amp > 0)
    uvdist = uvdist[mask]
    amp = amp[mask]

    fig, ax = plt.subplots(figsize=config.figsize, dpi=config.dpi)

    # Scatter plot
    ax.scatter(uvdist, amp, s=1, alpha=0.2, c="gray", label="Individual")

    # Binned median with error bars
    if show_binned and len(uvdist) > n_bins:
        bin_edges = np.linspace(uvdist.min(), uvdist.max(), n_bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        bin_medians = []
        bin_stds = []

        for i in range(n_bins):
            in_bin = (uvdist >= bin_edges[i]) & (uvdist < bin_edges[i + 1])
            if np.sum(in_bin) > 0:
                bin_medians.append(np.median(amp[in_bin]))
                bin_stds.append(np.std(amp[in_bin]))
            else:
                bin_medians.append(np.nan)
                bin_stds.append(np.nan)

        bin_medians = np.array(bin_medians)
        bin_stds = np.array(bin_stds)

        ax.errorbar(
            bin_centers,
            bin_medians,
            yerr=bin_stds,
            fmt="o-",
            color="red",
            markersize=4,
            linewidth=1.5,
            capsize=2,
            label="Binned Median ± σ",
        )

    ax.set_xlabel("UV Distance (kλ)", fontsize=config.effective_label_size)
    ax.set_ylabel("Amplitude (Jy)", fontsize=config.effective_label_size)
    ax.set_title(title, fontsize=config.effective_title_size)
    ax.legend(fontsize=config.effective_tick_size)
    ax.grid(True, alpha=0.3)

    # Log scale for amplitude often useful
    if np.nanmin(amp) > 0:
        ax.set_yscale("log")

    fig.tight_layout()

    if output:
        fig.savefig(output, dpi=config.dpi, bbox_inches="tight")
        logger.info(f"Saved amplitude vs uvdist plot: {output}")
        plt.close(fig)

    return fig


def plot_visibility_phase_vs_time(
    times: NDArray,
    visibility: NDArray,
    output: str | Path | None = None,
    config: FigureConfig | None = None,
    title: str = "Visibility Phase vs Time",
    baseline_indices: NDArray | None = None,
    antenna_names: list[str] | None = None,
    pol_index: int = 0,
    freq_index: int | None = None,
    max_baselines: int = 10,
) -> Figure:
    """Plot visibility phase as function of time.

    Parameters
    ----------
    times :
        Time array (MJD or seconds)
    visibility :
        Complex visibility array
    output :
        Output file path
    config :
        Figure configuration
    title :
        Plot title
    baseline_indices :
        Array of baseline indices for color-coding
    antenna_names :
        List of antenna names for labeling
    pol_index :
        Polarization index to plot
    freq_index :
        Frequency index to plot (None = average all)
    max_baselines :
        Maximum number of baselines to show

    Returns
    -------
        matplotlib Figure object

    """
    _setup_matplotlib()
    import matplotlib.pyplot as plt

    if config is None:
        config = FigureConfig(style=PlotStyle.QUICKLOOK)

    times = np.asarray(times)
    vis = np.asarray(visibility)

    # Extract phase
    if vis.ndim == 1:
        phase = np.angle(vis, deg=True)
    elif vis.ndim == 2:
        if freq_index is not None:
            phase = np.angle(vis[:, freq_index], deg=True)
        else:
            # Average phase over frequency (via complex mean)
            phase = np.angle(np.nanmean(vis, axis=1), deg=True)
    elif vis.ndim == 3:
        if freq_index is not None:
            phase = np.angle(vis[:, freq_index, pol_index], deg=True)
        else:
            phase = np.angle(np.nanmean(vis[:, :, pol_index], axis=1), deg=True)
    else:
        phase = np.angle(np.nanmean(vis.reshape(vis.shape[0], -1), axis=1), deg=True)

    # Convert time to relative minutes
    t_min = (times - times.min()) * 24 * 60 if times.max() > 1000 else times / 60

    fig, ax = plt.subplots(figsize=config.figsize, dpi=config.dpi)

    if baseline_indices is not None:
        unique_bl = np.unique(baseline_indices)[:max_baselines]
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_bl)))

        for i, bl in enumerate(unique_bl):
            mask = baseline_indices == bl
            label = (
                f"BL {bl}"
                if antenna_names is None
                else antenna_names[bl]
                if bl < len(antenna_names)
                else f"BL {bl}"
            )
            ax.scatter(t_min[mask], phase[mask], s=2, alpha=0.5, c=[colors[i]], label=label)

        ax.legend(fontsize=config.effective_tick_size - 2, ncol=2, loc="upper right")
    else:
        ax.scatter(t_min, phase, s=1, alpha=0.3, c="blue")

    ax.set_xlabel("Time (minutes)", fontsize=config.effective_label_size)
    ax.set_ylabel("Phase (degrees)", fontsize=config.effective_label_size)
    ax.set_title(title, fontsize=config.effective_title_size)
    ax.set_ylim(-180, 180)
    ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()

    if output:
        fig.savefig(output, dpi=config.dpi, bbox_inches="tight")
        logger.info(f"Saved phase vs time plot: {output}")
        plt.close(fig)

    return fig


def plot_visibility_amplitude_vs_time(
    times: NDArray,
    visibility: NDArray,
    output: str | Path | None = None,
    config: FigureConfig | None = None,
    title: str = "Visibility Amplitude vs Time",
    baseline_indices: NDArray | None = None,
    pol_index: int = 0,
    freq_index: int | None = None,
    max_baselines: int = 10,
    normalize: bool = False,
) -> Figure:
    """Plot visibility amplitude as function of time.

    Parameters
    ----------
    times :
        Time array (MJD or seconds)
    visibility :
        Complex visibility array
    output :
        Output file path
    config :
        Figure configuration
    title :
        Plot title
    baseline_indices :
        Array of baseline indices for color-coding
    pol_index :
        Polarization index to plot
    freq_index :
        Frequency index to plot (None = average all)
    max_baselines :
        Maximum number of baselines to show
    normalize :
        If True, normalize each baseline to its median

    Returns
    -------
        matplotlib Figure object

    """
    _setup_matplotlib()
    import matplotlib.pyplot as plt

    if config is None:
        config = FigureConfig(style=PlotStyle.QUICKLOOK)

    times = np.asarray(times)
    vis = np.asarray(visibility)

    # Extract amplitude
    if vis.ndim == 1:
        amp = np.abs(vis)
    elif vis.ndim == 2:
        if freq_index is not None:
            amp = np.abs(vis[:, freq_index])
        else:
            amp = np.nanmean(np.abs(vis), axis=1)
    elif vis.ndim == 3:
        if freq_index is not None:
            amp = np.abs(vis[:, freq_index, pol_index])
        else:
            amp = np.nanmean(np.abs(vis[:, :, pol_index]), axis=1)
    else:
        amp = np.nanmean(np.abs(vis.reshape(vis.shape[0], -1)), axis=1)

    # Convert time to relative minutes
    t_min = (times - times.min()) * 24 * 60 if times.max() > 1000 else times / 60

    fig, ax = plt.subplots(figsize=config.figsize, dpi=config.dpi)

    if baseline_indices is not None:
        unique_bl = np.unique(baseline_indices)[:max_baselines]
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_bl)))

        for i, bl in enumerate(unique_bl):
            mask = baseline_indices == bl
            bl_amp = amp[mask].copy()

            if normalize:
                med = np.nanmedian(bl_amp)
                if med > 0:
                    bl_amp = bl_amp / med

            ax.scatter(t_min[mask], bl_amp, s=2, alpha=0.5, c=[colors[i]], label=f"BL {bl}")

        ax.legend(fontsize=config.effective_tick_size - 2, ncol=2, loc="upper right")
    else:
        if normalize:
            med = np.nanmedian(amp)
            if med > 0:
                amp = amp / med
        ax.scatter(t_min, amp, s=1, alpha=0.3, c="blue")

    ylabel = "Normalized Amplitude" if normalize else "Amplitude (Jy)"
    ax.set_xlabel("Time (minutes)", fontsize=config.effective_label_size)
    ax.set_ylabel(ylabel, fontsize=config.effective_label_size)
    ax.set_title(title, fontsize=config.effective_title_size)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()

    if output:
        fig.savefig(output, dpi=config.dpi, bbox_inches="tight")
        logger.info(f"Saved amplitude vs time plot: {output}")
        plt.close(fig)

    return fig


def extract_uv_from_ms(ms_path: str | Path) -> dict:
    """Extract UV coordinates and visibility data from a Measurement Set.

    Parameters
    ----------
    ms_path :
        Path to Measurement Set
    ms_path : Union[str, Path]

    Returns
    -------
    Dictionary with keys
        'u_lambda', 'v_lambda', 'w_lambda', 'visibility',
    Dictionary with keys
        'u_lambda', 'v_lambda', 'w_lambda', 'visibility',
        'time', 'antenna1', 'antenna2', 'freq_hz'

    """
    try:
        from casacore.tables import table
    except ImportError:
        raise ImportError("casacore required. Install with: pip install python-casacore")

    ms_path = Path(ms_path)
    result = {}

    with table(str(ms_path), readonly=True, ack=False) as tb:
        # Get UVW in meters
        uvw = tb.getcol("UVW")  # (Nrows, 3)
        result["time"] = tb.getcol("TIME")
        result["antenna1"] = tb.getcol("ANTENNA1")
        result["antenna2"] = tb.getcol("ANTENNA2")

        # Get DATA or CORRECTED_DATA
        if "CORRECTED_DATA" in tb.colnames():
            result["visibility"] = tb.getcol("CORRECTED_DATA")
        elif "DATA" in tb.colnames():
            result["visibility"] = tb.getcol("DATA")
        else:
            result["visibility"] = None

    # Get frequencies from SPECTRAL_WINDOW subtable
    with table(str(ms_path / "SPECTRAL_WINDOW"), readonly=True, ack=False) as spw_tb:
        freq_hz = spw_tb.getcol("CHAN_FREQ")[0]  # First SPW
        result["freq_hz"] = freq_hz

    # Convert UVW from meters to wavelengths (using center frequency)
    c_mps = 299792458.0
    center_freq = np.mean(freq_hz)
    wavelength_m = c_mps / center_freq

    result["u_lambda"] = uvw[:, 0] / wavelength_m
    result["v_lambda"] = uvw[:, 1] / wavelength_m
    result["w_lambda"] = uvw[:, 2] / wavelength_m
    result["uvw_m"] = uvw

    return result
