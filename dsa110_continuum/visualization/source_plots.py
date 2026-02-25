"""
Source analysis plotting utilities.

Provides functions for:
- Lightcurves (flux vs time)
- Spectra (flux vs frequency)
- Source comparison plots (ASKAP vs reference catalogs)

Adapted from:
- VAST/vastfast/plot.py (lightcurves)
- ASKAP-continuum-validation/report.py (validation plots)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from astropy.time import Time
    from matplotlib.figure import Figure
    from numpy.typing import NDArray

from dsa110_contimg.core.visualization.config import FigureConfig, PlotStyle

logger = logging.getLogger(__name__)


def _setup_matplotlib() -> None:
    """Configure matplotlib for headless operation."""
    import matplotlib

    matplotlib.use("Agg")


def plot_lightcurve(
    flux: NDArray,
    times: NDArray | Time | list,
    errors: NDArray | None = None,
    output: str | Path | None = None,
    config: FigureConfig | None = None,
    title: str = "",
    flux_unit: str = "mJy",
    source_name: str | None = None,
    show_mean: bool = False,
    show_std: bool = False,
    annotations: list[str] | None = None,
) -> Figure:
    """Plot a source lightcurve with error bars.

    Adapted from VAST/vastfast/plot.py plot_lightcurve().

    Parameters
    ----------
    flux :
        Flux density values
    times :
        Timestamps (MJD, datetime, or astropy Time)
    errors :
        Flux density errors (optional)
    output :
        Output file path
    config :
        Figure configuration
    title :
        Plot title
    flux_unit :
        Unit label for flux axis
    source_name :
        Source name for labeling
    show_mean :
        Show horizontal line at mean flux
    show_std :
        Show shaded region ±1σ around mean
    annotations :
        Optional list of text annotations (one per point)

    Returns
    -------
        matplotlib Figure object

    """
    _setup_matplotlib()
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    from astropy.time import Time

    if config is None:
        config = FigureConfig(style=PlotStyle.QUICKLOOK)

    # Convert times to datetime for plotting
    if not isinstance(times, Time):
        times = Time(times)
    times.format = "datetime64"

    # Convert flux to display units
    flux = np.asarray(flux)
    if flux_unit == "mJy" and np.nanmax(flux) < 0.1:
        flux = flux * 1e3
    elif flux_unit == "Jy" and np.nanmax(flux) > 100:
        flux = flux / 1e3
        flux_unit = "mJy"

    if errors is not None:
        errors = np.asarray(errors)
        if flux_unit == "mJy":
            errors = errors * 1e3

    fig, ax = plt.subplots(figsize=config.figsize, dpi=config.dpi)

    if errors is not None:
        ax.errorbar(
            times.value,
            flux,
            yerr=errors,
            fmt="o",
            color="black",
            markersize=config.marker_size,
            alpha=config.alpha,
            capsize=3,
        )
    else:
        ax.plot(
            times.value,
            flux,
            "o-",
            color="black",
            markersize=config.marker_size,
            alpha=config.alpha,
        )

    ax.set_xlabel("Time (UTC)", fontsize=config.effective_label_size)
    ax.set_ylabel(f"Peak Flux Density ({flux_unit}/beam)", fontsize=config.effective_label_size)

    # Optional reference lines
    if show_mean:
        mean_flux = np.nanmean(flux)
        ax.axhline(
            mean_flux, color="red", linestyle="--", alpha=0.7, label=f"Mean: {mean_flux:.2f}"
        )

    if show_std:
        mean_flux = np.nanmean(flux)
        std_flux = np.nanstd(flux)
        ax.fill_between(
            times.value,
            mean_flux - std_flux,
            mean_flux + std_flux,
            color="gray",
            alpha=0.2,
            label=f"±1σ: {std_flux:.2f}",
        )

    if show_mean or show_std:
        ax.legend(fontsize=config.effective_tick_size, loc="best")

    # Optional annotations
    if annotations is not None and len(annotations) == len(times):
        for i, (t, f, label) in enumerate(zip(times.value, flux, annotations)):
            if label:  # Skip empty annotations
                ax.annotate(
                    label,
                    xy=(t, f),
                    xytext=(5, 5),
                    textcoords="offset points",
                    fontsize=config.effective_tick_size - 1,
                    alpha=0.7,
                )

    # Format time axis
    date_formatter = mdates.DateFormatter("%Y-%b-%d\n%H:%M")
    ax.xaxis.set_major_formatter(date_formatter)

    # Auto-adjust tick spacing
    if len(times) > 1:
        span_hours = (times[-1] - times[0]).to_value("hour")
        if span_hours > 24:
            ax.xaxis.set_major_locator(mdates.DayLocator())
        elif span_hours > 4:
            ax.xaxis.set_major_locator(mdates.HourLocator(interval=max(1, int(span_hours / 6))))

    fig.autofmt_xdate(rotation=15)

    # Title
    if title:
        ax.set_title(title, fontsize=config.effective_title_size)
    elif source_name:
        ax.set_title(f"Lightcurve: {source_name}", fontsize=config.effective_title_size)

    if config.grid:
        ax.grid(True, alpha=0.3)

    fig.tight_layout()

    if output:
        fig.savefig(output, dpi=config.dpi, bbox_inches="tight")
        logger.info(f"Saved lightcurve: {output}")
        plt.close(fig)

    return fig


def plot_spectrum(
    flux: NDArray,
    freq_ghz: NDArray,
    errors: NDArray | None = None,
    output: str | Path | None = None,
    config: FigureConfig | None = None,
    title: str = "",
    source_name: str | None = None,
    fit_powerlaw: bool = False,
) -> Figure:
    """Plot a source spectrum (flux vs frequency).

    Parameters
    ----------
    flux :
        Flux density values in Jy
    freq_ghz :
        Frequencies in GHz
    errors :
        Flux density errors (optional)
    output :
        Output file path
    config :
        Figure configuration
    title :
        Plot title
    source_name :
        Source name for labeling
    fit_powerlaw :
        Fit and overlay a power-law model

    Returns
    -------
        matplotlib Figure object

    """
    _setup_matplotlib()
    import matplotlib.pyplot as plt

    if config is None:
        config = FigureConfig(style=PlotStyle.QUICKLOOK)

    flux = np.asarray(flux)
    freq_ghz = np.asarray(freq_ghz)

    # Convert to mJy if values are small
    flux_unit = "Jy"
    if np.nanmax(flux) < 0.1:
        flux = flux * 1e3
        flux_unit = "mJy"
        if errors is not None:
            errors = np.asarray(errors) * 1e3

    fig, ax = plt.subplots(figsize=config.figsize, dpi=config.dpi)

    if errors is not None:
        ax.errorbar(
            freq_ghz,
            flux,
            yerr=errors,
            fmt="o",
            color="black",
            markersize=config.marker_size,
            alpha=config.alpha,
            capsize=3,
        )
    else:
        ax.plot(
            freq_ghz,
            flux,
            "o",
            color="black",
            markersize=config.marker_size,
            alpha=config.alpha,
        )

    # Power-law fit
    if fit_powerlaw and len(flux) > 2:
        try:
            mask = np.isfinite(flux) & np.isfinite(freq_ghz) & (flux > 0)
            if np.sum(mask) > 2:
                from scipy.optimize import curve_fit

                def powerlaw(f, S0, alpha):
                    return S0 * (f / freq_ghz[mask].mean()) ** alpha

                popt, _ = curve_fit(powerlaw, freq_ghz[mask], flux[mask])

                freq_fit = np.linspace(freq_ghz.min(), freq_ghz.max(), 100)
                flux_fit = powerlaw(freq_fit, *popt)

                ax.plot(freq_fit, flux_fit, "--", color="red", label=f"α = {popt[1]:.2f}")
                ax.legend(fontsize=config.effective_tick_size)
        except Exception as e:
            logger.warning(f"Power-law fit failed: {e}")

    ax.set_xlabel("Frequency (GHz)", fontsize=config.effective_label_size)
    ax.set_ylabel(f"Flux Density ({flux_unit})", fontsize=config.effective_label_size)

    # Log-log scale often useful for spectra
    ax.set_xscale("log")
    ax.set_yscale("log")

    if title:
        ax.set_title(title, fontsize=config.effective_title_size)
    elif source_name:
        ax.set_title(f"Spectrum: {source_name}", fontsize=config.effective_title_size)

    if config.grid:
        ax.grid(True, alpha=0.3, which="both")

    fig.tight_layout()

    if output:
        fig.savefig(output, dpi=config.dpi, bbox_inches="tight")
        logger.info(f"Saved spectrum: {output}")
        plt.close(fig)

    return fig


def plot_source_comparison(
    measured_flux: NDArray,
    reference_flux: NDArray,
    errors_measured: NDArray | None = None,
    errors_reference: NDArray | None = None,
    output: str | Path | None = None,
    config: FigureConfig | None = None,
    title: str = "Flux Comparison",
    measured_label: str = "DSA-110",
    reference_label: str = "Reference",
    show_ratio: bool = True,
) -> Figure:
    """Plot measured vs reference flux comparison.

    Adapted from ASKAP-continuum-validation/report.py.

    Parameters
    ----------
    measured_flux :
        Measured flux densities
    reference_flux :
        Reference catalog flux densities
    errors_measured :
        Measured flux errors
    errors_reference :
        Reference flux errors
    output :
        Output file path
    config :
        Figure configuration
    title :
        Plot title
    measured_label :
        Label for measured data
    reference_label :
        Label for reference data
    show_ratio :
        Show flux ratio histogram

    Returns
    -------
        matplotlib Figure object

    """
    _setup_matplotlib()
    import matplotlib.pyplot as plt
    from scipy import stats

    if config is None:
        config = FigureConfig(style=PlotStyle.QUICKLOOK)

    measured_flux = np.asarray(measured_flux)
    reference_flux = np.asarray(reference_flux)

    # Filter valid data
    mask = np.isfinite(measured_flux) & np.isfinite(reference_flux)
    mask &= (measured_flux > 0) & (reference_flux > 0)

    meas = measured_flux[mask]
    ref = reference_flux[mask]

    if show_ratio:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(config.figsize[0] * 2, config.figsize[1]))
    else:
        fig, ax1 = plt.subplots(figsize=config.figsize)

    # Scatter plot
    ax1.scatter(ref, meas, alpha=config.alpha, s=config.marker_size * 5, edgecolors="none")

    # 1:1 line
    lims = [min(ref.min(), meas.min()) * 0.8, max(ref.max(), meas.max()) * 1.2]
    ax1.plot(lims, lims, "k--", alpha=0.5, label="1:1")

    # Linear fit
    if len(meas) > 5:
        slope, intercept, r_value, _, _ = stats.linregress(np.log10(ref), np.log10(meas))
        fit_x = np.array(lims)
        fit_y = 10 ** (slope * np.log10(fit_x) + intercept)
        ax1.plot(
            fit_x, fit_y, "r-", alpha=0.7, label=f"Fit: slope={slope:.2f}, r²={r_value**2:.2f}"
        )

    ax1.set_xlim(lims)
    ax1.set_ylim(lims)
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel(f"{reference_label} Flux (Jy)", fontsize=config.effective_label_size)
    ax1.set_ylabel(f"{measured_label} Flux (Jy)", fontsize=config.effective_label_size)
    ax1.legend(fontsize=config.effective_tick_size)
    ax1.set_title(title, fontsize=config.effective_title_size)

    # Ratio histogram
    if show_ratio:
        ratio = meas / ref

        median_ratio = np.median(ratio)
        mad = np.median(np.abs(ratio - median_ratio))

        ax2.hist(ratio, bins=50, alpha=0.7, edgecolor="black", linewidth=0.5)
        ax2.axvline(1.0, color="black", linestyle="--", label="1.0")
        ax2.axvline(median_ratio, color="red", linestyle="-", label=f"Median: {median_ratio:.3f}")
        ax2.axvline(median_ratio - mad, color="red", linestyle=":", alpha=0.7)
        ax2.axvline(
            median_ratio + mad, color="red", linestyle=":", alpha=0.7, label=f"MAD: {mad:.3f}"
        )

        ax2.set_xlabel(
            f"Flux Ratio ({measured_label}/{reference_label})", fontsize=config.effective_label_size
        )
        ax2.set_ylabel("Count", fontsize=config.effective_label_size)
        ax2.legend(fontsize=config.effective_tick_size)
        ax2.set_title("Flux Ratio Distribution", fontsize=config.effective_title_size)

    fig.tight_layout()

    if output:
        fig.savefig(output, dpi=config.dpi, bbox_inches="tight")
        logger.info(f"Saved comparison plot: {output}")
        plt.close(fig)

    return fig


def _calculate_lightcurve_statistics(
    mjd: NDArray, flux_jy: NDArray, flux_err_jy: NDArray
) -> dict:
    """Calculate lightcurve statistics."""
    if len(flux_jy) == 0:
        return {}
    
    return {
        "n_points": len(flux_jy),
        "mean": np.mean(flux_jy),
        "min": np.min(flux_jy),
        "max": np.max(flux_jy),
        "std": np.std(flux_jy),
        "mean_error": np.mean(flux_err_jy) if flux_err_jy is not None else 0.0,
        "time_span": mjd[-1] - mjd[0] if len(mjd) > 1 else 0.0,
    }


def plot_monitoring_lightcurve(
    mjd: NDArray,
    flux_jy: NDArray,
    flux_err_jy: NDArray,
    source_id: str,
    source_name: str | None = None,
    ra_deg: float | None = None,
    dec_deg: float | None = None,
    output: str | Path | None = None,
    config: FigureConfig | None = None,
) -> Figure:
    """Create a monitoring lightcurve figure matching the UI appearance.

    Features:
    - Blue line (#0080FF) for flux measurements
    - Error bars with shaded fill (20% opacity)
    - Statistics panel layout
    - Professional styling matching Dagster UI components

    Parameters
    ----------
    mjd :
        Modified Julian Date values
    flux_jy :
        Flux density values in Jy
    flux_err_jy :
        Flux density errors in Jy
    source_id :
        Source identifier
    source_name :
        Source name (optional)
    ra_deg :
        Right Ascension in degrees (optional)
    dec_deg :
        Declination in degrees (optional)
    output :
        Output file path
    config :
        Figure configuration (optional)

    Returns
    -------
        matplotlib Figure object

    Example
    -------
    >>> from dsa110_contimg.core.visualization import (
    ...     get_photometry_data,
    ...     plot_monitoring_lightcurve
    ... )
    >>>
    >>> # 1. Fetch data
    >>> data = get_photometry_data(pipeline_db_path, "0834+555")
    >>>
    >>> # 2. Plot
    >>> mjd = [d['mjd'] for d in data]
    >>> flux = [d['flux_jy'] for d in data]
    >>> errs = [d['flux_err_jy'] for d in data]
    >>>
    >>> plot_monitoring_lightcurve(
    ...     mjd=mjd,
    ...     flux_jy=flux,
    ...     flux_err_jy=errs,
    ...     source_id="0834+555",
    ...     output="lightcurve.png"
    ... )
    """
    _setup_matplotlib()
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    if config is None:
        config = FigureConfig(style=PlotStyle.PUBLICATION)

    # Chart.js color scheme from UI
    CHART_BLUE = "#0080FF"
    ERROR_FILL_ALPHA = 0.2
    ERROR_LINE_ALPHA = 0.33

    # Create figure with custom layout
    # Match UI dialog proportions: wider for statistics panel
    fig = plt.figure(figsize=(12, 8), dpi=config.dpi)
    gs = GridSpec(3, 1, figure=fig, height_ratios=[0.15, 0.7, 0.15], hspace=0.3)

    # Top panel: Source information
    ax_info = fig.add_subplot(gs[0])
    ax_info.axis("off")

    # Format coordinates helper
    def format_coord(value, is_ra):
        if value is None:
            return "N/A"
        if is_ra:
            hours = int(value / 15)
            minutes = int(((value / 15) % 1) * 60)
            seconds = ((((value / 15) % 1) * 60) % 1) * 60
            return f"{hours:02d}:{minutes:02d}:{seconds:05.2f}"
        else:
            sign = "+" if value >= 0 else "-"
            abs_value = abs(value)
            degrees = int(abs_value)
            minutes = int((abs_value % 1) * 60)
            seconds = ((abs_value % 1) * 60) % 1 * 60
            return f"{sign}{degrees:02d}:{minutes:02d}:{seconds:05.2f}"

    ra_str = format_coord(ra_deg, True)
    dec_str = format_coord(dec_deg, False)
    display_name = source_name or source_id

    # Source info box
    info_text = (
        f"SOURCE ID: {source_id}\n"
        f"RA: {ra_str}  |  DEC: {dec_str}  |  NAME: {display_name}"
    )
    ax_info.text(
        0.5,
        0.5,
        info_text,
        transform=ax_info.transAxes,
        fontsize=11,
        fontfamily="monospace",
        ha="center",
        va="center",
        bbox=dict(boxstyle="round", facecolor="#f5f5f5", edgecolor="#cccccc", pad=10),
    )

    # Middle panel: Main chart
    ax_chart = fig.add_subplot(gs[1])

    # Plot error fill region first
    upper_bounds = flux_jy + flux_err_jy
    lower_bounds = flux_jy - flux_err_jy

    ax_chart.fill_between(
        mjd,
        lower_bounds,
        upper_bounds,
        color=CHART_BLUE,
        alpha=ERROR_FILL_ALPHA,
        label="Error bounds",
        zorder=1,
    )

    # Plot error bounds as dashed lines
    ax_chart.plot(
        mjd,
        upper_bounds,
        "--",
        color=CHART_BLUE,
        alpha=ERROR_LINE_ALPHA,
        linewidth=1.0,
        label="Flux + Error",
        zorder=2,
    )
    ax_chart.plot(
        mjd,
        lower_bounds,
        "--",
        color=CHART_BLUE,
        alpha=ERROR_LINE_ALPHA,
        linewidth=1.0,
        label="Flux - Error",
        zorder=2,
    )

    # Plot main flux line
    ax_chart.plot(
        mjd,
        flux_jy,
        "-o",
        color=CHART_BLUE,
        linewidth=2.0,
        markersize=6,
        markerfacecolor=CHART_BLUE,
        markeredgecolor="white",
        markeredgewidth=1,
        label="Flux Density",
        zorder=3,
    )

    # Chart styling
    ax_chart.set_xlabel("Modified Julian Date (MJD)", fontsize=12, fontweight="normal")
    ax_chart.set_ylabel("Flux Density (Jy)", fontsize=12, fontweight="normal")
    ax_chart.set_title(
        f"Flux Density vs Time ({len(mjd)} point{'s' if len(mjd) != 1 else ''})",
        fontsize=14,
        fontweight="normal",
        pad=10,
    )

    # Grid
    ax_chart.grid(True, alpha=0.3, linestyle="-", linewidth=0.5)

    # Legend
    ax_chart.legend(
        loc="upper right", fontsize=9, framealpha=0.9, fancybox=True, shadow=False
    )

    # Format axes
    ax_chart.tick_params(axis="both", which="major", labelsize=10)

    # Bottom panel: Statistics
    ax_stats = fig.add_subplot(gs[2])
    ax_stats.axis("off")

    # Calculate statistics
    stats = _calculate_lightcurve_statistics(mjd, flux_jy, flux_err_jy)

    if stats:
        # Create statistics grid layout
        stat_items = [
            ("Data Points", f"{stats['n_points']}"),
            ("Mean Flux", f"{stats['mean']:.4f} Jy"),
            ("Min Flux", f"{stats['min']:.4f} Jy"),
            ("Max Flux", f"{stats['max']:.4f} Jy"),
            ("Std Dev", f"{stats['std']:.4f} Jy"),
            ("Mean Error", f"{stats['mean_error']:.4f} Jy"),
            ("Time Span", f"{stats['time_span']:.2f} days"),
        ]

        # Layout statistics in a grid (2 rows, 4 columns)
        n_cols = 4
        n_rows = 2
        cell_width = 1.0 / n_cols
        cell_height = 1.0 / n_rows

        for idx, (label, value) in enumerate(stat_items):
            row = idx // n_cols
            col = idx % n_cols

            x_center = col * cell_width + cell_width / 2
            y_center = 1.0 - (row * cell_height + cell_height / 2)

            # Label (small, uppercase)
            ax_stats.text(
                x_center,
                y_center + 0.08,
                label.upper(),
                transform=ax_stats.transAxes,
                fontsize=8,
                ha="center",
                va="bottom",
                color="#666666",
                fontweight="normal",
            )

            # Value (larger, monospace)
            ax_stats.text(
                x_center,
                y_center - 0.02,
                value,
                transform=ax_stats.transAxes,
                fontsize=11,
                fontfamily="monospace",
                ha="center",
                va="top",
                color="#000000",
                fontweight="normal",
            )

        # Add subtle grid lines for statistics panel
        for i in range(n_cols + 1):
            ax_stats.plot(
                [i * cell_width, i * cell_width],
                [0, 1],
                color="#e0e0e0",
                linewidth=0.5,
                transform=ax_stats.transAxes,
                zorder=0,
            )

    # Final layout adjustments
    fig.tight_layout()

    if output:
        fig.savefig(output, dpi=config.dpi, bbox_inches="tight", facecolor="white")
        logger.info(f"Saved monitoring lightcurve: {output}")
        plt.close(fig)

    return fig
