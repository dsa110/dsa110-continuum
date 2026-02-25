"""
Convergence and optimization plotting utilities for iterative algorithms.

Provides visualization for:
- Self-calibration convergence (SNR, chi-squared, RMS vs iteration)
- Clean algorithm convergence (residual RMS, model flux vs iteration)
- Optimization metrics (chi-squared, regularization terms)
- Per-antenna solution quality evolution

Essential for:
- Validating self-calibration is converging (not diverging)
- Determining optimal stopping point
- Diagnosing failed calibration runs
- Comparing different calibration strategies

Works with data structures from:
- dsa110_contimg.core.calibration.selfcal.SelfCalResult
- dsa110_contimg.core.calibration.selfcal.SelfCalIterationResult
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

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


@dataclass
class ConvergenceData:
    """Container for convergence tracking data."""

    iterations: list[int]
    chi_squared: list[float] | None = None
    snr: list[float] | None = None
    rms: list[float] | None = None
    peak_flux: list[float] | None = None
    model_flux: list[float] | None = None
    antenna_snr_median: list[float] | None = None
    antenna_snr_min: list[float] | None = None
    phase_scatter: list[float] | None = None
    amp_scatter: list[float] | None = None
    mode: list[str] | None = None
    solint: list[str] | None = None


def extract_convergence_from_selfcal_result(result: Any) -> ConvergenceData:
    """Extract convergence data from SelfCalResult.

    Parameters
    ----------
    result :
        SelfCalResult object from selfcal_ms()

    Returns
    -------
        ConvergenceData container

    """
    # Handle both actual SelfCalResult and dict-like objects
    iterations = []
    chi_squared = []
    snr = []
    rms = []
    peak_flux = []
    antenna_snr_median = []
    antenna_snr_min = []
    phase_scatter = []
    amp_scatter = []
    mode = []
    solint = []

    # Get iterations list
    if hasattr(result, "iterations"):
        iter_list = result.iterations
    elif isinstance(result, dict) and "iterations" in result:
        iter_list = result["iterations"]
    else:
        raise ValueError("Result must have 'iterations' attribute")

    for iter_result in iter_list:
        if hasattr(iter_result, "iteration"):
            iterations.append(iter_result.iteration)
            chi_squared.append(getattr(iter_result, "chi_squared", 0.0))
            snr.append(getattr(iter_result, "snr", 0.0))
            rms.append(getattr(iter_result, "rms", 0.0))
            peak_flux.append(getattr(iter_result, "peak_flux", 0.0))
            antenna_snr_median.append(getattr(iter_result, "antenna_snr_median", 0.0))
            antenna_snr_min.append(getattr(iter_result, "antenna_snr_min", 0.0))
            phase_scatter.append(getattr(iter_result, "phase_scatter_deg", 0.0))
            amp_scatter.append(getattr(iter_result, "amp_scatter_frac", 0.0))
            mode.append(str(getattr(iter_result, "mode", "unknown")))
            solint.append(getattr(iter_result, "solint", ""))
        elif isinstance(iter_result, dict):
            iterations.append(iter_result.get("iteration", len(iterations)))
            chi_squared.append(iter_result.get("chi_squared", 0.0))
            snr.append(iter_result.get("snr", 0.0))
            rms.append(iter_result.get("rms", 0.0))
            peak_flux.append(iter_result.get("peak_flux", 0.0))
            antenna_snr_median.append(iter_result.get("antenna_snr_median", 0.0))
            antenna_snr_min.append(iter_result.get("antenna_snr_min", 0.0))
            phase_scatter.append(iter_result.get("phase_scatter_deg", 0.0))
            amp_scatter.append(iter_result.get("amp_scatter_frac", 0.0))
            mode.append(iter_result.get("mode", "unknown"))
            solint.append(iter_result.get("solint", ""))

    return ConvergenceData(
        iterations=iterations,
        chi_squared=chi_squared if any(chi_squared) else None,
        snr=snr if any(snr) else None,
        rms=rms if any(rms) else None,
        peak_flux=peak_flux if any(peak_flux) else None,
        antenna_snr_median=antenna_snr_median if any(antenna_snr_median) else None,
        antenna_snr_min=antenna_snr_min if any(antenna_snr_min) else None,
        phase_scatter=phase_scatter if any(phase_scatter) else None,
        amp_scatter=amp_scatter if any(amp_scatter) else None,
        mode=mode if any(mode) else None,
        solint=solint if any(solint) else None,
    )


def plot_selfcal_convergence(
    data: ConvergenceData | Any,
    output: str | Path | None = None,
    config: FigureConfig | None = None,
    title: str = "Self-Calibration Convergence",
    show_chi_squared: bool = True,
    show_snr: bool = True,
    show_rms: bool = True,
    highlight_best: bool = True,
) -> Figure:
    """Plot self-calibration convergence metrics.

    Creates a multi-panel figure showing:
    - Chi-squared vs iteration (should decrease)
    - SNR vs iteration (should increase)
    - RMS noise vs iteration (should decrease)

    Parameters
    ----------
    data :
        ConvergenceData or SelfCalResult object
    output :
        Output file path
    config :
        Figure configuration
    title :
        Overall figure title
    show_chi_squared :
        Show chi-squared panel
    show_snr :
        Show SNR panel
    show_rms :
        Show RMS panel
    highlight_best :
        Highlight the best iteration
    data : Union[ConvergenceData, Any]
    output : Optional[Union[str, Path]]
         (Default value = None)
    config: Optional[FigureConfig] :
         (Default value = None)

    Returns
    -------
        matplotlib Figure object

    """
    _setup_matplotlib()
    import matplotlib.pyplot as plt

    if config is None:
        config = FigureConfig(style=PlotStyle.QUICKLOOK)

    # Convert SelfCalResult to ConvergenceData if needed
    if not isinstance(data, ConvergenceData):
        data = extract_convergence_from_selfcal_result(data)

    # Determine number of panels
    panels = []
    if show_chi_squared and data.chi_squared:
        panels.append(("chi_squared", "Chi-squared", data.chi_squared, "red"))
    if show_snr and data.snr:
        panels.append(("snr", "SNR", data.snr, "green"))
    if show_rms and data.rms:
        panels.append(("rms", "RMS (Jy/beam)", data.rms, "blue"))

    n_panels = len(panels)
    if n_panels == 0:
        logger.warning("No data to plot")
        fig, ax = plt.subplots(figsize=config.figsize)
        ax.text(0.5, 0.5, "No convergence data available", ha="center", va="center")
        return fig

    fig, axes = plt.subplots(
        n_panels, 1, figsize=(config.figsize[0], config.figsize[1] * n_panels / 2), sharex=True
    )

    if n_panels == 1:
        axes = [axes]

    iterations = data.iterations

    # Find best iteration (highest SNR or lowest chi-squared)
    best_iter = None
    if highlight_best:
        if data.snr:
            best_iter = iterations[np.argmax(data.snr)]
        elif data.chi_squared:
            valid_chi = [c for c in data.chi_squared if c > 0]
            if valid_chi:
                best_iter = iterations[data.chi_squared.index(min(valid_chi))]

    # Color-code by mode if available
    mode_colors = {}
    if data.mode:
        unique_modes = list(set(data.mode))
        color_map = {
            "phase": "blue",
            "ap": "orange",
            "SelfCalMode.PHASE": "blue",
            "SelfCalMode.AMPLITUDE_PHASE": "orange",
        }
        mode_colors = {m: color_map.get(m, "gray") for m in unique_modes}

    for ax, (name, ylabel, values, color) in zip(axes, panels):
        if data.mode:
            # Plot with mode color-coding
            for i, (it, val, m) in enumerate(zip(iterations, values, data.mode)):
                c = mode_colors.get(m, color)
                ax.plot([it], [val], "o", color=c, markersize=8)
                if i > 0:
                    ax.plot(
                        [iterations[i - 1], it],
                        [values[i - 1], val],
                        "-",
                        color=c,
                        linewidth=1.5,
                        alpha=0.7,
                    )
        else:
            ax.plot(iterations, values, "o-", color=color, markersize=8, linewidth=1.5)

        # Highlight best iteration
        if highlight_best and best_iter is not None:
            idx = iterations.index(best_iter)
            ax.axvline(best_iter, color="gold", linestyle="--", linewidth=2, alpha=0.7)
            ax.scatter(
                [best_iter],
                [values[idx]],
                s=150,
                facecolors="none",
                edgecolors="gold",
                linewidth=3,
                zorder=5,
            )

        ax.set_ylabel(ylabel, fontsize=config.effective_label_size)
        ax.grid(True, alpha=0.3)

        # Improvement annotation
        if len(values) > 1 and values[0] != 0:
            if name == "chi_squared":
                improvement = values[0] / values[-1] if values[-1] > 0 else float("inf")
                ax.text(
                    0.98,
                    0.95,
                    f"Improvement: {improvement:.2f}×",
                    transform=ax.transAxes,
                    ha="right",
                    va="top",
                    fontsize=config.effective_tick_size,
                )
            elif name == "snr":
                improvement = values[-1] / values[0] if values[0] > 0 else float("inf")
                ax.text(
                    0.98,
                    0.95,
                    f"Improvement: {improvement:.2f}×",
                    transform=ax.transAxes,
                    ha="right",
                    va="top",
                    fontsize=config.effective_tick_size,
                )

    axes[-1].set_xlabel("Iteration", fontsize=config.effective_label_size)
    axes[-1].set_xticks(iterations)

    # Add mode legend if applicable
    if data.mode and mode_colors:
        from matplotlib.lines import Line2D

        legend_elements = [
            Line2D(
                [0],
                [0],
                marker="o",
                color=c,
                label=m.replace("SelfCalMode.", ""),
                markersize=8,
                linestyle="-",
            )
            for m, c in mode_colors.items()
        ]
        axes[0].legend(handles=legend_elements, fontsize=config.effective_tick_size, loc="best")

    fig.suptitle(title, fontsize=config.effective_title_size)
    fig.tight_layout()

    if output:
        fig.savefig(output, dpi=config.dpi, bbox_inches="tight")
        logger.info(f"Saved convergence plot: {output}")
        plt.close(fig)

    return fig


def plot_antenna_solution_quality(
    data: ConvergenceData | Any,
    output: str | Path | None = None,
    config: FigureConfig | None = None,
    title: str = "Antenna Solution Quality",
    snr_threshold: float = 3.0,
) -> Figure:
    """Plot per-antenna solution quality metrics vs iteration.

    Parameters
    ----------
    data :
        ConvergenceData or SelfCalResult object
    output :
        Output file path
    config :
        Figure configuration
    title :
        Plot title
    snr_threshold :
        SNR threshold for highlighting
    data : Union[ConvergenceData, Any]
    output : Optional[Union[str, Path]]
         (Default value = None)
    config: Optional[FigureConfig] :
         (Default value = None)

    Returns
    -------
        matplotlib Figure object

    """
    _setup_matplotlib()
    import matplotlib.pyplot as plt

    if config is None:
        config = FigureConfig(style=PlotStyle.QUICKLOOK)

    if not isinstance(data, ConvergenceData):
        data = extract_convergence_from_selfcal_result(data)

    fig, axes = plt.subplots(2, 1, figsize=config.figsize, sharex=True)

    iterations = data.iterations

    # Panel 1: Antenna SNR
    if data.antenna_snr_median and data.antenna_snr_min:
        ax1 = axes[0]
        ax1.fill_between(
            iterations,
            data.antenna_snr_min,
            data.antenna_snr_median,
            alpha=0.3,
            color="blue",
            label="Min-Median range",
        )
        ax1.plot(iterations, data.antenna_snr_median, "o-", color="blue", label="Median SNR")
        ax1.plot(iterations, data.antenna_snr_min, "o--", color="red", label="Min SNR")
        ax1.axhline(
            snr_threshold,
            color="red",
            linestyle=":",
            alpha=0.7,
            label=f"Threshold ({snr_threshold})",
        )
        ax1.set_ylabel("Per-Antenna SNR", fontsize=config.effective_label_size)
        ax1.legend(fontsize=config.effective_tick_size - 1)
        ax1.grid(True, alpha=0.3)

    # Panel 2: Phase/Amplitude scatter
    ax2 = axes[1]
    if data.phase_scatter:
        ax2.plot(iterations, data.phase_scatter, "o-", color="purple", label="Phase scatter (°)")
    if data.amp_scatter:
        ax2_twin = ax2.twinx()
        ax2_twin.plot(
            iterations,
            [a * 100 for a in data.amp_scatter],
            "s-",
            color="orange",
            label="Amp scatter (%)",
        )
        ax2_twin.set_ylabel(
            "Amplitude Scatter (%)", fontsize=config.effective_label_size, color="orange"
        )

    ax2.set_xlabel("Iteration", fontsize=config.effective_label_size)
    ax2.set_ylabel("Phase Scatter (degrees)", fontsize=config.effective_label_size, color="purple")
    ax2.set_xticks(iterations)
    ax2.grid(True, alpha=0.3)

    # Combined legend
    lines1, labels1 = ax2.get_legend_handles_labels()
    if data.amp_scatter:
        lines2, labels2 = ax2_twin.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, fontsize=config.effective_tick_size - 1)
    else:
        ax2.legend(fontsize=config.effective_tick_size - 1)

    fig.suptitle(title, fontsize=config.effective_title_size)
    fig.tight_layout()

    if output:
        fig.savefig(output, dpi=config.dpi, bbox_inches="tight")
        logger.info(f"Saved antenna solution quality plot: {output}")
        plt.close(fig)

    return fig


def plot_chi_squared_improvement(
    initial_chi_sq: float,
    chi_squared_per_iteration: list[float],
    output: str | Path | None = None,
    config: FigureConfig | None = None,
    title: str = "Chi-Squared Convergence",
    target_improvement: float = 0.95,
) -> Figure:
    """Plot chi-squared improvement ratio vs iteration.

    Shows how close the algorithm is to the convergence criterion.

    Parameters
    ----------
    initial_chi_sq :
        Initial chi-squared value
    chi_squared_per_iteration :
        Chi-squared values per iteration
    output :
        Output file path
    config :
        Figure configuration
    title :
        Plot title
    target_improvement :
        Target chi-sq ratio for convergence (e.g., 0.95 = 5% improvement)

    Returns
    -------
        matplotlib Figure object

    """
    _setup_matplotlib()
    import matplotlib.pyplot as plt

    if config is None:
        config = FigureConfig(style=PlotStyle.QUICKLOOK)

    iterations = list(range(len(chi_squared_per_iteration)))

    # Compute improvement ratios
    ratios = [chi_sq / initial_chi_sq for chi_sq in chi_squared_per_iteration]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=config.figsize, sharex=True)

    # Panel 1: Absolute chi-squared
    ax1.plot(iterations, chi_squared_per_iteration, "o-", color="red", markersize=8, linewidth=2)
    ax1.axhline(initial_chi_sq, color="gray", linestyle="--", alpha=0.7, label="Initial")
    ax1.set_ylabel("Chi-squared", fontsize=config.effective_label_size)
    ax1.legend(fontsize=config.effective_tick_size)
    ax1.grid(True, alpha=0.3)

    # Panel 2: Improvement ratio
    ax2.plot(iterations, ratios, "o-", color="blue", markersize=8, linewidth=2)
    ax2.axhline(1.0, color="gray", linestyle="--", alpha=0.7, label="No improvement")
    ax2.axhline(
        target_improvement,
        color="green",
        linestyle="--",
        alpha=0.7,
        label=f"Target ({target_improvement})",
    )

    # Shade convergence region
    ax2.fill_between(iterations, 0, target_improvement, alpha=0.1, color="green")

    ax2.set_xlabel("Iteration", fontsize=config.effective_label_size)
    ax2.set_ylabel("Chi-sq / Initial", fontsize=config.effective_label_size)
    ax2.set_ylim(0, max(1.1, max(ratios) * 1.1))
    ax2.legend(fontsize=config.effective_tick_size)
    ax2.grid(True, alpha=0.3)

    # Annotate final improvement
    final_improvement = (1 - ratios[-1]) * 100
    ax2.text(
        0.98,
        0.05,
        f"Total improvement: {final_improvement:.1f}%",
        transform=ax2.transAxes,
        ha="right",
        va="bottom",
        fontsize=config.effective_tick_size,
        bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"),
    )

    fig.suptitle(title, fontsize=config.effective_title_size)
    fig.tight_layout()

    if output:
        fig.savefig(output, dpi=config.dpi, bbox_inches="tight")
        logger.info(f"Saved chi-squared improvement plot: {output}")
        plt.close(fig)

    return fig


def plot_clean_convergence(
    iterations: list[int],
    residual_rms: list[float],
    model_flux: list[float] | None = None,
    peak_residual: list[float] | None = None,
    output: str | Path | None = None,
    config: FigureConfig | None = None,
    title: str = "CLEAN Algorithm Convergence",
    threshold_rms: float | None = None,
) -> Figure:
    """Plot CLEAN algorithm convergence metrics.

    Parameters
    ----------
    iterations :
        Iteration numbers (or major cycle numbers)
    residual_rms :
        RMS of residual image per iteration
    model_flux :
        Total model flux per iteration
    peak_residual :
        Peak residual per iteration
    output :
        Output file path
    config :
        Figure configuration
    title :
        Plot title
    threshold_rms :
        Target RMS threshold to mark
    iterations: List[int] :

    residual_rms: List[float] :

    model_flux: Optional[List[float]] :
         (Default value = None)
    peak_residual: Optional[List[float]] :
         (Default value = None)
    output : Optional[Union[str, Path]]
         (Default value = None)
    config: Optional[FigureConfig] :
         (Default value = None)

    Returns
    -------
        matplotlib Figure object

    """
    _setup_matplotlib()
    import matplotlib.pyplot as plt

    if config is None:
        config = FigureConfig(style=PlotStyle.QUICKLOOK)

    n_panels = 1 + (model_flux is not None) + (peak_residual is not None)
    fig, axes = plt.subplots(
        n_panels, 1, figsize=(config.figsize[0], config.figsize[1] * n_panels / 2), sharex=True
    )

    if n_panels == 1:
        axes = [axes]

    panel_idx = 0

    # Residual RMS
    ax = axes[panel_idx]
    ax.semilogy(
        iterations,
        residual_rms,
        "o-",
        color="blue",
        markersize=6,
        linewidth=1.5,
        label="Residual RMS",
    )
    if threshold_rms is not None:
        ax.axhline(
            threshold_rms,
            color="green",
            linestyle="--",
            alpha=0.7,
            label=f"Threshold ({threshold_rms:.2e})",
        )
    ax.set_ylabel("Residual RMS (Jy/beam)", fontsize=config.effective_label_size)
    ax.legend(fontsize=config.effective_tick_size)
    ax.grid(True, alpha=0.3)
    panel_idx += 1

    # Peak residual
    if peak_residual is not None:
        ax = axes[panel_idx]
        ax.semilogy(iterations, peak_residual, "o-", color="red", markersize=6, linewidth=1.5)
        ax.set_ylabel("Peak Residual (Jy/beam)", fontsize=config.effective_label_size)
        ax.grid(True, alpha=0.3)
        panel_idx += 1

    # Model flux
    if model_flux is not None:
        ax = axes[panel_idx]
        ax.plot(iterations, model_flux, "o-", color="green", markersize=6, linewidth=1.5)
        ax.set_ylabel("Model Flux (Jy)", fontsize=config.effective_label_size)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Iteration / Major Cycle", fontsize=config.effective_label_size)

    fig.suptitle(title, fontsize=config.effective_title_size)
    fig.tight_layout()

    if output:
        fig.savefig(output, dpi=config.dpi, bbox_inches="tight")
        logger.info(f"Saved CLEAN convergence plot: {output}")
        plt.close(fig)

    return fig


def plot_convergence_comparison(
    results: dict[str, ConvergenceData | Any],
    metric: str = "snr",
    output: str | Path | None = None,
    config: FigureConfig | None = None,
    title: str = "Convergence Comparison",
) -> Figure:
    """Compare convergence across multiple runs/configurations.

    Parameters
    ----------
    results : Dict[str, Union[ConvergenceData, Any]]
        Dictionary mapping run name -> ConvergenceData or SelfCalResult.
    metric : str, optional
        Metric to compare ("snr", "chi_squared", "rms").
    output : Optional[Union[str, Path]], optional
        Output file path.
    config : Optional[FigureConfig], optional
        Figure configuration.
    title : str, optional
        Plot title.

    Returns
    -------
    Figure
        Matplotlib Figure object.
    """
    _setup_matplotlib()
    import matplotlib.pyplot as plt

    if config is None:
        config = FigureConfig(style=PlotStyle.QUICKLOOK)

    fig, ax = plt.subplots(figsize=config.figsize, dpi=config.dpi)

    colors = plt.cm.tab10(np.linspace(0, 1, len(results)))

    for (name, data), color in zip(results.items(), colors):
        if not isinstance(data, ConvergenceData):
            data = extract_convergence_from_selfcal_result(data)

        iterations = data.iterations
        values = getattr(data, metric, None)

        if values is None:
            logger.warning(f"Metric {metric} not available for {name}")
            continue

        ax.plot(iterations, values, "o-", color=color, markersize=6, linewidth=1.5, label=name)

    ax.set_xlabel("Iteration", fontsize=config.effective_label_size)
    ax.set_ylabel(metric.replace("_", " ").title(), fontsize=config.effective_label_size)
    ax.set_title(title, fontsize=config.effective_title_size)
    ax.legend(fontsize=config.effective_tick_size)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()

    if output:
        fig.savefig(output, dpi=config.dpi, bbox_inches="tight")
        logger.info(f"Saved convergence comparison: {output}")
        plt.close(fig)

    return fig


@dataclass
class TimeFreqConvergenceData:
    """Container for time-frequency resolved convergence data."""

    iterations: list[int]
    time_bins: NDArray[np.floating]
    freq_bins: NDArray[np.floating]
    metric_grid: NDArray[np.floating]  # Shape: (n_iter, n_time, n_freq)
    metric_name: str = "rms"
    antenna_ids: list[int] | None = None
    per_antenna_grid: NDArray[np.floating] | None = None  # (n_iter, n_ant, n_time, n_freq)


def compute_time_freq_convergence(
    ms_paths: list[str | Path],
    data_column: str = "CORRECTED_DATA",
    model_column: str = "MODEL_DATA",
    n_time_bins: int = 10,
    n_freq_bins: int = 16,
    metric: str = "rms",
    per_antenna: bool = False,
) -> TimeFreqConvergenceData:
    """Compute time-frequency resolved convergence metrics from MS files.

    Computes residuals (DATA - MODEL) binned by time and frequency to show
    where calibration is converging well vs. poorly.

    Parameters
    ----------
    ms_paths : List[Union[str, Path]]
        List of MS file paths, one per iteration.
    data_column : str, optional
        Column with calibrated data (default is "CORRECTED_DATA").
    model_column : str, optional
        Column with model visibilities (default is "MODEL_DATA").
    n_time_bins : int, optional
        Number of time bins (default is 10).
    n_freq_bins : int, optional
        Number of frequency bins (default is 16).
    metric : str, optional
        Metric to compute ("rms", "mad", "chi_squared") (default is "rms").
    per_antenna : bool, optional
        Whether to compute per-antenna breakdown (default is False).

    Returns
    -------
    TimeFreqConvergenceData
        TimeFreqConvergenceData container.
    """
    from casacore.tables import table

    iterations = list(range(len(ms_paths)))
    metric_grids = []
    per_antenna_grids = [] if per_antenna else None
    time_bins = None
    freq_bins = None
    antenna_ids = None

    for ms_path in ms_paths:
        with table(str(ms_path), readonly=True, ack=False) as tb:
            data = tb.getcol(data_column)
            model = tb.getcol(model_column)
            flags = tb.getcol("FLAG")
            times = tb.getcol("TIME")
            ant1 = tb.getcol("ANTENNA1")
            ant2 = tb.getcol("ANTENNA2")

        # Get frequency info
        with table(f"{ms_path}/SPECTRAL_WINDOW", readonly=True, ack=False) as spw:
            freqs = spw.getcol("CHAN_FREQ").flatten() / 1e6  # MHz

        # Compute residuals
        residuals = data - model
        residuals = np.ma.array(residuals, mask=flags)

        # Determine bin edges
        if time_bins is None:
            time_edges = np.linspace(times.min(), times.max(), n_time_bins + 1)
            time_bins = 0.5 * (time_edges[:-1] + time_edges[1:])

            freq_edges = np.linspace(freqs.min(), freqs.max(), n_freq_bins + 1)
            freq_bins = 0.5 * (freq_edges[:-1] + freq_edges[1:])

            if per_antenna:
                antenna_ids = sorted(set(ant1) | set(ant2))

        # Compute time bin indices
        time_idx = np.clip(np.digitize(times, time_edges) - 1, 0, n_time_bins - 1)

        # Compute freq bin indices
        n_chan = residuals.shape[1]
        chan_to_freq_bin = np.clip(np.digitize(freqs[:n_chan], freq_edges) - 1, 0, n_freq_bins - 1)

        # Compute metric grid
        metric_grid = np.zeros((n_time_bins, n_freq_bins))
        count_grid = np.zeros((n_time_bins, n_freq_bins))

        for ti in range(n_time_bins):
            time_mask = time_idx == ti
            for fi in range(n_freq_bins):
                chan_mask = chan_to_freq_bin == fi
                subset = residuals[time_mask][:, chan_mask, :]

                if subset.count() == 0:
                    continue

                if metric == "rms":
                    val = np.sqrt(np.mean(np.abs(subset.compressed()) ** 2))
                elif metric == "mad":
                    val = np.median(np.abs(subset.compressed()))
                elif metric == "chi_squared":
                    val = np.mean(np.abs(subset.compressed()) ** 2)
                else:
                    val = np.sqrt(np.mean(np.abs(subset.compressed()) ** 2))

                metric_grid[ti, fi] = val
                count_grid[ti, fi] = subset.count()

        metric_grids.append(metric_grid)

        # Per-antenna breakdown
        if per_antenna and antenna_ids:
            ant_grid = np.zeros((len(antenna_ids), n_time_bins, n_freq_bins))
            for ai, ant in enumerate(antenna_ids):
                ant_mask = (ant1 == ant) | (ant2 == ant)
                for ti in range(n_time_bins):
                    time_mask = time_idx == ti
                    combined_mask = ant_mask & time_mask
                    for fi in range(n_freq_bins):
                        chan_mask = chan_to_freq_bin == fi
                        subset = residuals[combined_mask][:, chan_mask, :]
                        if subset.count() > 0:
                            ant_grid[ai, ti, fi] = np.sqrt(
                                np.mean(np.abs(subset.compressed()) ** 2)
                            )
            per_antenna_grids.append(ant_grid)

    return TimeFreqConvergenceData(
        iterations=iterations,
        time_bins=time_bins,
        freq_bins=freq_bins,
        metric_grid=np.array(metric_grids),
        metric_name=metric,
        antenna_ids=antenna_ids,
        per_antenna_grid=np.array(per_antenna_grids) if per_antenna_grids else None,
    )


def plot_time_freq_convergence_heatmap(
    data: TimeFreqConvergenceData,
    iteration: int | None = None,
    output: str | Path | None = None,
    config: FigureConfig | None = None,
    title: str | None = None,
    cmap: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
    log_scale: bool = False,
    show_colorbar: bool = True,
    interactive: bool = False,
) -> Figure | dict[str, Any]:
    """Plot time-frequency convergence heatmap.

    Shows how calibration quality varies across time and frequency for a
    single iteration or as an animation/comparison across iterations.

    Parameters
    ----------
    data :
        TimeFreqConvergenceData container
    iteration :
        Specific iteration to plot (None for last)
    output :
        Output file path
    config :
        Figure configuration
    title :
        Plot title
    cmap :
        Colormap name
    vmin :
        Minimum value for colormap
    vmax :
        Maximum value for colormap
    log_scale :
        Use logarithmic color scale
    show_colorbar :
        Show colorbar
    interactive :
        Return Vega-Lite spec instead of matplotlib Figure

    Returns
    -------
        matplotlib Figure or Vega-Lite spec dict

    """
    if interactive:
        return _create_time_freq_heatmap_vega_spec(
            data, iteration=iteration, title=title, cmap=cmap
        )

    _setup_matplotlib()
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    if config is None:
        config = FigureConfig(style=PlotStyle.QUICKLOOK)

    if iteration is None:
        iteration = data.iterations[-1]

    iter_idx = data.iterations.index(iteration)
    grid = data.metric_grid[iter_idx]

    if log_scale:
        grid = np.ma.array(grid, mask=grid <= 0)

    fig, ax = plt.subplots(figsize=config.figsize, dpi=config.dpi)

    norm = LogNorm(vmin=vmin, vmax=vmax) if log_scale else None

    im = ax.pcolormesh(
        data.freq_bins,
        data.time_bins,
        grid,
        cmap=cmap,
        norm=norm,
        vmin=None if log_scale else vmin,
        vmax=None if log_scale else vmax,
        shading="auto",
    )

    if show_colorbar:
        cbar = fig.colorbar(im, ax=ax, pad=0.02)
        cbar.set_label(f"{data.metric_name.upper()}", fontsize=config.effective_label_size)

    ax.set_xlabel("Frequency (MHz)", fontsize=config.effective_label_size)
    ax.set_ylabel("Time (MJD)", fontsize=config.effective_label_size)

    if title is None:
        title = f"Time-Frequency Convergence (Iteration {iteration})"
    ax.set_title(title, fontsize=config.effective_title_size)

    fig.tight_layout()

    if output:
        fig.savefig(output, dpi=config.dpi, bbox_inches="tight")
        logger.info(f"Saved time-frequency heatmap: {output}")
        plt.close(fig)

    return fig


def plot_time_freq_convergence_animation(
    data: TimeFreqConvergenceData,
    output: str | Path,
    config: FigureConfig | None = None,
    title: str = "Calibration Convergence",
    cmap: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
    log_scale: bool = False,
    fps: int = 2,
) -> None:
    """Create animated GIF showing convergence evolution.

    Parameters
    ----------
    data :
        TimeFreqConvergenceData container
    output :
        Output file path (should be .gif)
    config :
        Figure configuration
    title :
        Base title
    cmap :
        Colormap name
    vmin :
        Minimum value for colormap (auto if None)
    vmax :
        Maximum value for colormap (auto if None)
    log_scale :
        Use logarithmic color scale
    fps :
        Frames per second
    """
    _setup_matplotlib()
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    from matplotlib.colors import LogNorm

    if config is None:
        config = FigureConfig(style=PlotStyle.QUICKLOOK)

    # Auto-scale across all iterations
    if vmin is None:
        vmin = np.nanmin(data.metric_grid[data.metric_grid > 0])
    if vmax is None:
        vmax = np.nanmax(data.metric_grid)

    fig, ax = plt.subplots(figsize=config.figsize, dpi=config.dpi)

    norm = LogNorm(vmin=vmin, vmax=vmax) if log_scale else None

    # Initial plot
    im = ax.pcolormesh(
        data.freq_bins,
        data.time_bins,
        data.metric_grid[0],
        cmap=cmap,
        norm=norm,
        vmin=None if log_scale else vmin,
        vmax=None if log_scale else vmax,
        shading="auto",
    )

    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label(f"{data.metric_name.upper()}", fontsize=config.effective_label_size)

    ax.set_xlabel("Frequency (MHz)", fontsize=config.effective_label_size)
    ax.set_ylabel("Time (MJD)", fontsize=config.effective_label_size)
    title_obj = ax.set_title(f"{title} (Iteration 0)", fontsize=config.effective_title_size)

    def update(frame):
        im.set_array(data.metric_grid[frame].ravel())
        title_obj.set_text(f"{title} (Iteration {data.iterations[frame]})")
        return [im, title_obj]

    anim = FuncAnimation(fig, update, frames=len(data.iterations), interval=1000 // fps, blit=True)

    writer = PillowWriter(fps=fps)
    anim.save(str(output), writer=writer)
    plt.close(fig)
    logger.info(f"Saved convergence animation: {output}")


def plot_time_freq_difference_heatmap(
    data: TimeFreqConvergenceData,
    iter1: int = 0,
    iter2: int | None = None,
    output: str | Path | None = None,
    config: FigureConfig | None = None,
    title: str | None = None,
    cmap: str = "RdBu_r",
    symmetric: bool = True,
) -> Figure:
    """Plot difference heatmap between two iterations.

    Shows improvement (or degradation) in calibration quality between
    two iterations.

    Parameters
    ----------
    data :
        TimeFreqConvergenceData container
    iter1 :
        First iteration (baseline)
    iter2 :
        Second iteration (None for last)
    output :
        Output file path
    config :
        Figure configuration
    title :
        Plot title
    cmap :
        Colormap name (diverging recommended)
    symmetric :
        Make colorbar symmetric around zero

    Returns
    -------
        matplotlib Figure object

    """
    _setup_matplotlib()
    import matplotlib.pyplot as plt

    if config is None:
        config = FigureConfig(style=PlotStyle.QUICKLOOK)

    if iter2 is None:
        iter2 = data.iterations[-1]

    idx1 = data.iterations.index(iter1)
    idx2 = data.iterations.index(iter2)

    diff = data.metric_grid[idx2] - data.metric_grid[idx1]

    fig, ax = plt.subplots(figsize=config.figsize, dpi=config.dpi)

    vmax = np.nanmax(np.abs(diff)) if symmetric else None
    vmin = -vmax if symmetric else None

    im = ax.pcolormesh(
        data.freq_bins,
        data.time_bins,
        diff,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        shading="auto",
    )

    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label(
        f"Δ{data.metric_name.upper()} (Iter {iter2} - {iter1})",
        fontsize=config.effective_label_size,
    )

    ax.set_xlabel("Frequency (MHz)", fontsize=config.effective_label_size)
    ax.set_ylabel("Time (MJD)", fontsize=config.effective_label_size)

    if title is None:
        title = f"Convergence Improvement (Iteration {iter1} → {iter2})"
    ax.set_title(title, fontsize=config.effective_title_size)

    # Add summary annotation
    mean_improvement = np.nanmean(diff)
    pct_improved = np.sum(diff < 0) / diff.size * 100

    ax.text(
        0.02,
        0.98,
        f"Mean Δ: {mean_improvement:.2e}\n{pct_improved:.0f}% cells improved",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=config.effective_tick_size,
        bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"),
    )

    fig.tight_layout()

    if output:
        fig.savefig(output, dpi=config.dpi, bbox_inches="tight")
        logger.info(f"Saved difference heatmap: {output}")
        plt.close(fig)

    return fig


def plot_per_antenna_convergence_heatmap(
    data: TimeFreqConvergenceData,
    iteration: int | None = None,
    output: str | Path | None = None,
    config: FigureConfig | None = None,
    title: str | None = None,
    cmap: str = "viridis",
    ncols: int = 4,
) -> Figure:
    """Plot per-antenna time-frequency convergence heatmaps.

    Creates a grid of heatmaps, one per antenna, showing where each
    antenna's calibration is converging well vs. poorly.

    Parameters
    ----------
    data :
        TimeFreqConvergenceData with per_antenna_grid set
    iteration :
        Specific iteration (None for last)
    output :
        Output file path
    config :
        Figure configuration
    title :
        Overall title
    cmap :
        Colormap name
    ncols :
        Number of columns in grid

    Returns
    -------
        matplotlib Figure object

    """
    _setup_matplotlib()
    import matplotlib.pyplot as plt

    if data.per_antenna_grid is None or data.antenna_ids is None:
        raise ValueError("TimeFreqConvergenceData must have per_antenna_grid set")

    if config is None:
        config = FigureConfig(style=PlotStyle.QUICKLOOK)

    if iteration is None:
        iteration = data.iterations[-1]

    iter_idx = data.iterations.index(iteration)
    n_ant = len(data.antenna_ids)
    nrows = (n_ant + ncols - 1) // ncols

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(config.figsize[0] * ncols / 2, config.figsize[1] * nrows / 2),
        squeeze=False,
    )

    # Global colorbar limits
    vmin = np.nanmin(data.per_antenna_grid[iter_idx])
    vmax = np.nanmax(data.per_antenna_grid[iter_idx])

    for ai, ant_id in enumerate(data.antenna_ids):
        row, col = divmod(ai, ncols)
        ax = axes[row, col]

        grid = data.per_antenna_grid[iter_idx, ai]

        im = ax.pcolormesh(
            data.freq_bins,
            data.time_bins,
            grid,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            shading="auto",
        )

        ax.set_title(f"Ant {ant_id}", fontsize=config.effective_tick_size)

        if col == 0:
            ax.set_ylabel("Time", fontsize=config.effective_tick_size - 1)
        if row == nrows - 1:
            ax.set_xlabel("Freq", fontsize=config.effective_tick_size - 1)

    # Hide empty subplots
    for ai in range(n_ant, nrows * ncols):
        row, col = divmod(ai, ncols)
        axes[row, col].set_visible(False)

    # Add colorbar
    fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax, label=data.metric_name.upper())

    if title is None:
        title = f"Per-Antenna Convergence (Iteration {iteration})"
    fig.suptitle(title, fontsize=config.effective_title_size)

    if output:
        fig.savefig(output, dpi=config.dpi, bbox_inches="tight")
        logger.info(f"Saved per-antenna convergence heatmap: {output}")
        plt.close(fig)

    return fig


def _create_time_freq_heatmap_vega_spec(
    data: TimeFreqConvergenceData,
    iteration: int | None = None,
    title: str | None = None,
    cmap: str = "viridis",
) -> dict[str, Any]:
    """Create Vega-Lite spec for interactive time-frequency heatmap.

    Parameters
    ----------
    data :
        TimeFreqConvergenceData container
    iteration :
        Specific iteration (None for last)
    title :
        Plot title
    cmap :
        Colormap name (translated to Vega scheme)

    Returns
    -------
        Vega-Lite specification dictionary

    """
    if iteration is None:
        iteration = data.iterations[-1]

    iter_idx = data.iterations.index(iteration)
    grid = data.metric_grid[iter_idx]

    # Convert to tabular format for Vega
    records = []
    for ti, t in enumerate(data.time_bins):
        for fi, f in enumerate(data.freq_bins):
            records.append(
                {
                    "time": float(t),
                    "freq": float(f),
                    "value": float(grid[ti, fi]),
                }
            )

    # Map matplotlib colormap to Vega scheme
    vega_scheme_map = {
        "viridis": "viridis",
        "plasma": "plasma",
        "inferno": "inferno",
        "magma": "magma",
        "cividis": "cividis",
        "RdBu_r": "redblue",
        "coolwarm": "redblue",
    }
    vega_scheme = vega_scheme_map.get(cmap, "viridis")

    if title is None:
        title = f"Time-Frequency Convergence (Iteration {iteration})"

    return {
        "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
        "title": title,
        "description": "Time-frequency resolved calibration convergence heatmap",
        "width": 600,
        "height": 400,
        "data": {"values": records},
        "mark": "rect",
        "encoding": {
            "x": {
                "field": "freq",
                "type": "ordinal",
                "title": "Frequency (MHz)",
                "axis": {"labelAngle": -45},
            },
            "y": {
                "field": "time",
                "type": "ordinal",
                "title": "Time (MJD)",
            },
            "color": {
                "field": "value",
                "type": "quantitative",
                "title": data.metric_name.upper(),
                "scale": {"scheme": vega_scheme},
            },
            "tooltip": [
                {"field": "time", "type": "quantitative", "title": "Time (MJD)"},
                {"field": "freq", "type": "quantitative", "title": "Frequency (MHz)"},
                {
                    "field": "value",
                    "type": "quantitative",
                    "title": data.metric_name.upper(),
                    "format": ".4f",
                },
            ],
        },
    }


def compute_convergence_quality_score(
    data: TimeFreqConvergenceData,
    target_rms: float = 0.01,
) -> dict[str, Any]:
    """Compute overall convergence quality metrics.

    Parameters
    ----------
    data :
        TimeFreqConvergenceData container
    target_rms :
        Target RMS value for "good" calibration

    Returns
    -------
        Dictionary with quality metrics

    """
    initial_grid = data.metric_grid[0]
    final_grid = data.metric_grid[-1]

    # Overall improvement
    mean_initial = np.nanmean(initial_grid)
    mean_final = np.nanmean(final_grid)
    improvement_factor = mean_initial / mean_final if mean_final > 0 else float("inf")

    # Fraction of cells meeting target
    pct_good = np.sum(final_grid < target_rms) / final_grid.size * 100

    # Problem regions (cells that got worse)
    diff = final_grid - initial_grid
    pct_worse = np.sum(diff > 0) / diff.size * 100

    # Worst time/freq bin
    worst_idx = np.unravel_index(np.nanargmax(final_grid), final_grid.shape)
    worst_time = float(data.time_bins[worst_idx[0]])
    worst_freq = float(data.freq_bins[worst_idx[1]])

    # Per-antenna summary if available
    antenna_summary = None
    if data.per_antenna_grid is not None and data.antenna_ids is not None:
        final_ant_grid = data.per_antenna_grid[-1]
        ant_means = np.nanmean(final_ant_grid, axis=(1, 2))
        worst_ant_idx = np.argmax(ant_means)
        antenna_summary = {
            "worst_antenna": data.antenna_ids[worst_ant_idx],
            "worst_antenna_rms": float(ant_means[worst_ant_idx]),
            "best_antenna": data.antenna_ids[np.argmin(ant_means)],
            "best_antenna_rms": float(np.min(ant_means)),
            "antenna_rms_spread": float(np.max(ant_means) - np.min(ant_means)),
        }

    return {
        "improvement_factor": improvement_factor,
        "mean_initial": mean_initial,
        "mean_final": mean_final,
        "pct_cells_good": pct_good,
        "pct_cells_worse": pct_worse,
        "worst_time_bin": worst_time,
        "worst_freq_bin": worst_freq,
        "worst_value": float(np.nanmax(final_grid)),
        "target_rms": target_rms,
        "converged": pct_good > 80 and pct_worse < 10,
        "antenna_summary": antenna_summary,
    }
