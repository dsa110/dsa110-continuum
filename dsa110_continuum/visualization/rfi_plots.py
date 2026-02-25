"""
RFI visualization utilities.

Provides functions for:
- RFI Occupancy vs Frequency (Spectrum)
- RFI Time-Frequency Waterfall
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from dsa110_contimg.core.visualization.config import FigureConfig, PlotStyle
from dsa110_contimg.core.visualization.plot_context import PlotContext, should_generate_interactive


def _setup_matplotlib() -> None:
    """Configure matplotlib for headless operation."""
    import matplotlib

    matplotlib.use("Agg")


def plot_rfi_spectrum(
    freqs: np.ndarray,
    occupancy: np.ndarray,
    output_path: str | Path,
    config: FigureConfig | None = None,
    interactive: bool | None = None,
    context: PlotContext | None = None,
    dual_output: bool = False,
) -> None:
    """Plot RFI occupancy percentage vs Frequency.

    Parameters
    ----------
    freqs : array_like
        Array of frequencies in Hz.
    occupancy : array_like
        Array of occupancy percentages (0-100).
    output_path : str or Path
        Path to save the plot (PNG or JSON for interactive).
    config : FigureConfig, optional
        Figure configuration. Default is None.
    interactive : bool, optional
        If True, generate Vega-Lite JSON. If False, generate PNG.
        If None, use context-based heuristic. Default is None.
    context : PlotContext, optional
        Usage context (e.g., PlotContext.WEB_API, PlotContext.REPORT).
        Determines format when interactive=None. Default is None.
    dual_output : bool, optional
        If True, generate both PNG and Vega-Lite JSON. Default is False.

    Returns
    -------
        None

    Examples
    --------
        >>> # Web API - automatic Vega-Lite
        >>> plot_rfi_spectrum(freqs, occ, "out.png", context=PlotContext.WEB_API)

        >>> # Report - automatic PNG
        >>> plot_rfi_spectrum(freqs, occ, "out.png", context=PlotContext.REPORT)

        >>> # Explicit override
        >>> plot_rfi_spectrum(freqs, occ, "out.png", interactive=True)

        >>> # Dual output
        >>> plot_rfi_spectrum(freqs, occ, "out.png", dual_output=True)
    """
    from dsa110_contimg.core.visualization.plot_context import PerformanceLogger

    # Performance logging
    with PerformanceLogger("rfi_spectrum", output_path, context, interactive) as perf:
        # Determine format based on context
        use_interactive = should_generate_interactive(context, interactive)

        # Dual output: generate both formats
        if dual_output:
            perf.log_format_selection("dual", "dual_output=True")
            _generate_rfi_spectrum_png(freqs, occupancy, output_path, config)
            json_path = Path(str(output_path).replace(".png", ".vega.json"))
            _generate_rfi_spectrum_json(freqs, occupancy, json_path)
            return

        # Interactive mode
        if use_interactive:
            reason = "explicit" if interactive else "context"
            perf.log_format_selection("vega-lite", reason)
            json_path = Path(str(output_path).replace(".png", ".vega.json"))
            _generate_rfi_spectrum_json(freqs, occupancy, json_path)
            return

        # Static mode
        perf.log_format_selection("png", "default")
        _generate_rfi_spectrum_png(freqs, occupancy, output_path, config)


def _generate_rfi_spectrum_json(
    freqs: np.ndarray,
    occupancy: np.ndarray,
    output_path: Path,
) -> None:
    """Generate Vega-Lite JSON for RFI spectrum.

    Parameters
    ----------
    freqs: np.ndarray :

    occupancy: np.ndarray :

    """
    from dsa110_contimg.core.visualization.vega_specs import (
        create_rfi_spectrum_spec,
        save_vega_spec,
    )

    freqs_mhz = freqs / 1e6
    spec = create_rfi_spectrum_spec(
        freqs_mhz=freqs_mhz,
        occupancy=occupancy,
        title="RFI Occupancy vs Frequency",
    )
    save_vega_spec(spec, output_path)


def _generate_rfi_spectrum_png(
    freqs: np.ndarray,
    occupancy: np.ndarray,
    output_path: str | Path,
    config: FigureConfig | None,
) -> None:
    """Generate PNG for RFI spectrum.

    Parameters
    ----------
    freqs: np.ndarray :

    occupancy: np.ndarray :

    output_path : Union[str, Path]
    config: Optional[FigureConfig] :


    """
    _setup_matplotlib()

    if config is None:
        config = FigureConfig(style=PlotStyle.QUICKLOOK)

    freqs_mhz = freqs / 1e6

    with plt.rc_context(config.to_mpl_params()):
        fig, ax = plt.subplots(figsize=config.figsize)

        ax.plot(freqs_mhz, occupancy, color="red", linewidth=1.0)
        ax.fill_between(freqs_mhz, occupancy, color="red", alpha=0.3)

        ax.set_xlabel("Frequency (MHz)")
        ax.set_ylabel("RFI Occupancy (%)")
        ax.set_title("RFI Occupancy vs Frequency")
        ax.set_ylim(0, 100)
        ax.grid(True, alpha=0.3)

        # Add summary stats
        mean_occ = np.mean(occupancy)
        ax.text(
            0.02,
            0.95,
            f"Mean Occupancy: {mean_occ:.2f}%",
            transform=ax.transAxes,
            bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"),
            color="black",
        )

        plt.tight_layout()
        plt.savefig(output_path, dpi=config.dpi, bbox_inches="tight")
        plt.close(fig)


def plot_rfi_waterfall(
    waterfall: np.ndarray,
    times: np.ndarray,
    freqs: np.ndarray,
    output_path: str | Path,
    config: FigureConfig | None = None,
    interactive: bool | None = None,
    context: PlotContext | None = None,
    dual_output: bool = False,
) -> None:
    """Plot Time-Frequency RFI occupancy waterfall.

    Parameters
    ----------
    waterfall :
        2D array (n_times, n_freqs) of occupancy %.
    times :
        Array of MJD times.
    freqs :
        Array of frequencies in Hz.
    output_path :
        Path to save the plot (PNG or JSON for interactive).
    config :
        Figure configuration.
    interactive :
        If True, generate Vega-Lite JSON. If False, generate PNG.
        If None, use context-based heuristic.
    context :
        Usage context for automatic format selection.
    dual_output :
        If True, generate both PNG and Vega-Lite JSON.
    waterfall: np.ndarray :

    times: np.ndarray :

    freqs: np.ndarray :

    output_path : Union[str, Path]
    config: Optional[FigureConfig] :
         (Default value = None)
    interactive: Optional[bool] :
         (Default value = None)
    context: Optional[PlotContext] :
         (Default value = None)
    """
    from dsa110_contimg.core.visualization.plot_context import PerformanceLogger

    # Performance logging
    with PerformanceLogger("rfi_waterfall", output_path, context, interactive) as perf:
        # Determine format based on context
        use_interactive = should_generate_interactive(context, interactive)

        # Dual output: generate both formats
        if dual_output:
            perf.log_format_selection("dual", "dual_output=True")
            _generate_rfi_waterfall_png(waterfall, times, freqs, output_path, config)
            json_path = Path(str(output_path).replace(".png", ".vega.json"))
            _generate_rfi_waterfall_json(waterfall, times, freqs, json_path)
            return

        # Interactive mode
        if use_interactive:
            reason = "explicit" if interactive else "context"
            perf.log_format_selection("vega-lite", reason)
            json_path = Path(str(output_path).replace(".png", ".vega.json"))
            _generate_rfi_waterfall_json(waterfall, times, freqs, json_path)
            return

        # Static mode
        perf.log_format_selection("png", "default")
        _generate_rfi_waterfall_png(waterfall, times, freqs, output_path, config)


def _generate_rfi_waterfall_json(
    waterfall: np.ndarray,
    times: np.ndarray,
    freqs: np.ndarray,
    output_path: Path,
) -> None:
    """Generate Vega-Lite JSON for RFI waterfall.

    Parameters
    ----------
    waterfall: np.ndarray :

    times: np.ndarray :

    freqs: np.ndarray :

    """
    from astropy.time import Time

    from dsa110_contimg.core.visualization.vega_specs import (
        create_rfi_waterfall_spec,
        save_vega_spec,
    )

    freqs_mhz = freqs / 1e6
    time_objs = Time(times, format="mjd")
    time_strings = [t.iso for t in time_objs]

    spec = create_rfi_waterfall_spec(
        times=time_strings,
        freqs_mhz=freqs_mhz,
        occupancy_2d=waterfall,
        title="RFI Time-Frequency Waterfall",
    )
    save_vega_spec(spec, output_path)


def _generate_rfi_waterfall_png(
    waterfall: np.ndarray,
    times: np.ndarray,
    freqs: np.ndarray,
    output_path: str | Path,
    config: FigureConfig | None,
) -> None:
    """Generate PNG for RFI waterfall.

    Parameters
    ----------
    waterfall: np.ndarray :

    times: np.ndarray :

    freqs: np.ndarray :

    output_path : Union[str, Path]
    config: Optional[FigureConfig] :


    """
    _setup_matplotlib()

    if config is None:
        config = FigureConfig(style=PlotStyle.QUICKLOOK)

    freqs_mhz = freqs / 1e6

    # Convert MJD to relative time (minutes) for readability
    t0 = times[0]
    rel_times = (times - t0) * 24 * 60  # Days -> Minutes

    with plt.rc_context(config.to_mpl_params()):
        fig, ax = plt.subplots(figsize=config.figsize)

        # Extent: [left, right, bottom, top]
        extent = [freqs_mhz[0], freqs_mhz[-1], rel_times[-1], rel_times[0]]

        im = ax.imshow(
            waterfall,
            aspect="auto",
            origin="upper",
            extent=extent,
            cmap="inferno",
            vmin=0,
            vmax=100,
        )

        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Flagged Fraction (%)")

        ax.set_xlabel("Frequency (MHz)")
        ax.set_ylabel("Time (minutes from start)")
        ax.set_title("RFI Occupancy Waterfall")

        plt.tight_layout()
        plt.savefig(output_path, dpi=config.dpi, bbox_inches="tight")
        plt.close(fig)
