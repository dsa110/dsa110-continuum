"""QA diagnostic plotting utilities."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from dsa110_contimg.core.qa.image_metrics import get_center_cutout
from dsa110_contimg.core.visualization.config import FigureConfig, PlotStyle
from dsa110_contimg.core.visualization.plot_context import PlotContext, should_generate_interactive


def _setup_matplotlib() -> None:
    """Configure matplotlib for headless operation."""
    import matplotlib

    matplotlib.use("Agg")


def plot_psf_correlation(
    dirty_data: np.ndarray,
    psf_data: np.ndarray,
    correlation_val: float,
    output_path: str | Path,
    cutout_size: int = 64,
    config: FigureConfig | None = None,
    interactive: bool | None = None,
    context: PlotContext | None = None,
    dual_output: bool = False,
) -> None:
    """Generate a 3-panel plot showing Source, PSF, and their correlation.

    Parameters
    ----------
    dirty_data :
        2D array of dirty image.
    psf_data :
        2D array of PSF image.
    correlation_val :
        Calculated Pearson correlation.
    output_path :
        Path to save the plot (PNG or JSON for interactive).
    cutout_size :
        Size of the cutout to display.
    config :
        Figure configuration (optional).
    interactive :
        If True, generate Vega-Lite JSON for scatter plot only.
        If None, use context-based heuristic.
    context :
        Usage context for automatic format selection.
    dual_output :
        If True, generate both PNG and Vega-Lite JSON.
    dirty_data: np.ndarray :

    psf_data: np.ndarray :

    """
    from dsa110_contimg.core.visualization.plot_context import PerformanceLogger

    # Get cutouts
    source_cutout = get_center_cutout(dirty_data, cutout_size)
    psf_cutout = get_center_cutout(psf_data, cutout_size)

    # Normalize
    source_norm = source_cutout / np.max(source_cutout)
    psf_norm = psf_cutout / np.max(psf_cutout)

    # Performance logging
    with PerformanceLogger("psf_correlation", output_path, context, interactive) as perf:
        # Determine format based on context
        use_interactive = should_generate_interactive(context, interactive)

        # Dual output: generate both formats
        if dual_output:
            perf.log_format_selection("dual", "dual_output=True")
            _generate_psf_correlation_png(
                source_cutout,
                psf_cutout,
                source_norm,
                psf_norm,
                correlation_val,
                output_path,
                config,
            )
            json_path = Path(str(output_path).replace(".png", ".vega.json"))
            _generate_psf_correlation_json(psf_norm, source_norm, correlation_val, json_path)
            return

        # Interactive mode: generate Vega-Lite scatter plot only
        if use_interactive:
            reason = "explicit" if interactive else "context"
            perf.log_format_selection("vega-lite", reason)
            json_path = Path(str(output_path).replace(".png", ".vega.json"))
            _generate_psf_correlation_json(psf_norm, source_norm, correlation_val, json_path)
            return

        # Static mode
        perf.log_format_selection("png", "default")
        _generate_psf_correlation_png(
            source_cutout, psf_cutout, source_norm, psf_norm, correlation_val, output_path, config
        )


def _generate_psf_correlation_json(
    psf_norm: np.ndarray,
    source_norm: np.ndarray,
    correlation_val: float,
    output_path: Path,
) -> None:
    """Generate Vega-Lite JSON for PSF correlation scatter plot.

    Parameters
    ----------
    psf_norm: np.ndarray :

    source_norm: np.ndarray :

    """
    from dsa110_contimg.core.visualization.vega_specs import (
        create_scatter_spec,
        save_vega_spec,
    )

    spec = create_scatter_spec(
        x_data=psf_norm.flatten(),
        y_data=source_norm.flatten(),
        x_label="Normalized PSF Pixel Value",
        y_label="Normalized Source Pixel Value",
        title=f"PSF Correlation: R = {correlation_val:.4f}",
    )
    save_vega_spec(spec, output_path)


def _generate_psf_correlation_png(
    source_cutout: np.ndarray,
    psf_cutout: np.ndarray,
    source_norm: np.ndarray,
    psf_norm: np.ndarray,
    correlation_val: float,
    output_path: str | Path,
    config: FigureConfig | None,
) -> None:
    """Generate PNG for PSF correlation (3-panel plot).

    Parameters
    ----------
    source_cutout: np.ndarray :

    psf_cutout: np.ndarray :

    source_norm: np.ndarray :

    psf_norm: np.ndarray :

    """
    _setup_matplotlib()

    if config is None:
        config = FigureConfig(style=PlotStyle.QUICKLOOK)

    # Create figure
    with plt.rc_context(config.to_mpl_params()):
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Source
        im1 = axes[0].imshow(source_cutout, origin="lower", cmap=config.cmap)
        axes[0].set_title("Source (Dirty Image)")
        plt.colorbar(im1, ax=axes[0], label="Flux (Jy/beam)")

        # PSF
        im2 = axes[1].imshow(psf_cutout, origin="lower", cmap=config.cmap)
        axes[1].set_title("PSF (Dirty Beam)")
        plt.colorbar(im2, ax=axes[1], label="Response")

        # Correlation
        axes[2].scatter(psf_norm.flatten(), source_norm.flatten(), alpha=0.1, s=1, c="cyan")
        axes[2].plot([0, 1], [0, 1], "r--", lw=2)
        axes[2].set_xlabel("Normalized PSF Pixel Value")
        axes[2].set_ylabel("Normalized Source Pixel Value")
        axes[2].set_title(f"Correlation: R = {correlation_val:.4f}")
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path, dpi=config.dpi, bbox_inches="tight")
        plt.close(fig)


def plot_residual_histogram(
    residual_data: np.ndarray,
    stats: dict,
    output_path: str | Path,
    config: FigureConfig | None = None,
    interactive: bool | None = None,
    context: PlotContext | None = None,
    dual_output: bool = False,
) -> None:
    """Plot histogram of residual image values to assess Gaussian noise.

    Parameters
    ----------
    residual_data :
        2D residual image array.
    stats :
        Statistics dict from calculate_residual_stats.
    output_path :
        Path to save the plot (PNG or JSON for interactive).
    config :
        Figure configuration.
    interactive :
        If True, generate Vega-Lite JSON. If None, use context-based heuristic.
    context :
        Usage context for automatic format selection.
    dual_output :
        If True, generate both PNG and Vega-Lite JSON.
    residual_data: np.ndarray :

    """
    from dsa110_contimg.core.visualization.plot_context import PerformanceLogger

    flat = residual_data.flatten()

    # Performance logging
    with PerformanceLogger("residual_histogram", output_path, context, interactive) as perf:
        # Determine format based on context
        use_interactive = should_generate_interactive(context, interactive)

        # Dual output: generate both formats
        if dual_output:
            perf.log_format_selection("dual", "dual_output=True")
            _generate_residual_histogram_png(flat, stats, output_path, config)
            json_path = Path(str(output_path).replace(".png", ".vega.json"))
            _generate_residual_histogram_json(flat, stats, json_path)
            return

        # Interactive mode
        if use_interactive:
            reason = "explicit" if interactive else "context"
            perf.log_format_selection("vega-lite", reason)
            json_path = Path(str(output_path).replace(".png", ".vega.json"))
            _generate_residual_histogram_json(flat, stats, json_path)
            return

        # Static mode
        perf.log_format_selection("png", "default")
        _generate_residual_histogram_png(flat, stats, output_path, config)


def _generate_residual_histogram_json(
    flat: np.ndarray,
    stats: dict,
    output_path: Path,
) -> None:
    """Generate Vega-Lite JSON for residual histogram.

    Parameters
    ----------
    flat: np.ndarray :

    """
    from dsa110_contimg.core.visualization.vega_specs import (
        create_residual_histogram_spec,
        save_vega_spec,
    )

    # Prepare Gaussian fit for overlay
    gaussian_fit = {
        "mean": stats["mean"],
        "std": stats["std"],
        "amplitude": len(flat) * (flat.max() - flat.min()) / 100,
    }

    spec = create_residual_histogram_spec(
        residuals=flat,
        bin_count=100,
        gaussian_fit=gaussian_fit,
        title="Residual Histogram (Normality Check)",
    )
    save_vega_spec(spec, output_path)


def _generate_residual_histogram_png(
    flat: np.ndarray,
    stats: dict,
    output_path: str | Path,
    config: FigureConfig | None,
) -> None:
    """Generate PNG for residual histogram.

    Parameters
    ----------
    flat: np.ndarray :

    """
    _setup_matplotlib()

    if config is None:
        config = FigureConfig(style=PlotStyle.QUICKLOOK)

    with plt.rc_context(config.to_mpl_params()):
        fig, ax = plt.subplots(figsize=config.figsize)

        # Histogram
        ax.hist(flat, bins=100, density=True, alpha=0.7, color="blue", label="Residual")

        # Overlay Gaussian for comparison
        from scipy.stats import norm

        x = np.linspace(flat.min(), flat.max(), 200)
        gaussian = norm.pdf(x, loc=stats["mean"], scale=stats["std"])
        ax.plot(x, gaussian, "r--", linewidth=2, label="Gaussian Fit")

        ax.set_xlabel("Residual (Jy/beam)")
        ax.set_ylabel("Density")
        ax.set_title("Residual Histogram (Normality Check)")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Add stats text
        stats_text = (
            f"Mean: {stats['mean']:.2e}\n"
            f"RMS: {stats['rms']:.2e}\n"
            f"Normality p: {stats['normality_p']:.3f}"
        )
        ax.text(
            0.98,
            0.97,
            stats_text,
            transform=ax.transAxes,
            verticalalignment="top",
            horizontalalignment="right",
            bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"),
            color="black",
        )

        plt.tight_layout()
        plt.savefig(output_path, dpi=config.dpi, bbox_inches="tight")
        plt.close(fig)


def plot_dynamic_range_map(
    image_data: np.ndarray,
    source_mask: np.ndarray | None = None,
    output_path: str | Path = None,
    config: FigureConfig | None = None,
) -> None:
    """Plot image with overlaid dynamic range contours.

    Parameters
    ----------
    image_data :
        2D image array.
    source_mask :
        Boolean mask of source regions (optional).
    output_path :
        Path to save the plot.
    config :
        Figure configuration.
    image_data: np.ndarray :

    source_mask: Optional[np.ndarray] :
         (Default value = None)
    output_path : Union[str, Path]
         (Default value = None)
    config: Optional[FigureConfig] :
         (Default value = None)

    """
    _setup_matplotlib()

    if config is None:
        config = FigureConfig(style=PlotStyle.QUICKLOOK)

    with plt.rc_context(config.to_mpl_params()):
        fig, ax = plt.subplots(figsize=config.figsize)

        # Log-scale image
        vmax = np.max(image_data)
        vmin = max(
            np.min(image_data[image_data > 0]) if np.any(image_data > 0) else vmax / 1e6, vmax / 1e6
        )

        im = ax.imshow(
            np.abs(image_data),
            origin="lower",
            cmap=config.cmap,
            norm=plt.matplotlib.colors.LogNorm(vmin=vmin, vmax=vmax),
        )

        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Flux (Jy/beam)")

        # Overlay source mask if provided
        if source_mask is not None:
            ax.contour(
                source_mask.astype(float), levels=[0.5], colors="red", linewidths=1, alpha=0.5
            )

        ax.set_xlabel("X (pixels)")
        ax.set_ylabel("Y (pixels)")
        ax.set_title("Image Dynamic Range")

        plt.tight_layout()
        plt.savefig(output_path, dpi=config.dpi, bbox_inches="tight")
        plt.close(fig)
