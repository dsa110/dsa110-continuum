"""
Image comparison plotting utilities.

Provides functions for:
- Side-by-side image comparison
- Difference/residual maps
- Pixel-by-pixel scatter plots
- Quantitative comparison metrics (SSIM, chi-squared, etc.)

Essential for comparing model vs observed, different algorithms, or time epochs.
Adapted from eht-imaging comp_plots.py patterns.
"""

from __future__ import annotations

import logging
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


def compute_comparison_metrics(
    image1: NDArray,
    image2: NDArray,
    mask: NDArray | None = None,
) -> dict:
    """Compute quantitative comparison metrics between two images.

    Parameters
    ----------
    image1 :
        First image array (reference)
    image2 :
        Second image array (comparison)
    mask :
        Optional boolean mask (True = include pixel)

    Returns
    -------
    Dictionary with metrics
        rmse, mae, correlation, ssim, chi_squared,
    Dictionary with metrics
        rmse, mae, correlation, ssim, chi_squared,
        flux_ratio, peak_ratio

    """
    img1 = np.asarray(image1).astype(np.float64)
    img2 = np.asarray(image2).astype(np.float64)

    if img1.shape != img2.shape:
        raise ValueError(f"Image shapes must match: {img1.shape} vs {img2.shape}")

    if mask is not None:
        mask = np.asarray(mask).astype(bool)
        img1_flat = img1[mask]
        img2_flat = img2[mask]
    else:
        img1_flat = img1.flatten()
        img2_flat = img2.flatten()

    # Filter NaN/Inf
    valid = np.isfinite(img1_flat) & np.isfinite(img2_flat)
    img1_flat = img1_flat[valid]
    img2_flat = img2_flat[valid]

    if len(img1_flat) == 0:
        return {
            "rmse": np.nan,
            "mae": np.nan,
            "correlation": np.nan,
            "ssim": np.nan,
            "chi_squared": np.nan,
            "flux_ratio": np.nan,
            "peak_ratio": np.nan,
            "n_pixels": 0,
        }

    # Root Mean Square Error
    diff = img1_flat - img2_flat
    rmse = np.sqrt(np.mean(diff**2))

    # Mean Absolute Error
    mae = np.mean(np.abs(diff))

    # Pearson Correlation
    if np.std(img1_flat) > 0 and np.std(img2_flat) > 0:
        correlation = np.corrcoef(img1_flat, img2_flat)[0, 1]
    else:
        correlation = np.nan

    # Structural Similarity Index (simplified version)
    c1 = (0.01 * max(img1_flat.max(), img2_flat.max())) ** 2
    c2 = (0.03 * max(img1_flat.max(), img2_flat.max())) ** 2

    mu1, mu2 = np.mean(img1_flat), np.mean(img2_flat)
    sigma1, sigma2 = np.std(img1_flat), np.std(img2_flat)
    sigma12 = np.mean((img1_flat - mu1) * (img2_flat - mu2))

    ssim = ((2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)) / (
        (mu1**2 + mu2**2 + c1) * (sigma1**2 + sigma2**2 + c2)
    )

    # Chi-squared (assuming Poisson-like errors)
    with np.errstate(divide="ignore", invalid="ignore"):
        variance = np.abs(img1_flat) + 1e-10  # Avoid division by zero
        chi_sq = np.mean(diff**2 / variance)

    # Flux ratio (total flux)
    flux1 = np.sum(img1_flat)
    flux2 = np.sum(img2_flat)
    flux_ratio = flux2 / flux1 if flux1 != 0 else np.nan

    # Peak ratio
    peak1 = np.max(img1_flat)
    peak2 = np.max(img2_flat)
    peak_ratio = peak2 / peak1 if peak1 != 0 else np.nan

    return {
        "rmse": float(rmse),
        "mae": float(mae),
        "correlation": float(correlation),
        "ssim": float(ssim),
        "chi_squared": float(chi_sq),
        "flux_ratio": float(flux_ratio),
        "peak_ratio": float(peak_ratio),
        "n_pixels": int(len(img1_flat)),
    }


def plot_image_comparison(
    image1: NDArray,
    image2: NDArray,
    output: str | Path | None = None,
    config: FigureConfig | None = None,
    title1: str = "Image 1",
    title2: str = "Image 2",
    suptitle: str = "Image Comparison",
    show_difference: bool = True,
    show_ratio: bool = False,
    show_metrics: bool = True,
    shared_colorscale: bool = True,
    vmin: float | None = None,
    vmax: float | None = None,
    diff_symmetric: bool = True,
    wcs1: Any | None = None,
    wcs2: Any | None = None,
) -> Figure:
    """Create side-by-side image comparison with optional difference map.

    Parameters
    ----------
    image1 :
        First image array (reference)
    image2 :
        Second image array (comparison)
    output :
        Output file path
    config :
        Figure configuration
    title1 :
        Title for first image
    title2 :
        Title for second image
    suptitle :
        Overall figure title
    show_difference :
        Show difference map (image2 - image1)
    show_ratio :
        Show ratio map (image2 / image1)
    show_metrics :
        Display comparison metrics
    shared_colorscale :
        Use same colorscale for both images
    vmin :
        Manual minimum for colorscale
    vmax :
        Manual maximum for colorscale
    diff_symmetric :
        Use symmetric colorscale for difference
    wcs1 :
        WCS for image1 (for coordinate axes)
    wcs2 :
        WCS for image2 (for coordinate axes)

    Returns
    -------
        matplotlib Figure object

    """
    _setup_matplotlib()
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    if config is None:
        config = FigureConfig(style=PlotStyle.QUICKLOOK)

    img1 = np.asarray(image1)
    img2 = np.asarray(image2)

    # Determine number of panels
    n_panels = 2
    if show_difference:
        n_panels += 1
    if show_ratio:
        n_panels += 1

    fig_width = config.figsize[0] * n_panels / 2
    fig, axes = plt.subplots(1, n_panels, figsize=(fig_width, config.figsize[1]))

    if n_panels == 2:
        axes = [axes[0], axes[1]]

    # Determine colorscale
    if shared_colorscale:
        if vmin is None:
            vmin = min(np.nanmin(img1), np.nanmin(img2))
        if vmax is None:
            vmax = max(np.nanmax(img1), np.nanmax(img2))

    # Plot image 1
    ax_idx = 0
    im1 = axes[ax_idx].imshow(img1, origin="lower", cmap=config.cmap, vmin=vmin, vmax=vmax)
    axes[ax_idx].set_title(title1, fontsize=config.effective_title_size)
    divider1 = make_axes_locatable(axes[ax_idx])
    cax1 = divider1.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im1, cax=cax1, label="Flux (Jy/beam)")

    # Plot image 2
    ax_idx = 1
    im2 = axes[ax_idx].imshow(img2, origin="lower", cmap=config.cmap, vmin=vmin, vmax=vmax)
    axes[ax_idx].set_title(title2, fontsize=config.effective_title_size)
    divider2 = make_axes_locatable(axes[ax_idx])
    cax2 = divider2.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im2, cax=cax2, label="Flux (Jy/beam)")

    # Plot difference map
    if show_difference:
        ax_idx += 1
        diff = img2 - img1
        if diff_symmetric:
            diff_max = np.nanmax(np.abs(diff))
            diff_vmin, diff_vmax = -diff_max, diff_max
            diff_cmap = "RdBu_r"
        else:
            diff_vmin, diff_vmax = np.nanmin(diff), np.nanmax(diff)
            diff_cmap = "coolwarm"

        im_diff = axes[ax_idx].imshow(
            diff, origin="lower", cmap=diff_cmap, vmin=diff_vmin, vmax=diff_vmax
        )
        axes[ax_idx].set_title(
            f"Difference ({title2} - {title1})", fontsize=config.effective_title_size
        )
        divider_diff = make_axes_locatable(axes[ax_idx])
        cax_diff = divider_diff.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(im_diff, cax=cax_diff, label="Δ Flux")

    # Plot ratio map
    if show_ratio:
        ax_idx += 1
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(np.abs(img1) > 1e-10, img2 / img1, np.nan)

        # Clip extreme ratios for display
        ratio_clipped = np.clip(ratio, 0.1, 10)

        im_ratio = axes[ax_idx].imshow(
            ratio_clipped, origin="lower", cmap="PiYG", vmin=0.5, vmax=2.0
        )
        axes[ax_idx].set_title(f"Ratio ({title2} / {title1})", fontsize=config.effective_title_size)
        divider_ratio = make_axes_locatable(axes[ax_idx])
        cax_ratio = divider_ratio.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(im_ratio, cax=cax_ratio, label="Ratio")

    # Add axis labels
    for ax in axes:
        ax.set_xlabel("X (pixels)")
        ax.set_ylabel("Y (pixels)")

    # Compute and display metrics
    if show_metrics:
        metrics = compute_comparison_metrics(img1, img2)
        metrics_text = (
            f"RMSE: {metrics['rmse']:.2e}\n"
            f"Correlation: {metrics['correlation']:.4f}\n"
            f"SSIM: {metrics['ssim']:.4f}\n"
            f"Flux Ratio: {metrics['flux_ratio']:.4f}"
        )
        fig.text(
            0.02,
            0.02,
            metrics_text,
            transform=fig.transFigure,
            fontsize=config.effective_tick_size,
            verticalalignment="bottom",
            bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"),
        )

    fig.suptitle(suptitle, fontsize=config.effective_title_size + 2)
    fig.tight_layout()

    if output:
        fig.savefig(output, dpi=config.dpi, bbox_inches="tight")
        logger.info(f"Saved image comparison: {output}")
        plt.close(fig)

    return fig


def plot_pixel_scatter(
    image1: NDArray,
    image2: NDArray,
    output: str | Path | None = None,
    config: FigureConfig | None = None,
    title: str = "Pixel-by-Pixel Comparison",
    xlabel: str = "Reference Flux (Jy/beam)",
    ylabel: str = "Comparison Flux (Jy/beam)",
    mask: NDArray | None = None,
    show_unity: bool = True,
    show_fit: bool = True,
    density_plot: bool = True,
    log_scale: bool = False,
) -> Figure:
    """Create pixel-by-pixel scatter plot comparing two images.

    Parameters
    ----------
    image1 :
        First image array (reference, x-axis)
    image2 :
        Second image array (comparison, y-axis)
    output :
        Output file path
    config :
        Figure configuration
    title :
        Plot title
    xlabel :
        X-axis label
    ylabel :
        Y-axis label
    mask :
        Optional boolean mask (True = include pixel)
    show_unity :
        Show 1:1 diagonal line
    show_fit :
        Show linear fit
    density_plot :
        Use 2D histogram for dense data
    log_scale :
        Use log scale for axes

    Returns
    -------
        matplotlib Figure object

    """
    _setup_matplotlib()
    import matplotlib.pyplot as plt
    from scipy import stats

    if config is None:
        config = FigureConfig(style=PlotStyle.QUICKLOOK)

    img1 = np.asarray(image1).flatten()
    img2 = np.asarray(image2).flatten()

    if mask is not None:
        mask = np.asarray(mask).flatten().astype(bool)
        img1 = img1[mask]
        img2 = img2[mask]

    # Filter valid data
    valid = np.isfinite(img1) & np.isfinite(img2)
    if log_scale:
        valid &= (img1 > 0) & (img2 > 0)
    img1 = img1[valid]
    img2 = img2[valid]

    fig, ax = plt.subplots(figsize=config.figsize, dpi=config.dpi)

    if density_plot and len(img1) > 1000:
        # 2D histogram for dense data
        from matplotlib.colors import LogNorm

        h, xedges, yedges, im = ax.hist2d(img1, img2, bins=100, cmap="viridis", norm=LogNorm())
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("Count", fontsize=config.effective_label_size)
    else:
        ax.scatter(img1, img2, s=1, alpha=0.3, c="blue")

    # Determine axis limits
    all_vals = np.concatenate([img1, img2])
    if log_scale:
        lims = [all_vals.min() * 0.8, all_vals.max() * 1.2]
    else:
        margin = 0.1 * (all_vals.max() - all_vals.min())
        lims = [all_vals.min() - margin, all_vals.max() + margin]

    # Unity line
    if show_unity:
        ax.plot(lims, lims, "k--", alpha=0.7, linewidth=1.5, label="1:1")

    # Linear fit
    if show_fit:
        if log_scale:
            slope, intercept, r_value, _, _ = stats.linregress(np.log10(img1), np.log10(img2))
            fit_x = np.array(lims)
            fit_y = 10 ** (slope * np.log10(fit_x) + intercept)
            fit_label = f"Fit: slope={slope:.3f}, r²={r_value**2:.3f}"
        else:
            slope, intercept, r_value, _, _ = stats.linregress(img1, img2)
            fit_x = np.array(lims)
            fit_y = slope * fit_x + intercept
            fit_label = f"Fit: y={slope:.3f}x+{intercept:.2e}, r²={r_value**2:.3f}"

        ax.plot(fit_x, fit_y, "r-", alpha=0.7, linewidth=1.5, label=fit_label)

    ax.set_xlim(lims)
    ax.set_ylim(lims)

    if log_scale:
        ax.set_xscale("log")
        ax.set_yscale("log")

    ax.set_xlabel(xlabel, fontsize=config.effective_label_size)
    ax.set_ylabel(ylabel, fontsize=config.effective_label_size)
    ax.set_title(title, fontsize=config.effective_title_size)
    ax.legend(fontsize=config.effective_tick_size, loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")

    fig.tight_layout()

    if output:
        fig.savefig(output, dpi=config.dpi, bbox_inches="tight")
        logger.info(f"Saved pixel scatter plot: {output}")
        plt.close(fig)

    return fig


def plot_residual_map(
    image: NDArray,
    model: NDArray,
    output: str | Path | None = None,
    config: FigureConfig | None = None,
    title: str = "Residual Map (Data - Model)",
    show_histogram: bool = True,
    sigma_clip: float = 3.0,
    wcs: Any | None = None,
) -> Figure:
    """Create residual map with optional histogram panel.

    Parameters
    ----------
    image :
        Observed image array
    model :
        Model image array
    output :
        Output file path
    config :
        Figure configuration
    title :
        Plot title
    show_histogram :
        Show residual histogram
    sigma_clip :
        Sigma level for colorscale clipping
    wcs :
        WCS for coordinate axes

    Returns
    -------
        matplotlib Figure object

    """
    _setup_matplotlib()
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    if config is None:
        config = FigureConfig(style=PlotStyle.QUICKLOOK)

    img = np.asarray(image)
    mod = np.asarray(model)
    residual = img - mod

    # Calculate statistics
    res_flat = residual[np.isfinite(residual)]
    rms = np.std(res_flat)
    mean = np.mean(res_flat)

    if show_histogram:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(config.figsize[0] * 2, config.figsize[1]))
    else:
        fig, ax1 = plt.subplots(figsize=config.figsize)

    # Residual map
    vmax = sigma_clip * rms
    vmin = -vmax

    im = ax1.imshow(residual, origin="lower", cmap="RdBu_r", vmin=vmin, vmax=vmax)
    ax1.set_title(title, fontsize=config.effective_title_size)
    ax1.set_xlabel("X (pixels)")
    ax1.set_ylabel("Y (pixels)")

    divider = make_axes_locatable(ax1)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im, cax=cax, label="Residual (Jy/beam)")

    # Statistics annotation
    stats_text = f"RMS: {rms:.2e}\nMean: {mean:.2e}"
    ax1.text(
        0.02,
        0.98,
        stats_text,
        transform=ax1.transAxes,
        verticalalignment="top",
        fontsize=config.effective_tick_size,
        bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"),
    )

    # Histogram
    if show_histogram:
        from scipy.stats import norm

        ax2.hist(
            res_flat,
            bins=100,
            density=True,
            alpha=0.7,
            color="steelblue",
            edgecolor="none",
            label="Residuals",
        )

        # Gaussian fit overlay
        if rms > 1e-10:
            x = np.linspace(res_flat.min(), res_flat.max(), 200)
            gaussian = norm.pdf(x, loc=mean, scale=rms)
            ax2.plot(x, gaussian, "r-", linewidth=2, label=f"Gaussian (σ={rms:.2e})")

        ax2.set_xlabel("Residual Value (Jy/beam)", fontsize=config.effective_label_size)
        ax2.set_ylabel("Density", fontsize=config.effective_label_size)
        ax2.set_title("Residual Distribution", fontsize=config.effective_title_size)
        ax2.legend(fontsize=config.effective_tick_size)
        ax2.grid(True, alpha=0.3)

        # Mark sigma levels
        for sigma in [-3, -2, -1, 1, 2, 3]:
            ax2.axvline(mean + sigma * rms, color="gray", linestyle=":", alpha=0.5)

    fig.tight_layout()

    if output:
        fig.savefig(output, dpi=config.dpi, bbox_inches="tight")
        logger.info(f"Saved residual map: {output}")
        plt.close(fig)

    return fig


def compare_fits_images(
    fits_path1: str | Path,
    fits_path2: str | Path,
    output: str | Path | None = None,
    config: FigureConfig | None = None,
    **kwargs: Any,
) -> tuple[Figure, dict]:
    """Compare two FITS images with automatic loading.

    Parameters
    ----------
    fits_path1 :
        Path to first FITS file (reference)
    fits_path2 :
        Path to second FITS file (comparison)
    output :
        Output file path
    config :
        Figure configuration
    **kwargs :
        Additional arguments passed to plot_image_comparison
    fits_path1 : Union[str, Path]
    fits_path2: Union[str :

    output : Optional[Union[str, Path]]
         (Default value = None)
    config: Optional[FigureConfig] :
         (Default value = None)
    **kwargs: Any :


    Returns
    -------
        Tuple of (Figure, metrics_dict)

    """
    from dsa110_contimg.common.utils.fits_utils import get_2d_data_and_wcs

    fits_path1 = Path(fits_path1)
    fits_path2 = Path(fits_path2)

    data1, wcs1, _ = get_2d_data_and_wcs(fits_path1)
    data2, wcs2, _ = get_2d_data_and_wcs(fits_path2)

    # Compute metrics
    metrics = compute_comparison_metrics(data1, data2)

    # Set default titles from filenames
    kwargs.setdefault("title1", fits_path1.name)
    kwargs.setdefault("title2", fits_path2.name)

    fig = plot_image_comparison(
        data1,
        data2,
        output=output,
        config=config,
        wcs1=wcs1,
        wcs2=wcs2,
        **kwargs,
    )

    return fig, metrics
