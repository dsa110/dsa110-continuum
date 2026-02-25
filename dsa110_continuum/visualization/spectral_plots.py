"""
Spectral index and multi-frequency visualization utilities.

Provides functions for:
- Spectral index maps (α where S ∝ ν^α)
- Spectral Energy Distribution (SED) plots
- Multi-frequency image mosaics/comparisons
- Spectral index error maps

Spectral index is defined as:
    S(ν) = S₀ * (ν/ν₀)^α

where α is the spectral index (typically negative for synchrotron emission).

Common spectral indices:
- Synchrotron: α ≈ -0.7 to -1.0
- Free-free (thermal): α ≈ -0.1
- Thermal dust: α ≈ +2 to +4
- Pulsars: α ≈ -1.4 to -1.8

References
----------
- Condon & Ransom: "Essential Radio Astronomy"
- MUFFIN multi-frequency imaging patterns
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


def compute_spectral_index(
    flux_low: NDArray,
    flux_high: NDArray,
    freq_low_hz: float,
    freq_high_hz: float,
    flux_err_low: NDArray | None = None,
    flux_err_high: NDArray | None = None,
    min_snr: float = 3.0,
) -> tuple[NDArray, NDArray | None]:
    """Compute spectral index from two-frequency measurements.

    α = log(S₂/S₁) / log(ν₂/ν₁)

    Parameters
    ----------
    flux_low :
        Flux density at lower frequency (Jy or Jy/beam)
    flux_high :
        Flux density at higher frequency
    freq_low_hz :
        Lower frequency in Hz
    freq_high_hz :
        Higher frequency in Hz
    flux_err_low :
        Flux error at lower frequency (optional)
    flux_err_high :
        Flux error at higher frequency (optional)
    min_snr :
        Minimum S/N for valid spectral index calculation

    Returns
    -------
        Tuple of (spectral_index, spectral_index_error)
        Error is None if flux errors not provided

    """
    s1 = np.asarray(flux_low).astype(np.float64)
    s2 = np.asarray(flux_high).astype(np.float64)

    # Log frequency ratio
    log_freq_ratio = np.log(freq_high_hz / freq_low_hz)

    # Compute spectral index where valid
    with np.errstate(divide="ignore", invalid="ignore"):
        # Mask invalid regions
        valid = (s1 > 0) & (s2 > 0) & np.isfinite(s1) & np.isfinite(s2)

        # Apply SNR cut if errors provided
        if flux_err_low is not None and flux_err_high is not None:
            err1 = np.asarray(flux_err_low)
            err2 = np.asarray(flux_err_high)
            snr1 = np.abs(s1) / (err1 + 1e-30)
            snr2 = np.abs(s2) / (err2 + 1e-30)
            valid &= (snr1 >= min_snr) & (snr2 >= min_snr)

        alpha = np.where(valid, np.log(s2 / s1) / log_freq_ratio, np.nan)

    # Compute error propagation if errors provided
    alpha_err = None
    if flux_err_low is not None and flux_err_high is not None:
        err1 = np.asarray(flux_err_low)
        err2 = np.asarray(flux_err_high)

        with np.errstate(divide="ignore", invalid="ignore"):
            # Error propagation: σ_α = (1/ln(ν₂/ν₁)) * sqrt((σ₁/S₁)² + (σ₂/S₂)²)
            rel_err_sq = (err1 / s1) ** 2 + (err2 / s2) ** 2
            alpha_err = np.where(valid, np.sqrt(rel_err_sq) / abs(log_freq_ratio), np.nan)

    return alpha, alpha_err


def plot_spectral_index_map(
    alpha: NDArray,
    output: str | Path | None = None,
    config: FigureConfig | None = None,
    title: str = "Spectral Index Map",
    alpha_err: NDArray | None = None,
    vmin: float = -2.0,
    vmax: float = 1.0,
    wcs: Any | None = None,
    show_colorbar: bool = True,
    cmap: str = "RdBu_r",
    overlay_contours: NDArray | None = None,
    contour_levels: list[float] | None = None,
) -> Figure:
    """Plot spectral index map.

    Parameters
    ----------
    alpha :
        2D spectral index array
    output :
        Output file path
    config :
        Figure configuration
    title :
        Plot title
    alpha_err :
        Optional spectral index error map
    vmin :
        Minimum spectral index for colorscale
    vmax :
        Maximum spectral index for colorscale
    wcs :
        WCS object for coordinate axes
    show_colorbar :
        Show colorbar
    cmap :
        Colormap (RdBu_r good for showing α=0 at white)
    overlay_contours :
        Optional intensity image for contour overlay
    contour_levels :
        Contour levels (default: auto)

    Returns
    -------
        matplotlib Figure object

    """
    _setup_matplotlib()
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    if config is None:
        config = FigureConfig(style=PlotStyle.QUICKLOOK)

    alpha = np.asarray(alpha)

    # Create figure with WCS if available
    if wcs is not None:
        try:
            from astropy.visualization.wcsaxes import WCSAxes  # noqa: F401

            fig = plt.figure(figsize=config.figsize, dpi=config.dpi)
            ax = fig.add_subplot(111, projection=wcs)
        except (ImportError, TypeError):
            fig, ax = plt.subplots(figsize=config.figsize, dpi=config.dpi)
    else:
        fig, ax = plt.subplots(figsize=config.figsize, dpi=config.dpi)

    # Plot spectral index
    im = ax.imshow(
        alpha,
        origin="lower",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        aspect="equal",
    )

    # Overlay intensity contours
    if overlay_contours is not None:
        if contour_levels is None:
            # Auto-generate levels at 10%, 30%, 50%, 70%, 90% of peak
            peak = np.nanmax(overlay_contours)
            contour_levels = peak * np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        ax.contour(
            overlay_contours,
            levels=contour_levels,
            colors="black",
            linewidths=0.5,
            alpha=0.7,
        )

    # Colorbar
    if show_colorbar:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = fig.colorbar(im, cax=cax)
        cbar.set_label("Spectral Index (α)", fontsize=config.effective_label_size)

    # Labels
    if wcs is not None:
        ax.set_xlabel("RA", fontsize=config.effective_label_size)
        ax.set_ylabel("Dec", fontsize=config.effective_label_size)
    else:
        ax.set_xlabel("X (pixels)", fontsize=config.effective_label_size)
        ax.set_ylabel("Y (pixels)", fontsize=config.effective_label_size)

    ax.set_title(title, fontsize=config.effective_title_size)

    # Add statistics annotation
    valid_alpha = alpha[np.isfinite(alpha)]
    if len(valid_alpha) > 0:
        stats_text = (
            f"Median α: {np.median(valid_alpha):.2f}\n"
            f"Mean α: {np.mean(valid_alpha):.2f}\n"
            f"Std α: {np.std(valid_alpha):.2f}"
        )
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
        logger.info(f"Saved spectral index map: {output}")
        plt.close(fig)

    return fig


def plot_spectral_index_error_map(
    alpha_err: NDArray,
    output: str | Path | None = None,
    config: FigureConfig | None = None,
    title: str = "Spectral Index Error Map",
    vmax: float | None = None,
    wcs: Any | None = None,
) -> Figure:
    """Plot spectral index error map.

    Parameters
    ----------
    alpha_err :
        2D spectral index error array
    output :
        Output file path
    config :
        Figure configuration
    title :
        Plot title
    vmax :
        Maximum error for colorscale (default: auto)
    wcs :
        WCS object for coordinate axes

    Returns
    -------
        matplotlib Figure object

    """
    _setup_matplotlib()
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    if config is None:
        config = FigureConfig(style=PlotStyle.QUICKLOOK)

    alpha_err = np.asarray(alpha_err)

    if vmax is None:
        vmax = np.nanpercentile(alpha_err, 95)

    fig, ax = plt.subplots(figsize=config.figsize, dpi=config.dpi)

    im = ax.imshow(
        alpha_err,
        origin="lower",
        cmap="plasma",
        vmin=0,
        vmax=vmax,
        aspect="equal",
    )

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("σ_α", fontsize=config.effective_label_size)

    ax.set_xlabel("X (pixels)", fontsize=config.effective_label_size)
    ax.set_ylabel("Y (pixels)", fontsize=config.effective_label_size)
    ax.set_title(title, fontsize=config.effective_title_size)

    fig.tight_layout()

    if output:
        fig.savefig(output, dpi=config.dpi, bbox_inches="tight")
        logger.info(f"Saved spectral index error map: {output}")
        plt.close(fig)

    return fig


def plot_sed(
    frequencies_hz: NDArray,
    flux_densities: NDArray,
    output: str | Path | None = None,
    config: FigureConfig | None = None,
    title: str = "Spectral Energy Distribution",
    flux_errors: NDArray | None = None,
    source_name: str = "",
    fit_powerlaw: bool = True,
    freq_unit: str = "GHz",
    flux_unit: str = "mJy",
    reference_alpha: float | None = None,
) -> Figure:
    """Plot Spectral Energy Distribution (SED) for a source.

    Parameters
    ----------
    frequencies_hz :
        Frequency array in Hz
    flux_densities :
        Flux density array in Jy
    output :
        Output file path
    config :
        Figure configuration
    title :
        Plot title
    flux_errors :
        Flux density errors in Jy
    source_name :
        Source name for annotation
    fit_powerlaw :
        Fit and overlay power-law model
    freq_unit :
        Unit for frequency axis ("Hz", "MHz", "GHz")
    flux_unit :
        Unit for flux axis ("Jy", "mJy", "μJy")
    reference_alpha :
        Reference spectral index to plot (e.g., -0.7 for synchrotron)

    Returns
    -------
        matplotlib Figure object

    """
    _setup_matplotlib()
    import matplotlib.pyplot as plt
    from scipy.optimize import curve_fit

    if config is None:
        config = FigureConfig(style=PlotStyle.QUICKLOOK)

    freq = np.asarray(frequencies_hz)
    flux = np.asarray(flux_densities)

    # Convert units
    freq_scale = {"Hz": 1.0, "MHz": 1e-6, "GHz": 1e-9}[freq_unit]
    flux_scale = {"Jy": 1.0, "mJy": 1e3, "μJy": 1e6, "uJy": 1e6}[flux_unit]

    freq_plot = freq * freq_scale
    flux_plot = flux * flux_scale

    fig, ax = plt.subplots(figsize=config.figsize, dpi=config.dpi)

    # Plot data points
    if flux_errors is not None:
        flux_err_plot = np.asarray(flux_errors) * flux_scale
        ax.errorbar(
            freq_plot,
            flux_plot,
            yerr=flux_err_plot,
            fmt="o",
            color="steelblue",
            markersize=8,
            capsize=3,
            label="Data",
        )
    else:
        ax.scatter(freq_plot, flux_plot, s=50, color="steelblue", label="Data")

    # Fit power law
    if fit_powerlaw and len(freq) >= 2:

        def powerlaw(nu, s0, alpha):
            nu0 = freq_plot[0]
            return s0 * (nu / nu0) ** alpha

        try:
            valid = (flux > 0) & np.isfinite(flux)
            if flux_errors is not None:
                valid &= flux_errors > 0
                weights = 1.0 / (np.asarray(flux_errors)[valid] * flux_scale) ** 2
            else:
                weights = None

            p0 = [flux_plot[valid][0], -0.7]
            popt, pcov = curve_fit(
                powerlaw,
                freq_plot[valid],
                flux_plot[valid],
                p0=p0,
                sigma=1.0 / np.sqrt(weights) if weights is not None else None,
                absolute_sigma=True,
                maxfev=5000,
            )

            s0_fit, alpha_fit = popt
            alpha_err = np.sqrt(pcov[1, 1]) if np.isfinite(pcov[1, 1]) else 0

            # Plot fit
            freq_fine = np.logspace(
                np.log10(freq_plot.min() * 0.8), np.log10(freq_plot.max() * 1.2), 100
            )
            flux_fit = powerlaw(freq_fine, s0_fit, alpha_fit)
            ax.plot(
                freq_fine,
                flux_fit,
                "r-",
                linewidth=2,
                label=f"Fit: α = {alpha_fit:.2f} ± {alpha_err:.2f}",
            )

        except Exception as e:
            logger.warning(f"Power-law fit failed: {e}")

    # Reference spectral index
    if reference_alpha is not None:
        freq_fine = np.logspace(
            np.log10(freq_plot.min() * 0.8), np.log10(freq_plot.max() * 1.2), 100
        )
        s0_ref = flux_plot[0]
        nu0_ref = freq_plot[0]
        flux_ref = s0_ref * (freq_fine / nu0_ref) ** reference_alpha
        ax.plot(
            freq_fine,
            flux_ref,
            "k--",
            linewidth=1.5,
            alpha=0.5,
            label=f"Reference α = {reference_alpha}",
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(f"Frequency ({freq_unit})", fontsize=config.effective_label_size)
    ax.set_ylabel(f"Flux Density ({flux_unit})", fontsize=config.effective_label_size)

    plot_title = title
    if source_name:
        plot_title = f"{title}: {source_name}"
    ax.set_title(plot_title, fontsize=config.effective_title_size)

    ax.legend(fontsize=config.effective_tick_size, loc="best")
    ax.grid(True, alpha=0.3, which="both")

    fig.tight_layout()

    if output:
        fig.savefig(output, dpi=config.dpi, bbox_inches="tight")
        logger.info(f"Saved SED plot: {output}")
        plt.close(fig)

    return fig


def plot_multi_frequency_mosaic(
    images: list[NDArray],
    frequencies_hz: list[float],
    output: str | Path | None = None,
    config: FigureConfig | None = None,
    title: str = "Multi-Frequency Images",
    freq_unit: str = "GHz",
    share_colorscale: bool = False,
    vmin: float | None = None,
    vmax: float | None = None,
    ncols: int = 3,
) -> Figure:
    """Plot mosaic of images at different frequencies.

    Parameters
    ----------
    images :
        List of 2D image arrays
    frequencies_hz :
        List of frequencies in Hz
    output :
        Output file path
    config :
        Figure configuration
    title :
        Overall title
    freq_unit :
        Unit for frequency labels
    share_colorscale :
        Use same colorscale for all panels
    vmin :
        Minimum for shared colorscale
    vmax :
        Maximum for shared colorscale
    ncols :
        Number of columns in mosaic
    images: List["NDArray"] :

    frequencies_hz: List[float] :

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
    from mpl_toolkits.axes_grid1 import ImageGrid

    if config is None:
        config = FigureConfig(style=PlotStyle.QUICKLOOK)

    n_images = len(images)
    nrows = int(np.ceil(n_images / ncols))

    freq_scale = {"Hz": 1.0, "MHz": 1e-6, "GHz": 1e-9}[freq_unit]

    # Determine colorscale
    if share_colorscale:
        if vmin is None:
            vmin = min(np.nanmin(img) for img in images)
        if vmax is None:
            vmax = max(np.nanmax(img) for img in images)

    fig = plt.figure(figsize=(config.figsize[0] * ncols / 2, config.figsize[1] * nrows / 2))
    grid = ImageGrid(
        fig,
        111,
        nrows_ncols=(nrows, ncols),
        axes_pad=0.3,
        cbar_location="right",
        cbar_mode="single" if share_colorscale else "each",
        cbar_size="5%",
        cbar_pad=0.1,
    )

    for idx, (img, freq) in enumerate(zip(images, frequencies_hz)):
        ax = grid[idx]

        img_vmin = vmin if share_colorscale else np.nanpercentile(img, 1)
        img_vmax = vmax if share_colorscale else np.nanpercentile(img, 99)

        im = ax.imshow(
            img,
            origin="lower",
            cmap=config.cmap,
            vmin=img_vmin,
            vmax=img_vmax,
        )

        freq_label = freq * freq_scale
        ax.set_title(f"{freq_label:.2f} {freq_unit}", fontsize=config.effective_tick_size)
        ax.tick_params(labelsize=config.effective_tick_size - 2)

        if not share_colorscale:
            grid.cbar_axes[idx].colorbar(im)

    # Hide unused axes
    for idx in range(n_images, nrows * ncols):
        grid[idx].set_visible(False)

    if share_colorscale:
        grid.cbar_axes[0].colorbar(im)

    fig.suptitle(title, fontsize=config.effective_title_size)

    if output:
        fig.savefig(output, dpi=config.dpi, bbox_inches="tight")
        logger.info(f"Saved multi-frequency mosaic: {output}")
        plt.close(fig)

    return fig


def plot_spectral_index_histogram(
    alpha: NDArray,
    output: str | Path | None = None,
    config: FigureConfig | None = None,
    title: str = "Spectral Index Distribution",
    bins: int = 50,
    alpha_range: tuple[float, float] = (-3.0, 2.0),
    reference_values: dict | None = None,
) -> Figure:
    """Plot histogram of spectral index values.

    Parameters
    ----------
    alpha :
        Spectral index array (any shape, will be flattened)
    output :
        Output file path
    config :
        Figure configuration
    title :
        Plot title
    bins :
        Number of histogram bins
    alpha_range :
        Range for histogram
    reference_values :
        Dict of reference α values to mark, e.g.,
        {"Synchrotron": -0.7, "Free-free": -0.1}

    Returns
    -------
        matplotlib Figure object

    """
    _setup_matplotlib()
    import matplotlib.pyplot as plt

    if config is None:
        config = FigureConfig(style=PlotStyle.QUICKLOOK)

    alpha_flat = np.asarray(alpha).flatten()
    alpha_valid = alpha_flat[np.isfinite(alpha_flat)]

    fig, ax = plt.subplots(figsize=config.figsize, dpi=config.dpi)

    ax.hist(
        alpha_valid,
        bins=bins,
        range=alpha_range,
        color="steelblue",
        alpha=0.7,
        edgecolor="black",
        linewidth=0.5,
    )

    # Reference values
    if reference_values is not None:
        colors = plt.cm.Set1(np.linspace(0, 1, len(reference_values)))
        for (name, alpha_ref), color in zip(reference_values.items(), colors):
            ax.axvline(
                alpha_ref, color=color, linestyle="--", linewidth=2, label=f"{name} (α={alpha_ref})"
            )

    # Statistics
    mean_alpha = np.mean(alpha_valid)
    median_alpha = np.median(alpha_valid)
    std_alpha = np.std(alpha_valid)

    ax.axvline(
        median_alpha, color="red", linestyle="-", linewidth=2, label=f"Median={median_alpha:.2f}"
    )

    stats_text = f"N pixels: {len(alpha_valid):,}\nMean: {mean_alpha:.2f}\nMedian: {median_alpha:.2f}\nStd: {std_alpha:.2f}"
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

    ax.set_xlabel("Spectral Index (α)", fontsize=config.effective_label_size)
    ax.set_ylabel("Count", fontsize=config.effective_label_size)
    ax.set_title(title, fontsize=config.effective_title_size)
    ax.legend(fontsize=config.effective_tick_size - 1, loc="upper left")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()

    if output:
        fig.savefig(output, dpi=config.dpi, bbox_inches="tight")
        logger.info(f"Saved spectral index histogram: {output}")
        plt.close(fig)

    return fig


def compute_spectral_index_from_fits(
    fits_low: str | Path,
    fits_high: str | Path,
    freq_low_hz: float | None = None,
    freq_high_hz: float | None = None,
    rms_low: float | None = None,
    rms_high: float | None = None,
    min_snr: float = 3.0,
) -> tuple[NDArray, NDArray | None, dict]:
    """Compute spectral index map from two FITS images.

    Parameters
    ----------
    fits_low :
        Path to lower frequency FITS image
    fits_high :
        Path to higher frequency FITS image
    freq_low_hz :
        Lower frequency in Hz (extracted from FITS if None)
    freq_high_hz :
        Higher frequency in Hz (extracted from FITS if None)
    rms_low :
        RMS noise at lower frequency (estimated if None)
    rms_high :
        RMS noise at higher frequency (estimated if None)
    min_snr :
        Minimum SNR for valid spectral index
    fits_low : Union[str, Path]
    fits_high : Union[str, freq_low_hz: Optional[float]
         (Default value = None)
    freq_high_hz: Optional[float] :
         (Default value = None)
    rms_low: Optional[float] :
         (Default value = None)
    rms_high: Optional[float] :
         (Default value = None)

    Returns
    -------
        Tuple of (spectral_index_map, error_map, metadata_dict)

    """
    from astropy.io import fits
    from astropy.wcs import WCS

    # Load images
    with fits.open(fits_low) as hdul_low:
        data_low = hdul_low[0].data
        header_low = hdul_low[0].header
        wcs = WCS(header_low, naxis=2)

    with fits.open(fits_high) as hdul_high:
        data_high = hdul_high[0].data
        header_high = hdul_high[0].header

    # Handle multi-dimensional data
    while data_low.ndim > 2:
        data_low = data_low[0]
    while data_high.ndim > 2:
        data_high = data_high[0]

    # Extract frequencies from headers if not provided
    if freq_low_hz is None:
        freq_low_hz = (
            header_low.get("CRVAL3") or header_low.get("RESTFRQ") or header_low.get("FREQ", 1.4e9)
        )
    if freq_high_hz is None:
        freq_high_hz = (
            header_high.get("CRVAL3")
            or header_high.get("RESTFRQ")
            or header_high.get("FREQ", 1.4e9)
        )

    # Estimate RMS if not provided (from image corners)
    if rms_low is None:
        corner_size = min(50, data_low.shape[0] // 4, data_low.shape[1] // 4)
        rms_low = np.nanstd(data_low[:corner_size, :corner_size])
    if rms_high is None:
        corner_size = min(50, data_high.shape[0] // 4, data_high.shape[1] // 4)
        rms_high = np.nanstd(data_high[:corner_size, :corner_size])

    # Create error maps
    err_low = np.full_like(data_low, rms_low)
    err_high = np.full_like(data_high, rms_high)

    # Compute spectral index
    alpha, alpha_err = compute_spectral_index(
        data_low,
        data_high,
        freq_low_hz,
        freq_high_hz,
        err_low,
        err_high,
        min_snr=min_snr,
    )

    metadata = {
        "freq_low_hz": freq_low_hz,
        "freq_high_hz": freq_high_hz,
        "rms_low": rms_low,
        "rms_high": rms_high,
        "wcs": wcs,
    }

    return alpha, alpha_err, metadata
