"""
Photometry visualization utilities.

Provides functions for:
- Aperture photometry overlays with radial profiles
- SNR maps with contours
- Catalog cross-match comparisons (NVSS, etc.)
- Field source detection plots
- Photometry summary panels

Follows the established visualization patterns from fits_plots.py and source_plots.py.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from astropy.wcs import WCS
    from matplotlib.figure import Figure
    from numpy.typing import NDArray

from dsa110_contimg.core.visualization.config import FigureConfig, PlotStyle

logger = logging.getLogger(__name__)


def _setup_matplotlib() -> None:
    """Configure matplotlib for headless operation."""
    import matplotlib

    matplotlib.use("Agg")


def plot_aperture_photometry(
    data: NDArray,
    ra_deg: float,
    dec_deg: float,
    wcs: WCS,
    aperture_radius_pix: float,
    annulus_inner_pix: float,
    annulus_outer_pix: float,
    peak_flux: float,
    local_rms: float,
    output: str | Path | None = None,
    config: FigureConfig | None = None,
    source_name: str | None = None,
    title: str | None = None,
) -> Figure:
    """Plot aperture photometry with radial profile.

    Creates a 2-panel figure:
    - Left: Image cutout with aperture and annulus overlays
    - Right: Radial profile showing signal vs radius

    Parameters
    ----------
    data :
        2D image array
    ra_deg :
        Right ascension in degrees
    dec_deg :
        Declination in degrees
    wcs :
        WCS object for coordinate transformation
    aperture_radius_pix :
        Aperture radius in pixels
    annulus_inner_pix :
        Annulus inner radius in pixels
    annulus_outer_pix :
        Annulus outer radius in pixels
    peak_flux :
        Measured peak flux in Jy/beam
    local_rms :
        Local RMS noise in Jy/beam
    output :
        Output file path
    config :
        Figure configuration
    source_name :
        Source name for title
    title :
        Custom title (overrides source_name)

    Returns
    -------
        matplotlib Figure object

    """
    _setup_matplotlib()
    import matplotlib.pyplot as plt
    from astropy.coordinates import SkyCoord
    from matplotlib.patches import Circle

    if config is None:
        config = FigureConfig(style=PlotStyle.QUICKLOOK)

    # Convert sky coordinates to pixel
    coord = SkyCoord(ra=ra_deg, dec=dec_deg, unit="deg")
    pix = wcs.world_to_pixel(coord)
    cx, cy = float(pix[0]), float(pix[1])

    # Create cutout region
    cutout_size = int(annulus_outer_pix * 2.5)
    y0 = max(0, int(cy - cutout_size))
    y1 = min(data.shape[0], int(cy + cutout_size))
    x0 = max(0, int(cx - cutout_size))
    x1 = min(data.shape[1], int(cx + cutout_size))
    cutout = data[y0:y1, x0:x1]

    # Compute radial profile
    y_grid, x_grid = np.ogrid[: data.shape[0], : data.shape[1]]
    r_grid = np.sqrt((x_grid - cx) ** 2 + (y_grid - cy) ** 2)
    max_radius = int(annulus_outer_pix * 1.5)
    radii = np.arange(0, max_radius, 1)
    profile = []
    for r in radii:
        mask = (r_grid >= r) & (r_grid < r + 1)
        if np.sum(mask) > 0:
            profile.append(np.nanmean(data[mask]))
        else:
            profile.append(np.nan)
    profile = np.array(profile)

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=config.dpi)

    # Panel 1: Image with apertures
    ax = axes[0]
    vmin, vmax = -3 * local_rms, peak_flux * 1.2
    im = ax.imshow(cutout, origin="lower", cmap="gray_r", vmin=vmin, vmax=vmax)

    # Adjust circle centers relative to cutout
    cx_cut = cx - x0
    cy_cut = cy - y0

    # Aperture circles
    aperture = Circle(
        (cx_cut, cy_cut),
        aperture_radius_pix,
        fill=False,
        edgecolor="lime",
        linewidth=2,
        label=f"Aperture (r={aperture_radius_pix:.1f} pix)",
    )
    annulus_inner = Circle(
        (cx_cut, cy_cut),
        annulus_inner_pix,
        fill=False,
        edgecolor="red",
        linewidth=1.5,
        linestyle="--",
        label=f"Annulus ({annulus_inner_pix:.1f}-{annulus_outer_pix:.1f} pix)",
    )
    annulus_outer = Circle(
        (cx_cut, cy_cut),
        annulus_outer_pix,
        fill=False,
        edgecolor="red",
        linewidth=1.5,
        linestyle="--",
    )
    ax.add_patch(aperture)
    ax.add_patch(annulus_inner)
    ax.add_patch(annulus_outer)

    ax.plot(cx_cut, cy_cut, "x", color="cyan", markersize=8, markeredgewidth=2)
    ax.set_xlabel("X (pixels)", fontsize=config.effective_label_size)
    ax.set_ylabel("Y (pixels)", fontsize=config.effective_label_size)
    ax.legend(fontsize=config.effective_tick_size - 1, loc="upper right")

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Flux (Jy/beam)", fontsize=config.effective_label_size)

    # Panel 2: Radial profile
    ax = axes[1]
    ax.plot(radii, profile * 1000, "o-", color="black", markersize=4, label="Radial Profile")
    ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
    ax.axhline(
        local_rms * 1000, color="red", linestyle=":", label=f"RMS = {local_rms * 1000:.1f} mJy"
    )
    ax.axhline(-local_rms * 1000, color="red", linestyle=":", alpha=0.5)
    ax.axvline(aperture_radius_pix, color="lime", linestyle="--", alpha=0.7, label="Aperture")
    ax.axvline(annulus_inner_pix, color="red", linestyle="--", alpha=0.5, label="Annulus")
    ax.axvline(annulus_outer_pix, color="red", linestyle="--", alpha=0.5)

    ax.set_xlabel("Radius (pixels)", fontsize=config.effective_label_size)
    ax.set_ylabel("Flux (mJy/beam)", fontsize=config.effective_label_size)
    ax.legend(fontsize=config.effective_tick_size - 1, loc="best")
    if config.grid:
        ax.grid(True, alpha=0.3)

    # Title
    if title:
        fig.suptitle(title, fontsize=config.effective_title_size, fontweight="bold")
    elif source_name:
        snr = peak_flux / local_rms
        fig.suptitle(
            f"{source_name} Aperture Photometry\nPeak: {peak_flux:.3f} Jy/beam, SNR: {snr:.1f}",
            fontsize=config.effective_title_size,
            fontweight="bold",
        )

    fig.tight_layout()

    if output:
        fig.savefig(output, dpi=config.dpi, bbox_inches="tight")
        logger.info(f"Saved aperture photometry plot: {output}")
        plt.close(fig)

    return fig


def plot_snr_map(
    data: NDArray,
    rms: float,
    wcs: WCS | None = None,
    output: str | Path | None = None,
    config: FigureConfig | None = None,
    contour_levels: list[float] | None = None,
    title: str | None = None,
) -> Figure:
    """Plot SNR map with contours.

    Parameters
    ----------
    data :
        2D image array in Jy/beam
    rms :
        RMS noise level in Jy/beam
    wcs :
        WCS object for coordinate axes (optional)
    output :
        Output file path
    config :
        Figure configuration
    contour_levels :
        SNR contour levels (default: [3, 5, 10, 20, 50])
    title :
        Plot title

    Returns
    -------
        matplotlib Figure object

    """
    _setup_matplotlib()
    import matplotlib.pyplot as plt
    from matplotlib.colors import SymLogNorm

    if config is None:
        config = FigureConfig(style=PlotStyle.QUICKLOOK)

    if contour_levels is None:
        contour_levels = [3, 5, 10, 20, 50]

    # Compute SNR
    snr_map = data / rms

    # Create figure with optional WCS projection
    if wcs is not None:
        fig = plt.figure(figsize=config.figsize, dpi=config.dpi)
        ax = fig.add_subplot(111, projection=wcs)
        xlabel, ylabel = "RA", "Dec"
    else:
        fig, ax = plt.subplots(figsize=config.figsize, dpi=config.dpi)
        xlabel, ylabel = "X (pixels)", "Y (pixels)"

    # SNR map with symmetric log normalization
    norm = SymLogNorm(linthresh=3, vmin=-10, vmax=np.nanpercentile(snr_map, 99))
    im = ax.imshow(snr_map, origin="lower", cmap="RdBu_r", norm=norm)

    # Contours
    contour_data = snr_map * rms  # Convert back to Jy for contour labels
    levels = [level * rms for level in contour_levels]
    cs = ax.contour(contour_data, levels=levels, colors="black", linewidths=1, alpha=0.6)
    ax.clabel(
        cs, inline=True, fontsize=config.effective_tick_size - 2, fmt=lambda x: f"{x / rms:.0f}σ"
    )

    ax.set_xlabel(xlabel, fontsize=config.effective_label_size)
    ax.set_ylabel(ylabel, fontsize=config.effective_label_size)

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("SNR", fontsize=config.effective_label_size)

    if title:
        ax.set_title(title, fontsize=config.effective_title_size)
    else:
        ax.set_title(
            f"Signal-to-Noise Ratio Map (RMS = {rms * 1000:.1f} mJy)",
            fontsize=config.effective_title_size,
        )

    if config.grid:
        ax.grid(True, alpha=0.3)

    fig.tight_layout()

    if output:
        fig.savefig(output, dpi=config.dpi, bbox_inches="tight")
        logger.info(f"Saved SNR map: {output}")
        plt.close(fig)

    return fig


def plot_catalog_comparison(
    data: NDArray,
    wcs: WCS,
    rms: float,
    catalog_sources: list[dict],
    output: str | Path | None = None,
    config: FigureConfig | None = None,
    catalog_name: str = "Catalog",
    title: str | None = None,
) -> Figure:
    """Plot catalog cross-match comparison.

    Creates a 2-panel figure:
    - Left: Measured vs catalog flux scatter plot
    - Right: Image with catalog source positions overlaid

    Parameters
    ----------
    data :
        2D image array in Jy/beam
    wcs :
        WCS object for coordinate transformation
    rms :
        RMS noise level in Jy/beam
    catalog_sources :
        List of dicts with keys:
        - 'ra': RA in degrees
        - 'dec': Dec in degrees
        - 'catalog_flux': Catalog flux in mJy
        - 'measured_flux': Measured flux in mJy
        - 'snr': Signal-to-noise ratio
        - 'status': Match status ('✓ Match', 'Non-det', 'Bright', etc.)
        - 'pix_x': Pixel X coordinate
        - 'pix_y': Pixel Y coordinate
    output :
        Output file path
    config :
        Figure configuration
    catalog_name :
        Name of reference catalog
    title :
        Custom title

    Returns
    -------
        matplotlib Figure object

    """
    _setup_matplotlib()
    import matplotlib.pyplot as plt

    if config is None:
        config = FigureConfig(style=PlotStyle.QUICKLOOK)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=config.dpi)

    # Panel 1: Scatter plot
    ax = axes[0]

    cat_flux = [s["catalog_flux"] for s in catalog_sources]
    meas_flux = [s["measured_flux"] for s in catalog_sources]
    colors = []
    for s in catalog_sources:
        if "✓" in s["status"]:
            colors.append("green")
        elif s["status"] == "Non-det":
            colors.append("red")
        elif s["status"] == "Bright":
            colors.append("orange")
        else:
            colors.append("gray")

    ax.scatter(cat_flux, meas_flux, c=colors, s=80, edgecolors="black", alpha=0.7)

    # Reference lines
    flux_min, flux_max = min(cat_flux + meas_flux), max(cat_flux + meas_flux)
    ax.plot([flux_min, flux_max], [flux_min, flux_max], "k--", lw=1, label="1:1")
    ax.plot([flux_min, flux_max], [2 * flux_min, 2 * flux_max], "k:", lw=1, alpha=0.5, label="2:1")
    ax.plot(
        [flux_min, flux_max], [0.5 * flux_min, 0.5 * flux_max], "k:", lw=1, alpha=0.5, label="1:2"
    )
    ax.axhline(5 * rms * 1000, color="red", ls="--", lw=1, label=f"5σ = {5 * rms * 1000:.0f} mJy")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(f"{catalog_name} Flux (mJy)", fontsize=config.effective_label_size)
    ax.set_ylabel("Measured Flux (mJy)", fontsize=config.effective_label_size)
    ax.legend(fontsize=config.effective_tick_size - 1)
    if config.grid:
        ax.grid(True, alpha=0.3)

    # Panel 2: Image with source positions
    ax = axes[1]
    vmin, vmax = -3 * rms, 10 * rms
    ax.imshow(data, origin="lower", cmap="gray_r", vmin=vmin, vmax=vmax)

    for s in catalog_sources:
        if "✓" in s["status"]:
            c = "lime"
        elif s["status"] == "Non-det":
            c = "red"
        elif s["status"] == "Bright":
            c = "orange"
        else:
            c = "gray"
        ax.scatter(s["pix_x"], s["pix_y"], s=100, facecolors="none", edgecolors=c, lw=2)

    ax.set_xlabel("X (pixels)", fontsize=config.effective_label_size)
    ax.set_ylabel("Y (pixels)", fontsize=config.effective_label_size)
    ax.set_title(
        f"{catalog_name} Sources\nGreen=match, Orange=variable, Red=non-detect",
        fontsize=config.effective_label_size,
    )

    # Overall title
    matches = [s for s in catalog_sources if "✓" in s["status"]]
    detected = [s for s in catalog_sources if s["snr"] > 3]
    if title:
        fig.suptitle(title, fontsize=config.effective_title_size, fontweight="bold")
    else:
        median_ratio = np.nan
        ratios = [s["measured_flux"] / s["catalog_flux"] for s in detected if s["catalog_flux"] > 0]
        if ratios:
            median_ratio = np.median(ratios)
        fig.suptitle(
            f"{catalog_name} Catalog Comparison\n"
            f"{len(matches)}/{len(catalog_sources)} sources match within factor of 2 "
            f"(median ratio: {median_ratio:.2f})",
            fontsize=config.effective_title_size,
            fontweight="bold",
        )

    fig.tight_layout()

    if output:
        fig.savefig(output, dpi=config.dpi, bbox_inches="tight")
        logger.info(f"Saved catalog comparison: {output}")
        plt.close(fig)

    return fig


def plot_field_sources(
    data: NDArray,
    wcs: WCS | None,
    rms: float,
    sources: list[dict],
    output: str | Path | None = None,
    config: FigureConfig | None = None,
    title: str | None = None,
    flux_threshold_sigma: float = 5.0,
) -> Figure:
    """Plot field source detections.

    Creates a 2-panel figure:
    - Left: Histogram of source flux distribution
    - Right: Image with detected sources overlaid

    Parameters
    ----------
    data :
        2D image array in Jy/beam
    wcs :
        WCS object (optional, for coordinate axes)
    rms :
        RMS noise level in Jy/beam
    sources :
        List of dicts with keys:
        - 'pix_x': Pixel X coordinate
        - 'pix_y': Pixel Y coordinate
        - 'peak_flux': Peak flux in Jy/beam
        - 'snr': Signal-to-noise ratio
    output :
        Output file path
    config :
        Figure configuration
    title :
        Custom title
    flux_threshold_sigma :
        Minimum SNR for detection

    Returns
    -------
        matplotlib Figure object

    """
    _setup_matplotlib()
    import matplotlib.pyplot as plt
    from matplotlib.colors import SymLogNorm

    if config is None:
        config = FigureConfig(style=PlotStyle.QUICKLOOK)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=config.dpi)

    # Panel 1: Flux histogram
    ax = axes[0]
    fluxes = [s["peak_flux"] * 1000 for s in sources]  # mJy
    ax.hist(fluxes, bins=50, color="steelblue", edgecolor="black", alpha=0.7)
    ax.axvline(
        flux_threshold_sigma * rms * 1000,
        color="red",
        linestyle="--",
        lw=2,
        label=f"{flux_threshold_sigma}σ threshold",
    )
    ax.set_xlabel("Peak Flux (mJy/beam)", fontsize=config.effective_label_size)
    ax.set_ylabel("Count", fontsize=config.effective_label_size)
    ax.set_yscale("log")
    ax.legend(fontsize=config.effective_tick_size)
    if config.grid:
        ax.grid(True, alpha=0.3)

    # Panel 2: Image with sources
    if wcs is not None:
        ax = fig.add_subplot(122, projection=wcs)
        xlabel, ylabel = "RA", "Dec"
    else:
        ax = axes[1]
        xlabel, ylabel = "X (pixels)", "Y (pixels)"

    norm = SymLogNorm(linthresh=rms, vmin=-3 * rms, vmax=10 * rms)
    ax.imshow(data, origin="lower", cmap="gray_r", norm=norm)

    # Overlay sources
    for s in sources:
        ax.plot(
            s["pix_x"],
            s["pix_y"],
            "o",
            color="lime",
            markersize=4,
            markeredgewidth=0.5,
            markeredgecolor="black",
        )

    ax.set_xlabel(xlabel, fontsize=config.effective_label_size)
    ax.set_ylabel(ylabel, fontsize=config.effective_label_size)
    ax.set_title(
        f"{len(sources)} Sources > {flux_threshold_sigma}σ", fontsize=config.effective_label_size
    )

    # Overall title
    if title:
        fig.suptitle(title, fontsize=config.effective_title_size, fontweight="bold")
    else:
        fig.suptitle(
            f"Field Source Detection (RMS = {rms * 1000:.1f} mJy)",
            fontsize=config.effective_title_size,
            fontweight="bold",
        )

    fig.tight_layout()

    if output:
        fig.savefig(output, dpi=config.dpi, bbox_inches="tight")
        logger.info(f"Saved field sources plot: {output}")
        plt.close(fig)

    return fig


def plot_photometry_summary(
    source_name: str,
    peak_flux: float,
    peak_error: float,
    local_snr: float,
    catalog_flux: float | None = None,
    catalog_name: str = "Catalog",
    output: str | Path | None = None,
    config: FigureConfig | None = None,
    additional_metrics: dict | None = None,
) -> Figure:
    """Plot photometry summary panel.

    Creates a multi-panel figure showing:
    - Flux comparison bar chart
    - Key metrics table

    Parameters
    ----------
    source_name :
        Source name
    peak_flux :
        Measured peak flux in Jy/beam
    peak_error :
        Flux error in Jy/beam
    local_snr :
        Signal-to-noise ratio
    catalog_flux :
        Reference catalog flux in Jy (optional)
    catalog_name :
        Name of reference catalog
    output :
        Output file path
    config :
        Figure configuration
    additional_metrics :
        Dict of additional metrics to display

    Returns
    -------
        matplotlib Figure object

    """
    _setup_matplotlib()
    import matplotlib.pyplot as plt

    if config is None:
        config = FigureConfig(style=PlotStyle.QUICKLOOK)

    if catalog_flux is not None:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=config.dpi)
    else:
        fig, ax = plt.subplots(1, 1, figsize=(8, 5), dpi=config.dpi)
        axes = [ax]

    # Panel 1: Flux comparison
    ax = axes[0]
    labels = ["Measured"]
    values = [peak_flux]
    errors = [peak_error]
    colors = ["steelblue"]

    if catalog_flux is not None:
        labels.append(catalog_name)
        values.append(catalog_flux)
        errors.append(0)
        colors.append("coral")

    x_pos = np.arange(len(labels))
    bars = ax.bar(x_pos, values, yerr=errors, color=colors, edgecolor="black", alpha=0.7, capsize=5)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, fontsize=config.effective_label_size)
    ax.set_ylabel("Flux (Jy/beam)", fontsize=config.effective_label_size)
    ax.set_title("Flux Comparison", fontsize=config.effective_label_size)

    # Add value labels on bars
    for bar, val, err in zip(bars, values, errors):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + err,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=config.effective_tick_size,
        )

    if config.grid:
        ax.grid(True, alpha=0.3, axis="y")

    # Panel 2: Metrics table (if catalog flux provided)
    if catalog_flux is not None:
        ax = axes[1]
        ax.axis("off")

        ratio = peak_flux / catalog_flux if catalog_flux > 0 else np.nan
        metrics = [
            ["Metric", "Value"],
            ["Peak Flux", f"{peak_flux:.3f} ± {peak_error:.3f} Jy/beam"],
            [f"{catalog_name} Flux", f"{catalog_flux:.3f} Jy"],
            ["Flux Ratio", f"{ratio:.2f}"],
            ["Local SNR", f"{local_snr:.1f}"],
        ]

        if additional_metrics:
            for key, val in additional_metrics.items():
                metrics.append([key, str(val)])

        table = ax.table(
            cellText=metrics,
            cellLoc="left",
            loc="center",
            colWidths=[0.4, 0.6],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(config.effective_tick_size)
        table.scale(1, 2)

        # Style header row
        for i in range(2):
            table[(0, i)].set_facecolor("#4CAF50")
            table[(0, i)].set_text_props(weight="bold", color="white")

    fig.suptitle(
        f"{source_name} Photometry Summary", fontsize=config.effective_title_size, fontweight="bold"
    )
    fig.tight_layout()

    if output:
        fig.savefig(output, dpi=config.dpi, bbox_inches="tight")
        logger.info(f"Saved photometry summary: {output}")
        plt.close(fig)

    return fig
