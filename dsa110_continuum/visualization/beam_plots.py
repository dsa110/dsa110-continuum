"""
PSF and beam analysis plotting utilities.

Provides functions for:
- PSF radial profile plots (major/minor axis cuts)
- PSF ellipticity and position angle visualization
- Dirty beam vs clean beam comparison
- Sidelobe analysis
- Primary beam pattern visualization

Essential for understanding resolution, image fidelity, and artifacts.
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


def fit_2d_gaussian(
    image: NDArray,
    center: tuple[int, int] | None = None,
) -> dict:
    """Fit a 2D Gaussian to a PSF image.

    Parameters
    ----------
    image :
        2D PSF image array
    center :
        Optional (y, x) center pixel (default: image center)

    Returns
    -------
    Dictionary with fitted parameters
        amplitude, x0, y0, sigma_x, sigma_y,
    Dictionary with fitted parameters
        amplitude, x0, y0, sigma_x, sigma_y,
        theta (rotation angle in degrees), fwhm_major, fwhm_minor, pa

    """
    from scipy.optimize import curve_fit

    img = np.asarray(image)
    ny, nx = img.shape

    if center is None:
        # Find peak
        y0, x0 = np.unravel_index(np.argmax(img), img.shape)
    else:
        y0, x0 = center

    # Create coordinate grids
    y, x = np.mgrid[:ny, :nx]

    def gaussian_2d(coords, amplitude, xo, yo, sigma_x, sigma_y, theta, offset):
        x, y = coords
        xo = float(xo)
        yo = float(yo)
        a = np.cos(theta) ** 2 / (2 * sigma_x**2) + np.sin(theta) ** 2 / (2 * sigma_y**2)
        b = -np.sin(2 * theta) / (4 * sigma_x**2) + np.sin(2 * theta) / (4 * sigma_y**2)
        c = np.sin(theta) ** 2 / (2 * sigma_x**2) + np.cos(theta) ** 2 / (2 * sigma_y**2)
        g = offset + amplitude * np.exp(
            -(a * (x - xo) ** 2 + 2 * b * (x - xo) * (y - yo) + c * (y - yo) ** 2)
        )
        return g.ravel()

    # Initial guess
    amplitude = img.max()
    sigma_est = 5.0
    p0 = [amplitude, x0, y0, sigma_est, sigma_est, 0.0, 0.0]

    try:
        import warnings
        from scipy.optimize import OptimizeWarning

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=OptimizeWarning)
            popt, _ = curve_fit(
                gaussian_2d,
                (x, y),
                img.ravel(),
                p0=p0,
                maxfev=5000,
            )

        amplitude, x0_fit, y0_fit, sigma_x, sigma_y, theta, offset = popt

        # Ensure major axis is larger
        if sigma_x < sigma_y:
            sigma_x, sigma_y = sigma_y, sigma_x
            theta += np.pi / 2

        # Convert to FWHM
        fwhm_factor = 2.0 * np.sqrt(2.0 * np.log(2.0))
        fwhm_major = abs(sigma_x) * fwhm_factor
        fwhm_minor = abs(sigma_y) * fwhm_factor

        # Position angle (North through East)
        pa = np.degrees(theta) % 180

        return {
            "amplitude": amplitude,
            "x0": x0_fit,
            "y0": y0_fit,
            "sigma_x": abs(sigma_x),
            "sigma_y": abs(sigma_y),
            "theta": theta,
            "offset": offset,
            "fwhm_major": fwhm_major,
            "fwhm_minor": fwhm_minor,
            "pa": pa,
            "ellipticity": 1 - fwhm_minor / fwhm_major if fwhm_major > 0 else 0,
        }

    except Exception as e:
        logger.warning(f"Gaussian fit failed: {e}")
        return {
            "amplitude": img.max(),
            "x0": x0,
            "y0": y0,
            "sigma_x": np.nan,
            "sigma_y": np.nan,
            "theta": np.nan,
            "offset": 0,
            "fwhm_major": np.nan,
            "fwhm_minor": np.nan,
            "pa": np.nan,
            "ellipticity": np.nan,
        }


def plot_psf_radial_profile(
    psf_data: NDArray,
    output: str | Path | None = None,
    config: FigureConfig | None = None,
    title: str = "PSF Radial Profile",
    pixel_scale_arcsec: float = 1.0,
    show_fit: bool = True,
    show_major_minor: bool = True,
    max_radius_pixels: int | None = None,
) -> Figure:
    """Plot radial profile of PSF with optional Gaussian fit.

    Parameters
    ----------
    psf_data :
        2D PSF image array (should be centered)
    output :
        Output file path
    config :
        Figure configuration
    title :
        Plot title
    pixel_scale_arcsec :
        Pixel scale in arcseconds
    show_fit :
        Overlay Gaussian fit
    show_major_minor :
        Show major/minor axis cuts separately
    max_radius_pixels :
        Maximum radius to plot (None = auto)

    Returns
    -------
        matplotlib Figure object

    """
    _setup_matplotlib()
    import matplotlib.pyplot as plt

    if config is None:
        config = FigureConfig(style=PlotStyle.QUICKLOOK)

    psf = np.asarray(psf_data)
    ny, nx = psf.shape

    # Find center (peak)
    cy, cx = np.unravel_index(np.argmax(psf), psf.shape)

    # Fit Gaussian
    fit_params = fit_2d_gaussian(psf, center=(cy, cx))

    # Create radial coordinates
    y, x = np.mgrid[:ny, :nx]
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)

    if max_radius_pixels is None:
        max_radius_pixels = min(cx, cy, nx - cx, ny - cy)

    fig, ax = plt.subplots(figsize=config.figsize, dpi=config.dpi)

    if show_major_minor and not np.isnan(fit_params["theta"]):
        # Extract cuts along major and minor axes
        theta = fit_params["theta"]

        # Major axis direction
        major_x = np.arange(-max_radius_pixels, max_radius_pixels + 1)
        major_y = np.zeros_like(major_x)

        # Rotate to major axis
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        major_xi = cx + (major_x * cos_t - major_y * sin_t)
        major_yi = cy + (major_x * sin_t + major_y * cos_t)

        # Interpolate
        from scipy.ndimage import map_coordinates

        major_profile = map_coordinates(psf, [major_yi, major_xi], order=1)
        minor_xi = cx + (-major_x * sin_t)
        minor_yi = cy + (major_x * cos_t)
        minor_profile = map_coordinates(psf, [minor_yi, minor_xi], order=1)

        r_arcsec = major_x * pixel_scale_arcsec

        ax.plot(r_arcsec, major_profile / psf.max(), "b-", linewidth=2, label="Major axis")
        ax.plot(r_arcsec, minor_profile / psf.max(), "r-", linewidth=2, label="Minor axis")

    else:
        # Azimuthally averaged profile
        r_bins = np.arange(0, max_radius_pixels + 1, 1)
        r_centers = (r_bins[:-1] + r_bins[1:]) / 2
        profile = np.zeros(len(r_centers))

        for i in range(len(r_centers)):
            mask = (r >= r_bins[i]) & (r < r_bins[i + 1])
            if np.any(mask):
                profile[i] = np.mean(psf[mask])

        r_arcsec = r_centers * pixel_scale_arcsec
        ax.plot(r_arcsec, profile / psf.max(), "k-", linewidth=2, label="Azimuthal avg")

    # Gaussian fit overlay
    if show_fit and not np.isnan(fit_params["sigma_x"]):
        r_fit = np.linspace(-max_radius_pixels, max_radius_pixels, 200) * pixel_scale_arcsec
        sigma_major = fit_params["sigma_x"] * pixel_scale_arcsec
        sigma_minor = fit_params["sigma_y"] * pixel_scale_arcsec

        gauss_major = np.exp(-0.5 * (r_fit / sigma_major) ** 2)
        gauss_minor = np.exp(-0.5 * (r_fit / sigma_minor) ** 2)

        ax.plot(
            r_fit,
            gauss_major,
            "b--",
            alpha=0.5,
            label=f'Fit major (FWHM={fit_params["fwhm_major"] * pixel_scale_arcsec:.2f}")',
        )
        ax.plot(
            r_fit,
            gauss_minor,
            "r--",
            alpha=0.5,
            label=f'Fit minor (FWHM={fit_params["fwhm_minor"] * pixel_scale_arcsec:.2f}")',
        )

    # FWHM level
    ax.axhline(0.5, color="gray", linestyle=":", alpha=0.7, label="FWHM level")

    ax.set_xlabel("Radius (arcsec)", fontsize=config.effective_label_size)
    ax.set_ylabel("Normalized Response", fontsize=config.effective_label_size)
    ax.set_title(title, fontsize=config.effective_title_size)
    ax.legend(fontsize=config.effective_tick_size)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.1, 1.1)

    # Add beam parameters text
    if not np.isnan(fit_params["fwhm_major"]):
        beam_text = (
            f'FWHM major: {fit_params["fwhm_major"] * pixel_scale_arcsec:.2f}"\n'
            f'FWHM minor: {fit_params["fwhm_minor"] * pixel_scale_arcsec:.2f}"\n'
            f"PA: {fit_params['pa']:.1f}°\n"
            f"Ellipticity: {fit_params['ellipticity']:.3f}"
        )
        ax.text(
            0.98,
            0.98,
            beam_text,
            transform=ax.transAxes,
            verticalalignment="top",
            horizontalalignment="right",
            fontsize=config.effective_tick_size,
            bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"),
        )

    fig.tight_layout()

    if output:
        fig.savefig(output, dpi=config.dpi, bbox_inches="tight")
        logger.info(f"Saved PSF radial profile: {output}")
        plt.close(fig)

    return fig


def plot_psf_2d(
    psf_data: NDArray,
    output: str | Path | None = None,
    config: FigureConfig | None = None,
    title: str = "PSF (Dirty Beam)",
    pixel_scale_arcsec: float = 1.0,
    show_contours: bool = True,
    show_ellipse: bool = True,
    log_scale: bool = False,
    cutout_radius_pixels: int | None = None,
) -> Figure:
    """Plot 2D PSF image with contours and beam ellipse.

    Parameters
    ----------
    psf_data :
        2D PSF image array
    output :
        Output file path
    config :
        Figure configuration
    title :
        Plot title
    pixel_scale_arcsec :
        Pixel scale in arcseconds
    show_contours :
        Overlay contours
    show_ellipse :
        Overlay fitted beam ellipse
    log_scale :
        Use log scale for colormap
    cutout_radius_pixels :
        Radius of cutout to show (None = auto)

    Returns
    -------
        matplotlib Figure object

    """
    _setup_matplotlib()
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    if config is None:
        config = FigureConfig(style=PlotStyle.QUICKLOOK)

    psf = np.asarray(psf_data)
    ny, nx = psf.shape

    # Find center and fit
    cy, cx = np.unravel_index(np.argmax(psf), psf.shape)
    fit_params = fit_2d_gaussian(psf, center=(cy, cx))

    # Create cutout if requested
    if cutout_radius_pixels is not None:
        r = cutout_radius_pixels
        y1, y2 = max(0, cy - r), min(ny, cy + r)
        x1, x2 = max(0, cx - r), min(nx, cx + r)
        psf_plot = psf[y1:y2, x1:x2]
        cx_plot, cy_plot = cx - x1, cy - y1
    else:
        psf_plot = psf
        cx_plot, cy_plot = cx, cy

    # Create coordinate extent in arcsec
    npy, npx = psf_plot.shape
    extent = [
        -cx_plot * pixel_scale_arcsec,
        (npx - cx_plot) * pixel_scale_arcsec,
        -cy_plot * pixel_scale_arcsec,
        (npy - cy_plot) * pixel_scale_arcsec,
    ]

    fig, ax = plt.subplots(figsize=config.figsize, dpi=config.dpi)

    # Normalize
    psf_norm = psf_plot / psf_plot.max()

    if log_scale:
        from matplotlib.colors import LogNorm

        vmin = max(1e-4, np.abs(psf_norm[psf_norm > 0]).min())
        im = ax.imshow(
            np.abs(psf_norm),
            origin="lower",
            cmap=config.cmap,
            extent=extent,
            norm=LogNorm(vmin=vmin, vmax=1.0),
        )
    else:
        im = ax.imshow(
            psf_norm,
            origin="lower",
            cmap=config.cmap,
            extent=extent,
            vmin=-0.1,
            vmax=1.0,
        )

    # Contours
    if show_contours:
        levels = [0.1, 0.3, 0.5, 0.7, 0.9]
        ax.contour(
            psf_norm,
            levels=levels,
            colors="white",
            linewidths=0.5,
            alpha=0.7,
            extent=extent,
        )

    # Beam ellipse
    if show_ellipse and not np.isnan(fit_params["fwhm_major"]):
        ellipse = Ellipse(
            (0, 0),
            width=fit_params["fwhm_minor"] * pixel_scale_arcsec,
            height=fit_params["fwhm_major"] * pixel_scale_arcsec,
            angle=fit_params["pa"],
            facecolor="none",
            edgecolor="red",
            linewidth=2,
            linestyle="--",
        )
        ax.add_patch(ellipse)

    ax.set_xlabel("Δ RA (arcsec)", fontsize=config.effective_label_size)
    ax.set_ylabel("Δ Dec (arcsec)", fontsize=config.effective_label_size)
    ax.set_title(title, fontsize=config.effective_title_size)

    # Colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im, cax=cax, label="Normalized Response")

    # Crosshair at center
    ax.axhline(0, color="white", linestyle="-", alpha=0.3, linewidth=0.5)
    ax.axvline(0, color="white", linestyle="-", alpha=0.3, linewidth=0.5)

    ax.set_aspect("equal")
    fig.tight_layout()

    if output:
        fig.savefig(output, dpi=config.dpi, bbox_inches="tight")
        logger.info(f"Saved PSF 2D plot: {output}")
        plt.close(fig)

    return fig


def plot_beam_comparison(
    dirty_beam: NDArray,
    clean_beam: NDArray,
    output: str | Path | None = None,
    config: FigureConfig | None = None,
    title: str = "Dirty Beam vs Clean Beam",
    pixel_scale_arcsec: float = 1.0,
) -> Figure:
    """Compare dirty beam (PSF) with idealized clean beam.

    Parameters
    ----------
    dirty_beam :
        2D dirty beam array
    clean_beam :
        2D clean beam array (Gaussian restoring beam)
    output :
        Output file path
    config :
        Figure configuration
    title :
        Plot title
    pixel_scale_arcsec :
        Pixel scale in arcseconds

    Returns
    -------
        matplotlib Figure object

    """
    _setup_matplotlib()
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    if config is None:
        config = FigureConfig(style=PlotStyle.QUICKLOOK)

    dirty = np.asarray(dirty_beam)
    clean = np.asarray(clean_beam)

    # Ensure same shape
    if dirty.shape != clean.shape:
        logger.warning(f"Beam shapes differ: {dirty.shape} vs {clean.shape}")

    # Fit both
    dirty_fit = fit_2d_gaussian(dirty)
    clean_fit = fit_2d_gaussian(clean)

    fig, axes = plt.subplots(1, 3, figsize=(config.figsize[0] * 3 / 2, config.figsize[1]))

    # Dirty beam
    im1 = axes[0].imshow(dirty / dirty.max(), origin="lower", cmap=config.cmap, vmin=-0.1, vmax=1.0)
    axes[0].set_title("Dirty Beam (PSF)", fontsize=config.effective_title_size)
    div1 = make_axes_locatable(axes[0])
    cax1 = div1.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im1, cax=cax1)

    # Clean beam
    im2 = axes[1].imshow(clean / clean.max(), origin="lower", cmap=config.cmap, vmin=-0.1, vmax=1.0)
    axes[1].set_title("Clean Beam (Restoring)", fontsize=config.effective_title_size)
    div2 = make_axes_locatable(axes[1])
    cax2 = div2.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im2, cax=cax2)

    # Difference (sidelobes)
    diff = dirty / dirty.max() - clean / clean.max()
    vmax = np.max(np.abs(diff))
    im3 = axes[2].imshow(diff, origin="lower", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    axes[2].set_title("Difference (Sidelobes)", fontsize=config.effective_title_size)
    div3 = make_axes_locatable(axes[2])
    cax3 = div3.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im3, cax=cax3)

    for ax in axes:
        ax.set_xlabel("X (pixels)")
        ax.set_ylabel("Y (pixels)")

    # Summary text
    if not np.isnan(dirty_fit["fwhm_major"]) and not np.isnan(clean_fit["fwhm_major"]):
        summary = (
            f'Dirty: {dirty_fit["fwhm_major"] * pixel_scale_arcsec:.2f}" × {dirty_fit["fwhm_minor"] * pixel_scale_arcsec:.2f}"\n'
            f'Clean: {clean_fit["fwhm_major"] * pixel_scale_arcsec:.2f}" × {clean_fit["fwhm_minor"] * pixel_scale_arcsec:.2f}"\n'
            f"Max sidelobe: {np.max(np.abs(diff)):.3f}"
        )
        fig.text(
            0.5,
            0.02,
            summary,
            transform=fig.transFigure,
            ha="center",
            fontsize=config.effective_tick_size,
            bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"),
        )

    fig.suptitle(title, fontsize=config.effective_title_size + 2)
    fig.tight_layout()

    if output:
        fig.savefig(output, dpi=config.dpi, bbox_inches="tight")
        logger.info(f"Saved beam comparison: {output}")
        plt.close(fig)

    return fig


def plot_sidelobe_analysis(
    psf_data: NDArray,
    output: str | Path | None = None,
    config: FigureConfig | None = None,
    title: str = "PSF Sidelobe Analysis",
    pixel_scale_arcsec: float = 1.0,
    n_radial_bins: int = 50,
) -> Figure:
    """Analyze PSF sidelobes with radial statistics.

    Parameters
    ----------
    psf_data :
        2D PSF image array
    output :
        Output file path
    config :
        Figure configuration
    title :
        Plot title
    pixel_scale_arcsec :
        Pixel scale in arcseconds
    n_radial_bins :
        Number of radial bins

    Returns
    -------
        matplotlib Figure object

    """
    _setup_matplotlib()
    import matplotlib.pyplot as plt

    if config is None:
        config = FigureConfig(style=PlotStyle.QUICKLOOK)

    psf = np.asarray(psf_data)
    psf_norm = psf / psf.max()

    ny, nx = psf.shape
    cy, cx = np.unravel_index(np.argmax(psf), psf.shape)

    # Radial coordinates
    y, x = np.mgrid[:ny, :nx]
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    max_r = min(cx, cy, nx - cx, ny - cy)

    # Bin by radius
    r_edges = np.linspace(0, max_r, n_radial_bins + 1)
    r_centers = (r_edges[:-1] + r_edges[1:]) / 2

    means = []
    maxs = []
    mins = []
    stds = []

    for i in range(len(r_centers)):
        mask = (r >= r_edges[i]) & (r < r_edges[i + 1])
        if np.any(mask):
            vals = psf_norm[mask]
            means.append(np.mean(vals))
            maxs.append(np.max(vals))
            mins.append(np.min(vals))
            stds.append(np.std(vals))
        else:
            means.append(np.nan)
            maxs.append(np.nan)
            mins.append(np.nan)
            stds.append(np.nan)

    means = np.array(means)
    maxs = np.array(maxs)
    mins = np.array(mins)
    stds = np.array(stds)

    r_arcsec = r_centers * pixel_scale_arcsec

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(config.figsize[0], config.figsize[1] * 2), sharex=True
    )

    # Top panel: mean with envelope
    ax1.fill_between(r_arcsec, mins, maxs, alpha=0.3, color="blue", label="Min-Max")
    ax1.fill_between(
        r_arcsec, means - stds, means + stds, alpha=0.5, color="orange", label="Mean ± σ"
    )
    ax1.plot(r_arcsec, means, "k-", linewidth=2, label="Mean")
    ax1.axhline(0, color="gray", linestyle="--", alpha=0.5)

    ax1.set_ylabel("Normalized Response", fontsize=config.effective_label_size)
    ax1.set_title(title, fontsize=config.effective_title_size)
    ax1.legend(fontsize=config.effective_tick_size, loc="upper right")
    ax1.grid(True, alpha=0.3)

    # Bottom panel: absolute sidelobe level
    sidelobe_level = np.abs(maxs)
    ax2.semilogy(r_arcsec, sidelobe_level, "b-", linewidth=2, label="|Max|")
    ax2.semilogy(r_arcsec, np.abs(mins), "r-", linewidth=2, label="|Min|")
    ax2.axhline(0.01, color="gray", linestyle="--", alpha=0.7, label="1% level")
    ax2.axhline(0.001, color="gray", linestyle=":", alpha=0.7, label="0.1% level")

    ax2.set_xlabel("Radius (arcsec)", fontsize=config.effective_label_size)
    ax2.set_ylabel("Sidelobe Level", fontsize=config.effective_label_size)
    ax2.legend(fontsize=config.effective_tick_size, loc="upper right")
    ax2.grid(True, alpha=0.3, which="both")

    fig.tight_layout()

    if output:
        fig.savefig(output, dpi=config.dpi, bbox_inches="tight")
        logger.info(f"Saved sidelobe analysis: {output}")
        plt.close(fig)

    return fig


def plot_primary_beam_pattern(
    freq_hz: float = 1.4e9,
    dish_diameter_m: float = 4.65,
    output: str | Path | None = None,
    config: FigureConfig | None = None,
    title: str = "Primary Beam Pattern",
    max_offset_deg: float = 5.0,
    n_points: int = 200,
) -> Figure:
    """Plot theoretical primary beam pattern (Airy disk).

    Parameters
    ----------
    freq_hz :
        Observing frequency in Hz
    dish_diameter_m :
        Dish diameter in meters
    output :
        Output file path
    config :
        Figure configuration
    title :
        Plot title
    max_offset_deg :
        Maximum angular offset to plot
    n_points :
        Number of points for smooth curve

    Returns
    -------
        matplotlib Figure object

    """
    _setup_matplotlib()
    import matplotlib.pyplot as plt
    from scipy.special import j1

    if config is None:
        config = FigureConfig(style=PlotStyle.QUICKLOOK)

    c_mps = 299792458.0
    wavelength_m = c_mps / freq_hz

    # FWHM calculation
    fwhm_rad = 0.514 * wavelength_m / dish_diameter_m
    fwhm_deg = np.degrees(fwhm_rad)

    offsets_deg = np.linspace(0, max_offset_deg, n_points)
    offsets_rad = np.deg2rad(offsets_deg)

    # Airy pattern
    x = np.pi * dish_diameter_m * np.sin(offsets_rad) / wavelength_m

    with np.errstate(divide="ignore", invalid="ignore"):
        response = np.where(
            np.abs(x) < 1e-10,
            1.0,
            (2.0 * j1(x) / x) ** 2,
        )

    fig, ax = plt.subplots(figsize=config.figsize, dpi=config.dpi)

    ax.plot(offsets_deg, response, "b-", linewidth=2, label="Airy Pattern")
    ax.axhline(0.5, color="red", linestyle="--", alpha=0.7, label=f"FWHM = {fwhm_deg:.2f}°")
    ax.axvline(fwhm_deg, color="red", linestyle=":", alpha=0.5)

    ax.set_xlabel("Angular Offset (degrees)", fontsize=config.effective_label_size)
    ax.set_ylabel("Primary Beam Response", fontsize=config.effective_label_size)
    ax.set_title(
        f"{title}\n{dish_diameter_m}m dish @ {freq_hz / 1e9:.2f} GHz",
        fontsize=config.effective_title_size,
    )
    ax.legend(fontsize=config.effective_tick_size)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)
    ax.set_xlim(0, max_offset_deg)

    # Annotate parameters
    params_text = (
        f"Dish: {dish_diameter_m} m\n"
        f"Freq: {freq_hz / 1e9:.2f} GHz\n"
        f"λ: {wavelength_m * 100:.1f} cm\n"
        f"FWHM: {fwhm_deg:.3f}°"
    )
    ax.text(
        0.98,
        0.5,
        params_text,
        transform=ax.transAxes,
        verticalalignment="center",
        horizontalalignment="right",
        fontsize=config.effective_tick_size,
        bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"),
    )

    fig.tight_layout()

    if output:
        fig.savefig(output, dpi=config.dpi, bbox_inches="tight")
        logger.info(f"Saved primary beam pattern: {output}")
        plt.close(fig)

    return fig
