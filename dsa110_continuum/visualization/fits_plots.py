"""
FITS image plotting utilities.

Provides functions for:
- Full image display with WCS
- Source cutouts
- Quicklook PNG generation
- Mosaic overview plots

Adapted from:
- VAST/vastfast/plot.py (cutouts, WCS handling)
- radiopadre/fitsfile.py (normalization, colormaps)
- dsa110_contimg/imaging/export.py (existing PNG generation)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from astropy.coordinates import SkyCoord
    from matplotlib.figure import Figure
    from numpy.typing import NDArray

from dsa110_contimg.common.utils.fits_utils import get_2d_data_and_wcs as _get_2d_data_and_wcs
from dsa110_contimg.core.visualization.config import FigureConfig, PlotStyle

logger = logging.getLogger(__name__)


def _setup_matplotlib() -> None:
    """Configure matplotlib for headless operation."""
    import matplotlib

    matplotlib.use("Agg")


def _normalize_image(
    data: NDArray,
    vmin: float | None = None,
    vmax: float | None = None,
    stretch: str = "asinh",
    contrast: float = 0.2,
) -> tuple[NDArray, float, float]:
    """Normalize image data for display.

    Uses ZScale + asinh stretch by default (astronomy standard).

    Parameters
    ----------
    data :
        2D image array
    vmin :
        Manual minimum value
    vmax :
        Manual maximum value
    stretch :
        Stretch function ('linear', 'asinh', 'log', 'sqrt')
    contrast :
        ZScale contrast parameter

    Returns
    -------
        Tuple of (normalized_data, vmin, vmax)

    """
    from astropy.visualization import (
        AsinhStretch,
        ImageNormalize,
        LinearStretch,
        LogStretch,
        SqrtStretch,
        ZScaleInterval,
    )

    finite_mask = np.isfinite(data)
    if not np.any(finite_mask):
        return data, 0, 1

    finite_data = data[finite_mask]

    # Determine vmin/vmax using ZScale if not provided
    if vmin is None or vmax is None:
        interval = ZScaleInterval(contrast=contrast)
        auto_vmin, auto_vmax = interval.get_limits(finite_data)
        vmin = vmin if vmin is not None else auto_vmin
        vmax = vmax if vmax is not None else auto_vmax

    # Select stretch function
    stretch_map = {
        "linear": LinearStretch(),
        "asinh": AsinhStretch(),
        "log": LogStretch(),
        "sqrt": SqrtStretch(),
    }
    stretch_fn = stretch_map.get(stretch, AsinhStretch())

    # Create normalizer
    norm = ImageNormalize(vmin=vmin, vmax=vmax, stretch=stretch_fn)

    return norm, vmin, vmax


def plot_fits_image(
    fits_path: str | Path,
    output: str | Path | None = None,
    config: FigureConfig | None = None,
    title: str | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    stretch: str = "asinh",
    show_beam: bool = True,
    show_colorbar: bool = True,
) -> Figure:
    """Plot a FITS image with WCS axes.

    Parameters
    ----------
    fits_path :
        Path to FITS file
    output :
        Output file path (None for interactive display)
    config :
        Figure configuration
    title :
        Plot title (default: filename)
    vmin :
        Minimum display value
    vmax :
        Maximum display value
    stretch :
        Stretch function ('linear', 'asinh', 'log', 'sqrt')
    show_beam :
        Show beam ellipse if available
    show_colorbar :
        Show colorbar
    fits_path : Union[str, Path]
    output : Optional[Union[str, Path]]
         (Default value = None)
    config: Optional[FigureConfig] :
         (Default value = None)
    title: Optional[str] :
         (Default value = None)
    vmin: Optional[float] :
         (Default value = None)
    vmax: Optional[float] :
         (Default value = None)

    Returns
    -------
        matplotlib Figure object

    """
    _setup_matplotlib()
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse

    if config is None:
        config = FigureConfig(style=PlotStyle.QUICKLOOK)

    fits_path = Path(fits_path)
    data, wcs, header_info = _get_2d_data_and_wcs(fits_path)

    # Normalize
    norm, vmin_used, vmax_used = _normalize_image(data, vmin=vmin, vmax=vmax, stretch=stretch)

    # Create figure with WCS projection
    fig = plt.figure(figsize=config.figsize, dpi=config.dpi)
    ax = fig.add_subplot(111, projection=wcs)

    # Display image
    im = ax.imshow(
        data,
        origin="lower",
        cmap=config.cmap,
        norm=norm,
        interpolation="nearest",
    )

    # Axis labels
    ax.set_xlabel("Right Ascension (J2000)")
    ax.set_ylabel("Declination (J2000)")

    # Title
    if title is None:
        title = fits_path.name
    ax.set_title(title, fontsize=config.effective_title_size)

    # Colorbar
    if show_colorbar:
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(header_info["bunit"], fontsize=config.effective_label_size)

    # Beam ellipse
    if show_beam and header_info["bmaj"] and header_info["bmin"]:
        # Position in bottom-left corner (pixel coordinates)
        bmaj_pix = header_info["bmaj"] / abs(wcs.wcs.cdelt[0])
        bmin_pix = header_info["bmin"] / abs(wcs.wcs.cdelt[1])
        bpa = header_info["bpa"] or 0

        beam_x = 0.1 * data.shape[1]
        beam_y = 0.1 * data.shape[0]

        beam = Ellipse(
            (beam_x, beam_y),
            width=bmin_pix,
            height=bmaj_pix,
            angle=90 + bpa,
            facecolor="white",
            edgecolor="black",
            linewidth=1,
        )
        ax.add_patch(beam)

    if config.tight_layout:
        fig.tight_layout()

    # Save or return
    if output:
        fig.savefig(output, dpi=config.dpi, bbox_inches="tight")
        logger.info(f"Saved figure to {output}")
        plt.close(fig)

    return fig


def plot_cutout(
    fits_path: str | Path,
    ra: float | None = None,
    dec: float | None = None,
    coord: SkyCoord | None = None,
    radius_arcmin: float = 5.0,
    output: str | Path | None = None,
    config: FigureConfig | None = None,
    title: str | None = None,
    show_crosshair: bool = True,
    show_circle: bool = False,
    circle_radius_arcsec: float = 30.0,
    **kwargs: Any,
) -> Figure:
    """Create a cutout image centered on a position.

    Adapted from VAST/vastfast/plot.py plot_cutout().

    Parameters
    ----------
    fits_path :
        Path to FITS file
    ra :
        Right ascension in degrees (or use coord)
    dec :
        Declination in degrees (or use coord)
    coord :
        SkyCoord object (alternative to ra/dec)
    radius_arcmin :
        Cutout radius in arcminutes
    output :
        Output file path
    config :
        Figure configuration
    title :
        Plot title
    show_crosshair :
        Show crosshair at center
    show_circle :
        Show circle around source
    circle_radius_arcsec :
        Radius of circle in arcseconds
    **kwargs :
        Additional arguments passed to plot_fits_image
    fits_path : Union[str, Path]
    ra: Optional[float] :
         (Default value = None)
    dec: Optional[float] :
         (Default value = None)
    coord: Optional["SkyCoord"] :
         (Default value = None)

    Returns
    -------
        matplotlib Figure object

    """
    _setup_matplotlib()
    import astropy.units as u
    import matplotlib.pyplot as plt
    from astropy.coordinates import SkyCoord
    from astropy.nddata import Cutout2D

    if config is None:
        config = FigureConfig(style=PlotStyle.QUICKLOOK)

    # Get position
    if coord is None:
        if ra is None or dec is None:
            raise ValueError("Must provide either coord or both ra and dec")
        coord = SkyCoord(ra=ra, dec=dec, unit="deg")

    # Load data
    fits_path = Path(fits_path)
    data, wcs, header_info = _get_2d_data_and_wcs(fits_path)

    # Create cutout
    size = radius_arcmin * u.arcmin
    try:
        cutout = Cutout2D(data, position=coord, size=size, wcs=wcs)
    except Exception as e:
        logger.warning(f"Cutout failed: {e}. Using full image.")
        cutout = None

    if cutout is not None:
        data = cutout.data
        wcs = cutout.wcs

    # Normalize
    norm, _, _ = _normalize_image(
        data, **{k: v for k, v in kwargs.items() if k in ["vmin", "vmax", "stretch"]}
    )

    # Create figure
    fig = plt.figure(figsize=config.figsize, dpi=config.dpi)
    ax = fig.add_subplot(111, projection=wcs)

    # Display
    im = ax.imshow(
        data,
        origin="lower",
        cmap=config.cmap,
        norm=norm,
        interpolation="nearest",
    )

    # Coordinate formatting
    ax.coords[0].set_major_formatter("hh:mm:ss")
    ax.coords[1].set_major_formatter("dd:mm:ss")
    ax.set_xlabel("RA (J2000)")
    ax.set_ylabel("Dec (J2000)")

    # Crosshair at center
    if show_crosshair:
        ax.scatter(
            coord.ra.deg,
            coord.dec.deg,
            marker="+",
            c="red",
            s=100,
            linewidths=2,
            transform=ax.get_transform("fk5"),
        )

    # Circle around source
    if show_circle:
        from matplotlib.patches import Circle

        circle = Circle(
            (coord.ra.deg, coord.dec.deg),
            circle_radius_arcsec / 3600.0,
            transform=ax.get_transform("fk5"),
            facecolor="none",
            edgecolor="orange",
            linewidth=2,
        )
        ax.add_patch(circle)

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(f"Flux ({header_info['bunit']})")

    # Title
    if title is None:
        title = f"Cutout: {coord.to_string('hmsdms', precision=1)}"
    ax.set_title(title, fontsize=config.effective_title_size)

    if config.tight_layout:
        fig.tight_layout()

    if output:
        fig.savefig(output, dpi=config.dpi, bbox_inches="tight")
        logger.info(f"Saved cutout to {output}")
        plt.close(fig)

    return fig


def save_quicklook_png(
    fits_path: str | Path,
    output: str | Path | None = None,
    max_size: int = 2048,
) -> Path:
    """Generate a quick PNG from a FITS file.

    Optimized for speed with automatic downsampling for large images.

    Parameters
    ----------
    fits_path :
        Path to FITS file
    output :
        Output PNG path (default: fits_path + '.png')
    max_size :
        Maximum dimension before downsampling
    fits_path : Union[str, Path]
    output : Optional[Union[str, Path]]
         (Default value = None)

    Returns
    -------
        Path to output PNG

    """
    _setup_matplotlib()
    import matplotlib.pyplot as plt

    fits_path = Path(fits_path)
    if output is None:
        output = fits_path.with_suffix(".png")
    output = Path(output)

    # Load data
    data, wcs, header_info = _get_2d_data_and_wcs(fits_path)

    # Downsample if needed
    if max(data.shape) > max_size:
        factor = max(data.shape) // max_size + 1
        h, w = data.shape
        h_new, w_new = h // factor, w // factor
        if h_new > 0 and w_new > 0:
            data = (
                data[: h_new * factor, : w_new * factor]
                .reshape(h_new, factor, w_new, factor)
                .mean(axis=(1, 3))
            )
            logger.debug(f"Downsampled by factor {factor}")

    # Normalize
    norm, _, _ = _normalize_image(data)

    # Create simple figure
    fig, ax = plt.subplots(figsize=(6, 5), dpi=140)

    im = ax.imshow(
        data,
        origin="lower",
        cmap="inferno",
        norm=norm,
        interpolation="nearest",
    )

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(header_info["bunit"])

    ax.set_title(fits_path.name)
    fig.tight_layout()

    fig.savefig(output, dpi=140, bbox_inches="tight")
    plt.close(fig)

    logger.info(f"Saved quicklook PNG: {output}")
    return output


def plot_mosaic_overview(
    mosaic_path: str | Path,
    weight_path: str | Path | None = None,
    output: str | Path | None = None,
    config: FigureConfig | None = None,
) -> Figure:
    """Create an overview plot of a mosaic with optional weight map.

    Parameters
    ----------
    mosaic_path :
        Path to mosaic FITS
    weight_path :
        Path to weight map FITS
    output :
        Output file path
    config :
        Figure configuration
    mosaic_path : Union[str, Path]
    weight_path : Optional[Union[str, Path]]
         (Default value = None)
    output : Optional[Union[str, config: Optional[FigureConfig]
         (Default value = None)

    Returns
    -------
        matplotlib Figure object

    """
    _setup_matplotlib()
    import matplotlib.pyplot as plt

    if config is None:
        config = FigureConfig(style=PlotStyle.QUICKLOOK)

    mosaic_path = Path(mosaic_path)
    data, wcs, header_info = _get_2d_data_and_wcs(mosaic_path)

    # Determine if we have weight map
    has_weights = weight_path is not None and Path(weight_path).exists()

    if has_weights:
        weight_data, _, _ = _get_2d_data_and_wcs(weight_path)
        fig, (ax1, ax2) = plt.subplots(
            1,
            2,
            figsize=(config.figsize[0] * 2, config.figsize[1]),
            subplot_kw={"projection": wcs},
        )
    else:
        fig, ax1 = plt.subplots(
            1,
            1,
            figsize=config.figsize,
            subplot_kw={"projection": wcs},
        )

    # Plot mosaic
    norm, _, _ = _normalize_image(data)
    im1 = ax1.imshow(data, origin="lower", cmap=config.cmap, norm=norm, interpolation="nearest")
    ax1.set_xlabel("RA (J2000)")
    ax1.set_ylabel("Dec (J2000)")
    ax1.set_title(f"Mosaic: {mosaic_path.name}")

    cbar1 = fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    cbar1.set_label(header_info["bunit"])

    # Plot weight map
    if has_weights:
        im2 = ax2.imshow(weight_data, origin="lower", cmap="viridis", interpolation="nearest")
        ax2.set_xlabel("RA (J2000)")
        ax2.set_ylabel("Dec (J2000)")
        ax2.set_title("Weight Map")

        cbar2 = fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
        cbar2.set_label("Weight")

    fig.tight_layout()

    if output:
        fig.savefig(output, dpi=config.dpi, bbox_inches="tight")
        logger.info(f"Saved mosaic overview to {output}")
        plt.close(fig)

    return fig
