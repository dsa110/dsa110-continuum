"""
Mosaic-specific visualization utilities.

Provides functions for:
- Individual tile thumbnails
- Mosaic overview with tile footprints
- Coverage maps

Adapted from VAST/vaster patterns and radiopadre/fitsfile.py
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
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


def plot_tile_grid(
    tiles: Sequence[str | Path],
    output: str | Path | None = None,
    config: FigureConfig | None = None,
    ncols: int = 4,
    show_labels: bool = True,
    stretch: str = "asinh",
) -> Figure:
    """Plot a grid of tile thumbnails.

    Parameters
    ----------
    tiles : list
        List of tile FITS file paths.
    output : str or Path, optional
        Output file path.
    config : FigureConfig, optional
        Figure configuration.
    ncols : int
        Number of columns in grid.
    show_labels : bool
        Show tile names.
    stretch : str
        Image stretch ('linear', 'log', 'sqrt', 'asinh').

    Returns
    -------
    Figure
        matplotlib Figure object.
    """
    _setup_matplotlib()
    import matplotlib.pyplot as plt
    from astropy.io import fits
    from astropy.visualization import AsinhStretch, ImageNormalize, LogStretch, SqrtStretch

    if config is None:
        config = FigureConfig(style=PlotStyle.QUICKLOOK)

    tiles = [Path(t) for t in tiles if Path(t).exists()]
    n_tiles = len(tiles)
    if n_tiles == 0:
        logger.warning("No valid tile files found")
        fig, ax = plt.subplots(figsize=config.figsize)
        ax.text(0.5, 0.5, "No tiles found", ha="center", va="center")
        return fig

    nrows = (n_tiles + ncols - 1) // ncols

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(config.figsize[0] * ncols / 2, config.figsize[1] * nrows / 2),
        squeeze=False,
    )

    # Choose stretch
    stretch_map = {
        "linear": None,
        "log": LogStretch(),
        "sqrt": SqrtStretch(),
        "asinh": AsinhStretch(),
    }
    stretch_fn = stretch_map.get(stretch, AsinhStretch())

    for idx, tile_path in enumerate(tiles):
        row, col = divmod(idx, ncols)
        ax = axes[row, col]

        try:
            with fits.open(tile_path) as hdul:
                data = hdul[0].data.squeeze()

            # Robust normalization
            vmin, vmax = np.nanpercentile(data, [1, 99])
            if stretch_fn:
                norm = ImageNormalize(vmin=vmin, vmax=vmax, stretch=stretch_fn)
            else:
                norm = None

            ax.imshow(
                data,
                origin="lower",
                cmap=config.colormap,
                norm=norm,
                vmin=vmin if not stretch_fn else None,
                vmax=vmax if not stretch_fn else None,
            )

            if show_labels:
                ax.set_title(tile_path.stem, fontsize=config.effective_tick_size)

        except Exception as e:
            logger.warning(f"Failed to load tile {tile_path}: {e}")
            ax.text(0.5, 0.5, "Load Error", ha="center", va="center", transform=ax.transAxes)

        ax.set_xticks([])
        ax.set_yticks([])

    # Hide empty axes
    for idx in range(n_tiles, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row, col].set_visible(False)

    fig.tight_layout()

    if output:
        fig.savefig(output, dpi=config.dpi, bbox_inches="tight")
        logger.info(f"Saved tile grid: {output}")
        plt.close(fig)

    return fig


def plot_mosaic_footprints(
    tiles: Sequence[str | Path],
    output: str | Path | None = None,
    config: FigureConfig | None = None,
    mosaic_wcs: WCS | None = None,
    mosaic_shape: tuple | None = None,
    title: str = "Mosaic Tile Coverage",
) -> Figure:
    """Plot mosaic footprints showing tile coverage.

    Parameters
    ----------
    tiles : list
        List of tile FITS file paths.
    output : str or Path, optional
        Output file path.
    config : FigureConfig, optional
        Figure configuration.
    mosaic_wcs : WCS, optional
        WCS of the final mosaic (for coordinate frame).
    mosaic_shape : tuple, optional
        Shape of the final mosaic.
    title : str
        Plot title.

    Returns
    -------
    Figure
        matplotlib Figure object.
    """
    _setup_matplotlib()
    import matplotlib.pyplot as plt
    from astropy.io import fits
    from astropy.wcs import WCS

    if config is None:
        config = FigureConfig(style=PlotStyle.QUICKLOOK)

    fig = plt.figure(figsize=(config.figsize[0] * 1.5, config.figsize[1] * 1.5))

    if mosaic_wcs is not None:
        ax = fig.add_subplot(1, 1, 1, projection=mosaic_wcs)
    else:
        ax = fig.add_subplot(1, 1, 1)

    # Collect footprints
    colors = plt.cm.tab10(np.linspace(0, 1, len(tiles)))

    all_ra = []
    all_dec = []

    for idx, tile_path in enumerate(tiles):
        tile_path = Path(tile_path)
        if not tile_path.exists():
            continue

        try:
            with fits.open(tile_path) as hdul:
                header = hdul[0].header
                tile_wcs = WCS(header, naxis=2)
                shape = hdul[0].data.squeeze().shape

            # Get corner coordinates
            ny, nx = shape[-2:]
            corners_pix = np.array([[0, 0], [nx, 0], [nx, ny], [0, ny], [0, 0]])
            corners_world = tile_wcs.pixel_to_world_values(corners_pix[:, 0], corners_pix[:, 1])

            ra_corners = corners_world[0]
            dec_corners = corners_world[1]

            all_ra.extend(ra_corners)
            all_dec.extend(dec_corners)

            # Plot footprint
            if mosaic_wcs is not None:
                # Transform to mosaic pixel coordinates
                pix_x, pix_y = mosaic_wcs.world_to_pixel_values(ra_corners, dec_corners)
                ax.plot(pix_x, pix_y, "-", color=colors[idx], linewidth=2, alpha=0.8)
            else:
                ax.plot(ra_corners, dec_corners, "-", color=colors[idx], linewidth=2, alpha=0.8)

            # Label at center
            center_ra = np.mean(ra_corners[:-1])
            center_dec = np.mean(dec_corners[:-1])

            if mosaic_wcs is not None:
                cx, cy = mosaic_wcs.world_to_pixel_values([center_ra], [center_dec])
                ax.text(
                    cx[0],
                    cy[0],
                    tile_path.stem,
                    fontsize=6,
                    ha="center",
                    va="center",
                    color=colors[idx],
                )
            else:
                ax.text(
                    center_ra,
                    center_dec,
                    tile_path.stem,
                    fontsize=6,
                    ha="center",
                    va="center",
                    color=colors[idx],
                )

        except Exception as e:
            logger.warning(f"Failed to get footprint for {tile_path}: {e}")

    if mosaic_wcs is not None:
        ax.set_xlabel("RA", fontsize=config.effective_label_size)
        ax.set_ylabel("Dec", fontsize=config.effective_label_size)
    else:
        ax.set_xlabel("RA (deg)", fontsize=config.effective_label_size)
        ax.set_ylabel("Dec (deg)", fontsize=config.effective_label_size)
        ax.invert_xaxis()  # RA increases to the left

    ax.set_title(title, fontsize=config.effective_title_size)

    if config.grid:
        ax.grid(True, alpha=0.3)

    fig.tight_layout()

    if output:
        fig.savefig(output, dpi=config.dpi, bbox_inches="tight")
        logger.info(f"Saved footprint plot: {output}")
        plt.close(fig)

    return fig


def plot_coverage_map(
    coverage_data: NDArray,
    wcs: WCS | None = None,
    output: str | Path | None = None,
    config: FigureConfig | None = None,
    title: str = "Coverage Map",
) -> Figure:
    """Plot a coverage map showing number of contributing tiles per pixel.

    Parameters
    ----------
    coverage_data :
        2D array of coverage counts
    wcs :
        WCS for coordinate display
    output :
        Output file path
    config :
        Figure configuration
    title :
        Plot title

    Returns
    -------
        matplotlib Figure object

    """
    _setup_matplotlib()
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm

    if config is None:
        config = FigureConfig(style=PlotStyle.QUICKLOOK)

    coverage_data = np.asarray(coverage_data)
    max_coverage = int(np.nanmax(coverage_data))

    if wcs is not None:
        fig = plt.figure(figsize=config.figsize, dpi=config.dpi)
        ax = fig.add_subplot(1, 1, 1, projection=wcs)
    else:
        fig, ax = plt.subplots(figsize=config.figsize, dpi=config.dpi)

    # Discrete colormap for coverage counts
    cmap = plt.cm.viridis
    bounds = np.arange(0, max_coverage + 2) - 0.5
    norm = BoundaryNorm(bounds, cmap.N)

    im = ax.imshow(coverage_data, origin="lower", cmap=cmap, norm=norm)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8, ticks=range(max_coverage + 1))
    cbar.set_label("Number of Tiles", fontsize=config.effective_label_size)

    ax.set_title(title, fontsize=config.effective_title_size)

    if wcs is not None:
        ax.set_xlabel("RA", fontsize=config.effective_label_size)
        ax.set_ylabel("Dec", fontsize=config.effective_label_size)

    fig.tight_layout()

    if output:
        fig.savefig(output, dpi=config.dpi, bbox_inches="tight")
        logger.info(f"Saved coverage map: {output}")
        plt.close(fig)

    return fig
