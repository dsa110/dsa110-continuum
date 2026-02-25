"""
Generalized catalog tools for imaging: masks and overlays.

Provides catalog-agnostic functions for creating CRTF/FITS masks and
diagnostic overlay images from any supported survey catalog.

Examples
--------
>>> from dsa110_contimg.core.imaging.catalog_tools import create_catalog_mask
>>> # Create mask from unified catalog (default)
>>> mask_path = create_catalog_mask(
...     image_path="image.fits",
...     min_flux_mjy=5.0,
...     radius_arcsec=60.0,
... )
>>> # Create mask from specific catalog
>>> mask_path = create_catalog_mask(
...     image_path="image.fits",
...     catalog="first",
...     min_flux_mjy=1.0,
... )
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales

from dsa110_contimg.core.calibration.catalog_registry import (
    CatalogName,
    query_catalog,
)

logger = logging.getLogger(__name__)


def _image_center_and_radius_deg(hdr) -> tuple[SkyCoord, float]:
    """Get image center and diagonal radius in degrees from FITS header.

    Parameters
    ----------
    hdr :


    """
    w = WCS(hdr).celestial
    nx = int(hdr.get("NAXIS1", 0))
    ny = int(hdr.get("NAXIS2", 0))
    if nx <= 0 or ny <= 0:
        raise ValueError("Invalid image dimensions")

    cx, cy = (nx - 1) / 2.0, (ny - 1) / 2.0
    ctr = w.pixel_to_world(cx, cy)

    scales = proj_plane_pixel_scales(w)  # deg/pixel
    fov_ra = float(scales[0] * nx)
    fov_dec = float(scales[1] * ny)
    half_diag = 0.5 * float(np.hypot(fov_ra, fov_dec))

    return ctr, half_diag


def create_catalog_mask(
    image_path: str | Path,
    catalog: CatalogName | str = CatalogName.UNICAT,
    min_flux_mjy: float | None = None,
    radius_arcsec: float = 60.0,
    out_path: str | Path | None = None,
    output_format: str = "crtf",
) -> str:
    """Create a mask from catalog sources for CLEAN imaging.

        Generates either a CASA Region Text Format (CRTF) file or FITS mask
        with circular regions centered on catalog sources within the image FoV.

    Parameters
    ----------
    image_path : Union[str, Path]
        Path to input FITS image (for WCS/FoV)
    catalog : Union[CatalogName, str], optional
        Catalog to use (default: unicat). Options: nvss, first, vlass, unicat, atnf, rax
    min_flux_mjy : float, optional
        Minimum flux in mJy (uses catalog default if not specified)
    radius_arcsec : float, optional
        Mask radius around each source in arcseconds
    out_path : Union[str, Path], optional
        Output path (auto-generated if not specified)
    output_format : str, optional
        "crtf" for CASA region file, "fits" for FITS mask

    Returns
    -------
        str
        Path to created mask file

    Examples
    --------
        >>> mask = create_catalog_mask("observation.fits", catalog="unicat", min_flux_mjy=10.0)
        >>> print(f"Mask created: {mask}")
    """
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Normalize catalog name
    if isinstance(catalog, str):
        catalog = CatalogName(catalog.lower())

    # Read image header for WCS
    hdr = fits.getheader(str(image_path))
    center, radius_deg = _image_center_and_radius_deg(hdr)

    # Query catalog
    sources = query_catalog(
        catalog=catalog,
        ra_deg=center.ra.deg,
        dec_deg=center.dec.deg,
        radius_deg=radius_deg * 1.1,  # 10% padding
        min_flux_mjy=min_flux_mjy,
    )

    logger.info(
        "Found %d %s sources in FoV (flux >= %.1f mJy)",
        len(sources),
        catalog.value,
        min_flux_mjy or 0,
    )

    # Generate output path
    if out_path is None:
        suffix = f".{catalog.value}_mask"
        if output_format == "fits":
            out_path = str(image_path).replace(".fits", f"{suffix}.fits")
        else:
            out_path = str(image_path).replace(".fits", f"{suffix}.crtf")
    out_path = Path(out_path)

    os.makedirs(out_path.parent, exist_ok=True)

    if output_format == "fits":
        return _write_fits_mask(hdr, sources, radius_arcsec, out_path)
    else:
        return _write_crtf_mask(sources, radius_arcsec, out_path)


def _write_crtf_mask(sources, radius_arcsec: float, out_path: Path) -> str:
    """Write CASA Region Text Format mask file.

    Parameters
    ----------
    sources :

    """
    with open(out_path, "w") as f:
        f.write("#CRTFv0\n")
        for _, row in sources.iterrows():
            src = SkyCoord(row["ra_deg"] * u.deg, row["dec_deg"] * u.deg, frame="icrs")
            ra_str = src.ra.to_string(unit=u.hourangle, sep=":", precision=2, pad=True)
            dec_str = src.dec.to_string(unit=u.deg, sep=":", precision=2, pad=True, alwayssign=True)
            f.write(f"circle[[{ra_str}, {dec_str}], {float(radius_arcsec):.3f}arcsec]\n")

    logger.info("Wrote CRTF mask: %s (%d regions)", out_path, len(sources))
    return str(out_path)


def _write_fits_mask(hdr, sources, radius_arcsec: float, out_path: Path) -> str:
    """Write FITS mask file for WSClean.

    Parameters
    ----------
    hdr :

    sources :

    """
    wcs = WCS(hdr).celestial
    nx = int(hdr.get("NAXIS1", 0))
    ny = int(hdr.get("NAXIS2", 0))

    # Get cell size from WCS
    scales = proj_plane_pixel_scales(wcs)  # deg/pixel
    cell_arcsec = float(scales[0]) * 3600.0  # Convert to arcsec

    # Initialize mask (0 = not cleaned)
    mask = np.zeros((ny, nx), dtype=np.float32)

    radius_pixels = radius_arcsec / cell_arcsec

    for _, row in sources.iterrows():
        coord = SkyCoord(row["ra_deg"] * u.deg, row["dec_deg"] * u.deg, frame="icrs")
        try:
            x, y = wcs.world_to_pixel(coord)
        except Exception:
            continue

        # Skip if outside image bounds
        if x < 0 or x >= nx or y < 0 or y >= ny:
            continue

        # Create circular mask efficiently: only process bounding box
        x_min = int(max(0, x - radius_pixels))
        x_max = int(min(nx, x + radius_pixels + 1))
        y_min = int(max(0, y - radius_pixels))
        y_max = int(min(ny, y + radius_pixels + 1))

        # Create meshgrid only for bounding box
        yy, xx = np.mgrid[y_min:y_max, x_min:x_max]
        dist_sq = (xx - x) ** 2 + (yy - y) ** 2
        circle_mask = dist_sq <= radius_pixels**2

        # Apply to mask
        mask[y_min:y_max, x_min:x_max][circle_mask] = 1.0

    # Write FITS mask
    # Build minimal WCS header for 2D mask
    out_hdr = wcs.to_header()
    hdu = fits.PrimaryHDU(data=mask, header=out_hdr)
    hdu.writeto(str(out_path), overwrite=True)

    logger.info("Wrote FITS mask: %s (%d sources)", out_path, len(sources))
    return str(out_path)


def create_catalog_fits_mask(
    imagename: str,
    imsize: int,
    cell_arcsec: float,
    ra0_deg: float,
    dec0_deg: float,
    catalog: CatalogName | str = CatalogName.UNICAT,
    min_flux_mjy: float | None = None,
    radius_arcsec: float = 60.0,
    out_path: str | None = None,
) -> str:
    """Create FITS mask from catalog sources for WSClean.

    Creates a FITS mask file with circular regions around catalog sources.
    Zero values = not cleaned, non-zero values = cleaned.

    This function creates a mask from scratch given imaging parameters,
    without requiring an existing FITS image.

    Parameters
    ----------
    imagename :
        Base image name (used to determine output path)
    imsize :
        Image size in pixels
    cell_arcsec :
        Pixel scale in arcseconds
    ra0_deg :
        Phase center RA in degrees
    dec0_deg :
        Phase center Dec in degrees
    catalog :
        Catalog to use (default: unicat)
    min_flux_mjy :
        Minimum flux in mJy (uses catalog default if not specified)
    radius_arcsec :
        Mask radius around each source in arcseconds
    out_path :
        Optional output path (defaults to {imagename}.{catalog}_mask.fits)

    Returns
    -------
        Path to created FITS mask file

    """
    # Normalize catalog name
    if isinstance(catalog, str):
        catalog = CatalogName(catalog.lower())

    # Create WCS for mask
    wcs = WCS(naxis=2)
    wcs.wcs.crpix = [imsize / 2.0 + 0.5, imsize / 2.0 + 0.5]
    wcs.wcs.crval = [ra0_deg, dec0_deg]
    wcs.wcs.cdelt = [-cell_arcsec / 3600.0, cell_arcsec / 3600.0]  # Negative RA
    wcs.wcs.ctype = ["RA---SIN", "DEC--SIN"]

    # Calculate FoV radius
    fov_radius_deg = (cell_arcsec * imsize) / 3600.0 / 2.0

    # Query catalog
    sources = query_catalog(
        catalog=catalog,
        ra_deg=ra0_deg,
        dec_deg=dec0_deg,
        radius_deg=fov_radius_deg * 1.1,  # 10% padding
        min_flux_mjy=min_flux_mjy,
    )

    logger.info(
        "Found %d %s sources for mask (flux >= %.1f mJy)",
        len(sources),
        catalog.value,
        min_flux_mjy or 0,
    )

    # Initialize mask (all zeros = not cleaned)
    mask = np.zeros((imsize, imsize), dtype=np.float32)

    if len(sources) == 0:
        # No sources found, create empty mask
        if out_path is None:
            out_path = f"{imagename}.{catalog.value}_mask.fits"
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

        hdu = fits.PrimaryHDU(data=mask, header=wcs.to_header())
        hdu.writeto(out_path, overwrite=True)
        return out_path

    # Create circular masks
    radius_pixels = radius_arcsec / cell_arcsec

    for _, row in sources.iterrows():
        coord = SkyCoord(row["ra_deg"] * u.deg, row["dec_deg"] * u.deg, frame="icrs")
        try:
            x, y = wcs.world_to_pixel(coord)
        except Exception:
            continue

        # Skip if outside image bounds
        if x < 0 or x >= imsize or y < 0 or y >= imsize:
            continue

        # Create circular mask efficiently
        x_min = int(max(0, x - radius_pixels))
        x_max = int(min(imsize, x + radius_pixels + 1))
        y_min = int(max(0, y - radius_pixels))
        y_max = int(min(imsize, y + radius_pixels + 1))

        yy, xx = np.mgrid[y_min:y_max, x_min:x_max]
        dist_sq = (xx - x) ** 2 + (yy - y) ** 2
        circle_mask = dist_sq <= radius_pixels**2

        mask[y_min:y_max, x_min:x_max][circle_mask] = 1.0

    # Write FITS mask
    if out_path is None:
        out_path = f"{imagename}.{catalog.value}_mask.fits"

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    hdu = fits.PrimaryHDU(data=mask, header=wcs.to_header())
    hdu.writeto(out_path, overwrite=True)

    logger.info("Created %s mask: %s (radius=%.1f arcsec)", catalog.value, out_path, radius_arcsec)
    return out_path


def create_catalog_overlay(
    image_path: str | Path,
    out_path: str | Path,
    catalog: CatalogName | str = CatalogName.UNICAT,
    min_flux_mjy: float = 10.0,
    pb_path: str | Path | None = None,
    pblimit: float = 0.2,
) -> None:
    """Create a diagnostic overlay image with catalog sources marked.

    Parameters
    ----------
    image_path : Union[str, Path]
        Input FITS image path
    out_path : Union[str, Path]
        Output PNG path
    catalog : Union[CatalogName, str], optional
        Catalog to overlay (default: unicat)
    min_flux_mjy : float, optional
        Minimum flux in mJy to plot
    pb_path : Union[str, Path], optional
        Optional primary beam FITS for masking detections
    pblimit : float, optional
        PB cutoff when pb_path is provided

    Examples
    --------
        >>> create_catalog_overlay("image.fits", "overlay.png", catalog="nvss", min_flux_mjy=5.0)
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    image_path = Path(image_path)
    out_path = Path(out_path)

    # Normalize catalog name
    if isinstance(catalog, str):
        catalog = CatalogName(catalog.lower())

    # Load image
    data = fits.getdata(str(image_path))
    hdr = fits.getheader(str(image_path))
    while data.ndim > 2:
        data = data[0]

    # Get FoV
    center, radius_deg = _image_center_and_radius_deg(hdr)

    # Query catalog
    sources = query_catalog(
        catalog=catalog,
        ra_deg=center.ra.deg,
        dec_deg=center.dec.deg,
        radius_deg=radius_deg * 1.1,
        min_flux_mjy=min_flux_mjy,
    )

    # Load PB mask if provided
    pb_mask = None
    if pb_path:
        try:
            pb_data = fits.getdata(str(pb_path))
            while pb_data.ndim > 2:
                pb_data = pb_data[0]
            pb_mask = np.isfinite(pb_data) & (pb_data >= float(pblimit))
        except (OSError, ValueError, KeyError):
            pass

    # Plot
    wcs = WCS(hdr).celestial
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={"projection": wcs})

    # Image
    m_data = np.isfinite(data)
    if np.any(m_data):
        vals = data[m_data]
        vmin, vmax = np.percentile(vals, [1, 99])
        ax.imshow(data, origin="lower", cmap="gray", vmin=vmin, vmax=vmax)
        if pb_mask is not None and pb_mask.shape == data.shape:
            ax.contour(pb_mask, levels=[0.5], colors="cyan", linewidths=1, alpha=0.5)

    # Overlay catalog sources
    if len(sources) > 0:
        coords = SkyCoord(
            sources["ra_deg"].values * u.deg,
            sources["dec_deg"].values * u.deg,
            frame="icrs",
        )
        pix = wcs.world_to_pixel(coords)

        # Scale circle sizes by log10(flux)
        fluxes = np.maximum(sources["flux_mjy"].values, 1.0)
        log_flux = np.log10(fluxes)
        sizes = (
            50 * (log_flux - np.min(log_flux)) / (np.max(log_flux) - np.min(log_flux) + 1e-6) + 20
        )

        ax.scatter(
            pix[0],
            pix[1],
            s=sizes,
            facecolors="none",
            edgecolors="red",
            linewidths=1.5,
            alpha=0.7,
            label=catalog.value.upper(),
        )

        # Label brightest sources
        n_label = min(10, len(sources))
        brightest = sources.nlargest(n_label, "flux_mjy")
        for _, row in brightest.iterrows():
            coord = SkyCoord(row["ra_deg"] * u.deg, row["dec_deg"] * u.deg, frame="icrs")
            try:
                pix_single = wcs.world_to_pixel(coord)
                ax.text(
                    pix_single[0],
                    pix_single[1],
                    f"{row['flux_mjy']:.1f}",
                    color="yellow",
                    fontsize=8,
                    ha="center",
                    va="bottom",
                )
            except Exception:
                pass

    ax.set_xlabel("RA")
    ax.set_ylabel("Dec")
    ax.set_title(f"{image_path.name} + {catalog.value.upper()}")
    ax.legend()

    os.makedirs(out_path.parent, exist_ok=True)
    plt.tight_layout()
    plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close()

    logger.info("Wrote overlay: %s (%d %s sources)", out_path, len(sources), catalog.value)
