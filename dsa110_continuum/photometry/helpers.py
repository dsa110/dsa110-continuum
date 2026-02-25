"""Helper functions for photometry automation.

Provides utilities for extracting field centers from FITS files and querying
catalog sources for photometry measurements.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from astropy.io import fits
from astropy.wcs import WCS

from dsa110_contimg.core.catalog.query import query_sources

logger = logging.getLogger(__name__)


def get_field_center_from_fits(fits_path: Path) -> tuple[float, float]:
    """Extract RA, Dec center from FITS header.

        Uses WCS information from the FITS header to determine the field center.
        Falls back to CRVAL1/CRVAL2 if WCS transformation fails.

    Parameters
    ----------
    fits_path : str
        Path to FITS image file

    Returns
    -------
        tuple of float
        (ra_deg, dec_deg) - Field center coordinates in degrees

    Raises
    ------
        ValueError
        If FITS file cannot be read or has no valid coordinates
        FileNotFoundError
        If FITS file does not exist
    """
    if not fits_path.exists():
        raise FileNotFoundError(f"FITS file not found: {fits_path}")

    with fits.open(fits_path) as hdul:
        hdr = hdul[0].header

        # Try WCS-based extraction first (more accurate)
        try:
            wcs = WCS(hdr)
            if wcs.has_celestial:
                # Get image center in pixels
                naxis1 = hdr.get("NAXIS1", 0)
                naxis2 = hdr.get("NAXIS2", 0)
                if naxis1 > 0 and naxis2 > 0:
                    center_pix = [naxis1 / 2, naxis2 / 2]
                    if hdr.get("NAXIS", 0) >= 2:
                        result = wcs.all_pix2world(center_pix[0], center_pix[1], 0)
                        # Handle both tuple and array returns
                        if isinstance(result, tuple):
                            ra, dec = result[0], result[1]
                        else:
                            ra, dec = float(result[0]), float(result[1])
                        # Normalize RA to [0, 360) range for astronomical convention
                        ra = float(ra) % 360.0
                        logger.debug(f"Extracted field center from WCS: RA={ra:.6f}, Dec={dec:.6f}")
                        return (ra, float(dec))
        except (ValueError, TypeError, AttributeError, KeyError) as e:
            logger.debug(f"WCS extraction failed, trying CRVAL: {e}")

        # Fallback to CRVAL1/CRVAL2 (reference pixel values)
        try:
            ra = hdr.get("CRVAL1")
            dec = hdr.get("CRVAL2")
            if ra is not None and dec is not None:
                # Normalize RA to [0, 360) range for astronomical convention
                ra = float(ra) % 360.0
                logger.debug(f"Extracted field center from CRVAL: RA={ra:.6f}, Dec={dec:.6f}")
                return (ra, float(dec))
        except (ValueError, TypeError) as e:
            logger.debug(f"CRVAL extraction failed: {e}")

        raise ValueError(
            f"Cannot extract field center from FITS file {fits_path}: "
            "No valid WCS or CRVAL values found"
        )


def query_sources_for_fits(
    fits_path: Path,
    catalog: str = "nvss",
    radius_deg: float = 0.5,
    min_flux_mjy: float | None = None,
    max_sources: int | None = None,
    catalog_path: Path | None = None,
    ra_radius_deg: float | None = None,
    dec_radius_deg: float | None = None,
) -> list[dict[str, Any]]:
    """Query catalog sources for a FITS image field.

        Extracts the field center from the FITS file and queries the specified
        catalog for sources within the search radius.

    Parameters
    ----------
    fits_path : str or Path
        Path to FITS image file
    catalog : str
        Catalog type ("nvss", "first", "rax", "vlass", "master")
    radius_deg : float
        Search radius in degrees (used if ra_radius_deg/dec_radius_deg not specified)
    min_flux_mjy : float or None, optional
        Minimum flux threshold in mJy (optional)
    max_sources : int or None, optional
        Maximum number of sources to return (optional)
    catalog_path : str or None, optional
        Path to catalog database file (optional)
    ra_radius_deg : float or None, optional
        Search radius in RA direction in degrees (optional)
    dec_radius_deg : float or None, optional
        Search radius in Dec direction in degrees (optional)

    Returns
    -------
        list of dict
        List of source dictionaries with keys: ra, dec, flux_mjy, etc.
        Returns empty list if no sources found or query fails.

    Examples
    --------
        >>> sources = query_sources_for_fits(
        ...     Path("/data/image.fits"),
        ...     catalog="nvss",
        ...     radius_deg=0.5
        ... )
        >>> for src in sources:
        ...     print(f"Source at {src['ra']:.6f}, {src['dec']:.6f}")
    """
    try:
        # Extract field center
        ra_center, dec_center = get_field_center_from_fits(fits_path)

        # Use separate RA/Dec radii if provided, otherwise use isotropic radius
        effective_ra_radius = ra_radius_deg if ra_radius_deg is not None else radius_deg
        effective_dec_radius = dec_radius_deg if dec_radius_deg is not None else radius_deg

        logger.info(
            f"Querying {catalog} catalog for sources near "
            f"RA={ra_center:.6f}, Dec={dec_center:.6f}, "
            f"RA radius={effective_ra_radius:.3f}deg, Dec radius={effective_dec_radius:.3f}deg"
        )

        # Query catalog with elliptical search if radii differ
        if effective_ra_radius != effective_dec_radius:
            # Use larger radius for initial query, then filter by ellipse
            max_radius = max(effective_ra_radius, effective_dec_radius)
            df = query_sources(
                catalog_type=catalog,
                ra_center=ra_center,
                dec_center=dec_center,
                radius_deg=max_radius,
                min_flux_mjy=min_flux_mjy,
                max_sources=None,  # Filter after elliptical cut
                catalog_path=str(catalog_path) if catalog_path else None,
            )
            # Filter by ellipse: (delta_ra/ra_radius)^2 + (delta_dec/dec_radius)^2 <= 1
            if not df.empty:
                import numpy as np

                delta_ra = (df["ra"] - ra_center) * np.cos(np.radians(dec_center))
                delta_dec = df["dec"] - dec_center
                in_ellipse = (delta_ra / effective_ra_radius) ** 2 + (
                    delta_dec / effective_dec_radius
                ) ** 2 <= 1
                df = df[in_ellipse]
                if max_sources:
                    df = df.head(max_sources)
        else:
            # Circular search (isotropic)
            df = query_sources(
                catalog_type=catalog,
                ra_center=ra_center,
                dec_center=dec_center,
                radius_deg=radius_deg,
                min_flux_mjy=min_flux_mjy,
                max_sources=max_sources,
                catalog_path=str(catalog_path) if catalog_path else None,
            )

        # Convert DataFrame to list of dictionaries
        if df.empty:
            logger.info(f"No sources found in {catalog} catalog for field")
            return []

        sources = df.to_dict("records")
        logger.info(f"Found {len(sources)} sources in {catalog} catalog")
        return sources

    except Exception as e:
        logger.error(f"Failed to query sources for FITS {fits_path}: {e}", exc_info=True)
        return []


def query_sources_for_mosaic(
    mosaic_path: Path,
    catalog: str = "nvss",
    radius_deg: float = 1.0,
    ra_radius_deg: float | None = None,
    dec_radius_deg: float | None = None,
    min_flux_mjy: float | None = None,
    max_sources: int | None = None,
    catalog_path: Path | None = None,
) -> list[dict[str, Any]]:
    """Query catalog sources for a mosaic FITS file.

        Similar to `query_sources_for_fits()` but with a larger default search
        radius (1.0 deg) to account for the larger field of view in mosaics.

    Parameters
    ----------
    mosaic_path : str or Path
        Path to mosaic FITS file
    catalog : str
        Catalog type ("nvss", "first", "rax", "vlass", "master")
    radius_deg : float, optional
        Search radius in degrees, default is 1.0
    min_flux_mjy : float or None, optional
        Minimum flux threshold in mJy (optional)
    max_sources : int or None, optional
        Maximum number of sources to return (optional)
    catalog_path : str or None, optional
        Path to catalog database file (optional)

    Returns
    -------
        list of dict
        List of source dictionaries with keys: ra, dec, flux_mjy, etc.
        Returns empty list if no sources found or query fails.

    Examples
    --------
        >>> sources = query_sources_for_mosaic(
        ...     Path("/data/mosaic.fits"),
        ...     catalog="nvss",
        ...     radius_deg=1.5
        ... )
    """
    try:
        # Extract field center
        ra_center, dec_center = get_field_center_from_fits(mosaic_path)
        logger.info(
            f"Querying {catalog} catalog for sources near mosaic "
            f"RA={ra_center:.6f}, Dec={dec_center:.6f}, radius={radius_deg}deg"
        )

        # Query catalog with larger radius
        df = query_sources(
            catalog_type=catalog,
            ra_center=ra_center,
            dec_center=dec_center,
            radius_deg=radius_deg,
            min_flux_mjy=min_flux_mjy,
            max_sources=max_sources,
            catalog_path=str(catalog_path) if catalog_path else None,
        )

        # Convert DataFrame to list of dictionaries
        if df.empty:
            logger.info(f"No sources found in {catalog} catalog for mosaic")
            return []

        sources = df.to_dict("records")
        logger.info(f"Found {len(sources)} sources in {catalog} catalog for mosaic")
        return sources

    except Exception as e:
        logger.error(f"Failed to query sources for mosaic {mosaic_path}: {e}", exc_info=True)
        return []
