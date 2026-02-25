"""Shared utilities for creating synthetic FITS images.

This module provides a consolidated implementation of synthetic FITS image
generation to avoid code duplication across scripts and tests.
"""

import random
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.time import Time
from astropy.wcs import WCS


def create_synthetic_fits(
    output_path: Path,
    ra_deg: float = 180.0,
    dec_deg: float = 35.0,
    image_size: int = 512,
    pixel_scale_arcsec: float = 2.0,
    noise_level_jy: float = 0.001,
    n_sources: int = 5,
    source_flux_range_jy: tuple = (0.01, 0.1),
    beam_fwhm_pix: float = 10.0,
    sources: list = None,
    mark_synthetic: bool = True,
    add_wsclean_history: bool = True,
) -> Path:
    """Create a synthetic FITS image with point sources and noise.

    This is a consolidated implementation that replaces duplicate functions
    in scripts/create_synthetic_images.py and test files.

    Parameters
    ----------
    output_path :
        Path to output FITS file
    ra_deg :
        Right ascension of image center (degrees)
    dec_deg :
        Declination of image center (degrees)
    image_size :
        Image size in pixels (square)
    pixel_scale_arcsec :
        Pixel scale in arcseconds
    noise_level_jy :
        RMS noise level in Jy
    n_sources :
        Number of point sources to add (if sources not provided)
    source_flux_range_jy :
        (min, max) flux range for sources in Jy
    beam_fwhm_pix :
        Beam FWHM in pixels
    sources :
        Optional list of dicts with keys: ra_deg, dec_deg, flux_jy, name
        If provided, n_sources is ignored
    mark_synthetic :
        If True, add synthetic provenance markers to header
    add_wsclean_history :
        If True, add fake wsclean history for gridder/mask tests

    Returns
    -------
        Path to created FITS file

    """
    # Create WCS
    w = WCS(naxis=2)
    w.wcs.crpix = [image_size / 2, image_size / 2]
    w.wcs.cdelt = [
        -pixel_scale_arcsec / 3600.0,
        pixel_scale_arcsec / 3600.0,
    ]  # Negative for RA
    w.wcs.crval = [ra_deg, dec_deg]
    w.wcs.ctype = ["RA---SIN", "DEC--SIN"]

    # Create image data with noise
    data = np.random.normal(0, noise_level_jy, (image_size, image_size))

    # Add point sources
    if sources is None:
        # Generate random sources
        for _ in range(n_sources):
            # Random position (avoid edges)
            x = random.randint(image_size // 4, 3 * image_size // 4)
            y = random.randint(image_size // 4, 3 * image_size // 4)

            # Random flux
            flux_jy = random.uniform(*source_flux_range_jy)

            # Add Gaussian source
            sigma_pix = beam_fwhm_pix / 2.355
            y_grid, x_grid = np.ogrid[:image_size, :image_size]
            gaussian = flux_jy * np.exp(
                -((x_grid - x) ** 2 + (y_grid - y) ** 2) / (2 * sigma_pix**2)
            )
            data += gaussian
    else:
        # Use provided source list
        sigma_pix = beam_fwhm_pix / 2.355
        y_grid, x_grid = np.ogrid[:image_size, :image_size]

        for src in sources:
            ra_src = src.get("ra_deg", ra_deg)
            dec_src = src.get("dec_deg", dec_deg)
            flux_src = src.get("flux_jy", 0.01)

            # Convert RA/Dec to pixel coordinates
            pix_coords = w.world_to_pixel_values(ra_src, dec_src)
            x0, y0 = float(pix_coords[0]), float(pix_coords[1])

            if not (0 <= x0 < image_size and 0 <= y0 < image_size):
                continue  # Skip sources outside image bounds

            # Create 2D Gaussian
            gaussian = flux_src * np.exp(
                -((x_grid - x0) ** 2 + (y_grid - y0) ** 2) / (2 * sigma_pix**2)
            )
            data += gaussian

    # Create FITS HDU
    hdu = fits.PrimaryHDU(data=data, header=w.to_header())

    # Add standard FITS keywords
    hdu.header["BUNIT"] = "Jy/beam"
    hdu.header["BTYPE"] = "Intensity"
    hdu.header["BSCALE"] = 1.0
    hdu.header["BZERO"] = 0.0
    hdu.header["BMAJ"] = beam_fwhm_pix * pixel_scale_arcsec / 3600.0  # degrees
    hdu.header["BMIN"] = beam_fwhm_pix * pixel_scale_arcsec / 3600.0
    hdu.header["BPA"] = 0.0
    hdu.header["DATE-OBS"] = Time.now().isot

    # Provenance marking
    if mark_synthetic:
        hdu.header["OBJECT"] = "Synthetic Test Image"
        hdu.header["SYNTH"] = True  # Use <=8 char keyword to avoid HIERARCH
        hdu.header["COMMENT"] = "This is synthetic test data generated for testing purposes"
    else:
        hdu.header["OBJECT"] = "Test Image"

    if add_wsclean_history:
        hdu.header["HISTORY"] = "WSCLEAN: wsclean -gridder idg -clean-mask mymask.fits"
        hdu.header["WSCGRID"] = "IDG"

    # Write FITS file
    hdu.writeto(output_path, overwrite=True)

    return output_path
