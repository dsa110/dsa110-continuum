"""Direct FITS Image Generation for DSA-110 Tiles.

This module provides fast, direct generation of synthetic FITS images that
match the structure and characteristics of standard DSA-110 5-minute observation
tiles, without requiring full end-to-end visibility simulation.

The direct FITS approach is ideal for:
- Rapid testing of photometry and catalog tools
- Validation of image processing pipelines
- Generating test datasets with known source properties
- Benchmarking and performance testing

Examples
--------
>>> from dsa110_contimg.core.simulation.simulate_tile_fits import create_synthetic_tile
>>>
>>> # Generate tile with catalog sources
>>> from dsa110_contimg.infrastructure.database import data_config
>>> fits_path = create_synthetic_tile(
...     output_path=str(data_config.STAGE_BASE / "sim_tile.fits"),
...     ra_deg=180.0,
...     dec_deg=35.0,
...     obs_timestamp="2025-01-15T12:00:00",
...     source_mode="catalog",
...     catalog_name="nvss",
...     min_flux_mjy=10.0,
... )
>>>
>>> # Generate tile with parametric sources
>>> fits_path = create_synthetic_tile(
...     output_path=str(data_config.STAGE_BASE / "sim_tile_param.fits"),
...     ra_deg=45.0,
...     dec_deg=15.0,
...     source_mode="parametric",
...     num_sources=30,
...     flux_range_mjy=(5.0, 100.0),
... )
"""

import logging
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.time import Time
from astropy.wcs import WCS

from dsa110_contimg.core.simulation.tile_specs import STANDARD_TILE, TileSpecification

logger = logging.getLogger(__name__)


def create_wcs(
    ra_deg: float,
    dec_deg: float,
    image_size_px: int,
    pixel_scale_arcsec: float,
) -> WCS:
    """Create WCS for a DSA-110 tile image.

    Parameters
    ----------
    ra_deg : float
        Right ascension of field center (degrees)
    dec_deg : float
        Declination of field center (degrees)
    image_size_px : int
        Image size in pixels (square)
    pixel_scale_arcsec : float
        Pixel scale in arcseconds per pixel

    Returns
    -------
    wcs : astropy.wcs.WCS
        World Coordinate System object

    Examples
    --------
    >>> wcs = create_wcs(180.0, 35.0, 2048, 6.0)
    >>> print(wcs)
    """
    w = WCS(naxis=2)

    # Reference pixel at image center
    w.wcs.crpix = [image_size_px / 2.0, image_size_px / 2.0]

    # Pixel scale (convert arcsec to degrees, RA is negative)
    pixel_scale_deg = pixel_scale_arcsec / 3600.0
    w.wcs.cdelt = [-pixel_scale_deg, pixel_scale_deg]

    # Reference coordinates
    w.wcs.crval = [ra_deg, dec_deg]

    # Projection type (SIN for DSA-110 drift-scan)
    w.wcs.ctype = ["RA---SIN", "DEC--SIN"]

    return w


def add_gaussian_source(
    data: np.ndarray,
    x_center: float,
    y_center: float,
    flux_jy: float,
    beam_fwhm_pix: float,
) -> None:
    """Add a Gaussian source to image data (in-place).

    Parameters
    ----------
    data : np.ndarray
        2D image array (modified in-place)
    x_center : float
        X pixel coordinate of source center
    y_center : float
        Y pixel coordinate of source center
    flux_jy : float
        Integrated flux in Jy
    beam_fwhm_pix : float
        Beam FWHM in pixels

    Examples
    --------
    >>> data = np.zeros((512, 512))
    >>> add_gaussian_source(data, 256, 256, 0.05, 5.0)
    >>> print(f"Peak: {data.max():.3f} Jy/beam")
    """
    ny, nx = data.shape

    # Skip if source is outside image
    if not (0 <= x_center < nx and 0 <= y_center < ny):
        return

    # Gaussian sigma from FWHM
    sigma_pix = beam_fwhm_pix / 2.355

    # Create meshgrid centered on source
    y_grid, x_grid = np.ogrid[:ny, :nx]

    # Gaussian profile (normalized)
    gaussian = np.exp(-((x_grid - x_center) ** 2 + (y_grid - y_center) ** 2) / (2 * sigma_pix**2))

    # Scale to desired integrated flux
    # For a 2D Gaussian, peak = flux / (2 * π * σ^2)
    normalization = 1.0 / (2.0 * np.pi * sigma_pix**2)
    peak_jy_beam = flux_jy * normalization

    # Add to data
    data += peak_jy_beam * gaussian


def generate_parametric_sources(
    num_sources: int,
    flux_range_mjy: tuple[float, float],
    image_size_px: int,
    rng: np.random.Generator | None = None,
) -> list[dict]:
    """Generate parametric source list with realistic flux distribution.

    Uses a power-law flux distribution typical of radio source populations.

    Parameters
    ----------
    num_sources : int
        Number of sources to generate
    flux_range_mjy : Tuple[float, float]
        (min_flux, max_flux) in mJy
    image_size_px : int
        Image size for random positioning
    rng : np.random.Generator, optional
        Random number generator

    Returns
    -------
    sources : List[dict]
        List of source dictionaries with keys: x_px, y_px, flux_jy

    Examples
    --------
    >>> sources = generate_parametric_sources(20, (5.0, 100.0), 2048)
    >>> print(f"Generated {len(sources)} sources")
    >>> print(f"Flux range: {min(s['flux_jy'] for s in sources)*1000:.1f} - "
    ...       f"{max(s['flux_jy'] for s in sources)*1000:.1f} mJy")
    """
    if rng is None:
        rng = np.random.default_rng()

    sources = []

    # Power-law index for flux distribution (typical: -1.5 to -2.0)
    alpha = -1.7

    min_flux_jy = flux_range_mjy[0] / 1000.0
    max_flux_jy = flux_range_mjy[1] / 1000.0

    for _ in range(num_sources):
        # Random position (avoid edges)
        margin = image_size_px // 10
        x_px = rng.uniform(margin, image_size_px - margin)
        y_px = rng.uniform(margin, image_size_px - margin)

        # Power-law distributed flux
        # Use inverse transform sampling: F = (F_max^(α+1) - u*(F_max^(α+1) - F_min^(α+1)))^(1/(α+1))
        u = rng.uniform(0, 1)
        flux_jy = (
            max_flux_jy ** (alpha + 1)
            - u * (max_flux_jy ** (alpha + 1) - min_flux_jy ** (alpha + 1))
        ) ** (1.0 / (alpha + 1))

        sources.append(
            {
                "x_px": x_px,
                "y_px": y_px,
                "flux_jy": flux_jy,
            }
        )

    return sources


def create_fits_header(
    wcs: WCS,
    obs_timestamp: str,
    ra_deg: float,
    dec_deg: float,
    beam_fwhm_arcsec: float,
    noise_rms_mjy: float,
    tile_spec: TileSpecification,
    source_mode: str,
    **extra_keywords,
) -> fits.Header:
    """Create FITS header with DSA-110-specific keywords.

    Parameters
    ----------
    wcs : astropy.wcs.WCS
        World coordinate system
    obs_timestamp : str
        Observation timestamp (ISO format)
    ra_deg : float
        Field center RA (degrees)
    dec_deg : float
        Field center Dec (degrees)
    beam_fwhm_arcsec : float
        Beam FWHM (arcseconds)
    noise_rms_mjy : float
        RMS noise (mJy/beam)
    tile_spec : TileSpecification
        Tile specification
    source_mode : str
        Source generation mode
    **extra_keywords
        Additional FITS keywords

    Returns
    -------
    header : astropy.io.fits.Header
        FITS header
    """
    # Start with WCS header
    header = wcs.to_header()

    # Standard FITS keywords
    header["BUNIT"] = ("Jy/beam", "Brightness unit")
    header["BTYPE"] = ("Intensity", "Type of image")
    header["BSCALE"] = (1.0, "Scaling factor")
    header["BZERO"] = (0.0, "Offset")

    # Beam parameters
    beam_deg = beam_fwhm_arcsec / 3600.0
    header["BMAJ"] = (beam_deg, "Beam major axis (deg)")
    header["BMIN"] = (beam_deg, "Beam minor axis (deg)")
    header["BPA"] = (0.0, "Beam position angle (deg)")

    # Observation parameters
    header["DATE-OBS"] = (obs_timestamp, "Observation timestamp")
    header["MJD-OBS"] = (Time(obs_timestamp).mjd, "MJD of observation")
    header["OBSRA"] = (ra_deg, "Field center RA (deg)")
    header["OBSDEC"] = (dec_deg, "Field center Dec (deg)")

    # DSA-110 specific
    header["TELESCOP"] = ("DSA-110", "Telescope name")
    header["INSTRUME"] = ("DSA-110", "Instrument name")
    header["OBSERVER"] = ("DSA-110", "Observer")

    # Frequency information
    header["RESTFRQ"] = (tile_spec.reference_freq_hz, "Rest frequency (Hz)")
    header["FREQ"] = (tile_spec.reference_freq_hz, "Observation frequency (Hz)")

    # Synthetic data markers
    header["SYNTH"] = (True, "Synthetic/simulated data")
    header["SIMMODE"] = (source_mode, "Source generation mode")
    header["DATAGEN"] = ("simulate_tile_fits", "Data generation tool")
    header["TILEDUR"] = (tile_spec.total_duration_sec, "Tile duration (s)")
    header["NSUBBAND"] = (tile_spec.num_subbands, "Number of subbands")
    header["NANT"] = (tile_spec.num_antennas, "Number of antennas")

    # Noise and quality
    header["NOISE"] = (noise_rms_mjy / 1000.0, "RMS noise (Jy/beam)")
    header["RMSNOISE"] = (noise_rms_mjy, "RMS noise (mJy/beam)")

    # Imaging parameters
    header["WGHTSCHM"] = (tile_spec.weighting, "Visibility weighting scheme")
    header["ROBUST"] = (tile_spec.robust, "Briggs robustness")

    # Software version
    header["ORIGIN"] = ("dsa110-contimg", "Data origin")
    header["COMMENT"] = "Synthetic DSA-110 tile image for testing"

    # Add extra keywords
    for key, value in extra_keywords.items():
        if isinstance(value, tuple):
            header[key] = value  # (value, comment)
        else:
            header[key] = value

    return header


def create_synthetic_tile(
    output_path: Path,
    ra_deg: float,
    dec_deg: float,
    obs_timestamp: str,
    source_mode: str = "parametric",
    num_sources: int = 30,
    flux_range_mjy: tuple[float, float] = (5.0, 100.0),
    catalog_sources: list[dict] | None = None,
    noise_rms_mjy: float = 1.0,
    beam_fwhm_arcsec: float | None = None,
    tile_spec: TileSpecification | None = None,
    add_noise: bool = True,
    random_seed: int | None = None,
) -> Path:
    """Create a synthetic DSA-110 tile FITS image.

    Parameters
    ----------
    output_path : Path
        Output FITS file path
    ra_deg : float
        Field center right ascension (degrees)
    dec_deg : float
        Field center declination (degrees)
    obs_timestamp : str
        Observation timestamp (ISO format, e.g., "2025-01-15T12:00:00")
    source_mode : str, optional
        Source generation mode: "parametric", "catalog", or "custom"
        Default: "parametric"
    num_sources : int, optional
        Number of sources (for parametric mode). Default: 30
    flux_range_mjy : Tuple[float, float], optional
        (min, max) flux range in mJy (for parametric mode). Default: (5.0, 100.0)
    catalog_sources : List[dict], optional
        Custom source list (for custom mode). Each dict should have keys:
        ra_deg, dec_deg, flux_jy
    noise_rms_mjy : float, optional
        RMS noise level in mJy/beam. Default: 1.0
    beam_fwhm_arcsec : float, optional
        Beam FWHM in arcseconds (default: from tile_spec)
    tile_spec : TileSpecification, optional
        Tile specification (default: STANDARD_TILE)
    add_noise : bool, optional
        Add Gaussian noise. Default: True
    random_seed : int, optional
        Random seed for reproducibility

    Returns
    -------
    output_path : Path
        Path to created FITS file

    Examples
    --------
    >>> from pathlib import Path
    >>> from dsa110_contimg.infrastructure.database import data_config
    >>> fits_path = create_synthetic_tile(
    ...     output_path=data_config.STAGE_BASE / "test_tile.fits",
    ...     ra_deg=180.0,
    ...     dec_deg=35.0,
    ...     obs_timestamp="2025-01-15T12:00:00",
    ...     source_mode="parametric",
    ...     num_sources=20,
    ... )
    >>> print(f"Created: {fits_path}")
    """
    if tile_spec is None:
        tile_spec = STANDARD_TILE

    if beam_fwhm_arcsec is None:
        beam_fwhm_arcsec = tile_spec.typical_beam_fwhm_arcsec

    # Initialize RNG
    rng = np.random.default_rng(random_seed)

    logger.info(f"Creating synthetic tile: {output_path}")
    logger.info(f"  Field center: RA={ra_deg:.4f}, Dec={dec_deg:.4f}")
    logger.info(f"  Observation: {obs_timestamp}")
    logger.info(f"  Source mode: {source_mode}")

    # Create WCS
    wcs = create_wcs(
        ra_deg,
        dec_deg,
        tile_spec.image_size_px,
        tile_spec.pixel_scale_arcsec,
    )

    # Initialize image data
    data = np.zeros((tile_spec.image_size_px, tile_spec.image_size_px), dtype=np.float32)

    # Convert beam FWHM to pixels
    beam_fwhm_pix = beam_fwhm_arcsec / tile_spec.pixel_scale_arcsec

    # Generate or use sources
    if source_mode == "parametric":
        logger.info(f"  Generating {num_sources} parametric sources")
        sources = generate_parametric_sources(
            num_sources,
            flux_range_mjy,
            tile_spec.image_size_px,
            rng=rng,
        )
        # Add sources to image
        for src in sources:
            add_gaussian_source(
                data,
                src["x_px"],
                src["y_px"],
                src["flux_jy"],
                beam_fwhm_pix,
            )
        logger.info(f"  Added {len(sources)} sources")

    elif source_mode == "catalog" and catalog_sources is not None:
        logger.info(f"  Using {len(catalog_sources)} catalog sources")
        num_added = 0
        for src in catalog_sources:
            # Convert RA/Dec to pixel coordinates
            try:
                pix_coords = wcs.world_to_pixel_values(src["ra_deg"], src["dec_deg"])
                x_px, y_px = float(pix_coords[0]), float(pix_coords[1])

                # Add source if within image bounds
                if 0 <= x_px < tile_spec.image_size_px and 0 <= y_px < tile_spec.image_size_px:
                    add_gaussian_source(
                        data,
                        x_px,
                        y_px,
                        src["flux_jy"],
                        beam_fwhm_pix,
                    )
                    num_added += 1
            except Exception as e:
                logger.warning(f"  Skipping source: {e}")
                continue
        logger.info(f"  Added {num_added} catalog sources")

    elif source_mode == "custom" and catalog_sources is not None:
        # Custom sources provided directly in pixel coordinates
        for src in catalog_sources:
            add_gaussian_source(
                data,
                src.get("x_px", src.get("ra_deg", 0)),  # Allow either pixel or sky coords
                src.get("y_px", src.get("dec_deg", 0)),
                src["flux_jy"],
                beam_fwhm_pix,
            )
        logger.info(f"  Added {len(catalog_sources)} custom sources")

    # Add thermal noise
    if add_noise:
        noise_jy = noise_rms_mjy / 1000.0
        noise = rng.normal(0, noise_jy, data.shape).astype(np.float32)
        data += noise
        logger.info(f"  Added Gaussian noise: RMS = {noise_rms_mjy:.3f} mJy/beam")

    # Create FITS header
    header = create_fits_header(
        wcs,
        obs_timestamp,
        ra_deg,
        dec_deg,
        beam_fwhm_arcsec,
        noise_rms_mjy,
        tile_spec,
        source_mode,
    )

    # Add source count to header
    if source_mode == "parametric":
        header["NSOURCES"] = (num_sources, "Number of synthetic sources")
    elif catalog_sources is not None:
        header["NSOURCES"] = (len(catalog_sources), "Number of catalog sources")

    # Create HDU and write file
    hdu = fits.PrimaryHDU(data=data, header=header)
    hdu.writeto(output_path, overwrite=True)

    logger.info(f"  Created FITS image: {output_path}")
    logger.info("  Image statistics:")
    logger.info(f"    Min: {data.min():.6f} Jy/beam")
    logger.info(f"    Max: {data.max():.6f} Jy/beam")
    logger.info(f"    Mean: {data.mean():.6f} Jy/beam")
    logger.info(f"    Std: {data.std():.6f} Jy/beam ({data.std() * 1000:.3f} mJy/beam)")

    return Path(output_path)
