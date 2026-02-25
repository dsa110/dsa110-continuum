"""
Core mosaic building function using WSClean-style linear mosaicking.

This module implements proper radio astronomy mosaicking using:
1. Common pixel grid (no interpolation)
2. Primary beam weighting
3. Linear combination of images

NO reproject. NO interpolation. Just proper radio astronomy.

**Tier Routing Guide** (see also ``core/mosaic/README.md``):

- **QUICKLOOK tier**: Use ``build_mosaic()`` (this module). Fast image-domain
  combination from existing FITS images. No MS access required. Best for
  real-time monitoring and post-hoc combination of archived images.

- **SCIENCE / DEEP tiers**: Prefer ``build_wsclean_mosaic()`` from
  ``core/mosaic/wsclean_mosaic.py``. Visibility-domain joint deconvolution
  produces scientifically superior results for wide-field imaging. Requires
  Measurement Set files.

The two approaches are *complementary*, not duplicates — they operate at
different stages of the pipeline on different data products.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.stats import mad_std
from scipy.ndimage import map_coordinates

from dsa110_contimg.common.utils.decorators import timed

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)


@dataclass
class MosaicResult:
    """Result of mosaic build operation."""

    output_path: Path
    n_images: int
    median_rms: float
    coverage_sq_deg: float
    weight_map_path: Path | None = None
    effective_noise_jy: float | None = None
    external_weights_used: bool = False


@timed("mosaic.build_mosaic")
def build_mosaic(
    image_paths: list[Path],
    output_path: Path,
    alignment_order: int = 3,  # noqa: ARG001 - Now unused, kept for API compatibility
    timeout_minutes: int = 30,  # noqa: ARG001 - Reserved for future async support
    write_weight_map: bool = True,
    apply_pb_correction: bool = False,
    weight_image_paths: list[Path] | None = None,
    rescale_weights: bool = True,
) -> MosaicResult:
    """Build mosaic from list of FITS images using linear mosaicking.

    This implementation uses WSClean-style linear mosaicking:
    - No interpolative reprojection (radio astronomy best practice)
    - Primary beam weighted combination
    - Images must be on compatible pixel grids

    .. note:: **Tier: QUICKLOOK**

        This function performs image-domain linear mosaicking on already-deconvolved
        FITS images. It is the recommended approach for the QUICKLOOK tier where speed
        is critical and MS files may not be available.

        For SCIENCE and DEEP tiers, prefer :func:`~dsa110_contimg.core.mosaic.wsclean_mosaic.build_wsclean_mosaic`
        which performs visibility-domain joint deconvolution for better wide-field imaging.

    Parameters
    ----------
    image_paths : list[Path]
        List of input FITS files.
    output_path : Path
        Where to write output mosaic.
    alignment_order : int, optional
        DEPRECATED. Kept for API compatibility. Linear mosaicking does not interpolate.
        Default is 3.
    timeout_minutes : int, optional
        Maximum execution time (for future async support).
        Default is 30.
    write_weight_map : bool, optional
        If True, write a weight map for uncertainty estimation.
        Default is True.
    apply_pb_correction : bool, optional
        If True, apply primary beam correction using DSA-110
        Airy disk model (4.7m dish). This divides each pixel by the primary
        beam response to correct for attenuation away from the phase center.
        The correction is limited at PB < 0.1 to avoid amplifying edge noise.
        Default is False.
    weight_image_paths : list[Path] or None, optional
        Optional list of external weight map FITS files,
        one per input image. If provided, these are used instead of computing
        weights from RMS.
        Default is None.
    rescale_weights : bool, optional
        If True and using external weights, normalize them to
        have consistent scaling.
        Default is True.

    Returns
    -------
    MosaicResult
        MosaicResult with metadata.

    Raises
    ------
    ValueError
        If no images provided or images are invalid.
    FileNotFoundError
        If input files don't exist.

    Examples
    --------
    >>> result = build_mosaic(
    ...     image_paths=[Path("img1.fits"), Path("img2.fits")],
    ...     output_path=Path("mosaic.fits"),
    ... )
    >>> print(f"Created mosaic with {result.n_images} images")
    """
    if not image_paths:
        raise ValueError("No images provided for mosaicking")

    logger.info(f"Building mosaic from {len(image_paths)} images using linear mosaicking")

    # Validate input files exist
    for path in image_paths:
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")

    # Read input images and WCS
    # Squeeze degenerate axes (freq, Stokes) for 2D mosaicking
    hdus = []
    for path in image_paths:
        with fits.open(str(path)) as hdulist:
            data = hdulist[0].data.copy()
            header = hdulist[0].header.copy()

            # Squeeze degenerate axes (e.g., 4D CASA images to 2D)
            while data.ndim > 2:
                if data.shape[0] == 1:
                    data = data[0]
                else:
                    break

            # Create 2D WCS from potentially 4D header
            wcs_2d = WCS(header, naxis=2)
            header_2d = wcs_2d.to_header()
            # Preserve key metadata
            for key in [
                "BUNIT",
                "BMAJ",
                "BMIN",
                "BPA",
                "TELESCOP",
                "DATE-OBS",
                "RESTFRQ",
                "OBJECT",
            ]:
                if key in header:
                    header_2d[key] = header[key]

            hdu = fits.PrimaryHDU(data=data, header=header_2d)
            hdus.append(hdu)

    logger.debug("Loaded %d FITS images", len(hdus))

    # Compute optimal output WCS (covers all inputs)
    output_wcs, output_shape = compute_optimal_wcs(hdus)

    logger.debug(
        f"Output shape: {output_shape}, WCS center: "
        f"({output_wcs.wcs.crval[0]:.4f}, {output_wcs.wcs.crval[1]:.4f})"
    )

    # Regrid all images to common grid using nearest-neighbor (no interpolation)
    # This is the WSClean-style approach
    arrays = []
    footprints = []

    for i, hdu in enumerate(hdus):
        logger.debug(f"Regridding image {i + 1}/{len(hdus)} (nearest-neighbor, no interpolation)")
        array, footprint = regrid_to_common_grid(
            hdu.data,
            WCS(hdu.header, naxis=2),
            output_wcs,
            output_shape,
        )
        arrays.append(array)
        footprints.append(footprint)

    # Compute weights - either from external maps or from RMS (inverse-variance)
    external_weights_used = False
    if weight_image_paths is not None:
        if len(weight_image_paths) != len(image_paths):
            raise ValueError(
                f"Number of weight images ({len(weight_image_paths)}) must match "
                f"number of input images ({len(image_paths)})"
            )
        logger.info("Using external weight maps")
        weights = compute_weights_from_maps(
            weight_image_paths,
            output_wcs,
            output_shape,
            rescale=rescale_weights,
        )
        external_weights_used = True
    else:
        # Default: compute inverse-variance weights from image RMS
        weights = compute_weights(hdus)

    # Compute primary beam weights for each image
    # This is the key for radio astronomy mosaicking
    pb_weights = []
    for i, hdu in enumerate(hdus):
        # Get frequency from header
        freq_hz = 1.4e9
        if "CRVAL3" in hdu.header:
            freq_hz = float(hdu.header["CRVAL3"])
        elif "RESTFRQ" in hdu.header:
            freq_hz = float(hdu.header["RESTFRQ"])

        # Get phase center from WCS
        wcs = WCS(hdu.header, naxis=2)

        # Compute primary beam weight map for this image
        pb_weight = compute_pb_weight_map(
            output_wcs,
            output_shape,
            wcs.wcs.crval[0],  # Phase center RA
            wcs.wcs.crval[1],  # Phase center Dec
            freq_hz=freq_hz,
            dish_dia_m=4.7,  # DSA-110 dish diameter
        )
        pb_weights.append(pb_weight)

    # Combine with primary beam weighted linear combination
    combined, weight_map = linear_mosaic_combine(
        arrays, weights, footprints, pb_weights, return_weights=True
    )
    combined_footprint = np.sum(footprints, axis=0) > 0

    # Apply primary beam correction if requested
    if apply_pb_correction:
        logger.info("Applying primary beam correction to final mosaic")
        # Get frequency from first image header (default to 1.4 GHz if not found)
        freq_hz = 1.4e9
        if "CRVAL3" in hdus[0].header:
            freq_hz = float(hdus[0].header["CRVAL3"])
        elif "RESTFRQ" in hdus[0].header:
            freq_hz = float(hdus[0].header["RESTFRQ"])

        pb_correction = compute_pb_correction_map(
            output_wcs,
            output_shape,
            freq_hz=freq_hz,
            dish_dia_m=4.7,  # DSA-110 dish diameter
            pb_cutoff=0.1,
        )
        combined = combined * pb_correction
        logger.info(f"Applied PB correction with freq={freq_hz / 1e9:.3f} GHz")

    # Compute statistics
    rms_values = [compute_rms(arr) for arr in arrays]
    median_rms = float(np.median(rms_values))

    # Compute effective noise from weight map (propagated uncertainty)
    with np.errstate(invalid="ignore", divide="ignore"):
        effective_noise_map = np.where(weight_map > 0, 1.0 / np.sqrt(weight_map), np.nan)
    effective_noise_jy = float(np.nanmedian(effective_noise_map[combined_footprint]))

    # Handle pixel scale as Quantity or plain value
    pixel_scale_raw = output_wcs.proj_plane_pixel_scales()[0]
    if hasattr(pixel_scale_raw, "value"):
        pixel_scale = float(pixel_scale_raw.value)
    else:
        pixel_scale = float(pixel_scale_raw)
    coverage_sq_deg = float(np.sum(combined_footprint) * pixel_scale**2)

    logger.info(
        f"Mosaic stats: median_rms={median_rms:.6f} Jy, "
        f"effective_noise={effective_noise_jy:.6f} Jy, "
        f"coverage={coverage_sq_deg:.4f} sq deg"
    )

    # Write output FITS
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_header = output_wcs.to_header()
    output_header["NIMAGES"] = (len(image_paths), "Number of images combined")
    output_header["MEDRMS"] = (median_rms, "Median RMS noise (Jy)")
    output_header["EFFNOISE"] = (effective_noise_jy, "Effective noise from weights (Jy)")
    output_header["COVERAGE"] = (coverage_sq_deg, "Sky coverage (sq deg)")
    output_header["BUNIT"] = "Jy/beam"
    output_header["MOSAIC"] = ("LINEAR", "Linear mosaicking (no interpolation)")
    if external_weights_used:
        output_header["EXTWEIGH"] = (True, "External weight maps used")

    output_hdu = fits.PrimaryHDU(data=combined, header=output_header)
    output_hdu.writeto(str(output_path), overwrite=True)

    logger.info(f"Wrote mosaic to {output_path}")

    # Optionally write weight map for uncertainty propagation
    weight_map_path = None
    if write_weight_map:
        weight_map_path = output_path.with_suffix(".weights.fits")
        weight_header = output_wcs.to_header()
        weight_header["BUNIT"] = "1/Jy^2"
        weight_header["COMMENT"] = "Inverse-variance weight map for uncertainty estimation"
        weight_header["COMMENT"] = "Noise = 1/sqrt(weight) at each pixel"
        weight_hdu = fits.PrimaryHDU(data=weight_map.astype(np.float32), header=weight_header)
        weight_hdu.writeto(str(weight_map_path), overwrite=True)
        logger.info(f"Wrote weight map to {weight_map_path}")

    return MosaicResult(
        output_path=output_path,
        n_images=len(image_paths),
        median_rms=median_rms,
        coverage_sq_deg=coverage_sq_deg,
        weight_map_path=weight_map_path,
        effective_noise_jy=effective_noise_jy,
        external_weights_used=external_weights_used,
    )


def regrid_to_common_grid(
    data: NDArray,
    input_wcs: WCS,
    output_wcs: WCS,
    output_shape: tuple[int, int],
) -> tuple[NDArray, NDArray]:
    """Regrid image to common grid using nearest-neighbor (no interpolation).

    This is the WSClean-style approach - proper for radio astronomy where
    pixel values should not be interpolated.

    Parameters
    ----------
    data : NDArray
        Input image data (2D).
    input_wcs : WCS
        Input image WCS.
    output_wcs : WCS
        Target output WCS.
    output_shape : tuple[int, int]
        Target output shape (ny, nx).

    Returns
    -------
    tuple[NDArray, NDArray]
        (regridded_array, footprint) where footprint marks valid pixels.
    """
    ny, nx = output_shape

    # Create output pixel grid
    yy, xx = np.mgrid[0:ny, 0:nx]

    # Convert output pixels to world coordinates
    world = output_wcs.pixel_to_world(xx, yy)

    # Convert world coordinates to input pixel coordinates
    if hasattr(world, "ra"):
        # Single SkyCoord array
        input_pixels = input_wcs.world_to_pixel(world)
    else:
        # Handle list of coords
        input_pixels = input_wcs.world_to_pixel(world)

    # Get pixel coordinates
    if isinstance(input_pixels, tuple):
        input_x, input_y = input_pixels
    else:
        input_x = input_pixels[0]
        input_y = input_pixels[1]

    # Create footprint - mark pixels that fall within input image
    input_ny, input_nx = data.shape
    valid = (
        (input_x >= 0)
        & (input_x < input_nx)
        & (input_y >= 0)
        & (input_y < input_ny)
        & np.isfinite(input_x)
        & np.isfinite(input_y)
    )

    # Initialize output with NaNs
    output = np.full(output_shape, np.nan, dtype=np.float32)

    # Use nearest-neighbor regridding (no interpolation!)
    # This is the key difference from reproject
    valid_mask = valid.ravel()
    if np.any(valid_mask):
        coords = np.array([input_y.ravel()[valid_mask], input_x.ravel()[valid_mask]])
        # order=0 means nearest-neighbor (no interpolation)
        values = map_coordinates(data, coords, order=0, mode="constant", cval=np.nan)
        output.ravel()[valid_mask] = values

    footprint = valid.astype(np.float32)

    return output, footprint


def compute_optimal_wcs(hdus: list[fits.PrimaryHDU]) -> tuple[WCS, tuple[int, int]]:
    """Compute WCS that covers all input images.

    Parameters
    ----------
    hdus : list[fits.PrimaryHDU]
        List of FITS HDUs with valid WCS.

    Returns
    -------
    tuple[WCS, tuple[int, int]]
        Tuple of (output_wcs, (ny, nx)) shape.
    """
    # Find min/max RA/Dec across all images
    all_ra = []
    all_dec = []
    pixel_scales = []

    for hdu in hdus:
        wcs = WCS(hdu.header, naxis=2)
        ny, nx = hdu.data.shape[-2:]

        # Get corner coordinates
        corners_x = [0, nx - 1, nx - 1, 0]
        corners_y = [0, 0, ny - 1, ny - 1]

        coords = wcs.pixel_to_world(corners_x, corners_y)
        all_ra.extend([c.ra.deg for c in coords])
        all_dec.extend([c.dec.deg for c in coords])

        # Track pixel scales (convert from Quantity to float if needed)
        scales = wcs.proj_plane_pixel_scales()
        if hasattr(scales[0], "value"):
            pixel_scales.append(float(np.mean([s.value for s in scales])))
        else:
            pixel_scales.append(float(np.mean(scales)))

    # Compute bounding box
    ra_min, ra_max = min(all_ra), max(all_ra)
    dec_min, dec_max = min(all_dec), max(all_dec)

    # Handle RA wrap-around near 0/360
    if ra_max - ra_min > 180:
        ra_pos = [r for r in all_ra if r > 180]
        ra_neg = [r for r in all_ra if r <= 180]
        ra_min = min(ra_pos) if ra_pos else 0
        ra_max = max(ra_neg) + 360 if ra_neg else 360

    # Use median pixel scale
    pixel_scale = np.median(pixel_scales)

    # Compute output grid size
    ra_span = ra_max - ra_min
    dec_span = dec_max - dec_min

    nx = int(np.ceil(ra_span / pixel_scale)) + 10  # Add margin
    ny = int(np.ceil(dec_span / pixel_scale)) + 10

    # Limit size to prevent memory issues
    max_size = 8192
    if nx > max_size or ny > max_size:
        scale_factor = max(nx, ny) / max_size
        nx = int(nx / scale_factor)
        ny = int(ny / scale_factor)
        pixel_scale *= scale_factor

    # Create output WCS
    output_wcs = WCS(naxis=2)
    output_wcs.wcs.crpix = [nx / 2, ny / 2]
    output_wcs.wcs.crval = [(ra_min + ra_max) / 2, (dec_min + dec_max) / 2]
    output_wcs.wcs.cdelt = [-pixel_scale, pixel_scale]  # RA increases left
    output_wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    output_wcs.array_shape = (ny, nx)

    return output_wcs, (ny, nx)


def compute_weights(hdus: list[fits.PrimaryHDU]) -> NDArray[np.floating]:
    """Compute inverse-variance weights for images.

    Parameters
    ----------
    hdus : list[fits.PrimaryHDU]
        List of FITS HDUs.

    Returns
    -------
    NDArray[np.floating]
        Array of weights (normalized to sum to 1).
    """
    weights = []
    for hdu in hdus:
        rms = compute_rms(hdu.data)
        # Inverse variance weighting
        weight = 1.0 / (rms**2) if rms > 0 else 0.0
        weights.append(weight)

    weights = np.array(weights)

    # Normalize
    total = np.sum(weights)
    if total > 0:
        weights /= total
    else:
        # Equal weights if all RMS are zero
        weights = np.ones(len(hdus)) / len(hdus)

    return weights


def compute_weights_from_maps(
    weight_paths: list[Path],
    output_wcs: WCS,
    output_shape: tuple[int, int],
    rescale: bool = True,
) -> NDArray[np.floating]:
    """Compute per-image weights from external weight map FITS files.

    Parameters
    ----------
    weight_paths : list[Path]
        List of paths to weight map FITS files.
    output_wcs : WCS
        Target WCS for regridding.
    output_shape : tuple[int, int]
        Target image shape (ny, nx).
    rescale : bool, optional
        If True, normalize weights.
        Default is True.

    Returns
    -------
    NDArray[np.floating]
        Array of per-image weights (normalized if rescale=True).
    """
    weights = []
    for weight_path in weight_paths:
        if not weight_path.exists():
            logger.warning("Weight map not found: %s, using weight=0", weight_path)
            weights.append(0.0)
            continue

        with fits.open(str(weight_path)) as hdul:
            weight_data = hdul[0].data
            weight_wcs = WCS(hdul[0].header, naxis=2)

            # Regrid weight map using nearest-neighbor
            weight_reproj, _ = regrid_to_common_grid(
                weight_data,
                weight_wcs,
                output_wcs,
                output_shape,
            )

            # Use median of non-zero weights as overall image weight
            valid_weights = weight_reproj[np.isfinite(weight_reproj) & (weight_reproj > 0)]
            if len(valid_weights) > 0:
                weights.append(float(np.median(valid_weights)))
            else:
                weights.append(0.0)

    weights = np.array(weights)

    if rescale:
        total = np.sum(weights)
        if total > 0:
            weights /= total
        else:
            weights = np.ones(len(weight_paths)) / len(weight_paths)

    logger.debug("External weight map weights: %s", weights)
    return weights


def compute_pb_weight_map(
    wcs: WCS,
    shape: tuple[int, int],
    phase_center_ra: float,
    phase_center_dec: float,
    freq_hz: float = 1.4e9,
    dish_dia_m: float = 4.7,
) -> NDArray:
    """Compute primary beam weight map for an image.

    This creates the primary beam response pattern used for weighting
    in linear mosaicking. Each image is weighted by its PB response.

    Parameters
    ----------
    wcs : WCS
        WCS of the output mosaic.
    shape : tuple[int, int]
        (ny, nx) shape of the output mosaic.
    phase_center_ra : float
        Phase center RA in degrees.
    phase_center_dec : float
        Phase center Dec in degrees.
    freq_hz : float, optional
        Observation frequency in Hz.
        Default is 1.4e9.
    dish_dia_m : float, optional
        Dish diameter in meters.
        Default is 4.7.

    Returns
    -------
    NDArray
        2D array of primary beam weights (0-1).
    """
    ny, nx = shape

    # Convert phase center to radians
    center_ra = np.radians(phase_center_ra)
    center_dec = np.radians(phase_center_dec)

    # Compute wavelength
    c = 299792458.0  # m/s
    wavelength = c / freq_hz

    # Pre-compute factor for Airy disk
    factor = np.pi * dish_dia_m / wavelength

    # Create coordinate grid
    y_idx, x_idx = np.indices((ny, nx))

    # Convert pixels to world coordinates
    coords = wcs.pixel_to_world(x_idx.ravel(), y_idx.ravel())

    # Get RA/Dec arrays
    if hasattr(coords, "ra"):
        ra_deg = coords.ra.deg
        dec_deg = coords.dec.deg
    else:
        ra_deg = np.array([c.ra.deg for c in coords])
        dec_deg = np.array([c.dec.deg for c in coords])

    ra_rad = np.radians(ra_deg)
    dec_rad = np.radians(dec_deg)

    # Compute angular separation using haversine
    delta_dec = dec_rad - center_dec
    delta_ra = ra_rad - center_ra

    a = (
        np.sin(delta_dec / 2) ** 2
        + np.cos(center_dec) * np.cos(dec_rad) * np.sin(delta_ra / 2) ** 2
    )
    theta = 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))

    # Compute Airy disk primary beam response
    from scipy.special import j1

    x = factor * np.sin(theta)

    pb_response = np.ones_like(x)
    nonzero = x > 1e-10
    pb_response[nonzero] = (2 * j1(x[nonzero]) / x[nonzero]) ** 2

    # Reshape to image dimensions
    pb_weight = pb_response.reshape((ny, nx))

    return pb_weight.astype(np.float32)


def linear_mosaic_combine(
    arrays: list[NDArray],
    weights: NDArray[np.floating],
    footprints: list[NDArray],
    pb_weights: list[NDArray],
    return_weights: bool = False,
    mask_zero_weight: bool = True,
) -> NDArray[np.floating] | tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Combine images using linear mosaicking with primary beam weighting.

    This is the WSClean-style approach:
    - Each image weighted by its primary beam response
    - Linear combination (no interpolation)
    - Proper uncertainty propagation

    Parameters
    ----------
    arrays : list[NDArray]
        List of regridded data arrays.
    weights : NDArray[np.floating]
        Per-image RMS weights.
    footprints : list[NDArray]
        Per-image footprints.
    pb_weights : list[NDArray]
        Per-image primary beam weight maps.
    return_weights : bool, optional
        If True, return (combined, weight_map).
        Default is False.
    mask_zero_weight : bool, optional
        If True, mask pixels with zero weight to NaN.
        Default is True.

    Returns
    -------
    NDArray or tuple[NDArray, NDArray]
        Combined mosaic, or (combined, weight_map) if return_weights=True.
    """
    # Stack arrays
    stack = np.array(arrays)
    fp_stack = np.array(footprints)
    pb_stack = np.array(pb_weights)

    # Apply footprint mask
    stack = np.where(fp_stack, stack, np.nan)

    # Combine RMS weights with primary beam weights
    # Final weight = rms_weight * pb_weight^2
    # (PB weight squared because we're weighting variance)
    combined_weights = weights[:, np.newaxis, np.newaxis] * (pb_stack ** 2) * fp_stack

    with np.errstate(invalid="ignore", divide="ignore"):
        sum_weights = np.nansum(combined_weights, axis=0)
        combined = np.nansum(stack * combined_weights, axis=0) / sum_weights

    # Handle zero-weight pixels
    if mask_zero_weight:
        combined = np.where(sum_weights > 0, combined, np.nan)
    else:
        combined = np.nan_to_num(combined, nan=0.0)

    sum_weights = np.nan_to_num(sum_weights, nan=0.0)

    if return_weights:
        return combined, sum_weights
    return combined


def compute_rms(data: NDArray) -> float:
    """Compute RMS noise in image.

    Uses median absolute deviation (MAD) for robust noise estimate.

    Parameters
    ----------
    data : NDArray
        Image data array.

    Returns
    -------
    float
        RMS noise estimate.
    """
    finite_data = data[np.isfinite(data)]
    if len(finite_data) == 0:
        return 0.0

    # REFACTOR: Use astropy.stats.mad_std
    return float(mad_std(finite_data, ignore_nan=True))


def compute_pb_correction_map(
    wcs: WCS,
    shape: tuple[int, int],
    freq_hz: float = 1.4e9,
    dish_dia_m: float = 4.7,
    pb_cutoff: float = 0.1,
) -> NDArray:
    """Compute primary beam correction map for DSA-110.

    Creates a 2D map of primary beam correction factors (1/PB) that can
    be multiplied with the image to correct for primary beam attenuation.

    Parameters
    ----------
    wcs : WCS
        WCS of the output mosaic.
    shape : tuple[int, int]
        (ny, nx) shape of the output mosaic.
    freq_hz : float, optional
        Observation frequency in Hz.
        Default is 1.4e9.
    dish_dia_m : float, optional
        Dish diameter in meters.
        Default is 4.7.
    pb_cutoff : float, optional
        Minimum PB response to apply correction.
        Default is 0.1.

    Returns
    -------
    NDArray
        2D array of primary beam correction factors (1/PB).
    """
    ny, nx = shape

    # Get phase center from WCS
    center_ra_deg = wcs.wcs.crval[0]
    center_dec_deg = wcs.wcs.crval[1]
    center_ra = np.radians(center_ra_deg)
    center_dec = np.radians(center_dec_deg)

    # Compute wavelength
    c = 299792458.0  # m/s
    wavelength = c / freq_hz

    # Pre-compute factor for Airy disk
    factor = np.pi * dish_dia_m / wavelength

    # Create coordinate grid
    y_idx, x_idx = np.indices((ny, nx))

    # Convert pixels to world coordinates
    coords = wcs.pixel_to_world(x_idx.ravel(), y_idx.ravel())

    if hasattr(coords, "ra"):
        ra_deg = coords.ra.deg
        dec_deg = coords.dec.deg
    else:
        ra_deg = np.array([c.ra.deg for c in coords])
        dec_deg = np.array([c.dec.deg for c in coords])

    ra_rad = np.radians(ra_deg)
    dec_rad = np.radians(dec_deg)

    # Compute angular separation using haversine
    delta_dec = dec_rad - center_dec
    delta_ra = ra_rad - center_ra

    a = (
        np.sin(delta_dec / 2) ** 2
        + np.cos(center_dec) * np.cos(dec_rad) * np.sin(delta_ra / 2) ** 2
    )
    theta = 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))

    # Compute Airy disk primary beam response
    from scipy.special import j1

    x = factor * np.sin(theta)

    pb_response = np.ones_like(x)
    nonzero = x > 1e-10
    pb_response[nonzero] = (2 * j1(x[nonzero]) / x[nonzero]) ** 2

    # Apply cutoff to avoid extreme corrections at edges
    pb_response = np.maximum(pb_response, pb_cutoff)

    # Correction factor is 1/PB
    pb_correction = 1.0 / pb_response

    # Reshape to image dimensions
    pb_correction = pb_correction.reshape((ny, nx))

    logger.debug(
        f"PB correction map: min={pb_correction.min():.3f}, "
        f"max={pb_correction.max():.3f}, "
        f"center correction={pb_correction[ny // 2, nx // 2]:.3f}"
    )

    return pb_correction.astype(np.float32)
