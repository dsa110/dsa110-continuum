"""Masking utilities for imaging.

Adapted from dstools/mask.py and dstools/imaging.py for DSA-110 pipeline.
Designed to operate within the casa6 environment (relies on astropy/scipy,
avoids radio-beam).
"""

import logging
import shutil
from pathlib import Path
from typing import NamedTuple

import numpy as np
from astropy.convolution import Gaussian2DKernel
from astropy.io import fits
from astropy.stats import gaussian_fwhm_to_sigma
from scipy.ndimage import binary_erosion, minimum_filter
from scipy.signal import fftconvolve

LOG = logging.getLogger(__name__)


class SkewResult(NamedTuple):
    positive_pixel_frac: np.ndarray
    """The fraction of positive pixels in a boxcar function"""
    skew_mask: np.ndarray
    """Mask of pixel positions indicating which positions failed the skew test"""
    box_size: int
    """Size of the boxcar window applies"""
    skew_delta: float
    """The test threshold for skew"""


def create_boxcar_skew_mask(
    image: np.ndarray,
    skew_delta: float,
    box_size: int,
) -> SkewResult:
    assert 0.0 < skew_delta < 0.5, f"{skew_delta=}, but should be 0.0 to 0.5"
    assert len(image.shape) == 2, f"Expected two dimensions, got image shape of {image.shape}"
    LOG.debug("Computing boxcar skew with box_size=%s and skew_delta=%s", box_size, skew_delta)
    positive_pixels = (image > 0.0).astype(np.float32)

    # Counting positive pixel fraction here.
    window_shape = (box_size, box_size)
    positive_pixel_fraction = fftconvolve(
        in1=positive_pixels,
        in2=np.ones(window_shape, dtype=np.float32),
        mode="same",
    ) / np.prod(window_shape)
    positive_pixel_fraction = np.clip(
        positive_pixel_fraction,
        0.0,
        1.0,
    )  # trust nothing

    skew_mask = positive_pixel_fraction > (0.5 + skew_delta)
    LOG.debug(
        "%s pixels above skew_delta=%s with box_size=%s", np.sum(skew_mask), skew_delta, box_size
    )

    return SkewResult(
        positive_pixel_frac=positive_pixel_fraction,
        skew_mask=skew_mask,
        skew_delta=skew_delta,
        box_size=box_size,
    )


def _minimum_absolute_clip(
    image: np.ndarray,
    increase_factor: float = 2.0,
    box_size: int = 100,
) -> np.ndarray:
    """Given an input image or signal array, construct a simple image mask by
    applying a rolling boxcar minimum filter, and then selecting pixels above a
    cut of the absolute value value scaled by `increase_factor`. This is a
    pixel-wise operation.

    Parameters
    ----------
    image: np.ndarray :

    """
    LOG.debug("Minimum absolute clip, increase_factor=%s box_size=%s", increase_factor, box_size)
    rolling_box_min = minimum_filter(image, box_size)

    image_mask = image > (increase_factor * np.abs(rolling_box_min))

    return image_mask


def _adaptive_minimum_absolute_clip(
    image: np.ndarray,
    increase_factor: float = 2.0,
    box_size: int = 100,
    adaptive_max_depth: int = 3,
    adaptive_box_step: float = 2.0,
    adaptive_skew_delta: float = 0.2,
) -> np.ndarray:
    LOG.debug(
        "Using adaptive minimum absolute clip with box_size=%s adaptive_skew_delta=%s",
        box_size,
        adaptive_skew_delta,
    )
    min_value = minimum_filter(image, size=box_size)

    for box_round in range(adaptive_max_depth, 0, -1):
        skew_results = create_boxcar_skew_mask(
            image=image,
            skew_delta=adaptive_skew_delta,
            box_size=box_size,
        )
        if np.all(~skew_results.skew_mask):
            LOG.info("No skewed islands detected")
            break
        if any([box_size > dim for dim in image.shape]):
            LOG.info("box_size=%s larger than a dimension in image.shape=%s", box_size, image.shape)
            break

        LOG.debug(
            "(%s) Growing box_size=%s adaptive_box_step=%s", box_round, box_size, adaptive_box_step
        )
        box_size = int(box_size * adaptive_box_step)
        minval = minimum_filter(image, box_size)
        LOG.debug("Slicing minimum values into place")

        min_value[skew_results.skew_mask] = minval[skew_results.skew_mask]

    mask = image > (np.abs(min_value) * increase_factor)

    return mask


def minimum_absolute_clip(
    image: np.ndarray,
    increase_factor: float = 2.0,
    box_size: int = 100,
    adaptive_max_depth: int | None = None,
    adaptive_box_step: float = 2.0,
    adaptive_skew_delta: float = 0.2,
) -> np.ndarray:
    """Adaptive minimum absolute clip (author: Tim Galvin).

    Implements minimum absolute clip method. A minimum filter of a particular
    boxc size is applied to the input image. The absolute of the output is taken
    and increased by a guard factor, which forms the clipping level used to
    construct a clean mask.

    Parameters
    ----------
    image: np.ndarray :

    """
    if adaptive_max_depth is None:
        return _minimum_absolute_clip(
            image=image,
            box_size=box_size,
            increase_factor=increase_factor,
        )

    adaptive_max_depth = int(adaptive_max_depth)

    return _adaptive_minimum_absolute_clip(
        image=image,
        increase_factor=increase_factor,
        box_size=box_size,
        adaptive_max_depth=adaptive_max_depth,
        adaptive_box_step=adaptive_box_step,
        adaptive_skew_delta=adaptive_skew_delta,
    )


def create_beam_mask_kernel(
    fits_header: fits.Header,
    kernel_size=100,
    minimum_response: float = 0.6,
) -> np.ndarray:
    """Make a mask using the shape of a beam in a FITS Header object.

    Uses BMAJ, BMIN, BPA from header to generate a Gaussian kernel using Astropy.

    Parameters
    ----------
    fits_header: fits.Header :

    kernel_size :
         (Default value = 100)
    """
    assert 0.0 < minimum_response < 1.0, (
        f"{minimum_response=}, should be between 0 to 1 (exclusive)"
    )

    if not all(key in fits_header for key in ["BMAJ", "BMIN", "BPA"]):
        raise KeyError("BMAJ, BMIN, BPA must be present in FITS header")

    if "CDELT1" in fits_header:
        pixel_scale = abs(fits_header["CDELT1"])
    elif "CD1_1" in fits_header:
        pixel_scale = abs(fits_header["CD1_1"])
    else:
        raise KeyError("Pixel scale (CDELT1 or CD1_1) missing from FITS header")

    # Beam parameters in degrees
    bmaj = fits_header["BMAJ"]
    bmin = fits_header["BMIN"]
    bpa = fits_header["BPA"]

    # Convert to pixels (sigma)
    # FWHM to Sigma: FWHM = 2.355 * sigma
    sigma_major = (bmaj / pixel_scale) * gaussian_fwhm_to_sigma
    sigma_minor = (bmin / pixel_scale) * gaussian_fwhm_to_sigma
    theta = np.radians(bpa)

    # Create 2D Gaussian Kernel
    kernel = Gaussian2DKernel(
        x_stddev=sigma_major,
        y_stddev=sigma_minor,
        theta=theta,
        x_size=kernel_size,
        y_size=kernel_size,
    )

    # Normalize kernel to peak at 1.0 for thresholding
    kernel_array = kernel.array / kernel.array.max()

    return kernel_array > minimum_response


def beam_shape_erode(
    mask: np.ndarray,
    fits_header: fits.Header,
    minimum_response: float = 0.6,
) -> np.ndarray:
    """Construct a kernel representing the shape of the restoring beam at
    a particular level, and use it as the basis of a binary erosion of the
    input mask.

    Parameters
    ----------
    mask: np.ndarray :

    fits_header: fits.Header :

    """
    if not all([key in fits_header for key in ["BMAJ", "BMIN", "BPA"]]):
        LOG.warning("Beam parameters missing. Not performing the beam shape erosion. ")
        return mask

    LOG.debug("Eroding the mask using the beam shape with minimum_response=%s", minimum_response)

    try:
        beam_mask_kernel = create_beam_mask_kernel(
            fits_header=fits_header,
            minimum_response=minimum_response,
        )

        # This handles any unsqueezed dimensions
        beam_mask_kernel = beam_mask_kernel.reshape(mask.shape[:-2] + beam_mask_kernel.shape)

        erode_mask = binary_erosion(
            input=mask,
            iterations=1,
            structure=beam_mask_kernel,
        )

        return erode_mask.astype(mask.dtype)
    except Exception as e:
        LOG.warning("Failed to create beam mask kernel: %s. Skipping erosion.", e)
        return mask


def prepare_cleaning_mask(
    fits_mask: str | None,
    target_mask: str | None = None,
    galvin_clip_mask: str | None = None,
    galvin_box_size: int = 100,
    galvin_adaptive_depth: int = 3,
    erode_beam_shape: bool = False,
    work_dir: str | None = None,
) -> str | None:
    """Prepare a cleaning mask by combining optional target mask, adaptive clip
    mask, and beam erosion.

    Parameters
    ----------
    fits_mask :
        Path to input FITS mask.
    target_mask :
        Optional path to mask to intersect with (AND).
    galvin_clip_mask :
        Optional path to image for adaptive clipping
        (minimum_absolute_clip).
    galvin_box_size :
        Box size for Galvin adaptive clip (default 100 pixels).
    galvin_adaptive_depth :
        Max iterations for adaptive subdivision (default 3).
    erode_beam_shape :
        Whether to erode mask by beam shape.
    work_dir :
        Optional directory to write the combined mask to. If None,
        updates fits_mask in place.
    fits_mask: Optional[str] :

    target_mask: Optional[str] :
         (Default value = None)
    galvin_clip_mask: Optional[str] :
         (Default value = None)

    Returns
    -------
        Path to the final prepared mask.

    """
    if fits_mask is None:
        return None

    # Use str conversion for Path compatibility
    mask_path = Path(fits_mask).absolute()
    if not mask_path.exists():
        LOG.warning("Mask file not found: %s", mask_path)
        return None

    # If work_dir provided, copy mask there first to avoid modifying original
    if work_dir:
        work_path = Path(work_dir)
        work_path.mkdir(exist_ok=True, parents=True)
        new_mask_path = work_path / mask_path.name
        if new_mask_path != mask_path:
            shutil.copy2(mask_path, new_mask_path)
            mask_path = new_mask_path

    try:
        # Load mask
        with fits.open(mask_path) as hdul:
            header = hdul[0].header
            mask_data = hdul[0].data
            # Handle dimensions
            if mask_data.ndim == 4:
                mask_array = mask_data[0, 0, :, :]
            elif mask_data.ndim == 3:
                mask_array = mask_data[0, :, :]
            else:
                mask_array = mask_data

        # Apply Galvin clip if requested
        if galvin_clip_mask is not None:
            clip_path = Path(galvin_clip_mask).absolute()
            if clip_path.exists():
                with fits.open(clip_path) as hdul_clip:
                    clip_data = hdul_clip[0].data
                    if clip_data.ndim == 4:
                        clip_image = clip_data[0, 0, :, :]
                    elif clip_data.ndim == 3:
                        clip_image = clip_data[0, :, :]
                    else:
                        clip_image = clip_data

                clip_mask_array = minimum_absolute_clip(
                    clip_image,
                    box_size=galvin_box_size,
                    adaptive_max_depth=galvin_adaptive_depth,
                )
                LOG.info(
                    "Applied Galvin adaptive clip from %s (box_size=%d, depth=%d)",
                    clip_path,
                    galvin_box_size,
                    galvin_adaptive_depth,
                )
                mask_array = clip_mask_array
            else:
                LOG.warning("Galvin clip mask file not found: %s", clip_path)

        # Erode the beam shape
        if erode_beam_shape:
            mask_array = beam_shape_erode(
                mask=mask_array,
                fits_header=header,
            )

        # Remove user-specified region from mask by selecting pixels
        # that are in mask_array but not in target_mask (Intersection)
        if target_mask is not None:
            target_path = Path(target_mask).absolute()
            if target_path.exists():
                with fits.open(target_path) as hdul_target:
                    target_data = hdul_target[0].data
                    if target_data.ndim == 4:
                        target_array = target_data[0, 0, :, :]
                    elif target_data.ndim == 3:
                        target_array = target_data[0, :, :]
                    else:
                        target_array = target_data

                # Ensure shapes match
                if target_array.shape == mask_array.shape:
                    mask_array = np.logical_and(mask_array, target_array)
                else:
                    LOG.warning(
                        "Target mask shape %s mismatch with mask %s",
                        target_array.shape,
                        mask_array.shape,
                    )
            else:
                LOG.warning("Target mask file not found: %s", target_path)

        # Save updated mask
        with fits.open(mask_path, mode="update") as hdul:
            # Update data while preserving dimensions
            if hdul[0].data.ndim == 4:
                hdul[0].data[0, 0, :, :] = mask_array.astype(hdul[0].data.dtype)
            elif hdul[0].data.ndim == 3:
                hdul[0].data[0, :, :] = mask_array.astype(hdul[0].data.dtype)
            else:
                hdul[0].data = mask_array.astype(hdul[0].data.dtype)

            hdul.flush()

        return str(mask_path)

    except Exception as e:
        LOG.error("Failed to prepare cleaning mask: %s", e)
        import traceback

        LOG.debug(traceback.format_exc())
        return None
