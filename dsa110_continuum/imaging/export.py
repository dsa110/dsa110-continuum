"""Export CASA images to FITS and PNG formats."""

from __future__ import annotations

import os
from collections.abc import Iterable
from glob import glob


def _find_casa_images(source: str, prefix: str) -> list[str]:
    """Find CASA image directories matching prefix.

    Parameters
    ----------
    """
    patt = os.path.join(source, prefix + ".*")
    paths = sorted(glob(patt))
    return [p for p in paths if os.path.isdir(p)]


def export_fits(
    images: Iterable[str], register_metadata: bool = True, ms_path: str = "unknown"
) -> list[str]:
    """Export CASA images to FITS format and optionally register metadata.

    Parameters
    ----------
    images :
        Iterable of CASA image paths
    register_metadata :
        If True, register FITS metadata in database
    ms_path :
        Path to parent MS (for metadata registration)
    images: Iterable[str] :

    Returns
    -------
        List of exported FITS file paths

    """
    from dsa110_contimg.core.calibration.casa_service import CASAService

    service = CASAService()

    exported: list[str] = []
    for p in images:
        fits_out = p + ".fits"
        try:
            service.exportfits(imagename=p, fitsimage=fits_out, overwrite=True)
            print("Exported FITS:", fits_out)
            exported.append(fits_out)

            # Register metadata if requested
            if register_metadata:
                try:
                    from dsa110_contimg.infrastructure.database.register_products import (
                        register_image_with_metadata,
                    )

                    image_id = register_image_with_metadata(fits_out, ms_path=ms_path)
                    if image_id:
                        print(f"Registered metadata: image_id={image_id}")
                except Exception as reg_error:
                    print(
                        f"Metadata registration failed for {fits_out}: {reg_error}",
                        file=__import__("sys").stderr,
                    )
        except Exception as e:
            print("exportfits failed for", p, ":", e, file=__import__("sys").stderr)
    return exported


def save_png_from_fits(paths: Iterable[str]) -> list[str]:
    """Convert FITS files to PNG quicklook images.

    Parameters
    ----------
    paths: Iterable[str] :


    """
    saved: list[str] = []
    try:
        import matplotlib
        import numpy as np
        from astropy.io import fits
        from astropy.visualization import (
            AsinhStretch,
            ImageNormalize,
            ZScaleInterval,
        )

        from dsa110_contimg.common.utils.runtime_safeguards import validate_image_shape

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print("PNG conversion dependencies missing:", e, file=__import__("sys").stderr)
        return saved

    for f in paths:
        try:
            # Use memmap=True for large files to avoid loading everything into memory
            with fits.open(f, memmap=True) as hdul:
                data = None
                for hdu in hdul:
                    if getattr(hdu, "data", None) is not None and getattr(hdu.data, "ndim", 0) >= 2:
                        # Validate image shape before processing
                        try:
                            validate_image_shape(hdu.data, min_size=1)
                        except ValueError as e:
                            import logging

                            logging.warning(f"Skipping invalid image in {f}: {e}")
                            continue
                        data = hdu.data
                        break
                if data is None:
                    print("Skip (no 2D image in FITS):", f)
                    continue
                arr = np.array(data, dtype=float)
                while arr.ndim > 2:
                    arr = arr[0]
                m = np.isfinite(arr)
                if not np.any(m):
                    print("Skip (all NaN):", f)
                    continue

                # Downsample large arrays to speed up processing
                # For arrays > 10M pixels, downsample by factor of 4-16
                n_pixels = arr.size
                if n_pixels > 10_000_000:
                    # Calculate downsampling factor to get ~1-5M pixels
                    factor = max(2, int(np.sqrt(n_pixels / 2_000_000)))
                    # Use simple block averaging for downsampling
                    h, w = arr.shape
                    h_new, w_new = h // factor, w // factor
                    if h_new > 0 and w_new > 0:
                        arr_downsampled = (
                            arr[: h_new * factor, : w_new * factor]
                            .reshape(h_new, factor, w_new, factor)
                            .mean(axis=(1, 3))
                        )
                        arr = arr_downsampled
                        m = np.isfinite(arr)
                        print(f"Downsampled by factor {factor} for faster processing")

                # Use ZScale normalization (better for astronomical images)
                # This matches VAST Tools approach and handles outliers better
                # than percentile-based methods
                try:
                    # Create ZScale normalization with asinh stretch
                    # This is the standard approach for astronomical images
                    # Use finite values for interval calculation, but apply to full array
                    normalize = ImageNormalize(
                        arr[m],  # Use finite values for interval calculation
                        interval=ZScaleInterval(contrast=0.2),
                        stretch=AsinhStretch(),
                    )

                    # Set NaN values to 0 for display (they'll be outside the colormap range)
                    arr_display = arr.copy()
                    arr_display[~m] = 0

                    plt.figure(figsize=(6, 5), dpi=140)
                    im = plt.imshow(
                        arr_display,
                        origin="lower",
                        cmap="inferno",
                        interpolation="nearest",
                        norm=normalize,
                    )
                    plt.colorbar(im, fraction=0.046, pad=0.04, label="Flux (Jy/beam)")
                    plt.title(os.path.basename(f))
                    plt.tight_layout()
                    out = f + ".png"
                    plt.savefig(out, bbox_inches="tight")
                    plt.close()
                    print("Wrote PNG:", out)
                    saved.append(out)
                except Exception as norm_error:
                    # Fallback to simple percentile normalization if ZScale fails
                    import logging

                    logging.warning(
                        f"ZScale normalization failed, using percentile fallback: {norm_error}"
                    )
                    vals = arr[m]
                    lo, hi = np.percentile(
                        vals, [1.0, 99.9]
                    )  # Standardized to 99.9 (matches VAST Tools)
                    img = np.clip(arr, lo, hi)
                    img = np.arcsinh((img - lo) / max(1e-12, (hi - lo)))
                    img[~m] = np.nan
                    plt.figure(figsize=(6, 5), dpi=140)
                    plt.imshow(img, origin="lower", cmap="inferno", interpolation="nearest")
                    plt.colorbar(fraction=0.046, pad=0.04)
                    plt.title(os.path.basename(f))
                    plt.tight_layout()
                    out = f + ".png"
                    plt.savefig(out, bbox_inches="tight")
                    plt.close()
                    print("Wrote PNG:", out)
                    saved.append(out)
        except Exception as e:
            print("PNG conversion failed for", f, ":", e, file=__import__("sys").stderr)
    return saved
