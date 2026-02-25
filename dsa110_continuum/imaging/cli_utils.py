"""Utility functions for imaging CLI."""

import logging
from pathlib import Path

import numpy as np

# Ensure CASAPATH is set before importing CASA modules
from dsa110_contimg.common.utils.casa_init import ensure_casa_path

ensure_casa_path()

import casacore.tables as casatables

table = casatables.table  # noqa: N816

from dsa110_contimg.core.imaging.masks import (
    beam_shape_erode,
    minimum_absolute_clip,
)

LOG = logging.getLogger(__name__)

from dsa110_contimg.core.imaging.masks import (  # noqa: E402
    prepare_cleaning_mask,
)


def detect_datacolumn(ms_path: str) -> str:
    """Choose datacolumn for tclean.

    Preference order:
    - Use CORRECTED_DATA if present and contains any non-zero values.
    - Otherwise fall back to DATA.

    **CRITICAL SAFEGUARD**: If CORRECTED_DATA column exists but is unpopulated
    (all zeros), this indicates calibration was attempted but failed. In this case,
    we FAIL rather than silently falling back to DATA to prevent imaging uncalibrated
    data when calibration was expected.

    This avoids the common pitfall where applycal didn't populate
    CORRECTED_DATA (all zeros) and tclean would produce blank images.

    Parameters
    ----------
    """
    try:
        with table(ms_path, readonly=True) as t:
            cols = set(t.colnames())
            if "CORRECTED_DATA" in cols:
                try:
                    total = t.nrows()
                    if total <= 0:
                        # Empty MS - can't determine, but CORRECTED_DATA exists
                        # so calibration was attempted, fail to be safe
                        raise RuntimeError(
                            f"CORRECTED_DATA column exists but MS has zero rows: {ms_path}. "
                            f"Calibration appears to have been attempted but failed. "
                            f"Cannot proceed with imaging."
                        )
                    # Sample up to 8 evenly spaced windows of up to 2048 rows
                    windows = 8
                    block = 2048
                    indices = []
                    for i in range(windows):
                        start_idx = int(i * total / max(1, windows))
                        indices.append(max(0, start_idx - block // 2))

                    found_nonzero = False
                    for start in indices:
                        n = min(block, total - start)
                        if n <= 0:
                            continue
                        cd = t.getcol("CORRECTED_DATA", start, n)
                        flags = t.getcol("FLAG", start, n)
                        # Check unflagged data
                        unflagged = cd[~flags]
                        if len(unflagged) > 0 and np.count_nonzero(np.abs(unflagged) > 1e-10) > 0:
                            found_nonzero = True
                            break

                    if found_nonzero:
                        return "corrected"
                    else:
                        # CORRECTED_DATA exists but is all zeros - calibration failed
                        raise RuntimeError(
                            f"CORRECTED_DATA column exists but appears unpopulated in {ms_path}. "
                            f"Calibration appears to have been attempted but failed (all zeros). "
                            f"Cannot proceed with imaging uncalibrated data. "
                            f"Please verify calibration was applied successfully using: "
                            f"python -m dsa110_contimg.core.calibration.cli apply --ms {ms_path}"
                        )
                except RuntimeError:
                    raise  # Re-raise our errors
                except Exception as e:
                    # Other exceptions - be safe and fail
                    raise RuntimeError(
                        f"Error checking CORRECTED_DATA in {ms_path}: {e}. "
                        f"Cannot determine if calibration was applied. Cannot proceed."
                    ) from e
            # CORRECTED_DATA doesn't exist - calibration never attempted, fall back to DATA
            return "data"
    except RuntimeError:
        raise  # Re-raise our errors
    except Exception as e:
        # Other exceptions - be safe and fail
        raise RuntimeError(
            f"Error accessing MS {ms_path}: {e}. Cannot determine calibration status. Cannot proceed."
        ) from e


def default_cell_arcsec(ms_path: str) -> float:
    """Estimate cell size (arcsec) as a fraction of synthesized beam.

    Uses uv extents as proxy: theta ~ 0.5 * lambda / umax (radians).
    Returns 1/5 of theta in arcsec, clipped to [0.1, 60].

    Parameters
    ----------
    """
    try:
        from daskms import xds_from_ms  # type: ignore[import]

        dsets = xds_from_ms(ms_path, columns=["UVW", "DATA"], chunks={})
        umax = 0.0
        freq_list: list[float] = []
        for ds in dsets:
            uvw = np.asarray(ds.UVW.data.compute())
            umax = max(umax, float(np.nanmax(np.abs(uvw[:, 0]))))
            # derive mean freq per ddid
            with table(f"{ms_path}::DATA_DESCRIPTION", readonly=True) as dd:
                spw_map = dd.getcol("SPECTRAL_WINDOW_ID")
                spw_id = int(spw_map[ds.attrs["DATA_DESC_ID"]])
            with table(f"{ms_path}::SPECTRAL_WINDOW", readonly=True) as spw:
                chan = spw.getcol("CHAN_FREQ")[spw_id]
            freq_list.append(float(np.nanmean(chan)))
        if umax <= 0 or not freq_list:
            raise RuntimeError("bad umax or freq")
        c = 299_792_458.0
        lam = c / float(np.nanmean(freq_list))
        theta_rad = 0.5 * lam / umax
        cell = max(0.1, min(60.0, np.degrees(theta_rad) * 3600.0 / 5.0))
        return float(cell)
    except (OSError, RuntimeError, KeyError, ValueError):
        # CASA-only fallback using casacore tables if daskms missing
        try:
            with table(f"{ms_path}::MAIN", readonly=True) as main_tbl:
                uvw0 = main_tbl.getcol("UVW", 0, min(10000, main_tbl.nrows()))
                umax = float(np.nanmax(np.abs(uvw0[:, 0])))
            with table(f"{ms_path}::SPECTRAL_WINDOW", readonly=True) as spw:
                chan = spw.getcol("CHAN_FREQ")
                if hasattr(chan, "__array__"):
                    freq_scalar = float(np.nanmean(chan))
                else:
                    freq_scalar = float(np.nanmean(np.asarray(chan)))
            if umax <= 0 or not np.isfinite(freq_scalar):
                return 2.0
            c = 299_792_458.0
            lam = c / freq_scalar
            theta_rad = 0.5 * lam / umax
            cell = max(0.1, min(60.0, np.degrees(theta_rad) * 3600.0 / 5.0))
            return float(cell)
        except (OSError, RuntimeError, KeyError, ValueError):
            return 2.0


# Masking Utilities
# Functions imported from dsa110_contimg.core.imaging.masks to avoid duplication


def prepare_cleaning_mask(
    fits_mask: Path | None,
    target_mask: Path | None = None,
    galvin_clip_mask: Path | None = None,
    galvin_box_size: int = 100,
    galvin_adaptive_depth: int = 3,
    erode_beam_shape: bool = False,
) -> Path | None:
    """Prepare a cleaning mask by combining optional target mask, adaptive clip mask,
    and beam erosion.

    Parameters
    ----------
    fits_mask :
        Path to input FITS mask (modified in place or copied).
    target_mask :
        Optional path to mask to intersect with (AND).
    galvin_clip_mask :
        Optional path to image for adaptive clipping (minimum_absolute_clip).
    galvin_box_size :
        Box size for Galvin adaptive clip (default 100 pixels).
    galvin_adaptive_depth :
        Max iterations for adaptive subdivision (default 3).
    erode_beam_shape :
        Whether to erode mask by beam shape.
    fits_mask: Optional[Path] :

    target_mask: Optional[Path] :
         (Default value = None)
    galvin_clip_mask: Optional[Path] :
         (Default value = None)

    Returns
    -------
        Path to the final prepared mask (same as fits_mask input).

    """
    from astropy.io import fits

    if fits_mask is None:
        return None

    # Use str conversion for Path compatibility
    mask_path = Path(fits_mask).absolute()
    if not mask_path.exists():
        LOG.warning(f"Mask file not found: {mask_path}")
        return None

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

        # Adaptive clipping
        if galvin_clip_mask is not None:
            clip_path = Path(galvin_clip_mask).absolute()
            if clip_path.exists():
                try:
                    with fits.open(clip_path) as hdul_clip:
                        clip_data = hdul_clip[0].data
                        if clip_data.ndim == 4:
                            clip_array = clip_data[0, 0, :, :]
                        elif clip_data.ndim == 3:
                            clip_array = clip_data[0, :, :]
                        else:
                            clip_array = clip_data

                    # Apply Galvin clip
                    mask_array = minimum_absolute_clip(
                        clip_array,
                        box_size=galvin_box_size,
                        adaptive_max_depth=galvin_adaptive_depth,
                    )
                    LOG.info(
                        f"Applied Galvin adaptive clip using {clip_path} (box_size={galvin_box_size}, depth={galvin_adaptive_depth})"
                    )
                except Exception as e:
                    LOG.warning(f"Failed to apply Galvin clip from {clip_path}: {e}")
            else:
                LOG.warning(f"Galvin clip mask file not found: {clip_path}")

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
                        f"Target mask shape {target_array.shape} mismatch with mask {mask_array.shape}"
                    )

        # Save updated mask (in place update)
        with fits.open(mask_path, mode="update") as hdul:
            # Update data while preserving dimensions
            if hdul[0].data.ndim == 4:
                hdul[0].data[0, 0, :, :] = mask_array.astype(hdul[0].data.dtype)
            elif hdul[0].data.ndim == 3:
                hdul[0].data[0, :, :] = mask_array.astype(hdul[0].data.dtype)
            else:
                hdul[0].data = mask_array.astype(hdul[0].data.dtype)

            hdul.flush()

        return mask_path

    except Exception as e:
        LOG.error(f"Failed to prepare cleaning mask: {e}")
        return None
