"""Core imaging functions for imaging CLI."""

import logging
import math
import os
import shutil
import subprocess
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

# Initialize CASA environment before importing CASA modules
from dsa110_contimg.common.utils.casa_init import ensure_casa_path

ensure_casa_path()

# Prefer module import so mocks on casacore.tables.table are respected at call time
import casacore.tables as casatables  # noqa: E402
import numpy as np  # noqa: E402

# Back-compat symbol for tests that patch dsa110_contimg.core.imaging.cli_imaging.table
table = casatables.table  # noqa: N816 (kept for test patchability)

# DEFERRED IMPORTS: casatasks imports are deferred to avoid CASA log file creation
# at module import time. CASA writes log files to CWD when casatasks is imported.
# Import these inside functions that need them, wrapped in casa_log_environment().
# from casatasks import exportfits, tclean  # MOVED TO _get_casatasks()

if TYPE_CHECKING:
    pass


@dataclass
class ImagingResult:
    """Result of an imaging operation."""

    image_path: str
    rms_noise: float = 0.0
    peak_flux: float = 0.0
    beam_major: float | None = None
    beam_minor: float | None = None
    beam_pa: float | None = None


# For backwards compatibility, provide module-level names that use CASAService
def exportfits(*args, **kwargs):
    """Wrapper for casatasks.exportfits using CASAService.

    Parameters
    ----------
    *args :
    **kwargs :
    """
    from dsa110_contimg.core.calibration.casa_service import CASAService

    return CASAService().exportfits(*args, **kwargs)


def tclean(*args, **kwargs):
    """Wrapper for casatasks.tclean using CASAService.

    Parameters
    ----------
    *args :
    **kwargs :
    """
    from dsa110_contimg.core.calibration.casa_service import CASAService

    return CASAService().tclean(*args, **kwargs)


try:
    from casatools import msmetadata as _msmd  # type: ignore[import]
    from casatools import vpmanager as _vpmanager  # type: ignore[import]
except ImportError:  # pragma: no cover
    _vpmanager = None
    _msmd = None

try:
    from dsa110_contimg.common.utils.run_isolation import prepare_temp_environment  # noqa: E402
except ImportError:  # pragma: no cover - defensive import
    prepare_temp_environment = None  # type: ignore

from dsa110_contimg.core.imaging.cli_utils import default_cell_arcsec, detect_datacolumn  # noqa: E402
from dsa110_contimg.core.imaging.fov import derive_extent_deg  # noqa: E402
from dsa110_contimg.common.unified_config import settings  # noqa: E402
from dsa110_contimg.common.utils.error_context import format_ms_error_with_suggestions  # noqa: E402
from dsa110_contimg.common.utils.gpu_utils import (  # noqa: E402
    build_docker_command,
    get_gpu_config,
)
from dsa110_contimg.common.utils.performance import track_performance  # noqa: E402
from dsa110_contimg.common.utils.runtime_safeguards import require_casa6_python  # noqa: E402
from dsa110_contimg.common.utils.validation import ValidationError, validate_ms  # noqa: E402

LOG = logging.getLogger(__name__)

# Fixed image extent: defaults to 3.5° x 3.5° unless config opts into derivation
FIXED_IMAGE_EXTENT_DEG = settings.imaging.fixed_extent_deg


@track_performance("wsclean", log_result=True)
def run_wsclean(
    ms_path: str,
    imagename: str,
    datacolumn: str,
    field: str,
    imsize: int,
    cell_arcsec: float,
    weighting: str,
    robust: float,
    specmode: str,
    deconvolver: str,
    nterms: int,
    niter: int,
    threshold: str,
    pbcor: bool,
    uvrange: str,
    pblimit: float,
    quality_tier: str,
    wsclean_path: str | None = None,
    gridder: str = "idg",
    mask_path: str | None = None,
    target_mask: str | None = None,
    galvin_clip_mask: str | None = None,
    galvin_box_size: int = 100,
    galvin_adaptive_depth: int = 3,
    erode_beam_shape: bool = False,
    idg_mode: str = "gpu",
    gpu_ids: list[int] | None = None,
    use_gpu: bool = False,  # Ignored, for backward compatibility
    auto_mask: bool = True,  # Ignored, for backward compatibility
) -> ImagingResult:
    """Run WSClean with parameters mapped from tclean equivalents.

    This function builds a WSClean command-line that matches the tclean
    parameters as closely as possible. MODEL_DATA seeding should be done
    before calling this function via CASA ft().

    Parameters
    ----------
    quality_tier :
        Imaging quality tier
    high_precision :
        Development tier uses 4x coarser cell size
    and :
        fewer iterations for faster processing
    Data :
        is always reordered regardless of quality tier to ensure
    correct :
        multi
    """
    # Check GPU availability if requested
    gpu_config = get_gpu_config()
    if (gridder == "idg" and idg_mode != "cpu") or (gpu_ids is not None) or use_gpu:
        if not gpu_config.has_gpu:
            LOG.warning(
                "GPU acceleration requested but no GPUs detected (or Docker GPU unavailable). "
                "Falling back to CPU-only 'wgridder'."
            )
            gridder = "wgridder"
            idg_mode = "cpu"
            gpu_ids = None

    # Prepare mask if needed
    if mask_path or target_mask or galvin_clip_mask or erode_beam_shape:
        from dsa110_contimg.core.imaging.cli_utils import prepare_cleaning_mask

        # If we have advanced masking options but no base mask_path, we need to decide what to do.
        # For WSClean, we typically start with a base mask.
        # If mask_path is None but others are set, we might need a dummy or full-field mask?
        # For now, assume mask_path is the primary mask being modified.

        if mask_path:
            try:
                from pathlib import Path as PathlibPath

                prepared_mask = prepare_cleaning_mask(
                    fits_mask=PathlibPath(mask_path),
                    target_mask=PathlibPath(target_mask) if target_mask else None,
                    galvin_clip_mask=PathlibPath(galvin_clip_mask) if galvin_clip_mask else None,
                    galvin_box_size=galvin_box_size,
                    galvin_adaptive_depth=galvin_adaptive_depth,
                    erode_beam_shape=erode_beam_shape,
                )
                if prepared_mask:
                    mask_path = str(prepared_mask)
            except Exception as e:
                LOG.warning(
                    "Failed to prepare advanced mask: %s. Proceeding with original mask if any.", e
                )

    # Ensure TELESCOPE_NAME is DSA_110 for EveryBeam beam model compatibility
    # (merge_spws may have set it to OVRO_MMA for CASA listobs compatibility)
    try:
        from dsa110_contimg.core.conversion.helpers_telescope import set_ms_telescope_name

        set_ms_telescope_name(ms_path, "DSA_110")
    except Exception as e:
        LOG.debug("Could not set telescope name (non-fatal): %s", e)

    # Find WSClean executable
    # Priority: Prefer native WSClean over Docker for better performance (2-5x faster)
    # But for GPU acceleration, Docker with nvidia-container-toolkit is required
    gpu_config = get_gpu_config()

    # Always use Docker WSClean unless an explicit path is provided
    if wsclean_path and wsclean_path != "docker":
        wsclean_cmd = [wsclean_path]
    else:
        # Check if native wsclean exists first to avoid docker requirement
        native_wsclean = shutil.which("wsclean")
        if native_wsclean:
            wsclean_cmd = [native_wsclean]
            LOG.info("Using native WSClean at %s", native_wsclean)
        else:
            docker_cmd = shutil.which("docker")
            if not docker_cmd:
                raise RuntimeError("Docker not found and no native wsclean in PATH.")

            docker_user_flags = None
            if os.getenv("WSCLEAN_DOCKER_USER", "").lower() == "host":
                docker_user_flags = ["--user", f"{os.getuid()}:{os.getgid()}"]

            docker_base = build_docker_command(
                image="dsa110-contimg:gpu",
                command=["wsclean"],
                gpu_config=gpu_config,
                env_vars={
                    "NVIDIA_DISABLE_REQUIRE": "1",
                    "MAMBA_ROOT_PREFIX": "/dev/shm/micromamba",
                    "HOME": "/dev/shm/dsa110-contimg",
                },
                extra_flags=docker_user_flags,
            )
            wsclean_cmd = docker_base
            if gpu_config.has_gpu:
                LOG.info("Using Docker WSClean with GPU acceleration (IDG gridder)")
            else:
                LOG.info("Using Docker WSClean (CPU mode)")

    # Build command
    cmd = wsclean_cmd.copy()

    # Output name (use same path for Docker since volumes are mounted)
    cmd.extend(["-name", imagename])

    # Image size and pixel scale
    cmd.extend(["-size", str(imsize), str(imsize)])
    cmd.extend(["-scale", f"{cell_arcsec:.3f}arcsec"])

    # Data column
    if datacolumn == "corrected":
        cmd.extend(["-data-column", "CORRECTED_DATA"])

    # Field selection (if specified)
    if field:
        cmd.extend(["-field", field])

    # Weighting
    if weighting.lower() == "briggs":
        cmd.extend(["-weight", "briggs", str(robust)])
    elif weighting.lower() == "natural":
        cmd.extend(["-weight", "natural"])
    elif weighting.lower() == "uniform":
        cmd.extend(["-weight", "uniform"])

    # Multi-term deconvolution (mtmfs equivalent)
    if specmode == "mfs" and nterms > 1:
        cmd.extend(["-fit-spectral-pol", str(nterms)])
        cmd.extend(["-channels-out", "8"])  # Reasonable default for multi-term
        cmd.extend(["-join-channels"])

    # Deconvolver
    if deconvolver == "multiscale":
        cmd.append("-multiscale")
        # Default scales if not specified
        cmd.extend(["-multiscale-scales", "0,5,15,45"])
    elif deconvolver == "hogbom":
        # Default is hogbom, no flag needed
        pass

    # Iterations and threshold
    cmd.extend(["-niter", str(niter)])

    # Parse threshold string (e.g., "0.005Jy" or "0.1mJy")
    threshold_lower = threshold.lower().strip()
    if threshold_lower.endswith("jy") and not threshold_lower.endswith("mjy"):
        threshold_val = float(threshold_lower[:-2])
        if threshold_val > 0:
            cmd.extend(["-abs-threshold", str(threshold_val)])
    elif threshold_lower.endswith("mjy"):
        threshold_val = float(threshold_lower[:-3]) / 1000.0  # Convert to Jy
        if threshold_val > 0:
            cmd.extend(["-abs-threshold", f"{threshold_val:.6f}"])

    # Primary beam correction disabled for compatibility with older WSClean versions

    # UV range filtering
    if uvrange:
        # Parse ">1klambda" format
        import re

        match = re.match(r"([<>]?)(\d+(?:\.\d+)?)(?:\.)?(k?lambda)", uvrange.lower())
        if match:
            op, val, unit = match.groups()
            val_float = float(val)
            if unit == "klambda":
                val_float *= 1000.0
            if op == ">":
                cmd.extend(["-minuv-l", str(int(val_float))])
            elif op == "<":
                cmd.extend(["-maxuv-l", str(int(val_float))])

    # Gridder selection (WSClean native gridders: wgridder, idg, wstacking, standard)
    # Use WSClean gridder names directly - no tclean compatibility mapping
    cmd.extend(["-gridder", gridder])
    LOG.debug("Using WSClean gridder: %s", gridder)
    if gridder == "idg":
        cmd.extend(["-idg-mode", idg_mode])
        LOG.debug("Using WSClean idg-mode: %s", idg_mode)

    # Reordering (required for multi-spw, but can be slow - only if needed)
    # CRITICAL: Always reorder data - required for correct multi-SPW processing
    # Reorder ensures proper channel ordering across subbands
    cmd.append("-reorder")

    # Mask file (if provided)
    if mask_path:
        cmd.extend(["-fits-mask", mask_path])
        LOG.info("Using mask file: %s", mask_path)

    # Auto-masking (helps with convergence)
    # Note: Auto-masking can be combined with user-provided mask
    # WSClean will use the user mask as initial constraint and auto-expand if needed
    # Values optimized for DSA-110 transient detection (primary science goal):
    # - auto-mask 5σ: conservative mask creation for point sources
    # - auto-threshold 1.0σ: clean deeply within mask
    cmd.extend(["-auto-mask", "5"])
    cmd.extend(["-auto-threshold", "1.0"])
    cmd.extend(["-mgain", "0.8"])

    # Threading: use all available CPU cores (critical for performance!)
    import multiprocessing

    num_threads = os.getenv("WSCLEAN_THREADS", str(multiprocessing.cpu_count()))
    cmd.extend(["-j", num_threads])
    LOG.debug("Using %s threads for WSClean", num_threads)

    # Memory limit (optimized for performance)
    # Development tier: Use more memory for faster gridding/FFT (16GB default)
    # Production mode: Scale with image size (16-32GB)
    if quality_tier == "development":
        # Development tier: Allow more memory for speed (10-30% faster gridding)
        abs_mem = os.getenv("WSCLEAN_ABS_MEM", "16")
    else:
        # Production mode: Scale with image size
        abs_mem = os.getenv("WSCLEAN_ABS_MEM", "32" if imsize >= 2400 else "16")
    cmd.extend(["-abs-mem", abs_mem])
    LOG.debug("WSClean memory allocation: %sGB", abs_mem)

    # Polarity
    cmd.extend(["-pol", "I"])

    # Input MS (use same path for Docker since volumes are mounted)
    cmd.append(ms_path)

    # Log command
    cmd_str = " ".join(cmd)
    LOG.info("Running WSClean: %s", cmd_str)

    # Execute with configurable timeout
    # Set WSCLEAN_DOCKER_TIMEOUT env var to override default (in seconds)
    from dsa110_contimg.common.utils import get_env_int

    wsclean_timeout = get_env_int("WSCLEAN_DOCKER_TIMEOUT", default=1800)
    use_docker = len(wsclean_cmd) > 1 and wsclean_cmd[0] == "docker"
    env = None
    if not use_docker:
        from dsa110_contimg.common.utils.wsclean_utils import build_wsclean_native_env

        env = build_wsclean_native_env()

    # Get MS info for progress estimation
    try:
        import casacore.tables as ct

        with ct.table(ms_path, ack=False) as t:
            n_rows = t.nrows()
    except Exception:
        n_rows = 1_000_000

    # Use progress monitoring
    from dsa110_contimg.common.utils.progress import StageProgressMonitor, estimate_imaging_time

    estimated_seconds = estimate_imaging_time(n_rows, imsize, niter)

    # WSClean output naming: "-MFS-" suffix only when -channels-out is used (nterms > 1)
    wsclean_suffix = "-MFS-" if nterms > 1 else "-"

    monitor = StageProgressMonitor(
        "WSClean imaging",
        output_path=f"{imagename}{wsclean_suffix}image.fits",  # Primary output file
        poll_interval=10.0,  # Imaging is slow, use longer interval
        estimated_seconds=estimated_seconds,
    )
    monitor.set_context(rows=n_rows, imsize=imsize, niter=niter)

    t0 = time.perf_counter()
    try:
        with monitor:
            subprocess.run(
                cmd,
                check=True,
                capture_output=False,
                text=True,
                timeout=wsclean_timeout,
                env=env,
            )
        LOG.info("WSClean completed in %.2fs", time.perf_counter() - t0)
    except subprocess.TimeoutExpired:
        LOG.error(
            "WSClean timed out after %ds. If using Docker, attempting cleanup...",
            wsclean_timeout,
        )
        # Attempt to kill any orphaned Docker containers
        if use_docker:
            try:
                # Find and kill containers running wsclean image
                kill_cmd = [
                    "docker",
                    "ps",
                    "-q",
                    "--filter",
                    "ancestor=dsa110-contimg:gpu",
                ]
                result = subprocess.run(kill_cmd, capture_output=True, text=True, timeout=10)
                container_ids = result.stdout.strip().split()
                for cid in container_ids:
                    if cid:
                        LOG.warning("Killing orphaned WSClean container: %s", cid)
                        subprocess.run(["docker", "kill", cid], timeout=10, check=False)
            except Exception as cleanup_err:
                LOG.warning("Failed to cleanup Docker containers: %s", cleanup_err)
        raise RuntimeError(
            f"WSClean timed out after {wsclean_timeout}s. "
            "Consider increasing WSCLEAN_DOCKER_TIMEOUT or disabling NVSS seeding."
        )
    except subprocess.CalledProcessError as e:
        LOG.error("WSClean failed with exit code %d", e.returncode)
        raise RuntimeError(f"WSClean execution failed: {e}") from e
    except FileNotFoundError:
        suggestions = [
            "Check WSClean installation",
            "Verify WSClean is in PATH",
            "Use --wsclean-path to specify WSClean location",
            "Install WSClean: https://gitlab.com/aroffringa/wsclean",
        ]
        error_msg = format_ms_error_with_suggestions(
            FileNotFoundError(f"WSClean executable not found: {wsclean_cmd}"),
            ms_path,
            "WSClean execution",
            suggestions,
        )
        raise RuntimeError(error_msg) from None
    except Exception as e:
        suggestions = [
            "Check WSClean logs for detailed error information",
            "Verify MS path and file permissions",
            "Check disk space for output images",
            "Review WSClean parameters and configuration",
        ]
        error_msg = format_ms_error_with_suggestions(e, ms_path, "WSClean execution", suggestions)
        raise RuntimeError(error_msg) from e

    # Calculate stats and return result
    output_fits = f"{imagename}{wsclean_suffix}image.fits"
    rms = 0.0
    peak = 0.0
    beam_major = None
    beam_minor = None
    beam_pa = None

    try:
        from astropy.io import fits

        if os.path.exists(output_fits):
            with fits.open(output_fits) as hdul:
                data = hdul[0].data
                header = hdul[0].header
                # Simple stats (robust)
                valid = data[np.isfinite(data)]
                if len(valid) > 0:
                    peak = float(np.max(valid))
                    # MAD-estimated RMS (approximate)
                    median = np.median(valid)
                    rms = float(1.4826 * np.median(np.abs(valid - median)))

                # Beam info
                if "BMAJ" in header:
                    beam_major = float(header["BMAJ"]) * 3600.0
                if "BMIN" in header:
                    beam_minor = float(header["BMIN"]) * 3600.0
                if "BPA" in header:
                    beam_pa = float(header["BPA"])
    except Exception as e:
        LOG.warning("Failed to calculate image stats: %s", e)

    return ImagingResult(
        image_path=output_fits,
        rms_noise=rms * 1000.0,  # Convert to mJy
        peak_flux=peak * 1000.0,
        beam_major=beam_major,
        beam_minor=beam_minor,
        beam_pa=beam_pa,
    )


@track_performance("imaging", log_result=True)
@require_casa6_python
def image_ms(
    ms_path: str,
    *,
    imagename: str,
    field: str = "",
    spw: str = "",
    imsize: int = 2400,
    cell_arcsec: float | str | None = 6.0,
    weighting: str = "briggs",
    robust: float = 0.5,
    specmode: str = "mfs",
    deconvolver: str = "hogbom",
    nterms: int = 1,
    niter: int = 1000,
    threshold: str = "0.005Jy",
    pbcor: bool = True,
    phasecenter: str | None = None,
    gridder: str = "idg",
    wprojplanes: int = -1,
    uvrange: str = ">1klambda",
    pblimit: float = 0.2,
    psfcutoff: float | None = None,
    quality_tier: str = "standard",
    skip_fits: bool = False,
    vptable: str | None = None,
    wbawp: bool | None = None,
    cfcache: str | None = None,
    unicat_min_mjy: float | None = None,
    nvss_min_mjy: float | None = None,
    calib_ra_deg: float | None = None,
    calib_dec_deg: float | None = None,
    calib_flux_jy: float | None = None,
    backend: str = "wsclean",
    wsclean_path: str | None = None,
    export_model_image: bool = False,
    use_unicat_mask: bool = True,
    mask_path: str | None = None,
    mask_radius_arcsec: float = 60.0,
    target_mask: str | None = None,
    galvin_clip_mask: str | None = None,
    galvin_box_size: int = 100,
    galvin_adaptive_depth: int = 3,
    erode_beam_shape: bool = False,
) -> None:
    """Main imaging function for Measurement Sets.

    Supports both CASA tclean and WSClean backends. WSClean is the default.
    Automatically selects CORRECTED_DATA when present, otherwise uses DATA.

    Cell Size (Pixel Scale):
        Default is 6.0 arcseconds, which provides ~3-5 pixels across the
        synthesized beam for typical DSA-110 L-band observations.
        Set cell_arcsec='auto' to calculate automatically from UV coverage
        using the formula: cell = (λ / 2·umax) / 5.

    Quality Tiers:
        - development: 4x coarser cell size, max 300 iterations, NVSS threshold 10 mJy.
          NON-SCIENCE QUALITY - for code testing only. Data is always reordered.
        - standard: Full quality imaging (recommended for science).
        - high_precision: Enhanced settings with 2000+ iterations, NVSS threshold 5 mJy.

    NVSS Seeding:
        When pbcor=True, NVSS sources are limited to the primary beam extent
        (based on pblimit) to avoid including sources beyond the corrected region.
        The seeding radius is calculated from the primary beam FWHM and pblimit.

    Masking:
        When use_unicat_mask=True and unicat_min_mjy is provided (or nvss_min_mjy alias),
        generates a FITS mask from unified catalog sources for WSClean. This provides
        2-4x faster imaging by restricting cleaning to known source locations. Masking is
        only supported for the WSClean backend.

    Parameters
    ----------
    """
    from dsa110_contimg.common.utils.validation import validate_corrected_data_quality

    # Validate MS using shared validation module
    try:
        validate_ms(
            ms_path,
            check_empty=True,
            check_columns=["DATA", "ANTENNA1", "ANTENNA2", "TIME", "UVW"],
        )
    except ValidationError as e:
        suggestions = [
            "Check MS path is correct and file exists",
            "Verify file permissions",
            "Run validation: python -m dsa110_contimg.core.calibration.cli validate --ms <path>",
            "Check MS structure and integrity",
        ]
        error_msg = format_ms_error_with_suggestions(e, ms_path, "MS validation", suggestions)
        raise RuntimeError(error_msg) from e

    # Validate CORRECTED_DATA quality if present - FAIL if calibration appears unapplied
    warnings = validate_corrected_data_quality(ms_path)
    if warnings:
        # Distinguish between unpopulated data warnings and validation errors
        unpopulated_warnings = [
            w
            for w in warnings
            if "appears unpopulated" in w.lower()
            or "zero rows" in w.lower()
            or "all sampled data is flagged" in w.lower()
        ]
        validation_errors = [
            w for w in warnings if w.startswith("Error validating CORRECTED_DATA:")
        ]

        if unpopulated_warnings:
            # CORRECTED_DATA exists but is unpopulated - calibration failed
            suggestions = [
                "Re-run calibration on this MS",
                "Check calibration logs for errors",
                "Verify calibration tables were applied correctly",
                "Use --datacolumn=DATA to image uncalibrated data (not recommended)",
            ]
            error_msg = format_ms_error_with_suggestions(
                RuntimeError("CORRECTED_DATA column exists but appears unpopulated"),
                ms_path,
                "calibration validation",
                suggestions,
            )
            error_msg += f"\nDetails: {'; '.join(unpopulated_warnings)}"
            LOG.error(error_msg)
            raise RuntimeError(error_msg)
        elif validation_errors:
            # Validation error (e.g., permission denied, file access issue)
            suggestions = [
                "Check file permissions on MS directory",
                "Verify MS file is not corrupted",
                "Check disk space and file system",
                "Review detailed error logs",
            ]
            error_msg = format_ms_error_with_suggestions(
                RuntimeError("Failed to validate CORRECTED_DATA"),
                ms_path,
                "MS validation",
                suggestions,
            )
            error_msg += f"\nDetails: {'; '.join(validation_errors)}"
            LOG.error(error_msg)
            raise RuntimeError(error_msg)
        else:
            # Other warnings (shouldn't happen, but handle gracefully)
            suggestions = [
                "Review calibration validation warnings",
                "Check MS structure and integrity",
                "Verify calibration was applied correctly",
            ]
            error_msg = format_ms_error_with_suggestions(
                RuntimeError("Calibration validation warnings"),
                ms_path,
                "calibration validation",
                suggestions,
            )
            error_msg += f"\nDetails: {'; '.join(warnings)}"
            LOG.error(error_msg)
            raise RuntimeError(error_msg)

    # PRECONDITION CHECK: Verify sufficient disk space for images
    # This ensures we follow "measure twice, cut once" - verify resources upfront
    # before expensive imaging operations.
    try:
        output_dir = os.path.dirname(os.path.abspath(imagename))
        os.makedirs(output_dir, exist_ok=True)

        # Estimate image size: rough estimate based on imsize and number of images
        # Each image is approximately: imsize^2 * 4 bytes (float32) * number of images
        # We create: .image, .model, .residual, .pb, .pbcor = 5 images
        # Plus weights, etc. Use 10x safety margin for overhead
        bytes_per_pixel = 4  # float32
        # Conservative estimate (.image, .model, .residual, .pb, .pbcor, weights, etc.)
        num_images = 10
        image_size_estimate = (
            imsize * imsize * bytes_per_pixel * num_images * 10
        )  # 10x safety margin

        # CRITICAL: Check disk space (fatal check for imaging operations)
        from dsa110_contimg.core.mosaic.error_handling import check_disk_space

        _, space_msg = check_disk_space(
            imagename,
            required_bytes=image_size_estimate,
            operation=f"imaging of {ms_path}",
            fatal=True,  # Fail fast if insufficient space
        )
        LOG.info(space_msg)
    except RuntimeError:
        # Re-raise RuntimeError from fatal disk space check
        raise
    except Exception as e:
        # Other exceptions: log warning but don't fail (may be permission issues, etc.)
        LOG.warning("Failed to check disk space: %s", e)

    # Prepare temp dirs and working directory to keep TempLattice* off the repo
    try:
        if prepare_temp_environment is not None:
            from dsa110_contimg.common.unified_config import settings

            out_dir = os.path.dirname(os.path.abspath(imagename))
            root = os.getenv("CONTIMG_TMPFS_DIR") or str(settings.paths.tmpfs_dir)
            prepare_temp_environment(root, cwd_to=out_dir)
    except (OSError, RuntimeError):
        # Best-effort; continue even if temp prep fails
        pass
    datacolumn = detect_datacolumn(ms_path)
    if cell_arcsec is None or cell_arcsec == "auto":
        cell_arcsec = default_cell_arcsec(ms_path)

    # Backwards compatibility: accept deprecated nvss_min_mjy alias
    if nvss_min_mjy is not None:
        if unicat_min_mjy is None:
            unicat_min_mjy = nvss_min_mjy
        else:
            LOG.warning(
                "Both unicat_min_mjy and deprecated nvss_min_mjy provided; using unicat_min_mjy=%s",
                unicat_min_mjy,
            )

    # Enforce fixed or derived FoV extent
    extent_deg = derive_extent_deg()
    desired_extent_arcsec = extent_deg * 3600.0

    # Store original imsize for warning if user overrode it
    user_imsize = imsize

    # Calculate imsize to maintain configured extent
    # If user specified both imsize and cell_arcsec, use cell_arcsec and recalculate imsize
    calculated_imsize = int(np.ceil(desired_extent_arcsec / cell_arcsec))
    # Ensure even number (CASA requirement)
    if calculated_imsize % 2 != 0:
        calculated_imsize += 1

    # Warn if user specified imsize but we're overriding it
    if user_imsize != 2400:  # 2400 is the default, so only warn if user explicitly set it
        if calculated_imsize != user_imsize:
            LOG.warning(
                "User-specified imsize=%d overridden to maintain %.2f° extent: "
                "calculated imsize=%d from cell_arcsec=%.3f arcsec",
                user_imsize,
                extent_deg,
                calculated_imsize,
                cell_arcsec,
            )

    imsize = calculated_imsize
    cell = f"{cell_arcsec:.3f}arcsec"

    # Apply quality tier settings
    if quality_tier == "development":
        # :warning:  NON-SCIENCE QUALITY - For code testing only
        LOG.warning(
            "=" * 80 + "\n"
            ":warning:  DEVELOPMENT TIER: NON-SCIENCE QUALITY\n"
            "   This tier uses coarser resolution and fewer iterations.\n"
            "   NEVER use for actual science observations or ESE detection.\n"
            "   Results will have reduced angular resolution and deconvolution quality.\n"
            "=" * 80
        )
        # Coarser resolution (4x default cell size)
        default_cell = default_cell_arcsec(ms_path)
        if abs(cell_arcsec - default_cell) < 0.01:  # Only adjust if using default cell size
            cell_arcsec = cell_arcsec * 4.0
            # Recalculate imsize for new cell size
            calculated_imsize = int(np.ceil(desired_extent_arcsec / cell_arcsec))
            if calculated_imsize % 2 != 0:
                calculated_imsize += 1
            imsize = calculated_imsize
            cell = f"{cell_arcsec:.3f}arcsec"
            LOG.info(
                "Development tier: using coarser cell size (%.3f arcsec) - NON-SCIENCE QUALITY",
                cell_arcsec,
            )
        niter = min(niter, 300)  # Fewer iterations
        # Lower unified catalog seeding threshold for faster convergence
        if unicat_min_mjy is None:
            unicat_min_mjy = 10.0
            LOG.info(
                "Development tier: Unified catalog seeding threshold set to %s mJy (NON-SCIENCE)",
                unicat_min_mjy,
            )

    elif quality_tier == "standard":
        # Recommended for all science observations - no compromises
        LOG.info("Standard tier: full quality imaging (recommended for science)")
        # Use default settings optimized for science quality

    elif quality_tier == "high_precision":
        # Enhanced quality for critical observations
        LOG.info("High precision tier: enhanced quality settings (slower)")
        niter = max(niter, 2000)  # More iterations for better deconvolution
        if unicat_min_mjy is None:
            unicat_min_mjy = 5.0  # Lower threshold for cleaner sky model
            LOG.info(
                "High precision tier: Unified catalog seeding threshold set to %s mJy",
                unicat_min_mjy,
            )
    LOG.info("Imaging %s -> %s", ms_path, imagename)
    LOG.info(
        "datacolumn=%s cell=%s imsize=%d quality_tier=%s",
        datacolumn,
        cell,
        imsize,
        quality_tier,
    )

    # Build common kwargs for tclean, adding optional params only when needed
    kwargs = dict(
        vis=ms_path,
        imagename=imagename,
        datacolumn=datacolumn,
        field=field,
        spw=spw,
        imsize=[imsize, imsize],
        cell=[cell, cell],
        weighting=weighting,
        robust=robust,
        specmode=specmode,
        deconvolver=deconvolver,
        nterms=nterms,
        niter=niter,
        threshold=threshold,
        gridder=gridder,
        wprojplanes=wprojplanes,
        stokes="I",
        restoringbeam="",
        pbcor=pbcor,
        phasecenter=phasecenter if phasecenter else "",
        interactive=False,
    )
    if uvrange:
        kwargs["uvrange"] = uvrange
    if pblimit is not None:
        kwargs["pblimit"] = pblimit
    if psfcutoff is not None:
        kwargs["psfcutoff"] = psfcutoff
    if vptable:
        kwargs["vptable"] = vptable
    if wbawp is not None:
        kwargs["wbawp"] = bool(wbawp)
    if cfcache:
        kwargs["cfcache"] = cfcache

    # Avoid overwriting any seeded MODEL_DATA during tclean
    kwargs["savemodel"] = "none"

    # Compute approximate FoV radius from image geometry
    fov_x = (cell_arcsec * imsize) / 3600.0
    fov_y = (cell_arcsec * imsize) / 3600.0
    radius_deg = 0.5 * float(math.hypot(fov_x, fov_y))

    # Get phase center from MS (needed for mask generation and unified catalog seeding)
    ra0_deg = dec0_deg = None
    with casatables.table(f"{ms_path}::FIELD", readonly=True) as fld:
        try:
            ph = fld.getcol("PHASE_DIR")[0]
            ra0_deg = float(ph[0][0]) * (180.0 / np.pi)
            dec0_deg = float(ph[0][1]) * (180.0 / np.pi)
        except (KeyError, IndexError, TypeError):
            pass
    if ra0_deg is None or dec0_deg is None:
        LOG.warning("Could not determine phase center from MS FIELD table")

    # Optional: seed a single-component calibrator model if provided and in FoV
    did_seed = False
    if (
        calib_ra_deg is not None
        and calib_dec_deg is not None
        and calib_flux_jy is not None
        and calib_flux_jy > 0
    ):
        try:
            with casatables.table(f"{ms_path}::FIELD", readonly=True) as fld:
                ph = fld.getcol("PHASE_DIR")[0]
                ra0_deg = float(ph[0][0]) * (180.0 / np.pi)
                dec0_deg = float(ph[0][1]) * (180.0 / np.pi)
            # crude small-angle separation in deg
            d_ra = (float(calib_ra_deg) - ra0_deg) * np.cos(np.deg2rad(dec0_deg))
            d_dec = float(calib_dec_deg) - dec0_deg
            sep_deg = float(math.hypot(d_ra, d_dec))
            if sep_deg <= radius_deg * 1.05:
                from dsa110_contimg.core.calibration.skymodels import (
                    make_point_skymodel,
                    predict_from_skymodel_wsclean,
                )

                sky = make_point_skymodel(
                    name="calibrator",
                    ra_deg=float(calib_ra_deg),
                    dec_deg=float(calib_dec_deg),
                    flux_jy=float(calib_flux_jy),
                    freq_ghz=1.4,
                )
                predict_from_skymodel_wsclean(
                    ms_path=ms_path, skymodel=sky, field=field or "0",
                )
                LOG.info(
                    "Seeded MODEL_DATA with calibrator point model (flux=%.3f Jy)",
                    calib_flux_jy,
                )
                did_seed = True
        except Exception as exc:
            LOG.debug("Calibrator seeding skipped: %s", exc)

    # Optional: seed a sky model from unified catalog (> unicat_min_mjy mJy) via wsclean -predict, if no calibrator seed
    if (not did_seed) and (unicat_min_mjy is not None):
        try:

            # Use phase center already determined above
            if ra0_deg is None or dec0_deg is None:
                raise RuntimeError("FIELD::PHASE_DIR not available")

            # Limit unified catalog seeding radius to primary beam extent when pbcor is enabled
            # Primary beam FWHM at 1.4 GHz: ~3.2 degrees (1.22 * lambda / D)
            # Use pblimit to determine effective radius (typically 20% of peak = ~1.6 deg radius)
            # Mean observing frequency and bandwidth
            freq_ghz = 1.4
            bandwidth_hz = 250e6  # Default 250 MHz
            try:
                with casatables.table(f"{ms_path}::SPECTRAL_WINDOW", readonly=True) as spw:
                    ch = spw.getcol("CHAN_FREQ")[0]
                    freq_ghz = float(np.nanmean(ch)) / 1e9
                    if len(ch) > 1:
                        bandwidth_hz = float(np.max(ch) - np.min(ch) + abs(ch[1] - ch[0]))
                    else:
                        bandwidth_hz = float(spw.getcol("TOTAL_BANDWIDTH")[0])
            except (OSError, RuntimeError, KeyError):
                pass

            # Calculate primary beam radius based on pblimit
            # Primary beam FWHM = 1.22 * lambda / D
            # For DSA-110: D = 4.7 m, lambda = c / (freq_ghz * 1e9)
            # At pblimit=0.2, effective radius is approximately FWHM * sqrt(-ln(0.2)) / sqrt(-ln(0.5))
            c_mps = 299792458.0
            dish_dia_m = 4.7
            lambda_m = c_mps / (freq_ghz * 1e9)
            fwhm_rad = 1.22 * lambda_m / dish_dia_m
            fwhm_deg = math.degrees(fwhm_rad)

            # Calculate radius at pblimit (Airy pattern: PB = (2*J1(x)/x)^2, solve for PB = pblimit)
            # Approximate: radius at pblimit ≈ FWHM * sqrt(-ln(pblimit)) / sqrt(-ln(0.5))
            if pbcor and pblimit > 0:
                pb_radius_deg = fwhm_deg * math.sqrt(-math.log(pblimit)) / math.sqrt(-math.log(0.5))
                # Use the smaller of image radius or primary beam radius
                unicat_radius_deg = min(radius_deg, pb_radius_deg)
                LOG.info(
                    "Limiting unified catalog seeding to primary beam extent: %.2f deg (pblimit=%.2f, FWHM=%.2f deg)",
                    unicat_radius_deg,
                    pblimit,
                    fwhm_deg,
                )
            else:
                unicat_radius_deg = radius_deg

            # Note: wsclean -predict with -model-list is not supported by the installed wsclean version.
            # We use a 2-step process: -draw-model then -predict.
            if backend == "wsclean":
                # Use wsclean -predict (faster, multi-threaded)
                txt_path = f"{imagename}.unicat_{float(unicat_min_mjy):g}mJy.txt"
                LOG.info(
                    "Creating Unified source list (FIRST+RACS+NVSS) (>%s mJy, radius %.2f deg) for wsclean -predict",
                    unicat_min_mjy,
                    unicat_radius_deg,
                )
                from dsa110_contimg.core.calibration.skymodels import make_unified_wsclean_list

                make_unified_wsclean_list(
                    ra0_deg,
                    dec0_deg,
                    unicat_radius_deg,
                    min_mjy=float(unicat_min_mjy),
                    freq_ghz=freq_ghz,
                    out_path=txt_path,
                )

                # Determine wsclean executable
                wsclean_exec = wsclean_path
                if not wsclean_exec:
                    wsclean_exec = shutil.which("wsclean")

                # If not found locally, check for Docker
                use_docker = False
                if not wsclean_exec:
                    if shutil.which("docker"):
                        # Use Docker-based wsclean via build_docker_command
                        use_docker = True
                        LOG.info("Using Docker-based wsclean for model seeding")
                    else:
                        raise RuntimeError("wsclean executable not found for prediction")

                if use_docker:
                    # Docker command construction using build_docker_command (same as main run_wsclean)

                    # Convert decimal degrees to h:m:s and d:m:s format
                    # WSClean requires this format for -draw-centre
                    ra_hours = ra0_deg / 15.0
                    ra_h = int(ra_hours)
                    ra_m = int((ra_hours - ra_h) * 60)
                    ra_s = ((ra_hours - ra_h) * 60 - ra_m) * 60
                    ra_str = f"{ra_h}h{ra_m}m{ra_s:.3f}s"

                    dec_sign = "+" if dec0_deg >= 0 else "-"
                    dec_abs = abs(dec0_deg)
                    dec_d = int(dec_abs)
                    dec_m = int((dec_abs - dec_d) * 60)
                    dec_s = ((dec_abs - dec_d) * 60 - dec_m) * 60
                    dec_str = f"{dec_sign}{dec_d}d{dec_m}m{dec_s:.3f}s"

                    # Build Docker command using the same approach as main run_wsclean
                    gpu_config = get_gpu_config()
                    docker_user_flags = None
                    if os.getenv("WSCLEAN_DOCKER_USER", "").lower() == "host":
                        docker_user_flags = ["--user", f"{os.getuid()}:{os.getgid()}"]

                    docker_base = build_docker_command(
                        image="dsa110-contimg:gpu",
                        command=["wsclean"],
                        gpu_config=gpu_config,
                        env_vars={
                            "NVIDIA_DISABLE_REQUIRE": "1",
                            "MAMBA_ROOT_PREFIX": "/dev/shm/micromamba",
                            "HOME": "/dev/shm/dsa110-contimg",
                        },
                        extra_flags=docker_user_flags,
                    )

                    # Step 1: Render model image from text list
                    model_prefix = f"{imagename}.nvss_model"
                    cmd_draw = docker_base.copy()
                    cmd_draw.extend(
                        [
                            "-draw-model",
                            txt_path,
                            "-name",
                            model_prefix,
                            "-draw-frequencies",
                            f"{freq_ghz * 1e9}",
                            f"{bandwidth_hz}",
                            "-draw-spectral-terms",
                            "2",
                            "-size",
                            str(imsize),
                            str(imsize),
                            "-scale",
                            f"{cell_arcsec}arcsec",
                            "-draw-centre",
                            ra_str,
                            dec_str,
                        ]
                    )
                    LOG.info(
                        "Running wsclean -draw-model via Docker: %s",
                        " ".join(cmd_draw[:8]) + " ...",
                    )
                    subprocess.run(cmd_draw, check=True, timeout=300)  # 5 min timeout

                    # Step 1.5: Rename output file for prediction
                    # WSClean -draw-model creates prefix-term-0.fits
                    # WSClean -predict expects prefix-model.fits
                    term_file = f"{model_prefix}-term-0.fits"
                    model_file = f"{model_prefix}-model.fits"
                    if os.path.exists(term_file):
                        shutil.move(term_file, model_file)
                        LOG.info("Renamed %s -> %s", term_file, model_file)
                    else:
                        LOG.warning("Expected output file not found: %s", term_file)

                    # Step 2: Predict from rendered image
                    cmd_predict = docker_base.copy()
                    cmd_predict.extend(
                        [
                            "-predict",
                            "-reorder",  # Required for multi-SPW MS
                            "-name",
                            model_prefix,
                            ms_path,
                        ]
                    )
                    LOG.info(
                        "Running wsclean -predict via Docker: %s",
                        " ".join(cmd_predict[:8]) + " ...",
                    )
                    start_time = time.perf_counter()

                    try:
                        subprocess.run(
                            cmd_predict, check=True, timeout=600, capture_output=True, text=True
                        )
                        elapsed = time.perf_counter() - start_time
                        LOG.info(
                            "Docker WSClean -predict completed successfully in %.1fs", elapsed
                        )

                    except subprocess.TimeoutExpired:
                        elapsed = time.perf_counter() - start_time
                        LOG.error("Docker WSClean -predict timeout after %.1fs", elapsed)
                        raise
                    except subprocess.CalledProcessError as e:
                        elapsed = time.perf_counter() - start_time
                        LOG.error("Docker WSClean -predict failed after %.1fs: %s", elapsed, e)
                        raise
                        raise

                else:
                    # Native WSClean execution
                    from dsa110_contimg.common.utils.wsclean_utils import (
                        build_wsclean_native_env,
                    )

                    native_env = build_wsclean_native_env()

                    # Convert decimal degrees to h:m:s and d:m:s format
                    ra_hours = ra0_deg / 15.0
                    ra_h = int(ra_hours)
                    ra_m = int((ra_hours - ra_h) * 60)
                    ra_s = ((ra_hours - ra_h) * 60 - ra_m) * 60
                    ra_str = f"{ra_h}h{ra_m}m{ra_s:.3f}s"

                    dec_sign = "+" if dec0_deg >= 0 else "-"
                    dec_abs = abs(dec0_deg)
                    dec_d = int(dec_abs)
                    dec_m = int((dec_abs - dec_d) * 60)
                    dec_s = ((dec_abs - dec_d) * 60 - dec_m) * 60
                    dec_str = f"{dec_sign}{dec_d}d{dec_m}m{dec_s:.3f}s"

                    # Step 1: Render model
                    cmd_draw = [
                        wsclean_exec,
                        "-draw-model",
                        txt_path,
                        "-name",
                        f"{imagename}.nvss_model",
                        "-draw-frequencies",
                        f"{freq_ghz * 1e9}",
                        f"{bandwidth_hz}",
                        "-draw-spectral-terms",
                        "2",
                        "-size",
                        str(imsize),
                        str(imsize),
                        "-scale",
                        f"{cell_arcsec}arcsec",
                        "-draw-centre",
                        ra_str,
                        dec_str,
                    ]
                    LOG.info("Running wsclean -draw-model: %s", " ".join(cmd_draw))
                    subprocess.run(
                        cmd_draw,
                        check=True,
                        timeout=300,
                        env=native_env,
                    )  # 5 min timeout

                    # Step 1.5: Rename output file for prediction
                    term_file = f"{imagename}.nvss_model-term-0.fits"
                    model_file = f"{imagename}.nvss_model-model.fits"
                    if os.path.exists(term_file):
                        shutil.move(term_file, model_file)
                        LOG.info("Renamed %s -> %s", term_file, model_file)
                    else:
                        LOG.warning("Expected output file not found: %s", term_file)

                    # Step 2: Predict
                    cmd_predict = [
                        wsclean_exec,
                        "-predict",
                        "-reorder",  # Required for multi-SPW MS
                        "-name",
                        f"{imagename}.nvss_model",
                        ms_path,
                    ]
                    LOG.info("Running wsclean -predict: %s", " ".join(cmd_predict))
                    LOG.info(
                        "DIAGNOSTIC: Starting native WSClean -predict at %s",
                        time.strftime("%Y-%m-%d %H:%M:%S"),
                    )
                    start_time = time.perf_counter()

                    try:
                        subprocess.run(
                            cmd_predict,
                            check=True,
                            timeout=600,
                            capture_output=True,
                            text=True,
                            env=native_env,
                        )
                        elapsed = time.perf_counter() - start_time
                        LOG.info(
                            "DIAGNOSTIC: Native WSClean -predict completed successfully in %.1fs",
                            elapsed,
                        )
                    except subprocess.TimeoutExpired:
                        elapsed = time.perf_counter() - start_time
                        LOG.error("DIAGNOSTIC: subprocess.TimeoutExpired after %.1fs", elapsed)
                        raise
                    except Exception as e:
                        elapsed = time.perf_counter() - start_time
                        LOG.error(
                            "DIAGNOSTIC: Exception %s after %.1fs", type(e).__name__, elapsed
                        )
                        raise

                LOG.info("Seeded MODEL_DATA with wsclean -predict")

            else:
                # Non-wsclean imager (tclean, etc.): use unified skymodel + wsclean -predict
                from dsa110_contimg.core.calibration.skymodels import (
                    make_unified_skymodel,
                    predict_from_skymodel_wsclean,
                )

                LOG.info(
                    "Creating unified skymodel (>%s mJy, radius %.2f deg, center RA=%.6f° Dec=%.6f°)",
                    unicat_min_mjy,
                    unicat_radius_deg,
                    ra0_deg,
                    dec0_deg,
                )
                sky = make_unified_skymodel(
                    center_ra_deg=ra0_deg,
                    center_dec_deg=dec0_deg,
                    radius_deg=unicat_radius_deg,
                    min_mjy=float(unicat_min_mjy),
                    freq_ghz=freq_ghz,
                )
                predict_from_skymodel_wsclean(
                    ms_path=ms_path, skymodel=sky, field=field or "0",
                )
                LOG.info(
                    "Seeded MODEL_DATA with unified skymodel (>%s mJy, radius %.2f deg)",
                    unicat_min_mjy,
                    unicat_radius_deg,
                )

            # Export MODEL_DATA as FITS image if requested
            if export_model_image:
                try:
                    from dsa110_contimg.core.calibration.model import export_model_as_fits

                    output_path = f"{imagename}.nvss_model"
                    LOG.info("Exporting NVSS model image to %s.fits...", output_path)
                    export_model_as_fits(
                        ms_path,
                        output_path,
                        field=field or "0",
                        imsize=512,
                        cell_arcsec=1.0,
                    )
                except Exception as e:
                    LOG.warning("Failed to export NVSS model image: %s", e)
        except Exception as exc:
            LOG.warning("NVSS skymodel seeding skipped: %s", exc)
            import traceback

            LOG.debug("NVSS seeding traceback: %s", traceback.format_exc())

    # If a VP table is supplied, proactively register it as user default for the
    # telescope reported by the MS (and for DSA_110) to satisfy AWProject.
    if vptable and _vpmanager is not None and _msmd is not None:
        try:
            telname = None
            md = _msmd()
            md.open(ms_path)
            try:
                telname = md.telescope()  # pylint: disable=no-member
            finally:
                md.close()
            vp = _vpmanager()
            vp.loadfromtable(vptable)
            for tname in filter(None, [telname, "DSA_110"]):
                try:
                    vp.setuserdefault(telescope=tname)
                except (RuntimeError, ValueError):
                    pass
            LOG.debug(
                "Registered VP table %s for telescope(s): %s",
                vptable,
                [telname, "DSA_110"],
            )
        except Exception as exc:
            LOG.debug("VP preload skipped: %s", exc)

    # Generate mask if requested (before imaging)
    if (
        mask_path is None
        and use_unicat_mask
        and unicat_min_mjy is not None
        and backend == "wsclean"
    ):
        if ra0_deg is not None and dec0_deg is not None:
            try:
                from dsa110_contimg.core.imaging.catalog_tools import create_catalog_fits_mask

                mask_path = create_catalog_fits_mask(
                    catalog="unicat",
                    imagename=imagename,
                    imsize=imsize,
                    cell_arcsec=cell_arcsec,
                    ra0_deg=ra0_deg,
                    dec0_deg=dec0_deg,
                    min_mjy=unicat_min_mjy,
                    radius_arcsec=mask_radius_arcsec,
                )
                LOG.info(
                    "Generated unified catalog mask: %s (radius=%.1f arcsec, sources >= %.1f mJy)",
                    mask_path,
                    mask_radius_arcsec,
                    unicat_min_mjy,
                )
            except Exception as exc:
                LOG.warning(
                    "Failed to generate unified catalog mask, continuing without mask: %s", exc
                )
                import traceback

                LOG.debug("Mask generation traceback: %s", traceback.format_exc())
                mask_path = None
        else:
            LOG.warning("Cannot generate mask: phase center not available")

    # IDG requires a single-SPW MS. If using IDG gridder with a multi-SPW MS,
    # merge SPWs into a temporary single-SPW MS first.
    _idg_merged_ms = None
    if backend == "wsclean" and gridder == "idg":
        with casatables.table(
            os.path.join(ms_path, "SPECTRAL_WINDOW"), readonly=True, ack=False
        ) as _t_spw:
            _n_spws = _t_spw.nrows()
        if _n_spws > 1:
            from pathlib import Path as _P

            from dsa110_contimg.core.conversion.merge_spws import merge_spws

            _idg_merged_ms = str(
                _P(ms_path).parent / f"{_P(ms_path).stem}_idg_merged.ms"
            )
            LOG.info(
                "IDG gridder requires single-SPW MS; merging %d SPWs -> %s",
                _n_spws,
                _idg_merged_ms,
            )
            merge_spws(ms_path, _idg_merged_ms, datacolumn=datacolumn)
            ms_path = _idg_merged_ms
            # After merging, data lives in DATA column regardless of input
            datacolumn = "DATA"

    # Route to appropriate backend
    if backend == "wsclean":
        try:
            run_wsclean(
                ms_path=ms_path,
                imagename=imagename,
                datacolumn=datacolumn,
                field=field,
                imsize=imsize,
                cell_arcsec=cell_arcsec,
                weighting=weighting,
                robust=robust,
                specmode=specmode,
                deconvolver=deconvolver,
                nterms=nterms,
                niter=niter,
                threshold=threshold,
                pbcor=pbcor,
                uvrange=uvrange,
                pblimit=pblimit,
                quality_tier=quality_tier,
                wsclean_path=wsclean_path,
                gridder=gridder,
                mask_path=mask_path,
                target_mask=target_mask,
                galvin_clip_mask=galvin_clip_mask,
                galvin_box_size=galvin_box_size,
                galvin_adaptive_depth=galvin_adaptive_depth,
                erode_beam_shape=erode_beam_shape,
            )
        finally:
            # Clean up temporary IDG-merged MS
            if _idg_merged_ms is not None:
                LOG.info("Removing temporary IDG-merged MS: %s", _idg_merged_ms)
                shutil.rmtree(_idg_merged_ms, ignore_errors=True)
    else:
        # Prepare mask if needed for tclean
        if mask_path or target_mask or galvin_clip_mask or erode_beam_shape:
            from dsa110_contimg.core.imaging.cli_utils import prepare_cleaning_mask

            if mask_path:
                try:
                    from pathlib import Path as PathlibPath

                    prepared_mask = prepare_cleaning_mask(
                        fits_mask=PathlibPath(mask_path),
                        target_mask=PathlibPath(target_mask) if target_mask else None,
                        galvin_clip_mask=(
                            PathlibPath(galvin_clip_mask) if galvin_clip_mask else None
                        ),
                        galvin_box_size=galvin_box_size,
                        galvin_adaptive_depth=galvin_adaptive_depth,
                        erode_beam_shape=erode_beam_shape,
                    )
                    if prepared_mask:
                        mask_path = str(prepared_mask)
                        kwargs["mask"] = mask_path
                        kwargs["usemask"] = "user"
                        LOG.info("Using prepared mask for tclean: %s", mask_path)
                except Exception as e:
                    LOG.warning(
                        "Failed to prepare advanced mask for tclean: %s. Proceeding with default mask behavior.",
                        e,
                    )
            elif target_mask:
                # If only target_mask is provided, we could use it as the mask?
                # For now, only support modifying an existing mask_path.
                LOG.warning("Target mask provided without base mask_path for tclean. Ignoring.")

        # CASA tclean doesn't support FITS masks directly, BUT if we prepared it via prepare_cleaning_mask
        # it is still a FITS file. tclean's 'mask' parameter accepts an image name or a list of regions.
        # If it's a FITS file, tclean might accept it if it's in the right format or needs import.
        # However, `dstools` was using WSClean which takes FITS.
        # CASA tclean usually prefers CASA images or region files.
        # If `mask_path` is a FITS file, tclean *can* sometimes read it, but it's safer to import it.
        # Or let the user rely on 'auto-multithresh' if no mask.

        if mask_path and mask_path.endswith(".fits") and backend == "tclean":
            # Convert FITS mask to CASA image mask if needed?
            # Actually, tclean documentation says 'mask' can be an image name.
            # FITS might work if CASA can read it on the fly, but importfits is safer.
            try:
                try:
                    from dsa110_contimg.common.utils.casa_init import casa_log_environment

                    with casa_log_environment():
                        from casatasks import importfits
                except ImportError:
                    from casatasks import importfits

                casa_mask = mask_path.replace(".fits", ".mask.image")
                if not os.path.exists(casa_mask):
                    importfits(fitsimage=mask_path, imagename=casa_mask, overwrite=True)
                kwargs["mask"] = casa_mask
                kwargs["usemask"] = "user"
            except Exception as e:
                LOG.warning("Failed to convert FITS mask to CASA image: %s", e)

        if mask_path and not kwargs.get("mask"):
            LOG.warning(
                "Masking not supported or failed for CASA tclean backend with provided file."
            )

        from pathlib import Path as PathlibPath

        from dsa110_contimg.common.utils.ms_permissions import (
            ensure_dir_writable,
            ensure_ms_writable,
        )

        ensure_dir_writable(PathlibPath(imagename).parent)
        if kwargs.get("savemodel") and kwargs.get("savemodel") != "none":
            ensure_ms_writable(ms_path)

        t0 = time.perf_counter()
        tclean(**kwargs)  # type: ignore[arg-type]  # CASA uses dynamic kwargs
        LOG.info("tclean completed in %.2fs", time.perf_counter() - t0)

    # QA validation of image products
    try:
        from dsa110_contimg.core.qa.pipeline_quality import check_image_quality

        if backend == "wsclean":
            # WSClean output naming: "-MFS-" only when -channels-out is used (nterms > 1)
            qa_suffix = "-MFS-" if nterms > 1 else "-"
            image_path = imagename + qa_suffix + "image.fits"
            if os.path.isfile(image_path):
                check_image_quality(image_path, alert_on_issues=True)
        else:
            image_path = imagename + ".image"
            if os.path.isdir(image_path):
                check_image_quality(image_path, alert_on_issues=True)
    except Exception as e:
        LOG.warning("QA validation failed: %s", e)

    # Export FITS products if present (only for tclean backend)
    if backend == "tclean" and not skip_fits:
        for suffix in (".image", ".pb", ".pbcor", ".residual", ".model"):
            img = imagename + suffix
            if os.path.isdir(img):
                fits = imagename + suffix + ".fits"
                try:
                    exportfits(imagename=img, fitsimage=fits, overwrite=True)
                except Exception as exc:
                    LOG.debug("exportfits failed for %s: %s", img, exc)
