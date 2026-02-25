"""
Imaging worker: watches a directory of freshly converted 5-minute MS files,
looks up an active calibration apply list from the registry by observation
time, applies calibration, and makes quick continuum images.

This is a first-pass skeleton that can run in one-shot (scan) mode or in a
simple polling loop. It records products in a small SQLite DB for later
mosaicking.

GPU Safety:
    All imaging entry points are wrapped with @memory_safe decorator to ensure
    system RAM limits are respected before processing. This prevents OOM crashes
    that could cause disk disconnection (ref: Dec 2 2025 incident).

GPU Acceleration (Phase 3.3):
    The worker now supports GPU-accelerated dirty imaging via gpu_grid_visibilities().
    This provides ~10x speedup for gridding operations when CuPy is available.
    Falls back to CPU gridding or CASA tclean when GPU is unavailable.
"""

import argparse
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from dsa110_contimg.infrastructure.database import (
    ensure_pipeline_db,
    get_active_applylist,
    images_insert,
    ms_index_upsert,
)
from dsa110_contimg.common.utils.gpu_safety import (
    check_gpu_memory_available,
    gpu_safe,
    initialize_gpu_safety,
    is_gpu_available,
    memory_safe,
)
from dsa110_contimg.core.conversion.ms_utils import (
    configure_ms_for_imaging,
    inject_provenance_metadata,
)
from dsa110_contimg.infrastructure.tracking.provenance import (
    ProvenanceTracker,
    record_provenance,
)

logger = logging.getLogger("imaging_worker")

# Initialize GPU safety limits at module load time
initialize_gpu_safety()

# Check if GPU gridding is available
try:
    from dsa110_contimg.core.imaging.gpu_gridding import (
        GriddingConfig,
        cpu_grid_visibilities,
        gpu_grid_visibilities,
    )

    GPU_GRIDDING_AVAILABLE = True
except ImportError:
    GPU_GRIDDING_AVAILABLE = False
    gpu_grid_visibilities = None  # type: ignore[assignment]
    cpu_grid_visibilities = None  # type: ignore[assignment]
    GriddingConfig = None  # type: ignore[assignment,misc]

try:
    from dsa110_contimg.common.utils.run_isolation import prepare_temp_environment
except ImportError:  # pragma: no cover
    prepare_temp_environment = None  # type: ignore[assignment]


def setup_logging(level: str) -> None:
    """Configure logging with the specified level."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def _get_wavelength_from_ms(ms_path: str) -> float:
    """Get wavelength from MS SPECTRAL_WINDOW table."""
    from dsa110_contimg.common.utils.casa_init import ensure_casa_path

    ensure_casa_path()
    import casacore.tables as casatables

    try:
        with casatables.table(f"{ms_path}/SPECTRAL_WINDOW", readonly=True) as spw:
            ref_freq = spw.getcol("REF_FREQUENCY")[0]  # Hz
            return 299792458.0 / ref_freq  # meters
    except (OSError, KeyError, RuntimeError):
        # Default to 1.4 GHz (21cm line)
        return 0.2142


def _get_weights_from_table(tb, data_shape: tuple[int, ...]) -> np.ndarray:
    """Extract weights from MS table, handling different column formats."""
    colnames = tb.colnames()
    if "WEIGHT_SPECTRUM" in colnames:
        return tb.getcol("WEIGHT_SPECTRUM")
    if "WEIGHT" in colnames:
        weights = tb.getcol("WEIGHT")
        if weights.ndim == 2:  # (n_rows, n_pol)
            return np.broadcast_to(weights[:, np.newaxis, :], data_shape).copy()
        return weights
    return np.ones(data_shape, dtype=np.float32)


def _average_polarizations(
    data: np.ndarray,
    flags: np.ndarray,
    weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Average data over polarizations (Stokes I from XX+YY)."""
    n_pol = data.shape[2]
    if n_pol >= 2:
        vis_avg = 0.5 * (data[:, :, 0] + data[:, :, -1])
        flag_avg = flags[:, :, 0] | flags[:, :, -1]
        wt_avg = 0.5 * (weights[:, :, 0] + weights[:, :, -1])
    else:
        vis_avg = data[:, :, 0]
        flag_avg = flags[:, :, 0]
        wt_avg = weights[:, :, 0]
    return vis_avg, flag_avg, wt_avg


def _read_ms_visibilities(
    ms_path: str,
    datacolumn: str = "CORRECTED_DATA",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Read visibilities from MS for GPU gridding."""
    from dsa110_contimg.common.utils.casa_init import ensure_casa_path

    ensure_casa_path()
    import casacore.tables as casatables

    wavelength = _get_wavelength_from_ms(ms_path)

    with casatables.table(ms_path, readonly=True) as tb:
        uvw = tb.getcol("UVW") / wavelength  # Convert to wavelengths
        data = tb.getcol(datacolumn)
        flags = tb.getcol("FLAG")
        weights = _get_weights_from_table(tb, data.shape)

    # Average polarizations and channels
    vis_avg, flag_avg, wt_avg = _average_polarizations(data, flags, weights)
    vis_flat = np.nanmean(vis_avg, axis=1)
    flag_flat = np.any(flag_avg, axis=1)
    wt_flat = np.nanmean(wt_avg, axis=1)

    return uvw, vis_flat, wt_flat, flag_flat


def _run_gridding(
    uvw: np.ndarray,
    vis: np.ndarray,
    weights: np.ndarray,
    flags: np.ndarray,
    config,
    gpu_id: int,
) -> tuple:
    """Run gridding on GPU or CPU based on availability."""
    gpu_ok, gpu_reason = check_gpu_memory_available(2.0)
    use_gpu = gpu_ok and is_gpu_available()

    if use_gpu:
        logger.info("Using GPU %d for gridding", gpu_id)
        result = gpu_grid_visibilities(
            uvw, vis, weights, config=config, flags=flags.astype(np.int32)
        )
    else:
        logger.info("Using CPU for gridding (reason: %s)", gpu_reason)
        result = cpu_grid_visibilities(
            uvw, vis, weights, config=config, flags=flags.astype(np.int32)
        )

    return result, use_gpu


def _save_dirty_fits(
    result,
    output_path: str,
    image_size: int,
    cell_size_arcsec: float,
    use_gpu: bool,
    gpu_id: int,
) -> str:
    """Save gridding result to FITS file."""
    from astropy.io import fits as pyfits

    fits_path = f"{output_path}.dirty.fits"
    hdu = pyfits.PrimaryHDU(result.image.astype(np.float32))
    hdu.header["BUNIT"] = "JY/BEAM"
    hdu.header["CDELT1"] = -cell_size_arcsec / 3600.0
    hdu.header["CDELT2"] = cell_size_arcsec / 3600.0
    hdu.header["CRPIX1"] = image_size / 2 + 1
    hdu.header["CRPIX2"] = image_size / 2 + 1
    hdu.header["CTYPE1"] = "RA---SIN"
    hdu.header["CTYPE2"] = "DEC--SIN"
    hdu.header["NVIS"] = result.n_vis
    hdu.header["NFLAG"] = result.n_flagged
    hdu.header["WSUM"] = result.weight_sum
    hdu.header["GPU"] = use_gpu
    hdu.header["GPUID"] = gpu_id if use_gpu else -1
    hdu.header["PROCTIME"] = result.processing_time_s
    hdu.writeto(fits_path, overwrite=True)
    return fits_path


@gpu_safe(max_gpu_gb=9.0, max_system_gb=6.0)
def gpu_dirty_image(
    ms_path: str,
    output_path: str,
    *,
    image_size: int = 512,
    cell_size_arcsec: float = 12.0,
    gpu_id: int = 0,
    datacolumn: str = "CORRECTED_DATA",
) -> str | None:
    """Create dirty image using GPU gridding."""
    if not GPU_GRIDDING_AVAILABLE:
        logger.warning("GPU gridding not available, skipping GPU dirty image")
        return None

    start_time = time.time()
    logger.info("GPU dirty imaging %s -> %s", ms_path, output_path)

    try:
        uvw, vis, weights, flags = _read_ms_visibilities(ms_path, datacolumn)
        n_vis, n_flag = len(vis), int(np.sum(flags))
        logger.info(
            "Read %d visibilities, %d flagged (%.1f%%)", n_vis, n_flag, 100.0 * n_flag / n_vis
        )

        config = GriddingConfig(
            image_size=image_size,
            cell_size_arcsec=cell_size_arcsec,
            gpu_id=gpu_id,
        )

        result, use_gpu = _run_gridding(uvw, vis, weights, flags, config, gpu_id)

        if result.error:
            logger.error("Gridding failed: %s", result.error)
            return None

        fits_path = _save_dirty_fits(
            result, output_path, image_size, cell_size_arcsec, use_gpu, gpu_id
        )

        elapsed = time.time() - start_time
        logger.info(
            "GPU dirty image complete in %.2fs (gridding: %.2fs)", elapsed, result.processing_time_s
        )
        return fits_path

    except (OSError, RuntimeError, ValueError) as exc:
        logger.error("GPU dirty imaging failed: %s", exc)
        return None


def _setup_temp_environment(out_dir: Path) -> None:
    """Set up temp environment for imaging."""
    try:
        if prepare_temp_environment is not None:
            from dsa110_contimg.common.utils.paths import resolve_paths

            prepare_temp_environment(
                str(resolve_paths().tmpfs_dir),
                cwd_to=os.fspath(out_dir),
            )
    except (OSError, RuntimeError):
        pass


def _submit_imaging_tasks(
    executor,
    ms_path: str,
    imgroot: Path,
    out_dir: Path,
    use_gpu: bool,
    *,
    use_catalog_mask: bool = True,
    catalog_name: str = "unicat",
    catalog_min_flux_mjy: float | None = None,
    catalog_mask_radius_arcsec: float = 60.0,
):
    """Submit imaging tasks to executor.

    Parameters
    ----------
    executor : ThreadPoolExecutor
        Executor for parallel task submission
    ms_path : str
        Path to measurement set
    imgroot : Path
        Root path for output images
    out_dir : Path
        Output directory for products
    use_gpu : bool
        Whether to use GPU acceleration
    use_catalog_mask : bool, optional
        Enable catalog-based masking for CLEAN (default: True)
    catalog_name : str, optional
        Catalog to use for masking: 'unicat', 'nvss', 'first', 'vlass', etc. (default: 'unicat')
    catalog_min_flux_mjy : float, optional
        Minimum flux threshold in mJy for catalog sources (default: None, uses catalog default)
    catalog_mask_radius_arcsec : float, optional
        Radius in arcseconds for circular masks around catalog sources (default: 60.0)
    """
    # Lazy import to avoid triggering casatools auto-update at module import time
    from dsa110_contimg.core.imaging.cli import image_ms
    from dsa110_contimg.core.imaging.fast_imaging import run_fast_imaging

    # Configure catalog masking parameters for image_ms
    # The image_ms function already has built-in catalog masking support
    imaging_kwargs = {
        "imagename": str(imgroot),
        "field": "",
        "quality_tier": "standard",
        "skip_fits": True,
    }

    if use_catalog_mask:
        # Enable catalog-based masking using the existing functionality in image_ms
        # This leverages the catalog_tools.py infrastructure already in place
        imaging_kwargs.update({
            "use_unicat_mask": True,
            "unicat_min_mjy": catalog_min_flux_mjy,
            "mask_radius_arcsec": catalog_mask_radius_arcsec,
        })
        logger.info(
            "Catalog masking enabled: catalog=%s, min_flux=%.1f mJy, radius=%.1f arcsec",
            catalog_name,
            catalog_min_flux_mjy or 0.0,
            catalog_mask_radius_arcsec,
        )
    else:
        imaging_kwargs["use_unicat_mask"] = False
        logger.info("Catalog masking disabled")

    future_deep = executor.submit(
        image_ms,
        ms_path,
        **imaging_kwargs,
    )

    future_fast = executor.submit(
        run_fast_imaging,
        ms_path,
        interval_seconds=None,
        threshold_sigma=6.0,
        datacolumn="CORRECTED_DATA",
        work_dir=str(out_dir),
    )

    future_gpu = None
    if use_gpu and GPU_GRIDDING_AVAILABLE:
        future_gpu = executor.submit(
            gpu_dirty_image,
            ms_path,
            str(imgroot),
            image_size=512,
            cell_size_arcsec=12.0,
        )

    return future_deep, future_fast, future_gpu


def _wait_for_imaging_results(future_deep, future_fast, future_gpu, artifacts: list[str]) -> None:
    """Wait for imaging futures and collect results."""
    # Deep imaging is critical
    try:
        future_deep.result()
    except (RuntimeError, OSError, ValueError) as e:
        logger.error("Deep imaging failed: %s", e)
        raise

    # Fast imaging is auxiliary
    try:
        future_fast.result()
    except Exception as e:
        logger.warning("Fast imaging failed (non-fatal): %s", e)

    # GPU dirty image is auxiliary
    if future_gpu is not None:
        try:
            gpu_fits = future_gpu.result()
            if gpu_fits and os.path.exists(gpu_fits):
                artifacts.append(gpu_fits)
                logger.info("GPU dirty image created: %s", gpu_fits)
        except (RuntimeError, OSError, ValueError) as e:
            logger.warning("GPU dirty imaging failed (non-fatal): %s", e)


@memory_safe(max_system_gb=6.0)
def _apply_and_image(
    ms_path: str,
    out_dir: Path,
    gaintables: list[str],
    *,
    use_gpu: bool = True,
    use_catalog_mask: bool = True,
    catalog_name: str = "unicat",
    catalog_min_flux_mjy: float | None = None,
    catalog_mask_radius_arcsec: float = 60.0,
) -> list[str]:
    """Apply calibration and produce images; returns artifact paths.

    Parameters
    ----------
    ms_path : str
        Path to measurement set
    out_dir : Path
        Output directory for products
    gaintables : list[str]
        List of calibration tables to apply
    use_gpu : bool, optional
        Enable GPU acceleration (default: True)
    use_catalog_mask : bool, optional
        Enable catalog-based masking for CLEAN (default: True)
    catalog_name : str, optional
        Catalog to use for masking (default: 'unicat')
    catalog_min_flux_mjy : float, optional
        Minimum flux threshold in mJy for catalog sources (default: None)
    catalog_mask_radius_arcsec : float, optional
        Radius in arcseconds for circular masks around catalog sources (default: 60.0)
    """
    artifacts: list[str] = []
    _setup_temp_environment(out_dir)

    try:
        from dsa110_contimg.core.calibration.applycal import apply_to_target

        apply_to_target(ms_path, field="", gaintables=gaintables, calwt=True)
        imgroot = out_dir / (Path(ms_path).stem + ".img")

        with ThreadPoolExecutor(max_workers=3) as executor:
            future_deep, future_fast, future_gpu = _submit_imaging_tasks(
                executor,
                ms_path,
                imgroot,
                out_dir,
                use_gpu,
                use_catalog_mask=use_catalog_mask,
                catalog_name=catalog_name,
                catalog_min_flux_mjy=catalog_min_flux_mjy,
                catalog_mask_radius_arcsec=catalog_mask_radius_arcsec,
            )
            _wait_for_imaging_results(future_deep, future_fast, future_gpu, artifacts)

        # Collect CASA artifacts
        for ext in [".image", ".image.pbcor", ".residual", ".psf", ".pb"]:
            p = f"{imgroot}{ext}"
            if os.path.exists(p):
                artifacts.append(p)

    except (RuntimeError, OSError, ValueError) as e:
        logger.error("apply/image failed for %s: %s", ms_path, e)

    return artifacts


def _get_ms_time_info(ms: Path) -> tuple[float | None, float | None, float]:
    """Extract time info from MS, with fallback to current time."""
    from dsa110_contimg.common.utils.time_utils import extract_ms_time_range

    start_mjd, end_mjd, mid_mjd = extract_ms_time_range(os.fspath(ms))
    if mid_mjd is None:
        from astropy.time import Time

        mid_mjd = Time.now().mjd
    return start_mjd, end_mjd, mid_mjd


def _record_ms_status(conn, ms: Path, start_mjd, end_mjd, mid_mjd: float, status: str) -> None:
    """Record MS processing status in database."""
    ms_index_upsert(
        conn,
        os.fspath(ms),
        start_mjd=start_mjd,
        end_mjd=end_mjd,
        mid_mjd=mid_mjd,
        processed_at=time.time(),
        status=status,
    )
    conn.commit()


def _process_single_ms(
    ms: Path,
    out_dir: Path,
    registry_db: Path,
    conn,
    *,
    use_catalog_mask: bool = True,
    catalog_name: str = "unicat",
    catalog_min_flux_mjy: float | None = None,
    catalog_mask_radius_arcsec: float = 60.0,
) -> bool:
    """Process a single MS file. Returns True if processed.

    Parameters
    ----------
    ms : Path
        Path to measurement set
    out_dir : Path
        Output directory for products
    registry_db : Path
        Path to calibration registry database
    conn :
        Database connection
    use_catalog_mask : bool, optional
        Enable catalog-based masking (default: True)
    catalog_name : str, optional
        Catalog to use for masking (default: 'unicat')
    catalog_min_flux_mjy : float, optional
        Minimum flux threshold in mJy (default: None)
    catalog_mask_radius_arcsec : float, optional
        Mask radius in arcseconds (default: 60.0)
    """
    start_mjd, end_mjd, mid_mjd = _get_ms_time_info(ms)

    # --- Provenance Injection (DeepMind-style Rigor) ---
    try:
        # Initialize tracker with unique job ID
        # In a real system, this ID might come from a task queue, but UUID is fine here
        import uuid
        job_id = str(uuid.uuid4())

        # Capture current config state (implicit or explicit)
        # For now, we capture the runtime args and environment as the 'config'
        # Ideally, this should come from a UnifiedPipelineConfig object
        config_state = {
            "worker_version": "1.0.0",
            "registry_db": str(registry_db),
            "out_dir": str(out_dir),
            "gpu_gridding": GPU_GRIDDING_AVAILABLE,
            "catalog_masking": use_catalog_mask,
            "catalog_name": catalog_name if use_catalog_mask else None,
            "timestamp": time.time()
        }

        tracker = ProvenanceTracker(job_id=job_id)
        tracker.set_config(config_state)

        # Inject metadata into MS HISTORY immediately
        # This binds the data to the provenance record forever
        if tracker.provenance.config_hash:
            inject_provenance_metadata(
                os.fspath(ms),
                job_id,
                tracker.provenance.config_hash
            )
            logger.info(f"Injected provenance metadata into {ms} (Job: {job_id})")

        tracker.save() # Persist initial record

    except Exception as e:
        logger.warning(f"Provenance injection failed for {ms}: {e}")
        # Non-fatal, proceed with processing
    # ---------------------------------------------------

    applylist = get_active_applylist(registry_db, mid_mjd)
    if not applylist:
        logger.warning("No active caltables for %s (mid MJD %.5f)", ms, mid_mjd)
        _record_ms_status(conn, ms, start_mjd, end_mjd, mid_mjd, "skipped_no_caltables")
        return False

    artifacts = _apply_and_image(
        os.fspath(ms),
        out_dir,
        applylist,
        use_catalog_mask=use_catalog_mask,
        catalog_name=catalog_name,
        catalog_min_flux_mjy=catalog_min_flux_mjy,
        catalog_mask_radius_arcsec=catalog_mask_radius_arcsec,
    )
    status = "done" if artifacts else "failed"
    _record_ms_status(conn, ms, start_mjd, end_mjd, mid_mjd, status)

    for art in artifacts:
        img_type = "pbcor" if art.endswith(".image.pbcor") else "5min"
        images_insert(conn, art, os.fspath(ms), img_type, created_at=time.time())
    conn.commit()

    logger.info("Processed %s (artifacts: %d)", ms, len(artifacts))
    return True


@memory_safe(max_system_gb=6.0)
def process_once(
    ms_dir: Path,
    out_dir: Path,
    registry_db: Path,
    products_db: Path,
    *,
    use_catalog_mask: bool = True,
    catalog_name: str = "unicat",
    catalog_min_flux_mjy: float | None = None,
    catalog_mask_radius_arcsec: float = 60.0,
) -> int:
    """Process all MS files in directory once.

    Parameters
    ----------
    ms_dir : Path
        Directory containing MS files
    out_dir : Path
        Output directory for products
    registry_db : Path
        Path to calibration registry database
    products_db : Path
        Path to products database
    use_catalog_mask : bool, optional
        Enable catalog-based masking (default: True)
    catalog_name : str, optional
        Catalog to use for masking (default: 'unicat')
    catalog_min_flux_mjy : float, optional
        Minimum flux threshold in mJy (default: None)
    catalog_mask_radius_arcsec : float, optional
        Mask radius in arcseconds (default: 60.0)
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    conn = ensure_pipeline_db()
    processed = 0

    for ms in sorted(ms_dir.glob("**/*.ms")):
        row = conn.execute(
            "SELECT status FROM ms_index WHERE path = ?", (os.fspath(ms),)
        ).fetchone()
        if row and row[0] == "done":
            continue

        if _process_single_ms(
            ms,
            out_dir,
            registry_db,
            conn,
            use_catalog_mask=use_catalog_mask,
            catalog_name=catalog_name,
            catalog_min_flux_mjy=catalog_min_flux_mjy,
            catalog_mask_radius_arcsec=catalog_mask_radius_arcsec,
        ):
            processed += 1

    return processed


def cmd_scan(args: argparse.Namespace) -> int:
    """Run one-shot scan of MS directory."""
    setup_logging(args.log_level)
    n = process_once(
        Path(args.ms_dir),
        Path(args.out_dir),
        Path(args.registry_db),
        Path(args.products_db),
        use_catalog_mask=args.catalog_mask,
        catalog_name=args.catalog_name,
        catalog_min_flux_mjy=args.catalog_min_flux_mjy,
        catalog_mask_radius_arcsec=args.catalog_mask_radius_arcsec,
    )
    logger.info("Scan complete: %d MS processed", n)
    return 0 if n >= 0 else 1


def cmd_daemon(args: argparse.Namespace) -> int:
    """Run continuous polling daemon."""
    setup_logging(args.log_level)
    ms_dir = Path(args.ms_dir)
    out_dir = Path(args.out_dir)
    registry_db = Path(args.registry_db)
    products_db = Path(args.products_db)
    poll = float(args.poll_interval)

    while True:
        try:
            process_once(
                ms_dir,
                out_dir,
                registry_db,
                products_db,
                use_catalog_mask=args.catalog_mask,
                catalog_name=args.catalog_name,
                catalog_min_flux_mjy=args.catalog_min_flux_mjy,
                catalog_mask_radius_arcsec=args.catalog_mask_radius_arcsec,
            )
        except (RuntimeError, OSError, ValueError) as e:
            logger.error("Worker loop error: %s", e)
        time.sleep(poll)


def build_parser() -> argparse.ArgumentParser:
    """Build argument parser for CLI."""
    p = argparse.ArgumentParser(description="Imaging worker for 5-min MS")
    sub = p.add_subparsers(dest="cmd")

    # Shared catalog masking arguments
    def add_catalog_masking_args(parser):
        """Add catalog masking arguments to a subparser."""
        parser.add_argument(
            "--catalog-mask",
            action="store_true",
            default=True,
            help="Enable catalog-based masking for CLEAN (default: enabled)",
        )
        parser.add_argument(
            "--no-catalog-mask",
            dest="catalog_mask",
            action="store_false",
            help="Disable catalog-based masking",
        )
        parser.add_argument(
            "--catalog-name",
            default="unicat",
            choices=["unicat", "nvss", "first", "vlass", "atnf", "rax"],
            help="Catalog to use for masking (default: unicat)",
        )
        parser.add_argument(
            "--catalog-min-flux-mjy",
            type=float,
            default=None,
            help="Minimum flux threshold in mJy for catalog sources (default: catalog-specific)",
        )
        parser.add_argument(
            "--catalog-mask-radius-arcsec",
            type=float,
            default=60.0,
            help="Radius in arcseconds for circular masks around catalog sources (default: 60.0)",
        )

    sp = sub.add_parser("scan", help="One-shot scan of an MS directory")
    sp.add_argument("--ms-dir", required=True)
    sp.add_argument("--out-dir", required=True)
    sp.add_argument("--registry-db", required=True)
    sp.add_argument("--products-db", required=True)
    sp.add_argument("--log-level", default="INFO")
    add_catalog_masking_args(sp)
    sp.set_defaults(func=cmd_scan)

    sp = sub.add_parser("daemon", help="Poll and process arriving MS")
    sp.add_argument("--ms-dir", required=True)
    sp.add_argument("--out-dir", required=True)
    sp.add_argument("--registry-db", required=True)
    sp.add_argument("--products-db", required=True)
    sp.add_argument("--poll-interval", type=float, default=60.0)
    sp.add_argument("--log-level", default="INFO")
    add_catalog_masking_args(sp)
    sp.set_defaults(func=cmd_daemon)

    return p


def main(argv: list[str] | None = None) -> int:
    """Main entry point."""
    p = build_parser()
    args = p.parse_args(argv)
    if not hasattr(args, "func"):
        p.print_help()
        return 2
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
