"""
WSClean visibility-domain mosaicking.

This module implements proper radio interferometric mosaicking by:
1. Copying MS files to preserve originals
2. Unifying phase centers to mean meridian using chgcentre
3. Joint deconvolution with WSClean using IDG + beam correction
4. Separate calibration path with phase center at calibrator position

This replaces the image-domain "linear mosaicking" with proper
visibility-domain joint deconvolution.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from astropy.coordinates import SkyCoord

logger = logging.getLogger(__name__)

# EveryBeam configuration
# WSClean on this system is linked against /opt/everybeam/lib
# User-specified path can override via EVERYBEAM_PATH environment variable
EVERYBEAM_PATH = os.environ.get(
    "EVERYBEAM_PATH", "/opt/everybeam"
)
EVERYBEAM_LIB_PATH = Path(EVERYBEAM_PATH) / "lib"


@dataclass
class WSCleanMosaicConfig:
    """Configuration for WSClean mosaicking.

    Attributes
    ----------
    scratch_dir : Path
        Directory for MS copies (preserve originals)
    output_dir : Path
        Directory for output products
    size : int
        Image size in pixels (both dimensions)
    scale : str
        Pixel scale (e.g., "1asec", "2asec")
    niter : int
        Maximum CLEAN iterations
    mgain : float
        Major cycle gain
    auto_threshold : float
        Auto-threshold sigma level
    idg_mode : str
        IDG mode: "cpu" (default, no GPU) or "hybrid"
    parallel_deconvolution : int
        Parallel deconvolution subimage size
    """

    scratch_dir: Path = field(default_factory=lambda: Path("/dev/shm/mosaic"))
    output_dir: Path = field(default_factory=lambda: Path("/stage/dsa110-contimg/mosaics"))
    size: int = 4096
    scale: str = "1asec"
    niter: int = 50000
    mgain: float = 0.6
    auto_threshold: float = 3.0
    idg_mode: str = "cpu"  # No GPU available
    parallel_deconvolution: int = 2000
    local_rms: bool = True
    # Calibration settings
    calibration_preset: str = "standard"  # "standard", "fast", "high_snr", "low_snr"
    calibration_overrides: dict[str, Any] | None = None  # Advanced parameter overrides
    skip_calibration: bool = False  # Skip solve/apply (for pre-calibrated data)


@dataclass
class WSCleanMosaicResult:
    """Result of WSClean mosaic operation."""

    output_path: Path
    pb_corrected_path: Path
    n_ms_files: int
    phase_center_ra_deg: float
    phase_center_dec_deg: float
    median_rms_jy: float = 0.0
    wsclean_log: str | None = None


@dataclass
class CalibrationResult:
    """Result of calibration solve on phase-shifted MS."""

    bandpass_table: Path
    gains_table: Path
    calibrator_name: str
    ms_path: Path


# =============================================================================
# Calibration Helper Functions
# =============================================================================


def _solve_calibration(
    cal_ms: Path,
    caltable_dir: Path,
    preset_name: str = "standard",
    overrides: dict[str, Any] | None = None,
) -> list[str]:
    """Solve calibration on phase-shifted calibrator MS.

    Uses the calibration preset system for configuration.

    Parameters
    ----------
    cal_ms : Path
        Phase-shifted calibration MS (at calibrator position).
    caltable_dir : Path
        Directory to write calibration tables.
    preset_name : str
        Calibration preset name ("standard", "fast", "high_snr", "low_snr").
    overrides : dict or None
        Optional parameter overrides for the preset.

    Returns
    -------
    list[str]
        List of calibration table paths (BP, G).

    Raises
    ------
    CalibrationError
        If calibration solve fails.
    """
    from dsa110_contimg.common.utils.exceptions import CalibrationError
    from dsa110_contimg.core.calibration.presets import get_preset
    from dsa110_contimg.workflow.pipeline.stages.calibration_helpers import (
        solve_calibration_tables,
    )

    logger.info(f"Solving calibration on {cal_ms.name} with preset '{preset_name}'")

    # Get preset and apply overrides
    try:
        preset = get_preset(preset_name)
    except KeyError as e:
        raise CalibrationError(
            f"Unknown calibration preset: {preset_name}",
            ms_path=str(cal_ms),
        ) from e

    if overrides:
        preset = preset.with_overrides(**overrides)

    params = preset.to_dict()

    # Set table prefix in caltable directory
    ms_name = cal_ms.stem
    table_prefix = str(caltable_dir / ms_name)

    try:
        tables = solve_calibration_tables(
            ms_path=str(cal_ms),
            table_prefix=table_prefix,
            params=params,
        )
    except Exception as e:
        raise CalibrationError(
            f"Calibration solve failed for {cal_ms.name}: {e}",
            ms_path=str(cal_ms),
            original_exception=e,
        ) from e

    if not tables:
        raise CalibrationError(
            f"No calibration tables produced for {cal_ms.name}",
            ms_path=str(cal_ms),
        )

    # Collect table paths
    caltables = []
    for table_type in ["BP", "G", "K"]:
        if table_type in tables:
            caltables.append(tables[table_type])
            logger.info(f"  {table_type} table: {tables[table_type]}")

    logger.info(f"Calibration solve complete: {len(caltables)} tables produced")
    return caltables


def _apply_to_mosaic_copies(
    mosaic_copies: list[Path],
    caltables: list[str],
) -> None:
    """Apply calibration tables to all mosaic MS copies.

    Parameters
    ----------
    mosaic_copies : list[Path]
        List of MS copies to apply calibration to.
    caltables : list[str]
        List of calibration table paths.

    Raises
    ------
    CalibrationError
        If calibration apply fails for any MS.
    """
    from dsa110_contimg.common.utils.exceptions import CalibrationError
    from dsa110_contimg.core.calibration.applycal import apply_to_target

    logger.info(f"Applying calibration to {len(mosaic_copies)} mosaic copies")

    for i, ms_copy in enumerate(mosaic_copies, 1):
        logger.info(f"  [{i}/{len(mosaic_copies)}] Applying to {ms_copy.name}")
        try:
            apply_to_target(
                ms_target=str(ms_copy),
                field="",  # All fields
                gaintables=caltables,
                calwt=True,
            )
        except Exception as e:
            raise CalibrationError(
                f"Failed to apply calibration to {ms_copy.name}: {e}",
                ms_path=str(ms_copy),
                original_exception=e,
            ) from e

    logger.info(f"Calibration applied to all {len(mosaic_copies)} mosaic copies")


def get_env_with_everybeam() -> dict[str, str]:
    """Get environment with EveryBeam + WSClean native libs configured."""
    from dsa110_contimg.common.utils.wsclean_utils import build_wsclean_native_env

    env = build_wsclean_native_env()
    ld_path = env.get("LD_LIBRARY_PATH", "")
    everybeam_lib = str(EVERYBEAM_LIB_PATH)

    if everybeam_lib not in ld_path:
        env["LD_LIBRARY_PATH"] = f"{everybeam_lib}:{ld_path}" if ld_path else everybeam_lib

    return env


def copy_ms_to_scratch(
    ms_paths: list[Path],
    scratch_dir: Path,
    subdir: str = "mosaic",
) -> list[Path]:
    """Copy MS files to scratch directory to preserve originals.

    Parameters
    ----------
    ms_paths : list[Path]
        Original MS file paths.
    scratch_dir : Path
        Base scratch directory.
    subdir : str
        Subdirectory name within scratch.

    Returns
    -------
    list[Path]
        Paths to copied MS files in scratch.
    """
    work_dir = scratch_dir / subdir
    work_dir.mkdir(parents=True, exist_ok=True)

    copied_paths = []
    for ms_path in ms_paths:
        dest = work_dir / ms_path.name
        if dest.exists():
            shutil.rmtree(dest)
        logger.info(f"Copying {ms_path} to {dest}")
        shutil.copytree(ms_path, dest)
        copied_paths.append(dest)

    return copied_paths


def get_phase_center_from_ms(ms_path: Path) -> tuple[float, float]:
    """Get phase center (RA, Dec) from MS file.

    Uses the first field's phase center.

    Parameters
    ----------
    ms_path : Path
        Path to Measurement Set.

    Returns
    -------
    tuple[float, float]
        (ra_deg, dec_deg) of phase center.
    """
    try:
        from casacore.tables import table
    except ImportError:
        logger.warning("casacore not available, using fallback phase center extraction")
        # Fallback: try to read from MS FIELD table using casatools
        try:
            from casatools import table as tb

            t = tb()
            t.open(str(ms_path) + "/FIELD")
            phase_dir = t.getcol("PHASE_DIR")
            t.close()
            # Handle variable shape
            if phase_dir.ndim == 3:
                # Shape could be (nfields, 1, 2) or (2, 1, nfields)
                if phase_dir.shape[2] == 2:
                    ra_rad = phase_dir[0, 0, 0]
                    dec_rad = phase_dir[0, 0, 1]
                else:
                    ra_rad = phase_dir[0, 0, 0]
                    dec_rad = phase_dir[1, 0, 0]
            else:
                ra_rad = phase_dir[0, 0]
                dec_rad = phase_dir[0, 1]
            return np.degrees(ra_rad), np.degrees(dec_rad)
        except Exception as e:
            logger.error(f"Failed to get phase center from {ms_path}: {e}")
            raise

    t = table(str(ms_path) + "/FIELD", readonly=True, ack=False)
    phase_dir = t.getcol("PHASE_DIR")
    t.close()

    # phase_dir shape is (nfields, 1, 2) - [field, polynomial, coord]
    # Get first field's phase center
    ra_rad = phase_dir[0, 0, 0]
    dec_rad = phase_dir[0, 0, 1]

    return np.degrees(ra_rad), np.degrees(dec_rad)


def compute_mean_meridian(ms_paths: list[Path]) -> tuple[float, float]:
    """Compute mean meridian RA from list of MS files.

    For drift-scan mosaicking, the unified phase center should be
    the mean RA of all pointings at the common declination.

    Parameters
    ----------
    ms_paths : list[Path]
        List of MS file paths.

    Returns
    -------
    tuple[float, float]
        (mean_ra_deg, dec_deg) - mean RA and declination from first MS.
    """
    ra_values = []
    dec_values = []

    for ms_path in ms_paths:
        ra, dec = get_phase_center_from_ms(ms_path)
        ra_values.append(ra)
        dec_values.append(dec)

    mean_ra = np.mean(ra_values)
    # Use first Dec (should all be same for drift scan)
    dec = dec_values[0]

    logger.info(
        f"Mean meridian: RA={mean_ra:.4f}° Dec={dec:.4f}° "
        f"(from {len(ms_paths)} MS files, RA range: {min(ra_values):.4f}-{max(ra_values):.4f}°)"
    )

    return mean_ra, dec


def run_chgcentre(ms_path: Path, ra_deg: float, dec_deg: float) -> None:
    """Run chgcentre to shift phase center of MS.

    Parameters
    ----------
    ms_path : Path
        Path to Measurement Set (will be modified in-place).
    ra_deg : float
        Target RA in degrees.
    dec_deg : float
        Target Dec in degrees.
    """
    # Convert degrees to sexagesimal for chgcentre
    from astropy.coordinates import Angle

    ra_hms = Angle(ra_deg, unit="deg").to_string(unit="hour", sep="hms", precision=1)
    dec_dms = Angle(dec_deg, unit="deg").to_string(unit="deg", sep="dms", precision=0)

    # chgcentre expects format like "01h37m41.3s +33d09m35s"
    cmd = ["chgcentre", str(ms_path), ra_hms, dec_dms]

    logger.info(f"Running: {' '.join(cmd)}")

    env = get_env_with_everybeam()
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )

    if result.returncode != 0:
        logger.error(f"chgcentre failed: {result.stderr}")
        raise RuntimeError(f"chgcentre failed: {result.stderr}")

    logger.info(f"Phase-shifted {ms_path.name} to RA={ra_deg:.4f}° Dec={dec_deg:.4f}°")


def run_wsclean_mosaic(
    ms_paths: list[Path],
    output_name: str,
    config: WSCleanMosaicConfig,
) -> WSCleanMosaicResult:
    """Run WSClean joint deconvolution on multiple MS files.

    Parameters
    ----------
    ms_paths : list[Path]
        List of phase-shifted MS files (should have unified phase center).
    output_name : str
        Base name for output files.
    config : WSCleanMosaicConfig
        WSClean configuration.

    Returns
    -------
    WSCleanMosaicResult
        Result with output paths and metadata.
    """
    config.output_dir.mkdir(parents=True, exist_ok=True)
    output_prefix = config.output_dir / output_name

    # Build WSClean command
    cmd = [
        "wsclean",
        "-use-idg",
        f"-idg-mode", config.idg_mode,
        "-grid-with-beam",
        "-name", str(output_prefix),
        "-size", str(config.size), str(config.size),
        "-scale", config.scale,
        "-niter", str(config.niter),
        "-mgain", str(config.mgain),
        "-auto-threshold", str(config.auto_threshold),
        "-parallel-deconvolution", str(config.parallel_deconvolution),
    ]

    if config.local_rms:
        cmd.append("-local-rms")

    # Add all MS files
    cmd.extend([str(ms) for ms in ms_paths])

    logger.info(f"Running WSClean: {' '.join(cmd[:15])}... ({len(ms_paths)} MS files)")

    env = get_env_with_everybeam()
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        env=env,
        check=False,
        cwd=str(config.output_dir),
    )

    if result.returncode != 0:
        logger.error(f"WSClean failed: {result.stderr}")
        raise RuntimeError(f"WSClean failed: {result.stderr}")

    # Find output files
    image_path = Path(f"{output_prefix}-image.fits")
    pb_path = Path(f"{output_prefix}-image-pb.fits")

    if not image_path.exists():
        # Try MFS suffix
        image_path = Path(f"{output_prefix}-MFS-image.fits")
        pb_path = Path(f"{output_prefix}-MFS-image-pb.fits")

    if not image_path.exists():
        raise FileNotFoundError(f"WSClean output not found: {output_prefix}-image.fits")

    # Get phase center from first MS
    phase_center_ra, phase_center_dec = get_phase_center_from_ms(ms_paths[0])

    # Compute RMS from (PB-corrected) output image
    from .builder import compute_rms

    rms_path = pb_path if pb_path.exists() else image_path
    try:
        from astropy.io import fits

        with fits.open(rms_path) as hdul:
            data = hdul[0].data
            # Handle 4D (Stokes, freq) or 2D data
            if data.ndim == 4:
                data = data[0, 0, :, :]
            elif data.ndim == 3:
                data = data[0, :, :]
            median_rms_jy = compute_rms(data)
    except Exception as e:
        logger.warning(f"Could not compute RMS from {rms_path}: {e}")
        median_rms_jy = 0.0

    return WSCleanMosaicResult(
        output_path=image_path,
        pb_corrected_path=pb_path if pb_path.exists() else image_path,
        n_ms_files=len(ms_paths),
        phase_center_ra_deg=phase_center_ra,
        phase_center_dec_deg=phase_center_dec,
        median_rms_jy=median_rms_jy,
        wsclean_log=result.stdout,
    )


def build_wsclean_mosaic(
    ms_paths: list[Path],
    output_name: str,
    config: WSCleanMosaicConfig | None = None,
    calibrator_ms_idx: int | None = None,
    calibrator_position: tuple[float, float] | None = None,
) -> WSCleanMosaicResult:
    """Build mosaic using WSClean visibility-domain joint deconvolution.

    .. note:: **Tier: SCIENCE / DEEP**

        This function performs proper radio interferometric mosaicking via
        joint deconvolution across all pointings. It is the recommended
        approach for publication-quality science products and deep integrations.

        For QUICKLOOK tier (fast monitoring, no MS access),
        use :func:`~dsa110_contimg.core.mosaic.builder.build_mosaic` instead.

    This is the main entry point for WSClean mosaicking. It:
    1. Copies MS files to scratch (preserves originals)
    2. Optionally phase-shifts calibration MS to calibrator position
    3. Phase-shifts all mosaic MS copies to mean meridian
    4. Runs WSClean with IDG + beam correction

    Parameters
    ----------
    ms_paths : list[Path]
        List of original MS file paths.
    output_name : str
        Base name for output mosaic.
    config : WSCleanMosaicConfig or None
        Configuration. Uses defaults if None.
    calibrator_ms_idx : int or None
        Index of the transit MS to use for calibration.
        If provided, this MS will be phase-shifted to calibrator_position
        for calibration, while a separate copy is used for mosaicking.
    calibrator_position : tuple[float, float] or None
        (RA_deg, Dec_deg) of calibrator for phase centering.
        Required if calibrator_ms_idx is provided.

    Returns
    -------
    WSCleanMosaicResult
        Result with output paths and metadata.

    Examples
    --------
    >>> # Simple 3-tile mosaic
    >>> result = build_wsclean_mosaic(
    ...     ms_paths=[Path("tile1.ms"), Path("tile2.ms"), Path("tile3.ms")],
    ...     output_name="my_mosaic",
    ... )

    >>> # With calibration on transit tile
    >>> result = build_wsclean_mosaic(
    ...     ms_paths=[Path("tile1.ms"), Path("tile2.ms"), Path("tile3.ms")],
    ...     output_name="3c48_mosaic",
    ...     calibrator_ms_idx=1,  # Transit tile
    ...     calibrator_position=(24.4221, 33.1597),  # 3C48
    ... )
    """
    if config is None:
        config = WSCleanMosaicConfig()

    if len(ms_paths) == 0:
        raise ValueError("No MS files provided")

    logger.info(f"Building WSClean mosaic from {len(ms_paths)} MS files")

    # 1. Copy all MS files to scratch for mosaic
    mosaic_copies = copy_ms_to_scratch(
        ms_paths, config.scratch_dir, subdir=f"{output_name}_mosaic"
    )

    # 2. If calibration requested, make separate copy for calibration
    if calibrator_ms_idx is not None:
        if calibrator_position is None:
            raise ValueError("calibrator_position required when calibrator_ms_idx is set")

        cal_copies = copy_ms_to_scratch(
            [ms_paths[calibrator_ms_idx]],
            config.scratch_dir,
            subdir=f"{output_name}_cal",
        )
        cal_ms = cal_copies[0]

        # Phase-shift calibration MS to calibrator position
        run_chgcentre(cal_ms, calibrator_position[0], calibrator_position[1])
        logger.info(
            f"Calibration MS phase-shifted to calibrator: "
            f"RA={calibrator_position[0]:.4f}° Dec={calibrator_position[1]:.4f}°"
        )

        # Solve calibration and apply to mosaic copies
        if not config.skip_calibration:
            caltable_dir = config.scratch_dir / f"{output_name}_caltables"
            caltable_dir.mkdir(parents=True, exist_ok=True)

            caltables = _solve_calibration(
                cal_ms=cal_ms,
                caltable_dir=caltable_dir,
                preset_name=config.calibration_preset,
                overrides=config.calibration_overrides,
            )

            _apply_to_mosaic_copies(
                mosaic_copies=mosaic_copies,
                caltables=caltables,
            )

            logger.info(
                f"Calibration complete: solved from {cal_ms.name}, "
                f"applied to {len(mosaic_copies)} mosaic copies"
            )
        else:
            logger.info("Skipping calibration (skip_calibration=True)")

    # 3. Compute mean meridian phase center
    mean_ra, dec = compute_mean_meridian(ms_paths)

    # 4. Phase-shift all mosaic copies to mean meridian
    for ms_copy in mosaic_copies:
        run_chgcentre(ms_copy, mean_ra, dec)

    # 5. Run WSClean joint deconvolution
    result = run_wsclean_mosaic(mosaic_copies, output_name, config)

    logger.info(
        f"WSClean mosaic complete: {result.pb_corrected_path} "
        f"({result.n_ms_files} MS files)"
    )

    # Inject provenance if possible (using FITS headers)
    # Since mosaic is a FITS file, we use astropy to add keywords
    try:
        from dsa110_contimg.infrastructure.tracking.provenance import ProvenanceTracker
        import uuid
        import time
        from astropy.io import fits

        job_id = str(uuid.uuid4())
        tracker = ProvenanceTracker(job_id=job_id)
        tracker.set_config({
            "service": "mosaic_worker",
            "inputs": [str(p) for p in ms_paths],
            "timestamp": time.time()
        })
        tracker.save()

        with fits.open(result.pb_corrected_path, mode='update') as hdul:
            hdul[0].header['PROV_JID'] = (job_id, 'DSA-110 Job ID')
            hdul[0].header['PROV_HSH'] = (tracker.provenance.config_hash[:16], 'Config Hash')
    except Exception as e:
        logger.warning(f"Failed to inject mosaic provenance: {e}")

    return result


def cleanup_scratch(scratch_dir: Path, output_name: str) -> None:
    """Clean up scratch copies after mosaicking.

    Parameters
    ----------
    scratch_dir : Path
        Base scratch directory.
    output_name : str
        Mosaic output name (used for subdirectory names).
    """
    for subdir in [f"{output_name}_mosaic", f"{output_name}_cal"]:
        work_dir = scratch_dir / subdir
        if work_dir.exists():
            logger.info(f"Cleaning up {work_dir}")
            shutil.rmtree(work_dir)
