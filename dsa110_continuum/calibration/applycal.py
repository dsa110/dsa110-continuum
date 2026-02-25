"""
Apply calibration tables to target measurement sets.

GPU Safety:
    Entry point apply_to_target() is wrapped with @memory_safe to ensure
    system RAM limits are respected before processing. Calibration is memory-
    intensive and can cause OOM on large datasets.

GPU Acceleration (Phase 3.3):
    The module now supports GPU-accelerated gain application via apply_gains().
    This provides ~10x speedup for large datasets when CuPy is available.
    Falls back to CASA applycal or CPU when GPU is unavailable.
"""

import logging
from typing import Optional

import numpy as np

# Ensure CASAPATH is set before importing CASA modules
from dsa110_contimg.common.utils.casa_init import ensure_casa_path

ensure_casa_path()

# CASA import moved to function level to prevent logs in workspace root
# See: docs/dev-notes/analysis/casa_log_handling_investigation.md

from dsa110_contimg.core.calibration.casa_service import CASAService
from dsa110_contimg.core.calibration.validate import (
    validate_caltables_for_use,
)
from dsa110_contimg.common.utils import timed
from dsa110_contimg.common.utils.gpu_safety import (
    check_gpu_memory_available,
    gpu_safe,
    initialize_gpu_safety,
    is_gpu_available,
    memory_safe,
)
from dsa110_contimg.common.utils.ms_permissions import ensure_ms_writable

logger = logging.getLogger("applycal")

# Initialize GPU safety limits at module load time
initialize_gpu_safety()

# Check if GPU calibration is available
try:
    from dsa110_contimg.core.calibration.gpu_calibration import (
        ApplyCalResult,
        apply_gains,
    )

    GPU_CALIBRATION_AVAILABLE = True
except ImportError:
    GPU_CALIBRATION_AVAILABLE = False
    apply_gains = None  # type: ignore[assignment]
    ApplyCalResult = None  # type: ignore[assignment]


# =============================================================================
# Issue #2: Interpolated Calibration Application
# =============================================================================


def apply_interpolated_calibration(
    ms_target: str,
    field: str,
    paths_before: list[str],
    paths_after: list[str],
    weight_before: float,
    *,
    calwt: bool = True,
    verify: bool = True,
) -> None:
    """Apply interpolated calibration from two calibration sets.

    This addresses Issue #2: The pipeline only interpolates within a single
    calibration table, not between different calibration observations.

    Strategy:
    1. Read gains from both calibration sets
    2. Compute weighted average of complex gains
    3. Write merged gains to temporary calibration table
    4. Apply merged table to target MS

    The weight_before parameter determines how much weight to give the
    "before" calibration set (weight_after = 1.0 - weight_before).

    Parameters
    ----------
    ms_target :
        Path to target Measurement Set
    field :
        Field to calibrate (empty string for all)
    paths_before :
        Calibration table paths from earlier observation
    paths_after :
        Calibration table paths from later observation
    weight_before :
        Weight for "before" set (0.0-1.0)
    calwt :
        Whether to calibrate weights
    verify :
        Whether to verify CORRECTED_DATA after application

    Raises
    ------
    ValueError
        If weights are invalid or no paths provided
    RuntimeError
        If interpolation fails

    """
    import tempfile
    from pathlib import Path

    if not 0.0 <= weight_before <= 1.0:
        raise ValueError(f"weight_before must be in [0.0, 1.0], got {weight_before}")

    # Handle edge cases - pure single-set application
    if weight_before >= 0.999 or not paths_after:
        if not paths_before:
            raise ValueError("No calibration paths provided")
        logger.info("Applying single calibration set (before), weight=%.3f", weight_before)
        apply_to_target(ms_target, field, paths_before, calwt=calwt, verify=verify)
        return

    if weight_before <= 0.001 or not paths_before:
        if not paths_after:
            raise ValueError("No calibration paths provided")
        logger.info("Applying single calibration set (after), weight=%.3f", 1.0 - weight_before)
        apply_to_target(ms_target, field, paths_after, calwt=calwt, verify=verify)
        return

    # True interpolation case
    weight_after = 1.0 - weight_before
    logger.info(
        "Applying interpolated calibration: before=%.1f%%, after=%.1f%%",
        weight_before * 100,
        weight_after * 100,
    )

    # Match table types between before and after sets
    # Tables are ordered by apply order (K, BA, BP, GA, GP, etc.)
    # We need to interpolate matching types

    # Group tables by type suffix
    def get_table_type(path: str) -> str:
        """Extract table type from path suffix.

        Parameters
        ----------
        """
        name = Path(path).name.lower()
        for suffix in [
            ".k",
            "_kcal",
            ".2k",
            "_2kcal",
            ".b",
            "_bacal",
            "_bpcal",
            "_gacal",
            ".g",
            "_gpcal",
            ".2g",
            "_2gcal",
            "_fluxcal",
            ".prebp",
            "_prebp_phase",
        ]:
            if name.endswith(suffix):
                return suffix
        return "_unknown"

    before_by_type = {get_table_type(p): p for p in paths_before}
    after_by_type = {get_table_type(p): p for p in paths_after}

    # Find common table types for interpolation
    common_types = set(before_by_type.keys()) & set(after_by_type.keys())

    if not common_types:
        logger.warning(
            "No common table types between before and after sets. Falling back to before set only."
        )
        apply_to_target(ms_target, field, paths_before, calwt=calwt, verify=verify)
        return

    # Create temporary directory for merged tables
    with tempfile.TemporaryDirectory(prefix="calmrg_") as tmpdir:
        merged_paths: list[str] = []

        for table_type in sorted(common_types):
            before_path = before_by_type[table_type]
            after_path = after_by_type[table_type]
            merged_path = str(Path(tmpdir) / f"merged{table_type}")

            try:
                _merge_caltables_weighted(
                    before_path, after_path, merged_path, weight_before, weight_after
                )
                merged_paths.append(merged_path)
                logger.debug(
                    "Merged %s: %s + %s -> %s", table_type, before_path, after_path, merged_path
                )
            except (OSError, RuntimeError, ValueError) as e:
                logger.warning("Failed to merge %s tables, using before only: %s", table_type, e)
                merged_paths.append(before_path)

        # Add any non-common tables from before set
        for table_type, path in before_by_type.items():
            if table_type not in common_types:
                merged_paths.append(path)
                logger.debug("Including non-interpolated table from before: %s", path)

        # Apply merged calibration
        if merged_paths:
            apply_to_target(ms_target, field, merged_paths, calwt=calwt, verify=verify)
        else:
            raise RuntimeError("No merged calibration tables to apply")


def _merge_caltables_weighted(
    path_before: str,
    path_after: str,
    output_path: str,
    weight_before: float,
    weight_after: float,
) -> None:
    """Merge two calibration tables with weighted average of gains.

    Creates a new calibration table with gains that are a weighted
    average of the input tables.

    Parameters
    ----------
    path_before :
        Path to earlier calibration table
    path_after :
        Path to later calibration table
    output_path :
        Path to write merged table
    weight_before :
        Weight for earlier table (0.0-1.0)
    weight_after :
        Weight for later table (0.0-1.0)
    """
    import shutil

    import casacore.tables as casatables

    # Copy the before table as base
    shutil.copytree(path_before, output_path)

    # Read gains from both tables
    with casatables.table(path_before, readonly=True) as tb_before:
        gains_before = tb_before.getcol("CPARAM")
        flags_before = tb_before.getcol("FLAG")
        tb_before.getcol("ANTENNA1")

    with casatables.table(path_after, readonly=True) as tb_after:
        gains_after = tb_after.getcol("CPARAM")
        flags_after = tb_after.getcol("FLAG")
        tb_after.getcol("ANTENNA1")

    # Verify shapes match
    if gains_before.shape != gains_after.shape:
        raise ValueError(f"Gain shapes don't match: {gains_before.shape} vs {gains_after.shape}")

    # Compute weighted average of complex gains
    # For flagged data, use the unflagged value; if both flagged, use before
    merged_gains = np.empty_like(gains_before)
    merged_flags = flags_before.copy()

    # Where both are unflagged: weighted average
    both_good = ~flags_before & ~flags_after
    merged_gains[both_good] = (
        weight_before * gains_before[both_good] + weight_after * gains_after[both_good]
    )

    # Where only before is good: use before
    only_before_good = ~flags_before & flags_after
    merged_gains[only_before_good] = gains_before[only_before_good]
    merged_flags[only_before_good] = False

    # Where only after is good: use after
    only_after_good = flags_before & ~flags_after
    merged_gains[only_after_good] = gains_after[only_after_good]
    merged_flags[only_after_good] = False

    # Where both flagged: keep before (flagged)
    both_flagged = flags_before & flags_after
    merged_gains[both_flagged] = gains_before[both_flagged]
    merged_flags[both_flagged] = True

    # Write merged gains to output table
    with casatables.table(output_path, readonly=False) as tb_out:
        tb_out.putcol("CPARAM", merged_gains)
        tb_out.putcol("FLAG", merged_flags)

    logger.debug(
        "Merged gains: %.1f%% interpolated, %.1f%% from before, %.1f%% from after, %.1f%% flagged",
        100 * np.sum(both_good) / both_good.size,
        100 * np.sum(only_before_good) / both_good.size,
        100 * np.sum(only_after_good) / both_good.size,
        100 * np.sum(both_flagged) / both_good.size,
    )


def _verify_corrected_data_populated(ms_path: str, min_fraction: float = 0.01) -> None:
    """Verify CORRECTED_DATA column is populated after applycal.

    This ensures we follow "measure twice, cut once" - verify calibration
    was applied successfully before proceeding.

    Parameters
    ----------
    ms_path :
        Path to Measurement Set
    min_fraction :
        Minimum fraction of unflagged data that must be non-zero

    Raises
    ------
    RuntimeError
        If CORRECTED_DATA is not populated

    """
    import casacore.tables as casatables  # type: ignore[import]

    table = casatables.table  # noqa: N816

    try:
        with table(ms_path, readonly=True) as tb:
            _check_corrected_data_column(tb, ms_path)
            _verify_nonzero_fraction(tb, ms_path, min_fraction)
    except RuntimeError:
        raise
    except (OSError, ValueError, KeyError) as e:
        raise RuntimeError(
            f"Failed to verify CORRECTED_DATA population in MS: {ms_path}. Error: {e}"
        ) from e


def _check_corrected_data_column(tb, ms_path: str) -> None:
    """Check that CORRECTED_DATA column exists and MS has data.

    Parameters
    ----------
    tb :

    """
    if "CORRECTED_DATA" not in tb.colnames():
        raise RuntimeError(
            f"CORRECTED_DATA column not present in MS: {ms_path}. "
            f"Calibration may not have been applied successfully."
        )

    if tb.nrows() == 0:
        raise RuntimeError(f"MS has zero rows: {ms_path}. Cannot verify calibration.")


def _verify_nonzero_fraction(tb, ms_path: str, min_fraction: float) -> None:
    """Verify that a sufficient fraction of CORRECTED_DATA is non-zero.

    Parameters
    ----------
    tb :

    """
    n_rows = tb.nrows()
    sample_size = min(10000, n_rows)
    corrected_data = tb.getcol("CORRECTED_DATA", startrow=0, nrow=sample_size)
    flags = tb.getcol("FLAG", startrow=0, nrow=sample_size)

    unflagged = corrected_data[~flags]
    if len(unflagged) == 0:
        raise RuntimeError(
            f"All CORRECTED_DATA is flagged in MS: {ms_path}. "
            f"Cannot verify calibration was applied."
        )

    nonzero_count = np.count_nonzero(np.abs(unflagged) > 1e-10)
    nonzero_fraction = nonzero_count / len(unflagged) if len(unflagged) > 0 else 0.0

    if nonzero_fraction < min_fraction:
        raise RuntimeError(
            f"CORRECTED_DATA appears unpopulated in MS: {ms_path}. "
            f"Only {nonzero_fraction * 100:.1f}% of unflagged data is non-zero "
            f"(minimum {min_fraction * 100:.1f}% required). "
        )

    logger.info(
        "Verified CORRECTED_DATA populated: %.1f%% non-zero (%d/%d samples)",
        nonzero_fraction * 100,
        nonzero_count,
        len(unflagged),
    )


def _read_gains_from_caltable(caltable_path: str) -> tuple[np.ndarray, np.ndarray]:
    """Read complex gains from a CASA calibration table.

    Parameters
    ----------
    caltable_path :
        Path to calibration table

    Returns
    -------
        Tuple of (gains, antenna_ids) where gains shape is (n_ant, n_pol)

    """
    import casacore.tables as casatables

    with casatables.table(caltable_path, readonly=True) as tb:
        gains = tb.getcol("CPARAM")  # Complex gains
        ant_ids = tb.getcol("ANTENNA1")
        flags = tb.getcol("FLAG")

        # Average over time/spw if multiple solutions
        # Take first solution interval for simplicity
        if gains.ndim == 3:  # (n_rows, n_chan, n_pol)
            gains = gains[:, 0, :]  # Take first channel
        elif gains.ndim == 2:  # (n_rows, n_pol)
            pass
        else:
            gains = gains.reshape(-1, 1)

        # Apply flags - set flagged gains to 1.0 (identity)
        if flags.ndim == 3:
            flags = flags[:, 0, :]
        gains = np.where(flags, 1.0 + 0j, gains)

        return gains, ant_ids


def _read_ms_for_gpu_cal(
    ms_path: str,
    datacolumn: str = "DATA",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read visibilities and antenna indices from MS for GPU calibration.

    Parameters
    ----------
    ms_path :
        Path to Measurement Set
    datacolumn :
        Data column to read

    Returns
    -------
        Tuple of (vis, ant1, ant2) arrays

    """
    import casacore.tables as casatables

    with casatables.table(ms_path, readonly=True) as tb:
        vis = tb.getcol(datacolumn)
        ant1 = tb.getcol("ANTENNA1")
        ant2 = tb.getcol("ANTENNA2")

    return vis, ant1, ant2


def _write_corrected_data(ms_path: str, corrected: np.ndarray) -> None:
    """Write corrected visibilities back to MS.

    Parameters
    ----------
    """
    import casacore.tables as casatables

    with casatables.table(ms_path, readonly=False) as tb:
        tb.putcol("CORRECTED_DATA", corrected)


@gpu_safe(max_gpu_gb=6.0, max_system_gb=6.0)
def apply_gains_to_ms(
    ms_path: str,
    gaintable: str,
    *,
    datacolumn: str = "DATA",
    use_gpu: bool = True,
) -> Optional["ApplyCalResult"]:
    """Apply gains from a calibration table to an MS using GPU acceleration.

    GPU Acceleration:
        Uses CuPy-based gain application for ~10x speedup on large datasets.
        Falls back to CPU when GPU is unavailable.

    Parameters
    ----------
    ms_path :
        Path to Measurement Set
    gaintable :
        Path to calibration table with complex gains
    datacolumn :
        Data column to calibrate (default: DATA)
    use_gpu :
        Whether to attempt GPU acceleration

    Returns
    -------
        ApplyCalResult with statistics, or None if GPU calibration unavailable

    """
    if not GPU_CALIBRATION_AVAILABLE:
        logger.warning("GPU calibration not available")
        return None

    logger.info("GPU gain application %s with %s", ms_path, gaintable)

    try:
        # Read gains from caltable
        gains, gain_ant_ids = _read_gains_from_caltable(gaintable)
        n_ant = int(np.max(gain_ant_ids)) + 1

        # Reorganize gains by antenna ID
        gains_by_ant = np.ones((n_ant, gains.shape[1]), dtype=np.complex128)
        for i, ant_id in enumerate(gain_ant_ids):
            gains_by_ant[ant_id] = gains[i]

        # Read MS data
        vis, ant1, ant2 = _read_ms_for_gpu_cal(ms_path, datacolumn)

        # Flatten for GPU processing if needed
        original_shape = vis.shape
        if vis.ndim > 1:
            vis_flat = vis.reshape(-1)
            # Expand antenna indices to match flattened vis
            n_extra = vis.size // len(ant1)
            ant1_exp = np.repeat(ant1, n_extra)
            ant2_exp = np.repeat(ant2, n_extra)
        else:
            vis_flat = vis
            ant1_exp = ant1
            ant2_exp = ant2

        # Extract scalar gains (first polarization)
        gains_scalar = gains_by_ant[:, 0]

        # Check GPU availability
        gpu_ok, _ = check_gpu_memory_available(2.0)
        actual_use_gpu = use_gpu and gpu_ok and is_gpu_available()

        # Apply gains
        result = apply_gains(vis_flat, gains_scalar, ant1_exp, ant2_exp, use_gpu=actual_use_gpu)

        # Reshape and write back
        corrected = vis_flat.reshape(original_shape)
        _write_corrected_data(ms_path, corrected)

        logger.info(
            "GPU calibration complete: %d/%d vis calibrated in %.2fs",
            result.n_vis_calibrated,
            result.n_vis_processed,
            result.processing_time_s,
        )
        return result

    except (OSError, RuntimeError, ValueError) as exc:
        logger.error("GPU gain application failed: %s", exc)
        return None


@memory_safe(max_system_gb=6.0)
@timed("calibration.apply_to_target")
def apply_to_target(
    ms_target: str,
    field: str,
    gaintables: list[str],
    interp: list[str] | None = None,
    calwt: bool = True,
    # CASA accepts a single list (applied to all tables) or a list-of-lists
    # (one mapping per gaintable). Use Union typing to document both shapes.
    spwmap: list[int] | list[list[int]] | None = None,
    verify: bool = True,
) -> None:
    """Apply calibration tables to a target MS field.

    Memory Safety:
        Wrapped with @memory_safe to check system RAM availability before
        processing. Rejects if less than 30% RAM available or less than 2GB free.

    **PRECONDITION**: All calibration tables must exist and be compatible with
    the MS. This ensures consistent, reliable calibration application.

    **POSTCONDITION**: If `verify=True`, CORRECTED_DATA is verified to be populated
    after application. This ensures calibration was applied successfully.

    interp defaults will be set to 'linear' matching list length.

    Parameters
    ----------
    """
    # PRECONDITION CHECK: Validate all calibration tables before applying
    # This ensures we follow "measure twice, cut once" - establish requirements upfront
    # for consistent, reliable calibration application.
    if not gaintables:
        raise ValueError("No calibration tables provided for applycal")

    print(f"Validating {len(gaintables)} calibration table(s) before applying...")

    # STRICT SEPARATION: Reject NON_SCIENCE calibration tables for production use
    for gaintable in gaintables:
        if "NON_SCIENCE" in gaintable:
            raise ValueError(
                f":warning:  STRICT SEPARATION VIOLATION: Attempting to apply NON_SCIENCE calibration table '{gaintable}' to production data.\n"
                f"   NON_SCIENCE tables (prefixed with 'NON_SCIENCE_*') are created by development tier calibration.\n"
                f"   These tables CANNOT be applied to production/science data due to time-channel binning mismatches.\n"
                f"   Use standard or high_precision tier calibration for production data."
            )

    try:
        validate_caltables_for_use(gaintables, ms_target, require_all=True)
    except (FileNotFoundError, ValueError) as e:
        raise ValueError(
            f"Calibration table validation failed. This is a required precondition for "
            f"applycal. Error: {e}"
        ) from e

    if interp is None:
        # Prefer 'nearest' for bandpass-like tables, 'linear' for gains.
        # Heuristic by table name; callers can override explicitly.
        _defaults: list[str] = []
        for gt in gaintables:
            low = gt.lower()
            if low.endswith(".b") or "bpcal" in low or "bandpass" in low:
                _defaults.append("nearest")
            else:
                _defaults.append("linear")
        interp = _defaults
    kwargs = dict(
        vis=ms_target,
        field=field,
        gaintable=gaintables,
        interp=interp,
        calwt=calwt,
    )
    # Only pass spwmap if explicitly provided; CASA rejects explicit null
    if spwmap is not None:
        kwargs["spwmap"] = spwmap

    ensure_ms_writable(ms_target)

    print(f"Applying {len(gaintables)} calibration table(s) to {ms_target}...")

    # Import and call applycal with CASA log environment protection
    service = CASAService()
    service.applycal(**kwargs)

    # POSTCONDITION CHECK: Verify CORRECTED_DATA was populated successfully
    # This ensures we follow "measure twice, cut once" - verify calibration was
    # applied successfully before proceeding.
    if verify:
        _verify_corrected_data_populated(ms_target)
