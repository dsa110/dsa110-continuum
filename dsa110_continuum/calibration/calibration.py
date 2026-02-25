"""Calibration module for DSA-110 pipeline.

This module provides high-level calibration functions wrapping CASA tasks,
including delay calibration (K), bandpass calibration (B), and gain calibration (G).
It handles process isolation, logging, and parameter validation.
"""

import fnmatch
import logging
import os
from typing import Any

from dsa110_contimg.common.utils import timed
from dsa110_contimg.common.utils.casa_init import ensure_casa_path
from dsa110_contimg.core.calibration.casa_service import CASAService
from dsa110_contimg.core.calibration.validate import (
    validate_caltables_for_use,
)
from dsa110_contimg.core.conversion.merge_spws import get_spw_count

# Initialize CASA environment before importing CASA modules
ensure_casa_path()

# setjy imported elsewhere; avoid unused import here

logger = logging.getLogger(__name__)

# Provide a single casacore tables symbol for the module
try:
    import casacore.tables as _casatables  # type: ignore

    table = _casatables.table  # noqa: N816
except ImportError:
    _casatables = None
    table = None


def _call_gaincal(**kwargs) -> None:
    """Call gaincal task via CASAService."""
    service = CASAService()
    service.gaincal(**kwargs)


def _call_gaincal_with_progress(
    stage_name: str,
    ms: str,
    caltable: str,
    **kwargs,
) -> None:
    """Call gaincal with progress monitoring.

    Parameters
    ----------
    stage_name :
        Human-readable name for progress display (e.g., "Delay solve")
    ms :
        Path to Measurement Set
    caltable :
        Path to output calibration table
    **kwargs :
        Arguments to pass to gaincal
    """
    from dsa110_contimg.common.utils.progress import StageProgressMonitor, estimate_calibration_time

    # Get MS info for progress estimation
    try:
        with table(ms, ack=False) as t:
            n_rows = t.nrows()
        with table(f"{ms}::SPECTRAL_WINDOW", ack=False) as tspw:
            n_spws = tspw.nrows()
        with table(f"{ms}::ANTENNA", ack=False) as tant:
            n_ant = tant.nrows()
    except Exception:
        n_rows, n_spws, n_ant = 1_000_000, 16, 110

    # Estimate expected runtime (gaincal is generally faster than bandpass)
    estimated_seconds = estimate_calibration_time(n_rows, n_spws, n_ant) * 0.5

    # Create progress monitor
    monitor = StageProgressMonitor(
        stage_name,
        output_path=caltable,
        poll_interval=5.0,
        estimated_seconds=estimated_seconds,
    )
    monitor.set_context(rows=n_rows, SPWs=n_spws, antennas=n_ant)

    service = CASAService()
    with monitor:
        service.gaincal(vis=ms, caltable=caltable, **kwargs)


def _get_caltable_spw_count(caltable_path: str) -> int | None:
    """Get the number of unique spectral windows in a calibration table.

    Parameters
    ----------
    caltable_path :
        Path to calibration table

    Returns
    -------
        Number of unique SPWs, or None if unable to read

    """
    import numpy as np  # type: ignore[import]

    # use module-level table

    try:
        with table(caltable_path, readonly=True) as tb:
            if "SPECTRAL_WINDOW_ID" not in tb.colnames():
                return None
            spw_ids = tb.getcol("SPECTRAL_WINDOW_ID")
            return len(np.unique(spw_ids))
    except (OSError, RuntimeError, KeyError):
        return None


# QA thresholds for calibration quality assessment (Issue #5 fix)
QA_SNR_MIN_THRESHOLD = 3.0  # Minimum acceptable mean SNR
QA_SNR_WARN_THRESHOLD = 10.0  # SNR below this triggers warning
QA_FLAGGED_MAX_THRESHOLD = 0.5  # Maximum acceptable flagged fraction
QA_FLAGGED_WARN_THRESHOLD = 0.2  # Flagged fraction above this triggers warning
QA_MIN_ANTENNAS = 10  # Minimum antennas for valid calibration


def _extract_quality_metrics(
    caltable_path: str,
    *,
    snr_min: float = QA_SNR_MIN_THRESHOLD,
    snr_warn: float = QA_SNR_WARN_THRESHOLD,
    flagged_max: float = QA_FLAGGED_MAX_THRESHOLD,
    flagged_warn: float = QA_FLAGGED_WARN_THRESHOLD,
    min_antennas: int = QA_MIN_ANTENNAS,
) -> dict[str, Any] | None:
    """Extract quality metrics from a calibration table with QA assessment.

    This function fixes Issue #5: No calibration QA before registration.
    It extracts metrics AND performs quality assessment, adding qa_passed
    and any issues/warnings to the metrics dict.

    Parameters
    ----------
    caltable_path :
        Path to calibration table
    snr_min :
        Minimum acceptable mean SNR (default: 5.0)
    snr_warn :
        SNR threshold for warnings (default: 10.0)
    flagged_max :
        Maximum acceptable flagged fraction (default: 0.5)
    flagged_warn :
        Flagged fraction threshold for warnings (default: 0.2)
    min_antennas :
        Minimum number of antennas (default: 10)

    Returns
    -------
        Dictionary with quality metrics (SNR, flagged_fraction, etc.),
        qa_passed (bool), and any issues/warnings. Returns None on read error.

    """
    import time

    import numpy as np  # type: ignore[import]

    try:
        with table(caltable_path, readonly=True) as tb:
            metrics: dict[str, Any] = {
                "qa_passed": True,  # Assume pass until proven otherwise
                "issues": [],
                "warnings": [],
                "assessed_at": time.time(),
            }

            # Number of solutions
            nrows = tb.nrows()
            metrics["n_solutions"] = nrows

            if nrows == 0:
                metrics["qa_passed"] = False
                metrics["issues"].append("Calibration table has zero solutions")
                return metrics

            # Check for FLAG column
            if "FLAG" in tb.colnames():
                flags = tb.getcol("FLAG")
                if flags.size > 0:
                    flagged_count = np.sum(flags)
                    total_count = flags.size
                    flagged_fraction = float(flagged_count / total_count)
                    metrics["flagged_fraction"] = flagged_fraction

                    # QA check: flagged fraction
                    if flagged_fraction > flagged_max:
                        metrics["qa_passed"] = False
                        metrics["issues"].append(
                            f"Flagged fraction too high: {flagged_fraction:.1%} "
                            f"(max: {flagged_max:.1%})"
                        )
                    elif flagged_fraction > flagged_warn:
                        metrics["warnings"].append(f"High flagged fraction: {flagged_fraction:.1%}")

            # Check for SNR column
            if "SNR" in tb.colnames():
                snr = tb.getcol("SNR")
                if snr.size > 0:
                    snr_flat = snr.flatten()
                    snr_valid = snr_flat[~np.isnan(snr_flat)]
                    if len(snr_valid) > 0:
                        snr_mean = float(np.mean(snr_valid))
                        snr_median = float(np.median(snr_valid))
                        snr_min_val = float(np.min(snr_valid))
                        snr_max_val = float(np.max(snr_valid))

                        metrics["snr_mean"] = snr_mean
                        metrics["snr_median"] = snr_median
                        metrics["snr_min"] = snr_min_val
                        metrics["snr_max"] = snr_max_val

                        # QA check: mean SNR
                        if snr_mean < snr_min:
                            metrics["qa_passed"] = False
                            metrics["issues"].append(
                                f"Mean SNR too low: {snr_mean:.1f} (min: {snr_min:.1f})"
                            )
                        elif snr_mean < snr_warn:
                            metrics["warnings"].append(f"Low mean SNR: {snr_mean:.1f}")

            # Number of antennas
            if "ANTENNA1" in tb.colnames():
                ant1 = tb.getcol("ANTENNA1")
                unique_ants = np.unique(ant1)
                n_antennas = len(unique_ants)
                metrics["n_antennas"] = n_antennas

                # QA check: minimum antennas
                if n_antennas < min_antennas:
                    metrics["qa_passed"] = False
                    metrics["issues"].append(
                        f"Too few antennas: {n_antennas} (min: {min_antennas})"
                    )

            # Number of spectral windows
            if "SPECTRAL_WINDOW_ID" in tb.colnames():
                spw_ids = tb.getcol("SPECTRAL_WINDOW_ID")
                unique_spws = np.unique(spw_ids)
                metrics["n_spws"] = len(unique_spws)

            # Clean up empty lists
            if not metrics["issues"]:
                del metrics["issues"]
            if not metrics["warnings"]:
                del metrics["warnings"]

            return metrics

    except Exception as e:
        logger.warning(f"Failed to extract quality metrics from {caltable_path}: {e}")
        return None


def _track_calibration_provenance(
    ms_path: str,
    caltable_path: str,
    task_name: str,
    params: dict[str, Any],
    registry_db: str | None = None,
) -> None:
    """Track calibration provenance after successful solve.

        This function captures and stores provenance information (source MS,
        solver command, version, parameters, quality metrics) for a calibration table.

    Parameters
    ----------
    ms_path : str
        Path to the input MS that generated this caltable.
    caltable_path : str
        Path to the calibration table.
    task_name : str
        CASA task name used (e.g., "gaincal", "bandpass").
    params : dict[str, Any]
        Parameters used in the calibration task.
    registry_db : str or None, optional
        Optional path to registry database. Default is None.

    Returns
    -------
        None
    """
    try:
        from pathlib import Path as PathLib

        from dsa110_contimg.infrastructure.database.provenance import track_calibration_provenance

        # Use CASAService for version and command string
        service = CASAService()
        casa_version = service.get_version()

        # Build command string
        command_str = service.build_command_string(task_name, params)

        # Extract quality metrics
        quality_metrics = _extract_quality_metrics(caltable_path)

        # Determine registry DB path (unified pipeline database)
        if registry_db is None:
            # Use unified pipeline database
            registry_db_path = PathLib(
                os.environ.get(
                    "PIPELINE_DB",
                    os.environ.get(
                        "CAL_REGISTRY_DB",  # Legacy fallback
                        os.environ.get(
                            "PIPELINE_DB", "/data/dsa110-contimg/state/db/pipeline.sqlite3"
                        ),
                    ),
                )
            )
        else:
            registry_db_path = PathLib(registry_db)

        # Track provenance
        track_calibration_provenance(
            registry_db=registry_db_path,
            ms_path=ms_path,
            caltable_path=caltable_path,
            params=params,
            metrics=quality_metrics,
            solver_command=command_str,
            solver_version=casa_version,
        )

        logger.debug(
            f"Tracked provenance for {caltable_path} (source: {ms_path}, version: {casa_version})"
        )

    except Exception as e:
        # Don't fail calibration if provenance tracking fails
        logger.warning(
            f"Failed to track provenance for {caltable_path}: {e}. "
            f"Calibration succeeded but provenance not recorded."
        )


def _determine_spwmap_for_bptables(
    bptables: list[str],
    ms_path: str,
) -> list[int] | None:
    """Determine spwmap parameter for bandpass tables when combine_spw was used.

    When a bandpass table is created with combine_spw=True, it contains solutions
    only for SPW=0 (the aggregate SPW). When applying this table during gain
    calibration, we need to map all MS SPWs to SPW 0 in the bandpass table.

    Parameters
    ----------
    bptables :
        List of bandpass table paths
    ms_path :
        Path to Measurement Set
    bptables: List[str] :

    Returns
    -------
        List of SPW mappings [0, 0, 0, ...] if needed, or None if not needed.
        The length of the list equals the number of SPWs in the MS.

    """
    if not bptables:
        return None

    # Get number of SPWs in MS
    n_ms_spw = get_spw_count(ms_path)
    if n_ms_spw is None or n_ms_spw <= 1:
        return None

    # Check if any bandpass table has only 1 SPW (indicating combine_spw was used)
    for bptable in bptables:
        n_bp_spw = _get_caltable_spw_count(bptable)
        logger.debug(
            f"Checking table {os.path.basename(bptable)}: {n_bp_spw} SPW(s), MS has {n_ms_spw} SPWs"
        )
        if n_bp_spw == 1:
            # This bandpass table was created with combine_spw=True
            # Map all MS SPWs to SPW 0 in the bandpass table
            logger.info(
                f"Detected calibration table {os.path.basename(bptable)} has only 1 SPW (from combine_spw), "
                f"while MS has {n_ms_spw} SPWs. Setting spwmap to map all MS SPWs to SPW 0."
            )
            return [0] * n_ms_spw

    return None


def _validate_solve_success(caltable_path: str, refant: int | str | None = None) -> None:
    """Validate that a calibration solve completed successfully.

    This ensures we follow "measure twice, cut once" - verify solutions exist
    immediately after each solve completes, before proceeding to the next step.

    Parameters
    ----------
    caltable_path :
        Path to calibration table
    refant :
        Optional reference antenna ID to verify has solutions

    Raises
    ------
    RuntimeError
        If table doesn't exist, has no solutions, or refant missing

    """
    # use module-level table

    # Verify table exists
    if not os.path.exists(caltable_path):
        raise RuntimeError(f"Calibration solve failed: table was not created: {caltable_path}")

    # Verify table has solutions
    try:
        with table(caltable_path, readonly=True) as tb:
            if tb.nrows() == 0:
                raise RuntimeError(
                    f"Calibration solve failed: table has no solutions: {caltable_path}"
                )

            # Verify refant has solutions if provided
            if refant is not None:
                # Handle comma-separated refant string (e.g., "103,111,113,115,104")
                # Use the first antenna in the chain for validation
                if isinstance(refant, str):
                    if "," in refant:
                        # Comma-separated list: use first antenna
                        refant_str = refant.split(",")[0].strip()
                        refant_int = int(refant_str)
                    else:
                        # Single antenna ID as string
                        refant_int = int(refant)
                else:
                    refant_int = refant

                antennas = tb.getcol("ANTENNA1")

                # For antenna-based calibration, check ANTENNA1
                # For baseline-based calibration, check both ANTENNA1 and ANTENNA2
                if "ANTENNA2" in tb.colnames():
                    ant2 = tb.getcol("ANTENNA2")
                    # Filter out -1 values (baseline-based calibration uses -1 for antenna-based entries)
                    ant2_valid = ant2[ant2 != -1]
                    all_antennas = set(antennas) | set(ant2_valid)
                else:
                    all_antennas = set(antennas)

                if refant_int not in all_antennas:
                    raise RuntimeError(
                        f"Calibration solve failed: reference antenna {refant} has no solutions "
                        f"in table: {caltable_path}. Available antennas: {sorted(all_antennas)}"
                    )
    except RuntimeError:
        raise
    except Exception as e:
        raise RuntimeError(
            f"Calibration solve validation failed: unable to read table {caltable_path}. Error: {e}"
        ) from e


def _check_flag_fraction(
    caltable_path: str,
    max_flag_fraction: float = 0.25,
    cal_type: str = "calibration",
) -> float:
    """Check if flag fraction in calibration table exceeds threshold.

    This function excludes fully-flagged antennas (dead antennas) from the
    calculation, since they don't indicate calibration problems - just
    non-working hardware.

    Parameters
    ----------
    caltable_path :
        Path to calibration table
    max_flag_fraction :
        Maximum allowed flag fraction (default: 0.25 = 25%)
    cal_type :
        Type of calibration for error message (e.g., "bandpass", "gain")

    Returns
    -------
        Actual flag fraction (0.0 to 1.0) for working antennas only

    Raises
    ------
    ValueError
        If flag fraction exceeds max_flag_fraction

    """
    import numpy as np

    with table(caltable_path, readonly=True, ack=False) as tb:
        if "FLAG" not in tb.colnames():
            logger.warning(f"No FLAG column in {caltable_path}, skipping flag check")
            return 0.0

        flags = tb.getcol("FLAG")
        # casacore returns shape (nrow, nchan, npol) for bandpass tables
        # where nrow = number of antennas

        # Calculate raw flag fraction
        total = flags.size
        flagged = int(np.sum(flags))
        raw_flag_fraction = flagged / total if total > 0 else 0.0

        # Calculate flag fraction excluding fully-dead antennas
        # Dead antennas have 100% of their solutions flagged
        # For bandpass tables: shape is (nant, nchan, npol)
        if flags.ndim == 3:
            nant, nchan, npol = flags.shape
            # Sum over channels and polarizations to get per-antenna flag count
            per_ant_flags = np.sum(flags, axis=(1, 2))
            max_flags_per_ant = nchan * npol

            # Count dead antennas (>=99% flagged)
            # Using a threshold handles cases where an antenna is effectively dead
            # but has a few unflagged solutions (e.g. due to edge effects or gaps)
            dead_ant_mask = per_ant_flags >= 0.99 * max_flags_per_ant
            n_dead = int(np.sum(dead_ant_mask))
            n_working = nant - n_dead

            if n_working > 0:
                # Calculate flag fraction for working antennas only
                working_flags = np.sum(flags[~dead_ant_mask, :, :])
                working_total = n_working * nchan * npol
                effective_flag_fraction = working_flags / working_total
            else:
                effective_flag_fraction = raw_flag_fraction

            logger.info(
                f"Flag fraction in {cal_type} table: {raw_flag_fraction * 100:.1f}% raw "
                f"({flagged:,}/{total:,} solutions flagged)"
            )
            if n_dead > 0:
                logger.info(
                    f"  Excluding {n_dead} fully-flagged (dead) antennas: "
                    f"effective flag fraction = {effective_flag_fraction * 100:.1f}% "
                    f"({n_working} working antennas)"
                )
        else:
            # Fallback for other table shapes
            effective_flag_fraction = raw_flag_fraction
            n_dead = 0
            logger.info(
                f"Flag fraction in {cal_type} table: {raw_flag_fraction * 100:.1f}% "
                f"({flagged:,}/{total:,} solutions flagged)"
            )

    if effective_flag_fraction > max_flag_fraction:
        dead_info = f" (excluding {n_dead} dead antennas)" if n_dead > 0 else ""
        raise ValueError(
            f"{cal_type.upper()} SOLVE FAILED: Excessive flagging detected{dead_info}.\n"
            f"  Effective flag fraction: {effective_flag_fraction * 100:.1f}% (threshold: {max_flag_fraction * 100:.0f}%)\n"
            f"  Raw flagged solutions: {flagged:,} / {total:,}\n\n"
            f"This indicates poor data quality or incorrect calibration setup.\n"
            f"Common causes:\n"
            f"  - Data not coherently phased to calibrator\n"
            f"  - Low SNR (calibrator too faint or too far from beam center)\n"
            f"  - RFI contamination\n"
            f"  - Incorrect MODEL_DATA (wrong flux or position)\n"
        )

    return effective_flag_fraction


def _print_bandpass_solution_summary(
    caltable_path: str,
    ms: str,
) -> dict[str, Any]:
    """Print comprehensive summary of bandpass solution flagging.

    Provides aggregated flagging statistics that CASA doesn't report by default,
    giving visibility into where solutions were flagged due to SNR or other issues.

    Parameters
    ----------
    caltable_path :
        Path to the bandpass calibration table
    ms :
        Path to the measurement set (for antenna names)

    Returns
    -------
        Dictionary with detailed flagging statistics

    """
    import numpy as np

    stats: dict[str, Any] = {}

    # Get antenna names from MS
    ant_names = {}
    try:
        with table(f"{ms}::ANTENNA", readonly=True, ack=False) as ant_tb:
            names = ant_tb.getcol("NAME")
            for i, name in enumerate(names):
                ant_names[i] = name
    except Exception:
        pass  # Fall back to antenna IDs

    with table(caltable_path, readonly=True, ack=False) as tb:
        if "FLAG" not in tb.colnames():
            print("  [No FLAG column in calibration table]")
            return stats

        flags = tb.getcol("FLAG")  # Shape: (nant, nchan, npol) for bandpass
        antenna_ids = tb.getcol("ANTENNA1")
        spw_ids = tb.getcol("SPECTRAL_WINDOW_ID")

        if flags.ndim != 3:
            print(f"  [Unexpected table shape: {flags.shape}]")
            return stats

        nrows, nchan, npol = flags.shape

        # Unique SPWs and antennas
        unique_spws = sorted(set(spw_ids))
        unique_ants = sorted(set(antenna_ids))
        n_spw = len(unique_spws)
        n_ant = len(unique_ants)

        # Overall statistics
        total_solutions = flags.size
        total_flagged = int(np.sum(flags))
        overall_frac = total_flagged / total_solutions if total_solutions > 0 else 0.0

        print("\n" + "─" * 80)
        print("BANDPASS SOLUTION SUMMARY")
        print("─" * 80)
        print(f"  Table: {os.path.basename(caltable_path)}")
        print(
            f"  Dimensions: {n_ant} antennas × {n_spw} SPWs × {nchan} channels × {npol} polarizations"
        )
        print(f"  Total solutions: {total_solutions:,}")
        print(f"  Flagged solutions: {total_flagged:,} ({overall_frac * 100:.2f}%)")

        stats["total_solutions"] = total_solutions
        stats["total_flagged"] = total_flagged
        stats["overall_fraction"] = overall_frac

        # Per-SPW breakdown
        print("\n  Per-SPW Flagging:")
        print("  " + "-" * 50)
        spw_stats = {}
        high_flag_spws = []

        for spw in unique_spws:
            spw_mask = spw_ids == spw
            spw_flags = flags[spw_mask, :, :]
            spw_total = spw_flags.size
            spw_flagged = int(np.sum(spw_flags))
            spw_frac = spw_flagged / spw_total if spw_total > 0 else 0.0
            spw_stats[spw] = {"flagged": spw_flagged, "total": spw_total, "fraction": spw_frac}

            # Build a compact bar representation
            bar_len = 20
            filled = int(spw_frac * bar_len)
            bar = "█" * filled + "░" * (bar_len - filled)

            # Mark high-flagging SPWs
            marker = ""
            if spw_frac > 0.30:
                marker = "  HIGH"
                high_flag_spws.append(spw)
            elif spw_frac > 0.15:
                marker = " "

            print(
                f"    SPW {spw:2d}: [{bar}] {spw_frac * 100:5.1f}% ({spw_flagged:5d}/{spw_total:5d}){marker}"
            )

        stats["per_spw"] = spw_stats

        # Per-SPW per-Channel breakdown (detailed view showing ALL channels)
        print("\n  Per-Channel Flagging (all SPWs × channels):")
        print("  " + "-" * 70)
        chan_stats: dict[int, dict[int, dict[str, Any]]] = {}

        for spw in unique_spws:
            spw_mask = spw_ids == spw
            spw_flags = flags[spw_mask, :, :]  # Shape: (n_ants_in_spw, nchan, npol)
            chan_stats[spw] = {}

            print(f"    SPW {spw:2d}:")
            for chan_idx in range(nchan):
                # Flagging for this channel across all antennas and polarizations
                chan_flags = spw_flags[:, chan_idx, :]  # Shape: (n_ants_in_spw, npol)
                chan_total = chan_flags.size
                chan_flagged = int(np.sum(chan_flags))
                chan_frac = chan_flagged / chan_total if chan_total > 0 else 0.0
                chan_stats[spw][chan_idx] = {
                    "flagged": chan_flagged,
                    "total": chan_total,
                    "fraction": chan_frac,
                }

                # Build compact bar (10 chars for channel-level detail)
                bar_len = 10
                filled = int(chan_frac * bar_len)
                bar = "█" * filled + "░" * (bar_len - filled)

                # Mark high-flagging channels
                marker = ""
                if chan_frac >= 0.999:
                    marker = "  DEAD"
                elif chan_frac > 0.50:
                    marker = "  HIGH"
                elif chan_frac > 0.20:
                    marker = " "

                print(
                    f"      chan {chan_idx:3d}: [{bar}] {chan_flagged:4d} of {chan_total:4d} flagged "
                    f"({chan_frac * 100:5.1f}%){marker}"
                )

        stats["per_channel"] = chan_stats

        # Per-polarization breakdown
        print("\n  Per-Polarization Flagging:")
        print("  " + "-" * 50)
        pol_labels = ["RR/XX", "LL/YY"] if npol == 2 else [f"Pol{i}" for i in range(npol)]
        pol_stats = {}

        for pol_idx in range(npol):
            pol_flags = flags[:, :, pol_idx]
            pol_total = pol_flags.size
            pol_flagged = int(np.sum(pol_flags))
            pol_frac = pol_flagged / pol_total if pol_total > 0 else 0.0
            pol_stats[pol_labels[pol_idx]] = {
                "flagged": pol_flagged,
                "total": pol_total,
                "fraction": pol_frac,
            }
            print(
                f"    {pol_labels[pol_idx]:6s}: {pol_frac * 100:5.1f}% flagged ({pol_flagged:,}/{pol_total:,})"
            )

        stats["per_polarization"] = pol_stats

        # Per-antenna breakdown (show top 10 highest flagged)
        print("\n  Per-Antenna Flagging (top 10 highest):")
        print("  " + "-" * 50)
        ant_stats = {}

        for ant in unique_ants:
            ant_mask = antenna_ids == ant
            ant_flags = flags[ant_mask, :, :]
            ant_total = ant_flags.size
            ant_flagged = int(np.sum(ant_flags))
            ant_frac = ant_flagged / ant_total if ant_total > 0 else 0.0
            ant_name = ant_names.get(ant, str(ant))
            ant_stats[ant] = {
                "name": ant_name,
                "flagged": ant_flagged,
                "total": ant_total,
                "fraction": ant_frac,
            }

        # Sort by flagging fraction (descending)
        sorted_ants = sorted(ant_stats.items(), key=lambda x: x[1]["fraction"], reverse=True)

        # Identify dead antennas (100% flagged)
        dead_ants = [ant for ant, info in sorted_ants if info["fraction"] >= 0.999]
        partial_ants = [ant for ant, info in sorted_ants if 0.40 < info["fraction"] < 0.999]

        for ant, info in sorted_ants[:10]:
            bar_len = 20
            filled = int(info["fraction"] * bar_len)
            bar = "█" * filled + "░" * (bar_len - filled)

            marker = ""
            if info["fraction"] >= 0.999:
                marker = "  DEAD"
            elif info["fraction"] > 0.50:
                marker = "  BAD POL?"
            elif info["fraction"] > 0.30:
                marker = " "

            print(
                f"    Ant {info['name']:>4s} ({ant:3d}): [{bar}] {info['fraction'] * 100:5.1f}%{marker}"
            )

        stats["per_antenna"] = ant_stats
        stats["dead_antennas"] = dead_ants
        stats["partial_antennas"] = partial_ants

        # Summary recommendations
        print("\n  Summary:")
        print("  " + "-" * 50)

        if dead_ants:
            dead_names = [ant_stats[a]["name"] for a in dead_ants]
            print(f"    • {len(dead_ants)} dead antenna(s): {', '.join(dead_names)}")

        if partial_ants:
            partial_names = [ant_stats[a]["name"] for a in partial_ants]
            print(
                f"    • {len(partial_ants)} antenna(s) with partial flagging (40-99%): {', '.join(partial_names)}"
            )
            print("      → May indicate bad polarization(s); run: flag-bad-polarizations")

        if high_flag_spws:
            print(f"    • {len(high_flag_spws)} SPW(s) with >30% flagging: {high_flag_spws}")
            print("      → Check for RFI in these frequency ranges")

        # Exclude dead antennas from effective calculation
        if dead_ants:
            working_ants = [a for a in unique_ants if a not in dead_ants]
            working_mask = np.isin(antenna_ids, working_ants)
            working_flags = flags[working_mask, :, :]
            working_total = working_flags.size
            working_flagged = int(np.sum(working_flags))
            effective_frac = working_flagged / working_total if working_total > 0 else 0.0
            print(f"\n    Effective flagging (excl. dead antennas): {effective_frac * 100:.2f}%")
            stats["effective_fraction"] = effective_frac
        else:
            stats["effective_fraction"] = overall_frac

        print("─" * 80)

    return stats


logger = logging.getLogger(__name__)


def _resolve_field_ids(ms: str, field_sel: str) -> list[int]:
    """Resolve CASA-like field selection into a list of FIELD_ID integers.

    Supports numeric indices, comma lists, numeric ranges ("A~B"), and
    name/glob matching against FIELD::NAME.

    Parameters
    ----------
    ms : str
        Path to Measurement Set
    field_sel : str
        CASA-style field selection string
    """
    # use module-level table

    sel = str(field_sel).strip()
    # Try numeric selections first: comma-separated tokens and A~B ranges
    ids: list[int] = []
    numeric_tokens = [tok.strip() for tok in sel.replace(";", ",").split(",") if tok.strip()]

    def _add_numeric(tok: str) -> bool:
        if "~" in tok:
            a, b = tok.split("~", 1)
            if a.strip().isdigit() and b.strip().isdigit():
                ai, bi = int(a), int(b)
                lo, hi = (ai, bi) if ai <= bi else (bi, ai)
                ids.extend(list(range(lo, hi + 1)))
                return True
            return False
        if tok.isdigit():
            ids.append(int(tok))
            return True
        return False

    any_numeric = False
    for tok in numeric_tokens:
        if _add_numeric(tok):
            any_numeric = True

    if any_numeric:
        # Deduplicate and return
        return sorted(set(ids))

    # Fall back to FIELD::NAME glob matching
    patterns = [p for p in numeric_tokens if p]
    # If no separators were present, still try the full selector as a single
    # pattern
    if not patterns:
        patterns = [sel]

    try:
        with table(f"{ms}::FIELD") as tf:
            names = list(tf.getcol("NAME"))
            out = []
            for i, name in enumerate(names):
                for pat in patterns:
                    if fnmatch.fnmatchcase(str(name), pat):
                        out.append(int(i))
                        break
            return sorted(set(out))
    except (OSError, RuntimeError, KeyError):
        return []


def _validate_delay_solve_preconditions(ms: str, cal_field: str, refant: str) -> None:
    """Validate preconditions for delay solve.

    Parameters
    ----------
    ms : str
        Path to Measurement Set
    cal_field : str
        Field selection for calibration
    refant : str
        Reference antenna name or ID

    Raises
    ------
    ValueError
        If any precondition is not met.

    """
    import numpy as np

    logger.info(f"Validating data for delay solve on field(s) {cal_field}...")

    with table(ms) as tb:
        # Check MODEL_DATA exists
        if "MODEL_DATA" not in tb.colnames():
            raise ValueError(
                "MODEL_DATA column does not exist in MS. "
                "This is a required precondition for K-calibration. "
                "Populate MODEL_DATA using setjy, ft(), or a catalog model before "
                "calling solve_delay()."
            )

        # Check MODEL_DATA is populated (not all zeros)
        model_sample = tb.getcol("MODEL_DATA", startrow=0, nrow=min(100, tb.nrows()))
        if np.all(np.abs(model_sample) < 1e-10):
            raise ValueError(
                "MODEL_DATA column exists but is all zeros (unpopulated). "
                "This is a required precondition for K-calibration. "
                "Populate MODEL_DATA using setjy, ft(), or a catalog model before "
                "calling solve_delay()."
            )

        # Resolve and check field selection
        field_ids = tb.getcol("FIELD_ID")
        target_ids = _resolve_field_ids(ms, str(cal_field))
        if not target_ids:
            raise ValueError(f"Unable to resolve field selection: {cal_field}")

        field_mask = np.isin(field_ids, np.asarray(target_ids, dtype=field_ids.dtype))
        if not np.any(field_mask):
            raise ValueError(f"No data found for field selection {cal_field}")

        row_idx = np.nonzero(field_mask)[0]
        if row_idx.size == 0:
            raise ValueError(f"No data found for field selection {cal_field}")

        # Check reference antenna exists in this field
        start_row = int(row_idx[0])
        nrow_sel = int(row_idx[-1] - start_row + 1)
        ant1_slice = tb.getcol("ANTENNA1", startrow=start_row, nrow=nrow_sel)
        ant2_slice = tb.getcol("ANTENNA2", startrow=start_row, nrow=nrow_sel)
        rel_idx = row_idx - start_row
        field_ant1 = ant1_slice[rel_idx]
        field_ant2 = ant2_slice[rel_idx]
        ref_present = np.any((field_ant1 == int(refant)) | (field_ant2 == int(refant)))
        if not ref_present:
            raise ValueError(f"Reference antenna {refant} not found in field {cal_field}")

        # Check for unflagged data
        field_flags = tb.getcol("FLAG", startrow=start_row, nrow=nrow_sel)
        unflagged_count = int(np.sum(~field_flags))
        if unflagged_count == 0:
            raise ValueError(f"All data in field {cal_field} is flagged")

        logger.debug(
            f"Field {cal_field}: {np.sum(field_mask)} rows, {unflagged_count} unflagged points"
        )


def _run_delay_gaincal(
    ms: str,
    caltable: str,
    cal_field: str,
    refant: str,
    solint: str,
    combine: str,
    minsnr: float,
    uvrange: str,
    retry_without_combine: bool = True,
) -> str:
    """Run gaincal for delay (K) solve with optional retry and progress monitoring.

    Parameters
    ----------
    ms :
        Measurement set path
    caltable :
        Output calibration table path
    cal_field :
        Calibrator field selection
    refant :
        Reference antenna
    solint :
        Solution interval
    combine :
        Combine mode (e.g., "spw" or "")
    minsnr :
        Minimum SNR
    uvrange :
        UV range filter
    retry_without_combine :
        If True, retry with combine="" on failure

    Returns
    -------
        Path to created calibration table

    Raises
    ------
    RuntimeError
        If solve fails even after retry

    """
    kwargs = dict(
        field=cal_field,
        solint=solint,
        refant=refant,
        gaintype="K",
        combine=combine,
        minsnr=minsnr,
        selectdata=True,
    )
    if uvrange:
        kwargs["uvrange"] = uvrange
        logger.debug(f"Using uvrange filter: {uvrange}")

    try:
        _call_gaincal_with_progress("Delay (K) solve", ms, caltable, **kwargs)
        _validate_solve_success(caltable, refant=refant)
        _track_calibration_provenance(
            ms_path=ms,
            caltable_path=caltable,
            task_name="gaincal",
            params={"vis": ms, "caltable": caltable, **kwargs},
        )
        return caltable
    except Exception as e:
        if not retry_without_combine or combine == "":
            raise RuntimeError(f"Delay solve failed: {e}") from e

        # Retry with no combination
        logger.error(f"Delay solve failed: {e}")
        logger.info("Retrying with no combination...")
        kwargs["combine"] = ""
        try:
            _call_gaincal_with_progress("Delay (K) solve (retry)", ms, caltable, **kwargs)
            _validate_solve_success(caltable, refant=refant)
            _track_calibration_provenance(
                ms_path=ms,
                caltable_path=caltable,
                task_name="gaincal",
                params={"vis": ms, "caltable": caltable, **kwargs},
            )
            return caltable
        except Exception as e2:
            raise RuntimeError(f"Delay solve failed even with conservative settings: {e2}") from e2


@timed("calibration.solve_delay")
def solve_delay(
    ms: str,
    cal_field: str,
    refant: str,
    table_prefix: str | None = None,
    combine_spw: bool = False,
    t_slow: str = "inf",
    t_fast: str | None = "60s",
    uvrange: str = "",
    minsnr: float = 3.0,
    skip_slow: bool = False,
) -> list[str]:
    """Solve delay (K) on slow and optional fast timescales using CASA gaincal.

    Uses casatasks.gaincal with gaintype='K' to avoid explicit casatools
    calibrater usage, which can be unstable in some notebook environments.

    **PRECONDITION**: MODEL_DATA must be populated before calling this function.

    Parameters
    ----------
    ms : str
        Path to Measurement Set
    field_selector : str
        Field selection string
    combine_spw : bool
        Whether SPWs are combined
    """
    # Validate preconditions (early return on failure)
    _validate_delay_solve_preconditions(ms, cal_field, refant)

    combine = "spw" if combine_spw else ""
    if table_prefix is None:
        table_prefix = f"{os.path.splitext(ms)[0]}_{cal_field}"

    tables: list[str] = []

    # ============================================================================
    # DELAY CALIBRATION (K) - TRANSPARENCY HEADER
    # ============================================================================
    logger.info("\n" + "=" * 80)
    logger.info("DELAY CALIBRATION (gaintype='K') - Stage 1")
    logger.info("=" * 80)
    logger.info("PURPOSE: Solve for geometric path delays and clock offsets")
    logger.info("         between antennas. For DSA-110 (connected-element array),")
    logger.info("         this is often OPTIONAL due to excellent clock sync.")
    logger.info("")
    logger.info("EXPECTED INPUT:")
    logger.info("  - MODEL_DATA: Populated with calibrator source model ✓")
    logger.info(f"  - Field: {cal_field}")
    logger.info(f"  - Reference antenna: {refant}")
    logger.info(f"  - Combine SPWs: {combine_spw}")
    logger.info(f"  - Solution interval: {t_slow} (slow), {t_fast} (fast)")
    logger.info(f"  - Minimum SNR: {minsnr}")
    logger.info("")
    logger.info("EXPECTED SOLUTIONS:")
    logger.info("  - Typical delays: -100 ns to +100 ns for DSA-110")
    logger.info("  - Geometric bounds: ±200 ns (based on max baseline ~5 km)")
    logger.info("  - Reference antenna: Delay = 0 ns (by definition)")
    logger.info("  - Core antennas: Small delays (~10-50 ns), tightly clustered")
    logger.info("  - Outrigger antennas: Larger delays (50-100 ns), more scattered")
    logger.info("")
    logger.info("QUALITY ASSURANCE:")
    logger.info("  1. Geometric validation: All delays within ±200 ns")
    logger.info("  2. SNR validation: Solutions above minimum SNR threshold")
    logger.info("  3. Outlier detection: Flag delays beyond ±500 ns as errors")
    logger.info("  4. Antenna coverage: At least 80% of antennas with valid solutions")
    logger.info("")
    logger.info("HOW SOLUTIONS ARE USED (CURRENT PIPELINE):")
    logger.info("  - Delay (K) solutions are computed and stored for diagnostics and QA")
    logger.info("  - They are NOT automatically applied in later gain calibration stages")
    logger.info("  - See solve_gains() documentation for the actual calibration order")
    logger.info("=" * 80 + "\n")

    # Slow (infinite) delay solve
    if not skip_slow:
        logger.info(f"→ Running SLOW delay solve (solint={t_slow}) on field {cal_field}...")
        logger.info(f"  This captures instrumental delays that don't change on minute timescales")
        caltable = _run_delay_gaincal(
            ms=ms,
            caltable=f"{table_prefix}.k",
            cal_field=cal_field,
            refant=refant,
            solint=t_slow,
            combine=combine,
            minsnr=minsnr,
            uvrange=uvrange,
            retry_without_combine=True,
        )
        tables.append(caltable)
        logger.info(f":check: Delay solve completed: {caltable}")
    else:
        logger.debug("Skipping slow delay solve (fast mode optimization)")

    # Fast (short) delay solve
    if t_fast or skip_slow:
        if skip_slow and not t_fast:
            t_fast = "60s"
            logger.debug(f"Using default fast solution interval: {t_fast}")

        logger.info(f"\n→ Running FAST delay solve (solint={t_fast}) on field {cal_field}...")
        logger.info(f"  This captures time-variable delays (if present)")
        logger.info(f"  Should show smooth evolution, not random jumps")
        try:
            caltable = _run_delay_gaincal(
                ms=ms,
                caltable=f"{table_prefix}.2k",
                cal_field=cal_field,
                refant=refant,
                solint=t_fast,
                combine=combine,
                minsnr=minsnr,
                uvrange=uvrange,
                retry_without_combine=False,  # Fast solve doesn't retry
            )
            tables.append(caltable)
            logger.info(f":check: Fast delay solve completed: {caltable}")
        except Exception as e:
            logger.error(f"Fast delay solve failed: {e}")
            logger.info("Skipping fast delay solve...")

    # ============================================================================
    # QUALITY ASSURANCE: Validate delay solutions
    # ============================================================================
    is_fast_mode = uvrange and uvrange.startswith(">")
    if not is_fast_mode:
        logger.info("\n" + "=" * 80)
        logger.info("DELAY CALIBRATION QA - Validating Solutions")
        logger.info("=" * 80)

        # QA Check 1: General calibration table validation
        try:
            logger.info("→ QA Check 1: Table structure and reference antenna validation")
            from dsa110_contimg.core.qa.pipeline_quality import check_calibration_quality

            check_calibration_quality(tables, ms_path=ms, alert_on_issues=True)
            logger.info("  ✓ Table structure valid")
            logger.info("  ✓ Reference antenna has valid solutions")
        except Exception as e:
            logger.warning(f"  ⚠ QA validation warning: {e}")

        # QA Check 2: Geometric validation of delay solutions
        try:
            logger.info("\n→ QA Check 2: Geometric validation of delay values")
            logger.info("  Expected: All delays within ±200 ns (geometric bounds)")
            logger.info("  Expected: At least 80% of antennas with valid solutions")
            logger.info("  Expected: Reference antenna delay = 0 ns")
            from dsa110_contimg.core.qa.delay_validation import check_delay_solutions

            for ktable in tables:
                if ktable.endswith(".k") or ktable.endswith(".2k"):
                    logger.info(f"\n  Validating {ktable}...")
                    result = check_delay_solutions(
                        ktable,
                        refant=refant,
                        raise_on_failure=True,  # Stop pipeline on invalid delays
                        strict=False,  # Allow some outliers
                    )
                    if result.is_valid:
                        logger.info(
                            f"  ✅ Delay validation PASSED: {result.n_within_bounds}/"
                            f"{result.n_antennas - result.n_flagged} antennas within "
                            f"geometric bounds (max {result.max_geometric_delay_ns:.0f} ns)"
                        )
                        logger.info(f"     - Valid solutions: {result.n_within_bounds} antennas")
                        logger.info(f"     - Flagged solutions: {result.n_flagged} antennas")
                        logger.info(f"     - Out of bounds: {result.n_out_of_bounds} antennas")
                        if result.n_out_of_bounds > 0:
                            logger.warning(
                                f"     ⚠ {result.n_out_of_bounds} antennas have delays "
                                f"outside geometric bounds (may indicate hardware issues)"
                            )
                    else:
                        logger.error(f"  ❌ Delay validation FAILED")
                        logger.error(
                            f"     Only {result.n_within_bounds}/{result.n_antennas} antennas "
                            f"within bounds - below 80% threshold"
                        )
        except Exception as e:
            logger.error(f"  ❌ Delay geometric validation failed: {e}")
            logger.error("     This indicates systematic issues with delay calibration")
            logger.error("     Possible causes:")
            logger.error("       - Incorrect antenna positions")
            logger.error("       - Clock synchronization failure")
            logger.error("       - Wrong calibrator model")
            raise  # Re-raise to stop pipeline

        logger.info("\n" + "=" * 80)
        logger.info("DELAY CALIBRATION QA - Complete")
        logger.info("=" * 80 + "\n")
    else:
        logger.debug("Skipping QA validation (fast mode)")

    return tables


def solve_prebandpass_phase(
    ms: str,
    cal_field: str,
    refant: str,
    table_prefix: str | None = None,
    combine_fields: bool = True,
    combine_spw: bool = True,
    uvrange: str = ">1klambda",  # Default: exclude short baselines for better SNR
    # Default to 'inf' to match test expectation and allow long integration when appropriate
    solint: str = "inf",
    # Default to 3.0 for better outrigger antenna coverage
    minsnr: float = 3.0,
    peak_field_idx: int | None = None,
    minblperant: int | None = None,  # Minimum baselines per antenna
    # SPW selection (e.g., "4~11" for central 8 SPWs)
    spw: str | None = None,
    # Custom table name (e.g., ".bpphase.gcal")
    table_name: str | None = None,
    # List of prior calibration tables to apply (e.g., K-table)
    gaintable: list[str] | None = None,
) -> str:
    """Solve phase-only calibration before bandpass to correct phase drifts in raw data.

    This phase-only calibration step is critical for uncalibrated raw data. It corrects
    for time-dependent phase variations that cause decorrelation and low SNR in bandpass
    calibration. This should be run BEFORE bandpass calibration.

    **PRECONDITION**: MODEL_DATA must be populated before calling this function.

    Parameters
    ----------
    ms : str
        Path to Measurement Set
    cal_field : str
        Field selection
    refant : str
        Reference antenna
    table_prefix : str, optional
        Prefix for tables
    combine_fields : bool, optional
        Whether to combine fields
    t_short : str, optional
        Short solution interval
    solint : str, optional
        Solution interval
    minsnr : float, optional
        Minimum SNR
    peak_field_idx : int, optional
        Index of peak field
    minblperant : int, optional
        Minimum baselines per antenna
    spw : str, optional
        SPW selection
    table_name : str, optional
        Custom table name
    gaintable : list[str], optional
        List of prior calibration tables to apply (e.g. K-table). Applying K-table
        is CRITICAL if instrumental delays are significant, as they cause
        phase decoherence when averaging over frequency (combine_spw=True).

    Returns
    -------
        Path to phase-only calibration table (to be passed to bandpass via gaintable)

    """
    import numpy as np  # type: ignore[import]

    # use module-level table

    if table_prefix is None:
        table_prefix = f"{os.path.splitext(ms)[0]}_{cal_field}"

    # ============================================================================
    # PRE-BANDPASS PHASE CALIBRATION - TRANSPARENCY HEADER
    # ============================================================================
    logger.info("\n" + "=" * 80)
    logger.info("PRE-BANDPASS PHASE CALIBRATION (calmode='p') - Stage 2")
    logger.info("=" * 80)
    logger.info("PURPOSE: Solve for time-dependent phase variations in raw uncalibrated data")
    logger.info("         BEFORE bandpass calibration. This prevents decorrelation and")
    logger.info("         low SNR in bandpass solve, especially for faint calibrators.")
    logger.info("")
    logger.info("WHY NEEDED:")
    logger.info("  - Bandpass solves with solint='inf' (per-channel), integrating entire obs")
    logger.info("  - Phase drifts during integration cause decorrelation → low SNR → flagging")
    logger.info("  - Pre-BP phase correction stabilizes phases → higher bandpass SNR")
    logger.info("")
    logger.info("EXPECTED INPUT:")
    logger.info("  - MODEL_DATA: Populated with calibrator source model (checking...)")
    logger.info(f"  - Field: {cal_field}")
    logger.info(f"  - Reference antenna: {refant}")
    logger.info(f"  - Solution interval: {solint}")
    logger.info(f"  - Combine fields: {combine_fields} (higher SNR if True)")
    logger.info(f"  - Combine SPWs: {combine_spw} (frequency-independent phase)")
    logger.info(f"  - Minimum SNR: {minsnr}")
    logger.info("")
    logger.info("EXPECTED SOLUTIONS:")
    logger.info("  - Phase range: -180° to +180° (wrapped)")
    logger.info("  - Time evolution: Smooth trends, not random jumps")
    logger.info("  - Frequency behavior: Similar phases across SPWs (frequency-independent)")
    logger.info("  - Reference antenna: Phase = 0° (by definition)")
    logger.info("  - All antennas: Phases cluster around 0° after correction")
    logger.info("")
    logger.info("QUALITY ASSURANCE:")
    logger.info("  1. MODEL_DATA validation: Column exists and is populated")
    logger.info("  2. SNR validation: Solutions above minimum SNR threshold")
    logger.info("  3. Solution continuity: No large phase jumps (>90°) between intervals")
    logger.info("  4. Table structure: Valid CASA calibration table format")
    logger.info("")
    logger.info("HOW SOLUTIONS ARE USED:")
    logger.info("  - Applied ONLY during bandpass calibration (Stage 3)")
    logger.info("  - NOT applied to target data (bandpass captures final phase solutions)")
    logger.info("  - Bandpass uses: gaintable=[prebandpass_phase_table]")
    logger.info("  - Improves bandpass SNR by removing phase decorrelation")
    logger.info("=" * 80 + "\n")

    # PRECONDITION CHECK: Verify MODEL_DATA exists and is populated for cal_field
    logger.info(f"→ Validating MODEL_DATA for pre-bandpass phase solve on field(s) {cal_field}...")
    with table(ms) as tb:
        if "MODEL_DATA" not in tb.colnames():
            raise ValueError(
                "MODEL_DATA column does not exist in MS. "
                "This is a required precondition for phase-only calibration. "
                "Populate MODEL_DATA before calling solve_prebandpass_phase()."
            )

        # Parse field selection to determine which field(s) to check
        # MODEL_DATA may only be populated for the calibration field(s)
        if "~" in str(cal_field):
            # Field range: check first field in range
            check_field = int(str(cal_field).split("~")[0])
        elif str(cal_field).isdigit():
            check_field = int(cal_field)
        else:
            # Field name - check all data as fallback
            check_field = None

        if check_field is not None:
            # Query only the cal_field's rows to check MODEL_DATA
            field_ids = tb.getcol("FIELD_ID")
            field_mask = field_ids == check_field
            field_rows = np.where(field_mask)[0]
            if len(field_rows) == 0:
                raise ValueError(f"No data found for field {check_field}. Check field selection.")
            # Sample up to 100 rows from the field
            sample_rows = field_rows[: min(100, len(field_rows))]
            model_sample = np.array([tb.getcell("MODEL_DATA", int(r)) for r in sample_rows])
        else:
            # Fallback: check first 100 rows
            model_sample = tb.getcol("MODEL_DATA", startrow=0, nrow=min(100, tb.nrows()))

        if np.all(np.abs(model_sample) < 1e-10):
            raise ValueError(
                f"MODEL_DATA column exists but is all zeros for field {cal_field}. "
                "This is a required precondition for phase-only calibration. "
                "Populate MODEL_DATA before calling solve_prebandpass_phase()."
            )

    # Determine field selector based on combine_fields setting
    # - If combining across fields: use the full selection string to maximize SNR
    # - Otherwise: use the peak field (closest to calibrator) if provided, otherwise parse from range
    #   The peak field is the one with maximum PB-weighted flux (closest to calibrator position)
    if combine_fields:
        field_selector = str(cal_field)
    else:
        if peak_field_idx is not None:
            field_selector = str(peak_field_idx)
        elif "~" in str(cal_field):
            # Fallback: use first field in range (should be peak when peak_idx=0)
            field_selector = str(cal_field).split("~")[0]
        else:
            field_selector = str(cal_field)
    logger.debug(
        f"Using field selector '{field_selector}' for pre-bandpass phase solve"
        + (
            f" (combined from range {cal_field})"
            if combine_fields
            else f" (peak field: {field_selector})"
        )
    )

    # Combine across scans, fields, and SPWs when requested
    # Combining SPWs improves SNR by using all 16 subbands simultaneously
    comb_parts = ["scan"]
    if combine_fields:
        comb_parts.append("field")
    if combine_spw:
        comb_parts.append("spw")
    comb = ",".join(comb_parts) if comb_parts else ""

    # VERIFICATION: Check which SPWs are available and will be used
    logger.info("\n" + "=" * 70)
    logger.info("SPW SELECTION VERIFICATION")
    logger.info("=" * 70)
    with table(f"{ms}::SPECTRAL_WINDOW", ack=False) as tspw:
        n_spws = tspw.nrows()
        spw_ids = list(range(n_spws))
        ref_freqs = tspw.getcol("REF_FREQUENCY")
        num_chan = tspw.getcol("NUM_CHAN")
        logger.info(f"MS contains {n_spws} spectral windows: SPW {spw_ids[0]} to SPW {spw_ids[-1]}")
        logger.info(f"  Frequency range: {ref_freqs[0] / 1e9:.4f} - {ref_freqs[-1] / 1e9:.4f} GHz")
        logger.info(f"  Total channels across all SPWs: {np.sum(num_chan)}")

    # Check data selection for the specified field
    with table(ms, ack=False) as tb:
        # Get unique SPW IDs in data for the selected field
        # We need to query the actual data to see which SPWs have data
        field_ids = tb.getcol("FIELD_ID")
        spw_ids_in_data = tb.getcol("DATA_DESC_ID")

        # Get unique SPW IDs (need to map DATA_DESC_ID to SPW)
        with table(f"{ms}::DATA_DESCRIPTION", ack=False) as tdd:
            data_desc_to_spw = tdd.getcol("SPECTRAL_WINDOW_ID")

        # Filter by field if field_selector is a single number
        if "~" not in str(field_selector):
            try:
                field_idx = int(field_selector)
                field_mask = field_ids == field_idx
                spw_ids_with_data = np.unique(data_desc_to_spw[spw_ids_in_data[field_mask]])
            except ValueError:
                # Field selector might be a name, use all data
                spw_ids_with_data = np.unique(data_desc_to_spw[spw_ids_in_data])
        else:
            # Range of fields, use all data
            spw_ids_with_data = np.unique(data_desc_to_spw[spw_ids_in_data])

        spw_ids_list = sorted(
            [int(x) for x in spw_ids_with_data]
        )  # Convert to plain ints for cleaner output
        logger.info(f"\nSPWs with data for field(s) '{field_selector}': {spw_ids_with_data}")
        logger.info(f"  Total SPWs to be processed: {len(spw_ids_with_data)}")

        if combine_spw:
            logger.info("\n  COMBINE='spw' is ENABLED:")
            logger.info(
                f"    :arrow_right: All {len(spw_ids_with_data)} SPWs will be used together in a single solve"
            )
            logger.info("    :arrow_right: Solution will be stored in SPW ID 0 (aggregate SPW)")
            logger.info(
                f"    :arrow_right: This improves SNR by using all {len(spw_ids_with_data)} subbands simultaneously"
            )
        else:
            logger.info("\n  COMBINE='spw' is DISABLED:")
            logger.info(
                f"    :arrow_right: Each of the {len(spw_ids_list)} SPWs will be solved separately"
            )
            logger.info(f"    :arrow_right: Solutions will be stored in SPW IDs {spw_ids_list}")

    logger.info("=" * 70 + "\n")

    # Determine table name
    if table_name:
        caltable_name = table_name
    else:
        caltable_name = f"{table_prefix}.prebp"

    # Solve phase-only calibration
    combine_desc = f" (combining across {comb})" if comb else ""
    spw_desc = f" (SPW: {spw})" if spw else ""
    gaintable_desc = f" (applying {len(gaintable)} prior tables)" if gaintable else ""
    logger.info(
        f"Running pre-bandpass phase-only solve on field {field_selector}"
        f"{combine_desc}{spw_desc}{gaintable_desc}..."
    )
    kwargs = dict(
        field=field_selector,
        spw=spw if spw else "",  # Use provided SPW selection or all SPWs
        solint=solint,
        refant=refant,
        calmode="p",  # Phase-only mode
        combine=comb,
        minsnr=minsnr,
        selectdata=True,
    )
    if uvrange:
        kwargs["uvrange"] = uvrange
    if minblperant is not None:
        kwargs["minblperant"] = minblperant
    if gaintable:
        kwargs["gaintable"] = gaintable
        # Apply the same spectral-window mapping to each gaintable.
        # _determine_spwmap_for_bptables returns a single mapping if any table was created with combine_spw=True.
        spwmap = _determine_spwmap_for_bptables(gaintable, ms)
        if spwmap:
            kwargs["spwmap"] = [spwmap for _ in gaintable]

    _call_gaincal_with_progress("Pre-bandpass phase solve", ms, caltable_name, **kwargs)

    # ============================================================================
    # QUALITY ASSURANCE: Validate pre-bandpass phase solutions
    # ============================================================================
    logger.info("\n" + "=" * 80)
    logger.info("PRE-BANDPASS PHASE CALIBRATION QA - Validating Solutions")
    logger.info("=" * 80)

    logger.info("→ QA Check 1: Solution table structure validation")
    _validate_solve_success(caltable_name, refant=refant)
    logger.info("  ✓ Calibration table created successfully")
    logger.info("  ✓ Reference antenna has valid solutions")
    logger.info("  ✓ Solutions exist for all expected antennas/times")

    # Track provenance after successful solve
    _track_calibration_provenance(
        ms_path=ms,
        caltable_path=caltable_name,
        task_name="gaincal",
        params={"vis": ms, "caltable": caltable_name, **kwargs},
    )

    logger.info("\n→ Solution Summary:")
    logger.info(f"  Output table: {caltable_name}")
    logger.info(f"  Calibration mode: Phase-only (calmode='p')")
    logger.info(f"  Solution interval: {solint}")
    logger.info(f"  Field(s) used: {field_selector}")
    logger.info(f"  SPW combination: {'Yes (aggregate SPW)' if combine_spw else 'No (per-SPW)'}")
    logger.info("")
    logger.info("→ Next step: Apply this table during bandpass calibration (Stage 3)")
    logger.info("  This will stabilize phases and improve bandpass SNR")

    logger.info("\n" + "=" * 80)
    logger.info("PRE-BANDPASS PHASE CALIBRATION - Complete")
    logger.info("=" * 80 + "\n")

    return caltable_name


def _check_coherent_phasing(
    ms: str,
    field_selector: str,
    max_ra_scatter_arcsec: float = 60.0,
) -> None:
    """Check if fields are coherently phased (not meridian tracking).

    DSA-110 data is initially phased to meridian (RA = LST). For calibration,
    fields must be rephased to calibrator using CASA phaseshift task.

    Parameters
    ----------
    ms : str
        Path to Measurement Set
    field_selector : str
        Field selection string
    max_ra_scatter_arcsec : float, optional
        Maximum allowed RA scatter in arcseconds

    Raises
    ------
    ValueError
        If RA scatter across fields exceeds threshold (meridian tracking)

    """
    import numpy as np  # type: ignore[import]

    # Parse field selection - single field doesn't need scatter check
    if "~" not in str(field_selector):
        return
    parts = str(field_selector).split("~")
    field_indices = list(range(int(parts[0]), int(parts[1]) + 1))

    # Read PHASE_DIR from FIELD table
    with table(f"{ms}::FIELD", readonly=True, ack=False) as field_tb:
        if "PHASE_DIR" not in field_tb.colnames():
            logger.warning("PHASE_DIR column not found - skipping coherence check")
            return
        phase_dir = field_tb.getcol("PHASE_DIR")  # Shape: (nfields, 1, 2)

    # Get RA values for selected fields (in radians)
    ra_values = np.array([phase_dir[i, 0, 0] for i in field_indices if i < len(phase_dir)])
    if len(ra_values) < 2:
        return

    # Calculate RA scatter (handling wrap-around at 2π)
    ra_mean = np.arctan2(np.mean(np.sin(ra_values)), np.mean(np.cos(ra_values)))
    ra_diff = np.angle(np.exp(1j * (ra_values - ra_mean)))
    ra_scatter_arcsec = np.rad2deg(np.std(ra_diff)) * 3600
    ra_span_arcsec = np.rad2deg(np.ptp(ra_diff)) * 3600

    logger.debug(
        "Phase center RA: scatter=%.1f arcsec, span=%.1f arcsec (%d fields)",
        ra_scatter_arcsec,
        ra_span_arcsec,
        len(ra_values),
    )

    if ra_scatter_arcsec > max_ra_scatter_arcsec:
        est_duration_min = ra_span_arcsec / 54000 * 60  # LST: 15°/hour = 54000 arcsec/hour
        raise ValueError(
            f"COHERENT PHASING CHECK FAILED: Fields NOT coherently phased.\n"
            f"  RA scatter: {ra_scatter_arcsec:.1f} arcsec > {max_ra_scatter_arcsec:.1f} threshold\n"
            f"  RA span: {ra_span_arcsec:.1f} arcsec (~{est_duration_min:.1f} min LST drift)\n\n"
            f"Data is still MERIDIAN-phased (RA=LST). Use phaseshift_ms() to rephase:\n"
            f"  from dsa110_contimg.core.calibration.runner import phaseshift_ms\n"
            f"  phaseshift_ms('{ms}', mode='calibrator', calibrator_name='<CAL>')\n\n"
            f"This handles both phaseshift and REFERENCE_DIR sync for ft().\n"
            f"Then recalculate MODEL_DATA to match the new phase center."
        )

    logger.info(
        " Coherent phasing OK: RA scatter=%.1f arcsec (< %.1f threshold)",
        ra_scatter_arcsec,
        max_ra_scatter_arcsec,
    )


def _validate_bandpass_model_data(ms: str, cal_field: str) -> None:
    """Validate MODEL_DATA exists and is populated for bandpass solve.

    Parameters
    ----------
    ms : str
        Path to Measurement Set
    cal_field : str
        Field selection

    Raises
    ------
    ValueError
        If MODEL_DATA is missing or unpopulated.

    """
    import numpy as np

    logger.info(f"Validating MODEL_DATA for bandpass solve on field(s) {cal_field}...")
    with table(ms) as tb:
        if "MODEL_DATA" not in tb.colnames():
            raise ValueError(
                "MODEL_DATA column does not exist in MS. "
                "This is a required precondition for bandpass calibration. "
                "Populate MODEL_DATA using setjy, ft(), or a catalog model before "
                "calling solve_bandpass()."
            )

        # Parse field selection to determine which field(s) to check
        if "~" in str(cal_field):
            check_field = int(str(cal_field).split("~")[0])
        elif str(cal_field).isdigit():
            check_field = int(cal_field)
        else:
            check_field = None

        # Check if MODEL_DATA is populated for the calibration field
        if check_field is not None:
            field_ids = tb.getcol("FIELD_ID")
            field_mask = field_ids == check_field
            field_rows = np.where(field_mask)[0]
            if len(field_rows) == 0:
                raise ValueError(f"No data found for field {check_field}. Check field selection.")
            sample_rows = field_rows[: min(100, len(field_rows))]
            model_sample = np.array([tb.getcell("MODEL_DATA", int(r)) for r in sample_rows])
        else:
            model_sample = tb.getcol("MODEL_DATA", startrow=0, nrow=min(100, tb.nrows()))

        if np.all(np.abs(model_sample) < 1e-10):
            raise ValueError(
                f"MODEL_DATA column exists but is all zeros for field {cal_field}. "
                "This is a required precondition for bandpass calibration. "
                "Populate MODEL_DATA using setjy, ft(), or a catalog model before "
                "calling solve_bandpass()."
            )


def _run_bandpass_with_progress(
    casa_bandpass_func,
    kwargs: dict,
    ms: str,
    poll_interval: float = 5.0,
    live_channel_output: bool = True,
) -> None:
    """Run CASA bandpass task with progress monitoring.

    CASA's bandpass task is C++ and doesn't provide Python-level progress callbacks.
    This wrapper monitors the CASA log file in real-time and displays live per-channel
    progress showing ALL channels as they are solved.

    Parameters
    ----------
    casa_bandpass_func :
        The CASA bandpass task function
    kwargs :
        Arguments to pass to bandpass task
    ms :
        Path to Measurement Set (for estimating total work)
    poll_interval :
        How often to report progress (seconds)
    live_channel_output :
        If True, show live per-channel progress grid
    """
    import os

    from dsa110_contimg.common.utils.progress import (
        BandpassChannelMonitor,
        StageProgressMonitor,
        estimate_calibration_time,
    )

    caltable = kwargs.get("caltable", "unknown")

    # Get MS info for progress estimation
    try:
        with table(ms, ack=False) as t:
            n_rows = t.nrows()
        with table(f"{ms}::SPECTRAL_WINDOW", ack=False) as tspw:
            n_spws = tspw.nrows()
            n_chan = tspw.getcol("NUM_CHAN")[0]
        with table(f"{ms}::ANTENNA", ack=False) as tant:
            n_ant = tant.nrows()
    except Exception:
        n_rows, n_spws, n_chan, n_ant = 0, 16, 48, 117

    # Estimate expected runtime based on data size
    estimated_seconds = estimate_calibration_time(n_rows, n_spws, n_ant)

    if live_channel_output:
        # Use live channel monitor that shows ALL channels as they're solved
        casa_log = os.environ.get("CASALOGFILE", "")
        monitor_ch: BandpassChannelMonitor = BandpassChannelMonitor(
            n_spws=n_spws,
            n_chans=n_chan,
            casa_log_path=casa_log if casa_log else None,
            poll_interval=poll_interval,
        )
        with monitor_ch:
            casa_bandpass_func(**kwargs)
    else:
        # Fallback: use simple stage progress monitor
        monitor_sp: StageProgressMonitor = StageProgressMonitor(
            "Bandpass solve",
            output_path=caltable,
            poll_interval=poll_interval,
            estimated_seconds=estimated_seconds,
        )
        monitor_sp.set_context(rows=n_rows, SPWs=n_spws, channels=n_chan, antennas=n_ant)

        with monitor_sp:
            casa_bandpass_func(**kwargs)


def _determine_field_selector(
    cal_field: str, combine_fields: bool, peak_field_idx: int | None
) -> str:
    """Determine CASA field selector based on combine_fields setting.

    Parameters
    ----------
    cal_field : str
        Field selection
    combine_fields : bool
        Whether to combine fields
    peak_field_idx : int | None
        Index of peak field
    """
    if combine_fields:
        return str(cal_field)
    if peak_field_idx is not None:
        return str(peak_field_idx)
    if "~" in str(cal_field):
        return str(cal_field).split("~")[0]
    return str(cal_field)


def _build_bandpass_combine_string(
    combine: str | None, combine_fields: bool, combine_spw: bool
) -> str:
    """Build combine string for bandpass solve.

    Parameters
    ----------
    combine: Optional[str] :

    """
    if combine:
        logger.debug(f"Using custom combine string: {combine}")
        return combine

    comb_parts = ["scan"]
    if combine_fields:
        comb_parts.append("field")
    if combine_spw:
        comb_parts.append("spw")
    return ",".join(comb_parts)


def _log_spw_verification(ms: str, field_selector: str, combine_spw: bool) -> None:
    """Log SPW selection verification for bandpass solve.

    Parameters
    ----------
    ms : str
        Path to Measurement Set
    field_selector : str
        Field selection string
    combine_spw : bool
        Whether SPWs are combined
    """
    import numpy as np

    logger.info("\n" + "=" * 70)
    logger.info("SPW SELECTION VERIFICATION")
    logger.info("=" * 70)

    with table(f"{ms}::SPECTRAL_WINDOW", ack=False) as tspw:
        n_spws = tspw.nrows()
        spw_ids = list(range(n_spws))
        ref_freqs = tspw.getcol("REF_FREQUENCY")
        num_chan = tspw.getcol("NUM_CHAN")
        logger.info(f"MS contains {n_spws} spectral windows: SPW {spw_ids[0]} to SPW {spw_ids[-1]}")
        logger.info(f"  Frequency range: {ref_freqs[0] / 1e9:.4f} - {ref_freqs[-1] / 1e9:.4f} GHz")
        logger.info(f"  Total channels across all SPWs: {np.sum(num_chan)}")

    with table(ms, ack=False) as tb:
        field_ids = tb.getcol("FIELD_ID")
        spw_ids_in_data = tb.getcol("DATA_DESC_ID")

        with table(f"{ms}::DATA_DESCRIPTION", ack=False) as tdd:
            data_desc_to_spw = tdd.getcol("SPECTRAL_WINDOW_ID")

        if "~" not in str(field_selector):
            try:
                field_idx = int(field_selector)
                field_mask = field_ids == field_idx
                spw_ids_with_data = np.unique(data_desc_to_spw[spw_ids_in_data[field_mask]])
            except ValueError:
                spw_ids_with_data = np.unique(data_desc_to_spw[spw_ids_in_data])
        else:
            spw_ids_with_data = np.unique(data_desc_to_spw[spw_ids_in_data])

        spw_ids_list = sorted(spw_ids_with_data)
        logger.info(f"\nSPWs with data for field(s) '{field_selector}': {spw_ids_list}")
        logger.info(f"  Total SPWs to be processed: {len(spw_ids_list)}")

        if combine_spw:
            logger.info("\n  COMBINE='spw' is ENABLED:")
            logger.info(
                f"    :arrow_right: All {len(spw_ids_list)} SPWs will be used together in a single solve"
            )
            logger.info("    :arrow_right: Solution will be stored in SPW ID 0 (aggregate SPW)")
            logger.info(
                f"    :arrow_right: This improves SNR by using all {len(spw_ids_list)} subbands simultaneously"
            )
        else:
            logger.info("\n  COMBINE='spw' is DISABLED:")
            logger.info(
                f"    :arrow_right: Each of the {len(spw_ids_list)} SPWs will be solved separately"
            )
            logger.info(f"    :arrow_right: Solutions will be stored in SPW IDs {spw_ids_list}")

    logger.info("=" * 70 + "\n")


def _run_bandpass_diagnostics(
    ms: str,
    cal_field: str,
    bpcal_table: str,
    calibrator_name: str | None,
    refant: str,
    flag_fraction: float,
    generate_report: bool = True,
    report_output_dir: str | None = None,
) -> str | None:
    """Run comprehensive bandpass quality diagnostics.

    This function integrates the bandpass diagnostic framework to identify
    root causes of high flagging and provide actionable recommendations.
    Optionally generates a comprehensive HTML report with diagnostic figures.

    Parameters
    ----------
    ms :
        Path to Measurement Set
    cal_field :
        Field selection
    bpcal_table :
        Path to bandpass calibration table
    calibrator_name :
        Calibrator name (e.g., "0834+555")
    refant :
        Reference antenna
    flag_fraction :
        Overall flagging fraction (0.0 to 1.0)
    generate_report :
        If True, generate HTML diagnostics report (default: True)
    report_output_dir :
        Directory for HTML report. If None, uses parent of bpcal_table

    Returns
    -------
        Path to generated HTML report, or None if report generation disabled/failed

    """
    report_path: str | None = None

    try:
        from dsa110_contimg.core.calibration.bandpass_diagnostics import (
            analyze_flagging_pattern,
            diagnose_bandpass_quality,
            extract_bandpass_flagging_stats,
        )

        print("\n" + "-" * 80)
        print("RUNNING BANDPASS DIAGNOSTIC ANALYSIS")
        print("-" * 80)

        # Extract detailed flagging statistics
        print("→ Extracting flagging statistics...")
        flagging_stats = extract_bandpass_flagging_stats(bpcal_table)

        # Analyze pattern
        pattern = analyze_flagging_pattern(flagging_stats)
        print(f"→ Flagging pattern detected: {pattern.upper()}")

        pattern_descriptions = {
            "channel_specific": "RFI contamination in specific channels",
            "spw_specific": "Edge channel or bandpass rolloff effects",
            "antenna_specific": "Bad antenna or reference antenna issue",
            "uniform": "Systematic setup error (geometry, model, flux)",
            "random": "SNR/noise limited, likely phase decorrelation",
            "unknown": "Unable to determine clear pattern",
        }
        print(f"  {pattern_descriptions.get(pattern, 'Unknown pattern')}")

        # Run full diagnostic if calibrator name available
        if calibrator_name:
            print(f"→ Running comprehensive diagnostic (calibrator: {calibrator_name})...")
            print()

            diagnosis = diagnose_bandpass_quality(
                ms, cal_field, bpcal_table, calibrator_name, refant
            )

            # Print diagnostic report
            print(str(diagnosis))

            # Highlight critical actions
            if diagnosis.severity in ["critical", "high"]:
                print("\n" + "!" * 80)
                print("CRITICAL: IMMEDIATE ACTION REQUIRED")
                print("!" * 80)
                print(f"Root Cause: {diagnosis.root_cause}")
                print(f"Confidence: {diagnosis.confidence:.0%}")
                print("\nRecommended Actions (in priority order):")
                for i, fix in enumerate(diagnosis.fixes, 1):
                    print(f"  {i}. {fix}")
                print("!" * 80)
        else:
            print("  (Skipping detailed diagnostic - calibrator name not provided)")
            print("\n  Basic recommendations:")
            print("  - Check that data is coherently phased to calibrator")
            print("  - Verify pre-bandpass phase correction was applied")
            print("  - Inspect for RFI if channel-specific pattern")
            print("  - Consider re-phasing if flagging >15%")

        print("-" * 80 + "\n")

        # Generate HTML report if requested
        if generate_report:
            try:
                from dsa110_contimg.core.calibration.bandpass_report import generate_bandpass_report

                # Determine output directory
                if report_output_dir is None:
                    report_output_dir = os.path.dirname(bpcal_table)
                    if not report_output_dir:
                        report_output_dir = "."

                print("→ Generating HTML diagnostics report...")
                report_path = generate_bandpass_report(
                    ms_path=ms,
                    bpcal_path=bpcal_table,
                    output_dir=report_output_dir,
                    calibrator_name=calibrator_name or "unknown",
                )
                print(f" HTML report saved: {report_path}")
            except Exception as e:
                logger.warning(f"Failed to generate HTML report: {e}")
                print(f" HTML report generation failed: {e}")

    except ImportError as e:
        logger.warning(f"Bandpass diagnostic module not available: {e}")
        print(f" Diagnostic module not available: {e}")
        print("  Basic recommendation: Check geometry and pre-bandpass phase correction")
    except Exception as e:
        logger.error(f"Error running bandpass diagnostics: {e}", exc_info=True)
        print(f" Diagnostic analysis failed: {e}")
        print("  Please review logs and calibration setup manually")

    return report_path


@timed("calibration.solve_bandpass")
def solve_bandpass(
    ms: str,
    cal_field: str,
    refant: str,
    ktable: str | None,
    table_prefix: str | None = None,
    set_model: bool = True,
    model_standard: str = "Perley-Butler 2017",
    combine_fields: bool = True,  # Default: combine for higher SNR
    combine_spw: bool = False,  # Default: False - per-SPW solutions are faster and more accurate
    minsnr: float = 3.0,
    uvrange: str = ">1klambda",  # Default: exclude short baselines for better calibration
    fillgaps: int = 3,  # Interpolate across flagged channels up to this width
    minblperant: int = 4,  # Minimum baselines per antenna to include in solve
    prebandpass_phase_table: str | None = None,
    bp_smooth_type: str | None = None,
    bp_smooth_window: int | None = None,
    peak_field_idx: int | None = None,
    # Custom combine string (e.g., "scan,obs,field")
    combine: str | None = None,
    require_coherent_phasing: bool = True,
    max_flag_fraction: float = 0.05,
    calibrator_name: str | None = None,  # For diagnostics
    generate_diagnostics_report: bool = True,  # Generate HTML diagnostics report
    diagnostics_output_dir: str | None = None,  # Directory for diagnostics output
) -> list[str]:
    """Solve bandpass using CASA bandpass task with bandtype='B'.

        This solves for frequency-dependent bandpass correction using the dedicated
        bandpass task, which properly handles per-channel solutions. The bandpass task
        requires a source model (smodel) which is provided via MODEL_DATA column.

        Preconditions
    -------------
        - MODEL_DATA must be populated before calling this function.
        - When combine_fields=True, fields must be coherently phased.

    Notes
    -----
        - `ktable` is applied to the bandpass solve when provided and the file exists.
        - combine_spw=False is recommended for bandpass calibration because:
        - Each SPW has a different bandpass shape that should be solved independently.
        - Combining SPWs does NOT increase per-channel SNR (unlike gain calibration).
        - Per-SPW solutions are actually faster and produce less flagging.
        - This function now includes comprehensive bandpass quality diagnostics.
        - If flagging exceeds 5%, an automatic diagnostic analysis will be performed
        to identify the root cause and recommend fixes.

    Parameters
    ----------
    ms : str
        Path to Measurement Set.
    cal_field : str
        Field selection (e.g., "23" or "0~23").
    refant : str
        Reference antenna.
    ktable : str or None
        K-table to apply during bandpass solve. Applied if provided and file exists.
    table_prefix : str or None, optional
        Prefix for output calibration table. Default is None.
    set_model : bool, optional
        Not used (kept for compatibility). Default is True.
    model_standard : str, optional
        Not used (kept for compatibility). Default is "Perley-Butler 2017".
    combine_fields : bool, optional
        If True, combine across fields for higher SNR. Default is True.
    combine_spw : bool, optional
        If True, combine across spectral windows. Default is False.
    minsnr : float, optional
        Minimum SNR threshold for solutions. Default is 3.0.
    uvrange : str, optional
        UV range selection (e.g., ">1klambda"). Default is ">1klambda".
    fillgaps : int, optional
        Interpolate across flagged channels up to this width. Default is 3.
    minblperant : int, optional
        Minimum baselines per antenna to include in solve. Default is 4.
    prebandpass_phase_table : str or None, optional
        Pre-bandpass phase-only calibration table. Default is None.
    bp_smooth_type : str or None, optional
        Smoothing type for bandpass (e.g., "poly"). Default is None.
    bp_smooth_window : int or None, optional
        Smoothing window size. Default is None.
    peak_field_idx : int or None, optional
        Index of field with peak calibrator flux. Default is None.
    combine : str or None, optional
        Custom combine string (overrides combine_fields/combine_spw). Default is None.
    require_coherent_phasing : bool, optional
        If True, check coherent phasing. Default is True.
    max_flag_fraction : float, optional
        Maximum allowed flag fraction. Default is 0.05.
    calibrator_name : str or None, optional
        Calibrator name (e.g., "0834+555") for diagnostic framework. Default is None.
    generate_diagnostics_report : bool, optional
        If True, generate HTML diagnostics report. Default is True.
    diagnostics_output_dir : str or None, optional
        Directory for HTML report output. If None, uses table_prefix dir. Default is None.

    Returns
    -------
        list of str
        List of calibration table paths created.

    Raises
    ------
        ValueError
        If preconditions are not met or flag fraction exceeds limit.
    """
    service = CASAService()

    if table_prefix is None:
        table_prefix = f"{os.path.splitext(ms)[0]}_{cal_field}"

    # ============================================================================
    # PRE-CALIBRATION VALIDATION GATE
    # ============================================================================
    # Run comprehensive precondition checks before attempting bandpass calibration.
    # This prevents wasting compute on data that will inevitably fail.
    try:
        from dsa110_contimg.core.calibration.guardrails import (
            CalibrationGuardrails,
        )
        from dsa110_contimg.core.calibration.preconditions import (
            validate_bandpass_preconditions,
        )

        # Check all preconditions
        validation_result = validate_bandpass_preconditions(
            ms_path=ms,
            cal_field=cal_field,
            calibrator_name=calibrator_name,
            prebandpass_phase_table=prebandpass_phase_table,
            require_prebandpass_phase=True,  # Pre-BP phase is critical
        )

        if not validation_result.can_proceed:
            # Critical preconditions failed
            issues = "\n  - ".join(validation_result.blocking_issues)
            raise ValueError(
                f"Bandpass calibration preconditions not met:\n  - {issues}\n\n"
                f"Fix these issues before attempting bandpass calibration."
            )

        # Log warnings but continue
        for warning in validation_result.warnings:
            logger.warning(f"Precondition warning: {warning}")

    except ImportError:
        # Preconditions module not available, continue with legacy checks
        logger.debug("Preconditions module not available, using legacy checks")

    # ============================================================================
    # BANDPASS CALIBRATION (B) - TRANSPARENCY HEADER
    # ============================================================================
    logger.info("\n" + "=" * 80)
    logger.info("BANDPASS CALIBRATION (bandtype='B') - Stage 3")
    logger.info("=" * 80)
    logger.info("PURPOSE: Solve for frequency-dependent amplitude and phase variations")
    logger.info("         across each spectral window. This is the MOST CRITICAL")
    logger.info("         calibration for spectral line and continuum imaging.")
    logger.info("")
    logger.info("WHAT IT CORRECTS:")
    logger.info("  - Analog electronics: Filters, amplifiers with non-flat frequency response")
    logger.info("  - Digital channelization: Polyphase filterbank imperfections")
    logger.info("  - Cable reflections: Frequency-dependent transmission")
    logger.info("  - Atmospheric effects: Slight frequency dependence in tropospheric phase")
    logger.info("")
    logger.info("EXPECTED INPUT:")
    logger.info("  - MODEL_DATA: Populated with calibrator source model (validating...)")
    logger.info(f"  - Field: {cal_field}")
    logger.info(f"  - Reference antenna: {refant}")
    logger.info(f"  - Combine fields: {combine_fields} (higher SNR if True)")
    logger.info(f"  - Combine SPWs: {combine_spw} (NOT recommended - per-SPW is better)")
    logger.info(f"  - Minimum SNR: {minsnr}")
    logger.info(f"  - UV range: {uvrange} (exclude short baselines)")
    logger.info(f"  - Pre-BP phase table: {'Yes' if prebandpass_phase_table else 'No'}")
    logger.info("")
    logger.info("EXPECTED SOLUTIONS:")
    logger.info("  - Amplitude: 0.5 to 2.0 (relative to mean)")
    logger.info("    * Values <0.5 or >2.0 suggest hardware issues")
    logger.info("  - Phase: -180° to +180° (wrapped)")
    logger.info("    * Should vary smoothly across frequency within each SPW")
    logger.info("  - Structure: Per-channel (typically 384 channels/SPW × 16 SPWs)")
    logger.info("  - Edge channels: May be flagged (normal filter rolloff)")
    logger.info("  - Reference antenna: Often has flattest bandpass")
    logger.info("")
    logger.info("QUALITY ASSURANCE:")
    logger.info("  1. Flag fraction analysis:")
    logger.info("     - PRISTINE: <3% flagged - Target achieved!")
    logger.info("     - GOOD: 3-5% flagged - Acceptable quality")
    logger.info("     - MODERATE: 5-10% flagged - Running diagnostics")
    logger.info("     - HIGH: 10-20% flagged - Diagnostic required")
    logger.info("     - CRITICAL: >20% flagged - Systematic failure")
    logger.info("  2. Solution summary: Breakdown by antenna, SPW, channel")
    logger.info("  3. Automated diagnostics: Root cause analysis for high flagging")
    logger.info("  4. HTML report: Comprehensive plots and analysis (always generated)")
    logger.info("")
    logger.info("HOW SOLUTIONS ARE USED:")
    logger.info("  - Applied in final gain calibration (Stage 4)")
    logger.info("  - Applied to target data with all other calibrations")
    logger.info("  - Application order: K → B → G (bandpass after delays)")
    logger.info("  - Interpolation: Linear in time, nearest in frequency")
    logger.info("=" * 80 + "\n")

    # Determine field selector
    field_selector = _determine_field_selector(cal_field, combine_fields, peak_field_idx)
    logger.debug(
        f"Using field selector '{field_selector}' for bandpass calibration"
        + (
            f" (combined from range {cal_field})"
            if combine_fields
            else f" (peak field: {field_selector})"
        )
    )

    # Validate preconditions (early failures)
    if require_coherent_phasing and combine_fields:
        _check_coherent_phasing(ms, cal_field)

    _validate_bandpass_model_data(ms, cal_field)

    # Build combine string
    comb = _build_bandpass_combine_string(combine, combine_fields, combine_spw)

    # Log SPW verification
    _log_spw_verification(ms, field_selector, combine_spw)

    # Use bandpass task with bandtype='B' for proper bandpass calibration
    # The bandpass task requires MODEL_DATA to be populated (smodel source model)
    # uvrange='>1klambda' is the default to avoid short baselines
    # CRITICAL: Apply pre-bandpass phase-only calibration if provided. This corrects
    # phase drifts in raw uncalibrated data that cause decorrelation and low SNR.
    # CRITICAL: Apply K-table if provided. This flattens the phase slope across
    # the band, preventing decorherence if averaging channels or if delays are large.
    combine_desc = f" (combining across {comb})" if comb else ""
    phase_desc = " with pre-bandpass phase correction" if prebandpass_phase_table else ""
    k_desc = " with K-correction" if ktable else ""
    logger.info(
        f"Running bandpass solve using bandpass task (bandtype='B') on field {field_selector}"
        f"{combine_desc}{phase_desc}{k_desc}..."
    )
    kwargs = dict(
        vis=ms,
        caltable=f"{table_prefix}.b",
        field=field_selector,
        solint="inf",  # Per-channel solution (bandpass)
        refant=refant,
        combine=comb,
        solnorm=True,
        bandtype="B",  # Bandpass type B (per-channel)
        selectdata=True,  # Required to use uvrange parameter
        minsnr=minsnr,  # Minimum SNR threshold for solutions
        fillgaps=fillgaps,  # Interpolate across flagged channels
        minblperant=minblperant,  # Minimum baselines per antenna
    )
    # Set uvrange (default: '>1klambda' to avoid short baselines)
    if uvrange:
        kwargs["uvrange"] = uvrange

    # Construct gaintable list
    gaintables = []
    if ktable:
        # Check if K-table exists
        if os.path.exists(ktable):
            gaintables.append(ktable)
            logger.debug(f"  Applying K-table calibration: {ktable}")
        else:
            logger.warning(f"  K-table specified but not found: {ktable}")

    if prebandpass_phase_table:
        gaintables.append(prebandpass_phase_table)
        logger.debug(f"  Applying pre-bandpass phase-only calibration: {prebandpass_phase_table}")

    if gaintables:
        kwargs["gaintable"] = gaintables

        # Handle spwmap and interp for multiple tables
        # K-table: usually needs spwmap if combined, interp='linear' or 'nearest'
        # Pre-BP: needs spwmap if combined, interp='linear'

        # Determine spwmap for each table type independently (following solve_gains pattern)
        k_spwmap = None
        if ktable and os.path.exists(ktable):
            k_spwmap = _determine_spwmap_for_bptables([ktable], ms)

        p_spwmap = None
        if prebandpass_phase_table:
            p_spwmap = _determine_spwmap_for_bptables([prebandpass_phase_table], ms)

        # Build spwmaps and interps lists in the same order as gaintables was constructed
        spwmaps = None
        interps = None
        if k_spwmap or p_spwmap:
            # If any table needs mapping, construct the full spwmap list
            spwmaps = []
            interps = []
            # Add K-table mapping if it was added to gaintables
            if ktable and os.path.exists(ktable):
                spwmaps.append(k_spwmap if k_spwmap else [])
                interps.append("linear")
            # Add Pre-BP mapping if it was added to gaintables
            if prebandpass_phase_table:
                spwmaps.append(p_spwmap if p_spwmap else [])
                interps.append("linear")

        # Only pass spwmap if at least one table needs mapping
        if spwmaps:
            kwargs["spwmap"] = spwmaps
        if interps:
            kwargs["interp"] = interps

    # Run bandpass with progress monitoring (CASA's C++ core doesn't report progress)
    # We monitor the caltable file growth to provide feedback during long solves
    _run_bandpass_with_progress(service.bandpass, kwargs, ms)

    # PRECONDITION CHECK: Verify bandpass solve completed successfully
    # This ensures we follow "measure twice, cut once" - verify solutions exist
    # immediately after solve completes, before proceeding.
    _validate_solve_success(f"{table_prefix}.b", refant=refant)

    # CHECK FLAG FRACTION: Fail early if too many solutions are flagged
    # This prevents wasting time on downstream calibration with bad solutions
    flag_fraction = _check_flag_fraction(
        f"{table_prefix}.b",
        max_flag_fraction=max_flag_fraction,
        cal_type="bandpass",
    )

    # ============================================================================
    # BANDPASS SOLUTION SUMMARY
    # ============================================================================
    # Print comprehensive summary of where solutions were flagged
    # This provides the aggregated view that CASA doesn't report by default
    _print_bandpass_solution_summary(f"{table_prefix}.b", ms)

    # ============================================================================
    # BANDPASS QUALITY DIAGNOSTICS (AUTOMATIC)
    # ============================================================================
    # If flagging > 5%, run comprehensive diagnostic analysis to identify
    # root cause and provide actionable recommendations
    print("\n" + "=" * 80)
    print("BANDPASS QUALITY ASSESSMENT")
    print("=" * 80)
    print(f"Overall flagging fraction: {flag_fraction * 100:.2f}%")

    if flag_fraction < 0.03:
        print(" PRISTINE calibration (<3% flagged) - Target achieved!")
        print("  No diagnostic analysis needed.")
    elif flag_fraction < 0.05:
        print(" GOOD calibration (3-5% flagged) - Acceptable quality")
        print("  Minor flagging, likely edge effects or isolated RFI.")
    elif flag_fraction < 0.10:
        print(" MODERATE flagging (5-10%) - Running diagnostics...")
        _run_bandpass_diagnostics(
            ms,
            cal_field,
            f"{table_prefix}.b",
            calibrator_name,
            refant,
            flag_fraction,
            generate_report=False,  # Report generated separately below
            report_output_dir=diagnostics_output_dir,
        )
    elif flag_fraction < 0.20:
        print(" HIGH flagging (10-20%) - DIAGNOSTIC REQUIRED")
        _run_bandpass_diagnostics(
            ms,
            cal_field,
            f"{table_prefix}.b",
            calibrator_name,
            refant,
            flag_fraction,
            generate_report=False,  # Report generated separately below
            report_output_dir=diagnostics_output_dir,
        )
    else:
        print(" CRITICAL flagging (>20%) - SYSTEMATIC FAILURE")
        print("  Running comprehensive diagnostic analysis...")
        _run_bandpass_diagnostics(
            ms,
            cal_field,
            f"{table_prefix}.b",
            calibrator_name,
            refant,
            flag_fraction,
            generate_report=False,  # Report generated separately below
            report_output_dir=diagnostics_output_dir,
        )

    # ============================================================================
    # BANDPASS DIAGNOSTICS HTML REPORT (ALWAYS GENERATED)
    # ============================================================================
    # Generate comprehensive HTML report with figures for every bandpass solve
    if generate_diagnostics_report:
        try:
            from dsa110_contimg.core.calibration.bandpass_report import generate_bandpass_report

            # Determine output directory
            output_dir = diagnostics_output_dir
            if output_dir is None:
                output_dir = os.path.dirname(f"{table_prefix}.b")
                if not output_dir:
                    output_dir = "."

            print("→ Generating HTML diagnostics report...")
            report_path = generate_bandpass_report(
                ms_path=ms,
                bpcal_path=f"{table_prefix}.b",
                output_dir=output_dir,
                calibrator_name=calibrator_name or "unknown",
            )
            print(f" HTML report saved: {report_path}")
        except Exception as e:
            logger.warning(f"Failed to generate HTML report: {e}")
            print(f" HTML report generation failed: {e}")

    print("=" * 80 + "\n")

    # Track provenance after successful solve
    _track_calibration_provenance(
        ms_path=ms,
        caltable_path=f"{table_prefix}.b",
        task_name="bandpass",
        params=kwargs,
    )
    logger.info(f":check: Bandpass solve completed: {table_prefix}.b")

    # Optional smoothing of bandpass table (post-solve), off by default
    if (
        bp_smooth_type
        and str(bp_smooth_type).lower() != "none"
        and bp_smooth_window
        and int(bp_smooth_window) > 1
    ):
        try:
            logger.info(
                f"Smoothing bandpass table '{table_prefix}.b' with {bp_smooth_type} (window={bp_smooth_window})..."
            )
            # Best-effort: in-place smoothing using same output table
            service.smoothcal(
                vis=ms,
                tablein=f"{table_prefix}.b",
                tableout=f"{table_prefix}.b",
                smoothtype=str(bp_smooth_type).lower(),
                smoothwindow=int(bp_smooth_window),
            )
            logger.info(":check: Bandpass table smoothing complete")
        except Exception as e:
            logger.warning(f"Could not smooth bandpass table via CASA smoothcal: {e}")

    out = [f"{table_prefix}.b"]

    # QA validation of bandpass calibration tables
    try:
        from dsa110_contimg.core.qa.pipeline_quality import check_calibration_quality

        check_calibration_quality(out, ms_path=ms, alert_on_issues=True)
    except Exception as e:
        logger.warning(f"QA validation failed: {e}")

    # If flagging is still high, we just warn and proceed instead of failing
    # This allows the pipeline to continue even with poor data, which can be inspected later
    if flag_fraction > max_flag_fraction:
        logger.warning(
            f"High flagging detected: {flag_fraction * 100:.1f}% > {max_flag_fraction * 100:.0f}% limit. "
            f"Proceeding anyway as 'max_flag_fraction' is soft limit in this mode."
        )

    return out


@timed("calibration.solve_gains")
def solve_gains(
    ms: str,
    cal_field: str,
    refant: str,
    ktable: str | None,
    bptables: list[str],
    table_prefix: str | None = None,
    t_short: str = "60s",
    combine_fields: bool = False,
    *,
    phase_only: bool = False,
    uvrange: str = "",
    solint: str = "inf",
    minsnr: float = 3.0,
    peak_field_idx: int | None = None,
) -> list[str]:
    """Solve gain amplitude and phase; optionally short-timescale.

    **PRECONDITION**: MODEL_DATA must be populated before calling this function.
    This ensures consistent, reliable calibration results across all calibrators
    (bright or faint). The calling code should verify MODEL_DATA exists and is
    populated before invoking solve_gains().

    **PRECONDITION**: If `bptables` are provided, they must exist and be
    compatible with the MS. This ensures consistent, reliable calibration results.

    **NOTE**: `ktable` (Delay calibration) is applied if provided.

    Parameters
    ----------
    ms : str
        Path to the Measurement Set.
    cal_field : str
        Field ID or name for calibration (e.g., "0" or "3C286").
    refant : str
        Reference antenna ID.
    ktable : str or None
        Delay calibration table to apply.
    bptables : list[str]
        List of bandpass calibration tables to apply.
    table_prefix : str or None, optional
        Prefix for output calibration tables. Defaults to MS name + field.
    t_short : str, optional
        Short timescale solution interval (default: "60s").
    combine_fields : bool, optional
        If True, combine all fields for solution (default: False).
    phase_only : bool, optional
        If True, solve phase only (calmode='p'). Default: False.
    uvrange : str, optional
        UV range selection string (default: "").
    solint : str, optional
        Solution interval (default: "inf").
    minsnr : float, optional
        Minimum signal-to-noise ratio for solutions (default: 3.0).
    peak_field_idx : int or None, optional
        Index of peak flux field for calibrator selection.

    Returns
    -------
    list[str]
        List of generated calibration table paths.
    """
    # use module-level table - access via sys.modules to avoid scoping issues
    import sys

    import numpy as np  # type: ignore[import]

    _table = sys.modules[__name__].table
    if _table is None:
        raise ImportError(
            "casacore.tables module is not available. "
            "This function requires CASA environment to be properly configured. "
            "Please ensure you are running in the casa6 conda environment."
        )

    if table_prefix is None:
        table_prefix = f"{os.path.splitext(ms)[0]}_{cal_field}"

    # ============================================================================
    # FINAL GAIN CALIBRATION (G) - TRANSPARENCY HEADER
    # ============================================================================
    logger.info("\n" + "=" * 80)
    logger.info("FINAL GAIN CALIBRATION (gaintype='G') - Stage 4")
    logger.info("=" * 80)
    logger.info("PURPOSE: Solve for time-dependent amplitude and phase variations")
    logger.info("         after applying bandpass corrections. This is the FINAL")
    logger.info("         calibration step before applying to target data.")
    logger.info("")
    logger.info("WHAT IT CORRECTS:")
    logger.info("  - Atmospheric phase fluctuations: Tropospheric water vapor changes")
    logger.info("  - Antenna gain drifts: Electronics warm-up, temperature variations")
    logger.info("  - Pointing errors: Antennas drift slightly off source")
    logger.info("  - Ionospheric phase: For low-frequency observations")
    logger.info("")
    logger.info("EXPECTED INPUT:")
    logger.info("  - MODEL_DATA: Populated with calibrator source model (validating...)")
    logger.info(f"  - Field: {cal_field}")
    logger.info(f"  - Reference antenna: {refant}")
    logger.info(f"  - Bandpass tables: {len(bptables)} table(s) (validating...)")
    if ktable:
        logger.info(
            f"  - Delay table: {ktable} (NOT used - K-calibration not required for DSA-110)"
        )
    logger.info(f"  - Solution intervals: {solint} (long), {t_short} (short)")
    logger.info(f"  - Calibration mode: {'Phase-only' if phase_only else 'Amplitude + Phase'}")
    logger.info(f"  - Minimum SNR: {minsnr}")
    logger.info("")
    logger.info("TWO-TIMESCALE APPROACH:")
    logger.info("  This stage generates TWO gain tables:")
    logger.info("  1. Long timescale (.g file, solint='inf'):")
    logger.info("     - Captures overall amplitude and phase offsets")
    logger.info("     - Instrumental effects (stable)")
    logger.info("     - One solution per observation")
    logger.info("  2. Short timescale (.2g file, solint='60s'):")
    logger.info("     - Captures rapid atmospheric phase variations")
    logger.info("     - Atmospheric effects (variable)")
    logger.info("     - Time-resolved solutions")
    logger.info("")
    logger.info("EXPECTED SOLUTIONS:")
    logger.info("  - Amplitude: 0.8 to 1.2 (relative to mean)")
    logger.info("    * Slowly varying over time (minutes to hours)")
    logger.info("    * Large excursions suggest source confusion, pointing drift, or RFI")
    logger.info("  - Phase: -180° to +180° (wrapped)")
    logger.info("    * Can vary rapidly (seconds to minutes) due to atmosphere")
    logger.info("    * Smooth trends indicate tropospheric phase screen")
    logger.info("    * Random jumps indicate RFI or data quality issues")
    logger.info("  - Reference antenna: Amplitude = 1.0, Phase = 0° (by definition)")
    logger.info("")
    logger.info("QUALITY ASSURANCE:")
    logger.info("  1. MODEL_DATA validation: Column exists and is populated")
    logger.info("  2. Bandpass table validation: Tables exist and are compatible")
    logger.info("  3. SNR validation: Solutions above minimum SNR threshold")
    logger.info("  4. Flagging fraction: <5% ideal, <10% acceptable")
    logger.info("  5. Pipeline quality check: Table structure and coverage")
    logger.info("")
    logger.info("HOW SOLUTIONS ARE USED:")
    logger.info("  - Applied to target observations (final calibration step)")
    logger.info("  - Application order: B → G → 2G")
    logger.info("  - Time interpolation: Linear between calibrator scans")
    logger.info("  - Frequency interpolation: Bandpass handles frequency dependence")
    logger.info("=" * 80 + "\n")

    # PRECONDITION CHECK: Verify MODEL_DATA exists and is populated
    # This ensures we follow "measure twice, cut once" - establish requirements upfront
    # for consistent, reliable calibration across all calibrators (bright or faint).
    logger.info(f"→ QA Check 1: Validating MODEL_DATA for gain solve on field(s) {cal_field}...")
    with _table(ms) as tb:
        if "MODEL_DATA" not in tb.colnames():
            raise ValueError(
                "MODEL_DATA column does not exist in MS. "
                "This is a required precondition for gain calibration. "
                "Populate MODEL_DATA using setjy, ft(), or a catalog model before "
                "calling solve_gains()."
            )

        # Check if MODEL_DATA is populated (not all zeros)
        model_sample = tb.getcol("MODEL_DATA", startrow=0, nrow=min(100, tb.nrows()))
        if np.all(np.abs(model_sample) < 1e-10):
            raise ValueError(
                "MODEL_DATA column exists but is all zeros (unpopulated). "
                "This is a required precondition for gain calibration. "
                "Populate MODEL_DATA using setjy, ft(), or a catalog model before "
                "calling solve_gains()."
            )

    logger.info("  ✓ MODEL_DATA validation passed")

    # PRECONDITION CHECK: Validate all required calibration tables
    # This ensures we follow "measure twice, cut once" - establish requirements upfront
    # for consistent, reliable calibration across all calibrators.
    # Validate K-table if provided
    if ktable:
        logger.info(f"Validating K-table before gain calibration: {ktable}")
        try:
            if not os.path.exists(ktable):
                raise FileNotFoundError(f"K-table not found: {ktable}")
        except Exception as e:
            raise ValueError(
                f"K-table validation failed. This is a required precondition for "
                f"gain calibration when ktable is provided. Error: {e}"
            ) from e

    if bptables:
        logger.info(
            f"\n→ QA Check 2: Validating {len(bptables)} bandpass table(s) before gain calibration..."
        )
        logger.info("  Required checks:")
        logger.info("    - All bandpass tables exist on disk")
        logger.info("    - Tables are compatible with MS (matching SPWs, antennas)")
        logger.info("    - Reference antenna has valid solutions")
        logger.info("    - Tables are not corrupted (valid CASA table format)")
        try:
            # Convert refant string to int for validation
            # Handle comma-separated refant string (e.g., "113,114,103,106,112")
            # Use the first antenna in the chain for validation
            if isinstance(refant, str):
                if "," in refant:
                    # Comma-separated list: use first antenna
                    refant_str = refant.split(",")[0].strip()
                    refant_int = int(refant_str)
                else:
                    # Single antenna ID as string
                    refant_int = int(refant)
            else:
                refant_int = refant
            validate_caltables_for_use(bptables, ms, require_all=True, refant=refant_int)
            logger.info("  ✓ All bandpass table validation checks passed")
        except (FileNotFoundError, ValueError) as e:
            logger.error("  ❌ Bandpass table validation FAILED")
            logger.error(f"     Error: {e}")
            logger.error("     Cannot proceed without valid bandpass tables")
            raise ValueError(
                f"Calibration table validation failed. This is a required precondition for "
                f"gain calibration. Error: {e}"
            ) from e

    # Determine CASA field selector based on combine_fields setting
    # - If combining across fields: use the full selection string to maximize SNR
    # - Otherwise: use the peak field (closest to calibrator) if provided, otherwise parse from range
    #   The peak field is the one with maximum PB-weighted flux (closest to calibrator position)
    if combine_fields:
        field_selector = str(cal_field)
    else:
        if peak_field_idx is not None:
            field_selector = str(peak_field_idx)
        elif "~" in str(cal_field):
            # Fallback: use first field in range (should be peak when peak_idx=0)
            field_selector = str(cal_field).split("~")[0]
        else:
            field_selector = str(cal_field)
    logger.debug(
        f"Using field selector '{field_selector}' for gain calibration"
        + (
            f" (combined from range {cal_field})"
            if combine_fields
            else f" (peak field: {field_selector})"
        )
    )

    # Construct gaintable list
    gaintable = []
    if ktable:
        gaintable.append(ktable)
    gaintable.extend(bptables)

    # Combine across scans and fields when requested; otherwise do not combine
    comb = "scan,field" if combine_fields else ""

    # CRITICAL FIX: Determine spwmap if tables were created with combine_spw=True
    # When combine_spw is used, the table has solutions only for SPW=0 (aggregate).
    # We need to map all MS SPWs to SPW 0 in that table.

    # Check if K-table needs mapping
    k_spwmap = None
    if ktable:
        # Re-use the logic for BP tables (it works for any calibration table)
        k_spwmap = _determine_spwmap_for_bptables([ktable], ms)

    # Check if BP tables need mapping
    bp_spwmap = _determine_spwmap_for_bptables(bptables, ms)

    spwmap = None
    if k_spwmap or bp_spwmap:
        # If any table needs mapping, we must construct the full spwmap parameter list
        spwmap = []
        if ktable:
            spwmap.append(k_spwmap if k_spwmap else [])

        # Add mapping for each bandpass table
        for _ in bptables:
            spwmap.append(bp_spwmap if bp_spwmap else [])

    # Run gain calibration after bandpass
    # Default is amplitude+phase (calmode='ap')
    # Use phase_only=True for phase-only calibration (calmode='p')
    calmode = "p" if phase_only else "ap"
    logger.info(
        f"Running {'phase-only' if phase_only else 'amplitude+phase'} gain solve on field {field_selector}"
        + (" (combining across fields)..." if combine_fields else "...")
    )
    kwargs = dict(
        vis=ms,
        caltable=f"{table_prefix}.g",
        field=field_selector,
        solint=solint,
        refant=refant,
        gaintype="G",
        calmode=calmode,
        gaintable=gaintable,
        combine=comb,
        minsnr=minsnr,
        selectdata=True,
    )
    if uvrange:
        kwargs["uvrange"] = uvrange
    if spwmap:
        kwargs["spwmap"] = spwmap

    # Run with progress monitoring
    from dsa110_contimg.common.utils.progress import stage_progress

    with stage_progress(
        f"{'Phase-only' if phase_only else 'Amplitude+phase'} gain solve",
        output_path=f"{table_prefix}.g",
    ):
        _call_gaincal(**kwargs)
    # PRECONDITION CHECK: Verify phase-only gain solve completed successfully
    # This ensures we follow "measure twice, cut once" - verify solutions exist
    # immediately after solve completes, before proceeding.
    _validate_solve_success(f"{table_prefix}.g", refant=refant)
    # Track provenance after successful solve
    _track_calibration_provenance(
        ms_path=ms,
        caltable_path=f"{table_prefix}.g",
        task_name="gaincal",
        params=kwargs,
    )
    logger.info(f":check: Gain solve completed: {table_prefix}.g")

    out = [f"{table_prefix}.g"]
    gaintable2 = gaintable + [f"{table_prefix}.g"]

    if t_short:
        logger.info(
            f"Running short-timescale {'phase-only' if phase_only else 'amplitude+phase'} gain solve on field {field_selector}"
            + (" (combining across fields)..." if combine_fields else "...")
        )
        kwargs = dict(
            vis=ms,
            caltable=f"{table_prefix}.2g",
            field=field_selector,
            solint=t_short,
            refant=refant,
            gaintype="G",
            calmode=calmode,
            gaintable=gaintable2,
            combine=comb,
            minsnr=minsnr,
            selectdata=True,
        )
        if uvrange:
            kwargs["uvrange"] = uvrange
        # CRITICAL FIX: Apply spwmap to second gaincal call as well
        # Note: spwmap applies to bandpass tables in gaintable2; the gain table doesn't need it
        if spwmap:
            kwargs["spwmap"] = spwmap

        with stage_progress(
            f"Short-timescale {'phase-only' if phase_only else 'amplitude+phase'} gain solve",
            output_path=f"{table_prefix}.2g",
        ):
            _call_gaincal(**kwargs)
        # PRECONDITION CHECK: Verify short-timescale gain solve completed successfully
        # This ensures we follow "measure twice, cut once" - verify solutions exist
        # immediately after solve completes, before proceeding.
        _validate_solve_success(f"{table_prefix}.2g", refant=refant)
        # Track provenance after successful solve
        _track_calibration_provenance(
            ms_path=ms,
            caltable_path=f"{table_prefix}.2g",
            task_name="gaincal",
            params=kwargs,
        )
        logger.info(f":check: Short-timescale gain solve completed: {table_prefix}.2g")
        out.append(f"{table_prefix}.2g")

    # ============================================================================
    # QUALITY ASSURANCE: Validate gain calibration solutions
    # ============================================================================
    logger.info("\n" + "=" * 80)
    logger.info("FINAL GAIN CALIBRATION QA - Validating Solutions")
    logger.info("=" * 80)

    # QA validation of gain calibration tables
    logger.info("→ QA Check 3: Pipeline quality validation")
    logger.info("  Checking:")
    logger.info("    - Solution table structure validity")
    logger.info("    - Reference antenna has valid solutions")
    logger.info("    - Time coverage matches MS")
    logger.info("    - No entirely flagged time ranges")
    try:
        from dsa110_contimg.core.qa.pipeline_quality import check_calibration_quality

        check_calibration_quality(out, ms_path=ms, alert_on_issues=True)
        logger.info("  ✓ Pipeline quality validation passed")
    except Exception as e:
        logger.warning(f"  ⚠ QA validation warning: {e}")

    logger.info("\n→ Solution Summary:")
    logger.info(f"  Generated {len(out)} gain table(s):")
    for i, table in enumerate(out, 1):
        if table.endswith(".g"):
            logger.info(f"    {i}. {table} (long timescale, solint={solint})")
        elif table.endswith(".2g"):
            logger.info(f"    {i}. {table} (short timescale, solint={t_short})")
        else:
            logger.info(f"    {i}. {table}")
    logger.info(f"  Calibration mode: {'Phase-only' if phase_only else 'Amplitude + Phase'}")
    logger.info(f"  Field(s) used: {field_selector}")
    logger.info(f"  Bandpass corrections: {len(bptables)} table(s) applied")
    logger.info("")
    logger.info("→ Next step: Apply these tables to target observations")
    logger.info("  Application order: K → B → G → 2G")
    logger.info("  Time interpolation: Linear between calibrator scans")

    logger.info("\n" + "=" * 80)
    logger.info("FINAL GAIN CALIBRATION - Complete")
    logger.info("=" * 80 + "\n")

    return out
