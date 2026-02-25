"""
Calibration table validation utilities for precondition checking.

Following "measure twice, cut once" philosophy: verify preconditions upfront
before expensive calibration operations.
"""

from __future__ import annotations

import logging
import os

# Ensure CASAPATH is set before importing CASA modules
from dsa110_contimg.common.utils.casa_init import ensure_casa_path

ensure_casa_path()

import casacore.tables as casatables  # type: ignore[import]
import numpy as np  # type: ignore[import]

from dsa110_contimg.common.utils.antenna_classification import (
    get_outrigger_antennas,
    select_outrigger_refant,
)

table = casatables.table  # noqa: N816

logger = logging.getLogger(__name__)


def validate_caltable_exists(caltable_path: str) -> None:
    """Verify that a calibration table exists and is readable.

    CASA calibration tables are directories (like Measurement Sets), not files.

    Parameters
    ----------
    caltable_path : str
        Path to the CASA calibration table.

    Raises
    ------
    FileNotFoundError
        If table doesn't exist.
    ValueError
        If table is empty or unreadable.
    """
    if not os.path.exists(caltable_path):
        raise FileNotFoundError(f"Calibration table does not exist: {caltable_path}")

    # CASA calibration tables are directories, not files (like MS files)
    if not os.path.isdir(caltable_path):
        raise ValueError(
            f"Calibration table path is not a directory: {caltable_path}. "
            f"CASA calibration tables must be directories."
        )

    # Try to open the table to verify it's readable
    try:
        with table(caltable_path, readonly=True) as tb:
            if tb.nrows() == 0:
                raise ValueError(f"Calibration table has no solutions: {caltable_path}")
    except Exception as e:
        raise ValueError(
            f"Calibration table is unreadable or corrupted: {caltable_path}. Error: {e}"
        ) from e


def validate_caltable_compatibility(
    caltable_path: str,
    ms_path: str,
    *,
    check_antennas: bool = True,
    check_frequencies: bool = True,
    check_spw: bool = True,
    refant: int | str | None = None,
) -> list[str]:
    """Validate that a calibration table is compatible with an MS.

    Checks:
    - Antenna compatibility (if check_antennas=True)
    - Frequency compatibility (if check_frequencies=True) - NOTE: incomplete
    - Spectral window compatibility (if check_spw=True)
    - Reference antenna has solutions (if refant provided)

    Parameters
    ----------
    caltable_path :
        Path to calibration table
    ms_path :
        Path to Measurement Set
    check_antennas :
        Whether to check antenna compatibility
    check_frequencies :
        Whether to check frequency compatibility (currently incomplete)
    check_spw :
        Whether to check SPW compatibility
    refant :
        Optional reference antenna ID to verify has solutions

    Returns
    -------
        List of warning messages (empty if all checks pass)

    Raises
    ------
    FileNotFoundError
        If MS or caltable doesn't exist
    ValueError
        If compatibility issues are found (critical errors)

    """
    warnings: list[str] = []

    # Read MS antenna list
    ms_antennas = set()
    if check_antennas:
        try:
            with table(f"{ms_path}/ANTENNA", readonly=True) as tb:
                ms_antennas = set(range(tb.nrows()))
        except Exception as e:
            raise ValueError(f"Failed to read MS antenna table: {ms_path}. Error: {e}") from e

    # Read MS frequency range
    ms_freq_min = None
    ms_freq_max = None
    ms_spw_ids = set()
    if check_frequencies or check_spw:
        try:
            with table(f"{ms_path}/SPECTRAL_WINDOW", readonly=True) as tb:
                chan_freqs = tb.getcol("CHAN_FREQ")
                if len(chan_freqs) > 0:
                    ms_freq_min = float(np.min(chan_freqs))
                    ms_freq_max = float(np.max(chan_freqs))

                if check_spw:
                    # Get SPW IDs from DATA_DESCRIPTION
                    try:
                        with table(f"{ms_path}/DATA_DESCRIPTION", readonly=True) as dd:
                            spw_map = dd.getcol("SPECTRAL_WINDOW_ID")
                            ms_spw_ids = set(int(spw_id) for spw_id in spw_map)
                    except (OSError, RuntimeError, KeyError):
                        # If DATA_DESCRIPTION doesn't exist, try to infer from SPECTRAL_WINDOW
                        ms_spw_ids = set(range(len(chan_freqs)))
        except Exception as e:
            raise ValueError(
                f"Failed to read MS spectral window table: {ms_path}. Error: {e}"
            ) from e

    # Read calibration table
    try:
        with table(caltable_path, readonly=True) as tb:
            if tb.nrows() == 0:
                raise ValueError(f"Calibration table has no solutions: {caltable_path}")

            # Check antenna compatibility
            if check_antennas:
                cal_antennas = set()
                if "ANTENNA1" in tb.colnames():
                    cal_antennas.update(tb.getcol("ANTENNA1"))
                if "ANTENNA2" in tb.colnames():
                    # ANTENNA2 is optional for antenna-based calibration (gaincal, bandpass)
                    # For antenna-based calibration, ANTENNA2 may be -1 or absent
                    # For baseline-based calibration, ANTENNA2 is required
                    ant2_values = tb.getcol("ANTENNA2")
                    # Filter out -1 values (indicates antenna-based calibration)
                    valid_ant2 = ant2_values[ant2_values >= 0]
                    cal_antennas.update(valid_ant2)

                # Check for critical mismatches
                if len(cal_antennas) == 0:
                    # No antennas in caltable - this is critical
                    raise ValueError(f"Calibration table has no antenna solutions: {caltable_path}")

                missing_antennas = ms_antennas - cal_antennas
                if missing_antennas:
                    if len(missing_antennas) == len(ms_antennas):
                        # All MS antennas missing - this is critical
                        raise ValueError(
                            f"Calibration table has no solutions for any MS antennas. "
                            f"MS antennas: {sorted(ms_antennas)}, "
                            f"Cal table antennas: {sorted(cal_antennas)}"
                        )
                    else:
                        # Partial coverage - warn but allow (CASA will flag missing data)
                        warnings.append(
                            f"MS has {len(missing_antennas)} antennas not in calibration table: "
                            f"{sorted(missing_antennas)[:10]}"
                            + ("..." if len(missing_antennas) > 10 else "")
                            + " (CASA will flag data for these antennas)"
                        )

                # Check reference antenna has solutions
                if refant is not None:
                    # Handle both string and int refant values
                    # Also handle comma-separated lists ("104,105,106") by taking first
                    if isinstance(refant, str):
                        refant_str = refant.split(",")[0].strip()
                        refant_int = int(refant_str)
                    else:
                        refant_int = refant
                    if refant_int not in cal_antennas:
                        # Try to select an outrigger antenna as fallback
                        # Outriggers are preferred for reference antenna due to better
                        # phase stability and longer baselines
                        outrigger_refant = select_outrigger_refant(
                            list(cal_antennas), preferred_refant=refant_int
                        )

                        if outrigger_refant is not None:
                            error_msg = (
                                f"Reference antenna {refant_int} has no solutions in calibration table: "
                                f"{caltable_path}. Available antennas: {sorted(cal_antennas)}. "
                                f"Suggested outrigger reference antenna: {outrigger_refant}. "
                                f"Outriggers are preferred for reference antenna due to better "
                                f"phase stability. Available outriggers: {get_outrigger_antennas(list(cal_antennas))}"
                            )
                        else:
                            # No outriggers available, suggest first available antenna
                            suggested_refant = sorted(cal_antennas)[0]
                            error_msg = (
                                f"Reference antenna {refant_int} has no solutions in calibration table: "
                                f"{caltable_path}. Available antennas: {sorted(cal_antennas)}. "
                                f"Consider using refant={suggested_refant} instead, "
                                f"or check why antenna {refant_int} has no solutions "
                                f"(it may be flagged or not present in the calibration data)."
                            )

                        raise ValueError(error_msg)

            # Check frequency/SPW compatibility
            if check_frequencies or check_spw:
                if "SPECTRAL_WINDOW_ID" in tb.colnames():
                    cal_spw_ids = set(tb.getcol("SPECTRAL_WINDOW_ID"))

                    if check_spw:
                        missing_spws = ms_spw_ids - cal_spw_ids
                        if missing_spws:
                            if len(cal_spw_ids) == 0:
                                # No SPWs in caltable - this is critical
                                raise ValueError(
                                    f"Calibration table has no SPW solutions: {caltable_path}"
                                )
                            elif len(missing_spws) == len(ms_spw_ids):
                                # All MS SPWs missing - this is critical
                                raise ValueError(
                                    f"Calibration table has no solutions for any MS SPWs. "
                                    f"MS SPWs: {sorted(ms_spw_ids)}, "
                                    f"Cal table SPWs: {sorted(cal_spw_ids)}"
                                )
                            else:
                                # Partial coverage - warn but allow
                                warnings.append(
                                    f"MS has {len(missing_spws)} SPWs not in calibration table: "
                                    f"{sorted(missing_spws)}"
                                )

                    # Check frequency overlap (if we have frequency info)
                    # NOTE: Frequency checking is incomplete - CASA caltables don't always
                    # store frequencies directly. This would require matching SPW IDs and
                    # checking REF_FREQUENCY from SPW tables, which is complex.
                    if check_frequencies and ms_freq_min is not None and ms_freq_max is not None:
                        warnings.append(
                            "Frequency compatibility check not fully implemented. "
                            "SPW compatibility check should be sufficient."
                        )

    except Exception as e:
        raise ValueError(f"Failed to read calibration table: {caltable_path}. Error: {e}") from e

    return warnings


def validate_caltables_for_use(
    caltable_paths: list[str],
    ms_path: str,
    *,
    require_all: bool = True,
    check_compatibility: bool = True,
    refant: int | str | None = None,
) -> None:
    """Validate multiple calibration tables before use.

    This is a convenience function that validates existence and optionally
    compatibility for a list of calibration tables.

    Parameters
    ----------
    caltable_paths :
        List of calibration table paths (may include None)
    ms_path :
        Path to Measurement Set
    require_all :
        If True, all tables must exist. If False, None entries are skipped.
    check_compatibility :
        Whether to check compatibility with MS
    refant :
        Optional reference antenna ID to verify has solutions in all tables
    caltable_paths: List[str] :

    Raises
    ------
    FileNotFoundError
        If required tables don't exist
    ValueError
        If tables are invalid or incompatible

    """
    valid_tables = [ct for ct in caltable_paths if ct is not None]

    if not valid_tables:
        if require_all:
            raise ValueError("No calibration tables provided")
        return

    # Validate existence
    for ct in valid_tables:
        validate_caltable_exists(ct)

    # Validate compatibility
    if check_compatibility:
        all_warnings = []
        for ct in valid_tables:
            warnings = validate_caltable_compatibility(ct, ms_path, refant=refant)
            all_warnings.extend(warnings)

        if all_warnings:
            # Log warnings but don't fail (non-critical issues)
            logger.warning(
                f"Calibration table compatibility warnings ({len(all_warnings)}): "
                + "; ".join(all_warnings[:5])
                + ("..." if len(all_warnings) > 5 else "")
            )
