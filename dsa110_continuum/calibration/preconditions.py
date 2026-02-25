"""
Pre-calibration validation gate for bandpass calibration.

This module provides a comprehensive validation gate that runs BEFORE bandpass
calibration starts. It prevents wasting compute on data that will inevitably
fail due to setup issues like non-coherent phasing, missing MODEL_DATA, or
UVW geometry errors.

The validation gate checks:
1. Coherent phasing: Field phase centers should be aligned to calibrator
2. MODEL_DATA: Must exist and have non-zero values
3. Pre-bandpass phase: Table should be provided (critical for SNR)
4. Antenna inventory: Sufficient antennas with cross-correlation data
5. Initial flagging: Not too much data flagged before we start
6. UVW geometry: Values within physical baseline limits

Usage:
    from dsa110_contimg.core.calibration.preconditions import (
        validate_bandpass_preconditions,
        ValidationGateResult,
    )

    result = validate_bandpass_preconditions(
        ms_path="/path/to/ms",
        cal_field="0~23",
        calibrator_name="3C454.3",
        prebandpass_phase_table="/path/to/prebp.G",
    )

    if not result.can_proceed:
        print("Cannot proceed with bandpass calibration:")
        for issue in result.blocking_issues:
            print(f"  CRITICAL: {issue}")
        raise ValueError("Precondition check failed")

    for warning in result.warnings:
        print(f"  WARNING: {warning}")
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PreconditionResult:
    """Result of a single precondition check."""

    name: str
    """Name of the precondition check."""

    passed: bool
    """Whether the check passed."""

    severity: str
    """Severity level: 'critical', 'warning', or 'info'."""

    message: str
    """Human-readable description of the result."""

    details: dict[str, Any] = field(default_factory=dict)
    """Additional details about the check."""

    def __str__(self) -> str:
        status = "✅" if self.passed else ("❌" if self.severity == "critical" else "⚠️")
        return f"{status} {self.name}: {self.message}"


@dataclass
class ValidationGateResult:
    """Result of pre-calibration validation gate."""

    can_proceed: bool
    """Whether calibration can proceed (no critical failures)."""

    checks: list[PreconditionResult] = field(default_factory=list)
    """List of all precondition check results."""

    blocking_issues: list[str] = field(default_factory=list)
    """List of critical issues that block calibration."""

    warnings: list[str] = field(default_factory=list)
    """List of warnings that don't block calibration."""

    def __str__(self) -> str:
        status = "✅ PASS" if self.can_proceed else "❌ BLOCKED"
        lines = [
            "=" * 70,
            "PRE-CALIBRATION VALIDATION GATE",
            "=" * 70,
            f"Status: {status}",
            "",
        ]

        if self.blocking_issues:
            lines.append("CRITICAL ISSUES (must fix before proceeding):")
            for issue in self.blocking_issues:
                lines.append(f"  ❌ {issue}")
            lines.append("")

        if self.warnings:
            lines.append("WARNINGS (calibration may have reduced quality):")
            for warning in self.warnings:
                lines.append(f"  ⚠️  {warning}")
            lines.append("")

        lines.append("Check Details:")
        for check in self.checks:
            lines.append(f"  {check}")

        lines.append("=" * 70)
        return "\n".join(lines)


def _check_calibrator_transit(
    ms_path: str,
    calibrator_name: str | None = None,
    max_offset_deg: float = 5.0,
) -> PreconditionResult:
    """Check that the MS phase center matches a calibrator that was transiting.

    DSA-110 is a meridian transit instrument. The phase center of a calibrator MS
    should be within the observation's LST range, indicating the calibrator was
    actually in the primary beam during the observation.

    Parameters
    ----------
    ms_path : str
        Path to Measurement Set.
    calibrator_name : str or None
        Expected calibrator name (for error messages). Default None.
    max_offset_deg : float
        Maximum allowed offset between phase center RA and meridian RA. Default 5.0°.

    Returns
    -------
    PreconditionResult
        Result of the transit check.
    """
    from astropy.coordinates import EarthLocation
    from astropy.time import Time
    import astropy.units as u
    from dsa110_contimg.common.utils.casa_init import ensure_casa_path

    ensure_casa_path()

    try:
        import casacore.tables as ct

        # DSA-110 location (OVRO)
        dsa110_loc = EarthLocation(lat=37.2339 * u.deg, lon=-118.2825 * u.deg, height=1222 * u.m)

        # Get observation time range
        with ct.table(ms_path, readonly=True, ack=False) as tb:
            times = tb.getcol("TIME")
            t_start_mjd = np.min(times) / 86400.0
            t_end_mjd = np.max(times) / 86400.0

        t_start = Time(t_start_mjd, format="mjd")
        t_end = Time(t_end_mjd, format="mjd")

        # Get phase center RA
        with ct.table(f"{ms_path}::FIELD", readonly=True, ack=False) as tb:
            phase_dir = tb.getcol("PHASE_DIR")
            phase_ra_rad = phase_dir[0, 0, 0]
            phase_ra_deg = float(np.degrees(phase_ra_rad)) % 360.0
            if phase_ra_deg < 0:
                phase_ra_deg += 360.0

        # Calculate LST range
        lst_start = t_start.sidereal_time("mean", longitude=dsa110_loc.lon)
        lst_end = t_end.sidereal_time("mean", longitude=dsa110_loc.lon)
        lst_start_deg = lst_start.deg
        lst_end_deg = lst_end.deg
        mid_lst_deg = ((lst_start_deg + lst_end_deg) / 2) % 360.0

        # Calculate offset
        offset_deg = abs(phase_ra_deg - mid_lst_deg)
        if offset_deg > 180:
            offset_deg = 360 - offset_deg
        offset_hours = offset_deg / 15.0

        details = {
            "phase_ra_deg": phase_ra_deg,
            "lst_start_deg": lst_start_deg,
            "lst_end_deg": lst_end_deg,
            "mid_lst_deg": mid_lst_deg,
            "offset_deg": offset_deg,
            "offset_hours": offset_hours,
        }

        if offset_deg > max_offset_deg:
            cal_info = f" ({calibrator_name})" if calibrator_name else ""
            return PreconditionResult(
                name="Calibrator Transit",
                passed=False,
                severity="critical",
                message=(
                    f"Phase center{cal_info} RA={phase_ra_deg:.1f}° is {offset_deg:.1f}° "
                    f"({offset_hours:.1f} hours) from observation meridian ({mid_lst_deg:.1f}°). "
                    f"This calibrator was NOT transiting during the observation. "
                    f"Use data taken when the calibrator was transiting."
                ),
                details=details,
            )

        return PreconditionResult(
            name="Calibrator Transit",
            passed=True,
            severity="info",
            message=f"Phase center RA={phase_ra_deg:.1f}° within {offset_deg:.1f}° of meridian (OK)",
            details=details,
        )

    except Exception as e:
        return PreconditionResult(
            name="Calibrator Transit",
            passed=False,
            severity="critical",
            message=f"Failed to check transit: {e}",
            details={"error": str(e)},
        )


def _check_coherent_phasing(
    ms_path: str,
    cal_field: str,
    max_ra_scatter_arcsec: float = 60.0,
) -> PreconditionResult:
    """Check that all fields are coherently phased to the same position.

    For bandpass calibration with combine_fields=True, all fields must be
    phased to the calibrator position. RA scatter > 60 arcsec indicates
    non-coherent phasing which will cause destructive interference.

    Parameters
    ----------
    ms_path : str
        Path to Measurement Set.
    cal_field : str
        Field selection (e.g., "0~23").
    max_ra_scatter_arcsec : float
        Maximum allowed RA scatter in arcseconds.

    Returns
    -------
    PreconditionResult
        Result of the coherent phasing check.
    """
    from dsa110_contimg.common.utils.casa_init import ensure_casa_path

    ensure_casa_path()

    import casacore.tables as ct

    try:
        # Parse field selection
        if "~" in cal_field:
            start, end = map(int, cal_field.split("~"))
            field_indices = list(range(start, end + 1))
        elif cal_field.isdigit():
            field_indices = [int(cal_field)]
        else:
            field_indices = None  # All fields

        with ct.table(f"{ms_path}::FIELD", readonly=True, ack=False) as tb:
            phase_dir = tb.getcol("PHASE_DIR")  # Shape: (nfields, 1, 2)
            n_fields = tb.nrows()

            if field_indices is None:
                field_indices = list(range(n_fields))

            # Extract RA/Dec for selected fields
            ra_rad = phase_dir[field_indices, 0, 0]
            dec_rad = phase_dir[field_indices, 0, 1]

        # Compute RA scatter using circular statistics (handles wrap-around)
        ra_complex = np.exp(1j * ra_rad)
        mean_ra_complex = np.mean(ra_complex)
        ra_scatter_rad = np.std(np.angle(ra_complex / mean_ra_complex))
        ra_scatter_arcsec = np.degrees(ra_scatter_rad) * 3600

        # Compute Dec scatter (simpler, no wrap-around)
        dec_scatter_arcsec = np.std(dec_rad) * 3600 * 180 / np.pi

        details = {
            "n_fields": len(field_indices),
            "ra_scatter_arcsec": float(ra_scatter_arcsec),
            "dec_scatter_arcsec": float(dec_scatter_arcsec),
            "mean_ra_deg": float(np.degrees(np.angle(mean_ra_complex))),
            "mean_dec_deg": float(np.mean(np.degrees(dec_rad))),
        }

        if ra_scatter_arcsec > max_ra_scatter_arcsec:
            return PreconditionResult(
                name="Coherent Phasing",
                passed=False,
                severity="critical",
                message=(
                    f'RA scatter {ra_scatter_arcsec:.1f}" > {max_ra_scatter_arcsec:.0f}" limit. '
                    f"Fields are NOT coherently phased to calibrator. "
                    f"Run phaseshift_ms() first."
                ),
                details=details,
            )

        return PreconditionResult(
            name="Coherent Phasing",
            passed=True,
            severity="info",
            message=f'RA scatter {ra_scatter_arcsec:.1f}" < {max_ra_scatter_arcsec:.0f}" (OK)',
            details=details,
        )

    except Exception as e:
        return PreconditionResult(
            name="Coherent Phasing",
            passed=False,
            severity="critical",
            message=f"Failed to check phasing: {e}",
            details={"error": str(e)},
        )


def _check_model_data(
    ms_path: str,
    cal_field: str,
    min_amplitude: float = 1e-6,
    sample_size: int = 2000,
) -> PreconditionResult:
    """Check that MODEL_DATA column exists and has non-zero values.

    Bandpass calibration divides DATA by MODEL_DATA. If MODEL_DATA is zeros
    or doesn't exist, all solutions will be flagged.

    Parameters
    ----------
    ms_path : str
        Path to Measurement Set.
    cal_field : str
        Field selection.
    min_amplitude : float
        Minimum acceptable MODEL_DATA amplitude.
    sample_size : int
        Number of rows to sample for validation.

    Returns
    -------
    PreconditionResult
        Result of the MODEL_DATA check.
    """
    from dsa110_contimg.common.utils.casa_init import ensure_casa_path

    ensure_casa_path()

    import casacore.tables as ct

    try:
        with ct.table(ms_path, readonly=True, ack=False) as tb:
            cols = tb.colnames()

            if "MODEL_DATA" not in cols:
                return PreconditionResult(
                    name="MODEL_DATA",
                    passed=False,
                    severity="critical",
                    message=(
                        "MODEL_DATA column does not exist. Run populate_model_from_catalog() first."
                    ),
                    details={"columns": cols},
                )

            # Sample MODEL_DATA to check for non-zero values
            n_rows = tb.nrows()
            actual_sample = min(sample_size, n_rows)

            model_sample = np.array([tb.getcell("MODEL_DATA", i) for i in range(actual_sample)])

            max_amp = np.max(np.abs(model_sample))
            mean_amp = np.mean(np.abs(model_sample))
            nonzero_frac = np.mean(np.abs(model_sample) > min_amplitude)

        details = {
            "max_amplitude": float(max_amp),
            "mean_amplitude": float(mean_amp),
            "nonzero_fraction": float(nonzero_frac),
            "sample_size": actual_sample,
        }

        if max_amp < min_amplitude:
            return PreconditionResult(
                name="MODEL_DATA",
                passed=False,
                severity="critical",
                message=(
                    f"MODEL_DATA is all zeros (max amp = {max_amp:.2e}). "
                    f"Catalog lookup likely failed or wrong calibrator specified."
                ),
                details=details,
            )

        if nonzero_frac < 0.5:
            return PreconditionResult(
                name="MODEL_DATA",
                passed=False,
                severity="warning",
                message=(
                    f"Only {nonzero_frac * 100:.0f}% of MODEL_DATA has non-zero values. "
                    f"Check field selection and model population."
                ),
                details=details,
            )

        return PreconditionResult(
            name="MODEL_DATA",
            passed=True,
            severity="info",
            message=f"MODEL_DATA populated (max amp = {max_amp:.2f})",
            details=details,
        )

    except Exception as e:
        return PreconditionResult(
            name="MODEL_DATA",
            passed=False,
            severity="critical",
            message=f"Failed to check MODEL_DATA: {e}",
            details={"error": str(e)},
        )


def _check_prebandpass_phase(
    prebandpass_phase_table: str | None,
) -> PreconditionResult:
    """Check that pre-bandpass phase table is provided and exists.

    Pre-bandpass phase calibration is CRITICAL for bandpass calibration.
    Without it, phases will drift and decorrelate, causing 90%+ flagged
    solutions.

    Parameters
    ----------
    prebandpass_phase_table : str or None
        Path to pre-bandpass phase calibration table.

    Returns
    -------
    PreconditionResult
        Result of the pre-bandpass phase check.
    """
    if prebandpass_phase_table is None:
        return PreconditionResult(
            name="Pre-Bandpass Phase",
            passed=False,
            severity="critical",
            message=(
                "No pre-bandpass phase table provided. This is REQUIRED. "
                "Without it, phases decorrelate causing 90%+ flagged solutions. "
                "Run solve_prebandpass_phase() first."
            ),
            details={},
        )

    if not os.path.exists(prebandpass_phase_table):
        return PreconditionResult(
            name="Pre-Bandpass Phase",
            passed=False,
            severity="critical",
            message=f"Pre-bandpass phase table not found: {prebandpass_phase_table}",
            details={"path": prebandpass_phase_table},
        )

    # Verify it's a valid calibration table
    from dsa110_contimg.common.utils.casa_init import ensure_casa_path

    ensure_casa_path()

    import casacore.tables as ct

    try:
        with ct.table(prebandpass_phase_table, readonly=True, ack=False) as tb:
            cols = tb.colnames()
            n_rows = tb.nrows()

            # Check for expected columns
            if "CPARAM" not in cols and "FPARAM" not in cols:
                return PreconditionResult(
                    name="Pre-Bandpass Phase",
                    passed=False,
                    severity="critical",
                    message=(
                        f"Invalid calibration table: {prebandpass_phase_table}. "
                        f"Missing CPARAM or FPARAM column."
                    ),
                    details={"columns": cols},
                )

            # Check flag fraction
            if "FLAG" in cols:
                flags = tb.getcol("FLAG")
                flag_frac = np.mean(flags)
            else:
                flag_frac = 0.0

        details = {
            "path": prebandpass_phase_table,
            "n_rows": n_rows,
            "flag_fraction": float(flag_frac),
        }

        if flag_frac > 0.5:
            return PreconditionResult(
                name="Pre-Bandpass Phase",
                passed=False,
                severity="warning",
                message=(
                    f"Pre-bandpass phase table has {flag_frac * 100:.0f}% flagged solutions. "
                    f"This may degrade bandpass quality."
                ),
                details=details,
            )

        return PreconditionResult(
            name="Pre-Bandpass Phase",
            passed=True,
            severity="info",
            message=f"Pre-bandpass phase table valid ({n_rows} rows, {flag_frac * 100:.0f}% flagged)",
            details=details,
        )

    except Exception as e:
        return PreconditionResult(
            name="Pre-Bandpass Phase",
            passed=False,
            severity="critical",
            message=f"Failed to validate pre-bandpass table: {e}",
            details={"error": str(e)},
        )


def _check_antenna_data(
    ms_path: str,
    min_antennas: int = 50,
) -> PreconditionResult:
    """Check that sufficient antennas have cross-correlation data.

    Antennas without cross-correlation data will produce only flagged
    solutions and inflate flagging statistics.

    Parameters
    ----------
    ms_path : str
        Path to Measurement Set.
    min_antennas : int
        Minimum required antennas with data.

    Returns
    -------
    PreconditionResult
        Result of the antenna inventory check.
    """
    from dsa110_contimg.common.utils.casa_init import ensure_casa_path

    ensure_casa_path()

    import casacore.tables as ct

    try:
        with ct.table(ms_path, readonly=True, ack=False) as tb:
            ant1 = tb.getcol("ANTENNA1")
            ant2 = tb.getcol("ANTENNA2")

        with ct.table(f"{ms_path}::ANTENNA", readonly=True, ack=False) as tb:
            n_ant_table = tb.nrows()
            # ant_names available if needed for detailed reporting
            _ = tb.getcol("NAME")  # Read to verify table access

        # Find antennas with cross-correlation data
        cross_mask = ant1 != ant2
        cross_ants = set(int(a) for a in ant1[cross_mask]) | set(int(a) for a in ant2[cross_mask])
        no_data_ants = sorted(set(range(n_ant_table)) - cross_ants)

        n_with_data = len(cross_ants)
        n_no_data = len(no_data_ants)

        details = {
            "n_antennas_in_table": n_ant_table,
            "n_with_cross_data": n_with_data,
            "n_without_data": n_no_data,
            "no_data_antenna_ids": no_data_ants[:20],  # First 20
        }

        if n_with_data < min_antennas:
            return PreconditionResult(
                name="Antenna Inventory",
                passed=False,
                severity="critical",
                message=(
                    f"Only {n_with_data} antennas with data (need {min_antennas}). "
                    f"{n_no_data} antennas have no cross-correlation data."
                ),
                details=details,
            )

        if n_no_data > 0:
            return PreconditionResult(
                name="Antenna Inventory",
                passed=True,
                severity="info",
                message=(
                    f"{n_with_data} antennas with data, "
                    f"{n_no_data} without (will be excluded from solves)"
                ),
                details=details,
            )

        return PreconditionResult(
            name="Antenna Inventory",
            passed=True,
            severity="info",
            message=f"All {n_with_data} antennas have cross-correlation data",
            details=details,
        )

    except Exception as e:
        return PreconditionResult(
            name="Antenna Inventory",
            passed=False,
            severity="warning",
            message=f"Failed to check antenna inventory: {e}",
            details={"error": str(e)},
        )


def _check_initial_flagging(
    ms_path: str,
    cal_field: str,
    max_flag_fraction: float = 0.50,
    sample_size: int = 10000,
) -> PreconditionResult:
    """Check that not too much data is already flagged.

    If >50% of data is already flagged before calibration, there may not
    be enough good data for reliable solutions.

    Parameters
    ----------
    ms_path : str
        Path to Measurement Set.
    cal_field : str
        Field selection.
    max_flag_fraction : float
        Maximum allowed pre-existing flag fraction.
    sample_size : int
        Number of rows to sample.

    Returns
    -------
    PreconditionResult
        Result of the initial flagging check.
    """
    from dsa110_contimg.common.utils.casa_init import ensure_casa_path

    ensure_casa_path()

    import casacore.tables as ct

    try:
        with ct.table(ms_path, readonly=True, ack=False) as tb:
            n_rows = tb.nrows()
            actual_sample = min(sample_size, n_rows)

            # Sample flags
            flags = np.array([tb.getcell("FLAG", i) for i in range(actual_sample)])

            flag_frac = np.mean(flags)

        details = {
            "flag_fraction": float(flag_frac),
            "sample_size": actual_sample,
            "total_rows": n_rows,
        }

        if flag_frac > max_flag_fraction:
            return PreconditionResult(
                name="Initial Flagging",
                passed=False,
                severity="warning",
                message=(
                    f"{flag_frac * 100:.0f}% of data already flagged "
                    f"(threshold: {max_flag_fraction * 100:.0f}%). "
                    f"Calibration quality may be poor."
                ),
                details=details,
            )

        return PreconditionResult(
            name="Initial Flagging",
            passed=True,
            severity="info",
            message=f"{flag_frac * 100:.0f}% of data flagged (acceptable)",
            details=details,
        )

    except Exception as e:
        return PreconditionResult(
            name="Initial Flagging",
            passed=False,
            severity="warning",
            message=f"Failed to check flagging: {e}",
            details={"error": str(e)},
        )


def _check_uvw_geometry(
    ms_path: str,
    max_baseline_m: float = 2707.0,
    tolerance_factor: float = 1.1,
) -> PreconditionResult:
    """Check that UVW values are within physical baseline limits.

    Parameters
    ----------
    ms_path : str
        Path to Measurement Set.
    max_baseline_m : float
        Maximum physical baseline in meters.
    tolerance_factor : float
        Tolerance factor for the limit.

    Returns
    -------
    PreconditionResult
        Result of the UVW geometry check.
    """
    try:
        from dsa110_contimg.core.qa.uvw_validation import validate_uvw_geometry

        result = validate_uvw_geometry(
            ms_path,
            max_baseline_m=max_baseline_m,
            tolerance_factor=tolerance_factor,
            sample_size=10000,
        )

        details = {
            "max_uvw_m": result.max_uvw_distance_m,
            "max_baseline_m": max_baseline_m,
            "n_violations": result.n_violations,
            "violation_fraction": result.violation_fraction,
        }

        if not result.is_valid:
            return PreconditionResult(
                name="UVW Geometry",
                passed=False,
                severity="critical",
                message=(
                    f"UVW values exceed physical limits "
                    f"(max: {result.max_uvw_distance_m:.0f} m, "
                    f"limit: {max_baseline_m:.0f} m). "
                    f"Check phase shifting operation."
                ),
                details=details,
            )

        return PreconditionResult(
            name="UVW Geometry",
            passed=True,
            severity="info",
            message=(
                f"UVW values within limits "
                f"(max: {result.max_uvw_distance_m:.0f} m, "
                f"limit: {max_baseline_m * tolerance_factor:.0f} m)"
            ),
            details=details,
        )

    except ImportError:
        return PreconditionResult(
            name="UVW Geometry",
            passed=True,
            severity="info",
            message="UVW validation module not available, skipping check",
            details={},
        )
    except Exception as e:
        return PreconditionResult(
            name="UVW Geometry",
            passed=False,
            severity="warning",
            message=f"Failed to check UVW geometry: {e}",
            details={"error": str(e)},
        )


def validate_bandpass_preconditions(
    ms_path: str,
    cal_field: str,
    calibrator_name: str | None = None,
    prebandpass_phase_table: str | None = None,
    require_prebandpass_phase: bool = True,
    min_antennas: int = 50,
    max_initial_flag_fraction: float = 0.50,
    max_ra_scatter_arcsec: float = 60.0,
) -> ValidationGateResult:
    """Validate all preconditions before bandpass calibration.

    This is the main entry point for the pre-calibration validation gate.
    It runs all precondition checks and returns a comprehensive result
    indicating whether calibration can proceed.

    Parameters
    ----------
    ms_path : str
        Path to Measurement Set.
    cal_field : str
        Field selection (e.g., "0~23").
    calibrator_name : str or None, optional
        Calibrator name (for logging). Default None.
    prebandpass_phase_table : str or None, optional
        Path to pre-bandpass phase calibration table. Default None.
    require_prebandpass_phase : bool, optional
        If True, missing prebandpass_phase_table is critical. Default True.
    min_antennas : int, optional
        Minimum required antennas with data. Default 50.
    max_initial_flag_fraction : float, optional
        Maximum allowed initial flag fraction. Default 0.50.
    max_ra_scatter_arcsec : float, optional
        Maximum allowed RA scatter for coherent phasing. Default 60.0.

    Returns
    -------
    ValidationGateResult
        Comprehensive validation result.

    Examples
    --------
    >>> result = validate_bandpass_preconditions(
    ...     ms_path="/path/to/ms",
    ...     cal_field="0~23",
    ...     calibrator_name="3C454.3",
    ...     prebandpass_phase_table="/path/to/prebp.G",
    ... )
    >>> if not result.can_proceed:
    ...     raise ValueError(f"Precondition check failed: {result.blocking_issues}")
    """
    logger.info("\n" + "=" * 70)
    logger.info("PRE-CALIBRATION VALIDATION GATE")
    logger.info("=" * 70)
    logger.info(f"MS: {ms_path}")
    logger.info(f"Field: {cal_field}")
    if calibrator_name:
        logger.info(f"Calibrator: {calibrator_name}")
    logger.info("")

    checks = []

    # Check 0: Calibrator transit validation (most fundamental check)
    # This must pass before any other checks make sense - if the calibrator
    # wasn't transiting, all other checks are meaningless.
    logger.info("Checking calibrator transit...")
    checks.append(_check_calibrator_transit(ms_path, calibrator_name))

    # Check 1: Coherent phasing
    logger.info("Checking coherent phasing...")
    checks.append(_check_coherent_phasing(ms_path, cal_field, max_ra_scatter_arcsec))

    # Check 2: MODEL_DATA validation
    logger.info("Checking MODEL_DATA...")
    checks.append(_check_model_data(ms_path, cal_field))

    # Check 3: Pre-bandpass phase table
    logger.info("Checking pre-bandpass phase table...")
    prebp_check = _check_prebandpass_phase(prebandpass_phase_table)
    if not require_prebandpass_phase and prebp_check.severity == "critical":
        # Downgrade to warning if not required
        prebp_check = PreconditionResult(
            name=prebp_check.name,
            passed=False,
            severity="warning",
            message=prebp_check.message.replace("REQUIRED", "recommended"),
            details=prebp_check.details,
        )
    checks.append(prebp_check)

    # Check 4: Antenna inventory
    logger.info("Checking antenna inventory...")
    checks.append(_check_antenna_data(ms_path, min_antennas))

    # Check 5: Initial flag fraction
    logger.info("Checking initial flagging...")
    checks.append(_check_initial_flagging(ms_path, cal_field, max_initial_flag_fraction))

    # Check 6: UVW geometry
    logger.info("Checking UVW geometry...")
    checks.append(_check_uvw_geometry(ms_path))

    # Compile results
    blocking = [c for c in checks if not c.passed and c.severity == "critical"]
    warnings = [c for c in checks if not c.passed and c.severity == "warning"]

    result = ValidationGateResult(
        can_proceed=len(blocking) == 0,
        checks=checks,
        blocking_issues=[c.message for c in blocking],
        warnings=[c.message for c in warnings],
    )

    # Log result
    logger.info("")
    logger.info(str(result))

    return result


def require_valid_preconditions(
    ms_path: str,
    cal_field: str,
    calibrator_name: str | None = None,
    prebandpass_phase_table: str | None = None,
    **kwargs: Any,
) -> ValidationGateResult:
    """Validate preconditions and raise if any critical checks fail.

    This is a convenience wrapper that raises ValueError if any
    critical preconditions are not met.

    Parameters
    ----------
    ms_path : str
        Path to Measurement Set.
    cal_field : str
        Field selection.
    calibrator_name : str or None, optional
        Calibrator name.
    prebandpass_phase_table : str or None, optional
        Path to pre-bandpass phase table.
    **kwargs
        Additional arguments passed to validate_bandpass_preconditions.

    Returns
    -------
    ValidationGateResult
        Validation result (only if all critical checks pass).

    Raises
    ------
    ValueError
        If any critical precondition check fails.
    """
    result = validate_bandpass_preconditions(
        ms_path=ms_path,
        cal_field=cal_field,
        calibrator_name=calibrator_name,
        prebandpass_phase_table=prebandpass_phase_table,
        **kwargs,
    )

    if not result.can_proceed:
        issues = "\n  - ".join(result.blocking_issues)
        raise ValueError(
            f"Bandpass calibration preconditions not met:\n  - {issues}\n\n"
            f"Fix these issues before attempting bandpass calibration."
        )

    return result
