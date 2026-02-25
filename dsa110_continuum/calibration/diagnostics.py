"""Comprehensive calibration diagnostics and comparison utilities."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

# Ensure CASAPATH is set before importing CASA modules
from dsa110_contimg.common.utils.casa_init import ensure_casa_path

ensure_casa_path()

import casacore.tables as casatables
import numpy as np

table = casatables.table  # noqa: N816

from dsa110_contimg.core.qa.calibration_quality import (
    CalibrationQualityMetrics,
    check_corrected_data_quality,
    validate_caltable_quality,
)

logger = logging.getLogger(__name__)


@dataclass
class CalibrationDiagnostics:
    """Comprehensive calibration diagnostics report."""

    ms_path: str
    field: str
    refant: str

    # MS quality
    ms_valid: bool
    ms_has_fields: bool
    ms_n_rows: int
    ms_n_antennas: int
    ms_unflagged_fraction: float

    # Flagging statistics
    flagging_unflagged_fraction: float
    flagging_issues: list[str]

    # Calibration table quality (if tables exist)
    caltables: dict[str, CalibrationQualityMetrics]

    # CORRECTED_DATA quality (if applied)
    corrected_data_quality: dict | None
    corrected_data_issues: list[str]

    # Overall assessment
    ready_for_calibration: bool
    issues: list[str]
    warnings: list[str]

    def to_dict(self) -> dict:
        """Convert diagnostics to dictionary."""
        return {
            "ms_path": self.ms_path,
            "field": self.field,
            "refant": self.refant,
            "ms_quality": {
                "valid": self.ms_valid,
                "has_fields": self.ms_has_fields,
                "n_rows": self.ms_n_rows,
                "n_antennas": self.ms_n_antennas,
                "unflagged_fraction": self.ms_unflagged_fraction,
            },
            "flagging": {
                "unflagged_fraction": self.flagging_unflagged_fraction,
                "issues": self.flagging_issues,
            },
            "caltables": {path: metrics.to_dict() for path, metrics in self.caltables.items()},
            "corrected_data": self.corrected_data_quality,
            "corrected_data_issues": self.corrected_data_issues,
            "assessment": {
                "ready_for_calibration": self.ready_for_calibration,
                "issues": self.issues,
                "warnings": self.warnings,
            },
        }

    def print_report(self) -> None:
        """Print human-readable diagnostics report."""
        print("\n" + "=" * 70)
        print("Calibration Diagnostics Report")
        print("=" * 70)
        print(f"MS: {self.ms_path}")
        print(f"Field: {self.field}")
        print(f"Reference Antenna: {self.refant}")
        print()

        print("MS Quality:")
        print(f"  Valid: {self.ms_valid}")
        print(f"  Has Fields: {self.ms_has_fields}")
        print(f"  Rows: {self.ms_n_rows:,}")
        print(f"  Antennas: {self.ms_n_antennas}")
        print(f"  Unflagged Fraction: {self.ms_unflagged_fraction * 100:.1f}%")
        print()

        print("Flagging:")
        print(f"  Unflagged Fraction: {self.flagging_unflagged_fraction * 100:.1f}%")
        if self.flagging_issues:
            print(f"  Issues: {', '.join(self.flagging_issues)}")
        print()

        if self.caltables:
            print("Calibration Tables:")
            for path, metrics in self.caltables.items():
                print(f"  {os.path.basename(path)} ({metrics.cal_type}):")
                print(f"    Solutions: {metrics.n_solutions}")
                print(f"    Flagged: {metrics.fraction_flagged * 100:.1f}%")
                if metrics.has_issues:
                    print(f"    Issues: {', '.join(metrics.issues)}")
                if metrics.has_warnings:
                    print(f"    Warnings: {', '.join(metrics.warnings)}")
            print()

        if self.corrected_data_quality:
            print("CORRECTED_DATA Quality:")
            for key, value in self.corrected_data_quality.items():
                if isinstance(value, float):
                    print(f"  {key}: {value:.6f}")
                else:
                    print(f"  {key}: {value}")
            if self.corrected_data_issues:
                print(f"  Issues: {', '.join(self.corrected_data_issues)}")
            print()

        print("Overall Assessment:")
        print(f"  Ready for Calibration: {self.ready_for_calibration}")
        if self.issues:
            print(f"  Issues: {', '.join(self.issues)}")
        if self.warnings:
            print(f"  Warnings: {', '.join(self.warnings)}")
        print("=" * 70)


def generate_calibration_diagnostics(
    ms_path: str,
    field: str = "",
    refant: str | None = None,
    check_caltables: bool = True,
    check_corrected_data: bool = True,
) -> CalibrationDiagnostics:
    """Generate comprehensive calibration diagnostics report.

    Parameters
    ----------
    ms_path :
        Path to Measurement Set
    field :
        Field selection (default: "" = all fields)
    refant :
        Reference antenna ID (optional)
    check_caltables :
        Whether to check for existing caltables
    check_corrected_data :
        Whether to check CORRECTED_DATA quality

    Returns
    -------
        CalibrationDiagnostics object

    """
    issues = []
    warnings = []

    # MS quality checks
    ms_valid = True
    ms_has_fields = False
    ms_n_rows = 0
    ms_n_antennas = 0
    ms_unflagged_fraction = 0.0

    try:
        with table(ms_path, readonly=True) as tb:
            ms_n_rows = tb.nrows()

            # Check for fields
            with table(f"{ms_path}::FIELD", readonly=True) as field_tb:
                ms_has_fields = field_tb.nrows() > 0

            # Check antennas
            with table(f"{ms_path}::ANTENNA", readonly=True) as ant_tb:
                ms_n_antennas = ant_tb.nrows()

            # Sample unflagged fraction
            if ms_n_rows > 0:
                sample_size = min(10000, ms_n_rows)
                flags_sample = tb.getcol("FLAG", startrow=0, nrow=sample_size)
                ms_unflagged_fraction = float(np.mean(~flags_sample))

            if ms_n_rows == 0:
                issues.append("MS has zero rows")
                ms_valid = False
            if not ms_has_fields:
                issues.append("MS has no fields")
                ms_valid = False
            if ms_unflagged_fraction < 0.1:
                warnings.append(f"Low unflagged data: {ms_unflagged_fraction * 100:.1f}%")
    except Exception as e:
        issues.append(f"Failed to read MS: {e}")
        ms_valid = False

    # Flagging statistics (simulate flagging without actually doing it)
    flagging_unflagged_fraction = ms_unflagged_fraction
    flagging_issues = []

    # Check for existing caltables
    caltables = {}
    if check_caltables:
        ms_base = ms_path.rstrip("/").rstrip(".ms")
        for suffix in [".kcal", ".bpcal", ".gcal"]:
            caltable_path = ms_base + suffix
            if os.path.exists(caltable_path):
                try:
                    metrics = validate_caltable_quality(caltable_path)
                    caltables[caltable_path] = metrics
                    if metrics.has_issues:
                        issues.append(f"Caltable {os.path.basename(caltable_path)} has issues")
                    if metrics.has_warnings:
                        warnings.append(f"Caltable {os.path.basename(caltable_path)} has warnings")
                except Exception as e:
                    warnings.append(f"Failed to validate {caltable_path}: {e}")

    # Check CORRECTED_DATA quality
    corrected_data_quality = None
    corrected_data_issues = []
    if check_corrected_data:
        try:
            passed, metrics, issues_list = check_corrected_data_quality(ms_path)
            corrected_data_quality = metrics
            corrected_data_issues = issues_list
            if not passed:
                issues.extend(issues_list)
        except Exception as e:
            warnings.append(f"Failed to check CORRECTED_DATA: {e}")

    # Overall assessment
    ready_for_calibration = ms_valid and ms_has_fields and ms_n_rows > 0

    return CalibrationDiagnostics(
        ms_path=ms_path,
        field=field,
        refant=refant or "not specified",
        ms_valid=ms_valid,
        ms_has_fields=ms_has_fields,
        ms_n_rows=ms_n_rows,
        ms_n_antennas=ms_n_antennas,
        ms_unflagged_fraction=ms_unflagged_fraction,
        flagging_unflagged_fraction=flagging_unflagged_fraction,
        flagging_issues=flagging_issues,
        caltables=caltables,
        corrected_data_quality=corrected_data_quality,
        corrected_data_issues=corrected_data_issues,
        ready_for_calibration=ready_for_calibration,
        issues=issues,
        warnings=warnings,
    )


@dataclass
class CalibrationComparison:
    """Comparison between two calibration solutions."""

    caltable1_path: str
    caltable2_path: str

    # Table structure comparison
    same_structure: bool
    n_solutions_diff: int
    n_antennas_diff: int

    # Solution comparison
    amplitude_median_diff: float
    amplitude_rms_diff: float
    phase_median_diff: float
    phase_rms_diff: float

    # Agreement metrics
    solutions_agree: bool
    tolerance: float
    agreement_fraction: float

    # Issues
    issues: list[str]
    warnings: list[str]

    def to_dict(self) -> dict:
        """Convert comparison to dictionary."""
        return {
            "caltable1": self.caltable1_path,
            "caltable2": self.caltable2_path,
            "structure": {
                "same": self.same_structure,
                "n_solutions_diff": self.n_solutions_diff,
                "n_antennas_diff": self.n_antennas_diff,
            },
            "solutions": {
                "amplitude_median_diff": self.amplitude_median_diff,
                "amplitude_rms_diff": self.amplitude_rms_diff,
                "phase_median_diff": self.phase_median_diff,
                "phase_rms_diff": self.phase_rms_diff,
            },
            "agreement": {
                "solutions_agree": self.solutions_agree,
                "tolerance": self.tolerance,
                "agreement_fraction": self.agreement_fraction,
            },
            "issues": self.issues,
            "warnings": self.warnings,
        }

    def print_report(self) -> None:
        """Print human-readable comparison report."""
        print("\n" + "=" * 70)
        print("Calibration Comparison Report")
        print("=" * 70)
        print(f"Caltable 1: {self.caltable1_path}")
        print(f"Caltable 2: {self.caltable2_path}")
        print()

        print("Structure Comparison:")
        print(f"  Same Structure: {self.same_structure}")
        print(f"  Solutions Difference: {self.n_solutions_diff}")
        print(f"  Antennas Difference: {self.n_antennas_diff}")
        print()

        print("Solution Comparison:")
        print(f"  Amplitude Median Difference: {self.amplitude_median_diff:.6f}")
        print(f"  Amplitude RMS Difference: {self.amplitude_rms_diff:.6f}")
        print(f"  Phase Median Difference: {self.phase_median_diff:.2f} deg")
        print(f"  Phase RMS Difference: {self.phase_rms_diff:.2f} deg")
        print()

        print("Agreement:")
        print(f"  Solutions Agree: {self.solutions_agree}")
        print(f"  Agreement Fraction: {self.agreement_fraction * 100:.1f}%")
        print(f"  Tolerance: {self.tolerance:.6e}")
        print()

        if self.issues:
            print("Issues:")
            for issue in self.issues:
                print(f"  - {issue}")
            print()

        if self.warnings:
            print("Warnings:")
            for warning in self.warnings:
                print(f"  - {warning}")
            print()

        print("=" * 70)


def compare_caltables(
    caltable1_path: str,
    caltable2_path: str,
    tolerance: float = 1e-6,
) -> CalibrationComparison:
    """Compare two calibration tables for consistency.

    Parameters
    ----------
    caltable1_path :
        Path to first calibration table
    caltable2_path :
        Path to second calibration table
    tolerance :
        Tolerance for solution agreement (default: 1e-6)

    Returns
    -------
        CalibrationComparison object

    """
    issues = []
    warnings = []

    # Validate tables exist
    if not os.path.exists(caltable1_path):
        raise FileNotFoundError(f"Calibration table not found: {caltable1_path}")
    if not os.path.exists(caltable2_path):
        raise FileNotFoundError(f"Calibration table not found: {caltable2_path}")

    # Get table metrics
    try:
        metrics1 = validate_caltable_quality(caltable1_path)
        metrics2 = validate_caltable_quality(caltable2_path)
    except Exception as e:
        raise RuntimeError(f"Failed to validate calibration tables: {e}") from e

    # Structure comparison
    same_structure = (
        metrics1.n_antennas == metrics2.n_antennas and metrics1.n_spws == metrics2.n_spws
    )
    n_solutions_diff = abs(metrics1.n_solutions - metrics2.n_solutions)
    n_antennas_diff = abs(metrics1.n_antennas - metrics2.n_antennas)

    if not same_structure:
        issues.append("Calibration tables have different structure")

    # Solution comparison (for complex gain tables)
    if metrics1.cal_type in ["BP", "G"] and metrics2.cal_type in ["BP", "G"]:
        # Compare amplitudes and phases
        amplitude_median_diff = abs(metrics1.median_amplitude - metrics2.median_amplitude)
        amplitude_rms_diff = abs(metrics1.rms_amplitude - metrics2.rms_amplitude)
        phase_median_diff = abs(metrics1.median_phase_deg - metrics2.median_phase_deg)
        phase_rms_diff = abs(metrics1.rms_phase_deg - metrics2.rms_phase_deg)

        # Compute agreement fraction (simplified - compare key metrics)
        agreement_fraction = 1.0
        if amplitude_median_diff > tolerance * 10:
            agreement_fraction *= 0.5
        if phase_median_diff > 1.0:  # 1 degree tolerance
            agreement_fraction *= 0.5

        solutions_agree = (
            amplitude_median_diff < tolerance * 10
            and phase_median_diff < 1.0
            and amplitude_rms_diff < tolerance * 10
            and phase_rms_diff < 5.0  # 5 degrees RMS tolerance
        )
    else:
        # For K-calibration or mixed types, use simpler comparison
        amplitude_median_diff = 0.0
        amplitude_rms_diff = 0.0
        phase_median_diff = 0.0
        phase_rms_diff = 0.0
        solutions_agree = same_structure
        agreement_fraction = 1.0 if same_structure else 0.0

    if not solutions_agree:
        warnings.append("Calibration solutions differ significantly")

    return CalibrationComparison(
        caltable1_path=caltable1_path,
        caltable2_path=caltable2_path,
        same_structure=same_structure,
        n_solutions_diff=n_solutions_diff,
        n_antennas_diff=n_antennas_diff,
        amplitude_median_diff=amplitude_median_diff,
        amplitude_rms_diff=amplitude_rms_diff,
        phase_median_diff=phase_median_diff,
        phase_rms_diff=phase_rms_diff,
        solutions_agree=solutions_agree,
        tolerance=tolerance,
        agreement_fraction=agreement_fraction,
        issues=issues,
        warnings=warnings,
    )
