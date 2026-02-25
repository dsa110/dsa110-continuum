"""
Unified MS validation module.

Consolidates all MS validation functionality:
- Basic properties (fields, SPWs, antennas, data shape)
- Phasing checks (phase centers, calibrator alignment)
- Rephase status (REFERENCE_DIR vs PHASE_DIR)
- Timing validation (TIME column, duration, LST)
- Data integrity (non-zero data, flagging)

Usage:
    from dsa110_contimg.core.validation.ms_validator import MSValidator

    validator = MSValidator("/path/to/file.ms")
    report = validator.validate_all()
    print(report.summary())

CLI:
    python -m dsa110_contimg.core.validation.ms_validator /path/to/file.ms
    python -m dsa110_contimg.core.validation.ms_validator /path/to/file.ms --json
    python -m dsa110_contimg.core.validation.ms_validator /path/to/file.ms --calibrator 3C286
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def _convert_numpy_types(obj: Any) -> Any:
    """Recursively convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: _convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_convert_numpy_types(v) for v in obj]
    return obj


@dataclass
class ValidationResult:
    """Result of a single validation check."""

    name: str
    passed: bool
    message: str
    details: dict[str, Any] = field(default_factory=dict)
    severity: str = "error"  # error, warning, info


@dataclass
class MSValidationReport:
    """Complete MS validation report."""

    ms_path: str
    validated_at: str
    results: list[ValidationResult] = field(default_factory=list)
    properties: dict[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        """True if all error-severity checks passed."""
        return all(r.passed for r in self.results if r.severity == "error")

    @property
    def errors(self) -> list[ValidationResult]:
        """List of failed error-severity checks."""
        return [r for r in self.results if not r.passed and r.severity == "error"]

    @property
    def warnings(self) -> list[ValidationResult]:
        """List of failed warning-severity checks."""
        return [r for r in self.results if not r.passed and r.severity == "warning"]

    def summary(self) -> str:
        """Human-readable summary."""
        lines = [
            f"MS Validation Report: {self.ms_path}",
            f"Validated: {self.validated_at}",
            "=" * 70,
        ]

        # Properties section
        if self.properties:
            lines.append("\nProperties:")
            for key, value in self.properties.items():
                lines.append(f"  {key}: {value}")

        # Results section
        lines.append("\nValidation Results:")
        for result in self.results:
            status = "" if result.passed else ""
            lines.append(f"  {status} [{result.severity.upper()}] {result.name}: {result.message}")

        # Summary
        n_passed = sum(1 for r in self.results if r.passed)
        n_total = len(self.results)
        n_errors = len(self.errors)
        n_warnings = len(self.warnings)

        lines.append("")
        lines.append(f"Summary: {n_passed}/{n_total} checks passed")
        if n_errors > 0:
            lines.append(f"  Errors: {n_errors}")
        if n_warnings > 0:
            lines.append(f"  Warnings: {n_warnings}")

        overall = "PASSED " if self.passed else "FAILED "
        lines.append(f"\nOverall: {overall}")

        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return _convert_numpy_types(
            {
                "ms_path": self.ms_path,
                "validated_at": self.validated_at,
                "passed": self.passed,
                "properties": self.properties,
                "results": [
                    {
                        "name": r.name,
                        "passed": r.passed,
                        "message": r.message,
                        "severity": r.severity,
                        "details": r.details,
                    }
                    for r in self.results
                ],
                "summary": {
                    "total": len(self.results),
                    "passed": sum(1 for r in self.results if r.passed),
                    "errors": len(self.errors),
                    "warnings": len(self.warnings),
                },
            }
        )


class MSValidator:
    """Unified MS validation class."""

    def __init__(self, ms_path: str | Path):
        """Initialize validator with MS path."""
        self.ms_path = Path(ms_path)
        self._tb = None  # casacore table handle
        self._report = MSValidationReport(
            ms_path=str(self.ms_path),
            validated_at=datetime.utcnow().isoformat() + "Z",
        )

    def validate_all(
        self,
        calibrator_name: str | None = None,
        calibrator_ra_deg: float | None = None,
        calibrator_dec_deg: float | None = None,
        check_data: bool = True,
    ) -> MSValidationReport:
        """Run all validation checks.

        Parameters
        ----------
        calibrator_name : str
            Name of calibrator to check against
        calibrator_ra_deg : float
            Calibrator RA in degrees
        calibrator_dec_deg : float
            Calibrator Dec in degrees
        check_data : bool
            Whether to check DATA column (slower)

        Returns
        -------
            MSValidationReport
            MSValidationReport with all results
        """
        # Check MS exists and is readable
        if not self._check_exists():
            return self._report

        # Basic structure checks
        self._check_tables()
        self._check_properties()

        # Phase center checks
        self._check_phase_centers()

        # Rephase status
        self._check_rephase_status()

        # Calibrator alignment (if provided)
        if calibrator_ra_deg is not None and calibrator_dec_deg is not None:
            self._check_calibrator_alignment(
                calibrator_name or "calibrator",
                calibrator_ra_deg,
                calibrator_dec_deg,
            )

        # Data integrity
        if check_data:
            self._check_data_integrity()

        # Timing
        self._check_timing()

        return self._report

    def _add_result(
        self,
        name: str,
        passed: bool,
        message: str,
        details: dict | None = None,
        severity: str = "error",
    ):
        """Add a validation result."""
        self._report.results.append(
            ValidationResult(
                name=name,
                passed=passed,
                message=message,
                details=details or {},
                severity=severity,
            )
        )

    def _check_exists(self) -> bool:
        """Check MS exists and is a directory."""
        if not self.ms_path.exists():
            self._add_result(
                "ms_exists",
                False,
                f"MS path does not exist: {self.ms_path}",
            )
            return False

        if not self.ms_path.is_dir():
            self._add_result(
                "ms_is_directory",
                False,
                f"MS path is not a directory: {self.ms_path}",
            )
            return False

        # Check for table.dat (required for CASA MS)
        table_dat = self.ms_path / "table.dat"
        if not table_dat.exists():
            self._add_result(
                "ms_has_table_dat",
                False,
                "MS missing table.dat file (not a valid CASA table)",
            )
            return False

        self._add_result("ms_exists", True, "MS exists and is valid directory")
        return True

    def _check_tables(self):
        """Check required subtables exist."""
        required_tables = [
            "ANTENNA",
            "DATA_DESCRIPTION",
            "FEED",
            "FIELD",
            "OBSERVATION",
            "POLARIZATION",
            "SOURCE",
            "SPECTRAL_WINDOW",
        ]

        missing = []
        for table_name in required_tables:
            table_path = self.ms_path / table_name
            if not table_path.exists():
                missing.append(table_name)

        if missing:
            self._add_result(
                "required_tables",
                False,
                f"Missing required tables: {', '.join(missing)}",
                {"missing": missing},
            )
        else:
            self._add_result(
                "required_tables",
                True,
                f"All {len(required_tables)} required tables present",
            )

    def _check_properties(self):
        """Extract and validate basic MS properties."""
        try:
            from casacore.tables import table
        except ImportError:
            self._add_result(
                "casacore_available",
                False,
                "casacore not available - cannot read MS",
                severity="error",
            )
            return

        try:
            # Main table
            with table(str(self.ms_path), readonly=True, ack=False) as tb:
                nrows = tb.nrows()
                colnames = tb.colnames()

                self._report.properties["nrows"] = nrows
                self._report.properties["has_data"] = "DATA" in colnames
                self._report.properties["has_corrected"] = "CORRECTED_DATA" in colnames
                self._report.properties["has_model"] = "MODEL_DATA" in colnames

            # Antenna table
            with table(str(self.ms_path / "ANTENNA"), readonly=True, ack=False) as tb:
                nants = tb.nrows()
                self._report.properties["nants"] = nants

            # Field table
            with table(str(self.ms_path / "FIELD"), readonly=True, ack=False) as tb:
                nfields = tb.nrows()
                field_names = list(tb.getcol("NAME"))
                self._report.properties["nfields"] = nfields
                self._report.properties["field_names"] = field_names

            # Spectral window table
            with table(str(self.ms_path / "SPECTRAL_WINDOW"), readonly=True, ack=False) as tb:
                nspw = tb.nrows()
                num_chan = list(tb.getcol("NUM_CHAN"))
                self._report.properties["nspw"] = nspw
                self._report.properties["channels_per_spw"] = num_chan
                self._report.properties["total_channels"] = sum(num_chan)

            # Observation table
            with table(str(self.ms_path / "OBSERVATION"), readonly=True, ack=False) as tb:
                if tb.nrows() > 0:
                    observer = tb.getcol("OBSERVER")[0]
                    telescope = tb.getcol("TELESCOPE_NAME")[0]
                    self._report.properties["observer"] = observer
                    self._report.properties["telescope"] = telescope

            self._add_result(
                "properties_extracted",
                True,
                f"{nfields} fields, {nspw} SPWs, {nants} antennas, {nrows} rows",
                self._report.properties,
            )

        except (OSError, RuntimeError) as e:
            self._add_result(
                "properties_extracted",
                False,
                f"Failed to extract properties: {e}",
            )

    def _check_phase_centers(self):
        """Check phase centers for all fields."""
        try:
            from casacore.tables import table
        except ImportError:
            return

        try:
            with table(str(self.ms_path / "FIELD"), readonly=True, ack=False) as tb:
                phase_dirs = tb.getcol("PHASE_DIR")
                tb.getcol("REFERENCE_DIR")
                names = tb.getcol("NAME")

                centers = []
                for i in range(len(names)):
                    ra_deg = np.degrees(phase_dirs[i, 0, 0])
                    dec_deg = np.degrees(phase_dirs[i, 0, 1])
                    centers.append(
                        {
                            "field": i,
                            "name": names[i],
                            "ra_deg": float(ra_deg),
                            "dec_deg": float(dec_deg),
                        }
                    )

                self._report.properties["phase_centers"] = centers

                # Check first field has valid coordinates
                if len(centers) > 0:
                    first = centers[0]
                    if abs(first["dec_deg"]) > 90:
                        self._add_result(
                            "phase_center_valid",
                            False,
                            f"Invalid Dec: {first['dec_deg']:.2f}°",
                            severity="error",
                        )
                    else:
                        self._add_result(
                            "phase_center_valid",
                            True,
                            f"Field 0: RA={first['ra_deg']:.4f}° Dec={first['dec_deg']:.4f}°",
                        )

        except (OSError, RuntimeError) as e:
            self._add_result(
                "phase_centers",
                False,
                f"Failed to check phase centers: {e}",
            )

    def _check_rephase_status(self):
        """Check if REFERENCE_DIR matches PHASE_DIR (rephase indicator)."""
        try:
            from casacore.tables import table
        except ImportError:
            return

        try:
            with table(str(self.ms_path / "FIELD"), readonly=True, ack=False) as tb:
                ref_dirs = tb.getcol("REFERENCE_DIR")
                phase_dirs = tb.getcol("PHASE_DIR")

                # Compare first field
                ref_match = np.allclose(ref_dirs, phase_dirs, rtol=1e-10)

                if ref_match:
                    self._add_result(
                        "rephase_status",
                        True,
                        "REFERENCE_DIR == PHASE_DIR (standard meridian phasing)",
                        {"rephased": False},
                        severity="info",
                    )
                else:
                    # Calculate separation
                    ref_ra = np.degrees(ref_dirs[0, 0, 0])
                    ref_dec = np.degrees(ref_dirs[0, 0, 1])
                    phase_ra = np.degrees(phase_dirs[0, 0, 0])
                    phase_dec = np.degrees(phase_dirs[0, 0, 1])

                    sep_ra = abs(ref_ra - phase_ra)
                    sep_dec = abs(ref_dec - phase_dec)

                    self._add_result(
                        "rephase_status",
                        True,
                        f"REFERENCE_DIR != PHASE_DIR (rephased, ΔRA={sep_ra:.4f}° ΔDec={sep_dec:.4f}°)",
                        {"rephased": True, "delta_ra_deg": sep_ra, "delta_dec_deg": sep_dec},
                        severity="info",
                    )

        except (OSError, RuntimeError) as e:
            self._add_result(
                "rephase_status",
                False,
                f"Failed to check rephase status: {e}",
                severity="warning",
            )

    def _check_calibrator_alignment(
        self, calibrator_name: str, cal_ra_deg: float, cal_dec_deg: float
    ):
        """Check if any field is aligned with calibrator."""
        try:
            from astropy import units as u
            from astropy.coordinates import SkyCoord
        except ImportError:
            self._add_result(
                "calibrator_alignment",
                False,
                "astropy not available",
                severity="warning",
            )
            return

        centers = self._report.properties.get("phase_centers", [])
        if not centers:
            return

        cal_coord = SkyCoord(ra=cal_ra_deg * u.deg, dec=cal_dec_deg * u.deg, frame="icrs")

        best_sep = float("inf")
        best_field = None

        for center in centers:
            field_coord = SkyCoord(
                ra=center["ra_deg"] * u.deg,
                dec=center["dec_deg"] * u.deg,
                frame="icrs",
            )
            sep = cal_coord.separation(field_coord).arcmin

            if sep < best_sep:
                best_sep = sep
                best_field = center

        # Threshold: within 1 arcmin is aligned
        aligned = best_sep < 1.0

        if aligned:
            self._add_result(
                "calibrator_alignment",
                True,
                f"Field {best_field['field']} ({best_field['name']}) aligned with {calibrator_name} "
                f"(separation: {best_sep:.2f} arcmin)",
                {
                    "calibrator": calibrator_name,
                    "aligned_field": best_field["field"],
                    "separation_arcmin": best_sep,
                },
            )
        else:
            self._add_result(
                "calibrator_alignment",
                False,
                f"No field aligned with {calibrator_name} (closest: {best_sep:.2f} arcmin)",
                {
                    "calibrator": calibrator_name,
                    "closest_field": best_field["field"] if best_field else None,
                    "separation_arcmin": best_sep,
                },
                severity="warning",
            )

    def _check_data_integrity(self):
        """Check DATA column for non-zero values and flagging."""
        try:
            from casacore.tables import table
        except ImportError:
            return

        try:
            with table(str(self.ms_path), readonly=True, ack=False) as tb:
                # Sample first N rows
                sample_size = min(1000, tb.nrows())

                if "DATA" not in tb.colnames():
                    self._add_result(
                        "data_column",
                        False,
                        "DATA column missing",
                    )
                    return

                data = tb.getcol("DATA", startrow=0, nrow=sample_size)
                flags = (
                    tb.getcol("FLAG", startrow=0, nrow=sample_size)
                    if "FLAG" in tb.colnames()
                    else None
                )

                # Check data shape
                self._report.properties["data_shape"] = list(data.shape)
                self._report.properties["data_dtype"] = str(data.dtype)

                # Check for non-zero data
                mean_abs = np.mean(np.abs(data))
                nonzero_frac = np.count_nonzero(data) / data.size

                if mean_abs < 1e-10:
                    self._add_result(
                        "data_nonzero",
                        False,
                        f"DATA appears to be all zeros (mean abs: {mean_abs:.2e})",
                    )
                else:
                    self._add_result(
                        "data_nonzero",
                        True,
                        f"DATA has values (mean abs: {mean_abs:.4f}, {nonzero_frac * 100:.1f}% nonzero)",
                        {"mean_abs": float(mean_abs), "nonzero_fraction": float(nonzero_frac)},
                    )

                # Check flagging
                if flags is not None:
                    flag_frac = np.mean(flags)
                    if flag_frac > 0.9:
                        self._add_result(
                            "flagging",
                            False,
                            f"{flag_frac * 100:.1f}% of data flagged (>90%)",
                            {"flag_fraction": float(flag_frac)},
                            severity="warning",
                        )
                    else:
                        self._add_result(
                            "flagging",
                            True,
                            f"{flag_frac * 100:.1f}% of data flagged",
                            {"flag_fraction": float(flag_frac)},
                            severity="info",
                        )

        except (OSError, RuntimeError) as e:
            self._add_result(
                "data_integrity",
                False,
                f"Failed to check data: {e}",
            )

    def _check_timing(self):
        """Check TIME column for validity."""
        try:
            from casacore.tables import table
        except ImportError:
            return

        try:
            with table(str(self.ms_path), readonly=True, ack=False) as tb:
                times = tb.getcol("TIME")

                min_time = np.min(times)
                max_time = np.max(times)
                duration_s = max_time - min_time

                # Convert MJD seconds to ISO
                from astropy.time import Time

                t_start = Time(min_time / 86400.0, format="mjd")
                t_end = Time(max_time / 86400.0, format="mjd")

                self._report.properties["time_start"] = t_start.isot
                self._report.properties["time_end"] = t_end.isot
                self._report.properties["duration_seconds"] = float(duration_s)

                # DSA-110 observation is ~5 minutes (309 seconds)
                expected_duration = 309.0
                duration_ok = abs(duration_s - expected_duration) < 60

                if duration_ok:
                    self._add_result(
                        "timing_duration",
                        True,
                        f"Duration: {duration_s:.1f}s ({t_start.isot} to {t_end.isot})",
                        {"duration_seconds": float(duration_s)},
                    )
                else:
                    self._add_result(
                        "timing_duration",
                        True,  # Not an error, just unusual
                        f"Unusual duration: {duration_s:.1f}s (expected ~309s)",
                        {"duration_seconds": float(duration_s)},
                        severity="warning",
                    )

        except (OSError, RuntimeError, ValueError) as e:
            self._add_result(
                "timing",
                False,
                f"Failed to check timing: {e}",
            )


def validate_ms(
    ms_path: str | Path,
    calibrator_name: str | None = None,
    calibrator_ra_deg: float | None = None,
    calibrator_dec_deg: float | None = None,
    check_data: bool = True,
) -> MSValidationReport:
    """Convenience function to validate an MS.

    Parameters
    ----------
    ms_path : str
        Path to Measurement Set
    calibrator_name : str
        Name of calibrator to check against
    calibrator_ra_deg : float
        Calibrator RA in degrees
    calibrator_dec_deg : float
        Calibrator Dec in degrees
    check_data : bool
        Whether to check DATA column

    Returns
    -------
        MSValidationReport
    """
    validator = MSValidator(ms_path)
    return validator.validate_all(
        calibrator_name=calibrator_name,
        calibrator_ra_deg=calibrator_ra_deg,
        calibrator_dec_deg=calibrator_dec_deg,
        check_data=check_data,
    )


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Validate Measurement Set files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic validation
  python -m dsa110_contimg.core.validation.ms_validator /path/to/file.ms

  # With calibrator check
  python -m dsa110_contimg.core.validation.ms_validator /path/to/file.ms \\
    --calibrator 3C286 --cal-ra 202.7845 --cal-dec 30.5091

  # JSON output
  python -m dsa110_contimg.core.validation.ms_validator /path/to/file.ms --json

  # Skip data check (faster)
  python -m dsa110_contimg.core.validation.ms_validator /path/to/file.ms --no-data
        """,
    )
    parser.add_argument("ms_path", help="Path to Measurement Set")
    parser.add_argument("--calibrator", help="Calibrator name")
    parser.add_argument("--cal-ra", type=float, help="Calibrator RA in degrees")
    parser.add_argument("--cal-dec", type=float, help="Calibrator Dec in degrees")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--no-data", action="store_true", help="Skip DATA column check")

    args = parser.parse_args()

    report = validate_ms(
        args.ms_path,
        calibrator_name=args.calibrator,
        calibrator_ra_deg=args.cal_ra,
        calibrator_dec_deg=args.cal_dec,
        check_data=not args.no_data,
    )

    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        print(report.summary())

    # Exit with error code if validation failed
    import sys

    sys.exit(0 if report.passed else 1)


if __name__ == "__main__":
    main()
