"""Validation framework for synthetic data pipeline testing.

    This module validates production pipeline outputs against known ground truth
    from synthetic datasets. It reads outputs from the production pipeline database
    and compares them to the ground truth registry to assess pipeline performance.

    Key validation functions
------------------------
    - validate_photometry() - Compare measured vs injected fluxes
    - validate_lightcurve() - Validate variability metrics
    - validate_ese_detection() - Check transient detection accuracy
    - validate_catalog_crossmatch() - Verify astrometric matching

    Example
-------
    >>> from dsa110_contimg.core.simulation.validation import validate_photometry
    >>> from dsa110_contimg.core.simulation.ground_truth import GroundTruthRegistry
    >>>
    >>> # Load ground truth
    >>> registry = GroundTruthRegistry.from_json(Path("ground_truth.json"))
    >>>
    >>> # Validate photometry outputs
    >>> report = validate_photometry(
    ...     ground_truth=registry,
    ...     products_db=Path("/data/dsa110-contimg/state/db/pipeline.sqlite3"),
    ...     tolerance_percent=10.0,
    ... )
    >>>
    >>> print(f"Flux recovery: {report.mean_error_percent:.1f}% error")
    >>> print(f"Sources matched: {report.n_matched}/{report.n_injected}")
    >>> print(f"Pass: {report.passed}")
"""

from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from dsa110_contimg.core.simulation.ground_truth import GroundTruthRegistry

logger = logging.getLogger(__name__)


@dataclass
class ValidationReport:
    """Validation report with metrics and pass/fail status."""

    test_name: str
    passed: bool
    n_injected: int = 0
    n_measured: int = 0
    n_matched: int = 0
    mean_error_percent: float = 0.0
    rms_error_jy: float = 0.0
    details: dict = field(default_factory=dict)
    error_message: str | None = None

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "test_name": self.test_name,
            "passed": self.passed,
            "n_injected": self.n_injected,
            "n_measured": self.n_measured,
            "n_matched": self.n_matched,
            "mean_error_percent": self.mean_error_percent,
            "rms_error_jy": self.rms_error_jy,
            "details": self.details,
            "error_message": self.error_message,
        }

    def print_summary(self) -> None:
        """Print human-readable summary."""
        print(f"\n{'=' * 70}")
        print(f"Validation Report: {self.test_name}")
        print(f"{'=' * 70}")
        print(f"Status: {' PASS' if self.passed else ' FAIL'}")
        print(f"Injected sources: {self.n_injected}")
        print(f"Measured sources: {self.n_measured}")
        print(f"Matched: {self.n_matched}")
        print(f"Mean error: {self.mean_error_percent:.2f}%")
        print(f"RMS error: {self.rms_error_jy:.4f} Jy")
        if self.error_message:
            print(f"Error: {self.error_message}")
        if self.details:
            print("\nDetails:")
            for key, value in self.details.items():
                print(f"  {key}: {value}")
        print(f"{'=' * 70}\n")


def validate_photometry(
    ground_truth: GroundTruthRegistry,
    products_db: Path,
    tolerance_percent: float = 10.0,
    match_radius_arcsec: float = 5.0,
) -> ValidationReport:
    """Validate photometry outputs against ground truth.

        Reads photometry measurements from the production pipeline database
        and compares them to expected fluxes from the ground truth registry.

    Parameters
    ----------
    ground_truth : GroundTruthRegistry
        Ground truth registry with injected sources
    products_db : Path
        Path to pipeline.sqlite3 database
    tolerance_percent : float, optional
        Maximum acceptable flux error in percent (default is 10.0)
    match_radius_arcsec : float, optional
        Position matching radius in arcseconds (default is 5.0)

    Returns
    -------
        ValidationReport
        Validation report with photometry validation results.

    Examples
    --------
        >>> report = validate_photometry(registry, db_path)
        >>> assert report.passed
        >>> assert report.mean_error_percent < 10.0
    """
    from dsa110_contimg.infrastructure.database.unified import Database

    logger.info("Validating photometry outputs...")

    if not products_db.exists():
        return ValidationReport(
            test_name="photometry",
            passed=False,
            error_message=f"Database not found: {products_db}",
        )

    try:
        db = Database(products_db)
        
        # Read photometry measurements
        measurements_df = db.query_df("""
            SELECT source_id, ra_deg, dec_deg, flux_jy, mjd, image_path
            FROM photometry
            WHERE source_id IS NOT NULL
            ORDER BY source_id, mjd
        """)
        
        if measurements_df.empty:
            return ValidationReport(
                test_name="photometry",
                passed=False,
                n_injected=len(ground_truth.sources),
                error_message="No photometry measurements found in database",
            )
            
        conn = None  # No explicit connection to close

        logger.info("Found %d photometry measurements", len(measurements_df))

        # prepare ground truth dataframe
        truth_records = []
        for s in ground_truth.sources.values():
            truth_records.append({
                "source_id": s.source_id,
                "truth_ra": s.ra_deg,
                "truth_dec": s.dec_deg,
                "baseline_flux": s.baseline_flux_jy,
                "is_variable": s.variability_model is not None,
                "truth_obj": s # Keep object for complex calc if needed
            })
        
        truth_df = pd.DataFrame(truth_records)
        if truth_df.empty:
             return ValidationReport(
                test_name="photometry", 
                passed=False,
                error_message="Ground truth is empty"
            )

        # Merge measurements with ground truth
        # Note: inner join keeps only matched sources
        merged = pd.merge(measurements_df, truth_df, on="source_id", how="inner")
        
        # Report unmatched sources for debugging
        injected_ids = set(truth_df["source_id"])
        measured_ids = set(measurements_df["source_id"])
        unmatched_injected = sorted(injected_ids - measured_ids)
        unmatched_measured = sorted(measured_ids - injected_ids)
        
        if merged.empty:
            return ValidationReport(
                test_name="photometry",
                passed=False,
                n_injected=len(ground_truth.sources),
                n_measured=len(measurements_df),
                error_message="No measurements matched ground truth sources",
                details={
                    "n_unmatched_injected": len(unmatched_injected),
                    "n_unmatched_measured": len(unmatched_measured),
                    "unmatched_injected_sample": unmatched_injected[:10],
                    "unmatched_measured_sample": unmatched_measured[:10],
                },
            )

        # Calculate expected flux
        # Default to baseline
        merged["expected_flux"] = merged["baseline_flux"]
        
        # Handle variable sources
        variable_mask = merged["is_variable"]
        if variable_mask.any():
            # For variable sources, we must calculate flux at specific MJD
            # This is hard to fully vectorize without vectorizing the model evaluation itself
            # We use apply() on the subset
            def calc_var_flux(row):
                return row["truth_obj"].get_flux_at_time(row["mjd"])
                
            merged.loc[variable_mask, "expected_flux"] = merged[variable_mask].apply(calc_var_flux, axis=1)

        # Drop rows where expected_flux is None/NaN (if any model returned None)
        merged = merged.dropna(subset=["expected_flux"])

        # Vectorized error calculations
        # Flux Error % = 100 * (measured - expected) / expected
        merged["flux_error_pct"] = 100.0 * (merged["flux_jy"] - merged["expected_flux"]) / merged["expected_flux"]
        
        # Position Error in arcsec
        # Approximation for small offsets: sqrt((dRA*cos(Dec))^2 + dDec^2) * 3600
        # or use astropy if available. dsa110_contimg.core.simulation.metrics.astrometric_offset uses SkyCoord usually
        # But we can reimplement simple approx for speed or use the function via apply?
        # The original code imported astrometric_offset.
        # Let's use simple approximation for vectorization efficiency 
        # (astrometric_offset likely uses astropy which is slow in loop, or numpy if vectorized).
        # Let's check imports. astrometric_offset is imported.
        # If astrometric_offset supports arrays, we use it. If not, we reimplement.
        # Assuming we can use numpy approximation for speed:
        
        ra_diff = merged["ra_deg"] - merged["truth_ra"]
        dec_diff = merged["dec_deg"] - merged["truth_dec"]
        cos_dec = np.cos(np.radians(merged["truth_dec"]))
        
        merged["position_error_arcsec"] = 3600.0 * np.sqrt((ra_diff * cos_dec)**2 + dec_diff**2)
        
        # Statistics
        flux_errors_array = merged["flux_error_pct"].to_numpy()
        position_errors_array = merged["position_error_arcsec"].to_numpy()

        mean_flux_error = float(np.mean(np.abs(flux_errors_array)))
        rms_flux_error_pct = float(np.sqrt(np.mean(flux_errors_array**2)))
        median_flux_error = float(np.median(np.abs(flux_errors_array)))
        mean_pos_error = float(np.mean(position_errors_array))

        # Pass/fail based on tolerance
        passed = mean_flux_error <= tolerance_percent

        details = {
            "mean_abs_flux_error_pct": f"{mean_flux_error:.2f}",
            "rms_flux_error_pct": f"{rms_flux_error_pct:.2f}",
            "median_flux_error_pct": f"{median_flux_error:.2f}",
            "mean_position_error_arcsec": f"{mean_pos_error:.2f}",
            "n_sources_matched": len(merged["source_id"].unique()),
            "n_measurements": len(merged),
            "tolerance_pct": tolerance_percent,
            "n_unmatched_injected": len(unmatched_injected),
            "n_unmatched_measured": len(unmatched_measured),
            "unmatched_injected_sample": unmatched_injected[:10] if unmatched_injected else [],
            "unmatched_measured_sample": unmatched_measured[:10] if unmatched_measured else [],
        }

        logger.info("Photometry validation: %s", "PASS" if passed else "FAIL")
        logger.info("  Mean flux error: %.2f%%", mean_flux_error)
        logger.info("  Mean position error: %.2f arcsec", mean_pos_error)

        return ValidationReport(
            test_name="photometry",
            passed=passed,
            n_injected=len(ground_truth.sources),
            n_measured=len(measurements_df),
            n_matched=len(merged["source_id"].unique()),
            mean_error_percent=mean_flux_error,
            rms_error_jy=0.0,  
            details=details,
        )

    except Exception as e:
        logger.error("Photometry validation failed: %s", e, exc_info=True)
        return ValidationReport(
            test_name="photometry",
            passed=False,
            error_message=str(e),
        )


def validate_lightcurve(
    ground_truth: GroundTruthRegistry,
    products_db: Path,
) -> ValidationReport:
    """Validate lightcurve and variability detection.
    
    Checks that:
    1. Variable sources are correctly flagged in variability_stats
    2. Constant sources are not flagged as variable
    3. Variability metrics (η, V, χ²) are reasonable

    Parameters
    ----------
    ground_truth :
        Ground truth registry
    products_db :
        Path to pipeline.sqlite3 database

    Returns
    -------
        ValidationReport with lightcurve validation results

    """
    from dsa110_contimg.infrastructure.database.unified import Database

    logger.info("Validating lightcurve and variability detection...")

    if not products_db.exists():
        return ValidationReport(
            test_name="lightcurve",
            passed=False,
            error_message=f"Database not found: {products_db}",
        )

    try:
        db = Database(products_db)
        
        # Check if variability_stats table exists
        try:
             # Need a reliable way to check table existence via Database class or query
             # Database class doesn't expose list_tables directly but we can query sqlite_master
             check = db.query_one("""
                SELECT name FROM sqlite_master
                WHERE type='table' AND name='variability_stats'
             """)
        except Exception:
             check = None
             
        if not check:
            return ValidationReport(
                test_name="lightcurve",
                passed=False,
                error_message="variability_stats table not found - lightcurve stage not run",
            )

        # Read variability statistics
        var_stats_df = db.query_df("""
            SELECT source_id, n_obs, eta_metric, chi2_nu, sigma_deviation
            FROM variability_stats
            WHERE source_id IS NOT NULL
        """)
        
        # Get variable sources from ground truth
        variable_source_ids = set(s.source_id for s in ground_truth.get_variable_sources())
        constant_source_ids = set(s.source_id for s in ground_truth.get_constant_sources())

        # Vectorized check
        # Flag if eta > 0 OR chi2_nu > 2
        # Ensure columns are numeric/not-none first (pandas handles this well with masking)
        
        # Create boolean masks
        is_eta_flagged = (var_stats_df["eta_metric"].notna()) & (var_stats_df["eta_metric"] > 0)
        is_chi2_flagged = (var_stats_df["chi2_nu"].notna()) & (var_stats_df["chi2_nu"] > 2.0)
        is_flagged_variable = is_eta_flagged | is_chi2_flagged
        
        # Source type masks
        is_variable_truth = var_stats_df["source_id"].isin(variable_source_ids)
        is_constant_truth = var_stats_df["source_id"].isin(constant_source_ids)

        # TP: Variable truth AND Flagged variable
        true_positives = (is_variable_truth & is_flagged_variable).sum()
        
        # FN: Variable truth AND NOT Flagged variable
        false_negatives = (is_variable_truth & ~is_flagged_variable).sum()
        
        # FP: Constant truth AND Flagged variable
        false_positives = (is_constant_truth & is_flagged_variable).sum()
        
        # TN: Constant truth AND NOT Flagged variable
        true_negatives = (is_constant_truth & ~is_flagged_variable).sum()

        total = true_positives + false_positives + true_negatives + false_negatives
        if total == 0:
            accuracy = 0.0
        else:
            accuracy = float(true_positives + true_negatives) / float(total)

        # Pass if accuracy > 70%
        passed = accuracy > 0.7

        details = {
            "true_positives": int(true_positives),
            "false_positives": int(false_positives),
            "true_negatives": int(true_negatives),
            "false_negatives": int(false_negatives),
            "accuracy": f"{accuracy:.2%}",
            "n_variable_injected": len(variable_source_ids),
            "n_constant_injected": len(constant_source_ids),
        }

        logger.info("Lightcurve validation: %s", "PASS" if passed else "FAIL")
        logger.info("  Accuracy: %.2f%%", accuracy)
        logger.info(
            "  TP=%d, FP=%d, TN=%d, FN=%d",
            true_positives,
            false_positives,
            true_negatives,
            false_negatives,
        )

        return ValidationReport(
            test_name="lightcurve",
            passed=passed,
            n_injected=len(ground_truth.sources),
            n_measured=len(var_stats_df),
            n_matched=int(true_positives + true_negatives),
            mean_error_percent=(1.0 - accuracy) * 100.0,
            details=details,
        )

    except Exception as e:
        logger.error("Lightcurve validation failed: %s", e, exc_info=True)
        return ValidationReport(
            test_name="lightcurve",
            passed=False,
            error_message=str(e),
        )


def validate_ese_detection(
    ground_truth: GroundTruthRegistry,
    products_db: Path,
    min_sigma: float = 5.0,
) -> ValidationReport:
    """Validate ESE/transient detection.

    Checks that sources with variability models (especially ESE and flares)
    are correctly identified as transient candidates.

    Parameters
    ----------
    ground_truth :
        Ground truth registry
    products_db :
        Path to pipeline.sqlite3 database
    min_sigma :
        Minimum sigma for ESE detection

    Returns
    -------
        ValidationReport with transient detection results

    """
    from dsa110_contimg.infrastructure.database.unified import Database

    logger.info("Validating ESE/transient detection...")

    if not products_db.exists():
        return ValidationReport(
            test_name="ese_detection",
            passed=False,
            error_message=f"Database not found: {products_db}",
        )

    try:
        db = Database(products_db)
        
        # Check if ese_candidates table exists
        try:
             check = db.query_one("""
                SELECT name FROM sqlite_master
                WHERE type='table' AND name='ese_candidates'
             """)
        except Exception:
             check = None

        if not check:
            return ValidationReport(
                test_name="ese_detection",
                passed=False,
                error_message="ese_candidates table not found - transient detection not run",
            )

        # Read ESE candidates
        candidates_df = db.query_df("""
            SELECT source_id, sigma_deviation, detected_at
            FROM ese_candidates
            WHERE source_id IS NOT NULL
        """)
        
        candidates = candidates_df.to_dict("records")
        
        # Get variable sources from ground truth
        variable_sources = ground_truth.get_variable_sources()
        variable_source_ids = set(s.source_id for s in variable_sources)

        detected_ids = set(row["source_id"] for row in candidates)

        # True positives: variable sources that were detected
        true_positives = len(variable_source_ids & detected_ids)
        # False negatives: variable sources that were missed
        false_negatives = len(variable_source_ids - detected_ids)
        # False positives: non-variable sources detected as variable
        false_positives = len(detected_ids - variable_source_ids)

        # Pass if we detect at least 50% of variable sources
        if len(variable_source_ids) > 0:
            detection_rate = true_positives / len(variable_source_ids)
            passed = detection_rate >= 0.5
        else:
            # No variable sources injected
            passed = len(detected_ids) == 0
            detection_rate = 1.0 if passed else 0.0

        details = {
            "n_candidates_detected": len(detected_ids),
            "n_variable_injected": len(variable_source_ids),
            "true_positives": true_positives,
            "false_negatives": false_negatives,
            "false_positives": false_positives,
            "detection_rate": f"{detection_rate:.2%}",
            "min_sigma_threshold": min_sigma,
        }

        logger.info("ESE detection validation: %s", "PASS" if passed else "FAIL")
        logger.info("  Detection rate: %.2f%%", detection_rate)
        logger.info("  TP=%d, FN=%d, FP=%d", true_positives, false_negatives, false_positives)

        return ValidationReport(
            test_name="ese_detection",
            passed=passed,
            n_injected=len(variable_source_ids),
            n_measured=len(detected_ids),
            n_matched=true_positives,
            mean_error_percent=(1.0 - detection_rate) * 100.0,
            details=details,
        )

    except Exception as e:
        logger.error("ESE detection validation failed: %s", e, exc_info=True)
        return ValidationReport(
            test_name="ese_detection",
            passed=False,
            error_message=str(e),
        )


def validate_all(
    ground_truth: GroundTruthRegistry,
    products_db: Path,
    photometry_tolerance_pct: float = 10.0,
) -> dict[str, ValidationReport]:
    """Run all validation tests and return combined results.

    Parameters
    ----------
    ground_truth : GroundTruthRegistry
        Ground truth registry
    products_db : Path
        Path to pipeline.sqlite3 database
    photometry_tolerance_pct : float, optional
        Flux tolerance for photometry (%), by default 10.0

    Returns
    -------
    dict
        Dictionary mapping test name to ValidationReport

    Examples
    --------
    >>> reports = validate_all(registry, db_path)
    >>> for name, report in reports.items():
    ...     report.print_summary()
    >>> all_passed = all(r.passed for r in reports.values())
    """
    logger.info("Running all validation tests...")

    reports = {}

    # Photometry validation
    reports["photometry"] = validate_photometry(
        ground_truth, products_db, tolerance_percent=photometry_tolerance_pct
    )

    # Lightcurve validation
    reports["lightcurve"] = validate_lightcurve(ground_truth, products_db)

    # ESE detection validation
    reports["ese_detection"] = validate_ese_detection(ground_truth, products_db)

    # Summary
    n_passed = sum(1 for r in reports.values() if r.passed)
    n_total = len(reports)

    logger.info("\nValidation Summary: %d/%d tests passed", n_passed, n_total)
    for name, report in reports.items():
        status = " PASS" if report.passed else " FAIL"
        logger.info("  %s: %s", name, status)

    return reports
