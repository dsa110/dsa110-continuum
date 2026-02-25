"""
Pipeline hooks for post-stage processing and metrics ingestion.

This module provides hooks that are called after specific pipeline stages
complete, enabling:
- Calibration metrics ingestion for multi-epoch trending (Phase 3.1)
- Quality monitoring and alerting
- Automatic report generation
- Database updates for downstream analysis

Usage:
    These hooks are called automatically by pipeline stages. For manual use:

    from dsa110_contimg.core.qa.pipeline_hooks import (
        hook_calibration_complete,
        ingest_calibration_metrics,
    )

    # After calibration, ingest metrics for trending
    hook_calibration_complete(
        ms_path="/path/to/calibrated.ms",
        caltable_set="calibration_set_name",
    )
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from dsa110_contimg.common.utils import get_env_path
from dsa110_contimg.interfaces.api.db_adapters.query_builder import QueryBuilder

logger = logging.getLogger(__name__)

# Default database path - use PIPELINE_DB env var if set
_CONTIMG_BASE = str(get_env_path("CONTIMG_BASE_DIR", default="/data/dsa110-contimg"))
DEFAULT_DB_PATH = Path(os.environ.get("PIPELINE_DB", f"{_CONTIMG_BASE}/state/db/pipeline.sqlite3"))


@dataclass
class CalibrationMetricsRecord:
    """Record of calibration metrics for database ingestion.

    Captures key metrics from a single calibration run for long-term
    trending and anomaly detection.
    """

    # Identification
    ms_path: str
    observation_mjd: float
    calibrator_name: str | None = None
    calibrator_field_id: int | None = None

    # Bandpass metrics
    bp_mean_amp: float | None = None
    bp_std_amp: float | None = None
    bp_median_phase_scatter_deg: float | None = None
    bp_max_phase_scatter_deg: float | None = None
    bp_flagged_fraction: float | None = None
    bp_snr_median: float | None = None
    bp_snr_min: float | None = None

    # Gain metrics
    gain_mean_amp: float | None = None
    gain_std_amp: float | None = None
    gain_phase_scatter_deg: float | None = None
    gain_flagged_fraction: float | None = None
    gain_snr_median: float | None = None
    gain_snr_min: float | None = None

    # Delay metrics
    delay_max_ns: float | None = None
    delay_std_ns: float | None = None
    delay_flagged_fraction: float | None = None

    # Overall quality
    overall_quality_score: float | None = None
    quality_grade: str | None = None
    flags_total_fraction: float | None = None

    # Residual diagnostics
    residual_rms_jy: float | None = None
    residual_phase_scatter_deg: float | None = None

    # Per-antenna summary (will be JSON serialized)
    per_antenna_summary: list[dict[str, Any]] | None = None

    # Problem flags
    has_rfi_contamination: bool = False
    has_antenna_issues: bool = False
    problematic_antennas: list[int] | None = None

    # Metadata
    caltable_set_name: str | None = None
    recorded_at: float = field(default_factory=time.time)

    def to_db_row(self) -> dict[str, Any]:
        """Convert to database row dictionary."""
        row = asdict(self)

        # Serialize lists/dicts to JSON
        if row["per_antenna_summary"] is not None:
            row["per_antenna_summary"] = json.dumps(row["per_antenna_summary"])
        if row["problematic_antennas"] is not None:
            row["problematic_antennas"] = json.dumps(row["problematic_antennas"])

        # Convert booleans to integers for SQLite
        row["has_rfi_contamination"] = 1 if row["has_rfi_contamination"] else 0
        row["has_antenna_issues"] = 1 if row["has_antenna_issues"] else 0

        return row


def extract_calibration_metrics(
    ms_path: str,
    caltable_paths: dict[str, str] | None = None,
    calibrator_name: str | None = None,
) -> CalibrationMetricsRecord:
    """Extract calibration metrics from MS and calibration tables.

    Parameters
    ----------
    ms_path : str
        Path to the calibrated measurement set.
    caltable_paths : dict
        Dictionary mapping table type to path, e.g. {"k": "/path/to.kcal", "bp": "/path/to.bcal", "g": "/path/to.gcal"}.
    calibrator_name : str
        Name of the calibrator used.

    Returns
    -------
        CalibrationMetricsRecord
        Object containing extracted calibration metrics.
    """
    from dsa110_contimg.core.calibration.qa import compute_calibration_metrics
    from dsa110_contimg.common.utils import get_ms_mid_mjd

    # Get observation time
    try:
        observation_mjd = get_ms_mid_mjd(ms_path)
    except (OSError, RuntimeError) as e:
        logger.warning(f"Could not get observation MJD from {ms_path}: {e}")
        observation_mjd = 0.0

    record = CalibrationMetricsRecord(
        ms_path=ms_path,
        observation_mjd=observation_mjd,
        calibrator_name=calibrator_name,
    )

    if caltable_paths is None:
        caltable_paths = _find_caltables_for_ms(ms_path)

    # Extract bandpass metrics
    if "bp" in caltable_paths or "bpcal" in caltable_paths:
        bp_path = caltable_paths.get("bp") or caltable_paths.get("bpcal")
        try:
            bp_metrics = compute_calibration_metrics(bp_path, cal_type="bp")
            if bp_metrics:
                record.bp_mean_amp = bp_metrics.get("mean_amp")
                record.bp_std_amp = bp_metrics.get("std_amp")
                record.bp_median_phase_scatter_deg = bp_metrics.get("median_phase_scatter")
                record.bp_max_phase_scatter_deg = bp_metrics.get("max_phase_scatter")
                record.bp_flagged_fraction = bp_metrics.get("flagged_fraction")
                record.bp_snr_median = bp_metrics.get("snr_median")
                record.bp_snr_min = bp_metrics.get("snr_min")
        except (ValueError, KeyError, RuntimeError) as e:
            logger.warning(f"Failed to extract BP metrics: {e}")

    # Extract gain metrics
    if "g" in caltable_paths or "gpcal" in caltable_paths:
        g_path = caltable_paths.get("g") or caltable_paths.get("gpcal")
        try:
            g_metrics = compute_calibration_metrics(g_path, cal_type="g")
            if g_metrics:
                record.gain_mean_amp = g_metrics.get("mean_amp")
                record.gain_std_amp = g_metrics.get("std_amp")
                record.gain_phase_scatter_deg = g_metrics.get("phase_scatter")
                record.gain_flagged_fraction = g_metrics.get("flagged_fraction")
                record.gain_snr_median = g_metrics.get("snr_median")
                record.gain_snr_min = g_metrics.get("snr_min")
        except (ValueError, KeyError, RuntimeError) as e:
            logger.warning(f"Failed to extract gain metrics: {e}")

    # Extract delay metrics
    if "k" in caltable_paths or "kcal" in caltable_paths:
        k_path = caltable_paths.get("k") or caltable_paths.get("kcal")
        try:
            k_metrics = compute_calibration_metrics(k_path, cal_type="k")
            if k_metrics:
                record.delay_max_ns = k_metrics.get("max_delay_ns")
                record.delay_std_ns = k_metrics.get("std_delay_ns")
                record.delay_flagged_fraction = k_metrics.get("flagged_fraction")
        except (ValueError, KeyError, RuntimeError) as e:
            logger.warning(f"Failed to extract delay metrics: {e}")

    # Compute overall quality score
    record.overall_quality_score, record.quality_grade = _compute_quality_score(record)

    # Extract residual metrics from MS if CORRECTED_DATA exists
    try:
        residual_metrics = _extract_residual_metrics(ms_path)
        if residual_metrics:
            record.residual_rms_jy = residual_metrics.get("rms_jy")
            record.residual_phase_scatter_deg = residual_metrics.get("phase_scatter_deg")
    except (RuntimeError, ValueError) as e:
        logger.debug(f"Could not extract residual metrics: {e}")

    return record


def _find_caltables_for_ms(ms_path: str) -> dict[str, str]:
    """Find calibration tables associated with an MS.

    Looks in:
    1. Same directory as MS
    2. products/caltables/ with matching group_id
    3. Database lookup
    """
    caltables = {}
    ms_path_obj = Path(ms_path)

    # Look for caltables in same directory
    ms_dir = ms_path_obj.parent
    ms_stem = ms_path_obj.stem

    for suffix, cal_type in [
        (".kcal", "k"),
        (".bcal", "bp"),
        (".bpcal", "bp"),
        (".gcal", "g"),
        (".gpcal", "g"),
    ]:
        cal_path = ms_dir / f"{ms_stem}{suffix}"
        if cal_path.exists():
            caltables[cal_type] = str(cal_path)

    # Look in products/caltables
    contimg_base = str(get_env_path("CONTIMG_BASE_DIR", default="/data/dsa110-contimg"))
    products_caltables = Path(contimg_base) / "products" / "caltables"
    if products_caltables.exists():
        for cal_file in products_caltables.glob(f"*{ms_stem}*"):
            if cal_file.suffix in (".kcal", ".bcal", ".bpcal", ".gcal", ".gpcal"):
                cal_type = cal_file.suffix[1:].replace("cal", "")
                if cal_type == "bp":
                    cal_type = "bp"
                elif cal_type == "gp":
                    cal_type = "g"
                caltables.setdefault(cal_type, str(cal_file))

    return caltables


def _compute_quality_score(record: CalibrationMetricsRecord) -> tuple[float, str]:
    """Compute overall quality score from individual metrics.

    Returns
    -------
        Tuple of (score 0-100, grade string)
    """
    scores = []
    weights = []

    # Bandpass quality (weight: 3)
    if record.bp_snr_median is not None:
        bp_score = min(100, record.bp_snr_median * 10)  # SNR 10 = 100%
        scores.append(bp_score)
        weights.append(3)

    if record.bp_flagged_fraction is not None:
        bp_flag_score = 100 * (1 - record.bp_flagged_fraction)
        scores.append(bp_flag_score)
        weights.append(2)

    if record.bp_median_phase_scatter_deg is not None:
        # Lower scatter is better, 5 deg = 100, 30 deg = 0
        bp_phase_score = max(0, 100 - (record.bp_median_phase_scatter_deg - 5) * 4)
        scores.append(bp_phase_score)
        weights.append(2)

    # Gain quality (weight: 2)
    if record.gain_snr_median is not None:
        gain_score = min(100, record.gain_snr_median * 10)
        scores.append(gain_score)
        weights.append(2)

    if record.gain_flagged_fraction is not None:
        gain_flag_score = 100 * (1 - record.gain_flagged_fraction)
        scores.append(gain_flag_score)
        weights.append(1)

    # Default score if no metrics available
    if not scores:
        return 50.0, "unknown"

    # Weighted average
    total_score = sum(s * w for s, w in zip(scores, weights)) / sum(weights)

    # Determine grade
    if total_score >= 90:
        grade = "excellent"
    elif total_score >= 75:
        grade = "good"
    elif total_score >= 50:
        grade = "acceptable"
    elif total_score >= 25:
        grade = "poor"
    else:
        grade = "failed"

    return total_score, grade


def _extract_residual_metrics(ms_path: str) -> dict[str, float] | None:
    """Extract residual metrics from calibrated MS.

    Computes RMS and phase scatter of CORRECTED_DATA - MODEL_DATA.
    """
    try:
        from casacore.tables import table
    except ImportError:
        return None

    ms_path_str = str(ms_path)

    try:
        with table(ms_path_str, readonly=True, ack=False) as tb:
            cols = tb.colnames()

            if "CORRECTED_DATA" not in cols:
                return None

            # Sample to avoid memory issues
            n_rows = min(1000, tb.nrows())
            corrected = tb.getcol("CORRECTED_DATA", 0, n_rows)
            flags = tb.getcol("FLAG", 0, n_rows)

            if "MODEL_DATA" in cols:
                model = tb.getcol("MODEL_DATA", 0, n_rows)
                residuals = corrected - model
            else:
                # Just use corrected data variance
                residuals = corrected

            # Compute metrics on unflagged data
            mask = ~flags
            if np.sum(mask) == 0:
                return None

            residuals_unflagged = residuals[mask]
            rms = np.sqrt(np.mean(np.abs(residuals_unflagged) ** 2))
            phases = np.angle(residuals_unflagged, deg=True)
            phase_scatter = np.std(phases)

            return {
                "rms_jy": float(rms),
                "phase_scatter_deg": float(phase_scatter),
            }
    except (RuntimeError, ValueError, ImportError) as e:
        logger.debug(f"Failed to extract residual metrics: {e}")
        return None


def ingest_calibration_metrics(
    record: CalibrationMetricsRecord,
    db_path: Path | None = None,
) -> bool:
    """Ingest calibration metrics record into database.

    Parameters
    ----------
    record : CalibrationMetricsRecord
        CalibrationMetricsRecord to ingest
    db_path : str, optional
        Path to database (defaults to pipeline.sqlite3)

    Returns
    -------
        bool
        True if successful, False otherwise
    """
    import sqlite3

    if db_path is None:
        db_path = DEFAULT_DB_PATH

    if not db_path.exists():
        logger.warning(f"Database not found: {db_path}")
        return False

    row = record.to_db_row()

    # Build INSERT statement using QueryBuilder for SQL injection safety
    qb = QueryBuilder()
    columns = list(row.keys())
    sql = qb.insert("calibration_metrics", columns)

    try:
        conn = sqlite3.connect(str(db_path), timeout=30.0)
        conn.execute("PRAGMA journal_mode=WAL")
        cursor = conn.cursor()
        cursor.execute(sql, list(row.values()))
        conn.commit()
        conn.close()
        logger.info(f"Ingested calibration metrics for {record.ms_path}")
        return True
    except sqlite3.OperationalError as e:
        if "no such table" in str(e):
            logger.warning(
                "calibration_metrics table not found. Run database migration to create it."
            )
        else:
            logger.error(f"Database error ingesting calibration metrics: {e}")
        return False
    except (sqlite3.Error, OSError) as e:
        logger.error(f"Failed to ingest calibration metrics: {e}")
        return False


def hook_calibration_complete(
    ms_path: str | None = None,
    caltable_set: str | None = None,
    caltable_paths: dict[str, str] | None = None,
    calibrator_name: str | None = None,
    db_path: Path | None = None,
) -> bool:
    """Hook called after calibration stage completes.

        This hook:
        1. Extracts calibration metrics from the MS and cal tables
        2. Ingests metrics into the database for trending
        3. Optionally triggers alerts for quality issues

    Parameters
    ----------
    ms_path : str
        Path to the calibrated measurement set
    caltable_set : str
        Name of the calibration table set
    caltable_paths : dict
        Dictionary of calibration table paths
    calibrator_name : str
        Name of the calibrator used
    db_path : str, optional
        Path to database (defaults to pipeline.sqlite3)

    Returns
    -------
        bool
        True if metrics were successfully ingested
    """
    logger.debug("Calibration complete hook triggered")

    # If no MS path provided, try to get from context
    if ms_path is None:
        logger.debug("No MS path provided to calibration hook, skipping metrics ingestion")
        return False

    try:
        # Extract metrics
        record = extract_calibration_metrics(
            ms_path=ms_path,
            caltable_paths=caltable_paths,
            calibrator_name=calibrator_name,
        )

        if caltable_set:
            record.caltable_set_name = caltable_set

        # Ingest to database
        success = ingest_calibration_metrics(record, db_path=db_path)

        # Log quality summary
        if record.quality_grade:
            logger.info(
                f"Calibration quality: {record.quality_grade} "
                f"(score: {record.overall_quality_score:.1f})"
            )

        # Trigger alert for poor quality
        if record.quality_grade in ("poor", "failed"):
            logger.warning(
                f"Poor calibration quality detected for {ms_path}: "
                f"grade={record.quality_grade}, score={record.overall_quality_score:.1f}"
            )
            # Could trigger AlertManager here in future

        return success

    except (RuntimeError, ValueError, OSError) as e:
        logger.warning(f"Calibration metrics hook failed: {e}")
        return False


def update_calibration_trending(
    period_type: str = "daily",
    db_path: Path | None = None,
) -> bool:
    """Update calibration trending aggregates.

        Computes daily/weekly/monthly summaries from calibration_metrics
        and stores in calibration_trending table.

    Parameters
    ----------
    period_type : str
        "daily", "weekly", or "monthly"
    db_path : str
        Path to database

    Returns
    -------
        bool
        True if successful
    """
    import sqlite3
    from datetime import datetime, timedelta

    if db_path is None:
        db_path = DEFAULT_DB_PATH

    if not db_path.exists():
        return False

    # Determine period boundaries
    now = datetime.utcnow()

    if period_type == "daily":
        period_start = datetime(now.year, now.month, now.day) - timedelta(days=1)
        period_end = datetime(now.year, now.month, now.day)
    elif period_type == "weekly":
        # Start of current week
        start_of_week = now - timedelta(days=now.weekday() + 7)
        period_start = datetime(start_of_week.year, start_of_week.month, start_of_week.day)
        period_end = period_start + timedelta(days=7)
    elif period_type == "monthly":
        # Previous month
        first_of_month = datetime(now.year, now.month, 1)
        period_end = first_of_month
        if now.month == 1:
            period_start = datetime(now.year - 1, 12, 1)
        else:
            period_start = datetime(now.year, now.month - 1, 1)
    else:
        logger.error(f"Unknown period type: {period_type}")
        return False

    # Convert to MJD
    from astropy.time import Time

    period_start_mjd = Time(period_start).mjd
    period_end_mjd = Time(period_end).mjd

    try:
        conn = sqlite3.connect(str(db_path), timeout=30.0)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        # Query metrics for period
        cursor.execute(
            """
            SELECT
                COUNT(*) as n_calibrations,
                SUM(CASE WHEN quality_grade = 'excellent' OR quality_grade = 'good' THEN 1 ELSE 0 END) as n_good,
                SUM(CASE WHEN quality_grade = 'acceptable' THEN 1 ELSE 0 END) as n_acceptable,
                SUM(CASE WHEN quality_grade = 'poor' THEN 1 ELSE 0 END) as n_poor,
                SUM(CASE WHEN quality_grade = 'failed' THEN 1 ELSE 0 END) as n_failed,
                AVG(bp_mean_amp) as bp_amp_mean,
                AVG(bp_std_amp) as bp_amp_std,
                AVG(bp_median_phase_scatter_deg) as bp_phase_scatter_mean_deg,
                AVG(gain_mean_amp) as gain_amp_mean,
                AVG(gain_std_amp) as gain_amp_std,
                AVG(gain_phase_scatter_deg) as gain_phase_scatter_mean_deg,
                AVG(overall_quality_score) as quality_score_mean,
                AVG(flags_total_fraction) as flags_fraction_mean
            FROM calibration_metrics
            WHERE observation_mjd >= ? AND observation_mjd < ?
            """,
            (period_start_mjd, period_end_mjd),
        )

        row = cursor.fetchone()

        if row["n_calibrations"] == 0:
            logger.info(f"No calibrations found for {period_type} period")
            conn.close()
            return True

        # Insert or replace trending record
        cursor.execute(
            """
            INSERT OR REPLACE INTO calibration_trending (
                period_type, period_start_mjd, period_end_mjd,
                n_calibrations, n_good, n_acceptable, n_poor, n_failed,
                bp_amp_mean, bp_amp_std, bp_phase_scatter_mean_deg,
                gain_amp_mean, gain_amp_std, gain_phase_scatter_mean_deg,
                quality_score_mean, flags_fraction_mean,
                computed_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                period_type,
                period_start_mjd,
                period_end_mjd,
                row["n_calibrations"],
                row["n_good"] or 0,
                row["n_acceptable"] or 0,
                row["n_poor"] or 0,
                row["n_failed"] or 0,
                row["bp_amp_mean"],
                row["bp_amp_std"],
                row["bp_phase_scatter_mean_deg"],
                row["gain_amp_mean"],
                row["gain_amp_std"],
                row["gain_phase_scatter_mean_deg"],
                row["quality_score_mean"],
                row["flags_fraction_mean"],
                time.time(),
            ),
        )

        conn.commit()
        conn.close()

        logger.info(
            f"Updated {period_type} calibration trending: {row['n_calibrations']} calibrations"
        )
        return True

    except (sqlite3.Error, OSError) as e:
        logger.error(f"Failed to update calibration trending: {e}")
        return False


def query_calibration_trending(
    start_mjd: float | None = None,
    end_mjd: float | None = None,
    period_type: str = "daily",
    db_path: Path | None = None,
) -> list[dict[str, Any]]:
    """Query calibration trending data.

    Parameters
    ----------
    start_mjd : float
        Start of query range (MJD)
    end_mjd : float
        End of query range (MJD)
    period_type : str
        Filter by period type
    db_path : str
        Path to database

    Returns
    -------
        list of dict
        List of trending records as dictionaries
    """
    import sqlite3

    if db_path is None:
        db_path = DEFAULT_DB_PATH

    if not db_path.exists():
        return []

    try:
        conn = sqlite3.connect(str(db_path), timeout=30.0)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        conditions = ["period_type = ?"]
        params: list[Any] = [period_type]

        if start_mjd is not None:
            conditions.append("period_start_mjd >= ?")
            params.append(start_mjd)

        if end_mjd is not None:
            conditions.append("period_end_mjd <= ?")
            params.append(end_mjd)

        where_clause = " AND ".join(conditions)

        cursor.execute(
            f"""
            SELECT * FROM calibration_trending
            WHERE {where_clause}
            ORDER BY period_start_mjd DESC
            """,
            params,
        )

        results = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return results

    except (sqlite3.Error, OSError) as e:
        logger.error(f"Failed to query calibration trending: {e}")
        return []
