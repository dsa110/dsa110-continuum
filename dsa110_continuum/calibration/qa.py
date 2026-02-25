"""
Automated calibration quality assessment.

This module provides functions for computing QA metrics from calibration tables
and assessing calibration quality with configurable thresholds.

Integrates with the existing batch/qa.py patterns while providing a more
comprehensive QA framework for pipeline integration.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# Default QA thresholds
DEFAULT_MIN_SNR = 3.0
DEFAULT_MAX_FLAG_FRACTION = 0.3
DEFAULT_MIN_AMPLITUDE = 0.1
DEFAULT_MAX_AMPLITUDE = 10.0
DEFAULT_MAX_PHASE_SCATTER_DEG = 30.0


@dataclass
class QAThresholds:
    """Configurable QA thresholds for calibration assessment."""

    min_snr: float = DEFAULT_MIN_SNR
    max_flag_fraction: float = DEFAULT_MAX_FLAG_FRACTION
    min_amplitude: float = DEFAULT_MIN_AMPLITUDE
    max_amplitude: float = DEFAULT_MAX_AMPLITUDE
    max_phase_scatter_deg: float = DEFAULT_MAX_PHASE_SCATTER_DEG


@dataclass
class CalibrationMetrics:
    """Metrics extracted from a calibration table."""

    caltable_path: str
    cal_type: str  # 'k', 'bp', 'g'
    n_solutions: int = 0
    n_flagged: int = 0
    flag_fraction: float = 0.0
    mean_amplitude: float = 0.0
    std_amplitude: float = 0.0
    median_amplitude: float = 0.0
    min_amplitude: float = 0.0
    max_amplitude: float = 0.0
    median_phase_deg: float = 0.0
    phase_scatter_deg: float = 0.0
    median_snr: float | None = None
    min_snr: float | None = None
    max_snr: float | None = None
    n_antennas: int = 0
    n_spws: int = 0
    n_channels: int = 0
    extraction_time_s: float = 0.0
    extraction_error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CalibrationMetrics:
        """Create from dictionary.

        Parameters
        ----------
        data : Dict[str, Any]
            Dictionary of data to create the instance from.

        Returns
        -------
            object
            Created instance.
        """
        return cls(**data)

    @property
    def is_valid(self) -> bool:
        """Check if metrics were extracted successfully."""
        return self.extraction_error is None and self.n_solutions > 0


@dataclass
class QAIssue:
    """A quality issue found during assessment."""

    severity: str  # 'error', 'warning', 'info'
    cal_type: str
    metric: str
    value: float
    threshold: float
    message: str


@dataclass
class CalibrationQAResult:
    """Result of calibration quality assessment."""

    ms_path: str
    passed: bool
    severity: str  # 'success', 'warning', 'error'
    overall_grade: str  # 'excellent', 'good', 'marginal', 'poor', 'failed'
    issues: list[QAIssue] = field(default_factory=list)
    metrics: list[CalibrationMetrics] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)
    assessment_time_s: float = 0.0
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "ms_path": self.ms_path,
            "passed": self.passed,
            "severity": self.severity,
            "overall_grade": self.overall_grade,
            "issues": [asdict(i) for i in self.issues],
            "metrics": [m.to_dict() for m in self.metrics],
            "summary": self.summary,
            "assessment_time_s": self.assessment_time_s,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CalibrationQAResult:
        """Create from dictionary.

        Parameters
        ----------
        data : Dict[str, Any]
            Dictionary of data to create the instance from.

        Returns
        -------
            object
            Created instance.
        """
        issues = [QAIssue(**i) for i in data.get("issues", [])]
        metrics = [CalibrationMetrics.from_dict(m) for m in data.get("metrics", [])]
        return cls(
            ms_path=data["ms_path"],
            passed=data["passed"],
            severity=data["severity"],
            overall_grade=data["overall_grade"],
            issues=issues,
            metrics=metrics,
            summary=data.get("summary", {}),
            assessment_time_s=data.get("assessment_time_s", 0.0),
            timestamp=data.get("timestamp", time.time()),
        )

    @property
    def warnings(self) -> list[str]:
        """Get warning messages."""
        return [i.message for i in self.issues if i.severity == "warning"]

    @property
    def errors(self) -> list[str]:
        """Get error messages."""
        return [i.message for i in self.issues if i.severity == "error"]


def _open_caltable(caltable_path: str):
    """Open a calibration table and return the table object.

        Uses casacore.tables for table access (more reliable than casatools).

    Parameters
    ----------
    caltable_path : str
        Path to the calibration table.

    Returns
    -------
        object
        Calibration table object.
    """
    # Ensure CASAPATH is set before importing
    from dsa110_contimg.common.utils.casa_init import ensure_casa_path

    ensure_casa_path()

    import casacore.tables as casatables

    return casatables.table(caltable_path, readonly=True)


def compute_calibration_metrics(
    caltable_path: str,
    cal_type: str | None = None,
) -> CalibrationMetrics:
    """Compute QA metrics for a calibration table.

        Extracts comprehensive statistics from a CASA calibration table including:
        - Solution counts and flagging statistics
        - Amplitude statistics (mean, std, median, min, max)
        - Phase statistics (median, scatter)
        - SNR statistics (if available)
        - Metadata (antennas, SPWs, channels)

    Parameters
    ----------
    caltable_path : str
        Path to the calibration table.
    cal_type : str, optional
        Type of calibration ('k', 'bp', 'g'). Auto-detected if None.

    Returns
    -------
        dict
        Computed QA metrics.
    """
    start_time = time.time()

    # Auto-detect cal_type from filename if not provided
    if cal_type is None:
        path_lower = caltable_path.lower()
        if "kcal" in path_lower or "_k." in path_lower:
            cal_type = "k"
        elif "bpcal" in path_lower or "_bp." in path_lower:
            cal_type = "bp"
        elif "gcal" in path_lower or "_g." in path_lower or "gpcal" in path_lower:
            cal_type = "g"
        else:
            cal_type = "unknown"

    metrics = CalibrationMetrics(caltable_path=caltable_path, cal_type=cal_type)

    if not Path(caltable_path).exists():
        metrics.extraction_error = f"Calibration table not found: {caltable_path}"
        metrics.extraction_time_s = time.time() - start_time
        return metrics

    try:
        tb = _open_caltable(caltable_path)

        try:
            # Get column names
            colnames = tb.colnames()

            # Get gains (CPARAM for complex gains, or FPARAM for float gains like K)
            if "CPARAM" in colnames:
                gains = tb.getcol("CPARAM")
                is_complex = True
            elif "FPARAM" in colnames:
                gains = tb.getcol("FPARAM")
                is_complex = False
            else:
                metrics.extraction_error = "No CPARAM or FPARAM column found"
                return metrics

            # Get flags
            flags = tb.getcol("FLAG") if "FLAG" in colnames else np.zeros_like(gains, dtype=bool)

            # Get SNR if available
            snr = tb.getcol("SNR") if "SNR" in colnames else None

            # Get metadata
            if "ANTENNA1" in colnames:
                antennas = tb.getcol("ANTENNA1")
                metrics.n_antennas = len(np.unique(antennas))

            if "SPECTRAL_WINDOW_ID" in colnames:
                spw_ids = tb.getcol("SPECTRAL_WINDOW_ID")
                metrics.n_spws = len(np.unique(spw_ids))

            # Calculate solution statistics
            metrics.n_solutions = int(gains.size)
            metrics.n_flagged = int(np.sum(flags))
            metrics.flag_fraction = (
                float(metrics.n_flagged / metrics.n_solutions) if metrics.n_solutions > 0 else 0.0
            )

            # Get unflagged data for statistics
            valid_mask = ~flags
            if np.any(valid_mask):
                valid_gains = gains[valid_mask]

                if is_complex:
                    # Complex gains: compute amplitude and phase
                    amplitudes = np.abs(valid_gains)
                    phases_deg = np.angle(valid_gains, deg=True)

                    metrics.mean_amplitude = float(np.mean(amplitudes))
                    metrics.std_amplitude = float(np.std(amplitudes))
                    metrics.median_amplitude = float(np.median(amplitudes))
                    metrics.min_amplitude = float(np.min(amplitudes))
                    metrics.max_amplitude = float(np.max(amplitudes))
                    metrics.median_phase_deg = float(np.median(phases_deg))
                    metrics.phase_scatter_deg = float(np.std(phases_deg))
                else:
                    # Float gains (e.g., delays): treat as amplitudes
                    metrics.mean_amplitude = float(np.mean(valid_gains))
                    metrics.std_amplitude = float(np.std(valid_gains))
                    metrics.median_amplitude = float(np.median(valid_gains))
                    metrics.min_amplitude = float(np.min(valid_gains))
                    metrics.max_amplitude = float(np.max(valid_gains))

                # Number of channels (infer from shape)
                if len(gains.shape) >= 2:
                    metrics.n_channels = gains.shape[1] if gains.shape[1] > 1 else 1

                # SNR statistics
                if snr is not None:
                    valid_snr = snr[valid_mask]
                    if len(valid_snr) > 0:
                        metrics.median_snr = float(np.median(valid_snr))
                        metrics.min_snr = float(np.min(valid_snr))
                        metrics.max_snr = float(np.max(valid_snr))

        finally:
            tb.close()

    except (OSError, RuntimeError, ValueError, KeyError) as exc:
        metrics.extraction_error = str(exc)
        logger.warning(f"Failed to extract metrics from {caltable_path}: {exc}")

    metrics.extraction_time_s = time.time() - start_time
    return metrics


def assess_calibration_quality(
    ms_path: str,
    thresholds: QAThresholds | None = None,
    *,
    caltables: dict[str, str | None] | None = None,
) -> CalibrationQAResult:
    """Assess quality of calibration tables for an MS.

        Performs comprehensive quality assessment including:
        - SNR checks (minimum acceptable)
        - Flagging fraction checks (maximum acceptable)
        - Amplitude range checks (physical reasonableness)
        - Phase scatter checks (stability)

    Parameters
    ----------
    ms_path : str
        Path to the Measurement Set.
    thresholds : QAThresholds, optional
        QA thresholds (uses defaults if None).
    caltables : dict, optional
        Pre-discovered caltables dict. If None, discovers them.

    Returns
    -------
        dict
        Quality assessment results.
    """
    start_time = time.time()

    if thresholds is None:
        thresholds = QAThresholds()

    result = CalibrationQAResult(
        ms_path=ms_path,
        passed=True,
        severity="success",
        overall_grade="unknown",
    )

    # Discover caltables if not provided
    if caltables is None:
        from dsa110_contimg.core.calibration.caltables import discover_caltables

        caltables = discover_caltables(ms_path)

    # Check for required tables
    if not caltables.get("bp") and not caltables.get("g"):
        result.passed = False
        result.severity = "error"
        result.overall_grade = "failed"
        result.issues.append(
            QAIssue(
                severity="error",
                cal_type="all",
                metric="presence",
                value=0.0,
                threshold=1.0,
                message="Missing required calibration tables (BP and/or G)",
            )
        )
        result.summary = {
            "n_tables": 0,
            "n_warnings": 0,
            "n_errors": 1,
        }
        result.assessment_time_s = time.time() - start_time
        return result

    # Compute metrics for each table
    n_errors = 0
    n_warnings = 0
    total_flag_fraction = 0.0
    n_tables_with_flags = 0

    for cal_type, cal_path in caltables.items():
        if not cal_path:
            continue

        metrics = compute_calibration_metrics(cal_path, cal_type)
        result.metrics.append(metrics)

        if not metrics.is_valid:
            result.issues.append(
                QAIssue(
                    severity="error",
                    cal_type=cal_type,
                    metric="extraction",
                    value=0.0,
                    threshold=1.0,
                    message=f"{cal_type.upper()} table: {metrics.extraction_error}",
                )
            )
            n_errors += 1
            continue

        # Track flagging for overall grade
        total_flag_fraction += metrics.flag_fraction
        n_tables_with_flags += 1

        # Check SNR (if available)
        if metrics.median_snr is not None and metrics.median_snr < thresholds.min_snr:
            result.issues.append(
                QAIssue(
                    severity="error",
                    cal_type=cal_type,
                    metric="snr",
                    value=metrics.median_snr,
                    threshold=thresholds.min_snr,
                    message=f"{cal_type.upper()} table: Low SNR ({metrics.median_snr:.1f} < {thresholds.min_snr})",
                )
            )
            n_errors += 1

        # Check flagging fraction
        if metrics.flag_fraction > thresholds.max_flag_fraction:
            severity = "error" if metrics.flag_fraction > 0.5 else "warning"
            result.issues.append(
                QAIssue(
                    severity=severity,
                    cal_type=cal_type,
                    metric="flagging",
                    value=metrics.flag_fraction,
                    threshold=thresholds.max_flag_fraction,
                    message=f"{cal_type.upper()} table: High flagging ({metrics.flag_fraction:.1%} > {thresholds.max_flag_fraction:.1%})",
                )
            )
            if severity == "error":
                n_errors += 1
            else:
                n_warnings += 1

        # Check amplitude range (only for complex gain tables)
        if cal_type in ("bp", "g"):
            if metrics.mean_amplitude < thresholds.min_amplitude:
                result.issues.append(
                    QAIssue(
                        severity="warning",
                        cal_type=cal_type,
                        metric="amplitude_low",
                        value=metrics.mean_amplitude,
                        threshold=thresholds.min_amplitude,
                        message=f"{cal_type.upper()} table: Low amplitude ({metrics.mean_amplitude:.3f} < {thresholds.min_amplitude})",
                    )
                )
                n_warnings += 1

            if metrics.max_amplitude > thresholds.max_amplitude:
                result.issues.append(
                    QAIssue(
                        severity="warning",
                        cal_type=cal_type,
                        metric="amplitude_high",
                        value=metrics.max_amplitude,
                        threshold=thresholds.max_amplitude,
                        message=f"{cal_type.upper()} table: High amplitude ({metrics.max_amplitude:.3f} > {thresholds.max_amplitude})",
                    )
                )
                n_warnings += 1

        # Check phase scatter
        if metrics.phase_scatter_deg > thresholds.max_phase_scatter_deg:
            result.issues.append(
                QAIssue(
                    severity="warning",
                    cal_type=cal_type,
                    metric="phase_scatter",
                    value=metrics.phase_scatter_deg,
                    threshold=thresholds.max_phase_scatter_deg,
                    message=f"{cal_type.upper()} table: High phase scatter ({metrics.phase_scatter_deg:.1f}° > {thresholds.max_phase_scatter_deg}°)",
                )
            )
            n_warnings += 1

        # Check for non-finite values
        if not np.isfinite(metrics.mean_amplitude):
            result.issues.append(
                QAIssue(
                    severity="error",
                    cal_type=cal_type,
                    metric="finite",
                    value=float("nan"),
                    threshold=0.0,
                    message=f"{cal_type.upper()} table: Non-finite amplitudes detected",
                )
            )
            n_errors += 1

    # Determine pass/fail
    result.passed = n_errors == 0
    result.severity = "error" if n_errors > 0 else ("warning" if n_warnings > 0 else "success")

    # Calculate overall grade based on average flagging
    avg_flag_fraction = (
        total_flag_fraction / n_tables_with_flags if n_tables_with_flags > 0 else 1.0
    )
    if not result.passed:
        result.overall_grade = "failed"
    elif avg_flag_fraction < 0.1:
        result.overall_grade = "excellent"
    elif avg_flag_fraction < 0.2:
        result.overall_grade = "good"
    elif avg_flag_fraction < 0.3:
        result.overall_grade = "marginal"
    else:
        result.overall_grade = "poor"

    result.summary = {
        "n_tables": len(result.metrics),
        "n_valid_tables": sum(1 for m in result.metrics if m.is_valid),
        "n_warnings": n_warnings,
        "n_errors": n_errors,
        "avg_flag_fraction": avg_flag_fraction,
    }

    result.assessment_time_s = time.time() - start_time
    return result


class CalibrationQAStore:
    """Persistent storage for calibration QA results.

    Stores QA results in SQLite database alongside pipeline state.

    """

    TABLE_NAME = "calibration_qa"

    def __init__(self, db_path: str | None = None):
        """Initialize QA store.

        Parameters
        ----------
        db_path : str, optional
            Path to SQLite database. Uses pipeline.sqlite3 if None.
        """
        if db_path is None:
            from dsa110_contimg.infrastructure.database.session import get_db_path

            db_path = get_db_path("pipeline")

        self.db_path = db_path
        self.table_name = self.TABLE_NAME
        self._init_db()

    def _init_db(self) -> None:
        """Initialize database schema.

        Creates the canonical calibration_qa table.

        """
        with sqlite3.connect(self.db_path, timeout=30.0) as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self.table_name} (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ms_path TEXT NOT NULL,
                    passed BOOLEAN NOT NULL,
                    severity TEXT NOT NULL,
                    overall_grade TEXT NOT NULL,
                    n_warnings INTEGER DEFAULT 0,
                    n_errors INTEGER DEFAULT 0,
                    avg_flag_fraction REAL,
                    assessment_time_s REAL,
                    result_json TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    UNIQUE(ms_path, created_at)
                )
                """
            )
            conn.execute(
                f"""
                CREATE INDEX IF NOT EXISTS idx_{self.table_name}_ms_path
                ON {self.table_name}(ms_path)
                """
            )
            conn.execute(
                f"""
                CREATE INDEX IF NOT EXISTS idx_{self.table_name}_created_at
                ON {self.table_name}(created_at DESC)
                """
            )

            conn.commit()

    def save_result(self, result: CalibrationQAResult) -> int:
        """Save a QA result to the database.

        Parameters
        ----------
        result : CalibrationQAResult
            Calibration QA result to save.

        Returns
        -------
            None
        """
        with sqlite3.connect(self.db_path, timeout=30.0) as conn:
            cursor = conn.execute(
                f"""
                INSERT INTO {self.table_name} (
                    ms_path, passed, severity, overall_grade,
                    n_warnings, n_errors, avg_flag_fraction,
                    assessment_time_s, result_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    result.ms_path,
                    result.passed,
                    result.severity,
                    result.overall_grade,
                    result.summary.get("n_warnings", 0),
                    result.summary.get("n_errors", 0),
                    result.summary.get("avg_flag_fraction"),
                    result.assessment_time_s,
                    json.dumps(result.to_dict()),
                    result.timestamp,
                ),
            )
            conn.commit()
            return cursor.lastrowid

    def get_result(self, ms_path: str) -> CalibrationQAResult | None:
        """Get the most recent QA result for an MS.

        Parameters
        ----------
        ms_path : str
            Path to the Measurement Set.

        Returns
        -------
            None
        """
        with sqlite3.connect(self.db_path, timeout=30.0) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                f"""
                SELECT result_json FROM {self.table_name}
                WHERE ms_path = ?
                ORDER BY created_at DESC
                LIMIT 1
                """,
                (ms_path,),
            ).fetchone()

            if row:
                return CalibrationQAResult.from_dict(json.loads(row["result_json"]))
            return None

    def list_recent(
        self,
        limit: int = 10,
        *,
        passed_only: bool = False,
        failed_only: bool = False,
        min_grade: str | None = None,
    ) -> list[CalibrationQAResult]:
        """List recent QA results.

        Parameters
        ----------
        limit : int, optional
            Maximum number of results (default is 10).
        passed_only : bool, optional
            Only return passed results (default is False).
        failed_only : bool, optional
            Only return failed results (default is False).
        min_grade : Optional[str], optional
            Minimum grade to include ('excellent', 'good', 'marginal', 'poor') (default is None).
        """
        conditions = []
        params: list[Any] = []

        if passed_only:
            conditions.append("passed = 1")
        elif failed_only:
            conditions.append("passed = 0")

        if min_grade:
            grade_order = {"excellent": 1, "good": 2, "marginal": 3, "poor": 4, "failed": 5}
            min_order = grade_order.get(min_grade, 5)
            valid_grades = [g for g, o in grade_order.items() if o <= min_order]
            placeholders = ",".join("?" * len(valid_grades))
            conditions.append(f"overall_grade IN ({placeholders})")
            params.extend(valid_grades)

        where_clause = " AND ".join(conditions) if conditions else "1=1"

        with sqlite3.connect(self.db_path, timeout=30.0) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                f"""
                SELECT result_json FROM {self.table_name}
                WHERE {where_clause}
                ORDER BY created_at DESC
                LIMIT ?
                """,
                params + [limit],
            ).fetchall()

            return [CalibrationQAResult.from_dict(json.loads(row["result_json"])) for row in rows]

    def get_summary_stats(self, since_timestamp: float | None = None) -> dict[str, Any]:
        """Get summary statistics for QA results.

        Parameters
        ----------
        since_timestamp :
            Only include results after this timestamp
        since_timestamp : Optional[float] :
            (Default value = None)
        since_timestamp: Optional[float] :
             (Default value = None)

        """
        conditions = []
        params: list[Any] = []

        if since_timestamp:
            conditions.append("created_at >= ?")
            params.append(since_timestamp)

        where_clause = " AND ".join(conditions) if conditions else "1=1"

        with sqlite3.connect(self.db_path, timeout=30.0) as conn:
            conn.row_factory = sqlite3.Row

            # Get counts by grade
            grade_counts = conn.execute(
                f"""
                SELECT overall_grade, COUNT(*) as count
                FROM {self.table_name}
                WHERE {where_clause}
                GROUP BY overall_grade
                """,
                params,
            ).fetchall()

            # Get pass/fail counts
            pass_counts = conn.execute(
                f"""
                SELECT passed, COUNT(*) as count
                FROM {self.table_name}
                WHERE {where_clause}
                GROUP BY passed
                """,
                params,
            ).fetchall()

            # Get average flagging
            avg_stats = conn.execute(
                f"""
                SELECT
                    AVG(avg_flag_fraction) as avg_flagging,
                    AVG(assessment_time_s) as avg_time,
                    COUNT(*) as total
                FROM {self.table_name}
                WHERE {where_clause}
                """,
                params,
            ).fetchone()

            return {
                "by_grade": {row["overall_grade"]: row["count"] for row in grade_counts},
                "passed": sum(row["count"] for row in pass_counts if row["passed"]),
                "failed": sum(row["count"] for row in pass_counts if not row["passed"]),
                "total": avg_stats["total"] if avg_stats else 0,
                "avg_flagging": avg_stats["avg_flagging"] if avg_stats else None,
                "avg_assessment_time_s": avg_stats["avg_time"] if avg_stats else None,
            }

    def cleanup_old_results(self, days: int = 30) -> int:
        """Remove QA results older than specified days.

        Parameters
        ----------
        days :
            Remove results older than this many days
        days : int :
            (Default value = 30)
        """
        cutoff = time.time() - (days * 24 * 3600)

        with sqlite3.connect(self.db_path, timeout=30.0) as conn:
            cursor = conn.execute(
                f"DELETE FROM {self.table_name} WHERE created_at < ?",
                (cutoff,),
            )
            conn.commit()
            return cursor.rowcount


# Module-level singleton for QA store
_qa_store: CalibrationQAStore | None = None


def get_qa_store(db_path: str | None = None) -> CalibrationQAStore:
    """Get or create the singleton QA store.

    Parameters
    ----------
    db_path :
        Database path (only used on first call)
    db_path : Optional[str] :
        (Default value = None)
    db_path: Optional[str] :
         (Default value = None)

    """
    global _qa_store
    if _qa_store is None:
        _qa_store = CalibrationQAStore(db_path)
    return _qa_store


def close_qa_store() -> None:
    """Close the singleton QA store."""
    global _qa_store
    _qa_store = None
