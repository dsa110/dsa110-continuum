"""
Evaluation database module for isolated evaluation data storage.

This module provides a separate SQLite database for evaluation results,
reference data, and run history - completely isolated from the production
pipeline database.

Database: state/db/evaluation.sqlite3

Tables:
    - evaluation_runs: Run metadata and stage pass rate summaries
    - stage_results: Per-stage evaluation results
    - metric_results: Individual metric measurements
    - reference_baselines: Golden reference dataset baselines
    - baselines: Stored baseline results for regression detection
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

from .stage_evaluators import StageEvaluationResult
from .stages import PipelineStage
from dsa110_contimg.common.utils import get_env_path

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

_DEFAULT_DB_PATH = (
    get_env_path("CONTIMG_BASE_DIR", default="/data/dsa110-contimg") / "state/db/evaluation.sqlite3"
)


# =============================================================================
# Schema Definition
# =============================================================================

_SCHEMA = """
-- Evaluation runs table: one row per evaluation execution
CREATE TABLE IF NOT EXISTS evaluation_runs (
    run_id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_timestamp TEXT NOT NULL,
    dataset_name TEXT,
    dataset_path TEXT,
    num_samples INTEGER DEFAULT 0,
    num_stages_evaluated INTEGER DEFAULT 0,
    overall_score REAL,  -- Legacy pass-rate field (overall stage pass rate)
    quality_grade TEXT,
    passed INTEGER DEFAULT 0,  -- 0=failed, 1=passed
    duration_seconds REAL,
    config_baseline TEXT,  -- JSON of thresholds used
    notes TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

-- Stage results: per-stage scores for each run
CREATE TABLE IF NOT EXISTS stage_results (
    result_id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL,
    stage TEXT NOT NULL,
    composite_score REAL,  -- Legacy pass-rate field (per-stage check pass rate)
    quality_grade TEXT,
    passed INTEGER DEFAULT 0,
    num_metrics INTEGER DEFAULT 0,
    errors TEXT,  -- JSON array
    warnings TEXT,  -- JSON array
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (run_id) REFERENCES evaluation_runs(run_id) ON DELETE CASCADE
);

-- Metric results: individual metric measurements
CREATE TABLE IF NOT EXISTS metric_results (
    metric_id INTEGER PRIMARY KEY AUTOINCREMENT,
    result_id INTEGER NOT NULL,
    run_id INTEGER NOT NULL,
    stage TEXT NOT NULL,
    metric_name TEXT NOT NULL,
    metric_value REAL,
    score REAL,
    weight REAL,
    passed INTEGER DEFAULT 0,
    unit TEXT,
    message TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (result_id) REFERENCES stage_results(result_id) ON DELETE CASCADE,
    FOREIGN KEY (run_id) REFERENCES evaluation_runs(run_id) ON DELETE CASCADE
);

-- Reference baselines: golden reference data for comparison
CREATE TABLE IF NOT EXISTS reference_baselines (
    baseline_id INTEGER PRIMARY KEY AUTOINCREMENT,
    baseline_name TEXT NOT NULL UNIQUE,
    preset_name TEXT,  -- From reference_datasets.yaml
    source_path TEXT,
    capture_timestamp TEXT NOT NULL,
    stage TEXT NOT NULL,
    metrics_json TEXT NOT NULL,  -- JSON of metric name -> value
    metadata_json TEXT,  -- Additional context
    is_active INTEGER DEFAULT 1,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

-- Baselines: stored evaluation results for regression detection
CREATE TABLE IF NOT EXISTS baselines (
    baseline_id INTEGER PRIMARY KEY AUTOINCREMENT,
    baseline_name TEXT NOT NULL UNIQUE,
    description TEXT,
    run_id INTEGER,  -- Link to the evaluation run used as baseline
    stage TEXT NOT NULL,
    composite_score REAL NOT NULL,  -- Legacy pass-rate field for baselines
    quality_grade TEXT,
    metrics_json TEXT NOT NULL,  -- JSON of metric scores
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (run_id) REFERENCES evaluation_runs(run_id) ON DELETE SET NULL
);

-- Indexes for common queries
CREATE INDEX IF NOT EXISTS idx_stage_results_run ON stage_results(run_id);
CREATE INDEX IF NOT EXISTS idx_stage_results_stage ON stage_results(stage);
CREATE INDEX IF NOT EXISTS idx_metric_results_run ON metric_results(run_id);
CREATE INDEX IF NOT EXISTS idx_metric_results_stage ON metric_results(stage);
CREATE INDEX IF NOT EXISTS idx_reference_baselines_stage ON reference_baselines(stage);
CREATE INDEX IF NOT EXISTS idx_baselines_stage ON baselines(stage);
"""


# =============================================================================
# Database Connection
# =============================================================================


class EvaluationDatabase:
    """Database interface for evaluation results storage.

    This class provides methods to:
    - Store evaluation run results
    - Record per-stage and per-metric scores
    - Manage reference baselines
    - Store and compare against baselines

    """

    def __init__(self, db_path: Path | None = None):
        """Initialize database connection.

        Parameters
        ----------
        db_path : Path
            Path to SQLite database file. Defaults to state/db/evaluation.sqlite3
        """
        self.db_path = db_path or _DEFAULT_DB_PATH
        self._ensure_directory()
        self._initialize_schema()

    def _ensure_directory(self) -> None:
        """Ensure the database directory exists."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    def _initialize_schema(self) -> None:
        """Create database tables if they don't exist."""
        with self._connect() as conn:
            conn.executescript(_SCHEMA)
            conn.commit()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        """Context manager for database connections."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    # =========================================================================
    # Evaluation Run Operations
    # =========================================================================

    def start_run(
        self,
        dataset_name: str | None = None,
        dataset_path: str | None = None,
        config_baseline: dict[str, Any] | None = None,
        notes: str | None = None,
    ) -> int:
        """Start a new evaluation run and return its ID.

        Parameters
        ----------
        dataset_name : Optional[str]
            Name of the evaluation dataset (default is None)
        dataset_path : Optional[str]
            Path to the dataset file (default is None)
        config_baseline : Optional[Dict[str, Any]]
            Threshold configuration used (default is None)
        notes : Optional[str]
            Optional notes about the run (default is None)

        """
        with self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO evaluation_runs
                    (run_timestamp, dataset_name, dataset_path, config_baseline, notes)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    datetime.utcnow().isoformat(),
                    dataset_name,
                    dataset_path,
                    json.dumps(config_baseline) if config_baseline else None,
                    notes,
                ),
            )
            conn.commit()
            return cursor.lastrowid

    def complete_run(
        self,
        run_id: int,
        overall_score: float,
        quality_grade: str,
        passed: bool,
        num_samples: int = 0,
        num_stages: int = 0,
        duration_seconds: float = 0.0,
    ) -> None:
        """Update run with final results.

        Parameters
        ----------
        run_id : int
            The evaluation run ID
        overall_score : float
            Overall stage pass rate (legacy field)
        quality_grade : str
            Quality grade (excellent/good/etc)
        passed : bool
            Whether the evaluation passed
        num_samples : int
            Number of samples evaluated (default is 0)
        num_stages : int
            Number of stages evaluated (default is 0)
        duration_seconds : float
            Total evaluation duration (default is 0.0)

        """
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE evaluation_runs SET
                    overall_score = ?,
                    quality_grade = ?,
                    passed = ?,
                    num_samples = ?,
                    num_stages_evaluated = ?,
                    duration_seconds = ?
                WHERE run_id = ?
                """,
                (
                    overall_score,
                    quality_grade,
                    1 if passed else 0,
                    num_samples,
                    num_stages,
                    duration_seconds,
                    run_id,
                ),
            )
            conn.commit()

    def get_run(self, run_id: int) -> dict[str, Any] | None:
        """Get evaluation run by ID.

        Parameters
        ----------
        run_id : int
            Evaluation run ID

        """
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM evaluation_runs WHERE run_id = ?",
                (run_id,),
            ).fetchone()
            return dict(row) if row else None

    def get_recent_runs(self, limit: int = 10) -> list[dict[str, Any]]:
        """Get most recent evaluation runs.

        Parameters
        ----------
        limit : int :
            (Default value = 10)
        """
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM evaluation_runs
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
            return [dict(row) for row in rows]

    # =========================================================================
    # Stage Result Operations
    # =========================================================================

    def store_stage_result(
        self,
        run_id: int,
        result: StageEvaluationResult,
    ) -> int:
        """Store a stage evaluation result.

        Parameters
        ----------
        run_id : int
            The evaluation run ID
        result : StageEvaluationResult
            StageEvaluationResult to store

        """
        # Compute pass rate for legacy composite_score field
        num_checks = len(result.checks)
        pass_rate = result.num_passed / num_checks if num_checks > 0 else 0.0

        # Map pass rate to quality grade for legacy field
        if pass_rate >= 0.95:
            quality_grade = "excellent"
        elif pass_rate >= 0.85:
            quality_grade = "good"
        elif pass_rate >= 0.70:
            quality_grade = "acceptable"
        elif pass_rate >= 0.50:
            quality_grade = "poor"
        else:
            quality_grade = "failed"

        with self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO stage_results
                    (run_id, stage, composite_score, quality_grade, passed,
                     num_metrics, errors, warnings)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    result.stage.value,
                    pass_rate,  # Store pass rate in composite_score field
                    quality_grade,
                    1 if result.passed else 0,
                    len(result.checks),
                    json.dumps(result.errors),
                    json.dumps(result.warnings),
                ),
            )
            result_id = cursor.lastrowid

            # Store individual metric checks
            for check in result.checks:
                conn.execute(
                    """
                    INSERT INTO metric_results
                        (result_id, run_id, stage, metric_name, metric_value,
                         score, weight, passed, unit, message)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        result_id,
                        run_id,
                        result.stage.value,
                        check.name,
                        check.value,
                        1.0 if check.passed else 0.0,  # Store pass as score
                        1.0 if check.required else 0.5,  # Store required as weight
                        1 if check.passed else 0,
                        check.unit,
                        check.message,
                    ),
                )

            conn.commit()
            return result_id

    def get_stage_results(self, run_id: int) -> list[dict[str, Any]]:
        """Get all stage results for a run.

        Parameters
        ----------
        run_id : int
            Evaluation run ID

        """
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM stage_results
                WHERE run_id = ?
                ORDER BY result_id
                """,
                (run_id,),
            ).fetchall()
            return [dict(row) for row in rows]

    def get_metric_results(
        self,
        run_id: int,
        stage: str | None = None,
    ) -> list[dict[str, Any]]:
        """Get metric results for a run, optionally filtered by stage.

        Parameters
        ----------
        run_id : int
            Evaluation run ID
        stage : Optional[str]
            Stage to filter by (default is None)

        """
        with self._connect() as conn:
            if stage:
                rows = conn.execute(
                    """
                    SELECT * FROM metric_results
                    WHERE run_id = ? AND stage = ?
                    ORDER BY metric_id
                    """,
                    (run_id, stage),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT * FROM metric_results
                    WHERE run_id = ?
                    ORDER BY metric_id
                    """,
                    (run_id,),
                ).fetchall()
            return [dict(row) for row in rows]

    # =========================================================================
    # Reference Snapshot Operations
    # =========================================================================

    def store_reference_baseline(
        self,
        baseline_name: str,
        stage: PipelineStage,
        metrics: dict[str, float],
        preset_name: str | None = None,
        source_path: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> int:
        """Store a reference baseline for regression comparison.

        Parameters
        ----------
        baseline_name : str
            Unique identifier for this baseline
        stage : PipelineStage
            Pipeline stage the baseline represents
        metrics : Dict[str, float]
            Dictionary of metric name to expected value
        preset_name : Optional[str]
            Reference preset from reference_datasets.yaml (default is None)
        source_path : Optional[str]
            Path to source data used (default is None)
        metadata : Optional[Dict[str, Any]]
            Additional context (default is None)

        """
        with self._connect() as conn:
            # Delete existing baseline with same name (allows overwrite)
            conn.execute(
                "DELETE FROM reference_baselines WHERE baseline_name = ?",
                (baseline_name,),
            )

            cursor = conn.execute(
                """
                INSERT INTO reference_baselines
                    (baseline_name, preset_name, source_path, capture_timestamp,
                     stage, metrics_json, metadata_json)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    baseline_name,
                    preset_name,
                    source_path,
                    datetime.utcnow().isoformat(),
                    stage.value,
                    json.dumps(metrics),
                    json.dumps(metadata) if metadata else None,
                ),
            )
            conn.commit()
            return cursor.lastrowid

    def get_reference_baseline(
        self,
        baseline_name: str,
    ) -> dict[str, Any] | None:
        """Get an active reference baseline by name.

        Parameters
        ----------
        baseline_name : str
            Name of the reference baseline

        """
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT * FROM reference_baselines
                WHERE baseline_name = ? AND is_active = 1
                """,
                (baseline_name,),
            ).fetchone()
            if row:
                result = dict(row)
                result["metrics"] = json.loads(result["metrics_json"])
                return result
            return None

    def get_reference_baselines_for_stage(
        self,
        stage: PipelineStage,
    ) -> list[dict[str, Any]]:
        """Get all active reference baselines for a stage.

        Parameters
        ----------
        stage : PipelineStage
            Pipeline stage to get baselines for

        """
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM reference_baselines
                WHERE stage = ? AND is_active = 1
                ORDER BY capture_timestamp DESC
                """,
                (stage.value,),
            ).fetchall()
            results = []
            for row in rows:
                result = dict(row)
                result["metrics"] = json.loads(result["metrics_json"])
                results.append(result)
            return results

    # =========================================================================
    # Baseline Operations
    # =========================================================================

    def store_baseline(
        self,
        baseline_name: str,
        stage: PipelineStage,
        composite_score: float,
        quality_grade: str,
        metrics: dict[str, float],
        description: str | None = None,
        run_id: int | None = None,
    ) -> int:
        """Store a baseline for regression detection.

        Parameters
        ----------
        baseline_name : str
            Unique identifier for this baseline
        stage : PipelineStage
            Pipeline stage
        composite_score : float
            The baseline pass rate (legacy field)
        quality_grade : str
            Quality grade at baseline
        metrics : Dict[str, float]
            Dictionary of metric name to score
        description : Optional[str]
            Description of the baseline (default is None)
        run_id : Optional[int]
            Optional link to evaluation run (default is None)

        """
        with self._connect() as conn:
            # Delete existing baseline with same name
            conn.execute(
                "DELETE FROM baselines WHERE baseline_name = ?",
                (baseline_name,),
            )

            cursor = conn.execute(
                """
                INSERT INTO baselines
                    (baseline_name, description, run_id, stage,
                     composite_score, quality_grade, metrics_json)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    baseline_name,
                    description,
                    run_id,
                    stage.value,
                    composite_score,
                    quality_grade,
                    json.dumps(metrics),
                ),
            )
            conn.commit()
            return cursor.lastrowid

    def get_baseline(self, baseline_name: str) -> dict[str, Any] | None:
        """Get a baseline by name.

        Parameters
        ----------
        baseline_name : str
            Name of the baseline

        """
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM baselines WHERE baseline_name = ?",
                (baseline_name,),
            ).fetchone()
            if row:
                result = dict(row)
                result["metrics"] = json.loads(result["metrics_json"])
                return result
            return None

    def get_baselines_for_stage(
        self,
        stage: PipelineStage,
    ) -> list[dict[str, Any]]:
        """Get all baselines for a stage.

        Parameters
        ----------
        stage : PipelineStage
            Pipeline stage to get baselines for

        """
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM baselines
                WHERE stage = ?
                ORDER BY created_at DESC
                """,
                (stage.value,),
            ).fetchall()
            results = []
            for row in rows:
                result = dict(row)
                result["metrics"] = json.loads(result["metrics_json"])
                results.append(result)
            return results

    def compare_to_baseline(
        self,
        baseline_name: str,
        current_result: StageEvaluationResult,
        tolerance: float = 0.05,
    ) -> dict[str, Any]:
        """Compare current results against a stored baseline.

        Parameters
        ----------
        baseline_name : str
            Name of baseline to compare against
        current_result : StageEvaluationResult
            Current evaluation result
        tolerance : float
            Acceptable pass rate degradation (default is 0.05)

        """
        baseline = self.get_baseline(baseline_name)
        if not baseline:
            return {
                "comparison": "no_baseline",
                "message": f"No baseline found: {baseline_name}",
            }

        # Compute current pass rate
        num_checks = len(current_result.checks)
        current_pass_rate = current_result.num_passed / num_checks if num_checks > 0 else 0.0

        # Map pass rate to quality grade
        if current_pass_rate >= 0.95:
            current_grade = "excellent"
        elif current_pass_rate >= 0.85:
            current_grade = "good"
        elif current_pass_rate >= 0.70:
            current_grade = "acceptable"
        elif current_pass_rate >= 0.50:
            current_grade = "poor"
        else:
            current_grade = "failed"

        # Compare pass rates
        score_delta = current_pass_rate - baseline["composite_score"]
        is_regression = score_delta < -tolerance

        # Compare individual metric checks
        metric_comparisons = []
        baseline_metrics = baseline["metrics"]
        for check in current_result.checks:
            if check.name in baseline_metrics:
                baseline_score = baseline_metrics[check.name]
                current_score = 1.0 if check.passed else 0.0
                delta = current_score - baseline_score
                metric_comparisons.append(
                    {
                        "name": check.name,
                        "current_passed": check.passed,
                        "current_score": current_score,
                        "baseline_score": baseline_score,
                        "delta": delta,
                        "is_regression": delta < -tolerance,
                    }
                )

        return {
            "comparison": "completed",
            "baseline_name": baseline_name,
            "baseline_score": baseline["composite_score"],
            "current_score": current_pass_rate,
            "score_delta": score_delta,
            "is_regression": is_regression,
            "baseline_grade": baseline["quality_grade"],
            "current_grade": current_grade,
            "metric_comparisons": metric_comparisons,
            "regressed_metrics": [m["name"] for m in metric_comparisons if m["is_regression"]],
        }

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def get_statistics(self) -> dict[str, Any]:
        """Get database statistics."""
        with self._connect() as conn:
            stats = {}

            # Count runs
            row = conn.execute("SELECT COUNT(*) FROM evaluation_runs").fetchone()
            stats["total_runs"] = row[0]

            # Count passed runs
            row = conn.execute("SELECT COUNT(*) FROM evaluation_runs WHERE passed = 1").fetchone()
            stats["passed_runs"] = row[0]

            # Average pass rate
            row = conn.execute("SELECT AVG(overall_score) FROM evaluation_runs").fetchone()
            stats["average_score"] = row[0]

            # Count baselines
            row = conn.execute(
                "SELECT COUNT(*) FROM reference_baselines WHERE is_active = 1"
            ).fetchone()
            stats["active_baselines"] = row[0]

            # Count baselines
            row = conn.execute("SELECT COUNT(*) FROM baselines").fetchone()
            stats["baselines"] = row[0]

            return stats

    def cleanup_old_runs(self, keep_days: int = 30) -> int:
        """Remove evaluation runs older than specified days.

        Parameters
        ----------
        keep_days : int
            Number of days of history to keep (default 30)
        """
        with self._connect() as conn:
            cutoff = datetime.utcnow().isoformat()[:10]  # YYYY-MM-DD
            cursor = conn.execute(
                """
                DELETE FROM evaluation_runs
                WHERE date(created_at) < date(?, '-' || ? || ' days')
                """,
                (cutoff, keep_days),
            )
            conn.commit()
            return cursor.rowcount


# =============================================================================
# Module-Level Instance
# =============================================================================


class _DatabaseHolder:
    """Holder class to avoid global statement."""

    instance: EvaluationDatabase | None = None


def get_evaluation_db(db_path: Path | None = None) -> EvaluationDatabase:
    """Get the evaluation database instance (lazy-loaded singleton).

    Parameters
    ----------
    db_path : Optional[Path]
        Optional path to override default database location (default is None)

    """
    if _DatabaseHolder.instance is None or db_path is not None:
        _DatabaseHolder.instance = EvaluationDatabase(db_path)
    return _DatabaseHolder.instance
