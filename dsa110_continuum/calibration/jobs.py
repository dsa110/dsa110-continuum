"""
Dagster job implementations for calibration pipeline.

Three jobs:
1. CalibrationSolveJob - Auto-detect calibrator, solve K/BP/G tables
2. CalibrationApplyJob - Apply calibration tables to target MS
3. CalibrationValidateJob - Verify calibration quality metrics

These jobs inherit from the generic pipeline.Job base class and can be
orchestrated via CalibrationPipeline.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
import statistics
from dataclasses import dataclass, field
from pathlib import Path

from dsa110_contimg.workflow.pipeline import Job, JobResult, register_job

logger = logging.getLogger(__name__)


@dataclass
class CalibrationJobConfig:
    """Configuration for calibration jobs."""

    database_path: Path
    caltable_dir: Path
    catalog_path: Path | None = None


def _ensure_calibration_tables(conn: sqlite3.Connection) -> None:
    """Ensure calibration tracking tables exist.

    Parameters
    ----------
    conn :
        SQLite database connection
    conn: sqlite3.Connection :


    """
    conn.executescript("""
        -- Calibration solve tracking
        CREATE TABLE IF NOT EXISTS calibration_solves (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ms_path TEXT NOT NULL,
            calibrator_field TEXT,
            refant TEXT,
            status TEXT NOT NULL,
            k_table_path TEXT,
            bp_table_path TEXT,
            g_table_path TEXT,
            created_at REAL NOT NULL,
            completed_at REAL,
            error TEXT,
            metadata_json TEXT
        );

        CREATE INDEX IF NOT EXISTS idx_calibration_solves_ms
            ON calibration_solves(ms_path);
        CREATE INDEX IF NOT EXISTS idx_calibration_solves_status
            ON calibration_solves(status);

        -- Calibration apply tracking
        CREATE TABLE IF NOT EXISTS calibration_applications (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            solve_id INTEGER,
            target_ms_path TEXT NOT NULL,
            target_field TEXT NOT NULL,
            gaintables_json TEXT NOT NULL,
            status TEXT NOT NULL,
            created_at REAL NOT NULL,
            completed_at REAL,
            verified BOOLEAN DEFAULT 0,
            error TEXT,
            FOREIGN KEY(solve_id) REFERENCES calibration_solves(id)
        );

        CREATE INDEX IF NOT EXISTS idx_calibration_applications_target
            ON calibration_applications(target_ms_path);
        CREATE INDEX IF NOT EXISTS idx_calibration_applications_status
            ON calibration_applications(status);
    """)
    conn.commit()


@register_job
@dataclass
class CalibrationSolveJob(Job):
    """Solve calibration for a Measurement Set.

    Auto-detects calibrator field and reference antenna, then solves
    K (delay), BP (bandpass), and G (gain) calibration tables.

    Inputs:
        - ms_path: Path to the MS containing calibrator data
        - cal_field: Calibrator field (optional, auto-detected)
        - refant: Reference antenna (optional, auto-detected)
        - do_k: Whether to solve K (delay) calibration
        - calibrator_name: Expected calibrator name for model lookup (optional)

    Outputs:
        - solve_id: Database record ID
        - k_table_path: Path to K calibration table (if do_k=True)
        - bp_table_path: Path to bandpass table
        - g_table_path: Path to gain table
        - calibrator_name: Auto-detected calibrator name

    """

    job_type: str = "calibration_solve"

    ms_path: str = ""
    cal_field: str | None = None
    refant: str | None = None
    do_k: bool = False
    calibrator_name: str | None = None
    config: CalibrationJobConfig | None = None

    def validate(self) -> tuple[bool, str | None]:
        """Validate job parameters."""
        if not self.config:
            return False, "No configuration provided"
        if not self.ms_path:
            return False, "No MS path provided"
        if not Path(self.ms_path).exists():
            return False, f"MS not found: {self.ms_path}"
        return True, None

    def execute(self) -> JobResult:
        """Execute the calibration solve job."""
        from dsa110_contimg.core.calibration.caltables import discover_caltables
        from dsa110_contimg.core.calibration.streaming import solve_calibration_for_ms

        # Validate first
        is_valid, error = self.validate()
        if not is_valid:
            return JobResult.fail(error or "Validation failed")

        logger.info(
            f"Starting calibration solve for {self.ms_path} "
            f"(field={self.cal_field}, refant={self.refant}, do_k={self.do_k})"
        )

        conn = sqlite3.connect(str(self.config.database_path))
        _ensure_calibration_tables(conn)

        # Insert initial record
        cursor = conn.execute(
            """
            INSERT INTO calibration_solves
                (ms_path, calibrator_field, refant, status, created_at)
            VALUES (?, ?, ?, 'running', ?)
            """,
            (
                self.ms_path,
                self.cal_field,
                self.refant,
                time.time(),
            ),
        )
        solve_id = cursor.lastrowid
        conn.commit()

        try:
            # Solve calibration
            success, error_msg = solve_calibration_for_ms(
                ms_path=self.ms_path,
                cal_field=self.cal_field,
                refant=self.refant,
                do_k=self.do_k,
                catalog_path=str(self.config.catalog_path) if self.config.catalog_path else None,
                calibrator_name=self.calibrator_name,
            )

            if not success:
                conn.execute(
                    """
                    UPDATE calibration_solves
                    SET status = 'failed', completed_at = ?, error = ?
                    WHERE id = ?
                    """,
                    (time.time(), error_msg, solve_id),
                )
                conn.commit()
                conn.close()

                logger.error(f"Calibration solve failed: {error_msg}")
                return JobResult.fail(error_msg or "Calibration solve failed")

            # Discover produced calibration tables
            caltables = discover_caltables(self.ms_path)

            # Update database with table paths
            conn.execute(
                """
                UPDATE calibration_solves
                SET status = 'completed',
                    completed_at = ?,
                    k_table_path = ?,
                    bp_table_path = ?,
                    g_table_path = ?,
                    metadata_json = ?
                WHERE id = ?
                """,
                (
                    time.time(),
                    caltables.get("K"),
                    caltables.get("B"),
                    caltables.get("G"),
                    json.dumps({"caltables": caltables}),
                    solve_id,
                ),
            )
            conn.commit()
            conn.close()

            # Build output paths list
            outputs = {
                "solve_id": solve_id,
                "bp_table_path": caltables.get("B"),
                "g_table_path": caltables.get("G"),
            }
            if self.do_k:
                outputs["k_table_path"] = caltables.get("K")

            logger.info(f"Calibration solve completed for {self.ms_path}")
            return JobResult.ok(
                outputs=outputs,
                message=f"Calibration solved: BP={caltables.get('B')}, G={caltables.get('G')}",
            )

        except Exception as e:
            logger.exception(f"Calibration solve job failed: {e}")
            conn.execute(
                """
                UPDATE calibration_solves
                SET status = 'failed', completed_at = ?, error = ?
                WHERE id = ?
                """,
                (time.time(), str(e), solve_id),
            )
            conn.commit()
            conn.close()

            return JobResult.fail(str(e))


@register_job
@dataclass
class CalibrationApplyJob(Job):
    """Apply calibration tables to a target Measurement Set.

    Applies previously solved calibration tables to a target MS field.
    Validates tables before application and verifies CORRECTED_DATA
    is populated after.

    Inputs:
        - target_ms_path: Path to the target MS
        - target_field: Field to apply calibration to
        - gaintables: List of calibration table paths (or use solve_id)
        - solve_id: Database ID from CalibrationSolveJob (alternative to gaintables)
        - verify: Whether to verify CORRECTED_DATA after application

    Outputs:
        - application_id: Database record ID
        - verified: Whether verification passed

    """

    job_type: str = "calibration_apply"

    target_ms_path: str = ""
    target_field: str = ""
    gaintables: list[str] = field(default_factory=list)
    solve_id: int | None = None
    verify: bool = True
    config: CalibrationJobConfig | None = None

    def validate(self) -> tuple[bool, str | None]:
        """Validate job parameters."""
        if not self.config:
            return False, "No configuration provided"
        if not self.target_ms_path:
            return False, "No target MS path provided"
        if not Path(self.target_ms_path).exists():
            return False, f"Target MS not found: {self.target_ms_path}"
        if not self.target_field:
            return False, "No target field provided"
        if not self.gaintables and not self.solve_id:
            return False, "Either gaintables or solve_id must be provided"
        return True, None

    def execute(self) -> JobResult:
        """Execute the calibration apply job."""
        from dsa110_contimg.core.calibration.applycal import apply_to_target

        # Validate first
        is_valid, error = self.validate()
        if not is_valid:
            return JobResult.fail(error or "Validation failed")

        conn = sqlite3.connect(str(self.config.database_path))
        _ensure_calibration_tables(conn)

        # Resolve gaintables from solve_id if needed
        gaintables = list(self.gaintables)
        if self.solve_id and not gaintables:
            cursor = conn.execute(
                """
                SELECT k_table_path, bp_table_path, g_table_path
                FROM calibration_solves
                WHERE id = ?
                """,
                (self.solve_id,),
            )
            row = cursor.fetchone()
            if not row:
                conn.close()
                return JobResult.fail(f"Solve ID {self.solve_id} not found")

            k_path, bp_path, g_path = row
            if bp_path:
                gaintables.append(bp_path)
            if g_path:
                gaintables.append(g_path)
            if k_path:
                gaintables.insert(0, k_path)  # K table applied first

        if not gaintables:
            conn.close()
            return JobResult.fail("No calibration tables to apply")

        logger.info(
            f"Applying calibration to {self.target_ms_path} field={self.target_field} "
            f"({len(gaintables)} tables)"
        )

        # Insert application record
        cursor = conn.execute(
            """
            INSERT INTO calibration_applications
                (solve_id, target_ms_path, target_field, gaintables_json, status, created_at)
            VALUES (?, ?, ?, ?, 'running', ?)
            """,
            (
                self.solve_id,
                self.target_ms_path,
                self.target_field,
                json.dumps(gaintables),
                time.time(),
            ),
        )
        application_id = cursor.lastrowid
        conn.commit()

        try:
            # Apply calibration
            apply_to_target(
                ms_target=self.target_ms_path,
                field=self.target_field,
                gaintables=gaintables,
                verify=self.verify,
            )

            # Update record on success
            conn.execute(
                """
                UPDATE calibration_applications
                SET status = 'completed', completed_at = ?, verified = ?
                WHERE id = ?
                """,
                (time.time(), 1 if self.verify else 0, application_id),
            )
            conn.commit()
            conn.close()

            logger.info(f"Calibration applied to {self.target_ms_path} (verified={self.verify})")
            return JobResult.ok(
                outputs={
                    "application_id": application_id,
                    "verified": self.verify,
                },
                message=f"Applied {len(gaintables)} calibration tables to {self.target_field}",
            )

        except Exception as e:
            logger.exception(f"Calibration apply job failed: {e}")
            conn.execute(
                """
                UPDATE calibration_applications
                SET status = 'failed', completed_at = ?, error = ?
                WHERE id = ?
                """,
                (time.time(), str(e), application_id),
            )
            conn.commit()
            conn.close()
            return JobResult.fail(str(e))


@register_job
@dataclass
class CalibrationValidateJob(Job):
    """Validate calibration quality metrics.

    Analyzes calibration tables and MS data to verify calibration quality.
    Checks for:
    - Antenna health (flagging fraction, gain scatter)
    - Solution convergence
    - Phase/amplitude stability

    Inputs:
        - ms_path: Path to calibrated MS
        - solve_id: Database ID from CalibrationSolveJob
        - thresholds: Quality thresholds (optional)

    Outputs:
        - validation_passed: Boolean overall pass/fail
        - metrics: Dictionary of quality metrics
        - warnings: List of quality warnings

    """

    job_type: str = "calibration_validate"

    ms_path: str = ""
    solve_id: int | None = None
    flag_threshold: float = 0.3  # Max fraction of flagged solutions
    gain_scatter_threshold: float = 0.2  # Max gain amplitude scatter
    config: CalibrationJobConfig | None = None

    def validate(self) -> tuple[bool, str | None]:
        """Validate job parameters."""
        if not self.config:
            return False, "No configuration provided"
        if not self.ms_path:
            return False, "No MS path provided"
        return True, None

    def execute(self) -> JobResult:
        """Execute the calibration validation job."""
        from dsa110_contimg.core.calibration.refant_selection import (
            analyze_antenna_health_from_caltable,
        )

        # Validate first
        is_valid, error = self.validate()
        if not is_valid:
            return JobResult.fail(error or "Validation failed")

        logger.info(f"Validating calibration for {self.ms_path}")

        conn = sqlite3.connect(str(self.config.database_path))
        _ensure_calibration_tables(conn)

        # Get calibration table paths
        caltable_paths = []
        if self.solve_id:
            cursor = conn.execute(
                """
                SELECT k_table_path, bp_table_path, g_table_path
                FROM calibration_solves
                WHERE id = ?
                """,
                (self.solve_id,),
            )
            row = cursor.fetchone()
            if row:
                k_path, bp_path, g_path = row
                if k_path:
                    caltable_paths.append(("K", k_path))
                if bp_path:
                    caltable_paths.append(("BP", bp_path))
                if g_path:
                    caltable_paths.append(("G", g_path))

        if not caltable_paths:
            # Try to discover from MS path
            from dsa110_contimg.core.calibration.caltables import discover_caltables

            discovered = discover_caltables(self.ms_path)
            for cal_type, path in discovered.items():
                if path:
                    caltable_paths.append((cal_type, path))

        conn.close()

        if not caltable_paths:
            return JobResult.fail("No calibration tables found to validate")

        # Analyze each calibration table
        all_metrics = {}
        warnings = []
        validation_passed = True

        for cal_type, caltable_path in caltable_paths:
            if not Path(caltable_path).exists():
                warnings.append(f"{cal_type} table not found: {caltable_path}")
                continue

            try:
                # Analyze antenna health
                antenna_health = analyze_antenna_health_from_caltable(caltable_path)

                # Calculate summary metrics
                flag_fractions = [
                    ant.get("flag_fraction", 0) for ant in antenna_health if "flag_fraction" in ant
                ]
                gain_scatters = [
                    ant.get("amplitude_scatter", 0)
                    for ant in antenna_health
                    if "amplitude_scatter" in ant
                ]

                avg_flag_fraction = (
                    statistics.mean(flag_fractions) if flag_fractions else 0
                )
                max_flag_fraction = max(flag_fractions) if flag_fractions else 0
                avg_gain_scatter = statistics.mean(gain_scatters) if gain_scatters else 0

                metrics = {
                    "n_antennas": len(antenna_health),
                    "avg_flag_fraction": avg_flag_fraction,
                    "max_flag_fraction": max_flag_fraction,
                    "avg_gain_scatter": avg_gain_scatter,
                }
                all_metrics[cal_type] = metrics

                # Check thresholds
                if avg_flag_fraction > self.flag_threshold:
                    warnings.append(f"{cal_type}: High average flagging ({avg_flag_fraction:.1%})")
                    validation_passed = False

                if avg_gain_scatter > self.gain_scatter_threshold:
                    warnings.append(f"{cal_type}: High gain scatter ({avg_gain_scatter:.2f})")
                    validation_passed = False

                logger.info(
                    f"{cal_type} validation: flag={avg_flag_fraction:.1%}, "
                    f"scatter={avg_gain_scatter:.3f}"
                )

            except Exception as e:
                warnings.append(f"Failed to analyze {cal_type} table: {e}")
                logger.warning(f"Failed to analyze {caltable_path}: {e}")

        if not validation_passed:
            logger.warning(f"Calibration validation failed: {warnings}")

        return JobResult.ok(
            outputs={
                "validation_passed": validation_passed,
                "metrics": all_metrics,
                "warnings": warnings,
            },
            message=(
                f"Validation {'PASSED' if validation_passed else 'FAILED'}: "
                f"{len(caltable_paths)} tables, {len(warnings)} warnings"
            ),
        )
