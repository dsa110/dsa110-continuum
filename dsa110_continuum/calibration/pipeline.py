"""
Pipeline definitions for calibration.

Pipeline classes using the generic pipeline framework:
- CalibrationPipeline: Full calibration workflow (solve → apply → validate)
- StreamingCalibrationPipeline: Real-time calibration for incoming data

Job graph:
    CalibrationSolveJob → CalibrationApplyJob → CalibrationValidateJob
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path

from dsa110_contimg.workflow.pipeline import (
    Pipeline,
    register_pipeline,
)

from .jobs import (
    CalibrationApplyJob,
    CalibrationSolveJob,
    CalibrationValidateJob,
)

__all__ = [
    "CalibrationPipelineConfig",
    "CalibrationResult",
    "CalibrationPipeline",
    "StreamingCalibrationPipeline",
    "run_calibration_pipeline",
]

logger = logging.getLogger(__name__)


# =============================================================================
# Pipeline Status Enum
# =============================================================================


class CalibrationStatus(str, Enum):
    """Status of a calibration pipeline execution."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    VALIDATION_FAILED = "validation_failed"


# =============================================================================
# Configuration Dataclasses
# =============================================================================


@dataclass
class CalibrationPipelineConfig:
    """Configuration for calibration pipelines."""

    database_path: Path
    caltable_dir: Path
    catalog_path: Path | None = None
    default_refant: str | None = None
    do_k_calibration: bool = False


@dataclass
class CalibrationResult:
    """Result of a calibration pipeline execution."""

    success: bool
    solve_id: int | None = None
    application_id: int | None = None
    validation_passed: bool | None = None
    gaintables: list[str] = field(default_factory=list)
    message: str = ""
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    execution_id: str | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None


# =============================================================================
# Pipeline Classes
# =============================================================================


@register_pipeline
class CalibrationPipeline(Pipeline):
    """Full calibration pipeline for a Measurement Set.

        Orchestrates the complete calibration workflow:
        1. CalibrationSolveJob - Solve calibration tables from calibrator data
        2. CalibrationApplyJob - Apply tables to target data
        3. CalibrationValidateJob - Validate calibration quality

        Job graph
    ---------
        solve (CalibrationSolveJob)
        └─> apply (CalibrationApplyJob)
        └─> validate (CalibrationValidateJob)

    Examples
    --------
        >>> config = CalibrationPipelineConfig(
        ...     database_path=Path("pipeline.sqlite3"),
        ...     caltable_dir=Path("/data/caltables"),
        ... )
        >>> pipeline = CalibrationPipeline(
        ...     config=config,
        ...     ms_path="/data/obs_20240101.ms",
        ...     target_field="0",
        ... )
        >>> # Execute via PipelineExecutor
        >>> executor = PipelineExecutor(db_path=config.database_path)
        >>> execution_id = await executor.execute(pipeline)
    """

    pipeline_name = "calibration"
    schedule = None  # On-demand

    def __init__(
        self,
        config: CalibrationPipelineConfig | None = None,
        ms_path: str = "",
        cal_field: str | None = None,
        target_field: str = "",
        refant: str | None = None,
        do_k: bool | None = None,
        skip_apply: bool = False,
        skip_validate: bool = False,
        calibrator_name: str | None = None,
    ):
        """Initialize calibration pipeline.

        Parameters
        ----------
        config : object
            Pipeline configuration.
        ms_path : str
            Path to Measurement Set.
        cal_field : optional
            Calibrator field name (auto-detected if None).
        target_field : str
            Target field to apply calibration to.
        refant : optional
            Reference antenna (auto-detected if None).
        do_k : optional
            Whether to solve K calibration (uses config default if None).
        skip_apply : bool, optional
            If True, only solve calibration (no apply step).
        skip_validate : bool, optional
            If True, skip validation step.
        calibrator_name : optional
            Expected calibrator name for model lookup (e.g., "0834+555").
        """
        self.ms_path = ms_path
        self.cal_field = cal_field
        self.target_field = target_field
        self.refant = refant
        self.skip_apply = skip_apply
        self.skip_validate = skip_validate
        self.calibrator_name = calibrator_name

        # Use config default for do_k if not specified
        if do_k is not None:
            self.do_k = do_k
        elif config:
            self.do_k = config.do_k_calibration
        else:
            self.do_k = False

        super().__init__(config)

    def build(self) -> None:
        """Build the calibration pipeline job graph."""
        # Job 1: Solve calibration
        self.add_job(
            CalibrationSolveJob,
            job_id="solve",
            params={
                "ms_path": self.ms_path,
                "cal_field": self.cal_field,
                "refant": self.refant or (self.config.default_refant if self.config else None),
                "do_k": self.do_k,
                "calibrator_name": self.calibrator_name,
            },
        )

        if not self.skip_apply and self.target_field:
            # Job 2: Apply calibration (depends on solve)
            self.add_job(
                CalibrationApplyJob,
                job_id="apply",
                params={
                    "target_ms_path": self.ms_path,
                    "target_field": self.target_field,
                    "solve_id": "${solve.solve_id}",
                    "verify": True,
                },
                dependencies=["solve"],
            )

            if not self.skip_validate:
                # Job 3: Validate calibration (depends on apply)
                self.add_job(
                    CalibrationValidateJob,
                    job_id="validate",
                    params={
                        "ms_path": self.ms_path,
                        "solve_id": "${solve.solve_id}",
                    },
                    dependencies=["apply"],
                )

        elif not self.skip_validate:
            # Validate directly after solve (no apply step)
            self.add_job(
                CalibrationValidateJob,
                job_id="validate",
                params={
                    "ms_path": self.ms_path,
                    "solve_id": "${solve.solve_id}",
                },
                dependencies=["solve"],
            )

        # Configure retry and notifications
        self.set_retry_policy(max_retries=1, backoff="fixed")
        self.add_notification(
            on_failure="solve",
            channels=["slack"],
            recipients=["#dsa110-alerts"],
        )


@register_pipeline
class StreamingCalibrationPipeline(Pipeline):
    """Streaming calibration pipeline for real-time processing.

        Designed for automatic calibration when new calibrator data arrives.
        Only solves calibration (no apply step), storing tables for later use.

        This pipeline can be triggered by:
        - Event: DATA_INGESTED with calibrator detection
        - Scheduler: Periodic check for new calibrator MSes

        Job graph
    ---------
        solve (CalibrationSolveJob)
        └─> validate (CalibrationValidateJob)

    Examples
    --------
        >>> # Triggered when new calibrator MS arrives
        >>> # ... trigger logic ...
    """

    pipeline_name = "streaming_calibration"
    schedule = None  # Event-driven

    def __init__(
        self,
        config: CalibrationPipelineConfig | None = None,
        ms_path: str = "",
        refant: str | None = None,
    ):
        """Initialize streaming calibration pipeline.

        Parameters
        ----------
        config : object
            Pipeline configuration.
        ms_path : str
            Path to Measurement Set with calibrator data.
        refant : optional
            Reference antenna (auto-detected if None).
        """
        self.ms_path = ms_path
        self.refant = refant

        super().__init__(config)

    def build(self) -> None:
        """Build the streaming calibration job graph."""
        # Job 1: Solve calibration (calibrator field auto-detected)
        self.add_job(
            CalibrationSolveJob,
            job_id="solve",
            params={
                "ms_path": self.ms_path,
                "cal_field": None,  # Auto-detect
                "refant": self.refant or (self.config.default_refant if self.config else None),
                "do_k": False,  # K not needed for streaming
            },
        )

        # Job 2: Validate (quick validation for streaming)
        self.add_job(
            CalibrationValidateJob,
            job_id="validate",
            params={
                "ms_path": self.ms_path,
                "solve_id": "${solve.solve_id}",
                "flag_threshold": 0.5,  # More lenient for streaming
                "gain_scatter_threshold": 0.3,
            },
            dependencies=["solve"],
        )

        # Minimal retry for streaming (don't block)
        self.set_retry_policy(max_retries=0)


# =============================================================================
# Helper Functions
# =============================================================================


async def run_calibration_pipeline(
    ms_path: str,
    target_field: str,
    config: CalibrationPipelineConfig,
    cal_field: str | None = None,
    refant: str | None = None,
    do_k: bool = False,
    calibrator_name: str | None = None,
) -> CalibrationResult:
    """Run calibration pipeline and return result.

        Convenience function for run calibration without manually
        setting up executor.

    Parameters
    ----------
    ms_path : str
        Path to Measurement Set.
    target_field : str
        Target field to calibrate.
    config : object
        Pipeline configuration.
    cal_field : optional
        Calibrator field (auto-detected if None).
    refant : optional
        Reference antenna (auto-detected if None).
    do_k : optional
        Whether to solve K calibration.
    calibrator_name : optional
        Expected calibrator name for model lookup (e.g., "0834+555").

    Returns
    -------
        CalibrationResult
        Execution status and outputs.
    """
    from dsa110_contimg.workflow.pipeline import PipelineExecutor

    started_at = datetime.now(UTC)

    # Create pipeline
    pipeline = CalibrationPipeline(
        config=config,
        ms_path=ms_path,
        cal_field=cal_field,
        target_field=target_field,
        refant=refant,
        do_k=do_k,
        calibrator_name=calibrator_name,
    )

    # Execute
    executor = PipelineExecutor(db_path=config.database_path)
    try:
        execution_id = await executor.execute(pipeline)
        status = await executor.get_status(execution_id)

        completed_at = datetime.now(UTC)

        # Extract results from job outputs
        solve_id = None
        application_id = None
        validation_passed = None
        gaintables = []
        warnings = []

        for job in status.jobs:
            if job.get("job_id") == "solve" and job.get("outputs_json"):
                import json

                outputs = json.loads(job["outputs_json"])
                solve_id = outputs.get("solve_id")
                if outputs.get("bp_table_path"):
                    gaintables.append(outputs["bp_table_path"])
                if outputs.get("g_table_path"):
                    gaintables.append(outputs["g_table_path"])
                if outputs.get("k_table_path"):
                    gaintables.append(outputs["k_table_path"])

            elif job.get("job_id") == "apply" and job.get("outputs_json"):
                import json

                outputs = json.loads(job["outputs_json"])
                application_id = outputs.get("application_id")

            elif job.get("job_id") == "validate" and job.get("outputs_json"):
                import json

                outputs = json.loads(job["outputs_json"])
                validation_passed = outputs.get("validation_passed")
                warnings = outputs.get("warnings", [])

        return CalibrationResult(
            success=status.status == "completed",
            solve_id=solve_id,
            application_id=application_id,
            validation_passed=validation_passed,
            gaintables=gaintables,
            message=f"Calibration pipeline {status.status}",
            errors=[status.error] if status.error else [],
            warnings=warnings,
            execution_id=execution_id,
            started_at=started_at,
            completed_at=completed_at,
        )

    except Exception as e:
        logger.exception(f"Calibration pipeline failed: {e}")
        return CalibrationResult(
            success=False,
            message=f"Calibration pipeline failed: {e}",
            errors=[str(e)],
            started_at=started_at,
            completed_at=datetime.now(UTC),
        )
