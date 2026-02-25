"""
Pipeline definitions for user-triggered mosaicking.

OnDemandMosaicPipeline: User-requested mosaic via API

Job graph:
    MosaicPlanningJob → MosaicBuildJob → MosaicQAJob
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any

from dsa110_contimg.workflow.pipeline import (
    NotificationConfig,
    Pipeline,
    RetryBackoff,
    RetryPolicy,
    register_pipeline,
)

from .jobs import (
    MosaicBuildJob,
    MosaicPlanningJob,
    MosaicQAJob,
)
from .science_jobs import (
    ScienceMosaicBridgeJob,
    SciencePlanningJob,
)
from .tiers import select_tier_for_request

# Re-export for backward compatibility
__all__ = [
    "PipelineStatus",
    "MosaicPipelineConfig",
    "PipelineResult",
    "OnDemandMosaicPipeline",
    "run_on_demand_mosaic",
    "run_mosaic_pipeline",
    "execute_mosaic_pipeline_task",
    # Re-exported from pipeline framework
    "RetryPolicy",
    "RetryBackoff",
    "NotificationConfig",
]

logger = logging.getLogger(__name__)


# =============================================================================
# Pipeline Status Enum (for API compatibility)
# =============================================================================


class PipelineStatus(str, Enum):
    """Status of a pipeline execution."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


# =============================================================================
# Configuration Dataclasses
# =============================================================================


@dataclass
class MosaicPipelineConfig:
    """Configuration for mosaic pipelines."""

    database_path: Path
    mosaic_dir: Path
    images_table: str = "images"


@dataclass
class PipelineResult:
    """Result of a pipeline execution."""

    success: bool
    plan_id: int | None = None
    mosaic_id: int | None = None
    mosaic_path: str | None = None
    qa_status: str | None = None
    n_images: int | None = None
    message: str = ""
    errors: list[str] = field(default_factory=list)
    execution_id: str | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None

    @property
    def num_tiles(self) -> int | None:
        """Number of tiles/images combined in the mosaic.

        Alias for n_images to provide a more descriptive API for consumers.
        """
        return self.n_images


# =============================================================================
# Pipeline Classes
# =============================================================================


@register_pipeline
class OnDemandMosaicPipeline(Pipeline):
    """On-demand mosaic pipeline for API-triggered requests.

        Same job structure as nightly, but with user-specified parameters.
        Tier is auto-selected based on time range if not provided.

        Job graph:
        plan (MosaicPlanningJob)
        └─> build (MosaicBuildJob)
        └─> qa (MosaicQAJob)

    Examples
    --------
        >>> pipeline = OnDemandMosaicPipeline(
        ...     config=config,
        ...     name="custom_mosaic",
        ...     start_time=1700000000,
        ...     end_time=1700086400,
        ...     tier="deep",
        ... )
        >>> result = pipeline.execute()
    """

    pipeline_name = "on_demand_mosaic"
    schedule = None  # On-demand, not scheduled

    def __init__(
        self,
        config: MosaicPipelineConfig | None = None,
        name: str = "mosaic",
        start_time: int = 0,
        end_time: int = 0,
        tier: str | None = None,
    ):
        """Initialize on-demand pipeline.

        Parameters
        ----------
        config : MosaicPipelineConfig
            Pipeline configuration
        name : str
            Unique mosaic name
        start_time : int
            Start time (Unix timestamp)
        end_time : int
            End time (Unix timestamp)
        tier : str or None, optional
            Tier to use (auto-selected if not provided)
        """
        self.mosaic_name = name
        self.start_time = start_time
        self.end_time = end_time

        # Auto-select tier if not provided
        if tier is None:
            time_range_hours = (end_time - start_time) / 3600
            selected_tier = select_tier_for_request(time_range_hours)
            self.tier = selected_tier.value
        else:
            self.tier = tier

        super().__init__(config)

    def build(self) -> None:
        """Build the on-demand pipeline job graph."""
        if self.tier == "science":
            self._build_science_workflow()
        else:
            self._build_standard_workflow()

        # Configure retry
        self.set_retry_policy(max_retries=2, backoff="exponential")

    def _build_science_workflow(self) -> None:
        """Build the rigorous Science workflow."""
        # Job 1: Science Planning (creates pending plan)
        self.add_job(
            SciencePlanningJob,
            job_id="plan",
            params={
                "start_time": self.start_time,
                "end_time": self.end_time,
                "tier": self.tier,
                "mosaic_name": self.mosaic_name,
            },
        )

        # Job 2: Science Mosaic Bridge (executes Dagster workflow)
        self.add_job(
            ScienceMosaicBridgeJob,
            job_id="build",
            params={
                "plan_id": "${plan.plan_id}",
                "start_time": self.start_time,
                "end_time": self.end_time,
                "mosaic_name": self.mosaic_name,
            },
            dependencies=["plan"],
        )

        # Note: QA is handled within the Dagster workflow or skipped for now
        # Ideally, ScienceMosaicBridgeJob should output mosaic_id and we can run MosaicQAJob

    def _build_standard_workflow(self) -> None:
        """Build the standard linear mosaicking workflow."""
        # Job 1: Planning
        self.add_job(
            MosaicPlanningJob,
            job_id="plan",
            params={
                "start_time": self.start_time,
                "end_time": self.end_time,
                "tier": self.tier,
                "mosaic_name": self.mosaic_name,
            },
        )

        # Job 2: Build (depends on plan)
        self.add_job(
            MosaicBuildJob,
            job_id="build",
            params={
                "plan_id": "${plan.plan_id}",
            },
            dependencies=["plan"],
        )

        # Job 3: QA (depends on build)
        self.add_job(
            MosaicQAJob,
            job_id="qa",
            params={
                "mosaic_id": "${build.mosaic_id}",
            },
            dependencies=["build"],
        )


# =============================================================================
# Wrapper Functions
# =============================================================================


def run_on_demand_mosaic(
    config: MosaicPipelineConfig,
    name: str,
    start_time: int,
    end_time: int,
    tier: str | None = None,
) -> PipelineResult:
    """Run on-demand mosaic for user request.

    Parameters
    ----------
    config : MosaicPipelineConfig
        Pipeline configuration
    name : str
        Unique mosaic name
    start_time : int
        Start time (Unix timestamp)
    end_time : int
        End time (Unix timestamp)
    tier : str or None, optional
        Tier to use (auto-selected if not provided), by default None

    Returns
    -------
        None
    """
    import asyncio

    from dsa110_contimg.workflow.pipeline import PipelineExecutor

    pipeline = OnDemandMosaicPipeline(
        config=config,
        name=name,
        start_time=start_time,
        end_time=end_time,
        tier=tier,
    )

    # Use executor for proper tracking
    executor = PipelineExecutor(config.database_path)
    execution_id = asyncio.get_event_loop().run_until_complete(executor.execute(pipeline))

    # Get status
    status = asyncio.get_event_loop().run_until_complete(executor.get_status(execution_id))

    return _status_to_result(status, pipeline)


def run_mosaic_pipeline(
    config: MosaicPipelineConfig,
    mosaic_name: str,
    start_time: int,
    end_time: int,
    tier: str,
) -> PipelineResult:
    """Execute full mosaic pipeline: Plan → Build → QA.

    Parameters
    ----------
    config : MosaicPipelineConfig
        Pipeline configuration
    mosaic_name : str
        Unique name for the mosaic
    start_time : int
        Start time (Unix timestamp)
    end_time : int
        End time (Unix timestamp)
    tier : str
        Tier to use

    Returns
    -------
        None
    """
    return run_on_demand_mosaic(
        config=config,
        name=mosaic_name,
        start_time=start_time,
        end_time=end_time,
        tier=tier,
    )


def _status_to_result(status: Any, pipeline: Pipeline) -> PipelineResult:
    """Convert executor status to PipelineResult.

    Parameters
    ----------
    status : Any
        Executor status to convert
    pipeline : Pipeline
        Pipeline instance

    Returns
    -------
        PipelineResult
        Result corresponding to the executor status
    """
    from datetime import datetime

    # Extract job outputs
    plan_id = None
    mosaic_id = None
    mosaic_path = None
    qa_status = None
    n_images = None
    errors = []

    for job in status.jobs:
        if job.get("outputs_json"):
            import json

            outputs = json.loads(job["outputs_json"])
            if job["job_id"] == "plan":
                plan_id = outputs.get("plan_id")
            elif job["job_id"] == "build":
                mosaic_id = outputs.get("mosaic_id")
                mosaic_path = outputs.get("mosaic_path")
                n_images = outputs.get("n_images")
            elif job["job_id"] == "qa":
                qa_status = outputs.get("qa_status")

        if job.get("error"):
            errors.append(f"{job['job_id']}: {job['error']}")

    success = status.status == "completed" and not errors

    return PipelineResult(
        success=success,
        plan_id=plan_id,
        mosaic_id=mosaic_id,
        mosaic_path=mosaic_path,
        qa_status=qa_status,
        n_images=n_images,
        message=f"Pipeline {status.status}",
        errors=errors,
        execution_id=status.execution_id,
        started_at=datetime.fromtimestamp(status.started_at, tz=UTC) if status.started_at else None,
        completed_at=datetime.fromtimestamp(status.completed_at, tz=UTC)
        if status.completed_at
        else None,
    )


# =============================================================================
# Legacy Integration
# =============================================================================


async def execute_mosaic_pipeline_task(params: dict[str, Any]) -> dict[str, Any]:
    """Execute on-demand mosaic pipeline as a legacy task.

        This function wraps the pipeline for use with legacy task queue.

    Parameters
    ----------
    params : dict
        Task parameters including:
        - database_path: Path to database
        - mosaic_dir: Output directory
        - name: Mosaic name
        - start_time: Start time (Unix timestamp)
        - end_time: End time (Unix timestamp)
        - tier: Optional tier override (auto-selected if not provided)

    Returns
    -------
        dict
        Task result dict with status, outputs, and execution metadata
    """
    from dsa110_contimg.workflow.pipeline import PipelineExecutor

    # Build config
    config = MosaicPipelineConfig(
        database_path=Path(params["database_path"]),
        mosaic_dir=Path(params["mosaic_dir"]),
        images_table=params.get("images_table", "images"),
    )

    # Create on-demand pipeline
    pipeline = OnDemandMosaicPipeline(
        config=config,
        name=params["name"],
        start_time=params["start_time"],
        end_time=params["end_time"],
        tier=params.get("tier"),
    )

    # Execute via executor
    executor = PipelineExecutor(config.database_path)
    execution_id = await executor.execute(pipeline)
    status = await executor.get_status(execution_id)

    result = _status_to_result(status, pipeline)

    if result.success:
        return {
            "status": "success",
            "execution_id": result.execution_id,
            "outputs": {
                "plan_id": result.plan_id,
                "mosaic_id": result.mosaic_id,
                "mosaic_path": result.mosaic_path,
                "qa_status": result.qa_status,
            },
            "message": result.message,
            "started_at": result.started_at.isoformat() if result.started_at else None,
            "completed_at": result.completed_at.isoformat() if result.completed_at else None,
        }
    else:
        return {
            "status": "error",
            "execution_id": result.execution_id,
            "message": result.message,
            "errors": result.errors,
        }
