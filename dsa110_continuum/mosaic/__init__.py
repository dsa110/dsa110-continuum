"""
Mosaicking module for DSA-110 Continuum Imaging Pipeline.

This module provides a simple, unified mosaicking system with:
- Three clear tiers: Quicklook, Science, Deep
- Dagster-governed pipeline execution
- Unified database state
- Contract-tested with real FITS files

Architecture:
    tiers.py     - Tier definitions and selection logic
    builder.py   - Core mosaic building function
    qa.py        - Quality assessment checks
    jobs.py      - Dagster job implementations
    pipeline.py  - Pipeline definitions (on-demand only)
    api.py       - FastAPI endpoints
    schema.py    - Database schema definitions
"""

from .api import (
    MosaicRequest,
    MosaicResponse,
    MosaicStatusResponse,
    configure_mosaic_api,
)
from .api import (
    router as mosaic_router,
)
from .builder import (
    MosaicResult,
    build_mosaic,
    compute_rms,
)
from .jobs import (
    JobResult,
    MosaicBuildJob,
    MosaicJobConfig,
    MosaicPlanningJob,
    MosaicQAJob,
)
from .jobs_wsclean import (
    MosaicMSPlanningJob,
    MosaicWSCleanBuildJob,
    MosaicWSCleanJobConfig,
)
from .wsclean_mosaic import (
    WSCleanMosaicConfig,
    WSCleanMosaicResult,
    build_wsclean_mosaic,
    cleanup_scratch,
    compute_mean_meridian,
    copy_ms_to_scratch,
    run_chgcentre,
    run_wsclean_mosaic,
)
from .orchestrator import (
    MosaicOrchestrator,
    OrchestratorConfig,
)
from .pipeline import (
    MosaicPipelineConfig,
    NotificationConfig,
    OnDemandMosaicPipeline,
    PipelineResult,
    PipelineStatus,
    RetryBackoff,
    # Re-exported from pipeline framework
    RetryPolicy,
    execute_mosaic_pipeline_task,
    run_mosaic_pipeline,
    run_on_demand_mosaic,
)
from .qa import (
    ArtifactResult,
    AstrometryResult,
    PhotometryResult,
    QAResult,
    check_artifacts,
    check_astrometry,
    check_photometry,
    run_qa_checks,
)
from .schema import (
    MOSAIC_INDEXES,
    MOSAIC_TABLES,
    ensure_mosaic_tables,
    get_mosaic_schema_sql,
)
from .tiers import (
    TIER_CONFIGS,
    MosaicTier,
    TierConfig,
    get_tier_config,
    select_tier_for_request,
)
from .trigger import (
    DEC_TOLERANCE,
    SLEW_STABILITY_COUNT,
    STRIDE,
    WINDOW,
    SlidingWindowTrigger,
    TileRecordResult,
    TriggerState,
)

__all__ = [
    # Tiers
    "MosaicTier",
    "TierConfig",
    "TIER_CONFIGS",
    "select_tier_for_request",
    "get_tier_config",
    # Builder (legacy image-domain - deprecated)
    "MosaicResult",
    "build_mosaic",
    "compute_rms",
    # WSClean (visibility-domain - preferred)
    "WSCleanMosaicConfig",
    "WSCleanMosaicResult",
    "build_wsclean_mosaic",
    "run_wsclean_mosaic",
    "run_chgcentre",
    "copy_ms_to_scratch",
    "compute_mean_meridian",
    "cleanup_scratch",
    # QA
    "QAResult",
    "AstrometryResult",
    "PhotometryResult",
    "ArtifactResult",
    "run_qa_checks",
    "check_astrometry",
    "check_photometry",
    "check_artifacts",
    # Jobs (legacy image-domain)
    "JobResult",
    "MosaicJobConfig",
    "MosaicPlanningJob",
    "MosaicBuildJob",
    "MosaicQAJob",
    # Jobs (WSClean visibility-domain - preferred)
    "MosaicWSCleanJobConfig",
    "MosaicMSPlanningJob",
    "MosaicWSCleanBuildJob",
    # Pipeline
    "PipelineResult",
    "PipelineStatus",
    "MosaicPipelineConfig",
    "OnDemandMosaicPipeline",
    "run_on_demand_mosaic",
    "run_mosaic_pipeline",
    "execute_mosaic_pipeline_task",
    "RetryPolicy",
    "RetryBackoff",
    "NotificationConfig",
    # Schema
    "MOSAIC_TABLES",
    "MOSAIC_INDEXES",
    "ensure_mosaic_tables",
    "get_mosaic_schema_sql",
    # API
    "mosaic_router",
    "configure_mosaic_api",
    "MosaicRequest",
    "MosaicResponse",
    "MosaicStatusResponse",
    # Orchestrator
    "MosaicOrchestrator",
    "OrchestratorConfig",
    # Trigger
    "SlidingWindowTrigger",
    "TileRecordResult",
    "TriggerState",
    "STRIDE",
    "WINDOW",
    "DEC_TOLERANCE",
    "SLEW_STABILITY_COUNT",
]
