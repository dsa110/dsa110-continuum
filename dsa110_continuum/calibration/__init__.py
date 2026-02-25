# This file initializes the calibration module.
"""
DSA-110 Continuum Imaging Pipeline - Calibration Module.

.. note::
    For new code, prefer using the public API which provides a simpler interface:

        from dsa110_contimg.interfaces.public_api import calibrate_ms

    This module is primarily for internal use and advanced customization.

This module provides calibration functionality including:
- Bandpass and gain calibration
- Self-calibration
- Calibration QA and validation
- Catalog-based calibrator lookup
"""

# Bandpass diagnostics and auto-recovery
from dsa110_contimg.core.calibration.bandpass_diagnostics import (
    DiagnosticReport,
    analyze_flagging_pattern,
    auto_recover_bandpass_calibration,
    check_geometric_setup,
    check_snr_budget,
    diagnose_bandpass_quality,
    extract_bandpass_flagging_stats,
    get_recovery_recommendations,
)

# Bandpass diagnostics report
from dsa110_contimg.core.calibration.bandpass_report import (
    BandpassReportData,
    generate_bandpass_report,
)

# Beam model
from dsa110_contimg.core.calibration.beam_model import (
    BeamConfig,
    primary_beam_response,
)

# Catalog registry (unified catalog query interface)
from dsa110_contimg.core.calibration.catalog_registry import (
    CATALOG_REGISTRY,
    CatalogConfig,
    CatalogName,
    list_available_catalogs,
    query_catalog,
    query_multiple_catalogs,
)
from dsa110_contimg.core.calibration.checkpoints import (
    CalibrationCheckpoint,
    MSIntegrityError,
    validate_ms_integrity,
)

# Preflight checks and fail-fast validation
# OPENBLAS_NUM_THREADS is managed by WSClean execution paths to avoid oversubscription.
from dsa110_contimg.core.calibration.flagging import (
    PreflightError,
    preflight_check_all,
    preflight_check_aoflagger,
    preflight_check_aoflagger_docker_mounts,
    preflight_check_casa,
    preflight_check_disk_space,
    preflight_check_memory,
    preflight_check_output_dir,
    preflight_check_strategy_file,
    preflight_check_wsclean,
)

# Adaptive flagging
from dsa110_contimg.core.calibration.flagging_adaptive import (
    AdaptiveFlaggingResult,
    CalibrationFailure,
    FlaggingStrategy,
    flag_rfi_adaptive,
    flag_rfi_with_gpu_fallback,
)
from dsa110_contimg.core.calibration.flux_validation import (
    FluxScaleCheckResult,
    check_model_corrected_ratio,
)

# Flux calibration (CASA fluxscale bootstrap)
from dsa110_contimg.core.calibration.fluxscale import (
    PRIMARY_FLUX_CALIBRATORS,
    FluxBootstrapResult,
    FluxscaleResult,
    SetjyResult,
    bootstrap_flux_scale,
    bootstrap_flux_scale_single_ms,
    get_latest_flux_for_calibrator,
    get_primary_calibrator_info,
    is_primary_flux_calibrator,
    list_primary_flux_calibrators,
    record_flux_bootstrap,
    run_fluxscale,
    set_model_primary_calibrator,
    update_calibrator_catalog_flux,
)

# Pipeline guardrails for calibration quality
from dsa110_contimg.core.calibration.guardrails import (
    CalibrationGuardrails,
    QualityAction,
    QualityMetrics,
    QualityThresholds,
    extract_quality_metrics,
    get_quality_action,
    get_quality_tier,
)

# Pipeline jobs and orchestration
from dsa110_contimg.core.calibration.jobs import (
    CalibrationApplyJob,
    CalibrationJobConfig,
    CalibrationSolveJob,
    CalibrationValidateJob,
)
from dsa110_contimg.core.calibration.pipeline import (
    CalibrationPipeline,
    CalibrationPipelineConfig,
    CalibrationResult,
    CalibrationStatus,
    StreamingCalibrationPipeline,
    run_calibration_pipeline,
)

# Pre-calibration validation gate
from dsa110_contimg.core.calibration.preconditions import (
    PreconditionResult,
    ValidationGateResult,
    require_valid_preconditions,
    validate_bandpass_preconditions,
)

# QA module
from dsa110_contimg.core.calibration.qa import (
    CalibrationMetrics,
    CalibrationQAResult,
    CalibrationQAStore,
    QAIssue,
    QAThresholds,
    assess_calibration_quality,
    compute_calibration_metrics,
    get_qa_store,
)
from dsa110_contimg.core.calibration.qa_compare import compare_caltables

# Self-calibration
from dsa110_contimg.core.calibration.selfcal import (
    SelfCalConfig,
    SelfCalIterationResult,
    SelfCalMode,
    SelfCalResult,
    SelfCalStatus,
    selfcal_iteration,
    selfcal_ms,
)

# Self-calibration diagnostics
from dsa110_contimg.core.calibration.selfcal_diagnostics import (
    generate_observation_diagnostics,
    generate_selfcal_diagnostics,
)

# Production self-calibration routine
from dsa110_contimg.core.calibration.selfcal_routine import (
    SelfCalIterationConfig,
    SelfCalIterationResult as SelfCalRoutineIterationResult,
    SelfCalRoutineConfig,
    SelfCalRoutineError,
    SelfCalRoutineResult,
    run_selfcal_routine,
)
from dsa110_contimg.core.calibration.transit import (
    find_observations_containing_transit,
    find_transits_for_source,
    next_transit_time,
    observation_contains_transit,
    transit_times,
    upcoming_transits,
)

__all__ = [
    # Beam model
    "BeamConfig",
    "primary_beam_response",
    # Transit utilities
    "find_observations_containing_transit",
    "find_transits_for_source",
    "next_transit_time",
    "observation_contains_transit",
    "transit_times",
    "upcoming_transits",
    # Pipeline jobs
    "CalibrationApplyJob",
    "CalibrationJobConfig",
    "CalibrationSolveJob",
    "CalibrationValidateJob",
    # Pipelines
    "CalibrationPipeline",
    "CalibrationPipelineConfig",
    "CalibrationResult",
    "CalibrationStatus",
    "StreamingCalibrationPipeline",
    "run_calibration_pipeline",
    # QA
    "CalibrationMetrics",
    "CalibrationQAResult",
    "CalibrationQAStore",
    "QAIssue",
    "QAThresholds",
    "assess_calibration_quality",
    "compute_calibration_metrics",
    "get_qa_store",
    # Adaptive flagging
    "CalibrationFailure",
    "flag_rfi_adaptive",
    "flag_rfi_with_gpu_fallback",
    "FlaggingStrategy",
    "AdaptiveFlaggingResult",
    # Preflight checks and fail-fast validation
    "PreflightError",
    "preflight_check_aoflagger",
    "preflight_check_aoflagger_docker_mounts",
    "preflight_check_all",
    "preflight_check_casa",
    "preflight_check_wsclean",
    "preflight_check_disk_space",
    "preflight_check_output_dir",
    "preflight_check_memory",
    "preflight_check_strategy_file",
    "MSIntegrityError",
    "validate_ms_integrity",
    "CalibrationCheckpoint",
    # Self-calibration
    "SelfCalMode",
    "SelfCalStatus",
    "SelfCalConfig",
    "SelfCalIterationResult",
    "SelfCalResult",
    "selfcal_iteration",
    "selfcal_ms",
    "generate_selfcal_diagnostics",
    "generate_observation_diagnostics",
    "SelfCalIterationConfig",
    "SelfCalRoutineIterationResult",
    "SelfCalRoutineConfig",
    "SelfCalRoutineError",
    "SelfCalRoutineResult",
    "run_selfcal_routine",
    "compare_caltables",
    # Bandpass diagnostics report
    "BandpassReportData",
    "generate_bandpass_report",
    # Catalog registry
    "CatalogName",
    "CatalogConfig",
    "CATALOG_REGISTRY",
    "query_catalog",
    "query_multiple_catalogs",
    "list_available_catalogs",
    # Flux calibration (fluxscale bootstrap)
    "PRIMARY_FLUX_CALIBRATORS",
    "SetjyResult",
    "FluxscaleResult",
    "FluxBootstrapResult",
    "is_primary_flux_calibrator",
    "get_primary_calibrator_info",
    "list_primary_flux_calibrators",
    "set_model_primary_calibrator",
    "run_fluxscale",
    "bootstrap_flux_scale",
    "bootstrap_flux_scale_single_ms",
    "record_flux_bootstrap",
    "get_latest_flux_for_calibrator",
    "update_calibrator_catalog_flux",
    # Quick flux scale checks
    "FluxScaleCheckResult",
    "check_model_corrected_ratio",
    # Bandpass diagnostics and auto-recovery
    "DiagnosticReport",
    "analyze_flagging_pattern",
    "auto_recover_bandpass_calibration",
    "check_geometric_setup",
    "check_snr_budget",
    "diagnose_bandpass_quality",
    "extract_bandpass_flagging_stats",
    "get_recovery_recommendations",
    # Pipeline guardrails
    "CalibrationGuardrails",
    "QualityAction",
    "QualityMetrics",
    "QualityThresholds",
    "extract_quality_metrics",
    "get_quality_action",
    "get_quality_tier",
    # Pre-calibration validation gate
    "PreconditionResult",
    "ValidationGateResult",
    "require_valid_preconditions",
    "validate_bandpass_preconditions",
]
