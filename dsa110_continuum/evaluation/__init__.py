"""
Evaluation framework for DSA-110 Continuum Imaging Pipeline.

This module provides custom evaluators and evaluation harnesses for measuring
pipeline quality, failure detection, and regression coverage.

Core Components:
    - PipelineStage: Enum of the 6 pipeline stages
    - StageSpec: Specification for each stage (inputs, outputs, metrics)
    - StageRegistry: Central registry with YAML-configurable thresholds

Evaluators:
    - PipelineCompletionEvaluator: Measures stage completion accuracy
    - FailureDetectionEvaluator: Validates failure detection rate
    - RegressionCoverageEvaluator: Compares outputs against golden baselines

Usage:
    from dsa110_contimg.core.evaluation import (
        PipelineStage,
        StageRegistry,
        PipelineCompletionEvaluator,
        FailureDetectionEvaluator,
        RegressionCoverageEvaluator,
        run_evaluation,
    )

    # Get stage specifications
    registry = StageRegistry()
    calibrate_spec = registry.get_stage(PipelineStage.CALIBRATE)

    # Run evaluation with all metrics
    results = run_evaluation("evaluation_dataset.jsonl")

    # Use individual evaluators
    completion_eval = PipelineCompletionEvaluator()
    score = completion_eval(
        query="Process 16 subbands from 0834+555 transit",
        response={"status": "completed", "stages_completed": 6, "stages_total": 6}
    )
"""

from .database import EvaluationDatabase, get_evaluation_db
from .evaluators import (
    FailureDetectionEvaluator,
    PipelineCompletionEvaluator,
    RegressionCoverageEvaluator,
)
from .harness import (
    EvaluationResult,
    StageEvaluationReport,
    create_evaluation_dataset,
    create_stage_evaluators,
    load_thresholds_config,
    run_evaluation,
    run_full_pipeline_evaluation,
    run_stage_evaluation,
)
from .html_report import (
    generate_html_report,
    generate_stage_report,
    render_evaluation_report,
)
from .stage_evaluators import (
    BaseStageEvaluator,
    CalibrationEvaluator,
    CheckType,
    ConversionEvaluator,
    ImagingEvaluator,
    IngestEvaluator,
    MetricCheck,
    MosaicEvaluator,
    PhotometryEvaluator,
    StageEvaluationResult,
    count_passed_stages,
    evaluate_pipeline_run,
    get_evaluator,
)
from .stages import (
    DataProduct,
    MetricSpec,
    PipelineStage,
    StageRegistry,
    StageSpec,
    get_registry,
    get_stage_spec,
)

__all__ = [
    # Stage taxonomy
    "PipelineStage",
    "StageSpec",
    "DataProduct",
    "MetricSpec",
    "StageRegistry",
    "get_registry",
    "get_stage_spec",
    # Stage evaluators
    "BaseStageEvaluator",
    "IngestEvaluator",
    "ConversionEvaluator",
    "CalibrationEvaluator",
    "ImagingEvaluator",
    "PhotometryEvaluator",
    "MosaicEvaluator",
    "StageEvaluationResult",
    "MetricCheck",
    "CheckType",
    "get_evaluator",
    "evaluate_pipeline_run",
    "count_passed_stages",
    # High-level evaluators
    "PipelineCompletionEvaluator",
    "FailureDetectionEvaluator",
    "RegressionCoverageEvaluator",
    # Harness
    "run_evaluation",
    "run_stage_evaluation",
    "run_full_pipeline_evaluation",
    "create_evaluation_dataset",
    "create_stage_evaluators",
    "load_thresholds_config",
    "EvaluationResult",
    "StageEvaluationReport",
    # Database
    "EvaluationDatabase",
    "get_evaluation_db",
    # HTML Reports
    "generate_html_report",
    "generate_stage_report",
    "render_evaluation_report",
]
