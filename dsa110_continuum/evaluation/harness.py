"""
Evaluation harness for running pipeline evaluations.

This module provides the orchestration layer for:
- Loading test datasets (JSONL format)
- Running evaluators against pipeline responses
- Aggregating and reporting results
- Storing baselines for regression comparison

Usage:
    from dsa110_contimg.core.evaluation import run_evaluation, create_evaluation_dataset
from dsa110_contimg.common.utils import get_env_path

    # Create dataset from fixtures
    create_evaluation_dataset(
        output_path="evaluation_data.jsonl",
        include_failure_cases=True,
        include_synthetic=True
    )

    # Run full evaluation
    results = run_evaluation("evaluation_data.jsonl")
    print(results.metric_averages)
"""

from __future__ import annotations

import json
import logging
import os
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any
import statistics

from .config_loader import load_thresholds_config as load_thresholds_config_helper
from .evaluators import create_evaluators
from .stage_evaluators import (
    StageEvaluationResult,
    get_evaluator,
)
from .stages import PipelineStage

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================


# Default paths
_CONTIMG_BASE = get_env_path("CONTIMG_BASE_DIR", default="/data/dsa110-contimg")
DEFAULT_EVAL_DIR = _CONTIMG_BASE / "state" / "evaluation"
DEFAULT_BASELINE_DIR = DEFAULT_EVAL_DIR / "baselines"
DEFAULT_RESULTS_DIR = DEFAULT_EVAL_DIR / "results"

# Test data locations

FIXTURES_DIR = _CONTIMG_BASE / "backend" / "tests" / "fixtures"


# =============================================================================
# Data Structures
# =============================================================================


@dataclass
class EvaluationSample:
    """A single evaluation sample with query, response, and ground truth."""

    query: str
    response: dict[str, Any]
    ground_truth: dict[str, Any] | None = None
    sample_id: str | None = None
    category: str = "general"  # completion, failure, regression

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EvaluationSample:
        """Create sample from dictionary (JSONL row).

        Parameters
        ----------
        data: Dict[str :

        Any] :


        """
        return cls(
            query=data.get("query", ""),
            response=data.get("response", {}),
            ground_truth=data.get("ground_truth"),
            sample_id=data.get("sample_id"),
            category=data.get("category", "general"),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSONL serialization."""
        d = {
            "query": self.query,
            "response": self.response,
            "category": self.category,
        }
        if self.ground_truth:
            d["ground_truth"] = self.ground_truth
        if self.sample_id:
            d["sample_id"] = self.sample_id
        return d


@dataclass
class EvaluationResult:
    """Results from running evaluation."""

    # Per-sample results
    sample_results: list[dict[str, Any]] = field(default_factory=list)

    # Aggregate metrics (per evaluator, per metric)
    metric_averages: dict[str, dict[str, float]] = field(default_factory=dict)

    # Metadata
    timestamp: str = ""
    dataset_path: str = ""
    num_samples: int = 0
    duration_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "timestamp": self.timestamp,
            "dataset_path": self.dataset_path,
            "num_samples": self.num_samples,
            "duration_seconds": self.duration_seconds,
            "metric_averages": self.metric_averages,
            "sample_results": self.sample_results,
        }

    def save(self, output_path: Path) -> None:
        """Save results to JSON file.

        Parameters
        ----------
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info(f"Saved evaluation results to {output_path}")


# =============================================================================
# Dataset Loading
# =============================================================================


def load_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    """Load samples from JSONL file.

    Parameters
    ----------
    path :
        Path to JSONL file

    Yields
    ------
    path :
        Path to JSONL file

    Yields
    ------
        Dict for each line
    """
    with open(path) as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                logger.warning(f"Skipping invalid JSON at line {line_num}: {e}")


def load_evaluation_dataset(path: Path) -> list[EvaluationSample]:
    """Load evaluation dataset from JSONL file.

        Expected format per line:
        {
        "query": "Process transit obs_id=12345",
        "response": {...},
        "ground_truth": {...},
        "category": "completion"
        }

    Parameters
    ----------
    path : Path
        Path to JSONL dataset
    """
    samples = []
    for row in load_jsonl(path):
        samples.append(EvaluationSample.from_dict(row))
    logger.info(f"Loaded {len(samples)} samples from {path}")
    return samples


# =============================================================================
# Evaluation Runner
# =============================================================================


def run_evaluation(
    dataset_path: str | Path,
    evaluators: dict[str, Callable] | None = None,
    output_dir: Path | None = None,
) -> EvaluationResult:
    """Run evaluation on a dataset using all evaluators.

    Parameters
    ----------
    dataset_path : str or Path
        Path to JSONL dataset file
    evaluators : Optional[Dict[str, Callable]], optional
        Dict of evaluator name to evaluator callable, defaults to all pipeline evaluators
    output_dir : Optional[Path], optional
        Directory to save results
    """
    dataset_path = Path(dataset_path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    # Load evaluators
    if evaluators is None:
        evaluators = create_evaluators()

    # Load dataset
    samples = load_evaluation_dataset(dataset_path)
    if not samples:
        raise ValueError(f"Empty dataset: {dataset_path}")

    # Run evaluation
    start_time = time.time()
    sample_results = []
    metric_totals: dict[str, dict[str, list[float]]] = {name: {} for name in evaluators}

    for sample in samples:
        result_entry = {
            "sample_id": sample.sample_id,
            "category": sample.category,
            "query": sample.query,
            "scores": {},
        }

        # Run each evaluator
        for eval_name, evaluator in evaluators.items():
            try:
                scores = evaluator(
                    query=sample.query,
                    response=sample.response,
                    ground_truth=sample.ground_truth,
                )
                result_entry["scores"][eval_name] = scores

                # Aggregate numeric metrics for reporting/baselines
                for metric_name, metric_value in scores.items():
                    if isinstance(metric_value, (int, float)) and not isinstance(
                        metric_value, bool
                    ):
                        metric_totals[eval_name].setdefault(metric_name, []).append(
                            float(metric_value)
                        )

            except Exception as e:
                logger.error(f"Evaluator {eval_name} failed on sample {sample.sample_id}: {e}")
                result_entry["scores"][eval_name] = {"error": str(e)}

        sample_results.append(result_entry)

    duration = time.time() - start_time

    # Calculate metric averages
    metric_averages: dict[str, dict[str, float]] = {}
    for eval_name, metrics in metric_totals.items():
        averages = {}
        for metric_name, values in metrics.items():
            if values:
                averages[metric_name] = statistics.mean(values)
        if averages:
            metric_averages[eval_name] = averages

    # Build result
    result = EvaluationResult(
        sample_results=sample_results,
        metric_averages=metric_averages,
        timestamp=datetime.now().isoformat(),
        dataset_path=str(dataset_path),
        num_samples=len(samples),
        duration_seconds=duration,
    )

    # Save if output directory specified
    if output_dir:
        output_path = output_dir / f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        result.save(output_path)

    logger.info(
        f"Evaluation complete: {len(samples)} samples, "
        f"metric groups: {len(result.metric_averages)}, "
        f"duration: {duration:.1f}s"
    )

    return result


# =============================================================================
# Dataset Generation
# =============================================================================


def create_evaluation_dataset(
    output_path: str | Path,
    include_failure_cases: bool = True,
    include_synthetic: bool = True,
) -> Path:
    """Create evaluation dataset from fixtures and reference data.

    Parameters
    ----------
    output_path : str or Path
        Path for output JSONL file
    include_failure_cases : bool, optional
        Include failure detection samples (default True)
    include_synthetic : bool, optional
        Include synthetic test cases (default True)
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    samples: list[dict[str, Any]] = []

    # 1. Failure detection samples
    if include_failure_cases:
        samples.extend(_generate_failure_samples())

    # 2. Synthetic completion samples
    if include_synthetic:
        samples.extend(_generate_synthetic_completion_samples())

    # Write JSONL
    with open(output_path, "w") as f:
        for sample in samples:
            f.write(json.dumps(sample) + "\n")

    logger.info(f"Created evaluation dataset with {len(samples)} samples at {output_path}")
    return output_path


def _generate_failure_samples() -> list[dict[str, Any]]:
    """Generate failure detection test samples."""
    # Define failure scenarios
    failure_scenarios = [
        {
            "name": "conversion_timeout",
            "stage": "convert",
            "type": "timeout_error",
            "description": "UVH5 conversion exceeds time limit",
        },
        {
            "name": "calibration_divergence",
            "stage": "solve_cal",
            "type": "calibration_error",
            "description": "Gain solutions fail to converge",
        },
        {
            "name": "imaging_oom",
            "stage": "imaging",
            "type": "resource_exhaustion",
            "description": "GPU out of memory during gridding",
        },
        {
            "name": "subband_incomplete",
            "stage": "ingest",
            "type": "subband_grouping_error",
            "description": "Missing subbands in observation group",
        },
        {
            "name": "database_lock",
            "stage": "convert",
            "type": "database_error",
            "description": "SQLite database locked during write",
        },
    ]

    samples = []
    base_timestamp = 1700000000  # Arbitrary base time

    for i, scenario in enumerate(failure_scenarios):
        inject_time = base_timestamp + i * 1000
        detect_time = inject_time + 30  # 30s detection latency

        sample = {
            "sample_id": f"failure_{scenario['name']}",
            "category": "failure",
            "query": f"Monitor pipeline for failures: {scenario['description']}",
            "response": {
                "failures_detected": [
                    {
                        "type": scenario["type"],
                        "stage": scenario["stage"],
                        "timestamp": detect_time,
                        "message": scenario["description"],
                    }
                ],
                "alerts_raised": 1,
            },
            "ground_truth": {
                "expected_failures": [
                    {
                        "type": scenario["type"],
                        "stage": scenario["stage"],
                        "injected_at": inject_time,
                    }
                ],
            },
        }
        samples.append(sample)

    # Add a "no failure" baseline
    samples.append(
        {
            "sample_id": "failure_none_expected",
            "category": "failure",
            "query": "Monitor healthy pipeline run",
            "response": {
                "failures_detected": [],
                "alerts_raised": 0,
            },
            "ground_truth": {
                "expected_failures": [],
            },
        }
    )

    logger.info(f"Generated {len(samples)} failure detection samples")
    return samples


def _generate_synthetic_completion_samples() -> list[dict[str, Any]]:
    """Generate synthetic pipeline completion samples."""
    samples = []

    # Full completion scenario
    samples.append(
        {
            "sample_id": "completion_full",
            "category": "completion",
            "query": "Process 16 subbands from 0834+555 transit",
            "response": {
                "status": "completed",
                "stages_completed": [
                    "ingest",
                    "convert",
                    "flag_rfi",
                    "solve_cal",
                    "apply_cal",
                    "imaging",
                ],
                "subbands_processed": 16,
                "state_transitions": [
                    ("pending", "converting"),
                    ("converting", "converted"),
                    ("converted", "flagging_rfi"),
                    ("flagging_rfi", "solving_cal"),
                    ("solving_cal", "applying_cal"),
                    ("applying_cal", "imaging"),
                    ("imaging", "done"),
                ],
            },
            "ground_truth": {
                "expected_stages": 6,
                "expected_subbands": 16,
            },
        }
    )

    # Partial completion (stopped at calibration)
    samples.append(
        {
            "sample_id": "completion_partial",
            "category": "completion",
            "query": "Process transit with calibration-only mode",
            "response": {
                "status": "partial",
                "stages_completed": ["ingest", "convert", "flag_rfi", "solve_cal"],
                "subbands_processed": 16,
                "state_transitions": [
                    ("pending", "converting"),
                    ("converting", "converted"),
                    ("converted", "flagging_rfi"),
                    ("flagging_rfi", "solving_cal"),
                ],
            },
            "ground_truth": {
                "expected_stages": 4,  # Calibration-only expects 4 stages
                "expected_subbands": 16,
            },
        }
    )

    # Incomplete subbands (12/16)
    samples.append(
        {
            "sample_id": "completion_incomplete_subbands",
            "category": "completion",
            "query": "Process incomplete observation group",
            "response": {
                "status": "completed",
                "stages_completed": [
                    "ingest",
                    "convert",
                    "flag_rfi",
                    "solve_cal",
                    "apply_cal",
                    "imaging",
                ],
                "subbands_processed": 12,
                "state_transitions": [
                    ("pending", "converting"),
                    ("converting", "converted"),
                    ("converted", "flagging_rfi"),
                    ("flagging_rfi", "solving_cal"),
                    ("solving_cal", "applying_cal"),
                    ("applying_cal", "imaging"),
                    ("imaging", "done"),
                ],
            },
            "ground_truth": {
                "expected_stages": 6,
                "expected_subbands": 16,
            },
        }
    )

    # Failed with retry
    samples.append(
        {
            "sample_id": "completion_failed_retry",
            "category": "completion",
            "query": "Process transit with transient failure",
            "response": {
                "status": "completed",
                "stages_completed": [
                    "ingest",
                    "convert",
                    "flag_rfi",
                    "solve_cal",
                    "apply_cal",
                    "imaging",
                ],
                "subbands_processed": 16,
                "state_transitions": [
                    ("pending", "converting"),
                    ("converting", "failed"),
                    ("failed", "pending"),  # Retry
                    ("pending", "converting"),
                    ("converting", "converted"),
                    ("converted", "flagging_rfi"),
                    ("flagging_rfi", "solving_cal"),
                    ("solving_cal", "applying_cal"),
                    ("applying_cal", "imaging"),
                    ("imaging", "done"),
                ],
            },
            "ground_truth": {
                "expected_stages": 6,
                "expected_subbands": 16,
            },
        }
    )

    logger.info(f"Generated {len(samples)} completion samples")
    return samples


# =============================================================================
# Baseline Management
# =============================================================================


def save_baseline(
    result: EvaluationResult,
    name: str,
    baseline_dir: Path | None = None,
) -> Path:
    """Save evaluation result as a baseline for future comparison.

    Parameters
    ----------
    result : EvaluationResult
        Evaluation result to save
    name : str
        Baseline identifier
    baseline_dir : Optional[Path], optional
        Directory for baselines (default: state/evaluation/baselines)
    """
    baseline_dir = baseline_dir or DEFAULT_BASELINE_DIR
    baseline_dir.mkdir(parents=True, exist_ok=True)

    baseline_path = baseline_dir / f"{name}.json"
    result.save(baseline_path)

    logger.info(f"Saved baseline '{name}' to {baseline_path}")
    return baseline_path


def load_baseline(name: str, baseline_dir: Path | None = None) -> dict[str, Any]:
    """Load a baseline for comparison.

    Parameters
    ----------
    name : str
        Baseline identifier
    baseline_dir : Optional[Path], optional
        Directory containing baselines
    """
    baseline_dir = baseline_dir or DEFAULT_BASELINE_DIR
    baseline_path = baseline_dir / f"{name}.json"

    if not baseline_path.exists():
        raise FileNotFoundError(f"Baseline not found: {baseline_path}")

    with open(baseline_path) as f:
        return json.load(f)


def compare_to_baseline(
    result: EvaluationResult,
    baseline_name: str,
    tolerance: float = 0.05,
) -> dict[str, Any]:
    """Compare evaluation result to a saved baseline.

    Parameters
    ----------
    result : EvaluationResult
        Current evaluation result
    baseline_name : str
        Name of baseline to compare against
    tolerance : float, optional
        Acceptable deviation (default 0.05)
    """
    baseline = load_baseline(baseline_name)
    baseline_metrics = baseline.get("metric_averages", {})

    comparison = {
        "baseline_name": baseline_name,
        "baseline_timestamp": baseline.get("timestamp"),
        "current_timestamp": result.timestamp,
        "metrics": {},
        "regressions": [],
        "improvements": [],
    }

    for evaluator_name, metrics in result.metric_averages.items():
        baseline_for_eval = baseline_metrics.get(evaluator_name, {})
        for metric_name, current_value in metrics.items():
            baseline_value = baseline_for_eval.get(metric_name)
            diff = current_value - baseline_value if baseline_value is not None else None
            pct_change = diff / baseline_value if baseline_value not in (None, 0.0) else None

            comparison["metrics"].setdefault(evaluator_name, {})[metric_name] = {
                "current": current_value,
                "baseline": baseline_value,
                "diff": diff,
                "pct_change": pct_change,
            }

            if diff is None:
                continue
            if diff < -tolerance:
                comparison["regressions"].append(
                    {
                        "evaluator": evaluator_name,
                        "metric": metric_name,
                        "current": current_value,
                        "baseline": baseline_value,
                        "diff": diff,
                    }
                )
            elif diff > tolerance:
                comparison["improvements"].append(
                    {
                        "evaluator": evaluator_name,
                        "metric": metric_name,
                        "current": current_value,
                        "baseline": baseline_value,
                        "diff": diff,
                    }
                )

    comparison["has_regressions"] = len(comparison["regressions"]) > 0

    return comparison


# =============================================================================
# Stage-Based Evaluation (New Architecture)
# =============================================================================


@dataclass
class StageEvaluationReport:
    """Complete evaluation report for a pipeline run.

    Pass/fail is determined by whether ALL stages pass their individual
    thresholds. No composite scoring - each stage stands on its own.

    """

    run_id: int | None = None
    timestamp: str = ""
    passed: bool = False
    stage_results: dict[str, StageEvaluationResult] = field(default_factory=dict)
    failed_stages: list[str] = field(default_factory=list)
    duration_seconds: float = 0.0
    thresholds_config: str | None = None

    @property
    def num_passed(self) -> int:
        """Number of stages that passed."""
        return sum(1 for r in self.stage_results.values() if r.passed)

    @property
    def num_failed(self) -> int:
        """Number of stages that failed."""
        return len(self.failed_stages)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "run_id": self.run_id,
            "timestamp": self.timestamp,
            "passed": self.passed,
            "num_stages": len(self.stage_results),
            "num_passed": self.num_passed,
            "num_failed": self.num_failed,
            "failed_stages": self.failed_stages,
            "duration_seconds": self.duration_seconds,
            "thresholds_config": self.thresholds_config,
            "stages": {name: result.to_dict() for name, result in self.stage_results.items()},
        }


def load_thresholds_config(config_path: Path | None = None) -> dict[str, Any]:
    """Load evaluation thresholds from YAML config.

    Parameters
    ----------
    config_path : Optional[Path], optional
        Path to config file, defaults to config/evaluation_thresholds.yaml
    """
    return load_thresholds_config_helper(config_path)


def create_stage_evaluators() -> dict[PipelineStage, Any]:
    """Create stage evaluators for all pipeline stages."""
    return {stage: get_evaluator(stage) for stage in PipelineStage.core_stages()}


def run_stage_evaluation(
    stage_data: dict[str, dict[str, Any]],
    thresholds_path: Path | None = None,
    run_id: int | None = None,
) -> StageEvaluationReport:
    """Run stage-based evaluation on pipeline data.

        Each stage is evaluated independently against its thresholds.
        The pipeline passes only if ALL stages pass.

    Parameters
    ----------
    stage_data : Dict[str, Dict[str, Any]]
        Mapping stage name to stage metrics/data
    thresholds_path : Optional[Path], optional
        Path to thresholds config
    run_id : Optional[int], optional
        Pipeline run ID
    """
    start_time = time.time()
    evaluators = create_stage_evaluators()

    # Run evaluators for each stage
    stage_results: dict[str, StageEvaluationResult] = {}
    failed_stages: list[str] = []

    for stage_name, data in stage_data.items():
        try:
            stage = PipelineStage(stage_name)
        except ValueError:
            logger.warning("Unknown stage: %s, skipping", stage_name)
            continue

        evaluator = evaluators.get(stage)
        if evaluator is None:
            logger.warning("No evaluator for stage: %s", stage_name)
            continue

        result = evaluator.evaluate(data)
        stage_results[stage_name] = result

        if not result.passed:
            failed_stages.append(stage_name)

    # Pipeline passes only if ALL stages pass
    all_passed = len(failed_stages) == 0 and len(stage_results) > 0
    duration = time.time() - start_time

    report = StageEvaluationReport(
        run_id=run_id,
        timestamp=datetime.now().isoformat(),
        passed=all_passed,
        stage_results=stage_results,
        failed_stages=failed_stages,
        duration_seconds=duration,
        thresholds_config=str(thresholds_path) if thresholds_path else None,
    )

    if all_passed:
        logger.info(
            "Stage evaluation PASSED: %d/%d stages, %.2fs",
            len(stage_results),
            len(stage_results),
            duration,
        )
    else:
        logger.warning(
            "Stage evaluation FAILED: %d/%d stages failed (%s), %.2fs",
            len(failed_stages),
            len(stage_results),
            ", ".join(failed_stages),
            duration,
        )

    return report


def run_full_pipeline_evaluation(
    run_id: int,
    thresholds_path: Path | None = None,
    store_results: bool = True,
) -> StageEvaluationReport:
    """Run full pipeline evaluation on a completed run.

        This function:
        1. Loads stage data from the pipeline database
        2. Runs all stage evaluators
        3. Computes stage pass rate and grade
        4. Optionally stores results in evaluation database

    Parameters
    ----------
    run_id : int
        Pipeline run ID to evaluate
    thresholds_path : Optional[Path], optional
        Path to thresholds config
    store_results : bool, optional
        Whether to store results in evaluation database (default True)
    """
    # Load data from pipeline database
    thresholds = load_thresholds_config(thresholds_path)

    # Convert run_id to stage data dict by loading from pipeline DB
    # This is a placeholder - actual implementation would query the
    # unified database for stage outputs and metrics
    stage_data = _load_run_stage_data(run_id)

    report = run_stage_evaluation(
        stage_data=stage_data,
        thresholds_path=thresholds_path,
        run_id=run_id,
    )

    # Store in evaluation database
    if store_results:
        try:
            from .database import get_evaluation_db

            eval_db = get_evaluation_db()
            db_run_id = eval_db.start_run(
                dataset_name=f"pipeline_run_{run_id}",
                config_snapshot=thresholds,
            )

            for result in report.stage_results.values():
                eval_db.store_stage_result(db_run_id, result)

            total = len(report.stage_results)
            passed = sum(1 for r in report.stage_results.values() if r.passed)
            pass_rate = passed / total if total else 0.0

            eval_db.complete_run(
                run_id=db_run_id,
                overall_score=pass_rate,  # Legacy field stores pass rate
                quality_grade="passed" if report.passed else "failed",
                passed=report.passed,
                num_stages=len(report.stage_results),
                duration_seconds=report.duration_seconds,
            )
            logger.info("Stored evaluation results in database (run_id=%d)", db_run_id)

        except (ImportError, OSError) as exc:
            logger.warning("Failed to store results in database: %s", exc)

    return report


def _load_run_stage_data(run_id: int) -> dict[str, dict[str, Any]]:
    """Load stage data for a pipeline run from the database.

        This function queries the pipeline's unified database to retrieve
        metrics and outputs for each stage of the run.

    Parameters
    ----------
    run_id : int
        Pipeline run ID
    """
    # Placeholder implementation - would query unified database
    # for actual stage metrics and outputs
    stage_data: dict[str, dict[str, Any]] = {}

    for stage in PipelineStage.core_stages():
        # In real implementation, query:
        # - Ingest: subband_files table
        # - Convert: conversion_runs, ms_files tables
        # - Calibrate: calibration_qa table
        # - Image: image_metadata table
        # - Photometry: photometry table
        # - Mosaic: mosaic_metadata table
        stage_data[stage.value] = {
            "run_id": run_id,
            "stage": stage.value,
            # Actual metrics would be loaded from DB
        }

    logger.debug("Loaded stage data for run %d: %s", run_id, list(stage_data.keys()))
    return stage_data
