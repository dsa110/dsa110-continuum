"""
Stage-specific evaluators for DSA-110 pipeline evaluation framework.

Each stage has a dedicated evaluator that knows how to:
- Extract relevant metrics from pipeline outputs
- Score metrics against stage-specific thresholds
- Determine pass/fail status per metric and stage

The evaluators use the stage taxonomy from stages.py and threshold
configuration from config/evaluation_thresholds.yaml.
"""

from __future__ import annotations

import logging
import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from .stages import (
    MetricSpec,
    PipelineStage,
    StageRegistry,
    StageSpec,
    get_registry,
)

logger = logging.getLogger(__name__)


class CheckType:
    """Enumeration of check types for MetricCheck."""

    BOOLEAN = "boolean"  # Simple true/false check (file exists, entry present)
    COUNT = "count"  # Count check (16 of 16 subbands)
    THRESHOLD = "threshold"  # Numeric threshold (value >= min or value <= max)
    RANGE = "range"  # Value within bounds (min <= value <= max)
    MATCH = "match"  # Exact value match (state == "completed")


@dataclass
class MetricCheck:
    """Result of a single metric pass/fail check.

    Supports multiple check types for semantic clarity:
    - BOOLEAN: Simple exists/not exists, valid/invalid
    - COUNT: Expected vs actual count (e.g., 16/16 subbands)
    - THRESHOLD: Numeric comparison (≥ min or ≤ max)
    - RANGE: Value within bounds
    - MATCH: Exact value comparison

    """

    name: str
    passed: bool
    check_type: str | None = None  # One of CheckType constants

    # For BOOLEAN checks
    value: Any | None = None  # The actual value (bool, float, str, etc.)

    # For COUNT checks
    expected: int | None = None
    actual: int | None = None

    # For THRESHOLD checks
    threshold: float | None = None
    comparison: str | None = None  # "gte" (>=), "lte" (<=), "eq" (==)
    threshold_type: str | None = None  # Legacy alias: "min"/"max"/"eq"

    # For RANGE checks
    min_bound: float | None = None
    max_bound: float | None = None

    # Common fields
    unit: str = ""
    message: str = ""
    required: bool = True  # If False, failure is a warning not an error

    def __post_init__(self) -> None:
        if self.comparison is None and self.threshold_type is not None:
            mapping = {"min": "gte", "max": "lte", "eq": "eq"}
            self.comparison = mapping.get(self.threshold_type, self.comparison)

        if self.check_type is None:
            if self.min_bound is not None or self.max_bound is not None:
                self.check_type = CheckType.RANGE
            elif self.threshold is not None or self.comparison is not None:
                self.check_type = CheckType.THRESHOLD
            elif self.expected is not None or self.actual is not None:
                self.check_type = CheckType.COUNT
            else:
                self.check_type = CheckType.BOOLEAN

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result: dict[str, Any] = {
            "name": self.name,
            "passed": self.passed,
            "check_type": self.check_type,
            "message": self.message,
            "required": self.required,
        }

        # Add type-specific fields
        if self.check_type == CheckType.BOOLEAN:
            result["value"] = self.value

        elif self.check_type == CheckType.COUNT:
            result["expected"] = self.expected
            result["actual"] = self.actual

        elif self.check_type == CheckType.THRESHOLD:
            result["value"] = self.value
            result["threshold"] = self.threshold
            result["comparison"] = self.comparison
            if self.unit:
                result["unit"] = self.unit

        elif self.check_type == CheckType.RANGE:
            result["value"] = self.value
            result["min_bound"] = self.min_bound
            result["max_bound"] = self.max_bound
            if self.unit:
                result["unit"] = self.unit

        elif self.check_type == CheckType.MATCH:
            result["value"] = self.value
            result["expected"] = self.expected

        return result

    # Factory methods for cleaner construction
    @classmethod
    def boolean(
        cls,
        name: str,
        value: bool,
        message: str = "",
        required: bool = True,
    ) -> MetricCheck:
        """Create a boolean check (exists/valid).

        Parameters
        ----------
        name : str
            Name of the check.
        value : bool
            Boolean value to check.
        message : str
            Message to display. (Default value = "")
        required : bool
            Whether the check is required. (Default value = True)

        """
        return cls(
            name=name,
            passed=value,
            check_type=CheckType.BOOLEAN,
            value=value,
            message=message or (f"{name}: ✓" if value else f"{name}: ✗"),
            required=required,
        )

    @classmethod
    def count(
        cls,
        name: str,
        actual: int,
        expected: int,
        message: str = "",
        required: bool = True,
    ) -> MetricCheck:
        """Create a count check (actual vs expected).

        Parameters
        ----------
        name : str
            Name of the check.
        actual : int
            Actual count value.
        expected : int
            Expected count value.
        message : str
            Message to display. (Default value = "")
        required : bool
            Whether the check is required. (Default value = True)

        """
        passed = actual >= expected
        return cls(
            name=name,
            passed=passed,
            check_type=CheckType.COUNT,
            expected=expected,
            actual=actual,
            message=message or f"{actual}/{expected}",
            required=required,
        )

    @classmethod
    def threshold_gte(
        cls,
        name: str,
        value: float,
        threshold: float,
        unit: str = "",
        message: str = "",
        required: bool = True,
    ) -> MetricCheck:
        """Create a threshold check (value >= threshold).

        Parameters
        ----------
        name : str
            Name of the check.
        value : float
            Value to check.
        threshold : float
            Threshold value.
        unit : str
            Unit of measurement. (Default value = "")
        message : str
            Message to display. (Default value = "")
        required : bool
            Whether the check is required. (Default value = True)

        """
        passed = value >= threshold
        default_msg = (
            f"{value:.4g} {unit} ✓ (≥{threshold:.4g})".strip()
            if passed
            else f"{value:.4g} {unit} < {threshold:.4g}".strip()
        )
        return cls(
            name=name,
            passed=passed,
            check_type=CheckType.THRESHOLD,
            value=value,
            threshold=threshold,
            comparison="gte",
            unit=unit,
            message=message or default_msg,
            required=required,
        )

    @classmethod
    def threshold_lte(
        cls,
        name: str,
        value: float,
        threshold: float,
        unit: str = "",
        message: str = "",
        required: bool = True,
    ) -> MetricCheck:
        """Create a threshold check (value <= threshold).

        Parameters
        ----------
        name : str
            Name of the check.
        value : float
            Value to check.
        threshold : float
            Threshold value.
        unit : str
            Unit of measurement. (Default value = "")
        message : str
            Message to display. (Default value = "")
        required : bool
            Whether the check is required. (Default value = True)

        """
        passed = value <= threshold
        default_msg = (
            f"{value:.4g} {unit} ✓ (≤{threshold:.4g})".strip()
            if passed
            else f"{value:.4g} {unit} > {threshold:.4g}".strip()
        )
        return cls(
            name=name,
            passed=passed,
            check_type=CheckType.THRESHOLD,
            value=value,
            threshold=threshold,
            comparison="lte",
            unit=unit,
            message=message or default_msg,
            required=required,
        )

    @classmethod
    def in_range(
        cls,
        name: str,
        value: float,
        min_bound: float,
        max_bound: float,
        unit: str = "",
        message: str = "",
        required: bool = True,
    ) -> MetricCheck:
        """Create a range check (min <= value <= max).

        Parameters
        ----------
        name : str
            Name of the check.
        value : float
            Value to check.
        min_bound : float
            Minimum bound.
        max_bound : float
            Maximum bound.
        unit : str
            Unit of measurement. (Default value = "")
        message : str
            Message to display. (Default value = "")
        required : bool
            Whether the check is required. (Default value = True)

        """
        passed = min_bound <= value <= max_bound
        default_msg = (
            f"{value:.4g} {unit} ✓ ({min_bound:.4g}-{max_bound:.4g})".strip()
            if passed
            else f"{value:.4g} {unit} outside [{min_bound:.4g}, {max_bound:.4g}]".strip()
        )
        return cls(
            name=name,
            passed=passed,
            check_type=CheckType.RANGE,
            value=value,
            min_bound=min_bound,
            max_bound=max_bound,
            unit=unit,
            message=message or default_msg,
            required=required,
        )

    @classmethod
    def match(
        cls,
        name: str,
        value: Any,
        expected: Any,
        message: str = "",
        required: bool = True,
    ) -> MetricCheck:
        """Create a match check (value == expected).

        Parameters
        ----------
        name : str
            Name of the check.
        value : Any
            Value to check.
        expected : Any
            Expected value.
        message : str
            Message to display. (Default value = "")
        required : bool
            Whether the check is required. (Default value = True)

        """
        passed = value == expected
        return cls(
            name=name,
            passed=passed,
            check_type=CheckType.MATCH,
            value=value,
            expected=expected,
            message=message
            or (f"{name}: {value} ✓" if passed else f"{name}: {value} ≠ {expected}"),
            required=required,
        )


@dataclass
class StageEvaluationResult:
    """Complete result of evaluating a pipeline stage.

    A stage passes only if ALL required metrics pass their individual checks.
    No composite scoring - just clear pass/fail at each level.

    """

    stage: PipelineStage
    passed: bool
    checks: list[MetricCheck] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def failed_checks(self) -> list[MetricCheck]:
        """ """
        return [c for c in self.checks if not c.passed]

    @property
    def required_failures(self) -> list[MetricCheck]:
        """ """
        return [c for c in self.checks if not c.passed and c.required]

    @property
    def num_passed(self) -> int:
        """Number of checks that passed."""
        return sum(1 for c in self.checks if c.passed)

    @property
    def num_failed(self) -> int:
        """Number of checks that failed."""
        return sum(1 for c in self.checks if not c.passed)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "stage": self.stage.value,
            "passed": self.passed,
            "num_checks": len(self.checks),
            "num_passed": self.num_passed,
            "num_failed": self.num_failed,
            "checks": [c.to_dict() for c in self.checks],
            "failed_checks": [c.name for c in self.failed_checks],
            "errors": self.errors,
            "warnings": self.warnings,
        }


class BaseStageEvaluator(ABC):
    """Abstract base class for stage-specific evaluators.

    Subclasses must implement extract_metrics() to pull
    relevant metrics from pipeline outputs.

    """

    def __init__(
        self,
        registry: StageRegistry | None = None,
    ):
        """Initialize evaluator.

        Parameters
        ----------
        registry : StageRegistry
            Stage registry for threshold lookup.
        """
        self._registry = registry or get_registry()

    @property
    @abstractmethod
    def stage(self) -> PipelineStage:
        """The pipeline stage this evaluator handles."""

    @property
    def spec(self) -> StageSpec:
        """Get the stage specification."""
        return self._registry.get_stage(self.stage)

    @property
    @abstractmethod
    def extract_metrics(
        self,
        data: dict[str, Any],
        context: dict[str, Any] | None = None,
    ) -> dict[str, float]:
        """Extract metric values from pipeline output data.

        Parameters
        ----------
        data : Dict[str, Any]
            Pipeline output data (DB records, file metadata, etc.).
        context : Optional[Dict[str, Any]]
            Additional context (reference data, expected values). (Default value = None)

        """

    def evaluate(
        self,
        data: dict[str, Any],
        context: dict[str, Any] | None = None,
    ) -> StageEvaluationResult:
        """Evaluate pipeline output against stage thresholds.

            Each metric is checked against its threshold(s) for pass/fail.
            Stage passes only if ALL required metrics pass.

        Parameters
        ----------
        data : dict
            Pipeline output data to evaluate.
        context : dict, optional
            Additional context for evaluation (default is None).

        """
        errors: list[str] = []
        warnings: list[str] = []

        # Extract metrics from data
        try:
            metric_values = self.extract_metrics(data, context)
        except (KeyError, TypeError, ValueError) as exc:
            logger.exception("Failed to extract metrics for %s", self.stage.value)
            return StageEvaluationResult(
                stage=self.stage,
                passed=False,
                errors=[f"Metric extraction failed: {exc}"],
            )

        # Check each metric against its threshold
        checks: list[MetricCheck] = []
        all_required_passed = True

        for metric_spec in self.spec.metrics:
            value = metric_values.get(metric_spec.name)
            required = metric_spec.weight >= 0.5  # High weight = required

            if value is None:
                # Missing value = check fails
                check = MetricCheck.boolean(
                    name=metric_spec.name,
                    value=False,
                    message="Value not available",
                    required=required,
                )
                warnings.append(f"Missing value for metric: {metric_spec.name}")
            else:
                # Evaluate against threshold - use appropriate check type
                check = self._create_metric_check(metric_spec, value, required)

            checks.append(check)

            if check.required and not check.passed:
                all_required_passed = False

        # Stage passes only if all required checks pass and no errors
        stage_passed = all_required_passed and not errors

        return StageEvaluationResult(
            stage=self.stage,
            passed=stage_passed,
            checks=checks,
            errors=errors,
            warnings=warnings,
        )

    def _create_metric_check(
        self,
        spec: MetricSpec,
        value: float,
        required: bool,
    ) -> MetricCheck:
        """Create appropriate MetricCheck based on MetricSpec thresholds.

        Parameters
        ----------
        spec : MetricSpec
            Metric specification containing thresholds.
        value : float
            Value to check against thresholds.
        required : bool
            Whether the metric is required.

        """
        # Range check (both min and max)
        if spec.min_value is not None and spec.max_value is not None:
            return MetricCheck.in_range(
                name=spec.name,
                value=value,
                min_bound=spec.min_value,
                max_bound=spec.max_value,
                unit=spec.unit,
                required=required,
            )

        # Minimum threshold
        if spec.min_value is not None:
            return MetricCheck.threshold_gte(
                name=spec.name,
                value=value,
                threshold=spec.min_value,
                unit=spec.unit,
                required=required,
            )

        # Maximum threshold
        if spec.max_value is not None:
            return MetricCheck.threshold_lte(
                name=spec.name,
                value=value,
                threshold=spec.max_value,
                unit=spec.unit,
                required=required,
            )

        # Target value (exact match)
        if spec.target_value is not None:
            return MetricCheck.match(
                name=spec.name,
                value=value,
                expected=spec.target_value,
                required=required,
            )

        # No threshold defined - just record the value as passing
        return MetricCheck.boolean(
            name=spec.name,
            value=True,
            message=f"{spec.name}: {value:.4g} {spec.unit}".strip(),
            required=required,
        )


class IngestEvaluator(BaseStageEvaluator):
    """Evaluator for the ingest stage (UVH5 file discovery and registration).

        The ingest stage is responsible for:
        1. File Discovery - Finding HDF5/UVH5 files in incoming directory
        2. File Validation - Checking readability, format, naming convention
        3. Database Population - Recording files in hdf5_files and processing_queue
        4. Group Formation - Clustering 16 subbands into observation groups

        Granular Checks
    ---------------

        File System:
        - file_exists: All expected subband files found on disk
        - file_readable: All files have read permissions
        - filename_valid: Filenames match expected pattern
        - hdf5_header_valid: HDF5 headers readable with required metadata

        Database:
        - db_file_records: Records exist in hdf5_files table
        - db_queue_entry: Group registered in processing_queue
        - db_subband_records: Subband records in PostgreSQL (if applicable)

        Quality Metrics:
        - subband_completeness: All 16 subbands present (required)
        - timestamp_clustering: Files within time tolerance (required)
        - file_size_consistency: Files have consistent sizes (warning)
    """

    # Expected number of subbands per observation group
    EXPECTED_SUBBANDS = 16
    # Time tolerance for grouping subbands (seconds)
    CLUSTER_TOLERANCE_S = 120.0
    # Expected HDF5 file extension patterns
    VALID_EXTENSIONS = {".hdf5", ".uvh5", ".h5"}
    # Filename pattern: YYYY-MM-DDTHH:MM:SS_sbNN.hdf5
    FILENAME_PATTERN = r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}_sb\d{2}\.(hdf5|uvh5|h5)$"

    @property
    def stage(self) -> PipelineStage:
        return PipelineStage.INGEST

    def extract_metrics(
        self,
        data: dict[str, Any],
        context: dict[str, Any] | None = None,
    ) -> dict[str, float]:
        """Extract ingest metrics from subband group data.

            Expected data keys:
            File System Checks:
            - file_paths: List of file paths to validate
            - files_exist: Boolean or count of files that exist
            - files_readable: Boolean or count of readable files
            - filenames_valid: Boolean or count of valid filenames
            - hdf5_headers_valid: Boolean or count of valid headers

            Database Checks:
            - db_file_records_count: Number of records in hdf5_files
            - db_queue_entry_exists: Boolean for processing_queue entry
            - db_subband_records_count: Number of subband records (legacy)

            Quality Metrics:
            - subbands_found: Number of subbands discovered
            - subbands_expected: Number expected (default 16)
            - timestamp_spread_s: Max timestamp spread within group
            - file_sizes: List of file sizes for consistency check
            - group_id: Observation group identifier

        Parameters
        ----------
        data : dict
            Pipeline output data with file/database information.
        context : dict, optional
            Additional context such as expected values or database connections (default is None).

        """
        metrics: dict[str, float] = {}
        ctx = context or {}
        expected_subbands = data.get(
            "subbands_expected", ctx.get("expected_subbands", self.EXPECTED_SUBBANDS)
        )

        # -----------------------------------------------------------------
        # File System Checks (convert to fractions where applicable)
        # -----------------------------------------------------------------
        self._extract_file_existence_metrics(metrics, data, expected_subbands)
        self._extract_file_readability_metrics(metrics, data, expected_subbands)
        self._extract_filename_validity_metrics(metrics, data, expected_subbands)
        self._extract_hdf5_header_metrics(metrics, data, expected_subbands)

        # -----------------------------------------------------------------
        # Database Checks (convert to fractions)
        # -----------------------------------------------------------------
        self._extract_db_file_record_metrics(metrics, data, expected_subbands)
        self._extract_db_queue_metrics(metrics, data)
        self._extract_db_subband_metrics(metrics, data, expected_subbands)

        # -----------------------------------------------------------------
        # Quality Metrics
        # -----------------------------------------------------------------
        self._extract_subband_completeness(metrics, data, expected_subbands)
        self._extract_timestamp_clustering(metrics, data)
        self._extract_file_size_consistency(metrics, data)

        return metrics

    def _extract_file_existence_metrics(
        self,
        metrics: dict[str, float],
        data: dict[str, Any],
        expected: int,
    ) -> None:
        """Extract file existence check metrics.

        Parameters
        ----------
        metrics : dict
            Dictionary to store metric values.
        data : dict
            Input data containing file existence information.
        expected : int
            Expected number of files.

        """
        # Accept boolean, count, or list of paths
        if "files_exist" in data:
            val = data["files_exist"]
            if isinstance(val, bool):
                metrics["file_exists"] = 1.0 if val else 0.0
            elif isinstance(val, (int, float)):
                metrics["file_exists"] = val / expected if expected > 0 else 0.0
        elif "file_paths" in data:
            paths = data["file_paths"]
            if paths:
                exist_count = sum(1 for p in paths if os.path.exists(p))
                metrics["file_exists"] = exist_count / len(paths)

    def _extract_file_readability_metrics(
        self,
        metrics: dict[str, float],
        data: dict[str, Any],
        expected: int,
    ) -> None:
        """Extract file readability check metrics.

        Parameters
        ----------
        metrics : dict
            Dictionary to store metric values.
        data : dict
            Input data containing file readability information.
        expected : int
            Expected number of files.

        """
        if "files_readable" in data:
            val = data["files_readable"]
            if isinstance(val, bool):
                metrics["file_readable"] = 1.0 if val else 0.0
            elif isinstance(val, (int, float)):
                metrics["file_readable"] = val / expected if expected > 0 else 0.0
        elif "file_paths" in data:
            paths = data["file_paths"]
            if paths:
                readable_count = sum(
                    1 for p in paths if os.path.exists(p) and os.access(p, os.R_OK)
                )
                metrics["file_readable"] = readable_count / len(paths)

    def _extract_filename_validity_metrics(
        self,
        metrics: dict[str, float],
        data: dict[str, Any],
        expected: int,
    ) -> None:
        """Extract filename pattern validity metrics.

        Parameters
        ----------
        metrics : dict
            Dictionary to store metric values.
        data : dict
            Input data containing filename validity information.
        expected : int
            Expected number of files.

        """
        if "filenames_valid" in data:
            val = data["filenames_valid"]
            if isinstance(val, bool):
                metrics["filename_valid"] = 1.0 if val else 0.0
            elif isinstance(val, (int, float)):
                metrics["filename_valid"] = val / expected if expected > 0 else 0.0
        elif "file_paths" in data:
            paths = data["file_paths"]
            if paths:
                pattern = re.compile(self.FILENAME_PATTERN)
                valid_count = sum(1 for p in paths if pattern.match(os.path.basename(p)))
                metrics["filename_valid"] = valid_count / len(paths)

    def _extract_hdf5_header_metrics(
        self,
        metrics: dict[str, float],
        data: dict[str, Any],
        expected: int,
    ) -> None:
        """Extract HDF5 header validity metrics.

        Parameters
        ----------
        metrics : dict
            Dictionary to store metric values.
        data : dict
            Input data containing HDF5 header validity information.
        expected : int
            Expected number of files.

        """
        if "hdf5_headers_valid" in data:
            val = data["hdf5_headers_valid"]
            if isinstance(val, bool):
                metrics["hdf5_header_valid"] = 1.0 if val else 0.0
            elif isinstance(val, (int, float)):
                metrics["hdf5_header_valid"] = val / expected if expected > 0 else 0.0
        # Note: Actual HDF5 validation requires h5py and is expensive
        # Usually pre-computed by ingestion tasks

    def _extract_db_file_record_metrics(
        self,
        metrics: dict[str, float],
        data: dict[str, Any],
        expected: int,
    ) -> None:
        """Extract database file record metrics (hdf5_files table).

        Parameters
        ----------
        metrics : dict
            Dictionary to store metric values.
        data : dict
            Input data containing database file record information.
        expected : int
            Expected number of records.

        """
        if "db_file_records_count" in data:
            count = data["db_file_records_count"]
            metrics["db_records_created"] = count / expected if expected > 0 else 0.0
        elif "db_file_records_exist" in data:
            metrics["db_records_created"] = 1.0 if data["db_file_records_exist"] else 0.0

    def _extract_db_queue_metrics(
        self,
        metrics: dict[str, float],
        data: dict[str, Any],
    ) -> None:
        """Extract processing queue entry metrics.

        Parameters
        ----------
        metrics : dict
            Dictionary to store metric values.
        data : dict
            Input data containing processing queue information.

        """
        if "db_queue_entry_exists" in data:
            metrics["db_queue_entry"] = 1.0 if data["db_queue_entry_exists"] else 0.0
        elif "queue_state" in data:
            # Any state other than None means entry exists
            metrics["db_queue_entry"] = 1.0 if data["queue_state"] else 0.0

    def _extract_db_subband_metrics(
        self,
        metrics: dict[str, float],
        data: dict[str, Any],
        expected: int,
    ) -> None:
        """Extract legacy PostgreSQL subband record metrics.

        Parameters
        ----------
        metrics : dict
            Dictionary to store metric values.
        data : dict
            Input data containing subband record information.
        expected : int
            Expected number of subband records.

        """
        if "db_subband_records_count" in data:
            count = data["db_subband_records_count"]
            metrics["db_subband_records"] = count / expected if expected > 0 else 0.0

    def _extract_subband_completeness(
        self,
        metrics: dict[str, float],
        data: dict[str, Any],
        expected: int,
    ) -> None:
        """Extract subband completeness metric.

        Parameters
        ----------
        metrics : dict
            Dictionary to store metric values.
        data : dict
            Input data containing subband completeness information.
        expected : int
            Expected number of subbands.

        """
        found = data.get("subbands_found", 0)
        if expected > 0:
            metrics["subband_completeness"] = found / expected

    def _extract_timestamp_clustering(
        self,
        metrics: dict[str, float],
        data: dict[str, Any],
    ) -> None:
        """Extract timestamp clustering metric.

        Parameters
        ----------
        metrics : dict
            Dictionary to store metric values.
        data : dict
            Input data containing timestamp information.

        """
        timestamp_spread = data.get("timestamp_spread_s")
        if timestamp_spread is not None:
            metrics["timestamp_clustering"] = timestamp_spread

    def _extract_file_size_consistency(
        self,
        metrics: dict[str, float],
        data: dict[str, Any],
    ) -> None:
        """Extract file size consistency metric.

        Parameters
        ----------
        metrics : dict
            Dictionary to store metric values.
        data : dict
            Input data containing file size information.

        """
        file_sizes = [s for s in data.get("file_sizes", []) if s is not None]
        if file_sizes and len(file_sizes) > 1:
            median_size = sorted(file_sizes)[len(file_sizes) // 2]
            if median_size > 0:
                # Files within 10% of median are considered consistent
                consistent = sum(1 for s in file_sizes if 0.9 <= s / median_size <= 1.1)
                metrics["file_size_consistency"] = consistent / len(file_sizes)

    def evaluate(
        self,
        data: dict[str, Any],
        context: dict[str, Any] | None = None,
    ) -> StageEvaluationResult:
        """Evaluate ingest stage with granular pass/fail checks.

            Uses explicit check types:
            - BOOLEAN: File exists, readable, valid filename, HDF5 header valid
            - COUNT: Subband completeness, DB record counts
            - THRESHOLD: Timestamp clustering, file size consistency

        Parameters
        ----------
        data : dict
            Ingest output data to evaluate.
        context : dict, optional
            Additional context for evaluation (default is None).

        """
        errors: list[str] = []
        warnings: list[str] = []
        ctx = context or {}
        expected = data.get(
            "subbands_expected", ctx.get("expected_subbands", self.EXPECTED_SUBBANDS)
        )

        # Extract raw data for checks
        try:
            raw = self._extract_raw_values(data, ctx, expected)
        except (KeyError, TypeError, ValueError) as exc:
            logger.exception("Failed to extract ingest data")
            return StageEvaluationResult(
                stage=self.stage,
                passed=False,
                errors=[f"Data extraction failed: {exc}"],
            )

        checks: list[MetricCheck] = []

        # File System Checks (BOOLEAN)
        checks.extend(self._create_file_system_checks(raw))

        # Database Checks (BOOLEAN and COUNT)
        checks.extend(self._create_database_checks(raw, expected))

        # Quality Checks (COUNT and THRESHOLD)
        checks.extend(self._create_quality_checks(raw, expected))

        # Determine pass/fail
        all_required_passed = all(c.passed for c in checks if c.required)
        warnings.extend(f"{c.name}: {c.message}" for c in checks if not c.required and not c.passed)

        return StageEvaluationResult(
            stage=self.stage,
            passed=all_required_passed and not errors,
            checks=checks,
            errors=errors,
            warnings=warnings,
        )

    def _extract_raw_values(
        self,
        data: dict[str, Any],
        ctx: dict[str, Any],
        expected: int,
    ) -> dict[str, Any]:
        """Extract raw values from input data for check creation.

        Parameters
        ----------
        data : dict
            Input data containing raw values.
        ctx : dict
            Context dictionary.
        expected : int
            Expected count or value.

        """
        raw: dict[str, Any] = {"expected_subbands": expected}

        # File paths for auto-validation
        if "file_paths" in data:
            paths = data["file_paths"]
            raw["file_paths"] = paths
            raw["files_exist_count"] = sum(1 for p in paths if os.path.exists(p))
            raw["files_readable_count"] = sum(
                1 for p in paths if os.path.exists(p) and os.access(p, os.R_OK)
            )
            pattern = re.compile(self.FILENAME_PATTERN)
            raw["filenames_valid_count"] = sum(
                1 for p in paths if pattern.match(os.path.basename(p))
            )
            raw["total_files"] = len(paths)

        # Pre-computed boolean/count values
        for key in ["files_exist", "files_readable", "filenames_valid", "hdf5_headers_valid"]:
            if key in data:
                raw[key] = data[key]

        # Database values
        raw["db_file_records_count"] = data.get("db_file_records_count", 0)
        raw["db_queue_entry_exists"] = data.get("db_queue_entry_exists", False)
        raw["db_subband_records_count"] = data.get("db_subband_records_count")

        # Quality metrics
        raw["subbands_found"] = data.get("subbands_found", 0)
        raw["timestamp_spread_s"] = data.get("timestamp_spread_s")
        raw["file_sizes"] = data.get("file_sizes", [])

        return raw

    def _create_file_system_checks(self, raw: dict[str, Any]) -> list[MetricCheck]:
        """Create file system boolean checks.

        Parameters
        ----------
        raw : Dict[str, Any]
            Raw input data.

        """
        checks: list[MetricCheck] = []
        total = raw.get("total_files", raw["expected_subbands"])

        # File exists check
        if "files_exist" in raw:
            val = raw["files_exist"]
            passed = val if isinstance(val, bool) else val >= total
            checks.append(
                MetricCheck.boolean(
                    "file_exists", passed, "All expected files exist on disk", required=True
                )
            )
        elif "files_exist_count" in raw:
            actual = raw["files_exist_count"]
            checks.append(
                MetricCheck.count(
                    "file_exists", actual, total, f"{actual}/{total} files found", required=True
                )
            )

        # File readable check
        if "files_readable" in raw:
            val = raw["files_readable"]
            passed = val if isinstance(val, bool) else val >= total
            checks.append(
                MetricCheck.boolean(
                    "file_readable", passed, "All files have read permissions", required=True
                )
            )
        elif "files_readable_count" in raw:
            actual = raw["files_readable_count"]
            checks.append(
                MetricCheck.count(
                    "file_readable",
                    actual,
                    total,
                    f"{actual}/{total} files readable",
                    required=True,
                )
            )

        # Filename validity check
        if "filenames_valid" in raw:
            val = raw["filenames_valid"]
            passed = val if isinstance(val, bool) else val >= total
            checks.append(
                MetricCheck.boolean(
                    "filename_valid", passed, "All filenames match expected pattern", required=True
                )
            )
        elif "filenames_valid_count" in raw:
            actual = raw["filenames_valid_count"]
            checks.append(
                MetricCheck.count(
                    "filename_valid",
                    actual,
                    total,
                    f"{actual}/{total} valid filenames",
                    required=True,
                )
            )

        # HDF5 header validity
        if "hdf5_headers_valid" in raw:
            val = raw["hdf5_headers_valid"]
            passed = val if isinstance(val, bool) else val >= total
            checks.append(
                MetricCheck.boolean(
                    "hdf5_header_valid", passed, "All HDF5 headers readable", required=True
                )
            )

        return checks

    def _create_database_checks(self, raw: dict[str, Any], expected: int) -> list[MetricCheck]:
        """Create database boolean/count checks.

        Parameters
        ----------
        raw : Dict[str, Any]
            Raw input data.
        expected : int
            Expected count value.

        """
        checks: list[MetricCheck] = []

        # DB file records (COUNT)
        db_count = raw.get("db_file_records_count", 0)
        if db_count > 0 or "db_file_records_count" in raw:
            checks.append(
                MetricCheck.count(
                    "db_records_created",
                    db_count,
                    expected,
                    f"{db_count}/{expected} records in hdf5_files",
                    required=True,
                )
            )

        # Queue entry exists (BOOLEAN)
        queue_exists = raw.get("db_queue_entry_exists", False)
        checks.append(
            MetricCheck.boolean(
                "db_queue_entry",
                queue_exists,
                "Group registered in processing_queue",
                required=True,
            )
        )

        # Legacy subband records (COUNT, optional)
        subband_count = raw.get("db_subband_records_count")
        if subband_count is not None:
            checks.append(
                MetricCheck.count(
                    "db_subband_records",
                    subband_count,
                    expected,
                    f"{subband_count}/{expected} legacy records",
                    required=False,
                )
            )

        return checks

    def _create_quality_checks(self, raw: dict[str, Any], expected: int) -> list[MetricCheck]:
        """Create quality metric checks (COUNT and THRESHOLD).

        Parameters
        ----------
        raw : Dict[str, Any]
            Raw input data.
        expected : int
            Expected count value.

        """
        checks: list[MetricCheck] = []

        # Subband completeness (COUNT)
        found = raw.get("subbands_found", 0)
        checks.append(
            MetricCheck.count(
                "subband_completeness",
                found,
                expected,
                f"{found}/{expected} subbands",
                required=True,
            )
        )

        # Timestamp clustering (THRESHOLD, max)
        ts_spread = raw.get("timestamp_spread_s")
        if ts_spread is not None:
            checks.append(
                MetricCheck.threshold_lte(
                    "timestamp_clustering",
                    ts_spread,
                    self.CLUSTER_TOLERANCE_S,
                    unit="s",
                    required=True,
                )
            )

        # File size consistency (THRESHOLD, min percentage)
        file_sizes = [s for s in raw.get("file_sizes", []) if s is not None]
        if file_sizes and len(file_sizes) > 1:
            median_size = sorted(file_sizes)[len(file_sizes) // 2]
            if median_size > 0:
                consistent = sum(1 for s in file_sizes if 0.9 <= s / median_size <= 1.1)
                ratio = consistent / len(file_sizes)
                checks.append(
                    MetricCheck.threshold_gte(
                        "file_size_consistency",
                        ratio,
                        0.9,
                        message=f"{ratio:.0%} files within 10% of median size",
                        required=False,
                    )
                )

        return checks


class ConversionEvaluator(BaseStageEvaluator):
    """Evaluator for the convert stage (UVH5 to MS)."""

    # DSA-110 defaults
    EXPECTED_ANTENNAS: int = 63
    EXPECTED_SPW: int = 16
    MIN_UV_COVERAGE: float = 0.7
    MIN_ANTENNA_FRACTION: float = 0.95

    @property
    def stage(self) -> PipelineStage:
        return PipelineStage.CONVERT

    def extract_metrics(
        self,
        data: dict[str, Any],
        context: dict[str, Any] | None = None,
    ) -> dict[str, float]:
        """Extract conversion metrics from MS metadata.

            Expected data keys:
            - ms_valid: Whether MS passes structural validation
            - n_antennas: Number of antennas in MS
            - n_spw: Number of spectral windows
            - time_range_valid: Whether time range is correct
            - uv_coverage: UV plane coverage metric (0-1)

        Parameters
        ----------
        data : Dict[str, Any]
            Input data containing MS metadata.
        context : Optional[Dict[str, Any]], optional
            Additional context information, by default None

        """
        metrics: dict[str, float] = {}
        ctx = context or {}

        # MS structure validity (boolean -> 1.0/0.0)
        if "ms_valid" in data:
            metrics["ms_structure_valid"] = 1.0 if data["ms_valid"] else 0.0

        # Antenna completeness
        n_antennas = data.get("n_antennas", 0)
        expected_antennas = ctx.get("expected_antennas", self.EXPECTED_ANTENNAS)
        if expected_antennas > 0:
            metrics["antenna_completeness"] = min(1.0, n_antennas / expected_antennas)

        # Spectral window completeness
        n_spw = data.get("n_spw", 0)
        expected_spw = ctx.get("expected_spw", self.EXPECTED_SPW)
        if expected_spw > 0:
            metrics["spw_completeness"] = min(1.0, n_spw / expected_spw)

        # Time range validity (boolean -> 1.0/0.0)
        if "time_range_valid" in data:
            metrics["time_range_valid"] = 1.0 if data["time_range_valid"] else 0.0

        # UV coverage score
        if "uv_coverage" in data:
            metrics["uv_coverage_score"] = data["uv_coverage"]

        return metrics

    def evaluate(
        self,
        data: dict[str, Any],
        context: dict[str, Any] | None = None,
    ) -> StageEvaluationResult:
        """Evaluate conversion stage with granular pass/fail checks.

            Uses explicit check types:
            - BOOLEAN: MS structure valid, time range valid, MS file exists
            - COUNT: Antenna completeness, SPW completeness
            - THRESHOLD: UV coverage score

        Parameters
        ----------
        data : Dict[str, Any]
            Conversion output data to evaluate.
        context : Optional[Dict[str, Any]], optional
            Additional context (expected_antennas, expected_spw), by default None

        """
        errors: list[str] = []
        warnings: list[str] = []
        ctx = context or {}

        expected_antennas = ctx.get("expected_antennas", self.EXPECTED_ANTENNAS)
        expected_spw = ctx.get("expected_spw", self.EXPECTED_SPW)

        checks: list[MetricCheck] = []

        # MS Structure Checks (BOOLEAN)
        checks.extend(self._create_structure_checks(data))

        # Completeness Checks (COUNT)
        checks.extend(self._create_completeness_checks(data, expected_antennas, expected_spw))

        # Quality Checks (THRESHOLD)
        checks.extend(self._create_uv_quality_checks(data))

        # Determine pass/fail
        all_required_passed = all(c.passed for c in checks if c.required)
        warnings.extend(f"{c.name}: {c.message}" for c in checks if not c.required and not c.passed)

        return StageEvaluationResult(
            stage=self.stage,
            passed=all_required_passed and not errors,
            checks=checks,
            errors=errors,
            warnings=warnings,
        )

    def _create_structure_checks(self, data: dict[str, Any]) -> list[MetricCheck]:
        """Create MS structure validity checks (BOOLEAN).

        Parameters
        ----------
        data : Dict[str, Any]
            Input data for MS structure validation.

        """
        checks: list[MetricCheck] = []

        # MS file exists
        if "ms_path" in data:
            ms_exists = os.path.exists(data["ms_path"])
            checks.append(
                MetricCheck.boolean(
                    "ms_file_exists", ms_exists, "Measurement Set file exists", required=True
                )
            )

        # MS structure validity
        if "ms_valid" in data:
            checks.append(
                MetricCheck.boolean(
                    "ms_structure_valid",
                    data["ms_valid"],
                    "MS passes structural validation",
                    required=True,
                )
            )

        # Time range validity
        if "time_range_valid" in data:
            checks.append(
                MetricCheck.boolean(
                    "time_range_valid",
                    data["time_range_valid"],
                    "Time range matches expected observation",
                    required=True,
                )
            )

        return checks

    def _create_completeness_checks(
        self, data: dict[str, Any], expected_antennas: int, expected_spw: int
    ) -> list[MetricCheck]:
        """Create completeness checks for antennas and spectral windows (COUNT).

        Parameters
        ----------
        data : Dict[str, Any]
            Input data containing completeness information.
        expected_antennas : int
            Expected number of antennas.
        expected_spw : int
            Expected number of spectral windows.

        """
        checks: list[MetricCheck] = []

        # Antenna completeness
        n_antennas = data.get("n_antennas", 0)
        min_antennas = int(expected_antennas * self.MIN_ANTENNA_FRACTION)
        checks.append(
            MetricCheck.count(
                "antenna_completeness",
                n_antennas,
                min_antennas,
                f"{n_antennas}/{expected_antennas} antennas (min {min_antennas})",
                required=True,
            )
        )

        # SPW completeness
        n_spw = data.get("n_spw", 0)
        checks.append(
            MetricCheck.count(
                "spw_completeness",
                n_spw,
                expected_spw,
                f"{n_spw}/{expected_spw} spectral windows",
                required=True,
            )
        )

        return checks

    def _create_uv_quality_checks(self, data: dict[str, Any]) -> list[MetricCheck]:
        """Create UV coverage quality checks (THRESHOLD).

        Parameters
        ----------
        data : Dict[str, Any]
            Input data containing UV coverage metrics.

        """
        checks: list[MetricCheck] = []

        # UV coverage score
        if "uv_coverage" in data:
            checks.append(
                MetricCheck.threshold_gte(
                    "uv_coverage_score",
                    data["uv_coverage"],
                    self.MIN_UV_COVERAGE,
                    message=f"UV coverage {data['uv_coverage']:.2f} (min {self.MIN_UV_COVERAGE})",
                    required=False,
                )
            )

        return checks


class CalibrationEvaluator(BaseStageEvaluator):
    """Evaluator for the calibrate stage (bandpass, delay, gain)."""

    # Calibration quality thresholds
    MIN_DELAY_SNR: float = 10.0
    MAX_DELAY_NS: float = 100.0
    MIN_BP_SNR: float = 20.0
    MAX_BP_FLAGGED: float = 0.3
    MAX_BP_PHASE_SCATTER: float = 30.0  # degrees
    MIN_GAIN_SNR: float = 10.0
    MAX_GAIN_PHASE_SCATTER: float = 20.0  # degrees
    MAX_FLAG_FRACTION: float = 0.5
    FLUX_SCALE_RANGE: tuple = (0.9, 1.1)

    @property
    def stage(self) -> PipelineStage:
        return PipelineStage.CALIBRATE

    def extract_metrics(
        self,
        data: dict[str, Any],
        context: dict[str, Any] | None = None,
    ) -> dict[str, float]:
        """Extract calibration metrics from cal tables and QA data.

            Expected data keys:
            - delay_snr_median: Median SNR of delay solutions
            - delay_max_ns: Maximum delay value
            - bp_snr_median: Median bandpass SNR
            - bp_flagged_fraction: Fraction of BP solutions flagged
            - bp_phase_scatter_deg: Phase scatter in BP
            - gain_snr_median: Median gain SNR
            - gain_phase_scatter_deg: Gain phase scatter
            - overall_flag_fraction: Total flag fraction
            - flux_scale_factor: Applied flux scaling

        Parameters
        ----------
        data : Dict[str, Any]
            Input data containing calibration metrics.
        context : Optional[Dict[str, Any]], optional
            Additional context information, by default None

        """
        metrics: dict[str, float] = {}

        # Direct mapping for metrics that match spec names
        direct_metrics = [
            "delay_snr_median",
            "delay_max_ns",
            "bp_snr_median",
            "bp_flagged_fraction",
            "bp_phase_scatter_deg",
            "gain_snr_median",
            "gain_phase_scatter_deg",
            "overall_flag_fraction",
        ]

        for metric_name in direct_metrics:
            if metric_name in data:
                metrics[metric_name] = data[metric_name]

        # Flux scale accuracy (renamed from flux_scale_factor)
        if "flux_scale_factor" in data:
            metrics["flux_scale_accuracy"] = data["flux_scale_factor"]

        return metrics

    def evaluate(
        self,
        data: dict[str, Any],
        context: dict[str, Any] | None = None,
    ) -> StageEvaluationResult:
        """Evaluate calibration stage with granular pass/fail checks.

            Uses explicit check types:
            - BOOLEAN: Calibration tables exist, solutions applied
            - THRESHOLD: SNR values, phase scatter, flag fractions
            - RANGE: Flux scale accuracy

        Parameters
        ----------
        data : Dict[str, Any]
            Calibration output data to evaluate.
        context : Optional[Dict[str, Any]], optional
            Additional context for evaluation, by default None

        """
        errors: list[str] = []
        warnings: list[str] = []

        checks: list[MetricCheck] = []

        # Table Existence Checks (BOOLEAN)
        checks.extend(self._create_table_checks(data))

        # Delay Calibration Checks (THRESHOLD)
        checks.extend(self._create_delay_checks(data))

        # Bandpass Calibration Checks (THRESHOLD)
        checks.extend(self._create_bandpass_checks(data))

        # Gain Calibration Checks (THRESHOLD)
        checks.extend(self._create_gain_checks(data))

        # Overall Quality Checks (THRESHOLD and RANGE)
        checks.extend(self._create_overall_quality_checks(data))

        # Determine pass/fail
        all_required_passed = all(c.passed for c in checks if c.required)
        warnings.extend(f"{c.name}: {c.message}" for c in checks if not c.required and not c.passed)

        return StageEvaluationResult(
            stage=self.stage,
            passed=all_required_passed and not errors,
            checks=checks,
            errors=errors,
            warnings=warnings,
        )

    def _create_table_checks(self, data: dict[str, Any]) -> list[MetricCheck]:
        """Create calibration table existence checks (BOOLEAN).

        Parameters
        ----------
        data : Dict[str, Any]
            Input data for calibration table checks.

        """
        checks: list[MetricCheck] = []

        # Check for cal table paths
        if "delay_table_path" in data:
            exists = os.path.exists(data["delay_table_path"])
            checks.append(
                MetricCheck.boolean(
                    "delay_table_exists", exists, "Delay cal table exists", required=True
                )
            )

        if "bp_table_path" in data:
            exists = os.path.exists(data["bp_table_path"])
            checks.append(
                MetricCheck.boolean(
                    "bandpass_table_exists", exists, "Bandpass cal table exists", required=True
                )
            )

        if "gain_table_path" in data:
            exists = os.path.exists(data["gain_table_path"])
            checks.append(
                MetricCheck.boolean(
                    "gain_table_exists", exists, "Gain cal table exists", required=True
                )
            )

        # Solutions applied check
        if "solutions_applied" in data:
            checks.append(
                MetricCheck.boolean(
                    "solutions_applied",
                    data["solutions_applied"],
                    "Calibration solutions successfully applied",
                    required=True,
                )
            )

        return checks

    def _create_delay_checks(self, data: dict[str, Any]) -> list[MetricCheck]:
        """Create delay calibration quality checks (THRESHOLD).

        Parameters
        ----------
        data : Dict[str, Any]
            Input data containing delay calibration metrics.

        """
        checks: list[MetricCheck] = []

        # Delay SNR
        if "delay_snr_median" in data:
            snr = data["delay_snr_median"]
            checks.append(
                MetricCheck.threshold_gte(
                    "delay_snr_median",
                    snr,
                    self.MIN_DELAY_SNR,
                    message=f"Delay SNR {snr:.1f} (min {self.MIN_DELAY_SNR})",
                    required=True,
                )
            )

        # Max delay value
        if "delay_max_ns" in data:
            delay = data["delay_max_ns"]
            checks.append(
                MetricCheck.threshold_lte(
                    "delay_max_ns",
                    delay,
                    self.MAX_DELAY_NS,
                    unit="ns",
                    required=True,
                )
            )

        return checks

    def _create_bandpass_checks(self, data: dict[str, Any]) -> list[MetricCheck]:
        """Create bandpass calibration quality checks (THRESHOLD).

        Parameters
        ----------
        data : Dict[str, Any]
            Input data containing bandpass calibration metrics.

        """
        checks: list[MetricCheck] = []

        # BP SNR
        if "bp_snr_median" in data:
            snr = data["bp_snr_median"]
            checks.append(
                MetricCheck.threshold_gte(
                    "bp_snr_median",
                    snr,
                    self.MIN_BP_SNR,
                    message=f"Bandpass SNR {snr:.1f} (min {self.MIN_BP_SNR})",
                    required=True,
                )
            )

        # BP flagged fraction
        if "bp_flagged_fraction" in data:
            flagged = data["bp_flagged_fraction"]
            checks.append(
                MetricCheck.threshold_lte(
                    "bp_flagged_fraction",
                    flagged,
                    self.MAX_BP_FLAGGED,
                    message=f"BP flagged {flagged:.1%} (max {self.MAX_BP_FLAGGED:.0%})",
                    required=True,
                )
            )

        # BP phase scatter
        if "bp_phase_scatter_deg" in data:
            scatter = data["bp_phase_scatter_deg"]
            checks.append(
                MetricCheck.threshold_lte(
                    "bp_phase_scatter_deg",
                    scatter,
                    self.MAX_BP_PHASE_SCATTER,
                    unit="deg",
                    required=False,
                )
            )

        return checks

    def _create_gain_checks(self, data: dict[str, Any]) -> list[MetricCheck]:
        """Create gain calibration quality checks (THRESHOLD).

        Parameters
        ----------
        data : Dict[str, Any]
            Input data containing gain calibration metrics.

        """
        checks: list[MetricCheck] = []

        # Gain SNR
        if "gain_snr_median" in data:
            snr = data["gain_snr_median"]
            checks.append(
                MetricCheck.threshold_gte(
                    "gain_snr_median",
                    snr,
                    self.MIN_GAIN_SNR,
                    message=f"Gain SNR {snr:.1f} (min {self.MIN_GAIN_SNR})",
                    required=True,
                )
            )

        # Gain phase scatter
        if "gain_phase_scatter_deg" in data:
            scatter = data["gain_phase_scatter_deg"]
            checks.append(
                MetricCheck.threshold_lte(
                    "gain_phase_scatter_deg",
                    scatter,
                    self.MAX_GAIN_PHASE_SCATTER,
                    unit="deg",
                    required=False,
                )
            )

        return checks

    def _create_overall_quality_checks(self, data: dict[str, Any]) -> list[MetricCheck]:
        """Create overall calibration quality checks (THRESHOLD and RANGE).

        Parameters
        ----------
        data : Dict[str, Any]
            Input data containing overall calibration quality metrics.

        """
        checks: list[MetricCheck] = []

        # Overall flag fraction
        if "overall_flag_fraction" in data:
            flagged = data["overall_flag_fraction"]
            checks.append(
                MetricCheck.threshold_lte(
                    "overall_flag_fraction",
                    flagged,
                    self.MAX_FLAG_FRACTION,
                    message=f"Overall flagged {flagged:.1%} (max {self.MAX_FLAG_FRACTION:.0%})",
                    required=True,
                )
            )

        # Flux scale accuracy (RANGE check)
        if "flux_scale_factor" in data:
            scale = data["flux_scale_factor"]
            min_scale, max_scale = self.FLUX_SCALE_RANGE
            checks.append(
                MetricCheck.in_range(
                    "flux_scale_accuracy",
                    scale,
                    min_scale,
                    max_scale,
                    message=f"Flux scale {scale:.3f} (expected {min_scale}-{max_scale})",
                    required=False,
                )
            )

        return checks


class ImagingEvaluator(BaseStageEvaluator):
    """Evaluator for the image stage (FITS image creation)."""

    # Imaging quality thresholds
    MAX_RMS_NOISE_JY: float = 0.002
    TARGET_RMS_NOISE_JY: float = 0.001
    MIN_DYNAMIC_RANGE: float = 100.0
    MIN_PEAK_FLUX_JY: float = 0.001
    BEAM_RANGE_ARCSEC: tuple = (1.0, 60.0)
    TARGET_BEAM_ARCSEC: float = 15.0
    MAX_NAN_FRACTION: float = 0.01

    @property
    def stage(self) -> PipelineStage:
        return PipelineStage.IMAGE

    def extract_metrics(
        self,
        data: dict[str, Any],
        context: dict[str, Any] | None = None,
    ) -> dict[str, float]:
        """Extract imaging metrics from FITS metadata.

            Expected data keys:
            - rms_jy: RMS noise in Jy/beam
            - peak_jy: Peak flux in Jy/beam
            - beam_major_arcsec: Beam major axis
            - beam_minor_arcsec: Beam minor axis
            - nan_fraction: Fraction of NaN pixels

        Parameters
        ----------
        data : Dict[str, Any]
            Input data containing FITS metadata.
        context : Optional[Dict[str, Any]], optional
            Additional context for extraction, by default None

        Returns
        -------
            None
        """
        metrics: dict[str, float] = {}

        # RMS noise
        if "rms_jy" in data:
            metrics["rms_noise_jy"] = data["rms_jy"]

        # Peak flux
        if "peak_jy" in data:
            metrics["peak_flux_jy"] = data["peak_jy"]

        # Dynamic range (computed if both available)
        if "rms_jy" in data and "peak_jy" in data and data["rms_jy"] > 0:
            metrics["dynamic_range"] = data["peak_jy"] / data["rms_jy"]

        # Beam size
        if "beam_major_arcsec" in data:
            metrics["beam_major_arcsec"] = data["beam_major_arcsec"]
        if "beam_minor_arcsec" in data:
            metrics["beam_minor_arcsec"] = data["beam_minor_arcsec"]

        # NaN fraction
        if "nan_fraction" in data:
            metrics["nan_fraction"] = data["nan_fraction"]

        return metrics

    def evaluate(
        self,
        data: dict[str, Any],
        context: dict[str, Any] | None = None,
    ) -> StageEvaluationResult:
        """Evaluate imaging stage with granular pass/fail checks.

            Uses explicit check types:
            - BOOLEAN: FITS file exists, image is valid
            - THRESHOLD: RMS noise, dynamic range, NaN fraction
            - RANGE: Beam dimensions

        Parameters
        ----------
        data : Dict[str, Any]
            Imaging output data to evaluate.
        context : Optional[Dict[str, Any]], optional
            Additional context for evaluation, by default None

        Returns
        -------
            None
        """
        errors: list[str] = []
        warnings: list[str] = []

        checks: list[MetricCheck] = []

        # File Existence Checks (BOOLEAN)
        checks.extend(self._create_file_checks(data))

        # Noise Quality Checks (THRESHOLD)
        checks.extend(self._create_noise_checks(data))

        # Dynamic Range Check (THRESHOLD)
        checks.extend(self._create_dynamic_range_checks(data))

        # Beam Quality Checks (RANGE)
        checks.extend(self._create_beam_checks(data))

        # Data Integrity Checks (THRESHOLD)
        checks.extend(self._create_integrity_checks(data))

        # Determine pass/fail
        all_required_passed = all(c.passed for c in checks if c.required)
        warnings.extend(f"{c.name}: {c.message}" for c in checks if not c.required and not c.passed)

        return StageEvaluationResult(
            stage=self.stage,
            passed=all_required_passed and not errors,
            checks=checks,
            errors=errors,
            warnings=warnings,
        )

    def _create_file_checks(self, data: dict[str, Any]) -> list[MetricCheck]:
        """Create FITS file existence checks (BOOLEAN).

        Parameters
        ----------
        data : Dict[str, Any]
            Input data for file existence checks.

        Returns
        -------
            None
        """
        checks: list[MetricCheck] = []

        # FITS file exists
        if "fits_path" in data:
            exists = os.path.exists(data["fits_path"])
            checks.append(
                MetricCheck.boolean(
                    "fits_file_exists", exists, "FITS image file exists", required=True
                )
            )

        # PSF file exists (optional)
        if "psf_path" in data:
            exists = os.path.exists(data["psf_path"])
            checks.append(
                MetricCheck.boolean(
                    "psf_file_exists", exists, "PSF image file exists", required=False
                )
            )

        # Image validity (header readable, data accessible)
        if "image_valid" in data:
            checks.append(
                MetricCheck.boolean(
                    "image_valid",
                    data["image_valid"],
                    "FITS image is structurally valid",
                    required=True,
                )
            )

        return checks

    def _create_noise_checks(self, data: dict[str, Any]) -> list[MetricCheck]:
        """Create noise quality checks (THRESHOLD).

        Parameters
        ----------
        data : Dict[str, Any]
            Input data for noise quality checks.

        Returns
        -------
            None
        """
        checks: list[MetricCheck] = []

        # RMS noise (lower is better)
        if "rms_jy" in data:
            rms = data["rms_jy"]
            checks.append(
                MetricCheck.threshold_lte(
                    "rms_noise_jy",
                    rms,
                    self.MAX_RMS_NOISE_JY,
                    unit="Jy/beam",
                    required=True,
                )
            )

        return checks

    def _create_dynamic_range_checks(self, data: dict[str, Any]) -> list[MetricCheck]:
        """Create dynamic range checks (THRESHOLD).

        Parameters
        ----------
        data : Dict[str, Any]
            Input data for dynamic range checks.

        Returns
        -------
            None
        """
        checks: list[MetricCheck] = []

        # Compute dynamic range if not provided
        dynamic_range = data.get("dynamic_range")
        if dynamic_range is None and "rms_jy" in data and "peak_jy" in data:
            if data["rms_jy"] > 0:
                dynamic_range = data["peak_jy"] / data["rms_jy"]

        if dynamic_range is not None:
            checks.append(
                MetricCheck.threshold_gte(
                    "dynamic_range",
                    dynamic_range,
                    self.MIN_DYNAMIC_RANGE,
                    message=f"Dynamic range {dynamic_range:.0f} (min {self.MIN_DYNAMIC_RANGE})",
                    required=True,
                )
            )

        # Peak flux check
        if "peak_jy" in data:
            peak = data["peak_jy"]
            checks.append(
                MetricCheck.threshold_gte(
                    "peak_flux_jy",
                    peak,
                    self.MIN_PEAK_FLUX_JY,
                    unit="Jy/beam",
                    required=False,
                )
            )

        return checks

    def _create_beam_checks(self, data: dict[str, Any]) -> list[MetricCheck]:
        """Create synthesized beam quality checks (RANGE).

        Parameters
        ----------
        data : Dict[str, Any]
            Input data for beam quality checks.

        Returns
        -------
            None
        """
        checks: list[MetricCheck] = []
        min_beam, max_beam = self.BEAM_RANGE_ARCSEC

        # Beam major axis
        if "beam_major_arcsec" in data:
            major = data["beam_major_arcsec"]
            checks.append(
                MetricCheck.in_range(
                    "beam_major_arcsec",
                    major,
                    min_beam,
                    max_beam,
                    unit="arcsec",
                    required=False,
                )
            )

        # Beam minor axis
        if "beam_minor_arcsec" in data:
            minor = data["beam_minor_arcsec"]
            checks.append(
                MetricCheck.in_range(
                    "beam_minor_arcsec",
                    minor,
                    min_beam,
                    max_beam,
                    unit="arcsec",
                    required=False,
                )
            )

        return checks

    def _create_integrity_checks(self, data: dict[str, Any]) -> list[MetricCheck]:
        """Create data integrity checks (THRESHOLD).

        Parameters
        ----------
        data : Dict[str, Any]
            Input data for data integrity checks.

        Returns
        -------
            None
        """
        checks: list[MetricCheck] = []

        # NaN fraction (lower is better)
        if "nan_fraction" in data:
            nan_frac = data["nan_fraction"]
            checks.append(
                MetricCheck.threshold_lte(
                    "nan_fraction",
                    nan_frac,
                    self.MAX_NAN_FRACTION,
                    message=f"NaN fraction {nan_frac:.2%} (max {self.MAX_NAN_FRACTION:.0%})",
                    required=True,
                )
            )

        return checks


class PhotometryEvaluator(BaseStageEvaluator):
    """Evaluator for the photometry stage (source extraction)."""

    # Photometry quality thresholds
    FLUX_ACCURACY_RANGE: tuple = (0.9, 1.1)
    MAX_POSITION_OFFSET: float = 2.0  # arcsec
    TARGET_POSITION_OFFSET: float = 0.5  # arcsec
    MIN_DETECTION_SNR: float = 5.0
    TARGET_DETECTION_SNR: float = 10.0
    RMS_CONSISTENCY_RANGE: tuple = (0.8, 1.2)
    COMPACTNESS_RANGE: tuple = (0.8, 1.5)

    @property
    def stage(self) -> PipelineStage:
        return PipelineStage.PHOTOMETRY

    def extract_metrics(
        self,
        data: dict[str, Any],
        context: dict[str, Any] | None = None,
    ) -> dict[str, float]:
        """Extract photometry metrics from source catalog.

            Expected data keys:
            - measured_flux_jy: Measured flux of reference source
            - expected_flux_jy: Expected flux from catalog
            - position_offset_arcsec: Positional offset
            - detection_snr: Signal-to-noise of detection
            - local_rms_jy: Local background RMS
            - global_rms_jy: Global image RMS
            - integrated_flux_jy: Integrated source flux
            - peak_flux_jy: Peak pixel flux

        Parameters
        ----------
        data : Dict[str, Any]
            Input data containing source catalog metrics.
        context : Optional[Dict[str, Any]], optional
            Additional context for extraction, by default None

        Returns
        -------
            None
        """
        metrics: dict[str, float] = {}
        ctx = context or {}

        # Flux accuracy
        self._add_flux_accuracy(metrics, data, ctx)

        # Direct metrics
        if "position_offset_arcsec" in data:
            metrics["position_offset_arcsec"] = data["position_offset_arcsec"]
        if "detection_snr" in data:
            metrics["snr_detection"] = data["detection_snr"]

        # Computed ratios
        self._add_rms_consistency(metrics, data)
        self._add_compactness(metrics, data)

        return metrics

    def _add_flux_accuracy(
        self,
        metrics: dict[str, float],
        data: dict[str, Any],
        ctx: dict[str, Any],
    ) -> None:
        """Add flux accuracy metric if data available.

        Parameters
        ----------
        metrics : Dict[str, float]
            Dictionary to update with flux accuracy metric.
        data : Dict[str, Any]
            Input data containing flux information.
        ctx : Dict[str, Any]
            Contextual information for metric calculation.

        Returns
        -------
            None
        """
        measured = data.get("measured_flux_jy")
        expected = data.get("expected_flux_jy", ctx.get("expected_flux_jy"))
        if measured is not None and expected is not None and expected > 0:
            metrics["flux_accuracy"] = measured / expected

    def _add_rms_consistency(
        self,
        metrics: dict[str, float],
        data: dict[str, Any],
    ) -> None:
        """Add local RMS consistency metric if data available.

        Parameters
        ----------
        metrics : Dict[str, float]
            Dictionary to update with RMS consistency metric.
        data : Dict[str, Any]
            Input data containing RMS information.

        Returns
        -------
            None
        """
        local_rms = data.get("local_rms_jy")
        global_rms = data.get("global_rms_jy")
        if local_rms is not None and global_rms is not None and global_rms > 0:
            metrics["local_rms_consistency"] = local_rms / global_rms

    def _add_compactness(
        self,
        metrics: dict[str, float],
        data: dict[str, Any],
    ) -> None:
        """Add compactness metric if data available.

        Parameters
        ----------
        metrics : Dict[str, float]
            Dictionary to update with compactness metric.
        data : Dict[str, Any]
            Input data containing compactness information.

        Returns
        -------
            None
        """
        integrated = data.get("integrated_flux_jy")
        peak = data.get("peak_flux_jy")
        if integrated is not None and peak is not None and peak > 0:
            metrics["compactness"] = integrated / peak

    def evaluate(
        self,
        data: dict[str, Any],
        context: dict[str, Any] | None = None,
    ) -> StageEvaluationResult:
        """Evaluate photometry stage with granular pass/fail checks.

            Uses explicit check types:
            - BOOLEAN: Source detected, catalog created
            - THRESHOLD: Detection SNR, position offset
            - RANGE: Flux accuracy, RMS consistency, compactness

        Parameters
        ----------
        data : Dict[str, Any]
            Photometry output data to evaluate.
        context : Optional[Dict[str, Any]], optional
            Additional context (expected_flux_jy), by default None

        Returns
        -------
            None
        """
        errors: list[str] = []
        warnings: list[str] = []
        ctx = context or {}

        checks: list[MetricCheck] = []

        # Detection Checks (BOOLEAN)
        checks.extend(self._create_detection_checks(data))

        # Flux Quality Checks (RANGE)
        checks.extend(self._create_flux_checks(data, ctx))

        # Position Checks (THRESHOLD)
        checks.extend(self._create_position_checks(data))

        # Source Quality Checks (THRESHOLD and RANGE)
        checks.extend(self._create_source_quality_checks(data))

        # Determine pass/fail
        all_required_passed = all(c.passed for c in checks if c.required)
        warnings.extend(f"{c.name}: {c.message}" for c in checks if not c.required and not c.passed)

        return StageEvaluationResult(
            stage=self.stage,
            passed=all_required_passed and not errors,
            checks=checks,
            errors=errors,
            warnings=warnings,
        )

    def _create_detection_checks(self, data: dict[str, Any]) -> list[MetricCheck]:
        """Create source detection checks (BOOLEAN).

        Parameters
        ----------
        data : Dict[str, Any]
            Input data for source detection checks.

        Returns
        -------
            None
        """
        checks: list[MetricCheck] = []

        # Source detected
        if "source_detected" in data:
            checks.append(
                MetricCheck.boolean(
                    "source_detected",
                    data["source_detected"],
                    "Reference source was detected",
                    required=True,
                )
            )

        # Catalog file exists
        if "catalog_path" in data:
            exists = os.path.exists(data["catalog_path"])
            checks.append(
                MetricCheck.boolean(
                    "catalog_exists", exists, "Source catalog file exists", required=True
                )
            )

        # Sources found count
        if "n_sources" in data:
            has_sources = data["n_sources"] > 0
            checks.append(
                MetricCheck.boolean(
                    "sources_found",
                    has_sources,
                    f"{data['n_sources']} sources extracted",
                    required=True,
                )
            )

        return checks

    def _create_flux_checks(self, data: dict[str, Any], ctx: dict[str, Any]) -> list[MetricCheck]:
        """Create flux accuracy checks (RANGE).

        Parameters
        ----------
        data : Dict[str, Any]
            Input data for flux accuracy checks.
        ctx : Dict[str, Any]
            Contextual information for flux accuracy checks.

        Returns
        -------
            None
        """
        checks: list[MetricCheck] = []

        # Compute flux accuracy
        measured = data.get("measured_flux_jy")
        expected = data.get("expected_flux_jy", ctx.get("expected_flux_jy"))

        if measured is not None and expected is not None and expected > 0:
            accuracy = measured / expected
            min_acc, max_acc = self.FLUX_ACCURACY_RANGE
            checks.append(
                MetricCheck.in_range(
                    "flux_accuracy",
                    accuracy,
                    min_acc,
                    max_acc,
                    message=f"Flux accuracy {accuracy:.3f} (expected {min_acc}-{max_acc})",
                    required=True,
                )
            )

        return checks

    def _create_position_checks(self, data: dict[str, Any]) -> list[MetricCheck]:
        """Create positional accuracy checks (THRESHOLD).

        Parameters
        ----------
        data : Dict[str, Any]
            Input data for positional accuracy checks.

        Returns
        -------
            None
        """
        checks: list[MetricCheck] = []

        # Position offset
        if "position_offset_arcsec" in data:
            offset = data["position_offset_arcsec"]
            checks.append(
                MetricCheck.threshold_lte(
                    "position_offset_arcsec",
                    offset,
                    self.MAX_POSITION_OFFSET,
                    unit="arcsec",
                    required=True,
                )
            )

        return checks

    def _create_source_quality_checks(self, data: dict[str, Any]) -> list[MetricCheck]:
        """Create source quality checks (THRESHOLD and RANGE).

        Parameters
        ----------
        data : Dict[str, Any]
            Input data dictionary.

        """
        checks: list[MetricCheck] = []

        # Detection SNR
        if "detection_snr" in data:
            snr = data["detection_snr"]
            checks.append(
                MetricCheck.threshold_gte(
                    "snr_detection",
                    snr,
                    self.MIN_DETECTION_SNR,
                    message=f"Detection SNR {snr:.1f} (min {self.MIN_DETECTION_SNR})",
                    required=True,
                )
            )

        # Local RMS consistency
        local_rms = data.get("local_rms_jy")
        global_rms = data.get("global_rms_jy")
        if local_rms is not None and global_rms is not None and global_rms > 0:
            consistency = local_rms / global_rms
            min_cons, max_cons = self.RMS_CONSISTENCY_RANGE
            checks.append(
                MetricCheck.in_range(
                    "local_rms_consistency",
                    consistency,
                    min_cons,
                    max_cons,
                    message=f"RMS consistency {consistency:.2f} (expected {min_cons}-{max_cons})",
                    required=False,
                )
            )

        # Compactness
        integrated = data.get("integrated_flux_jy")
        peak = data.get("peak_flux_jy")
        if integrated is not None and peak is not None and peak > 0:
            compactness = integrated / peak
            min_comp, max_comp = self.COMPACTNESS_RANGE
            checks.append(
                MetricCheck.in_range(
                    "compactness",
                    compactness,
                    min_comp,
                    max_comp,
                    message=f"Compactness {compactness:.2f} (expected {min_comp}-{max_comp})",
                    required=False,
                )
            )

        return checks


class MosaicEvaluator(BaseStageEvaluator):
    """Evaluator for the mosaic stage (image combination)."""

    # Mosaic quality thresholds
    MAX_EFFECTIVE_NOISE_JY: float = 0.001
    TARGET_EFFECTIVE_NOISE_JY: float = 0.0005
    MIN_NOISE_IMPROVEMENT: float = 1.2
    TARGET_NOISE_IMPROVEMENT: float = 1.41  # sqrt(2) for 2 images
    MIN_WEIGHT_UNIFORMITY: float = 0.8
    TARGET_WEIGHT_UNIFORMITY: float = 0.95
    MIN_SEAM_SCORE: float = 0.9
    MIN_COVERAGE_FRACTION: float = 0.9

    @property
    def stage(self) -> PipelineStage:
        return PipelineStage.MOSAIC

    def extract_metrics(
        self,
        data: dict[str, Any],
        context: dict[str, Any] | None = None,
    ) -> dict[str, float]:
        """Extract mosaic metrics from combined image.

            Expected data keys:
            - effective_rms_jy: RMS in combined image
            - single_image_rms_jy: RMS of input images
            - weight_uniformity: Weight map uniformity (0-1)
            - seam_score: Freedom from seam artifacts (0-1)
            - coverage_fraction: Area coverage fraction

        Parameters
        ----------
        data : Dict[str, Any]
            Input data dictionary.
        context : Optional[Dict[str, Any]], optional
            Additional context, by default None

        """
        metrics: dict[str, float] = {}

        # Effective noise
        if "effective_rms_jy" in data:
            metrics["effective_noise_jy"] = data["effective_rms_jy"]

        # Noise improvement factor
        effective = data.get("effective_rms_jy")
        single = data.get("single_image_rms_jy")
        if effective is not None and single is not None and effective > 0:
            metrics["noise_improvement"] = single / effective

        # Weight uniformity
        if "weight_uniformity" in data:
            metrics["weight_uniformity"] = data["weight_uniformity"]

        # Seam artifact score
        if "seam_score" in data:
            metrics["seam_artifact_score"] = data["seam_score"]

        # Coverage fraction
        if "coverage_fraction" in data:
            metrics["coverage_fraction"] = data["coverage_fraction"]

        return metrics

    def evaluate(
        self,
        data: dict[str, Any],
        context: dict[str, Any] | None = None,
    ) -> StageEvaluationResult:
        """Evaluate mosaic stage with granular pass/fail checks.

            Uses explicit check types:
            - BOOLEAN: Mosaic file exists, weight map exists
            - THRESHOLD: Effective noise, noise improvement, weight uniformity,
            seam score, coverage fraction
            - COUNT: Input images combined

        Parameters
        ----------
        data : Dict[str, Any]
            Mosaic output data to evaluate.
        context : Optional[Dict[str, Any]], optional
            Additional context for evaluation, by default None

        """
        errors: list[str] = []
        warnings: list[str] = []

        checks: list[MetricCheck] = []

        # File Existence Checks (BOOLEAN)
        checks.extend(self._create_file_checks(data))

        # Noise Quality Checks (THRESHOLD)
        checks.extend(self._create_noise_checks(data))

        # Weight and Coverage Checks (THRESHOLD)
        checks.extend(self._create_weight_coverage_checks(data))

        # Artifact Checks (THRESHOLD)
        checks.extend(self._create_artifact_checks(data))

        # Determine pass/fail
        all_required_passed = all(c.passed for c in checks if c.required)
        warnings.extend(f"{c.name}: {c.message}" for c in checks if not c.required and not c.passed)

        return StageEvaluationResult(
            stage=self.stage,
            passed=all_required_passed and not errors,
            checks=checks,
            errors=errors,
            warnings=warnings,
        )

    def _create_file_checks(self, data: dict[str, Any]) -> list[MetricCheck]:
        """Create mosaic file existence checks (BOOLEAN).

        Parameters
        ----------
        data : Dict[str, Any]
            Input data dictionary.

        """
        checks: list[MetricCheck] = []

        # Mosaic file exists
        if "mosaic_path" in data:
            exists = os.path.exists(data["mosaic_path"])
            checks.append(
                MetricCheck.boolean(
                    "mosaic_file_exists", exists, "Mosaic FITS file exists", required=True
                )
            )

        # Weight map exists (optional)
        if "weight_map_path" in data:
            exists = os.path.exists(data["weight_map_path"])
            checks.append(
                MetricCheck.boolean(
                    "weight_map_exists", exists, "Weight map file exists", required=False
                )
            )

        # Input images combined (COUNT - need at least 2 for a mosaic)
        if "n_images_combined" in data:
            n = data["n_images_combined"]
            min_images = 2
            checks.append(
                MetricCheck.count(
                    "images_combined",
                    n,
                    min_images,
                    f"{n} images combined (min {min_images})",
                    required=True,
                )
            )

        return checks

    def _create_noise_checks(self, data: dict[str, Any]) -> list[MetricCheck]:
        """Create noise quality checks (THRESHOLD).

        Parameters
        ----------
        data : Dict[str, Any]
            Input data dictionary.

        """
        checks: list[MetricCheck] = []

        # Effective noise (lower is better)
        if "effective_rms_jy" in data:
            noise = data["effective_rms_jy"]
            checks.append(
                MetricCheck.threshold_lte(
                    "effective_noise_jy",
                    noise,
                    self.MAX_EFFECTIVE_NOISE_JY,
                    unit="Jy",
                    required=True,
                )
            )

        # Noise improvement (higher is better)
        effective = data.get("effective_rms_jy")
        single = data.get("single_image_rms_jy")
        if effective is not None and single is not None and effective > 0:
            improvement = single / effective
            checks.append(
                MetricCheck.threshold_gte(
                    "noise_improvement",
                    improvement,
                    self.MIN_NOISE_IMPROVEMENT,
                    message=f"Noise improvement {improvement:.2f}x (min {self.MIN_NOISE_IMPROVEMENT}x)",
                    required=True,
                )
            )

        return checks

    def _create_weight_coverage_checks(self, data: dict[str, Any]) -> list[MetricCheck]:
        """Create weight and coverage checks (THRESHOLD).

        Parameters
        ----------
        data : Dict[str, Any]
            Input data dictionary.

        """
        checks: list[MetricCheck] = []

        # Weight uniformity
        if "weight_uniformity" in data:
            uniformity = data["weight_uniformity"]
            checks.append(
                MetricCheck.threshold_gte(
                    "weight_uniformity",
                    uniformity,
                    self.MIN_WEIGHT_UNIFORMITY,
                    message=f"Weight uniformity {uniformity:.2f} (min {self.MIN_WEIGHT_UNIFORMITY})",
                    required=False,
                )
            )

        # Coverage fraction
        if "coverage_fraction" in data:
            coverage = data["coverage_fraction"]
            checks.append(
                MetricCheck.threshold_gte(
                    "coverage_fraction",
                    coverage,
                    self.MIN_COVERAGE_FRACTION,
                    message=f"Coverage {coverage:.1%} (min {self.MIN_COVERAGE_FRACTION:.0%})",
                    required=True,
                )
            )

        return checks

    def _create_artifact_checks(self, data: dict[str, Any]) -> list[MetricCheck]:
        """Create artifact quality checks (THRESHOLD).

        Parameters
        ----------
        data : Dict[str, Any]
            Input data dictionary.

        """
        checks: list[MetricCheck] = []

        # Seam artifact score (higher is better = fewer artifacts)
        if "seam_score" in data:
            score = data["seam_score"]
            checks.append(
                MetricCheck.threshold_gte(
                    "seam_artifact_score",
                    score,
                    self.MIN_SEAM_SCORE,
                    message=f"Seam score {score:.2f} (min {self.MIN_SEAM_SCORE})",
                    required=False,
                )
            )

        return checks


# =============================================================================
# Evaluator Registry
# =============================================================================


STAGE_EVALUATORS: dict[PipelineStage, type] = {
    PipelineStage.INGEST: IngestEvaluator,
    PipelineStage.CONVERT: ConversionEvaluator,
    PipelineStage.CALIBRATE: CalibrationEvaluator,
    PipelineStage.IMAGE: ImagingEvaluator,
    PipelineStage.PHOTOMETRY: PhotometryEvaluator,
    PipelineStage.MOSAIC: MosaicEvaluator,
}


def get_evaluator(
    stage: PipelineStage,
    registry: StageRegistry | None = None,
) -> BaseStageEvaluator:
    """Get an evaluator instance for a specific stage.

    Parameters
    ----------
    stage : PipelineStage
        Pipeline stage to evaluate.
    registry : Optional[StageRegistry]
        Optional stage registry for custom thresholds. (Default value = None)

    """
    evaluator_class = STAGE_EVALUATORS.get(stage)
    if evaluator_class is None:
        raise ValueError(f"No evaluator registered for stage: {stage.value}")
    return evaluator_class(registry=registry)


def evaluate_pipeline_run(
    stage_data: dict[PipelineStage, dict[str, Any]],
    context: dict[str, Any] | None = None,
    registry: StageRegistry | None = None,
) -> dict[PipelineStage, StageEvaluationResult]:
    """Evaluate a complete pipeline run across all stages.

    Parameters
    ----------
    stage_data : Dict[PipelineStage, Dict[str, Any]]
        Dictionary mapping stages to their output data.
    context : Optional[Dict[str, Any]]
        Additional context for evaluation. (Default value = None)
    registry : Optional[StageRegistry]
        Stage registry for threshold lookup. (Default value = None)

    """
    results: dict[PipelineStage, StageEvaluationResult] = {}

    for stage, data in stage_data.items():
        evaluator = get_evaluator(stage, registry)
        results[stage] = evaluator.evaluate(data, context)

    return results


def compute_overall_score(
    results: dict[PipelineStage, StageEvaluationResult],
    registry: StageRegistry | None = None,
) -> float:
    """Compute fraction of stages that passed (deprecated - use count_passed_stages).

    Parameters
    ----------
    results : Dict[PipelineStage, StageEvaluationResult]
        Stage evaluation results.
    registry : Optional[StageRegistry]
        Unused, kept for backward compatibility. (Default value = None)

    """
    if not results:
        return 0.0
    passed = sum(1 for r in results.values() if r.passed)
    return passed / len(results)


def count_passed_stages(
    results: dict[PipelineStage, StageEvaluationResult],
) -> tuple[int, int, list[str]]:
    """Count passed and failed stages.

    Parameters
    ----------
    results : Dict[PipelineStage, StageEvaluationResult]
        Stage evaluation results.

    """
    passed = 0
    failed = 0
    failed_names: list[str] = []

    for stage, result in results.items():
        if result.passed:
            passed += 1
        else:
            failed += 1
            failed_names.append(stage.value)

    return passed, failed, failed_names


# Default quality grade thresholds (deprecated - kept for backward compatibility)
_DEFAULT_QUALITY_GRADES = {
    "excellent": 0.95,
    "good": 0.85,
    "acceptable": 0.70,
    "poor": 0.50,
    "failed": 0.0,
}


def get_quality_grade(
    score: float,
    thresholds: dict[str, float] | None = None,
) -> str:
    """Get quality grade string for a score (deprecated).

        This function is kept for backward compatibility but should not
        be used in new code. Use pass/fail checks instead.

    Parameters
    ----------
    score : float
        Composite score (0.0 to 1.0).
    thresholds : Optional[Dict[str, float]]
        Optional threshold overrides. (Default value = None)

    """
    grades = thresholds or _DEFAULT_QUALITY_GRADES

    if score >= grades.get("excellent", 0.95):
        return "excellent"
    if score >= grades.get("good", 0.85):
        return "good"
    if score >= grades.get("acceptable", 0.70):
        return "acceptable"
    if score >= grades.get("poor", 0.50):
        return "poor"
    return "failed"
