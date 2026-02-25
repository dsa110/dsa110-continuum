"""
Pipeline stage taxonomy for DSA-110 evaluation framework.

Defines the 6 pipeline stages with their:
- Expected inputs and outputs
- Quality metrics and thresholds
- Validation requirements
- Database table associations

This module provides the structural foundation for stage-specific evaluators.
"""

from __future__ import annotations

import fnmatch
import logging
import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import yaml

from .config_loader import load_thresholds_config
from dsa110_contimg.common.utils import get_env_path

logger = logging.getLogger(__name__)


# =============================================================================
# Constants
# =============================================================================

# Default config paths
_CONFIG_ROOT = get_env_path("CONTIMG_BASE_DIR", default="/data/dsa110-contimg") / "config"
DEFAULT_THRESHOLDS_PATH = _CONFIG_ROOT / "evaluation_thresholds.yaml"
DEFAULT_REFERENCE_PATH = _CONFIG_ROOT / "reference_datasets.yaml"

# DSA-110 specific
NUM_SUBBANDS = 16
NUM_ANTENNAS = 63
SUBBAND_CLUSTER_TOLERANCE_S = 120.0


# =============================================================================
# Stage Enumeration
# =============================================================================


class PipelineStage(Enum):
    """Pipeline processing stages in execution order.

    Each stage transforms data and produces specific outputs that
    become inputs for subsequent stages.

    """

    INGEST = "ingest"
    CONVERT = "convert"
    CALIBRATE = "calibrate"
    IMAGE = "image"
    MOSAIC = "mosaic"
    PHOTOMETRY = "photometry"  # Runs on mosaic output

    @classmethod
    def ordered(cls) -> list[PipelineStage]:
        """ """
        return [
            cls.INGEST,
            cls.CONVERT,
            cls.CALIBRATE,
            cls.IMAGE,
            cls.MOSAIC,
            cls.PHOTOMETRY,
        ]

    @classmethod
    def core_stages(cls) -> list[PipelineStage]:
        """ """
        return [
            cls.INGEST,
            cls.CONVERT,
            cls.CALIBRATE,
            cls.IMAGE,
            cls.MOSAIC,
        ]

    def next_stage(self) -> PipelineStage | None:
        """ """
        ordered = self.ordered()
        try:
            idx = ordered.index(self)
            if idx < len(ordered) - 1:
                return ordered[idx + 1]
        except ValueError:
            pass
        return None

    def previous_stage(self) -> PipelineStage | None:
        """ """
        ordered = self.ordered()
        try:
            idx = ordered.index(self)
            if idx > 0:
                return ordered[idx - 1]
        except ValueError:
            pass
        return None


# =============================================================================
# Data Product Definitions
# =============================================================================


@dataclass
class DataProduct:
    """Specification for a data product produced or consumed by a stage."""

    name: str
    file_patterns: list[str] = field(default_factory=list)
    db_table: str | None = None
    required: bool = True
    description: str = ""

    def matches_path(self, path: Path) -> bool:
        """Check if a path matches any of the file patterns.

        Parameters
        ----------
        path : Path
            Path to check against file patterns

        Returns
        -------
            bool
            True if path matches any pattern, False otherwise
        """
        path_str = str(path)
        return any(fnmatch.fnmatch(path_str, pattern) for pattern in self.file_patterns)


@dataclass
class MetricSpec:
    """Specification for a quality metric."""

    name: str
    min_value: float | None = None
    max_value: float | None = None
    target_value: float | None = None
    weight: float = 1.0
    unit: str = ""
    description: str = ""

    def evaluate(self, value: float) -> float:
        """Evaluate a value against this metric spec.

        Parameters
        ----------
        value : float
            Value to evaluate

        Returns
        -------
            float
            Evaluation score or metric result
        """
        if value is None:
            return 0.0

        # Check hard bounds and score appropriately
        if self.min_value is not None and value < self.min_value:
            return self._score_below_min(value)

        if self.max_value is not None and value > self.max_value:
            return self._score_above_max(value)

        # Within bounds - score based on target if defined
        if self.target_value is not None:
            return self._score_versus_target(value)

        # No target, but within bounds = good
        return 1.0

    def _score_below_min(self, value: float) -> float:
        """Score for value below minimum threshold.

        Parameters
        ----------
        value : float
            Value to score

        Returns
        -------
            float
            Score for value below minimum threshold
        """
        if self.min_value is None or self.min_value == 0:
            return 0.0
        ratio = value / self.min_value
        return max(0.0, ratio * 0.5)

    def _score_above_max(self, value: float) -> float:
        """Score for value above maximum threshold.

        Parameters
        ----------
        value : float
            Value to score

        Returns
        -------
            float
            Score for value above maximum threshold
        """
        if self.max_value is None or self.max_value == 0:
            return 0.0
        ratio = self.max_value / value
        return max(0.0, ratio * 0.5)

    def _score_versus_target(self, value: float) -> float:
        """Score based on distance from target value.

        Parameters
        ----------
        value : float
            Value to score

        Returns
        -------
            float
            Score based on distance from target
        """
        if self.target_value is None:
            return 1.0
        if self.target_value == 0:
            # Target is 0, closer to 0 is better
            if self.max_value and self.max_value > 0:
                return 1.0 - (value / self.max_value) * 0.5
            return 1.0 if value == 0 else 0.5
        deviation = abs(value - self.target_value) / abs(self.target_value)
        return max(0.5, 1.0 - deviation * 0.5)


# =============================================================================
# Stage Specifications
# =============================================================================


@dataclass
class StageSpec:
    """Complete specification for a pipeline stage.

    Defines what the stage consumes, produces, and how to evaluate it.

    """

    stage: PipelineStage
    description: str

    # Data flow
    inputs: list[DataProduct] = field(default_factory=list)
    outputs: list[DataProduct] = field(default_factory=list)

    # Quality metrics
    metrics: list[MetricSpec] = field(default_factory=list)

    # Database associations
    db_tables: list[str] = field(default_factory=list)

    # State machine states
    entry_state: str = ""
    exit_state: str = ""

    def get_metric(self, name: str) -> MetricSpec | None:
        """Get a metric spec by name.

        Parameters
        ----------
        name : str
            Name of the metric

        Returns
        -------
            MetricSpec
            Metric specification object
        """
        for metric in self.metrics:
            if metric.name == name:
                return metric
        return None

    def required_outputs(self) -> list[DataProduct]:
        """Get required output products."""
        return [p for p in self.outputs if p.required]


# =============================================================================
# Stage Definitions
# =============================================================================


def _define_ingest_stage() -> StageSpec:
    """Define the ingest stage specification."""
    return StageSpec(
        stage=PipelineStage.INGEST,
        description="UVH5 subband file discovery and registration",
        inputs=[
            DataProduct(
                name="uvh5_files",
                file_patterns=["*_sb*.hdf5", "*_sb*.uvh5"],
                description="Raw UVH5 subband files from correlator",
            ),
        ],
        outputs=[
            DataProduct(
                name="subband_group",
                db_table="processing_queue",
                description="Registered subband group ready for conversion",
            ),
            DataProduct(
                name="file_records",
                db_table="hdf5_files",
                description="Individual file metadata records",
            ),
        ],
        metrics=[
            MetricSpec(
                name="subband_completeness",
                min_value=1.0,
                target_value=1.0,
                weight=1.0,
                description="Fraction of expected subbands present",
            ),
            MetricSpec(
                name="timestamp_clustering",
                max_value=120.0,
                target_value=60.0,
                weight=0.5,
                unit="seconds",
                description="Max time spread within subband group",
            ),
            MetricSpec(
                name="file_size_consistency",
                min_value=0.9,
                target_value=1.0,
                weight=0.3,
                description="Fraction of files with expected size",
            ),
        ],
        db_tables=["processing_queue", "hdf5_files", "subband_files"],
        entry_state="pending",
        exit_state="converting",
    )


def _define_convert_stage() -> StageSpec:
    """Define the convert stage specification."""
    return StageSpec(
        stage=PipelineStage.CONVERT,
        description="UVH5 to Measurement Set conversion",
        inputs=[
            DataProduct(
                name="subband_group",
                db_table="processing_queue",
                description="Registered subband group",
            ),
        ],
        outputs=[
            DataProduct(
                name="measurement_set",
                file_patterns=["*.ms", "*.ms/"],
                db_table="ms_index",
                description="CASA Measurement Set",
            ),
        ],
        metrics=[
            MetricSpec(
                name="ms_structure_valid",
                min_value=1.0,
                target_value=1.0,
                weight=1.0,
                description="MS passes structural validation",
            ),
            MetricSpec(
                name="antenna_completeness",
                min_value=0.95,
                target_value=1.0,
                weight=0.8,
                description="Fraction of expected antennas present",
            ),
            MetricSpec(
                name="spw_completeness",
                min_value=1.0,
                target_value=1.0,
                weight=0.9,
                description="Fraction of expected spectral windows",
            ),
            MetricSpec(
                name="time_range_valid",
                min_value=1.0,
                target_value=1.0,
                weight=0.7,
                description="Time range matches expected observation",
            ),
            MetricSpec(
                name="uv_coverage_score",
                min_value=0.7,
                target_value=0.9,
                weight=0.5,
                description="UV plane coverage quality",
            ),
        ],
        db_tables=["ms_index"],
        entry_state="converting",
        exit_state="converted",
    )


def _define_calibrate_stage() -> StageSpec:
    """Define the calibrate stage specification."""
    return StageSpec(
        stage=PipelineStage.CALIBRATE,
        description="Bandpass, delay, and gain calibration",
        inputs=[
            DataProduct(
                name="measurement_set",
                file_patterns=["*.ms", "*.ms/"],
                db_table="ms_index",
                description="Uncalibrated Measurement Set",
            ),
        ],
        outputs=[
            DataProduct(
                name="calibration_tables",
                file_patterns=["*.bcal", "*.gcal", "*.kcal"],
                db_table="calibration_tables",
                description="Calibration solution tables",
            ),
            DataProduct(
                name="calibrated_ms",
                file_patterns=["*.ms", "*.ms/"],
                db_table="ms_index",
                description="MS with CORRECTED_DATA column",
            ),
            DataProduct(
                name="calibration_qa",
                db_table="calibration_qa",
                description="Calibration quality metrics",
            ),
        ],
        metrics=[
            # Delay metrics
            MetricSpec(
                name="delay_snr_median",
                min_value=10.0,
                target_value=50.0,
                weight=0.8,
                description="Median SNR of delay solutions",
            ),
            MetricSpec(
                name="delay_max_ns",
                max_value=100.0,
                target_value=50.0,
                weight=0.6,
                unit="ns",
                description="Maximum delay value",
            ),
            # Bandpass metrics
            MetricSpec(
                name="bp_snr_median",
                min_value=20.0,
                target_value=100.0,
                weight=1.0,
                description="Median SNR of bandpass solutions",
            ),
            MetricSpec(
                name="bp_flagged_fraction",
                max_value=0.3,
                target_value=0.1,
                weight=0.9,
                description="Fraction of bandpass solutions flagged",
            ),
            MetricSpec(
                name="bp_phase_scatter_deg",
                max_value=30.0,
                target_value=10.0,
                weight=0.7,
                unit="degrees",
                description="Phase scatter in bandpass solutions",
            ),
            # Gain metrics
            MetricSpec(
                name="gain_snr_median",
                min_value=10.0,
                target_value=30.0,
                weight=0.9,
                description="Median SNR of gain solutions",
            ),
            MetricSpec(
                name="gain_phase_scatter_deg",
                max_value=20.0,
                target_value=5.0,
                weight=0.6,
                unit="degrees",
                description="Phase scatter in gain solutions",
            ),
            # Overall metrics
            MetricSpec(
                name="overall_flag_fraction",
                max_value=0.5,
                target_value=0.2,
                weight=0.8,
                description="Total fraction of data flagged",
            ),
            MetricSpec(
                name="flux_scale_accuracy",
                min_value=0.9,
                max_value=1.1,
                target_value=1.0,
                weight=1.0,
                description="Flux scale factor (1.0 = perfect)",
            ),
        ],
        db_tables=[
            "calibration_tables",
            "calibration_applied",
            "calibration_qa",
            "calibration_metrics",
        ],
        entry_state="converted",
        exit_state="applying_cal",
    )


def _define_image_stage() -> StageSpec:
    """Define the image stage specification."""
    return StageSpec(
        stage=PipelineStage.IMAGE,
        description="Visibility gridding and FFT to produce FITS images",
        inputs=[
            DataProduct(
                name="calibrated_ms",
                file_patterns=["*.ms", "*.ms/"],
                db_table="ms_index",
                description="Calibrated Measurement Set",
            ),
        ],
        outputs=[
            DataProduct(
                name="fits_image",
                file_patterns=["*.fits", "*.fits.gz"],
                db_table="images",
                description="FITS image (dirty, restored, residual)",
            ),
            DataProduct(
                name="psf_image",
                file_patterns=["*-psf.fits"],
                db_table="images",
                required=False,
                description="Point spread function image",
            ),
        ],
        metrics=[
            MetricSpec(
                name="rms_noise_jy",
                max_value=0.002,
                target_value=0.001,
                weight=1.0,
                unit="Jy/beam",
                description="RMS noise in image",
            ),
            MetricSpec(
                name="dynamic_range",
                min_value=100.0,
                target_value=1000.0,
                weight=0.9,
                description="Peak / RMS ratio",
            ),
            MetricSpec(
                name="peak_flux_jy",
                min_value=0.001,
                weight=0.7,
                unit="Jy/beam",
                description="Peak flux in image",
            ),
            MetricSpec(
                name="beam_major_arcsec",
                min_value=1.0,
                max_value=60.0,
                target_value=15.0,
                weight=0.5,
                unit="arcsec",
                description="Synthesized beam major axis",
            ),
            MetricSpec(
                name="beam_minor_arcsec",
                min_value=1.0,
                max_value=60.0,
                target_value=15.0,
                weight=0.5,
                unit="arcsec",
                description="Synthesized beam minor axis",
            ),
            MetricSpec(
                name="nan_fraction",
                max_value=0.01,
                target_value=0.0,
                weight=0.8,
                description="Fraction of NaN pixels",
            ),
        ],
        db_tables=["images", "image_metadata"],
        entry_state="applying_cal",
        exit_state="imaging",
    )


def _define_photometry_stage() -> StageSpec:
    """Define the photometry stage specification."""
    return StageSpec(
        stage=PipelineStage.PHOTOMETRY,
        description="Source extraction and flux measurement",
        inputs=[
            DataProduct(
                name="fits_image",
                file_patterns=["*.fits", "*.fits.gz"],
                db_table="images",
                description="FITS image for source extraction",
            ),
        ],
        outputs=[
            DataProduct(
                name="source_catalog",
                file_patterns=["*.csv", "*.fits"],
                db_table="photometry",
                description="Extracted source catalog",
            ),
            DataProduct(
                name="source_metrics",
                db_table="source_metrics",
                description="Per-source quality metrics",
            ),
        ],
        metrics=[
            MetricSpec(
                name="flux_accuracy",
                min_value=0.9,
                max_value=1.1,
                target_value=1.0,
                weight=1.0,
                description="Measured / expected flux ratio",
            ),
            MetricSpec(
                name="position_offset_arcsec",
                max_value=2.0,
                target_value=0.5,
                weight=0.9,
                unit="arcsec",
                description="Positional offset from reference",
            ),
            MetricSpec(
                name="snr_detection",
                min_value=5.0,
                target_value=10.0,
                weight=0.8,
                description="Signal-to-noise ratio of detection",
            ),
            MetricSpec(
                name="local_rms_consistency",
                min_value=0.8,
                max_value=1.2,
                target_value=1.0,
                weight=0.6,
                description="Local RMS / global RMS ratio",
            ),
            MetricSpec(
                name="compactness",
                min_value=0.8,
                max_value=1.5,
                target_value=1.0,
                weight=0.5,
                description="Integrated / peak flux ratio",
            ),
        ],
        db_tables=["photometry", "source_metrics", "catalog_crossmatch"],
        entry_state="imaging",
        exit_state="done",
    )


def _define_mosaic_stage() -> StageSpec:
    """Define the mosaic stage specification."""
    return StageSpec(
        stage=PipelineStage.MOSAIC,
        description="Multi-image combination",
        inputs=[
            DataProduct(
                name="fits_images",
                file_patterns=["*.fits", "*.fits.gz"],
                db_table="images",
                description="Multiple FITS images to combine",
            ),
        ],
        outputs=[
            DataProduct(
                name="mosaic_image",
                file_patterns=["*_mosaic.fits"],
                db_table="mosaics",
                description="Combined mosaic image",
            ),
            DataProduct(
                name="weight_map",
                file_patterns=["*_weights.fits"],
                required=False,
                description="Mosaic weight map",
            ),
        ],
        metrics=[
            MetricSpec(
                name="effective_noise_jy",
                max_value=0.001,
                target_value=0.0005,
                weight=1.0,
                unit="Jy",
                description="Effective RMS noise in combined image",
            ),
            MetricSpec(
                name="noise_improvement",
                min_value=1.2,
                target_value=1.41,
                weight=0.8,
                description="Noise improvement factor vs single image",
            ),
            MetricSpec(
                name="weight_uniformity",
                min_value=0.8,
                target_value=0.95,
                weight=0.6,
                description="Uniformity of weight map",
            ),
            MetricSpec(
                name="seam_artifact_score",
                min_value=0.9,
                target_value=1.0,
                weight=0.7,
                description="Freedom from seam artifacts",
            ),
            MetricSpec(
                name="coverage_fraction",
                min_value=0.9,
                target_value=1.0,
                weight=0.5,
                description="Fraction of expected area covered",
            ),
        ],
        db_tables=["mosaics"],
        entry_state="done",
        exit_state="done",
    )


# =============================================================================
# Stage Registry
# =============================================================================


class StageRegistry:
    """Registry of all pipeline stage specifications.

    Provides access to stage definitions and threshold configuration.

    """

    def __init__(
        self,
        thresholds_path: Path | None = None,
        reference_path: Path | None = None,
    ):
        """Initialize the stage registry.

        Parameters
        ----------
        thresholds_path : str, optional
            Path to evaluation_thresholds.yaml
        reference_path : str, optional
            Path to reference_datasets.yaml
        """
        self._stages: dict[PipelineStage, StageSpec] = {}
        self._thresholds: dict[str, Any] = {}
        self._references: dict[str, Any] = {}

        # Load stage definitions
        self._stages[PipelineStage.INGEST] = _define_ingest_stage()
        self._stages[PipelineStage.CONVERT] = _define_convert_stage()
        self._stages[PipelineStage.CALIBRATE] = _define_calibrate_stage()
        self._stages[PipelineStage.IMAGE] = _define_image_stage()
        self._stages[PipelineStage.PHOTOMETRY] = _define_photometry_stage()
        self._stages[PipelineStage.MOSAIC] = _define_mosaic_stage()

        # Load YAML configurations
        thresholds_path = thresholds_path or DEFAULT_THRESHOLDS_PATH
        reference_path = reference_path or DEFAULT_REFERENCE_PATH

        self._load_thresholds(thresholds_path)

        if reference_path.exists():
            self._load_references(reference_path)
        else:
            logger.warning("Reference config not found: %s", reference_path)

    def _load_thresholds(self, path: Path) -> None:
        """Load and apply threshold overrides from YAML.

        Parameters
        ----------
        path : Path
            Path to thresholds YAML file

        Returns
        -------
            dict
            Loaded threshold overrides
        """
        config = load_thresholds_config(path)
        self._thresholds = config

        # Apply threshold overrides to stage specs
        stages_config = config.get("stages", {})
        for stage_name, stage_thresholds in stages_config.items():
            try:
                stage = PipelineStage(stage_name)
                spec = self._stages.get(stage)
                if spec:
                    self._apply_thresholds(spec, stage_thresholds)
            except ValueError:
                logger.warning("Unknown stage in config: %s", stage_name)

    def _apply_thresholds(self, spec: StageSpec, thresholds: dict[str, Any]) -> None:
        """Apply threshold values to a stage spec.

        Parameters
        ----------
        spec : StageSpec
            Stage specification to update
        thresholds : dict
            Threshold values to apply

        Returns
        -------
            StageSpec
            Updated stage specification
        """
        for metric_name, threshold_values in thresholds.items():
            if metric_name in ("description",):
                continue

            if not isinstance(threshold_values, dict):
                continue

            metric = spec.get_metric(metric_name)
            if metric:
                if "min" in threshold_values:
                    metric.min_value = threshold_values["min"]
                if "max" in threshold_values:
                    metric.max_value = threshold_values["max"]
                if "target" in threshold_values:
                    metric.target_value = threshold_values["target"]
                if "weight" in threshold_values:
                    metric.weight = threshold_values["weight"]

    def _load_references(self, path: Path) -> None:
        """Load reference dataset specifications from YAML.

        Parameters
        ----------
        path : Path
            Path to reference datasets YAML file

        Returns
        -------
            dict
            Loaded reference dataset specifications
        """
        from dsa110_contimg.common.utils.yaml_loader import load_yaml_with_env

        self._references = load_yaml_with_env(path, expand_vars=True)

    def get_stage(self, stage: PipelineStage) -> StageSpec:
        """Get the specification for a stage.

        Parameters
        ----------
        stage : PipelineStage
            Stage to retrieve specification for

        Returns
        -------
            StageSpec
            Specification of the stage
        """
        return self._stages[stage]

    def get_all_stages(self) -> list[StageSpec]:
        """Get all stage specifications in order."""
        return [self._stages[s] for s in PipelineStage.ordered()]

    def get_reference_preset(self, preset_path: str) -> dict[str, Any] | None:
        """Get a reference dataset preset by path.

        Parameters
        ----------
        preset_path : str
            Dot-separated path like "calibrators.0834+555"

        Returns
        -------
            ReferencePreset
            Reference dataset preset object
        """
        parts = preset_path.split(".")
        current = self._references

        for part in parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return None

        return current if isinstance(current, dict) else None

    def get_quality_grade(self, score: float) -> str:
        """Map a numeric score to a quality grade.

        Parameters
        ----------
        score : float
            Numeric score to map.

        """
        thresholds = self._thresholds.get("quality_grades", {}).get("thresholds", {})

        if score >= thresholds.get("excellent", 0.95):
            return "excellent"
        if score >= thresholds.get("good", 0.85):
            return "good"
        if score >= thresholds.get("acceptable", 0.70):
            return "acceptable"
        if score >= thresholds.get("poor", 0.50):
            return "poor"
        return "failed"


# =============================================================================
# Module-Level Instance
# =============================================================================


# Lazy-loaded global registry
_registry: StageRegistry | None = None


def get_registry() -> StageRegistry:
    """Get the global stage registry (lazy-loaded)."""
    global _registry
    if _registry is None:
        _registry = StageRegistry()
    return _registry


def get_stage_spec(stage: PipelineStage) -> StageSpec:
    """Convenience function to get a stage specification.

    Parameters
    ----------
    stage : PipelineStage
        Stage to get specification for

    Returns
    -------
        StageSpec
        Specification of the given stage
    """
    return get_registry().get_stage(stage)
