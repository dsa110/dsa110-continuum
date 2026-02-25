"""
Custom code-based evaluators for DSA-110 Pipeline.

These evaluators implement objective, measurable metrics for:
1. Pipeline stage completion accuracy
2. Failure detection and classification rate
3. Regression coverage against golden datasets

All evaluators follow the Azure AI Evaluation SDK pattern:
- Callable with (query, response, ground_truth?) signature
- Return dict with metric name and score (0.0-1.0)
- Optionally include reasoning/details

Reference: Azure AI Evaluation SDK code-based evaluator pattern
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any
import statistics

logger = logging.getLogger(__name__)


# =============================================================================
# Constants
# =============================================================================


class PipelineStage(Enum):
    """Pipeline processing stages for completion tracking."""

    INGEST = "ingest"
    CONVERT = "convert"
    FLAG_RFI = "flag_rfi"
    SOLVE_CAL = "solve_cal"
    APPLY_CAL = "apply_cal"
    IMAGING = "imaging"

    @classmethod
    def all_stages(cls) -> list[PipelineStage]:
        """Return all stages in processing order."""
        return [
            cls.INGEST,
            cls.CONVERT,
            cls.FLAG_RFI,
            cls.SOLVE_CAL,
            cls.APPLY_CAL,
            cls.IMAGING,
        ]


# Expected stage count for full completion
TOTAL_STAGES = len(PipelineStage.all_stages())

# DSA-110 specific constants
NUM_SUBBANDS = 16
SUBBAND_CLUSTER_TOLERANCE_S = 120.0


# =============================================================================
# Pipeline Completion Evaluator
# =============================================================================


@dataclass
class PipelineCompletionEvaluator:
    """
    Evaluates pipeline stage completion accuracy.

    Metrics:
        - Stage completion rate (0-1): fraction of expected stages completed
        - Subband completeness: all 16 subbands processed
        - State machine validity: no invalid transitions detected

    Usage:
        evaluator = PipelineCompletionEvaluator()
        result = evaluator(
            query="Process transit obs_id=12345",
            response={
                "status": "completed",
                "stages_completed": ["ingest", "convert", "flag_rfi", "solve_cal", "apply_cal", "imaging"],
                "subbands_processed": 16,
                "state_transitions": [...]
            }
        )
        # result = {"stage_completion_rate": 1.0, "subband_completeness": 1.0, ...}
    """

    # Valid state transitions (from -> list of valid to states)
    valid_transitions: dict[str, list[str]] = field(
        default_factory=lambda: {
            "pending": ["converting", "failed", "error"],
            "converting": ["converted", "failed", "error"],
            "converted": ["flagging_rfi", "failed", "error"],
            "flagging_rfi": ["solving_cal", "failed", "error"],
            "solving_cal": ["applying_cal", "failed", "error"],
            "applying_cal": ["imaging", "failed", "error"],
            "imaging": ["done", "failed", "error"],
            "failed": ["pending", "error"],  # Retry allowed
            "done": [],  # Terminal
            "error": [],  # Terminal
        }
    )

    def __call__(
        self,
        *,
        query: str,
        response: dict[str, Any],
        ground_truth: dict[str, Any] | None = None,
    ) -> dict[str, float | str]:
        """Evaluate pipeline completion metrics.

        Parameters
        ----------
        query : str
            Description of the pipeline task
        response : dict
            Pipeline execution result with:
            - status: "completed" | "failed" | "partial"
            - stages_completed: list of stage names or count
            - subbands_processed: number of subbands (0-16)
            - state_transitions: list of (from_state, to_state) tuples (optional)
        ground_truth : optional
            Expected completion state

        Returns
        -------
            dict
            Scores including:
            - stage_completion_rate: fraction of stages completed
            - subband_completeness: fraction of subbands processed
            - state_machine_validity: 1.0 if all transitions valid, else penalty
            - reasoning: explanation of scores
        """
        # Extract response fields
        status = response.get("status", "unknown")
        stages = response.get("stages_completed", [])
        subbands = response.get("subbands_processed", 0)
        transitions = response.get("state_transitions", [])

        # Calculate stage completion rate
        if isinstance(stages, list):
            stages_completed = len(stages)
        else:
            stages_completed = int(stages) if stages else 0

        expected_stages = (
            ground_truth.get("expected_stages", TOTAL_STAGES) if ground_truth else TOTAL_STAGES
        )
        stage_rate = min(stages_completed / expected_stages, 1.0) if expected_stages > 0 else 0.0

        # Calculate subband completeness
        expected_subbands = (
            ground_truth.get("expected_subbands", NUM_SUBBANDS) if ground_truth else NUM_SUBBANDS
        )
        subband_rate = min(subbands / expected_subbands, 1.0) if expected_subbands > 0 else 0.0

        # Validate state machine transitions
        state_validity = self._validate_transitions(transitions)

        # Build reasoning
        reasoning_parts = [
            f"Status: {status}",
            f"Stages: {stages_completed}/{expected_stages} ({stage_rate:.1%})",
            f"Subbands: {subbands}/{expected_subbands} ({subband_rate:.1%})",
            f"State validity: {state_validity:.1%}",
        ]

        return {
            "stage_completion_rate": round(stage_rate, 4),
            "subband_completeness": round(subband_rate, 4),
            "state_machine_validity": round(state_validity, 4),
            "reasoning": "; ".join(reasoning_parts),
        }

    def _validate_transitions(self, transitions: list[tuple]) -> float:
        """
        Validate state machine transitions.

        Returns 1.0 if all transitions are valid, reduced score for invalid ones.
        """
        if not transitions:
            return 1.0  # No transitions to validate

        valid_count = 0
        for from_state, to_state in transitions:
            from_state = from_state.lower() if isinstance(from_state, str) else str(from_state)
            to_state = to_state.lower() if isinstance(to_state, str) else str(to_state)

            allowed = self.valid_transitions.get(from_state, [])
            if to_state in allowed:
                valid_count += 1
            else:
                logger.debug("Invalid transition: %s -> %s", from_state, to_state)

        return valid_count / len(transitions)


# =============================================================================
# Failure Detection Evaluator
# =============================================================================


@dataclass
class FailureDetectionEvaluator:
    """
    Evaluates failure detection and classification accuracy.

    Metrics:
        - Detection rate: fraction of actual failures detected
        - Classification accuracy: correct failure type identification
        - False positive rate: incorrect failure detections
        - Time to detection: how quickly failures are caught

    This evaluator requires ground_truth with known failure points.

    Usage:
        evaluator = FailureDetectionEvaluator()
        result = evaluator(
            query="Monitor pipeline for failures",
            response={
                "failures_detected": [
                    {"type": "conversion_error", "stage": "convert", "timestamp": 1234567890}
                ],
                "alerts_raised": 1
            },
            ground_truth={
                "expected_failures": [
                    {"type": "conversion_error", "stage": "convert", "injected_at": 1234567880}
                ]
            }
        )
    """

    # Known failure types in the pipeline
    failure_types: list[str] = field(
        default_factory=lambda: [
            "conversion_error",
            "calibration_error",
            "imaging_error",
            "database_error",
            "queue_error",
            "validation_error",
            "subband_grouping_error",
            "timeout_error",
            "resource_exhaustion",
        ]
    )

    # Maximum acceptable detection latency (seconds)
    max_detection_latency_s: float = 60.0

    def __call__(
        self,
        *,
        query: str,
        response: dict[str, Any],
        ground_truth: dict[str, Any] | None = None,
    ) -> dict[str, float | str]:
        """Evaluate failure detection performance.

        Parameters
        ----------
        query : str
            Description of the monitoring task
        response : dict
            Pipeline monitoring result with:
            - failures_detected: list of detected failure records
            - alerts_raised: count of alerts
        ground_truth : optional
            Expected failures with:
            - expected_failures: list of injected/known failures
            - false_positives_allowed: tolerance for FPs (default 0)

        Returns
        -------
            dict
            Scores including:
            - failure_detection_rate: recall (detected / expected)
            - classification_accuracy: correct type identification rate
            - false_positive_rate: FP / total detections
            - avg_detection_latency: mean time to detect (normalized)
            - reasoning: explanation
        """
        detected = response.get("failures_detected", [])
        expected = ground_truth.get("expected_failures", []) if ground_truth else []

        # Handle case with no expected failures
        if not expected:
            # If nothing expected and nothing detected = perfect
            if not detected:
                return {
                    "failure_detection_rate": 1.0,
                    "classification_accuracy": 1.0,
                    "false_positive_rate": 0.0,
                    "avg_detection_latency": 1.0,
                    "reasoning": "No failures expected, none detected - correct behavior",
                }
            # If nothing expected but something detected = all FPs
            return {
                "failure_detection_rate": 1.0,  # N/A but not a miss
                "classification_accuracy": 0.0,
                "false_positive_rate": 1.0,
                "avg_detection_latency": 0.0,
                "reasoning": f"No failures expected but {len(detected)} detected - false positives",
            }

        # Match detected failures to expected ones
        matched, unmatched_detected, unmatched_expected = self._match_failures(detected, expected)

        # Calculate metrics
        detection_rate = len(matched) / len(expected)
        false_positive_rate = len(unmatched_detected) / len(detected) if detected else 0.0

        # Classification accuracy among matched
        correct_type_count = sum(
            1 for d, e in matched if d.get("type", "").lower() == e.get("type", "").lower()
        )
        classification_accuracy = correct_type_count / len(matched) if matched else 0.0

        # Detection latency (normalized: 1.0 = instant, 0.0 = at threshold)
        latencies = []
        for det, exp in matched:
            det_time = det.get("timestamp", 0)
            exp_time = exp.get("injected_at", exp.get("timestamp", 0))
            latency = abs(det_time - exp_time)
            normalized = max(0.0, 1.0 - latency / self.max_detection_latency_s)
            latencies.append(normalized)

        avg_latency_score = statistics.mean(latencies) if latencies else 0.0

        # Build reasoning
        reasoning = (
            f"Detected {len(matched)}/{len(expected)} expected failures; "
            f"{len(unmatched_detected)} false positives; "
            f"Classification: {correct_type_count}/{len(matched)} correct types"
        )

        return {
            "failure_detection_rate": round(detection_rate, 4),
            "classification_accuracy": round(classification_accuracy, 4),
            "false_positive_rate": round(false_positive_rate, 4),
            "avg_detection_latency": round(avg_latency_score, 4),
            "reasoning": reasoning,
        }

    def _match_failures(
        self,
        detected: list[dict],
        expected: list[dict],
    ) -> tuple:
        """
        Match detected failures to expected failures.

        Uses stage + approximate timestamp matching.

        Returns
        -------
            (matched_pairs, unmatched_detected, unmatched_expected)
        """
        matched = []
        unmatched_detected = list(detected)
        unmatched_expected = list(expected)

        for exp in list(unmatched_expected):
            for det in list(unmatched_detected):
                # Match by stage
                if det.get("stage") == exp.get("stage"):
                    # Check timestamp proximity (within tolerance)
                    det_time = det.get("timestamp", 0)
                    exp_time = exp.get("injected_at", exp.get("timestamp", 0))
                    if abs(det_time - exp_time) <= self.max_detection_latency_s:
                        matched.append((det, exp))
                        unmatched_detected.remove(det)
                        unmatched_expected.remove(exp)
                        break

        return matched, unmatched_detected, unmatched_expected


# =============================================================================
# Regression Coverage Evaluator
# =============================================================================


@dataclass
class RegressionCoverageEvaluator:
    """
    Evaluates output consistency against golden baseline datasets.

    Metrics:
        - Numerical tolerance: flux values within acceptable range
        - Structural match: expected output files present
        - Metadata consistency: headers and keywords match
        - Image fidelity: RMS noise within expected bounds

    Usage:
        evaluator = RegressionCoverageEvaluator()
        result = evaluator(
            query="Compare output to golden reference",
            response={
                "output_files": ["image.fits", "catalog.csv"],
                "flux_values": {"src1": 1.234, "src2": 5.678},
                "rms_noise_jy": 0.00085,
                "metadata": {"NAXIS1": 4096, "NAXIS2": 4096}
            },
            ground_truth={
                "expected_files": ["image.fits", "catalog.csv"],
                "expected_flux": {"src1": 1.235, "src2": 5.680},
                "expected_rms_jy": 0.0009,
                "expected_metadata": {"NAXIS1": 4096, "NAXIS2": 4096},
                "flux_tolerance_pct": 5.0,
                "rms_tolerance_pct": 10.0
            }
        )
    """

    # Default tolerances
    default_flux_tolerance_pct: float = 5.0
    default_rms_tolerance_pct: float = 10.0
    default_position_tolerance_arcsec: float = 1.0

    def __call__(
        self,
        *,
        query: str,
        response: dict[str, Any],
        ground_truth: dict[str, Any] | None = None,
    ) -> dict[str, float | str]:
        """Evaluate regression against golden baseline.

        Parameters
        ----------
        query : str
            Description of the comparison task
        response : dict
            Pipeline output with:
            - output_files: list of generated files
            - flux_values: dict of source fluxes
            - rms_noise_jy: image RMS noise
            - metadata: FITS header or similar
        ground_truth : optional
            Golden baseline with:
            - expected_files: required output files
            - expected_flux: reference flux values
            - expected_rms_jy: reference noise
            - expected_metadata: reference metadata
            - flux_tolerance_pct: allowed flux deviation
            - rms_tolerance_pct: allowed noise deviation

        Returns
        -------
            dict
            Scores including:
            - structural_match: file presence score
            - flux_accuracy: flux comparison score
            - noise_fidelity: RMS comparison score
            - metadata_consistency: metadata match score
            - reasoning: explanation
        """
        if not ground_truth:
            return {
                "structural_match": 0.0,
                "flux_accuracy": 0.0,
                "noise_fidelity": 0.0,
                "metadata_consistency": 0.0,
                "reasoning": "No ground truth provided for regression comparison",
            }

        # Extract tolerances
        flux_tol = ground_truth.get("flux_tolerance_pct", self.default_flux_tolerance_pct) / 100.0
        rms_tol = ground_truth.get("rms_tolerance_pct", self.default_rms_tolerance_pct) / 100.0

        # 1. Structural match (files present)
        output_files = set(Path(f).name for f in response.get("output_files", []))
        expected_files = set(Path(f).name for f in ground_truth.get("expected_files", []))
        structural_score = self._jaccard_similarity(output_files, expected_files)

        # 2. Flux accuracy
        flux_values = response.get("flux_values", {})
        expected_flux = ground_truth.get("expected_flux", {})
        flux_score = self._compare_numeric_dict(flux_values, expected_flux, flux_tol)

        # 3. Noise fidelity
        rms = response.get("rms_noise_jy")
        expected_rms = ground_truth.get("expected_rms_jy")
        if rms is not None and expected_rms is not None and expected_rms > 0:
            rms_deviation = abs(rms - expected_rms) / expected_rms
            noise_score = 1.0 if rms_deviation <= rms_tol else max(0.0, 1.0 - rms_deviation)
        else:
            noise_score = 0.0 if expected_rms else 1.0  # No expectation = pass

        # 4. Metadata consistency
        metadata = response.get("metadata", {})
        expected_meta = ground_truth.get("expected_metadata", {})
        metadata_score = self._compare_metadata(metadata, expected_meta)

        reasoning = (
            f"Structure: {structural_score:.1%} ({len(output_files & expected_files)}/{len(expected_files)} files); "
            f"Flux: {flux_score:.1%}; "
            f"RMS: {noise_score:.1%} (actual={rms}, expected={expected_rms}); "
            f"Metadata: {metadata_score:.1%}"
        )

        return {
            "structural_match": round(structural_score, 4),
            "flux_accuracy": round(flux_score, 4),
            "noise_fidelity": round(noise_score, 4),
            "metadata_consistency": round(metadata_score, 4),
            "reasoning": reasoning,
        }

    def _jaccard_similarity(self, set1: set, set2: set) -> float:
        """Calculate Jaccard similarity between two sets."""
        if not set1 and not set2:
            return 1.0
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        return intersection / union if union > 0 else 0.0

    def _compare_numeric_dict(
        self,
        actual: dict[str, float],
        expected: dict[str, float],
        tolerance: float,
    ) -> float:
        """
        Compare numeric dictionaries within tolerance.

        Returns fraction of values within tolerance.
        """
        if not expected:
            return 1.0  # No expectations

        within_tol = 0
        for key, exp_val in expected.items():
            act_val = actual.get(key)
            if act_val is not None and exp_val != 0:
                deviation = abs(act_val - exp_val) / abs(exp_val)
                if deviation <= tolerance:
                    within_tol += 1
            elif act_val == exp_val:  # Both zero or both None
                within_tol += 1

        return within_tol / len(expected)

    def _compare_metadata(self, actual: dict, expected: dict) -> float:
        """
        Compare metadata dictionaries.

        Returns fraction of expected keys that match.
        """
        if not expected:
            return 1.0

        matching = 0
        for key, exp_val in expected.items():
            if key in actual and actual[key] == exp_val:
                matching += 1

        return matching / len(expected)


# =============================================================================
# Convenience Functions
# =============================================================================


def create_evaluators() -> dict[str, Any]:
    """
    Create all pipeline evaluators.

    Returns
    -------
        Dict mapping evaluator names to instances.
    """
    return {
        "pipeline_completion": PipelineCompletionEvaluator(),
        "failure_detection": FailureDetectionEvaluator(),
    }
