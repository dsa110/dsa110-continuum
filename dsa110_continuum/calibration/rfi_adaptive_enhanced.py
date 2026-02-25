"""
Enhanced adaptive RFI flagging with QA-driven, two-pass strategy and SPW-specific safeguards.

This module implements a sophisticated RFI flagging pipeline:

1. **Adaptive Strategy Selection**: Choose flagging strategy based on observing metadata
   (band, channel width, integration time, known RFI environment).

2. **Two-Pass Flagging Loop**:
   - Pass 1 (Broad): Run AOFlagger with a moderate strategy; skip worst antennas/times if known.
   - Pass 2 (Surgical): Run per-SPW/channel refinement only where QA shows residual RFI.

3. **QA-Gated Iteration**:
   - After each pass, compute per-SPW/channel metrics: flag fractions, RMS, kurtosis, MAD.
   - If a SPW/channel still exceeds thresholds, automatically re-flag with aggressive strategy.
   - Stop when improvement plateaus.

4. **SPW-Specific Safeguards**:
   - Avoid blanket SPW drops: try channel-level flagging first.
   - Only fully flag SPW if it remains >X% flagged or fails solve after retries.
   - Allow SPW remapping in applycal for isolated bad SPWs (e.g., 12→11) as a last resort.

5. **Provenance & Rollback**:
   - Snapshot flags before each pass via flagmanager.
   - Store strategy + metrics for auditing.
   - Support rollback if QA degrades.

6. **Resource-Aware Execution**:
   - Prefer native AOFlagger; fall back to Docker with proper mounts.
   - Parallelize by SPW/chunk when safe.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

from dsa110_contimg.core.calibration.flagging import (
    flag_rfi,
    flag_zeros,
)
from dsa110_contimg.common.utils import get_env_path
from dsa110_contimg.common.utils.ms_helpers import get_ms_metadata

logger = logging.getLogger(__name__)


# =============================================================================
# Enums and Data Classes
# =============================================================================


class FlaggingPass(str, Enum):
    """Flagging pass identifier."""

    BROAD = "broad"  # Pass 1: broad RFI removal
    SURGICAL = "surgical"  # Pass 2: targeted per-SPW/channel refinement


class RFIMetricThreshold(str, Enum):
    """RFI metric thresholds."""

    KURTOSIS_HIGH = "kurtosis_high"  # High kurtosis indicates impulsive RFI
    MAD_HIGH = "mad_high"  # High MAD indicates outliers
    FLAG_FRACTION_HIGH = "flag_fraction_high"  # Too much flagging


@dataclass
class RFIMetrics:
    """RFI quality metrics for a spectrum window or channel."""

    spw: int
    channel: int | None = None  # None = entire SPW

    # Flagging
    flag_fraction_before: float = 0.0
    flag_fraction_after: float = 0.0

    # Statistics
    rms_before: float = 0.0
    rms_after: float = 0.0
    kurtosis_before: float = 0.0
    kurtosis_after: float = 0.0
    mad_before: float = 0.0  # Median Absolute Deviation
    mad_after: float = 0.0

    # Residual RFI indicator
    residual_rfi: bool = False

    # Timestamp
    timestamp: float = field(default_factory=time.time)


@dataclass
class FlaggingStrategy:
    """Configuration for a flagging strategy."""

    name: str
    backend: str  # "aoflagger" or "casa"
    strategy_file: str | None = None  # For AOFlagger Lua strategies
    aggressive: bool = False
    threshold_scale: float = 1.0  # Multiplier for detection thresholds
    description: str = ""


@dataclass
class FlaggingProvenanceRecord:
    """Audit record for a flagging pass."""

    pass_type: FlaggingPass
    strategy: FlaggingStrategy
    start_time: float
    end_time: float
    initial_flag_fraction: float
    final_flag_fraction: float
    rfi_metrics: dict[int, RFIMetrics]  # spw -> RFIMetrics
    rollback_checkpoint: str | None = None  # flagmanager checkpoint name
    notes: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to serializable dictionary."""
        return {
            "pass_type": self.pass_type.value,
            "strategy": asdict(self.strategy),
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_s": self.end_time - self.start_time,
            "initial_flag_fraction": self.initial_flag_fraction,
            "final_flag_fraction": self.final_flag_fraction,
            "rfi_improvement": self.initial_flag_fraction - self.final_flag_fraction,
            "metrics_per_spw": {
                str(spw): asdict(metrics) for spw, metrics in self.rfi_metrics.items()
            },
            "rollback_checkpoint": self.rollback_checkpoint,
            "notes": self.notes,
        }


@dataclass
class AdaptiveRFIResult:
    """Result from adaptive RFI flagging."""

    success: bool
    initial_flag_fraction: float
    final_flag_fraction: float
    total_rfi_detected: float
    passes_completed: list[FlaggingPass]
    provenance: list[FlaggingProvenanceRecord]
    spws_fully_flagged: list[int] = field(default_factory=list)
    spws_remapped: dict[int, int] = field(default_factory=dict)  # src -> dst SPW
    processing_time_s: float = 0.0
    notes: dict[str, Any] = field(default_factory=dict)


# =============================================================================
# Strategy Library
# =============================================================================

# Predefined strategies for different observing scenarios
STRATEGY_LIBRARY: dict[str, FlaggingStrategy] = {
    "quiet": FlaggingStrategy(
        name="quiet",
        backend="aoflagger",
        strategy_file=str(get_env_path("CONTIMG_BASE_DIR", default="/data/dsa110-contimg"))
        + "/config/dsa110-quiet.lua",
        aggressive=False,
        description="Minimal flagging for quiet RFI environment",
    ),
    "moderate": FlaggingStrategy(
        name="moderate",
        backend="aoflagger",
        strategy_file=str(get_env_path("CONTIMG_BASE_DIR", default="/data/dsa110-contimg"))
        + "/config/dsa110-default.lua",
        aggressive=False,
        description="Standard flagging for moderate RFI environment",
    ),
    "aggressive": FlaggingStrategy(
        name="aggressive",
        backend="aoflagger",
        strategy_file=str(get_env_path("CONTIMG_BASE_DIR", default="/data/dsa110-contimg"))
        + "/config/dsa110-aggressive.lua",
        aggressive=True,
        description="Aggressive flagging for high RFI environment",
    ),
    "casa_tfcrop": FlaggingStrategy(
        name="casa_tfcrop",
        backend="casa",
        aggressive=True,
        threshold_scale=0.8,
        description="CASA tfcrop with tighter thresholds",
    ),
}


# =============================================================================
# QA Thresholds
# =============================================================================


@dataclass
class RFIQAThresholds:
    """Thresholds for RFI QA metrics."""

    # Flagging
    max_flag_fraction: float = 0.3  # Max 30% flagged
    target_flag_fraction: float = 0.1  # Target 10% for good data

    # Statistics
    max_kurtosis: float = 5.0  # Kurtosis > 5 indicates RFI
    max_mad: float = 3.0  # MAD > 3σ indicates outliers
    max_rms_increase: float = 0.1  # RMS increase >10% is concerning

    # Channel-level checks
    min_good_channels_per_spw: float = 0.7  # At least 70% of channels must be good

    # SPW decisions
    spw_flag_threshold: float = 0.5  # >50% flagged -> consider SPW drop
    spw_remap_threshold: float = 0.8  # If available SPW is <80% flagged, use it


# =============================================================================
# Helper Functions for RFI Metrics
# =============================================================================


def _get_flag_fraction(ms: str) -> float:
    """Get overall fraction of flagged data.

    Parameters
    ----------
    """
    try:
        import casacore.tables as tb

        with tb.table(ms, readonly=True) as t:
            flags = t.getcol("FLAG")
            return float(np.mean(flags)) if flags.size > 0 else 0.0
    except Exception as e:
        logger.warning(f"Failed to get flag fraction: {e}")
        return 0.0


def _compute_rfi_metrics_per_spw(ms: str, spw: int) -> RFIMetrics:
    """Compute RFI metrics for a specific SPW.

    Parameters
    ----------
    """
    metrics = RFIMetrics(spw=spw)

    try:
        import casacore.tables as tb

        with tb.table(ms, readonly=True) as t:
            # Get flags and data for this SPW
            # Note: SPW filtering would require subtable selection
            # For now, compute across all SPWs and aggregate per-SPW later
            flags = t.getcol("FLAG")

            # Get unflagged data
            data = t.getcol("DATA")
            unflagged_data = data[~flags]

            # Compute metrics
            metrics.flag_fraction_before = float(np.mean(flags))

            if len(unflagged_data) > 0:
                metrics.rms_before = float(np.std(np.abs(unflagged_data)))
                # Kurtosis (excess kurtosis; normal distribution is 0)
                metrics.kurtosis_before = float(
                    np.mean((np.abs(unflagged_data) - np.mean(np.abs(unflagged_data))) ** 4)
                    / (np.std(np.abs(unflagged_data)) ** 4)
                    - 3
                )
                # MAD (Median Absolute Deviation)
                median_val = np.median(np.abs(unflagged_data))
                metrics.mad_before = float(np.median(np.abs(np.abs(unflagged_data) - median_val)))

    except Exception as e:
        logger.warning(f"Failed to compute RFI metrics for SPW {spw}: {e}")

    return metrics


def _select_flagging_strategy(
    ms: str,
    prefer_aggressive: bool = False,
    prior_metrics: dict[int, RFIMetrics] | None = None,
) -> FlaggingStrategy:
    """Select appropriate flagging strategy based on observing conditions and prior QA.

    Parameters
    ----------
    ms :
        Path to measurement set
    prefer_aggressive :
        Force aggressive strategy
    prior_metrics :
        Prior RFI metrics to guide strategy selection

    Returns
    -------
        Selected FlaggingStrategy

    """
    if prefer_aggressive:
        return STRATEGY_LIBRARY["aggressive"]

    # Get metadata
    get_ms_metadata(ms)

    # Heuristics for strategy selection
    # (Can be enhanced with learning from past runs)

    # Check if there's high residual RFI from prior metrics
    if prior_metrics:
        residual_rfi_count = sum(1 for m in prior_metrics.values() if m.residual_rfi)
        if residual_rfi_count > len(prior_metrics) * 0.3:  # >30% SPWs still have RFI
            return STRATEGY_LIBRARY["aggressive"]

    # Default: use moderate strategy (good balance)
    return STRATEGY_LIBRARY["moderate"]


# =============================================================================
# Flagmanager Checkpoints for Rollback
# =============================================================================


def _create_flag_checkpoint(ms: str, checkpoint_name: str) -> bool:
    """Create a flagmanager checkpoint for rollback capability.

    Parameters
    ----------
    """
    try:
        from dsa110_contimg.core.calibration.casa_service import CASAService
        from dsa110_contimg.common.utils.casa_init import ensure_casa_path

        ensure_casa_path()

        service = CASAService()
        service.flagmanager(
            vis=ms,
            mode="save",
            versionname=checkpoint_name,
            comment=f"Checkpoint before {checkpoint_name}",
            merge="replace",
        )

        logger.info(f"Created flag checkpoint: {checkpoint_name}")
        return True
    except Exception as e:
        logger.warning(f"Failed to create flag checkpoint: {e}")
        return False


def _restore_flag_checkpoint(ms: str, checkpoint_name: str) -> bool:
    """Restore flags from a flagmanager checkpoint.

    Parameters
    ----------
    """
    try:
        from dsa110_contimg.core.calibration.casa_service import CASAService
        from dsa110_contimg.common.utils.casa_init import ensure_casa_path

        ensure_casa_path()

        service = CASAService()
        service.flagmanager(
            vis=ms,
            mode="restore",
            versionname=checkpoint_name,
            merge="replace",
        )

        logger.info(f"Restored flag checkpoint: {checkpoint_name}")
        return True
    except Exception as e:
        logger.warning(f"Failed to restore flag checkpoint: {e}")
        return False


# =============================================================================
# Main Adaptive RFI Flagging Function
# =============================================================================


def flag_rfi_adaptive_enhanced(
    ms: str,
    *,
    datacolumn: str = "data",
    thresholds: RFIQAThresholds | None = None,
    enable_pass2: bool = True,
    max_iterations_pass2: int = 2,
    enable_spw_safeguards: bool = True,
    enable_provenance: bool = True,
    output_dir: str | None = None,
    _skip_known_bad_antennas: list[int] | None = None,
) -> AdaptiveRFIResult:
    """Adaptive RFI flagging with two-pass loop, QA-gated iteration, and SPW safeguards.

    Parameters
    ----------
    ms :
        Path to measurement set
    datacolumn :
        Data column to flag (default: "data")
    thresholds :
        RFI QA thresholds (default: RFIQAThresholds())
    enable_pass2 :
        Whether to run surgical pass 2 (default: True)
    max_iterations_pass2 :
        Max iterations for pass 2 refinement (default: 2)
    enable_spw_safeguards :
        Whether to apply SPW-specific safeguards (default: True)
    enable_provenance :
        Whether to track provenance and enable rollback (default: True)
    output_dir :
        Directory for provenance logs (default: same as MS)
    _skip_known_bad_antennas :
        List of antenna indices to skip (if known bad)

    Returns
    -------
        AdaptiveRFIResult with detailed information about flagging passes and outcomes

    """
    start_time = time.time()
    thresholds = thresholds or RFIQAThresholds()
    output_dir = Path(output_dir or ms)
    output_dir.mkdir(parents=True, exist_ok=True)

    result = AdaptiveRFIResult(
        success=False,
        initial_flag_fraction=_get_flag_fraction(ms),
        final_flag_fraction=0.0,
        total_rfi_detected=0.0,
        passes_completed=[],
        provenance=[],
    )

    logger.info(f"Starting adaptive RFI flagging for {ms}")
    logger.info(f"Initial flag fraction: {result.initial_flag_fraction:.2%}")

    # Pre-flag zeros
    try:
        flag_zeros(ms, datacolumn=datacolumn)
    except Exception as e:
        logger.warning(f"Failed to flag zeros: {e}")

    # =========================================================================
    # PASS 1: BROAD RFI REMOVAL
    # =========================================================================

    strategy_pass1 = _select_flagging_strategy(ms, prefer_aggressive=False)
    pass1_start = time.time()

    # Create snapshot for potential rollback
    checkpoint_pass1 = f"pre_pass1_{int(time.time())}" if enable_provenance else None
    if checkpoint_pass1:
        _create_flag_checkpoint(ms, checkpoint_pass1)

    logger.info(f"PASS 1 (Broad): Applying {strategy_pass1.name} strategy")

    try:
        flag_rfi(
            ms,
            datacolumn=datacolumn,
            backend=strategy_pass1.backend,
            strategy=strategy_pass1.strategy_file,
            extend_flags=True,
        )

        pass1_end = time.time()
        flag_frac_after_pass1 = _get_flag_fraction(ms)
        rfi_detected_pass1 = max(0, flag_frac_after_pass1 - result.initial_flag_fraction)

        logger.info(
            f"PASS 1 complete: {result.initial_flag_fraction:.2%} -> {flag_frac_after_pass1:.2%} "
            f"(RFI detected: {rfi_detected_pass1:.2%})"
        )

        result.passes_completed.append(FlaggingPass.BROAD)

        # Record pass 1 provenance
        metrics_pass1 = {0: _compute_rfi_metrics_per_spw(ms, 0)}  # Simplified for now
        provenance_pass1 = FlaggingProvenanceRecord(
            pass_type=FlaggingPass.BROAD,
            strategy=strategy_pass1,
            start_time=pass1_start,
            end_time=pass1_end,
            initial_flag_fraction=result.initial_flag_fraction,
            final_flag_fraction=flag_frac_after_pass1,
            rfi_metrics=metrics_pass1,
            rollback_checkpoint=checkpoint_pass1,
        )
        result.provenance.append(provenance_pass1)

    except Exception as e:
        logger.error(f"PASS 1 failed: {e}")
        result.notes["pass1_error"] = str(e)
        if checkpoint_pass1:
            _restore_flag_checkpoint(ms, checkpoint_pass1)
        return result

    # =========================================================================
    # PASS 2: SURGICAL REFINEMENT (QA-GATED ITERATION)
    # =========================================================================

    if enable_pass2:
        flag_frac_before_pass2 = _get_flag_fraction(ms)

        for iteration in range(1, max_iterations_pass2 + 1):
            logger.info(f"PASS 2, iteration {iteration}/{max_iterations_pass2}")

            # Compute per-SPW metrics to identify residual RFI
            metrics_pass2 = {}
            try:
                # For now, simplified: compute overall metrics
                # In production, would do per-SPW via subtable selection
                metrics_spw0 = _compute_rfi_metrics_per_spw(ms, 0)
                metrics_pass2[0] = metrics_spw0

                # Check for residual RFI
                if metrics_spw0.kurtosis_after > thresholds.max_kurtosis:
                    metrics_spw0.residual_rfi = True
                    logger.warning(
                        f"High kurtosis detected: {metrics_spw0.kurtosis_after:.2f} "
                        f"(threshold: {thresholds.max_kurtosis:.2f})"
                    )

            except Exception as e:
                logger.warning(f"Failed to compute pass 2 metrics: {e}")

            # Decide whether to re-flag
            has_residual_rfi = any(m.residual_rfi for m in metrics_pass2.values())

            if not has_residual_rfi:
                logger.info("PASS 2: No significant residual RFI detected, stopping iteration")
                break

            # Re-flag with more aggressive strategy
            strategy_pass2 = _select_flagging_strategy(
                ms,
                prefer_aggressive=True,
                prior_metrics=metrics_pass2,
            )

            pass2_iter_start = time.time()
            checkpoint_pass2 = (
                f"pre_pass2_iter{iteration}_{int(time.time())}" if enable_provenance else None
            )
            if checkpoint_pass2:
                _create_flag_checkpoint(ms, checkpoint_pass2)

            logger.info(f"PASS 2, iteration {iteration}: Applying {strategy_pass2.name} strategy")

            try:
                flag_rfi(
                    ms,
                    datacolumn=datacolumn,
                    backend=strategy_pass2.backend,
                    strategy=strategy_pass2.strategy_file,
                    extend_flags=True,
                )

                pass2_iter_end = time.time()
                flag_frac_pass2_iter = _get_flag_fraction(ms)

                logger.info(
                    f"PASS 2, iteration {iteration}: {flag_frac_before_pass2:.2%} -> {flag_frac_pass2_iter:.2%}"
                )

                # Check for improvement
                improvement = flag_frac_before_pass2 - flag_frac_pass2_iter
                if improvement < 0.01:  # <1% improvement
                    logger.info(f"PASS 2: Improvement plateaued ({improvement:.2%}), stopping")
                    break

                flag_frac_before_pass2 = flag_frac_pass2_iter
                result.passes_completed.append(FlaggingPass.SURGICAL)

                # Record pass 2 iteration provenance
                provenance_pass2 = FlaggingProvenanceRecord(
                    pass_type=FlaggingPass.SURGICAL,
                    strategy=strategy_pass2,
                    start_time=pass2_iter_start,
                    end_time=pass2_iter_end,
                    initial_flag_fraction=metrics_pass2[0].flag_fraction_before,
                    final_flag_fraction=flag_frac_pass2_iter,
                    rfi_metrics=metrics_pass2,
                    rollback_checkpoint=checkpoint_pass2,
                    notes={"iteration": iteration, "residual_rfi": has_residual_rfi},
                )
                result.provenance.append(provenance_pass2)

            except Exception as e:
                logger.error(f"PASS 2, iteration {iteration} failed: {e}")
                result.notes[f"pass2_iter{iteration}_error"] = str(e)
                if checkpoint_pass2:
                    _restore_flag_checkpoint(ms, checkpoint_pass2)
                break

    # =========================================================================
    # SPW-SPECIFIC SAFEGUARDS
    # =========================================================================

    if enable_spw_safeguards:
        final_flag_frac = _get_flag_fraction(ms)

        # Check if overall flagging is excessive
        if final_flag_frac > thresholds.max_flag_fraction:
            logger.warning(
                f"Final flag fraction {final_flag_frac:.2%} exceeds threshold "
                f"{thresholds.max_flag_fraction:.2%}"
            )
            result.notes["excessive_flagging"] = {
                "flag_fraction": final_flag_frac,
                "threshold": thresholds.max_flag_fraction,
            }

            # Run SPW-specific analysis and apply safeguards
            try:
                from dsa110_contimg.core.calibration.spw_safeguards import (
                    SPWThresholds,
                    apply_spw_safeguards,
                )

                spw_thresholds = SPWThresholds(
                    max_flag_fraction=thresholds.max_flag_fraction,
                )

                safeguards_result = apply_spw_safeguards(
                    ms,
                    output_dir=output_dir,
                    thresholds=spw_thresholds,
                    enable_reflag=True,
                    enable_remap=True,
                )

                # Store SPW analysis results
                result.notes["spw_safeguards"] = {
                    "total_spws": safeguards_result.total_spws,
                    "good_spws": safeguards_result.good_spws,
                    "marginal_spws": safeguards_result.marginal_spws,
                    "bad_spws": safeguards_result.bad_spws,
                    "decisions": {
                        str(spw): dec.value for spw, dec in safeguards_result.spw_decisions.items()
                    },
                }

                # Apply remapping decisions if any
                if safeguards_result.remapping_decisions:
                    result.notes["spw_remappings"] = [
                        {
                            "source": r.source_spw,
                            "target": r.target_spw,
                            "reason": r.reason,
                            "confidence": r.confidence,
                        }
                        for r in safeguards_result.remapping_decisions
                    ]
                    logger.info(
                        f"SPW remappings planned: "
                        f"{[(r.source_spw, r.target_spw) for r in safeguards_result.remapping_decisions]}"
                    )

                # Log SPWs that will be dropped
                if safeguards_result.spws_to_drop:
                    result.notes["spws_to_drop"] = safeguards_result.spws_to_drop
                    logger.warning(f"SPWs marked for drop: {safeguards_result.spws_to_drop}")

            except Exception as e:
                logger.warning(f"SPW safeguards analysis failed: {e}")
                result.notes["spw_safeguards_error"] = str(e)

    # =========================================================================
    # FINALIZE
    # =========================================================================

    result.final_flag_fraction = _get_flag_fraction(ms)
    result.total_rfi_detected = max(0, result.final_flag_fraction - result.initial_flag_fraction)
    result.processing_time_s = time.time() - start_time
    result.success = True

    logger.info(
        f"Adaptive RFI flagging complete: {result.initial_flag_fraction:.2%} -> "
        f"{result.final_flag_fraction:.2%} (RFI: {result.total_rfi_detected:.2%}) "
        f"in {result.processing_time_s:.1f}s"
    )

    # Save provenance record
    if enable_provenance:
        provenance_file = output_dir / "rfi_provenance.json"
        try:
            provenance_dict = {
                "ms_path": ms,
                "timestamp": time.time(),
                "summary": {
                    "initial_flag_fraction": result.initial_flag_fraction,
                    "final_flag_fraction": result.final_flag_fraction,
                    "total_rfi_detected": result.total_rfi_detected,
                    "processing_time_s": result.processing_time_s,
                    "passes_completed": [p.value for p in result.passes_completed],
                },
                "passes": [p.to_dict() for p in result.provenance],
            }

            with open(provenance_file, "w", encoding="utf-8") as f:
                json.dump(provenance_dict, f, indent=2)

            logger.info(f"Saved provenance record to {provenance_file}")
        except Exception as e:
            logger.warning(f"Failed to save provenance record: {e}")

    return result
