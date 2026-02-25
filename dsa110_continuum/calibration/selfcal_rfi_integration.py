"""
Integration of adaptive RFI flagging with selfcal pipeline.

This module provides:
- RFI pre-processing before each selfcal iteration
- QA-driven decision points for flagging strategies
- Provenance tracking and rollback capability
- Integration with selfcal's SNR monitoring
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from dsa110_contimg.core.calibration.rfi_adaptive_enhanced import (
    AdaptiveRFIResult,
    RFIQAThresholds,
    flag_rfi_adaptive_enhanced,
)
from dsa110_contimg.core.calibration.spw_safeguards import (
    SPWSafeguardsResult,
    SPWThresholds,
    apply_spw_safeguards,
)

logger = logging.getLogger(__name__)


@dataclass
class SelfCalRFIConfig:
    """Configuration for RFI handling in selfcal context."""

    # Pre-selfcal RFI treatment
    preflag_enabled: bool = True  # Run RFI flagging before selfcal
    preflag_enable_pass2: bool = True  # Enable surgical pass 2
    preflag_max_iterations: int = 2  # Max pass 2 iterations

    # Per-iteration RFI checks
    reflag_between_iterations: bool = False  # Re-flag between selfcal iterations
    reflag_if_snr_plateaus: bool = True  # Re-flag if SNR stops improving

    # SPW safeguards
    apply_spw_safeguards: bool = True
    enable_spw_remap: bool = True

    # QA-driven strategy selection
    use_qa_driven_strategies: bool = True

    # Provenance
    save_provenance: bool = True

    # Thresholds
    rfi_thresholds: RFIQAThresholds = field(default_factory=RFIQAThresholds)
    spw_thresholds: SPWThresholds = field(default_factory=SPWThresholds)


@dataclass
class SelfCalRFICheckpoint:
    """Checkpoint for RFI state during selfcal."""

    iteration: int
    timestamp: float
    flagged_fraction: float
    rfi_metrics: AdaptiveRFIResult
    spw_metrics: SPWSafeguardsResult | None = None
    notes: dict[str, Any] | None = None


def preflag_before_selfcal(
    ms: str,
    config: SelfCalRFIConfig | None = None,
    output_dir: str | None = None,
) -> AdaptiveRFIResult | None:
    """Run comprehensive RFI flagging before starting selfcal.

    This is the entry point for pre-processing: it applies the full
    two-pass adaptive flagging loop and SPW safeguards before selfcal begins.

    Parameters
    ----------
    ms :
        Path to measurement set
    config :
        RFI configuration (default: SelfCalRFIConfig())
    output_dir :
        Directory for logs/reports

    Returns
    -------
        AdaptiveRFIResult if successful, None if skipped or failed

    """
    if config is None:
        config = SelfCalRFIConfig()

    if not config.preflag_enabled:
        logger.info("Pre-selfcal RFI flagging disabled")
        return None

    logger.info("=" * 60)
    logger.info("Starting pre-selfcal RFI flagging")
    logger.info("=" * 60)

    output_dir = Path(output_dir or ms)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run adaptive flagging
    result = flag_rfi_adaptive_enhanced(
        ms,
        datacolumn="data",
        thresholds=config.rfi_thresholds,
        enable_pass2=config.preflag_enable_pass2,
        max_iterations_pass2=config.preflag_max_iterations,
        enable_spw_safeguards=False,  # Do SPW safeguards separately
        enable_provenance=config.save_provenance,
        output_dir=str(output_dir),
    )

    if not result.success:
        logger.error("Pre-selfcal RFI flagging failed")
        return None

    logger.info(
        f"Pre-selfcal RFI flagging: {result.initial_flag_fraction:.2%} -> "
        f"{result.final_flag_fraction:.2%} (detected {result.total_rfi_detected:.2%})"
    )

    # Apply SPW safeguards
    if config.apply_spw_safeguards:
        try:
            spw_result = apply_spw_safeguards(
                ms,
                thresholds=config.spw_thresholds,
                enable_reflag=True,
                enable_remap=config.enable_spw_remap,
                output_dir=str(output_dir),
            )

            logger.info(
                f"SPW safeguards: {spw_result.good_spws} good, "
                f"{spw_result.marginal_spws} marginal, {spw_result.bad_spws} bad"
            )

            if spw_result.spws_to_drop:
                logger.warning(
                    f"SPWs to drop: {spw_result.spws_to_drop}. Selfcal should exclude these."
                )
                result.notes["spws_to_drop"] = spw_result.spws_to_drop

            if spw_result.remapping_decisions:
                logger.info(f"SPW remapping planned: {len(spw_result.remapping_decisions)} remaps")
                result.notes["spw_remappings"] = [
                    {
                        "source": r.source_spw,
                        "target": r.target_spw,
                        "reason": r.reason,
                    }
                    for r in spw_result.remapping_decisions
                ]

        except Exception as e:
            logger.warning(f"SPW safeguards analysis failed: {e}")

    logger.info("=" * 60)

    return result


def check_rfi_during_selfcal(
    ms: str,
    iteration: int,
    previous_checkpoint: SelfCalRFICheckpoint | None = None,
    config: SelfCalRFIConfig | None = None,
    output_dir: str | None = None,
) -> SelfCalRFICheckpoint:
    """Check RFI status during selfcal iteration and optionally re-flag.

    This function:
    1. Computes current RFI metrics
    2. Compares to previous checkpoint
    3. Decides whether to re-flag based on QA
    4. Optionally performs targeted re-flagging

    Parameters
    ----------
    ms :
        Path to measurement set
    iteration :
        Current selfcal iteration number
    previous_checkpoint :
        Previous checkpoint (if any)
    config :
        RFI configuration
    output_dir :
        Directory for logs

    Returns
    -------
        SelfCalRFICheckpoint with current RFI status

    """
    if config is None:
        config = SelfCalRFIConfig()

    output_dir = Path(output_dir or ms)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = SelfCalRFICheckpoint(
        iteration=iteration,
        timestamp=time.time(),
        flagged_fraction=0.0,
        rfi_metrics=None,  # type: ignore
        notes={},
    )

    # Get current flagged fraction
    try:
        import casacore.tables as tb

        with tb.table(ms, readonly=True, ack=False) as t:
            flags = t.getcol("FLAG")
            checkpoint.flagged_fraction = float(np.mean(flags))
    except Exception as e:
        logger.warning(f"Failed to get flagged fraction: {e}")
        return checkpoint

    # Check for re-flagging opportunity
    should_reflag = False
    reflag_reason = None

    if config.reflag_between_iterations:
        # Check if flagging has degraded since last checkpoint
        if previous_checkpoint:
            flag_increase = checkpoint.flagged_fraction - previous_checkpoint.flagged_fraction

            if flag_increase > 0.05:  # >5% increase in flagging
                should_reflag = True
                reflag_reason = f"Flags increased by {flag_increase:.1%} since iteration {previous_checkpoint.iteration}"

    logger.info(
        f"Selfcal iteration {iteration}: Flagged fraction = {checkpoint.flagged_fraction:.2%}"
    )

    # Optionally perform targeted re-flagging
    if should_reflag and reflag_reason:
        logger.info(f"Re-flagging during iteration {iteration}: {reflag_reason}")

        try:
            rfi_result = flag_rfi_adaptive_enhanced(
                ms,
                datacolumn="corrected",  # Use corrected data if available
                thresholds=config.rfi_thresholds,
                enable_pass2=False,  # Lighter re-flag during iteration
                max_iterations_pass2=1,
                enable_spw_safeguards=False,
                enable_provenance=config.save_provenance,
                output_dir=str(output_dir),
            )

            checkpoint.rfi_metrics = rfi_result
            if checkpoint.notes is None:
                checkpoint.notes = {}
            checkpoint.notes["reflagging_applied"] = True
            checkpoint.notes["reflag_reason"] = reflag_reason

        except Exception as e:
            logger.warning(f"Re-flagging during iteration {iteration} failed: {e}")
            if checkpoint.notes is None:
                checkpoint.notes = {}
            checkpoint.notes["reflag_error"] = str(e)

    return checkpoint


def should_skip_selfcal_due_to_rfi(
    ms: str,
    config: SelfCalRFIConfig | None = None,
) -> tuple[bool, str]:
    """Determine if selfcal should be skipped due to excessive RFI.

    Parameters
    ----------
    ms :
        Path to measurement set
    config :
        RFI configuration

    Returns
    -------
        Tuple of (should_skip, reason_string)

    """
    if config is None:
        config = SelfCalRFIConfig()

    try:
        import casacore.tables as tb

        with tb.table(ms, readonly=True, ack=False) as t:
            flags = t.getcol("FLAG")
            flag_frac = float(np.mean(flags))

        # If more than 70% is flagged, selfcal won't work well
        if flag_frac > 0.7:
            return True, f"Too much RFI: {flag_frac:.1%} flagged (>70%)"

        # If more than 50%, warn but don't skip
        if flag_frac > 0.5:
            logger.warning(f"High RFI level: {flag_frac:.1%} flagged (>50%)")

        return False, ""

    except Exception as e:
        logger.warning(f"Failed to assess RFI for selfcal: {e}")
        return False, ""


# =============================================================================
# Integration Helpers
# =============================================================================


def get_spws_to_exclude_from_selfcal(
    spw_safeguards_result: SPWSafeguardsResult | None,
) -> str:
    """Format SPW exclusion string for selfcal based on safeguards analysis.

    Parameters
    ----------
    spw_safeguards_result: Optional[SPWSafeguardsResult] :


    Returns
    -------
        SPW string for gaincal/applycal (e.g., "!0,!2" to exclude SPWs 0 and 2)

    """
    if not spw_safeguards_result or not spw_safeguards_result.spws_to_drop:
        return ""

    # Format as "!spw1,!spw2,..."
    spw_str = ",".join(f"!{spw}" for spw in sorted(spw_safeguards_result.spws_to_drop))
    logger.info(f"Excluding SPWs from selfcal: {spw_str}")

    return spw_str


__all__ = [
    "SelfCalRFIConfig",
    "SelfCalRFICheckpoint",
    "preflag_before_selfcal",
    "check_rfi_during_selfcal",
    "should_skip_selfcal_due_to_rfi",
    "get_spws_to_exclude_from_selfcal",
]
