"""
Adaptive RFI flagging with calibration-aware strategy selection.

This module implements an adaptive flagging approach that:
1. Attempts flagging with a default (less aggressive) strategy
2. Tests if calibration succeeds after flagging
3. Falls back to more aggressive strategies if calibration fails
4. Optionally uses GPU-accelerated RFI detection

The goal is to flag enough RFI to allow successful calibration without
over-flagging good data.
"""

from __future__ import annotations

import logging
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, TypedDict

from dsa110_contimg.core.calibration.flagging import (
    flag_rfi,
    flag_zeros,
    reset_flags,
)

logger = logging.getLogger(__name__)


class CalibrationFailure(Exception):
    """

    Raises
    ------
    This
        exception is used to signal that the current flagging strategy
    is
        insufficient and a more aggressive approach should be tried

    """


@dataclass
class AdaptiveFlaggingResult:
    """Result from adaptive flagging operation."""

    success: bool
    strategy: str
    attempts: int
    flagged_fraction: float
    processing_time_s: float
    calibration_error: str | None = None
    caltables: dict[str, str] | None = None


@dataclass
class FlaggingStrategy:
    """Configuration for a flagging strategy."""

    name: str
    backend: str  # "aoflagger" or "casa"
    strategy_file: str | None = None  # For AOFlagger Lua strategies
    aggressive: bool = False
    threshold_scale: float = 1.0  # Multiplier for detection thresholds
    use_gpu: bool = False  # Whether to use GPU RFI detection


# Default adaptive strategy chain configuration.
#
# Start with a conservative AOFlagger pass, then escalate to aggressive flagging
# if calibration still fails. CASA tfcrop is a final fallback for stubborn cases.
DEFAULT_STRATEGY_CHAIN: list[FlaggingStrategy] = [
    FlaggingStrategy(
        name="default",
        backend="aoflagger",
        strategy_file=None,  # AOFlagger auto-detects optimal strategy for DSA-110
        aggressive=False,
    ),
    FlaggingStrategy(
        name="aggressive",
        backend="aoflagger",
        strategy_file=None,
        aggressive=True,
    ),
]

# Optional: Additional strategies that can be explicitly requested (not used by default)
AGGRESSIVE_STRATEGY = FlaggingStrategy(
    name="aggressive",
    backend="aoflagger",
    strategy_file=None,
    aggressive=True,
)

CASA_FALLBACK_STRATEGY = FlaggingStrategy(
    name="casa_tfcrop",
    backend="casa",
    aggressive=True,
    threshold_scale=0.8,  # Tighter thresholds
)


def _get_flag_fraction(ms_path: str) -> float:
    """Get the fraction of flagged data in a measurement set.

    Parameters
    ----------
    """
    try:
        import casacore.tables as casatables
        import numpy as np

        with casatables.table(ms_path, readonly=True) as tb:
            flags = tb.getcol("FLAG")
            if flags.size == 0:
                return 0.0
            return float(np.sum(flags) / flags.size)
    except (OSError, RuntimeError, KeyError) as e:
        logger.warning("Failed to get flag fraction: %s", e)
        return 0.0


def _apply_gpu_flagging(
    ms_path: str,
    strategy: FlaggingStrategy,
    datacolumn: str,
) -> bool:
    """Apply GPU-accelerated RFI flagging.

    Parameters
    ----------

    Returns
    -------
        True if GPU flagging succeeded, False if fallback needed.

    """
    try:
        from dsa110_contimg.core.rfi import RFIDetectionConfig, gpu_rfi_detection

        config = RFIDetectionConfig(
            threshold=5.0 / strategy.threshold_scale,
            apply_flags=True,
        )
        result = gpu_rfi_detection(ms_path, config=config)

        if result.success:
            return True
        logger.warning("GPU RFI detection failed: %s", result.error)
        return False
    except ImportError:
        logger.warning("GPU RFI detection not available, using standard flagging")
        return False


class FlaggingStrategyError(Exception):
    """

    Raises
    ------
    This
        is distinct from CalibrationFailure
    calibration
        still failed
    itself
        crashed or returned an error

    """


def _apply_flagging_strategy(
    ms_path: str,
    strategy: FlaggingStrategy,
    datacolumn: str = "data",
    skip_reset: bool = False,
) -> float:
    """Apply a flagging strategy and return the new flagged fraction.

    Parameters
    ----------
    ms_path :
        Path to measurement set
    strategy :
        Flagging strategy to apply
    datacolumn :
        Data column to flag
    skip_reset :
        If True, don't reset flags before applying (useful for retries)

    Returns
    -------
        Flagged fraction after applying strategy

    Raises
    ------
    FlaggingStrategyError
        If the flagging tool itself fails (e.g., AOFlagger crash)

    """
    logger.info("Applying flagging strategy: %s (backend=%s)", strategy.name, strategy.backend)

    # Reset flags before applying new strategy (unless skipped)
    if not skip_reset:
        reset_flags(ms_path)
        flag_zeros(ms_path, datacolumn=datacolumn)

    initial_fraction = _get_flag_fraction(ms_path)

    # Try GPU if requested, fall back to standard if it fails
    gpu_succeeded = False
    if strategy.use_gpu:
        gpu_succeeded = _apply_gpu_flagging(ms_path, strategy, datacolumn)

    # Use standard flagging if GPU not requested or failed
    if not strategy.use_gpu or not gpu_succeeded:
        try:
            flag_rfi(
                ms_path,
                datacolumn=datacolumn,
                backend=strategy.backend,
                strategy=strategy.strategy_file,
            )
        except subprocess.CalledProcessError as e:
            # AOFlagger or other tool crashed - this is a strategy failure
            error_msg = (
                f"Flagging strategy '{strategy.name}' failed with exit code {e.returncode}. "
                f"This may indicate a missing/corrupt strategy file or tool issue."
            )
            logger.warning(error_msg)
            raise FlaggingStrategyError(error_msg) from e

    final_fraction = _get_flag_fraction(ms_path)
    logger.info(
        "Strategy '%s': flagged fraction %.2f%% -> %.2f%%",
        strategy.name,
        initial_fraction * 100,
        final_fraction * 100,
    )

    return final_fraction


def flag_rfi_adaptive(
    ms_path: str,
    refant: str,
    calibrate_fn: Callable[[str, str], dict[str, str]],
    calibrate_kwargs: dict[str, Any] | None = None,
    aggressive_strategy: str | None = None,
    backend: str = "aoflagger",
    datacolumn: str = "data",
    max_attempts: int = 3,
    strategy_chain: list[FlaggingStrategy] | None = None,
    use_gpu_rfi: bool = False,
    checkpoint: Any | None = None,  # CalibrationCheckpoint instance
) -> AdaptiveFlaggingResult:
    """Perform adaptive RFI flagging with calibration-aware strategy selection.

    This function implements an iterative approach:
    1. Apply a flagging strategy
    2. Attempt calibration
    3. If calibration fails, try a more aggressive strategy
    4. Repeat until calibration succeeds or all strategies exhausted

    NEW: Supports checkpointing to avoid re-trying failed strategies on retries.

    Parameters
    ----------
    ms_path :
        Path to measurement set
    refant :
        Reference antenna for calibration
    calibrate_fn :
        Calibration function to call. Should accept (ms_path, refant, **kwargs)
        and return dict of caltables. Raises CalibrationFailure if calibration fails.
    calibrate_kwargs :
        Additional kwargs to pass to calibrate_fn
    aggressive_strategy :
        Path to aggressive AOFlagger strategy file (for backward compat)
    backend :
        Default flagging backend ("aoflagger" or "casa")
    datacolumn :
        Data column to flag
    max_attempts :
        Maximum number of flagging attempts
    strategy_chain :
        Custom strategy chain to use (default: DEFAULT_STRATEGY_CHAIN)
    use_gpu_rfi :
        Whether to try GPU RFI detection first
    checkpoint :
        Optional CalibrationCheckpoint for tracking tried strategies

    Returns
    -------
        AdaptiveFlaggingResult with success status, strategy used, and statistics

    """
    start_time = time.time()
    calibrate_kwargs = calibrate_kwargs or {}

    # Build strategy chain
    if strategy_chain is None:
        strategy_chain = list(DEFAULT_STRATEGY_CHAIN)  # Copy to avoid mutation

        # Override aggressive strategy if provided
        if aggressive_strategy:
            for s in strategy_chain:
                if s.name == "aggressive":
                    s.strategy_file = aggressive_strategy

    # Optionally prepend GPU strategy
    if use_gpu_rfi:
        gpu_strategy = FlaggingStrategy(
            name="gpu_default",
            backend="aoflagger",
            use_gpu=True,
            aggressive=False,
        )
        strategy_chain.insert(0, gpu_strategy)

    # Limit to max_attempts
    strategy_chain = strategy_chain[:max_attempts]

    # Filter out already-tried strategies if checkpoint provided
    if checkpoint is not None:
        all_names = [s.name for s in strategy_chain]
        untried_names = checkpoint.get_untried_strategies(all_names)
        if len(untried_names) < len(all_names):
            logger.info(
                "Checkpoint found %d already-tried strategies, skipping: %s",
                len(all_names) - len(untried_names),
                [n for n in all_names if n not in untried_names],
            )
            strategy_chain = [s for s in strategy_chain if s.name in untried_names]

        # Check if we already have a successful strategy
        if checkpoint.has_successful_strategy():
            success_name = checkpoint.get_successful_strategy()
            logger.info(
                "Checkpoint indicates strategy '%s' already succeeded, skipping adaptive flagging",
                success_name,
            )
            return AdaptiveFlaggingResult(
                success=True,
                strategy=success_name,
                attempts=0,
                flagged_fraction=0.0,
                calibration_error=None,
                processing_time_s=0.0,
            )

    if not strategy_chain:
        logger.error("No flagging strategies available (all already tried)")
        return AdaptiveFlaggingResult(
            success=False,
            strategy="none",
            attempts=0,
            flagged_fraction=0.0,
            calibration_error="All strategies already exhausted",
            processing_time_s=time.time() - start_time,
        )

    last_error: str | None = None
    final_flagged_fraction = 0.0

    for attempt, strategy in enumerate(strategy_chain, 1):
        strategy_start = time.time()
        logger.info(
            "Adaptive flagging attempt %d/%d: %s", attempt, len(strategy_chain), strategy.name
        )

        try:
            # Apply flagging strategy
            final_flagged_fraction = _apply_flagging_strategy(
                ms_path, strategy, datacolumn=datacolumn
            )

            # Attempt calibration
            logger.info("Testing calibration after %s flagging...", strategy.name)
            caltables = calibrate_fn(ms_path, refant, **calibrate_kwargs)

            # Calibration succeeded!
            logger.info("Calibration succeeded with strategy: %s", strategy.name)

            # Record success in checkpoint
            if checkpoint is not None:
                checkpoint.mark_strategy_tried(
                    strategy.name,
                    success=True,
                    flagged_fraction=final_flagged_fraction,
                    duration_s=time.time() - strategy_start,
                )

            return AdaptiveFlaggingResult(
                success=True,
                strategy=strategy.name,
                attempts=attempt,
                flagged_fraction=final_flagged_fraction,
                calibration_error=None,
                processing_time_s=time.time() - start_time,
                caltables=caltables,
            )

        except FlaggingStrategyError as e:
            # FAIL-FAST: Flagging tool itself failed (e.g., AOFlagger not found)
            # This is a hard failure - no fallbacks, give actionable diagnostics
            last_error = str(e)
            logger.error(
                "FAIL-FAST: Flagging tool failed for strategy '%s': %s\n"
                "Action: Fix the tool issue before retrying. "
                "Run preflight_check_aoflagger() to diagnose.",
                strategy.name,
                e,
            )
            if checkpoint is not None:
                checkpoint.mark_strategy_tried(
                    strategy.name,
                    success=False,
                    error=str(e),
                    duration_s=time.time() - strategy_start,
                )
            # FAIL-FAST: Return immediately with actionable error
            return AdaptiveFlaggingResult(
                success=False,
                strategy=strategy.name,
                attempts=attempt,
                flagged_fraction=final_flagged_fraction,
                calibration_error=f"Flagging tool error (run preflight check): {last_error}",
                processing_time_s=time.time() - start_time,
            )

        except CalibrationFailure as e:
            last_error = str(e)
            logger.warning(
                "Calibration failed with strategy '%s': %s. Trying next strategy if available.",
                strategy.name,
                e,
            )
            if checkpoint is not None:
                checkpoint.mark_strategy_tried(
                    strategy.name,
                    success=False,
                    error=str(e),
                    flagged_fraction=final_flagged_fraction,
                    duration_s=time.time() - strategy_start,
                )
            continue

        except (OSError, RuntimeError, ValueError) as e:
            # FAIL-FAST: Unexpected error - fail immediately with full traceback
            last_error = str(e)
            logger.error(
                "FAIL-FAST: Unexpected error during flagging attempt %d: %s",
                attempt,
                e,
                exc_info=True,
            )
            if checkpoint is not None:
                checkpoint.mark_strategy_tried(
                    strategy.name,
                    success=False,
                    error=str(e),
                    duration_s=time.time() - strategy_start,
                )
            # FAIL-FAST: Return immediately with error details
            return AdaptiveFlaggingResult(
                success=False,
                strategy=strategy.name,
                attempts=attempt,
                flagged_fraction=final_flagged_fraction,
                calibration_error=f"Unexpected error (check logs): {last_error}",
                processing_time_s=time.time() - start_time,
            )

    return AdaptiveFlaggingResult(
        success=False,
        strategy=strategy_chain[-1].name if strategy_chain else "none",
        attempts=len(strategy_chain),
        flagged_fraction=final_flagged_fraction,
        calibration_error=last_error or "No flagging result produced",
        processing_time_s=time.time() - start_time,
    )


def flag_rfi_with_gpu_fallback(
    ms_path: str,
    *,
    threshold: float = 5.0,
    backend: str = "aoflagger",
    strategy: str | None = None,
    prefer_gpu: bool = True,
) -> dict[str, Any]:
    """Flag RFI with automatic GPU/CPU fallback.

    Attempts GPU-accelerated flagging first (if available and preferred),
    falling back to standard AOFlagger/CASA flagging if GPU fails.

    Parameters
    ----------
    ms_path :
        Path to measurement set
    threshold :
        Detection threshold in MAD units (for GPU)
    backend :
        Fallback backend ("aoflagger" or "casa")
    strategy :
        AOFlagger strategy file path
    prefer_gpu :
        Whether to try GPU first

    Returns
    -------
        Dict with flagging results and method used

    """
    result: dict[str, Any] = {
        "method": "unknown",
        "success": False,
        "flagged_fraction": 0.0,
        "error": None,
    }

    # Try GPU first if preferred
    if prefer_gpu:
        gpu_result = _try_gpu_flagging(ms_path, threshold, result)
        if gpu_result is not None:
            return gpu_result

    # Fall back to standard flagging
    logger.info("Using standard RFI flagging (backend=%s)", backend)
    try:
        initial_fraction = _get_flag_fraction(ms_path)
        flag_rfi(ms_path, backend=backend, strategy=strategy)
        final_fraction = _get_flag_fraction(ms_path)

        result["method"] = backend
        result["success"] = True
        result["flagged_fraction"] = final_fraction
        result["rfi_flagged"] = final_fraction - initial_fraction

    except (OSError, RuntimeError, subprocess.CalledProcessError) as e:
        result["error"] = str(e)
        logger.error("Standard RFI flagging failed: %s", e)

    return result


def _try_gpu_flagging(
    ms_path: str,
    threshold: float,
    result: dict[str, Any],
) -> dict[str, Any] | None:
    """Try GPU-accelerated flagging.

    Parameters
    ----------

    Returns
    -------
        Updated result dict if GPU succeeded, None if fallback needed.

    """
    try:
        from dsa110_contimg.core.rfi import RFIDetectionConfig, gpu_rfi_detection
        from dsa110_contimg.core.rfi.gpu_detection import CUPY_AVAILABLE

        if not CUPY_AVAILABLE:
            logger.debug("CuPy not available, skipping GPU RFI detection")
            return None

        logger.info("Attempting GPU-accelerated RFI flagging...")
        config = RFIDetectionConfig(
            threshold=threshold,
            apply_flags=True,
        )
        gpu_result = gpu_rfi_detection(ms_path, config=config)

        if gpu_result.success:
            result["method"] = "gpu"
            result["success"] = True
            result["flagged_fraction"] = gpu_result.flag_percent / 100.0
            result["processing_time_s"] = gpu_result.processing_time_s
            logger.info(
                "GPU RFI flagging complete: %.2f%% flagged in %.2fs",
                gpu_result.flag_percent,
                gpu_result.processing_time_s,
            )
            return result

        logger.warning("GPU RFI flagging failed: %s", gpu_result.error)
        return None

    except ImportError:
        logger.debug("GPU RFI module not available")
        return None
    except (OSError, RuntimeError) as e:
        logger.warning("GPU RFI flagging failed: %s", e)
        return None
