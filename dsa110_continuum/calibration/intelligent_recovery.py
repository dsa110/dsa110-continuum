"""Intelligent calibration error recovery.

    Analyzes calibration failures and applies domain-specific recovery strategies:
    - Singular matrix → Try relaxed flagging or alternate refant.
    - Low SNR → Try longer solint, all fields, or lower minsnr threshold.
    - Phase instability → Try shorter solint or different refant.
    - Excessive flagging → Try GPU-accelerated RFI detection.
    - Bad refant → Auto-select healthy refant from antenna analysis.

Examples
--------
    >>> from dsa110_contimg.core.calibration.intelligent_recovery import CalibrationRecoveryManager
    >>>
    >>> # Define calibration function
    >>> def calibrate(ms_path, **params):
    >>>     return solve_calibration_tables(ms_path, **params)
    >>>
    >>> # Attempt calibration with recovery
    >>> recovery_mgr = CalibrationRecoveryManager(ms_path, cal_params)
    >>>
    >>> try:
    >>>     result = calibrate(ms_path, **cal_params)
    >>> except Exception as e:
    >>>     success, recovered_params = recovery_mgr.attempt_recovery(
    >>>         calibrate, error=e, max_recovery_attempts=3
    >>>     )
    >>>     if success:
    >>>         result = calibrate(ms_path, **recovered_params)
    >>>     else:
    >>>         raise
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class RecoveryStrategy:
    """Calibration recovery strategy."""

    name: str
    description: str
    apply_fn: Callable[[], dict[str, Any]]
    max_attempts: int = 1


class CalibrationRecoveryManager:
    """Manages intelligent calibration error recovery.

        This class analyzes calibration errors and applies domain-specific
        recovery strategies based on the error type. It tracks attempt history
        and provides detailed logging for debugging.

    Examples
    --------
        manager = CalibrationRecoveryManager(ms_path, original_params)

        try:
        calibrate(ms_path, **original_params)
        except Exception as e:
        success, params = manager.attempt_recovery(calibrate, e)
        if success:
        logger.info("Recovered with params: %s", params)
    """

    def __init__(self, ms_path: str, original_params: dict[str, Any]):
        """Initialize recovery manager.

        Parameters
        ----------
        ms_path : str
            Path to Measurement Set
        original_params : dict
            Original calibration parameters
        """
        self.ms_path = ms_path
        self.original_params = original_params.copy()
        self.attempt_history: list[dict[str, Any]] = []

    def analyze_error(self, error: Exception) -> str:
        """Classify calibration error by type.

        Parameters
        ----------
        error : Exception
            Exception raised during calibration

        """
        error_str = str(error).lower()
        error_type_name = type(error).__name__.lower()

        # Check error message and exception type
        if "singular" in error_str or "matrix" in error_str:
            return "singular_matrix"
        elif "snr" in error_str or "signal" in error_str or "noise" in error_str:
            return "low_snr"
        elif "phase" in error_str or "wrap" in error_str or "unwrap" in error_str:
            return "phase_instability"
        elif "flag" in error_str or "insufficient" in error_str or "no data" in error_str:
            return "excessive_flagging"
        elif "refant" in error_str or "reference antenna" in error_str:
            return "bad_refant"
        elif "convergence" in error_str or "diverge" in error_str:
            return "convergence_failure"
        else:
            logger.debug(f"Unknown error type: {error_type_name}, message: {error_str[:100]}")
            return "unknown"

    def get_recovery_strategies(self, error_type: str) -> list[RecoveryStrategy]:
        """Get ordered list of recovery strategies for error type.

        Parameters
        ----------
        error_type : str
            Error type from analyze_error()

        """
        if error_type == "singular_matrix":
            return [
                RecoveryStrategy(
                    name="relax_flagging",
                    description="Reduce flagging aggressiveness (more data)",
                    apply_fn=self._try_relaxed_flagging,
                ),
                RecoveryStrategy(
                    name="change_refant",
                    description="Switch to healthier reference antenna",
                    apply_fn=self._try_alternate_refant,
                ),
                RecoveryStrategy(
                    name="increase_solint",
                    description="Increase solution interval (average more data)",
                    apply_fn=self._try_longer_solint,
                ),
            ]

        elif error_type == "low_snr":
            return [
                RecoveryStrategy(
                    name="increase_solint",
                    description="Increase solution interval (average more data)",
                    apply_fn=self._try_longer_solint,
                ),
                RecoveryStrategy(
                    name="use_all_fields",
                    description="Use all 24 fields (maximum SNR)",
                    apply_fn=self._try_all_fields,
                ),
                RecoveryStrategy(
                    name="lower_minsnr",
                    description="Reduce SNR threshold",
                    apply_fn=self._try_lower_minsnr,
                ),
            ]

        elif error_type == "phase_instability":
            return [
                RecoveryStrategy(
                    name="shorter_solint",
                    description="Track faster phase variations",
                    apply_fn=self._try_shorter_solint,
                ),
                RecoveryStrategy(
                    name="change_refant",
                    description="Switch to more stable reference antenna",
                    apply_fn=self._try_alternate_refant,
                ),
            ]

        elif error_type == "excessive_flagging":
            return [
                RecoveryStrategy(
                    name="relax_flagging",
                    description="Less aggressive flagging",
                    apply_fn=self._try_relaxed_flagging,
                ),
                RecoveryStrategy(
                    name="gpu_rfi_detection",
                    description="Use GPU-accelerated RFI detection (more accurate)",
                    apply_fn=self._try_gpu_rfi,
                ),
            ]

        elif error_type == "bad_refant":
            return [
                RecoveryStrategy(
                    name="auto_select_refant",
                    description="Automatically select best refant from health analysis",
                    apply_fn=self._try_auto_refant,
                ),
            ]

        elif error_type == "convergence_failure":
            return [
                RecoveryStrategy(
                    name="increase_solint",
                    description="More stable solutions with longer averaging",
                    apply_fn=self._try_longer_solint,
                ),
                RecoveryStrategy(
                    name="relax_flagging",
                    description="More data for better convergence",
                    apply_fn=self._try_relaxed_flagging,
                ),
            ]

        else:
            # Unknown error - try general strategies
            return [
                RecoveryStrategy(
                    name="relax_snr_thresholds",
                    description="Use relaxed SNR thresholds",
                    apply_fn=self._try_relaxed_snr,
                ),
                RecoveryStrategy(
                    name="change_refant",
                    description="Try different reference antenna",
                    apply_fn=self._try_alternate_refant,
                ),
                RecoveryStrategy(
                    name="relax_flagging",
                    description="Reduce flagging aggressiveness",
                    apply_fn=self._try_relaxed_flagging,
                ),
            ]

    def attempt_recovery(
        self,
        calibrate_fn: Callable[[str], Any],
        error: Exception,
        max_recovery_attempts: int = 3,
    ) -> tuple[bool, dict[str, Any] | None]:
        """Attempt intelligent recovery from calibration failure.

        Parameters
        ----------
        calibrate_fn : Callable[[str, Any], Any]
            Calibration function to retry
        Signature: (ms_path, **params) -> result
        error : Exception
            Original calibration error
        max_recovery_attempts : int, optional
            Maximum recovery strategies to try
        Default: 3

        """
        error_type = self.analyze_error(error)
        logger.info(f"Calibration failed with error type: {error_type}")
        logger.info(f"Original error: {error}")

        strategies = self.get_recovery_strategies(error_type)
        logger.info(
            f"Attempting {min(len(strategies), max_recovery_attempts)} recovery strategies: "
            f"{[s.name for s in strategies[:max_recovery_attempts]]}"
        )

        for i, strategy in enumerate(strategies[:max_recovery_attempts], 1):
            logger.info(
                f"Recovery attempt {i}/{min(len(strategies), max_recovery_attempts)}: "
                f"{strategy.name}"
            )
            logger.info(f"  Strategy: {strategy.description}")

            try:
                # Apply recovery strategy (modifies parameters)
                modified_params = strategy.apply_fn()

                # Log parameter changes
                self._log_parameter_changes(self.original_params, modified_params)

                # Try calibration with modified parameters
                logger.info("  Retrying calibration with modified params...")
                calibrate_fn(self.ms_path, **modified_params)

                # Success!
                logger.info(f" Calibration succeeded with strategy: {strategy.name}")
                self.attempt_history.append(
                    {
                        "strategy": strategy.name,
                        "success": True,
                        "params": modified_params,
                        "error_type": error_type,
                    }
                )

                return True, modified_params

            except Exception as e:
                logger.warning(f"  Strategy {strategy.name} failed: {e}")
                self.attempt_history.append(
                    {
                        "strategy": strategy.name,
                        "success": False,
                        "error": str(e),
                        "error_type": error_type,
                    }
                )
                continue

        # All strategies exhausted
        logger.error(
            f"All {min(len(strategies), max_recovery_attempts)} recovery strategies failed"
        )
        return False, None

    # =========================================================================
    # Recovery Strategy Implementations
    # =========================================================================

    def _try_relaxed_flagging(self) -> dict[str, Any]:
        """Reduce flagging aggressiveness to preserve more data."""
        params = self.original_params.copy()

        # Reduce flagging threshold by 30%
        current_threshold = params.get("flag_threshold", 5.0)
        params["flag_threshold"] = current_threshold * 1.3

        logger.info(
            f"  Relaxing flag threshold: {current_threshold:.1f} → {params['flag_threshold']:.1f}"
        )

        return params

    def _try_alternate_refant(self) -> dict[str, Any]:
        """Switch to healthier reference antenna based on antenna health analysis."""
        params = self.original_params.copy()

        try:
            # Import here to avoid circular dependency
            from dsa110_contimg.core.calibration.refant_selection import (
                recommend_refants_from_ms,
            )

            refant_str = recommend_refants_from_ms(self.ms_path)
            params["refant"] = refant_str
            logger.info(f"  Switching to recommended refant: {refant_str}")

        except Exception as e:
            logger.warning(f"  Could not get refant recommendations: {e}")
            # Fallback: use default outrigger chain
            params["refant"] = "103,101,100,102,104,105,106,107,108,109,110"
            logger.info(f"  Using default outrigger chain: {params['refant']}")

        return params

    def _try_longer_solint(self) -> dict[str, Any]:
        """Increase solution interval for better SNR."""
        params = self.original_params.copy()

        current_solint = params.get("gain_solint", "60s")

        # Parse and double the solution interval
        match = re.match(r"(\d+)([smh]?)", current_solint)
        if match:
            value = int(match.group(1))
            unit = match.group(2) or "s"
            new_value = value * 2
            params["gain_solint"] = f"{new_value}{unit}"
            logger.info(f"  Increasing solint: {current_solint} → {params['gain_solint']}")
        else:
            params["gain_solint"] = "120s"
            logger.info(f"  Setting solint: {params['gain_solint']} (parse failed)")

        return params

    def _try_shorter_solint(self) -> dict[str, Any]:
        """Decrease solution interval to track faster phase variations."""
        params = self.original_params.copy()

        current_solint = params.get("gain_solint", "60s")

        # Parse and halve the solution interval
        match = re.match(r"(\d+)([smh]?)", current_solint)
        if match:
            value = int(match.group(1))
            unit = match.group(2) or "s"
            new_value = max(10, value // 2)  # Don't go below 10s
            params["gain_solint"] = f"{new_value}{unit}"
            logger.info(f"  Decreasing solint: {current_solint} → {params['gain_solint']}")
        else:
            params["gain_solint"] = "30s"
            logger.info(f"  Setting solint: {params['gain_solint']} (parse failed)")

        return params

    def _try_all_fields(self) -> dict[str, Any]:
        """Use all 24 fields for maximum SNR."""
        params = self.original_params.copy()
        params["field"] = "0~23"
        logger.info("  Using all fields: 0~23 (maximum SNR)")
        return params

    def _try_lower_minsnr(self) -> dict[str, Any]:
        """Reduce SNR thresholds to accept noisier solutions."""
        params = self.original_params.copy()

        # Reduce minsnr thresholds by 40%
        for key in ["k_minsnr", "bp_minsnr", "gain_minsnr"]:
            if key in params:
                current = params[key]
                params[key] = current * 0.6
                logger.info(f"  Reducing {key}: {current:.1f} → {params[key]:.1f}")

        return params

    def _try_gpu_rfi(self) -> dict[str, Any]:
        """Enable GPU-accelerated RFI detection."""
        params = self.original_params.copy()
        params["use_gpu_rfi"] = True
        logger.info("  Enabling GPU-accelerated RFI detection")
        return params

    def _try_auto_refant(self) -> dict[str, Any]:
        """Auto-select best refant from antenna health analysis."""
        return self._try_alternate_refant()  # Same implementation

    def _try_relaxed_snr(self) -> dict[str, Any]:
        """Use relaxed SNR thresholds for difficult calibration cases."""
        params = self.original_params.copy()

        # Relax SNR thresholds
        params["bp_minsnr"] = 2.0
        params["gain_minsnr"] = 2.0
        params["prebp_minsnr"] = 2.0
        logger.info("  Using relaxed SNR thresholds (bp=2.0, gain=2.0)")

        return params

    # =========================================================================
    # Utilities
    # =========================================================================

    def _log_parameter_changes(self, original: dict[str, Any], modified: dict[str, Any]) -> None:
        """Log changes between original and modified parameters.

        Parameters
        ----------
        original : Dict[str, Any]
            Original parameters.
        modified : Dict[str, Any]
            Modified parameters.

        Returns
        -------
            None
        """
        changes = []

        for key in set(original.keys()) | set(modified.keys()):
            orig_val = original.get(key)
            mod_val = modified.get(key)

            if orig_val != mod_val:
                changes.append(f"    {key}: {orig_val} → {mod_val}")

        if changes:
            logger.info("  Parameter changes:")
            for change in changes:
                logger.info(change)

    def get_attempt_summary(self) -> dict[str, Any]:
        """Get summary of recovery attempts."""
        successful = [a for a in self.attempt_history if a.get("success", False)]
        failed = [a for a in self.attempt_history if not a.get("success", False)]

        return {
            "total_attempts": len(self.attempt_history),
            "successful_strategy": successful[0]["strategy"] if successful else None,
            "failed_strategies": [a["strategy"] for a in failed],
            "history": self.attempt_history,
        }
