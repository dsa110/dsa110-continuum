"""
Pipeline guardrails for calibration quality.

Enforces best practices and provides staged escalation for quality issues.
This module provides guardrails that can be used at the pipeline level to:

1. Enforce mandatory pre-bandpass phase calibration
2. Warn against using chgcentre (known UVW issues with DSA-110)
3. Provide staged quality escalation based on flagging fraction
4. Track and report calibration quality metrics

Usage:
    from dsa110_contimg.core.calibration.guardrails import (
        CalibrationGuardrails,
        QualityAction,
        get_quality_action,
    )

    # Enforce pre-bandpass phase requirement
    CalibrationGuardrails.require_prebandpass_phase(prebp_table)

    # Check quality and determine action
    action, message = get_quality_action(flag_fraction=0.15)
    if action == QualityAction.PAUSE:
        # Run diagnostics before continuing
        pass
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class QualityAction(Enum):
    """Actions to take based on calibration quality level."""

    PASS = "pass"  # Continue normally, quality is acceptable
    WARN = "warn"  # Log warning but continue
    PAUSE = "pause"  # Run diagnostics, attempt recovery
    FAIL = "fail"  # Stop pipeline, require investigation


@dataclass
class QualityThresholds:
    """Configurable quality thresholds for bandpass calibration.

    These thresholds define the boundaries between different quality
    levels based on the fraction of flagged solutions.

    Attributes
    ----------
    pristine : float
        Flagging below this is excellent quality (<3% default).
    good : float
        Flagging below this is acceptable (<5% default).
    moderate : float
        Flagging below this is concerning but can continue (<20% default).
    critical : float
        Flagging below this requires intervention (<50% default).
        Above this threshold, calibration is considered failed.
    """

    pristine: float = 0.03  # <3% = excellent
    good: float = 0.05  # <5% = acceptable
    moderate: float = 0.20  # <20% = concerning but can continue
    critical: float = 0.50  # <50% = requires intervention


# Default thresholds instance
DEFAULT_THRESHOLDS = QualityThresholds()


def get_quality_action(
    flag_fraction: float,
    thresholds: QualityThresholds | None = None,
) -> tuple[QualityAction, str]:
    """Determine action based on flagging fraction.

    This function maps a flagging fraction to an appropriate action
    and provides a human-readable message describing the quality level.

    Parameters
    ----------
    flag_fraction : float
        Fraction of flagged solutions (0.0 to 1.0).
    thresholds : QualityThresholds or None, optional
        Custom thresholds to use. Default uses DEFAULT_THRESHOLDS.

    Returns
    -------
    tuple[QualityAction, str]
        (action, message) tuple where action indicates what to do
        and message describes the quality level.

    Examples
    --------
    >>> action, msg = get_quality_action(0.02)
    >>> print(action, msg)
    QualityAction.PASS Pristine quality (2.0% flagged)

    >>> action, msg = get_quality_action(0.35)
    >>> print(action)
    QualityAction.PAUSE
    """
    if thresholds is None:
        thresholds = DEFAULT_THRESHOLDS

    pct = flag_fraction * 100

    if flag_fraction < thresholds.pristine:
        return QualityAction.PASS, f"Pristine quality ({pct:.1f}% flagged)"
    elif flag_fraction < thresholds.good:
        return QualityAction.PASS, f"Good quality ({pct:.1f}% flagged)"
    elif flag_fraction < thresholds.moderate:
        return QualityAction.WARN, f"Moderate flagging ({pct:.1f}%) - review recommended"
    elif flag_fraction < thresholds.critical:
        return QualityAction.PAUSE, f"High flagging ({pct:.1f}%) - running diagnostics"
    else:
        return QualityAction.FAIL, f"Critical flagging ({pct:.1f}%) - investigation required"


def get_quality_tier(flag_fraction: float) -> str:
    """Get a simple quality tier string for reporting.

    Parameters
    ----------
    flag_fraction : float
        Fraction of flagged solutions (0.0 to 1.0).

    Returns
    -------
    str
        Quality tier: "pristine", "good", "moderate", "high", or "critical".
    """
    if flag_fraction < 0.03:
        return "pristine"
    elif flag_fraction < 0.05:
        return "good"
    elif flag_fraction < 0.20:
        return "moderate"
    elif flag_fraction < 0.50:
        return "high"
    else:
        return "critical"


class CalibrationGuardrails:
    """Guardrails for calibration pipeline.

    This class provides static methods that enforce best practices
    in the calibration pipeline. These can be called at key points
    to ensure data quality and proper setup.

    Examples
    --------
    >>> # Before bandpass calibration
    >>> CalibrationGuardrails.require_prebandpass_phase("/path/to/prebp.G")

    >>> # Before phase shifting
    >>> CalibrationGuardrails.warn_chgcentre_usage(use_chgcentre=True)

    >>> # After bandpass calibration
    >>> CalibrationGuardrails.check_bandpass_quality(flag_fraction=0.08)
    """

    @staticmethod
    def require_prebandpass_phase(
        prebandpass_table: str | None,
        allow_override: bool = False,
    ) -> None:
        """Enforce that pre-bandpass phase calibration is provided.

        Pre-bandpass phase calibration is CRITICAL for bandpass calibration.
        Without it, phases will decorrelate causing 90%+ flagged solutions.
        This is the most common cause of bandpass calibration failure.

        Parameters
        ----------
        prebandpass_table : str or None
            Path to pre-bandpass phase calibration table.
        allow_override : bool, optional
            If True, only warn instead of raising error. Default False.

        Raises
        ------
        ValueError
            If prebandpass_table is None and allow_override is False.

        Warnings
        --------
        UserWarning
            If prebandpass_table is None and allow_override is True.
        """
        if prebandpass_table is None:
            msg = (
                "Pre-bandpass phase calibration is REQUIRED for bandpass calibration.\n"
                "Without it, phases will decorrelate causing 90%+ flagged solutions.\n\n"
                "Resolution:\n"
                "  1. Run solve_prebandpass_phase() before solve_bandpass()\n"
                "  2. Pass the resulting table as prebandpass_phase_table argument\n\n"
                "Example:\n"
                "  prebp_tables = solve_prebandpass_phase(ms, cal_field, refant, ...)\n"
                "  bp_tables = solve_bandpass(ms, cal_field, refant, ktable=None,\n"
                "                             prebandpass_phase_table=prebp_tables[0])"
            )
            if allow_override:
                warnings.warn(msg, UserWarning, stacklevel=2)
                logger.warning(msg)
            else:
                raise ValueError(msg)

    @staticmethod
    def warn_chgcentre_usage(use_chgcentre: bool) -> None:
        """Deprecated: This warning is no longer issued.

        The previous claim about chgcentre producing 2x UVW errors was unverified.
        This method is kept for backward compatibility but does nothing.

        Parameters
        ----------
        use_chgcentre : bool
            Whether chgcentre is being used (ignored).
        """
        # Deprecated - unverified claims removed
        pass

    @staticmethod
    def check_bandpass_quality(
        flag_fraction: float,
        thresholds: QualityThresholds | None = None,
        raise_on_fail: bool = True,
        warn_on_pause: bool = True,
    ) -> QualityAction:
        """Check bandpass calibration quality and take appropriate action.

        This method should be called after bandpass calibration completes
        to determine the quality level and decide on next steps.

        Parameters
        ----------
        flag_fraction : float
            Fraction of flagged solutions (0.0 to 1.0).
        thresholds : QualityThresholds or None, optional
            Custom thresholds. Default uses DEFAULT_THRESHOLDS.
        raise_on_fail : bool, optional
            If True, raise ValueError when quality is FAIL. Default True.
        warn_on_pause : bool, optional
            If True, log warning when quality is PAUSE. Default True.

        Returns
        -------
        QualityAction
            The recommended action based on quality.

        Raises
        ------
        ValueError
            If quality is FAIL and raise_on_fail is True.
        """
        action, message = get_quality_action(flag_fraction, thresholds)

        if action == QualityAction.PASS:
            logger.info(f"✅ Bandpass quality: {message}")
        elif action == QualityAction.WARN:
            logger.warning(f"⚠️  Bandpass quality: {message}")
        elif action == QualityAction.PAUSE:
            if warn_on_pause:
                logger.warning(f"⏸️  Bandpass quality: {message}")
                logger.warning("Running diagnostics to identify root cause...")
        elif action == QualityAction.FAIL:
            logger.error(f"❌ Bandpass quality: {message}")
            if raise_on_fail:
                raise ValueError(
                    f"Bandpass calibration failed: {message}\n\n"
                    f"This indicates a fundamental problem with the data or setup.\n"
                    f"Common causes:\n"
                    f"  - Data not coherently phased to calibrator\n"
                    f"  - Pre-bandpass phase calibration not applied\n"
                    f"  - MODEL_DATA not populated or incorrect\n"
                    f"  - UVW values corrupted (chgcentre bug)\n"
                    f"  - Severe RFI contamination\n\n"
                    f"Run bandpass diagnostics for detailed analysis."
                )

        return action

    @staticmethod
    def validate_refant(
        refant: str | int | None,
        available_antennas: list[int] | None = None,
    ) -> None:
        """Validate that reference antenna is specified and available.

        Parameters
        ----------
        refant : str, int, or None
            Reference antenna specification.
        available_antennas : list[int] or None, optional
            List of available antenna IDs. If provided, validates refant is in list.

        Raises
        ------
        ValueError
            If refant is None or not in available_antennas.
        """
        if refant is None:
            raise ValueError(
                "Reference antenna (refant) must be specified for calibration.\n"
                "Recommended: Use an outrigger antenna with stable phases.\n"
                "For DSA-110, antenna 103 or 104 are typical choices."
            )

        if available_antennas is not None:
            # Parse refant to get primary antenna ID
            if isinstance(refant, str):
                # Handle comma-separated list (priority order)
                primary = refant.split(",")[0].strip()
                try:
                    refant_id = int(primary)
                except ValueError:
                    # May be a name, can't validate against IDs
                    return
            else:
                refant_id = int(refant)

            if refant_id not in available_antennas:
                raise ValueError(
                    f"Reference antenna {refant_id} not in available antennas.\n"
                    f"Available: {sorted(available_antennas)[:20]}..."
                )

    @staticmethod
    def validate_field_selection(
        cal_field: str,
        n_fields: int,
    ) -> list[int]:
        """Validate and parse field selection string.

        Parameters
        ----------
        cal_field : str
            Field selection string (e.g., "0~23", "5", "0,1,2,3").
        n_fields : int
            Total number of fields in MS.

        Returns
        -------
        list[int]
            List of field indices.

        Raises
        ------
        ValueError
            If field selection is invalid or references non-existent fields.
        """
        field_indices = []

        if "~" in cal_field:
            # Range notation: "0~23"
            parts = cal_field.split("~")
            if len(parts) != 2:
                raise ValueError(f"Invalid field range: {cal_field}")
            try:
                start = int(parts[0])
                end = int(parts[1])
                field_indices = list(range(start, end + 1))
            except ValueError:
                raise ValueError(f"Invalid field range: {cal_field}")
        elif "," in cal_field:
            # Comma-separated: "0,1,2,3"
            try:
                field_indices = [int(f.strip()) for f in cal_field.split(",")]
            except ValueError:
                raise ValueError(f"Invalid field list: {cal_field}")
        elif cal_field.isdigit():
            # Single field
            field_indices = [int(cal_field)]
        else:
            # Assume all fields or special selection
            field_indices = list(range(n_fields))

        # Validate all indices are in range
        for idx in field_indices:
            if idx < 0 or idx >= n_fields:
                raise ValueError(
                    f"Field index {idx} out of range. MS has {n_fields} fields (0-{n_fields - 1})."
                )

        return field_indices


@dataclass
class QualityMetrics:
    """Container for calibration quality metrics.

    Attributes
    ----------
    flag_fraction : float
        Overall fraction of flagged solutions.
    n_antennas_good : int
        Number of antennas with good solutions.
    n_antennas_flagged : int
        Number of fully-flagged antennas.
    median_snr : float
        Median SNR of unflagged solutions.
    quality_tier : str
        Quality tier string.
    action : QualityAction
        Recommended action.
    """

    flag_fraction: float
    n_antennas_good: int
    n_antennas_flagged: int
    median_snr: float
    quality_tier: str
    action: QualityAction

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "flag_fraction": self.flag_fraction,
            "n_antennas_good": self.n_antennas_good,
            "n_antennas_flagged": self.n_antennas_flagged,
            "median_snr": self.median_snr,
            "quality_tier": self.quality_tier,
            "action": self.action.value,
        }


def extract_quality_metrics(
    caltable_path: str,
    thresholds: QualityThresholds | None = None,
) -> QualityMetrics:
    """Extract quality metrics from a calibration table.

    Parameters
    ----------
    caltable_path : str
        Path to calibration table.
    thresholds : QualityThresholds or None, optional
        Custom thresholds for quality assessment.

    Returns
    -------
    QualityMetrics
        Extracted quality metrics.
    """
    from dsa110_contimg.common.utils.casa_init import ensure_casa_path

    ensure_casa_path()

    import casacore.tables as ct
    import numpy as np

    with ct.table(caltable_path, readonly=True, ack=False) as tb:
        flags = tb.getcol("FLAG")
        antenna_ids = tb.getcol("ANTENNA1")

        # Get SNR if available
        if "SNR" in tb.colnames():
            snr = tb.getcol("SNR")
            unflagged_snr = snr[~flags]
            median_snr = float(np.median(unflagged_snr)) if len(unflagged_snr) > 0 else 0.0
        else:
            median_snr = 0.0

    # Overall flag fraction
    flag_fraction = float(np.mean(flags))

    # Per-antenna analysis
    unique_ants = np.unique(antenna_ids)
    n_good = 0
    n_flagged = 0

    for ant in unique_ants:
        ant_mask = antenna_ids == ant
        ant_flags = flags[:, :, ant_mask] if flags.ndim == 3 else flags[ant_mask]
        if np.mean(ant_flags) > 0.99:
            n_flagged += 1
        else:
            n_good += 1

    # Get quality tier and action
    quality_tier = get_quality_tier(flag_fraction)
    action, _ = get_quality_action(flag_fraction, thresholds)

    return QualityMetrics(
        flag_fraction=flag_fraction,
        n_antennas_good=n_good,
        n_antennas_flagged=n_flagged,
        median_snr=median_snr,
        quality_tier=quality_tier,
        action=action,
    )
