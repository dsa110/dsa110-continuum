"""Calibration hardening utilities.

This module provides hardening functionality for calibration, including:
- Calibration validity windows and bidirectional selection
- RFI flagging and mitigation

Moved from workflow.pipeline.hardening to proper location in core.calibration.
"""

from dsa110_contimg.core.calibration.hardening.calibration import (
    BP_VALIDITY_DAYS,
    BP_VALIDITY_HOURS,
    DEFAULT_CAL_VALIDITY_DAYS,
    DEFAULT_CAL_VALIDITY_HOURS,
    G_VALIDITY_DAYS,
    G_VALIDITY_HOURS,
    K_VALIDITY_DAYS,
    K_VALIDITY_HOURS,
    CalibrationSelection,
    CalibratorCandidate,
    InterpolatedCalibration,
    TableTypeValidity,
    check_calibration_overlap,
    find_backup_calibrators,
    get_active_applylist_bidirectional,
    get_calibration_for_science,
    get_interpolated_calibration,
    get_min_validity_for_types,
    get_validity_hours_for_type,
    resolve_calibration_overlap,
)
from dsa110_contimg.core.calibration.hardening.rfi import (
    RFIStats,
    preflag_rfi,
    preflag_rfi_adaptive,
)

__all__ = [
    # Validity constants
    "BP_VALIDITY_DAYS",
    "BP_VALIDITY_HOURS",
    "DEFAULT_CAL_VALIDITY_DAYS",
    "DEFAULT_CAL_VALIDITY_HOURS",
    "G_VALIDITY_DAYS",
    "G_VALIDITY_HOURS",
    "K_VALIDITY_DAYS",
    "K_VALIDITY_HOURS",
    # Calibration types
    "CalibrationSelection",
    "CalibratorCandidate",
    "InterpolatedCalibration",
    "TableTypeValidity",
    # Calibration functions
    "check_calibration_overlap",
    "find_backup_calibrators",
    "get_active_applylist_bidirectional",
    "get_calibration_for_science",
    "get_interpolated_calibration",
    "get_min_validity_for_types",
    "get_validity_hours_for_type",
    "resolve_calibration_overlap",
    # RFI functions
    "RFIStats",
    "preflag_rfi",
    "preflag_rfi_adaptive",
]
