"""
Pointing Module - Monitors telescope pointing and calibrator transits.

This module provides:
- Calibrator transit prediction based on LST
- Pointing history tracking
- Active observation monitoring
- Transit window calculations for scheduling

The pointing monitor runs as a Docker service that:
1. Tracks upcoming calibrator transits
2. Updates status files for health monitoring
3. Logs transit events for pipeline coordination
"""

from .monitor import (
    PointingMonitor,
    PointingStatus,
    TransitPrediction,
    calculate_elevation,
    calculate_lst,
    get_active_calibrator,
    get_upcoming_transits,
    predict_calibrator_transit,
)
from .transit_selection import (
    TransitObservation,
    find_transit_observations,
    get_pointing_from_hdf5,
    validate_pointing_matches_calibrator,
)

__all__ = [
    "PointingMonitor",
    "PointingStatus",
    "TransitPrediction",
    "calculate_elevation",
    "calculate_lst",
    "get_active_calibrator",
    "get_upcoming_transits",
    "predict_calibrator_transit",
    "TransitObservation",
    "find_transit_observations",
    "get_pointing_from_hdf5",
    "validate_pointing_matches_calibrator",
]
