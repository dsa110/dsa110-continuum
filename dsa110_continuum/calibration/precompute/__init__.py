"""Calibration precomputation utilities.

This package provides proactive resource preparation when telescope pointing changes:
- Detecting declination changes from incoming HDF5 metadata
- Precomputing bandpass calibrator selection for new declinations
- Triggering background catalog strip database builds
- Caching transit predictions for upcoming calibrators

Moved from workflow.pipeline.precompute to proper location in core.calibration.
"""

from dsa110_contimg.core.calibration.precompute.precompute import (
    CalibratorPrediction,
    PointingChange,
    PointingTracker,
    ensure_catalogs_for_dec,
    get_pointing_tracker,
    precompute_all_transits,
    read_uvh5_metadata_fast,
)

__all__ = [
    "CalibratorPrediction",
    "PointingChange",
    "PointingTracker",
    "ensure_catalogs_for_dec",
    "get_pointing_tracker",
    "precompute_all_transits",
    "read_uvh5_metadata_fast",
]
