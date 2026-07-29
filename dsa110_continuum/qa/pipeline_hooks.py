"""Pipeline hooks for post-stage processing.

This module previously contained an unused calibration-metrics ingestion
path (`CalibrationMetricsRecord`, `extract_calibration_metrics`,
`hook_calibration_complete`, etc.) that had drifted out of sync with the
real `CalibrationMetrics` dataclass in `dsa110_continuum.calibration.qa`.

The calibration dashboard now uses `validate_caltable_quality` and the
backends exposed by `dsa110_continuum.qa.calibration_quality`; the dead
functions have been removed.  Non-calibration hooks (e.g. ESE detection)
remain here as thin call-throughs when needed.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def hook_ese_detection_complete() -> None:
    """Trigger the ESE-detection post-processing hook.

    Currently a no-op; downstream dashboard integration is tracked
    separately.
    """
    logger.debug("ESE detection complete hook triggered")
