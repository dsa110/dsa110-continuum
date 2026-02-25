"""Enhanced ESE detection with observability, resilience, and caching.

This module demonstrates integration of all cost-free improvements
into the ESE detection component.
"""

from __future__ import annotations

import time
from pathlib import Path

# Import original function
from dsa110_contimg.core.photometry.ese_detection import detect_ese_candidates as _detect_ese_candidates
from dsa110_contimg.workflow.pipeline.caching import (
    get_cached_variability_stats,
)
from dsa110_contimg.workflow.pipeline.metrics import record_ese_detection
from dsa110_contimg.workflow.pipeline.structured_logging import (
    get_logger,
    log_ese_detection,
    set_correlation_id,
)

logger = get_logger(__name__)


def detect_ese_candidates_enhanced(
    products_db: Path,
    min_sigma: float = 5.0,
    source_id: str | None = None,
    recompute: bool = False,
    correlation_id: str | None = None,
) -> list[dict]:
    """Enhanced ESE detection with observability and resilience.

        Features:
        - Metrics recording
        - Structured logging
        - Caching

        Note: Dagster retry policies handle failures; Dagster assets publish events via Output metadata

        Note: Dagster retry policies handle failures; Dagster assets publish events via Output metadata

    Parameters
    ----------
    products_db : str
        Path to products database
    min_sigma : float
        Minimum sigma threshold
    source_id : str or None
        Optional specific source ID
    recompute : bool
        Recompute variability stats
    correlation_id : str
        Correlation ID for tracing

    Returns
    -------
        list of dict
        List of ESE candidate dictionaries
    """
    # Set correlation ID for tracing
    if correlation_id:
        set_correlation_id(correlation_id)

    start_time = time.time()

    try:
        # Call ESE detection directly - Dagster retry policies handle failures
        candidates = _detect_ese_candidates(
            products_db=products_db,
            min_sigma=min_sigma,
            source_id=source_id,
            recompute=recompute,
        )

        duration = time.time() - start_time

        # Record metrics
        record_ese_detection(
            duration=duration,
            candidates=len(candidates),
            source=source_id or "all",
            min_sigma=min_sigma,
        )

        # Structured logging
        log_ese_detection(
            logger=logger,
            source_id=source_id,
            candidates_found=len(candidates),
            duration_seconds=duration,
            min_sigma=min_sigma,
        )

        # Note: Event publishing should be handled by Dagster assets via metadata
        # Dagster assets calling this function should publish metadata like:
        #   Output(candidates, metadata={"num_candidates": len(candidates), ...})

        return candidates

    except Exception as e:
        duration = time.time() - start_time

        # Record failure metrics
        record_ese_detection(
            duration=duration,
            candidates=0,
            source=source_id or "all",
            min_sigma=min_sigma,
        )

        # Log error
        logger.error(
            "ese_detection_failed",
            component="ese_detection",
            error_type=type(e).__name__,
            error_message=str(e),
            source_id=source_id,
            min_sigma=min_sigma,
        )

        # Note: Dagster run storage automatically tracks failures with full context.
        # Failed runs can be queried via: instance.get_runs(filters=RunsFilter(statuses=[DagsterRunStatus.FAILURE]))

        raise


def detect_ese_with_caching(
    products_db: Path, source_id: str, min_sigma: float = 5.0
) -> dict | None:
    """Detect ESE for a source with caching.

        Uses cached variability stats if available and recent.

    Parameters
    ----------
    products_db : str
        Path to products database
    source_id : str
        Source ID to check
    min_sigma : float
        Minimum sigma threshold

    Returns
    -------
        dict or None
        ESE candidate dict if detected, None otherwise
    """
    # Try to get cached variability stats
    cached_stats = get_cached_variability_stats(source_id)

    if cached_stats:
        # Use cached stats if sigma deviation meets threshold
        sigma_dev = cached_stats.get("sigma_deviation", 0.0)
        if sigma_dev >= min_sigma:
            return {
                "source_id": source_id,
                "significance": sigma_dev,
                "sigma_deviation": sigma_dev,
                "cached": True,
            }

    # Fall back to full detection
    candidates = detect_ese_candidates_enhanced(
        products_db=products_db,
        min_sigma=min_sigma,
        source_id=source_id,
        recompute=False,
    )

    if candidates:
        candidate = candidates[0]
        # Cache the variability stats for future use
        # (This would need to fetch from database)
        return candidate

    return None
