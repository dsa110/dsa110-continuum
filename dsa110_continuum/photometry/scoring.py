"""Multi-metric scoring system for ESE detection.

This module provides composite scoring that combines multiple variability
metrics into a single confidence score for ESE candidate detection.
"""

from __future__ import annotations

# Default weights for composite scoring
DEFAULT_WEIGHTS = {
    "sigma_deviation": 0.5,
    "chi2_nu": 0.3,
    "eta_metric": 0.2,
}

# Confidence level thresholds
CONFIDENCE_THRESHOLDS = {
    "high": 7.0,
    "medium": 4.0,
    "low": 0.0,
}


def normalize_metric(
    value: float,
    min_val: float = 0.0,
    max_val: float = 10.0,
) -> float:
    """Normalize a metric value to [0, 1] range.

    Parameters
    ----------
    value : float
        Metric value to normalize
    min_val : float
        Minimum expected value (clips below this)
    max_val : float
        Maximum expected value (clips above this)

    Returns
    -------
        float
        Normalized value in [0, 1] range
    """
    if max_val == min_val:
        return 0.0

    # Clip value to range
    clipped_value = max(min_val, min(value, max_val))

    # Normalize to [0, 1]
    normalized = (clipped_value - min_val) / (max_val - min_val)

    return float(max(0.0, min(1.0, normalized)))


def calculate_composite_score(
    metrics: dict[str, float],
    weights: dict[str, float] | None = None,
    normalize: bool = True,
) -> float:
    """Calculate composite score from multiple variability metrics.

        The composite score is a weighted sum of normalized metrics, providing
        a single confidence value for ESE detection.

    Parameters
    ----------
    metrics : dict
        Dictionary of metric names to values
    weights : dict, optional
        Dictionary of metric names to weights (defaults to DEFAULT_WEIGHTS)
    normalize : bool, optional
        Whether to normalize metrics before combining (default: True)

    Returns
    -------
        float
        Composite score

        Example
    -------
        >>> metrics = {'sigma_deviation': 5.0, 'chi2_nu': 3.0, 'eta_metric': 2.0}
        >>> score = calculate_composite_score(metrics)
        >>> print(f"Composite score: {score}")
    """
    if weights is None:
        weights = DEFAULT_WEIGHTS

    if not metrics:
        return 0.0

    # Normalize metrics if requested
    normalized_metrics = {}
    for metric_name, value in metrics.items():
        if metric_name in weights:
            if normalize:
                # Normalize based on typical ranges
                # sigma_deviation: [0, 10], chi2_nu: [0, 10], eta_metric: [0, 5]
                max_val = 10.0 if metric_name != "eta_metric" else 5.0
                normalized_metrics[metric_name] = normalize_metric(value, 0.0, max_val)
            else:
                normalized_metrics[metric_name] = value

    # Calculate weighted sum
    composite_score = 0.0
    total_weight = 0.0

    for metric_name, weight in weights.items():
        if metric_name in normalized_metrics:
            composite_score += normalized_metrics[metric_name] * weight
            total_weight += weight

    # Normalize by total weight if weights don't sum to 1.0
    if total_weight > 0:
        composite_score = composite_score / total_weight * 10.0  # Scale to [0, 10]

    return float(composite_score)


def get_confidence_level(score: float) -> str:
    """Get confidence level from composite score.

    Parameters
    ----------
    score : float
        Composite score value

    Returns
    -------
        str
        Confidence level: 'high', 'medium', or 'low'
    """
    if score >= CONFIDENCE_THRESHOLDS["high"]:
        return "high"
    elif score >= CONFIDENCE_THRESHOLDS["medium"]:
        return "medium"
    else:
        return "low"
