"""
Variability Metrics Module for DSA-110

This module provides statistical functions to quantify the variability of radio sources
over time, supporting the Long-term Flux Monitoring science goal.
"""

import numpy as np
from astropy.stats import mad_std as astropy_mad_std

def calculate_weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    """
    Calculate the weighted mean of an array of values.

    Args:
        values: Array of measurements.
        weights: Array of weights (typically 1/sigma^2).

    Returns:
        Weighted mean value. Returns NaN if weights sum to 0.
    """
    if np.sum(weights) == 0:
        return np.nan
    return np.average(values, weights=weights)

def calculate_weighted_variance(values: np.ndarray, weights: np.ndarray) -> float:
    """
    Calculate the weighted variance of an array of values.

    Args:
        values: Array of measurements.
        weights: Array of weights.

    Returns:
        Weighted variance.
    """
    if np.sum(weights) == 0:
        return np.nan
    
    mean = calculate_weighted_mean(values, weights)
    # Variance = Sum(w * (x - mean)^2) / Sum(w)
    variance = np.average((values - mean)**2, weights=weights)
    return variance

def calculate_chi_squared(values: np.ndarray, weights: np.ndarray, model_value: float = None) -> float:
    """
    Calculate the generalized Chi-Squared statistic for goodness of fit.
    
    If model_value is None, it defaults to the weighted mean (testing against constant flux hypothesis).

    Args:
        values: Array of measurements.
        weights: Array of weights (typically 1/err^2).
        model_value: The expected model value (float).

    Returns:
        Reduced Chi-Squared statistic (Chi^2 / N). 
        Note: This is technically 1/N * Sum( (x - model)^2 * weight )
    """
    if len(values) == 0:
        return np.nan

    if model_value is None:
        model_value = calculate_weighted_mean(values, weights)
        
    chi_sq = np.sum(weights * (values - model_value)**2)
    return chi_sq / len(values) # Reduced Chi-Sq roughly

def calculate_mad_std(values: np.ndarray, ignore_nan: bool = True) -> float:
    """
    Calculate the Median Absolute Deviation (MAD) standard deviation.
    A robust estimator for noise.

    Args:
        values: Array of measurements.
        ignore_nan: Whether to ignore NaN values.

    Returns:
        Robust standard deviation (sigma equivalent).
    """
    return astropy_mad_std(values, ignore_nan=ignore_nan)

def calculate_eta_metric(values: np.ndarray, weights: np.ndarray) -> float:
    """
    Calculate the Eta variability metric (weighted variance index).
    
    Formula:
        η = (N / (N-1)) * ( mean(w * f^2) - (mean(w * f)^2 / mean(w)) )
        where w = weights, f = flux values, and mean() is the arithmetic mean.

    Args:
        values: Array of flux measurements.
        weights: Array of weights (typically 1/sigma^2).

    Returns:
        Eta metric value.
    """
    # Filter for finite values and positive weights
    mask = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    v = values[mask]
    w = weights[mask]
    
    n = len(v)
    if n < 2:
        return 0.0

    # Arithmetic means of weighted quantities
    mean_w_f2 = np.mean(w * v**2)
    mean_w_f = np.mean(w * v)
    mean_w = np.mean(w)

    eta = (n / (n - 1)) * (mean_w_f2 - (mean_w_f**2 / mean_w))
    return float(eta)

def calculate_v_metric(values: np.ndarray) -> float:
    """
    Calculate the V metric (coefficient of variation).
    
    Formula:
        V = std(f) / mean(f)

    Args:
        values: Array of flux measurements.

    Returns:
        V metric value.
    """
    valid = values[np.isfinite(values)]
    if len(valid) < 2:
        return 0.0
    
    mean = np.mean(valid)
    if mean == 0:
        return 0.0
        
    std = np.std(valid, ddof=1) # Using sample std dev to match typical statistics
    return float(std / mean)

def calculate_sigma_deviation(values: np.ndarray, mean: float = None, std: float = None) -> float:
    """
    Calculate sigma deviation (max deviation from mean in units of std).

    Args:
        values: Array of flux measurements.
        mean: Optional pre-computed mean.
        std: Optional pre-computed std.

    Returns:
        Sigma deviation score.
    """
    valid = values[np.isfinite(values)]
    if len(valid) == 0:
        return 0.0

    # If stats are not provided, we need at least 2 points to compute them
    if (mean is None or std is None) and len(valid) < 2:
        return 0.0

    if mean is None:
        mean = float(np.mean(valid))
    if std is None:
        std = float(np.std(valid, ddof=1))

    if std == 0:
        return 0.0

    max_val = float(np.max(valid))
    min_val = float(np.min(valid))

    return float(max(abs(max_val - mean), abs(min_val - mean)) / std)
