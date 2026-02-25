"""
Variability metrics for DSA-110 photometry analysis.

Adopted from VAST Tools for calculating variability metrics on source measurements.
These metrics complement χ²-based variability detection for ESE analysis.

Reference: archive/references/vast-tools/vasttools/utils.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def calculate_eta_metric(
    df: pd.DataFrame,
    flux_col: str = "normalized_flux_jy",
    err_col: str = "normalized_flux_err_jy",
) -> float:
    """Calculate η metric (weighted variance) - adopted from VAST Tools.

        The η metric is a weighted variance metric that accounts for measurement
        uncertainties. It provides a complementary measure to χ² for variability
        detection.

        See VAST Tools: vasttools/utils.py::pipeline_get_eta_metric()

        Formula
    -------
        η = (N / (N-1)) * (
        (w * f²).mean() - ((w * f).mean()² / w.mean())
        )
        where w = 1 / σ² (weights), f = flux values

    Parameters
    ----------
    df : DataFrame
        DataFrame containing flux measurements
    flux_col : str
        Column name for flux values
    err_col : str
        Column name for flux errors

    Returns
    -------
        float
        η metric value

    Raises
    ------
        ValueError
        If insufficient data or missing columns
    """
    if len(df) <= 1:
        return 0.0

    if flux_col not in df.columns:
        raise ValueError(f"Flux column '{flux_col}' not found in DataFrame")

    if err_col not in df.columns:
        raise ValueError(f"Error column '{err_col}' not found in DataFrame")

    # Filter out invalid values
    valid_mask = np.isfinite(df[flux_col]) & np.isfinite(df[err_col]) & (df[err_col] > 0)

    if valid_mask.sum() < 2:
        return 0.0

    df_valid = df[valid_mask]

    # Calculate weights (1 / σ²)
    weights = 1.0 / (df_valid[err_col].values ** 2)
    fluxes = df_valid[flux_col].values

    n = len(df_valid)

    # Calculate η metric
    eta = (n / (n - 1)) * (
        (weights * fluxes**2).mean() - ((weights * fluxes).mean() ** 2 / weights.mean())
    )

    return float(eta)


def calculate_vs_metric(
    flux_a: float, flux_b: float, flux_err_a: float, flux_err_b: float
) -> float:
    """Calculate Vs metric (two-epoch t-statistic) - adopted from VAST Tools.

        The Vs metric is the t-statistic that two flux measurements are variable.
        See Section 5 of Mooley et al. (2016) for details.
    DOI: 10.3847/0004-637X/818/2/105

        See VAST Tools: vasttools/utils.py::calculate_vs_metric()

        Formula
    -------
        Vs = (flux_a - flux_b) / sqrt(σ_a² + σ_b²)

    Parameters
    ----------
    flux_a : float
        Flux value at epoch A
    flux_b : float
        Flux value at epoch B
    flux_err_a : float
        Uncertainty of flux_a
    flux_err_b : float
        Uncertainty of flux_b

    Returns
    -------
        float
        Vs metric value

    Raises
    ------
        ValueError
        If errors are invalid (non-positive or NaN)
    """
    if not (np.isfinite(flux_err_a) and np.isfinite(flux_err_b)):
        raise ValueError("Flux errors must be finite")

    if flux_err_a <= 0 or flux_err_b <= 0:
        raise ValueError("Flux errors must be positive")

    return (flux_a - flux_b) / np.hypot(flux_err_a, flux_err_b)


def calculate_m_metric(flux_a: float, flux_b: float) -> float:
    """Calculate m metric (modulation index) - adopted from VAST Tools.

        The m metric is the modulation index between two fluxes, proportional
        to the fractional variability.
        See Section 5 of Mooley et al. (2016) for details.
    DOI: 10.3847/0004-637X/818/2/105

        See VAST Tools: vasttools/utils.py::calculate_m_metric()

        Formula
    -------
        m = 2 * ((flux_a - flux_b) / (flux_a + flux_b))

    Parameters
    ----------
    flux_a : float
        Flux value at epoch A
    flux_b : float
        Flux value at epoch B

    Returns
    -------
        float
        m metric value

    Raises
    ------
        ValueError
        If sum of fluxes is zero
    """
    flux_sum = flux_a + flux_b
    if flux_sum == 0:
        raise ValueError("Sum of fluxes cannot be zero")

    return 2 * ((flux_a - flux_b) / flux_sum)


def calculate_v_metric(fluxes: np.ndarray) -> float:
    """Calculate V metric (coefficient of variation).

        The V metric is the fractional variability: std / mean.
        This is already implemented in DSA-110's variability_stats table,
        but provided here for completeness.

    Parameters
    ----------
    fluxes : array_like
        Array of flux values

    Returns
    -------
        float
        V metric value
    """
    if len(fluxes) == 0:
        return 0.0

    valid_fluxes = fluxes[np.isfinite(fluxes)]
    if len(valid_fluxes) < 2:
        return 0.0

    mean_flux = np.mean(valid_fluxes)
    if mean_flux == 0:
        return 0.0

    return float(np.std(valid_fluxes) / mean_flux)


def calculate_sigma_deviation(
    fluxes: np.ndarray,
    mean: float | None = None,
    std: float | None = None,
) -> float:
    """Calculate sigma deviation (maximum deviation from mean in units of standard deviation).

        This measures how many standard deviations the maximum or minimum flux
        deviates from the mean. This is a key metric for ESE detection.

        Formula
    -------
        sigma_deviation = max(
        |max_flux - mean_flux| / std_flux,
        |min_flux - mean_flux| / std_flux
        )

    Parameters
    ----------
    fluxes : array_like
        Array of flux values
    mean : float, optional
        Pre-computed mean (computed if not provided)
    std : float, optional
        Pre-computed standard deviation (computed if not provided)

    Returns
    -------
        float
        Sigma deviation value

    Raises
    ------
        ValueError
        If input is empty or all NaN
    """
    # Filter out NaN values
    valid_fluxes = fluxes[np.isfinite(fluxes)]

    if len(valid_fluxes) == 0:
        raise ValueError("Input array is empty or contains only NaN values")

    if len(valid_fluxes) == 1:
        # Single measurement: no variance, return 0.0
        return 0.0

    # Compute mean and std if not provided
    # Note: If precomputed stats are provided, they should be computed from valid_fluxes
    if mean is None:
        mean = float(np.mean(valid_fluxes))
    if std is None:
        std = float(np.std(valid_fluxes, ddof=1))  # Sample standard deviation

    # If std is zero (all values identical), return 0.0
    if std == 0.0:
        return 0.0

    # Calculate maximum deviation from mean using valid fluxes only
    max_flux = float(np.max(valid_fluxes))
    min_flux = float(np.min(valid_fluxes))

    sigma_deviation = max(abs(max_flux - mean) / std, abs(min_flux - mean) / std)

    return float(sigma_deviation)


def calculate_relative_flux(
    target_fluxes: np.ndarray,
    neighbor_fluxes: np.ndarray,
    neighbor_weights: np.ndarray | None = None,
    neighbor_errors: np.ndarray | None = None,
    use_robust_stats: bool = True,
) -> tuple[np.ndarray, float, float]:
    """Calculate relative flux (Target / Neighbor) time series and statistics.

        This method divides the target source flux by the weighted ensemble average of
        neighboring source fluxes at each epoch. It employs robust statistics (median)
        to reject outlier neighbors if requested.

        Robustness features:
        - Weighted ensemble average (optionally weighted by 1/sigma^2 if errors provided)
        - Median-based averaging (if use_robust_stats=True) to reject flaring neighbors
        - NaNs are masked and weights re-normalized per epoch

    Parameters
    ----------
    target_fluxes : array_like
        Array of target flux values (per epoch)
    neighbor_fluxes : array_like, 2D
        2D array of neighbor fluxes (n_epochs x n_neighbors)
    neighbor_weights : array_like, optional
        Optional manual weights for each neighbor (length n_neighbors)
    neighbor_errors : array_like, 2D, optional
        Optional 2D array of neighbor flux errors (n_epochs x n_neighbors).
        If provided, used for inverse-variance weighting.
    use_robust_stats : bool, optional
        If True, uses weighted median instead of mean for ensemble average.

    Returns
    -------
        tuple
        Tuple of:
        - relative_fluxes : array_like
        Array of relative flux values (Target / Neighbor_Avg)
        - mean_relative_flux : float
        Mean of the relative flux series
        - std_relative_flux : float
        Standard deviation of the relative flux series
    """
    if len(target_fluxes) == 0:
        raise ValueError("Target fluxes array is empty")

    # Ensure 1D target array
    target_fluxes = np.asarray(target_fluxes).flatten()
    n_epochs = len(target_fluxes)

    # Handle single neighbor case (1D array) -> reshape to (n_epochs, 1)
    neighbor_fluxes = np.asarray(neighbor_fluxes)
    if neighbor_fluxes.ndim == 1:
        if len(neighbor_fluxes) != n_epochs:
            raise ValueError(
                f"1D neighbor_fluxes length ({len(neighbor_fluxes)}) must match n_epochs ({n_epochs})"
            )
        neighbor_fluxes = neighbor_fluxes.reshape(n_epochs, 1)

    if neighbor_fluxes.shape[0] != n_epochs:
        raise ValueError(
            f"Neighbor flux epochs ({neighbor_fluxes.shape[0]}) do not match target epochs ({n_epochs})"
        )

    n_neighbors = neighbor_fluxes.shape[1]

    # Determine weights
    # Priority:
    # 1. 1/sigma^2 from neighbor_errors (if provided)
    # 2. Manual neighbor_weights (if provided)
    # 3. Uniform weights

    if neighbor_errors is not None:
        neighbor_errors = np.asarray(neighbor_errors)
        if neighbor_errors.shape != neighbor_fluxes.shape:
            raise ValueError("neighbor_errors shape must match neighbor_fluxes shape")
        # Avoid division by zero or negative variance
        with np.errstate(divide="ignore", invalid="ignore"):
            variance = neighbor_errors**2
            # Small epsilon to avoid div by zero
            weights_2d = 1.0 / (variance + 1e-20)
            # Mask invalid weights
            weights_2d[~np.isfinite(weights_2d)] = 0.0
    elif neighbor_weights is not None:
        base_weights = np.asarray(neighbor_weights).flatten()
        if len(base_weights) != n_neighbors:
            raise ValueError(
                f"Length of neighbor_weights ({len(base_weights)}) must match n_neighbors ({n_neighbors})"
            )
        # Broadcast 1D weights to 2D (same weight for a neighbor across all epochs)
        weights_2d = np.tile(base_weights, (n_epochs, 1))
    else:
        weights_2d = np.ones((n_epochs, n_neighbors))

    # Calculate ensemble average flux per epoch
    neighbor_avg_fluxes = np.zeros(n_epochs)

    for i in range(n_epochs):
        fluxes = neighbor_fluxes[i, :]
        epoch_weights = weights_2d[i, :]

        # Valid mask: finite flux AND finite positive weight
        valid = np.isfinite(fluxes) & (epoch_weights > 0)

        if not np.any(valid):
            neighbor_avg_fluxes[i] = np.nan
            continue

        v_fluxes = fluxes[valid]
        v_weights = epoch_weights[valid]

        # Normalize weights
        w_sum = np.sum(v_weights)
        if w_sum <= 0:
            neighbor_avg_fluxes[i] = np.nan
            continue

        norm_weights = v_weights / w_sum

        if use_robust_stats and len(v_fluxes) >= 3:
            # Weighted Median
            # Sort data and weights
            sort_idx = np.argsort(v_fluxes)
            sorted_fluxes = v_fluxes[sort_idx]
            sorted_weights = norm_weights[sort_idx]

            cumsum = np.cumsum(sorted_weights)
            cutoff = 0.5
            median_idx = np.searchsorted(cumsum, cutoff)
            # Handle edge case where searchsorted returns len
            median_idx = min(median_idx, len(sorted_fluxes) - 1)
            neighbor_avg_fluxes[i] = sorted_fluxes[median_idx]
        else:
            # Weighted Mean
            neighbor_avg_fluxes[i] = np.sum(v_fluxes * norm_weights)

    # Calculate relative flux
    with np.errstate(divide="ignore", invalid="ignore"):
        relative_fluxes = target_fluxes / neighbor_avg_fluxes

    # Calculate statistics on the relative flux lightcurve
    valid_rel = relative_fluxes[np.isfinite(relative_fluxes)]

    if len(valid_rel) > 0:
        mean_val = float(np.mean(valid_rel))
        std_val = float(np.std(valid_rel))
    else:
        mean_val = 0.0
        std_val = 0.0

    return relative_fluxes, mean_val, std_val
