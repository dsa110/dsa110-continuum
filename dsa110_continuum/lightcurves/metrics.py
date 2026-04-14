"""Mooley et al. (2016) multi-epoch variability metrics for DSA-110.

Reference
---------
Mooley, K. P. et al. (2016), ApJ, 818, 105.
DOI: 10.3847/0004-637X/818/2/105

Metrics
-------
m : modulation index
    ``m = σ_S / <S>``
    Fractional RMS variability.  Sensitive to any kind of variability but
    depends on the noise level of individual measurements.

Vs : flux variability significance
    ``Vs = (S_max - S_min) / sqrt(σ_max² + σ_min²)``
    t-statistic that the brightest and faintest detections are inconsistent
    with constant flux.  Less biased by measurement noise than *m*.

η : reduced chi-squared (eta)
    ``η = χ²_r = (1/(N-1)) Σ_i [(S_i - <S>_w)² / σ_i²]``
    where ``<S>_w = (Σ S_i/σ_i²) / (Σ 1/σ_i²)`` is the weighted mean.
    η ≫ 1 indicates significant variability relative to measurement noise.

Variable candidate thresholds (Mooley et al. 2016 defaults)
------------------------------------------------------------
*  ``Vs > 4.0``
*  ``η  > 2.5``

Usage
-----
::

    from dsa110_continuum.lightcurves.metrics import compute_metrics, flag_candidates

    metrics_df = compute_metrics(stacked_df)
    candidates = flag_candidates(metrics_df)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Thresholds (Mooley et al. 2016 §5)
# ---------------------------------------------------------------------------

VS_THRESHOLD: float = 4.0
ETA_THRESHOLD: float = 2.5

# Minimum number of epochs required for meaningful metric calculation.
MIN_EPOCHS: int = 2


# ---------------------------------------------------------------------------
# Per-source result dataclass
# ---------------------------------------------------------------------------


@dataclass
class VariabilityMetrics:
    """Variability metrics for one source.

    Attributes
    ----------
    source_id : int
        Stable integer source identifier.
    ra_deg : float
        Right ascension in degrees.
    dec_deg : float
        Declination in degrees.
    n_epochs : int
        Number of epochs with a measurement.
    mean_flux : float
        Unweighted mean flux in Jy.
    std_flux : float
        Sample standard deviation of flux in Jy (NaN if n < 2).
    m : float
        Modulation index ``σ_S / <S>`` (NaN if n < 2 or mean ≤ 0).
    Vs : float
        Flux variability significance (NaN if n < 2 or errors not available).
    eta : float
        Reduced chi-squared η (NaN if n < 2 or errors not available).
    is_variable_candidate : bool
        ``True`` if ``Vs > VS_THRESHOLD`` or ``η > ETA_THRESHOLD``.
    catalog_flux_jy : float or None
        Reference catalog flux if available.
    spectral_index : float or None
        Spectral index if available.
    """

    source_id: int
    ra_deg: float
    dec_deg: float
    n_epochs: int
    mean_flux: float
    std_flux: float
    m: float
    Vs: float
    eta: float
    is_variable_candidate: bool
    catalog_flux_jy: Optional[float] = None
    spectral_index: Optional[float] = None


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------


def compute_source_metrics(
    fluxes: np.ndarray,
    errors: np.ndarray,
) -> tuple[float, float, float]:
    """Compute (m, Vs, η) for a single source's light curve.

    Parameters
    ----------
    fluxes : ndarray, shape (N,)
        Flux density measurements in Jy.
    errors : ndarray, shape (N,)
        1-sigma flux uncertainties in Jy.

    Returns
    -------
    m : float
        Modulation index.  NaN if N < 2 or mean ≤ 0.
    Vs : float
        Variability significance.  NaN if N < 2 or errors invalid.
    eta : float
        Reduced chi-squared.  NaN if N < 2 or errors invalid.

    Notes
    -----
    * Entries with non-finite flux or non-positive error are silently dropped.
    * If the cleaned array has fewer than 2 points all three metrics are NaN.
    """
    # Clean: keep only rows with finite flux AND positive error
    valid = np.isfinite(fluxes) & np.isfinite(errors) & (errors > 0)
    f = fluxes[valid].astype(float)
    e = errors[valid].astype(float)
    n = len(f)

    if n < MIN_EPOCHS:
        return float("nan"), float("nan"), float("nan")

    # ------------------------------------------------------------------
    # Modulation index  m = σ / <S>
    # ------------------------------------------------------------------
    mean_s = float(np.mean(f))
    std_s = float(np.std(f, ddof=1))
    m = std_s / mean_s if mean_s > 0 else float("nan")

    # ------------------------------------------------------------------
    # Vs = (S_max - S_min) / sqrt(σ_max² + σ_min²)
    # ------------------------------------------------------------------
    idx_max = int(np.argmax(f))
    idx_min = int(np.argmin(f))
    denom = np.hypot(e[idx_max], e[idx_min])
    Vs = (f[idx_max] - f[idx_min]) / denom if denom > 0 else float("nan")

    # ------------------------------------------------------------------
    # η = reduced χ²  = (1/(N-1)) Σ [(S_i - <S>_w)² / σ_i²]
    # ------------------------------------------------------------------
    weights = 1.0 / (e ** 2)
    mean_w = float(np.average(f, weights=weights))
    chi2 = float(np.sum(((f - mean_w) / e) ** 2))
    eta = chi2 / (n - 1)

    return m, Vs, eta


def compute_metrics(
    lc: pd.DataFrame,
    flux_col: str = "measured_flux_jy",
    err_col: str = "flux_err_jy",
    source_col: str = "source_id",
) -> pd.DataFrame:
    """Compute Mooley et al. (2016) variability metrics for every source.

    Parameters
    ----------
    lc : DataFrame
        Stacked light curve DataFrame with one row per source-epoch.
        Must contain *source_col*, ``ra_deg``, ``dec_deg``, *flux_col*, and
        *err_col*.  Optional columns ``catalog_flux_jy`` and
        ``spectral_index`` are preserved if present.
    flux_col : str
        Column name for flux density values.
    err_col : str
        Column name for flux uncertainties.
    source_col : str
        Column name for source group identifier.

    Returns
    -------
    DataFrame
        One row per source, indexed by *source_col*, with columns:
        ``ra_deg``, ``dec_deg``, ``n_epochs``, ``mean_flux``, ``std_flux``,
        ``m``, ``Vs``, ``eta``, ``catalog_flux_jy`` (optional),
        ``spectral_index`` (optional), ``is_variable_candidate``.

    Raises
    ------
    KeyError
        If required columns are absent from *lc*.
    ValueError
        If *lc* is empty.
    """
    required = {source_col, "ra_deg", "dec_deg", flux_col, err_col}
    missing = required - set(lc.columns)
    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}")
    if lc.empty:
        raise ValueError("Light curve DataFrame is empty.")

    records: list[dict] = []

    for sid, group in lc.groupby(source_col):
        n = len(group)
        fluxes = group[flux_col].values.astype(float)
        errors = group[err_col].values.astype(float)
        ra = float(group["ra_deg"].iloc[0])
        dec = float(group["dec_deg"].iloc[0])

        mean_s = float(np.nanmean(fluxes))
        std_s = float(np.nanstd(fluxes, ddof=1)) if n > 1 else float("nan")

        m, Vs, eta = compute_source_metrics(fluxes, errors)

        rec: dict = {
            source_col: sid,
            "ra_deg": ra,
            "dec_deg": dec,
            "n_epochs": n,
            "mean_flux": mean_s,
            "std_flux": std_s,
            "m": m,
            "Vs": Vs,
            "eta": eta,
        }

        if "catalog_flux_jy" in group.columns:
            rec["catalog_flux_jy"] = float(group["catalog_flux_jy"].iloc[0])
        if "spectral_index" in group.columns:
            rec["spectral_index"] = float(group["spectral_index"].iloc[0])

        records.append(rec)

    result = pd.DataFrame(records).set_index(source_col)
    result = flag_candidates(result)
    return result


# ---------------------------------------------------------------------------
# Candidate flagging
# ---------------------------------------------------------------------------


def flag_candidates(
    metrics: pd.DataFrame,
    vs_threshold: float = VS_THRESHOLD,
    eta_threshold: float = ETA_THRESHOLD,
) -> pd.DataFrame:
    """Add ``is_variable_candidate`` boolean column to a metrics DataFrame.

    A source is flagged as a variable candidate if
    ``Vs > vs_threshold`` OR ``η > eta_threshold``.  NaN metrics are treated
    as not exceeding the threshold (i.e. the source is not flagged).

    Parameters
    ----------
    metrics : DataFrame
        Output of :func:`compute_metrics`.
    vs_threshold : float
        Threshold on Vs (default 4.0).
    eta_threshold : float
        Threshold on η (default 2.5).

    Returns
    -------
    DataFrame
        Copy of *metrics* with ``is_variable_candidate`` column added or
        replaced.
    """
    metrics = metrics.copy()
    vs_flag = metrics["Vs"].fillna(0.0) > vs_threshold
    eta_flag = metrics["eta"].fillna(0.0) > eta_threshold
    metrics["is_variable_candidate"] = vs_flag | eta_flag
    return metrics


# ---------------------------------------------------------------------------
# Convenience: summary statistics
# ---------------------------------------------------------------------------


def variability_summary(metrics: pd.DataFrame) -> dict:
    """Return a summary dictionary of variability statistics.

    Intended for logging and QA reporting.

    Parameters
    ----------
    metrics : DataFrame
        Output of :func:`compute_metrics` / :func:`flag_candidates`.

    Returns
    -------
    dict
        Keys: ``n_sources``, ``n_candidates``, ``fraction_variable``,
        ``median_m``, ``median_Vs``, ``median_eta``.
    """
    n_total = len(metrics)
    n_cand = int(metrics["is_variable_candidate"].sum()) if "is_variable_candidate" in metrics.columns else 0

    return {
        "n_sources": n_total,
        "n_candidates": n_cand,
        "fraction_variable": n_cand / n_total if n_total > 0 else float("nan"),
        "median_m": float(np.nanmedian(metrics["m"].values)) if "m" in metrics.columns else float("nan"),
        "median_Vs": float(np.nanmedian(metrics["Vs"].values)) if "Vs" in metrics.columns else float("nan"),
        "median_eta": float(np.nanmedian(metrics["eta"].values)) if "eta" in metrics.columns else float("nan"),
    }
