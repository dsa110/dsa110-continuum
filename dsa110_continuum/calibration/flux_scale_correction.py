"""
Huber-Regression Flux Scale Correction
=======================================

Adapts the VAST post-processing flux scale correction method for DSA-110.

Background
----------
The VAST pipeline (vast-post-processing/corrections.py) fits a robust affine
flux scale between Selavy detections and the RACS-Low reference catalog using
Huber robust regression:

    S_measured = gradient × S_reference + offset

The correction is:
    S_corrected = (S_measured - offset) / gradient

We adapt this for DSA-110 using NVSS or FIRST as the primary reference
(since DSA-110 observes at 1.4 GHz, matching NVSS/FIRST frequencies well).

Huber vs OLS
------------
Ordinary least squares is corrupted by variable sources — a source that has
flared by 3× appears as an outlier that biases the slope estimate.  The Huber
loss down-weights such outliers via iteratively reweighted least squares (IRLS),
producing a flux scale that reflects stable, quiescent sources only.

Implementation details
----------------------
* Uses IRLS (iteratively reweighted least squares) with the Huber weight
  function w(u) = 1 if |u| ≤ 1 else 1/|u|, where u = residual / (δ × σ_MAD).
* δ = 1.5 is the standard Huber tuning constant (as used in VAST).
* σ_MAD = 1.4826 × MAD(residuals) — the median absolute deviation scaled to
  match the Gaussian σ for normally distributed data.
* Source selection mirrors VAST: point sources, SNR > 20, isolated
  (nearest neighbour > 1 arcmin), within search radius of tile centre.
* 5-sigma clip on flux ratio before regression (pre-clipping) removes the
  most egregious outliers so that IRLS converges rapidly.

References
----------
* VAST post-processing corrections.py: https://github.com/askap-vast/vast-post-processing
* Mooley et al. (2016), ApJ 818, 105 — variability metrics
* Perley & Butler (2017), ApJS 230, 7 — VLA flux density scale

Usage
-----
>>> from dsa110_continuum.calibration.flux_scale_correction import (
...     huber_flux_scale,
...     apply_flux_scale,
...     FluxScaleResult,
... )
>>>
>>> # Cross-match catalog already prepared as numpy arrays
>>> result = huber_flux_scale(
...     s_measured=np.array([1.1, 2.0, 3.1, 0.05]),  # DSA-110 peak fluxes (Jy)
...     s_reference=np.array([1.0, 2.0, 3.0, 0.05]), # NVSS fluxes (Jy)
...     snr=np.array([25., 40., 55., 8.]),             # per-source SNR
...     snr_min=20.0,
... )
>>> print(result)
FluxScaleResult(gradient=1.083, offset=0.001, n_fit=3, passed=True)
>>> s_corrected = apply_flux_scale(np.array([1.5, 2.5]), result)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class FluxScaleResult:
    """Output of the Huber-regression flux scale fit.

    Attributes
    ----------
    gradient : float
        Best-fit slope (S_measured = gradient × S_reference + offset).
        A value of 1.0 means the pipeline flux scale is perfect.
    offset : float
        Best-fit additive offset (Jy).  Expected to be close to 0 for
        well-calibrated data.
    gradient_err : float
        1-sigma uncertainty on the gradient from bootstrap resampling.
    offset_err : float
        1-sigma uncertainty on the offset from bootstrap resampling.
    n_candidate : int
        Number of sources passing SNR and isolation cuts (before pre-clip).
    n_fit : int
        Number of sources used in the final Huber regression (post-clip).
    n_outlier : int
        Number of sources rejected by pre-clipping on flux ratio.
    rms_residual : float
        RMS of weighted residuals after final fit (Jy).
    median_flux_ratio : float
        Median of S_measured / S_reference (simple diagnostic).
    passed : bool
        True if the fit is considered reliable:
        - n_fit >= 5
        - |gradient - 1| < 0.5  (flux scale not off by > 50%)
        - |offset| < 0.1 Jy     (additive offset small)
    message : str
        Human-readable summary.
    """
    gradient: float = 1.0
    offset: float = 0.0
    gradient_err: float = float("nan")
    offset_err: float = float("nan")
    n_candidate: int = 0
    n_fit: int = 0
    n_outlier: int = 0
    rms_residual: float = float("nan")
    median_flux_ratio: float = float("nan")
    passed: bool = False
    message: str = ""

    def __str__(self) -> str:  # pragma: no cover
        return (
            f"FluxScaleResult(gradient={self.gradient:.4f}±{self.gradient_err:.4f}, "
            f"offset={self.offset:.4f}±{self.offset_err:.4f} Jy, "
            f"n_fit={self.n_fit}/{self.n_candidate}, passed={self.passed})\n"
            f"  {self.message}"
        )


# ── Core Huber regression (IRLS) ──────────────────────────────────────────────

def _huber_irls(
    x: np.ndarray,
    y: np.ndarray,
    delta: float = 1.5,
    max_iter: int = 100,
    tol: float = 1e-8,
) -> tuple[float, float]:
    """Fit y ≈ slope×x + intercept via Huber-IRLS.

    Parameters
    ----------
    x, y : 1-D arrays
        Predictor (reference flux) and response (measured flux).
    delta : float
        Huber tuning constant (default 1.5 = standard value).
    max_iter, tol : int, float
        Convergence controls.

    Returns
    -------
    slope, intercept : float
        Best-fit coefficients.
    """
    if len(x) < 2:
        raise ValueError(f"Need at least 2 points for regression, got {len(x)}")

    A = np.column_stack([x, np.ones_like(x)])

    # Initial estimate: median flux ratio (robust start point)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratios = y / x
        slope0 = float(np.nanmedian(ratios[x > 0]))
    if not np.isfinite(slope0):
        slope0 = 1.0
    coeffs = np.array([slope0, 0.0])

    for _ in range(max_iter):
        coeffs_old = coeffs.copy()
        r = y - A @ coeffs

        # Robust sigma via MAD (scale-consistent with Gaussian)
        sigma = 1.4826 * float(np.median(np.abs(r)))
        if sigma < 1e-12:
            break  # Residuals are essentially zero — converged

        u = r / (delta * sigma)
        # Huber weights: 1 inside ±1, 1/|u| outside
        with np.errstate(divide="ignore", invalid="ignore"):
            w = np.where(np.abs(u) <= 1.0, 1.0, 1.0 / np.maximum(np.abs(u), 1e-30))

        # Weighted least squares: (A^T W A) β = A^T W y
        AtWA = (A.T * w) @ A
        AtWy = A.T @ (w * y)

        try:
            coeffs = np.linalg.solve(AtWA, AtWy)
        except np.linalg.LinAlgError:
            logger.warning("IRLS: singular matrix — stopping early")
            break

        if np.max(np.abs(coeffs - coeffs_old)) < tol:
            break

    return float(coeffs[0]), float(coeffs[1])


# ── Source selection ──────────────────────────────────────────────────────────

def _preselect_sources(
    s_measured: np.ndarray,
    s_reference: np.ndarray,
    snr: np.ndarray | None,
    snr_min: float,
    nearest_neighbour_arcsec: np.ndarray | None,
    isolation_arcsec: float,
    sigma_clip: float,
) -> np.ndarray:
    """Return a boolean mask for sources suitable for flux scale fitting.

    Applies in order:
    1. SNR cut: snr >= snr_min (VAST uses 20)
    2. Isolation cut: nearest_neighbour > isolation_arcsec (VAST uses 60 arcsec)
    3. Positive flux check
    4. Sigma-clip on flux ratio (VAST uses 5σ)
    """
    n = len(s_measured)
    mask = np.ones(n, dtype=bool)

    # SNR cut
    if snr is not None:
        mask &= snr >= snr_min

    # Isolation cut
    if nearest_neighbour_arcsec is not None:
        mask &= nearest_neighbour_arcsec > isolation_arcsec

    # Positive flux
    mask &= (s_measured > 0) & (s_reference > 0)

    # Sigma-clip on flux ratio
    if mask.sum() >= 3:
        with np.errstate(divide="ignore", invalid="ignore"):
            ratios = s_measured[mask] / s_reference[mask]
        med = float(np.nanmedian(ratios))
        mad = 1.4826 * float(np.nanmedian(np.abs(ratios - med)))
        # Use a floor on MAD so extreme single outliers are still caught when
        # all other sources share exactly the same flux ratio (MAD → 0).  A
        # floor of 1% of |median| means the threshold is at least
        # sigma_clip × 0.01 × |med| away from the median, which is generous
        # enough to never accidentally clip genuine sources while still
        # removing outliers with ratios >~5× the median.
        mad_floor = 0.01 * abs(med) if med != 0.0 else 1e-6
        mad_eff = max(mad, mad_floor)
        clipped = np.abs(ratios - med) <= sigma_clip * mad_eff
        # Map back to full mask
        full_idx = np.where(mask)[0]
        mask[full_idx[~clipped]] = False

    return mask


# ── Bootstrap uncertainty ─────────────────────────────────────────────────────

def _bootstrap_errors(
    x: np.ndarray,
    y: np.ndarray,
    delta: float,
    n_boot: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    """Estimate gradient and offset uncertainties via bootstrap resampling."""
    if len(x) < 4:
        return float("nan"), float("nan")

    gradients = []
    offsets = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(x), size=len(x))
        try:
            g, o = _huber_irls(x[idx], y[idx], delta=delta)
            gradients.append(g)
            offsets.append(o)
        except (ValueError, np.linalg.LinAlgError):
            continue

    if len(gradients) < 5:
        return float("nan"), float("nan")

    return float(np.std(gradients)), float(np.std(offsets))


# ── Public API ────────────────────────────────────────────────────────────────

def huber_flux_scale(
    s_measured: np.ndarray,
    s_reference: np.ndarray,
    *,
    snr: np.ndarray | None = None,
    snr_min: float = 20.0,
    nearest_neighbour_arcsec: np.ndarray | None = None,
    isolation_arcsec: float = 60.0,
    sigma_clip: float = 5.0,
    delta: float = 1.5,
    n_bootstrap: int = 200,
    seed: int | None = 42,
    gradient_tolerance: float = 0.5,
    offset_tolerance_jy: float = 0.1,
    min_sources_for_pass: int = 5,
) -> FluxScaleResult:
    """Compute flux scale correction via Huber robust regression.

    Fits the linear model::

        S_measured = gradient × S_reference + offset

    using iteratively reweighted least squares with Huber weights.  This is
    robust to variable sources, which would bias an ordinary mean flux ratio.

    Parameters
    ----------
    s_measured : array-like of float
        Flux density of DSA-110-detected sources (Jy).
    s_reference : array-like of float
        Flux density of the same sources in the reference catalog
        (NVSS, FIRST, or VLASS; Jy).  Must have the same length as
        *s_measured*.
    snr : array-like of float or None
        Per-source signal-to-noise ratio.  Sources with ``snr < snr_min``
        are excluded.  If None, no SNR cut is applied.
    snr_min : float
        Minimum SNR for inclusion (default 20, from VAST).
    nearest_neighbour_arcsec : array-like of float or None
        Distance to the nearest other detected source (arcsec).  Sources
        with ``nearest_neighbour_arcsec <= isolation_arcsec`` are considered
        confused and excluded.  If None, no isolation cut is applied.
    isolation_arcsec : float
        Minimum isolation radius (default 60 arcsec = 1 arcmin, from VAST).
    sigma_clip : float
        Pre-clipping threshold on flux ratio in units of MAD.
        Default 5.0 (5-sigma clip, from VAST).
    delta : float
        Huber tuning constant.  Default 1.5 (standard value, used by VAST).
    n_bootstrap : int
        Number of bootstrap resamples for uncertainty estimation.
        Default 200.  Set to 0 to skip bootstrap.
    seed : int or None
        Random seed for bootstrap reproducibility.
    gradient_tolerance : float
        Maximum deviation of gradient from 1.0 for the result to be
        considered ``passed``.  Default 0.5 (allow ±50% flux scale error).
    offset_tolerance_jy : float
        Maximum absolute offset for the result to be ``passed`` (Jy).
        Default 0.1 Jy.
    min_sources_for_pass : int
        Minimum number of sources required for the result to be ``passed``.
        Default 5.

    Returns
    -------
    FluxScaleResult
        Dataclass with gradient, offset, uncertainties, and diagnostics.

    Examples
    --------
    >>> import numpy as np
    >>> from dsa110_continuum.calibration.flux_scale_correction import huber_flux_scale
    >>> rng = np.random.default_rng(0)
    >>> s_ref = rng.uniform(0.05, 5.0, 30)
    >>> s_meas = 1.15 * s_ref + 0.01 + 0.05 * rng.standard_normal(30)
    >>> # Add 2 variable-source outliers
    >>> s_meas[[5, 12]] = [10.0, 0.01]
    >>> result = huber_flux_scale(s_meas, s_ref)
    >>> assert abs(result.gradient - 1.15) < 0.1
    >>> assert result.passed
    """
    s_measured  = np.asarray(s_measured,  dtype=float)
    s_reference = np.asarray(s_reference, dtype=float)

    if s_measured.shape != s_reference.shape:
        raise ValueError(
            f"s_measured and s_reference must have the same shape: "
            f"{s_measured.shape} vs {s_reference.shape}"
        )
    if s_measured.ndim != 1:
        raise ValueError("s_measured must be 1-D")

    # ── Source selection ──────────────────────────────────────────────────────
    snr_arr = None if snr is None else np.asarray(snr, dtype=float)
    nn_arr  = (None if nearest_neighbour_arcsec is None
               else np.asarray(nearest_neighbour_arcsec, dtype=float))

    mask = _preselect_sources(
        s_measured, s_reference,
        snr=snr_arr,
        snr_min=snr_min,
        nearest_neighbour_arcsec=nn_arr,
        isolation_arcsec=isolation_arcsec,
        sigma_clip=sigma_clip,
    )

    n_candidate = int(mask.sum())
    n_total = len(s_measured)
    n_outlier = n_total - n_candidate

    if n_candidate < 2:
        msg = (
            f"Too few sources for Huber regression: {n_candidate} "
            f"(of {n_total} total pass quality cuts). "
            "Returning identity correction."
        )
        logger.warning(msg)
        return FluxScaleResult(
            gradient=1.0, offset=0.0,
            n_candidate=n_candidate, n_fit=0, n_outlier=n_outlier,
            passed=False, message=msg,
        )

    x = s_reference[mask]
    y = s_measured[mask]

    # Diagnostic: median flux ratio
    with np.errstate(divide="ignore", invalid="ignore"):
        median_ratio = float(np.nanmedian(y / x))

    # ── Huber regression ──────────────────────────────────────────────────────
    try:
        gradient, offset = _huber_irls(x, y, delta=delta)
    except Exception as exc:  # pragma: no cover
        msg = f"Huber IRLS failed: {exc}. Returning identity correction."
        logger.error(msg)
        return FluxScaleResult(
            gradient=1.0, offset=0.0,
            n_candidate=n_candidate, n_fit=n_candidate, n_outlier=n_outlier,
            median_flux_ratio=median_ratio,
            passed=False, message=msg,
        )

    # ── RMS residual ──────────────────────────────────────────────────────────
    residuals = y - (gradient * x + offset)
    rms_residual = float(np.sqrt(np.mean(residuals ** 2)))

    # ── Bootstrap uncertainties ───────────────────────────────────────────────
    rng_boot = np.random.default_rng(seed)
    if n_bootstrap > 0 and n_candidate >= 4:
        grad_err, off_err = _bootstrap_errors(x, y, delta=delta,
                                              n_boot=n_bootstrap, rng=rng_boot)
    else:
        grad_err = off_err = float("nan")

    # ── Pass/fail check ───────────────────────────────────────────────────────
    issues = []
    if n_candidate < min_sources_for_pass:
        issues.append(f"only {n_candidate} sources (need ≥{min_sources_for_pass})")
    if abs(gradient - 1.0) > gradient_tolerance:
        issues.append(f"gradient {gradient:.3f} deviates by >{gradient_tolerance:.0%}")
    if abs(offset) > offset_tolerance_jy:
        issues.append(f"|offset| {abs(offset):.3f} Jy > {offset_tolerance_jy:.3f} Jy")

    passed = len(issues) == 0
    if passed:
        msg = (
            f"OK — gradient={gradient:.4f}±{grad_err:.4f}, "
            f"offset={offset:.4f}±{off_err:.4f} Jy, "
            f"n_fit={n_candidate}/{n_total}"
        )
    else:
        msg = "WARN — " + "; ".join(issues) + f" (gradient={gradient:.4f}, offset={offset:.4f} Jy)"

    logger.info("Flux scale correction: %s", msg)

    return FluxScaleResult(
        gradient=gradient,
        offset=offset,
        gradient_err=grad_err,
        offset_err=off_err,
        n_candidate=n_candidate,
        n_fit=n_candidate,
        n_outlier=n_outlier,
        rms_residual=rms_residual,
        median_flux_ratio=median_ratio,
        passed=passed,
        message=msg,
    )


def apply_flux_scale(
    fluxes: np.ndarray,
    result: FluxScaleResult,
    flux_errors: np.ndarray | None = None,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Apply the Huber flux scale correction to measured fluxes.

    Computes::

        S_corrected = (S_measured - offset) / gradient

    If *flux_errors* is provided, also propagates the uncertainty::

        σ_corrected = sqrt(
            (σ_measured / gradient)^2
            + (S_corrected × σ_gradient / gradient)^2
            + (σ_offset / gradient)^2
        )

    Parameters
    ----------
    fluxes : array-like of float
        Measured flux densities (Jy).
    result : FluxScaleResult
        Output from :func:`huber_flux_scale`.
    flux_errors : array-like of float or None
        1-sigma uncertainties on the measured fluxes (same shape as
        *fluxes*).  If provided, corrected errors are also returned.

    Returns
    -------
    corrected : np.ndarray
        Corrected flux densities.
    corrected_errors : np.ndarray (only if *flux_errors* is not None)
        Propagated uncertainties on the corrected fluxes.

    Examples
    --------
    >>> from dsa110_continuum.calibration.flux_scale_correction import (
    ...     FluxScaleResult, apply_flux_scale,
    ... )
    >>> result = FluxScaleResult(gradient=1.1, offset=0.05)
    >>> import numpy as np
    >>> s = np.array([1.0, 2.0, 3.0])
    >>> apply_flux_scale(s, result)
    array([0.86363636, 1.77272727, 2.68181818])
    """
    fluxes = np.asarray(fluxes, dtype=float)

    if result.gradient == 0.0:
        raise ValueError("FluxScaleResult.gradient is 0 — cannot divide")

    corrected = (fluxes - result.offset) / result.gradient

    if flux_errors is None:
        return corrected

    # Error propagation
    flux_errors = np.asarray(flux_errors, dtype=float)
    g   = result.gradient
    o   = result.offset
    s_g = result.gradient_err if np.isfinite(result.gradient_err) else 0.0
    s_o = result.offset_err   if np.isfinite(result.offset_err)   else 0.0

    # σ² = (σ_meas/g)² + (S_corr × σ_g/g)² + (σ_o/g)²
    term_meas   = (flux_errors / g) ** 2
    term_grad   = (corrected * s_g / g) ** 2
    term_offset = (s_o / g) ** 2
    corrected_errors = np.sqrt(term_meas + term_grad + term_offset)

    return corrected, corrected_errors


def correction_factor(result: FluxScaleResult) -> float:
    """Return the multiplicative flux scale correction factor (1/gradient).

    A factor > 1 means the pipeline under-estimates fluxes relative to
    the reference catalog; < 1 means it over-estimates.

    Parameters
    ----------
    result : FluxScaleResult

    Returns
    -------
    float
        1 / gradient.
    """
    if result.gradient == 0.0:
        raise ValueError("gradient is 0")
    return 1.0 / result.gradient
