"""Canonical correctness tests for the variability-metric implementations.

Correctness criteria (hardening-research-code):

1. Analytic closed forms — Mooley et al. (2016), ApJ 818, 105, §5. Expected
   values below are derived by hand; the derivation is stated in each test.
2. Reference implementation — the VAST pipeline/tools checkout at
   /data/radio-pipelines/askap-vast (vasttools/utils.py::pipeline_get_eta_metric,
   vast_pipeline/pipeline/pairs.py::calculate_vs_metric/calculate_m_metric,
   pipeline/finalise.py V metric with pandas .std(), i.e. ddof=1). Formulas are
   transcribed into expected values here; the tests do not import VAST.
3. Algebraic identity — with w = 1/sigma^2 the VAST eta form
   (N/(N-1)) * [mean(w f^2) - mean(w f)^2 / mean(w)]
   equals the reduced chi-squared about the weighted mean
   (1/(N-1)) * sum((f_i - <f>_w)^2 / sigma_i^2)
   exactly in real arithmetic, so the three repo implementations
   (photometry/metrics.py, photometry/variability.py, lightcurves/metrics.py)
   must agree to double-precision rounding.
4. Statistical calibration — for Gaussian noise with correctly stated errors,
   eta follows a reduced chi-squared distribution: E[eta] = 1,
   std(eta) = sqrt(2/(N-1)).

No golden data files: every expected value is derivable from the formulas
above.
"""

from __future__ import annotations

import importlib
import sys

import numpy as np
import pandas as pd
import pytest
from dsa110_continuum.lightcurves.metrics import compute_source_metrics
from dsa110_continuum.photometry import metrics as pmetrics
from dsa110_continuum.photometry import variability as pvar


def _var_df(fluxes, errors):
    return pd.DataFrame(
        {
            "normalized_flux_jy": np.asarray(fluxes, dtype=float),
            "normalized_flux_err_jy": np.asarray(errors, dtype=float),
        }
    )


def _eta_three_ways(fluxes, errors):
    fluxes = np.asarray(fluxes, dtype=float)
    errors = np.asarray(errors, dtype=float)
    eta_metrics = pmetrics.calculate_eta_metric(fluxes, 1.0 / errors**2)
    eta_var = pvar.calculate_eta_metric(_var_df(fluxes, errors))
    _, _, eta_lc = compute_source_metrics(fluxes, errors)
    return eta_metrics, eta_var, eta_lc


class TestEtaAnalytic:
    """Criterion 1: closed-form eta values.

    Two points f=[1,3], sigma=[1,1]: weighted mean 2, chi^2 = 1+1 = 2,
    eta = chi^2/(N-1) = 2.
    Three points f=[2,4,6], sigma=[1,2,1]: w=[1,1/4,1], <f>_w = 9/2.25 = 4,
    chi^2 = 4*1 + 0*0.25 + 4*1 = 8, eta = 8/2 = 4.
    Tolerance rtol=1e-12: the arithmetic is a handful of double ops
    (eps ~ 2.2e-16), so 1e-12 leaves ~1e3 headroom without admitting any
    formula error.
    """

    def test_two_point_equal_errors(self):
        for eta in _eta_three_ways([1.0, 3.0], [1.0, 1.0]):
            np.testing.assert_allclose(eta, 2.0, rtol=1e-12)

    def test_three_point_unequal_errors(self):
        for eta in _eta_three_ways([2.0, 4.0, 6.0], [1.0, 2.0, 1.0]):
            np.testing.assert_allclose(eta, 4.0, rtol=1e-12)

    def test_constant_flux_eta_zero(self):
        """Constant flux has chi^2 = 0 analytically. Cancellation floor is
        eps * mean(w f^2) ~ 2e-16 * 25 ~ 6e-15, so atol=1e-12 sits well above
        the floor and far below any physical eta."""
        for eta in _eta_three_ways([5.0, 5.0, 5.0, 5.0], [1.0, 1.0, 1.0, 1.0]):
            np.testing.assert_allclose(eta, 0.0, atol=1e-12)


class TestEtaAlgebraicIdentity:
    """Criterion 3: the VAST algebraic form and the two-pass reduced
    chi-squared are the same quantity in real arithmetic, so all three repo
    implementations must agree with an independent two-pass computation to
    double rounding. rtol=1e-10: N=50 accumulations at eps ~ 2.2e-16 give
    ~1e-14 relative error on well-conditioned data; 1e-10 leaves margin for
    the mild cancellation in the algebraic form."""

    def _dataset(self):
        rng = np.random.default_rng(42)
        fluxes = 10.0 + rng.normal(0.0, 2.0, 50)
        errors = rng.uniform(0.5, 1.5, 50)
        return fluxes, errors

    def test_all_implementations_match_two_pass_chi2(self):
        fluxes, errors = self._dataset()
        w = 1.0 / errors**2
        mean_w = np.sum(w * fluxes) / np.sum(w)
        eta_ref = np.sum(((fluxes - mean_w) / errors) ** 2) / (len(fluxes) - 1)
        for eta in _eta_three_ways(fluxes, errors):
            np.testing.assert_allclose(eta, eta_ref, rtol=1e-10)

    def test_scale_invariance(self):
        """eta(c*f, c*sigma) = eta(f, sigma) exactly in real arithmetic:
        both chi^2 numerator and denominator scale as c^2."""
        fluxes, errors = self._dataset()
        base = _eta_three_ways(fluxes, errors)
        scaled = _eta_three_ways(137.0 * fluxes, 137.0 * errors)
        np.testing.assert_allclose(scaled, base, rtol=1e-12)


class TestEtaStatisticalCalibration:
    """Criterion 4: for flux = const + N(0, sigma^2) with the true sigma
    supplied, eta ~ reduced chi-squared with E=1, std=sqrt(2/(N-1)). With a
    fixed seed the test is deterministic; the 5-sigma acceptance band
    1 +/- 5*sqrt(2/999) ~ [0.78, 1.22] comes from the chi-squared
    distribution, not from the observed value."""

    def test_pure_noise_eta_near_one(self):
        rng = np.random.default_rng(7)
        n = 1000
        fluxes = 10.0 + rng.normal(0.0, 1.0, n)
        errors = np.ones(n)
        band = 5.0 * np.sqrt(2.0 / (n - 1))
        for eta in _eta_three_ways(fluxes, errors):
            assert abs(eta - 1.0) < band


class TestEtaFloat32Stability:
    """Stability check: float32 input must agree with float64 within the
    conditioning of the algebraic form. The subtraction
    mean(w f^2) - mean(w f)^2/mean(w) cancels by a factor
    kappa ~ mean(w f^2)/(eta*(N-1)/N) ~ 100 for this dataset, so the float32
    error bound is ~eps32 * kappa ~ 1.2e-5; rtol=1e-4 gives ~8x margin."""

    def test_float32_matches_float64(self):
        rng = np.random.default_rng(3)
        fluxes = 10.0 + rng.normal(0.0, 1.0, 50)
        errors = rng.uniform(0.8, 1.2, 50)
        eta64 = _eta_three_ways(fluxes, errors)
        eta32 = _eta_three_ways(fluxes.astype(np.float32), errors.astype(np.float32))
        np.testing.assert_allclose(eta32, eta64, rtol=1e-4)


class TestTwoEpochMetrics:
    """Criterion 1 + 2: Mooley two-epoch metrics, identical formulas in the
    VAST reference (vast_pipeline/pipeline/pairs.py).

    Vs(3,1,1,1) = 2/sqrt(2) = sqrt(2).
    m(3,1) = 2*(3-1)/(3+1) = 1.
    Antisymmetry and the bound |m| <= 2 for positive fluxes hold for all
    inputs (property checks)."""

    def test_vs_analytic(self):
        np.testing.assert_allclose(
            pvar.calculate_vs_metric(3.0, 1.0, 1.0, 1.0), np.sqrt(2.0), rtol=1e-12
        )

    def test_m_analytic(self):
        np.testing.assert_allclose(pvar.calculate_m_metric(3.0, 1.0), 1.0, rtol=1e-12)

    def test_antisymmetry(self):
        rng = np.random.default_rng(11)
        for _ in range(20):
            a, b = rng.uniform(0.1, 10.0, 2)
            ea, eb = rng.uniform(0.05, 1.0, 2)
            np.testing.assert_allclose(
                pvar.calculate_vs_metric(a, b, ea, eb),
                -pvar.calculate_vs_metric(b, a, eb, ea),
                rtol=1e-12,
            )
            np.testing.assert_allclose(
                pvar.calculate_m_metric(a, b), -pvar.calculate_m_metric(b, a), rtol=1e-12
            )

    def test_m_bounded_for_positive_fluxes(self):
        rng = np.random.default_rng(13)
        for _ in range(50):
            a, b = rng.uniform(1e-3, 1e3, 2)
            assert abs(pvar.calculate_m_metric(a, b)) <= 2.0

    def test_vs_rejects_invalid_errors(self):
        with pytest.raises(ValueError):
            pvar.calculate_vs_metric(3.0, 1.0, 0.0, 1.0)
        with pytest.raises(ValueError):
            pvar.calculate_vs_metric(3.0, 1.0, -1.0, 1.0)
        with pytest.raises(ValueError):
            pvar.calculate_vs_metric(3.0, 1.0, np.nan, 1.0)

    def test_m_zero_sum_raises(self):
        with pytest.raises(ValueError):
            pvar.calculate_m_metric(1.0, -1.0)

    def test_lightcurves_vs_analytic(self):
        """Vs from the max/min pair: f=[1,2,5], sigma=[1,1,2] gives
        (5-1)/sqrt(2^2+1^2) = 4/sqrt(5)."""
        _, vs, _ = compute_source_metrics(np.array([1.0, 2.0, 5.0]), np.array([1.0, 1.0, 2.0]))
        np.testing.assert_allclose(vs, 4.0 / np.sqrt(5.0), rtol=1e-12)


class TestSourceModuleMetricImports:
    """photometry/source.py must not silently stub calculate_eta/vs/m_metric to
    0.0 when its imports fail: a 0.0-returning eta means variability is never
    flagged, with no exception anywhere. A failed import must abort loudly."""

    def test_import_failure_propagates_instead_of_stubbing(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "dsa110_continuum.catalog.multiwavelength", None)
        sys.modules.pop("dsa110_continuum.photometry.source", None)
        try:
            with pytest.raises(ImportError):
                importlib.import_module("dsa110_continuum.photometry.source")
        finally:
            sys.modules.pop("dsa110_continuum.photometry.source", None)

    def test_source_metrics_are_the_variability_implementations(self):
        psource = importlib.import_module("dsa110_continuum.photometry.source")
        assert psource.calculate_eta_metric is pvar.calculate_eta_metric
        assert psource.calculate_vs_metric is pvar.calculate_vs_metric
        assert psource.calculate_m_metric is pvar.calculate_m_metric


class TestRelativeLightcurveWiring:
    """Source.calculate_relative_lightcurve builds neighbor_flux_matrix but
    must actually bind and pass the transposed matrix to
    calculate_relative_flux; a missing neighbor_fluxes binding is a NameError
    on every call with >=1 valid neighbor (path previously uncovered).

    Analytic expectation: two identical neighbors at 2.0 Jy give a weighted
    median ensemble of 2.0 per epoch, so relative flux = target / 2.0."""

    def test_reaches_calculate_relative_flux_with_neighbor_matrix(self, monkeypatch):
        psource = importlib.import_module("dsa110_continuum.photometry.source")
        target = psource.Source("T1", ra_deg=10.0, dec_deg=16.0)
        target.measurements = pd.DataFrame(
            {
                "image_path": ["a.fits", "b.fits", "c.fits"],
                "mjd": [60000.0, 60001.0, 60002.0],
                "normalized_flux_jy": [1.0, 1.1, 0.9],
                "normalized_flux_err_jy": [0.1, 0.1, 0.1],
            }
        )
        monkeypatch.setattr(target, "find_stable_neighbors", lambda **kwargs: ["N1", "N2"])

        class _FakeNeighbor:
            measurements = pd.DataFrame(
                {
                    "image_path": ["a.fits", "b.fits", "c.fits"],
                    "normalized_flux_jy": [2.0, 2.0, 2.0],
                    "normalized_flux_err_jy": [0.2, 0.2, 0.2],
                }
            )

        monkeypatch.setattr(psource, "Source", lambda nid, products_db=None: _FakeNeighbor())

        result = target.calculate_relative_lightcurve()

        assert result["n_neighbors"] == 2
        np.testing.assert_allclose(result["relative_flux"], [0.5, 0.55, 0.45], rtol=1e-12)


class TestVMetricConvention:
    """Criterion 2: the VAST reference computes V with pandas .std(), i.e.
    the ddof=1 sample standard deviation. For f=[1,3]: std=sqrt(2), mean=2,
    V = sqrt(2)/2. photometry/metrics.py follows this convention."""

    def test_metrics_v_matches_vast_ddof1(self):
        v = pmetrics.calculate_v_metric(np.array([1.0, 3.0]))
        np.testing.assert_allclose(v, np.sqrt(2.0) / 2.0, rtol=1e-12)

    def test_variability_v_matches_vast_ddof1(self):
        """photometry/variability.py delegates to the canonical metrics.py
        implementation, so both copies follow the VAST convention (pandas
        .std(), ddof=1). Replaces the former UNVERIFIED ddof=0 pin."""
        v = pvar.calculate_v_metric(np.array([1.0, 3.0]))
        np.testing.assert_allclose(v, np.sqrt(2.0) / 2.0, rtol=1e-12)


class TestWeightedHelpers:
    """Criterion 1 + invariants for the weighted-statistics helpers in
    photometry/metrics.py.

    Weighted mean of [1,3] with w=[1,3] is (1+9)/4 = 2.5.
    Weighted variance (normalized by sum(w)) of [1,3] with w=[1,1]:
    mean 2, variance = (1+1)/2 = 1.
    Both are invariant under w -> c*w (weights enter homogeneously)."""

    def test_weighted_mean_analytic(self):
        m = pmetrics.calculate_weighted_mean(np.array([1.0, 3.0]), np.array([1.0, 3.0]))
        np.testing.assert_allclose(m, 2.5, rtol=1e-12)

    def test_weighted_variance_analytic(self):
        v = pmetrics.calculate_weighted_variance(np.array([1.0, 3.0]), np.array([1.0, 1.0]))
        np.testing.assert_allclose(v, 1.0, rtol=1e-12)

    def test_weight_scale_invariance(self):
        values = np.array([1.0, 2.0, 4.0, 8.0])
        weights = np.array([1.0, 0.5, 2.0, 0.25])
        for func in (pmetrics.calculate_weighted_mean, pmetrics.calculate_weighted_variance):
            np.testing.assert_allclose(
                func(values, 137.0 * weights), func(values, weights), rtol=1e-12
            )

    def test_zero_weights_return_nan(self):
        assert np.isnan(pmetrics.calculate_weighted_mean(np.array([1.0]), np.array([0.0])))
        assert np.isnan(pmetrics.calculate_weighted_variance(np.array([1.0]), np.array([0.0])))

    def test_chi_squared_documented_contract(self):
        """Criterion: the documented formula (module docstring states the
        chi^2-per-N normalization, NOT the (N-1)-dof reduced chi^2). Perfect
        model gives 0; f=[1,3], w=[1,1], model=2 gives (1+1)/2 = 1."""
        chi = pmetrics.calculate_chi_squared(np.array([1.0, 3.0]), np.array([1.0, 1.0]), 2.0)
        np.testing.assert_allclose(chi, 1.0, rtol=1e-12)
        chi0 = pmetrics.calculate_chi_squared(np.array([2.0, 2.0]), np.array([1.0, 1.0]), 2.0)
        np.testing.assert_allclose(chi0, 0.0, atol=1e-12)


class TestSigmaDeviation:
    """Criterion 1: f=[0,0,0,4]: mean 1, sample std (ddof=1)
    sqrt((1+1+1+9)/3) = 2, max deviation |4-1| = 3, so sigma_dev = 1.5.
    Applies to both copies (photometry/metrics.py and variability.py);
    their empty-input contracts differ and are pinned as documented."""

    def test_analytic_value_both_implementations(self):
        data = np.array([0.0, 0.0, 0.0, 4.0])
        np.testing.assert_allclose(pmetrics.calculate_sigma_deviation(data), 1.5, rtol=1e-12)
        np.testing.assert_allclose(pvar.calculate_sigma_deviation(data), 1.5, rtol=1e-12)

    def test_constant_is_zero(self):
        data = np.array([3.0, 3.0, 3.0])
        assert pmetrics.calculate_sigma_deviation(data) == 0.0
        assert pvar.calculate_sigma_deviation(data) == 0.0

    def test_empty_input_contracts(self):
        assert pmetrics.calculate_sigma_deviation(np.array([])) == 0.0
        with pytest.raises(ValueError):
            pvar.calculate_sigma_deviation(np.array([]))
