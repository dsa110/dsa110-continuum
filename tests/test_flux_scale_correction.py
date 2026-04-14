"""
Tests for dsa110_continuum.calibration.flux_scale_correction
=============================================================

Tests the Huber-regression flux scale correction module.
All tests are cloud-safe (numpy + scipy only, no CASA, no H17).
"""
from __future__ import annotations

import numpy as np
import pytest

from dsa110_continuum.calibration.flux_scale_correction import (
    FluxScaleResult,
    _huber_irls,
    _preselect_sources,
    apply_flux_scale,
    correction_factor,
    huber_flux_scale,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def clean_data():
    """50 sources with gradient=1.2, offset=0.05, small Gaussian noise."""
    rng = np.random.default_rng(0)
    s_ref  = rng.uniform(0.1, 5.0, 50)
    s_meas = 1.2 * s_ref + 0.05 + 0.02 * rng.standard_normal(50)
    snr    = rng.uniform(25.0, 80.0, 50)
    return s_meas, s_ref, snr


@pytest.fixture
def data_with_outliers(clean_data):
    """Same as clean_data but with 5 variable-source outliers injected."""
    s_meas, s_ref, snr = [arr.copy() for arr in clean_data]
    rng = np.random.default_rng(7)
    outlier_idx = rng.choice(len(s_meas), size=5, replace=False)
    s_meas[outlier_idx] *= rng.uniform(3.0, 10.0, size=5)  # flared sources
    return s_meas, s_ref, snr


# ══════════════════════════════════════════════════════════════════════════════
# 1.  Core IRLS
# ══════════════════════════════════════════════════════════════════════════════

class TestHuberIRLS:
    """Unit tests for the _huber_irls function."""

    def test_perfect_linear_data(self):
        x = np.arange(1.0, 11.0)
        y = 2.0 * x + 0.5
        slope, intercept = _huber_irls(x, y)
        np.testing.assert_allclose(slope, 2.0, atol=1e-6)
        np.testing.assert_allclose(intercept, 0.5, atol=1e-6)

    def test_noisy_data_recovers_slope(self):
        rng = np.random.default_rng(42)
        x = rng.uniform(0.1, 5.0, 100)
        y = 1.5 * x + 0.1 + 0.05 * rng.standard_normal(100)
        slope, intercept = _huber_irls(x, y)
        np.testing.assert_allclose(slope, 1.5, atol=0.05)
        np.testing.assert_allclose(intercept, 0.1, atol=0.1)

    def test_outlier_does_not_dominate(self):
        """A single extreme outlier should not break the fit."""
        rng = np.random.default_rng(0)
        x = np.arange(1.0, 21.0)
        y = 2.0 * x + 0.0
        # Inject an extreme outlier
        y_noisy = y.copy()
        y_noisy[10] = 200.0   # should be 22, is 200
        slope, intercept = _huber_irls(x, y_noisy)
        # Huber should still recover slope ≈ 2
        np.testing.assert_allclose(slope, 2.0, atol=0.3,
                                   err_msg=f"Outlier corrupted slope: {slope:.3f}")

    def test_raises_on_insufficient_data(self):
        with pytest.raises(ValueError, match="at least 2"):
            _huber_irls(np.array([1.0]), np.array([2.0]))

    def test_unit_slope_unit_intercept(self):
        x = np.linspace(0.5, 5.0, 20)
        slope, intercept = _huber_irls(x, x)
        np.testing.assert_allclose(slope, 1.0, atol=1e-6)
        np.testing.assert_allclose(intercept, 0.0, atol=1e-6)

    def test_different_delta_affects_outlier_sensitivity(self):
        """Larger δ makes regression MORE sensitive to outliers (fewer down-weights).

        With 25% contamination (+8 additive shift on 10 of 40 points), a
        tight δ (0.5) aggressively down-weights the contaminated points and
        stays close to the true slope=2, while a very loose δ (50) barely
        down-weights them and ends up with a biased slope.
        """
        rng = np.random.default_rng(99)
        n = 40
        x = np.linspace(1.0, 10.0, n)
        y = 2.0 * x + rng.standard_normal(n) * 0.3
        # 25% contamination: shift 10 points upward
        outlier_idx = rng.choice(n, 10, replace=False)
        y[outlier_idx] += 8.0

        slope_tight, _ = _huber_irls(x, y, delta=0.5)
        slope_loose, _ = _huber_irls(x, y, delta=50.0)
        # Tight δ down-weights outlier more aggressively → closer to true slope 2.0
        assert abs(slope_tight - 2.0) < abs(slope_loose - 2.0)


# ══════════════════════════════════════════════════════════════════════════════
# 2.  Source pre-selection
# ══════════════════════════════════════════════════════════════════════════════

class TestPreselect:
    """Tests for the _preselect_sources filtering function."""

    def test_no_cuts_all_pass(self):
        mask = _preselect_sources(
            np.array([1.0, 2.0, 3.0]),
            np.array([1.0, 2.0, 3.0]),
            snr=None, snr_min=0.0,
            nearest_neighbour_arcsec=None, isolation_arcsec=0.0,
            sigma_clip=100.0,
        )
        assert mask.all()

    def test_snr_cut_removes_low_snr(self):
        snr = np.array([5.0, 25.0, 50.0, 15.0])
        mask = _preselect_sources(
            np.ones(4), np.ones(4),
            snr=snr, snr_min=20.0,
            nearest_neighbour_arcsec=None, isolation_arcsec=0.0,
            sigma_clip=100.0,
        )
        np.testing.assert_array_equal(mask, [False, True, True, False])

    def test_isolation_cut_removes_confused(self):
        nn = np.array([30.0, 90.0, 120.0, 45.0])  # arcsec
        mask = _preselect_sources(
            np.ones(4), np.ones(4),
            snr=None, snr_min=0.0,
            nearest_neighbour_arcsec=nn, isolation_arcsec=60.0,
            sigma_clip=100.0,
        )
        np.testing.assert_array_equal(mask, [False, True, True, False])

    def test_negative_flux_excluded(self):
        mask = _preselect_sources(
            np.array([-1.0, 0.0, 2.0]),
            np.array([1.0, 1.0, 2.0]),
            snr=None, snr_min=0.0,
            nearest_neighbour_arcsec=None, isolation_arcsec=0.0,
            sigma_clip=100.0,
        )
        np.testing.assert_array_equal(mask, [False, False, True])

    def test_sigma_clip_removes_extreme_ratios(self):
        """A source with flux ratio 10× higher than median should be clipped."""
        s_ref  = np.ones(20)
        s_meas = np.ones(20)
        s_meas[0] = 100.0   # extreme outlier
        mask = _preselect_sources(
            s_meas, s_ref,
            snr=None, snr_min=0.0,
            nearest_neighbour_arcsec=None, isolation_arcsec=0.0,
            sigma_clip=5.0,
        )
        assert not mask[0], "Outlier source should be clipped"
        assert mask[1:].all(), "Other sources should pass"


# ══════════════════════════════════════════════════════════════════════════════
# 3.  huber_flux_scale (public API)
# ══════════════════════════════════════════════════════════════════════════════

class TestHuberFluxScale:
    """Tests for the main huber_flux_scale function."""

    def test_returns_flux_scale_result(self, clean_data):
        s_meas, s_ref, snr = clean_data
        result = huber_flux_scale(s_meas, s_ref, snr=snr)
        assert isinstance(result, FluxScaleResult)

    def test_correct_gradient_clean_data(self, clean_data):
        """Should recover gradient ≈ 1.2 from clean data."""
        s_meas, s_ref, snr = clean_data
        result = huber_flux_scale(s_meas, s_ref, snr=snr)
        np.testing.assert_allclose(result.gradient, 1.2, atol=0.05,
                                   err_msg=f"gradient={result.gradient:.4f}")

    def test_correct_offset_clean_data(self, clean_data):
        """Should recover offset ≈ 0.05 from clean data."""
        s_meas, s_ref, snr = clean_data
        result = huber_flux_scale(s_meas, s_ref, snr=snr)
        np.testing.assert_allclose(result.offset, 0.05, atol=0.05,
                                   err_msg=f"offset={result.offset:.4f}")

    def test_robust_to_outliers(self, data_with_outliers):
        """Gradient should still be ≈ 1.2 even with 5 variable-source outliers."""
        s_meas, s_ref, snr = data_with_outliers
        result = huber_flux_scale(s_meas, s_ref, snr=snr)
        np.testing.assert_allclose(result.gradient, 1.2, atol=0.15,
                                   err_msg=f"Outliers corrupted gradient: {result.gradient:.4f}")

    def test_ols_corrupted_by_outliers(self, data_with_outliers):
        """OLS should be MORE corrupted by outliers than Huber.

        This test verifies that Huber adds value over naive mean-ratio.
        """
        s_meas, s_ref, snr = data_with_outliers
        # Huber estimate
        result = huber_flux_scale(s_meas, s_ref, snr=snr)

        # Naive OLS
        mask = (s_meas > 0) & (s_ref > 0) & (snr >= 20.0)
        x = s_ref[mask]; y = s_meas[mask]
        A = np.column_stack([x, np.ones_like(x)])
        ols_coeffs, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
        ols_gradient = ols_coeffs[0]

        # Both should recover ~1.2, but Huber should be closer
        huber_err = abs(result.gradient - 1.2)
        ols_err = abs(ols_gradient - 1.2)
        assert huber_err <= ols_err + 0.05, (
            f"Huber gradient error {huber_err:.4f} should be ≤ OLS error {ols_err:.4f}"
        )

    def test_passed_flag_set_for_good_data(self, clean_data):
        s_meas, s_ref, snr = clean_data
        result = huber_flux_scale(s_meas, s_ref, snr=snr)
        assert result.passed, f"Expected passed=True; message: {result.message}"

    def test_passed_false_for_insufficient_sources(self):
        """With only 3 sources, result should not be trusted."""
        result = huber_flux_scale(
            np.array([1.0, 2.0, 3.0]),
            np.array([1.0, 2.0, 3.0]),
            min_sources_for_pass=5,
        )
        # 3 sources < min_sources_for_pass=5 → passed=False
        assert not result.passed

    def test_passed_false_for_large_gradient(self):
        """Gradient = 5 (500% off) should fail."""
        s_ref  = np.ones(20)
        s_meas = 5.0 * s_ref  # gradient = 5
        result = huber_flux_scale(s_meas, s_ref, gradient_tolerance=0.5)
        assert not result.passed

    def test_n_candidate_counts_correct(self):
        """n_candidate should reflect SNR filtering."""
        s_ref  = np.ones(10)
        s_meas = np.ones(10)
        snr    = np.array([5.,5.,30.,30.,30.,30.,30.,30.,30.,30.])
        result = huber_flux_scale(s_meas, s_ref, snr=snr, snr_min=20.0)
        assert result.n_candidate == 8  # 8 pass SNR ≥ 20

    def test_too_few_sources_returns_identity(self):
        """With < 2 sources after cuts, should return gradient=1, offset=0."""
        s_ref  = np.array([1.0])
        s_meas = np.array([1.5])
        result = huber_flux_scale(s_meas, s_ref)
        assert result.gradient == 1.0
        assert result.offset == 0.0
        assert not result.passed

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="same shape"):
            huber_flux_scale(np.array([1., 2.]), np.array([1.]))

    def test_2d_input_raises(self):
        with pytest.raises(ValueError, match="1-D"):
            huber_flux_scale(np.ones((3, 2)), np.ones((3, 2)))

    def test_gradient_error_finite_with_enough_data(self, clean_data):
        s_meas, s_ref, snr = clean_data
        result = huber_flux_scale(s_meas, s_ref, snr=snr, n_bootstrap=100)
        assert np.isfinite(result.gradient_err)
        assert np.isfinite(result.offset_err)

    def test_bootstrap_zero_skips_uncertainty(self, clean_data):
        s_meas, s_ref, snr = clean_data
        result = huber_flux_scale(s_meas, s_ref, snr=snr, n_bootstrap=0)
        assert np.isnan(result.gradient_err)
        assert np.isnan(result.offset_err)

    def test_median_flux_ratio_reasonable(self, clean_data):
        s_meas, s_ref, snr = clean_data
        result = huber_flux_scale(s_meas, s_ref, snr=snr)
        # Median ratio should be close to the true gradient (≈1.2)
        np.testing.assert_allclose(result.median_flux_ratio, 1.2, atol=0.2)

    def test_isolation_cut_applied(self):
        """Sources with small nearest-neighbour distance should be excluded."""
        rng = np.random.default_rng(5)
        n = 20
        s_ref  = rng.uniform(0.1, 5.0, n)
        s_meas = 1.1 * s_ref + 0.01 * rng.standard_normal(n)
        nn_arcsec = np.full(n, 120.0)
        nn_arcsec[:5] = 10.0  # 5 confused sources

        result_no_isolation = huber_flux_scale(s_meas, s_ref, isolation_arcsec=0.0)
        result_with_isolation = huber_flux_scale(
            s_meas, s_ref,
            nearest_neighbour_arcsec=nn_arcsec, isolation_arcsec=60.0,
        )
        assert result_with_isolation.n_candidate <= result_no_isolation.n_candidate

    def test_reproducible_with_same_seed(self, clean_data):
        s_meas, s_ref, snr = clean_data
        r1 = huber_flux_scale(s_meas, s_ref, snr=snr, seed=0)
        r2 = huber_flux_scale(s_meas, s_ref, snr=snr, seed=0)
        assert r1.gradient == r2.gradient
        assert r1.gradient_err == r2.gradient_err

    def test_rms_residual_positive_and_finite(self, clean_data):
        s_meas, s_ref, snr = clean_data
        result = huber_flux_scale(s_meas, s_ref, snr=snr)
        assert np.isfinite(result.rms_residual)
        assert result.rms_residual > 0.0


# ══════════════════════════════════════════════════════════════════════════════
# 4.  apply_flux_scale
# ══════════════════════════════════════════════════════════════════════════════

class TestApplyFluxScale:
    """Tests for the apply_flux_scale function."""

    def test_identity_correction(self):
        result = FluxScaleResult(gradient=1.0, offset=0.0)
        s = np.array([1.0, 2.0, 3.0])
        np.testing.assert_array_equal(apply_flux_scale(s, result), s)

    def test_gradient_only(self):
        result = FluxScaleResult(gradient=2.0, offset=0.0)
        s = np.array([2.0, 4.0, 6.0])
        expected = np.array([1.0, 2.0, 3.0])
        np.testing.assert_allclose(apply_flux_scale(s, result), expected)

    def test_offset_only(self):
        result = FluxScaleResult(gradient=1.0, offset=0.5)
        s = np.array([1.5, 2.5])
        expected = np.array([1.0, 2.0])
        np.testing.assert_allclose(apply_flux_scale(s, result), expected)

    def test_gradient_and_offset(self):
        """S_corrected = (S_measured - offset) / gradient"""
        result = FluxScaleResult(gradient=1.1, offset=0.1)
        s = np.array([1.2])   # -> (1.2 - 0.1) / 1.1 = 1.0
        np.testing.assert_allclose(apply_flux_scale(s, result), [1.0], atol=1e-10)

    def test_returns_np_array(self):
        result = FluxScaleResult(gradient=1.5, offset=0.0)
        out = apply_flux_scale(np.array([3.0]), result)
        assert isinstance(out, np.ndarray)

    def test_zero_gradient_raises(self):
        result = FluxScaleResult(gradient=0.0, offset=0.0)
        with pytest.raises(ValueError, match="gradient is 0"):
            apply_flux_scale(np.array([1.0]), result)

    def test_error_propagation_shape(self):
        result = FluxScaleResult(gradient=1.1, offset=0.0,
                                 gradient_err=0.05, offset_err=0.01)
        s = np.array([1.0, 2.0, 3.0])
        s_err = np.array([0.1, 0.1, 0.1])
        corrected, corr_err = apply_flux_scale(s, result, flux_errors=s_err)
        assert corrected.shape == s.shape
        assert corr_err.shape == s.shape

    def test_error_propagation_positive(self):
        result = FluxScaleResult(gradient=1.1, offset=0.0,
                                 gradient_err=0.05, offset_err=0.01)
        s = np.array([1.0, 2.0])
        s_err = np.array([0.05, 0.05])
        _, corr_err = apply_flux_scale(s, result, flux_errors=s_err)
        assert np.all(corr_err > 0)

    def test_error_propagation_larger_with_gradient_uncertainty(self):
        """Errors should increase when gradient_err is non-zero."""
        result_no_grad_err  = FluxScaleResult(gradient=1.1, offset=0.0,
                                               gradient_err=0.0, offset_err=0.0)
        result_with_grad_err = FluxScaleResult(gradient=1.1, offset=0.0,
                                               gradient_err=0.1, offset_err=0.0)
        s = np.array([5.0])
        s_err = np.array([0.1])
        _, err_no  = apply_flux_scale(s, result_no_grad_err,  flux_errors=s_err)
        _, err_yes = apply_flux_scale(s, result_with_grad_err, flux_errors=s_err)
        assert err_yes[0] > err_no[0]

    def test_roundtrip_clean_data(self, clean_data):
        """Correcting data generated with a known scale should recover reference fluxes."""
        s_meas, s_ref, snr = clean_data
        result = huber_flux_scale(s_meas, s_ref, snr=snr, n_bootstrap=0)
        s_corrected = apply_flux_scale(s_meas, result)
        # After correction, corrected should be close to s_ref (within noise)
        relative_error = np.abs(s_corrected - s_ref) / s_ref
        assert np.median(relative_error) < 0.1, (
            f"Median relative error after correction: {np.median(relative_error):.3f}"
        )

    def test_no_errors_returns_array_not_tuple(self):
        result = FluxScaleResult(gradient=1.2, offset=0.0)
        out = apply_flux_scale(np.array([1.0, 2.0]), result)
        assert not isinstance(out, tuple)

    def test_with_errors_returns_tuple(self):
        result = FluxScaleResult(gradient=1.2, offset=0.0, gradient_err=0.05)
        out = apply_flux_scale(np.array([1.0]), result, flux_errors=np.array([0.1]))
        assert isinstance(out, tuple)
        assert len(out) == 2


# ══════════════════════════════════════════════════════════════════════════════
# 5.  correction_factor
# ══════════════════════════════════════════════════════════════════════════════

class TestCorrectionFactor:
    def test_identity(self):
        result = FluxScaleResult(gradient=1.0, offset=0.0)
        assert correction_factor(result) == pytest.approx(1.0)

    def test_gradient_2(self):
        result = FluxScaleResult(gradient=2.0, offset=0.0)
        assert correction_factor(result) == pytest.approx(0.5)

    def test_gradient_half(self):
        result = FluxScaleResult(gradient=0.5, offset=0.0)
        assert correction_factor(result) == pytest.approx(2.0)

    def test_zero_gradient_raises(self):
        result = FluxScaleResult(gradient=0.0)
        with pytest.raises(ValueError, match="gradient is 0"):
            correction_factor(result)


# ══════════════════════════════════════════════════════════════════════════════
# 6.  Integration: end-to-end flux scale workflow
# ══════════════════════════════════════════════════════════════════════════════

class TestFluxScaleIntegration:
    """Integration tests simulating a realistic flux scale correction workflow."""

    def test_full_workflow_with_variable_sources(self):
        """Simulate a real field: 100 sources, 10 variable, recover correct scale."""
        rng = np.random.default_rng(42)
        n = 100
        true_gradient = 1.08
        true_offset   = 0.005  # Jy

        s_ref  = rng.uniform(0.05, 8.0, n)
        snr    = rng.uniform(5.0, 100.0, n)
        noise  = 0.03 * rng.standard_normal(n)
        s_meas = true_gradient * s_ref + true_offset + noise

        # 10% variable: fluxes boosted by random factor 2–5×
        var_idx = rng.choice(n, size=10, replace=False)
        s_meas[var_idx] *= rng.uniform(2.0, 5.0, size=10)

        # Force SNR ≥ 20 for 60 sources, < 20 for remaining
        snr[:60] = rng.uniform(20.0, 100.0, 60)
        snr[60:] = rng.uniform(5.0, 15.0, 40)

        result = huber_flux_scale(
            s_meas, s_ref, snr=snr, snr_min=20.0,
            sigma_clip=5.0, n_bootstrap=100, seed=0,
        )

        assert result.passed, f"Expected passed=True; message: {result.message}"
        np.testing.assert_allclose(result.gradient, true_gradient, atol=0.05,
                                   err_msg=f"gradient={result.gradient:.4f}")
        np.testing.assert_allclose(result.offset, true_offset, atol=0.05,
                                   err_msg=f"offset={result.offset:.4f}")

    def test_corrected_fluxes_match_reference(self):
        """After applying correction, corrected fluxes should agree with reference."""
        rng = np.random.default_rng(3)
        n = 40
        true_gradient = 0.92
        true_offset   = -0.02

        s_ref  = rng.uniform(0.1, 5.0, n)
        s_meas = true_gradient * s_ref + true_offset + 0.02 * rng.standard_normal(n)

        result = huber_flux_scale(s_meas, s_ref, n_bootstrap=0)
        s_corrected = apply_flux_scale(s_meas, result)

        rel_err = np.abs(s_corrected - s_ref) / s_ref
        assert np.median(rel_err) < 0.05

    def test_message_populated(self, clean_data):
        s_meas, s_ref, snr = clean_data
        result = huber_flux_scale(s_meas, s_ref, snr=snr)
        assert isinstance(result.message, str)
        assert len(result.message) > 0

    def test_n_outlier_nonzero_when_clipping_occurs(self):
        """n_outlier should be non-zero when extreme outliers are present."""
        rng = np.random.default_rng(1)
        s_ref  = rng.uniform(0.1, 3.0, 30)
        s_meas = 1.1 * s_ref.copy()
        # Inject 3 extreme outliers
        s_meas[[0, 5, 20]] = [50.0, 0.001, 100.0]
        result = huber_flux_scale(s_meas, s_ref, sigma_clip=5.0)
        assert result.n_outlier >= 3
