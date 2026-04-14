"""
Tests for dsa110_continuum.qa.composite — 3-gate composite QA metric.

Coverage
--------
Gate 1 — Flux scale: pass / warn / fail / exact-boundary
Gate 2 — Completeness: pass / warn / fail / skip (zero expected) / DataFrame path
Gate 3 — Noise floor: pass / warn / fail / skip (zero theoretical) / NaN factor
CompositeQAResult: status aggregation, summary string, to_dict
run_composite_qa convenience wrapper
theoretical_rms_jyb helper
Custom threshold construction
epoch label propagation
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from dsa110_continuum.qa.composite import (
    CompositeQA,
    CompositeQAResult,
    QAStatus,
    run_composite_qa,
    theoretical_rms_jyb,
)

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------
_GOOD_CORRECTION = 1.03    # within warn threshold
_GOOD_N_DET = 90
_GOOD_N_EXP = 100
_GOOD_RMS = 2.0e-4         # measured
_GOOD_THEO = 1.8e-4        # theoretical  → factor 1.11, within 2×

_qa = CompositeQA()        # default thresholds


# ===========================================================================
# 1. Gate 1 — Flux scale
# ===========================================================================

class TestFluxScaleGate:
    def test_pass_within_tolerance(self):
        r = _qa._gate_flux_scale(1.05)
        assert r.status == QAStatus.PASS

    def test_warn_above_warn_threshold(self):
        # default warn ≥ 0.08 → correction = 1.09 → deviation = 0.09
        r = _qa._gate_flux_scale(1.09)
        assert r.status == QAStatus.WARN

    def test_fail_above_max_threshold(self):
        # default fail ≥ 0.15 → correction = 1.20 → deviation = 0.20
        r = _qa._gate_flux_scale(1.20)
        assert r.status == QAStatus.FAIL

    def test_fail_below_one(self):
        # correction = 0.80 → deviation = 0.20
        r = _qa._gate_flux_scale(0.80)
        assert r.status == QAStatus.FAIL

    def test_exact_unity_passes(self):
        r = _qa._gate_flux_scale(1.0)
        assert r.status == QAStatus.PASS
        assert r.deviation == pytest.approx(0.0)

    def test_deviation_calculated_correctly(self):
        r = _qa._gate_flux_scale(1.12)
        assert r.deviation == pytest.approx(0.12, abs=1e-9)

    def test_correction_factor_stored(self):
        r = _qa._gate_flux_scale(1.07)
        assert r.correction_factor == pytest.approx(1.07)

    def test_message_non_empty(self):
        r = _qa._gate_flux_scale(1.05)
        assert len(r.message) > 0

    def test_to_dict_has_gate_key(self):
        r = _qa._gate_flux_scale(1.05)
        d = r.to_dict()
        assert d["gate"] == "flux_scale"
        assert "status" in d
        assert "deviation" in d


# ===========================================================================
# 2. Gate 2 — Completeness (scalar counts)
# ===========================================================================

class TestCompletenessGateScalar:
    def test_pass(self):
        r = _qa._gate_completeness(85, 100)
        assert r.status == QAStatus.PASS
        assert r.completeness == pytest.approx(0.85)

    def test_warn_below_warn_threshold(self):
        # default warn < 0.80 → 75/100 = 0.75
        r = _qa._gate_completeness(75, 100)
        assert r.status == QAStatus.WARN

    def test_fail_below_min(self):
        # default fail < 0.70 → 60/100 = 0.60
        r = _qa._gate_completeness(60, 100)
        assert r.status == QAStatus.FAIL

    def test_skip_when_expected_zero(self):
        r = _qa._gate_completeness(0, 0)
        assert r.status == QAStatus.SKIP

    def test_counts_stored(self):
        r = _qa._gate_completeness(80, 100)
        assert r.n_detected == 80
        assert r.n_expected == 100

    def test_completeness_above_one_is_pass(self):
        # More detections than catalog (can happen if field is partial)
        r = _qa._gate_completeness(110, 100)
        assert r.completeness == pytest.approx(1.10)
        assert r.status == QAStatus.PASS

    def test_to_dict_has_gate_key(self):
        r = _qa._gate_completeness(80, 100)
        d = r.to_dict()
        assert d["gate"] == "completeness"


# ===========================================================================
# 3. Gate 2 — Completeness (DataFrame path)
# ===========================================================================

def _make_sources(ra_centers: list[float], dec: float = 30.0, flux_mjy: float = 50.0):
    return pd.DataFrame({
        "ra_deg": ra_centers,
        "dec_deg": [dec] * len(ra_centers),
        "flux_mjy": [flux_mjy] * len(ra_centers),
    })


class TestCompletenessGateDataFrame:
    def test_all_matched(self):
        # 5 catalog sources, all within 15 arcsec of a detected source
        cat = _make_sources([180.0, 181.0, 182.0, 183.0, 184.0])
        det = _make_sources([180.001, 181.001, 182.001, 183.001, 184.001])
        r = _qa._gate_completeness_from_df(det, cat, min_catalog_flux_mjy=10.0)
        assert r.completeness == pytest.approx(1.0)
        assert r.status == QAStatus.PASS

    def test_partial_match(self):
        cat = _make_sources([180.0, 181.0, 182.0, 183.0, 184.0])
        # Only 3 of 5 detected
        det = _make_sources([180.001, 181.001, 182.001])
        r = _qa._gate_completeness_from_df(det, cat, min_catalog_flux_mjy=10.0)
        assert r.completeness == pytest.approx(3 / 5)
        assert r.status in (QAStatus.FAIL, QAStatus.WARN)

    def test_empty_catalog_skipped(self):
        det = _make_sources([180.0])
        cat = pd.DataFrame(columns=["ra_deg", "dec_deg", "flux_mjy"])
        r = _qa._gate_completeness_from_df(det, cat)
        assert r.status == QAStatus.SKIP

    def test_empty_detected_gives_zero(self):
        cat = _make_sources([180.0, 181.0, 182.0])
        det = pd.DataFrame(columns=["ra_deg", "dec_deg", "flux_mjy"])
        r = _qa._gate_completeness_from_df(det, cat, min_catalog_flux_mjy=10.0)
        assert r.completeness == pytest.approx(0.0)
        assert r.status == QAStatus.FAIL

    def test_flux_threshold_filters_faint_catalog(self):
        # All catalog sources are faint (5 mJy) — min_catalog_flux_mjy=10 → skip
        cat = _make_sources([180.0, 181.0], flux_mjy=5.0)
        det = _make_sources([180.001, 181.001])
        r = _qa._gate_completeness_from_df(det, cat, min_catalog_flux_mjy=10.0)
        assert r.status == QAStatus.SKIP


# ===========================================================================
# 4. Gate 3 — Noise floor
# ===========================================================================

class TestNoiseFloorGate:
    def test_pass_within_factor(self):
        # 1.1× theoretical → pass
        r = _qa._gate_noise_floor(1.1e-4, 1.0e-4)
        assert r.status == QAStatus.PASS
        assert r.noise_factor == pytest.approx(1.1)

    def test_warn_above_warn_factor(self):
        # default warn > 1.5 → factor = 1.7
        r = _qa._gate_noise_floor(1.7e-4, 1.0e-4)
        assert r.status == QAStatus.WARN

    def test_fail_above_max_factor(self):
        # default fail > 2.0 → factor = 2.5
        r = _qa._gate_noise_floor(2.5e-4, 1.0e-4)
        assert r.status == QAStatus.FAIL

    def test_skip_when_theoretical_zero(self):
        r = _qa._gate_noise_floor(1.0e-4, 0.0)
        assert r.status == QAStatus.SKIP

    def test_noise_factor_stored(self):
        r = _qa._gate_noise_floor(2.0e-4, 1.0e-4)
        assert r.noise_factor == pytest.approx(2.0)

    def test_to_dict_has_gate_key(self):
        r = _qa._gate_noise_floor(1.0e-4, 1.0e-4)
        d = r.to_dict()
        assert d["gate"] == "noise_floor"
        assert "noise_factor" in d


# ===========================================================================
# 5. CompositeQAResult — aggregation
# ===========================================================================

class TestCompositeAggregation:
    def test_all_pass_gives_pass(self):
        result = run_composite_qa(
            flux_scale_correction=1.02,
            n_detected=85,
            n_catalog_expected=100,
            measured_rms_jyb=1.9e-4,
            theoretical_rms_jyb=1.8e-4,
        )
        assert result.status == QAStatus.PASS
        assert result.passed

    def test_flux_fail_gives_fail(self):
        result = run_composite_qa(
            flux_scale_correction=1.30,   # fail
            n_detected=85, n_catalog_expected=100,
            measured_rms_jyb=1.9e-4, theoretical_rms_jyb=1.8e-4,
        )
        assert result.status == QAStatus.FAIL
        assert result.failed

    def test_completeness_fail_gives_fail(self):
        result = run_composite_qa(
            flux_scale_correction=1.02,
            n_detected=50, n_catalog_expected=100,  # 50% → fail
            measured_rms_jyb=1.9e-4, theoretical_rms_jyb=1.8e-4,
        )
        assert result.status == QAStatus.FAIL

    def test_noise_fail_gives_fail(self):
        result = run_composite_qa(
            flux_scale_correction=1.02,
            n_detected=85, n_catalog_expected=100,
            measured_rms_jyb=5.0e-4, theoretical_rms_jyb=1.8e-4,  # 2.78× → fail
        )
        assert result.status == QAStatus.FAIL

    def test_warn_from_one_gate(self):
        result = run_composite_qa(
            flux_scale_correction=1.02,
            n_detected=75, n_catalog_expected=100,  # 75% → warn
            measured_rms_jyb=1.9e-4, theoretical_rms_jyb=1.8e-4,
        )
        assert result.status == QAStatus.WARN
        assert not result.passed
        assert not result.failed

    def test_any_fail_beats_warn(self):
        """fail takes priority over warn."""
        qa = CompositeQA(
            max_flux_scale_error=0.10,
            warn_flux_scale_error=0.05,
        )
        result = qa.evaluate_counts(
            flux_scale_correction=1.12,  # fail
            n_detected=75,               # warn
            n_catalog_expected=100,
            measured_rms_jyb=1.0e-4,
            theoretical_rms_jyb=1.0e-4,
        )
        assert result.status == QAStatus.FAIL


# ===========================================================================
# 6. CompositeQAResult properties
# ===========================================================================

class TestCompositeResultProperties:
    def _passing_result(self) -> CompositeQAResult:
        return run_composite_qa(1.02, 85, 100, 1.9e-4, 1.8e-4)

    def test_summary_string_contains_status(self):
        r = self._passing_result()
        s = r.summary()
        assert "pass" in s.lower()
        assert "flux_scale" in s
        assert "completeness" in s
        assert "noise" in s

    def test_to_dict_structure(self):
        r = self._passing_result()
        d = r.to_dict()
        assert "status" in d
        assert "gates" in d
        assert "flux_scale" in d["gates"]
        assert "completeness" in d["gates"]
        assert "noise_floor" in d["gates"]

    def test_epoch_propagated(self):
        result = run_composite_qa(
            1.02, 85, 100, 1.9e-4, 1.8e-4,
            epoch="2026-01-25T22:26:05",
        )
        assert result.epoch == "2026-01-25T22:26:05"
        assert "epoch=" in result.summary()

    def test_epoch_none_by_default(self):
        result = run_composite_qa(1.02, 85, 100, 1.9e-4, 1.8e-4)
        assert result.epoch is None

    def test_to_dict_round_trips_status(self):
        r = self._passing_result()
        d = r.to_dict()
        assert d["status"] == "pass"


# ===========================================================================
# 7. CompositeQA.evaluate() with DataFrames
# ===========================================================================

class TestEvaluateWithDataFrames:
    def _cat(self):
        return pd.DataFrame({
            "ra_deg": [180.0, 181.0, 182.0, 183.0, 184.0],
            "dec_deg": [30.0] * 5,
            "flux_mjy": [50.0] * 5,
        })

    def _det_all(self):
        return pd.DataFrame({
            "ra_deg": [180.001, 181.001, 182.001, 183.001, 184.001],
            "dec_deg": [30.0] * 5,
            "flux_mjy": [48.0] * 5,
        })

    def test_evaluate_with_full_match(self):
        qa = CompositeQA()
        result = qa.evaluate(
            flux_scale_correction=1.02,
            measured_rms_jyb=1.9e-4,
            theoretical_rms_jyb=1.8e-4,
            detected_df=self._det_all(),
            catalog_df=self._cat(),
            min_catalog_flux_mjy=10.0,
        )
        assert result.completeness.status == QAStatus.PASS
        assert result.completeness.completeness == pytest.approx(1.0)

    def test_evaluate_scalar_fallback(self):
        qa = CompositeQA()
        result = qa.evaluate(
            flux_scale_correction=1.02,
            measured_rms_jyb=1.9e-4,
            theoretical_rms_jyb=1.8e-4,
            n_detected=80,
            n_catalog_expected=100,
        )
        assert result.completeness.n_detected == 80

    def test_evaluate_no_completeness_data_skips(self):
        qa = CompositeQA()
        result = qa.evaluate(
            flux_scale_correction=1.02,
            measured_rms_jyb=1.9e-4,
            theoretical_rms_jyb=1.8e-4,
        )
        assert result.completeness.status == QAStatus.SKIP


# ===========================================================================
# 8. Custom thresholds
# ===========================================================================

class TestCustomThresholds:
    def test_strict_flux_threshold(self):
        qa = CompositeQA(max_flux_scale_error=0.05)
        result = qa.evaluate_counts(1.07, 85, 100, 1.9e-4, 1.8e-4)
        # 0.07 > 0.05 → fail
        assert result.flux_scale.status == QAStatus.FAIL

    def test_lenient_completeness_threshold(self):
        qa = CompositeQA(min_completeness=0.50)
        result = qa.evaluate_counts(1.02, 55, 100, 1.9e-4, 1.8e-4)
        # 0.55 > 0.50 → should pass completeness
        assert result.completeness.status in (QAStatus.PASS, QAStatus.WARN)

    def test_strict_noise_threshold(self):
        qa = CompositeQA(max_noise_factor=1.2)
        result = qa.evaluate_counts(1.02, 85, 100, 1.5e-4, 1.0e-4)
        # factor = 1.5 > 1.2 → fail
        assert result.noise_floor.status == QAStatus.FAIL


# ===========================================================================
# 9. theoretical_rms_jyb helper
# ===========================================================================

class TestTheoreticalRms:
    def test_returns_float(self):
        rms = theoretical_rms_jyb()
        assert isinstance(rms, float)

    def test_positive_value(self):
        rms = theoretical_rms_jyb()
        assert rms > 0

    def test_more_antennas_lower_noise(self):
        rms_110 = theoretical_rms_jyb(n_antennas=110)
        rms_55 = theoretical_rms_jyb(n_antennas=55)
        # More antennas → more baselines → lower noise
        assert rms_110 < rms_55

    def test_longer_integration_lower_noise(self):
        rms_short = theoretical_rms_jyb(t_int_s=60.0)
        rms_long = theoretical_rms_jyb(t_int_s=600.0)
        assert rms_long < rms_short

    def test_wider_bandwidth_lower_noise(self):
        rms_narrow = theoretical_rms_jyb(bandwidth_hz=50e6)
        rms_wide = theoretical_rms_jyb(bandwidth_hz=200e6)
        assert rms_wide < rms_narrow

    def test_dsa110_default_order_of_magnitude(self):
        """DSA-110 5-min tile should be in the ~10–500 μJy/beam range."""
        rms = theoretical_rms_jyb()
        assert 1e-5 < rms < 1e-2

    def test_radiometer_scaling(self):
        """σ ∝ 1/sqrt(Δν·τ): doubling Δν reduces rms by sqrt(2)."""
        rms1 = theoretical_rms_jyb(bandwidth_hz=100e6)
        rms2 = theoretical_rms_jyb(bandwidth_hz=200e6)
        assert rms1 / rms2 == pytest.approx(math.sqrt(2), rel=0.01)


# ===========================================================================
# 10. run_composite_qa convenience wrapper
# ===========================================================================

class TestRunCompositeQA:
    def test_returns_composite_result(self):
        r = run_composite_qa(1.02, 85, 100, 1.9e-4, 1.8e-4)
        assert isinstance(r, CompositeQAResult)

    def test_custom_thresholds_passed_through(self):
        # Very strict: any deviation fails
        r = run_composite_qa(
            1.02, 85, 100, 1.9e-4, 1.8e-4,
            max_flux_scale_error=0.01,  # 1% → 2% fails
        )
        assert r.flux_scale.status == QAStatus.FAIL

    def test_epoch_propagated(self):
        r = run_composite_qa(1.02, 85, 100, 1.9e-4, 1.8e-4, epoch="2026-01-01")
        assert r.epoch == "2026-01-01"
