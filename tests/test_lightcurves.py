"""Tests for dsa110_continuum.lightcurves — stacking and variability metrics.

All tests are synthetic.  No HDF5 files, no CASA, no network.

Coverage
--------
* parse_epoch_utc — various filename formats + error cases
* assign_source_ids — source_name path, coordinate path, edge cases
* stack_csvs — empty + multi-epoch stacking
* compute_source_metrics — m, Vs, η correctness and NaN edge cases
* compute_metrics — full pipeline on synthetic DataFrames
* flag_candidates — threshold logic
* variability_summary — statistics on metrics DataFrame
* VariabilityMetrics dataclass
"""

from __future__ import annotations

import io
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from dsa110_continuum.lightcurves.stacker import (
    assign_source_ids,
    parse_epoch_utc,
    stack_csvs,
)
from dsa110_continuum.lightcurves.metrics import (
    VS_THRESHOLD,
    ETA_THRESHOLD,
    VariabilityMetrics,
    compute_metrics,
    compute_source_metrics,
    flag_candidates,
    variability_summary,
)

# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------


def _make_lc_csv(
    epoch_str: str,
    source_names: list[str],
    fluxes: list[float],
    errors: list[float],
    cat_fluxes: list[float] | None = None,
) -> str:
    """Return CSV text for a single epoch forced-phot CSV."""
    rows = []
    for i, (name, f, e) in enumerate(zip(source_names, fluxes, errors)):
        ra = 10.0 + i
        dec = 20.0 + i
        cat = cat_fluxes[i] if cat_fluxes else f
        rows.append(f"{name},{ra:.4f},{dec:.4f},{f:.6f},{e:.6f},{cat:.6f}")
    header = "source_name,ra_deg,dec_deg,measured_flux_jy,flux_err_jy,catalog_flux_jy"
    return header + "\n" + "\n".join(rows)


SOURCES = ["J0001+2000", "J0002+2001", "J0003+2002"]
EPOCH1 = "2026-01-25T2200"
EPOCH2 = "2026-01-26T0000"


def _write_epoch_csv(tmp_path: Path, epoch_str: str, fluxes: list[float]) -> Path:
    errors = [f * 0.05 for f in fluxes]
    content = _make_lc_csv(epoch_str, SOURCES, fluxes, errors)
    fname = f"{epoch_str}_forced_phot.csv"
    p = tmp_path / fname
    p.write_text(content)
    return p


# ===========================================================================
# parse_epoch_utc
# ===========================================================================

class TestParseEpochUtc:
    def test_compact_hhmm(self):
        assert parse_epoch_utc("2026-01-25T2200_forced_phot.csv") == "2026-01-25T22:00:00"

    def test_compact_hhmm_different_hour(self):
        assert parse_epoch_utc("2026-02-12T0000_forced_phot.csv") == "2026-02-12T00:00:00"

    def test_full_iso_with_colons(self):
        assert parse_epoch_utc("2026-01-25T22:00:00_forced_phot.csv") == "2026-01-25T22:00:00"

    def test_full_path_uses_filename(self):
        result = parse_epoch_utc("/data/products/2026-01-25T2200_forced_phot.csv")
        assert result == "2026-01-25T22:00:00"

    def test_pathlib_input(self):
        result = parse_epoch_utc(Path("/data/2026-01-25T2200_forced_phot.csv"))
        assert result == "2026-01-25T22:00:00"

    def test_no_pattern_raises(self):
        with pytest.raises(ValueError, match="Cannot parse epoch"):
            parse_epoch_utc("no_timestamp_here.csv")

    def test_result_is_string(self):
        r = parse_epoch_utc("2026-01-25T2200_forced_phot.csv")
        assert isinstance(r, str)

    def test_midnight_0000(self):
        assert parse_epoch_utc("2026-03-01T0000_forced_phot.csv") == "2026-03-01T00:00:00"


# ===========================================================================
# assign_source_ids
# ===========================================================================

class TestAssignSourceIds:
    def test_source_name_path(self):
        df = pd.DataFrame({
            "source_name": ["S1", "S2", "S1", "S3"],
            "ra_deg": [10, 11, 10, 12],
            "dec_deg": [20, 21, 20, 22],
        })
        result = assign_source_ids(df)
        assert "source_id" in result.columns
        # S1 rows should have same id
        ids = result[result["source_name"] == "S1"]["source_id"].values
        assert ids[0] == ids[1]
        # Different names → different ids
        assert result[result["source_name"] == "S1"]["source_id"].iloc[0] != \
               result[result["source_name"] == "S2"]["source_id"].iloc[0]

    def test_source_name_path_n_unique(self):
        df = pd.DataFrame({
            "source_name": ["A", "B", "A", "C", "B"],
        })
        result = assign_source_ids(df)
        assert result["source_id"].nunique() == 3

    def test_coordinate_path(self):
        # Two sources, each appearing twice (same position)
        df = pd.DataFrame({
            "ra_deg":  [10.0, 20.0, 10.0001, 20.0001],
            "dec_deg": [20.0, 30.0, 20.0001, 30.0001],
        })
        result = assign_source_ids(df, match_arcsec=1.0)
        assert result["source_id"].nunique() == 2
        # First and third rows → same source
        assert result.iloc[0]["source_id"] == result.iloc[2]["source_id"]

    def test_missing_columns_raises(self):
        df = pd.DataFrame({"x": [1, 2]})
        with pytest.raises(ValueError, match="source_name"):
            assign_source_ids(df)

    def test_returns_copy(self):
        df = pd.DataFrame({
            "source_name": ["A", "B"],
        })
        result = assign_source_ids(df)
        assert "source_id" not in df.columns   # original unchanged

    def test_single_row(self):
        df = pd.DataFrame({"source_name": ["X"], "ra_deg": [1.0], "dec_deg": [2.0]})
        result = assign_source_ids(df)
        assert len(result) == 1
        assert result.iloc[0]["source_id"] == 0


# ===========================================================================
# stack_csvs
# ===========================================================================

class TestStackCsvs:
    def test_empty_raises(self, tmp_path):
        with pytest.raises(ValueError, match="empty"):
            stack_csvs([])

    def test_single_csv(self, tmp_path):
        p = _write_epoch_csv(tmp_path, EPOCH1, [0.1, 0.2, 0.3])
        df = stack_csvs([p])
        assert "epoch_utc" in df.columns
        assert "source_id" in df.columns
        assert len(df) == len(SOURCES)

    def test_two_epochs_stacked(self, tmp_path):
        p1 = _write_epoch_csv(tmp_path, EPOCH1, [0.1, 0.2, 0.3])
        p2 = _write_epoch_csv(tmp_path, EPOCH2, [0.11, 0.19, 0.28])
        df = stack_csvs([p1, p2])
        assert len(df) == len(SOURCES) * 2
        assert df["epoch_utc"].nunique() == 2

    def test_date_column(self, tmp_path):
        p = _write_epoch_csv(tmp_path, EPOCH1, [0.1, 0.2, 0.3])
        df = stack_csvs([p])
        assert "date" in df.columns
        assert df["date"].iloc[0] == "2026-01-25"

    def test_source_ids_consistent_across_epochs(self, tmp_path):
        p1 = _write_epoch_csv(tmp_path, EPOCH1, [0.1, 0.2, 0.3])
        p2 = _write_epoch_csv(tmp_path, EPOCH2, [0.11, 0.19, 0.28])
        df = stack_csvs([p1, p2])
        # Source "J0001+2000" should always have the same source_id
        ids = df[df["source_name"] == "J0001+2000"]["source_id"].unique()
        assert len(ids) == 1


# ===========================================================================
# compute_source_metrics
# ===========================================================================

class TestComputeSourceMetrics:
    def _fluxes_errors(self, fluxes, frac_err=0.05):
        f = np.array(fluxes, dtype=float)
        e = np.abs(f) * frac_err
        e[e == 0] = 1e-6   # avoid zero error
        return f, e

    def test_constant_source_m_zero(self):
        f, e = self._fluxes_errors([1.0, 1.0, 1.0, 1.0])
        m, Vs, eta = compute_source_metrics(f, e)
        assert m == pytest.approx(0.0, abs=1e-10)

    def test_constant_source_Vs_zero(self):
        f, e = self._fluxes_errors([1.0, 1.0, 1.0, 1.0])
        m, Vs, eta = compute_source_metrics(f, e)
        assert Vs == pytest.approx(0.0, abs=1e-10)

    def test_constant_source_eta_near_zero(self):
        f, e = self._fluxes_errors([1.0, 1.0, 1.0, 1.0])
        m, Vs, eta = compute_source_metrics(f, e)
        assert eta == pytest.approx(0.0, abs=1e-10)

    def test_single_epoch_all_nan(self):
        f = np.array([0.5])
        e = np.array([0.025])
        m, Vs, eta = compute_source_metrics(f, e)
        assert np.isnan(m)
        assert np.isnan(Vs)
        assert np.isnan(eta)

    def test_variable_source_m_positive(self):
        f = np.array([0.1, 0.5, 0.3, 0.8])
        e = np.array([0.01, 0.02, 0.015, 0.03])
        m, Vs, eta = compute_source_metrics(f, e)
        assert m > 0

    def test_variable_source_Vs_positive(self):
        f = np.array([0.1, 0.5, 0.3, 0.8])
        e = np.array([0.01, 0.02, 0.015, 0.03])
        m, Vs, eta = compute_source_metrics(f, e)
        assert Vs > 0

    def test_variable_source_eta_positive(self):
        f = np.array([0.1, 0.5, 0.3, 0.8])
        e = np.array([0.01, 0.02, 0.015, 0.03])
        m, Vs, eta = compute_source_metrics(f, e)
        assert eta > 0

    def test_highly_variable_Vs_exceeds_threshold(self):
        # Max=1.0, min=0.1, errors=0.01 → Vs ≈ 63.6
        f = np.array([0.1, 1.0, 0.5, 0.3])
        e = np.full(4, 0.01)
        m, Vs, eta = compute_source_metrics(f, e)
        assert Vs > VS_THRESHOLD

    def test_nan_flux_skipped(self):
        f = np.array([0.3, np.nan, 0.35, 0.32])
        e = np.array([0.015, 0.015, 0.0175, 0.016])
        m, Vs, eta = compute_source_metrics(f, e)
        # Should succeed with 3 valid points
        assert np.isfinite(m)
        assert np.isfinite(Vs)
        assert np.isfinite(eta)

    def test_zero_error_rows_excluded(self):
        f = np.array([0.3, 0.35, 0.32])
        e = np.array([0.015, 0.0, 0.016])   # one zero error
        m, Vs, eta = compute_source_metrics(f, e)
        # Should succeed with 2 valid rows
        assert np.isfinite(m)

    def test_m_formula(self):
        """m = std / mean — verify against manual calculation."""
        f = np.array([0.8, 1.0, 1.2])
        e = np.array([0.04, 0.05, 0.06])
        m, _, _ = compute_source_metrics(f, e)
        expected_m = float(np.std(f, ddof=1) / np.mean(f))
        assert m == pytest.approx(expected_m, rel=1e-6)

    def test_Vs_formula(self):
        """Vs = (max - min) / sqrt(e_max² + e_min²)."""
        f = np.array([0.5, 1.5, 0.8])
        e = np.array([0.05, 0.15, 0.08])
        _, Vs, _ = compute_source_metrics(f, e)
        idx_max = int(np.argmax(f))  # 1
        idx_min = int(np.argmin(f))  # 0
        expected_Vs = (f[idx_max] - f[idx_min]) / np.hypot(e[idx_max], e[idx_min])
        assert Vs == pytest.approx(expected_Vs, rel=1e-6)

    def test_eta_formula(self):
        """η = Σ[(S_i - <S>_w)² / σ_i²] / (N-1)."""
        f = np.array([0.5, 0.9, 0.7])
        e = np.array([0.05, 0.09, 0.07])
        _, _, eta = compute_source_metrics(f, e)
        w = 1.0 / e ** 2
        mean_w = np.average(f, weights=w)
        chi2 = np.sum(((f - mean_w) / e) ** 2)
        expected_eta = chi2 / (len(f) - 1)
        assert eta == pytest.approx(expected_eta, rel=1e-6)


# ===========================================================================
# compute_metrics (full DataFrame pipeline)
# ===========================================================================

def _make_stacked_lc(
    n_sources: int = 5,
    n_epochs: int = 8,
    variable_ids: list[int] | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Create a synthetic stacked light curve DataFrame."""
    rng = np.random.default_rng(seed)
    rows = []
    for sid in range(n_sources):
        base_flux = rng.uniform(0.1, 1.0)
        for eid in range(n_epochs):
            if variable_ids and sid in variable_ids:
                # Vary by 50%
                flux = base_flux * rng.uniform(0.5, 1.5)
            else:
                flux = base_flux * rng.normal(1.0, 0.02)
            err = base_flux * 0.05
            rows.append({
                "source_id": sid,
                "ra_deg": 10.0 + sid,
                "dec_deg": 20.0 + sid,
                "measured_flux_jy": max(flux, 0.001),
                "flux_err_jy": max(err, 1e-5),
                "catalog_flux_jy": base_flux,
            })
    return pd.DataFrame(rows)


class TestComputeMetrics:
    def test_returns_dataframe(self):
        lc = _make_stacked_lc(n_sources=3, n_epochs=5)
        result = compute_metrics(lc)
        assert isinstance(result, pd.DataFrame)

    def test_one_row_per_source(self):
        n = 4
        lc = _make_stacked_lc(n_sources=n, n_epochs=6)
        result = compute_metrics(lc)
        assert len(result) == n

    def test_has_required_columns(self):
        lc = _make_stacked_lc()
        result = compute_metrics(lc)
        for col in ["ra_deg", "dec_deg", "n_epochs", "mean_flux", "std_flux",
                    "m", "Vs", "eta", "is_variable_candidate"]:
            assert col in result.columns, f"Missing column: {col}"

    def test_n_epochs_correct(self):
        lc = _make_stacked_lc(n_sources=3, n_epochs=7)
        result = compute_metrics(lc)
        assert (result["n_epochs"] == 7).all()

    def test_variable_flagged(self):
        lc = _make_stacked_lc(n_sources=5, n_epochs=10, variable_ids=[2])
        result = compute_metrics(lc)
        # Source 2 should be a candidate
        assert result.loc[2, "is_variable_candidate"]

    def test_steady_not_flagged(self):
        # Synthetic steady sources with tiny noise
        rng = np.random.default_rng(0)
        rows = []
        for sid in range(3):
            base = 0.5
            for eid in range(10):
                flux = base + rng.normal(0, 0.001)
                rows.append({
                    "source_id": sid,
                    "ra_deg": 10.0 + sid,
                    "dec_deg": 20.0 + sid,
                    "measured_flux_jy": flux,
                    "flux_err_jy": 0.025,  # large error → small Vs, small η
                })
        lc = pd.DataFrame(rows)
        result = compute_metrics(lc)
        assert not result["is_variable_candidate"].any()

    def test_missing_column_raises(self):
        lc = _make_stacked_lc()
        lc = lc.drop(columns=["flux_err_jy"])
        with pytest.raises(KeyError):
            compute_metrics(lc)

    def test_empty_dataframe_raises(self):
        lc = pd.DataFrame(columns=["source_id", "ra_deg", "dec_deg",
                                    "measured_flux_jy", "flux_err_jy"])
        with pytest.raises(ValueError, match="empty"):
            compute_metrics(lc)

    def test_catalog_flux_preserved(self):
        lc = _make_stacked_lc()
        result = compute_metrics(lc)
        assert "catalog_flux_jy" in result.columns

    def test_single_epoch_all_nan_metrics(self):
        lc = pd.DataFrame({
            "source_id": [0],
            "ra_deg": [10.0],
            "dec_deg": [20.0],
            "measured_flux_jy": [0.5],
            "flux_err_jy": [0.025],
        })
        result = compute_metrics(lc)
        assert np.isnan(result.loc[0, "m"])
        assert np.isnan(result.loc[0, "Vs"])
        assert np.isnan(result.loc[0, "eta"])

    def test_custom_flux_column(self):
        lc = _make_stacked_lc()
        lc = lc.rename(columns={"measured_flux_jy": "flux_jy"})
        # Should raise because default col name is wrong
        with pytest.raises(KeyError):
            compute_metrics(lc)
        # Should work with correct col name
        result = compute_metrics(lc, flux_col="flux_jy")
        assert len(result) > 0


# ===========================================================================
# flag_candidates
# ===========================================================================

class TestFlagCandidates:
    def _make_metrics_df(self) -> pd.DataFrame:
        return pd.DataFrame({
            "source_id": [0, 1, 2, 3],
            "Vs": [1.0, 5.0, np.nan, 2.0],
            "eta": [1.0, 1.0, 3.0, np.nan],
        }).set_index("source_id")

    def test_vs_flag(self):
        df = self._make_metrics_df()
        result = flag_candidates(df)
        assert result.loc[1, "is_variable_candidate"]   # Vs=5 > 4

    def test_eta_flag(self):
        df = self._make_metrics_df()
        result = flag_candidates(df)
        assert result.loc[2, "is_variable_candidate"]   # eta=3 > 2.5

    def test_steady_not_flagged(self):
        df = self._make_metrics_df()
        result = flag_candidates(df)
        assert not result.loc[0, "is_variable_candidate"]

    def test_nan_vs_not_flagged(self):
        df = self._make_metrics_df()
        result = flag_candidates(df)
        # nan Vs and nan eta → not flagged
        assert not result.loc[3, "is_variable_candidate"]   # eta is nan, Vs=2 < 4

    def test_custom_thresholds(self):
        df = self._make_metrics_df()
        result = flag_candidates(df, vs_threshold=10.0, eta_threshold=10.0)
        assert not result["is_variable_candidate"].any()

    def test_returns_copy(self):
        df = self._make_metrics_df()
        result = flag_candidates(df)
        assert "is_variable_candidate" not in df.columns


# ===========================================================================
# variability_summary
# ===========================================================================

class TestVariabilitySummary:
    def test_basic_keys(self):
        lc = _make_stacked_lc()
        metrics = compute_metrics(lc)
        summary = variability_summary(metrics)
        for k in ["n_sources", "n_candidates", "fraction_variable",
                   "median_m", "median_Vs", "median_eta"]:
            assert k in summary

    def test_n_sources(self):
        lc = _make_stacked_lc(n_sources=6)
        metrics = compute_metrics(lc)
        summary = variability_summary(metrics)
        assert summary["n_sources"] == 6

    def test_fraction_between_0_and_1(self):
        lc = _make_stacked_lc()
        metrics = compute_metrics(lc)
        summary = variability_summary(metrics)
        assert 0.0 <= summary["fraction_variable"] <= 1.0


# ===========================================================================
# VariabilityMetrics dataclass
# ===========================================================================

class TestVariabilityMetricsDataclass:
    def test_construction(self):
        vm = VariabilityMetrics(
            source_id=42,
            ra_deg=150.5,
            dec_deg=25.3,
            n_epochs=10,
            mean_flux=0.5,
            std_flux=0.05,
            m=0.1,
            Vs=2.5,
            eta=1.8,
            is_variable_candidate=False,
        )
        assert vm.source_id == 42
        assert vm.is_variable_candidate is False
        assert vm.catalog_flux_jy is None

    def test_optional_fields(self):
        vm = VariabilityMetrics(
            source_id=1,
            ra_deg=10.0,
            dec_deg=20.0,
            n_epochs=5,
            mean_flux=0.3,
            std_flux=0.01,
            m=0.03,
            Vs=1.0,
            eta=0.9,
            is_variable_candidate=False,
            catalog_flux_jy=0.29,
            spectral_index=-0.7,
        )
        assert vm.catalog_flux_jy == pytest.approx(0.29)
        assert vm.spectral_index == pytest.approx(-0.7)


# ===========================================================================
# Module-level imports
# ===========================================================================

class TestModuleImports:
    def test_package_imports(self):
        from dsa110_continuum import lightcurves  # noqa: F401

    def test_all_symbols_importable(self):
        from dsa110_continuum.lightcurves import (
            stack_csvs,
            assign_source_ids,
            parse_epoch_utc,
            compute_metrics,
            flag_candidates,
            VariabilityMetrics,
        )
        assert callable(stack_csvs)
        assert callable(assign_source_ids)
        assert callable(parse_epoch_utc)
        assert callable(compute_metrics)
        assert callable(flag_candidates)
        assert VariabilityMetrics is not None
