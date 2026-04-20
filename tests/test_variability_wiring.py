"""Tests for variability metric computation and candidate flagging."""
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def _make_lightcurve_df(n_sources: int = 5, n_epochs: int = 4,
                         variable_idx: int | None = 0) -> pd.DataFrame:
    """Build a synthetic stacked light-curve DataFrame."""
    rows = []
    rng = np.random.default_rng(42)
    for sid in range(n_sources):
        for epoch in range(n_epochs):
            flux = 0.1 + sid * 0.02
            err = 0.003
            if sid == variable_idx:
                flux += rng.uniform(-0.05, 0.05)  # add variability
            rows.append({
                "source_id": sid,
                "ra_deg": 344.0 + sid * 0.05,
                "dec_deg": 16.15,
                "epoch_utc": f"2026-01-{25 + epoch:02d}T22:26:05",
                "measured_flux_jy": max(flux, 0.001),
                "flux_err_jy": err,
                "catalog_flux_jy": 0.1 + sid * 0.02,
            })
    return pd.DataFrame(rows)


def test_compute_metrics_produces_dataframe():
    """compute_metrics returns a DataFrame with required columns."""
    from dsa110_continuum.lightcurves.metrics import compute_metrics
    lc = _make_lightcurve_df(n_sources=5, n_epochs=4)
    metrics = compute_metrics(lc)
    for col in ("Vs", "eta", "m", "n_epochs", "mean_flux", "is_variable_candidate"):
        assert col in metrics.columns, f"Missing column: {col}"
    assert len(metrics) == 5


def test_variability_summary_dict_keys():
    """variability_summary returns all expected keys."""
    from dsa110_continuum.lightcurves.metrics import compute_metrics, variability_summary
    lc = _make_lightcurve_df(n_sources=5, n_epochs=4)
    metrics = compute_metrics(lc)
    summary = variability_summary(metrics)
    for key in ("n_sources", "n_candidates", "fraction_variable", "median_Vs", "median_eta"):
        assert key in summary, f"Missing key: {key}"


def test_variability_csv_written():
    """Variability summary CSV is written to out_dir."""
    from dsa110_continuum.lightcurves.metrics import compute_metrics
    from dsa110_continuum.lightcurves.variability_output import write_variability_summary

    lc = _make_lightcurve_df(n_sources=5, n_epochs=4)
    metrics = compute_metrics(lc)
    with tempfile.TemporaryDirectory() as d:
        out_path = write_variability_summary(metrics, out_dir=d)
        assert Path(out_path).exists()
        df = pd.read_csv(out_path)
        assert "eta" in df.columns
        assert "is_variable_candidate" in df.columns
        assert len(df) == 5


def test_candidates_flagged_correctly():
    """Sources with high variability are flagged as candidates."""
    from dsa110_continuum.lightcurves.metrics import compute_metrics, flag_candidates

    # Construct a clearly variable source: large spread relative to errors
    rows = []
    for epoch, flux in enumerate([0.2, 0.05, 0.18, 0.04]):
        rows.append({
            "source_id": 0, "ra_deg": 344.0, "dec_deg": 16.15,
            "measured_flux_jy": flux, "flux_err_jy": 0.003,
            "catalog_flux_jy": 0.1,
            "epoch_utc": f"2026-01-{25+epoch:02d}T22:26:05",
        })
    lc = pd.DataFrame(rows)
    metrics = compute_metrics(lc)
    assert metrics["is_variable_candidate"].iloc[0], "Highly variable source should be flagged"
