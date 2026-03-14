import pandas as pd
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from variability_metrics import compute_metrics, flag_candidates


def make_source_df(fluxes, errors):
    n = len(fluxes)
    return pd.DataFrame({
        "source_id": [0] * n,
        "ra_deg": [10.0] * n,
        "dec_deg": [5.0] * n,
        "catalog_flux_jy": [1.0] * n,
        "epoch_utc": [f"2026-01-0{i+1}T00:00:00" for i in range(n)],
        "measured_flux_jy": fluxes,
        "flux_err_jy": errors,
        "flux_ratio": [f / 1.0 for f in fluxes],
        "date": [f"2026-01-0{i+1}" for i in range(n)],
    })


def test_constant_source_has_low_eta():
    df = make_source_df([1.0, 1.0, 1.0], [0.1, 0.1, 0.1])
    metrics = compute_metrics(df)
    assert metrics.loc[0, "eta"] < 0.1


def test_variable_source_has_high_vs():
    df = make_source_df([1.0, 2.0], [0.05, 0.05])
    metrics = compute_metrics(df)
    assert metrics.loc[0, "Vs"] > 4.0


def test_single_epoch_source_gets_nan():
    df = make_source_df([1.0], [0.1])
    metrics = compute_metrics(df)
    assert np.isnan(metrics.loc[0, "eta"])
    assert np.isnan(metrics.loc[0, "Vs"])


def test_flag_candidates():
    metrics = pd.DataFrame({
        "source_id": [0, 1, 2],
        "Vs": [5.0, 1.0, 2.0],
        "eta": [1.0, 3.0, 1.0],
    }).set_index("source_id")
    flagged = flag_candidates(metrics)
    assert flagged.loc[0, "is_variable_candidate"]
    assert flagged.loc[1, "is_variable_candidate"]
    assert not flagged.loc[2, "is_variable_candidate"]
