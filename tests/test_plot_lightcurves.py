import pandas as pd
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from plot_lightcurves import (
    plot_source_lightcurve,
    build_summary_html,
)


def make_lc():
    return pd.DataFrame({
        "source_id": [0, 0, 1],
        "ra_deg": [10.0, 10.0, 20.0],
        "dec_deg": [5.0, 5.0, 15.0],
        "catalog_flux_jy": [1.0, 1.0, 2.0],
        "epoch_utc": ["2026-01-25T02:00:00", "2026-02-12T00:00:00", "2026-01-25T02:00:00"],
        "measured_flux_jy": [0.9, 0.95, 1.8],
        "flux_err_jy": [0.01, 0.01, 0.02],
        "flux_ratio": [0.9, 0.95, 0.9],
        "date": ["2026-01-25", "2026-02-12", "2026-01-25"],
    })


def make_metrics():
    return pd.DataFrame({
        "source_id": [0, 1],
        "ra_deg": [10.0, 20.0],
        "dec_deg": [5.0, 15.0],
        "catalog_flux_jy": [1.0, 2.0],
        "n_epochs": [2, 1],
        "mean_flux": [0.925, 1.8],
        "std_flux": [0.035, np.nan],
        "m": [0.038, np.nan],
        "Vs": [3.5, np.nan],
        "eta": [6.1, np.nan],
        "is_variable_candidate": [True, False],
    })


def test_plot_source_lightcurve_creates_file(tmp_path):
    lc = make_lc()
    source_group = lc[lc["source_id"] == 0]
    out_path = tmp_path / "000000.png"
    plot_source_lightcurve(source_group, nvss_flux=1.0, out_path=str(out_path))
    assert out_path.exists()
    assert out_path.stat().st_size > 1000


def test_build_summary_html_contains_candidates(tmp_path):
    lc = make_lc()
    metrics = make_metrics()
    plots_dir = tmp_path / "plots"
    plots_dir.mkdir()
    # Create a minimal valid PNG (just header bytes)
    png_bytes = (
        b'\x89PNG\r\n\x1a\n'  # PNG magic
        b'\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02'
        b'\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\xf8\x0f\x00'
        b'\x00\x01\x01\x00\x05\x18\xd8N\x00\x00\x00\x00IEND\xaeB`\x82'
    )
    (plots_dir / "000000.png").write_bytes(png_bytes)
    html = build_summary_html(metrics, lc, plots_dir=str(plots_dir))
    assert "<table" in html
    assert "000000" in html or "source_id" in html.lower()
    assert "eta" in html.lower() or "&eta;" in html
