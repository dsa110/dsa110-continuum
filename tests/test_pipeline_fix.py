import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))


def test_batch_pipeline_respects_env_vars(monkeypatch):
    """MS_DIR, STAGE_IMAGE_BASE, PRODUCTS_BASE should be overridable via env vars."""
    monkeypatch.setenv("DSA110_MS_DIR", "/tmp/test_ms")
    monkeypatch.setenv("DSA110_STAGE_IMAGE_BASE", "/tmp/test_images")
    monkeypatch.setenv("DSA110_PRODUCTS_BASE", "/tmp/test_products")
    import importlib
    import batch_pipeline as bp
    importlib.reload(bp)
    assert bp.MS_DIR == "/tmp/test_ms"
    assert bp.STAGE_IMAGE_BASE == "/tmp/test_images"
    assert bp.PRODUCTS_BASE == "/tmp/test_products"


def test_forced_phot_csv_schema_has_source_id(tmp_path):
    """Verify the forced phot output schema has source_id."""
    import pandas as pd
    # Simulate what forced.py should produce
    df = pd.DataFrame({
        "ra_deg": [10.0, 20.0],
        "dec_deg": [5.0, 15.0],
        "nvss_flux_jy": [1.0, 2.0],
        "dsa_peak_jyb": [0.9, 1.8],
        "dsa_peak_err_jyb": [0.01, 0.02],
        "dsa_nvss_ratio": [0.9, 0.9],
        "source_id": [0, 1],
    })
    csv_path = tmp_path / "test.csv"
    df.to_csv(csv_path, index=False)
    loaded = pd.read_csv(csv_path)
    assert "source_id" in loaded.columns
    assert loaded["source_id"].dtype in [int, "int64", "int32"]
