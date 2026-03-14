import pandas as pd
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from stack_lightcurves import (
    parse_epoch_utc,
    assign_source_ids,
    stack_csvs,
)


def test_parse_epoch_utc():
    assert parse_epoch_utc("2026-02-12T0000_forced_phot.csv") == "2026-02-12T00:00:00"
    assert parse_epoch_utc("2026-01-25T2200_forced_phot.csv") == "2026-01-25T22:00:00"


def test_assign_source_ids_groups_nearby():
    df = pd.DataFrame({
        "ra_deg": [10.001, 10.001, 20.000],
        "dec_deg": [5.000, 5.000, 15.000],
        "catalog_flux_jy": [1.0, 1.0, 2.0],
    })
    result = assign_source_ids(df, match_arcsec=5.0)
    assert result.loc[0, "source_id"] == result.loc[1, "source_id"]
    assert result.loc[0, "source_id"] != result.loc[2, "source_id"]


def test_assign_source_ids_uses_source_name():
    df = pd.DataFrame({
        "ra_deg": [10.0, 10.0, 20.0],
        "dec_deg": [5.0, 5.0, 15.0],
        "source_name": ["J10.0000+5.0000", "J10.0000+5.0000", "J20.0000+15.0000"],
        "catalog_flux_jy": [1.0, 1.0, 2.0],
    })
    result = assign_source_ids(df)
    assert result.loc[0, "source_id"] == result.loc[1, "source_id"]
    assert result.loc[0, "source_id"] != result.loc[2, "source_id"]


def test_stack_csvs_produces_required_columns(tmp_path):
    csv1 = tmp_path / "2026-01-25T0200_forced_phot.csv"
    csv2 = tmp_path / "2026-02-12T0000_forced_phot.csv"
    rows = "source_name,ra_deg,dec_deg,catalog_flux_jy,measured_flux_jy,flux_err_jy,flux_ratio\n"
    rows += "J10.0000+5.0000,10.0,5.0,1.0,0.9,0.01,0.9\n"
    csv1.write_text(rows)
    csv2.write_text(rows)
    df = stack_csvs([str(csv1), str(csv2)])
    for col in ["source_id", "epoch_utc", "date", "measured_flux_jy", "flux_err_jy"]:
        assert col in df.columns, f"Missing column: {col}"
    assert len(df) == 2
    assert df["source_id"].iloc[0] == df["source_id"].iloc[1]
