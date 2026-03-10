"""Tests: VLASS included in make_unified_skymodel()."""
import pandas as pd
from unittest.mock import patch


def _make_df(ra, dec, flux):
    return pd.DataFrame({"ra_deg": [ra], "dec_deg": [dec], "flux_mjy": [flux]})


def test_make_unified_skymodel_queries_vlass():
    """make_unified_skymodel should query VLASS and include unique VLASS sources."""
    from dsa110_continuum.calibration.skymodels import make_unified_skymodel

    calls = []

    def fake_query(catalog_type, ra_center, dec_center, radius_deg, min_flux_mjy):
        calls.append(catalog_type)
        if catalog_type == "vlass":
            return _make_df(10.5, 30.0, 200.0)   # unique VLASS-only source
        return pd.DataFrame(columns=["ra_deg", "dec_deg", "flux_mjy"])

    with patch(
        "dsa110_continuum.catalog.query.query_sources",
        side_effect=fake_query,
    ):
        sky = make_unified_skymodel(10.0, 30.0, 1.0, min_mjy=50.0)

    assert "vlass" in calls, "VLASS catalog was not queried"
    assert sky.Ncomponents >= 1, "VLASS-only source should appear in unified model"


def test_make_unified_skymodel_vlass_deduplication():
    """VLASS sources that match NVSS should be suppressed."""
    from dsa110_continuum.calibration.skymodels import make_unified_skymodel

    def fake_query(catalog_type, ra_center, dec_center, radius_deg, min_flux_mjy):
        if catalog_type == "nvss":
            return _make_df(10.0, 30.0, 500.0)
        if catalog_type == "vlass":
            return _make_df(10.0, 30.0, 480.0)  # same position → should be deduplicated
        return pd.DataFrame(columns=["ra_deg", "dec_deg", "flux_mjy"])

    with patch(
        "dsa110_continuum.catalog.query.query_sources",
        side_effect=fake_query,
    ):
        sky = make_unified_skymodel(10.0, 30.0, 1.0, min_mjy=50.0)

    assert sky.Ncomponents == 1, "Duplicate VLASS/NVSS source should be deduplicated"
