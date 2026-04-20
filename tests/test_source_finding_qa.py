"""Tests for source-finding QA: completeness and size distribution checks."""
import os
import tempfile
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from dsa110_continuum.source_finding.core import SourceCatalogEntry


def _make_entries(n: int = 5, a_arcsec: float = 36.9, b_arcsec: float = 25.5,
                  snr: float = 10.0) -> list[SourceCatalogEntry]:
    rms = 0.001
    return [
        SourceCatalogEntry(
            source_name=f"J{i:04d}",
            ra_deg=344.0 + i * 0.05,
            dec_deg=16.15,
            peak_flux_jy=rms * snr,
            peak_flux_err_jy=rms * 0.5,
            int_flux_jy=rms * snr * 1.1,
            a_arcsec=a_arcsec,
            b_arcsec=b_arcsec,
            pa_deg=130.75,
            local_rms_jy=rms,
        )
        for i in range(n)
    ]


def test_check_source_completeness_all_recovered():
    """All NVSS sources recovered → completeness=1.0, gate=PASS."""
    from dsa110_continuum.source_finding.core import check_source_completeness, CompletenessResult

    catalog = _make_entries(n=5)
    # Mock cone_search to return the same RA/Dec positions → all matched
    nvss_df = pd.DataFrame({
        "ra_deg": [344.0 + i * 0.05 for i in range(5)],
        "dec_deg": [16.15] * 5,
        "flux_mjy": [100.0] * 5,
    })
    with patch("dsa110_continuum.source_finding.core._cone_search_nvss", return_value=nvss_df):
        result = check_source_completeness(
            catalog, ra_center=344.1, dec_center=16.15, radius_deg=1.0
        )
    assert isinstance(result, CompletenessResult)
    assert result.completeness_frac == pytest.approx(1.0)
    assert result.gate == "PASS"


def test_check_source_completeness_fail():
    """Only 2/5 NVSS sources recovered → gate=WARN or FAIL (<60% threshold)."""
    from dsa110_continuum.source_finding.core import check_source_completeness

    # Catalog has only 2 sources; NVSS has 5 → 40% recovered
    catalog = _make_entries(n=2)
    nvss_df = pd.DataFrame({
        "ra_deg": [344.0 + i * 0.05 for i in range(5)],
        "dec_deg": [16.15] * 5,
        "flux_mjy": [100.0] * 5,
    })
    with patch("dsa110_continuum.source_finding.core._cone_search_nvss", return_value=nvss_df):
        result = check_source_completeness(
            catalog, ra_center=344.1, dec_center=16.15, radius_deg=1.0
        )
    assert result.n_nvss_reference == 5
    assert result.n_recovered == 2
    # Gate: WARN at 40-60%, FAIL below 40%
    assert result.gate in ("WARN", "FAIL")


def test_check_size_distribution_normal():
    """Normal-size sources → frac_subbeam and frac_elongated near 0, PASS gate."""
    from dsa110_continuum.source_finding.core import check_size_distribution

    catalog = _make_entries(n=10, a_arcsec=36.9, b_arcsec=25.5)
    result = check_size_distribution(catalog, beam_a_arcsec=36.9, beam_b_arcsec=25.5)
    assert result.frac_subbeam < 0.05
    assert result.frac_elongated < 0.05
    assert result.gate == "PASS"


def test_check_size_distribution_warns_subbeam():
    """Many sub-beam sources → gate=WARN (frac_subbeam > 5%)."""
    from dsa110_continuum.source_finding.core import check_size_distribution

    # Create sources with a_arcsec = 25 (< 0.9 × 36.9 = 33.2) → subbeam
    catalog = _make_entries(n=10, a_arcsec=25.0, b_arcsec=20.0)
    result = check_size_distribution(catalog, beam_a_arcsec=36.9, beam_b_arcsec=25.5)
    assert result.frac_subbeam > 0.05
    assert result.gate == "WARN"


def test_completeness_empty_nvss():
    """Empty NVSS cone result → gate=WARN (no reference sources)."""
    from dsa110_continuum.source_finding.core import check_source_completeness

    catalog = _make_entries(n=5)
    empty_df = pd.DataFrame(columns=["ra_deg", "dec_deg", "flux_mjy"])
    with patch("dsa110_continuum.source_finding.core._cone_search_nvss", return_value=empty_df):
        result = check_source_completeness(
            catalog, ra_center=344.1, dec_center=16.15, radius_deg=1.0
        )
    assert result.n_nvss_reference == 0
    assert result.gate == "WARN"  # no reference sources → WARN, not FAIL
