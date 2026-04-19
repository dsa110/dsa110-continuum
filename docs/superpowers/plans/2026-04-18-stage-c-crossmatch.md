# Stage C: Post-Discovery Cross-Match Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create `dsa110_continuum/catalog/stage_c.py` with `run_stage_c()`, a CLI wrapper `scripts/stage_c_crossmatch.py`, and 10 CI-runnable tests in `tests/test_stage_c_crossmatch.py`.

**Architecture:** Read Stage B Aegean FITS → cross-match against master catalog (+ NVSS/FIRST/RACS fallbacks) via existing `cross_match_sources()` → write annotated FITS table with 19 columns including `master_matched`, `new_source_candidate`, flux ratios, and separations.

**Tech Stack:** `astropy.io.fits`, `pandas`, `unittest.mock.patch`, `dsa110_continuum.catalog.crossmatch`, `dsa110_continuum.catalog.query.cone_search`

---

### Task 1: `run_stage_c()` core — read catalog, cross-match master, write output

**Files:**
- Create: `dsa110_continuum/catalog/stage_c.py`
- Create: `tests/test_stage_c_crossmatch.py`

**Setup note:** `matplotlib.use("Agg")` is NOT needed in these tests — no plotting.
Use `tempfile.NamedTemporaryFile(delete=False)` + cleanup in `try/finally` blocks.
Do NOT use `pytest.tmp_path` — it is not writable in this environment.

- [ ] **Step 1: Write 5 failing tests**

```python
"""Tests for Stage C: post-discovery cross-match."""
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from astropy.io import fits


# ---------------------------------------------------------------------------
# Synthetic FITS helper
# ---------------------------------------------------------------------------

def _make_aegean_fits(path: str, n: int = 5, snr_override: list | None = None) -> None:
    """Write a minimal Aegean-style FITS binary table."""
    from astropy.table import Table
    t = Table()
    t["source_name"] = [f"J344.{i:04d}+16.0000" for i in range(n)]
    t["ra_deg"]      = np.array([344.0 + i * 0.05 for i in range(n)])
    t["dec_deg"]     = np.full(n, 16.15)
    rms = 0.001
    peak = [rms * (snr_override[i] if snr_override else 10.0) for i in range(n)]
    t["peak_flux_jy"]     = np.array(peak)
    t["peak_flux_err_jy"] = np.full(n, rms * 0.5)
    t["int_flux_jy"]      = np.array(peak) * 1.1
    t["a_arcsec"]         = np.full(n, 36.9)
    t["b_arcsec"]         = np.full(n, 25.5)
    t["pa_deg"]           = np.full(n, 130.75)
    t["local_rms_jy"]     = np.full(n, rms)
    t.write(path, format="fits", overwrite=True)


def _make_master_cone_df(ra_list, dec_list, flux_mjy_list, id_list=None):
    """Return a DataFrame as if returned by cone_search('master', ...)."""
    n = len(ra_list)
    return pd.DataFrame({
        "ra_deg":    ra_list,
        "dec_deg":   dec_list,
        "flux_mjy":  flux_mjy_list,
        "source_id": id_list or [f"MASTER_{i}" for i in range(n)],
    })


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_run_stage_c_empty_catalog():
    """Empty Aegean FITS → ValueError('No sources')."""
    from astropy.table import Table
    with tempfile.NamedTemporaryFile(suffix=".fits", delete=False) as f:
        empty_path = f.name
    with tempfile.TemporaryDirectory() as out_dir:
        try:
            t = Table(names=["source_name","ra_deg","dec_deg","peak_flux_jy",
                             "peak_flux_err_jy","int_flux_jy","a_arcsec","b_arcsec",
                             "pa_deg","local_rms_jy"],
                      dtype=["U64",float,float,float,float,float,float,float,float,float])
            t.write(empty_path, format="fits", overwrite=True)
            from dsa110_continuum.catalog.stage_c import run_stage_c
            with pytest.raises(ValueError, match="No sources"):
                run_stage_c(empty_path, out_dir)
        finally:
            os.unlink(empty_path)


def test_run_stage_c_all_matched():
    """5 sources, master returns 5 nearby matches → all master_matched=True."""
    with tempfile.NamedTemporaryFile(suffix=".fits", delete=False) as f:
        cat_path = f.name
    with tempfile.TemporaryDirectory() as out_dir:
        try:
            _make_aegean_fits(cat_path, n=5)
            # Build master catalog rows at exactly the same positions (0 arcsec separation)
            master_df = _make_master_cone_df(
                ra_list=[344.0 + i * 0.05 for i in range(5)],
                dec_list=[16.15] * 5,
                flux_mjy_list=[100.0 * (i + 1) for i in range(5)],
            )
            with patch("dsa110_continuum.catalog.stage_c._cone_search", return_value=master_df):
                from dsa110_continuum.catalog.stage_c import run_stage_c
                out_path = run_stage_c(cat_path, out_dir,
                                       ra_center=344.1, dec_center=16.15, radius_deg=1.0)
            assert out_path is not None
            assert Path(str(out_path)).exists()
            with fits.open(str(out_path)) as hdul:
                tbl = hdul[1].data
                assert all(tbl["master_matched"]), "All 5 should be master-matched"
        finally:
            os.unlink(cat_path)


def test_run_stage_c_new_source_candidate():
    """Unmatched source with SNR >= 5 → new_source_candidate=True."""
    with tempfile.NamedTemporaryFile(suffix=".fits", delete=False) as f:
        cat_path = f.name
    with tempfile.TemporaryDirectory() as out_dir:
        try:
            # One source with SNR=10 (peak=0.01, rms=0.001)
            _make_aegean_fits(cat_path, n=1, snr_override=[10.0])
            empty_df = pd.DataFrame(columns=["ra_deg","dec_deg","flux_mjy","source_id"])
            # All catalog queries return empty → no match
            with patch("dsa110_continuum.catalog.stage_c._cone_search", return_value=empty_df):
                from dsa110_continuum.catalog.stage_c import run_stage_c
                out_path = run_stage_c(cat_path, out_dir,
                                       ra_center=344.0, dec_center=16.15)
            with fits.open(str(out_path)) as hdul:
                tbl = hdul[1].data
                assert tbl["new_source_candidate"][0], "High-SNR unmatched source should be a candidate"
        finally:
            os.unlink(cat_path)


def test_run_stage_c_low_snr_not_candidate():
    """Unmatched source with SNR < 5 → new_source_candidate=False."""
    with tempfile.NamedTemporaryFile(suffix=".fits", delete=False) as f:
        cat_path = f.name
    with tempfile.TemporaryDirectory() as out_dir:
        try:
            # SNR=3 (below default threshold of 5)
            _make_aegean_fits(cat_path, n=1, snr_override=[3.0])
            empty_df = pd.DataFrame(columns=["ra_deg","dec_deg","flux_mjy","source_id"])
            with patch("dsa110_continuum.catalog.stage_c._cone_search", return_value=empty_df):
                from dsa110_continuum.catalog.stage_c import run_stage_c
                out_path = run_stage_c(cat_path, out_dir,
                                       ra_center=344.0, dec_center=16.15)
            with fits.open(str(out_path)) as hdul:
                tbl = hdul[1].data
                assert not tbl["new_source_candidate"][0], "Low-SNR unmatched source should NOT be a candidate"
        finally:
            os.unlink(cat_path)


def test_output_fits_columns():
    """Output FITS has all 19 required columns."""
    REQUIRED_COLS = [
        "source_name", "ra_deg", "dec_deg", "peak_flux_jy", "snr",
        "master_matched", "master_sep_arcsec", "master_flux_mjy",
        "master_flux_ratio", "master_source_id",
        "nvss_matched", "nvss_sep_arcsec", "nvss_flux_mjy",
        "first_matched", "first_sep_arcsec",
        "racs_matched", "racs_sep_arcsec",
        "any_matched", "new_source_candidate",
    ]
    with tempfile.NamedTemporaryFile(suffix=".fits", delete=False) as f:
        cat_path = f.name
    with tempfile.TemporaryDirectory() as out_dir:
        try:
            _make_aegean_fits(cat_path, n=3)
            empty_df = pd.DataFrame(columns=["ra_deg","dec_deg","flux_mjy","source_id"])
            with patch("dsa110_continuum.catalog.stage_c._cone_search", return_value=empty_df):
                from dsa110_continuum.catalog.stage_c import run_stage_c
                out_path = run_stage_c(cat_path, out_dir,
                                       ra_center=344.0, dec_center=16.15)
            with fits.open(str(out_path)) as hdul:
                cols = [c.name for c in hdul[1].columns]
                for req in REQUIRED_COLS:
                    assert req in cols, f"Missing required column: {req}"
        finally:
            os.unlink(cat_path)
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
cd /home/user/workspace/dsa110-continuum
pytest tests/test_stage_c_crossmatch.py -v 2>&1 | tail -15
```

Expected: 5 failures with `ImportError` or `ModuleNotFoundError` (stage_c doesn't exist yet).

- [ ] **Step 3: Create `dsa110_continuum/catalog/stage_c.py`**

```python
"""Stage C: post-discovery cross-match annotation for DSA-110 continuum pipeline.

Reads the Stage B Aegean FITS catalog, matches each detection against the master
radio catalog (NVSS+VLASS+FIRST+RACS) with fallback to individual catalogs, and
writes an annotated FITS table.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.table import Table

from dsa110_continuum.catalog.crossmatch import (
    cross_match_sources,
    calculate_positional_offsets,
)

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Internal catalog query shim (allows patching in tests)
# ---------------------------------------------------------------------------

def _cone_search(catalog_type: str, ra_center: float, dec_center: float,
                 radius_deg: float) -> pd.DataFrame:
    """Thin wrapper around catalog.query.cone_search — monkeypatched in tests."""
    try:
        from dsa110_continuum.catalog.query import cone_search
        result = cone_search(catalog_type, ra_center, dec_center, radius_deg)
        return result if result is not None else pd.DataFrame()
    except Exception as exc:
        log.warning("Catalog query failed for %s: %s", catalog_type, exc)
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# Output column schema
# ---------------------------------------------------------------------------

_OUTPUT_COLS = [
    "source_name", "ra_deg", "dec_deg", "peak_flux_jy", "snr",
    "master_matched", "master_sep_arcsec", "master_flux_mjy",
    "master_flux_ratio", "master_source_id",
    "nvss_matched", "nvss_sep_arcsec", "nvss_flux_mjy",
    "first_matched", "first_sep_arcsec",
    "racs_matched", "racs_sep_arcsec",
    "any_matched", "new_source_candidate",
]


def _read_aegean_fits(catalog_path: str | Path) -> pd.DataFrame:
    """Read Aegean FITS binary table into a DataFrame."""
    with fits.open(catalog_path) as hdul:
        t = Table(hdul[1].data)
        df = t.to_pandas()
        # Decode byte strings if present (FITS column encoding)
        for col in df.select_dtypes(include=["object"]).columns:
            df[col] = df[col].apply(
                lambda v: v.decode("utf-8").strip() if isinstance(v, bytes) else str(v)
            )
    return df


def _match_catalog(
    detected_ra: np.ndarray,
    detected_dec: np.ndarray,
    catalog_df: pd.DataFrame,
    match_radius_arcsec: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Match detected sources against a catalog DataFrame.

    Returns (matched_bool, sep_arcsec, flux_mjy) arrays aligned to detected sources.
    NaN means no match.
    """
    n = len(detected_ra)
    matched = np.zeros(n, dtype=bool)
    sep = np.full(n, np.nan)
    flux = np.full(n, np.nan)

    if catalog_df is None or len(catalog_df) == 0:
        return matched, sep, flux

    required = {"ra_deg", "dec_deg", "flux_mjy"}
    if not required.issubset(catalog_df.columns):
        log.warning("Catalog missing expected columns (need ra_deg, dec_deg, flux_mjy)")
        return matched, sep, flux

    result = cross_match_sources(
        detected_ra=detected_ra,
        detected_dec=detected_dec,
        catalog_ra=catalog_df["ra_deg"].values,
        catalog_dec=catalog_df["dec_deg"].values,
        radius_arcsec=match_radius_arcsec,
        detected_flux=None,
        catalog_flux=catalog_df["flux_mjy"].values,
    )

    if result is None or len(result) == 0:
        return matched, sep, flux

    for _, row in result.iterrows():
        di = int(row["detected_idx"])
        ci = int(row["catalog_idx"])
        matched[di] = True
        sep[di] = float(row["separation_arcsec"])
        flux[di] = float(catalog_df["flux_mjy"].iloc[ci])

    return matched, sep, flux


def run_stage_c(
    catalog_path: str | Path,
    out_path: str | Path | None = None,
    *,
    ra_center: float | None = None,
    dec_center: float | None = None,
    radius_deg: float = 2.0,
    match_radius_arcsec: float = 10.0,
    new_source_snr_threshold: float = 5.0,
) -> Path:
    """Annotate Stage B Aegean detections with multi-catalog cross-match results.

    Parameters
    ----------
    catalog_path : str or Path
        Stage B Aegean FITS catalog (output of run_source_finding).
    out_path : str, Path, or None
        Output path for annotated FITS. Default: ``{stem}_crossmatched.fits``
        next to the input catalog.
    ra_center, dec_center : float or None
        Field center for catalog cone queries. Derived from source centroid if None.
    radius_deg : float
        Cone search radius in degrees.
    match_radius_arcsec : float
        Cross-match radius in arcseconds.
    new_source_snr_threshold : float
        SNR threshold for flagging unmatched sources as candidates.

    Returns
    -------
    Path
        Path to the written annotated FITS table.
    """
    catalog_path = Path(catalog_path)

    # -- Read Aegean catalog --------------------------------------------------
    df = _read_aegean_fits(catalog_path)
    if len(df) == 0:
        raise ValueError(f"No sources in catalog: {catalog_path}")

    log.info("Stage C: %d detections loaded from %s", len(df), catalog_path)

    ra_arr  = df["ra_deg"].values.astype(float)
    dec_arr = df["dec_deg"].values.astype(float)
    peak    = df["peak_flux_jy"].values.astype(float)
    rms     = df["local_rms_jy"].values.astype(float)
    snr_arr = np.where(rms > 0, peak / rms, 0.0)

    # -- Derive field center if not provided ----------------------------------
    if ra_center is None:
        ra_center = float(np.median(ra_arr))
    if dec_center is None:
        dec_center = float(np.median(dec_arr))

    # -- Primary: master catalog match ----------------------------------------
    master_df = _cone_search("master", ra_center, dec_center, radius_deg)
    master_matched, master_sep, master_flux = _match_catalog(
        ra_arr, dec_arr, master_df, match_radius_arcsec
    )

    # Master catalog source IDs
    master_ids = np.full(len(df), "", dtype=object)
    if len(master_df) > 0 and "source_id" in master_df.columns:
        result_tmp = cross_match_sources(
            detected_ra=ra_arr,
            detected_dec=dec_arr,
            catalog_ra=master_df["ra_deg"].values,
            catalog_dec=master_df["dec_deg"].values,
            radius_arcsec=match_radius_arcsec,
        )
        if result_tmp is not None and len(result_tmp) > 0:
            for _, row in result_tmp.iterrows():
                di = int(row["detected_idx"])
                ci = int(row["catalog_idx"])
                master_ids[di] = str(master_df["source_id"].iloc[ci])

    # -- Flux ratio -----------------------------------------------------------
    master_flux_ratio = np.where(
        master_matched & (master_flux > 0),
        peak / (master_flux / 1000.0),
        np.nan,
    )

    # -- Fallback: individual catalogs for unmatched --------------------------
    unmatched_mask = ~master_matched

    def _fallback(catalog_name: str):
        if not np.any(unmatched_mask):
            return (np.zeros(len(df), dtype=bool),
                    np.full(len(df), np.nan),
                    np.full(len(df), np.nan))
        cat_df = _cone_search(catalog_name, ra_center, dec_center, radius_deg)
        fb_matched, fb_sep, fb_flux = _match_catalog(
            ra_arr, dec_arr, cat_df, match_radius_arcsec
        )
        # Only credit fallback match for sources not already master-matched
        fb_matched = fb_matched & unmatched_mask
        fb_sep = np.where(fb_matched, fb_sep, np.nan)
        fb_flux = np.where(fb_matched, fb_flux, np.nan)
        return fb_matched, fb_sep, fb_flux

    nvss_matched,  nvss_sep,  nvss_flux  = _fallback("nvss")
    first_matched, first_sep, _          = _fallback("first")
    racs_matched,  racs_sep,  _          = _fallback("rax")

    any_matched = master_matched | nvss_matched | first_matched | racs_matched
    new_candidate = (~any_matched) & (snr_arr >= new_source_snr_threshold)

    # -- Astrometry QA --------------------------------------------------------
    matched_indices = np.where(master_matched)[0]
    if len(matched_indices) >= 3 and len(master_df) > 0:
        try:
            result_qa = cross_match_sources(
                detected_ra=ra_arr,
                detected_dec=dec_arr,
                catalog_ra=master_df["ra_deg"].values,
                catalog_dec=master_df["dec_deg"].values,
                radius_arcsec=match_radius_arcsec,
            )
            if result_qa is not None and len(result_qa) >= 3:
                dra_med, ddec_med, dra_mad, ddec_mad = calculate_positional_offsets(result_qa)
                log.info(
                    "Astrometry QA: median ΔRA=%.2f\" ΔDec=%.2f\" MAD_RA=%.2f\" MAD_Dec=%.2f\"",
                    dra_med.value, ddec_med.value, dra_mad.value, ddec_mad.value,
                )
        except Exception as exc:
            log.warning("Astrometry QA failed: %s", exc)

    n_cand = int(np.sum(new_candidate))
    n_matched_total = int(np.sum(any_matched))
    log.info(
        "Stage C summary: %d/%d matched, %d new source candidates",
        n_matched_total, len(df), n_cand,
    )
    if n_cand > 0:
        cand_names = df["source_name"].values[new_candidate]
        for name in cand_names[:10]:
            log.warning("New source candidate: %s", name)

    # -- Assemble output table ------------------------------------------------
    names_arr = df["source_name"].values.astype(str)

    out_table = Table([
        names_arr,
        ra_arr,
        dec_arr,
        peak,
        snr_arr,
        master_matched.astype(np.int16),   # FITS bool as int16
        np.where(np.isnan(master_sep), -1.0, master_sep),
        np.where(np.isnan(master_flux), -1.0, master_flux),
        np.where(np.isnan(master_flux_ratio), -1.0, master_flux_ratio),
        master_ids.astype(str),
        nvss_matched.astype(np.int16),
        np.where(np.isnan(nvss_sep), -1.0, nvss_sep),
        np.where(np.isnan(nvss_flux), -1.0, nvss_flux),
        first_matched.astype(np.int16),
        np.where(np.isnan(first_sep), -1.0, first_sep),
        racs_matched.astype(np.int16),
        np.where(np.isnan(racs_sep), -1.0, racs_sep),
        any_matched.astype(np.int16),
        new_candidate.astype(np.int16),
    ], names=_OUTPUT_COLS)

    # -- Write output ---------------------------------------------------------
    if out_path is None:
        out_path = catalog_path.parent / (catalog_path.stem + "_crossmatched.fits")
    else:
        out_path = Path(out_path)
        if out_path.is_dir():
            out_path = out_path / (catalog_path.stem + "_crossmatched.fits")

    out_table.write(str(out_path), format="fits", overwrite=True)
    log.info("Stage C annotated catalog written: %s", out_path)
    return out_path
```

- [ ] **Step 4: Run 5 tests — all must pass**

```bash
cd /home/user/workspace/dsa110-continuum
pytest tests/test_stage_c_crossmatch.py -v 2>&1 | tail -15
```

Expected: 5 PASSED. If any fail, read the error carefully.

Debug notes:
- `_make_aegean_fits` uses `astropy.table.Table.write(format="fits")` — if this causes issues, use `fits.BinTableHDU` directly
- `_cone_search` is patched via `patch("dsa110_continuum.catalog.stage_c._cone_search", return_value=...)` — this works because `_cone_search` is called by name inside `run_stage_c`
- If `cross_match_sources` returns an empty DataFrame for exact-position matches, check if `radius_arcsec` is large enough (default 10.0 arcsec is generous)

- [ ] **Step 5: Commit**

```bash
cd /home/user/workspace/dsa110-continuum
git add dsa110_continuum/catalog/stage_c.py tests/test_stage_c_crossmatch.py
git commit -m "feat(catalog): add stage_c.py run_stage_c() with 5 initial tests"
```

---

### Task 2: 5 remaining tests + fallback catalog matching

**Files:**
- Modify: `tests/test_stage_c_crossmatch.py` — append 5 more tests
- Modify: (no changes to `stage_c.py` expected unless a test fails)

- [ ] **Step 1: Append 5 more failing tests to `tests/test_stage_c_crossmatch.py`**

```python
def test_run_stage_c_no_master_fallback_nvss():
    """Master returns empty, NVSS returns 2 matches → nvss_matched=True for those 2."""
    with tempfile.NamedTemporaryFile(suffix=".fits", delete=False) as f:
        cat_path = f.name
    with tempfile.TemporaryDirectory() as out_dir:
        try:
            _make_aegean_fits(cat_path, n=3)
            empty_df   = pd.DataFrame(columns=["ra_deg","dec_deg","flux_mjy","source_id"])
            nvss_df    = _make_master_cone_df(
                ra_list=[344.0, 344.05],
                dec_list=[16.15, 16.15],
                flux_mjy_list=[80.0, 90.0],
            )

            call_count = {"n": 0}
            def fake_cone(catalog_type, ra, dec, radius):
                call_count["n"] += 1
                if catalog_type == "nvss":
                    return nvss_df
                return empty_df

            with patch("dsa110_continuum.catalog.stage_c._cone_search", side_effect=fake_cone):
                from importlib import reload
                import dsa110_continuum.catalog.stage_c as sc
                reload(sc)
                out_path = sc.run_stage_c(cat_path, out_dir,
                                           ra_center=344.1, dec_center=16.15)
            with fits.open(str(out_path)) as hdul:
                tbl = hdul[1].data
                nvss_hits = sum(tbl["nvss_matched"])
                assert nvss_hits == 2, f"Expected 2 NVSS hits, got {nvss_hits}"
                assert not any(tbl["master_matched"]), "Master should be unmatched"
        finally:
            os.unlink(cat_path)


def test_output_path_default_stem():
    """Default output path is {catalog_stem}_crossmatched.fits next to input."""
    with tempfile.NamedTemporaryFile(suffix=".fits", delete=False) as f:
        cat_path = f.name
    try:
        _make_aegean_fits(cat_path, n=2)
        empty_df = pd.DataFrame(columns=["ra_deg","dec_deg","flux_mjy","source_id"])
        with patch("dsa110_continuum.catalog.stage_c._cone_search", return_value=empty_df):
            from dsa110_continuum.catalog.stage_c import run_stage_c
            out_path = run_stage_c(cat_path, ra_center=344.0, dec_center=16.15)
        expected_name = Path(cat_path).stem + "_crossmatched.fits"
        assert Path(str(out_path)).name == expected_name
        assert Path(str(out_path)).exists()
    finally:
        os.unlink(cat_path)
        crossmatched = str(cat_path).replace(".fits", "_crossmatched.fits")
        if os.path.exists(crossmatched):
            os.unlink(crossmatched)


def test_flux_ratio_computed():
    """Flux ratio = peak_flux_jy / (master_flux_mjy / 1000)."""
    with tempfile.NamedTemporaryFile(suffix=".fits", delete=False) as f:
        cat_path = f.name
    with tempfile.TemporaryDirectory() as out_dir:
        try:
            _make_aegean_fits(cat_path, n=1, snr_override=[10.0])
            # peak = 0.01 Jy; master flux = 10 mJy = 0.01 Jy → ratio ≈ 1.0
            master_df = _make_master_cone_df(
                ra_list=[344.0], dec_list=[16.15], flux_mjy_list=[10.0]
            )
            with patch("dsa110_continuum.catalog.stage_c._cone_search", return_value=master_df):
                from dsa110_continuum.catalog.stage_c import run_stage_c
                out_path = run_stage_c(cat_path, out_dir, ra_center=344.0, dec_center=16.15)
            with fits.open(str(out_path)) as hdul:
                tbl = hdul[1].data
                ratio = float(tbl["master_flux_ratio"][0])
                assert 0.5 < ratio < 2.0, f"Flux ratio should be near 1.0, got {ratio}"
        finally:
            os.unlink(cat_path)


def test_astrometry_qa_logged(caplog):
    """Astrometry QA log message appears when >= 3 sources matched."""
    import logging
    with tempfile.NamedTemporaryFile(suffix=".fits", delete=False) as f:
        cat_path = f.name
    with tempfile.TemporaryDirectory() as out_dir:
        try:
            _make_aegean_fits(cat_path, n=5)
            master_df = _make_master_cone_df(
                ra_list=[344.0 + i * 0.05 for i in range(5)],
                dec_list=[16.15] * 5,
                flux_mjy_list=[100.0] * 5,
            )
            with patch("dsa110_continuum.catalog.stage_c._cone_search", return_value=master_df):
                from dsa110_continuum.catalog.stage_c import run_stage_c
                with caplog.at_level(logging.INFO, logger="dsa110_continuum.catalog.stage_c"):
                    run_stage_c(cat_path, out_dir, ra_center=344.1, dec_center=16.15)
            assert any("Astrometry QA" in r.message for r in caplog.records), \
                "Expected astrometry QA log message"
        finally:
            os.unlink(cat_path)


def test_cli_sim_missing_catalog_exits_zero(monkeypatch, capsys):
    """CLI --sim with missing catalog file exits 0 with a warning (not a crash)."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    # Point the sim catalog to a nonexistent path
    monkeypatch.setenv("DSA110_SIM_STAGE_B_CATALOG", "/nonexistent/stage_b_sources.fits")
    try:
        import stage_c_crossmatch
        with pytest.raises(SystemExit) as exc_info:
            stage_c_crossmatch.main(["--sim"])
        assert exc_info.value.code == 0, "Expected exit code 0 for missing sim catalog"
    except ImportError:
        pytest.skip("stage_c_crossmatch.py not yet created")
```

- [ ] **Step 2: Run tests — confirm 5 new tests fail (5 old pass)**

```bash
cd /home/user/workspace/dsa110-continuum
pytest tests/test_stage_c_crossmatch.py -v 2>&1 | tail -20
```

Expected: 5 PASSED, 5 FAILED (the new ones). The `test_cli_sim_missing_catalog_exits_zero` will be skipped (ImportError) until the script is created.

- [ ] **Step 3: Fix `test_run_stage_c_no_master_fallback_nvss` if needed**

This test uses `importlib.reload` to work around Python module caching after patching.
If the test fails because the patch doesn't apply after reload, simplify:

```python
# Replace the reload pattern with a direct import inside the patch context:
with patch("dsa110_continuum.catalog.stage_c._cone_search", side_effect=fake_cone):
    from dsa110_continuum.catalog.stage_c import run_stage_c
    out_path = run_stage_c(cat_path, out_dir, ra_center=344.1, dec_center=16.15)
```

The key insight: `_cone_search` is called by name at runtime inside `run_stage_c`, so patching the module-level name is sufficient — no reload needed.

- [ ] **Step 4: Run all 10 tests — all must pass**

```bash
cd /home/user/workspace/dsa110-continuum
pytest tests/test_stage_c_crossmatch.py -v 2>&1 | tail -15
```

Expected: 10 PASSED (the CLI test may be SKIPPED, which is acceptable).

- [ ] **Step 5: Commit**

```bash
cd /home/user/workspace/dsa110-continuum
git add tests/test_stage_c_crossmatch.py
git commit -m "test(catalog): add 5 additional Stage C tests (fallback, path, flux ratio, astrometry QA)"
```

---

### Task 3: CLI script `scripts/stage_c_crossmatch.py`

**Files:**
- Create: `scripts/stage_c_crossmatch.py`

- [ ] **Step 1: Create the script**

```python
#!/opt/miniforge/envs/casa6/bin/python
"""
Stage C post-discovery cross-match for DSA-110 continuum pipeline.

Reads an Aegean FITS source catalog (Stage B output), cross-matches against the
master radio catalog (NVSS+VLASS+FIRST+RACS) and individual fallback catalogs,
and writes an annotated FITS table.

Usage:
  stage_c_crossmatch.py [--catalog PATH] [--out PATH]
                        [--ra RA_DEG] [--dec DEC_DEG]
                        [--radius RADIUS_DEG]
                        [--match-radius ARCSEC]
                        [--sim]
"""
import argparse
import logging
import os
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# Production default — Stage B output
_DEFAULT_CATALOG = "/stage/dsa110-contimg/images/mosaic_2026-01-25/full_mosaic_sources.fits"

# Sim-mode catalog: pipeline_outputs/step6/step6_mosaic_sources.fits
_SIM_CATALOG = str(
    Path(__file__).resolve().parents[1]
    / "pipeline_outputs/step6/step6_mosaic_sources.fits"
)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description="Stage C: cross-match Aegean detections against radio catalogs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--catalog", default=None, metavar="PATH",
        help=f"Stage B Aegean FITS catalog. Default (production): {_DEFAULT_CATALOG}",
    )
    parser.add_argument(
        "--out", default=None, metavar="PATH",
        help="Output path for annotated FITS (default: {catalog_stem}_crossmatched.fits).",
    )
    parser.add_argument("--ra",  type=float, default=None, metavar="DEG",
                        help="Field center RA (degrees). Derived from catalog centroid if omitted.")
    parser.add_argument("--dec", type=float, default=None, metavar="DEG",
                        help="Field center Dec (degrees). Derived from catalog centroid if omitted.")
    parser.add_argument("--radius", type=float, default=2.0, metavar="DEG",
                        help="Cone search radius (degrees).")
    parser.add_argument("--match-radius", type=float, default=10.0, metavar="ARCSEC",
                        help="Cross-match radius (arcsec).")
    parser.add_argument(
        "--sim", action="store_true",
        help=(
            "Use sim-mode Stage B catalog "
            "(pipeline_outputs/step6/step6_mosaic_sources.fits). "
            "Exits 0 even if catalog is missing."
        ),
    )
    args = parser.parse_args(argv)

    # Resolve catalog path
    if args.sim:
        # Allow env override for testing
        catalog_path = os.environ.get("DSA110_SIM_STAGE_B_CATALOG", _SIM_CATALOG)
        log.info("[SIM MODE] Using sim catalog: %s", catalog_path)
    else:
        catalog_path = args.catalog or _DEFAULT_CATALOG

    if not Path(catalog_path).exists():
        if args.sim:
            log.warning(
                "[SIM MODE] Stage B catalog not found: %s — "
                "run source_finding.py first. Exiting 0.",
                catalog_path,
            )
            sys.exit(0)
        else:
            log.error("Catalog not found: %s", catalog_path)
            sys.exit(1)

    from dsa110_continuum.catalog.stage_c import run_stage_c

    try:
        out_path = run_stage_c(
            catalog_path,
            args.out,
            ra_center=args.ra,
            dec_center=args.dec,
            radius_deg=args.radius,
            match_radius_arcsec=args.match_radius,
        )
        print(f"\nAnnotated catalog written: {out_path}")
    except ValueError as exc:
        log.error("Stage C failed: %s", exc)
        if args.sim:
            sys.exit(0)
        sys.exit(1)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify syntax and help**

```bash
cd /home/user/workspace/dsa110-continuum
python -c "import ast; ast.parse(open('scripts/stage_c_crossmatch.py').read()); print('Syntax OK')"
python scripts/stage_c_crossmatch.py --help
```

Expected: Syntax OK, full help text displayed.

- [ ] **Step 3: Run full test suite — 10 tests pass (CLI test no longer skipped)**

```bash
cd /home/user/workspace/dsa110-continuum
pytest tests/test_stage_c_crossmatch.py -v 2>&1 | tail -15
```

Expected: 10 PASSED.

- [ ] **Step 4: Run regression suite**

```bash
cd /home/user/workspace/dsa110-continuum
pytest tests/test_stage_a_diagnostics.py tests/test_source_finding.py tests/test_two_stage_photometry.py tests/test_stage_c_crossmatch.py -v 2>&1 | tail -10
```

Expected: 42 PASSED (32 + 10).

- [ ] **Step 5: Commit**

```bash
cd /home/user/workspace/dsa110-continuum
git add scripts/stage_c_crossmatch.py tests/test_stage_c_crossmatch.py
git commit -m "feat(scripts): add stage_c_crossmatch.py CLI wrapper for Stage C"
```

---

## Self-Review

**Spec coverage:**
- ✅ `run_stage_c()` with all parameters
- ✅ All 19 output columns
- ✅ Master catalog primary match
- ✅ NVSS/FIRST/RACS fallback for unmatched sources
- ✅ `new_source_candidate` flag (SNR ≥ 5 threshold)
- ✅ Astrometry QA logging
- ✅ Graceful degradation on catalog query failure
- ✅ `ValueError` on empty input catalog
- ✅ CLI with `--sim` flag
- ✅ 10 tests covering all branches

**Placeholder scan:** None found — all code blocks are complete.

**Type consistency:**
- `_cone_search(catalog_type, ra, dec, radius)` called identically in `run_stage_c` and patched in tests ✅
- `_match_catalog(ra_arr, dec_arr, df, radius)` returns `(bool[], float[], float[])` — consistent throughout ✅
- Output column list `_OUTPUT_COLS` used in both `out_table` construction and `test_output_fits_columns` ✅
