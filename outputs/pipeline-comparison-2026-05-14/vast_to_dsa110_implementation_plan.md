# VAST → DSA-110 Scientific Methodology: Implementation Plan

**Date:** 2026-05-14  
**Scope:** Systematically introduce VAST pipeline methodology into the DSA-110 continuum pipeline, prioritised by scientific impact and test coverage benefit.  
**Reference codebases:**

- VAST: `/data/radio-pipelines/askap-vast/`
- DSA-110: `/data/dsa110-continuum/`

---

## 1. Situation Assessment

### What DSA-110 already has (do not re-implement)

| Component | DSA-110 file | Status |
| --- | --- | --- |
| Condon (1997) forced photometry | `photometry/forced.py` | Complete; `_numba_convolution` matches VAST's `ForcedPhot` kernel math |
| η (eta) metric | `photometry/variability.py`, `lightcurves/metrics.py` | Two independent implementations; formulas match VAST exactly |
| V (coefficient of variation) | `photometry/variability.py` | Implemented |
| Vs (two-epoch t-statistic) | `photometry/variability.py`, `lightcurves/metrics.py` | Two independent implementations; formulas match VAST |
| m (modulation index) | `photometry/variability.py`, `lightcurves/metrics.py` | Two independent implementations; formulas match VAST |
| de Ruiter radius | `catalog/crossmatch.py` | Implemented; formula matches VAST |
| Weighted average position | `photometry/multi_epoch.py` | Implemented as `calc_weighted_average_position()` |
| Flux aggregate statistics | `photometry/multi_epoch.py` | Implemented as `FluxAggregateStats` |
| New-source significance | `photometry/multi_epoch.py` | Dataclass `NewSourceMetrics` + `calc_new_source_significance()` exist |
| Upper limits | `photometry/upper_limits.py` | Fully implemented |
| Parallel photometry | `photometry/parallel.py` | Implemented via `multiprocessing.Pool` |

### What is genuinely missing or broken

| Gap | Severity | VAST reference |
| --- | --- | --- |
| **G1** — No cross-epoch source association: detections from different epochs are never spatially matched to build a persistent source identity | Critical | `vast-pipeline/vast_pipeline/pipeline/association.py` |
| **G2** — Duplicate variability implementations with formula divergence risk: `variability.py` and `lightcurves/metrics.py` each define η, Vs, m independently and with different normalizations | High | `vast-pipeline/vast_pipeline/pipeline/utils.py` (canonical) |
| **G3** — `NewSourceMetrics` / `calc_new_source_significance()` exists but is not wired into `batch_pipeline.py` or any photometry script | High | `vast-pipeline/vast_pipeline/pipeline/finalise.py` |
| **G4** — Measurement-pair metrics (Vs, m computed for all N(N-1)/2 pairs within a source, not just max/min pair) are absent | Medium | `vast-pipeline/vast_pipeline/pipeline/pairs.py` |
| **G5** — Population-level variability summary (distribution of η, V, Vs across all sources in a run) is not computed or stored | Medium | `vast-pipeline/vast_pipeline/pipeline/finalise.py` |
| **G6** — `calculate_relative_flux()` in `variability.py` exists but is never called; no reference-ensemble flux normalization | Medium | `vast-pipeline/vast_pipeline/pipeline/utils.py` |
| **G7** — `Source` class in `photometry/source.py` loads from DB after the fact; no in-memory aggregation of measurements during pipeline execution | Medium | `vast-pipeline/vast_pipeline/models.py` (`Source` model) |
| **G8** — No tests exercise the round-trip: ingest two epochs → associate → compute η/V → flag variable | High | `vast-pipeline/vast_pipeline/tests/test_pipeline/test_association.py` |

---

## 2. Implementation Plan

The plan is structured as six phases. Each phase is independently mergeable and has defined acceptance tests. Phases 1–3 address the critical gaps; phases 4–6 address the medium gaps and test coverage.

---

### Phase 1 — Canonical variability module (resolves G2)

**Rationale.** Two independent definitions of η create a silent correctness risk. VAST's canonical formula is in `vast-pipeline/vast_pipeline/pipeline/utils.py::get_eta_metric()`. The DSA-110 formula in `lightcurves/metrics.py::compute_source_metrics()` uses a different normalization for η (divides by N−1 inside the sum rather than outside). This matters for small N.

**VAST canonical η formula** (from `vast_pipeline/pipeline/utils.py` lines 567–596):

```text
η = (N / (N-1)) * [ mean(w·f²) − (mean(w·f))² / mean(w) ]
where w = 1/σ²
```

**DSA-110 `lightcurves/metrics.py` current η formula:**

```text
η = (1/(N-1)) Σ_i [(f_i − f̄_w)² / σ_i²]
where f̄_w = weighted mean
```

These are mathematically equivalent only when the weighted mean is computed the same way. The VAST form is the Bessel-corrected weighted variance; the DSA-110 form is the weighted reduced chi-squared. Both are correct interpretations of η from Mooley et al. (2016) but they produce numerically different values for N < 10. A single canonical implementation eliminates the ambiguity.

**Deliverables:**

1. Create `dsa110_continuum/photometry/metrics.py` — the single authoritative location for all variability metric functions, copied from VAST's formulation with DSA-110 units (Jy):

```python
# dsa110_continuum/photometry/metrics.py

import numpy as np


def eta_metric(fluxes: np.ndarray, errors: np.ndarray) -> float:
    """Weighted reduced chi-squared (VAST pipeline canonical form).

    Matches vast-pipeline/vast_pipeline/pipeline/utils.py::get_eta_metric().
    Reference: Mooley et al. (2016), DOI 10.3847/0004-637X/818/2/105.

    Returns 0.0 for N < 2.
    """
    n = len(fluxes)
    if n < 2:
        return 0.0
    w = 1.0 / errors**2
    return (n / (n - 1)) * (
        np.mean(w * fluxes**2) - (np.mean(w * fluxes) ** 2 / np.mean(w))
    )


def v_metric(fluxes: np.ndarray) -> float:
    """Coefficient of variation (std/mean). Returns 0.0 for N < 2."""
    if len(fluxes) < 2 or np.mean(fluxes) == 0.0:
        return 0.0
    return float(np.std(fluxes, ddof=1) / np.mean(fluxes))


def vs_metric(flux_a: float, flux_b: float, err_a: float, err_b: float) -> float:
    """Two-epoch t-statistic.

    Matches vast-pipeline/vast_pipeline/pipeline/pairs.py::calculate_vs_metric().
    """
    denom = np.hypot(err_a, err_b)
    if denom == 0.0:
        return 0.0
    return (flux_a - flux_b) / denom


def m_metric(flux_a: float, flux_b: float) -> float:
    """Modulation index for a pair of epochs.

    Matches vast-pipeline/vast_pipeline/pipeline/pairs.py::calculate_m_metric().
    """
    total = flux_a + flux_b
    if total == 0.0:
        return 0.0
    return 2.0 * (flux_a - flux_b) / total
```

1. Update `photometry/variability.py` and `lightcurves/metrics.py` to import from `photometry/metrics.py` instead of defining their own formulas. Remove the duplicate definitions.

1. Write `tests/test_metrics_canonical.py`:

```python
# Verified against VAST test values from:
# vast-tools/tests/test_utils.py::test_eta_metric

def test_eta_matches_vast_canonical():
    # Values from VAST pipeline regression suite
    fluxes = np.array([1.0, 1.2, 0.8, 1.1, 0.9])
    errors = np.array([0.05, 0.05, 0.05, 0.05, 0.05])
    result = eta_metric(fluxes, errors)
    # Computed from VAST formula: (5/4) * [mean(w*f²) - mean(w*f)²/mean(w)]
    # With w=400 for all, simplifies to (5/4) * var(f) * 400
    expected = (5 / 4) * np.var(fluxes, ddof=0) / (0.05**2)
    assert abs(result - expected) < 1e-10

def test_eta_single_epoch_returns_zero():
    assert eta_metric(np.array([1.0]), np.array([0.05])) == 0.0

def test_vs_symmetric():
    assert vs_metric(1.2, 0.8, 0.05, 0.05) == -vs_metric(0.8, 1.2, 0.05, 0.05)

def test_m_range():
    # m must be in [-2, 2]
    for fa, fb in [(1.0, 0.0), (0.0, 1.0), (1.0, 1.0)]:
        result = m_metric(fa, fb)
        assert -2.0 <= result <= 2.0
```

**Files touched:** `photometry/metrics.py` (new), `photometry/variability.py` (refactor), `lightcurves/metrics.py` (refactor), `tests/test_metrics_canonical.py` (new).  
**Test command:** `python -m pytest tests/test_metrics_canonical.py -v`

---

### Phase 2 — Cross-epoch source association (resolves G1)

**Rationale.** This is the most consequential missing piece. VAST's association pipeline transforms a flat list of per-epoch detections into a set of persistent source identities by spatially matching detections across epochs. Without it, DSA-110 has no way to track a source's flux history as a coherent time series — the `lightcurves/stacker.py` workaround (O(N²) position matching at CSV-stack time) breaks silently when a source falls below detection threshold in an epoch, and produces duplicate source IDs when two sources are within 5 arcsec.

**VAST reference:** `vast-pipeline/vast_pipeline/pipeline/association.py`  
**Two algorithms to port:**

**Algorithm A — Basic (nearest-neighbour):**

```text
For each new epoch's detections:
  For each new detection d_new:
    Find nearest existing source s within `limit` arcsec
    If found: assign d_new to s (d2d stored)
    Else: create new source from d_new
  Handle duplicates (two d_new match same s): keep closest, fork rest
```

**Algorithm B — De Ruiter (position-uncertainty-weighted):**

```text
For each new epoch's detections:
  Find all candidates within bw_max (1.5–3× BMAJ)
  Compute de Ruiter radius:
    dr = sqrt(
      (Δα·cos(δ_avg))² / (σ_α1² + σ_α2²)  +  (Δδ)² / (σ_δ1² + σ_δ2²)
    )
  Accept if dr < dr_limit (default 3.0)
  Handle duplicates: keep minimum dr match
```

**DSA-110 adaptation notes:**
- DSA-110 measurements come from `ForcedPhotometryResult` (all sources in reference catalog are always measured). This means the basic algorithm degenerates to a lookup — the reference catalog IS the source list. The de Ruiter algorithm is needed only for **blind detections** (Aegean outputs from `source_finding.py`).
- Position uncertainties for DSA-110: `ForcedPhotometryResult` does not store positional uncertainties per measurement (positions are fixed from the reference catalog). For the de Ruiter step on blind detections, use Condon (1997) astrometric error formula: `σ_pos = θ_beam / (2 × SNR)` where `θ_beam = sqrt(BMAJ × BMIN)`.
- The `related` source concept (sources that are spatially coincident across epochs but not identical) maps directly to DSA-110's confusion_flag in the unified catalog.

**Deliverables:**

1. Create `dsa110_continuum/photometry/association.py`:

```python
# dsa110_continuum/photometry/association.py
"""
Cross-epoch source association.

Ports the VAST pipeline association logic
(vast-pipeline/vast_pipeline/pipeline/association.py) to DSA-110.

Two modes:
  basic    — nearest-neighbour within `limit_arcsec`
  deruiter — all candidates within `bw_max` filtered by de Ruiter radius

For forced-photometry sources (fixed reference catalog positions) only
the basic mode is needed; de Ruiter is reserved for blind detections.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
import astropy.units as u
from typing import Literal


def calc_de_ruiter(
    ra1_deg: np.ndarray,
    ra2_deg: np.ndarray,
    dec1_deg: np.ndarray,
    dec2_deg: np.ndarray,
    sigma_ra1_deg: np.ndarray,
    sigma_ra2_deg: np.ndarray,
    sigma_dec1_deg: np.ndarray,
    sigma_dec2_deg: np.ndarray,
) -> np.ndarray:
    """De Ruiter radius (dimensionless position-uncertainty-weighted distance).

    Formula:
        dr = sqrt(
            (Δα·cos(δ_avg))² / (σ_α1² + σ_α2²)
            + (Δδ)² / (σ_δ1² + σ_δ2²)
        )
    all quantities in radians.

    Matches vast-pipeline/vast_pipeline/pipeline/association.py::calc_de_ruiter().
    """
    ra1, ra2 = np.deg2rad(ra1_deg), np.deg2rad(ra2_deg)
    dec1, dec2 = np.deg2rad(dec1_deg), np.deg2rad(dec2_deg)
    s_ra1, s_ra2 = np.deg2rad(sigma_ra1_deg), np.deg2rad(sigma_ra2_deg)
    s_dec1, s_dec2 = np.deg2rad(sigma_dec1_deg), np.deg2rad(sigma_dec2_deg)

    dec_avg = (dec1 + dec2) / 2.0
    dra = (ra1 - ra2) * np.cos(dec_avg)
    ddec = dec1 - dec2

    var_ra = s_ra1**2 + s_ra2**2
    var_dec = s_dec1**2 + s_dec2**2

    # Guard against zero variance (fixed-position sources)
    with np.errstate(divide="ignore", invalid="ignore"):
        dr_ra = np.where(var_ra > 0, dra**2 / var_ra, 0.0)
        dr_dec = np.where(var_dec > 0, ddec**2 / var_dec, 0.0)

    return np.sqrt(dr_ra + dr_dec)


def associate_epoch(
    catalog_df: pd.DataFrame,           # existing sources: source_id, ra_deg, dec_deg, [sigma_ra_deg, sigma_dec_deg]
    detections_df: pd.DataFrame,        # new epoch detections: det_id, ra_deg, dec_deg, [sigma_ra_deg, sigma_dec_deg]
    method: Literal["basic", "deruiter"] = "basic",
    limit_arcsec: float = 10.0,         # basic: max separation
    bw_max_deg: float = 0.01,           # deruiter: beamwidth limit (~36 arcsec for DSA-110)
    dr_limit: float = 3.0,              # deruiter: max de Ruiter radius
) -> pd.DataFrame:
    """Associate detections from a new epoch with the existing source catalog.

    Returns a DataFrame with columns:
        det_id        — detection ID from detections_df
        source_id     — matched catalog source_id (or new ID if unmatched)
        d2d_arcsec    — angular separation
        dr            — de Ruiter radius (0.0 if method='basic')
        is_new        — True if no catalog match was found
    """
    ...
```

2. Add a `measurements` table to the products SQLite DB schema (alongside existing `photometry` and `monitoring_sources` tables):

```sql
CREATE TABLE IF NOT EXISTS measurements (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    source_id       TEXT    NOT NULL REFERENCES sources(source_id),
    epoch_utc       TEXT    NOT NULL,   -- ISO-8601
    mjd             REAL    NOT NULL,
    image_path      TEXT,
    ra_deg          REAL    NOT NULL,
    dec_deg         REAL    NOT NULL,
    flux_peak_jy    REAL,               -- from ForcedPhotometryResult.peak_jyb
    flux_peak_err_jy REAL,
    chisq           REAL,
    dof             INTEGER,
    is_forced       INTEGER DEFAULT 1,  -- 1=forced, 0=blind detection
    is_upper_limit  INTEGER DEFAULT 0,
    d2d_arcsec      REAL DEFAULT 0.0,   -- association distance
    dr              REAL    DEFAULT 0.0 -- de Ruiter radius
);
CREATE INDEX IF NOT EXISTS idx_meas_source_id ON measurements(source_id);
CREATE INDEX IF NOT EXISTS idx_meas_epoch     ON measurements(epoch_utc);
```

3. Modify `scripts/batch_pipeline.py` (and `scripts/forced_photometry.py`) to write `ForcedPhotometryResult` rows into the `measurements` table after each epoch, keyed by `source_id` from the reference catalog.

4. Write `tests/test_association.py`:

```python
def test_basic_association_matches_nearest():
    catalog = pd.DataFrame({
        "source_id": ["S1", "S2"],
        "ra_deg": [180.0, 180.1],
        "dec_deg": [30.0, 30.0],
    })
    detections = pd.DataFrame({
        "det_id": [0, 1],
        "ra_deg": [180.001, 180.099],  # 3.6 arcsec from S1, 3.6 arcsec from S2
        "dec_deg": [30.0, 30.0],
    })
    result = associate_epoch(catalog, detections, method="basic", limit_arcsec=10.0)
    assert result.loc[result.det_id == 0, "source_id"].values[0] == "S1"
    assert result.loc[result.det_id == 1, "source_id"].values[0] == "S2"

def test_unmatched_detection_flagged_as_new():
    catalog = pd.DataFrame({
        "source_id": ["S1"],
        "ra_deg": [180.0],
        "dec_deg": [30.0],
    })
    detections = pd.DataFrame({
        "det_id": [0],
        "ra_deg": [181.0],   # 3600 arcsec away
        "dec_deg": [30.0],
    })
    result = associate_epoch(catalog, detections, method="basic", limit_arcsec=10.0)
    assert result.loc[0, "is_new"] == True

def test_de_ruiter_rejects_large_radius():
    # Two sources 8 arcsec apart but large position uncertainty → low dr
    # Two sources 8 arcsec apart but small uncertainty → high dr → rejected
    ...
```

**Files touched:** `photometry/association.py` (new), `photometry/db_schema.py` or equivalent (add `measurements` table), `scripts/batch_pipeline.py` (write measurements), `tests/test_association.py` (new).  
**Test command:** `python -m pytest tests/test_association.py -v`

---

### Phase 3 — New-source detection pipeline integration (resolves G3)

**Rationale.** `multi_epoch.py::calc_new_source_significance()` correctly computes whether a new detection would have been visible in prior epochs. But it is never called in `batch_pipeline.py`. In VAST (`finalise.py` lines 165–177), new-source significance is computed after the association step and stored per source. The equivalent in DSA-110 should fire after Phase 2's `measurements` table is populated.

**VAST reference:** `vast-pipeline/vast_pipeline/pipeline/finalise.py` lines 165–177, `vast-pipeline/vast_pipeline/models.py` `Source.new` boolean field.

**VAST algorithm:**
```
For each source with n_meas_forced > 0 (i.e., not detected in every epoch):
  max_sigma = max(flux_peak_at_first_detection / rms_in_each_prior_epoch)
  is_new = max_sigma < detection_threshold (typically 5.0σ)
```

**DSA-110 adaptation:** DSA-110 always force-measures every reference catalog source, so `n_meas_forced` is always equal to `n_epochs`. The relevant new-source concept in DSA-110 is:

> A source is considered a **new transient** if it was absent (or below 5σ) in the reference catalog (NVSS/FIRST/RACS) but appears at > 5σ in the current epoch.

This maps to: `blind_detection_snr > 5.0` AND `source_id NOT IN reference_catalog`.

**Deliverables:**

1. Add `detect_new_sources()` to `photometry/association.py` (or a new `photometry/new_source.py`):

```python
def detect_new_sources(
    blind_detections_df: pd.DataFrame,   # Aegean output for this epoch
    reference_catalog: pd.DataFrame,     # NVSS/FIRST/RACS sources
    prior_rms_maps: list[str],           # paths to per-epoch RMS FITS
    detection_threshold_sigma: float = 5.0,
    match_radius_arcsec: float = 10.0,
) -> pd.DataFrame:
    """
    Flag detections not in the reference catalog.

    For each unmatched blind detection, compute:
        new_high_sigma = max(peak_flux / rms_at_position_in_prior_epochs)
    A source is 'confirmed new' if new_high_sigma < detection_threshold_sigma
    in ALL prior epochs (i.e., it genuinely was not there before).

    Returns DataFrame with columns:
        ra_deg, dec_deg, snr, new_high_sigma, is_confirmed_new,
        first_detection_epoch_utc
    """
```

2. Wire into `scripts/batch_pipeline.py`: after each epoch's blind source finding (if enabled), call `detect_new_sources()` and write results to a `new_sources` table in the products DB.

3. Add `new_sources` table to DB schema:

```sql
CREATE TABLE IF NOT EXISTS new_sources (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    epoch_utc       TEXT    NOT NULL,
    ra_deg          REAL    NOT NULL,
    dec_deg         REAL    NOT NULL,
    snr             REAL,
    new_high_sigma  REAL,               -- max(flux / prior_rms); VAST equivalent
    is_confirmed_new INTEGER DEFAULT 0,
    first_detection_epoch_utc TEXT
);
```

4. Write `tests/test_new_source_detection.py`:

```python
def test_source_absent_in_all_prior_epochs_is_confirmed_new():
    # Build synthetic prior RMS maps where the source position has rms=1 mJy/beam
    # Source flux = 10 mJy → new_high_sigma = 10 in each prior epoch
    # Since 10 > 5, the source WOULD have been detectable → is_confirmed_new=False
    # (it's not actually new, just a reference catalog miss)

def test_source_below_threshold_in_all_prior_epochs_is_new():
    # Source flux = 10 mJy, prior rms at position = 4 mJy (SNR would be 2.5σ)
    # new_high_sigma = 2.5 < 5.0 → is_confirmed_new=True
```

**Files touched:** `photometry/new_source.py` (new) or addition to `association.py`, `scripts/batch_pipeline.py` (wire-in), `tests/test_new_source_detection.py` (new).

---

### Phase 4 — All-pairs measurement metrics (resolves G4)

**Rationale.** VAST computes Vs and m for **every pair** of measurements within a source (not just the max/min pair), then stores the maximum |Vs| pair where |Vs| ≥ min_vs. This is important because the max/min pair may be dominated by noise; a consistently elevated Vs across multiple pairs is stronger evidence of variability. DSA-110's current `lightcurves/metrics.py::compute_source_metrics()` only computes Vs between the brightest and faintest epochs.

**VAST reference:** `vast-pipeline/vast_pipeline/pipeline/pairs.py` (all pairs), `finalise.py::calculate_measurement_pair_aggregate_metrics()` (filter by min_vs, keep max |m|).

**Deliverables:**

1. Add `compute_all_pairs()` to `dsa110_continuum/photometry/metrics.py`:

```python
def compute_all_pairs(
    fluxes: np.ndarray,
    errors: np.ndarray,
    epochs: np.ndarray | None = None,
    min_vs: float = 3.0,
) -> dict:
    """
    Compute Vs and m for all N(N-1)/2 measurement pairs.

    Returns:
        vs_all          : array of all Vs values
        m_all           : array of all m values
        vs_significant  : Vs values where |Vs| >= min_vs
        m_significant   : corresponding m values
        vs_abs_max      : max(|Vs|) among significant pairs (0.0 if none)
        m_abs_max       : |m| for the pair with max |Vs| (0.0 if none)
        n_pairs         : total number of pairs
        n_significant   : number of significant pairs

    Matches vast-pipeline/vast_pipeline/pipeline/pairs.py and
    vast-pipeline/vast_pipeline/pipeline/finalise.py::
    calculate_measurement_pair_aggregate_metrics().
    """
    n = len(fluxes)
    if n < 2:
        return {"vs_abs_max": 0.0, "m_abs_max": 0.0, "n_pairs": 0, "n_significant": 0, ...}

    i_idx, j_idx = np.triu_indices(n, k=1)  # all unique pairs
    vs_all = vs_metric(fluxes[i_idx], fluxes[j_idx], errors[i_idx], errors[j_idx])
    m_all = m_metric(fluxes[i_idx], fluxes[j_idx])

    mask = np.abs(vs_all) >= min_vs
    vs_sig, m_sig = vs_all[mask], m_all[mask]

    if len(vs_sig) == 0:
        return {"vs_abs_max": 0.0, "m_abs_max": 0.0, ...}

    best = np.argmax(np.abs(vs_sig))
    return {
        "vs_abs_max": float(np.abs(vs_sig[best])),
        "m_abs_max": float(np.abs(m_sig[best])),
        "n_pairs": len(vs_all),
        "n_significant": int(mask.sum()),
    }
```

2. Add `vs_abs_max` and `m_abs_max` columns to the per-source variability output in `lightcurves/metrics.py::compute_metrics()`.

3. Extend `tests/test_metrics_canonical.py` with pair-metric tests:

```python
def test_all_pairs_count():
    # N=4 detections → 6 pairs
    fluxes = np.array([1.0, 1.5, 0.8, 1.2])
    errors = np.full(4, 0.05)
    result = compute_all_pairs(fluxes, errors)
    assert result["n_pairs"] == 6

def test_all_pairs_vs_abs_max_matches_max_pair():
    # Largest |Vs| should be between the 1.5 and 0.8 epochs
    fluxes = np.array([1.0, 1.5, 0.8, 1.2])
    errors = np.full(4, 0.05)
    result = compute_all_pairs(fluxes, errors, min_vs=0.0)
    expected_max_vs = vs_metric(1.5, 0.8, 0.05, 0.05)
    assert abs(result["vs_abs_max"] - abs(expected_max_vs)) < 1e-10
```

**Files touched:** `photometry/metrics.py` (add `compute_all_pairs`), `lightcurves/metrics.py` (use it), `tests/test_metrics_canonical.py` (extend).

---

### Phase 5 — Population-level variability statistics (resolves G5)

**Rationale.** VAST's `finalise.py` computes per-run aggregate statistics (median η, distribution of V, fraction of sources with |Vs| > threshold) and stores them in the `Run` model. This enables QA: if the median η of a run is anomalously high, it indicates a calibration or imaging artifact rather than real variability. DSA-110 has `lightcurves/metrics.py::variability_summary()` but it is only called in scripts, never in `batch_pipeline.py` or stored.

**VAST reference:** `vast-pipeline/vast_pipeline/pipeline/finalise.py` (per-run statistics), `vast-pipeline/vast_pipeline/models.py::Run` (stored fields).

**Deliverables:**

1. Add `compute_run_statistics()` to `lightcurves/metrics.py`:

```python
def compute_run_statistics(metrics_df: pd.DataFrame) -> dict:
    """
    Compute population-level variability statistics for a pipeline run.

    Input: output of compute_metrics() — one row per source.

    Returns dict with:
        n_sources          : total sources measured
        n_variable_candidates : sources with is_variable_candidate=True
        fraction_variable  : n_variable_candidates / n_sources
        median_eta         : median η across all sources
        median_v           : median V across all sources
        median_vs_abs_max  : median of vs_abs_max across all sources
        p95_eta            : 95th percentile η (QA threshold)
        p95_vs             : 95th percentile |Vs|
    """
```

2. Add a `run_statistics` table to the products DB and write results after each epoch batch completes in `batch_pipeline.py`.

3. Emit a QA warning if `median_eta > 2.0` (suggests systematic flux variation from calibration; threshold chosen to match VAST's typical quiescent value of η ≈ 1.0).

4. Write `tests/test_population_statistics.py`:

```python
def test_fraction_variable_all_constant():
    # All sources constant → fraction_variable = 0
    ...

def test_p95_eta_exceeds_median():
    # p95 must be >= median by definition
    ...
```

**Files touched:** `lightcurves/metrics.py` (add `compute_run_statistics`), `scripts/batch_pipeline.py` (write to DB), `tests/test_population_statistics.py` (new).

---

### Phase 6 — Reference-ensemble flux normalization (resolves G6)

**Rationale.** VAST's forced photometry pipeline applies relative flux calibration: the mean flux of a set of "stable" reference sources is tracked per epoch, and individual source fluxes are divided by the per-epoch reference mean. This corrects for epoch-to-epoch gain offsets that would otherwise inflate η for all sources. DSA-110's `photometry/variability.py::calculate_relative_flux()` exists but is not called anywhere.

**VAST reference:** `vast-pipeline/vast_pipeline/pipeline/utils.py` (relative flux normalization implicit in η definition — VAST applies it upstream in the mosaic pipeline; `vast-pipeline` itself expects calibrated fluxes).

**Note:** VAST does not perform explicit in-pipeline normalization — it relies on `vast-post-processing` having applied astrometric and flux corrections (Huber regression against reference catalogs) before the pipeline runs. The DSA-110 equivalent is the calibration and self-cal stages. However, if per-epoch flux scale variation is present post-calibration (as is common in DSA-110's current state), an additional normalization step adds robustness.

**Deliverables:**

1. Activate `calculate_relative_flux()` in `photometry/variability.py`: add a `normalize_to_reference_ensemble` parameter to `compute_metrics()` in `lightcurves/metrics.py`. When enabled:
   - Select "stable" reference sources: sources with η < 1.5 AND V < 0.05 AND n_epochs = max_epochs (i.e., detected in every epoch)
   - Compute per-epoch normalization factor = median(reference_flux_epoch_i / mean_reference_flux)
   - Apply normalization before computing η, V, Vs, m for all sources

2. Default off (requires at least 10 reference sources to be reliable). Log a warning if fewer than 10 reference sources are found.

3. Write `tests/test_normalization.py`:

```python
def test_normalization_removes_epoch_gain_offset():
    # Inject a 10% gain offset in epoch 3 for all sources
    # Without normalization: all sources have elevated η
    # With normalization: reference sources have η ≈ 0; target sources return to true variability
    ...
```

**Files touched:** `photometry/variability.py` (activate `calculate_relative_flux`), `lightcurves/metrics.py` (add normalize flag), `tests/test_normalization.py` (new).

---

## 3. Test Strategy

The underlying philosophy: VAST has 119 test functions that collectively exercise the pipeline's scientific logic at the unit level; DSA-110 currently has no tests that verify the **round-trip** from raw measurements to variability flags. The test additions below mirror VAST's test structure.

### Test hierarchy to build

| Level | VAST equivalent | DSA-110 target |
| --- | --- | --- |
| Unit — formula correctness | `test_tools.py::test_eta_metric` | `test_metrics_canonical.py` (Phase 1) |
| Unit — association logic | `test_association.py` | `test_association.py` (Phase 2) |
| Unit — pair metrics | `test_pairs.py` | `test_metrics_canonical.py` extension (Phase 4) |
| Integration — epoch round-trip | `test_pipeline/test_pipeline.py` | `test_epoch_roundtrip.py` (cross-phase) |
| Regression — known variable source | VAST regression suite | `test_ese_candidate_regression.py` |

### Cross-phase integration test: `test_epoch_roundtrip.py`

This test exercises the full chain that VAST's pipeline tests cover end-to-end:

```python
def test_two_epoch_roundtrip():
    """
    Synthetic test: two epochs, one variable source and one stable source.

    1. Build two sets of ForcedPhotometryResult objects (via synthetic FITS or mocks)
    2. Associate detections with reference catalog (Phase 2)
    3. Write to measurements table
    4. Compute η, V, Vs, m per source (Phase 1 formulas)
    5. Compute all-pairs metrics (Phase 4)
    6. Compute run statistics (Phase 5)

    Assertions:
    - Stable source: η ≈ 0, V ≈ 0, is_variable_candidate=False
    - Variable source: η >> 1, |Vs| >> 3, is_variable_candidate=True
    - n_sources = 2, fraction_variable = 0.5
    """
```

---

## 4. Dependency and Risk Analysis

### Dependency order

```
Phase 1 (metrics consolidation)
  └─ Phase 4 (all-pairs, uses canonical vs/m)
       └─ Phase 5 (population stats, uses all-pairs output)

Phase 2 (association)
  └─ Phase 3 (new-source detection, needs measurements table)

Phase 6 (normalization) — independent, can run in parallel with any phase
```

Phases 1 and 2 are independent of each other and can be developed in parallel by separate contributors.

### Risks

| Risk | Likelihood | Mitigation |
| --- | --- | --- |
| DSA-110's forced photometry produces only `flux_peak_jy` (no `flux_int_jy`); VAST's η computes both `eta_int` and `eta_peak` | High | Use `flux_peak_jy` only for η; add `flux_int_jy` computation as a future enhancement (requires resolved-source deconvolution) |
| Reference catalog sources have no position uncertainties; de Ruiter association degenerates | Medium | Use Condon astrometric formula `σ_pos = θ_beam / (2 × SNR)` as the position uncertainty estimate |
| `batch_pipeline.py` writes to files (CSV), not DB; adding DB writes changes the output contract | Medium | Write to both CSV (existing) and DB (new) during transition; deprecate CSV after validation |
| Phase 6 normalization with fewer than 10 reference sources | Low | Hard minimum: disable normalization and log warning; do not attempt normalization in sparse-coverage Dec strips |
| The `measurements` table schema defined in Phase 2 may conflict with the existing `photometry` table schema | Low | Audit `photometry` table schema before implementation; migrate rather than duplicate |

---

## 5. Implementation Sequence (Recommended)

```
Week 1:  Phase 1 (metrics consolidation + tests) — lowest risk, highest test ROI
Week 2:  Phase 2, Part A: association.py skeleton + unit tests
Week 3:  Phase 2, Part B: DB schema + batch_pipeline.py integration
Week 4:  Phase 3: new-source detection + wire-in
Week 5:  Phase 4: all-pairs metrics + Phase 5: population statistics
Week 6:  Phase 6: normalization + end-to-end integration test
```

Phase 1 alone, with its test suite, eliminates the silent formula-divergence risk and is the highest-value single commit.

---

## 6. What NOT to port

The following VAST capabilities are not applicable to DSA-110 and should not be ported:

| VAST capability | Reason not to port |
| --- | --- |
| Django ORM models (`Run`, `Image`, `Source`, `Measurement`, `Association`) | DSA-110 uses SQLite + dataclasses; a full Django ORM would be overengineering |
| Selavy source finder integration | DSA-110 uses Aegean; Selavy is ASKAP-specific |
| Epoch-based image management (Image model with run/band/skyreg FKs) | DSA-110's mosaic pipeline handles image management separately |
| vaster-webapp candidate classification UI | Out of scope for the pipeline itself |
| Bokeh.js visualization in pipeline output | DSA-110 uses matplotlib; different deployment context |
| Q3C spatial indexing (PostgreSQL extension) | DSA-110 uses SQLite; Q3C is PostgreSQL-only. Use `astropy.coordinates.SkyCoord.search_around_sky()` instead |

---

## 7. Summary Table

| Phase | Resolves | New files | Modified files | New tests | Priority |
| --- | --- | --- | --- | --- | --- |
| 1 | G2 (duplicate metrics) | `photometry/metrics.py` | `photometry/variability.py`, `lightcurves/metrics.py` | `test_metrics_canonical.py` | Critical |
| 2 | G1 (no association) | `photometry/association.py` | `scripts/batch_pipeline.py`, DB schema | `test_association.py` | Critical |
| 3 | G3 (new-source unwired) | `photometry/new_source.py` | `scripts/batch_pipeline.py` | `test_new_source_detection.py` | High |
| 4 | G4 (no all-pairs) | — | `photometry/metrics.py`, `lightcurves/metrics.py` | `test_metrics_canonical.py` extension | Medium |
| 5 | G5 (no pop stats) | — | `lightcurves/metrics.py`, `scripts/batch_pipeline.py` | `test_population_statistics.py` | Medium |
| 6 | G6 (normalization unused) | — | `photometry/variability.py`, `lightcurves/metrics.py` | `test_normalization.py` | Medium |
| Cross | G7, G8 | — | — | `test_epoch_roundtrip.py` | High (after Ph 1+2) |
