# DSA-110 Data Quality Assurance: Broader Design

**Date:** 2026-04-19  
**Scope:** Pipeline-wide DQA gap analysis and architecture for Stages A/B/C and cross-cutting concerns

---

## Executive Summary

The codebase contains rich, well-implemented QA infrastructure that is largely disconnected from the running pipeline. The dominant pattern throughout is: *capability exists in a library module; nothing calls it from a script or stage boundary.* The design below closes those gaps systematically, adds two missing capabilities (noise floor comparison and source-finding completeness), and establishes a lightweight epoch-level QA accumulation mechanism.

---

## Current State by Stage

### What exists and is wired

| Location | What it does |
|----------|-------------|
| `forced_photometry.py` | Logs median flux ratio, outlier count, ratio range |
| `source_finding/core.py::check_catalog` | Logs source count, bright count, sky-window count |
| `stage_c.py` | Logs astrometric QA (median ΔRA/ΔDec), new-source candidate names |
| `photometry/epoch_qa.py::measure_epoch_qa` | Three-gate QA (flux scale, completeness, noise floor) — correct logic, not called from Stage A |
| `qa/pipeline_hooks.py` | Full calibration metrics ingestion pipeline — complete, wired to calibration only |

### What exists but is unwired

| Module | Capability | Gap |
|--------|------------|-----|
| `qa/noise_model.py::calculate_theoretical_rms` | Radiometer-equation noise prediction | Never compared to measured mosaic RMS |
| `qa/noise_model.py::validate_noise_prediction` | Pass/fail comparison of measured vs predicted RMS | Not called anywhere |
| `qa/image_metrics.py::calculate_dynamic_range` | Peak/RMS dynamic range | Not called before source finding |
| `qa/catalog_validation.py::run_full_validation` | Astrometry, flux scale, source counts vs catalogs | Not called from Stage A or B |
| `lightcurves/metrics.py::compute_metrics` | Mooley et al. η, Vs, m variability metrics | Not called from `plot_lightcurves.py` |
| `lightcurves/metrics.py::variability_summary` | Summary dict of variable candidates | Not called |
| `catalog/variable_source_detection.py` | Full variability alert pipeline | Not called |

### Known bugs / inaccuracies

| Location | Issue |
|----------|-------|
| `qa/noise_model.py` | `num_antennas=117` default — should be 96 (47 E-W + 35 N-S + 14 outriggers, verified 2026-05-05 against H17 HDF5 metadata) |
| `photometry/epoch_qa.py` | `QA_RMS_LIMIT_MJY = 17.1` comment says "TODO: recompute" — this depends on the 117→96 fix |
| `stage_c.py` | Flux ratio distribution never summarized (median, std, outlier fraction across all matched sources) |

---

## Architecture: Four DQA Planes

The gaps fall cleanly into four independent planes. Each is self-contained and can be implemented and tested separately.

```
Plane 1: Image Quality Gate     (before source finding)
Plane 2: Noise Floor Validation  (end of Stage A forced photometry)
Plane 3: Source-Finding QA       (end of Stage B)
Plane 4: Variability Metrics     (end of light-curve generation)
```

A fifth cross-cutting concern — epoch-level QA accumulation — threads through all four planes.

---

## Plane 1: Image Quality Gate (pre-source-finding)

### Problem

Stage B (source finding) currently runs on whatever mosaic it receives. A poorly deconvolved image (dynamic range < ~50:1, RMS > 3× theoretical, or non-Gaussian residuals) will produce a noisy or spurious source catalog. Nothing currently checks image quality before BANE/Aegean are invoked.

### Design

A new function `check_image_quality_for_source_finding(mosaic_path) -> ImageQAResult` in a new module `dsa110_continuum/qa/image_gate.py`.

**Checks performed:**

1. **Dynamic range gate** — `calculate_dynamic_range(mosaic_path)` → warn if < 100, fail if < 30
2. **Noise floor ratio** — compare MAD-RMS of the mosaic to `calculate_theoretical_rms(integration_time_s=..., num_antennas=96)` → warn if ratio > 1.5, fail if > 3.0
3. **Pixel coverage** — fraction of finite pixels in the mosaic → fail if < 0.5 (indicates a badly mosaicked or empty image)
4. **Beam sanity** — if BMAJ/BMIN present in header, check BMAJ/pixel_scale is in a sane range (1–10 beams across); log if absent

All checks produce a named result. The gate is non-blocking by default (the function returns a result struct; the caller decides whether to abort). `source_finding.py` uses it to log a warning block before invoking BANE.

**Result struct:**

```python
@dataclass
class ImageQAResult:
    dynamic_range: float
    dynamic_range_gate: Literal["PASS", "WARN", "FAIL"]
    rms_mjy: float
    theoretical_rms_mjy: float
    rms_ratio: float
    rms_ratio_gate: Literal["PASS", "WARN", "FAIL"]
    pixel_coverage_frac: float
    pixel_coverage_gate: Literal["PASS", "FAIL"]
    overall: Literal["PASS", "WARN", "FAIL"]
```

**Wiring:** Called from `scripts/source_finding.py` at the top of `main()`, before `run_source_finding()`. A `--skip-image-qa` flag allows bypass. Result is logged as a structured block.

---

## Plane 2: Noise Floor Validation (Stage A)

### Problem

`forced_photometry.py` measures the mosaic RMS via MAD-std but never compares it to the theoretical noise floor predicted by the radiometer equation. A ratio of 2–3× is normal (calibration residuals, sidelobe confusion). A ratio > 5× indicates a seriously degraded epoch.

Additionally, `num_antennas=117` in `noise_model.py` is wrong and produces an 11% too-optimistic noise prediction.

### Design

**Fix 1:** Correct `num_antennas` default to 96 in `qa/noise_model.py`. Add a `bandwidth_hz` parameter update comment noting the 188 MHz usable bandwidth (768 channels × 244 kHz, minus ~20% RFI flagging).

**Fix 2:** Call `validate_noise_prediction` at the end of `run_forced_photometry` (after the QA summary block), passing `measured_rms` from the mosaic MAD-std (already computed in the function body). Use `integration_time_s=12.88` as a fallback when no MS path is available (drift-scan constant).

**Addition to the QA summary dict** returned by `run_forced_photometry`:

```python
{
    "n_sources": int,
    "median_ratio": float,
    "csv_path": str,
    "rms_mjy": float,           # NEW: measured mosaic MAD-RMS
    "theoretical_rms_mjy": float,  # NEW: radiometer prediction
    "rms_ratio": float,         # NEW: measured / theoretical
    "noise_gate": str,          # NEW: "PASS" / "WARN" / "FAIL"
}
```

**Thresholds:**
- `PASS`: rms_ratio ≤ 1.5
- `WARN`: 1.5 < rms_ratio ≤ 3.0 (log at WARNING level)
- `FAIL`: rms_ratio > 3.0 (log at ERROR level — pipeline continues but marks epoch)

---

## Plane 3: Source-Finding QA (Stage B)

### Problem

`check_catalog()` only counts sources. It does not measure:
- **Completeness**: fraction of known bright sources recovered by Aegean
- **Size distribution**: whether fitted sizes are consistent with the PSF (detecting sidelobe pickup vs. real sources)

### Design

**3a: Source Recovery Completeness**

New function `check_source_completeness(catalog: list[SourceCatalogEntry], mosaic_path: str, *, sigma_threshold: float = 5.0) -> CompletenessResult` in `dsa110_continuum/source_finding/core.py`.

Algorithm:
1. Query NVSS sources above `5 × local_rms` in the mosaic footprint (using `_query_nvss_in_footprint` from `epoch_qa.py`, or a thin wrapper around `cone_search("nvss", ...)`)
2. Cross-match the Aegean catalog against those NVSS positions (using `cross_match_sources`, 15 arcsec radius)
3. Report fraction recovered

```python
@dataclass
class CompletenessResult:
    n_nvss_reference: int
    n_recovered: int
    completeness_frac: float
    gate: Literal["PASS", "WARN", "FAIL"]   # PASS ≥0.6, WARN ≥0.4, FAIL <0.4
```

**3b: Source Size Distribution**

New function `check_size_distribution(catalog: list[SourceCatalogEntry], beam_a_arcsec: float, beam_b_arcsec: float) -> SizeQAResult` in `dsa110_continuum/source_finding/core.py`.

Checks:
- Fraction of sources with fitted `a_arcsec < 0.9 × beam_a_arcsec` (sub-beam, likely sidelobe artefacts): warn if > 5%
- Fraction of sources with `a_arcsec / b_arcsec > 5` (extremely elongated): warn if > 10%

```python
@dataclass  
class SizeQAResult:
    n_sources: int
    frac_subbeam: float
    frac_elongated: float
    beam_a_arcsec: float
    beam_b_arcsec: float
    gate: Literal["PASS", "WARN"]
```

**Wiring:** Both functions called from `run_source_finding()` after `write_catalog()`. `beam_a_arcsec=36.9, beam_b_arcsec=25.5` are known DSA-110 constants (from ground truth). The FITS header BMAJ/BMIN values are used if present; otherwise these defaults are used.

---

## Plane 4: Variability Metrics (Light Curves)

### Problem

`lightcurves/metrics.py` implements the complete Mooley et al. (2016) η/Vs/m framework correctly. `plot_lightcurves.py` generates per-source light curve PNGs but never computes variability metrics, never flags candidates, and never writes a summary.

### Design

At the end of `plot_lightcurves.py::main()`, after all per-source PNGs are written:

1. Call `compute_metrics(stacked_df)` on the full stacked DataFrame
2. Call `variability_summary(metrics_df)` for a log-friendly dict
3. Log the summary (n_sources, n_candidates, fraction_variable, median η, median Vs)
4. Write `{output_dir}/variability_summary.csv` — one row per source, sorted by η descending
5. Flag candidates (Vs > 4.0 OR η > 2.5) with `log.warning("Variable candidate: {name} Vs={:.2f} η={:.2f}")`

**No new modules needed.** All the needed functions already exist. This is purely wiring.

**Expected output columns in `variability_summary.csv`:**
`source_id, ra_deg, dec_deg, n_epochs, mean_flux, std_flux, m, Vs, eta, is_variable_candidate, catalog_flux_jy`

---

## Cross-cutting: Epoch-Level QA Accumulation

### Problem

Each epoch's QA results are logged to stdout/stderr but never written to a persistent record. There is no way to look back and see which epochs were degraded, or to detect trends (e.g., steadily rising noise floor over 3 weeks).

### Design

A lightweight `EpochQARecord` appended to a JSONL file (`pipeline_outputs/qa_log.jsonl`) at the end of each stage's run. JSONL (one JSON object per line) requires no schema, no database, and is trivially readable.

**Record schema** (emitted by each pipeline stage):

```json
{
    "epoch_utc": "2026-01-25T22:26:05",
    "stage": "forced_photometry",
    "mosaic_path": "pipeline_outputs/step6/step6_mosaic.fits",
    "n_sources": 47,
    "median_flux_ratio": 0.97,
    "rms_mjy": 8.9,
    "theoretical_rms_mjy": 7.1,
    "rms_ratio": 1.25,
    "noise_gate": "PASS",
    "overall_gate": "PASS",
    "timestamp": "2026-04-19T15:07:00Z"
}
```

```json
{
    "epoch_utc": "2026-01-25T22:26:05",
    "stage": "source_finding",
    "n_sources_aegean": 52,
    "n_nvss_reference": 61,
    "completeness_frac": 0.852,
    "completeness_gate": "PASS",
    "frac_subbeam": 0.02,
    "size_gate": "PASS",
    "overall_gate": "PASS",
    "timestamp": "2026-04-19T15:08:00Z"
}
```

**New module:** `dsa110_continuum/qa/epoch_log.py` — a single public function:

```python
def append_epoch_qa(record: dict, log_path: str | Path = "pipeline_outputs/qa_log.jsonl") -> None:
    """Append one QA record to the epoch QA log (JSONL format, one record per line)."""
```

This is a 15-line function. Each stage calls it with its relevant dict after computing QA results.

**Reading the log:** Any downstream tool (or a future dashboard) can `pd.read_json("qa_log.jsonl", lines=True)` to get a DataFrame of all epochs for trending.

---

## Bug Fixes (Not New Features)

These are correctness fixes that should land before any new QA planes are wired in:

| File | Fix |
|------|-----|
| `qa/noise_model.py` | Change `num_antennas=117` default to `96` |
| `qa/noise_model.py` | Update `bandwidth_hz` comment: 188 MHz effective (20% RFI flagging of 235 MHz) |
| `photometry/epoch_qa.py` | Recompute `QA_RMS_LIMIT_MJY` using 96 antennas, 188 MHz bandwidth, T_sys=25K: new value ~18.5 mJy/beam (2× empirical floor with 96 antennas) |

---

## Implementation Order and Dependencies

```
Bug fixes (noise_model.py, epoch_qa.py)
    ↓
Plane 2: Noise floor validation in forced_photometry.py
    ↓
Plane 1: Image quality gate in source_finding.py        ← independent of Plane 2
    ↓
Plane 3: Source-finding completeness + size QA          ← depends on Plane 1 infrastructure
    ↓
Cross-cutting: epoch_log.py + wiring in all stages      ← depends on Planes 1-3 for data
    ↓
Plane 4: Variability metrics wiring in plot_lightcurves.py  ← independent
```

Planes 1 and 4 are independent and could be built in parallel. The epoch log should come last since it just wires the outputs of the other planes.

---

## Test Strategy

Each plane gets its own test file with 5–8 CI-runnable tests using synthetic FITS and mocked catalog queries. No network, no casatools, no real data required.

| Test file | Tests |
|-----------|-------|
| `tests/test_image_gate.py` | 6 tests: dynamic range, RMS ratio, pixel coverage, beam sanity, PASS/WARN/FAIL overall |
| `tests/test_noise_model_fix.py` | 3 tests: 96-antenna prediction, ratio computation, gate thresholds |
| `tests/test_source_finding_qa.py` | 5 tests: completeness PASS/FAIL/WARN, size distribution flags, integration with `run_source_finding` |
| `tests/test_variability_wiring.py` | 4 tests: metrics computed, CSV written, candidates flagged, summary dict correct |
| `tests/test_epoch_log.py` | 4 tests: append writes valid JSONL, multiple appends accumulate, readable by pandas, graceful on missing dir |

Target: **+22 tests** → total suite **64 tests**.

---

## File Map

| File | Action | Plane |
|------|---------|-------|
| `dsa110_continuum/qa/noise_model.py` | Modify — fix `num_antennas`, add bandwidth comment | Bug fix |
| `dsa110_continuum/photometry/epoch_qa.py` | Modify — update `QA_RMS_LIMIT_MJY` | Bug fix |
| `dsa110_continuum/qa/image_gate.py` | **Create** — `ImageQAResult`, `check_image_quality_for_source_finding` | Plane 1 |
| `scripts/source_finding.py` | Modify — call `check_image_quality_for_source_finding` | Plane 1 |
| `scripts/forced_photometry.py` | Modify — call `validate_noise_prediction`, extend return dict | Plane 2 |
| `dsa110_continuum/source_finding/core.py` | Modify — add `CompletenessResult`, `check_source_completeness`, `SizeQAResult`, `check_size_distribution` | Plane 3 |
| `scripts/plot_lightcurves.py` | Modify — wire `compute_metrics`, write `variability_summary.csv` | Plane 4 |
| `dsa110_continuum/qa/epoch_log.py` | **Create** — `append_epoch_qa` | Cross-cutting |
| `scripts/forced_photometry.py` | Modify — call `append_epoch_qa` | Cross-cutting |
| `scripts/source_finding.py` | Modify — call `append_epoch_qa` | Cross-cutting |
| `scripts/stage_c_crossmatch.py` | Modify — call `append_epoch_qa` | Cross-cutting |
| `tests/test_image_gate.py` | **Create** — 6 tests | Plane 1 |
| `tests/test_noise_model_fix.py` | **Create** — 3 tests | Bug fix |
| `tests/test_source_finding_qa.py` | **Create** — 5 tests | Plane 3 |
| `tests/test_variability_wiring.py` | **Create** — 4 tests | Plane 4 |
| `tests/test_epoch_log.py` | **Create** — 4 tests | Cross-cutting |
