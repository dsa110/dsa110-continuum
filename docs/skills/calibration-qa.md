# Calibration QA and Pipeline Provenance

> **Status**: Current as of 2026-03-14
> **Key files**: `dsa110_continuum/calibration/qa.py`, `dsa110_continuum/qa/provenance.py`, `scripts/batch_pipeline.py`
> **See also**: [Calibration](calibration.md) | [Epoch QA](../drafts/02-epoch-qa.md) | [Outputs](../drafts/03-outputs-and-artifacts.md)

## Why this exists

All mosaics except Jan 25 were catastrophically bad — heavy striping, edge
artifacts, 99% of catalog sources undetectable. But there was **no way to
diagnose why**: no record of which calibration tables were used, what their
quality was, which tiles went in, or what happened at each stage.

The calibration QA infrastructure was already built (`calibration/qa.py`
had `compute_calibration_metrics()` and `assess_calibration_quality()`)
but was never called from the pipeline. The fix was wiring — not new
science code.

---

## Architecture overview

Calibration quality is assessed at three layers:

```
Layer 1: Metric extraction         compute_calibration_metrics()
         Opens CASA cal table via casacore, extracts flag fraction,
         phase scatter, amplitude stats, SNR → CalibrationMetrics

Layer 2: Threshold assessment      assess_calibration_quality()
         Runs Layer 1 on each cal table, checks against QAThresholds,
         grades as excellent/good/marginal/poor/failed → CalibrationQAResult

Layer 3: Pipeline integration      RunManifest.assess_cal_quality()
         Called from batch_pipeline.py after cal-table validation.
         Stores results in JSON manifest and FITS headers.
```

---

## Layer 1: Metric extraction

### `compute_calibration_metrics(caltable_path, cal_type=None)`

**File**: `dsa110_continuum/calibration/qa.py:200`

Opens a CASA calibration table via casacore (`casacore.tables`) and
extracts comprehensive statistics from the solutions. Returns a
`CalibrationMetrics` dataclass.

#### What it reads

| Column | Content | Table types |
|--------|---------|-------------|
| `CPARAM` | Complex gain solutions | Bandpass (`.b`), Gain (`.g`) |
| `FPARAM` | Float parameter solutions | Delay (K) |
| `FLAG` | Per-solution flags (set by solver) | All |
| `SNR` | Per-solution signal-to-noise | All (if CASA wrote it) |
| `ANTENNA1` | Antenna index | All |
| `SPECTRAL_WINDOW_ID` | Spectral window index | All |

#### What it computes

| Metric | Derivation | Meaning |
|--------|------------|---------|
| `flag_fraction` | `n_flagged / n_solutions` | Fraction of solutions the solver couldn't find. > 30% is a warning |
| `phase_scatter_deg` | `std(angle(CPARAM))` on unflagged data | Stability of phase solutions. > 30° means phases are noisy |
| `mean_amplitude` | `mean(abs(CPARAM))` on unflagged data | For normalized bandpass: should be near 1.0 |
| `std_amplitude` | `std(abs(CPARAM))` on unflagged data | Amplitude variation across antennas/channels |
| `median_snr` | `median(SNR)` on unflagged data | Solver confidence. < 3.0 means fitting noise |
| `n_antennas` | `len(unique(ANTENNA1))` | Sanity check: should be ~117 for DSA-110 |
| `n_spws` | `len(unique(SPECTRAL_WINDOW_ID))` | Sanity check: should be 16 for DSA-110 |
| `n_channels` | Inferred from `CPARAM` shape | 48 per SPW for DSA-110 |

#### Cal-type auto-detection

When `cal_type` is not provided, it is inferred from the filename:

| Pattern | Detected type |
|---------|--------------|
| `_bp.` or `bpcal` in path | `"bp"` (bandpass) |
| `_g.` or `gcal` or `gpcal` in path | `"g"` (gain) |
| `_k.` or `kcal` in path | `"k"` (delay) |
| Path ends with `.b` | `"bp"` (DSA-110 convention) |
| Path ends with `.g` | `"g"` (DSA-110 convention) |

The `.b`/`.g` suffix rules are DSA-110-specific — the default CASA naming
(`_bp.`, `_g.`) doesn't match the DSA-110 naming convention
(`{date}T22:26:05_0~23.b` and `.g`).

#### Error handling

If the table doesn't exist or can't be read, the function returns a
`CalibrationMetrics` with `extraction_error` set and `is_valid == False`.
It never raises — callers check `.is_valid` or `.extraction_error`.

#### Usage

```python
from dsa110_continuum.calibration.qa import compute_calibration_metrics

metrics = compute_calibration_metrics(
    "/stage/dsa110-contimg/ms/2026-01-25T22:26:05_0~23.b"
)

if metrics.is_valid:
    print(f"Type:          {metrics.cal_type}")        # "bp"
    print(f"Flag fraction: {metrics.flag_fraction:.1%}")
    print(f"Phase scatter: {metrics.phase_scatter_deg:.1f}°")
    print(f"Median SNR:    {metrics.median_snr}")
    print(f"Amplitude:     {metrics.median_amplitude:.3f} ± {metrics.std_amplitude:.3f}")
    print(f"Antennas:      {metrics.n_antennas}")
    print(f"Channels:      {metrics.n_channels}")
else:
    print(f"ERROR: {metrics.extraction_error}")

# Serialize for JSON
d = metrics.to_dict()
```

---

## Layer 2: Threshold assessment

### `assess_calibration_quality(ms_path, thresholds=None, caltables=None)`

**File**: `dsa110_continuum/calibration/qa.py:338`

Runs `compute_calibration_metrics()` on each cal table associated with
an MS and checks against configurable thresholds.

#### Default thresholds (`QAThresholds`)

| Metric | Default | Severity |
|--------|---------|----------|
| `max_flag_fraction` | 0.3 (30%) | > 30% = warning; > 50% = error |
| `max_phase_scatter_deg` | 30.0° | > 30° = warning |
| `min_snr` | 3.0 | Median SNR < 3.0 = error |
| `min_amplitude` | 0.1 | Mean amplitude < 0.1 = warning |
| `max_amplitude` | 10.0 | Max amplitude > 10.0 = warning |
| Non-finite values | any NaN/Inf | = error |

#### Overall grading

Based on the average flag fraction across all tables:

| Average flag fraction | Grade |
|-----------------------|-------|
| < 10% | `excellent` |
| 10–20% | `good` |
| 20–30% | `marginal` |
| > 30% | `poor` |
| Any error-level issue | `failed` |

#### Result structure (`CalibrationQAResult`)

```python
result = assess_calibration_quality(
    ms_path="/stage/dsa110-contimg/ms/2026-01-25T22:26:05.ms",
    caltables={
        "bp": "/stage/dsa110-contimg/ms/2026-01-25T22:26:05_0~23.b",
        "g":  "/stage/dsa110-contimg/ms/2026-01-25T22:26:05_0~23.g",
    },
)

print(f"Passed:  {result.passed}")         # True/False
print(f"Grade:   {result.overall_grade}")   # "excellent", "good", etc.
print(f"Severity:{result.severity}")        # "success", "warning", "error"

for issue in result.issues:
    print(f"  [{issue.severity}] {issue.message}")
    # e.g. "[warning] BP table: High phase scatter (32.1° > 30.0°)"

# Serializable
d = result.to_dict()
```

#### Persistent storage (`CalibrationQAStore`)

Results can be stored in SQLite via `CalibrationQAStore` for trending:

```python
from dsa110_continuum.calibration.qa import get_qa_store

store = get_qa_store()
store.save_result(result)

# Query later
latest = store.get_result(ms_path)
recent = store.list_recent(limit=10, failed_only=True)
stats = store.get_summary_stats()
```

---

## Layer 3: Pipeline provenance manifest

### `RunManifest`

**File**: `dsa110_continuum/qa/provenance.py`

A dataclass that accumulates provenance throughout a `batch_pipeline.py` run
and serializes to a single JSON file alongside the pipeline products.

#### Lifecycle in `batch_pipeline.py`

```
1. After arg parsing + cal-table validation:
   manifest = RunManifest.start(date, cal_date)
   manifest.assess_cal_quality(_bp, _ga)          ← calls compute_calibration_metrics()

2. After MS list built:
   manifest.ms_files = list(ms_list)

3. After epoch gaincal:
   manifest.gaincal_status = _epoch_gaincal_status
   manifest.epoch_g_table = _epoch_g_table

4. Inside tile loop (per tile):
   manifest.record_tile(ms_path, fits_path, status, elapsed_sec, error=...)

5. After all epochs:
   manifest.record_epoch(hour, epoch_result_dict)

6. After print_summary:
   manifest.finalize(wall_time_sec)
   manifest.save(products_dir)
```

#### Manifest JSON structure

```json
{
  "git_sha": "da94269",
  "started_at": "2026-02-12T04:30:00+00:00",
  "finished_at": "2026-02-12T06:15:42+00:00",
  "wall_time_sec": 6342.1,
  "command_line": ["batch_pipeline.py", "--date", "2026-02-12", "--cal-date", "2026-01-25"],
  "hostname": "h17",
  "date": "2026-02-12",
  "cal_date": "2026-01-25",
  "bp_table": "/stage/dsa110-contimg/ms/2026-01-25T22:26:05_0~23.b",
  "g_table": "/stage/dsa110-contimg/ms/2026-01-25T22:26:05_0~23.g",
  "epoch_g_table": "/stage/.../epoch_gaincal/2026-02-12_epoch.ap.G",
  "gaincal_status": "ok",
  "ms_files": ["...", "..."],
  "cal_quality": {
    "bp": {
      "cal_type": "bp",
      "flag_fraction": 0.12,
      "phase_scatter_deg": 8.3,
      "mean_amplitude": 1.02,
      "median_snr": 45.2,
      "n_antennas": 117,
      "n_spws": 16,
      "n_channels": 48,
      "extraction_error": null
    },
    "g": {
      "cal_type": "g",
      "flag_fraction": 0.08,
      "phase_scatter_deg": 25.3,
      "mean_amplitude": 0.98,
      "median_snr": 38.7,
      "n_antennas": 117,
      "extraction_error": null
    }
  },
  "tiles": [
    {"ms_path": "...", "fits_path": "...", "status": "ok", "elapsed_sec": 142.3},
    {"ms_path": "...", "fits_path": null, "status": "failed", "elapsed_sec": 1800.0, "error": "timeout or crash"}
  ],
  "epochs": [
    {
      "hour": 0,
      "n_tiles": 13,
      "status": "ok",
      "mosaic_path": "...",
      "peak": 0.52,
      "rms": 0.0034,
      "n_sources": 87,
      "median_ratio": 0.95,
      "qa_result": "PASS",
      "rms_mjy": 3.4,
      "completeness_frac": 0.82
    }
  ]
}
```

#### Output location

```
/data/dsa110-proc/products/mosaics/{date}/
    {date}_manifest.json          ← provenance manifest
    {date}_run_summary.json       ← execution summary (moved from /tmp)
    {date}T{HH}00_mosaic.fits     ← FITS headers enriched with provenance
    {date}T{HH}00_forced_phot.csv
    {date}T{HH}00_mosaic_qa_diag.png
```

A backward-compatibility symlink is maintained at
`/tmp/pipeline_last_run.json` pointing to the latest run summary.

---

## FITS header provenance cards

Every epoch mosaic FITS file written by `batch_pipeline.py` includes
provenance and QA header keywords:

### Written at mosaic creation time

| Keyword | Example | Description |
|---------|---------|-------------|
| `PIPEVER` | `da94269` | Pipeline git commit hash |
| `CALDATE` | `2026-01-25` | Date of calibration tables used |
| `NTILES` | `13` | Number of input tiles (including overlap) |
| `BPFLAG` | `0.12` | Bandpass table flagged fraction |
| `GPHSCTR` | `8.3` | Gain table phase scatter (degrees) |

### Updated in-place after epoch QA

| Keyword | Example | Description |
|---------|---------|-------------|
| `QARESULT` | `FAIL` | Overall epoch QA verdict (`PASS` / `FAIL`) |
| `QARMS` | `15.1` | Mosaic RMS (mJy/beam) |
| `QARAT` | `0.92` | Median DSA/catalog flux ratio |

These can be inspected with standard FITS tools:

```bash
# Quick header check
python -c "from astropy.io import fits; h=fits.getheader('mosaic.fits'); print(h['QARESULT'], h['CALDATE'], h['BPFLAG'])"
```

---

## Interpreting cal quality for diagnosis

### What "phase scatter" tells you

Phase scatter is `std(angle(gains))` across all unflagged solutions. It is
the most important single metric for predicting mosaic quality.

| Phase scatter | Interpretation |
|---------------|----------------|
| < 10° | Excellent — solutions are stable |
| 10–20° | Good — typical for DSA-110 same-date cal |
| 20–30° | Marginal — expect some flux loss |
| > 30° | Poor — significant coherence loss, mosaics will be degraded |
| > 60° | Effectively random phases — cross-date misapplication signature |

**Why cross-date calibration fails**: DSA-110 is a transit array. Phase
solutions encode the ionosphere + instrumental delays at the moment of
observation. Applied to a different date, the phases are essentially random
offsets. When visibilities are coherently averaged with random phase errors,
amplitude drops as ~1/sqrt(N). This explains why cross-date runs show
median DSA/NVSS flux ratios of ~0.06.

### What "flag fraction" tells you

| Flag fraction | Interpretation |
|---------------|----------------|
| < 10% | Excellent — clean data, solver converged well |
| 10–20% | Good — some RFI or bad antennas |
| 20–30% | Marginal — significant data loss, check for systematic RFI |
| 30–50% | Poor — solver struggling; may need manual flagging |
| > 50% | Failed — majority of solutions flagged; tables unreliable |

### Diagnosing a bad mosaic from the manifest

Open `{date}_manifest.json` and check:

1. **`cal_quality.bp.flag_fraction`** and **`cal_quality.g.flag_fraction`**
   — Are they above 30%? If so, the cal tables themselves are poor quality.

2. **`cal_quality.g.phase_scatter_deg`**
   — Is it above 30°? If so, gain phases are too noisy for coherent imaging.

3. **`cal_date` vs `date`**
   — Are they different? Cross-date calibration transfers amplitude but not
   phase. Unless epoch gaincal succeeded, expect degraded results.

4. **`gaincal_status`**
   — Is it `"ok"`, `"fallback"`, `"skipped"`, or `"error"`?
   `"fallback"` or `"error"` means per-epoch gain calibration failed and
   the static daily G table was used instead.

5. **`tiles`**
   — How many have `"status": "failed"`? Many tile failures suggest CASA
   instability or resource exhaustion.

6. **`epochs[].qa_result`**
   — Are they `"PASS"` or `"FAIL"`? Cross-reference with `rms_mjy` and
   `median_ratio` to identify the failure mode.

---

## Key files

| File | Purpose |
|------|---------|
| `dsa110_continuum/calibration/qa.py` | `compute_calibration_metrics()`, `assess_calibration_quality()`, `CalibrationMetrics`, `CalibrationQAResult`, `CalibrationQAStore` |
| `dsa110_continuum/qa/provenance.py` | `RunManifest` dataclass — accumulates and serializes pipeline provenance |
| `scripts/batch_pipeline.py` | Wiring: creates manifest, records cal quality / tiles / epochs, writes FITS headers |
| `tests/test_provenance.py` | Tests for manifest roundtrip, missing cal table handling, cal-type detection |
