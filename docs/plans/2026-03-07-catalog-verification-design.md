# Catalog Verification + Gaincal Fallback + Multi-Epoch Sprint Design

**Date:** 2026-03-07  
**Status:** Approved

---

## Goal

Produce reliable DSA/NVSS flux ratios for every epoch mosaic, gate multi-epoch processing on those ratios, and prevent the epoch gaincal from destroying pristine bandpass calibration on faint RA fields.

---

## Diagnosis

Feb 15 2026 diagnostic re-run (post-spwmap fix) confirmed:

| Metric | Value |
|--------|-------|
| Epoch | 2026-02-15T0000 |
| Mosaic peak | 0.116 Jy/beam |
| Global RMS | 10.0 mJy/beam |
| NVSS sources in footprint | 187 |
| S/N ≥ 3 detections | 10 |
| Median DSA/NVSS ratio | **0.258** |
| Target ratio | 0.8–1.2 |

The 4× flux suppression is caused by the amplitude+phase `gaincal` step in `epoch_gaincal.py` operating at insufficient SNR. When the solver cannot distinguish the faint sky model from thermal noise, it depresses the antenna gains, directly suppressing the reconstructed flux. The spwmap fix (preventing SPW 1–15 flagging) recovered from 0.000 → 0.258 but did not resolve the underlying SNR problem.

**Root-cause confirmation (physics):** In a stable instrument like DSA-110, the daily bandpass table derived from a bright calibrator (~13 Jy) already captures delays, frequency-dependent gains, and base phase offsets. For a faint RA field with distributed flux (~4 Jy across 13 sources), an additional per-epoch `ap` solve produces solutions with SNR < 3 per antenna, flagging >30% of solutions and corrupting the calibration already provided by the bandpass. Falling back to bandpass-only for such fields is standard survey pipeline practice.

---

## Architecture: Three Pieces

### Piece 1 — `scripts/verify_sources.py`

A footprint-aware, catalog-driven source verification script. Replaces the current forced-photometry logic (which returns pixel values outside the valid mosaic area, causing nonsensical ratios).

**Key design choices:**
- Uses `master_sources.sqlite3` (1.8M unified NVSS+FIRST+RACS+VLASS positions) for continuum sources
- Uses `atnf_full.sqlite3` (4351 pulsars) for pulsar passes
- Footprint filter: converts each catalog position to pixels, rejects NaN (primary-beam-blanked) pixels before measuring
- Simple `nanmax` in a ±5 pixel box (no GPU, no Condon convolution) — reliable for a median-ratio diagnostic
- Sources below 3σ are kept as upper limits (`is_upper_limit=True`), not discarded
- Prints one machine-parseable QA line: `VERIFY PASS/WARN/FAIL: median_ratio=X.XXX n_sources=NNN`

**New files:**
- `dsa110_continuum/photometry/footprint.py` — `load_mosaic()`, `sources_in_footprint()`
- `dsa110_continuum/photometry/simple_peak.py` — `measure_peak_box()`
- `scripts/verify_sources.py` — CLI orchestrator

**Output columns:**
```
source_name | ra_deg | dec_deg | catalog_flux_jy | dsa_peak_jyb
snr | ratio | source_type | is_upper_limit | catalog
```

**QA gate for Piece 3:** median ratio of S/N ≥ 3 continuum detections must be ≥ 0.70.

---

### Piece 2 — Gaincal Flag-Fraction Monitor + Bandpass-Only Fallback

Modified `epoch_gaincal.py`: after the `p.G` solve, read the FLAG column from the CASA table directly and compute the flagged fraction. If `flag_fraction > 0.30`, abort the epoch gaincal and return `None` so the batch pipeline applies bandpass-only.

**CASA implementation:**
```python
import casatools as cto

tb = cto.table()
tb.open(p_table)
flags = tb.getcol("FLAG")   # shape: (n_pol, n_spw, n_rows) — booleans
tb.close()
flag_fraction = flags.sum() / flags.size

if flag_fraction > GAINCAL_FLAG_FRACTION_LIMIT:   # 0.30
    log.warning(
        "Epoch gaincal [%s]: p.G flagged %.1f%% of solutions (limit %.0f%%). "
        "SNR too low — returning None (pipeline will apply BP-only).",
        stem, flag_fraction * 100, GAINCAL_FLAG_FRACTION_LIMIT * 100,
    )
    return None
```

**Constant to add at module top:**
```python
GAINCAL_FLAG_FRACTION_LIMIT = 0.30
```

**Batch pipeline behaviour (already correct):** `calibrate_epoch()` returning `None` causes `batch_pipeline.py` to skip the `ap.G` applycal and apply only `bp_table`. No changes needed there.

**Test to add:** `test_gaincal_returns_none_when_p_table_heavily_flagged` — mock `tb.getcol("FLAG")` returning an array that is 35% True; assert `calibrate_epoch()` returns `None`.

---

### Piece 3 — Multi-Epoch Batch Run

Run `batch_pipeline.py` on all available dates at the Dec=+16° strip. Gate advancement per epoch on `verify_sources.py` median ratio ≥ 0.70.

**No new code required.** This is an operational step: run the pipeline, then run `verify_sources.py` on each output mosaic.

**Script to add:** `scripts/batch_run_all_dates.sh` — a simple shell loop that calls `batch_pipeline.py` for each indexed date, then `verify_sources.py` on the output mosaic, and appends a line to `products/qa_summary.csv`.

**`products/qa_summary.csv` schema:**
```
date | epoch_utc | mosaic_path | median_ratio | n_detections | gaincal_used | qa_result
```

---

## Data Flow

```
batch_pipeline.py (per date)
  │
  ├── epoch_gaincal.py
  │     ├── p.G solve
  │     ├── FLAG fraction check  ← NEW (Piece 2)
  │     │     ├── > 30%: return None  (BP-only path)
  │     │     └── ≤ 30%: continue → ap.G
  │     └── return ap_table (or None)
  │
  ├── apply cal + WSClean imaging
  │
  ├── mosaic builder
  │
  └── [after each epoch mosaic] verify_sources.py  ← NEW (Piece 1)
        ├── master_sources.sqlite3  (continuum)
        ├── atnf_full.sqlite3       (pulsars)
        ├── footprint filter
        ├── peak-in-box measurement
        └── products/qa_summary.csv  ← NEW (Piece 3)
```

---

## Files to Create / Modify

| File | Action |
|------|--------|
| `dsa110_continuum/photometry/footprint.py` | CREATE |
| `dsa110_continuum/photometry/simple_peak.py` | CREATE |
| `scripts/verify_sources.py` | CREATE |
| `dsa110_continuum/calibration/epoch_gaincal.py` | MODIFY — add `GAINCAL_FLAG_FRACTION_LIMIT`, flag-fraction check after p.G solve |
| `tests/test_epoch_gaincal.py` | MODIFY — add flag-fraction fallback test |
| `tests/test_verify_sources.py` | CREATE |
| `scripts/batch_run_all_dates.sh` | CREATE |

---

## Success Criteria

1. `verify_sources.py` run on Jan 25 02h mosaic returns median ratio 0.8–1.2 with ≥ 20 S/N≥3 detections.
2. `epoch_gaincal.py` returns `None` (logs "BP-only") for Feb 15 (expected >30% p.G flagging).
3. A re-run of Feb 15 with BP-only applied produces a mosaic with `verify_sources.py` median ratio ≥ 0.70.
4. Multi-epoch batch run on ≥ 3 dates produces `qa_summary.csv` with at least 2 QA-passing epochs.

---

## Deferred

- Pulsar flux calibration: ATNF pulsars with known S1400 flux will be matched and checked, but no special pulsar-optimised photometry (gating, dedispersion) is in scope.
- Per-catalog ratio breakdown (separate NVSS vs. RACS vs. FIRST medians): deferred to post-sprint.
- VLASS-based sky model for gaincal: deferred (VLASS DB exists but not yet wired into `epoch_gaincal.py::populate_model_data`).
