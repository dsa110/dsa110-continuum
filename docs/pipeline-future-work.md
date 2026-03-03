# DSA-110 Continuum Pipeline: Future Work

Last updated: 2026-02-28
Status of items already implemented is noted inline.

---

## 1. Pipeline Resilience (High Priority)

### 1a. Per-tile timeout + kill   [DONE — 2026-02-28]
Hard timeout on applycal/WSClean per tile using ThreadPoolExecutor.
If a tile exceeds the limit, CASA and WSClean are killed and the tile
is skipped (or retried). Flag: --tile-timeout SECONDS (default 1800).

### 1b. Checkpoint file   [DONE — 2026-02-28]
After every successful tile, write .tile_checkpoint.json to the stage dir.
On restart without --start-hour, completed tiles are picked up automatically
so the full tile list is available for mosaicking even after a crash.

### 1c. Single-retry on failure   [DONE — 2026-02-28]
Each failed tile gets one retry after a 60s cool-down.
Flag: --retry-failed. Off by default to avoid doubling wall time on
systemic (not transient) failures.

### 1d. Root-cause logging for CASA subprocess failures   [NOT DONE]
Currently if CASA dies the parent sees None with no diagnostic.
Worth capturing: exit code, stderr, dmesg OOM events, and CASA's
own casapy.log. These should be copied to the stage dir per tile
so post-mortem analysis is possible without re-running.

### 1e. Memory pressure guard   [PARTIAL — memory_safe decorator exists]
The @memory_safe(max_system_gb=6.0) decorator in applycal.py checks
available RAM before running CASA. If the system is under pressure
it waits and retries. In practice this doesn't prevent OOM kills
if RAM degrades mid-task. A better approach: spawn a watchdog thread
that polls psutil every 30s and pre-empts the tile if free RAM < 2 GB.

---

## 2. Calibration Traceability (High Priority)

### 2a. Cal-date validation at startup   [NOT DONE]
Before processing any tile, verify that the BP/gain tables listed in
--cal-date actually exist on disk and are not symlinks to an unexpected
date. A 5-line check at the top of main() would have caught the
Jan-25-tables-applied-to-Feb-12 problem immediately.

### 2b. Cal provenance in FITS header   [NOT DONE]
Write CALDATE and CALTBL FITS header keywords to every tile and mosaic.
This makes it trivial to audit which calibration was used for any image
without consulting log files.

### 2c. Automated calibration solution QA   [NOT DONE]
Before imaging, run a quick bandpass amplitude check: median BP gain
across all antennas should be close to 1.0. Flag suspiciously low or
high solutions (>3-sigma outliers) and abort if more than N% of
antennas are bad. This catches wrong-cal-date issues before they
propagate to the mosaic.

---

## 3. Observability & Dashboards (Medium Priority)

### 3a. Live QA dashboard   [DONE — 2026-02-28]
FastAPI server on port 8767 serving per-epoch metrics (peak, RMS,
dynamic range, DSA/NVSS ratio, pass/fail) and mosaic thumbnails.
Accessible via Tailscale or Cloudflare tunnel. Auto-refreshes 60s.

### 3b. Post-run summary notification   [NOT DONE]
When the pipeline finishes (or dies), send a text/Slack message with:
  - Epochs processed, pass/fail count, median DSA/NVSS ratio
  - Link to QA dashboard
  - Wall time
Currently you have to check the log manually.

### 3c. Per-tile timing histogram   [NOT DONE]
Log min/median/max tile processing time per run. Useful for detecting
when CASA is slower than usual (memory pressure, disk I/O contention).

### 3d. Persistent QA database   [NOT DONE — see note on VAST approach]
VAST uses a Django/Postgres state machine. For DSA-110 this is overkill
today, but if the number of epochs grows to hundreds, a lightweight
SQLite database storing per-epoch metrics would allow:
  - Trend analysis (is DSA/NVSS ratio drifting over weeks?)
  - Automated flagging of epochs that regress
  - Historical comparison across re-runs
A pandas DataFrame serialized to Parquet is a good intermediate step
before committing to a full DB.

---

## 4. Pipeline Architecture (Lower Priority / Longer Term)

### 4a. True parallel tile processing   [NOT DONE]
Right now tiles are processed serially. H17 likely has enough CPU/RAM
for 2-4 concurrent tiles. A ProcessPoolExecutor with max_workers=2
would roughly halve wall time. Requires careful management of CASA
temp directories and log files per worker.

### 4b. Idempotent re-run from any stage   [PARTIAL]
Currently: tile skip-if-exists works; mosaic skip-if-exists works;
photometry is always re-run. Photometry should also be skippable if
CSV exists and mosaic hasn't changed (check mtime).

### 4c. Dagster integration   [DEFERRED — existing code has too many flaws]
The dsa110-contimg Dagster repo has the right skeleton (asset graph,
per-epoch ops) but the actual QA views are broken. Worth a ground-up
rewrite of just the Dagster asset definitions to wrap batch_pipeline.py
as atomic ops. Benefit: UI shows asset lineage, re-materialization is
one click, and run history is queryable.

### 4d. Containerization   [NOT DONE]
The pipeline depends on CASA6 in a conda env, WSClean, and several
DSA-specific Python packages. A Docker/Apptainer image would make
the pipeline reproducible on other machines and simplify dependency
management. Important if DSA-110 data ever moves to a cloud compute
environment.

---

## 5. Data Quality (Science-facing)

### 5a. Automated source finding and cross-match   [PARTIAL]
Currently doing forced photometry against NVSS. Consider adding a
PyBDSF or Aegean blind source-finding step to catch sources that are
in DSA but not NVSS (new transients, steep-spectrum sources).

### 5b. Ionospheric flagging   [NOT DONE]
During periods of high ionospheric activity, some tiles will have
anomalously high noise or position offsets. An automated iono-quality
metric (e.g. median position offset of bright point sources vs NVSS)
would allow auto-flagging bad tiles before mosaicking.

### 5c. Multi-epoch flux monitoring   [NOT DONE]
Once multiple epochs are processed correctly, the forced-photometry
CSVs can be stacked to build a light curve for every NVSS source in
the field. A simple pandas merge across dates gives variability metrics.
This is the core science product; worth building a light-curve viewer
into the QA dashboard.

