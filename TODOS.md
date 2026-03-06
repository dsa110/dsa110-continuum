
Last updated: 2026-03-04

---

# Roadmap: Multi-Day Light Curves

Goal: for each source in the reference catalog, a flux measurement (or upper limit) exists
for every epoch that passes QA, with variability metrics computed.

---

## Phase 1 — Reliable single-epoch pipeline (foundation)

- [x] **Primary beam correction**: `pbcor=True` confirmed working. 02h epoch forced photometry
  (2026-02-27): median DSA/NVSS ratio = **0.893** on 1047 NVSS sources — within the 0.8–1.2 target.

- [x] **Re-run conversion for 3 corrupt MS files**: Timestamps `2026-01-25T22:20:55`, `22:48:11`,
  `22:53:20` — re-converted 2026-02-26. FIELD tables now present.

- [x] **mosaic_day.py cleanup**: `--keep-intermediates` flag, MS cleanup after tile success,
  mosaic moved to `products/mosaics/{DATE}/` on QA pass.

- [x] **Dec-strip awareness (Phase 1)**: Added `read_ms_dec()` utility
  (`dsa110_continuum/calibration/dec_utils.py`), dec-strip guard in
  `batch_pipeline.py` (aborts if observed Dec differs from cal-table strip by >5°,
  `--expected-dec` flag, default 16.1°), and `dec_deg` column in `inventory.py`
  output. Commits: `5188fa6`, `ad20e92`, `7bfa57b`.

- [x] **02h test epoch run (2026-01-25)**: 11 tiles, peak 7.79 Jy/beam, RMS 11.9 mJy/beam,
  DR 656, QA PASSED. Output: `/stage/dsa110-contimg/images/mosaic_2026-01-25/2026-01-25T0200_mosaic.fits`.

- [ ] **Resolve flux ratio regression**: Re-run Jan 25 02h epoch after duplicate-package removal to
  confirm ratio is back at 0.893. If not, bisect: compare CORRECTED_DATA amplitudes before/after
  applycal, check `detect_datacolumn()` fallback, verify pbcor FITS is being read (not dirty image).

- [ ] **Complete import rename**: 828 references in `dsa110_continuum/` still import from
  `dsa110_contimg.core.*`. Pipeline silently uses stale code if the old package is installed.
  Do a mass rename; confirm 120/122 module imports pass under casa6 env after.

- [ ] **Make data paths configurable**: Replace hardcoded `/stage/dsa110-contimg/ms/` and
  `/data/incoming/` across all scripts with a single config file or environment variable.
  Required before running on any new machine or date range.

- [ ] **Investigate leftmost mosaic tiles**: First 1–2 tiles are significantly noisier.
  Check observation log for low elevation, heavy RFI flags, or data quality at those timestamps.

- [ ] **Clarify noise consistency threshold**: The 3.0 ratio in mosaic QA has unclear origin.
  Verify against VAST pipeline or radio survey literature; document in `docs/reference/mosaicking.md`.

---

## Phase 2 — Data inventory + multi-epoch automation

- [ ] **Data inventory script**: Scan `/data/incoming/`, count HDF5 files by date and subband,
  flag incomplete observations (missing subbands for a timestamp), report total volume and
  per-date readiness. ~3 TB on disk currently, more to copy from other machines.

- [ ] **Per-date calibration**: Document and automate generating or symlinking cal tables for each
  new date. Standard procedure: symlink `_0~23.b` and `_0~23.g` from `2026-01-25`, then run
  `batch_pipeline.py --cal-date 2026-01-25`. Flag that cross-date phase cal degrades dynamic range;
  note where per-date gain solutions become necessary.

- [ ] **Per-strip calibration tables**: Derive B/G solutions for Dec=+33° and +54°
  strips (currently blocked — only +16° tables exist). Until then,
  `batch_pipeline.py --expected-dec 33.0` or `54.0` will abort at the dec-strip
  guard. Procedure: run full calibration workflow for a night observed at each strip;
  store tables at `/stage/dsa110-contimg/ms/{date}T22:26:05_0~23.{b,g}`.

- [ ] **Automated multi-epoch batch run**: Run `batch_pipeline.py` on all available dates with QA
  gating — only advance epochs where DSA/NVSS ratio is 0.8–1.2. Log pass/fail and ratio per epoch
  to a summary CSV at `products/qa_summary.csv`.

- [ ] **Handle DSA-110 elevation slews** *(detection implemented; routing pending)*: The dec-strip guard in `batch_pipeline.py` now detects slews and aborts with a clear error if observed Dec differs from the expected cal-table strip by >5°. Remaining work: (1) derive per-strip cal tables, (2) implement automatic strip routing so the pipeline selects the correct table set without manual intervention.

- [ ] **Bulk data processing**: Once inventory is complete and Phase 1 regressions are resolved,
  run full pipeline on all available dates.

---

## Phase 3 — Reference source catalog

- [ ] **Source finding on best epoch**: Run BANE + Aegean on the highest-DR mosaic to produce a
  blind source catalog (see `scripts/source_finding.py`). Record detection threshold and mosaic
  date/epoch used.

- [ ] **Cross-match with NVSS/RACS**: Match blind catalog against NVSS and RACS. Output a reference
  catalog with: source ID, RA/Dec, NVSS/RACS flux, spectral index (where available), angular
  separation from DSA detection. Use this for forced photometry and flux-scale validation.

---

## Phase 4 — Forced photometry across all epochs

- [ ] **Per-epoch forced photometry**: Apply reference catalog positions to every QA-passing epoch
  mosaic. Output: one CSV per epoch at `products/mosaics/{date}/{date}T{HH}00_forced_phot.csv`
  with columns: source_id, ra, dec, flux_Jy, local_rms, SNR, epoch_utc, mosaic_path.

- [ ] **Per-source per-epoch QA flags**: Add flag column to photometry CSV for: local RMS unusually
  high (> 2× median), source within 1 beam of mosaic edge, source blended with bright neighbor.
  Flagged rows are included but excluded from variability analysis.

---

## Phase 5 — Light curve extraction

- [ ] **Stack photometry CSVs**: Merge all per-epoch CSVs by source_id into a single light curve
  table (source × epoch). Store as `products/lightcurves/lightcurves_{date_range}.csv` and as a
  wide-format Parquet for fast per-source slicing.

- [ ] **Variability metrics**: For each source, compute:
    - Mooley modulation index *m* = σ_S / ⟨S⟩
    - Variability significance *Vs* = (S_max − S_min) / sqrt(σ_max² + σ_min²)
    - Reduced chi-squared *η* against constant-flux null
  Use formulas from `docs/reference/vast-crossref.md`. Flag sources where *Vs* > 4.0 or *η* > 2.5.

- [ ] **Light curve visualization**: Per-source flux vs. time plots. Highlight variable candidates.
  Save to `products/lightcurves/plots/{source_id}.png`. Produce a summary page with the top-N most
  variable sources ranked by *η*.

---

## Deferred / low priority

- [ ] **Build VLASS SQLite catalog database**: The epoch gaincal tile-selection and sky model steps
  query VLASS via SQLite but the database has not been built on H17. RACS is also missing.
  Run the catalog build tools for both catalogs (dec ≈ +16°, RA range 0–360°) and place the
  resulting `.db` files in the expected location. Until then, gaincal falls back to NVSS+FIRST
  only (8 sources), which is sufficient but degrades sky model completeness.


- [ ] **GPU calibration**: `dsa110_continuum/calibration/gpu_calibration.py` exists but is untested
  end-to-end. Profile whether GPU bandpass solve gives meaningful speedup on the 110-antenna array.

- [ ] **Spectral index maps**: Once multi-frequency data are available, produce in-band spectral
  index images from WSClean multi-scale / MFS output.
