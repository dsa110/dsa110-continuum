
Last updated: 2026-03-09

---

# Roadmap: Multi-Day Light Curves

Goal: for each source in the reference catalog, a flux measurement (or upper limit) exists
for every epoch that passes QA, with variability metrics computed.

**Testing efficiency principle:** All regression tests should run against the *canary tile*
(`2026-01-25T22:26:05`, contains 3C454.3 at ~12.5 Jy) unless the change being tested is
specifically about multi-tile mosaicking. A single-tile run takes ~15–20 min vs. ~3 hours
for a full 11-tile epoch. Never run a full epoch to answer a question a single tile can answer.

---

## Phase 0 — Test infrastructure (prerequisite to all other phases)

*Without this phase, we cannot efficiently validate any pipeline change or epoch quality.*

- [ ] **Define canary tile**: Formally document `2026-01-25T22:26:05` as the standard
  regression tile. Add a `scripts/run_canary.sh` one-liner that runs `batch_pipeline.py`
  for exactly this tile, then `verify_sources.py` on the output, and prints the three QA
  metrics. Expected outputs: ratio ~0.93, n_detections ≥ 1 (3C454.3 alone is sufficient),
  RMS ≤ 15 mJy/beam. This becomes the fast sanity check after every pipeline code change.

- [ ] **Composite epoch QA metric (three-gate pass/fail)**: Implement and enforce three
  independent QA gates per epoch mosaic:
  1. **Flux scale**: median DSA/NVSS ratio in [0.8, 1.2] on sources ≥ 50 mJy
  2. **Detection completeness**: ≥ 60% of NVSS sources ≥ 50 mJy within footprint are
     recovered above 5σ local RMS
  3. **Noise floor**: median tile RMS ≤ 2× the expected thermal noise (~8 mJy/beam at
     current parameters)
  All three must pass for an epoch to be marked QA-PASS. Store verdict + all three metric
  values in `products/qa_summary.csv`. Generate a one-page diagnostic PNG per epoch
  (ratio histogram, completeness fraction bar, RMS map across tiles) saved alongside the
  mosaic. This replaces the current ad-hoc ratio-only check.

---

## Phase 1 — Reliable single-epoch pipeline (foundation)

*Use the canary tile for all regression checks in this phase.*

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

- [x] **--force-recal hardened**: Purges stale `*.ap.G` gain tables, `*_meridian.ms` intermediate
  MSes, epoch mosaic, photometry CSV, and `.tile_checkpoint.json` checkpoint so reruns are truly
  clean. (`batch_pipeline.py`, 2026-03-09)

- [x] **Epoch gaincal pre-conditioner phase solve**: Short-timescale (`solint='60s'`, `combine='spw'`)
  phase-only solve inserted before the main `p.G` solve to prevent vector decorrelation over the
  full `inf` interval. Pre-conditioner table (`precond.G`) is fed into all downstream gaintable
  chains. (`epoch_gaincal.py` lines 164–384, 2026-03-09)

- [x] **Epoch gaincal BP-only fallback**: Flag-fraction monitor (`GAINCAL_FLAG_FRACTION_LIMIT=0.30`)
  falls back to bandpass-only when gaincal SNR is too low. Validated on Jan 25 22h:
  median DSA/NVSS ratio **0.930**, 243 sources, RMS 8.54 mJy/beam. (`epoch_gaincal.py`, 2026-03-09)

- [ ] **Corrupt meridian MS guard** *(was incorrectly marked done)*: Add `_ms_is_valid(path)`
  helper to `batch_pipeline.py` that checks for `table.dat` and at least one `table.f*` file
  before trusting the skip-if-exists guard on `*_meridian.ms`. If the check fails, delete the
  partial directory with `shutil.rmtree` and regenerate. The `--force-recal` purge (already
  implemented) only fixes the problem for forced reruns; this guard fixes it for all runs.

- [ ] **Complete import rename** *(URGENT — silent failure risk)*: 929 references in
  `dsa110_continuum/` still import from `dsa110_contimg.core.*`. If the old package is
  installed, the pipeline silently runs stale code with no error. Do a mass rename and
  confirm 120/122 module imports pass under the casa6 env. This must happen before bulk
  data processing — a silent import error will corrupt all downstream products.

- [ ] **Resolve flux ratio regression**: Run the canary tile (single tile, ~15 min) after
  the import rename to confirm ratio is still ~0.930. If not, bisect: compare
  CORRECTED_DATA amplitudes before/after applycal, check `detect_datacolumn()` fallback,
  verify pbcor FITS is being read (not dirty image).

- [ ] **Fix mosaic reproject: RA wrap bug + 15-min/tile perf** (`scripts/mosaic_day.py`):
  Two bugs in `coadd_tiles` / `build_common_wcs` cause mosaicking to take ~15 min per tile
  (target: 3–5 min) and produce an oversized output grid (RA 34–341° instead of the true
  ~15° footprint):
  (1) **4D WCS mismatch**: WSClean outputs 4D FITS; `WCS(hdr).celestial` on a 4D header
      misaligns axis indices, producing bogus corner RA values that inflate the bounding box.
  (2) **Full-grid reproject**: `reproject_interp` allocates a ~500 M pixel array per tile;
      at 11 tiles this is the dominant cost.
  **Fix**: remove `build_common_wcs`; replace `coadd_tiles` with `reproject_and_coadd` +
  `find_optimal_celestial_wcs` from `reproject.mosaicking` (lazy footprint, circular RA
  mean); build a clean 2D header per tile before reprojecting.
  File to edit: `scripts/mosaic_day.py` — `build_common_wcs` (lines 219–264) and
  `coadd_tiles` (lines 267–335); update call site in `main()` (~line 486).

- [ ] **Fix `needs_calibration()` bare except** (`scripts/mosaic_day.py`, lines 92–107):
  The function catches `except Exception: return True`, silently swallowing disk read
  failures, table corruption, NaN division, and any other error — then claims calibration
  is needed with no diagnostic output. During batch runs this masks real problems (e.g. a
  corrupted CORRECTED_DATA column triggers unnecessary re-calibration instead of alerting).
  **Fix**: catch only `FileNotFoundError` / `RuntimeError` from casacore table access;
  let unexpected errors propagate. Log a warning on expected fallbacks.

- [ ] **Epoch gaincal remaining sub-items**:
  (1) Check whether Jan 25 02h gaincal had similarly high flagging rates — if the bandpass
      alone produced ratio 0.893, epoch gaincal may be actively harming faint fields and
      the fallback threshold should be lowered.
  (2) Build VLASS SQLite catalog (currently deferred) to improve sky model completeness
      for faint-field gaincal; move up in priority if sub-item (1) shows gaincal is needed.

- [ ] **Make data paths configurable**: Replace hardcoded `/stage/dsa110-contimg/ms/` and
  `/data/incoming/` across all scripts with a single config file or environment variable.
  Required before running on any machine other than H17.

- [ ] **Clarify and document noise threshold**: The 3.0 RMS ratio used in mosaic QA has an
  unclear origin. Anchor it to the expected thermal noise floor (baseline: ~8 mJy/beam at
  current parameters), document in `docs/reference/mosaicking.md`, and wire it into the
  composite QA metric from Phase 0.

- [ ] **Investigate leftmost mosaic tiles**: First 1–2 tiles are consistently noisier.
  Check observation log for low elevation or heavy RFI flagging at those timestamps.
  Low priority unless the noise drives those tiles to QA-FAIL systematically.

---

## Phase 2 — Data inventory + multi-epoch automation

- [ ] **Data inventory script**: Scan `/data/incoming/`, count HDF5 files by date and subband,
  flag incomplete observations (missing subbands for a timestamp), report total volume and
  per-date readiness. ~3 TB on disk currently, more to copy from other machines.

- [ ] **Per-date calibration**: Document and automate generating or symlinking cal tables for each
  new date. Standard procedure: symlink `_0~23.b` and `_0~23.g` from `2026-01-25`, then run
  `batch_pipeline.py --cal-date 2026-01-25`. Note where cross-date phase cal degrades dynamic
  range and per-date gain solutions become necessary.

- [ ] **Clear stale `qa_summary.csv` before batch run**: The current `products/qa_summary.csv`
  contains pre-fix epoch data (ratios of 0.212, NaN) and will silently mislead any tooling
  that reads it. Archive it to `products/qa_summary_stale_pre_fix.csv` before starting the
  multi-epoch batch run.

- [ ] **Automated multi-epoch batch run**: Run `batch_pipeline.py` on all available dates.
  Gate on the composite QA metric from Phase 0 — only epochs that pass all three gates
  contribute to Phase 4 forced photometry. Log to `products/qa_summary.csv`.
  *Implementation skeleton exists at `scripts/batch_run_all_dates.sh` — review and extend
  rather than starting from scratch.*

- [ ] **Per-strip calibration tables**: Derive B/G solutions for Dec=+33° and +54° strips
  (currently blocked — only +16° tables exist). Until then, `batch_pipeline.py` aborts at
  the dec-strip guard for those strips. Run calibration workflow for one night at each strip;
  store tables at `/stage/dsa110-contimg/ms/{date}T22:26:05_0~23.{b,g}`.

- [ ] **Handle DSA-110 elevation slews** *(detection done; routing pending)*: Implement automatic
  strip routing so the pipeline selects the correct cal-table set without manual intervention.
  Blocked by per-strip calibration tables above.

- [ ] **Bulk data processing**: Once inventory is complete and Phase 1 regressions are resolved,
  run the full pipeline on all available dates.

---

## Phase 3 — Reference source catalog

- [ ] **Source finding on best epoch**: Run BANE + Aegean on the highest-DR mosaic to produce a
  blind source catalog (`scripts/source_finding.py`). Record detection threshold and mosaic
  date/epoch used.

- [ ] **Cross-match with NVSS/RACS**: Match blind catalog against NVSS and RACS. Output a
  reference catalog with: source ID, RA/Dec, NVSS/RACS flux, spectral index (where available),
  angular separation from DSA detection. Use for forced photometry and flux-scale validation.

---

## Phase 4 — Forced photometry across all epochs

- [ ] **Per-epoch forced photometry**: Apply reference catalog positions to every QA-passing epoch
  mosaic. Output: one CSV per epoch at `products/mosaics/{date}/{date}T{HH}00_forced_phot.csv`
  with columns: source_id, ra, dec, flux_Jy, local_rms, SNR, epoch_utc, mosaic_path.

- [ ] **Per-source per-epoch QA flags**: Flag column for: local RMS > 2× median, source within
  1 beam of mosaic edge, source blended with bright neighbor. Flagged rows included but excluded
  from variability analysis.

---

## Phase 5 — Light curve extraction

- [ ] **Stack photometry CSVs**: Merge all per-epoch CSVs by source_id into a single light curve
  table (source × epoch). Store as `products/lightcurves/lightcurves_{date_range}.csv` and as
  wide-format Parquet for fast per-source slicing.

- [ ] **Variability metrics**: For each source, compute:
    - Mooley modulation index *m* = σ_S / ⟨S⟩
    - Variability significance *Vs* = (S_max − S_min) / sqrt(σ_max² + σ_min²)
    - Reduced chi-squared *η* against constant-flux null
  Use formulas from `docs/reference/vast-crossref.md`. Flag sources where *Vs* > 4.0 or *η* > 2.5.

- [ ] **Light curve visualization**: Per-source flux vs. time plots. Highlight variable candidates.
  Save to `products/lightcurves/plots/{source_id}.png`. Produce a summary page with the top-N
  most variable sources ranked by *η*.

---

## Cross-cutting — Documentation and operator docs

*These tasks do not block core science validation, but they reduce re-discovery cost and make
the pipeline easier to operate, audit, and hand off to collaborators.*

- [ ] **Render-validate the Quarto docs site and promote it as the operator-facing docs baseline**:
  A first-pass Quarto site now exists under `docs/quarto/` (`index.qmd`, `pipeline-overview.qmd`,
  `epoch-qa.qmd`, `outputs-and-artifacts.qmd`) and reflects the current repo state, but it has
  only been statically converted and editorially reviewed — it has **not** yet been rendered on a
  machine with Quarto installed. The next step is to validate the site end-to-end in a real render
  environment:
  1. Install Quarto on a machine where local HTML rendering is possible.
  2. Run `quarto render` (and ideally `quarto preview`) from `docs/quarto/`.
  3. Fix any real render-time issues: broken cross-links, bad heading anchors, Mermaid rendering
     problems, malformed callouts, sidebar/nav issues, code-fence formatting regressions, or pages
     that render but are hard to read.
  4. Verify the generated site matches the current implementation honestly: if a page says
     "Implemented" or "Verified", confirm the wording still matches actual code/test evidence and
     does not overclaim runtime validation.
  5. Keep the site audience-neutral but operator-friendly: the home page should clearly point a new
     collaborator to the pipeline overview, epoch QA interpretation, artifact/debugging map, and
     the correct runtime environment (`/opt/miniforge/envs/casa6/bin/python`).
  6. Treat Quarto as the **current internal docs baseline**, not as a final publishing decision.
     Do **not** migrate to Mintlify or another platform yet; first prove that the current docs are
     useful, accurate, and easy to maintain. If the site proves valuable after a few real pipeline
     runs, revisit whether to keep Quarto or port the content later.
  Deliverable: a render-validated `_site/` output plus a short follow-up pass on any wording or
  layout issues discovered only during real HTML rendering.

- [ ] **Decide Quarto vs Mintlify after 2–3 real pipeline runs**: Do not make a docs-platform
  decision in the abstract. After the Quarto site has been used during 2–3 actual pipeline runs
  (including at least one debugging session and one QA interpretation session), evaluate whether it
  is meeting the real needs of operators and developers:
  - Is it fast to update when code changes?
  - Are the diagrams and artifact maps actually being used?
  - Does Quarto provide enough structure/navigation, or would Mintlify materially improve
    discoverability and maintenance?
  - Is there any real friction around local rendering, hosting, or sharing?
  Only after that usage-based review should we decide whether to keep Quarto as the long-term docs
  baseline or migrate the content to Mintlify.

---

## Deferred / low priority

- [ ] **Build VLASS SQLite catalog database**: Gaincal tile-selection and sky model steps query
  VLASS via SQLite but the database has not been built on H17. RACS is also missing. Until
  then, gaincal falls back to NVSS+FIRST only (~8 sources), which is sufficient but degrades
  sky model completeness. Promote to Phase 1 if epoch gaincal sub-item (2) confirms it is needed.

- [ ] **GPU calibration**: `dsa110_continuum/calibration/gpu_calibration.py` exists but is
  untested end-to-end. Profile GPU bandpass solve speedup on the 110-antenna array.

- [ ] **Spectral index maps**: Once multi-frequency data are available, produce in-band spectral
  index images from WSClean MFS output.

- [ ] **Fix `pyproject.toml` package path**: `[tool.setuptools.packages.find]` says
  `where = ["src"]` but no `src/` directory exists — the package lives at `dsa110_continuum/`
  at the repo root. The package is not pip-installable in its current state. Fix: change to
  `where = ["."]` or move the package under `src/`. Not blocking H17 (pipeline runs via
  sys.path), but required before anyone else can `pip install -e .`.
