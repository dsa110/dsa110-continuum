# Dec-Strip Awareness Design

**Approved:** 2026-03-04

## Root cause

DSA-110 observes different declination strips on different nights (+16°, +33°, +54°, …).
The pipeline was built and validated on Jan 25 at Dec≈+16° and carries that assumption in
three places:

1. **Calibration tables** — `batch_pipeline.py` hardcodes the Jan-25 `.b`/`.g` tables,
   which were solved at Dec≈+16°. Applying them to +33° or +54° data degrades the
   absolute flux scale by up to ~20× (confirmed: Feb 23 and Feb 26 median DSA/NVSS ≈ 0.06).

2. **Dec-strip guard absent** — the pipeline processes any MS regardless of its Dec,
   silently producing calibrated but scientifically useless images when Dec mismatches.

3. **Inventory has no Dec column** — operators cannot tell at a glance which Dec strip
   each epoch belongs to.

## What this design does NOT fix

- Deriving cal tables for +33° and +54° strips (deferred; requires a calibrator transit
  on each strip).
- The separate Feb 15 (Dec=+16°) low-flux issue, which requires a diagnostic re-run
  under the current code.

## Approach

**Detect & reject**, not silently process.

1. Expose `read_ms_dec(ms_path) -> float` in a shared utility so every component can
   read the observed Dec without importing a heavy calibration module.
2. In `batch_pipeline.py`, detect the epoch Dec from the first MS, compare to a
   `--expected-dec` (default +16.1°), and abort with a clear message if the mismatch
   exceeds `DEC_CHANGE_THRESHOLD_DEG` (5°, already in `mosaic_constants.py`).
3. Add a `--expected-dec` CLI flag so future strips can be processed explicitly once
   cal tables exist for them.
4. Update `inventory.py` to read and report the Dec strip per MS so operators can audit
   the dataset before launching batch runs.

## Out of scope

- Deriving new cal tables for non-+16° strips.
- Automated per-strip cal table selection.
- Multi-strip light curve merging.
