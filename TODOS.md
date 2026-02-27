
Last updated: 2026-02-27 (02h test run)

---

## Blocking (must fix before science results are trustworthy)

- [x] **Primary beam correction**: `pbcor=True` confirmed working. 02h epoch forced photometry (2026-02-27): median DSA/NVSS ratio = **0.893** on 1047 NVSS sources — within the 0.8–1.2 target. Previous ratio of 0.39 was from before pbcor was enabled.

---

## High priority

- [x] **Re-run conversion for 3 corrupt MS files**: Timestamps `2026-01-25T22:20:55`, `22:48:11`, `22:53:20` — re-converted 2026-02-26 using `--no-skip-existing` on window `22:20–22:55`. FIELD tables now present.

- [ ] **Data inventory script**: Write a script that scans `/data/incoming/`, counts HDF5 files by date and subband, identifies incomplete observations (missing subbands for a given timestamp), and reports total data volume. ~3 TB currently on disk, more to copy from other machines.

- [ ] **Make data paths configurable**: Replace hardcoded `/stage/dsa110-contimg/ms/` and `/data/incoming/` paths across all scripts with a single config file or environment variable.

- [x] **mosaic_day.py cleanup**: Added `--keep-intermediates` flag (default False). Each `*_meridian.ms` is deleted immediately after its tile images successfully. After QA the full mosaic directory is moved from `/stage/dsa110-contimg/images/mosaic_{DATE}/` to `/data/dsa110-continuum/products/mosaics/{DATE}/`. Also consolidated the repeated `2026-01-25` date string into a single `DATE` constant.

---

## Medium priority

- [ ] **Complete import rename**: 828 references in `dsa110_continuum/` still import from `dsa110_contimg.core.*`. Do a mass rename to make the package self-contained and independent of the old repo being installed.

- [ ] **Investigate leftmost mosaic tiles**: First 1–2 tiles appear significantly noisier than the rest. Check observation log for elevation, RFI flags, or data quality issues at those timestamps.

- [ ] **Clarify noise consistency threshold**: The 3.0 ratio threshold used in mosaic QA has unclear origin. Verify against VAST pipeline or radio survey literature.

---

## Low priority / future

- [x] **02h test epoch run (2026-01-25)**: `batch_pipeline.py --start-hour 2 --end-hour 3` completed 2026-02-27. 11 tiles, mosaic peak 7.79 Jy/beam, RMS 11.9 mJy/beam, DR 656, QA PASSED. ATNF cone search (8 deg radius): 8 pulsars found; none detected above 5×RMS (only one with known 1.4 GHz flux, J0304+1932 at 15 mJy, fell off the mosaic footprint). Output: `/stage/dsa110-contimg/images/mosaic_2026-01-25/2026-01-25T0200_mosaic.fits`.

- [ ] **Bulk data processing**: Once inventory is complete and primary beam correction is verified, run full pipeline on all available dates.

- [ ] **Light curve extraction**: Once daily mosaics are produced reliably, build a script to extract flux vs. time for each cataloged source.

- [ ] **Handle DSA-110 elevation slews**: Pipeline currently assumes fixed declination. Add logic to switch calibrator source when the array slews to a new elevation.
