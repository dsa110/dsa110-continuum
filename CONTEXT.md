# CONTEXT — dsa110-continuum domain glossary

Vocabulary for the DSA-110 continuum imaging pipeline. Use these terms verbatim in issues, PRDs, refactor proposals, test names, and code comments. If a term you need isn't here, either reconsider the wording or add it via `/grill-with-docs`.

Each entry that names a code-level fact ends with a citation in `<path>::<Symbol>` form (e.g. `dsa110_continuum/imaging/params.py::ImagingParams.imsize`). Run `scripts/verify_glossary.py` to check that the cited symbols still resolve. Claims that are operational (not enforced in code) are tagged `(operational)` and cite a docs source instead.

## Instrument

- **DSA-110** — Deep Synoptic Array of 110 elements at OVRO. Meridian drift-scan transit array; the sky drifts through fixed beams, the array does not track.
- **Active antennas** — 96 online antennas of 117 array elements present in HDF5 metadata. Inactive elements correspond to non-existent antennas and are filtered before MS conversion. The HDF5 header carries the per-observation antenna metadata.
- **Subband** — one of 16 frequency chunks (48 channels each, 244 kHz/channel), totalling 187.5 MHz across 1.31–1.50 GHz (L-band). Files arrive as `{timestamp}_sb{00..15}.hdf5`.
- **Integration time** — 12.885 s per HDF5 sample; ~24 samples per ~5-minute tile.
- **Transit** — passage of a fixed-RA strip of sky through the beam; ~5 minutes at zenith.
- **Meridian** — instantaneous local meridian; calibration ideally uses a bright source at meridian transit.

## Pipeline stages and products

- **HDF5** — raw data unit: 16 subbands × N timestamps, 12.885 s integrations. Subband filename pattern `{YYYY-MM-DDTHH:MM:SS}_sb{NN}.hdf5`. All 16 subbands of one observation share a bit-identical `time_array[0]`.
- **MS / Measurement Set** — CASA-format calibrated visibility container produced from an HDF5 group. SPW-merged via `dsa110_continuum/conversion/merge_spws.py::merge_spws` before IDG imaging.
- **Tile** — single ~5-minute transit's image: a 4800 × 4800 px FITS at 3 arcsec/px (`dsa110_continuum/imaging/params.py::ImagingParams.imsize`, `dsa110_continuum/imaging/params.py::ImagingParams.cell_arcsec`). Produced sequentially from each group of 16 subbands as the sky drifts through the beam. Tiles flow continuously, not in batches. Not "snapshot" or "frame".
- **Hourly-epoch mosaic** — coadd of ~12 sequential tiles (~1 hour of transit) along the current Dec strip. Adjacent mosaics overlap so flux measurements stay continuous across boundaries. Two operational modes:
  - *Batch* (current production) — bin tiles by UTC hour; each epoch additionally includes the last 2 tiles of the previous hour and the first 2 of the next, giving ~4-tile / ~20-min overlap. First and last epoch of the day have one-sided overlap only (`scripts/batch_pipeline.py::bin_tiles_by_hour`, `scripts/batch_pipeline.py::build_epoch_tile_sets`).
  - *Sliding* (design target for streaming operation) — fixed 12-tile window, stride 6 → 50% overlap between adjacent mosaics (`dsa110_continuum/mosaic/trigger.py::SlidingWindowTrigger`; module-level `WINDOW`, `STRIDE`).

  Output filename: `{date}T{HH}00_mosaic.fits`.
- **Dec strip** — DSA-110 dwells at a fixed declination for long stretches; tiles within a strip share Dec (with small drift tracked in the HDF5 header) and step in RA via Earth rotation. The same sources are observed each sidereal day until a Dec change. NOT to be confused with a "daily mosaic" — mosaics are hourly, not 24-hour.
- **24-hour cycle** (operational) — the sidereal-day cadence has two consequences:
  1. *Source set is roughly stable* across a sidereal day — useful for variability tracking.
  2. *Bandpass cadence* — exactly one bandpass-calibrator transit per day in the configured preset.

  Cadence is operational, not code-enforced.
- **Run products** — per-date directory containing `run_<utc>.log`, `{date}_manifest.json`, `{date}_run_summary.json`, `run_report.md` (`dsa110_continuum/qa/run_report.py`).

## Calibration

- **Bandpass / Gain / Delay tables** — `.b` / `.g` / `.k` CASA tables produced by `dsa110_continuum/calibration/`.
- **Bandpass cycle** (24-hour, operational) — exactly one bandpass-calibrator transit per sidereal day in the configured Dec strip; that transit's MS is what the bandpass is solved against. Cadence is operational (a consequence of drift-scan + one preset bpcal per Dec strip), not enforced by code. The code *does* enforce that the calibrator be transiting within the MS being solved (`dsa110_continuum/calibration/runner.py::_validate_calibrator_transit`).
- **Calibration table borrowing** (bidirectional, day-level) — when no validated BP/G tables exist for the target date, the pipeline searches outward in both directions (±1, ±2, … days) for the nearest date with real tables and symlinks them onto the target date. Up to `max_borrow_days=30` by default. Borrows previously-validated tables; does NOT re-solve the bandpass from a different day's calibrator MS (`dsa110_continuum/calibration/ensure.py::_find_nearest_real_tables`).
- **Applycal** — application of bandpass + gain solutions to the MS, populating `CORRECTED_DATA`.
- **Phaseshift** — re-pointing the visibility phase centre to a target sky position before imaging. Implemented via `chgcentre`. After phaseshift, `FIELD::PHASE_DIR` and `FIELD::REFERENCE_DIR` are synced explicitly (`dsa110_continuum/calibration/runner.py::update_phase_dir_to_target`, `dsa110_continuum/calibration/runner.py::sync_reference_dir_with_phase_dir`).
- **Self-calibration** — iterative gain refinement against the imaged sky model (`dsa110_continuum/calibration/selfcal.py::SelfCalConfig`).
- **Default calibration preset** — `field="0~23"`, `refant="103"`, `prebp_phase=True`, `bp_minsnr=5.0`, `gain_calmode="ap"`, `gain_solint="inf"` (`dsa110_continuum/calibration/presets.py::DEFAULT_PRESET`).
- **Flux anchor** — provenance of the calibration's absolute flux scale: `"perley_butler_primary"` (model-anchored) or `"vla_catalog"` (catalog-anchored, fallback) (`dsa110_continuum/calibration/ensure.py::_lookup_calibrator_coords`).
- **Selection pool** — `"primary"` or `"bright_fallback"`; records which calibrator family produced the BP/G tables (`dsa110_continuum/calibration/ensure.py::_lookup_calibrator_coords`).
- **Refant** — reference antenna for phase solutions. Default `103` (`dsa110_continuum/calibration/presets.py::DEFAULT_PRESET`).
- **Provenance sidecar** — JSON file adjacent to a BP table recording selection metadata (calibrator name, flux anchor, selection pool, source date, etc.). Used to validate borrowed-table strip-compatibility (`dsa110_continuum/calibration/ensure.py::provenance_sidecar_path`, `dsa110_continuum/calibration/ensure.py::write_provenance_sidecar`, `dsa110_continuum/calibration/ensure.py::load_provenance_sidecar`).

## Imaging

- **WSClean** — primary imager. Two backends: `wgridder` (default) and `idg` (image-domain gridder, requires SPW merge). Selectable via `--gridder` (`dsa110_continuum/imaging/cli.py`).
- **EveryBeam** — primary-beam model library used by IDG. Requires `TELESCOPE_NAME = DSA_110` on the MS. The MS-level patch is `dsa110_continuum/conversion/helpers_telescope.py::set_ms_telescope_name`.
- **Sky model seeding** — predict-then-image two-step workflow: predict model visibilities from a catalog into `MODEL_DATA` via `wsclean -predict`, then deconvolve. Reduces major cycles by 2–4× (see `docs/reference/imaging.md`).
- **Phase centre invariants** — `FIELD::PHASE_DIR` and `FIELD::REFERENCE_DIR` must be synced after `chgcentre`; `TELESCOPE_NAME` must be reset to `"DSA_110"` after `merge_spws` for EveryBeam compatibility. Silent-failure mode if any are skipped (`dsa110_continuum/calibration/runner.py::update_phase_dir_to_target`, `dsa110_continuum/calibration/runner.py::sync_reference_dir_with_phase_dir`, `dsa110_continuum/conversion/helpers_telescope.py::set_ms_telescope_name`).

## Mosaicking

- **Quicklook** — image-domain mosaic on pre-deconvolved tile FITS; nearest-neighbour regridding, weighted coadd with `weight = w_rms × w_PB²` (Airy-disk primary beam, dish diameter 4.65 m). Pixels below 10% peak beam response blanked. Fast (`dsa110_continuum/mosaic/builder.py::build_mosaic`; see `docs/reference/mosaicking.md` for parameters).
- **Science / Deep** — visibility-domain joint deconvolution: all tile MS files phaseshifted to a common centre and jointly imaged with WSClean+IDG using `-grid-with-beam` for direction-dependent beam correction. Slower; scientifically correct for wide fields (see `docs/reference/mosaicking.md`).
- **RA-wrap (circular mean)** — when tiles span 0° RA, the mosaic centre is computed as `arctan2(mean(sin(RA)), mean(cos(RA)))` rather than arithmetic mean. Fixes the bug where a [350°, 5°, 10°] tile set centred at 121.7° instead of 1.7° (`dsa110_continuum/mosaic/builder.py::build_mosaic`).
- **Coadd** — combine reprojected tiles into a single mosaic FITS. Uses `reproject.reproject_interp` per tile (full-grid; optimisation deferred) (`dsa110_continuum/mosaic/builder.py::build_mosaic`).
- **Strip grouping** (legacy day-batch) — `scripts/mosaic_day.py::group_tiles_by_ra` partitions a day's tiles into contiguous RA strips with a 10° gap threshold. Used by the per-day batch path; the streaming path does not partition (it produces sliding-window mosaics directly).

## Photometry and variability

- **Forced photometry** — flux extraction at a fixed sky position using a Condon matched filter, regardless of detection. Error propagation: `dsa110_continuum/photometry/condon_errors.py::CondonErrors`, `dsa110_continuum/photometry/condon_errors.py::calc_condon_errors`.
- **Light curve** — per-source flux vs epoch CSV, optionally stacked across dates (`scripts/stack_lightcurves.py`, `scripts/plot_lightcurves.py`).
- **ESE** — extreme scattering event; refractive lensing of a compact extragalactic source by an interstellar plasma lens. Days-to-weeks timescale. Multiple detection pipelines exist:
  - `dsa110_continuum/photometry/ese_detection_enhanced.py::detect_ese_candidates_enhanced` (single-frequency, enhanced thresholds)
  - `dsa110_continuum/photometry/multi_frequency.py::detect_ese_multi_frequency`
  - `dsa110_continuum/photometry/multi_observable.py::detect_ese_multi_observable`
  - `dsa110_continuum/photometry/parallel.py::detect_ese_parallel` (parallel orchestrator)
- **Variability metrics** — Mooley *η* (reduced χ² of light curve), V-statistic, and modulation index *m* (`dsa110_continuum/photometry/variability.py`, `dsa110_continuum/photometry/source_monitoring_report.py`). Definitions tracked in `docs/reference/vast-crossref.md`.

## QA and operations

- **Three-gate epoch QA** — composite metric over each hourly-epoch mosaic; passes only if all three gates pass: (1) flux scale, (2) catalog completeness, (3) noise floor (`dsa110_continuum/qa/composite.py::CompositeQA`; `dsa110_continuum/photometry/epoch_qa.py::measure_epoch_qa`).
- **Calibration gate** — separate check on the BP/G tables before applycal; not part of the three-gate composite (`scripts/batch_pipeline.py::check_cal_gate`).
- **`pipeline_verdict`** — manifest field: `"CLEAN"` (no gates triggered) or `"DEGRADED"` (at least one gate triggered, but pipeline continued). Surfaced in the run report (`dsa110_continuum/qa/run_report.py`).
- **Strict QA** — `--strict-qa` flag aborts on cal gate failure and skips photometry for QA-FAIL epochs. Default in production (`scripts/batch_pipeline.py::check_cal_gate`).
- **Lenient QA** — `--lenient-qa` is a diagnostic override; it lets photometry/archive proceed despite QA-FAIL and emits a `lenient_qa` gate marker in the manifest (`scripts/batch_pipeline.py`).
- **Quarantine** — MS entries whose checkpoint failure count reaches `--quarantine-after-failures` are skipped on subsequent runs until `--clear-quarantine` (`scripts/batch_pipeline.py`).
- **Canary** — fast regression smoke test that runs the QA-measurement path on a known-good pre-existing FITS tile (2026-01-25T22:26:05, 3C454.3 at ~12.5 Jy/beam). Does NOT re-run calibration or imaging (`scripts/run_canary.sh`).
- **Calibrator-tile smoke** — single-tile reproducibility test using 3C48 as bandpass calibrator, pinned to one calibrator with no fallback ladder (`scripts/hdf5_calibrator_tile_smoke.py`).
- **Pipeline DB** — SQLite database `pipeline.sqlite3` indexing HDF5 inventory and conversion status. Tables include `hdf5_files` and `group_time_ranges`. Read by `scripts/inventory.py` and several photometry/visualization helpers. New dates must be indexed before they are visible to the pipeline.
