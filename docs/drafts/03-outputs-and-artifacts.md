# Outputs and Artifacts

> **Status**: Draft v2 — 2026-03-09
> **Audience**: Operators locating and interpreting pipeline outputs
> **Verified against**: `batch_pipeline.py`, `mosaic_day.py`, `source_finding.py`, `forced_photometry.py` as of 2026-03-09
> **See also**: [Pipeline Overview](01-pipeline-overview.md) | [Epoch QA](02-epoch-qa.md)

## Artifact map

```mermaid
flowchart LR
    subgraph raw["/data/incoming/"]
        HDF5[/"HDF5 files"/]
    end

    subgraph stage["/stage/dsa110-contimg/"]
        MS[/"Measurement Sets\nms/*.ms"/]
        CAL[/"Cal tables\nms/*.b, *.g"/]
        TILE[/"Tile FITS\nimages/mosaic_{date}/*-image-pb.fits"/]
        EMOSAIC[/"Epoch mosaics\nimages/mosaic_{date}/*_mosaic.fits"/]
        DIAG[/"QA PNGs\nimages/mosaic_{date}/*_qa_diag.png"/]
    end

    subgraph products["/data/dsa110-continuum/products/"]
        PMOSAIC[/"Archived mosaics\nmosaics/{date}/"/]
        PCSV[/"Photometry CSVs\nmosaics/{date}/"/]
        QA_CSV[/"qa_summary.csv"/]
    end

    subgraph catalogs["/data/dsa110-contimg/state/catalogs/"]
        NVSS[("nvss_full.sqlite3")]
    end

    HDF5 -->|convert| MS
    CAL -->|applycal| MS
    MS -->|WSClean| TILE
    TILE -->|mosaic| EMOSAIC
    EMOSAIC -->|epoch QA| DIAG & QA_CSV
    EMOSAIC -->|forced phot| PCSV
    EMOSAIC -->|archive| PMOSAIC
    NVSS -.->|reference| PCSV & QA_CSV
```

---

## Directory layout

### `/data/incoming/` — Raw HDF5

| Pattern | Description |
|---------|-------------|
| `{YYYY-MM-DD}T{HH:MM:SS}_sb{NN}.hdf5` | Raw UVH5 subband file. 16 subbands per complete observation |

- ~3 TB on disk (H17), growing
- **Inspect when**: Missing subbands, corrupt files, incomplete observations

### `/stage/dsa110-contimg/ms/` — Measurement Sets and calibration tables

| Pattern | Description |
|---------|-------------|
| `{date}T{HH:MM:SS}.ms` | Measurement Set (one per observation timestamp) |
| `{date}T{HH:MM:SS}_meridian.ms` | Phase-shifted intermediate MS. **Temporary** — deleted after successful imaging unless `--keep-intermediates` is set |
| `{date}T22:26:05_0~23.b` | Bandpass calibration table (CASA format) |
| `{date}T22:26:05_0~23.g` | Gain calibration table (CASA format) |

- **Current limitation**: Only 2026-01-25 cal tables exist for Dec ≈ +16° strip. Other dates must symlink to these (see `CLAUDE.md` for symlink commands)
- **Inspect when**: Calibration failures, flux scale issues, missing `CORRECTED_DATA` column

### `/stage/dsa110-contimg/images/mosaic_{date}/` — Tile images and mosaics

**Per-tile outputs** (one set per observation timestamp):

| Pattern | Description |
|---------|-------------|
| `{date}T{HH:MM:SS}-image-pb.fits` | PB-corrected tile image — **canonical input to mosaicking** |
| `{date}T{HH:MM:SS}-image.fits` | Uncorrected tile image (fallback if `-pb` not produced) |
| `{date}T{HH:MM:SS}-beam-0.fits` | WSClean beam model (used for PB cutoff during mosaicking) |
| `{date}T{HH:MM:SS}-dirty.fits` | Dirty image (WSClean intermediate) |
| `{date}T{HH:MM:SS}-residual.fits` | Residual image (WSClean intermediate) |

**Epoch-level outputs** (from `batch_pipeline.py`):

| Pattern | Description |
|---------|-------------|
| `{date}T{HH}00_mosaic.fits` | Hourly epoch mosaic |
| `{date}T{HH}00_mosaic_qa_diag.png` | Three-panel QA diagnostic — see [Epoch QA → Diagnostic PNG](02-epoch-qa.md#diagnostic-png-layout) |
| `.tile_checkpoint.json` | Crash recovery: list of completed tile FITS paths |

**Full-day outputs** (from `mosaic_day.py` standalone):

| Pattern | Description |
|---------|-------------|
| `full_mosaic.fits` | Full-day mosaic when all tiles belong to one RA strip |
| `full_mosaic_ra{NNN}.fits` | Per-strip mosaic when tiles span disjoint RA ranges (e.g., 02h and 22h) |

**Epoch gaincal subdirectory** (`epoch_gaincal/`):

| Pattern | Description |
|---------|-------------|
| `*.ap.G` | Per-epoch amplitude+phase gain table |
| `precond.G` | Short-timescale phase pre-conditioner table |

### `/data/dsa110-continuum/products/` — Archived science products

| Path | Description |
|------|-------------|
| `mosaics/{date}/{date}T{HH}00_mosaic.fits` | Archived epoch mosaic (copied from staging). FITS headers include provenance cards (`PIPEVER`, `CALDATE`, `NTILES`, `BPFLAG`, `GPHSCTR`) and QA cards (`QARESULT`, `QARMS`, `QARAT`) |
| `mosaics/{date}/{date}T{HH}00_forced_phot.csv` | Per-epoch forced photometry CSV |
| `mosaics/{date}/{date}_manifest.json` | **Pipeline provenance manifest** — git SHA, cal table quality, per-tile status, per-epoch QA. See [Calibration QA](../skills/calibration-qa.md) |
| `mosaics/{date}/{date}_run_summary.json` | Execution summary (epoch pass/fail counts, wall time). Symlinked at `/tmp/pipeline_last_run.json` for backward compat |
| `qa_summary.csv` | Master QA log — all dates, all epochs |
| `lightcurves/` | Cross-epoch light curve products *(Planned — directory exists, not yet populated)* |

- **Inspect when**: Cross-checking final results, auditing QA verdicts, debugging flux ratios

### `/data/dsa110-contimg/state/catalogs/` — Reference catalogs

| File | Table | Key columns | Used by |
|------|-------|-------------|---------|
| `nvss_full.sqlite3` | `sources` | `ra_deg, dec_deg, flux_mjy` | epoch QA, forced photometry, source verification |
| `master_sources.sqlite3` | `sources` | `source_id, ra_deg, dec_deg, flux_jy` | source verification |
| `atnf_full.sqlite3` | `sources` | `name, ra_deg, dec_deg, flux_mjy` | source verification (pulsars) |

- **Current limitation**: VLASS and RACS catalogs not yet built on H17

---

## Key output file schemas

### Forced photometry CSV

**Path**: `products/mosaics/{date}/{date}T{HH}00_forced_phot.csv`
**Produced by**: `batch_pipeline.py` → `run_photometry_phase()` → `measure_forced_peak()`

| Column | Type | Description |
|--------|------|-------------|
| `ra_deg` | float | Source RA (decimal degrees) |
| `dec_deg` | float | Source Dec (decimal degrees) |
| `nvss_flux_jy` | float | NVSS catalog flux (Jy) |
| `dsa_peak_jyb` | float | Measured DSA peak flux (Jy/beam) |
| `dsa_peak_err_jyb` | float | Peak flux uncertainty (Jy/beam) |
| `dsa_nvss_ratio` | float | DSA / NVSS flux ratio |
| `source_id` | int | Sequential integer within epoch |

Note: The `source_id` is sequential per epoch, not globally stable. Cross-epoch source matching is done by `stack_lightcurves.py` using positional clustering (5″ match radius).

### QA summary CSV

**Path**: `products/qa_summary.csv`
**Produced by**: `batch_pipeline.py` → `write_qa_summary_row()`
**Schema**: Defined in `batch_pipeline.py:QA_CSV_FIELDS`

| Column | Type | Description |
|--------|------|-------------|
| `date` | str | Observation date (`YYYY-MM-DD`) |
| `epoch_utc` | str | Epoch label (e.g., `2026-01-25T2200`) |
| `mosaic_path` | str | Full path to epoch mosaic FITS in staging |
| `n_catalog` | int | NVSS sources ≥ 50 mJy in footprint |
| `n_recovered` | int | Sources recovered above 5σ local RMS |
| `completeness_frac` | float | Detection completeness (0.0–1.0) |
| `median_ratio` | float | Median DSA/NVSS flux ratio |
| `ratio_gate` | str | `PASS` or `FAIL` |
| `completeness_gate` | str | `PASS`, `FAIL`, or `SKIP` |
| `rms_gate` | str | `PASS` or `FAIL` |
| `mosaic_rms_mjy` | float | Global MAD-RMS (mJy/beam) |
| `qa_result` | str | Overall `PASS` or `FAIL` |
| `gaincal_used` | str | `ok`, `fallback`, `skipped`, or `error` |

For gate threshold definitions, see [Epoch QA → Constants reference](02-epoch-qa.md#constants-reference).

### Source verification CSV

**Produced by**: `scripts/verify_sources.py --out PATH`

| Column | Type | Description |
|--------|------|-------------|
| `source_name` | str | Catalog source name |
| `ra_deg` | float | Source RA |
| `dec_deg` | float | Source Dec |
| `catalog_flux_jy` | float | Reference catalog flux |
| `dsa_peak_jyb` | float | Measured DSA peak flux |
| `snr` | float | Signal-to-noise ratio |
| `ratio` | float | DSA / catalog flux ratio |
| `source_type` | str | `continuum` or `pulsar` |
| `is_upper_limit` | str | `True` if SNR < 3 |
| `catalog` | str | `master`, `nvss`, or `atnf` |

### Source catalog FITS (Aegean)

**Produced by**: `scripts/source_finding.py`
**Path**: `{mosaic_stem}_sources.fits`

| Column | Type | Description |
|--------|------|-------------|
| `source_name` | str | Aegean-assigned name |
| `ra_deg`, `dec_deg` | float | Fitted position |
| `peak_flux_jy` | float | Peak flux density |
| `int_flux_jy` | float | Integrated flux density |
| `a_arcsec`, `b_arcsec`, `pa_deg` | float | Fitted Gaussian shape |
| `local_rms_jy` | float | Local RMS at source position |

---

## Other diagnostic outputs

| File | Location | Purpose |
|------|----------|---------|
| `/tmp/pipeline_last_run.json` | Symlink to `products/mosaics/{date}/{date}_run_summary.json` | Machine-readable run summary: date, wall time, per-epoch pass/fail. The canonical copy now lives in products; the `/tmp` path is a backward-compat symlink |
| `inventory.csv` | Repo root | Data census: HDF5 file counts by date/subband, MS availability, Dec strip |
| `.tile_checkpoint.json` | Staging dir | Crash recovery: list of completed tile FITS paths and cal-date used |

---

## What to inspect when debugging

### Is the run complete?

| Check | Where to look |
|-------|---------------|
| How many tiles processed? | `.tile_checkpoint.json` in staging dir → `completed` array length |
| Did all epochs get built? | `products/qa_summary.csv` → count rows for the date |
| Any tiles time out or crash? | `batch_pipeline.py` log output → search for `TIMEOUT` or `FAILED` |
| Is there a stale partial run? | `*_meridian.ms` directories still present in `/stage/dsa110-contimg/ms/` → indicates incomplete tile that wasn't cleaned up |

### What went wrong with the calibration?

| Check | Where to look |
|-------|---------------|
| Cal table quality at a glance | `{date}_manifest.json` → `cal_quality.bp` and `cal_quality.g` |
| Bandpass flag fraction | `manifest.json` → `cal_quality.bp.flag_fraction` (target: < 0.3) |
| Gain phase scatter | `manifest.json` → `cal_quality.g.phase_scatter_deg` (target: < 30°) |
| Cross-date cal applied? | `manifest.json` → compare `date` vs `cal_date`. If different, check `gaincal_status` |
| Epoch gaincal success? | `manifest.json` → `gaincal_status`: `"ok"` = good, `"fallback"` = failed (used static table), `"error"` = crashed |
| FITS-level provenance | `fitsheader mosaic.fits` → `CALDATE`, `BPFLAG`, `GPHSCTR`, `QARESULT` |
| Detailed per-tile outcomes | `manifest.json` → `tiles[]` array (status, elapsed, error) |

See [Calibration QA](../skills/calibration-qa.md) for threshold interpretation tables and full manifest schema.

### Is the flux scale correct?

| Check | Where to look |
|-------|---------------|
| Overall flux scale | `qa_summary.csv` → `median_ratio` column (target: 0.8–1.2) |
| Per-source ratios | `*_forced_phot.csv` → `dsa_nvss_ratio` column |
| Visual histogram | `*_qa_diag.png` → left panel |
| Cross-strip misapplication? | `median_ratio` ≈ 0.06 is the signature of wrong Dec-strip cal tables |
| Wrong gaincal? | `qa_summary.csv` → `gaincal_used = "fallback"` means epoch gaincal failed and BP-only was used |

### Is source recovery working?

| Check | Where to look |
|-------|---------------|
| Completeness fraction | `qa_summary.csv` → `completeness_frac` (target: ≥ 0.60) |
| How many recovered? | `qa_summary.csv` → `n_recovered` / `n_catalog` |
| Visual bar | `*_qa_diag.png` → center panel |
| Source-level detail | Run `verify_sources.py --fits MOSAIC --out check.csv` and inspect `is_upper_limit` column |

### Is the noise level acceptable?

| Check | Where to look |
|-------|---------------|
| Global mosaic RMS | `qa_summary.csv` → `mosaic_rms_mjy` (target: ≤ 17.1) |
| Per-tile noise variation | `*_qa_diag.png` → right panel (per-tile RMS bars) |
| Tile-level noise | Open individual `*-image-pb.fits` and compute MAD-RMS of central 400×400 px |
| Edge tiles noisier? | First/last tiles in an epoch are consistently noisier (known, low priority) |

### Is the beam model correct?

| Check | Where to look |
|-------|---------------|
| Telescope name | `OBSERVATION::TELESCOPE_NAME` in MS should be `DSA_110`, not `OVRO_MMA` |
| Beam FITS present? | `*-beam-0.fits` must exist alongside each `*-image-pb.fits` |
| PB cutoff applied? | Look for `PB cutoff` log lines during mosaicking; missing beam file → no cutoff → noisy edges |

### Which files are canonical at each stage?

| Stage | Canonical file | Fallback |
|-------|---------------|----------|
| Tile image for mosaicking | `*-image-pb.fits` | `*-image.fits` (no PB correction) |
| Epoch mosaic | `{stage_dir}/{date}T{HH}00_mosaic.fits` | — |
| Archived mosaic | `products/mosaics/{date}/{date}T{HH}00_mosaic.fits` | Staging copy |
| QA verdict | `products/qa_summary.csv` | — |
| Forced photometry | `products/mosaics/{date}/{date}T{HH}00_forced_phot.csv` | — |
| Source catalog | `{mosaic_stem}_sources.fits` (from Aegean) | — |
