# Stage C: Post-Discovery Cross-Match Design Spec

## Purpose

Stage C annotates the blind-detection catalog produced by Stage B (Aegean FITS table)
by matching each detected source against the master radio catalog (NVSS+VLASS+FIRST+RACS)
and a small set of auxiliary catalogs. The output is an annotated FITS table that records,
for each Aegean detection:

- Whether it has a counterpart in the master catalog (and separation/flux ratio)
- Whether it has a counterpart in NVSS, FIRST, or RACS individually
- Whether it is unmatched (a "new" source candidate)
- Astrometric offset metrics (median ΔRA, ΔDec, RMS) across all matched pairs

Stage C does **not** make variability decisions, trigger alerts, or modify Stage A photometry.
It is purely an annotation and QA pass.

---

## Scope

**In scope:**
- Read Stage B catalog FITS (list of `SourceCatalogEntry` fields)
- Cross-match against master catalog via `cone_search("master", ...)` → `cross_match_sources()`
- For unmatched sources, attempt fallback match against NVSS, FIRST, RACS individually
- Write annotated FITS table: one row per Aegean detection, columns for match status and metadata
- Compute and log astrometry QA metrics
- CLI script: `scripts/stage_c_crossmatch.py`

**Out of scope:**
- Variability classification
- Light-curve generation
- Visual inspection or CARTA integration
- Multi-epoch stack analysis

---

## Inputs / Outputs

### Input
- `catalog_path` (FITS) — Stage B Aegean catalog; columns per `SourceCatalogEntry`:
  `source_name, ra_deg, dec_deg, peak_flux_jy, peak_flux_err_jy, int_flux_jy,
  a_arcsec, b_arcsec, pa_deg, local_rms_jy`
- `ra_center`, `dec_center`, `radius_deg` — field center and search radius for catalog cone queries
  (derived from mosaic WCS if not provided explicitly)
- `match_radius_arcsec` — default 10.0 arcsec

### Output
- `{catalog_stem}_crossmatched.fits` — FITS binary table, one row per Aegean detection.

**Output columns:**
| Column | Type | Description |
|--------|------|-------------|
| `source_name` | str | Aegean source name |
| `ra_deg` | float | Aegean RA |
| `dec_deg` | float | Aegean Dec |
| `peak_flux_jy` | float | Aegean peak flux |
| `snr` | float | peak_flux_jy / local_rms_jy |
| `master_matched` | bool | Has master catalog counterpart |
| `master_sep_arcsec` | float | Separation to nearest master match (NaN if unmatched) |
| `master_flux_mjy` | float | Master catalog flux (NaN if unmatched) |
| `master_flux_ratio` | float | peak_flux_jy / (master_flux_mjy/1000) (NaN if unmatched) |
| `master_source_id` | str | Master catalog ID (empty if unmatched) |
| `nvss_matched` | bool | Fallback NVSS match |
| `nvss_sep_arcsec` | float | |
| `nvss_flux_mjy` | float | |
| `first_matched` | bool | Fallback FIRST match |
| `first_sep_arcsec` | float | |
| `racs_matched` | bool | Fallback RACS match |
| `racs_sep_arcsec` | float | |
| `any_matched` | bool | master OR nvss OR first OR racs |
| `new_source_candidate` | bool | not any_matched AND snr >= 5 |

---

## Architecture

### New module: `dsa110_continuum/catalog/stage_c.py`

One public function:

```python
def run_stage_c(
    catalog_path: str | Path,
    out_path: str | Path | None = None,
    *,
    ra_center: float | None = None,
    dec_center: float | None = None,
    radius_deg: float = 2.0,
    match_radius_arcsec: float = 10.0,
    new_source_snr_threshold: float = 5.0,
) -> Path:
```

It:
1. Reads the Aegean FITS catalog into a DataFrame
2. Derives field center from the catalog centroid if `ra_center`/`dec_center` not provided
3. Runs `cone_search("master", ...)` → `cross_match_sources()` for primary match
4. For unmatched detections, queries NVSS, FIRST, RACS individually and matches
5. Assembles the annotated table
6. Computes astrometry QA (median ΔRA, ΔDec, RMS) via `calculate_positional_offsets()`
7. Logs QA summary
8. Writes `*_crossmatched.fits`
9. Returns output path

### Script: `scripts/stage_c_crossmatch.py`

Thin CLI wrapper:
```
usage: stage_c_crossmatch.py [--catalog PATH] [--out PATH]
                              [--ra RA_DEG] [--dec DEC_DEG]
                              [--radius RADIUS_DEG]
                              [--match-radius ARCSEC]
                              [--sim]
```

`--sim` uses `pipeline_outputs/step6/step6_mosaic_sources.fits` as input (Stage B sim output).

### Tests: `tests/test_stage_c_crossmatch.py`

10 CI-runnable tests (no network, no real catalogs):

1. `test_read_aegean_catalog_empty` — empty FITS → `ValueError`
2. `test_run_stage_c_all_matched` — 5 sources, mock `cone_search` returns all 5 → `master_matched` all True
3. `test_run_stage_c_no_master_fallback` — 3 sources, master empty, NVSS returns 2 → `nvss_matched` for 2
4. `test_run_stage_c_new_source_candidate` — unmatched source with SNR ≥ 5 → `new_source_candidate=True`
5. `test_run_stage_c_low_snr_not_candidate` — unmatched with SNR < 5 → `new_source_candidate=False`
6. `test_output_fits_columns` — confirms all 19 output columns present in written FITS
7. `test_output_path_default_stem` — default out_path is `{stem}_crossmatched.fits`
8. `test_astrometry_qa_logged` — checks `log.info` called with median ΔRA/ΔDec
9. `test_flux_ratio_computed` — flux_ratio = peak_flux / (master_flux_mjy/1000)
10. `test_cli_sim_exits_zero` — `--sim` with missing catalog exits 0 with a warning

---

## Key Design Decisions

### Catalog query strategy
- Primary: `cone_search("master", ra, dec, radius_deg)` → covers NVSS+VLASS+FIRST+RACS unified
- Fallback (only for sources NOT matched by master): individual `cone_search("nvss")`, `cone_search("first")`, `cone_search("rax")` within the same cone
- If catalog query raises (DB not found), catch and treat as zero results — Stage C is non-fatal

### Field center derivation
- If `ra_center` / `dec_center` not given: use median RA/Dec of detected sources + `radius_deg` = half the field diagonal
- This avoids requiring the mosaic FITS to be present at crossmatch time

### New source candidates
- `new_source_candidate = not any_matched AND snr >= new_source_snr_threshold` (default 5.0)
- SNR = `peak_flux_jy / local_rms_jy`
- Logged as warnings: count of candidates and their names

### Graceful degradation
- If the Stage B catalog has zero rows: `ValueError("No sources in catalog")`
- If all catalog queries fail: table is written with all `*_matched=False` — script exits 0
- `run_stage_c` never raises on catalog query failure — only on missing/unreadable input file

### NaN convention
- Separation, flux, flux_ratio columns use `NaN` for unmatched sources (not 0 or -1)

---

## File Map

| File | Action |
|------|--------|
| `dsa110_continuum/catalog/stage_c.py` | **Create** — `run_stage_c()` |
| `scripts/stage_c_crossmatch.py` | **Create** — CLI wrapper |
| `tests/test_stage_c_crossmatch.py` | **Create** — 10 tests |

No existing files modified.

---

## Test Infrastructure Notes

- Mock `dsa110_continuum.catalog.query.cone_search` via `unittest.mock.patch`
- Build synthetic Aegean FITS with `astropy.io.fits` + 5-row binary table
- Use `tempfile.NamedTemporaryFile(delete=False)` + `tempfile.TemporaryDirectory()`
- No AegeanTools, no real catalogs, no network access needed
