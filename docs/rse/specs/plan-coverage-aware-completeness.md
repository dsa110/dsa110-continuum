# Plan: coverage-aware epoch QA completeness gate

## Overview

Port the validated coverage-aware completeness fix (from
`outputs/green-refactor-issue-96/rfi-rebuild-2026-07-10/epoch_qa_patched.py`)
into `dsa110_continuum/photometry/epoch_qa.py` on branch
`agent/coverage-aware-completeness`. Catalog sources landing on non-finite
mosaic pixels are excluded from the completeness denominator; `n_covered` is
added to `EpochQAResult`; the gate SKIPs when `n_covered <
QA_MIN_CATALOG_SOURCES`. Thresholds are unchanged. Consumers that assume the
old schema (CSV field list, QA plot label, provenance record, canary history)
are updated.

## Current State Analysis

- `dsa110_continuum/photometry/epoch_qa.py:216` — `completeness_frac =
  n_recovered / n_catalog`; `epoch_qa.py:227` — SKIP on `n_catalog < 5`;
  dataclass (`epoch_qa.py:46-59`) has no `n_covered`.
- `scripts/batch_pipeline.py:89-95` — `QA_CSV_FIELDS` lacks `n_covered`;
  `write_qa_summary_row` (`batch_pipeline.py:964-993`) drops unknown keys via
  `extrasaction="ignore"`.
- `dsa110_continuum/photometry/epoch_qa_plot.py:87` — panel label
  `n_recovered/n_catalog`.
- `dsa110_continuum/qa/provenance.py:195-200` — records `completeness_frac`
  only.
- `scripts/canary_history.py:115-133` — JSON entry without `n_covered`.
- `tests/test_epoch_qa.py` — 15 tests, no coverage case; synthetic mosaic is
  fully finite.
- `tests/test_epoch_qa_plot.py:9-24` — fixture without `n_covered`.

Full analysis: [research-coverage-aware-completeness.md](research-coverage-aware-completeness.md).

## Desired End State

- `measure_epoch_qa` on a mosaic whose catalog box includes blank sky reports
  `n_covered < n_catalog` and computes `completeness_frac = n_recovered /
  n_covered`.
- `EpochQAResult.to_dict()` includes `n_covered`; the batch QA CSV schema
  carries it; the QA plot shows `recovered/covered`.
- `tests/test_epoch_qa.py` gains `TestCoverageAwareCompleteness` (3 tests);
  all epoch-QA-related tests pass under casa6 python.

## What We're NOT Doing

- Not changing `QA_COMPLETENESS_MIN` (0.60), `QA_MIN_CATALOG_SOURCES` (5), or
  any other gate constant.
- Not addressing the ~70 genuine on-support misses (edge RMS / near-zero
  weight) — that is image-quality work, tracked separately.
- Not touching `dsa110_continuum/source_finding/core.py` (separate
  completeness checker).
- Not claiming the 2026-01-25T2200 mosaic is science-ready — the
  `dsa110-mosaic-quality-gate/v1` gate still FAILs on central/edge RMS.
- Not rewriting existing `qa_summary.csv` files (old files keep their header;
  new column applies to fresh files).
- Not bulk-fixing pre-existing ruff violations.

## Implementation Approach

Faithful port of the patched file (it is byte-identical to main except for the
coverage logic), then additive schema updates at consumer sites, test-first
where new behavior is introduced. Work on branch
`agent/coverage-aware-completeness` off `main` (635c5d9). No commits unless
the user asks.

## Implementation Phases

### Phase 1 — branch + failing tests

**Objective**: encode the new behavior as failing tests.

1. Create the branch:

   ```bash
   cd /data/dsa110-continuum && git checkout -b agent/coverage-aware-completeness
   ```

2. Append to `tests/test_epoch_qa.py` (after `TestToDictCSV`,
   `tests/test_epoch_qa.py:203-218`) — a helper that blanks a pixel-column
   band to NaN plus the new test class:

   ```python
   def _make_partial_coverage_fits(
       tmp_path: Path,
       n_finite_sources: int = 6,
       n_blank_sources: int = 6,
   ) -> tuple[Path, list[tuple[float, float, float]], int]:
       """Synthetic mosaic where columns >= 350 are NaN (unobserved sky).

       Returns (fits_path, catalog, n_expected_covered): catalog mixes
       sources embedded on finite pixels with sources placed on the blank
       band; only the former should enter the completeness denominator.
       """
       ny, nx = 500, 500
       rng = np.random.default_rng(42)
       data = rng.normal(0, 0.0085, (ny, nx)).astype(np.float32)

       w = WCS(naxis=2)
       w.wcs.crpix = [nx // 2, ny // 2]
       w.wcs.cdelt = [-6.0 / 3600, 6.0 / 3600]
       w.wcs.crval = [45.0, 16.1]
       w.wcs.ctype = ["RA---SIN", "DEC--SIN"]

       catalog: list[tuple[float, float, float]] = []
       for y in np.linspace(100, 400, n_finite_sources, dtype=int):
           x = 150
           data[y, x] += 0.5
           sky = w.pixel_to_world(x, y)
           catalog.append((sky.ra.deg, sky.dec.deg, 500.0))
       for y in np.linspace(120, 380, n_blank_sources, dtype=int):
           x = 420
           sky = w.pixel_to_world(x, y)
           catalog.append((sky.ra.deg, sky.dec.deg, 500.0))

       data[:, 350:] = np.nan

       hdr = w.to_header()
       hdr["BUNIT"] = "Jy/beam"
       hdu = fits.PrimaryHDU(data=data[np.newaxis, np.newaxis], header=hdr)
       out = tmp_path / "partial_mosaic.fits"
       hdu.writeto(str(out), overwrite=True)
       return out, catalog, n_finite_sources


   class TestCoverageAwareCompleteness:
       """Sources on non-finite pixels must not count as completeness misses."""

       def test_blank_pixel_sources_excluded_from_denominator(self, tmp_path):
           fits_path, catalog, n_finite = _make_partial_coverage_fits(tmp_path)
           nvss_db = _make_nvss_db(tmp_path, catalog)
           result = measure_epoch_qa(str(fits_path), str(nvss_db))

           assert result.n_catalog == len(catalog)
           assert result.n_covered == n_finite
           assert result.n_recovered == n_finite
           assert result.completeness_frac == 1.0
           assert result.completeness_gate == "PASS"

       def test_gate_skips_when_covered_below_minimum(self, tmp_path):
           # 2 finite + 8 blank: n_catalog=10 >= 5 but n_covered=2 < 5 → SKIP
           fits_path, catalog, n_finite = _make_partial_coverage_fits(
               tmp_path, n_finite_sources=2, n_blank_sources=8,
           )
           nvss_db = _make_nvss_db(tmp_path, catalog)
           result = measure_epoch_qa(str(fits_path), str(nvss_db))

           assert result.n_catalog == 10
           assert result.n_covered == 2
           assert result.completeness_gate == "SKIP"

       def test_n_covered_serialised_in_to_dict(self, tmp_path):
           fits_path, catalog, n_finite = _make_partial_coverage_fits(tmp_path)
           nvss_db = _make_nvss_db(tmp_path, catalog)
           d = measure_epoch_qa(str(fits_path), str(nvss_db)).to_dict()

           assert d["n_covered"] == n_finite
           assert "ratios" not in d
   ```

3. Run and watch the new tests fail (AttributeError / KeyError on
   `n_covered`):

   ```bash
   PYTHONPATH=/data/dsa110-continuum /opt/miniforge/envs/casa6/bin/python \
     -m pytest tests/test_epoch_qa.py::TestCoverageAwareCompleteness -v
   ```

**Verification**: 3 tests collected, 3 fail; the 15 pre-existing tests in the
file still pass.

### Phase 2 — core fix in `epoch_qa.py`

**Objective**: make Phase 1 tests pass by porting the patched logic.
Depends on Phase 1.

1. Module docstring (`epoch_qa.py:1-9`): replace the gate-2 line with the
   patched wording (`epoch_qa_patched.py:1-15`) — completeness is over
   *covered* NVSS sources; "covered" = source lands on a finite mosaic pixel.

2. Dataclass (`epoch_qa.py:46-59`): insert after `qa_result`:

   ```python
       n_covered: int = 0              # catalog sources landing on finite pixels
   ```

   (before `ratios`, keeping `ratios` last; matches `epoch_qa_patched.py:65`).

3. Measurement loop (`epoch_qa.py:192-212`): initialise `n_covered = 0`
   alongside `n_recovered`; after the edge-margin check
   (`epoch_qa.py:201-202`) insert:

   ```python
           # The bounding-box catalog query includes sky the mosaic never
           # observed (blank strip ends, coverage gaps).  A source on a
           # non-finite pixel cannot be recovered at any image quality, so it
           # is excluded from the completeness denominator.
           if not np.isfinite(data[cy, cx]):
               continue
           n_covered += 1
   ```

4. Gate evaluation (`epoch_qa.py:216` and `epoch_qa.py:227`):

   ```python
       completeness_frac = n_recovered / n_covered if n_covered > 0 else 0.0
   ```

   ```python
       # Gate 2: detection completeness (over covered sources only)
       if n_covered < QA_MIN_CATALOG_SOURCES:
   ```

5. Constructor call (`epoch_qa.py:245-256`): add `n_covered=n_covered,` after
   `n_catalog=n_catalog,`.

6. Run:

   ```bash
   PYTHONPATH=/data/dsa110-continuum /opt/miniforge/envs/casa6/bin/python \
     -m pytest tests/test_epoch_qa.py -v
   ```

**Verification**: all 18 tests in `tests/test_epoch_qa.py` pass.

### Phase 3 — consumers

**Objective**: carry `n_covered` through CSV, plot, provenance, canary.
Depends on Phase 2.

1. `scripts/batch_pipeline.py:89-95` — insert `"n_covered"` after
   `"n_catalog"`:

   ```python
   QA_CSV_FIELDS = [
       "date", "epoch_utc", "mosaic_path",
       "n_catalog", "n_covered", "n_recovered", "completeness_frac",
       "median_ratio", "ratio_gate", "completeness_gate",
       "rms_gate", "mosaic_rms_mjy",
       "qa_result", "gaincal_used",
   ]
   ```

2. `dsa110_continuum/photometry/epoch_qa_plot.py:87` — label shows
   recovered/covered, falling back to `n_catalog` for pre-fix result objects
   loaded from old records:

   ```python
       n_str = f"{result.n_recovered}/{result.n_covered or result.n_catalog}"
   ```

3. `dsa110_continuum/qa/provenance.py:195-200` — inside the existing
   `try/except AttributeError`, after the `completeness_frac` line:

   ```python
                   rec["n_covered"] = epoch_qa.n_covered
   ```

4. `scripts/canary_history.py:115-133` — in the entry dict, after
   `"n_catalog": result.n_catalog,`:

   ```python
           "n_covered": result.n_covered,
   ```

5. `tests/test_epoch_qa_plot.py:9-24` — add `n_covered=18,` to the
   `_dummy_result` constructor kwargs (any value between `n_recovered` and
   `n_catalog`); the two inline constructions at
   `tests/test_epoch_qa_plot.py:43-60` are left as-is to exercise the
   `n_covered=0` fallback path.

6. Run:

   ```bash
   PYTHONPATH=/data/dsa110-continuum /opt/miniforge/envs/casa6/bin/python \
     -m pytest tests/test_epoch_qa.py tests/test_epoch_qa_plot.py -v
   ```

**Verification**: both files' tests pass.

### Phase 4 — validation sweep

**Objective**: prove no consumer regressed. Depends on Phase 3.

1. Targeted suite:

   ```bash
   PYTHONPATH=/data/dsa110-continuum /opt/miniforge/envs/casa6/bin/python \
     -m pytest tests/test_epoch_qa.py tests/test_epoch_qa_plot.py \
     tests/test_epoch_log.py tests/test_dev_tools.py \
     tests/test_maistro_control_plane.py -q
   ```

2. Lint only the touched files:

   ```bash
   ruff check dsa110_continuum/photometry/epoch_qa.py \
     dsa110_continuum/photometry/epoch_qa_plot.py \
     dsa110_continuum/qa/provenance.py scripts/batch_pipeline.py \
     scripts/canary_history.py tests/test_epoch_qa.py tests/test_epoch_qa_plot.py
   ```

   New violations introduced by this change are fixed; pre-existing ones are
   left alone.

3. Write `docs/rse/specs/validation-coverage-aware-completeness.md` (via
   `ai-research-workflows:validating-implementations`).

**Verification**: targeted suite green; no new ruff violations.

## Success Criteria

### Automated Verification

- [ ] `pytest tests/test_epoch_qa.py -q` → 18 passed (15 existing + 3 new).
- [ ] `pytest tests/test_epoch_qa_plot.py tests/test_epoch_log.py
      tests/test_dev_tools.py tests/test_maistro_control_plane.py -q` → all pass.
- [ ] `rg -n "n_recovered / n_catalog" dsa110_continuum/photometry/epoch_qa.py`
      → no hits.
- [ ] `rg -n "n_covered" scripts/batch_pipeline.py` → hit in `QA_CSV_FIELDS`.
- [ ] `ruff check` on touched files reports no *new* violations vs `main`.

### Manual Verification

- [ ] Diff review: gate constants (`QA_COMPLETENESS_MIN`,
      `QA_MIN_CATALOG_SOURCES`, `QA_RMS_LIMIT_MJY`) unchanged.
- [ ] (Optional, H17 data) re-run the gate on
      `/stage/dsa110-contimg/images/mosaic_2026-01-25/2026-01-25T2200_mosaic.fits`
      and confirm completeness ≈ 0.71 PASS with `n_covered` ≈ 241 — matches
      the investigation numbers.

## Testing Strategy

- **Unit**: `TestCoverageAwareCompleteness` (Phase 1) covers denominator
  exclusion, SKIP-on-low-coverage, and serialisation. Existing 15 tests guard
  gate-1/gate-3 and fully-covered behavior (on a fully finite image
  `n_covered == n_catalog`, so their assertions are unaffected).
- **Integration**: plot/provenance/CSV consumers exercised by
  `tests/test_epoch_qa_plot.py`, `tests/test_dev_tools.py`,
  `tests/test_maistro_control_plane.py`.
- **Manual**: optional real-mosaic re-run (above).

## References

- [research-coverage-aware-completeness.md](research-coverage-aware-completeness.md)
- `outputs/green-refactor-issue-96/rfi-rebuild-2026-07-10/COMPLETENESS_INVESTIGATION.md`
- `outputs/green-refactor-issue-96/rfi-rebuild-2026-07-10/epoch_qa_patched.py`
- Files analyzed: `dsa110_continuum/photometry/epoch_qa.py`,
  `dsa110_continuum/photometry/epoch_qa_plot.py`,
  `dsa110_continuum/qa/provenance.py`, `dsa110_continuum/qa/epoch_log.py`,
  `scripts/batch_pipeline.py`, `scripts/canary_history.py`,
  `scripts/run_canary.sh`, `tests/test_epoch_qa.py`,
  `tests/test_epoch_qa_plot.py`
