# Research: coverage-aware epoch QA completeness gate

- **Scope**: internal codebase only (diagnosis already done in
  `outputs/green-refactor-issue-96/rfi-rebuild-2026-07-10/COMPLETENESS_INVESTIGATION.md`).
- **Codebase state**: commit `635c5d9`, branch `main`, 2026-07-11. Working tree
  clean for tracked files.
- **Question**: what must change in `main` to land the coverage-aware
  completeness fix, and which callers assume the old `EpochQAResult` schema?

## Problem (from the completed investigation)

`measure_epoch_qa()` queries NVSS in a rectangular RA/Dec bounding box derived
from the mosaic WCS (`dsa110_continuum/photometry/epoch_qa.py:185-188`), but
the mosaic support is a rounded Dec strip covering only ~67% of that box on
2026-01-25T2200. Sources on blank (NaN) pixels are counted as misses:
`_peak_in_box` returns 0.0 on all-NaN boxes (`epoch_qa.py:107-110`), so
unobserved sky silently deflates completeness. Measured breakdown of 383
catalog sources: 171 recovered, 70 genuine misses on valid pixels, 120 on NaN
pixels, 22 off-grid. Raw completeness 44.6% (FAIL); coverage-corrected
171/241 = 71.0% (PASS at the unchanged 60% threshold).

## Main vs patched state

| Aspect | `main` (`photometry/epoch_qa.py`) | Patched (`outputs/.../epoch_qa_patched.py`) |
| --- | --- | --- |
| Denominator | `n_recovered / n_catalog` (`epoch_qa.py:216`) | `n_recovered / n_covered` (patched:231) |
| Coverage test | none | `if not np.isfinite(data[cy, cx]): continue` before counting (patched:215-217) |
| Gate SKIP | `n_catalog < QA_MIN_CATALOG_SOURCES` (`epoch_qa.py:227`) | `n_covered < QA_MIN_CATALOG_SOURCES` (patched:242) |
| Dataclass | no `n_covered` (`epoch_qa.py:46-59`) | `n_covered: int = 0` after `qa_result`, before `ratios` (patched:65) |
| Module docstring | gate 2 described vs all NVSS | gate 2 described vs *covered* NVSS |

The patched file is otherwise byte-identical in helpers, constants
(`QA_COMPLETENESS_MIN = 0.60`, `QA_MIN_CATALOG_SOURCES = 5` unchanged), and
gate-1/gate-3 logic. `to_dict()` picks up `n_covered` automatically via
`asdict` (patched:68-72).

## Consumers of the `EpochQAResult` schema

- `scripts/batch_pipeline.py:89-95` — `QA_CSV_FIELDS` lists the CSV schema;
  `write_qa_summary_row` (`batch_pipeline.py:964-993`) does
  `row.update(qa.to_dict())` into a `csv.DictWriter(..., extrasaction="ignore")`.
  Without a field-list update, `n_covered` would be silently dropped from the
  QA summary CSV. Needs `n_covered` added to `QA_CSV_FIELDS`. Caveat: an
  existing CSV file keeps its old header (header only written when the file is
  created), so the new column applies to fresh files.
- `dsa110_continuum/photometry/epoch_qa_plot.py:87-88` — panel 2 label is
  `f"{result.n_recovered}/{result.n_catalog}"`; should become
  recovered/covered per the investigation resolution.
- `dsa110_continuum/qa/provenance.py:195-200` — `record_epoch` copies
  `mosaic_rms_mjy` and `completeness_frac` from the QA object inside
  `try/except AttributeError`; additive `n_covered` recording is safe and
  keeps provenance interpretable.
- `scripts/canary_history.py:115-133` — builds a JSON entry with
  `n_recovered`/`n_catalog`; all readers use `.get(...)`, so adding
  `n_covered` is additive and non-breaking.
- `scripts/run_canary.sh:55-80` — prints `n_recovered / n_catalog` and gates
  on `n_recovered >= 3`; attribute access only, unaffected, optional label
  update.
- `dsa110_continuum/qa/epoch_log.py:22` — schema-free JSONL append; no change.
- `dsa110_continuum/source_finding/core.py:317-434` — a *separate*
  source-finding completeness checker (Aegean-based, its own dataclass); out
  of scope.

## Tests

- `tests/test_epoch_qa.py` — 15 tests, keyword-free of `n_covered`. Synthetic
  mosaic helper `_make_test_fits` (`test_epoch_qa.py:22-60`) produces a fully
  finite 500×500 image; no NaN-coverage case exists. The new
  `TestCoverageAwareCompleteness` class needs a variant that blanks a region
  of the image to NaN and places catalog sources on it.
- `tests/test_epoch_qa_plot.py:9-24,43-60` — constructs `EpochQAResult` via
  keywords; `n_covered=0` default keeps these passing, but display fixtures
  should set `n_covered` once the plot label uses it.
- `tests/test_maistro_control_plane.py`, `tests/test_dev_tools.py` — plain
  dict fixtures read with `.get`; tolerant, no change required.

## Synthesis

The fix is a faithful port of the already-validated patched file into
`dsa110_continuum/photometry/epoch_qa.py`, plus schema-aware updates at three
consumer sites (`QA_CSV_FIELDS`, plot label, provenance record) and additive
fields in the canary history entry. Threshold (0.60) and minimum-source count
(5) are unchanged — the change corrects the measurement, not the operating
point. The 70 genuine on-support misses (near-zero relative weight, elevated
edge RMS) are a separate image-quality problem and are explicitly NOT
addressed here; the stricter `dsa110-mosaic-quality-gate/v1` science gate
still fails this epoch (central RMS 8.98 > 8.0, edge/interior 2.35 > 2.0).

## Gaps / risks

- Appending `n_covered` to `QA_CSV_FIELDS` shifts the column layout for rows
  appended to a pre-existing `qa_summary.csv` (old header, new row shape is
  avoided only because `DictWriter` writes by fieldname — old files simply
  gain an unlabeled trailing column). Acceptable operationally; noted for the
  plan.
- `n_covered` placed after `qa_result` with a default keeps positional
  construction (`EpochQAResult(n_catalog, n_recovered, ...)`) working; no
  callers construct positionally today, but the patched ordering is retained.
