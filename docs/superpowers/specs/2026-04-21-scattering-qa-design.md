# Scattering Transform QA — Design Spec

**Date:** 2026-04-21  
**Author:** Jakob Faber  
**Status:** Approved

---

## 1. Motivation

The existing DQA pipeline catches well-defined, quantitative failures: RMS ratio out of range, dynamic range below threshold, pixel coverage missing. It cannot detect morphological anomalies — PSF artefacts, RFI streak patterns, stitching rings, sidelobe impostors, directional calibration failures — because these are non-Gaussian, spatially structured, and poorly captured by any scalar threshold.

The scattering transform provides a compact, multi-scale, rotation-sensitive descriptor of image texture. Because it is stable to small deformations but sensitive to structural changes, it captures exactly the class of anomalies the pipeline currently misses. Crucially, the synthesis mode allows the transform to generate a phase-randomized reference image with identical scattering statistics to the input — making the comparison self-referential and requiring no labelled training data or accumulated history.

---

## 2. Scope

- New module: `dsa110_continuum/qa/scattering_qa.py`
- New tests: `tests/test_scattering_qa.py` (6 tests)
- Wire into: `scripts/source_finding.py`, after the existing image gate check
- New dependency: `scattering` package (DSA-2000 fork at `gitlab.com/dsa-2000/dat/scattering_transform`), imported lazily

---

## 3. Algorithm

For each patch extracted from the mosaic:

1. Extract a `patch_size × patch_size` numpy array (float32) from the mosaic data
2. Compute scattering covariance coefficients: `stc.scattering_cov(patch[None,...])['for_synthesis_iso']`
3. Synthesize a phase-randomized reference: `scattering.synthesis(estimator_name='s_cov_iso', target=patch[None,...], mode='image', steps=synthesis_steps, ...)`
4. Compute scattering covariance coefficients of the synthesized reference
5. Similarity score = normalized dot product of the two coefficient vectors: `dot(co/||co||, co_syn/||co_syn||)`. Ranges from 1.0 (texture is fully self-consistent) to 0.0 (completely anomalous)

A perfectly Gaussian noise patch will have a score near 1.0 because the scattering transform of Gaussian noise is fully captured by its second-order statistics and synthesis reproduces it faithfully. A patch containing structured artefacts (sidelobes, streaks, stitching rings) will score lower because the synthesis cannot reproduce the non-Gaussian structure.

**Why synthesis over a stored reference:** Synthesis requires no stored reference and no history. Every patch is compared against a statistically consistent version of itself. This makes the check unconditionally valid from the first epoch.

---

## 4. Patch Extraction Strategy

### Primary: WCS-derived tile footprints

`_get_tile_footprints(mosaic_path, tile_dir)` walks `<tile_dir>/tile*/wsclean_out/*image.fits`, uses `astropy.wcs.WCS` to reproject each tile's pixel corners into the mosaic coordinate system, and returns a list of `TileFootprint` named-tuples:

```python
TileFootprint(
    tile_name: str,        # e.g. "tile00"
    x_min: int,            # pixel bounds in mosaic (clipped to mosaic shape)
    x_max: int,
    y_min: int,
    y_max: int,
    ra_center: float,      # sky center of tile
    dec_center: float,
)
```

Patches are extracted from within each tile's footprint. Tiles in the simulated data overlap by ~50% (as verified from header inspection), so patches near tile boundaries will be scored within whichever tile's footprint they are attributed to first.

### Fallback: patch grid

If `tile_dir` is None, tile FITS files are absent, or WCS reprojection fails, fall back to a regular grid of non-overlapping `patch_size × patch_size` patches tiling the mosaic. With a 517×1188 mosaic and `patch_size=256`, this produces 8 patches covering ~88% of the mosaic area (164 px remainder in X, 5 px in Y — both discarded). Validation against real tile footprints shows every patch maps to one or two tiles, so the fallback is a faithful approximation of tile-level granularity.

The fallback is transparent: the `ScatteringQAResult` records `tile_source="wcs"` or `tile_source="grid"` to indicate which path was taken.

---

## 5. Components

### `TileFootprint` (named tuple)

Fields: `tile_name`, `x_min`, `x_max`, `y_min`, `y_max`, `ra_center`, `dec_center`

### `PatchScore` (dataclass)

```python
@dataclass
class PatchScore:
    tile_name: str           # e.g. "tile00" or "grid_0_0"
    x_min: int               # patch pixel bounds in mosaic
    x_max: int
    y_min: int
    y_max: int
    score: float             # normalized dot product, [0, 1]
    n_finite: int            # number of finite pixels in patch
```

### `ScatteringQAResult` (dataclass)

```python
@dataclass
class ScatteringQAResult:
    patch_scores: list[PatchScore]
    median_score: float
    min_score: float
    min_score_patch: PatchScore         # the most anomalous patch
    tile_source: Literal["wcs", "grid"] # which extraction path was used
    gate: Literal["PASS", "WARN", "FAIL"]
```

Gate thresholds (named constants):
```python
_SCORE_WARN = 0.85   # WARN if min_score below this
_SCORE_FAIL = 0.70   # FAIL if min_score below this
```

These are initial estimates. Real values will be calibrated from production data — they are named constants precisely so they can be updated without touching logic.

### `_get_scattering_calculator(npix, J, L)` (module-level cached)

Returns a `scattering.Scattering2d` instance. Cached by `(npix, J, L)` key using a module-level dict. Filter bank generation is expensive (~1s); caching ensures it only runs once per unique patch size per process.

### `score_patch(patch: np.ndarray, stc, synthesis_steps: int) -> float`

Core computation. Accepts a 2D float32 array. Handles the NaN mask: replaces NaNs with zeros before passing to the scattering transform (the library requires finite input). If the patch is >50% NaN, returns `float("nan")` rather than a misleading score.

### `_get_tile_footprints(mosaic_path, tile_dir) -> list[TileFootprint]`

WCS reprojection logic. Returns empty list on any failure (triggering fallback). Catches `WcsError`, `FileNotFoundError`, `OSError`, and any astropy exception.

### `check_tile_scattering(mosaic_path, tile_dir=None, patch_size=256, J=7, L=4, synthesis_steps=50) -> ScatteringQAResult`

Public entry point. Full algorithm:
1. Load mosaic FITS data once
2. Call `_get_tile_footprints`; if empty, build patch grid
3. For each tile footprint (or patch grid cell), extract `patch_size × patch_size` subarray
4. Call `score_patch` for each
5. Aggregate into `ScatteringQAResult`
6. Log per-patch scores and overall gate at INFO level; log WARN/ERROR for anomalous patches

---

## 6. Parameter Defaults and Rationale

| Parameter | Default | Rationale |
|-----------|---------|-----------|
| `patch_size` | 256 | Power of 2, required by scattering library; covers ~1.4°×1.4° at 20"/pix |
| `J` | 7 | `int(log2(256)) - 1 = 7`; maximum meaningful scale depth for 256px patches |
| `L` | 4 | 4 orientations; balances sensitivity and runtime |
| `synthesis_steps` | 50 | Empirically determined: similarity converges to >0.9999 at 10 steps; 50 provides margin. Runtime: ~5 min for 8 patches on CPU (vs. 35 min for 300 steps) |

---

## 7. Performance

Benchmarked on CPU (no GPU in pipeline sandbox):
- `scattering_cov` on 256×256 patch: ~1.0s
- `synthesis` at 50 steps: ~40s per patch
- 8 patches total: ~5.4 minutes

This is acceptable for a once-per-epoch QA step that runs after the computationally expensive BANE+Aegean source finding. If GPU becomes available (production cluster), runtime drops to under 30 seconds total.

---

## 8. Wiring into `source_finding.py`

Inserted after the existing image gate block, before `run_source_finding`, in the same try/except pattern:

```python
# ── Scattering texture QA ─────────────────────────────────────────────────
try:
    from dsa110_continuum.qa.scattering_qa import check_tile_scattering
    from dsa110_continuum.qa.epoch_log import append_epoch_qa
    scat_qa = check_tile_scattering(
        mosaic_path,
        tile_dir=str(Path(mosaic_path).parent.parent / "step5"),
    )
    log.info(
        "Scattering QA: median_score=%.4f  min_score=%.4f  [%s]  source=%s",
        scat_qa.median_score, scat_qa.min_score, scat_qa.gate, scat_qa.tile_source,
    )
    if scat_qa.gate == "FAIL":
        log.error(
            "Scattering QA FAIL: most anomalous patch is %s (score=%.4f)",
            scat_qa.min_score_patch.tile_name, scat_qa.min_score,
        )
    append_epoch_qa({
        "stage": "scattering_qa",
        "mosaic_path": str(mosaic_path),
        "median_score": round(scat_qa.median_score, 4),
        "min_score": round(scat_qa.min_score, 4),
        "min_score_tile": scat_qa.min_score_patch.tile_name,
        "n_patches": len(scat_qa.patch_scores),
        "tile_source": scat_qa.tile_source,
        "gate": scat_qa.gate,
    })
except Exception as exc:
    log.warning("Scattering texture QA skipped: %s", exc)
```

The entire block is non-blocking. A missing `scattering` library, absent tile directory, or any runtime error produces a warning and continues.

---

## 9. Tests (6 tests in `tests/test_scattering_qa.py`)

All tests use synthetic numpy arrays — no FITS I/O, no `scattering` library dependency where avoidable. The `score_patch` tests mock `scattering.Scattering2d` to avoid the 40s synthesis runtime in CI.

| Test | What it checks |
|------|----------------|
| `test_score_patch_identical_images` | Mocked: same image vs itself → score = 1.0 |
| `test_score_patch_nan_heavy_returns_nan` | >50% NaN patch → `float("nan")` |
| `test_scattering_result_has_all_fields` | `ScatteringQAResult` has all 6 documented fields |
| `test_gate_fail_on_low_score` | min_score < `_SCORE_FAIL` → gate = "FAIL" |
| `test_gate_warn_on_mid_score` | min_score between `_SCORE_FAIL` and `_SCORE_WARN` → gate = "WARN" |
| `test_get_tile_footprints_fallback` | Missing tile_dir → returns empty list (triggers grid fallback) |

One integration test (marked `@pytest.mark.slow`, skipped in normal CI) exercises the full `check_tile_scattering` call on a real 256×256 patch to verify the real scattering library produces a score in [0, 1].

---

## 10. Dependencies

- `scattering` (DSA-2000 fork): install from `gitlab.com/dsa-2000/dat/scattering_transform` via `pip install .`
- `torch` (already present as scattering dependency)
- `appdirs` (required by scattering backend)
- All other dependencies already in the pipeline environment

Add to `requirements.txt` or `setup.cfg` with a comment pointing to the GitLab source, since it is not on PyPI.

---

## 11. Future: Coefficient Accumulation (Option C)

Once a baseline of N ≥ 20 epochs exists, `score_patch` can optionally compare against the running population mean and covariance of `for_synthesis_iso` coefficients rather than against a synthesized reference. This is strictly more powerful — it learns the normal texture distribution of DSA-110 images — but requires history. The module is designed so this path can be added by extending `score_patch` with an optional `reference_coef` argument, without changing the public interface of `check_tile_scattering`.

---

## 12. Files Changed

| File | Action |
|------|--------|
| `dsa110_continuum/qa/scattering_qa.py` | Create |
| `tests/test_scattering_qa.py` | Create |
| `scripts/source_finding.py` | Modify (insert QA block) |
| `requirements.txt` | Modify (add scattering dependency note) |
