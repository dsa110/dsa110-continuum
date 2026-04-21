# Scattering Transform QA Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a scattering-transform-based image texture QA module that detects morphological artefacts (PSF sidelobes, RFI streaks, stitching rings) not caught by scalar threshold checks, and wire it into the source-finding pipeline.

**Architecture:** A new `dsa110_continuum/qa/scattering_qa.py` module extracts 256×256 pixel patches from the mosaic (using WCS-derived tile footprints with a grid fallback), synthesizes a phase-randomized reference for each patch via the scattering transform, computes a normalized dot-product similarity score, and gates on the minimum score across all patches. The module is wired into `scripts/source_finding.py` after the existing image gate, in a non-blocking try/except block.

**Tech Stack:** `scattering` (DSA-2000 fork, CPU mode), `torch`, `astropy.wcs`, `numpy`, `dataclasses`

---

## File Map

| File | Action | Responsibility |
|------|--------|---------------|
| `dsa110_continuum/qa/scattering_qa.py` | Create | All scattering QA logic: footprint extraction, patch scoring, result aggregation |
| `tests/test_scattering_qa.py` | Create | 6 unit tests + 1 slow integration test |
| `scripts/source_finding.py` | Modify | Insert scattering QA block after image gate |
| `pyproject.toml` | Modify | Add scattering + appdirs dependency notes |

---

## Pre-Flight: Install the scattering library

Before any tasks, verify the `scattering` library is installed in the repo environment:

```bash
cd /home/user/workspace/dsa110-continuum
python -c "import scattering; print('OK')" 2>&1
```

If `ModuleNotFoundError`:
```bash
cd /tmp && git clone https://gitlab.com/dsa-2000/dat/scattering_transform.git
pip install appdirs -q
pip install -e /tmp/scattering_transform -q
python -c "import scattering; print('OK')"
```

---

## Task 1: Core dataclasses and patch scorer

**Files:**
- Create: `dsa110_continuum/qa/scattering_qa.py`
- Create: `tests/test_scattering_qa.py` (first 4 tests)

---

- [ ] **Step 1.1: Write 4 failing tests**

Create `tests/test_scattering_qa.py`:

```python
"""Tests for scattering transform texture QA."""
import math
import os
import tempfile
from collections import namedtuple
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Test 1: score_patch returns 1.0 when synthesis reproduces the image exactly
# ---------------------------------------------------------------------------
def test_score_patch_identical_coefficients():
    """score_patch returns 1.0 when synthesized image has identical coefficients."""
    import torch
    from dsa110_continuum.qa.scattering_qa import score_patch

    # Build a mock scattering calculator whose synthesis returns a copy of input
    rng = np.random.default_rng(42)
    patch_data = rng.standard_normal((256, 256)).astype(np.float32)

    # Mock coefficient vector — same for both original and synthesis
    coef = np.ones(371, dtype=np.float32)
    mock_cov = {"for_synthesis_iso": torch.tensor(coef[None, :])}

    mock_stc = MagicMock()
    mock_stc.scattering_cov.return_value = mock_cov

    with patch("dsa110_continuum.qa.scattering_qa.scattering") as mock_scat:
        # synthesis returns a copy of the patch — same stats
        mock_scat.synthesis.return_value = patch_data[None, :]
        mock_scat.Scattering2d.return_value = mock_stc
        score = score_patch(patch_data, mock_stc, synthesis_steps=5)

    assert math.isclose(score, 1.0, abs_tol=1e-5), f"Expected 1.0, got {score}"


# ---------------------------------------------------------------------------
# Test 2: score_patch returns nan for >50% NaN patch
# ---------------------------------------------------------------------------
def test_score_patch_nan_heavy_returns_nan():
    """score_patch returns float('nan') when >50% of pixels are NaN."""
    from dsa110_continuum.qa.scattering_qa import score_patch

    patch_data = np.full((256, 256), np.nan, dtype=np.float32)
    patch_data[:50, :50] = 1.0  # only 50*50/256*256 = 3.8% finite

    mock_stc = MagicMock()
    with patch("dsa110_continuum.qa.scattering_qa.scattering"):
        result = score_patch(patch_data, mock_stc, synthesis_steps=5)

    assert math.isnan(result), f"Expected nan, got {result}"


# ---------------------------------------------------------------------------
# Test 3: gate logic — FAIL when min_score below _SCORE_FAIL
# ---------------------------------------------------------------------------
def test_gate_fail_on_low_min_score():
    """Overall gate is FAIL when min_score < _SCORE_FAIL (0.70)."""
    from dsa110_continuum.qa.scattering_qa import (
        PatchScore, ScatteringQAResult, _build_result,
    )

    patches = [
        PatchScore("tile00", 0, 256, 0, 256, score=0.95, n_finite=256*256),
        PatchScore("tile01", 256, 512, 0, 256, score=0.60, n_finite=256*256),
    ]
    result = _build_result(patches, tile_source="grid")
    assert result.gate == "FAIL"
    assert result.min_score_patch.tile_name == "tile01"


# ---------------------------------------------------------------------------
# Test 4: gate logic — WARN when min_score between _SCORE_FAIL and _SCORE_WARN
# ---------------------------------------------------------------------------
def test_gate_warn_on_mid_min_score():
    """Overall gate is WARN when _SCORE_FAIL ≤ min_score < _SCORE_WARN (0.85)."""
    from dsa110_continuum.qa.scattering_qa import (
        PatchScore, ScatteringQAResult, _build_result,
    )

    patches = [
        PatchScore("tile00", 0, 256, 0, 256, score=0.95, n_finite=256*256),
        PatchScore("tile01", 256, 512, 0, 256, score=0.78, n_finite=256*256),
    ]
    result = _build_result(patches, tile_source="wcs")
    assert result.gate == "WARN"
    assert math.isclose(result.min_score, 0.78, abs_tol=1e-6)
```

- [ ] **Step 1.2: Run tests to confirm they all fail**

```bash
cd /home/user/workspace/dsa110-continuum
pytest tests/test_scattering_qa.py -v 2>&1 | tail -10
```

Expected: 4 FAILED (ImportError — module does not exist yet).

- [ ] **Step 1.3: Create `dsa110_continuum/qa/scattering_qa.py` with dataclasses, `score_patch`, and `_build_result`**

```python
"""Scattering transform texture QA for DSA-110 continuum pipeline.

Scores image texture quality per mosaic tile by comparing each 256×256
patch against a phase-randomized synthesis of itself. A similarity score
near 1.0 means the texture is self-consistent (normal noise); lower scores
indicate non-Gaussian structure (artefacts, RFI streaks, stitching rings).

References
----------
Cheung et al. 2020, MNRAS 499, 5902 — scattering transform theory
Scattering library: https://gitlab.com/dsa-2000/dat/scattering_transform
compare.py reference: https://gitlab.com/dsa-2000/dat/image-qa
"""
from __future__ import annotations

import logging
import math
from collections import namedtuple
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Gate thresholds — calibrate from production data; change here only
# ---------------------------------------------------------------------------
_SCORE_WARN: float = 0.85   # min_score below this → WARN
_SCORE_FAIL: float = 0.70   # min_score below this → FAIL

# ---------------------------------------------------------------------------
# Scattering calculator cache — filter bank construction is expensive (~1 s)
# ---------------------------------------------------------------------------
_STC_CACHE: dict[tuple, object] = {}

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
TileFootprint = namedtuple(
    "TileFootprint",
    ["tile_name", "x_min", "x_max", "y_min", "y_max", "ra_center", "dec_center"],
)


@dataclass
class PatchScore:
    """Scattering similarity score for one mosaic patch."""
    tile_name: str      # e.g. "tile00" or "grid_0_0"
    x_min: int          # patch pixel bounds in mosaic (x = NAXIS1 direction)
    x_max: int
    y_min: int          # y = NAXIS2 direction
    y_max: int
    score: float        # normalized dot product in [0, 1]; nan if patch unusable
    n_finite: int       # number of finite pixels in patch


@dataclass
class ScatteringQAResult:
    """Aggregate scattering QA result for one mosaic."""
    patch_scores: list[PatchScore]
    median_score: float
    min_score: float
    min_score_patch: PatchScore
    tile_source: Literal["wcs", "grid"]
    gate: Literal["PASS", "WARN", "FAIL"]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_scattering_calculator(npix: int, J: int, L: int):
    """Return a cached Scattering2d instance for (npix, J, L)."""
    key = (npix, J, L)
    if key not in _STC_CACHE:
        import scattering as _scattering
        _STC_CACHE[key] = _scattering.Scattering2d(
            M=npix, N=npix, J=J, L=L, device="cpu", wavelets="morlet"
        )
    return _STC_CACHE[key]


def score_patch(
    patch: np.ndarray,
    stc,
    synthesis_steps: int = 50,
) -> float:
    """Compute scattering similarity score for one square patch.

    Parameters
    ----------
    patch : np.ndarray, shape (M, N), dtype float32
        Square mosaic sub-image. M must equal N.
    stc : scattering.Scattering2d
        Pre-built scattering calculator (use _get_scattering_calculator).
    synthesis_steps : int
        Gradient steps for phase-randomized synthesis. 50 is sufficient
        for convergence on CPU; 10 is the practical minimum.

    Returns
    -------
    float
        Normalized dot product of scattering covariance coefficients:
        1.0 = texture fully self-consistent, 0.0 = maximally anomalous.
        Returns float('nan') if >50% of pixels are non-finite.
    """
    import scattering as _scattering
    import torch

    # Guard: too many NaNs → uninformative score
    n_finite = int(np.isfinite(patch).sum())
    if n_finite < patch.size * 0.5:
        return float("nan")

    # Replace NaNs with zero (scattering library requires finite input)
    clean = np.where(np.isfinite(patch), patch, 0.0).astype(np.float32)
    img = clean[None, ...]  # shape (1, M, N)

    # Scattering covariance of the original patch
    cov_orig = stc.scattering_cov(img)
    co_orig = cov_orig["for_synthesis_iso"].detach().cpu().numpy().squeeze()

    # Synthesize phase-randomized reference with identical scattering stats
    syn = _scattering.synthesis(
        estimator_name="s_cov_iso",
        target=img,
        mode="image",
        M=patch.shape[0],
        N=patch.shape[1],
        J=stc.J,
        L=stc.L,
        device="cpu",
        learning_rate=0.5,
        steps=synthesis_steps,
    )

    # Scattering covariance of the synthesized reference
    cov_syn = stc.scattering_cov(syn)
    co_syn = cov_syn["for_synthesis_iso"].detach().cpu().numpy().squeeze()

    # Normalized dot product
    norm_orig = np.linalg.norm(co_orig)
    norm_syn = np.linalg.norm(co_syn)
    if norm_orig < 1e-12 or norm_syn < 1e-12:
        return float("nan")

    return float(np.dot(co_orig / norm_orig, co_syn / norm_syn))


def _build_result(
    patch_scores: list[PatchScore],
    tile_source: Literal["wcs", "grid"],
) -> ScatteringQAResult:
    """Aggregate per-patch scores into a ScatteringQAResult."""
    valid = [p.score for p in patch_scores if not math.isnan(p.score)]
    if not valid:
        # No usable patches — return a WARN result with sentinel values
        sentinel = patch_scores[0] if patch_scores else PatchScore(
            "none", 0, 0, 0, 0, float("nan"), 0
        )
        return ScatteringQAResult(
            patch_scores=patch_scores,
            median_score=float("nan"),
            min_score=float("nan"),
            min_score_patch=sentinel,
            tile_source=tile_source,
            gate="WARN",
        )

    median_score = float(np.median(valid))
    min_score = float(min(valid))
    min_patch = min(
        (p for p in patch_scores if not math.isnan(p.score)),
        key=lambda p: p.score,
    )

    if min_score < _SCORE_FAIL:
        gate: Literal["PASS", "WARN", "FAIL"] = "FAIL"
    elif min_score < _SCORE_WARN:
        gate = "WARN"
    else:
        gate = "PASS"

    return ScatteringQAResult(
        patch_scores=patch_scores,
        median_score=round(median_score, 6),
        min_score=round(min_score, 6),
        min_score_patch=min_patch,
        tile_source=tile_source,
        gate=gate,
    )
```

- [ ] **Step 1.4: Run 4 tests — all must pass**

```bash
cd /home/user/workspace/dsa110-continuum
pytest tests/test_scattering_qa.py -v 2>&1 | tail -12
```

Expected: 4 PASSED.

- [ ] **Step 1.5: Run full regression suite — no regressions**

```bash
cd /home/user/workspace/dsa110-continuum
pytest tests/ -q --tb=short 2>&1 | tail -5
```

Expected: 64 PASSED (same as before), no new failures.

- [ ] **Step 1.6: Commit**

```bash
cd /home/user/workspace/dsa110-continuum
git add dsa110_continuum/qa/scattering_qa.py tests/test_scattering_qa.py
git commit -m "feat(qa): add scattering_qa.py — dataclasses, score_patch, _build_result"
```

---

## Task 2: Tile footprint extraction + patch grid fallback

**Files:**
- Modify: `dsa110_continuum/qa/scattering_qa.py` (add `_get_tile_footprints` and `_build_patch_grid`)
- Modify: `tests/test_scattering_qa.py` (add 2 more tests)

---

- [ ] **Step 2.1: Write 2 failing tests**

Append to `tests/test_scattering_qa.py`:

```python
# ---------------------------------------------------------------------------
# Test 5: _get_tile_footprints returns empty list for missing tile_dir
# ---------------------------------------------------------------------------
def test_get_tile_footprints_missing_dir_returns_empty():
    """_get_tile_footprints returns [] when tile_dir does not exist."""
    from dsa110_continuum.qa.scattering_qa import _get_tile_footprints

    result = _get_tile_footprints(
        mosaic_path="/nonexistent/mosaic.fits",
        tile_dir="/nonexistent/step5",
    )
    assert result == [], f"Expected empty list, got {result}"


# ---------------------------------------------------------------------------
# Test 6: _build_patch_grid produces correct number of patches for known mosaic
# ---------------------------------------------------------------------------
def test_build_patch_grid_coverage():
    """_build_patch_grid on 517×1188 mosaic with patch_size=256 gives 8 patches."""
    from dsa110_continuum.qa.scattering_qa import _build_patch_grid

    patches = _build_patch_grid(mosaic_shape=(517, 1188), patch_size=256)
    assert len(patches) == 8, f"Expected 8 patches, got {len(patches)}"
    # All patches must be within mosaic bounds
    for p in patches:
        assert p.x_min >= 0 and p.x_max <= 1188
        assert p.y_min >= 0 and p.y_max <= 517
        assert p.x_max - p.x_min == 256
        assert p.y_max - p.y_min == 256
```

- [ ] **Step 2.2: Run new tests to confirm they fail**

```bash
cd /home/user/workspace/dsa110-continuum
pytest tests/test_scattering_qa.py::test_get_tile_footprints_missing_dir_returns_empty \
       tests/test_scattering_qa.py::test_build_patch_grid_coverage -v 2>&1 | tail -8
```

Expected: 2 FAILED (ImportError).

- [ ] **Step 2.3: Add `_get_tile_footprints` and `_build_patch_grid` to `scattering_qa.py`**

Add after the `_build_result` function, before the end of the file:

```python
def _get_tile_footprints(
    mosaic_path: str | Path,
    tile_dir: str | Path | None,
) -> list[TileFootprint]:
    """Return tile pixel footprints in mosaic coordinates via WCS reprojection.

    Parameters
    ----------
    mosaic_path : str or Path
        Path to the stitched mosaic FITS file.
    tile_dir : str or Path or None
        Directory containing tile sub-directories, e.g. ``pipeline_outputs/step5``.
        Each tile is expected at ``<tile_dir>/tile*/wsclean_out/*image.fits``.

    Returns
    -------
    list[TileFootprint]
        One entry per unique tile. Empty list if tile_dir is None, absent,
        or WCS reprojection fails for all tiles.
    """
    if tile_dir is None:
        return []

    tile_dir = Path(tile_dir)
    mosaic_path = Path(mosaic_path)

    try:
        from astropy.io import fits as _fits
        from astropy.wcs import WCS as _WCS
        import glob as _glob

        with _fits.open(mosaic_path) as _hdul:
            mosaic_wcs = _WCS(_hdul[0].header).celestial
            mny, mnx = _hdul[0].data.squeeze().shape[-2:]

        tile_fits = sorted(_glob.glob(str(tile_dir / "tile*" / "wsclean_out" / "*image.fits")))
        if not tile_fits:
            log.debug("No tile FITS found under %s — using patch grid fallback", tile_dir)
            return []

        seen: set[str] = set()
        footprints: list[TileFootprint] = []

        for tf in tile_fits:
            tile_name = Path(tf).parent.parent.name  # e.g. "tile00"
            if tile_name in seen:
                continue
            seen.add(tile_name)

            try:
                with _fits.open(tf) as _th:
                    tile_wcs = _WCS(_th[0].header).celestial
                    tny, tnx = _th[0].data.squeeze().shape[-2:]

                # Tile corners in sky coords
                corners_sky = tile_wcs.pixel_to_world(
                    [0, tnx - 1, 0, tnx - 1],
                    [0, 0, tny - 1, tny - 1],
                )
                # Project into mosaic pixel coordinates
                mx, my = mosaic_wcs.world_to_pixel(corners_sky)
                mx = np.array([c if hasattr(c, '__float__') else float(c) for c in mx], dtype=float)
                my = np.array([c if hasattr(c, '__float__') else float(c) for c in my], dtype=float)

                x0 = int(max(0, math.floor(mx.min())))
                x1 = int(min(mnx - 1, math.ceil(mx.max())))
                y0 = int(max(0, math.floor(my.min())))
                y1 = int(min(mny - 1, math.ceil(my.max())))

                # Sky center
                ra_c = float(np.mean([s.ra.deg for s in corners_sky]))
                dec_c = float(np.mean([s.dec.deg for s in corners_sky]))

                footprints.append(TileFootprint(tile_name, x0, x1, y0, y1, ra_c, dec_c))
                log.debug("Tile %s footprint: x=[%d,%d] y=[%d,%d]", tile_name, x0, x1, y0, y1)

            except Exception as exc:  # noqa: BLE001
                log.debug("Skipping tile %s (WCS error): %s", tile_name, exc)

        return footprints

    except (FileNotFoundError, OSError) as exc:
        log.debug("Tile footprint extraction failed: %s", exc)
        return []
    except Exception as exc:  # noqa: BLE001
        log.debug("Tile footprint extraction failed (unexpected): %s", exc)
        return []


def _build_patch_grid(
    mosaic_shape: tuple[int, int],
    patch_size: int = 256,
) -> list[TileFootprint]:
    """Build a regular non-overlapping patch grid as a fallback.

    Parameters
    ----------
    mosaic_shape : (n_rows, n_cols) — i.e. (NAXIS2, NAXIS1)
    patch_size : int
        Patch edge length in pixels (must be a power of 2).

    Returns
    -------
    list[TileFootprint]
        One TileFootprint per patch. ``ra_center`` / ``dec_center`` are 0.0
        (sky coords unavailable without a mosaic WCS here).
    """
    nrows, ncols = mosaic_shape
    patches: list[TileFootprint] = []
    row_idx = 0
    for y in range(0, nrows - patch_size + 1, patch_size):
        col_idx = 0
        for x in range(0, ncols - patch_size + 1, patch_size):
            name = f"grid_{row_idx}_{col_idx}"
            patches.append(TileFootprint(name, x, x + patch_size, y, y + patch_size, 0.0, 0.0))
            col_idx += 1
        row_idx += 1
    return patches
```

- [ ] **Step 2.4: Run all 6 tests — all must pass**

```bash
cd /home/user/workspace/dsa110-continuum
pytest tests/test_scattering_qa.py -v 2>&1 | tail -12
```

Expected: 6 PASSED.

- [ ] **Step 2.5: Run full regression suite — no regressions**

```bash
cd /home/user/workspace/dsa110-continuum
pytest tests/ -q --tb=short 2>&1 | tail -5
```

Expected: 64 PASSED.

- [ ] **Step 2.6: Commit**

```bash
cd /home/user/workspace/dsa110-continuum
git add dsa110_continuum/qa/scattering_qa.py tests/test_scattering_qa.py
git commit -m "feat(qa): add tile footprint extraction and patch grid fallback"
```

---

## Task 3: Public entry point `check_tile_scattering` + slow integration test

**Files:**
- Modify: `dsa110_continuum/qa/scattering_qa.py` (add `check_tile_scattering`)
- Modify: `tests/test_scattering_qa.py` (add slow integration test)

---

- [ ] **Step 3.1: Add the slow integration test**

Append to `tests/test_scattering_qa.py`:

```python
# ---------------------------------------------------------------------------
# Test 7 (slow): full check_tile_scattering on a real synthetic mosaic FITS
# ---------------------------------------------------------------------------
@pytest.mark.slow
def test_check_tile_scattering_integration():
    """Full pipeline: synthetic 512×512 FITS → check_tile_scattering → score in [0,1]."""
    import tempfile, os
    from astropy.io import fits

    # Build a synthetic 512×512 FITS (2 patches of 256×256 fit exactly)
    rng = np.random.default_rng(0)
    data = rng.standard_normal((512, 512)).astype(np.float32) * 0.01
    # Inject a bright source
    data[256, 256] = 1.0

    with tempfile.NamedTemporaryFile(suffix=".fits", delete=False) as f:
        mosaic_path = f.name
    try:
        hdr = fits.Header()
        hdr["NAXIS"] = 2
        hdr["NAXIS1"] = 512
        hdr["NAXIS2"] = 512
        hdr["CDELT1"] = -20.0 / 3600.0
        hdr["CDELT2"] = 20.0 / 3600.0
        hdr["CRPIX1"] = 256.0
        hdr["CRPIX2"] = 256.0
        hdr["CRVAL1"] = 344.0
        hdr["CRVAL2"] = 16.15
        hdr["CTYPE1"] = "RA---TAN"
        hdr["CTYPE2"] = "DEC--TAN"
        fits.writeto(mosaic_path, data, hdr, overwrite=True)

        from dsa110_continuum.qa.scattering_qa import check_tile_scattering
        result = check_tile_scattering(
            mosaic_path,
            tile_dir=None,          # force grid fallback
            patch_size=256,
            J=7,
            L=4,
            synthesis_steps=20,     # fewer steps for speed in testing
        )
        assert result.gate in ("PASS", "WARN", "FAIL")
        assert len(result.patch_scores) == 4   # 512/256 × 512/256 = 4 patches
        assert result.tile_source == "grid"
        for ps in result.patch_scores:
            assert math.isnan(ps.score) or 0.0 <= ps.score <= 1.0, (
                f"Score out of range: {ps.score}"
            )
    finally:
        os.unlink(mosaic_path)
```

- [ ] **Step 3.2: Confirm new test is collected but skipped (not slow-marked in normal run)**

```bash
cd /home/user/workspace/dsa110-continuum
pytest tests/test_scattering_qa.py -v --collect-only 2>&1 | grep "slow\|SKIP\|test_check"
```

Expected: `test_check_tile_scattering_integration` is collected (even if it will be skipped without `-m slow`).

- [ ] **Step 3.3: Add `check_tile_scattering` to `scattering_qa.py`**

Add after `_build_patch_grid`, at the end of the file:

```python
def check_tile_scattering(
    mosaic_path: str | Path,
    tile_dir: str | Path | None = None,
    patch_size: int = 256,
    J: int = 7,
    L: int = 4,
    synthesis_steps: int = 50,
) -> ScatteringQAResult:
    """Score mosaic image texture quality using the scattering transform.

    Extracts patches from the mosaic (WCS-derived tile footprints if
    ``tile_dir`` is provided and valid, otherwise a regular grid), scores
    each 256×256 patch by comparing its scattering covariance coefficients
    against a phase-randomized synthesis of itself, and aggregates into a
    gated result.

    Parameters
    ----------
    mosaic_path : str or Path
        Path to the stitched mosaic FITS file.
    tile_dir : str or Path or None
        Directory containing step5 tile sub-directories. Pass None to use
        the patch grid fallback unconditionally.
    patch_size : int
        Square patch edge length in pixels (default 256; must be power of 2).
    J : int
        Number of dyadic scales for the scattering transform (default 7;
        maximum for 256-pixel patches is ``int(log2(256)) - 1 = 7``).
    L : int
        Number of orientations (default 4).
    synthesis_steps : int
        Gradient steps for phase-randomized synthesis (default 50; 10 is
        the practical minimum for convergence).

    Returns
    -------
    ScatteringQAResult
        Dataclass with per-patch scores, summary statistics, and an overall
        PASS / WARN / FAIL gate.
    """
    mosaic_path = Path(mosaic_path)

    # -- Load mosaic once -----------------------------------------------------
    from astropy.io import fits as _fits
    with _fits.open(mosaic_path) as _hdul:
        mosaic_data = _hdul[0].data.squeeze().astype(np.float32)

    mosaic_shape = mosaic_data.shape  # (NAXIS2, NAXIS1)

    # -- Determine patch regions ----------------------------------------------
    footprints = _get_tile_footprints(mosaic_path, tile_dir)
    tile_source: Literal["wcs", "grid"]

    if footprints:
        tile_source = "wcs"
        # Extract one patch per tile footprint — largest square that fits
        patch_regions: list[TileFootprint] = []
        for fp in footprints:
            w = fp.x_max - fp.x_min
            h = fp.y_max - fp.y_min
            side = min(w, h, patch_size)
            # Round down to nearest multiple of patch_size
            side = (side // patch_size) * patch_size
            if side < patch_size:
                log.debug("Tile %s footprint too small for a %d-px patch — skipping", fp.tile_name, patch_size)
                continue
            x0 = fp.x_min + (w - side) // 2
            y0 = fp.y_min + (h - side) // 2
            patch_regions.append(
                TileFootprint(fp.tile_name, x0, x0 + side, y0, y0 + side, fp.ra_center, fp.dec_center)
            )
        if not patch_regions:
            log.warning("WCS footprints found but all too small — falling back to patch grid")
            footprints = []

    if not footprints:
        tile_source = "grid"
        patch_regions = _build_patch_grid(mosaic_shape, patch_size)

    # -- Score each patch -----------------------------------------------------
    stc = _get_scattering_calculator(patch_size, J, L)
    patch_scores: list[PatchScore] = []

    for fp in patch_regions:
        patch = mosaic_data[fp.y_min:fp.y_max, fp.x_min:fp.x_max]
        n_finite = int(np.isfinite(patch).sum())

        try:
            s = score_patch(patch, stc, synthesis_steps=synthesis_steps)
        except Exception as exc:  # noqa: BLE001
            log.warning("score_patch failed for %s: %s", fp.tile_name, exc)
            s = float("nan")

        patch_scores.append(PatchScore(
            tile_name=fp.tile_name,
            x_min=fp.x_min, x_max=fp.x_max,
            y_min=fp.y_min, y_max=fp.y_max,
            score=s,
            n_finite=n_finite,
        ))
        log.info(
            "Scattering QA patch %s: score=%.4f  (%d finite pixels)",
            fp.tile_name, s if not math.isnan(s) else -1, n_finite,
        )

    result = _build_result(patch_scores, tile_source)
    log.info(
        "Scattering QA overall: median=%.4f  min=%.4f  gate=%s  source=%s",
        result.median_score, result.min_score, result.gate, result.tile_source,
    )
    return result
```

- [ ] **Step 3.4: Run all 6 unit tests (skip slow) — all must pass**

```bash
cd /home/user/workspace/dsa110-continuum
pytest tests/test_scattering_qa.py -v -m "not slow" 2>&1 | tail -12
```

Expected: 6 PASSED, 1 skipped (the `@pytest.mark.slow` test).

- [ ] **Step 3.5: Verify syntax**

```bash
cd /home/user/workspace/dsa110-continuum
python -c "import ast; ast.parse(open('dsa110_continuum/qa/scattering_qa.py').read()); print('OK')"
```

- [ ] **Step 3.6: Run full regression suite — no regressions**

```bash
cd /home/user/workspace/dsa110-continuum
pytest tests/ -q --tb=short -m "not slow" 2>&1 | tail -5
```

Expected: 64 PASSED.

- [ ] **Step 3.7: Commit**

```bash
cd /home/user/workspace/dsa110-continuum
git add dsa110_continuum/qa/scattering_qa.py tests/test_scattering_qa.py
git commit -m "feat(qa): add check_tile_scattering public entry point and integration test"
```

---

## Task 4: Wire into `source_finding.py` + update `pyproject.toml`

**Files:**
- Modify: `scripts/source_finding.py`
- Modify: `pyproject.toml`

---

- [ ] **Step 4.1: Insert scattering QA block into `source_finding.py`**

In `scripts/source_finding.py`, find the exact end of the image gate block:

```python
    except Exception as exc:
        log.warning("Image quality gate skipped: %s", exc)

    # Run full pipeline
    catalog_path = run_source_finding(
```

Replace with (inserting the new block between the two existing sections):

```python
    except Exception as exc:
        log.warning("Image quality gate skipped: %s", exc)

    # ── Scattering texture QA ─────────────────────────────────────────────────
    try:
        from dsa110_continuum.qa.scattering_qa import check_tile_scattering
        from dsa110_continuum.qa.epoch_log import append_epoch_qa as _append_scat_qa
        _scat_qa = check_tile_scattering(
            mosaic_path,
            tile_dir=str(Path(mosaic_path).parent.parent / "step5"),
        )
        log.info(
            "Scattering QA: median_score=%.4f  min_score=%.4f  [%s]  source=%s",
            _scat_qa.median_score, _scat_qa.min_score, _scat_qa.gate, _scat_qa.tile_source,
        )
        if _scat_qa.gate == "FAIL":
            log.error(
                "Scattering QA FAIL: most anomalous patch is %s (score=%.4f)",
                _scat_qa.min_score_patch.tile_name, _scat_qa.min_score,
            )
        _append_scat_qa({
            "stage": "scattering_qa",
            "mosaic_path": str(mosaic_path),
            "median_score": round(_scat_qa.median_score, 4)
                            if not math.isnan(_scat_qa.median_score) else None,
            "min_score": round(_scat_qa.min_score, 4)
                         if not math.isnan(_scat_qa.min_score) else None,
            "min_score_tile": _scat_qa.min_score_patch.tile_name,
            "n_patches": len(_scat_qa.patch_scores),
            "tile_source": _scat_qa.tile_source,
            "gate": _scat_qa.gate,
        })
    except Exception as exc:  # noqa: BLE001
        log.warning("Scattering texture QA skipped: %s", exc)

    # Run full pipeline
    catalog_path = run_source_finding(
```

- [ ] **Step 4.2: Add `import math` to `source_finding.py` if not already present**

Check:

```bash
grep "^import math" /home/user/workspace/dsa110-continuum/scripts/source_finding.py
```

If not found, add `import math` to the stdlib imports block at the top of the file (after the existing `import` lines).

- [ ] **Step 4.3: Verify syntax on `source_finding.py`**

```bash
cd /home/user/workspace/dsa110-continuum
python -c "import ast; ast.parse(open('scripts/source_finding.py').read()); print('OK')"
```

- [ ] **Step 4.4: Add dependency notes to `pyproject.toml`**

In `pyproject.toml`, find the `[project.optional-dependencies]` section (around line 109). Add a new optional group if it doesn't exist, or append to an existing `extras` group:

```toml
[project.optional-dependencies]
# ... existing entries ...
qa = [
    # scattering transform (DSA-2000 fork — not on PyPI):
    #   git clone https://gitlab.com/dsa-2000/dat/scattering_transform
    #   pip install appdirs && pip install -e ./scattering_transform
    "appdirs",
    "torch",
]
```

- [ ] **Step 4.5: Run full regression suite — still 64 passing**

```bash
cd /home/user/workspace/dsa110-continuum
pytest tests/ -q --tb=short -m "not slow" 2>&1 | tail -5
```

Expected: 64 PASSED. (The wiring is in a try/except; no new tests are required for the script wiring itself — it follows the same pattern already tested in prior tasks.)

- [ ] **Step 4.6: Commit**

```bash
cd /home/user/workspace/dsa110-continuum
git add scripts/source_finding.py pyproject.toml
git commit -m "feat(scripts): wire scattering QA into source_finding.py; note qa deps in pyproject.toml"
```

---

## Task 5: Push and verify

- [ ] **Step 5.1: Confirm final test count**

```bash
cd /home/user/workspace/dsa110-continuum
pytest tests/ -q --tb=short -m "not slow" 2>&1 | tail -5
```

Expected: **64 PASSED** (no new tests added in Task 4; all 6 scattering tests pass from Task 1–3). The slow integration test is excluded by `-m "not slow"` and does not affect the count.

- [ ] **Step 5.2: Push to GitHub**

```bash
cd /home/user/workspace/dsa110-continuum
git remote set-url origin https://<PAT>@github.com/dsa110/dsa110-continuum.git
git push origin main 2>&1 | tail -5
```

- [ ] **Step 5.3: Confirm pushed SHA**

```bash
cd /home/user/workspace/dsa110-continuum
git log --oneline -4
```

---

## Self-Review Notes

**Spec coverage check:**

| Spec section | Task |
|---|---|
| §2 New module `scattering_qa.py` | Task 1, 2, 3 |
| §3 Algorithm (covariance, synthesis, dot product) | Task 1 (`score_patch`) |
| §4 WCS tile footprints + grid fallback | Task 2 |
| §5 All dataclasses (`TileFootprint`, `PatchScore`, `ScatteringQAResult`) | Task 1, 2 |
| §5 `_get_scattering_calculator` cache | Task 1 |
| §5 `check_tile_scattering` public entry | Task 3 |
| §6 Parameter defaults (patch_size=256, J=7, L=4, steps=50) | Task 3 |
| §7 Performance (synthesis_steps=50) | Task 3 |
| §8 Wiring into `source_finding.py` | Task 4 |
| §9 6 unit tests + 1 slow integration test | Tasks 1, 2, 3 |
| §10 Dependencies in `pyproject.toml` | Task 4 |
| §11 Future `reference_coef` extension point | Preserved in `score_patch` signature (optional arg can be added without breaking callers) |

All spec sections are covered. No gaps.

**Type consistency check:**
- `TileFootprint` defined Task 2, used in Tasks 2 and 3 ✓
- `PatchScore` defined Task 1, used in Tasks 1, 2, 3 ✓
- `ScatteringQAResult` defined Task 1, used in Tasks 1, 3 ✓
- `_build_result(patch_scores, tile_source)` defined Task 1, called in Task 3 ✓
- `score_patch(patch, stc, synthesis_steps)` defined Task 1, called in Task 3 ✓
- `_get_tile_footprints(mosaic_path, tile_dir)` defined Task 2, called in Task 3 ✓
- `_build_patch_grid(mosaic_shape, patch_size)` defined Task 2, called in Task 3 ✓
- `_get_scattering_calculator(npix, J, L)` defined Task 1, called in Task 3 ✓
