"""Tests for scattering transform texture QA."""
import math
import os
import sys
import tempfile
import types
from collections import namedtuple
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


def _mock_scattering_module(synthesis):
    module = types.ModuleType("scattering")
    module.synthesis = synthesis
    return patch.dict(sys.modules, {"scattering": module})


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
    mock_stc.J = 7
    mock_stc.L = 4
    mock_stc.scattering_cov.return_value = mock_cov

    with _mock_scattering_module(MagicMock(return_value=patch_data[None, :])):
        score, _co_orig, _co_syn = score_patch(patch_data, mock_stc, synthesis_steps=5)

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
    with _mock_scattering_module(MagicMock()):
        result, co_orig, co_syn = score_patch(patch_data, mock_stc, synthesis_steps=5)

    assert math.isnan(result), f"Expected nan, got {result}"
    assert co_orig is None
    assert co_syn is None


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
    """Overall gate is WARN when _SCORE_FAIL <= min_score < _SCORE_WARN (0.85)."""
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
    """_build_patch_grid on 517x1188 mosaic with patch_size=256 gives 8 patches."""
    from dsa110_continuum.qa.scattering_qa import _build_patch_grid

    patches = _build_patch_grid(mosaic_shape=(517, 1188), patch_size=256)
    assert len(patches) == 8, f"Expected 8 patches, got {len(patches)}"
    # All patches must be within mosaic bounds
    for p in patches:
        assert p.x_min >= 0 and p.x_max <= 1188
        assert p.y_min >= 0 and p.y_max <= 517
        assert p.x_max - p.x_min == 256
        assert p.y_max - p.y_min == 256


# ---------------------------------------------------------------------------
# Test 7: check_tile_scattering on a synthetic mosaic FITS with mocked scattering
# ---------------------------------------------------------------------------
def test_check_tile_scattering_integration(tmp_path):
    """check_tile_scattering on a 512x512 FITS exercises the full grid + gate path.

    Real code paths covered: FITS read, ``_build_patch_grid`` fallback (since
    ``tile_dir=None``), per-patch slicing, score aggregation, and gate logic.
    The CPU-heavy ``scattering`` package is mocked using the same pattern as
    the surrounding score_patch tests so the test runs in the default suite.
    """
    import torch
    from astropy.io import fits

    # Build a synthetic 512x512 FITS (4 patches of 256x256 fit exactly)
    rng = np.random.default_rng(0)
    data = rng.standard_normal((512, 512)).astype(np.float32) * 0.01
    data[256, 256] = 1.0  # inject a bright source

    mosaic_path = tmp_path / "mosaic.fits"
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
    fits.writeto(str(mosaic_path), data, hdr, overwrite=True)

    # Mock stc: returns identical coefficients for orig and synthesis
    coef = np.ones(371, dtype=np.float32)
    mock_cov = {"for_synthesis_iso": torch.tensor(coef[None, :])}
    mock_stc = MagicMock()
    mock_stc.J = 7
    mock_stc.L = 4
    mock_stc.scattering_cov.return_value = mock_cov

    # ``synthesis`` must return a (1, patch_size, patch_size)-shaped array
    fake_syn = np.zeros((1, 256, 256), dtype=np.float32)

    from dsa110_continuum.qa.scattering_qa import check_tile_scattering

    with _mock_scattering_module(MagicMock(return_value=fake_syn)), \
         patch(
             "dsa110_continuum.qa.scattering_qa._get_scattering_calculator",
             return_value=mock_stc,
         ):
        result = check_tile_scattering(
            mosaic_path,
            tile_dir=None,          # force grid fallback
            patch_size=256,
            J=7,
            L=4,
            synthesis_steps=5,
        )

    assert result.gate in ("PASS", "WARN", "FAIL")
    assert len(result.patch_scores) == 4   # 512/256 x 512/256 = 4 patches
    assert result.tile_source == "grid"
    for ps in result.patch_scores:
        # float32 dot product can land 1 ulp above 1.0 when vectors are identical
        assert math.isnan(ps.score) or 0.0 <= ps.score <= 1.0 + 1e-5, (
            f"Score out of range: {ps.score}"
        )
    # Identical mock coefficients -> exact-match score of 1.0 -> PASS gate.
    assert result.gate == "PASS"
    assert math.isclose(result.min_score, 1.0, abs_tol=1e-5)


# ---------------------------------------------------------------------------
# Test 8: score_patch returns coefficient vectors alongside score
# ---------------------------------------------------------------------------
def test_score_patch_returns_coefficients():
    """score_patch returns (score, co_orig, co_syn) where co_orig/co_syn are ndarray."""
    import torch
    from dsa110_continuum.qa.scattering_qa import score_patch

    rng = np.random.default_rng(7)
    patch_data = rng.standard_normal((256, 256)).astype(np.float32)

    coef_orig = np.ones(371, dtype=np.float32) * 2.0
    coef_syn  = np.ones(371, dtype=np.float32) * 2.0
    call_count = {"n": 0}

    def mock_scattering_cov(img):
        call_count["n"] += 1
        coef = coef_orig if call_count["n"] == 1 else coef_syn
        return {"for_synthesis_iso": torch.tensor(coef[None, :])}

    mock_stc = MagicMock()
    mock_stc.J = 7
    mock_stc.L = 4
    mock_stc.scattering_cov.side_effect = mock_scattering_cov

    with _mock_scattering_module(MagicMock(return_value=patch_data[None, :])):
        score, co_orig, co_syn = score_patch(patch_data, mock_stc, synthesis_steps=5)

    assert isinstance(co_orig, np.ndarray), "co_orig should be ndarray"
    assert isinstance(co_syn, np.ndarray), "co_syn should be ndarray"
    assert co_orig.shape == (371,)
    assert co_syn.shape == (371,)
    assert math.isclose(score, 1.0, abs_tol=1e-5)


# ---------------------------------------------------------------------------
# Test 9: PatchScore.to_dict excludes numpy fields; None default works
# ---------------------------------------------------------------------------
def test_patch_score_to_dict_excludes_numpy():
    """PatchScore.to_dict() omits co_orig/co_syn; None default leaves them absent."""
    from dsa110_continuum.qa.scattering_qa import PatchScore

    # With no coefficients (default)
    ps = PatchScore("t0", 0, 256, 0, 256, score=0.9, n_finite=65536)
    assert ps.co_orig is None
    assert ps.co_syn is None
    d = ps.to_dict()
    assert "co_orig" not in d
    assert "co_syn" not in d
    assert d["tile_name"] == "t0"
    assert d["score"] == 0.9

    # With coefficients stored — to_dict still excludes them
    arr = np.ones(371, dtype=np.float32)
    ps2 = PatchScore("t1", 0, 256, 0, 256, score=0.8, n_finite=65536,
                     co_orig=arr, co_syn=arr * 0.9)
    assert ps2.co_orig is not None
    d2 = ps2.to_dict()
    assert "co_orig" not in d2
    assert "co_syn" not in d2
    assert d2["score"] == 0.8
