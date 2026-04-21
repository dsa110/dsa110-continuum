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
    mock_stc.J = 7
    mock_stc.L = 4
    mock_stc.scattering_cov.return_value = mock_cov

    with patch("scattering.synthesis", return_value=patch_data[None, :]):
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
