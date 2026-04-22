"""Tests for scattering transform diagnostic visualizations."""
import math
import os
import tempfile

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Test 1: plot_scattering_overview writes a PNG for a WARN result
# ---------------------------------------------------------------------------
def test_plot_scattering_overview_writes_png():
    """plot_scattering_overview writes a PNG file to the given output path."""
    from dsa110_continuum.qa.scattering_qa import PatchScore, _build_result
    from dsa110_continuum.visualization.scattering_diagnostics import plot_scattering_overview

    patches = [
        PatchScore("grid_0_0", 0,   256,   0, 256, score=0.92, n_finite=65536),
        PatchScore("grid_0_1", 256, 512,   0, 256, score=0.78, n_finite=65536),
        PatchScore("grid_1_0", 0,   256, 256, 512, score=0.88, n_finite=65536),
        PatchScore("grid_1_1", 256, 512, 256, 512, score=0.95, n_finite=65536),
    ]
    result = _build_result(patches, tile_source="grid")  # min=0.78 -> WARN

    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = os.path.join(tmpdir, "scattering_overview.png")
        plot_scattering_overview(result, out_path)
        assert os.path.exists(out_path), "PNG was not written"
        assert os.path.getsize(out_path) > 1000, "PNG suspiciously small"


# ---------------------------------------------------------------------------
# Test 2: plot_patch_coefficients writes a PNG for two random coefficient vectors
# ---------------------------------------------------------------------------
def test_plot_patch_coefficients_writes_png():
    """plot_patch_coefficients writes a PNG given two 1-D numpy arrays."""
    from dsa110_continuum.qa.scattering_qa import PatchScore
    from dsa110_continuum.visualization.scattering_diagnostics import plot_patch_coefficients

    rng = np.random.default_rng(0)
    co_orig = rng.random(371).astype(np.float32)
    co_syn  = rng.random(371).astype(np.float32)
    ps = PatchScore("grid_0_1", 256, 512, 0, 256, score=0.78, n_finite=65536,
                    co_orig=co_orig, co_syn=co_syn)

    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = os.path.join(tmpdir, "coeff_diag.png")
        plot_patch_coefficients(co_orig, co_syn, ps, out_path)
        assert os.path.exists(out_path), "PNG was not written"
        assert os.path.getsize(out_path) > 1000, "PNG suspiciously small"


# ---------------------------------------------------------------------------
# Test 3: plot_scattering_overview handles all-NaN scores gracefully
# ---------------------------------------------------------------------------
def test_plot_scattering_overview_all_nan():
    """plot_scattering_overview does not raise when all patch scores are NaN."""
    from dsa110_continuum.qa.scattering_qa import PatchScore, _build_result
    from dsa110_continuum.visualization.scattering_diagnostics import plot_scattering_overview

    patches = [
        PatchScore("grid_0_0", 0, 256, 0, 256, score=float("nan"), n_finite=0),
        PatchScore("grid_0_1", 256, 512, 0, 256, score=float("nan"), n_finite=0),
    ]
    result = _build_result(patches, tile_source="grid")

    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = os.path.join(tmpdir, "scattering_nan.png")
        plot_scattering_overview(result, out_path)  # must not raise
        assert os.path.exists(out_path)
