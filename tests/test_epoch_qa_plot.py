"""Tests for dsa110_continuum.photometry.epoch_qa_plot."""
from __future__ import annotations

import numpy as np
from dsa110_continuum.photometry.epoch_qa import EpochQAResult
from dsa110_continuum.photometry.epoch_qa_plot import plot_epoch_qa


def _dummy_result(qa_result: str = "PASS") -> EpochQAResult:
    return EpochQAResult(
        n_catalog=20,
        n_recovered=14,
        completeness_frac=0.70,
        median_ratio=0.93,
        mosaic_rms_mjy=8.6,
        ratio_gate="PASS",
        completeness_gate="PASS",
        rms_gate="PASS",
        qa_result=qa_result,
    )


def test_plot_epoch_qa_creates_png(tmp_path):
    out = tmp_path / "qa_diag.png"
    rng = np.random.default_rng(99)
    ratios = rng.uniform(0.7, 1.3, 20).tolist()
    tile_rms = rng.uniform(7, 12, 11).tolist()
    plot_epoch_qa(_dummy_result(), ratios, tile_rms, str(out), epoch_label="2026-01-25T2200")
    assert out.exists()
    assert out.stat().st_size > 1000


def test_plot_epoch_qa_fail_result(tmp_path):
    out = tmp_path / "qa_fail.png"
    plot_epoch_qa(_dummy_result("FAIL"), [], [], str(out), epoch_label="2026-02-15T0000")
    assert out.exists()
    assert out.stat().st_size > 1000


def test_plot_epoch_qa_sparse_inputs(tmp_path):
    """Single ratio, single tile — should not crash."""
    out = tmp_path / "qa_sparse.png"
    result = EpochQAResult(
        n_catalog=1, n_recovered=1, completeness_frac=1.0,
        median_ratio=0.95, mosaic_rms_mjy=9.0,
        ratio_gate="PASS", completeness_gate="SKIP", rms_gate="PASS",
        qa_result="PASS",
    )
    plot_epoch_qa(result, [0.95], [9.0], str(out))
    assert out.exists()


def test_plot_epoch_qa_empty_everything(tmp_path):
    """Completely empty inputs — should not crash."""
    out = tmp_path / "qa_empty.png"
    result = EpochQAResult(
        n_catalog=0, n_recovered=0, completeness_frac=0.0,
        median_ratio=float("nan"), mosaic_rms_mjy=50.0,
        ratio_gate="FAIL", completeness_gate="SKIP", rms_gate="FAIL",
        qa_result="FAIL",
    )
    plot_epoch_qa(result, [], [], str(out))
    assert out.exists()
