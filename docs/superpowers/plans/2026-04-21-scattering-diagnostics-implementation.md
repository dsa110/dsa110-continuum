# Scattering Diagnostics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `co_orig`/`co_syn` coefficient storage to `PatchScore`, build `scattering_diagnostics.py` with `plot_scattering_overview` and `plot_patch_coefficients`, and wire `plot_scattering_overview` into `EpochOrchestrator` so WARN/REJECT epochs auto-save a PNG.

**Architecture:** Three layers — data model extension (`scattering_qa.py`), visualization module (`scattering_diagnostics.py`), and orchestrator wiring (`epoch_orchestrator.py`). The visualizer takes plain numpy arrays and has no dependency on `torch` or the `scattering` library. The orchestrator wiring is wrapped in try/except so a missing library never breaks a run.

**Tech Stack:** numpy, matplotlib, SciencePlots (`["science", "notebook"]`), existing `FigureConfig`/`PlotStyle` pattern from `stage_a_diagnostics.py`.

---

## File Map

- **Modify:** `dsa110_continuum/qa/scattering_qa.py` — extend `PatchScore`, update `score_patch`, add `PatchScore.to_dict()`
- **Create:** `dsa110_continuum/visualization/scattering_diagnostics.py` — `plot_scattering_overview`, `plot_patch_coefficients`
- **Modify:** `dsa110_continuum/pipeline/epoch_orchestrator.py` — wire scattering overview into `run_epoch`
- **Modify:** `tests/test_scattering_qa.py` — add 2 new tests for coefficient storage
- **Create:** `tests/test_scattering_diagnostics.py` — 3 tests for the visualizer + orchestrator wiring

---

### Task 1: Extend PatchScore + update score_patch + tests

**Files:**
- Modify: `dsa110_continuum/qa/scattering_qa.py`
- Modify: `tests/test_scattering_qa.py`

#### Step 1.1 — Extend PatchScore with optional numpy fields

In `dsa110_continuum/qa/scattering_qa.py`, replace the `PatchScore` dataclass with:

```python
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
    co_orig: "np.ndarray | None" = field(default=None, repr=False)
    co_syn:  "np.ndarray | None" = field(default=None, repr=False)

    def to_dict(self) -> dict:
        """Return a JSON-serialisable dict, excluding numpy array fields."""
        return {
            "tile_name": self.tile_name,
            "x_min": self.x_min,
            "x_max": self.x_max,
            "y_min": self.y_min,
            "y_max": self.y_max,
            "score": self.score,
            "n_finite": self.n_finite,
        }
```

Add `field` to the existing `from dataclasses import dataclass` import:
```python
from dataclasses import dataclass, field
```

- [ ] Make these changes now. Do not touch any other code in the file.

#### Step 1.2 — Update score_patch to return (score, co_orig, co_syn)

`score_patch` currently returns a plain `float`. Change it to return a tuple `(float, np.ndarray | None, np.ndarray | None)`.

Replace the function signature and docstring:

```python
def score_patch(
    patch: np.ndarray,
    stc,
    synthesis_steps: int = 50,
) -> tuple[float, "np.ndarray | None", "np.ndarray | None"]:
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
    tuple of (score, co_orig, co_syn)
        score : float
            Normalized dot product in [0, 1]; nan if patch unusable.
        co_orig : np.ndarray or None
            Raw scattering covariance vector for the original patch.
            None if the patch was rejected (>50% NaN).
        co_syn : np.ndarray or None
            Raw scattering covariance vector for the synthesized reference.
            None if the patch was rejected.
    """
```

Replace the early-return NaN guard:
```python
    if n_finite < patch.size * 0.5:
        return float("nan"), None, None
```

Replace the final return:
```python
    return float(np.dot(co_orig / norm_orig, co_syn / norm_syn)), co_orig, co_syn
```

Also replace the zero-norm guard:
```python
    if norm_orig < 1e-12 or norm_syn < 1e-12:
        return float("nan"), None, None
```

- [ ] Make these changes now.

#### Step 1.3 — Update check_tile_scattering to unpack the tuple and store coefficients

In `check_tile_scattering`, the scoring loop currently reads:

```python
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
```

Replace with:

```python
        co_orig_arr: "np.ndarray | None" = None
        co_syn_arr:  "np.ndarray | None" = None
        try:
            s, co_orig_arr, co_syn_arr = score_patch(patch, stc, synthesis_steps=synthesis_steps)
        except Exception as exc:  # noqa: BLE001
            log.warning("score_patch failed for %s: %s", fp.tile_name, exc)
            s = float("nan")

        patch_scores.append(PatchScore(
            tile_name=fp.tile_name,
            x_min=fp.x_min, x_max=fp.x_max,
            y_min=fp.y_min, y_max=fp.y_max,
            score=s,
            n_finite=n_finite,
            co_orig=co_orig_arr,
            co_syn=co_syn_arr,
        ))
```

- [ ] Make these changes now.

#### Step 1.4 — Write two new tests in tests/test_scattering_qa.py

Add these two tests at the end of the file:

```python
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
        # Alternate to simulate orig vs syn responses
        coef = coef_orig if call_count["n"] == 1 else coef_syn
        return {"for_synthesis_iso": torch.tensor(coef[None, :])}

    mock_stc = MagicMock()
    mock_stc.J = 7
    mock_stc.L = 4
    mock_stc.scattering_cov.side_effect = mock_scattering_cov

    with patch("scattering.synthesis", return_value=patch_data[None, :]):
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
```

- [ ] Add these tests now.

#### Step 1.5 — Run the tests

```bash
cd /home/user/workspace/dsa110-continuum
python -m pytest tests/test_scattering_qa.py -m "not slow" -v 2>&1 | tail -20
```

Expected: 8 passed (tests 1–6 still pass, 7 and 8 are the two new ones — tests numbered 8 and 9 in the file).

- [ ] Run and verify all pass.

#### Step 1.6 — Commit

```bash
cd /home/user/workspace/dsa110-continuum
git add dsa110_continuum/qa/scattering_qa.py tests/test_scattering_qa.py
git commit -m "feat(scattering_qa): store co_orig/co_syn in PatchScore; score_patch returns tuple"
```

- [ ] Commit.

---

### Task 2: scattering_diagnostics.py — plot_scattering_overview + plot_patch_coefficients

**Files:**
- Create: `dsa110_continuum/visualization/scattering_diagnostics.py`
- Create: `tests/test_scattering_diagnostics.py`

#### Step 2.1 — Write the failing tests first

Create `tests/test_scattering_diagnostics.py`:

```python
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
```

- [ ] Create the test file now.

#### Step 2.2 — Run tests to confirm they fail

```bash
cd /home/user/workspace/dsa110-continuum
python -m pytest tests/test_scattering_diagnostics.py -v 2>&1 | tail -10
```

Expected: 3 errors (ImportError — module does not exist yet).

- [ ] Run and confirm failure.

#### Step 2.3 — Create scattering_diagnostics.py

Create `dsa110_continuum/visualization/scattering_diagnostics.py` with this exact content:

```python
"""Scattering transform QA diagnostic plots for the DSA-110 continuum pipeline.

Provides two visualization functions:

- plot_scattering_overview  — spatial heatmap of per-patch scores for one mosaic
- plot_patch_coefficients   — lollipop + delta bar chart for a single flagged patch

Both functions:
- Accept outputs from dsa110_continuum.qa.scattering_qa (no torch/scattering dependency)
- Apply PlotStyle.PUBLICATION (SciencePlots ["science", "notebook"]) via FigureConfig
- Write PNG to the caller-supplied output path
- Degrade gracefully: scienceplots absent → plain matplotlib style
"""
from __future__ import annotations

import logging
import math
from pathlib import Path

import numpy as np

from dsa110_continuum.visualization.config import FigureConfig, PlotStyle

log = logging.getLogger(__name__)

# Threshold constants — mirrors scattering_qa._SCORE_WARN / _SCORE_FAIL
_SCORE_WARN: float = 0.85
_SCORE_FAIL: float = 0.70


def _setup_matplotlib() -> None:
    """Configure matplotlib for headless rendering (idempotent)."""
    import matplotlib
    matplotlib.use("Agg")


def plot_scattering_overview(
    result,
    output_path: str | Path,
    config: FigureConfig | None = None,
) -> None:
    """Write a spatial heatmap of per-patch scattering scores to *output_path*.

    Parameters
    ----------
    result : ScatteringQAResult
        Output of check_tile_scattering().
    output_path : str or Path
        Destination PNG path. Parent directory is created if absent.
    config : FigureConfig or None
        Plot configuration. Defaults to PlotStyle.PUBLICATION.
    """
    _setup_matplotlib()
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    import matplotlib.patches as mpatches

    if config is None:
        config = FigureConfig(style=PlotStyle.PUBLICATION)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    patches = result.patch_scores

    # Determine mosaic extent from patch bounds
    all_x_max = max(p.x_max for p in patches) if patches else 256
    all_y_max = max(p.y_max for p in patches) if patches else 256

    # Colormap: green (1.0) → yellow (WARN) → red (FAIL → 0)
    cmap = plt.cm.RdYlGn
    norm = mcolors.Normalize(vmin=0.0, vmax=1.0)

    with config.style_context():
        fig, ax = plt.subplots(figsize=(8, 5))

        for ps in patches:
            score = ps.score
            color = "lightgrey" if math.isnan(score) else cmap(norm(score))
            rect = mpatches.FancyBboxPatch(
                (ps.x_min, ps.y_min),
                ps.x_max - ps.x_min,
                ps.y_max - ps.y_min,
                boxstyle="round,pad=2",
                facecolor=color,
                edgecolor="white",
                linewidth=0.8,
            )
            ax.add_patch(rect)
            label = "NaN" if math.isnan(score) else f"{score:.3f}"
            ax.text(
                (ps.x_min + ps.x_max) / 2,
                (ps.y_min + ps.y_max) / 2,
                label,
                ha="center", va="center",
                fontsize=8,
                color="black" if not math.isnan(score) and score > 0.5 else "white",
            )

        # Threshold lines on colorbar via dummy ScalarMappable
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
        cbar.set_label("Scattering similarity score")
        cbar.ax.axhline(_SCORE_WARN, color="gold",   linewidth=1.5, linestyle="--", label=f"WARN ({_SCORE_WARN})")
        cbar.ax.axhline(_SCORE_FAIL, color="crimson", linewidth=1.5, linestyle="--", label=f"FAIL ({_SCORE_FAIL})")
        cbar.ax.legend(loc="lower left", fontsize=7, framealpha=0.7)

        gate_color = {"PASS": "green", "WARN": "orange", "FAIL": "red"}.get(result.gate, "grey")
        ax.set_title(
            f"Scattering QA — gate: {result.gate}  "
            f"(median={result.median_score:.3f}, min={result.min_score:.3f})",
            color=gate_color,
            fontsize=10,
        )
        ax.set_xlim(0, all_x_max)
        ax.set_ylim(0, all_y_max)
        ax.set_xlabel("Mosaic x (pixels)")
        ax.set_ylabel("Mosaic y (pixels)")
        ax.set_aspect("equal")
        ax.invert_yaxis()  # FITS convention: y=0 at top

        fig.tight_layout()
        fig.savefig(str(output_path), dpi=config.dpi, bbox_inches="tight")
        plt.close(fig)

    log.info("Scattering overview saved to %s", output_path)


def plot_patch_coefficients(
    co_orig: np.ndarray,
    co_syn: np.ndarray,
    patch_score,
    output_path: str | Path,
    config: FigureConfig | None = None,
) -> None:
    """Write a two-panel coefficient diagnostic for one flagged patch.

    Left panel: lollipop chart of co_orig (data) vs co_syn (reference).
    Right panel: |co_orig - co_syn| delta bar chart, highlighting indices
    where delta > 2σ in red.

    Parameters
    ----------
    co_orig : np.ndarray, shape (N,)
        Scattering covariance vector for the original patch.
    co_syn : np.ndarray, shape (N,)
        Scattering covariance vector for the synthesized reference.
    patch_score : PatchScore
        Used for the figure title (tile_name and score).
    output_path : str or Path
        Destination PNG path. Parent directory is created if absent.
    config : FigureConfig or None
        Plot configuration. Defaults to PlotStyle.PUBLICATION.
    """
    _setup_matplotlib()
    import matplotlib.pyplot as plt

    if config is None:
        config = FigureConfig(style=PlotStyle.PUBLICATION)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    indices = np.arange(len(co_orig))
    delta = np.abs(co_orig - co_syn)
    delta_mean = float(np.mean(delta))
    delta_std  = float(np.std(delta))
    anomalous = delta > (delta_mean + 2 * delta_std)

    with config.style_context():
        fig, (ax_coef, ax_delta) = plt.subplots(1, 2, figsize=(12, 4))

        # --- Left: lollipop chart ---
        ax_coef.vlines(indices, 0, co_orig, colors="steelblue",  linewidth=0.6, alpha=0.7, label="data ($c_{orig}$)")
        ax_coef.vlines(indices, 0, co_syn,  colors="darkorange", linewidth=0.6, alpha=0.5, label="reference ($c_{syn}$)")
        ax_coef.plot(indices, co_orig, "o", color="steelblue",  markersize=2, alpha=0.8)
        ax_coef.plot(indices, co_syn,  "o", color="darkorange", markersize=2, alpha=0.6)
        ax_coef.set_xlabel("Coefficient index")
        ax_coef.set_ylabel("Normalized value")
        ax_coef.set_title("Scattering covariance coefficients")
        ax_coef.legend(fontsize=8)
        ax_coef.axhline(0, color="grey", linewidth=0.5, linestyle=":")

        # --- Right: delta bar chart ---
        bar_colors = np.where(anomalous, "crimson", "steelblue")
        ax_delta.bar(indices, delta, color=bar_colors, width=1.0, alpha=0.8)
        ax_delta.axhline(delta_mean + 2 * delta_std, color="crimson",
                         linewidth=1.0, linestyle="--", label=r"mean + 2$\sigma$")
        ax_delta.set_xlabel("Coefficient index")
        ax_delta.set_ylabel(r"$|c_{orig} - c_{syn}|$")
        ax_delta.set_title(f"Coefficient delta  ({int(anomalous.sum())} anomalous)")
        ax_delta.legend(fontsize=8)

        score_str = f"{patch_score.score:.4f}" if not math.isnan(patch_score.score) else "NaN"
        fig.suptitle(
            f"Patch: {patch_score.tile_name}  —  score={score_str}  "
            f"[x={patch_score.x_min}:{patch_score.x_max}, y={patch_score.y_min}:{patch_score.y_max}]",
            fontsize=10,
        )
        fig.tight_layout()
        fig.savefig(str(output_path), dpi=config.dpi, bbox_inches="tight")
        plt.close(fig)

    log.info("Patch coefficient diagnostic saved to %s", output_path)
```

- [ ] Create the file now.

#### Step 2.4 — Run tests to confirm they pass

```bash
cd /home/user/workspace/dsa110-continuum
python -m pytest tests/test_scattering_diagnostics.py -v 2>&1 | tail -10
```

Expected: 3 passed.

- [ ] Run and verify all pass.

#### Step 2.5 — Run full non-slow suite to check for regressions

```bash
cd /home/user/workspace/dsa110-continuum
python -m pytest tests/ -m "not slow" -q 2>&1 | tail -10
```

Expected: same pass count as before + 5 new (tests 8, 9 from Task 1 and 3 from Task 2) = no regressions.

- [ ] Run and verify no regressions.

#### Step 2.6 — Commit

```bash
cd /home/user/workspace/dsa110-continuum
git add dsa110_continuum/visualization/scattering_diagnostics.py tests/test_scattering_diagnostics.py
git commit -m "feat(visualization): add scattering_diagnostics — overview heatmap + coefficient bar chart"
```

- [ ] Commit.

---

### Task 3: Wire plot_scattering_overview into EpochOrchestrator.run_epoch

**Files:**
- Modify: `dsa110_continuum/pipeline/epoch_orchestrator.py`
- Modify: `tests/test_epoch_orchestrator.py`

#### Step 3.1 — Locate the wiring point in run_epoch

The wiring goes between Step 4 (decision) and Step 5 (persist) in `run_epoch`. After `decision = _qa_status_to_decision(qa_result.status)` and before `result = EpochRunResult(...)`.

Find this block (approximately lines 500–515 in the current file):

```python
        decision = _qa_status_to_decision(qa_result.status)
        logger.info(
            "Epoch %s → %s (%s), tiles=%d, rms=%.3g Jy/b",
            epoch_id,
            decision.value,
            qa_result.status.value,
            n_tiles,
            measured_rms,
        )

        result = EpochRunResult(
```

- [ ] Read the file to confirm the exact lines before editing.

#### Step 3.2 — Insert scattering QA + PNG block

Insert this block between the `logger.info(...)` call and the `result = EpochRunResult(...)` line:

```python
        # ── Step 3b: Scattering texture QA + diagnostic PNG for WARN/REJECT ──
        scattering_png_path: Optional[str] = None
        if mosaic_path is not None and decision in (EpochDecision.WARN, EpochDecision.REJECT):
            try:
                from dsa110_continuum.qa.scattering_qa import check_tile_scattering
                from dsa110_continuum.visualization.scattering_diagnostics import (
                    plot_scattering_overview,
                )
                _scat_result = check_tile_scattering(mosaic_path)
                _out_dir = Path(mosaic_path).parent
                scattering_png_path = str(_out_dir / "scattering_overview.png")
                plot_scattering_overview(_scat_result, scattering_png_path)
                notes.append(
                    f"Scattering QA: gate={_scat_result.gate} "
                    f"median={_scat_result.median_score:.4f} "
                    f"min={_scat_result.min_score:.4f} "
                    f"png={scattering_png_path}"
                )
                logger.info(
                    "Scattering QA PNG saved: %s (gate=%s)",
                    scattering_png_path,
                    _scat_result.gate,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("Scattering QA skipped: %s", exc)
                notes.append(f"Scattering QA skipped: {exc}")
```

- [ ] Insert this block now.

#### Step 3.3 — Write the failing test

Add one test to `tests/test_epoch_orchestrator.py` at the end of the file:

```python
# ---------------------------------------------------------------------------
# Test: scattering overview PNG auto-saved on WARN/REJECT epoch
# ---------------------------------------------------------------------------
def test_run_epoch_saves_scattering_png_on_warn(tmp_path):
    """EpochOrchestrator saves scattering_overview.png when decision is WARN."""
    import os
    from unittest.mock import MagicMock, patch
    from dsa110_continuum.pipeline.epoch_orchestrator import EpochOrchestrator, EpochDecision
    from dsa110_continuum.qa.scattering_qa import PatchScore, _build_result

    # Build a synthetic WARN ScatteringQAResult
    patches = [
        PatchScore("grid_0_0", 0, 256, 0, 256, score=0.92, n_finite=65536),
        PatchScore("grid_0_1", 256, 512, 0, 256, score=0.78, n_finite=65536),
    ]
    warn_result = _build_result(patches, tile_source="grid")

    # Synthetic mosaic FITS
    mosaic_dir = tmp_path / "epoch"
    mosaic_dir.mkdir()
    mosaic_path = str(mosaic_dir / "epoch_mosaic.fits")

    # Write a minimal valid FITS so EpochOrchestrator can read RMS
    try:
        from astropy.io import fits as af
        import numpy as np
        data = np.random.default_rng(0).standard_normal((64, 64)).astype(np.float32) * 1e-3
        af.writeto(mosaic_path, data, overwrite=True)
    except ImportError:
        pytest.skip("astropy not available")

    orch = EpochOrchestrator(
        output_dir=str(tmp_path),
        db_path=None,
    )

    with (
        patch.object(orch, "_write_mosaic", return_value=mosaic_path),
        patch(
            "dsa110_continuum.pipeline.epoch_orchestrator.check_tile_scattering",
            return_value=warn_result,
        ),
        patch(
            "dsa110_continuum.pipeline.epoch_orchestrator.plot_scattering_overview",
        ) as mock_plot,
    ):
        result = orch.run_epoch(
            "2026-01-25T22:00:00",
            tile_paths=["dummy.fits"],
            write_mosaic=True,
        )

    assert result.decision in (EpochDecision.WARN, EpochDecision.REJECT, EpochDecision.ACCEPT)
    expected_png = str(mosaic_dir / "scattering_overview.png")
    mock_plot.assert_called_once()
    call_args = mock_plot.call_args
    assert call_args[0][1] == expected_png or str(call_args[0][1]) == expected_png
```

**Important note on the patch path:** the test patches `check_tile_scattering` and `plot_scattering_overview` at the `epoch_orchestrator` module level, not at their source modules. This requires importing them at module level in `epoch_orchestrator.py`. However, since those imports are lazy (inside the `try` block), use `patch` with the source module path and ensure the lazy imports are replaced.

Actually, the cleanest approach for testability: lift the lazy imports out of the try block into module-level imports under a try/except guard at the top of `epoch_orchestrator.py`:

```python
# Optional dependencies — absent when scattering library not installed
try:
    from dsa110_continuum.qa.scattering_qa import check_tile_scattering as _check_tile_scattering
    from dsa110_continuum.visualization.scattering_diagnostics import plot_scattering_overview as _plot_scattering_overview
    _SCATTERING_AVAILABLE = True
except ImportError:
    _SCATTERING_AVAILABLE = False
    _check_tile_scattering = None  # type: ignore[assignment]
    _plot_scattering_overview = None  # type: ignore[assignment]
```

Then replace the step 3b block to use those module-level names:

```python
        if mosaic_path is not None and decision in (EpochDecision.WARN, EpochDecision.REJECT) and _SCATTERING_AVAILABLE:
            try:
                _scat_result = _check_tile_scattering(mosaic_path)
                _out_dir = Path(mosaic_path).parent
                scattering_png_path = str(_out_dir / "scattering_overview.png")
                _plot_scattering_overview(_scat_result, scattering_png_path)
                notes.append(
                    f"Scattering QA: gate={_scat_result.gate} "
                    f"median={_scat_result.median_score:.4f} "
                    f"min={_scat_result.min_score:.4f} "
                    f"png={scattering_png_path}"
                )
                logger.info(
                    "Scattering QA PNG saved: %s (gate=%s)",
                    scattering_png_path,
                    _scat_result.gate,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("Scattering QA skipped: %s", exc)
                notes.append(f"Scattering QA skipped: {exc}")
```

And the test patches at `dsa110_continuum.pipeline.epoch_orchestrator._check_tile_scattering` and `dsa110_continuum.pipeline.epoch_orchestrator._plot_scattering_overview`.

Update the test accordingly:

```python
    with (
        patch.object(orch, "_write_mosaic", return_value=mosaic_path),
        patch(
            "dsa110_continuum.pipeline.epoch_orchestrator._check_tile_scattering",
            return_value=warn_result,
        ),
        patch(
            "dsa110_continuum.pipeline.epoch_orchestrator._plot_scattering_overview",
        ) as mock_plot,
        patch(
            "dsa110_continuum.pipeline.epoch_orchestrator._SCATTERING_AVAILABLE",
            True,
        ),
    ):
```

- [ ] Add the module-level try/except imports to `epoch_orchestrator.py`.
- [ ] Replace the step 3b block to use `_check_tile_scattering` and `_plot_scattering_overview`.
- [ ] Add the test to `tests/test_epoch_orchestrator.py`.

#### Step 3.4 — Run the new orchestrator test

```bash
cd /home/user/workspace/dsa110-continuum
python -m pytest tests/test_epoch_orchestrator.py -v -k "scattering" 2>&1 | tail -15
```

Expected: 1 passed.

- [ ] Run and verify pass.

#### Step 3.5 — Run the full non-slow suite

```bash
cd /home/user/workspace/dsa110-continuum
python -m pytest tests/ -m "not slow" -q 2>&1 | tail -10
```

Expected: no regressions (all previously passing tests still pass).

- [ ] Run and verify.

#### Step 3.6 — Commit

```bash
cd /home/user/workspace/dsa110-continuum
git add dsa110_continuum/pipeline/epoch_orchestrator.py tests/test_epoch_orchestrator.py
git commit -m "feat(orchestrator): auto-save scattering overview PNG on WARN/REJECT epochs"
```

- [ ] Commit.

---

### Task 4: Push and verify

#### Step 4.1 — Final test count

```bash
cd /home/user/workspace/dsa110-continuum
python -m pytest tests/ -m "not slow" -q 2>&1 | tail -5
```

Record the final pass count.

- [ ] Run and record.

#### Step 4.2 — Push to GitHub

```bash
cd /home/user/workspace/dsa110-continuum
git remote set-url origin https://<PAT>@github.com/dsa110/dsa110-continuum.git
git push origin main 2>&1 | tail -5
```

- [ ] Push.

#### Step 4.3 — Confirm pushed SHA

```bash
cd /home/user/workspace/dsa110-continuum
git log --oneline -4
```

- [ ] Record and report the pushed SHA.
