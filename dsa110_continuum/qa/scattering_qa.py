"""Scattering transform texture QA for DSA-110 continuum pipeline.

Scores image texture quality per mosaic tile by comparing each 256x256
patch against a phase-randomized synthesis of itself. A similarity score
near 1.0 means the texture is self-consistent (normal noise); lower scores
indicate non-Gaussian structure (artefacts, RFI streaks, stitching rings).

References
----------
Cheung et al. 2020, MNRAS 499, 5902 -- scattering transform theory
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
import scattering

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Gate thresholds -- calibrate from production data; change here only
# ---------------------------------------------------------------------------
_SCORE_WARN: float = 0.85   # min_score below this -> WARN
_SCORE_FAIL: float = 0.70   # min_score below this -> FAIL

# ---------------------------------------------------------------------------
# Scattering calculator cache -- filter bank construction is expensive (~1 s)
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
        _STC_CACHE[key] = scattering.Scattering2d(
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
    import torch

    # Guard: too many NaNs -> uninformative score
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
    syn = scattering.synthesis(
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
        # No usable patches -- return a WARN result with sentinel values
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
