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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np

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
    import scattering as _scattering
    key = (npix, J, L)
    if key not in _STC_CACHE:
        _STC_CACHE[key] = _scattering.Scattering2d(
            M=npix, N=npix, J=J, L=L, device="cpu", wavelets="morlet"
        )
    return _STC_CACHE[key]


def score_patch(
    patch: np.ndarray,
    stc,
    synthesis_steps: int = 50,
) -> "tuple[float, np.ndarray | None, np.ndarray | None]":
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
    import torch
    import scattering as _scattering

    # Guard: too many NaNs -> uninformative score
    n_finite = int(np.isfinite(patch).sum())
    if n_finite < patch.size * 0.5:
        return float("nan"), None, None

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
        return float("nan"), None, None

    return float(np.dot(co_orig / norm_orig, co_syn / norm_syn)), co_orig, co_syn


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
        import glob as _glob
        from astropy.io import fits as _fits
        from astropy.wcs import WCS as _WCS

        with _fits.open(mosaic_path) as _hdul:
            mosaic_wcs = _WCS(_hdul[0].header).celestial
            mny, mnx = _hdul[0].data.squeeze().shape[-2:]

        tile_fits = sorted(_glob.glob(str(tile_dir / "tile*" / "wsclean_out" / "*image.fits")))
        if not tile_fits:
            log.debug("No tile FITS found under %s -- using patch grid fallback", tile_dir)
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
                mx = np.array(
                    [float(c) if not hasattr(c, '__len__') else float(c) for c in mx],
                    dtype=float,
                )
                my = np.array(
                    [float(c) if not hasattr(c, '__len__') else float(c) for c in my],
                    dtype=float,
                )

                x0 = int(max(0, math.floor(mx.min())))
                x1 = int(min(mnx - 1, math.ceil(mx.max())))
                y0 = int(max(0, math.floor(my.min())))
                y1 = int(min(mny - 1, math.ceil(my.max())))

                # Sky center
                ra_c = float(np.mean([s.ra.deg for s in corners_sky]))
                dec_c = float(np.mean([s.dec.deg for s in corners_sky]))

                footprints.append(TileFootprint(tile_name, x0, x1, y0, y1, ra_c, dec_c))
                log.debug(
                    "Tile %s footprint: x=[%d,%d] y=[%d,%d]",
                    tile_name, x0, x1, y0, y1,
                )

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
    mosaic_shape : tuple (n_rows, n_cols) -- i.e. (NAXIS2, NAXIS1)
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
            patches.append(
                TileFootprint(name, x, x + patch_size, y, y + patch_size, 0.0, 0.0)
            )
            col_idx += 1
        row_idx += 1
    return patches


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
    each patch by comparing its scattering covariance coefficients against
    a phase-randomized synthesis of itself, and aggregates into a gated result.

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
        Number of dyadic scales (default 7; max for 256-px patches).
    L : int
        Number of orientations (default 4).
    synthesis_steps : int
        Gradient steps for phase-randomized synthesis (default 50).

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
        # Extract one patch per tile footprint -- largest square that fits
        patch_regions: list[TileFootprint] = []
        for fp in footprints:
            w = fp.x_max - fp.x_min
            h = fp.y_max - fp.y_min
            side = min(w, h, patch_size)
            # Round down to nearest multiple of patch_size
            side = (side // patch_size) * patch_size
            if side < patch_size:
                log.debug(
                    "Tile %s footprint too small for a %d-px patch -- skipping",
                    fp.tile_name, patch_size,
                )
                continue
            x0 = fp.x_min + (w - side) // 2
            y0 = fp.y_min + (h - side) // 2
            patch_regions.append(
                TileFootprint(fp.tile_name, x0, x0 + side, y0, y0 + side, fp.ra_center, fp.dec_center)
            )
        if not patch_regions:
            log.warning("WCS footprints found but all too small -- falling back to patch grid")
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
        log.info(
            "Scattering QA patch %s: score=%.4f  (%d finite pixels)",
            fp.tile_name,
            s if not math.isnan(s) else -1.0,
            n_finite,
        )

    result = _build_result(patch_scores, tile_source)
    log.info(
        "Scattering QA overall: median=%.4f  min=%.4f  gate=%s  source=%s",
        result.median_score if not math.isnan(result.median_score) else -1.0,
        result.min_score if not math.isnan(result.min_score) else -1.0,
        result.gate, result.tile_source,
    )
    return result
