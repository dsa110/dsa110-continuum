"""Composite epoch QA metric for DSA-110 mosaics.

Three independent gates must all pass for an epoch to be QA-PASS:
  1. Flux scale:        median DSA/NVSS ratio in [0.8, 1.2]
  2. Detection compl.:  >= 60% of NVSS sources >= 50 mJy recovered above 5-sigma local RMS
  3. Noise floor:       median mosaic RMS <= 18.6 mJy/beam

Design decisions are documented in docs/plans/2026-03-09-phase0-qa-infrastructure.md.
"""
from __future__ import annotations

import sqlite3
from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

QA_RATIO_LOW: float = 0.8
QA_RATIO_HIGH: float = 1.2
QA_COMPLETENESS_MIN: float = 0.60
QA_MIN_CATALOG_SOURCES: int = 5      # fewer → completeness gate is SKIP
QA_MIN_FLUX_MJY: float = 50.0       # 50 mJy NVSS threshold
QA_RECOVERY_SIGMA: float = 5.0      # detection threshold in units of local RMS
QA_MIN_RATIO_DETECTIONS: int = 3    # minimum detections for ratio gate

# 2 × empirical noise floor (recalculated for 96 antennas, 188 MHz, T_sys=25 K)
# Theoretical: ~9.3 mJy/beam → 2× floor = 18.6 mJy/beam
QA_RMS_LIMIT_MJY: float = 18.6

# Default NVSS SQLite DB path
DEFAULT_NVSS_DB: str = "/data/dsa110-contimg/state/catalogs/nvss_full.sqlite3"

_GateResult = Literal["PASS", "FAIL", "SKIP"]


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class EpochQAResult:
    """Three-gate QA verdict for one epoch mosaic."""

    n_catalog: int
    n_recovered: int
    completeness_frac: float
    median_ratio: float
    mosaic_rms_mjy: float
    ratio_gate: _GateResult
    completeness_gate: _GateResult
    rms_gate: _GateResult
    qa_result: _GateResult          # PASS only if all non-SKIP gates pass
    ratios: list[float] | None = None  # per-source DSA/NVSS ratios (for plotting)

    def to_dict(self) -> dict:
        """Return a flat dict suitable for CSV serialisation."""
        d = asdict(self)
        d.pop("ratios", None)  # exclude list from CSV row
        return d


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _image_rms_mad(data: np.ndarray) -> float:
    """Global MAD-based RMS estimate (Jy/beam). Ignores NaN."""
    flat = data[np.isfinite(data)].ravel()
    if flat.size == 0:
        return float("nan")
    return float(1.4826 * np.median(np.abs(flat - np.median(flat))))


def _local_rms(data: np.ndarray, cy: int, cx: int, outer: int = 25, inner: int = 5) -> float:
    """Local RMS in an annular box around (cy, cx)."""
    ny, nx = data.shape
    y0, y1 = max(0, cy - outer), min(ny, cy + outer)
    x0, x1 = max(0, cx - outer), min(nx, cx + outer)
    box = data[y0:y1, x0:x1].copy()
    # Mask central region
    iy0 = cy - y0 - inner
    iy1 = cy - y0 + inner
    ix0 = cx - x0 - inner
    ix1 = cx - x0 + inner
    if iy0 >= 0 and iy1 <= box.shape[0] and ix0 >= 0 and ix1 <= box.shape[1]:
        box[max(0, iy0):iy1, max(0, ix0):ix1] = np.nan
    flat = box[np.isfinite(box)].ravel()
    if len(flat) < 10:
        return _image_rms_mad(data)
    return float(1.4826 * np.median(np.abs(flat - np.median(flat))))


def _peak_in_box(data: np.ndarray, cy: int, cx: int, half: int = 1) -> float:
    """Peak value in a (2*half+1)^2 box around (cy, cx)."""
    ny, nx = data.shape
    y0, y1 = max(0, cy - half), min(ny, cy + half + 1)
    x0, x1 = max(0, cx - half), min(nx, cx + half + 1)
    sub = data[y0:y1, x0:x1]
    if sub.size == 0:
        return 0.0
    valid = sub[np.isfinite(sub)]
    if valid.size == 0:
        return 0.0
    return float(np.nanmax(valid))


def _query_nvss_in_footprint(
    nvss_db: str,
    ra_min: float,
    ra_max: float,
    dec_min: float,
    dec_max: float,
    min_flux_mjy: float = QA_MIN_FLUX_MJY,
) -> list[tuple[float, float, float]]:
    """Return (ra_deg, dec_deg, flux_mjy) for NVSS sources in footprint."""
    try:
        con = sqlite3.connect(nvss_db)
        rows = con.execute(
            "SELECT ra_deg, dec_deg, flux_mjy FROM sources "
            "WHERE ra_deg BETWEEN ? AND ? AND dec_deg BETWEEN ? AND ? "
            "AND flux_mjy >= ?",
            (ra_min, ra_max, dec_min, dec_max, min_flux_mjy),
        ).fetchall()
        con.close()
        return [(float(r[0]), float(r[1]), float(r[2])) for r in rows]
    except Exception:
        return []


def _sky_footprint(wcs: WCS, ny: int, nx: int) -> tuple[float, float, float, float]:
    """Return (ra_min, ra_max, dec_min, dec_max) for the image WCS."""
    sample_x = [0, nx - 1, 0, nx - 1, nx // 2, nx // 2, 0, nx - 1]
    sample_y = [0, 0, ny - 1, ny - 1, 0, ny - 1, ny // 2, ny // 2]
    sky = wcs.pixel_to_world(sample_x, sample_y)
    ra_vals = np.array([c.ra.deg for c in sky])
    dec_vals = np.array([c.dec.deg for c in sky])
    return float(ra_vals.min()), float(ra_vals.max()), float(dec_vals.min()), float(dec_vals.max())


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def measure_epoch_qa(
    mosaic_fits: str,
    nvss_db: str = DEFAULT_NVSS_DB,
    min_flux_mjy: float = QA_MIN_FLUX_MJY,
) -> EpochQAResult:
    """Compute the three-gate composite QA metric for one epoch mosaic.

    Parameters
    ----------
    mosaic_fits:
        Path to the epoch mosaic FITS file (Jy/beam, primary beam corrected).
    nvss_db:
        Path to the NVSS SQLite database (table ``sources``).
    min_flux_mjy:
        Minimum NVSS catalog flux in mJy for inclusion (default 50 mJy).

    Returns
    -------
    EpochQAResult
    """
    # --- Load image ---
    with fits.open(mosaic_fits) as hdul:
        hdr = hdul[0].header
        raw = hdul[0].data
    while raw.ndim > 2:
        raw = raw[0]
    data = raw.astype(np.float64)
    wcs = WCS(hdr, naxis=2)

    # --- Global RMS ---
    rms_jy = _image_rms_mad(data)
    rms_mjy = rms_jy * 1000.0

    # --- Sky footprint ---
    ny, nx = data.shape
    ra_min, ra_max, dec_min, dec_max = _sky_footprint(wcs, ny, nx)

    # --- Catalog sources ---
    catalog = _query_nvss_in_footprint(nvss_db, ra_min, ra_max, dec_min, dec_max, min_flux_mjy)
    n_catalog = len(catalog)

    # --- Measure each source ---
    ratios: list[float] = []
    n_recovered = 0

    for ra, dec, cat_flux_mjy in catalog:
        try:
            pix = wcs.world_to_pixel_values(ra, dec)
            cx, cy = int(round(float(pix[0]))), int(round(float(pix[1])))
        except Exception:
            continue
        if not (2 <= cy < ny - 2 and 2 <= cx < nx - 2):
            continue

        local = _local_rms(data, cy, cx)
        peak = _peak_in_box(data, cy, cx, half=1)
        recovered = peak > QA_RECOVERY_SIGMA * local

        if recovered:
            n_recovered += 1
            cat_flux_jy = cat_flux_mjy / 1000.0
            if cat_flux_jy > 0:
                ratios.append(peak / cat_flux_jy)

    # --- Gate evaluation ---
    median_ratio = float(np.median(ratios)) if ratios else float("nan")
    completeness_frac = n_recovered / n_catalog if n_catalog > 0 else 0.0

    # Gate 1: flux scale
    if np.isnan(median_ratio) or len(ratios) < QA_MIN_RATIO_DETECTIONS:
        ratio_gate: _GateResult = "FAIL"
    elif QA_RATIO_LOW <= median_ratio <= QA_RATIO_HIGH:
        ratio_gate = "PASS"
    else:
        ratio_gate = "FAIL"

    # Gate 2: detection completeness
    if n_catalog < QA_MIN_CATALOG_SOURCES:
        completeness_gate: _GateResult = "SKIP"
    elif completeness_frac >= QA_COMPLETENESS_MIN:
        completeness_gate = "PASS"
    else:
        completeness_gate = "FAIL"

    # Gate 3: noise floor
    # TODO: recompute QA_RMS_LIMIT_MJY when Tsys/SEFD is measured
    if rms_mjy <= QA_RMS_LIMIT_MJY:
        rms_gate: _GateResult = "PASS"
    else:
        rms_gate = "FAIL"

    # Overall verdict
    active_gates = [g for g in (ratio_gate, completeness_gate, rms_gate) if g != "SKIP"]
    qa_result: _GateResult = "PASS" if active_gates and all(g == "PASS" for g in active_gates) else "FAIL"

    return EpochQAResult(
        n_catalog=n_catalog,
        n_recovered=n_recovered,
        completeness_frac=round(completeness_frac, 4),
        median_ratio=round(median_ratio, 4) if not np.isnan(median_ratio) else float("nan"),
        mosaic_rms_mjy=round(rms_mjy, 3),
        ratio_gate=ratio_gate,
        completeness_gate=completeness_gate,
        rms_gate=rms_gate,
        qa_result=qa_result,
        ratios=ratios,
    )
