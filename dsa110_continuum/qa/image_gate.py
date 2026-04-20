"""Pre-source-finding image quality gate for DSA-110 continuum pipeline.

Checks three criteria before BANE/Aegean run:
  1. Dynamic range  (peak / MAD-RMS)
  2. Noise floor ratio  (measured vs. theoretical radiometer prediction)
  3. Pixel coverage  (fraction of finite pixels)
  4. Beam sanity  (BMAJ/BMIN in header, logged only)

Returns an ImageQAResult dataclass. The gate is non-blocking — callers
decide whether to abort on FAIL.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from astropy.io import fits

log = logging.getLogger(__name__)

# Thresholds
_DR_WARN = 100.0    # dynamic range WARN below this
_DR_FAIL = 30.0     # dynamic range FAIL below this
_RMS_WARN = 1.5     # RMS ratio WARN above this
_RMS_FAIL = 3.0     # RMS ratio FAIL above this
_COVERAGE_FAIL = 0.5  # pixel coverage FAIL below this

_Gate = Literal["PASS", "WARN", "FAIL"]


@dataclass
class ImageQAResult:
    """Image quality gate result for one mosaic."""
    dynamic_range: float
    dynamic_range_gate: _Gate
    rms_mjy: float
    theoretical_rms_mjy: float
    rms_ratio: float
    rms_ratio_gate: _Gate
    pixel_coverage_frac: float
    pixel_coverage_gate: _Gate
    overall: _Gate


def check_image_quality_for_source_finding(
    mosaic_path: str | Path,
    *,
    integration_time_s: float = 12.88,
    num_antennas: int = 96,
    bandwidth_hz: float = 188e6,
    sefd_per_element_jy: float = 5800.0,
    efficiency: float = 0.7,
) -> ImageQAResult:
    """Check image quality before running BANE + Aegean.

    Parameters
    ----------
    mosaic_path : str or Path
        Path to the mosaic FITS file.
    integration_time_s : float
        Total on-sky integration time (used for theoretical noise).
    num_antennas, bandwidth_hz, sefd_per_element_jy, efficiency :
        Radiometer equation parameters (defaults are DSA-110 values).

    Returns
    -------
    ImageQAResult
        Dataclass with individual gate results and an overall verdict.
    """
    mosaic_path = Path(mosaic_path)

    # -- Load image -----------------------------------------------------------
    with fits.open(mosaic_path) as hdul:
        hdr = hdul[0].header
        data = hdul[0].data.squeeze().astype(np.float64)

    # -- Pixel coverage -------------------------------------------------------
    finite_mask = np.isfinite(data)
    coverage_frac = float(finite_mask.sum()) / float(data.size)
    coverage_gate: _Gate = "PASS" if coverage_frac >= _COVERAGE_FAIL else "FAIL"

    # -- Dynamic range --------------------------------------------------------
    finite = data[finite_mask]
    if finite.size == 0:
        rms_jy = float("nan")
        peak_jy = float("nan")
        dr = float("nan")
    else:
        rms_jy = float(1.4826 * np.median(np.abs(finite - np.median(finite))))
        peak_jy = float(np.max(np.abs(finite)))
        dr = peak_jy / rms_jy if rms_jy > 0 else float("nan")

    if np.isnan(dr) or dr < _DR_FAIL:
        dr_gate: _Gate = "FAIL"
    elif dr < _DR_WARN:
        dr_gate = "WARN"
    else:
        dr_gate = "PASS"

    log.info("Image gate: dynamic_range=%.1f [%s]  coverage=%.1f%% [%s]",
             dr, dr_gate, coverage_frac * 100, coverage_gate)

    # -- Noise floor ratio ----------------------------------------------------
    try:
        from dsa110_continuum.qa.noise_model import calculate_theoretical_rms
        theoretical_mjy = calculate_theoretical_rms(
            ms_path=None,
            bandwidth_hz=bandwidth_hz,
            integration_time_s=integration_time_s,
            num_antennas=num_antennas,
            sefd_per_element_jy=sefd_per_element_jy,
            efficiency=efficiency,
        )
        measured_mjy = rms_jy * 1000.0
        rms_ratio = measured_mjy / theoretical_mjy if theoretical_mjy > 0 else float("nan")
    except Exception as exc:
        log.warning("Could not compute theoretical RMS: %s", exc)
        theoretical_mjy = float("nan")
        measured_mjy = rms_jy * 1000.0
        rms_ratio = float("nan")

    if np.isnan(rms_ratio) or rms_ratio > _RMS_FAIL:
        rms_gate: _Gate = "FAIL"
    elif rms_ratio > _RMS_WARN:
        rms_gate = "WARN"
    else:
        rms_gate = "PASS"

    log.info("Image gate: rms=%.3f mJy  theoretical=%.3f mJy  ratio=%.2f [%s]",
             measured_mjy, theoretical_mjy, rms_ratio, rms_gate)

    # -- Beam sanity (log only, no gate) --------------------------------------
    bmaj = hdr.get("BMAJ")
    bmin = hdr.get("BMIN")
    cdelt = abs(hdr.get("CDELT2", hdr.get("CDELT1", 20.0 / 3600.0)))
    if bmaj is not None and bmin is not None:
        beam_pix = bmaj / cdelt
        log.info("Image gate: beam BMAJ=%.1f\" BMIN=%.1f\" (%.1f pix across major axis)",
                 bmaj * 3600, bmin * 3600, beam_pix)
    else:
        log.info("Image gate: no BMAJ/BMIN in header (using DSA-110 defaults 36.9\"×25.5\")")

    # -- Overall verdict ------------------------------------------------------
    gates = [dr_gate, coverage_gate, rms_gate]
    if "FAIL" in gates:
        overall: _Gate = "FAIL"
    elif "WARN" in gates:
        overall = "WARN"
    else:
        overall = "PASS"

    log.info("Image gate overall: %s", overall)

    return ImageQAResult(
        dynamic_range=round(dr, 2) if not np.isnan(dr) else float("nan"),
        dynamic_range_gate=dr_gate,
        rms_mjy=round(measured_mjy, 3),
        theoretical_rms_mjy=round(theoretical_mjy, 3) if not np.isnan(theoretical_mjy) else float("nan"),
        rms_ratio=round(rms_ratio, 3) if not np.isnan(rms_ratio) else float("nan"),
        rms_ratio_gate=rms_gate,
        pixel_coverage_frac=round(coverage_frac, 4),
        pixel_coverage_gate=coverage_gate,
        overall=overall,
    )
