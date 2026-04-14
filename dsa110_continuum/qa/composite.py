"""
Composite QA metric for the DSA-110 continuum pipeline.

Three-gate pass/warn/fail decision:

  Gate 1 — Flux scale
    Compares the pipeline flux scale (derived from Huber regression against
    NVSS) to a reference value.  Fails if the multiplicative correction
    deviates from 1.0 by more than ``max_flux_scale_error`` (default 15%).

  Gate 2 — Detection completeness
    Counts what fraction of NVSS sources above a brightness threshold were
    recovered in the pipeline image.  Fails if completeness falls below
    ``min_completeness`` (default 0.70 = 70%).

  Gate 3 — Noise floor
    Measures the image RMS noise and compares it against a theoretical
    expectation.  Fails if the measured RMS exceeds
    ``max_noise_factor × theoretical_rms``.

All three gates must pass for the composite result to pass.  If any gate
warns but none fails the composite result is "warn".

Usage
-----
    from dsa110_continuum.qa.composite import CompositeQA, run_composite_qa

    result = run_composite_qa(
        flux_scale_correction=1.08,
        n_detected=85,
        n_catalog_expected=100,
        measured_rms_jyb=2.1e-4,
        theoretical_rms_jyb=1.8e-4,
    )
    print(result.status)        # "pass" | "warn" | "fail"
    print(result.summary())

Or from real image + catalog DataFrames:

    result = CompositeQA().evaluate(
        flux_scale_correction=correction_factor,
        detected_df=source_df,        # columns: ra_deg, dec_deg, flux_mjy
        catalog_df=nvss_df,           # columns: ra_deg, dec_deg, flux_mjy
        measured_rms_jyb=image_rms,
        theoretical_rms_jyb=expected_rms,
    )
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Default thresholds (can be overridden at construction time)
DEFAULT_MAX_FLUX_SCALE_ERROR: float = 0.15   # 15% deviation from unity
DEFAULT_MIN_COMPLETENESS: float = 0.70       # 70% recovery fraction
DEFAULT_MAX_NOISE_FACTOR: float = 2.0        # allow up to 2× theoretical rms
DEFAULT_WARN_FLUX_SCALE_ERROR: float = 0.08  # warn at 8%
DEFAULT_WARN_COMPLETENESS: float = 0.80      # warn below 80%
DEFAULT_WARN_NOISE_FACTOR: float = 1.5       # warn above 1.5×


class QAStatus(str, Enum):
    PASS = "pass"
    WARN = "warn"
    FAIL = "fail"
    SKIP = "skip"     # gate skipped (insufficient data)


# ---------------------------------------------------------------------------
# Per-gate results
# ---------------------------------------------------------------------------

@dataclass
class FluxScaleGateResult:
    """Result for Gate 1: flux scale accuracy."""
    correction_factor: float = 1.0
    deviation: float = 0.0        # |correction - 1|
    status: QAStatus = QAStatus.PASS
    message: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "gate": "flux_scale",
            "correction_factor": self.correction_factor,
            "deviation": self.deviation,
            "status": self.status.value,
            "message": self.message,
        }


@dataclass
class CompletenessGateResult:
    """Result for Gate 2: detection completeness."""
    n_detected: int = 0
    n_expected: int = 0
    completeness: float = 1.0
    status: QAStatus = QAStatus.PASS
    message: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "gate": "completeness",
            "n_detected": self.n_detected,
            "n_expected": self.n_expected,
            "completeness": round(self.completeness, 4),
            "status": self.status.value,
            "message": self.message,
        }


@dataclass
class NoiseFloorGateResult:
    """Result for Gate 3: noise floor."""
    measured_rms_jyb: float = 0.0
    theoretical_rms_jyb: float = 0.0
    noise_factor: float = 1.0      # measured / theoretical
    status: QAStatus = QAStatus.PASS
    message: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "gate": "noise_floor",
            "measured_rms_jyb": self.measured_rms_jyb,
            "theoretical_rms_jyb": self.theoretical_rms_jyb,
            "noise_factor": round(self.noise_factor, 3),
            "status": self.status.value,
            "message": self.message,
        }


# ---------------------------------------------------------------------------
# Composite result
# ---------------------------------------------------------------------------

@dataclass
class CompositeQAResult:
    """Aggregated result from all three QA gates."""
    status: QAStatus = QAStatus.PASS
    flux_scale: FluxScaleGateResult = field(default_factory=FluxScaleGateResult)
    completeness: CompletenessGateResult = field(default_factory=CompletenessGateResult)
    noise_floor: NoiseFloorGateResult = field(default_factory=NoiseFloorGateResult)
    epoch: str | None = None    # ISO timestamp of the epoch being evaluated
    notes: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return self.status == QAStatus.PASS

    @property
    def failed(self) -> bool:
        return self.status == QAStatus.FAIL

    def summary(self) -> str:
        """Return a human-readable one-liner."""
        parts = [
            f"flux_scale={self.flux_scale.status.value}({self.flux_scale.deviation:.1%})",
            f"completeness={self.completeness.status.value}({self.completeness.completeness:.0%})",
            f"noise={self.noise_floor.status.value}({self.noise_floor.noise_factor:.2f}×)",
        ]
        epoch_str = f" epoch={self.epoch}" if self.epoch else ""
        return f"CompositeQA[{self.status.value}]{epoch_str}: {' | '.join(parts)}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "epoch": self.epoch,
            "gates": {
                "flux_scale": self.flux_scale.to_dict(),
                "completeness": self.completeness.to_dict(),
                "noise_floor": self.noise_floor.to_dict(),
            },
            "notes": self.notes,
        }


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class CompositeQA:
    """Three-gate composite QA evaluator.

    Parameters
    ----------
    max_flux_scale_error : float
        Maximum allowed |correction - 1|.  Default 0.15 (15%).
    min_completeness : float
        Minimum detection completeness fraction.  Default 0.70.
    max_noise_factor : float
        Maximum allowed measured_rms / theoretical_rms.  Default 2.0.
    warn_flux_scale_error : float
        Warn threshold for flux scale error.  Default 0.08.
    warn_completeness : float
        Warn threshold for completeness.  Default 0.80.
    warn_noise_factor : float
        Warn threshold for noise factor.  Default 1.5.
    crossmatch_radius_arcsec : float
        Match radius for completeness cross-match.  Default 15 arcsec.
    """

    def __init__(
        self,
        max_flux_scale_error: float = DEFAULT_MAX_FLUX_SCALE_ERROR,
        min_completeness: float = DEFAULT_MIN_COMPLETENESS,
        max_noise_factor: float = DEFAULT_MAX_NOISE_FACTOR,
        warn_flux_scale_error: float = DEFAULT_WARN_FLUX_SCALE_ERROR,
        warn_completeness: float = DEFAULT_WARN_COMPLETENESS,
        warn_noise_factor: float = DEFAULT_WARN_NOISE_FACTOR,
        crossmatch_radius_arcsec: float = 15.0,
    ) -> None:
        self.max_flux_scale_error = max_flux_scale_error
        self.min_completeness = min_completeness
        self.max_noise_factor = max_noise_factor
        self.warn_flux_scale_error = warn_flux_scale_error
        self.warn_completeness = warn_completeness
        self.warn_noise_factor = warn_noise_factor
        self.crossmatch_radius_arcsec = crossmatch_radius_arcsec

    # ------------------------------------------------------------------
    # Gate evaluators
    # ------------------------------------------------------------------

    def _gate_flux_scale(self, correction_factor: float) -> FluxScaleGateResult:
        """Gate 1: flux scale accuracy."""
        dev = abs(correction_factor - 1.0)
        if dev >= self.max_flux_scale_error:
            status = QAStatus.FAIL
            msg = (
                f"Flux scale correction {correction_factor:.3f} deviates "
                f"{dev:.1%} from unity (limit {self.max_flux_scale_error:.1%})"
            )
        elif dev >= self.warn_flux_scale_error:
            status = QAStatus.WARN
            msg = (
                f"Flux scale correction {correction_factor:.3f} deviates "
                f"{dev:.1%} from unity (warn >{self.warn_flux_scale_error:.1%})"
            )
        else:
            status = QAStatus.PASS
            msg = f"Flux scale correction {correction_factor:.3f} within tolerance"
        return FluxScaleGateResult(
            correction_factor=correction_factor,
            deviation=dev,
            status=status,
            message=msg,
        )

    def _gate_completeness(
        self,
        n_detected: int,
        n_expected: int,
    ) -> CompletenessGateResult:
        """Gate 2: detection completeness from counts."""
        if n_expected <= 0:
            return CompletenessGateResult(
                n_detected=n_detected,
                n_expected=n_expected,
                completeness=1.0,
                status=QAStatus.SKIP,
                message="No expected sources (cannot compute completeness)",
            )
        comp = n_detected / n_expected
        if comp < self.min_completeness:
            status = QAStatus.FAIL
            msg = (
                f"Completeness {comp:.1%} below minimum {self.min_completeness:.1%} "
                f"({n_detected}/{n_expected} sources)"
            )
        elif comp < self.warn_completeness:
            status = QAStatus.WARN
            msg = (
                f"Completeness {comp:.1%} below warning threshold "
                f"{self.warn_completeness:.1%} ({n_detected}/{n_expected})"
            )
        else:
            status = QAStatus.PASS
            msg = f"Completeness {comp:.1%} ({n_detected}/{n_expected} sources)"
        return CompletenessGateResult(
            n_detected=n_detected,
            n_expected=n_expected,
            completeness=float(comp),
            status=status,
            message=msg,
        )

    def _gate_completeness_from_df(
        self,
        detected_df: pd.DataFrame,
        catalog_df: pd.DataFrame,
        min_catalog_flux_mjy: float = 10.0,
    ) -> CompletenessGateResult:
        """Gate 2: detection completeness via crossmatch of DataFrames."""
        if catalog_df.empty:
            return CompletenessGateResult(
                status=QAStatus.SKIP,
                message="Empty catalog DataFrame",
            )

        # Filter catalog to bright sources only
        flux_col = _detect_flux_col(catalog_df)
        if flux_col and flux_col in catalog_df.columns:
            cat_bright = catalog_df[catalog_df[flux_col] >= min_catalog_flux_mjy]
        else:
            cat_bright = catalog_df

        n_expected = len(cat_bright)
        if n_expected == 0:
            return CompletenessGateResult(
                status=QAStatus.SKIP,
                message=f"No catalog sources above {min_catalog_flux_mjy} mJy",
            )

        if detected_df.empty:
            return self._gate_completeness(0, n_expected)

        # Nearest-neighbour crossmatch
        n_matched = _count_matched(
            detected_df, cat_bright, self.crossmatch_radius_arcsec
        )
        return self._gate_completeness(n_matched, n_expected)

    def _gate_noise_floor(
        self,
        measured_rms_jyb: float,
        theoretical_rms_jyb: float,
    ) -> NoiseFloorGateResult:
        """Gate 3: noise floor vs theoretical expectation."""
        if theoretical_rms_jyb <= 0:
            return NoiseFloorGateResult(
                measured_rms_jyb=measured_rms_jyb,
                theoretical_rms_jyb=theoretical_rms_jyb,
                noise_factor=float("nan"),
                status=QAStatus.SKIP,
                message="Theoretical RMS is zero/negative; gate skipped",
            )
        factor = measured_rms_jyb / theoretical_rms_jyb
        if factor > self.max_noise_factor:
            status = QAStatus.FAIL
            msg = (
                f"Measured RMS {measured_rms_jyb*1e6:.1f} μJy/beam is "
                f"{factor:.2f}× theoretical (limit {self.max_noise_factor:.1f}×)"
            )
        elif factor > self.warn_noise_factor:
            status = QAStatus.WARN
            msg = (
                f"Measured RMS {measured_rms_jyb*1e6:.1f} μJy/beam is "
                f"{factor:.2f}× theoretical (warn >{self.warn_noise_factor:.1f}×)"
            )
        else:
            status = QAStatus.PASS
            msg = (
                f"Measured RMS {measured_rms_jyb*1e6:.1f} μJy/beam is "
                f"{factor:.2f}× theoretical"
            )
        return NoiseFloorGateResult(
            measured_rms_jyb=measured_rms_jyb,
            theoretical_rms_jyb=theoretical_rms_jyb,
            noise_factor=float(factor),
            status=status,
            message=msg,
        )

    # ------------------------------------------------------------------
    # Public entry points
    # ------------------------------------------------------------------

    def evaluate_counts(
        self,
        flux_scale_correction: float,
        n_detected: int,
        n_catalog_expected: int,
        measured_rms_jyb: float,
        theoretical_rms_jyb: float,
        *,
        epoch: str | None = None,
    ) -> CompositeQAResult:
        """Evaluate all three gates from scalar inputs.

        Parameters
        ----------
        flux_scale_correction : float
            Multiplicative correction factor from Huber regression (1.0 = perfect).
        n_detected : int
            Number of pipeline-detected sources above the brightness threshold.
        n_catalog_expected : int
            Number of NVSS/catalog sources expected in the field above the same threshold.
        measured_rms_jyb : float
            Measured image RMS noise in Jy/beam.
        theoretical_rms_jyb : float
            Theoretical thermal noise in Jy/beam.
        epoch : str or None
            ISO timestamp label for this evaluation.
        """
        g1 = self._gate_flux_scale(flux_scale_correction)
        g2 = self._gate_completeness(n_detected, n_catalog_expected)
        g3 = self._gate_noise_floor(measured_rms_jyb, theoretical_rms_jyb)
        return self._aggregate(g1, g2, g3, epoch=epoch)

    def evaluate(
        self,
        flux_scale_correction: float,
        measured_rms_jyb: float,
        theoretical_rms_jyb: float,
        *,
        detected_df: pd.DataFrame | None = None,
        catalog_df: pd.DataFrame | None = None,
        n_detected: int | None = None,
        n_catalog_expected: int | None = None,
        min_catalog_flux_mjy: float = 10.0,
        epoch: str | None = None,
    ) -> CompositeQAResult:
        """Evaluate all three gates.

        Completeness can be supplied either as scalar counts (n_detected,
        n_catalog_expected) or derived from DataFrames (detected_df, catalog_df).
        DataFrames take priority when both are provided.

        Parameters
        ----------
        flux_scale_correction : float
        measured_rms_jyb : float
        theoretical_rms_jyb : float
        detected_df : DataFrame or None
            Detected sources with columns ra_deg, dec_deg, flux_mjy.
        catalog_df : DataFrame or None
            Reference catalog with columns ra_deg, dec_deg, flux_mjy.
        n_detected : int or None
            Scalar count fallback.
        n_catalog_expected : int or None
            Scalar count fallback.
        min_catalog_flux_mjy : float
            Flux threshold for completeness calculation.
        epoch : str or None
        """
        g1 = self._gate_flux_scale(flux_scale_correction)

        if detected_df is not None and catalog_df is not None:
            g2 = self._gate_completeness_from_df(
                detected_df, catalog_df, min_catalog_flux_mjy
            )
        elif n_detected is not None and n_catalog_expected is not None:
            g2 = self._gate_completeness(n_detected, n_catalog_expected)
        else:
            g2 = CompletenessGateResult(
                status=QAStatus.SKIP,
                message="No completeness data provided",
            )

        g3 = self._gate_noise_floor(measured_rms_jyb, theoretical_rms_jyb)
        return self._aggregate(g1, g2, g3, epoch=epoch)

    # ------------------------------------------------------------------
    # Aggregation
    # ------------------------------------------------------------------

    def _aggregate(
        self,
        g1: FluxScaleGateResult,
        g2: CompletenessGateResult,
        g3: NoiseFloorGateResult,
        epoch: str | None = None,
    ) -> CompositeQAResult:
        """Combine gate results into a composite status."""
        gate_statuses = [g1.status, g2.status, g3.status]

        if QAStatus.FAIL in gate_statuses:
            composite = QAStatus.FAIL
        elif QAStatus.WARN in gate_statuses:
            composite = QAStatus.WARN
        else:
            composite = QAStatus.PASS

        result = CompositeQAResult(
            status=composite,
            flux_scale=g1,
            completeness=g2,
            noise_floor=g3,
            epoch=epoch,
        )
        logger.info(result.summary())
        return result


# ---------------------------------------------------------------------------
# Convenience function (mirrors VAST run_full_validation pattern)
# ---------------------------------------------------------------------------

def run_composite_qa(
    flux_scale_correction: float,
    n_detected: int,
    n_catalog_expected: int,
    measured_rms_jyb: float,
    theoretical_rms_jyb: float,
    *,
    epoch: str | None = None,
    max_flux_scale_error: float = DEFAULT_MAX_FLUX_SCALE_ERROR,
    min_completeness: float = DEFAULT_MIN_COMPLETENESS,
    max_noise_factor: float = DEFAULT_MAX_NOISE_FACTOR,
) -> CompositeQAResult:
    """Run the composite 3-gate QA check and return a CompositeQAResult.

    This is a convenience wrapper around ``CompositeQA.evaluate_counts()``
    that constructs a QA instance with custom thresholds.

    Parameters
    ----------
    flux_scale_correction : float
        Multiplicative flux-scale correction (1.0 = perfect).
    n_detected : int
        Detected sources above brightness threshold.
    n_catalog_expected : int
        Expected sources from reference catalog in same field + threshold.
    measured_rms_jyb : float
        Measured image RMS in Jy/beam.
    theoretical_rms_jyb : float
        Theoretical thermal noise in Jy/beam.
    epoch : str or None
        ISO timestamp of the epoch.
    max_flux_scale_error : float
    min_completeness : float
    max_noise_factor : float

    Returns
    -------
    CompositeQAResult
    """
    qa = CompositeQA(
        max_flux_scale_error=max_flux_scale_error,
        min_completeness=min_completeness,
        max_noise_factor=max_noise_factor,
    )
    return qa.evaluate_counts(
        flux_scale_correction=flux_scale_correction,
        n_detected=n_detected,
        n_catalog_expected=n_catalog_expected,
        measured_rms_jyb=measured_rms_jyb,
        theoretical_rms_jyb=theoretical_rms_jyb,
        epoch=epoch,
    )


# ---------------------------------------------------------------------------
# DSA-110 theoretical noise helper
# ---------------------------------------------------------------------------

def theoretical_rms_jyb(
    n_antennas: int = 110,
    t_int_s: float = 5.0 * 60.0,      # 5-min tile
    bandwidth_hz: float = 187.5e6,    # 1311–1499 MHz
    t_sys_k: float = 100.0,           # approximate for L-band
    eta: float = 0.9,                 # aperture efficiency
    antenna_diameter_m: float = 4.65, # DSA-110 dish diameter
) -> float:
    """Estimate the theoretical thermal noise for a DSA-110 mosaic tile.

    Uses the standard radiometer equation:

        σ = T_sys / (η_a · A_eff · sqrt(N(N-1)/2 · Δν · τ))

    Returns
    -------
    float
        Theoretical RMS noise in Jy/beam.
    """
    import math

    n_baselines = n_antennas * (n_antennas - 1) / 2
    a_eff = eta * math.pi * (antenna_diameter_m / 2.0) ** 2   # m²
    # System temperature to SEFD: SEFD = 2 k T_sys / A_eff  (k = 1380.6 Jy m² K⁻¹)
    k_boltzmann_jy = 1380.6  # Jy m² K⁻¹
    sefd = 2.0 * k_boltzmann_jy * t_sys_k / a_eff  # Jy

    # Radiometer: σ = SEFD / sqrt(2 · N_bl · Δν · τ)
    rms = sefd / math.sqrt(2.0 * n_baselines * bandwidth_hz * t_int_s)
    return float(rms)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _detect_flux_col(df: pd.DataFrame) -> str | None:
    """Identify the flux column in a DataFrame (mJy preferred)."""
    for col in ("flux_mjy", "s_nvss_mjy", "flux_mJy", "flux"):
        if col in df.columns:
            return col
    return None


def _count_matched(
    detected_df: pd.DataFrame,
    catalog_df: pd.DataFrame,
    radius_arcsec: float,
) -> int:
    """Count catalog sources with at least one match in detected_df within radius."""
    try:
        import astropy.units as u
        from astropy.coordinates import SkyCoord, match_coordinates_sky

        cat_coords = SkyCoord(
            ra=catalog_df["ra_deg"].values * u.deg,
            dec=catalog_df["dec_deg"].values * u.deg,
        )
        det_coords = SkyCoord(
            ra=detected_df["ra_deg"].values * u.deg,
            dec=detected_df["dec_deg"].values * u.deg,
        )
        _, sep2d, _ = match_coordinates_sky(cat_coords, det_coords)
        return int(np.sum(sep2d.to(u.arcsec).value <= radius_arcsec))
    except Exception as exc:
        logger.warning("Completeness crossmatch failed: %s", exc)
        return 0
