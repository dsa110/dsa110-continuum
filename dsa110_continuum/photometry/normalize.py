"""
Flux normalization for forced photometry using reference source ensembles.

Implements differential photometry to achieve 1-2% relative precision by
normalizing out atmospheric and instrumental systematics.

Based on established radio variability survey methods and differential photometry
techniques from optical astronomy.

Algorithm: Differential Flux Ratios
-----------------------------------
1. Establish baseline flux for N reference sources (median of first 10 epochs)
2. For each new epoch: measure all references, compute correction factor
3. Apply correction to target sources
4. Achieves 1-2% relative precision vs 5-7% absolute

See docs/reports/ESE_LITERATURE_SUMMARY.md for full methodology.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .forced import measure_forced_peak


@dataclass
class ReferenceSource:
    """Reference source for differential photometry."""

    source_id: int
    ra_deg: float
    dec_deg: float
    nvss_name: str
    nvss_flux_mjy: float
    snr_nvss: float
    flux_baseline: float | None = None  # Will be set after baseline establishment
    baseline_rms: float | None = None
    is_valid: bool = True  # False if flagged as variable


@dataclass
class CorrectionResult:
    """Result of ensemble correction calculation."""

    correction_factor: float
    correction_rms: float  # Scatter in reference ensemble
    n_references: int
    reference_measurements: list[float]
    valid_references: list[int]  # source_ids used


def query_reference_sources(
    db_path: Path,
    ra_center: float,
    dec_center: float,
    fov_radius_deg: float = 1.5,
    min_snr: float = 50.0,
    max_sources: int = 20,
) -> list[ReferenceSource]:
    """Query reference sources from master_sources catalog within FoV.

    Parameters
    ----------
        db_path :
        Path to master_sources.sqlite3
    ra_center : float
        Field center RA in degrees
    dec_center : float
        Field center Dec in degrees
    fov_radius_deg : float, optional
        Search radius in degrees
    min_snr : float, optional
        Minimum NVSS SNR for references
    max_sources : int, optional
        Maximum number of references to return

    Returns
    -------
        list
        List of ReferenceSource objects sorted by SNR (highest first)
    """
    if not db_path.exists():
        raise FileNotFoundError(f"Catalog not found: {db_path}")

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    # Approximate box search (faster than exact angular separation)
    # For small angles: Δdec ≈ radius, Δra ≈ radius/cos(dec)
    dec_half = fov_radius_deg
    ra_half = fov_radius_deg / np.cos(np.radians(dec_center))

    # Query sources directly (final_references view requires alpha which needs VLASS)
    # If NVSS-only catalog, use sources table with SNR and compactness filters
    query = """
    SELECT source_id, ra_deg, dec_deg, s_nvss, snr_nvss
    FROM sources
    WHERE ra_deg BETWEEN ? AND ?
      AND dec_deg BETWEEN ? AND ?
      AND snr_nvss >= ?
      AND resolved_flag = 0
      AND confusion_flag = 0
    ORDER BY snr_nvss DESC
    LIMIT ?
    """

    rows = conn.execute(
        query,
        (
            ra_center - ra_half,
            ra_center + ra_half,
            dec_center - dec_half,
            dec_center + dec_half,
            min_snr,
            max_sources,
        ),
    ).fetchall()

    conn.close()

    sources = []
    for row in rows:
        # Exact angular separation check
        dra = (row["ra_deg"] - ra_center) * np.cos(np.radians(dec_center))
        ddec = row["dec_deg"] - dec_center
        sep = np.sqrt(dra**2 + ddec**2)

        if sep <= fov_radius_deg:
            # Construct NVSS name from coordinates
            ra_h = int(row["ra_deg"] / 15.0)
            ra_m = int((row["ra_deg"] / 15.0 - ra_h) * 60.0)
            ra_s = ((row["ra_deg"] / 15.0 - ra_h) * 60.0 - ra_m) * 60.0
            dec_sign = "+" if row["dec_deg"] >= 0 else "-"
            dec_d = int(abs(row["dec_deg"]))
            dec_m = int((abs(row["dec_deg"]) - dec_d) * 60.0)
            dec_s = ((abs(row["dec_deg"]) - dec_d) * 60.0 - dec_m) * 60.0

            nvss_name = f"NVSS J{ra_h:02d}{ra_m:02d}{ra_s:04.1f}{dec_sign}{dec_d:02d}{dec_m:02d}{dec_s:02.0f}"

            sources.append(
                ReferenceSource(
                    source_id=row["source_id"],
                    ra_deg=row["ra_deg"],
                    dec_deg=row["dec_deg"],
                    nvss_name=nvss_name,
                    nvss_flux_mjy=row["s_nvss"] * 1000.0,  # Jy to mJy
                    snr_nvss=row["snr_nvss"],
                )
            )

    return sources


def establish_baselines(
    sources: list[ReferenceSource],
    db_conn: sqlite3.Connection,
    n_baseline_epochs: int = 10,
) -> list[ReferenceSource]:
    """Establish baseline flux for reference sources from first N epochs.

    Parameters
    ----------
    sources : list
        List of ReferenceSource objects
        db_conn :
        Connection to products database
    n_baseline_epochs : int, optional
        Number of epochs to use for baseline

    Returns
    -------
        list
        Updated sources list with baselines set
    """
    for source in sources:
        # Query first N measurements for this source
        rows = db_conn.execute(
            """
            SELECT peak_jyb FROM photometry
            WHERE source_id = ?
            ORDER BY mjd ASC
            LIMIT ?
            """,
            (source.source_id, n_baseline_epochs),
        ).fetchall()

        if len(rows) >= 3:  # Need at least 3 points for robust median
            fluxes = np.array([r[0] for r in rows])
            source.flux_baseline = float(np.median(fluxes))
            # Robust RMS estimate using MAD (Median Absolute Deviation)
            mad = np.median(np.abs(fluxes - source.flux_baseline))
            source.baseline_rms = float(1.4826 * mad)  # Convert MAD to stddev
        else:
            # Not enough data yet, mark as invalid
            source.is_valid = False

    return [s for s in sources if s.is_valid]


def compute_ensemble_correction(
    fits_path: str,
    ref_sources: list[ReferenceSource],
    *,
    box_size_pix: int = 5,
    annulus_pix: tuple[int, int] = (30, 50),
    max_deviation_sigma: float = 3.0,
) -> CorrectionResult:
    """Compute correction factor from reference source ensemble.

        Measures all reference sources, computes ratio to baseline, and returns
        median correction factor with scatter estimate.

    Parameters
    ----------
        fits_path :
        Path to FITS image
    ref_sources : list
        List of reference sources with baselines established
    box_size_pix : int, optional
        Pixel box size for forced photometry
    annulus_pix : tuple, optional
        Annulus radii for RMS estimation
    max_deviation_sigma : float, optional
        Reject references deviating > this many sigma

    Returns
    -------
        CorrectionResult
        CorrectionResult with correction factor and statistics
    """
    ratios = []
    measurements = []
    valid_ids = []

    for ref in ref_sources:
        if not ref.is_valid or ref.flux_baseline is None:
            continue

        # Measure current flux
        result = measure_forced_peak(
            fits_path,
            ref.ra_deg,
            ref.dec_deg,
            box_size_pix=box_size_pix,
            annulus_pix=annulus_pix,
        )

        if not np.isfinite(result.peak_jyb) or result.peak_jyb <= 0:
            continue

        # Compute ratio to baseline
        ratio = result.peak_jyb / ref.flux_baseline
        ratios.append(ratio)
        measurements.append(result.peak_jyb)
        valid_ids.append(ref.source_id)

    if len(ratios) < 3:
        raise ValueError(f"Insufficient valid reference measurements: {len(ratios)} < 3")

    # Robust statistics
    ratios_arr = np.array(ratios)
    median_ratio = float(np.median(ratios_arr))
    mad = np.median(np.abs(ratios_arr - median_ratio))
    rms_ratio = float(1.4826 * mad)

    # Reject outliers (sigma clipping)
    mask = np.abs(ratios_arr - median_ratio) < (max_deviation_sigma * rms_ratio)
    if np.sum(mask) >= 3:
        # Recompute after rejection
        ratios_clean = ratios_arr[mask]
        median_ratio = float(np.median(ratios_clean))
        mad = np.median(np.abs(ratios_clean - median_ratio))
        rms_ratio = float(1.4826 * mad)
        valid_ids = [vid for vid, m in zip(valid_ids, mask) if m]
        measurements = [meas for meas, m in zip(measurements, mask) if m]

    return CorrectionResult(
        correction_factor=median_ratio,
        correction_rms=rms_ratio,
        n_references=len(valid_ids),
        reference_measurements=measurements,
        valid_references=valid_ids,
    )


def normalize_measurement(
    raw_flux: float,
    raw_error: float,
    correction: CorrectionResult,
) -> tuple[float, float]:
    """Apply ensemble correction to normalize a flux measurement.

    Parameters
    ----------
    raw_flux : float
        Measured flux in Jy/beam
    raw_error : float
        Measurement error in Jy/beam
    correction : CorrectionResult
        CorrectionResult from compute_ensemble_correction

    Returns
    -------
        tuple
        (normalized_flux, normalized_error) in Jy/beam
    """
    # Normalize flux
    flux_norm = raw_flux / correction.correction_factor

    # Propagate errors
    # σ_norm^2 = (σ_raw / corr)^2 + (F_raw * σ_corr / corr^2)^2
    err_from_meas = raw_error / correction.correction_factor
    err_from_corr = raw_flux * correction.correction_rms / (correction.correction_factor**2)
    error_norm = np.sqrt(err_from_meas**2 + err_from_corr**2)

    return float(flux_norm), float(error_norm)


def check_reference_stability(
    ref_sources: list[ReferenceSource],
    db_conn: sqlite3.Connection,
    time_window_days: float = 30.0,
    max_chi2: float = 2.0,
) -> list[ReferenceSource]:
    """Check for variability in reference sources over recent time window.

        Flags references as invalid if they show significant variability.

    Parameters
    ----------
    ref_sources : list
        List of reference sources
        db_conn :
        Connection to products database
    time_window_days : float, optional
        Time window to check (days)
    max_chi2 : float, optional
        Maximum allowed reduced chi-squared

    Returns
    -------
        list
        Updated sources list with is_valid flag set
    """
    for ref in ref_sources:
        if not ref.is_valid or ref.flux_baseline is None:
            continue

        # Query recent normalized measurements
        rows = db_conn.execute(
            """
            SELECT normalized_flux_jy, normalized_flux_err_jy, mjd
            FROM photometry
            WHERE source_id = ?
              AND normalized_flux_jy IS NOT NULL
              AND mjd > (SELECT MAX(mjd) FROM photometry) - ?
            ORDER BY mjd ASC
            """,
            (ref.source_id, time_window_days),
        ).fetchall()

        if len(rows) < 5:
            continue  # Not enough data to assess

        fluxes = np.array([r[0] for r in rows])
        errors = np.array([r[1] for r in rows])

        # Compute reduced chi-squared
        mean_flux = np.mean(fluxes)
        chi2 = np.sum(((fluxes - mean_flux) / errors) ** 2) / (len(fluxes) - 1)

        if chi2 > max_chi2:
            ref.is_valid = False
            print(
                f"WARNING: Reference {ref.nvss_name} (ID {ref.source_id}) "
                f"shows variability: χ²_ν = {chi2:.2f} > {max_chi2}"
            )

    return [s for s in ref_sources if s.is_valid]


__all__ = [
    "ReferenceSource",
    "CorrectionResult",
    "query_reference_sources",
    "establish_baselines",
    "compute_ensemble_correction",
    "normalize_measurement",
    "check_reference_stability",
]
