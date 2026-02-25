"""Astrometric self-calibration for DSA-110 continuum imaging pipeline.

This module provides functions to refine astrometric accuracy by calculating
systematic offsets from high-precision catalogs (FIRST) and applying WCS corrections.

Implements Proposal #5: Astrometric Self-Calibration
Target: <1" accuracy (from current ~2-3")
"""

import logging
import math
import os
import sqlite3
import time

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def create_astrometry_tables(
    db_path: str = os.environ.get("PIPELINE_DB", "/data/dsa110-contimg/state/db/pipeline.sqlite3"),
) -> bool:
    """Create database tables for astrometric calibration tracking.

        Tables created:
        - astrometric_solutions: WCS correction solutions per mosaic
        - astrometric_residuals: Per-source offsets for quality assessment

    Parameters
    ----------
    db_path : str
        Path to products database

    Returns
    -------
        bool
        True if successful
    """
    conn = sqlite3.connect(db_path, timeout=30.0)
    cur = conn.cursor()

    try:
        # Astrometric solutions table
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS astrometric_solutions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                mosaic_id INTEGER NOT NULL,
                reference_catalog TEXT NOT NULL,
                n_matches INTEGER NOT NULL,
                ra_offset_mas REAL NOT NULL,
                dec_offset_mas REAL NOT NULL,
                ra_offset_err_mas REAL NOT NULL,
                dec_offset_err_mas REAL NOT NULL,
                rotation_deg REAL,
                scale_factor REAL,
                rms_residual_mas REAL NOT NULL,
                applied BOOLEAN DEFAULT 0,
                computed_at REAL NOT NULL,
                applied_at REAL,
                notes TEXT,
                FOREIGN KEY (mosaic_id) REFERENCES products(id)
            )
        """
        )

        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_astrometry_mosaic
            ON astrometric_solutions(mosaic_id, computed_at DESC)
        """
        )
        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_astrometry_applied
            ON astrometric_solutions(applied, computed_at DESC)
        """
        )

        # Per-source residuals table
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS astrometric_residuals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                solution_id INTEGER NOT NULL,
                source_ra_deg REAL NOT NULL,
                source_dec_deg REAL NOT NULL,
                reference_ra_deg REAL NOT NULL,
                reference_dec_deg REAL NOT NULL,
                ra_offset_mas REAL NOT NULL,
                dec_offset_mas REAL NOT NULL,
                separation_mas REAL NOT NULL,
                source_flux_mjy REAL,
                reference_flux_mjy REAL,
                measured_at REAL NOT NULL,
                FOREIGN KEY (solution_id) REFERENCES astrometric_solutions(id)
            )
        """
        )

        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_residuals_solution
            ON astrometric_residuals(solution_id)
        """
        )

        conn.commit()
        logger.info("Created astrometric calibration tables")
        return True

    except Exception as e:
        logger.error(f"Error creating astrometric tables: {e}")
        return False
    finally:
        conn.close()


def calculate_astrometric_offsets(
    observed_sources: pd.DataFrame,
    reference_sources: pd.DataFrame,
    match_radius_arcsec: float = 5.0,
    min_matches: int = 10,
    flux_weight: bool = True,
) -> dict | None:
    """Calculate systematic astrometric offsets from reference catalog.

        Cross-matches observed sources with reference catalog (typically FIRST)
        and calculates median RA/Dec offsets.

    Parameters
    ----------
    observed_sources : pandas.DataFrame
        DataFrame with columns: ra_deg, dec_deg, flux_mjy
    reference_sources : pandas.DataFrame
        DataFrame with columns: ra_deg, dec_deg, flux_mjy
    match_radius_arcsec : float
        Matching radius [arcsec]
    min_matches : int
        Minimum number of matches required
    flux_weight : bool
        Weight offsets by source flux

    Returns
    -------
        dict or None
        Dictionary with offset solution, or None if insufficient matches
    """
    from astropy import units as u
    from astropy.coordinates import SkyCoord

    if len(observed_sources) == 0 or len(reference_sources) == 0:
        logger.warning("Insufficient sources for astrometric calibration")
        return None

    # Create SkyCoord catalog objects for efficient matching
    obs_coords = SkyCoord(
        ra=observed_sources["ra_deg"].values * u.deg,
        dec=observed_sources["dec_deg"].values * u.deg,
    )
    
    ref_coords = SkyCoord(
        ra=reference_sources["ra_deg"].values * u.deg,
        dec=reference_sources["dec_deg"].values * u.deg,
    )

    # Use astropy's efficient KD-tree based catalog matching
    idx, sep2d, _ = obs_coords.match_to_catalog_sky(ref_coords)
    
    # Filter matches by separation threshold
    match_radius_deg = match_radius_arcsec * u.arcsec
    matched_mask = sep2d < match_radius_deg
    
    n_matches = np.sum(matched_mask)
    
    if n_matches < min_matches:
        logger.warning(
            f"Insufficient matches for astrometric calibration: {n_matches} < {min_matches}"
        )
        return None

    logger.info(f"Found {n_matches} astrometric matches")

    # Get matched pairs
    obs_matched = observed_sources.iloc[matched_mask]
    ref_matched = reference_sources.iloc[idx[matched_mask]]
    
    # Calculate offsets in milliarcseconds
    # For RA, account for cos(dec) factor
    ra_diff = obs_matched["ra_deg"].values - ref_matched["ra_deg"].values
    dec_diff = obs_matched["dec_deg"].values - ref_matched["dec_deg"].values
    
    # Apply cos(dec) correction to RA offset
    cos_dec = np.cos(np.radians(obs_matched["dec_deg"].values))
    ra_offsets_mas = ra_diff * 3600.0 * 1000.0 * cos_dec
    dec_offsets_mas = dec_diff * 3600.0 * 1000.0

    # Build matches list for storage
    matches = []
    for i, (obs_idx, ref_idx) in enumerate(zip(np.where(matched_mask)[0], idx[matched_mask])):
        matches.append({
            "ra_obs": float(observed_sources.iloc[obs_idx]["ra_deg"]),
            "dec_obs": float(observed_sources.iloc[obs_idx]["dec_deg"]),
            "ra_ref": float(reference_sources.iloc[ref_idx]["ra_deg"]),
            "dec_ref": float(reference_sources.iloc[ref_idx]["dec_deg"]),
            "ra_offset_mas": float(ra_offsets_mas[i]),
            "dec_offset_mas": float(dec_offsets_mas[i]),
            "separation_mas": float(sep2d[matched_mask][i].to(u.mas).value),
            "flux_obs": float(observed_sources.iloc[obs_idx].get("flux_mjy", 1.0)),
            "flux_ref": float(reference_sources.iloc[ref_idx].get("flux_mjy", 1.0)),
        })

    # Calculate weighted median offsets
    if flux_weight:
        # Weight by flux (brighter sources more reliable)
        weights = obs_matched["flux_mjy"].values
        weights = weights / np.sum(weights)

        # Weighted median
        ra_offset = _weighted_median(ra_offsets_mas, weights)
        dec_offset = _weighted_median(dec_offsets_mas, weights)
    else:
        ra_offset = np.median(ra_offsets_mas)
        dec_offset = np.median(dec_offsets_mas)

    # Calculate uncertainties (MAD estimator)
    ra_offset_err = 1.4826 * np.median(np.abs(ra_offsets_mas - ra_offset))
    dec_offset_err = 1.4826 * np.median(np.abs(dec_offsets_mas - dec_offset))

    # Calculate RMS residual after offset correction
    ra_residuals = ra_offsets_mas - ra_offset
    dec_residuals = dec_offsets_mas - dec_offset
    rms_residual = np.sqrt(np.mean(ra_residuals**2 + dec_residuals**2))

    solution = {
        "n_matches": n_matches,
        "ra_offset_mas": float(ra_offset),
        "dec_offset_mas": float(dec_offset),
        "ra_offset_err_mas": float(ra_offset_err),
        "dec_offset_err_mas": float(dec_offset_err),
        "rms_residual_mas": float(rms_residual),
        "matches": matches,
    }

    logger.info(
        f"Astrometric solution: RA offset = {ra_offset:.1f} ± {ra_offset_err:.1f} mas, "
        f"Dec offset = {dec_offset:.1f} ± {dec_offset_err:.1f} mas, "
        f"RMS = {rms_residual:.1f} mas"
    )

    return solution


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    """Calculate weighted median.

    Parameters
    ----------
    values : array_like
        Array of values
    weights : array_like
        Array of weights (must sum to 1)

    Returns
    -------
        float
        Weighted median value
    """
    sorted_indices = np.argsort(values)
    sorted_values = values[sorted_indices]
    sorted_weights = weights[sorted_indices]

    cumulative_weights = np.cumsum(sorted_weights)
    median_idx = np.searchsorted(cumulative_weights, 0.5)

    return float(sorted_values[median_idx])


def store_astrometric_solution(
    solution: dict,
    mosaic_id: int,
    reference_catalog: str = "FIRST",
    db_path: str = os.environ.get("PIPELINE_DB", "/data/dsa110-contimg/state/db/pipeline.sqlite3"),
) -> int | None:
    """Store astrometric solution in database.

    Parameters
    ----------
    solution : dict
        Solution dictionary from calculate_astrometric_offsets()
    mosaic_id : int or str
        Associated mosaic product ID
    reference_catalog : str
        Name of reference catalog
    db_path : str
        Path to products database

    Returns
    -------
        int or None
        Solution ID, or None if failed
    """
    conn = sqlite3.connect(db_path, timeout=30.0)
    cur = conn.cursor()

    current_time = time.time()

    try:
        # Store solution
        cur.execute(
            """
            INSERT INTO astrometric_solutions (
                mosaic_id, reference_catalog, n_matches,
                ra_offset_mas, dec_offset_mas,
                ra_offset_err_mas, dec_offset_err_mas,
                rms_residual_mas, computed_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                mosaic_id,
                reference_catalog,
                solution["n_matches"],
                solution["ra_offset_mas"],
                solution["dec_offset_mas"],
                solution["ra_offset_err_mas"],
                solution["dec_offset_err_mas"],
                solution["rms_residual_mas"],
                current_time,
            ),
        )

        solution_id = cur.lastrowid

        # Store individual residuals
        for match in solution.get("matches", []):
            cur.execute(
                """
                INSERT INTO astrometric_residuals (
                    solution_id, source_ra_deg, source_dec_deg,
                    reference_ra_deg, reference_dec_deg,
                    ra_offset_mas, dec_offset_mas, separation_mas,
                    source_flux_mjy, reference_flux_mjy, measured_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    solution_id,
                    match["ra_obs"],
                    match["dec_obs"],
                    match["ra_ref"],
                    match["dec_ref"],
                    match["ra_offset_mas"],
                    match["dec_offset_mas"],
                    match["separation_mas"],
                    match["flux_obs"],
                    match["flux_ref"],
                    current_time,
                ),
            )

        conn.commit()
        logger.info(f"Stored astrometric solution {solution_id}")
        return solution_id

    except Exception as e:
        logger.error(f"Error storing astrometric solution: {e}")
        return None
    finally:
        conn.close()


def apply_wcs_correction(
    ra_offset_mas: float,
    dec_offset_mas: float,
    fits_path: str,
) -> bool:
    """Apply astrometric correction to FITS WCS headers.

        Updates CRVAL1/CRVAL2 in FITS header to correct systematic offsets.

    Parameters
    ----------
    ra_offset_mas : float
        RA offset to apply [mas]
    dec_offset_mas : float
        Dec offset to apply [mas]
    fits_path : str
        Path to FITS file to update

    Returns
    -------
        bool
        True if successful
    """
    try:
        from astropy.io import fits

        # Convert offsets to degrees
        ra_offset_deg = ra_offset_mas / (3600.0 * 1000.0)
        dec_offset_deg = dec_offset_mas / (3600.0 * 1000.0)

        # Update FITS header
        with fits.open(fits_path, mode="update") as hdul:
            header = hdul[0].header

            # Get current CRVAL
            crval1 = header.get("CRVAL1", 0.0)
            crval2 = header.get("CRVAL2", 0.0)

            # Apply correction (subtract offset, since offset = observed - reference)
            crval1_new = crval1 - ra_offset_deg / np.cos(np.radians(crval2))
            crval2_new = crval2 - dec_offset_deg

            # Update header
            header["CRVAL1"] = crval1_new
            header["CRVAL2"] = crval2_new

            # Add history
            header.add_history(
                f"Astrometric correction applied: "
                f"RA offset = {ra_offset_mas:.1f} mas, "
                f"Dec offset = {dec_offset_mas:.1f} mas"
            )

            hdul.flush()

        logger.info(f"Applied astrometric correction to {fits_path}")
        return True

    except Exception as e:
        logger.error(f"Error applying WCS correction: {e}")
        return False


def mark_solution_applied(
    solution_id: int,
    db_path: str = os.environ.get("PIPELINE_DB", "/data/dsa110-contimg/state/db/pipeline.sqlite3"),
) -> bool:
    """Mark astrometric solution as applied.

    Parameters
    ----------
    solution_id : int
        Solution ID
    db_path : str
        Path to products database

    Returns
    -------
        bool
        True if successful
    """
    conn = sqlite3.connect(db_path, timeout=30.0)
    cur = conn.cursor()

    try:
        cur.execute(
            """
            UPDATE astrometric_solutions
            SET applied = 1, applied_at = ?
            WHERE id = ?
        """,
            (time.time(), solution_id),
        )

        conn.commit()
        logger.info(f"Marked solution {solution_id} as applied")
        return True

    except Exception as e:
        logger.error(f"Error marking solution applied: {e}")
        return False
    finally:
        conn.close()


def get_astrometric_accuracy_stats(
    time_window_days: float | None = 30.0,
    db_path: str = os.environ.get("PIPELINE_DB", "/data/dsa110-contimg/state/db/pipeline.sqlite3"),
) -> dict:
    """Get astrometric accuracy statistics.

    Parameters
    ----------
    time_window_days : int or None
        Time window for statistics [days], None for all time
    db_path : str
        Path to products database

    Returns
    -------
        dict
        Dictionary with accuracy statistics
    """
    conn = sqlite3.connect(db_path)

    query = "SELECT * FROM astrometric_solutions"
    params = []

    if time_window_days:
        cutoff_time = time.time() - (time_window_days * 86400.0)
        query += " WHERE computed_at >= ?"
        params.append(cutoff_time)

    query += " ORDER BY computed_at DESC"

    try:
        df = pd.read_sql_query(query, conn, params=params)

        if len(df) == 0:
            return {
                "n_solutions": 0,
                "mean_rms_mas": None,
                "median_rms_mas": None,
                "mean_ra_offset_mas": None,
                "mean_dec_offset_mas": None,
            }

        stats = {
            "n_solutions": len(df),
            "mean_rms_mas": float(df["rms_residual_mas"].mean()),
            "median_rms_mas": float(df["rms_residual_mas"].median()),
            "mean_ra_offset_mas": float(df["ra_offset_mas"].mean()),
            "mean_dec_offset_mas": float(df["dec_offset_mas"].mean()),
            "std_ra_offset_mas": float(df["ra_offset_mas"].std()),
            "std_dec_offset_mas": float(df["dec_offset_mas"].std()),
            "mean_n_matches": float(df["n_matches"].mean()),
        }

        return stats

    finally:
        conn.close()


def get_recent_astrometric_solutions(
    limit: int = 10,
    db_path: str = os.environ.get("PIPELINE_DB", "/data/dsa110-contimg/state/db/pipeline.sqlite3"),
) -> pd.DataFrame:
    """Get recent astrometric solutions.

    Parameters
    ----------
    limit : int
        Maximum number of solutions to return
    db_path : str
        Path to products database

    Returns
    -------
        pandas.DataFrame
        DataFrame with solution information
    """
    conn = sqlite3.connect(db_path)

    query = """
        SELECT * FROM astrometric_solutions
        ORDER BY computed_at DESC
        LIMIT ?
    """

    try:
        df = pd.read_sql_query(query, conn, params=[limit])
        return df
    finally:
        conn.close()
