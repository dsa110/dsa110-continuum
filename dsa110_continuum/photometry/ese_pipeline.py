"""Automated ESE detection pipeline integration.

This module provides functions to automatically compute variability statistics
and detect ESE candidates after photometry measurements are completed.
"""

from __future__ import annotations

import logging
import sqlite3
import time
from pathlib import Path

from dsa110_contimg.infrastructure.database import ensure_pipeline_db
from dsa110_contimg.core.photometry.caching import (
    invalidate_cache,
)
from dsa110_contimg.core.photometry.ese_detection import detect_ese_candidates
from dsa110_contimg.core.photometry.metrics import (
    calculate_weighted_mean,
    calculate_eta_metric,
    calculate_v_metric,
    calculate_chi_squared,
    calculate_sigma_deviation,
)
from dsa110_contimg.common.utils.decorators import timed

logger = logging.getLogger(__name__)


def update_variability_stats_for_source(
    conn: sqlite3.Connection,
    source_id: str,
    use_cache: bool = True,
    cache_ttl: int = 3600,
    products_db: Path | None = None,
) -> bool:
    """Update variability statistics for a single source from photometry measurements.

    Parameters
    ----------
    conn : object
        Database connection
    source_id : str
        Source ID to update
    use_cache : bool, optional
        If True, check cache before recomputing (default True)
    cache_ttl : int, optional
        Cache time-to-live in seconds (default 3600)

    Returns
    -------
        bool
        True if stats were updated, False otherwise
    """
    # Check if photometry table exists
    tables = {
        row[0]
        for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    }

    if "photometry" not in tables:
        logger.debug(f"photometry table not found - skipping variability stats for {source_id}")
        return False

    # Check cache first if enabled
    if use_cache:
        # Get database path from connection (approximate)
        # Note: SQLite connections don't expose the path directly, so we'll skip cache check
        # for now and rely on database-level caching in get_cached_variability_stats
        pass

    # Get photometry history for this source
    rows = conn.execute(
        """
        SELECT
            ra_deg,
            dec_deg,
            nvss_flux_mjy,
            peak_jyb,
            peak_err_jyb,
            measured_at,
            mjd
        FROM photometry
        WHERE source_id = ?
        ORDER BY measured_at
        """,
        (source_id,),
    ).fetchall()

    if not rows:
        logger.debug(f"No photometry data found for source {source_id}")
        return False

    # Convert to DataFrame-like structure for calculations
    import pandas as pd
    import numpy as np

    df = pd.DataFrame(
        rows,
        columns=[
            "ra_deg",
            "dec_deg",
            "nvss_flux_mjy",
            "peak_jyb",
            "peak_err_jyb",
            "measured_at",
            "mjd",
        ],
    )

    # Use first row for position and NVSS flux
    ra_deg = df["ra_deg"].iloc[0]
    dec_deg = df["dec_deg"].iloc[0]
    nvss_flux_mjy = (
        df["nvss_flux_mjy"].iloc[0] if not pd.isna(df["nvss_flux_mjy"].iloc[0]) else None
    )

    # Convert to Jy for calculations/storage (keep as numpy arrays)
    flux_jy = df["peak_jyb"].values
    flux_err_jy = df["peak_err_jyb"].values if "peak_err_jyb" in df.columns else None
    
    # NVSS flux for normalization (convert mJy to Jy)
    nvss_flux_jy = nvss_flux_mjy / 1000.0 if nvss_flux_mjy is not None else None

    # Calculate statistics using standardized metrics
    # Note: We work in Jy, but output logs might prefer mJy. Database uses Jy.
    n_detections = len(df)
    
    # Basic stats
    valid_flux = flux_jy[np.isfinite(flux_jy)]
    mean_flux_jy = float(np.mean(valid_flux)) if len(valid_flux) > 0 else 0.0
    std_flux_jy = float(np.std(valid_flux)) if len(valid_flux) > 0 else 0.0
    max_flux_jy = float(np.max(valid_flux)) if len(valid_flux) > 0 else 0.0
    
    # Advanced metrics
    # Weights are 1/sigma^2
    if flux_err_jy is not None:
        # Handle zeros in error to avoid division by zero
        err = flux_err_jy.copy()
        err[err <= 0] = 1e-9 # Small value to avoid inf weights
        weights = 1.0 / (err**2)
    else:
        weights = np.ones_like(flux_jy)
        
    eta = calculate_eta_metric(flux_jy, weights)
    v_index = calculate_v_metric(flux_jy)
    
    # Chi-squared
    # For chi-squared, we want to know if it varies around the weighted mean
    weighted_mean = calculate_weighted_mean(flux_jy, weights)
    # calculate_chi_squared returns raw chi2. We might want reduced.
    # The monitoring_sources table has `chi_squared`. Let's store Reduced Chi2 there if that was the convention,
    # or Raw. `metrics.py` docs say "Chi-squared statistic".
    # Previous code stored `chi2_nu` (reduced). Let's calculate reduced.
    chi2 = calculate_chi_squared(flux_jy, weights, model_value=weighted_mean)
    chi_squared = float(chi2 / (n_detections - 1)) if n_detections > 1 else 0.0

    # Sigma Deviation (deviation from mean in units of std)
    sigma_deviation = calculate_sigma_deviation(flux_jy, mean=mean_flux_jy, std=std_flux_jy)

    # Get timestamps
    last_measured_at = float(df["measured_at"].max())

    # Insert or update monitoring_sources
    conn.execute(
        """
        INSERT INTO monitoring_sources
        (source_id, ra_deg, dec_deg, n_detections, mean_flux_jy, std_flux_jy, 
         max_flux_jy, nvss_flux_jy, chi_squared, sigma_deviation, eta, v_index, 
         last_detected_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(source_id) DO UPDATE SET
            n_detections = excluded.n_detections,
            mean_flux_jy = excluded.mean_flux_jy,
            std_flux_jy = excluded.std_flux_jy,
            max_flux_jy = excluded.max_flux_jy,
            nvss_flux_jy = excluded.nvss_flux_jy,
            chi_squared = excluded.chi_squared,
            sigma_deviation = excluded.sigma_deviation,
            eta = excluded.eta,
            v_index = excluded.v_index,
            last_detected_at = excluded.last_detected_at,
            ra_deg = excluded.ra_deg,
            dec_deg = excluded.dec_deg
        """,
        (
            source_id,
            ra_deg,
            dec_deg,
            n_detections,
            mean_flux_jy,
            std_flux_jy,
            max_flux_jy,
            nvss_flux_jy,
            chi_squared,
            sigma_deviation,
            eta,
            v_index,
            last_measured_at,
        ),
    )

    logger.debug(
        f"Updated variability stats for source {source_id}: sigma_deviation={sigma_deviation:.2f}"
    )

    # Invalidate cache for this source since we just updated stats
    # Only invalidate if products_db is provided (cache needs it for key generation)
    if products_db:
        invalidate_cache(source_id, products_db)

    return True


@timed("photometry.auto_detect_ese_after_photometry")
def auto_detect_ese_after_photometry(
    products_db: Path,
    source_ids: list[str] | None = None,
    min_sigma: float = 5.0,
    update_variability_stats: bool = True,
) -> list[dict]:
    """Automatically detect ESE candidates after photometry measurements.

        This function:
        1. Updates variability statistics for specified sources (or all sources)
        2. Detects ESE candidates based on updated statistics

    Parameters
    ----------
    products_db : str
        Path to products database
    source_ids : list of str or None, optional
        Optional list of source IDs to process (if None, processes all)
    min_sigma : float
        Minimum sigma deviation threshold for ESE detection
    update_variability_stats : bool
        If True, update variability stats before detection

    Returns
    -------
        list of dict
        List of detected ESE candidate dictionaries
    """
    if not products_db.exists():
        logger.warning(f"Products database not found: {products_db}")
        return []

    conn = ensure_pipeline_db()

    try:
        # Ensure tables exist
        tables = {
            row[0]
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        }

        if "photometry" not in tables:
            logger.debug("photometry table not found - skipping ESE detection")
            return []

        if "monitoring_sources" not in tables:
            logger.debug("monitoring_sources table not found - initializing database")
            # Re-initialize to ensure schema is created
            ensure_pipeline_db().close()

        # Update variability stats for sources
        if update_variability_stats:
            if source_ids:
                # Update specific sources
                for source_id in source_ids:
                    try:
                        update_variability_stats_for_source(
                            conn, source_id, products_db=products_db
                        )
                    except Exception as e:
                        logger.warning(f"Failed to update variability stats for {source_id}: {e}")
            else:
                # Update all sources with photometry data
                source_rows = conn.execute(
                    """
                    SELECT DISTINCT source_id
                    FROM photometry
                    WHERE source_id IS NOT NULL
                    """
                ).fetchall()

                logger.info(f"Updating variability stats for {len(source_rows)} sources...")
                for (source_id,) in source_rows:
                    try:
                        update_variability_stats_for_source(
                            conn, source_id, products_db=products_db
                        )
                    except Exception as e:
                        logger.warning(f"Failed to update variability stats for {source_id}: {e}")

            conn.commit()

        # Detect ESE candidates
        candidates = detect_ese_candidates(
            products_db=products_db,
            min_sigma=min_sigma,
            source_id=None if not source_ids else source_ids[0] if len(source_ids) == 1 else None,
            recompute=False,  # Already updated above
        )

        logger.info(f"Auto-detected {len(candidates)} ESE candidates")
        return candidates

    except Exception as e:
        logger.error(f"Error in auto ESE detection: {e}", exc_info=True)
        return []
    finally:
        conn.close()


def auto_detect_ese_for_new_measurements(
    products_db: Path,
    source_id: str,
    min_sigma: float = 5.0,
) -> dict | None:
    """Automatically detect ESE candidate for a single source after new measurement.

        This is optimized for single-source updates after a new photometry measurement.

    Parameters
    ----------
    products_db : str
        Path to products database
    source_id : str
        Source ID that was just measured
    min_sigma : float
        Minimum sigma deviation threshold

    Returns
    -------
        dict or None
        ESE candidate dict if detected, None otherwise
    """
    if not products_db.exists():
        return None

    conn = ensure_pipeline_db()

    try:
        # Update variability stats for this source
        updated = update_variability_stats_for_source(conn, source_id, products_db=products_db)
        if not updated:
            return None

        conn.commit()

        # Check if this source qualifies as ESE candidate
        row = conn.execute(
            """
            SELECT sigma_deviation
            FROM monitoring_sources
            WHERE source_id = ?
            """,
            (source_id,),
        ).fetchone()

        if not row or row[0] < min_sigma:
            return None

        # Detect ESE candidate for this source
        candidates = detect_ese_candidates(
            products_db=products_db,
            min_sigma=min_sigma,
            source_id=source_id,
            recompute=False,
        )

        return candidates[0] if candidates else None

    except Exception as e:
        logger.warning(f"Error in auto ESE detection for {source_id}: {e}")
        return None
    finally:
        conn.close()
