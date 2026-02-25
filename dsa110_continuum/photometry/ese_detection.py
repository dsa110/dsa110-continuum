"""ESE (Extreme Scattering Event) detection from variability statistics.

This module provides functions to detect ESE candidates by analyzing
variability statistics computed from photometry measurements.
"""

from __future__ import annotations

import logging
import sqlite3
import time
from pathlib import Path

import numpy as np

from dsa110_contimg.core.photometry.scoring import (
    calculate_composite_score,
    get_confidence_level,
)
from dsa110_contimg.core.photometry.scoring import (
    calculate_composite_score,
    get_confidence_level,
)
from datetime import timezone
from dsa110_contimg.common.utils.decorators import timed

logger = logging.getLogger(__name__)


@timed("photometry.detect_ese_candidates")
def detect_ese_candidates(
    products_db: Path,
    min_sigma: float = 5.0,
    source_id: str | None = None,
    recompute: bool = False,
    use_composite_scoring: bool = False,
    scoring_weights: dict | None = None,
) -> list[dict]:
    """Detect ESE candidates from variability statistics.

        Queries the variability_stats table for sources with sigma_deviation >= min_sigma
        and flags them as ESE candidates in the ese_candidates table.

    Parameters
    ----------
    products_db : str
        Path to products database
    min_sigma : float, optional
        Minimum sigma deviation threshold (default 5.0)
    source_id : str or None, optional
        Optional specific source ID to check (if None, checks all sources)
    recompute : bool, optional
        If True, recompute variability stats before detection (default False)
    use_composite_scoring : bool, optional
        If True, compute composite score from multiple metrics (default False)
    scoring_weights : dict or None, optional
        Optional custom weights for composite scoring (default None)

    Returns
    -------
        None
    """
    if not products_db.exists():
        logger.warning(f"Products database not found: {products_db}")
        return []

    conn = sqlite3.connect(products_db, timeout=30.0)
    conn.row_factory = sqlite3.Row

    try:
        # Ensure tables exist
        tables = {
            row[0]
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        }

        if "monitoring_sources" not in tables:
            logger.warning("monitoring_sources table not found in database")
            return []

        if "ese_candidates" not in tables:
            logger.warning("ese_candidates table not found - initializing database")
            from dsa110_contimg.infrastructure.database import ensure_pipeline_db

            ensure_pipeline_db().close()  # Ensure schema is created

        # If recompute requested, update variability stats first
        if recompute:
            logger.info("Recomputing variability statistics...")
            _recompute_variability_stats(conn)

        # Query for sources with high variability
        # Include eta_metric if available for composite scoring
        if source_id:
            query = """
                SELECT
                    source_id,
                    ra_deg,
                    dec_deg,
                    nvss_flux_jy,
                    mean_flux_jy,
                    std_flux_jy,
                    chi_squared,
                    sigma_deviation,
                    eta,
                    n_detections,
                    last_detected_at
                FROM monitoring_sources
                WHERE source_id = ? AND sigma_deviation >= ?
            """
            params = (source_id, min_sigma)
        else:
            query = """
                SELECT
                    source_id,
                    ra_deg,
                    dec_deg,
                    nvss_flux_jy,
                    mean_flux_jy,
                    std_flux_jy,
                    chi_squared,
                    sigma_deviation,
                    eta,
                    n_detections,
                    last_detected_at
                FROM monitoring_sources
                WHERE sigma_deviation >= ?
                ORDER BY sigma_deviation DESC
            """
            params = (min_sigma,)

        rows = conn.execute(query, params).fetchall()

        if not rows:
            logger.info(f"No sources found with sigma_deviation >= {min_sigma}")
            return []

        detected = []
        flagged_at = time.time()

        for row in rows:
            source_id_val = row["source_id"]
            significance = float(row["sigma_deviation"])

            # Compute composite score if enabled
            composite_score = None
            confidence_level = None
            if use_composite_scoring:
                metrics = {
                    "sigma_deviation": significance,
                }

                # Add chi2_nu if available
                # Add chi2_nu (mapped from chi_squared) if available
                if row["chi_squared"] is not None:
                    metrics["chi2_nu"] = float(row["chi_squared"])

                # Add eta_metric (mapped from eta) if available
                if row.get("eta") is not None:
                    metrics["eta_metric"] = float(row["eta"])

                if metrics:
                    composite_score = calculate_composite_score(
                        metrics,
                        weights=scoring_weights,
                        normalize=True,
                    )
                    confidence_level = get_confidence_level(composite_score)

            # Check if already flagged
            existing = conn.execute(
                """
                SELECT id, status FROM ese_candidates
                WHERE source_id = ? AND status = 'active'
                """,
                (source_id_val,),
            ).fetchone()

            if existing:
                # Update existing candidate if significance increased
                if significance > min_sigma:
                    conn.execute(
                        """
                        UPDATE ese_candidates
                        SET significance = ?, flagged_at = ?, flag_type = 'auto'
                        WHERE id = ?
                        """,
                        (significance, flagged_at, existing["id"]),
                    )
                    logger.debug(
                        f"Updated ESE candidate {source_id_val} (significance: {significance:.2f})"
                    )
                else:
                    logger.debug(f"Skipping {source_id_val} - already flagged as active candidate")
                    continue
            else:
                # Insert new candidate
                conn.execute(
                    """
                    INSERT INTO ese_candidates
                    (source_id, flagged_at, flagged_by, significance, flag_type, status)
                    VALUES (?, ?, 'auto', ?, 'auto', 'active')
                    """,
                    (source_id_val, flagged_at, significance),
                )
                logger.info(
                    f"Flagged ESE candidate {source_id_val} (significance: {significance:.2f})"
                )

            candidate_dict = {
                "source_id": source_id_val,
                "ra_deg": float(row["ra_deg"]),
                "dec_deg": float(row["dec_deg"]),
                "significance": significance,
                "nvss_flux_mjy": float(row["nvss_flux_jy"] * 1000.0) if row["nvss_flux_jy"] else None,
                "mean_flux_mjy": float(row["mean_flux_jy"] * 1000.0) if row["mean_flux_jy"] else None,
                "std_flux_mjy": float(row["std_flux_jy"] * 1000.0) if row["std_flux_jy"] else None,
                "chi2_nu": float(row["chi_squared"]) if row["chi_squared"] else None,
                "n_obs": int(row["n_detections"]),
                "last_mjd": (float(row["last_detected_at"]) / 86400.0 + 40587.0) if row["last_detected_at"] else None,
            }

            # Add composite scoring fields if enabled
            if use_composite_scoring and composite_score is not None:
                candidate_dict["composite_score"] = composite_score
                candidate_dict["confidence_level"] = confidence_level

            detected.append(candidate_dict)

        conn.commit()
        logger.info(f"Detected {len(detected)} ESE candidates")

        # Hook: Update ESE candidate dashboard after detection
        if detected:
            try:
                from dsa110_contimg.core.qa.pipeline_hooks import hook_ese_detection_complete

                hook_ese_detection_complete()
            except Exception as e:
                logger.debug(f"ESE dashboard update hook failed: {e}")

        return detected

    except Exception as e:
        logger.error(f"Error detecting ESE candidates: {e}", exc_info=True)
        conn.rollback()
        raise
    finally:
        conn.close()


def _recompute_variability_stats(conn: sqlite3.Connection) -> None:
    """Recompute variability statistics from photometry measurements.

    This function queries the photometry table and computes variability
    statistics for all sources, updating the variability_stats table.

    Parameters
    ----------
    conn :
        Database connection
    conn: sqlite3.Connection :


    """
    # Check if photometry table exists
    tables = {
        row[0]
        for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    }

    # Updated internal logic to use ese_pipeline
    from dsa110_contimg.core.photometry.ese_pipeline import update_variability_stats_for_source

    if "photometry" not in tables:
        logger.warning("photometry table not found - cannot recompute stats")
        return

    # Get all unique sources from photometry
    sources = conn.execute(
        """
        SELECT DISTINCT source_id
        FROM photometry
        WHERE source_id IS NOT NULL
        """
    ).fetchall()

    logger.info(f"Recomputing variability stats for {len(sources)} sources...")

    for (source_id,) in sources:
        try:
            update_variability_stats_for_source(conn, source_id, products_db=None)
        except Exception as e:
            logger.warning(f"Failed to update stats for {source_id}: {e}")

    conn.commit()
    logger.info("Finished recomputing variability statistics")
