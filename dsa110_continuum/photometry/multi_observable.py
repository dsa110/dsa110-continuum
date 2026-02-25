"""Multi-observable correlation analysis for ESE detection.

This module provides multi-observable analysis capabilities to detect ESEs
by correlating variability across different observables (flux, scintillation, DM).
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np


def analyze_scintillation_variability(
    source_id: str,
    products_db: Path,
) -> dict[str, any]:
    """Analyze scintillation variability for a source.

    Parameters
    ----------
    source_id : str
        Source identifier
    products_db : str
        Path to products database

    Returns
    -------
        dict
        Dictionary with variability analysis results
    """
    if not products_db.exists():
        return {
            "variability": "unknown",
            "std": None,
            "mean": None,
        }

    try:
        conn = sqlite3.connect(products_db)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        # Query scintillation data
        cursor.execute(
            """
            SELECT scintillation_bandwidth_mhz, scintillation_timescale_sec
            FROM scintillation_data
            WHERE source_id = ?
            ORDER BY measured_at
            """,
            (source_id,),
        )

        rows = cursor.fetchall()
        conn.close()

        if len(rows) < 2:
            return {
                "variability": "low",
                "std": 0.0,
                "mean": float(rows[0]["scintillation_bandwidth_mhz"]) if rows else None,
            }

        # Extract bandwidth values
        bandwidths = [float(row["scintillation_bandwidth_mhz"]) for row in rows]

        # Calculate variability metrics
        mean_bw = np.mean(bandwidths)
        std_bw = np.std(bandwidths, ddof=1)

        # Classify variability
        if std_bw > mean_bw * 0.3:  # >30% relative std
            variability = "high"
        elif std_bw > mean_bw * 0.1:  # >10% relative std
            variability = "moderate"
        else:
            variability = "low"

        return {
            "variability": variability,
            "std": float(std_bw),
            "mean": float(mean_bw),
            "sigma_deviation": float(std_bw / mean_bw) if mean_bw > 0 else 0.0,
        }

    except Exception as e:
        return {
            "variability": "unknown",
            "error": str(e),
        }


def analyze_dm_variability(
    source_id: str,
    products_db: Path,
) -> dict[str, any]:
    """Analyze dispersion measure (DM) variability for a source.

    Parameters
    ----------
    source_id : str
        Source identifier
    products_db : str
        Path to products database

    Returns
    -------
        dict
        Dictionary with variability analysis results
    """
    if not products_db.exists():
        return {
            "variability": "unknown",
            "std": None,
            "mean": None,
        }

    try:
        conn = sqlite3.connect(products_db)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        # Query DM data
        cursor.execute(
            """
            SELECT dm_pc_cm3
            FROM dm_data
            WHERE source_id = ?
            ORDER BY measured_at
            """,
            (source_id,),
        )

        rows = cursor.fetchall()
        conn.close()

        if len(rows) < 2:
            return {
                "variability": "low",
                "std": 0.0,
                "mean": float(rows[0]["dm_pc_cm3"]) if rows else None,
            }

        # Extract DM values
        dms = [float(row["dm_pc_cm3"]) for row in rows]

        # Calculate variability metrics
        mean_dm = np.mean(dms)
        std_dm = np.std(dms, ddof=1)

        # Classify variability
        if std_dm > mean_dm * 0.1:  # >10% relative std (DM is typically more stable)
            variability = "high"
        elif std_dm > mean_dm * 0.05:  # >5% relative std
            variability = "moderate"
        else:
            variability = "low"

        return {
            "variability": variability,
            "std": float(std_dm),
            "mean": float(mean_dm),
            "sigma_deviation": float(std_dm / mean_dm) if mean_dm > 0 else 0.0,
        }

    except Exception as e:
        return {
            "variability": "unknown",
            "error": str(e),
        }


def calculate_observable_correlation(
    observable_results: dict[str, dict[str, any]],
) -> dict[str, any]:
    """Calculate correlation between multiple observables.

    Parameters
    ----------
    observable_results : dict
        Dictionary mapping observable names to their analysis results

    Returns
    -------
        dict
        Dictionary with correlation analysis
    """
    if not observable_results:
        return {
            "is_correlated": False,
            "strength": 0.0,
        }

    # Count observables with high variability
    high_variability_count = 0
    total_observables = 0

    for obs_name, obs_result in observable_results.items():
        if isinstance(obs_result, dict):
            variability = obs_result.get("variability", "low")
            sigma_dev = obs_result.get("sigma_deviation", 0.0)

            # Consider high variability if variability='high' or sigma_dev >= 3.0
            if variability == "high" or sigma_dev >= 3.0:
                high_variability_count += 1
            total_observables += 1

    if total_observables == 0:
        return {
            "is_correlated": False,
            "strength": 0.0,
        }

    # Calculate correlation strength
    correlation_strength = high_variability_count / total_observables

    # Consider correlated if strength > 0.5 (majority show high variability)
    is_correlated = correlation_strength > 0.5

    return {
        "is_correlated": is_correlated,
        "strength": correlation_strength,
        "high_variability_count": high_variability_count,
        "total_observables": total_observables,
    }


def detect_ese_multi_observable(
    source_id: str,
    observables: dict[str, bool],
    products_db: Path,
) -> dict[str, any]:
    """Detect ESE using multi-observable analysis.

    Parameters
    ----------
    source_id : str
        Source identifier
    observables : dict
        Dictionary mapping observable names to enabled flags
    products_db : str
        Path to products database

    Returns
    -------
        dict
        Dictionary with detection results including correlation analysis
    """
    if not products_db.exists():
        return {
            "source_id": source_id,
            "detected": False,
            "significance": None,
            "correlation": None,
        }

    try:
        # Analyze each enabled observable
        observable_results = {}

        # Analyze flux variability (from variability_stats table)
        if observables.get("flux", False):
            conn = sqlite3.connect(products_db)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            cursor.execute(
                """
                SELECT sigma_deviation
                FROM variability_stats
                WHERE source_id = ?
                """,
                (source_id,),
            )

            row = cursor.fetchone()
            conn.close()

            if row and row["sigma_deviation"] is not None:
                sigma_dev = float(row["sigma_deviation"])
                observable_results["flux"] = {
                    "variability": "high" if sigma_dev >= 3.0 else "low",
                    "sigma_deviation": sigma_dev,
                }

        # Analyze scintillation variability
        if observables.get("scintillation", False):
            scint_result = analyze_scintillation_variability(source_id, products_db)
            observable_results["scintillation"] = scint_result

        # Analyze DM variability
        if observables.get("dm", False):
            dm_result = analyze_dm_variability(source_id, products_db)
            observable_results["dm"] = dm_result

        # Calculate correlation
        correlation = calculate_observable_correlation(observable_results)

        # Get base significance from flux (if available)
        base_significance = None
        if "flux" in observable_results:
            base_significance = observable_results["flux"].get("sigma_deviation")

        # Calculate composite significance
        if base_significance is not None:
            composite_significance = base_significance * (1.0 + correlation["strength"] * 0.3)
        else:
            # If no flux, use highest sigma_deviation from other observables
            max_sigma = max(
                (obs.get("sigma_deviation", 0.0) for obs in observable_results.values()),
                default=0.0,
            )
            composite_significance = (
                max_sigma * (1.0 + correlation["strength"] * 0.3) if max_sigma > 0 else None
            )

        # Determine if detected (threshold: 3.0 sigma)
        detected = False
        if composite_significance is not None:
            detected = composite_significance >= 3.0

        return {
            "source_id": source_id,
            "detected": detected,
            "significance": composite_significance,
            "base_significance": base_significance,
            "correlation": correlation,
            "observable_results": observable_results,
        }

    except Exception as e:
        return {
            "source_id": source_id,
            "detected": False,
            "error": str(e),
        }
