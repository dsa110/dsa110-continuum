"""Multi-frequency analysis for ESE detection.

This module provides multi-frequency analysis capabilities to detect ESEs
by correlating variability across different observing frequencies.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path


def analyze_frequency_correlation(
    source_id: str,
    frequencies: list[float],
    products_db: Path,
) -> dict[str, any]:
    """Analyze correlation of variability across frequencies.

    Parameters
    ----------
    source_id : str
        Source identifier
    frequencies : list
        List of frequencies in MHz
    products_db : str
        Path to products database

    Returns
    -------
        dict
        Dictionary with correlation analysis results
    """
    if not products_db.exists():
        return {
            "is_correlated": False,
            "strength": 0.0,
            "frequencies_analyzed": 0,
        }

    try:
        conn = sqlite3.connect(products_db)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        # Query variability stats for each frequency
        frequency_results = []
        for freq in frequencies:
            # Construct source_id with frequency suffix
            freq_source_id = f"{source_id}_{freq:.0f}"

            cursor.execute(
                """
                SELECT sigma_deviation, n_obs
                FROM variability_stats
                WHERE source_id = ?
                """,
                (freq_source_id,),
            )

            row = cursor.fetchone()
            if row:
                frequency_results.append(
                    {
                        "frequency": freq,
                        "sigma_deviation": float(row["sigma_deviation"]),
                        "n_obs": int(row["n_obs"]),
                    }
                )

        conn.close()

        if len(frequency_results) < 2:
            return {
                "is_correlated": False,
                "strength": 0.0,
                "frequencies_analyzed": len(frequency_results),
            }

        # Check if multiple frequencies show high variability
        high_variability_count = sum(1 for r in frequency_results if r["sigma_deviation"] >= 3.0)

        # Calculate correlation strength
        # Strength = proportion of frequencies with high variability
        correlation_strength = high_variability_count / len(frequency_results)

        # Consider correlated if strength > 0.5 (majority show variability)
        is_correlated = correlation_strength > 0.5

        return {
            "is_correlated": is_correlated,
            "strength": correlation_strength,
            "frequencies_analyzed": len(frequency_results),
            "high_variability_count": high_variability_count,
        }

    except Exception as e:
        return {
            "is_correlated": False,
            "strength": 0.0,
            "error": str(e),
        }


def calculate_composite_significance(
    base_significance: float,
    correlation_strength: float,
) -> float:
    """Calculate composite significance with correlation boost.

        Formula:
        composite = base_significance * (1.0 + correlation_strength * 0.3)

    Parameters
    ----------
    base_significance : float
        Base significance from flux variability
    correlation_strength : float
        Correlation strength (0.0 to 1.0)

    Returns
    -------
        float
        Composite significance value
    """
    correlation_boost = 1.0 + correlation_strength * 0.3
    return float(base_significance * correlation_boost)


def detect_ese_multi_frequency(
    source_id: str,
    frequencies: list[float],
    products_db: Path,
    min_sigma: float = 3.0,
) -> dict[str, any]:
    """Detect ESE using multi-frequency analysis.

    Parameters
    ----------
    source_id : str
        Source identifier
    frequencies : list
        List of frequencies in MHz
    products_db : str
        Path to products database
    min_sigma : float
        Minimum sigma threshold

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
        conn = sqlite3.connect(products_db)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        # Get base significance from primary frequency (first frequency)
        primary_freq = frequencies[0] if frequencies else None
        base_significance = None

        if primary_freq:
            freq_source_id = f"{source_id}_{primary_freq:.0f}"
            cursor.execute(
                """
                SELECT sigma_deviation
                FROM variability_stats
                WHERE source_id = ?
                """,
                (freq_source_id,),
            )
            row = cursor.fetchone()
            if row:
                base_significance = float(row["sigma_deviation"])

        conn.close()

        # Analyze frequency correlation
        correlation = analyze_frequency_correlation(source_id, frequencies, products_db)

        # Calculate composite significance if we have base significance
        if base_significance is not None:
            composite_significance = calculate_composite_significance(
                base_significance, correlation["strength"]
            )
        else:
            composite_significance = None

        # Determine if detected
        detected = False
        if composite_significance is not None:
            detected = composite_significance >= min_sigma
        elif base_significance is not None:
            detected = base_significance >= min_sigma

        return {
            "source_id": source_id,
            "detected": detected,
            "significance": composite_significance if composite_significance else base_significance,
            "base_significance": base_significance,
            "correlation": correlation,
        }

    except Exception as e:
        return {
            "source_id": source_id,
            "detected": False,
            "error": str(e),
        }
