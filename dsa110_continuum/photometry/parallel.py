"""Parallel processing for ESE detection.

This module provides parallel processing capabilities to speed up ESE detection
for large numbers of sources using multiprocessing.
"""

from __future__ import annotations

import multiprocessing
import sqlite3
from pathlib import Path


def _detect_single_source(
    source_id: str,
    products_db: Path,
    min_sigma: float,
) -> dict:
    """Detect ESE for a single source (worker function).

    Parameters
    ----------
    source_id : str
        Source identifier
    products_db : str
        Path to products database
    min_sigma : float
        Minimum sigma threshold

    Returns
    -------
        dict
        Detection result dictionary
    """
    try:
        conn = sqlite3.connect(products_db)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        # Query variability_stats for this source
        cursor.execute(
            """
            SELECT source_id, ra_deg, dec_deg, sigma_deviation
            FROM variability_stats
            WHERE source_id = ? AND sigma_deviation >= ?
            """,
            (source_id, min_sigma),
        )

        row = cursor.fetchone()
        conn.close()

        if row is None:
            return {
                "source_id": source_id,
                "detected": False,
                "significance": None,
            }

        return {
            "source_id": row["source_id"],
            "detected": True,
            "significance": float(row["sigma_deviation"]),
            "ra_deg": float(row["ra_deg"]),
            "dec_deg": float(row["dec_deg"]),
        }

    except Exception as e:
        # Return error result
        return {
            "source_id": source_id,
            "detected": False,
            "error": str(e),
        }


def get_optimal_worker_count() -> int:
    """
    Get optimal number of worker processes.

    Returns
    -------
        Optimal worker count (typically CPU count, capped at reasonable max)
    """
    cpu_count = multiprocessing.cpu_count()
    # Cap at 32 to avoid excessive overhead
    return min(cpu_count, 32)


def detect_ese_parallel(
    source_ids: list[str],
    products_db: Path,
    min_sigma: float = 5.0,
    n_workers: int | None = None,
) -> list[dict]:
    """Detect ESE candidates for multiple sources in parallel.

    Parameters
    ----------
    source_ids : list
        List of source identifiers to check
    products_db : str
        Path to products database
    min_sigma : float
        Minimum sigma threshold for detection
    n_workers : int
        Number of worker processes (defaults to optimal)

    Returns
    -------
        list of dict
        List of detection result dictionaries

        Example
    -------
        >>> source_ids = ["source001", "source002", "source003"]
        >>> results = detect_ese_parallel(source_ids, products_db, min_sigma=3.0)
        >>> for result in results:
        ...     if result['detected']:
        ...         print(f"{result['source_id']}: {result['significance']}")
    """
    if not source_ids:
        return []

    if n_workers is None:
        n_workers = get_optimal_worker_count()

    # For small lists, use sequential processing to avoid overhead
    if len(source_ids) < n_workers * 2:
        results = []
        for source_id in source_ids:
            result = _detect_single_source(source_id, products_db, min_sigma)
            results.append(result)
        return results

    # Use multiprocessing for larger lists
    with multiprocessing.Pool(processes=n_workers) as pool:
        # Create argument tuples for each source
        args = [(source_id, products_db, min_sigma) for source_id in source_ids]

        # Process in parallel
        results = pool.starmap(_detect_single_source, args)

    return results
