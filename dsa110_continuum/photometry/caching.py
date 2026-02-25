"""Caching system for variability statistics.

This module provides caching functionality to improve performance when
accessing variability statistics for frequently queried sources.
"""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path


class CacheStats:
    """Cache statistics tracker."""

    _hits = 0
    _misses = 0

    @classmethod
    def get_stats(cls) -> dict[str, int]:
        """Get cache statistics."""
        return {
            "hits": cls._hits,
            "misses": cls._misses,
            "total": cls._hits + cls._misses,
        }

    @classmethod
    def reset(cls) -> None:
        """Reset cache statistics."""
        cls._hits = 0
        cls._misses = 0

    @classmethod
    def record_hit(cls) -> None:
        """Record a cache hit."""
        cls._hits += 1

    @classmethod
    def record_miss(cls) -> None:
        """Record a cache miss."""
        cls._misses += 1


def get_cached_variability_stats(
    source_id: str,
    products_db: Path,
    ttl_seconds: float = 3600.0,  # 1 hour default TTL
) -> dict[str, float] | None:
    """Get cached variability statistics for a source.

    Parameters
    ----------
    source_id : str
        Source identifier
    products_db : Path
        Path to products database
    ttl_seconds : float, optional
        Time-to-live in seconds (default: 3600 = 1 hour)
        (Default value = 3600.0)

    """
    if not products_db.exists():
        CacheStats.record_miss()
        return None

    try:
        conn = sqlite3.connect(products_db)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        # Query variability_stats table
        cursor.execute(
            """
            SELECT source_id, ra_deg, dec_deg, n_obs, mean_flux_mjy,
                   std_flux_mjy, sigma_deviation, updated_at
            FROM variability_stats
            WHERE source_id = ?
            """,
            (source_id,),
        )

        row = cursor.fetchone()
        conn.close()

        if row is None:
            CacheStats.record_miss()
            return None

        # Check TTL
        updated_at = float(row["updated_at"])
        age_seconds = time.time() - updated_at

        if age_seconds > ttl_seconds:
            CacheStats.record_miss()
            return None

        # Return cached stats
        CacheStats.record_hit()
        return {
            "source_id": row["source_id"],
            "ra_deg": float(row["ra_deg"]),
            "dec_deg": float(row["dec_deg"]),
            "n_obs": int(row["n_obs"]),
            "mean_flux_mjy": float(row["mean_flux_mjy"]) if row["mean_flux_mjy"] else None,
            "std_flux_mjy": float(row["std_flux_mjy"]) if row["std_flux_mjy"] else None,
            "sigma_deviation": float(row["sigma_deviation"]) if row["sigma_deviation"] else None,
            "updated_at": updated_at,
            "stale": False,
        }

    except (sqlite3.Error, ValueError):
        # On error, treat as cache miss
        CacheStats.record_miss()
        return None


def invalidate_cache(
    source_id: str,
    products_db: Path,
) -> None:
    """Invalidate cache for a source by updating its timestamp.

    This marks the cache entry as stale, forcing recomputation on next access.

    Parameters
    ----------
    source_id :
        Source identifier
    products_db :
        Path to products database
    """
    if not products_db.exists():
        return

    try:
        conn = sqlite3.connect(products_db)
        cursor = conn.cursor()

        # Update timestamp to very old value to force expiration
        cursor.execute(
            """
            UPDATE variability_stats
            SET updated_at = 0
            WHERE source_id = ?
            """,
            (source_id,),
        )

        conn.commit()
        conn.close()

    except sqlite3.Error:
        # Ignore errors on invalidation
        pass
