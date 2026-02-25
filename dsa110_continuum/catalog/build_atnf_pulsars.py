#!/usr/bin/env python
"""
Build ATNF Pulsar Catalogue SQLite database.

Downloads the latest ATNF Pulsar Catalogue and creates a SQLite database
optimized for spatial queries and pulsar property lookups.

Usage:
    python -m dsa110_contimg.core.catalog.build_atnf_pulsars
    python -m dsa110_contimg.core.catalog.build_atnf_pulsars --output /path/to/atnf_pulsars.sqlite3
    python -m dsa110_contimg.core.catalog.build_atnf_pulsars --min-flux-mjy 1.0
"""

from __future__ import annotations

import argparse
import logging
import os
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def _download_atnf_catalog() -> pd.DataFrame:
    """Download ATNF Pulsar Catalogue using psrqpy.

    Returns
    -------
        DataFrame
        DataFrame with pulsar properties

    Raises
    ------
        ImportError
        If psrqpy is not installed
        Exception
        If download fails
    """
    try:
        import psrqpy
    except ImportError:
        raise ImportError(
            "psrqpy is required to download ATNF Pulsar Catalogue. "
            "Install it with: pip install psrqpy"
        )

    logger.info("Downloading ATNF Pulsar Catalogue...")
    print("Downloading ATNF Pulsar Catalogue (this may take a minute)...")

    # Query all pulsars with essential parameters
    params = [
        "JNAME",  # Pulsar J2000 name
        "NAME",  # Pulsar B1950 name (if available)
        "RAJ",  # Right ascension (J2000, hh:mm:ss.s)
        "DECJ",  # Declination (J2000, dd:mm:ss)
        "RAJD",  # RA in degrees
        "DECJD",  # Dec in degrees
        "P0",  # Period (s)
        "P1",  # Period derivative
        "DM",  # Dispersion measure (pc/cm^3)
        "S400",  # Flux at 400 MHz (mJy)
        "S1400",  # Flux at 1400 MHz (mJy)
        "S2000",  # Flux at 2000 MHz (mJy)
        "DIST",  # Distance (kpc)
        "TYPE",  # Pulsar type
        "BINARY",  # Binary companion type
        "ASSOC",  # Associations (SNR, GC, etc.)
    ]

    try:
        query = psrqpy.QueryATNF(params=params, loadfromdb=None)
        df = query.table.to_pandas()

        logger.info(f"Downloaded {len(df)} pulsars from ATNF catalogue")
        print(f":check: Downloaded {len(df)} pulsars")

        return df

    except Exception as e:
        logger.error(f"Failed to download ATNF catalogue: {e}")
        raise


def _process_atnf_data(df: pd.DataFrame, min_flux_mjy: float | None = None) -> pd.DataFrame:
    """Process ATNF data for database insertion.

    Parameters
    ----------
    df : DataFrame
        Raw ATNF DataFrame
    min_flux_mjy : Optional[float], optional
        Minimum flux at 1400 MHz (mJy), None = no filter. Default is None.

    Returns
    -------
        DataFrame
        Processed DataFrame with cleaned columns
    """
    logger.info("Processing ATNF data...")

    # Create processed dataframe
    processed = pd.DataFrame()

    # Name (prefer JNAME, fallback to NAME)
    processed["pulsar_name"] = df["JNAME"].fillna(df.get("NAME", ""))

    # Coordinates in degrees
    processed["ra_deg"] = pd.to_numeric(df["RAJD"], errors="coerce")
    processed["dec_deg"] = pd.to_numeric(df["DECJD"], errors="coerce")

    # Period and period derivative
    processed["period_s"] = pd.to_numeric(df["P0"], errors="coerce")
    processed["period_dot"] = pd.to_numeric(df["P1"], errors="coerce")

    # Dispersion measure
    processed["dm_pc_cm3"] = pd.to_numeric(df["DM"], errors="coerce")

    # Flux densities at different frequencies
    processed["flux_400mhz_mjy"] = pd.to_numeric(df.get("S400"), errors="coerce")
    processed["flux_1400mhz_mjy"] = pd.to_numeric(df.get("S1400"), errors="coerce")
    processed["flux_2000mhz_mjy"] = pd.to_numeric(df.get("S2000"), errors="coerce")

    # Distance
    processed["distance_kpc"] = pd.to_numeric(df.get("DIST"), errors="coerce")

    # Type and binary
    processed["pulsar_type"] = df.get("TYPE", "").fillna("")
    processed["binary_type"] = df.get("BINARY", "").fillna("")
    processed["association"] = df.get("ASSOC", "").fillna("")

    # Filter invalid coordinates
    valid_coords = processed["ra_deg"].notna() & processed["dec_deg"].notna()
    processed = processed[valid_coords].copy()

    # Filter by flux if requested
    if min_flux_mjy is not None:
        # Use 1400 MHz flux for filtering (closest to DSA-110 observing frequency)
        has_flux = processed["flux_1400mhz_mjy"].notna()
        bright_enough = processed["flux_1400mhz_mjy"] >= min_flux_mjy
        processed = processed[has_flux & bright_enough].copy()
        logger.info(f"Filtered to {len(processed)} pulsars with S1400 >= {min_flux_mjy} mJy")

    logger.info(f"Processed {len(processed)} pulsars with valid coordinates")
    return processed


def build_atnf_pulsar_db(
    output_path: str | os.PathLike[str] | None = None,
    min_flux_mjy: float | None = None,
    force_rebuild: bool = False,
) -> Path:
    """Build SQLite database for ATNF Pulsar Catalogue.

    Parameters
    ----------
    output_path : Optional[str | os.PathLike[str]], optional
        Output SQLite database path (auto-generated if None). Default is None.
    min_flux_mjy : Optional[float], optional
        Minimum flux at 1400 MHz in mJy (None = all pulsars). Default is None.
    force_rebuild : bool, optional
        Force rebuild even if database exists. Default is False.

    Returns
    -------
        str
        Path to created SQLite database

    Raises
    ------
        ImportError
        If psrqpy is not installed
        Exception
        If download or database creation fails
    """
    from dsa110_contimg.core.catalog.builders import _acquire_db_lock, _release_db_lock

    # Resolve output path
    if output_path is None:
        output_path = Path("state/catalogs/atnf_pulsars.sqlite3")
    else:
        output_path = Path(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Check if database already exists
    if output_path.exists() and not force_rebuild:
        logger.info(f"ATNF Pulsar database already exists: {output_path}")
        print(f"Database already exists: {output_path}")
        print("Use --force to rebuild")
        return output_path

    # Acquire lock to prevent concurrent builds
    lock_path = output_path.with_suffix(".lock")
    lock_fd = _acquire_db_lock(lock_path)

    if lock_fd is None:
        raise RuntimeError(f"Could not acquire lock for building {output_path}")

    try:
        # Double-check database doesn't exist (another process may have created it)
        if output_path.exists() and not force_rebuild:
            logger.info(
                f"Database {output_path} was created by another process while waiting for lock"
            )
            return output_path

        # Download and process ATNF data
        df_raw = _download_atnf_catalog()
        df_processed = _process_atnf_data(df_raw, min_flux_mjy=min_flux_mjy)

        # Create SQLite database
        print(f"Creating SQLite database: {output_path}")
        logger.info(f"Creating database: {output_path}")

        # Enable WAL mode for concurrent reads
        conn = sqlite3.connect(str(output_path))
        conn.execute("PRAGMA journal_mode=WAL")
        conn.close()

        with sqlite3.connect(str(output_path)) as conn:
            # Create pulsars table with comprehensive schema
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS pulsars (
                    pulsar_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    pulsar_name TEXT NOT NULL UNIQUE,
                    ra_deg REAL NOT NULL,
                    dec_deg REAL NOT NULL,
                    period_s REAL,
                    period_dot REAL,
                    dm_pc_cm3 REAL,
                    flux_400mhz_mjy REAL,
                    flux_1400mhz_mjy REAL,
                    flux_2000mhz_mjy REAL,
                    distance_kpc REAL,
                    pulsar_type TEXT,
                    binary_type TEXT,
                    association TEXT
                )
                """
            )

            # Create spatial indices for fast cone searches
            conn.execute("CREATE INDEX IF NOT EXISTS idx_pulsars_radec ON pulsars(ra_deg, dec_deg)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_pulsars_dec ON pulsars(dec_deg)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_pulsars_ra ON pulsars(ra_deg)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_pulsars_name ON pulsars(pulsar_name)")
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_pulsars_flux1400 ON pulsars(flux_1400mhz_mjy)"
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_pulsars_period ON pulsars(period_s)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_pulsars_dm ON pulsars(dm_pc_cm3)")

            # Clear existing data
            conn.execute("DELETE FROM pulsars")

            # Insert pulsars
            insert_data = []
            for _, row in df_processed.iterrows():
                values = (
                    row["pulsar_name"],
                    float(row["ra_deg"]),
                    float(row["dec_deg"]),
                    float(row["period_s"]) if pd.notna(row["period_s"]) else None,
                    float(row["period_dot"]) if pd.notna(row["period_dot"]) else None,
                    float(row["dm_pc_cm3"]) if pd.notna(row["dm_pc_cm3"]) else None,
                    float(row["flux_400mhz_mjy"]) if pd.notna(row["flux_400mhz_mjy"]) else None,
                    float(row["flux_1400mhz_mjy"]) if pd.notna(row["flux_1400mhz_mjy"]) else None,
                    float(row["flux_2000mhz_mjy"]) if pd.notna(row["flux_2000mhz_mjy"]) else None,
                    float(row["distance_kpc"]) if pd.notna(row["distance_kpc"]) else None,
                    str(row["pulsar_type"]),
                    str(row["binary_type"]),
                    str(row["association"]),
                )
                insert_data.append(values)

            conn.executemany(
                """
                INSERT INTO pulsars(
                    pulsar_name, ra_deg, dec_deg,
                    period_s, period_dot, dm_pc_cm3,
                    flux_400mhz_mjy, flux_1400mhz_mjy, flux_2000mhz_mjy,
                    distance_kpc, pulsar_type, binary_type, association
                ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                insert_data,
            )

            # Create metadata table
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT
                )
                """
            )

            # Store metadata
            metadata = {
                "catalog_name": "ATNF Pulsar Catalogue",
                "build_date": datetime.now(UTC).isoformat(),
                "source_url": "https://www.atnf.csiro.au/research/pulsar/psrcat/",
                "n_pulsars": str(len(df_processed)),
                "min_flux_mjy": str(min_flux_mjy) if min_flux_mjy is not None else "None",
                "builder_version": "1.0.0",
            }

            for key, value in metadata.items():
                conn.execute(
                    "INSERT OR REPLACE INTO metadata(key, value) VALUES(?, ?)", (key, value)
                )

            conn.commit()

        # Verify database was created successfully
        with sqlite3.connect(str(output_path)) as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM pulsars")
            count = cursor.fetchone()[0]
            print("\n:check: Successfully created ATNF Pulsar database")
            print(f"  Database: {output_path}")
            print(f"  Pulsars: {count}")

            if min_flux_mjy is not None:
                print(f"  Min flux (1400 MHz): {min_flux_mjy} mJy")

        logger.info(f"Successfully built ATNF Pulsar database with {count} pulsars")
        return output_path

    finally:
        # Always release the lock
        _release_db_lock(lock_fd, lock_path)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for building ATNF Pulsar Catalogue database."""
    ap = argparse.ArgumentParser(
        description="Build ATNF Pulsar Catalogue SQLite database",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Build default database (all pulsars)
  python -m dsa110_contimg.core.catalog.build_atnf_pulsars

  # Build with flux threshold
  python -m dsa110_contimg.core.catalog.build_atnf_pulsars --min-flux-mjy 1.0

  # Specify output location
  python -m dsa110_contimg.core.catalog.build_atnf_pulsars --output /custom/path.sqlite3

  # Force rebuild existing database
  python -m dsa110_contimg.core.catalog.build_atnf_pulsars --force
        """,
    )
    ap.add_argument(
        "--output",
        help="Output SQLite database path (default: state/catalogs/atnf_pulsars.sqlite3)",
    )
    ap.add_argument(
        "--min-flux-mjy",
        type=float,
        help="Minimum flux at 1400 MHz in mJy (filters out faint pulsars)",
    )
    ap.add_argument(
        "--force",
        action="store_true",
        help="Force rebuild even if database already exists",
    )
    ap.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    args = ap.parse_args(argv)

    # Configure logging
    if args.verbose:
        logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    try:
        build_atnf_pulsar_db(
            output_path=args.output,
            min_flux_mjy=args.min_flux_mjy,
            force_rebuild=args.force,
        )
        return 0

    except ImportError as e:
        print(f"\n:cross: Error: {e}")
        print("\nTo install psrqpy:")
        print("  pip install psrqpy")
        return 1

    except Exception as e:
        print(f"\n:cross: Error building database: {e}")
        import traceback

        if args.verbose:
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    import sys

    sys.exit(main())
