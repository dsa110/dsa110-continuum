#!/usr/bin/env python
"""
Schema migration helper for master_sources.sqlite3.

Adds provenance tables without altering existing sources data. Optionally
rebuilds the master catalog with provenance enabled.
"""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

from dsa110_contimg.core.catalog.build_master import (
    backfill_master_provenance_from_sqlite,
    build_master_union_from_sqlite,
)


def _ensure_provenance_tables(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS catalog_matches (
            source_id INTEGER NOT NULL,
            catalog TEXT NOT NULL,
            catalog_row_id INTEGER NOT NULL,
            sep_arcsec REAL,
            match_rank INTEGER DEFAULT 0,
            is_primary INTEGER DEFAULT 0,
            match_version INTEGER DEFAULT 1,
            PRIMARY KEY (source_id, catalog, catalog_row_id)
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_catalog_matches_source ON catalog_matches(source_id)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_catalog_matches_catalog_row ON catalog_matches(catalog, catalog_row_id)"
    )
    cols = {row[1] for row in conn.execute("PRAGMA table_info(catalog_matches)").fetchall()}
    if "match_version" not in cols:
        conn.execute(
            "ALTER TABLE catalog_matches ADD COLUMN match_version INTEGER DEFAULT 1"
        )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS catalogs (
            catalog TEXT PRIMARY KEY,
            source_db_path TEXT,
            source_hash TEXT,
            n_rows INTEGER,
            build_time_iso TEXT,
            raw_rows_format TEXT,
            raw_rows_path TEXT,
            raw_rows_hash TEXT
        )
        """
    )
    cols = {row[1] for row in conn.execute("PRAGMA table_info(catalogs)").fetchall()}
    if "raw_rows_format" not in cols:
        conn.execute("ALTER TABLE catalogs ADD COLUMN raw_rows_format TEXT")
    if "raw_rows_path" not in cols:
        conn.execute("ALTER TABLE catalogs ADD COLUMN raw_rows_path TEXT")
    if "raw_rows_hash" not in cols:
        conn.execute("ALTER TABLE catalogs ADD COLUMN raw_rows_hash TEXT")
    conn.execute(
        "INSERT OR REPLACE INTO meta(key, value) VALUES('schema_version', ?)",
        ("3",),
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Migrate master_sources schema")
    ap.add_argument(
        "--db",
        type=Path,
        default=Path("state/catalogs/master_sources.sqlite3"),
        help="Path to master_sources.sqlite3",
    )
    ap.add_argument(
        "--rebuild",
        action="store_true",
        help="Rebuild master catalog with provenance enabled (overwrites sources table)",
    )
    ap.add_argument(
        "--backfill",
        action="store_true",
        help="Backfill catalog_matches for an existing master catalog",
    )
    ap.add_argument(
        "--clear-existing",
        action="store_true",
        help="Clear existing provenance rows before backfill",
    )
    ap.add_argument("--nvss-db", type=Path, default=None)
    ap.add_argument("--vlass-db", type=Path, default=None)
    ap.add_argument("--first-db", type=Path, default=None)
    ap.add_argument("--rax-db", type=Path, default=None)
    ap.add_argument("--match-radius", type=float, default=7.5)
    ap.add_argument("--chunk-size", type=int, default=100_000)
    ap.add_argument("--force", action="store_true", help="Overwrite existing DB on rebuild")

    args = ap.parse_args(argv)

    if args.rebuild:
        build_master_union_from_sqlite(
            output_path=args.db,
            nvss_db=args.nvss_db,
            vlass_db=args.vlass_db,
            first_db=args.first_db,
            rax_db=args.rax_db,
            match_radius_arcsec=args.match_radius,
            chunk_size=args.chunk_size,
            force_rebuild=args.force,
            with_provenance=True,
        )
        return 0

    if args.backfill:
        backfill_master_provenance_from_sqlite(
            master_db=args.db,
            nvss_db=args.nvss_db,
            vlass_db=args.vlass_db,
            first_db=args.first_db,
            rax_db=args.rax_db,
            match_radius_arcsec=args.match_radius,
            chunk_size=args.chunk_size,
            clear_existing=args.clear_existing,
        )
        return 0

    if not args.db.exists():
        raise FileNotFoundError(f"Master catalog not found: {args.db}")

    with sqlite3.connect(str(args.db)) as conn:
        _ensure_provenance_tables(conn)
        conn.commit()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
