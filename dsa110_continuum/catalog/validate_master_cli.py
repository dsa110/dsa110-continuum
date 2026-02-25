#!/usr/bin/env python
"""Validate master_sources.sqlite3 provenance consistency."""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
        (name,),
    ).fetchone()
    return row is not None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Validate master_sources provenance tables")
    ap.add_argument(
        "--db",
        type=Path,
        default=Path("state/catalogs/master_sources.sqlite3"),
        help="Path to master_sources.sqlite3",
    )
    ap.add_argument(
        "--sample-raw",
        type=int,
        default=5,
        help="Number of raw-row samples to validate per catalog (parquet)",
    )
    args = ap.parse_args(argv)

    if not args.db.exists():
        raise FileNotFoundError(f"Master catalog not found: {args.db}")

    required_tables = ["sources", "catalog_matches", "catalogs", "meta"]
    catalogs = ["nvss", "vlass", "first", "rax"]
    exit_code = 0

    with sqlite3.connect(str(args.db)) as conn:
        missing = [t for t in required_tables if not _table_exists(conn, t)]
        if missing:
            print(f"Missing tables: {', '.join(missing)}")
            return 1

        for cat in catalogs:
            has_col = f"has_{cat}"
            cols = {row[1] for row in conn.execute("PRAGMA table_info(sources)").fetchall()}
            if has_col not in cols:
                print(f"Missing column in sources: {has_col}")
                exit_code = 1
                continue

            has_count = conn.execute(
                f"SELECT COUNT(*) FROM sources WHERE {has_col}=1"
            ).fetchone()[0]
            match_count = conn.execute(
                "SELECT COUNT(DISTINCT source_id) FROM catalog_matches WHERE catalog=?",
                (cat,),
            ).fetchone()[0]
            mismatch_has = conn.execute(
                f"""
                SELECT COUNT(*) FROM sources s
                WHERE s.{has_col}=1
                  AND NOT EXISTS(
                      SELECT 1 FROM catalog_matches m
                      WHERE m.source_id=s.source_id AND m.catalog=?
                  )
                """,
                (cat,),
            ).fetchone()[0]
            mismatch_extra = conn.execute(
                f"""
                SELECT COUNT(*) FROM sources s
                WHERE s.{has_col}=0
                  AND EXISTS(
                      SELECT 1 FROM catalog_matches m
                      WHERE m.source_id=s.source_id AND m.catalog=?
                  )
                """,
                (cat,),
            ).fetchone()[0]

            print(
                f"{cat}: has_*={has_count:,} matches={match_count:,} "
                f"missing_has={mismatch_has:,} extra_matches={mismatch_extra:,}"
            )
            if mismatch_has or mismatch_extra:
                exit_code = 1

        # Raw-row metadata checks
        cat_meta = conn.execute(
            "SELECT catalog, raw_rows_format, raw_rows_path FROM catalogs"
        ).fetchall()
        raw_meta = {row[0]: {"format": row[1], "path": row[2]} for row in cat_meta}
        for cat in catalogs:
            match_count = conn.execute(
                "SELECT COUNT(*) FROM catalog_matches WHERE catalog=?",
                (cat,),
            ).fetchone()[0]
            meta = raw_meta.get(cat)
            if match_count and not meta:
                print(f"{cat}: missing catalogs metadata row")
                exit_code = 1
                continue
            if not meta:
                continue
            if meta["format"] and meta["format"].lower() != "parquet":
                print(f"{cat}: raw_rows_format is {meta['format']} (expected parquet)")
                exit_code = 1
            if meta["path"]:
                if not Path(meta["path"]).exists():
                    print(f"{cat}: raw_rows_path missing: {meta['path']}")
                    exit_code = 1
            elif match_count:
                print(f"{cat}: raw_rows_path not set")
                exit_code = 1

        # Parquet validation samples (best-effort)
        if args.sample_raw > 0:
            try:
                import pandas as pd  # noqa: WPS433 (runtime import)
            except Exception as exc:
                print(f"Skipping parquet validation (pandas import failed): {exc}")
                return exit_code

            for cat, meta in raw_meta.items():
                if not meta.get("path"):
                    continue
                try:
                    df = pd.read_parquet(meta["path"], columns=["catalog_row_id"])
                    if "catalog_row_id" not in df.columns:
                        print(f"{cat}: parquet missing catalog_row_id column")
                        exit_code = 1
                except ImportError as exc:
                    print(f"Skipping parquet validation (engine missing): {exc}")
                    break
                except Exception as exc:  # pragma: no cover - unexpected
                    print(f"{cat}: failed to read parquet: {exc}")
                    exit_code = 1

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
