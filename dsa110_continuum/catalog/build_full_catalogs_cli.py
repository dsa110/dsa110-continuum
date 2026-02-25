#!/usr/bin/env python
"""
Build full SQLite databases for all survey catalogs.

This creates comprehensive databases with all sources, indexed for fast spatial queries:
  - nvss_full.sqlite3   (~218 MB, 1.77M sources)
  - first_full.sqlite3  (requires Vizier download or cached file)
  - vlass_full.sqlite3  (requires cached file)
  - rax_full.sqlite3    (requires cached file)
  - atnf_full.sqlite3   (~few MB, all known pulsars)

Once built, dec strip databases can be created much faster using SQL queries
instead of re-downloading/parsing raw catalog files.

Usage:
    # Build all available catalogs
    python -m dsa110_contimg.core.catalog.build_full_catalogs_cli --all

    # Build specific catalog
    python -m dsa110_contimg.core.catalog.build_full_catalogs_cli nvss
    python -m dsa110_contimg.core.catalog.build_full_catalogs_cli first
    python -m dsa110_contimg.core.catalog.build_full_catalogs_cli atnf

    # Check status of all catalogs
    python -m dsa110_contimg.core.catalog.build_full_catalogs_cli --status

    # Force rebuild
    python -m dsa110_contimg.core.catalog.build_full_catalogs_cli nvss --force
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

from dsa110_contimg.core.catalog.builders import (
    atnf_full_db_exists,
    # ATNF
    build_atnf_full_db,
    # FIRST
    build_first_full_db,
    # NVSS
    build_nvss_full_db,
    # RAX
    build_rax_full_db,
    # VLASS
    build_vlass_full_db,
    first_full_db_exists,
    get_atnf_full_db_path,
    get_first_full_db_path,
    get_nvss_full_db_path,
    get_rax_full_db_path,
    get_vlass_full_db_path,
    nvss_full_db_exists,
    rax_full_db_exists,
    vlass_full_db_exists,
)

CATALOG_INFO = {
    "nvss": {
        "build": build_nvss_full_db,
        "path": get_nvss_full_db_path,
        "exists": nvss_full_db_exists,
        "description": "NVSS (NRAO VLA Sky Survey) - 1.4 GHz continuum",
    },
    "first": {
        "build": build_first_full_db,
        "path": get_first_full_db_path,
        "exists": first_full_db_exists,
        "description": "FIRST (Faint Images of the Radio Sky at Twenty-cm)",
    },
    "vlass": {
        "build": build_vlass_full_db,
        "path": get_vlass_full_db_path,
        "exists": vlass_full_db_exists,
        "description": "VLASS (VLA Sky Survey) - 3 GHz continuum",
    },
    "rax": {
        "build": build_rax_full_db,
        "path": get_rax_full_db_path,
        "exists": rax_full_db_exists,
        "description": "RAX (DSA-110 specific catalog)",
    },
    "atnf": {
        "build": build_atnf_full_db,
        "path": get_atnf_full_db_path,
        "exists": atnf_full_db_exists,
        "description": "ATNF Pulsar Catalogue",
    },
}


def get_db_stats(db_path: Path) -> dict:
    """Get statistics for a database."""
    if not db_path.exists():
        return {"exists": False}

    stats = {"exists": True, "size_mb": db_path.stat().st_size / (1024 * 1024)}

    try:
        with sqlite3.connect(str(db_path)) as conn:
            stats["n_sources"] = conn.execute("SELECT COUNT(*) FROM sources").fetchone()[0]

            # Try to get build time
            try:
                row = conn.execute("SELECT value FROM meta WHERE key='build_time_iso'").fetchone()
                if row:
                    stats["build_time"] = row[0]
            except Exception:
                pass
    except Exception as e:
        stats["error"] = str(e)

    return stats


def print_status() -> None:
    """Print status of all catalog databases."""
    print("\n Full Catalog Database Status")
    print("=" * 70)

    for name, info in CATALOG_INFO.items():
        db_path = info["path"]()
        stats = get_db_stats(db_path)

        if stats.get("exists"):
            size = stats.get("size_mb", 0)
            n_sources = stats.get("n_sources", "?")
            build_time = stats.get("build_time", "unknown")
            print(f"\n {name.upper()}: {info['description']}")
            print(f"   Path: {db_path}")
            print(f"   Size: {size:.2f} MB | Sources: {n_sources:,}")
            print(f"   Built: {build_time}")
        else:
            print(f"\n {name.upper()}: {info['description']}")
            print(f"   Path: {db_path}")
            print("   Status: Not built")

    print("\n" + "=" * 70)


def build_catalog(name: str, force: bool = False) -> bool:
    """Build a specific catalog database."""
    if name not in CATALOG_INFO:
        print(f" Unknown catalog: {name}")
        print(f"   Available: {', '.join(CATALOG_INFO.keys())}")
        return False

    info = CATALOG_INFO[name]
    db_path = info["path"]()

    if info["exists"]() and not force:
        stats = get_db_stats(db_path)
        print(f" {name.upper()} database already exists")
        print(f"   Path: {db_path}")
        print(
            f"   Size: {stats.get('size_mb', 0):.2f} MB | Sources: {stats.get('n_sources', '?'):,}"
        )
        print("   Use --force to rebuild")
        return True

    print(f" Building {name.upper()} full database...")
    try:
        result_path = info["build"](force_rebuild=force)
        stats = get_db_stats(result_path)
        print(f" Built {name.upper()} database")
        print(f"   Path: {result_path}")
        print(
            f"   Size: {stats.get('size_mb', 0):.2f} MB | Sources: {stats.get('n_sources', '?'):,}"
        )
        return True
    except FileNotFoundError as e:
        print(f"  Could not build {name.upper()}: {e}")
        print("   The source catalog file may not be available.")
        return False
    except Exception as e:
        print(f" Failed to build {name.upper()}: {e}")
        return False


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Build full SQLite databases for survey catalogs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Show status of all catalogs
    python -m dsa110_contimg.core.catalog.build_full_catalogs_cli --status

    # Build NVSS database
    python -m dsa110_contimg.core.catalog.build_full_catalogs_cli nvss

    # Build ATNF database (requires psrqpy)
    python -m dsa110_contimg.core.catalog.build_full_catalogs_cli atnf

    # Build all available catalogs
    python -m dsa110_contimg.core.catalog.build_full_catalogs_cli --all

    # Force rebuild
    python -m dsa110_contimg.core.catalog.build_full_catalogs_cli nvss --force
""",
    )

    ap.add_argument(
        "catalog",
        nargs="?",
        choices=list(CATALOG_INFO.keys()),
        help="Catalog to build (nvss, first, vlass, rax, atnf)",
    )
    ap.add_argument(
        "--all",
        action="store_true",
        help="Build all available catalog databases",
    )
    ap.add_argument(
        "--status",
        action="store_true",
        help="Show status of all catalog databases",
    )
    ap.add_argument(
        "--force",
        action="store_true",
        help="Force rebuild even if database exists",
    )

    args = ap.parse_args(argv)

    # Status check
    if args.status:
        print_status()
        return 0

    # Build all
    if args.all:
        success_count = 0
        for name in CATALOG_INFO:
            print()
            if build_catalog(name, force=args.force):
                success_count += 1
        print(f"\n Built {success_count}/{len(CATALOG_INFO)} catalogs")
        return 0 if success_count > 0 else 1

    # Build specific catalog
    if args.catalog:
        return 0 if build_catalog(args.catalog, force=args.force) else 1

    # No action specified
    print_status()
    print("\nTo build a catalog, run:")
    print("  python -m dsa110_contimg.core.catalog.build_full_catalogs_cli <catalog>")
    print("  python -m dsa110_contimg.core.catalog.build_full_catalogs_cli --all")
    return 0


if __name__ == "__main__":
    sys.exit(main())
