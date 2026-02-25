#!/usr/bin/env python
"""
Unified CLI for building the master catalog.

This is the canonical entry point for master catalog builds, supporting both:
  - files: CSV/FITS catalogs
  - sqlite: full-sky SQLite catalogs
"""

from __future__ import annotations

import argparse
import logging
import os
import sqlite3
import sys
from pathlib import Path

from dsa110_contimg.core.catalog.build_master import (
    build_master,
    build_master_from_sqlite,
    build_master_union_from_sqlite,
)

logger = logging.getLogger(__name__)


def _unbuffer_std_streams() -> None:
    """Force stdout/stderr to line-buffer so output streams live (no block buffering)."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(line_buffering=True)


def _add_map_args(parser: argparse.ArgumentParser, prefix: str) -> None:
    parser.add_argument(f"--map-{prefix}-ra", dest=f"map_{prefix}_ra")
    parser.add_argument(f"--map-{prefix}-dec", dest=f"map_{prefix}_dec")
    parser.add_argument(f"--map-{prefix}-flux", dest=f"map_{prefix}_flux")
    if prefix == "first":
        parser.add_argument(f"--map-{prefix}-maj", dest=f"map_{prefix}_maj")
        parser.add_argument(f"--map-{prefix}-min", dest=f"map_{prefix}_min")


def _parse_map_args(args: argparse.Namespace, prefix: str) -> dict:
    prefix_key = f"map_{prefix}_"
    return {
        key.split(prefix_key)[1]: value
        for key, value in vars(args).items()
        if key.startswith(prefix_key) and value
    }


def _run_files(args: argparse.Namespace) -> int:
    map_nv = _parse_map_args(args, "nvss")
    map_vl = _parse_map_args(args, "vlass")
    map_fi = _parse_map_args(args, "first")

    try:
        outp = build_master(
            args.nvss,
            vlass_path=args.vlass,
            first_path=args.first,
            out_db=args.out,
            match_radius_arcsec=args.match_radius_arcsec,
            map_nvss=map_nv or None,
            map_vlass=map_vl or None,
            map_first=map_fi or None,
            nvss_flux_unit=args.nvss_flux_unit,
            vlass_flux_unit=args.vlass_flux_unit,
            goodref_snr_min=args.goodref_snr_min,
            goodref_alpha_min=args.goodref_alpha_min,
            goodref_alpha_max=args.goodref_alpha_max,
            finalref_snr_min=args.finalref_snr_min,
            finalref_ids_file=args.finalref_ids,
            materialize_final=args.materialize_final,
        )
        logger.info("Wrote master catalog to: %s", outp)
        print(f"Wrote master catalog to: {outp}", flush=True)

        if args.export_view:
            allowed_views = {
                "sources",
                "good_references",
                "final_references",
                "final_references_table",
            }
            if args.export_view not in allowed_views:
                raise ValueError(
                    f"Invalid export view: {args.export_view}. "
                    f"Allowed views: {', '.join(sorted(allowed_views))}"
                )
            try:
                import pandas as _pd

                with sqlite3.connect(os.fspath(outp)) as _conn:
                    df = _pd.read_sql_query(f"SELECT * FROM {args.export_view}", _conn)
                export_path = (
                    Path(args.export_csv)
                    if args.export_csv
                    else Path(outp)
                    .with_suffix("")
                    .with_name(f"{Path(outp).stem}_{args.export_view}.csv")
                )
                export_path.parent.mkdir(parents=True, exist_ok=True)
                df.to_csv(export_path, index=False)
                logger.info("Exported %s to: %s", args.export_view, export_path)
                print(f"Exported {args.export_view} to: {export_path}", flush=True)
            except Exception as exc:
                logger.error("Export failed: %s", exc, exc_info=True)
                print(f"Export failed: {exc}", flush=True)
        return 0
    except Exception as exc:
        logger.error("Failed to build master catalog: %s", exc, exc_info=True)
        print(f"Failed to build master catalog: {exc}", flush=True)
        return 1


def _run_sqlite(args: argparse.Namespace) -> int:
    if args.verbose:
        root = logging.getLogger()
        root.setLevel(logging.DEBUG)
        if not root.handlers:
            h = logging.StreamHandler(sys.stdout)
            h.setLevel(logging.INFO)
            root.addHandler(h)

    def _progress(processed: int, total: int) -> None:
        pct = 100 * processed / total if total else 0
        print(f"  Processed {processed:,}/{total:,} ({pct:.1f}%)", file=sys.stderr, flush=True)

    print("Building master catalog from SQLite databases...", file=sys.stderr, flush=True)
    try:
        build_fn = build_master_union_from_sqlite if args.union else build_master_from_sqlite
        output_path = build_fn(
            output_path=args.out,
            nvss_db=args.nvss_db,
            vlass_db=args.vlass_db,
            first_db=args.first_db,
            rax_db=args.rax_db,
            match_radius_arcsec=args.match_radius,
            chunk_size=args.chunk_size,
            force_rebuild=args.force,
            with_provenance=args.provenance,
            progress_callback=_progress,
            resume=getattr(args, "resume", False),
        )
        print(f"\nBuilt master catalog: {output_path}", flush=True)

        with sqlite3.connect(str(output_path)) as conn:
            total = conn.execute("SELECT COUNT(*) FROM sources").fetchone()[0]
            with_nvss = conn.execute("SELECT COUNT(*) FROM sources WHERE has_nvss=1").fetchone()[0]
            # Legacy DBs may have has_nvss=0 for all rows (builder bug); infer from s_nvss
            if with_nvss == 0 and total > 0:
                with_nvss = conn.execute(
                    "SELECT COUNT(*) FROM sources WHERE s_nvss IS NOT NULL"
                ).fetchone()[0]
            with_vlass = conn.execute("SELECT COUNT(*) FROM sources WHERE has_vlass=1").fetchone()[
                0
            ]
            with_first = conn.execute("SELECT COUNT(*) FROM sources WHERE has_first=1").fetchone()[
                0
            ]
            with_rax = conn.execute("SELECT COUNT(*) FROM sources WHERE has_rax=1").fetchone()[0]
            with_alpha = conn.execute(
                "SELECT COUNT(*) FROM sources WHERE alpha IS NOT NULL"
            ).fetchone()[0]

        print("Summary:", flush=True)
        print(f"  Total sources: {total:,}", flush=True)
        print(f"  With NVSS: {with_nvss:,}", flush=True)
        print(f"  With VLASS: {with_vlass:,}", flush=True)
        print(f"  With FIRST: {with_first:,}", flush=True)
        print(f"  With RAX: {with_rax:,}", flush=True)
        print(f"  With alpha: {with_alpha:,}", flush=True)
        return 0
    except Exception as exc:
        logger.error("Failed to build master catalog: %s", exc, exc_info=True)
        print(f"Failed to build master catalog: {exc}", flush=True)
        return 1


def main(argv: list[str] | None = None) -> int:
    _unbuffer_std_streams()
    ap = argparse.ArgumentParser(
        description="Build master catalog (files or SQLite sources)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m dsa110_contimg.core.catalog.build_master_cli files --nvss NVSS.csv --vlass VLASS.csv
  python -m dsa110_contimg.core.catalog.build_master_cli sqlite --force --match-radius 10.0
""",
    )
    subparsers = ap.add_subparsers(dest="source", required=True)

    files = subparsers.add_parser("files", help="Build from CSV/FITS catalogs")
    files.add_argument("--nvss", required=True, help="Path to NVSS catalog (CSV/FITS)")
    files.add_argument("--vlass", help="Path to VLASS catalog (CSV/FITS)")
    files.add_argument("--first", help="Path to FIRST catalog (CSV/FITS)")
    files.add_argument(
        "--out",
        default="state/catalogs/master_sources.sqlite3",
        help="Output SQLite DB path",
    )
    files.add_argument("--match-radius-arcsec", type=float, default=7.5)
    files.add_argument(
        "--nvss-flux-unit",
        choices=["jy", "mjy", "ujy"],
        default="jy",
        help="Units of NVSS flux column (converted to Jy)",
    )
    files.add_argument(
        "--vlass-flux-unit",
        choices=["jy", "mjy", "ujy"],
        default="jy",
        help="Units of VLASS flux column (converted to Jy)",
    )
    files.add_argument(
        "--goodref-snr-min",
        type=float,
        default=50.0,
        help="SNR threshold for good reference view",
    )
    files.add_argument(
        "--goodref-alpha-min",
        type=float,
        default=-1.2,
        help="Min alpha for good reference view",
    )
    files.add_argument(
        "--goodref-alpha-max",
        type=float,
        default=0.2,
        help="Max alpha for good reference view",
    )
    files.add_argument(
        "--finalref-snr-min",
        type=float,
        default=80.0,
        help="SNR threshold for final references view",
    )
    files.add_argument(
        "--finalref-ids",
        help="Optional file with source_id list (one per line) to define long-term stable set",
    )
    files.add_argument(
        "--materialize-final",
        action="store_true",
        help="Create final_references_table materialized from view",
    )
    files.add_argument(
        "--export-view",
        choices=[
            "sources",
            "good_references",
            "final_references",
            "final_references_table",
        ],
        help="Optionally export a table/view to CSV after building the DB",
    )
    files.add_argument(
        "--export-csv",
        help="Path to CSV to write for --export-view (defaults to <out>_<view>.csv)",
    )
    _add_map_args(files, "nvss")
    _add_map_args(files, "vlass")
    _add_map_args(files, "first")
    files.set_defaults(runner=_run_files)

    sqlite = subparsers.add_parser("sqlite", help="Build from SQLite full catalogs")
    sqlite.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output path (default: state/catalogs/master_sources.sqlite3)",
    )
    sqlite.add_argument(
        "--nvss-db",
        type=Path,
        default=None,
        help="Path to nvss_full.sqlite3 (auto-detected if not specified)",
    )
    sqlite.add_argument(
        "--vlass-db",
        type=Path,
        default=None,
        help="Path to vlass_full.sqlite3 (auto-detected if not specified)",
    )
    sqlite.add_argument(
        "--first-db",
        type=Path,
        default=None,
        help="Path to first_full.sqlite3 (auto-detected if not specified)",
    )
    sqlite.add_argument(
        "--rax-db",
        type=Path,
        default=None,
        help="Path to rax_full.sqlite3 (RACS, auto-detected if not specified)",
    )
    sqlite.add_argument(
        "--match-radius",
        type=float,
        default=7.5,
        help="Crossmatch radius in arcseconds (default: 7.5)",
    )
    sqlite.add_argument(
        "--chunk-size",
        type=int,
        default=100_000,
        help="Process sources in chunks of this size (default: 100000)",
    )
    sqlite.add_argument(
        "--union",
        action="store_true",
        help="Build a true union master catalog (includes sources from all catalogs, not NVSS-centered)",
    )
    sqlite.add_argument(
        "--provenance",
        dest="provenance",
        action="store_true",
        default=True,
        help="Store per-catalog provenance tables in master_sources.sqlite3 (default: enabled)",
    )
    sqlite.add_argument(
        "--no-provenance",
        dest="provenance",
        action="store_false",
        help="Disable per-catalog provenance tables",
    )
    sqlite.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing master_sources.sqlite3",
    )
    sqlite.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing DB (append remaining chunks; use after interrupted run)",
    )
    sqlite.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    sqlite.set_defaults(runner=_run_sqlite)

    args = ap.parse_args(argv)
    return args.runner(args)


if __name__ == "__main__":
    # Force stderr line-buffered so progress appears immediately (no block buffering)
    try:
        sys.stderr = open(sys.stderr.fileno(), "w", buffering=1)
    except (AttributeError, OSError):
        pass
    print("build_master_cli: starting", file=sys.stderr, flush=True)
    sys.exit(main())
