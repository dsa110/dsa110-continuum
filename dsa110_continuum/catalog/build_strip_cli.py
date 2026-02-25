#!/usr/bin/env python
"""
Unified CLI for building declination strip catalog databases.

This is the canonical entry point for strip catalog builds. Individual catalog
CLIs now delegate to this module.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from pathlib import Path

from dsa110_contimg.core.catalog.builders import (
    build_atnf_strip_db,
    build_first_strip_db,
    build_nvss_strip_db,
    build_rax_strip_db,
)
from dsa110_contimg.core.pointing.utils import load_pointing


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--hdf5", required=True, help="Path to HDF5 file to read declination from")
    parser.add_argument(
        "--dec-range",
        type=float,
        default=6.0,
        help="Declination range (+/- degrees around center, default: 6.0)",
    )
    parser.add_argument(
        "--output", help="Output SQLite database path (auto-generated if not provided)"
    )


def _load_dec_center(hdf5_path: Path) -> float:
    info = load_pointing(str(hdf5_path))
    if "dec_deg" not in info:
        raise KeyError(f"Could not read declination from {hdf5_path}")
    return float(info["dec_deg"])


def _run_strip_builder(
    builder: Callable[..., Path],
    args: argparse.Namespace,
    catalog_label: str,
) -> int:
    hdf5_path = Path(args.hdf5)
    if not hdf5_path.exists():
        print(f"Error: HDF5 file not found: {hdf5_path}")
        return 1

    try:
        dec_center = _load_dec_center(hdf5_path)
        print(f"Declination from {hdf5_path.name}: {dec_center:.6f} degrees")
    except Exception as exc:
        print(f"Error reading HDF5 file: {exc}")
        import traceback

        traceback.print_exc()
        return 1

    dec_min = dec_center - args.dec_range
    dec_max = dec_center + args.dec_range
    dec_range = (dec_min, dec_max)

    print(f"Building {catalog_label} SQLite database for declination strip:")
    print(f"  Center: {dec_center:.6f} degrees")
    print(f"  Range: {dec_min:.6f} to {dec_max:.6f} degrees (+/-{args.dec_range} degrees)")

    kwargs = {
        "dec_center": dec_center,
        "dec_range": dec_range,
        "output_path": args.output,
    }
    if hasattr(args, "min_flux_mjy"):
        kwargs["min_flux_mjy"] = args.min_flux_mjy
    if hasattr(args, "first_catalog_path"):
        kwargs["first_catalog_path"] = args.first_catalog_path
    if hasattr(args, "rax_catalog_path"):
        kwargs["rax_catalog_path"] = args.rax_catalog_path
    if hasattr(args, "cache_dir"):
        kwargs["cache_dir"] = args.cache_dir

    try:
        output_path = builder(**kwargs)
        print(f"\nSuccessfully built {catalog_label} declination strip database")
        print(f"  Database: {output_path}")
        return 0
    except Exception as exc:
        print(f"\nError building {catalog_label} database: {exc}")
        import traceback

        traceback.print_exc()
        return 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Build declination strip SQLite databases for survey catalogs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m dsa110_contimg.core.catalog.build_strip_cli nvss --hdf5 /path/file.hdf5
  python -m dsa110_contimg.core.catalog.build_strip_cli first --hdf5 /path/file.hdf5 --dec-range 4.0
  python -m dsa110_contimg.core.catalog.build_strip_cli atnf --hdf5 /path/file.hdf5 --min-flux-mjy 10
""",
    )

    subparsers = ap.add_subparsers(dest="catalog", required=True)

    nvss = subparsers.add_parser("nvss", help="Build NVSS declination strip database")
    _add_common_args(nvss)
    nvss.add_argument("--min-flux-mjy", type=float, help="Minimum flux threshold in mJy (optional)")
    nvss.set_defaults(builder=build_nvss_strip_db, catalog_label="NVSS")

    first = subparsers.add_parser("first", help="Build FIRST declination strip database")
    _add_common_args(first)
    first.add_argument(
        "--min-flux-mjy", type=float, help="Minimum flux threshold in mJy (optional)"
    )
    first.add_argument(
        "--first-catalog-path",
        help="Path to FIRST catalog file (CSV/FITS). If not provided, attempts to auto-download/cache.",
    )
    first.add_argument(
        "--cache-dir",
        default=".cache/catalogs",
        help="Directory for caching catalog files (default: .cache/catalogs)",
    )
    first.set_defaults(builder=build_first_strip_db, catalog_label="FIRST")

    atnf = subparsers.add_parser("atnf", help="Build ATNF declination strip database")
    _add_common_args(atnf)
    atnf.add_argument(
        "--min-flux-mjy",
        type=float,
        help="Minimum flux threshold at 1400 MHz in mJy (optional)",
    )
    atnf.add_argument(
        "--cache-dir",
        default=".cache/catalogs",
        help="Directory for caching catalog files (default: .cache/catalogs)",
    )
    atnf.set_defaults(builder=build_atnf_strip_db, catalog_label="ATNF")

    rax = subparsers.add_parser("rax", help="Build RAX declination strip database")
    _add_common_args(rax)
    rax.add_argument("--min-flux-mjy", type=float, help="Minimum flux threshold in mJy (optional)")
    rax.add_argument(
        "--rax-catalog-path",
        help="Path to RAX catalog file (CSV/FITS). If not provided, attempts to find cached catalog.",
    )
    rax.add_argument(
        "--cache-dir",
        default=".cache/catalogs",
        help="Directory for caching catalog files (default: .cache/catalogs)",
    )
    rax.set_defaults(builder=build_rax_strip_db, catalog_label="RAX")

    args = ap.parse_args(argv)
    return _run_strip_builder(args.builder, args, args.catalog_label)


if __name__ == "__main__":
    sys.exit(main())
