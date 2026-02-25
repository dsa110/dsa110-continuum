#!/usr/bin/env python
"""Find all available transits for a calibrator in the HDF5 database.

This script uses find_transits_for_source() to efficiently query the database
for observations that actually contain the calibrator transit.
"""

import argparse
import sys

from dsa110_contimg.core.calibration.catalogs import load_vla_catalog_from_sqlite
from dsa110_contimg.core.calibration.transit import find_transits_for_source
from dsa110_contimg.infrastructure.database.unified import get_pipeline_db_path


def main():
    parser = argparse.ArgumentParser(
        description="Find all available transits for a calibrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Find all 0834+555 transits
  %(prog)s 0834+555

  # Search with custom tolerance
  %(prog)s 0834+555 --ra-tolerance 2.0 --dec-tolerance 2.0
""",
    )
    parser.add_argument("calibrator", help="Calibrator name (e.g., 0834+555)")
    parser.add_argument(
        "--catalog",
        default="/data/dsa110-contimg/state/catalogs/vla_calibrators.sqlite3",
        help="Path to calibrator catalog (default: %(default)s)",
    )
    parser.add_argument(
        "--db",
        default=None,
        help="Path to pipeline database (default: auto-detect)",
    )
    parser.add_argument(
        "--ra-tolerance",
        type=float,
        default=2.0,
        help="RA search tolerance in degrees (default: 2.0)",
    )
    parser.add_argument(
        "--dec-tolerance",
        type=float,
        default=2.0,
        help="Dec search tolerance in degrees (default: 2.0)",
    )

    args = parser.parse_args()

    # Load calibrator coordinates
    try:
        catalog = load_vla_catalog_from_sqlite(args.catalog)
        if args.calibrator not in catalog.index:
            print(f"Error: Calibrator {args.calibrator} not found in catalog")
            return 1
        source = catalog.loc[args.calibrator]
        ra_deg = float(source["ra_deg"])
        dec_deg = float(source["dec_deg"])
    except Exception as e:
        print(f"Error loading catalog: {e}")
        return 1

    print(f"Calibrator: {args.calibrator}")
    print(f"Coordinates: RA={ra_deg:.4f}°, Dec={dec_deg:.4f}°")
    print()

    # Get database path
    db_path = args.db or str(get_pipeline_db_path())
    print(f"Using database: {db_path}")
    print()

    # Find transits with data using find_transits_for_source
    print("Querying observations containing calibrator transits...")
    try:
        results = find_transits_for_source(
            db_path=db_path,
            ra_deg=ra_deg,
            dec_deg=dec_deg,
            ra_tolerance_deg=args.ra_tolerance,
            dec_tolerance_deg=args.dec_tolerance,
        )
    except Exception as e:
        print(f"Error querying database: {e}")
        return 1

    if not results:
        print(f"No observations found containing {args.calibrator} transits")
        return 0

    print(f"Found {len(results)} observation(s) with transits:")
    print()

    # Group by date
    by_date = {}
    for item in results:
        date = item["group_id"].split("T")[0]
        if date not in by_date:
            by_date[date] = []
        by_date[date].append(item)

    for date in sorted(by_date.keys()):
        items = by_date[date]
        print(f" {date}:")
        for item in items:
            transit_time = item["transit_time_iso"].replace(" ", "T").split("T")[1][:8]
            delta = item["delta_minutes"]
            print(
                f"   ✓ Obs: {item['group_id']} | Transit: {transit_time} | "
                f"Δ={delta:.1f} min from start"
            )
        print()

    print(f"Summary: {len(results)} observations with transits")
    return 0


if __name__ == "__main__":
    sys.exit(main())
