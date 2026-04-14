#!/opt/miniforge/envs/casa6/bin/python
"""Stack per-epoch forced photometry CSVs into a cross-epoch light curve Parquet.

Usage:
    python scripts/stack_lightcurves.py [--products-dir /data/dsa110-continuum/products]

Delegates to dsa110_continuum.lightcurves.stacker for all logic.
"""
import argparse
import os
import sys
from pathlib import Path

from dsa110_continuum.lightcurves.stacker import (
    assign_source_ids,
    parse_epoch_utc,
    stack_csvs,
)

PRODUCTS_DIR = Path(os.environ.get("DSA110_PRODUCTS_BASE", "/data/dsa110-proc/products/mosaics")).parent


def main():
    parser = argparse.ArgumentParser(description="Stack forced-phot CSVs into light curve Parquet.")
    parser.add_argument("--products-dir", default=str(PRODUCTS_DIR))
    parser.add_argument("--match-arcsec", type=float, default=5.0)
    args = parser.parse_args()

    products = Path(args.products_dir)
    csv_paths = sorted(products.glob("mosaics/*/*.csv"))
    if not csv_paths:
        print(f"No forced-phot CSVs found under {products}/mosaics/", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(csv_paths)} CSVs:")
    for p in csv_paths:
        print(f"  {p}")

    df = stack_csvs([str(p) for p in csv_paths], match_arcsec=args.match_arcsec)
    print(f"\nStacked {len(df)} source-epoch rows, {df['source_id'].nunique()} unique sources.")

    out_dir = products / "lightcurves"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "lightcurves.parquet"
    df.to_parquet(out_path, index=False)
    print(f"Written: {out_path}")


if __name__ == "__main__":
    main()
