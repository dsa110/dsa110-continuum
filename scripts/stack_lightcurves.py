#!/opt/miniforge/envs/casa6/bin/python
"""Stack per-epoch forced photometry CSVs into a cross-epoch light curve Parquet.

Usage:
    python scripts/stack_lightcurves.py [--products-dir /data/dsa110-continuum/products]
"""
import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
import astropy.units as u

PRODUCTS_DIR = Path("/data/dsa110-continuum/products")


def parse_epoch_utc(filename: str) -> str:
    """Extract ISO8601 UTC string from forced-phot CSV filename.

    Examples:
        2026-02-12T0000_forced_phot.csv  ->  2026-02-12T00:00:00
        2026-01-25T2200_forced_phot.csv  ->  2026-01-25T22:00:00
    """
    m = re.search(r"(\d{4}-\d{2}-\d{2})T(\d{2})(\d{2})_forced_phot", filename)
    if not m:
        raise ValueError(f"Cannot parse epoch from filename: {filename}")
    date, hh, mm = m.group(1), m.group(2), m.group(3)
    return f"{date}T{hh}:{mm}:00"


def assign_source_ids(df: pd.DataFrame, match_arcsec: float = 5.0) -> pd.DataFrame:
    """Assign a stable integer source_id to each row by clustering RA/Dec positions.

    Rows within match_arcsec of each other get the same source_id.
    Uses greedy first-occurrence assignment via SkyCoord matching.
    """
    coords = SkyCoord(ra=df["ra_deg"].values * u.deg, dec=df["dec_deg"].values * u.deg)
    source_ids = np.full(len(df), -1, dtype=int)
    next_id = 0
    for i in range(len(df)):
        if source_ids[i] != -1:
            continue
        source_ids[i] = next_id
        sep = coords[i].separation(coords).arcsec
        matches = (sep < match_arcsec) & (source_ids == -1)
        source_ids[matches] = next_id
        next_id += 1
    df = df.copy()
    df["source_id"] = source_ids
    return df


def stack_csvs(csv_paths: list, match_arcsec: float = 5.0) -> pd.DataFrame:
    """Read all forced-phot CSVs, assign source_ids, return stacked DataFrame."""
    frames = []
    for path in csv_paths:
        df = pd.read_csv(path)
        fname = Path(path).name
        df["epoch_utc"] = parse_epoch_utc(fname)
        df["date"] = fname[:10]
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    combined = assign_source_ids(combined, match_arcsec=match_arcsec)
    return combined


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
