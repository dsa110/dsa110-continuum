"""Scan /data/incoming-style tree and emit manifest + static site."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

from dsacamera_monitor.manifest import (
    BeamAgg,
    DayAgg,
    ScanAccum,
    build_manifest,
    try_parse_filename,
)


def scan_directory(root: Path, *, no_stat: bool) -> ScanAccum:
    """Single pass over directory entries; only matching *.hdf5 names are counted."""
    accum = ScanAccum()
    with os.scandir(root) as it:
        for entry in it:
            if not entry.is_file():
                continue
            name = entry.name
            if not name.endswith(".hdf5"):
                continue
            parsed = try_parse_filename(name)
            if parsed is None:
                continue
            dt_utc, beam = parsed
            day = dt_utc.date()

            if day not in accum.by_day:
                accum.by_day[day] = DayAgg()
            day_agg = accum.by_day[day]
            day_agg.count += 1

            if beam not in accum.by_beam:
                accum.by_beam[beam] = BeamAgg()
            beam_agg = accum.by_beam[beam]
            beam_agg.count += 1

            if accum.latest_filename_dt is None or dt_utc > accum.latest_filename_dt:
                accum.latest_filename_dt = dt_utc
            if accum.earliest_filename_dt is None or dt_utc < accum.earliest_filename_dt:
                accum.earliest_filename_dt = dt_utc

            if not no_stat:
                try:
                    st = entry.stat(follow_symlinks=False)
                except OSError:
                    continue
                size = st.st_size
                mtime_dt = datetime.fromtimestamp(st.st_mtime, tz=timezone.utc)
                day_agg.bytes += size
                beam_agg.bytes += size
                accum.total_bytes += size
                if accum.latest_mtime is None or mtime_dt > accum.latest_mtime:
                    accum.latest_mtime = mtime_dt
                if accum.earliest_mtime is None or mtime_dt < accum.earliest_mtime:
                    accum.earliest_mtime = mtime_dt

            accum.file_count += 1

    return accum


def build_out(
    *,
    root: Path,
    out_dir: Path,
    no_stat: bool,
    site_dir: Path | None = None,
) -> Path:
    """Scan, write manifest.json, copy static site into out_dir."""
    accum = scan_directory(root, no_stat=no_stat)
    manifest = build_manifest(
        source_root=str(root.resolve()),
        accum=accum,
        no_stat=no_stat,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")

    if site_dir is None:
        site_dir = Path(__file__).resolve().parent / "site"
    if site_dir.is_dir():
        for child in site_dir.iterdir():
            dest = out_dir / child.name
            if child.is_file():
                shutil.copy2(child, dest)
            else:
                shutil.copytree(child, dest, dirs_exist_ok=True)

    return manifest_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Scan DSA-110 incoming HDF5 files and build static dashboard output."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("/data/incoming"),
        help="Directory to scan (default: /data/incoming)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output directory (manifest.json + copied site assets)",
    )
    parser.add_argument(
        "--no-stat",
        action="store_true",
        help="Do not stat() files; counts only (bytes and mtime freshness omitted)",
    )
    parser.add_argument(
        "--site",
        type=Path,
        default=None,
        help="Override path to static site directory (default: package site/)",
    )
    args = parser.parse_args(argv)

    root = args.root
    if not root.is_dir():
        print(f"error: not a directory: {root}", file=sys.stderr)
        return 1

    build_out(root=root, out_dir=args.out, no_stat=args.no_stat, site_dir=args.site)
    print(f"Wrote {args.out / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
