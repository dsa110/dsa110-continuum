"""Scan /data/incoming-style tree and emit manifest + static site."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from datetime import date, datetime, timezone
from pathlib import Path

from dsacamera_monitor.hdf5_pointing import DEC_ROUND_DIGITS, read_pointing_metadata
from dsacamera_monitor.manifest import (
    BeamAgg,
    DayAgg,
    ScanAccum,
    build_manifest,
    try_parse_filename,
)


def _record_dec(accum: ScanAccum, day: date, dec: float) -> None:
    dr = round(dec, DEC_ROUND_DIGITS)
    accum.global_decs_rounded.add(dr)
    if accum.global_dec_min is None:
        accum.global_dec_min = dec
        accum.global_dec_max = dec
    else:
        accum.global_dec_min = min(accum.global_dec_min, dec)
        accum.global_dec_max = max(accum.global_dec_max, dec)
    dagg = accum.by_day[day]
    dagg.decs_rounded.add(dr)
    if dagg.dec_min is None:
        dagg.dec_min = dec
        dagg.dec_max = dec
    else:
        dagg.dec_min = min(dagg.dec_min, dec)
        dagg.dec_max = max(dagg.dec_max, dec)


def scan_directory(
    root: Path,
    *,
    no_stat: bool,
    hdf5_metadata: bool = True,
    pointing_timeseries: bool = False,
    pointing_timeseries_max_files: int = 5000,
) -> ScanAccum:
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
            full_path = Path(entry.path)

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

            if hdf5_metadata:
                meta = read_pointing_metadata(full_path)
                if meta["dec_status"] == "ok":
                    dec = meta["dec_deg"]
                    assert dec is not None
                    accum.files_with_dec += 1
                    _record_dec(accum, day, dec)
                elif meta["dec_status"] == "missing":
                    accum.files_dec_missing += 1
                else:
                    accum.files_dec_read_failed += 1

                if pointing_timeseries and len(accum.timeseries_rows) < pointing_timeseries_max_files:
                    accum.timeseries_rows.append(
                        {
                            "filename": meta["filename"],
                            "t_mid_utc": meta["t_mid_utc"],
                            "ra_deg": meta["ra_deg"],
                            "dec_deg": meta["dec_deg"],
                        }
                    )
                if meta["pointing_status"] == "read_failed":
                    accum.files_pointing_read_failed += 1

    if pointing_timeseries and hdf5_metadata and accum.file_count > len(accum.timeseries_rows):
        accum.timeseries_truncated = True

    return accum


def build_out(
    *,
    root: Path,
    out_dir: Path,
    no_stat: bool,
    site_dir: Path | None = None,
    hdf5_metadata: bool = True,
    pointing_timeseries: bool = False,
    pointing_timeseries_max_files: int = 5000,
) -> tuple[Path, bool]:
    """Scan, write manifest.json, optional pointing_timeseries.json, copy static site into out_dir."""
    accum = scan_directory(
        root,
        no_stat=no_stat,
        hdf5_metadata=hdf5_metadata,
        pointing_timeseries=pointing_timeseries,
        pointing_timeseries_max_files=pointing_timeseries_max_files,
    )
    manifest = build_manifest(
        source_root=str(root.resolve()),
        accum=accum,
        no_stat=no_stat,
        hdf5_metadata=hdf5_metadata,
        pointing_timeseries=pointing_timeseries,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")

    wrote_timeseries = False
    if accum.timeseries_rows:
        ts_path = out_dir / "pointing_timeseries.json"
        with open(ts_path, "w", encoding="utf-8") as f:
            json.dump(accum.timeseries_rows, f, indent=2)
            f.write("\n")
        wrote_timeseries = True

    if site_dir is None:
        site_dir = Path(__file__).resolve().parent / "site"
    if site_dir.is_dir():
        for child in site_dir.iterdir():
            dest = out_dir / child.name
            if child.is_file():
                shutil.copy2(child, dest)
            else:
                shutil.copytree(child, dest, dirs_exist_ok=True)

    return manifest_path, wrote_timeseries


def main(argv: list[str] | None = None) -> int:
    """Parse arguments and run a scan; used as setuptools console script."""
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
        "--no-hdf5-metadata",
        action="store_true",
        help="Do not open HDF5 files; skip declination and pointing timeseries",
    )
    parser.add_argument(
        "--pointing-timeseries",
        action="store_true",
        help="Emit pointing_timeseries.json (per-file RA/Dec, mid-time); requires HDF5 metadata",
    )
    parser.add_argument(
        "--pointing-timeseries-max-files",
        type=int,
        default=5000,
        metavar="N",
        help="Cap rows in pointing_timeseries.json (default: 5000)",
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

    hdf5_metadata = not args.no_hdf5_metadata
    pointing_ts = bool(args.pointing_timeseries) and hdf5_metadata
    if args.pointing_timeseries and not hdf5_metadata:
        print(
            "warning: --pointing-timeseries ignored because --no-hdf5-metadata was set",
            file=sys.stderr,
        )

    _, wrote_timeseries = build_out(
        root=root,
        out_dir=args.out,
        no_stat=args.no_stat,
        site_dir=args.site,
        hdf5_metadata=hdf5_metadata,
        pointing_timeseries=pointing_ts,
        pointing_timeseries_max_files=max(1, args.pointing_timeseries_max_files),
    )
    print(f"Wrote {args.out / 'manifest.json'}")
    if pointing_ts and wrote_timeseries:
        print(f"Wrote {args.out / 'pointing_timeseries.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
