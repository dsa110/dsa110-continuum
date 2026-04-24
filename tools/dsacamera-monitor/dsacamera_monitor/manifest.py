"""Manifest schema for DSA-110 incoming HDF5 inventory (v2 with optional pointing)."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Any

from dsacamera_monitor import MANIFEST_SCHEMA_VERSION
from dsacamera_monitor.gaps import compute_gaps

FILENAME_PATTERN = (
    r"^(?P<ymd>\d{4}-\d{2}-\d{2})T(?P<hms>\d{2}:\d{2}:\d{2})_sb(?P<beam>\d+)\.hdf5$"
)
_COMPILED = re.compile(FILENAME_PATTERN)


def try_parse_filename(name: str) -> tuple[datetime, int] | None:
    """Parse DSA-110 incoming HDF5 name; return (UTC datetime from name, beam id) or None."""
    m = _COMPILED.match(name)
    if not m:
        return None
    ymd = m.group("ymd")
    hms = m.group("hms")
    beam = int(m.group("beam"))
    dt = datetime.strptime(f"{ymd}T{hms}", "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc)
    return dt, beam


@dataclass
class DayAgg:
    """Per-calendar-day aggregates (file sizes and optional Dec from HDF5)."""

    count: int = 0
    bytes: int = 0
    dec_min: float | None = None
    dec_max: float | None = None
    decs_rounded: set[float] = field(default_factory=set)


@dataclass
class BeamAgg:
    """Per-beam (subband) file aggregates."""

    count: int = 0
    bytes: int = 0


@dataclass
class ScanAccum:
    """Full scan result before JSON serialization."""

    by_day: dict[date, DayAgg] = field(default_factory=dict)
    by_beam: dict[int, BeamAgg] = field(default_factory=dict)
    latest_filename_dt: datetime | None = None
    earliest_filename_dt: datetime | None = None
    latest_mtime: datetime | None = None
    earliest_mtime: datetime | None = None
    file_count: int = 0
    total_bytes: int = 0
    files_with_dec: int = 0
    files_dec_missing: int = 0
    global_dec_min: float | None = None
    global_dec_max: float | None = None
    global_decs_rounded: set[float] = field(default_factory=set)
    timeseries_rows: list[dict[str, Any]] = field(default_factory=list)
    timeseries_truncated: bool = False


def build_manifest(
    *,
    source_root: str,
    accum: ScanAccum,
    no_stat: bool,
    hdf5_metadata: bool = False,
    pointing_timeseries: bool = False,
    generated_at: datetime | None = None,
) -> dict[str, Any]:
    """Assemble the manifest dict from scan aggregates (schema v2 when hdf5_metadata)."""
    gen = generated_at or datetime.now(timezone.utc)
    if gen.tzinfo is None:
        gen = gen.replace(tzinfo=timezone.utc)

    by_day_rows: list[dict[str, Any]] = []
    for d in sorted(accum.by_day):
        agg = accum.by_day[d]
        row: dict[str, Any] = {"date": d.isoformat(), "count": agg.count}
        if not no_stat:
            row["bytes"] = agg.bytes
        else:
            row["bytes"] = 0
        if hdf5_metadata and agg.decs_rounded:
            row["dec_deg_min"] = agg.dec_min
            row["dec_deg_max"] = agg.dec_max
            row["dec_unique_count"] = len(agg.decs_rounded)
        by_day_rows.append(row)

    days_with_files = set(accum.by_day.keys())
    if days_with_files:
        date_min = min(days_with_files)
        date_max = max(days_with_files)
        gap_list = compute_gaps(days_with_files, date_min, date_max)
    else:
        gap_list = []

    by_beam_rows: list[dict[str, Any]] = []
    for beam in sorted(accum.by_beam):
        beam_agg = accum.by_beam[beam]
        row_b: dict[str, Any] = {"beam": beam, "count": beam_agg.count}
        if not no_stat:
            row_b["bytes"] = beam_agg.bytes
        else:
            row_b["bytes"] = 0
        by_beam_rows.append(row_b)

    freshness: dict[str, Any] = {
        "latest_filename_timestamp_utc": (
            accum.latest_filename_dt.isoformat().replace("+00:00", "Z")
            if accum.latest_filename_dt
            else None
        ),
        "earliest_filename_timestamp_utc": (
            accum.earliest_filename_dt.isoformat().replace("+00:00", "Z")
            if accum.earliest_filename_dt
            else None
        ),
        "latest_mtime_utc": (
            accum.latest_mtime.isoformat().replace("+00:00", "Z")
            if accum.latest_mtime
            else None
        ),
        "earliest_mtime_utc": (
            accum.earliest_mtime.isoformat().replace("+00:00", "Z")
            if accum.earliest_mtime
            else None
        ),
    }

    options: dict[str, Any] = {
        "no_stat": no_stat,
        "hdf5_metadata": hdf5_metadata,
        "pointing_timeseries": pointing_timeseries,
    }

    manifest: dict[str, Any] = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "generated_at": gen.isoformat().replace("+00:00", "Z"),
        "source_root": source_root,
        "options": options,
        "totals": {
            "file_count": accum.file_count,
            "total_bytes": accum.total_bytes if not no_stat else 0,
        },
        "by_day": by_day_rows,
        "by_beam": by_beam_rows,
        "gaps": gap_list,
        "freshness": freshness,
    }
    if hdf5_metadata:
        manifest["pointing"] = {
            "dec_deg_min": accum.global_dec_min,
            "dec_deg_max": accum.global_dec_max,
            "dec_unique": sorted(accum.global_decs_rounded),
            "unique_strip_count": len(accum.global_decs_rounded),
            "files_with_dec": accum.files_with_dec,
            "files_dec_missing": accum.files_dec_missing,
            "sampled": False,
        }
    if pointing_timeseries and accum.timeseries_rows:
        manifest["pointing_timeseries"] = {
            "file": "pointing_timeseries.json",
            "row_count": len(accum.timeseries_rows),
            "truncated": accum.timeseries_truncated,
        }
    return manifest
