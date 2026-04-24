"""Manifest schema v1 for DSA-110 incoming HDF5 inventory."""

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
    count: int = 0
    bytes: int = 0


@dataclass
class BeamAgg:
    count: int = 0
    bytes: int = 0


@dataclass
class ScanAccum:
    by_day: dict[date, DayAgg] = field(default_factory=dict)
    by_beam: dict[int, BeamAgg] = field(default_factory=dict)
    latest_filename_dt: datetime | None = None
    earliest_filename_dt: datetime | None = None
    latest_mtime: datetime | None = None
    earliest_mtime: datetime | None = None
    file_count: int = 0
    total_bytes: int = 0


def build_manifest(
    *,
    source_root: str,
    accum: ScanAccum,
    no_stat: bool,
    generated_at: datetime | None = None,
) -> dict[str, Any]:
    """Assemble the version-1 manifest dict from scan aggregates."""
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

    manifest: dict[str, Any] = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "generated_at": gen.isoformat().replace("+00:00", "Z"),
        "source_root": source_root,
        "options": {"no_stat": no_stat},
        "totals": {
            "file_count": accum.file_count,
            "total_bytes": accum.total_bytes if not no_stat else 0,
        },
        "by_day": by_day_rows,
        "by_beam": by_beam_rows,
        "gaps": gap_list,
        "freshness": freshness,
    }
    return manifest
