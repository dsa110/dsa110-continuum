"""Contiguous date gaps with zero files (between first and last day with data)."""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any


def compute_gaps(
    days_with_files: set[date],
    date_min: date,
    date_max: date,
) -> list[dict[str, Any]]:
    """Return list of gap ranges {start, end, days} where no files exist."""
    gaps: list[dict[str, Any]] = []
    gap_start: date | None = None
    d = date_min
    while d <= date_max:
        if d not in days_with_files:
            if gap_start is None:
                gap_start = d
        elif gap_start is not None:
            gap_end = d - timedelta(days=1)
            gaps.append(
                {
                    "start": gap_start.isoformat(),
                    "end": gap_end.isoformat(),
                    "days": (gap_end - gap_start).days + 1,
                }
            )
            gap_start = None
        d += timedelta(days=1)

    if gap_start is not None:
        gap_end = date_max
        gaps.append(
            {
                "start": gap_start.isoformat(),
                "end": gap_end.isoformat(),
                "days": (gap_end - gap_start).days + 1,
            }
        )

    return gaps


def gaps_from_by_day_rows(by_day: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build a gap list from manifest-style by_day rows (test helper)."""
    days_with_files: set[date] = set()
    for row in by_day:
        days_with_files.add(date.fromisoformat(row["date"]))
    if not days_with_files:
        return []
    return compute_gaps(days_with_files, min(days_with_files), max(days_with_files))
