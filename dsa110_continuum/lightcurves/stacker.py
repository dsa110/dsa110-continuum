"""Stack per-epoch forced-photometry CSVs into a cross-epoch light curve table.

This module is a clean, importable extraction of the logic in
``scripts/stack_lightcurves.py`` with the following improvements:

* Type-annotated public API
* ``parse_epoch_utc`` handles edge cases (no match raises ValueError with detail)
* ``assign_source_ids`` prefers ``source_name`` column (O(N)) and falls back to
  SkyCoord cross-matching (O(N²))
* ``stack_csvs`` accepts path-like objects as well as strings
* All helpers are independently unit-testable

Reference
---------
Mooley et al. (2016), ApJ 818, 105
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

try:
    from astropy.coordinates import SkyCoord
    import astropy.units as u

    _HAS_ASTROPY = True
except ImportError:  # pragma: no cover
    _HAS_ASTROPY = False


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def parse_epoch_utc(filename: str | Path) -> str:
    """Extract an ISO-8601 UTC epoch string from a forced-phot CSV filename.

    Accepted formats
    ----------------
    * ``2026-02-12T0000_forced_phot.csv``  →  ``2026-02-12T00:00:00``
    * ``2026-01-25T2200_forced_phot.csv``  →  ``2026-01-25T22:00:00``
    * ``2026-01-25T22:00:00_forced_phot.csv``  →  ``2026-01-25T22:00:00``

    Parameters
    ----------
    filename : str or Path
        Filename (or full path) to parse.

    Returns
    -------
    str
        ISO-8601 UTC timestamp with seconds precision.

    Raises
    ------
    ValueError
        If the filename does not contain a recognisable epoch pattern.
    """
    fname = Path(filename).name

    # Pattern 1: ISO date + compact HHMM e.g. 2026-01-25T2200
    m = re.search(r"(\d{4}-\d{2}-\d{2})T(\d{2})(\d{2})(?:[^:\d]|$)", fname)
    if m:
        date, hh, mm = m.group(1), m.group(2), m.group(3)
        return f"{date}T{hh}:{mm}:00"

    # Pattern 2: full ISO with colons e.g. 2026-01-25T22:00:00
    m2 = re.search(r"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})", fname)
    if m2:
        return m2.group(1)

    raise ValueError(
        f"Cannot parse epoch UTC from filename: {fname!r}. "
        "Expected pattern like '2026-01-25T2200_forced_phot.csv'."
    )


def assign_source_ids(df: pd.DataFrame, match_arcsec: float = 5.0) -> pd.DataFrame:
    """Assign a stable integer ``source_id`` to each row by clustering positions.

    Strategy
    --------
    1. If the DataFrame has a ``source_name`` column, use ``pd.factorize``
       for O(N) exact grouping.
    2. Otherwise fall back to O(N²) SkyCoord cross-matching (legacy CSVs).

    Parameters
    ----------
    df : DataFrame
        Must have columns ``ra_deg`` and ``dec_deg`` (or ``source_name``).
    match_arcsec : float
        Position match radius in arcseconds for the fallback algorithm.

    Returns
    -------
    DataFrame
        Copy of *df* with a new ``source_id`` integer column.

    Raises
    ------
    ValueError
        If required position columns are absent and ``source_name`` is not present.
    """
    df = df.copy()

    if "source_name" in df.columns:
        codes, _ = pd.factorize(df["source_name"])
        df["source_id"] = codes
        return df

    # Fallback: coordinate cross-matching
    if not {"ra_deg", "dec_deg"}.issubset(df.columns):
        raise ValueError("DataFrame must have 'source_name' OR both 'ra_deg' and 'dec_deg'.")

    if not _HAS_ASTROPY:  # pragma: no cover
        raise ImportError("astropy is required for coordinate-based source matching.")

    coords = SkyCoord(
        ra=df["ra_deg"].values * u.deg,
        dec=df["dec_deg"].values * u.deg,
    )
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

    df["source_id"] = source_ids
    return df


def stack_csvs(
    csv_paths: Sequence[str | Path],
    match_arcsec: float = 5.0,
) -> pd.DataFrame:
    """Read all forced-photometry CSVs and return a stacked cross-epoch DataFrame.

    Each CSV is expected to contain per-source rows for a single epoch.  The
    epoch timestamp is inferred from the filename via :func:`parse_epoch_utc`.

    Parameters
    ----------
    csv_paths : sequence of str or Path
        Paths to forced-photometry CSV files.
    match_arcsec : float
        Position match radius forwarded to :func:`assign_source_ids`.

    Returns
    -------
    DataFrame
        Stacked DataFrame with added columns:
        * ``epoch_utc`` — ISO-8601 UTC timestamp string
        * ``date``      — ``YYYY-MM-DD`` date string extracted from filename
        * ``source_id`` — stable integer source identifier

    Raises
    ------
    ValueError
        If *csv_paths* is empty.
    """
    if not csv_paths:
        raise ValueError("csv_paths must not be empty.")

    frames: list[pd.DataFrame] = []
    for path in csv_paths:
        df = pd.read_csv(path)
        fname = Path(path).name
        df["epoch_utc"] = parse_epoch_utc(fname)
        df["date"] = fname[:10]
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    combined = assign_source_ids(combined, match_arcsec=match_arcsec)
    return combined
