"""
Upper-Limit Storage for Non-Detections
========================================

When a known source (from the reference catalog) is not detected in a given
epoch's image, the pipeline records an *upper limit* rather than simply
omitting that epoch from the light curve.  This is essential for variability
analysis: knowing that a source was *observed* but fell below N×σ on a given
day constrains the Mooley modulation index even in non-detected epochs.

Gap vs VAST
-----------
In the VAST pipeline every non-detection produces an explicit forced-photometry
row with ``is_upper_limit=True`` and ``flux_upper_limit = N × local_rms``.
The DSA-110 pipeline mirrors this convention, with two differences:

1. **Detection threshold**: DSA-110 uses SNR ≥ 5σ (VAST uses 5σ by default
   too, but operating on Selavy blobs; here we apply it to forced-peak SNR).
2. **Storage format**: upper limits are accumulated in a lightweight SQLite
   database (``upper_limits.db``) *alongside* the detected-source HDF5 table,
   so both can be joined on (``source_id``, ``epoch_mjd``) without touching
   the detection table schema.

Data model
----------
Each non-detection row stores:

* ``source_id``         — integer key into the reference catalog
* ``epoch_mjd``         — MJD mid-point of the observing tile
* ``ra_deg``, ``dec_deg`` — sky position (from catalog, not fitted)
* ``rms_jyb``           — local RMS at the forced-photometry position (Jy/beam)
* ``upper_limit_jyb``   — N × rms_jyb (the N-σ upper limit on peak flux)
* ``n_sigma``           — the threshold used (e.g. 5.0)
* ``forced_peak_jyb``   — the raw forced-peak value (may be negative — noise)
* ``image_path``        — FITS image used (for provenance)

Usage
-----
>>> from dsa110_continuum.photometry.upper_limits import (
...     UpperLimitRecord, UpperLimitStore,
...     forced_peak_to_upper_limit,
... )
>>> store = UpperLimitStore("/data/epoch_results/upper_limits.db")
>>> rec = UpperLimitRecord(
...     source_id=1042,
...     epoch_mjd=60340.5,
...     ra_deg=150.2,
...     dec_deg=30.1,
...     rms_jyb=0.003,
...     upper_limit_jyb=0.015,
...     n_sigma=5.0,
...     forced_peak_jyb=-0.001,
...     image_path="/data/images/20240115_tile03.fits",
... )
>>> store.add(rec)
>>> store.close()
"""
from __future__ import annotations

import sqlite3
import threading
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np


# ── Detection threshold defaults ──────────────────────────────────────────────

#: Default SNR below which a forced-photometry measurement is an upper limit.
DEFAULT_DETECTION_THRESHOLD_SIGMA: float = 5.0


# ── Record dataclass ──────────────────────────────────────────────────────────

@dataclass
class UpperLimitRecord:
    """A single non-detection upper limit.

    Attributes
    ----------
    source_id : int
        Stable integer identifier into the reference catalog (e.g. NVSS row
        index).
    epoch_mjd : float
        Modified Julian Date of the tile mid-point.
    ra_deg : float
        Source right ascension from catalog (degrees).
    dec_deg : float
        Source declination from catalog (degrees).
    rms_jyb : float
        Local RMS noise at the source position (Jy/beam).  Derived from the
        annulus-based RMS in ``measure_forced_peak``.
    upper_limit_jyb : float
        N-sigma upper limit on peak flux density (= n_sigma × rms_jyb).
        This is the value to use in variability calculations.
    n_sigma : float
        Detection threshold used to set the upper limit (default 5.0).
    forced_peak_jyb : float
        Raw forced-peak value at the source position.  May be negative
        (noise fluctuation).  Stored for diagnostic purposes.
    image_path : str
        Absolute path to the FITS image used.  Empty string if unavailable.
    """
    source_id: int
    epoch_mjd: float
    ra_deg: float
    dec_deg: float
    rms_jyb: float
    upper_limit_jyb: float
    n_sigma: float = DEFAULT_DETECTION_THRESHOLD_SIGMA
    forced_peak_jyb: float = float("nan")
    image_path: str = ""

    def __post_init__(self) -> None:
        if self.rms_jyb < 0:
            raise ValueError(f"rms_jyb must be non-negative, got {self.rms_jyb}")
        if self.upper_limit_jyb < 0:
            raise ValueError(
                f"upper_limit_jyb must be non-negative, got {self.upper_limit_jyb}"
            )
        if self.n_sigma <= 0:
            raise ValueError(f"n_sigma must be positive, got {self.n_sigma}")


# ── Conversion helper ─────────────────────────────────────────────────────────

def forced_peak_to_upper_limit(
    source_id: int,
    epoch_mjd: float,
    ra_deg: float,
    dec_deg: float,
    forced_peak_jyb: float,
    rms_jyb: float,
    image_path: str = "",
    n_sigma: float = DEFAULT_DETECTION_THRESHOLD_SIGMA,
) -> UpperLimitRecord | None:
    """Convert a below-threshold forced-peak measurement to an upper-limit record.

    Returns ``None`` if the source *is* detected (SNR ≥ n_sigma) or if the
    RMS is not finite.

    Parameters
    ----------
    source_id : int
        Reference catalog row index.
    epoch_mjd : float
        Tile mid-point (MJD).
    ra_deg, dec_deg : float
        Source sky position (degrees).
    forced_peak_jyb : float
        Raw forced-peak flux density (Jy/beam).  May be NaN or negative.
    rms_jyb : float
        Local noise RMS at the source position (Jy/beam).
    image_path : str
        Path to the FITS image (for provenance).
    n_sigma : float
        Detection threshold in units of RMS (default 5.0).

    Returns
    -------
    UpperLimitRecord or None
        ``None`` if the source is detected (``forced_peak_jyb >= n_sigma × rms_jyb``).
        An :class:`UpperLimitRecord` otherwise.

    Examples
    --------
    >>> rec = forced_peak_to_upper_limit(
    ...     source_id=7, epoch_mjd=60340.5, ra_deg=150.0, dec_deg=30.0,
    ...     forced_peak_jyb=0.002, rms_jyb=0.003, n_sigma=5.0,
    ... )
    >>> rec is not None          # SNR = 0.002/0.003 = 0.67 < 5 → upper limit
    True
    >>> rec.upper_limit_jyb      # 5 × 0.003
    0.015
    """
    if not np.isfinite(rms_jyb) or rms_jyb <= 0:
        return None

    threshold = n_sigma * rms_jyb
    # Detected if the *positive* forced peak exceeds the threshold
    if np.isfinite(forced_peak_jyb) and forced_peak_jyb >= threshold:
        return None  # Source is detected — caller handles it normally

    return UpperLimitRecord(
        source_id=source_id,
        epoch_mjd=epoch_mjd,
        ra_deg=ra_deg,
        dec_deg=dec_deg,
        rms_jyb=rms_jyb,
        upper_limit_jyb=threshold,
        n_sigma=n_sigma,
        forced_peak_jyb=forced_peak_jyb if np.isfinite(forced_peak_jyb) else float("nan"),
        image_path=image_path,
    )


# ── SQLite store ──────────────────────────────────────────────────────────────

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS upper_limits (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    source_id        INTEGER NOT NULL,
    epoch_mjd        REAL    NOT NULL,
    ra_deg           REAL    NOT NULL,
    dec_deg          REAL    NOT NULL,
    rms_jyb          REAL    NOT NULL,
    upper_limit_jyb  REAL    NOT NULL,
    n_sigma          REAL    NOT NULL,
    forced_peak_jyb  REAL,
    image_path       TEXT    NOT NULL DEFAULT '',
    created_at       TEXT    DEFAULT (datetime('now')),
    UNIQUE (source_id, epoch_mjd)
);
CREATE INDEX IF NOT EXISTS idx_source ON upper_limits (source_id);
CREATE INDEX IF NOT EXISTS idx_epoch  ON upper_limits (epoch_mjd);
"""

_INSERT_SQL = """
INSERT OR REPLACE INTO upper_limits
    (source_id, epoch_mjd, ra_deg, dec_deg, rms_jyb,
     upper_limit_jyb, n_sigma, forced_peak_jyb, image_path)
VALUES
    (:source_id, :epoch_mjd, :ra_deg, :dec_deg, :rms_jyb,
     :upper_limit_jyb, :n_sigma, :forced_peak_jyb, :image_path)
"""


class UpperLimitStore:
    """Thread-safe SQLite store for non-detection upper limits.

    Uses ``INSERT OR REPLACE`` so re-running the pipeline over the same
    (source_id, epoch_mjd) pair is idempotent — the latest measurement wins.

    Parameters
    ----------
    db_path : str or Path
        Path to the SQLite database file.  Created if it does not exist.
        Pass ``:memory:`` for an in-memory database (useful for tests).

    Examples
    --------
    >>> store = UpperLimitStore(":memory:")
    >>> rec = UpperLimitRecord(
    ...     source_id=1, epoch_mjd=60000.0,
    ...     ra_deg=180.0, dec_deg=0.0,
    ...     rms_jyb=0.003, upper_limit_jyb=0.015,
    ... )
    >>> store.add(rec)
    >>> store.count()
    1
    >>> store.close()
    """

    def __init__(self, db_path: str | Path = ":memory:") -> None:
        self._path = str(db_path)
        self._local = threading.local()  # one connection per thread
        self._connect()

    # ── Connection management ─────────────────────────────────────────────────

    def _connect(self) -> None:
        conn = sqlite3.connect(self._path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.executescript(_CREATE_TABLE_SQL)
        conn.commit()
        self._local.conn = conn

    @property
    def _conn(self) -> sqlite3.Connection:
        if not hasattr(self._local, "conn"):
            self._connect()
        return self._local.conn

    def close(self) -> None:
        """Commit and close the underlying connection."""
        if hasattr(self._local, "conn"):
            self._local.conn.commit()
            self._local.conn.close()
            del self._local.conn

    def __enter__(self) -> "UpperLimitStore":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    # ── Write ──────────────────────────────────────────────────────────────────

    def add(self, record: UpperLimitRecord) -> None:
        """Insert or replace a single upper-limit record."""
        row = {f.name: getattr(record, f.name) for f in fields(record)}
        self._conn.execute(_INSERT_SQL, row)
        self._conn.commit()

    def add_many(self, records: Sequence[UpperLimitRecord]) -> int:
        """Insert or replace multiple records in a single transaction.

        Returns the number of records written.
        """
        if not records:
            return 0
        rows = [{f.name: getattr(r, f.name) for f in fields(r)} for r in records]
        self._conn.executemany(_INSERT_SQL, rows)
        self._conn.commit()
        return len(rows)

    # ── Read ───────────────────────────────────────────────────────────────────

    def count(self) -> int:
        """Return total number of stored upper limits."""
        row = self._conn.execute("SELECT COUNT(*) FROM upper_limits").fetchone()
        return int(row[0])

    def get_by_source(self, source_id: int) -> list[UpperLimitRecord]:
        """Return all upper limits for a given source, ordered by epoch."""
        rows = self._conn.execute(
            "SELECT * FROM upper_limits WHERE source_id=? ORDER BY epoch_mjd",
            (source_id,),
        ).fetchall()
        return [_row_to_record(r) for r in rows]

    def get_by_epoch(self, epoch_mjd: float, tol_days: float = 1e-4) -> list[UpperLimitRecord]:
        """Return all upper limits within *tol_days* of *epoch_mjd*."""
        rows = self._conn.execute(
            "SELECT * FROM upper_limits "
            "WHERE ABS(epoch_mjd - ?) < ? ORDER BY source_id",
            (epoch_mjd, tol_days),
        ).fetchall()
        return [_row_to_record(r) for r in rows]

    def get_epoch_range(
        self,
        source_id: int,
        mjd_min: float,
        mjd_max: float,
    ) -> list[UpperLimitRecord]:
        """Return upper limits for *source_id* between *mjd_min* and *mjd_max*."""
        rows = self._conn.execute(
            "SELECT * FROM upper_limits "
            "WHERE source_id=? AND epoch_mjd>=? AND epoch_mjd<=? "
            "ORDER BY epoch_mjd",
            (source_id, mjd_min, mjd_max),
        ).fetchall()
        return [_row_to_record(r) for r in rows]

    def iter_all(self) -> Iterator[UpperLimitRecord]:
        """Iterate over all records (memory-efficient, uses cursor)."""
        cur = self._conn.execute(
            "SELECT * FROM upper_limits ORDER BY source_id, epoch_mjd"
        )
        for row in cur:
            yield _row_to_record(row)

    def sources_with_upper_limits(self) -> list[int]:
        """Return sorted list of source_ids that have at least one upper limit."""
        rows = self._conn.execute(
            "SELECT DISTINCT source_id FROM upper_limits ORDER BY source_id"
        ).fetchall()
        return [int(r[0]) for r in rows]

    def delete_by_source(self, source_id: int) -> int:
        """Delete all upper limits for *source_id*. Returns rows deleted."""
        cur = self._conn.execute(
            "DELETE FROM upper_limits WHERE source_id=?", (source_id,)
        )
        self._conn.commit()
        return cur.rowcount

    def delete_by_epoch(self, epoch_mjd: float, tol_days: float = 1e-4) -> int:
        """Delete all upper limits within *tol_days* of *epoch_mjd*. Returns rows deleted."""
        cur = self._conn.execute(
            "DELETE FROM upper_limits WHERE ABS(epoch_mjd - ?) < ?",
            (epoch_mjd, tol_days),
        )
        self._conn.commit()
        return cur.rowcount

    # ── Numpy convenience ──────────────────────────────────────────────────────

    def upper_limits_array(self, source_id: int) -> np.ndarray:
        """Return upper-limit values (Jy/beam) for *source_id* as a numpy array.

        Returns an empty array if the source has no upper limits.
        """
        rows = self.get_by_source(source_id)
        if not rows:
            return np.array([], dtype=float)
        return np.array([r.upper_limit_jyb for r in rows], dtype=float)

    def epochs_array(self, source_id: int) -> np.ndarray:
        """Return epoch MJDs for *source_id* non-detections as a numpy array."""
        rows = self.get_by_source(source_id)
        if not rows:
            return np.array([], dtype=float)
        return np.array([r.epoch_mjd for r in rows], dtype=float)


# ── Private helpers ───────────────────────────────────────────────────────────

def _row_to_record(row: sqlite3.Row) -> UpperLimitRecord:
    """Convert a sqlite3.Row to an UpperLimitRecord."""
    return UpperLimitRecord(
        source_id=int(row["source_id"]),
        epoch_mjd=float(row["epoch_mjd"]),
        ra_deg=float(row["ra_deg"]),
        dec_deg=float(row["dec_deg"]),
        rms_jyb=float(row["rms_jyb"]),
        upper_limit_jyb=float(row["upper_limit_jyb"]),
        n_sigma=float(row["n_sigma"]),
        forced_peak_jyb=float(row["forced_peak_jyb"])
        if row["forced_peak_jyb"] is not None
        else float("nan"),
        image_path=str(row["image_path"]) if row["image_path"] else "",
    )
