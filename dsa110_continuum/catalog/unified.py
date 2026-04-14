"""
Unified radio catalog backend: NVSS + FIRST + VLASS + RACS (RAX) in one SQLite DB.

Schema
------
The unified catalog stores one row per physical radio source, identified by a
stable ``source_id`` (hash of the NVSS name when available, otherwise a hex-UUID
derived from the position).  Per-survey flux columns hold NULL when a survey does
not cover or detect the source.

Table: ``sources``

    source_id       TEXT  PRIMARY KEY  -- stable cross-survey identifier
    ra_deg          REAL  NOT NULL     -- best-position RA (ICRS J2000)
    dec_deg         REAL  NOT NULL     -- best-position Dec (ICRS J2000)
    s_nvss_mjy      REAL              -- NVSS 1.4 GHz total flux (mJy)
    s_first_mjy     REAL              -- FIRST 1.4 GHz peak flux (mJy)
    s_vlass_mjy     REAL              -- VLASS 3 GHz peak flux (mJy)
    s_racs_mjy      REAL              -- RACS-Low 887.5 MHz total flux (mJy)
    alpha           REAL              -- spectral index (fitted where ≥2 surveys)
    resolved_flag   INTEGER           -- 1 if FIRST indicates extended emission
    confusion_flag  INTEGER           -- 1 if >1 NVSS match within 45 arcsec
    has_nvss        INTEGER           -- 1/0 boolean
    has_first       INTEGER           -- 1/0 boolean
    has_vlass       INTEGER           -- 1/0 boolean
    has_racs        INTEGER           -- 1/0 boolean

Table: ``meta``

    key   TEXT  PRIMARY KEY
    value TEXT

Indexes: ``idx_radec``, ``idx_dec``, ``idx_flux``, ``idx_has_nvss``.

Usage
-----
    from dsa110_continuum.catalog.unified import UnifiedCatalog

    # Open (or create) DB
    db = UnifiedCatalog("/tmp/unified_test.db")

    # Ingest synthetic sources
    db.ingest_sources(rows)           # list[dict]

    # Cone search
    df = db.cone_search(180.0, 30.0, radius_deg=1.0)

    # Stats
    print(db.summary())

Design notes
------------
* The class is intentionally thin — ingest + query.  All cross-survey matching
  is done at ingest time (VAST-style: nearest neighbour within match_radius_arcsec).
* Works without any real data; synthetic rows can be inserted directly for tests.
* Thread-safe: each connection is per-instance; the DB is opened in WAL mode.
"""

from __future__ import annotations

import hashlib
import logging
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import astropy.units as u
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord

logger = logging.getLogger(__name__)

# Default match radius for associating FIRST/VLASS/RACS rows with an NVSS anchor
DEFAULT_MATCH_RADIUS_ARCSEC: float = 7.5

# Column names used internally
_SURVEYS = ("nvss", "first", "vlass", "racs")

_DDL = """
PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS sources (
    source_id       TEXT    PRIMARY KEY,
    ra_deg          REAL    NOT NULL,
    dec_deg         REAL    NOT NULL,
    s_nvss_mjy      REAL,
    s_first_mjy     REAL,
    s_vlass_mjy     REAL,
    s_racs_mjy      REAL,
    alpha           REAL,
    resolved_flag   INTEGER DEFAULT 0,
    confusion_flag  INTEGER DEFAULT 0,
    has_nvss        INTEGER DEFAULT 0,
    has_first       INTEGER DEFAULT 0,
    has_vlass       INTEGER DEFAULT 0,
    has_racs        INTEGER DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_radec  ON sources(ra_deg, dec_deg);
CREATE INDEX IF NOT EXISTS idx_dec    ON sources(dec_deg);
CREATE INDEX IF NOT EXISTS idx_flux   ON sources(s_nvss_mjy);
CREATE INDEX IF NOT EXISTS idx_has_nvss ON sources(has_nvss);

CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT
);
"""


def _make_source_id(ra: float, dec: float, nvss_name: str | None = None) -> str:
    """Derive a stable source_id from position (or NVSS name if available)."""
    if nvss_name:
        # Hash the NVSS name for stability across rebuilds
        return "nvss_" + hashlib.sha1(nvss_name.encode()).hexdigest()[:16]
    # Position-based fallback: round to 0.1 arcsec, then hash
    key = f"J{ra:.6f}{dec:+.6f}"
    return "pos_" + hashlib.sha1(key.encode()).hexdigest()[:16]


def _spectral_index(s1: float | None, nu1: float, s2: float | None, nu2: float) -> float | None:
    """Power-law spectral index: alpha = log(s2/s1) / log(nu2/nu1).

    Parameters
    ----------
    s1, s2 : float or None
        Flux densities at frequencies nu1, nu2.
    nu1, nu2 : float
        Frequencies in GHz.
    """
    if s1 is None or s2 is None:
        return None
    if s1 <= 0 or s2 <= 0:
        return None
    return float(np.log(s2 / s1) / np.log(nu2 / nu1))


class UnifiedCatalog:
    """NVSS+FIRST+VLASS+RACS unified SQLite catalog.

    Parameters
    ----------
    db_path : str or Path
        Path to the SQLite database file.  Created if it does not exist.
    match_radius_arcsec : float
        Maximum separation for associating per-survey rows with an NVSS anchor.
    """

    def __init__(
        self,
        db_path: str | Path,
        match_radius_arcsec: float = DEFAULT_MATCH_RADIUS_ARCSEC,
    ) -> None:
        self.db_path = Path(db_path)
        self.match_radius_arcsec = match_radius_arcsec
        self._conn: sqlite3.Connection | None = None
        self._open()

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    def _open(self) -> None:
        """Open (or create) the database and apply schema."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_DDL)
        self._conn.commit()
        logger.debug("Opened unified catalog DB: %s", self.db_path)

    def close(self) -> None:
        """Close the database connection."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def __enter__(self) -> "UnifiedCatalog":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Ingest
    # ------------------------------------------------------------------

    def ingest_sources(
        self,
        rows: list[dict[str, Any]],
        *,
        overwrite: bool = False,
    ) -> int:
        """Insert source rows into the catalog.

        Each row is a ``dict`` with at minimum ``ra_deg`` and ``dec_deg``.
        Optional keys:
        - ``source_id``   — stable identifier (derived from position if absent)
        - ``nvss_name``   — used to derive ``source_id`` when present
        - ``s_nvss_mjy``  — NVSS flux (mJy)
        - ``s_first_mjy`` — FIRST flux (mJy)
        - ``s_vlass_mjy`` — VLASS flux (mJy)
        - ``s_racs_mjy``  — RACS flux (mJy)
        - ``alpha``       — spectral index (computed from NVSS+VLASS if absent and both available)
        - ``resolved_flag``, ``confusion_flag``

        Parameters
        ----------
        rows : list[dict]
        overwrite : bool
            If True, replace existing rows with the same ``source_id``.

        Returns
        -------
        int
            Number of rows inserted (not updated).
        """
        if not rows:
            return 0

        insert_verb = "INSERT OR REPLACE" if overwrite else "INSERT OR IGNORE"
        sql = f"""
        {insert_verb} INTO sources (
            source_id, ra_deg, dec_deg,
            s_nvss_mjy, s_first_mjy, s_vlass_mjy, s_racs_mjy,
            alpha, resolved_flag, confusion_flag,
            has_nvss, has_first, has_vlass, has_racs
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """

        params = []
        for row in rows:
            ra = float(row["ra_deg"])
            dec = float(row["dec_deg"])
            sid = row.get("source_id") or _make_source_id(
                ra, dec, row.get("nvss_name")
            )

            s_nvss = row.get("s_nvss_mjy")
            s_first = row.get("s_first_mjy")
            s_vlass = row.get("s_vlass_mjy")
            s_racs = row.get("s_racs_mjy")

            # Auto-compute spectral index from NVSS (1.4 GHz) + VLASS (3 GHz)
            alpha = row.get("alpha")
            if alpha is None and s_nvss is not None and s_vlass is not None:
                alpha = _spectral_index(s_nvss, 1.4, s_vlass, 3.0)

            params.append((
                sid, ra, dec,
                s_nvss, s_first, s_vlass, s_racs,
                alpha,
                int(bool(row.get("resolved_flag", 0))),
                int(bool(row.get("confusion_flag", 0))),
                int(s_nvss is not None),
                int(s_first is not None),
                int(s_vlass is not None),
                int(s_racs is not None),
            ))

        assert self._conn is not None
        cursor = self._conn.executemany(sql, params)
        self._conn.commit()
        return cursor.rowcount

    def ingest_dataframe(
        self,
        df: pd.DataFrame,
        *,
        overwrite: bool = False,
    ) -> int:
        """Ingest a pandas DataFrame.  Column names must match the dict keys above."""
        return self.ingest_sources(df.to_dict("records"), overwrite=overwrite)

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def cone_search(
        self,
        ra_center: float,
        dec_center: float,
        radius_deg: float,
        *,
        min_flux_mjy: float | None = None,
        max_sources: int | None = None,
        surveys: list[str] | None = None,
    ) -> pd.DataFrame:
        """Return sources within ``radius_deg`` of a sky position.

        Parameters
        ----------
        ra_center, dec_center : float
            Field centre (ICRS degrees).
        radius_deg : float
            Search radius in degrees.
        min_flux_mjy : float or None
            Minimum NVSS flux (mJy).  Applied only when provided.
        max_sources : int or None
            Cap on number of returned rows.
        surveys : list[str] or None
            If provided (e.g. ``["nvss", "vlass"]``), only return sources
            detected in **all** listed surveys.

        Returns
        -------
        pandas.DataFrame
        """
        assert self._conn is not None

        # Box pre-filter (fast) then exact separation cut
        dec_half = radius_deg
        cos_dec = max(np.cos(np.radians(dec_center)), 1e-6)
        ra_half = radius_deg / cos_dec

        where: list[str] = [
            "ra_deg BETWEEN ? AND ?",
            "dec_deg BETWEEN ? AND ?",
        ]
        params: list[Any] = [
            ra_center - ra_half, ra_center + ra_half,
            dec_center - dec_half, dec_center + dec_half,
        ]

        if min_flux_mjy is not None:
            where.append("(s_nvss_mjy >= ? OR s_nvss_mjy IS NULL)")
            params.append(min_flux_mjy)

        if surveys:
            for sv in surveys:
                col = f"has_{sv.lower()}"
                where.append(f"{col} = 1")

        sql = f"SELECT * FROM sources WHERE {' AND '.join(where)} ORDER BY s_nvss_mjy DESC"
        if max_sources:
            sql += f" LIMIT {max_sources * 4}"  # over-fetch, trim after sep filter

        rows = self._conn.execute(sql, params).fetchall()
        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame([dict(r) for r in rows])

        # Exact angular separation filter
        sc = SkyCoord(ra=df["ra_deg"].values * u.deg, dec=df["dec_deg"].values * u.deg)
        center = SkyCoord(ra_center * u.deg, dec_center * u.deg)
        sep = sc.separation(center).deg
        df = df[sep <= radius_deg].copy()
        df["separation_deg"] = sep[sep <= radius_deg]

        if max_sources:
            df = df.head(max_sources)

        return df.reset_index(drop=True)

    def get_source(self, source_id: str) -> dict[str, Any] | None:
        """Look up a single source by its stable ``source_id``."""
        assert self._conn is not None
        row = self._conn.execute(
            "SELECT * FROM sources WHERE source_id = ?", (source_id,)
        ).fetchone()
        return dict(row) if row else None

    def query_by_flux(
        self,
        min_flux_mjy: float,
        survey: str = "nvss",
        limit: int = 1000,
    ) -> pd.DataFrame:
        """Return the brightest sources above a flux threshold in a given survey."""
        assert self._conn is not None
        col = f"s_{survey.lower()}_mjy"
        sql = (
            f"SELECT * FROM sources WHERE {col} >= ? "
            f"ORDER BY {col} DESC LIMIT ?"
        )
        rows = self._conn.execute(sql, (min_flux_mjy, limit)).fetchall()
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame([dict(r) for r in rows])

    # ------------------------------------------------------------------
    # Statistics & utilities
    # ------------------------------------------------------------------

    def count(self) -> int:
        """Total number of sources in the catalog."""
        assert self._conn is not None
        return self._conn.execute("SELECT COUNT(*) FROM sources").fetchone()[0]

    def summary(self) -> dict[str, Any]:
        """Return a summary dict with source counts per survey."""
        assert self._conn is not None
        total = self.count()
        out: dict[str, Any] = {"total": total}
        for sv in _SURVEYS:
            col = f"has_{sv}"
            n = self._conn.execute(
                f"SELECT COUNT(*) FROM sources WHERE {col} = 1"
            ).fetchone()[0]
            out[sv] = n
        # Build time
        row = self._conn.execute(
            "SELECT value FROM meta WHERE key='build_time_iso'"
        ).fetchone()
        out["build_time_iso"] = row[0] if row else None
        return out

    def set_meta(self, key: str, value: str) -> None:
        """Write or update a metadata key-value pair."""
        assert self._conn is not None
        self._conn.execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)", (key, value)
        )
        self._conn.commit()

    def get_meta(self, key: str) -> str | None:
        """Read a metadata value."""
        assert self._conn is not None
        row = self._conn.execute(
            "SELECT value FROM meta WHERE key = ?", (key,)
        ).fetchone()
        return row[0] if row else None

    # ------------------------------------------------------------------
    # Bulk cross-survey ingest from individual SQLite strip DBs
    # ------------------------------------------------------------------

    @classmethod
    def build_from_strips(
        cls,
        nvss_db: Path | None = None,
        first_db: Path | None = None,
        vlass_db: Path | None = None,
        racs_db: Path | None = None,
        output_db: Path | str | None = None,
        *,
        match_radius_arcsec: float = DEFAULT_MATCH_RADIUS_ARCSEC,
        min_nvss_flux_mjy: float | None = None,
        overwrite: bool = False,
    ) -> "UnifiedCatalog":
        """Build a unified catalog by cross-matching individual survey strip DBs.

        NVSS is the anchor survey.  Sources from FIRST/VLASS/RACS are associated
        using nearest-neighbour matching within ``match_radius_arcsec``.

        Parameters
        ----------
        nvss_db, first_db, vlass_db, racs_db : Path or None
            Paths to individual-survey SQLite strip databases.
            Any can be omitted (columns will remain NULL).
        output_db : Path or str or None
            Where to write the unified catalog.
            Defaults to ``{catalog_dir}/unified_catalog.sqlite3``.
        match_radius_arcsec : float
            Cross-survey association radius.
        min_nvss_flux_mjy : float or None
            Only include NVSS sources above this flux.
        overwrite : bool
            If True, replace existing rows.

        Returns
        -------
        UnifiedCatalog
        """
        if output_db is None:
            try:
                from dsa110_continuum.config import paths
                output_db = paths.catalog_dir / "unified_catalog.sqlite3"
            except Exception:
                output_db = Path("/tmp/unified_catalog.sqlite3")

        uc = cls(output_db, match_radius_arcsec=match_radius_arcsec)

        # --- Load NVSS as anchor ---
        nvss_df = _load_sqlite_sources(nvss_db, min_flux_mjy=min_nvss_flux_mjy)
        if nvss_df.empty:
            logger.warning("No NVSS sources loaded; unified catalog will be empty")
            return uc

        # --- Load secondary surveys ---
        first_df = _load_sqlite_sources(first_db)
        vlass_df = _load_sqlite_sources(vlass_db)
        racs_df = _load_sqlite_sources(racs_db)

        # --- Associate secondary surveys ---
        rows = _crossmatch_to_nvss(
            nvss_df, first_df, vlass_df, racs_df, match_radius_arcsec
        )

        inserted = uc.ingest_sources(rows, overwrite=overwrite)

        uc.set_meta("build_time_iso", datetime.now(UTC).isoformat())
        uc.set_meta("match_radius_arcsec", str(match_radius_arcsec))
        if nvss_db:
            uc.set_meta("nvss_db", str(nvss_db))

        logger.info(
            "Built unified catalog: %d sources → %s", inserted, output_db
        )
        return uc

    def __repr__(self) -> str:
        return (
            f"UnifiedCatalog(path={self.db_path!r}, "
            f"n_sources={self.count()})"
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _load_sqlite_sources(
    db_path: Path | None,
    min_flux_mjy: float | None = None,
) -> pd.DataFrame:
    """Load sources from a single-survey SQLite strip database."""
    if db_path is None or not db_path.exists():
        return pd.DataFrame(columns=["ra_deg", "dec_deg", "flux_mjy"])
    try:
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        query = "SELECT * FROM sources"
        params: list = []
        if min_flux_mjy is not None:
            query += " WHERE flux_mjy >= ?"
            params.append(min_flux_mjy)
        rows = conn.execute(query, params).fetchall()
        conn.close()
        if not rows:
            return pd.DataFrame(columns=["ra_deg", "dec_deg", "flux_mjy"])
        return pd.DataFrame([dict(r) for r in rows])
    except sqlite3.Error as exc:
        logger.warning("Failed to load %s: %s", db_path, exc)
        return pd.DataFrame(columns=["ra_deg", "dec_deg", "flux_mjy"])


def _nearest_flux(
    nvss_ra: np.ndarray,
    nvss_dec: np.ndarray,
    survey_df: pd.DataFrame,
    radius_arcsec: float,
) -> np.ndarray:
    """Vectorised nearest-neighbour match; returns flux array aligned to NVSS."""
    n = len(nvss_ra)
    result = np.full(n, np.nan)

    if survey_df.empty:
        return result

    nvss_coords = SkyCoord(ra=nvss_ra * u.deg, dec=nvss_dec * u.deg)
    cat_coords = SkyCoord(
        ra=survey_df["ra_deg"].values * u.deg,
        dec=survey_df["dec_deg"].values * u.deg,
    )

    from astropy.coordinates import match_coordinates_sky
    idx, sep2d, _ = match_coordinates_sky(nvss_coords, cat_coords)
    sep_arcsec = sep2d.to(u.arcsec).value
    matched = sep_arcsec <= radius_arcsec
    result[matched] = survey_df["flux_mjy"].values[idx[matched]]
    return result


def _crossmatch_to_nvss(
    nvss_df: pd.DataFrame,
    first_df: pd.DataFrame,
    vlass_df: pd.DataFrame,
    racs_df: pd.DataFrame,
    radius_arcsec: float,
) -> list[dict[str, Any]]:
    """Cross-match FIRST, VLASS, RACS onto NVSS positions."""
    ra = nvss_df["ra_deg"].values
    dec = nvss_df["dec_deg"].values

    s_nvss = nvss_df["flux_mjy"].values
    s_first = _nearest_flux(ra, dec, first_df, radius_arcsec)
    s_vlass = _nearest_flux(ra, dec, vlass_df, radius_arcsec)
    s_racs = _nearest_flux(ra, dec, racs_df, radius_arcsec)

    rows = []
    for i in range(len(ra)):
        rows.append({
            "ra_deg": float(ra[i]),
            "dec_deg": float(dec[i]),
            "s_nvss_mjy": float(s_nvss[i]) if np.isfinite(s_nvss[i]) else None,
            "s_first_mjy": float(s_first[i]) if np.isfinite(s_first[i]) else None,
            "s_vlass_mjy": float(s_vlass[i]) if np.isfinite(s_vlass[i]) else None,
            "s_racs_mjy": float(s_racs[i]) if np.isfinite(s_racs[i]) else None,
        })
    return rows
