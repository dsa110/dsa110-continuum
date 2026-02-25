#!/usr/bin/env python
# pylint: disable=no-member  # astropy.units uses dynamic attributes (deg, arcsec, etc.)
"""
Build a master reference catalog by crossmatching NVSS with VLASS and FIRST.

Outputs an SQLite DB at state/catalogs/master_sources.sqlite3 (by default)
containing one row per NVSS source with optional VLASS/FIRST matches and
derived spectral index and compactness/confusion flags.

Usage examples:

  python -m dsa110_contimg.core.catalog.build_master \
      --nvss /data/catalogs/NVSS.csv \
      --vlass /data/catalogs/VLASS.csv \
      --first /data/catalogs/FIRST.csv \
      --out state/catalogs/master_sources.sqlite3 \
      --match-radius-arcsec 7.5 \
      --export-view final_references --export-csv state/catalogs/final_refs.csv

Notes
-----
- This tool is intentionally tolerant of column naming. It attempts to map
  common column names for RA/Dec/flux/SNR in each survey. If your files use
  different names, you can provide explicit mappings via --map-<cat>-<field>.
- Input formats: CSV/TSV (auto-delimited) or FITS (via astropy.table).
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import math
import os
import sqlite3
import sys
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import astropy.units as u  # pylint: disable=no-member
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy.table import Table
from dsa110_contimg.common.utils import get_env_path

logger = logging.getLogger(__name__)


# ----------------------------- IO helpers ---------------------------------


FITS_SUFFIXES = (".fits", ".fit", ".fz", ".fits.gz", ".fit.gz")


def _read_table(path: str) -> pd.DataFrame:
    """Read a catalog (CSV/TSV/FITS) into a pandas DataFrame.

    Delimiter is auto-detected for text. FITS loaded via astropy.table.
    """
    path = os.fspath(path)
    lower = path.lower()
    if lower.endswith(FITS_SUFFIXES):
        t = Table.read(path)
        return t.to_pandas()
    # text
    return pd.read_csv(path, sep=None, engine="python")


def _normalize_columns(df: pd.DataFrame, mapping: dict[str, Iterable[str]]) -> dict[str, str]:
    """Return a mapping of canonical->actual column names found in df.

    mapping: {canonical: [candidate1, candidate2, ...]}
    """
    result: dict[str, str] = {}
    cols = {c.lower(): c for c in df.columns}
    for canon, cands in mapping.items():
        chosen: str | None = None
        for cand in cands:
            key = cand.lower()
            if key in cols:
                chosen = cols[key]
                break
        if chosen is not None:
            result[canon] = chosen
    return result


def _skycoord_from_df(df: pd.DataFrame, ra_col: str, dec_col: str) -> SkyCoord:
    return SkyCoord(ra=df[ra_col].values * u.deg, dec=df[dec_col].values * u.deg, frame="icrs")


# ---------------------------- Crossmatching --------------------------------


@dataclass
class SourceRow:
    ra_deg: float
    dec_deg: float
    s_nvss: float | None
    snr_nvss: float | None
    s_vlass: float | None
    alpha: float | None
    resolved_flag: int
    confusion_flag: int


def _compute_alpha(
    s1: float | None, nu1_hz: float, s2: float | None, nu2_hz: float
) -> float | None:
    if s1 is None or s2 is None:
        return None
    if s1 <= 0 or s2 <= 0:
        return None
    try:
        return float(math.log(s2 / s1) / math.log(nu2_hz / nu1_hz))
    except (ValueError, ZeroDivisionError, OverflowError):
        return None


def _crossmatch(
    df_nvss: pd.DataFrame,
    df_vlass: pd.DataFrame | None,
    df_first: pd.DataFrame | None,
    *,
    maps: dict[str, dict[str, str]],
    match_radius_arcsec: float,
    scale_nvss_to_jy: float,
    scale_vlass_to_jy: float,
) -> pd.DataFrame:
    """Crossmatch NVSS with optional VLASS and FIRST; compute alpha/flags.

    Returns a DataFrame with canonical columns ready to write to SQLite.
    """
    # Canonical column names we will emit
    out_rows: list[SourceRow] = []

    # Build SkyCoord for NVSS
    n_ra = maps["nvss"]["ra"]
    n_dec = maps["nvss"]["dec"]
    nvss_sc = _skycoord_from_df(df_nvss, n_ra, n_dec)

    # Flux/SNR columns (optional)
    n_flux = maps["nvss"].get("flux")
    n_snr = maps["nvss"].get("snr")

    # Prepare VLASS coord and flux if provided
    vlass_sc = None
    v_flux_col = None
    if (
        df_vlass is not None
        and "vlass" in maps
        and "ra" in maps["vlass"]
        and "dec" in maps["vlass"]
    ):
        vlass_sc = _skycoord_from_df(df_vlass, maps["vlass"]["ra"], maps["vlass"]["dec"])
        v_flux_col = maps["vlass"].get("flux")

    # Prepare FIRST coord and morphology if provided
    first_sc = None
    f_maj = f_min = None
    if (
        df_first is not None
        and "first" in maps
        and "ra" in maps["first"]
        and "dec" in maps["first"]
    ):
        first_sc = _skycoord_from_df(df_first, maps["first"]["ra"], maps["first"]["dec"])
        f_maj = maps["first"].get("maj")
        f_min = maps["first"].get("min")

    radius = match_radius_arcsec * u.arcsec

    # Pre-index matches for VLASS and FIRST using astropy search_around_sky
    v_idx_by_n: dict[int, list[int]] = {}
    if vlass_sc is not None:
        # NOTE: astropy returns (idx_other, idx_self)
        idx_v, idx_nv, _, _ = nvss_sc.search_around_sky(vlass_sc, radius)
        # For each match, map NVSS index -> list of VLASS indices within radius
        for i_v, i_n in zip(idx_v, idx_nv):
            v_idx_by_n.setdefault(int(i_n), []).append(int(i_v))

    f_idx_by_n: dict[int, list[int]] = {}
    if first_sc is not None:
        # NOTE: astropy returns (idx_other, idx_self)
        idx_f, idx_nv, _, _ = nvss_sc.search_around_sky(first_sc, radius)
        for i_f, i_n in zip(idx_f, idx_nv):
            f_idx_by_n.setdefault(int(i_n), []).append(int(i_f))

    # Iterate NVSS rows and assemble outputs
    for i in range(len(df_nvss)):
        ra = float(df_nvss.at[i, n_ra])
        dec = float(df_nvss.at[i, n_dec])
        s_nv = None
        snr_nv = None
        if n_flux and n_flux in df_nvss.columns:
            try:
                s_nv = float(df_nvss.at[i, n_flux]) * float(scale_nvss_to_jy)
            except (ValueError, TypeError, KeyError):
                s_nv = None
        if n_snr and n_snr in df_nvss.columns:
            try:
                snr_nv = float(df_nvss.at[i, n_snr])
            except (ValueError, TypeError, KeyError):
                snr_nv = None

        # VLASS match: choose single best (closest) if multiple; flag confusion if >1
        s_vl = None
        confusion = 0
        if vlass_sc is not None:
            cand = v_idx_by_n.get(i, [])
            if len(cand) > 1:
                confusion = 1
            if len(cand) >= 1:
                # pick closest by angular sep
                seps = SkyCoord(ra=ra * u.deg, dec=dec * u.deg).separation(vlass_sc[cand])
                j = int(cand[int(np.argmin(seps.to_value(u.arcsec)))])
                if v_flux_col and v_flux_col in df_vlass.columns:
                    try:
                        s_vl = float(df_vlass.at[j, v_flux_col]) * float(scale_vlass_to_jy)
                    except (ValueError, TypeError, KeyError):
                        s_vl = None

        # FIRST compactness: treat as resolved if deconvolved major/minor above thresholds
        resolved = 0
        if first_sc is not None:
            cand = f_idx_by_n.get(i, [])
            if len(cand) > 1:
                confusion = 1
            if len(cand) >= 1 and (f_maj or f_min):
                seps = SkyCoord(ra=ra * u.deg, dec=dec * u.deg).separation(first_sc[cand])
                j = int(cand[int(np.argmin(seps.to_value(u.arcsec)))])
                maj = None
                mn = None
                try:
                    if f_maj and f_maj in df_first.columns:
                        maj = float(df_first.at[j, f_maj])
                    if f_min and f_min in df_first.columns:
                        mn = float(df_first.at[j, f_min])
                except (ValueError, TypeError, KeyError):
                    maj = None
                    mn = None
                # Heuristic: resolved if either axis > 6 arcsec (FIRST beam ~5")
                if (maj is not None and maj > 6.0) or (mn is not None and mn > 6.0):
                    resolved = 1

        alpha = _compute_alpha(s_nv, 1.4e9, s_vl, 3.0e9)

        out_rows.append(
            SourceRow(
                ra_deg=ra,
                dec_deg=dec,
                s_nvss=s_nv,
                snr_nvss=snr_nv,
                s_vlass=s_vl,
                alpha=alpha,
                resolved_flag=int(resolved),
                confusion_flag=int(confusion),
            )
        )

    # Assemble output DataFrame
    out = pd.DataFrame(
        {
            "ra_deg": [r.ra_deg for r in out_rows],
            "dec_deg": [r.dec_deg for r in out_rows],
            "s_nvss": [r.s_nvss for r in out_rows],
            "snr_nvss": [r.snr_nvss for r in out_rows],
            "s_vlass": [r.s_vlass for r in out_rows],
            "alpha": [r.alpha for r in out_rows],
            "resolved_flag": [r.resolved_flag for r in out_rows],
            "confusion_flag": [r.confusion_flag for r in out_rows],
        }
    )
    # Assign source_id monotonically (NVSS row index surrogate)
    out.insert(0, "source_id", np.arange(len(out), dtype=int))
    return out


# ---------------------------- DB persistence -------------------------------


def _ensure_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _hash_file(path: Path) -> str:
    """Return SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_sqlite(
    out: pd.DataFrame,
    db_path: Path,
    *,
    goodref_snr_min: float = 50.0,
    goodref_alpha_min: float = -1.2,
    goodref_alpha_max: float = 0.2,
    finalref_snr_min: float = 80.0,
    finalref_ids: Iterable[int] | None = None,
    materialize_final: bool = False,
    meta_extra: dict[str, str] | None = None,
) -> None:
    _ensure_dir(db_path)
    with sqlite3.connect(os.fspath(db_path)) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS sources (
                source_id INTEGER PRIMARY KEY,
                ra_deg REAL NOT NULL,
                dec_deg REAL NOT NULL,
                s_nvss REAL,
                snr_nvss REAL,
                s_vlass REAL,
                alpha REAL,
                resolved_flag INTEGER NOT NULL,
                confusion_flag INTEGER NOT NULL
            )
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_sources_radec ON sources(ra_deg, dec_deg)")
        # Overwrite (replace on conflict by recreating contents)
        conn.execute("DELETE FROM sources")
        out.to_sql("sources", conn, if_exists="append", index=False)
        # meta table for provenance
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS meta (
                key TEXT PRIMARY KEY,
                value TEXT
            )
            """
        )
        # Persist thresholds and provenance in meta
        conn.execute(
            "INSERT OR REPLACE INTO meta(key, value) VALUES('goodref_snr_min', ?)",
            (str(goodref_snr_min),),
        )
        conn.execute(
            "INSERT OR REPLACE INTO meta(key, value) VALUES('goodref_alpha_min', ?)",
            (str(goodref_alpha_min),),
        )
        conn.execute(
            "INSERT OR REPLACE INTO meta(key, value) VALUES('goodref_alpha_max', ?)",
            (str(goodref_alpha_max),),
        )
        conn.execute(
            "INSERT OR REPLACE INTO meta(key, value) VALUES('finalref_snr_min', ?)",
            (str(finalref_snr_min),),
        )
        conn.execute(
            "INSERT OR REPLACE INTO meta(key, value) VALUES('build_time_iso', ?)",
            (datetime.now(UTC).isoformat(),),
        )
        if meta_extra:
            for k, v in meta_extra.items():
                conn.execute(
                    "INSERT OR REPLACE INTO meta(key, value) VALUES(?, ?)",
                    (str(k), str(v)),
                )
        # Create/replace a view for good reference sources
        try:
            conn.execute("DROP VIEW IF EXISTS good_references")
        except sqlite3.Error:
            pass
        conn.execute(
            f"""
            CREATE VIEW good_references AS
            SELECT * FROM sources
            WHERE snr_nvss IS NOT NULL AND snr_nvss > {goodref_snr_min}
              AND resolved_flag = 0 AND confusion_flag = 0
              AND alpha IS NOT NULL AND alpha BETWEEN {goodref_alpha_min} AND {goodref_alpha_max}
            """
        )
        # Optional: stable IDs constraint for final references
        if finalref_ids is not None:
            try:
                conn.execute("DROP TABLE IF EXISTS stable_ids")
            except sqlite3.Error:
                pass
            conn.execute("CREATE TABLE IF NOT EXISTS stable_ids(source_id INTEGER PRIMARY KEY)")
            rows = [(int(i),) for i in finalref_ids if i is not None]
            if rows:
                conn.executemany("INSERT OR IGNORE INTO stable_ids(source_id) VALUES(?)", rows)
            conn.execute(
                "INSERT OR REPLACE INTO meta(key, value) VALUES('finalref_ids_count', ?)",
                (str(len(rows)),),
            )
        else:
            conn.execute(
                "INSERT OR REPLACE INTO meta(key, value) VALUES('finalref_ids_count', '0')"
            )
        # Create final_references view: stricter SNR and (optionally) membership in stable_ids
        try:
            conn.execute("DROP VIEW IF EXISTS final_references")
        except sqlite3.Error:
            pass
        if finalref_ids is not None:
            conn.execute(
                f"""
                CREATE VIEW final_references AS
                SELECT s.* FROM sources s
                JOIN stable_ids t ON t.source_id = s.source_id
                WHERE s.snr_nvss IS NOT NULL AND s.snr_nvss > {finalref_snr_min}
                  AND s.resolved_flag = 0 AND s.confusion_flag = 0
                  AND s.alpha IS NOT NULL AND s.alpha BETWEEN {goodref_alpha_min} AND {goodref_alpha_max}
                """
            )
        else:
            conn.execute(
                f"""
                CREATE VIEW final_references AS
                SELECT * FROM sources
                WHERE snr_nvss IS NOT NULL AND snr_nvss > {finalref_snr_min}
                  AND resolved_flag = 0 AND confusion_flag = 0
                  AND alpha IS NOT NULL AND alpha BETWEEN {goodref_alpha_min} AND {goodref_alpha_max}
                """
            )
        # Optionally materialize final_references into a table snapshot
        if materialize_final:
            try:
                conn.execute("DROP TABLE IF EXISTS final_references_table")
            except sqlite3.Error:
                pass
            conn.execute("CREATE TABLE final_references_table AS SELECT * FROM final_references")


# ----------------------------- Column maps ---------------------------------


NVSS_CANDIDATES = {
    "ra": ["ra", "raj2000", "ra_deg"],
    "dec": ["dec", "dej2000", "dec_deg"],
    "flux": ["s1.4", "flux", "flux_jy", "peak_flux", "spk", "s_pk"],
    "snr": ["snr", "s/n", "snratio"],
}

VLASS_CANDIDATES = {
    "ra": ["ra", "ra_deg", "raj2000"],
    "dec": ["dec", "dec_deg", "dej2000"],
    # Prefer peak flux density for compactness comparisons
    "flux": ["peak_flux", "peak_mjy_per_beam", "flux_peak", "flux", "total_flux"],
}

FIRST_CANDIDATES = {
    "ra": ["ra", "ra_deg", "raj2000"],
    "dec": ["dec", "dec_deg", "dej2000"],
    # deconvolved major/minor FWHM in arcsec if present
    "maj": ["deconv_maj", "maj", "fwhm_maj", "deconvolved_major"],
    "min": ["deconv_min", "min", "fwhm_min", "deconvolved_minor"],
}


# ------------------------------- CLI ---------------------------------------


def build_master(
    nvss_path: str,
    *,
    vlass_path: str | None = None,
    first_path: str | None = None,
    out_db: str = "state/catalogs/master_sources.sqlite3",
    match_radius_arcsec: float = 7.5,
    map_nvss: dict[str, str] | None = None,
    map_vlass: dict[str, str] | None = None,
    map_first: dict[str, str] | None = None,
    nvss_flux_unit: str = "jy",
    vlass_flux_unit: str = "jy",
    goodref_snr_min: float = 50.0,
    goodref_alpha_min: float = -1.2,
    goodref_alpha_max: float = 0.2,
    finalref_snr_min: float = 80.0,
    finalref_ids_file: str | None = None,
    materialize_final: bool = False,
) -> Path:
    df_nvss = _read_table(nvss_path)
    df_vlass = _read_table(vlass_path) if vlass_path else None
    df_first = _read_table(first_path) if first_path else None

    # Resolve column names
    nv_map = _normalize_columns(df_nvss, NVSS_CANDIDATES)
    if map_nvss:
        nv_map.update(map_nvss)
    v_map: dict[str, str] = {}
    if df_vlass is not None:
        v_map = _normalize_columns(df_vlass, VLASS_CANDIDATES)
        if map_vlass:
            v_map.update(map_vlass)
    f_map: dict[str, str] = {}
    if df_first is not None:
        f_map = _normalize_columns(df_first, FIRST_CANDIDATES)
        if map_first:
            f_map.update(map_first)

    # Unit scales to Jy
    def _scale(unit: str) -> float:
        u = unit.lower()
        if u in ("jy",):
            return 1.0
        if u in ("mjy",):
            return 1e-3
        if u in ("ujy", "µjy", "uJy"):
            return 1e-6
        # default assume already Jy
        return 1.0

    out = _crossmatch(
        df_nvss,
        df_vlass,
        df_first,
        maps={"nvss": nv_map, "vlass": v_map, "first": f_map},
        match_radius_arcsec=match_radius_arcsec,
        scale_nvss_to_jy=_scale(nvss_flux_unit),
        scale_vlass_to_jy=_scale(vlass_flux_unit),
    )

    # Build meta provenance: file hashes and row counts
    def _hash(path: str | None) -> tuple[str, int, int]:
        if not path:
            return ("", 0, 0)
        h = hashlib.sha256()
        try:
            with open(path, "rb") as f:
                for chunk in iter(lambda: f.read(1024 * 1024), b""):
                    h.update(chunk)
            size = os.path.getsize(path)
            mtime = int(os.path.getmtime(path))
            return (h.hexdigest(), int(size), mtime)
        except OSError:
            return ("", 0, 0)

    meta_extra: dict[str, str] = {}
    hv, sv, mv = _hash(nvss_path)
    meta_extra.update(
        {
            "nvss_path": os.fspath(nvss_path),
            "nvss_sha256": hv,
            "nvss_size": str(sv),
            "nvss_mtime": str(mv),
            "nvss_rows": str(len(df_nvss)),
        }
    )
    if vlass_path:
        hv, sv, mv = _hash(vlass_path)
        meta_extra.update(
            {
                "vlass_path": os.fspath(vlass_path),
                "vlass_sha256": hv,
                "vlass_size": str(sv),
                "vlass_mtime": str(mv),
                "vlass_rows": str(len(df_vlass) if df_vlass is not None else 0),
            }
        )
    if first_path:
        hv, sv, mv = _hash(first_path)
        meta_extra.update(
            {
                "first_path": os.fspath(first_path),
                "first_sha256": hv,
                "first_size": str(sv),
                "first_mtime": str(mv),
                "first_rows": str(len(df_first) if df_first is not None else 0),
            }
        )

    # Optional final reference IDs
    final_ids: list[int] | None = None
    if finalref_ids_file:
        try:
            with open(finalref_ids_file, encoding="utf-8") as f:
                final_ids = [
                    int(x.strip()) for x in f if x.strip() and not x.strip().startswith("#")
                ]
        except (OSError, ValueError):
            final_ids = None

    out_db_path = Path(out_db)
    _write_sqlite(
        out,
        out_db_path,
        goodref_snr_min=goodref_snr_min,
        goodref_alpha_min=goodref_alpha_min,
        goodref_alpha_max=goodref_alpha_max,
        finalref_snr_min=finalref_snr_min,
        finalref_ids=final_ids,
        materialize_final=materialize_final,
        meta_extra=meta_extra,
    )
    return out_db_path


def _add_map_args(p: argparse.ArgumentParser, prefix: str) -> None:
    p.add_argument(f"--map-{prefix}-ra", dest=f"map_{prefix}_ra")
    p.add_argument(f"--map-{prefix}-dec", dest=f"map_{prefix}_dec")
    p.add_argument(f"--map-{prefix}-flux", dest=f"map_{prefix}_flux")
    if prefix == "first":
        p.add_argument(f"--map-{prefix}-maj", dest=f"map_{prefix}_maj")
        p.add_argument(f"--map-{prefix}-min", dest=f"map_{prefix}_min")


def main(argv: list[str] | None = None) -> int:
    from dsa110_contimg.core.catalog.build_master_cli import main as unified_main

    argv = [] if argv is None else list(argv)
    return unified_main(["files", *argv])


# ---------------------------------------------------------------------------
# Build master catalog from SQLite databases (preferred method)
# ---------------------------------------------------------------------------


def build_master_from_sqlite(
    output_path: Path | None = None,
    *,
    nvss_db: Path | None = None,
    vlass_db: Path | None = None,
    first_db: Path | None = None,
    rax_db: Path | None = None,
    match_radius_arcsec: float = 7.5,
    chunk_size: int = 100_000,
    goodref_snr_min: float = 12.0,
    goodref_alpha_min: float = -1.2,
    goodref_alpha_max: float = 0.2,
    finalref_snr_min: float = 15.0,
    force_rebuild: bool = False,
    with_provenance: bool = False,
    progress_callback: Callable[[int, int], None] | None = None,
    resume: bool = False,
) -> Path:
    """Build master catalog by crossmatching directly from SQLite databases.

        This is the preferred method for building the master catalog, as it:
        - Avoids memory-intensive CSV/FITS loading
        - Uses indexed SQLite queries for efficient spatial lookups
        - Processes in chunks for memory efficiency

    Parameters
    ----------
    output_path : str, optional
        Output database path. Default is 'state/catalogs/master_sources.sqlite3'.
    nvss_db : Optional[str], optional
        Path to nvss_full.sqlite3 (auto-detected if None).
    vlass_db : Optional[str], optional
        Path to vlass_full.sqlite3 (auto-detected if None).
    first_db : Optional[str], optional
        Path to first_full.sqlite3 (auto-detected if None).
    rax_db : Optional[str], optional
        Path to rax_full.sqlite3 (auto-detected if None).
    match_radius_arcsec : float, optional
        Crossmatch radius in arcseconds. Default is 7.5.
    chunk_size : int, optional
        Process NVSS in chunks of this size. Default is 100000.
    goodref_snr_min : Optional[float], optional
        SNR threshold for good_references view.
    goodref_alpha_min : Optional[float], optional
        Minimum spectral index for good_references.
    goodref_alpha_max : Optional[float], optional
        Maximum spectral index for good_references.
    finalref_snr_min : Optional[float], optional
        SNR threshold for final_references view.
    force_rebuild : bool, optional
        If True, overwrite existing database. Default is False.
    progress_callback : callable, optional
        Optional callback(current, total) for progress.

    Returns
    -------
        str
        Path to the created master_sources.sqlite3

    Raises
    ------
        FileNotFoundError
        If NVSS database is not found (required)
        sqlite3.DatabaseError
        If source databases are corrupted

        Example
    -------
        >>> from dsa110_contimg.core.catalog.build_master import build_master_from_sqlite
        >>> db_path = build_master_from_sqlite()
        >>> print(f"Built master catalog at {db_path}")
    """
    from dsa110_contimg.core.catalog.builders import (
        get_first_full_db_path,
        get_nvss_full_db_path,
        get_rax_full_db_path,
        get_vlass_full_db_path,
    )

    # Resolve default paths
    if output_path is None:
        output_path = (
            get_env_path("CONTIMG_BASE_DIR", default="/data/dsa110-contimg")
            / "state/catalogs/master_sources.sqlite3"
        )
    output_path = Path(output_path)

    if nvss_db is None:
        nvss_db = get_nvss_full_db_path()
    if vlass_db is None:
        vlass_db = get_vlass_full_db_path()
    if first_db is None:
        first_db = get_first_full_db_path()
    if rax_db is None:
        rax_db = get_rax_full_db_path()


    # Validate NVSS exists (required as base catalog)
    if not nvss_db.exists():
        raise FileNotFoundError(f"NVSS database not found: {nvss_db}")

    # Check if output exists (skip if resume and we will append)
    if output_path.exists() and not force_rebuild and not resume:
        logger.info(f"Master catalog already exists: {output_path}")
        return output_path

    logger.info("Building master catalog from SQLite databases...")
    logger.info(f"  NVSS:  {nvss_db} (required)")
    logger.info(f"  VLASS: {vlass_db} {'(found)' if vlass_db.exists() else '(not found)'}")
    logger.info(f"  FIRST: {first_db} {'(found)' if first_db.exists() else '(not found)'}")
    logger.info(f"  RAX:   {rax_db} {'(found)' if rax_db.exists() else '(not found)'}")

    # Helper to safely open database and check integrity
    def open_catalog_db(db_path: Path, name: str) -> sqlite3.Connection | None:
        if not db_path.exists():
            logger.warning(f"{name} database not found, skipping: {db_path}")
            return None
        try:
            conn = sqlite3.connect(str(db_path))
            # Quick integrity check
            conn.execute("SELECT 1 FROM sources LIMIT 1")
            return conn
        except sqlite3.DatabaseError as e:
            logger.warning(f"{name} database corrupted, skipping: {e}")
            return None

    # Open source databases
    nvss_conn = sqlite3.connect(str(nvss_db))
    vlass_conn = open_catalog_db(vlass_db, "VLASS")
    first_conn = open_catalog_db(first_db, "FIRST")
    rax_conn = open_catalog_db(rax_db, "RAX")

    try:

        def _ra_where_clause(ra_min: float, ra_max: float) -> str:
            # Handle RA wrap-around at 0/360 degrees.
            if (ra_max - ra_min) >= 360.0:
                return "1=1"
            if ra_min < 0.0 and ra_max > 360.0:
                return "1=1"
            if ra_min < 0.0:
                return f"(ra_deg >= {ra_min + 360.0} OR ra_deg <= {ra_max})"
            if ra_max > 360.0:
                return f"(ra_deg >= {ra_min} OR ra_deg <= {ra_max - 360.0})"
            return f"ra_deg BETWEEN {ra_min} AND {ra_max}"

        # Get total NVSS count for progress
        total_nvss = nvss_conn.execute("SELECT COUNT(*) FROM sources").fetchone()[0]
        logger.info(f"Processing {total_nvss:,} NVSS sources in chunks of {chunk_size:,}")
        print(f"Processing {total_nvss:,} NVSS sources in chunks of {chunk_size:,}...", file=sys.stderr, flush=True)

        # Resume from existing DB or create new one
        output_path.parent.mkdir(parents=True, exist_ok=True)
        processed = 0
        source_id = 0
        if resume and output_path.exists():
            try:
                _conn = sqlite3.connect(str(output_path))
                cur = _conn.execute("SELECT COUNT(*), MAX(source_id) FROM sources")
                row_count, max_sid = cur.fetchone()
                _conn.close()
                if row_count and row_count > 0 and max_sid is not None and row_count < total_nvss:
                    processed = int(row_count)
                    source_id = int(max_sid)
                    print(f"Resuming from {processed:,} sources (source_id={source_id:,})...", file=sys.stderr, flush=True)
            except sqlite3.DatabaseError:
                pass

        if processed == 0:
            if output_path.exists():
                output_path.unlink()
            out_conn = sqlite3.connect(str(output_path))
            out_conn.execute("""
                CREATE TABLE sources (
                    source_id INTEGER PRIMARY KEY,
                    ra_deg REAL NOT NULL,
                    dec_deg REAL NOT NULL,
                    s_nvss REAL,
                    snr_nvss REAL,
                    s_vlass REAL,
                    s_rax REAL,
                    alpha REAL,
                    resolved_flag INTEGER NOT NULL DEFAULT 0,
                    confusion_flag INTEGER NOT NULL DEFAULT 0,
                    has_nvss INTEGER NOT NULL DEFAULT 0,
                    has_vlass INTEGER NOT NULL DEFAULT 0,
                    has_first INTEGER NOT NULL DEFAULT 0,
                    has_rax INTEGER NOT NULL DEFAULT 0
                )
            """)
        else:
            out_conn = sqlite3.connect(str(output_path))

        # Match radius in degrees for SQL queries
        match_radius_deg = match_radius_arcsec / 3600.0

        # Process NVSS in chunks (from processed offset when resuming)
        if progress_callback:
            progress_callback(processed, total_nvss)

        for chunk_start in range(processed, total_nvss, chunk_size):
            print(f"  Chunk {chunk_start:,}-{chunk_start + chunk_size:,}: read NVSS...", file=sys.stderr, flush=True)
            # Read NVSS chunk ordered by (dec_deg, ra_deg) so the chunk is spatially compact.
            # Without ORDER BY, LIMIT/OFFSET returns arbitrary rows and a chunk can span the full
            # sky (ra_min..ra_max >= 360°), so catalog queries return millions of rows and
            # search_around_sky(100k, millions) hangs (root cause of hang at ~73%).
            nvss_chunk = pd.read_sql_query(
                f"""
                SELECT ra_deg, dec_deg, flux_mjy, flux_err_mjy
                FROM sources
                ORDER BY dec_deg, ra_deg
                LIMIT {chunk_size} OFFSET {chunk_start}
                """,
                nvss_conn,
            )

            if len(nvss_chunk) == 0:
                break

            # Build SkyCoord for this chunk
            nvss_sc = SkyCoord(
                ra=nvss_chunk["ra_deg"].values * u.deg,
                dec=nvss_chunk["dec_deg"].values * u.deg,
                frame="icrs",
            )

            # Get bounding box for efficient SQL queries
            ra_min = float(nvss_chunk["ra_deg"].min() - match_radius_deg)
            ra_max = float(nvss_chunk["ra_deg"].max() + match_radius_deg)
            dec_min = float(nvss_chunk["dec_deg"].min() - match_radius_deg)
            dec_max = float(nvss_chunk["dec_deg"].max() + match_radius_deg)

            ra_where = _ra_where_clause(ra_min, ra_max)

            # Cap rows per catalog so search_around_sky doesn't hang (astropy very slow on huge
            # arrays). Combined with ORDER BY dec_deg, ra_deg on NVSS chunks so each chunk is
            # spatially compact; without that, some chunks span full RA and pull millions of rows.
            _max_catalog_rows = 600_000

            # Load nearby sources from other catalogs
            print(f"  Chunk {chunk_start:,}: load VLASS/FIRST/RAX...", file=sys.stderr, flush=True)
            vlass_df = None
            if vlass_conn:
                vlass_df = pd.read_sql_query(
                    f"""
                    SELECT ra_deg, dec_deg, flux_mjy
                    FROM sources
                    WHERE dec_deg BETWEEN {dec_min} AND {dec_max}
                                            AND {ra_where}
                    LIMIT {_max_catalog_rows}
                    """,
                    vlass_conn,
                )
                if len(vlass_df) >= _max_catalog_rows:
                    print(f"  Chunk {chunk_start:,}: VLASS capped at {_max_catalog_rows:,}", file=sys.stderr, flush=True)

            first_df = None
            if first_conn:
                first_df = pd.read_sql_query(
                    f"""
                    SELECT ra_deg, dec_deg, flux_mjy, maj_arcsec, min_arcsec
                    FROM sources
                    WHERE dec_deg BETWEEN {dec_min} AND {dec_max}
                                            AND {ra_where}
                    LIMIT {_max_catalog_rows}
                    """,
                    first_conn,
                )
                if first_df is not None and len(first_df) >= _max_catalog_rows:
                    print(f"  Chunk {chunk_start:,}: FIRST capped at {_max_catalog_rows:,}", file=sys.stderr, flush=True)

            rax_df = None
            if rax_conn:
                rax_df = pd.read_sql_query(
                    f"""
                    SELECT ra_deg, dec_deg, flux_mjy
                    FROM sources
                    WHERE dec_deg BETWEEN {dec_min} AND {dec_max}
                                            AND {ra_where}
                    LIMIT {_max_catalog_rows}
                    """,
                    rax_conn,
                )
                if rax_df is not None and len(rax_df) >= _max_catalog_rows:
                    print(f"  Chunk {chunk_start:,}: RAX capped at {_max_catalog_rows:,}", file=sys.stderr, flush=True)

            # Build SkyCoords for matching catalogs
            vlass_sc = None
            if vlass_df is not None and len(vlass_df) > 0:
                vlass_sc = SkyCoord(
                    ra=vlass_df["ra_deg"].values * u.deg,
                    dec=vlass_df["dec_deg"].values * u.deg,
                    frame="icrs",
                )

            first_sc = None
            if first_df is not None and len(first_df) > 0:
                first_sc = SkyCoord(
                    ra=first_df["ra_deg"].values * u.deg,
                    dec=first_df["dec_deg"].values * u.deg,
                    frame="icrs",
                )

            rax_sc = None
            if rax_df is not None and len(rax_df) > 0:
                rax_sc = SkyCoord(
                    ra=rax_df["ra_deg"].values * u.deg,
                    dec=rax_df["dec_deg"].values * u.deg,
                    frame="icrs",
                )

            # Crossmatch using search_around_sky for efficiency
            radius = match_radius_arcsec * u.arcsec

            # VLASS matches
            v_idx_by_n: dict[int, list] = {}
            if vlass_sc is not None:
                try:
                    # NOTE: astropy returns (idx_other, idx_self)
                    idx_v, idx_nv, _, _ = nvss_sc.search_around_sky(vlass_sc, radius)
                    for i_v, i_n in zip(idx_v, idx_nv):
                        v_idx_by_n.setdefault(int(i_n), []).append(int(i_v))
                except Exception:
                    pass

            # FIRST matches
            f_idx_by_n: dict[int, list] = {}
            if first_sc is not None:
                print(f"  Chunk {chunk_start:,}: match FIRST ({len(first_sc):,} sources)...", file=sys.stderr, flush=True)
                try:
                    # NOTE: astropy returns (idx_other, idx_self)
                    idx_f, idx_nf, _, _ = nvss_sc.search_around_sky(first_sc, radius)
                    for i_f, i_n in zip(idx_f, idx_nf):
                        f_idx_by_n.setdefault(int(i_n), []).append(int(i_f))
                except Exception:
                    pass

            # RAX matches
            r_idx_by_n: dict[int, list] = {}
            if rax_sc is not None:
                print(f"  Chunk {chunk_start:,}: match RAX ({len(rax_sc):,} sources)...", file=sys.stderr, flush=True)
                try:
                    # NOTE: astropy returns (idx_other, idx_self)
                    idx_r, idx_nr, _, _ = nvss_sc.search_around_sky(rax_sc, radius)
                    for i_r, i_n in zip(idx_r, idx_nr):
                        r_idx_by_n.setdefault(int(i_n), []).append(int(i_r))
                except Exception:
                    pass

            # Build output rows
            out_rows = []
            _progress_every = 25_000  # Report progress within chunk so it doesn't look stuck
            for i in range(len(nvss_chunk)):
                if progress_callback and (i + 1) % _progress_every == 0:
                    progress_callback(chunk_start + i + 1, total_nvss)
                ra = float(nvss_chunk.iloc[i]["ra_deg"])
                dec = float(nvss_chunk.iloc[i]["dec_deg"])
                s_nvss = float(nvss_chunk.iloc[i]["flux_mjy"]) / 1000.0  # mJy -> Jy

                snr_nvss = None
                try:
                    ferr = nvss_chunk.iloc[i].get("flux_err_mjy")
                    ferr_val = float(ferr) if ferr is not None and not pd.isna(ferr) else None
                    if ferr_val is not None and ferr_val > 0.0:
                        snr_nvss = float(nvss_chunk.iloc[i]["flux_mjy"]) / ferr_val
                except Exception:
                    snr_nvss = None

                # VLASS match
                s_vlass = None
                has_vlass = 0
                confusion = 0
                if i in v_idx_by_n:
                    cands = v_idx_by_n[i]
                    if len(cands) > 1:
                        confusion = 1
                    # Pick closest
                    best_idx = cands[0]
                    if len(cands) > 1 and vlass_sc is not None:
                        seps = SkyCoord(ra=ra * u.deg, dec=dec * u.deg).separation(vlass_sc[cands])
                        best_idx = cands[int(np.argmin(seps.to_value(u.arcsec)))]
                    fmjy = vlass_df.iloc[best_idx].get("flux_mjy")
                    if fmjy is not None and not pd.isna(fmjy):
                        s_vlass = float(fmjy) / 1000.0
                    has_vlass = 1

                # FIRST match (for resolved flag)
                resolved = 0
                has_first = 0
                if i in f_idx_by_n:
                    cands = f_idx_by_n[i]
                    has_first = 1
                    if len(cands) > 1:
                        confusion = 1

                    best_idx = cands[0]
                    if len(cands) > 1 and first_sc is not None:
                        seps = SkyCoord(ra=ra * u.deg, dec=dec * u.deg).separation(first_sc[cands])
                        best_idx = cands[int(np.argmin(seps.to_value(u.arcsec)))]

                    try:
                        maj = first_df.iloc[best_idx].get("maj_arcsec")
                        mn = first_df.iloc[best_idx].get("min_arcsec")
                        maj_val = float(maj) if maj is not None and not pd.isna(maj) else None
                        min_val = float(mn) if mn is not None and not pd.isna(mn) else None
                    except Exception:
                        maj_val = None
                        min_val = None

                    # Heuristic: resolved if either axis > 6 arcsec (FIRST beam ~5")
                    if (maj_val is not None and maj_val > 6.0) or (
                        min_val is not None and min_val > 6.0
                    ):
                        resolved = 1

                # RAX match
                s_rax = None
                has_rax = 0
                if i in r_idx_by_n:
                    cands = r_idx_by_n[i]
                    if len(cands) > 1:
                        confusion = 1
                    best_idx = cands[0]
                    if len(cands) > 1 and rax_sc is not None:
                        seps = SkyCoord(ra=ra * u.deg, dec=dec * u.deg).separation(rax_sc[cands])
                        best_idx = cands[int(np.argmin(seps.to_value(u.arcsec)))]
                    fmjy = rax_df.iloc[best_idx].get("flux_mjy")
                    if fmjy is not None and not pd.isna(fmjy):
                        s_rax = float(fmjy) / 1000.0
                    has_rax = 1

                # Compute spectral index (NVSS 1.4 GHz -> VLASS 3 GHz)
                alpha = _compute_alpha(s_nvss, 1.4e9, s_vlass, 3.0e9)

                source_id += 1
                out_rows.append(
                    (
                        source_id,
                        ra,
                        dec,
                        s_nvss,
                        snr_nvss,
                        s_vlass,
                        s_rax,
                        alpha,
                        resolved,
                        confusion,
                        1,  # has_nvss: every row is NVSS-based in this build
                        has_vlass,
                        has_first,
                        has_rax,
                    )
                )

            # Insert batch
            out_conn.executemany(
                """
                INSERT INTO sources (
                    source_id, ra_deg, dec_deg, s_nvss, snr_nvss, s_vlass, s_rax,
                    alpha, resolved_flag, confusion_flag, has_nvss, has_vlass, has_first, has_rax
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                out_rows,
            )
            out_conn.commit()

            processed += len(nvss_chunk)
            if progress_callback:
                progress_callback(processed, total_nvss)
            logger.info(
                f"Processed {processed:,}/{total_nvss:,} sources ({100 * processed / total_nvss:.1f}%)"
            )

        # Create indexes
        logger.info("Creating indexes...")
        out_conn.execute("CREATE INDEX IF NOT EXISTS idx_sources_radec ON sources(ra_deg, dec_deg)")
        out_conn.execute("CREATE INDEX IF NOT EXISTS idx_sources_dec ON sources(dec_deg)")
        out_conn.execute("CREATE INDEX IF NOT EXISTS idx_sources_flux ON sources(s_nvss)")

        # Create views
        out_conn.execute("DROP VIEW IF EXISTS good_references")
        out_conn.execute(f"""
            CREATE VIEW good_references AS
            SELECT * FROM sources
                        WHERE s_nvss IS NOT NULL
                            AND snr_nvss IS NOT NULL AND snr_nvss > {goodref_snr_min}
              AND resolved_flag = 0 AND confusion_flag = 0
              AND alpha IS NOT NULL AND alpha BETWEEN {goodref_alpha_min} AND {goodref_alpha_max}
        """)

        out_conn.execute("DROP VIEW IF EXISTS final_references")
        out_conn.execute(f"""
            CREATE VIEW final_references AS
            SELECT * FROM sources
                        WHERE s_nvss IS NOT NULL
                            AND snr_nvss IS NOT NULL AND snr_nvss > {finalref_snr_min}
              AND resolved_flag = 0 AND confusion_flag = 0
              AND alpha IS NOT NULL AND alpha BETWEEN {goodref_alpha_min} AND {goodref_alpha_max}
        """)

        # Write metadata
        out_conn.execute("""
            CREATE TABLE IF NOT EXISTS meta (
                key TEXT PRIMARY KEY,
                value TEXT
            )
        """)
        meta_values = [
            ("build_time_iso", datetime.now(UTC).isoformat()),
            ("build_method", "nvss_centered"),
            ("match_radius_arcsec", str(match_radius_arcsec)),
            ("nvss_db", str(nvss_db)),
            ("vlass_db", str(vlass_db) if vlass_conn else "not_available"),
            ("first_db", str(first_db) if first_conn else "not_available"),
            ("rax_db", str(rax_db) if rax_conn else "not_available"),
            ("total_sources", str(source_id)),
            ("goodref_snr_min", str(goodref_snr_min)),
            ("finalref_snr_min", str(finalref_snr_min)),
            ("goodref_alpha_min", str(goodref_alpha_min)),
            ("goodref_alpha_max", str(goodref_alpha_max)),
        ]
        out_conn.executemany(
            "INSERT OR REPLACE INTO meta(key, value) VALUES(?, ?)",
            meta_values,
        )
        out_conn.commit()

        logger.info(f"Built master catalog with {source_id:,} sources: {output_path}")
        return output_path

    finally:
        # Clean up connections
        nvss_conn.close()
        if vlass_conn:
            vlass_conn.close()
        if first_conn:
            first_conn.close()
        if rax_conn:
            rax_conn.close()
        if "out_conn" in locals():
            out_conn.close()


def build_master_union_from_sqlite(
    output_path: Path | None = None,
    *,
    nvss_db: Path | None = None,
    vlass_db: Path | None = None,
    first_db: Path | None = None,
    rax_db: Path | None = None,
    match_radius_arcsec: float = 7.5,
    chunk_size: int = 100_000,
    goodref_snr_min: float = 12.0,
    goodref_alpha_min: float = -1.2,
    goodref_alpha_max: float = 0.2,
    finalref_snr_min: float = 15.0,
    force_rebuild: bool = False,
    with_provenance: bool = True,
    progress_callback: callable | None = None,
    resume: bool = False,
) -> Path:
    """Build a union (not NVSS-centered) master catalog.

    Produces one row per *unique* sky source, where uniqueness is defined by
    positional crossmatch within `match_radius_arcsec` across the available
    survey catalogs.

    The resulting `sources` table includes per-catalog flux columns (Jy),
    per-catalog presence flags, and a `flux_jy` column used for fast querying.
    """
    import sqlite3

    from dsa110_contimg.core.catalog.builders import (
        get_first_full_db_path,
        get_nvss_full_db_path,
        get_rax_full_db_path,
        get_vlass_full_db_path,
    )

    if output_path is None:
        output_path = (
            get_env_path("CONTIMG_BASE_DIR", default="/data/dsa110-contimg")
            / "state/catalogs/master_sources.sqlite3"
        )
    output_path = Path(output_path)

    if nvss_db is None:
        nvss_db = get_nvss_full_db_path()
    if vlass_db is None:
        vlass_db = get_vlass_full_db_path()
    if first_db is None:
        first_db = get_first_full_db_path()
    if rax_db is None:
        rax_db = get_rax_full_db_path()

    if output_path.exists() and not force_rebuild:
        return output_path

    # Prepare output directory
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Overwrite existing output
    if output_path.exists():
        output_path.unlink()

    match_radius_deg = match_radius_arcsec / 3600.0
    radius = match_radius_arcsec * u.arcsec

    def _ra_where_clause_local(ra_min: float, ra_max: float) -> str:
        """Return a SQL RA WHERE clause that is safe across 0/360 wrap."""
        ra_min_mod = ra_min % 360.0
        ra_max_mod = ra_max % 360.0
        if ra_min_mod <= ra_max_mod:
            return f"ra_deg BETWEEN {ra_min_mod} AND {ra_max_mod}"
        return f"(ra_deg >= {ra_min_mod} OR ra_deg <= {ra_max_mod})"

    def _iter_sqlite_chunks(conn: sqlite3.Connection, sql: str, params: tuple, chunk: int):
        last_rowid = 0
        while True:
            df = pd.read_sql_query(
                sql,
                conn,
                params=(last_rowid, *params, chunk),
            )
            if df is None or len(df) == 0:
                break
            last_rowid = int(df["rowid"].max())
            yield df

    def _ensure_provenance_tables(conn: sqlite3.Connection) -> None:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS catalog_matches (
                source_id INTEGER NOT NULL,
                catalog TEXT NOT NULL,
                catalog_row_id INTEGER NOT NULL,
                sep_arcsec REAL,
                match_rank INTEGER DEFAULT 0,
                is_primary INTEGER DEFAULT 0,
                match_version INTEGER DEFAULT 1,
                PRIMARY KEY (source_id, catalog, catalog_row_id)
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_catalog_matches_source ON catalog_matches(source_id)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_catalog_matches_catalog_row ON catalog_matches(catalog, catalog_row_id)"
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS catalogs (
                catalog TEXT PRIMARY KEY,
                source_db_path TEXT,
                source_hash TEXT,
                n_rows INTEGER,
                build_time_iso TEXT,
                raw_rows_format TEXT,
                raw_rows_path TEXT,
                raw_rows_hash TEXT
            )
            """
        )
        cols = {row[1] for row in conn.execute("PRAGMA table_info(catalogs)").fetchall()}
        if "raw_rows_format" not in cols:
            conn.execute("ALTER TABLE catalogs ADD COLUMN raw_rows_format TEXT")
        if "raw_rows_path" not in cols:
            conn.execute("ALTER TABLE catalogs ADD COLUMN raw_rows_path TEXT")
        if "raw_rows_hash" not in cols:
            conn.execute("ALTER TABLE catalogs ADD COLUMN raw_rows_hash TEXT")

    def _source_columns(conn: sqlite3.Connection) -> set[str]:
        return {row[1] for row in conn.execute("PRAGMA table_info(sources)").fetchall()}

    def _record_catalog_meta(catalog: str, db_path: Path | None) -> None:
        if not with_provenance or db_path is None:
            return
        if not Path(db_path).exists():
            return
        build_time = None
        n_rows = None
        with sqlite3.connect(str(db_path)) as conn:
            try:
                n_rows = conn.execute("SELECT COUNT(*) FROM sources").fetchone()[0]
            except Exception:
                n_rows = None
            try:
                row = conn.execute(
                    "SELECT value FROM meta WHERE key='build_time_iso'"
                ).fetchone()
                if row:
                    build_time = row[0]
            except Exception:
                build_time = None
            raw_rows_format = None
            raw_rows_path = None
            raw_rows_hash = None
            try:
                row = conn.execute(
                    "SELECT value FROM meta WHERE key='raw_rows_format'"
                ).fetchone()
                if row:
                    raw_rows_format = row[0]
            except Exception:
                raw_rows_format = None
            try:
                row = conn.execute(
                    "SELECT value FROM meta WHERE key='raw_rows_path'"
                ).fetchone()
                if row:
                    raw_rows_path = row[0]
            except Exception:
                raw_rows_path = None
            try:
                row = conn.execute(
                    "SELECT value FROM meta WHERE key='raw_rows_hash'"
                ).fetchone()
                if row:
                    raw_rows_hash = row[0]
            except Exception:
                raw_rows_hash = None
        source_hash = _hash_file(Path(db_path))
        out_conn.execute(
            """
            INSERT OR REPLACE INTO catalogs(
                catalog, source_db_path, source_hash, n_rows, build_time_iso,
                raw_rows_format, raw_rows_path, raw_rows_hash
            ) VALUES(?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                catalog,
                os.fspath(db_path),
                source_hash,
                n_rows,
                build_time,
                raw_rows_format,
                raw_rows_path,
                raw_rows_hash,
            ),
        )
        out_conn.execute(
            "INSERT OR REPLACE INTO meta(key, value) VALUES(?, ?)",
            (f"{catalog}_db_hash", source_hash or ""),
        )

    def _update_flux_jy_sql() -> str:
        return """
        flux_jy = MAX(
            COALESCE(s_vlass, 0.0),
            COALESCE(s_first, 0.0),
            COALESCE(s_nvss, 0.0),
            COALESCE(s_rax, 0.0)
        )
        """.strip()

    with sqlite3.connect(str(output_path)) as out_conn:
        out_conn.execute("PRAGMA journal_mode=WAL")
        out_conn.execute("PRAGMA synchronous=NORMAL")

        out_conn.execute(
            """
            CREATE TABLE sources (
                source_id INTEGER PRIMARY KEY,
                ra_deg REAL NOT NULL,
                dec_deg REAL NOT NULL,
                flux_jy REAL,
                s_nvss REAL,
                snr_nvss REAL,
                s_vlass REAL,
                s_first REAL,
                s_rax REAL,
                alpha REAL,
                resolved_flag INTEGER DEFAULT 0,
                confusion_flag INTEGER DEFAULT 0,
                has_nvss INTEGER DEFAULT 0,
                has_vlass INTEGER DEFAULT 0,
                has_first INTEGER DEFAULT 0,
                has_rax INTEGER DEFAULT 0
            )
            """
        )
        out_conn.execute("CREATE INDEX idx_sources_pos ON sources(dec_deg, ra_deg)")
        out_conn.execute("CREATE INDEX idx_sources_flux ON sources(flux_jy)")

        out_conn.execute("CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT)")
        out_conn.executemany(
            "INSERT OR REPLACE INTO meta(key, value) VALUES(?, ?)",
            [
                ("build_time_iso", datetime.now(UTC).isoformat()),
                ("build_method", "union"),
                ("match_radius_arcsec", str(match_radius_arcsec)),
                ("chunk_size", str(chunk_size)),
                ("goodref_snr_min", str(goodref_snr_min)),
                ("finalref_snr_min", str(finalref_snr_min)),
            ],
        )

        if with_provenance:
            _ensure_provenance_tables(out_conn)
            out_conn.execute(
                "INSERT OR REPLACE INTO meta(key, value) VALUES('schema_version', ?)",
                ("3",),
            )

        # Ensure schema/meta are committed before chunked bulk operations.
        out_conn.commit()

        next_source_id = 1

        def _allocate_source_ids(n_rows: int) -> list[int]:
            nonlocal next_source_id
            ids = list(range(next_source_id, next_source_id + n_rows))
            next_source_id += n_rows
            return ids

        _record_catalog_meta("vlass", vlass_db)
        _record_catalog_meta("first", first_db)
        _record_catalog_meta("nvss", nvss_db)
        _record_catalog_meta("rax", rax_db)

        def _insert_seed_chunk(*, df: pd.DataFrame, cat: str) -> None:
            rows = []
            match_rows = []
            source_ids = _allocate_source_ids(len(df))
            for idx, (_, r) in enumerate(df.iterrows()):
                s = (
                    float(r["flux_mjy"]) / 1000.0
                    if "flux_mjy" in df.columns and not pd.isna(r["flux_mjy"])
                    else None
                )
                source_id = source_ids[idx]
                base = {
                    "source_id": source_id,
                    "ra_deg": float(r["ra_deg"]),
                    "dec_deg": float(r["dec_deg"]),
                    "flux_jy": s,
                    "s_nvss": None,
                    "snr_nvss": None,
                    "s_vlass": None,
                    "s_first": None,
                    "s_rax": None,
                    "alpha": None,
                    "resolved_flag": 0,
                    "confusion_flag": 0,
                    "has_nvss": 0,
                    "has_vlass": 0,
                    "has_first": 0,
                    "has_rax": 0,
                }

                if cat == "vlass":
                    base["s_vlass"] = s
                    base["has_vlass"] = 1
                elif cat == "first":
                    base["s_first"] = s
                    base["has_first"] = 1
                elif cat == "nvss":
                    base["s_nvss"] = s
                    base["has_nvss"] = 1
                elif cat == "rax":
                    base["s_rax"] = s
                    base["has_rax"] = 1

                rows.append(base)
                if with_provenance and "catalog_row_id" in df.columns:
                    match_rows.append(
                        (
                            source_id,
                            cat,
                            int(r["catalog_row_id"]),
                            0.0,
                            0,
                            1,
                        )
                    )

            out_conn.executemany(
                """
                INSERT INTO sources(
                    source_id, ra_deg, dec_deg, flux_jy, s_nvss, snr_nvss, s_vlass, s_first, s_rax,
                    alpha, resolved_flag, confusion_flag, has_nvss, has_vlass, has_first, has_rax
                ) VALUES(
                    :source_id, :ra_deg, :dec_deg, :flux_jy, :s_nvss, :snr_nvss, :s_vlass, :s_first, :s_rax,
                    :alpha, :resolved_flag, :confusion_flag, :has_nvss, :has_vlass, :has_first, :has_rax
                )
                """,
                rows,
            )

            if with_provenance and match_rows:
                out_conn.executemany(
                    """
                    INSERT OR IGNORE INTO catalog_matches(
                        source_id, catalog, catalog_row_id, sep_arcsec, match_rank, is_primary
                    ) VALUES(?, ?, ?, ?, ?, ?)
                    """,
                    match_rows,
                )

        # Helper: match one catalog chunk against existing union, update or insert
        def _merge_catalog_chunk(
            *,
            df: pd.DataFrame,
            cat: str,
            ra_col: str = "ra_deg",
            dec_col: str = "dec_deg",
            flux_mjy_col: str = "flux_mjy",
            extra_cols: dict | None = None,
        ) -> None:
            if extra_cols is None:
                extra_cols = {}

            if len(df) == 0:
                return

            ra_min = float(df[ra_col].min() - match_radius_deg)
            ra_max = float(df[ra_col].max() + match_radius_deg)
            dec_min = float(df[dec_col].min() - match_radius_deg)
            dec_max = float(df[dec_col].max() + match_radius_deg)

            ra_where = _ra_where_clause_local(ra_min, ra_max)
            cand = pd.read_sql_query(
                f"""
                SELECT source_id, ra_deg, dec_deg
                FROM sources
                WHERE dec_deg BETWEEN {dec_min} AND {dec_max}
                  AND {ra_where}
                """,
                out_conn,
            )

            if len(cand) == 0:
                # Everything is new
                rows = []
                match_rows = []
                for _, r in df.iterrows():
                    s = (
                        float(r[flux_mjy_col]) / 1000.0
                        if flux_mjy_col in r and not pd.isna(r[flux_mjy_col])
                        else None
                    )
                    catalog_row_id = None
                    if "catalog_row_id" in df.columns and not pd.isna(r.get("catalog_row_id")):
                        try:
                            catalog_row_id = int(r.get("catalog_row_id"))
                        except Exception:
                            catalog_row_id = None
                    resolved = 0
                    if cat == "first":
                        try:
                            maj = r.get("maj_arcsec")
                            mn = r.get("min_arcsec")
                            maj_val = float(maj) if maj is not None and not pd.isna(maj) else None
                            min_val = float(mn) if mn is not None and not pd.isna(mn) else None
                            if (maj_val is not None and maj_val > 6.0) or (
                                min_val is not None and min_val > 6.0
                            ):
                                resolved = 1
                        except Exception:
                            resolved = 0

                    base = {
                        "ra_deg": float(r[ra_col]),
                        "dec_deg": float(r[dec_col]),
                        "flux_jy": s,
                        "s_nvss": None,
                        "snr_nvss": None,
                        "s_vlass": None,
                        "s_first": None,
                        "s_rax": None,
                        "alpha": None,
                        "resolved_flag": resolved,
                        "confusion_flag": 0,
                        "has_nvss": 0,
                        "has_vlass": 0,
                        "has_first": 0,
                        "has_rax": 0,
                    }
                    if cat == "vlass":
                        base["s_vlass"] = s
                        base["has_vlass"] = 1
                    elif cat == "first":
                        base["s_first"] = s
                        base["has_first"] = 1
                    elif cat == "nvss":
                        base["s_nvss"] = s
                        base["has_nvss"] = 1
                        try:
                            ferr = r.get("flux_err_mjy")
                            ferr_val = (
                                float(ferr) if ferr is not None and not pd.isna(ferr) else None
                            )
                            if ferr_val is not None and ferr_val > 0.0 and s is not None:
                                base["snr_nvss"] = (s * 1000.0) / ferr_val
                        except Exception:
                            base["snr_nvss"] = None
                    elif cat == "rax":
                        base["s_rax"] = s
                        base["has_rax"] = 1

                    base["_catalog_row_id"] = catalog_row_id
                    rows.append(base)

                if rows:
                    insert_ids = _allocate_source_ids(len(rows))
                    for idx, row in enumerate(rows):
                        row["source_id"] = insert_ids[idx]
                        if with_provenance and row.get("_catalog_row_id") is not None:
                            match_rows.append(
                                (
                                    row["source_id"],
                                    cat,
                                    int(row["_catalog_row_id"]),
                                    0.0,
                                    0,
                                    1,
                                )
                            )
                        row.pop("_catalog_row_id", None)

                    out_conn.executemany(
                        """
                        INSERT INTO sources(
                            source_id, ra_deg, dec_deg, flux_jy, s_nvss, snr_nvss, s_vlass, s_first, s_rax,
                            alpha, resolved_flag, confusion_flag, has_nvss, has_vlass, has_first, has_rax
                        ) VALUES(
                            :source_id, :ra_deg, :dec_deg, :flux_jy, :s_nvss, :snr_nvss, :s_vlass, :s_first, :s_rax,
                            :alpha, :resolved_flag, :confusion_flag, :has_nvss, :has_vlass, :has_first, :has_rax
                        )
                        """,
                        rows,
                    )

                    if with_provenance and match_rows:
                        out_conn.executemany(
                            """
                            INSERT OR IGNORE INTO catalog_matches(
                                source_id, catalog, catalog_row_id, sep_arcsec, match_rank, is_primary
                            ) VALUES(?, ?, ?, ?, ?, ?)
                            """,
                            match_rows,
                        )
                return

            df_sc = SkyCoord(
                ra=df[ra_col].values * u.deg, dec=df[dec_col].values * u.deg, frame="icrs"
            )
            cand_sc = SkyCoord(
                ra=cand["ra_deg"].values * u.deg, dec=cand["dec_deg"].values * u.deg, frame="icrs"
            )

            # Map each incoming source to candidate(s)
            idx_df, idx_cand, _, _ = cand_sc.search_around_sky(df_sc, radius)
            cand_by_df: dict[int, list[int]] = {}
            for i_df, i_c in zip(idx_df, idx_cand):
                cand_by_df.setdefault(int(i_df), []).append(int(i_c))

            # Track multiple assignments to same union row in this chunk
            assigned_counts: dict[int, int] = {}

            updates = []
            inserts = []
            match_rows = []

            for i in range(len(df)):
                r = df.iloc[i]
                s = (
                    float(r[flux_mjy_col]) / 1000.0
                    if flux_mjy_col in df.columns and not pd.isna(r[flux_mjy_col])
                    else None
                )
                catalog_row_id = None
                if "catalog_row_id" in df.columns and not pd.isna(r.get("catalog_row_id")):
                    try:
                        catalog_row_id = int(r.get("catalog_row_id"))
                    except Exception:
                        catalog_row_id = None

                cands = cand_by_df.get(i, [])
                if not cands:
                    resolved = 0
                    if cat == "first":
                        try:
                            maj = r.get("maj_arcsec")
                            mn = r.get("min_arcsec")
                            maj_val = float(maj) if maj is not None and not pd.isna(maj) else None
                            min_val = float(mn) if mn is not None and not pd.isna(mn) else None
                            if (maj_val is not None and maj_val > 6.0) or (
                                min_val is not None and min_val > 6.0
                            ):
                                resolved = 1
                        except Exception:
                            resolved = 0

                    base = {
                        "ra_deg": float(r[ra_col]),
                        "dec_deg": float(r[dec_col]),
                        "flux_jy": s,
                        "s_nvss": None,
                        "snr_nvss": None,
                        "s_vlass": None,
                        "s_first": None,
                        "s_rax": None,
                        "alpha": None,
                        "resolved_flag": resolved,
                        "confusion_flag": 0,
                        "has_nvss": 0,
                        "has_vlass": 0,
                        "has_first": 0,
                        "has_rax": 0,
                        "_catalog_row_id": catalog_row_id,
                    }
                    if cat == "vlass":
                        base["s_vlass"] = s
                        base["has_vlass"] = 1
                    elif cat == "first":
                        base["s_first"] = s
                        base["has_first"] = 1
                    elif cat == "nvss":
                        base["s_nvss"] = s
                        base["has_nvss"] = 1
                        try:
                            ferr = r.get("flux_err_mjy")
                            ferr_val = (
                                float(ferr) if ferr is not None and not pd.isna(ferr) else None
                            )
                            if ferr_val is not None and ferr_val > 0.0 and s is not None:
                                base["snr_nvss"] = (s * 1000.0) / ferr_val
                        except Exception:
                            base["snr_nvss"] = None
                    elif cat == "rax":
                        base["s_rax"] = s
                        base["has_rax"] = 1
                    inserts.append(base)
                    continue

                # Compute separations for all candidates
                seps = SkyCoord(
                    ra=float(r[ra_col]) * u.deg, dec=float(r[dec_col]) * u.deg
                ).separation(cand_sc[cands])
                sep_arcsec = seps.to_value(u.arcsec)
                ranked = sorted(zip(sep_arcsec, cands), key=lambda x: x[0])
                confusion = 1 if len(cands) > 1 else 0
                best_c = ranked[0][1]

                union_id = int(cand.iloc[best_c]["source_id"])
                assigned_counts[union_id] = assigned_counts.get(union_id, 0) + 1

                if with_provenance and catalog_row_id is not None:
                    for rank, (sep_val, cand_idx) in enumerate(ranked):
                        match_rows.append(
                            (
                                int(cand.iloc[cand_idx]["source_id"]),
                                cat,
                                catalog_row_id,
                                float(sep_val),
                                rank,
                                1 if rank == 0 else 0,
                            )
                        )

                resolved = None
                if cat == "first":
                    try:
                        maj = r.get("maj_arcsec")
                        mn = r.get("min_arcsec")
                        maj_val = float(maj) if maj is not None and not pd.isna(maj) else None
                        min_val = float(mn) if mn is not None and not pd.isna(mn) else None
                        if (maj_val is not None and maj_val > 6.0) or (
                            min_val is not None and min_val > 6.0
                        ):
                            resolved = 1
                        else:
                            resolved = 0
                    except Exception:
                        resolved = None

                snr = None
                if cat == "nvss":
                    try:
                        ferr = r.get("flux_err_mjy")
                        ferr_val = float(ferr) if ferr is not None and not pd.isna(ferr) else None
                        if ferr_val is not None and ferr_val > 0.0 and s is not None:
                            snr = (s * 1000.0) / ferr_val
                    except Exception:
                        snr = None

                updates.append(
                    {
                        "source_id": union_id,
                        "s": s,
                        "snr": snr,
                        "resolved": resolved,
                        "confusion": confusion,
                    }
                )

            if inserts:
                insert_ids = _allocate_source_ids(len(inserts))
                for idx, row in enumerate(inserts):
                    row["source_id"] = insert_ids[idx]
                    if with_provenance and row.get("_catalog_row_id") is not None:
                        match_rows.append(
                            (
                                row["source_id"],
                                cat,
                                int(row["_catalog_row_id"]),
                                0.0,
                                0,
                                1,
                            )
                        )
                    row.pop("_catalog_row_id", None)
                out_conn.executemany(
                    """
                    INSERT INTO sources(
                        source_id, ra_deg, dec_deg, flux_jy, s_nvss, snr_nvss, s_vlass, s_first, s_rax,
                        alpha, resolved_flag, confusion_flag, has_nvss, has_vlass, has_first, has_rax
                    ) VALUES(
                        :source_id, :ra_deg, :dec_deg, :flux_jy, :s_nvss, :snr_nvss, :s_vlass, :s_first, :s_rax,
                        :alpha, :resolved_flag, :confusion_flag, :has_nvss, :has_vlass, :has_first, :has_rax
                    )
                    """,
                    inserts,
                )

            # Apply updates
            for urow in updates:
                sid = urow["source_id"]
                confusion = 1 if (urow["confusion"] or assigned_counts.get(sid, 0) > 1) else 0

                if cat == "vlass":
                    out_conn.execute(
                        f"""
                        UPDATE sources
                        SET s_vlass = COALESCE(?, s_vlass),
                            has_vlass = 1,
                            confusion_flag = MAX(confusion_flag, ?),
                            {_update_flux_jy_sql()}
                        WHERE source_id = ?
                        """,
                        (urow["s"], confusion, sid),
                    )
                elif cat == "first":
                    out_conn.execute(
                        f"""
                        UPDATE sources
                        SET s_first = COALESCE(?, s_first),
                            has_first = 1,
                            resolved_flag = CASE WHEN ? IS NULL THEN resolved_flag ELSE MAX(resolved_flag, ?) END,
                            confusion_flag = MAX(confusion_flag, ?),
                            {_update_flux_jy_sql()}
                        WHERE source_id = ?
                        """,
                        (urow["s"], urow["resolved"], urow["resolved"], confusion, sid),
                    )
                elif cat == "nvss":
                    out_conn.execute(
                        f"""
                        UPDATE sources
                        SET s_nvss = COALESCE(?, s_nvss),
                            snr_nvss = COALESCE(?, snr_nvss),
                            has_nvss = 1,
                            confusion_flag = MAX(confusion_flag, ?),
                            {_update_flux_jy_sql()}
                        WHERE source_id = ?
                        """,
                        (urow["s"], urow["snr"], confusion, sid),
                    )
                elif cat == "rax":
                    out_conn.execute(
                        f"""
                        UPDATE sources
                        SET s_rax = COALESCE(?, s_rax),
                            has_rax = 1,
                            confusion_flag = MAX(confusion_flag, ?),
                            {_update_flux_jy_sql()}
                        WHERE source_id = ?
                        """,
                        (urow["s"], confusion, sid),
                    )

            if with_provenance and match_rows:
                out_conn.executemany(
                    """
                    INSERT OR IGNORE INTO catalog_matches(
                        source_id, catalog, catalog_row_id, sep_arcsec, match_rank, is_primary
                    ) VALUES(?, ?, ?, ?, ?, ?)
                    """,
                    match_rows,
                )

        # Seed with VLASS (preferred positional anchor when available)
        if vlass_db is not None and Path(vlass_db).exists():
            with sqlite3.connect(str(vlass_db)) as conn:
                cols = _source_columns(conn)
                has_catalog_row_id = "catalog_row_id" in cols
                catalog_id_expr = "catalog_row_id" if has_catalog_row_id else "rowid"
                sql = f"""
                SELECT s.rowid AS rowid,
                       s.{catalog_id_expr} AS catalog_row_id,
                       s.ra_deg, s.dec_deg, s.flux_mjy
                FROM sources s
                WHERE s.rowid > ?
                ORDER BY s.rowid
                LIMIT ?
                """
                total = conn.execute("SELECT COUNT(*) FROM sources").fetchone()[0]
                processed = 0
                for df in _iter_sqlite_chunks(conn, sql, tuple(), chunk_size):
                    _insert_seed_chunk(df=df, cat="vlass")
                    out_conn.commit()
                    processed += len(df)
                    if progress_callback:
                        progress_callback(processed, total)

        # Merge FIRST
        if first_db is not None and Path(first_db).exists():
            with sqlite3.connect(str(first_db)) as conn:
                cols = _source_columns(conn)
                has_catalog_row_id = "catalog_row_id" in cols
                catalog_id_expr = "catalog_row_id" if has_catalog_row_id else "rowid"
                sql = f"""
                SELECT s.rowid AS rowid,
                       s.{catalog_id_expr} AS catalog_row_id,
                       s.ra_deg, s.dec_deg, s.flux_mjy, s.maj_arcsec, s.min_arcsec
                FROM sources s
                WHERE s.rowid > ?
                ORDER BY s.rowid
                LIMIT ?
                """
                total = conn.execute("SELECT COUNT(*) FROM sources").fetchone()[0]
                processed = 0
                for df in _iter_sqlite_chunks(conn, sql, tuple(), chunk_size):
                    _merge_catalog_chunk(df=df, cat="first")
                    out_conn.commit()
                    processed += len(df)
                    if progress_callback:
                        progress_callback(processed, total)

        # Merge NVSS
        if nvss_db is not None and Path(nvss_db).exists():
            with sqlite3.connect(str(nvss_db)) as conn:
                cols = _source_columns(conn)
                has_catalog_row_id = "catalog_row_id" in cols
                catalog_id_expr = "catalog_row_id" if has_catalog_row_id else "rowid"
                sql = f"""
                SELECT s.rowid AS rowid,
                       s.{catalog_id_expr} AS catalog_row_id,
                       s.ra_deg, s.dec_deg, s.flux_mjy, s.flux_err_mjy
                FROM sources s
                WHERE s.rowid > ?
                ORDER BY s.rowid
                LIMIT ?
                """
                total = conn.execute("SELECT COUNT(*) FROM sources").fetchone()[0]
                processed = 0
                for df in _iter_sqlite_chunks(conn, sql, tuple(), chunk_size):
                    _merge_catalog_chunk(df=df, cat="nvss")
                    out_conn.commit()
                    processed += len(df)
                    if progress_callback:
                        progress_callback(processed, total)

        # Merge RAX
        if rax_db is not None and Path(rax_db).exists():
            with sqlite3.connect(str(rax_db)) as conn:
                cols = _source_columns(conn)
                has_catalog_row_id = "catalog_row_id" in cols
                catalog_id_expr = "catalog_row_id" if has_catalog_row_id else "rowid"
                sql = f"""
                SELECT s.rowid AS rowid,
                       s.{catalog_id_expr} AS catalog_row_id,
                       s.ra_deg, s.dec_deg, s.flux_mjy
                FROM sources s
                WHERE s.rowid > ?
                ORDER BY s.rowid
                LIMIT ?
                """
                total = conn.execute("SELECT COUNT(*) FROM sources").fetchone()[0]
                processed = 0
                for df in _iter_sqlite_chunks(conn, sql, tuple(), chunk_size):
                    _merge_catalog_chunk(df=df, cat="rax")
                    out_conn.commit()
                    processed += len(df)
                    if progress_callback:
                        progress_callback(processed, total)

        # Compute spectral index where possible (NVSS vs VLASS) in Python.
        # Avoid relying on SQLite math function availability.
        cur = out_conn.execute(
            """
            SELECT source_id, s_nvss, s_vlass
            FROM sources
            WHERE s_nvss IS NOT NULL AND s_vlass IS NOT NULL
            """
        )
        batch = cur.fetchmany(100_000)
        while batch:
            updates = []
            for source_id, s_nvss, s_vlass in batch:
                alpha = _compute_alpha(float(s_nvss), 1.4e9, float(s_vlass), 3.0e9)
                updates.append((alpha, int(source_id)))
            out_conn.executemany(
                "UPDATE sources SET alpha=? WHERE source_id=?",
                updates,
            )
            batch = cur.fetchmany(100_000)

        # Reference views (still NVSS+VLASS-based by design)
        out_conn.execute("DROP VIEW IF EXISTS good_references")
        out_conn.execute("DROP VIEW IF EXISTS final_references")
        out_conn.execute(
            f"""
            CREATE VIEW good_references AS
            SELECT * FROM sources
            WHERE s_nvss IS NOT NULL
              AND snr_nvss IS NOT NULL AND snr_nvss > {goodref_snr_min}
              AND resolved_flag = 0 AND confusion_flag = 0
              AND alpha IS NOT NULL AND alpha BETWEEN {goodref_alpha_min} AND {goodref_alpha_max}
            """
        )
        out_conn.execute(
            f"""
            CREATE VIEW final_references AS
            SELECT * FROM sources
            WHERE s_nvss IS NOT NULL
              AND snr_nvss IS NOT NULL AND snr_nvss > {finalref_snr_min}
              AND resolved_flag = 0 AND confusion_flag = 0
              AND alpha IS NOT NULL AND alpha BETWEEN {goodref_alpha_min} AND {goodref_alpha_max}
            """
        )

        out_conn.execute(
            "INSERT OR REPLACE INTO meta(key, value) VALUES('build_time_iso', ?)",
            (datetime.now(UTC).isoformat(),),
        )

        out_conn.commit()

    return output_path

def backfill_master_provenance_from_sqlite(
    *,
    master_db: str | Path,
    nvss_db: str | Path | None = None,
    vlass_db: str | Path | None = None,
    first_db: str | Path | None = None,
    rax_db: str | Path | None = None,
    match_radius_arcsec: float = 7.5,
    chunk_size: int = 100_000,
    clear_existing: bool = False,
    progress_callback: callable | None = None,
) -> Path:
    """Backfill provenance tables for an existing master catalog."""
    output_path = Path(master_db)
    if not output_path.exists():
        raise FileNotFoundError(f"Master catalog not found: {output_path}")

    match_radius_deg = match_radius_arcsec / 3600.0
    radius = match_radius_arcsec * u.arcsec

    def _ra_where_clause_local(ra_min: float, ra_max: float) -> str:
        ra_min_mod = ra_min % 360.0
        ra_max_mod = ra_max % 360.0
        if ra_min_mod <= ra_max_mod:
            return f"ra_deg BETWEEN {ra_min_mod} AND {ra_max_mod}"
        return f"(ra_deg >= {ra_min_mod} OR ra_deg <= {ra_max_mod})"

    def _iter_sqlite_chunks(conn: sqlite3.Connection, sql: str, params: tuple, chunk: int):
        last_rowid = 0
        while True:
            df = pd.read_sql_query(
                sql,
                conn,
                params=(last_rowid, *params, chunk),
            )
            if df is None or len(df) == 0:
                break
            last_rowid = int(df["rowid"].max())
            yield df

    def _ensure_provenance_tables(conn: sqlite3.Connection) -> None:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS catalog_matches (
                source_id INTEGER NOT NULL,
                catalog TEXT NOT NULL,
                catalog_row_id INTEGER NOT NULL,
                sep_arcsec REAL,
                match_rank INTEGER DEFAULT 0,
                is_primary INTEGER DEFAULT 0,
                match_version INTEGER DEFAULT 1,
                PRIMARY KEY (source_id, catalog, catalog_row_id)
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_catalog_matches_source ON catalog_matches(source_id)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_catalog_matches_catalog_row ON catalog_matches(catalog, catalog_row_id)"
        )
        cols = {row[1] for row in conn.execute("PRAGMA table_info(catalog_matches)").fetchall()}
        if "match_version" not in cols:
            conn.execute(
                "ALTER TABLE catalog_matches ADD COLUMN match_version INTEGER DEFAULT 1"
            )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS catalogs (
                catalog TEXT PRIMARY KEY,
                source_db_path TEXT,
                source_hash TEXT,
                n_rows INTEGER,
                build_time_iso TEXT,
                raw_rows_format TEXT,
                raw_rows_path TEXT,
                raw_rows_hash TEXT
            )
            """
        )
        cols = {row[1] for row in conn.execute("PRAGMA table_info(catalogs)").fetchall()}
        if "raw_rows_format" not in cols:
            conn.execute("ALTER TABLE catalogs ADD COLUMN raw_rows_format TEXT")
        if "raw_rows_path" not in cols:
            conn.execute("ALTER TABLE catalogs ADD COLUMN raw_rows_path TEXT")
        if "raw_rows_hash" not in cols:
            conn.execute("ALTER TABLE catalogs ADD COLUMN raw_rows_hash TEXT")
        conn.execute(
            "INSERT OR REPLACE INTO meta(key, value) VALUES('schema_version', ?)",
            ("3",),
        )

    def _source_columns(conn: sqlite3.Connection) -> set[str]:
        return {row[1] for row in conn.execute("PRAGMA table_info(sources)").fetchall()}

    def _record_catalog_meta(conn: sqlite3.Connection, catalog: str, db_path: Path | None) -> None:
        if db_path is None or not Path(db_path).exists():
            return
        build_time = None
        n_rows = None
        with sqlite3.connect(str(db_path)) as src_conn:
            try:
                n_rows = src_conn.execute("SELECT COUNT(*) FROM sources").fetchone()[0]
            except Exception:
                n_rows = None
            try:
                row = src_conn.execute(
                    "SELECT value FROM meta WHERE key='build_time_iso'"
                ).fetchone()
                if row:
                    build_time = row[0]
            except Exception:
                build_time = None
            raw_rows_format = None
            raw_rows_path = None
            raw_rows_hash = None
            try:
                row = src_conn.execute(
                    "SELECT value FROM meta WHERE key='raw_rows_format'"
                ).fetchone()
                if row:
                    raw_rows_format = row[0]
            except Exception:
                raw_rows_format = None
            try:
                row = src_conn.execute(
                    "SELECT value FROM meta WHERE key='raw_rows_path'"
                ).fetchone()
                if row:
                    raw_rows_path = row[0]
            except Exception:
                raw_rows_path = None
            try:
                row = src_conn.execute(
                    "SELECT value FROM meta WHERE key='raw_rows_hash'"
                ).fetchone()
                if row:
                    raw_rows_hash = row[0]
            except Exception:
                raw_rows_hash = None
        source_hash = _hash_file(Path(db_path))
        conn.execute(
            """
            INSERT OR REPLACE INTO catalogs(
                catalog, source_db_path, source_hash, n_rows, build_time_iso,
                raw_rows_format, raw_rows_path, raw_rows_hash
            ) VALUES(?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                catalog,
                os.fspath(db_path),
                source_hash,
                n_rows,
                build_time,
                raw_rows_format,
                raw_rows_path,
                raw_rows_hash,
            ),
        )
        conn.execute(
            "INSERT OR REPLACE INTO meta(key, value) VALUES(?, ?)",
            (f"{catalog}_db_hash", source_hash or ""),
        )

    def _backfill_catalog_chunk(
        out_conn: sqlite3.Connection,
        df: pd.DataFrame,
        *,
        cat: str,
        ra_col: str = "ra_deg",
        dec_col: str = "dec_deg",
    ) -> None:
        if len(df) == 0:
            return

        ra_min = float(df[ra_col].min() - match_radius_deg)
        ra_max = float(df[ra_col].max() + match_radius_deg)
        dec_min = float(df[dec_col].min() - match_radius_deg)
        dec_max = float(df[dec_col].max() + match_radius_deg)

        ra_where = _ra_where_clause_local(ra_min, ra_max)
        cand = pd.read_sql_query(
            f"""
            SELECT source_id, ra_deg, dec_deg
            FROM sources
            WHERE dec_deg BETWEEN {dec_min} AND {dec_max}
              AND {ra_where}
            """,
            out_conn,
        )
        if len(cand) == 0:
            return

        df_sc = SkyCoord(
            ra=df[ra_col].values * u.deg, dec=df[dec_col].values * u.deg, frame="icrs"
        )
        cand_sc = SkyCoord(
            ra=cand["ra_deg"].values * u.deg, dec=cand["dec_deg"].values * u.deg, frame="icrs"
        )
        idx_df, idx_cand, _, _ = cand_sc.search_around_sky(df_sc, radius)
        cand_by_df: dict[int, list[int]] = {}
        for i_df, i_c in zip(idx_df, idx_cand):
            cand_by_df.setdefault(int(i_df), []).append(int(i_c))

        match_rows: list[tuple] = []
        for i in range(len(df)):
            cands = cand_by_df.get(i, [])
            if not cands:
                continue
            row = df.iloc[i]
            catalog_row_id = None
            if "catalog_row_id" in df.columns and not pd.isna(row.get("catalog_row_id")):
                try:
                    catalog_row_id = int(row.get("catalog_row_id"))
                except Exception:
                    catalog_row_id = None
            if catalog_row_id is None:
                continue
            seps = SkyCoord(
                ra=float(row[ra_col]) * u.deg, dec=float(row[dec_col]) * u.deg
            ).separation(cand_sc[cands])
            sep_arcsec = seps.to_value(u.arcsec)
            ranked = sorted(zip(sep_arcsec, cands), key=lambda x: x[0])
            for rank, (sep_val, cand_idx) in enumerate(ranked):
                match_rows.append(
                    (
                        int(cand.iloc[cand_idx]["source_id"]),
                        cat,
                        catalog_row_id,
                        float(sep_val),
                        rank,
                        1 if rank == 0 else 0,
                    )
                )

        if match_rows:
            out_conn.executemany(
                """
                INSERT OR IGNORE INTO catalog_matches(
                    source_id, catalog, catalog_row_id, sep_arcsec, match_rank, is_primary
                ) VALUES(?, ?, ?, ?, ?, ?)
                """,
                match_rows,
            )

    with sqlite3.connect(str(output_path)) as out_conn:
        out_conn.execute("PRAGMA journal_mode=WAL")
        _ensure_provenance_tables(out_conn)
        if clear_existing:
            out_conn.execute("DELETE FROM catalog_matches")
            out_conn.execute("DELETE FROM catalogs")
        out_conn.execute(
            "INSERT OR REPLACE INTO meta(key, value) VALUES('build_time_iso', ?)",
            (datetime.now(UTC).isoformat(),),
        )
        out_conn.commit()

        for cat, db_path in [
            ("vlass", vlass_db),
            ("first", first_db),
            ("nvss", nvss_db),
            ("rax", rax_db),
        ]:
            if db_path is None:
                continue
            if not Path(db_path).exists():
                continue
            _record_catalog_meta(out_conn, cat, Path(db_path))

            with sqlite3.connect(str(db_path)) as conn:
                cols = _source_columns(conn)
                has_catalog_row_id = "catalog_row_id" in cols
                catalog_id_expr = "catalog_row_id" if has_catalog_row_id else "rowid"
                sql = f"""
                SELECT s.rowid AS rowid,
                       s.{catalog_id_expr} AS catalog_row_id,
                       s.ra_deg, s.dec_deg
                FROM sources s
                WHERE s.rowid > ?
                ORDER BY s.rowid
                LIMIT ?
                """
                total = conn.execute("SELECT COUNT(*) FROM sources").fetchone()[0]
                processed = 0
                for df in _iter_sqlite_chunks(conn, sql, tuple(), chunk_size):
                    _backfill_catalog_chunk(out_conn, df, cat=cat)
                    out_conn.commit()
                    processed += len(df)
                    if progress_callback:
                        progress_callback(processed, total)

    return output_path


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
