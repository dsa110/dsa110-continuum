# pylint: disable=no-member  # astropy.units uses dynamic attributes (deg, etc.)
import gzip
import logging
import os
from pathlib import Path
from urllib.request import urlretrieve

import astropy.units as u
import numpy as np
import pandas as pd
from astropy.coordinates import Angle, SkyCoord
from astropy.time import Time

from dsa110_contimg.core.calibration.beam_model import (
    BeamConfig,
    primary_beam_response,
)
from dsa110_contimg.common.utils import get_env_path
from dsa110_contimg.common.unified_config import settings


from .schedule import DSA110_LOCATION

logger = logging.getLogger(__name__)

NVSS_URL = "https://heasarc.gsfc.nasa.gov/FTP/heasarc/dbase/tdat_files/heasarc_nvss.tdat.gz"

# FIRST catalog URLs (if available)
# Note: FIRST catalog is typically available as FITS files from NRAO
# Users may need to download manually or provide path
# Example, verify actual URL
FIRST_CATALOG_BASE_URL = "https://third.ucllnl.org/first/catalogs/"

# RAX catalog URL (DSA-110 specific, may need to be provided manually)
# RAX catalog location to be determined based on DSA-110 data access


def resolve_vla_catalog_path(
    explicit_path: str | os.PathLike[str] | None = None, prefer_sqlite: bool = True
) -> Path:
    """Resolve the path to the VLA calibrator catalog.

    This function provides a single source of truth for locating the VLA calibrator catalog.
    SQLite database is the only supported format.

    Resolution order:
    1. Explicit path provided as argument (highest priority)
    2. VLA_CATALOG environment variable
    3. Standard SQLite location: <state_dir>/catalogs/vla_calibrators.sqlite3

    Parameters
    ----------
    explicit_path : Optional[str | os.PathLike[str]]
        Optional explicit path to catalog (overrides all defaults) (default: None)
    prefer_sqlite : bool
        Deprecated, kept for API compatibility (SQLite is always used) (default: True)

    Returns
    -------
    Path
        Path object pointing to the SQLite catalog file

    Raises
    ------
    FileNotFoundError
        If no catalog file can be found

    Examples
    --------
    >>> path = resolve_vla_catalog_path()
    >>> path = resolve_vla_catalog_path("/custom/path/to/catalog.sqlite3")
    """
    import logging

    logger = logging.getLogger(__name__)

    # 1. Explicit path takes highest priority
    if explicit_path:
        path = Path(explicit_path)
        if path.exists():
            return path
        raise FileNotFoundError(f"Explicit catalog path does not exist: {explicit_path}")

    # 1. Environment variable (explicit override)
    env_path = os.getenv("VLA_CATALOG")
    if env_path:
        path = Path(env_path)
        if path.exists():
            logger.info(f"Using VLA catalog from environment variable VLA_CATALOG: {path}")
            return path
        # Fail fast if VLA_CATALOG is explicitly set but path doesn't exist
        raise FileNotFoundError(
            f"VLA_CATALOG env var set but path does not exist: {env_path}"
        )

    # 2. Standard location (state_dir must be configured)
    sqlite_path = settings.paths.state_dir / "catalogs" / "vla_calibrators.sqlite3"
    if sqlite_path.exists():
        if not env_path and not explicit_path:
            logger.info(
                f"Using default VLA catalog path (no VLA_CATALOG env var set): {sqlite_path}"
            )
        return sqlite_path

    raise FileNotFoundError(
        f"VLA calibrator catalog not found at {sqlite_path}. "
        f"Ensure CONTIMG_STATE_DIR is set correctly or use VLA_CATALOG env var."
    )


def validate_vla_catalog() -> list[str]:
    """Validate VLA catalog is accessible and has expected structure.

    Returns
    -------
    list[str]
        List of validation errors (empty if valid)
    """
    errors = []

    try:
        catalog_path = resolve_vla_catalog_path()
    except FileNotFoundError as e:
        return [f"VLA catalog: {e}"]

    # Verify it's a SQLite database with expected tables
    try:
        import sqlite3
        conn = sqlite3.connect(str(catalog_path))
        cursor = conn.cursor()

        # Check for required tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {row[0] for row in cursor.fetchall()}

        required_tables = {"calibrators", "fluxes"}
        missing = required_tables - tables
        if missing:
            errors.append(f"VLA catalog: Missing tables: {missing}")

        # Check calibrators table has data
        if "calibrators" in tables:
            cursor.execute("SELECT COUNT(*) FROM calibrators")
            count = cursor.fetchone()[0]
            if count == 0:
                errors.append("VLA catalog: calibrators table is empty")

        conn.close()
    except Exception as e:
        errors.append(f"VLA catalog: Database validation failed: {e}")

    return errors


def read_nvss_catalog(cache_dir: str = ".cache/catalogs") -> pd.DataFrame:
    """Download (if needed) and parse the NVSS catalog to a DataFrame.

    Returns flux_20_cm in mJy to match historical conventions.

    Parameters
    ----------
    """
    os.makedirs(cache_dir, exist_ok=True)
    gz_path = str(Path(cache_dir) / "heasarc_nvss.tdat.gz")
    txt_path = str(Path(cache_dir) / "heasarc_nvss.tdat")

    if not os.path.exists(txt_path):
        if not os.path.exists(gz_path):
            urlretrieve(NVSS_URL, gz_path)
        with gzip.open(gz_path, "rb") as f_in, open(txt_path, "wb") as f_out:
            f_out.write(f_in.read())

    df = pd.read_csv(
        txt_path,
        sep="|",
        skiprows=67,
        names=[
            "ra",
            "dec",
            "lii",
            "bii",
            "ra_error",
            "dec_error",
            "flux_20_cm",
            "flux_20_cm_error",
            "limit_major_axis",
            "major_axis",
            "major_axis_error",
            "limit_minor_axis",
            "minor_axis",
            "minor_axis_error",
            "position_angle",
            "position_angle_error",
            "residual_code",
            "residual_flux",
            "pol_flux",
            "pol_flux_error",
            "pol_angle",
            "pol_angle_error",
            "field_name",
            "x_pixel",
            "y_pixel",
            "extra",
        ],
    )
    if len(df) > 0:
        df = df.iloc[:-1]  # drop trailer row
    if "extra" in df.columns:
        df = df.drop(columns=["extra"])  # trailing blank
    return df


def read_first_catalog(
    cache_dir: str = ".cache/catalogs",
    first_catalog_path: str | None = None,
) -> pd.DataFrame:
    """Download (if needed) and parse the FIRST catalog to a DataFrame.

    If first_catalog_path is provided, reads from that file directly.
    Otherwise, uses cached file.

    Parameters
    ----------
    cache_dir :
        Directory to cache downloaded catalog files
    first_catalog_path :
        Optional explicit path to FIRST catalog file (CSV/FITS)

    Returns
    -------
        DataFrame with FIRST catalog data

    """
    from dsa110_contimg.core.catalog.build_master import _read_table

    # If explicit path provided, use it directly
    if first_catalog_path:
        if not os.path.exists(first_catalog_path):
            raise FileNotFoundError(f"FIRST catalog file not found: {first_catalog_path}")
        return _read_table(first_catalog_path)

    # Try to find cached FIRST catalog
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = Path(cache_dir) / "first_catalog"

    # Try common extensions (check vizier cache first)
    for ext in [".csv", ".fits", ".fits.gz", ".csv.gz"]:
        cached_file = cache_path.with_suffix(ext)
        if cached_file.exists():
            return _read_table(str(cached_file))

    # Also check for vizier cache (legacy name)
    vizier_cache = Path(cache_dir) / "first_catalog_from_vizier.csv"
    if vizier_cache.exists():
        return _read_table(str(vizier_cache))

    # Try to download from Vizier
    try:
        from dsa110_contimg.core.catalog.download import download_first
        logger.info("Attempting to download FIRST catalog from Vizier...")
        downloaded_path = download_first(cache_dir=cache_dir)
        if downloaded_path and downloaded_path.exists():
            return _read_table(str(downloaded_path))
    except ImportError:
        logger.warning("Could not import download module or astroquery not available")
    except Exception as e:
        logger.warning(f"Failed to download FIRST catalog: {e}")

    raise FileNotFoundError(
        "FIRST catalog not found in cache and automatic download failed. "
        f"Please place the FIRST catalog file at {cache_path}.csv or provide first_catalog_path."
    )


def read_rax_catalog(
    cache_dir: str = ".cache/catalogs",
    rax_catalog_path: str | None = None,
) -> pd.DataFrame:
    """Download (if needed) and parse the RAX catalog to a DataFrame.

    If rax_catalog_path is provided, reads from that file directly.
    Otherwise, attempts to find cached file or raises error.

    Parameters
    ----------
    cache_dir : str
        Directory to cache downloaded catalog files (default: ".cache/catalogs")
    rax_catalog_path : Optional[str]
        Optional explicit path to RAX catalog file (CSV/FITS) (default: None)

    Returns
    -------
    DataFrame
        DataFrame with RAX catalog data

    Notes
    -----
    RAX catalog is DSA-110 specific and not available via Vizier.
    Provide the path manually or ensure the catalog is cached in the cache_dir.
    """
    from dsa110_contimg.core.catalog.build_master import _read_table

    # If explicit path provided, use it directly
    if rax_catalog_path:
        if not os.path.exists(rax_catalog_path):
            raise FileNotFoundError(f"RAX catalog file not found: {rax_catalog_path}")
        return _read_table(rax_catalog_path)

    # Try to find cached RAX catalog
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = Path(cache_dir) / "rax_catalog"

    # Try common extensions
    for ext in [".fits", ".csv", ".fits.gz", ".csv.gz"]:
        cached_file = cache_path.with_suffix(ext)
        if cached_file.exists():
            return _read_table(str(cached_file))

    # Try RACS catalog from Vizier (often used as RAX equivalent/source)
    racs_vizier = Path(cache_dir) / "racs_catalog_from_vizier.csv"
    if racs_vizier.exists():
        return _read_table(str(racs_vizier))

    # Try to download RACS from Vizier
    try:
        from dsa110_contimg.core.catalog.download import download_racs
        logger.info("Attempting to download RACS catalog from Vizier...")
        downloaded_path = download_racs(cache_dir=cache_dir)
        if downloaded_path and downloaded_path.exists():
            return _read_table(str(downloaded_path))
    except ImportError:
        logger.warning("Could not import download module or astroquery not available")
    except Exception as e:
        logger.warning(f"Failed to download RACS catalog: {e}")

    # If not found, raise error with helpful message
    raise FileNotFoundError(
        f"RAX/RACS catalog not found. Please provide path via rax_catalog_path argument, "
        f"or place RAX catalog file in {cache_dir}/rax_catalog.fits or .csv, "
        f"or ensure internet access for automatic RACS download from Vizier."
    )


def read_vlass_catalog(
    cache_dir: str = ".cache/catalogs",
    vlass_catalog_path: str | None = None,
) -> pd.DataFrame:
    """Download (if needed) and parse the VLASS catalog to a DataFrame.

    If vlass_catalog_path is provided, reads from that file directly.
    Otherwise, uses cached file.

    Parameters
    ----------
    cache_dir :
        Directory to cache downloaded catalog files
    vlass_catalog_path :
        Optional explicit path to VLASS catalog file (CSV/FITS)

    Returns
    -------
        DataFrame with VLASS catalog data

    """
    from dsa110_contimg.core.catalog.build_master import _read_table

    # If explicit path provided, use it directly
    if vlass_catalog_path:
        if not os.path.exists(vlass_catalog_path):
            raise FileNotFoundError(f"VLASS catalog file not found: {vlass_catalog_path}")
        return _read_table(vlass_catalog_path)

    # Try to find cached VLASS catalog
    os.makedirs(cache_dir, exist_ok=True)
    Path(cache_dir) / "vlass_catalog"

    # Try common extensions (also check for vizier-cached version)
    for filename in [
        "vlass_catalog_from_vizier.csv",
        "vlass_catalog.csv",
        "vlass_catalog.fits",
        "vlass_catalog.fits.gz",
    ]:
        cached_file = Path(cache_dir) / filename
        if cached_file.exists():
            print(f"Using cached VLASS catalog: {cached_file}")
            return _read_table(str(cached_file))

    # Try to download from Vizier
    try:
        from dsa110_contimg.core.catalog.download import download_vlass
        logger.info("Attempting to download VLASS catalog from Vizier...")
        downloaded_path = download_vlass(cache_dir=cache_dir)
        if downloaded_path and downloaded_path.exists():
            return _read_table(str(downloaded_path))
    except ImportError:
        logger.warning("Could not import download module or astroquery not available")
    except Exception as e:
        logger.warning(f"Failed to download VLASS catalog: {e}")

    # If not found, raise error with helpful message
    raise FileNotFoundError(
        f"VLASS catalog not found. Options:\n"
        f"  1. Provide path via vlass_catalog_path argument\n"
        f"  2. Download VLASS catalog and place it in {cache_dir}/vlass_catalog.csv\n"
        f"VLASS catalog can be obtained from:\n"
        f"  - Vizier: https://vizier.cds.unistra.fr/viz-bin/VizieR?-source=J/ApJS/255/30\n"
        f"  - NRAO: https://archive-new.nrao.edu/vlass/quicklook/"
    )


def read_vla_calibrator_catalog(path: str, cache_dir: str | None = None) -> pd.DataFrame:
    """Parse the NRAO VLA calibrator list from a local text file.

    This follows the structure used in historical VLA calibrator files:
    - A header line per source: "<source> ... <ra> <dec> ..."
      where RA/Dec are sexagesimal strings parseable by astropy Angle.
    - Followed by 4 lines of other metadata.
    - Followed by a block of frequency lines until a blank line; the line
      containing "20cm " includes 4 code tokens and a flux (Jy).

    Parameters
    ----------
    """
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)

    records = []
    with open(path, encoding="utf-8") as f:
        # Skip the first 3 lines if they are header text (as in many files)
        [f.readline() for _ in range(3)]
        # Continue reading entries
        while True:
            line = f.readline()
            if not line:
                break
            if not line.strip():
                continue
            parts = line.split()
            # Expect at least: source, ?, ?, ra, dec
            if len(parts) < 5:
                continue
            try:
                source = parts[0]
                ra_str = parts[3]
                dec_str = parts[4]
                ra_deg = Angle(ra_str).to_value(u.deg)  # pylint: disable=no-member
                dec_deg = Angle(dec_str).to_value(u.deg)  # pylint: disable=no-member
            except Exception:
                continue

            # Skip 4 lines per entry as per file layout
            for _ in range(4):
                _ = f.readline()

            flux_20_cm = None
            code_20_cm = None
            # Read frequency block until blank
            while True:
                f.tell()
                fl = f.readline()
                if (not fl) or fl.isspace():
                    break
                if "20cm " in fl:
                    toks = fl.split()
                    try:
                        # Expected format: "20cm <...> <code_a> <code_b> <code_c> <code_d> <flux> ..."
                        code_a, code_b, code_c, code_d = (
                            toks[2],
                            toks[3],
                            toks[4],
                            toks[5],
                        )
                        flux_20_cm = toks[6]
                        code_20_cm = code_a + code_b + code_c + code_d
                    except Exception:
                        # Fallback: last token as flux
                        flux_20_cm = toks[-1]
                        code_20_cm = None
            # Position now at blank; continue
            if flux_20_cm not in [None, "?"]:
                try:
                    flux_mJy = 1000.0 * float(flux_20_cm)
                except Exception:
                    flux_mJy = np.nan
                records.append(
                    {
                        "source": source,
                        "ra": ra_deg,
                        "dec": dec_deg,
                        "flux_20_cm": flux_mJy,
                        "code_20_cm": code_20_cm,
                    }
                )

    df = pd.DataFrame.from_records(records)
    if not df.empty:
        df = df.set_index("source")
    return df


def load_vla_catalog(
    explicit_path: str | os.PathLike[str] | None = None,
    prefer_sqlite: bool = True,
    band: str = "20cm",
    require_flux: bool = True,
) -> pd.DataFrame:
    """Load the VLA calibrator catalog from SQLite database.

    Parameters
    ----------
    explicit_path : Optional[str | os.PathLike[str]]
        Optional explicit path to catalog (overrides all defaults) (default: None)
    prefer_sqlite : bool
        Deprecated, kept for API compatibility (SQLite is always used) (default: True)
    band : str
        Frequency band to filter by (default: "20cm" for L-band)
    require_flux : bool
        If True (default), only return calibrators with measured flux
        at the specified band. Set False to include all (with caution).

    Returns
    -------
    DataFrame
        DataFrame with calibrator catalog (indexed by name, columns ra_deg, dec_deg, flux_jy)

    Examples
    --------
    >>> df = load_vla_catalog()  # L-band calibrators with documented flux
    >>> df = load_vla_catalog(band="90cm")  # P-band calibrators
    >>> df = load_vla_catalog("/custom/path/to/catalog.sqlite3")
    """
    catalog_path = resolve_vla_catalog_path(explicit_path, prefer_sqlite=prefer_sqlite)
    return load_vla_catalog_from_sqlite(str(catalog_path), band=band, require_flux=require_flux)


def load_vla_catalog_from_sqlite(
    db_path: str, band: str = "20cm", require_flux: bool = True
) -> pd.DataFrame:
    """Load VLA calibrator catalog from SQLite database.

        By default, loads ONLY calibrators with documented flux at the specified band.
        This ensures we don't accidentally use calibrators that are weak or unmeasured
        at our observing frequency.

    Parameters
    ----------
    db_path : str
        Path to SQLite database.
    band : str
        Frequency band to use for flux. Default is "20cm" (L-band).
    require_flux : bool
        If True (default), only include calibrators with measured flux
        at the specified band. If False, include all calibrators with default 1.0 Jy
        flux for those without measurements (use with caution!).

    Returns
    -------
        DataFrame
        DataFrame with calibrator catalog (indexed by name, columns include ra_deg, dec_deg, flux_jy).

    Notes
    -----
        Setting require_flux=False was the cause of selecting 1911+161 (a Q-band
        calibrator with 0.18 Jy at 43 GHz) for L-band calibration where it's likely
        <0.1 Jy. Always use require_flux=True for production calibration.
    """
    import sqlite3

    conn = sqlite3.connect(db_path)
    try:
        if require_flux:
            # INNER JOIN: Only calibrators with documented flux at the specified band
            # This is the safe default - don't use calibrators with unknown flux
            df = pd.read_sql_query(
                """
                SELECT c.name, c.alt_name, c.ra_deg, c.dec_deg,
                       c.position_code, f.flux_jy, f.quality_codes
                FROM calibrators c
                INNER JOIN fluxes f ON c.name = f.name
                WHERE f.band = ?
                """,
                conn,
                params=(band,),
            )
        else:
            # LEFT JOIN: Include all calibrators, defaulting to 1.0 Jy
            # WARNING: This can select calibrators with unknown/low flux at observing band!
            df = pd.read_sql_query(
                """
                SELECT c.name, c.alt_name, c.ra_deg, c.dec_deg,
                       c.position_code, COALESCE(f.flux_jy, 1.0) as flux_jy,
                       f.quality_codes
                FROM calibrators c
                LEFT JOIN fluxes f ON c.name = f.name AND f.band = ?
                """,
                conn,
                params=(band,),
            )

        # CRITICAL: Set index to calibrator name (required by _load_radec)
        # The DataFrame MUST be indexed by calibrator name, not by numeric index
        if "name" in df.columns:
            df = df.set_index("name")
        elif "source_name" in df.columns:
            df = df.set_index("source_name")
        else:
            # If no name column, use first column as index
            df = df.set_index(df.columns[0])

        return df
    finally:
        conn.close()


def read_vla_parsed_catalog_csv(path: str) -> pd.DataFrame:
    """Read a CSV VLA calibrator catalog (parsed) and normalize RA/Dec columns to degrees.

    Expected columns (case-insensitive, best-effort):
    - 'J2000_NAME' (used as index)
    - RA in sexagesimal (e.g., 'RA_J2000') or degrees (e.g., 'RA_deg')
    - DEC in sexagesimal (e.g., 'DEC_J2000') or degrees (e.g., 'DEC_deg')

    Parameters
    ----------
    """
    df = pd.read_csv(path)
    # Identify columns heuristically
    cols = {c.lower(): c for c in df.columns}
    name_col = cols.get("j2000_name") or cols.get("name") or list(df.columns)[0]
    ra_col = cols.get("ra_j2000") or cols.get("ra") or cols.get("raj2000") or cols.get("ra_hms")
    dec_col = cols.get("dec_j2000") or cols.get("dec") or cols.get("dej2000") or cols.get("dec_dms")
    ra_deg_col = next((c for c in df.columns if "ra_deg" in c.lower()), None)
    dec_deg_col = next((c for c in df.columns if "dec_deg" in c.lower()), None)

    def _to_deg(ra_val, dec_val) -> tuple[float, float]:
        import astropy.units as u  # pylint: disable=no-member
        from astropy.coordinates import SkyCoord

        # try sexagesimal first
        try:
            sc = SkyCoord(
                str(ra_val).strip() + " " + str(dec_val).strip(),
                unit=(u.hourangle, u.deg),  # pylint: disable=no-member
                frame="icrs",
            )
            return float(sc.ra.deg), float(sc.dec.deg)
        except Exception:
            try:
                sc = SkyCoord(
                    str(ra_val).strip() + " " + str(dec_val).strip(),
                    unit=(u.deg, u.deg),  # pylint: disable=no-member
                    frame="icrs",
                )
                return float(sc.ra.deg), float(sc.dec.deg)
            except Exception:
                return float("nan"), float("nan")

    out = []
    for _, r in df.iterrows():
        name = str(r.get(name_col, "")).strip()
        if ra_deg_col and dec_deg_col:
            ra_deg = pd.to_numeric(r[ra_deg_col], errors="coerce")
            dec_deg = pd.to_numeric(r[dec_deg_col], errors="coerce")
        elif ra_col and dec_col:
            ra_deg, dec_deg = _to_deg(r.get(ra_col, ""), r.get(dec_col, ""))
        else:
            ra_deg, dec_deg = float("nan"), float("nan")
        out.append({"name": name, "ra_deg": ra_deg, "dec_deg": dec_deg})
    out_df = pd.DataFrame(out).set_index("name")
    return out_df


def read_vla_parsed_catalog_with_flux(path: str, band: str = "20cm") -> pd.DataFrame:
    """Read a parsed VLA calibrator CSV and return RA/Dec in degrees and flux in Jy for a given band.

    Expected columns include J2000_NAME, RA_J2000, DEC_J2000, BAND, FLUX_JY.
    Returns a DataFrame indexed by name with columns ra_deg, dec_deg, flux_jy.

    Parameters
    ----------
    """
    df = pd.read_csv(path)
    # Filter band if present
    if "BAND" in df.columns:
        df = df[df["BAND"].astype(str).str.lower() == band.lower()].copy()
    # Normalize coordinates
    cols = {c.lower(): c for c in df.columns}
    name_col = cols.get("j2000_name") or cols.get("name") or list(df.columns)[0]
    ra_col = cols.get("ra_j2000") or cols.get("ra")
    dec_col = cols.get("dec_j2000") or cols.get("dec")
    # Optional spectral index columns
    sidx_col = cols.get("sidx") or cols.get("spectral_index")
    sidx_f0_col = cols.get("sidx_f0_ghz") or cols.get("nu0") or cols.get("si_freq")
    out = []
    for _, r in df.iterrows():
        name = str(r.get(name_col, "")).strip()
        ra = r.get(ra_col, "")
        dec = r.get(dec_col, "")
        try:
            sc = SkyCoord(
                str(ra).strip() + " " + str(dec).strip(),
                unit=(u.hourangle, u.deg),  # pylint: disable=no-member
                frame="icrs",
            )
            ra_deg = float(sc.ra.deg)
            dec_deg = float(sc.dec.deg)
        except Exception:
            # Try degrees
            try:
                sc = SkyCoord(
                    float(ra) * u.deg,
                    float(dec) * u.deg,
                    frame="icrs",  # pylint: disable=no-member
                )
                ra_deg = float(sc.ra.deg)
                dec_deg = float(sc.dec.deg)
            except Exception:
                continue
        try:
            flux_jy = float(r.get("FLUX_JY", r.get("flux_jy", "nan")))
        except Exception:
            flux_jy = float("nan")
        sidx = None
        sidx_f0_hz = None
        if sidx_col is not None:
            try:
                sidx = float(r.get(sidx_col))
            except Exception:
                sidx = None
        if sidx_f0_col is not None:
            try:
                f0 = float(r.get(sidx_f0_col))
                # If in GHz convert to Hz (assume GHz unless very large)
                sidx_f0_hz = f0 * 1e9 if f0 < 1e6 else f0
            except Exception:
                sidx_f0_hz = None
        out.append(
            {
                "name": name,
                "ra_deg": ra_deg,
                "dec_deg": dec_deg,
                "flux_jy": flux_jy,
                "sidx": sidx,
                "sidx_f0_hz": sidx_f0_hz,
            }
        )
    odf = pd.DataFrame(out).set_index("name")
    return odf


def nearest_calibrator_within_radius(
    pointing_ra_deg: float,
    pointing_dec_deg: float,
    cal_df: pd.DataFrame,
    radius_deg: float,
) -> tuple[str, float, float, float] | None:
    """

    Parameters
    ----------

    Returns
    -------
    type
        cal_df must have columns ra_deg, dec_deg, and optionally flux_jy.

    """
    if cal_df.empty:
        return None
    ra = pd.to_numeric(cal_df["ra_deg"], errors="coerce")
    dec = pd.to_numeric(cal_df["dec_deg"], errors="coerce")
    # Small-angle approximation for speed
    cosd = max(np.cos(np.deg2rad(pointing_dec_deg)), 1e-3)
    sep = np.hypot((ra - pointing_ra_deg) * cosd, (dec - pointing_dec_deg))
    sel = cal_df.copy()
    sel["sep"] = sep
    sel = sel[sel["sep"] <= radius_deg]
    if sel.empty:
        return None
    row = sel.sort_values("sep").iloc[0]
    # Get name from source_name column if available, otherwise use index
    if "source_name" in row.index:
        name = str(row["source_name"])
    else:
        name = str(row.name)
    return (
        name,
        float(row["ra_deg"]),
        float(row["dec_deg"]),
        float(row.get("flux_jy", np.nan)),
    )


def get_calibrator_radec(df: pd.DataFrame, name: str) -> tuple[float, float]:
    """Lookup a calibrator by name (index) and return (ra_deg, dec_deg).

    Parameters
    ----------
    df: pd.DataFrame :

    """
    if name in df.index:
        row = df.loc[name]
        # Handle case where multiple rows match (duplicates) - take first
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        # Handle Series - extract scalar value
        ra_val = row["ra_deg"]
        dec_val = row["dec_deg"]
        if isinstance(ra_val, pd.Series):
            ra_val = ra_val.iloc[0]
        if isinstance(dec_val, pd.Series):
            dec_val = dec_val.iloc[0]
        return float(ra_val), float(dec_val)
    # Fallback: try case-insensitive and stripped on index (name) and alt_name column
    key = name.strip().upper()

    # 1. Check index (names) again case-insensitively
    for idx in df.index:
        if str(idx).strip().upper() == key:
            row = df.loc[idx]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            return float(row["ra_deg"]), float(row["dec_deg"])

    # 2. Check alt_name column if it exists
    if "alt_name" in df.columns:
        # Vectorized check is faster
        mask = df["alt_name"].str.strip().str.upper() == key
        if mask.any():
            row = df[mask].iloc[0]
            return float(row["ra_deg"]), float(row["dec_deg"])

    raise KeyError(f"Calibrator '{name}' not found in catalog")


def generate_caltable(
    vla_df: pd.DataFrame,
    pt_dec: u.Quantity,
    csv_path: str,
    radius: u.Quantity = 2.5 * u.deg,  # pylint: disable=no-member
    min_weighted_flux: u.Quantity = 1.0 * u.Jy,  # pylint: disable=no-member
    min_percent_flux: float = 0.15,
) -> str:
    """Build a declination-specific calibrator table and save to CSV.

    Weighted by primary beam response at 1.4 GHz.

    Parameters
    ----------
    vla_df: pd.DataFrame :

    pt_dec: u.Quantity :

    """
    pt_dec_deg = pt_dec.to_value(u.deg)  # pylint: disable=no-member
    # ensure numeric
    vla_df = vla_df.copy()
    vla_df["ra"] = pd.to_numeric(vla_df["ra"], errors="coerce")
    vla_df["dec"] = pd.to_numeric(vla_df["dec"], errors="coerce")
    vla_df["flux_20_cm"] = pd.to_numeric(vla_df["flux_20_cm"], errors="coerce")

    cal_df = vla_df[
        (vla_df["dec"] < (pt_dec_deg + radius.to_value(u.deg)))  # pylint: disable=no-member
        & (vla_df["dec"] > (pt_dec_deg - radius.to_value(u.deg)))  # pylint: disable=no-member
        & (vla_df["flux_20_cm"] > 1000.0)
    ].copy()
    if cal_df.empty:
        # Still write an empty CSV to satisfy pipeline expectations
        cal_df.to_csv(csv_path, index=True)
        return csv_path

    # Compute weighted flux per calibrator and field flux
    cal_df["weighted_flux"] = 0.0
    cal_df["field_flux"] = 0.0

    ant_ra = 0.0  # use RA=self for beam centering approximation; drop explicit RA dependence
    ant_dec = np.deg2rad(pt_dec_deg)
    for name, row in cal_df.iterrows():
        src_ra = np.deg2rad(row["ra"]) if np.isfinite(row["ra"]) else 0.0
        src_dec = np.deg2rad(row["dec"]) if np.isfinite(row["dec"]) else ant_dec
        config = BeamConfig(
            frequency_ghz=1.4,
            antenna_ra=ant_ra,
            antenna_dec=ant_dec,
            beam_mode="analytic",
            use_docker=False,
        )
        resp = primary_beam_response(src_ra, src_dec, config=config)
        cal_df.at[name, "weighted_flux"] = (row["flux_20_cm"] / 1e3) * resp

        # Field: local patch of radius scaled by cos(dec)
        field = vla_df[
            (vla_df["dec"] < (pt_dec_deg + radius.to_value(u.deg)))  # pylint: disable=no-member
            & (vla_df["dec"] > (pt_dec_deg - radius.to_value(u.deg)))  # pylint: disable=no-member
            & (
                vla_df["ra"]
                < row["ra"]
                + radius.to_value(u.deg)  # pylint: disable=no-member
                / max(np.cos(np.deg2rad(pt_dec_deg)), 1e-3)
            )
            & (
                vla_df["ra"]
                > row["ra"]
                - radius.to_value(u.deg)  # pylint: disable=no-member
                / max(np.cos(np.deg2rad(pt_dec_deg)), 1e-3)
            )
        ].copy()
        wsum = 0.0
        for _, crow in field.iterrows():
            f_ra = np.deg2rad(crow["ra"]) if np.isfinite(crow["ra"]) else 0.0
            f_dec = np.deg2rad(crow["dec"]) if np.isfinite(crow["dec"]) else ant_dec
            config_f = BeamConfig(
                frequency_ghz=1.4,
                antenna_ra=ant_ra,
                antenna_dec=ant_dec,
                beam_mode="analytic",
                use_docker=False,
            )
            wsum += (crow["flux_20_cm"] / 1e3) * primary_beam_response(f_ra, f_dec, config=config_f)
        cal_df.at[name, "field_flux"] = wsum

    cal_df["percent_flux"] = cal_df["weighted_flux"] / cal_df["field_flux"].replace(0, np.nan)

    sel = cal_df[
        (cal_df["weighted_flux"] > min_weighted_flux.to_value(u.Jy))  # pylint: disable=no-member
        & (cal_df["percent_flux"] > min_percent_flux)
    ].copy()

    # Fallback: if selection empty, choose top by weighted flux within dec band
    if sel.empty:
        sel = cal_df.sort_values("weighted_flux", ascending=False).head(10).copy()
        # If any field_flux is zero (rare), set percent_flux=1 for ranking purposes
        z = sel["field_flux"] == 0
        sel.loc[z, "percent_flux"] = 1.0

    # Reformat columns and units
    out = sel.copy()
    out["flux (Jy)"] = out["flux_20_cm"] / 1e3
    out = out.rename(columns={"code_20_cm": "code_20_cm", "ra": "ra(deg)", "dec": "dec(deg)"})
    out = out[
        [
            "ra(deg)",
            "dec(deg)",
            "flux (Jy)",
            "weighted_flux",
            "percent_flux",
            "code_20_cm",
        ]
    ]
    out.to_csv(csv_path, index=True)
    return csv_path


def update_caltable(
    vla_df: pd.DataFrame, pt_dec: u.Quantity, out_dir: str = ".cache/catalogs"
) -> str:
    """Ensure a declination-specific caltable exists; return its path.

    Parameters
    ----------
    vla_df: pd.DataFrame :

    pt_dec: u.Quantity :

    """
    os.makedirs(out_dir, exist_ok=True)
    decsign = "+" if pt_dec.to_value(u.deg) >= 0 else "-"  # pylint: disable=no-member
    decval = f"{abs(pt_dec.to_value(u.deg)):05.1f}".replace(".", "p")  # pylint: disable=no-member
    csv_path = str(Path(out_dir) / f"calibrator_sources_dec{decsign}{decval}.csv")
    if not os.path.exists(csv_path):
        generate_caltable(vla_df=vla_df, pt_dec=pt_dec, csv_path=csv_path)
    return csv_path


def query_nvss_sources(
    ra_deg: float,
    dec_deg: float,
    radius_deg: float,
    min_flux_mjy: float | None = None,
    max_sources: int | None = None,
    catalog_path: str | os.PathLike[str] | None = None,
    auto_regenerate: bool = False,
) -> pd.DataFrame:
    """Query NVSS catalog for sources within a radius using SQLite database.

    This function requires SQLite databases. If a strip database is corrupted,
    it automatically falls back to the full NVSS database. Corrupted strips
    can be regenerated using `regenerate_nvss_strip_db()`.

    Parameters
    ----------
    ra_deg :
        Field center RA in degrees
    dec_deg :
        Field center Dec in degrees
    radius_deg :
        Search radius in degrees
    min_flux_mjy :
        Minimum flux in mJy (optional)
    max_sources :
        Maximum number of sources to return (optional)
    catalog_path :
        Explicit path to SQLite database (overrides auto-resolution)
    auto_regenerate :
        If True, automatically regenerate corrupted strip databases
        (default: False)

    Returns
    -------
    DataFrame with columns
        ra_deg, dec_deg, flux_mjy

    """
    import logging
    import sqlite3

    logger = logging.getLogger(__name__)

    # Try SQLite first (much faster)
    db_path = None

    # 1. Explicit path provided
    if catalog_path:
        db_path = Path(catalog_path)
        if not db_path.exists():
            db_path = None

    # 2. Auto-resolve based on declination strip
    if db_path is None:
        dec_rounded = round(float(dec_deg), 1)
        db_name = f"nvss_dec{dec_rounded:+.1f}.sqlite3"

        # Try standard locations
        candidates = []
        try:
            current_file = Path(__file__).resolve()
            potential_root = current_file.parents[3]
            if (potential_root / "src" / "dsa110_contimg").exists():
                candidates.append(potential_root / "state" / "catalogs" / db_name)
        except Exception:
            pass

        for root_str in ["/data/dsa110-contimg", "/app"]:
            root_path = Path(root_str)
            if root_path.exists():
                candidates.append(root_path / "state" / "catalogs" / db_name)

        candidates.append(Path.cwd() / "state" / "catalogs" / db_name)
        candidates.append(
            get_env_path("CONTIMG_BASE_DIR", default="/data/dsa110-contimg")
            / "state/catalogs"
            / db_name
        )

        for candidate in candidates:
            if candidate.exists():
                db_path = candidate
                break

        # If exact match not found, try to find nearest declination match (within 1.0 degree tolerance)
        if db_path is None:
            catalog_dirs = []
            for root_str in ["/data/dsa110-contimg", "/app"]:
                root_path = Path(root_str)
                if root_path.exists():
                    catalog_dirs.append(root_path / "state" / "catalogs")
            try:
                current_file = Path(__file__).resolve()
                potential_root = current_file.parents[3]
                if (potential_root / "src" / "dsa110_contimg").exists():
                    catalog_dirs.append(potential_root / "state" / "catalogs")
            except Exception:
                pass
            catalog_dirs.append(Path.cwd() / "state" / "catalogs")
            catalog_dirs.append(
                get_env_path("CONTIMG_BASE_DIR", default="/data/dsa110-contimg") / "state/catalogs"
            )

            best_match = None
            best_diff = float("inf")
            for catalog_dir in catalog_dirs:
                if not catalog_dir.exists():
                    continue
                # Find all nvss_dec*.sqlite3 files
                for nvss_file in catalog_dir.glob("nvss_dec*.sqlite3"):
                    try:
                        # Extract declination from filename: nvss_dec+54.6.sqlite3 -> 54.6
                        dec_str = nvss_file.stem.replace("nvss_dec", "").replace("+", "")
                        file_dec = float(dec_str)
                        diff = abs(file_dec - float(dec_deg))
                        if (
                            diff < best_diff and diff <= 6.0
                        ):  # Within 6 degree tolerance (databases cover ±6°)
                            best_diff = diff
                            best_match = nvss_file
                    except (ValueError, AttributeError):
                        continue

            if best_match is not None:
                db_path = best_match

    # Helper function to execute the query on a given database
    def _execute_nvss_query(db_path_arg: Path) -> pd.DataFrame:
        """Execute NVSS query on specified database.

        Parameters
        ----------
        """
        conn = sqlite3.connect(str(db_path_arg))
        conn.row_factory = sqlite3.Row

        try:
            # Approximate box search (faster than exact angular separation)
            # Account for RA wrapping at dec
            cos_dec = max(np.cos(np.radians(dec_deg)), 1e-3)
            ra_half = radius_deg / cos_dec
            dec_half = radius_deg

            # Build query with spatial index
            where_clauses = [
                "ra_deg BETWEEN ? AND ?",
                "dec_deg BETWEEN ? AND ?",
            ]
            params = [
                ra_deg - ra_half,
                ra_deg + ra_half,
                dec_deg - dec_half,
                dec_deg + dec_half,
            ]

            if min_flux_mjy is not None:
                where_clauses.append("flux_mjy >= ?")
                params.append(min_flux_mjy)

            query_sql = f"""
            SELECT ra_deg, dec_deg, flux_mjy
            FROM sources
            WHERE {" AND ".join(where_clauses)}
            ORDER BY flux_mjy DESC
            """

            if max_sources:
                query_sql += f" LIMIT {max_sources}"

            rows = conn.execute(query_sql, params).fetchall()

            if not rows:
                return pd.DataFrame(columns=["ra_deg", "dec_deg", "flux_mjy"])

            return pd.DataFrame([dict(row) for row in rows])
        finally:
            conn.close()

    # Query SQLite if available
    if db_path is not None:
        try:
            df = _execute_nvss_query(db_path)

            # Exact angular separation filter (post-query refinement)
            if len(df) > 0:
                sc = SkyCoord(
                    ra=df["ra_deg"].values * u.deg,  # pylint: disable=no-member
                    dec=df["dec_deg"].values * u.deg,  # pylint: disable=no-member
                    frame="icrs",
                )
                center = SkyCoord(
                    ra_deg * u.deg,
                    dec_deg * u.deg,
                    frame="icrs",  # pylint: disable=no-member,no-member
                )
                sep = sc.separation(center).deg
                df = df[sep <= radius_deg].copy()

                # Re-apply flux filter if needed (for exact separation)
                if min_flux_mjy is not None and len(df) > 0:
                    df = df[df["flux_mjy"] >= min_flux_mjy].copy()

                # Re-apply limit if needed
                if max_sources and len(df) > max_sources:
                    df = df.head(max_sources)

            return df

        except sqlite3.DatabaseError as db_err:
            # Database corruption - try the full database as fallback
            corrupted_db_path = db_path
            logger.warning(
                f"Database error with {db_path.name}: {db_err}. Trying full database fallback."
            )

            # Look for nvss_full.sqlite3 in catalog directories
            full_db_path = None
            catalog_dirs = [
                get_env_path("CONTIMG_BASE_DIR", default="/data/dsa110-contimg") / "state/catalogs",
                Path("/app/state/catalogs"),
                Path.cwd() / "state" / "catalogs",
            ]
            try:
                current_file = Path(__file__).resolve()
                potential_root = current_file.parents[3]
                if (potential_root / "src" / "dsa110_contimg").exists():
                    catalog_dirs.insert(0, potential_root / "state" / "catalogs")
            except Exception:
                pass

            for catalog_dir in catalog_dirs:
                candidate = catalog_dir / "nvss_full.sqlite3"
                if candidate.exists():
                    full_db_path = candidate
                    break

            if full_db_path is not None and full_db_path != db_path:
                try:
                    logger.info(f"Falling back to full database: {full_db_path.name}")
                    df = _execute_nvss_query(full_db_path)

                    # Apply separation filter
                    if len(df) > 0:
                        sc = SkyCoord(
                            ra=df["ra_deg"].values * u.deg,
                            dec=df["dec_deg"].values * u.deg,
                            frame="icrs",
                        )
                        center = SkyCoord(ra_deg * u.deg, dec_deg * u.deg, frame="icrs")
                        sep = sc.separation(center).deg
                        df = df[sep <= radius_deg].copy()

                        if min_flux_mjy is not None and len(df) > 0:
                            df = df[df["flux_mjy"] >= min_flux_mjy].copy()

                        if max_sources and len(df) > max_sources:
                            df = df.head(max_sources)

                    # Auto-regenerate the corrupted strip if requested
                    if auto_regenerate and corrupted_db_path is not None:
                        try:
                            from dsa110_contimg.core.catalog.builders import regenerate_nvss_strip_db

                            dec_str = corrupted_db_path.stem.replace("nvss_dec", "")
                            dec_center = float(dec_str)
                            logger.info(
                                f"Auto-regenerating corrupted strip database for Dec {dec_center}..."
                            )
                            regenerate_nvss_strip_db(dec_center, force=True)
                        except Exception as regen_err:
                            logger.warning(f"Auto-regeneration failed: {regen_err}")

                    return df
                except Exception as full_err:
                    logger.warning(f"Full database also failed: {full_err}")

            # No fallback available
            logger.error(
                f"SQLite query failed ({db_err}). "
                f"Corrupted strip database: {corrupted_db_path.name}. "
                f"Regenerate with: from dsa110_contimg.core.catalog import regenerate_nvss_strip_db; "
                f"regenerate_nvss_strip_db({dec_deg:.1f})"
            )
            return pd.DataFrame(columns=["ra_deg", "dec_deg", "flux_mjy"])

        except Exception as e:
            # Other SQLite query failures
            logger.error(
                f"SQLite query failed ({e}). "
                f"Try regenerating the database: from dsa110_contimg.core.catalog import regenerate_nvss_strip_db; "
                f"regenerate_nvss_strip_db({dec_deg:.1f})"
            )
            return pd.DataFrame(columns=["ra_deg", "dec_deg", "flux_mjy"])

    # No SQLite database found
    if db_path is None:
        logger.error(
            f"NVSS SQLite database not found for Dec {dec_deg:.1f}. "
            f"Build with: from dsa110_contimg.core.catalog import build_nvss_strip_db; "
            f"build_nvss_strip_db({dec_deg:.1f})"
        )
        return pd.DataFrame(columns=["ra_deg", "dec_deg", "flux_mjy"])

    # Should not reach here, but return empty dataframe just in case
    return pd.DataFrame(columns=["ra_deg", "dec_deg", "flux_mjy"])


def query_first_sources(
    ra_deg: float,
    dec_deg: float,
    radius_deg: float,
    min_flux_mjy: float | None = None,
    max_sources: int | None = None,
    catalog_path: str | os.PathLike[str] | None = None,
    use_csv_fallback: bool = False,
) -> pd.DataFrame:
    """Query FIRST catalog for sources within a radius using SQLite database.

    FIRST (Faint Images of the Radio Sky at Twenty cm) provides higher resolution
    (5" beam) than NVSS (45" beam) at 1.4 GHz.

    This function requires SQLite databases for optimal performance (~170× faster than CSV).
    CSV fallback is available but disabled by default. Set use_csv_fallback=True to enable.

    Parameters
    ----------
    ra_deg :
        Field center RA in degrees
    dec_deg :
        Field center Dec in degrees
    radius_deg :
        Search radius in degrees
    min_flux_mjy :
        Minimum flux in mJy (optional)
    max_sources :
        Maximum number of sources to return (optional)
    catalog_path :
        Explicit path to SQLite database (overrides auto-resolution)
    use_csv_fallback :
        If True, fall back to CSV when SQLite fails (default: False)

    Returns
    -------
    DataFrame with columns
        ra_deg, dec_deg, flux_mjy

    """
    import logging
    import sqlite3

    logger = logging.getLogger(__name__)

    # Try SQLite first (much faster)
    db_path = None

    # 1. Explicit path provided
    if catalog_path:
        db_path = Path(catalog_path)
        if not db_path.exists():
            db_path = None

    # 2. Auto-resolve based on declination strip
    if db_path is None:
        dec_rounded = round(float(dec_deg), 1)
        db_name = f"first_dec{dec_rounded:+.1f}.sqlite3"

        # Try standard locations
        candidates = []
        try:
            current_file = Path(__file__).resolve()
            potential_root = current_file.parents[3]
            if (potential_root / "src" / "dsa110_contimg").exists():
                candidates.append(potential_root / "state" / "catalogs" / db_name)
        except Exception:
            pass

        for root_str in ["/data/dsa110-contimg", "/app"]:
            root_path = Path(root_str)
            if root_path.exists():
                candidates.append(root_path / "state" / "catalogs" / db_name)

        candidates.append(Path.cwd() / "state" / "catalogs" / db_name)
        candidates.append(
            get_env_path("CONTIMG_BASE_DIR", default="/data/dsa110-contimg")
            / "state/catalogs"
            / db_name
        )

        for candidate in candidates:
            if candidate.exists():
                db_path = candidate
                break

        # If exact match not found, try to find nearest declination match
        if db_path is None:
            catalog_dirs = []
            for root_str in ["/data/dsa110-contimg", "/app"]:
                root_path = Path(root_str)
                if root_path.exists():
                    catalog_dirs.append(root_path / "state" / "catalogs")
            try:
                current_file = Path(__file__).resolve()
                potential_root = current_file.parents[3]
                if (potential_root / "src" / "dsa110_contimg").exists():
                    catalog_dirs.append(potential_root / "state" / "catalogs")
            except Exception:
                pass
            catalog_dirs.append(Path.cwd() / "state" / "catalogs")
            catalog_dirs.append(
                get_env_path("CONTIMG_BASE_DIR", default="/data/dsa110-contimg") / "state/catalogs"
            )

            best_match = None
            best_diff = float("inf")
            for catalog_dir in catalog_dirs:
                if not catalog_dir.exists():
                    continue
                for first_file in catalog_dir.glob("first_dec*.sqlite3"):
                    try:
                        dec_str = first_file.stem.replace("first_dec", "").replace("+", "")
                        file_dec = float(dec_str)
                        diff = abs(file_dec - float(dec_deg))
                        if diff < best_diff and diff <= 6.0:
                            best_diff = diff
                            best_match = first_file
                    except (ValueError, IndexError):
                        continue

            if best_match:
                db_path = best_match
                logger.info(
                    f"Using nearest FIRST database: {best_match.name} (Δdec={best_diff:.1f}°)"
                )

    # Query SQLite database if found
    if db_path and db_path.exists():
        try:
            conn = sqlite3.connect(str(db_path))
            query = """
                SELECT ra_deg, dec_deg, flux_mjy
                FROM sources
                WHERE ra_deg BETWEEN ? AND ?
                  AND dec_deg BETWEEN ? AND ?
            """
            params = [
                ra_deg - radius_deg,
                ra_deg + radius_deg,
                dec_deg - radius_deg,
                dec_deg + radius_deg,
            ]

            if min_flux_mjy is not None:
                query += " AND flux_mjy >= ?"
                params.append(float(min_flux_mjy))

            query += " ORDER BY flux_mjy DESC"

            if max_sources is not None:
                query += " LIMIT ?"
                params.append(int(max_sources))

            df = pd.read_sql_query(query, conn, params=params)
            conn.close()

            # Filter by actual circle (not just box)
            if len(df) > 0:
                # Compute angular separation using haversine formula
                ra1_rad = np.radians(ra_deg)
                dec1_rad = np.radians(dec_deg)
                ra2_rad = np.radians(df["ra_deg"].values)
                dec2_rad = np.radians(df["dec_deg"].values)

                delta_ra = ra2_rad - ra1_rad
                delta_dec = dec2_rad - dec1_rad

                a = (
                    np.sin(delta_dec / 2) ** 2
                    + np.cos(dec1_rad) * np.cos(dec2_rad) * np.sin(delta_ra / 2) ** 2
                )
                c = 2 * np.arcsin(np.sqrt(a))
                distances = np.degrees(c)

                df = df[distances <= radius_deg].copy()

            return df

        except Exception as e:
            logger.warning(f"FIRST SQLite query failed: {e}")
            if not use_csv_fallback:
                logger.warning(
                    "Returning empty catalog. "
                    "Set use_csv_fallback=True to enable CSV fallback (slower, ~1s vs ~0.01s)."
                )
                return pd.DataFrame(columns=["ra_deg", "dec_deg", "flux_mjy"])

    # Fall back to CSV if requested
    if use_csv_fallback:
        logger.warning("FIRST SQLite not available, falling back to CSV query (slow)")
        # CSV fallback not implemented for FIRST yet
        return pd.DataFrame(columns=["ra_deg", "dec_deg", "flux_mjy"])

    # If we get here, no database found and fallback disabled
    logger.warning(
        f"FIRST database not found for dec={dec_deg:.1f}°. "
        "Set use_csv_fallback=True to enable CSV fallback (slower, ~1s vs ~0.01s)."
    )
    return pd.DataFrame(columns=["ra_deg", "dec_deg", "flux_mjy"])


def query_merged_nvss_first_sources(
    ra_deg: float,
    dec_deg: float,
    radius_deg: float,
    min_flux_mjy: float | None = None,
    max_sources: int | None = None,
    match_radius_arcsec: float = 10.0,
) -> pd.DataFrame:
    """Query and merge NVSS + FIRST catalogs, deduplicating by position.

    Queries both NVSS (1.4 GHz, 45" beam) and FIRST (1.4 GHz, 5" beam) catalogs
    and merges them, keeping the higher-resolution FIRST flux when sources match.

    Parameters
    ----------
    ra_deg :
        Field center RA in degrees
    dec_deg :
        Field center Dec in degrees
    radius_deg :
        Search radius in degrees
    min_flux_mjy :
        Minimum flux in mJy (applied to both catalogs)
    max_sources :
        Maximum number of sources to return (after merging)
    match_radius_arcsec :
        Matching radius in arcseconds for deduplication (default: 10")

    Returns
    -------
    DataFrame with columns
        ra_deg, dec_deg, flux_mjy, catalog (source: 'nvss', 'first', or 'both')

    """
    import logging

    logger = logging.getLogger(__name__)

    # Query both catalogs
    nvss_df = query_nvss_sources(
        ra_deg=ra_deg,
        dec_deg=dec_deg,
        radius_deg=radius_deg,
        min_flux_mjy=min_flux_mjy,
        max_sources=None,  # Don't limit yet, will limit after merge
    )

    first_df = query_first_sources(
        ra_deg=ra_deg,
        dec_deg=dec_deg,
        radius_deg=radius_deg,
        min_flux_mjy=min_flux_mjy,
        max_sources=None,
    )

    logger.info(f"Found {len(nvss_df)} NVSS sources, {len(first_df)} FIRST sources")

    # If one catalog is empty, return the other
    if len(nvss_df) == 0 and len(first_df) == 0:
        return pd.DataFrame(columns=["ra_deg", "dec_deg", "flux_mjy", "catalog"])
    if len(nvss_df) == 0:
        first_df["catalog"] = "first"
        if max_sources:
            first_df = first_df.head(max_sources)
        return first_df
    if len(first_df) == 0:
        nvss_df["catalog"] = "nvss"
        if max_sources:
            nvss_df = nvss_df.head(max_sources)
        return nvss_df

    # Import astropy for efficient matching
    from astropy.coordinates import SkyCoord
    from astropy import units as u

    # Create SkyCoord objects
    nvss_coords = SkyCoord(
        ra=nvss_df["ra_deg"].values * u.deg,
        dec=nvss_df["dec_deg"].values * u.deg,
    )
    first_coords = SkyCoord(
        ra=first_df["ra_deg"].values * u.deg,
        dec=first_df["dec_deg"].values * u.deg,
    )

    match_radius_deg = match_radius_arcsec / 3600.0 * u.deg

    # 1. Match FIRST -> NVSS to tag FIRST sources
    idx_nvss, sep2d_first, _ = first_coords.match_to_catalog_sky(nvss_coords)
    matched_to_nvss = sep2d_first <= match_radius_deg

    first_df = first_df.copy()
    first_df["catalog"] = "first"
    first_df.loc[matched_to_nvss, "catalog"] = "both"

    # 2. Match NVSS -> FIRST to find unmatched NVSS sources
    # If an NVSS source matches ANY FIRST source, it is duplicate/lower-res and should be dropped
    idx_first, sep2d_nvss, _ = nvss_coords.match_to_catalog_sky(first_coords)
    matched_to_first = sep2d_nvss <= match_radius_deg

    # Keep only NVSS sources that are NOT matched to FIRST
    unmatched_nvss_df = nvss_df[~matched_to_first].copy()
    unmatched_nvss_df["catalog"] = "nvss"

    # Merge
    merged_df = pd.concat([first_df, unmatched_nvss_df], ignore_index=True)

    # Sort by flux (descending) and apply max_sources limit
    merged_df = merged_df.sort_values("flux_mjy", ascending=False)
    if max_sources:
        merged_df = merged_df.head(max_sources)

    logger.info(
        f"Merged catalog: {len(merged_df)} total sources "
        f"({sum(merged_df['catalog'] == 'both')} matched, "
        f"{sum(merged_df['catalog'] == 'first')} FIRST-only, "
        f"{sum(merged_df['catalog'] == 'nvss')} NVSS-only)"
    )

    return merged_df.reset_index(drop=True)


def query_rax_sources(
    ra_deg: float,
    dec_deg: float,
    radius_deg: float,
    min_flux_mjy: float | None = None,
    max_sources: int | None = None,
    catalog_path: str | os.PathLike[str] | None = None,
    use_csv_fallback: bool = False,
) -> pd.DataFrame:
    """Query RACS/RAX catalog for sources within a radius using SQLite database.

    This function requires SQLite databases for optimal performance (~170× faster than CSV).
    CSV fallback is available but disabled by default. Set use_csv_fallback=True to enable.

    Parameters
    ----------
    ra_deg :
        Field center RA in degrees
    dec_deg :
        Field center Dec in degrees
    radius_deg :
        Search radius in degrees
    min_flux_mjy :
        Minimum flux in mJy (optional)
    max_sources :
        Maximum number of sources to return (optional)
    catalog_path :
        Explicit path to SQLite database (overrides auto-resolution)
    use_csv_fallback :
        If True, fall back to CSV when SQLite fails (default: False)

    Returns
    -------
    DataFrame with columns
        ra_deg, dec_deg, flux_mjy

    """
    import sqlite3

    # Try SQLite first (much faster)
    db_path = None

    # 1. Explicit path provided
    if catalog_path:
        db_path = Path(catalog_path)
        if not db_path.exists():
            db_path = None

    # 2. Auto-resolve based on declination strip
    if db_path is None:
        dec_rounded = round(float(dec_deg), 1)
        db_name = f"rax_dec{dec_rounded:+.1f}.sqlite3"

        # Try standard locations
        candidates = []
        try:
            current_file = Path(__file__).resolve()
            potential_root = current_file.parents[3]
            if (potential_root / "src" / "dsa110_contimg").exists():
                candidates.append(potential_root / "state" / "catalogs" / db_name)
        except Exception:
            pass

        for root_str in ["/data/dsa110-contimg", "/app"]:
            root_path = Path(root_str)
            if root_path.exists():
                candidates.append(root_path / "state" / "catalogs" / db_name)

        candidates.append(Path.cwd() / "state" / "catalogs" / db_name)
        candidates.append(
            get_env_path("CONTIMG_BASE_DIR", default="/data/dsa110-contimg")
            / "state/catalogs"
            / db_name
        )

        for candidate in candidates:
            if candidate.exists():
                db_path = candidate
                break

    # Query SQLite if available
    if db_path is not None:
        try:
            conn = sqlite3.connect(str(db_path))
            conn.row_factory = sqlite3.Row

            try:
                # Approximate box search (faster than exact angular separation)
                cos_dec = max(np.cos(np.radians(dec_deg)), 1e-3)
                ra_half = radius_deg / cos_dec
                dec_half = radius_deg

                # Build query with spatial index
                where_clauses = [
                    "ra_deg BETWEEN ? AND ?",
                    "dec_deg BETWEEN ? AND ?",
                ]
                params = [
                    ra_deg - ra_half,
                    ra_deg + ra_half,
                    dec_deg - dec_half,
                    dec_deg + dec_half,
                ]

                if min_flux_mjy is not None:
                    where_clauses.append("flux_mjy >= ?")
                    params.append(min_flux_mjy)

                query = f"""
                SELECT ra_deg, dec_deg, flux_mjy
                FROM sources
                WHERE {" AND ".join(where_clauses)}
                ORDER BY flux_mjy DESC
                """

                if max_sources:
                    query += f" LIMIT {max_sources}"

                rows = conn.execute(query, params).fetchall()

                if not rows:
                    return pd.DataFrame(columns=["ra_deg", "dec_deg", "flux_mjy"])

                df = pd.DataFrame([dict(row) for row in rows])

                # Exact angular separation filter (post-query refinement)
                if len(df) > 0:
                    sc = SkyCoord(
                        ra=df["ra_deg"].values * u.deg,  # pylint: disable=no-member
                        dec=df["dec_deg"].values * u.deg,  # pylint: disable=no-member
                        frame="icrs",
                    )
                    center = SkyCoord(
                        ra_deg * u.deg,
                        dec_deg * u.deg,
                        frame="icrs",  # pylint: disable=no-member,no-member
                    )
                    sep = sc.separation(center).deg
                    df = df[sep <= radius_deg].copy()

                    # Re-apply flux filter if needed
                    if min_flux_mjy is not None and len(df) > 0:
                        df = df[df["flux_mjy"] >= min_flux_mjy].copy()

                    # Re-apply limit if needed
                    if max_sources and len(df) > max_sources:
                        df = df.head(max_sources)

                return df

            finally:
                conn.close()

        except Exception as e:
            # SQLite query failed
            if use_csv_fallback:
                print(
                    "Note: CSV catalog is available as an alternative. "
                    "Set use_csv_fallback=True to enable CSV fallback (slower, ~1s vs ~0.01s)."
                )
                logger.warning(
                    f"SQLite query failed ({e}), falling back to CSV. "
                    f"This will be slower (~1s vs ~0.01s)."
                )
                # Fallback to CSV (slower but always works)
                df_full = read_rax_catalog()

                # Normalize column names for RAX
                from dsa110_contimg.core.catalog.build_master import _normalize_columns

                RAX_CANDIDATES = {
                    "ra": ["ra", "ra_deg", "raj2000", "ra_hms"],
                    "dec": ["dec", "dec_deg", "dej2000", "dec_dms"],
                    "flux": [
                        "flux",
                        "flux_mjy",
                        "flux_jy",
                        "peak_flux",
                        "fpeak",
                        "s1.4",
                    ],
                }
                col_map = _normalize_columns(df_full, RAX_CANDIDATES)

                ra_col = col_map.get("ra", "ra")
                dec_col = col_map.get("dec", "dec")
                flux_col = col_map.get("flux", None)

                # Convert to SkyCoord for separation calculation
                ra_vals = pd.to_numeric(df_full[ra_col], errors="coerce")
                dec_vals = pd.to_numeric(df_full[dec_col], errors="coerce")
                sc = SkyCoord(
                    ra=ra_vals.values * u.deg,  # pylint: disable=no-member
                    dec=dec_vals.values * u.deg,
                    frame="icrs",
                )  # pylint: disable=no-member
                center = SkyCoord(
                    ra_deg * u.deg,
                    dec_deg * u.deg,
                    frame="icrs",  # pylint: disable=no-member
                )  # pylint: disable=no-member
                sep = sc.separation(center).deg

                # Filter by separation
                keep = sep <= radius_deg

                # Filter by flux if specified
                if min_flux_mjy is not None and flux_col:
                    flux_vals = pd.to_numeric(df_full[flux_col], errors="coerce")
                    # Convert to mJy if needed (assume > 1000 means Jy)
                    if len(flux_vals) > 0 and flux_vals.max() > 1000:
                        flux_vals = flux_vals * 1000.0
                    keep = keep & (flux_vals >= min_flux_mjy)

                result = df_full[keep].copy()

                # Standardize column names
                result["ra_deg"] = pd.to_numeric(result[ra_col], errors="coerce")
                result["dec_deg"] = pd.to_numeric(result[dec_col], errors="coerce")

                if flux_col and flux_col in result.columns:
                    flux_vals = pd.to_numeric(result[flux_col], errors="coerce")
                    if len(flux_vals) > 0 and flux_vals.max() > 1000:
                        result["flux_mjy"] = flux_vals * 1000.0
                    else:
                        result["flux_mjy"] = flux_vals
                else:
                    result["flux_mjy"] = None

                # Sort by flux and limit
                if "flux_mjy" in result.columns and result["flux_mjy"].notna().any():
                    result = result.sort_values("flux_mjy", ascending=False, na_position="last")
                if max_sources:
                    result = result.head(max_sources)

                # Select only the columns we need
                if len(result) > 0:
                    result = result[["ra_deg", "dec_deg", "flux_mjy"]].copy()
                else:
                    result = pd.DataFrame(columns=["ra_deg", "dec_deg", "flux_mjy"])

                return result
            else:
                # No fallback - return empty DataFrame
                logger.error(
                    f"SQLite query failed ({e}). "
                    f"SQLite database required. CSV fallback is available but disabled. "
                    f"Set use_csv_fallback=True to enable CSV fallback."
                )
                print(
                    "Note: CSV catalog is available as an alternative. "
                    "Set use_csv_fallback=True to enable CSV fallback (slower, ~1s vs ~0.01s)."
                )
                return pd.DataFrame(columns=["ra_deg", "dec_deg", "flux_mjy"])

    # No SQLite database found and CSV fallback disabled - FATAL ERROR
    if db_path is None:
        error_msg = (
            f"RAX/RACS SQLite database not found for dec={dec_deg:.1f}°. "
            "SQLite database is required for catalog queries. "
            "Either: (1) Build the database using catalog build tools, or "
            "(2) Set use_csv_fallback=True to enable slower CSV fallback."
        )
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)


def query_vlass_sources(
    ra_deg: float,
    dec_deg: float,
    radius_deg: float,
    min_flux_mjy: float | None = None,
    max_sources: int | None = None,
    catalog_path: str | os.PathLike[str] | None = None,
    use_csv_fallback: bool = False,
) -> pd.DataFrame:
    """Query VLASS catalog for sources within a radius using SQLite database.

    This function requires SQLite databases for optimal performance (~170× faster than CSV).
    CSV fallback is available but disabled by default. Set use_csv_fallback=True to enable.

    Parameters
    ----------
    ra_deg :
        Field center RA in degrees
    dec_deg :
        Field center Dec in degrees
    radius_deg :
        Search radius in degrees
    min_flux_mjy :
        Minimum flux in mJy (optional)
    max_sources :
        Maximum number of sources to return (optional)
    catalog_path :
        Explicit path to SQLite database (overrides auto-resolution)
    use_csv_fallback :
        If True, fall back to CSV when SQLite fails (default: False)

    Returns
    -------
    DataFrame with columns
        ra_deg, dec_deg, flux_mjy

    """
    import sqlite3

    # Try SQLite first (much faster)
    db_path = None

    # 1. Explicit path provided
    if catalog_path:
        db_path = Path(catalog_path)
        if not db_path.exists():
            db_path = None

    # 2. Auto-resolve based on declination strip
    if db_path is None:
        dec_rounded = round(float(dec_deg), 1)
        db_name = f"vlass_dec{dec_rounded:+.1f}.sqlite3"

        # Try standard locations
        candidates = []
        try:
            current_file = Path(__file__).resolve()
            potential_root = current_file.parents[3]
            if (potential_root / "src" / "dsa110_contimg").exists():
                candidates.append(potential_root / "state" / "catalogs" / db_name)
        except Exception:
            pass

        for root_str in ["/data/dsa110-contimg", "/app"]:
            root_path = Path(root_str)
            if root_path.exists():
                candidates.append(root_path / "state" / "catalogs" / db_name)

        candidates.append(Path.cwd() / "state" / "catalogs" / db_name)
        candidates.append(
            get_env_path("CONTIMG_BASE_DIR", default="/data/dsa110-contimg")
            / "state/catalogs"
            / db_name
        )

        for candidate in candidates:
            if candidate.exists():
                db_path = candidate
                break

    # Query SQLite if available
    if db_path is not None:
        try:
            conn = sqlite3.connect(str(db_path))
            conn.row_factory = sqlite3.Row

            try:
                # Approximate box search (faster than exact angular separation)
                cos_dec = max(np.cos(np.radians(dec_deg)), 1e-3)
                ra_half = radius_deg / cos_dec
                dec_half = radius_deg

                # Build query with spatial index
                where_clauses = [
                    "ra_deg BETWEEN ? AND ?",
                    "dec_deg BETWEEN ? AND ?",
                ]
                params = [
                    ra_deg - ra_half,
                    ra_deg + ra_half,
                    dec_deg - dec_half,
                    dec_deg + dec_half,
                ]

                if min_flux_mjy is not None:
                    where_clauses.append("flux_mjy >= ?")
                    params.append(min_flux_mjy)

                query = f"""
                SELECT ra_deg, dec_deg, flux_mjy
                FROM sources
                WHERE {" AND ".join(where_clauses)}
                ORDER BY flux_mjy DESC
                """

                if max_sources:
                    query += f" LIMIT {max_sources}"

                rows = conn.execute(query, params).fetchall()

                if not rows:
                    return pd.DataFrame(columns=["ra_deg", "dec_deg", "flux_mjy"])

                df = pd.DataFrame([dict(row) for row in rows])

                # Exact angular separation filter (post-query refinement)
                if len(df) > 0:
                    sc = SkyCoord(
                        ra=df["ra_deg"].values * u.deg,  # pylint: disable=no-member
                        dec=df["dec_deg"].values * u.deg,  # pylint: disable=no-member
                        frame="icrs",
                    )
                    center = SkyCoord(
                        ra_deg * u.deg,
                        dec_deg * u.deg,
                        frame="icrs",  # pylint: disable=no-member,no-member
                    )
                    sep = sc.separation(center).deg
                    df = df[sep <= radius_deg].copy()

                    # Re-apply flux filter if needed
                    if min_flux_mjy is not None and len(df) > 0:
                        df = df[df["flux_mjy"] >= min_flux_mjy].copy()

                    # Re-apply limit if needed
                    if max_sources and len(df) > max_sources:
                        df = df.head(max_sources)

                return df

            finally:
                conn.close()

        except Exception as e:
            # SQLite query failed
            if use_csv_fallback:
                print(
                    "Note: CSV catalog is available as an alternative. "
                    "Set use_csv_fallback=True to enable CSV fallback (slower, ~1s vs ~0.01s)."
                )
                logger.warning(
                    f"SQLite query failed ({e}), falling back to CSV. "
                    f"This will be slower (~1s vs ~0.01s)."
                )
                # Fallback to CSV (slower but always works)
                # VLASS catalog reading - need to implement read_vlass_catalog or use generic reader
                from dsa110_contimg.core.catalog.build_master import _read_table

                # Try to find cached VLASS catalog
                cache_dir = ".cache/catalogs"
                os.makedirs(cache_dir, exist_ok=True)
                cache_path = Path(cache_dir) / "vlass_catalog"

                # Try common extensions
                vlass_path = None
                for ext in [".csv", ".fits", ".fits.gz", ".csv.gz"]:
                    candidate = cache_path.with_suffix(ext)
                    if candidate.exists():
                        vlass_path = str(candidate)
                        break

                if vlass_path is None:
                    # Return empty DataFrame if no catalog found
                    logger.warning(
                        "VLASS catalog not found. Please provide catalog_path or place "
                        f"VLASS catalog in {cache_dir}/vlass_catalog.csv or .fits"
                    )
                    return pd.DataFrame(columns=["ra_deg", "dec_deg", "flux_mjy"])

                df_full = _read_table(vlass_path)

                # Normalize column names for VLASS
                from dsa110_contimg.core.catalog.build_master import _normalize_columns

                VLASS_CANDIDATES = {
                    "ra": ["ra", "ra_deg", "raj2000"],
                    "dec": ["dec", "dec_deg", "dej2000"],
                    "flux": [
                        "peak_flux",
                        "peak_mjy_per_beam",
                        "flux_peak",
                        "flux",
                        "total_flux",
                    ],
                }
                col_map = _normalize_columns(df_full, VLASS_CANDIDATES)

                ra_col = col_map.get("ra", "ra")
                dec_col = col_map.get("dec", "dec")
                flux_col = col_map.get("flux", None)

                # Convert to SkyCoord for separation calculation
                ra_vals = pd.to_numeric(df_full[ra_col], errors="coerce")
                dec_vals = pd.to_numeric(df_full[dec_col], errors="coerce")
                sc = SkyCoord(
                    ra=ra_vals.values * u.deg,  # pylint: disable=no-member
                    dec=dec_vals.values * u.deg,
                    frame="icrs",
                )  # pylint: disable=no-member
                center = SkyCoord(
                    ra_deg * u.deg,
                    dec_deg * u.deg,
                    frame="icrs",  # pylint: disable=no-member
                )  # pylint: disable=no-member
                sep = sc.separation(center).deg

                # Filter by separation
                keep = sep <= radius_deg

                # Filter by flux if specified
                if min_flux_mjy is not None and flux_col:
                    flux_vals = pd.to_numeric(df_full[flux_col], errors="coerce")
                    # VLASS flux is typically in mJy, but check if conversion needed
                    if len(flux_vals) > 0 and flux_vals.max() > 1000:
                        flux_vals = flux_vals * 1000.0  # Convert Jy to mJy
                    keep = keep & (flux_vals >= min_flux_mjy)

                result = df_full[keep].copy()

                # Standardize column names
                result["ra_deg"] = pd.to_numeric(result[ra_col], errors="coerce")
                result["dec_deg"] = pd.to_numeric(result[dec_col], errors="coerce")

                if flux_col and flux_col in result.columns:
                    flux_vals = pd.to_numeric(result[flux_col], errors="coerce")
                    if len(flux_vals) > 0 and flux_vals.max() > 1000:
                        result["flux_mjy"] = flux_vals * 1000.0
                    else:
                        result["flux_mjy"] = flux_vals
                else:
                    result["flux_mjy"] = None

                # Sort by flux and limit
                if "flux_mjy" in result.columns and result["flux_mjy"].notna().any():
                    result = result.sort_values("flux_mjy", ascending=False, na_position="last")
                if max_sources:
                    result = result.head(max_sources)

                # Select only the columns we need
                if len(result) > 0:
                    result = result[["ra_deg", "dec_deg", "flux_mjy"]].copy()
                else:
                    result = pd.DataFrame(columns=["ra_deg", "dec_deg", "flux_mjy"])

                return result
            else:
                # No fallback - return empty DataFrame
                logger.error(
                    f"SQLite query failed ({e}). "
                    f"SQLite database required. CSV fallback is available but disabled. "
                    f"Set use_csv_fallback=True to enable CSV fallback."
                )
                print(
                    "Note: CSV catalog is available as an alternative. "
                    "Set use_csv_fallback=True to enable CSV fallback (slower, ~1s vs ~0.01s)."
                )
                return pd.DataFrame(columns=["ra_deg", "dec_deg", "flux_mjy"])

    # No SQLite database found and CSV fallback disabled - FATAL ERROR
    if db_path is None:
        error_msg = (
            f"VLASS SQLite database not found for dec={dec_deg:.1f}°. "
            "SQLite database is required for catalog queries. "
            "Either: (1) Build the database using catalog build tools, or "
            "(2) Set use_csv_fallback=True to enable slower CSV fallback."
        )
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)


def query_catalog_sources(
    catalog_type: str,
    ra_deg: float,
    dec_deg: float,
    radius_deg: float,
    min_flux_mjy: float | None = None,
    max_sources: int | None = None,
    catalog_path: str | os.PathLike[str] | None = None,
    auto_regenerate: bool = False,
) -> pd.DataFrame:
    """Unified interface to query catalog sources (NVSS, RAX, VLASS).

        This function provides a common API for querying different radio source catalogs.
        It automatically selects the appropriate query function based on catalog_type.

    Parameters
    ----------
    catalog_type : str
        One of "nvss", "rax", "vlass".
    ra_deg : float
        Field center RA in degrees.
    dec_deg : float
        Field center Dec in degrees.
    radius_deg : float
        Search radius in degrees.
    min_flux_mjy : Optional[float]
        Minimum flux in mJy (optional). Default is None.
    max_sources : Optional[int]
        Maximum number of sources to return (optional). Default is None.
    catalog_path : Optional[str or os.PathLike]
        Explicit path to SQLite database (overrides auto-resolution). Default is None.
    auto_regenerate : bool
        If True, automatically regenerate corrupted strip databases. Default is False.

    Returns
    -------
        DataFrame
        DataFrame with columns ra_deg, dec_deg, flux_mjy.

    Examples
    --------
        >>> # Query NVSS sources
        >>> df = query_catalog_sources("nvss", ra_deg=83.5, dec_deg=54.6, radius_deg=1.0)

        >>> # Query RAX sources
        >>> df = query_catalog_sources("rax", ra_deg=83.5, dec_deg=54.6, radius_deg=1.0)

        >>> # Query VLASS sources
        >>> df = query_catalog_sources("vlass", ra_deg=83.5, dec_deg=54.6, radius_deg=1.0)
    """
    catalog_type_lower = catalog_type.lower()

    if catalog_type_lower == "nvss":
        return query_nvss_sources(
            ra_deg=ra_deg,
            dec_deg=dec_deg,
            radius_deg=radius_deg,
            min_flux_mjy=min_flux_mjy,
            max_sources=max_sources,
            catalog_path=catalog_path,
            auto_regenerate=auto_regenerate,
        )
    elif catalog_type_lower == "rax":
        return query_rax_sources(
            ra_deg=ra_deg,
            dec_deg=dec_deg,
            radius_deg=radius_deg,
            min_flux_mjy=min_flux_mjy,
            max_sources=max_sources,
            catalog_path=catalog_path,
        )
    elif catalog_type_lower == "vlass":
        return query_vlass_sources(
            ra_deg=ra_deg,
            dec_deg=dec_deg,
            radius_deg=radius_deg,
            min_flux_mjy=min_flux_mjy,
            max_sources=max_sources,
            catalog_path=catalog_path,
        )
    else:
        raise ValueError(
            f"Unsupported catalog_type: {catalog_type}. Supported types: nvss, rax, vlass"
        )


def get_calibrator_position(name: str) -> tuple[float, float] | None:
    """Lookup a calibrator position by name.

    Convenience wrapper around load_vla_catalog and get_calibrator_radec.

    Parameters
    ----------
    name : str
        Calibrator name

    Returns
    -------
    Tuple[float, float]
        (ra_deg, dec_deg) or None if not found
    """
    try:
        df = load_vla_catalog()
        return get_calibrator_radec(df, name)
    except (KeyError, FileNotFoundError):
        return None


def query_calibrator_by_position(
    ra_deg: float,
    dec_deg: float,
    radius_deg: float = 1.0,
    catalog_path: str | None = None,
) -> tuple[str, float, float, float] | None:
    """Find nearest calibrator to a position.

    Convenience wrapper around load_vla_catalog and nearest_calibrator_within_radius.

    Parameters
    ----------
    ra_deg : float
        RA in degrees
    dec_deg : float
        Dec in degrees
    radius_deg : float
        Search radius in degrees
    catalog_path : Optional[str]
        Explicit path to catalog

    Returns
    -------
    Tuple[str, float, float, float]
        (name, ra_deg, dec_deg, flux_jy) or None if not found
    """
    try:
        df = load_vla_catalog(explicit_path=catalog_path)
        return nearest_calibrator_within_radius(ra_deg, dec_deg, df, radius_deg)
    except FileNotFoundError:
        return None
