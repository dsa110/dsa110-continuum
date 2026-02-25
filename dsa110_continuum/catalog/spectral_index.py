"""Spectral index calculation and management.

This module provides functions to calculate spectral indices from
multi-frequency catalog cross-matches (e.g., NVSS/FIRST at 1.4 GHz,
RACS at 888 MHz, VLASS at 3 GHz).

Spectral index α is defined as: S_ν ∝ ν^α
where S_ν is flux density at frequency ν.

For two frequencies: α = log(S2/S1) / log(ν2/ν1)

Implements Proposal #1: Spectral Index Mapping
"""

import logging
import sqlite3
import time

import numpy as np

from dsa110_contimg.common.unified_config import settings

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = str(settings.paths.pipeline_db)


def create_spectral_indices_table(db_path: str = DEFAULT_DB_PATH):
    """Create spectral_indices table in products database.

        Table structure:
        - id: Primary key
        - source_id: Source identifier (RA_DEC format)
        - ra_deg: Right ascension [degrees]
        - dec_deg: Declination [degrees]
        - spectral_index: Spectral index α (S_ν ∝ ν^α)
        - spectral_index_err: Uncertainty in α
        - freq1_ghz: Lower frequency [GHz]
        - freq2_ghz: Higher frequency [GHz]
        - flux1_mjy: Flux at freq1 [mJy]
        - flux2_mjy: Flux at freq2 [mJy]
        - flux1_err_mjy: Uncertainty in flux1 [mJy]
        - flux2_err_mjy: Uncertainty in flux2 [mJy]
        - catalog1: Source catalog for freq1 (e.g., 'NVSS', 'RACS')
        - catalog2: Source catalog for freq2 (e.g., 'VLASS', 'DSA110')
        - match_separation_arcsec: Angular separation of match [arcsec]
        - n_frequencies: Number of frequency points used
        - fit_quality: Quality of spectral fit ('good', 'fair', 'poor')
        - calculated_at: Unix timestamp
        - notes: Optional notes

    Parameters
    ----------
    db_path : str
        Path to pipeline database (unified pipeline.sqlite3)

    Returns
    -------
        bool
        True if successful
    """
    conn = sqlite3.connect(db_path, timeout=30.0)
    cur = conn.cursor()

    try:
        # Create main spectral indices table
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS spectral_indices (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_id TEXT NOT NULL,
                ra_deg REAL NOT NULL,
                dec_deg REAL NOT NULL,
                spectral_index REAL NOT NULL,
                spectral_index_err REAL,
                freq1_ghz REAL NOT NULL,
                freq2_ghz REAL NOT NULL,
                flux1_mjy REAL NOT NULL,
                flux2_mjy REAL NOT NULL,
                flux1_err_mjy REAL,
                flux2_err_mjy REAL,
                catalog1 TEXT NOT NULL,
                catalog2 TEXT NOT NULL,
                match_separation_arcsec REAL,
                n_frequencies INTEGER DEFAULT 2,
                fit_quality TEXT,
                calculated_at REAL NOT NULL,
                notes TEXT,
                UNIQUE(source_id, catalog1, catalog2)
            )
        """
        )

        # Create indices for efficient queries
        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_spec_idx_source
            ON spectral_indices(source_id)
        """
        )
        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_spec_idx_coords
            ON spectral_indices(ra_deg, dec_deg)
        """
        )
        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_spec_idx_alpha
            ON spectral_indices(spectral_index)
        """
        )
        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_spec_idx_quality
            ON spectral_indices(fit_quality, spectral_index)
        """
        )

        conn.commit()
        logger.info("Created spectral_indices table")
        return True

    except Exception as e:
        logger.error(f"Error creating spectral_indices table: {e}")
        return False
    finally:
        conn.close()


def calculate_spectral_index(
    freq1_ghz: float,
    freq2_ghz: float,
    flux1_mjy: float,
    flux2_mjy: float,
    flux1_err_mjy: float | None = None,
    flux2_err_mjy: float | None = None,
) -> tuple[float, float | None]:
    """Calculate spectral index between two frequencies.

        α = log(S2/S1) / log(ν2/ν1)

        Error propagation (if flux errors provided):
        σ_α² = (1/(ln(ν2/ν1)))² * [(σ_S1/S1)² + (σ_S2/S2)²]

    Parameters
    ----------
    freq1_ghz : float
        First frequency [GHz]
    freq2_ghz : float
        Second frequency [GHz]
    flux1_mjy : float
        Flux at freq1 [mJy]
    flux2_mjy : float
        Flux at freq2 [mJy]
    flux1_err_mjy : float, optional
        Uncertainty in flux1 [mJy]
    flux2_err_mjy : float, optional
        Uncertainty in flux2 [mJy]

    Returns
    -------
        tuple
        Tuple of (spectral_index, spectral_index_err)
        Returns (NaN, None) if calculation fails
    """
    if flux1_mjy <= 0 or flux2_mjy <= 0:
        logger.warning(f"Non-positive flux: {flux1_mjy}, {flux2_mjy}")
        return np.nan, None

    if freq1_ghz <= 0 or freq2_ghz <= 0 or freq1_ghz == freq2_ghz:
        logger.warning(f"Invalid frequencies: {freq1_ghz}, {freq2_ghz}")
        return np.nan, None

    try:
        # Calculate spectral index
        alpha = np.log10(flux2_mjy / flux1_mjy) / np.log10(freq2_ghz / freq1_ghz)

        # Calculate uncertainty if errors provided
        alpha_err = None
        if flux1_err_mjy is not None and flux2_err_mjy is not None:
            if flux1_err_mjy > 0 and flux2_err_mjy > 0:
                # Fractional errors
                frac_err1 = flux1_err_mjy / flux1_mjy
                frac_err2 = flux2_err_mjy / flux2_mjy

                # Error propagation
                log_freq_ratio = np.log10(freq2_ghz / freq1_ghz)
                alpha_err = np.sqrt(frac_err1**2 + frac_err2**2) / (log_freq_ratio * np.log(10))

        return float(alpha), float(alpha_err) if alpha_err is not None else None

    except Exception as e:
        logger.error(f"Error calculating spectral index: {e}")
        return np.nan, None


def fit_spectral_index_multifreq(
    frequencies_ghz: list[float],
    fluxes_mjy: list[float],
    flux_errors_mjy: list[float] | None = None,
) -> tuple[float, float, str]:
    """Fit spectral index to multiple frequency points.

        Uses weighted least-squares fit: log(S) = log(S0) + α*log(ν)

    Parameters
    ----------
    frequencies_ghz : list of float
        List of frequencies [GHz]
    fluxes_mjy : list of float
        List of flux densities [mJy]
    flux_errors_mjy : list of float, optional
        Optional list of flux uncertainties [mJy]

    Returns
    -------
        tuple
        Tuple of (spectral_index, spectral_index_err, fit_quality)
    fit_quality: 'good' | 'fair' | 'poor'
    """
    if len(frequencies_ghz) < 2:
        return np.nan, np.nan, "poor"

    if len(frequencies_ghz) != len(fluxes_mjy):
        logger.error("Frequency and flux arrays must have same length")
        return np.nan, np.nan, "poor"

    # Filter out non-positive values
    valid = [(f, s) for f, s in zip(frequencies_ghz, fluxes_mjy) if f > 0 and s > 0]
    if len(valid) < 2:
        return np.nan, np.nan, "poor"

    freqs = np.array([v[0] for v in valid])
    fluxes = np.array([v[1] for v in valid])

    # Log space
    log_freqs = np.log10(freqs)
    log_fluxes = np.log10(fluxes)

    # Weighted fit if errors provided
    weights = None
    if flux_errors_mjy is not None:
        errors = np.array(
            [
                flux_errors_mjy[i]
                for i in range(len(frequencies_ghz))
                if frequencies_ghz[i] > 0 and fluxes_mjy[i] > 0
            ]
        )
        if len(errors) == len(fluxes) and np.all(errors > 0):
            # Convert to log-space weights
            weights = fluxes / (errors * np.log(10))

    try:
        # Fit: log(S) = log(S0) + α*log(ν)
        if weights is not None:
            coeffs, cov = np.polyfit(log_freqs, log_fluxes, deg=1, w=weights, cov=True)
        else:
            coeffs, cov = np.polyfit(log_freqs, log_fluxes, deg=1, cov=True)

        alpha = coeffs[0]
        alpha_err = np.sqrt(cov[0, 0])

        # Assess fit quality based on scatter
        predicted_log_flux = np.polyval(coeffs, log_freqs)
        residuals = log_fluxes - predicted_log_flux
        rms = np.sqrt(np.mean(residuals**2))

        # Quality thresholds (in log space)
        if rms < 0.05:  # <12% scatter
            quality = "good"
        elif rms < 0.15:  # <40% scatter
            quality = "fair"
        else:
            quality = "poor"

        return float(alpha), float(alpha_err), quality

    except Exception as e:
        logger.error(f"Error fitting spectral index: {e}")
        return np.nan, np.nan, "poor"


def store_spectral_index(
    source_id: str,
    ra_deg: float,
    dec_deg: float,
    spectral_index: float,
    freq1_ghz: float,
    freq2_ghz: float,
    flux1_mjy: float,
    flux2_mjy: float,
    catalog1: str,
    catalog2: str,
    spectral_index_err: float | None = None,
    flux1_err_mjy: float | None = None,
    flux2_err_mjy: float | None = None,
    match_separation_arcsec: float | None = None,
    n_frequencies: int = 2,
    fit_quality: str | None = None,
    notes: str | None = None,
    db_path: str = DEFAULT_DB_PATH,
) -> int | None:
    """Store spectral index in database.

    Parameters
    ----------
    source_id : str
        Source identifier
    ra_deg : float
        Right ascension [degrees]
    dec_deg : float
        Declination [degrees]
    spectral_index : float
        Spectral index α
    freq1_ghz : float
        Lower frequency [GHz]
    freq2_ghz : float
        Higher frequency [GHz]
    flux1_mjy : float
        Flux at freq1 [mJy]
    flux2_mjy : float
        Flux at freq2 [mJy]
    catalog1 : str
        Source catalog for freq1
    catalog2 : str
        Source catalog for freq2
    spectral_index_err : float, optional
        Uncertainty in α
    flux1_err_mjy : float, optional
        Uncertainty in flux1 [mJy]
    flux2_err_mjy : float, optional
        Uncertainty in flux2 [mJy]
    match_separation_arcsec : float, optional
        Angular separation [arcsec]
    n_frequencies : int, optional
        Number of frequency points
    fit_quality : str, optional
        Quality assessment
    notes : str, optional
        Optional notes
    db_path : str
        Path to database

    Returns
    -------
        int or None
        Record ID if successful, None otherwise
    """
    # Validate
    if np.isnan(spectral_index):
        logger.warning(f"Cannot store NaN spectral index for {source_id}")
        return None

    conn = sqlite3.connect(db_path, timeout=30.0)
    cur = conn.cursor()

    try:
        cur.execute(
            """
            INSERT OR REPLACE INTO spectral_indices
            (source_id, ra_deg, dec_deg, spectral_index, spectral_index_err,
             freq1_ghz, freq2_ghz, flux1_mjy, flux2_mjy,
             flux1_err_mjy, flux2_err_mjy,
             catalog1, catalog2, match_separation_arcsec,
             n_frequencies, fit_quality, calculated_at, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                source_id,
                ra_deg,
                dec_deg,
                spectral_index,
                spectral_index_err,
                freq1_ghz,
                freq2_ghz,
                flux1_mjy,
                flux2_mjy,
                flux1_err_mjy,
                flux2_err_mjy,
                catalog1,
                catalog2,
                match_separation_arcsec,
                n_frequencies,
                fit_quality,
                time.time(),
                notes,
            ),
        )

        record_id = cur.lastrowid
        conn.commit()

        logger.debug(
            f"Stored spectral index {record_id}: {source_id} "
            f"α={spectral_index:.2f} ({catalog1}-{catalog2})"
        )

        return record_id

    except Exception as e:
        logger.error(f"Error storing spectral index: {e}")
        return None
    finally:
        conn.close()


def query_spectral_indices(
    ra_deg: float | None = None,
    dec_deg: float | None = None,
    radius_deg: float = 1.0,
    alpha_min: float | None = None,
    alpha_max: float | None = None,
    fit_quality: str | None = None,
    limit: int = 1000,
    db_path: str = DEFAULT_DB_PATH,
) -> list[dict]:
    """Query spectral indices from database.

    Parameters
    ----------
    ra_deg : float
        Center RA for cone search [degrees]
    dec_deg : float
        Center Dec for cone search [degrees]
    radius_deg : float
        Search radius [degrees]
    alpha_min : float
        Minimum spectral index
    alpha_max : float
        Maximum spectral index
    fit_quality : str
        Filter by quality ('good', 'fair', 'poor')
    limit : int
        Maximum number of results
    db_path : str
        Path to database

    Returns
    -------
        list
        List of spectral index dictionaries
    """
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    query = "SELECT * FROM spectral_indices WHERE 1=1"
    params = []

    # Cone search
    if ra_deg is not None and dec_deg is not None:
        query += """
            AND (
                (ra_deg - ?)*(ra_deg - ?) + (dec_deg - ?)*(dec_deg - ?)
                <= ?*?
            )
        """
        params.extend([ra_deg, ra_deg, dec_deg, dec_deg, radius_deg, radius_deg])

    # Spectral index range
    if alpha_min is not None:
        query += " AND spectral_index >= ?"
        params.append(alpha_min)
    if alpha_max is not None:
        query += " AND spectral_index <= ?"
        params.append(alpha_max)

    # Fit quality
    if fit_quality:
        query += " AND fit_quality = ?"
        params.append(fit_quality)

    query += " ORDER BY calculated_at DESC LIMIT ?"
    params.append(limit)

    cur.execute(query, params)
    rows = cur.fetchall()
    conn.close()

    # Get column names
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("PRAGMA table_info(spectral_indices)")
    columns = [col[1] for col in cur.fetchall()]
    conn.close()

    results = []
    for row in rows:
        results.append(dict(zip(columns, row)))

    return results


def get_spectral_index_for_source(source_id: str, db_path: str = DEFAULT_DB_PATH) -> dict | None:
    """Get spectral index for a specific source.

        If multiple entries exist, returns the one with best fit_quality.

    Parameters
    ----------
    source_id : str
        Source identifier
    db_path : str
        Path to database

    Returns
    -------
        dict or None
        Spectral index dictionary or None
    """
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.execute(
        """
        SELECT * FROM spectral_indices
        WHERE source_id = ?
        ORDER BY
            CASE fit_quality
                WHEN 'good' THEN 1
                WHEN 'fair' THEN 2
                WHEN 'poor' THEN 3
                ELSE 4
            END,
            calculated_at DESC
        LIMIT 1
    """,
        (source_id,),
    )

    row = cur.fetchone()

    if row is None:
        conn.close()
        return None

    # Get column names
    cur.execute("PRAGMA table_info(spectral_indices)")
    columns = [col[1] for col in cur.fetchall()]
    conn.close()

    return dict(zip(columns, row))


def calculate_and_store_from_catalogs(
    source_id: str,
    ra_deg: float,
    dec_deg: float,
    catalog_fluxes: dict[str, tuple[float, float, float]],
    db_path: str = DEFAULT_DB_PATH,
) -> list[int]:
    """Calculate spectral indices from multiple catalog matches.

    Parameters
    ----------
    source_id : str or int
        Source identifier.
    ra_deg : float
        Source RA in degrees.
    dec_deg : float
        Source Dec in degrees.
    catalog_fluxes : dict
        Dictionary mapping catalog name to (freq_ghz, flux_mjy, flux_err_mjy).
        For example::

            {
                'NVSS': (1.4, 150.0, 5.0),
                'VLASS': (3.0, 80.0, 4.0),
                'RACS': (0.888, 200.0, 10.0)
            }

    db_path : str, optional
        Path to database.

    Returns
    -------
    list
        List of record IDs created.
    """
    if len(catalog_fluxes) < 2:
        logger.warning(f"Need at least 2 catalogs for spectral index, got {len(catalog_fluxes)}")
        return []

    record_ids = []

    # Calculate pairwise spectral indices
    catalogs = list(catalog_fluxes.keys())
    for i in range(len(catalogs)):
        for j in range(i + 1, len(catalogs)):
            cat1, cat2 = catalogs[i], catalogs[j]

            freq1, flux1, err1 = catalog_fluxes[cat1]
            freq2, flux2, err2 = catalog_fluxes[cat2]

            # Ensure freq1 < freq2
            if freq1 > freq2:
                cat1, cat2 = cat2, cat1
                freq1, flux1, err1, freq2, flux2, err2 = freq2, flux2, err2, freq1, flux1, err1

            # Calculate
            alpha, alpha_err = calculate_spectral_index(freq1, freq2, flux1, flux2, err1, err2)

            if not np.isnan(alpha):
                # Assess quality based on error
                if alpha_err is not None and alpha_err < 0.2:
                    quality = "good"
                elif alpha_err is not None and alpha_err < 0.5:
                    quality = "fair"
                else:
                    quality = "poor"

                record_id = store_spectral_index(
                    source_id=source_id,
                    ra_deg=ra_deg,
                    dec_deg=dec_deg,
                    spectral_index=alpha,
                    freq1_ghz=freq1,
                    freq2_ghz=freq2,
                    flux1_mjy=flux1,
                    flux2_mjy=flux2,
                    catalog1=cat1,
                    catalog2=cat2,
                    spectral_index_err=alpha_err,
                    flux1_err_mjy=err1,
                    flux2_err_mjy=err2,
                    fit_quality=quality,
                    db_path=db_path,
                )

                if record_id:
                    record_ids.append(record_id)

    logger.info(f"Created {len(record_ids)} spectral index entries for {source_id}")
    return record_ids


def get_spectral_index_statistics(
    db_path: str = DEFAULT_DB_PATH,
) -> dict:
    """Get statistics on spectral indices in database.

    Returns
    -------
        Dictionary with statistics:
        - total_count: Total number of spectral indices
        - by_quality: Count by fit quality
        - median_alpha: Median spectral index
        - steep_spectrum_count: Count with α < -0.7 (steep spectrum)
        - flat_spectrum_count: Count with -0.5 < α < 0.5 (flat spectrum)
        - inverted_spectrum_count: Count with α > 0.5 (inverted spectrum)
    """
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    try:
        # Total count
        cur.execute("SELECT COUNT(*) FROM spectral_indices")
        total_count = cur.fetchone()[0]

        # By quality
        cur.execute(
            """
            SELECT fit_quality, COUNT(*)
            FROM spectral_indices
            GROUP BY fit_quality
        """
        )
        by_quality = {row[0]: row[1] for row in cur.fetchall()}

        # Get all spectral indices
        cur.execute("SELECT spectral_index FROM spectral_indices")
        alphas = [row[0] for row in cur.fetchall()]

        if len(alphas) == 0:
            return {
                "total_count": 0,
                "by_quality": {},
                "median_alpha": None,
                "steep_spectrum_count": 0,
                "flat_spectrum_count": 0,
                "inverted_spectrum_count": 0,
            }

        alphas = np.array(alphas)
        median_alpha = float(np.median(alphas))

        steep_count = int(np.sum(alphas < -0.7))
        flat_count = int(np.sum((alphas >= -0.5) & (alphas <= 0.5)))
        inverted_count = int(np.sum(alphas > 0.5))

        return {
            "total_count": total_count,
            "by_quality": by_quality,
            "median_alpha": median_alpha,
            "steep_spectrum_count": steep_count,
            "flat_spectrum_count": flat_count,
            "inverted_spectrum_count": inverted_count,
        }

    except Exception as e:
        logger.error(f"Error getting spectral index statistics: {e}")
        return {}
    finally:
        conn.close()
