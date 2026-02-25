"""Data retrieval for lightcurve visualization.

Provides functions to fetch photometry data from the pipeline database.
"""

from typing import List, Optional
import sqlite3
import logging

logger = logging.getLogger(__name__)


def get_photometry_data(
    pipeline_db: str,
    source_id: str,
    start_mjd: Optional[float] = None,
    end_mjd: Optional[float] = None,
) -> List[dict]:
    """Get photometry data for a source from the database.

    Parameters
    ----------
    pipeline_db : str
        Path to the pipeline SQLite database
    source_id : str
        Source identifier (e.g. "0834+555")
    start_mjd : float, optional
        Start MJD for filtering
    end_mjd : float, optional
        End MJD for filtering

    Returns
    -------
    List[dict]
        List of photometry measurements with keys:
        - mjd: Modified Julian Date
        - flux_jy: Flux density in Jy
        - flux_err_jy: Flux density error in Jy
    """
    try:
        conn = sqlite3.connect(pipeline_db, timeout=30)
        conn.row_factory = sqlite3.Row

        query = """
            SELECT mjd, flux_jy, flux_err_jy, peak_jyb, peak_err_jyb, measured_at
            FROM photometry
            WHERE source_id = ?
        """
        params = [source_id]

        if start_mjd is not None:
            query += " AND mjd >= ?"
            params.append(start_mjd)
        if end_mjd is not None:
            query += " AND mjd <= ?"
            params.append(end_mjd)

        query += " ORDER BY mjd"

        cursor = conn.execute(query, params)
        rows = cursor.fetchall()
        conn.close()

        data_points = []
        for row in rows:
            # Use integrated flux if available, otherwise peak flux
            flux_jy = row["flux_jy"] if row["flux_jy"] is not None else row["peak_jyb"]
            flux_err_jy = (
                row["flux_err_jy"]
                if row["flux_err_jy"] is not None
                else row["peak_err_jyb"]
            )

            if flux_jy is None:
                continue

            data_points.append(
                {
                    "mjd": row["mjd"],
                    "flux_jy": flux_jy,
                    "flux_err_jy": flux_err_jy or 0.0,
                }
            )

        logger.info(f"Retrieved {len(data_points)} photometry points for {source_id}")
        return data_points

    except sqlite3.Error as e:
        logger.error(f"Database error retrieving photometry: {e}")
        return []
    except Exception as e:
        logger.error(f"Error retrieving photometry: {e}")
        return []
