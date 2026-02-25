"""Flux calibration monitoring and alerting system.

This module provides functions to track flux scale stability over time,
detect calibration drift, and generate alerts when issues are detected.

Implements Proposal #6: Flux Calibration Monitoring & Alerts
"""

import logging
import sqlite3
import time

import numpy as np
from astropy.time import Time

from dsa110_contimg.common.unified_config import settings

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = str(settings.paths.pipeline_db)


def create_flux_monitoring_tables(db_path: str = DEFAULT_DB_PATH):
    """Create calibration_monitoring table in products database.

        Table structure:
        - id: Primary key
        - calibrator_name: Name of calibrator source
        - ms_path: Path to measurement set
        - observed_flux_jy: Measured flux density [Jy]
        - catalog_flux_jy: Expected flux from catalog [Jy]
        - flux_ratio: observed/catalog
        - frequency_ghz: Observation frequency [GHz]
        - mjd: Modified Julian Date
        - timestamp_iso: ISO timestamp
        - phase_rms_deg: Phase RMS from calibration [degrees]
        - amp_rms: Amplitude RMS from calibration
        - flagged_fraction: Fraction of data flagged
        - created_at: Unix timestamp when recorded

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
        # Create main monitoring table
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS calibration_monitoring (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                calibrator_name TEXT NOT NULL,
                ms_path TEXT NOT NULL,
                observed_flux_jy REAL NOT NULL,
                catalog_flux_jy REAL NOT NULL,
                flux_ratio REAL NOT NULL,
                frequency_ghz REAL NOT NULL,
                mjd REAL NOT NULL,
                timestamp_iso TEXT,
                phase_rms_deg REAL,
                amp_rms REAL,
                flagged_fraction REAL,
                created_at REAL NOT NULL,
                notes TEXT
            )
        """
        )

        # Create indices for efficient queries
        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_cal_mon_calibrator
            ON calibration_monitoring(calibrator_name, mjd DESC)
        """
        )
        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_cal_mon_mjd
            ON calibration_monitoring(mjd DESC)
        """
        )
        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_cal_mon_ms_path
            ON calibration_monitoring(ms_path)
        """
        )

        # Create table for flux monitoring alerts
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS flux_monitoring_alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                alert_type TEXT NOT NULL,
                severity TEXT NOT NULL,
                calibrator_name TEXT,
                time_window_days REAL NOT NULL,
                flux_drift_percent REAL,
                n_measurements INTEGER NOT NULL,
                message TEXT NOT NULL,
                triggered_at REAL NOT NULL,
                acknowledged_at REAL,
                acknowledged_by TEXT,
                resolved_at REAL,
                resolution_note TEXT
            )
        """
        )

        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_flux_alerts_triggered
            ON flux_monitoring_alerts(triggered_at DESC)
        """
        )
        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_flux_alerts_severity
            ON flux_monitoring_alerts(severity, triggered_at DESC)
        """
        )

        conn.commit()
        logger.info("Created calibration_monitoring and flux_monitoring_alerts tables")
        return True

    except Exception as e:
        logger.error(f"Error creating flux monitoring tables: {e}")
        return False
    finally:
        conn.close()


def record_calibration_measurement(
    calibrator_name: str,
    ms_path: str,
    observed_flux_jy: float,
    catalog_flux_jy: float,
    frequency_ghz: float,
    mjd: float,
    timestamp_iso: str | None = None,
    phase_rms_deg: float | None = None,
    amp_rms: float | None = None,
    flagged_fraction: float | None = None,
    notes: str | None = None,
    db_path: str = DEFAULT_DB_PATH,
) -> int | None:
    """Record a calibration flux measurement.

    Parameters
    ----------
    calibrator_name : str
        Name of calibrator (e.g., "3C286", "J1331+3030")
    ms_path : str
        Path to measurement set
    observed_flux_jy : float
        Measured flux density [Jy]
    catalog_flux_jy : float
        Expected catalog flux [Jy]
    frequency_ghz : float
        Observation frequency [GHz]
    mjd : float
        Modified Julian Date
    timestamp_iso : str
        ISO timestamp string
    phase_rms_deg : float
        Phase RMS from calibration solution [degrees]
    amp_rms : float
        Amplitude RMS from calibration solution
    flagged_fraction : float
        Fraction of data flagged (0.0-1.0)
    notes : str, optional
        Optional notes
    db_path : str
        Path to database

    Returns
    -------
        int or None
        Measurement ID if successful, None otherwise
    """
    flux_ratio = observed_flux_jy / catalog_flux_jy if catalog_flux_jy > 0 else 0.0

    conn = sqlite3.connect(db_path, timeout=30.0)
    cur = conn.cursor()

    try:
        cur.execute(
            """
            INSERT INTO calibration_monitoring
            (calibrator_name, ms_path, observed_flux_jy, catalog_flux_jy,
             flux_ratio, frequency_ghz, mjd, timestamp_iso,
             phase_rms_deg, amp_rms, flagged_fraction, created_at, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                calibrator_name,
                ms_path,
                observed_flux_jy,
                catalog_flux_jy,
                flux_ratio,
                frequency_ghz,
                mjd,
                timestamp_iso,
                phase_rms_deg,
                amp_rms,
                flagged_fraction,
                time.time(),
                notes,
            ),
        )

        measurement_id = cur.lastrowid
        conn.commit()

        logger.info(
            f"Recorded flux measurement {measurement_id}: {calibrator_name} "
            f"ratio={flux_ratio:.3f} (obs={observed_flux_jy:.3f} Jy, "
            f"cat={catalog_flux_jy:.3f} Jy)"
        )

        return measurement_id

    except Exception as e:
        logger.error(f"Error recording calibration measurement: {e}")
        return None
    finally:
        conn.close()


def calculate_flux_trends(
    calibrator_name: str | None = None,
    time_window_days: float = 7.0,
    db_path: str = DEFAULT_DB_PATH,
) -> dict[str, dict]:
    """Calculate flux scale trends over specified time window.

    Parameters
    ----------
    calibrator_name : str or None
        Specific calibrator (None = all calibrators)
    time_window_days : float
        Time window for analysis [days]
    db_path : str
        Path to database

    Returns
    -------
        dict
        Dictionary mapping calibrator_name to trend statistics:
        {
        'calibrator_name': {
        'n_measurements': int,
        'mean_ratio': float,
        'std_ratio': float,
        'min_ratio': float,
        'max_ratio': float,
        'drift_percent': float,  # (max - min) / mean * 100
        'recent_ratio': float,   # Most recent measurement
        'first_mjd': float,
        'last_mjd': float
        }
        }
    """
    import pandas as pd

    conn = sqlite3.connect(db_path)

    # Calculate MJD cutoff
    current_mjd = Time(time.time(), format="unix").mjd
    min_mjd = current_mjd - time_window_days

    # Build query
    if calibrator_name:
        query = """
            SELECT calibrator_name, flux_ratio, mjd
            FROM calibration_monitoring
            WHERE calibrator_name = ? AND mjd >= ?
            ORDER BY calibrator_name, mjd
        """
        params = (calibrator_name, min_mjd)
    else:
        query = """
            SELECT calibrator_name, flux_ratio, mjd
            FROM calibration_monitoring
            WHERE mjd >= ?
            ORDER BY calibrator_name, mjd
        """
        params = (min_mjd,)

    # Load data with Pandas
    df = pd.read_sql_query(query, conn, params=params)
    conn.close()

    if len(df) == 0:
        return {}

    # Group by calibrator and compute statistics vectorized
    trends = {}
    for cal_name, group in df.groupby("calibrator_name"):
        ratios = group["flux_ratio"].values
        mjds = group["mjd"].values

        ratios_arr = np.array(ratios)
        mjds_arr = np.array(mjds)

        mean_ratio = np.mean(ratios_arr)
        std_ratio = np.std(ratios_arr)
        min_ratio = np.min(ratios_arr)
        max_ratio = np.max(ratios_arr)

        # Calculate drift relative to the most stable baseline (min value).
        # This is more sensitive to long-term drift than mean-normalized spread.
        if min_ratio > 0:
            drift_percent = (max_ratio / min_ratio - 1.0) * 100.0
        elif mean_ratio > 0:
            drift_percent = (max_ratio - min_ratio) / mean_ratio * 100.0
        else:
            drift_percent = 0.0

        trends[cal_name] = {
            "n_measurements": len(ratios),
            "mean_ratio": float(mean_ratio),
            "std_ratio": float(std_ratio),
            "min_ratio": float(min_ratio),
            "max_ratio": float(max_ratio),
            "drift_percent": float(drift_percent),
            "recent_ratio": float(ratios[-1]),
            "first_mjd": float(mjds_arr[0]),
            "last_mjd": float(mjds_arr[-1]),
        }

    return trends


def check_flux_stability(
    drift_threshold_percent: float = 20.0,
    time_window_days: float = 7.0,
    min_measurements: int = 3,
    db_path: str = DEFAULT_DB_PATH,
) -> tuple[bool, list[dict]]:
    """Check flux calibration stability and detect issues.

    Parameters
    ----------
    drift_threshold_percent : float
        Alert if drift exceeds this [%]
    time_window_days : float
        Time window for analysis [days]
    min_measurements : int
        Minimum measurements required for alert
    db_path : str
        Path to database

    Returns
    -------
        tuple
        Tuple of (all_stable: bool, issues: List[Dict])

        issues contains:
        {
        'calibrator_name': str,
        'drift_percent': float,
        'n_measurements': int,
        'mean_ratio': float,
        'severity': 'warning' | 'critical'
        }
    """
    trends = calculate_flux_trends(
        calibrator_name=None, time_window_days=time_window_days, db_path=db_path
    )

    issues = []

    for cal_name, stats in trends.items():
        if stats["n_measurements"] < min_measurements:
            continue

        drift = stats["drift_percent"]

        if drift > drift_threshold_percent:
            # Determine severity
            if drift > 2 * drift_threshold_percent:
                severity = "critical"
            else:
                severity = "warning"

            issues.append(
                {
                    "calibrator_name": cal_name,
                    "drift_percent": drift,
                    "n_measurements": stats["n_measurements"],
                    "mean_ratio": stats["mean_ratio"],
                    "recent_ratio": stats["recent_ratio"],
                    "time_window_days": time_window_days,
                    "severity": severity,
                }
            )

    all_stable = len(issues) == 0

    return all_stable, issues


def create_flux_stability_alert(issue: dict, db_path: str = DEFAULT_DB_PATH) -> int | None:
    """Create a flux stability alert in the database.

    Parameters
    ----------
    issue : dict
        Issue dictionary from check_flux_stability()
    db_path : str
        Path to database

    Returns
    -------
        int or None
        Alert ID if successful, None otherwise
    """
    message = (
        f"Flux calibration drift detected for {issue['calibrator_name']}: "
        f"{issue['drift_percent']:.1f}% over {issue['time_window_days']:.1f} days "
        f"(mean ratio: {issue['mean_ratio']:.3f}, recent: {issue['recent_ratio']:.3f}, "
        f"n={issue['n_measurements']})"
    )

    conn = sqlite3.connect(db_path, timeout=30.0)
    cur = conn.cursor()

    try:
        cur.execute(
            """
            INSERT INTO flux_monitoring_alerts
            (alert_type, severity, calibrator_name, time_window_days,
             flux_drift_percent, n_measurements, message, triggered_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                "flux_drift",
                issue["severity"],
                issue["calibrator_name"],
                issue["time_window_days"],
                issue["drift_percent"],
                issue["n_measurements"],
                message,
                time.time(),
            ),
        )

        alert_id = cur.lastrowid
        conn.commit()

        logger.warning(f"Created flux stability alert {alert_id}: {message}")

        return alert_id

    except Exception as e:
        logger.error(f"Error creating flux stability alert: {e}")
        return None
    finally:
        conn.close()


def get_recent_flux_alerts(
    days: float = 7.0,
    severity: str | None = None,
    unresolved_only: bool = True,
    db_path: str = DEFAULT_DB_PATH,
) -> list[dict]:
    """Get recent flux stability alerts.

    Parameters
    ----------
    days : int
        Look back this many days
    severity : str or None
        Filter by severity ('warning', 'critical', None=all)
    unresolved_only : bool
        Only return unresolved alerts
    db_path : str
        Path to database

    Returns
    -------
        list
        List of alert dictionaries
    """
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cutoff_time = time.time() - (days * 86400.0)

    query = """
        SELECT id, alert_type, severity, calibrator_name,
               time_window_days, flux_drift_percent, n_measurements,
               message, triggered_at, acknowledged_at, resolved_at
        FROM flux_monitoring_alerts
        WHERE triggered_at >= ?
    """
    params = [cutoff_time]

    if severity:
        query += " AND severity = ?"
        params.append(severity)

    if unresolved_only:
        query += " AND resolved_at IS NULL"

    query += " ORDER BY triggered_at DESC"

    cur.execute(query, params)
    rows = cur.fetchall()
    conn.close()

    alerts = []
    for row in rows:
        alerts.append(
            {
                "id": row[0],
                "alert_type": row[1],
                "severity": row[2],
                "calibrator_name": row[3],
                "time_window_days": row[4],
                "flux_drift_percent": row[5],
                "n_measurements": row[6],
                "message": row[7],
                "triggered_at": row[8],
                "acknowledged_at": row[9],
                "resolved_at": row[10],
            }
        )

    return alerts


def run_flux_monitoring_check(
    drift_threshold_percent: float = 20.0,
    time_window_days: float = 7.0,
    min_measurements: int = 3,
    create_alerts: bool = True,
    db_path: str = DEFAULT_DB_PATH,
) -> tuple[bool, list[dict]]:
    """Run flux monitoring check and optionally create alerts.

        This is the main entry point for automated flux monitoring.
        Call this periodically (e.g., daily) to check calibration stability.

    Parameters
    ----------
    drift_threshold_percent : float
        Alert threshold [%]
    time_window_days : float
        Analysis window [days]
    min_measurements : int
        Minimum measurements for alert
    create_alerts : bool
        Create database alerts for issues
    db_path : str
        Path to database

    Returns
    -------
        tuple
        Tuple of (all_stable: bool, issues: List[Dict])
    """
    all_stable, issues = check_flux_stability(
        drift_threshold_percent=drift_threshold_percent,
        time_window_days=time_window_days,
        min_measurements=min_measurements,
        db_path=db_path,
    )

    if not all_stable:
        logger.warning(f"Flux stability check found {len(issues)} issue(s)")

        if create_alerts:
            for issue in issues:
                create_flux_stability_alert(issue, db_path=db_path)
    else:
        logger.info("Flux stability check: all calibrators stable")

    return all_stable, issues
