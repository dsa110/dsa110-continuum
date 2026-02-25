"""Variable source detection module for DSA-110 continuum imaging pipeline.

This module provides functions to detect variable radio sources
by comparing current observations with baseline catalogs (NVSS, FIRST).

Implements Proposal #2: Transient Detection & Classification
"""

import logging
import sqlite3
import time

import numpy as np
import pandas as pd

from dsa110_contimg.common.unified_config import settings

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = str(settings.paths.pipeline_db)


def create_variable_source_detection_tables(
    db_path: str = DEFAULT_DB_PATH,
) -> bool:
    """Create database tables for variable source detection.

        Tables created:
        - variable_source_candidates: Detected variable sources
        - variable_source_alerts: High-priority alerts for follow-up
        - variable_source_lightcurves: Flux measurements over time

    Parameters
    ----------
    db_path : str
        Path to products database

    Returns
    -------
        bool
        True if successful
    """
    from dsa110_contimg.infrastructure.database.unified import Database
    
    try:
        db = Database(db_path)
        conn = db.conn  # Use raw connection for DDL as query_df is for queries

        # Main transient candidates table
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS variable_source_candidates (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_name TEXT NOT NULL,
                ra_deg REAL NOT NULL,
                dec_deg REAL NOT NULL,
                detection_type TEXT NOT NULL,
                flux_obs_mjy REAL NOT NULL,
                flux_baseline_mjy REAL,
                flux_ratio REAL,
                significance_sigma REAL NOT NULL,
                baseline_catalog TEXT,
                detected_at REAL NOT NULL,
                mosaic_id INTEGER,
                classification TEXT,
                classified_by TEXT,
                classified_at REAL,
                variability_index REAL,
                last_updated REAL NOT NULL,
                follow_up_status TEXT,
                notes TEXT,
                FOREIGN KEY (mosaic_id) REFERENCES products(id)
            )
        """
        )

        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_transients_type
            ON variable_source_candidates(detection_type, significance_sigma DESC)
        """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_transients_coords
            ON variable_source_candidates(ra_deg, dec_deg)
        """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_transients_detected
            ON variable_source_candidates(detected_at DESC)
        """
        )

        # High-priority alerts table
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS variable_source_alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                candidate_id INTEGER NOT NULL,
                alert_level TEXT NOT NULL,
                alert_message TEXT NOT NULL,
                created_at REAL NOT NULL,
                acknowledged BOOLEAN DEFAULT 0,
                acknowledged_at REAL,
                acknowledged_by TEXT,
                follow_up_status TEXT,
                notes TEXT,
                FOREIGN KEY (candidate_id) REFERENCES variable_source_candidates(id)
            )
        """
        )

        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_alerts_level
            ON variable_source_alerts(alert_level, created_at DESC)
        """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_alerts_status
            ON variable_source_alerts(acknowledged, created_at DESC)
        """
        )

        # Lightcurve measurements table
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS variable_source_lightcurves (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                candidate_id INTEGER NOT NULL,
                mjd REAL NOT NULL,
                flux_mjy REAL NOT NULL,
                flux_err_mjy REAL,
                frequency_ghz REAL NOT NULL,
                mosaic_id INTEGER,
                measured_at REAL NOT NULL,
                FOREIGN KEY (candidate_id) REFERENCES variable_source_candidates(id),
                FOREIGN KEY (mosaic_id) REFERENCES products(id)
            )
        """
        )

        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_lightcurves_candidate
            ON variable_source_lightcurves(candidate_id, mjd)
        """
        )

        # No need to commit manually if using sqlite3 connection context or Database usually autocommits on DDL
        # But standard sqlite connection needs commit for DDL sometimes? 
        # Actually in sqlite3, DDL is transactional.
        conn.commit()
        logger.info("Created variable source detection tables")
        return True

    except Exception as e:
        logger.error(f"Error creating variable source detection tables: {e}")
        return False


def detect_variable_sources(
    observed_sources: pd.DataFrame,
    baseline_sources: pd.DataFrame,
    detection_threshold_sigma: float = 5.0,
    variability_threshold: float = 3.0,
    match_radius_arcsec: float = 10.0,
    baseline_catalog: str = "NVSS",
) -> tuple[list[dict], list[dict], list[dict]]:
    """Detect variable sources.

        Compares observed sources with baseline catalog to find:
        1. New sources (not in baseline, significant detection)
        2. Variable sources (flux significantly changed)
        3. Fading sources (baseline source not detected)

    Parameters
    ----------
    observed_sources : pandas.DataFrame
        DataFrame with columns: ra_deg, dec_deg, flux_mjy, flux_err_mjy
    baseline_sources : pandas.DataFrame
        DataFrame with columns: ra_deg, dec_deg, flux_mjy
    detection_threshold_sigma : float
        Significance threshold for new sources [sigma]
    variability_threshold : float
        Threshold for flux variability [sigma]
    match_radius_arcsec : float
        Matching radius [arcsec]
    baseline_catalog : str
        Name of baseline catalog

    Returns
    -------
        tuple of (list of dict, list of dict, list of dict)
        Tuple of (new_sources, variable_sources, fading_sources)
    """
    new_sources = []
    variable_sources = []
    fading_sources = []

    if len(observed_sources) == 0:
        logger.warning("No observed sources provided")
        return new_sources, variable_sources, fading_sources

    if len(baseline_sources) == 0:
        logger.warning("No baseline sources provided")
        # All observed sources are "new" if no baseline
        # Vectorized new sources check
        if "flux_err_mjy" in observed_sources.columns:
             sig = observed_sources["flux_mjy"] / observed_sources["flux_err_mjy"].replace(0, np.nan)
        else:
             sig = pd.Series([np.inf]*len(observed_sources), index=observed_sources.index)

        mask_new = sig >= detection_threshold_sigma
        if mask_new.any():
            new_df = observed_sources.loc[mask_new].copy()
            new_df["detection_type"] = "new"
            new_df["significance_sigma"] = sig.loc[mask_new]
            new_df["flux_obs_mjy"] = new_df["flux_mjy"]
            new_df["flux_baseline_mjy"] = None
            new_sources = new_df[["ra_deg", "dec_deg", "flux_obs_mjy", "flux_baseline_mjy", "significance_sigma", "detection_type"]].to_dict("records")

        return new_sources, variable_sources, fading_sources

    # Import astropy for efficient matching
    from astropy.coordinates import SkyCoord
    from astropy import units as u

    # Create SkyCoord objects
    obs_coords = SkyCoord(
        ra=observed_sources["ra_deg"].values * u.deg,
        dec=observed_sources["dec_deg"].values * u.deg,
    )
    
    base_coords = SkyCoord(
        ra=baseline_sources["ra_deg"].values * u.deg,
        dec=baseline_sources["dec_deg"].values * u.deg,
    )

    match_radius_deg = match_radius_arcsec * u.arcsec

    # 1. Match Observed -> Baseline (Check for New and Variable)
    idx_base, sep2d, _ = obs_coords.match_to_catalog_sky(base_coords)
    matched_mask = sep2d <= match_radius_deg

    # Prepare DataFrame for analysis
    obs_matched = observed_sources.copy()
    obs_matched["matched_base_idx"] = -1
    obs_matched.loc[matched_mask, "matched_base_idx"] = idx_base[matched_mask]
    obs_matched["separation_arcsec"] = sep2d.to(u.arcsec).value
    
    # === Variable Sources (Matched) ===
    matched_obs = obs_matched[matched_mask].copy()
    if not matched_obs.empty:
        # Get baseline fluxes
        baseline_fluxes = baseline_sources.iloc[matched_obs["matched_base_idx"]]["flux_mjy"].values
        matched_obs["flux_baseline_mjy"] = baseline_fluxes
        
        flux_obs = matched_obs["flux_mjy"]
        flux_base = matched_obs["flux_baseline_mjy"]
        
        # Flux ratio
        matched_obs["flux_ratio"] = flux_obs / flux_base
        matched_obs["flux_ratio"] = matched_obs["flux_ratio"].fillna(np.inf) # Handle div by zero if any
        
        # Significance
        flux_err_obs = matched_obs.get("flux_err_mjy", flux_obs * 0.1)
        flux_err_base = flux_base * 0.05
        flux_err_total = np.sqrt(flux_err_obs**2 + flux_err_base**2)
        
        flux_diff = flux_obs - flux_base
        variability_sigma = np.abs(flux_diff) / flux_err_total
        
        matched_obs["significance_sigma"] = variability_sigma.fillna(0.0)
        
        # Filter by threshold
        var_mask = matched_obs["significance_sigma"] >= variability_threshold
        if var_mask.any():
            vars_df = matched_obs[var_mask].copy()
            
            # Classification
            conditions = [
                vars_df["flux_ratio"] > 1.5,
                vars_df["flux_ratio"] < 0.67
            ]
            choices = ["brightening", "fading"]
            vars_df["detection_type"] = np.select(conditions, choices, default="variable")
            
            vars_df["flux_obs_mjy"] = vars_df["flux_mjy"]
            variable_sources = vars_df[[
                "ra_deg", "dec_deg", "flux_obs_mjy", "flux_baseline_mjy", 
                "flux_ratio", "significance_sigma", "detection_type", "separation_arcsec"
            ]].to_dict("records")

    # === New Sources (Unmatched) ===
    unmatched_obs = obs_matched[~matched_mask].copy()
    if not unmatched_obs.empty:
        flux_obs = unmatched_obs["flux_mjy"]
        flux_err_obs = unmatched_obs.get("flux_err_mjy", flux_obs * 0.1)
        
        significance = flux_obs / flux_err_obs.replace(0, np.nan)
        unmatched_obs["significance_sigma"] = significance.fillna(0.0)
        
        new_mask = unmatched_obs["significance_sigma"] >= detection_threshold_sigma
        if new_mask.any():
            new_df = unmatched_obs[new_mask].copy()
            new_df["detection_type"] = "new"
            new_df["flux_obs_mjy"] = new_df["flux_mjy"]
            new_df["flux_baseline_mjy"] = None
            
            new_sources = new_df[[
                "ra_deg", "dec_deg", "flux_obs_mjy", "flux_baseline_mjy", 
                "significance_sigma", "detection_type"
            ]].to_dict("records")

    # 2. Match Baseline -> Observed (Check for Fading/Disappeared)
    # We want to find baseline sources that have NO match in observed sources
    idx_obs, sep2d_rev, _ = base_coords.match_to_catalog_sky(obs_coords)
    unmatched_base_mask = sep2d_rev > match_radius_deg
    
    # Iterate only over unmatched baseline sources
    # Filter baseline sources > 10mJy FIRST
    strong_baseline_mask = (baseline_sources["flux_mjy"] >= 10.0)
    
    # Candidates for fading: Unmatched AND Strong
    fading_candidates_mask = unmatched_base_mask & strong_baseline_mask
    
    if fading_candidates_mask.any():
        fading_df = baseline_sources[fading_candidates_mask].copy()
        fading_df["flux_baseline_mjy"] = fading_df["flux_mjy"]
        fading_df["flux_obs_mjy"] = 0.0
        fading_df["flux_ratio"] = 0.0
        # Significance = Flux / (5% error) = 20 sigma
        fading_df["significance_sigma"] = fading_df["flux_baseline_mjy"] / (fading_df["flux_baseline_mjy"] * 0.05)
        fading_df["detection_type"] = "fading"
        
        fading_sources = fading_df[[
            "ra_deg", "dec_deg", "flux_obs_mjy", "flux_baseline_mjy",
            "flux_ratio", "significance_sigma", "detection_type"
        ]].to_dict("records")

    logger.info(
        f"Variable source detection: {len(new_sources)} new, "
        f"{len(variable_sources)} variable, {len(fading_sources)} fading"
    )

    return new_sources, variable_sources, fading_sources


def store_variable_source_candidates(
    candidates: list[dict],
    baseline_catalog: str = "NVSS",
    mosaic_id: int | None = None,
    db_path: str = DEFAULT_DB_PATH,
) -> list[int]:
    """Store transient candidates in database.

    Parameters
    ----------
    candidates : list of dict
        List of candidate dictionaries from detect_variable_sources()
    baseline_catalog : str
        Name of baseline catalog
    mosaic_id : int
        Associated mosaic product ID
    db_path : str
        Path to products database

    Returns
    -------
        list
        List of candidate IDs
    """
    from dsa110_contimg.infrastructure.database.unified import Database

    try:
        db = Database(db_path)
        current_time = time.time()
        
        insert_data = []
        for candidate in candidates:
            # Generate source name
            ra = candidate["ra_deg"]
            dec = candidate["dec_deg"]
            source_name = f"DSA_TRANSIENT_J{ra:08.4f}{dec:+09.4f}".replace(".", "")
    
            # Calculate variability index if applicable
            variability_index = None
            if candidate.get("flux_baseline_mjy") and candidate.get("flux_obs_mjy"):
                # Avoid division by zero
                base = candidate["flux_baseline_mjy"]
                obs = candidate["flux_obs_mjy"]
                if base > 0 and obs > 0:
                    flux_ratio = obs / base
                    variability_index = abs(np.log10(flux_ratio))
            
            insert_data.append((
                source_name,
                candidate["ra_deg"],
                candidate["dec_deg"],
                candidate["detection_type"],
                candidate["flux_obs_mjy"],
                candidate.get("flux_baseline_mjy"),
                candidate.get("flux_ratio"),
                candidate["significance_sigma"],
                baseline_catalog,
                current_time,
                mosaic_id,
                variability_index,
                current_time
            ))
            
        if not insert_data:
            return []

        # We need to return inserted IDs. sqlite3 executemany doesn't return list of IDs easily.
        # However, if we strongly need IDs we might have to insert one by one or fetch them back.
        # Given this is likely not time-critical compared to bulk ingestion, optimize for correctness/simplicity first.
        # But wait, the function spec returns list[int].
        # If we use executemany, we lose ability to get distinct lastrowids for each easily in standard sqlite3.
        # We can implement a transaction and insert one by one for now, but wrapped in valid transaction.
        # Or bulk insert and then query back by detected_at + properties? That's risky.
        # Let's use transaction with single inserts but using Database context helper if available or simple loop.
        # Database.execute_many is optimal but doesn't return IDs.
        # Let's revert to loop but using Database class to manage connection/transaction.
        
        candidate_ids = []
        with db.transaction() as conn:
            for params in insert_data:
                cur = conn.execute(
                    """
                    INSERT INTO variable_source_candidates (
                        source_name, ra_deg, dec_deg, detection_type,
                        flux_obs_mjy, flux_baseline_mjy, flux_ratio,
                        significance_sigma, baseline_catalog, detected_at,
                        mosaic_id, variability_index, last_updated
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    params
                )
                candidate_ids.append(cur.lastrowid)

        logger.info(f"Stored {len(candidate_ids)} transient candidates")
        return candidate_ids

    except Exception as e:
        logger.error(f"Error storing transient candidates: {e}")
        return []


def generate_variable_source_alerts(
    candidate_ids: list[int],
    alert_threshold_sigma: float = 7.0,
    db_path: str = DEFAULT_DB_PATH,
) -> list[int]:
    """Generate alerts for high-priority transient candidates.

        Alert levels:
        - CRITICAL: >10σ detection, new source
        - HIGH: >7σ detection, significant variability
        - MEDIUM: 5-7σ detection

    Parameters
    ----------
    candidate_ids : list
        List of candidate IDs to check
    alert_threshold_sigma : float
        Minimum significance for alerts [sigma]
    db_path : str
        Path to products database

    Returns
    -------
        list
        List of alert IDs created
    """
    from dsa110_contimg.infrastructure.database.unified import Database

    try:
        db = Database(db_path)
        
        if not candidate_ids:
            return []

        # Bulk query candidates
        placeholders = ",".join("?" * len(candidate_ids))
        df_candidates = db.query_df(f"""
            SELECT id, source_name, detection_type, flux_obs_mjy, flux_baseline_mjy,
                   flux_ratio, significance_sigma
            FROM variable_source_candidates
            WHERE id IN ({placeholders})
        """, tuple(candidate_ids))

        if df_candidates.empty:
            return []

        alert_data = []
        current_time = time.time()
        
        # Vectorized logic in standard python (since it's complex string formatting)
        # or just normal loop over dataframe which is faster than db roundtrips
        
        for _, row in df_candidates.iterrows():
            cid = row["id"]
            source_name = row["source_name"]
            detection_type = row["detection_type"]
            flux_obs = row["flux_obs_mjy"]
            flux_baseline = row["flux_baseline_mjy"]
            flux_ratio = row["flux_ratio"]
            significance = row["significance_sigma"]

            alert_level = None
            alert_message = None

            if significance >= 10.0 and detection_type == "new":
                alert_level = "CRITICAL"
                alert_message = (
                    f"New source {source_name}: {flux_obs:.1f} mJy ({significance:.1f}σ detection)"
                )
            elif significance >= alert_threshold_sigma:
                if detection_type in ["brightening", "fading"]:
                    alert_level = "HIGH"
                    action = "brightened" if detection_type == "brightening" else "faded"
                    
                    # Handle None/NaN baseline for safety
                    flux_bl_val = flux_baseline if pd.notna(flux_baseline) else 0.0
                    flux_ratio_val = flux_ratio if pd.notna(flux_ratio) else 0.0
                    
                    alert_message = (
                        f"Variable source {source_name}: {action} from "
                        f"{flux_bl_val:.1f} to {flux_obs:.1f} mJy "
                        f"({flux_ratio_val:.2f}×, {significance:.1f}σ)"
                    )
                elif detection_type == "new":
                    alert_level = "HIGH"
                    alert_message = (
                        f"New source {source_name}: {flux_obs:.1f} mJy "
                        f"({significance:.1f}σ detection)"
                    )
                else:
                    alert_level = "MEDIUM"
                    alert_message = (
                        f"Variable source {source_name}: {flux_obs:.1f} mJy "
                        f"({significance:.1f}σ variability)"
                    )
            
            if alert_level:
                alert_data.append((cid, alert_level, alert_message, current_time))
        
        if not alert_data:
            return []
            
        alert_ids = []
        with db.transaction() as conn:
            for params in alert_data:
                cur = conn.execute(
                    """
                    INSERT INTO variable_source_alerts (
                        candidate_id, alert_level, alert_message, created_at
                    ) VALUES (?, ?, ?, ?)
                    """,
                    params
                )
                alert_ids.append(cur.lastrowid)

        logger.info(f"Generated {len(alert_ids)} transient alerts")
        return alert_ids

    except Exception as e:
        logger.error(f"Error generating transient alerts: {e}")
        return []


def get_variable_source_candidates(
    min_significance: float = 5.0,
    detection_types: list[str] | None = None,
    limit: int = 100,
    db_path: str = DEFAULT_DB_PATH,
) -> pd.DataFrame:
    """Query transient candidates from database.

    Parameters
    ----------
    min_significance : float
        Minimum significance threshold [sigma]
    detection_types : list of str
        Filter by types (e.g., ['new', 'brightening'])
    limit : int
        Maximum number of candidates to return
    db_path : str
        Path to products database

    Returns
    -------
        pandas.DataFrame
        DataFrame with candidate information
    """
    from dsa110_contimg.infrastructure.database.unified import Database

    db = Database(db_path)
    
    query = """
        SELECT * FROM variable_source_candidates
        WHERE significance_sigma >= ?
    """
    params = [min_significance]

    if detection_types:
        placeholders = ",".join("?" * len(detection_types))
        query += f" AND detection_type IN ({placeholders})"
        params.extend(detection_types)

    query += " ORDER BY significance_sigma DESC LIMIT ?"
    params.append(limit)

    return db.query_df(query, tuple(params))


def get_variable_source_alerts(
    alert_level: str | None = None,
    acknowledged: bool = False,
    limit: int = 50,
    db_path: str = DEFAULT_DB_PATH,
) -> pd.DataFrame:
    """Query transient alerts from database.

    Parameters
    ----------
    alert_level : str
        Filter by level ('CRITICAL', 'HIGH', 'MEDIUM')
    acknowledged : bool
        If True, show only acknowledged; if False, show unacknowledged
    limit : int
        Maximum number of alerts to return
    db_path : str
        Path to products database

    Returns
    -------
        pandas.DataFrame
        DataFrame with alert information
    """
    from dsa110_contimg.infrastructure.database.unified import Database

    db = Database(db_path)

    query = "SELECT * FROM variable_source_alerts WHERE acknowledged = ?"
    params = [1 if acknowledged else 0]

    if alert_level:
        query += " AND alert_level = ?"
        params.append(alert_level)

    query += " ORDER BY created_at DESC LIMIT ?"
    params.append(limit)

    return db.query_df(query, tuple(params))


def acknowledge_alert(
    alert_id: int,
    acknowledged_by: str,
    notes: str | None = None,
    db_path: str = DEFAULT_DB_PATH,
) -> bool:
    """Mark alert as acknowledged by operator.

        Updates the alert record to indicate it has been reviewed by an operator.
        Sets acknowledged=True, records the operator name, timestamp, and optional notes.

    Parameters
    ----------
    alert_id : int
        ID of the alert to acknowledge
    acknowledged_by : str
        Username/identifier of person acknowledging
    notes : str, optional
        Optional notes about the acknowledgment
    db_path : str
        Path to products database

    Returns
    -------
        bool
        True if successful

    Raises
    ------
        ValueError
        If alert_id does not exist in database
    """
    from dsa110_contimg.infrastructure.database.unified import Database
    
    db = Database(db_path)

    try:
        # Check if alert exists
        alert = db.query_one("SELECT id FROM variable_source_alerts WHERE id = ?", (alert_id,))
        if not alert:
            raise ValueError(f"Alert ID {alert_id} not found in database")

        # Update alert
        acknowledged_at = time.time()
        
        # Use simple execute since Database commits automatically or use transaction for safety
        count = db.execute(
            """
            UPDATE variable_source_alerts
            SET acknowledged = 1,
                acknowledged_by = ?,
                acknowledged_at = ?,
                notes = CASE
                    WHEN notes IS NULL THEN ?
                    WHEN ? IS NULL THEN notes
                    ELSE notes || char(10) || '[' || datetime(?, 'unixepoch') || '] ' || ?
                END
            WHERE id = ?
            """,
            (
                acknowledged_by,
                acknowledged_at,
                notes,
                notes,
                acknowledged_at,
                notes,
                alert_id,
            ),
        )

        logger.info(f"Alert {alert_id} acknowledged by {acknowledged_by}")
        return True

    except Exception as e:
        logger.error(f"Failed to acknowledge alert {alert_id}: {e}")
        raise


def classify_candidate(
    candidate_id: int,
    classification: str,
    classified_by: str,
    notes: str | None = None,
    db_path: str = DEFAULT_DB_PATH,
) -> bool:
    """Assign classification to transient candidate.

        Updates the candidate record with a classification label chosen by an operator.
        Valid classifications: 'real', 'artifact', 'variable', 'uncertain'

    Parameters
    ----------
    candidate_id : int
        ID of the candidate to classify
    classification : str
        Classification label (real, artifact, variable, uncertain)
    classified_by : str
        Username/identifier of person classifying
    notes : str, optional
        Optional notes about the classification
    db_path : str
        Path to products database

    Returns
    -------
        bool
        True if successful

    Raises
    ------
        ValueError
        If candidate_id does not exist or classification is invalid
    """
    valid_classifications = {"real", "artifact", "variable", "uncertain"}
    if classification.lower() not in valid_classifications:
        raise ValueError(
            f"Invalid classification '{classification}'. "
            f"Must be one of: {', '.join(valid_classifications)}"
        )

    from dsa110_contimg.infrastructure.database.unified import Database
    db = Database(db_path)

    try:
        # Check if candidate exists
        cand = db.query_one("SELECT id FROM variable_source_candidates WHERE id = ?", (candidate_id,))
        if not cand:
            raise ValueError(f"Candidate ID {candidate_id} not found in database")

        # Update candidate
        timestamp = time.time()
        classification_note = f"Classified as '{classification}' by {classified_by}"
        if notes:
            classification_note += f": {notes}"

        db.execute(
            """
            UPDATE variable_source_candidates
            SET classification = ?,
                classified_by = ?,
                classified_at = ?,
                last_updated = ?,
                notes = CASE
                    WHEN notes IS NULL THEN ?
                    ELSE notes || char(10) || '[' || datetime(?, 'unixepoch') || '] ' || ?
                END
            WHERE id = ?
            """,
            (
                classification.lower(),
                classified_by,
                timestamp,
                timestamp,
                classification_note,
                timestamp,
                classification_note,
                candidate_id,
            ),
        )

        logger.info(f"Candidate {candidate_id} classified as '{classification}' by {classified_by}")
        return True

    except Exception as e:
        logger.error(f"Failed to classify candidate {candidate_id}: {e}")
        raise


def update_follow_up_status(
    item_id: int,
    item_type: str,
    status: str,
    notes: str | None = None,
    db_path: str = DEFAULT_DB_PATH,
) -> bool:
    """Update follow-up observation status.

        Updates the follow-up status for either an alert or candidate.
        Valid statuses: 'pending', 'scheduled', 'completed', 'declined'

    Parameters
    ----------
    item_id : int
        ID of the alert or candidate
    item_type : str
        Type of item - 'alert' or 'candidate'
    status : str
        Follow-up status (pending, scheduled, completed, declined)
    notes : str, optional
        Optional notes about the status update
    db_path : str
        Path to products database

    Returns
    -------
        bool
        True if successful

    Raises
    ------
        ValueError
        If item_id does not exist, item_type is invalid, or status is invalid
    """
    valid_statuses = {"pending", "scheduled", "completed", "declined"}
    if status.lower() not in valid_statuses:
        raise ValueError(f"Invalid status '{status}'. Must be one of: {', '.join(valid_statuses)}")

    item_type = item_type.lower()
    if item_type not in {"alert", "candidate"}:
        raise ValueError(f"Invalid item_type '{item_type}'. Must be 'alert' or 'candidate'")

    table_name = "variable_source_alerts" if item_type == "alert" else "variable_source_candidates"
    timestamp_field = "acknowledged_at" if item_type == "alert" else "last_updated"

    from dsa110_contimg.infrastructure.database.unified import Database
    db = Database(db_path)

    try:
        # Check if item exists
        item = db.query_one(f"SELECT id FROM {table_name} WHERE id = ?", (item_id,))
        if not item:
            raise ValueError(f"{item_type.capitalize()} ID {item_id} not found in database")

        # Update status
        timestamp = time.time()
        status_note = f"Follow-up status: {status}"
        if notes:
            status_note += f" - {notes}"

        db.execute(
            f"""
            UPDATE {table_name}
            SET follow_up_status = ?,
                {timestamp_field} = ?,
                notes = CASE
                    WHEN notes IS NULL THEN ?
                    ELSE notes || char(10) || '[' || datetime(?, 'unixepoch') || '] ' || ?
                END
            WHERE id = ?
            """,
            (status.lower(), timestamp, status_note, timestamp, status_note, item_id),
        )

        logger.info(f"{item_type.capitalize()} {item_id} follow-up status set to '{status}'")
        return True

    except Exception as e:
        logger.error(f"Failed to update follow-up status for {item_type} {item_id}: {e}")
        raise


def add_notes(
    item_id: int,
    item_type: str,
    notes: str,
    username: str,
    append: bool = True,
    db_path: str = DEFAULT_DB_PATH,
) -> bool:
    """Add or update notes for transient records.

        Adds notes to either an alert or candidate record. By default, appends to
        existing notes with timestamp. Can optionally replace all existing notes.

    Parameters
    ----------
    item_id : int
        ID of the alert or candidate
    item_type : str
        Type of item - 'alert' or 'candidate'
    notes : str
        Notes text to add
    username : str
        Username/identifier of person adding notes
    append : bool
        If True, append to existing notes; if False, replace
    db_path : str
        Path to products database

    Returns
    -------
        bool
        True if successful

    Raises
    ------
        ValueError
        If item_id does not exist or item_type is invalid
    """
    item_type = item_type.lower()
    if item_type not in {"alert", "candidate"}:
        raise ValueError(f"Invalid item_type '{item_type}'. Must be 'alert' or 'candidate'")

    table_name = "variable_source_alerts" if item_type == "alert" else "variable_source_candidates"
    timestamp_field = "acknowledged_at" if item_type == "alert" else "last_updated"

    from dsa110_contimg.infrastructure.database.unified import Database
    db = Database(db_path)

    try:
        # Check if item exists
        item = db.query_one(f"SELECT id FROM {table_name} WHERE id = ?", (item_id,))
        if not item:
            raise ValueError(f"{item_type.capitalize()} ID {item_id} not found in database")

        # Prepare note with timestamp and username
        timestamp = time.time()
        timestamped_note = f"[{username}] {notes}"

        if append:
            # Append to existing notes
            db.execute(
                f"""
                UPDATE {table_name}
                SET {timestamp_field} = ?,
                    notes = CASE
                        WHEN notes IS NULL THEN ?
                        ELSE notes || char(10) || '[' || datetime(?, 'unixepoch') || '] ' || ?
                    END
                WHERE id = ?
                """,
                (timestamp, timestamped_note, timestamp, timestamped_note, item_id),
            )
        else:
            # Replace existing notes
            db.execute(
                f"""
                UPDATE {table_name}
                SET {timestamp_field} = ?,
                    notes = ?
                WHERE id = ?
                """,
                (timestamp, timestamped_note, item_id),
            )

        logger.info(
            f"Notes {'appended to' if append else 'replaced for'} "
            f"{item_type} {item_id} by {username}"
        )
        return True

    except Exception as e:
        logger.error(f"Failed to add notes to {item_type} {item_id}: {e}")
        raise
