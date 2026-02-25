"""Blacklist variable and unsuitable sources for calibration.

This module provides functions to query external catalogs (ATNF pulsar catalog,
WISE AGN catalog) and blacklist variable sources from the calibrator registry.
"""

import logging
import os

logger = logging.getLogger(__name__)


def blacklist_atnf_pulsars(
    db_path: str = os.environ.get(
        "CAL_CATALOG_DB", "/data/dsa110-contimg/state/db/vla_calibrator_catalog.sqlite3"
    ),
    radius_deg: float = 0.1,
) -> int:
    """Blacklist known pulsars from ATNF pulsar catalog.

        Pulsars are highly variable and unsuitable for calibration.

    Parameters
    ----------
    db_path : str
        Path to calibrator registry database
    radius_deg : float
        Cross-match radius [degrees]

    Returns
    -------
        int
        Number of pulsars blacklisted
    """
    from dsa110_contimg.core.catalog.calibrator_registry import blacklist_source

    # Query ATNF pulsar catalog
    pulsars = _query_atnf_pulsars()

    if not pulsars:
        logger.warning("No pulsars found in ATNF catalog query")
        return 0

    blacklisted = 0
    for pulsar in pulsars:
        name, ra, dec = pulsar

        success = blacklist_source(
            source_name=name,
            ra_deg=ra,
            dec_deg=dec,
            reason="pulsar",
            source_type="pulsar",
            notes="ATNF pulsar catalog, variable",
            db_path=db_path,
        )

        if success:
            blacklisted += 1

    logger.info(f"Blacklisted {blacklisted} pulsars from ATNF catalog")
    return blacklisted


def _query_atnf_pulsars() -> list[tuple[str, float, float]]:
    """Query ATNF pulsar catalog for pulsar positions.

    Set SKIP_ATNF_QUERY=1 environment variable to skip network query
    and use hardcoded pulsar list (faster, avoids network timeouts).

    Returns
    -------
        List of (name, ra_deg, dec_deg) tuples
    """
    import os

    # Allow skipping network query via environment variable
    if os.environ.get("SKIP_ATNF_QUERY", "0") == "1":
        logger.info("SKIP_ATNF_QUERY=1 - using hardcoded pulsar list")
        return _get_hardcoded_pulsars()

    try:
        # Try importing psrqpy (ATNF catalog query tool)
        import psrqpy

        # Query all pulsars with valid positions
        # Note: This can be slow or hang if ATNF server is unresponsive
        logger.info("Querying ATNF pulsar catalog (this may take a few minutes)...")
        query = psrqpy.QueryATNF(
            params=["NAME", "RAJ", "DECJ"], condition="RAJ != '' && DECJ != ''"
        )

        pulsars = []
        from dsa110_contimg.common.utils.coordinates import dms_to_deg, hms_to_deg

        for i in range(len(query)):
            try:
                name = query["NAME"][i]
                ra_hms = query["RAJ"][i]
                dec_dms = query["DECJ"][i]

                # Convert to degrees
                ra_deg = hms_to_deg(ra_hms)
                dec_deg = dms_to_deg(dec_dms)

                pulsars.append((name, ra_deg, dec_deg))
            except Exception as e:
                logger.debug(f"Skipping pulsar {i}: {e}")
                continue

        logger.info(f"Retrieved {len(pulsars)} pulsars from ATNF catalog")
        return pulsars

    except ImportError:
        logger.warning("psrqpy not installed - using hardcoded pulsar list")
        return _get_hardcoded_pulsars()
    except Exception as e:
        logger.error(f"Error querying ATNF catalog: {e}")
        return _get_hardcoded_pulsars()


def _get_hardcoded_pulsars() -> list[tuple[str, float, float]]:
    """Hardcoded list of brightest pulsars for fallback.

    Returns
    -------
        List of (name, ra_deg, dec_deg) tuples
    """
    # Brightest pulsars at 1.4 GHz (most likely to contaminate calibrators)
    return [
        ("J0534+2200", 83.633, 22.014),  # Crab pulsar
        ("J0835-4510", 128.836, -45.176),  # Vela pulsar
        ("J1939+2134", 294.909, 21.574),  # PSR B1937+21
        ("J0437-4715", 69.316, -47.253),  # Brightest millisecond pulsar
        ("J0030+0451", 7.708, 4.856),
        ("J2145-0750", 326.480, -7.838),
        ("J1744-1134", 266.120, -11.574),
        ("J1713+0747", 258.287, 7.787),
    ]


def blacklist_wise_agn(
    db_path: str = os.environ.get(
        "CAL_CATALOG_DB", "/data/dsa110-contimg/state/db/vla_calibrator_catalog.sqlite3"
    ),
    radius_deg: float = 0.05,
    min_variability: float = 0.3,
) -> int:
    """Blacklist variable AGN from WISE blazar catalog.

        Blazars can be highly variable and unsuitable for calibration.

    Parameters
    ----------
    db_path : str
        Path to calibrator registry database
    radius_deg : float
        Cross-match radius [degrees]
    min_variability : float
        Minimum variability index to blacklist

    Returns
    -------
        int
        Number of AGN blacklisted
    """
    from dsa110_contimg.core.catalog.calibrator_registry import blacklist_source

    # Query WISE AGN catalog
    agn = _query_wise_agn(min_variability=min_variability)

    if not agn:
        logger.warning("No AGN found in WISE catalog query")
        return 0

    blacklisted = 0
    for source in agn:
        name, ra, dec, var = source

        success = blacklist_source(
            source_name=name,
            ra_deg=ra,
            dec_deg=dec,
            reason="variable_agn",
            source_type="AGN/blazar",
            notes=f"WISE AGN catalog, variability index={var:.2f}",
            db_path=db_path,
        )

        if success:
            blacklisted += 1

    logger.info(f"Blacklisted {blacklisted} variable AGN from WISE catalog")
    return blacklisted


def _query_wise_agn(min_variability: float = 0.3) -> list[tuple[str, float, float, float]]:
    """Query WISE AGN catalog for variable blazars.

    Returns
    -------
        List of (name, ra_deg, dec_deg, variability_index) tuples
    """
    # Note: This is a placeholder - real implementation would query VizieR
    # For now, return empty list (user can populate manually)
    logger.warning("WISE AGN query not implemented - using empty list")
    logger.info("To manually blacklist AGN, use blacklist_source() function")
    return []


def blacklist_extended_sources(
    db_path: str = os.environ.get(
        "CAL_CATALOG_DB", "/data/dsa110-contimg/state/db/vla_calibrator_catalog.sqlite3"
    ),
    max_size_arcmin: float = 1.0,
) -> int:
    """Blacklist extended sources that are unsuitable for calibration.

        Extended sources (>1 arcmin) can cause calibration errors.

    Parameters
    ----------
    db_path : str
        Path to calibrator registry database
    max_size_arcmin : float
        Maximum source size [arcmin]

    Returns
    -------
        int
        Number of extended sources blacklisted
    """
    # This would require querying catalog for source sizes
    # For now, just log a warning
    logger.warning("Extended source blacklisting not yet implemented")
    logger.info("Use compactness_score in calibrator_registry to filter extended sources")
    return 0


def run_full_blacklist_update(
    db_path: str = os.environ.get(
        "CAL_CATALOG_DB", "/data/dsa110-contimg/state/db/vla_calibrator_catalog.sqlite3"
    ),
) -> dict:
    """Run all blacklisting operations.

        This is the main function to update the blacklist from all sources.
        Run periodically (e.g., monthly) to keep blacklist up to date.

    Parameters
    ----------
    db_path : str
        Path to calibrator registry database

    Returns
    -------
        dict
        Dictionary with blacklist statistics
    """
    logger.info("Running full calibrator blacklist update...")

    results = {
        "pulsars": 0,
        "agn": 0,
        "extended": 0,
        "total": 0,
    }

    # Blacklist pulsars
    try:
        results["pulsars"] = blacklist_atnf_pulsars(db_path=db_path)
    except Exception as e:
        logger.error(f"Error blacklisting pulsars: {e}")

    # Blacklist variable AGN
    try:
        results["agn"] = blacklist_wise_agn(db_path=db_path)
    except Exception as e:
        logger.error(f"Error blacklisting AGN: {e}")

    # Blacklist extended sources
    try:
        results["extended"] = blacklist_extended_sources(db_path=db_path)
    except Exception as e:
        logger.error(f"Error blacklisting extended sources: {e}")

    results["total"] = results["pulsars"] + results["agn"] + results["extended"]

    logger.info(
        f"Blacklist update complete: {results['total']} sources "
        f"({results['pulsars']} pulsars, {results['agn']} AGN, "
        f"{results['extended']} extended)"
    )

    return results


def manual_blacklist_source(
    source_name: str,
    ra_deg: float,
    dec_deg: float,
    reason: str,
    db_path: str = os.environ.get(
        "CAL_CATALOG_DB", "/data/dsa110-contimg/state/db/vla_calibrator_catalog.sqlite3"
    ),
) -> bool:
    """Manually blacklist a source.

        Use this to blacklist sources that are found to be problematic
        through operational experience.

    Parameters
    ----------
    source_name : str
        Source identifier
    ra_deg : float
        Right ascension [degrees]
    dec_deg : float
        Declination [degrees]
    reason : str
        Reason for blacklisting
    db_path : str
        Path to calibrator registry database

    Returns
    -------
        bool
        True if successful
    """
    from dsa110_contimg.core.catalog.calibrator_registry import blacklist_source

    return blacklist_source(
        source_name=source_name,
        ra_deg=ra_deg,
        dec_deg=dec_deg,
        reason=reason,
        source_type="manual",
        notes="Manually blacklisted via operational experience",
        db_path=db_path,
    )
