"""Reference antenna selection utilities for DSA-110 calibration.

This module provides functions for selecting optimal reference antennas,
with emphasis on using outrigger antennas (103-117) for long-baseline
calibration quality.

Key Concepts:
    - DSA-110 has 117 antennas: core (1-102) and outriggers (103-117)
    - Outrigger antennas provide crucial long baselines for calibration
    - Reference antenna selection should prioritize healthy outriggers
    - CASA automatically falls back through refant chain if first fails

Usage:
    from dsa110_contimg.core.calibration.refant_selection import (
        get_default_outrigger_refants,
        select_best_outrigger_refant,
    )

    # Get default outrigger chain (no data inspection)
    refant_string = get_default_outrigger_refants()

    # Or get optimized chain based on MS antenna health (recommended)
    result = select_best_outrigger_refant(ms_path)
    refant_string = result['refant_string']
    print(f"Using {result['best_refant']} ({result['reason']})")
"""

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# DSA-110 outrigger antenna IDs (from DSA110_Station_Coordinates.csv)
# These antennas are widely separated from the core array and provide
# critical long baselines needed for high-quality calibration
OUTRIGGER_ANTENNAS = list(range(103, 118))  # 103-117 (15 antennas)

# Default priority order for outrigger reference antennas
# Prioritized by geometric position for optimal baseline coverage:
#   - Eastern outriggers (104-108): Best overall baseline coverage
#   - Northern outriggers (109-113): Good azimuthal distribution
#   - Western/peripheral (114-117, 103): Extreme baselines
DEFAULT_OUTRIGGER_PRIORITY = [
    104,
    105,
    106,
    107,
    108,  # Eastern (best coverage)
    109,
    110,
    111,
    112,
    113,  # Northern (good azimuth)
    114,
    115,
    116,
    103,
    117,  # Western/peripheral (extreme)
]


def get_default_outrigger_refants() -> str:
    """Get default outrigger reference antenna chain as CASA-format string.

    This provides the baseline fallback chain without any data inspection.
    CASA will automatically try antennas in order until it finds a healthy one.

    Returns
    -------
    str
        CASA-format string of default outrigger reference antenna chain.

    Examples
    --------
    >>> from casatasks import bandpass
    >>> refant = get_default_outrigger_refants()
    >>> bandpass(vis='obs.ms', refant=refant, ...)
    """
    return ",".join(map(str, DEFAULT_OUTRIGGER_PRIORITY))


def get_outrigger_antenna_ids() -> list[int]:
    """Get list of DSA-110 outrigger antenna IDs.

    Returns
    -------
        List of outrigger antenna IDs (103-117)

    """
    return OUTRIGGER_ANTENNAS.copy()


def analyze_antenna_health_from_ms(ms_path: str) -> list[dict[str, Any]]:
    """Analyze antenna health from MS flagging statistics (pre-calibration).

    This function examines antenna flags in the MS data BEFORE calibration,
    allowing intelligent refant selection based on data quality rather than
    using defaults. Useful for selecting the best reference antenna before
    any calibration is attempted.

    Parameters
    ----------
    ms_path : str
        Path to Measurement Set

    Returns
    -------
    list[dict[str, Any]]
        List of antenna statistics dictionaries with keys:
        - antenna_id: Antenna number
        - flagged_fraction: Fraction of visibilities flagged (0.0-1.0)
        - total_visibilities: Total number of visibility rows
        - is_outrigger: Whether this is an outrigger antenna

    Raises
    ------
    ImportError
        If casacore not available
    FileNotFoundError
        If MS doesn't exist

    Examples
    --------
    >>> stats = analyze_antenna_health_from_ms('obs.ms')
    >>> best = min(stats, key=lambda x: x['flagged_fraction'])
    >>> print(f"Best antenna: {best['antenna_id']} ({best['flagged_fraction']:.1%} flagged)")
    """
    try:
        import casacore.tables as casatables

        table = casatables.table
    except ImportError as e:
        raise ImportError("casacore.tables not available - cannot analyze antenna health") from e

    ms = Path(ms_path)
    if not ms.exists():
        raise FileNotFoundError(f"Measurement Set does not exist: {ms_path}")

    import numpy as np

    antenna_stats = {}

    with table(str(ms), readonly=True) as tb:
        antenna1 = tb.getcol("ANTENNA1")
        antenna2 = tb.getcol("ANTENNA2")
        flags = tb.getcol("FLAG")

        # Aggregate flags across all correlations/channels
        row_flagged = np.all(flags, axis=(1, 2))  # Row fully flagged if all channels/pols flagged

        # Count per antenna (from both ANTENNA1 and ANTENNA2)
        unique_ants = np.unique(np.concatenate([antenna1, antenna2]))

        for ant_id in unique_ants:
            ant_mask = (antenna1 == ant_id) | (antenna2 == ant_id)
            ant_flags = row_flagged[ant_mask]

            total = len(ant_flags)
            flagged = np.sum(ant_flags)

            if total > 0:
                flagged_fraction = flagged / total
            else:
                flagged_fraction = 1.0

            antenna_stats[int(ant_id)] = {
                "antenna_id": int(ant_id),
                "flagged_fraction": float(flagged_fraction),
                "total_visibilities": int(total),
                "is_outrigger": int(ant_id) in OUTRIGGER_ANTENNAS,
            }

    return list(antenna_stats.values())


def select_best_outrigger_refant(
    ms_path: str,
    max_flag_fraction: float = 0.3,
    prefer_eastern: bool = True,
) -> dict[str, Any]:
    """Select the best outrigger reference antenna based on MS data quality.

    This is the high-level function for automatic refant selection. It examines
    actual antenna health from the MS and returns the best choice along with
    a prioritized fallback chain.

    Parameters
    ----------
    ms_path : str
        Path to Measurement Set
    max_flag_fraction : float
        Maximum acceptable flag fraction for "healthy" antenna. Default 0.3 (30%).
    prefer_eastern : bool
        If True, prefer eastern outriggers (104-108) when health is similar.
        These have best overall baseline coverage for DSA-110. Default True.

    Returns
    -------
    dict[str, Any]
        Dictionary with:
        - best_refant: int - Best single reference antenna ID
        - refant_string: str - CASA-format comma-separated chain
        - healthy_outriggers: list[int] - All healthy outrigger IDs
        - method: str - "data_driven" or "default"
        - reason: str - Human-readable explanation

    Examples
    --------
    >>> result = select_best_outrigger_refant('obs.ms')
    >>> print(f"Using refant={result['refant_string']} ({result['reason']})")
    Using refant=105,104,106,107,108 (105 has lowest flag fraction: 2.1%)
    """
    try:
        antenna_stats = analyze_antenna_health_from_ms(ms_path)
    except Exception as e:
        logger.warning(f"Failed to analyze MS antenna health: {e}. Using defaults.")
        return {
            "best_refant": DEFAULT_OUTRIGGER_PRIORITY[0],
            "refant_string": get_default_outrigger_refants(),
            "healthy_outriggers": [],
            "method": "default",
            "reason": f"Could not analyze MS: {e}",
        }

    # Filter to outriggers only
    outrigger_stats = [s for s in antenna_stats if s["is_outrigger"]]

    if not outrigger_stats:
        logger.warning("No outrigger antennas found in MS")
        return {
            "best_refant": DEFAULT_OUTRIGGER_PRIORITY[0],
            "refant_string": get_default_outrigger_refants(),
            "healthy_outriggers": [],
            "method": "default",
            "reason": "No outriggers in MS",
        }

    # Filter to healthy antennas
    healthy = [s for s in outrigger_stats if s["flagged_fraction"] <= max_flag_fraction]

    if not healthy:
        # Fall back to least-bad antenna
        least_flagged = min(outrigger_stats, key=lambda x: x["flagged_fraction"])
        logger.warning(
            f"No outriggers below {max_flag_fraction:.0%} flagged. "
            f"Best available: {least_flagged['antenna_id']} ({least_flagged['flagged_fraction']:.1%})"
        )
        return {
            "best_refant": least_flagged["antenna_id"],
            "refant_string": str(least_flagged["antenna_id"]),
            "healthy_outriggers": [],
            "method": "fallback",
            "reason": f"All outriggers >30% flagged; {least_flagged['antenna_id']} is least bad ({least_flagged['flagged_fraction']:.1%})",
        }

    # Sort by flag fraction, with tie-breaker for eastern preference
    eastern_outriggers = set(range(104, 109))  # 104-108

    def sort_key(s):
        # Primary: flag fraction (lower is better)
        # Secondary: prefer eastern if flag fractions are close (<5% diff)
        base_score = s["flagged_fraction"]
        if prefer_eastern and s["antenna_id"] in eastern_outriggers:
            # Small bonus for eastern antennas (-0.01 to prioritize when close)
            base_score -= 0.01
        return base_score

    sorted_healthy = sorted(healthy, key=sort_key)

    # Build refant chain (top 5, or all if fewer)
    chain = [s["antenna_id"] for s in sorted_healthy[:5]]
    best = sorted_healthy[0]

    logger.info(
        f"Selected refant {best['antenna_id']} ({best['flagged_fraction']:.1%} flagged) "
        f"from {len(healthy)} healthy outriggers"
    )

    return {
        "best_refant": best["antenna_id"],
        "refant_string": ",".join(map(str, chain)),
        "healthy_outriggers": [s["antenna_id"] for s in sorted_healthy],
        "method": "data_driven",
        "reason": f"{best['antenna_id']} has lowest flag fraction: {best['flagged_fraction']:.1%}",
    }


def analyze_antenna_health_from_caltable(caltable_path: str) -> list[dict[str, Any]]:
    """Analyze antenna health from calibration table flagging statistics.

    Parameters
    ----------
    caltable_path :
        Path to CASA calibration table

    Returns
    -------
    List of antenna statistics dictionaries with keys
        - antenna_id: Antenna number
        - flagged_fraction: Fraction of solutions flagged (0.0-1.0)
        - total_solutions: Total number of solutions
        - flagged_solutions: Number of flagged solutions

    Raises
    ------
    ImportError
        If casacore not available
    FileNotFoundError
        If calibration table doesn't exist

    """
    try:
        import casacore.tables as casatables

        table = casatables.table
    except ImportError as e:
        raise ImportError("casacore.tables not available - cannot analyze antenna health") from e

    caltable = Path(caltable_path)
    if not caltable.exists():
        raise FileNotFoundError(f"Calibration table does not exist: {caltable_path}")

    import numpy as np

    antenna_stats = []

    with table(str(caltable), readonly=True) as tb:
        antenna_ids = tb.getcol("ANTENNA1")
        flags = tb.getcol("FLAG")

        unique_ants = np.unique(antenna_ids)

        for ant_id in unique_ants:
            ant_mask = antenna_ids == ant_id
            ant_flags = flags[ant_mask]

            total_solutions = ant_flags.size
            flagged_solutions = np.sum(ant_flags)
            if total_solutions > 0:
                flagged_fraction = flagged_solutions / total_solutions
            else:
                flagged_fraction = 1.0

            antenna_stats.append(
                {
                    "antenna_id": int(ant_id),
                    "flagged_fraction": float(flagged_fraction),
                    "total_solutions": int(total_solutions),
                    "flagged_solutions": int(flagged_solutions),
                }
            )

    return antenna_stats


def recommend_outrigger_refants(
    antenna_analysis: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Recommend outrigger reference antennas for DSA-110 calibration.

        Provides intelligent refant selection prioritizing healthy outrigger
        antennas. If no antenna statistics are provided, returns default priority.

    Parameters
    ----------
    antenna_analysis : Optional[List[Dict[str, Any]]]
        Optional list of antenna statistics from
        analyze_antenna_health_from_caltable() or similar.
        Each dict should have 'antenna_id' and 'flagged_fraction' keys.
        Default is None.

    Returns
    -------
        dict
        Dictionary with refant recommendations:
        - outrigger_antennas: List of all outrigger antenna IDs
        - default_refant_list: Default priority order (list of ints)
        - default_refant_string: Default chain (CASA format string)
        - recommended_refant: Best single antenna (if stats provided)
        - recommended_refant_string: Optimized chain (if stats provided)
        - healthy_outriggers: List of healthy outriggers (if stats provided)
        - problematic_outriggers: List of bad outriggers (if stats provided)
        - note: Human-readable explanation

    Examples
    --------
        >>> # Without antenna statistics (use defaults)
        >>> recs = recommend_outrigger_refants()
        >>> print(recs['default_refant_string'])
        '104,105,106,107,108,109,110,111,112,113,114,115,116,103,117'

        >>> # With antenna health analysis
        >>> from dsa110_contimg.core.calibration.refant_selection import (
        ...     analyze_antenna_health_from_caltable,
        ...     recommend_outrigger_refants
        ... )
        >>> stats = analyze_antenna_health_from_caltable('cal.bcal')
        >>> recs = recommend_outrigger_refants(stats)
        >>> print(recs['recommended_refant_string'])
        '105,104,106,107,108'  # Optimized based on antenna health
    """
    recommendations = {
        "outrigger_antennas": OUTRIGGER_ANTENNAS.copy(),
        "default_refant_list": DEFAULT_OUTRIGGER_PRIORITY.copy(),
        "default_refant_string": get_default_outrigger_refants(),
    }

    # If no antenna statistics provided, return defaults
    if not antenna_analysis:
        recommendations["recommended_refant"] = DEFAULT_OUTRIGGER_PRIORITY[0]
        recommendations["recommended_refant_string"] = recommendations["default_refant_string"]
        recommendations["note"] = "No antenna statistics available - using default priority order"
        return recommendations

    # Extract outrigger antenna stats
    outrigger_stats = [ant for ant in antenna_analysis if ant["antenna_id"] in OUTRIGGER_ANTENNAS]

    if not outrigger_stats:
        logger.warning("No outrigger antennas found in antenna statistics")
        recommendations["recommended_refant"] = DEFAULT_OUTRIGGER_PRIORITY[0]
        recommendations["recommended_refant_string"] = recommendations["default_refant_string"]
        recommendations["note"] = "No outrigger stats found - using default priority"
        return recommendations

    # Sort by flagged fraction (lower is better)
    healthy_outriggers = sorted(outrigger_stats, key=lambda x: x["flagged_fraction"])

    # Filter to reasonably healthy antennas (<50% flagged)
    good_outriggers = [ant for ant in healthy_outriggers if ant["flagged_fraction"] < 0.5]

    if good_outriggers:
        # Determine health status
        def get_health_status(frac):
            if frac < 0.1:
                return "excellent"
            elif frac < 0.3:
                return "good"
            else:
                return "fair"

        recommendations["healthy_outriggers"] = [
            {
                "antenna_id": ant["antenna_id"],
                "flagged_fraction": ant["flagged_fraction"],
                "health_status": get_health_status(ant["flagged_fraction"]),
            }
            for ant in good_outriggers
        ]

        # Build optimized refant string from healthy antennas
        top_5 = [str(ant["antenna_id"]) for ant in good_outriggers[:5]]
        top_ant = good_outriggers[0]
        recommendations["recommended_refant"] = top_ant["antenna_id"]
        recommendations["recommended_refant_string"] = ",".join(top_5)

        note = (
            f"Top choice: antenna {top_ant['antenna_id']} "
            f"({top_ant['flagged_fraction'] * 100:.1f}% flagged)"
        )
        recommendations["note"] = note
    else:
        recommendations["warning"] = "No healthy outrigger antennas found (<50% flagged)"
        recommendations["recommended_refant"] = DEFAULT_OUTRIGGER_PRIORITY[0]
        recommendations["recommended_refant_string"] = recommendations["default_refant_string"]
        recommendations["note"] = "Using default priority - check array status"

    # Identify problematic outriggers (>80% flagged)
    bad_outriggers = [ant for ant in outrigger_stats if ant["flagged_fraction"] > 0.8]

    if bad_outriggers:
        recommendations["problematic_outriggers"] = [
            {
                "antenna_id": ant["antenna_id"],
                "flagged_fraction": ant["flagged_fraction"],
            }
            for ant in bad_outriggers
        ]

    return recommendations


def recommend_refants_from_ms(
    ms_path: str,
    caltable_path: str | None = None,
    use_defaults_on_error: bool = True,
) -> str:
    """Get recommended refant string for calibration based on MS/caltable.

        This is the high-level convenience function for CLI/orchestrator usage.
        It attempts to analyze antenna health and provide optimized refant chain,
        falling back to defaults if analysis fails.

    Parameters
    ----------
    ms_path : str
        Path to Measurement Set.
    caltable_path : Optional[str]
        Optional path to calibration table for health analysis.
        Default is None.
    use_defaults_on_error : bool
        If True, use default refants if analysis fails.
        Default is True.

    Returns
    -------
        str
        CASA-format refant string (comma-separated antenna IDs).

    Examples
    --------
        >>> # Use defaults (no caltable inspection)
        >>> refant = recommend_refants_from_ms('obs.ms')
        >>> print(refant)
        '104,105,106,107,108,109,110,111,112,113,114,115,116,103,117'

        >>> # Optimize based on previous calibration
        >>> refant = recommend_refants_from_ms('obs.ms', 'prev.bcal')
        >>> print(refant)
        '105,104,106,107,108'  # Best 5 based on health
    """
    # If no caltable provided, return defaults
    if not caltable_path:
        logger.info("No calibration table provided - using default outrigger chain")
        return get_default_outrigger_refants()

    try:
        # Analyze antenna health from caltable
        antenna_stats = analyze_antenna_health_from_caltable(caltable_path)

        # Get recommendations
        recs = recommend_outrigger_refants(antenna_stats)

        # Use recommended chain if available, otherwise default
        refant_string = recs.get("recommended_refant_string", recs["default_refant_string"])

        logger.info(
            f"Recommended refant chain: {refant_string} "
            f"({recs.get('note', 'optimized from antenna health')})"
        )

        return refant_string

    except Exception as e:
        if use_defaults_on_error:
            logger.warning(f"Failed to analyze antenna health: {e}. Using default outrigger chain.")
            return get_default_outrigger_refants()
        else:
            raise


def format_refant_for_casa(antenna_ids: list[int]) -> str:
    """Format list of antenna IDs as CASA refant parameter string.

    Parameters
    ----------
    antenna_ids : List[int]
        List of antenna IDs (integers).

    Returns
    -------
    str
        Comma-separated string for CASA refant parameter.

    Examples
    --------
    >>> format_refant_for_casa([104, 105, 106])
    '104,105,106'
    """
    return ",".join(str(a) for a in antenna_ids)
