"""Coverage-aware catalog selection.

This module provides functions to determine which catalogs are available
for a given sky position and recommend optimal catalogs based on coverage,
resolution, sensitivity, and intended use.

Implements Proposal #8: Coverage-Aware Catalog Selection
"""

import logging

logger = logging.getLogger(__name__)


# Catalog coverage definitions [degrees]
CATALOG_COVERAGE = {
    "nvss": {
        "name": "NVSS",
        "frequency_ghz": 1.4,
        "dec_min": -40.0,
        "dec_max": 90.0,
        "resolution_arcsec": 45.0,
        "typical_rms_mjy": 0.45,
        "flux_limit_mjy": 2.5,  # 5-sigma
        "best_for": ["general", "calibration", "transients"],
        "notes": "All-sky survey Dec > -40°, excellent for calibration",
    },
    "first": {
        "name": "FIRST",
        "frequency_ghz": 1.4,
        "dec_min": -40.0,
        "dec_max": 90.0,
        "resolution_arcsec": 5.0,
        "typical_rms_mjy": 0.15,
        "flux_limit_mjy": 1.0,
        "best_for": ["astrometry", "morphology", "compact"],
        "notes": "High-resolution survey, 10,000 sq deg, excellent astrometry",
    },
    "racs": {
        "name": "RACS",
        "frequency_ghz": 0.888,
        "dec_min": -90.0,
        "dec_max": 41.0,
        "resolution_arcsec": 15.0,
        "typical_rms_mjy": 0.25,
        "flux_limit_mjy": 1.5,
        "best_for": ["southern", "spectral_index", "general"],
        "notes": "Southern sky survey Dec < +41°, ASKAP data",
    },
    "rax": {  # Alias for RACS (used internally in code)
        "name": "RACS",
        "frequency_ghz": 0.888,
        "dec_min": -90.0,
        "dec_max": 41.0,
        "resolution_arcsec": 15.0,
        "typical_rms_mjy": 0.25,
        "flux_limit_mjy": 1.5,
        "best_for": ["southern", "spectral_index", "general"],
        "notes": "Southern sky survey Dec < +41°, ASKAP data (alias: rax)",
    },
    "vlass": {
        "name": "VLASS",
        "frequency_ghz": 3.0,
        "dec_min": -40.0,
        "dec_max": 90.0,
        "resolution_arcsec": 2.5,
        "typical_rms_mjy": 0.12,
        "flux_limit_mjy": 1.0,
        "best_for": ["spectral_index", "high_freq", "morphology"],
        "notes": "VLA Sky Survey, ongoing, excellent for spectral indices",
    },
    "sumss": {
        "name": "SUMSS",
        "frequency_ghz": 0.843,
        "dec_min": -90.0,
        "dec_max": -30.0,
        "resolution_arcsec": 45.0,
        "typical_rms_mjy": 1.0,
        "flux_limit_mjy": 6.0,
        "best_for": ["southern", "general"],
        "notes": "Sydney University Molonglo Sky Survey, Dec < -30°",
    },
}


def get_catalog_coverage(catalog_type: str) -> dict | None:
    """Get coverage information for a specific catalog.

    Parameters
    ----------
    catalog_type : str
        Catalog identifier (e.g., 'nvss', 'first', 'racs')

    Returns
    -------
        dict or None
        Dictionary with coverage info, or None if catalog unknown
    """
    return CATALOG_COVERAGE.get(catalog_type.lower())


def is_position_in_catalog(
    ra_deg: float, dec_deg: float, catalog_type: str, margin_deg: float = 0.0
) -> bool:
    """Check if a sky position is covered by a catalog.

    Parameters
    ----------
    ra_deg : float
        Right ascension [degrees]
    dec_deg : float
        Declination [degrees]
    catalog_type : str
        Catalog identifier
    margin_deg : float
        Safety margin to subtract from coverage edges [degrees]

    Returns
    -------
        bool
        True if position is covered by catalog
    """
    coverage = get_catalog_coverage(catalog_type)
    if coverage is None:
        logger.warning(f"Unknown catalog: {catalog_type}")
        return False

    dec_min = coverage["dec_min"] + margin_deg
    dec_max = coverage["dec_max"] - margin_deg

    # RA coverage is all-sky for these surveys
    in_coverage = dec_min <= dec_deg <= dec_max

    return in_coverage


def get_available_catalogs(ra_deg: float, dec_deg: float, margin_deg: float = 1.0) -> list[str]:
    """Get list of catalogs that cover a given position.

    Parameters
    ----------
    ra_deg : float
        Right ascension [degrees]
    dec_deg : float
        Declination [degrees]
    margin_deg : float
        Safety margin from coverage edges [degrees]

    Returns
    -------
        list of str
        List of catalog identifiers that cover this position
    """
    available = []

    for catalog_type in CATALOG_COVERAGE.keys():
        if is_position_in_catalog(ra_deg, dec_deg, catalog_type, margin_deg):
            available.append(catalog_type)

    return available


def recommend_catalogs(
    ra_deg: float,
    dec_deg: float,
    purpose: str = "general",
    require_spectral_index: bool = False,
    min_resolution_arcsec: float | None = None,
    max_resolution_arcsec: float | None = None,
) -> list[dict]:
    """Recommend optimal catalogs for a position and purpose.

        This is the main function for intelligent catalog selection. It considers:
        - Sky coverage (declination limits)
        - Survey purpose (calibration, astrometry, spectral index, etc.)
        - Resolution requirements
        - Multi-frequency coverage for spectral indices

    Parameters
    ----------
    ra_deg : float
        Right ascension [degrees]
    dec_deg : float
        Declination [degrees]
    purpose : str
        Intended use - one of:
        - 'general': General source queries
        - 'calibration': Selecting calibrators (prefer flat-spectrum)
        - 'astrometry': Accurate positions (prefer high-resolution)
        - 'spectral_index': Multi-frequency matching
        - 'morphology': Detailed source structure
        - 'transients': Transient/variable source searches
    require_spectral_index : bool
        If True, only recommend catalogs with complementary frequency coverage
    min_resolution_arcsec : float
        Minimum acceptable resolution [arcsec]
    max_resolution_arcsec : float
        Maximum acceptable resolution [arcsec]

    Returns
    -------
        list of dict
        List of recommended catalogs (dicts) sorted by priority, each containing:
        - catalog_type: Identifier (e.g., 'nvss')
        - name: Human-readable name
        - priority: Recommendation priority (1=highest)
        - reason: Why this catalog is recommended
        - coverage_info: Full coverage dictionary
    """
    available = get_available_catalogs(ra_deg, dec_deg, margin_deg=1.0)

    if len(available) == 0:
        logger.warning(f"No catalogs available at RA={ra_deg:.2f}, Dec={dec_deg:.2f}")
        return []

    recommendations = []

    for catalog_type in available:
        coverage = CATALOG_COVERAGE[catalog_type]

        # Filter by resolution if specified
        if min_resolution_arcsec is not None:
            if coverage["resolution_arcsec"] < min_resolution_arcsec:
                continue
        if max_resolution_arcsec is not None:
            if coverage["resolution_arcsec"] > max_resolution_arcsec:
                continue

        # Calculate priority based on purpose
        priority = 999
        reason = ""

        if purpose in coverage["best_for"]:
            priority = 1
            reason = f"Optimized for {purpose}"
        elif purpose == "general":
            priority = 2
            reason = "Good general-purpose catalog"
        elif purpose == "calibration":
            # NVSS and FIRST excellent for calibration
            if catalog_type in ["nvss", "first"]:
                priority = 1
                reason = "Excellent calibrator database"
            else:
                priority = 3
                reason = "Can be used for calibration"
        elif purpose == "astrometry":
            # Prefer high-resolution catalogs
            if coverage["resolution_arcsec"] <= 10.0:
                priority = 1
                reason = f'High resolution ({coverage["resolution_arcsec"]:.1f}")'
            else:
                priority = 3
                reason = "Lower resolution, less accurate astrometry"
        elif purpose == "spectral_index":
            # Prefer catalogs at different frequencies
            priority = 2
            reason = f"Provides {coverage['frequency_ghz']} GHz data"
        elif purpose == "morphology":
            # Prefer high resolution
            if coverage["resolution_arcsec"] <= 15.0:
                priority = 1
                reason = f'Good resolution ({coverage["resolution_arcsec"]:.1f}")'
            else:
                priority = 3
                reason = "Lower resolution"
        elif purpose == "transients":
            # NVSS is baseline for transients
            if catalog_type == "nvss":
                priority = 1
                reason = "Standard transient baseline catalog"
            else:
                priority = 2
                reason = "Can be used for transient searches"
        else:
            priority = 5
            reason = "Available catalog"

        recommendations.append(
            {
                "catalog_type": catalog_type,
                "name": coverage["name"],
                "priority": priority,
                "reason": reason,
                "coverage_info": coverage,
            }
        )

    # Sort by priority (lower is better)
    recommendations.sort(key=lambda x: x["priority"])

    # If spectral index required, ensure we have multi-frequency coverage
    if require_spectral_index and len(recommendations) >= 2:
        # Keep only catalogs with different frequencies
        unique_freqs = []
        filtered = []
        for rec in recommendations:
            freq = rec["coverage_info"]["frequency_ghz"]
            if freq not in unique_freqs:
                unique_freqs.append(freq)
                filtered.append(rec)
                if len(filtered) >= 3:  # 3 catalogs is usually enough
                    break
        recommendations = filtered

    return recommendations


def get_catalog_overlap_region(catalog_types: list[str]) -> tuple[float, float]:
    """Get the declination range where all specified catalogs overlap.

    Parameters
    ----------
    catalog_types : list of str
        List of catalog identifiers

    Returns
    -------
        tuple
        Tuple of (dec_min, dec_max) for overlap region
        Returns (None, None) if no overlap exists
    """
    if not catalog_types:
        return None, None

    dec_min = -90.0
    dec_max = 90.0

    for catalog_type in catalog_types:
        coverage = get_catalog_coverage(catalog_type)
        if coverage is None:
            continue

        dec_min = max(dec_min, coverage["dec_min"])
        dec_max = min(dec_max, coverage["dec_max"])

    if dec_min >= dec_max:
        logger.warning(f"No overlap between catalogs: {catalog_types}")
        return None, None

    return dec_min, dec_max


def suggest_catalog_for_declination(dec_deg: float, purpose: str = "general") -> str:
    """Suggest best single catalog for a given declination.

        Convenience function for quick catalog selection.

    Parameters
    ----------
    dec_deg : float
        Declination [degrees]
    purpose : str
        Intended use (see recommend_catalogs for options)

    Returns
    -------
        str
        Catalog identifier (e.g., 'nvss', 'racs') or 'none' if no coverage
    """
    # Use equator for RA (doesn't matter for these all-sky surveys)
    recommendations = recommend_catalogs(ra_deg=0.0, dec_deg=dec_deg, purpose=purpose)

    if len(recommendations) == 0:
        return "none"

    return recommendations[0]["catalog_type"]


def validate_catalog_choice(
    catalog_type: str, ra_deg: float, dec_deg: float
) -> tuple[bool, str | None]:
    """Validate if a catalog is appropriate for a given position.

    Parameters
    ----------
    catalog_type : str
        Catalog identifier
    ra_deg : float
        Right ascension [degrees]
    dec_deg : float
        Declination [degrees]

    Returns
    -------
        tuple
        Tuple of (is_valid, error_message). If valid, error_message is None.
    """
    coverage = get_catalog_coverage(catalog_type)

    if coverage is None:
        return False, f"Unknown catalog: {catalog_type}"

    if not is_position_in_catalog(ra_deg, dec_deg, catalog_type, margin_deg=0.0):
        return (
            False,
            f"{coverage['name']} does not cover Dec={dec_deg:.2f}° "
            f"(range: {coverage['dec_min']:.0f}° to {coverage['dec_max']:.0f}°)",
        )

    return True, None


def get_catalog_summary() -> dict[str, dict]:
    """Get summary of all available catalogs.

    Returns
    -------
        Dictionary mapping catalog_type to coverage info
    """
    return CATALOG_COVERAGE.copy()


def print_coverage_summary():
    """Print human-readable summary of catalog coverage.

    Useful for debugging and documentation.
    """
    print("\n" + "=" * 70)
    print("DSA-110 CATALOG COVERAGE SUMMARY")
    print("=" * 70)

    for cat_type, info in sorted(CATALOG_COVERAGE.items()):
        print(f"\n{info['name']} ({cat_type.upper()})")
        print(f"  Frequency:   {info['frequency_ghz']} GHz")
        print(f"  Declination: {info['dec_min']:+.0f}° to {info['dec_max']:+.0f}°")
        print(f'  Resolution:  {info["resolution_arcsec"]:.1f}"')
        print(f"  RMS:         {info['typical_rms_mjy']:.2f} mJy")
        print(f"  Limit:       {info['flux_limit_mjy']:.1f} mJy (5σ)")
        print(f"  Best for:    {', '.join(info['best_for'])}")
        print(f"  Notes:       {info['notes']}")

    print("\n" + "=" * 70)
    print("\nOVERLAP REGIONS:")
    print("-" * 70)

    # Show overlap regions
    regions = [
        (["nvss", "first", "vlass"], "Northern (high-res)"),
        (["nvss", "racs"], "Equatorial overlap"),
        (["racs", "sumss"], "Southern"),
    ]

    for cats, desc in regions:
        dec_min, dec_max = get_catalog_overlap_region(cats)
        if dec_min is not None:
            print(
                f"{desc:25s} ({' + '.join([c.upper() for c in cats]):20s}): "
                f"{dec_min:+.0f}° to {dec_max:+.0f}°"
            )

    print("=" * 70 + "\n")
