"""External catalog queries for source identification and cross-matching.

    This module provides functions to query external astronomical catalogs:
    - SIMBAD: Object identification and basic properties
    - NED: Extragalactic database with redshifts and classifications
    - Gaia: Astrometry and parallax measurements

    These queries are useful for:
    - Source identification and classification
    - Redshift determination (NED)
    - Proper motion measurements (Gaia)
    - Multi-wavelength cross-matching

Examples
--------
    >>> from dsa110_contimg.core.catalog.external import simbad_search, ned_search, gaia_search
    >>> from astropy.coordinates import SkyCoord
    >>> import astropy.units as u
    >>>
    >>> coord = SkyCoord(ra=123.456*u.deg, dec=12.345*u.deg)
    >>> simbad_result = simbad_search(coord, radius_arcsec=5.0)
    >>> ned_result = ned_search(coord, radius_arcsec=5.0)
    >>> gaia_result = gaia_search(coord, radius_arcsec=5.0)
"""

import logging

import astropy.units as u
from astropy.coordinates import SkyCoord

logger = logging.getLogger(__name__)

# Lazy import flag - astroquery can be slow to import (downloads remote tables)
HAS_ASTROQUERY = None  # None = unchecked, True/False after first check


def _check_astroquery():
    """Check if astroquery is available (lazy import)."""
    global HAS_ASTROQUERY
    if HAS_ASTROQUERY is None:
        try:
            from astroquery.gaia import Gaia  # noqa: F401
            from astroquery.ned import Ned  # noqa: F401
            from astroquery.simbad import Simbad  # noqa: F401

            HAS_ASTROQUERY = True
        except ImportError:
            HAS_ASTROQUERY = False
            logger.warning(
                "astroquery not available. External catalog queries will not work. "
                "Install with: pip install astroquery"
            )
    return HAS_ASTROQUERY


def simbad_search(
    coord: SkyCoord,
    radius_arcsec: float = 5.0,
    timeout: float = 30.0,
) -> dict[str, any] | None:
    """Query SIMBAD for object identification.

        SIMBAD (Set of Identifications, Measurements and Bibliography for Astronomical Data)
        provides object identification, basic properties, and bibliographic references.

    Parameters
    ----------
    coord : astropy.coordinates.SkyCoord
        SkyCoord object with source position
    radius_arcsec : float, optional
        Search radius in arcseconds (default: 5.0)
    timeout : float, optional
        Query timeout in seconds (default: 30.0)

    Returns
    -------
        dict or None
        Dictionary with SIMBAD results, or None if no match found or error.
        Keys:
        - main_id: Primary identifier
        - otype: Object type (e.g., 'Radio', 'QSO', 'Star')
        - ra: Right ascension (degrees)
        - dec: Declination (degrees)
        - separation_arcsec: Separation from query position
        - flux_v: V-band magnitude (if available)
        - redshift: Redshift (if available)
        - names: List of alternative names
        - bibcode: Bibliographic code (if available)

    Examples
    --------
        >>> from astropy.coordinates import SkyCoord
        >>> import astropy.units as u
        >>> coord = SkyCoord(ra=123.456*u.deg, dec=12.345*u.deg)
        >>> result = simbad_search(coord, radius_arcsec=10.0)
        >>> if result:
        ...     print(f"Found: {result['main_id']}, type: {result['otype']}")
    """
    if not _check_astroquery():
        logger.warning("astroquery not available, skipping SIMBAD query")
        return None

    from astroquery.simbad import Simbad

    try:
        # Configure SIMBAD query
        Simbad.TIMEOUT = timeout
        Simbad.add_votable_fields("otype", "flux(V)", "z_value", "ids")

        # Perform query
        result_table = Simbad.query_region(coord, radius=f"{radius_arcsec}arcsec")

        if result_table is None or len(result_table) == 0:
            return None

        # Get closest match (first row)
        row = result_table[0]

        # Calculate separation
        simbad_coord = SkyCoord(ra=row["RA"], dec=row["DEC"], unit=(u.hourangle, u.deg))
        separation = coord.separation(simbad_coord).to(u.arcsec).value

        # Extract names
        names = []
        if "MAIN_ID" in row.colnames:
            names.append(str(row["MAIN_ID"]))
        if "IDS" in row.colnames and row["IDS"]:
            # IDs field contains semicolon-separated names
            ids_str = str(row["IDS"])
            names.extend([n.strip() for n in ids_str.split(";")])

        result = {
            "main_id": str(row["MAIN_ID"]) if "MAIN_ID" in row.colnames else None,
            "otype": str(row["OTYPE"]) if "OTYPE" in row.colnames else None,
            "ra": simbad_coord.ra.deg,
            "dec": simbad_coord.dec.deg,
            "separation_arcsec": separation,
            "flux_v": (
                float(row["FLUX_V"]) if "FLUX_V" in row.colnames and row["FLUX_V"] else None
            ),
            "redshift": (
                float(row["Z_VALUE"]) if "Z_VALUE" in row.colnames and row["Z_VALUE"] else None
            ),
            "names": names,
            "bibcode": (str(row["COO_BIBCODE"]) if "COO_BIBCODE" in row.colnames else None),
        }

        logger.debug(f"SIMBAD query successful: {result['main_id']} at {separation:.2f} arcsec")
        return result

    except Exception as e:
        logger.warning(f"SIMBAD query failed for {coord}: {e}")
        return None


def ned_search(
    coord: SkyCoord,
    radius_arcsec: float = 5.0,
    timeout: float = 30.0,
) -> dict[str, any] | None:
    """Query NED (NASA/IPAC Extragalactic Database) for extragalactic objects.

        NED provides redshifts, classifications, and multi-wavelength data for
        extragalactic sources.

    Parameters
    ----------
    coord : astropy.coordinates.SkyCoord
        SkyCoord object with source position
    radius_arcsec : float, optional
        Search radius in arcseconds (default: 5.0)
    timeout : float, optional
        Query timeout in seconds (default: 30.0)

    Returns
    -------
        dict or None
        Dictionary with NED results, or None if no match found or error.
        Keys:
        - ned_name: NED object name
        - object_type: Object classification
        - ra: Right ascension (degrees)
        - dec: Declination (degrees)
        - separation_arcsec: Separation from query position
        - redshift: Redshift value
        - redshift_type: Redshift type (e.g., 'z', 'v', 'q')
        - velocity: Recession velocity (km/s, if available)
        - distance: Distance (Mpc, if available)
        - magnitude: Optical magnitude (if available)
        - flux_1_4ghz: 1.4 GHz flux density (mJy, if available)

    Examples
    --------
        >>> from astropy.coordinates import SkyCoord
        >>> import astropy.units as u
        >>> coord = SkyCoord(ra=123.456*u.deg, dec=12.345*u.deg)
        >>> result = ned_search(coord, radius_arcsec=10.0)
        >>> if result and result['redshift']:
        ...     print(f"Found: {result['ned_name']}, z={result['redshift']}")
    """
    if not _check_astroquery():
        logger.warning("astroquery not available, skipping NED query")
        return None

    from astroquery.ned import Ned

    try:
        # Configure NED query
        Ned.TIMEOUT = timeout

        # Perform query
        result_table = Ned.query_region(coord, radius=f"{radius_arcsec}arcsec")

        if result_table is None or len(result_table) == 0:
            return None

        # Get closest match (first row)
        row = result_table[0]

        # Calculate separation
        ned_coord = SkyCoord(ra=row["RA"], dec=row["DEC"], unit=u.deg)
        separation = coord.separation(ned_coord).to(u.arcsec).value

        result = {
            "ned_name": (str(row["Object Name"]) if "Object Name" in row.colnames else None),
            "object_type": str(row["Type"]) if "Type" in row.colnames else None,
            "ra": ned_coord.ra.deg,
            "dec": ned_coord.dec.deg,
            "separation_arcsec": separation,
            "redshift": (
                float(row["Redshift"]) if "Redshift" in row.colnames and row["Redshift"] else None
            ),
            "redshift_type": (
                str(row["Redshift Type"]) if "Redshift Type" in row.colnames else None
            ),
            "velocity": (
                float(row["Velocity"]) if "Velocity" in row.colnames and row["Velocity"] else None
            ),
            "distance": (
                float(row["Distance"]) if "Distance" in row.colnames and row["Distance"] else None
            ),
            "magnitude": (
                float(row["Magnitude"])
                if "Magnitude" in row.colnames and row["Magnitude"]
                else None
            ),
            "flux_1_4ghz": (
                float(row["1.4GHz"]) if "1.4GHz" in row.colnames and row["1.4GHz"] else None
            ),
        }

        logger.debug(f"NED query successful: {result['ned_name']} at {separation:.2f} arcsec")
        return result

    except Exception as e:
        logger.warning(f"NED query failed for {coord}: {e}")
        return None


def gaia_search(
    coord: SkyCoord,
    radius_arcsec: float = 5.0,
    timeout: float = 30.0,
    max_results: int = 1,
) -> dict[str, any] | None:
    """Query Gaia for astrometry and parallax measurements.

        Gaia provides high-precision astrometry, proper motions, and parallaxes
        for stars and other objects.

    Parameters
    ----------
    coord : astropy.coordinates.SkyCoord
        SkyCoord object with source position
    radius_arcsec : float, optional
        Search radius in arcseconds (default: 5.0)
    timeout : float, optional
        Query timeout in seconds (default: 30.0)
    max_results : int, optional
        Maximum number of results to return (default: 1)

    Returns
    -------
        dict or None
        Dictionary with Gaia results, or None if no match found or error.
        Keys:
        - source_id: Gaia source ID
        - ra: Right ascension (degrees)
        - dec: Declination (degrees)
        - separation_arcsec: Separation from query position
        - parallax: Parallax (mas)
        - parallax_error: Parallax error (mas)
        - pmra: Proper motion in RA (mas/yr)
        - pmdec: Proper motion in Dec (mas/yr)
        - pmra_error: Proper motion RA error (mas/yr)
        - pmdec_error: Proper motion Dec error (mas/yr)
        - phot_g_mean_mag: G-band magnitude
        - phot_bp_mean_mag: BP-band magnitude
        - phot_rp_mean_mag: RP-band magnitude
        - distance: Distance estimate (pc, if parallax > 0)

    Examples
    --------
        >>> from astropy.coordinates import SkyCoord
        >>> import astropy.units as u
        >>> coord = SkyCoord(ra=123.456*u.deg, dec=12.345*u.deg)
        >>> result = gaia_search(coord, radius_arcsec=10.0)
        >>> if result and result['parallax']:
        ...     print(f"Distance: {result['distance']:.1f} pc")
    """
    if not _check_astroquery():
        logger.warning("astroquery not available, skipping Gaia query")
        return None

    from astroquery.gaia import Gaia

    try:
        # Configure Gaia query
        Gaia.TIMEOUT = timeout

        # Build ADQL query
        radius_deg = radius_arcsec / 3600.0
        query = f"""
        SELECT TOP {max_results}
            source_id, ra, dec,
            parallax, parallax_error,
            pmra, pmdec, pmra_error, pmdec_error,
            phot_g_mean_mag, phot_bp_mean_mag, phot_rp_mean_mag
        FROM gaiadr3.gaia_source
        WHERE 1=CONTAINS(
            POINT('ICRS', ra, dec),
            CIRCLE('ICRS', {coord.ra.deg}, {coord.dec.deg}, {radius_deg})
        )
        ORDER BY
            SQRT(POWER(ra - {coord.ra.deg}, 2) + POWER(dec - {coord.dec.deg}, 2))
        """

        # Perform query
        job = Gaia.launch_job(query)
        result_table = job.get_results()

        if result_table is None or len(result_table) == 0:
            return None

        # Get closest match (first row)
        row = result_table[0]

        # Calculate separation
        gaia_coord = SkyCoord(ra=row["ra"], dec=row["dec"], unit=u.deg)
        separation = coord.separation(gaia_coord).to(u.arcsec).value

        # Calculate distance from parallax
        distance = None
        if row["parallax"] and row["parallax"] > 0:
            distance = 1000.0 / row["parallax"]  # Convert mas to pc

        result = {
            "source_id": str(row["source_id"]),
            "ra": float(row["ra"]),
            "dec": float(row["dec"]),
            "separation_arcsec": separation,
            "parallax": float(row["parallax"]) if row["parallax"] else None,
            "parallax_error": (float(row["parallax_error"]) if row["parallax_error"] else None),
            "pmra": float(row["pmra"]) if row["pmra"] else None,
            "pmdec": float(row["pmdec"]) if row["pmdec"] else None,
            "pmra_error": float(row["pmra_error"]) if row["pmra_error"] else None,
            "pmdec_error": float(row["pmdec_error"]) if row["pmdec_error"] else None,
            "phot_g_mean_mag": (float(row["phot_g_mean_mag"]) if row["phot_g_mean_mag"] else None),
            "phot_bp_mean_mag": (
                float(row["phot_bp_mean_mag"]) if row["phot_bp_mean_mag"] else None
            ),
            "phot_rp_mean_mag": (
                float(row["phot_rp_mean_mag"]) if row["phot_rp_mean_mag"] else None
            ),
            "distance": distance,
        }

        logger.debug(f"Gaia query successful: {result['source_id']} at {separation:.2f} arcsec")
        return result

    except Exception as e:
        logger.warning(f"Gaia query failed for {coord}: {e}")
        return None


def query_all_catalogs(
    coord: SkyCoord,
    radius_arcsec: float = 5.0,
    timeout: float = 30.0,
) -> dict[str, dict[str, any] | None]:
    """Query all external catalogs (SIMBAD, NED, Gaia) simultaneously.

        This is a convenience function that queries all three catalogs and returns
        results in a single dictionary.

    Parameters
    ----------
    coord : astropy.coordinates.SkyCoord
        SkyCoord object with source position
    radius_arcsec : float, optional
        Search radius in arcseconds (default: 5.0)
    timeout : float, optional
        Query timeout in seconds (default: 30.0)

    Returns
    -------
        dict
        Dictionary with keys 'simbad', 'ned', 'gaia', each containing
        the result from the respective catalog (or None if no match/error).

    Examples
    --------
        >>> from astropy.coordinates import SkyCoord
        >>> import astropy.units as u
        >>> coord = SkyCoord(ra=123.456*u.deg, dec=12.345*u.deg)
        >>> results = query_all_catalogs(coord)
        >>> if results["simbad"]:
        ...     print(f"SIMBAD: {results['simbad']['main_id']}")
        >>> if results["ned"] and results["ned"]["redshift"]:
        ...     print(f"NED redshift: {results['ned']['redshift']}")
    """
    results = {
        "simbad": simbad_search(coord, radius_arcsec=radius_arcsec, timeout=timeout),
        "ned": ned_search(coord, radius_arcsec=radius_arcsec, timeout=timeout),
        "gaia": gaia_search(coord, radius_arcsec=radius_arcsec, timeout=timeout),
    }

    return results
