"""
Multi-wavelength catalog search tools.

Repurposed from vast-mw (https://github.com/askap-vast/vast-mw) by David Kaplan.
Provides unified access to various astronomical catalogs (Gaia, Simbad, NVSS, etc.)
for source cross-matching and classification.
"""

import logging
import warnings

from astropy import units as u
from astropy.coordinates import EarthLocation, SkyCoord, get_body, solar_system_ephemeris
from astropy.time import Time

logger = logging.getLogger(__name__)

# Lazy imports for network-dependent modules
# These are imported on first use to avoid network calls at module load time
_astroquery_loaded = False
Casda = None
Gaia = None
Simbad = None
vo = None
requests = None


def _ensure_astroquery():
    """Lazily load astroquery and related modules."""
    global _astroquery_loaded, Casda, Gaia, Simbad, vo, requests
    if _astroquery_loaded:
        return

    import pyvo as _vo
    import requests as _requests
    from astroquery.casda import Casda as _Casda
    from astroquery.gaia import Gaia as _Gaia
    from astroquery.simbad import Simbad as _Simbad

    requests = _requests
    vo = _vo
    Casda = _Casda
    Gaia = _Gaia
    Simbad = _Simbad

    # Configure Gaia
    Gaia.MAIN_GAIA_TABLE = GAIA_MAIN_TABLE

    # Configure Simbad (done below after constant definitions)
    _astroquery_loaded = True


# Constants
GAIA_MAIN_TABLE = "gaiadr3.gaia_source"
PULSAR_SCRAPER_URL = "https://pulsar.cgca-hub.org/api"
SIMBAD_URL = "https://simbad.u-strasbg.fr/simbad/sim-id"
GAIA_URL = "https://gaia.ari.uni-heidelberg.de/singlesource.html"

# Lazy-initialized Simbad client (initialized on first use)
_cSimbad = None


def _get_simbad_client():
    """Get or create the configured Simbad client."""
    global _cSimbad
    if _cSimbad is None:
        _ensure_astroquery()
        _cSimbad = Simbad()
        _cSimbad.add_votable_fields("pmra", "pmdec")
    return _cSimbad


def format_radec(coord: SkyCoord) -> str:
    """Return coordinates as 'HHhMMmSS.SSs DDdMMmSS.Ss'."""
    sra = coord.icrs.ra.to_string(u.hour, decimal=False, sep="hms", precision=2)
    sdec = coord.icrs.dec.to_string(
        u.degree, decimal=False, sep="dms", precision=1, pad=True, alwayssign=True
    )
    return f"{sra}, {sdec}"


def _query_local_catalog(
    catalog_name: str,
    source: SkyCoord,
    radius: u.Quantity,
    name_col: str | None = None,
    name_prefix: str | None = None,
) -> dict[str, u.Quantity]:
    """Helper to query local SQLite catalogs."""
    try:
        from dsa110_contimg.core.calibration.catalogs import query_catalog_sources

        # query_catalog_sources expects degrees
        df = query_catalog_sources(
            catalog_name, source.ra.deg, source.dec.deg, radius.to(u.deg).value
        )

        if df.empty:
            return {}

        out = {}
        for _, row in df.iterrows():
            sc = SkyCoord(row["ra_deg"], row["dec_deg"], unit="deg")
            if name_col and name_col in row:
                name = row[name_col]
            elif name_prefix:
                name = f"{name_prefix} J{format_radec(sc)}"
            else:
                name = f"{catalog_name.upper()} J{format_radec(sc)}"

            sep = sc.separation(source).to(u.arcsec)
            out[name] = sep
        return out
    except Exception as e:
        logger.warning(f"Local {catalog_name} query failed: {e}")
        return {}


def check_gaia(
    source: SkyCoord, t: Time = None, radius: u.Quantity = 15 * u.arcsec
) -> dict[str, u.Quantity]:
    """Check a source against Gaia, correcting for proper motion."""
    _ensure_astroquery()
    if t is None:
        if source.obstime is None:
            logger.error(
                "Must supply either SkyCoord with obstime or separate time for coordinate check"
            )
            return {}
        t = source.obstime

    try:
        q = Gaia.cone_search(coordinate=source, radius=radius)
        r = q.get_results()
    except Exception as e:
        logger.warning(f"Gaia query failed: {e}")
        return {}

    separations = {}
    if len(r) > 0:
        designation = "DESIGNATION" if "DESIGNATION" in r.colnames else "designation"
    else:
        return {}

    for i in range(len(r)):
        try:
            gaia_source = SkyCoord(
                r[i]["ra"] * u.deg,
                r[i]["dec"] * u.deg,
                pm_ra_cosdec=r[i]["pmra"] * u.mas / u.yr,
                pm_dec=r[i]["pmdec"] * u.mas / u.yr,
                distance=(
                    (r[i]["parallax"] * u.mas).to(u.kpc, equivalencies=u.parallax())
                    if r[i]["parallax"] > 0
                    else 1 * u.kpc
                ),
                obstime=Time(r[0]["ref_epoch"], format="decimalyear"),
            )
            sep = gaia_source.apply_space_motion(t).separation(source).arcsec * u.arcsec
            separations[r[i][designation]] = sep
        except Exception as e:
            logger.warning(f"Error processing Gaia source {i}: {e}")
            continue

    return separations


def check_pulsarscraper(
    source: SkyCoord, radius: u.Quantity = 15 * u.arcsec
) -> dict[str, u.Quantity]:
    """Check a source against the Pulsar survey scraper."""
    _ensure_astroquery()  # For requests
    try:
        response = requests.get(
            PULSAR_SCRAPER_URL,
            params={
                "type": "search",
                "ra": source.ra.deg,
                "dec": source.dec.deg,
                "radius": radius.to_value(u.deg),
            },
            timeout=10,
        )
        if not response.ok:
            logger.error(
                f"Unable to query pulsarsurveyscraper: received code={response.status_code} ({response.reason})"
            )
            return {}

        out = {}
        data = response.json()
        for k in data:
            if k.startswith("search") or k.startswith("nmatches"):
                continue
            # The key structure might vary, handle carefully
            try:
                survey = data[k].get("survey", {}).get("value", "unknown")
                dist_val = data[k].get("distance", {}).get("value", 0)
                out[f"{k}[{survey}]"] = (dist_val * u.deg).to(u.arcsec)
            except (KeyError, TypeError):
                continue
        return out
    except Exception as e:
        logger.warning(f"Pulsar scraper query failed: {e}")
        return {}


def check_simbad(
    source: SkyCoord, t: Time = None, radius: u.Quantity = 15 * u.arcsec
) -> dict[str, u.Quantity]:
    """Check a source against Simbad, correcting for proper motion."""
    _ensure_astroquery()
    if t is None:
        if source.obstime is None:
            logger.error(
                "Must supply either SkyCoord with obstime or separate time for coordinate check"
            )
            return {}
        t = source.obstime

    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            r = _get_simbad_client().query_region(source, radius=radius)
    except Exception as e:
        logger.warning(f"Simbad query failed: {e}")
        return {}

    if r is None:
        return {}

    ra_col, dec_col, pmra_col, pmdec_col = "RA", "DEC", "PMRA", "PMDEC"
    if "RA" not in r.colnames:
        ra_col, dec_col, pmra_col, pmdec_col = "ra", "dec", "pmra", "pmdec"

    separations = {}
    for i in range(len(r)):
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                simbad_source = SkyCoord(
                    r[i][ra_col],
                    r[i][dec_col],
                    unit=("hour", "deg"),
                    pm_ra_cosdec=r[i][pmra_col] * u.mas / u.yr,
                    pm_dec=r[i][pmdec_col] * u.mas / u.yr,
                    obstime=Time(2000, format="decimalyear"),
                )
                sep = simbad_source.apply_space_motion(t).separation(source).arcsec * u.arcsec
                separations[r[i]["MAIN_ID"]] = sep
        except (KeyError, ValueError, TypeError):
            # If proper motion is missing or other error, just calculate static separation
            try:
                simbad_source = SkyCoord(r[i][ra_col], r[i][dec_col], unit=("hour", "deg"))
                sep = simbad_source.separation(source).arcsec * u.arcsec
                separations[r[i]["MAIN_ID"]] = sep
            except Exception as e:
                logger.warning(f"Error processing Simbad source {i}: {e}")
                continue

    return separations


def check_atnf(source: SkyCoord, radius: u.Quantity = 15 * u.arcsec) -> dict[str, u.Quantity]:
    """Check a source against ATNF pulsar catalog using local database."""
    # Note: Local ATNF query doesn't currently support proper motion correction
    # as fully as psrqpy might, but it's faster and local.
    return _query_local_catalog("atnf", source, radius, name_col="pulsar_name")


def check_planets(
    source: SkyCoord,
    t: Time = None,
    radius: u.Quantity = 1 * u.arcmin,
) -> dict[str, u.Quantity]:
    """Check a source against solar system planets."""
    if t is None:
        if source.obstime is None:
            logger.error(
                "Must supply either SkyCoord with obstime or separate time for coordinate check"
            )
            return {}
        t = source.obstime

    try:
        # OVRO location approx: 118.283 W, 37.234 N
        loc = EarthLocation.from_geodetic(
            lat=37.234 * u.deg, lon=-118.283 * u.deg, height=1222 * u.m
        )
    except (ValueError, TypeError):
        loc = None  # Fallback

    separations = {}
    try:
        with solar_system_ephemeris.set("builtin"):
            for planet_name in solar_system_ephemeris.bodies:
                if planet_name == "earth":
                    continue
                planet = get_body(planet_name, t, loc)
                if planet.separation(source) < radius:
                    separations[planet_name] = planet.separation(source).to(u.arcsec)
    except Exception as e:
        logger.warning(f"Planet check failed: {e}")

    return separations


def check_first(source: SkyCoord, radius: u.Quantity = 15 * u.arcsec) -> dict[str, u.Quantity]:
    """Check FIRST using local database."""
    return _query_local_catalog("first", source, radius, name_prefix="FIRST")


def check_nvss(source: SkyCoord, radius: u.Quantity = 15 * u.arcsec) -> dict[str, u.Quantity]:
    """Check NVSS using local database."""
    return _query_local_catalog("nvss", source, radius, name_prefix="NVSS")


def check_vlass(source: SkyCoord, radius: u.Quantity = 15 * u.arcsec) -> dict[str, u.Quantity]:
    """Check VLASS using local database."""
    return _query_local_catalog("vlass", source, radius, name_prefix="VLASS")


def check_rax(source: SkyCoord, radius: u.Quantity = 15 * u.arcsec) -> dict[str, u.Quantity]:
    """Check RAX using local database."""
    return _query_local_catalog("rax", source, radius, name_prefix="RAX")


def check_all_services(
    source: SkyCoord, t: Time = None, radius: u.Quantity = 15 * u.arcsec
) -> dict[str, dict[str, u.Quantity]]:
    """Check source against all configured services."""
    services = {
        "Gaia": check_gaia,
        "Simbad": check_simbad,
        "Pulsar Scraper": check_pulsarscraper,
        "ATNF": check_atnf,
        "Planets": check_planets,
        "FIRST": check_first,
        "NVSS": check_nvss,
        "VLASS": check_vlass,
        "RAX": check_rax,
    }

    results = {}
    for name, func in services.items():
        try:
            # Some functions require time 't', others don't or handle it optionally
            # Simbad, Gaia, Planets need 't' if source.obstime is missing
            if func in [check_gaia, check_simbad, check_planets]:
                res = func(source, t=t, radius=radius)
            else:
                res = func(source, radius=radius)

            if res:
                results[name] = res
        except Exception as e:
            logger.warning(f"Service {name} failed: {e}")

    return results
