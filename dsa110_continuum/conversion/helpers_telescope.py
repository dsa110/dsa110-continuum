"""Telescope utility helper functions for conversion."""

import logging
from contextlib import contextmanager

import astropy.units as u
import numpy as np
from astropy.coordinates import EarthLocation

from dsa110_contimg.common.utils.runtime_safeguards import require_casa6_python

logger = logging.getLogger("dsa110_contimg.conversion.helpers")


@require_casa6_python
def cleanup_casa_file_handles() -> None:
    """Force close any open CASA file handles to prevent locking issues.

    This is critical when running parallel MS operations or using tmpfs staging.
    CASA tools can hold file handles open even after operations complete,
    causing file locking errors in subsequent operations.
    """
    # Ensure CASAPATH is set before importing CASA modules
    from dsa110_contimg.common.utils.casa_init import ensure_casa_path

    ensure_casa_path()

    try:
        import casatools

        tool_names = ["ms", "table", "image", "msmetadata", "simulator"]

        for tool_name in tool_names:
            try:
                tool_factory = getattr(casatools, tool_name, None)
                if tool_factory is not None:
                    tool_instance = tool_factory()
                    if hasattr(tool_instance, "close"):
                        tool_instance.close()
                    if hasattr(tool_instance, "done"):
                        tool_instance.done()
            except (RuntimeError, OSError, AttributeError):
                # Individual tool cleanup failures are non-fatal
                # RuntimeError: CASA internal errors, OSError: file issues,
                # AttributeError: missing methods
                pass

        logger.debug("CASA file handles cleanup completed")
    except ImportError:
        # casatools not available - nothing to clean up
        pass
    except Exception as e:
        logger.debug(f"CASA cleanup failed (non-fatal): {e}")


@contextmanager
def casa_operation():
    """Context manager for CASA operations with automatic cleanup.

        Ensures CASA file handles are cleaned up after operations complete,
        even if exceptions occur. This prevents file locking issues in parallel
        operations and tmpfs staging scenarios.

    Examples
    --------
        with casa_operation():
        # CASA operations here
        ms.open("observation.ms")
        # ... do work ...
        ms.close()
        # cleanup_casa_file_handles() is automatically called here

    Notes
    -----
        This is a best-effort cleanup. Individual tool cleanup failures
        are logged but don't raise exceptions.
    """
    try:
        yield
    finally:
        cleanup_casa_file_handles()


def set_telescope_identity(
    uv,
    name: str | None = None,
    lon_deg: float | None = None,
    lat_deg: float | None = None,
    alt_m: float | None = None,
) -> None:
    """Set a consistent telescope identity and location on a UVData object.

    This writes both name and location metadata in places used by
    pyuvdata and downstream tools:
    - ``uv.telescope_name``
    - ``uv.telescope_location`` (ITRF meters)
    - ``uv.telescope_location_lat_lon_alt`` (radians + meters)
    - ``uv.telescope_location_lat_lon_alt_deg`` (degrees + meters, when present)
    - If a ``uv.telescope`` sub-object exists (pyuvdata>=3), mirror name and
      location fields there as well.

    Parameters
    ----------
    uv : UVData-like
        The in-memory UVData object.
    name : str, optional
        Telescope name. Defaults to ENV PIPELINE_TELESCOPE_NAME or 'DSA_110'.
    lon_deg, lat_deg, alt_m : float, optional
        Observatory geodetic coordinates (WGS84). If not provided, uses DSA110_LOCATION
        from constants.py (single source of truth for DSA-110 coordinates).
    """
    import os as _os

    # Use constants if coordinates not provided (single source of truth)
    if lon_deg is None or lat_deg is None or alt_m is None:
        from dsa110_contimg.common.utils.constants import DSA110_LOCATION

        if lon_deg is None:
            lon_deg = DSA110_LOCATION.lon.to(u.deg).value
        if lat_deg is None:
            lat_deg = DSA110_LOCATION.lat.to(u.deg).value
        if alt_m is None:
            alt_m = DSA110_LOCATION.height.to(u.m).value

    tel_name = name or _os.getenv("PIPELINE_TELESCOPE_NAME", "DSA_110")
    try:
        setattr(uv, "telescope_name", tel_name)
    except (AttributeError, TypeError):
        pass

    try:
        _loc = EarthLocation.from_geodetic(
            lon=lon_deg * u.deg, lat=lat_deg * u.deg, height=alt_m * u.m
        )
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Failed to construct EarthLocation: %s", exc)
        return

    # Populate top-level ITRF (meters)
    try:
        uv.telescope_location = np.array(
            [
                _loc.x.to_value(u.m),
                _loc.y.to_value(u.m),
                _loc.z.to_value(u.m),
            ],
            dtype=float,
        )
    except (AttributeError, TypeError):
        pass

    # Populate geodetic lat/lon/alt in radians/meters if available
    try:
        uv.telescope_location_lat_lon_alt = (
            float(_loc.lat.to_value(u.rad)),
            float(_loc.lon.to_value(u.rad)),
            float(_loc.height.to_value(u.m)),
        )
    except (AttributeError, TypeError):
        pass
    # And in degrees where convenient
    try:
        uv.telescope_location_lat_lon_alt_deg = (
            float(_loc.lat.to_value(u.deg)),
            float(_loc.lon.to_value(u.deg)),
            float(_loc.height.to_value(u.m)),
        )
    except (AttributeError, TypeError):
        pass

    # Mirror onto uv.telescope sub-object when present
    tel = getattr(uv, "telescope", None)
    if tel is not None:
        try:
            setattr(tel, "name", tel_name)
        except (AttributeError, TypeError):
            pass
        try:
            setattr(
                tel,
                "location",
                np.array(
                    [
                        _loc.x.to_value(u.m),
                        _loc.y.to_value(u.m),
                        _loc.z.to_value(u.m),
                    ],
                    dtype=float,
                ),
            )
        except (AttributeError, TypeError):
            pass
        try:
            setattr(
                tel,
                "location_lat_lon_alt",
                (
                    float(_loc.lat.to_value(u.rad)),
                    float(_loc.lon.to_value(u.rad)),
                    float(_loc.height.to_value(u.m)),
                ),
            )
        except (AttributeError, TypeError):
            pass
        try:
            setattr(
                tel,
                "location_lat_lon_alt_deg",
                (
                    float(_loc.lat.to_value(u.deg)),
                    float(_loc.lon.to_value(u.deg)),
                    float(_loc.height.to_value(u.m)),
                ),
            )
        except (AttributeError, TypeError):
            pass

    logger.debug(
        "Set telescope identity: %s @ (lon,lat,alt)=(%.4f, %.4f, %.1f)",
        tel_name,
        lon_deg,
        lat_deg,
        alt_m,
    )


def set_ms_telescope_name(ms_path: str, name: str = "DSA_110") -> None:
    """Set TELESCOPE_NAME in a Measurement Set's OBSERVATION table.

    This is used to restore DSA_110 before running WSClean/EveryBeam,
    which requires the correct telescope name for beam model selection.

    The merge_spws step sets TELESCOPE_NAME to OVRO_MMA for CASA compatibility,
    but EveryBeam needs DSA_110 to use the native DSA-110 beam model.

    Parameters
    ----------
    ms_path : str
        Path to Measurement Set
    name : str, optional
        Telescope name to set. Defaults to 'DSA_110' for EveryBeam compatibility.
    """
    import os

    try:
        import casacore.tables as ct
    except ImportError:
        logger.warning("casacore.tables not available, cannot set telescope name")
        return

    obs_table = os.path.join(ms_path, "OBSERVATION")
    if not os.path.exists(obs_table):
        logger.warning("OBSERVATION table not found in %s", ms_path)
        return

    try:
        with ct.table(obs_table, readonly=False) as tb:
            current = tb.getcol("TELESCOPE_NAME")
            if current and current[0] != name:
                tb.putcol("TELESCOPE_NAME", [name] * len(current))
                logger.info(
                    "Updated TELESCOPE_NAME: %s -> %s (for EveryBeam beam model)",
                    current[0],
                    name,
                )
    except Exception as e:
        logger.warning("Failed to update TELESCOPE_NAME in %s: %s", ms_path, e)
