"""Antenna position helper functions for conversion."""

import logging

import numpy as np

from dsa110_contimg.common.utils.antpos_local import get_itrf

logger = logging.getLogger("dsa110_contimg.conversion.helpers")


def _get_relative_antenna_positions(uv) -> np.ndarray:
    """Return antenna positions relative to the telescope location."""
    if hasattr(uv, "antenna_positions") and uv.antenna_positions is not None:
        return uv.antenna_positions
    telescope = getattr(uv, "telescope", None)
    if telescope is not None and getattr(telescope, "antenna_positions", None) is not None:
        return telescope.antenna_positions
    raise AttributeError("UVData object has no antenna_positions information")


def _set_relative_antenna_positions(uv, rel_positions: np.ndarray) -> None:
    """Write relative antenna positions back to the UVData structure."""
    if hasattr(uv, "antenna_positions") and uv.antenna_positions is not None:
        uv.antenna_positions[: rel_positions.shape[0]] = rel_positions
    elif hasattr(uv, "antenna_positions"):
        uv.antenna_positions = rel_positions
    else:
        setattr(uv, "antenna_positions", rel_positions)

    telescope = getattr(uv, "telescope", None)
    if telescope is not None:
        if getattr(telescope, "antenna_positions", None) is not None:
            telescope.antenna_positions[: rel_positions.shape[0]] = rel_positions
        elif hasattr(telescope, "antenna_positions"):
            telescope.antenna_positions = rel_positions
        else:
            setattr(telescope, "antenna_positions", rel_positions)


def set_antenna_positions(uvdata) -> np.ndarray:
    """Populate antenna positions for the Measurement Set."""
    logger.info("Setting DSA-110 antenna positions")
    try:
        df_itrf = get_itrf(latlon_center=None)
    except Exception as exc:  # pragma: no cover - defensive
        logger.error("Failed to load antenna coordinates: %s", exc)
        raise

    abs_positions = np.array(
        [
            df_itrf["x_m"],
            df_itrf["y_m"],
            df_itrf["z_m"],
        ]
    ).T.astype(np.float64)

    telescope_location = getattr(uvdata, "telescope_location", None)
    if telescope_location is None and getattr(uvdata, "telescope", None) is not None:
        telescope_location = getattr(uvdata.telescope, "location", None)
    if telescope_location is None:
        raise AttributeError("UVData object lacks telescope location information")
    if hasattr(telescope_location, "value"):
        telescope_location = telescope_location.value
    telescope_location = np.asarray(telescope_location)
    if getattr(telescope_location, "dtype", None) is not None and telescope_location.dtype.names:
        telescope_location = np.array(
            [telescope_location["x"], telescope_location["y"], telescope_location["z"]]
        )

    rel_positions_target: np.ndarray | None = None
    try:
        rel_positions_target = _get_relative_antenna_positions(uvdata)
    except AttributeError:
        pass

    if rel_positions_target is not None and rel_positions_target.shape[0] != abs_positions.shape[0]:
        raise ValueError(
            f"Mismatch between antenna counts ({rel_positions_target.shape[0]!r} vs "
            f"{abs_positions.shape[0]!r}) when loading antenna catalogue"
        )

    relative_positions = abs_positions - telescope_location
    _set_relative_antenna_positions(uvdata, relative_positions)

    logger.info("Loaded dynamic antenna positions for %s antennas", abs_positions.shape[0])

    # Ensure antenna mount metadata is populated (alt-az for DSA-110)
    # pyuvdata's write_ms_antenna reads telescope.mount_type (a list of mount types per antenna)
    # Valid values: "alt-az", "equatorial", "x-y", "orbiting", "phased", "fixed", "other"
    # CASA requires lowercase "alt-az" to compute parallactic angles correctly
    nants = abs_positions.shape[0]
    mounts = ["alt-az"] * nants  # pyuvdata expects a list, not numpy array
    telescope = getattr(uvdata, "telescope", None)
    if telescope is not None and hasattr(telescope, "mount_type"):
        telescope.mount_type = mounts
    return abs_positions


def _ensure_antenna_diameters(uvdata, diameter_m: float = 4.65) -> None:
    """Ensure antenna diameter metadata is populated."""
    nants: int | None = None
    if (
        hasattr(uvdata, "telescope")
        and getattr(uvdata.telescope, "antenna_numbers", None) is not None
    ):
        nants = len(uvdata.telescope.antenna_numbers)
    elif getattr(uvdata, "antenna_numbers", None) is not None:
        nants = len(np.unique(uvdata.antenna_numbers))

    if nants is None:
        raise AttributeError("Unable to determine antenna count to assign diameters")

    diam_array = np.full(nants, diameter_m, dtype=np.float64)

    telescope = getattr(uvdata, "telescope", None)
    if telescope is not None and hasattr(telescope, "antenna_diameters"):
        telescope.antenna_diameters = diam_array
    if hasattr(uvdata, "antenna_diameters"):
        uvdata.antenna_diameters = diam_array
