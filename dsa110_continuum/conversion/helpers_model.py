"""Model helper functions for conversion."""

import logging

import astropy.units as u

# Ensure CASAPATH is set before importing CASA modules
from dsa110_contimg.common.utils.casa_init import ensure_casa_path

ensure_casa_path()

import casacore.tables as casatables  # type: ignore
import numpy as np

from dsa110_contimg.core.calibration import BeamConfig, primary_beam_response

table = casatables.table  # noqa: N816

logger = logging.getLogger("dsa110_contimg.conversion.helpers")


def amplitude_sky_model(
    source_ra: u.Quantity,
    source_dec: u.Quantity,
    flux_jy: float,
    lst: np.ndarray,
    pt_dec: u.Quantity,
    freq_ghz: np.ndarray,
    dish_diameter_m: float = 4.7,
) -> np.ndarray:
    """Construct a primary-beam weighted amplitude model using EveryBeam.

    Uses the robust EveryBeam beam model (with Airy fallback) for accurate
    primary beam corrections during model construction.
    """
    # Convert coordinates to radians
    src_ra = source_ra.to_value(u.rad)
    src_dec = source_dec.to_value(u.rad)
    ant_dec = pt_dec.to_value(u.rad)

    # Create beam configuration with EveryBeam preference
    config = BeamConfig(
        frequency_ghz=freq_ghz,
        antenna_ra=lst,  # LST array for antenna RA
        antenna_dec=ant_dec,
        dish_diameter_m=dish_diameter_m,
        use_docker=True,  # Prefer Docker EveryBeam for accuracy
    )

    # Calculate primary beam response
    pb = primary_beam_response(
        src_ra=src_ra,
        src_dec=src_dec,
        config=config,
    )
    return (flux_jy * pb).astype(np.float32)


def set_model_column(
    msname: str,
    uvdata,
    pt_dec: u.Quantity,
    ra: u.Quantity,
    dec: u.Quantity,
    flux_jy: float | None = None,
) -> None:
    """Populate MODEL_DATA (and related columns) for the produced MS."""
    logger.info("Setting MODEL_DATA column")
    if flux_jy is not None:
        fobs = uvdata.freq_array.squeeze() / 1e9
        lst = uvdata.lst_array
        model = amplitude_sky_model(ra, dec, flux_jy, lst, pt_dec, fobs)
        model = np.tile(model[:, :, np.newaxis], (1, 1, uvdata.Npols)).astype(np.complex64)
    else:
        model = np.ones((uvdata.Nblts, uvdata.Nfreqs, uvdata.Npols), dtype=np.complex64)

    ms_path = f"{msname}.ms"
    with table(ms_path, readonly=False) as tb:
        data_shape = tb.getcol("DATA").shape
        model_transposed = np.transpose(model, (2, 1, 0))

        if model_transposed.shape != data_shape:
            logger.warning(
                "Model shape %s does not match DATA shape %s; skipping MODEL_DATA write",
                model_transposed.shape,
                data_shape,
            )
        else:
            tb.putcol("MODEL_DATA", model_transposed)

        if "CORRECTED_DATA" in tb.colnames():
            try:
                corr = tb.getcol("CORRECTED_DATA")
                if not np.any(corr):
                    tb.putcol("CORRECTED_DATA", tb.getcol("DATA"))
            except (RuntimeError, KeyError):  # pragma: no cover - best effort
                # RuntimeError: CASA table errors, KeyError: missing column
                pass

        if "WEIGHT_SPECTRUM" in tb.colnames():
            flags = tb.getcol("FLAG")
            weights = tb.getcol("WEIGHT")
            ncorr = weights.shape[0]
            nchan = flags.shape[0]

            wspec = np.repeat(weights[np.newaxis, :, :], nchan, axis=0)
            if wspec.shape != (nchan, ncorr, weights.shape[1]):
                logger.debug(
                    "Skipping WEIGHT_SPECTRUM update due to unexpected shape: %s",
                    wspec.shape,
                )
            else:
                wspec[flags] = 0.0
                tb.putcol("WEIGHT_SPECTRUM", wspec.astype(np.float32))
                logger.info("Reconstructed WEIGHT_SPECTRUM column.")

    logger.info("MODEL_DATA column set successfully")
