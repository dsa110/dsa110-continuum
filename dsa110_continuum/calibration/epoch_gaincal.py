"""Per-epoch gain calibration for DSA-110 mosaic pipeline.

Public API
----------
select_calibration_tile_from_ms(epoch_ms_paths) -> str
    Return the MS path (from the two central tiles) with the most catalog sources.

calibrate_epoch(epoch_ms_paths, bp_table, work_dir, ...) -> str | None
    Full 5-step catalog-bootstrap + self-cal gain solve. Returns ap.G table path.
"""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np

from dsa110_continuum.calibration.applycal import apply_to_target
from dsa110_continuum.calibration.model import count_bright_sources_in_tile
from dsa110_continuum.calibration.mosaic_constants import (
    MOSAIC_TILE_COUNT,
    SKYMODEL_MIN_FLUX_MJY,
    SOURCE_QUERY_RADIUS_DEG,
)
from dsa110_continuum.calibration.runner import phaseshift_ms
from dsa110_continuum.calibration.skymodels import (
    make_unified_skymodel,
    predict_from_skymodel_wsclean,
)

log = logging.getLogger(__name__)


_WSCLEAN_FLAG_FRACTION_LIMIT = 0.60  # skip WSClean self-cal if MS is more flagged than this


def _ms_flag_fraction(ms_path: str) -> float:
    """Return the fraction of FLAG=True elements in the MS DATA column."""
    import casacore.tables as ct

    with ct.table(ms_path, readonly=True, ack=False) as t:
        flags = t.getcol("FLAG")
    return float(flags.sum()) / flags.size


def _read_ms_phase_center(ms_path: str) -> tuple[float, float]:
    """Return (ra_deg, dec_deg) of the median field phase center in an MS."""
    import casacore.tables as ct

    with ct.table(f"{ms_path}::FIELD", readonly=True, ack=False) as t:
        phase_dir = t.getcol("PHASE_DIR")  # shape (nfields, 1, 2) radians
    ra_rad = phase_dir[:, 0, 0]
    dec_rad = phase_dir[:, 0, 1]
    # Circular mean for RA to handle 0/360 wrap
    median_ra = float(np.degrees(np.angle(np.mean(np.exp(1j * ra_rad)))) % 360)
    median_dec = float(np.degrees(np.median(dec_rad)))
    return median_ra, median_dec


def select_calibration_tile_from_ms(
    epoch_ms_paths: list[str],
    *,
    min_flux_mjy: float = SKYMODEL_MIN_FLUX_MJY,
    source_radius_deg: float = SOURCE_QUERY_RADIUS_DEG,
) -> str:
    """Return the central tile MS with the most bright catalog sources.

    Checks the two tiles nearest the centre of the sorted list and returns
    the MS path whose pointing has more catalog sources above *min_flux_mjy*
    within *source_radius_deg*.  Optimised for MOSAIC_TILE_COUNT (12) tiles
    but gracefully handles any count >= 2.

    Parameters
    ----------
    epoch_ms_paths:
        Sorted list of >= 2 MS paths for the epoch.
    min_flux_mjy:
        Minimum source flux for the source count query (default: 5 mJy).
    source_radius_deg:
        Catalog search radius around the tile pointing (default: 0.3 deg).

    Returns
    -------
    str
        MS path of the selected calibration tile.

    Raises
    ------
    ValueError
        If epoch_ms_paths contains fewer than 2 entries.
    """
    n = len(epoch_ms_paths)
    if n < 2:
        raise ValueError(f"Need at least 2 MS paths for tile selection, got {n}")

    # Pick the two tiles nearest the centre of the list
    mid = n // 2
    center_indices = [mid - 1, mid]
    best_ms: str | None = None
    best_count = -1

    for idx in center_indices:
        ms = epoch_ms_paths[idx]
        try:
            ra, dec = _read_ms_phase_center(ms)
            n = count_bright_sources_in_tile(
                ra,
                dec,
                min_flux_mjy=min_flux_mjy,
                radius_deg=source_radius_deg,
            )
            log.info("Tile %d (%s): %d catalog sources", idx, Path(ms).stem, n)
            if n > best_count:
                best_count = n
                best_ms = ms
        except Exception as exc:
            log.warning("Cannot count sources for tile %d (%s): %s", idx, ms, exc)

    if best_ms is None:
        # Both catalog queries failed (e.g. VLASS/RACS databases absent).
        # Fall back to the geometrically central tile rather than a hardcoded
        # index that is only correct for MOSAIC_TILE_COUNT=12.
        fallback_idx = len(epoch_ms_paths) // 2
        best_ms = epoch_ms_paths[fallback_idx]
        log.warning(
            "Source count failed for all candidate tiles — "
            "defaulting to central tile index %d (%s)",
            fallback_idx,
            Path(best_ms).stem,
        )

    log.info(
        "Selected calibration tile: %s (%d sources)",
        Path(best_ms).stem,
        best_count,
    )
    return best_ms


def calibrate_epoch(
    epoch_ms_paths: list[str],
    bp_table: str,
    work_dir: str,
    *,
    refant: str = "103,104,105,106,107,10,11,12",
    min_flux_mjy: float = SKYMODEL_MIN_FLUX_MJY,
    source_radius_deg: float = SOURCE_QUERY_RADIUS_DEG,
    wsclean_niter: int = 1000,
    wsclean_threshold_sigma: float = 3.0,
) -> str | None:
    """Derive per-epoch gain solutions using catalog bootstrap + one self-cal round.

    Workflow
    --------
    1. Select central tile (by catalog source count).
    2. Phaseshift to median meridian (reuses existing meridian MS if present).
    3. Apply bandpass-only to CORRECTED_DATA.
    4. Populate MODEL_DATA from unified catalog (FIRST+RACS+NVSS+VLASS).
    5. Phase-only gaincal (solint='inf') → epoch_p.G.
    6. Apply BP + epoch_p.G, then WSClean quick image (-save-model).
    7. Amplitude+phase gaincal (solint='inf') → epoch_ap.G  ← returned.

    Any exception causes an early return of None so callers can fall back to
    the static daily G table.

    Parameters
    ----------
    epoch_ms_paths:
        Sorted list of MOSAIC_TILE_COUNT MS paths (raw, unphaseshifted).
    bp_table:
        Path to the daily bandpass table. Must exist.
    work_dir:
        Scratch directory for intermediate files and output G table.
    refant:
        Reference antenna. CASA uses the first unflagged antenna in a
        comma-separated list, so the default is an outrigger priority chain:
        103 (primary outrigger), then 104–107, then core antennas 10–12 as
        last-resort fallbacks.
    min_flux_mjy:
        Minimum flux for catalog source selection (default: 5 mJy).
    source_radius_deg:
        Catalog search radius (default: 0.3 deg).
    wsclean_niter:
        CLEAN iterations for the self-cal imaging pass (default: 1000).
    wsclean_threshold_sigma:
        Auto-threshold sigma for WSClean (default: 3.0).

    Returns
    -------
    str or None
        Path to the solved ap.G table, or None if any step failed.
    """
    from dsa110_continuum.calibration.casa_service import CASAService

    work = Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)

    try:
        # ── 0. Select central tile ────────────────────────────────────────────
        central_raw_ms = select_calibration_tile_from_ms(
            epoch_ms_paths,
            min_flux_mjy=min_flux_mjy,
            source_radius_deg=source_radius_deg,
        )
        stem = Path(central_raw_ms).stem
        meridian_ms = str(work / f"{stem}_meridian.ms")
        p_table = str(work / f"{stem}.p.G")
        ap_table = str(work / f"{stem}.ap.G")
        wsclean_prefix = str(work / f"{stem}_model")

        # Return cached result if the ap.G table already exists
        if os.path.exists(ap_table):
            log.info("Epoch gaincal [%s]: cached ap.G found — reusing %s", stem, ap_table)
            return ap_table

        # ── 1. Phaseshift to median meridian ──────────────────────────────────
        if not os.path.exists(meridian_ms):
            log.info("Epoch gaincal [%s]: phaseshifting", stem)
            phaseshift_ms(
                ms_path=central_raw_ms,
                mode="median_meridian",
                output_ms=meridian_ms,
            )
        else:
            log.info("Epoch gaincal [%s]: meridian MS exists, reusing", stem)

        # ── 1b. Pre-calibration RFI flagging ─────────────────────────────────
        # Must run on the raw meridian MS before any calibration solve.
        # Unflagged RFI spikes corrupt the least-squares gain solver; the old
        # dsa110-contimg pipeline validated this as critical for drift-scan data
        # where the time axis has only ~24 samples.
        try:
            _svc = CASAService()
            log.info("Epoch gaincal [%s]: flagging autocorrelations", stem)
            _svc.flagdata(vis=meridian_ms, autocorr=True, flagbackup=False)
            try:
                from dsa110_contimg.core.calibration.flagging import flag_rfi as _flag_rfi
                log.info("Epoch gaincal [%s]: AOFlagger RFI flagging", stem)
                _flag_rfi(meridian_ms, backend="aoflagger")
                log.info("Epoch gaincal [%s]: AOFlagger complete", stem)
            except Exception as _aof_err:
                log.warning(
                    "Epoch gaincal [%s]: AOFlagger unavailable (%s) — "
                    "falling back to CASA tfcrop+rflag",
                    stem, _aof_err,
                )
                _svc.flagdata(
                    vis=meridian_ms, mode="tfcrop", datacolumn="data",
                    timecutoff=4.0, freqcutoff=4.0,
                    extendflags=False, flagbackup=False,
                )
                _svc.flagdata(
                    vis=meridian_ms, mode="rflag", datacolumn="data",
                    timedevscale=4.0, freqdevscale=4.0,
                    extendflags=False, flagbackup=False,
                )
                log.info("Epoch gaincal [%s]: CASA tfcrop+rflag complete", stem)
        except Exception as _flag_err:
            log.warning(
                "Epoch gaincal [%s]: pre-calibration flagging failed (%s) — continuing",
                stem, _flag_err,
            )

        # ── 2. Initialise MODEL_DATA column before any applycal ──────────────
        # predict_from_skymodel_wsclean needs MODEL_DATA to exist; if it's absent
        # it attempts clearcal which would destroy CORRECTED_DATA. We add it now
        # while the MS is still "uncalibrated" so the protection guard never fires.
        log.info("Epoch gaincal [%s]: initialising MODEL_DATA column", stem)
        try:
            import casacore.tables as _ct
            with _ct.table(meridian_ms, readonly=True, ack=False) as _t:
                _has_model = "MODEL_DATA" in _t.colnames()
            if not _has_model:
                from dsa110_continuum.calibration.casa_service import CASAService as _CS
                _CS().clearcal(vis=meridian_ms, addmodel=True)
        except Exception as _e:
            log.warning("Epoch gaincal [%s]: MODEL_DATA init failed (%s) — continuing", stem, _e)

        # ── 3. Apply bandpass only → CORRECTED_DATA ───────────────────────────
        log.info("Epoch gaincal [%s]: applying BP table", stem)
        apply_to_target(
            ms_target=meridian_ms,
            field="",
            gaintables=[bp_table],
            interp=["nearest"],
        )

        # ── 5. Catalog MODEL_DATA ─────────────────────────────────────────────
        log.info("Epoch gaincal [%s]: building catalog sky model", stem)
        ra, dec = _read_ms_phase_center(meridian_ms)
        sky = make_unified_skymodel(ra, dec, source_radius_deg, min_mjy=min_flux_mjy)
        if sky.Ncomponents == 0:
            log.error(
                "Epoch gaincal [%s]: catalog sky model is empty — cannot calibrate",
                stem,
            )
            return None
        log.info("Epoch gaincal [%s]: sky model has %d components", stem, sky.Ncomponents)
        predict_from_skymodel_wsclean(meridian_ms, sky)

        # ── 6. Phase-only gaincal ─────────────────────────────────────────────
        log.info("Epoch gaincal [%s]: phase-only gaincal → %s", stem, Path(p_table).name)
        service = CASAService()
        service.gaincal(
            vis=meridian_ms,
            caltable=p_table,
            field="",
            refant=refant,
            calmode="p",
            solint="inf",
            minsnr=3.0,
            gaintype="G",
            gaintable=[bp_table],
            interp=["nearest"],
        )
        if not os.path.exists(p_table):
            log.error("Epoch gaincal [%s]: phase-only solve produced no table", stem)
            return None

        # Apply BP + p.G before WSClean imaging
        apply_to_target(
            ms_target=meridian_ms,
            field="",
            gaintables=[bp_table, p_table],
            interp=["nearest", "linear"],
        )

        # ── 7. Quick WSClean self-cal image to update MODEL_DATA ──────────────
        # Skip WSClean if the MS is too heavily flagged: WSClean crashes during
        # gridding when the uv-plane is under-sampled (UV-starvation). The 60%
        # threshold is conservative; the Feb 15 gaincal MS was 70% flagged.
        _flag_frac = _ms_flag_fraction(meridian_ms)
        log.info(
            "Epoch gaincal [%s]: MS flag fraction before WSClean = %.1f%%",
            stem, 100 * _flag_frac,
        )
        wsclean_exec = shutil.which("wsclean")
        if _flag_frac >= _WSCLEAN_FLAG_FRACTION_LIMIT:
            log.warning(
                "Epoch gaincal [%s]: %.1f%% of data flagged (≥%.0f%% limit) — "
                "skipping WSClean self-cal, re-predicting catalog model for ap solve",
                stem, 100 * _flag_frac, 100 * _WSCLEAN_FLAG_FRACTION_LIMIT,
            )
            predict_from_skymodel_wsclean(meridian_ms, sky)
        elif not wsclean_exec:
            log.warning(
                "Epoch gaincal [%s]: wsclean not on PATH — "
                "re-predicting catalog model for ap solve",
                stem,
            )
            predict_from_skymodel_wsclean(meridian_ms, sky)
        else:
            cmd = [
                wsclean_exec,
                "-niter", str(wsclean_niter),
                "-auto-threshold", str(wsclean_threshold_sigma),
                "-save-model-column", "MODEL_DATA",
                "-name", wsclean_prefix,
                "-size", "1024", "1024",
                "-scale", "6arcsec",
                "-weight", "briggs", "0.5",
                "-no-update-model-required",
                meridian_ms,
            ]
            log.info("Epoch gaincal [%s]: WSClean self-cal imaging", stem)
            wsclean_result = subprocess.run(cmd, capture_output=True, timeout=600)
            if wsclean_result.returncode != 0:
                log.warning(
                    "Epoch gaincal [%s]: WSClean exited %d — "
                    "falling back to catalog MODEL_DATA for ap solve\n%s",
                    stem,
                    wsclean_result.returncode,
                    wsclean_result.stderr.decode("utf-8", errors="replace")[-500:],
                )
                predict_from_skymodel_wsclean(meridian_ms, sky)

        # ── 8. Amplitude+phase gaincal ────────────────────────────────────────
        log.info("Epoch gaincal [%s]: ap gaincal → %s", stem, Path(ap_table).name)
        service.gaincal(
            vis=meridian_ms,
            caltable=ap_table,
            field="",
            refant=refant,
            calmode="ap",
            solint="inf",
            minsnr=3.0,
            gaintype="G",
            gaintable=[bp_table, p_table],
            interp=["nearest", "linear"],
        )
        if not os.path.exists(ap_table):
            log.error("Epoch gaincal [%s]: ap solve produced no table", stem)
            return None

        log.info("Epoch gaincal [%s]: SUCCESS → %s", stem, ap_table)
        return ap_table

    except Exception as exc:
        log.error(
            "Epoch gaincal: FAILED (%s) — caller should fall back to static daily G table",
            exc,
        )
        return None
