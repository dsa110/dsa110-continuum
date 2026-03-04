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

    Checks tiles at indices 5 and 6 (0-indexed) of the sorted 12-tile epoch
    list and returns the MS path whose pointing has more catalog sources above
    *min_flux_mjy* within *source_radius_deg*.

    Parameters
    ----------
    epoch_ms_paths:
        Sorted list of exactly MOSAIC_TILE_COUNT (12) MS paths for the epoch.
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
        If epoch_ms_paths does not contain exactly MOSAIC_TILE_COUNT entries.
    """
    if len(epoch_ms_paths) != MOSAIC_TILE_COUNT:
        raise ValueError(
            f"Expected {MOSAIC_TILE_COUNT} MS paths, got {len(epoch_ms_paths)}"
        )

    center_indices = [5, 6]
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
        log.warning("Source count failed for both central tiles; defaulting to tile 5")
        best_ms = epoch_ms_paths[5]

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
    refant: str = "103",
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
        Reference antenna (default: "103").
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

        # ── 2. Apply bandpass only → CORRECTED_DATA ───────────────────────────
        log.info("Epoch gaincal [%s]: applying BP table", stem)
        apply_to_target(
            ms_target=meridian_ms,
            field="",
            gaintables=[bp_table],
            interp=["nearest"],
        )

        # ── 3. Catalog MODEL_DATA ─────────────────────────────────────────────
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

        # ── 4. Phase-only gaincal ─────────────────────────────────────────────
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

        # ── 5. Quick WSClean self-cal image to update MODEL_DATA ──────────────
        wsclean_exec = shutil.which("wsclean")
        if not wsclean_exec:
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

        # ── 6. Amplitude+phase gaincal ────────────────────────────────────────
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
