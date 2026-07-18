"""Per-epoch gain calibration for DSA-110 mosaic pipeline.

Public API
----------
select_calibration_tile_from_ms(epoch_ms_paths) -> str
    Return the epoch MS with the strongest catalog calibration target.

calibrate_epoch(epoch_ms_paths, bp_table, work_dir, ...) -> EpochGaincalResult
    Full 5-step catalog-bootstrap + self-cal gain solve. Returns a structured
    result carrying the ap.G table path (or None) plus the status enum and a
    human-readable reason. The result distinguishes "low SNR" (operational
    limit) from "exception / no table" (code-path / data fault) so downstream
    manifests and promotion records can classify the outcome honestly per
    docs/validation/pipeline-validation-from-scratch.md.
"""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from dsa110_continuum.calibration.applycal import apply_to_target
from dsa110_continuum.calibration.field_directions import (
    extract_field_ra_dec as _extract_field_ra_dec,
)
from dsa110_continuum.calibration.model import count_bright_sources_in_tile
from dsa110_continuum.calibration.mosaic_constants import (
    SKYMODEL_MIN_FLUX_MJY,
    SOURCE_QUERY_RADIUS_DEG,
)
from dsa110_continuum.calibration.runner import phaseshift_ms
from dsa110_continuum.calibration.skymodels import (
    make_unified_skymodel,
    predict_from_skymodel_wsclean,
)

log = logging.getLogger(__name__)

if TYPE_CHECKING:
    from dsa110_continuum.calibration.flagging import RfiMode


class EpochGaincalStatus(str, Enum):
    """Spec-aligned status for one calibrate_epoch invocation.

    LOW_SNR is the operational case where the data could not support a
    reliable gain solution — empty sky model, p.G flag fraction over the
    GAINCAL_FLAG_FRACTION_LIMIT, or solver wrote no table because every
    solution was flagged. SOLVER_NO_TABLE is reserved for the case where
    CASA reported success but the table file is absent (rare; usually a
    code/data fault). EXCEPTION captures any uncaught Python exception in
    the calibrate_epoch try/except — the legacy "code-path fallback".

    Mapping to the spec's epoch_gaincal_state enum (see
    dsa110_continuum.qa.promotion.derive_epoch_gaincal_state_from_status):
      SOLVED          -> "solved"
      LOW_SNR         -> "skipped_or_failed_low_snr"
      SOLVER_NO_TABLE -> "skipped_or_failed_low_snr"  (no table = all flagged)
      EXCEPTION       -> "fell_back_to_static_with_reason"
    """

    SOLVED = "solved"
    LOW_SNR = "low_snr"
    SOLVER_NO_TABLE = "solver_no_table"
    EXCEPTION = "exception"


@dataclass(frozen=True)
class EpochGaincalResult:
    """Structured outcome of a calibrate_epoch invocation.

    g_table is the path to the solved ap.G table when status == SOLVED;
    otherwise None and the caller should fall back to the static daily G.
    reason is a short human-readable string suitable for the manifest gate's
    reason field (e.g. "p.G flagged 44.4% of solutions (limit 30%)").
    """

    g_table: str | None
    status: EpochGaincalStatus
    reason: str | None = None


_WSCLEAN_FLAG_FRACTION_LIMIT = 0.60  # skip WSClean self-cal if MS is more flagged than this
GAINCAL_FLAG_FRACTION_LIMIT = 0.30  # abort epoch gaincal if p.G table is more flagged than this


def _ms_flag_fraction(ms_path: str) -> float:
    """Return the fraction of FLAG=True elements in the MS DATA column."""
    from dsa110_continuum.adapters import casa_tables as ct

    with ct.table(ms_path, readonly=True, ack=False) as t:
        flags = t.getcol("FLAG")
    return float(flags.sum()) / flags.size


def _gain_flag_fractions(candidate_table: str, bp_table: str) -> dict[str, object]:
    """Return raw and BP-relative flag fractions for a candidate gain table."""
    from dsa110_continuum.adapters import casa_tables as ct
    from dsa110_continuum.calibration.solve_bandpass import (
        _flag_fraction_excluding_dead_receptors,
    )

    with ct.table(candidate_table, readonly=True, ack=False) as table:
        candidate_flags = table.getcol("FLAG")
        candidate_antennas = table.getcol("ANTENNA1")
    if np.asarray(candidate_flags).size == 0:
        raise ValueError(f"Candidate gain table has no FLAG values: {candidate_table}")

    raw_fraction = float(np.mean(candidate_flags))
    try:
        with ct.table(bp_table, readonly=True, ack=False) as table:
            bp_flags = table.getcol("FLAG")
            bp_antennas = table.getcol("ANTENNA1")
        if np.asarray(bp_flags).size == 0:
            raise ValueError(f"Bandpass table has no FLAG values: {bp_table}")
        baseline = _flag_fraction_excluding_dead_receptors(bp_flags, bp_antennas)
        effective = _flag_fraction_excluding_dead_receptors(
            candidate_flags,
            candidate_antennas,
            excluded_receptors=set(baseline["dead_receptors"]),
        )
    except Exception as exc:
        log.warning(
            "Could not validate BP receptor baseline for %s (%s); using raw flag fraction",
            candidate_table,
            exc,
        )
        return {
            "raw_fraction": raw_fraction,
            "effective_fraction": raw_fraction,
            "baseline_valid": False,
            "baseline_dead_receptors": 0,
        }

    return {
        "raw_fraction": raw_fraction,
        "effective_fraction": effective["effective_flag_fraction"],
        "baseline_valid": True,
        "baseline_dead_receptors": baseline["dead_receptor_count"],
        "working_flagged": effective["working_flagged"],
        "working_total": effective["working_total"],
    }


def _modeled_field_count(ms_path: str) -> tuple[int, int]:
    """Return modeled and total field counts using bounded MODEL_DATA samples."""
    from dsa110_continuum.adapters import casa_tables as ct

    with ct.table(ms_path, readonly=True, ack=False) as table:
        field_ids = np.asarray(table.getcol("FIELD_ID"))
        unique_fields = np.unique(field_ids)
        modeled = 0
        for field_id in unique_fields:
            field_rows = np.flatnonzero(field_ids == field_id)
            rows = field_rows[np.linspace(0, len(field_rows) - 1, min(8, len(field_rows))).astype(int)]
            if any(np.any(np.abs(table.getcell("MODEL_DATA", int(row))) > 0) for row in rows):
                modeled += 1
    return modeled, len(unique_fields)


def _format_gain_flag_stats(stats: dict[str, object]) -> str:
    baseline = "BP-relative" if stats["baseline_valid"] else "raw fallback"
    return (
        f"raw {float(stats['raw_fraction']) * 100:.1f}%, "
        f"{baseline} {float(stats['effective_fraction']) * 100:.1f}%"
    )


def _read_ms_phase_center(ms_path: str) -> tuple[float, float]:
    """Return (ra_deg, dec_deg) of the median field phase center in an MS."""
    from dsa110_continuum.adapters import casa_tables as ct

    with ct.table(f"{ms_path}::FIELD", readonly=True, ack=False) as t:
        phase_dir = t.getcol("PHASE_DIR")
    # Shape-tolerant: PHASE_DIR is (nfields, 1, 2) on rows-first table backends
    # and (nfields, 2, 1) when CASA returns column-major. _extract_field_ra_dec
    # handles both; raw [:, 0, 1] indexing on the column-major shape raises
    # IndexError on axis-2 size 1 (the original epoch_gaincal failure mode).
    ra_rad, dec_rad = _extract_field_ra_dec(phase_dir)
    # Circular mean for RA to handle 0/360 wrap
    median_ra = float(np.degrees(np.angle(np.mean(np.exp(1j * ra_rad)))) % 360)
    median_dec = float(np.degrees(np.median(dec_rad)))
    return median_ra, median_dec


def _find_vla_calibrator_in_ms(
    ms_path: str,
    *,
    search_radius_deg: float,
) -> tuple[str, float, float]:
    """Return ``(name, flux_jy, separation_deg)`` for the best VLA calibrator."""
    from dsa110_continuum.calibration.selection import select_bandpass_from_catalog

    _, _, _, cal_info, _ = select_bandpass_from_catalog(
        ms_path,
        search_radius_deg=search_radius_deg,
    )
    name, cal_ra_deg, cal_dec_deg, flux_jy = cal_info
    tile_ra_deg, tile_dec_deg = _read_ms_phase_center(ms_path)
    tile_ra = np.radians(tile_ra_deg)
    tile_dec = np.radians(tile_dec_deg)
    cal_ra = np.radians(cal_ra_deg)
    cal_dec = np.radians(cal_dec_deg)
    cos_sep = (
        np.sin(tile_dec) * np.sin(cal_dec)
        + np.cos(tile_dec) * np.cos(cal_dec) * np.cos(tile_ra - cal_ra)
    )
    separation_deg = float(np.degrees(np.arccos(np.clip(cos_sep, -1.0, 1.0))))
    return str(name), float(flux_jy), separation_deg


def select_calibration_tile_from_ms(
    epoch_ms_paths: list[str],
    *,
    min_flux_mjy: float = SKYMODEL_MIN_FLUX_MJY,
    source_radius_deg: float = SOURCE_QUERY_RADIUS_DEG,
) -> str:
    """Return the epoch MS with the strongest catalog calibration target.

    All tiles are checked for VLA calibrators first. The highest-flux match
    wins, with angular proximity to the tile midpoint as the tiebreaker. If
    the VLA catalog is unavailable or no calibrator is present, all tiles are
    ranked by their bright-source counts.

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

    calibrator_search_radius_deg = max(1.0, source_radius_deg)
    best_calibrator_ms: str | None = None
    best_calibrator_score: tuple[float, float] | None = None

    for idx, ms in enumerate(epoch_ms_paths):
        try:
            name, flux_jy, separation_deg = _find_vla_calibrator_in_ms(
                ms,
                search_radius_deg=calibrator_search_radius_deg,
            )
        except (
            FileNotFoundError,
            KeyError,
            OSError,
            RuntimeError,
            TypeError,
            ValueError,
        ) as exc:
            log.debug("No VLA calibrator match for tile %d (%s): %s", idx, ms, exc)
            continue

        score = (flux_jy, -separation_deg)
        log.info(
            "Tile %d (%s): VLA calibrator %s, %.2f Jy, %.3f deg from midpoint",
            idx,
            Path(ms).stem,
            name,
            flux_jy,
            separation_deg,
        )
        if best_calibrator_score is None or score > best_calibrator_score:
            best_calibrator_score = score
            best_calibrator_ms = ms

    if best_calibrator_ms is not None:
        log.info(
            "Selected calibration tile %s from VLA calibrator ranking",
            Path(best_calibrator_ms).stem,
        )
        return best_calibrator_ms

    best_ms: str | None = None
    best_count = -1

    for idx, ms in enumerate(epoch_ms_paths):
        try:
            ra, dec = _read_ms_phase_center(ms)
            source_count = count_bright_sources_in_tile(
                ra,
                dec,
                min_flux_mjy=min_flux_mjy,
                radius_deg=source_radius_deg,
            )
            log.info("Tile %d (%s): %d catalog sources", idx, Path(ms).stem, source_count)
            if source_count > best_count:
                best_count = source_count
                best_ms = ms
        except Exception as exc:
            log.warning("Cannot count sources for tile %d (%s): %s", idx, ms, exc)

    if best_ms is None:
        # All catalog queries failed (e.g. VLASS/NVSS databases absent).
        # Fall back to the geometrically central tile rather than a hardcoded
        # index that is only correct for MOSAIC_TILE_COUNT=12.
        fallback_idx = len(epoch_ms_paths) // 2
        best_ms = epoch_ms_paths[fallback_idx]
        log.warning(
            "Source count failed for all epoch tiles — "
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
    rfi_mode: RfiMode = "conditional",
) -> EpochGaincalResult:
    """Derive per-epoch gain solutions using catalog bootstrap + one self-cal round.

    Workflow
    --------
    1.  Select the strongest calibration tile from the full epoch.
    2.  Phaseshift to median meridian (reuses existing meridian MS if present).
    1b. Pre-calibration RFI flagging (autocorr + AOFlagger/tfcrop+rflag).
    3.  Apply bandpass-only to CORRECTED_DATA.
    4.  Populate MODEL_DATA from unified catalog (FIRST+RACS+NVSS+VLASS).
    6.  Direct phase-only gaincal (solint='inf', gaintable=[bp]) → direct.p.G.
    6b. Gate on newly failed receptors relative to the independently measured
        BP receptor mask. If direct solving fails and multiple fields contain
        MODEL_DATA, try the 60s pre-conditioner once as a rescue.
    7.  Apply the selected chain, then WSClean quick image (-save-model).
    8.  BP-referenced amplitude+phase gaincal → self-contained ap.G.

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
    rfi_mode:
        Shared pre-calibration RFI policy (default: ``conditional``).

    Returns
    -------
    EpochGaincalResult
        ``g_table`` is the ap.G table path when ``status == SOLVED``, else
        ``None`` and the caller should fall back to the static daily G.
        ``reason`` is a short human-readable string suitable for the manifest
        gate's reason field. Status distinguishes operational SNR-floor
        failures from code-path exceptions per the validation spec at
        ``docs/validation/pipeline-validation-from-scratch.md``.
    """
    from dsa110_continuum.calibration.casa_service import CASAService

    work = Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)

    try:
        # ── 0. Select calibration tile ────────────────────────────────────────
        central_raw_ms = select_calibration_tile_from_ms(
            epoch_ms_paths,
            min_flux_mjy=min_flux_mjy,
            source_radius_deg=source_radius_deg,
        )
        stem = Path(central_raw_ms).stem
        meridian_ms   = str(work / f"{stem}_meridian.ms")
        precond_table = str(work / f"{stem}.direct_first.precond.G")
        direct_p_table = str(work / f"{stem}.direct.p.G")
        rescue_p_table = str(work / f"{stem}.precond_rescue.p.G")
        ap_table = str(work / f"{stem}.direct_first.ap.G")
        wsclean_prefix = str(work / f"{stem}_model")

        # Return cached result if the ap.G table already exists
        if os.path.exists(ap_table):
            try:
                cached_stats = _gain_flag_fractions(ap_table, bp_table)
                if float(cached_stats["effective_fraction"]) <= GAINCAL_FLAG_FRACTION_LIMIT:
                    log.info(
                        "Epoch gaincal [%s]: validated cached ap.G — reusing %s",
                        stem,
                        ap_table,
                    )
                    return EpochGaincalResult(
                        ap_table, EpochGaincalStatus.SOLVED, "validated cached ap.G reused"
                    )
                log.warning(
                    "Epoch gaincal [%s]: cached ap.G fails gate (%s) — recomputing",
                    stem,
                    _format_gain_flag_stats(cached_stats),
                )
            except Exception as exc:
                log.warning(
                    "Epoch gaincal [%s]: cached ap.G is unreadable (%s) — recomputing",
                    stem,
                    exc,
                )
            if os.path.isdir(ap_table):
                shutil.rmtree(ap_table)
            else:
                os.remove(ap_table)

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
            from dsa110_continuum.calibration.flagging import execute_rfi_policy

            execute_rfi_policy(meridian_ms, rfi_mode, f"epoch gaincal {stem}")
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
            from dsa110_continuum.adapters import casa_tables as _ct
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
            return EpochGaincalResult(
                None,
                EpochGaincalStatus.LOW_SNR,
                "catalog sky model is empty (no bright sources within search radius)",
            )
        log.info("Epoch gaincal [%s]: sky model has %d components", stem, sky.Ncomponents)
        predict_from_skymodel_wsclean(meridian_ms, sky, field="all")
        modeled_fields, total_fields = _modeled_field_count(meridian_ms)
        if modeled_fields != total_fields:
            raise RuntimeError(
                f"MODEL_DATA populated for {modeled_fields}/{total_fields} fields; "
                "refusing partial epoch gain calibration"
            )

        # ── 6. Direct phase-only gaincal ──────────────────────────────────────
        service = CASAService()
        p_table = direct_p_table
        _precond: list[str] = []
        _precond_interp: list[str] = []
        _precond_spwmap: list[list[int]] = []
        log.info("Epoch gaincal [%s]: direct phase-only gaincal → %s", stem, Path(p_table).name)
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
        # ── 6b. Independent receptor gate and bounded rescue ──────────────────
        if os.path.exists(p_table):
            direct_stats = _gain_flag_fractions(p_table, bp_table)
        else:
            log.warning("Epoch gaincal [%s]: direct phase-only solve produced no table", stem)
            direct_stats = {
                "raw_fraction": 1.0,
                "effective_fraction": 1.0,
                "baseline_valid": False,
                "baseline_dead_receptors": 0,
            }
        log.info("Epoch gaincal [%s]: direct p.G %s", stem, _format_gain_flag_stats(direct_stats))
        if float(direct_stats["effective_fraction"]) > GAINCAL_FLAG_FRACTION_LIMIT:
            if modeled_fields < 2:
                reason = (
                    f"direct p.G {_format_gain_flag_stats(direct_stats)} exceeds "
                    f"{GAINCAL_FLAG_FRACTION_LIMIT * 100:.0f}% limit; MODEL_DATA present in "
                    f"{modeled_fields}/{total_fields} fields, so 60s rescue cannot span fields"
                )
                return EpochGaincalResult(None, EpochGaincalStatus.LOW_SNR, reason)

            log.info(
                "Epoch gaincal [%s]: direct gate failed; trying 60s pre-conditioner rescue",
                stem,
            )
            try:
                service.gaincal(
                    vis=meridian_ms,
                    caltable=precond_table,
                    field="",
                    refant=refant,
                    calmode="p",
                    solint="60s",
                    combine="spw",
                    minsnr=3.0,
                    gaintype="G",
                    gaintable=[bp_table],
                    interp=["nearest"],
                )
            except Exception as exc:
                reason = (
                    f"direct p.G {_format_gain_flag_stats(direct_stats)} exceeds "
                    f"{GAINCAL_FLAG_FRACTION_LIMIT * 100:.0f}% limit; rescue failed: {exc}"
                )
                return EpochGaincalResult(None, EpochGaincalStatus.LOW_SNR, reason)
            if not os.path.exists(precond_table):
                reason = (
                    f"direct p.G {_format_gain_flag_stats(direct_stats)} exceeds "
                    f"{GAINCAL_FLAG_FRACTION_LIMIT * 100:.0f}% limit; rescue produced no table"
                )
                return EpochGaincalResult(None, EpochGaincalStatus.LOW_SNR, reason)

            from dsa110_continuum.adapters import casa_tables as _ct2

            with _ct2.table(
                f"{meridian_ms}::SPECTRAL_WINDOW", readonly=True, ack=False
            ) as _tspw:
                _n_spw = _tspw.nrows()
            _precond = [precond_table]
            _precond_interp = ["linear"]
            _precond_spwmap = [[0] * _n_spw]
            service.gaincal(
                vis=meridian_ms,
                caltable=rescue_p_table,
                field="",
                refant=refant,
                calmode="p",
                solint="inf",
                minsnr=3.0,
                gaintype="G",
                gaintable=[bp_table, precond_table],
                interp=["nearest", "linear"],
                spwmap=[[], *_precond_spwmap],
            )
            if not os.path.exists(rescue_p_table):
                return EpochGaincalResult(
                    None,
                    EpochGaincalStatus.LOW_SNR,
                    "pre-conditioner rescue phase solve produced no table",
                )
            rescue_stats = _gain_flag_fractions(rescue_p_table, bp_table)
            log.info(
                "Epoch gaincal [%s]: rescue p.G %s",
                stem,
                _format_gain_flag_stats(rescue_stats),
            )
            rescue_fraction = float(rescue_stats["effective_fraction"])
            if (
                rescue_fraction > GAINCAL_FLAG_FRACTION_LIMIT
                or rescue_fraction >= float(direct_stats["effective_fraction"])
            ):
                reason = (
                    f"rescue p.G {_format_gain_flag_stats(rescue_stats)} did not strictly improve "
                    f"and pass the {GAINCAL_FLAG_FRACTION_LIMIT * 100:.0f}% limit"
                )
                return EpochGaincalResult(None, EpochGaincalStatus.LOW_SNR, reason)
            p_table = rescue_p_table

        # Apply BP + precond (if present) + p.G before WSClean imaging
        apply_to_target(
            ms_target=meridian_ms,
            field="",
            gaintables=[bp_table, *_precond, p_table],
            interp=["nearest", *_precond_interp, "linear"],
            spwmap=([[], *_precond_spwmap, []] if _precond_spwmap else None),
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
            predict_from_skymodel_wsclean(meridian_ms, sky, field="all")
        elif not wsclean_exec:
            log.warning(
                "Epoch gaincal [%s]: wsclean not on PATH — "
                "re-predicting catalog model for ap solve",
                stem,
            )
            predict_from_skymodel_wsclean(meridian_ms, sky, field="all")
        else:
            cmd = [
                wsclean_exec,
                "-reorder",
                "-niter", str(wsclean_niter),
                "-auto-threshold", str(wsclean_threshold_sigma),
                "-model-column", "MODEL_DATA",
                "-update-model-required",
                "-field", "all",
                "-name", wsclean_prefix,
                "-size", "1024", "1024",
                "-scale", "6arcsec",
                "-weight", "briggs", "0.5",
                "-mgain", "0.8",
                meridian_ms,
            ]
            log.info("Epoch gaincal [%s]: WSClean self-cal imaging", stem)
            wsclean_result = subprocess.run(cmd, capture_output=True, timeout=600)
            if wsclean_result.returncode != 0:
                diagnostic = (wsclean_result.stdout or b"") + (wsclean_result.stderr or b"")
                log.warning(
                    "Epoch gaincal [%s]: WSClean exited %d — "
                    "falling back to catalog MODEL_DATA for ap solve\n%s",
                    stem,
                    wsclean_result.returncode,
                    diagnostic.decode("utf-8", errors="replace")[-500:],
                )
                predict_from_skymodel_wsclean(meridian_ms, sky, field="all")

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
            gaintable=[bp_table],
            interp=["nearest"],
        )
        if not os.path.exists(ap_table):
            log.error("Epoch gaincal [%s]: ap solve produced no table", stem)
            return EpochGaincalResult(
                None,
                EpochGaincalStatus.SOLVER_NO_TABLE,
                "amplitude+phase gaincal produced no table (likely all solutions flagged at minsnr=3.0)",
            )

        ap_stats = _gain_flag_fractions(ap_table, bp_table)
        log.info("Epoch gaincal [%s]: ap.G %s", stem, _format_gain_flag_stats(ap_stats))
        if float(ap_stats["effective_fraction"]) > GAINCAL_FLAG_FRACTION_LIMIT:
            reason = (
                f"ap.G {_format_gain_flag_stats(ap_stats)} exceeds "
                f"{GAINCAL_FLAG_FRACTION_LIMIT * 100:.0f}% limit"
            )
            return EpochGaincalResult(None, EpochGaincalStatus.LOW_SNR, reason)

        log.info("Epoch gaincal [%s]: SUCCESS → %s", stem, ap_table)
        return EpochGaincalResult(ap_table, EpochGaincalStatus.SOLVED, None)

    except Exception as exc:
        log.error(
            "Epoch gaincal: FAILED (%s) — caller should fall back to static daily G table",
            exc,
        )
        return EpochGaincalResult(None, EpochGaincalStatus.EXCEPTION, str(exc))
