"""Bandpass (B) calibration solver."""

# ruff: noqa

from __future__ import annotations

import logging
import os

import numpy as np

from dsa110_continuum.calibration.casa_service import CASAService
from dsa110_continuum.calibration.solver_common import (
    _determine_spwmap_for_bptables,
    _track_calibration_provenance,
    _validate_solve_success,
    table,
    timed,
)

logger = logging.getLogger(__name__)

def _check_flag_fraction(
    caltable_path: str,
    max_flag_fraction: float = 0.25,
    cal_type: str = "calibration",
) -> float:
    """Check if flag fraction in calibration table exceeds threshold.

    Excludes fully-flagged (antenna, receptor) pairs from the calculation,
    since those indicate dead hardware rather than calibration problems.

    Parameters
    ----------
    caltable_path :
        Path to calibration table
    max_flag_fraction :
        Maximum allowed flag fraction (default: 0.25 = 25%)
    cal_type :
        Type of calibration for error message (e.g., "bandpass", "gain")

    Returns
    -------
        Actual flag fraction (0.0 to 1.0) for working receptors only

    Raises
    ------
    ValueError
        If flag fraction exceeds max_flag_fraction

    """
    import numpy as np

    with table(caltable_path, readonly=True, ack=False) as tb:
        if "FLAG" not in tb.colnames():
            logger.warning(f"No FLAG column in {caltable_path}, skipping flag check")
            return 0.0

        flags = tb.getcol("FLAG")

        # Calculate raw flag fraction
        total = flags.size
        flagged = int(np.sum(flags))
        raw_flag_fraction = flagged / total if total > 0 else 0.0

        # Calculate flag fraction excluding fully-dead antenna receptors.
        # A CASA calibration table stores one row per antenna/SPW, while the
        # FLAG cell stores receptor/polarization and channel axes. Treating the
        # first array axis as "antenna" incorrectly counts rows or channels as
        # dead antennas after the casa_tables row-axis normalization.
        antenna_ids = None
        if "ANTENNA1" in tb.colnames():
            antenna_ids = tb.getcol("ANTENNA1")

        if flags.ndim == 3 and antenna_ids is not None and len(antenna_ids) == flags.shape[0]:
            flag_stats = _flag_fraction_excluding_dead_receptors(flags, antenna_ids)
            effective_flag_fraction = flag_stats["effective_flag_fraction"]
            n_dead = flag_stats["dead_receptor_count"]
            n_dead_antennas = flag_stats["dead_antenna_count"]

            logger.info(
                f"Flag fraction in {cal_type} table: {raw_flag_fraction * 100:.1f}% raw "
                f"({flagged:,}/{total:,} solutions flagged)"
            )
            if n_dead > 0:
                logger.info(
                    f"  Excluding {n_dead} fully-flagged (dead) antenna receptors "
                    f"across {n_dead_antennas} antennas: "
                    f"effective flag fraction = {effective_flag_fraction * 100:.1f}% "
                    f"({flag_stats['working_receptor_count']} working receptors)"
                )
        else:
            # Fallback for other table shapes
            effective_flag_fraction = raw_flag_fraction
            n_dead = 0
            n_dead_antennas = 0
            logger.info(
                f"Flag fraction in {cal_type} table: {raw_flag_fraction * 100:.1f}% "
                f"({flagged:,}/{total:,} solutions flagged)"
            )

    if effective_flag_fraction > max_flag_fraction:
        dead_info = (
            f" (excluding {n_dead} dead antenna receptors across {n_dead_antennas} antennas)"
            if n_dead > 0
            else ""
        )
        raise ValueError(
            f"{cal_type.upper()} SOLVE FAILED: Excessive flagging detected{dead_info}.\n"
            f"  Effective flag fraction: {effective_flag_fraction * 100:.1f}% (threshold: {max_flag_fraction * 100:.0f}%)\n"
            f"  Raw flagged solutions: {flagged:,} / {total:,}\n\n"
            f"This indicates poor data quality or incorrect calibration setup.\n"
            f"Common causes:\n"
            f"  - Data not coherently phased to calibrator\n"
            f"  - Low SNR (calibrator too faint or too far from beam center)\n"
            f"  - RFI contamination\n"
            f"  - Incorrect MODEL_DATA (wrong flux or position)\n"
        )

    return effective_flag_fraction


def _flag_fraction_excluding_dead_receptors(
    flags: Any,
    antenna_ids: Any,
    *,
    dead_threshold: float = 0.99,
    excluded_receptors: set[tuple[int, int]] | None = None,
) -> dict[str, Any]:
    """Compute flag fraction after removing independently known dead receptors."""
    import numpy as np

    flags = np.asarray(flags, dtype=bool)
    antenna_ids = np.asarray(antenna_ids)
    if flags.ndim != 3:
        raise ValueError(f"Expected row-major 3D FLAG array, found shape {flags.shape}")
    if flags.shape[0] != antenna_ids.shape[0]:
        raise ValueError(
            f"Expected one ANTENNA1 value per FLAG row, found {antenna_ids.shape[0]} "
            f"antenna IDs for {flags.shape[0]} rows"
        )

    receptor_axis = 1
    receptor_count = flags.shape[receptor_axis]
    if receptor_count > 4:
        raise ValueError(f"Expected receptor axis at index 1, found shape {flags.shape}")

    unique_antennas = sorted(set(antenna_ids.tolist()))
    candidate_receptors = {
        (int(antenna_id), receptor_idx)
        for antenna_id in unique_antennas
        for receptor_idx in range(receptor_count)
    }
    dead_receptors = set(excluded_receptors or ()) & candidate_receptors
    working_flagged = 0
    working_total = 0

    for antenna_id in unique_antennas:
        antenna_mask = antenna_ids == antenna_id
        antenna_flags = flags[antenna_mask]
        for receptor_idx in range(receptor_count):
            receptor_flags = np.take(antenna_flags, receptor_idx, axis=receptor_axis)
            receptor_total = int(receptor_flags.size)
            receptor_flagged = int(np.sum(receptor_flags))
            receptor_fraction = receptor_flagged / receptor_total if receptor_total else 0.0
            receptor = (int(antenna_id), receptor_idx)
            if excluded_receptors is None and receptor_fraction >= dead_threshold:
                dead_receptors.add(receptor)
            if receptor not in dead_receptors:
                working_flagged += receptor_flagged
                working_total += receptor_total

    effective_flag_fraction = (
        working_flagged / working_total if working_total else float(np.mean(flags))
    )
    dead_antennas = {antenna_id for antenna_id, _ in dead_receptors}
    return {
        "effective_flag_fraction": float(effective_flag_fraction),
        "dead_receptor_count": len(dead_receptors),
        "dead_antenna_count": len(dead_antennas),
        "dead_receptors": sorted(dead_receptors),
        "working_receptor_count": len(unique_antennas) * receptor_count - len(dead_receptors),
        "working_flagged": int(working_flagged),
        "working_total": int(working_total),
    }


def _print_bandpass_solution_summary(
    caltable_path: str,
    ms: str,
) -> dict[str, Any]:
    """Print comprehensive summary of bandpass solution flagging.

    Provides aggregated flagging statistics that CASA doesn't report by default,
    giving visibility into where solutions were flagged due to SNR or other issues.

    Parameters
    ----------
    caltable_path :
        Path to the bandpass calibration table
    ms :
        Path to the measurement set (for antenna names)

    Returns
    -------
        Dictionary with detailed flagging statistics

    """
    import numpy as np

    stats: dict[str, Any] = {}

    # Get antenna names from MS
    ant_names = {}
    try:
        with table(f"{ms}::ANTENNA", readonly=True, ack=False) as ant_tb:
            names = ant_tb.getcol("NAME")
            for i, name in enumerate(names):
                ant_names[i] = name
    except Exception:
        pass  # Fall back to antenna IDs

    with table(caltable_path, readonly=True, ack=False) as tb:
        if "FLAG" not in tb.colnames():
            print("  [No FLAG column in calibration table]")
            return stats

        flags = tb.getcol("FLAG")  # Shape: (nant, nchan, npol) for bandpass
        antenna_ids = tb.getcol("ANTENNA1")
        spw_ids = tb.getcol("SPECTRAL_WINDOW_ID")

        if flags.ndim != 3:
            print(f"  [Unexpected table shape: {flags.shape}]")
            return stats

        _nrows, nchan, npol = flags.shape

        # Unique SPWs and antennas
        unique_spws = sorted(set(spw_ids))
        unique_ants = sorted(set(antenna_ids))
        n_spw = len(unique_spws)
        n_ant = len(unique_ants)

        # Overall statistics
        total_solutions = flags.size
        total_flagged = int(np.sum(flags))
        overall_frac = total_flagged / total_solutions if total_solutions > 0 else 0.0

        print("\n" + "─" * 80)
        print("BANDPASS SOLUTION SUMMARY")
        print("─" * 80)
        print(f"  Table: {os.path.basename(caltable_path)}")
        print(
            f"  Dimensions: {n_ant} antennas × {n_spw} SPWs × {nchan} channels × {npol} polarizations"
        )
        print(f"  Total solutions: {total_solutions:,}")
        print(f"  Flagged solutions: {total_flagged:,} ({overall_frac * 100:.2f}%)")

        stats["total_solutions"] = total_solutions
        stats["total_flagged"] = total_flagged
        stats["overall_fraction"] = overall_frac

        # Per-SPW breakdown
        print("\n  Per-SPW Flagging:")
        print("  " + "-" * 50)
        spw_stats = {}
        high_flag_spws = []

        for spw in unique_spws:
            spw_mask = spw_ids == spw
            spw_flags = flags[spw_mask, :, :]
            spw_total = spw_flags.size
            spw_flagged = int(np.sum(spw_flags))
            spw_frac = spw_flagged / spw_total if spw_total > 0 else 0.0
            spw_stats[spw] = {"flagged": spw_flagged, "total": spw_total, "fraction": spw_frac}

            # Build a compact bar representation
            bar_len = 20
            filled = int(spw_frac * bar_len)
            bar = "█" * filled + "░" * (bar_len - filled)

            # Mark high-flagging SPWs
            marker = ""
            if spw_frac > 0.30:
                marker = "  HIGH"
                high_flag_spws.append(spw)
            elif spw_frac > 0.15:
                marker = " "

            print(
                f"    SPW {spw:2d}: [{bar}] {spw_frac * 100:5.1f}% ({spw_flagged:5d}/{spw_total:5d}){marker}"
            )

        stats["per_spw"] = spw_stats

        # Per-SPW per-Channel breakdown (detailed view showing ALL channels)
        print("\n  Per-Channel Flagging (all SPWs × channels):")
        print("  " + "-" * 70)
        chan_stats: dict[int, dict[int, dict[str, Any]]] = {}

        for spw in unique_spws:
            spw_mask = spw_ids == spw
            spw_flags = flags[spw_mask, :, :]  # Shape: (n_ants_in_spw, nchan, npol)
            chan_stats[spw] = {}

            print(f"    SPW {spw:2d}:")
            for chan_idx in range(nchan):
                # Flagging for this channel across all antennas and polarizations
                chan_flags = spw_flags[:, chan_idx, :]  # Shape: (n_ants_in_spw, npol)
                chan_total = chan_flags.size
                chan_flagged = int(np.sum(chan_flags))
                chan_frac = chan_flagged / chan_total if chan_total > 0 else 0.0
                chan_stats[spw][chan_idx] = {
                    "flagged": chan_flagged,
                    "total": chan_total,
                    "fraction": chan_frac,
                }

                # Build compact bar (10 chars for channel-level detail)
                bar_len = 10
                filled = int(chan_frac * bar_len)
                bar = "█" * filled + "░" * (bar_len - filled)

                # Mark high-flagging channels
                marker = ""
                if chan_frac >= 0.999:
                    marker = "  DEAD"
                elif chan_frac > 0.50:
                    marker = "  HIGH"
                elif chan_frac > 0.20:
                    marker = " "

                print(
                    f"      chan {chan_idx:3d}: [{bar}] {chan_flagged:4d} of {chan_total:4d} flagged "
                    f"({chan_frac * 100:5.1f}%){marker}"
                )

        stats["per_channel"] = chan_stats

        # Per-polarization breakdown
        print("\n  Per-Polarization Flagging:")
        print("  " + "-" * 50)
        pol_labels = ["RR/XX", "LL/YY"] if npol == 2 else [f"Pol{i}" for i in range(npol)]
        pol_stats = {}

        for pol_idx in range(npol):
            pol_flags = flags[:, :, pol_idx]
            pol_total = pol_flags.size
            pol_flagged = int(np.sum(pol_flags))
            pol_frac = pol_flagged / pol_total if pol_total > 0 else 0.0
            pol_stats[pol_labels[pol_idx]] = {
                "flagged": pol_flagged,
                "total": pol_total,
                "fraction": pol_frac,
            }
            print(
                f"    {pol_labels[pol_idx]:6s}: {pol_frac * 100:5.1f}% flagged ({pol_flagged:,}/{pol_total:,})"
            )

        stats["per_polarization"] = pol_stats

        # Per-antenna breakdown (show top 10 highest flagged)
        print("\n  Per-Antenna Flagging (top 10 highest):")
        print("  " + "-" * 50)
        ant_stats = {}

        for ant in unique_ants:
            ant_mask = antenna_ids == ant
            ant_flags = flags[ant_mask, :, :]
            ant_total = ant_flags.size
            ant_flagged = int(np.sum(ant_flags))
            ant_frac = ant_flagged / ant_total if ant_total > 0 else 0.0
            ant_name = ant_names.get(ant, str(ant))
            ant_stats[ant] = {
                "name": ant_name,
                "flagged": ant_flagged,
                "total": ant_total,
                "fraction": ant_frac,
            }

        # Sort by flagging fraction (descending)
        sorted_ants = sorted(ant_stats.items(), key=lambda x: x[1]["fraction"], reverse=True)

        # Identify dead antennas (100% flagged)
        dead_ants = [ant for ant, info in sorted_ants if info["fraction"] >= 0.999]
        partial_ants = [ant for ant, info in sorted_ants if 0.40 < info["fraction"] < 0.999]

        for ant, info in sorted_ants[:10]:
            bar_len = 20
            filled = int(info["fraction"] * bar_len)
            bar = "█" * filled + "░" * (bar_len - filled)

            marker = ""
            if info["fraction"] >= 0.999:
                marker = "  DEAD"
            elif info["fraction"] > 0.50:
                marker = "  BAD POL?"
            elif info["fraction"] > 0.30:
                marker = " "

            print(
                f"    Ant {info['name']:>4s} ({ant:3d}): [{bar}] {info['fraction'] * 100:5.1f}%{marker}"
            )

        stats["per_antenna"] = ant_stats
        stats["dead_antennas"] = dead_ants
        stats["partial_antennas"] = partial_ants

        # Summary recommendations
        print("\n  Summary:")
        print("  " + "-" * 50)

        if dead_ants:
            dead_names = [ant_stats[a]["name"] for a in dead_ants]
            print(f"    • {len(dead_ants)} dead antenna(s): {', '.join(dead_names)}")

        if partial_ants:
            partial_names = [ant_stats[a]["name"] for a in partial_ants]
            print(
                f"    • {len(partial_ants)} antenna(s) with partial flagging (40-99%): {', '.join(partial_names)}"
            )
            print("      → May indicate bad polarization(s); run: flag-bad-polarizations")

        if high_flag_spws:
            print(f"    • {len(high_flag_spws)} SPW(s) with >30% flagging: {high_flag_spws}")
            print("      → Check for RFI in these frequency ranges")

        # Exclude dead antennas from effective calculation
        if dead_ants:
            working_ants = [a for a in unique_ants if a not in dead_ants]
            working_mask = np.isin(antenna_ids, working_ants)
            working_flags = flags[working_mask, :, :]
            working_total = working_flags.size
            working_flagged = int(np.sum(working_flags))
            effective_frac = working_flagged / working_total if working_total > 0 else 0.0
            print(f"\n    Effective flagging (excl. dead antennas): {effective_frac * 100:.2f}%")
            stats["effective_fraction"] = effective_frac
        else:
            stats["effective_fraction"] = overall_frac

        print("─" * 80)

    return stats

def _check_coherent_phasing(
    ms: str,
    field_selector: str,
    max_ra_scatter_arcsec: float = 60.0,
) -> None:
    """Check if fields are coherently phased (not meridian tracking).

    DSA-110 data is initially phased to meridian (RA = LST). For calibration,
    fields must be rephased to calibrator using CASA phaseshift task.

    Parameters
    ----------
    ms : str
        Path to Measurement Set
    field_selector : str
        Field selection string
    max_ra_scatter_arcsec : float, optional
        Maximum allowed RA scatter in arcseconds

    Raises
    ------
    ValueError
        If RA scatter across fields exceeds threshold (meridian tracking)

    """
    import numpy as np  # type: ignore[import]

    # Parse field selection - single field doesn't need scatter check
    if "~" not in str(field_selector):
        return
    parts = str(field_selector).split("~")
    field_indices = list(range(int(parts[0]), int(parts[1]) + 1))

    # Read PHASE_DIR from FIELD table
    from dsa110_continuum.calibration.field_directions import (
        extract_field_ra_dec as _extract_field_ra_dec,
    )

    with table(f"{ms}::FIELD", readonly=True, ack=False) as field_tb:
        if "PHASE_DIR" not in field_tb.colnames():
            logger.warning("PHASE_DIR column not found - skipping coherence check")
            return
        phase_dir = field_tb.getcol("PHASE_DIR")

    # Get RA values for selected fields (in radians); helper handles both
    # rows-first (nfields, 1, 2) and CASA column-major (nfields, 2, 1) shapes.
    ra_all, _ = _extract_field_ra_dec(phase_dir)
    ra_values = np.array([ra_all[i] for i in field_indices if i < len(phase_dir)])
    if len(ra_values) < 2:
        return

    # Calculate RA scatter (handling wrap-around at 2π)
    ra_mean = np.arctan2(np.mean(np.sin(ra_values)), np.mean(np.cos(ra_values)))
    ra_diff = np.angle(np.exp(1j * (ra_values - ra_mean)))
    ra_scatter_arcsec = np.rad2deg(np.std(ra_diff)) * 3600
    ra_span_arcsec = np.rad2deg(np.ptp(ra_diff)) * 3600

    logger.debug(
        "Phase center RA: scatter=%.1f arcsec, span=%.1f arcsec (%d fields)",
        ra_scatter_arcsec,
        ra_span_arcsec,
        len(ra_values),
    )

    if ra_scatter_arcsec > max_ra_scatter_arcsec:
        est_duration_min = ra_span_arcsec / 54000 * 60  # LST: 15°/hour = 54000 arcsec/hour
        raise ValueError(
            f"COHERENT PHASING CHECK FAILED: Fields NOT coherently phased.\n"
            f"  RA scatter: {ra_scatter_arcsec:.1f} arcsec > {max_ra_scatter_arcsec:.1f} threshold\n"
            f"  RA span: {ra_span_arcsec:.1f} arcsec (~{est_duration_min:.1f} min LST drift)\n\n"
            f"Data is still MERIDIAN-phased (RA=LST). Use phaseshift_ms() to rephase:\n"
            f"  from dsa110_continuum.calibration.runner import phaseshift_ms\n"
            f"  phaseshift_ms('{ms}', mode='calibrator', calibrator_name='<CAL>')\n\n"
            f"This handles both phaseshift and REFERENCE_DIR sync for ft().\n"
            f"Then recalculate MODEL_DATA to match the new phase center."
        )

    logger.info(
        " Coherent phasing OK: RA scatter=%.1f arcsec (< %.1f threshold)",
        ra_scatter_arcsec,
        max_ra_scatter_arcsec,
    )


def _validate_bandpass_model_data(ms: str, cal_field: str) -> None:
    """Validate MODEL_DATA exists and is populated for bandpass solve.

    Parameters
    ----------
    ms : str
        Path to Measurement Set
    cal_field : str
        Field selection

    Raises
    ------
    ValueError
        If MODEL_DATA is missing or unpopulated.

    """
    import numpy as np

    logger.info(f"Validating MODEL_DATA for bandpass solve on field(s) {cal_field}...")
    with table(ms) as tb:
        if "MODEL_DATA" not in tb.colnames():
            raise ValueError(
                "MODEL_DATA column does not exist in MS. "
                "This is a required precondition for bandpass calibration. "
                "Populate MODEL_DATA using setjy, ft(), or a catalog model before "
                "calling solve_bandpass()."
            )

        # Parse field selection to determine which field(s) to check
        if "~" in str(cal_field):
            check_field = int(str(cal_field).split("~")[0])
        elif str(cal_field).isdigit():
            check_field = int(cal_field)
        else:
            check_field = None

        # Check if MODEL_DATA is populated for the calibration field
        if check_field is not None:
            field_ids = tb.getcol("FIELD_ID")
            field_mask = field_ids == check_field
            field_rows = np.where(field_mask)[0]
            if len(field_rows) == 0:
                raise ValueError(f"No data found for field {check_field}. Check field selection.")
            sample_rows = field_rows[: min(100, len(field_rows))]
            model_sample = np.array([tb.getcell("MODEL_DATA", int(r)) for r in sample_rows])
        else:
            model_sample = tb.getcol("MODEL_DATA", startrow=0, nrow=min(100, tb.nrows()))

        if np.all(np.abs(model_sample) < 1e-10):
            raise ValueError(
                f"MODEL_DATA column exists but is all zeros for field {cal_field}. "
                "This is a required precondition for bandpass calibration. "
                "Populate MODEL_DATA using setjy, ft(), or a catalog model before "
                "calling solve_bandpass()."
            )


def _run_bandpass_with_progress(
    casa_bandpass_func,
    kwargs: dict,
    ms: str,
    poll_interval: float = 5.0,
    live_channel_output: bool = True,
) -> None:
    """Run CASA bandpass task with progress monitoring.

    CASA's bandpass task is C++ and doesn't provide Python-level progress callbacks.
    This wrapper monitors the CASA log file in real-time and displays live per-channel
    progress showing ALL channels as they are solved.

    Parameters
    ----------
    casa_bandpass_func :
        The CASA bandpass task function
    kwargs :
        Arguments to pass to bandpass task
    ms :
        Path to Measurement Set (for estimating total work)
    poll_interval :
        How often to report progress (seconds)
    live_channel_output :
        If True, show live per-channel progress grid
    """
    import os

    from dsa110_continuum.utils.progress import (
        BandpassChannelMonitor,
        StageProgressMonitor,
        estimate_calibration_time,
    )

    caltable = kwargs.get("caltable", "unknown")

    # Get MS info for progress estimation
    try:
        with table(ms, ack=False) as t:
            n_rows = t.nrows()
        with table(f"{ms}::SPECTRAL_WINDOW", ack=False) as tspw:
            n_spws = tspw.nrows()
            n_chan = tspw.getcol("NUM_CHAN")[0]
        with table(f"{ms}::ANTENNA", ack=False) as tant:
            n_ant = tant.nrows()
    except Exception:
        n_rows, n_spws, n_chan, n_ant = 0, 16, 48, 117

    # Estimate expected runtime based on data size
    estimated_seconds = estimate_calibration_time(n_rows, n_spws, n_ant)

    if live_channel_output:
        # Use live channel monitor that shows ALL channels as they're solved
        casa_log = os.environ.get("CASALOGFILE", "")
        monitor_ch: BandpassChannelMonitor = BandpassChannelMonitor(
            n_spws=n_spws,
            n_chans=n_chan,
            casa_log_path=casa_log if casa_log else None,
            poll_interval=poll_interval,
        )
        with monitor_ch:
            casa_bandpass_func(**kwargs)
    else:
        # Fallback: use simple stage progress monitor
        monitor_sp: StageProgressMonitor = StageProgressMonitor(
            "Bandpass solve",
            output_path=caltable,
            poll_interval=poll_interval,
            estimated_seconds=estimated_seconds,
        )
        monitor_sp.set_context(rows=n_rows, SPWs=n_spws, channels=n_chan, antennas=n_ant)

        with monitor_sp:
            casa_bandpass_func(**kwargs)


def _determine_field_selector(
    cal_field: str, combine_fields: bool, peak_field_idx: int | None
) -> str:
    """Determine CASA field selector based on combine_fields setting.

    Parameters
    ----------
    cal_field : str
        Field selection
    combine_fields : bool
        Whether to combine fields
    peak_field_idx : int | None
        Index of peak field
    """
    if combine_fields:
        return str(cal_field)
    if peak_field_idx is not None:
        return str(peak_field_idx)
    if "~" in str(cal_field):
        return str(cal_field).split("~")[0]
    return str(cal_field)


def _build_bandpass_combine_string(
    combine: str | None, combine_fields: bool, combine_spw: bool
) -> str:
    """Build combine string for bandpass solve.

    Parameters
    ----------
    combine: Optional[str] :

    """
    if combine:
        logger.debug(f"Using custom combine string: {combine}")
        return combine

    comb_parts = ["scan"]
    if combine_fields:
        comb_parts.append("field")
    if combine_spw:
        comb_parts.append("spw")
    return ",".join(comb_parts)


def _log_spw_verification(ms: str, field_selector: str, combine_spw: bool) -> None:
    """Log SPW selection verification for bandpass solve.

    Parameters
    ----------
    ms : str
        Path to Measurement Set
    field_selector : str
        Field selection string
    combine_spw : bool
        Whether SPWs are combined
    """
    import numpy as np

    logger.info("\n" + "=" * 70)
    logger.info("SPW SELECTION VERIFICATION")
    logger.info("=" * 70)

    with table(f"{ms}::SPECTRAL_WINDOW", ack=False) as tspw:
        n_spws = tspw.nrows()
        spw_ids = list(range(n_spws))
        ref_freqs = tspw.getcol("REF_FREQUENCY")
        num_chan = tspw.getcol("NUM_CHAN")
        logger.info(f"MS contains {n_spws} spectral windows: SPW {spw_ids[0]} to SPW {spw_ids[-1]}")
        logger.info(f"  Frequency range: {ref_freqs[0] / 1e9:.4f} - {ref_freqs[-1] / 1e9:.4f} GHz")
        logger.info(f"  Total channels across all SPWs: {np.sum(num_chan)}")

    with table(ms, ack=False) as tb:
        field_ids = tb.getcol("FIELD_ID")
        spw_ids_in_data = tb.getcol("DATA_DESC_ID")

        with table(f"{ms}::DATA_DESCRIPTION", ack=False) as tdd:
            data_desc_to_spw = tdd.getcol("SPECTRAL_WINDOW_ID")

        if "~" not in str(field_selector):
            try:
                field_idx = int(field_selector)
                field_mask = field_ids == field_idx
                spw_ids_with_data = np.unique(data_desc_to_spw[spw_ids_in_data[field_mask]])
            except ValueError:
                spw_ids_with_data = np.unique(data_desc_to_spw[spw_ids_in_data])
        else:
            spw_ids_with_data = np.unique(data_desc_to_spw[spw_ids_in_data])

        spw_ids_list = sorted(spw_ids_with_data)
        logger.info(f"\nSPWs with data for field(s) '{field_selector}': {spw_ids_list}")
        logger.info(f"  Total SPWs to be processed: {len(spw_ids_list)}")

        if combine_spw:
            logger.info("\n  COMBINE='spw' is ENABLED:")
            logger.info(
                f"    :arrow_right: All {len(spw_ids_list)} SPWs will be used together in a single solve"
            )
            logger.info("    :arrow_right: Solution will be stored in SPW ID 0 (aggregate SPW)")
            logger.info(
                f"    :arrow_right: This improves SNR by using all {len(spw_ids_list)} subbands simultaneously"
            )
        else:
            logger.info("\n  COMBINE='spw' is DISABLED:")
            logger.info(
                f"    :arrow_right: Each of the {len(spw_ids_list)} SPWs will be solved separately"
            )
            logger.info(f"    :arrow_right: Solutions will be stored in SPW IDs {spw_ids_list}")

    logger.info("=" * 70 + "\n")


def _run_bandpass_diagnostics(
    ms: str,
    cal_field: str,
    bpcal_table: str,
    calibrator_name: str | None,
    refant: str,
    flag_fraction: float,
    generate_report: bool = True,
    report_output_dir: str | None = None,
) -> str | None:
    """Run comprehensive bandpass quality diagnostics.

    This function integrates the bandpass diagnostic framework to identify
    root causes of high flagging and provide actionable recommendations.
    Optionally generates a comprehensive HTML report with diagnostic figures.

    Parameters
    ----------
    ms :
        Path to Measurement Set
    cal_field :
        Field selection
    bpcal_table :
        Path to bandpass calibration table
    calibrator_name :
        Calibrator name (e.g., "0834+555")
    refant :
        Reference antenna
    flag_fraction :
        Overall flagging fraction (0.0 to 1.0)
    generate_report :
        If True, generate HTML diagnostics report (default: True)
    report_output_dir :
        Directory for HTML report. If None, uses parent of bpcal_table

    Returns
    -------
        Path to generated HTML report, or None if report generation disabled/failed

    """
    report_path: str | None = None

    try:
        from dsa110_continuum.calibration.bandpass_diagnostics import (
            analyze_flagging_pattern,
            diagnose_bandpass_quality,
            extract_bandpass_flagging_stats,
        )

        print("\n" + "-" * 80)
        print("RUNNING BANDPASS DIAGNOSTIC ANALYSIS")
        print("-" * 80)

        # Extract detailed flagging statistics
        print("→ Extracting flagging statistics...")
        flagging_stats = extract_bandpass_flagging_stats(bpcal_table)

        # Analyze pattern
        pattern = analyze_flagging_pattern(flagging_stats)
        print(f"→ Flagging pattern detected: {pattern.upper()}")

        pattern_descriptions = {
            "channel_specific": "RFI contamination in specific channels",
            "spw_specific": "Edge channel or bandpass rolloff effects",
            "antenna_specific": "Bad antenna or reference antenna issue",
            "uniform": "Systematic setup error (geometry, model, flux)",
            "random": "SNR/noise limited, likely phase decorrelation",
            "unknown": "Unable to determine clear pattern",
        }
        print(f"  {pattern_descriptions.get(pattern, 'Unknown pattern')}")

        # Run full diagnostic if calibrator name available
        if calibrator_name:
            print(f"→ Running comprehensive diagnostic (calibrator: {calibrator_name})...")
            print()

            diagnosis = diagnose_bandpass_quality(
                ms, cal_field, bpcal_table, calibrator_name, refant
            )

            # Print diagnostic report
            print(str(diagnosis))

            # Highlight critical actions
            if diagnosis.severity in ["critical", "high"]:
                print("\n" + "!" * 80)
                print("CRITICAL: IMMEDIATE ACTION REQUIRED")
                print("!" * 80)
                print(f"Root Cause: {diagnosis.root_cause}")
                print(f"Confidence: {diagnosis.confidence:.0%}")
                print("\nRecommended Actions (in priority order):")
                for i, fix in enumerate(diagnosis.fixes, 1):
                    print(f"  {i}. {fix}")
                print("!" * 80)
        else:
            print("  (Skipping detailed diagnostic - calibrator name not provided)")
            print("\n  Basic recommendations:")
            print("  - Check that data is coherently phased to calibrator")
            print("  - Verify pre-bandpass phase correction was applied")
            print("  - Inspect for RFI if channel-specific pattern")
            print("  - Consider re-phasing if flagging >15%")

        print("-" * 80 + "\n")

        # Generate HTML report if requested
        if generate_report:
            try:
                from dsa110_continuum.calibration.bandpass_report import generate_bandpass_report

                # Determine output directory
                if report_output_dir is None:
                    report_output_dir = os.path.dirname(bpcal_table)
                    if not report_output_dir:
                        report_output_dir = "."

                print("→ Generating HTML diagnostics report...")
                report_path = generate_bandpass_report(
                    ms_path=ms,
                    bpcal_path=bpcal_table,
                    output_dir=report_output_dir,
                    calibrator_name=calibrator_name or "unknown",
                )
                print(f" HTML report saved: {report_path}")
            except Exception as e:
                logger.warning(f"Failed to generate HTML report: {e}")
                print(f" HTML report generation failed: {e}")

    except ImportError as e:
        logger.warning(f"Bandpass diagnostic module not available: {e}")
        print(f" Diagnostic module not available: {e}")
        print("  Basic recommendation: Check geometry and pre-bandpass phase correction")
    except Exception as e:
        logger.error(f"Error running bandpass diagnostics: {e}", exc_info=True)
        print(f" Diagnostic analysis failed: {e}")
        print("  Please review logs and calibration setup manually")

    return report_path


@timed("calibration.solve_bandpass")
def solve_bandpass(
    ms: str,
    cal_field: str,
    refant: str,
    ktable: str | None,
    table_prefix: str | None = None,
    set_model: bool = True,
    model_standard: str = "Perley-Butler 2017",
    combine_fields: bool = True,  # Default: combine for higher SNR
    combine_spw: bool = False,  # Default: False - per-SPW solutions are faster and more accurate
    minsnr: float = 3.0,
    uvrange: str = ">1klambda",  # Default: exclude short baselines for better calibration
    fillgaps: int = 3,  # Interpolate across flagged channels up to this width
    minblperant: int = 4,  # Minimum baselines per antenna to include in solve
    prebandpass_phase_table: str | None = None,
    bp_smooth_type: str | None = None,
    bp_smooth_window: int | None = None,
    peak_field_idx: int | None = None,
    # Custom combine string (e.g., "scan,obs,field")
    combine: str | None = None,
    require_coherent_phasing: bool = True,
    max_flag_fraction: float = 0.05,
    calibrator_name: str | None = None,  # For diagnostics
    generate_diagnostics_report: bool = True,  # Generate HTML diagnostics report
    diagnostics_output_dir: str | None = None,  # Directory for diagnostics output
) -> list[str]:
    """Solve bandpass using CASA bandpass task with bandtype='B'.

        This solves for frequency-dependent bandpass correction using the dedicated
        bandpass task, which properly handles per-channel solutions. The bandpass task
        requires a source model (smodel) which is provided via MODEL_DATA column.

        Preconditions
    -------------
        - MODEL_DATA must be populated before calling this function.
        - When combine_fields=True, fields must be coherently phased.

    Notes
    -----
        - `ktable` is applied to the bandpass solve when provided and the file exists.
        - combine_spw=False is recommended for bandpass calibration because:
        - Each SPW has a different bandpass shape that should be solved independently.
        - Combining SPWs does NOT increase per-channel SNR (unlike gain calibration).
        - Per-SPW solutions are actually faster and produce less flagging.
        - This function now includes comprehensive bandpass quality diagnostics.
        - If flagging exceeds 5%, an automatic diagnostic analysis will be performed
        to identify the root cause and recommend fixes.

    Parameters
    ----------
    ms : str
        Path to Measurement Set.
    cal_field : str
        Field selection (e.g., "23" or "0~23").
    refant : str
        Reference antenna.
    ktable : str or None
        K-table to apply during bandpass solve. Applied if provided and file exists.
    table_prefix : str or None, optional
        Prefix for output calibration table. Default is None.
    set_model : bool, optional
        Not used (kept for compatibility). Default is True.
    model_standard : str, optional
        Not used (kept for compatibility). Default is "Perley-Butler 2017".
    combine_fields : bool, optional
        If True, combine across fields for higher SNR. Default is True.
    combine_spw : bool, optional
        If True, combine across spectral windows. Default is False.
    minsnr : float, optional
        Minimum SNR threshold for solutions. Default is 3.0.
    uvrange : str, optional
        UV range selection (e.g., ">1klambda"). Default is ">1klambda".
    fillgaps : int, optional
        Interpolate across flagged channels up to this width. Default is 3.
    minblperant : int, optional
        Minimum baselines per antenna to include in solve. Default is 4.
    prebandpass_phase_table : str or None, optional
        Pre-bandpass phase-only calibration table. Default is None.
    bp_smooth_type : str or None, optional
        Smoothing type for bandpass (e.g., "poly"). Default is None.
    bp_smooth_window : int or None, optional
        Smoothing window size. Default is None.
    peak_field_idx : int or None, optional
        Index of field with peak calibrator flux. Default is None.
    combine : str or None, optional
        Custom combine string (overrides combine_fields/combine_spw). Default is None.
    require_coherent_phasing : bool, optional
        If True, check coherent phasing. Default is True.
    max_flag_fraction : float, optional
        Maximum allowed flag fraction. Default is 0.05.
    calibrator_name : str or None, optional
        Calibrator name (e.g., "0834+555") for diagnostic framework. Default is None.
    generate_diagnostics_report : bool, optional
        If True, generate HTML diagnostics report. Default is True.
    diagnostics_output_dir : str or None, optional
        Directory for HTML report output. If None, uses table_prefix dir. Default is None.

    Returns
    -------
        list of str
        List of calibration table paths created.

    Raises
    ------
        ValueError
        If preconditions are not met or flag fraction exceeds limit.
    """
    service = CASAService()

    if table_prefix is None:
        table_prefix = f"{os.path.splitext(ms)[0]}_{cal_field}"

    # ============================================================================
    # PRE-CALIBRATION VALIDATION GATE
    # ============================================================================
    # Run comprehensive precondition checks before attempting bandpass calibration.
    # This prevents wasting compute on data that will inevitably fail.
    try:
        from dsa110_continuum.calibration.preconditions import (
            validate_bandpass_preconditions,
        )

        # Check all preconditions
        validation_result = validate_bandpass_preconditions(
            ms_path=ms,
            cal_field=cal_field,
            calibrator_name=calibrator_name,
            prebandpass_phase_table=prebandpass_phase_table,
            require_prebandpass_phase=True,  # Pre-BP phase is critical
        )

        if not validation_result.can_proceed:
            # Critical preconditions failed
            issues = "\n  - ".join(validation_result.blocking_issues)
            raise ValueError(
                f"Bandpass calibration preconditions not met:\n  - {issues}\n\n"
                f"Fix these issues before attempting bandpass calibration."
            )

        # Log warnings but continue
        for warning in validation_result.warnings:
            logger.warning(f"Precondition warning: {warning}")

    except ImportError:
        # Preconditions module not available, continue with legacy checks
        logger.debug("Preconditions module not available, using legacy checks")

    # ============================================================================
    # BANDPASS CALIBRATION (B) - TRANSPARENCY HEADER
    # ============================================================================
    logger.info("\n" + "=" * 80)
    logger.info("BANDPASS CALIBRATION (bandtype='B') - Stage 3")
    logger.info("=" * 80)
    logger.info("PURPOSE: Solve for frequency-dependent amplitude and phase variations")
    logger.info("         across each spectral window. This is the MOST CRITICAL")
    logger.info("         calibration for spectral line and continuum imaging.")
    logger.info("")
    logger.info("WHAT IT CORRECTS:")
    logger.info("  - Analog electronics: Filters, amplifiers with non-flat frequency response")
    logger.info("  - Digital channelization: Polyphase filterbank imperfections")
    logger.info("  - Cable reflections: Frequency-dependent transmission")
    logger.info("  - Atmospheric effects: Slight frequency dependence in tropospheric phase")
    logger.info("")
    logger.info("EXPECTED INPUT:")
    logger.info("  - MODEL_DATA: Populated with calibrator source model (validating...)")
    logger.info(f"  - Field: {cal_field}")
    logger.info(f"  - Reference antenna: {refant}")
    logger.info(f"  - Combine fields: {combine_fields} (higher SNR if True)")
    logger.info(f"  - Combine SPWs: {combine_spw} (NOT recommended - per-SPW is better)")
    logger.info(f"  - Minimum SNR: {minsnr}")
    logger.info(f"  - UV range: {uvrange} (exclude short baselines)")
    logger.info(f"  - Pre-BP phase table: {'Yes' if prebandpass_phase_table else 'No'}")
    logger.info("")
    logger.info("EXPECTED SOLUTIONS:")
    logger.info("  - Amplitude: 0.5 to 2.0 (relative to mean)")
    logger.info("    * Values <0.5 or >2.0 suggest hardware issues")
    logger.info("  - Phase: -180° to +180° (wrapped)")
    logger.info("    * Should vary smoothly across frequency within each SPW")
    logger.info("  - Structure: Per-channel (typically 384 channels/SPW × 16 SPWs)")
    logger.info("  - Edge channels: May be flagged (normal filter rolloff)")
    logger.info("  - Reference antenna: Often has flattest bandpass")
    logger.info("")
    logger.info("QUALITY ASSURANCE:")
    logger.info("  1. Flag fraction analysis:")
    logger.info("     - PRISTINE: <3% flagged - Target achieved!")
    logger.info("     - GOOD: 3-5% flagged - Acceptable quality")
    logger.info("     - MODERATE: 5-10% flagged - Running diagnostics")
    logger.info("     - HIGH: 10-20% flagged - Diagnostic required")
    logger.info("     - CRITICAL: >20% flagged - Systematic failure")
    logger.info("  2. Solution summary: Breakdown by antenna, SPW, channel")
    logger.info("  3. Automated diagnostics: Root cause analysis for high flagging")
    logger.info("  4. HTML report: Comprehensive plots and analysis (always generated)")
    logger.info("")
    logger.info("HOW SOLUTIONS ARE USED:")
    logger.info("  - Applied in final gain calibration (Stage 4)")
    logger.info("  - Applied to target data with all other calibrations")
    logger.info("  - Application order: K → B → G (bandpass after delays)")
    logger.info("  - Interpolation: Linear in time, nearest in frequency")
    logger.info("=" * 80 + "\n")

    # Determine field selector
    field_selector = _determine_field_selector(cal_field, combine_fields, peak_field_idx)
    logger.debug(
        f"Using field selector '{field_selector}' for bandpass calibration"
        + (
            f" (combined from range {cal_field})"
            if combine_fields
            else f" (peak field: {field_selector})"
        )
    )

    # Validate preconditions (early failures)
    if require_coherent_phasing and combine_fields:
        _check_coherent_phasing(ms, cal_field)

    _validate_bandpass_model_data(ms, cal_field)

    # Build combine string
    comb = _build_bandpass_combine_string(combine, combine_fields, combine_spw)

    # Log SPW verification
    _log_spw_verification(ms, field_selector, combine_spw)

    # Use bandpass task with bandtype='B' for proper bandpass calibration
    # The bandpass task requires MODEL_DATA to be populated (smodel source model)
    # uvrange='>1klambda' is the default to avoid short baselines
    # CRITICAL: Apply pre-bandpass phase-only calibration if provided. This corrects
    # phase drifts in raw uncalibrated data that cause decorrelation and low SNR.
    # CRITICAL: Apply K-table if provided. This flattens the phase slope across
    # the band, preventing decorherence if averaging channels or if delays are large.
    combine_desc = f" (combining across {comb})" if comb else ""
    phase_desc = " with pre-bandpass phase correction" if prebandpass_phase_table else ""
    k_desc = " with K-correction" if ktable else ""
    logger.info(
        f"Running bandpass solve using bandpass task (bandtype='B') on field {field_selector}"
        f"{combine_desc}{phase_desc}{k_desc}..."
    )
    kwargs = dict(
        vis=ms,
        caltable=f"{table_prefix}.b",
        field=field_selector,
        solint="inf",  # Per-channel solution (bandpass)
        refant=refant,
        combine=comb,
        solnorm=True,
        bandtype="B",  # Bandpass type B (per-channel)
        selectdata=True,  # Required to use uvrange parameter
        minsnr=minsnr,  # Minimum SNR threshold for solutions
        fillgaps=fillgaps,  # Interpolate across flagged channels
        minblperant=minblperant,  # Minimum baselines per antenna
    )
    # Set uvrange (default: '>1klambda' to avoid short baselines)
    if uvrange:
        kwargs["uvrange"] = uvrange

    # Construct gaintable list
    gaintables = []
    if ktable:
        # Check if K-table exists
        if os.path.exists(ktable):
            gaintables.append(ktable)
            logger.debug(f"  Applying K-table calibration: {ktable}")
        else:
            logger.warning(f"  K-table specified but not found: {ktable}")

    if prebandpass_phase_table:
        gaintables.append(prebandpass_phase_table)
        logger.debug(f"  Applying pre-bandpass phase-only calibration: {prebandpass_phase_table}")

    if gaintables:
        kwargs["gaintable"] = gaintables

        # Handle spwmap and interp for multiple tables
        # K-table: usually needs spwmap if combined, interp='linear' or 'nearest'
        # Pre-BP: needs spwmap if combined, interp='linear'

        # Determine spwmap for each table type independently (following solve_gains pattern)
        k_spwmap = None
        if ktable and os.path.exists(ktable):
            k_spwmap = _determine_spwmap_for_bptables([ktable], ms)

        p_spwmap = None
        if prebandpass_phase_table:
            p_spwmap = _determine_spwmap_for_bptables([prebandpass_phase_table], ms)

        # Build spwmaps and interps lists in the same order as gaintables was constructed
        spwmaps = None
        interps = None
        if k_spwmap or p_spwmap:
            # If any table needs mapping, construct the full spwmap list
            spwmaps = []
            interps = []
            # Add K-table mapping if it was added to gaintables
            if ktable and os.path.exists(ktable):
                spwmaps.append(k_spwmap if k_spwmap else [])
                interps.append("linear")
            # Add Pre-BP mapping if it was added to gaintables
            if prebandpass_phase_table:
                spwmaps.append(p_spwmap if p_spwmap else [])
                interps.append("linear")

        # Only pass spwmap if at least one table needs mapping
        if spwmaps:
            kwargs["spwmap"] = spwmaps
        if interps:
            kwargs["interp"] = interps

    # Run bandpass with progress monitoring (CASA's C++ core doesn't report progress)
    # We monitor the caltable file growth to provide feedback during long solves
    _run_bandpass_with_progress(service.bandpass, kwargs, ms)

    # PRECONDITION CHECK: Verify bandpass solve completed successfully
    # This ensures we follow "measure twice, cut once" - verify solutions exist
    # immediately after solve completes, before proceeding.
    _validate_solve_success(f"{table_prefix}.b", refant=refant)

    # CHECK FLAG FRACTION: Fail early if too many solutions are flagged
    # This prevents wasting time on downstream calibration with bad solutions
    flag_fraction = _check_flag_fraction(
        f"{table_prefix}.b",
        max_flag_fraction=max_flag_fraction,
        cal_type="bandpass",
    )

    # ============================================================================
    # BANDPASS SOLUTION SUMMARY
    # ============================================================================
    # Print comprehensive summary of where solutions were flagged
    # This provides the aggregated view that CASA doesn't report by default
    _print_bandpass_solution_summary(f"{table_prefix}.b", ms)

    # ============================================================================
    # BANDPASS QUALITY DIAGNOSTICS (AUTOMATIC)
    # ============================================================================
    # If flagging > 5%, run comprehensive diagnostic analysis to identify
    # root cause and provide actionable recommendations
    print("\n" + "=" * 80)
    print("BANDPASS QUALITY ASSESSMENT")
    print("=" * 80)
    print(f"Overall flagging fraction: {flag_fraction * 100:.2f}%")

    if flag_fraction < 0.03:
        print(" PRISTINE calibration (<3% flagged) - Target achieved!")
        print("  No diagnostic analysis needed.")
    elif flag_fraction < 0.05:
        print(" GOOD calibration (3-5% flagged) - Acceptable quality")
        print("  Minor flagging, likely edge effects or isolated RFI.")
    elif flag_fraction < 0.10:
        print(" MODERATE flagging (5-10%) - Running diagnostics...")
        _run_bandpass_diagnostics(
            ms,
            cal_field,
            f"{table_prefix}.b",
            calibrator_name,
            refant,
            flag_fraction,
            generate_report=False,  # Report generated separately below
            report_output_dir=diagnostics_output_dir,
        )
    elif flag_fraction < 0.20:
        print(" HIGH flagging (10-20%) - DIAGNOSTIC REQUIRED")
        _run_bandpass_diagnostics(
            ms,
            cal_field,
            f"{table_prefix}.b",
            calibrator_name,
            refant,
            flag_fraction,
            generate_report=False,  # Report generated separately below
            report_output_dir=diagnostics_output_dir,
        )
    else:
        print(" CRITICAL flagging (>20%) - SYSTEMATIC FAILURE")
        print("  Running comprehensive diagnostic analysis...")
        _run_bandpass_diagnostics(
            ms,
            cal_field,
            f"{table_prefix}.b",
            calibrator_name,
            refant,
            flag_fraction,
            generate_report=False,  # Report generated separately below
            report_output_dir=diagnostics_output_dir,
        )

    # ============================================================================
    # BANDPASS DIAGNOSTICS HTML REPORT (ALWAYS GENERATED)
    # ============================================================================
    # Generate comprehensive HTML report with figures for every bandpass solve
    if generate_diagnostics_report:
        try:
            from dsa110_continuum.calibration.bandpass_report import generate_bandpass_report

            # Determine output directory
            output_dir = diagnostics_output_dir
            if output_dir is None:
                output_dir = os.path.dirname(f"{table_prefix}.b")
                if not output_dir:
                    output_dir = "."

            print("→ Generating HTML diagnostics report...")
            report_path = generate_bandpass_report(
                ms_path=ms,
                bpcal_path=f"{table_prefix}.b",
                output_dir=output_dir,
                calibrator_name=calibrator_name or "unknown",
            )
            print(f" HTML report saved: {report_path}")
        except Exception as e:
            logger.warning(f"Failed to generate HTML report: {e}")
            print(f" HTML report generation failed: {e}")

    print("=" * 80 + "\n")

    # Track provenance after successful solve
    _track_calibration_provenance(
        ms_path=ms,
        caltable_path=f"{table_prefix}.b",
        task_name="bandpass",
        params=kwargs,
    )
    logger.info(f":check: Bandpass solve completed: {table_prefix}.b")

    # Optional smoothing of bandpass table (post-solve), off by default
    if (
        bp_smooth_type
        and str(bp_smooth_type).lower() != "none"
        and bp_smooth_window
        and int(bp_smooth_window) > 1
    ):
        try:
            logger.info(
                f"Smoothing bandpass table '{table_prefix}.b' with {bp_smooth_type} (window={bp_smooth_window})..."
            )
            # Best-effort: in-place smoothing using same output table
            service.smoothcal(
                vis=ms,
                tablein=f"{table_prefix}.b",
                tableout=f"{table_prefix}.b",
                smoothtype=str(bp_smooth_type).lower(),
                smoothwindow=int(bp_smooth_window),
            )
            logger.info(":check: Bandpass table smoothing complete")
        except Exception as e:
            logger.warning(f"Could not smooth bandpass table via CASA smoothcal: {e}")

    out = [f"{table_prefix}.b"]

    # QA validation of bandpass calibration tables
    try:
        from dsa110_continuum.qa.pipeline_quality import check_calibration_quality

        check_calibration_quality(out, ms_path=ms, alert_on_issues=True)
    except Exception as e:
        logger.warning(f"QA validation failed: {e}")

    # If flagging is still high, we just warn and proceed instead of failing
    # This allows the pipeline to continue even with poor data, which can be inspected later
    if flag_fraction > max_flag_fraction:
        logger.warning(
            f"High flagging detected: {flag_fraction * 100:.1f}% > {max_flag_fraction * 100:.0f}% limit. "
            f"Proceeding anyway as 'max_flag_fraction' is soft limit in this mode."
        )

    return out
