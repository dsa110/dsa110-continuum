# ruff: noqa: D103,I001
"""Single-polarization failure detection for pre-calibration flagging."""

from __future__ import annotations

from contextlib import contextmanager


def CASAService(*args, **kwargs):
    from dsa110_continuum.calibration import flagging

    return flagging.CASAService(*args, **kwargs)


def flag_summary(*args, **kwargs):
    from dsa110_continuum.calibration import flagging

    return flagging.flag_summary(*args, **kwargs)


@contextmanager
def suppress_subprocess_stderr():
    from dsa110_continuum.calibration import flagging

    with flagging.suppress_subprocess_stderr():
        yield


def detect_and_flag_bad_polarizations(
    ms_path: str,
    snr_ratio_threshold: float = 5.0,
    min_good_snr: float = 10.0,
    phase_table: str | None = None,
    dry_run: bool = False,
) -> dict:
    """Detect and flag antennas with single-polarization failures.

    Some antennas may have one polarization (XX or YY) that is decorrelated or
    has hardware issues, causing very low SNR for that polarization during
    calibration. This function identifies such antennas using two methods:

    1. **Primary (if phase_table provided)**: Analyze SNR in a pre-bandpass phase
       calibration table. This is the most reliable method since it directly
       measures which polarizations failed during calibration.

    2. **Fallback (MS-only analysis)**: Compute phase coherence from raw visibilities.
       Less reliable but works without a calibration table.

    Parameters
    ----------
    ms_path :
        Path to Measurement Set
    snr_ratio_threshold :
        If one polarization's SNR is this many times
        lower than the other in the phase table, flag it (default: 5.0)
    min_good_snr :
        Minimum SNR for a "good" polarization (default: 10.0)
    phase_table :
        Path to pre-bandpass phase calibration table. If provided,
        analyze this table directly (most reliable). Otherwise, analyze
        the MS data using coherence metrics.
    dry_run :
        If True, only report statistics without flagging (default: False)

    Returns
    -------
    Dictionary with
        - 'bad_polarizations': List of (antenna_id, pol_idx, pol_name) tuples
        - 'antenna_stats': Dict mapping antenna ID to per-pol statistics
        - 'n_antennas_affected': Number of antennas with one bad polarization
        - 'total_flagged_before': Overall flagged fraction before action
        - 'total_flagged_after': Overall flagged fraction after action
        - 'action_taken': Whether any flags were applied
        - 'detection_method': 'phase_table' or 'ms_coherence'
        - 'amplitude_imbalanced_antennas': Antennas skipped by the MS fallback
          because their receptor amplitudes differ by at least 2x

    """
    import logging
    import os

    logger = logging.getLogger(__name__)

    result = {
        "bad_polarizations": [],
        "antenna_stats": {},
        "n_antennas_affected": 0,
        "total_flagged_before": 0.0,
        "total_flagged_after": 0.0,
        "action_taken": False,
        "detection_method": "unknown",
        "amplitude_imbalanced_antennas": [],
    }

    # Method 1: Analyze phase calibration table if provided
    if phase_table and os.path.exists(phase_table):
        logger.info(f"Analyzing phase table: {phase_table}")
        result["detection_method"] = "phase_table"
        bad_polarizations = _detect_bad_pols_from_caltable(
            phase_table, snr_ratio_threshold, min_good_snr, result
        )
    else:
        # Method 2: Analyze MS coherence (less reliable)
        if phase_table:
            logger.warning(f"Phase table not found: {phase_table}, falling back to MS analysis")
        logger.info("Using MS coherence analysis for polarization detection")
        result["detection_method"] = "ms_coherence"
        bad_polarizations = _detect_bad_pols_from_ms_coherence(
            ms_path, snr_ratio_threshold, min_good_snr, result
        )

    result["bad_polarizations"] = bad_polarizations
    result["n_antennas_affected"] = len(bad_polarizations)

    # Log findings
    if bad_polarizations:
        xx_bad = [a for a, p, n in bad_polarizations if p == 0]
        yy_bad = [a for a, p, n in bad_polarizations if p == 1]
        if xx_bad:
            logger.info(f"Detected {len(xx_bad)} antennas with bad XX polarization: {xx_bad}")
        if yy_bad:
            logger.info(f"Detected {len(yy_bad)} antennas with bad YY polarization: {yy_bad}")
    else:
        logger.info("No single-polarization failures detected")

    # Get initial flag fraction
    stats_before = flag_summary(ms_path)
    result["total_flagged_before"] = stats_before.get("total_fraction_flagged", 0.0)

    # Flag bad polarizations if not dry_run
    if bad_polarizations and not dry_run:
        service = CASAService()
        with suppress_subprocess_stderr():
            flagged_count = 0
            skipped_count = 0
            for ant_id, pol_idx, pol_name in bad_polarizations:
                try:
                    logger.info(f"Flagging antenna {ant_id} polarization {pol_name}")
                    service.flagdata(
                        vis=ms_path,
                        mode="manual",
                        antenna=str(ant_id),
                        correlation=pol_name,
                        action="apply",
                    )
                    flagged_count += 1
                except RuntimeError as e:
                    # MSSelectionNullSelection means no unflagged data for this selection
                    # This can happen if the antenna is already completely flagged
                    if "NullSelection" in str(e) or "zero rows" in str(e):
                        logger.debug(
                            f"Skipping antenna {ant_id} {pol_name} - already flagged or no data"
                        )
                        skipped_count += 1
                    else:
                        raise

            if flagged_count > 0:
                result["action_taken"] = True
            if skipped_count > 0:
                logger.info(f"Skipped {skipped_count} antennas (already flagged)")

    # Get final flag fraction
    if not dry_run and bad_polarizations:
        stats_after = flag_summary(ms_path)
        result["total_flagged_after"] = stats_after.get("total_fraction_flagged", 0.0)
    else:
        result["total_flagged_after"] = result["total_flagged_before"]

    return result


def _detect_bad_pols_from_caltable(
    caltable: str,
    snr_ratio_threshold: float,
    min_good_snr: float,
    result: dict,
) -> list:
    """Detect bad polarizations by analyzing a calibration table.

    Examines the SNR and flag columns in a CASA calibration table (typically
    a G-type phase solution) to find antennas where one polarization has
    significantly lower SNR or is completely flagged.

    Parameters
    ----------
    caltable :
        Path to calibration table
    snr_ratio_threshold :
        Flag if SNR ratio exceeds this (one pol much better)
    min_good_snr :
        Minimum SNR for a "good" polarization
    result :
        Dictionary to store antenna stats

    Returns
    -------
        List of (antenna_id, pol_idx, pol_name) tuples for bad polarizations

    """
    import logging

    import numpy as np
    from dsa110_continuum.calibration.casa_service import get_casa_tool

    logger = logging.getLogger(__name__)
    bad_polarizations = []

    table = get_casa_tool("table")
    tb = table()
    try:
        tb.open(caltable)

        ant1 = tb.getcol("ANTENNA1")
        snr = tb.getcol("SNR")  # shape: (npol, nchan, nrow)
        flags = tb.getcol("FLAG")

        # Get unique antennas
        unique_ants = np.unique(ant1)
        antenna_stats = {}

        for ant_id in unique_ants:
            ant_mask = ant1 == ant_id
            if ant_mask.sum() == 0:
                continue

            ant_snr = snr[:, :, ant_mask]
            ant_flags = flags[:, :, ant_mask]

            pol_stats = []
            for pol in range(min(2, ant_snr.shape[0])):
                pol_snr = ant_snr[pol, :, :]
                pol_flags = ant_flags[pol, :, :]

                # Get unflagged SNR values
                unflagged_mask = ~pol_flags
                if unflagged_mask.sum() == 0:
                    pol_stats.append({"mean_snr": 0, "flag_frac": 1.0, "status": "all_flagged"})
                    continue

                mean_snr = float(pol_snr[unflagged_mask].mean())
                flag_frac = float(pol_flags.mean())

                pol_stats.append({"mean_snr": mean_snr, "flag_frac": flag_frac, "status": "ok"})

            antenna_stats[int(ant_id)] = pol_stats

            # Check for single-polarization failure
            if len(pol_stats) >= 2:
                snr0 = pol_stats[0]["mean_snr"]
                snr1 = pol_stats[1]["mean_snr"]
                flag0 = pol_stats[0]["flag_frac"]
                flag1 = pol_stats[1]["flag_frac"]

                # Primary check: One polarization completely flagged, other not
                # This is the clearest signal of single-pol failure - we don't require
                # the good polarization to have high SNR since DSA-110 often has low SNR
                # in the pre-bandpass phase solution
                if flag1 > 0.9 and flag0 < 0.5:
                    # YY is 100% flagged, XX is mostly unflagged
                    bad_polarizations.append((int(ant_id), 1, "YY"))
                    logger.debug(
                        f"Antenna {ant_id}: YY 100% flagged in cal table "
                        f"(XX flag={flag0 * 100:.0f}%, SNR={snr0:.1f})"
                    )
                elif flag0 > 0.9 and flag1 < 0.5:
                    # XX is 100% flagged, YY is mostly unflagged
                    bad_polarizations.append((int(ant_id), 0, "XX"))
                    logger.debug(
                        f"Antenna {ant_id}: XX 100% flagged in cal table "
                        f"(YY flag={flag1 * 100:.0f}%, SNR={snr1:.1f})"
                    )
                # Secondary check: SNR ratio for partially flagged cases
                elif snr0 > 0 and snr1 > 0:
                    ratio = snr0 / snr1
                    if ratio > snr_ratio_threshold and snr1 < min_good_snr:
                        bad_polarizations.append((int(ant_id), 1, "YY"))
                        logger.debug(
                            f"Antenna {ant_id}: YY low SNR={snr1:.1f} vs XX SNR={snr0:.1f}"
                        )
                    elif ratio < 1 / snr_ratio_threshold and snr0 < min_good_snr:
                        bad_polarizations.append((int(ant_id), 0, "XX"))
                        logger.debug(
                            f"Antenna {ant_id}: XX low SNR={snr0:.1f} vs YY SNR={snr1:.1f}"
                        )

        result["antenna_stats"] = antenna_stats

    finally:
        tb.close()

    return bad_polarizations


def _detect_bad_pols_from_ms_coherence(
    ms_path: str,
    snr_ratio_threshold: float,
    min_good_snr: float,
    result: dict,
) -> list:
    """Detect bad polarizations from MS coherence analysis.

    Less reliable than calibration table analysis, but useful when no
    calibration table is available. Coherence is computed independently for
    each baseline before taking the median across baselines. This avoids
    cancellation between legitimate baseline-dependent sky phases. The fallback
    only assigns a receptor when amplitudes agree within 2x and at least one
    receptor has coherence of 0.5 or greater; otherwise a phase table is required.

    Parameters
    ----------
    ms_path :
        Path to Measurement Set
    snr_ratio_threshold :
        Used for coherence ratio threshold (2.5x default)
    min_good_snr :
        Minimum SNR for a "good" polarization
    result :
        Dictionary to store antenna stats

    Returns
    -------
        List of (antenna_id, pol_idx, pol_name) tuples for bad polarizations

    """
    import logging

    from dsa110_continuum.adapters import casa_tables as casatables
    import numpy as np

    logger = logging.getLogger(__name__)
    bad_polarizations = []

    # Use looser coherence ratio threshold - this detection is less reliable
    coherence_ratio_threshold = 2.0  # One pol 2x less coherent
    amplitude_ratio_threshold = 2.0
    min_reliable_coherence = 0.5

    try:
        with casatables.table(ms_path, readonly=True) as tb:
            n_rows = tb.nrows()
            if n_rows == 0:
                logger.warning(f"MS {ms_path} has no rows")
                return bad_polarizations

            ant1 = tb.getcol("ANTENNA1")
            ant2 = tb.getcol("ANTENNA2")
            data = tb.getcol("DATA")  # shape: (n_rows, n_pol, n_chan)
            flags = tb.getcol("FLAG")

            # Get number of polarizations
            n_pol = data.shape[1]
            if n_pol < 2:
                logger.warning("MS has only one polarization, skipping bad pol detection")
                return bad_polarizations

            # Get unique antenna IDs (excluding autocorrelations)
            cross_mask = ant1 != ant2
            all_antennas = np.unique(np.concatenate([ant1[cross_mask], ant2[cross_mask]]))

            # Compute per-antenna, per-polarization amplitude and coherence statistics
            antenna_stats = {}

            for ant_id in all_antennas:
                # Get all baselines involving this antenna
                mask = ((ant1 == ant_id) | (ant2 == ant_id)) & cross_mask
                ant_data = data[mask]
                ant_flags = flags[mask]
                partners = np.where(ant1[mask] == ant_id, ant2[mask], ant1[mask])

                if ant_data.size == 0:
                    continue

                # Compute coherence for each polarization
                pol_stats = []
                for pol in range(n_pol):
                    baseline_coherences = []
                    valid_amplitudes = []
                    n_valid = 0
                    for partner in np.unique(partners):
                        baseline_mask = partners == partner
                        pol_data = ant_data[baseline_mask, pol, :]
                        valid_data = pol_data[~ant_flags[baseline_mask, pol, :]]
                        amplitudes = np.abs(valid_data)
                        nonzero = amplitudes > 0
                        if not np.any(nonzero):
                            continue

                        unit_phasors = valid_data[nonzero] / amplitudes[nonzero]
                        baseline_coherences.append(float(np.abs(np.mean(unit_phasors))))
                        valid_amplitudes.append(amplitudes[nonzero])
                        n_valid += int(nonzero.sum())

                    if not baseline_coherences:
                        pol_stats.append({"coherence": 0, "n_valid": 0})
                        continue

                    pol_stats.append(
                        {
                            "coherence": float(np.median(baseline_coherences)),
                            "median_amplitude": float(np.median(np.concatenate(valid_amplitudes))),
                            "n_valid": n_valid,
                        }
                    )

                antenna_stats[int(ant_id)] = pol_stats

                # Check for single-polarization coherence issue
                if len(pol_stats) >= 2:
                    coh0 = pol_stats[0].get("coherence", 1.0)
                    coh1 = pol_stats[1].get("coherence", 1.0)
                    amp0 = pol_stats[0].get("median_amplitude", 0.0)
                    amp1 = pol_stats[1].get("median_amplitude", 0.0)

                    min_amplitude = min(amp0, amp1)
                    amplitude_ratio = (
                        max(amp0, amp1) / min_amplitude if min_amplitude > 0 else np.inf
                    )
                    if amplitude_ratio >= amplitude_ratio_threshold:
                        result.setdefault("amplitude_imbalanced_antennas", []).append(int(ant_id))
                        logger.warning(
                            "Antenna %s has %.2fx receptor amplitude imbalance; "
                            "skipping MS-coherence labeling and requiring a phase table",
                            ant_id,
                            amplitude_ratio,
                        )
                        continue

                    if max(coh0, coh1) < min_reliable_coherence:
                        logger.debug(
                            "Antenna %s has no reliably coherent receptor "
                            "(XX=%.3f, YY=%.3f); skipping MS-coherence labeling",
                            ant_id,
                            coh0,
                            coh1,
                        )
                        continue

                    if coh0 > 0 and coh1 > 0:
                        coh_ratio = coh0 / coh1
                        if coh_ratio > coherence_ratio_threshold:
                            # Pol 1 (YY) is decorrelated
                            bad_polarizations.append((int(ant_id), 1, "YY"))
                            logger.debug(
                                f"Antenna {ant_id}: YY polarization decorrelated "
                                f"(coherence={coh1:.3f} vs XX coherence={coh0:.3f})"
                            )
                        elif coh_ratio < 1 / coherence_ratio_threshold:
                            # Pol 0 (XX) is decorrelated
                            bad_polarizations.append((int(ant_id), 0, "XX"))
                            logger.debug(
                                f"Antenna {ant_id}: XX polarization decorrelated "
                                f"(coherence={coh0:.3f} vs YY coherence={coh1:.3f})"
                            )

            result["antenna_stats"] = antenna_stats

    except Exception as e:
        logger.error(f"Error in MS coherence analysis: {e}", exc_info=True)
        raise

    return bad_polarizations
