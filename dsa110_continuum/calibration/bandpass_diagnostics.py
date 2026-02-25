"""
Bandpass calibration quality diagnostics and automated recovery.

This module provides comprehensive diagnostics for bandpass calibration failures,
identifying root causes and automatically applying fixes to achieve pristine
calibration solutions (<3% flagging).
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from dsa110_contimg.common.utils.casa_init import ensure_casa_path

ensure_casa_path()

import casacore.tables as casatables

table = casatables.table

logger = logging.getLogger(__name__)


@dataclass
class DiagnosticReport:
    """Bandpass calibration diagnostic report."""

    # Primary diagnosis
    root_cause: str = "unknown"
    confidence: float = 0.0  # 0-1
    severity: str = "unknown"  # 'critical', 'high', 'medium', 'low'

    # Recommended fixes
    fixes: list[str] = field(default_factory=list)

    # Detailed metrics
    metrics: dict[str, Any] = field(default_factory=dict)

    # Flagging statistics
    overall_fraction_flagged: float = 0.0
    flagging_pattern: str = "unknown"

    # Issues and warnings
    issues: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        """Generate human-readable report."""
        lines = []
        lines.append("=" * 70)
        lines.append("BANDPASS CALIBRATION DIAGNOSTIC REPORT")
        lines.append("=" * 70)
        lines.append(f"Root Cause: {self.root_cause}")
        lines.append(f"Confidence: {self.confidence:.1%}")
        lines.append(f"Severity: {self.severity.upper()}")
        lines.append(f"Overall Flagging: {self.overall_fraction_flagged:.1%}")
        lines.append(f"Flagging Pattern: {self.flagging_pattern}")
        lines.append("")

        if self.issues:
            lines.append("Issues:")
            for issue in self.issues:
                lines.append(f"  - {issue}")
            lines.append("")

        if self.warnings:
            lines.append("Warnings:")
            for warning in self.warnings:
                lines.append(f"  - {warning}")
            lines.append("")

        if self.fixes:
            lines.append("Recommended Fixes:")
            for i, fix in enumerate(self.fixes, 1):
                lines.append(f"  {i}. {fix}")
            lines.append("")

        if self.metrics:
            lines.append("Detailed Metrics:")
            for key, value in self.metrics.items():
                if isinstance(value, float):
                    lines.append(f"  {key}: {value:.4f}")
                elif isinstance(value, (list, np.ndarray)):
                    if len(value) < 10:
                        lines.append(f"  {key}: {value}")
                    else:
                        lines.append(f"  {key}: [array with {len(value)} elements]")
                else:
                    lines.append(f"  {key}: {value}")
            lines.append("")

        lines.append("=" * 70)
        return "\n".join(lines)


def parse_casa_bandpass_flagging(casa_log_path: str | None = None) -> dict[str, Any]:
    """Parse CASA log output for bandpass flagging statistics.

    CASA outputs messages like:
    "22 of 192 solutions flagged due to SNR < 5 in spw=0 (chan=9) at 2025/10/02/15:42:35.2"

    Parameters
    ----------
    casa_log_path :
        Path to CASA log file. If None, searches current directory.
    casa_log_path: Optional[str] :
         (Default value = None)

    Returns
    -------
        Dict with flagging statistics per SPW and channel

    """
    # Find most recent CASA log if not specified
    if casa_log_path is None:
        import glob

        casa_logs = sorted(glob.glob("casa-*.log"), key=os.path.getmtime, reverse=True)
        if casa_logs:
            casa_log_path = casa_logs[0]
        else:
            logger.warning("No CASA log file found")
            return {}

    if not os.path.exists(casa_log_path):
        logger.warning(f"CASA log not found: {casa_log_path}")
        return {}

    # Parse log file
    pattern = r"(\d+) of (\d+) solutions flagged.*spw=(\d+).*\(chan=(\d+)\).*at ([\d/:]+)"

    flagging_data = {}

    with open(casa_log_path) as f:
        for line in f:
            match = re.search(pattern, line)
            if match:
                n_flagged = int(match.group(1))
                n_total = int(match.group(2))
                spw_id = int(match.group(3))
                chan_id = int(match.group(4))
                timestamp = match.group(5)

                if spw_id not in flagging_data:
                    flagging_data[spw_id] = {}

                flagging_data[spw_id][chan_id] = {
                    "n_flagged": n_flagged,
                    "n_total": n_total,
                    "fraction": n_flagged / n_total if n_total > 0 else 0.0,
                    "timestamp": timestamp,
                }

    # Compute overall statistics
    if flagging_data:
        total_flagged = sum(
            data["n_flagged"] for spw_data in flagging_data.values() for data in spw_data.values()
        )
        total_solutions = sum(
            data["n_total"] for spw_data in flagging_data.values() for data in spw_data.values()
        )

        return {
            "per_spw_chan": flagging_data,
            "overall_fraction_flagged": total_flagged / total_solutions
            if total_solutions > 0
            else 0.0,
            "total_flagged": total_flagged,
            "total_solutions": total_solutions,
        }

    return {}


def extract_bandpass_flagging_stats(bpcal_table: str) -> dict[str, Any]:
    """Extract detailed flagging statistics from bandpass calibration table.

    Parameters
    ----------
    bpcal_table :
        Path to bandpass calibration table

    Returns
    -------
        Dict with per-SPW, per-channel, per-antenna flagging statistics

    """
    if not os.path.exists(bpcal_table):
        raise FileNotFoundError(f"Bandpass table not found: {bpcal_table}")

    with table(bpcal_table, readonly=True) as tb:
        spw_ids = tb.getcol("SPECTRAL_WINDOW_ID")
        flags = tb.getcol("FLAG")  # Shape: (n_solutions, n_channels, n_pols)
        antenna_ids = tb.getcol("ANTENNA1")

        # Overall statistics
        overall_fraction = float(np.mean(flags))

        # Per-SPW statistics
        per_spw = {}
        unique_spws = np.unique(spw_ids)
        for spw_id in unique_spws:
            spw_mask = spw_ids == spw_id
            spw_flags = flags[spw_mask]
            per_spw[int(spw_id)] = {
                "fraction_flagged": float(np.mean(spw_flags)),
                "n_solutions": int(np.sum(spw_mask)),
            }

        # Per-antenna statistics
        per_antenna = {}
        unique_ants = np.unique(antenna_ids)
        for ant_id in unique_ants:
            ant_mask = antenna_ids == ant_id
            ant_flags = flags[ant_mask]
            per_antenna[int(ant_id)] = {
                "fraction_flagged": float(np.mean(ant_flags)),
                "n_solutions": int(np.sum(ant_mask)),
            }

        # Per-channel statistics (across all SPWs)
        n_channels = flags.shape[1]
        per_channel = {}
        for chan_id in range(n_channels):
            chan_flags = flags[:, chan_id, :]
            per_channel[chan_id] = {
                "fraction_flagged": float(np.mean(chan_flags)),
            }

    return {
        "overall_fraction_flagged": overall_fraction,
        "per_spw": per_spw,
        "per_antenna": per_antenna,
        "per_channel": per_channel,
    }


def analyze_flagging_pattern(flagging_stats: dict[str, Any]) -> str:
    """Determine spatial/spectral pattern of flagging.

    Patterns:
    - 'channel_specific': Certain channels flagged across SPWs (RFI)
    - 'spw_specific': Entire SPWs flagged (edge SPWs, attenuation)
    - 'antenna_specific': Certain antennas flagged (bad antenna)
    - 'random': Random distribution (low SNR, noise)
    - 'uniform': All flagged uniformly (model/geometry failure)

    Parameters
    ----------
    flagging_stats :
        Output from extract_bandpass_flagging_stats
    flagging_stats: Dict[str :

    Any] :


    Returns
    -------
        Pattern identifier string

    """
    # Compute variance across different dimensions
    per_channel = flagging_stats.get("per_channel", {})
    per_spw = flagging_stats.get("per_spw", {})
    per_antenna = flagging_stats.get("per_antenna", {})

    if not per_channel or not per_spw or not per_antenna:
        return "unknown"

    # Get flagging fractions
    channel_fractions = [data["fraction_flagged"] for data in per_channel.values()]
    spw_fractions = [data["fraction_flagged"] for data in per_spw.values()]
    antenna_fractions = [data["fraction_flagged"] for data in per_antenna.values()]

    # Compute standard deviations
    channel_std = np.std(channel_fractions)
    spw_std = np.std(spw_fractions)
    antenna_std = np.std(antenna_fractions)

    overall_std = np.std(channel_fractions + spw_fractions + antenna_fractions)

    # Determine dominant pattern
    if channel_std > 0.3 and channel_std > spw_std and channel_std > antenna_std:
        return "channel_specific"  # RFI
    elif spw_std > 0.3 and spw_std > channel_std and spw_std > antenna_std:
        return "spw_specific"  # Bandpass edges
    elif antenna_std > 0.3 and antenna_std > channel_std and antenna_std > spw_std:
        return "antenna_specific"  # Bad antenna
    elif overall_std < 0.05:
        return "uniform"  # Systematic issue
    else:
        return "random"  # SNR/noise limited


def check_geometric_setup(
    ms_path: str, cal_field: str, calibrator_name: str
) -> tuple[bool, dict[str, Any]]:
    """Verify data is coherently phased to calibrator.

    Checks:
    1. RA scatter across fields < 60 arcsec (coherent phasing)
    2. Calibrator near field centers
    3. Primary beam attenuation < 50% for all fields

    Parameters
    ----------
    ms_path :
        Path to Measurement Set
    cal_field :
        Field selection (e.g., "0~23")
    calibrator_name :
        Calibrator name for position lookup

    Returns
    -------
        (geometry_ok, metrics) tuple

    """
    from dsa110_contimg.core.calibration.runner import _get_calibrator_position

    # Get calibrator position
    try:
        cal_ra, cal_dec = _get_calibrator_position(calibrator_name)
    except Exception as e:
        logger.error(f"Failed to get calibrator position: {e}")
        return False, {"error": str(e)}

    # Parse field selection
    if "~" in cal_field:
        start, end = map(int, cal_field.split("~"))
        field_indices = list(range(start, end + 1))
    elif cal_field.isdigit():
        field_indices = [int(cal_field)]
    else:
        field_indices = list(range(24))  # Default: all fields

    # Check field phase centers
    with table(f"{ms_path}::FIELD", readonly=True) as tb:
        phase_dir = tb.getcol("PHASE_DIR")  # Shape: (nfields, 1, 2)
        ra_rad = phase_dir[:, 0, 0]
        dec_rad = phase_dir[:, 0, 1]

        # Filter to selected fields
        field_ra = np.degrees(ra_rad[field_indices])
        field_dec = np.degrees(dec_rad[field_indices])

    # Check RA scatter (coherent phasing indicator)
    # Use circular statistics for RA (handles wrap-around)
    ra_rad_selected = np.radians(field_ra)
    ra_complex = np.exp(1j * ra_rad_selected)
    mean_ra_complex = np.mean(ra_complex)
    ra_scatter_rad = np.std(np.angle(ra_complex / mean_ra_complex))
    ra_scatter_arcsec = np.degrees(ra_scatter_rad) * 3600

    # Check calibrator-field separation and primary beam weights
    separations = []
    pb_weights = []

    for fra, fdec in zip(field_ra, field_dec):
        # Compute separation (great circle distance)
        dra = np.radians(cal_ra - fra)
        ddec = np.radians(cal_dec - fdec)
        a = (
            np.sin(ddec / 2) ** 2
            + np.cos(np.radians(cal_dec)) * np.cos(np.radians(fdec)) * np.sin(dra / 2) ** 2
        )
        sep_rad = 2 * np.arcsin(np.sqrt(a))
        sep_arcsec = np.degrees(sep_rad) * 3600
        separations.append(sep_arcsec)

        # Primary beam weight at 1.4 GHz (FWHM ~ 2.5 deg for DSA-110)
        # Gaussian beam: exp(-(theta/theta_FWHM)^2 * 4*ln(2))
        theta_fwhm_arcsec = 2.5 * 3600  # 2.5 degrees
        pb_weight = np.exp(-4 * np.log(2) * (sep_arcsec / theta_fwhm_arcsec) ** 2)
        pb_weights.append(pb_weight)

    metrics = {
        "ra_scatter_arcsec": float(ra_scatter_arcsec),
        "calibrator_field_separations": separations,
        "pb_weights": pb_weights,
        "min_pb_weight": float(np.min(pb_weights)),
        "median_separation": float(np.median(separations)),
        "calibrator_ra": cal_ra,
        "calibrator_dec": cal_dec,
        "field_ra_range": [float(np.min(field_ra)), float(np.max(field_ra))],
        "field_dec_range": [float(np.min(field_dec)), float(np.max(field_dec))],
    }

    # Diagnosis
    issues = []
    geometry_ok = True

    if ra_scatter_arcsec > 60:
        issues.append(
            f'Data not coherently phased: RA scatter {ra_scatter_arcsec:.1f}" > 60" threshold'
        )
        geometry_ok = False

    if np.min(pb_weights) < 0.5:
        issues.append(f"Some fields have >50% PB attenuation: min weight {np.min(pb_weights):.2f}")
        geometry_ok = False

    if np.median(separations) > 1800:  # > 0.5 degrees
        issues.append(
            f'Calibrator far from field centers: median separation {np.median(separations):.1f}"'
        )
        geometry_ok = False

    metrics["issues"] = issues
    metrics["geometry_ok"] = geometry_ok

    return geometry_ok, metrics


def check_snr_budget(
    ms_path: str,
    cal_field: str,
    calibrator_name: str,
    flagging_stats: dict[str, Any],
) -> tuple[bool, dict[str, Any]]:
    """Compute expected SNR budget and compare to actual.

    SNR budget:
    SNR_per_baseline_channel = (S_cal / sigma_vis)
    SNR_total = SNR_per_bl_chan * sqrt(N_baseline * N_channel * N_time)

    Parameters
    ----------
    ms_path :
        Path to Measurement Set
    cal_field :
        Field selection
    calibrator_name :
        Calibrator name for flux lookup
    flagging_stats :
        Flagging statistics from extract_bandpass_flagging_stats

    Returns
    -------
        (snr_ok, metrics) tuple

    """
    try:
        # Get calibrator flux (assumes 1.4 GHz for DSA-110)
        # You may need to implement get_calibrator_flux or use existing catalog
        S_cal = 1.0  # Placeholder - implement catalog lookup
        logger.debug(f"Calibrator {calibrator_name} flux: {S_cal:.2f} Jy (placeholder)")
    except Exception as e:
        logger.warning(f"Could not get calibrator flux: {e}")
        S_cal = 1.0  # Fallback

    # Estimate visibility noise from unflagged data
    with table(ms_path, readonly=True) as tb:
        # Sample data to estimate noise
        n_rows = tb.nrows()
        sample_size = min(1000, n_rows)
        data_sample = tb.getcol("DATA", startrow=0, nrow=sample_size)
        flags_sample = tb.getcol("FLAG", startrow=0, nrow=sample_size)

        # Estimate noise from unflagged data
        unflagged_data = data_sample[~flags_sample]
        if len(unflagged_data) > 0:
            # Estimate noise as robust MAD (median absolute deviation)
            median_amp = np.median(np.abs(unflagged_data))
            mad = np.median(np.abs(np.abs(unflagged_data) - median_amp))
            sigma_vis = 1.4826 * mad  # Convert MAD to std dev
        else:
            sigma_vis = 1.0  # Fallback

        # Count parameters
        antenna_ids = tb.getcol("ANTENNA1")
        N_ant = len(np.unique(antenna_ids))
        N_baseline = N_ant * (N_ant - 1) // 2

        # Parse field selection
        field_ids = tb.getcol("FIELD_ID")
        if "~" in cal_field:
            start, end = map(int, cal_field.split("~"))
            target_fields = list(range(start, end + 1))
        elif cal_field.isdigit():
            target_fields = [int(cal_field)]
        else:
            target_fields = list(np.unique(field_ids))

        field_mask = np.isin(field_ids, target_fields)
        times = tb.getcol("TIME")[field_mask]
        N_time = len(np.unique(times))

    # Get number of channels
    with table(f"{ms_path}::SPECTRAL_WINDOW", readonly=True) as tb:
        num_chan = tb.getcol("NUM_CHAN")
        N_chan = int(np.sum(num_chan))

    # Compute expected SNR
    snr_per_bl_chan = S_cal / sigma_vis
    snr_expected = snr_per_bl_chan * np.sqrt(N_baseline * N_chan * N_time)

    # Get actual SNR from flagging stats (estimate from flagged fraction)
    # Lower SNR → higher flagging
    # Assume SNR threshold = 5, and flagging increases exponentially below threshold
    overall_fraction_flagged = flagging_stats["overall_fraction_flagged"]
    if overall_fraction_flagged < 0.01:
        snr_actual_estimate = 10.0  # High SNR
    elif overall_fraction_flagged < 0.1:
        snr_actual_estimate = 7.0
    elif overall_fraction_flagged < 0.3:
        snr_actual_estimate = 5.0
    else:
        snr_actual_estimate = 3.0  # Low SNR

    snr_deficit = snr_expected - snr_actual_estimate

    metrics = {
        "expected_snr": float(snr_expected),
        "actual_snr_estimate": float(snr_actual_estimate),
        "snr_deficit": float(snr_deficit),
        "calibrator_flux": float(S_cal),
        "visibility_noise": float(sigma_vis),
        "n_antennas": int(N_ant),
        "n_baselines": int(N_baseline),
        "n_times": int(N_time),
        "n_channels": int(N_chan),
    }

    # Diagnosis
    snr_ok = True
    if snr_deficit > 3:
        # Significant SNR loss
        metrics["issue"] = (
            f"SNR deficit of {snr_deficit:.1f} - likely decorrelation or low integration"
        )
        snr_ok = False

    return snr_ok, metrics


def diagnose_bandpass_quality(
    ms_path: str,
    cal_field: str,
    bpcal_table: str,
    calibrator_name: str,
    refant: str,
) -> DiagnosticReport:
    """Comprehensive bandpass calibration quality diagnostic.

    This function systematically checks for common failure modes and
    identifies the most likely root cause with confidence level.

    Parameters
    ----------
    ms_path :
        Path to Measurement Set
    cal_field :
        Field selection (e.g., "0~23")
    bpcal_table :
        Path to bandpass calibration table
    calibrator_name :
        Calibrator name (e.g., "0834+555")
    refant :
        Reference antenna

    Returns
    -------
        DiagnosticReport with root cause, confidence, and recommended fixes

    """
    report = DiagnosticReport()

    # Extract flagging statistics
    logger.info("Extracting flagging statistics from bandpass table...")
    flagging_stats = extract_bandpass_flagging_stats(bpcal_table)
    overall_fraction = flagging_stats["overall_fraction_flagged"]

    report.overall_fraction_flagged = overall_fraction
    report.metrics["flagging_stats"] = flagging_stats

    logger.info(f"Overall flagging: {overall_fraction * 100:.1f}%")

    # Analyze flagging pattern
    pattern = analyze_flagging_pattern(flagging_stats)
    report.flagging_pattern = pattern
    logger.info(f"Flagging pattern: {pattern}")

    # Determine severity
    if overall_fraction < 0.03:
        report.severity = "low"  # Pristine
    elif overall_fraction < 0.10:
        report.severity = "medium"  # Acceptable with investigation
    elif overall_fraction < 0.30:
        report.severity = "high"  # Problematic
    else:
        report.severity = "critical"  # Failure

    # Phase 1: Check geometry (most common issue)
    logger.info("Checking geometric setup...")
    geometry_ok, geometry_metrics = check_geometric_setup(ms_path, cal_field, calibrator_name)
    report.metrics["geometry"] = geometry_metrics

    if not geometry_ok:
        report.root_cause = "geometric_phase_error"
        report.confidence = 0.9
        report.fixes = [
            "rephase_to_calibrator",
            "verify_field_selection",
            "check_calibrator_position",
        ]
        report.issues.extend(geometry_metrics.get("issues", []))
        return report

    # Phase 2: Check SNR budget
    logger.info("Checking SNR budget...")
    snr_ok, snr_metrics = check_snr_budget(ms_path, cal_field, calibrator_name, flagging_stats)
    report.metrics["snr_budget"] = snr_metrics

    if not snr_ok:
        report.root_cause = "insufficient_snr"
        report.confidence = 0.8
        report.fixes = [
            "apply_prebandpass_phase",
            "combine_more_fields",
            "increase_integration_time",
        ]
        if "issue" in snr_metrics:
            report.issues.append(snr_metrics["issue"])
        return report

    # Phase 3: Pattern-specific diagnosis
    if pattern == "channel_specific":
        report.root_cause = "rfi_contamination"
        report.confidence = 0.7
        report.fixes = ["rerun_aoflagger", "flag_rfi_channels", "inspect_rfi_mask"]
        report.warnings.append("Channel-specific flagging suggests RFI contamination")

    elif pattern == "spw_specific":
        report.root_cause = "spw_edge_effects"
        report.confidence = 0.6
        report.fixes = ["flag_edge_spws", "adjust_bandpass_parameters"]
        report.warnings.append("SPW-specific flagging suggests edge channel issues")

    elif pattern == "antenna_specific":
        report.root_cause = "bad_antenna_or_refant"
        report.confidence = 0.75
        report.fixes = ["change_refant", "flag_bad_antennas", "inspect_antenna_data"]
        report.warnings.append("Antenna-specific flagging suggests bad antenna or refant")

    elif pattern == "uniform":
        report.root_cause = "systematic_model_geometry_failure"
        report.confidence = 0.85
        report.fixes = [
            "verify_model_data",
            "rephase_to_calibrator",
            "check_calibrator_flux",
        ]
        report.issues.append("Uniform flagging suggests fundamental setup error")

    else:  # random
        report.root_cause = "low_snr_noise_limited"
        report.confidence = 0.65
        report.fixes = [
            "apply_prebandpass_phase",
            "combine_more_data",
            "check_integration_time",
        ]
        report.warnings.append("Random flagging pattern suggests noise/SNR limitation")

    return report


def generate_diagnostic_report(diagnosis: DiagnosticReport) -> str:
    """Generate human-readable diagnostic report.

    Parameters
    ----------
    diagnosis :
        DiagnosticReport object

    Returns
    -------
        Formatted report string

    """
    return str(diagnosis)


def _apply_recovery_fix(
    ms_path: str,
    cal_field: str,
    fix: str,
    refant: str,
    diagnostic_log: list[dict[str, Any]],
) -> bool:
    """Apply a single recovery fix based on diagnosis.

    Parameters
    ----------
    ms_path : str
        Path to Measurement Set.
    cal_field : str
        Field selection.
    fix : str
        Fix identifier from diagnosis.fixes.
    refant : str
        Reference antenna.
    diagnostic_log : list
        Log to append fix results to.

    Returns
    -------
    bool
        True if fix was applied successfully.
    """
    logger.info(f"Applying recovery fix: {fix}")

    try:
        if fix == "rephase_to_calibrator":
            # Re-run phase shift using CASA (not chgcentre)
            logger.info("  → Re-phasing to calibrator using CASA phaseshift...")
            # This would require importing and calling phaseshift_ms
            # For now, we log the recommendation
            diagnostic_log.append(
                {
                    "action": "rephase_to_calibrator",
                    "status": "recommended",
                    "message": "Run phaseshift_ms(..., use_chgcentre=False) and retry",
                }
            )
            return False  # Manual intervention needed

        elif fix == "apply_prebandpass_phase":
            # Solve and apply pre-bandpass phase
            logger.info("  → Solving pre-bandpass phase calibration...")
            from dsa110_contimg.core.calibration.calibration import solve_prebandpass_phase

            prebp_tables = solve_prebandpass_phase(
                ms=ms_path,
                cal_field=cal_field,
                refant=refant,
                table_prefix=f"{os.path.splitext(ms_path)[0]}_recovery",
            )
            diagnostic_log.append(
                {
                    "action": "apply_prebandpass_phase",
                    "status": "success",
                    "tables": prebp_tables,
                }
            )
            return True

        elif fix == "flag_bad_antennas":
            # Flag antennas with high flagging in previous solve
            logger.info("  → Flagging problematic antennas...")
            # This would analyze the caltable and flag worst antennas
            diagnostic_log.append(
                {
                    "action": "flag_bad_antennas",
                    "status": "recommended",
                    "message": "Inspect antenna flagging stats and flag worst performers",
                }
            )
            return False  # Manual intervention needed

        elif fix == "change_refant":
            # Try alternative reference antenna
            logger.info("  → Trying alternative reference antenna...")
            diagnostic_log.append(
                {
                    "action": "change_refant",
                    "status": "recommended",
                    "message": f"Current refant={refant}. Try alternative (e.g., 104, 105)",
                }
            )
            return False  # Manual intervention needed

        elif fix == "rerun_aoflagger":
            # Run more aggressive RFI flagging
            logger.info("  → Running aggressive RFI flagging...")
            try:
                from dsa110_contimg.core.calibration.flagging import flag_rfi_aoflagger

                flag_rfi_aoflagger(ms_path, datacolumn="data")
                diagnostic_log.append(
                    {
                        "action": "rerun_aoflagger",
                        "status": "success",
                    }
                )
                return True
            except Exception as e:
                diagnostic_log.append(
                    {
                        "action": "rerun_aoflagger",
                        "status": "failed",
                        "error": str(e),
                    }
                )
                return False

        elif fix == "verify_model_data":
            # Check and repopulate MODEL_DATA if needed
            logger.info("  → Verifying MODEL_DATA...")
            diagnostic_log.append(
                {
                    "action": "verify_model_data",
                    "status": "recommended",
                    "message": "Run populate_model_from_catalog() with correct calibrator",
                }
            )
            return False  # Manual intervention needed

        else:
            logger.warning(f"  Unknown fix: {fix}")
            diagnostic_log.append(
                {
                    "action": fix,
                    "status": "unknown",
                    "message": f"Fix '{fix}' not implemented in auto-recovery",
                }
            )
            return False

    except Exception as e:
        logger.error(f"  Fix '{fix}' failed: {e}")
        diagnostic_log.append(
            {
                "action": fix,
                "status": "exception",
                "error": str(e),
            }
        )
        return False


def _get_flag_fraction_from_table(caltable_path: str) -> float:
    """Extract flag fraction from a calibration table.

    Parameters
    ----------
    caltable_path : str
        Path to calibration table.

    Returns
    -------
    float
        Flag fraction (0.0 to 1.0).
    """
    with table(caltable_path, readonly=True, ack=False) as tb:
        if "FLAG" not in tb.colnames():
            return 0.0
        flags = tb.getcol("FLAG")
        return float(np.mean(flags))


def auto_recover_bandpass_calibration(
    ms_path: str,
    cal_field: str,
    calibrator_name: str,
    refant: str,
    table_prefix: str | None = None,
    prebandpass_phase_table: str | None = None,
    max_iterations: int = 3,
    target_flag_fraction: float = 0.05,
    auto_apply_fixes: bool = False,
) -> tuple[bool, str | None, list[dict[str, Any]]]:
    """Automated bandpass calibration recovery workflow.

    Iteratively diagnoses and fixes calibration issues until
    flagging < target_flag_fraction or max iterations reached.

    Recovery Strategy:
    1. Attempt bandpass solve with current settings
    2. If flagging exceeds target, diagnose root cause
    3. Apply recommended fix (if auto_apply_fixes=True)
    4. Retry bandpass solve
    5. Repeat until success or max_iterations

    Recovery actions by root cause:
    - geometric_phase_error: Re-phaseshift with CASA (not chgcentre)
    - insufficient_snr: Re-solve pre-bandpass phase with longer solint
    - bad_antenna_or_refant: Flag bad antennas, try alternate refant
    - rfi_contamination: Run aggressive RFI flagging

    Parameters
    ----------
    ms_path : str
        Path to Measurement Set.
    cal_field : str
        Field selection (e.g., "0~23").
    calibrator_name : str
        Calibrator name for model and diagnostics.
    refant : str
        Reference antenna.
    table_prefix : str or None, optional
        Prefix for output calibration tables. If None, uses ms_path stem.
    prebandpass_phase_table : str or None, optional
        Pre-bandpass phase table. If None, will attempt to solve one.
    max_iterations : int, optional
        Maximum fix-and-retry attempts. Default 3.
    target_flag_fraction : float, optional
        Target flag fraction to achieve. Default 0.05 (5%).
    auto_apply_fixes : bool, optional
        If True, automatically apply fixes. If False, only recommend.
        Default False (safer, requires manual intervention).

    Returns
    -------
    tuple[bool, str | None, list[dict[str, Any]]]
        (success, final_bpcal_table, diagnostic_log) tuple where:
        - success: True if target flag fraction achieved
        - final_bpcal_table: Path to final bandpass table (or None if failed)
        - diagnostic_log: List of diagnostic entries from each iteration

    Examples
    --------
    >>> success, bp_table, log = auto_recover_bandpass_calibration(
    ...     ms_path="/path/to/ms",
    ...     cal_field="0~23",
    ...     calibrator_name="3C454.3",
    ...     refant="103",
    ...     auto_apply_fixes=True,
    ... )
    >>> if success:
    ...     print(f"Recovery succeeded: {bp_table}")
    ... else:
    ...     print("Recovery failed. Review diagnostic log:")
    ...     for entry in log:
    ...         print(f"  {entry}")
    """
    logger.info("\n" + "=" * 70)
    logger.info("AUTOMATED BANDPASS CALIBRATION RECOVERY")
    logger.info("=" * 70)
    logger.info(f"MS: {ms_path}")
    logger.info(f"Calibrator: {calibrator_name}")
    logger.info(f"Field: {cal_field}")
    logger.info(f"Refant: {refant}")
    logger.info(f"Max iterations: {max_iterations}")
    logger.info(f"Target flag fraction: {target_flag_fraction * 100:.0f}%")
    logger.info(f"Auto-apply fixes: {auto_apply_fixes}")
    logger.info("=" * 70 + "\n")

    if table_prefix is None:
        table_prefix = os.path.splitext(ms_path)[0]

    diagnostic_log: list[dict[str, Any]] = []
    final_bp_table: str | None = None
    current_prebp = prebandpass_phase_table

    for iteration in range(max_iterations):
        logger.info(f"\n{'─' * 60}")
        logger.info(f"RECOVERY ITERATION {iteration + 1}/{max_iterations}")
        logger.info(f"{'─' * 60}\n")

        iter_prefix = f"{table_prefix}_recovery_iter{iteration}"
        bp_table = f"{iter_prefix}.b"

        # Step 1: Check if we need to solve pre-bandpass phase
        if current_prebp is None and iteration == 0:
            logger.info("No pre-bandpass phase table provided. Solving one first...")
            try:
                from dsa110_contimg.core.calibration.calibration import solve_prebandpass_phase

                prebp_tables = solve_prebandpass_phase(
                    ms=ms_path,
                    cal_field=cal_field,
                    refant=refant,
                    table_prefix=iter_prefix,
                )
                current_prebp = prebp_tables[0] if prebp_tables else None
                diagnostic_log.append(
                    {
                        "iteration": iteration,
                        "action": "solve_prebandpass_phase",
                        "result": "success",
                        "table": current_prebp,
                    }
                )
                logger.info(f"✓ Pre-bandpass phase solved: {current_prebp}")
            except Exception as e:
                diagnostic_log.append(
                    {
                        "iteration": iteration,
                        "action": "solve_prebandpass_phase",
                        "result": "failed",
                        "error": str(e),
                    }
                )
                logger.error(f"✗ Pre-bandpass phase solve failed: {e}")
                # Continue anyway, but quality will likely be poor

        # Step 2: Attempt bandpass solve
        logger.info("Attempting bandpass solve...")
        try:
            from dsa110_contimg.core.calibration.calibration import solve_bandpass

            tables = solve_bandpass(
                ms=ms_path,
                cal_field=cal_field,
                refant=refant,
                ktable=None,
                table_prefix=iter_prefix,
                prebandpass_phase_table=current_prebp,
                calibrator_name=calibrator_name,
                max_flag_fraction=0.99,  # Don't fail, we're recovering
                generate_diagnostics_report=False,  # Skip report during recovery
            )
            bp_table = tables[0]

            # Get flag fraction
            flag_fraction = _get_flag_fraction_from_table(bp_table)

            diagnostic_log.append(
                {
                    "iteration": iteration,
                    "action": "solve_bandpass",
                    "result": "success",
                    "table": bp_table,
                    "flag_fraction": flag_fraction,
                }
            )

            logger.info(f"✓ Bandpass solve completed: {bp_table}")
            logger.info(f"  Flag fraction: {flag_fraction * 100:.1f}%")

            # Check if we achieved target
            if flag_fraction < target_flag_fraction:
                logger.info(
                    f"\n✅ SUCCESS! Flag fraction {flag_fraction * 100:.1f}% "
                    f"< {target_flag_fraction * 100:.0f}% target"
                )
                final_bp_table = bp_table
                return True, final_bp_table, diagnostic_log

            # Update best result so far
            if final_bp_table is None:
                final_bp_table = bp_table

        except Exception as e:
            diagnostic_log.append(
                {
                    "iteration": iteration,
                    "action": "solve_bandpass",
                    "result": "exception",
                    "error": str(e),
                }
            )
            logger.error(f"✗ Bandpass solve failed: {e}")
            flag_fraction = 1.0  # Assume complete failure

        # Step 3: Diagnose root cause
        if flag_fraction >= target_flag_fraction:
            logger.info("\nRunning diagnostic analysis...")

            if os.path.exists(bp_table):
                try:
                    diagnosis = diagnose_bandpass_quality(
                        ms_path, cal_field, bp_table, calibrator_name, refant
                    )

                    diagnostic_log.append(
                        {
                            "iteration": iteration,
                            "action": "diagnose",
                            "root_cause": diagnosis.root_cause,
                            "confidence": diagnosis.confidence,
                            "severity": diagnosis.severity,
                            "fixes": diagnosis.fixes,
                        }
                    )

                    logger.info(f"  Root cause: {diagnosis.root_cause}")
                    logger.info(f"  Confidence: {diagnosis.confidence:.0%}")
                    logger.info(f"  Severity: {diagnosis.severity}")
                    logger.info(f"  Recommended fixes: {diagnosis.fixes}")

                    # Step 4: Apply fix if enabled
                    if auto_apply_fixes and diagnosis.fixes:
                        fix = diagnosis.fixes[0]
                        logger.info(f"\nApplying fix: {fix}")
                        fix_success = _apply_recovery_fix(
                            ms_path, cal_field, fix, refant, diagnostic_log
                        )
                        if fix_success:
                            logger.info(f"✓ Fix applied successfully")
                        else:
                            logger.warning(
                                f"⚠ Fix requires manual intervention. "
                                f"See diagnostic log for details."
                            )
                    elif not auto_apply_fixes and diagnosis.fixes:
                        logger.info(
                            f"\n⚠ Auto-apply disabled. Recommended fix: {diagnosis.fixes[0]}"
                        )
                        logger.info("  Set auto_apply_fixes=True to apply automatically")
                        diagnostic_log.append(
                            {
                                "iteration": iteration,
                                "action": "fix_recommended",
                                "fix": diagnosis.fixes[0],
                                "message": "Auto-apply disabled, manual intervention needed",
                            }
                        )

                except Exception as e:
                    logger.error(f"Diagnostic analysis failed: {e}")
                    diagnostic_log.append(
                        {
                            "iteration": iteration,
                            "action": "diagnose",
                            "result": "exception",
                            "error": str(e),
                        }
                    )

    # Max iterations reached
    logger.warning(
        f"\n❌ Recovery failed after {max_iterations} iterations. Best result: {final_bp_table}"
    )
    logger.warning("Review diagnostic log for details and apply manual fixes.")

    return False, final_bp_table, diagnostic_log


def get_recovery_recommendations(
    diagnosis: DiagnosticReport,
) -> list[dict[str, Any]]:
    """Get detailed recovery recommendations based on diagnosis.

    Converts diagnosis.fixes into detailed, actionable recommendations
    with specific commands and explanations.

    Parameters
    ----------
    diagnosis : DiagnosticReport
        Diagnostic report from diagnose_bandpass_quality.

    Returns
    -------
    list[dict[str, Any]]
        List of recommendation dicts with keys:
        - fix: str, the fix identifier
        - description: str, human-readable description
        - command: str, example command or code
        - priority: int, 1 = highest priority
    """
    recommendations = []

    fix_details = {
        "rephase_to_calibrator": {
            "description": (
                "Re-phaseshift data to calibrator position using CASA phaseshift. "
                "The chgcentre tool has known UVW convention issues with DSA-110."
            ),
            "command": (
                "from dsa110_contimg.core.calibration.runner import phaseshift_ms\n"
                "phaseshift_ms(ms_path, field='0~23', mode='calibrator', "
                "calibrator_name='...', use_chgcentre=False)"
            ),
        },
        "apply_prebandpass_phase": {
            "description": (
                "Solve and apply pre-bandpass phase calibration. "
                "This corrects phase drifts that cause decorrelation and low SNR."
            ),
            "command": (
                "from dsa110_contimg.core.calibration.calibration import "
                "solve_prebandpass_phase\n"
                "tables = solve_prebandpass_phase(ms, cal_field, refant, ...)"
            ),
        },
        "change_refant": {
            "description": (
                "Try a different reference antenna. The current refant may have "
                "unstable phases or hardware issues."
            ),
            "command": "refant='104'  # or '105', '106' - try outrigger antennas",
        },
        "flag_bad_antennas": {
            "description": (
                "Flag antennas with consistently poor solutions. "
                "Check per-antenna flagging stats in the diagnostic report."
            ),
            "command": "flagdata(vis=ms, mode='manual', antenna='<bad_ant_ids>')",
        },
        "rerun_aoflagger": {
            "description": (
                "Run more aggressive RFI flagging. Channel-specific flagging "
                "patterns suggest RFI contamination."
            ),
            "command": (
                "from dsa110_contimg.core.calibration.flagging import flag_rfi_aoflagger\n"
                "flag_rfi_aoflagger(ms_path, datacolumn='data')"
            ),
        },
        "verify_model_data": {
            "description": (
                "Verify MODEL_DATA is populated correctly. Zero or incorrect "
                "MODEL_DATA causes all solutions to be flagged."
            ),
            "command": (
                "from dsa110_contimg.core.calibration.model import "
                "populate_model_from_catalog\n"
                "populate_model_from_catalog(ms, field='0~23', "
                "calibrator_name='...')"
            ),
        },
        "combine_more_fields": {
            "description": (
                "Combine more fields to increase SNR. Use combine='scan,field' in bandpass solve."
            ),
            "command": "solve_bandpass(..., combine_fields=True)",
        },
        "verify_field_selection": {
            "description": (
                "Verify field selection includes all calibrator observations. "
                "Check that fields are coherently phased."
            ),
            "command": "# Check PHASE_DIR in FIELD subtable for all selected fields",
        },
        "check_calibrator_position": {
            "description": (
                "Verify calibrator position matches the VLA catalog. "
                "Wrong position causes phase errors."
            ),
            "command": (
                "from dsa110_contimg.core.calibration.catalogs import get_vla_calibrator\n"
                "cal = get_vla_calibrator('...')\n"
                "print(cal.ra_deg, cal.dec_deg)"
            ),
        },
    }

    for i, fix in enumerate(diagnosis.fixes, 1):
        details = fix_details.get(
            fix,
            {
                "description": f"Apply fix: {fix}",
                "command": f"# See documentation for {fix}",
            },
        )
        recommendations.append(
            {
                "fix": fix,
                "priority": i,
                "description": details["description"],
                "command": details["command"],
            }
        )

    return recommendations
