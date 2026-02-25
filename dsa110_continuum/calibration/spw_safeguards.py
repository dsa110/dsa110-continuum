"""
Per-SPW QA checks and safeguards to avoid blanket SPW drops.

This module provides:
- Per-SPW flag fraction and RFI metrics computation
- Decision logic for channel-level vs. SPW-level flagging
- SPW remapping strategies for isolated bad SPWs
- Integration with applycal for SPW substitution
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class SPWDecision(str, Enum):
    """Decision for a problematic SPW."""

    KEEP = "keep"  # Keep SPW, may have residual RFI
    REFLAG_CHANNELS = "reflag_channels"  # Re-flag specific channels
    REMAP = "remap"  # Remap to alternate SPW
    DROP = "drop"  # Flag entire SPW


@dataclass
class SPWQAMetrics:
    """Quality metrics for a single spectral window."""

    spw_id: int
    n_channels: int
    n_antennas: int

    # Flagging statistics
    flag_fraction: float = 0.0
    n_flagged_channels: int = 0
    flagged_channels: list[int] = field(default_factory=list)

    # Per-channel statistics
    per_channel_flag_fraction: np.ndarray = field(default_factory=lambda: np.array([]))
    per_channel_rms: np.ndarray = field(default_factory=lambda: np.array([]))
    per_channel_kurtosis: np.ndarray = field(default_factory=lambda: np.array([]))

    # RFI indicators
    max_kurtosis: float = 0.0
    median_kurtosis: float = 0.0
    max_flag_fraction_per_channel: float = 0.0

    # Status
    is_bad: bool = False
    is_marginal: bool = False


@dataclass
class SPWRemappingDecision:
    """Decision to remap a bad SPW to an alternate."""

    source_spw: int
    target_spw: int
    reason: str
    confidence: float  # 0.0-1.0, higher is better


@dataclass
class SPWSafeguardsResult:
    """Result from SPW safeguards analysis."""

    ms_path: str
    timestamp: float

    # Per-SPW analysis
    spw_metrics: dict[int, SPWQAMetrics]

    # Decisions
    spw_decisions: dict[int, SPWDecision]
    remapping_decisions: list[SPWRemappingDecision] = field(default_factory=list)
    spws_to_drop: list[int] = field(default_factory=list)

    # Summary
    total_spws: int = 0
    good_spws: int = 0
    marginal_spws: int = 0
    bad_spws: int = 0

    notes: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to serializable dictionary."""
        return {
            "ms_path": self.ms_path,
            "timestamp": self.timestamp,
            "spw_metrics": {
                str(spw): {
                    "spw_id": m.spw_id,
                    "n_channels": m.n_channels,
                    "flag_fraction": m.flag_fraction,
                    "n_flagged_channels": m.n_flagged_channels,
                    "max_kurtosis": m.max_kurtosis,
                    "median_kurtosis": m.median_kurtosis,
                    "is_bad": m.is_bad,
                    "is_marginal": m.is_marginal,
                }
                for spw, m in self.spw_metrics.items()
            },
            "spw_decisions": {str(spw): d.value for spw, d in self.spw_decisions.items()},
            "remapping_decisions": [
                {
                    "source_spw": r.source_spw,
                    "target_spw": r.target_spw,
                    "reason": r.reason,
                    "confidence": r.confidence,
                }
                for r in self.remapping_decisions
            ],
            "spws_to_drop": self.spws_to_drop,
            "summary": {
                "total_spws": self.total_spws,
                "good_spws": self.good_spws,
                "marginal_spws": self.marginal_spws,
                "bad_spws": self.bad_spws,
            },
            "notes": self.notes,
        }


# =============================================================================
# Per-SPW Metrics Computation
# =============================================================================


def _extract_spw_data(ms: str, spw_id: int) -> tuple[np.ndarray, np.ndarray]:
    """Extract DATA and FLAG arrays for a specific SPW.

    Parameters
    ----------

    Returns
    -------
        Tuple of (data, flags) arrays, shape (rows, channels, pols)

    """
    try:
        import casacore.tables as tb

        with tb.table(ms, readonly=True, ack=False) as t:
            # This is a simplified approach; in production might need to
            # select by DATA_DESC_ID which maps to SPW
            data = t.getcol("DATA")
            flags = t.getcol("FLAG")

            # Get DATA_DESC_ID column to map to SPWs
            if "DATA_DESC_ID" in t.colnames():
                data_desc = t.getcol("DATA_DESC_ID")
                spw_mask = data_desc == spw_id
                data = data[spw_mask]
                flags = flags[spw_mask]

        return data, flags
    except Exception as e:
        logger.warning(f"Failed to extract SPW {spw_id} data: {e}")
        return np.array([]), np.array([])


def compute_spw_qa_metrics(ms: str, spw_id: int) -> SPWQAMetrics:
    """Compute comprehensive QA metrics for a single SPW.

    Parameters
    ----------
    ms :
        Path to measurement set
    spw_id :
        Spectral window ID

    Returns
    -------
        SPWQAMetrics object with computed statistics

    """
    metrics = SPWQAMetrics(spw_id=spw_id, n_channels=0, n_antennas=0)

    try:
        import casacore.tables as tb

        # Get SPW metadata
        with tb.table(f"{ms}/SPECTRAL_WINDOW", readonly=True, ack=False) as sw:
            if spw_id < sw.nrows():
                n_chans = len(sw.getcell("CHAN_FREQ", spw_id))
                metrics.n_channels = n_chans

        # Get antenna count
        with tb.table(f"{ms}/ANTENNA", readonly=True, ack=False) as ant:
            metrics.n_antennas = ant.nrows()

        # Extract SPW data
        data, flags = _extract_spw_data(ms, spw_id)

        if data.size == 0:
            logger.warning(f"No data found for SPW {spw_id}")
            return metrics

        # Overall flagging
        metrics.flag_fraction = float(np.mean(flags))

        # Per-channel statistics
        n_rows, n_chans, n_pols = data.shape
        metrics.per_channel_flag_fraction = np.zeros(n_chans)
        metrics.per_channel_rms = np.zeros(n_chans)
        metrics.per_channel_kurtosis = np.zeros(n_chans)

        for chan in range(n_chans):
            chan_flags = flags[:, chan, :]
            chan_data = data[:, chan, :]

            # Channel-level flagging
            metrics.per_channel_flag_fraction[chan] = float(np.mean(chan_flags))
            if metrics.per_channel_flag_fraction[chan] > 0.95:
                metrics.flagged_channels.append(chan)

            # Statistics on unflagged data
            unflagged_chan = chan_data[~chan_flags]
            if len(unflagged_chan) > 0:
                metrics.per_channel_rms[chan] = float(np.std(np.abs(unflagged_chan)))

                # Kurtosis
                abs_vals = np.abs(unflagged_chan)
                mean_abs = np.mean(abs_vals)
                std_abs = np.std(abs_vals)
                if std_abs > 0:
                    kurtosis = np.mean((abs_vals - mean_abs) ** 4) / (std_abs**4) - 3
                    metrics.per_channel_kurtosis[chan] = float(kurtosis)

        # Summary statistics
        metrics.n_flagged_channels = len(metrics.flagged_channels)
        metrics.max_kurtosis = float(np.max(metrics.per_channel_kurtosis))
        metrics.median_kurtosis = float(np.median(metrics.per_channel_kurtosis))
        metrics.max_flag_fraction_per_channel = float(np.max(metrics.per_channel_flag_fraction))

    except Exception as e:
        logger.warning(f"Failed to compute SPW {spw_id} metrics: {e}")

    return metrics


def compute_all_spws_qa(ms: str) -> dict[int, SPWQAMetrics]:
    """Compute QA metrics for all SPWs in MS.

    Parameters
    ----------
    """
    metrics = {}

    try:
        import casacore.tables as tb

        with tb.table(f"{ms}/SPECTRAL_WINDOW", readonly=True, ack=False) as sw:
            n_spws = sw.nrows()

        logger.info(f"Computing QA metrics for {n_spws} SPWs...")

        for spw_id in range(n_spws):
            metrics[spw_id] = compute_spw_qa_metrics(ms, spw_id)
            logger.debug(
                f"SPW {spw_id}: {metrics[spw_id].flag_fraction:.1%} flagged, "
                f"kurtosis={metrics[spw_id].median_kurtosis:.2f}"
            )

    except Exception as e:
        logger.warning(f"Failed to compute all SPWs QA: {e}")

    return metrics


# =============================================================================
# SPW Classification and Decision Logic
# =============================================================================


@dataclass
class SPWThresholds:
    """Thresholds for SPW classification."""

    good_max_flag_frac: float = 0.15  # <=15% -> good
    marginal_max_flag_frac: float = 0.50  # 15-50% -> marginal
    bad_max_flag_frac: float = 0.50  # >50% -> bad

    good_max_kurtosis: float = 3.0  # Normal kurtosis
    bad_max_kurtosis: float = 5.0  # High kurtosis -> bad

    min_good_channels_per_spw: float = 0.7  # >=70% channels good -> keep


def classify_spws(
    spw_metrics: dict[int, SPWQAMetrics],
    thresholds: SPWThresholds | None = None,
) -> dict[int, SPWDecision]:
    """Classify each SPW and decide on action.

    Decision logic:
    - If >70% of channels are good AND overall flag fraction <50%: KEEP
    - If >70% of channels are good but >50% flagged: REFLAG_CHANNELS for bad ones
    - If <50% of channels good AND can find alternate: REMAP
    - If no good alternate: DROP

    Parameters
    ----------
    spw_metrics: Dict[int :

    SPWQAMetrics] :

    thresholds: Optional[SPWThresholds] :
         (Default value = None)

    """
    thresholds = thresholds or SPWThresholds()
    decisions = {}

    for spw_id, metrics in spw_metrics.items():
        # Classify the SPW itself
        if metrics.flag_fraction <= thresholds.good_max_flag_frac:
            metrics.is_bad = False
            metrics.is_marginal = False
        elif metrics.flag_fraction <= thresholds.marginal_max_flag_frac:
            metrics.is_bad = False
            metrics.is_marginal = True
        else:
            metrics.is_bad = True
            metrics.is_marginal = False

        # Classify based on per-channel health
        if metrics.n_channels > 0:
            good_channels = sum(
                1
                for frac in metrics.per_channel_flag_fraction
                if frac <= thresholds.good_max_flag_frac
            )
            frac_good_channels = good_channels / metrics.n_channels
        else:
            frac_good_channels = 0.0

        # Decision logic
        if frac_good_channels >= thresholds.min_good_channels_per_spw:
            if metrics.flag_fraction > thresholds.marginal_max_flag_frac:
                # Some channels are bad; reflag them specifically
                decisions[spw_id] = SPWDecision.REFLAG_CHANNELS
            else:
                # Mostly good
                decisions[spw_id] = SPWDecision.KEEP
        else:
            # Too many bad channels
            if metrics.is_bad:
                # Consider remap or drop
                decisions[spw_id] = SPWDecision.REMAP
            else:
                decisions[spw_id] = SPWDecision.KEEP

        logger.info(
            f"SPW {spw_id}: Decision={decisions[spw_id].value}, "
            f"flag={metrics.flag_fraction:.1%}, "
            f"good_chans={frac_good_channels:.1%}"
        )

    return decisions


# =============================================================================
# Per-Channel Re-flagging Logic
# =============================================================================


def reflag_bad_channels(
    ms: str,
    spw_ids: list[int],
    spw_metrics: dict[int, SPWQAMetrics],
    thresholds: SPWThresholds,
) -> dict[str, Any]:
    """Re-flag specific bad channels within marginal SPWs.

    This performs targeted flagging on channels that exceed quality thresholds
    while preserving good channels in the same SPW.

    Parameters
    ----------
    ms :
        Path to measurement set
    spw_ids :
        List of SPW IDs to re-flag
    spw_metrics :
        Pre-computed QA metrics for each SPW
    thresholds :
        Classification thresholds

    Returns
    -------
        Dictionary with re-flagging results per SPW

    """
    results: dict[str, Any] = {
        "spws_processed": [],
        "channels_flagged": {},
        "total_new_flags": 0,
        "errors": [],
    }

    try:
        import casacore.tables as tb  # noqa: F401 - used in _flag_channels_in_spw
    except ImportError:
        logger.warning("casacore not available - skipping per-channel re-flagging")
        results["errors"].append("casacore not available")
        return results

    for spw_id in spw_ids:
        if spw_id not in spw_metrics:
            logger.warning(f"No metrics available for SPW {spw_id}")
            continue

        metrics = spw_metrics[spw_id]
        bad_channels = _identify_bad_channels(metrics, thresholds)

        if not bad_channels:
            logger.info(f"SPW {spw_id}: No individual bad channels identified")
            continue

        logger.info(
            f"SPW {spw_id}: Flagging {len(bad_channels)} bad channels: {bad_channels[:10]}..."
        )

        try:
            n_flagged = _flag_channels_in_spw(ms, spw_id, bad_channels)
            results["spws_processed"].append(spw_id)
            results["channels_flagged"][str(spw_id)] = bad_channels
            results["total_new_flags"] += n_flagged
            logger.info(
                f"SPW {spw_id}: Flagged {n_flagged} new visibilities in {len(bad_channels)} channels"
            )
        except Exception as e:
            error_msg = f"SPW {spw_id}: Failed to flag channels: {e}"
            logger.error(error_msg)
            results["errors"].append(error_msg)

    return results


def _identify_bad_channels(
    metrics: SPWQAMetrics,
    thresholds: SPWThresholds,
) -> list[int]:
    """Identify which channels in an SPW should be flagged.

    Uses multiple criteria:
    - High per-channel flag fraction (already mostly flagged)
    - High kurtosis (non-Gaussian, likely RFI)
    - High RMS compared to median

    Parameters
    ----------
    metrics :
        QA metrics for the SPW
    thresholds :
        Classification thresholds

    Returns
    -------
        List of channel indices to flag

    """
    bad_channels = []

    if len(metrics.per_channel_flag_fraction) == 0:
        return bad_channels

    # Compute median RMS for comparison
    valid_rms = metrics.per_channel_rms[metrics.per_channel_rms > 0]
    median_rms = float(np.median(valid_rms)) if len(valid_rms) > 0 else 0.0

    for chan in range(metrics.n_channels):
        is_bad = False
        reasons = []

        # Criterion 1: Already heavily flagged (>80% flagged)
        if chan < len(metrics.per_channel_flag_fraction):
            flag_frac = metrics.per_channel_flag_fraction[chan]
            if flag_frac > 0.8:
                is_bad = True
                reasons.append(f"high_flag_frac={flag_frac:.2f}")

        # Criterion 2: High kurtosis (>5 indicates impulsive RFI)
        if chan < len(metrics.per_channel_kurtosis):
            kurtosis = metrics.per_channel_kurtosis[chan]
            if kurtosis > thresholds.bad_max_kurtosis:
                is_bad = True
                reasons.append(f"high_kurtosis={kurtosis:.2f}")

        # Criterion 3: Anomalously high RMS (>3x median)
        if chan < len(metrics.per_channel_rms) and median_rms > 0:
            rms = metrics.per_channel_rms[chan]
            if rms > 3.0 * median_rms:
                is_bad = True
                reasons.append(f"high_rms={rms:.2f}")

        # Criterion 4: Moderate flagging + moderate kurtosis (compound indicator)
        if not is_bad and chan < len(metrics.per_channel_flag_fraction):
            flag_frac = metrics.per_channel_flag_fraction[chan]
            kurtosis = (
                metrics.per_channel_kurtosis[chan]
                if chan < len(metrics.per_channel_kurtosis)
                else 0.0
            )
            if flag_frac > 0.5 and kurtosis > thresholds.good_max_kurtosis:
                is_bad = True
                reasons.append(f"compound: flag={flag_frac:.2f}, kurt={kurtosis:.2f}")

        if is_bad:
            bad_channels.append(chan)

    return bad_channels


def _flag_channels_in_spw(
    ms: str,
    spw_id: int,
    channels: list[int],
) -> int:
    """Flag specific channels within an SPW in the measurement set.

    Parameters
    ----------
    ms :
        Path to measurement set
    spw_id :
        Spectral window ID
    channels :
        List of channel indices to flag

    Returns
    -------
        Number of visibilities newly flagged

    """
    import casacore.tables as tb

    n_flagged = 0

    # Open main table and find rows for this SPW
    with tb.table(ms, readonly=False, ack=False) as main_table:  # noqa: F841 - used in taql
        # Get DATA_DESC_ID for this SPW
        with tb.table(f"{ms}/DATA_DESCRIPTION", readonly=True, ack=False) as dd:
            dd_spws = dd.getcol("SPECTRAL_WINDOW_ID")
            dd_ids = [i for i, s in enumerate(dd_spws) if s == spw_id]

        if not dd_ids:
            logger.warning(f"No DATA_DESC_ID found for SPW {spw_id}")
            return 0

        # Query rows for this SPW
        dd_query = " OR ".join([f"DATA_DESC_ID=={dd_id}" for dd_id in dd_ids])

        with tb.taql(f"SELECT FROM $main_table WHERE {dd_query}") as selection:
            n_rows = selection.nrows()
            if n_rows == 0:
                logger.debug(f"No rows for SPW {spw_id}")
                return 0

            # Get FLAG column
            flags = selection.getcol("FLAG")  # shape: (nrows, nchan, npol)

            # Flag specified channels
            channels_array = np.array(channels)
            valid_channels = channels_array[channels_array < flags.shape[1]]

            if len(valid_channels) == 0:
                return 0

            # Count new flags
            existing_flags = flags[:, valid_channels, :].sum()
            flags[:, valid_channels, :] = True
            new_flags = flags[:, valid_channels, :].sum()
            n_flagged = int(new_flags - existing_flags)

            # Write back
            selection.putcol("FLAG", flags)

    return n_flagged


# =============================================================================
# SPW Remapping Logic
# =============================================================================


def find_alternate_spw(
    problem_spw: int,
    spw_metrics: dict[int, SPWQAMetrics],
    min_quality_threshold: float = 0.8,
) -> int | None:
    """Find an alternate SPW to use if the primary one is bad.

    Parameters
    ----------

    Returns
    -------
        Alternate SPW ID if found, else None

    """
    candidates = []

    for spw_id, metrics in spw_metrics.items():
        if spw_id == problem_spw:
            continue

        # Candidate must be "good" quality
        if metrics.flag_fraction <= (1.0 - min_quality_threshold):
            candidates.append((spw_id, metrics.flag_fraction))

    if not candidates:
        return None

    # Return the best candidate (lowest flag fraction)
    candidates.sort(key=lambda x: x[1])
    return candidates[0][0]


def plan_spw_remappings(
    problem_spws: list[int],
    spw_metrics: dict[int, SPWQAMetrics],
) -> list[SPWRemappingDecision]:
    """Plan remappings for problem SPWs.

    Parameters
    ----------
    problem_spws: List[int] :

    spw_metrics: Dict[int :

    SPWQAMetrics] :


    Returns
    -------
        List of SPWRemappingDecision objects

    """
    remappings = []

    for problem_spw in problem_spws:
        alternate = find_alternate_spw(problem_spw, spw_metrics)

        if alternate is not None:
            decision = SPWRemappingDecision(
                source_spw=problem_spw,
                target_spw=alternate,
                reason=f"Bad SPW {problem_spw} remapped to {alternate}",
                confidence=0.8,  # Can be improved with heuristics
            )
            remappings.append(decision)
            logger.info(
                f"Plan remap: SPW {problem_spw} (flag={spw_metrics[problem_spw].flag_fraction:.1%}) "
                f"-> SPW {alternate} (flag={spw_metrics[alternate].flag_fraction:.1%})"
            )
        else:
            logger.warning(f"No alternate SPW found for problem SPW {problem_spw}; will drop")

    return remappings


# =============================================================================
# Main SPW Safeguards Function
# =============================================================================


def apply_spw_safeguards(
    ms: str,
    *,
    thresholds: SPWThresholds | None = None,
    enable_reflag: bool = True,
    enable_remap: bool = True,
    output_dir: str | None = None,
) -> SPWSafeguardsResult:
    """Apply per-SPW safeguards: compute metrics, classify SPWs, and plan actions.

    Parameters
    ----------
    ms :
        Path to measurement set
    thresholds :
        SPW classification thresholds
    enable_reflag :
        Whether to re-flag specific bad channels
    enable_remap :
        Whether to plan SPW remappings
    output_dir :
        Directory for logging (default: same as MS directory)

    Returns
    -------
        SPWSafeguardsResult with decisions and recommendations

    """
    import time

    output_dir = Path(output_dir or ms)
    output_dir.mkdir(parents=True, exist_ok=True)

    result = SPWSafeguardsResult(
        ms_path=ms,
        timestamp=time.time(),
        spw_metrics={},
        spw_decisions={},
    )

    logger.info(f"Applying SPW safeguards for {ms}")

    # Compute metrics for all SPWs
    result.spw_metrics = compute_all_spws_qa(ms)
    result.total_spws = len(result.spw_metrics)

    # Classify SPWs
    thresholds = thresholds or SPWThresholds()
    result.spw_decisions = classify_spws(result.spw_metrics, thresholds)

    # Count classifications
    result.good_spws = sum(
        1 for m in result.spw_metrics.values() if not m.is_bad and not m.is_marginal
    )
    result.marginal_spws = sum(1 for m in result.spw_metrics.values() if m.is_marginal)
    result.bad_spws = sum(1 for m in result.spw_metrics.values() if m.is_bad)

    logger.info(
        f"SPW Classification: {result.good_spws} good, {result.marginal_spws} marginal, "
        f"{result.bad_spws} bad"
    )

    # Plan re-flagging for marginal SPWs
    if enable_reflag:
        marginal_spws = [
            spw_id
            for spw_id, decision in result.spw_decisions.items()
            if decision == SPWDecision.REFLAG_CHANNELS
        ]

        if marginal_spws:
            logger.info(f"Planning re-flagging for SPWs: {marginal_spws}")
            # Perform per-channel re-flagging for marginal SPWs
            reflag_results = reflag_bad_channels(
                ms,
                marginal_spws,
                result.spw_metrics,
                thresholds,
            )
            result.notes["reflag_results"] = reflag_results

    # Plan remappings for bad SPWs
    if enable_remap:
        bad_spws = [
            spw_id
            for spw_id, decision in result.spw_decisions.items()
            if decision == SPWDecision.REMAP
        ]

        if bad_spws:
            result.remapping_decisions = plan_spw_remappings(bad_spws, result.spw_metrics)
            result.spws_to_drop = [
                spw_id
                for spw_id in bad_spws
                if spw_id not in [r.source_spw for r in result.remapping_decisions]
            ]

    # Save report
    report_file = output_dir / "spw_safeguards_report.json"
    try:
        with open(report_file, "w") as f:
            json.dump(result.to_dict(), f, indent=2)
        logger.info(f"Saved SPW safeguards report to {report_file}")
    except Exception as e:
        logger.warning(f"Failed to save SPW safeguards report: {e}")

    return result
