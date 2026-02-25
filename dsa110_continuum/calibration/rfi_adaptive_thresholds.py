"""
Adaptive RFI Threshold Tracker with Rolling Baseline.

Tracks RFI statistics over time to enable adaptive threshold calculation
rather than using static thresholds. Uses ring buffers to maintain
rolling statistics that adapt to changing RFI environment conditions.

Inspired by Chimbuko's in-situ anomaly detection concepts:

- Rolling baseline computation
- Adaptive threshold based on recent history
- Per-antenna/channel tracking

Notes
-----
Key Features:

1. Rolling MAD/kurtosis baseline per antenna and per channel
2. Adaptive thresholds: tighten during quiet periods, loosen during noisy
3. Environment drift detection (new interference sources)
4. Per-SPW RFI condition tracking

Examples
--------
>>> from dsa110_contimg.core.calibration.rfi_adaptive_thresholds import (
...     RFIThresholdTracker,
...     get_global_rfi_tracker,
... )
>>> tracker = get_global_rfi_tracker()
>>> # Update with observation statistics
>>> tracker.update_observation(
...     observation_mjd=60000.0,
...     per_spw_stats={0: {"mad": 2.5, "kurtosis": 3.2, "flag_fraction": 0.05}},
... )
>>> # Get adaptive thresholds
>>> thresholds = tracker.get_adaptive_thresholds(spw=0)
>>> print(f"MAD threshold: {thresholds.mad_threshold}")
"""

from __future__ import annotations

import logging
import sqlite3
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Any

import numpy as np

from dsa110_contimg.common.unified_config import settings

logger = logging.getLogger(__name__)

# Default database path for persistence
DEFAULT_DB_PATH = settings.paths.pipeline_db

# Configuration constants
DEFAULT_HISTORY_SIZE = 100  # Number of observations to track
MIN_SAMPLES_FOR_ADAPTIVE = 10  # Minimum samples before using adaptive thresholds
BASELINE_PERCENTILE_LOW = 25  # Percentile for baseline lower bound
BASELINE_PERCENTILE_HIGH = 75  # Percentile for baseline upper bound
OUTLIER_SIGMA = 3.0  # Standard deviations for outlier detection

# Static fallback thresholds (used when insufficient history)
DEFAULT_MAD_THRESHOLD = 3.0
DEFAULT_KURTOSIS_THRESHOLD = 5.0
DEFAULT_FLAG_FRACTION_THRESHOLD = 0.3


@dataclass
class RFIObservationStats:
    """RFI statistics for a single observation."""

    observation_mjd: float
    spw: int
    antenna_id: int | None = None  # None = array-wide
    channel: int | None = None  # None = full SPW

    # Statistics
    mad: float = 0.0  # Median Absolute Deviation
    kurtosis: float = 0.0  # Excess kurtosis
    rms: float = 0.0
    flag_fraction: float = 0.0
    max_amplitude: float = 0.0

    # Metadata
    recorded_at: float = field(default_factory=time.time)


@dataclass
class AdaptiveThresholds:
    """Adaptive thresholds computed from rolling history."""

    # Thresholds
    mad_threshold: float
    kurtosis_threshold: float
    flag_fraction_threshold: float

    # Baseline statistics (for context)
    baseline_mad_mean: float
    baseline_mad_std: float
    baseline_kurtosis_mean: float
    baseline_kurtosis_std: float

    # Confidence
    n_samples: int
    is_adaptive: bool  # True if using adaptive, False if using static fallback

    # Environment condition
    environment_condition: str  # "quiet", "moderate", "noisy"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "mad_threshold": self.mad_threshold,
            "kurtosis_threshold": self.kurtosis_threshold,
            "flag_fraction_threshold": self.flag_fraction_threshold,
            "baseline_mad_mean": self.baseline_mad_mean,
            "baseline_mad_std": self.baseline_mad_std,
            "baseline_kurtosis_mean": self.baseline_kurtosis_mean,
            "baseline_kurtosis_std": self.baseline_kurtosis_std,
            "n_samples": self.n_samples,
            "is_adaptive": self.is_adaptive,
            "environment_condition": self.environment_condition,
        }


@dataclass
class RFIEnvironmentChange:
    """Detected change in RFI environment."""

    detection_time: float
    spw: int
    metric: str  # "mad", "kurtosis", etc.
    old_baseline: float
    new_value: float
    sigma_deviation: float
    message: str


class RFIThresholdTracker:
    """Track RFI statistics and compute adaptive thresholds.

    Uses ring buffers to maintain per-SPW rolling history of RFI metrics,
    enabling adaptive threshold calculation based on recent conditions.

    Thread-safe for use in concurrent pipeline operations.

    """

    def __init__(
        self,
        history_size: int = DEFAULT_HISTORY_SIZE,
        db_path: Path | None = None,
        persist: bool = True,
        sigma_multiplier: float = 3.0,
    ):
        """Initialize the tracker.

        Parameters
        ----------
        history_size : int
            Maximum observations to track per SPW
        db_path : str
            Path to SQLite database for persistence
        persist : bool
            Whether to persist history to database
        sigma_multiplier : float
            Number of sigmas above mean for threshold
        """
        self.history_size = history_size
        self.db_path = db_path or DEFAULT_DB_PATH
        self.persist = persist
        self.sigma_multiplier = sigma_multiplier
        self._lock = Lock()

        # Per-SPW ring buffers: spw -> deque of RFIObservationStats
        self._spw_history: dict[int, deque[RFIObservationStats]] = defaultdict(
            lambda: deque(maxlen=history_size)
        )

        # Per-antenna per-SPW history (more granular)
        # (spw, antenna_id) -> deque of RFIObservationStats
        self._antenna_spw_history: dict[tuple[int, int], deque[RFIObservationStats]] = defaultdict(
            lambda: deque(maxlen=history_size)
        )

        # Environment change detection
        self._environment_changes: deque[RFIEnvironmentChange] = deque(maxlen=100)

        # Initialize database if persisting
        if self.persist:
            self._init_db()
            self._load_from_db()

    def _init_db(self) -> None:
        """Create database tables for persistence."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS rfi_threshold_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    observation_mjd REAL NOT NULL,
                    spw INTEGER NOT NULL,
                    antenna_id INTEGER,
                    channel INTEGER,
                    mad REAL NOT NULL,
                    kurtosis REAL NOT NULL,
                    rms REAL NOT NULL,
                    flag_fraction REAL NOT NULL,
                    max_amplitude REAL NOT NULL,
                    recorded_at REAL NOT NULL,
                    UNIQUE(observation_mjd, spw, antenna_id, channel)
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_rfi_history_spw
                ON rfi_threshold_history(spw)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_rfi_history_mjd
                ON rfi_threshold_history(observation_mjd)
            """)
            conn.commit()

    def _load_from_db(self) -> None:
        """Load recent history from database."""
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                cursor = conn.execute(
                    """
                    SELECT observation_mjd, spw, antenna_id, channel,
                           mad, kurtosis, rms, flag_fraction, max_amplitude, recorded_at
                    FROM rfi_threshold_history
                    ORDER BY observation_mjd DESC
                    LIMIT ?
                """,
                    (self.history_size * 16,),
                )  # Assume max 16 SPWs

                for row in cursor:
                    stats = RFIObservationStats(
                        observation_mjd=row[0],
                        spw=row[1],
                        antenna_id=row[2],
                        channel=row[3],
                        mad=row[4],
                        kurtosis=row[5],
                        rms=row[6],
                        flag_fraction=row[7],
                        max_amplitude=row[8],
                        recorded_at=row[9],
                    )
                    self._spw_history[stats.spw].append(stats)
                    if stats.antenna_id is not None:
                        self._antenna_spw_history[(stats.spw, stats.antenna_id)].append(stats)

                logger.info(f"Loaded RFI threshold history: {len(self._spw_history)} SPWs")
        except Exception as e:
            logger.warning(f"Could not load RFI threshold history: {e}")

    def _save_stats(self, stats: RFIObservationStats) -> None:
        """Save statistics to database.

        Parameters
        ----------
        stats : RFIObservationStats
            Statistics to save

        """
        if not self.persist:
            return
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO rfi_threshold_history
                    (observation_mjd, spw, antenna_id, channel,
                     mad, kurtosis, rms, flag_fraction, max_amplitude, recorded_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        stats.observation_mjd,
                        stats.spw,
                        stats.antenna_id,
                        stats.channel,
                        stats.mad,
                        stats.kurtosis,
                        stats.rms,
                        stats.flag_fraction,
                        stats.max_amplitude,
                        stats.recorded_at,
                    ),
                )
                conn.commit()
        except Exception as e:
            logger.warning(f"Could not save RFI stats: {e}")

    def update_observation(
        self,
        observation_mjd: float,
        per_spw_stats: dict[int, dict[str, float]],
        per_antenna_stats: dict[tuple[int, int], dict[str, float]] | None = None,
    ) -> list[RFIEnvironmentChange]:
        """Update tracker with new observation statistics.

        Parameters
        ----------
        observation_mjd : float
            Observation MJD
        per_spw_stats : Dict[int, Dict[str, float]]
            Dictionary mapping SPW to statistics dict with keys:
            "mad", "kurtosis", "rms", "flag_fraction", "max_amplitude"
        per_antenna_stats : Optional[Dict[Tuple[int, int], float]]
            Optional per-antenna stats keyed by (spw, antenna_id), default is None

        """
        changes = []

        with self._lock:
            # Process per-SPW stats
            for spw, stats_dict in per_spw_stats.items():
                stats = RFIObservationStats(
                    observation_mjd=observation_mjd,
                    spw=spw,
                    mad=stats_dict.get("mad", 0.0),
                    kurtosis=stats_dict.get("kurtosis", 0.0),
                    rms=stats_dict.get("rms", 0.0),
                    flag_fraction=stats_dict.get("flag_fraction", 0.0),
                    max_amplitude=stats_dict.get("max_amplitude", 0.0),
                )

                # Check for environment change before updating
                change = self._check_environment_change(spw, stats)
                if change:
                    changes.append(change)
                    self._environment_changes.append(change)

                # Update ring buffer
                self._spw_history[spw].append(stats)

                # Persist
                self._save_stats(stats)

            # Process per-antenna stats if provided
            if per_antenna_stats:
                for (spw, antenna_id), stats_dict in per_antenna_stats.items():
                    stats = RFIObservationStats(
                        observation_mjd=observation_mjd,
                        spw=spw,
                        antenna_id=antenna_id,
                        mad=stats_dict.get("mad", 0.0),
                        kurtosis=stats_dict.get("kurtosis", 0.0),
                        rms=stats_dict.get("rms", 0.0),
                        flag_fraction=stats_dict.get("flag_fraction", 0.0),
                        max_amplitude=stats_dict.get("max_amplitude", 0.0),
                    )
                    self._antenna_spw_history[(spw, antenna_id)].append(stats)
                    self._save_stats(stats)

        return changes

    def _check_environment_change(
        self, spw: int, new_stats: RFIObservationStats
    ) -> RFIEnvironmentChange | None:
        """Check if new stats indicate an environment change.

        Parameters
        ----------
        spw : int
            Spectral window index
        new_stats : RFIObservationStats
            New observation statistics to compare

        """
        history = self._spw_history.get(spw)
        if not history or len(history) < MIN_SAMPLES_FOR_ADAPTIVE:
            return None

        # Compute baseline from history (excluding most recent if any)
        mad_values = np.array([s.mad for s in history])
        baseline_mean = np.mean(mad_values)
        baseline_std = np.std(mad_values)

        if baseline_std < 1e-10:
            return None

        # Check if new value is an outlier
        deviation = abs(new_stats.mad - baseline_mean) / baseline_std

        if deviation > OUTLIER_SIGMA:
            direction = "increased" if new_stats.mad > baseline_mean else "decreased"
            return RFIEnvironmentChange(
                detection_time=time.time(),
                spw=spw,
                metric="mad",
                old_baseline=baseline_mean,
                new_value=new_stats.mad,
                sigma_deviation=deviation,
                message=(
                    f"SPW {spw}: MAD {direction} from {baseline_mean:.2f} to "
                    f"{new_stats.mad:.2f} ({deviation:.1f}σ deviation)"
                ),
            )

        return None

    def get_adaptive_thresholds(
        self,
        spw: int,
        antenna_id: int | None = None,
    ) -> AdaptiveThresholds:
        """Get adaptive thresholds for a specific SPW (and optionally antenna).

        Parameters
        ----------
        spw : int
            Spectral window index
        antenna_id : Optional[int]
            Optional antenna ID for per-antenna thresholds, default is None

        """
        with self._lock:
            return self._get_adaptive_thresholds_unlocked(spw, antenna_id)

    def _get_adaptive_thresholds_unlocked(
        self,
        spw: int,
        antenna_id: int | None = None,
    ) -> AdaptiveThresholds:
        """Internal method without lock (caller must hold lock).

        Parameters
        ----------
        spw : int
            Spectral window index
        antenna_id : Optional[int]
            Optional antenna ID for per-antenna thresholds, default is None

        """
        # Get relevant history
        if antenna_id is not None:
            history = list(self._antenna_spw_history.get((spw, antenna_id), []))
        else:
            history = list(self._spw_history.get(spw, []))

        n_samples = len(history)

        # Check if we have enough samples for adaptive thresholds
        if n_samples < MIN_SAMPLES_FOR_ADAPTIVE:
            # Return static fallback thresholds
            return AdaptiveThresholds(
                mad_threshold=DEFAULT_MAD_THRESHOLD,
                kurtosis_threshold=DEFAULT_KURTOSIS_THRESHOLD,
                flag_fraction_threshold=DEFAULT_FLAG_FRACTION_THRESHOLD,
                baseline_mad_mean=DEFAULT_MAD_THRESHOLD,
                baseline_mad_std=0.0,
                baseline_kurtosis_mean=DEFAULT_KURTOSIS_THRESHOLD,
                baseline_kurtosis_std=0.0,
                n_samples=n_samples,
                is_adaptive=False,
                environment_condition="unknown",
            )

        # Compute baseline statistics
        mad_values = np.array([s.mad for s in history])
        kurtosis_values = np.array([s.kurtosis for s in history])
        flag_fractions = np.array([s.flag_fraction for s in history])

        mad_mean = float(np.mean(mad_values))
        mad_std = float(np.std(mad_values))
        kurtosis_mean = float(np.mean(kurtosis_values))
        kurtosis_std = float(np.std(kurtosis_values))
        flag_mean = float(np.mean(flag_fractions))

        # Compute adaptive thresholds: baseline + sigma_multiplier * std
        # But ensure they're at least as strict as defaults
        mad_threshold = max(
            mad_mean + self.sigma_multiplier * mad_std,
            DEFAULT_MAD_THRESHOLD * 0.5,  # Don't go below half the default
        )
        kurtosis_threshold = max(
            kurtosis_mean + self.sigma_multiplier * kurtosis_std,
            DEFAULT_KURTOSIS_THRESHOLD * 0.5,
        )
        flag_threshold = min(
            flag_mean + self.sigma_multiplier * np.std(flag_fractions),
            DEFAULT_FLAG_FRACTION_THRESHOLD,  # Don't exceed default
        )

        # Determine environment condition based on baseline
        if mad_mean < 1.5:
            condition = "quiet"
        elif mad_mean < 2.5:
            condition = "moderate"
        else:
            condition = "noisy"

        return AdaptiveThresholds(
            mad_threshold=mad_threshold,
            kurtosis_threshold=kurtosis_threshold,
            flag_fraction_threshold=flag_threshold,
            baseline_mad_mean=mad_mean,
            baseline_mad_std=mad_std,
            baseline_kurtosis_mean=kurtosis_mean,
            baseline_kurtosis_std=kurtosis_std,
            n_samples=n_samples,
            is_adaptive=True,
            environment_condition=condition,
        )

    def get_all_spw_thresholds(self) -> dict[int, AdaptiveThresholds]:
        """Get adaptive thresholds for all tracked SPWs."""
        with self._lock:
            return {
                spw: self._get_adaptive_thresholds_unlocked(spw) for spw in self._spw_history.keys()
            }

    def get_environment_changes(self, since_mjd: float | None = None) -> list[RFIEnvironmentChange]:
        """Get detected environment changes.

        Parameters
        ----------
        since_mjd :
            Only return changes after this MJD (optional)
        since_mjd : Optional[float] :
            (Default value = None)
        since_mjd: Optional[float] :
             (Default value = None)

        """
        with self._lock:
            changes = list(self._environment_changes)

        if since_mjd is not None:
            # Convert MJD to Unix time for comparison
            # MJD 0 = Unix time -3506716800
            cutoff = (since_mjd - 40587) * 86400
            changes = [c for c in changes if c.detection_time >= cutoff]

        return changes

    def get_spw_summary(self, spw: int) -> dict[str, Any]:
        """Get summary statistics for a specific SPW.

        Parameters
        ----------
        spw : int
            Spectral window index

        """
        with self._lock:
            history = list(self._spw_history.get(spw, []))

        if not history:
            return {"spw": spw, "n_observations": 0}

        mad_values = [s.mad for s in history]
        kurtosis_values = [s.kurtosis for s in history]
        flag_fractions = [s.flag_fraction for s in history]

        return {
            "spw": spw,
            "n_observations": len(history),
            "oldest_mjd": min(s.observation_mjd for s in history),
            "newest_mjd": max(s.observation_mjd for s in history),
            "mad": {
                "mean": float(np.mean(mad_values)),
                "std": float(np.std(mad_values)),
                "min": float(np.min(mad_values)),
                "max": float(np.max(mad_values)),
            },
            "kurtosis": {
                "mean": float(np.mean(kurtosis_values)),
                "std": float(np.std(kurtosis_values)),
                "min": float(np.min(kurtosis_values)),
                "max": float(np.max(kurtosis_values)),
            },
            "flag_fraction": {
                "mean": float(np.mean(flag_fractions)),
                "std": float(np.std(flag_fractions)),
                "min": float(np.min(flag_fractions)),
                "max": float(np.max(flag_fractions)),
            },
        }

    def suggest_strategy(self, spw: int) -> str:
        """Suggest a flagging strategy based on RFI conditions.

        Parameters
        ----------
        spw : int
            Spectral window index

        """
        thresholds = self.get_adaptive_thresholds(spw)
        return thresholds.environment_condition

    def clear(self) -> None:
        """Clear all in-memory history."""
        with self._lock:
            self._spw_history.clear()
            self._antenna_spw_history.clear()
            self._environment_changes.clear()


# Global singleton instance
_global_tracker: RFIThresholdTracker | None = None
_tracker_lock = Lock()


def get_global_rfi_tracker(
    history_size: int = DEFAULT_HISTORY_SIZE,
    db_path: Path | None = None,
    persist: bool = True,
) -> RFIThresholdTracker:
    """Get or create the global RFI threshold tracker.

    Parameters
    ----------
    history_size :
        Maximum observations per SPW
    db_path :
        Database path for persistence
    persist :
        Whether to persist to database
    history_size : int :
        (Default value = DEFAULT_HISTORY_SIZE)
    db_path : Optional[Path] :
        (Default value = None)
    persist : bool :
        (Default value = True)
    """
    global _global_tracker

    with _tracker_lock:
        if _global_tracker is None:
            _global_tracker = RFIThresholdTracker(
                history_size=history_size,
                db_path=db_path,
                persist=persist,
            )
        return _global_tracker


def reset_global_rfi_tracker() -> None:
    """Reset the global tracker (useful for testing)."""
    global _global_tracker
    with _tracker_lock:
        _global_tracker = None


def compute_rfi_stats_from_ms(
    ms_path: str,
    spw: int | None = None,
) -> dict[int, dict[str, float]]:
    """Compute RFI statistics from a measurement set.

    Parameters
    ----------
    ms_path : str
        Path to measurement set
    spw : Optional[int]
        Specific SPW to analyze (None = all SPWs), default is None

    """
    results = {}

    try:
        import casacore.tables as casatables

        with casatables.table(ms_path, readonly=True, ack=False) as tb:
            flags = tb.getcol("FLAG")
            data = tb.getcol("DATA")

            # For now, compute array-wide stats (can be extended per-SPW)
            unflagged_mask = ~flags
            unflagged_data = data[unflagged_mask]

            if len(unflagged_data) > 0:
                amplitudes = np.abs(unflagged_data)

                # MAD
                median_amp = np.median(amplitudes)
                mad = float(np.median(np.abs(amplitudes - median_amp)))

                # Kurtosis (excess)
                mean_amp = np.mean(amplitudes)
                std_amp = np.std(amplitudes)
                if std_amp > 0:
                    kurtosis = float(np.mean(((amplitudes - mean_amp) / std_amp) ** 4) - 3)
                else:
                    kurtosis = 0.0

                results[0] = {
                    "mad": mad,
                    "kurtosis": kurtosis,
                    "rms": float(std_amp),
                    "flag_fraction": float(np.mean(flags)),
                    "max_amplitude": float(np.max(amplitudes)),
                }

    except Exception as e:
        logger.warning(f"Failed to compute RFI stats from {ms_path}: {e}")

    return results
