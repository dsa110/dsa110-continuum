"""
Calibration Stability Tracker with Ring Buffer History.

Tracks gain solutions over time to detect hardware degradation, antenna drift,
and calibrator source variability. Uses rolling windows to distinguish
instrument changes from environmental noise.

Inspired by Chimbuko's in-situ performance analysis concepts:

- Rolling statistics for adaptive baseline
- Drift detection over sliding windows
- Per-antenna trend analysis

Notes
-----
Use Cases:

1. Antenna degradation: Slow amplitude drop indicating feed problems
2. Phase drift: Unexpected jumps indicating cable/correlator issues
3. Calibrator variability: Distinguish source changes from instrument

Examples
--------
>>> from dsa110_contimg.core.qa.calibration_stability_tracker import (
...     CalibrationStabilityTracker,
...     get_global_tracker,
... )
>>> tracker = get_global_tracker()
>>> warnings = tracker.update_from_caltable("/path/to/gains.gcal")
>>> for w in warnings:
...     print(f"Warning: {w}")
"""

from __future__ import annotations

import logging
import os
import sqlite3
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Any

import numpy as np
from dsa110_contimg.common.utils import get_env_path

logger = logging.getLogger(__name__)

# Default database path for persistence - use PIPELINE_DB env var if set
_CONTIMG_BASE = str(get_env_path("CONTIMG_BASE_DIR", default="/data/dsa110-contimg"))
DEFAULT_DB_PATH = Path(os.environ.get("PIPELINE_DB", f"{_CONTIMG_BASE}/state/db/pipeline.sqlite3"))

# Configuration constants
DEFAULT_HISTORY_SIZE = 50  # Number of observations to keep per antenna
DRIFT_THRESHOLD_AMP_PERCENT = 1.0  # Alert if >1% amplitude change per observation
DRIFT_THRESHOLD_PHASE_DEG = 5.0  # Alert if >5° phase drift per observation
MIN_SAMPLES_FOR_TREND = 10  # Minimum samples before computing trends
OUTLIER_SIGMA = 3.0  # Standard deviations for outlier detection


@dataclass
class AntennaGainSnapshot:
    """Single observation's gain statistics for one antenna."""

    antenna_id: int
    observation_mjd: float
    mean_amplitude: float
    std_amplitude: float
    mean_phase_deg: float
    std_phase_deg: float
    n_solutions: int
    flagged_fraction: float
    caltable_path: str | None = None
    recorded_at: float = field(default_factory=time.time)


@dataclass
class AntennaTrendAnalysis:
    """Trend analysis results for a single antenna."""

    antenna_id: int
    n_samples: int

    # Amplitude trends
    amp_mean: float
    amp_std: float
    amp_trend_per_obs: float  # Fractional change per observation
    amp_trend_significance: float  # t-statistic

    # Phase trends
    phase_mean_deg: float
    phase_std_deg: float
    phase_trend_deg_per_obs: float  # Degrees change per observation
    phase_trend_significance: float  # t-statistic

    # Quality flags
    is_drifting_amplitude: bool = False
    is_drifting_phase: bool = False
    is_outlier: bool = False
    warning_message: str | None = None


@dataclass
class CalibrationStabilityReport:
    """Full stability report across all tracked antennas."""

    n_antennas_tracked: int
    n_observations: int
    oldest_observation_mjd: float
    newest_observation_mjd: float

    # Summary statistics
    median_amp_stability: float
    median_phase_stability_deg: float

    # Problem antennas
    drifting_antennas: list[int]
    outlier_antennas: list[int]

    # Per-antenna details
    antenna_analyses: dict[int, AntennaTrendAnalysis]

    # Warnings
    warnings: list[str]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "n_antennas_tracked": self.n_antennas_tracked,
            "n_observations": self.n_observations,
            "oldest_observation_mjd": self.oldest_observation_mjd,
            "newest_observation_mjd": self.newest_observation_mjd,
            "median_amp_stability": self.median_amp_stability,
            "median_phase_stability_deg": self.median_phase_stability_deg,
            "drifting_antennas": self.drifting_antennas,
            "outlier_antennas": self.outlier_antennas,
            "warnings": self.warnings,
            "antenna_details": {
                ant_id: {
                    "amp_trend_per_obs": analysis.amp_trend_per_obs,
                    "phase_trend_deg_per_obs": analysis.phase_trend_deg_per_obs,
                    "is_drifting_amplitude": analysis.is_drifting_amplitude,
                    "is_drifting_phase": analysis.is_drifting_phase,
                    "is_outlier": analysis.is_outlier,
                }
                for ant_id, analysis in self.antenna_analyses.items()
            },
        }


class CalibrationStabilityTracker:
    """Track calibration gain stability over time with ring buffers.

    Maintains per-antenna rolling history of gain statistics to detect
    drift, degradation, and anomalies.

    Thread-safe for use in concurrent pipeline operations.

    """

    def __init__(
        self,
        history_size: int = DEFAULT_HISTORY_SIZE,
        db_path: Path | None = None,
        persist: bool = True,
    ):
        """Initialize the tracker.

        Parameters
        ----------
        history_size : int
            Maximum observations to track per antenna
        db_path : str
            Path to SQLite database for persistence
        persist : bool
            Whether to persist history to database
        """
        self.history_size = history_size
        self.db_path = db_path or DEFAULT_DB_PATH
        self.persist = persist
        self._lock = Lock()

        # Per-antenna ring buffers: antenna_id -> deque of AntennaGainSnapshot
        self._antenna_history: dict[int, deque[AntennaGainSnapshot]] = defaultdict(
            lambda: deque(maxlen=history_size)
        )

        # Global observation tracking
        self._observation_count = 0
        self._last_observation_mjd = 0.0

        # Initialize database table if persisting
        if self.persist:
            self._init_db()
            self._load_from_db()

    def _init_db(self) -> None:
        """Create database table for persistence."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS calibration_stability_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    antenna_id INTEGER NOT NULL,
                    observation_mjd REAL NOT NULL,
                    mean_amplitude REAL NOT NULL,
                    std_amplitude REAL NOT NULL,
                    mean_phase_deg REAL NOT NULL,
                    std_phase_deg REAL NOT NULL,
                    n_solutions INTEGER NOT NULL,
                    flagged_fraction REAL NOT NULL,
                    caltable_path TEXT,
                    recorded_at REAL NOT NULL,
                    UNIQUE(antenna_id, observation_mjd)
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_cal_stability_antenna
                ON calibration_stability_history(antenna_id)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_cal_stability_mjd
                ON calibration_stability_history(observation_mjd)
            """)
            conn.commit()

    def _load_from_db(self) -> None:
        """Load recent history from database."""
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                # Get recent entries for each antenna
                cursor = conn.execute(
                    """
                    SELECT antenna_id, observation_mjd, mean_amplitude, std_amplitude,
                           mean_phase_deg, std_phase_deg, n_solutions, flagged_fraction,
                           caltable_path, recorded_at
                    FROM calibration_stability_history
                    ORDER BY observation_mjd DESC
                    LIMIT ?
                """,
                    (self.history_size * 128,),
                )  # Assume max 128 antennas

                for row in cursor:
                    snapshot = AntennaGainSnapshot(
                        antenna_id=row[0],
                        observation_mjd=row[1],
                        mean_amplitude=row[2],
                        std_amplitude=row[3],
                        mean_phase_deg=row[4],
                        std_phase_deg=row[5],
                        n_solutions=row[6],
                        flagged_fraction=row[7],
                        caltable_path=row[8],
                        recorded_at=row[9],
                    )
                    self._antenna_history[snapshot.antenna_id].append(snapshot)

                logger.info(
                    f"Loaded calibration stability history: {len(self._antenna_history)} antennas"
                )
        except Exception as e:
            logger.warning(f"Could not load calibration stability history: {e}")

    def _save_snapshot(self, snapshot: AntennaGainSnapshot) -> None:
        """Save a snapshot to database.

        Parameters
        ----------
        snapshot : AntennaGainSnapshot
            Snapshot to save
        """
        if not self.persist:
            return
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO calibration_stability_history
                    (antenna_id, observation_mjd, mean_amplitude, std_amplitude,
                     mean_phase_deg, std_phase_deg, n_solutions, flagged_fraction,
                     caltable_path, recorded_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        snapshot.antenna_id,
                        snapshot.observation_mjd,
                        snapshot.mean_amplitude,
                        snapshot.std_amplitude,
                        snapshot.mean_phase_deg,
                        snapshot.std_phase_deg,
                        snapshot.n_solutions,
                        snapshot.flagged_fraction,
                        snapshot.caltable_path,
                        snapshot.recorded_at,
                    ),
                )
                conn.commit()
        except Exception as e:
            logger.warning(f"Could not save calibration stability snapshot: {e}")

    def update(
        self,
        antenna_snapshots: list[AntennaGainSnapshot],
    ) -> list[str]:
        """Update tracker with new gain snapshots and check for drift.

        Parameters
        ----------
        antenna_snapshots : List[AntennaGainSnapshot]
            List of per-antenna gain statistics
        """
        warnings = []

        with self._lock:
            for snapshot in antenna_snapshots:
                # Add to ring buffer
                self._antenna_history[snapshot.antenna_id].append(snapshot)

                # Persist
                self._save_snapshot(snapshot)

                # Check for issues
                warning = self._check_antenna_drift(snapshot.antenna_id)
                if warning:
                    warnings.append(warning)

            self._observation_count += 1
            if antenna_snapshots:
                self._last_observation_mjd = max(s.observation_mjd for s in antenna_snapshots)

        return warnings

    def update_from_caltable(
        self,
        caltable_path: str,
        observation_mjd: float | None = None,
    ) -> list[str]:
        """Extract gains from a caltable and update tracker.

        Parameters
        ----------
        caltable_path : str
            Path to CASA calibration table
        observation_mjd : Optional[float], optional
            Override observation MJD (auto-detected if None, default is None)
        """
        snapshots = self._extract_gains_from_caltable(caltable_path, observation_mjd)
        return self.update(snapshots)

    def _extract_gains_from_caltable(
        self,
        caltable_path: str,
        observation_mjd: float | None = None,
    ) -> list[AntennaGainSnapshot]:
        """Extract per-antenna gain statistics from a caltable.

        Parameters
        ----------
        caltable_path : str
            Path to caltable
        observation_mjd : Optional[float], optional
            Observation MJD override (default is None)
        """
        try:
            import casacore.tables as casatables
        except ImportError:
            logger.warning("casacore not available, cannot extract gains")
            return []

        snapshots = []

        try:
            with casatables.table(caltable_path, readonly=True, ack=False) as tb:
                if "CPARAM" not in tb.colnames():
                    logger.warning(f"No CPARAM column in {caltable_path}")
                    return []

                gains = tb.getcol("CPARAM")  # Complex gains
                flags = tb.getcol("FLAG")
                antenna_ids = tb.getcol("ANTENNA1")
                times = tb.getcol("TIME")

                # Auto-detect observation MJD from midpoint
                if observation_mjd is None:
                    # Times are in MJD seconds
                    observation_mjd = (np.min(times) + np.max(times)) / 2 / 86400.0

                # Process per antenna
                unique_antennas = np.unique(antenna_ids)

                for ant_id in unique_antennas:
                    ant_mask = antenna_ids == ant_id
                    ant_gains = gains[ant_mask]
                    ant_flags = flags[ant_mask]

                    # Apply flags
                    valid_mask = ~ant_flags
                    if not np.any(valid_mask):
                        continue

                    valid_gains = ant_gains[valid_mask]

                    # Compute statistics
                    amplitudes = np.abs(valid_gains).flatten()
                    phases_deg = np.angle(valid_gains, deg=True).flatten()

                    # Wrap phases to [-180, 180]
                    phases_deg = np.mod(phases_deg + 180, 360) - 180

                    snapshot = AntennaGainSnapshot(
                        antenna_id=int(ant_id),
                        observation_mjd=observation_mjd,
                        mean_amplitude=float(np.mean(amplitudes)),
                        std_amplitude=float(np.std(amplitudes)),
                        mean_phase_deg=float(np.mean(phases_deg)),
                        std_phase_deg=float(np.std(phases_deg)),
                        n_solutions=int(valid_gains.size),
                        flagged_fraction=float(1 - np.mean(valid_mask)),
                        caltable_path=caltable_path,
                    )
                    snapshots.append(snapshot)

        except Exception as e:
            logger.error(f"Error extracting gains from {caltable_path}: {e}")

        return snapshots

    def _check_antenna_drift(self, antenna_id: int) -> str | None:
        """Check if antenna is showing problematic drift.

        Parameters
        ----------
        antenna_id : int
            Antenna identifier.

        Returns
        -------
            None
        """
        history = self._antenna_history.get(antenna_id)
        if not history or len(history) < MIN_SAMPLES_FOR_TREND:
            return None

        # Get recent data
        recent = list(history)
        n = len(recent)

        # Extract arrays
        amplitudes = np.array([s.mean_amplitude for s in recent])
        phases = np.array([s.mean_phase_deg for s in recent])

        # Compute amplitude trend
        x = np.arange(n)
        if np.std(amplitudes) > 1e-10:
            amp_slope, amp_intercept = np.polyfit(x, amplitudes, 1)
            amp_mean = np.mean(amplitudes)
            amp_trend_frac = amp_slope / amp_mean if amp_mean > 0 else 0
            amp_trend_pct = amp_trend_frac * 100

            if abs(amp_trend_pct) > DRIFT_THRESHOLD_AMP_PERCENT:
                direction = "increasing" if amp_trend_pct > 0 else "decreasing"
                return (
                    f"Antenna {antenna_id}: amplitude {direction} at "
                    f"{abs(amp_trend_pct):.2f}%/observation (over {n} observations)"
                )

        # Compute phase trend (unwrap first)
        phases_unwrapped = np.unwrap(np.deg2rad(phases))
        phases_unwrapped_deg = np.rad2deg(phases_unwrapped)

        if np.std(phases_unwrapped_deg) > 1e-10:
            phase_slope, _ = np.polyfit(x, phases_unwrapped_deg, 1)

            if abs(phase_slope) > DRIFT_THRESHOLD_PHASE_DEG:
                direction = "increasing" if phase_slope > 0 else "decreasing"
                return (
                    f"Antenna {antenna_id}: phase {direction} at "
                    f"{abs(phase_slope):.2f}°/observation (over {n} observations)"
                )

        return None

    def _analyze_antenna_unlocked(self, antenna_id: int) -> AntennaTrendAnalysis | None:
        """Internal analysis without lock (caller must hold lock).

        Parameters
        ----------
        antenna_id : int
            Antenna identifier.

        Returns
        -------
            None
        """
        history = self._antenna_history.get(antenna_id)
        if not history:
            return None

        recent = list(history)
        n = len(recent)

        if n < 2:
            return None

        # Extract arrays
        amplitudes = np.array([s.mean_amplitude for s in recent])
        phases = np.array([s.mean_phase_deg for s in recent])

        # Amplitude statistics and trend
        amp_mean = float(np.mean(amplitudes))
        amp_std = float(np.std(amplitudes))

        x = np.arange(n)
        if amp_std > 1e-10 and n >= MIN_SAMPLES_FOR_TREND:
            slope, intercept, r, p, se = self._linregress(x, amplitudes)
            amp_trend = slope / amp_mean if amp_mean > 0 else 0
            amp_significance = abs(slope / se) if se > 0 else 0
        else:
            amp_trend = 0.0
            amp_significance = 0.0

        # Phase statistics and trend
        phases_unwrapped = np.unwrap(np.deg2rad(phases))
        phases_unwrapped_deg = np.rad2deg(phases_unwrapped)

        phase_mean = float(np.mean(phases))
        phase_std = float(np.std(phases))

        if phase_std > 1e-10 and n >= MIN_SAMPLES_FOR_TREND:
            slope, intercept, r, p, se = self._linregress(x, phases_unwrapped_deg)
            phase_trend = slope
            phase_significance = abs(slope / se) if se > 0 else 0
        else:
            phase_trend = 0.0
            phase_significance = 0.0

        # Check for drift
        is_drifting_amp = abs(amp_trend * 100) > DRIFT_THRESHOLD_AMP_PERCENT
        is_drifting_phase = abs(phase_trend) > DRIFT_THRESHOLD_PHASE_DEG

        # Check for outlier (most recent vs history)
        if n >= MIN_SAMPLES_FOR_TREND:
            recent_amp = amplitudes[-1]
            baseline_mean = np.mean(amplitudes[:-1])
            baseline_std = np.std(amplitudes[:-1])
            is_outlier = (
                abs(recent_amp - baseline_mean) > OUTLIER_SIGMA * baseline_std
                if baseline_std > 0
                else False
            )
        else:
            is_outlier = False

        # Generate warning message
        warning = None
        if is_drifting_amp or is_drifting_phase or is_outlier:
            parts = []
            if is_drifting_amp:
                parts.append(f"amplitude drift {amp_trend * 100:.2f}%/obs")
            if is_drifting_phase:
                parts.append(f"phase drift {phase_trend:.2f}°/obs")
            if is_outlier:
                parts.append("recent outlier")
            warning = f"Antenna {antenna_id}: " + ", ".join(parts)

        return AntennaTrendAnalysis(
            antenna_id=antenna_id,
            n_samples=n,
            amp_mean=amp_mean,
            amp_std=amp_std,
            amp_trend_per_obs=amp_trend,
            amp_trend_significance=amp_significance,
            phase_mean_deg=phase_mean,
            phase_std_deg=phase_std,
            phase_trend_deg_per_obs=phase_trend,
            phase_trend_significance=phase_significance,
            is_drifting_amplitude=is_drifting_amp,
            is_drifting_phase=is_drifting_phase,
            is_outlier=is_outlier,
            warning_message=warning,
        )

    def analyze_antenna(self, antenna_id: int) -> AntennaTrendAnalysis | None:
        """Get detailed trend analysis for a single antenna.

        Parameters
        ----------
        antenna_id : int
            Antenna identifier.

        Returns
        -------
            None
        """
        with self._lock:
            return self._analyze_antenna_unlocked(antenna_id)

    @staticmethod
    def _linregress(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float, float, float]:
        """Simple linear regression returning (slope, intercept, r, p, stderr).

        Parameters
        ----------
        x : np.ndarray
            Independent variable data.
        y : np.ndarray
            Dependent variable data.

        Returns
        -------
            tuple
            Tuple containing slope, intercept, correlation coefficient (r), p-value, and standard error.
        """
        from scipy import stats

        return stats.linregress(x, y)

    def generate_report(self) -> CalibrationStabilityReport:
        """Generate a full stability report across all antennas."""
        with self._lock:
            antenna_analyses = {}
            warnings = []
            drifting = []
            outliers = []

            amp_stabilities = []
            phase_stabilities = []
            all_mjds = []

            for ant_id, history in self._antenna_history.items():
                if not history:
                    continue

                analysis = self._analyze_antenna_unlocked(ant_id)
                if analysis:
                    antenna_analyses[ant_id] = analysis

                    amp_stabilities.append(analysis.amp_std)
                    phase_stabilities.append(analysis.phase_std_deg)

                    if analysis.is_drifting_amplitude or analysis.is_drifting_phase:
                        drifting.append(ant_id)
                    if analysis.is_outlier:
                        outliers.append(ant_id)
                    if analysis.warning_message:
                        warnings.append(analysis.warning_message)

                # Collect MJDs
                all_mjds.extend(s.observation_mjd for s in history)

            return CalibrationStabilityReport(
                n_antennas_tracked=len(self._antenna_history),
                n_observations=self._observation_count,
                oldest_observation_mjd=min(all_mjds) if all_mjds else 0.0,
                newest_observation_mjd=max(all_mjds) if all_mjds else 0.0,
                median_amp_stability=float(np.median(amp_stabilities)) if amp_stabilities else 0.0,
                median_phase_stability_deg=float(np.median(phase_stabilities))
                if phase_stabilities
                else 0.0,
                drifting_antennas=sorted(drifting),
                outlier_antennas=sorted(outliers),
                antenna_analyses=antenna_analyses,
                warnings=warnings,
            )

    def get_antenna_history(self, antenna_id: int) -> list[AntennaGainSnapshot]:
        """Get full history for an antenna.

        Parameters
        ----------
        antenna_id : int
            Antenna identifier.

        Returns
        -------
            None
        """
        with self._lock:
            history = self._antenna_history.get(antenna_id)
            return list(history) if history else []

    def clear(self) -> None:
        """Clear all in-memory history."""
        with self._lock:
            self._antenna_history.clear()
            self._observation_count = 0
            self._last_observation_mjd = 0.0


# Global singleton instance
_global_tracker: CalibrationStabilityTracker | None = None
_tracker_lock = Lock()


def get_global_tracker(
    history_size: int = DEFAULT_HISTORY_SIZE,
    db_path: Path | None = None,
    persist: bool = True,
) -> CalibrationStabilityTracker:
    """Get or create the global calibration stability tracker.

    Parameters
    ----------
    history_size : int, optional
        Maximum observations per antenna (default is DEFAULT_HISTORY_SIZE)
    db_path : Optional[Path], optional
        Database path for persistence (default is None)
    persist : bool, optional
        Whether to persist to database (default is True)
    """
    global _global_tracker

    with _tracker_lock:
        if _global_tracker is None:
            _global_tracker = CalibrationStabilityTracker(
                history_size=history_size,
                db_path=db_path,
                persist=persist,
            )
        return _global_tracker


def reset_global_tracker() -> None:
    """Reset the global tracker (useful for testing)."""
    global _global_tracker
    with _tracker_lock:
        _global_tracker = None
