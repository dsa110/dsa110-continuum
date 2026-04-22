"""Per-epoch automated batch orchestrator with CompositeQA gating.

Architecture
------------
``EpochOrchestrator`` is the reusable, importable counterpart to the monolithic
``scripts/batch_pipeline.py``.  It owns one epoch at a time:

1. **Mosaic** — reproject + co-add tile FITS into a single epoch mosaic via
   :func:`~dsa110_continuum.mosaic.builder.fast_reproject_and_coadd`.
2. **QA** — evaluate the three-gate composite metric
   (:class:`~dsa110_continuum.qa.composite.CompositeQA`).
3. **Decide** — ``ACCEPT`` (all gates pass), ``WARN`` (no failures, at least one
   warning), or ``REJECT`` (at least one gate fails).
4. **Persist** — write per-epoch results to a SQLite table
   ``epoch_runs`` in the database pointed to by ``PathConfig.pipeline_db``.

Design decisions
----------------
* No CASA dependency — the mosaic step calls the already-fixed
  :func:`fast_reproject_and_coadd` which uses ``reproject`` (pure Python).
* The QA evaluator works entirely on scalar statistics so it can be called
  with pre-computed values in unit tests (no real image required).
* SQLite persistence is optional; set ``db_path=None`` to skip.
* Thread-safe at the per-orchestrator level (each orchestrator opens its own
  DB connection).
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

from dsa110_continuum.qa.composite import (
    CompositeQA,
    CompositeQAResult,
    QAStatus,
    theoretical_rms_jyb,
)

# Optional scattering QA dependencies — absent when scattering library not installed
try:
    from dsa110_continuum.qa.scattering_qa import check_tile_scattering as _check_tile_scattering
    from dsa110_continuum.visualization.scattering_diagnostics import (
        plot_scattering_overview as _plot_scattering_overview,
    )
    _SCATTERING_AVAILABLE = True
except ImportError:
    _SCATTERING_AVAILABLE = False
    _check_tile_scattering = None  # type: ignore[assignment]
    _plot_scattering_overview = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default QA thresholds (mirrors composite.py defaults)
# ---------------------------------------------------------------------------

DEFAULT_MAX_FLUX_SCALE_ERROR: float = 0.15
DEFAULT_MIN_COMPLETENESS: float = 0.70
DEFAULT_MAX_NOISE_FACTOR: float = 2.0

# ---------------------------------------------------------------------------
# Decision enum
# ---------------------------------------------------------------------------


class EpochDecision(str, Enum):
    """Outcome of one epoch run through the orchestrator."""

    ACCEPT = "accept"   # all QA gates pass
    WARN = "warn"       # no failures but at least one warning
    REJECT = "reject"   # at least one QA gate failed
    SKIP = "skip"       # no tiles supplied, epoch skipped entirely


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class EpochRunResult:
    """Full record for one epoch processed by :class:`EpochOrchestrator`.

    Attributes
    ----------
    epoch_id : str
        ISO-8601 UTC string identifying the epoch (e.g. ``"2026-01-25T22:00:00"``).
    decision : EpochDecision
        Accept / warn / reject / skip.
    n_tiles : int
        Number of tile FITS files supplied to this run.
    mosaic_path : str or None
        Path to the written epoch mosaic FITS, or ``None`` if skipped.
    qa : CompositeQAResult or None
        Full three-gate QA result object.
    flux_scale_correction : float
        Multiplicative flux-scale correction applied (1.0 = identity).
    measured_rms_jyb : float
        Measured mosaic RMS noise in Jy/beam.
    theoretical_rms : float
        Theoretical thermal noise in Jy/beam (from radiometer equation).
    n_detected : int
        Detected sources in epoch mosaic used for completeness gate.
    n_catalog_expected : int
        Expected sources from reference catalog in same field.
    elapsed_s : float
        Wall-clock seconds from start of ``run_epoch()`` call.
    notes : list[str]
        Any diagnostic notes accumulated during the run.
    """

    epoch_id: str
    decision: EpochDecision = EpochDecision.SKIP
    n_tiles: int = 0
    mosaic_path: Optional[str] = None
    qa: Optional[CompositeQAResult] = None
    flux_scale_correction: float = 1.0
    measured_rms_jyb: float = float("nan")
    theoretical_rms: float = float("nan")
    n_detected: int = 0
    n_catalog_expected: int = 0
    elapsed_s: float = 0.0
    notes: list[str] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def accepted(self) -> bool:
        return self.decision == EpochDecision.ACCEPT

    @property
    def rejected(self) -> bool:
        return self.decision == EpochDecision.REJECT

    @property
    def warned(self) -> bool:
        return self.decision == EpochDecision.WARN

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable flat dictionary suitable for SQLite storage."""
        d: dict[str, Any] = {
            "epoch_id": self.epoch_id,
            "decision": self.decision.value,
            "n_tiles": self.n_tiles,
            "mosaic_path": self.mosaic_path,
            "flux_scale_correction": self.flux_scale_correction,
            "measured_rms_jyb": self.measured_rms_jyb,
            "theoretical_rms": self.theoretical_rms,
            "n_detected": self.n_detected,
            "n_catalog_expected": self.n_catalog_expected,
            "elapsed_s": self.elapsed_s,
            "notes": json.dumps(self.notes),
        }
        if self.qa is not None:
            d["qa_status"] = self.qa.status.value
            d["qa_json"] = json.dumps(self.qa.to_dict())
        else:
            d["qa_status"] = None
            d["qa_json"] = None
        return d

    def summary(self) -> str:
        """One-line human-readable summary."""
        qa_str = self.qa.summary() if self.qa is not None else "qa=skipped"
        return (
            f"EpochRun[{self.decision.value}] epoch={self.epoch_id} "
            f"tiles={self.n_tiles} {qa_str} elapsed={self.elapsed_s:.1f}s"
        )


# ---------------------------------------------------------------------------
# SQLite schema helpers
# ---------------------------------------------------------------------------

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS epoch_runs (
    epoch_id              TEXT PRIMARY KEY,
    decision              TEXT NOT NULL,
    n_tiles               INTEGER,
    mosaic_path           TEXT,
    flux_scale_correction REAL,
    measured_rms_jyb      REAL,
    theoretical_rms       REAL,
    n_detected            INTEGER,
    n_catalog_expected    INTEGER,
    elapsed_s             REAL,
    qa_status             TEXT,
    qa_json               TEXT,
    notes                 TEXT,
    run_timestamp         TEXT DEFAULT (datetime('now'))
)
"""

_UPSERT = """
INSERT INTO epoch_runs (
    epoch_id, decision, n_tiles, mosaic_path,
    flux_scale_correction, measured_rms_jyb, theoretical_rms,
    n_detected, n_catalog_expected, elapsed_s,
    qa_status, qa_json, notes
) VALUES (
    :epoch_id, :decision, :n_tiles, :mosaic_path,
    :flux_scale_correction, :measured_rms_jyb, :theoretical_rms,
    :n_detected, :n_catalog_expected, :elapsed_s,
    :qa_status, :qa_json, :notes
)
ON CONFLICT(epoch_id) DO UPDATE SET
    decision              = excluded.decision,
    n_tiles               = excluded.n_tiles,
    mosaic_path           = excluded.mosaic_path,
    flux_scale_correction = excluded.flux_scale_correction,
    measured_rms_jyb      = excluded.measured_rms_jyb,
    theoretical_rms       = excluded.theoretical_rms,
    n_detected            = excluded.n_detected,
    n_catalog_expected    = excluded.n_catalog_expected,
    elapsed_s             = excluded.elapsed_s,
    qa_status             = excluded.qa_status,
    qa_json               = excluded.qa_json,
    notes                 = excluded.notes,
    run_timestamp         = datetime('now')
"""


def _init_db(con: sqlite3.Connection) -> None:
    con.execute(_CREATE_TABLE)
    con.commit()


def _persist_result(con: sqlite3.Connection, result: EpochRunResult) -> None:
    con.execute(_UPSERT, result.to_dict())
    con.commit()


# ---------------------------------------------------------------------------
# Mosaic + image statistics helpers (thin wrappers; real impl in mosaic/)
# ---------------------------------------------------------------------------


def _compute_image_rms(fits_path: str) -> float:
    """Compute MAD-based RMS from a FITS image (Jy/beam).

    Falls back gracefully if astropy is not available or file is corrupt.
    """
    try:
        from astropy.io import fits as af  # type: ignore

        with af.open(fits_path) as hdul:
            data = hdul[0].data  # type: ignore
        if data is None:
            return float("nan")
        # Squeeze degenerate axes (FITS can be 4-D)
        arr = np.squeeze(data.astype(float))
        flat = arr[np.isfinite(arr)].ravel()
        if flat.size == 0:
            return float("nan")
        return float(1.4826 * np.median(np.abs(flat - np.median(flat))))
    except Exception as exc:  # pragma: no cover
        logger.warning("Could not compute image RMS from %s: %s", fits_path, exc)
        return float("nan")


def _count_detected_sources(fits_path: str, rms_jyb: float, sigma: float = 5.0) -> int:
    """Count pixels above *sigma* × rms — cheap proxy for source count.

    A production orchestrator would call Aegean or PyBDSF; here we use a
    simple threshold for testability without heavy source-finder dependencies.
    """
    if not np.isfinite(rms_jyb) or rms_jyb <= 0:
        return 0
    try:
        from astropy.io import fits as af  # type: ignore

        with af.open(fits_path) as hdul:
            data = hdul[0].data  # type: ignore
        if data is None:
            return 0
        arr = np.squeeze(data.astype(float))
        threshold = sigma * rms_jyb
        # Count connected islands above threshold (simplified: just pixels here)
        above = np.sum(arr > threshold)
        # Rough estimate: assume each source occupies ~9 resolution elements
        return max(0, int(above // 9))
    except Exception as exc:  # pragma: no cover
        logger.warning("Could not count sources in %s: %s", fits_path, exc)
        return 0


# ---------------------------------------------------------------------------
# Main orchestrator class
# ---------------------------------------------------------------------------


class EpochOrchestrator:
    """Run the full mosaic → QA → persist pipeline for one epoch.

    Parameters
    ----------
    output_dir : str or Path, optional
        Directory for written mosaic FITS.  Defaults to the value of
        ``DSA110_PRODUCTS_BASE`` env var, falling back to ``/tmp/dsa110_mosaics``.
    db_path : str or Path or None, optional
        Path to the SQLite persistence database.  Pass ``None`` to disable
        persistence entirely (useful in tests).  Defaults to
        ``PathConfig.pipeline_db``.
    max_flux_scale_error : float
        Passed through to :class:`~dsa110_continuum.qa.composite.CompositeQA`.
    min_completeness : float
        Passed through to :class:`~dsa110_continuum.qa.composite.CompositeQA`.
    max_noise_factor : float
        Passed through to :class:`~dsa110_continuum.qa.composite.CompositeQA`.
    n_antennas : int
        Used for theoretical RMS calculation.  Default 110.
    catalog_expected_per_epoch : int
        Fallback number of expected catalog sources when the actual catalog
        is unavailable (used by completeness gate).  Default 30.
    """

    def __init__(
        self,
        output_dir: str | Path | None = None,
        db_path: str | Path | None = "default",
        max_flux_scale_error: float = DEFAULT_MAX_FLUX_SCALE_ERROR,
        min_completeness: float = DEFAULT_MIN_COMPLETENESS,
        max_noise_factor: float = DEFAULT_MAX_NOISE_FACTOR,
        n_antennas: int = 110,
        catalog_expected_per_epoch: int = 30,
    ) -> None:
        import os

        # Output directory
        if output_dir is None:
            base = os.environ.get("DSA110_PRODUCTS_BASE", "/tmp/dsa110_mosaics")
            self.output_dir = Path(base) / "mosaics"
        else:
            self.output_dir = Path(output_dir)

        # Database path
        if db_path == "default":
            try:
                from dsa110_continuum.config import paths

                db_path = paths.pipeline_db
            except Exception:
                db_path = None
        self.db_path: Optional[Path] = Path(db_path) if db_path is not None else None

        # QA thresholds
        self.qa = CompositeQA(
            max_flux_scale_error=max_flux_scale_error,
            min_completeness=min_completeness,
            max_noise_factor=max_noise_factor,
        )
        self.n_antennas = n_antennas
        self.catalog_expected_per_epoch = catalog_expected_per_epoch

        # Lazy DB connection (opened on first write)
        self._db_con: Optional[sqlite3.Connection] = None

    # ------------------------------------------------------------------
    # Database lifecycle
    # ------------------------------------------------------------------

    def _get_connection(self) -> Optional[sqlite3.Connection]:
        if self.db_path is None:
            return None
        if self._db_con is None:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            self._db_con = sqlite3.connect(str(self.db_path))
            _init_db(self._db_con)
        return self._db_con

    def close(self) -> None:
        """Close the SQLite connection (call when orchestrator is no longer needed)."""
        if self._db_con is not None:
            self._db_con.close()
            self._db_con = None

    def __enter__(self) -> "EpochOrchestrator":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Public API — single epoch
    # ------------------------------------------------------------------

    def run_epoch(
        self,
        epoch_id: str,
        tile_paths: Sequence[str],
        *,
        flux_scale_correction: float = 1.0,
        n_catalog_expected: Optional[int] = None,
        t_int_s: float = 5.0 * 60.0,
        bandwidth_hz: float = 187.5e6,
        write_mosaic: bool = True,
        dry_run: bool = False,
    ) -> EpochRunResult:
        """Process one epoch end-to-end.

        Parameters
        ----------
        epoch_id : str
            ISO-8601 UTC string, e.g. ``"2026-01-25T22:00:00"``.
        tile_paths : sequence of str
            Paths to per-tile calibrated FITS images.
        flux_scale_correction : float
            Multiplicative correction from
            :func:`~dsa110_continuum.calibration.flux_scale_correction.compute_flux_scale_correction`.
            1.0 means no correction needed.
        n_catalog_expected : int or None
            Override the expected source count for the completeness gate.
        t_int_s : float
            Total integration time for this epoch (for theoretical RMS).
        bandwidth_hz : float
            Bandwidth in Hz (for theoretical RMS).
        write_mosaic : bool
            If ``False``, skip the mosaic-writing step (useful for QA-only reruns
            when the mosaic already exists).
        dry_run : bool
            If ``True``, skip mosaic writing AND persistence; return the result
            object without any side effects.

        Returns
        -------
        EpochRunResult
        """
        t0 = time.monotonic()
        notes: list[str] = []

        tile_paths = list(tile_paths)
        n_tiles = len(tile_paths)

        # ------------------------------------------------------------------
        # Early exit: no tiles
        # ------------------------------------------------------------------
        if n_tiles == 0:
            logger.info("Epoch %s: no tiles supplied — skipping.", epoch_id)
            result = EpochRunResult(
                epoch_id=epoch_id,
                decision=EpochDecision.SKIP,
                n_tiles=0,
                elapsed_s=time.monotonic() - t0,
                notes=["No tiles supplied"],
            )
            if not dry_run:
                self._persist(result)
            return result

        # ------------------------------------------------------------------
        # Step 1: mosaic
        # ------------------------------------------------------------------
        mosaic_path: Optional[str] = None
        if write_mosaic and not dry_run:
            mosaic_path = self._write_mosaic(epoch_id, tile_paths, notes)
        elif dry_run:
            notes.append("dry_run=True — mosaic write skipped")
        else:
            notes.append("write_mosaic=False — mosaic write skipped")

        # ------------------------------------------------------------------
        # Step 2: image statistics
        # ------------------------------------------------------------------
        theo_rms = theoretical_rms_jyb(
            n_antennas=self.n_antennas,
            t_int_s=t_int_s,
            bandwidth_hz=bandwidth_hz,
        )

        if mosaic_path is not None and Path(mosaic_path).exists():
            measured_rms = _compute_image_rms(mosaic_path)
            n_detected = _count_detected_sources(mosaic_path, measured_rms)
        else:
            # No real image available (dry-run, test context, mosaic failed)
            measured_rms = float("nan")
            n_detected = 0
            notes.append("No mosaic available for image statistics — QA gates may skip")

        n_expected = n_catalog_expected if n_catalog_expected is not None else self.catalog_expected_per_epoch

        # ------------------------------------------------------------------
        # Step 3: CompositeQA
        # ------------------------------------------------------------------
        qa_result = self._run_qa(
            epoch_id=epoch_id,
            flux_scale_correction=flux_scale_correction,
            n_detected=n_detected,
            n_catalog_expected=n_expected,
            measured_rms_jyb=measured_rms,
            theoretical_rms_jyb=theo_rms,
            notes=notes,
        )

        # ------------------------------------------------------------------
        # Step 4: decision
        # ------------------------------------------------------------------
        decision = _qa_status_to_decision(qa_result.status)
        logger.info(
            "Epoch %s → %s (%s), tiles=%d, rms=%.3g Jy/b",
            epoch_id,
            decision.value,
            qa_result.status.value,
            n_tiles,
            measured_rms,
        )

        # ── Step 3b: Scattering texture QA + diagnostic PNG for WARN/REJECT ──
        scattering_png_path: Optional[str] = None
        if (
            mosaic_path is not None
            and decision in (EpochDecision.WARN, EpochDecision.REJECT)
            and _SCATTERING_AVAILABLE
        ):
            try:
                _scat_result = _check_tile_scattering(mosaic_path)
                _out_dir = Path(mosaic_path).parent
                scattering_png_path = str(_out_dir / "scattering_overview.png")
                _plot_scattering_overview(_scat_result, scattering_png_path)
                notes.append(
                    f"Scattering QA: gate={_scat_result.gate} "
                    f"median={_scat_result.median_score:.4f} "
                    f"min={_scat_result.min_score:.4f} "
                    f"png={scattering_png_path}"
                )
                logger.info(
                    "Scattering QA PNG saved: %s (gate=%s)",
                    scattering_png_path,
                    _scat_result.gate,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("Scattering QA skipped: %s", exc)
                notes.append(f"Scattering QA skipped: {exc}")

        result = EpochRunResult(
            epoch_id=epoch_id,
            decision=decision,
            n_tiles=n_tiles,
            mosaic_path=mosaic_path,
            qa=qa_result,
            flux_scale_correction=flux_scale_correction,
            measured_rms_jyb=measured_rms,
            theoretical_rms=theo_rms,
            n_detected=n_detected,
            n_catalog_expected=n_expected,
            elapsed_s=time.monotonic() - t0,
            notes=notes,
        )

        # ------------------------------------------------------------------
        # Step 5: persist
        # ------------------------------------------------------------------
        if not dry_run:
            self._persist(result)

        return result

    # ------------------------------------------------------------------
    # Public API — batch day
    # ------------------------------------------------------------------

    def run_day(
        self,
        date: str,
        tile_dir: str | Path,
        *,
        epoch_hours: list[int] | None = None,
        glob_pattern: str = "*.fits",
        **epoch_kwargs: Any,
    ) -> list[EpochRunResult]:
        """Process all epochs for one observing date.

        Tiles under *tile_dir* are grouped by integer hour (UTC) and processed
        as separate epochs.  This mirrors the logic in
        ``scripts/batch_pipeline.py:bin_tiles_by_hour()`` but is fully
        reusable and test-friendly.

        Parameters
        ----------
        date : str
            Date string used as a label (e.g. ``"2026-01-25"``).
        tile_dir : str or Path
            Directory containing per-tile FITS files named with ISO-8601
            timestamps (e.g. ``2026-01-25T2226_*.fits``).
        epoch_hours : list of int or None
            If provided, only process tiles matching those UTC hours.
        glob_pattern : str
            Glob pattern for tile FITS discovery under ``tile_dir``.
        **epoch_kwargs
            Additional keyword arguments forwarded to :meth:`run_epoch`.

        Returns
        -------
        list of EpochRunResult
            One result per epoch (may include SKIP results for empty bins).
        """
        import re

        tile_dir = Path(tile_dir)
        all_tiles = sorted(tile_dir.glob(glob_pattern))
        if not all_tiles:
            logger.warning("run_day: no tiles found under %s matching %s", tile_dir, glob_pattern)
            return []

        # Bin by hour
        bins: dict[int, list[str]] = {}
        for tile in all_tiles:
            m = re.search(r"T(\d{2})\d{2}", tile.name)
            hour = int(m.group(1)) if m else -1
            bins.setdefault(hour, []).append(str(tile))

        results: list[EpochRunResult] = []
        for hour in sorted(bins.keys()):
            if epoch_hours is not None and hour not in epoch_hours:
                continue
            epoch_id = f"{date}T{hour:02d}:00:00"
            logger.info("Processing epoch %s (%d tiles) ...", epoch_id, len(bins[hour]))
            res = self.run_epoch(epoch_id, bins[hour], **epoch_kwargs)
            results.append(res)

        return results

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    def get_result(self, epoch_id: str) -> Optional[EpochRunResult]:
        """Retrieve a previously persisted result from the SQLite database."""
        con = self._get_connection()
        if con is None:
            return None
        row = con.execute(
            "SELECT * FROM epoch_runs WHERE epoch_id = ?", (epoch_id,)
        ).fetchone()
        if row is None:
            return None
        return _row_to_result(row, con)

    def list_epochs(self, decision: Optional[EpochDecision] = None) -> list[str]:
        """List epoch IDs stored in the database, optionally filtered by decision."""
        con = self._get_connection()
        if con is None:
            return []
        if decision is not None:
            rows = con.execute(
                "SELECT epoch_id FROM epoch_runs WHERE decision = ? ORDER BY epoch_id",
                (decision.value,),
            ).fetchall()
        else:
            rows = con.execute(
                "SELECT epoch_id FROM epoch_runs ORDER BY epoch_id"
            ).fetchall()
        return [r[0] for r in rows]

    def acceptance_rate(self) -> float:
        """Fraction of non-skipped epochs that were accepted."""
        con = self._get_connection()
        if con is None:
            return float("nan")
        row = con.execute(
            "SELECT COUNT(*) FROM epoch_runs WHERE decision != 'skip'"
        ).fetchone()
        total = row[0] if row else 0
        if total == 0:
            return float("nan")
        row2 = con.execute(
            "SELECT COUNT(*) FROM epoch_runs WHERE decision = 'accept'"
        ).fetchone()
        accepted = row2[0] if row2 else 0
        return accepted / total

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _write_mosaic(
        self,
        epoch_id: str,
        tile_paths: list[str],
        notes: list[str],
    ) -> Optional[str]:
        """Mosaic tile FITS into an epoch mosaic; return output path or None."""
        out_dir = self.output_dir / epoch_id.replace(":", "").replace("-", "")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = str(out_dir / "epoch_mosaic.fits")

        try:
            from dsa110_continuum.mosaic.builder import fast_reproject_and_coadd

            fast_reproject_and_coadd(tile_paths, out_path)
            notes.append(f"Mosaic written: {out_path}")
            return out_path
        except ImportError:
            notes.append("reproject not available — mosaic step skipped")
            logger.warning("fast_reproject_and_coadd unavailable (missing reproject?)")
            return None
        except Exception as exc:
            notes.append(f"Mosaic failed: {exc}")
            logger.error("Mosaic failed for epoch %s: %s", epoch_id, exc, exc_info=True)
            return None

    def _run_qa(
        self,
        epoch_id: str,
        flux_scale_correction: float,
        n_detected: int,
        n_catalog_expected: int,
        measured_rms_jyb: float,
        theoretical_rms_jyb: float,
        notes: list[str],
    ) -> CompositeQAResult:
        """Run CompositeQA and log the outcome."""
        result = self.qa.evaluate_counts(
            flux_scale_correction=flux_scale_correction,
            n_detected=n_detected,
            n_catalog_expected=n_catalog_expected,
            measured_rms_jyb=measured_rms_jyb,
            theoretical_rms_jyb=theoretical_rms_jyb,
            epoch=epoch_id,
        )
        notes.append(result.summary())
        return result

    def _persist(self, result: EpochRunResult) -> None:
        """Write result to SQLite, swallowing errors gracefully."""
        con = self._get_connection()
        if con is None:
            return
        try:
            _persist_result(con, result)
        except Exception as exc:  # pragma: no cover
            logger.error("Failed to persist epoch %s: %s", result.epoch_id, exc, exc_info=True)


# ---------------------------------------------------------------------------
# Helper: map QAStatus → EpochDecision
# ---------------------------------------------------------------------------


def _qa_status_to_decision(status: QAStatus) -> EpochDecision:
    mapping = {
        QAStatus.PASS: EpochDecision.ACCEPT,
        QAStatus.WARN: EpochDecision.WARN,
        QAStatus.FAIL: EpochDecision.REJECT,
        QAStatus.SKIP: EpochDecision.WARN,  # skipped gate = cautious accept
    }
    return mapping.get(status, EpochDecision.REJECT)


# ---------------------------------------------------------------------------
# Helper: reconstruct EpochRunResult from a SQLite row
# ---------------------------------------------------------------------------


def _row_to_result(row: tuple, con: sqlite3.Connection) -> EpochRunResult:
    """Reconstruct an EpochRunResult from a SQLite row (column order = _CREATE_TABLE)."""
    (
        epoch_id,
        decision,
        n_tiles,
        mosaic_path,
        flux_scale_correction,
        measured_rms_jyb,
        theoretical_rms,
        n_detected,
        n_catalog_expected,
        elapsed_s,
        qa_status,
        qa_json,
        notes_json,
        _run_timestamp,
    ) = row

    notes: list[str] = json.loads(notes_json) if notes_json else []

    qa: Optional[CompositeQAResult] = None
    if qa_json:
        try:
            _qa_dict = json.loads(qa_json)
            # Reconstruct a minimal CompositeQAResult from stored JSON
            qa = _minimal_composite_from_dict(_qa_dict)
        except Exception:
            pass

    return EpochRunResult(
        epoch_id=epoch_id,
        decision=EpochDecision(decision),
        n_tiles=n_tiles or 0,
        mosaic_path=mosaic_path,
        qa=qa,
        flux_scale_correction=flux_scale_correction or 1.0,
        measured_rms_jyb=measured_rms_jyb if measured_rms_jyb is not None else float("nan"),
        theoretical_rms=theoretical_rms if theoretical_rms is not None else float("nan"),
        n_detected=n_detected or 0,
        n_catalog_expected=n_catalog_expected or 0,
        elapsed_s=elapsed_s or 0.0,
        notes=notes,
    )


def _minimal_composite_from_dict(d: dict[str, Any]) -> Optional[CompositeQAResult]:
    """Best-effort reconstruction of CompositeQAResult from stored JSON dict."""
    try:
        from dsa110_continuum.qa.composite import (
            CompletenessGateResult,
            FluxScaleGateResult,
            NoiseFloorGateResult,
        )

        status = QAStatus(d.get("status", "skip"))
        gates = d.get("gates", {})

        def _gs(key: str, cls):  # type: ignore[no-untyped-def]
            g = gates.get(key, {})
            return cls(status=QAStatus(g.get("status", "skip")))

        return CompositeQAResult(
            status=status,
            flux_scale=_gs("flux_scale", FluxScaleGateResult),
            completeness=_gs("completeness", CompletenessGateResult),
            noise_floor=_gs("noise_floor", NoiseFloorGateResult),
            epoch=d.get("epoch"),
            notes=d.get("notes", []),
        )
    except Exception:
        return None
