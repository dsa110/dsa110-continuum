"""Pipeline provenance and run manifest.

Records calibration quality, per-tile status, and per-epoch results
into a single JSON manifest alongside pipeline products. When a mosaic
looks bad, open the manifest to immediately see what went wrong.

Tile-granular retrieval
-----------------------
For testing and diagnostics, use :func:`get_cal_qa_for_tile` to obtain
calibration QA stats for any tile FITS or MS path::

    from dsa110_continuum.qa.provenance import get_cal_qa_for_tile

    qa = get_cal_qa_for_tile("/stage/.../2026-01-25T21:17:33-image-pb.fits")
    print(qa.bp_flag_fraction)   # BP flagging fraction
    print(qa.g_phase_scatter_deg) # G-table phase scatter
    print(qa.tile_record)         # per-tile status dict (or None if not in manifest)
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import re
import socket
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Default products base — overridable via DSA_PRODUCTS_DIR env var.
_DEFAULT_PRODUCTS_BASE = "/data/dsa110-continuum/products/mosaics"

logger = logging.getLogger(__name__)


@dataclass
class RunManifest:
    """Accumulates provenance during a pipeline run and serializes to JSON."""

    # Identity
    git_sha: str = ""
    started_at: str = ""
    finished_at: str | None = None
    wall_time_sec: float | None = None
    command_line: list[str] = field(default_factory=list)
    hostname: str = ""

    # Inputs
    date: str = ""
    cal_date: str = ""
    bp_table: str = ""
    g_table: str = ""
    epoch_g_table: str | None = None
    ms_files: list[str] = field(default_factory=list)

    # Calibration quality (from compute_calibration_metrics)
    cal_quality: dict[str, Any] = field(default_factory=dict)

    # Calibration selection provenance (from ensure_bandpass)
    cal_selection: dict[str, Any] = field(default_factory=dict)

    # Per-tile records
    tiles: list[dict[str, Any]] = field(default_factory=list)

    # Per-epoch records
    epochs: list[dict[str, Any]] = field(default_factory=list)

    # QA gates triggered during the run
    gates: list[dict[str, Any]] = field(default_factory=list)

    # Overall
    gaincal_status: str = ""
    pipeline_verdict: str = ""  # "CLEAN", "DEGRADED", or "FAILED"

    @classmethod
    def start(
        cls,
        date: str,
        cal_date: str,
        argv: list[str] | None = None,
    ) -> RunManifest:
        """Create a manifest capturing initial run metadata."""
        git_sha = ""
        try:
            git_sha = (
                subprocess.check_output(
                    ["git", "rev-parse", "--short", "HEAD"],
                    stderr=subprocess.DEVNULL,
                )
                .decode()
                .strip()
            )
        except Exception:
            pass

        return cls(
            git_sha=git_sha,
            started_at=datetime.now(timezone.utc).isoformat(),
            command_line=list(argv or sys.argv),
            hostname=socket.gethostname(),
            date=date,
            cal_date=cal_date,
        )

    def assess_cal_quality(self, bp_path: str, g_path: str) -> None:
        """Compute and store calibration quality metrics.

        Calls ``compute_calibration_metrics`` from
        ``dsa110_continuum.calibration.qa`` and logs warnings for poor
        quality indicators.
        """
        from dsa110_continuum.calibration.qa import compute_calibration_metrics

        self.bp_table = bp_path
        self.g_table = g_path

        for label, path in [("bp", bp_path), ("g", g_path)]:
            try:
                metrics = compute_calibration_metrics(path)
                d = metrics.to_dict()
                self.cal_quality[label] = d

                if metrics.extraction_error:
                    logger.warning(
                        "Cal QA %s: extraction error — %s",
                        label.upper(),
                        metrics.extraction_error,
                    )
                    continue

                if metrics.flag_fraction > 0.3:
                    logger.warning(
                        "Cal QA %s: high flag fraction %.1f%%",
                        label.upper(),
                        metrics.flag_fraction * 100,
                    )
                if metrics.phase_scatter_deg > 30.0:
                    logger.warning(
                        "Cal QA %s: high phase scatter %.1f°",
                        label.upper(),
                        metrics.phase_scatter_deg,
                    )
            except Exception as exc:
                self.cal_quality[label] = {"error": str(exc)}
                logger.warning("Cal QA %s failed: %s", label.upper(), exc)

    def record_tile(
        self,
        ms_path: str,
        fits_path: str | None,
        status: str,
        elapsed_sec: float,
        error: str | None = None,
    ) -> None:
        """Record the outcome of a single tile."""
        rec: dict[str, Any] = {
            "ms_path": ms_path,
            "fits_path": fits_path,
            "status": status,
            "elapsed_sec": round(elapsed_sec, 1),
        }
        if error is not None:
            rec["error"] = error
        self.tiles.append(rec)

    def record_epoch(
        self,
        hour: int,
        epoch_result: dict[str, Any],
        epoch_qa: Any | None = None,
    ) -> None:
        """Record the outcome of an epoch mosaic."""
        rec: dict[str, Any] = {
            "hour": hour,
            "n_tiles": epoch_result.get("n_tiles"),
            "status": epoch_result.get("status"),
            "mosaic_path": epoch_result.get("mosaic_path"),
            "peak": epoch_result.get("peak"),
            "rms": epoch_result.get("rms"),
            "n_sources": epoch_result.get("n_sources"),
            "median_ratio": epoch_result.get("median_ratio"),
            "gaincal_status": epoch_result.get("gaincal_status"),
            "qa_result": epoch_result.get("qa_result"),
        }
        if epoch_qa is not None:
            try:
                rec["rms_mjy"] = epoch_qa.mosaic_rms_mjy
                rec["completeness_frac"] = epoch_qa.completeness_frac
            except AttributeError:
                pass
        self.epochs.append(rec)

    def add_gate(
        self,
        gate: str,
        verdict: str,
        reason: str,
        **extras: Any,
    ) -> None:
        """Append a structured gate entry.

        Any QA check that records a ``gate`` here will cause
        :meth:`finalize` to mark the run ``DEGRADED``. Use this for
        every degradation signal (cal quality, strip mismatch, archive
        block, gaincal fallback, etc.) so the verdict cannot silently
        disagree with an operator reading the manifest.
        """
        entry: dict[str, Any] = {"gate": gate, "verdict": verdict, "reason": reason}
        entry.update(extras)
        self.gates.append(entry)

    def epoch_verdict(self, hour: int) -> str | None:
        """Return ``qa_result`` for a previously-recorded epoch, else ``None``.

        Used by the orchestrator to decide whether a mosaic file from a
        prior crashed run can be trusted (PASS) or must be rebuilt
        (FAIL / missing / WARN-ish).
        """
        for rec in self.epochs:
            if rec.get("hour") == hour:
                verdict = rec.get("qa_result")
                return str(verdict) if verdict is not None else None
        return None

    def finalize(self, wall_time_sec: float) -> None:
        """Mark the run as finished and compute pipeline verdict."""
        self.finished_at = datetime.now(timezone.utc).isoformat()
        self.wall_time_sec = round(wall_time_sec, 1)
        if self.gates:
            self.pipeline_verdict = "DEGRADED"
        else:
            self.pipeline_verdict = "CLEAN"

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict."""
        return asdict(self)

    def save(self, output_dir: str) -> str:
        """Write manifest JSON to *output_dir*/{date}_manifest.json."""
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, f"{self.date}_manifest.json")
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
        logger.info("Manifest written: %s", path)
        return path

    @classmethod
    def load(cls, path: str) -> RunManifest:
        """Load a manifest from a saved JSON file.

        Unknown keys in the JSON (e.g., from a newer pipeline version) are
        silently ignored so older code can still read newer manifests.

        Parameters
        ----------
        path : str
            Path to the manifest JSON file (e.g. ``{products_dir}/{date}_manifest.json``).

        Returns
        -------
        RunManifest
            Populated manifest instance.

        Raises
        ------
        FileNotFoundError
            If *path* does not exist.
        """
        with open(path) as f:
            data = json.load(f)
        known = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})

    def get_tile_record(self, key: str) -> dict[str, Any] | None:
        """Return the tile record whose ``fits_path`` or ``ms_path`` matches *key*.

        Parameters
        ----------
        key : str
            A tile FITS path or MS path exactly as recorded by :meth:`record_tile`.

        Returns
        -------
        dict or None
            The matching tile record dict, or ``None`` if not found.
        """
        for rec in self.tiles:
            if rec.get("fits_path") == key or rec.get("ms_path") == key:
                return rec
        return None


# ---------------------------------------------------------------------------
# Tile-granular retrieval helpers
# ---------------------------------------------------------------------------


@dataclass
class TileCalQA:
    """Calibration QA stats for a single tile, resolved from the day manifest.

    All tiles processed in a given pipeline day share the same BP and G
    calibration tables, so ``cal_quality`` is run-level, not tile-level.
    ``tile_record`` is the per-tile status entry from the manifest (or
    ``None`` if the tile was not found in the manifest).
    """

    date: str
    cal_date: str
    bp_table: str
    g_table: str
    cal_quality: dict[str, Any]
    tile_record: dict[str, Any] | None
    gates: list[dict[str, Any]]
    pipeline_verdict: str

    @property
    def bp_flag_fraction(self) -> float | None:
        """Flagged fraction of the BP calibration table (0–1), or None."""
        return self.cal_quality.get("bp", {}).get("flag_fraction")

    @property
    def g_phase_scatter_deg(self) -> float | None:
        """Phase scatter of the G calibration table in degrees, or None."""
        return self.cal_quality.get("g", {}).get("phase_scatter_deg")

    @property
    def tile_status(self) -> str | None:
        """Processing status of this tile (``"ok"`` / ``"failed"``), or None."""
        return self.tile_record.get("status") if self.tile_record else None


def load_manifest(date: str, products_dir: str | None = None) -> RunManifest:
    """Load the day manifest for *date* from *products_dir*.

    Parameters
    ----------
    date : str
        Observation date in ``YYYY-MM-DD`` format.
    products_dir : str, optional
        Base directory containing per-date product sub-directories.
        Defaults to the ``DSA_PRODUCTS_DIR`` environment variable, or
        ``/data/dsa110-continuum/products/mosaics`` if unset.

    Returns
    -------
    RunManifest

    Raises
    ------
    FileNotFoundError
        If the manifest file does not exist for *date*.
    """
    if products_dir is None:
        products_dir = os.environ.get("DSA_PRODUCTS_DIR", _DEFAULT_PRODUCTS_BASE)
    manifest_path = os.path.join(products_dir, date, f"{date}_manifest.json")
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    return RunManifest.load(manifest_path)


def try_load_prior_manifest(
    date: str,
    products_dir: str | None = None,
) -> RunManifest | None:
    """Load the prior-run manifest for *date*, or return ``None``.

    Unlike :func:`load_manifest`, this does not raise when the file is
    missing, unreadable, or structurally invalid — it returns ``None``
    so callers can treat "no prior state" and "prior state unusable"
    identically. Used by the orchestrator to consult prior epoch
    verdicts on re-run without making resume conditional on manifest
    presence.
    """
    if products_dir is None:
        products_dir = os.environ.get("DSA_PRODUCTS_DIR", _DEFAULT_PRODUCTS_BASE)
    manifest_path = os.path.join(products_dir, date, f"{date}_manifest.json")
    if not os.path.exists(manifest_path):
        return None
    try:
        return RunManifest.load(manifest_path)
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        logger.warning("Prior manifest %s unreadable (%s); ignoring", manifest_path, exc)
        return None


def get_cal_qa_for_tile(
    tile_path: str,
    products_dir: str | None = None,
) -> TileCalQA:
    """Return calibration QA stats for a tile FITS path or MS path.

    Parses the observation date from *tile_path*, loads the day manifest,
    then returns a :class:`TileCalQA` containing the run-level calibration
    metrics and the per-tile status record (if present in the manifest).

    Parameters
    ----------
    tile_path : str
        Path to a tile FITS file (e.g. ``…/2026-01-25T21:17:33-image-pb.fits``)
        or a Measurement Set (e.g. ``…/2026-01-25T21:17:33.ms``).
    products_dir : str, optional
        Base directory for day manifests.  See :func:`load_manifest`.

    Returns
    -------
    TileCalQA

    Raises
    ------
    ValueError
        If the date cannot be parsed from *tile_path*.
    FileNotFoundError
        If no manifest exists for the inferred date.
    """
    name = Path(tile_path).name
    m = re.match(r"(\d{4}-\d{2}-\d{2})", name)
    if not m:
        raise ValueError(
            f"Cannot parse observation date (YYYY-MM-DD) from tile path: {tile_path!r}"
        )
    date = m.group(1)

    manifest = load_manifest(date, products_dir)
    tile_record = manifest.get_tile_record(tile_path)

    return TileCalQA(
        date=manifest.date,
        cal_date=manifest.cal_date,
        bp_table=manifest.bp_table,
        g_table=manifest.g_table,
        cal_quality=manifest.cal_quality,
        tile_record=tile_record,
        gates=manifest.gates,
        pipeline_verdict=manifest.pipeline_verdict,
    )
