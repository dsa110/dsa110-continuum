"""Discovery preflight for HDF5-to-calibrator-tile smoke evidence.

This module implements the cloud-safe front half of the #38 smoke workflow:
validate current catalog/index inputs, rank calibrator candidates, and reject
bad HDF5 groups before any conversion or CASA work begins.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

import astropy.units as u
import numpy as np
from astropy.time import Time
from dsa110_continuum.calibration.transit import next_transit_time
from dsa110_continuum.config import PathConfig

EXPECTED_SUBBANDS = 16
EXPECTED_INTEGRATIONS = 24
EXPECTED_CHANNELS_PER_SUBBAND = 48
EXPECTED_FIELD_ROWS = 24
DSA110_L_BAND_FWHM_DEG = 3.5
HDF5_MAGIC = b"\x89HDF\r\n\x1a\n"


# Lazy-loaded conversion API (populated on first conversion stage call).
convert_subband_groups_to_ms = None
conversion_settings = None


def _load_conversion_api() -> tuple[Any, Any]:
    """Load conversion dependencies only when the conversion stage runs."""
    global convert_subband_groups_to_ms, conversion_settings

    from dsa110_continuum.conversion.conversion_orchestrator import (
        convert_subband_groups_to_ms as loaded_converter,
    )
    from dsa110_continuum.conversion.conversion_orchestrator import (
        settings as loaded_settings,
    )

    if convert_subband_groups_to_ms is None:
        convert_subband_groups_to_ms = loaded_converter
    if conversion_settings is None:
        conversion_settings = loaded_settings
    return convert_subband_groups_to_ms, conversion_settings


@dataclass(frozen=True)
class VLABPreflight:
    """Validation result for the VLA calibrator database."""

    path: Path
    ok: bool
    config_owner: str
    checksum_sha256: str | None = None
    calibrator_count: int = 0
    lband_flux_count: int = 0
    reject_reasons: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class DiscoveryConfig:
    """Inputs for calibrator candidate discovery."""

    pipeline_db: Path
    vla_calibrator_db: Path
    output_dir: Path | None = None
    fwhm_deg: float = DSA110_L_BAND_FWHM_DEG
    min_fallback_flux_jy: float = 5.0
    indexed_dates: list[str] | None = None
    max_candidates: int = 100
    config_owner: str = "dsa110_continuum.config.PathConfig"


@dataclass(frozen=True)
class Calibrator:
    """Catalog calibrator row used during discovery."""

    name: str
    ra_deg: float
    dec_deg: float
    flux_jy: float
    position_code: str | None
    quality_codes: str | None
    selection_pool: str


@dataclass
class CandidateAssessment:
    """Selection audit row for one calibrator/group pairing."""

    calibrator_name: str
    selection_pool: str
    group_id: str | None
    date: str | None
    ra_deg: float
    dec_deg: float
    flux_jy: float
    observed_strip_dec_deg: float | None
    dec_offset_deg: float | None
    fwhm_deg: float
    beam_response: float | None
    predicted_transit_iso: str | None
    window_start_iso: str | None
    window_end_iso: str | None
    transit_center_margin_fraction: float | None
    subband_count: int | None
    integration_count: int | None
    file_count_on_disk: int
    reject_reasons: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """Return whether this candidate passed all preflight gates."""
        return not self.reject_reasons


@dataclass(frozen=True)
class DiscoveryResult:
    """Full discovery output."""

    db_preflight: VLABPreflight
    candidates: list[CandidateAssessment]
    selected: CandidateAssessment | None



@dataclass(frozen=True)
class SmokeRunConfig:
    """Inputs for one immutable HDF5-to-image evidence run."""

    calibrator: str
    group_id: str
    evidence_root: Path
    pipeline_db: Path
    vla_calibrator_db: Path
    input_dir: Path = Path("/data/incoming")
    fwhm_deg: float = DSA110_L_BAND_FWHM_DEG
    max_workers: int | None = None
    refant: str = "103"
    wsclean_path: str | None = None
    run_id: str | None = None
    work_root: Path | None = Path("/dev/shm/dsa110-continuum/hdf5_calibrator_tile_smoke")
    use_fast_work_root: bool = True
    phase_shift_backend: str = "chgcentre"


@dataclass(frozen=True)
class StageResult:
    """Outcome for one smoke-run stage."""

    name: str
    status: str
    reason: str | None = None
    elapsed_sec: float | None = None
    artifacts: dict[str, str] = field(default_factory=dict)


@dataclass
class SmokeRunManifest:
    """Append-only manifest for one evidence run."""

    run_id: str
    run_dir: Path
    config: SmokeRunConfig
    status: str = "INCOMPLETE"
    failed_stage: str | None = None
    started_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    completed_at: str | None = None
    git: dict[str, Any] = field(default_factory=dict)
    environment: dict[str, Any] = field(default_factory=dict)
    selected: dict[str, Any] | None = None
    group_files: list[dict[str, Any]] = field(default_factory=list)
    artifacts: dict[str, str] = field(default_factory=dict)
    stages: list[StageResult] = field(default_factory=list)

    @classmethod
    def new(cls, run_id: str, run_dir: Path, config: SmokeRunConfig) -> "SmokeRunManifest":
        """Create a manifest with reproducibility metadata."""
        return cls(
            run_id=run_id,
            run_dir=run_dir,
            config=config,
            git=_git_summary(),
            environment=_environment_summary(),
        )

    def record_stage(self, result: StageResult) -> None:
        """Record a stage and preserve the earliest failing stage."""
        self.stages.append(result)
        self.artifacts.update(result.artifacts)
        if result.status == "FAILED" and self.failed_stage is None:
            self.status = "FAILED"
            self.failed_stage = result.name

    def mark_succeeded(self) -> None:
        """Mark the run as complete if no earlier stage failed."""
        if self.failed_stage is None:
            self.status = "SUCCEEDED"
        self.completed_at = datetime.now(UTC).isoformat()

    def write(self) -> None:
        """Write the manifest JSON under the run directory."""
        manifest_dir = self.run_dir / "manifest"
        manifest_dir.mkdir(parents=True, exist_ok=True)
        payload = asdict(self)
        (manifest_dir / "run_status.json").write_text(
            json.dumps(payload, indent=2, default=str)
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_summary() -> dict[str, Any]:
    """Return lightweight git provenance without mutating the checkout."""
    summary: dict[str, Any] = {}
    for key, cmd in {
        "commit": ["git", "rev-parse", "HEAD"],
        "branch": ["git", "branch", "--show-current"],
        "status_short": ["git", "status", "--short"],
    }.items():
        try:
            result = subprocess.run(
                cmd,
                cwd=Path(__file__).resolve().parents[2],
                check=False,
                capture_output=True,
                text=True,
                timeout=15,
            )
            summary[key] = result.stdout.strip()
        except (OSError, subprocess.SubprocessError) as exc:
            summary[key] = f"UNAVAILABLE:{exc}"
    return summary


def _environment_summary() -> dict[str, Any]:
    """Return runtime metadata needed to reproduce the smoke run."""
    return {
        "python": sys.executable,
        "python_version": sys.version.split()[0],
        "cwd": str(Path.cwd()),
        "argv": sys.argv,
        "env": {
            key: os.environ.get(key)
            for key in (
                "PYTHONPATH",
                "PIPELINE_DB",
                "DSA110_INCOMING_DIR",
                "DSA110_VLA_CAL_DB",
                "WSCLEAN_DOCKER_TIMEOUT",
            )
            if os.environ.get(key) is not None
        },
    }


def _utc_run_id(calibrator: str, group_id: str) -> str:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    safe_group = group_id.replace(":", "").replace("-", "")
    return f"{stamp}_{calibrator.lower()}_{safe_group}"


def _validated_run_id(run_id: str) -> str:
    """Return a run id that cannot escape evidence/work roots."""
    candidate = Path(run_id)
    if candidate.is_absolute() or candidate.name != run_id or run_id in {"", ".", ".."}:
        raise ValueError(f"Invalid run_id: {run_id!r}")
    return run_id


def create_immutable_run_dir(config: SmokeRunConfig, run_id: str | None = None) -> Path:
    """Create a new run directory and refuse in-place overwrite."""
    chosen_run_id = run_id or config.run_id or _utc_run_id(config.calibrator, config.group_id)
    run_dir = config.evidence_root / chosen_run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    for name in ("inputs", "discovery", "ms", "calibration", "images", "qa", "logs", "manifest"):
        (run_dir / name).mkdir()
    return run_dir


def create_work_run_dir(config: SmokeRunConfig, run_id: str) -> Path:
    """Create an immutable fast work directory for heavy MS operations."""
    if not config.use_fast_work_root or config.work_root is None:
        return config.evidence_root / run_id
    work_dir = config.work_root / run_id
    work_dir.mkdir(parents=True, exist_ok=False)
    for name in ("conversion", "ms", "calibration", "images", "qa", "logs", "scratch"):
        (work_dir / name).mkdir(parents=True, exist_ok=True)
    return work_dir


def _table_columns(conn: sqlite3.Connection, table_name: str) -> set[str]:
    return {row[1] for row in conn.execute(f"PRAGMA table_info({table_name})")}


def _sql_col(columns: set[str], *names: str, default: str = "NULL") -> str:
    for name in names:
        if name in columns:
            return name
    return default


def validate_vla_calibrator_db(
    db_path: Path,
    *,
    config_owner: str = "dsa110_continuum.config.PathConfig",
) -> VLABPreflight:
    """Validate that the current VLA calibrator DB is usable for discovery."""
    path = Path(db_path)
    reasons: list[str] = []
    if not path.exists():
        return VLABPreflight(
            path=path,
            ok=False,
            config_owner=config_owner,
            reject_reasons=["DB_MISSING"],
        )

    checksum = _sha256(path)
    calibrator_count = 0
    lband_flux_count = 0
    try:
        with sqlite3.connect(path) as conn:
            tables = {
                row[0]
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type IN ('table', 'view')"
                )
            }
            if "calibrators" not in tables:
                reasons.append("DB_SCHEMA_MISSING_CALIBRATORS")
            if "fluxes" not in tables:
                reasons.append("DB_SCHEMA_MISSING_FLUXES")

            if not reasons:
                cal_cols = _table_columns(conn, "calibrators")
                flux_cols = _table_columns(conn, "fluxes")
                required_cal = {"name", "ra_deg", "dec_deg"}
                required_flux = {"name", "band", "flux_jy"}
                if not required_cal <= cal_cols:
                    reasons.append("DB_SCHEMA_BAD_CALIBRATORS")
                if not required_flux <= flux_cols:
                    reasons.append("DB_SCHEMA_BAD_FLUXES")

            if not reasons:
                calibrator_count = conn.execute("SELECT COUNT(*) FROM calibrators").fetchone()[0]
                lband_flux_count = conn.execute(
                    "SELECT COUNT(*) FROM fluxes WHERE band = '20cm'"
                ).fetchone()[0]
                if calibrator_count <= 0:
                    reasons.append("DB_EMPTY_CALIBRATORS")
                if lband_flux_count <= 0:
                    reasons.append("DB_EMPTY_20CM_FLUX")
    except sqlite3.Error as exc:
        reasons.append(f"DB_SQLITE_ERROR:{exc}")

    if "dsa110_contimg" in config_owner:
        reasons.append("CONFIG_OWNER_LEGACY_DSA110_CONTIMG")

    return VLABPreflight(
        path=path,
        ok=not reasons,
        config_owner=config_owner,
        checksum_sha256=checksum,
        calibrator_count=calibrator_count,
        lband_flux_count=lband_flux_count,
        reject_reasons=reasons,
    )


def _primary_names() -> set[str]:
    from dsa110_continuum.calibration.fluxscale import PRIMARY_FLUX_CALIBRATORS

    names = {name.upper() for name in PRIMARY_FLUX_CALIBRATORS}
    for info in PRIMARY_FLUX_CALIBRATORS.values():
        names.update(str(alias).upper() for alias in info.get("alt_names", []))
    return names


def _load_calibrators(db_path: Path, min_fallback_flux_jy: float) -> list[Calibrator]:
    primary_names = _primary_names()
    rows: list[Calibrator] = []
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.execute(
            """
            SELECT c.name, c.ra_deg, c.dec_deg, c.position_code,
                   c.alt_name, f.flux_jy, f.quality_codes
            FROM calibrators c
            JOIN fluxes f ON c.name = f.name
            WHERE f.band = '20cm'
            ORDER BY f.flux_jy DESC
            """
        )
        for row in cursor:
            name = str(row["name"])
            aliases = {name.upper()}
            if row["alt_name"]:
                aliases.add(str(row["alt_name"]).upper())
            is_primary = bool(aliases & primary_names)
            if not is_primary and float(row["flux_jy"]) < min_fallback_flux_jy:
                continue
            rows.append(
                Calibrator(
                    name=name,
                    ra_deg=float(row["ra_deg"]),
                    dec_deg=float(row["dec_deg"]),
                    flux_jy=float(row["flux_jy"]),
                    position_code=row["position_code"],
                    quality_codes=row["quality_codes"],
                    selection_pool="primary" if is_primary else "bright_fallback",
                )
            )
    return rows


def _load_named_calibrator(db_path: Path, name_or_alias: str) -> Calibrator:
    primary_names = _primary_names()
    requested = name_or_alias.upper()
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            """
            SELECT c.name, c.ra_deg, c.dec_deg, c.position_code,
                   c.alt_name, f.flux_jy, f.quality_codes
            FROM calibrators c
            JOIN fluxes f ON c.name = f.name
            WHERE f.band = '20cm'
              AND (upper(c.name) = ? OR upper(c.alt_name) = ?)
            ORDER BY f.flux_jy DESC
            LIMIT 1
            """,
            (requested, requested),
        ).fetchone()
    if row is None:
        raise RuntimeError(f"Calibrator not found in VLA DB 20cm catalog: {name_or_alias}")
    aliases = {str(row["name"]).upper()}
    if row["alt_name"]:
        aliases.add(str(row["alt_name"]).upper())
    is_primary = bool(aliases & primary_names)
    return Calibrator(
        name=name_or_alias,
        ra_deg=float(row["ra_deg"]),
        dec_deg=float(row["dec_deg"]),
        flux_jy=float(row["flux_jy"]),
        position_code=row["position_code"],
        quality_codes=row["quality_codes"],
        selection_pool="primary" if is_primary else "bright_fallback",
    )


def _pipeline_group_rows(conn: sqlite3.Connection, indexed_dates: list[str] | None) -> list[dict]:
    columns = _table_columns(conn, "group_time_ranges")
    start_col = _sql_col(columns, "start_time_iso", "start_iso")
    end_col = _sql_col(columns, "end_time_iso", "end_iso")
    integration_col = _sql_col(columns, "integration_count", default="NULL")
    ra_col = _sql_col(columns, "ra_deg", default="NULL")
    dec_col = _sql_col(columns, "dec_deg", default="NULL")

    filters = ""
    params: list[Any] = []
    if indexed_dates:
        placeholders = ",".join("?" for _ in indexed_dates)
        filters = f"WHERE substr(g.group_id, 1, 10) IN ({placeholders})"
        params.extend(indexed_dates)

    rows = conn.execute(
        f"""
        SELECT
            g.group_id,
            g.file_count,
            g.{start_col} AS start_time_iso,
            g.{end_col} AS end_time_iso,
            {f"g.{integration_col}" if integration_col != "NULL" else "NULL"} AS integration_count,
            {f"g.{ra_col}" if ra_col != "NULL" else "NULL"} AS group_ra_deg,
            {f"g.{dec_col}" if dec_col != "NULL" else "NULL"} AS group_dec_deg
        FROM group_time_ranges g
        {filters}
        ORDER BY g.group_id
        """,
        params,
    ).fetchall()
    return [dict(row) for row in rows]


def _group_files(conn: sqlite3.Connection, group_id: str) -> list[dict]:
    columns = _table_columns(conn, "hdf5_files")
    subband_col = _sql_col(columns, "subband_num", "subband_code")
    filename_col = _sql_col(columns, "filename", default="NULL")
    rows = conn.execute(
        f"""
        SELECT path,
               {subband_col} AS subband_code,
               {filename_col if filename_col == "NULL" else f"{filename_col}"} AS filename,
               timestamp_iso,
               ra_deg,
               dec_deg
        FROM hdf5_files
        WHERE group_id = ?
        ORDER BY subband_code
        """,
        (group_id,),
    ).fetchall()
    return [dict(row) for row in rows]


def _looks_like_hdf5(path: Path) -> bool:
    try:
        with path.open("rb") as handle:
            return handle.read(len(HDF5_MAGIC)) == HDF5_MAGIC
    except OSError:
        return False


def _subband_number(value: Any) -> int:
    if isinstance(value, int):
        return value
    text = str(value)
    if text.startswith("sb"):
        text = text[2:]
    return int(text)


def _read_hdf5_tile_metadata(files: list[dict]) -> dict[str, Any]:
    """Read strict tile metadata from the first available indexed HDF5 file."""
    first_path = next((Path(row["path"]) for row in files if Path(row["path"]).exists()), None)
    if first_path is None:
        raise FileNotFoundError("No indexed HDF5 files exist on disk")

    import h5py

    with h5py.File(first_path, "r") as handle:
        ntimes = int(handle["Header/Ntimes"][()])
        time_array = np.asarray(handle["Header/time_array"][()])
        unique_times = np.unique(time_array)
        start = Time(float(unique_times.min()), format="jd", scale="utc")
        end = Time(float(unique_times.max()), format="jd", scale="utc")
        return {
            "integration_count": ntimes,
            "start_time_iso": start.isot,
            "end_time_iso": end.isot,
            "metadata_file": str(first_path),
        }


def _hydrate_group_with_hdf5_metadata(group: dict, files: list[dict]) -> dict:
    """Fill missing tile window/integration metadata once per indexed group."""
    hydrated = dict(group)
    if hydrated.get("integration_count") is None or hydrated["start_time_iso"] == hydrated["end_time_iso"]:
        metadata = _read_hdf5_tile_metadata(files)
        hydrated["integration_count"] = metadata["integration_count"]
        hydrated["start_time_iso"] = metadata["start_time_iso"]
        hydrated["end_time_iso"] = metadata["end_time_iso"]
        hydrated["hdf5_metadata_file"] = metadata["metadata_file"]
    return hydrated


def _gaussian_response(theta_deg: float, fwhm_deg: float) -> float:
    import math

    return math.exp(-4.0 * math.log(2.0) * (theta_deg / fwhm_deg) ** 2)


def _transit_center_margin_fraction(start: Time, end: Time, transit: Time) -> float:
    duration = (end - start).to_value(u.s)
    if duration <= 0:
        return -1.0
    center = start + (duration / 2.0) * u.s
    offset = abs((transit - center).to_value(u.s))
    return offset / duration


@lru_cache(maxsize=4096)
def _cached_transit_time(ra_millideg: int, date: str) -> Time:
    """Return the calibrator transit for a rounded RA/date pair."""
    day_start = Time(f"{date}T00:00:00", format="isot", scale="utc")
    return next_transit_time(ra_millideg / 1000.0, day_start.mjd)


@lru_cache(maxsize=65536)
def _cached_isot_time(value: str) -> Time:
    """Parse an ISO timestamp once for discovery candidate assessment."""
    return Time(value, format="isot", scale="utc")


def _audit_group_files(group_id: str, files: list[dict]) -> dict[str, Any]:
    """Validate indexed HDF5 file presence/header details once per group."""
    reasons: list[str] = []
    expected_sbs = set(range(EXPECTED_SUBBANDS))
    observed_sbs = {_subband_number(row["subband_code"]) for row in files}
    if observed_sbs != expected_sbs:
        reasons.append("SUBBAND_CODES_NOT_EXACTLY_0_15")

    existing_files = 0
    for row in files:
        path = Path(row["path"])
        subband_number = _subband_number(row["subband_code"])
        if path.exists():
            existing_files += 1
        else:
            reasons.append("HDF5_FILE_MISSING")
            continue
        if f"{group_id}_sb{subband_number:02d}.hdf5" != path.name:
            reasons.append("HDF5_FILENAME_TIMESTAMP_MISMATCH")
        if not _looks_like_hdf5(path):
            reasons.append("HDF5_METADATA_UNREADABLE")
    return {"file_count_on_disk": existing_files, "reject_reasons": reasons}


def _assess_pair(
    calibrator: Calibrator,
    group: dict,
    files: list[dict],
    fwhm_deg: float,
) -> CandidateAssessment:
    reasons: list[str] = []
    observed_dec = group.get("group_dec_deg")
    group_id = str(group["group_id"])
    start_iso = str(group["start_time_iso"])
    end_iso = str(group["end_time_iso"])
    subband_count = int(group["file_count"])
    integration_count = (
        int(group["integration_count"]) if group.get("integration_count") is not None else None
    )

    hdf5_metadata: dict[str, Any] | None = None
    if integration_count is None or start_iso == end_iso:
        try:
            hdf5_metadata = _read_hdf5_tile_metadata(files)
            integration_count = int(hdf5_metadata["integration_count"])
            start_iso = str(hdf5_metadata["start_time_iso"])
            end_iso = str(hdf5_metadata["end_time_iso"])
        except Exception as exc:
            reasons.append(f"HDF5_METADATA_UNREADABLE:{exc}")
            integration_count = integration_count or 0

    dec_offset = None
    beam_response = None
    if observed_dec is None:
        reasons.append("MISSING_STRIP_DEC")
    else:
        dec_offset = abs(calibrator.dec_deg - float(observed_dec))
        beam_response = _gaussian_response(dec_offset, fwhm_deg)
        if dec_offset > fwhm_deg / 2.0:
            reasons.append("DEC_OUTSIDE_FWHM_HALF_POWER")

    if subband_count != EXPECTED_SUBBANDS:
        reasons.append("INCOMPLETE_SUBBANDS")
    if integration_count != EXPECTED_INTEGRATIONS:
        reasons.append("BAD_INTEGRATION_COUNT")

    try:
        start = _cached_isot_time(start_iso)
        end = _cached_isot_time(end_iso)
        transit = _cached_transit_time(round(calibrator.ra_deg * 1000), group_id[:10])
        transit_iso = transit.isot
        center_margin = _transit_center_margin_fraction(start, end, transit)
        if not (start <= transit <= end):
            reasons.append("TRANSIT_OUTSIDE_WINDOW")
        elif center_margin > 0.25:
            reasons.append("TRANSIT_OUTSIDE_CENTRAL_HALF")
    except Exception as exc:
        transit_iso = None
        center_margin = None
        reasons.append(f"TRANSIT_METADATA_ERROR:{exc}")

    expected_sbs = set(range(EXPECTED_SUBBANDS))
    observed_sbs = {_subband_number(row["subband_code"]) for row in files}
    if observed_sbs != expected_sbs:
        reasons.append("SUBBAND_CODES_NOT_EXACTLY_0_15")

    existing_files = 0
    for row in files:
        path = Path(row["path"])
        subband_number = _subband_number(row["subband_code"])
        if path.exists():
            existing_files += 1
        else:
            reasons.append("HDF5_FILE_MISSING")
            continue
        if f"{group_id}_sb{subband_number:02d}.hdf5" != path.name:
            reasons.append("HDF5_FILENAME_TIMESTAMP_MISMATCH")
        if not _looks_like_hdf5(path):
            reasons.append("HDF5_METADATA_UNREADABLE")

    return CandidateAssessment(
        calibrator_name=calibrator.name,
        selection_pool=calibrator.selection_pool,
        group_id=group_id,
        date=group_id[:10],
        ra_deg=calibrator.ra_deg,
        dec_deg=calibrator.dec_deg,
        flux_jy=calibrator.flux_jy,
        observed_strip_dec_deg=float(observed_dec) if observed_dec is not None else None,
        dec_offset_deg=dec_offset,
        fwhm_deg=fwhm_deg,
        beam_response=beam_response,
        predicted_transit_iso=transit_iso,
        window_start_iso=start_iso,
        window_end_iso=end_iso,
        transit_center_margin_fraction=center_margin,
        subband_count=subband_count,
        integration_count=integration_count,
        file_count_on_disk=existing_files,
        reject_reasons=sorted(set(reasons)),
    )


def discover_calibrator_candidates(config: DiscoveryConfig) -> DiscoveryResult:
    """Build a reject/accept matrix and select one pinned candidate."""
    db_preflight = validate_vla_calibrator_db(
        config.vla_calibrator_db,
        config_owner=config.config_owner,
    )
    if not db_preflight.ok:
        return DiscoveryResult(db_preflight=db_preflight, candidates=[], selected=None)

    candidates = _load_calibrators(config.vla_calibrator_db, config.min_fallback_flux_jy)
    candidates.sort(key=lambda c: (0 if c.selection_pool == "primary" else 1, -c.flux_jy))

    assessments: list[CandidateAssessment] = []
    with sqlite3.connect(config.pipeline_db) as conn:
        conn.row_factory = sqlite3.Row
        groups = _pipeline_group_rows(conn, config.indexed_dates)
        group_files = {str(group["group_id"]): _group_files(conn, str(group["group_id"])) for group in groups}
        hydrated_groups = []
        for group in groups:
            files = group_files[str(group["group_id"])]
            try:
                hydrated_groups.append(_hydrate_group_with_hdf5_metadata(group, files))
            except Exception:
                hydrated_groups.append(group)
        for calibrator in candidates[: config.max_candidates]:
            for group in hydrated_groups:
                files = group_files[str(group["group_id"])]
                assessments.append(_assess_pair(calibrator, group, files, config.fwhm_deg))

    viable = [candidate for candidate in assessments if candidate.ok]
    viable.sort(
        key=lambda c: (
            0 if c.selection_pool == "primary" else 1,
            -(c.beam_response or 0.0) * c.flux_jy,
            c.dec_offset_deg or 999.0,
        )
    )
    selected = viable[0] if viable else None
    return DiscoveryResult(db_preflight=db_preflight, candidates=assessments, selected=selected)


def discover_pinned_candidate(config: SmokeRunConfig) -> DiscoveryResult:
    """Assess only the requested calibrator and group for an executable smoke run."""
    db_preflight = validate_vla_calibrator_db(config.vla_calibrator_db)
    if not db_preflight.ok:
        return DiscoveryResult(db_preflight=db_preflight, candidates=[], selected=None)
    calibrator = _load_named_calibrator(config.vla_calibrator_db, config.calibrator)
    with sqlite3.connect(config.pipeline_db) as conn:
        conn.row_factory = sqlite3.Row
        groups = [
            group
            for group in _pipeline_group_rows(conn, [config.group_id[:10]])
            if str(group["group_id"]) == config.group_id
        ]
        if not groups:
            assessment = CandidateAssessment(
                calibrator_name=config.calibrator,
                selection_pool=calibrator.selection_pool,
                group_id=config.group_id,
                date=config.group_id[:10],
                ra_deg=calibrator.ra_deg,
                dec_deg=calibrator.dec_deg,
                flux_jy=calibrator.flux_jy,
                observed_strip_dec_deg=None,
                dec_offset_deg=None,
                fwhm_deg=config.fwhm_deg,
                beam_response=None,
                predicted_transit_iso=None,
                window_start_iso=None,
                window_end_iso=None,
                transit_center_margin_fraction=None,
                subband_count=None,
                integration_count=None,
                file_count_on_disk=0,
                reject_reasons=["PINNED_GROUP_NOT_INDEXED"],
            )
            return DiscoveryResult(db_preflight=db_preflight, candidates=[assessment], selected=None)
        files = _group_files(conn, config.group_id)
    group = _hydrate_group_with_hdf5_metadata(groups[0], files)
    assessment = _assess_pair(
        calibrator,
        group,
        files,
        config.fwhm_deg,
        file_audit=_audit_group_files(config.group_id, files),
    )
    return DiscoveryResult(
        db_preflight=db_preflight,
        candidates=[assessment],
        selected=assessment if assessment.ok else None,
    )


def _write_outputs(result: DiscoveryResult, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "db_preflight": asdict(result.db_preflight),
        "selected": asdict(result.selected) if result.selected else None,
        "candidates": [asdict(candidate) for candidate in result.candidates],
    }
    (output_dir / "discovery.json").write_text(json.dumps(payload, indent=2, default=str))

    with (output_dir / "candidate_matrix.csv").open("w", newline="") as handle:
        fieldnames = list(asdict(result.candidates[0]).keys()) if result.candidates else [
            "calibrator_name",
            "reject_reasons",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for candidate in result.candidates:
            row = asdict(candidate)
            row["reject_reasons"] = ";".join(candidate.reject_reasons)
            writer.writerow(row)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str))


def _load_group_files(pipeline_db: Path, group_id: str) -> list[dict[str, Any]]:
    with sqlite3.connect(pipeline_db) as conn:
        conn.row_factory = sqlite3.Row
        return _group_files(conn, group_id)


def _select_pinned_candidate(result: DiscoveryResult, config: SmokeRunConfig) -> CandidateAssessment:
    for candidate in result.candidates:
        if (
            candidate.group_id == config.group_id
            and candidate.calibrator_name.upper() == config.calibrator.upper()
        ):
            if not candidate.ok:
                raise RuntimeError(
                    "Pinned candidate failed preflight: "
                    + ";".join(candidate.reject_reasons)
                )
            return candidate
    raise RuntimeError(
        f"No viable pinned candidate found for {config.calibrator} {config.group_id}"
    )


def _run_stage(
    manifest: SmokeRunManifest,
    name: str,
    func,
) -> Any:
    start = time.time()
    try:
        value = func()
    except (Exception, KeyboardInterrupt) as exc:
        message = str(exc)
        reason = f"{type(exc).__name__}: {message}" if message else type(exc).__name__
        result = StageResult(
            name=name,
            status="FAILED",
            reason=reason,
            elapsed_sec=round(time.time() - start, 3),
        )
        manifest.record_stage(result)
        manifest.mark_succeeded()
        manifest.write()
        raise
    artifacts = value if isinstance(value, dict) else {}
    result = StageResult(
        name=name,
        status="SUCCEEDED",
        elapsed_sec=round(time.time() - start, 3),
        artifacts={str(k): str(v) for k, v in artifacts.items()},
    )
    manifest.record_stage(result)
    manifest.write()
    return value


@contextlib.contextmanager
def _stage_log(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    scoped_env = ("CASALOGFILE", "TMPDIR", "TMP", "TEMP", "CASA_TMPDIR")
    previous_env = {name: os.environ.get(name) for name in scoped_env}
    stage_tmp = path.parent / f"{path.stem}_tmp"
    stage_tmp.mkdir(parents=True, exist_ok=True)
    os.environ["CASALOGFILE"] = str(path.parent / f"{path.stem}.casa.log")
    for name in ("TMPDIR", "TMP", "TEMP", "CASA_TMPDIR"):
        os.environ[name] = str(stage_tmp)
    try:
        with path.open("a") as handle:
            with contextlib.redirect_stdout(handle), contextlib.redirect_stderr(handle):
                yield
    finally:
        for name, value in previous_env.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _validate_direction_sync(ms_path: Path, *, require_synced: bool) -> dict[str, Any]:
    from dsa110_continuum.adapters import casa_tables as tb

    with tb.table(f"{ms_path}::FIELD", readonly=True) as field_tb:
        phase_dir = field_tb.getcol("PHASE_DIR")
        reference_dir = field_tb.getcol("REFERENCE_DIR")
        field_rows = field_tb.nrows()
    if field_rows != EXPECTED_FIELD_ROWS:
        raise ValueError(f"Expected {EXPECTED_FIELD_ROWS} FIELD rows, found {field_rows}")
    if require_synced and not np.allclose(phase_dir, reference_dir, rtol=0.0, atol=1e-10):
        raise ValueError("FIELD::REFERENCE_DIR is not synchronized with PHASE_DIR")
    return {"field_rows": field_rows, "directions_synced": bool(np.allclose(phase_dir, reference_dir))}


def validate_source_ms(ms_path: Path, *, require_direction_sync: bool = False) -> dict[str, Any]:
    """Validate conversion/phase-shift prerequisites before downstream stages."""
    from dsa110_continuum.adapters import casa_tables as tb

    if not ms_path.exists():
        raise FileNotFoundError(f"Measurement Set does not exist: {ms_path}")

    with tb.table(str(ms_path), readonly=True) as main_tb:
        rows = main_tb.nrows()
        data_cell_shape = tuple(int(v) for v in main_tb.getcell("DATA", 0).shape)
    if rows <= 0:
        raise ValueError("Measurement Set has no MAIN rows")

    with tb.table(f"{ms_path}::SPECTRAL_WINDOW", readonly=True) as spw_tb:
        n_spw = spw_tb.nrows()
        num_chan = [int(v) for v in spw_tb.getcol("NUM_CHAN")]
    if n_spw != EXPECTED_SUBBANDS:
        raise ValueError(f"Expected {EXPECTED_SUBBANDS} SPWs, found {n_spw}")
    if any(ch != EXPECTED_CHANNELS_PER_SUBBAND for ch in num_chan):
        raise ValueError(f"Expected 48 channels per SPW, found {num_chan}")

    with tb.table(f"{ms_path}::OBSERVATION", readonly=True) as obs_tb:
        telescope_names = [str(v) for v in obs_tb.getcol("TELESCOPE_NAME")]
    if telescope_names != ["DSA_110"]:
        raise ValueError(f"Expected OBSERVATION::TELESCOPE_NAME=DSA_110, found {telescope_names}")

    direction_summary = _validate_direction_sync(
        ms_path,
        require_synced=require_direction_sync,
    )
    return {
        "ms_path": str(ms_path),
        "rows": rows,
        "spw_count": n_spw,
        "num_chan": num_chan,
        "data_cell_shape": data_cell_shape,
        "telescope_names": telescope_names,
        **direction_summary,
    }


def validate_fresh_cal_tables(
    table_paths: list[str | Path], run_dir: Path
) -> tuple[Path, list[Path]]:
    """Return fresh BP table and the ordered list of G/2G tables.

    Rejects symlinks, dummy fallbacks, and tables outside the immutable run.
    Multiple gain tables (e.g. an initial ``.g`` plus a refined ``.2g``) are
    returned in input order so downstream applycal can chain them.
    """
    run_root = run_dir.resolve()
    valid: list[Path] = []
    for raw_path in table_paths:
        path = Path(raw_path)
        if not path.exists():
            raise ValueError(f"Calibration table does not exist: {path}")
        if path.is_symlink():
            raise ValueError(f"Calibration table must not be a symlink: {path}")
        if "dummy" in path.name.lower():
            raise ValueError(f"Calibration table must not be a dummy fallback: {path}")
        resolved = path.resolve()
        if run_root not in resolved.parents and resolved != run_root:
            raise ValueError(f"Calibration table is outside evidence run: {path}")
        valid.append(path)

    bp_tables = [p for p in valid if p.name.lower().endswith(".b")]
    g_tables = [
        p
        for p in valid
        if p.name.lower().endswith(".g") or p.name.lower().endswith(".2g")
    ]
    if len(bp_tables) != 1:
        raise ValueError(f"Expected exactly one fresh .b table, found {bp_tables}")
    if not g_tables:
        raise ValueError("Expected at least one fresh .g/.2g table, found none")
    return bp_tables[0], g_tables


def _conversion_stage(
    config: SmokeRunConfig, run_dir: Path, work_dir: Path
) -> dict[str, str]:
    # Avoid forcing the heavy conversion import when the converter is already
    # bound (e.g. tests monkeypatch ``convert_subband_groups_to_ms`` directly).
    if convert_subband_groups_to_ms is None:
        converter, settings_obj = _load_conversion_api()
    else:
        converter = convert_subband_groups_to_ms
        settings_obj = conversion_settings

    if config.max_workers is not None and settings_obj is not None:
        settings_obj.conversion.max_workers = config.max_workers

    source_dir = run_dir / "ms" / "source"
    result = converter(
        input_dir=str(config.input_dir),
        output_dir=str(source_dir),
        start_time=config.group_id,
        end_time=config.group_id,
        skip_incomplete=True,
        skip_existing=False,
        stage_to_tmpfs=config.use_fast_work_root,
        defer_final_copy=config.use_fast_work_root,
        tmpfs_path=str(work_dir / "conversion"),
        scratch_dir=str(work_dir / "scratch"),
    )
    _write_json(run_dir / "manifest" / "conversion_result.json", result)
    if config.group_id not in result.get("converted", []):
        raise RuntimeError(f"Conversion did not produce pinned group: {result}")
    converted_paths = result.get("converted_paths") or {}
    staged_ms = converted_paths.get(config.group_id)
    if staged_ms is None:
        # Fall back to the canonical source_dir layout when staging was disabled.
        ms_path = source_dir / f"{config.group_id}.ms"
        if not ms_path.exists():
            raise FileNotFoundError(f"Converted source MS missing: {ms_path}")
        staged_ms = str(ms_path)
    return {
        "source_ms": staged_ms,
        "conversion_result": str(run_dir / "manifest" / "conversion_result.json"),
    }


def copy_ms_tree(source: Path, dest: Path) -> str:
    """Copy a writable MS tree, preferring CoW reflink and never using symlinks."""
    if dest.exists():
        raise FileExistsError(f"Destination MS already exists: {dest}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(
            ["cp", "-a", "--dereference", "--reflink=always", str(source), str(dest)],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        return "reflink"
    except (FileNotFoundError, subprocess.CalledProcessError):
        if dest.exists():
            shutil.rmtree(dest, ignore_errors=True)
        shutil.copytree(source, dest, symlinks=False)
        return "copytree"


def promote_final_artifacts(work_dir: Path, run_dir: Path, paths: list[Path]) -> dict[str, str]:
    """Copy selected final artifacts from fast work storage into immutable evidence storage."""
    promoted: dict[str, str] = {}
    work_root = work_dir.resolve()
    run_root = run_dir.resolve()
    for source in paths:
        source_resolved = source.resolve(strict=True)
        if not source_resolved.is_relative_to(work_root):
            raise ValueError(f"Promoted artifact must be under work_dir: {source}")
        relative = source_resolved.relative_to(work_root)
        dest = run_root / relative
        dest_resolved = dest.resolve(strict=False)
        if not dest_resolved.is_relative_to(run_root):
            raise ValueError(f"Promoted artifact destination escapes run_dir: {dest}")
        if source_resolved == dest_resolved:
            promoted[str(source)] = str(dest)
            continue
        if dest.exists():
            raise FileExistsError(f"Promoted artifact already exists: {dest}")
        dest.parent.mkdir(parents=True, exist_ok=True)
        if source_resolved.is_dir():
            shutil.copytree(source_resolved, dest, symlinks=False)
        else:
            shutil.copy2(source_resolved, dest)
        promoted[str(source)] = str(dest)
    return promoted


def _promote_image_outputs(
    work_dir: Path,
    run_dir: Path,
    image_outputs: dict[str, str],
) -> dict[str, str]:
    """Promote image products and return canonical evidence-root artifact paths."""
    paths = [Path(image_outputs["fits_image"]), Path(image_outputs["image_summary"])]
    if "quicklook_png" in image_outputs:
        paths.append(Path(image_outputs["quicklook_png"]))
    promoted = promote_final_artifacts(work_dir, run_dir, paths)
    promoted_fits = Path(promoted[image_outputs["fits_image"]])
    promoted_summary = Path(promoted[image_outputs["image_summary"]])
    promoted_png = None
    if "quicklook_png" in image_outputs:
        promoted_png = Path(promoted[image_outputs["quicklook_png"]])
    summary = json.loads(promoted_summary.read_text())
    summary["fits_path"] = str(promoted_fits)
    if promoted_png is not None:
        summary["quicklook_png_path"] = str(promoted_png)
    _write_json(promoted_summary, summary)
    promoted_manifest = run_dir / "manifest" / "promoted_artifacts.json"
    _write_json(promoted_manifest, promoted)
    result = {
        "fits_image": str(promoted_fits),
        "image_summary": str(promoted_summary),
        "promoted_artifacts": str(promoted_manifest),
    }
    if promoted_png is not None:
        result["quicklook_png"] = str(promoted_png)
    return result


def _branch_ms(source_ms: Path, work_dir: Path) -> dict[str, str]:
    solve_raw = work_dir / "ms" / "solve" / "calibrator_solve_raw.ms"
    image_raw = work_dir / "ms" / "image" / "calibrator_image_raw.ms"
    solve_copy_mode = copy_ms_tree(source_ms, solve_raw)
    image_copy_mode = copy_ms_tree(source_ms, image_raw)
    return {
        "solve_raw_ms": str(solve_raw),
        "image_raw_ms": str(image_raw),
        "solve_copy_mode": solve_copy_mode,
        "image_copy_mode": image_copy_mode,
    }


def _phaseshift_stage(
    config: SmokeRunConfig,
    run_dir: Path,
    work_dir: Path,
    solve_raw: Path,
    image_raw: Path,
) -> dict[str, str]:
    from dsa110_continuum.calibration.runner import phaseshift_ms

    solve_ms = work_dir / "ms" / "solve" / "calibrator_solve.ms"
    image_ms = work_dir / "ms" / "image" / "calibrator_image.ms"
    solve_ms_out, solve_phase = phaseshift_ms(
        ms_path=str(solve_raw),
        field="0~23",
        output_ms=str(solve_ms),
        mode="calibrator",
        calibrator_name=config.calibrator,
        use_chgcentre=False,
    )
    image_ms_out, image_phase = phaseshift_ms(
        ms_path=str(image_raw),
        field="0~23",
        output_ms=str(image_ms),
        mode="median_meridian",
        use_chgcentre=False,
    )
    validate_source_ms(Path(solve_ms_out), require_direction_sync=True)
    validate_source_ms(Path(image_ms_out), require_direction_sync=True)
    _write_json(
        run_dir / "qa" / "phasecenters.json",
        {"solve_phasecenter": solve_phase, "image_phasecenter": image_phase},
    )
    return {
        "solve_ms": solve_ms_out,
        "image_ms": image_ms_out,
        "phasecenters": str(run_dir / "qa" / "phasecenters.json"),
    }


def _fresh_calibration_stage(
    config: SmokeRunConfig,
    run_dir: Path,
    solve_ms: Path,
) -> dict[str, str]:
    from dsa110_continuum.calibration.runner import run_calibrator

    table_prefix = run_dir / "calibration" / f"{config.calibrator.lower()}_{config.group_id.replace(':', '')}_0~23"
    tables = run_calibrator(
        ms_path=str(solve_ms),
        cal_field="0~23",
        refant=config.refant,
        do_flagging=True,
        do_k=False,
        table_prefix=str(table_prefix),
        calibrator_name=config.calibrator,
        do_phaseshift=False,
    )
    bp_table, g_table = validate_fresh_cal_tables(tables, run_dir)
    _write_json(
        run_dir / "calibration" / "fresh_calibration_tables.json",
        {"tables": [str(t) for t in tables], "bp_table": str(bp_table), "g_table": str(g_table)},
    )
    return {
        "bp_table": str(bp_table),
        "g_table": str(g_table),
        "calibration_tables": str(run_dir / "calibration" / "fresh_calibration_tables.json"),
    }


def _applycal_stage(image_ms: Path, bp_table: Path, g_table: Path) -> dict[str, str]:
    from dsa110_continuum.calibration.applycal import apply_to_target
    from dsa110_continuum.utils.validation import validate_corrected_data_quality

    apply_to_target(
        ms_target=str(image_ms),
        field="",
        gaintables=[str(bp_table), str(g_table)],
        interp=["nearest", "linear"],
        calwt=True,
        verify=True,
    )
    warnings = validate_corrected_data_quality(str(image_ms))
    if warnings:
        raise RuntimeError("; ".join(warnings))
    return {"calibrated_image_ms": str(image_ms)}


def _image_summary(fits_path: Path, run_dir: Path, candidate: CandidateAssessment) -> dict[str, str]:
    from astropy.io import fits

    with fits.open(fits_path) as hdul:
        data = hdul[0].data
        header = hdul[0].header
        if data is None:
            raise ValueError(f"FITS has no data: {fits_path}")
        array = np.asarray(data, dtype=float)
        finite = np.isfinite(array)
        finite_values = array[finite]
        if finite_values.size == 0:
            raise ValueError(f"FITS has no finite pixels: {fits_path}")
        peak = float(np.nanmax(finite_values))
        rms = float(np.nanstd(finite_values))
        summary = {
            "fits_path": str(fits_path),
            "peak_jy_per_beam": peak,
            "rms_jy_per_beam": rms,
            "finite_pixel_fraction": float(finite.mean()),
            "header": {
                key: header.get(key)
                for key in ("NAXIS", "NAXIS1", "NAXIS2", "CTYPE1", "CTYPE2", "CRVAL1", "CRVAL2")
            },
            "calibrator": {
                "name": candidate.calibrator_name,
                "ra_deg": candidate.ra_deg,
                "dec_deg": candidate.dec_deg,
                "detected_positive_peak": peak > 0.0,
            },
        }
    out = run_dir / "qa" / "image_summary.json"
    _write_json(out, summary)
    return {"image_summary": str(out), "fits_image": str(fits_path)}


def _imaging_stage(
    config: SmokeRunConfig,
    run_dir: Path,
    image_ms: Path,
    candidate: CandidateAssessment,
) -> dict[str, str]:
    from dsa110_continuum.imaging.cli_imaging import image_ms as run_image_ms
    from dsa110_continuum.imaging.export import save_png_from_fits

    imagename = run_dir / "images" / "3c48"
    run_image_ms(
        ms_path=str(image_ms),
        imagename=str(imagename),
        imsize=2400,
        cell_arcsec=6.0,
        weighting="briggs",
        robust=0.5,
        niter=1000,
        threshold="0.005Jy",
        pbcor=True,
        gridder="wgridder",
        backend="wsclean",
        wsclean_path=config.wsclean_path,
        use_unicat_mask=False,
    )
    candidates = sorted((run_dir / "images").glob("3c48*-image.fits"))
    if not candidates:
        raise FileNotFoundError("WSClean did not produce a 3c48*-image.fits product")
    fits_path = candidates[0]
    quicklooks = save_png_from_fits([str(fits_path)])
    if len(quicklooks) != 1:
        raise RuntimeError(f"Expected one PNG quicklook for {fits_path}, got {quicklooks}")
    summary = _image_summary(fits_path, run_dir, candidate)
    summary["quicklook_png"] = str(Path(quicklooks[0]))
    return summary


def run_smoke(config: SmokeRunConfig) -> SmokeRunManifest:
    """Run the full from-scratch 3C48 HDF5-to-image smoke workflow."""
    run_dir = create_immutable_run_dir(config)
    work_dir = create_work_run_dir(config, run_dir.name)
    manifest = SmokeRunManifest.new(run_dir.name, run_dir, config)
    manifest.write()

    try:
        def preflight() -> dict[str, str]:
            discovery = discover_pinned_candidate(config)
            _write_outputs(discovery, run_dir / "discovery")
            selected = _select_pinned_candidate(discovery, config)
            group_files = _load_group_files(config.pipeline_db, config.group_id)
            manifest.selected = asdict(selected)
            manifest.group_files = group_files
            _write_json(run_dir / "inputs" / "group_files.json", group_files)
            return {
                "discovery": str(run_dir / "discovery" / "discovery.json"),
                "candidate_matrix": str(run_dir / "discovery" / "candidate_matrix.csv"),
                "group_files": str(run_dir / "inputs" / "group_files.json"),
            }

        _run_stage(manifest, "preflight", preflight)

        with _stage_log(run_dir / "logs" / "01_convert.log"):
            conversion = _run_stage(manifest, "conversion", lambda: _conversion_stage(config, run_dir))
        source_ms = Path(conversion["source_ms"])

        def validate_source_stage() -> dict[str, str]:
            validation_path = run_dir / "qa" / "source_ms_validation.json"
            _write_json(validation_path, validate_source_ms(source_ms))
            return {"source_ms_validation": str(validation_path)}

        _run_stage(manifest, "source_ms_validation", validate_source_stage)

        branches = _run_stage(manifest, "ms_branching", lambda: _branch_ms(source_ms, work_dir))
        with _stage_log(run_dir / "logs" / "02_phaseshift.log"):
            phased = _run_stage(
                manifest,
                "phaseshift",
                lambda: _phaseshift_stage(
                    config,
                    run_dir,
                    work_dir,
                    Path(branches["solve_raw_ms"]),
                    Path(branches["image_raw_ms"]),
                ),
            )
        with _stage_log(run_dir / "logs" / "03_fresh_calibration.log"):
            cal = _run_stage(
                manifest,
                "fresh_calibration",
                lambda: _fresh_calibration_stage(config, run_dir, Path(phased["solve_ms"])),
            )
        with _stage_log(run_dir / "logs" / "04_applycal.log"):
            _run_stage(
                manifest,
                "applycal",
                lambda: _applycal_stage(
                    Path(phased["image_ms"]),
                    Path(cal["bp_table"]),
                    Path(cal["g_table"]),
                ),
            )
        with _stage_log(run_dir / "logs" / "05_imaging.log"):
            image_outputs = _run_stage(
                manifest,
                "imaging",
                lambda: _imaging_stage(
                    config,
                    work_dir,
                    Path(phased["image_ms"]),
                    CandidateAssessment(**manifest.selected),
                ),
            )
        _run_stage(
            manifest,
            "promote_final_artifacts",
            lambda: _promote_image_outputs(
                work_dir,
                run_dir,
                image_outputs,
            ),
        )
        manifest.mark_succeeded()
        manifest.write()
        return manifest
    except Exception:
        manifest.write()
        raise


def validate_run(run_dir: Path) -> dict[str, Any]:
    """Read a smoke run manifest and summarize its current status."""
    manifest_path = run_dir / "manifest" / "run_status.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing run manifest: {manifest_path}")
    payload = json.loads(manifest_path.read_text())
    run_root = run_dir.resolve()
    errors: list[str] = []
    image_summary = run_dir / "qa" / "image_summary.json"
    payload["has_image_summary"] = image_summary.exists()
    payload["image_summary_path"] = str(image_summary) if image_summary.exists() else None
    artifacts = payload.get("artifacts", {})
    fits_image = artifacts.get("fits_image")
    quicklook_png = artifacts.get("quicklook_png")
    if payload.get("status") == "SUCCEEDED":
        if not image_summary.exists():
            errors.append(f"Missing promoted image summary: {image_summary}")
        if not fits_image:
            errors.append("Manifest missing canonical fits_image artifact")
        else:
            fits_path = Path(fits_image)
            if not fits_path.exists():
                errors.append(f"Missing promoted FITS image: {fits_path}")
            if not fits_path.resolve(strict=False).is_relative_to(run_root):
                errors.append(f"fits_image is outside run_dir: {fits_path}")
        if not quicklook_png:
            errors.append("Manifest missing canonical quicklook_png artifact")
        else:
            png_path = Path(quicklook_png)
            if not png_path.exists():
                errors.append(f"Missing promoted PNG quicklook: {png_path}")
            if not png_path.resolve(strict=False).is_relative_to(run_root):
                errors.append(f"quicklook_png is outside run_dir: {png_path}")
        if image_summary.exists():
            summary = json.loads(image_summary.read_text())
            summary_fits = summary.get("fits_path")
            if summary_fits != fits_image:
                errors.append(
                    f"image_summary fits_path {summary_fits!r} does not match manifest fits_image"
                )
            summary_png = summary.get("quicklook_png_path")
            if summary_png != quicklook_png:
                errors.append(
                    "image_summary quicklook_png_path "
                    f"{summary_png!r} does not match manifest quicklook_png"
                )
    payload["validation_errors"] = errors
    payload["validation_ok"] = not errors
    return payload


def main(argv: list[str] | None = None) -> int:
    """Run the smoke evidence CLI."""
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] not in {"discover", "run", "validate-run"}:
        argv = ["discover", *argv]

    parser = argparse.ArgumentParser(description="HDF5-to-calibrator-tile smoke evidence")
    defaults = PathConfig()
    subparsers = parser.add_subparsers(dest="command", required=True)

    discover_parser = subparsers.add_parser("discover")
    discover_parser.add_argument("--pipeline-db", type=Path, default=defaults.pipeline_db)
    discover_parser.add_argument("--vla-calibrator-db", type=Path, default=defaults.vla_cal_db)
    discover_parser.add_argument("--output-dir", type=Path, required=True)
    discover_parser.add_argument("--date", action="append", dest="dates")
    discover_parser.add_argument("--fwhm-deg", type=float, default=DSA110_L_BAND_FWHM_DEG)

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--calibrator", required=True)
    run_parser.add_argument("--group-id", required=True)
    run_parser.add_argument("--evidence-root", type=Path, required=True)
    run_parser.add_argument("--pipeline-db", type=Path, default=defaults.pipeline_db)
    run_parser.add_argument("--vla-calibrator-db", type=Path, default=defaults.vla_cal_db)
    run_parser.add_argument("--input-dir", type=Path, default=defaults.incoming_dir)
    run_parser.add_argument("--fwhm-deg", type=float, default=DSA110_L_BAND_FWHM_DEG)
    run_parser.add_argument("--max-workers", type=int)
    run_parser.add_argument("--refant", default="103")
    run_parser.add_argument("--wsclean-path")
    run_parser.add_argument("--run-id")

    validate_parser = subparsers.add_parser("validate-run")
    validate_parser.add_argument("run_dir", type=Path)

    args = parser.parse_args(argv)

    if args.command == "discover":
        result = discover_calibrator_candidates(
            DiscoveryConfig(
                pipeline_db=args.pipeline_db,
                vla_calibrator_db=args.vla_calibrator_db,
                output_dir=args.output_dir,
                fwhm_deg=args.fwhm_deg,
                indexed_dates=args.dates,
            )
        )
        _write_outputs(result, args.output_dir)
        return 0 if result.selected else 2

    if args.command == "run":
        manifest = run_smoke(
            SmokeRunConfig(
                calibrator=args.calibrator,
                group_id=args.group_id,
                evidence_root=args.evidence_root,
                pipeline_db=args.pipeline_db,
                vla_calibrator_db=args.vla_calibrator_db,
                input_dir=args.input_dir,
                fwhm_deg=args.fwhm_deg,
                max_workers=args.max_workers,
                refant=args.refant,
                wsclean_path=args.wsclean_path,
                run_id=args.run_id,
            )
        )
        print(json.dumps({"run_dir": str(manifest.run_dir), "status": manifest.status}, indent=2))
        return 0 if manifest.status == "SUCCEEDED" else 1

    if args.command == "validate-run":
        payload = validate_run(args.run_dir)
        print(json.dumps(payload, indent=2, default=str))
        return 0 if payload["validation_ok"] else 1

    raise AssertionError(f"unhandled command {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
