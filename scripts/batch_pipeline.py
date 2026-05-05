#!/opt/miniforge/envs/casa6/bin/python
"""
Batch pipeline: calibrate → image → hourly-epoch mosaics → forced photometry.

Usage:
    python scripts/batch_pipeline.py [--date DATE] [--cal-date DATE] [--keep-intermediates] [--skip-photometry]

Steps:
    1. Find all valid MS files for DATE and process each one:
       phaseshift → applycal → WSClean image  (skip if tile FITS already exists)
    2. Bin tile images into 1-hour epochs by observation timestamp.
       Each epoch mosaic also includes the last 2 tiles from the previous epoch
       and the first 2 tiles from the next epoch (~4-tile / ~20-min overlap).
       The first and last epochs of the day have overlap on one side only.
    3. For each epoch (skip if output mosaic already exists):
       a. Build mosaic FITS  →  {stage}/mosaic_{date}/{date}T{HH}00_mosaic.fits
       b. Run QA (noise consistency)
       c. Forced photometry against master catalog → {products}/{date}T{HH}00_forced_phot.csv
    4. Print per-epoch summary + overall totals.

Output layout (after mosaic move to products/):
    /data/dsa110-continuum/products/mosaics/{date}/
        {date}T{HH}00_mosaic.fits
        {date}T{HH}00_forced_phot.csv
        ...
"""
import argparse
import csv
import json
import logging
import os
import shutil
import sys
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, TimeoutError as FuturesTimeoutError
from datetime import datetime, timezone
from pathlib import Path

# ── Load scripts/.env before anything else ───────────────────────────────────
_ENV_FILE = Path(__file__).parent / ".env"
if _ENV_FILE.exists():
    for _line in _ENV_FILE.read_text().splitlines():
        _line = _line.strip()
        if _line.startswith("export "):
            _line = _line[len("export "):]
        if "=" in _line and not _line.startswith("#"):
            _key, _, _val = _line.partition("=")
            os.environ.setdefault(_key.strip(), _val.strip())

# ── Project root + scripts/ on path ──────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))  # enables `import mosaic_day`

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

from dsa110_continuum.photometry.epoch_qa import EpochQAResult, measure_epoch_qa
from dsa110_continuum.photometry.epoch_qa_plot import plot_epoch_qa
from dsa110_continuum.qa.provenance import RunManifest, try_load_prior_manifest

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── Configuration ─────────────────────────────────────────────────────────────

DEFAULT_DATE = "2026-01-25"
MS_DIR = os.environ.get("DSA110_MS_DIR", "/stage/dsa110-contimg/ms")
STAGE_IMAGE_BASE = os.environ.get("DSA110_STAGE_IMAGE_BASE", "/stage/dsa110-contimg/images")
PRODUCTS_BASE = os.environ.get("DSA110_PRODUCTS_BASE", "/data/dsa110-proc/products/mosaics")
CELL_ARCSEC = 6.0  # must match mosaic_day.py
TILE_TIMEOUT_SEC = 1800  # 30 min max per tile before we kill & skip

# QA summary CSV schema (expanded for three-gate epoch QA)
QA_SUMMARY_CSV = os.environ.get(
    "DSA110_QA_SUMMARY",
    "/data/dsa110-proc/products/qa_summary.csv",
)
QA_CSV_FIELDS = [
    "date", "epoch_utc", "mosaic_path",
    "n_catalog", "n_recovered", "completeness_frac",
    "median_ratio", "ratio_gate", "completeness_gate",
    "rms_gate", "mosaic_rms_mjy",
    "qa_result", "gaincal_used",
]


def get_paths(date: str) -> dict:
    return {
        "ms_dir": MS_DIR,
        "stage_dir": f"{STAGE_IMAGE_BASE}/mosaic_{date}",
        "products_dir": f"{PRODUCTS_BASE}/{date}",
    }


def epoch_mosaic_path(paths: dict, date: str, hour: int) -> str:
    return f"{paths['stage_dir']}/{date}T{hour:02d}00_mosaic.fits"


def epoch_phot_path(paths: dict, date: str, hour: int) -> str:
    return f"{paths['products_dir']}/{date}T{hour:02d}00_forced_phot.csv"


# ── Timestamp parsing ─────────────────────────────────────────────────────────

def timestamp_from_fits(fits_path: str) -> datetime | None:
    """Extract UTC datetime from a tile FITS path like .../2026-01-25T21:17:33-image-pb.fits."""
    name = Path(fits_path).name  # e.g. 2026-01-25T21:17:33-image-pb.fits
    ts_str = name.split("-image")[0]  # e.g. 2026-01-25T21:17:33
    try:
        return datetime.strptime(ts_str, "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc)
    except ValueError:
        return None


# ── Run logging ───────────────────────────────────────────────────────────────

def _run_log_filename(started_at: datetime) -> str:
    """Return ``run_<UTC-ISO>.log`` with a filename-safe timestamp.

    Colons replaced with underscores so the file is portable across
    filesystems that disallow ``:`` (Windows shares, some cloud
    bucket mounts). Always UTC, always second-resolution; sorts
    correctly under ``ls`` since the year-first ISO form is monotonic.
    """
    stamp = started_at.astimezone(timezone.utc).strftime("%Y-%m-%dT%H_%M_%SZ")
    return f"run_{stamp}.log"


def _attach_run_logfile(date_dir: str, started_at: datetime) -> str:
    """Attach a per-run :class:`logging.FileHandler` to the root logger.

    *date_dir* is the per-date products directory (already date-nested,
    e.g. ``/data/products/2026-01-25/``); the log file is written
    directly into it as ``run_<utc-stamp>.log`` — matching the convention
    used by :meth:`RunManifest.save` and :func:`emit_run_summary`.

    Idempotent: if a ``FileHandler`` with the target ``baseFilename`` is
    already attached (e.g. ``main()`` called twice in the same process,
    as happens in some test harnesses) this is a no-op. Returns the
    absolute log file path so the caller can record it in the run
    summary.
    """
    os.makedirs(date_dir, exist_ok=True)
    log_path = os.path.abspath(os.path.join(date_dir, _run_log_filename(started_at)))

    root = logging.getLogger()
    for h in root.handlers:
        if isinstance(h, logging.FileHandler) and getattr(h, "baseFilename", None) == log_path:
            return log_path  # already attached for this exact path

    fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    fh.setLevel(logging.INFO)
    # Use full ISO date+time in the file (the console keeps the short HH:MM:SS
    # format from basicConfig); a multi-day cron run's log file is unambiguous.
    fh.setFormatter(logging.Formatter(
        fmt="%(asctime)s %(levelname)s %(name)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    ))
    root.addHandler(fh)
    return log_path


# ── Checkpoint helpers ────────────────────────────────────────────────────────

def _write_tile_checkpoint(
    checkpoint_path: str,
    date: str,
    cal_date: str,
    completed: list[str],
    prior_failures: list[dict],
    current_failures: list[dict],
    cleared_ms_paths: list[str] | None = None,
) -> None:
    """Atomically write the tile checkpoint JSON.

    The checkpoint tracks both successes and failures so that chronic-offender
    MS files are visible across re-runs rather than silently re-attempted each
    time. Each failed[] entry carries a ``failure_count`` that increments on
    every repeat failure, supporting Batch E quarantine policy.

    Consecutive-failure semantics: a successful processing run for an MS
    drops it from the merged failures (via ``cleared_ms_paths``). That way
    ``failure_count`` truly counts *consecutive* failures — a fail / fail /
    succeed / fail sequence resets to count=1 after the success, instead of
    accumulating to count=3 and tripping the quarantine threshold spuriously.
    """
    cleared = set(cleared_ms_paths or [])
    failure_by_ms: dict[str, dict] = {}
    for rec in prior_failures:
        ms = rec.get("ms_path")
        if not ms or ms in cleared:
            continue
        # Ensure failure_count exists so older checkpoints upgrade in place
        rec = dict(rec)
        rec.setdefault("failure_count", 1)
        failure_by_ms[ms] = rec
    for rec in current_failures:
        ms = rec.get("ms_path")
        if not ms or ms in cleared:
            continue
        prior = failure_by_ms.get(ms)
        merged = dict(rec)
        if prior is not None:
            merged["failure_count"] = int(prior.get("failure_count", 1)) + 1
            # Preserve the first-failure timestamp for operator forensics
            merged["first_failed_at"] = prior.get(
                "first_failed_at", prior.get("failed_at")
            )
        else:
            merged["failure_count"] = 1
            merged["first_failed_at"] = merged.get("failed_at")
        failure_by_ms[ms] = merged
    payload = {
        "date": date,
        "cal_date": cal_date,
        "completed": completed,
        "failed": list(failure_by_ms.values()),
    }
    try:
        tmp = checkpoint_path + ".tmp"
        with open(tmp, "w") as ck_f:
            json.dump(payload, ck_f, indent=2)
        os.replace(tmp, checkpoint_path)
    except Exception as e:
        log.warning("Could not write checkpoint: %s", e)


def _should_skip_photometry(
    qa_verdict: str | None,
    skip_photometry_flag: bool,
    lenient_qa: bool,
) -> tuple[bool, str]:
    """Return ``(skip, reason)`` for the per-epoch forced-photometry gate.

    Default-strict policy: a QA-FAIL epoch is skipped unless the operator
    explicitly passes ``--lenient-qa``. This prevents bad-flux measurements
    from leaking into the master lightcurve table on unattended runs.

    Reasons:
    - ``"skip-photometry-flag"`` → operator passed ``--skip-photometry``
    - ``"qa-fail-default-strict"`` → QA-FAIL with no override
    - ``"lenient-qa-override"``    → QA-FAIL but ``--lenient-qa`` is set;
      caller should record a gate so verdict reflects the override.
    - ``""`` (empty)               → run photometry as normal
    """
    if skip_photometry_flag:
        return True, "skip-photometry-flag"
    if qa_verdict == "FAIL":
        if lenient_qa:
            return False, "lenient-qa-override"
        return True, "qa-fail-default-strict"
    return False, ""


def _epoch_should_rebuild(
    mosaic_path: str,
    prior_manifest,
    hour: int,
    force_recal: bool,
) -> bool:
    """Decide whether to (re)build an epoch mosaic.

    Rules:
    - ``force_recal``              → rebuild (caller handles stale-file cleanup)
    - mosaic file missing          → rebuild
    - no prior manifest            → skip (backward-compat; trust file)
    - prior verdict was ``PASS``   → skip (trust prior PASS mosaic)
    - prior verdict was ``FAIL`` /
      ``None`` (crashed mid-epoch) → rebuild (don't trust stale mosaic)
    """
    if force_recal:
        return True
    if not os.path.exists(mosaic_path):
        return True
    if prior_manifest is None:
        return False
    prior = prior_manifest.epoch_verdict(hour)
    # None = hour not recorded (crash before QA ran). FAIL = previously bad.
    # Anything else (PASS, WARN etc) is trusted.
    return prior in (None, "FAIL")


# ── Quarantine policy ────────────────────────────────────────────────────────

def _compute_quarantine_set(
    failures: list[dict],
    threshold: int,
) -> set[str]:
    """Return the set of ms_paths whose ``failure_count`` >= ``threshold``.

    These MS files are skipped by the orchestrator: they have failed at least
    ``threshold`` times in a row across previous runs, so retrying them is
    almost certainly a waste of compute and an entry in the failure log.
    A ``threshold <= 0`` disables quarantine entirely (returns an empty set).
    Quarantine is purely advisory — nothing is moved or deleted; operators can
    re-enable an MS by clearing its failure_count via ``--clear-quarantine``
    or by hand-editing the checkpoint.
    """
    if threshold <= 0:
        return set()
    return {
        rec["ms_path"]
        for rec in failures
        if isinstance(rec, dict)
        and rec.get("ms_path")
        and int(rec.get("failure_count", 0)) >= threshold
    }


def _clear_failure_counts(failures: list[dict]) -> list[dict]:
    """Return a copy of ``failures`` with every ``failure_count`` reset to 0.

    Used by ``--clear-quarantine``: the failure history is preserved (so
    operators can still see what happened) but the counts no longer trigger
    quarantine on the next run.
    """
    cleared = []
    for rec in failures:
        new_rec = dict(rec)
        new_rec["failure_count"] = 0
        cleared.append(new_rec)
    return cleared


# ── Dry-run plan ─────────────────────────────────────────────────────────────

def _collect_dry_run_plan(
    *,
    date: str,
    cal_date: str,
    obs_dec_deg: float | None,
    paths: dict,
    bp_table: str,
    g_table: str,
    ms_list_after_filters: list[str],
    epoch_hours: list[int],
    epoch_decisions: list[dict],
    prior_manifest_verdict: str | None,
    prior_manifest_present: bool,
    checkpoint_completed: int,
    checkpoint_failures: list[dict],
    quarantine_threshold: int,
    quarantine_set: set[str],
    skip_epoch_gaincal: bool,
    skip_photometry: bool,
    lenient_qa: bool,
) -> dict:
    """Build a structured plan dict describing what a normal run would do.

    Pure data; no I/O. The caller decides whether to print it (dry-run) or
    discard it. A dict (rather than a string) is returned so tests can assert
    on individual fields without parsing log output.
    """
    n_total = len(ms_list_after_filters)
    n_quarantined = sum(1 for ms in ms_list_after_filters if ms in quarantine_set)
    n_to_attempt = n_total - n_quarantined

    n_epoch_rebuild = sum(1 for d in epoch_decisions if d.get("action") == "rebuild")
    n_epoch_skip = sum(1 for d in epoch_decisions if d.get("action") == "skip")

    return {
        "date": date,
        "cal_date": cal_date,
        "obs_dec_deg": obs_dec_deg,
        "stage_dir": paths.get("stage_dir"),
        "products_dir": paths.get("products_dir"),
        "cal_tables": {
            "bp": bp_table,
            "g": g_table,
            "bp_exists": os.path.exists(bp_table) if bp_table else False,
            "g_exists": os.path.exists(g_table) if g_table else False,
        },
        "ms_files_after_filters": n_total,
        "epoch_hours": list(epoch_hours),
        "epoch_decisions": list(epoch_decisions),
        "prior_manifest_present": prior_manifest_present,
        "prior_manifest_verdict": prior_manifest_verdict,
        "checkpoint_completed_count": checkpoint_completed,
        "checkpoint_failed_count": len(checkpoint_failures),
        "quarantine_threshold": quarantine_threshold,
        "quarantine_count": n_quarantined,
        "quarantine_ms_paths": sorted(ms for ms in ms_list_after_filters if ms in quarantine_set),
        "phase0_gaincal": "skipped (--skip-epoch-gaincal)" if skip_epoch_gaincal else "would run",
        "phase1_tiles_to_attempt": n_to_attempt,
        "phase1_tiles_quarantined": n_quarantined,
        "phase2_epochs_to_rebuild": n_epoch_rebuild,
        "phase2_epochs_to_skip": n_epoch_skip,
        "phase3_photometry": (
            "skipped (--skip-photometry)" if skip_photometry
            else f"would run; QA gating={'lenient' if lenient_qa else 'strict'}"
        ),
    }


def _format_dry_run_plan(plan: dict) -> list[str]:
    """Render a dry-run plan dict as human-readable lines."""
    bp = plan["cal_tables"]
    lines: list[str] = [
        "=== DRY RUN — DSA-110 batch_pipeline ===",
        f"Date:           {plan['date']}",
        f"Cal date:       {plan['cal_date']}",
        f"Obs Dec:        {plan['obs_dec_deg']}°",
        f"Stage dir:      {plan['stage_dir']}",
        f"Products dir:   {plan['products_dir']}",
        f"Cal tables (BP): {bp['bp']}  [{'exists' if bp['bp_exists'] else 'MISSING — would generate'}]",
        f"Cal tables (G):  {bp['g']}  [{'exists' if bp['g_exists'] else 'MISSING — would generate'}]",
        f"MS files (post-filter): {plan['ms_files_after_filters']}",
        f"Epoch hours: {plan['epoch_hours']}",
        f"Prior manifest: {'present (verdict=' + str(plan['prior_manifest_verdict']) + ')' if plan['prior_manifest_present'] else 'absent'}",
        f"Checkpoint: completed={plan['checkpoint_completed_count']}  failed={plan['checkpoint_failed_count']}",
        f"Quarantine: {plan['quarantine_count']} MS (threshold={plan['quarantine_threshold']})",
    ]
    for ms in plan["quarantine_ms_paths"]:
        lines.append(f"  QUARANTINED: {ms}")
    lines.append("Resume plan:")
    for d in plan["epoch_decisions"]:
        lines.append(
            f"  hour {int(d['hour']):02d}: {d['action']} ({d.get('reason', '')})"
        )
    lines.extend([
        f"Phase 0 (gaincal):    {plan['phase0_gaincal']}",
        f"Phase 1 (per-tile):   would attempt {plan['phase1_tiles_to_attempt']} tiles "
        f"({plan['phase1_tiles_quarantined']} quarantined)",
        f"Phase 2 (mosaic):     {plan['phase2_epochs_to_rebuild']} rebuild, "
        f"{plan['phase2_epochs_to_skip']} skip",
        f"Phase 3 (photometry): {plan['phase3_photometry']}",
        "",
        "Pipeline NOT executed (--dry-run set). No products written.",
    ])
    return lines


# ── Dry-run orchestration ────────────────────────────────────────────────────

def _dry_run_main(args, date: str, cal_date: str, obs_dec_deg: float | None) -> None:
    """Read-only dry-run: build a plan and print it, no side effects.

    Bypasses ``ensure_bandpass`` (which generates cal tables) and uses only
    the filesystem-glob ``resolve_cal_table_paths`` to discover what cal
    tables would be used. Does not create products/stage directories or a
    run log file.
    """
    from dsa110_continuum.calibration.ensure import resolve_cal_table_paths

    paths = get_paths(date)
    bp_table, g_table = resolve_cal_table_paths(MS_DIR, cal_date)

    # Discover MS files using the same find + filter logic as a real run.
    # Importing here keeps the module-level import light for the regular path.
    import mosaic_day as _md  # type: ignore  (scripts/ is on sys.path)

    cfg = _md.TileConfig.build(
        date=date,
        cal_date=cal_date,
        image_dir=paths["stage_dir"],
        products_dir=paths["products_dir"],
    )
    try:
        ms_list = _md.find_valid_ms(cfg)
    except Exception as exc:
        log.error("Dry-run: could not enumerate MS files (%s)", exc)
        ms_list = []

    def _ms_ts(ms_path: str):
        ts_str = Path(ms_path).stem
        try:
            return datetime.strptime(ts_str, "%Y-%m-%dT%H:%M:%S")
        except ValueError:
            return None

    ms_list = [
        p for p in ms_list
        if (t := _ms_ts(p)) is not None and t.strftime("%Y-%m-%d") == date
    ]
    if args.start_hour is not None or args.end_hour is not None:
        ms_list = [
            p for p in ms_list
            if (t := _ms_ts(p)) is not None
            and (args.start_hour is None or t.hour >= args.start_hour)
            and (args.end_hour is None or t.hour < args.end_hour)
        ]

    epoch_hours = sorted({_ms_ts(p).hour for p in ms_list if _ms_ts(p) is not None})

    # Prior manifest + checkpoint (read-only)
    prior_manifest = try_load_prior_manifest(date, products_dir=PRODUCTS_BASE)
    checkpoint_path = os.path.join(paths["stage_dir"], ".tile_checkpoint.json")
    checkpoint_completed: list[str] = []
    checkpoint_failures: list[dict] = []
    if os.path.exists(checkpoint_path):
        try:
            with open(checkpoint_path) as f:
                ck = json.load(f)
            checkpoint_completed = list(ck.get("completed", []))
            checkpoint_failures = [
                rec for rec in ck.get("failed", [])
                if isinstance(rec, dict) and rec.get("ms_path")
            ]
        except Exception as exc:
            log.warning("Dry-run: could not read checkpoint (%s)", exc)

    # Apply --clear-quarantine to the in-memory copy so the dry-run plan
    # reflects what *would* happen with that flag — without writing.
    effective_failures = (
        _clear_failure_counts(checkpoint_failures)
        if args.clear_quarantine else checkpoint_failures
    )
    quarantine_set = _compute_quarantine_set(
        effective_failures, args.quarantine_after_failures,
    )

    # Per-epoch resume decisions
    epoch_decisions: list[dict] = []
    for hour in epoch_hours:
        mosaic_path = epoch_mosaic_path(paths, date, hour)
        rebuild = _epoch_should_rebuild(
            mosaic_path, prior_manifest, hour, args.force_recal,
        )
        prior_v = (
            None if prior_manifest is None else prior_manifest.epoch_verdict(hour)
        )
        if not rebuild:
            reason = f"prior verdict={prior_v}"
            action = "skip"
        elif args.force_recal:
            reason = "--force-recal"
            action = "rebuild"
        elif not os.path.exists(mosaic_path):
            reason = "no mosaic on disk"
            action = "rebuild"
        else:
            reason = f"prior verdict={prior_v}"
            action = "rebuild"
        epoch_decisions.append({"hour": hour, "action": action, "reason": reason})

    plan = _collect_dry_run_plan(
        date=date,
        cal_date=cal_date,
        obs_dec_deg=obs_dec_deg,
        paths=paths,
        bp_table=bp_table,
        g_table=g_table,
        ms_list_after_filters=ms_list,
        epoch_hours=epoch_hours,
        epoch_decisions=epoch_decisions,
        prior_manifest_verdict=(
            prior_manifest.pipeline_verdict if prior_manifest else None
        ),
        prior_manifest_present=prior_manifest is not None,
        checkpoint_completed=len(checkpoint_completed),
        checkpoint_failures=effective_failures,
        quarantine_threshold=args.quarantine_after_failures,
        quarantine_set=quarantine_set,
        skip_epoch_gaincal=args.skip_epoch_gaincal,
        skip_photometry=args.skip_photometry,
        lenient_qa=args.lenient_qa,
    )
    for line in _format_dry_run_plan(plan):
        log.info("%s", line)


# ── Epoch binning ─────────────────────────────────────────────────────────────

def bin_tiles_by_hour(tile_fits: list[str]) -> dict[int, list[str]]:
    """Group tile FITS paths by the UTC hour of their observation timestamp.

    Returns a dict mapping hour (0–23) → sorted list of tile paths.
    Tiles whose timestamp cannot be parsed are dropped with a warning.
    """
    epochs: dict[int, list[str]] = {}
    for path in tile_fits:
        dt = timestamp_from_fits(path)
        if dt is None:
            log.warning("Cannot parse timestamp from %s — skipping", Path(path).name)
            continue
        h = dt.hour
        epochs.setdefault(h, []).append(path)
    # Sort within each epoch (tiles arrive in time order within an hour)
    for h in epochs:
        epochs[h].sort()
    return epochs


def build_epoch_tile_sets(epochs: dict[int, list[str]]) -> list[tuple[int, list[str]]]:
    """Return list of (hour, tiles_with_overlap) in chronological order.

    Each epoch's tile list is:
        last 2 tiles of previous epoch  (if exists)
      + all tiles for this epoch
      + first 2 tiles of next epoch     (if exists)

    The epoch hour label is the start of the 1-hour window; the center is hour+0.5h.
    """
    sorted_hours = sorted(epochs.keys())
    result = []
    for i, h in enumerate(sorted_hours):
        tiles = list(epochs[h])  # core tiles

        # Previous-epoch overlap
        if i > 0:
            prev_tiles = epochs[sorted_hours[i - 1]]
            tiles = prev_tiles[-2:] + tiles

        # Next-epoch overlap
        if i < len(sorted_hours) - 1:
            next_tiles = epochs[sorted_hours[i + 1]]
            tiles = tiles + next_tiles[:2]

        result.append((h, tiles))
    return result


# ── Per-epoch mosaic writer (path-explicit version of mosaic_day.write_mosaic) ─

def write_epoch_mosaic(
    mosaic: np.ndarray,
    out_wcs: WCS,
    ref_fits_paths: list[str],
    out_path: str,
    date: str,
    hour: int,
    n_tiles: int,
    cal_date: str | None = None,
    cal_quality: dict | None = None,
    git_sha: str | None = None,
    cal_selection: dict | None = None,
) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with fits.open(ref_fits_paths[0]) as ref:
        ref_hdr = ref[0].header.copy()

    new_hdr = fits.Header()
    new_hdr["SIMPLE"] = True
    new_hdr["BITPIX"] = -32
    new_hdr["NAXIS"] = 2
    new_hdr["NAXIS1"] = mosaic.shape[1]
    new_hdr["NAXIS2"] = mosaic.shape[0]
    for key in ("BMAJ", "BMIN", "BPA", "BUNIT", "RESTFRQ", "EQUINOX"):
        if key in ref_hdr:
            new_hdr[key] = ref_hdr[key]
    new_hdr.update(out_wcs.to_header())
    new_hdr["HISTORY"] = (
        f"DSA-110 hourly mosaic {date}T{hour:02d}:00 UTC, {n_tiles} tiles (with overlap)"
    )

    # Provenance header cards
    if git_sha:
        new_hdr["PIPEVER"] = (git_sha, "Pipeline git commit")
    if cal_date:
        new_hdr["CALDATE"] = (cal_date, "Calibration table date")
    new_hdr["NTILES"] = (n_tiles, "Number of input tiles")
    if cal_quality:
        bp_q = cal_quality.get("bp", {})
        g_q = cal_quality.get("g", {})
        if "flag_fraction" in bp_q:
            new_hdr["BPFLAG"] = (round(bp_q["flag_fraction"], 4), "BP table flagged fraction")
        if "phase_scatter_deg" in g_q:
            new_hdr["GPHSCTR"] = (round(g_q["phase_scatter_deg"], 1), "Gain phase scatter [deg]")

    # Calibration selection provenance
    if cal_selection:
        bpcal = cal_selection.get("calibrator_name")
        if bpcal:
            new_hdr["BPCAL"] = (bpcal, "Bandpass calibrator name")
        calsrc = cal_selection.get("source")
        if calsrc:
            new_hdr["CALSRC"] = (calsrc, "Cal table source (generated/existing/borrowed)")
        caldec = cal_selection.get("obs_dec_deg_used")
        if caldec is not None:
            new_hdr["CALDEC"] = (round(caldec, 2), "Obs Dec used for cal selection [deg]")

    hdu = fits.PrimaryHDU(data=mosaic.astype(np.float32), header=new_hdr)
    hdu.writeto(out_path, overwrite=True)
    log.info("Epoch mosaic written: %s", out_path)


# ── Forced photometry (delegates to forced_photometry.run_forced_photometry) ──


# ── Mosaic stats helper ───────────────────────────────────────────────────────

def mosaic_stats(mosaic_path: str) -> tuple[float, float]:
    """Return (peak_jyb, rms_jyb) for a FITS mosaic."""
    with fits.open(mosaic_path) as hdul:
        data = hdul[0].data.squeeze()
    finite = data[np.isfinite(data)]
    peak = float(np.nanmax(data))
    rms = float(1.4826 * np.nanmedian(np.abs(finite - np.nanmedian(finite))))
    return peak, rms


# ── QA summary CSV ────────────────────────────────────────────────────────────

def write_qa_summary_row(
    date: str,
    epoch_label: str,
    mosaic_path: str,
    qa: EpochQAResult | None,
    gaincal_status: str,
) -> None:
    """Append one row to the QA summary CSV, creating the file if needed."""
    row = {
        "date": date,
        "epoch_utc": epoch_label,
        "mosaic_path": mosaic_path,
        "gaincal_used": gaincal_status,
    }
    if qa is not None:
        row.update(qa.to_dict())
    else:
        for field in QA_CSV_FIELDS:
            row.setdefault(field, "")

    file_exists = os.path.isfile(QA_SUMMARY_CSV)
    os.makedirs(os.path.dirname(QA_SUMMARY_CSV), exist_ok=True)
    try:
        with open(QA_SUMMARY_CSV, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=QA_CSV_FIELDS, extrasaction="ignore")
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)
    except Exception as e:
        log.warning("Could not write QA summary row: %s", e)


# ── Summary ───────────────────────────────────────────────────────────────────

def print_summary(date: str, epoch_results: list[dict]) -> None:
    """Print a per-epoch table plus overall totals."""
    print("\n" + "=" * 88)
    print(f"  DSA-110 Batch Pipeline Summary — {date}")
    print("=" * 88)
    hdr = (
        f"  {'Epoch':12s}  {'Tiles':>5}  {'GainCal':>8}"
        f"  {'Peak(Jy/b)':>10}  {'RMS(mJy/b)':>10}  {'Sources':>7}  {'DSA/Cat':>8}  {'QA':>4}"
    )
    print(hdr)
    print("  " + "-" * 84)

    all_ratios: list[float] = []
    for r in epoch_results:
        status = r.get("status", "ok")
        gcal_str = r.get("gaincal_status", "n/a")[:8]
        if status == "skipped":
            print(f"  {r['label']:12s}  {'--':>5}  {gcal_str:>8}  {'(skipped)':>10}")
            continue
        if status == "failed":
            print(f"  {r['label']:12s}  {r['n_tiles']:>5}  {gcal_str:>8}  {'FAILED':>10}")
            continue

        peak_str = f"{r['peak']:.4f}" if r['peak'] is not None else "n/a"
        rms_str = f"{r['rms']*1000:.2f}" if r['rms'] is not None else "n/a"
        src_str = str(r['n_sources']) if r['n_sources'] is not None else "n/a"
        ratio = r.get('median_ratio')
        ratio_str = f"{ratio:.3f}" if ratio is not None else "n/a"
        if ratio is not None:
            all_ratios.append(ratio)

        qa_str = r.get("qa_result") or "n/a"
        print(
            f"  {r['label']:12s}  {r['n_tiles']:>5}  {gcal_str:>8}"
            f"  {peak_str:>10}  {rms_str:>10}  {src_str:>7}  {ratio_str:>8}  {qa_str:>4}"
        )

    print("  " + "-" * 84)
    if all_ratios:
        overall = float(np.median(all_ratios))
        flag = "  OK" if 0.8 <= overall <= 1.2 else "  WARNING: outside 0.8–1.2 target"
        print(f"  Median DSA/Cat ratio across all epochs: {overall:.3f}{flag}")
    total_tiles = sum(r.get("n_tiles", 0) for r in epoch_results if r.get("status") != "skipped")
    n_epochs = len(epoch_results)
    n_skipped = sum(1 for r in epoch_results if r.get("status") == "skipped")
    n_failed = sum(1 for r in epoch_results if r.get("status") == "failed")
    print(f"  Epochs: {n_epochs} total, {n_skipped} skipped, {n_failed} failed")
    # QA aggregate counts (separate from execution success/failure)
    qa_pass = sum(1 for r in epoch_results if r.get("qa_result") == "PASS")
    qa_fail = sum(1 for r in epoch_results if r.get("qa_result") == "FAIL")
    qa_none = n_epochs - qa_pass - qa_fail
    qa_parts = [f"{qa_pass} QA-pass", f"{qa_fail} QA-fail"]
    if qa_none:
        qa_parts.append(f"{qa_none} QA-unavailable")
    print(f"  QA:     {', '.join(qa_parts)}")
    print("=" * 78 + "\n")



# ── Tile execution: timeout + retry ──────────────────────────────────────────

def _run_process_ms(
    ms_path: str,
    cfg_dict: dict,
    keep: bool,
    force_recal: bool = False,
) -> dict:
    """Thin wrapper so process_ms can be submitted to a subprocess pool.

    Accepts *cfg_dict* (a plain dict from TileConfig.to_dict()) so that
    configuration is explicitly passed to the subprocess rather than relying
    on module-global mutation.  Returns a dict (TileResult.to_dict()) for
    safe transport across the ProcessPoolExecutor boundary.
    """
    import mosaic_day as _md
    cfg = _md.TileConfig.from_dict(cfg_dict)
    result = _md.process_ms(ms_path, cfg, keep_intermediates=keep, force_recal=force_recal)
    return result.to_dict()


def _ms_is_valid(path: str) -> bool:
    """Return True only if *path* looks like a complete CASA Measurement Set.

    Mirrors mosaic_day._ms_is_valid.  A Measurement Set produced by a partial
    or failed HDF5→MS conversion will be missing ``table.dat`` or the
    ``table.f*`` data files — detecting this early avoids stalling an entire
    CASA applycal/wsclean subprocess pool on a broken input.
    """
    import glob as _g
    return (
        os.path.isdir(path)
        and os.path.exists(os.path.join(path, "table.dat"))
        and len(_g.glob(os.path.join(path, "table.f*"))) > 0
    )


def process_tile_safe(
    cfg_dict: dict,
    ms_path: str,
    keep: bool,
    timeout_sec: int,
    retry: bool,
    force_recal: bool = False,
):
    """Run process_ms with a hard timeout and optional single retry.

    If the tile hangs beyond *timeout_sec*, any CASA/WSClean subprocesses are
    killed with SIGKILL and a failed TileResult is returned.  With *retry=True*
    a second attempt is made after a 60-second cool-down.
    """
    import mosaic_day as _md
    tag = Path(ms_path).stem

    def _attempt() -> _md.TileResult:
        with ProcessPoolExecutor(max_workers=1) as pool:
            fut = pool.submit(_run_process_ms, ms_path, cfg_dict, keep, force_recal)
            try:
                result_dict = fut.result(timeout=timeout_sec)
                return _md.TileResult.from_dict(result_dict)
            except FuturesTimeoutError:
                log.error("[%s] TIMEOUT after %ds — killing CASA/WSClean", tag, timeout_sec)
                for pattern in ["applycal", "wsclean", "mpicasa"]:
                    subprocess.run(["pkill", "-9", "-f", pattern], capture_output=True)
                return _md.TileResult("failed", failed_stage="timeout",
                                      error=f"exceeded {timeout_sec}s")

    result = _attempt()
    if not result.ok and retry:
        log.warning("[%s] First attempt failed — waiting 60s then retrying once", tag)
        time.sleep(60)
        result = _attempt()
        if not result.ok:
            log.error("[%s] Retry also failed — skipping tile", tag)
    return result


# ── Dec-strip guard ───────────────────────────────────────────────────────────

def check_dec_strip(
    observed_dec: float,
    expected_dec: float,
    threshold_deg: float = 5.0,
) -> None:
    """Abort if the observed Dec strip differs from the expected calibration strip.

    DSA-110 observes at different declination strips on different nights. Calibration
    tables are strip-specific — applying tables from one strip to another silently
    produces near-zero flux (confirmed: median DSA/NVSS ≈ 0.06 for cross-strip runs).
    """
    delta = abs(observed_dec - expected_dec)
    if delta > threshold_deg:
        log.error(
            "ABORT: observed Dec %.1f° differs from expected %.1f° by %.1f° "
            "(threshold %.1f°). Cal tables were derived at Dec≈%.1f° — "
            "applying them at Dec≈%.1f° will produce invalid flux scale. "
            "Re-run with --expected-dec %.1f once cal tables for that strip exist.",
            observed_dec, expected_dec, delta, threshold_deg,
            expected_dec, observed_dec, observed_dec,
        )
        sys.exit(1)
    log.info(
        "Dec-strip check passed: observed %.1f° vs expected %.1f° (Δ=%.1f°)",
        observed_dec, expected_dec, delta,
    )


# ── Cal quality gate ──────────────────────────────────────────────────────────

def check_cal_gate(manifest: RunManifest, cal_date: str, date: str, strict: bool) -> None:
    """Check calibration quality and fire gate if thresholds are exceeded."""
    issues: list[str] = []
    g_q = manifest.cal_quality.get("g", {})
    bp_q = manifest.cal_quality.get("bp", {})
    if cal_date != date and g_q.get("phase_scatter_deg", 0) > 30.0:
        issues.append(
            f"Cross-date G table phase scatter {g_q['phase_scatter_deg']:.1f}\u00b0 > 30\u00b0 "
            f"(cal from {cal_date}, data from {date})"
        )
    for label, q in [("BP", bp_q), ("G", g_q)]:
        ff = q.get("flag_fraction", 0)
        if ff > 0.5:
            issues.append(f"{label} table {ff:.0%} flagged (> 50%)")
    if issues:
        for msg in issues:
            log.warning("CAL GATE: %s", msg)
        manifest.gates.append({"gate": "cal_quality", "verdict": "WARN", "reasons": issues})
        if strict:
            log.error("--strict-qa: aborting due to cal quality gate")
            sys.exit(1)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    _main_start = time.time()
    parser = argparse.ArgumentParser(
        description="Hourly-epoch mosaic pipeline for DSA-110 drift observations."
    )
    parser.add_argument("--date", default=DEFAULT_DATE, help="Observation date (YYYY-MM-DD)")
    parser.add_argument(
        "--cal-date",
        default=None,
        metavar="YYYY-MM-DD",
        help=(
            "Date whose calibration tables (BP/gain) to use. "
            "Defaults to --date if not provided. "
            "Use this when processing a new date whose cal tables are symlinked "
            "from 2026-01-25 (see CLAUDE.md)."
        ),
    )
    parser.add_argument(
        "--expected-dec",
        type=float,
        default=16.1,
        metavar="DEG",
        help=(
            "Expected pointing declination for this cal-table strip (default: 16.1°). "
            "Pipeline aborts if the first MS differs by more than 5° (DEC_CHANGE_THRESHOLD_DEG). "
            "Set this explicitly when processing a non-default Dec strip once cal tables exist."
        ),
    )
    parser.add_argument(
        "--keep-intermediates",
        action="store_true",
        default=False,
        help="Keep *_meridian.ms files (useful for debugging).",
    )
    parser.add_argument(
        "--skip-photometry",
        action="store_true",
        default=False,
        help="Skip forced photometry step.",
    )
    parser.add_argument(
        "--skip-epoch-gaincal",
        action="store_true",
        default=False,
        help=(
            "Skip per-epoch gain calibration. "
            "Falls back to the static daily G table (--cal-date). "
            "Use for debugging or when cal tables already exist."
        ),
    )
    parser.add_argument(
        "--start-hour",
        type=int,
        default=None,
        metavar="H",
        help="Only process MS files with timestamp >= this UTC hour (0–23). Default: all hours.",
    )
    parser.add_argument(
        "--end-hour",
        type=int,
        default=None,
        metavar="H",
        help="Only process MS files with timestamp < this UTC hour (0–23). Default: all hours.",
    )
    parser.add_argument(
        "--tile-timeout",
        type=int,
        default=TILE_TIMEOUT_SEC,
        metavar="SECONDS",
        help=f"Hard timeout per tile (applycal + WSClean). Default: {TILE_TIMEOUT_SEC}s (30 min). "
             "If a tile exceeds this, CASA/WSClean are killed and the tile is skipped (or retried).",
    )
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        default=False,
        help="Retry each failed tile once (60s cool-down between attempts). "
             "Useful for transient CASA crashes or memory pressure.",
    )
    parser.add_argument(
        "--force-recal",
        action="store_true",
        default=False,
        help=(
            "Force full re-calibration and re-imaging of every tile, even when FITS outputs "
            "already exist. Also clears the epoch_gaincal ap.G cache so the fallback check "
            "runs fresh. Use when re-running a date after code changes (e.g. BP-only fallback)."
        ),
    )
    parser.add_argument(
        "--strict-qa",
        action="store_true",
        default=False,
        help=(
            "Abort the whole run on cal-quality gate WARN. Independent of the "
            "default-strict per-epoch photometry skip on QA-FAIL (use "
            "--lenient-qa to override that)."
        ),
    )
    parser.add_argument(
        "--lenient-qa",
        action="store_true",
        default=False,
        help=(
            "Operator override: run forced photometry on QA-FAIL epochs even "
            "though they failed the three-gate epoch QA. Emits a 'lenient_qa' "
            "gate so the run finishes with pipeline_verdict=DEGRADED. Use only "
            "for investigative re-runs; the default is strict (skip)."
        ),
    )
    parser.add_argument(
        "--archive-all",
        action="store_true",
        default=False,
        help="Archive mosaics to products even if epoch QA fails.",
    )
    parser.add_argument(
        "--skip-auto-cal",
        action="store_true",
        default=False,
        help=(
            "Skip automatic bandpass table generation. "
            "Requires --cal-date to point at existing tables. "
            "Use when you want manual control over calibration."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help=(
            "Do not execute any pipeline stage; instead print a plan summary "
            "(MS files found, epoch resume decisions, quarantine state, what "
            "each phase would do) and exit 0. Reads prior manifest and "
            "checkpoint to inform the plan but writes nothing."
        ),
    )
    parser.add_argument(
        "--quarantine-after-failures",
        type=int,
        default=3,
        metavar="N",
        help=(
            "Quarantine an MS file after N consecutive failures across runs "
            "(default: 3). Quarantined MS are skipped without retry. Set 0 "
            "to disable. Quarantine is reversible via --clear-quarantine."
        ),
    )
    parser.add_argument(
        "--clear-quarantine",
        action="store_true",
        default=False,
        help=(
            "Reset failure_count to 0 for every MS in the checkpoint, "
            "releasing all quarantined files. Failure history (timestamps, "
            "errors) is preserved for diagnostics. Note: combining with "
            "--force-recal is a no-op for the clear because force-recal "
            "deletes the checkpoint first."
        ),
    )
    parser.add_argument(
        "--photometry-workers",
        type=int,
        default=1,
        metavar="N",
        help=(
            "Worker processes for the per-epoch forced photometry call "
            "(default: 1 = serial). Threaded through run_two_stage_parallel; "
            "see scripts/forced_photometry.py for benchmark numbers."
        ),
    )
    parser.add_argument(
        "--photometry-chunk-size",
        type=int,
        default=0,
        metavar="N",
        help=(
            "Sources per worker chunk when --photometry-workers > 1. "
            "0 = auto (~4 chunks per worker)."
        ),
    )
    args = parser.parse_args()

    date = args.date
    cal_date = args.cal_date if args.cal_date is not None else date

    # ── Determine observation declination ──────────────────────────────────
    # The Dec must be known before calibrator selection so that the bandpass
    # calibrator matches the science strip's beam response.
    from dsa110_continuum.calibration.dec_utils import read_ms_dec as _read_ms_dec
    _obs_dec: float | None = None
    # Robust to missing/unreadable MS_DIR: dry-run preflight on a fresh
    # workstation should produce a plan rather than crashing here.
    try:
        _ms_dir_listing = os.listdir(MS_DIR)
    except (FileNotFoundError, NotADirectoryError, PermissionError) as _ms_dir_err:
        log.warning("MS_DIR %s not accessible (%s) — using --expected-dec",
                    MS_DIR, _ms_dir_err)
        _ms_dir_listing = []
    _first_ms_list = sorted(
        f for f in _ms_dir_listing
        if f.endswith(".ms") and f.startswith(date) and "meridian" not in f
    )
    if _first_ms_list:
        try:
            _obs_dec = _read_ms_dec(os.path.join(MS_DIR, _first_ms_list[0]))
            check_dec_strip(_obs_dec, args.expected_dec)
        except RuntimeError as _e:
            log.warning("Could not determine observed Dec (%s) — using --expected-dec", _e)
            _obs_dec = args.expected_dec
    else:
        log.warning("No MS files found for %s yet — using --expected-dec for cal selection", date)
        _obs_dec = args.expected_dec
    log.info("Observation Dec for calibrator selection: %.1f°", _obs_dec)

    # ── Dry-run plan (Batch E) ───────────────────────────────────────────────
    # Read-only inspection: resolve existing cal tables (glob only — no
    # generation), find MS, consult prior manifest + checkpoint for resume
    # and quarantine state, print the plan, exit. No mkdir, no FileHandler,
    # no Phase 0/1/2/3 dispatch.
    if args.dry_run:
        _dry_run_main(args, date, cal_date, _obs_dec)
        return

    # ── Automatic bandpass table generation ────────────────────────────────
    _auto_cal_result = None
    if not args.skip_auto_cal:
        try:
            from dsa110_continuum.calibration.ensure import ensure_bandpass
            _auto_cal_result = ensure_bandpass(
                cal_date, ms_dir=MS_DIR, refant="103", obs_dec_deg=_obs_dec,
            )
            log.info(
                "Auto-cal: %s tables from %s (calibrator=%s, source=%s)",
                cal_date, _auto_cal_result.cal_date,
                _auto_cal_result.calibrator_name, _auto_cal_result.source,
            )
        except Exception as _acal_err:
            log.warning("Auto-cal failed (%s) — falling back to manual table lookup", _acal_err)

    # ── Cal-table validation ───────────────────────────────────────────────
    if _auto_cal_result is not None:
        _bp = _auto_cal_result.bp_table
        _ga = _auto_cal_result.g_table
    else:
        from dsa110_continuum.calibration.ensure import (
            resolve_cal_table_paths,
            validate_table_strip_compatibility,
        )
        _bp, _ga = resolve_cal_table_paths(MS_DIR, cal_date)
        # Apply the same strip compatibility policy as the auto-cal path
        try:
            validate_table_strip_compatibility(_bp, _obs_dec)
        except Exception as _strip_err:
            log.error("ABORT: strip compatibility check failed for fallback tables: %s", _strip_err)
            sys.exit(1)
    _missing = [t for t in [_bp, _ga] if not os.path.exists(t)]
    if _missing:
        for _t in _missing:
            log.error("ABORT: calibration table not found: %s", _t)
        log.error("Available .b tables in %s:", MS_DIR)
        for _f in sorted(os.listdir(MS_DIR)):
            if _f.endswith(".b"):
                log.error("  %s", _f)
        sys.exit(1)
    log.info("Cal tables verified for %s", cal_date)

    # ── Provenance manifest ──────────────────────────────────────────────
    manifest = RunManifest.start(date, cal_date)
    manifest.assess_cal_quality(_bp, _ga)

    # Record calibration selection provenance in manifest
    if _auto_cal_result is not None and _auto_cal_result.provenance:
        manifest.cal_selection = dict(_auto_cal_result.provenance)
    else:
        # Try loading from sidecar for manual/legacy table paths.
        # Overlay runtime source so the manifest reflects actual table origin,
        # not the original generation source recorded in the sidecar.
        from dsa110_continuum.calibration.ensure import load_provenance_sidecar
        _loaded_prov = load_provenance_sidecar(_bp)
        if _loaded_prov is not None:
            _runtime_source = "borrowed" if os.path.islink(_bp) else "existing"
            manifest.cal_selection = dict(_loaded_prov)
            manifest.cal_selection["source"] = _runtime_source
            manifest.cal_selection["bp_table"] = _bp
            manifest.cal_selection["g_table"] = _ga
            manifest.cal_selection["cal_date"] = cal_date

    # ── Cal quality gate ────────────────────────────────────────────────────
    check_cal_gate(manifest, cal_date, date, args.strict_qa)

    # ───────────────────────────────────────────────────────────────────────
    keep = args.keep_intermediates
    paths = get_paths(date)

    log.info("=== DSA-110 Batch Pipeline — %s ===", date)
    if cal_date != date:
        log.info("Calibration tables from: %s", cal_date)
    log.info("Stage dir:    %s", paths["stage_dir"])
    log.info("Products dir: %s", paths["products_dir"])

    os.makedirs(paths["stage_dir"], exist_ok=True)
    os.makedirs(paths["products_dir"], exist_ok=True)

    # ── Per-run log file ─────────────────────────────────────────────────────
    # Attach a FileHandler so the entire pipeline run lands in a single
    # diagnostic file under the date's products dir. All subsequent stages
    # (Phase 0/1/2/3) emit through it; the console handler from
    # logging.basicConfig() is preserved unchanged.
    _run_started_at = datetime.now(timezone.utc)
    _run_log_path = _attach_run_logfile(paths["products_dir"], _run_started_at)
    manifest.run_log = _run_log_path
    log.info("Run log: %s", _run_log_path)

    # ── Migrate stale qa_summary.csv if schema doesn't match ─────────────────
    if os.path.isfile(QA_SUMMARY_CSV):
        try:
            with open(QA_SUMMARY_CSV) as _f:
                existing_header = _f.readline().strip()
            expected_header = ",".join(QA_CSV_FIELDS)
            if existing_header != expected_header:
                archive_path = QA_SUMMARY_CSV + ".pre_phase0.bak"
                if not os.path.exists(archive_path):
                    shutil.copy2(QA_SUMMARY_CSV, archive_path)
                    log.info("Archived old qa_summary.csv to %s", archive_path)
                os.remove(QA_SUMMARY_CSV)
                log.info("Removed stale qa_summary.csv (old schema)")
        except Exception as e:
            log.warning("Could not check/migrate qa_summary.csv: %s", e)

    # ── Build immutable config for this run ─────────────────────────────────
    import mosaic_day as _md  # type: ignore  (scripts/ is on sys.path)

    cfg = _md.TileConfig.build(
        date=date,
        cal_date=cal_date,
        image_dir=paths["stage_dir"],
        products_dir=paths["products_dir"],
    )

    # ── Phase 1: Find + validate MS files ────────────────────────────────────
    ms_list = _md.find_valid_ms(cfg)
    if not ms_list:
        log.error("No valid MS files found for %s — aborting", date)
        sys.exit(1)
    log.info("Found %d valid MS files", len(ms_list))
    manifest.ms_files = list(ms_list)

    # Apply date filter (--date), then --start-hour / --end-hour
    def _ms_ts(ms_path: str):
        ts_str = Path(ms_path).stem  # e.g. 2026-01-25T21:17:33
        try:
            return datetime.strptime(ts_str, "%Y-%m-%dT%H:%M:%S")
        except ValueError:
            return None

    before = len(ms_list)
    ms_list = [p for p in ms_list if (t := _ms_ts(p)) is not None and t.strftime("%Y-%m-%d") == date]
    log.info("Date filter (%s): %d → %d MS files", date, before, len(ms_list))
    if not ms_list:
        log.error("No MS files for date %s — aborting", date)
        sys.exit(1)

    start_hour = args.start_hour
    end_hour = args.end_hour
    if start_hour is not None or end_hour is not None:
        before = len(ms_list)
        ms_list = [
            p for p in ms_list
            if (t := _ms_ts(p)) is not None
            and (start_hour is None or t.hour >= start_hour)
            and (end_hour is None or t.hour < end_hour)
        ]
        log.info(
            "Hour filter [%s, %s): %d → %d MS files",
            f"{start_hour:02d}" if start_hour is not None else "*",
            f"{end_hour:02d}" if end_hour is not None else "*",
            before,
            len(ms_list),
        )
        if not ms_list:
            log.error("No MS files remain after hour filter — aborting")
            sys.exit(1)

    # ── Phase 0: Per-epoch gain calibration ───────────────────────────────────
    epoch_gaincal_dir = os.path.join(paths["stage_dir"], "epoch_gaincal")
    os.makedirs(epoch_gaincal_dir, exist_ok=True)

    # --force-recal: purge cached gaincal products so the solve runs fresh
    if args.force_recal:
        import glob as _glob
        _cached_ap = _glob.glob(os.path.join(epoch_gaincal_dir, "*.ap.G"))
        for _f in _cached_ap:
            try:
                # shutil is imported at module scope (line 32); a redundant
                # local import here shadowed the name across all of main(),
                # causing UnboundLocalError at the qa_summary migration site
                # whenever a stale-schema CSV triggered shutil.copy2.
                shutil.rmtree(_f) if os.path.isdir(_f) else os.remove(_f)
                log.info("--force-recal: removed cached ap.G: %s", _f)
            except Exception as _e:
                log.warning("--force-recal: could not remove %s: %s", _f, _e)

        # Purge stale per-tile *_meridian.ms intermediate Measurement Sets so
        # that a corrupt MS cannot cause applycal to fail on the next run.
        _stale_meridian = _glob.glob(os.path.join(MS_DIR, f"{date}*_meridian.ms"))
        for _p in _stale_meridian:
            try:
                shutil.rmtree(_p)
                log.info("--force-recal: removed stale meridian MS: %s", _p)
            except Exception as _e:
                log.warning("--force-recal: could not remove meridian MS %s: %s", _p, _e)

        # Purge applycal sentinel files so calibration reruns from scratch
        _stale_sentinels = _glob.glob(os.path.join(MS_DIR, f"{date}*_meridian.ms.applycal_done"))
        for _s in _stale_sentinels:
            try:
                os.remove(_s)
                log.info("--force-recal: removed stale applycal sentinel: %s", _s)
            except Exception as _e:
                log.warning("--force-recal: could not remove sentinel %s: %s", _s, _e)

    _epoch_g_table: str | None = None
    _epoch_gaincal_reason: str | None = None
    if not args.skip_epoch_gaincal:
        log.info("=== Phase 0/3: Per-epoch gain calibration ===")
        try:
            from dsa110_continuum.calibration.epoch_gaincal import (
                EpochGaincalStatus,
                calibrate_epoch,
            )
            from dsa110_continuum.calibration.mosaic_constants import MOSAIC_TILE_COUNT
            _epoch_ms = ms_list[:MOSAIC_TILE_COUNT] if len(ms_list) >= MOSAIC_TILE_COUNT else ms_list
            if len(_epoch_ms) >= 2:
                _eg_result = calibrate_epoch(
                    epoch_ms_paths=_epoch_ms,
                    bp_table=_bp,
                    work_dir=epoch_gaincal_dir,
                    refant="103",
                )
                _epoch_g_table = _eg_result.g_table
                _epoch_gaincal_reason = _eg_result.reason
                if _eg_result.status == EpochGaincalStatus.SOLVED:
                    log.info("Epoch gaincal SUCCESS: %s", _epoch_g_table)
                    cfg = cfg.replace(g_table=_epoch_g_table)
                    _epoch_gaincal_status = "ok"
                elif _eg_result.status in (
                    EpochGaincalStatus.LOW_SNR,
                    EpochGaincalStatus.SOLVER_NO_TABLE,
                ):
                    log.warning(
                        "Epoch gaincal low-SNR fall-back (%s) — applying static daily G (%s)",
                        _eg_result.reason or _eg_result.status.value, _ga,
                    )
                    _epoch_gaincal_status = "low_snr"
                else:
                    log.warning(
                        "Epoch gaincal code-path fall-back (%s) — applying static daily G (%s)",
                        _eg_result.reason or _eg_result.status.value, _ga,
                    )
                    _epoch_gaincal_status = "fallback"
            else:
                _epoch_gaincal_reason = f"need at least 2 MS files, found {len(_epoch_ms)}"
                log.warning("Epoch gaincal skipped: %s", _epoch_gaincal_reason)
                _epoch_gaincal_status = "skipped"
        except Exception as _eg_exc:
            log.error("Epoch gaincal error: %s — using static table", _eg_exc)
            _epoch_gaincal_status = "error"
            _epoch_gaincal_reason = str(_eg_exc)
    else:
        log.info("--skip-epoch-gaincal set: using static daily G table (%s)", _ga)
        _epoch_gaincal_status = "skipped"
        _epoch_gaincal_reason = "operator passed --skip-epoch-gaincal"
    # ──────────────────────────────────────────────────────────────────────────

    manifest.gaincal_status = _epoch_gaincal_status
    manifest.epoch_g_table = _epoch_g_table

    # Gaincal fall-back/error is a degradation signal: record it as a gate so
    # the pipeline verdict reflects that the per-epoch phase solution was not
    # used. low_snr is operational (data limit, not bug); fallback is the
    # legacy code-path catch-all (now reserved for true exceptions). "skipped"
    # is intentional (--skip-epoch-gaincal or len<2) and does not degrade.
    if _epoch_gaincal_status in ("low_snr", "fallback", "error"):
        _verdict_map = {"low_snr": "LOW_SNR", "fallback": "FALLBACK", "error": "ERROR"}
        manifest.add_gate(
            gate="gaincal",
            verdict=_verdict_map[_epoch_gaincal_status],
            reason=(
                _epoch_gaincal_reason
                or f"epoch gaincal {_epoch_gaincal_status}; static daily G table used"
            ),
            static_g_table=_ga,
        )

    # ── Phase 1: Calibrate + image all tiles ──────────────────────────────────
    tile_timeout = args.tile_timeout
    retry_failed = args.retry_failed
    checkpoint_path = os.path.join(paths["stage_dir"], ".tile_checkpoint.json")

    # Consult the prior-run manifest (if any). Used for QA-aware epoch skip
    # (below) and to carry tile-failure history across re-runs.
    prior_manifest = (
        None if args.force_recal
        else try_load_prior_manifest(date, products_dir=PRODUCTS_BASE)
    )
    if prior_manifest is not None:
        log.info(
            "Prior manifest loaded: verdict=%s, %d epochs, %d tiles recorded",
            prior_manifest.pipeline_verdict or "?",
            len(prior_manifest.epochs),
            len(prior_manifest.tiles),
        )

    # --force-recal means "fresh rerun", so discard any old checkpoint state
    if args.force_recal and os.path.exists(checkpoint_path):
        try:
            os.remove(checkpoint_path)
            log.info("--force-recal: removed stale checkpoint: %s", checkpoint_path)
        except Exception as e:
            log.warning("--force-recal: could not remove checkpoint %s: %s", checkpoint_path, e)

    # Load completed tiles (and prior failure history) from a previous run.
    # Failures are preserved so operators can see chronic offenders across
    # re-runs; they do not prevent re-attempting the same MS.
    tile_fits: list[str] = []
    prior_tile_failures: list[dict] = []
    if os.path.exists(checkpoint_path):
        try:
            with open(checkpoint_path) as f:
                ck = json.load(f)
            tile_fits = [p for p in ck.get("completed", []) if os.path.exists(p)]
            prior_tile_failures = [
                rec for rec in ck.get("failed", [])
                if isinstance(rec, dict) and rec.get("ms_path")
            ]
            if tile_fits:
                log.info("Checkpoint: resuming with %d previously completed tiles", len(tile_fits))
            if prior_tile_failures:
                log.info(
                    "Checkpoint: %d tiles previously failed (will be re-attempted)",
                    len(prior_tile_failures),
                )
        except Exception as e:
            log.warning("Could not read checkpoint file: %s", e)

    completed_fits = set(tile_fits)
    current_tile_failures: list[dict] = []
    # MS paths that succeeded in *this* run; passed to _write_tile_checkpoint
    # so prior failure history is cleared on success and failure_count
    # measures consecutive failures only (Batch E.2 fix).
    current_completed_ms_paths: list[str] = []

    # ── Quarantine policy (Batch E) ──────────────────────────────────────────
    # --clear-quarantine zeros every failure_count before the threshold check
    # so the next run is allowed to retry every MS regardless of history.
    if args.clear_quarantine and prior_tile_failures:
        log.info("--clear-quarantine: zeroing failure_count for %d MS",
                 len(prior_tile_failures))
        prior_tile_failures = _clear_failure_counts(prior_tile_failures)
        # Persist the cleared counts immediately so even if the run aborts the
        # release survives.
        _write_tile_checkpoint(
            checkpoint_path, date, cal_date, tile_fits,
            prior_tile_failures, current_tile_failures,
        )
    quarantine_set = _compute_quarantine_set(
        prior_tile_failures, args.quarantine_after_failures,
    )
    if quarantine_set:
        log.warning(
            "Quarantine: %d MS skipped (>= %d consecutive failures). "
            "Run --clear-quarantine to re-enable.",
            len(quarantine_set), args.quarantine_after_failures,
        )
        manifest.add_gate(
            gate="quarantine",
            verdict="BLOCKED",
            reason=(
                f"{len(quarantine_set)} MS file(s) skipped after "
                f">={args.quarantine_after_failures} failures"
            ),
            quarantined_ms_paths=sorted(quarantine_set),
        )

    log.info("=== Phase 1/3: Calibrate + Image all tiles (timeout=%ds, retry=%s) ===",
             tile_timeout, retry_failed)
    n_imaged = n_skipped_tiles = n_failed_tiles = n_quarantined = 0

    for i, ms_path in enumerate(ms_list, 1):
        tag = Path(ms_path).stem
        log.info("[%d/%d] %s", i, len(ms_list), tag)

        # Quarantined MS: skip without retry (it has crossed the failure
        # threshold). Recorded in manifest so operators see what was skipped.
        if ms_path in quarantine_set:
            log.warning("  QUARANTINED — skipping %s", ms_path)
            n_quarantined += 1
            manifest.record_tile(
                ms_path, None, "quarantined",
                0.0, error="quarantined",
            )
            continue

        # Guard: skip corrupt or incomplete Measurement Sets before spawning
        # a subprocess.  A corrupt MS (missing MAIN table, incomplete write,
        # truncated HDF5→MS conversion) would stall CASA silently and waste
        # the tile_timeout budget.  Detect early and fail fast.
        if not _ms_is_valid(ms_path):
            log.error(
                "  SKIPPED: %s looks corrupt or incomplete (missing required MS tables)",
                ms_path,
            )
            n_failed_tiles += 1
            manifest.record_tile(
                ms_path, None, "failed",
                0.0, error="corrupt_ms_skipped",
            )
            current_tile_failures.append({
                "ms_path": ms_path,
                "error": "corrupt_ms_skipped",
                "elapsed_sec": 0.0,
                "failed_at": datetime.now(timezone.utc).isoformat(),
            })
            _write_tile_checkpoint(
                checkpoint_path, date, cal_date, tile_fits,
                prior_tile_failures, current_tile_failures,
                cleared_ms_paths=current_completed_ms_paths,
            )
            continue

        t0 = time.time()
        result = process_tile_safe(
            cfg.to_dict(), ms_path, keep, tile_timeout, retry_failed,
            force_recal=(args.force_recal or _epoch_gaincal_status == "ok"),
        )
        elapsed = time.time() - t0

        if not result.ok:
            err_msg = result.error or result.failed_stage
            log.error("  FAILED after %.0fs (%s: %s)", elapsed,
                       result.failed_stage, err_msg or "unknown")
            n_failed_tiles += 1
            manifest.record_tile(ms_path, None, "failed", elapsed, error=err_msg)
            current_tile_failures.append({
                "ms_path": ms_path,
                "error": err_msg,
                "elapsed_sec": round(elapsed, 1),
                "failed_at": datetime.now(timezone.utc).isoformat(),
            })
            _write_tile_checkpoint(
                checkpoint_path, date, cal_date, tile_fits,
                prior_tile_failures, current_tile_failures,
                cleared_ms_paths=current_completed_ms_paths,
            )
        else:
            if result.fits_path not in completed_fits:
                tile_fits.append(result.fits_path)
                completed_fits.add(result.fits_path)
            if result.status == "cached":
                n_skipped_tiles += 1
            else:
                log.info("  Done in %.0fs → %s", elapsed, Path(result.fits_path).name)
                n_imaged += 1
            manifest.record_tile(ms_path, result.fits_path, "ok", elapsed)
            # Track the MS as cleared so prior failure history (from a previous
            # run) drops out of the merged failures, resetting the consecutive
            # count to zero.
            current_completed_ms_paths.append(ms_path)

            _write_tile_checkpoint(
                checkpoint_path, date, cal_date, tile_fits,
                prior_tile_failures, current_tile_failures,
                cleared_ms_paths=current_completed_ms_paths,
            )

    log.info(
        "Tiles: %d imaged, %d already done, %d failed, %d quarantined",
        n_imaged, n_skipped_tiles, n_failed_tiles, n_quarantined,
    )

    if len(tile_fits) < 2:
        log.error("Too few tiles to mosaic (%d) — aborting", len(tile_fits))
        sys.exit(1)

    # ── Phase 3: Bin tiles into 1-hour epochs with overlap ────────────────────
    log.info("=== Phase 2/3: Build hourly-epoch mosaics ===")
    by_hour = bin_tiles_by_hour(tile_fits)
    epoch_sets = build_epoch_tile_sets(by_hour)  # [(hour, [tile_paths...]), ...]

    log.info(
        "Epochs: %d  (hours: %s)",
        len(epoch_sets),
        ", ".join(f"{h:02d}h" for h, _ in epoch_sets),
    )

    epoch_results: list[dict] = []

    for hour, epoch_tiles in epoch_sets:
        label = f"{date}T{hour:02d}00"
        n_core = len(by_hour.get(hour, []))
        n_overlap = len(epoch_tiles) - n_core
        log.info(
            "--- Epoch %s: %d core + %d overlap = %d tiles ---",
            label, n_core, n_overlap, len(epoch_tiles),
        )

        mosaic_path = epoch_mosaic_path(paths, date, hour)
        phot_csv_path = epoch_phot_path(paths, date, hour)
        mosaic_fits_dst = Path(paths["products_dir"]) / Path(mosaic_path).name

        # Decide (rebuild vs skip) based on prior-run verdict. A mosaic file
        # that exists but whose prior QA verdict was FAIL (or absent, meaning
        # the prior run crashed mid-epoch) is NOT trusted — it gets rebuilt.
        should_rebuild = _epoch_should_rebuild(
            mosaic_path, prior_manifest, hour, args.force_recal,
        )

        # Remove stale epoch-level outputs before rebuilding so the fresh
        # mosaic / photometry CSV is unambiguous. --force-recal and a FAIL
        # prior verdict both use the same cleanup path.
        if should_rebuild and os.path.exists(mosaic_path):
            prior_v = None if prior_manifest is None else prior_manifest.epoch_verdict(hour)
            reason = "--force-recal" if args.force_recal else f"prior verdict={prior_v!s}"
            for stale_path in (mosaic_path, phot_csv_path, str(mosaic_fits_dst)):
                if os.path.exists(stale_path):
                    try:
                        os.remove(stale_path)
                        log.info("  %s: removed stale output %s", reason, stale_path)
                    except Exception as e:
                        log.warning("  %s: could not remove %s: %s", reason, stale_path, e)

        if not should_rebuild:
            prior_v = None if prior_manifest is None else prior_manifest.epoch_verdict(hour)
            log.info("  Mosaic already exists (prior verdict=%s) — skipping epoch %s",
                     prior_v or "no-manifest", label)
            epoch_results.append({
                "label": label, "status": "skipped", "n_tiles": len(epoch_tiles),
                "gaincal_status": _epoch_gaincal_status,
                "qa_result": prior_v,  # carry prior verdict forward
            })
            continue

        # Build mosaic
        try:
            out_wcs, ny, nx = _md.build_common_wcs(epoch_tiles)
            mosaic_arr = _md.coadd_tiles(epoch_tiles, out_wcs, ny, nx)
            write_epoch_mosaic(
                mosaic_arr, out_wcs, epoch_tiles, mosaic_path, date, hour, len(epoch_tiles),
                cal_date=cal_date, cal_quality=manifest.cal_quality, git_sha=manifest.git_sha,
                cal_selection=manifest.cal_selection or None,
            )
        except Exception as e:
            log.error("  Mosaic failed for epoch %s: %s", label, e)
            epoch_results.append({"label": label, "status": "failed", "n_tiles": len(epoch_tiles), "gaincal_status": _epoch_gaincal_status})
            continue

        # QA
        _md.check_mosaic_quality(mosaic_path)
        peak, rms = mosaic_stats(mosaic_path)
        log.info("  Peak: %.4f Jy/beam  RMS: %.2f mJy/beam  DR: %.0f", peak, rms * 1000, peak / rms if rms else 0)

        # ── Epoch QA (three-gate) — run BEFORE photometry ─────────────────────
        epoch_qa: EpochQAResult | None = None
        try:
            epoch_qa = measure_epoch_qa(mosaic_path)
            log.info(
                "  Epoch QA: ratio=%.3f [%s] | compl=%.1f%% [%s] | rms=%.1f mJy [%s] → %s",
                epoch_qa.median_ratio, epoch_qa.ratio_gate,
                epoch_qa.completeness_frac * 100, epoch_qa.completeness_gate,
                epoch_qa.mosaic_rms_mjy, epoch_qa.rms_gate,
                epoch_qa.qa_result,
            )
        except Exception as e:
            log.warning("  Epoch QA failed: %s", e)

        # ── Update FITS header with QA results ────────────────────────────────
        if epoch_qa is not None and os.path.exists(mosaic_path):
            try:
                with fits.open(mosaic_path, mode="update") as hdul:
                    hdr = hdul[0].header
                    hdr["QARESULT"] = (epoch_qa.qa_result, "Epoch QA verdict")
                    hdr["QARMS"] = (round(epoch_qa.mosaic_rms_mjy, 2), "Mosaic RMS [mJy/beam]")
                    hdr["QARAT"] = (round(epoch_qa.median_ratio, 4), "Median DSA/catalog flux ratio")
            except Exception as e:
                log.warning("  Could not update FITS header with QA: %s", e)

        # ── Forced photometry — default-strict on QA-FAIL ────────────────────
        # Default policy: do NOT run photometry on a QA-FAIL epoch (would
        # leak bad fluxes into the master lightcurve table). --lenient-qa
        # is the explicit operator override for investigation.
        n_sources: int | None = None
        median_ratio: float | None = None
        qa_verdict = epoch_qa.qa_result if epoch_qa else None
        skip_phot, skip_reason = _should_skip_photometry(
            qa_verdict, args.skip_photometry, args.lenient_qa,
        )

        if skip_reason == "qa-fail-default-strict":
            log.warning(
                "  QA FAIL — skipping photometry for epoch %s (use --lenient-qa to override)",
                label,
            )
        elif skip_reason == "lenient-qa-override":
            log.warning(
                "  --lenient-qa: running photometry on QA-FAIL epoch %s "
                "(verdict will be DEGRADED)",
                label,
            )
            manifest.add_gate(
                gate="lenient_qa",
                verdict="OVERRIDE",
                reason=f"photometry ran on QA-FAIL epoch {label} via --lenient-qa",
                epoch_label=label,
            )

        if not skip_phot:
            try:
                from forced_photometry import run_forced_photometry
                phot_result = run_forced_photometry(
                    mosaic_path, output_csv=phot_csv_path, min_flux_mjy=10.0,
                    workers=args.photometry_workers,
                    chunk_size=(args.photometry_chunk_size or None),
                )
                n_sources = phot_result["n_sources"]
                median_ratio = phot_result["median_ratio"]
                if np.isfinite(median_ratio):
                    log.info("  Median DSA/Cat ratio: %.3f  (%d sources)", median_ratio, n_sources)
            except Exception as e:
                log.error("  Forced photometry failed for epoch %s: %s", label, e)
                manifest.add_gate(
                    gate="photometry",
                    verdict="FAILED",
                    reason=f"forced photometry crashed for epoch {label}: {e}",
                    epoch_label=label,
                )

        # ── Diagnostic PNG ────────────────────────────────────────────────────
        if epoch_qa is not None:
            diag_png = mosaic_path.replace(".fits", "_qa_diag.png")
            try:
                tile_rms_list = []
                for tp in epoch_tiles:
                    if os.path.exists(tp):
                        _, trms = mosaic_stats(tp)
                        tile_rms_list.append(trms * 1000.0)
                plot_epoch_qa(
                    epoch_qa,
                    epoch_qa.ratios or [],
                    tile_rms_list,
                    diag_png,
                    epoch_label=label,
                )
                log.info("  QA diagnostic PNG: %s", diag_png)
            except Exception as e:
                log.warning("  Could not generate QA PNG: %s", e)

        # ── QA summary CSV row ────────────────────────────────────────────────
        try:
            write_qa_summary_row(date, label, mosaic_path, epoch_qa, _epoch_gaincal_status)
        except Exception as e:
            log.warning("  Could not write QA summary: %s", e)

        # ── Archive gate — skip archive for QA-FAIL unless --archive-all ──────
        mosaic_fits_src = Path(mosaic_path)
        should_archive = qa_verdict != "FAIL" or args.archive_all
        if not should_archive:
            log.warning("  QA FAIL — mosaic NOT archived to products: %s", label)
            manifest.gates.append({"gate": "archive", "verdict": "BLOCKED", "reason": f"epoch {label} QA FAIL"})
        elif mosaic_fits_src.exists() and (not mosaic_fits_dst.exists() or args.force_recal):
            shutil.copy2(str(mosaic_fits_src), str(mosaic_fits_dst))
            log.info("Archived mosaic FITS: %s", mosaic_fits_dst)

        epoch_results.append({
            "label": label,
            "status": "ok",
            "n_tiles": len(epoch_tiles),
            "n_core": n_core,
            "n_overlap": n_overlap,
            "peak": peak,
            "rms": rms,
            "n_sources": n_sources,
            "median_ratio": median_ratio,
            "mosaic_path": mosaic_path,
            "gaincal_status": _epoch_gaincal_status,
            "qa_result": epoch_qa.qa_result if epoch_qa else None,
        })

    # ── Record epochs in manifest ────────────────────────────────────────────
    for er in epoch_results:
        manifest.record_epoch(
            hour=int(er["label"].split("T")[1][:2]),
            epoch_result=er,
        )

    # ── Print summary ─────────────────────────────────────────────────────────
    _wall = time.time() - _main_start
    print_summary(date, epoch_results)

    manifest.finalize(_wall)
    manifest.save(paths["products_dir"])

    emit_run_summary(
        date, cal_date, epoch_results, _wall,
        products_dir=paths["products_dir"],
        run_log_path=_run_log_path,
    )

    # Static run report (Batch F). A failure to render must not fail the
    # actual run — operators still have manifest.json + run_summary.json.
    try:
        from dsa110_continuum.qa.run_report import write_run_report
        write_run_report(manifest, paths["products_dir"])
    except Exception as _report_err:
        log.warning("Run report render failed (non-fatal): %s", _report_err)

    _emit_promotion_record(manifest, paths, args)


def emit_run_summary(
    date: str,
    cal_date: str,
    epoch_results: list,
    wall_time_sec: float,
    products_dir: str | None = None,
    run_log_path: str | None = None,
) -> None:
    """Write run summary JSON to products dir and symlink at /tmp for backward compat.

    ``run_log_path`` is the absolute path of the per-run log file created by
    :func:`_attach_run_logfile`; if provided it is recorded under the
    ``run_log`` key so an operator inspecting the summary can locate the
    diagnostic log without searching.
    """
    import json as _json
    from datetime import datetime as _dt

    epochs_list = epoch_results
    n_exec_ok = sum(1 for v in epoch_results if v.get("status") == "ok")
    n_exec_fail = sum(1 for v in epoch_results if v.get("status") != "ok")
    n_qa_pass = sum(1 for v in epoch_results if v.get("qa_result") == "PASS")
    n_qa_fail = sum(1 for v in epoch_results if v.get("qa_result") == "FAIL")

    payload = {
        "date": date,
        "cal_date": cal_date,
        "finished_at": _dt.utcnow().isoformat() + "Z",
        "wall_time_sec": round(wall_time_sec),
        "n_epochs": len(epoch_results),
        "n_pass": n_exec_ok,
        "n_fail": n_exec_fail,
        "n_qa_pass": n_qa_pass,
        "n_qa_fail": n_qa_fail,
        "run_log": run_log_path,
        "epochs": epochs_list,
    }

    # Write to products dir if available, otherwise /tmp
    if products_dir:
        os.makedirs(products_dir, exist_ok=True)
        summary_path = os.path.join(products_dir, f"{date}_run_summary.json")
    else:
        summary_path = "/tmp/pipeline_last_run.json"

    with open(summary_path, "w") as _f:
        _json.dump(payload, _f, indent=2)

    # Backward-compat symlink at /tmp
    tmp_link = "/tmp/pipeline_last_run.json"
    if summary_path != tmp_link:
        try:
            if os.path.islink(tmp_link) or os.path.exists(tmp_link):
                os.remove(tmp_link)
            os.symlink(summary_path, tmp_link)
        except OSError:
            pass  # best-effort symlink

    log.info(
        "Run complete — date=%s cal=%s  epochs=%d  exec_ok=%d exec_fail=%d"
        "  qa_pass=%d qa_fail=%d  wall=%.0fm  → %s",
        date, cal_date, len(epoch_results), n_exec_ok, n_exec_fail,
        n_qa_pass, n_qa_fail, wall_time_sec / 60, summary_path,
    )

    notify_url = os.environ.get("DSA_NOTIFY_URL")
    if notify_url:
        try:
            import requests as _req
            _req.post(notify_url, json=payload, timeout=10)
        except Exception:
            pass  # notification is best-effort


def _emit_promotion_record(manifest, paths: dict, args) -> None:
    """Auto-emit per-(date, hour) promotion side-car JSON and ledger row.

    Non-fatal: a writer failure must not fail an otherwise completed run.
    The manifest and run summary remain the source of truth.
    """
    try:
        from dsa110_continuum.qa.promotion import emit_for_run

        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        products_root = os.path.dirname(os.path.dirname(paths["products_dir"]))
        emit_for_run(
            manifest,
            paths["products_dir"],
            repo_root,
            cli_invocation=list(sys.argv),
            skip_epoch_gaincal=bool(getattr(args, "skip_epoch_gaincal", False)),
            products_root=products_root,
        )
    except Exception as exc:
        log.warning("Promotion auto-emit failed (non-fatal): %s", exc)


if __name__ == "__main__":
    main()
