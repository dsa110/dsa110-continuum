"""Automatic promotion-record emission.

Implements the writer side of the spec at
``docs/validation/pipeline-validation-from-scratch.md``: at the end of a
``batch_pipeline.py`` run, derives a per-``(date, hour)`` promotion record
and writes it as a side-car JSON in the products directory plus a row in
the repo-tracked promotion ledger.

Photometric-anchor evaluation is operator-authored. The auto-emitted record
captures everything the manifest knows — calibration tier, epoch gaincal
state, gate snapshot, paths — and sets ``promotion_class`` to
``"auto_emitted_pending_review"`` until the operator finalizes the anchor
block. This keeps the emit non-promotional by default while removing
operator burden for the auditable fields.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)

EPOCH_GAINCAL_STATES = (
    "solved",
    "fell_back_to_static_with_reason",
    "skipped_intentionally",
    "skipped_or_failed_low_snr",
    "unknown",
)
DAILY_CAL_TIERS = ("A", "B", "C", "unknown")
PROMOTION_CLASSES = (
    "trusted_baseline",
    "comparator_only",
    "weak_baseline",
    "auto_emitted_pending_review",
)

LEDGER_RELPATH = "docs/validation/promotion-log.md"
LEDGER_HEADER = (
    "| date       | hour | class                          | tier | epoch_gc                          "
    "| anchor                                                | side-car (relative to products root)               "
    "| git_sha  |\n"
    "|------------|------|--------------------------------|------|------------------------------------"
    "|--------------------------------------------------------|----------------------------------------------------"
    "|----------|\n"
)


def derive_daily_cal_tier(cal_selection: dict[str, Any] | None) -> str:
    """Map ``RunManifest.cal_selection`` to a spec daily-cal tier.

    The current manifest records ``source`` as one of ``{"generated",
    "existing", "borrowed"}``. ``"generated"`` and ``"existing"`` are
    same-date table outcomes (tier A). ``"borrowed"`` represents
    cross-date table reuse and is treated as tier C — a conservative
    classification that lumps tier-B (BP same-date, G borrowed) into
    tier-C since the spec only gates trusted-baseline on tier-A.
    """
    if not cal_selection:
        return "unknown"
    source = cal_selection.get("source")
    if source in ("generated", "existing"):
        return "A"
    if source == "borrowed":
        return "C"
    return "unknown"


def derive_epoch_gaincal_state(
    legacy_status: str | None,
    skip_intentionally: bool,
) -> str:
    """Map ``RunManifest.gaincal_status`` to the spec's four-state enum.

    The legacy field overloads ``"skipped"`` for two genuinely different
    cases (operator passed ``--skip-epoch-gaincal`` vs. auto-skip because
    the epoch slice had fewer than two MS). The caller passes
    ``skip_intentionally`` to disambiguate.
    """
    if not legacy_status:
        return "unknown"
    if legacy_status == "ok":
        return "solved"
    if legacy_status in ("fallback", "error"):
        return "fell_back_to_static_with_reason"
    if legacy_status == "skipped":
        return "skipped_intentionally" if skip_intentionally else "skipped_or_failed_low_snr"
    return "unknown"


def _anchor_has_any_fired(anchor: dict[str, Any] | None) -> bool:
    if not anchor:
        return False
    return any(anchor.get(k) for k in ("primary_model", "catalog_xmatch", "tile_self_consistency"))


def derive_promotion_class(
    tier: str,
    gaincal_state: str,
    anchor: dict[str, Any] | None,
) -> str:
    """Compute the spec's promotion class for an emit-time record.

    The pipeline does not auto-fill the photometric anchor block — that
    evaluation is operator-authored per the spec. So an emit-time record
    with no anchor returns ``"auto_emitted_pending_review"`` regardless
    of cal tier or gaincal state. Once the operator fills the anchor
    block, calling this helper again classifies into the spec's named
    states.
    """
    if not _anchor_has_any_fired(anchor):
        return "auto_emitted_pending_review"
    if tier == "A" and gaincal_state == "solved":
        return "trusted_baseline"
    return "comparator_only"


def _eligibility_reason(tier: str, gaincal_state: str, gates_with_fail: list[str]) -> str | None:
    """Return None when eligible for trusted_baseline, else a short reason."""
    if tier != "A":
        return f"daily_cal_tier={tier!r} (trusted_baseline requires 'A')"
    if gaincal_state != "solved":
        return f"epoch_gaincal_state={gaincal_state!r} (trusted_baseline requires 'solved')"
    if gates_with_fail:
        return f"manifest gates failed: {', '.join(gates_with_fail)}"
    return None


def build_promotion_record(
    manifest: Any,
    hour: int,
    products_dir: str,
    *,
    cli_invocation: list[str] | None = None,
    skip_epoch_gaincal: bool = False,
) -> dict[str, Any]:
    """Build a schema-compliant promotion record dict for one ``(date, hour)``.

    Reads only ``manifest`` attributes already populated during a normal
    ``batch_pipeline.py`` run. The ``anchor`` block is left empty; the
    operator fills it in before finalizing promotion.
    """
    epoch_rec: dict[str, Any] | None = None
    for rec in getattr(manifest, "epochs", []) or []:
        if rec.get("hour") == hour:
            epoch_rec = rec
            break

    tier = derive_daily_cal_tier(getattr(manifest, "cal_selection", None) or None)
    gc_state = derive_epoch_gaincal_state(
        getattr(manifest, "gaincal_status", None),
        skip_intentionally=bool(skip_epoch_gaincal),
    )
    gates_failed = [
        str(g.get("gate"))
        for g in (getattr(manifest, "gates", []) or [])
        if str(g.get("verdict", "")).upper() in ("FAIL", "FALLBACK", "ERROR")
    ]
    elig_reason = _eligibility_reason(tier, gc_state, gates_failed)
    eligible = elig_reason is None
    promotion_class = derive_promotion_class(tier, gc_state, anchor=None)

    cal_selection = getattr(manifest, "cal_selection", None) or {}
    record: dict[str, Any] = {
        "date": getattr(manifest, "date", ""),
        "hour": int(hour),
        "promoted_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": getattr(manifest, "git_sha", ""),
        "operator": "auto",
        "notes": "auto-emitted at end of batch_pipeline run; awaiting operator anchor evaluation",
        "manifest_path": os.path.join(products_dir, f"{manifest.date}_manifest.json"),
        "run_log_path": getattr(manifest, "run_log", None),
        "run_summary_path": os.path.join(products_dir, f"{manifest.date}_run_summary.json"),
        "mosaic_path": (epoch_rec or {}).get("mosaic_path"),
        "batch_pipeline_invocation": list(cli_invocation or getattr(manifest, "command_line", []) or []),
        "manifest_verdict": getattr(manifest, "pipeline_verdict", "") or None,
        "pipeline_verdict": getattr(manifest, "pipeline_verdict", "") or None,
        "daily_cal_tier": tier,
        "cal_provenance": {
            "bp": getattr(manifest, "bp_table", "") or None,
            "g": getattr(manifest, "g_table", "") or None,
            "borrowed_from": cal_selection.get("borrowed_from"),
        },
        "epoch_gaincal_state": gc_state,
        "epoch_gaincal_reason": _gaincal_reason(manifest),
        "anchor": {
            "primary_model": None,
            "catalog_xmatch": None,
            "tile_self_consistency": None,
        },
        "eligible_for_trusted_baseline": eligible,
        "eligible_for_trusted_baseline_reason": elig_reason,
        "promotion_class": promotion_class,
        "_emit_metadata": {
            "schema_version": 1,
            "emitter": "dsa110_continuum.qa.promotion",
            "epoch_record_n_tiles": (epoch_rec or {}).get("n_tiles"),
            "epoch_record_status": (epoch_rec or {}).get("status"),
            "epoch_record_qa_result": (epoch_rec or {}).get("qa_result"),
        },
    }
    return record


def _gaincal_reason(manifest: Any) -> str | None:
    """Surface the gaincal fallback reason from the gate entry, if any."""
    for g in getattr(manifest, "gates", []) or []:
        if g.get("gate") == "gaincal":
            v = str(g.get("verdict", "")).upper()
            if v in ("FALLBACK", "ERROR"):
                return str(g.get("reason", "")) or None
    return None


def sidecar_path(products_dir: str, date: str, hour: int) -> str:
    """Return the canonical sidecar path for a ``(date, hour)`` window."""
    return os.path.join(products_dir, f"promotion_{date}T{hour:02d}00.json")


def write_promotion_sidecar(
    manifest: Any,
    hour: int,
    products_dir: str,
    *,
    cli_invocation: list[str] | None = None,
    skip_epoch_gaincal: bool = False,
) -> str:
    """Write a promotion sidecar JSON for one ``(date, hour)`` window.

    Returns the path written. Does not raise on common runtime
    irregularities — the caller is expected to wrap in try/except for
    absolute non-fatality.
    """
    os.makedirs(products_dir, exist_ok=True)
    record = build_promotion_record(
        manifest,
        hour,
        products_dir,
        cli_invocation=cli_invocation,
        skip_epoch_gaincal=skip_epoch_gaincal,
    )
    path = sidecar_path(products_dir, manifest.date, hour)
    with open(path, "w") as f:
        json.dump(record, f, indent=2, default=str)
    logger.info("Promotion sidecar written: %s", path)
    return path


def _ledger_row(record: dict[str, Any], side_car_relpath: str) -> str:
    anchor_summary = "—"
    a = record.get("anchor") or {}
    if a.get("primary_model"):
        anchor_summary = f"primary_model:{a['primary_model']}"
    elif a.get("catalog_xmatch"):
        cx = a["catalog_xmatch"]
        anchor_summary = f"catalog_xmatch_{cx.get('catalog')}_n={cx.get('n')}"
    elif a.get("tile_self_consistency"):
        ts = a["tile_self_consistency"]
        anchor_summary = f"tile_self_consistency_n={ts.get('n_overlaps')}"
    return (
        f"| {record['date']} | {record['hour']:02d}   "
        f"| {record['promotion_class']:<30} "
        f"| {record['daily_cal_tier']:<4} "
        f"| {record['epoch_gaincal_state']:<34} "
        f"| {anchor_summary:<54} "
        f"| {side_car_relpath:<50} "
        f"| {record['git_sha']:<8} |\n"
    )


def append_promotion_ledger(
    repo_root: str,
    record: dict[str, Any],
    side_car_path: str,
    products_root: str | None = None,
    ledger_relpath: str = LEDGER_RELPATH,
) -> str:
    """Append one row to the repo-tracked markdown ledger.

    Creates the file with a header on first write. Idempotent: if a row
    for the same (date, hour, git_sha) already exists, no row is added.
    """
    ledger_path = os.path.join(repo_root, ledger_relpath)
    os.makedirs(os.path.dirname(ledger_path), exist_ok=True)
    if products_root and side_car_path.startswith(products_root):
        side_car_relpath = os.path.relpath(side_car_path, products_root)
    else:
        side_car_relpath = side_car_path
    row = _ledger_row(record, side_car_relpath)
    dedup_key = f"| {record['date']} | {record['hour']:02d}   "
    if os.path.exists(ledger_path):
        with open(ledger_path) as f:
            existing = f.read()
        # Skip if a row for the same (date, hour) and same git_sha already exists.
        for line in existing.splitlines():
            if line.startswith(dedup_key) and f"| {record['git_sha']:<8} |" in line:
                logger.info("Ledger row already present for %s hour %02d at sha %s — skipping",
                            record["date"], record["hour"], record["git_sha"])
                return ledger_path
        with open(ledger_path, "a") as f:
            f.write(row)
    else:
        with open(ledger_path, "w") as f:
            f.write("# Promotion ledger\n\n")
            f.write(
                "Auto-appended by `dsa110_continuum.qa.promotion.append_promotion_ledger`. "
                "One row per `(date, hour)` window at end-of-run. Operator finalizes "
                "`promotion_class` after evaluating the photometric anchor block in the "
                "side-car JSON.\n\n"
            )
            f.write(LEDGER_HEADER)
            f.write(row)
    logger.info("Ledger row appended: %s", ledger_path)
    return ledger_path


def emit_for_run(
    manifest: Any,
    products_dir: str,
    repo_root: str,
    *,
    cli_invocation: list[str] | None = None,
    skip_epoch_gaincal: bool = False,
    products_root: str | None = None,
) -> list[str]:
    """End-of-run helper that emits one promotion sidecar per recorded epoch.

    Writes one sidecar per recorded epoch hour and appends a ledger row for
    each. Returns the list of sidecar paths written.

    This function is meant to be called from ``scripts/batch_pipeline.py``
    after ``manifest.save()``. It is non-fatal — raises only on
    catastrophic IO failures the caller should surface.
    """
    written: list[str] = []
    for rec in getattr(manifest, "epochs", []) or []:
        hour = rec.get("hour")
        if hour is None:
            continue
        try:
            path = write_promotion_sidecar(
                manifest,
                int(hour),
                products_dir,
                cli_invocation=cli_invocation,
                skip_epoch_gaincal=skip_epoch_gaincal,
            )
            written.append(path)
            record = build_promotion_record(
                manifest,
                int(hour),
                products_dir,
                cli_invocation=cli_invocation,
                skip_epoch_gaincal=skip_epoch_gaincal,
            )
            append_promotion_ledger(
                repo_root,
                record,
                path,
                products_root=products_root,
            )
        except Exception as exc:
            logger.warning("Promotion emit failed for %s hour %s: %s",
                           getattr(manifest, "date", "?"), hour, exc)
    return written
