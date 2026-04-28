"""Static Markdown run report (Batch F).

Generates ``{products_dir}/{date}/run_report.md`` summarising one batch
pipeline run from the saved :class:`RunManifest`. Operators get one
human-readable file describing what happened, what passed QA, what was
quarantined, and where the diagnostic artifacts live — without having
to cross-reference manifest.json + run_summary.json + the run log.

Design:
- :func:`render_run_report` is pure (manifest in → markdown string out)
  so tests don't need a filesystem.
- :func:`write_run_report` is a thin I/O wrapper.
- Robust to missing optional fields (older manifests, incomplete runs):
  rendering never raises on absent keys; sections degrade to "(none)" or
  "(not recorded)" rather than failing.
- Paths in the rendered output are absolute so the report can be opened
  from anywhere and the links still resolve.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from dsa110_continuum.qa.provenance import RunManifest

logger = logging.getLogger(__name__)


# ─── Public API ─────────────────────────────────────────────────────────────


def render_run_report(manifest: RunManifest, date_dir: str) -> str:
    """Return the rendered Markdown report for *manifest*.

    Pure function: no I/O, no global state read. The caller decides whether
    to write the result to disk.

    Parameters
    ----------
    manifest : RunManifest
        Finalised manifest (post-:meth:`RunManifest.finalize`).
    date_dir : str
        Per-date products directory (already date-nested, e.g.
        ``/data/products/2026-01-25/``); used only to derive sibling
        artifact paths (manifest JSON, run summary JSON, photometry CSVs).

    Returns
    -------
    str
        Markdown text, terminated by a trailing newline.
    """
    sections: list[str] = []
    sections.append(_render_header(manifest))
    sections.append(_render_artifacts(manifest, date_dir))
    sections.append(_render_tile_summary(manifest))
    sections.append(_render_epoch_summary(manifest))
    sections.append(_render_gates(manifest))
    sections.append(_render_qa_fail_photometry_note(manifest))
    sections.append(_render_quarantine(manifest))
    sections.append(_render_failed_tiles(manifest))
    sections.append(_render_photometry(manifest, date_dir))
    sections.append(_render_diagnostic_plots(manifest))
    return "\n\n".join(sections) + "\n"


def write_run_report(
    manifest: RunManifest,
    date_dir: str,
    *,
    filename: str = "run_report.md",
) -> str:
    """Render and write the report directly into *date_dir*.

    *date_dir* must already be the date-nested products directory (the same
    path that ``RunManifest.save`` and ``emit_run_summary`` write into).
    Returns the absolute path of the written file.
    """
    os.makedirs(date_dir, exist_ok=True)
    out_path = os.path.abspath(os.path.join(date_dir, filename))
    text = render_run_report(manifest, date_dir)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text)
    logger.info("Wrote run report: %s", out_path)
    return out_path


# ─── Section renderers ──────────────────────────────────────────────────────
#
# Each renderer takes the manifest and returns one Markdown section. Empty /
# missing data degrades to a single line saying "(none)" or "(not recorded)"
# rather than producing zero output, so the section count and ordering stay
# stable across runs.


def _render_header(manifest: RunManifest) -> str:
    verdict = manifest.pipeline_verdict or "(unfinished)"
    wall = manifest.wall_time_sec
    wall_str = f"{wall / 60:.1f} min" if wall else "(unknown)"
    git = manifest.git_sha or "(unknown)"
    started = manifest.started_at or "(unknown)"
    finished = manifest.finished_at or "(unfinished)"
    return "\n".join([
        f"# DSA-110 Run Report — {manifest.date}",
        "",
        f"- **Pipeline verdict:** `{verdict}`",
        f"- **Calibration date:** {manifest.cal_date or '(unknown)'}",
        f"- **Started:** {started}",
        f"- **Finished:** {finished}",
        f"- **Wall time:** {wall_str}",
        f"- **Git:** `{git}`",
        f"- **Gaincal status:** {manifest.gaincal_status or '(not recorded)'}",
    ])


def _render_artifacts(manifest: RunManifest, date_dir: str) -> str:
    manifest_path = os.path.join(date_dir, f"{manifest.date}_manifest.json")
    summary_path = os.path.join(date_dir, f"{manifest.date}_run_summary.json")
    run_log = manifest.run_log or "(not recorded)"
    return "\n".join([
        "## Artifacts",
        "",
        f"- Run log: `{run_log}`",
        f"- Manifest: `{manifest_path}`",
        f"- Run summary: `{summary_path}`",
    ])


def _render_tile_summary(manifest: RunManifest) -> str:
    counts: dict[str, int] = {}
    for t in manifest.tiles:
        counts[t.get("status", "unknown")] = counts.get(t.get("status", "unknown"), 0) + 1
    if not counts:
        return "## Tile summary\n\n(no tiles recorded)"
    lines = ["## Tile summary", "", "| Status | Count |", "|---|---:|"]
    # Sorted for deterministic output
    for status in sorted(counts):
        lines.append(f"| {status} | {counts[status]} |")
    return "\n".join(lines)


def _render_epoch_summary(manifest: RunManifest) -> str:
    if not manifest.epochs:
        return "## Epoch summary\n\n(no epochs recorded)"
    lines = [
        "## Epoch summary",
        "",
        "| Hour | Status | QA | n_tiles | Peak (Jy/beam) | RMS (mJy/beam) | Sources | Mosaic |",
        "|---:|---|---|---:|---:|---:|---:|---|",
    ]
    for ep in sorted(manifest.epochs, key=lambda e: e.get("hour", -1)):
        hour = ep.get("hour")
        peak = ep.get("peak")
        rms = ep.get("rms")
        rms_mjy = (rms * 1000.0) if isinstance(rms, (int, float)) else None
        mosaic = ep.get("mosaic_path") or "(none)"
        lines.append(
            f"| {hour:02d} | {ep.get('status', '?')} | "
            f"{ep.get('qa_result') or '?'} | "
            f"{_fmt_int(ep.get('n_tiles'))} | "
            f"{_fmt_float(peak, 4)} | "
            f"{_fmt_float(rms_mjy, 2)} | "
            f"{_fmt_int(ep.get('n_sources'))} | "
            f"`{mosaic}` |"
        )
    return "\n".join(lines)


def _render_gates(manifest: RunManifest) -> str:
    if not manifest.gates:
        return "## QA gates triggered\n\nNo gates triggered."
    lines = [
        "## QA gates triggered",
        "",
        "| Gate | Verdict | Reason |",
        "|---|---|---|",
    ]
    for g in manifest.gates:
        gate = str(g.get("gate", "?"))
        verdict = str(g.get("verdict", "?"))
        reason = str(g.get("reason", g.get("reasons", "(no reason)")))
        # Markdown-table-safe: collapse newlines, escape pipes
        reason = reason.replace("|", "\\|").replace("\n", " ")
        lines.append(f"| {gate} | {verdict} | {reason} |")
    return "\n".join(lines)


def _render_qa_fail_photometry_note(manifest: RunManifest) -> str:
    """Surface QA-FAIL epochs whose photometry was skipped by default-strict gating.

    Cross-checks the lenient_qa gate: if no lenient_qa gate was recorded, any
    epoch with qa_result=FAIL had its photometry skipped by Batch B's
    default-strict policy. This section makes that explicit so operators
    don't have to mentally compose two pieces of state.
    """
    fail_epochs = [
        ep for ep in manifest.epochs if ep.get("qa_result") == "FAIL"
    ]
    if not fail_epochs:
        return "## QA-FAIL epochs (photometry skipped)\n\n(none)"
    lenient_used = any(g.get("gate") == "lenient_qa" for g in manifest.gates)
    lines = ["## QA-FAIL epochs (photometry skipped)", ""]
    if lenient_used:
        lines.append(
            "*Note:* `--lenient-qa` was used; photometry ran on these epochs "
            "despite QA-FAIL verdicts. See gate `lenient_qa` above."
        )
        lines.append("")
    for ep in sorted(fail_epochs, key=lambda e: e.get("hour", -1)):
        hour = ep.get("hour")
        mosaic = ep.get("mosaic_path") or "(no mosaic path)"
        action = "ran via --lenient-qa" if lenient_used else "skipped (default-strict)"
        lines.append(f"- **Hour {hour:02d}** — QA verdict FAIL; photometry {action}.")
        lines.append(f"  Mosaic: `{mosaic}`")
    return "\n".join(lines)


def _render_quarantine(manifest: RunManifest) -> str:
    quarantine_gates = [g for g in manifest.gates if g.get("gate") == "quarantine"]
    quarantined_paths: list[str] = []
    for g in quarantine_gates:
        quarantined_paths.extend(g.get("quarantined_ms_paths", []))
    if not quarantined_paths:
        return "## Quarantined MS\n\n(none)"
    lines = ["## Quarantined MS", "",
             f"{len(quarantined_paths)} MS file(s) skipped after meeting the failure threshold:",
             ""]
    for ms in sorted(quarantined_paths):
        lines.append(f"- `{ms}`")
    lines.append("")
    lines.append(
        "Re-enable with: `batch_pipeline.py --date <date> --clear-quarantine`"
    )
    return "\n".join(lines)


def _render_failed_tiles(manifest: RunManifest) -> str:
    failed = [t for t in manifest.tiles if t.get("status") == "failed"]
    if not failed:
        return "## Failed tiles\n\n(none)"
    lines = [
        "## Failed tiles",
        "",
        "| MS | Error | Elapsed (s) |",
        "|---|---|---:|",
    ]
    for t in failed:
        ms = t.get("ms_path", "(unknown)")
        err = str(t.get("error", "(no error)")).replace("|", "\\|").replace("\n", " ")
        elapsed = t.get("elapsed_sec")
        lines.append(f"| `{ms}` | {err} | {_fmt_float(elapsed, 1)} |")
    return "\n".join(lines)


def _render_photometry(manifest: RunManifest, date_dir: str) -> str:
    """List per-epoch photometry CSV paths for epochs that actually got measured."""
    rows: list[tuple[int, int | None, str]] = []
    for ep in manifest.epochs:
        hour = ep.get("hour")
        n_sources = ep.get("n_sources")
        if hour is None or n_sources is None:
            continue
        csv_path = os.path.join(
            date_dir, f"{manifest.date}T{int(hour):02d}00_forced_phot.csv",
        )
        rows.append((int(hour), int(n_sources), csv_path))
    if not rows:
        return "## Forced photometry\n\n(no photometry results recorded)"
    lines = ["## Forced photometry", ""]
    for hour, n, path in sorted(rows):
        lines.append(f"- Hour {hour:02d}: **{n}** sources → `{path}`")
    return "\n".join(lines)


def _render_diagnostic_plots(manifest: RunManifest) -> str:
    """Link per-epoch QA diagnostic PNGs derived from each mosaic path."""
    plots: list[tuple[int, str]] = []
    for ep in manifest.epochs:
        mosaic = ep.get("mosaic_path")
        hour = ep.get("hour")
        if not mosaic or hour is None:
            continue
        # Mirror the orchestrator's derivation (batch_pipeline.py: mosaic_path.replace(".fits", "_qa_diag.png"))
        png = mosaic.replace(".fits", "_qa_diag.png")
        plots.append((int(hour), png))
    if not plots:
        return "## Diagnostic plots\n\n(none)"
    lines = ["## Diagnostic plots", ""]
    for hour, png in sorted(plots):
        lines.append(f"- Hour {hour:02d}: `{png}`")
    return "\n".join(lines)


# ─── Formatting helpers ─────────────────────────────────────────────────────


def _fmt_int(v: Any) -> str:
    if v is None:
        return "—"
    try:
        return str(int(v))
    except (TypeError, ValueError):
        return "—"


def _fmt_float(v: Any, ndigits: int = 2) -> str:
    if v is None:
        return "—"
    try:
        f = float(v)
    except (TypeError, ValueError):
        return "—"
    if f != f:  # NaN
        return "—"
    return f"{f:.{ndigits}f}"
