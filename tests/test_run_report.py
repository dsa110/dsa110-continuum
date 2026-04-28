"""Tests for Batch F static run report.

Behavior under test:
- Required H2 sections all render in a fixed order, regardless of data.
- Verdict / wall-time / artifact paths surface from the manifest cleanly.
- Gates render in tabular form; QA-FAIL epochs surface explicitly with
  an action note that switches based on whether --lenient-qa was used.
- Quarantined MS appear in the dedicated section, with operator help text.
- Missing optional fields (run_log absent, no epochs, no tiles, NaN values)
  do not crash rendering — the report degrades to "(none)" / "—".
- ``write_run_report`` writes to ``{products_dir}/{date}/run_report.md``.
"""

from __future__ import annotations

import math

from dsa110_continuum.qa.provenance import RunManifest
from dsa110_continuum.qa.run_report import (
    render_run_report,
    write_run_report,
)


def _make_clean_manifest() -> RunManifest:
    m = RunManifest.start("2026-01-25", "2026-01-25")
    m.run_log = "/data/products/2026-01-25/run_2026-04-27T14_00_00Z.log"
    m.gaincal_status = "ok"
    m.bp_table = "/stage/cal/2026-01-25.b"
    m.g_table = "/stage/cal/2026-01-25.g"
    # Two successful tiles
    m.record_tile("/stage/ms/a.ms", "/stage/img/a.fits", "ok", 142.3)
    m.record_tile("/stage/ms/b.ms", "/stage/img/b.fits", "ok", 130.1)
    # Two clean epochs
    m.record_epoch(2, {
        "n_tiles": 11, "status": "ok",
        "mosaic_path": "/stage/img/2026-01-25T0200_mosaic.fits",
        "peak": 0.523, "rms": 0.0012, "n_sources": 1234,
        "qa_result": "PASS",
    })
    m.record_epoch(22, {
        "n_tiles": 11, "status": "ok",
        "mosaic_path": "/stage/img/2026-01-25T2200_mosaic.fits",
        "peak": 12.5, "rms": 0.001, "n_sources": 4567,
        "qa_result": "PASS",
    })
    m.finalize(1820.4)
    return m


# ─── Required sections always present ────────────────────────────────────────


def test_all_required_h2_sections_present():
    """Every section heading exists, in the documented order."""
    m = _make_clean_manifest()
    text = render_run_report(m, "/data/products/2026-01-25")

    required_headings = [
        "## Artifacts",
        "## Tile summary",
        "## Epoch summary",
        "## QA gates triggered",
        "## QA-FAIL epochs (photometry skipped)",
        "## Quarantined MS",
        "## Failed tiles",
        "## Forced photometry",
        "## Diagnostic plots",
    ]
    last = -1
    for h in required_headings:
        idx = text.find(h)
        assert idx > -1, f"Section missing: {h}"
        assert idx > last, f"Section out of order: {h}"
        last = idx


def test_header_includes_verdict_and_run_log():
    m = _make_clean_manifest()
    text = render_run_report(m, "/data/products/2026-01-25")

    assert "# DSA-110 Run Report — 2026-01-25" in text
    assert "**Pipeline verdict:** `CLEAN`" in text
    assert m.run_log in text


def test_clean_run_says_no_gates():
    m = _make_clean_manifest()
    text = render_run_report(m, "/data/products/2026-01-25")
    assert "No gates triggered." in text


def test_clean_run_qa_fail_section_says_none():
    m = _make_clean_manifest()
    text = render_run_report(m, "/data/products/2026-01-25")
    # Section header present, body says (none)
    assert "## QA-FAIL epochs (photometry skipped)" in text
    qa_section = text.split("## QA-FAIL")[1].split("## Quarantined")[0]
    assert "(none)" in qa_section


# ─── DEGRADED with gates ─────────────────────────────────────────────────────


def test_gate_renders_in_table_with_reason():
    m = _make_clean_manifest()
    m.add_gate("photometry", "FAILED",
               "forced photometry crashed for epoch 2026-01-25T22: ZeroDivisionError",
               epoch_label="2026-01-25T22")
    # Must re-finalize so verdict updates
    m.finalize(1820.4)
    text = render_run_report(m, "/data/products/2026-01-25")
    assert "**Pipeline verdict:** `DEGRADED`" in text
    assert "| photometry | FAILED |" in text
    assert "ZeroDivisionError" in text


def test_gate_with_pipe_in_reason_is_escaped():
    """Reasons containing literal '|' must not break the markdown table."""
    m = _make_clean_manifest()
    m.add_gate("custom", "WARN", "thing|other thing")
    m.finalize(1.0)
    text = render_run_report(m, "/data/products/2026-01-25")
    # The pipe inside the reason was escaped, so the table row has exactly
    # the right number of unescaped pipes (4: leading, after gate, after
    # verdict, trailing)
    rows = [
        ln for ln in text.splitlines()
        if ln.startswith("| custom |")
    ]
    assert len(rows) == 1
    # Escaped pipe present
    assert r"thing\|other" in rows[0]


# ─── QA-FAIL epoch surfacing ─────────────────────────────────────────────────


def test_qa_fail_epoch_listed_as_default_strict_skipped():
    m = _make_clean_manifest()
    # Tweak hour-22 epoch to FAIL
    m.epochs[1]["qa_result"] = "FAIL"
    m.finalize(1.0)
    text = render_run_report(m, "/data/products/2026-01-25")
    # The QA-FAIL section calls out hour 22 with the "skipped (default-strict)" wording
    section = text.split("## QA-FAIL")[1].split("##")[0]
    assert "Hour 22" in section
    assert "default-strict" in section
    assert "lenient" not in section.lower() or "lenient-qa" in section.lower()


def test_qa_fail_epoch_with_lenient_gate_says_ran():
    """If --lenient-qa was used, the section reflects that the photometry ran."""
    m = _make_clean_manifest()
    m.epochs[1]["qa_result"] = "FAIL"
    m.add_gate("lenient_qa", "OVERRIDE",
               "photometry ran on QA-FAIL epoch 2026-01-25T22 via --lenient-qa",
               epoch_label="2026-01-25T22")
    m.finalize(1.0)
    text = render_run_report(m, "/data/products/2026-01-25")
    section = text.split("## QA-FAIL")[1].split("## Quarantined")[0]
    assert "ran via --lenient-qa" in section
    assert "default-strict" not in section


# ─── Quarantine ──────────────────────────────────────────────────────────────


def test_quarantine_section_lists_paths_and_release_help():
    m = _make_clean_manifest()
    m.add_gate(
        "quarantine", "BLOCKED",
        "2 MS file(s) skipped after >=3 failures",
        quarantined_ms_paths=["/stage/ms/bad1.ms", "/stage/ms/bad2.ms"],
    )
    m.finalize(1.0)
    text = render_run_report(m, "/data/products/2026-01-25")
    section = text.split("## Quarantined")[1].split("## Failed")[0]
    assert "/stage/ms/bad1.ms" in section
    assert "/stage/ms/bad2.ms" in section
    # Operator-facing help text on how to release
    assert "--clear-quarantine" in section


def test_quarantine_section_says_none_when_empty():
    m = _make_clean_manifest()
    text = render_run_report(m, "/data/products/2026-01-25")
    section = text.split("## Quarantined")[1].split("## Failed")[0]
    assert "(none)" in section


# ─── Tile summary, failed tiles, photometry ─────────────────────────────────


def test_tile_summary_counts_by_status():
    m = _make_clean_manifest()
    m.record_tile("/ms/c.ms", None, "failed", 1800.0, error="timeout")
    m.record_tile("/ms/d.ms", None, "quarantined", 0.0, error="quarantined")
    text = render_run_report(m, "/data/products/2026-01-25")
    section = text.split("## Tile summary")[1].split("## Epoch summary")[0]
    # Status counts present
    assert "| ok | 2 |" in section
    assert "| failed | 1 |" in section
    assert "| quarantined | 1 |" in section


def test_failed_tiles_table_lists_errors():
    m = _make_clean_manifest()
    m.record_tile("/ms/bad.ms", None, "failed", 1800.0, error="casa_hang")
    text = render_run_report(m, "/data/products/2026-01-25")
    section = text.split("## Failed tiles")[1].split("## Forced")[0]
    assert "/ms/bad.ms" in section
    assert "casa_hang" in section


def test_photometry_section_lists_csv_paths():
    m = _make_clean_manifest()
    text = render_run_report(m, "/data/products/2026-01-25")
    section = text.split("## Forced photometry")[1].split("## Diagnostic")[0]
    assert "Hour 02" in section
    assert "1234" in section
    assert "/data/products/2026-01-25/2026-01-25T0200_forced_phot.csv" in section  # date_dir/{date}T... — no extra date nesting


def test_diagnostic_plots_derived_from_mosaic_paths():
    m = _make_clean_manifest()
    text = render_run_report(m, "/data/products/2026-01-25")
    section = text.split("## Diagnostic plots")[1]
    # mosaic_path .fits → _qa_diag.png
    assert "/stage/img/2026-01-25T0200_mosaic_qa_diag.png" in section
    assert "/stage/img/2026-01-25T2200_mosaic_qa_diag.png" in section


# ─── Robustness to missing/sparse fields ────────────────────────────────────


def test_missing_run_log_says_not_recorded():
    m = RunManifest.start("2026-01-25", "2026-01-25")  # no run_log set
    m.finalize(1.0)
    text = render_run_report(m, "/data/products/2026-01-25")
    assert "Run log: `(not recorded)`" in text


def test_no_epochs_no_tiles_renders_cleanly():
    m = RunManifest.start("2026-01-25", "2026-01-25")
    m.finalize(0.0)
    text = render_run_report(m, "/data/products/2026-01-25")
    # All required sections still appear with degraded content
    assert "(no tiles recorded)" in text
    assert "(no epochs recorded)" in text
    assert "No gates triggered." in text


def test_nan_peak_or_rms_does_not_crash():
    m = RunManifest.start("2026-01-25", "2026-01-25")
    m.record_epoch(2, {
        "n_tiles": 11, "status": "ok",
        "mosaic_path": "/m.fits",
        "peak": float("nan"), "rms": float("nan"),
        "n_sources": None,  # also missing
        "qa_result": None,
    })
    m.finalize(1.0)
    text = render_run_report(m, "/data/products/2026-01-25")
    assert "## Epoch summary" in text
    # NaN / None render as em-dash, not as "nan"
    section = text.split("## Epoch summary")[1].split("## QA gates")[0]
    assert "—" in section


def test_legacy_manifest_without_run_log_field_renders():
    """A manifest loaded from an older save (no run_log attribute) still renders."""
    m = RunManifest.start("2026-01-25", "2026-01-25")
    # Simulate truly-missing attribute by deleting it (older codepath)
    # — note: the attribute exists with value None by default, which is the
    # back-compat path covered in test_batch_e1_hygiene.py. This tests the
    # render path for that scenario.
    m.run_log = None
    m.finalize(0.5)
    text = render_run_report(m, "/data/products/2026-01-25")
    assert "(not recorded)" in text


# ─── write_run_report I/O ───────────────────────────────────────────────────


def test_write_run_report_creates_file_at_expected_path(tmp_path):
    """write_run_report writes directly into date_dir (no extra date nesting)."""
    m = _make_clean_manifest()
    date_dir = tmp_path / "2026-01-25"
    out = write_run_report(m, str(date_dir))
    expected = date_dir / "run_report.md"
    assert out == str(expected.resolve())
    assert expected.exists()
    contents = expected.read_text()
    assert "# DSA-110 Run Report — 2026-01-25" in contents
    # Trailing newline (POSIX file convention)
    assert contents.endswith("\n")
    # Regression guard: no double-nested {date}/{date}/ structure
    assert not (date_dir / "2026-01-25").exists()


def test_write_run_report_overwrites(tmp_path):
    """Writing twice replaces the file (no append/duplicate content)."""
    m = _make_clean_manifest()
    date_dir = tmp_path / "2026-01-25"
    write_run_report(m, str(date_dir))
    # Re-render with one more gate
    m.add_gate("photometry", "FAILED", "second run")
    m.finalize(2.0)
    write_run_report(m, str(date_dir))
    out = date_dir / "run_report.md"
    contents = out.read_text()
    assert contents.count("# DSA-110 Run Report") == 1
    assert "second run" in contents


def test_write_run_report_creates_missing_date_dir(tmp_path):
    """If date_dir doesn't exist yet, writer mkdirs it."""
    m = _make_clean_manifest()
    date_dir = tmp_path / "products" / "2026-01-25"
    # date_dir and its parent do not yet exist
    out = write_run_report(m, str(date_dir))
    assert date_dir.is_dir()
    assert out.endswith("run_report.md")


# ─── Determinism ─────────────────────────────────────────────────────────────


def test_render_is_deterministic():
    """Two renders of the same manifest produce byte-identical output."""
    m = _make_clean_manifest()
    a = render_run_report(m, "/data/products/2026-01-25")
    b = render_run_report(m, "/data/products/2026-01-25")
    assert a == b


def test_render_does_not_mutate_manifest():
    m = _make_clean_manifest()
    n_tiles_before = len(m.tiles)
    n_epochs_before = len(m.epochs)
    n_gates_before = len(m.gates)
    render_run_report(m, "/data/products/2026-01-25")
    assert len(m.tiles) == n_tiles_before
    assert len(m.epochs) == n_epochs_before
    assert len(m.gates) == n_gates_before


# Suppress unused-import warning on `math` (keep import for any future
# numeric assertions; avoids reintroducing it later)
_ = math
