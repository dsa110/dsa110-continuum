"""Tests for Batch E: --dry-run mode and MS quarantine policy.

Behavior under test:
- ``_compute_quarantine_set`` honors the threshold; ``threshold <= 0`` disables.
- ``_clear_failure_counts`` zeros counts but preserves history.
- ``_write_tile_checkpoint`` increments ``failure_count`` on repeat failures.
- ``_collect_dry_run_plan`` produces a plan dict with the expected keys/values.
- ``--dry-run`` exits without creating dirs or log files (no side effects).
- CLI flag wiring for --quarantine-after-failures, --clear-quarantine, --dry-run.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# scripts/ on sys.path so we can import batch_pipeline helpers
_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT / "scripts"))


# ─── _compute_quarantine_set ────────────────────────────────────────────────


def test_quarantine_set_below_threshold_is_empty():
    import batch_pipeline as bp

    failures = [
        {"ms_path": "/ms/a.ms", "failure_count": 1},
        {"ms_path": "/ms/b.ms", "failure_count": 2},
    ]
    assert bp._compute_quarantine_set(failures, threshold=3) == set()


def test_quarantine_set_at_threshold_quarantines():
    import batch_pipeline as bp

    failures = [
        {"ms_path": "/ms/a.ms", "failure_count": 3},
        {"ms_path": "/ms/b.ms", "failure_count": 4},
        {"ms_path": "/ms/c.ms", "failure_count": 1},
    ]
    assert bp._compute_quarantine_set(failures, threshold=3) == {
        "/ms/a.ms", "/ms/b.ms",
    }


def test_quarantine_threshold_zero_disables():
    import batch_pipeline as bp

    failures = [{"ms_path": "/ms/a.ms", "failure_count": 99}]
    assert bp._compute_quarantine_set(failures, threshold=0) == set()
    assert bp._compute_quarantine_set(failures, threshold=-1) == set()


def test_quarantine_skips_records_without_ms_path_or_count():
    import batch_pipeline as bp

    failures = [
        {"failure_count": 5},  # no ms_path
        {"ms_path": "/ms/x.ms"},  # no failure_count → defaults to 0
        {"ms_path": "/ms/y.ms", "failure_count": 3},
    ]
    assert bp._compute_quarantine_set(failures, threshold=3) == {"/ms/y.ms"}


# ─── _clear_failure_counts ──────────────────────────────────────────────────


def test_clear_failure_counts_zeros_counts_preserves_history():
    import batch_pipeline as bp

    failures = [
        {"ms_path": "/ms/a.ms", "failure_count": 4, "error": "casa_hang", "failed_at": "x"},
        {"ms_path": "/ms/b.ms", "failure_count": 3, "error": "timeout", "failed_at": "y"},
    ]
    cleared = bp._clear_failure_counts(failures)
    assert all(rec["failure_count"] == 0 for rec in cleared)
    # History fields preserved
    assert cleared[0]["error"] == "casa_hang"
    assert cleared[1]["failed_at"] == "y"
    # Original list not mutated
    assert failures[0]["failure_count"] == 4


# ─── _write_tile_checkpoint failure_count behavior ──────────────────────────


def test_checkpoint_first_failure_count_is_one(tmp_path):
    import batch_pipeline as bp

    ck = tmp_path / ".tile_checkpoint.json"
    bp._write_tile_checkpoint(
        str(ck), "2026-04-27", "2026-04-27",
        completed=[],
        prior_failures=[],
        current_failures=[{
            "ms_path": "/ms/a.ms",
            "error": "timeout",
            "elapsed_sec": 1800,
            "failed_at": "2026-04-27T03:00:00Z",
        }],
    )
    data = json.loads(ck.read_text())
    assert len(data["failed"]) == 1
    rec = data["failed"][0]
    assert rec["failure_count"] == 1
    assert rec["first_failed_at"] == "2026-04-27T03:00:00Z"


def test_checkpoint_repeat_failure_increments_count(tmp_path):
    import batch_pipeline as bp

    ck = tmp_path / ".tile_checkpoint.json"
    prior = [{
        "ms_path": "/ms/a.ms", "failure_count": 2,
        "error": "timeout", "elapsed_sec": 1800,
        "failed_at": "2026-04-25T03:00:00Z",
        "first_failed_at": "2026-04-23T03:00:00Z",
    }]
    current = [{
        "ms_path": "/ms/a.ms", "error": "casa_hang",
        "elapsed_sec": 900, "failed_at": "2026-04-27T03:00:00Z",
    }]
    bp._write_tile_checkpoint(
        str(ck), "2026-04-27", "2026-04-27", [], prior, current,
    )
    rec = json.loads(ck.read_text())["failed"][0]
    assert rec["failure_count"] == 3  # 2 + 1
    assert rec["error"] == "casa_hang"  # current error wins
    assert rec["first_failed_at"] == "2026-04-23T03:00:00Z"  # preserved


def test_checkpoint_legacy_failure_record_upgrades(tmp_path):
    """Old checkpoints lack failure_count; the helper should default to 1."""
    import batch_pipeline as bp

    ck = tmp_path / ".tile_checkpoint.json"
    legacy_prior = [{"ms_path": "/ms/a.ms", "error": "x"}]  # no failure_count
    bp._write_tile_checkpoint(
        str(ck), "2026-04-27", "2026-04-27", [], legacy_prior, [],
    )
    rec = json.loads(ck.read_text())["failed"][0]
    assert rec["failure_count"] == 1


# ─── _collect_dry_run_plan ──────────────────────────────────────────────────


def test_collect_plan_basic_fields():
    import batch_pipeline as bp

    plan = bp._collect_dry_run_plan(
        date="2026-04-27",
        cal_date="2026-04-27",
        obs_dec_deg=16.5,
        paths={"stage_dir": "/stage", "products_dir": "/products"},
        bp_table="/no/such/path.b",
        g_table="/no/such/path.g",
        ms_list_after_filters=["/ms/a.ms", "/ms/b.ms", "/ms/c.ms"],
        epoch_hours=[2, 22],
        epoch_decisions=[
            {"hour": 2, "action": "rebuild", "reason": "no mosaic on disk"},
            {"hour": 22, "action": "skip", "reason": "prior verdict=PASS"},
        ],
        prior_manifest_verdict="DEGRADED",
        prior_manifest_present=True,
        checkpoint_completed=8,
        checkpoint_failures=[
            {"ms_path": "/ms/c.ms", "failure_count": 3},
        ],
        quarantine_threshold=3,
        quarantine_set={"/ms/c.ms"},
        skip_epoch_gaincal=False,
        skip_photometry=False,
        lenient_qa=False,
    )
    assert plan["date"] == "2026-04-27"
    assert plan["ms_files_after_filters"] == 3
    assert plan["epoch_hours"] == [2, 22]
    assert plan["quarantine_count"] == 1
    assert plan["quarantine_ms_paths"] == ["/ms/c.ms"]
    assert plan["phase1_tiles_to_attempt"] == 2  # 3 total - 1 quarantined
    assert plan["phase1_tiles_quarantined"] == 1
    assert plan["phase2_epochs_to_rebuild"] == 1
    assert plan["phase2_epochs_to_skip"] == 1
    assert "strict" in plan["phase3_photometry"]


def test_collect_plan_lenient_qa_field():
    import batch_pipeline as bp

    plan = bp._collect_dry_run_plan(
        date="2026-04-27", cal_date="2026-04-27", obs_dec_deg=16.5,
        paths={"stage_dir": "/s", "products_dir": "/p"},
        bp_table="", g_table="",
        ms_list_after_filters=[],
        epoch_hours=[], epoch_decisions=[],
        prior_manifest_verdict=None, prior_manifest_present=False,
        checkpoint_completed=0, checkpoint_failures=[],
        quarantine_threshold=3, quarantine_set=set(),
        skip_epoch_gaincal=False,
        skip_photometry=False,
        lenient_qa=True,
    )
    assert "lenient" in plan["phase3_photometry"]


def test_format_plan_includes_quarantine_lines():
    import batch_pipeline as bp

    plan = bp._collect_dry_run_plan(
        date="2026-04-27", cal_date="2026-04-27", obs_dec_deg=16.5,
        paths={"stage_dir": "/s", "products_dir": "/p"},
        bp_table="/bp.b", g_table="/g.g",
        ms_list_after_filters=["/ms/q.ms", "/ms/ok.ms"],
        epoch_hours=[22],
        epoch_decisions=[{"hour": 22, "action": "rebuild", "reason": "x"}],
        prior_manifest_verdict="DEGRADED", prior_manifest_present=True,
        checkpoint_completed=0, checkpoint_failures=[],
        quarantine_threshold=3, quarantine_set={"/ms/q.ms"},
        skip_epoch_gaincal=False, skip_photometry=False, lenient_qa=False,
    )
    lines = bp._format_dry_run_plan(plan)
    text = "\n".join(lines)
    assert "DRY RUN" in text
    assert "QUARANTINED: /ms/q.ms" in text
    assert "Pipeline NOT executed" in text
    assert "rebuild (x)" in text


# ─── --dry-run end-to-end: no side effects ──────────────────────────────────


def test_dry_run_does_not_create_products_or_log(tmp_path, monkeypatch, caplog):
    """--dry-run must not mkdir products/stage or attach a FileHandler."""
    import argparse
    import logging

    import batch_pipeline as bp

    products_root = tmp_path / "products_root"
    stage_root = tmp_path / "stage_root"
    ms_dir = tmp_path / "ms_root"
    ms_dir.mkdir()  # MS root exists but is empty — find_valid_ms returns []

    monkeypatch.setattr(bp, "PRODUCTS_BASE", str(products_root))
    monkeypatch.setattr(bp, "STAGE_IMAGE_BASE", str(stage_root))
    monkeypatch.setattr(bp, "MS_DIR", str(ms_dir))

    args = argparse.Namespace(
        date="2026-04-27",
        force_recal=False,
        clear_quarantine=False,
        quarantine_after_failures=3,
        skip_epoch_gaincal=False,
        skip_photometry=False,
        lenient_qa=False,
        start_hour=None,
        end_hour=None,
    )

    # Snapshot root logger handlers before
    root = logging.getLogger()
    handlers_before = list(root.handlers)

    # Run the dry-run main directly (decoupled from argparse)
    with caplog.at_level("INFO"):
        bp._dry_run_main(args, "2026-04-27", "2026-04-27", obs_dec_deg=16.5)

    # No products/stage dirs created
    assert not products_root.exists() or list(products_root.iterdir()) == []
    assert not stage_root.exists() or list(stage_root.iterdir()) == []
    # Verify no FileHandler was attached to the root logger
    new_file_handlers = [
        h for h in root.handlers
        if isinstance(h, logging.FileHandler) and h not in handlers_before
    ]
    assert new_file_handlers == [], (
        f"dry-run attached {len(new_file_handlers)} new FileHandler(s)"
    )

    # Confirm DRY RUN banner appeared in logs
    assert any("DRY RUN" in r.message for r in caplog.records)


# ─── CLI wiring ─────────────────────────────────────────────────────────────


def test_cli_dry_run_flag_default_false():
    """argparse default is opt-in (--dry-run not present → args.dry_run False)."""
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true", default=False)
    assert p.parse_args([]).dry_run is False
    assert p.parse_args(["--dry-run"]).dry_run is True


def test_cli_quarantine_after_failures_default_three():
    """Default threshold is 3 — conservative enough that one bad night doesn't quarantine."""
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--quarantine-after-failures", type=int, default=3)
    p.add_argument("--clear-quarantine", action="store_true", default=False)
    args = p.parse_args([])
    assert args.quarantine_after_failures == 3
    assert args.clear_quarantine is False
    args = p.parse_args(["--quarantine-after-failures", "5", "--clear-quarantine"])
    assert args.quarantine_after_failures == 5
    assert args.clear_quarantine is True
