"""Tests for Batch C per-run file logging.

Behavior under test:
- ``_run_log_filename`` produces a UTC, filename-safe timestamp.
- ``_attach_run_logfile`` writes under ``{products_dir}/{date}/`` and is
  idempotent (no duplicate handlers if main() is invoked twice in-process).
- Logs emitted after attach actually land in the file.
- ``emit_run_summary`` records the log path under the ``run_log`` key.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

# scripts/ on sys.path so we can import batch_pipeline helpers
_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT / "scripts"))


# ─── _run_log_filename ───────────────────────────────────────────────────────


def test_run_log_filename_is_filename_safe_utc():
    """No ``:`` in the name, ``Z`` suffix, sortable ISO-like form."""
    import batch_pipeline as bp

    dt = datetime(2026, 4, 27, 5, 30, 17, tzinfo=timezone.utc)
    name = bp._run_log_filename(dt)
    assert name == "run_2026-04-27T05_30_17Z.log"
    assert ":" not in name  # portable on Windows-ish filesystems


def test_run_log_filename_converts_to_utc():
    """A non-UTC datetime is converted to UTC for the filename."""
    from datetime import timedelta

    import batch_pipeline as bp

    # 02:30:17 in UTC+10 == 16:30:17 UTC the previous day
    aest = timezone(timedelta(hours=10))
    dt = datetime(2026, 4, 28, 2, 30, 17, tzinfo=aest)
    assert bp._run_log_filename(dt) == "run_2026-04-27T16_30_17Z.log"


# ─── _attach_run_logfile ─────────────────────────────────────────────────────


def _detach_test_handlers(target_path: str) -> None:
    """Remove FileHandlers we attached during a test (best-effort cleanup)."""
    root = logging.getLogger()
    for h in list(root.handlers):
        if isinstance(h, logging.FileHandler) and getattr(h, "baseFilename", "") == target_path:
            try:
                h.close()
            except Exception:
                pass
            root.removeHandler(h)


def test_attach_run_logfile_path_under_products_date(tmp_path):
    """Returned path is under {products_dir}/{date}/ with the run_<utc>.log name."""
    import batch_pipeline as bp

    started = datetime(2026, 4, 27, 5, 30, 17, tzinfo=timezone.utc)
    products = tmp_path / "products"
    log_path = bp._attach_run_logfile(str(products), "2026-04-27", started)

    try:
        assert Path(log_path).parent == (products / "2026-04-27")
        assert Path(log_path).name == "run_2026-04-27T05_30_17Z.log"
        assert Path(log_path).is_absolute()
        # Date dir must have been created
        assert (products / "2026-04-27").is_dir()
    finally:
        _detach_test_handlers(log_path)


def test_attach_run_logfile_is_idempotent(tmp_path):
    """Calling attach twice with the same target leaves exactly one FileHandler."""
    import batch_pipeline as bp

    started = datetime(2026, 4, 27, 5, 30, 17, tzinfo=timezone.utc)
    products = tmp_path / "products"

    log_path = bp._attach_run_logfile(str(products), "2026-04-27", started)
    bp._attach_run_logfile(str(products), "2026-04-27", started)
    bp._attach_run_logfile(str(products), "2026-04-27", started)

    try:
        root = logging.getLogger()
        matching = [
            h for h in root.handlers
            if isinstance(h, logging.FileHandler)
            and getattr(h, "baseFilename", "") == log_path
        ]
        assert len(matching) == 1, f"expected 1 handler, got {len(matching)}"
    finally:
        _detach_test_handlers(log_path)


def test_attach_run_logfile_captures_log_lines(tmp_path):
    """A log line emitted after attach reaches the file.

    Note: under pytest, ``logging.basicConfig`` from batch_pipeline import is a
    no-op (pytest pre-attaches its own handler), so the root logger's level
    stays at its default. We force INFO here to mirror the production runtime
    state where basicConfig sets ``level=logging.INFO`` on import.
    """
    import batch_pipeline as bp

    started = datetime(2026, 4, 27, 5, 30, 17, tzinfo=timezone.utc)
    products = tmp_path / "products"
    log_path = bp._attach_run_logfile(str(products), "2026-04-27", started)

    test_logger = logging.getLogger("batch_pipeline")
    prior_level = test_logger.level
    test_logger.setLevel(logging.INFO)
    try:
        test_logger.info("hello-from-test-marker")
        # FileHandler is buffered; flush before reading
        for h in logging.getLogger().handlers:
            if isinstance(h, logging.FileHandler) and h.baseFilename == log_path:
                h.flush()

        contents = Path(log_path).read_text()
        assert "hello-from-test-marker" in contents
        assert "INFO" in contents
        assert "batch_pipeline" in contents
    finally:
        test_logger.setLevel(prior_level)
        _detach_test_handlers(log_path)


# ─── emit_run_summary records run_log ────────────────────────────────────────


def test_emit_run_summary_records_run_log(tmp_path):
    import batch_pipeline as bp

    products = tmp_path / "products" / "2026-04-27"
    products.mkdir(parents=True)
    fake_log_path = "/tmp/products/2026-04-27/run_2026-04-27T05_30_17Z.log"

    bp.emit_run_summary(
        date="2026-04-27",
        cal_date="2026-04-27",
        epoch_results=[
            {"label": "2026-04-27T22", "status": "ok", "qa_result": "PASS"},
        ],
        wall_time_sec=12.3,
        products_dir=str(products),
        run_log_path=fake_log_path,
    )

    summary_path = products / "2026-04-27_run_summary.json"
    assert summary_path.exists()
    payload = json.loads(summary_path.read_text())
    assert payload["run_log"] == fake_log_path


def test_emit_run_summary_run_log_is_none_when_not_provided(tmp_path):
    """Backward-compat: callers that don't pass run_log_path get a null value."""
    import batch_pipeline as bp

    products = tmp_path / "products" / "2026-04-27"
    products.mkdir(parents=True)
    bp.emit_run_summary(
        date="2026-04-27",
        cal_date="2026-04-27",
        epoch_results=[],
        wall_time_sec=0.0,
        products_dir=str(products),
    )
    payload = json.loads((products / "2026-04-27_run_summary.json").read_text())
    assert payload["run_log"] is None
