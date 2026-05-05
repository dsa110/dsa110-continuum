"""Tests for the end-of-run promotion auto-emit hook in batch_pipeline.py.

Behavior under test:
- ``_emit_promotion_record`` invokes ``dsa110_continuum.qa.promotion.emit_for_run``
  with the run manifest and products directory so a per-``(date, hour)``
  promotion side-car JSON and ledger row are written at end of run.
- The hook is non-fatal: a writer failure must not fail an otherwise
  completed pipeline run.
- ``main()`` invokes ``_emit_promotion_record`` so the helper is actually
  reached on a real run (regression guard against silently dropping the
  call site, as happened when the structured-status / promotion-record
  feature was split across commits 385a5f8, 8237c09, and 1bfa519 on main).
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

# scripts/ on sys.path so we can import batch_pipeline helpers
_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT / "scripts"))


def test_emit_promotion_record_invokes_emit_for_run(monkeypatch, tmp_path):
    """Hook must invoke emit_for_run with the manifest and products_dir."""
    import batch_pipeline as bp

    products_dir = tmp_path / "products" / "2026-01-25"
    products_dir.mkdir(parents=True)

    calls: list[dict] = []

    def fake_emit(manifest, products_dir_arg, repo_root, **kwargs):
        calls.append(
            {
                "manifest": manifest,
                "products_dir": products_dir_arg,
                "repo_root": repo_root,
                "kwargs": kwargs,
            }
        )

    monkeypatch.setattr("dsa110_continuum.qa.promotion.emit_for_run", fake_emit)

    manifest = SimpleNamespace(date="2026-01-25")
    paths = {"products_dir": str(products_dir)}
    args = SimpleNamespace(skip_epoch_gaincal=False)

    bp._emit_promotion_record(manifest, paths, args)

    assert len(calls) == 1, "emit_for_run must be invoked exactly once per run"
    assert calls[0]["manifest"] is manifest
    assert calls[0]["products_dir"] == str(products_dir)
    assert calls[0]["kwargs"].get("skip_epoch_gaincal") is False


def test_emit_promotion_record_failure_is_non_fatal(monkeypatch, tmp_path):
    """A sidecar/ledger failure must not fail an otherwise completed run."""
    import batch_pipeline as bp

    products_dir = tmp_path / "products" / "2026-01-25"
    products_dir.mkdir(parents=True)

    def raising_emit(*_args, **_kwargs):
        raise RuntimeError("simulated promotion writer failure")

    monkeypatch.setattr("dsa110_continuum.qa.promotion.emit_for_run", raising_emit)

    manifest = SimpleNamespace(date="2026-01-25")
    paths = {"products_dir": str(products_dir)}
    args = SimpleNamespace(skip_epoch_gaincal=False)

    # Must not raise — the try/except inside the helper swallows writer faults.
    bp._emit_promotion_record(manifest, paths, args)


def test_main_calls_emit_promotion_record_helper():
    """Regression guard: main() must invoke _emit_promotion_record so the
    promotion auto-emit hook actually runs at end of pipeline.

    This test exists because the issue-26 split commits (385a5f8, 8237c09,
    1bfa519) deferred the call-site wiring from main(), and unit tests
    against emit_for_run() alone could not detect the integration gap.
    """
    import inspect

    import batch_pipeline as bp

    src = inspect.getsource(bp.main)
    assert "_emit_promotion_record" in src, (
        "main() must invoke _emit_promotion_record() to wire promotion auto-emit. "
        "If the helper has been renamed, update this test accordingly."
    )
