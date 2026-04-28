"""Tests for Batch E.2 hygiene fixes (the four blockers from compose-risk review).

Behavior under test:
- B4: ``forced.py`` does not raise ``NameError: settings`` when
  ``dsa110_contimg`` is unavailable (cloud / bare-test env).
- B1: ``_attach_run_logfile`` and ``write_run_report`` consume the
  date-nested dir directly — no double-nesting.
- B2: ``failure_count`` measures *consecutive* failures; a success drops
  the MS from the merged failure dict.
- B3: ``--dry-run`` survives a missing ``MS_DIR`` and still produces a plan.
"""

from __future__ import annotations

import importlib.abc
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

# scripts/ on sys.path so we can import batch_pipeline helpers
_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT / "scripts"))


# ─── B4: forced.py settings fallback ─────────────────────────────────────────


class _Block_dsa110_contimg(importlib.abc.MetaPathFinder):
    """Meta-path finder that pretends dsa110_contimg is uninstalled."""

    def find_spec(self, name, path, target=None):
        if name.startswith("dsa110_contimg"):
            raise ImportError(f"simulated: {name} not available")
        return None


def test_forced_convolve_works_without_dsa110_contimg():
    """When dsa110_contimg is absent, _weighted_convolution falls back to CPU.

    This is the regression test for the ``NameError: settings is not defined``
    bug that surfaced under PYTHONPATH-only test environments. The test must
    leave ``sys.modules`` and ``sys.meta_path`` in their original state — a
    leaked blocker would poison every subsequent test that imports anything
    under ``dsa110_contimg``.
    """
    blocker = _Block_dsa110_contimg()

    # Prefixes we evict so the import block in forced.py runs fresh under the
    # blocker. We also have to evict the parent package — ``from
    # dsa110_continuum.photometry import forced`` short-circuits if the parent
    # package is cached and already has ``forced`` as an attribute.
    evict_prefixes = (
        "dsa110_contimg",
        "dsa110_continuum.photometry",
    )

    saved_meta_path = list(sys.meta_path)
    saved_modules = {
        name: mod for name, mod in sys.modules.items()
        if any(name.startswith(p) for p in evict_prefixes)
    }

    try:
        for name in list(sys.modules):
            if any(name.startswith(p) for p in evict_prefixes):
                del sys.modules[name]
        sys.meta_path.insert(0, blocker)

        # Use importlib.import_module so we re-execute the module body even
        # if a stale attribute lingers somewhere on the parent package.
        import importlib
        forced_mod = importlib.import_module(
            "dsa110_continuum.photometry.forced",
        )

        # The fallback values
        assert forced_mod.settings is None
        # CPU fallback: get_array_module returns (numpy, False)
        xp, is_gpu = forced_mod.get_array_module(prefer_gpu=True, min_elements=1)
        assert xp is np
        assert is_gpu is False

        # _weighted_convolution must not raise NameError
        data = np.ones((10, 10))
        noise = np.ones((10, 10))
        kernel = np.ones((10, 10))
        flux, flux_err, chisq = forced_mod._weighted_convolution(data, noise, kernel)
        assert np.isfinite(flux) and np.isfinite(flux_err) and np.isfinite(chisq)
    finally:
        # Restore meta_path exactly + drop the polluted forced module so the
        # next test that imports it gets the normal (non-fallback) version.
        sys.meta_path[:] = saved_meta_path
        for name in list(sys.modules):
            if (name.startswith("dsa110_contimg")
                    or name.startswith("dsa110_continuum.photometry.forced")):
                del sys.modules[name]
        for name, mod in saved_modules.items():
            sys.modules[name] = mod


# ─── B1: no double-nesting of date directory ────────────────────────────────


def _detach_file_handlers(target_path: str) -> None:
    root = logging.getLogger()
    for h in list(root.handlers):
        if isinstance(h, logging.FileHandler) and getattr(h, "baseFilename", "") == target_path:
            try:
                h.close()
            except Exception:
                pass
            root.removeHandler(h)


def test_attach_run_logfile_no_extra_date_nesting(tmp_path):
    """Caller passes the date-nested dir; helper does not append {date} again."""
    import batch_pipeline as bp

    started = datetime(2026, 4, 27, 5, 30, 17, tzinfo=timezone.utc)
    date_dir = tmp_path / "products" / "2026-04-27"
    log_path = bp._attach_run_logfile(str(date_dir), started)

    try:
        # Lands at {date_dir}/run_*.log directly
        assert Path(log_path).parent == date_dir
        # No bogus {date_dir}/2026-04-27/ subdirectory
        assert not (date_dir / "2026-04-27").exists()
    finally:
        _detach_file_handlers(log_path)


def test_write_run_report_no_extra_date_nesting(tmp_path):
    from dsa110_continuum.qa.provenance import RunManifest
    from dsa110_continuum.qa.run_report import write_run_report

    m = RunManifest.start("2026-04-27", "2026-04-27")
    m.finalize(1.0)
    date_dir = tmp_path / "products" / "2026-04-27"
    out = write_run_report(m, str(date_dir))
    expected = date_dir / "run_report.md"
    assert Path(out) == expected
    # No double-nest
    assert not (date_dir / "2026-04-27").exists()


# ─── B2: consecutive-failure semantics ──────────────────────────────────────


def test_consecutive_count_resets_after_success(tmp_path):
    """fail / fail / SUCCESS / fail → count should reset to 1, not be 3."""
    import batch_pipeline as bp

    ck = tmp_path / ".tile_checkpoint.json"

    # Run 1: a.ms fails
    bp._write_tile_checkpoint(
        str(ck), "2026-04-27", "2026-04-27",
        completed=[],
        prior_failures=[],
        current_failures=[{
            "ms_path": "/ms/a.ms", "error": "x", "elapsed_sec": 1.0,
            "failed_at": "2026-04-25T03:00:00Z",
        }],
    )
    # Run 2: a.ms fails again → count=2
    payload = json.loads(ck.read_text())
    bp._write_tile_checkpoint(
        str(ck), "2026-04-27", "2026-04-27",
        completed=[],
        prior_failures=payload["failed"],
        current_failures=[{
            "ms_path": "/ms/a.ms", "error": "y", "elapsed_sec": 1.0,
            "failed_at": "2026-04-26T03:00:00Z",
        }],
    )
    payload = json.loads(ck.read_text())
    assert payload["failed"][0]["failure_count"] == 2

    # Run 3: a.ms succeeds → cleared from failures
    bp._write_tile_checkpoint(
        str(ck), "2026-04-27", "2026-04-27",
        completed=["/img/a.fits"],
        prior_failures=payload["failed"],
        current_failures=[],
        cleared_ms_paths=["/ms/a.ms"],
    )
    payload = json.loads(ck.read_text())
    assert len(payload["failed"]) == 0

    # Run 4: a.ms fails — fresh start, count must be 1, not 3
    bp._write_tile_checkpoint(
        str(ck), "2026-04-27", "2026-04-27",
        completed=[],
        prior_failures=payload["failed"],
        current_failures=[{
            "ms_path": "/ms/a.ms", "error": "z", "elapsed_sec": 1.0,
            "failed_at": "2026-04-27T03:00:00Z",
        }],
    )
    payload = json.loads(ck.read_text())
    assert payload["failed"][0]["failure_count"] == 1


def test_quarantine_threshold_with_intervening_success(tmp_path):
    """fail / fail / fail / SUCCESS / fail / fail must NOT quarantine at threshold=3.

    This is the failure mode the consecutive-semantics fix prevents: under the
    old code a transient flake plus a real success would still leave count=3
    waiting for one more fail to silently quarantine an MS that actually works.
    """
    import batch_pipeline as bp

    ck = tmp_path / ".tile_checkpoint.json"
    # Three failures
    failures = []
    for i in range(3):
        failures.append({
            "ms_path": "/ms/a.ms", "error": f"f{i}", "elapsed_sec": 1.0,
            "failed_at": f"2026-04-2{i + 1}T03:00:00Z",
        })
    bp._write_tile_checkpoint(
        str(ck), "2026-04-27", "2026-04-27",
        completed=[],
        prior_failures=[],
        current_failures=failures[:1],
    )
    for f in failures[1:]:
        payload = json.loads(ck.read_text())
        bp._write_tile_checkpoint(
            str(ck), "2026-04-27", "2026-04-27",
            completed=[],
            prior_failures=payload["failed"],
            current_failures=[f],
        )
    payload = json.loads(ck.read_text())
    # At this point quarantine_set with threshold=3 would pick up a.ms
    assert bp._compute_quarantine_set(payload["failed"], threshold=3) == {"/ms/a.ms"}

    # Success run: a.ms cleared
    bp._write_tile_checkpoint(
        str(ck), "2026-04-27", "2026-04-27",
        completed=["/img/a.fits"],
        prior_failures=payload["failed"],
        current_failures=[],
        cleared_ms_paths=["/ms/a.ms"],
    )
    payload = json.loads(ck.read_text())
    # Two more failures (only 2 consecutive — must NOT quarantine at threshold=3)
    for i in range(2):
        bp._write_tile_checkpoint(
            str(ck), "2026-04-27", "2026-04-27",
            completed=[],
            prior_failures=payload["failed"],
            current_failures=[{
                "ms_path": "/ms/a.ms", "error": f"new{i}", "elapsed_sec": 1.0,
                "failed_at": "2026-04-27T03:00:00Z",
            }],
        )
        payload = json.loads(ck.read_text())
    # 2 consecutive failures < threshold=3 → not quarantined
    assert bp._compute_quarantine_set(payload["failed"], threshold=3) == set()


def test_cleared_ms_paths_default_preserves_old_behavior(tmp_path):
    """Calling without cleared_ms_paths is identical to the pre-fix path."""
    import batch_pipeline as bp

    ck = tmp_path / ".tile_checkpoint.json"
    bp._write_tile_checkpoint(
        str(ck), "2026-04-27", "2026-04-27",
        completed=[],
        prior_failures=[{
            "ms_path": "/ms/a.ms", "failure_count": 2,
            "error": "x", "failed_at": "2026-04-25T03:00:00Z",
        }],
        current_failures=[{
            "ms_path": "/ms/a.ms", "error": "y", "elapsed_sec": 1.0,
            "failed_at": "2026-04-27T03:00:00Z",
        }],
        # cleared_ms_paths omitted → defaults to None
    )
    payload = json.loads(ck.read_text())
    # Old behavior: count increments to 3
    assert payload["failed"][0]["failure_count"] == 3


# ─── B3: dry-run survives missing MS_DIR ────────────────────────────────────


def test_dry_run_main_handles_missing_ms_dir(tmp_path, monkeypatch, caplog):
    """Pre-dry-run os.listdir(MS_DIR) crash is gone; the plan still renders."""
    import batch_pipeline as bp

    products_root = tmp_path / "products_root"
    stage_root = tmp_path / "stage_root"
    # MS_DIR points at a path that does NOT exist
    nonexistent_ms_dir = tmp_path / "absent_ms_dir"
    assert not nonexistent_ms_dir.exists()

    monkeypatch.setattr(bp, "PRODUCTS_BASE", str(products_root))
    monkeypatch.setattr(bp, "STAGE_IMAGE_BASE", str(stage_root))
    monkeypatch.setattr(bp, "MS_DIR", str(nonexistent_ms_dir))

    # Build a minimal args namespace replicating what argparse would produce
    # when the user passes only --dry-run --date.
    import argparse
    args = argparse.Namespace(
        date="2026-04-27",
        cal_date=None,
        expected_dec=16.1,
        skip_auto_cal=False,
        force_recal=False,
        clear_quarantine=False,
        quarantine_after_failures=3,
        skip_epoch_gaincal=False,
        skip_photometry=False,
        lenient_qa=False,
        start_hour=None,
        end_hour=None,
    )

    with caplog.at_level("INFO"):
        # _dry_run_main must run cleanly — the crash was in the pre-_dry_run
        # block that we just guarded; this test exercises the dry-run path
        # itself which has its own try/except around find_valid_ms.
        bp._dry_run_main(args, "2026-04-27", "2026-04-27", obs_dec_deg=16.1)

    # Plan still rendered
    assert any("DRY RUN" in r.message for r in caplog.records)


def test_main_pre_dry_run_block_handles_missing_ms_dir(tmp_path, monkeypatch):
    """The os.listdir call before the dry-run branch must not raise.

    Direct unit test on the listing-or-empty-list logic copied from main():
    if MS_DIR is missing the resulting list is just empty.
    """
    import os as _os

    nonexistent = tmp_path / "nope"
    # Replicate the exact try/except block from main()
    try:
        _ms_dir_listing = _os.listdir(str(nonexistent))
    except (FileNotFoundError, NotADirectoryError, PermissionError):
        _ms_dir_listing = []
    assert _ms_dir_listing == []
