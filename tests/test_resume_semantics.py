"""Tests for Batch A resume semantics:

- ``RunManifest.add_gate`` + ``epoch_verdict`` helpers
- ``try_load_prior_manifest`` safe-load (returns None on missing/corrupt)
- ``_epoch_should_rebuild`` decision matrix
- ``_write_tile_checkpoint`` round-trip with merged failures

These behaviors keep daily production runs restart-safe: a crashed run's
FAIL-verdict or mid-epoch mosaic must not be silently reused, and tile
failure history must not be silently dropped across re-runs.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# scripts/ directory on path so we can import batch_pipeline helpers
_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT / "scripts"))


# ─── RunManifest helpers ─────────────────────────────────────────────────────


def test_add_gate_causes_degraded_verdict():
    """Any gate entry must flip pipeline_verdict to DEGRADED on finalize."""
    from dsa110_continuum.qa.provenance import RunManifest

    m = RunManifest.start("2026-02-12", "2026-02-12")
    m.add_gate("gaincal", "FALLBACK", "epoch gaincal failed", static_g_table="/tmp/x.g")
    m.finalize(1.0)

    assert m.pipeline_verdict == "DEGRADED"
    assert len(m.gates) == 1
    entry = m.gates[0]
    assert entry["gate"] == "gaincal"
    assert entry["verdict"] == "FALLBACK"
    assert entry["reason"] == "epoch gaincal failed"
    assert entry["static_g_table"] == "/tmp/x.g"


def test_no_gates_means_clean():
    from dsa110_continuum.qa.provenance import RunManifest

    m = RunManifest.start("2026-02-12", "2026-02-12")
    m.finalize(1.0)
    assert m.pipeline_verdict == "CLEAN"


def test_epoch_verdict_returns_recorded_and_none():
    from dsa110_continuum.qa.provenance import RunManifest

    m = RunManifest.start("2026-02-12", "2026-02-12")
    m.record_epoch(21, {"n_tiles": 13, "status": "ok", "qa_result": "PASS"})
    m.record_epoch(22, {"n_tiles": 11, "status": "ok", "qa_result": "FAIL"})
    m.record_epoch(23, {"n_tiles": 5, "status": "ok"})  # no qa_result → None

    assert m.epoch_verdict(21) == "PASS"
    assert m.epoch_verdict(22) == "FAIL"
    assert m.epoch_verdict(23) is None
    assert m.epoch_verdict(0) is None  # never recorded


# ─── try_load_prior_manifest ──────────────────────────────────────────────────


def test_try_load_prior_manifest_missing_returns_none(tmp_path):
    from dsa110_continuum.qa.provenance import try_load_prior_manifest

    # No manifest exists → None, not a raise
    got = try_load_prior_manifest("2026-02-12", products_dir=str(tmp_path))
    assert got is None


def test_try_load_prior_manifest_corrupt_returns_none(tmp_path):
    from dsa110_continuum.qa.provenance import try_load_prior_manifest

    date_dir = tmp_path / "2026-02-12"
    date_dir.mkdir()
    (date_dir / "2026-02-12_manifest.json").write_text("not { valid json")

    got = try_load_prior_manifest("2026-02-12", products_dir=str(tmp_path))
    assert got is None  # logs warning but does not raise


def test_try_load_prior_manifest_valid_returns_manifest(tmp_path):
    from dsa110_continuum.qa.provenance import RunManifest, try_load_prior_manifest

    m = RunManifest.start("2026-02-12", "2026-02-12")
    m.record_epoch(21, {"n_tiles": 13, "status": "ok", "qa_result": "PASS"})
    m.finalize(10.0)
    m.save(str(tmp_path / "2026-02-12"))

    got = try_load_prior_manifest("2026-02-12", products_dir=str(tmp_path))
    assert got is not None
    assert got.epoch_verdict(21) == "PASS"


# ─── _epoch_should_rebuild decision matrix ───────────────────────────────────


def test_epoch_should_rebuild_force_recal(tmp_path):
    import batch_pipeline as bp
    from dsa110_continuum.qa.provenance import RunManifest

    mosaic = tmp_path / "epoch_21.fits"
    mosaic.write_bytes(b"fake")
    prior = RunManifest.start("2026-02-12", "2026-02-12")
    prior.record_epoch(21, {"n_tiles": 13, "status": "ok", "qa_result": "PASS"})

    # force_recal always rebuilds, even for a PASS prior
    assert bp._epoch_should_rebuild(str(mosaic), prior, 21, force_recal=True) is True


def test_epoch_should_rebuild_missing_file(tmp_path):
    import batch_pipeline as bp

    # File doesn't exist → always rebuild
    missing = str(tmp_path / "nope.fits")
    assert bp._epoch_should_rebuild(missing, None, 21, force_recal=False) is True


def test_epoch_should_rebuild_no_prior_manifest(tmp_path):
    """Backward-compat: if no prior manifest and file exists, skip (trust it)."""
    import batch_pipeline as bp

    mosaic = tmp_path / "epoch_21.fits"
    mosaic.write_bytes(b"fake")
    assert bp._epoch_should_rebuild(str(mosaic), None, 21, force_recal=False) is False


def test_epoch_should_rebuild_prior_pass(tmp_path):
    import batch_pipeline as bp
    from dsa110_continuum.qa.provenance import RunManifest

    mosaic = tmp_path / "epoch_21.fits"
    mosaic.write_bytes(b"fake")
    prior = RunManifest.start("2026-02-12", "2026-02-12")
    prior.record_epoch(21, {"n_tiles": 13, "status": "ok", "qa_result": "PASS"})

    assert bp._epoch_should_rebuild(str(mosaic), prior, 21, force_recal=False) is False


def test_epoch_should_rebuild_prior_fail(tmp_path):
    """Key case: FAIL-verdict mosaic from prior run must be rebuilt."""
    import batch_pipeline as bp
    from dsa110_continuum.qa.provenance import RunManifest

    mosaic = tmp_path / "epoch_22.fits"
    mosaic.write_bytes(b"fake")
    prior = RunManifest.start("2026-02-12", "2026-02-12")
    prior.record_epoch(22, {"n_tiles": 11, "status": "ok", "qa_result": "FAIL"})

    assert bp._epoch_should_rebuild(str(mosaic), prior, 22, force_recal=False) is True


def test_epoch_should_rebuild_prior_crash(tmp_path):
    """Crash mid-epoch: file exists but prior manifest has no entry → rebuild."""
    import batch_pipeline as bp
    from dsa110_continuum.qa.provenance import RunManifest

    mosaic = tmp_path / "epoch_23.fits"
    mosaic.write_bytes(b"fake")
    prior = RunManifest.start("2026-02-12", "2026-02-12")
    # Only record epoch 21, not 23
    prior.record_epoch(21, {"n_tiles": 13, "status": "ok", "qa_result": "PASS"})

    assert bp._epoch_should_rebuild(str(mosaic), prior, 23, force_recal=False) is True


# ─── _write_tile_checkpoint round-trip ────────────────────────────────────────


def test_checkpoint_roundtrip_merges_failures(tmp_path):
    import batch_pipeline as bp

    ck = tmp_path / ".tile_checkpoint.json"

    prior = [
        {"ms_path": "/ms/a.ms", "error": "timeout", "elapsed_sec": 1800},
        {"ms_path": "/ms/b.ms", "error": "oops", "elapsed_sec": 12},
    ]
    current = [
        # a.ms failed again this run with a different error → current wins
        {"ms_path": "/ms/a.ms", "error": "corrupt_ms_skipped", "elapsed_sec": 0},
        {"ms_path": "/ms/c.ms", "error": "casa_hang", "elapsed_sec": 900},
    ]
    completed = ["/img/x.fits", "/img/y.fits"]

    bp._write_tile_checkpoint(str(ck), "2026-02-12", "2026-02-12",
                              completed, prior, current)

    assert ck.exists()
    data = json.loads(ck.read_text())
    assert data["date"] == "2026-02-12"
    assert data["completed"] == completed
    # a.ms deduplicated with current error; b.ms preserved; c.ms added
    failed_by_ms = {r["ms_path"]: r for r in data["failed"]}
    assert set(failed_by_ms) == {"/ms/a.ms", "/ms/b.ms", "/ms/c.ms"}
    assert failed_by_ms["/ms/a.ms"]["error"] == "corrupt_ms_skipped"


def test_checkpoint_atomic_write_does_not_leak_tmp(tmp_path):
    """The .tmp file should be replaced, not left behind."""
    import batch_pipeline as bp

    ck = tmp_path / ".tile_checkpoint.json"
    bp._write_tile_checkpoint(str(ck), "2026-02-12", "2026-02-12", [], [], [])
    assert ck.exists()
    assert not (tmp_path / ".tile_checkpoint.json.tmp").exists()
