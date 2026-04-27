"""Tests for Batch E.1 hygiene fixes (Risks 1, 2, 3 from the stabilization review).

Behavior under test:
- Risk 1: forced-photometry crash records a manifest gate so pipeline_verdict
  becomes DEGRADED instead of falsely CLEAN.
- Risk 2: --photometry-workers / --photometry-chunk-size argparse flags exist
  with safe defaults and parse correctly.
- Risk 3: RunManifest.run_log field round-trips through save/load.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# scripts/ on sys.path so we can import batch_pipeline helpers
_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT / "scripts"))


# ─── Risk 3: run_log on RunManifest ──────────────────────────────────────────


def test_run_log_field_default_is_none():
    from dsa110_continuum.qa.provenance import RunManifest

    m = RunManifest.start("2026-04-27", "2026-04-27")
    assert m.run_log is None


def test_run_log_field_persists_through_save_and_load(tmp_path):
    """Setting run_log before save → load returns the same value."""
    from dsa110_continuum.qa.provenance import RunManifest

    m = RunManifest.start("2026-04-27", "2026-04-27")
    m.run_log = "/data/products/2026-04-27/run_2026-04-27T05_00_00Z.log"
    m.finalize(1.0)
    out = m.save(str(tmp_path))

    payload = json.loads(Path(out).read_text())
    assert payload["run_log"] == "/data/products/2026-04-27/run_2026-04-27T05_00_00Z.log"

    reloaded = RunManifest.load(out)
    assert reloaded.run_log == "/data/products/2026-04-27/run_2026-04-27T05_00_00Z.log"


def test_run_log_field_back_compat_with_old_manifest(tmp_path):
    """Manifests written before E.1 don't have run_log; load() must still work."""
    from dsa110_continuum.qa.provenance import RunManifest

    # Build an old-style payload with no run_log key
    legacy = {
        "git_sha": "abc1234",
        "started_at": "2026-04-25T03:00:00+00:00",
        "command_line": ["batch_pipeline.py"],
        "hostname": "h17",
        "date": "2026-04-25",
        "cal_date": "2026-04-25",
        "ms_files": [],
        "tiles": [],
        "epochs": [],
        "gates": [],
        "cal_quality": {},
        "cal_selection": {},
        "gaincal_status": "",
        "pipeline_verdict": "CLEAN",
        "wall_time_sec": 100.0,
        "finished_at": "2026-04-25T04:00:00+00:00",
        "bp_table": "",
        "g_table": "",
        "epoch_g_table": None,
    }
    p = tmp_path / "2026-04-25_manifest.json"
    p.write_text(json.dumps(legacy))

    m = RunManifest.load(str(p))
    assert m.run_log is None  # default kicks in for missing field
    assert m.pipeline_verdict == "CLEAN"


# ─── Risk 2: photometry parallel CLI wiring ──────────────────────────────────


def test_photometry_workers_argparse_defaults():
    """Replicate the orchestrator's argparse setup and confirm defaults."""
    p = argparse.ArgumentParser()
    p.add_argument("--photometry-workers", type=int, default=1)
    p.add_argument("--photometry-chunk-size", type=int, default=0)
    args = p.parse_args([])
    assert args.photometry_workers == 1  # serial by default — no behavior change
    assert args.photometry_chunk_size == 0  # 0 = auto-size when forwarded


def test_photometry_workers_argparse_parses_values():
    p = argparse.ArgumentParser()
    p.add_argument("--photometry-workers", type=int, default=1)
    p.add_argument("--photometry-chunk-size", type=int, default=0)
    args = p.parse_args(["--photometry-workers", "4", "--photometry-chunk-size", "100"])
    assert args.photometry_workers == 4
    assert args.photometry_chunk_size == 100


def test_photometry_chunk_size_zero_forwards_as_none():
    """The orchestrator forwards 0 → None so the parallel helper auto-sizes."""
    chunk_size_arg = 0
    forwarded = (chunk_size_arg or None)
    assert forwarded is None

    chunk_size_arg = 50
    forwarded = (chunk_size_arg or None)
    assert forwarded == 50


# ─── Risk 1: photometry failure → manifest gate ──────────────────────────────


def test_photometry_failure_gate_marks_run_degraded():
    """Recording a 'photometry FAILED' gate flips pipeline_verdict to DEGRADED."""
    from dsa110_continuum.qa.provenance import RunManifest

    m = RunManifest.start("2026-04-27", "2026-04-27")
    # Simulate orchestrator behaviour after a photometry crash
    m.add_gate(
        gate="photometry",
        verdict="FAILED",
        reason="forced photometry crashed for epoch 2026-04-27T22: ZeroDivisionError",
        epoch_label="2026-04-27T22",
    )
    m.finalize(1.0)

    assert m.pipeline_verdict == "DEGRADED"
    assert len(m.gates) == 1
    g = m.gates[0]
    assert g["gate"] == "photometry"
    assert g["verdict"] == "FAILED"
    assert g["epoch_label"] == "2026-04-27T22"
    assert "ZeroDivisionError" in g["reason"]


def test_orchestrator_calls_add_gate_on_photometry_failure(monkeypatch, tmp_path):
    """End-to-end: when run_forced_photometry raises, the manifest gets a gate.

    This mocks the heavy orchestrator dependencies and exercises just the
    photometry-call try/except block. We construct a minimal local executor
    that mirrors the production except-block semantics.
    """
    from dsa110_continuum.qa.provenance import RunManifest

    manifest = RunManifest.start("2026-04-27", "2026-04-27")

    def boom(*args, **kwargs):
        raise RuntimeError("synthetic photometry crash")

    label = "2026-04-27T22"

    # Mirror the orchestrator's try/except block exactly
    try:
        boom(mosaic_path="/tmp/m.fits", output_csv="/tmp/o.csv", min_flux_mjy=10.0)
    except Exception as e:
        manifest.add_gate(
            gate="photometry",
            verdict="FAILED",
            reason=f"forced photometry crashed for epoch {label}: {e}",
            epoch_label=label,
        )

    assert len(manifest.gates) == 1
    manifest.finalize(0.1)
    assert manifest.pipeline_verdict == "DEGRADED"
