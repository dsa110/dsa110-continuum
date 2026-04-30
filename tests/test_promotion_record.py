"""Tests for dsa110_continuum.qa.promotion."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any

import pytest
from dsa110_continuum.qa.promotion import (
    DAILY_CAL_TIERS,
    EPOCH_GAINCAL_STATES,
    PROMOTION_CLASSES,
    build_promotion_record,
    derive_daily_cal_tier,
    derive_epoch_gaincal_state,
    derive_promotion_class,
    emit_for_run,
    sidecar_path,
    write_promotion_sidecar,
)

# ── derivation helpers ────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "cal_selection, expected",
    [
        ({"source": "generated"}, "A"),
        ({"source": "existing"}, "A"),
        ({"source": "borrowed"}, "C"),
        ({"source": "unknown_value"}, "unknown"),
        ({}, "unknown"),
        (None, "unknown"),
    ],
)
def test_derive_daily_cal_tier(cal_selection, expected):
    assert derive_daily_cal_tier(cal_selection) == expected
    assert derive_daily_cal_tier(cal_selection) in DAILY_CAL_TIERS


@pytest.mark.parametrize(
    "legacy_status, skip_intentionally, expected",
    [
        ("ok", False, "solved"),
        ("low_snr", False, "skipped_or_failed_low_snr"),
        ("fallback", False, "fell_back_to_static_with_reason"),
        ("error", False, "fell_back_to_static_with_reason"),
        ("skipped", True, "skipped_intentionally"),
        ("skipped", False, "skipped_or_failed_low_snr"),
        ("", False, "unknown"),
        (None, False, "unknown"),
    ],
)
def test_derive_epoch_gaincal_state(legacy_status, skip_intentionally, expected):
    state = derive_epoch_gaincal_state(legacy_status, skip_intentionally=skip_intentionally)
    assert state == expected
    assert state in EPOCH_GAINCAL_STATES


def test_derive_promotion_class_no_anchor_returns_pending_review():
    assert derive_promotion_class("A", "solved", anchor=None) == "auto_emitted_pending_review"
    assert derive_promotion_class("A", "solved", anchor={}) == "auto_emitted_pending_review"
    assert (
        derive_promotion_class(
            "A",
            "solved",
            anchor={
                "primary_model": None,
                "catalog_xmatch": None,
                "tile_self_consistency": None,
            },
        )
        == "auto_emitted_pending_review"
    )


def test_derive_promotion_class_trusted_when_tier_a_solved_and_anchor_filled():
    anchor = {"catalog_xmatch": {"catalog": "nvss", "n": 27, "median_ratio": 0.95}}
    assert derive_promotion_class("A", "solved", anchor=anchor) == "trusted_baseline"


def test_derive_promotion_class_comparator_when_tier_or_gaincal_off():
    anchor = {"primary_model": "3C286"}
    assert derive_promotion_class("C", "solved", anchor=anchor) == "comparator_only"
    assert (
        derive_promotion_class("A", "fell_back_to_static_with_reason", anchor=anchor)
        == "comparator_only"
    )


# ── synthetic manifest for build/write/emit tests ────────────────────────────


@dataclass
class _FakeManifest:
    """Minimal shape the promotion module reads from RunManifest."""

    date: str = "2026-01-25"
    git_sha: str = "abc1234"
    bp_table: str = "/stage/dsa110-contimg/ms/2026-01-25T22:26:05_0~23.b"
    g_table: str = "/stage/dsa110-contimg/ms/2026-01-25T22:26:05_0~23.g"
    cal_selection: dict[str, Any] = field(default_factory=lambda: {"source": "existing"})
    gaincal_status: str = "ok"
    pipeline_verdict: str = "CLEAN"
    command_line: list[str] = field(
        default_factory=lambda: [
            "/opt/miniforge/envs/casa6/bin/python",
            "scripts/batch_pipeline.py",
            "--date",
            "2026-01-25",
            "--start-hour",
            "2",
            "--end-hour",
            "3",
        ]
    )
    run_log: str = "/data/dsa110-proc/products/mosaics/2026-01-25/run_2026-04-30T00.log"
    epochs: list[dict[str, Any]] = field(
        default_factory=lambda: [
            {
                "hour": 2,
                "n_tiles": 11,
                "status": "ok",
                "qa_result": "PASS",
                "mosaic_path": "/stage/dsa110-contimg/images/mosaic_2026-01-25/2026-01-25T0200_mosaic.fits",
            },
        ]
    )
    gates: list[dict[str, Any]] = field(default_factory=list)


def test_build_promotion_record_tier_a_solved_clean(tmp_path):
    m = _FakeManifest()
    rec = build_promotion_record(m, hour=2, products_dir=str(tmp_path))
    assert rec["date"] == "2026-01-25"
    assert rec["hour"] == 2
    assert rec["daily_cal_tier"] == "A"
    assert rec["epoch_gaincal_state"] == "solved"
    assert rec["eligible_for_trusted_baseline"] is True
    assert rec["eligible_for_trusted_baseline_reason"] is None
    assert rec["promotion_class"] == "auto_emitted_pending_review"
    assert rec["mosaic_path"].endswith("2026-01-25T0200_mosaic.fits")
    assert rec["batch_pipeline_invocation"][:2] == m.command_line[:2]
    assert rec["cal_provenance"]["bp"] == m.bp_table
    assert rec["cal_provenance"]["g"] == m.g_table


def test_build_promotion_record_marks_ineligible_when_gaincal_fallback(tmp_path):
    m = _FakeManifest()
    m.gaincal_status = "fallback"
    m.gates = [{"gate": "gaincal", "verdict": "FALLBACK", "reason": "phase_dir bug"}]
    m.pipeline_verdict = "DEGRADED"
    rec = build_promotion_record(m, hour=2, products_dir=str(tmp_path))
    assert rec["epoch_gaincal_state"] == "fell_back_to_static_with_reason"
    assert rec["eligible_for_trusted_baseline"] is False
    assert "epoch_gaincal_state" in (rec["eligible_for_trusted_baseline_reason"] or "")
    assert rec["epoch_gaincal_reason"] == "phase_dir bug"


def test_build_promotion_record_marks_ineligible_when_tier_borrowed(tmp_path):
    m = _FakeManifest()
    m.cal_selection = {"source": "borrowed", "borrowed_from": "2026-01-25"}
    rec = build_promotion_record(m, hour=2, products_dir=str(tmp_path))
    assert rec["daily_cal_tier"] == "C"
    assert rec["eligible_for_trusted_baseline"] is False
    assert rec["cal_provenance"]["borrowed_from"] == "2026-01-25"


def test_write_promotion_sidecar_creates_valid_json(tmp_path):
    m = _FakeManifest()
    products_dir = tmp_path / "products" / "mosaics" / "2026-01-25"
    products_dir.mkdir(parents=True)
    out = write_promotion_sidecar(m, hour=2, products_dir=str(products_dir))
    assert os.path.exists(out)
    with open(out) as f:
        rec = json.load(f)
    assert rec["date"] == "2026-01-25"
    assert rec["hour"] == 2
    assert rec["promotion_class"] in PROMOTION_CLASSES
    assert rec["daily_cal_tier"] in DAILY_CAL_TIERS
    assert rec["epoch_gaincal_state"] in EPOCH_GAINCAL_STATES
    assert out == sidecar_path(str(products_dir), m.date, 2)


def test_emit_for_run_writes_sidecar_and_appends_ledger(tmp_path):
    m = _FakeManifest()
    products_root = tmp_path / "products"
    products_dir = products_root / "mosaics" / "2026-01-25"
    products_dir.mkdir(parents=True)
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    written = emit_for_run(
        m,
        str(products_dir),
        str(repo_root),
        cli_invocation=m.command_line,
        skip_epoch_gaincal=False,
        products_root=str(products_root),
    )
    assert len(written) == 1
    assert os.path.exists(written[0])
    ledger_path = repo_root / "docs" / "validation" / "promotion-log.md"
    assert ledger_path.exists()
    text = ledger_path.read_text()
    assert "2026-01-25" in text
    assert "abc1234" in text
    assert "auto_emitted_pending_review" in text


def test_emit_for_run_is_idempotent_for_same_sha(tmp_path):
    m = _FakeManifest()
    products_root = tmp_path / "products"
    products_dir = products_root / "mosaics" / "2026-01-25"
    products_dir.mkdir(parents=True)
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    emit_for_run(
        m,
        str(products_dir),
        str(repo_root),
        cli_invocation=m.command_line,
        products_root=str(products_root),
    )
    emit_for_run(
        m,
        str(products_dir),
        str(repo_root),
        cli_invocation=m.command_line,
        products_root=str(products_root),
    )
    ledger_path = repo_root / "docs" / "validation" / "promotion-log.md"
    rows = [ln for ln in ledger_path.read_text().splitlines() if ln.startswith("| 2026-01-25 |")]
    assert len(rows) == 1


def test_emit_for_run_no_epochs_writes_nothing(tmp_path):
    m = _FakeManifest()
    m.epochs = []
    products_dir = tmp_path / "products"
    products_dir.mkdir(parents=True)
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    written = emit_for_run(m, str(products_dir), str(repo_root))
    assert written == []
    assert not (repo_root / "docs" / "validation" / "promotion-log.md").exists()
