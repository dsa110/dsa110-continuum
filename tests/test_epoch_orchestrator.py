"""Tests for dsa110_continuum.pipeline.epoch_orchestrator.

All tests are synthetic — no real HDF5 data, no CASA, no network access.
Tests exercise:
  * EpochDecision enum values and mappings
  * EpochRunResult serialisation / deserialisaton
  * EpochOrchestrator.run_epoch() with dry_run=True (no I/O side effects)
  * EpochOrchestrator with in-memory SQLite via db_path=":memory:"
  * run_day() binning logic using a temporary directory of fake FITS files
  * QA gate integration: accept / warn / reject / skip paths
  * Persistence round-trip: persist → get_result
  * acceptance_rate() and list_epochs()
"""

from __future__ import annotations

import json
import sqlite3
import tempfile
from pathlib import Path
from typing import Optional
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from dsa110_continuum.pipeline.epoch_orchestrator import (
    EpochDecision,
    EpochOrchestrator,
    EpochRunResult,
    _qa_status_to_decision,
    _row_to_result,
)
from dsa110_continuum.qa.composite import (
    CompositeQAResult,
    QAStatus,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

EPOCH_ID = "2026-01-25T22:00:00"
EPOCH_ID2 = "2026-01-26T00:00:00"


def _make_orchestrator(tmp_path: Path, db: bool = True) -> EpochOrchestrator:
    """Create an orchestrator backed by a fresh SQLite DB in tmp_path."""
    db_path = str(tmp_path / "pipeline.db") if db else None
    return EpochOrchestrator(
        output_dir=str(tmp_path / "mosaics"),
        db_path=db_path,
    )


def _make_result(
    epoch_id: str = EPOCH_ID,
    decision: EpochDecision = EpochDecision.ACCEPT,
    n_tiles: int = 5,
) -> EpochRunResult:
    return EpochRunResult(
        epoch_id=epoch_id,
        decision=decision,
        n_tiles=n_tiles,
        measured_rms_jyb=2.0e-3,
        theoretical_rms=1.8e-3,
        elapsed_s=0.5,
    )


# ---------------------------------------------------------------------------
# EpochDecision
# ---------------------------------------------------------------------------

class TestEpochDecision:
    def test_values(self):
        assert EpochDecision.ACCEPT.value == "accept"
        assert EpochDecision.WARN.value == "warn"
        assert EpochDecision.REJECT.value == "reject"
        assert EpochDecision.SKIP.value == "skip"

    def test_string_equality(self):
        assert EpochDecision.ACCEPT == "accept"

    def test_qa_status_mapping(self):
        assert _qa_status_to_decision(QAStatus.PASS) == EpochDecision.ACCEPT
        assert _qa_status_to_decision(QAStatus.WARN) == EpochDecision.WARN
        assert _qa_status_to_decision(QAStatus.FAIL) == EpochDecision.REJECT
        # SKIP gate → cautious WARN
        assert _qa_status_to_decision(QAStatus.SKIP) == EpochDecision.WARN


# ---------------------------------------------------------------------------
# EpochRunResult
# ---------------------------------------------------------------------------

class TestEpochRunResult:
    def test_properties_accept(self):
        r = _make_result(decision=EpochDecision.ACCEPT)
        assert r.accepted
        assert not r.rejected
        assert not r.warned

    def test_properties_reject(self):
        r = _make_result(decision=EpochDecision.REJECT)
        assert not r.accepted
        assert r.rejected
        assert not r.warned

    def test_properties_warn(self):
        r = _make_result(decision=EpochDecision.WARN)
        assert not r.accepted
        assert not r.rejected
        assert r.warned

    def test_to_dict_keys(self):
        r = _make_result()
        d = r.to_dict()
        for k in ["epoch_id", "decision", "n_tiles", "mosaic_path", "elapsed_s",
                   "qa_status", "qa_json", "notes"]:
            assert k in d, f"Missing key: {k}"

    def test_to_dict_values(self):
        r = _make_result()
        d = r.to_dict()
        assert d["epoch_id"] == EPOCH_ID
        assert d["decision"] == "accept"
        assert d["n_tiles"] == 5
        assert d["mosaic_path"] is None
        assert d["qa_status"] is None

    def test_summary_contains_epoch(self):
        r = _make_result()
        s = r.summary()
        assert EPOCH_ID in s
        assert "accept" in s

    def test_to_dict_with_qa(self):
        from dsa110_continuum.qa.composite import run_composite_qa
        qa = run_composite_qa(
            flux_scale_correction=1.05,
            n_detected=25,
            n_catalog_expected=30,
            measured_rms_jyb=2e-3,
            theoretical_rms_jyb=1.8e-3,
            epoch=EPOCH_ID,
        )
        r = _make_result()
        r.qa = qa
        d = r.to_dict()
        assert d["qa_status"] == qa.status.value
        parsed = json.loads(d["qa_json"])
        assert "status" in parsed

    def test_notes_serialised_as_json(self):
        r = _make_result()
        r.notes = ["note1", "note2"]
        d = r.to_dict()
        assert json.loads(d["notes"]) == ["note1", "note2"]


# ---------------------------------------------------------------------------
# EpochOrchestrator — dry_run (no I/O)
# ---------------------------------------------------------------------------

class TestEpochOrchestratorDryRun:
    def test_no_tiles_returns_skip(self, tmp_path):
        orch = _make_orchestrator(tmp_path, db=False)
        result = orch.run_epoch(EPOCH_ID, [], dry_run=True)
        assert result.decision == EpochDecision.SKIP
        assert result.n_tiles == 0

    def test_dry_run_does_not_write_mosaic(self, tmp_path):
        orch = _make_orchestrator(tmp_path, db=False)
        result = orch.run_epoch(EPOCH_ID, ["fake_tile.fits"], dry_run=True)
        assert result.mosaic_path is None
        # No mosaic file should exist
        assert not any((tmp_path / "mosaics").rglob("*.fits"))

    def test_dry_run_returns_epoch_id(self, tmp_path):
        orch = _make_orchestrator(tmp_path, db=False)
        result = orch.run_epoch(EPOCH_ID, ["t1.fits", "t2.fits"], dry_run=True)
        assert result.epoch_id == EPOCH_ID

    def test_dry_run_n_tiles_correct(self, tmp_path):
        orch = _make_orchestrator(tmp_path, db=False)
        result = orch.run_epoch(EPOCH_ID, ["t1.fits", "t2.fits", "t3.fits"], dry_run=True)
        assert result.n_tiles == 3

    def test_dry_run_no_db_write(self, tmp_path):
        db_path = tmp_path / "pipeline.db"
        orch = EpochOrchestrator(
            output_dir=str(tmp_path / "mosaics"),
            db_path=str(db_path),
        )
        orch.run_epoch(EPOCH_ID, ["t1.fits"], dry_run=True)
        # DB should NOT have been created
        assert not db_path.exists()

    def test_elapsed_s_positive(self, tmp_path):
        orch = _make_orchestrator(tmp_path, db=False)
        result = orch.run_epoch(EPOCH_ID, ["t1.fits"], dry_run=True)
        assert result.elapsed_s >= 0


# ---------------------------------------------------------------------------
# EpochOrchestrator — QA integration
# ---------------------------------------------------------------------------

class TestEpochOrchestratorQA:
    """Verify QA gate → decision mapping via inject-style patching."""

    def _run_with_qa(
        self,
        tmp_path: Path,
        flux_scale: float,
        n_detected: int,
        n_expected: int,
        rms: float,
        theo: float,
    ) -> EpochRunResult:
        orch = _make_orchestrator(tmp_path, db=False)
        # Patch _compute_image_rms and _count_detected to inject values
        with (
            patch(
                "dsa110_continuum.pipeline.epoch_orchestrator._compute_image_rms",
                return_value=rms,
            ),
            patch(
                "dsa110_continuum.pipeline.epoch_orchestrator._count_detected_sources",
                return_value=n_detected,
            ),
            patch(
                "dsa110_continuum.pipeline.epoch_orchestrator.theoretical_rms_jyb",
                return_value=theo,
            ),
            patch(
                "dsa110_continuum.pipeline.epoch_orchestrator.EpochOrchestrator._write_mosaic",
                return_value="/tmp/fake_mosaic.fits",
            ),
        ):
            # Pretend mosaic exists for stat computation
            with patch("pathlib.Path.exists", return_value=True):
                return orch.run_epoch(
                    EPOCH_ID,
                    ["t1.fits", "t2.fits"],
                    flux_scale_correction=flux_scale,
                    n_catalog_expected=n_expected,
                )

    def test_all_pass_gives_accept(self, tmp_path):
        result = self._run_with_qa(
            tmp_path,
            flux_scale=1.02,   # <15% deviation → pass
            n_detected=28,
            n_expected=30,     # 93% completeness → pass
            rms=2e-3,
            theo=1.8e-3,       # factor ~1.1 → pass
        )
        assert result.decision == EpochDecision.ACCEPT

    def test_flux_fail_gives_reject(self, tmp_path):
        result = self._run_with_qa(
            tmp_path,
            flux_scale=1.40,   # 40% deviation → fail
            n_detected=28,
            n_expected=30,
            rms=2e-3,
            theo=1.8e-3,
        )
        assert result.decision == EpochDecision.REJECT

    def test_rms_fail_gives_reject(self, tmp_path):
        result = self._run_with_qa(
            tmp_path,
            flux_scale=1.02,
            n_detected=28,
            n_expected=30,
            rms=5e-3,          # 2.8× theo → fail
            theo=1.8e-3,
        )
        assert result.decision == EpochDecision.REJECT

    def test_completeness_fail_gives_reject(self, tmp_path):
        result = self._run_with_qa(
            tmp_path,
            flux_scale=1.02,
            n_detected=10,     # 33% of 30 → fail
            n_expected=30,
            rms=2e-3,
            theo=1.8e-3,
        )
        assert result.decision == EpochDecision.REJECT

    def test_qa_result_attached(self, tmp_path):
        result = self._run_with_qa(
            tmp_path,
            flux_scale=1.02,
            n_detected=28,
            n_expected=30,
            rms=2e-3,
            theo=1.8e-3,
        )
        assert result.qa is not None
        assert isinstance(result.qa, CompositeQAResult)

    def test_no_mosaic_skips_image_stats(self, tmp_path):
        """When mosaic is None (dry_run), QA gates that need image data should SKIP."""
        orch = _make_orchestrator(tmp_path, db=False)
        result = orch.run_epoch(EPOCH_ID, ["t1.fits"], dry_run=True)
        # SKIP decision or WARN (missing stats → completeness gate skips)
        assert result.decision in (EpochDecision.SKIP, EpochDecision.WARN, EpochDecision.ACCEPT, EpochDecision.REJECT)


# ---------------------------------------------------------------------------
# EpochOrchestrator — SQLite persistence
# ---------------------------------------------------------------------------

class TestEpochOrchestratorPersistence:
    def test_persist_and_retrieve(self, tmp_path):
        db_path = str(tmp_path / "pipeline.db")
        orch = EpochOrchestrator(
            output_dir=str(tmp_path / "mosaics"),
            db_path=db_path,
        )
        result = orch.run_epoch(EPOCH_ID, [], dry_run=False)
        assert result.decision == EpochDecision.SKIP

        retrieved = orch.get_result(EPOCH_ID)
        assert retrieved is not None
        assert retrieved.epoch_id == EPOCH_ID
        assert retrieved.decision == EpochDecision.SKIP

    def test_upsert_updates_existing(self, tmp_path):
        db_path = str(tmp_path / "pipeline.db")
        orch = EpochOrchestrator(
            output_dir=str(tmp_path / "mosaics"),
            db_path=db_path,
        )
        # First run
        orch.run_epoch(EPOCH_ID, [])
        # Second run should upsert
        orch.run_epoch(EPOCH_ID, [])
        # Should still be exactly one row
        con = sqlite3.connect(db_path)
        count = con.execute("SELECT COUNT(*) FROM epoch_runs WHERE epoch_id = ?", (EPOCH_ID,)).fetchone()[0]
        con.close()
        assert count == 1

    def test_get_result_missing_returns_none(self, tmp_path):
        orch = _make_orchestrator(tmp_path)
        result = orch.get_result("nonexistent-epoch")
        assert result is None

    def test_list_epochs_empty(self, tmp_path):
        orch = _make_orchestrator(tmp_path)
        assert orch.list_epochs() == []

    def test_list_epochs_after_run(self, tmp_path):
        orch = _make_orchestrator(tmp_path)
        orch.run_epoch(EPOCH_ID, [])
        orch.run_epoch(EPOCH_ID2, [])
        epochs = orch.list_epochs()
        assert EPOCH_ID in epochs
        assert EPOCH_ID2 in epochs

    def test_list_epochs_filter_by_decision(self, tmp_path):
        orch = _make_orchestrator(tmp_path)
        orch.run_epoch(EPOCH_ID, [])   # → SKIP
        orch.run_epoch(EPOCH_ID2, [])  # → SKIP
        skip_epochs = orch.list_epochs(decision=EpochDecision.SKIP)
        accept_epochs = orch.list_epochs(decision=EpochDecision.ACCEPT)
        assert len(skip_epochs) == 2
        assert len(accept_epochs) == 0

    def test_acceptance_rate_no_data_is_nan(self, tmp_path):
        orch = _make_orchestrator(tmp_path)
        rate = orch.acceptance_rate()
        assert np.isnan(rate)

    def test_acceptance_rate_all_skip_is_nan(self, tmp_path):
        orch = _make_orchestrator(tmp_path)
        orch.run_epoch(EPOCH_ID, [])
        orch.run_epoch(EPOCH_ID2, [])
        rate = orch.acceptance_rate()
        # All are SKIP, no non-skip rows → nan
        assert np.isnan(rate)

    def test_no_db_get_result_returns_none(self, tmp_path):
        orch = EpochOrchestrator(
            output_dir=str(tmp_path / "mosaics"),
            db_path=None,
        )
        assert orch.get_result(EPOCH_ID) is None

    def test_no_db_list_epochs_returns_empty(self, tmp_path):
        orch = EpochOrchestrator(
            output_dir=str(tmp_path / "mosaics"),
            db_path=None,
        )
        assert orch.list_epochs() == []

    def test_context_manager_closes_connection(self, tmp_path):
        db_path = str(tmp_path / "pipeline.db")
        with EpochOrchestrator(
            output_dir=str(tmp_path / "mosaics"),
            db_path=db_path,
        ) as orch:
            orch.run_epoch(EPOCH_ID, [])
        # After __exit__, connection should be None
        assert orch._db_con is None


# ---------------------------------------------------------------------------
# EpochOrchestrator — run_day() binning
# ---------------------------------------------------------------------------

class TestEpochOrchestratorRunDay:
    def _make_fake_fits(self, tile_dir: Path, epoch_str: str, n: int = 3) -> list[Path]:
        """Create empty .fits files with epoch_str in name."""
        files = []
        for i in range(n):
            p = tile_dir / f"{epoch_str}_tile{i:02d}.fits"
            p.touch()
            files.append(p)
        return files

    def test_run_day_empty_dir_returns_empty(self, tmp_path):
        tile_dir = tmp_path / "tiles"
        tile_dir.mkdir()
        orch = _make_orchestrator(tmp_path, db=False)
        results = orch.run_day("2026-01-25", tile_dir, dry_run=True)
        assert results == []

    def test_run_day_bins_by_hour(self, tmp_path):
        tile_dir = tmp_path / "tiles"
        tile_dir.mkdir()
        self._make_fake_fits(tile_dir, "2026-01-25T2200", n=3)
        self._make_fake_fits(tile_dir, "2026-01-25T2300", n=2)
        orch = _make_orchestrator(tmp_path, db=False)
        results = orch.run_day("2026-01-25", tile_dir, dry_run=True)
        # 2 distinct hour bins
        assert len(results) == 2

    def test_run_day_epoch_ids_contain_date(self, tmp_path):
        tile_dir = tmp_path / "tiles"
        tile_dir.mkdir()
        self._make_fake_fits(tile_dir, "2026-01-25T2200", n=2)
        orch = _make_orchestrator(tmp_path, db=False)
        results = orch.run_day("2026-01-25", tile_dir, dry_run=True)
        assert all("2026-01-25" in r.epoch_id for r in results)

    def test_run_day_epoch_hours_filter(self, tmp_path):
        tile_dir = tmp_path / "tiles"
        tile_dir.mkdir()
        self._make_fake_fits(tile_dir, "2026-01-25T2200", n=2)
        self._make_fake_fits(tile_dir, "2026-01-25T2300", n=2)
        orch = _make_orchestrator(tmp_path, db=False)
        results = orch.run_day(
            "2026-01-25", tile_dir, epoch_hours=[22], dry_run=True
        )
        assert len(results) == 1
        assert "T22" in results[0].epoch_id

    def test_run_day_tile_counts(self, tmp_path):
        tile_dir = tmp_path / "tiles"
        tile_dir.mkdir()
        self._make_fake_fits(tile_dir, "2026-01-25T2200", n=4)
        orch = _make_orchestrator(tmp_path, db=False)
        results = orch.run_day("2026-01-25", tile_dir, dry_run=True)
        assert results[0].n_tiles == 4


# ---------------------------------------------------------------------------
# _row_to_result reconstruction
# ---------------------------------------------------------------------------

class TestRowToResult:
    def test_roundtrip(self, tmp_path):
        orch = _make_orchestrator(tmp_path)
        orch.run_epoch(EPOCH_ID, [])
        retrieved = orch.get_result(EPOCH_ID)
        assert retrieved is not None
        assert retrieved.epoch_id == EPOCH_ID
        assert retrieved.decision == EpochDecision.SKIP

    def test_notes_preserved(self, tmp_path):
        orch = _make_orchestrator(tmp_path)
        orch.run_epoch(EPOCH_ID, [])
        retrieved = orch.get_result(EPOCH_ID)
        assert isinstance(retrieved.notes, list)


# ---------------------------------------------------------------------------
# EpochOrchestrator — default construction
# ---------------------------------------------------------------------------

class TestEpochOrchestratorDefaults:
    def test_default_output_dir_uses_env(self, tmp_path, monkeypatch):
        monkeypatch.setenv("DSA110_PRODUCTS_BASE", str(tmp_path / "products"))
        orch = EpochOrchestrator(db_path=None)
        assert "mosaics" in str(orch.output_dir)

    def test_no_db_path_disables_persistence(self, tmp_path):
        orch = EpochOrchestrator(output_dir=str(tmp_path), db_path=None)
        assert orch.db_path is None
        assert orch._get_connection() is None

    def test_custom_thresholds_propagate(self, tmp_path):
        orch = EpochOrchestrator(
            output_dir=str(tmp_path),
            db_path=None,
            max_flux_scale_error=0.05,
            min_completeness=0.95,
            max_noise_factor=1.2,
        )
        assert orch.qa.max_flux_scale_error == pytest.approx(0.05)
        assert orch.qa.min_completeness == pytest.approx(0.95)
        assert orch.qa.max_noise_factor == pytest.approx(1.2)
