"""Tests for the development-tools bundle (check_import_migration, canary_history,
inspect_epoch_artifacts).

These tests cover the pure-logic functions that are testable without real
pipeline data on disk.  File-system traversal is exercised with tmp_path.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Make scripts importable (ruff: noqa for non-standard import order)
REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from canary_history import _append_log, _assess_change, _canary_pass, _fmt_delta, _load_log  # noqa: I001
from check_import_migration import file_to_module, scan_stale_imports
from inspect_epoch_artifacts import (
    _file_status,
    _find_qa_row,
    _gaincal_status,
    _infer_context_from_mosaic_path,
)


# ── check_import_migration ─────────────────────────────────────────────────────


class TestScanStaleImports:
    def test_detects_from_import(self, tmp_path):
        f = tmp_path / "a.py"
        f.write_text("from dsa110_contimg.core.calibration import foo\n")
        hits = scan_stale_imports(tmp_path)
        assert f in hits
        assert len(hits[f]) == 1
        assert hits[f][0][0] == 1  # line number

    def test_detects_bare_import(self, tmp_path):
        f = tmp_path / "b.py"
        f.write_text("import dsa110_contimg.core\n")
        hits = scan_stale_imports(tmp_path)
        assert f in hits

    def test_ignores_comment_line(self, tmp_path):
        f = tmp_path / "c.py"
        f.write_text("# from dsa110_contimg.core import foo\n")
        hits = scan_stale_imports(tmp_path)
        assert f not in hits

    def test_ignores_string_literal(self, tmp_path):
        f = tmp_path / "d.py"
        f.write_text('x = "from dsa110_contimg.core import foo"\n')
        hits = scan_stale_imports(tmp_path)
        assert f not in hits

    def test_counts_multiple_lines(self, tmp_path):
        f = tmp_path / "e.py"
        f.write_text(
            "from dsa110_contimg.core.a import x\n"
            "from dsa110_contimg.core.b import y\n"
        )
        hits = scan_stale_imports(tmp_path)
        assert len(hits[f]) == 2

    def test_clean_file_not_in_hits(self, tmp_path):
        f = tmp_path / "clean.py"
        f.write_text("from dsa110_continuum.photometry import epoch_qa\n")
        hits = scan_stale_imports(tmp_path)
        assert f not in hits

    def test_empty_directory_returns_empty(self, tmp_path):
        hits = scan_stale_imports(tmp_path)
        assert hits == {}

    def test_recursive_scan(self, tmp_path):
        sub = tmp_path / "sub"
        sub.mkdir()
        f = sub / "nested.py"
        f.write_text("from dsa110_contimg.core import x\n")
        hits = scan_stale_imports(tmp_path)
        assert f in hits


class TestFileToModule:
    def test_simple_module(self, tmp_path):
        pkg = tmp_path / "dsa110_continuum"
        pkg.mkdir()
        f = pkg / "photometry" / "epoch_qa.py"
        f.parent.mkdir(parents=True)
        result = file_to_module(f, pkg)
        assert result == "dsa110_continuum.photometry.epoch_qa"

    def test_init_file_drops_init(self, tmp_path):
        pkg = tmp_path / "dsa110_continuum"
        pkg.mkdir()
        f = pkg / "calibration" / "__init__.py"
        f.parent.mkdir(parents=True)
        result = file_to_module(f, pkg)
        assert result == "dsa110_continuum.calibration"

    def test_returns_none_for_outside_root(self, tmp_path):
        f = Path("/tmp/unrelated.py")
        result = file_to_module(f, tmp_path)
        assert result is None


# ── canary_history ─────────────────────────────────────────────────────────────


class TestCanaryLog:
    def test_load_empty_log(self, tmp_path):
        log = tmp_path / "canary.jsonl"
        assert _load_log(log) == []

    def test_load_nonexistent_log(self, tmp_path):
        log = tmp_path / "no_such_file.jsonl"
        assert _load_log(log) == []

    def test_append_and_reload(self, tmp_path):
        log = tmp_path / "canary.jsonl"
        entry = {"timestamp": "2026-01-25T00:00:00Z", "median_ratio": 1.01, "n_recovered": 5}
        _append_log(log, entry)
        loaded = _load_log(log)
        assert len(loaded) == 1
        assert loaded[0]["median_ratio"] == pytest.approx(1.01)

    def test_append_multiple(self, tmp_path):
        log = tmp_path / "canary.jsonl"
        for i in range(3):
            _append_log(log, {"seq": i})
        loaded = _load_log(log)
        assert len(loaded) == 3
        assert [e["seq"] for e in loaded] == [0, 1, 2]

    def test_creates_parent_dirs(self, tmp_path):
        log = tmp_path / "nested" / "deep" / "canary.jsonl"
        _append_log(log, {"x": 1})
        assert log.exists()

    def test_skips_malformed_line(self, tmp_path):
        log = tmp_path / "canary.jsonl"
        log.write_text('{"good": 1}\nNOT JSON\n{"good": 2}\n')
        loaded = _load_log(log)
        assert len(loaded) == 2
        assert all("good" in e for e in loaded)


class TestCanaryPass:
    def _entry(self, ratio=1.0, n=5, rms=10.0):
        return {"median_ratio": ratio, "n_recovered": n, "rms_mjy": rms}

    def test_nominal_passes(self):
        assert _canary_pass(self._entry()) is True

    def test_ratio_too_low(self):
        assert _canary_pass(self._entry(ratio=0.80)) is False

    def test_ratio_too_high(self):
        assert _canary_pass(self._entry(ratio=1.20)) is False

    def test_ratio_at_boundary_lo(self):
        assert _canary_pass(self._entry(ratio=0.85)) is True

    def test_ratio_at_boundary_hi(self):
        assert _canary_pass(self._entry(ratio=1.15)) is True

    def test_n_recovered_too_few(self):
        assert _canary_pass(self._entry(n=2)) is False

    def test_n_recovered_exactly_threshold(self):
        assert _canary_pass(self._entry(n=3)) is True

    def test_rms_too_high(self):
        assert _canary_pass(self._entry(rms=18.0)) is False

    def test_nan_ratio_fails(self):
        assert _canary_pass(self._entry(ratio=float("nan"))) is False


class TestFmtDelta:
    def test_delta_contains_ratio_line(self):
        prev = {"git_commit": "abc", "median_ratio": 1.0, "n_recovered": 5, "rms_mjy": 10.0, "canary_pass": True}
        curr = {"git_commit": "def", "median_ratio": 1.05, "n_recovered": 6, "rms_mjy": 9.5, "canary_pass": True}
        out = _fmt_delta(prev, curr)
        assert "Δratio" in out
        assert "+0.0500" in out or "+0.050" in out

    def test_status_change_flag(self):
        prev = {"git_commit": "a", "median_ratio": 1.0, "n_recovered": 5, "rms_mjy": 10.0, "canary_pass": True}
        curr = {"git_commit": "b", "median_ratio": 0.7, "n_recovered": 2, "rms_mjy": 20.0, "canary_pass": False}
        out = _fmt_delta(prev, curr)
        assert "PASS→FAIL" in out


class TestAssessChange:
    def test_marks_status_flip_as_concerning(self):
        prev = {"median_ratio": 1.0, "n_recovered": 5, "rms_mjy": 10.0, "canary_pass": True}
        curr = {"median_ratio": 0.9, "n_recovered": 4, "rms_mjy": 10.5, "canary_pass": False}
        assessment = _assess_change(prev, curr)
        assert assessment["concerning"] is True
        assert "status changed" in assessment["summary"].lower()

    def test_marks_small_shift_as_stable(self):
        prev = {"median_ratio": 1.00, "n_recovered": 5, "rms_mjy": 10.0, "canary_pass": True}
        curr = {"median_ratio": 1.02, "n_recovered": 5, "rms_mjy": 10.4, "canary_pass": True}
        assessment = _assess_change(prev, curr)
        assert assessment["concerning"] is False
        assert "stable" in assessment["summary"].lower()


# ── inspect_epoch_artifacts ────────────────────────────────────────────────────


class TestFindQaRow:
    def _rows(self):
        return [
            {"date": "2026-01-25", "epoch_utc": "2026-01-25T2200", "qa_result": "warn"},
            {"date": "2026-01-25", "epoch_utc": "2026-01-25T0200", "qa_result": "pass"},
            {"date": "2026-02-12", "epoch_utc": "2026-02-12T0000", "qa_result": "fail"},
        ]

    def test_finds_matching_row(self):
        row = _find_qa_row(self._rows(), "2026-01-25", "2026-01-25T22")
        assert row is not None
        assert row["qa_result"] == "warn"

    def test_returns_none_for_missing_date(self):
        row = _find_qa_row(self._rows(), "2099-01-01", "2099-01-01T00")
        assert row is None

    def test_returns_none_for_wrong_epoch(self):
        row = _find_qa_row(self._rows(), "2026-01-25", "2026-01-25T06")
        assert row is None

    def test_empty_list_returns_none(self):
        assert _find_qa_row([], "2026-01-25", "2026-01-25T22") is None


class TestGaincalStatus:
    def test_missing_tables(self, tmp_path):
        info = _gaincal_status(tmp_path, "2099-01-01")
        assert info["bandpass"]["status"] == "missing"
        assert info["gain"]["status"] == "missing"

    def test_present_tables(self, tmp_path):
        # Create dummy table directories (CASA tables are dirs)
        bp = tmp_path / "2026-01-25T22:26:05_0~23.b"
        g = tmp_path / "2026-01-25T22:26:05_0~23.g"
        bp.mkdir()
        g.mkdir()
        info = _gaincal_status(tmp_path, "2026-01-25")
        assert info["bandpass"]["status"] == "present"
        assert info["gain"]["status"] == "present"

    def test_symlink_table(self, tmp_path):
        # Create a real target and a symlink
        target = tmp_path / "2026-01-25T22:26:05_0~23.b"
        target.mkdir()
        link = tmp_path / "2026-02-12T22:26:05_0~23.b"
        link.symlink_to(target)
        info = _gaincal_status(tmp_path, "2026-02-12")
        assert info["bandpass"]["status"] == "symlink"
        assert "target" in info["bandpass"]


class TestFileStatus:
    def test_existing_file(self, tmp_path):
        f = tmp_path / "test.fits"
        f.write_bytes(b"x" * 1024)
        status = _file_status(f)
        assert status.startswith("OK")
        assert "KB" in status or "B" in status

    def test_missing_file(self, tmp_path):
        f = tmp_path / "nonexistent.fits"
        assert _file_status(f) == "MISSING"


class TestInferContextFromMosaicPath:
    def test_products_mosaic_path_infers_epoch_context(self, tmp_path):
        products_base = tmp_path / "products" / "mosaics"
        stage_base = tmp_path / "stage"
        mosaic_path = products_base / "2026-01-25" / "2026-01-25T2200_mosaic.fits"
        mosaic_path.parent.mkdir(parents=True)
        mosaic_path.write_text("dummy")

        context = _infer_context_from_mosaic_path(mosaic_path, products_base, stage_base)

        assert context["date"] == "2026-01-25"
        assert context["hour"] == 22
        assert context["products_dir"] == products_base / "2026-01-25"
        assert context["active_mosaic"] == mosaic_path

    def test_stage_mosaic_path_infers_products_dir(self, tmp_path):
        products_base = tmp_path / "products" / "mosaics"
        stage_base = tmp_path / "stage"
        mosaic_path = stage_base / "mosaic_2026-01-25" / "2026-01-25T2200_mosaic.fits"
        mosaic_path.parent.mkdir(parents=True)
        mosaic_path.write_text("dummy")

        context = _infer_context_from_mosaic_path(mosaic_path, products_base, stage_base)

        assert context["date"] == "2026-01-25"
        assert context["hour"] == 22
        assert context["stage_dir"] == stage_base / "mosaic_2026-01-25"
        assert context["products_dir"] == products_base / "2026-01-25"
