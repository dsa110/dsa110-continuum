"""Tests for Measurement Set discovery diagnostics in mosaic_day.py."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import mosaic_day
from mosaic_day import TileConfig


class _OpenableTable:
    def __init__(self, tablename: str, readonly: bool = True, ack: bool = False):
        self.tablename = tablename
        self.readonly = readonly
        self.ack = ack

    def __enter__(self):
        return self

    def __exit__(self, *exc: object) -> None:
        return None


class _FailingTable:
    def __init__(self, *args: object, **kwargs: object):
        raise RuntimeError("adapter unavailable")


def _cfg(tmp_path: Path) -> TileConfig:
    return TileConfig(
        date="2026-01-25",
        ms_dir=str(tmp_path),
        image_dir=str(tmp_path / "images"),
        mosaic_out=str(tmp_path / "mosaic.fits"),
        products_dir=str(tmp_path / "products"),
        bp_table=str(tmp_path / "cal.b"),
        g_table=str(tmp_path / "cal.g"),
    )


def _ms(tmp_path: Path, name: str, *, field: str = "dir") -> Path:
    path = tmp_path / name
    path.mkdir()
    if field == "dir":
        (path / "FIELD").mkdir()
    elif field == "file":
        (path / "FIELD").write_text("not a table")
    elif field != "missing":
        raise ValueError(f"unknown field mode: {field}")
    return path


def test_find_valid_ms_returns_openable_field_table(tmp_path, monkeypatch):
    ms_path = _ms(tmp_path, "2026-01-25T22:00:18.ms")
    monkeypatch.setattr("dsa110_continuum.adapters.casa_tables.table", _OpenableTable)

    assert mosaic_day.find_valid_ms(_cfg(tmp_path)) == [str(ms_path)]


def test_find_valid_ms_reports_missing_field_table(tmp_path, monkeypatch, caplog):
    _ms(tmp_path, "2026-01-25T22:00:18.ms", field="missing")
    monkeypatch.setattr("dsa110_continuum.adapters.casa_tables.table", _OpenableTable)

    with caplog.at_level(logging.WARNING, logger="mosaic_day"):
        assert mosaic_day.find_valid_ms(_cfg(tmp_path)) == []

    assert "missing FIELD table" in caplog.text


def test_find_valid_ms_reports_field_table_open_failure(tmp_path, monkeypatch, caplog):
    _ms(tmp_path, "2026-01-25T22:00:18.ms")
    monkeypatch.setattr("dsa110_continuum.adapters.casa_tables.table", _FailingTable)

    with caplog.at_level(logging.WARNING, logger="mosaic_day"):
        assert mosaic_day.find_valid_ms(_cfg(tmp_path)) == []

    assert "unreadable FIELD table" in caplog.text
    assert "RuntimeError: adapter unavailable" in caplog.text


def test_find_valid_ms_filters_meridian_and_flagversion_paths(tmp_path, monkeypatch):
    valid_ms = _ms(tmp_path, "2026-01-25T22:00:18.ms")
    _ms(tmp_path, "2026-01-25T22:05:28_meridian.ms")
    _ms(tmp_path, "2026-01-25T22:10:37_flagversion.ms")
    monkeypatch.setattr("dsa110_continuum.adapters.casa_tables.table", _OpenableTable)

    assert mosaic_day.find_valid_ms(_cfg(tmp_path)) == [str(valid_ms)]


def test_find_valid_ms_reports_field_path_that_is_not_directory(
    tmp_path,
    monkeypatch,
    caplog,
):
    _ms(tmp_path, "2026-01-25T22:00:18.ms", field="file")
    monkeypatch.setattr("dsa110_continuum.adapters.casa_tables.table", _OpenableTable)

    with caplog.at_level(logging.WARNING, logger="mosaic_day"):
        assert mosaic_day.find_valid_ms(_cfg(tmp_path)) == []

    assert "FIELD table is not a directory" in caplog.text
