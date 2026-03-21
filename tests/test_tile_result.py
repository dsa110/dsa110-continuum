"""Tests for TileResult dataclass (mosaic_day.py)."""

import dataclasses
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from mosaic_day import TileResult


class TestTileResultOk:
    """TileResult.ok property correctly distinguishes success from failure."""

    def test_imaged_is_ok(self):
        r = TileResult("imaged", fits_path="/tmp/tile.fits")
        assert r.ok is True

    def test_cached_is_ok(self):
        r = TileResult("cached", fits_path="/tmp/tile.fits")
        assert r.ok is True

    def test_failed_is_not_ok(self):
        r = TileResult("failed", failed_stage="imaging", error="WSClean crashed")
        assert r.ok is False


class TestTileResultSerialization:
    """TileResult round-trips through dict serialization."""

    @pytest.mark.parametrize("status,kwargs", [
        ("imaged", {"fits_path": "/data/tile-image-pb.fits"}),
        ("cached", {"fits_path": "/data/tile-image.fits"}),
        ("failed", {"failed_stage": "phaseshift", "error": "bad MS"}),
        ("failed", {"failed_stage": "timeout", "error": "exceeded 1800s"}),
    ])
    def test_roundtrip(self, status, kwargs):
        original = TileResult(status, **kwargs)
        restored = TileResult.from_dict(original.to_dict())
        # Compare via dicts to avoid class-identity issues when mosaic_day
        # gets reloaded by other tests (test_qa_gates uses importlib.reload)
        assert restored.to_dict() == original.to_dict()

    def test_to_dict_returns_plain_dict(self):
        r = TileResult("imaged", fits_path="/tmp/x.fits")
        d = r.to_dict()
        assert isinstance(d, dict)
        assert d["status"] == "imaged"
        assert d["fits_path"] == "/tmp/x.fits"
        assert d["failed_stage"] is None
        assert d["error"] is None


class TestTileResultFrozen:
    """TileResult is immutable (frozen dataclass)."""

    def test_cannot_set_status(self):
        r = TileResult("imaged", fits_path="/tmp/tile.fits")
        with pytest.raises(dataclasses.FrozenInstanceError):
            r.status = "failed"

    def test_cannot_set_fits_path(self):
        r = TileResult("cached", fits_path="/tmp/tile.fits")
        with pytest.raises(dataclasses.FrozenInstanceError):
            r.fits_path = "/tmp/other.fits"
