"""Tests for epoch binning and tile overlap logic in batch_pipeline.

These functions determine which tiles go into each hourly mosaic.
Edge cases (single-hour days, midnight crossings, unparseable timestamps)
can silently produce wrong epoch assignments or missing overlap tiles.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from batch_pipeline import timestamp_from_fits, bin_tiles_by_hour, build_epoch_tile_sets


# ── Helpers ───────────────────────────────────────────────────────────────────

def _tile(datestr: str, hour: int, minute: int, second: int) -> str:
    """Build a fake tile FITS path with a parseable timestamp."""
    return f"/fake/{datestr}T{hour:02d}:{minute:02d}:{second:02d}-image-pb.fits"


# ── timestamp_from_fits ───────────────────────────────────────────────────────

class TestTimestampFromFits:
    def test_valid_timestamp(self):
        dt = timestamp_from_fits("/data/2026-01-25T21:17:33-image-pb.fits")
        assert dt is not None
        assert dt.hour == 21
        assert dt.minute == 17

    def test_unparseable_returns_none(self):
        assert timestamp_from_fits("/data/garbage-image-pb.fits") is None

    def test_missing_image_suffix(self):
        # No "-image" in name → split fails gracefully
        assert timestamp_from_fits("/data/2026-01-25T21:17:33.fits") is None


# ── bin_tiles_by_hour ─────────────────────────────────────────────────────────

class TestBinTilesByHour:
    def test_single_hour(self):
        tiles = [_tile("2026-01-25", 21, m, 0) for m in range(0, 50, 10)]
        epochs = bin_tiles_by_hour(tiles)
        assert list(epochs.keys()) == [21]
        assert len(epochs[21]) == 5

    def test_multiple_hours(self):
        tiles = [
            _tile("2026-01-25", 20, 30, 0),
            _tile("2026-01-25", 21, 10, 0),
            _tile("2026-01-25", 21, 20, 0),
            _tile("2026-01-25", 22, 5, 0),
        ]
        epochs = bin_tiles_by_hour(tiles)
        assert sorted(epochs.keys()) == [20, 21, 22]
        assert len(epochs[21]) == 2

    def test_unparseable_tiles_dropped(self):
        tiles = [
            _tile("2026-01-25", 21, 0, 0),
            "/fake/garbage-image-pb.fits",
        ]
        epochs = bin_tiles_by_hour(tiles)
        assert len(epochs[21]) == 1

    def test_sorted_within_hour(self):
        tiles = [
            _tile("2026-01-25", 21, 30, 0),
            _tile("2026-01-25", 21, 10, 0),
            _tile("2026-01-25", 21, 20, 0),
        ]
        epochs = bin_tiles_by_hour(tiles)
        assert epochs[21] == sorted(epochs[21])


# ── build_epoch_tile_sets ─────────────────────────────────────────────────────

class TestBuildEpochTileSets:
    def test_single_epoch_no_overlap(self):
        """A day with only one hour of data should have no overlap tiles."""
        epochs = {21: ["tile_a", "tile_b", "tile_c"]}
        result = build_epoch_tile_sets(epochs)
        assert len(result) == 1
        hour, tiles = result[0]
        assert hour == 21
        assert tiles == ["tile_a", "tile_b", "tile_c"]

    def test_two_epochs_overlap(self):
        """Two adjacent hours: each epoch borrows 2 tiles from the other."""
        epochs = {
            20: [f"h20_{i}" for i in range(5)],
            21: [f"h21_{i}" for i in range(5)],
        }
        result = build_epoch_tile_sets(epochs)
        assert len(result) == 2

        # Hour 20: core + first 2 of hour 21
        h20_tiles = result[0][1]
        assert h20_tiles == [f"h20_{i}" for i in range(5)] + ["h21_0", "h21_1"]

        # Hour 21: last 2 of hour 20 + core
        h21_tiles = result[1][1]
        assert h21_tiles == ["h20_3", "h20_4"] + [f"h21_{i}" for i in range(5)]

    def test_three_epochs_middle_has_both_overlaps(self):
        """Middle epoch should borrow from both neighbors."""
        epochs = {
            10: ["a", "b", "c"],
            11: ["d", "e", "f"],
            12: ["g", "h", "i"],
        }
        result = build_epoch_tile_sets(epochs)
        assert len(result) == 3

        # Middle epoch (hour 11): last 2 of h10 + core + first 2 of h12
        mid_tiles = result[1][1]
        assert mid_tiles == ["b", "c", "d", "e", "f", "g", "h"]

    def test_non_contiguous_hours(self):
        """Hours don't have to be adjacent (e.g., gap from 10 to 14)."""
        epochs = {
            10: ["a", "b"],
            14: ["c", "d"],
        }
        result = build_epoch_tile_sets(epochs)
        assert len(result) == 2

        # Hour 10 borrows from 14 (its next neighbor), hour 14 borrows from 10
        h10_tiles = result[0][1]
        assert h10_tiles == ["a", "b", "c", "d"]

        h14_tiles = result[1][1]
        assert h14_tiles == ["a", "b", "c", "d"]

    def test_short_epoch_fewer_than_2_tiles(self):
        """If an epoch has only 1 tile, overlap should still work (borrow what's available)."""
        epochs = {
            20: ["only_tile"],
            21: ["a", "b", "c", "d", "e"],
        }
        result = build_epoch_tile_sets(epochs)

        # Hour 20: 1 core + first 2 of hour 21
        h20_tiles = result[0][1]
        assert h20_tiles == ["only_tile", "a", "b"]

        # Hour 21: last 1 of hour 20 (only 1 available) + 5 core
        h21_tiles = result[1][1]
        assert h21_tiles == ["only_tile", "a", "b", "c", "d", "e"]

    def test_chronological_order_preserved(self):
        """Output should be sorted by hour regardless of input dict order."""
        epochs = {22: ["z"], 20: ["x"], 21: ["y"]}
        result = build_epoch_tile_sets(epochs)
        hours = [h for h, _ in result]
        assert hours == [20, 21, 22]
