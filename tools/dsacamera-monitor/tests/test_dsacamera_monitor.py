# ruff: noqa: D103
"""Tests for DSA-110 incoming HDF5 manifest and gap logic."""

from __future__ import annotations

from datetime import date, datetime, timezone
from pathlib import Path

from dsacamera_monitor.gaps import compute_gaps, gaps_from_by_day_rows
from dsacamera_monitor.hdf5_pointing import read_phase_center_dec_deg
from dsacamera_monitor.manifest import (
    BeamAgg,
    DayAgg,
    ScanAccum,
    build_manifest,
    try_parse_filename,
)


def test_try_parse_filename_valid() -> None:
    p = try_parse_filename("2025-05-05T12:34:56_sb03.hdf5")
    assert p is not None
    dt, beam = p
    assert beam == 3
    assert dt == datetime(2025, 5, 5, 12, 34, 56, tzinfo=timezone.utc)


def test_try_parse_filename_invalid() -> None:
    assert try_parse_filename("not_a_match.hdf5") is None
    assert try_parse_filename("2025-05-05T12:34:56_sb03.txt") is None


def test_compute_gaps_middle() -> None:
    days = {date(2025, 1, 1), date(2025, 1, 5)}
    gaps = compute_gaps(days, date(2025, 1, 1), date(2025, 1, 5))
    assert len(gaps) == 1
    assert gaps[0]["start"] == "2025-01-02"
    assert gaps[0]["end"] == "2025-01-04"
    assert gaps[0]["days"] == 3


def test_compute_gaps_none() -> None:
    days = {date(2025, 6, 1)}
    gaps = compute_gaps(days, date(2025, 6, 1), date(2025, 6, 1))
    assert gaps == []


def test_gaps_from_by_day_rows() -> None:
    rows = [
        {"date": "2025-01-01", "count": 1, "bytes": 0},
        {"date": "2025-01-03", "count": 1, "bytes": 0},
    ]
    g = gaps_from_by_day_rows(rows)
    assert len(g) == 1
    assert g[0]["start"] == "2025-01-02"
    assert g[0]["end"] == "2025-01-02"
    assert g[0]["days"] == 1


def test_build_manifest_roundtrip() -> None:
    accum = ScanAccum()
    d = date(2025, 1, 1)
    accum.by_day[d] = DayAgg()
    accum.by_day[d].count = 2
    accum.by_day[d].bytes = 100
    accum.by_beam[4] = BeamAgg()
    accum.by_beam[4].count = 2
    accum.by_beam[4].bytes = 100
    accum.file_count = 2
    accum.total_bytes = 100
    accum.latest_filename_dt = datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    accum.earliest_filename_dt = accum.latest_filename_dt
    accum.latest_mtime = datetime(2025, 1, 2, 0, 0, 0, tzinfo=timezone.utc)
    accum.earliest_mtime = accum.latest_mtime

    m = build_manifest(
        source_root="/tmp/incoming",
        accum=accum,
        no_stat=False,
        hdf5_metadata=False,
        generated_at=datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc),
    )
    assert m["schema_version"] == 2
    assert m["totals"]["file_count"] == 2
    assert m["by_day"][0]["date"] == "2025-01-01"
    assert m["gaps"] == []
    assert "pointing" not in m
    assert m["options"]["hdf5_metadata"] is False


def test_scan_directory(tmp_path: Path) -> None:
    from dsacamera_monitor.scan import scan_directory

    (tmp_path / "2025-01-01T00:00:00_sb01.hdf5").write_bytes(b"x")
    (tmp_path / "2025-01-01T01:00:00_sb02.hdf5").write_bytes(b"yy")
    (tmp_path / "skip.txt").write_text("x")
    accum = scan_directory(tmp_path, no_stat=False, hdf5_metadata=False)
    assert accum.file_count == 2
    assert accum.total_bytes == 3
    assert set(accum.by_beam.keys()) == {1, 2}


def test_scan_directory_with_hdf5_dec(tmp_path: Path) -> None:
    import h5py
    import numpy as np
    from dsacamera_monitor.scan import scan_directory

    fn = "2025-06-15T10:00:00_sb05.hdf5"
    p = tmp_path / fn
    with h5py.File(p, "w") as f:
        f.create_dataset(
            "Header/extra_keywords/phase_center_dec",
            data=np.float64(np.deg2rad(40.5)),
        )
    accum = scan_directory(tmp_path, no_stat=True, hdf5_metadata=True)
    assert accum.file_count == 1
    assert accum.files_with_dec == 1
    assert accum.files_dec_missing == 0
    assert accum.global_dec_min is not None and abs(accum.global_dec_min - 40.5) < 1e-6
    assert read_phase_center_dec_deg(p) is not None


def test_build_out_copies_site(tmp_path: Path) -> None:
    from dsacamera_monitor.scan import build_out

    src = tmp_path / "src"
    src.mkdir()
    (src / "2025-01-01T00:00:00_sb01.hdf5").write_bytes(b"x")
    out = tmp_path / "out"
    build_out(root=src, out_dir=out, no_stat=True, hdf5_metadata=False)
    assert (out / "manifest.json").is_file()
    assert (out / "index.html").is_file()
    assert (out / "js" / "dashboard.js").is_file()
    man = (out / "manifest.json").read_text()
    assert '"schema_version": 2' in man


def test_pointing_timeseries_file(tmp_path: Path) -> None:
    import h5py
    import numpy as np
    from dsacamera_monitor.scan import build_out

    src = tmp_path / "src"
    src.mkdir()
    fn = "2025-03-01T08:00:00_sb00.hdf5"
    p = src / fn
    with h5py.File(p, "w") as f:
        f.create_dataset("Header/extra_keywords/phase_center_dec", data=np.float64(0.7))
        f.create_dataset("Header/extra_keywords/ha_phase_center", data=np.float64(0.01))
        f.create_dataset("Header/time_array", data=np.array([2450000.5, 2450000.6]))

    out = tmp_path / "out"
    build_out(
        root=src,
        out_dir=out,
        no_stat=True,
        hdf5_metadata=True,
        pointing_timeseries=True,
        pointing_timeseries_max_files=10,
    )
    assert (out / "pointing_timeseries.json").is_file()
    m = __import__("json").loads((out / "manifest.json").read_text())
    assert m["pointing_timeseries"]["row_count"] == 1
    assert "pointing" in m
