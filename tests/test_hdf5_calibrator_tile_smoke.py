from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path

import numpy as np
import pytest
from astropy.time import Time
from dsa110_continuum.calibration import model as calibration_model
from dsa110_continuum.evidence import hdf5_calibrator_tile_smoke as smoke
from dsa110_continuum.evidence.hdf5_calibrator_tile_smoke import (
    EXPECTED_INTEGRATIONS,
    EXPECTED_SUBBANDS,
    Calibrator,
    DiscoveryConfig,
    SmokeRunConfig,
    SmokeRunManifest,
    StageResult,
    _assess_pair,
    _cached_transit_time,
    _run_stage,
    create_immutable_run_dir,
    create_work_run_dir,
    discover_calibrator_candidates,
    validate_fresh_cal_tables,
    validate_vla_calibrator_db,
)


def _make_vla_db(path: Path) -> None:
    with sqlite3.connect(path) as conn:
        conn.executescript(
            """
            CREATE TABLE calibrators (
                name TEXT PRIMARY KEY,
                ra_deg REAL NOT NULL,
                dec_deg REAL NOT NULL,
                position_code TEXT,
                alt_name TEXT
            );
            CREATE TABLE fluxes (
                name TEXT NOT NULL,
                band TEXT NOT NULL,
                flux_jy REAL NOT NULL,
                quality_codes TEXT
            );
            """
        )
        conn.execute(
            "INSERT INTO calibrators VALUES (?, ?, ?, ?, ?)",
            ("3C138", 80.2912, 16.6394, "A", "J0521+1638"),
        )
        conn.execute(
            "INSERT INTO calibrators VALUES (?, ?, ?, ?, ?)",
            ("J9999+1600", 80.2912, 16.0, "A", None),
        )
        conn.execute("INSERT INTO fluxes VALUES (?, ?, ?, ?)", ("3C138", "20cm", 8.3, ""))
        conn.execute("INSERT INTO fluxes VALUES (?, ?, ?, ?)", ("J9999+1600", "20cm", 6.0, ""))


def _make_pipeline_db(path: Path, hdf5_root: Path, *, complete: bool = True) -> None:
    hdf5_root.mkdir()
    group_id = "2026-01-25T04:56:12"
    subband_count = EXPECTED_SUBBANDS if complete else EXPECTED_SUBBANDS - 1
    with sqlite3.connect(path) as conn:
        conn.executescript(
            """
            CREATE TABLE hdf5_files (
                path TEXT PRIMARY KEY,
                group_id TEXT NOT NULL,
                subband_code INTEGER NOT NULL,
                timestamp_iso TEXT NOT NULL,
                ra_deg REAL,
                dec_deg REAL,
                file_size_bytes INTEGER
            );
            CREATE TABLE group_time_ranges (
                group_id TEXT PRIMARY KEY,
                file_count INTEGER NOT NULL,
                start_time_iso TEXT NOT NULL,
                end_time_iso TEXT NOT NULL,
                integration_count INTEGER NOT NULL,
                ra_deg REAL,
                dec_deg REAL
            );
            """
        )
        conn.execute(
            "INSERT INTO group_time_ranges VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                group_id,
                subband_count,
                "2026-01-25T04:53:42",
                "2026-01-25T04:58:42",
                EXPECTED_INTEGRATIONS,
                80.3,
                16.1,
            ),
        )
        for sb in range(subband_count):
            hdf5_path = hdf5_root / f"{group_id}_sb{sb:02d}.hdf5"
            hdf5_path.write_bytes(b"\x89HDF\r\n\x1a\n")
            conn.execute(
                "INSERT INTO hdf5_files VALUES (?, ?, ?, ?, ?, ?, ?)",
                (str(hdf5_path), group_id, sb, group_id, 80.3, 16.1, 10),
            )


def _make_live_shape_pipeline_db(path: Path, hdf5_root: Path) -> None:
    hdf5_root.mkdir()
    group_id = "2026-01-25T04:56:12"
    with sqlite3.connect(path) as conn:
        conn.executescript(
            """
            CREATE TABLE hdf5_files (
                path TEXT PRIMARY KEY,
                filename TEXT,
                group_id TEXT NOT NULL,
                subband_code TEXT,
                subband_num INTEGER NOT NULL,
                timestamp_iso TEXT NOT NULL,
                ra_deg REAL,
                dec_deg REAL
            );
            CREATE TABLE group_time_ranges (
                group_id TEXT PRIMARY KEY,
                start_iso TEXT NOT NULL,
                end_iso TEXT NOT NULL,
                file_count INTEGER NOT NULL,
                dec_deg REAL,
                complete BOOLEAN
            );
            """
        )
        conn.execute(
            "INSERT INTO group_time_ranges VALUES (?, ?, ?, ?, ?, ?)",
            (
                group_id,
                "2026-01-25T04:53:42",
                "2026-01-25T04:58:42",
                EXPECTED_SUBBANDS,
                16.1,
                1,
            ),
        )
        for sb in range(EXPECTED_SUBBANDS):
            hdf5_path = hdf5_root / f"{group_id}_sb{sb:02d}.hdf5"
            h5py = pytest.importorskip("h5py")

            with h5py.File(hdf5_path, "w") as handle:
                header = handle.create_group("Header")
                header.create_dataset("Ntimes", data=EXPECTED_INTEGRATIONS)
                times = Time(
                    [
                        "2026-01-25T04:53:42",
                        "2026-01-25T04:58:42",
                    ],
                    format="isot",
                    scale="utc",
                ).jd
                header.create_dataset(
                    "time_array",
                    data=np.linspace(times[0], times[1], EXPECTED_INTEGRATIONS),
                )
            conn.execute(
                "INSERT INTO hdf5_files VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    str(hdf5_path),
                    hdf5_path.name,
                    group_id,
                    f"sb{sb:02d}",
                    sb,
                    group_id,
                    80.3,
                    16.1,
                ),
            )


def test_validate_vla_calibrator_db_records_current_config_provenance(tmp_path):
    vla_db = tmp_path / "vla.sqlite3"
    _make_vla_db(vla_db)

    result = validate_vla_calibrator_db(vla_db)

    assert result.ok is True
    assert result.path == vla_db
    assert result.config_owner == "dsa110_continuum.config.PathConfig"
    assert result.calibrator_count == 2
    assert result.lband_flux_count == 2
    assert result.reject_reasons == []


def test_discovery_prefers_primary_calibrator_with_complete_centered_window(tmp_path):
    vla_db = tmp_path / "vla.sqlite3"
    pipeline_db = tmp_path / "pipeline.sqlite3"
    _make_vla_db(vla_db)
    _make_pipeline_db(pipeline_db, tmp_path / "incoming")

    result = discover_calibrator_candidates(
        DiscoveryConfig(
            pipeline_db=pipeline_db,
            vla_calibrator_db=vla_db,
            fwhm_deg=3.5,
            indexed_dates=["2026-01-25"],
        )
    )

    assert result.selected is not None
    assert result.selected.calibrator_name == "3C138"
    assert result.selected.selection_pool == "primary"
    assert result.selected.group_id == "2026-01-25T04:56:12"
    assert result.selected.reject_reasons == []
    assert result.selected.dec_offset_deg <= result.selected.fwhm_deg / 2
    assert result.selected.subband_count == EXPECTED_SUBBANDS
    assert result.selected.integration_count == EXPECTED_INTEGRATIONS


def test_discovery_rejects_incomplete_subband_group_before_selection(tmp_path):
    vla_db = tmp_path / "vla.sqlite3"
    pipeline_db = tmp_path / "pipeline.sqlite3"
    _make_vla_db(vla_db)
    _make_pipeline_db(pipeline_db, tmp_path / "incoming", complete=False)

    result = discover_calibrator_candidates(
        DiscoveryConfig(
            pipeline_db=pipeline_db,
            vla_calibrator_db=vla_db,
            fwhm_deg=3.5,
            indexed_dates=["2026-01-25"],
        )
    )

    assert result.selected is None
    assert any("INCOMPLETE_SUBBANDS" in c.reject_reasons for c in result.candidates)


def test_discovery_supports_live_pipeline_index_column_names(tmp_path):
    vla_db = tmp_path / "vla.sqlite3"
    pipeline_db = tmp_path / "pipeline.sqlite3"
    _make_vla_db(vla_db)
    _make_live_shape_pipeline_db(pipeline_db, tmp_path / "incoming")

    result = discover_calibrator_candidates(
        DiscoveryConfig(
            pipeline_db=pipeline_db,
            vla_calibrator_db=vla_db,
            fwhm_deg=3.5,
            indexed_dates=["2026-01-25"],
        )
    )

    assert result.selected is not None
    assert result.selected.group_id == "2026-01-25T04:56:12"
    assert result.selected.integration_count == EXPECTED_INTEGRATIONS


def test_assess_pair_reuses_transit_calculation_for_same_calibrator_date(
    monkeypatch, tmp_path: Path
):

    calls = 0

    def fake_next_transit_time(ra_deg: float, start_time_mjd: float):
        nonlocal calls
        calls += 1
        return Time("2026-01-25T04:56:12", format="isot", scale="utc")

    def files_for(group_id: str) -> list[dict[str, str]]:
        files = []
        for sb in range(EXPECTED_SUBBANDS):
            path = tmp_path / f"{group_id}_sb{sb:02d}.hdf5"
            path.write_bytes(b"\x89HDF\r\n\x1a\n")
            files.append({"path": str(path), "subband_code": f"sb{sb:02d}"})
        return files

    monkeypatch.setattr(smoke, "next_transit_time", fake_next_transit_time)
    _cached_transit_time.cache_clear()
    calibrator = Calibrator(
        name="3C138",
        ra_deg=80.2912,
        dec_deg=16.1,
        flux_jy=8.3,
        position_code="A",
        quality_codes="",
        selection_pool="primary",
    )
    groups = [
        {
            "group_id": "2026-01-25T04:56:12",
            "group_dec_deg": 16.1,
            "start_time_iso": "2026-01-25T04:53:42",
            "end_time_iso": "2026-01-25T04:58:42",
            "file_count": EXPECTED_SUBBANDS,
            "integration_count": EXPECTED_INTEGRATIONS,
        },
        {
            "group_id": "2026-01-25T05:01:12",
            "group_dec_deg": 16.1,
            "start_time_iso": "2026-01-25T04:58:42",
            "end_time_iso": "2026-01-25T05:03:42",
            "file_count": EXPECTED_SUBBANDS,
            "integration_count": EXPECTED_INTEGRATIONS,
        },
    ]

    for group in groups:
        _assess_pair(calibrator, group, files_for(group["group_id"]), 3.5)

    assert calls == 1
    _cached_transit_time.cache_clear()


def test_manual_model_data_preserves_correlation_channel_axis_order(monkeypatch):
    written_chunks: list[np.ndarray] = []

    class FakeTable:
        def __init__(self, table_name: str, readonly: bool = True) -> None:
            self.table_name = table_name
            self.readonly = readonly

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def colnames(self) -> list[str]:
            if self.table_name.endswith("::FIELD"):
                return ["PHASE_DIR", "REFERENCE_DIR"]
            return ["DATA", "MODEL_DATA", "CORRECTED_DATA"]

        def nrows(self) -> int:
            return 3

        def getcol(self, column: str, startrow: int = 0, nrow: int = -1):
            del startrow, nrow
            if self.table_name.endswith("::FIELD") and column == "PHASE_DIR":
                return np.array([[[0.0, 0.0]]])
            if self.table_name.endswith("::SPECTRAL_WINDOW") and column == "CHAN_FREQ":
                return np.array([np.linspace(1.3e9, 1.31e9, 48)])
            if self.table_name.endswith("::DATA_DESCRIPTION") and column == "SPECTRAL_WINDOW_ID":
                return np.array([0])
            if column == "UVW":
                return np.zeros((3, 3))
            if column == "DATA_DESC_ID":
                return np.zeros(3, dtype=int)
            if column == "FIELD_ID":
                return np.zeros(3, dtype=int)
            raise AssertionError(f"unexpected getcol {self.table_name} {column}")

        def getcell(self, column: str, row: int):
            assert column == "DATA"
            assert row == 0
            return np.zeros((2, 48), dtype=np.complex64)

        def putcol(self, column: str, value, startrow: int = 0, nrow: int = -1) -> None:
            del startrow, nrow
            if column == "MODEL_DATA":
                written_chunks.append(np.asarray(value))

        def flush(self) -> None:
            return None

    def fake_table(table_name: str, readonly: bool = True, **kwargs):
        del kwargs
        return FakeTable(table_name, readonly=readonly)

    monkeypatch.setattr(calibration_model, "_ensure_imaging_columns", lambda ms_path: None)
    monkeypatch.setattr(calibration_model, "_initialize_corrected_from_data", lambda ms_path: None)
    monkeypatch.setattr(calibration_model, "get_ms_metadata", None)
    monkeypatch.setattr(calibration_model.tb, "table", fake_table)

    calibration_model._calculate_manual_model_data(
        "fake.ms",
        ra_deg=0.0,
        dec_deg=0.0,
        flux_jy=2.5,
        initialize_corrected=True,
    )

    assert written_chunks
    assert written_chunks[0].shape == (3, 2, 48)
    assert np.allclose(written_chunks[0], 2.5 + 0j)


def test_create_immutable_run_dir_refuses_existing_directory(tmp_path):
    config = SmokeRunConfig(
        calibrator="3C48",
        group_id="2026-02-22T23:17:51",
        evidence_root=tmp_path,
        pipeline_db=tmp_path / "pipeline.sqlite3",
        vla_calibrator_db=tmp_path / "vla.sqlite3",
    )

    run_dir = create_immutable_run_dir(config, run_id="fixed_run")

    assert run_dir == tmp_path / "fixed_run"
    with pytest.raises(FileExistsError):
        create_immutable_run_dir(config, run_id="fixed_run")


def test_create_work_run_dir_mirrors_run_id_and_refuses_overwrite(tmp_path: Path):
    config = SmokeRunConfig(
        calibrator="3C48",
        group_id="2026-02-22T23:17:51",
        evidence_root=tmp_path / "evidence",
        pipeline_db=tmp_path / "pipeline.sqlite3",
        vla_calibrator_db=tmp_path / "vla.sqlite3",
        work_root=tmp_path / "work",
        run_id="fixed-run",
    )

    run_dir = create_immutable_run_dir(config)
    work_dir = create_work_run_dir(config, run_dir.name)

    assert work_dir == tmp_path / "work" / "fixed-run"
    assert (work_dir / "conversion").is_dir()
    assert (work_dir / "ms").is_dir()
    assert (work_dir / "logs").is_dir()

    with pytest.raises(FileExistsError):
        create_work_run_dir(config, run_dir.name)


def test_conversion_stage_uses_fast_work_root_and_records_actual_source_ms(
    monkeypatch, tmp_path: Path
):

    calls = {}

    def fake_convert(**kwargs):
        calls.update(kwargs)
        staged_ms = (
            tmp_path
            / "work"
            / "run"
            / "conversion"
            / "dsa110-contimg"
            / "2026-02-22T23:17:51.staged.ms"
        )
        staged_ms.mkdir(parents=True)
        return {
            "converted": ["2026-02-22T23:17:51"],
            "skipped": [],
            "failed": [],
            "converted_paths": {"2026-02-22T23:17:51": str(staged_ms)},
        }

    monkeypatch.setattr(smoke, "convert_subband_groups_to_ms", fake_convert, raising=False)
    monkeypatch.setattr(
        smoke,
        "_load_conversion_api",
        lambda: pytest.fail("conversion API should not load when converter is monkeypatched"),
    )

    config = SmokeRunConfig(
        calibrator="3C48",
        group_id="2026-02-22T23:17:51",
        evidence_root=tmp_path / "evidence",
        pipeline_db=tmp_path / "pipeline.sqlite3",
        vla_calibrator_db=tmp_path / "vla.sqlite3",
        work_root=tmp_path / "work",
    )
    run_dir = tmp_path / "evidence" / "run"
    work_dir = tmp_path / "work" / "run"
    work_dir.mkdir(parents=True)

    result = smoke._conversion_stage(config, run_dir, work_dir)

    assert calls["stage_to_tmpfs"] is True
    assert calls["defer_final_copy"] is True
    assert calls["tmpfs_path"] == str(work_dir / "conversion")
    assert calls["scratch_dir"] == str(work_dir / "scratch")
    assert result["source_ms"].endswith("2026-02-22T23:17:51.staged.ms")


def test_manifest_records_earliest_failed_stage(tmp_path):
    manifest = SmokeRunManifest.new(
        run_id="run1",
        run_dir=tmp_path / "run1",
        config=SmokeRunConfig(
            calibrator="3C48",
            group_id="2026-02-22T23:17:51",
            evidence_root=tmp_path,
            pipeline_db=tmp_path / "pipeline.sqlite3",
            vla_calibrator_db=tmp_path / "vla.sqlite3",
        ),
    )

    manifest.record_stage(StageResult(name="conversion", status="SUCCEEDED"))
    manifest.record_stage(
        StageResult(name="fresh_calibration", status="FAILED", reason="no .b table")
    )
    manifest.record_stage(StageResult(name="imaging", status="FAILED", reason="not reached"))

    assert manifest.status == "FAILED"
    assert manifest.failed_stage == "fresh_calibration"


def test_run_stage_records_failure_without_marking_success(tmp_path):
    config = SmokeRunConfig(
        calibrator="3C48",
        group_id="2026-02-22T23:17:51",
        evidence_root=tmp_path,
        pipeline_db=tmp_path / "pipeline.sqlite3",
        vla_calibrator_db=tmp_path / "vla.sqlite3",
    )
    run_dir = tmp_path / "run"
    manifest = SmokeRunManifest.new("run", run_dir, config)

    def fail() -> None:
        raise RuntimeError("conversion stalled")

    with pytest.raises(RuntimeError, match="conversion stalled"):
        _run_stage(manifest, "conversion", fail)

    payload = json.loads((run_dir / "manifest" / "run_status.json").read_text())
    assert payload["status"] == "FAILED"
    assert payload["failed_stage"] == "conversion"
    assert payload["completed_at"] is not None
    assert payload["stages"][0]["status"] == "FAILED"
    assert "RuntimeError: conversion stalled" in payload["stages"][0]["reason"]


def test_run_stage_records_keyboard_interrupt_as_failed_stage(tmp_path):
    config = SmokeRunConfig(
        calibrator="3C48",
        group_id="2026-02-22T23:17:51",
        evidence_root=tmp_path,
        pipeline_db=tmp_path / "pipeline.sqlite3",
        vla_calibrator_db=tmp_path / "vla.sqlite3",
    )
    run_dir = tmp_path / "run"
    manifest = SmokeRunManifest.new("run", run_dir, config)

    def interrupt() -> None:
        raise KeyboardInterrupt()

    with pytest.raises(KeyboardInterrupt):
        _run_stage(manifest, "conversion", interrupt)

    payload = json.loads((run_dir / "manifest" / "run_status.json").read_text())
    assert payload["status"] == "FAILED"
    assert payload["failed_stage"] == "conversion"
    assert payload["completed_at"] is not None
    assert payload["stages"][0]["reason"] == "KeyboardInterrupt"


def test_stage_log_sets_run_scoped_casa_log(monkeypatch, tmp_path: Path):
    from dsa110_continuum.evidence.hdf5_calibrator_tile_smoke import _stage_log

    monkeypatch.setenv("CASALOGFILE", "/data/dsa110-contimg/state/logs/casa.log")
    log_path = tmp_path / "logs" / "01_convert.log"

    with _stage_log(log_path):
        assert Path(os.environ["CASALOGFILE"]).parent == tmp_path / "logs"
        print("stdout marker")

    assert "stdout marker" in log_path.read_text()
    assert os.environ["CASALOGFILE"] == "/data/dsa110-contimg/state/logs/casa.log"


def test_validate_fresh_cal_tables_rejects_outside_symlink_dummy_and_missing(tmp_path):
    run_dir = tmp_path / "run"
    cal_dir = run_dir / "calibration"
    cal_dir.mkdir(parents=True)
    good_bp = cal_dir / "3c48.b"
    good_g = cal_dir / "3c48.g"
    good_2g = cal_dir / "3c48.2g"
    good_bp.mkdir()
    good_g.mkdir()
    good_2g.mkdir()

    assert validate_fresh_cal_tables([good_bp, good_g], run_dir) == (good_bp, [good_g])
    assert validate_fresh_cal_tables([good_bp, good_g, good_2g], run_dir) == (
        good_bp,
        [good_g, good_2g],
    )

    outside = tmp_path / "outside.g"
    outside.mkdir()
    with pytest.raises(ValueError, match="outside evidence run"):
        validate_fresh_cal_tables([good_bp, outside], run_dir)

    dummy = cal_dir / "3c48.dummy.G"
    dummy.mkdir()
    with pytest.raises(ValueError, match="dummy"):
        validate_fresh_cal_tables([good_bp, dummy], run_dir)

    link = cal_dir / "linked.g"
    link.symlink_to(good_g, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        validate_fresh_cal_tables([good_bp, link], run_dir)

    with pytest.raises(ValueError, match="does not exist"):
        validate_fresh_cal_tables([good_bp, cal_dir / "missing.g"], run_dir)
