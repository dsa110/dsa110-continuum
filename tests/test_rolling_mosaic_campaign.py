import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

import run_rolling_mosaic_campaign as campaign
from run_rolling_mosaic_campaign import rolling_window


def test_rolling_window_uses_neighboring_available_hours():
    hours = [2, 3, 9]

    assert rolling_window(hours, 0) == (2, 4)
    assert rolling_window(hours, 1) == (2, 10)
    assert rolling_window(hours, 2) == (3, 10)


def test_working_set_group_ids_limits_conversion_to_core_and_overlap(tmp_path, monkeypatch):
    database = tmp_path / "pipeline.sqlite3"
    timestamps = [
        "2026-01-25T00:00:10",
        "2026-01-25T00:05:19",
        "2026-01-25T00:10:28",
        "2026-01-25T04:03:15",
        "2026-01-25T04:08:25",
        "2026-01-25T11:06:05",
        "2026-01-25T11:11:14",
        "2026-01-25T11:16:23",
    ]
    with campaign.sqlite3.connect(database) as connection:
        connection.execute(
            "CREATE TABLE hdf5_files (group_id TEXT, timestamp_iso TEXT, subband_num INTEGER)"
        )
        connection.executemany(
            "INSERT INTO hdf5_files VALUES (?, ?, ?)",
            [(timestamp, timestamp, subband) for timestamp in timestamps for subband in range(16)],
        )
    monkeypatch.setattr(campaign, "DB_PATH", database)

    selected = campaign._working_set_group_ids("2026-01-25", 4, 0, 12)

    assert selected == set(timestamps[1:7])


def test_batch_command_runs_photometry_before_pruning():
    command = campaign.batch_command("2026-01-25", 4, 0, 12)

    assert "--skip-photometry" not in command
    assert command[command.index("--photometry-workers") + 1] == "4"
    assert command[command.index("--photometry-chunk-size") + 1] == "0"


def test_refresh_lightcurves_stacks_forced_photometry(tmp_path, monkeypatch):
    forced_phot = tmp_path / "mosaics/2026-01-25/2026-01-25T0500_forced_phot.csv"
    forced_phot.parent.mkdir(parents=True)
    forced_phot.touch()

    def run(_command):
        stacked = tmp_path / "lightcurves/lightcurves.parquet"
        stacked.parent.mkdir(parents=True)
        stacked.touch()

    run_mock = MagicMock(side_effect=run)
    monkeypatch.setattr(campaign, "PRODUCTS_DIR", tmp_path)
    monkeypatch.setattr(campaign, "run", run_mock)

    assert campaign.refresh_lightcurves()
    run_mock.assert_called_once_with(
        [
            "/opt/miniforge/envs/casa6/bin/python",
            "scripts/stack_lightcurves.py",
            "--products-dir",
            str(tmp_path),
        ]
    )


def test_prune_hour_removes_approved_symlink_target(tmp_path, monkeypatch):
    stage_ms = tmp_path / "stage-ms"
    proc_ms = tmp_path / "proc-ms"
    images = tmp_path / "images"
    stage_ms.mkdir()
    proc_ms.mkdir()
    target = proc_ms / "2026-01-25T00:00:10.ms"
    target.mkdir()
    (target / "table.dat").touch()
    link = stage_ms / target.name
    link.symlink_to(target)
    tile = images / "mosaic_2026-01-25" / "2026-01-25T00:00:10-image-pb.fits"
    tile.parent.mkdir(parents=True)
    tile.touch()
    monkeypatch.setattr(campaign, "MS_DIR", stage_ms)
    monkeypatch.setattr(campaign, "PROC_MS_DIR", proc_ms)
    monkeypatch.setattr(campaign, "IMAGE_DIR", images)

    campaign.prune_hour("2026-01-25", 0)

    assert not link.exists()
    assert not target.exists()
    assert not tile.exists()


def test_prune_hour_retains_only_named_overlap_tiles(tmp_path, monkeypatch):
    stage_ms = tmp_path / "stage-ms"
    images = tmp_path / "images" / "mosaic_2026-01-25"
    stage_ms.mkdir()
    images.mkdir(parents=True)
    keep_stem = "2026-01-25T00:55:00"
    drop_stem = "2026-01-25T00:05:00"
    for stem in (keep_stem, drop_stem):
        (stage_ms / f"{stem}.ms").mkdir()
        (images / f"{stem}-image-pb.fits").touch()
    monkeypatch.setattr(campaign, "MS_DIR", stage_ms)
    monkeypatch.setattr(campaign, "PROC_MS_DIR", tmp_path / "proc-ms")
    monkeypatch.setattr(campaign, "IMAGE_DIR", tmp_path / "images")

    campaign.prune_hour("2026-01-25", 0, {keep_stem})

    assert (stage_ms / f"{keep_stem}.ms").exists()
    assert (images / f"{keep_stem}-image-pb.fits").exists()
    assert not (stage_ms / f"{drop_stem}.ms").exists()
    assert not (images / f"{drop_stem}-image-pb.fits").exists()


def test_strict_qa_requires_matching_pass_epoch(tmp_path, monkeypatch):
    monkeypatch.setattr(campaign, "CAMPAIGN_DIR", tmp_path)
    manifest = tmp_path / "2026-01-25T04_2026-01-25_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "epochs": [
                    {"hour": 4, "status": "ok", "qa_result": "FAIL"},
                    {"hour": 5, "status": "ok", "qa_result": "PASS"},
                ]
            }
        )
    )

    assert not campaign.strict_qa_passed("2026-01-25", 4)
    assert not campaign.strict_qa_passed("2026-01-25", 5)

    manifest.write_text(json.dumps({"epochs": [{"hour": 4, "status": "ok", "qa_result": "PASS"}]}))
    assert campaign.strict_qa_passed("2026-01-25", 4)


def test_preserved_metadata_requires_matching_pass_epoch(tmp_path, monkeypatch):
    monkeypatch.setattr(campaign, "CAMPAIGN_DIR", tmp_path)
    products = tmp_path / "products"
    monkeypatch.setattr(campaign, "PRODUCTS_MOSAIC_DIR", products)
    prefix = tmp_path / "2026-01-25T04_"
    preserved_manifest = Path(f"{prefix}2026-01-25_manifest.json")
    preserved_manifest.write_text(
        json.dumps({"epochs": [{"hour": 5, "status": "ok", "qa_result": "PASS"}]})
    )
    current_manifest = products / "2026-01-25/2026-01-25_manifest.json"
    current_manifest.parent.mkdir(parents=True)
    current_manifest.write_text(
        json.dumps({"epochs": [{"hour": 4, "status": "ok", "qa_result": "PASS"}]})
    )
    Path(f"{prefix}2026-01-25_run_summary.json").write_text(
        json.dumps(
            {
                "epochs": [
                    {
                        "label": "2026-01-25T0400",
                        "status": "ok",
                        "qa_result": "PASS",
                    }
                ]
            }
        )
    )
    report = Path(f"{prefix}run_report.md")
    report.write_text("| 04 | ok | PASS | 12 |\n")

    assert not campaign.preserved_run_metadata_complete("2026-01-25", 4)

    preserved_manifest.write_text(
        json.dumps({"epochs": [{"hour": 4, "status": "ok", "qa_result": "PASS"}]})
    )
    assert campaign.preserved_run_metadata_complete("2026-01-25", 4)


def test_failed_strict_qa_is_recorded_without_stopping_later_epochs(tmp_path, monkeypatch):
    status_path = tmp_path / "status.json"
    prune = MagicMock()
    preserve = MagicMock()
    accepted_products = MagicMock(return_value=(True, None))
    monkeypatch.setattr(campaign, "CAMPAIGN_DIR", tmp_path)
    monkeypatch.setattr(campaign, "STATUS_PATH", status_path)
    monkeypatch.setattr(campaign, "index_inventory", lambda: None)
    monkeypatch.setattr(campaign, "complete_hours", lambda: {"2026-01-25": [4, 5]})
    monkeypatch.setattr(campaign, "accepted_artifacts_complete", lambda *args: False)
    monkeypatch.setattr(
        campaign,
        "mosaic_is_valid",
        MagicMock(side_effect=[False, False, False, True]),
    )
    monkeypatch.setattr(campaign, "convert_window", lambda *args: None)
    monkeypatch.setattr(campaign, "promote_working_set_to_nvme", lambda *args: None)
    monkeypatch.setattr(campaign, "run", lambda command: None)
    monkeypatch.setattr(campaign, "preserve_run_metadata", preserve)
    monkeypatch.setattr(campaign, "accepted_products_ready", accepted_products)
    monkeypatch.setattr(campaign, "prune_hour", prune)

    campaign.run_campaign(plan_only=False)

    status = json.loads(status_path.read_text())
    assert status["state"] == "complete_with_failures"
    assert status["completed"] == ["2026-01-25T0500"]
    assert status["failed_epochs"] == [
        {
            "epoch": "2026-01-25T0400",
            "reason": "mosaic failed product integrity or strict QA",
        }
    ]
    assert accepted_products.call_count == 1
    prune.assert_called_once_with("2026-01-25", 5, set())


def test_nvme_promotion_replaces_approved_symlink_atomically(tmp_path, monkeypatch):
    stage_ms = tmp_path / "stage-ms"
    proc_ms = tmp_path / "proc-ms"
    stage_ms.mkdir()
    proc_ms.mkdir()
    target = proc_ms / "2026-01-25T04:00:00.ms"
    target.mkdir()
    (target / "table.dat").write_bytes(b"science")
    link = stage_ms / target.name
    link.symlink_to(target)
    monkeypatch.setattr(campaign, "MS_DIR", stage_ms)
    monkeypatch.setattr(campaign, "PROC_MS_DIR", proc_ms)
    monkeypatch.setattr(campaign, "NVME_RESERVE_BYTES", 0)
    monkeypatch.setattr(
        campaign,
        "_measurement_set_is_readable",
        lambda path: True,
    )

    campaign.promote_working_set_to_nvme("2026-01-25", 4, 4, 5)

    assert link.is_dir()
    assert not link.is_symlink()
    assert (link / "table.dat").read_bytes() == b"science"
    assert target.exists()


def test_nvme_promotion_refuses_to_cross_reserve(tmp_path, monkeypatch):
    stage_ms = tmp_path / "stage-ms"
    proc_ms = tmp_path / "proc-ms"
    stage_ms.mkdir()
    proc_ms.mkdir()
    target = proc_ms / "2026-01-25T04:00:00.ms"
    target.mkdir()
    link = stage_ms / target.name
    link.symlink_to(target)
    monkeypatch.setattr(campaign, "MS_DIR", stage_ms)
    monkeypatch.setattr(campaign, "PROC_MS_DIR", proc_ms)
    monkeypatch.setattr(campaign, "NVME_RESERVE_BYTES", 100)
    monkeypatch.setattr(campaign, "_allocated_bytes", lambda path: 1)
    monkeypatch.setattr(
        campaign.shutil,
        "disk_usage",
        lambda path: type("Usage", (), {"free": 100})(),
    )

    with pytest.raises(RuntimeError, match="NVMe working set needs"):
        campaign.promote_working_set_to_nvme("2026-01-25", 4, 4, 5)

    assert link.is_symlink()


def test_inactive_nvme_ms_is_atomically_demoted(tmp_path, monkeypatch):
    stage_ms = tmp_path / "stage-ms"
    proc_ms = tmp_path / "proc-ms"
    stage_ms.mkdir()
    proc_ms.mkdir()
    name = "2026-01-25T04:00:00.ms"
    stage_copy = stage_ms / name
    slow_copy = proc_ms / name
    stage_copy.mkdir()
    slow_copy.mkdir()
    (stage_copy / "table.dat").write_bytes(b"nvme")
    (slow_copy / "table.dat").write_bytes(b"slow")
    monkeypatch.setattr(campaign, "MS_DIR", stage_ms)
    monkeypatch.setattr(campaign, "PROC_MS_DIR", proc_ms)
    monkeypatch.setattr(
        campaign,
        "_measurement_set_is_readable",
        lambda path: True,
    )

    campaign.demote_inactive_nvme_ms("2026-01-25", set())

    assert stage_copy.is_symlink()
    assert stage_copy.resolve() == slow_copy
    assert (stage_copy / "table.dat").read_bytes() == b"slow"


def test_conversion_capacity_refuses_to_cross_reserve(tmp_path, monkeypatch):
    stage_ms = tmp_path / "stage-ms"
    proc_ms = tmp_path / "proc-ms"
    stage_ms.mkdir()
    proc_ms.mkdir()
    (proc_ms / "2026-01-25T03:00:00.ms").mkdir()
    monkeypatch.setattr(campaign, "MS_DIR", stage_ms)
    monkeypatch.setattr(campaign, "PROC_MS_DIR", proc_ms)
    monkeypatch.setattr(campaign, "NVME_RESERVE_BYTES", 100)
    monkeypatch.setattr(campaign, "_allocated_bytes", lambda path: 10)
    monkeypatch.setattr(
        campaign.shutil,
        "disk_usage",
        lambda path: type("Usage", (), {"free": 100})(),
    )

    with pytest.raises(RuntimeError, match="conversion working set estimates"):
        campaign.ensure_conversion_capacity({"2026-01-25T04:00:00"})
