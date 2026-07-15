import os
from datetime import datetime
from pathlib import Path

import dsa110_continuum.observability.dagster_defs as dagster_defs
import pytest
from dsa110_continuum.observability.dagster_defs import ASSET_NAMES, _readiness_markdown, defs
from dsa110_continuum.observability.hour_state import HourStateConfig, collect_hour_state
from dsa110_continuum.observability.mosaic_preview import (
    mosaic_preview_markdown,
    qa_thumb_url,
    resolve_thumb_source,
    sync_mosaic_thumb,
)


def _tmp_config(tmp_path: Path, **overrides) -> HourStateConfig:
    kwargs = {
        "stage": tmp_path / "stage",
        "products": tmp_path / "products",
        "incoming": tmp_path / "incoming",
        "campaign_outputs": tmp_path / "campaign",
        "disk_roots": (tmp_path,),
    }
    kwargs.update(overrides)
    return HourStateConfig(**kwargs)


def _no_processes(monkeypatch):
    monkeypatch.setattr("dsa110_continuum.observability.hour_state._process_status", lambda: [])


def test_collect_hour_state_reports_external_products(tmp_path: Path, monkeypatch):
    stage = tmp_path / "stage"
    ms_dir = stage / "ms"
    image_dir = stage / "images" / "mosaic_2026-07-13"
    incoming = tmp_path / "incoming"
    products = tmp_path / "products"
    campaign = tmp_path / "campaign"
    for directory in (ms_dir, image_dir, incoming, products / "2026-07-13", campaign):
        directory.mkdir(parents=True, exist_ok=True)

    (ms_dir / "2026-07-13T11:01:00.ms").mkdir()
    (ms_dir / "2026-07-13T11:01:00.b").write_text("bandpass")
    (ms_dir / "2026-07-13T11:01:00.g").write_text("gain")
    (image_dir / "2026-07-13T11:01:00-image.fits").write_text("tile")
    (image_dir / "2026-07-13T1100_mosaic.fits").write_text("mosaic")
    (incoming / "2026-07-13T11:01:00_sb00.hdf5").write_text("hdf5")
    (campaign / "batch_run_h11.log").write_text("campaign running\n")
    (campaign / "batch_attempt5.pid").write_text("999999\n")
    monkeypatch.setattr(
        "dsa110_continuum.observability.hour_state._process_status",
        lambda: [{"pid": 42, "command": "scripts/batch_pipeline.py"}],
    )

    state = collect_hour_state(
        HourStateConfig(
            stage=stage,
            products=products,
            incoming=incoming,
            campaign_outputs=campaign,
            disk_roots=(tmp_path,),
        )
    )

    assert state["measurement_sets"]["count"] == 1
    assert state["calibration"]["bandpass_count"] == 1
    assert state["calibration"]["gain_count"] == 1
    assert state["tiles"]["count"] == 1
    assert state["mosaic"]["path"].endswith("2026-07-13T1100_mosaic.fits")
    assert state["incoming"]["count"] == 1
    assert state["campaign"]["log_tail"] == ["campaign running"]
    assert state["campaign"]["state"] == "running"
    assert state["summary"]["campaign_state"] == "running"
    assert state["summary"]["campaign_process_visible"] is True


def test_collect_hour_state_distinguishes_absent_and_finished_campaigns(
    tmp_path: Path, monkeypatch
):
    campaign = tmp_path / "campaign"
    campaign.mkdir()
    monkeypatch.setattr("dsa110_continuum.observability.hour_state._process_status", lambda: [])
    config = HourStateConfig(
        stage=tmp_path / "stage",
        products=tmp_path / "products",
        incoming=tmp_path / "incoming",
        campaign_outputs=campaign,
        disk_roots=(tmp_path,),
    )

    assert collect_hour_state(config)["campaign"]["state"] == "absent"

    (campaign / "batch_run_h11.log").write_text("complete\n")

    assert collect_hour_state(config)["campaign"]["state"] == "finished"


def test_dagster_definitions_expose_six_observability_assets():
    asset_graph = defs.resolve_asset_graph()
    asset_keys = {key.to_user_string() for key in asset_graph.get_all_asset_keys()}

    assert asset_keys == set(ASSET_NAMES)
    assert all(asset_graph.get(key).description for key in asset_graph.get_all_asset_keys())
    assert defs.resolve_sensor_def("refresh_hour_11_observability_sensor") is not None


def test_readiness_markdown_is_an_operator_checklist():
    state = {
        "date": "2026-07-13",
        "hour": 11,
        "summary": {
            "campaign_state": "finished",
            "measurement_sets_present": True,
            "calibration_present": True,
            "tiles_present": True,
            "mosaic_present": False,
            "incoming_hdf5_present": True,
            "campaign_process_visible": False,
            "campaign_pid_visible": False,
        },
    }

    checklist = _readiness_markdown(state)

    assert "2026-07-13 UTC hour 11" in checklist
    assert "Campaign state:** `finished`" in checklist
    assert "| hourly-epoch mosaic | not visible |" in checklist
    assert "does not launch or retry pipeline work" in checklist


def test_qa_thumb_url_uses_epoch_token():
    assert qa_thumb_url("2026-07-13", 11) == (
        "http://127.0.0.1:8767/artifacts/mosaic/2026-07-13/T1100/thumb.png"
    )


def test_sync_mosaic_thumb_copies_png_into_static_root(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(
        "dsa110_continuum.observability.mosaic_preview.webapp_build_dir", lambda: None
    )
    campaign = tmp_path / "campaign"
    preview_dir = campaign / "previews"
    preview_dir.mkdir(parents=True)
    source = preview_dir / "2026-07-13T1100_mosaic_qa_thumb.png"
    source.write_bytes(b"\x89PNG\r\n\x1a\n" + b"fake-thumb")
    static_root = tmp_path / "dagster-static"
    config = HourStateConfig(
        stage=tmp_path / "stage",
        products=tmp_path / "products",
        incoming=tmp_path / "incoming",
        campaign_outputs=campaign,
        disk_roots=(tmp_path,),
    )

    assert resolve_thumb_source(config) == source
    preview = sync_mosaic_thumb(
        config,
        fits_path="/stage/example.fits",
        static_root=static_root,
    )

    assert preview["synced"] is True
    assert preview["epoch"] == "T1100"
    dest = static_root / "dsa110" / "2026-07-13_T1100_mosaic_thumb.png"
    assert dest.is_file()
    assert (static_root / "dsa110" / "hour11_mosaic.html").is_file()
    md = mosaic_preview_markdown(preview, mosaic_path="/stage/example.fits")
    assert "![mosaic thumbnail]" in md
    assert preview["dagster_page_url"] in md


class TestHourStateConfigValidation:
    """Correctness criterion: input-domain boundary behavior (fail-loud).

    Valid inputs are exactly the UTC hours 0..23 inclusive and real Gregorian
    calendar dates in YYYY-MM-DD form. Anything outside must raise ValueError
    at construction; accepting it would silently produce an all-absent snapshot
    for a nonsense epoch. Basis: UTC hour definition and calendar arithmetic
    (closed-form domain), not a pinned output.
    """

    def test_hour_boundaries_accepted(self, tmp_path):
        assert _tmp_config(tmp_path, hour=0).hour == 0
        assert _tmp_config(tmp_path, hour=23).hour == 23

    @pytest.mark.parametrize("hour", [-1, 24, 100])
    def test_out_of_range_hour_rejected(self, tmp_path, hour):
        with pytest.raises(ValueError):
            _tmp_config(tmp_path, hour=hour)

    @pytest.mark.parametrize("date", ["2026-13-01", "2026-02-30", "2026-00-10"])
    def test_non_calendar_date_rejected(self, tmp_path, date):
        with pytest.raises(ValueError):
            _tmp_config(tmp_path, date=date)

    @pytest.mark.parametrize("date", ["2026-7-13", "20260713", "2026-07-13T11"])
    def test_malformed_date_rejected(self, tmp_path, date):
        with pytest.raises(ValueError):
            _tmp_config(tmp_path, date=date)


class TestUtcHourBinSelection:
    """Correctness criterion: UTC-hour bin definition.

    An artifact belongs to hour H iff its filename timestamp falls in
    [H:00:00, H+1:00:00) — equivalently, the HH field of the
    YYYY-MM-DDTHH:MM:SS stem equals H, zero-padded. Basis: the stage/incoming
    filename convention shared with scripts/qa_server.py; verified with
    boundary timestamps on both bin edges (analytic bin edges, not pinned
    counts of an arbitrary fixture).
    """

    def test_hour_bin_edges_include_hour_exclude_neighbours(self, tmp_path, monkeypatch):
        _no_processes(monkeypatch)
        ms_dir = tmp_path / "stage" / "ms"
        ms_dir.mkdir(parents=True)
        for stamp in (
            "2026-07-13T10:59:59",
            "2026-07-13T11:00:00",
            "2026-07-13T11:59:59",
            "2026-07-13T12:00:00",
        ):
            (ms_dir / f"{stamp}.ms").mkdir()

        state = collect_hour_state(_tmp_config(tmp_path, date="2026-07-13", hour=11))

        assert state["measurement_sets"]["count"] == 2
        names = [Path(p).name for p in state["measurement_sets"]["paths"]]
        assert names == ["2026-07-13T11:00:00.ms", "2026-07-13T11:59:59.ms"]

    def test_single_digit_hour_is_zero_padded(self, tmp_path, monkeypatch):
        _no_processes(monkeypatch)
        ms_dir = tmp_path / "stage" / "ms"
        ms_dir.mkdir(parents=True)
        (ms_dir / "2026-07-13T05:30:00.ms").mkdir()

        state = collect_hour_state(_tmp_config(tmp_path, date="2026-07-13", hour=5))

        assert state["measurement_sets"]["count"] == 1

    def test_campaign_log_glob_accepts_padded_and_unpadded_hour(self, tmp_path, monkeypatch):
        _no_processes(monkeypatch)
        campaign = tmp_path / "campaign"
        campaign.mkdir()
        padded = campaign / "batch_run_h05_a.log"
        unpadded = campaign / "batch_run_h5_b.log"
        padded.write_text("old\n")
        unpadded.write_text("new\n")
        os.utime(padded, (1_000, 1_000))
        os.utime(unpadded, (2_000, 2_000))

        state = collect_hour_state(_tmp_config(tmp_path, date="2026-07-13", hour=5))

        assert state["campaign"]["log"]["path"].endswith("batch_run_h5_b.log")

    def test_campaign_log_glob_rejects_other_hours_sharing_digit_prefix(
        self, tmp_path, monkeypatch
    ):
        _no_processes(monkeypatch)
        campaign = tmp_path / "campaign"
        campaign.mkdir()
        own = campaign / "batch_run_h1_a.log"
        other = campaign / "batch_run_h11_b.log"
        own.write_text("own\n")
        other.write_text("other\n")
        os.utime(own, (1_000, 1_000))
        os.utime(other, (2_000, 2_000))

        state = collect_hour_state(_tmp_config(tmp_path, date="2026-07-13", hour=1))

        assert state["campaign"]["log"]["path"].endswith("batch_run_h1_a.log")


class TestCampaignStateMachine:
    """Correctness criterion: campaign-state precedence invariant.

    running > finished > absent: any live evidence (matching process OR a
    recorded PID visible in /proc) forces `running` regardless of artifacts;
    artifacts without live evidence give `finished`; neither gives `absent`.
    Basis: the state machine's documented precedence — adding stronger
    evidence must never demote the state (monotonicity), verified by
    constructing each evidence combination explicitly.
    """

    def test_visible_pid_hint_alone_means_running(self, tmp_path, monkeypatch):
        _no_processes(monkeypatch)
        campaign = tmp_path / "campaign"
        campaign.mkdir()
        (campaign / "watch.pid").write_text(f"{os.getpid()}\n")

        state = collect_hour_state(_tmp_config(tmp_path))

        assert state["summary"]["campaign_pid_visible"] is True
        assert state["summary"]["campaign_process_visible"] is False
        assert state["campaign"]["state"] == "running"

    def test_manifest_alone_means_finished(self, tmp_path, monkeypatch):
        _no_processes(monkeypatch)
        product_dir = tmp_path / "products" / "2026-07-13"
        product_dir.mkdir(parents=True)
        (product_dir / "2026-07-13_manifest.json").write_text("{}")

        state = collect_hour_state(_tmp_config(tmp_path, date="2026-07-13"))

        assert state["campaign"]["state"] == "finished"

    def test_running_takes_precedence_over_finished_artifacts(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "dsa110_continuum.observability.hour_state._process_status",
            lambda: [{"pid": 42, "elapsed_seconds": 1, "state": "S", "command": "wsclean"}],
        )
        campaign = tmp_path / "campaign"
        campaign.mkdir()
        (campaign / "batch_run_h11.log").write_text("done\n")

        state = collect_hour_state(_tmp_config(tmp_path, hour=11))

        assert state["campaign"]["state"] == "running"


class TestFileRecordTailOrdering:
    """Correctness criteria for the filesystem probe helpers.

    (a) `_file_record` timestamps: a file whose mtime is set to the Unix epoch
        must report modified_utc == 1970-01-01T00:00:00+00:00 (closed-form:
        definition of the Unix epoch in UTC) and size_bytes equal to the exact
        byte count written. Exact string/int equality is valid — no float
        arithmetic is involved for an integer mtime.
    (b) `_latest` ordering invariant: selection is by mtime, never by name —
        verified by making the lexicographically-first name the newest file.
    (c) `_tail` boundary: exactly the last 24 lines of a longer file, in file
        order, with trailing whitespace stripped (definition of a tail).
    """

    def test_file_record_epoch_mtime_and_exact_size(self, tmp_path, monkeypatch):
        _no_processes(monkeypatch)
        image_dir = tmp_path / "stage" / "images" / "mosaic_2026-07-13"
        image_dir.mkdir(parents=True)
        mosaic = image_dir / "2026-07-13T1100_mosaic.fits"
        mosaic.write_bytes(b"12345")
        os.utime(mosaic, (0, 0))

        state = collect_hour_state(_tmp_config(tmp_path, date="2026-07-13", hour=11))

        assert state["mosaic"]["size_bytes"] == 5
        assert state["mosaic"]["modified_utc"] == "1970-01-01T00:00:00+00:00"

    def test_latest_tile_selected_by_mtime_not_name(self, tmp_path, monkeypatch):
        _no_processes(monkeypatch)
        image_dir = tmp_path / "stage" / "images" / "mosaic_2026-07-13"
        image_dir.mkdir(parents=True)
        first_name = image_dir / "2026-07-13T11:01:00-image.fits"
        last_name = image_dir / "2026-07-13T11:02:00-image.fits"
        first_name.write_text("tile")
        last_name.write_text("tile")
        os.utime(first_name, (2_000, 2_000))
        os.utime(last_name, (1_000, 1_000))

        state = collect_hour_state(_tmp_config(tmp_path, date="2026-07-13", hour=11))

        assert state["tiles"]["latest"]["path"].endswith("T11:01:00-image.fits")

    def test_log_tail_is_last_24_lines_rstripped(self, tmp_path, monkeypatch):
        _no_processes(monkeypatch)
        campaign = tmp_path / "campaign"
        campaign.mkdir()
        lines = [f"line-{i:02d}   " for i in range(30)]
        (campaign / "batch_run_h11.log").write_text("\n".join(lines) + "\n")

        state = collect_hour_state(_tmp_config(tmp_path, hour=11))

        assert state["campaign"]["log_tail"] == [f"line-{i:02d}" for i in range(6, 30)]


class TestSnapshotIdempotence:
    """Correctness criterion: the probe is read-only and idempotent.

    Basis: module contract ("Read-only filesystem and process probes").
    Two consecutive calls on an unchanged filesystem must return identical
    state apart from `generated_at` (wall clock) and `disks` (usage of a live
    volume), and must not create, remove, or modify any probed file.
    """

    def test_collect_twice_identical_and_creates_nothing(self, tmp_path, monkeypatch):
        _no_processes(monkeypatch)
        campaign = tmp_path / "campaign"
        campaign.mkdir()
        (campaign / "batch_run_h11.log").write_text("done\n")
        ms_dir = tmp_path / "stage" / "ms"
        ms_dir.mkdir(parents=True)
        (ms_dir / "2026-07-13T11:01:00.ms").mkdir()
        config = _tmp_config(tmp_path, date="2026-07-13", hour=11)

        before = sorted(str(p) for p in tmp_path.rglob("*"))
        first = collect_hour_state(config)
        second = collect_hour_state(config)
        after = sorted(str(p) for p in tmp_path.rglob("*"))

        assert after == before
        for state in (first, second):
            state.pop("generated_at")
            state.pop("disks")
        assert first == second


class TestSensorRunKey:
    """Correctness criterion: sensor run-key granularity is one UTC minute.

    Dagster de-duplicates runs by run_key, so the key must be constant within
    a minute (seconds and microseconds truncated) to cap the refresh rate at
    the sensor's minimum_interval_seconds=60. Basis: Dagster run_key dedupe
    semantics; verified with a frozen clock at an arbitrary in-minute instant.
    """

    def test_run_key_truncates_to_minute(self, monkeypatch):
        class _Frozen(datetime):
            @classmethod
            def now(cls, tz=None):
                return datetime(2026, 7, 13, 11, 42, 37, 123456, tzinfo=tz)

        monkeypatch.setattr(dagster_defs, "datetime", _Frozen)

        request = dagster_defs.refresh_hour_11_observability_sensor()

        assert request.run_key == "2026-07-13T11:42:00+00:00"
