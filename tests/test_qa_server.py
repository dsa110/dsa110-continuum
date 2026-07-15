import os
import subprocess
import types
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest
import scripts.qa_server as qa_server
from astropy.io import fits
from fastapi.testclient import TestClient
from scripts.qa_server import DashboardConfig, create_app, get_metrics


def test_dashboard_startup_and_mosaic_artifact_routes(tmp_path: Path):
    stage = tmp_path / "stage"
    products = tmp_path / "products"
    incoming = tmp_path / "incoming"
    mosaic_dir = stage / "images" / "mosaic_2026-01-25"
    mosaic_dir.mkdir(parents=True)
    (stage / "ms").mkdir()
    products.mkdir()
    incoming.mkdir()
    photometry_dir = products / "2026-01-25"
    photometry_dir.mkdir()
    (photometry_dir / "2026-01-25T0200_phot.csv").write_text("nvss_flux_jy,dsa_nvss_ratio\n0.1,\n")
    fits.writeto(
        mosaic_dir / "2026-01-25T0200_mosaic.fits",
        np.arange(64, dtype=np.float32).reshape(8, 8) / 1000,
    )
    config = DashboardConfig(
        stage=stage,
        products=products,
        incoming=incoming,
        thumb_dir=tmp_path / "thumbs",
        campaign_outputs=tmp_path / "campaign",
    )

    with TestClient(create_app(config)) as client:
        root = client.get("/")
        status = client.get("/artifacts/mosaic/2026-01-25/T0200/status")
        thumbnail = client.get("/artifacts/mosaic/2026-01-25/T0200/thumb.png")

    assert root.status_code == 200
    assert "DSA-110 Continuum Observatory" in root.text
    assert status.status_code == 200
    assert status.json()["fits"] is True
    assert status.json()["ratio"] is None
    assert status.json()["mosaic_path"].endswith("2026-01-25T0200_mosaic.fits")
    assert thumbnail.status_code == 200
    assert thumbnail.headers["content-type"] == "image/png"
    assert thumbnail.content.startswith(b"\x89PNG")


def _make_config(tmp_path: Path) -> DashboardConfig:
    return DashboardConfig(
        stage=tmp_path / "stage",
        products=tmp_path / "products",
        incoming=tmp_path / "incoming",
        thumb_dir=tmp_path / "thumbs",
        campaign_outputs=tmp_path / "campaign",
        campaign_date="2026-07-13",
        campaign_hour=11,
    )


def _write_mosaic(config: DashboardConfig, date: str, epoch: str, data: np.ndarray) -> Path:
    mosaic_dir = config.stage / "images" / f"mosaic_{date}"
    mosaic_dir.mkdir(parents=True, exist_ok=True)
    path = mosaic_dir / f"{date}{epoch}_mosaic.fits"
    fits.writeto(path, data, overwrite=True)
    return path


def _write_phot(config: DashboardConfig, date: str, epoch: str, text: str) -> Path:
    directory = config.products / date
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{date}{epoch}_phot.csv"
    path.write_text(text)
    return path


def _analytic_mosaic() -> np.ndarray:
    """Synthetic mosaic with hand-computable QA metrics.

    254 background pixels split evenly between -0.01 and +0.01 (mean 0,
    std exactly 0.01 -> RMS 10.0 mJy), one 5.0 Jy peak, one NaN.
    Dynamic range = 5.0 / 0.01 = 500.
    """
    flat = np.empty(256, dtype=np.float32)
    flat[:127] = -0.01
    flat[127:254] = 0.01
    flat[254] = 5.0
    flat[255] = np.nan
    return flat.reshape(16, 16)


class TestPathParameterSafety:
    """Criterion: user-supplied date/epoch (path or query params) are interpolated
    into filesystem paths (stage/images/mosaic_{date}/..., products/{date}/...),
    so any value outside the canonical YYYY-MM-DD / THHMM formats must be rejected
    with 4xx before any filesystem access, must never leak file content, and must
    never produce a 500. Basis: independent path-traversal invariant for any
    file-serving endpoint, not a pin of current behaviour."""

    def test_traversal_date_in_mosaic_status_is_rejected(self, tmp_path: Path):
        with TestClient(create_app(_make_config(tmp_path))) as client:
            for value in ("..", "....", "%2e%2e%2f%2e%2e%2fetc%2fpasswd", "2026-01-25x"):
                response = client.get(f"/artifacts/mosaic/{value}/T0200/status")
                assert 400 <= response.status_code < 500, value
                assert "root:" not in response.text

    def test_traversal_epoch_in_mosaic_routes_is_rejected(self, tmp_path: Path):
        with TestClient(create_app(_make_config(tmp_path))) as client:
            for value in ("..", "passwd", "T12345", "%2e%2e"):
                status = client.get(f"/artifacts/mosaic/2026-01-25/{value}/status")
                thumb = client.get(f"/artifacts/mosaic/2026-01-25/{value}/thumb.png")
                assert 400 <= status.status_code < 500, value
                assert 400 <= thumb.status_code < 500, value
                assert "root:" not in status.text

    def test_traversal_on_legacy_thumb_route_is_rejected(self, tmp_path: Path):
        with TestClient(create_app(_make_config(tmp_path))) as client:
            response = client.get("/thumb/%2e%2e/T0200.png")
            assert 400 <= response.status_code < 500
            response = client.get("/thumb/2026-01-25/%2e%2e.png")
            assert 400 <= response.status_code < 500

    def test_ops_status_rejects_malformed_date_query(self, tmp_path: Path):
        with TestClient(create_app(_make_config(tmp_path))) as client:
            for value in ("../../../etc/passwd", "2026-01-25%0A", "26-01-25", "...."):
                response = client.get(f"/api/status?date={value}")
                assert response.status_code == 400, value
                assert "root:" not in response.text

    def test_ops_status_rejects_out_of_range_hour(self, tmp_path: Path):
        with TestClient(create_app(_make_config(tmp_path))) as client:
            assert client.get("/api/status?hour=24").status_code == 400
            assert client.get("/api/status?hour=-1").status_code == 400
            assert client.get("/api/status?hour=abc").status_code == 422

    def test_wellformed_unknown_date_yields_empty_state_not_500(self, tmp_path: Path):
        with TestClient(create_app(_make_config(tmp_path))) as client:
            ops = client.get("/api/status?date=1999-01-01")
            metrics = client.get("/artifacts/mosaic/1999-01-01/T0000/status")
        assert ops.status_code == 200
        assert all(stage["state"] == "not_yet" for stage in ops.json()["stages"])
        assert metrics.status_code == 200
        assert metrics.json()["status"] == "missing"


class TestMosaicMetricsGroundTruth:
    """Criterion: /artifacts/mosaic/{date}/{epoch}/status reports peak/rms/dr that
    match analytic values hand-computed from a synthetic mosaic (background split
    evenly between -0.01 and +0.01 -> std exactly 0.01 -> 10.0 mJy RMS; peak 5.0 Jy;
    DR 500), and the DSA/NVSS ratio is the median over bright (>0.06 Jy) sources
    only. Basis: analytic construction of the input array, not a regression pin.
    Floats compared with tolerances (float32 storage)."""

    def test_metrics_match_analytic_values(self, tmp_path: Path):
        config = _make_config(tmp_path)
        _write_mosaic(config, "2026-01-25", "T0200", _analytic_mosaic())
        _write_phot(
            config,
            "2026-01-25",
            "T0200",
            "nvss_flux_jy,dsa_nvss_ratio\n0.1,1.0\n0.2,1.1\n0.5,0.9\n0.05,5.0\n",
        )
        with TestClient(create_app(config)) as client:
            payload = client.get("/artifacts/mosaic/2026-01-25/T0200/status").json()
        assert payload["fits"] is True
        assert payload["csv"] is True
        assert payload["peak"] == pytest.approx(5.0, rel=1e-5)
        assert payload["rms"] == pytest.approx(10.0, rel=1e-3)
        assert payload["dr"] == pytest.approx(500.0, rel=1e-3)
        assert payload["n_bright"] == 3
        assert payload["ratio"] == pytest.approx(1.0, rel=1e-9)
        assert payload["status"] == "pass"
        assert payload["mosaic_path"].endswith("2026-01-25T0200_mosaic.fits")

    def test_missing_mosaic_reports_missing(self, tmp_path: Path):
        with TestClient(create_app(_make_config(tmp_path))) as client:
            payload = client.get("/artifacts/mosaic/2026-01-25/T0200/status").json()
        assert payload["fits"] is False
        assert payload["status"] == "missing"
        assert payload["peak"] is None
        assert payload["mosaic_path"] is None

    def test_corrupt_fits_yields_controlled_nulls_not_500(self, tmp_path: Path):
        config = _make_config(tmp_path)
        mosaic_dir = config.stage / "images" / "mosaic_2026-01-25"
        mosaic_dir.mkdir(parents=True)
        (mosaic_dir / "2026-01-25T0200_mosaic.fits").write_bytes(b"this is not a FITS file")
        with TestClient(create_app(config)) as client:
            response = client.get("/artifacts/mosaic/2026-01-25/T0200/status")
        assert response.status_code == 200
        payload = response.json()
        assert payload["fits"] is True
        assert payload["peak"] is None
        assert payload["rms"] is None
        assert payload["dr"] is None
        assert payload["status"] == "no_phot"

    def test_all_nan_mosaic_yields_null_metrics_not_500(self, tmp_path: Path):
        config = _make_config(tmp_path)
        _write_mosaic(config, "2026-01-25", "T0200", np.full((8, 8), np.nan, dtype=np.float32))
        with TestClient(create_app(config)) as client:
            response = client.get("/artifacts/mosaic/2026-01-25/T0200/status")
        assert response.status_code == 200
        payload = response.json()
        assert payload["peak"] is None
        assert payload["rms"] is None
        assert payload["status"] == "no_phot"


class TestStatusTruthTable:
    """Criterion: QA verdict aggregation follows the documented truth table --
    'pass' iff 0.8 <= median bright-source DSA/NVSS ratio <= 1.2 (the dashboard's
    published target window, closed interval), 'fail' outside it, 'no_phot' when
    the ratio is unavailable or non-finite, and 'missing' dominates everything
    when no mosaic FITS exists. Basis: documented QA operating point (0.8-1.2)
    and the >0.06 Jy bright-source definition. The strict exclusion of sources
    at exactly 0.06 Jy is pinned current behaviour (UNVERIFIED as a science
    contract). Exercised at unit level via get_metrics."""

    @pytest.mark.parametrize(
        ("ratio_text", "expected_status"),
        [
            ("1.0", "pass"),
            ("0.8", "pass"),
            ("1.2", "pass"),
            ("0.79", "fail"),
            ("1.21", "fail"),
            ("0.5", "fail"),
            ("2.0", "fail"),
        ],
    )
    def test_ratio_window_verdicts(self, tmp_path: Path, ratio_text: str, expected_status: str):
        config = _make_config(tmp_path)
        _write_mosaic(config, "2026-01-25", "T0200", _analytic_mosaic())
        _write_phot(
            config, "2026-01-25", "T0200", f"nvss_flux_jy,dsa_nvss_ratio\n0.1,{ratio_text}\n"
        )
        metrics = get_metrics(config, "2026-01-25", "T0200")
        assert metrics["status"] == expected_status
        assert metrics["ratio"] == pytest.approx(float(ratio_text), rel=1e-12)

    def test_no_csv_is_no_phot(self, tmp_path: Path):
        config = _make_config(tmp_path)
        _write_mosaic(config, "2026-01-25", "T0200", _analytic_mosaic())
        metrics = get_metrics(config, "2026-01-25", "T0200")
        assert metrics["csv"] is False
        assert metrics["ratio"] is None
        assert metrics["status"] == "no_phot"

    def test_empty_csv_file_is_no_phot_not_error(self, tmp_path: Path):
        config = _make_config(tmp_path)
        _write_mosaic(config, "2026-01-25", "T0200", _analytic_mosaic())
        _write_phot(config, "2026-01-25", "T0200", "")
        metrics = get_metrics(config, "2026-01-25", "T0200")
        assert metrics["n_bright"] == 0
        assert metrics["status"] == "no_phot"

    def test_csv_missing_flux_column_is_no_phot(self, tmp_path: Path):
        config = _make_config(tmp_path)
        _write_mosaic(config, "2026-01-25", "T0200", _analytic_mosaic())
        _write_phot(config, "2026-01-25", "T0200", "other_col\n1.0\n2.0\n")
        metrics = get_metrics(config, "2026-01-25", "T0200")
        assert metrics["n_bright"] == 0
        assert metrics["ratio"] is None
        assert metrics["status"] == "no_phot"

    def test_csv_missing_ratio_column_counts_bright_but_no_phot(self, tmp_path: Path):
        config = _make_config(tmp_path)
        _write_mosaic(config, "2026-01-25", "T0200", _analytic_mosaic())
        _write_phot(config, "2026-01-25", "T0200", "nvss_flux_jy\n0.1\n0.2\n")
        metrics = get_metrics(config, "2026-01-25", "T0200")
        assert metrics["n_bright"] == 2
        assert metrics["ratio"] is None
        assert metrics["status"] == "no_phot"

    def test_non_finite_ratio_is_normalized_to_no_phot(self, tmp_path: Path):
        config = _make_config(tmp_path)
        _write_mosaic(config, "2026-01-25", "T0200", _analytic_mosaic())
        _write_phot(config, "2026-01-25", "T0200", "nvss_flux_jy,dsa_nvss_ratio\n0.1,inf\n")
        metrics = get_metrics(config, "2026-01-25", "T0200")
        assert metrics["ratio"] is None
        assert metrics["status"] == "no_phot"

    def test_faint_only_sources_are_no_phot(self, tmp_path: Path):
        config = _make_config(tmp_path)
        _write_mosaic(config, "2026-01-25", "T0200", _analytic_mosaic())
        _write_phot(
            config, "2026-01-25", "T0200", "nvss_flux_jy,dsa_nvss_ratio\n0.05,1.0\n0.06,1.0\n"
        )
        metrics = get_metrics(config, "2026-01-25", "T0200")
        assert metrics["n_bright"] == 0
        assert metrics["status"] == "no_phot"

    def test_missing_fits_dominates_valid_photometry(self, tmp_path: Path):
        config = _make_config(tmp_path)
        _write_phot(config, "2026-01-25", "T0200", "nvss_flux_jy,dsa_nvss_ratio\n0.1,1.0\n")
        metrics = get_metrics(config, "2026-01-25", "T0200")
        assert metrics["fits"] is False
        assert metrics["csv"] is True
        assert metrics["status"] == "missing"
        assert metrics["ratio"] is None


class TestThumbnailEndpoints:
    """Criterion: thumbnail routes return image/png for renderable mosaics on both
    the canonical and legacy URLs with identical content; absent, corrupt, or
    unrenderable (all-NaN) mosaics yield 404 -- never 500 and never a partial
    body. A regenerated mosaic (new mtime) must invalidate the cached PNG: a
    stale thumbnail is a silently wrong dashboard. Basis: HTTP contract plus the
    cache-key-includes-mtime freshness invariant."""

    def test_missing_mosaic_is_404_on_both_routes(self, tmp_path: Path):
        with TestClient(create_app(_make_config(tmp_path))) as client:
            assert client.get("/artifacts/mosaic/2026-01-25/T0200/thumb.png").status_code == 404
            assert client.get("/thumb/2026-01-25/T0200.png").status_code == 404

    def test_corrupt_fits_is_404_not_500(self, tmp_path: Path):
        config = _make_config(tmp_path)
        mosaic_dir = config.stage / "images" / "mosaic_2026-01-25"
        mosaic_dir.mkdir(parents=True)
        (mosaic_dir / "2026-01-25T0200_mosaic.fits").write_bytes(b"garbage")
        with TestClient(create_app(config)) as client:
            assert client.get("/artifacts/mosaic/2026-01-25/T0200/thumb.png").status_code == 404

    def test_all_nan_mosaic_is_404_not_500(self, tmp_path: Path):
        config = _make_config(tmp_path)
        _write_mosaic(config, "2026-01-25", "T0200", np.full((8, 8), np.nan, dtype=np.float32))
        with TestClient(create_app(config)) as client:
            assert client.get("/artifacts/mosaic/2026-01-25/T0200/thumb.png").status_code == 404

    def test_legacy_and_canonical_routes_serve_identical_png(self, tmp_path: Path):
        config = _make_config(tmp_path)
        _write_mosaic(config, "2026-01-25", "T0200", _analytic_mosaic())
        with TestClient(create_app(config)) as client:
            canonical = client.get("/artifacts/mosaic/2026-01-25/T0200/thumb.png")
            legacy = client.get("/thumb/2026-01-25/T0200.png")
        assert canonical.status_code == 200
        assert legacy.status_code == 200
        assert canonical.content.startswith(b"\x89PNG")
        assert canonical.content == legacy.content

    def test_thumbnail_regenerates_when_mosaic_changes(self, tmp_path: Path):
        config = _make_config(tmp_path)
        path = _write_mosaic(config, "2026-01-25", "T0200", _analytic_mosaic())
        with TestClient(create_app(config)) as client:
            first = client.get("/artifacts/mosaic/2026-01-25/T0200/thumb.png")
            changed = _analytic_mosaic()
            changed[changed == 5.0] = 20.0
            fits.writeto(path, changed, overwrite=True)
            stat = path.stat()
            os.utime(path, (stat.st_atime, stat.st_mtime + 100))
            second = client.get("/artifacts/mosaic/2026-01-25/T0200/thumb.png")
        assert first.status_code == 200
        assert second.status_code == 200
        assert second.content != first.content
        assert len(list(config.thumb_dir.glob("2026-01-25_T0200_*.png"))) == 1


class TestOpsStatusContract:
    """Criterion: /api/status reflects filesystem truth -- stage states and counts
    derive from actual artifact presence for the requested date/hour only; the
    calibration stage is 'ready' only when BOTH bandpass (.b) and gain (.g)
    tables exist (BP/G pair contract, docs/reference/calibration.md); missing
    roots produce an empty-state 200, never 500; file records carry real path
    and byte size; the log panel serves a suffix of the newest matching log
    (tail length 24 is the module default -- pinned, UNVERIFIED as contract).
    process_status is stubbed for determinism; it has its own parsing tests."""

    def test_empty_environment_yields_empty_state(self, tmp_path: Path, monkeypatch):
        monkeypatch.setattr(qa_server, "process_status", lambda: [])
        with TestClient(create_app(_make_config(tmp_path))) as client:
            response = client.get("/api/status")
        assert response.status_code == 200
        payload = response.json()
        assert payload["date"] == "2026-07-13"
        assert payload["hour"] == 11
        assert [stage["state"] for stage in payload["stages"]] == ["not_yet"] * 4
        assert [stage["count"] for stage in payload["stages"]] == [0] * 4
        assert payload["measurement_sets"] == {"count": 0, "paths": []}
        assert payload["mosaic"] is None
        assert payload["run_products"] == {"manifest": None, "summary": None, "report": None}
        assert payload["log"] == {"file": None, "tail": []}
        assert payload["incoming"]["hdf5_count"] == 0
        assert "disks" in payload

    def test_populated_environment_counts_only_requested_hour(self, tmp_path: Path, monkeypatch):
        monkeypatch.setattr(qa_server, "process_status", lambda: [])
        config = _make_config(tmp_path)
        ms_dir = config.stage / "ms"
        ms_dir.mkdir(parents=True)
        (ms_dir / "2026-07-13T11:02:03.ms").write_text("ms")
        (ms_dir / "2026-07-13T11:30:00.ms").write_text("ms")
        (ms_dir / "2026-07-13T12:00:00.ms").write_text("decoy other hour")
        (ms_dir / "2026-07-13T11:26:05_0~23.b").write_text("bp")
        (ms_dir / "2026-07-13T11:26:05_0~23.g").write_text("gain")
        image_dir = config.stage / "images" / "mosaic_2026-07-13"
        image_dir.mkdir(parents=True)
        (image_dir / "2026-07-13T11:05:00-image.fits").write_text("tile")
        (image_dir / "2026-07-13T12:05:00-image.fits").write_text("decoy other hour")
        (image_dir / "2026-07-13T1100_mosaic.fits").write_text("mosaic")
        config.incoming.mkdir(parents=True)
        (config.incoming / "2026-07-13T11:00:00_sb01.hdf5").write_text("h5")
        (config.incoming / "2026-07-13T11:00:00_sb02.hdf5").write_text("h5")
        (config.incoming / "2026-07-13T12:00:00_sb01.hdf5").write_text("decoy other hour")
        product_dir = config.products / "2026-07-13"
        product_dir.mkdir(parents=True)
        manifest_text = '{"pipeline_verdict": "DEGRADED"}'
        (product_dir / "2026-07-13_manifest.json").write_text(manifest_text)
        (product_dir / "2026-07-13_run_summary.json").write_text("{}")
        (product_dir / "run_report.md").write_text("# report")
        with TestClient(create_app(config)) as client:
            payload = client.get("/api/status").json()
        states = {stage["name"]: stage for stage in payload["stages"]}
        assert states["Measurement Sets"] == {
            "name": "Measurement Sets",
            "state": "ready",
            "count": 2,
        }
        assert states["Calibration tables"]["state"] == "ready"
        assert states["Calibration tables"]["count"] == 2
        assert states["Tile images"] == {"name": "Tile images", "state": "ready", "count": 1}
        assert states["Hourly-epoch mosaic"]["state"] == "ready"
        assert payload["mosaic"]["path"].endswith("2026-07-13T1100_mosaic.fits")
        assert payload["incoming"]["hdf5_count"] == 2
        manifest = payload["run_products"]["manifest"]
        assert manifest["path"].endswith("2026-07-13_manifest.json")
        assert manifest["size_bytes"] == len(manifest_text)
        assert payload["run_products"]["summary"]["path"].endswith("2026-07-13_run_summary.json")
        assert payload["run_products"]["report"]["path"].endswith("run_report.md")

    def test_calibration_stage_requires_both_bandpass_and_gains(self, tmp_path: Path, monkeypatch):
        monkeypatch.setattr(qa_server, "process_status", lambda: [])
        config = _make_config(tmp_path)
        ms_dir = config.stage / "ms"
        ms_dir.mkdir(parents=True)
        (ms_dir / "2026-07-13T11:26:05_0~23.b").write_text("bp only, no gains")
        with TestClient(create_app(config)) as client:
            payload = client.get("/api/status").json()
        cal = next(stage for stage in payload["stages"] if stage["name"] == "Calibration tables")
        assert cal["state"] == "not_yet"
        assert cal["count"] == 1

    def test_log_tail_is_suffix_of_newest_log(self, tmp_path: Path, monkeypatch):
        monkeypatch.setattr(qa_server, "process_status", lambda: [])
        config = _make_config(tmp_path)
        product_dir = config.products / "2026-07-13"
        product_dir.mkdir(parents=True)
        old_log = product_dir / "run_old.log"
        old_log.write_text("stale line\n")
        new_lines = [f"line-{i:02d}\n" for i in range(30)]
        new_log = product_dir / "run_new.log"
        new_log.write_text("".join(new_lines))
        now = new_log.stat().st_mtime
        os.utime(old_log, (now - 100, now - 100))
        os.utime(new_log, (now + 100, now + 100))
        with TestClient(create_app(config)) as client:
            payload = client.get("/api/status").json()
        assert payload["log"]["file"]["path"].endswith("run_new.log")
        assert payload["log"]["tail"] == new_lines[-24:]


class TestProcessAndPidParsing:
    """Criterion: operator heartbeat parsing contracts. ps rows (from
    `ps -eo pid=,etimes=,stat=,args=`) are parsed positionally into
    pid/elapsed_seconds/state/command for pipeline-relevant commands only;
    malformed rows are skipped; a failing ps yields an error record instead of
    raising. *.pid hint files yield labeled pids with /proc visibility, and a
    missing campaign directory yields an empty list. Basis: the procps output
    column contract and the campaign runbook LABEL_PID=N file format."""

    def test_process_status_parses_ps_rows_positionally(self, monkeypatch):
        stdout = (
            "  1234    56 S    /opt/miniforge/envs/casa6/bin/python scripts/batch_pipeline.py --date 2026-07-13\n"
            "  4321   999 R    wsclean -size 4800 4800 obs.ms\n"
            "  5555    12 S    sshd: ubuntu@pts/0\n"
            "  9999 5 wsclean\n"
        )
        fake = types.SimpleNamespace(
            run=lambda *args, **kwargs: subprocess.CompletedProcess(
                args=args, returncode=0, stdout=stdout, stderr=""
            ),
            SubprocessError=subprocess.SubprocessError,
        )
        monkeypatch.setattr(qa_server, "subprocess", fake)
        assert qa_server.process_status() == [
            {
                "pid": 1234,
                "elapsed_seconds": 56,
                "state": "S",
                "command": "/opt/miniforge/envs/casa6/bin/python scripts/batch_pipeline.py --date 2026-07-13",
            },
            {
                "pid": 4321,
                "elapsed_seconds": 999,
                "state": "R",
                "command": "wsclean -size 4800 4800 obs.ms",
            },
        ]

    def test_process_status_returns_error_record_when_ps_fails(self, monkeypatch):
        def boom(*args, **kwargs):
            raise OSError("ps unavailable")

        fake = types.SimpleNamespace(run=boom, SubprocessError=subprocess.SubprocessError)
        monkeypatch.setattr(qa_server, "subprocess", fake)
        assert qa_server.process_status() == [{"error": "ps unavailable"}]

    def test_pid_hints_parse_labels_and_proc_visibility(self, tmp_path: Path):
        config = _make_config(tmp_path)
        config.campaign_outputs.mkdir(parents=True)
        own_pid = os.getpid()
        (config.campaign_outputs / "run.pid").write_text(
            f"BATCH_PID={own_pid}\nWSCLEAN_PID=4000000000\nnot a pid line\n"
        )
        hints = qa_server._pid_hints(config)
        assert [(hint["label"], hint["pid"]) for hint in hints] == [
            ("BATCH_PID", own_pid),
            ("WSCLEAN_PID", 4000000000),
        ]
        assert hints[0]["visible"] is True
        assert hints[1]["visible"] is False
        assert hints[0]["source"].endswith("run.pid")

    def test_pid_hints_missing_directory_is_empty(self, tmp_path: Path):
        assert qa_server._pid_hints(_make_config(tmp_path)) == []


class TestDashboardAndHealth:
    """Criterion: the root dashboard renders 200 HTML and must HTML-escape log
    file content before embedding it -- batch logs echo external inputs
    (filenames, casa/wsclean output), so unescaped injection is stored XSS in
    an auto-refreshing operator dashboard. /health returns status ok with a
    parseable timezone-aware UTC timestamp. Basis: output-encoding security
    invariant and the health-probe contract."""

    def test_dashboard_escapes_log_content(self, tmp_path: Path, monkeypatch):
        monkeypatch.setattr(qa_server, "process_status", lambda: [])
        config = _make_config(tmp_path)
        config.campaign_outputs.mkdir(parents=True)
        payload = "<script>alert('qa-xss')</script>"
        (config.campaign_outputs / "batch_run_h11_test.log").write_text(payload + "\n")
        with TestClient(create_app(config)) as client:
            response = client.get("/")
        assert response.status_code == 200
        assert payload not in response.text
        assert "&lt;script&gt;alert(&#x27;qa-xss&#x27;)&lt;/script&gt;" in response.text

    def test_health_contract(self, tmp_path: Path):
        with TestClient(create_app(_make_config(tmp_path))) as client:
            response = client.get("/health")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ok"
        parsed = datetime.fromisoformat(body["time"])
        assert parsed.tzinfo is not None


class TestFindCsv:
    """Criterion: find_csv's contract is the newest matching table (docstring);
    newest means max mtime, not lexicographically last."""

    def test_returns_newest_by_mtime_not_lexicographic(self, tmp_path: Path):
        config = _make_config(tmp_path)
        directory = config.products / "2026-01-25"
        directory.mkdir(parents=True)
        newer_lex_first = directory / "2026-01-25T0200_a_phot.csv"
        older_lex_last = directory / "2026-01-25T0200_z_phot.csv"
        newer_lex_first.write_text("x\n")
        older_lex_last.write_text("x\n")
        os.utime(older_lex_last, (1_000, 1_000))
        os.utime(newer_lex_first, (2_000, 2_000))

        assert qa_server.find_csv(config, "2026-01-25", "T0200") == newer_lex_first


class TestCampaignStatusLogSelection:
    """Criterion: an epoch's status may only adopt campaign logs whose filename
    hour equals the requested hour (padded or unpadded form); hour 1 must never
    adopt h11-h19 logs even when they are newer."""

    def test_rejects_other_hours_sharing_digit_prefix(self, tmp_path: Path, monkeypatch):
        monkeypatch.setattr(qa_server, "process_status", lambda: [])
        config = _make_config(tmp_path)
        config.campaign_outputs.mkdir(parents=True)
        own = config.campaign_outputs / "batch_run_h1_a.log"
        other = config.campaign_outputs / "batch_run_h11_b.log"
        own.write_text("own\n")
        other.write_text("other\n")
        os.utime(own, (1_000, 1_000))
        os.utime(other, (2_000, 2_000))

        status = qa_server.campaign_status(config, "2026-07-13", 1)

        assert status["log"]["file"]["path"].endswith("batch_run_h1_a.log")


class TestControlAuth:
    """Criterion: mutating control routes fail closed -- no DSA110_CONTROL_TOKEN
    in the environment means 403 for every mutating request regardless of the
    header; a wrong or missing bearer is 403; the correct bearer is accepted;
    request fields that fail RunRequest's closed grammar are 4xx; the audit log
    records mutating requests without ever containing the token; run listing
    stays readable without a token (read-only surface). Basis: fail-closed
    auth invariant for a publicly-tunneled control surface."""

    def _client(self, tmp_path, monkeypatch, token_env):
        if token_env is None:
            monkeypatch.delenv("DSA110_CONTROL_TOKEN", raising=False)
        else:
            monkeypatch.setenv("DSA110_CONTROL_TOKEN", token_env)
        monkeypatch.setenv("DSA110_CONTROL_DIR", str(tmp_path / "control"))
        monkeypatch.setenv("DSA110_REPO_ROOT", str(tmp_path))
        import sys as _sys

        monkeypatch.setenv("DSA110_PIPELINE_PYTHON", _sys.executable)
        scripts = tmp_path / "scripts"
        scripts.mkdir(exist_ok=True)
        (scripts / "batch_pipeline.py").write_text("print('plan ok')\n")
        return TestClient(create_app(_make_config(tmp_path)))

    def test_launch_without_token_env_is_403_even_with_header(self, tmp_path, monkeypatch):
        with self._client(tmp_path, monkeypatch, token_env=None) as client:
            response = client.post(
                "/api/runs",
                json={"date": "2026-01-25", "dry_run": True},
                headers={"Authorization": "Bearer anything"},
            )
        assert response.status_code == 403

    def test_wrong_or_missing_token_is_403(self, tmp_path, monkeypatch):
        with self._client(tmp_path, monkeypatch, token_env="s3cret") as client:
            wrong = client.post(
                "/api/runs",
                json={"date": "2026-01-25", "dry_run": True},
                headers={"Authorization": "Bearer wrong"},
            )
            missing = client.post("/api/runs", json={"date": "2026-01-25", "dry_run": True})
        assert wrong.status_code == 403
        assert missing.status_code == 403

    def test_dry_run_with_token_returns_plan_and_audits(self, tmp_path, monkeypatch):
        with self._client(tmp_path, monkeypatch, token_env="s3cret") as client:
            response = client.post(
                "/api/runs",
                json={"date": "2026-01-25", "dry_run": True},
                headers={"Authorization": "Bearer s3cret"},
            )
        assert response.status_code == 200
        assert "plan ok" in response.json()["plan"]
        audit = (tmp_path / "control" / "audit.jsonl").read_text()
        assert '"dry_run": true' in audit
        assert "s3cret" not in audit

    def test_injection_shaped_date_is_4xx(self, tmp_path, monkeypatch):
        with self._client(tmp_path, monkeypatch, token_env="s3cret") as client:
            response = client.post(
                "/api/runs",
                json={"date": "2026-01-25; rm -rf /", "dry_run": True},
                headers={"Authorization": "Bearer s3cret"},
            )
        assert response.status_code in (400, 422)

    def test_run_listing_is_readable_without_token(self, tmp_path, monkeypatch):
        with self._client(tmp_path, monkeypatch, token_env="s3cret") as client:
            response = client.get("/api/runs")
        assert response.status_code == 200
        assert response.json() == {"runs": []}

    def test_launch_terminate_roundtrip_with_token(self, tmp_path, monkeypatch):
        with self._client(tmp_path, monkeypatch, token_env="s3cret") as client:
            (tmp_path / "scripts" / "batch_pipeline.py").write_text("import time; time.sleep(60)\n")
            headers = {"Authorization": "Bearer s3cret"}
            launched = client.post("/api/runs", json={"date": "2026-01-25"}, headers=headers)
            assert launched.status_code == 200
            run_id = launched.json()["run_id"]
            detail = client.get(f"/api/runs/{run_id}")
            assert detail.status_code == 200
            assert detail.json()["status"] == "running"
            terminated = client.post(f"/api/runs/{run_id}/terminate", headers=headers)
            assert terminated.status_code == 200
            assert terminated.json()["status"] == "terminated"
            unauth = client.post(f"/api/runs/{run_id}/terminate")
            assert unauth.status_code == 403

    def test_unknown_run_detail_is_404(self, tmp_path, monkeypatch):
        with self._client(tmp_path, monkeypatch, token_env="s3cret") as client:
            client.post(
                "/api/runs",
                json={"date": "2026-01-25", "dry_run": True},
                headers={"Authorization": "Bearer s3cret"},
            )
            assert client.get("/api/runs/nope").status_code == 404


class TestEpochDiscovery:
    """Criterion: the historical QA list derives from actual mosaics on stage
    (newest mtime first, canonical names only), falling back to the static
    EPOCHS list when stage is empty so the dashboard never renders blank.
    Basis: filesystem-truth navigation replacing the hardcoded epoch list."""

    def test_discovers_epochs_from_stage_newest_first(self, tmp_path: Path):
        config = _make_config(tmp_path)
        old = _write_mosaic(config, "2026-01-25", "T0200", np.ones((4, 4), np.float32))
        _write_mosaic(config, "2026-07-13", "T1100", np.ones((4, 4), np.float32))
        os.utime(old, (1, 1))
        discovered = qa_server.discover_epochs(config)
        assert discovered[0] == ("2026-07-13", "T1100")
        assert ("2026-01-25", "T0200") in discovered

    def test_falls_back_to_static_epochs_when_stage_empty(self, tmp_path: Path):
        assert qa_server.discover_epochs(_make_config(tmp_path)) == qa_server.EPOCHS

    def test_ignores_malformed_names_and_respects_limit(self, tmp_path: Path):
        config = _make_config(tmp_path)
        directory = config.stage / "images" / "mosaic_2026-01-25"
        directory.mkdir(parents=True)
        (directory / "junk_mosaic.fits").write_bytes(b"x")
        assert qa_server.discover_epochs(config) == qa_server.EPOCHS
        for hour in range(5):
            _write_mosaic(config, "2026-01-25", f"T{hour:02d}00", np.ones((2, 2), np.float32))
        assert len(qa_server.discover_epochs(config, limit=3)) == 3


class TestControlUi:
    """Criterion: the dashboard exposes the control surface (form posting to
    /api/runs) and per-run detail pages render registry state plus an
    HTML-escaped log tail. Basis: operator-facing contract of the control UI."""

    def test_dashboard_shows_pipeline_control_panel(self, tmp_path: Path, monkeypatch):
        monkeypatch.setattr(qa_server, "process_status", lambda: [])
        monkeypatch.setenv("DSA110_CONTROL_DIR", str(tmp_path / "control"))
        with TestClient(create_app(_make_config(tmp_path))) as client:
            page = client.get("/").text
        assert "Pipeline control" in page
        assert 'id="run-form"' in page and "/api/runs" in page
        assert 'id="control-token"' in page

    def test_run_detail_page_renders_log_tail_escaped(self, tmp_path: Path, monkeypatch):
        import sys as _sys
        import time as _time

        monkeypatch.setenv("DSA110_CONTROL_DIR", str(tmp_path / "control"))
        monkeypatch.setenv("DSA110_PIPELINE_PYTHON", _sys.executable)
        monkeypatch.setenv("DSA110_REPO_ROOT", str(tmp_path))
        (tmp_path / "scripts").mkdir()
        (tmp_path / "scripts" / "batch_pipeline.py").write_text(
            "print('hello from run <script>bad</script>')\n"
        )
        from dsa110_continuum.observability.control import (
            ControlConfig,
            RunRequest,
            launch_run,
        )

        record = launch_run(
            RunRequest(date="2026-01-25"),
            ControlConfig(
                repo_root=tmp_path, python=_sys.executable, control_dir=tmp_path / "control"
            ),
        )
        _time.sleep(1.0)
        with TestClient(create_app(_make_config(tmp_path))) as client:
            page = client.get(f"/control/runs/{record['run_id']}")
        assert page.status_code == 200
        assert "hello from run" in page.text
        assert "<script>bad</script>" not in page.text

    def test_unknown_run_page_is_404(self, tmp_path: Path, monkeypatch):
        monkeypatch.setenv("DSA110_CONTROL_DIR", str(tmp_path / "control"))
        with TestClient(create_app(_make_config(tmp_path))) as client:
            assert client.get("/control/runs/nope").status_code == 404


class TestRunProvenancePage:
    """Criterion: /runs/{date} surfaces manifest truth (pipeline_verdict, gate
    reasons, per-epoch QA) plus the run report; a partial manifest renders with
    placeholder fields instead of 500; unknown dates are 404 and traversal
    inputs are 4xx. Basis: manifest schema written by qa/provenance.py."""

    def _write_manifest(self, config: DashboardConfig, date: str = "2026-01-25") -> None:
        import json as _json

        directory = config.products / date
        directory.mkdir(parents=True, exist_ok=True)
        manifest = {
            "date": date,
            "cal_date": date,
            "git_sha": "abc1234",
            "command_line": "batch_pipeline.py --date " + date,
            "pipeline_verdict": "DEGRADED",
            "gates": [
                {
                    "gate": "epoch_qa:22",
                    "verdict": "FAIL",
                    "reason": "catalog completeness 0.42 < 0.5",
                }
            ],
            "epochs": [
                {
                    "hour": 22,
                    "n_tiles": 11,
                    "status": "ok",
                    "qa_result": "FAIL",
                    "peak": 12.5,
                    "rms": 0.008,
                    "mosaic_path": "/x.fits",
                    "gaincal_status": "ok",
                    "n_sources": 40,
                    "median_ratio": 0.9,
                    "weight_path": "/x.w.fits",
                }
            ],
            "tiles": [],
            "run_log": "run_x.log",
        }
        (directory / f"{date}_manifest.json").write_text(_json.dumps(manifest))
        (directory / "run_report.md").write_text("# Run report\nverdict DEGRADED\n")

    def test_renders_verdict_gates_and_epochs(self, tmp_path: Path):
        config = _make_config(tmp_path)
        self._write_manifest(config)
        with TestClient(create_app(config)) as client:
            page = client.get("/runs/2026-01-25")
        assert page.status_code == 200
        assert "DEGRADED" in page.text
        assert "catalog completeness" in page.text
        assert "T2200" in page.text
        assert "Run report" in page.text

    def test_partial_manifest_renders_without_500(self, tmp_path: Path):
        import json as _json

        config = _make_config(tmp_path)
        directory = config.products / "2026-01-25"
        directory.mkdir(parents=True)
        (directory / "2026-01-25_manifest.json").write_text(
            _json.dumps({"pipeline_verdict": "CLEAN"})
        )
        with TestClient(create_app(config)) as client:
            page = client.get("/runs/2026-01-25")
        assert page.status_code == 200
        assert "CLEAN" in page.text

    def test_missing_manifest_is_404_and_traversal_rejected(self, tmp_path: Path):
        with TestClient(create_app(_make_config(tmp_path))) as client:
            assert client.get("/runs/2026-01-26").status_code == 404
            assert 400 <= client.get("/runs/..%2f..%2fetc").status_code < 500


class TestLightcurveView:
    """Criterion: /sources/lightcurve matches the nearest forced-photometry
    source per epoch within a positional radius (RA scaled by cos(dec)), renders
    a table + PNG + eta/V variability metrics from the canonical formulas, and
    rejects out-of-range coordinates. Basis: forced-phot CSV schema on disk
    (ra_deg, dec_deg, dsa_peak_jyb, ...) and beam-scale positional matching."""

    HEADER = "ra_deg,dec_deg,nvss_flux_jy,dsa_peak_jyb,dsa_peak_err_jyb,dsa_nvss_ratio\n"

    def _write_epoch_csv(self, config: DashboardConfig, date: str, epoch: str, flux: float):
        directory = config.products / date
        directory.mkdir(parents=True, exist_ok=True)
        (directory / f"{date}{epoch}_forced_phot.csv").write_text(
            self.HEADER
            + f"47.499,17.099833,4.87,{flux},0.02,0.95\n"
            + "120.0,-5.0,1.0,1.1,0.05,1.1\n"
        )

    def test_matches_position_across_epochs(self, tmp_path: Path):
        config = _make_config(tmp_path)
        self._write_epoch_csv(config, "2026-01-25", "T0200", 3.9)
        self._write_epoch_csv(config, "2026-02-12", "T0000", 4.1)
        points = qa_server.lightcurve_points(
            config, ra_deg=47.499, dec_deg=17.0998, radius_arcsec=30.0
        )
        assert len(points) == 2
        assert {point["epoch"] for point in points} == {"2026-01-25T0200", "2026-02-12T0000"}
        assert all(abs(point["flux_jy"] - 4.0) < 0.2 for point in points)
        assert all(point["separation_arcsec"] < 30.0 for point in points)

    def test_no_match_outside_radius(self, tmp_path: Path):
        config = _make_config(tmp_path)
        self._write_epoch_csv(config, "2026-01-25", "T0200", 3.9)
        assert qa_server.lightcurve_points(config, 47.499, 18.5, 30.0) == []

    def test_csv_missing_columns_is_skipped(self, tmp_path: Path):
        config = _make_config(tmp_path)
        directory = config.products / "2026-01-25"
        directory.mkdir(parents=True)
        (directory / "2026-01-25T0200_forced_phot.csv").write_text("other\n1\n")
        assert qa_server.lightcurve_points(config, 47.499, 17.0998, 30.0) == []

    def test_lightcurve_page_and_png(self, tmp_path: Path):
        config = _make_config(tmp_path)
        self._write_epoch_csv(config, "2026-01-25", "T0200", 3.9)
        self._write_epoch_csv(config, "2026-02-12", "T0000", 4.1)
        with TestClient(create_app(config)) as client:
            page = client.get("/sources/lightcurve?ra=47.499&dec=17.0998")
            png = client.get("/sources/lightcurve.png?ra=47.499&dec=17.0998")
        assert page.status_code == 200 and "2026-01-25" in page.text
        assert "eta" in page.text.lower()
        assert png.status_code == 200 and png.content.startswith(b"\x89PNG")

    def test_no_points_page_is_200_png_is_404(self, tmp_path: Path):
        with TestClient(create_app(_make_config(tmp_path))) as client:
            page = client.get("/sources/lightcurve?ra=47.499&dec=17.0998")
            png = client.get("/sources/lightcurve.png?ra=47.499&dec=17.0998")
        assert page.status_code == 200 and "No matching source" in page.text
        assert png.status_code == 404

    def test_bad_coords_rejected(self, tmp_path: Path):
        with TestClient(create_app(_make_config(tmp_path))) as client:
            assert 400 <= client.get("/sources/lightcurve?ra=999&dec=0").status_code < 500
            assert 400 <= client.get("/sources/lightcurve?ra=10&dec=95").status_code < 500
            assert (
                400
                <= client.get("/sources/lightcurve?ra=10&dec=10&radius_arcsec=9999").status_code
                < 500
            )

    def test_dashboard_links_lightcurve_lookup(self, tmp_path: Path, monkeypatch):
        monkeypatch.setattr(qa_server, "process_status", lambda: [])
        with TestClient(create_app(_make_config(tmp_path))) as client:
            page = client.get("/").text
        assert "/sources/lightcurve" in page

    def test_nan_flux_epochs_are_excluded(self, tmp_path: Path):
        config = _make_config(tmp_path)
        self._write_epoch_csv(config, "2026-01-25", "T0200", 3.9)
        directory = config.products / "2026-02-12"
        directory.mkdir(parents=True)
        (directory / "2026-02-12T0100_forced_phot.csv").write_text(
            self.HEADER + "47.499,17.099833,4.87,nan,nan,nan\n"
        )
        points = qa_server.lightcurve_points(config, 47.499, 17.0998, 30.0)
        assert [point["epoch"] for point in points] == ["2026-01-25T0200"]
