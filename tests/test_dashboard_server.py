"""Tests for scripts/dashboard_server.py (unified pipeline console).

Uses a synthetic on-disk tree; no telescope paths, no CASA. FastAPI's
TestClient drives the app in-process.
"""

import importlib.util
import json
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
DATE = "2026-01-25"


def _mkfits(path: Path, size: int = 32, peak: float = 10.0) -> None:
    from astropy.io import fits

    rng = np.random.default_rng(0)
    img = rng.normal(0, 0.003, (size, size)).astype(np.float32)
    img[size // 2, size // 2] = peak
    path.parent.mkdir(parents=True, exist_ok=True)
    fits.PrimaryHDU(img).writeto(path, overwrite=True)


@pytest.fixture()
def console(tmp_path, monkeypatch):
    """Load a fresh dashboard_server module pointed at a synthetic tree."""
    incoming = tmp_path / "incoming"
    ms_dir = tmp_path / "ms"
    image_base = tmp_path / "images"
    products = tmp_path / "products"
    job_dir = tmp_path / "jobs"
    pipeline_db = tmp_path / "pipeline.sqlite3"

    # incoming: one complete 16-subband group + one partial
    incoming.mkdir()
    for sb in range(16):
        (incoming / f"{DATE}T22:00:05_sb{sb:02d}.hdf5").write_bytes(b"x")
    for sb in range(4):
        (incoming / f"{DATE}T22:05:05_sb{sb:02d}.hdf5").write_bytes(b"x")

    conn = sqlite3.connect(pipeline_db)
    conn.execute(
        "CREATE TABLE hdf5_files (group_id TEXT, subband_num INTEGER, obs_date TEXT, stored INTEGER)"
    )
    conn.executemany(
        "INSERT INTO hdf5_files VALUES (?, ?, ?, 1)",
        [("22:00", sb, DATE) for sb in range(16)] + [("22:05", sb, DATE) for sb in range(4)],
    )
    conn.commit()
    conn.close()

    # MS + cal tables
    for m in ("00", "05"):
        d = ms_dir / f"{DATE}T22:{m}:05.ms"
        d.mkdir(parents=True)
        (d / "table.dat").write_bytes(b"t")
    (ms_dir / f"{DATE}T22:26:05_0~23.b").mkdir()
    (ms_dir / f"{DATE}T22:26:05_0~23.g").mkdir()
    # non-date-prefixed entries must not become phantom dates
    (ms_dir / "verify_meridian.ms").mkdir()
    (ms_dir / f"verify_{DATE}T22:26:05.g").mkdir()

    # tiles + mosaic + weights + checkpoint
    sd = image_base / f"mosaic_{DATE}"
    _mkfits(sd / f"{DATE}T22:00:05_image.fits")
    mosaic = sd / f"{DATE}T2200_mosaic.fits"
    _mkfits(mosaic)
    _mkfits(sd / f"{DATE}T2200_mosaic.weights.fits", peak=1.0)
    (sd / ".tile_checkpoint.json").write_text(
        json.dumps(
            {
                "date": DATE,
                "cal_date": DATE,
                "completed": ["a.ms"],
                "failed": [
                    {"ms_path": "b.ms", "failure_count": 3, "error": "boom"},
                    {"ms_path": "c.ms", "failure_count": 1, "error": "meh"},
                ],
            }
        )
    )

    # products
    pd_dir = products / DATE
    pd_dir.mkdir(parents=True)
    for hr, boost in (("2200", 1.0), ("2300", 1.5)):
        (pd_dir / f"{DATE}T{hr}_forced_phot.csv").write_text(
            "source_id,flux_jy,flux_err_jy,nvss_flux_jy,dsa_nvss_ratio\n"
            + "\n".join(
                f"S{i},{0.5 * (boost if i == 0 else 1.0):.3f},0.01,0.5,1.0{i}" for i in range(5)
            )
        )
    (pd_dir / f"{DATE}_manifest.json").write_text(
        json.dumps({"date": DATE, "pipeline_verdict": "CLEAN", "epochs": [], "gates": []})
    )
    (pd_dir / "run_report.md").write_text("# report")
    (pd_dir / f"run_{DATE}.log").write_text("line1\nline2\n")

    env = {
        "DSA110_INCOMING_DIR": str(incoming),
        "DSA110_MS_DIR": str(ms_dir),
        "DSA110_IMAGE_BASE": str(image_base),
        "DSA110_PRODUCTS_BASE": str(products),
        "PIPELINE_DB": str(pipeline_db),
        "DSA110_JOB_DIR": str(job_dir),
        "DSA110_THUMB_DIR": str(tmp_path / "thumbs"),
        "DSA110_DASH_TOKEN": "sekrit",
        "DSA110_PYTHON": sys.executable,
        "DSA110_REPO_DIR": str(tmp_path / "repo"),
        "DSA110_LOG_GLOBS": str(tmp_path / "none" / "*.log"),
    }
    for k, v in env.items():
        monkeypatch.setenv(k, v)

    # stub batch_pipeline.py so control/run spawns something harmless
    stub = tmp_path / "repo" / "scripts"
    stub.mkdir(parents=True)
    (stub / "batch_pipeline.py").write_text("import sys; print('stub ok', sys.argv[1:])")
    (stub / "auto_pipeline.py").write_text("import sys; print('auto stub ok', sys.argv[1:])")

    spec = importlib.util.spec_from_file_location(
        "dashboard_server_test", REPO / "scripts" / "dashboard_server.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    client = TestClient(mod.app)
    return mod, client


def test_health_reports_control_enabled(console):
    _, client = console
    d = client.get("/api/health").json()
    assert d["status"] == "ok"
    assert d["control_enabled"] is True


def test_dates_matrix_aggregates_all_stages(console):
    _, client = console
    d = client.get("/api/dates").json()
    rows = {r["date"]: r for r in d["dates"]}
    assert DATE in rows
    r = rows[DATE]
    assert r["incoming"] == {"timestamps": 2, "files": 20, "complete_groups": 1}
    assert r["indexed"] == {"files": 20, "groups": 2, "complete_groups": 1}
    assert r["n_ms"] == 2
    assert r["cal"] == {"bandpass": 1, "gain": 1}
    assert r["n_tiles"] == 1  # mosaic + weights excluded
    assert r["n_tile_artifacts"] == 1
    assert r["n_failures"] == 2
    assert r["n_quarantine_risk"] == 1
    assert r["n_mosaics"] == 1
    assert r["verdict"] == "CLEAN"
    assert r["n_phot"] == 2
    assert [s["key"] for s in r["stages"]] == [
        "ingest",
        "index",
        "conversion",
        "flagging",
        "calibration",
        "imaging",
        "mosaic",
        "qa",
        "photometry",
        "archive",
    ]


def test_date_detail_includes_metrics_and_products(console):
    _, client = console
    d = client.get(f"/api/date/{DATE}").json()
    assert d["mosaics"][0]["weights"] is True
    assert d["mosaics"][0]["peak"] == pytest.approx(10.0, abs=0.5)
    assert d["products"]["phot_csvs"][0]["n_sources"] == 5
    assert d["tiles"]["checkpoint"]["failed"][0]["failure_count"] == 3
    assert client.get("/api/date/evil-date").status_code == 400


def test_artifact_viewer_and_traversal_guard(console, tmp_path):
    mod, client = console
    log = client.get(
        "/api/artifact", params={"path": str(mod.PRODUCTS_BASE / DATE / f"run_{DATE}.log")}
    ).json()
    assert log["tail"] == ["line1", "line2"]
    man = client.get(
        "/api/artifact",
        params={"path": str(mod.PRODUCTS_BASE / DATE / f"{DATE}_manifest.json")},
    ).json()
    assert man["json"]["pipeline_verdict"] == "CLEAN"
    # FITS header route
    fh = client.get(
        "/api/artifact",
        params={"path": str(mod.IMAGE_BASE / f"mosaic_{DATE}" / f"{DATE}T2200_mosaic.fits")},
    ).json()
    assert "fits_header" in fh and fh["peak"] == pytest.approx(10.0, abs=0.5)
    # traversal denied
    assert client.get("/api/artifact", params={"path": "/etc/passwd"}).status_code == 403


def test_thumbnail_renders_png(console):
    _, client = console
    r = client.get(f"/api/thumb/{DATE}/{DATE}T2200_mosaic.fits.png")
    assert r.status_code == 200
    assert r.content[:8] == b"\x89PNG\r\n\x1a\n"
    assert client.get(f"/api/thumb/{DATE}/nope.fits.png").status_code == 404


def test_control_requires_token(console):
    _, client = console
    r = client.post("/api/control/run", json={"date": DATE})
    assert r.status_code == 403
    r = client.post("/api/control/run", json={"date": DATE}, headers={"X-DSA110-Token": "wrong"})
    assert r.status_code == 403


def test_control_disabled_when_token_unset(console, monkeypatch):
    mod, client = console
    monkeypatch.setattr(mod, "DASH_TOKEN", "")
    r = client.post("/api/control/run", json={"date": DATE}, headers={"X-DSA110-Token": ""})
    assert r.status_code == 403
    assert "disabled" in r.json()["detail"].lower()


def test_control_run_spawns_job_and_logs(console):
    _, client = console
    r = client.post(
        "/api/control/run",
        json={
            "date": DATE,
            "dry_run": True,
            "start_hour": 22,
            "end_hour": 23,
            "rfi_mode": "cflag",
        },
        headers={"X-DSA110-Token": "sekrit"},
    )
    assert r.status_code == 200
    job = r.json()
    assert job["status"] == "running"
    assert "--dry-run" in job["argv"] and "--start-hour" in job["argv"]
    assert job["argv"][job["argv"].index("--rfi-mode") + 1] == "cflag"
    # wait for the stub to finish, then check job bookkeeping + log capture
    import time

    for _ in range(50):
        d = client.get(f"/api/jobs/{job['id']}").json()
        if d["status"] != "running":
            break
        time.sleep(0.1)
    assert d["status"] == "completed"
    assert any("stub ok" in line for line in d.get("log_tail", []))


def test_control_run_spawns_end_to_end_flow(console):
    _, client = console
    r = client.post(
        "/api/control/run",
        json={
            "date": DATE,
            "flow": "end_to_end",
            "dry_run": False,
            "rfi_mode": "cflag",
        },
        headers={"X-DSA110-Token": "sekrit"},
    )
    assert r.status_code == 200
    job = r.json()
    assert job["kind"] == "end-to-end"
    assert job["argv"][-4:] == ["--date", DATE, "--rfi-mode", "cflag"]


def test_end_to_end_rejects_dry_run_and_hour_bounds(console):
    _, client = console
    headers = {"X-DSA110-Token": "sekrit"}
    assert (
        client.post(
            "/api/control/run",
            json={"date": DATE, "flow": "end_to_end", "dry_run": True},
            headers=headers,
        ).status_code
        == 400
    )
    assert (
        client.post(
            "/api/control/run",
            json={
                "date": DATE,
                "flow": "end_to_end",
                "dry_run": False,
                "start_hour": 22,
            },
            headers=headers,
        ).status_code
        == 400
    )


def test_control_run_validates_date(console):
    _, client = console
    r = client.post(
        "/api/control/run",
        json={"date": "2026-01-25; rm -rf /"},
        headers={"X-DSA110-Token": "sekrit"},
    )
    assert r.status_code == 400


def test_clear_quarantine_zeroes_failure_counts(console):
    mod, client = console
    r = client.post(
        "/api/control/clear-quarantine",
        json={"date": DATE},
        headers={"X-DSA110-Token": "sekrit"},
    )
    assert r.status_code == 200
    assert r.json()["cleared"] == 2
    ck = json.loads((mod.IMAGE_BASE / f"mosaic_{DATE}" / ".tile_checkpoint.json").read_text())
    assert all(f["failure_count"] == 0 for f in ck["failed"])
    # entries preserved (history kept, counts reset)
    assert {f["ms_path"] for f in ck["failed"]} == {"b.ms", "c.ms"}


def test_index_serves_ui(console):
    _, client = console
    r = client.get("/")
    assert r.status_code == 200
    assert "DSA-110 Pipeline Console" in r.text


def test_sky_state_and_sources(console):
    _, client = console
    d = client.get("/api/sky").json()
    assert 0.0 <= d["lst_hours"] < 24.0
    assert abs(d["meridian_ra_deg"] - d["lst_hours"] * 15.0) < 0.1
    names = {s["name"] for s in d["sources"]}
    assert {"Cas A", "Cyg A", "3C286"} <= names
    kinds = {s["kind"] for s in d["sources"]}
    assert "ateam" in kinds and "cal" in kinds


def test_sky_map_has_simulated_context_and_hover_tooltips(console):
    _, client = console
    page = client.get("/").text

    assert "galacticPlanePaths" in page
    assert "simulated-sources" in page
    assert 'id="sky-tip" role="tooltip"' in page
    assert 'class="sky-source' in page
    assert "pointerover" in page and "focusin" in page
    assert "very bright radio source" in page


def test_antennas_json_source_preferred(console, tmp_path, monkeypatch):
    mod, client = console
    status = tmp_path / "ant_status.json"
    status.write_text(
        json.dumps(
            {
                "asof": "2026-07-15T00:00:00Z",
                "antennas": {"1": "good", "2": "bad", "3": {"status": "warn", "flag_frac": 0.4}},
            }
        )
    )
    monkeypatch.setattr(mod, "ANT_STATUS_JSON", str(status))
    monkeypatch.setattr(mod, "N_ANT", 5)
    d = client.get("/api/antennas").json()
    assert d["source"] == "json"
    by = {a["name"]: a for a in d["antennas"]}
    assert by["1"]["status"] == "good"
    assert by["2"]["status"] == "bad"
    assert by["3"]["status"] == "warn"
    assert by["5"]["status"] == "unknown"  # padded to N_ANT
    assert d["counts"]["good"] == 1


def test_antennas_fallback_unknown_when_no_source(console, monkeypatch):
    mod, client = console
    monkeypatch.setattr(mod, "ANT_STATUS_JSON", "/nonexistent/ant.json")
    monkeypatch.setattr(mod, "N_ANT", 4)
    d = client.get("/api/antennas").json()
    assert d["source"].startswith(("none", "caltable"))
    assert d["n"] >= 4


def test_science_feed_aggregates(console):
    _, client = console
    d = client.get("/api/science").json()
    row = next(r for r in d["dates"] if r["date"] == DATE)
    assert row["verdict"] == "CLEAN"
    assert row["mosaics"][0]["weights"] is True
    assert row["phot"][0]["n_sources"] == 5


def test_three_pages_served(console):
    _, client = console
    for path, marker in [
        ("/", "/ telescope"),
        ("/pipeline", "/ pipeline"),
        ("/science", "/ science"),
    ]:
        r = client.get(path)
        assert r.status_code == 200
        assert marker in r.text


def test_antenna_positions_from_csv(console, tmp_path, monkeypatch):
    mod, client = console
    csv = tmp_path / "antpos.csv"
    csv.write_text("ant_id,east_m,north_m\n1,-100,0\n2,0,0\n3,100,0\n4,0,-100\n")
    monkeypatch.setattr(mod, "ANTPOS_CSV", str(csv))
    monkeypatch.setattr(mod, "N_ANT", 4)
    monkeypatch.setattr(mod, "ANT_STATUS_JSON", "/nonexistent.json")
    d = client.get("/api/antennas").json()
    assert d["has_positions"] is True
    by = {a["name"]: a for a in d["antennas"]}
    assert by["1"]["x_m"] == -100.0 and by["4"]["y_m"] == -100.0


def test_antenna_positions_latlon_projection(console, tmp_path, monkeypatch):
    mod, client = console
    csv = tmp_path / "antpos_ll.csv"
    csv.write_text(
        "antenna,lat,lon\n1,37.2339,-118.2817\n2,37.2339,-118.2717\n3,37.2439,-118.2817\n"
    )
    monkeypatch.setattr(mod, "ANTPOS_CSV", str(csv))
    pos = mod._ant_positions()
    assert abs(pos["3"]["y_m"] - pos["1"]["y_m"] - 1105.4) < 15  # ~0.01 deg north
    assert pos["2"]["x_m"] > pos["1"]["x_m"]  # east positive


def test_dates_ignore_non_date_names(console):
    _, client = console
    d = client.get("/api/dates").json()
    assert [r["date"] for r in d["dates"]] == [DATE]


def test_catalog_sources_reads_views(console, tmp_path, monkeypatch):
    import sqlite3

    mod, _ = console
    db = tmp_path / "cal.sqlite3"
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE calibrators (name TEXT, ra_deg REAL, dec_deg REAL)")
    conn.execute("CREATE TABLE fluxes (name TEXT, band TEXT, flux_jy REAL)")
    conn.execute("INSERT INTO calibrators VALUES ('3C273', 187.28, 2.05)")
    conn.execute("INSERT INTO fluxes VALUES ('3C273', '20cm', 32.0)")
    conn.execute(
        "CREATE VIEW vla_20cm AS SELECT c.name, c.ra_deg, c.dec_deg, f.flux_jy "
        "FROM calibrators c JOIN fluxes f ON f.name = c.name"
    )
    conn.commit()
    conn.close()
    monkeypatch.setattr(mod, "VLA_CAL_DB", str(db))
    srcs = mod._catalog_sources()
    assert any(s["name"] == "3C273" and s["flux_jy"] == 32.0 for s in srcs)


def test_antenna_positions_station_coordinates_export(console, tmp_path, monkeypatch):
    mod, _ = console
    csv = tmp_path / "station.csv"
    csv.write_text(
        ",,,,\n,,,,\n,DSA-110 Station Coordinates,,,\n,Last updated:,,2/15/2022,\n,,,,\n"
        ",Station Number,Latitude,Longitude,Elevation (meters)\n"
        ",DSA-001,37.2333752,-118.2856408,1182.6\n"
        ",DSA-002,37.2333752,-118.2855760,1182.6\n"
        ",DSA-003,37.2343752,-118.2855112,1182.6\n"
    )
    monkeypatch.setattr(mod, "ANTPOS_CSV", str(csv))
    pos = mod._ant_positions()
    assert set(pos) == {"1", "2", "3"}
    assert pos["2"]["x_m"] > pos["1"]["x_m"]  # east positive
    assert pos["3"]["y_m"] > pos["1"]["y_m"]  # north positive


def test_variability_accepts_forced_phot_schema(console):
    mod, client = console
    pd_dir = mod.PRODUCTS_BASE / DATE
    for hr, boost in (("2200", 1.0), ("2300", 2.0)):
        (pd_dir / f"{DATE}T{hr}_forced_phot.csv").write_text(
            "ra_deg,dec_deg,nvss_flux_jy,dsa_peak_jyb,dsa_peak_err_jyb,dsa_nvss_ratio,source_id\n"
            "31.21,15.24,4.07,nan,nan,nan,1\n"
            f"36.05,27.84,3.02,{0.5 * boost},0.01,1.0,2\n"
            "40.00,20.00,2.50,0.8,0.01,1.0,3\n"
        )
    d = client.get("/api/variability").json()
    assert d["n_epochs"] == 2
    assert {s["source_id"] for s in d["sources"]} == {"2", "3"}  # nan-flux row excluded
    assert d["sources"][0]["source_id"] == "2"  # the boosted source ranks first


def test_variability_accepts_source_name_schema(console):
    mod, client = console
    pd_dir = mod.PRODUCTS_BASE / DATE
    for hr, boost in (("2200", 1.0), ("2300", 2.0)):
        (pd_dir / f"{DATE}T{hr}_forced_phot.csv").write_text(
            "source_name,ra_deg,dec_deg,catalog_flux_jy,measured_flux_jy,flux_err_jy,"
            "flux_ratio,snr\n"
            f"NVSS J1,36.05,27.84,3.02,{0.5 * boost},0.01,1.0,50\n"
            "NVSS J2,40.00,20.00,2.50,0.8,0.01,1.0,80\n"
        )
    d = client.get("/api/variability").json()
    assert d["n_epochs"] == 2
    assert {s["source_id"] for s in d["sources"]} == {"NVSS J1", "NVSS J2"}
    assert d["sources"][0]["source_id"] == "NVSS J1"


def test_variability_stacks_epochs(console):
    _, client = console
    d = client.get("/api/variability").json()
    assert d["n_epochs"] == 2
    assert d["n_sources"] == 5
    top = d["sources"][0]
    assert top["source_id"] == "S0"  # the boosted source is most variable
    assert top["n_epochs"] == 2
    assert top["v"] > d["sources"][1]["v"]
    assert len(top["flux_jy"]) == 2


def test_lightcurve_endpoint(console):
    _, client = console
    d = client.get("/api/lightcurve/S0").json()
    assert d["flux_jy"] == [0.5, 0.75]
    assert client.get("/api/lightcurve/NOPE").status_code == 404
