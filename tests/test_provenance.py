"""Tests for pipeline provenance manifest and cal-type detection fix."""

import json
import os

import pytest


def test_manifest_roundtrip(tmp_path):
    """Create, populate, save, reload — verify structure."""
    from dsa110_continuum.qa.provenance import RunManifest

    m = RunManifest.start("2026-02-12", "2026-01-25", argv=["batch_pipeline.py", "--date", "2026-02-12"])
    m.ms_files = ["/stage/ms/a.ms", "/stage/ms/b.ms"]
    m.gaincal_status = "ok"
    m.record_tile("/stage/ms/a.ms", "/stage/img/a.fits", "ok", 142.3)
    m.record_tile("/stage/ms/b.ms", None, "failed", 1800.0, error="timeout")
    m.record_epoch(0, {"n_tiles": 13, "status": "ok", "peak": 0.5, "rms": 0.001, "qa_result": "PASS"})
    m.finalize(300.5)

    out = m.save(str(tmp_path))
    assert os.path.exists(out)
    assert out.endswith("2026-02-12_manifest.json")

    with open(out) as f:
        data = json.load(f)

    assert data["date"] == "2026-02-12"
    assert data["cal_date"] == "2026-01-25"
    assert data["hostname"] != ""
    assert data["started_at"] != ""
    assert data["finished_at"] is not None
    assert data["wall_time_sec"] == 300.5
    assert len(data["tiles"]) == 2
    assert data["tiles"][0]["status"] == "ok"
    assert data["tiles"][1]["error"] == "timeout"
    assert len(data["epochs"]) == 1
    assert data["epochs"][0]["qa_result"] == "PASS"
    assert data["command_line"][0] == "batch_pipeline.py"


def test_manifest_missing_cal_table(tmp_path):
    """assess_cal_quality with nonexistent paths stores error, doesn't crash."""
    from dsa110_continuum.qa.provenance import RunManifest

    m = RunManifest.start("2026-02-12", "2026-01-25")
    m.assess_cal_quality("/nonexistent/path.b", "/nonexistent/path.g")

    # Should have entries for both, each with an extraction_error
    assert "bp" in m.cal_quality
    assert "g" in m.cal_quality
    bp_q = m.cal_quality["bp"]
    # Either has extraction_error (from compute_calibration_metrics) or error key
    has_error = "extraction_error" in bp_q or "error" in bp_q
    assert has_error


def test_cal_type_detection_dsa110_suffix():
    """Verify .b -> 'bp' and .g -> 'g' in compute_calibration_metrics."""
    from dsa110_continuum.calibration.qa import compute_calibration_metrics

    # These paths don't exist, so we get extraction_error, but cal_type should be set
    metrics_b = compute_calibration_metrics("/fake/2026-01-25T22:26:05_0~23.b")
    assert metrics_b.cal_type == "bp"

    metrics_g = compute_calibration_metrics("/fake/2026-01-25T22:26:05_0~23.g")
    assert metrics_g.cal_type == "g"

    # Original patterns still work
    metrics_bp = compute_calibration_metrics("/fake/cal_bp.tbl")
    assert metrics_bp.cal_type == "bp"

    metrics_gcal = compute_calibration_metrics("/fake/gpcal.tbl")
    assert metrics_gcal.cal_type == "g"


def test_manifest_gates_serialized(tmp_path):
    """Verify gates field appears in JSON output."""
    from dsa110_continuum.qa.provenance import RunManifest

    m = RunManifest.start("2026-02-12", "2026-01-25")
    m.gates.append({"gate": "cal_quality", "verdict": "WARN", "reasons": ["high scatter"]})
    m.gates.append({"gate": "archive", "verdict": "BLOCKED", "reason": "QA FAIL"})
    m.finalize(50.0)

    out = m.save(str(tmp_path))
    with open(out) as f:
        data = json.load(f)

    assert len(data["gates"]) == 2
    assert data["gates"][0]["gate"] == "cal_quality"
    assert data["gates"][1]["verdict"] == "BLOCKED"
    assert data["pipeline_verdict"] == "DEGRADED"


def test_manifest_gates_empty_clean(tmp_path):
    """No gates → gates is empty list and pipeline_verdict is CLEAN."""
    from dsa110_continuum.qa.provenance import RunManifest

    m = RunManifest.start("2026-02-12", "2026-01-25")
    m.finalize(10.0)

    out = m.save(str(tmp_path))
    with open(out) as f:
        data = json.load(f)

    assert data["gates"] == []
    assert data["pipeline_verdict"] == "CLEAN"


def test_manifest_save_creates_file(tmp_path):
    """Verify JSON written to correct path with valid content."""
    from dsa110_continuum.qa.provenance import RunManifest

    m = RunManifest.start("2026-03-01", "2026-01-25")
    m.finalize(10.0)

    # Save to a subdirectory that doesn't exist yet
    out_dir = str(tmp_path / "products" / "mosaics" / "2026-03-01")
    path = m.save(out_dir)

    assert os.path.isfile(path)
    with open(path) as f:
        data = json.load(f)
    assert data["date"] == "2026-03-01"
    assert data["wall_time_sec"] == 10.0


# ---------------------------------------------------------------------------
# Tile-granular retrieval
# ---------------------------------------------------------------------------


def _make_manifest(tmp_path, date="2026-02-12", cal_date="2026-01-25"):
    """Helper: build and save a manifest with two tiles and synthetic cal QA."""
    from dsa110_continuum.qa.provenance import RunManifest

    m = RunManifest.start(date, cal_date)
    m.bp_table = f"/stage/ms/{cal_date}T22:26:05_0~23.b"
    m.g_table = f"/stage/ms/{cal_date}T22:26:05_0~23.g"
    m.cal_quality = {
        "bp": {"flag_fraction": 0.05, "phase_scatter_deg": 2.1, "extraction_error": None},
        "g": {"flag_fraction": 0.08, "phase_scatter_deg": 18.3, "extraction_error": None},
    }
    m.record_tile(
        f"/stage/ms/{date}T21:00:00.ms",
        f"/stage/img/{date}T21:00:00-image-pb.fits",
        "ok",
        95.0,
    )
    m.record_tile(
        f"/stage/ms/{date}T21:05:00.ms",
        None,
        "failed",
        1800.0,
        error="timeout",
    )
    m.finalize(200.0)

    products_dir = str(tmp_path / "products" / "mosaics")
    m.save(os.path.join(products_dir, date))
    return m, products_dir


def test_load_manifest_roundtrip(tmp_path):
    """RunManifest.load() reconstructs a manifest saved by .save()."""
    from dsa110_continuum.qa.provenance import RunManifest, load_manifest

    original, products_dir = _make_manifest(tmp_path)
    loaded = load_manifest("2026-02-12", products_dir)

    assert isinstance(loaded, RunManifest)
    assert loaded.date == original.date
    assert loaded.cal_date == original.cal_date
    assert loaded.bp_table == original.bp_table
    assert loaded.cal_quality["bp"]["flag_fraction"] == pytest.approx(0.05)
    assert loaded.cal_quality["g"]["phase_scatter_deg"] == pytest.approx(18.3)
    assert len(loaded.tiles) == 2
    assert loaded.pipeline_verdict == "CLEAN"


def test_load_manifest_missing_raises(tmp_path):
    """load_manifest raises FileNotFoundError when manifest is absent."""
    from dsa110_continuum.qa.provenance import load_manifest

    with pytest.raises(FileNotFoundError, match="Manifest not found"):
        load_manifest("2099-01-01", str(tmp_path))


def test_get_tile_record_by_fits_path(tmp_path):
    """get_tile_record finds a tile by its FITS path."""
    original, _ = _make_manifest(tmp_path)

    fits_path = "/stage/img/2026-02-12T21:00:00-image-pb.fits"
    rec = original.get_tile_record(fits_path)

    assert rec is not None
    assert rec["status"] == "ok"
    assert rec["fits_path"] == fits_path
    assert rec["elapsed_sec"] == pytest.approx(95.0)


def test_get_tile_record_by_ms_path(tmp_path):
    """get_tile_record finds a tile by its MS path."""
    original, _ = _make_manifest(tmp_path)

    ms_path = "/stage/ms/2026-02-12T21:05:00.ms"
    rec = original.get_tile_record(ms_path)

    assert rec is not None
    assert rec["status"] == "failed"
    assert rec["error"] == "timeout"


def test_get_tile_record_not_found(tmp_path):
    """get_tile_record returns None for a path not in the manifest."""
    original, _ = _make_manifest(tmp_path)

    assert original.get_tile_record("/nonexistent/tile.fits") is None


def test_get_cal_qa_for_tile_found(tmp_path):
    """get_cal_qa_for_tile returns correct stats and tile record for a known tile."""
    from dsa110_continuum.qa.provenance import get_cal_qa_for_tile

    _make_manifest(tmp_path)
    fits_path = "/stage/img/2026-02-12T21:00:00-image-pb.fits"
    products_dir = str(tmp_path / "products" / "mosaics")

    qa = get_cal_qa_for_tile(fits_path, products_dir=products_dir)

    assert qa.date == "2026-02-12"
    assert qa.cal_date == "2026-01-25"
    assert qa.bp_flag_fraction == pytest.approx(0.05)
    assert qa.g_phase_scatter_deg == pytest.approx(18.3)
    assert qa.tile_status == "ok"
    assert qa.pipeline_verdict == "CLEAN"
    assert qa.tile_record is not None
    assert qa.tile_record["fits_path"] == fits_path


def test_get_cal_qa_for_tile_not_in_manifest(tmp_path):
    """get_cal_qa_for_tile returns None tile_record for an unknown tile."""
    from dsa110_continuum.qa.provenance import get_cal_qa_for_tile

    _make_manifest(tmp_path)
    products_dir = str(tmp_path / "products" / "mosaics")

    qa = get_cal_qa_for_tile(
        "/stage/img/2026-02-12T23:59:00-image-pb.fits",
        products_dir=products_dir,
    )

    assert qa.date == "2026-02-12"
    assert qa.tile_record is None
    assert qa.tile_status is None
    # cal stats are still available even without a matching tile record
    assert qa.bp_flag_fraction == pytest.approx(0.05)


def test_get_cal_qa_for_tile_ms_path(tmp_path):
    """get_cal_qa_for_tile accepts an MS path as the key."""
    from dsa110_continuum.qa.provenance import get_cal_qa_for_tile

    _make_manifest(tmp_path)
    products_dir = str(tmp_path / "products" / "mosaics")
    ms_path = "/stage/ms/2026-02-12T21:05:00.ms"

    qa = get_cal_qa_for_tile(ms_path, products_dir=products_dir)

    assert qa.tile_status == "failed"
    assert qa.tile_record["error"] == "timeout"


def test_get_cal_qa_for_tile_bad_path(tmp_path):
    """get_cal_qa_for_tile raises ValueError when no date can be parsed."""
    from dsa110_continuum.qa.provenance import get_cal_qa_for_tile

    with pytest.raises(ValueError, match="Cannot parse observation date"):
        get_cal_qa_for_tile("/some/path/without_a_date.fits", products_dir=str(tmp_path))


# ---------------------------------------------------------------------------
# Calibration selection provenance in manifest
# ---------------------------------------------------------------------------


def test_manifest_cal_selection_roundtrip(tmp_path):
    """cal_selection survives save/load cycle and is accessible in dict."""
    from dsa110_continuum.qa.provenance import RunManifest

    m = RunManifest.start("2026-03-16", "2026-03-16")
    m.cal_selection = {
        "selection_mode": "dec_aware",
        "obs_dec_deg_used": 16.1,
        "calibrator_name": "3C138",
        "calibrator_dec_deg": 16.64,
        "source": "generated",
    }
    m.finalize(10.0)

    out = m.save(str(tmp_path))
    with open(out) as f:
        data = json.load(f)

    assert data["cal_selection"]["calibrator_name"] == "3C138"
    assert data["cal_selection"]["selection_mode"] == "dec_aware"
    assert data["cal_selection"]["obs_dec_deg_used"] == 16.1

    # Round-trip via load
    loaded = RunManifest.load(out)
    assert loaded.cal_selection["calibrator_name"] == "3C138"
    assert loaded.cal_selection["source"] == "generated"


def test_manifest_cal_selection_empty_default(tmp_path):
    """cal_selection defaults to empty dict — no breakage for old manifests."""
    from dsa110_continuum.qa.provenance import RunManifest

    m = RunManifest.start("2026-03-16", "2026-03-16")
    m.finalize(5.0)

    out = m.save(str(tmp_path))
    with open(out) as f:
        data = json.load(f)

    assert data["cal_selection"] == {}

    loaded = RunManifest.load(out)
    assert loaded.cal_selection == {}


def test_tile_retrieval_with_cal_selection(tmp_path):
    """get_cal_qa_for_tile still works after adding cal_selection to manifest."""
    from dsa110_continuum.qa.provenance import RunManifest, get_cal_qa_for_tile

    m = RunManifest.start("2026-03-16", "2026-03-16")
    m.bp_table = "/stage/ms/2026-03-16T05:21:10_0~23.b"
    m.g_table = "/stage/ms/2026-03-16T05:21:10_0~23.g"
    m.cal_quality = {
        "bp": {"flag_fraction": 0.03, "extraction_error": None},
        "g": {"flag_fraction": 0.05, "phase_scatter_deg": 12.0, "extraction_error": None},
    }
    m.cal_selection = {"calibrator_name": "3C138", "source": "generated"}
    m.record_tile(
        "/stage/ms/2026-03-16T04:00:00.ms",
        "/stage/img/2026-03-16T04:00:00-image-pb.fits",
        "ok",
        100.0,
    )
    m.finalize(50.0)

    products_dir = str(tmp_path / "products" / "mosaics")
    m.save(os.path.join(products_dir, "2026-03-16"))

    qa = get_cal_qa_for_tile(
        "/stage/img/2026-03-16T04:00:00-image-pb.fits",
        products_dir=products_dir,
    )
    assert qa.tile_status == "ok"
    assert qa.bp_flag_fraction == pytest.approx(0.03)
