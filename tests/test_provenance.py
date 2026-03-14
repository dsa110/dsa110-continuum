"""Tests for pipeline provenance manifest and cal-type detection fix."""

import json
import os


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
