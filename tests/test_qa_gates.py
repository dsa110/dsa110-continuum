"""Tests for pipeline QA gates."""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
from astropy.io import fits


# ── Cal quality gate tests ───────────────────────────────────────────────────

def _make_manifest(phase_scatter=10.0, flag_frac_bp=0.1, flag_frac_g=0.1):
    """Create a RunManifest with pre-populated cal_quality."""
    from dsa110_continuum.qa.provenance import RunManifest
    m = RunManifest.start("2026-02-12", "2026-01-25")
    m.cal_quality = {
        "bp": {"flag_fraction": flag_frac_bp, "phase_scatter_deg": 5.0},
        "g": {"flag_fraction": flag_frac_g, "phase_scatter_deg": phase_scatter},
    }
    return m


def test_cal_gate_cross_date_high_scatter():
    """Cross-date G table with high phase scatter triggers WARN gate."""
    # Import the function from the script
    sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
    from batch_pipeline import check_cal_gate

    m = _make_manifest(phase_scatter=45.0)
    check_cal_gate(m, cal_date="2026-01-25", date="2026-02-12", strict=False)

    assert len(m.gates) == 1
    assert m.gates[0]["gate"] == "cal_quality"
    assert m.gates[0]["verdict"] == "WARN"
    assert any("phase scatter" in r.lower() for r in m.gates[0]["reasons"])


def test_cal_gate_same_date_no_trigger():
    """Same-date G table with high phase scatter does NOT trigger gate."""
    sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
    from batch_pipeline import check_cal_gate

    m = _make_manifest(phase_scatter=45.0)
    check_cal_gate(m, cal_date="2026-02-12", date="2026-02-12", strict=False)

    assert len(m.gates) == 0


def test_cal_gate_high_flag_fraction():
    """BP or G table with >50% flags triggers WARN gate."""
    sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
    from batch_pipeline import check_cal_gate

    m = _make_manifest(flag_frac_bp=0.7)
    check_cal_gate(m, cal_date="2026-02-12", date="2026-02-12", strict=False)

    assert len(m.gates) == 1
    assert any("flagged" in r for r in m.gates[0]["reasons"])


def test_cal_gate_strict_exits():
    """With strict=True and a gate trigger, sys.exit(1) is called."""
    sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
    from batch_pipeline import check_cal_gate

    m = _make_manifest(phase_scatter=45.0)
    with pytest.raises(SystemExit) as exc_info:
        check_cal_gate(m, cal_date="2026-01-25", date="2026-02-12", strict=True)
    assert exc_info.value.code == 1


# ── Tile image validation tests ──────────────────────────────────────────────

def test_tile_validation_rejects_zeros(tmp_path):
    """All-zero FITS image is rejected by validate_image_quality."""
    from dsa110_continuum.validation.image_validator import validate_image_quality

    # Create a minimal FITS file with all zeros
    data = np.zeros((100, 100), dtype=np.float32)
    hdr = fits.Header()
    hdr["CRPIX1"] = 50
    hdr["CRPIX2"] = 50
    hdr["CRVAL1"] = 180.0
    hdr["CRVAL2"] = 30.0
    hdr["CDELT1"] = -0.001
    hdr["CDELT2"] = 0.001
    hdr["CTYPE1"] = "RA---SIN"
    hdr["CTYPE2"] = "DEC--SIN"
    fpath = tmp_path / "zeros.fits"
    fits.PrimaryHDU(data=data, header=hdr).writeto(fpath)

    ok, errors = validate_image_quality(fpath, min_snr=3.0, max_flagged_fraction=0.5)
    assert not ok
    assert any("all zeros" in e.lower() for e in errors)


def test_tile_validation_passes_good_image(tmp_path):
    """FITS with signal + noise passes validate_image_quality."""
    from dsa110_continuum.validation.image_validator import validate_image_quality

    rng = np.random.default_rng(42)
    data = rng.normal(0, 0.001, (100, 100)).astype(np.float32)
    data[50, 50] = 0.1  # bright source

    hdr = fits.Header()
    hdr["CRPIX1"] = 50
    hdr["CRPIX2"] = 50
    hdr["CRVAL1"] = 180.0
    hdr["CRVAL2"] = 30.0
    hdr["CDELT1"] = -0.001
    hdr["CDELT2"] = 0.001
    hdr["CTYPE1"] = "RA---SIN"
    hdr["CTYPE2"] = "DEC--SIN"
    fpath = tmp_path / "good.fits"
    fits.PrimaryHDU(data=data, header=hdr).writeto(fpath)

    ok, errors = validate_image_quality(fpath, min_snr=3.0, max_flagged_fraction=0.5)
    assert ok
    assert errors == []


# ── needs_calibration exception propagation ──────────────────────────────────

def test_needs_calibration_propagates_type_error():
    """TypeError should NOT be caught by needs_calibration (only OSError/RuntimeError)."""
    sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
    import importlib
    import mosaic_day
    importlib.reload(mosaic_day)

    # Trigger a TypeError by passing a non-string path
    with pytest.raises(TypeError):
        mosaic_day.needs_calibration(12345)


# ── Manifest pipeline_verdict tests ──────────────────────────────────────────

def test_manifest_pipeline_verdict_clean():
    """No gates triggered → pipeline_verdict is CLEAN."""
    from dsa110_continuum.qa.provenance import RunManifest

    m = RunManifest.start("2026-02-12", "2026-01-25")
    m.finalize(100.0)
    assert m.pipeline_verdict == "CLEAN"


def test_manifest_pipeline_verdict_degraded():
    """Gates present → pipeline_verdict is DEGRADED."""
    from dsa110_continuum.qa.provenance import RunManifest

    m = RunManifest.start("2026-02-12", "2026-01-25")
    m.gates.append({"gate": "cal_quality", "verdict": "WARN", "reasons": ["test"]})
    m.finalize(100.0)
    assert m.pipeline_verdict == "DEGRADED"
