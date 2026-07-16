from pathlib import Path


def test_vlass_resolver_falls_back_to_full_catalog(monkeypatch):
    from dsa110_continuum.calibration.catalogs import _resolve_vlass_catalog_path

    full_path = Path("/data/dsa110-contimg/state/catalogs/vlass_full.sqlite3")
    monkeypatch.setattr(Path, "exists", lambda path: path == full_path)

    assert _resolve_vlass_catalog_path(16.1) == full_path


def test_vlass_resolver_prefers_strip_catalog(monkeypatch):
    from dsa110_continuum.calibration.catalogs import _resolve_vlass_catalog_path

    strip_path = Path("/data/dsa110-contimg/state/catalogs/vlass_dec+16.1.sqlite3")
    full_path = Path("/data/dsa110-contimg/state/catalogs/vlass_full.sqlite3")
    monkeypatch.setattr(Path, "exists", lambda path: path in {strip_path, full_path})

    assert _resolve_vlass_catalog_path(16.1) == strip_path
