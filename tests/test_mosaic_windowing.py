"""Tests for science-window mosaicking in mosaic_day.py."""

import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from mosaic_day import generate_windows, validate_window_params


class TestGenerateWindows:
    """generate_windows() produces correct sliding windows over tiles."""

    def test_basic_20_tiles(self):
        tiles = [f"t{i:02d}" for i in range(20)]
        windows = generate_windows(tiles, window_tiles=12, stride_tiles=6)
        assert len(windows) == 3
        assert len(windows[0]) == 12
        assert len(windows[1]) == 12
        assert len(windows[2]) == 8
        assert windows[0][0] == "t00"
        assert windows[1][0] == "t06"
        assert windows[2][0] == "t12"

    def test_exact_fit_one_window(self):
        tiles = [f"t{i:02d}" for i in range(12)]
        windows = generate_windows(tiles, window_tiles=12, stride_tiles=6)
        assert len(windows) == 1
        assert len(windows[0]) == 12

    def test_fewer_tiles_than_window(self):
        tiles = [f"t{i}" for i in range(5)]
        windows = generate_windows(tiles, window_tiles=12, stride_tiles=6)
        assert len(windows) == 1
        assert len(windows[0]) == 5

    def test_no_overlap_stride_eq_window(self):
        tiles = [f"t{i:02d}" for i in range(24)]
        windows = generate_windows(tiles, window_tiles=12, stride_tiles=12)
        assert len(windows) == 2
        assert windows[0][-1] == "t11"
        assert windows[1][0] == "t12"

    def test_empty_returns_empty(self):
        assert generate_windows([], window_tiles=12, stride_tiles=6) == []

    def test_single_tile_returns_empty(self):
        assert generate_windows(["t0"], window_tiles=12, stride_tiles=6) == []

    def test_two_tiles_minimum_window(self):
        windows = generate_windows(["a", "b"], window_tiles=12, stride_tiles=6)
        assert len(windows) == 1
        assert windows[0] == ["a", "b"]

    def test_trailing_single_tile_dropped(self):
        tiles = [f"t{i:02d}" for i in range(13)]
        windows = generate_windows(tiles, window_tiles=12, stride_tiles=12)
        # Window 0: [t00..t11] (12 tiles)
        # Window 1: [t12] (1 tile) — dropped (< 2)
        assert len(windows) == 1

    def test_all_tiles_covered(self):
        tiles = [f"t{i:02d}" for i in range(15)]
        windows = generate_windows(tiles, window_tiles=12, stride_tiles=6)
        covered = set()
        for w in windows:
            covered.update(w)
        assert covered == set(tiles)

    def test_order_preserved_within_windows(self):
        tiles = [f"t{i:02d}" for i in range(20)]
        windows = generate_windows(tiles, window_tiles=12, stride_tiles=6)
        for w in windows:
            indices = [int(t[1:]) for t in w]
            assert indices == sorted(indices)

    def test_stride_1_maximum_overlap(self):
        tiles = [f"t{i}" for i in range(5)]
        windows = generate_windows(tiles, window_tiles=3, stride_tiles=1)
        assert len(windows) == 3
        assert windows[0] == ["t0", "t1", "t2"]
        assert windows[1] == ["t1", "t2", "t3"]
        assert windows[2] == ["t2", "t3", "t4"]


class TestValidateWindowParams:
    """validate_window_params() catches invalid CLI inputs."""

    def test_valid_default(self):
        assert validate_window_params(12, 6) == []

    def test_valid_minimum(self):
        assert validate_window_params(2, 1) == []

    def test_valid_stride_eq_window(self):
        assert validate_window_params(12, 12) == []

    def test_window_below_minimum(self):
        errors = validate_window_params(1, 1)
        assert any("--window-tiles" in e for e in errors)

    def test_stride_below_minimum(self):
        errors = validate_window_params(12, 0)
        assert any("--stride-tiles" in e for e in errors)

    def test_stride_exceeds_window(self):
        errors = validate_window_params(6, 12)
        assert any(">" in e or "must be <=" in e for e in errors)

    def test_multiple_violations(self):
        errors = validate_window_params(1, 0)
        assert len(errors) >= 2

    def test_window_0_stride_0(self):
        errors = validate_window_params(0, 0)
        assert len(errors) >= 2


# ── Helpers for main()-level integration tests ───────────────────────────────

def _make_tmpdir():
    """Create a temp dir under /tmp (avoids broken $TMPDIR ownership)."""
    return tempfile.mkdtemp(dir="/tmp", prefix="test_mosaic_day_")


def _stub_main(monkeypatch, tmpdir, extra_argv=None):
    """Wire up monkeypatches so main() runs without CASA/WSClean/filesystem.

    Returns (written_paths, move_calls) for assertion.
    """
    import mosaic_day as md

    date = "2026-01-25"
    image_dir = os.path.join(tmpdir, "images")
    os.makedirs(image_dir, exist_ok=True)
    ms_dir = os.path.join(tmpdir, "ms")
    os.makedirs(ms_dir, exist_ok=True)

    fake_cfg = md.TileConfig(
        date=date,
        ms_dir=ms_dir,
        image_dir=image_dir,
        mosaic_out=os.path.join(image_dir, "full_mosaic.fits"),
        products_dir=os.path.join(tmpdir, "products"),
        bp_table=os.path.join(tmpdir, "cal.b"),
        g_table=os.path.join(tmpdir, "cal.g"),
    )
    # Create cal-table sentinels so validation passes
    Path(fake_cfg.bp_table).touch()
    Path(fake_cfg.g_table).touch()

    monkeypatch.setattr(md.TileConfig, "build", staticmethod(lambda **kw: fake_cfg))

    # 4 fake MSes → 4 fake tile images
    fake_ms = [os.path.join(ms_dir, f"{date}T{h:02d}:00:00.ms") for h in range(4)]
    tile_fits = [os.path.join(image_dir, f"tile_{i}.fits") for i in range(4)]

    monkeypatch.setattr(md, "find_valid_ms", lambda cfg: fake_ms)
    monkeypatch.setattr(
        md,
        "process_ms",
        lambda ms_path, cfg, keep_intermediates=False, force_recal=False: md.TileResult(
            "imaged", fits_path=tile_fits[fake_ms.index(ms_path)]
        ),
    )
    monkeypatch.setattr(md, "group_tiles_by_ra", lambda paths, gap_deg=10.0: [paths])

    fake_wcs = WCS(naxis=2)
    fake_wcs.wcs.crpix = [5, 5]
    fake_wcs.wcs.cdelt = [-0.001, 0.001]
    fake_wcs.wcs.crval = [180.0, 45.0]
    fake_wcs.wcs.ctype = ["RA---SIN", "DEC--SIN"]
    monkeypatch.setattr(
        md, "build_common_wcs", lambda paths, margin_deg=0.5: (fake_wcs, 10, 10)
    )
    monkeypatch.setattr(md, "coadd_tiles", lambda *a, **kw: np.zeros((10, 10)))

    written_paths: list[str] = []

    def _fake_write(mosaic, out_wcs, fits_paths, output_path, date=""):
        written_paths.append(output_path)
        hdu = fits.PrimaryHDU(data=np.zeros((10, 10), dtype=np.float32))
        hdu.writeto(output_path, overwrite=True)
        return output_path

    monkeypatch.setattr(md, "write_mosaic", _fake_write)
    monkeypatch.setattr(md, "check_mosaic_quality", lambda path: True)

    move_calls: list[dict] = []

    def _tracking_move(cfg, *, full_day):
        move_calls.append({"full_day": full_day})

    monkeypatch.setattr(md, "_move_mosaic_to_products", _tracking_move)

    argv = ["mosaic_day.py", "--date", date]
    if extra_argv:
        argv.extend(extra_argv)
    monkeypatch.setattr(sys, "argv", argv)

    return written_paths, move_calls


class TestMainScienceWindowBranch:
    """main() without --full-day follows the science-window code path."""

    def test_default_produces_windowed_names(self, monkeypatch):
        tmpdir = _make_tmpdir()
        try:
            written, _ = _stub_main(monkeypatch, tmpdir)
            import mosaic_day as md

            md.main()
            assert len(written) >= 1
            for p in written:
                name = Path(p).name
                assert "full_mosaic" not in name, f"science mode wrote full-day name: {name}"
                assert "_w" in name, f"science mode missing window index: {name}"
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_default_window_naming_format(self, monkeypatch):
        tmpdir = _make_tmpdir()
        try:
            written, _ = _stub_main(monkeypatch, tmpdir)
            import mosaic_day as md

            md.main()
            # With 4 tiles, window=12, stride=6 → 1 window (4 < 12)
            assert len(written) == 1
            assert Path(written[0]).name == "2026-01-25_w00_mosaic.fits"
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_move_called_with_full_day_false(self, monkeypatch):
        tmpdir = _make_tmpdir()
        try:
            _, move_calls = _stub_main(monkeypatch, tmpdir)
            import mosaic_day as md

            md.main()
            assert len(move_calls) == 1
            assert move_calls[0]["full_day"] is False
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)


class TestMainFullDayBranch:
    """main() with --full-day follows the legacy full-day code path."""

    def test_full_day_produces_full_mosaic_name(self, monkeypatch):
        tmpdir = _make_tmpdir()
        try:
            written, _ = _stub_main(monkeypatch, tmpdir, extra_argv=["--full-day"])
            import mosaic_day as md

            md.main()
            assert len(written) >= 1
            for p in written:
                name = Path(p).name
                assert "full_mosaic" in name, f"full-day mode missing full_mosaic: {name}"
                assert "_w" not in name, f"full-day mode has window index: {name}"
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_move_called_with_full_day_true(self, monkeypatch):
        tmpdir = _make_tmpdir()
        try:
            _, move_calls = _stub_main(monkeypatch, tmpdir, extra_argv=["--full-day"])
            import mosaic_day as md

            md.main()
            assert len(move_calls) == 1
            assert move_calls[0]["full_day"] is True
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)


class TestMoveModeSeparation:
    """_move_mosaic_to_products only archives files matching the active mode."""

    def test_science_mode_ignores_stale_full_mosaic(self):
        tmpdir = _make_tmpdir()
        try:
            import mosaic_day as md

            image_dir = os.path.join(tmpdir, "images")
            products_dir = os.path.join(tmpdir, "products")
            os.makedirs(image_dir)

            # Plant both stale full-day AND fresh windowed artifacts
            Path(os.path.join(image_dir, "full_mosaic.fits")).touch()
            Path(os.path.join(image_dir, "2026-01-25_w00_mosaic.fits")).touch()

            cfg = md.TileConfig(
                date="2026-01-25",
                ms_dir=tmpdir,
                image_dir=image_dir,
                mosaic_out=os.path.join(image_dir, "full_mosaic.fits"),
                products_dir=products_dir,
                bp_table="x.b",
                g_table="x.g",
            )

            md._move_mosaic_to_products(cfg, full_day=False)

            # Products dir was created and contains ONLY windowed artifact
            assert os.path.isdir(products_dir)
            archived = os.listdir(products_dir)
            assert "2026-01-25_w00_mosaic.fits" in archived
            assert "full_mosaic.fits" not in archived
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_full_day_mode_ignores_windowed_artifacts(self):
        tmpdir = _make_tmpdir()
        try:
            import mosaic_day as md

            image_dir = os.path.join(tmpdir, "images")
            products_dir = os.path.join(tmpdir, "products")
            os.makedirs(image_dir)

            Path(os.path.join(image_dir, "full_mosaic.fits")).touch()
            Path(os.path.join(image_dir, "2026-01-25_w00_mosaic.fits")).touch()

            cfg = md.TileConfig(
                date="2026-01-25",
                ms_dir=tmpdir,
                image_dir=image_dir,
                mosaic_out=os.path.join(image_dir, "full_mosaic.fits"),
                products_dir=products_dir,
                bp_table="x.b",
                g_table="x.g",
            )

            md._move_mosaic_to_products(cfg, full_day=True)

            assert os.path.isdir(products_dir)
            archived = os.listdir(products_dir)
            assert "full_mosaic.fits" in archived
            assert "2026-01-25_w00_mosaic.fits" not in archived
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)
