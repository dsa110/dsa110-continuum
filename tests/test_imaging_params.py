"""Tests for ImagingParams config object and lazy initialization."""

import pathlib

import pytest

# ---------------------------------------------------------------------------
# ImagingParams field tests (#4)
# ---------------------------------------------------------------------------

class TestImagingParamsFields:
    """Verify ImagingParams carries cleaning / resource fields end-to-end."""

    def _make_params(self, **kw):
        from dsa110_continuum.imaging.params import ImagingParams
        defaults = {"imagename": "test_output"}
        defaults.update(kw)
        return ImagingParams(**defaults)

    # ── default values match the old hardcoded WSClean values ──

    def test_default_auto_mask(self):
        p = self._make_params()
        assert p.auto_mask == 5.0

    def test_default_auto_threshold(self):
        p = self._make_params()
        assert p.auto_threshold == 1.0

    def test_default_mgain(self):
        p = self._make_params()
        assert p.mgain == 0.8

    def test_default_threads_none(self):
        p = self._make_params()
        assert p.threads is None

    def test_default_mem_gb_none(self):
        p = self._make_params()
        assert p.mem_gb is None

    # ── to_dict round-trips ──

    def test_to_dict_includes_cleaning_fields(self):
        p = self._make_params(auto_mask=4.0, auto_threshold=0.5, mgain=0.6)
        d = p.to_dict()
        assert d["auto_mask"] == 4.0
        assert d["auto_threshold"] == 0.5
        assert d["mgain"] == 0.6

    def test_to_dict_includes_resource_fields(self):
        p = self._make_params(threads=8, mem_gb=32)
        d = p.to_dict()
        assert d["threads"] == 8
        assert d["mem_gb"] == 32

    def test_to_dict_none_resources(self):
        d = self._make_params().to_dict()
        assert d["threads"] is None
        assert d["mem_gb"] is None

    # ── from_dict inverse ──

    def test_from_dict_round_trip(self):
        from dsa110_continuum.imaging.params import ImagingParams
        p = self._make_params(auto_mask=3.5, threads=4, mem_gb=16)
        p2 = ImagingParams.from_dict(p.to_dict())
        assert p2.auto_mask == 3.5
        assert p2.threads == 4
        assert p2.mem_gb == 16

    # ── with_overrides ──

    def test_with_overrides_cleaning(self):
        p = self._make_params()
        p2 = p.with_overrides(auto_mask=3.0, mgain=0.9)
        assert p2.auto_mask == 3.0
        assert p2.mgain == 0.9
        # originals unchanged
        assert p.auto_mask == 5.0
        assert p.mgain == 0.8

    # ── factory presets ──

    def test_for_survey_overrides(self):
        from dsa110_continuum.imaging.params import ImagingParams
        p = ImagingParams.for_survey("survey_img")
        assert p.auto_mask == 4.0
        assert p.auto_threshold == 0.5
        assert p.deconvolver == "multiscale"
        assert p.nterms == 2

    def test_for_standard_uses_defaults(self):
        from dsa110_continuum.imaging.params import ImagingParams
        p = ImagingParams.for_standard("std_img")
        assert p.auto_mask == 5.0
        assert p.auto_threshold == 1.0
        assert p.mgain == 0.8

    def test_for_development_uses_defaults(self):
        from dsa110_continuum.imaging.params import ImagingParams
        p = ImagingParams.for_development("dev_img")
        assert p.auto_mask == 5.0
        assert p.threads is None

    # ── validation ──

    def test_auto_mask_must_be_positive(self):
        with pytest.raises(ValueError, match="auto_mask"):
            self._make_params(auto_mask=0)

    def test_auto_threshold_must_be_positive(self):
        with pytest.raises(ValueError, match="auto_threshold"):
            self._make_params(auto_threshold=-1.0)

    def test_mgain_must_be_in_unit_interval(self):
        with pytest.raises(ValueError, match="mgain"):
            self._make_params(mgain=1.5)

    def test_mgain_must_be_positive(self):
        with pytest.raises(ValueError, match="mgain"):
            self._make_params(mgain=0)

    def test_threads_must_be_positive(self):
        with pytest.raises(ValueError, match="threads"):
            self._make_params(threads=0)

    def test_mem_gb_must_be_positive(self):
        with pytest.raises(ValueError, match="mem_gb"):
            self._make_params(mem_gb=0)

    def test_threads_none_is_valid(self):
        p = self._make_params(threads=None)
        assert p.threads is None

    def test_mem_gb_none_is_valid(self):
        p = self._make_params(mem_gb=None)
        assert p.mem_gb is None


# ---------------------------------------------------------------------------
# run_wsclean parameter flow (#4)
# ---------------------------------------------------------------------------

class TestRunWscleanParamFlow:
    """Verify that ImagingParams fields reach the WSClean command line."""

    def _get_run_wsclean_sig(self):
        """Return the run_wsclean function's parameter names."""
        import inspect

        from dsa110_continuum.imaging.cli_imaging import run_wsclean
        sig = inspect.signature(run_wsclean)
        return set(sig.parameters.keys())

    def test_auto_mask_in_signature(self):
        params = self._get_run_wsclean_sig()
        assert "auto_mask" in params

    def test_auto_threshold_in_signature(self):
        params = self._get_run_wsclean_sig()
        assert "auto_threshold" in params

    def test_mgain_in_signature(self):
        params = self._get_run_wsclean_sig()
        assert "mgain" in params

    def test_threads_in_signature(self):
        params = self._get_run_wsclean_sig()
        assert "threads" in params

    def test_mem_gb_in_signature(self):
        params = self._get_run_wsclean_sig()
        assert "mem_gb" in params

    def test_dead_params_removed(self):
        """use_gpu and auto_mask:bool backward-compat params are gone."""
        import inspect

        from dsa110_continuum.imaging.cli_imaging import run_wsclean
        sig = inspect.signature(run_wsclean)
        # auto_mask should be float, not bool
        assert sig.parameters["auto_mask"].annotation is not bool
        assert "use_gpu" not in sig.parameters

    def test_image_ms_accepts_new_params(self):
        """image_ms signature includes the new fields."""
        import inspect

        from dsa110_continuum.imaging.cli_imaging import image_ms
        sig = inspect.signature(image_ms)
        for name in ("auto_mask", "auto_threshold", "mgain", "threads", "mem_gb"):
            assert name in sig.parameters, f"{name} missing from image_ms signature"


class TestWscleanCommandLine:
    """Verify parameters deterministically reach the WSClean CLI command.

    These tests mock subprocess.run to capture the actual command list
    that run_wsclean would pass to WSClean, then assert on specific
    flag values.  This is the science-critical guarantee: what the user
    sets in ImagingParams is what WSClean receives.
    """

    def _capture_wsclean_cmd(self, monkeypatch, *, clean_env=True, **overrides):
        """Call run_wsclean with mocks, return the captured cmd list.

        Parameters
        ----------
        clean_env : bool
            If True (default), remove WSCLEAN_THREADS / WSCLEAN_ABS_MEM
            so tests are deterministic.  Set False for env-var-fallback tests.
        """
        import subprocess as _sp

        captured = {}

        def fake_run(cmd, **_kw):
            captured["cmd"] = cmd
            return _sp.CompletedProcess(cmd, 0)

        monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/wsclean")
        monkeypatch.setattr("subprocess.run", fake_run)
        monkeypatch.setattr(
            "dsa110_continuum.imaging.cli_imaging.get_gpu_config",
            lambda: type("C", (), {"has_gpu": False})(),
        )
        monkeypatch.setattr(
            "dsa110_continuum.imaging.cli_imaging.set_ms_telescope_name",
            lambda *a, **k: None,
            raising=False,
        )
        if clean_env:
            monkeypatch.delenv("WSCLEAN_THREADS", raising=False)
            monkeypatch.delenv("WSCLEAN_ABS_MEM", raising=False)

        from dsa110_continuum.imaging.cli_imaging import run_wsclean

        defaults = dict(
            ms_path="/tmp/fake.ms",
            imagename="/tmp/out",
            datacolumn="corrected",
            field="",
            imsize=2400,
            cell_arcsec=6.0,
            weighting="briggs",
            robust=0.5,
            specmode="mfs",
            deconvolver="hogbom",
            nterms=1,
            niter=1000,
            threshold="0.005Jy",
            pbcor=False,
            uvrange=">1klambda",
            pblimit=0.2,
            quality_tier="standard",
            gridder="wgridder",
        )
        defaults.update(overrides)
        try:
            run_wsclean(**defaults)
        except Exception:
            pass  # May fail on FITS stat; we only need the cmd
        return captured.get("cmd", [])

    def _flag_value(self, cmd, flag):
        """Return the value immediately after *flag* in *cmd*."""
        for i, tok in enumerate(cmd):
            if tok == flag and i + 1 < len(cmd):
                return cmd[i + 1]
        return None

    # ── auto_mask / auto_threshold / mgain ──

    def test_default_auto_mask_reaches_cli(self, monkeypatch):
        cmd = self._capture_wsclean_cmd(monkeypatch)
        assert self._flag_value(cmd, "-auto-mask") == "5.0"

    def test_default_auto_threshold_reaches_cli(self, monkeypatch):
        cmd = self._capture_wsclean_cmd(monkeypatch)
        assert self._flag_value(cmd, "-auto-threshold") == "1.0"

    def test_default_mgain_reaches_cli(self, monkeypatch):
        cmd = self._capture_wsclean_cmd(monkeypatch)
        assert self._flag_value(cmd, "-mgain") == "0.8"

    def test_survey_auto_mask_reaches_cli(self, monkeypatch):
        """for_survey() auto_mask=4.0 must NOT be silently overridden."""
        cmd = self._capture_wsclean_cmd(monkeypatch, auto_mask=4.0)
        assert self._flag_value(cmd, "-auto-mask") == "4.0"

    def test_survey_auto_threshold_reaches_cli(self, monkeypatch):
        cmd = self._capture_wsclean_cmd(monkeypatch, auto_threshold=0.5)
        assert self._flag_value(cmd, "-auto-threshold") == "0.5"

    def test_custom_mgain_reaches_cli(self, monkeypatch):
        cmd = self._capture_wsclean_cmd(monkeypatch, mgain=0.6)
        assert self._flag_value(cmd, "-mgain") == "0.6"

    # ── threads / mem_gb ──

    def test_explicit_threads_reaches_cli(self, monkeypatch):
        cmd = self._capture_wsclean_cmd(monkeypatch, threads=12)
        assert self._flag_value(cmd, "-j") == "12"

    def test_explicit_mem_gb_reaches_cli(self, monkeypatch):
        cmd = self._capture_wsclean_cmd(monkeypatch, mem_gb=48)
        assert self._flag_value(cmd, "-abs-mem") == "48"

    def test_default_threads_uses_cpu_count(self, monkeypatch):
        """When threads=None and no env var, falls back to cpu_count."""
        import multiprocessing
        cmd = self._capture_wsclean_cmd(monkeypatch)
        assert self._flag_value(cmd, "-j") == str(multiprocessing.cpu_count())

    def test_env_threads_fallback(self, monkeypatch):
        """When threads=None, env var WSCLEAN_THREADS is respected."""
        monkeypatch.setenv("WSCLEAN_THREADS", "7")
        cmd = self._capture_wsclean_cmd(monkeypatch, clean_env=False)
        assert self._flag_value(cmd, "-j") == "7"

    def test_explicit_threads_overrides_env(self, monkeypatch):
        """Explicit threads param takes precedence over env var."""
        monkeypatch.setenv("WSCLEAN_THREADS", "7")
        cmd = self._capture_wsclean_cmd(monkeypatch, clean_env=False, threads=16)
        assert self._flag_value(cmd, "-j") == "16"

    def test_explicit_mem_overrides_env(self, monkeypatch):
        monkeypatch.setenv("WSCLEAN_ABS_MEM", "99")
        cmd = self._capture_wsclean_cmd(monkeypatch, clean_env=False, mem_gb=24)
        assert self._flag_value(cmd, "-abs-mem") == "24"

    def test_tier_based_mem_when_no_override(self, monkeypatch):
        """Development tier defaults to 16 GB."""
        cmd = self._capture_wsclean_cmd(monkeypatch, quality_tier="development")
        assert self._flag_value(cmd, "-abs-mem") == "16"


# ---------------------------------------------------------------------------
# Imaging provenance sidecar tests (#6)
# ---------------------------------------------------------------------------

class TestImagingProvenance:
    """Verify the provenance JSON sidecar written by run_wsclean."""

    @pytest.fixture()
    def work_dir(self):
        """Create a temp directory that avoids the pytest tmp_path ownership issue."""
        import shutil
        import tempfile

        d = tempfile.mkdtemp(prefix="dsa110_prov_test_")
        yield pathlib.Path(d)
        shutil.rmtree(d, ignore_errors=True)

    def test_provenance_file_written(self, work_dir, monkeypatch):
        """run_wsclean writes a .provenance.json beside the output image."""
        from dsa110_continuum.imaging.cli_imaging import _write_imaging_provenance

        out = _write_imaging_provenance(
            imagename=str(work_dir / "tile_001"),
            ms_path="/stage/ms/2026-01-25.ms",
            imsize=2400,
            cell_arcsec=6.0,
            weighting="briggs",
            robust=0.5,
            specmode="mfs",
            deconvolver="hogbom",
            nterms=1,
            niter=1000,
            threshold="0.005Jy",
            auto_mask=5.0,
            auto_threshold=1.0,
            mgain=0.8,
            pbcor=True,
            gridder="wgridder",
            uvrange=">1klambda",
            quality_tier="standard",
            threads=40,
            mem_gb=64,
        )
        assert out is not None
        assert out.endswith(".provenance.json")
        assert pathlib.Path(out).exists()

    def test_provenance_values_match_effective_params(self, work_dir):
        """Sidecar values reflect the *effective* resolved parameters."""
        import json

        from dsa110_continuum.imaging.cli_imaging import _write_imaging_provenance

        path = _write_imaging_provenance(
            imagename=str(work_dir / "survey_003"),
            ms_path="/stage/ms/obs.ms",
            imsize=4800,
            cell_arcsec=3.0,
            weighting="briggs",
            robust=0.0,
            specmode="mfs",
            deconvolver="multiscale",
            nterms=2,
            niter=10000,
            threshold="0.05mJy",
            auto_mask=4.0,
            auto_threshold=0.5,
            mgain=0.8,
            pbcor=True,
            gridder="idg",
            uvrange=">1klambda",
            quality_tier="high_precision",
            threads=12,
            mem_gb=48,
        )
        with open(path) as f:
            prov = json.load(f)

        ip = prov["imaging_params"]
        assert ip["auto_mask"] == 4.0
        assert ip["auto_threshold"] == 0.5
        assert ip["deconvolver"] == "multiscale"
        assert ip["nterms"] == 2
        assert ip["niter"] == 10000
        assert ip["imsize"] == 4800
        assert ip["cell_arcsec"] == 3.0
        assert ip["gridder"] == "idg"
        assert ip["quality_tier"] == "high_precision"
        assert prov["resources"]["threads"] == 12
        assert prov["resources"]["mem_gb"] == 48
        assert prov["inputs"]["ms_path"] == "/stage/ms/obs.ms"
        assert prov["meta"]["pipeline"] == "dsa110-continuum"

    def test_provenance_round_trip_for_qa(self, work_dir):
        """Provenance JSON can be loaded and compared across epochs."""
        import json

        from dsa110_continuum.imaging.cli_imaging import _write_imaging_provenance

        common = dict(
            ms_path="/stage/ms/obs.ms",
            imsize=2400,
            cell_arcsec=6.0,
            weighting="briggs",
            robust=0.5,
            specmode="mfs",
            deconvolver="hogbom",
            nterms=1,
            niter=1000,
            threshold="0.005Jy",
            auto_mask=5.0,
            auto_threshold=1.0,
            mgain=0.8,
            pbcor=True,
            gridder="wgridder",
            uvrange=">1klambda",
            quality_tier="standard",
            threads=40,
            mem_gb=64,
        )
        p1 = _write_imaging_provenance(imagename=str(work_dir / "epoch1"), **common)
        p2 = _write_imaging_provenance(imagename=str(work_dir / "epoch2"), **common)

        with open(p1) as f:
            d1 = json.load(f)
        with open(p2) as f:
            d2 = json.load(f)

        # Imaging params must be identical across epochs with same config
        assert d1["imaging_params"] == d2["imaging_params"]
        assert d1["resources"] == d2["resources"]
        # Only meta.timestamp should differ
        assert d1["meta"]["timestamp"] != d2["meta"]["timestamp"]

    def test_provenance_contains_timestamp(self, work_dir):
        """Sidecar includes ISO-8601 UTC timestamp."""
        import json
        from datetime import datetime, timezone

        from dsa110_continuum.imaging.cli_imaging import _write_imaging_provenance

        before = datetime.now(timezone.utc)
        path = _write_imaging_provenance(
            imagename=str(work_dir / "ts_test"),
            ms_path="x.ms", imsize=512, cell_arcsec=6.0, weighting="briggs",
            robust=0.5, specmode="mfs", deconvolver="hogbom", nterms=1,
            niter=100, threshold="1Jy", auto_mask=5.0, auto_threshold=1.0,
            mgain=0.8, pbcor=False, gridder="wgridder", uvrange=">1klambda",
            quality_tier="development", threads=1, mem_gb=4,
        )
        after = datetime.now(timezone.utc)

        with open(path) as f:
            ts_str = json.load(f)["meta"]["timestamp"]
        ts = datetime.fromisoformat(ts_str)
        assert before <= ts <= after

    def test_provenance_path_in_imaging_result(self):
        """ImagingResult dataclass has provenance_path field."""
        from dsa110_continuum.imaging.cli_imaging import ImagingResult

        r = ImagingResult(image_path="test.fits", provenance_path="/tmp/test.provenance.json")
        assert r.provenance_path == "/tmp/test.provenance.json"

    def test_provenance_path_default_none(self):
        """ImagingResult.provenance_path defaults to None."""
        from dsa110_continuum.imaging.cli_imaging import ImagingResult

        r = ImagingResult(image_path="test.fits")
        assert r.provenance_path is None

    def test_run_wsclean_returns_provenance_path(self, work_dir, monkeypatch):
        """Full run_wsclean mock: ImagingResult.provenance_path is set."""
        import subprocess as _sp

        def fake_run(cmd, **_kw):
            return _sp.CompletedProcess(cmd, 0)

        monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/wsclean")
        monkeypatch.setattr("subprocess.run", fake_run)
        monkeypatch.setattr(
            "dsa110_continuum.imaging.cli_imaging.get_gpu_config",
            lambda: type("C", (), {"has_gpu": False})(),
        )
        monkeypatch.setattr(
            "dsa110_continuum.imaging.cli_imaging.set_ms_telescope_name",
            lambda *a, **k: None,
            raising=False,
        )
        monkeypatch.delenv("WSCLEAN_THREADS", raising=False)
        monkeypatch.delenv("WSCLEAN_ABS_MEM", raising=False)

        from dsa110_continuum.imaging.cli_imaging import run_wsclean

        result = run_wsclean(
            ms_path="/tmp/fake.ms",
            imagename=str(work_dir / "prov_test"),
            datacolumn="corrected",
            field="",
            imsize=2400,
            cell_arcsec=6.0,
            weighting="briggs",
            robust=0.5,
            specmode="mfs",
            deconvolver="hogbom",
            nterms=1,
            niter=1000,
            threshold="0.005Jy",
            pbcor=False,
            uvrange=">1klambda",
            pblimit=0.2,
            quality_tier="standard",
            gridder="wgridder",
        )
        assert result.provenance_path is not None
        assert result.provenance_path.endswith(".provenance.json")
        assert pathlib.Path(result.provenance_path).exists()

        # Verify sidecar content matches what run_wsclean used
        import json
        with open(result.provenance_path) as f:
            prov = json.load(f)
        assert prov["imaging_params"]["auto_mask"] == 5.0
        assert prov["imaging_params"]["niter"] == 1000
        assert prov["imaging_params"]["gridder"] == "wgridder"

    def test_sidecar_matches_effective_cmd_values(self, work_dir, monkeypatch):
        """Integration: every science-critical sidecar field matches run_wsclean's
        effective command-line values, including resolved threads/mem defaults."""
        import json
        import subprocess as _sp

        captured_cmd = {}

        def fake_run(cmd, **_kw):
            captured_cmd["cmd"] = cmd
            return _sp.CompletedProcess(cmd, 0)

        monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/wsclean")
        monkeypatch.setattr("subprocess.run", fake_run)
        monkeypatch.setattr(
            "dsa110_continuum.imaging.cli_imaging.get_gpu_config",
            lambda: type("C", (), {"has_gpu": False})(),
        )
        monkeypatch.setattr(
            "dsa110_continuum.imaging.cli_imaging.set_ms_telescope_name",
            lambda *a, **k: None,
            raising=False,
        )
        monkeypatch.delenv("WSCLEAN_THREADS", raising=False)
        monkeypatch.delenv("WSCLEAN_ABS_MEM", raising=False)

        from dsa110_continuum.imaging.cli_imaging import run_wsclean

        result = run_wsclean(
            ms_path="/tmp/fake.ms",
            imagename=str(work_dir / "integ"),
            datacolumn="corrected",
            field="",
            imsize=4800,
            cell_arcsec=3.0,
            weighting="briggs",
            robust=0.0,
            specmode="mfs",
            deconvolver="multiscale",
            nterms=2,
            niter=10000,
            threshold="0.05mJy",
            auto_mask=4.0,
            auto_threshold=0.5,
            mgain=0.6,
            pbcor=True,
            uvrange=">1klambda",
            pblimit=0.2,
            quality_tier="high_precision",
            gridder="wgridder",
            threads=16,
            mem_gb=48,
        )

        # --- Load sidecar ---
        with open(result.provenance_path) as f:
            prov = json.load(f)
        ip = prov["imaging_params"]
        res = prov["resources"]

        # --- Parse the same values from the captured WSClean cmd ---
        cmd = captured_cmd["cmd"]

        def flag_val(flag):
            for i, tok in enumerate(cmd):
                if tok == flag and i + 1 < len(cmd):
                    return cmd[i + 1]
            return None

        # Assert sidecar matches effective CLI for all 8 science-critical fields
        assert ip["auto_mask"] == 4.0
        assert flag_val("-auto-mask") == "4.0"

        assert ip["auto_threshold"] == 0.5
        assert flag_val("-auto-threshold") == "0.5"

        assert ip["mgain"] == 0.6
        assert flag_val("-mgain") == "0.6"

        assert res["threads"] == 16
        assert flag_val("-j") == "16"

        assert res["mem_gb"] == 48
        assert flag_val("-abs-mem") == "48"

        assert ip["imsize"] == 4800
        assert flag_val("-size") == "4800"

        assert ip["cell_arcsec"] == 3.0
        assert flag_val("-scale") == "3.000arcsec"

        assert ip["quality_tier"] == "high_precision"


# ---------------------------------------------------------------------------
# _lazy_init module tests (#5)
# ---------------------------------------------------------------------------

class TestLazyInit:
    """Test the deferred initialization guards."""

    def test_require_casa_is_idempotent(self):
        """Calling require_casa twice doesn't raise."""
        from dsa110_continuum._lazy_init import require_casa
        require_casa()
        require_casa()  # second call is a no-op

    def test_require_gpu_safety_is_idempotent(self):
        from dsa110_continuum._lazy_init import require_gpu_safety
        require_gpu_safety()
        require_gpu_safety()

    def test_require_headless_sets_env(self):
        import os

        from dsa110_continuum._lazy_init import require_headless
        require_headless()
        assert os.environ.get("QT_QPA_PLATFORM") == "offscreen"
        assert os.environ.get("CASA_NO_X") == "1"

    def test_require_headless_is_idempotent(self):
        from dsa110_continuum._lazy_init import require_headless
        require_headless()
        require_headless()

    def test_module_level_ensure_casa_path_removed(self):
        """No new-package file calls ensure_casa_path() at module level."""
        import ast
        from pathlib import Path

        pkg_dir = Path(__file__).parent.parent / "dsa110_continuum"
        violations = []

        for py in pkg_dir.rglob("*.py"):
            if py.name == "_lazy_init.py":
                continue
            if "__init__" in py.name:
                continue
            try:
                tree = ast.parse(py.read_text())
            except SyntaxError:
                continue

            for node in ast.iter_child_nodes(tree):
                # Look for module-level Expr nodes containing ensure_casa_path()
                if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
                    func = node.value.func
                    name = ""
                    if isinstance(func, ast.Name):
                        name = func.id
                    elif isinstance(func, ast.Attribute):
                        name = func.attr
                    if name == "ensure_casa_path":
                        violations.append(f"{py.relative_to(pkg_dir.parent)}:{node.lineno}")

        assert violations == [], (
            "Module-level ensure_casa_path() calls found:\n" + "\n".join(violations)
        )

    def test_module_level_initialize_gpu_safety_removed(self):
        """No new-package file calls initialize_gpu_safety() at module level."""
        import ast
        from pathlib import Path

        pkg_dir = Path(__file__).parent.parent / "dsa110_continuum"
        violations = []

        for py in pkg_dir.rglob("*.py"):
            if py.name == "_lazy_init.py":
                continue
            if "__init__" in py.name:
                continue
            try:
                tree = ast.parse(py.read_text())
            except SyntaxError:
                continue

            for node in ast.iter_child_nodes(tree):
                if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
                    func = node.value.func
                    name = ""
                    if isinstance(func, ast.Name):
                        name = func.id
                    elif isinstance(func, ast.Attribute):
                        name = func.attr
                    if name == "initialize_gpu_safety":
                        violations.append(f"{py.relative_to(pkg_dir.parent)}:{node.lineno}")

        assert violations == [], (
            "Module-level initialize_gpu_safety() calls found:\n" + "\n".join(violations)
        )
