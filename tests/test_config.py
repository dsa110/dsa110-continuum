"""
Tests for centralized path configuration (Task 13).

Tests cover:
  1. PathConfig defaults
  2. PathConfig env-var overrides
  3. Derived properties (state_dir, db_dir, image_dir, products_dir)
  4. get_env_path helper
  5. Immutability (frozen dataclass)
  6. ensure.py uses central config defaults
  7. Module-level singleton
"""
from __future__ import annotations

import os
import pytest
from pathlib import Path

from dsa110_continuum.config import PathConfig, get_env_path, paths


# ══════════════════════════════════════════════════════════════════════════════
# 1. Default values
# ══════════════════════════════════════════════════════════════════════════════

class TestPathConfigDefaults:

    def test_ms_dir_default(self):
        cfg = PathConfig()
        assert str(cfg.ms_dir) == os.environ.get("DSA110_MS_DIR", "/stage/dsa110-contimg/ms")

    def test_incoming_dir_default(self):
        cfg = PathConfig()
        assert str(cfg.incoming_dir) == os.environ.get("DSA110_INCOMING_DIR", "/data/incoming")

    def test_pipeline_db_default_contains_sqlite(self):
        cfg = PathConfig()
        assert ".sqlite3" in str(cfg.pipeline_db)

    def test_catalog_dir_default_contains_catalogs(self):
        cfg = PathConfig()
        assert "catalogs" in str(cfg.catalog_dir)

    def test_all_paths_are_Path_objects(self):
        cfg = PathConfig()
        for attr in ("base_dir", "ms_dir", "stage_image_base",
                     "incoming_dir", "pipeline_db", "catalog_dir",
                     "products_base", "vla_cal_db", "upper_limits_db"):
            assert isinstance(getattr(cfg, attr), Path), f"{attr} is not a Path"

    def test_vla_cal_db_ends_with_sqlite3(self):
        cfg = PathConfig()
        assert str(cfg.vla_cal_db).endswith(".sqlite3")

    def test_upper_limits_db_ends_with_db(self):
        cfg = PathConfig()
        assert str(cfg.upper_limits_db).endswith(".db")


# ══════════════════════════════════════════════════════════════════════════════
# 2. Env-var overrides
# ══════════════════════════════════════════════════════════════════════════════

class TestPathConfigEnvOverrides:

    def _with_env(self, env: dict, fn):
        """Run fn() with temporary environment variables set."""
        old = {k: os.environ.get(k) for k in env}
        try:
            for k, v in env.items():
                os.environ[k] = v
            return fn()
        finally:
            for k, v in old.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v

    def test_ms_dir_from_env(self):
        result = self._with_env(
            {"DSA110_MS_DIR": "/tmp/custom_ms"},
            lambda: PathConfig().ms_dir,
        )
        assert str(result) == "/tmp/custom_ms"

    def test_incoming_dir_from_env(self):
        result = self._with_env(
            {"DSA110_INCOMING_DIR": "/tmp/hdf5_in"},
            lambda: PathConfig().incoming_dir,
        )
        assert str(result) == "/tmp/hdf5_in"

    def test_pipeline_db_from_env(self):
        result = self._with_env(
            {"PIPELINE_DB": "/tmp/test_pipeline.sqlite3"},
            lambda: PathConfig().pipeline_db,
        )
        assert str(result) == "/tmp/test_pipeline.sqlite3"

    def test_catalog_dir_from_env(self):
        result = self._with_env(
            {"DSA110_CATALOG_DIR": "/tmp/catalogs"},
            lambda: PathConfig().catalog_dir,
        )
        assert str(result) == "/tmp/catalogs"

    def test_products_base_from_env(self):
        result = self._with_env(
            {"DSA110_PRODUCTS_BASE": "/tmp/products"},
            lambda: PathConfig().products_base,
        )
        assert str(result) == "/tmp/products"

    def test_contimg_base_dir_propagates(self):
        """Setting CONTIMG_BASE_DIR should affect derived paths."""
        result = self._with_env(
            {"CONTIMG_BASE_DIR": "/tmp/mybase",
             # Clear derived-path overrides so they derive from base
             "PIPELINE_DB": "",
             "DSA110_CATALOG_DIR": ""},
            lambda: PathConfig(),
        )
        # base_dir should match
        assert str(result.base_dir) == "/tmp/mybase"

    def test_manual_override_via_constructor(self):
        """PathConfig fields can be overridden at construction time."""
        cfg = PathConfig(ms_dir=Path("/custom/ms"))
        assert str(cfg.ms_dir) == "/custom/ms"
        # Other fields still use defaults/env
        assert isinstance(cfg.pipeline_db, Path)

    def test_vla_cal_db_from_env(self):
        result = self._with_env(
            {"DSA110_VLA_CAL_DB": "/tmp/vla.sqlite3"},
            lambda: PathConfig().vla_cal_db,
        )
        assert str(result) == "/tmp/vla.sqlite3"


# ══════════════════════════════════════════════════════════════════════════════
# 3. Derived properties
# ══════════════════════════════════════════════════════════════════════════════

class TestPathConfigDerived:

    def test_state_dir_is_base_slash_state(self):
        cfg = PathConfig(base_dir=Path("/my/base"))
        assert cfg.state_dir == Path("/my/base/state")

    def test_db_dir_is_base_slash_state_db(self):
        cfg = PathConfig(base_dir=Path("/my/base"))
        assert cfg.db_dir == Path("/my/base/state/db")

    def test_image_dir_uses_date(self):
        cfg = PathConfig(stage_image_base=Path("/stage/images"))
        result = cfg.image_dir("2026-01-25")
        assert result == Path("/stage/images/mosaic_2026-01-25")

    def test_products_dir_uses_date(self):
        cfg = PathConfig(products_base=Path("/products"))
        result = cfg.products_dir("2026-01-25")
        assert result == Path("/products/2026-01-25")

    def test_repr_contains_key_fields(self):
        cfg = PathConfig()
        r = repr(cfg)
        assert "ms_dir" in r
        assert "pipeline_db" in r


# ══════════════════════════════════════════════════════════════════════════════
# 4. get_env_path helper
# ══════════════════════════════════════════════════════════════════════════════

class TestGetEnvPath:

    def test_returns_path_from_env(self, monkeypatch):
        monkeypatch.setenv("_TEST_PATH_VAR", "/tmp/x")
        assert get_env_path("_TEST_PATH_VAR", "/default") == Path("/tmp/x")

    def test_returns_default_when_not_set(self, monkeypatch):
        monkeypatch.delenv("_TEST_PATH_VAR", raising=False)
        assert get_env_path("_TEST_PATH_VAR", "/default") == Path("/default")

    def test_returns_path_object(self, monkeypatch):
        monkeypatch.delenv("_TEST_PATH_VAR", raising=False)
        result = get_env_path("_TEST_PATH_VAR", "/default")
        assert isinstance(result, Path)

    def test_empty_env_var_gives_empty_path(self, monkeypatch):
        monkeypatch.setenv("_TEST_PATH_VAR", "")
        result = get_env_path("_TEST_PATH_VAR", "/default")
        # Empty string → Path("") which is valid (current directory)
        assert isinstance(result, Path)


# ══════════════════════════════════════════════════════════════════════════════
# 5. Immutability
# ══════════════════════════════════════════════════════════════════════════════

class TestPathConfigImmutability:

    def test_cannot_assign_field(self):
        cfg = PathConfig()
        with pytest.raises((AttributeError, TypeError)):
            cfg.ms_dir = Path("/new/path")

    def test_two_default_configs_are_equal(self):
        cfg1 = PathConfig()
        cfg2 = PathConfig()
        assert cfg1 == cfg2


# ══════════════════════════════════════════════════════════════════════════════
# 6. ensure.py uses central config defaults
# ══════════════════════════════════════════════════════════════════════════════

class TestEnsureUsesConfig:

    def test_ensure_defaults_match_config(self):
        """ensure.py constants should equal PathConfig defaults (or env vars)."""
        from dsa110_continuum.calibration.ensure import (
            DEFAULT_MS_DIR,
            DEFAULT_INPUT_DIR,
            DEFAULT_DB_PATH,
        )
        cfg = PathConfig()
        assert DEFAULT_MS_DIR == str(cfg.ms_dir)
        assert DEFAULT_INPUT_DIR == str(cfg.incoming_dir)
        assert DEFAULT_DB_PATH == str(cfg.pipeline_db)

    def test_ensure_default_vla_cal_db_matches_config(self):
        from dsa110_continuum.calibration.ensure import DEFAULT_VLA_CAL_DB
        cfg = PathConfig()
        assert DEFAULT_VLA_CAL_DB == str(cfg.vla_cal_db)


# ══════════════════════════════════════════════════════════════════════════════
# 7. Module-level singleton
# ══════════════════════════════════════════════════════════════════════════════

class TestModuleSingleton:

    def test_paths_is_path_config(self):
        assert isinstance(paths, PathConfig)

    def test_paths_ms_dir_is_path(self):
        assert isinstance(paths.ms_dir, Path)

    def test_paths_pipeline_db_is_path(self):
        assert isinstance(paths.pipeline_db, Path)
