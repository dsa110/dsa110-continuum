"""
DSA-110 Continuum Pipeline — Centralized Path & Configuration
==============================================================

All data paths used by the pipeline are resolved here, with environment
variable overrides so the same codebase runs on H17 (real data) and in CI /
cloud environments (simulated data).

Environment variables
---------------------
``CONTIMG_BASE_DIR``
    Root of all pipeline outputs on H17.  Default: ``/data/dsa110-contimg``.
    All other ``/data/…`` paths below are derived from this.

``DSA110_MS_DIR``
    Directory containing raw Measurement Sets.  Default:
    ``/stage/dsa110-contimg/ms``.

``DSA110_STAGE_IMAGE_BASE``
    Staging directory for per-date WSClean images.  Default:
    ``/stage/dsa110-contimg/images``.

``DSA110_INCOMING_DIR``
    Directory where HDF5 visibility data arrives from the correlator.
    Default: ``/data/incoming``.

``PIPELINE_DB``
    SQLite pipeline state database.  Default:
    ``{CONTIMG_BASE_DIR}/state/db/pipeline.sqlite3``.

``DSA110_CATALOG_DIR``
    Directory containing reference catalogs (NVSS, FIRST, VLASS, etc.).
    Default: ``{CONTIMG_BASE_DIR}/state/catalogs``.

``DSA110_PRODUCTS_BASE``
    Root directory for final science data products.  Default:
    ``{CONTIMG_BASE_DIR}/products``.

``DSA110_VLA_CAL_DB``
    Path to the VLA calibrator SQLite database.  Default:
    ``{DSA110_CATALOG_DIR}/vla_calibrators.sqlite3``.

``DSA110_UPPER_LIMITS_DB``
    Path to the non-detection upper limits SQLite database.  Default:
    ``{CONTIMG_BASE_DIR}/state/db/upper_limits.db``.

Usage
-----
>>> from dsa110_continuum.config import paths, get_env_path
>>> print(paths.ms_dir)          # /stage/dsa110-contimg/ms  (or $DSA110_MS_DIR)
>>> print(paths.pipeline_db)     # /data/dsa110-contimg/state/db/pipeline.sqlite3

All paths are :class:`pathlib.Path` objects so that ``/``-concatenation works::

    from dsa110_continuum.config import paths
    catalog_db = paths.catalog_dir / "nvss_strip.sqlite3"
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

__all__ = [
    "PathConfig",
    "paths",
    "get_env_path",
]


# ── get_env_path helper ───────────────────────────────────────────────────────

def get_env_path(env_var: str, default: str) -> Path:
    """Return ``os.environ[env_var]`` as a :class:`Path`, or *default* if unset.

    Parameters
    ----------
    env_var : str
        Environment variable name.
    default : str
        Fallback path string used when the environment variable is not set.

    Returns
    -------
    Path
        Resolved path (may not exist — callers check existence themselves).

    Examples
    --------
    >>> import os
    >>> from dsa110_continuum.config import get_env_path
    >>> os.environ['MY_DIR'] = '/tmp/test'
    >>> get_env_path('MY_DIR', '/default').as_posix()
    '/tmp/test'
    >>> del os.environ['MY_DIR']
    >>> get_env_path('MY_DIR', '/default').as_posix()
    '/default'
    """
    return Path(os.environ.get(env_var, default))


# ── PathConfig dataclass ──────────────────────────────────────────────────────

@dataclass(frozen=True)
class PathConfig:
    """Immutable bundle of all pipeline data paths.

    Every field has a default that reads from an environment variable at
    construction time, falling back to the canonical H17 path.  Override any
    path programmatically by constructing a custom :class:`PathConfig` and
    passing it to the relevant pipeline function, or by setting the
    corresponding environment variable before importing.

    Attributes
    ----------
    base_dir : Path
        Root of all pipeline outputs.  Env: ``CONTIMG_BASE_DIR``.
    ms_dir : Path
        Raw Measurement Set directory.  Env: ``DSA110_MS_DIR``.
    stage_image_base : Path
        Staging directory for per-date WSClean images.
        Env: ``DSA110_STAGE_IMAGE_BASE``.
    incoming_dir : Path
        HDF5 correlator output directory.  Env: ``DSA110_INCOMING_DIR``.
    pipeline_db : Path
        SQLite pipeline state database.  Env: ``PIPELINE_DB``.
    catalog_dir : Path
        Reference catalog directory.  Env: ``DSA110_CATALOG_DIR``.
    products_base : Path
        Root directory for final science products.  Env: ``DSA110_PRODUCTS_BASE``.
    vla_cal_db : Path
        VLA calibrator database.  Env: ``DSA110_VLA_CAL_DB``.
    upper_limits_db : Path
        Non-detection upper limits database.  Env: ``DSA110_UPPER_LIMITS_DB``.
    """

    base_dir: Path = field(
        default_factory=lambda: get_env_path("CONTIMG_BASE_DIR", "/data/dsa110-contimg")
    )
    ms_dir: Path = field(
        default_factory=lambda: get_env_path("DSA110_MS_DIR", "/stage/dsa110-contimg/ms")
    )
    stage_image_base: Path = field(
        default_factory=lambda: get_env_path(
            "DSA110_STAGE_IMAGE_BASE", "/stage/dsa110-contimg/images"
        )
    )
    incoming_dir: Path = field(
        default_factory=lambda: get_env_path("DSA110_INCOMING_DIR", "/data/incoming")
    )
    pipeline_db: Path = field(
        default_factory=lambda: get_env_path(
            "PIPELINE_DB",
            str(get_env_path("CONTIMG_BASE_DIR", "/data/dsa110-contimg")
                / "state/db/pipeline.sqlite3"),
        )
    )
    catalog_dir: Path = field(
        default_factory=lambda: get_env_path(
            "DSA110_CATALOG_DIR",
            str(get_env_path("CONTIMG_BASE_DIR", "/data/dsa110-contimg")
                / "state/catalogs"),
        )
    )
    products_base: Path = field(
        default_factory=lambda: get_env_path(
            "DSA110_PRODUCTS_BASE",
            str(get_env_path("CONTIMG_BASE_DIR", "/data/dsa110-contimg") / "products"),
        )
    )
    vla_cal_db: Path = field(
        default_factory=lambda: get_env_path(
            "DSA110_VLA_CAL_DB",
            str(get_env_path(
                "DSA110_CATALOG_DIR",
                str(get_env_path("CONTIMG_BASE_DIR", "/data/dsa110-contimg")
                    / "state/catalogs"),
            ) / "vla_calibrators.sqlite3"),
        )
    )
    upper_limits_db: Path = field(
        default_factory=lambda: get_env_path(
            "DSA110_UPPER_LIMITS_DB",
            str(get_env_path("CONTIMG_BASE_DIR", "/data/dsa110-contimg")
                / "state/db/upper_limits.db"),
        )
    )

    # ── Derived convenience paths (not environment-overridable, derived from
    # the above base paths)

    @property
    def state_dir(self) -> Path:
        """``{base_dir}/state``."""
        return self.base_dir / "state"

    @property
    def db_dir(self) -> Path:
        """``{base_dir}/state/db``."""
        return self.base_dir / "state" / "db"

    def image_dir(self, date: str) -> Path:
        """Per-date WSClean staging directory: ``{stage_image_base}/mosaic_{date}``."""
        return self.stage_image_base / f"mosaic_{date}"

    def products_dir(self, date: str) -> Path:
        """Per-date final products directory: ``{products_base}/{date}``."""
        return self.products_base / date

    def __repr__(self) -> str:
        lines = [
            "PathConfig(",
            f"  base_dir         = {self.base_dir}",
            f"  ms_dir           = {self.ms_dir}",
            f"  stage_image_base = {self.stage_image_base}",
            f"  incoming_dir     = {self.incoming_dir}",
            f"  pipeline_db      = {self.pipeline_db}",
            f"  catalog_dir      = {self.catalog_dir}",
            f"  products_base    = {self.products_base}",
            f"  vla_cal_db       = {self.vla_cal_db}",
            f"  upper_limits_db  = {self.upper_limits_db}",
            ")",
        ]
        return "\n".join(lines)


# ── Module-level singleton ─────────────────────────────────────────────────────

#: Module-level singleton.  Import and use this rather than constructing a new
#: :class:`PathConfig` each time.  It is evaluated *once* at import time, so
#: environment variables must be set before the first ``import`` of this module.
#:
#: To override paths in tests, either set environment variables before import
#: or construct a fresh :class:`PathConfig`::
#:
#:     import os
#:     os.environ["PIPELINE_DB"] = "/tmp/test.sqlite3"
#:     from dsa110_continuum.config import paths  # reads env at this point
#:     # -- or --
#:     from dsa110_continuum.config import PathConfig
#:     custom = PathConfig(pipeline_db=Path("/tmp/test.sqlite3"))
paths: PathConfig = PathConfig()
