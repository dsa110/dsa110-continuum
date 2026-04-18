"""
Pure-function source-finding core for DSA-110 continuum pipeline.

Steps:
  1. run_bane()            — background/RMS estimation (AegeanTools.BANE)
  2. run_aegean()          — blind detection (AegeanTools.source_finder)
  3. write_catalog() /
     write_empty_catalog() — FITS table output
  4. check_catalog()       — QA / logging
  5. run_source_finding()  — orchestrator
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from astropy.table import Table

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class SourceCatalogEntry:
    """One detected source component from Aegean."""
    source_name:      str
    ra_deg:           float
    dec_deg:          float
    peak_flux_jy:     float
    peak_flux_err_jy: float
    int_flux_jy:      float
    a_arcsec:         float
    b_arcsec:         float
    pa_deg:           float
    local_rms_jy:     float


# ---------------------------------------------------------------------------
# Catalog I/O
# ---------------------------------------------------------------------------

_CATALOG_COLS = [
    "source_name", "ra_deg", "dec_deg", "peak_flux_jy",
    "peak_flux_err_jy", "int_flux_jy", "a_arcsec", "b_arcsec",
    "pa_deg", "local_rms_jy",
]
_CATALOG_DTYPES = ["U64", float, float, float, float, float, float, float, float, float]


def write_catalog(entries: list[SourceCatalogEntry], out_path: str | Path) -> None:
    """Write source entries to a FITS binary table."""
    rows = [
        {
            "source_name": e.source_name,
            "ra_deg": e.ra_deg,
            "dec_deg": e.dec_deg,
            "peak_flux_jy": e.peak_flux_jy,
            "peak_flux_err_jy": e.peak_flux_err_jy,
            "int_flux_jy": e.int_flux_jy,
            "a_arcsec": e.a_arcsec,
            "b_arcsec": e.b_arcsec,
            "pa_deg": e.pa_deg,
            "local_rms_jy": e.local_rms_jy,
        }
        for e in entries
    ]
    t = Table(rows)
    t.write(str(out_path), overwrite=True)
    log.info("Catalog written: %s  (%d sources)", out_path, len(entries))


def write_empty_catalog(out_path: str | Path) -> None:
    """Write a zero-row FITS table with the correct column schema."""
    t = Table(names=_CATALOG_COLS, dtype=_CATALOG_DTYPES)
    t.write(str(out_path), overwrite=True)
    log.info("Empty catalog written: %s", out_path)


# ---------------------------------------------------------------------------
# Stubs — implemented in later tasks
# ---------------------------------------------------------------------------

def run_bane(mosaic_path, *, box_size=600, step_size=300, cores=1, skip_existing=True):
    raise NotImplementedError("run_bane not yet implemented")

def run_aegean(mosaic_path, bkg_path, rms_path, *, sigma=7.0):
    raise NotImplementedError("run_aegean not yet implemented")

def check_catalog(catalog_path, *, sky_ra_range=(300.0, 360.0), sky_dec_range=(0.0, 40.0)):
    raise NotImplementedError("check_catalog not yet implemented")

def run_source_finding(mosaic_path, out_path, *, bane_box=600, bane_step=300, aegean_sigma=7.0):
    raise NotImplementedError("run_source_finding not yet implemented")
