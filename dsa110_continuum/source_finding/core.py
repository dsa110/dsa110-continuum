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

    def __post_init__(self) -> None:
        if self.local_rms_jy <= 0:
            raise ValueError(f"local_rms_jy must be positive, got {self.local_rms_jy}")
        if not (0.0 <= self.ra_deg < 360.0):
            raise ValueError(f"ra_deg out of range [0, 360): {self.ra_deg}")


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
    """Write source entries to a FITS binary table.

    If *entries* is empty, writes a zero-row table with the correct schema
    (delegates to write_empty_catalog).
    """
    if not entries:
        write_empty_catalog(out_path)
        return
    t = Table(
        names=_CATALOG_COLS,
        dtype=_CATALOG_DTYPES,
        rows=[[getattr(e, col) for col in _CATALOG_COLS] for e in entries],
    )
    t.write(str(out_path), format="fits", overwrite=True)
    log.info("Catalog written: %s  (%d sources)", out_path, len(entries))


def write_empty_catalog(out_path: str | Path) -> None:
    """Write a zero-row FITS table with the correct column schema."""
    t = Table(names=_CATALOG_COLS, dtype=_CATALOG_DTYPES)
    t.write(str(out_path), format="fits", overwrite=True)
    log.info("Empty catalog written: %s", out_path)


# ---------------------------------------------------------------------------
# BANE background estimator
# ---------------------------------------------------------------------------

def run_bane(
    mosaic_path: str | Path,
    *,
    box_size: int = 600,
    step_size: int = 300,
    cores: int = 1,
    skip_existing: bool = True,
) -> tuple[str, str]:
    """Run BANE background/RMS estimator. Returns (bkg_path, rms_path).

    Parameters
    ----------
    mosaic_path : str | Path
        Input mosaic FITS file.
    box_size : int
        BANE box size in pixels (default 600 ≈ 50 beams at 20"/pix).
    step_size : int
        BANE step size in pixels (default 300).
    cores : int
        Worker cores (default 1 to avoid shared-memory deadlock on
        NaN-heavy mosaics).
    skip_existing : bool
        If True and both output files exist, return immediately without
        calling BANE.

    Returns
    -------
    tuple[str, str]
        Paths to (background_map.fits, rms_map.fits).

    Raises
    ------
    RuntimeError
        If BANE runs but does not produce the expected output files.
    ImportError
        If AegeanTools is not installed.
    """
    mosaic_path = Path(mosaic_path)
    stem = str(mosaic_path.parent / mosaic_path.stem)
    bkg_path = stem + "_bkg.fits"
    rms_path = stem + "_rms.fits"
    mosaic_path = str(mosaic_path)  # keep as str for BANE API

    if not Path(mosaic_path).exists():
        raise FileNotFoundError(f"Mosaic not found: {mosaic_path}")

    if skip_existing and Path(bkg_path).exists() and Path(rms_path).exists():
        log.info("BANE outputs already exist — skipping: %s, %s", bkg_path, rms_path)
        return bkg_path, rms_path

    log.info(
        "Running BANE on %s (box=%d, step=%d, cores=%d) ...",
        mosaic_path, box_size, step_size, cores,
    )
    try:
        from AegeanTools import BANE as _bane
    except ImportError as exc:
        raise ImportError(
            "AegeanTools not installed. "
            "Install with: pip install git+https://github.com/PaulHancock/Aegean.git"
        ) from exc

    _bane.filter_image(
        im_name=mosaic_path,
        out_base=stem,
        step_size=[step_size, step_size],
        box_size=[box_size, box_size],
        cores=cores,
        mask=True,
    )

    if not Path(bkg_path).exists() or not Path(rms_path).exists():
        raise RuntimeError(
            f"BANE did not produce expected outputs: {bkg_path}, {rms_path}"
        )

    log.info("BANE done: bkg=%s, rms=%s", bkg_path, rms_path)
    return bkg_path, rms_path


# ---------------------------------------------------------------------------
# Stubs — implemented in Task 3
# ---------------------------------------------------------------------------

def run_aegean(
    mosaic_path: str | Path,
    bkg_path: str | Path,
    rms_path: str | Path,
    *,
    sigma: float = 7.0,
) -> list[SourceCatalogEntry]:
    """Run Aegean blind source detection on *mosaic_path*. Implemented in Task 3."""
    raise NotImplementedError("run_aegean not yet implemented")


def check_catalog(
    catalog_path: str | Path,
    *,
    sky_ra_range: tuple[float, float] = (300.0, 360.0),
    sky_dec_range: tuple[float, float] = (0.0, 40.0),
) -> bool:
    """QA check on catalog at *catalog_path*. Implemented in Task 3."""
    raise NotImplementedError("check_catalog not yet implemented")


def run_source_finding(
    mosaic_path: str | Path,
    out_path: str | Path,
    *,
    bane_box: int = 600,
    bane_step: int = 300,
    aegean_sigma: float = 7.0,
) -> str:
    """Orchestrator: BANE → Aegean → write catalog → QA. Implemented in Task 3."""
    raise NotImplementedError("run_source_finding not yet implemented")
