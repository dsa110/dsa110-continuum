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
# Aegean blind source detection
# ---------------------------------------------------------------------------

def run_aegean(
    mosaic_path: str | Path,
    bkg_path: str | Path,
    rms_path: str | Path,
    *,
    sigma: float = 7.0,
) -> list[SourceCatalogEntry]:
    """Run Aegean blind source detection.

    Parameters
    ----------
    mosaic_path : str | Path
        Input mosaic FITS file.
    bkg_path : str | Path
        BANE background map (from run_bane).
    rms_path : str | Path
        BANE RMS map (from run_bane).
    sigma : float
        Detection threshold in local RMS units (default 7.0sigma, raised from
        the standard 5sigma to suppress spurious edge detections from pbcor
        noise amplification at tile boundaries).

    Returns
    -------
    list[SourceCatalogEntry]
        Detected source components (may be empty).

    Raises
    ------
    ImportError
        If AegeanTools is not installed.
    """
    try:
        from AegeanTools.source_finder import SourceFinder as _SourceFinder
    except ImportError as exc:
        raise ImportError(
            "AegeanTools not installed. "
            "Install with: pip install git+https://github.com/PaulHancock/Aegean.git"
        ) from exc

    log.info("Running Aegean (%.1fsigma threshold) on %s ...", sigma, mosaic_path)
    sf = _SourceFinder(log=logging.getLogger("AegeanTools"))
    found = sf.find_sources_in_image(
        filename=str(mosaic_path),
        hdu_index=0,
        outfile=None,
        innerclip=sigma,
        outerclip=sigma - 1.0,
        rmsin=str(rms_path),
        bkgin=str(bkg_path),
        cores=1,
    )

    if not found:
        log.warning("Aegean found no sources at %.1fsigma", sigma)
        return []

    log.info("Aegean found %d source components", len(found))
    entries: list[SourceCatalogEntry] = []
    for s in found:
        ra = float(s.ra)
        dec = float(s.dec)
        entries.append(SourceCatalogEntry(
            source_name=f"AEG_J{ra:.4f}{dec:+.4f}",
            ra_deg=ra,
            dec_deg=dec,
            peak_flux_jy=float(s.peak_flux),
            peak_flux_err_jy=float(getattr(s, "err_peak_flux", 0.0)),
            int_flux_jy=float(getattr(s, "int_flux", s.peak_flux)),
            a_arcsec=float(getattr(s, "a", 0.0)),
            b_arcsec=float(getattr(s, "b", 0.0)),
            pa_deg=float(getattr(s, "pa", 0.0)),
            local_rms_jy=float(getattr(s, "local_rms", 0.0)),
        ))
    return entries


# ---------------------------------------------------------------------------
# Catalog QA
# ---------------------------------------------------------------------------

def check_catalog(
    catalog_path: str | Path,
    *,
    sky_ra_range: tuple[float, float] = (300.0, 360.0),
    sky_dec_range: tuple[float, float] = (0.0, 40.0),
) -> bool:
    """Check catalog quality. Returns True if catalog is non-empty.

    Logs source count, bright-source (>1 Jy) count, and sources in the
    expected sky window. The bright-source criterion is a warning only --
    not a hard failure -- since simulated dirty-image fluxes are suppressed
    and production calibration may not yet be complete.

    Parameters
    ----------
    catalog_path : str | Path
        Path to catalog FITS table.
    sky_ra_range : tuple[float, float]
        Expected RA window (deg) for positional sanity check.
        Default (300, 360) covers the DSA-110 nominal pointing for
        the 2026-01-25 observation.
    sky_dec_range : tuple[float, float]
        Expected Dec window (deg). Default (0, 40).

    Returns
    -------
    bool
        True if catalog has at least one source, False if empty.
    """
    import numpy as np
    t = Table.read(str(catalog_path))
    log.info("Catalog: %d source(s)", len(t))

    if len(t) == 0:
        log.error("QA FAIL: empty catalog")
        return False

    if "peak_flux_jy" in t.colnames:
        bright = t[t["peak_flux_jy"] > 1.0]
        log.info("Sources > 1 Jy: %d", len(bright))
        if len(bright) == 0:
            log.warning("QA WARNING: no bright sources (>1 Jy) -- expected for dirty-image mosaics")
        else:
            for row in bright:
                log.info("  %s  RA=%.3f  Dec=%.3f  peak=%.2f Jy",
                         row["source_name"], row["ra_deg"], row["dec_deg"],
                         row["peak_flux_jy"])

    if "ra_deg" in t.colnames and "dec_deg" in t.colnames:
        ra_lo, ra_hi = sky_ra_range
        dec_lo, dec_hi = sky_dec_range
        in_range = int(np.sum(
            (t["ra_deg"] > ra_lo) & (t["ra_deg"] < ra_hi) &
            (t["dec_deg"] > dec_lo) & (t["dec_deg"] < dec_hi)
        ))
        log.info(
            "Sources in sky window (RA %.0f-%.0f, Dec %.0f-%.0f): %d/%d",
            ra_lo, ra_hi, dec_lo, dec_hi, in_range, len(t),
        )

    log.info("QA PASSED: catalog has %d source(s)", len(t))
    return True


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_source_finding(
    mosaic_path: str | Path,
    out_path: str | Path,
    *,
    bane_box: int = 600,
    bane_step: int = 300,
    aegean_sigma: float = 7.0,
) -> str:
    """Run the full source-finding pipeline: BANE -> Aegean -> catalog -> QA.

    Parameters
    ----------
    mosaic_path : str | Path
        Input mosaic FITS file.
    out_path : str | Path
        Output catalog FITS path.
    bane_box : int
        BANE box size in pixels (default 600).
    bane_step : int
        BANE step size in pixels (default 300).
    aegean_sigma : float
        Aegean detection threshold in sigma (default 7.0).

    Returns
    -------
    str
        Path to the written catalog file.
    """
    mosaic_path = Path(mosaic_path)
    out_path = Path(out_path)

    bkg_path, rms_path = run_bane(mosaic_path, box_size=bane_box, step_size=bane_step)
    entries = run_aegean(mosaic_path, bkg_path, rms_path, sigma=aegean_sigma)

    if entries:
        write_catalog(entries, out_path)
    else:
        write_empty_catalog(out_path)

    check_catalog(out_path)
    return str(out_path)
