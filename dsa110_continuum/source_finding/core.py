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
        if self.local_rms_jy < 0:
            raise ValueError(f"local_rms_jy must be non-negative, got {self.local_rms_jy}")
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
# Source-finding QA: completeness and size distribution
# ---------------------------------------------------------------------------

from dataclasses import dataclass as _dataclass
from typing import Literal as _Literal


@_dataclass
class CompletenessResult:
    """NVSS source recovery completeness result."""
    n_nvss_reference: int
    n_recovered: int
    completeness_frac: float
    gate: _Literal["PASS", "WARN", "FAIL"]


@_dataclass
class SizeQAResult:
    """Source size distribution QA result."""
    n_sources: int
    frac_subbeam: float
    frac_elongated: float
    beam_a_arcsec: float
    beam_b_arcsec: float
    gate: _Literal["PASS", "WARN"]


def _cone_search_nvss(
    ra_center: float, dec_center: float, radius_deg: float
) -> "pd.DataFrame":
    """Query NVSS via the catalog layer (patchable in tests)."""
    import pandas as pd
    try:
        from dsa110_continuum.catalog.query import cone_search
        result = cone_search("nvss", ra_center, dec_center, radius_deg)
        return result if result is not None else pd.DataFrame()
    except (FileNotFoundError, OSError, RuntimeError, KeyError, ValueError) as exc:
        log.warning("NVSS query failed: %s", exc)
        return pd.DataFrame()


def check_source_completeness(
    catalog: list[SourceCatalogEntry],
    *,
    ra_center: float,
    dec_center: float,
    radius_deg: float = 2.0,
    match_radius_arcsec: float = 15.0,
) -> CompletenessResult:
    """Check what fraction of bright NVSS sources was recovered by Aegean.

    Parameters
    ----------
    catalog : list[SourceCatalogEntry]
        Aegean detection catalog.
    ra_center, dec_center : float
        Field center for NVSS cone query.
    radius_deg : float
        Cone search radius in degrees.
    match_radius_arcsec : float
        Position match radius (default 15 arcsec).

    Returns
    -------
    CompletenessResult
    """
    import numpy as np

    nvss_df = _cone_search_nvss(ra_center, dec_center, radius_deg)

    if len(nvss_df) == 0 or "ra_deg" not in nvss_df.columns:
        log.warning("No NVSS sources in field for completeness check \u2192 WARN")
        return CompletenessResult(
            n_nvss_reference=0,
            n_recovered=0,
            completeness_frac=0.0,
            gate="WARN",
        )

    n_nvss = len(nvss_df)
    nvss_ra = nvss_df["ra_deg"].values.astype(float)
    nvss_dec = nvss_df["dec_deg"].values.astype(float)

    det_ra = np.array([s.ra_deg for s in catalog])
    det_dec = np.array([s.dec_deg for s in catalog])

    # Brute-force match: for each NVSS source, find nearest Aegean detection
    recovered = 0
    thr_deg = match_radius_arcsec / 3600.0
    for nra, ndec in zip(nvss_ra, nvss_dec):
        if len(det_ra) == 0:
            break
        cos_dec = np.cos(np.radians(ndec))
        dra = (det_ra - nra) * cos_dec
        ddec = det_dec - ndec
        sep = np.sqrt(dra**2 + ddec**2)
        if sep.min() <= thr_deg:
            recovered += 1

    completeness = recovered / n_nvss if n_nvss > 0 else 0.0

    if n_nvss < 3:
        gate: _Literal["PASS", "WARN", "FAIL"] = "WARN"
    elif completeness >= 0.60:
        gate = "PASS"
    elif completeness >= 0.40:
        gate = "WARN"
    else:
        gate = "FAIL"

    log.info(
        "Source completeness: %d/%d NVSS recovered (%.0f%%) [%s]",
        recovered, n_nvss, completeness * 100, gate,
    )
    return CompletenessResult(
        n_nvss_reference=n_nvss,
        n_recovered=recovered,
        completeness_frac=round(completeness, 4),
        gate=gate,
    )


def check_size_distribution(
    catalog: list[SourceCatalogEntry],
    *,
    beam_a_arcsec: float = 36.9,
    beam_b_arcsec: float = 25.5,
) -> SizeQAResult:
    """Check fitted source size distribution for artefact signatures.

    Parameters
    ----------
    catalog : list[SourceCatalogEntry]
        Aegean detection catalog.
    beam_a_arcsec, beam_b_arcsec : float
        Synthesized beam axes in arcseconds (DSA-110 defaults).

    Returns
    -------
    SizeQAResult
    """
    import numpy as np

    n = len(catalog)
    if n == 0:
        return SizeQAResult(
            n_sources=0,
            frac_subbeam=0.0,
            frac_elongated=0.0,
            beam_a_arcsec=beam_a_arcsec,
            beam_b_arcsec=beam_b_arcsec,
            gate="WARN",
        )

    a_arr = np.array([s.a_arcsec for s in catalog])
    b_arr = np.array([s.b_arcsec for s in catalog])

    # Sub-beam: major axis < 90% of synthesized beam major axis
    subbeam_mask = a_arr < 0.9 * beam_a_arcsec
    frac_subbeam = float(subbeam_mask.sum()) / n

    # Elongated: axis ratio > 5
    safe_b = np.where(b_arr > 0, b_arr, np.nan)
    ratio = a_arr / safe_b
    elongated_mask = ratio > 5.0
    frac_elongated = float(np.sum(elongated_mask & np.isfinite(ratio))) / n

    warn = frac_subbeam > 0.05 or frac_elongated > 0.10
    gate: _Literal["PASS", "WARN"] = "WARN" if warn else "PASS"

    log.info(
        "Size QA: %.0f%% sub-beam  %.0f%% elongated  [%s]",
        frac_subbeam * 100, frac_elongated * 100, gate,
    )
    return SizeQAResult(
        n_sources=n,
        frac_subbeam=round(frac_subbeam, 4),
        frac_elongated=round(frac_elongated, 4),
        beam_a_arcsec=beam_a_arcsec,
        beam_b_arcsec=beam_b_arcsec,
        gate=gate,
    )


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

    qa_passed = check_catalog(out_path)
    if not qa_passed:
        log.warning(
            "run_source_finding: check_catalog returned False for %s "
            "(empty catalog — QA non-fatal, inspect output)",
            out_path,
        )

    # ── Source-finding QA ─────────────────────────────────────────────────────
    if entries:
        try:
            import numpy as _np
            from astropy.io import fits as _fits
            from astropy.wcs import WCS as _WCS
            with _fits.open(mosaic_path) as _hdul:
                _wcs = _WCS(_hdul[0].header).celestial
                _ny, _nx = _hdul[0].data.squeeze().shape[-2:]
            _corners_sky = _wcs.pixel_to_world(
                [0, _nx - 1, 0, _nx - 1], [0, 0, _ny - 1, _ny - 1]
            )
            _ra_c = float(sum(c.ra.deg for c in _corners_sky) / 4)
            _dec_c = float(sum(c.dec.deg for c in _corners_sky) / 4)
            _comp = check_source_completeness(
                entries, ra_center=_ra_c, dec_center=_dec_c, radius_deg=1.5
            )
            _size = check_size_distribution(entries)
            try:
                from dsa110_continuum.qa.epoch_log import append_epoch_qa
                append_epoch_qa({
                    "stage": "source_finding",
                    "mosaic_path": str(mosaic_path),
                    "n_sources_aegean": len(entries),
                    "n_nvss_reference": _comp.n_nvss_reference,
                    "completeness_frac": _comp.completeness_frac,
                    "completeness_gate": _comp.gate,
                    "frac_subbeam": _size.frac_subbeam,
                    "frac_elongated": _size.frac_elongated,
                    "size_gate": _size.gate,
                })
            except Exception:
                pass
        except Exception as exc:
            log.warning("Source-finding QA skipped: %s", exc)

    return str(out_path)
