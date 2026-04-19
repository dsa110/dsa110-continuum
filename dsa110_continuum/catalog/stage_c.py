"""Stage C: post-discovery cross-match annotation for DSA-110 continuum pipeline.

Reads the Stage B Aegean FITS catalog, matches each detection against the master
radio catalog (NVSS+VLASS+FIRST+RACS) with fallback to individual catalogs, and
writes an annotated FITS table.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.table import Table

from dsa110_continuum.catalog.crossmatch import (
    cross_match_sources,
    calculate_positional_offsets,
)

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Internal catalog query shim (allows patching in tests)
# ---------------------------------------------------------------------------

def _cone_search(catalog_type: str, ra_center: float, dec_center: float,
                 radius_deg: float) -> pd.DataFrame:
    """Thin wrapper around catalog.query.cone_search — monkeypatched in tests."""
    try:
        from dsa110_continuum.catalog.query import cone_search
        result = cone_search(catalog_type, ra_center, dec_center, radius_deg)
        return result if result is not None else pd.DataFrame()
    except (FileNotFoundError, OSError, RuntimeError, KeyError, ValueError) as exc:
        log.warning("Catalog query failed for %s: %s", catalog_type, exc)
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# Output column schema
# ---------------------------------------------------------------------------

_OUTPUT_COLS = [
    "source_name", "ra_deg", "dec_deg", "peak_flux_jy", "snr",
    "master_matched", "master_sep_arcsec", "master_flux_mjy",
    "master_flux_ratio", "master_source_id",
    "nvss_matched", "nvss_sep_arcsec", "nvss_flux_mjy",
    "first_matched", "first_sep_arcsec",
    "racs_matched", "racs_sep_arcsec",
    "any_matched", "new_source_candidate",
]


def _read_aegean_fits(catalog_path: str | Path) -> pd.DataFrame:
    """Read Aegean FITS binary table into a DataFrame."""
    with fits.open(catalog_path) as hdul:
        t = Table(hdul[1].data)
        df = t.to_pandas()
        # Decode byte strings if present (FITS column encoding)
        for col in df.select_dtypes(include=["object"]).columns:
            df[col] = df[col].apply(
                lambda v: v.decode("utf-8").strip() if isinstance(v, bytes) else str(v)
            )
    return df


def _match_catalog(
    detected_ra: np.ndarray,
    detected_dec: np.ndarray,
    catalog_df: pd.DataFrame,
    match_radius_arcsec: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Match detected sources against a catalog DataFrame.

    Returns (matched_bool, sep_arcsec, flux_mjy, catalog_indices) arrays
    aligned to detected sources. -1.0 / -1 sentinel means no match.
    """
    n = len(detected_ra)
    matched = np.zeros(n, dtype=bool)
    sep = np.full(n, -1.0)
    flux = np.full(n, -1.0)
    cat_idx = np.full(n, -1, dtype=int)   # NEW: catalog row index for each match

    if catalog_df is None or len(catalog_df) == 0:
        return matched, sep, flux, cat_idx

    required = {"ra_deg", "dec_deg", "flux_mjy"}
    if not required.issubset(catalog_df.columns):
        log.warning("Catalog missing expected columns (need ra_deg, dec_deg, flux_mjy)")
        return matched, sep, flux, cat_idx

    result = cross_match_sources(
        detected_ra=detected_ra,
        detected_dec=detected_dec,
        catalog_ra=catalog_df["ra_deg"].values,
        catalog_dec=catalog_df["dec_deg"].values,
        radius_arcsec=match_radius_arcsec,
        detected_flux=None,
        catalog_flux=catalog_df["flux_mjy"].values,
    )

    if result is None or len(result) == 0:
        return matched, sep, flux, cat_idx

    for _, row in result.iterrows():
        di = int(row["detected_idx"])
        ci = int(row["catalog_idx"])
        matched[di] = True
        sep[di] = float(row["separation_arcsec"])
        flux[di] = float(catalog_df["flux_mjy"].iloc[ci])
        cat_idx[di] = ci   # NEW

    return matched, sep, flux, cat_idx


def run_stage_c(
    catalog_path: str | Path,
    out_path: str | Path | None = None,
    *,
    ra_center: float | None = None,
    dec_center: float | None = None,
    radius_deg: float = 2.0,
    match_radius_arcsec: float = 10.0,
    new_source_snr_threshold: float = 5.0,
) -> Path:
    """Annotate Stage B Aegean detections with multi-catalog cross-match results.

    Parameters
    ----------
    catalog_path : str or Path
        Stage B Aegean FITS catalog (output of run_source_finding).
    out_path : str, Path, or None
        Output path. If a directory, writes ``{stem}_crossmatched.fits`` inside it.
        If None, writes next to the input catalog.
    ra_center, dec_center : float or None
        Field center for catalog cone queries. Derived from source centroid if None.
    radius_deg : float
        Cone search radius in degrees.
    match_radius_arcsec : float
        Cross-match radius in arcseconds.
    new_source_snr_threshold : float
        SNR threshold for flagging unmatched sources as candidates.

    Returns
    -------
    Path
        Path to the written annotated FITS table.
    """
    catalog_path = Path(catalog_path)

    # -- Read Aegean catalog --------------------------------------------------
    df = _read_aegean_fits(catalog_path)
    if len(df) == 0:
        raise ValueError(f"No sources in catalog: {catalog_path}")

    log.info("Stage C: %d detections loaded from %s", len(df), catalog_path)

    ra_arr  = df["ra_deg"].values.astype(float)
    dec_arr = df["dec_deg"].values.astype(float)
    peak    = df["peak_flux_jy"].values.astype(float)
    rms     = df["local_rms_jy"].values.astype(float)
    snr_arr = np.where(rms > 0, peak / rms, 0.0)

    # -- Derive field center if not provided ----------------------------------
    if ra_center is None:
        ra_center = float(np.median(ra_arr))
    if dec_center is None:
        dec_center = float(np.median(dec_arr))

    # -- Primary: master catalog match ----------------------------------------
    master_df = _cone_search("master", ra_center, dec_center, radius_deg)
    master_matched, master_sep, master_flux, master_cat_idx = _match_catalog(
        ra_arr, dec_arr, master_df, match_radius_arcsec
    )

    # Extract master source IDs from the match results (no second query needed)
    master_ids = np.full(len(df), "", dtype=object)
    if len(master_df) > 0 and "source_id" in master_df.columns:
        for i in range(len(df)):
            if master_matched[i] and master_cat_idx[i] >= 0:
                master_ids[i] = str(master_df["source_id"].iloc[master_cat_idx[i]])

    # -- Flux ratio -----------------------------------------------------------
    # Only defined where master_matched=True and master_flux > 0
    master_flux_ratio = np.full(len(df), -1.0)
    for i in range(len(df)):
        if master_matched[i] and master_flux[i] > 0:
            master_flux_ratio[i] = float(peak[i] / (master_flux[i] / 1000.0))

    # -- Fallback: individual catalogs for unmatched --------------------------
    unmatched_mask = ~master_matched

    def _fallback(catalog_name: str):
        if not np.any(unmatched_mask):
            return (np.zeros(len(df), dtype=bool),
                    np.full(len(df), -1.0),
                    np.full(len(df), -1.0),
                    np.full(len(df), -1, dtype=int))
        cat_df = _cone_search(catalog_name, ra_center, dec_center, radius_deg)
        fb_matched, fb_sep, fb_flux, _ = _match_catalog(
            ra_arr, dec_arr, cat_df, match_radius_arcsec
        )
        # Only credit fallback match for sources not already master-matched
        for i in range(len(df)):
            if master_matched[i]:
                fb_matched[i] = False
                fb_sep[i] = -1.0
                fb_flux[i] = -1.0
        return fb_matched, fb_sep, fb_flux, np.full(len(df), -1, dtype=int)

    nvss_matched,  nvss_sep,  nvss_flux,  _ = _fallback("nvss")
    first_matched, first_sep, _,           _ = _fallback("first")
    racs_matched,  racs_sep,  _,           _ = _fallback("rax")  # "rax" is the cone_search key for RACS

    any_matched = master_matched | nvss_matched | first_matched | racs_matched
    new_candidate = (~any_matched) & (snr_arr >= new_source_snr_threshold)

    # -- Astrometry QA --------------------------------------------------------
    if np.sum(master_matched) >= 3 and len(master_df) > 0:
        try:
            result_qa = cross_match_sources(
                detected_ra=ra_arr,
                detected_dec=dec_arr,
                catalog_ra=master_df["ra_deg"].values,
                catalog_dec=master_df["dec_deg"].values,
                radius_arcsec=match_radius_arcsec,
            )
            if result_qa is not None and len(result_qa) >= 3:
                dra_med, ddec_med, dra_mad, ddec_mad = calculate_positional_offsets(result_qa)
                log.info(
                    "Astrometry QA: median ΔRA=%.2f\" ΔDec=%.2f\" MAD_RA=%.2f\" MAD_Dec=%.2f\"",
                    dra_med.value, ddec_med.value, dra_mad.value, ddec_mad.value,
                )
        except (ValueError, RuntimeError, AttributeError) as exc:
            log.warning("Astrometry QA failed: %s", exc)

    n_cand = int(np.sum(new_candidate))
    n_matched_total = int(np.sum(any_matched))
    log.info(
        "Stage C summary: %d/%d matched, %d new source candidates",
        n_matched_total, len(df), n_cand,
    )
    if n_cand > 0:
        cand_names = df["source_name"].values[new_candidate]
        for name in cand_names[:10]:
            log.warning("New source candidate: %s", name)

    # -- Assemble output table ------------------------------------------------
    # Use fixed-length numpy string dtype to avoid astropy writing variable-length
    # FITS format (P17A()) which is invalid in binary tables.
    names_raw = df["source_name"].values.astype(str)
    max_name_len = max((len(s) for s in names_raw), default=1)
    names_arr = np.array(names_raw, dtype=f"U{max_name_len}")

    ids_raw = master_ids.astype(str)
    max_id_len = max((len(s) for s in ids_raw), default=1)
    ids_arr = np.array(ids_raw, dtype=f"U{max_id_len}")

    out_table = Table([
        names_arr,
        ra_arr,
        dec_arr,
        peak,
        snr_arr,
        master_matched.astype(np.int16),
        master_sep,
        master_flux,
        master_flux_ratio,
        ids_arr,
        nvss_matched.astype(np.int16),
        nvss_sep,
        nvss_flux,
        first_matched.astype(np.int16),
        first_sep,
        racs_matched.astype(np.int16),
        racs_sep,
        any_matched.astype(np.int16),
        new_candidate.astype(np.int16),
    ], names=_OUTPUT_COLS)

    # -- Write output ---------------------------------------------------------
    if out_path is None:
        out_path = catalog_path.parent / (catalog_path.stem + "_crossmatched.fits")
    else:
        out_path = Path(out_path)
        if out_path.is_dir():
            out_path = out_path / (catalog_path.stem + "_crossmatched.fits")

    out_table.write(str(out_path), format="fits", overwrite=True)
    log.info("Stage C annotated catalog written: %s", out_path)
    return out_path
