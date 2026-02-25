"""Cross-match sources in DSA-110 images with reference catalogs.

This module provides general-purpose cross-matching utilities for matching
detected sources with reference catalogs (NVSS, FIRST, RACS, etc.).

Based on VAST Post-Processing crossmatch.py patterns.

Features:
- Simple nearest-neighbor matching (fast)
- de Ruiter radius matching (uncertainty-weighted, statistically robust)
- One-to-many association handling
- Multi-catalog matching
"""

import logging

import astropy.units as u
import numpy as np
import pandas as pd
from astropy.coordinates import Angle, SkyCoord, match_coordinates_sky, search_around_sky
from astropy.stats import mad_std
from uncertainties import ufloat
from uncertainties.core import AffineScalarFunc

logger = logging.getLogger(__name__)


# =============================================================================
# de Ruiter Radius (VAST-style uncertainty-weighted association)
# =============================================================================


def calc_de_ruiter(
    ra1: np.ndarray,
    ra2: np.ndarray,
    dec1: np.ndarray,
    dec2: np.ndarray,
    sigma_ra1: np.ndarray,
    sigma_ra2: np.ndarray,
    sigma_dec1: np.ndarray,
    sigma_dec2: np.ndarray,
) -> np.ndarray:
    """Calculate the unitless de Ruiter radius for source association.

        The de Ruiter radius is a statistically robust measure for source association
        that accounts for positional uncertainties of both sources. A de Ruiter radius
        of ~5.68 corresponds to a 3-sigma match (99.7% confidence).

        Based on VAST pipeline implementation (de Ruiter et al. 1977).

        Formula:
        dr = sqrt(
        (Δra * cos(dec_avg))² / (σ_ra1² + σ_ra2²) +
        (Δdec)² / (σ_dec1² + σ_dec2²)
        )

    Parameters
    ----------
    ra1 : array_like
        RA of first sources (degrees)
    ra2 : array_like
        RA of second sources (degrees)
    dec1 : array_like
        Dec of first sources (degrees)
    dec2 : array_like
        Dec of second sources (degrees)
    sigma_ra1 : array_like
        RA uncertainty of first sources (degrees)
    sigma_ra2 : array_like
        RA uncertainty of second sources (degrees)
    sigma_dec1 : array_like
        Dec uncertainty of first sources (degrees)
    sigma_dec2 : array_like
        Dec uncertainty of second sources (degrees)

    Returns
    -------
        array_like
        Array of de Ruiter radii (unitless). Values < ~5.68 indicate
        statistically significant associations.

    Examples
    --------
        >>> dr = calc_de_ruiter(
        ...     ra1=np.array([180.0]), ra2=np.array([180.001]),
        ...     dec1=np.array([45.0]), dec2=np.array([45.0]),
        ...     sigma_ra1=np.array([0.001]), sigma_ra2=np.array([0.001]),
        ...     sigma_dec1=np.array([0.001]), sigma_dec2=np.array([0.001])
        ... )
        >>> print(f"de Ruiter radius: {dr[0]:.2f}")  # Should be small

    References
    ----------
        de Ruiter, H. R., Willis, A. G., & Arp, H. C. 1977, A&AS, 28, 211
    """
    # Copy arrays to avoid modifying inputs
    ra1 = np.asarray(ra1, dtype=np.float64).copy()
    ra2 = np.asarray(ra2, dtype=np.float64).copy()
    dec1 = np.asarray(dec1, dtype=np.float64)
    dec2 = np.asarray(dec2, dtype=np.float64)
    sigma_ra1 = np.asarray(sigma_ra1, dtype=np.float64)
    sigma_ra2 = np.asarray(sigma_ra2, dtype=np.float64)
    sigma_dec1 = np.asarray(sigma_dec1, dtype=np.float64)
    sigma_dec2 = np.asarray(sigma_dec2, dtype=np.float64)

    # Avoid RA wrapping issues by shifting coordinates near 0/360
    ra1[ra1 > 270.0] -= 180.0
    ra2[ra2 > 270.0] -= 180.0
    ra1[ra1 < 90.0] += 180.0
    ra2[ra2 < 90.0] += 180.0

    # Convert to radians
    ra1_rad = np.deg2rad(ra1)
    ra2_rad = np.deg2rad(ra2)
    dec1_rad = np.deg2rad(dec1)
    dec2_rad = np.deg2rad(dec2)
    sigma_ra1_rad = np.deg2rad(sigma_ra1)
    sigma_ra2_rad = np.deg2rad(sigma_ra2)
    sigma_dec1_rad = np.deg2rad(sigma_dec1)
    sigma_dec2_rad = np.deg2rad(sigma_dec2)

    # RA term: (Δra * cos(dec_avg))² / (σ_ra1² + σ_ra2²)
    delta_ra = ra1_rad - ra2_rad
    cos_dec_avg = np.cos((dec1_rad + dec2_rad) / 2.0)
    ra_term = (delta_ra * cos_dec_avg) ** 2 / (sigma_ra1_rad**2 + sigma_ra2_rad**2)

    # Dec term: (Δdec)² / (σ_dec1² + σ_dec2²)
    delta_dec = dec1_rad - dec2_rad
    dec_term = delta_dec**2 / (sigma_dec1_rad**2 + sigma_dec2_rad**2)

    # de Ruiter radius
    dr = np.sqrt(ra_term + dec_term)

    return dr


def calc_de_ruiter_beamwidth(
    ra1: np.ndarray,
    ra2: np.ndarray,
    dec1: np.ndarray,
    dec2: np.ndarray,
    bmaj1: np.ndarray,
    bmin1: np.ndarray,
    bmaj2: np.ndarray,
    bmin2: np.ndarray,
    snr1: np.ndarray,
    snr2: np.ndarray,
) -> np.ndarray:
    """Calculate de Ruiter radius using beam-derived uncertainties.

        When explicit positional uncertainties aren't available, they can be
        estimated from beam size and SNR: σ ≈ beam / (2 * SNR).

    Parameters
    ----------
    ra1 : array_like
        RA coordinates (degrees)
    ra2 : array_like
        RA coordinates (degrees)
    dec1 : array_like
        Dec coordinates (degrees)
    dec2 : array_like
        Dec coordinates (degrees)
    bmaj1 : array_like
        Beam major axis (degrees)
    bmaj2 : array_like
        Beam major axis (degrees)
    bmin1 : array_like
        Beam minor axis (degrees)
    bmin2 : array_like
        Beam minor axis (degrees)
    snr1 : array_like
        Signal-to-noise ratios
    snr2 : array_like
        Signal-to-noise ratios

    Returns
    -------
        array_like
        Array of de Ruiter radii.
    """
    # Estimate positional uncertainties from beam and SNR
    # σ ≈ beam / (2 * SNR) (Condon 1997)
    sigma_ra1 = bmaj1 / (2.0 * np.maximum(snr1, 1.0))
    sigma_ra2 = bmaj2 / (2.0 * np.maximum(snr2, 1.0))
    sigma_dec1 = bmin1 / (2.0 * np.maximum(snr1, 1.0))
    sigma_dec2 = bmin2 / (2.0 * np.maximum(snr2, 1.0))

    return calc_de_ruiter(ra1, ra2, dec1, dec2, sigma_ra1, sigma_ra2, sigma_dec1, sigma_dec2)


def join_match_coordinates_sky(
    coords1: SkyCoord, coords2: SkyCoord, seplimit: u.arcsec
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Helper function to perform cross-match using astropy.

    Parameters
    ----------
    coords1 : astropy.coordinates.SkyCoord
        Input coordinates (detected sources)
    coords2 : astropy.coordinates.SkyCoord
        Reference coordinates (catalog sources)
    seplimit : float
        Cross-match radius limit

    Returns
    -------
        tuple
        Tuple of:
        - Indices of coords1 that have matches
        - Indices of coords2 that match coords1
        - Separation distances for matches
        - 3D distances for matches
    """
    idx, separation, dist_3d = match_coordinates_sky(coords1, coords2)
    mask = separation < seplimit
    return (
        np.where(mask)[0],
        idx[mask],
        separation[mask],
        dist_3d[mask],
    )


def cross_match_sources(
    detected_ra: np.ndarray,
    detected_dec: np.ndarray,
    catalog_ra: np.ndarray,
    catalog_dec: np.ndarray,
    radius_arcsec: float = 10.0,
    detected_flux: np.ndarray | None = None,
    catalog_flux: np.ndarray | None = None,
    detected_flux_err: np.ndarray | None = None,
    catalog_flux_err: np.ndarray | None = None,
    detected_ids: np.ndarray | None = None,
    catalog_ids: np.ndarray | None = None,
    *,
    use_de_ruiter: bool = False,
    de_ruiter_limit: float = 5.68,
    detected_sigma_ra: np.ndarray | None = None,
    detected_sigma_dec: np.ndarray | None = None,
    catalog_sigma_ra: np.ndarray | None = None,
    catalog_sigma_dec: np.ndarray | None = None,
) -> pd.DataFrame:
    """General-purpose cross-matching utility with optional de Ruiter matching.

        Matches detected sources with catalog sources using either:
        - Simple nearest-neighbor matching (default, fast)
        - de Ruiter radius matching (uncertainty-weighted, statistically robust)

    Parameters
    ----------
    detected_ra : array_like
        RA of detected sources (degrees)
    detected_dec : array_like
        Dec of detected sources (degrees)
    catalog_ra : array_like
        RA of catalog sources (degrees)
    catalog_dec : array_like
        Dec of catalog sources (degrees)
    radius_arcsec : float, optional
        Matching radius in arcseconds (for simple matching)
    detected_flux : array_like, optional
        Flux of detected sources
    catalog_flux : array_like, optional
        Flux of catalog sources
    detected_flux_err : array_like, optional
        Flux error of detected sources
    catalog_flux_err : array_like, optional
        Flux error of catalog sources
    detected_ids : array_like, optional
        IDs of detected sources
    catalog_ids : array_like, optional
        IDs of catalog sources
    use_de_ruiter : bool, optional
        If True, use de Ruiter radius instead of simple matching.
        Requires positional uncertainties. Default is False.
    de_ruiter_limit : float, optional
        Maximum de Ruiter radius for a valid match.
        Default is 5.68 (3-sigma, 99.7% confidence).
    detected_sigma_ra : array_like, optional
        RA uncertainty of detected sources (degrees).
        Required if use_de_ruiter=True.
    detected_sigma_dec : array_like, optional
        Dec uncertainty of detected sources (degrees).
        Required if use_de_ruiter=True.
    catalog_sigma_ra : array_like, optional
        RA uncertainty of catalog sources (degrees).
        Required if use_de_ruiter=True.
    catalog_sigma_dec : array_like, optional
        Dec uncertainty of catalog sources (degrees).
        Required if use_de_ruiter=True.

    Returns
    -------
        DataFrame
        DataFrame with cross-matched sources containing:
        - detected_idx: Index of detected source
        - catalog_idx: Index of catalog source
        - separation_arcsec: Separation distance (arcsec)
        - dra_arcsec: RA offset (arcsec)
        - ddec_arcsec: Dec offset (arcsec)
        - de_ruiter: de Ruiter radius (if use_de_ruiter=True)
        - detected_flux, catalog_flux: Flux values (if provided)
        - detected_flux_err, catalog_flux_err: Flux errors (if provided)
        - detected_id, catalog_id: Source IDs (if provided)
        - flux_ratio: Flux ratio (if both fluxes provided)
    """
    # Create SkyCoord objects
    detected_coords = SkyCoord(
        detected_ra * u.deg,
        detected_dec * u.deg,
    )
    catalog_coords = SkyCoord(
        catalog_ra * u.deg,
        catalog_dec * u.deg,
    )

    # Perform cross-match (nearest neighbor)
    idx, sep2d, _ = match_coordinates_sky(detected_coords, catalog_coords)
    sep_arcsec = sep2d.to(u.arcsec).value

    if use_de_ruiter:
        # Validate uncertainties are provided
        if any(
            x is None
            for x in [detected_sigma_ra, detected_sigma_dec, catalog_sigma_ra, catalog_sigma_dec]
        ):
            raise ValueError(
                "Positional uncertainties (detected_sigma_ra, detected_sigma_dec, "
                "catalog_sigma_ra, catalog_sigma_dec) are required when use_de_ruiter=True"
            )

        # Calculate de Ruiter radius for all nearest-neighbor matches
        dr = calc_de_ruiter(
            detected_ra,
            catalog_ra[idx],
            detected_dec,
            catalog_dec[idx],
            detected_sigma_ra,
            catalog_sigma_ra[idx],
            detected_sigma_dec,
            catalog_sigma_dec[idx],
        )

        # Filter by de Ruiter radius
        match_mask = dr < de_ruiter_limit
        n_matched = np.sum(match_mask)

        if n_matched == 0:
            logger.warning(f"No sources matched with de Ruiter radius < {de_ruiter_limit}")
            return pd.DataFrame()

        # Build results DataFrame
        results = pd.DataFrame(
            {
                "detected_idx": np.where(match_mask)[0],
                "catalog_idx": idx[match_mask],
                "separation_arcsec": sep_arcsec[match_mask],
                "de_ruiter": dr[match_mask],
            }
        )

        logger.info(f"Cross-matched {n_matched} sources with de Ruiter radius < {de_ruiter_limit}")

    else:
        # Simple radius matching (original behavior)
        match_mask = sep_arcsec < radius_arcsec
        n_matched = np.sum(match_mask)

        if n_matched == 0:
            logger.warning(f"No sources matched within {radius_arcsec} arcsec")
            return pd.DataFrame()

        # Build results DataFrame
        results = pd.DataFrame(
            {
                "detected_idx": np.where(match_mask)[0],
                "catalog_idx": idx[match_mask],
                "separation_arcsec": sep_arcsec[match_mask],
            }
        )

        logger.info(f"Cross-matched {n_matched} sources within {radius_arcsec} arcsec")

    # Calculate RA/Dec offsets
    matched_detected = detected_coords[match_mask]
    matched_catalog = catalog_coords[idx[match_mask]]

    results["dra_arcsec"] = (matched_detected.ra - matched_catalog.ra).to(u.arcsec).value
    results["ddec_arcsec"] = (matched_detected.dec - matched_catalog.dec).to(u.arcsec).value

    # Add flux information if provided
    if detected_flux is not None:
        results["detected_flux"] = detected_flux[results["detected_idx"].values]
    if catalog_flux is not None:
        results["catalog_flux"] = catalog_flux[results["catalog_idx"].values]
    if detected_flux_err is not None:
        results["detected_flux_err"] = detected_flux_err[results["detected_idx"].values]
    if catalog_flux_err is not None:
        results["catalog_flux_err"] = catalog_flux_err[results["catalog_idx"].values]

    # Add IDs if provided
    if detected_ids is not None:
        results["detected_id"] = detected_ids[results["detected_idx"].values]
    if catalog_ids is not None:
        results["catalog_id"] = catalog_ids[results["catalog_idx"].values]

    # Calculate flux ratio if both fluxes provided
    if detected_flux is not None and catalog_flux is not None:
        results["flux_ratio"] = results["detected_flux"] / results["catalog_flux"]

    logger.info(f"Cross-matched {n_matched} sources within {radius_arcsec} arcsec")

    return results


def cross_match_dataframes(
    detected_df: pd.DataFrame,
    catalog_df: pd.DataFrame,
    radius_arcsec: float = 10.0,
    detected_ra_col: str = "ra_deg",
    detected_dec_col: str = "dec_deg",
    catalog_ra_col: str = "ra_deg",
    catalog_dec_col: str = "dec_deg",
    detected_flux_col: str | None = None,
    catalog_flux_col: str | None = None,
    detected_id_col: str | None = None,
    catalog_id_col: str | None = None,
) -> pd.DataFrame:
    """Cross-match two DataFrames containing source positions.

        Convenience wrapper around cross_match_sources for DataFrame inputs.

    Parameters
    ----------
    detected_df : pandas.DataFrame
        DataFrame with detected sources
    catalog_df : pandas.DataFrame
        DataFrame with catalog sources
    radius_arcsec : float
        Matching radius in arcseconds
    detected_ra_col : str
        Column name for detected RA
    detected_dec_col : str
        Column name for detected Dec
    catalog_ra_col : str
        Column name for catalog RA
    catalog_dec_col : str
        Column name for catalog Dec
    detected_flux_col : str or None, optional
        Column name for detected flux (optional)
    catalog_flux_col : str or None, optional
        Column name for catalog flux (optional)
    detected_id_col : str or None, optional
        Column name for detected ID (optional)
    catalog_id_col : str or None, optional
        Column name for catalog ID (optional)

    Returns
    -------
        pandas.DataFrame
        DataFrame with cross-matched sources.
    """
    return cross_match_sources(
        detected_ra=detected_df[detected_ra_col].values,
        detected_dec=detected_df[detected_dec_col].values,
        catalog_ra=catalog_df[catalog_ra_col].values,
        catalog_dec=catalog_df[catalog_dec_col].values,
        radius_arcsec=radius_arcsec,
        detected_flux=(
            detected_df[detected_flux_col].values
            if detected_flux_col and detected_flux_col in detected_df.columns
            else None
        ),
        catalog_flux=(
            catalog_df[catalog_flux_col].values
            if catalog_flux_col and catalog_flux_col in catalog_df.columns
            else None
        ),
        detected_flux_err=(
            detected_df.get("flux_err_jy") if "flux_err_jy" in detected_df.columns else None
        ),
        catalog_flux_err=(
            catalog_df.get("flux_err_mjy") if "flux_err_mjy" in catalog_df.columns else None
        ),
        detected_ids=(
            detected_df[detected_id_col].values
            if detected_id_col and detected_id_col in detected_df.columns
            else None
        ),
        catalog_ids=(
            catalog_df[catalog_id_col].values
            if catalog_id_col and catalog_id_col in catalog_df.columns
            else None
        ),
    )


def calculate_positional_offsets(
    matches_df: pd.DataFrame,
) -> tuple[u.Quantity, u.Quantity, u.Quantity, u.Quantity]:
    """Calculate median positional offsets and MAD between matched sources.

    Parameters
    ----------
    matches_df : pandas.DataFrame
        DataFrame with cross-matched sources containing:
        - dra_arcsec: RA offsets (arcsec)
        - ddec_arcsec: Dec offsets (arcsec)

    Returns
    -------
        tuple
        Tuple of:
        - Median RA offset (Quantity)
        - Median Dec offset (Quantity)
        - MAD of RA offsets (Quantity)
        - MAD of Dec offsets (Quantity)
    """
    dra_median = np.median(matches_df["dra_arcsec"]) * u.arcsec
    dra_madfm = mad_std(matches_df["dra_arcsec"]) * u.arcsec
    ddec_median = np.median(matches_df["ddec_arcsec"]) * u.arcsec
    ddec_madfm = mad_std(matches_df["ddec_arcsec"]) * u.arcsec

    return dra_median, ddec_median, dra_madfm, ddec_madfm


def calculate_flux_scale(
    matches_df: pd.DataFrame,
    flux_ratio_col: str = "flux_ratio",
) -> tuple[AffineScalarFunc, AffineScalarFunc]:
    """Calculate flux scale correction factor.

        Uses median flux ratio as a simple flux scale estimate.
        For robust fitting, see calculate_flux_scale_robust().

    Parameters
    ----------
    matches_df : pandas.DataFrame
        DataFrame with cross-matched sources containing flux_ratio
    flux_ratio_col : str
        Column name for flux ratio

    Returns
    -------
        tuple
        Tuple of:
        - Flux correction factor (multiplicative)
        - Flux correction error
    """
    if flux_ratio_col not in matches_df.columns:
        raise ValueError(f"Column {flux_ratio_col} not found in matches_df")

    flux_ratios = matches_df[flux_ratio_col].values
    flux_ratios = flux_ratios[~np.isnan(flux_ratios)]
    flux_ratios = flux_ratios[flux_ratios > 0]

    if len(flux_ratios) == 0:
        raise ValueError("No valid flux ratios found")

    median_ratio = np.median(flux_ratios)
    mad_ratio = mad_std(flux_ratios)

    # Flux correction is inverse of ratio
    flux_corr = 1.0 / median_ratio
    flux_corr_err = mad_ratio / (median_ratio**2)

    # Ensure std_dev is never exactly 0 to avoid uncertainties warning
    # Use a small epsilon if MAD is zero (all ratios identical)
    if mad_ratio == 0.0:
        mad_ratio = 1e-10  # Small epsilon to avoid zero std_dev warning
        flux_corr_err = max(flux_corr_err, 1e-10)

    return ufloat(flux_corr, flux_corr_err), ufloat(median_ratio, mad_ratio)


def search_around_sky_all(
    coords1: SkyCoord,
    coords2: SkyCoord,
    radius: Angle,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Find all matches within radius (not just nearest neighbor).

        Useful for advanced association methods that need to consider
        multiple potential matches.

    Parameters
    ----------
    coords1 : astropy.coordinates.SkyCoord
        Input coordinates
    coords2 : astropy.coordinates.SkyCoord
        Reference coordinates
    radius : float
        Search radius

    Returns
    -------
        tuple
        Tuple of:
        - Indices of coords1 with matches
        - Indices of coords2 that match
        - Separation distances
    """
    idx1, idx2, sep2d, _ = coords1.search_around_sky(coords2, radius)
    return idx1, idx2, sep2d


def multi_catalog_match(
    detected_ra: np.ndarray,
    detected_dec: np.ndarray,
    catalogs: dict[str, dict[str, np.ndarray]],
    radius_arcsec: float = 10.0,
) -> pd.DataFrame:
    """Match sources against multiple catalogs simultaneously.

    Parameters
    ----------
    detected_ra : array_like
        RA of detected sources (degrees)
    detected_dec : array_like
        Dec of detected sources (degrees)
    catalogs : dict
        Dictionary mapping catalog names to dictionaries containing:
        - 'ra': RA array (degrees)
        - 'dec': Dec array (degrees)
        - 'flux': Flux array (optional)
        - 'id': ID array (optional)
    radius_arcsec : float
        Matching radius in arcseconds

    Returns
    -------
        pandas.DataFrame
        DataFrame with best match for each detected source across all catalogs:
        - detected_idx: Index of detected source
        - best_catalog: Name of catalog with best match
        - best_catalog_idx: Index in best catalog
        - best_separation_arcsec: Best separation distance
        - Additional columns for each catalog with match info
    """
    detected_coords = SkyCoord(detected_ra * u.deg, detected_dec * u.deg)

    results = pd.DataFrame(
        {
            "detected_idx": np.arange(len(detected_ra)),
        }
    )

    best_separations = np.full(len(detected_ra), np.inf)
    best_catalogs = np.full(len(detected_ra), "", dtype=object)
    best_indices = np.full(len(detected_ra), -1, dtype=int)

    # Match against each catalog
    for catalog_name, catalog_data in catalogs.items():
        catalog_coords = SkyCoord(
            catalog_data["ra"] * u.deg,
            catalog_data["dec"] * u.deg,
        )

        idx, sep2d, _ = match_coordinates_sky(detected_coords, catalog_coords)
        sep_arcsec = sep2d.to(u.arcsec).value
        match_mask = sep_arcsec < radius_arcsec

        # Update best matches
        better_mask = sep_arcsec < best_separations
        best_separations[better_mask] = sep_arcsec[better_mask]
        best_catalogs[better_mask] = catalog_name
        best_indices[better_mask] = idx[better_mask]

        # Store match info for this catalog
        results[f"{catalog_name}_matched"] = match_mask
        results[f"{catalog_name}_separation_arcsec"] = sep_arcsec
        results[f"{catalog_name}_idx"] = idx

        if "flux" in catalog_data:
            flux_data = catalog_data["flux"]
            if isinstance(flux_data, (list, np.ndarray)):
                results[f"{catalog_name}_flux"] = np.array(flux_data)[idx]
            else:
                results[f"{catalog_name}_flux"] = flux_data
        if "id" in catalog_data:
            id_data = catalog_data["id"]
            if isinstance(id_data, (list, np.ndarray)):
                results[f"{catalog_name}_id"] = np.array(id_data)[idx]
            else:
                results[f"{catalog_name}_id"] = id_data

    # Add best match columns
    results["best_catalog"] = best_catalogs
    results["best_catalog_idx"] = best_indices
    results["best_separation_arcsec"] = best_separations

    n_matched = np.sum(best_separations < np.inf)
    logger.info(f"Multi-catalog match: {n_matched}/{len(detected_ra)} sources matched")

    return results


def identify_duplicate_catalog_sources(
    catalog_matches: dict[str, pd.DataFrame],
    deduplication_radius_arcsec: float = 2.0,
) -> dict[str, str]:
    """Identify when multiple catalog entries refer to the same physical source.

        This function analyzes matches from multiple catalogs and identifies when
        different catalog entries (e.g., NVSS J123456+012345 and FIRST J123456+012345)
        refer to the same physical source based on their positions.

    Parameters
    ----------
    catalog_matches : dict
        Dictionary mapping catalog names to DataFrames with matches.
        Each DataFrame should contain columns: 'catalog_ra_deg', 'catalog_dec_deg', 'catalog_source_id'
    deduplication_radius_arcsec : float
        Maximum separation to consider sources as duplicates

    Returns
    -------
        dict
        Dictionary mapping catalog entries to master catalog IDs.
    Format: {f"{catalog_type}:{catalog_source_id}": master_catalog_id}
        The master_catalog_id is typically the NVSS ID if available, otherwise
        the FIRST ID, otherwise the RACS ID, or a generated ID.
    """
    import astropy.units as u
    from astropy.coordinates import SkyCoord

    # Collect all catalog entries with their positions
    all_entries = []
    # Map entry index to (catalog_type, catalog_source_id)
    entry_to_catalog = {}

    for catalog_type, matches_df in catalog_matches.items():
        if matches_df is None or len(matches_df) == 0:
            continue

        for idx, row in matches_df.iterrows():
            # Get catalog source position
            if "catalog_ra_deg" in matches_df.columns and "catalog_dec_deg" in matches_df.columns:
                ra = row["catalog_ra_deg"]
                dec = row["catalog_dec_deg"]
            elif "ra_deg" in matches_df.columns and "dra_arcsec" in matches_df.columns:
                # Approximate catalog position from detected position + offset
                ra = row["ra_deg"] - (row.get("dra_arcsec", 0) / 3600.0)
                dec = row["dec_deg"] - (row.get("ddec_arcsec", 0) / 3600.0)
            else:
                continue

            catalog_source_id = row.get("catalog_source_id", f"{catalog_type}_{idx}")
            entry_key = f"{catalog_type}:{catalog_source_id}"

            all_entries.append((ra, dec))
            entry_to_catalog[len(all_entries) - 1] = (catalog_type, catalog_source_id)

    if len(all_entries) == 0:
        return {}

    # Create SkyCoord objects
    all_coords = SkyCoord(
        [e[0] for e in all_entries] * u.deg,
        [e[1] for e in all_entries] * u.deg,
    )

    # Find duplicates using search_around_sky
    from astropy.coordinates import Angle as MatchAngle

    radius = MatchAngle(deduplication_radius_arcsec * u.arcsec)
    idx1, idx2, sep2d, _ = search_around_sky(all_coords, all_coords, radius)

    # Build groups of duplicate entries
    # Use union-find to group entries that are within radius of each other
    parent = list(range(len(all_entries)))

    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py

    # Union entries that are within radius
    for i, j in zip(idx1, idx2):
        if i != j:  # Don't match entry with itself
            union(i, j)

    # Assign master catalog IDs
    master_ids = {}
    groups = {}
    for i in range(len(all_entries)):
        root = find(i)
        if root not in groups:
            groups[root] = []
        groups[root].append(i)

    # For each group, assign master catalog ID based on priority:
    # NVSS > FIRST > RACS > generated
    catalog_priority = {"nvss": 0, "first": 1, "rax": 2}

    for root, group_indices in groups.items():
        if len(group_indices) == 1:
            # Single entry, use its own ID
            idx = group_indices[0]
            catalog_type, catalog_source_id = entry_to_catalog[idx]
            entry_key = f"{catalog_type}:{catalog_source_id}"
            master_ids[entry_key] = f"{catalog_type}:{catalog_source_id}"
        else:
            # Multiple entries, find highest priority catalog
            best_priority = float("inf")
            best_idx = None
            for idx in group_indices:
                catalog_type, catalog_source_id = entry_to_catalog[idx]
                priority = catalog_priority.get(catalog_type.lower(), 999)
                if priority < best_priority:
                    best_priority = priority
                    best_idx = idx

            # Use best catalog entry as master
            catalog_type, catalog_source_id = entry_to_catalog[best_idx]
            master_id = f"{catalog_type}:{catalog_source_id}"

            # Assign master ID to all entries in group
            for idx in group_indices:
                cat_type, cat_source_id = entry_to_catalog[idx]
                entry_key = f"{cat_type}:{cat_source_id}"
                master_ids[entry_key] = master_id

    return master_ids
