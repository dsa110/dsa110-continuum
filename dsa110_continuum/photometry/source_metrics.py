"""
Source-level quality metrics for radio source characterization.

Provides per-source metrics adopted from VAST Pipeline methodology,
including morphological metrics, flux ratios, and spatial statistics.

Reference: askap-vast/vast-pipeline models.py, pipeline/utils.py
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

logger = logging.getLogger(__name__)


@dataclass
class SourceMorphologyMetrics:
    """Morphological metrics for a radio source.

    Attributes
    ----------
    compactness : float
        Ratio of integrated to peak flux (flux_int / flux_peak). Values close to 1 indicate point sources; >1 indicates extended.
    is_resolved : bool
        Whether source is significantly resolved
    a_over_b : float
        Ratio of major to minor axis (ellipticity)
    chi_squared_fit : float
        Chi-squared of Gaussian fit to source
    reduced_chi_squared : float
        Reduced chi-squared (chi2 / dof)
    """

    compactness: float
    is_resolved: bool = False
    a_over_b: float | None = None
    chi_squared_fit: float | None = None
    reduced_chi_squared: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "compactness": self.compactness,
            "is_resolved": self.is_resolved,
            "a_over_b": self.a_over_b,
            "chi_squared_fit": self.chi_squared_fit,
            "reduced_chi_squared": self.reduced_chi_squared,
        }


@dataclass
class IslandMetrics:
    """Island (connected component) metrics for a source.

        In radio source finding, an "island" is a connected region of pixels
        above the detection threshold. A single island may contain multiple
        fitted components (sources).

    Attributes
    ----------
    flux_int_isl_ratio : float
        Ratio of component integrated flux to total island flux
    flux_peak_isl_ratio : float
        Ratio of component peak flux to island peak flux
    has_siblings : bool
        Whether source has other components in same island
    n_siblings : int
        Number of sibling components
    island_id : Any
        Island identifier (if available)
    total_island_flux : float
        Total integrated flux of island
    """

    flux_int_isl_ratio: float = 1.0
    flux_peak_isl_ratio: float = 1.0
    has_siblings: bool = False
    n_siblings: int = 0
    island_id: str | None = None
    total_island_flux: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "flux_int_isl_ratio": self.flux_int_isl_ratio,
            "flux_peak_isl_ratio": self.flux_peak_isl_ratio,
            "has_siblings": self.has_siblings,
            "n_siblings": self.n_siblings,
            "island_id": self.island_id,
            "total_island_flux": self.total_island_flux,
        }


@dataclass
class SpatialMetrics:
    """Spatial relationship metrics for a source.

    Attributes
    ----------
    n_neighbour_dist : float
        Distance to nearest neighbor (degrees)
    n_neighbours_3arcmin : int
        Number of neighbors within 3 arcmin
    n_neighbours_1arcmin : int
        Number of neighbors within 1 arcmin
    is_isolated : bool
        Whether source is isolated (no neighbors within threshold)
    crowding_parameter : float
        Local source density metric
    """

    n_neighbour_dist: float  # degrees
    n_neighbours_3arcmin: int = 0
    n_neighbours_1arcmin: int = 0
    is_isolated: bool = True
    crowding_parameter: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "n_neighbour_dist_deg": self.n_neighbour_dist,
            "n_neighbour_dist_arcsec": self.n_neighbour_dist * 3600,
            "n_neighbours_3arcmin": self.n_neighbours_3arcmin,
            "n_neighbours_1arcmin": self.n_neighbours_1arcmin,
            "is_isolated": self.is_isolated,
            "crowding_parameter": self.crowding_parameter,
        }


@dataclass
class SourceQAMetrics:
    """Complete source-level QA metrics.

        Combines all source metrics into a single container.

    Attributes
    ----------
    source_id : Any
        Source identifier
    ra_deg : float
        Right ascension (degrees)
    dec_deg : float
        Declination (degrees)
    flux_peak : float
        Peak flux (Jy/beam)
    flux_int : float
        Integrated flux (Jy)
    local_rms : float
        Local RMS at source position (Jy/beam)
    snr : float
        Signal-to-noise ratio (flux_peak / local_rms)
    morphology : SourceMorphologyMetrics
        Morphological metrics
    island : IslandMetrics
        Island/component metrics
    spatial : SpatialMetrics
        Spatial relationship metrics
    """

    source_id: str
    ra_deg: float
    dec_deg: float
    flux_peak: float
    flux_int: float
    local_rms: float
    snr: float
    morphology: SourceMorphologyMetrics | None = None
    island: IslandMetrics | None = None
    spatial: SpatialMetrics | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "source_id": self.source_id,
            "ra_deg": self.ra_deg,
            "dec_deg": self.dec_deg,
            "flux_peak": self.flux_peak,
            "flux_int": self.flux_int,
            "local_rms": self.local_rms,
            "snr": self.snr,
            "morphology": self.morphology.to_dict() if self.morphology else None,
            "island": self.island.to_dict() if self.island else None,
            "spatial": self.spatial.to_dict() if self.spatial else None,
        }


def calculate_compactness(
    flux_int: float,
    flux_peak: float,
) -> float:
    """Calculate source compactness (integrated / peak flux).

        Adopted from VAST Pipeline: pipeline/utils.py

        Compactness indicates source morphology:
        - ~1.0: Point source (unresolved)
        - >1.0: Extended source (resolved)
        - <1.0: Indicates fitting issues (should be rare)

    Parameters
    ----------
    flux_int : float
        Integrated flux (Jy)
    flux_peak : float
        Peak flux (Jy/beam)

    Returns
    -------
        float
        Compactness ratio

    Raises
    ------
        ValueError
        If peak flux is zero or negative
    """
    if flux_peak <= 0:
        raise ValueError("Peak flux must be positive")

    return flux_int / flux_peak


def calculate_snr(
    flux_peak: float,
    local_rms: float,
) -> float:
    """Calculate signal-to-noise ratio.

    Parameters
    ----------
    flux_peak : float
        Peak flux (Jy/beam)
    local_rms : float
        Local RMS noise (Jy/beam)

    Returns
    -------
        float
        SNR value

    Raises
    ------
        ValueError
        If local_rms is zero or negative
    """
    if local_rms <= 0:
        raise ValueError("Local RMS must be positive")

    return flux_peak / local_rms


def calculate_island_flux_ratios(
    component_flux_int: float,
    component_flux_peak: float,
    island_flux_int: float,
    island_flux_peak: float,
) -> tuple[float, float]:
    """Calculate island flux ratios for a component.

        Adopted from VAST Pipeline: models.py Measurement model

        These ratios indicate what fraction of island flux belongs to this component.
        Values significantly less than 1.0 indicate blended sources.

    Parameters
    ----------
    component_flux_int : float
        Component integrated flux
    component_flux_peak : float
        Component peak flux
    island_flux_int : float
        Total island integrated flux
    island_flux_peak : float
        Total island peak flux

    Returns
    -------
        tuple of float
        Tuple of (flux_int_isl_ratio, flux_peak_isl_ratio)
    """
    flux_int_ratio = component_flux_int / island_flux_int if island_flux_int > 0 else 1.0
    flux_peak_ratio = component_flux_peak / island_flux_peak if island_flux_peak > 0 else 1.0

    return flux_int_ratio, flux_peak_ratio


def compute_morphology_metrics(
    flux_peak: float,
    flux_int: float,
    major_arcsec: float | None = None,
    minor_arcsec: float | None = None,
    beam_major_arcsec: float | None = None,
    beam_minor_arcsec: float | None = None,
    chi_squared: float | None = None,
    dof: int | None = None,
) -> SourceMorphologyMetrics:
    """Compute morphological metrics for a source.

    Parameters
    ----------
    flux_peak : float
        Peak flux (Jy/beam)
    flux_int : float
        Integrated flux (Jy)
    major_arcsec : float
        Fitted major axis (arcsec)
    minor_arcsec : float
        Fitted minor axis (arcsec)
    beam_major_arcsec : float
        Beam major axis (arcsec)
    beam_minor_arcsec : float
        Beam minor axis (arcsec)
    chi_squared : float
        Chi-squared of Gaussian fit
    dof : int
        Degrees of freedom

    Returns
    -------
        SourceMorphologyMetrics
    """
    compactness = calculate_compactness(flux_int, flux_peak)

    # Determine if resolved
    is_resolved = compactness > 1.2  # 20% larger than point source

    # Compute axis ratio if available
    a_over_b = None
    if major_arcsec is not None and minor_arcsec is not None and minor_arcsec > 0:
        a_over_b = major_arcsec / minor_arcsec

    # Also check if deconvolved size > beam (indicates resolved)
    if major_arcsec and beam_major_arcsec:
        if major_arcsec > beam_major_arcsec * 1.1:  # 10% larger than beam
            is_resolved = True

    # Reduced chi-squared
    reduced_chi2 = None
    if chi_squared is not None and dof is not None and dof > 0:
        reduced_chi2 = chi_squared / dof

    return SourceMorphologyMetrics(
        compactness=compactness,
        is_resolved=is_resolved,
        a_over_b=a_over_b,
        chi_squared_fit=chi_squared,
        reduced_chi_squared=reduced_chi2,
    )


def compute_island_metrics(
    component_flux_int: float,
    component_flux_peak: float,
    island_components: list[dict[str, float]],
    island_id: str | None = None,
) -> IslandMetrics:
    """Compute island metrics for a source component.

    Parameters
    ----------
    component_flux_int : float
        This component's integrated flux
    component_flux_peak : float
        This component's peak flux
    island_components : list of dict
        List of all components in island, each with 'flux_int' and 'flux_peak' keys
    island_id : Any, optional
        Optional island identifier

    Returns
    -------
        IslandMetrics
    """
    n_components = len(island_components)
    has_siblings = n_components > 1
    n_siblings = n_components - 1 if n_components > 1 else 0

    # Calculate total island flux
    island_flux_int = sum(c.get("flux_int", 0) for c in island_components)
    island_flux_peak = max((c.get("flux_peak", 0) for c in island_components), default=0)

    # Calculate ratios
    flux_int_ratio, flux_peak_ratio = calculate_island_flux_ratios(
        component_flux_int, component_flux_peak, island_flux_int, island_flux_peak
    )

    return IslandMetrics(
        flux_int_isl_ratio=flux_int_ratio,
        flux_peak_isl_ratio=flux_peak_ratio,
        has_siblings=has_siblings,
        n_siblings=n_siblings,
        island_id=island_id,
        total_island_flux=island_flux_int,
    )


def compute_nearest_neighbor_distance(
    ra_deg: float,
    dec_deg: float,
    all_ra_deg: np.ndarray,
    all_dec_deg: np.ndarray,
) -> float:
    """Compute distance to nearest neighbor.

        Adopted from VAST Pipeline: Source.n_neighbour_dist

    Parameters
    ----------
    ra_deg : float
        RA of target source
    dec_deg : float
        Dec of target source
    all_ra_deg : array-like
        RA of all sources (including target)
    all_dec_deg : array-like
        Dec of all sources (including target)

    Returns
    -------
        float
        Distance to nearest neighbor in degrees
    """
    if len(all_ra_deg) < 2:
        return float("inf")

    # Convert to 3D Cartesian for spherical distance
    cos_dec = np.cos(np.radians(all_dec_deg))
    x = cos_dec * np.cos(np.radians(all_ra_deg))
    y = cos_dec * np.sin(np.radians(all_ra_deg))
    z = np.sin(np.radians(all_dec_deg))

    coords_3d = np.column_stack([x, y, z])

    # Build KD-tree
    tree = cKDTree(coords_3d)

    # Query for this source
    target_cos_dec = np.cos(np.radians(dec_deg))
    target_x = target_cos_dec * np.cos(np.radians(ra_deg))
    target_y = target_cos_dec * np.sin(np.radians(ra_deg))
    target_z = np.sin(np.radians(dec_deg))
    target_coord = np.array([target_x, target_y, target_z])

    # Get 2 nearest (first is self)
    distances, _ = tree.query(target_coord, k=min(2, len(all_ra_deg)))

    if len(distances) < 2:
        return float("inf")

    # Convert chord distance back to angular distance
    # chord = 2 * sin(theta/2), so theta = 2 * arcsin(chord/2)
    chord_distance = distances[1]  # Skip self
    angular_dist_rad = 2 * np.arcsin(min(chord_distance / 2, 1.0))

    return float(np.degrees(angular_dist_rad))


def compute_spatial_metrics(
    ra_deg: float,
    dec_deg: float,
    all_ra_deg: np.ndarray,
    all_dec_deg: np.ndarray,
    isolation_threshold_arcmin: float = 1.0,
) -> SpatialMetrics:
    """Compute spatial relationship metrics.

    Parameters
    ----------
    ra_deg : float
        RA of target source
    dec_deg : float
        Dec of target source
    all_ra_deg : array-like
        RA of all sources
    all_dec_deg : array-like
        Dec of all sources
    isolation_threshold_arcmin : float
        Threshold for isolation (arcmin)

    Returns
    -------
        SpatialMetrics
    """
    n_neighbour_dist = compute_nearest_neighbor_distance(ra_deg, dec_deg, all_ra_deg, all_dec_deg)

    # Count neighbors within thresholds
    # Use simple angular separation for counting
    cos_dec = np.cos(np.radians(dec_deg))
    delta_ra = (all_ra_deg - ra_deg) * cos_dec
    delta_dec = all_dec_deg - dec_deg
    sep_deg = np.sqrt(delta_ra**2 + delta_dec**2)

    # Exclude self (separation ~ 0)
    sep_deg = sep_deg[sep_deg > 1e-10]

    n_neighbours_3arcmin = int(np.sum(sep_deg < 3.0 / 60))
    n_neighbours_1arcmin = int(np.sum(sep_deg < 1.0 / 60))

    is_isolated = n_neighbour_dist > (isolation_threshold_arcmin / 60)

    # Crowding parameter: sources per square arcmin within 3 arcmin
    area_3arcmin = np.pi * 3.0**2  # square arcmin
    crowding = n_neighbours_3arcmin / area_3arcmin if area_3arcmin > 0 else 0

    return SpatialMetrics(
        n_neighbour_dist=n_neighbour_dist,
        n_neighbours_3arcmin=n_neighbours_3arcmin,
        n_neighbours_1arcmin=n_neighbours_1arcmin,
        is_isolated=is_isolated,
        crowding_parameter=crowding,
    )


def compute_source_qa_metrics(
    source_id: str,
    ra_deg: float,
    dec_deg: float,
    flux_peak: float,
    flux_int: float,
    local_rms: float,
    *,
    major_arcsec: float | None = None,
    minor_arcsec: float | None = None,
    beam_major_arcsec: float | None = None,
    beam_minor_arcsec: float | None = None,
    chi_squared: float | None = None,
    dof: int | None = None,
    island_components: list[dict[str, float]] | None = None,
    island_id: str | None = None,
    all_source_coords: tuple[np.ndarray, np.ndarray] | None = None,
) -> SourceQAMetrics:
    """Compute complete source QA metrics.

    Parameters
    ----------
    source_id : str
        Source identifier.
    ra_deg : float
        Right ascension (degrees).
    dec_deg : float
        Declination (degrees).
    flux_peak : float
        Peak flux (Jy/beam).
    flux_int : float
        Integrated flux (Jy).
    local_rms : float
        Local RMS noise (Jy/beam).
    major_arcsec : float, optional
        Fitted major axis.
    minor_arcsec : float, optional
        Fitted minor axis.
    beam_major_arcsec : float, optional
        Beam major axis.
    beam_minor_arcsec : float, optional
        Beam minor axis.
    chi_squared : float, optional
        Chi-squared of fit.
    dof : int, optional
        Degrees of freedom.
    island_components : list of dict, optional
        List of components in same island.
    island_id : str, optional
        Island identifier.
    all_source_coords : tuple of (np.ndarray, np.ndarray), optional
        Tuple of (all_ra, all_dec) for neighbor calculations.

    Returns
    -------
    SourceQAMetrics
        All computed metrics.
    """
    # Calculate SNR
    snr = calculate_snr(flux_peak, local_rms)

    # Morphology metrics
    morphology = compute_morphology_metrics(
        flux_peak=flux_peak,
        flux_int=flux_int,
        major_arcsec=major_arcsec,
        minor_arcsec=minor_arcsec,
        beam_major_arcsec=beam_major_arcsec,
        beam_minor_arcsec=beam_minor_arcsec,
        chi_squared=chi_squared,
        dof=dof,
    )

    # Island metrics
    island = None
    if island_components is not None:
        island = compute_island_metrics(
            component_flux_int=flux_int,
            component_flux_peak=flux_peak,
            island_components=island_components,
            island_id=island_id,
        )

    # Spatial metrics
    spatial = None
    if all_source_coords is not None:
        all_ra, all_dec = all_source_coords
        spatial = compute_spatial_metrics(
            ra_deg=ra_deg,
            dec_deg=dec_deg,
            all_ra_deg=all_ra,
            all_dec_deg=all_dec,
        )

    return SourceQAMetrics(
        source_id=source_id,
        ra_deg=ra_deg,
        dec_deg=dec_deg,
        flux_peak=flux_peak,
        flux_int=flux_int,
        local_rms=local_rms,
        snr=snr,
        morphology=morphology,
        island=island,
        spatial=spatial,
    )


def batch_compute_source_metrics(
    sources: list[dict[str, Any]],
    beam_major_arcsec: float | None = None,
    beam_minor_arcsec: float | None = None,
) -> list[SourceQAMetrics]:
    """Batch compute metrics for multiple sources.

        More efficient than computing one at a time because neighbor
        distances can be computed using a single KD-tree.

    Parameters
    ----------
    sources : list of dict
        List of source dicts with keys:
        - source_id, ra_deg, dec_deg
        - flux_peak, flux_int, local_rms
        - Optional: major_arcsec, minor_arcsec, chi_squared, dof
        beam_major_arcsec :
        Beam major axis (arcsec)
        beam_minor_arcsec :
        Beam minor axis (arcsec)

    Returns
    -------
        list of SourceQAMetrics
    """
    if not sources:
        return []

    # Extract all coordinates for neighbor calculation
    all_ra = np.array([s["ra_deg"] for s in sources])
    all_dec = np.array([s["dec_deg"] for s in sources])
    all_coords = (all_ra, all_dec)

    results = []
    for src in sources:
        metrics = compute_source_qa_metrics(
            source_id=src.get("source_id", ""),
            ra_deg=src["ra_deg"],
            dec_deg=src["dec_deg"],
            flux_peak=src["flux_peak"],
            flux_int=src.get("flux_int", src["flux_peak"]),
            local_rms=src["local_rms"],
            major_arcsec=src.get("major_arcsec"),
            minor_arcsec=src.get("minor_arcsec"),
            beam_major_arcsec=beam_major_arcsec,
            beam_minor_arcsec=beam_minor_arcsec,
            chi_squared=src.get("chi_squared"),
            dof=src.get("dof"),
            all_source_coords=all_coords,
        )
        results.append(metrics)

    return results
