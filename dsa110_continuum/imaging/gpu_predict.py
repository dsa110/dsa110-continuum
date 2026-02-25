"""
GPU-accelerated sky model visibility prediction.

This module provides high-level functions for predicting visibilities from
sky models using GPU-accelerated degridding. It integrates with the
calibration pipeline to enable multi-source sky model gain calibration.

Workflow:
    1. Query source catalog for bright sources in field
    2. Look up spectral indices from master_sources or VLA calibrators
    3. Render sources to sky model image
    4. FFT to UV plane
    5. Degrid at MS UVW coordinates
    6. Write to MODEL_DATA column

Performance:
    - GPU degridding: ~30-60× faster than WSClean -predict
    - Typical prediction: ~160ms for 1M visibilities
    - W-projection with 32 planes handles DSA-110 ±200λ w-range

Spectral Index Lookup Hierarchy:
    1. Use 'alpha' column from source dict if present
    2. Query master_sources.sqlite3 by position (0.5" crossmatch)
    3. Query VLA calibrators multi-band fluxes (L/C-band) and compute α
    4. Skip source with warning if no spectral index available
"""

from __future__ import annotations

import logging
import os
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from dsa110_contimg.core.imaging.gpu_gridding import (
    DegridConfig,
    DegridResult,
    gpu_degrid_visibilities,
)

logger = logging.getLogger(__name__)

# Default catalog paths
_DEFAULT_MASTER_DB = Path(
    os.environ.get(
        "CONTIMG_STATE_DIR",
        "/data/dsa110-contimg/state",
    )
) / "catalogs" / "master_sources.sqlite3"

_DEFAULT_VLA_DB = Path(
    os.environ.get(
        "CAL_CATALOG_DB",
        "/data/dsa110-contimg/state/catalogs/vla_calibrators.sqlite3",
    )
)


@dataclass
class SourceModel:
    """Point source model for sky prediction.

    Parameters
    ----------
    ra_deg : float
        Right ascension in degrees
    dec_deg : float
        Declination in degrees
    flux_jy : float
        Flux density in Jy (Stokes I)
    name : str, optional
        Source name for logging
    spectral_index : float, optional
        Spectral index α where S ∝ ν^α (default 0.0 = flat spectrum).
        Typical synchrotron: -0.7 to -0.8
    """

    ra_deg: float
    dec_deg: float
    flux_jy: float
    name: str = ""
    spectral_index: float = 0.0


@dataclass
class PredictConfig:
    """Configuration for visibility prediction.

    Parameters
    ----------
    image_size : int
        Image size in pixels (default 512)
    cell_size_arcsec : float
        Cell size in arcseconds (default 12.0)
    phase_center_ra : float
        Phase center RA in degrees
    phase_center_dec : float
        Phase center Dec in degrees
    gpu_id : int
        GPU device ID (default 0)
    use_w_projection : bool
        Enable W-projection (default True)
    w_planes : int
        Number of W-projection planes (default 32)
    w_max : float
        Maximum |w| in wavelengths (default 200.0)
    max_gpu_gb : float
        Maximum GPU memory to use in GB (default 4.0).
        Prediction uses smaller images (512²) so 4 GB is typically sufficient.
    """

    image_size: int = 512
    cell_size_arcsec: float = 12.0
    phase_center_ra: float = 0.0
    phase_center_dec: float = 0.0
    gpu_id: int = 0
    use_w_projection: bool = True
    w_planes: int = 32
    w_max: float = 200.0
    max_gpu_gb: float = 4.0


@dataclass
class PredictResult:
    """Result of visibility prediction.

    Parameters
    ----------
    vis_model : np.ndarray | None
        Complex model visibilities shape (nrows, nchan, npol)
    n_sources : int
        Number of sources in sky model
    n_vis : int
        Number of visibilities predicted
    processing_time_s : float
        Total processing time
    error : str | None
        Error message if failed
    """

    vis_model: np.ndarray | None = None
    n_sources: int = 0
    n_vis: int = 0
    processing_time_s: float = 0.0
    error: str | None = None

    @property
    def success(self) -> bool:
        """Check if prediction completed successfully."""
        return self.error is None and self.vis_model is not None


# ============================================================================
# Catalog Source Adapter with Spectral Index Lookup
# ============================================================================


class CatalogSourceAdapter:
    """Adapter to normalize catalog source dictionaries with spectral index lookup.

    This class provides factory methods to convert various catalog source formats
    into SourceModel instances, with automatic spectral index lookup from:
    1. 'alpha' column in source dict (if present)
    2. master_sources.sqlite3 crossmatch (0.5" radius)
    3. VLA calibrators multi-band fluxes (L/C-band → compute α)

    Sources without spectral index are skipped with a warning.

    Parameters
    ----------
    master_db : Path, optional
        Path to master_sources.sqlite3 (default: state/catalogs/master_sources.sqlite3)
    vla_db : Path, optional
        Path to vla_calibrators.sqlite3 (default: state/catalogs/vla_calibrators.sqlite3)
    crossmatch_radius_arcsec : float
        Crossmatch radius in arcseconds (default 0.5)
    """

    # VLA band frequencies for spectral index calculation
    BAND_FREQ_GHZ = {
        "20cm": 1.4,
        "6cm": 5.0,
        "3.7cm": 8.4,
    }

    def __init__(
        self,
        master_db: Path | None = None,
        vla_db: Path | None = None,
        crossmatch_radius_arcsec: float = 0.5,
    ):
        self.master_db = master_db or _DEFAULT_MASTER_DB
        self.vla_db = vla_db or _DEFAULT_VLA_DB
        self.crossmatch_radius_arcsec = crossmatch_radius_arcsec

        # Cache for VLA calibrator fluxes (name -> {band: flux_jy})
        self._vla_flux_cache: dict[str, dict[str, float]] = {}

        # Statistics
        self.n_from_dict = 0
        self.n_from_master = 0
        self.n_from_vla = 0
        self.n_skipped = 0

    def from_dict(self, source: dict[str, Any]) -> SourceModel | None:
        """Convert a single source dictionary to SourceModel with spectral index.

        Normalizes key variations:
        - 'ra' or 'ra_deg' → ra_deg
        - 'dec' or 'dec_deg' → dec_deg
        - 'flux_mjy' or 'flux_jy' → flux_jy (converts mJy to Jy)
        - 'alpha' or 'spectral_index' → spectral_index

        Parameters
        ----------
        source : dict
            Source dictionary with coordinate and flux keys

        Returns
        -------
        SourceModel | None
            SourceModel if spectral index is available, None otherwise
        """
        # Normalize RA
        ra_deg = source.get("ra_deg") or source.get("ra")
        if ra_deg is None:
            logger.warning("Source missing RA: %s", source)
            return None

        # Normalize Dec
        dec_deg = source.get("dec_deg") or source.get("dec")
        if dec_deg is None:
            logger.warning("Source missing Dec: %s", source)
            return None

        # Normalize flux (convert mJy to Jy if needed)
        flux_jy = source.get("flux_jy")
        if flux_jy is None:
            flux_mjy = source.get("flux_mjy", 0)
            flux_jy = flux_mjy / 1000.0
        if flux_jy <= 0:
            return None

        # Get source name
        name = source.get("name", "")

        # Spectral index lookup hierarchy
        spectral_index = self._get_spectral_index(source, ra_deg, dec_deg, name)

        if spectral_index is None:
            self.n_skipped += 1
            logger.warning(
                "Skipping source %s at (%.4f, %.4f): no spectral index available",
                name or "unnamed",
                ra_deg,
                dec_deg,
            )
            return None

        return SourceModel(
            ra_deg=float(ra_deg),
            dec_deg=float(dec_deg),
            flux_jy=float(flux_jy),
            name=name,
            spectral_index=float(spectral_index),
        )

    def _get_spectral_index(
        self,
        source: dict[str, Any],
        ra_deg: float,
        dec_deg: float,
        name: str,
    ) -> float | None:
        """Look up spectral index using hierarchy: dict → master_sources → VLA.

        Returns
        -------
        float | None
            Spectral index α, or None if not available
        """
        # 1. Check source dict for 'alpha' or 'spectral_index'
        alpha = source.get("alpha") or source.get("spectral_index")
        if alpha is not None and not np.isnan(alpha):
            self.n_from_dict += 1
            return float(alpha)

        # 2. Query master_sources.sqlite3 by position
        alpha = self._query_master_sources(ra_deg, dec_deg)
        if alpha is not None:
            self.n_from_master += 1
            return alpha

        # 3. Query VLA calibrators for multi-band fluxes
        alpha = self._query_vla_calibrators(ra_deg, dec_deg, name)
        if alpha is not None:
            self.n_from_vla += 1
            return alpha

        return None

    def _query_master_sources(self, ra_deg: float, dec_deg: float) -> float | None:
        """Query master_sources.sqlite3 for spectral index by position crossmatch."""
        if not self.master_db.exists():
            return None

        # Convert crossmatch radius to degrees
        radius_deg = self.crossmatch_radius_arcsec / 3600.0

        try:
            with sqlite3.connect(str(self.master_db), timeout=10.0) as conn:
                # Positional crossmatch with small box query
                row = conn.execute(
                    """
                    SELECT alpha
                    FROM sources
                    WHERE ra_deg BETWEEN ? AND ?
                      AND dec_deg BETWEEN ? AND ?
                      AND alpha IS NOT NULL
                    ORDER BY ABS(ra_deg - ?) + ABS(dec_deg - ?)
                    LIMIT 1
                    """,
                    (
                        ra_deg - radius_deg,
                        ra_deg + radius_deg,
                        dec_deg - radius_deg,
                        dec_deg + radius_deg,
                        ra_deg,
                        dec_deg,
                    ),
                ).fetchone()

                if row and row[0] is not None and not np.isnan(row[0]):
                    return float(row[0])

        except (sqlite3.Error, OSError) as e:
            logger.debug("Error querying master_sources: %s", e)

        return None

    def _query_vla_calibrators(
        self, ra_deg: float, dec_deg: float, name: str
    ) -> float | None:
        """Query VLA calibrators for multi-band fluxes and compute spectral index."""
        if not self.vla_db.exists():
            return None

        # Convert crossmatch radius to degrees
        radius_deg = self.crossmatch_radius_arcsec / 3600.0

        try:
            with sqlite3.connect(str(self.vla_db), timeout=10.0) as conn:
                conn.row_factory = sqlite3.Row

                # First try name match if provided
                calibrator_name = None
                if name:
                    row = conn.execute(
                        "SELECT name FROM calibrators WHERE name = ? OR alt_name = ?",
                        (name, name),
                    ).fetchone()
                    if row:
                        calibrator_name = row["name"]

                # If no name match, try positional match
                if not calibrator_name:
                    row = conn.execute(
                        """
                        SELECT name FROM calibrators
                        WHERE ra_deg BETWEEN ? AND ?
                          AND dec_deg BETWEEN ? AND ?
                        ORDER BY ABS(ra_deg - ?) + ABS(dec_deg - ?)
                        LIMIT 1
                        """,
                        (
                            ra_deg - radius_deg,
                            ra_deg + radius_deg,
                            dec_deg - radius_deg,
                            dec_deg + radius_deg,
                            ra_deg,
                            dec_deg,
                        ),
                    ).fetchone()
                    if row:
                        calibrator_name = row["name"]

                if not calibrator_name:
                    return None

                # Check cache
                if calibrator_name in self._vla_flux_cache:
                    fluxes = self._vla_flux_cache[calibrator_name]
                else:
                    # Query multi-band fluxes
                    rows = conn.execute(
                        """
                        SELECT band, flux_jy FROM fluxes
                        WHERE name = ? AND flux_jy > 0
                        """,
                        (calibrator_name,),
                    ).fetchall()

                    fluxes = {row["band"]: float(row["flux_jy"]) for row in rows}
                    self._vla_flux_cache[calibrator_name] = fluxes

                # Calculate spectral index from L-band (1.4 GHz) and C-band (5 GHz)
                flux_L = fluxes.get("20cm")
                flux_C = fluxes.get("6cm")

                if flux_L and flux_C and flux_L > 0 and flux_C > 0:
                    # α = log(S2/S1) / log(ν2/ν1)
                    from dsa110_contimg.core.catalog.spectral_index import (
                        calculate_spectral_index,
                    )

                    alpha, _ = calculate_spectral_index(
                        freq1_ghz=1.4,
                        freq2_ghz=5.0,
                        flux1_mjy=flux_L * 1000,  # Convert Jy to mJy
                        flux2_mjy=flux_C * 1000,
                    )
                    if not np.isnan(alpha):
                        logger.debug(
                            "VLA calibrator %s: L=%.2f Jy, C=%.2f Jy → α=%.2f",
                            calibrator_name,
                            flux_L,
                            flux_C,
                            alpha,
                        )
                        return alpha

        except (sqlite3.Error, OSError, ImportError) as e:
            logger.debug("Error querying VLA calibrators: %s", e)

        return None

    def from_dict_list(
        self,
        sources: list[dict[str, Any]],
        *,
        min_flux_jy: float = 0.0,
        max_sources: int | None = None,
    ) -> list[SourceModel]:
        """Convert list of source dictionaries to SourceModel list.

        Sources without spectral index are skipped with a warning.

        Parameters
        ----------
        sources : list[dict]
            List of source dictionaries
        min_flux_jy : float
            Minimum flux threshold in Jy
        max_sources : int, optional
            Maximum number of sources to return

        Returns
        -------
        list[SourceModel]
            List of SourceModel instances with spectral indices
        """
        result = []

        for src_dict in sources:
            # Apply flux threshold before conversion
            flux_jy = src_dict.get("flux_jy") or (src_dict.get("flux_mjy", 0) / 1000.0)
            if flux_jy < min_flux_jy:
                continue

            model = self.from_dict(src_dict)
            if model is not None:
                result.append(model)

            if max_sources and len(result) >= max_sources:
                break

        # Sort by flux (brightest first)
        result.sort(key=lambda s: s.flux_jy, reverse=True)

        return result

    def get_statistics(self) -> dict[str, int]:
        """Return lookup statistics."""
        return {
            "n_from_dict": self.n_from_dict,
            "n_from_master": self.n_from_master,
            "n_from_vla": self.n_from_vla,
            "n_skipped": self.n_skipped,
            "n_total": self.n_from_dict + self.n_from_master + self.n_from_vla,
        }


def render_sources_to_image(
    sources: list[SourceModel],
    config: PredictConfig,
) -> np.ndarray:
    """Render point sources to a sky model image.

    Creates a sky model image with point sources at their (l,m) positions
    relative to the phase center. Sources are delta functions (single pixels).

    Parameters
    ----------
    sources : list[SourceModel]
        List of point sources
    config : PredictConfig
        Prediction configuration

    Returns
    -------
    np.ndarray
        (image_size, image_size) sky model image in Jy/pixel
    """
    image = np.zeros((config.image_size, config.image_size), dtype=np.float64)

    cell_rad = config.cell_size_arcsec * np.pi / (180.0 * 3600.0)
    center = config.image_size // 2

    # Phase center in radians
    ra0 = np.deg2rad(config.phase_center_ra)
    dec0 = np.deg2rad(config.phase_center_dec)

    for src in sources:
        # Convert RA/Dec to l,m relative to phase center
        ra = np.deg2rad(src.ra_deg)
        dec = np.deg2rad(src.dec_deg)

        # l,m direction cosines
        # l = cos(dec) * sin(ra - ra0)
        # m = sin(dec) * cos(dec0) - cos(dec) * sin(dec0) * cos(ra - ra0)
        dra = ra - ra0
        l = np.cos(dec) * np.sin(dra)
        m = np.sin(dec) * np.cos(dec0) - np.cos(dec) * np.sin(dec0) * np.cos(dra)

        # Convert to pixel coordinates
        # l increases to the left (East), m increases up (North)
        x_pix = center - int(round(l / cell_rad))  # Note: l increases left
        y_pix = center + int(round(m / cell_rad))

        # Bounds check
        if 0 <= x_pix < config.image_size and 0 <= y_pix < config.image_size:
            image[y_pix, x_pix] += src.flux_jy
        else:
            logger.warning(
                "Source %s at l=%.4f, m=%.4f outside image bounds",
                src.name or f"({src.ra_deg:.4f}, {src.dec_deg:.4f})",
                l,
                m,
            )

    return image


def predict_visibilities_gpu(
    uvw: np.ndarray,
    sources: list[SourceModel],
    config: PredictConfig,
) -> DegridResult:
    """Predict visibilities from sky model using GPU degridding.

    Parameters
    ----------
    uvw : np.ndarray
        (N, 3) UVW coordinates in wavelengths
    sources : list[SourceModel]
        List of point sources
    config : PredictConfig
        Prediction configuration

    Returns
    -------
    DegridResult
        Result containing predicted visibilities
    """
    # Render sources to image
    sky_image = render_sources_to_image(sources, config)

    # Configure degridding
    degrid_config = DegridConfig(
        image_size=config.image_size,
        cell_size_arcsec=config.cell_size_arcsec,
        gpu_id=config.gpu_id,
        use_w_projection=config.use_w_projection,
        w_planes=config.w_planes,
        w_max=config.w_max,
    )

    # Degrid
    return gpu_degrid_visibilities(uvw, sky_image, config=degrid_config)


def predict_model_for_ms(
    ms_path: str | Path,
    sources: list[SourceModel],
    config: PredictConfig,
    *,
    write_model: bool = True,
    channel_average: bool = True,
) -> PredictResult:
    """Predict model visibilities for a measurement set.

    Reads UVW coordinates from the MS, predicts visibilities using GPU
    degridding, and optionally writes to MODEL_DATA column.

    Parameters
    ----------
    ms_path : str | Path
        Path to measurement set
    sources : list[SourceModel]
        List of point sources
    config : PredictConfig
        Prediction configuration
    write_model : bool
        Write predicted visibilities to MODEL_DATA (default True)
    channel_average : bool
        Predict single channel and broadcast (default True)

    Returns
    -------
    PredictResult
        Result with model visibilities
    """
    start_time = time.time()
    ms_path = Path(ms_path)

    if not ms_path.exists():
        return PredictResult(error=f"MS not found: {ms_path}")

    if not sources:
        return PredictResult(error="No sources provided for prediction")

    try:
        from casacore.tables import table
    except ImportError:
        return PredictResult(error="casacore not available for MS access")

    try:
        # Read UVW from MS
        with table(str(ms_path), readonly=True, ack=False) as tb:
            uvw = tb.getcol("UVW")  # (nrows, 3)
            # Get data shape for MODEL_DATA
            data = tb.getcol("DATA")  # (nrows, nchan, npol)
            nrows, nchan, npol = data.shape

        logger.info(
            "Predicting model for %s: %d sources, %s visibilities, %d channels",
            ms_path.name,
            len(sources),
            f"{nrows:,}",
            nchan,
        )

        # For channel_average mode, predict once and broadcast
        if channel_average:
            # Use center frequency UVW (they're the same for continuum)
            result = predict_visibilities_gpu(uvw, sources, config)

            if not result.success:
                return PredictResult(
                    error=result.error,
                    n_sources=len(sources),
                    n_vis=nrows,
                )

            # Broadcast to all channels and polarizations
            # MODEL_DATA is (nrows, nchan, npol), vis_predicted is (nrows,)
            vis_model = np.zeros((nrows, nchan, npol), dtype=np.complex64)

            # Stokes I: XX = YY = I/2 for unpolarized source
            # For simplicity, put full flux in each correlation
            # (This is correct for Stokes I gain calibration)
            vis_model[:, :, 0] = result.vis_predicted[:, np.newaxis]  # XX
            vis_model[:, :, 1] = 0.0  # XY
            vis_model[:, :, 2] = 0.0  # YX
            vis_model[:, :, 3] = result.vis_predicted[:, np.newaxis]  # YY

        else:
            # Per-channel prediction (slower but more accurate for wide bands)
            # TODO: Implement per-channel UVW scaling
            return PredictResult(
                error="Per-channel prediction not yet implemented",
                n_sources=len(sources),
            )

        # Write MODEL_DATA if requested
        if write_model:
            with table(str(ms_path), readonly=False, ack=False) as tb:
                if "MODEL_DATA" not in tb.colnames():
                    # Add MODEL_DATA column if missing
                    logger.info("Adding MODEL_DATA column to %s", ms_path.name)
                    from casacore.tables import makecoldesc, maketabdesc

                    desc = makecoldesc(
                        "MODEL_DATA",
                        tb.getcoldesc("DATA"),
                        valuetype="complex",
                    )
                    tb.addcols(maketabdesc(desc))

                tb.putcol("MODEL_DATA", vis_model)
                logger.info("Wrote MODEL_DATA to %s", ms_path.name)

        processing_time = time.time() - start_time

        logger.info(
            "Prediction complete: %d sources → %s visibilities in %.3fs",
            len(sources),
            f"{nrows:,}",
            processing_time,
        )

        return PredictResult(
            vis_model=vis_model,
            n_sources=len(sources),
            n_vis=nrows,
            processing_time_s=processing_time,
        )

    except Exception as err:
        logger.exception("Error predicting model for %s", ms_path)
        return PredictResult(
            error=f"Prediction failed: {err}",
            n_sources=len(sources),
        )


def predict_model_from_catalog(
    ms_path: str | Path,
    catalog_sources: list[dict],
    phase_center_ra: float,
    phase_center_dec: float,
    *,
    min_flux_jy: float = 0.005,
    max_sources: int = 50,
    write_model: bool = True,
    master_db: Path | None = None,
    vla_db: Path | None = None,
) -> PredictResult:
    """Predict model from catalog sources with spectral index lookup.

    Uses CatalogSourceAdapter to normalize source dictionaries and look up
    spectral indices from master_sources.sqlite3 or VLA calibrator database.
    Sources without valid spectral index are skipped.

    Parameters
    ----------
    ms_path : str | Path
        Path to measurement set
    catalog_sources : list[dict]
        Source catalog entries. Accepts various key formats:
        - 'ra' or 'ra_deg' for RA in degrees
        - 'dec' or 'dec_deg' for Dec in degrees
        - 'flux_mjy' or 'flux_jy' for flux
        - 'alpha' or 'spectral_index' for spectral index (optional)
        - 'name' for source name (optional)
    phase_center_ra : float
        Phase center RA in degrees
    phase_center_dec : float
        Phase center Dec in degrees
    min_flux_jy : float
        Minimum flux threshold in Jy (default 5 mJy)
    max_sources : int
        Maximum number of sources (default 50)
    write_model : bool
        Write to MODEL_DATA column (default True)
    master_db : Path, optional
        Path to master_sources.sqlite3 for spectral index lookup
    vla_db : Path, optional
        Path to vla_calibrators.sqlite3 for spectral index lookup

    Returns
    -------
    PredictResult
        Result with model visibilities
    """
    # Use adapter to convert and look up spectral indices
    adapter = CatalogSourceAdapter(
        master_db=master_db,
        vla_db=vla_db,
    )

    sources = adapter.from_dict_list(
        catalog_sources,
        min_flux_jy=min_flux_jy,
        max_sources=max_sources,
    )

    # Log adapter statistics
    stats = adapter.get_statistics()
    if stats["n_skipped"] > 0:
        logger.warning(
            "Skipped %d sources without spectral index "
            "(dict: %d, master: %d, VLA: %d)",
            stats["n_skipped"],
            stats["n_from_dict"],
            stats["n_from_master"],
            stats["n_from_vla"],
        )
    else:
        logger.info(
            "Spectral index sources: dict=%d, master=%d, VLA=%d",
            stats["n_from_dict"],
            stats["n_from_master"],
            stats["n_from_vla"],
        )

    if not sources:
        return PredictResult(
            error=f"No sources above {min_flux_jy} Jy with valid spectral index",
            n_sources=0,
        )

    logger.info(
        "Using %d catalog sources (%.1f - %.1f mJy, α: %.2f to %.2f) for model prediction",
        len(sources),
        sources[-1].flux_jy * 1000,
        sources[0].flux_jy * 1000,
        min(s.spectral_index for s in sources),
        max(s.spectral_index for s in sources),
    )

    # Configure prediction
    config = PredictConfig(
        image_size=512,
        cell_size_arcsec=12.0,
        phase_center_ra=phase_center_ra,
        phase_center_dec=phase_center_dec,
    )

    return predict_model_for_ms(
        ms_path,
        sources,
        config,
        write_model=write_model,
    )
