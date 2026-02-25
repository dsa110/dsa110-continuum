"""Mosaic calibration scheduler for managing gain and bandpass calibration.

This module implements the MosaicCalibrationManager class which handles:
- Detection of mosaic boundaries (12 consecutive tiles at same declination)
- Prediction of field centers for each observation
- Selection of gain calibration tiles (center tiles with most bright sources)
- Selection of bandpass calibrators based on declination
- Computation of bandpass calibrator transit schedules
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from dsa110_contimg.core.calibration.mosaic_constants import (
    EARTH_ROTATION_DEG_PER_SEC,
    INTEGRATION_TIME_SEC,
    MOSAIC_TILE_COUNT,
    N_FIELDS,
    SOURCE_QUERY_RADIUS_DEG,
    SKYMODEL_MIN_FLUX_MJY,
)

logger = logging.getLogger(__name__)


class MosaicCalibrationManager:
    """Manager for mosaic calibration scheduling and tile selection.
    
    This class provides methods for:
    - Identifying mosaic boundaries
    - Predicting field centers
    - Selecting gain calibration tiles
    - Selecting bandpass calibrators
    - Computing bandpass transit schedules
    """

    def __init__(self, db_session: Any):
        """Initialize mosaic calibration manager.
        
        Parameters
        ----------
        db_session : Any
            Database session for querying HDF5 index and sources
        """
        self.db_session = db_session

    def identify_mosaic_boundary(
        self, hdf5_groups: list[dict[str, Any]]
    ) -> list[list[dict[str, Any]]]:
        """Detect mosaic boundaries in HDF5 groups.
        
        A mosaic is defined as MOSAIC_TILE_COUNT consecutive groups at the same
        declination (within ±0.1°).
        
        Parameters
        ----------
        hdf5_groups : list[dict[str, Any]]
            List of HDF5 group records with 'group_id', 'pointing_dec_deg', etc.
            
        Returns
        -------
        list[list[dict[str, Any]]]
            List of detected mosaics, where each mosaic is a list of MOSAIC_TILE_COUNT
            consecutive groups
        """
        if len(hdf5_groups) < MOSAIC_TILE_COUNT:
            return []
        
        mosaics = []
        i = 0
        
        while i <= len(hdf5_groups) - MOSAIC_TILE_COUNT:
            # Check if next MOSAIC_TILE_COUNT groups have same declination
            candidate_mosaic = hdf5_groups[i : i + MOSAIC_TILE_COUNT]
            
            # Get declinations (handle None values properly)
            decs = []
            for g in candidate_mosaic:
                dec = g.get("pointing_dec_deg")
                if dec is None:
                    dec = g.get("dec_deg")
                decs.append(dec)
            
            # Skip if any declination is missing
            if None in decs:
                i += 1
                continue
            
            # Check if all declinations are within ±0.1° of the first
            ref_dec = decs[0]
            if all(abs(dec - ref_dec) <= 0.1 for dec in decs):
                mosaics.append(candidate_mosaic)
                logger.info(
                    f"Detected mosaic boundary: {MOSAIC_TILE_COUNT} tiles at "
                    f"dec={ref_dec:.2f}°, groups {candidate_mosaic[0]['group_id']} "
                    f"to {candidate_mosaic[-1]['group_id']}"
                )
                # Skip ahead by MOSAIC_TILE_COUNT to avoid overlapping mosaics
                i += MOSAIC_TILE_COUNT
            else:
                i += 1
        
        return mosaics

    def predict_field_centers(
        self, pointing_ra_deg: float, pointing_dec_deg: float
    ) -> list[tuple[float, float]]:
        """Predict field centers for all 24 fields in an observation.
        
        DSA-110 observes 24 fields sequentially with a fixed integration time per field.
        The field centers drift in RA due to Earth rotation while Dec remains constant.
        
        Parameters
        ----------
        pointing_ra_deg : float
            Pointing RA in degrees (mid-time LST from HDF5 index)
        pointing_dec_deg : float
            Pointing declination in degrees
            
        Returns
        -------
        list[tuple[float, float]]
            List of 24 (ra_deg, dec_deg) tuples for each field center
        """
        field_centers = []
        
        # Calculate offset from center of observation
        half_obs_sec = (N_FIELDS - 1) * INTEGRATION_TIME_SEC / 2.0
        
        for field_idx in range(N_FIELDS):
            # Time offset from center of observation
            offset_sec = field_idx * INTEGRATION_TIME_SEC - half_obs_sec
            
            # Calculate RA drift due to Earth rotation
            ra_drift_deg = offset_sec * EARTH_ROTATION_DEG_PER_SEC
            field_ra_deg = pointing_ra_deg + ra_drift_deg
            
            # Normalize RA to 0-360 range
            field_ra_deg = field_ra_deg % 360.0
            
            field_centers.append((field_ra_deg, pointing_dec_deg))
        
        return field_centers

    def select_gain_calibration_tile(
        self, mosaic_tiles: list[dict[str, Any]]
    ) -> dict[str, Any] | None:
        """Select center tile (5 or 6) with most bright sources for gain calibration.
        
        Parameters
        ----------
        mosaic_tiles : list[dict[str, Any]]
            List of MOSAIC_TILE_COUNT tiles in the mosaic
            
        Returns
        -------
        dict[str, Any] | None
            Selected tile record, or None if selection fails
        """
        if len(mosaic_tiles) != MOSAIC_TILE_COUNT:
            logger.error(
                f"Expected {MOSAIC_TILE_COUNT} tiles, got {len(mosaic_tiles)}"
            )
            return None
        
        # Check center tiles (indices 5 and 6 in 0-indexed list of 12)
        center_tiles = [mosaic_tiles[5], mosaic_tiles[6]]
        
        best_tile = None
        max_sources = 0
        
        for tile in center_tiles:
            pointing_ra = tile.get("pointing_ra_deg")
            if pointing_ra is None:
                pointing_ra = tile.get("ra_deg")
            pointing_dec = tile.get("pointing_dec_deg")
            if pointing_dec is None:
                pointing_dec = tile.get("dec_deg")
            
            if pointing_ra is None or pointing_dec is None:
                logger.warning(f"Missing coordinates for tile {tile['group_id']}")
                continue
            
            # Count sources in this tile
            try:
                source_count = self._count_sources_for_tile(pointing_ra, pointing_dec)
                
                logger.info(
                    f"Tile {tile['group_id']}: {source_count} sources "
                    f">={SKYMODEL_MIN_FLUX_MJY} mJy within {SOURCE_QUERY_RADIUS_DEG}°"
                )
                
                if source_count > max_sources:
                    max_sources = source_count
                    best_tile = tile
            except (RuntimeError, ValueError, KeyError, AttributeError) as e:
                logger.error(f"Error counting sources for tile {tile['group_id']}: {e}")
                continue
        
        if best_tile:
            logger.info(
                f"Selected tile {best_tile['group_id']} with {max_sources} sources"
            )
        else:
            logger.warning("No suitable calibration tile found")
        
        return best_tile

    def _count_sources_for_tile(
        self, pointing_ra_deg: float, pointing_dec_deg: float
    ) -> int:
        """Count unique sources in a tile's field of view.
        
        Parameters
        ----------
        pointing_ra_deg : float
            Pointing RA in degrees
        pointing_dec_deg : float
            Pointing declination in degrees
            
        Returns
        -------
        int
            Number of unique sources above flux threshold
        """
        from dsa110_contimg.core.calibration.model import count_bright_sources_in_tile
        
        return count_bright_sources_in_tile(
            pointing_ra_deg=pointing_ra_deg,
            pointing_dec_deg=pointing_dec_deg,
        )

    def select_bandpass_calibrator(
        self, pointing_dec_deg: float
    ) -> dict[str, Any] | None:
        """Select best bandpass calibrator for the given declination.
        
        Queries VLA calibrator catalog for calibrators visible at the given declination
        and selects the brightest one.
        
        Parameters
        ----------
        pointing_dec_deg : float
            Pointing declination in degrees
            
        Returns
        -------
        dict[str, Any] | None
            Calibrator info with 'name', 'ra_deg', 'dec_deg', 'flux_jy', or None if not found
        """
        try:
            from dsa110_contimg.core.calibration.catalogs import (
                load_vla_catalog,
            )
        except ImportError:
            logger.error("Cannot import VLA catalog functions")
            return None
        
        try:
            # Load and filter catalog
            catalog_df = load_vla_catalog()
            
            if catalog_df is None or len(catalog_df) == 0:
                logger.error("VLA calibrator catalog is empty or not loaded")
                return None
            
            visible_cal = self._filter_calibrators_by_dec(catalog_df, pointing_dec_deg)
            if visible_cal is None:
                return None
            
            # Find brightest calibrator
            calibrator_info = self._extract_brightest_calibrator(visible_cal)
            
            if calibrator_info:
                logger.info(
                    f"Selected bandpass calibrator: {calibrator_info['name']} "
                    f"(RA={calibrator_info['ra_deg']:.2f}°, "
                    f"Dec={calibrator_info['dec_deg']:.2f}°, "
                    f"Flux={calibrator_info['flux_jy']:.2f} Jy)"
                )
            
            return calibrator_info
            
        except (ValueError, KeyError, RuntimeError, IndexError) as e:
            logger.error(f"Error selecting bandpass calibrator: {e}", exc_info=True)
            return None

    def _filter_calibrators_by_dec(
        self, catalog_df: Any, pointing_dec_deg: float
    ) -> Any | None:
        """Filter VLA calibrators by declination.
        
        Parameters
        ----------
        catalog_df : pandas.DataFrame
            VLA calibrator catalog
        pointing_dec_deg : float
            Pointing declination in degrees
            
        Returns
        -------
        pandas.DataFrame | None
            Filtered catalog or None if no calibrators found
        """
        dec_tolerance = 15.0
        visible_cal = catalog_df[
            np.abs(catalog_df["dec_deg"] - pointing_dec_deg) <= dec_tolerance
        ]
        
        if len(visible_cal) == 0:
            logger.warning(
                f"No VLA calibrators found within ±{dec_tolerance}° of "
                f"dec={pointing_dec_deg:.2f}°"
            )
            return None
        
        return visible_cal

    def _extract_brightest_calibrator(self, visible_cal: Any) -> dict[str, Any] | None:
        """Extract brightest calibrator from filtered catalog.
        
        Parameters
        ----------
        visible_cal : pandas.DataFrame
            Filtered VLA calibrator catalog
            
        Returns
        -------
        dict[str, Any] | None
            Calibrator info or None if extraction fails
        """
        # Find flux column
        flux_col = None
        for col in ["flux_jy", "flux_density_jy", "S_1400", "flux_20cm"]:
            if col in visible_cal.columns:
                flux_col = col
                break
        
        if flux_col is None:
            logger.error("VLA catalog missing flux column")
            return None
        
        # Get brightest calibrator
        brightest_idx = visible_cal[flux_col].idxmax()
        brightest = visible_cal.loc[brightest_idx]
        
        # Extract name (try different column names)
        name = None
        for name_col in ["name", "source_name", "Source"]:
            if name_col in brightest.index:
                name = brightest[name_col]
                break
        
        if name is None:
            name = f"J{brightest['ra_deg']:.2f}+{brightest['dec_deg']:.2f}"
        
        return {
            "name": str(name),
            "ra_deg": float(brightest["ra_deg"]),
            "dec_deg": float(brightest["dec_deg"]),
            "flux_jy": float(brightest[flux_col]),
        }

    def compute_bp_transit_groups(
        self,
        calibrator_ra_deg: float,
        pointing_dec_deg: float,
        horizon_hours: float = 48.0,
    ) -> list[str]:
        """Find HDF5 groups containing a calibrator transit.
        
        Uses spatial matching via ``select_hdf5_groups_by_position`` to find
        groups whose beam covers the calibrator RA/Dec — the physically correct
        approach for drift-scan data.
        
        Parameters
        ----------
        calibrator_ra_deg : float
            Calibrator RA in degrees.
        pointing_dec_deg : float
            Pointing declination in degrees.
        horizon_hours : float, optional
            Deprecated — kept for API compatibility, ignored.
            
        Returns
        -------
        list[str]
            Representative timestamps for groups containing the calibrator transit.
        """
        from dsa110_contimg.infrastructure.database.unified import get_pipeline_db_path
        
        try:
            from dsa110_contimg.infrastructure.database.hdf5_index import (
                select_hdf5_groups_by_position,
            )
        except ImportError:
            logger.warning("Cannot import select_hdf5_groups_by_position")
            return []
        
        # Use spatial matching: find groups whose beam covers the calibrator.
        # This is the correct physical approach for a drift-scan telescope.
        try:
            db_path = get_pipeline_db_path()
            transit_groups = select_hdf5_groups_by_position(
                db_path=db_path,
                source_ra_deg=calibrator_ra_deg,
                source_dec_deg=pointing_dec_deg,
                beam_radius_deg=1.75,
                n_groups=24,
                require_complete=True,
            )
        except (ValueError, RuntimeError, ImportError) as e:
            logger.warning(f"Error querying HDF5 groups by position: {e}")
            transit_groups = []
        
        logger.info(
            f"Found {len(transit_groups)} groups containing calibrator transit "
            f"(RA={calibrator_ra_deg:.3f}°, Dec={pointing_dec_deg:.3f}°)"
        )
        
        return transit_groups
