"""
Calibrator MS Generator - Generate Measurement Sets around calibrator transits.

This module provides a unified workflow for generating MS files specifically
targeting the transit of bandpass calibrators for science-quality calibration.

Usage:
    >>> from dsa110_contimg.core.conversion import CalibratorMSGenerator
    >>> from pathlib import Path
    >>> from astropy.time import Time
    >>>
    >>> generator = CalibratorMSGenerator(
    ...     input_dir=Path("/data/incoming"),
    ...     output_dir=Path("/stage/dsa110-contimg/ms"),
    ... )
    >>>
    >>> # Generate MS from a specific calibrator transit
    >>> result = generator.generate_from_transit(
    ...     calibrator_name="0834+555",
    ...     transit_time=Time("2025-01-15T14:30:00"),
    ...     window_minutes=30,
    ... )
    >>>
    >>> # Or auto-detect best calibrator for current pointing
    >>> result = generator.generate_for_pointing(
    ...     dec_deg=43.5,
    ...     lookback_days=7,
    ... )
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import astropy.units as u
from astropy.time import Time

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = logging.getLogger(__name__)

# Default paths
DEFAULT_DB_PATH = Path("/data/dsa110-contimg/state/db/pipeline.sqlite3")
DEFAULT_VLA_CATALOG = Path("/data/dsa110-contimg/state/catalogs/vla_calibrators.sqlite3")


@dataclass
class CalibratorInfo:
    """Information about a selected calibrator."""

    name: str
    ra_deg: float
    dec_deg: float
    flux_jy: float
    position_code: str | None = None
    quality_codes: str | None = None


@dataclass
class TransitInfo:
    """Information about a calibrator transit."""

    calibrator: CalibratorInfo
    transit_time: Time
    transit_time_iso: str
    transit_time_mjd: float


@dataclass
class CalibratorMSResult:
    """Result of calibrator MS generation."""

    success: bool
    ms_path: Path | None = None
    ms_paths: list[Path] = field(default_factory=list)
    calibrator: CalibratorInfo | None = None
    transit: TransitInfo | None = None
    groups_converted: int = 0
    error_message: str | None = None
    calibrator_in_ms: bool = False
    peak_field_index: int | None = None

    def __bool__(self) -> bool:
        """Allow truthiness check on result."""
        return self.success


class CalibratorMSGenerator:
    """Generate Measurement Sets around calibrator transits.

    This class provides a unified workflow for:
    1. Selecting the best calibrator from the VLA catalog
    2. Calculating transit times
    3. Selecting HDF5 groups around the transit
    4. Converting to Measurement Sets
    5. Verifying calibrator presence in the MS

    Parameters
    ----------
    input_dir : Path
        Directory containing HDF5 subband files
    output_dir : Path
        Directory for output Measurement Sets
    db_path : Path, optional
        Path to pipeline database (default: pipeline.sqlite3)
    vla_catalog_path : Path, optional
        Path to VLA calibrator catalog database
    scratch_dir : Path, optional
        Fast scratch storage for intermediate files
    max_workers : int, optional
        Number of parallel conversion workers (default: 8)

    Examples
    --------
    >>> generator = CalibratorMSGenerator(
    ...     input_dir=Path("/data/incoming"),
    ...     output_dir=Path("/stage/dsa110-contimg/ms"),
    ... )
    >>>
    >>> # Generate from known calibrator transit
    >>> result = generator.generate_from_transit("0834+555", transit_time)
    >>> print(f"MS: {result.ms_path}")
    >>>
    >>> # Auto-detect best calibrator for declination
    >>> result = generator.generate_for_pointing(dec_deg=43.5)
    >>> print(f"Calibrator: {result.calibrator.name}")
    """

    def __init__(
        self,
        input_dir: Path,
        output_dir: Path,
        *,
        db_path: Path | None = None,
        vla_catalog_path: Path | None = None,
        scratch_dir: Path | None = None,
        max_workers: int = 8,
    ) -> None:
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.db_path = Path(db_path) if db_path else DEFAULT_DB_PATH
        self.vla_catalog_path = (
            Path(vla_catalog_path) if vla_catalog_path else DEFAULT_VLA_CATALOG
        )
        self.scratch_dir = Path(scratch_dir) if scratch_dir else None
        self.max_workers = max_workers

        # Validate paths
        if not self.input_dir.exists():
            raise FileNotFoundError(f"Input directory not found: {self.input_dir}")
        if not self.vla_catalog_path.exists():
            raise FileNotFoundError(
                f"VLA calibrator catalog not found: {self.vla_catalog_path}"
            )

    def get_calibrator(
        self,
        calibrator_name: str | None = None,
        dec_deg: float | None = None,
        dec_tolerance: float = 5.0,
        min_flux_jy: float = 1.0,
    ) -> CalibratorInfo:
        """Get calibrator by name or by declination.

        Parameters
        ----------
        calibrator_name : str, optional
            Specific calibrator name (e.g., "0834+555", "3C286")
        dec_deg : float, optional
            Target declination for auto-selection
        dec_tolerance : float
            Declination search range in degrees
        min_flux_jy : float
            Minimum flux threshold at L-band

        Returns
        -------
        CalibratorInfo
            Selected calibrator information

        Raises
        ------
        ValueError
            If neither calibrator_name nor dec_deg provided
        RuntimeError
            If no matching calibrator found
        """
        from dsa110_contimg.core.catalog.build_vla_calibrators import (
            get_best_vla_calibrator,
            query_calibrators_by_dec,
        )

        if calibrator_name is not None:
            # Look up specific calibrator
            # Search wide range then filter
            calibrators = query_calibrators_by_dec(
                dec_deg=0,  # Will be filtered by name
                max_separation=90,
                min_flux_jy=0,
                db_path=self.vla_catalog_path,
            )
            for cal in calibrators:
                # Normalize names for comparison
                if calibrator_name.upper().replace("J", "") in cal["name"].upper().replace("J", ""):
                    return CalibratorInfo(
                        name=cal["name"],
                        ra_deg=cal["ra_deg"],
                        dec_deg=cal["dec_deg"],
                        flux_jy=cal["flux_jy"],
                        position_code=cal.get("position_code"),
                        quality_codes=cal.get("quality_codes"),
                    )
            raise RuntimeError(f"Calibrator '{calibrator_name}' not found in VLA catalog")

        if dec_deg is not None:
            cal = get_best_vla_calibrator(
                dec_deg=dec_deg,
                dec_tolerance=dec_tolerance,
                min_flux_jy=min_flux_jy,
                db_path=self.vla_catalog_path,
            )
            if cal is None:
                raise RuntimeError(
                    f"No calibrator found for Dec {dec_deg}° ±{dec_tolerance}° "
                    f"with flux ≥{min_flux_jy} Jy"
                )
            return CalibratorInfo(
                name=cal["name"],
                ra_deg=cal["ra_deg"],
                dec_deg=cal["dec_deg"],
                flux_jy=cal["flux_jy"],
                position_code=cal.get("position_code"),
                quality_codes=cal.get("quality_codes"),
            )

        raise ValueError("Must provide either calibrator_name or dec_deg")

    def find_transits(
        self,
        calibrator: CalibratorInfo,
        start_time: Time,
        end_time: Time,
    ) -> list[Time]:
        """Find all transit times for a calibrator in a time window.

        Parameters
        ----------
        calibrator : CalibratorInfo
            Calibrator to find transits for
        start_time : Time
            Start of search window
        end_time : Time
            End of search window

        Returns
        -------
        list[Time]
            List of transit times
        """
        from dsa110_contimg.core.calibration.transit import transit_times

        return transit_times(
            ra_deg=calibrator.ra_deg,
            start_time=start_time,
            end_time=end_time,
        )

    def find_last_transit(
        self,
        calibrator: CalibratorInfo,
        lookback_days: int = 7,
    ) -> Time:
        """Find the most recent transit for a calibrator.

        Parameters
        ----------
        calibrator : CalibratorInfo
            Calibrator to find transit for
        lookback_days : int
            Number of days to look back

        Returns
        -------
        Time
            Most recent transit time

        Raises
        ------
        RuntimeError
            If no transit found in lookback window
        """
        end_time = Time.now()
        start_time = end_time - lookback_days * u.day

        transits = self.find_transits(calibrator, start_time, end_time)

        if not transits:
            raise RuntimeError(
                f"No transits found for {calibrator.name} in the last {lookback_days} days"
            )

        return transits[-1]

    def select_groups_by_position(
        self,
        source_ra_deg: float,
        source_dec_deg: float,
        beam_radius_deg: float = 1.75,
        n_groups: int = 12,
    ) -> list:
        """Select HDF5 groups where a source is within the primary beam.

        Uses spatial matching on the actual RA/Dec stored in the HDF5 index
        instead of computing transit times.  This is the preferred method
        for DSA-110 as a drift-scan instrument.

        Parameters
        ----------
        source_ra_deg : float
            Source RA in degrees [0, 360).
        source_dec_deg : float
            Source Dec in degrees.
        beam_radius_deg : float
            Half-power beam radius in degrees.  Default 1.75° (half of
            DSA-110 primary beam FWHM ≈ 3.5° at 1.4 GHz).
        n_groups : int
            Maximum number of groups to return.  Default 12.

        Returns
        -------
        list
            List of group representative timestamps (ISO format).
        """
        from dsa110_contimg.infrastructure.database.hdf5_index import (
            select_hdf5_groups_by_position,
        )

        return select_hdf5_groups_by_position(
            db_path=str(self.db_path),
            source_ra_deg=source_ra_deg,
            source_dec_deg=source_dec_deg,
            beam_radius_deg=beam_radius_deg,
            n_groups=n_groups,
        )

    def convert_groups(
        self,
        groups: Sequence,
        skip_existing: bool = True,
    ) -> list[Path]:
        """Convert HDF5 groups to Measurement Sets.

        Parameters
        ----------
        groups : Sequence
            List of SubbandGroup objects or file lists
        skip_existing : bool
            Skip conversion if MS already exists

        Returns
        -------
        list[Path]
            Paths to created MS files
        """
        from dsa110_contimg.core.conversion import convert_subband_groups_to_ms

        self.output_dir.mkdir(parents=True, exist_ok=True)
        ms_paths = []

        for group in groups:
            # Handle both SubbandGroup objects and raw file lists
            files = group.files if hasattr(group, "files") else group
            if not files:
                continue

            # Extract timestamp from first file
            first_file = Path(files[0])
            timestamp = first_file.stem.rsplit("_sb", 1)[0]

            # Check if already exists
            ms_path = self.output_dir / f"{timestamp}.ms"
            if skip_existing and ms_path.exists():
                logger.info("Skipping existing MS: %s", ms_path)
                ms_paths.append(ms_path)
                continue

            # Convert using time-based API
            end_ts = (Time(timestamp, format="isot") + 2 * u.minute).isot

            try:
                convert_subband_groups_to_ms(
                    input_dir=str(self.input_dir),
                    output_dir=str(self.output_dir),
                    start_time=timestamp,
                    end_time=end_ts,
                    skip_existing=False,
                )

                # Find the created MS
                pattern = f"{timestamp}*.ms"
                created = sorted(
                    self.output_dir.glob(pattern),
                    key=lambda p: p.stat().st_mtime,
                    reverse=True,
                )
                if created:
                    ms_paths.append(created[0])
                    logger.info("Created MS: %s", created[0])

            except Exception as exc:
                logger.error("Failed to convert group %s: %s", timestamp, exc)

        return ms_paths

    def verify_calibrator_in_ms(
        self,
        ms_path: Path,
        calibrator: CalibratorInfo,
        search_radius_deg: float = 5.0,
    ) -> tuple[bool, int | None]:
        """Verify that a calibrator is present in an MS.

        Parameters
        ----------
        ms_path : Path
            Path to Measurement Set
        calibrator : CalibratorInfo
            Expected calibrator
        search_radius_deg : float
            Search radius in degrees

        Returns
        -------
        tuple[bool, int | None]
            (is_present, peak_field_index)
        """
        from dsa110_contimg.core.calibration.selection import (
            select_bandpass_from_catalog,
        )

        try:
            _, _, _, _, peak_field = select_bandpass_from_catalog(
                ms_path=str(ms_path),
                calibrator_name=calibrator.name,
                search_radius_deg=search_radius_deg,
                freq_GHz=1.4,
            )
            return True, peak_field
        except RuntimeError:
            return False, None

    def generate_from_transit(
        self,
        calibrator_name: str,
        transit_time: Time,
        window_minutes: int = 30,
        *,
        verify: bool = True,
    ) -> CalibratorMSResult:
        """Generate MS from a specific calibrator transit.

        Parameters
        ----------
        calibrator_name : str
            Name of calibrator (e.g., "0834+555", "3C286")
        transit_time : Time
            Known transit time
        window_minutes : int
            Time window around transit (total, split before/after)
        verify : bool
            Verify calibrator presence after conversion

        Returns
        -------
        CalibratorMSResult
            Result with MS path and calibrator info
        """
        try:
            # Get calibrator info
            calibrator = self.get_calibrator(calibrator_name=calibrator_name)

            transit_info = TransitInfo(
                calibrator=calibrator,
                transit_time=transit_time,
                transit_time_iso=transit_time.iso,
                transit_time_mjd=transit_time.mjd,
            )

            # Select groups by spatial position (preferred for drift-scan)
            n_groups = max(1, window_minutes // 5)

            groups = self.select_groups_by_position(
                source_ra_deg=calibrator.ra_deg,
                source_dec_deg=calibrator.dec_deg,
                beam_radius_deg=1.75,
                n_groups=n_groups,
            )

            if not groups:
                return CalibratorMSResult(
                    success=False,
                    calibrator=calibrator,
                    transit=transit_info,
                    error_message="No HDF5 groups found within primary beam",
                )

            # Convert closest group to transit
            ms_paths = self.convert_groups(groups[:1])  # Just closest

            if not ms_paths:
                return CalibratorMSResult(
                    success=False,
                    calibrator=calibrator,
                    transit=transit_info,
                    groups_converted=0,
                    error_message="Conversion failed",
                )

            ms_path = ms_paths[0]

            # Verify calibrator in MS
            calibrator_in_ms = False
            peak_field = None
            if verify:
                calibrator_in_ms, peak_field = self.verify_calibrator_in_ms(
                    ms_path, calibrator
                )

            return CalibratorMSResult(
                success=True,
                ms_path=ms_path,
                ms_paths=ms_paths,
                calibrator=calibrator,
                transit=transit_info,
                groups_converted=len(ms_paths),
                calibrator_in_ms=calibrator_in_ms,
                peak_field_index=peak_field,
            )

        except Exception as exc:
            logger.exception("Failed to generate MS from transit")
            return CalibratorMSResult(
                success=False,
                error_message=str(exc),
            )

    def generate_for_pointing(
        self,
        dec_deg: float,
        *,
        lookback_days: int = 7,
        window_minutes: int = 30,
        dec_tolerance: float = 5.0,
        min_flux_jy: float = 1.0,
        verify: bool = True,
    ) -> CalibratorMSResult:
        """Generate MS for the best calibrator at a given pointing.

        This method:
        1. Selects the best calibrator from VLA catalog for the declination
        2. Finds the most recent transit
        3. Generates MS around that transit

        Parameters
        ----------
        dec_deg : float
            Telescope pointing declination in degrees
        lookback_days : int
            Number of days to look back for transits
        window_minutes : int
            Time window around transit
        dec_tolerance : float
            Declination search tolerance
        min_flux_jy : float
            Minimum calibrator flux
        verify : bool
            Verify calibrator presence after conversion

        Returns
        -------
        CalibratorMSResult
            Result with MS path and calibrator info
        """
        try:
            # Select best calibrator
            calibrator = self.get_calibrator(
                dec_deg=dec_deg,
                dec_tolerance=dec_tolerance,
                min_flux_jy=min_flux_jy,
            )

            logger.info(
                "Selected calibrator %s (%.1f Jy) for Dec %.2f°",
                calibrator.name,
                calibrator.flux_jy,
                dec_deg,
            )

            # Find last transit
            transit_time = self.find_last_transit(
                calibrator, lookback_days=lookback_days
            )

            logger.info("Last transit: %s", transit_time.iso)

            # Generate MS
            return self.generate_from_transit(
                calibrator_name=calibrator.name,
                transit_time=transit_time,
                window_minutes=window_minutes,
                verify=verify,
            )

        except Exception as exc:
            logger.exception("Failed to generate MS for pointing")
            return CalibratorMSResult(
                success=False,
                error_message=str(exc),
            )

    def generate_multiple(
        self,
        calibrator_name: str,
        transit_time: Time,
        n_groups: int = 12,
        *,
        verify: bool = True,
    ) -> CalibratorMSResult:
        """Generate multiple MS files around a calibrator transit.

        Use this when you need multiple observations for mosaic creation.

        Parameters
        ----------
        calibrator_name : str
            Name of calibrator
        transit_time : Time
            Target transit time
        n_groups : int
            Total number of groups to convert
        verify : bool
            Verify calibrator presence in central MS

        Returns
        -------
        CalibratorMSResult
            Result with list of MS paths
        """
        try:
            calibrator = self.get_calibrator(calibrator_name=calibrator_name)

            transit_info = TransitInfo(
                calibrator=calibrator,
                transit_time=transit_time,
                transit_time_iso=transit_time.iso,
                transit_time_mjd=transit_time.mjd,
            )

            groups = self.select_groups_by_position(
                source_ra_deg=calibrator.ra_deg,
                source_dec_deg=calibrator.dec_deg,
                beam_radius_deg=1.75,
                n_groups=n_groups,
            )

            if not groups:
                return CalibratorMSResult(
                    success=False,
                    calibrator=calibrator,
                    transit=transit_info,
                    error_message="No HDF5 groups found within primary beam",
                )

            # Convert all groups
            ms_paths = self.convert_groups(groups)

            if not ms_paths:
                return CalibratorMSResult(
                    success=False,
                    calibrator=calibrator,
                    transit=transit_info,
                    error_message="No MS files created",
                )

            # Find the MS closest to transit for verification
            central_ms = ms_paths[len(ms_paths) // 2] if ms_paths else None

            calibrator_in_ms = False
            peak_field = None
            if verify and central_ms:
                calibrator_in_ms, peak_field = self.verify_calibrator_in_ms(
                    central_ms, calibrator
                )

            return CalibratorMSResult(
                success=True,
                ms_path=central_ms,
                ms_paths=ms_paths,
                calibrator=calibrator,
                transit=transit_info,
                groups_converted=len(ms_paths),
                calibrator_in_ms=calibrator_in_ms,
                peak_field_index=peak_field,
            )

        except Exception as exc:
            logger.exception("Failed to generate multiple MS")
            return CalibratorMSResult(
                success=False,
                error_message=str(exc),
            )
