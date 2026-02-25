"""
Field naming utilities for DSA-110 Measurement Sets.

This module provides functions to rename fields with calibrator names,
preserving time index information for drift-scan observations.
"""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def rename_calibrator_field(
    ms_path: str,
    calibrator_name: str,
    field_idx: int,
    *,
    include_time_suffix: bool = True,
) -> None:
    """Rename a single field to include calibrator name.

        For DSA-110 drift-scan observations, fields are initially named
        meridian_icrs_t0, meridian_icrs_t1, etc. (one per 12.88s timestamp).
        This function renames the field containing a calibrator to include
        its catalog name, optionally preserving the time index.

    Parameters
    ----------
    ms_path : str
        Path to Measurement Set
    calibrator_name : str
        Name from VLA catalog (e.g., "3C286", "J1331+3030")
    field_idx : int
        Field index containing the calibrator (from auto-detection)
    include_time_suffix : bool, optional
        If True, append _t{idx} (e.g., "3C286_t17")
        If False, use only calibrator name (e.g., "3C286")
    Default: True (recommended for drift-scan observations)

    Examples
    --------
        >>> # Rename field 17 to "3C286_t17" (preserves time index)
        >>> rename_calibrator_field("2025-10-18T14:35:20.ms", "3C286", 17)

        >>> # Rename field 5 to just "J1331+3030" (no time suffix)
        >>> rename_calibrator_field("obs.ms", "J1331+3030", 5, include_time_suffix=False)

    Notes
    -----
        - Time suffix is recommended for drift-scan observations to preserve
        which 12.88s timestamp contained the calibrator at optimal alignment
        - For concatenated fields (after rephasing), time suffix may not be meaningful
        - Uses casacore.tables for direct FIELD table access (no CASA tasks required)
    """
    try:
        import casacore.tables as casatables
    except ImportError:
        logger.error("casacore.tables not available - cannot rename field")
        return

    ms_path = str(Path(ms_path).resolve())

    try:
        with casatables.table(f"{ms_path}::FIELD", readonly=False) as field_tb:
            field_names = field_tb.getcol("NAME")

            if field_idx < 0 or field_idx >= len(field_names):
                logger.warning(
                    f"Field index {field_idx} out of range [0, {len(field_names) - 1}] "
                    f"for MS {ms_path}"
                )
                return

            original_name = field_names[field_idx]

            if include_time_suffix:
                new_name = f"{calibrator_name}_t{field_idx}"
            else:
                new_name = calibrator_name

            field_names[field_idx] = new_name
            field_tb.putcol("NAME", field_names)

            logger.info(
                f":check_mark: Renamed field {field_idx} from '{original_name}' to '{new_name}' "
                f"in {Path(ms_path).name}"
            )

    except Exception as e:
        logger.warning(
            f"Could not rename field {field_idx} to '{calibrator_name}' in {ms_path}: {e}"
        )


def rename_calibrator_fields_from_catalog(
    ms_path: str,
    catalog_path: str | None = None,
    *,
    search_radius_deg: float = 1.0,
    freq_GHz: float = 1.4,
    include_time_suffix: bool = True,
) -> tuple[str, int] | None:
    """Auto-detect and rename field containing brightest calibrator from catalog.

        This function uses the same auto-detection logic as cli_calibrate.py
        to find which field contains a known calibrator, then renames that field.

    Parameters
    ----------
    ms_path : str
        Path to Measurement Set
    catalog_path : str, optional
        Path to VLA calibrator catalog (SQLite or CSV)
        If None, uses automatic resolution (prefers SQLite)
    search_radius_deg : float, optional
        Search radius in degrees for catalog matching
    Default: 1.0
    freq_GHz : float, optional
        Frequency in GHz for primary beam calculation
    Default: 1.4 (DSA-110 center frequency)
    include_time_suffix : bool, optional
        If True, append _t{idx} to calibrator name
    Default: True

    Returns
    -------
        tuple or None
        (calibrator_name, field_idx) if successful, None if no calibrator found

    Examples
    --------
        >>> # Auto-detect and rename calibrator field
        >>> result = rename_calibrator_fields_from_catalog("2025-10-18T14:35:20.ms")
        >>> if result:
        ...     name, idx = result
        ...     print(f"Renamed field {idx} to {name}_t{idx}")

    Notes
    -----
        - Uses select_bandpass_from_catalog() for field detection
        - Only renames the peak field (highest PB-weighted flux)
        - Handles all-meridian drift-scan mode correctly (checks all 24 fields)
        - Silently returns None if no calibrator found (logs warning)
    """
    from dsa110_contimg.core.calibration.selection import select_bandpass_from_catalog

    try:
        _, _, _, cal_info, peak_field = select_bandpass_from_catalog(
            ms_path,
            catalog_path,
            search_radius_deg=search_radius_deg,
            freq_GHz=freq_GHz,
        )

        calibrator_name, ra_deg, dec_deg, flux_jy = cal_info

        logger.info(
            f"Auto-detected calibrator '{calibrator_name}' in field {peak_field} "
            f"at ({ra_deg:.4f}, {dec_deg:.4f}) deg, {flux_jy:.2f} Jy"
        )

        rename_calibrator_field(
            ms_path,
            calibrator_name,
            peak_field,
            include_time_suffix=include_time_suffix,
        )

        return (calibrator_name, peak_field)

    except Exception as e:
        logger.warning(f"Could not auto-detect calibrator for renaming in {ms_path}: {e}")
        return None


__all__ = [
    "rename_calibrator_field",
    "rename_calibrator_fields_from_catalog",
]
