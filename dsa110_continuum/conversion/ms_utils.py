"""Shared utilities to configure Measurement Sets for imaging.

This module centralizes robust, repeatable post-write MS preparation:
- Ensure imaging columns exist (MODEL_DATA, CORRECTED_DATA)
- Populate imaging columns for every row with array values matching DATA
- Ensure FLAG and WEIGHT_SPECTRUM arrays are present and correctly shaped
- Initialize weights, including WEIGHT_SPECTRUM, via casatasks.initweights
- Normalize ANTENNA.MOUNT to CASA-compatible values

All callers should prefer `configure_ms_for_imaging()` rather than duplicating
these steps inline in scripts. This provides a single source of truth for MS
readiness across the pipeline.
"""

from __future__ import annotations

import os

from dsa110_contimg.common.utils.runtime_safeguards import require_casa6_python


def _ensure_imaging_columns_exist(ms_path: str) -> None:
    """Add MODEL_DATA and CORRECTED_DATA columns if missing.

    Uses CASA's clearcal() to create columns, NOT casacore's addImagingColumns.
    This is critical because casacore creates columns with different storage format
    that CASA cannot read properly.

    Parameters
    ----------

    Raises
    ------
    RuntimeError
        If column creation fails and columns don't already exist

    """
    import logging

    logger = logging.getLogger(__name__)

    # Ensure CASAPATH is set before importing CASA modules
    from dsa110_contimg.common.utils.casa_init import ensure_casa_path

    ensure_casa_path()

    # Check if columns already exist using casacore (fast readonly check)
    try:
        import casacore.tables as _casatables

        _tb = _casatables.table

        with _tb(ms_path, readonly=True) as tb:
            colnames = set(tb.colnames())
            has_model = "MODEL_DATA" in colnames
            has_corrected = "CORRECTED_DATA" in colnames

            if has_model and has_corrected:
                logger.debug(f"Imaging columns already exist in {ms_path}")
                return
    except Exception as e:
        logger.debug(f"Could not check columns with casacore: {e}")

    # Use CASA's clearcal to create the columns - ensures CASA compatibility
    # In conversion context (new MS), no calibration has been applied yet,
    # so we can safely disable protection.
    try:
        from dsa110_contimg.core.calibration.casa_service import CASAService

        service = CASAService()
        logger.info(f"Creating imaging columns using clearcal: {ms_path}")
        # protect_calibration=False: This is a new MS, no calibration yet
        service.clearcal(vis=ms_path, addmodel=True, protect_calibration=False)
        logger.debug(f"Imaging columns created via clearcal for {ms_path}")
    except Exception as e:
        # DO NOT fall back to casacore - it creates incompatible columns
        error_msg = f"Failed to create imaging columns with clearcal in {ms_path}: {e}"
        logger.error(error_msg)
        raise RuntimeError(error_msg) from e

    # Verify columns were created
    try:
        import casacore.tables as _casatables

        _tb = _casatables.table

        with _tb(ms_path, readonly=True) as tb:
            colnames = set(tb.colnames())
            missing = []
            if "MODEL_DATA" not in colnames:
                missing.append("MODEL_DATA")
            if "CORRECTED_DATA" not in colnames:
                missing.append("CORRECTED_DATA")
            if missing:
                raise RuntimeError(f"clearcal() completed but columns still missing: {missing}")
    except Exception as e:
        if "still missing" in str(e):
            raise
        logger.warning(f"Could not verify columns after clearcal: {e}")


def _ensure_imaging_columns_populated(ms_path: str) -> None:
    """Ensure MODEL_DATA and CORRECTED_DATA contain array values for every
    row, with shapes/dtypes matching the DATA column cells.

    This function uses vectorized operations for performance (~50x faster
    than row-by-row iteration on large MS files). It checks if columns need
    initialization by sampling rows, then uses bulk putcol operations.

    Parameters
    ----------

    Raises
    ------
    RuntimeError
        If columns exist but cannot be populated

    """
    import logging

    logger = logging.getLogger(__name__)

    try:
        import casacore.tables as _casatables  # type: ignore
        import numpy as _np

        _tb = _casatables.table
    except ImportError as e:
        error_msg = f"Failed to import required modules for column population: {e}"
        logger.error(error_msg)
        raise RuntimeError(error_msg) from e

    try:
        with _tb(ms_path, readonly=False) as tb:
            nrow = tb.nrows()
            if nrow == 0:
                logger.warning(f"MS {ms_path} has no rows - cannot populate columns")
                return

            colnames = set(tb.colnames())
            if "MODEL_DATA" not in colnames or "CORRECTED_DATA" not in colnames:
                missing = []
                if "MODEL_DATA" not in colnames:
                    missing.append("MODEL_DATA")
                if "CORRECTED_DATA" not in colnames:
                    missing.append("CORRECTED_DATA")
                raise RuntimeError(f"Cannot populate columns - they don't exist: {missing}")

            # Get DATA shape and dtype from first row
            try:
                data0 = tb.getcell("DATA", 0)
                data_shape = getattr(data0, "shape", None)
                data_dtype = getattr(data0, "dtype", None)
                if not data_shape or data_dtype is None:
                    raise RuntimeError("Cannot determine DATA column shape/dtype")
            except Exception as e:
                error_msg = f"Failed to read DATA column from {ms_path}: {e}"
                logger.error(error_msg)
                raise RuntimeError(error_msg) from e

            # Populate each column using vectorized operations
            for col in ("MODEL_DATA", "CORRECTED_DATA"):
                if col not in tb.colnames():
                    continue

                # Quick check: sample first, middle, and last rows to determine
                # if the column needs initialization. This catches the common cases:
                # 1. All rows properly initialized (no work needed)
                # 2. All rows need initialization (bulk write)
                # 3. Mixed state (fall back to row-by-row for safety)
                needs_init = False
                has_valid_data = False
                sample_indices = [0, nrow // 2, nrow - 1] if nrow > 2 else list(range(nrow))

                for idx in sample_indices:
                    try:
                        val = tb.getcell(col, idx)
                        if val is None or getattr(val, "shape", None) != data_shape:
                            needs_init = True
                        elif _np.any(val != 0):
                            # Column has non-zero data - it's been populated
                            has_valid_data = True
                    except (RuntimeError, KeyError, IndexError):
                        # RuntimeError: CASA errors, KeyError: missing col, IndexError: bad row
                        needs_init = True

                # If column already has valid non-zero data, skip initialization
                if has_valid_data and not needs_init:
                    logger.debug(f"Column {col} already populated in {ms_path}")
                    continue

                # If all sampled rows need initialization, use fast bulk write
                if needs_init:
                    try:
                        # Use vectorized putcol for ~50x speedup over row-by-row
                        # Process in chunks to manage memory for very large MS files
                        chunk_size = 100000  # ~100k rows per chunk
                        fixed = 0

                        for start_row in range(0, nrow, chunk_size):
                            end_row = min(start_row + chunk_size, nrow)
                            chunk_nrow = end_row - start_row

                            # Create zero array for this chunk
                            # Shape is (nrow, nfreq, npol) for casacore putcol
                            zeros = _np.zeros((chunk_nrow,) + data_shape, dtype=data_dtype)
                            tb.putcol(col, zeros, startrow=start_row, nrow=chunk_nrow)
                            fixed += chunk_nrow

                        logger.debug(f"Bulk-populated {fixed} rows in {col} column for {ms_path}")

                    except Exception as bulk_err:
                        # Fall back to row-by-row if bulk operation fails
                        logger.warning(
                            f"Bulk population failed for {col}, falling back to "
                            f"row-by-row: {bulk_err}"
                        )
                        fixed, errors = _populate_column_row_by_row(
                            tb, col, nrow, data_shape, data_dtype, logger, ms_path
                        )
                        if fixed > 0:
                            logger.debug(
                                f"Row-by-row populated {fixed} rows in {col} for {ms_path}"
                            )

    except RuntimeError:
        # Re-raise RuntimeError (our own errors)
        raise
    except Exception as e:
        error_msg = f"Failed to populate imaging columns in {ms_path}: {e}"
        logger.error(error_msg, exc_info=True)
        raise RuntimeError(error_msg) from e


def _populate_column_row_by_row(
    tb, col: str, nrow: int, data_shape: tuple, data_dtype, logger, ms_path: str
) -> tuple:
    """Fallback row-by-row population for columns with mixed initialization states.

    This preserves the original behavior for edge cases where bulk operations
    might overwrite valid data.

    Parameters
    ----------
    tb :

    Returns
    -------
    tuple
        (fixed_count, error_count)

    """
    import numpy as _np

    fixed = 0
    errors = 0
    error_examples = []

    for r in range(nrow):
        try:
            val = tb.getcell(col, r)
            if (val is None) or (getattr(val, "shape", None) != data_shape):
                tb.putcell(col, r, _np.zeros(data_shape, dtype=data_dtype))
                fixed += 1
        except (RuntimeError, KeyError, IndexError):
            try:
                tb.putcell(col, r, _np.zeros(data_shape, dtype=data_dtype))
                fixed += 1
            except (RuntimeError, OSError) as e2:
                errors += 1
                if len(error_examples) < 5:
                    error_examples.append(f"row {r}: {e2}")

    if errors > 0:
        error_summary = (
            f"Failed to populate {errors} out of {nrow} rows in {col} column for {ms_path}"
        )
        if error_examples:
            error_summary += f". Examples: {'; '.join(error_examples)}"
        logger.warning(error_summary)

    return fixed, errors


def _ensure_flag_and_weight_spectrum(ms_path: str) -> None:
    """Ensure FLAG and WEIGHT_SPECTRUM cells exist with correct shapes for all rows.

    - FLAG: boolean array shaped like DATA; fill with False when undefined
    - WEIGHT_SPECTRUM: float array shaped like DATA; when undefined,
      repeat WEIGHT across channels; if WEIGHT_SPECTRUM appears
      inconsistent across rows, drop the column to let CASA fall back
      to WEIGHT.

    Parameters
    ----------
    """
    try:
        import casacore.tables as _casatables  # type: ignore
        import numpy as _np

        _tb = _casatables.table
    except ImportError:
        return

    try:
        with _tb(ms_path, readonly=False) as tb:
            nrow = tb.nrows()
            colnames = set(tb.colnames())
            has_ws = "WEIGHT_SPECTRUM" in colnames
            ws_bad = False
            for i in range(nrow):
                try:
                    data = tb.getcell("DATA", i)
                except (RuntimeError, KeyError, IndexError):
                    # RuntimeError: CASA errors, KeyError: missing col, IndexError: bad row
                    continue
                target_shape = getattr(data, "shape", None)
                if not target_shape or len(target_shape) != 2:
                    continue
                nchan, npol = int(target_shape[0]), int(target_shape[1])
                # FLAG
                try:
                    f = tb.getcell("FLAG", i)
                    if f is None or getattr(f, "shape", None) != (nchan, npol):
                        raise RuntimeError("FLAG shape mismatch")
                except (RuntimeError, KeyError, IndexError):
                    tb.putcell("FLAG", i, _np.zeros((nchan, npol), dtype=bool))
                # WEIGHT_SPECTRUM
                if has_ws:
                    try:
                        ws_val = tb.getcell("WEIGHT_SPECTRUM", i)
                        if ws_val is None or getattr(ws_val, "shape", None) != (
                            nchan,
                            npol,
                        ):
                            raise RuntimeError("WS shape mismatch")
                    except (RuntimeError, KeyError, IndexError):
                        try:
                            w = tb.getcell("WEIGHT", i)
                            w = _np.asarray(w).reshape(-1)
                            if w.size != npol:
                                w = _np.ones((npol,), dtype=float)
                        except (RuntimeError, KeyError, IndexError):
                            w = _np.ones((npol,), dtype=float)
                        ws = _np.repeat(w[_np.newaxis, :], nchan, axis=0)
                        tb.putcell("WEIGHT_SPECTRUM", i, ws)
                        ws_bad = True
            if has_ws and ws_bad:
                try:
                    tb.removecols(["WEIGHT_SPECTRUM"])
                except (RuntimeError, OSError):
                    # RuntimeError: CASA errors, OSError: file issues
                    pass
    except (RuntimeError, OSError, ImportError):
        # RuntimeError: CASA errors, OSError: file issues, ImportError: casacore
        return


@require_casa6_python
def _initialize_weights(ms_path: str) -> None:
    """Initialize WEIGHT_SPECTRUM via casatasks.initweights.

    NOTE: CASA's initweights does NOT have doweight or doflag parameters.
    When wtmode='weight', it initializes WEIGHT_SPECTRUM from the existing WEIGHT column.

    Parameters
    ----------
    """
    try:
        from dsa110_contimg.core.calibration.casa_service import CASAService

        service = CASAService()

        # NOTE: When wtmode='weight', initweights initializes WEIGHT_SPECTRUM from WEIGHT column
        # dowtsp=True creates/updates WEIGHT_SPECTRUM column
        service.initweights(vis=ms_path, wtmode="weight", dowtsp=True)
    except Exception:
        # Non-fatal: initweights can fail on edge cases; downstream tools may
        # still work.
        pass


def _fix_mount_type_in_ms(ms_path: str) -> None:
    """Normalize ANTENNA.MOUNT values to CASA-supported strings.

    Parameters
    ----------
    """
    try:
        import casacore.tables as _casatables  # type: ignore

        _tb = _casatables.table

        with _tb(ms_path + "/ANTENNA", readonly=False) as ant_table:
            mounts = ant_table.getcol("MOUNT")
            fixed = []
            for m in mounts:
                normalized = str(m or "").lower().strip()
                if normalized in (
                    "alt-az",
                    "altaz",
                    "alt_az",
                    "alt az",
                    "az-el",
                    "azel",
                ):
                    fixed.append("alt-az")
                elif normalized in ("equatorial", "eq"):
                    fixed.append("equatorial")
                elif normalized in ("x-y", "xy"):
                    fixed.append("x-y")
                elif normalized in ("spherical", "sphere"):
                    fixed.append("spherical")
                else:
                    fixed.append("alt-az")
            ant_table.putcol("MOUNT", fixed)
    except (RuntimeError, OSError, ImportError):
        # Non-fatal normalization
        # RuntimeError: CASA errors, OSError: file issues, ImportError: casacore
        pass


def _fix_field_phase_centers_from_times(ms_path: str) -> None:
    """Fix FIELD table PHASE_DIR/REFERENCE_DIR with correct time-dependent RA values.

    This function corrects a bug where pyuvdata.write_ms() may assign incorrect RA
    values to fields when using time-dependent phase centers. For meridian-tracking
    phasing (RA = LST), each field should have RA corresponding to LST at that field's
    time, not a single midpoint RA.

    The function:
    1. Reads the main table to determine which times correspond to which FIELD_ID
    2. For each field, calculates the correct RA = LST(time) at that field's time
    3. Updates PHASE_DIR and REFERENCE_DIR in the FIELD table with correct values

    Parameters
    ----------
    ms_path :
        Path to Measurement Set
    """
    try:
        import astropy.units as u  # type: ignore
        import numpy as _np
        
        from dsa110_contimg.core.adapters.casa import casa_adapter
        from dsa110_contimg.core.conversion.helpers_coordinates import get_meridian_coords
        from dsa110_contimg.common.utils.time_utils import detect_casa_time_format
        
        if not casa_adapter.is_available:
            return
            
    except ImportError:
        # Non-fatal: if dependencies aren't available, skip this fix
        return

    tb = casa_adapter.table()
    
    try:
        # Read main table to get FIELD_ID and TIME mapping
        # Use nomodify=True for read-only access
        tb.open(ms_path, nomodify=True)
        if tb.nrows() == 0:
            tb.close()
            return

        field_ids = tb.getcol("FIELD_ID")
        times = tb.getcol("TIME")  # CASA TIME is in seconds since MJD epoch
        tb.close()

        # Get unique field IDs and their corresponding times
        unique_field_ids = _np.unique(field_ids)
        field_times = {}
        for fid in unique_field_ids:
            mask = field_ids == fid
            field_times[int(fid)] = _np.mean(times[mask])  # Use mean time for the field

        # Read FIELD table
        # Use nomodify=False for write access
        tb.open(ms_path + "/FIELD", nomodify=False)
        nfields = tb.nrows()
        if nfields == 0:
            tb.close()
            return

        # Get current PHASE_DIR and REFERENCE_DIR
        colnames = tb.colnames()
        has_phase_dir = "PHASE_DIR" in colnames
        has_ref_dir = "REFERENCE_DIR" in colnames

        if not has_phase_dir and not has_ref_dir:
            tb.close()
            return  # Can't fix if neither column exists

        phase_dir = tb.getcol("PHASE_DIR") if has_phase_dir else None
        ref_dir = tb.getcol("REFERENCE_DIR") if has_ref_dir else None

        # Get pointing declination from first field (should be constant)
        if phase_dir is not None:
            pt_dec_rad = phase_dir[0, 0, 1]  # Dec from first field
        elif ref_dir is not None:
            pt_dec_rad = ref_dir[0, 0, 1]
        else:
            tb.close()
            return

        pt_dec = pt_dec_rad * u.rad

        # Fix each field's phase center
        updated = False

        for field_idx in range(nfields):
            # Get time for this field
            if field_idx in field_times:
                time_sec = field_times[field_idx]
                _, time_mjd = detect_casa_time_format(time_sec)
            else:
                # Fallback: use mean time from main table
                mean_time_sec = _np.mean(times)
                _, time_mjd = detect_casa_time_format(mean_time_sec)

            # Calculate correct RA = LST(time) at meridian
            phase_ra, phase_dec = get_meridian_coords(pt_dec, time_mjd)
            ra_rad = float(phase_ra.to_value(u.rad))
            dec_rad = float(phase_dec.to_value(u.rad))

            # Update PHASE_DIR if it exists
            if has_phase_dir:
                current_ra = phase_dir[field_idx, 0, 0]
                current_dec = phase_dir[field_idx, 0, 1]
                # Only update if significantly different (more than 1 arcsec)
                ra_diff_rad = abs(ra_rad - current_ra)
                dec_diff_rad = abs(dec_rad - current_dec)
                if ra_diff_rad > _np.deg2rad(1.0 / 3600.0) or dec_diff_rad > _np.deg2rad(
                    1.0 / 3600.0
                ):
                    phase_dir[field_idx, 0, 0] = ra_rad
                    phase_dir[field_idx, 0, 1] = dec_rad
                    updated = True

            # Update REFERENCE_DIR if it exists
            if has_ref_dir:
                current_ra = ref_dir[field_idx, 0, 0]
                current_dec = ref_dir[field_idx, 0, 1]
                # Only update if significantly different (more than 1 arcsec)
                ra_diff_rad = abs(ra_rad - current_ra)
                dec_diff_rad = abs(dec_rad - current_dec)
                if ra_diff_rad > _np.deg2rad(1.0 / 3600.0) or dec_diff_rad > _np.deg2rad(
                    1.0 / 3600.0
                ):
                    ref_dir[field_idx, 0, 0] = ra_rad
                    ref_dir[field_idx, 0, 1] = dec_rad
                    updated = True

        # Write back updated values
        if updated:
            if has_phase_dir:
                tb.putcol("PHASE_DIR", phase_dir)
            if has_ref_dir:
                tb.putcol("REFERENCE_DIR", ref_dir)
                
        tb.close()
        
    except (RuntimeError, OSError, ValueError, KeyError) as e:
        # Close table if open
        try:
            tb.close()
        except:
            pass
            
        # Non-fatal: if fixing fails, log warning but don't crash
        import logging
        logger = logging.getLogger(__name__)
        logger.warning("Could not fix FIELD table phase centers (non-fatal): %s", e)


def _ensure_observation_table_valid(ms_path: str) -> None:
    """Ensure OBSERVATION table exists and has at least one valid row.

    This fixes MS files where the OBSERVATION table is empty or malformed,
    which causes CASA msmetadata to fail with "Observation ID -1 out of range".

    Parameters
    ----------
    """
    try:
        import casacore.tables as _casatables
        import numpy as _np

        _tb = _casatables.table
    except ImportError:
        return

    try:
        with _tb(f"{ms_path}::OBSERVATION", readonly=False) as obs_tb:
            # If table is empty, create a default observation row
            if obs_tb.nrows() == 0:
                import logging

                logger = logging.getLogger(__name__)
                logger.warning(f"OBSERVATION table is empty in {ms_path}, creating default row")

                # Get telescope name from environment or use default
                telescope_name = os.getenv("PIPELINE_TELESCOPE_NAME", "DSA_110")

                # Create a default observation row
                # CASA requires specific columns - use minimal valid values
                default_values = {
                    "TIME_RANGE": _np.array([0.0, 0.0], dtype=_np.float64),
                    "LOG": "",
                    "SCHEDULE": "",
                    "FLAG_ROW": False,
                    "OBSERVER": "",
                    "PROJECT": "",
                    "RELEASE_DATE": 0.0,
                    "SCHEDULE_TYPE": "",
                    "TELESCOPE_NAME": telescope_name,
                }

                # Add row with default values
                obs_tb.addrows(1)
                for col, val in default_values.items():
                    if col in obs_tb.colnames():
                        obs_tb.putcell(col, 0, val)

                logger.info(f"Created default OBSERVATION row in {ms_path}")

    except (RuntimeError, OSError, KeyError):
        # Non-fatal: best-effort fix only
        # RuntimeError: CASA errors, OSError: file issues, KeyError: missing columns
        import logging

        logger = logging.getLogger(__name__)
        logger.warning("Could not ensure OBSERVATION table validity (non-fatal)", exc_info=True)


def _ensure_state_table_valid(ms_path: str) -> None:
    """Ensure STATE table exists and has at least one valid row.

    This fixes MS files where the STATE table is empty, which causes CASA
    msmetadata.timesforscans() to fail with "No matching scans found" even
    when the OBSERVATION table TIME_RANGE is correct.

    The STATE table is referenced by STATE_ID column in the main table.
    If STATE_ID values point to non-existent STATE rows, CASA scan queries fail.

    Parameters
    ----------
    """
    try:
        import casacore.tables as _casatables

        _tb = _casatables.table
    except ImportError:
        return

    try:
        with _tb(f"{ms_path}::STATE", readonly=False) as state_tb:
            # If table is empty but main table has STATE_ID = 0, create a default row
            if state_tb.nrows() == 0:
                # Check if main table references STATE_ID = 0
                with _tb(ms_path, readonly=True) as main_tb:
                    if "STATE_ID" not in main_tb.colnames():
                        return
                    state_ids = main_tb.getcol("STATE_ID")
                    if state_ids is None or len(state_ids) == 0:
                        return
                    # Only fix if STATE_ID 0 is referenced
                    if 0 not in set(state_ids):
                        return

                import logging

                logger = logging.getLogger(__name__)
                logger.warning(f"STATE table is empty in {ms_path}, creating default row")

                # Create a default state row with minimal valid values
                # These columns are required by the MS2 standard
                state_tb.addrows(1)
                if "CAL" in state_tb.colnames():
                    state_tb.putcell("CAL", 0, 0.0)
                if "FLAG_ROW" in state_tb.colnames():
                    state_tb.putcell("FLAG_ROW", 0, False)
                if "LOAD" in state_tb.colnames():
                    state_tb.putcell("LOAD", 0, 0.0)
                if "OBS_MODE" in state_tb.colnames():
                    state_tb.putcell("OBS_MODE", 0, "OBSERVE_TARGET#ON_SOURCE")
                if "REF" in state_tb.colnames():
                    state_tb.putcell("REF", 0, False)
                if "SIG" in state_tb.colnames():
                    state_tb.putcell("SIG", 0, True)
                if "SUB_SCAN" in state_tb.colnames():
                    state_tb.putcell("SUB_SCAN", 0, 0)

                logger.info(f"Created default STATE row in {ms_path}")

    except (RuntimeError, OSError, KeyError):
        # Non-fatal: best-effort fix only
        import logging

        logger = logging.getLogger(__name__)
        logger.warning("Could not ensure STATE table validity (non-fatal)", exc_info=True)


def normalize_phaseshifted_ms_to_single_field(ms_path: str, ra_deg: float, dec_deg: float) -> None:
    """Normalize a phaseshift+concat MS so calibration can treat it as one field.

    CASA's multi-field workaround (phaseshift each field then concat) can produce
    an MS where concat offsets/renumbers FIELD_ID values (e.g., 0, 25..47). The
    pipeline calibration solver assumes the phaseshifted MS has a single field
    (index 0). This routine enforces that contract by:
    - Setting FIELD_ID=0 in the MAIN table (and key subtables when present)
    - Ensuring FIELD row 0 has the requested phase center

    This is safe for calibration because the data have already been phaseshifted
    to the target phase center; we are only normalizing metadata/IDs.

    Parameters
    ----------
    """
    import numpy as _np

    _tb = None
    _use_casatools = False
    try:
        import casacore.tables as _casatables

        _tb = _casatables.table
    except ImportError:
        try:
            from casatools import table as _casatools_table

            _tb = _casatools_table
            _use_casatools = True
        except Exception:
            return

    import logging

    logger = logging.getLogger(__name__)

    try:
        ra_rad = float(_np.radians(ra_deg))
        dec_rad = float(_np.radians(dec_deg))
        phase_dir_cell = _np.array([[ra_rad, dec_rad]], dtype=_np.float64)  # shape (1, 2)

        if _use_casatools:
            main = _tb()
            main.open(ms_path, nomodify=False)
            if "FIELD_ID" in main.colnames() and main.nrows() > 0:
                main.putcol("FIELD_ID", _np.zeros(main.nrows(), dtype=_np.int32))
            main.close()

            field_tb = _tb()
            field_tb.open(f"{ms_path}::FIELD", nomodify=False)
            if field_tb.nrows() == 0:
                field_tb.close()
                return
            for col in ("PHASE_DIR", "DELAY_DIR", "REFERENCE_DIR"):
                if col in field_tb.colnames():
                    cell = phase_dir_cell.reshape((1, 2))
                    field_tb.putcell(col, 0, cell)
            field_tb.close()
        else:
            with _tb(ms_path, readonly=False) as main:
                if "FIELD_ID" in main.colnames() and main.nrows() > 0:
                    main.putcol("FIELD_ID", _np.zeros(main.nrows(), dtype=_np.int32))

            with _tb(f"{ms_path}::FIELD", readonly=False) as field_tb:
                if field_tb.nrows() == 0:
                    return
                for col in ("PHASE_DIR", "DELAY_DIR", "REFERENCE_DIR"):
                    if col in field_tb.colnames():
                        field_tb.putcell(col, 0, phase_dir_cell)

        # Keep other common FIELD_ID-bearing subtables consistent when present.
        for subtable_name in ("POINTING", "SOURCE", "FLAG_CMD"):
            try:
                if _use_casatools:
                    t = _tb()
                    t.open(f"{ms_path}::{subtable_name}", nomodify=False)
                    if "FIELD_ID" in t.colnames() and t.nrows() > 0:
                        t.putcol("FIELD_ID", _np.zeros(t.nrows(), dtype=_np.int32))
                    t.close()
                else:
                    with _tb(f"{ms_path}::{subtable_name}", readonly=False) as t:
                        if "FIELD_ID" in t.colnames() and t.nrows() > 0:
                            t.putcol("FIELD_ID", _np.zeros(t.nrows(), dtype=_np.int32))
            except (RuntimeError, OSError, KeyError):
                continue

        logger.info(
            "Normalized phaseshifted MS to single FIELD_ID=0: %s (RA=%.4f deg, Dec=%.4f deg)",
            ms_path,
            ra_deg,
            dec_deg,
        )

    except (RuntimeError, OSError, KeyError):
        logger.warning(
            "Could not normalize phaseshifted MS field IDs (non-fatal)",
            exc_info=True,
        )


def normalize_scan_numbers_to_zero(ms_path: str) -> None:
    import numpy as _np

    _tb = None
    _use_casatools = False
    try:
        import casacore.tables as _casatables

        _tb = _casatables.table
    except ImportError:
        try:
            from casatools import table as _casatools_table

            _tb = _casatools_table
            _use_casatools = True
        except Exception:
            return

    import logging

    logger = logging.getLogger(__name__)

    try:
        if _use_casatools:
            main = _tb()
            main.open(ms_path, nomodify=False)
            if "SCAN_NUMBER" not in main.colnames() or main.nrows() == 0:
                main.close()
                return
            scans = main.getcol("SCAN_NUMBER")
            if scans is None or len(scans) == 0:
                main.close()
                return
            min_scan = int(_np.min(scans))
            if min_scan > 0:
                main.putcol("SCAN_NUMBER", scans - min_scan)
                logger.info("Normalized SCAN_NUMBER to start at 0: %s (offset=%d)", ms_path, min_scan)
            main.close()
        else:
            with _tb(ms_path, readonly=False) as main:
                if "SCAN_NUMBER" not in main.colnames() or main.nrows() == 0:
                    return
                scans = main.getcol("SCAN_NUMBER")
                if scans is None or len(scans) == 0:
                    return
                min_scan = int(_np.min(scans))
                if min_scan > 0:
                    main.putcol("SCAN_NUMBER", scans - min_scan)
                    logger.info(
                        "Normalized SCAN_NUMBER to start at 0: %s (offset=%d)", ms_path, min_scan
                    )

    except (RuntimeError, OSError, KeyError):
        logger.warning("Could not normalize SCAN_NUMBER (non-fatal)", exc_info=True)


def _fix_observation_id_column(ms_path: str) -> None:
    """Ensure OBSERVATION_ID column in main table has valid values (>= 0).

    This fixes MS files where OBSERVATION_ID values are negative or invalid,
    which causes CASA msmetadata to fail.

    Parameters
    ----------
    """
    try:
        import casacore.tables as _casatables
        import numpy as _np

        _tb = _casatables.table
    except ImportError:
        return

    try:
        with _tb(ms_path, readonly=False) as main_tb:
            if "OBSERVATION_ID" not in main_tb.colnames():
                return

            obs_ids = main_tb.getcol("OBSERVATION_ID")
            if obs_ids is None or len(obs_ids) == 0:
                return

            # Check if any values are negative
            negative_mask = obs_ids < 0
            if _np.any(negative_mask):
                import logging

                logger = logging.getLogger(__name__)
                n_negative = _np.sum(negative_mask)
                logger.warning(
                    f"Found {n_negative} rows with negative OBSERVATION_ID in {ms_path}, fixing"
                )

                # Fix negative values to 0
                fixed_ids = obs_ids.copy()
                fixed_ids[negative_mask] = 0
                main_tb.putcol("OBSERVATION_ID", fixed_ids)

                logger.info(f"Fixed {n_negative} negative OBSERVATION_ID values in {ms_path}")

    except (RuntimeError, OSError, KeyError):
        # Non-fatal: best-effort fix only
        # RuntimeError: CASA errors, OSError: file issues, KeyError: missing columns
        import logging

        logger = logging.getLogger(__name__)
        logger.warning("Could not fix OBSERVATION_ID column (non-fatal)", exc_info=True)


def _fix_observation_time_range(ms_path: str) -> None:
    """Fix OBSERVATION table TIME_RANGE by reading from main table TIME column.

    This corrects MS files where OBSERVATION table TIME_RANGE is [0, 0] or invalid.
    The TIME column in the main table is the authoritative source.

    Uses the same format detection logic as extract_ms_time_range() to handle
    both TIME formats (seconds since MJD 0 vs seconds since MJD 51544.0).

    Parameters
    ----------
    """
    try:
        import casacore.tables as _casatables
        import numpy as _np

        _tb = _casatables.table

        from dsa110_contimg.common.utils.time_utils import (
            DEFAULT_YEAR_RANGE,
            detect_casa_time_format,
            validate_time_mjd,
        )
    except ImportError:
        return

    try:
        # First ensure OBSERVATION table exists and has at least one row
        _ensure_observation_table_valid(ms_path)

        # Read TIME column from main table (authoritative source)
        with _tb(ms_path, readonly=True) as main_tb:
            if "TIME" not in main_tb.colnames() or main_tb.nrows() == 0:
                return

            times = main_tb.getcol("TIME")
            if len(times) == 0:
                return

            t0_sec = float(_np.min(times))
            t1_sec = float(_np.max(times))

        # Detect correct format using the same logic as extract_ms_time_range()
        # This handles both formats: seconds since MJD 0 vs seconds since MJD 51544.0
        _, start_mjd = detect_casa_time_format(t0_sec, DEFAULT_YEAR_RANGE)
        _, end_mjd = detect_casa_time_format(t1_sec, DEFAULT_YEAR_RANGE)

        # Validate using astropy
        if not (
            validate_time_mjd(start_mjd, DEFAULT_YEAR_RANGE)
            and validate_time_mjd(end_mjd, DEFAULT_YEAR_RANGE)
        ):
            # Invalid dates, skip update
            return

        # OBSERVATION table TIME_RANGE should be in the same format as the main table TIME
        # (seconds, not MJD days). Use the raw seconds values directly.
        # Shape should be [2] (start, end), not [1, 2]
        time_range_sec = _np.array([t0_sec, t1_sec], dtype=_np.float64)

        # Update OBSERVATION table
        with _tb(f"{ms_path}::OBSERVATION", readonly=False) as obs_tb:
            if obs_tb.nrows() == 0:
                # Should not happen after _ensure_observation_table_valid, but handle gracefully
                return

            if "TIME_RANGE" not in obs_tb.colnames():
                return

            # Check if TIME_RANGE is invalid (all zeros or very small)
            existing_tr = obs_tb.getcol("TIME_RANGE")
            if existing_tr is not None and existing_tr.size > 0:
                # Handle shape (nrows, 2) - getcol returns array of shape (nrows, 2)
                # where each row is [start_time, end_time]
                tr_flat = _np.asarray(existing_tr).flatten()
                if len(tr_flat) >= 2:
                    existing_t0 = float(tr_flat[0])
                    existing_t1 = float(tr_flat[1])
                else:
                    existing_t0 = 0.0
                    existing_t1 = 0.0

                # Only update if TIME_RANGE is invalid (zero or very small)
                if existing_t0 > 1.0 and existing_t1 > existing_t0:
                    # TIME_RANGE is already valid, don't overwrite
                    return

            # Update TIME_RANGE for all observation rows
            for row in range(obs_tb.nrows()):
                obs_tb.putcell("TIME_RANGE", row, time_range_sec)

        import logging

        logger = logging.getLogger(__name__)
        logger.debug(
            f"Fixed OBSERVATION table TIME_RANGE for {ms_path}: "
            f"{t0_sec:.1f} to {t1_sec:.1f} seconds "
            f"({start_mjd:.8f} to {end_mjd:.8f} MJD)"
        )
    except (RuntimeError, OSError, KeyError, ValueError):
        # Non-fatal: best-effort fix only
        # RuntimeError: CASA errors, OSError: file issues,
        # KeyError: missing columns, ValueError: time conversion
        import logging

        logger = logging.getLogger(__name__)
        logger.warning("Could not fix OBSERVATION table TIME_RANGE (non-fatal)", exc_info=True)


@require_casa6_python
def configure_ms_for_imaging(
    ms_path: str,
    *,
    ensure_columns: bool = True,
    ensure_flag_and_weight: bool = True,
    do_initweights: bool = True,
    fix_mount: bool = True,
    stamp_observation_telescope: bool = True,
    validate_columns: bool = True,
    rename_calibrator_fields: bool = True,
    catalog_path: str | None = None,
) -> None:
    """Make a Measurement Set safe and ready for imaging and calibration.

        This function performs essential post-conversion setup to ensure an MS is
        ready for downstream processing (calibration, imaging). It uses consistent
        error handling: critical failures raise exceptions, while non-critical issues
        log warnings and continue.

        What this function does
    ----------------------
        1. Ensures imaging columns exist: Creates MODEL_DATA and CORRECTED_DATA
        columns if missing, and populates them with properly-shaped arrays
        2. Ensures flag/weight arrays: Creates FLAG and WEIGHT_SPECTRUM arrays
        with correct shapes matching the DATA column
        3. Initializes weights: Runs CASA's initweights task to set proper
        weight values based on data quality
        4. Fixes antenna mount types: Normalizes ANTENNA.MOUNT values to
        CASA-compatible format
        5. Stamps telescope identity: Sets consistent telescope name and location
        6. Fixes phase centers: Updates FIELD table phase centers based on
        observation times
        7. Fixes observation time range: Updates OBSERVATION table with correct
        time range

    Parameters
    ----------
    ms_path : str
        Path to the Measurement Set (directory path).
    ensure_columns : bool, optional
        Ensure MODEL_DATA and CORRECTED_DATA columns exist and are populated.
        Default is True.
    ensure_flag_and_weight : bool, optional
        Ensure FLAG and WEIGHT_SPECTRUM arrays exist and are well-shaped.
        Default is True.
    do_initweights : bool, optional
        Run casatasks.initweights with WEIGHT_SPECTRUM initialization enabled.
        Default is True.
    fix_mount : bool, optional
        Normalize ANTENNA.MOUNT values to CASA-compatible format.
        Default is True.
    stamp_observation_telescope : bool, optional
        Set consistent telescope name and location metadata.
        Default is True.
    validate_columns : bool, optional
        Validate that columns exist and contain data after creation.
        Set to False for high-throughput scenarios where validation overhead
        is unacceptable. Default is True.
    rename_calibrator_fields : bool, optional
        Auto-detect and rename fields containing known calibrators.
        Uses VLA calibrator catalog to identify which field contains a calibrator,
        then renames it from 'meridian_icrs_t{i}' to '{calibrator}_t{i}'.
        Recommended for drift-scan observations. Default is True.
    catalog_path : str or None, optional
        Path to VLA calibrator catalog (SQLite or CSV).
        If None, uses automatic resolution (prefers SQLite).
        Only used if rename_calibrator_fields=True. Default is None.

    Raises
    ------
        ConversionError
        If MS path does not exist, is not readable, or becomes unreadable
        after configuration (critical failures).

    Examples
    --------
        Basic usage after converting UVH5 to MS:

        >>> from dsa110_contimg.core.conversion.ms_utils import configure_ms_for_imaging
        >>> configure_ms_for_imaging("/path/to/observation.ms")

        Configure only essential columns (skip weight initialization):

        >>> configure_ms_for_imaging(
        ...     "/path/to/observation.ms",
        ...     do_initweights=False
        ... )

        Minimal configuration (only columns and flags):

        >>> configure_ms_for_imaging(
        ...     "/path/to/observation.ms",
        ...     do_initweights=False,
        ...     fix_mount=False,
        ...     stamp_observation_telescope=False
        ... )

    Notes
    -----
        - This function should be called after converting UVH5 to MS format.
        - All operations are idempotent (safe to call multiple times).
        - Non-critical failures (e.g., column population issues) are logged as
        warnings but don't stop execution.
        - Critical failures (e.g., MS not found) raise ConversionError with
        context and suggestions.
    """
    if not isinstance(ms_path, str):
        ms_path = os.fspath(ms_path)

    # CRITICAL: Validate MS exists and is readable
    from dsa110_contimg.common.utils.exceptions import ConversionError

    if not os.path.exists(ms_path):
        raise ConversionError(
            f"MS does not exist: {ms_path}",
            context={"ms_path": ms_path, "operation": "configure_ms_for_imaging"},
            suggestion="Check that the MS path is correct and the file exists",
        )
    if not os.path.isdir(ms_path):
        raise ConversionError(
            f"MS path is not a directory: {ms_path}",
            context={"ms_path": ms_path, "operation": "configure_ms_for_imaging"},
            suggestion="Measurement Sets are directories, not files. Check the path.",
        )
    if not os.access(ms_path, os.R_OK):
        raise ConversionError(
            f"MS is not readable: {ms_path}",
            context={"ms_path": ms_path, "operation": "configure_ms_for_imaging"},
            suggestion="Check file permissions: ls -ld " + ms_path,
        )

    # Initialize logger early for use in error handling
    import logging

    logger = logging.getLogger(__name__)

    # Track which operations succeeded for summary logging
    operations_status = {
        "columns": "skipped",
        "flag_weight": "skipped",
        "initweights": "skipped",
        "mount_fix": "skipped",
        "telescope_stamp": "skipped",
        "field_phase_centers": "skipped",
        "observation_time_range": "skipped",
    }

    if ensure_columns:
        try:
            _ensure_imaging_columns_exist(ms_path)
            _ensure_imaging_columns_populated(ms_path)

            # CRITICAL: Validate columns actually exist and are populated (if enabled)
            if validate_columns:
                import casacore.tables as _casatables

                _tb = _casatables.table

                with _tb(ms_path, readonly=True) as tb:
                    colnames = set(tb.colnames())
                    missing = []
                    if "MODEL_DATA" not in colnames:
                        missing.append("MODEL_DATA")
                    if "CORRECTED_DATA" not in colnames:
                        missing.append("CORRECTED_DATA")

                    if missing:
                        error_msg = (
                            f"CRITICAL: Required imaging columns missing after creation: {missing}. "
                            f"MS {ms_path} is not ready for calibration/imaging."
                        )
                        logger.error(error_msg)
                        raise ConversionError(
                            error_msg,
                            context={"ms_path": ms_path, "missing_columns": missing},
                            suggestion="Check MS file permissions and disk space. "
                            "Try recreating the MS if the issue persists.",
                        )

                    # Verify columns have data (at least one row)
                    if tb.nrows() > 0:
                        try:
                            model_sample = tb.getcell("MODEL_DATA", 0)
                            corrected_sample = tb.getcell("CORRECTED_DATA", 0)
                            if model_sample is None or corrected_sample is None:
                                logger.warning(
                                    f"Imaging columns exist but contain None values in {ms_path}"
                                )
                        except Exception as e:
                            logger.warning(f"Could not verify column data in {ms_path}: {e}")
                    logger.info(f":check_mark: Imaging columns verified in {ms_path}")
            else:
                logger.debug(f"Imaging columns created (validation skipped) in {ms_path}")

            operations_status["columns"] = "success"
        except ConversionError:
            # Re-raise ConversionError (critical failures)
            raise
        except Exception as e:
            operations_status["columns"] = f"failed: {e}"
            error_msg = (
                f"CRITICAL: Failed to create/verify imaging columns in {ms_path}: {e}. "
                "MS is not ready for calibration/imaging."
            )
            logger.error(error_msg, exc_info=True)
            raise ConversionError(
                error_msg,
                context={"ms_path": ms_path, "error": str(e)},
                suggestion="Check MS file permissions, disk space, and CASA installation. "
                "Try recreating the MS if the issue persists.",
            ) from e

    if ensure_flag_and_weight:
        try:
            _ensure_flag_and_weight_spectrum(ms_path)
            operations_status["flag_weight"] = "success"
        except Exception as e:
            operations_status["flag_weight"] = f"failed: {e}"
            # Non-fatal: continue with other operations

    if do_initweights:
        try:
            _initialize_weights(ms_path)
            operations_status["initweights"] = "success"
        except Exception as e:
            operations_status["initweights"] = f"failed: {e}"
            # Non-fatal: initweights often fails on edge cases

    if fix_mount:
        try:
            _fix_mount_type_in_ms(ms_path)
            operations_status["mount_fix"] = "success"
        except Exception as e:
            operations_status["mount_fix"] = f"failed: {e}"
            # Non-fatal: mount type normalization is optional

    if stamp_observation_telescope:
        try:
            import casacore.tables as _casatables  # type: ignore

            _tb = _casatables.table

            name = os.getenv("PIPELINE_TELESCOPE_NAME", "DSA_110")
            with _tb(ms_path + "::OBSERVATION", readonly=False) as tb:
                n = tb.nrows()
                if n:
                    tb.putcol("TELESCOPE_NAME", [name] * n)
            operations_status["telescope_stamp"] = "success"
        except Exception as e:
            operations_status["telescope_stamp"] = f"failed: {e}"
            # Non-fatal: telescope name stamping is optional

    # Fix FIELD table phase centers (corrects RA assignment bug)
    try:
        _fix_field_phase_centers_from_times(ms_path)
        operations_status["field_phase_centers"] = "success"
    except Exception as e:
        operations_status["field_phase_centers"] = f"failed: {e}"
        # Non-fatal: field phase center fix is best-effort

    # Fix OBSERVATION table and OBSERVATION_ID column (critical for CASA msmetadata)
    try:
        _ensure_observation_table_valid(ms_path)
        _fix_observation_id_column(ms_path)
        operations_status["observation_table"] = "success"
    except Exception as e:
        operations_status["observation_table"] = f"failed: {e}"
        # Non-fatal: observation table fix is best-effort

    # Fix STATE table (critical for CASA msmetadata.timesforscans())
    try:
        _ensure_state_table_valid(ms_path)
        operations_status["state_table"] = "success"
    except Exception as e:
        operations_status["state_table"] = f"failed: {e}"
        # Non-fatal: state table fix is best-effort

    # Fix OBSERVATION table TIME_RANGE (corrects missing/invalid time range)
    try:
        _fix_observation_time_range(ms_path)
        operations_status["observation_time_range"] = "success"
    except Exception as e:
        operations_status["observation_time_range"] = f"failed: {e}"
        # Non-fatal: observation time range fix is best-effort

    # Auto-detect and rename calibrator fields (recommended for drift-scan observations)
    if rename_calibrator_fields:
        try:
            from dsa110_contimg.core.calibration.field_naming import (
                rename_calibrator_fields_from_catalog,
            )

            result = rename_calibrator_fields_from_catalog(
                ms_path,
                catalog_path=catalog_path,
            )
            if result:
                cal_name, field_idx = result
                operations_status["calibrator_renaming"] = "success"
                logger.info(
                    f":check_mark: Auto-renamed field {field_idx} to '{cal_name}_t{field_idx}'"
                )
            else:
                operations_status["calibrator_renaming"] = "no calibrator found"
                logger.debug("No calibrator found in MS for field renaming")
        except Exception as e:
            operations_status["calibrator_renaming"] = f"failed: {e}"
            logger.debug(f"Calibrator field renaming not available: {e}")
            # Non-fatal: field renaming is optional

    # Summary logging - report what worked and what didn't
    success_ops = [op for op, status in operations_status.items() if status == "success"]
    failed_ops = [
        f"{op}({status.split(': ')[1]})"
        for op, status in operations_status.items()
        if status.startswith("failed")
    ]

    if success_ops:
        logger.info(f":check_mark: MS configuration completed: {', '.join(success_ops)}")
    if failed_ops:
        logger.warning(f":warning_sign: MS configuration partial failures: {'; '.join(failed_ops)}")

    # Final validation: verify MS is still readable after all operations
    try:
        import casacore.tables as _casatables

        _tb = _casatables.table

        with _tb(ms_path, readonly=True) as tb:
            if tb.nrows() == 0:
                raise RuntimeError(f"MS has no data after configuration: {ms_path}")
    except Exception as e:
        raise RuntimeError(f"MS became unreadable after configuration: {e}")


def inject_provenance_metadata(ms_path: str, job_id: str, config_hash: str) -> None:
    """
    Inject cryptographic provenance metadata into the MS HISTORY table.
    
    This creates an unbreakable link between the data file and the 
    provenance database record.
    
    Parameters
    ----------
    ms_path : str
        Path to the Measurement Set
    job_id : str
        Unique Job ID from ProvenanceTracker
    config_hash : str
        SHA-256 hash of the pipeline configuration
    """
    from dsa110_contimg.core.adapters.casa import casa_adapter
    
    if not casa_adapter.is_available:
        return
        
    try:
        tb = casa_adapter.table()
        history_path = f"{ms_path}/HISTORY"
        
        # Open HISTORY table (it's a sub-table)
        tb.open(history_path, nomodify=False)
        
        # Add rows for provenance
        # Standard CASA HISTORY table has columns: 
        #   TIME (double), OBSERVATION_ID (int), MESSAGE (string), PRIORITY (string), 
        #   ORIGIN (string), OBJECT_ID (int), APPLICATION (string), CLI_COMMAND (string), 
        #   APP_PARAMS (string array)
        
        import time
        now = time.time() / 86400.0 + 2400000.5 # MJD approx
        
        # We inject as generic history messages
        rows = [
            f"DSA-110 PROVENANCE: JOB_ID={job_id}",
            f"DSA-110 PROVENANCE: CONFIG_HASH={config_hash}"
        ]
        
        tb.addrows(len(rows))
        
        # Get last N rows indices
        n_rows = tb.nrows()
        start_row = n_rows - len(rows)
        
        for i, msg in enumerate(rows):
            row_idx = start_row + i
            # Note: We must be careful with column types. 
            # Safest is to use putcell for robust writing if addrows doesn't init
            tb.putcell("MESSAGE", row_idx, msg)
            tb.putcell("ORIGIN", row_idx, "dsa110-pipeline")
            tb.putcell("PRIORITY", row_idx, "INFO")
            tb.putcell("APPLICATION", row_idx, "dsa110-contimg")
            # TIME is required
            tb.putcell("TIME", row_idx, now)
        
        tb.close()
        
    except Exception:
        # Don't fail the pipeline for metadata injection
        pass


def get_provenance_metadata(ms_path: str) -> dict[str, str]:
    """
    Extract provenance metadata from the MS HISTORY table.
    
    Reads the 'DSA-110 PROVENANCE: KEY=VALUE' entries injected by
    inject_provenance_metadata().
    
    Parameters
    ----------
    ms_path : str
        Path to the Measurement Set
        
    Returns
    -------
    dict
        Dictionary containing 'job_id' and 'config_hash' if found.
    """
    from dsa110_contimg.core.adapters.casa import casa_adapter
    
    metadata = {}
    
    if not casa_adapter.is_available:
        return metadata
        
    try:
        tb = casa_adapter.table()
        history_path = f"{ms_path}/HISTORY"
        
        # Open HISTORY table (read-only)
        tb.open(history_path, nomodify=True)
        
        # Get MESSAGE column
        # This might be large, but we only need to scan strings
        # Optimization: Read in chunks if needed, but HISTORY is usually small enough
        msgs = tb.getcol("MESSAGE")
        tb.close()
        
        if msgs is None or len(msgs) == 0:
            return metadata
            
        # Parse messages for provenance tags
        # Format: "DSA-110 PROVENANCE: KEY=VALUE"
        prefix = "DSA-110 PROVENANCE: "
        
        for msg in msgs:
            if not isinstance(msg, str):
                continue
                
            if msg.startswith(prefix):
                content = msg[len(prefix):]
                if "=" in content:
                    key, value = content.split("=", 1)
                    metadata[key.lower()] = value.strip()
                    
    except Exception:
        pass
        
    return metadata


__all__ = [
    "configure_ms_for_imaging",
    "inject_provenance_metadata",
    "get_provenance_metadata",
    "_ensure_imaging_columns_exist",
    "_ensure_imaging_columns_populated",
    "_ensure_flag_and_weight_spectrum",
    "_initialize_weights",
    "_fix_mount_type_in_ms",
    "_fix_field_phase_centers_from_times",
    "_ensure_observation_table_valid",
    "_ensure_state_table_valid",
    "_fix_observation_id_column",
    "_fix_observation_time_range",
]
