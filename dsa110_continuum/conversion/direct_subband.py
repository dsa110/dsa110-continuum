"""
Parallel MS writer for DSA-110 subband UVH5 files.

This strategy creates per-subband MS files in parallel, concatenates them,
and then merges all SPWs into a single SPW Measurement Set.
"""

import logging
import os
import shutil
import time
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any

import astropy.units as u
import numpy as np
from astropy.time import Time

from dsa110_contimg.core.conversion.helpers import (
    _ensure_antenna_diameters,
    cleanup_casa_file_handles,
    phase_to_meridian,
    set_antenna_positions,
    set_telescope_identity,
)
from dsa110_contimg.common.utils.paths import CONTIMG_TMPFS_DIR

from .writers import MSWriter

logger = logging.getLogger(__name__)


if TYPE_CHECKING:
    from pyuvdata import UVData


class DirectSubbandWriter(MSWriter):
    """Writes an MS by creating and concatenating per-subband parts, optionally merging SPWs.

    This writer creates per-subband MS files in parallel, concatenates them into
    a multi-SPW MS, and optionally merges all SPWs into a single SPW.

    Note: By default, SPW merging is disabled (merge_spws=False) to avoid
    mstransform incompatibility with CASA gaincal. Calibration should be performed
    on the multi-SPW MS before merging if needed.

    """

    def __init__(self, uv: "UVData", ms_path: str, **kwargs: Any) -> None:
        super().__init__(uv, ms_path, **kwargs)
        self.file_list: list[str] = self.kwargs.get("file_list", [])
        if not self.file_list:
            raise ValueError("DirectSubbandWriter requires 'file_list' in kwargs.")
        self.scratch_dir: str | None = self.kwargs.get("scratch_dir")
        self.max_workers: int = self.kwargs.get("max_workers", 4)
        # Optional tmpfs staging
        self.stage_to_tmpfs: bool = bool(self.kwargs.get("stage_to_tmpfs", False))
        self.tmpfs_path: str = str(
            self.kwargs.get(
                "tmpfs_path",
                os.environ.get(
                    "CONTIMG_TMPFS_DIR",
                    str(CONTIMG_TMPFS_DIR),
                ),
            )
        )
        # Defer copying from tmpfs to final location - keeps MS in tmpfs for faster
        # calibration/imaging, caller is responsible for final copy and cleanup
        self.defer_final_copy: bool = bool(self.kwargs.get("defer_final_copy", False))
        # Optional: disable SPW merging (for backward compatibility)
        # Default: False (don't merge) to avoid mstransform incompatibility with gaincal
        self.merge_spws: bool = bool(
            self.kwargs.get("merge_spws", False)  # Default: don't merge SPWs
        )
        # Optional: control SIGMA_SPECTRUM removal after merge
        self.remove_sigma_spectrum: bool = bool(
            # Default: remove to save space
            self.kwargs.get("remove_sigma_spectrum", True)
        )

    def get_files_to_process(self) -> list[str] | None:
        return self.file_list

    def _detect_synthetic_data(self) -> bool:
        """Detect if processing synthetic/simulated data.

        Checks the first file's extra_keywords for 'synthetic' flag or
        examines filename patterns for synthetic/simulation indicators.

        Returns
        -------
            True if synthetic data detected, False otherwise

        """
        if not self.file_list:
            return False

        # Check filename patterns first (fast)
        first_file = str(self.file_list[0])
        if "synthetic" in first_file.lower() or "simulation" in first_file.lower():
            logger.info(f"Detected synthetic data from filename: {first_file}")
            return True

        # Check HDF5 metadata (slower but definitive)
        try:
            from pyuvdata import UVData

            test_uv = UVData()
            test_uv.read(
                self.file_list[0],
                file_type="uvh5",
                read_data=False,
                run_check=False,
                check_extra=False,
                run_check_acceptability=False,
                strict_uvw_antpos_check=False,
            )
            is_synthetic = test_uv.extra_keywords.get("synthetic", False)
            if is_synthetic:
                logger.info(f"Detected synthetic=True in HDF5 metadata: {first_file}")
            del test_uv
            return bool(is_synthetic)
        except Exception as e:
            logger.warning(f"Could not check synthetic flag in {first_file}: {e}")
            return False

    def write(self) -> str:
        """Execute the parallel subband write and concatenation."""
        import multiprocessing
        from concurrent.futures import ProcessPoolExecutor, as_completed

        # Detect if processing synthetic data for appropriate multiprocessing context
        is_synthetic = self._detect_synthetic_data()

        # Use 'fork' for synthetic data (simpler pickling), 'spawn' for production
        # Production data uses 'spawn' to avoid fork issues with CASA/astropy in workers
        mp_method = "fork" if is_synthetic else "spawn"
        mp_ctx = multiprocessing.get_context(mp_method)
        logger.info(f"Using '{mp_method}' multiprocessing context (synthetic={is_synthetic})")

        from dsa110_contimg.core.calibration.casa_service import CASAService

        service = CASAService()

        # Determine staging locations
        ms_final_path = Path(self.ms_path)
        ms_stage_path = ms_final_path

        # Decide whether to use tmpfs for staging
        use_tmpfs = False
        tmpfs_root = Path(self.tmpfs_path)
        if not self.stage_to_tmpfs:
            logger.debug("Tmpfs staging disabled (stage_to_tmpfs=False); using scratch/output dir")
        elif not tmpfs_root.is_dir():
            logger.warning(
                f"Tmpfs staging path is not a directory: {self.tmpfs_path}. "
                "Falling back to scratch directory."
            )
        else:
            # PRECONDITION CHECK: Validate tmpfs is writable before staging
            # This ensures we follow "measure twice, cut once" - establish requirements upfront
            # before expensive staging operations.
            if not os.access(str(tmpfs_root), os.W_OK):
                logger.warning(
                    f"Tmpfs staging directory is not writable: {self.tmpfs_path}. "
                    "Falling back to scratch directory."
                )
            else:
                try:
                    # Rough size estimate: sum of input subband sizes × 2 margin
                    est_needed = 0
                    for p in self.file_list:
                        try:
                            est_needed += max(0, os.path.getsize(p))
                        except OSError:
                            pass
                    est_needed = int(est_needed * 2.0)
                    du = shutil.disk_usage(str(tmpfs_root))
                    free_bytes = int(du.free)
                    if free_bytes > est_needed:
                        use_tmpfs = True
                        logger.info(
                            "Using tmpfs staging at %s (free≈%.1f GiB, need≈%.1f GiB)",
                            self.tmpfs_path,
                            free_bytes / (1024**3),
                            est_needed / (1024**3),
                        )
                    else:
                        logger.warning(
                            "Insufficient tmpfs space at %s (free≈%.1f GiB, need≈%.1f GiB). "
                            "Falling back to scratch directory.",
                            self.tmpfs_path,
                            free_bytes / (1024**3),
                            est_needed / (1024**3),
                        )
                except OSError as e:
                    logger.warning(
                        "Could not inspect tmpfs space at %s (%s). Falling back to scratch directory.",
                        self.tmpfs_path,
                        e,
                    )

        if use_tmpfs:
            # Stage parts and final concat under tmpfs
            # Solution 2: Use unique identifier to avoid conflicts between groups
            unique_id = f"{ms_final_path.stem}_{uuid.uuid4().hex[:8]}"
            staging_root = (
                tmpfs_root
                if tmpfs_root.name == "dsa110-contimg"
                else (tmpfs_root / "dsa110-contimg")
            )
            part_base = staging_root / unique_id
            part_base.mkdir(parents=True, exist_ok=True)
            ms_stage_path = part_base.parent / (ms_final_path.stem + ".staged.ms")
        else:
            # Use provided scratch or output directory parent
            if self.scratch_dir:
                base_dir = Path(self.scratch_dir)
            elif ms_final_path.is_absolute():
                base_dir = ms_final_path.parent
            else:
                # Fallback to configured global staging directory instead of CWD
                from dsa110_contimg.common.unified_config import get_config

                base_dir = get_config().paths.staging_dir

            part_base = base_dir / ms_final_path.stem
        part_base.mkdir(parents=True, exist_ok=True)

        # Compute shared pointing declination for entire group
        # Time-dependent phase centers will be set per-subband via phase_to_meridian()
        group_pt_dec = None

        try:
            # Calculate group midpoint time by averaging all subband midpoints
            mid_times = []
            for sb_file in self.file_list:
                try:
                    # Use lightweight peek to get midpoint time without full read
                    from dsa110_contimg.common.utils.fast_meta import peek_uvh5_phase_and_midtime

                    # Use FUSE-aware read lock to avoid racing with moves
                    try:
                        from dsa110_contimg.common.utils.fuse_lock import get_fuse_lock_manager

                        lm = get_fuse_lock_manager()
                    except Exception:
                        lm = None

                    if lm is not None:
                        with lm.read_lock(sb_file, timeout=5.0):
                            _, pt_dec, mid_mjd = peek_uvh5_phase_and_midtime(sb_file)
                    else:
                        _, pt_dec, mid_mjd = peek_uvh5_phase_and_midtime(sb_file)

                    if group_pt_dec is None:
                        group_pt_dec = pt_dec
                    if np.isfinite(mid_mjd) and mid_mjd > 0:
                        mid_times.append(mid_mjd)
                except (OSError, KeyError, ValueError, RuntimeError):
                    # Fallback: read first file fully if peek fails
                    # OSError: file issues, KeyError: missing metadata,
                    # ValueError: invalid data, RuntimeError: HDF5 errors
                    if group_pt_dec is None:
                        try:
                            from pyuvdata import UVData

                            try:
                                from dsa110_contimg.common.utils.fuse_lock import get_fuse_lock_manager

                                lm2 = get_fuse_lock_manager()
                            except Exception:
                                lm2 = None

                            if lm2 is not None:
                                with lm2.read_lock(sb_file, timeout=10.0):
                                    temp_uv = UVData()
                                    temp_uv.read(
                                        sb_file,
                                        file_type="uvh5",
                                        read_data=False,
                                        run_check=False,
                                        check_extra=False,
                                        run_check_acceptability=False,
                                        strict_uvw_antpos_check=False,
                                    )
                            else:
                                temp_uv = UVData()
                                temp_uv.read(
                                    sb_file,
                                    file_type="uvh5",
                                    read_data=False,
                                    run_check=False,
                                    check_extra=False,
                                    run_check_acceptability=False,
                                    strict_uvw_antpos_check=False,
                                )

                            if group_pt_dec is None:
                                group_pt_dec = (
                                    temp_uv.extra_keywords.get("phase_center_dec", 0.0) * u.rad
                                )
                            mid_mjd = Time(float(np.mean(temp_uv.time_array)), format="jd").mjd
                            mid_times.append(mid_mjd)
                            del temp_uv
                        except (OSError, ValueError, RuntimeError):
                            # OSError: file issues, ValueError: invalid data,
                            # RuntimeError: HDF5/pyuvdata errors
                            pass

            if group_pt_dec is not None and len(mid_times) > 0:
                # Compute group midpoint time (average of all subband midpoints)
                group_mid_mjd = float(np.mean(mid_times))
                # group_pt_dec is a float in radians, convert to degrees
                if isinstance(group_pt_dec, u.Quantity):
                    pt_dec_deg = group_pt_dec.to(u.deg).value
                else:
                    pt_dec_deg = np.degrees(group_pt_dec)
                logger.info(
                    f"Using shared pointing declination for group: "
                    f"Dec={pt_dec_deg:.6f}° "
                    f"(MJD={group_mid_mjd:.6f})"
                )
        except (OSError, ValueError, RuntimeError) as e:
            logger.warning(f"Failed to compute shared pointing declination: {e}")
            logger.info("Falling back to per-subband pointing declination")
            group_pt_dec = None

        # Use processes, not threads: casatools/casacore are not thread-safe
        # for concurrent Simulator usage.
        # CRITICAL: DSA-110 subbands use DESCENDING frequency order:
        #   sb00 = highest frequency (~1498 MHz)
        #   sb15 = lowest frequency (~1311 MHz)
        # For MFS imaging, we need ASCENDING frequency order (low to high).
        # Therefore, we must REVERSE the subband number sort.
        from dsa110_contimg.core.conversion.conversion_orchestrator import (
            _extract_subband_code,
        )

        def sort_by_subband(fpath):
            fname = os.path.basename(fpath)
            sb = _extract_subband_code(fname)
            sb_num = int(sb.replace("sb", "")) if sb else 999
            return sb_num

        # CRITICAL: Sort in REVERSE subband order (15, 14, ..., 1, 0) to get
        # ascending frequency order (lowest to highest) for proper MFS imaging
        # and bandpass calibration. If frequencies are out of order, imaging will
        # produce fringes and bandpass calibration will fail.
        sorted_files = sorted(self.file_list, key=sort_by_subband, reverse=True)

        # Convert astropy Quantity to plain float (radians) for pickling
        # Astropy units don't reconstruct properly in worker processes
        group_pt_dec_rad = None
        if group_pt_dec is not None:
            if isinstance(group_pt_dec, u.Quantity):
                group_pt_dec_rad = float(group_pt_dec.to(u.rad).value)
            else:
                group_pt_dec_rad = float(group_pt_dec)

        futures = []
        with ProcessPoolExecutor(max_workers=self.max_workers, mp_context=mp_ctx) as ex:
            for idx, sb_file in enumerate(sorted_files):
                part_out = part_base / f"{Path(ms_stage_path).stem}.sb{idx:02d}.ms"
                futures.append(
                    (
                        idx,
                        ex.submit(
                            _write_ms_subband_part, sb_file, str(part_out), group_pt_dec_rad
                        ),  # Pass shared pointing declination as plain float
                    )
                )

        # Collect results in order (idx 0, 1, 2, ..., 15) to maintain spectral order
        parts = [None] * len(futures)
        completed = 0
        failed_subbands = []
        for future in as_completed([f for _, f in futures]):
            try:
                result = future.result()
                # Find which idx this future corresponds to
                for orig_idx, orig_future in futures:
                    if orig_future == future:
                        parts[orig_idx] = result
                        completed += 1
                        break
                if completed % 4 == 0 or completed == len(futures):
                    msg = f"Per-subband writes completed: {completed}/{len(futures)}"
                    logger.info(msg)
            except Exception as e:
                # Track which subband failed for better error reporting
                for orig_idx, orig_future in futures:
                    if orig_future == future:
                        failed_subbands.append((orig_idx, sorted_files[orig_idx], str(e)))
                        logger.error(f"Subband {orig_idx} ({sorted_files[orig_idx]}) failed: {e}")
                        break

        # Report all failures if any occurred
        if failed_subbands:
            failure_details = "; ".join(
                f"sb{idx}({Path(f).name}): {err[:100]}" for idx, f, err in failed_subbands
            )
            raise RuntimeError(
                f"{len(failed_subbands)}/{len(futures)} subband writer processes failed. "
                f"Check logs for details. Failures: {failure_details}"
            )

        # Remove None entries (shouldn't happen, but safety check)
        parts = [p for p in parts if p is not None]

        # Solution 4: Ensure subband write processes fully terminate before concat
        # Allow processes to fully terminate and release file handles
        time.sleep(0.5)

        # CRITICAL: Clean up any lingering CASA file handles before concat
        # This prevents file locking issues during concatenation
        cleanup_casa_file_handles()

        # CRITICAL: Remove existing staged MS if it exists (from previous failed run)
        # CASA's concat doesn't handle existing output directories well
        if ms_stage_path.exists():
            logger.warning(f"Removing existing staged MS before concatenation: {ms_stage_path}")
            cleanup_casa_file_handles()
            shutil.rmtree(ms_stage_path, ignore_errors=True)
            # Ensure the directory is fully removed
            time.sleep(0.5)
            cleanup_casa_file_handles()

        # Solution 3: Retry logic for concat failures
        # Concatenate parts into the final MS with retry on file locking errors
        logger.info(f"Concatenating {len(parts)} parts into {ms_stage_path}")
        max_retries = 3  # Increased from 2 to 3 for better reliability
        concat_success = False
        last_error = None

        for attempt in range(max_retries):
            try:
                # Additional cleanup before each concat attempt
                if attempt > 0:
                    # Clean up any partial MS from previous failed attempt
                    if ms_stage_path.exists():
                        logger.warning(
                            f"Removing partial staged MS from failed attempt: {ms_stage_path}"
                        )
                        cleanup_casa_file_handles()
                        shutil.rmtree(ms_stage_path, ignore_errors=True)
                        time.sleep(1.0)
                    cleanup_casa_file_handles()
                    time.sleep(1.0)  # Give more time for handles to close

                # CRITICAL: Parts are already in correct subband order (0-15)
                # Do NOT sort here - parts are already ordered by subband number
                # from the futures collection above. Sorting would break spectral order.

                # Set up temp environment and CWD for CASA concat
                # CASA creates temporary files (TMPPOINTING*) in the current working directory
                # We need to ensure CWD is writable and temp env vars are set
                from dsa110_contimg.common.utils.run_isolation import prepare_temp_environment

                # Use the MS directory as the working directory (should be writable)
                if ms_stage_path.is_absolute():
                    ms_dir = ms_stage_path.parent
                else:
                    # Fallback to configured global staging directory instead of CWD
                    from dsa110_contimg.common.unified_config import get_config

                    ms_dir = get_config().paths.staging_dir

                scratch_root = Path(self.scratch_dir) if self.scratch_dir else ms_dir

                # Set up temp environment variables (TMPDIR, TMP, TEMP, CASA_TMPDIR)
                # and change CWD to the MS directory (writable location)
                # CASA will create temporary files in CWD, so it must be writable
                prepare_temp_environment(
                    preferred_root=str(scratch_root),
                    cwd_to=str(ms_dir),  # Change CWD to MS directory (writable)
                    setup_casa_logs=True,  # Set up CASA logging environment
                )

                # Now call service.concat with CWD set to writable MS directory
                service.concat(
                    vis=parts,  # Already in correct subband order
                    concatvis=str(ms_stage_path),
                    copypointing=False,
                    freqtol="",  # Merge all SPWs into single wideband SPW
                    dirtol="",  # Merge fields with same phase center
                )
                concat_success = True

                # Fix SPECTRAL_WINDOW table to use positive channel widths (USB convention)
                # This eliminates CASA warnings about "Negative or zero total bandwidth"
                try:
                    from casatools import table as casatable

                    tb = casatable()
                    spw_table = f"{ms_stage_path}/SPECTRAL_WINDOW"
                    tb.open(spw_table, nomodify=False)
                    nspw = tb.nrows()
                    for i in range(nspw):
                        # Fix CHAN_WIDTH to be positive
                        chan_width = tb.getcell("CHAN_WIDTH", i)
                        if np.any(chan_width < 0):
                            tb.putcell("CHAN_WIDTH", i, np.abs(chan_width))

                        # Recalculate TOTAL_BANDWIDTH as positive value
                        chan_freq = tb.getcell("CHAN_FREQ", i)
                        if len(chan_freq) > 1:
                            total_bw = abs(chan_freq[-1] - chan_freq[0]) + abs(
                                chan_freq[1] - chan_freq[0]
                            )
                            tb.putcell("TOTAL_BANDWIDTH", i, total_bw)
                    tb.close()
                    logger.info(
                        f"Fixed CHAN_WIDTH and TOTAL_BANDWIDTH in {nspw} SPWs (positive/USB convention)"
                    )
                except Exception as bw_err:
                    logger.warning(f"Could not fix SPECTRAL_WINDOW table: {bw_err}")

                # Fix STATE and OBSERVATION tables for CASA compatibility
                # pyuvdata.write_ms() creates empty STATE tables and [0,0] TIME_RANGE,
                # which causes CASA msmetadata.timesforscans() to fail with
                # "No matching scans found" - breaking calibration and imaging.
                try:
                    from dsa110_contimg.core.conversion.ms_utils import (
                        _ensure_state_table_valid,
                        _fix_observation_time_range,
                    )

                    _ensure_state_table_valid(str(ms_stage_path))
                    _fix_observation_time_range(str(ms_stage_path))
                    logger.info(f"Applied STATE and OBSERVATION table fixes to {ms_stage_path}")
                except Exception as table_fix_err:
                    logger.warning(
                        f"Could not apply STATE/OBSERVATION table fixes (non-fatal): {table_fix_err}"
                    )

                break
            except (RuntimeError, OSError) as e:
                last_error = e
                error_msg = str(e)
                # Check for retryable errors
                retryable = (
                    "cannot be opened" in error_msg
                    or "readBlock" in error_msg
                    or "read/write" in error_msg
                    or "lock" in error_msg.lower()
                    or "Directory not empty" in error_msg
                    or "Invalid cross-device link" in error_msg
                    or "Errno 39" in error_msg
                    or "Errno 18" in error_msg
                )
                if retryable and attempt < max_retries - 1:
                    logger.warning(
                        f"Concat failed (attempt {attempt + 1}/{max_retries}), "
                        f"retrying after cleanup: {e}"
                    )
                    # Enhanced cleanup and retry
                    if ms_stage_path.exists():
                        shutil.rmtree(ms_stage_path, ignore_errors=True)
                    from dsa110_contimg.core.conversion.helpers_telescope import (
                        casa_operation,
                    )

                    with casa_operation():
                        # Cleanup happens automatically
                        pass
                    time.sleep(2.0)  # Longer wait for file handles to release
                    continue
                raise

        if not concat_success:
            from dsa110_contimg.core.conversion.helpers_telescope import casa_operation

            with casa_operation():
                # Final cleanup attempt - automatic cleanup
                pass
            raise RuntimeError(
                f"Concat failed after {max_retries} attempts. Last error: {last_error}"
            )

        # Explicit cleanup verification after concat
        # Use context manager for guaranteed cleanup
        from dsa110_contimg.core.conversion.helpers_telescope import casa_operation

        with casa_operation():
            # Cleanup happens automatically
            pass

        # If staged on tmpfs, move final MS atomically (or via copy on
        # cross-device) - unless defer_final_copy is set
        if use_tmpfs and not self.defer_final_copy:
            try:
                # Ensure destination parent exists
                ms_final_path.parent.mkdir(parents=True, exist_ok=True)
                src_path = str(ms_stage_path)
                dst_path = str(ms_final_path)

                # Use fuse_safe_move for robust FUSE filesystem handling
                try:
                    from dsa110_contimg.common.utils.fuse_lock import fuse_safe_move

                    fuse_safe_move(src_path, dst_path, timeout=60.0)
                    ms_stage_path = ms_final_path
                except ImportError:
                    # Fallback to plain move if fuse_lock not available
                    shutil.move(src_path, dst_path)
                    ms_stage_path = ms_final_path
                except Exception as move_err:
                    # Log error and fall back to plain move
                    logger.warning(
                        f"fuse_safe_move failed: {move_err}, falling back to shutil.move"
                    )
                    shutil.move(src_path, dst_path)
                    ms_stage_path = ms_final_path

                logger.info(f"Moved staged MS to final location: {ms_final_path}")
            except OSError:
                # If move failed, try copytree (for directory MS)
                if ms_final_path.exists():
                    shutil.rmtree(ms_final_path, ignore_errors=True)
                src_path = str(ms_stage_path)
                dst_path = str(ms_final_path)
                shutil.copytree(src_path, dst_path)
                shutil.rmtree(ms_stage_path, ignore_errors=True)
                ms_stage_path = ms_final_path
                logger.info(f"Copied staged MS to final location: {ms_final_path}")

        # Merge SPWs into a single SPW if requested
        if self.merge_spws:
            try:
                from dsa110_contimg.core.conversion.merge_spws import (
                    get_spw_count,
                    merge_spws,
                )

                n_spw_before = get_spw_count(str(ms_stage_path))
                if n_spw_before and n_spw_before > 1:
                    logger.info(f"Merging {n_spw_before} SPWs into a single SPW...")
                    ms_multi_spw = str(ms_stage_path)
                    ms_single_spw = str(ms_stage_path) + ".merged"

                    merge_spws(
                        ms_in=ms_multi_spw,
                        ms_out=ms_single_spw,
                        datacolumn="DATA",
                        regridms=True,
                        keepflags=True,
                        remove_sigma_spectrum=self.remove_sigma_spectrum,
                    )

                    # Replace multi-SPW MS with single-SPW MS
                    shutil.rmtree(ms_multi_spw, ignore_errors=True)

                    # Use fuse_safe_move for robust FUSE filesystem handling
                    try:
                        from dsa110_contimg.common.utils.fuse_lock import fuse_safe_move

                        fuse_safe_move(ms_single_spw, ms_multi_spw, timeout=60.0)
                    except ImportError:
                        shutil.move(ms_single_spw, ms_multi_spw)
                    except Exception as move_err:
                        logger.warning(
                            f"fuse_safe_move failed: {move_err}, falling back to shutil.move"
                        )
                        shutil.move(ms_single_spw, ms_multi_spw)

                    n_spw_after = get_spw_count(str(ms_stage_path))
                    if n_spw_after == 1:
                        logger.info(f"Successfully merged SPWs: {n_spw_before} :arrow_right: 1")
                    else:
                        logger.warning(f"Expected 1 SPW after merge, got {n_spw_after}")
            except Exception as merge_err:
                logger.warning(f"SPW merging failed (non-fatal): {merge_err}", exc_info=True)

        # Solution 1: Clean up temporary per-subband Measurement Sets and staging dir
        # with verification that cleanup completed
        cleanup_attempts = 0
        max_cleanup_attempts = 3
        while cleanup_attempts < max_cleanup_attempts:
            try:
                for part in parts:
                    if Path(part).exists():
                        shutil.rmtree(part, ignore_errors=True)
                if part_base.exists():
                    shutil.rmtree(part_base, ignore_errors=True)

                # Verify cleanup completed
                if part_base.exists():
                    cleanup_attempts += 1
                    if cleanup_attempts < max_cleanup_attempts:
                        logger.warning(
                            f"Cleanup incomplete (attempt {cleanup_attempts}), "
                            f"retrying: {part_base}"
                        )
                        time.sleep(0.5)
                        continue
                    else:
                        logger.warning(
                            f"Cleanup incomplete after {max_cleanup_attempts} attempts: {part_base}"
                        )
                break
            except Exception as cleanup_err:
                cleanup_attempts += 1
                if cleanup_attempts < max_cleanup_attempts:
                    logger.warning(
                        f"Cleanup failed (attempt {cleanup_attempts}), retrying: {cleanup_err}"
                    )
                    time.sleep(0.5)
                else:
                    logger.warning(
                        f"Failed to clean subband parts after {max_cleanup_attempts} attempts: "
                        f"{cleanup_err}"
                    )

        return "direct-subband"


def _write_ms_subband_part(
    subband_file: str,
    part_out: str,
    shared_pt_dec: float | None = None,
) -> str:
    """Write a single-subband MS using pyuvdata.write_ms.

    This is a top-level function to be safely used with multiprocessing.
    Uses time-dependent phase centers that track LST throughout the observation.

    Parameters
    ----------
    subband_file :
        Path to input UVH5 subband file
    part_out :
        Path to output MS file
    shared_pt_dec :
        Optional shared pointing declination in radians (for UVW computation)

    Returns
    -------
        Path to created MS file

    """
    import traceback

    from pyuvdata import UVData

    # Acquire a process-level shared lock on the input file to prevent it being
    # moved/deleted while this process is reading it (avoids .fuse_hidden* on FUSE).
    lock_fd = None
    try:
        try:
            from dsa110_contimg.common.utils.fuse_lock import acquire_process_lock
        except Exception:
            acquire_process_lock = None

        if acquire_process_lock is not None:
            try:
                lock_fd = acquire_process_lock(subband_file, exclusive=False, timeout=10.0)
            except Exception:
                lock_fd = None

        uv = UVData()
        try:
            uv.read(
                subband_file,
                file_type="uvh5",
                run_check=False,
                run_check_acceptability=False,
                strict_uvw_antpos_check=False,
                check_extra=False,
            )
        except Exception as e:
            logger.error(f"Failed to read subband file {subband_file}: {e}")
            logger.error(f"Traceback:\n{traceback.format_exc()}")
            raise RuntimeError(f"Cannot read HDF5 file {subband_file}: {e}") from e

    finally:
        if lock_fd is not None:
            try:
                from dsa110_contimg.common.utils.fuse_lock import release_process_lock

                release_process_lock(lock_fd)
            except Exception:
                pass

    # Normalize baseline conjugation to CASA convention (ant1 < ant2)
    # DSA-110 correlator outputs some baselines with antennas 101 and 115 in
    # reversed order (ant1 > ant2). This causes pyuvdata to warn about "mix of
    # baseline conjugation states" when writing MS. Fix it here at read time.
    if np.any(uv.ant_1_array > uv.ant_2_array):
        logger.debug(
            f"Normalizing baseline conjugation for {subband_file} "
            f"({np.sum(uv.ant_1_array > uv.ant_2_array)} reversed baselines)"
        )
        uv.conjugate_bls("ant1<ant2")

    # Validate and fix telescope metadata for synthetic data
    try:
        if not hasattr(uv, "telescope") or uv.telescope is None:
            logger.warning(f"Missing telescope object in {subband_file}, reconstructing...")
            from pyuvdata import Telescope

            from dsa110_contimg.common.utils.constants import DSA110_LOCATION

            tel = Telescope()
            tel.name = "DSA-110"
            tel.instrument = "DSA-110"
            tel.location = DSA110_LOCATION

            # Copy telescope-level metadata from UVData if available
            if hasattr(uv, "Nants_telescope"):
                tel.Nants = uv.Nants_telescope
            if hasattr(uv, "antenna_names"):
                tel.antenna_names = uv.antenna_names
            if hasattr(uv, "antenna_numbers"):
                tel.antenna_numbers = uv.antenna_numbers
            if hasattr(uv, "antenna_positions"):
                tel.antenna_positions = uv.antenna_positions

            uv.telescope = tel
            logger.info(f"Reconstructed telescope object for {subband_file}")
    except Exception as e:
        logger.warning(f"Could not validate/fix telescope metadata: {e}")

    # Stamp telescope identity prior to phasing/UVW
    # Uses DSA110_LOCATION from constants.py (single source of truth)
    try:
        set_telescope_identity(
            uv,
            os.getenv("PIPELINE_TELESCOPE_NAME", "DSA_110"),
        )
    except (AttributeError, TypeError, ValueError):
        # AttributeError/TypeError: UVData attr issues, ValueError: coord conversion
        pass

    part_out_path = Path(part_out)
    if part_out_path.exists():
        shutil.rmtree(part_out_path, ignore_errors=True)
    part_out_path.parent.mkdir(parents=True, exist_ok=True)

    # Reorder freqs ascending to keep CASA concat happy
    # pyuvdata's reorder_freqs with 'freq' sorts by frequency value (ascending)
    # This handles DSA-110 descending subbands correctly
    uv.reorder_freqs(channel_order="freq", run_check=False)

    # Ensure positive channel widths (standard USB convention)
    # DSA-110 correlator outputs negative widths (LSB convention) which causes
    # CASA warnings about "Negative or zero total bandwidth". Converting to
    # positive widths after frequency reordering maintains correct frequency
    # labels while matching industry-standard conventions.
    if hasattr(uv, "channel_width") and uv.channel_width is not None:
        if np.any(uv.channel_width < 0):
            uv.channel_width = np.abs(uv.channel_width)
            logger.debug("Converted channel_width to positive (USB convention)")

    # Set antenna metadata
    set_antenna_positions(uv)
    _ensure_antenna_diameters(uv)

    # Determine pointing declination
    if shared_pt_dec is not None:
        pt_dec = shared_pt_dec
    else:
        pt_dec = uv.extra_keywords.get("phase_center_dec", 0.0) * u.rad

    # Use phase_to_meridian() to set time-dependent phase centers
    # This ensures phase center RA tracks LST throughout the observation,
    # following interferometry best practices for continuous phase tracking
    phase_to_meridian(uv, pt_dec)

    # Write the single-subband MS
    # Suppress the "uncalib" units warning - DSA-110 correlator outputs raw
    # correlation coefficients (not flux-calibrated Jy). This is expected and
    # correct; calibration is applied later by the pipeline. The DATA column
    # will contain uncalibrated data, while CORRECTED_DATA (created during
    # calibration) will be in Jy. See README.md#visibility-units-warning.
    import warnings

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Writing in the MS file that the units of the data are uncalib",
            category=UserWarning,
        )
        uv.write_ms(
            str(part_out_path),
            clobber=True,
            run_check=False,
            check_extra=False,
            run_check_acceptability=False,
            strict_uvw_antpos_check=False,
            check_autos=False,
            fix_autos=False,
        )
    return str(part_out_path)
