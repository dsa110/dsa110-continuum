"""
HDF5 Subband Group Orchestrator for DSA-110 Continuum Imaging Pipeline.

Orchestrates the conversion of HDF5 subband files to Measurement Sets,
handling subband grouping, combination, and MS writing with proper
error handling and logging.

Configuration is loaded from settings.conversion for rarely-changed parameters.
Only essential arguments (input, output, time range) are passed to functions.
"""

from __future__ import annotations

import logging
import os
import re
import time
from pathlib import Path

import numpy as np
import pyuvdata

from dsa110_contimg.common.unified_config import settings
from dsa110_contimg.common.utils import FastMeta, timed, timed_debug
from dsa110_contimg.common.utils.antpos_local import get_itrf
from dsa110_contimg.common.utils.exceptions import (
    ConversionError,
    IncompleteSubbandGroupError,
    MSWriteError,
    UVH5ReadError,
    is_recoverable,
    wrap_exception,
)
from dsa110_contimg.common.utils.logging import log_context, log_exception
from dsa110_contimg.core.conversion.file_validator import (
    MissingInputFilesError,
    RollingFileValidator,
)
from dsa110_contimg.core.conversion.writers import get_writer
from dsa110_contimg.infrastructure.database.hdf5_index import (
    parse_subband_filename,
    query_subband_groups,
)
from dsa110_contimg.infrastructure.monitoring.pipeline_metrics import (
    PipelineStage,
    record_stage_timing,
)

logger = logging.getLogger(__name__)

# Regex to match subband codes in filenames (e.g., "_sb00", "_sb15")
_SUBBAND_PATTERN = re.compile(r"_sb(\d{2})")


def _extract_subband_code(filename: str) -> str | None:
    """Extract the subband code (e.g., 'sb00') from a filename.

    DSA-110 subband files follow the pattern: {timestamp}_sb{NN}.hdf5
    where NN is a two-digit subband number (00-15).

    Parameters
    ----------
    filename :
        Filename or path to extract subband code from.

    Returns
    -------
        Subband code like 'sb00', 'sb15', or None if not found.

    """
    basename = os.path.basename(filename)
    match = _SUBBAND_PATTERN.search(basename)
    if match:
        return f"sb{match.group(1)}"
    return None


def _remap_paths(group: list[str], remap_input_dir: Path) -> list[str]:
    """Remap file paths in a subband group to a different directory.

    This is used for golden datasets where the HDF5 index contains paths
    to /data/incoming/ but the actual files are in the golden dataset's
    raw/ directory (possibly downsampled).

    Parameters
    ----------
    group :
        List of file paths from the database.
    remap_input_dir :
        Directory to look for files in instead.
    group: list[str] :

    Returns
    -------
        List of remapped file paths with the same filenames but in remap_input_dir.

    """
    remapped = []
    for path in group:
        filename = os.path.basename(path)
        new_path = remap_input_dir / filename
        if new_path.exists():
            remapped.append(str(new_path))
        else:
            logger.warning(f"Remapped file not found: {new_path} (original: {path})")
            # Fall back to original path if remapped file doesn't exist
            remapped.append(path)
    return remapped


@timed("conversion.convert_subband_groups")
def convert_subband_groups_to_ms(
    input_dir: str,
    output_dir: str,
    start_time: str,
    end_time: str,
    *,
    # Override settings if needed (most callers just use defaults)
    tolerance_s: float | None = None,
    skip_incomplete: bool | None = None,
    skip_existing: bool | None = None,
    stage_to_tmpfs: bool | None = None,
    defer_final_copy: bool | None = None,
    remap_input_dir: str | None = None,
    ms_suffix: str | None = None,
) -> dict:
    """Orchestrate the conversion of HDF5 subband files to Measurement Sets.

    Most parameters are pulled from settings.conversion. Only override
    explicitly if you need non-default behavior.

    Parameters
    ----------
    input_dir : str
        Directory containing the HDF5 subband files.
    output_dir : str
        Directory where the Measurement Sets will be saved.
    start_time : str
        Start time for the conversion window.
    end_time : str
        End time for the conversion window.
    tolerance_s : float, optional
        Time tolerance for grouping subbands.
    skip_incomplete : bool, optional
        Skip incomplete groups.
    skip_existing : bool, optional
        Skip existing MS files.
    stage_to_tmpfs : bool, optional
        Stage files to tmpfs for faster I/O.
    defer_final_copy : bool, optional
        If True and stage_to_tmpfs is set, defer the final copy. This is
        faster for pipelines that will work on the data and copy later.
    remap_input_dir : str, optional
        Alternate input directory for looking up HDF5 files. Used for
        golden datasets where HDF5 files are stored in a different location.
    ms_suffix : str, optional
        Suffix to append to MS filename (inserted before .ms extension).

    Returns
    -------
    dict
        Dictionary with conversion statistics:
        - converted: List of successfully converted group IDs
        - skipped: List of skipped group IDs (incomplete or existing)
        - failed: List of failed group IDs with error details

    Raises
    ------
    ConversionError
        If no groups are found or critical error occurs.
    """
    # Apply settings defaults for optional parameters
    if tolerance_s is None:
        tolerance_s = settings.conversion.cluster_tolerance_s
    if skip_incomplete is None:
        skip_incomplete = settings.conversion.skip_incomplete
    if skip_existing is None:
        skip_existing = settings.conversion.skip_existing
    if stage_to_tmpfs is None:
        stage_to_tmpfs = settings.conversion.stage_to_tmpfs
    if defer_final_copy is None:
        # Default: defer copy when using tmpfs (faster for pipeline processing)
        defer_final_copy = False

    results = {
        "converted": [],
        "skipped": [],
        "failed": [],
    }

    # Validate paths
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    if not input_path.exists():
        raise ConversionError(
            f"Input directory does not exist: {input_dir}",
            input_path=input_dir,
        )

    # Create output directory if needed
    output_path.mkdir(parents=True, exist_ok=True)

    # Query subband groups based on the provided time window
    # Use unified pipeline database
    from dsa110_contimg.infrastructure.database.unified import get_pipeline_db_path

    hdf5_db = str(get_pipeline_db_path())

    try:
        result = query_subband_groups(
            hdf5_db,
            start_time,
            end_time,
        )
    except Exception as e:
        raise ConversionError(
            f"Failed to query subband groups from database: {e}",
            input_path=input_dir,
            original_exception=e,
        ) from e

    if not result:
        logger.warning(
            "No subband groups found in time window",
            extra={
                "input_dir": input_dir,
                "start_time": start_time,
                "end_time": end_time,
                "tolerance_s": tolerance_s,
            },
        )
        return results

    logger.info(
        f"Found {len(result)} subband groups to process "
        f"({result.metrics.complete_groups} complete, "
        f"{result.metrics.fraction_complete:.1%})",
        extra={
            "input_dir": input_dir,
            "start_time": start_time,
            "end_time": end_time,
            "group_count": len(result),
            "complete_groups": result.metrics.complete_groups,
            "fraction_complete": result.metrics.fraction_complete,
        },
    )

    # Validate file existence before processing
    # Use rolling window validator with 100-group window for production (60k files)
    validator = RollingFileValidator(window_size=100, max_workers=16)
    validation_results = validator.validate_groups(
        result,
        extract_id=lambda g: g.representative_time,
    )

    # Filter to valid groups and log validation failures
    valid_groups = []
    for group, val_result in zip(result, validation_results):
        if val_result.valid:
            valid_groups.append(group)
        else:
            logger.warning(
                f"Skipping group {val_result.group_id}: {val_result.error_message}",
                extra={
                    "group_id": val_result.group_id,
                    "missing_files": val_result.missing_files,
                    "file_count": val_result.file_count,
                },
            )
            results["skipped"].append(val_result.group_id)

    if len(valid_groups) < len(result):
        logger.warning(
            f"Validation filtered {len(result) - len(valid_groups)} groups with missing files",
            extra=validator.get_stats(),
        )

    for group in valid_groups:
        group_id = _extract_group_id(group.files)
        # Apply suffix to group_id for MS filename (e.g., "2025-10-02T15:40:06_12x")
        ms_group_id = f"{group_id}{ms_suffix}" if ms_suffix else group_id

        with log_context(group_id=group_id, pipeline_stage="conversion"):
            try:
                # Use progress monitoring for conversion (can take minutes per group)
                from dsa110_contimg.common.utils.progress import stage_progress

                output_path = os.path.join(output_dir, f"{ms_group_id}.ms")
                with stage_progress(
                    f"HDF5→MS conversion ({ms_group_id})",
                    output_path=output_path,
                    poll_interval=5.0,
                    subbands=len(group.files),
                ):
                    result_status = _convert_single_group(
                        group=group.files,
                        group_id=ms_group_id,
                        output_dir=output_dir,
                        skip_incomplete=skip_incomplete,
                        skip_existing=skip_existing,
                        stage_to_tmpfs=stage_to_tmpfs,
                        defer_final_copy=defer_final_copy,
                    )

                if result_status == "converted":
                    results["converted"].append(ms_group_id)
                elif result_status == "skipped":
                    results["skipped"].append(ms_group_id)

            except IncompleteSubbandGroupError as e:
                # Log warning but continue with next group
                logger.warning(
                    str(e),
                    extra=e.context,
                )
                results["skipped"].append(ms_group_id)

            except (UVH5ReadError, MSWriteError, ConversionError) as e:
                # Log error with full context
                log_exception(logger, e, group_id=ms_group_id)
                results["failed"].append(
                    {
                        "group_id": ms_group_id,
                        "error": str(e),
                        "error_type": type(e).__name__,
                        "recoverable": e.recoverable,
                    }
                )

                # Re-raise if not recoverable
                if not e.recoverable:
                    raise

            except Exception as e:
                # Unexpected error - wrap and log
                wrapped = wrap_exception(
                    e,
                    ConversionError,
                    f"Unexpected error during conversion: {e}",
                    group_id=ms_group_id,
                )
                log_exception(logger, wrapped, group_id=ms_group_id)
                results["failed"].append(
                    {
                        "group_id": ms_group_id,
                        "error": str(e),
                        "error_type": type(e).__name__,
                        "recoverable": is_recoverable(e),
                    }
                )

    # Log summary
    logger.info(
        f"Conversion complete: {len(results['converted'])} converted, "
        f"{len(results['skipped'])} skipped, {len(results['failed'])} failed",
        extra={
            "converted_count": len(results["converted"]),
            "skipped_count": len(results["skipped"]),
            "failed_count": len(results["failed"]),
        },
    )

    return results


@timed_debug("conversion.convert_single_group")
def _convert_single_group(
    group: list[str],
    group_id: str,
    output_dir: str,
    skip_incomplete: bool,
    skip_existing: bool,
    stage_to_tmpfs: bool,
    defer_final_copy: bool = False,
) -> str:
    """Convert a single subband group to Measurement Set.

    Parameters
    ----------
    group: list[str] :

    Returns
    -------
        "converted" if successful, "skipped" if skipped

    Raises
    ------
    IncompleteSubbandGroupError
        If group is incomplete and skip_incomplete=True
    UVH5ReadError
        If reading UVH5 fails
    MSWriteError
        If writing MS fails

    """
    # Check for complete group (use settings for expected count)
    expected_subbands = settings.conversion.expected_subbands
    if len(group) < expected_subbands:
        if skip_incomplete:
            raise IncompleteSubbandGroupError(
                group_id=group_id,
                expected_count=expected_subbands,
                actual_count=len(group),
                missing_subbands=_find_missing_subbands(group),
            )
        else:
            missing_sbs = _find_missing_subbands(group)
            logger.warning(
                f"Processing incomplete group: {len(group)}/{expected_subbands} subbands "
                f"(missing: {', '.join(missing_sbs)})",
                extra={
                    "group_id": group_id,
                    "subband_count": len(group),
                    "expected_count": expected_subbands,
                    "missing_subbands": missing_sbs,
                },
            )

    # Prepare output path
    output_path = os.path.join(output_dir, f"{group_id}.ms")

    if skip_existing and os.path.exists(output_path):
        logger.info(f"Skipping existing MS: {output_path}", extra={"output_path": output_path})
        return "skipped"

    logger.info(
        f"Converting {len(group)} subbands to {output_path}",
        extra={
            "subband_count": len(group),
            "output_path": output_path,
            "file_list": group,
        },
    )

    # Select writer up front so we can avoid redundant I/O.
    #
    # NOTE: DirectSubbandWriter reads and writes per-subband parts itself, so doing a
    # full UVData load+combine here would double-read the entire observation and
    # significantly slow real-time conversion.
    writer_type = settings.conversion.writer
    writer_cls = get_writer(writer_type)

    # Track I/O vs CPU time
    io_time = 0.0
    cpu_time = 0.0
    t0_all = time.perf_counter()

    if writer_type == "direct-subband":
        logger.info(
            "Direct-subband writer selected; skipping upfront UVData load (per-subband workers will read inputs)"
        )
        uvdata = pyuvdata.UVData()
    else:
        # Combine subbands using pyuvdata (with parallel I/O from settings)
        t0_io = time.perf_counter()
        uvdata = _load_and_combine_subbands(
            group,
            group_id,
            parallel=settings.conversion.parallel_loading,
            max_workers=settings.conversion.io_max_workers,
        )
        io_time += time.perf_counter() - t0_io

    # Get antenna positions (CPU)
    t0_cpu = time.perf_counter()
    try:
        antpos = get_itrf()
        logger.debug("Loaded antenna positions", extra={"ant_count": len(antpos)})
    except Exception as e:
        raise ConversionError(
            f"Failed to load antenna positions: {e}",
            group_id=group_id,
            original_exception=e,
        ) from e
    cpu_time += time.perf_counter() - t0_cpu

    # Write Measurement Set (use settings for writer type)
    try:
        # Pass file_list for DirectSubbandWriter, scratch_dir for temp files
        # Include tmpfs staging settings for 3-5x faster writes
        writer_kwargs = {
            "file_list": group,
            "scratch_dir": None,  # Use default temp dir
            "max_workers": settings.conversion.max_workers,
            "stage_to_tmpfs": stage_to_tmpfs,
            "tmpfs_path": getattr(
                settings.paths,
                "tmpfs_dir",
                os.environ.get(
                    "CONTIMG_TMPFS_DIR",
                    os.environ.get("CONTIMG_SCRATCH_DIR", str(settings.paths.tmpfs_dir)),
                ),
            ),
            "defer_final_copy": defer_final_copy,
        }

        t0_write = time.perf_counter()
        writer_instance = writer_cls(uvdata, output_path, **writer_kwargs)
        actual_writer = writer_instance.write()

        # For direct-subband, the write() call includes I/O and processing
        # For others, it's mostly I/O
        write_time = time.perf_counter() - t0_write
        if writer_type == "direct-subband":
            io_time += write_time * 0.7  # Estimate 70% I/O for direct-subband
            cpu_time += write_time * 0.3
        else:
            io_time += write_time

        # Determine the actual MS path (tmpfs if deferred, otherwise output_path)
        if defer_final_copy and stage_to_tmpfs:
            # Use configured scratch dir
            tmpfs_base = writer_kwargs["tmpfs_path"]
            tmpfs_staging_root = Path(tmpfs_base)
            if tmpfs_staging_root.name != "dsa110-contimg":
                tmpfs_staging_root = tmpfs_staging_root / "dsa110-contimg"
            actual_ms_path = tmpfs_staging_root / f"{Path(output_path).stem}.staged.ms"
            logger.info(
                f"Successfully wrote MS (tmpfs, deferred copy): {actual_ms_path}",
                extra={
                    "output_path": str(actual_ms_path),
                    "final_path": output_path,
                    "writer_type": actual_writer,
                    "deferred": True,
                },
            )
        else:
            actual_ms_path = Path(output_path)
            logger.info(
                f"Successfully wrote MS: {output_path}",
                extra={
                    "output_path": output_path,
                    "writer_type": actual_writer,
                },
            )
    except Exception as e:
        raise MSWriteError(
            output_path=output_path,
            reason=str(e),
            original_exception=e,
            group_id=group_id,
        ) from e

    # Record instrumentation metrics
    record_stage_timing(
        ms_path=output_path,
        stage=PipelineStage.CONVERSION,
        cpu_time_s=cpu_time,
        gpu_time_s=0.0,
        io_time_s=io_time,
        total_time_s=time.perf_counter() - t0_all,
    )

    return "converted"


def _load_single_subband(subband_file: str, group_id: str) -> pyuvdata.UVData:
    """Load a single subband file.

    This function is designed to be called in parallel for I/O-bound speedup.
    Thread-safe: Each call creates its own UVData object.

    Parameters
    ----------
    subband_file :
        Path to the UVH5 subband file.
    group_id :
        Group identifier for error messages.

    Returns
    -------
        Loaded UVData object.

    Raises
    ------
    UVH5ReadError
        If reading fails.

    """
    try:
        # Validate file with fast metadata read first
        with FastMeta(subband_file) as meta:
            _ = meta.time_array  # Quick validation

        # Read full data with explicit file_type (pyuvdata doesn't auto-detect .hdf5)
        # run_check=False: Skip dtype validation (DSA-110 files use float32 for uvw_array)
        subband_data = pyuvdata.UVData()
        subband_data.read(
            subband_file,
            file_type="uvh5",
            strict_uvw_antpos_check=False,
            run_check=False,
        )
        # DSA-110 files have uvw_array as float32, but pyuvdata requires float64
        # for __iadd__ operations during subband combination
        if subband_data.uvw_array.dtype != np.float64:
            subband_data.uvw_array = subband_data.uvw_array.astype(np.float64)
        # Set blt_order so pyuvdata does not emit "blt_order could not be identified"
        # when combining subbands (__iadd__). Matches write order in simulation/uvdata_writer.
        subband_data.blt_order = ("time", "baseline")
        return subband_data

    except FileNotFoundError as e:
        raise UVH5ReadError(
            file_path=subband_file,
            reason="File not found",
            original_exception=e,
            group_id=group_id,
        ) from e
    except Exception as e:
        raise UVH5ReadError(
            file_path=subband_file,
            reason=str(e),
            original_exception=e,
            group_id=group_id,
        ) from e


@timed_debug("conversion.load_and_combine_subbands")
def _load_and_combine_subbands(
    group: list[str],
    group_id: str,
    *,
    parallel: bool = True,
    max_workers: int = 4,
) -> pyuvdata.UVData:
    """Load and combine subband files into a single UVData object.

    Parameters
    ----------
    group :
        List of subband file paths.
    group_id :
        Group identifier for logging/errors.
    parallel :
        If True, load subbands in parallel (default: True).
    max_workers :
        Maximum number of parallel I/O threads (default: 4).
        Higher values may not help due to HDD seek limits.
    group: list[str] :

    Returns
    -------
        Combined UVData object with all subbands merged.

    Raises
    ------
    UVH5ReadError
        If any subband file fails to read.
    ConversionError
        If no valid data is loaded.

    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    sorted_files = sorted(group)
    n_files = len(sorted_files)

    if n_files == 0:
        raise ConversionError(
            "No subband files provided",
            group_id=group_id,
        )

    # For small groups or if parallel disabled, use sequential loading
    if not parallel or n_files <= 2:
        if not parallel:
            logger.warning(
                f"Parallel loading disabled, using sequential fallback for {n_files} subbands (slower)"
            )
        return _load_subbands_sequential(sorted_files, group_id)

    # Parallel loading: load all subbands concurrently, then combine
    logger.info(
        f"Loading {n_files} subbands in parallel (max_workers={max_workers})",
        extra={
            "group_id": group_id,
            "subband_count": n_files,
            "max_workers": max_workers,
        },
    )

    # Use dict to preserve order: {file_index: UVData}
    loaded_subbands: dict[int, pyuvdata.UVData] = {}
    errors: list[tuple[str, str]] = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all load tasks
        future_to_index = {
            executor.submit(_load_single_subband, f, group_id): i
            for i, f in enumerate(sorted_files)
        }

        # Collect results as they complete
        for future in as_completed(future_to_index):
            idx = future_to_index[future]
            file_path = sorted_files[idx]

            try:
                uvdata = future.result()
                loaded_subbands[idx] = uvdata
                logger.debug(
                    f"Loaded subband {idx + 1}/{n_files}: {file_path}",
                    extra={"subband_index": idx, "subband_file": file_path},
                )
            except UVH5ReadError as e:
                errors.append((file_path, str(e)))
                logger.error(
                    f"Failed to load subband {idx + 1}/{n_files}: {e}",
                    extra={"subband_index": idx, "subband_file": file_path},
                )

    # Check for load failures
    if errors:
        failed_files = [f for f, _ in errors]
        raise UVH5ReadError(
            file_path=failed_files[0],
            reason=f"Failed to load {len(errors)} subband(s): {errors[0][1]}",
            group_id=group_id,
        )

    if not loaded_subbands:
        raise ConversionError(
            "No valid subband data loaded",
            group_id=group_id,
        )

    # Combine subbands in order (must be sequential for memory safety)
    logger.debug(
        f"Combining {len(loaded_subbands)} subbands sequentially",
        extra={"group_id": group_id},
    )

    uvdata = None
    for idx in range(n_files):
        if idx not in loaded_subbands:
            continue

        subband_data = loaded_subbands[idx]
        if uvdata is None:
            uvdata = subband_data
        else:
            uvdata += subband_data

        # Free memory as we go
        del loaded_subbands[idx]

    if uvdata is None:
        raise ConversionError(
            "No valid subband data loaded after combining",
            group_id=group_id,
        )

    return uvdata


def _load_subbands_sequential(
    sorted_files: list[str],
    group_id: str,
) -> pyuvdata.UVData:
    """Sequential subband loading (fallback when parallel is disabled).

    Parameters
    ----------
    sorted_files :
        List of subband file paths (sorted).
    group_id :
        Group identifier for logging/errors.
    sorted_files: list[str] :

    Returns
    -------
        Combined UVData object.

    Raises
    ------
    UVH5ReadError
        If any file fails to read.
    ConversionError
        If no data loaded.

    """
    uvdata = None

    for i, subband_file in enumerate(sorted_files):
        logger.debug(
            f"Loading subband {i + 1}/{len(sorted_files)}: {subband_file}",
            extra={
                "subband_index": i,
                "subband_file": subband_file,
            },
        )

        subband_data = _load_single_subband(subband_file, group_id)

        if uvdata is None:
            uvdata = subband_data
        else:
            uvdata += subband_data

    if uvdata is None:
        raise ConversionError(
            "No valid subband data loaded",
            group_id=group_id,
        )

    return uvdata


def _extract_group_id(group: list[str]) -> str:
    """Extract group ID (timestamp) from first file in group.

    Delegates to :func:`parse_subband_filename` for robust parsing,
    with a simple string-split fallback for non-standard filenames.

    Parameters
    ----------
    group : list[str]
        List of file paths in the subband group.

    Returns
    -------
    str
        Group ID (ISO timestamp) or ``"unknown"``.
    """
    if not group:
        return "unknown"

    first_file = os.path.basename(group[0])
    parsed = parse_subband_filename(first_file)
    if parsed is not None:
        return parsed[0]
    # Fallback for non-standard filenames
    return first_file.rsplit("_sb", 1)[0]


def _find_missing_subbands(group: list[str]) -> list[str]:
    """Find which subband indices are missing from a group.

    Parameters
    ----------
    group : list[str]
        List of file paths in the subband group.

    Returns
    -------
    list[str]
        Sorted list of missing subband codes (e.g., ``['sb02', 'sb14']``).
    """
    found: set[int] = set()
    for file_path in group:
        parsed = parse_subband_filename(os.path.basename(file_path))
        if parsed is not None:
            found.add(parsed[1])
    return sorted(f"sb{i:02d}" for i in range(16) if i not in found)
