"""
SPW selection and per-SPW imaging for adaptive binning.

This module provides functions to:
1. Query SPW information from Measurement Sets
2. Image individual SPWs
3. Image all SPWs and return paths for adaptive binning
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Ensure CASAPATH is set before importing CASA modules
from dsa110_contimg.common.utils.casa_init import ensure_casa_path

ensure_casa_path()

try:
    import casacore.tables as casatables

    table = casatables.table  # noqa: N816
except ImportError:
    table = None  # type: ignore[assignment, misc]

from dsa110_contimg.core.imaging.cli_imaging import image_ms

LOG = logging.getLogger(__name__)


@dataclass
class SPWInfo:
    """Information about a spectral window."""

    spw_id: int
    center_freq_mhz: float
    bandwidth_mhz: float
    num_channels: int
    freq_min_mhz: float
    freq_max_mhz: float


def get_spw_info(ms_path: str) -> list[SPWInfo]:
    """Get SPW information from Measurement Set.

    Parameters
    ----------
    ms_path :
        Path to Measurement Set

    Returns
    -------
        List of SPWInfo objects, one per SPW

    Raises
    ------
    RuntimeError
        If casacore.tables is not available or MS cannot be read

    """
    if table is None:
        raise RuntimeError(
            "casacore.tables required for SPW queries. "
            "Install via: conda install -c conda-forge casacore"
        )

    spw_info_list = []

    try:
        with table(f"{ms_path}::SPECTRAL_WINDOW", readonly=True) as spw_tab:
            n_spws = spw_tab.nrows()

            # Get channel frequencies for each SPW
            chan_freqs = spw_tab.getcol("CHAN_FREQ")  # Shape: (n_spw, n_chan)
            num_chans = spw_tab.getcol("NUM_CHAN")  # Shape: (n_spw,)

            for spw_id in range(n_spws):
                if spw_id >= len(chan_freqs):
                    continue

                freqs_hz = chan_freqs[spw_id]
                if len(freqs_hz) == 0:
                    continue

                freq_min_hz = float(np.min(freqs_hz))
                freq_max_hz = float(np.max(freqs_hz))
                center_freq_hz = float(np.mean(freqs_hz))
                bandwidth_hz = freq_max_hz - freq_min_hz

                # Convert to MHz
                center_freq_mhz = center_freq_hz / 1e6
                bandwidth_mhz = bandwidth_hz / 1e6
                freq_min_mhz = freq_min_hz / 1e6
                freq_max_mhz = freq_max_hz / 1e6

                spw_info_list.append(
                    SPWInfo(
                        spw_id=spw_id,
                        center_freq_mhz=center_freq_mhz,
                        bandwidth_mhz=bandwidth_mhz,
                        num_channels=int(num_chans[spw_id]),
                        freq_min_mhz=freq_min_mhz,
                        freq_max_mhz=freq_max_mhz,
                    )
                )

    except Exception as e:
        raise RuntimeError(f"Failed to read SPW information from {ms_path}: {e}") from e

    return spw_info_list


def image_spw(
    ms_path: str,
    spw_id: int,
    output_dir: Path,
    base_name: str = "spw",
    **imaging_kwargs,
) -> Path:
    """Image a single SPW.

    Parameters
    ----------
    ms_path : str
        Path to Measurement Set
    spw_id : int
        SPW ID to image (0-indexed)
    output_dir : Path
        Directory for output images
    base_name : str, optional
        Base name for output images (will append spw_id)
        **imaging_kwargs :
        Additional arguments passed to image_ms()

    Returns
    -------
        Path
        Path to primary beam corrected FITS image

    Examples
    --------
        >>> image_path = image_spw(
        ...     ms_path="data.ms",
        ...     spw_id=0,
        ...     output_dir=Path("images/"),
        ...     imsize=1024,
        ...     quality_tier="standard",
        ... )
        >>> print(image_path)
        Path('images/spw0.img-image-pbcor.fits')
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create imagename with SPW suffix
    imagename = str(output_dir / f"{base_name}{spw_id}.img")

    # Select SPW using CASA SPW selection syntax (e.g., "0" for SPW 0)
    spw_selection = str(spw_id)

    # Image this SPW
    image_ms(
        ms_path,
        imagename=imagename,
        spw=spw_selection,
        **imaging_kwargs,
    )

    # Find the PB-corrected FITS image
    pbcor_fits = f"{imagename}-image-pbcor.fits"
    if Path(pbcor_fits).exists():
        return Path(pbcor_fits)

    # Fallback: try CASA image format
    pbcor_casa = f"{imagename}.image.pbcor"
    if Path(pbcor_casa).exists():
        # Convert to FITS if needed
        from dsa110_contimg.core.imaging.export import export_fits

        casa_images = [pbcor_casa]
        exported = export_fits(casa_images)
        if exported and exported[0] and Path(exported[0]).exists():
            return Path(exported[0])

    raise RuntimeError(
        f"Failed to create image for SPW {spw_id}. Expected output: {pbcor_fits} or {pbcor_casa}"
    )  # noqa: E501


def _image_spw_parallel_wrapper(
    args: tuple[str, int, Path, str, dict],
) -> tuple[int, Path]:
    """Wrapper function for parallel SPW imaging (must be at module level for pickling).

    Parameters
    ----------
    args :
        Tuple of (ms_path, spw_id, output_dir, base_name, imaging_kwargs)
    args: Tuple[str :

    int :

    Path :

    str :

    dict] :


    Returns
    -------
        Tuple of (spw_id, image_path)

    """
    ms_path, spw_id, output_dir, base_name, imaging_kwargs = args
    try:
        image_path = image_spw(
            ms_path=ms_path,
            spw_id=spw_id,
            output_dir=output_dir,
            base_name=base_name,
            **imaging_kwargs,
        )
        LOG.info(f"SPW {spw_id} complete: {image_path}")
        return (spw_id, image_path)
    except Exception as e:
        LOG.error(f"Failed to image SPW {spw_id}: {e}", exc_info=True)
        raise


def image_all_spws(
    ms_path: str,
    output_dir: Path,
    base_name: str = "spw",
    spw_ids: list[int] | None = None,
    parallel: bool = False,
    max_workers: int | None = None,
    serialize_ms_access: bool = False,
    **imaging_kwargs,
) -> list[tuple[int, Path]]:
    """Image all SPWs (or specified subset) and return paths.

    Parameters
    ----------
    ms_path : str
        Path to Measurement Set
    output_dir : Path
        Directory for output images
    base_name : str, optional
        Base name for output images (will append spw_id)
    spw_ids : list of int, optional
        Optional list of SPW IDs to image. If None, images all SPWs.
    parallel : bool, optional
        If True, image SPWs in parallel (default: False)
    max_workers : int, optional
        Maximum number of parallel workers (default: CPU count)
    serialize_ms_access : bool, optional
        If True, serialize MS access using file locking to
        prevent CASA table lock conflicts when multiple
        processes access the same MS (default: False)
        **imaging_kwargs :
        Additional arguments passed to image_ms()

    Returns
    -------
        list of tuple
        List of (spw_id, image_path) tuples, sorted by SPW ID

    Examples
    --------
        >>> spw_images = image_all_spws(
        ...     ms_path="data.ms",
        ...     output_dir=Path("images/"),
        ...     imsize=1024,
        ...     quality_tier="standard",
        ...     parallel=True,
        ...     max_workers=4,
        ...     serialize_ms_access=True,
        ... )
        >>> print(f"Imaged {len(spw_images)} SPWs")
        >>> for spw_id, img_path in spw_images:
        ...     print(f"SPW {spw_id}: {img_path}")
    """
    # Import MS locking utility if serialization is enabled
    if serialize_ms_access:
        from dsa110_contimg.common.utils.ms_locking import cleanup_stale_locks, ms_lock

        # Clean up any stale locks before starting
        cleanup_stale_locks(ms_path)

        # Use lock context manager for entire SPW imaging operation
        with ms_lock(ms_path, timeout=3600.0):
            return _image_all_spws_impl(
                ms_path=ms_path,
                output_dir=output_dir,
                base_name=base_name,
                spw_ids=spw_ids,
                parallel=parallel,
                max_workers=max_workers,
                **imaging_kwargs,
            )
    else:
        # No locking, proceed directly
        return _image_all_spws_impl(
            ms_path=ms_path,
            output_dir=output_dir,
            base_name=base_name,
            spw_ids=spw_ids,
            parallel=parallel,
            max_workers=max_workers,
            **imaging_kwargs,
        )


def _image_all_spws_impl(
    ms_path: str,
    output_dir: Path,
    base_name: str = "spw",
    spw_ids: list[int] | None = None,
    parallel: bool = False,
    max_workers: int | None = None,
    **imaging_kwargs,
) -> list[tuple[int, Path]]:
    """Internal implementation of image_all_spws (without locking).

    This function is separated out so that locking can be applied at the
    outer level when serialize_ms_access=True.

    Parameters
    ----------
    """
    # Get SPW information
    spw_info_list = get_spw_info(ms_path)

    if not spw_info_list:
        raise RuntimeError(f"No SPWs found in {ms_path}")

    # Determine which SPWs to image
    if spw_ids is None:
        spw_ids_to_image = [info.spw_id for info in spw_info_list]
    else:
        # Validate SPW IDs
        available_ids = {info.spw_id for info in spw_info_list}
        invalid_ids = set(spw_ids) - available_ids
        if invalid_ids:
            msg = f"Invalid SPW IDs: {sorted(invalid_ids)}. Available SPWs: {sorted(available_ids)}"
            raise ValueError(msg)
        spw_ids_to_image = sorted(spw_ids)

    LOG.info(f"Imaging {len(spw_ids_to_image)} SPW(s): {spw_ids_to_image}")

    if parallel and len(spw_ids_to_image) > 1:
        # Parallel imaging
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor, as_completed

        if max_workers is None:
            max_workers = min(mp.cpu_count(), len(spw_ids_to_image))

        LOG.info(f"Using parallel imaging with {max_workers} worker(s)")

        results = []
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Prepare arguments for each SPW
            futures = {}
            for spw_id in spw_ids_to_image:
                args = (ms_path, spw_id, output_dir, base_name, imaging_kwargs)
                future = executor.submit(_image_spw_parallel_wrapper, args)
                futures[future] = spw_id

            for future in as_completed(futures):
                spw_id = futures[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    LOG.error(f"SPW {spw_id} failed: {e}")
                    # Continue with other SPWs
                    continue
    else:
        # Sequential imaging
        results = []
        for spw_id in spw_ids_to_image:
            try:
                LOG.info(f"Imaging SPW {spw_id}...")
                image_path = image_spw(
                    ms_path=ms_path,
                    spw_id=spw_id,
                    output_dir=output_dir,
                    base_name=base_name,
                    **imaging_kwargs,
                )
                results.append((spw_id, image_path))
                LOG.info(f"SPW {spw_id} complete: {image_path}")
            except Exception as e:
                LOG.error(f"Failed to image SPW {spw_id}: {e}", exc_info=True)
                # Continue with other SPWs
                continue

    if not results:
        raise RuntimeError("No SPWs were successfully imaged")

    LOG.info(f"Successfully imaged {len(results)}/{len(spw_ids_to_image)} SPW(s)")

    return sorted(results, key=lambda x: x[0])
