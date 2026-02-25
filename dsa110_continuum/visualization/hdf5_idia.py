"""
HDF5 IDIA Schema Conversion for CARTA Visualization.

This module provides tools to convert FITS images and cubes to HDF5 format
using the IDIA schema, which is optimized for CARTA's visualization pipeline.

The IDIA schema provides:
- Pre-computed statistics at multiple resolution levels
- Tiled/chunked storage for efficient random access
- Built-in compression (LZ4 by default)
- 10-100x faster loading in CARTA compared to standard FITS

Note: This conversion is specifically useful for CARTA visualization.
The IDIA schema is not a general-purpose HDF5 format and other tools
may not be able to read these files correctly.

References
----------
- CARTA: https://cartavis.org/
- IDIA Schema: https://github.com/idia-astro/carta-backend
- fits2idia: https://github.com/CARTAvis/fits2idia
"""

import asyncio
import logging
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class ConversionResult:
    """Result of FITS to HDF5 IDIA conversion."""

    success: bool
    output_path: str | None = None
    original_path: str | None = None
    conversion_time_seconds: float | None = None
    compression_ratio: float | None = None
    error: str | None = None


def find_fits2idia() -> str | None:
    """Locate the fits2idia executable."""
    # Check if fits2idia is on PATH
    fits2idia_path = shutil.which("fits2idia")
    if fits2idia_path:
        return fits2idia_path

    # Check common installation locations
    common_paths = [
        "/usr/local/bin/fits2idia",
        "/usr/bin/fits2idia",
        "/opt/carta/bin/fits2idia",
        os.path.expanduser("~/.local/bin/fits2idia"),
    ]

    for path in common_paths:
        if os.path.isfile(path) and os.access(path, os.X_OK):
            return path

    return None


def _generate_output_path(fits_path: str, suffix: str = "_idia") -> str:
    """Generate output HDF5 path from FITS path.

    Parameters
    ----------
    fits_path : str
    suffix : str, optional
        Suffix to append, by default "_idia"

    """
    path = Path(fits_path)
    # Handle .fits, .fits.gz, .fits.fz extensions
    stem = path.stem
    if stem.endswith(".fits"):
        stem = stem[:-5]
    return str(path.parent / f"{stem}{suffix}.hdf5")


async def convert_fits_to_idia_hdf5(
    fits_path: str,
    output_path: str | None = None,
    compute_stats: bool = True,
    compression: str = "lz4",
    chunk_size: int = 256,
    overwrite: bool = False,
) -> ConversionResult:
    """Convert FITS file to HDF5 IDIA format for optimal CARTA performance.

        This function wraps the `fits2idia` tool provided by CARTA.
        If fits2idia is not available, it falls back to a pure Python
        implementation (slower but functional).

    Parameters
    ----------
    fits_path : str
        Path to input FITS file
    output_path : Optional[str], optional
        Path for output HDF5 (auto-generated if None), by default None
    compute_stats : bool, optional
        Pre-compute per-channel statistics (recommended), by default False
    compression : str, optional
        Compression type: 'lz4' (fast) or 'gzip' (smaller), by default "lz4"
    chunk_size : int, optional
        HDF5 chunk size in pixels (256 is optimal for most cases), by default 256
    overwrite : bool, optional
        If True, overwrite existing output file, by default False

    Returns
    -------
        ConversionResult
        Conversion result with output path and statistics

        Example
    -------
        >>> result = await convert_fits_to_idia_hdf5(
        ...     "/data/cubes/source.fits",
        ...     compute_stats=True,
        ... )
        >>> if result.success:
        ...     print(f"Converted to: {result.output_path}")
    """
    import time

    start_time = time.time()

    # Validate input
    if not os.path.exists(fits_path):
        return ConversionResult(
            success=False,
            original_path=fits_path,
            error=f"Input file not found: {fits_path}",
        )

    # Generate output path if not provided
    if output_path is None:
        output_path = _generate_output_path(fits_path)

    # Check if output exists
    if os.path.exists(output_path) and not overwrite:
        return ConversionResult(
            success=False,
            original_path=fits_path,
            output_path=output_path,
            error=f"Output file already exists: {output_path}",
        )

    # Try fits2idia first (fastest, most compatible)
    fits2idia_path = find_fits2idia()

    if fits2idia_path:
        result = await _convert_with_fits2idia(
            fits2idia_path,
            fits_path,
            output_path,
            compute_stats=compute_stats,
            compression=compression,
            chunk_size=chunk_size,
        )
    else:
        # Fall back to Python implementation
        logger.warning("fits2idia not found, using Python fallback (slower)")
        result = await _convert_with_python(
            fits_path,
            output_path,
            compute_stats=compute_stats,
            compression=compression,
            chunk_size=chunk_size,
        )

    # Add timing
    if result.success:
        result.conversion_time_seconds = time.time() - start_time
        result.original_path = fits_path

        # Compute compression ratio
        if os.path.exists(output_path):
            original_size = os.path.getsize(fits_path)
            converted_size = os.path.getsize(output_path)
            if converted_size > 0:
                result.compression_ratio = original_size / converted_size

    return result


async def _convert_with_fits2idia(
    fits2idia_path: str,
    fits_path: str,
    output_path: str,
    compute_stats: bool = True,
    compression: str = "lz4",
    chunk_size: int = 256,
) -> ConversionResult:
    """
    Convert using the fits2idia tool.

    fits2idia is the official CARTA conversion tool and produces
    optimal results for CARTA visualization.
    """
    # Build command
    cmd = [
        fits2idia_path,
        "-o",
        output_path,
    ]

    if compute_stats:
        cmd.append("-s")  # Compute statistics

    if compression == "gzip":
        cmd.extend(["-c", "gzip"])
    # lz4 is default

    cmd.extend(["-k", str(chunk_size)])  # Chunk size
    cmd.append(fits_path)

    try:
        # Run conversion
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        stdout, stderr = await process.communicate()

        if process.returncode == 0:
            logger.info(f"Converted {fits_path} to {output_path} using fits2idia")
            return ConversionResult(
                success=True,
                output_path=output_path,
            )
        else:
            error_msg = (
                stderr.decode().strip() or f"fits2idia exited with code {process.returncode}"
            )
            logger.error(f"fits2idia failed: {error_msg}")
            return ConversionResult(
                success=False,
                error=error_msg,
            )

    except Exception as e:
        logger.error(f"fits2idia execution failed: {e}")
        return ConversionResult(
            success=False,
            error=str(e),
        )


async def _convert_with_python(
    fits_path: str,
    output_path: str,
    compute_stats: bool = True,
    compression: str = "lz4",
    chunk_size: int = 256,
) -> ConversionResult:
    """
    Pure Python fallback conversion.

    This produces HDF5 files that CARTA can read, but they won't be
    as optimized as fits2idia output (no mipmaps, simpler statistics).
    """
    try:
        import h5py
        import numpy as np
        from astropy.io import fits
    except ImportError as e:
        return ConversionResult(
            success=False,
            error=f"Required packages not installed: {e}",
        )

    try:
        # Read FITS
        with fits.open(fits_path, memmap=True) as hdul:
            # Find the image HDU
            image_hdu = None
            for hdu in hdul:
                if hasattr(hdu, "data") and hdu.data is not None and hdu.data.ndim >= 2:
                    image_hdu = hdu
                    break

            if image_hdu is None:
                return ConversionResult(
                    success=False,
                    error="No image data found in FITS file",
                )

            data = image_hdu.data
            header = image_hdu.header

            # Set up HDF5 compression
            if compression == "lz4":
                # h5py may not have lz4, fall back to gzip
                try:
                    compression_opts = {"compression": "lz4"}
                except (ValueError, TypeError):
                    compression_opts = {"compression": "gzip", "compression_opts": 6}
            else:
                compression_opts = {"compression": "gzip", "compression_opts": 6}

            # Determine chunk shape
            ndim = data.ndim
            if ndim == 2:
                chunks = (min(chunk_size, data.shape[0]), min(chunk_size, data.shape[1]))
            elif ndim == 3:
                chunks = (1, min(chunk_size, data.shape[1]), min(chunk_size, data.shape[2]))
            elif ndim == 4:
                chunks = (1, 1, min(chunk_size, data.shape[2]), min(chunk_size, data.shape[3]))
            else:
                chunks = True  # Let h5py decide

            # Create HDF5 file
            with h5py.File(output_path, "w") as hf:
                # Store data with chunking
                hf.create_dataset(
                    "DATA",
                    data=data,
                    chunks=chunks,
                    **compression_opts,
                )

                # Store FITS header as attributes
                header_grp = hf.create_group("HEADER")
                for key, value in header.items():
                    if key and isinstance(value, (int, float, str, bool)):
                        try:
                            header_grp.attrs[key] = value
                        except (TypeError, ValueError):
                            pass

                # Compute and store statistics if requested
                if compute_stats:
                    stats_grp = hf.create_group("STATISTICS")

                    # Global statistics
                    valid_data = data[np.isfinite(data)]
                    if len(valid_data) > 0:
                        stats_grp.attrs["MIN"] = float(np.min(valid_data))
                        stats_grp.attrs["MAX"] = float(np.max(valid_data))
                        stats_grp.attrs["MEAN"] = float(np.mean(valid_data))
                        stats_grp.attrs["STDDEV"] = float(np.std(valid_data))
                        stats_grp.attrs["NVALID"] = int(len(valid_data))

                    # Per-channel statistics for cubes
                    if ndim >= 3:
                        n_channels = data.shape[-3] if ndim == 4 else data.shape[0]
                        chan_min = np.zeros(n_channels)
                        chan_max = np.zeros(n_channels)
                        chan_mean = np.zeros(n_channels)
                        chan_std = np.zeros(n_channels)

                        for i in range(n_channels):
                            if ndim == 4:
                                chan_data = data[0, i, :, :]
                            else:
                                chan_data = data[i, :, :]

                            valid = chan_data[np.isfinite(chan_data)]
                            if len(valid) > 0:
                                chan_min[i] = np.min(valid)
                                chan_max[i] = np.max(valid)
                                chan_mean[i] = np.mean(valid)
                                chan_std[i] = np.std(valid)

                        stats_grp.create_dataset("CHANNEL_MIN", data=chan_min)
                        stats_grp.create_dataset("CHANNEL_MAX", data=chan_max)
                        stats_grp.create_dataset("CHANNEL_MEAN", data=chan_mean)
                        stats_grp.create_dataset("CHANNEL_STDDEV", data=chan_std)

                # Store IDIA schema version
                hf.attrs["IDIA_SCHEMA_VERSION"] = "1.0"
                hf.attrs["CONVERTER"] = "dsa110-contimg-python"

        logger.info(f"Converted {fits_path} to {output_path} using Python")
        return ConversionResult(
            success=True,
            output_path=output_path,
        )

    except Exception as e:
        logger.error(f"Python conversion failed: {e}")
        # Clean up partial output
        if os.path.exists(output_path):
            try:
                os.remove(output_path)
            except OSError:
                pass
        return ConversionResult(
            success=False,
            error=str(e),
        )


async def batch_convert(
    fits_files: list[str],
    output_dir: str | None = None,
    max_concurrent: int = 4,
    compute_stats: bool = True,
) -> list[ConversionResult]:
    """Convert multiple FITS files to HDF5 IDIA format concurrently.

    Parameters
    ----------
    fits_files : List[str]
        List of FITS file paths
    output_dir : Optional[str], optional
        Directory for output files (same as input if None), by default None
    max_concurrent : Optional[int], optional
        Maximum concurrent conversions, by default None
    compute_stats : bool, optional
        Pre-compute statistics, by default False

    Returns
    -------
        List[ConversionResult]
        List of conversion results for each file
    """
    semaphore = asyncio.Semaphore(max_concurrent)

    async def convert_one(fits_path: str) -> ConversionResult:
        async with semaphore:
            output_path = None
            if output_dir:
                output_path = os.path.join(output_dir, Path(fits_path).stem + "_idia.hdf5")
            return await convert_fits_to_idia_hdf5(
                fits_path,
                output_path=output_path,
                compute_stats=compute_stats,
            )

    return await asyncio.gather(*[convert_one(f) for f in fits_files])


def check_idia_format(hdf5_path: str) -> dict:
    """Check if an HDF5 file is in IDIA format.

    Parameters
    ----------
    hdf5_path : str
        Path to HDF5 file

    """
    try:
        import h5py
    except ImportError:
        return {
            "is_idia": False,
            "error": "h5py not installed",
        }

    try:
        with h5py.File(hdf5_path, "r") as hf:
            schema_version = hf.attrs.get("IDIA_SCHEMA_VERSION")
            has_data = "DATA" in hf
            has_stats = "STATISTICS" in hf

            return {
                "is_idia": schema_version is not None or has_data,
                "schema_version": schema_version.decode()
                if isinstance(schema_version, bytes)
                else schema_version,
                "has_data": has_data,
                "has_statistics": has_stats,
                "converter": hf.attrs.get("CONVERTER"),
            }
    except Exception as e:
        return {
            "is_idia": False,
            "error": str(e),
        }


# Convenience function for CLI usage
def main():
    """CLI entry point for FITS to IDIA conversion."""
    import argparse

    parser = argparse.ArgumentParser(description="Convert FITS files to HDF5 IDIA format for CARTA")
    parser.add_argument("input", help="Input FITS file")
    parser.add_argument("-o", "--output", help="Output HDF5 file")
    parser.add_argument("--no-stats", action="store_true", help="Skip statistics computation")
    parser.add_argument("--compression", choices=["lz4", "gzip"], default="lz4")

    args = parser.parse_args()

    result = asyncio.run(
        convert_fits_to_idia_hdf5(
            args.input,
            output_path=args.output,
            compute_stats=not args.no_stats,
            compression=args.compression,
        )
    )

    if result.success:
        print(f"Success: {result.output_path}")
        if result.compression_ratio:
            print(f"Compression ratio: {result.compression_ratio:.2f}x")
        if result.conversion_time_seconds:
            print(f"Time: {result.conversion_time_seconds:.1f}s")
    else:
        print(f"Error: {result.error}")
        exit(1)


if __name__ == "__main__":
    main()
