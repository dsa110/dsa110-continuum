"""
Aegean forced photometry integration for DSA-110 pipeline.

This module provides forced fitting capabilities using the Aegean source finder,
following the WABIFAT approach for improved flux measurements on extended/blended sources.

Requirements:
- Aegean source finder (install via: pip install git+https://github.com/PaulHancock/Aegean.git)
- BANE (Background And Noise Estimation) tool (usually bundled with Aegean)

Installation:
    pip install git+https://github.com/PaulHancock/Aegean.git
    # Or from cloned repo:
    cd ~/proj/Aegean && pip install .
"""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from astropy.io import fits  # type: ignore[reportMissingTypeStubs]

# Ensure user site-packages is in path (for pip install --user)
try:
    import site

    user_site = site.getusersitepackages()
    if user_site and user_site not in sys.path:
        sys.path.insert(0, user_site)
except (AttributeError, TypeError):
    pass  # Ignore if site module fails


@dataclass
class AegeanResult:
    """Result from Aegean forced fitting."""

    ra_deg: float
    dec_deg: float
    peak_flux_jy: float
    err_peak_flux_jy: float
    local_rms_jy: float
    integrated_flux_jy: float | None = None
    err_integrated_flux_jy: float | None = None
    a_arcsec: float | None = None  # Major axis (arcsec)
    b_arcsec: float | None = None  # Minor axis (arcsec)
    pa_deg: float | None = None  # Position angle (degrees)
    success: bool = True
    error_message: str | None = None


def _check_aegean_available() -> tuple[bool, str | None]:
    """Check if Aegean is available.

    Checks multiple methods:
    1. Command-line tool 'Aegean' in PATH
    2. Python module 'AegeanTools.Aegean' via python -m
    3. Python module import (for programmatic use)

    """
    # Method 1: Check command-line tool
    try:
        result = subprocess.run(
            ["Aegean", "--version"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return True, None
    except FileNotFoundError:
        pass
    except (subprocess.SubprocessError, OSError):
        pass

    # Method 2: Check command-line script (in ~/.local/bin)
    try:
        import os

        home = os.path.expanduser("~")
        aegean_script = os.path.join(home, ".local", "bin", "aegean")
        if os.path.exists(aegean_script):
            result = subprocess.run(
                [aegean_script, "--version"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            # Exit code 120 is normal for --version (it shows version then exits)
            if result.returncode in (0, 120) or "Aegean" in result.stdout:
                return True, None
    except (subprocess.SubprocessError, OSError):
        pass

    # Method 3: Check if module can be imported (for programmatic use)
    try:
        import AegeanTools  # noqa: F401 - checking availability

        return True, None
    except ImportError:
        pass

    return False, (
        "Aegean not found. Install via: "
        "pip install git+https://github.com/PaulHancock/Aegean.git "
        "or pip install AegeanTools"
    )


def _check_bane_available() -> tuple[bool, str | None]:
    """Check if BANE is available.

    Checks multiple methods:
    1. Command-line tool 'BANE' in PATH
    2. Python module 'AegeanTools.BANE' via python -m
    3. Python module import (for programmatic use)

    """
    # Method 1: Check command-line tool
    try:
        result = subprocess.run(
            ["BANE", "--version"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return True, None
    except FileNotFoundError:
        pass
    except (subprocess.SubprocessError, OSError):
        pass

    # Method 2: Check Python module via -m flag
    try:
        import sys

        python_exe = sys.executable
        result = subprocess.run(
            [
                python_exe,
                "-m",
                "AegeanTools.CLI.BANE",
                "--version",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return True, None
    except (subprocess.SubprocessError, OSError):
        pass

    # Method 2b: Check command-line script (in ~/.local/bin)
    try:
        import os

        home = os.path.expanduser("~")
        bane_script = os.path.join(home, ".local", "bin", "BANE")
        if os.path.exists(bane_script):
            result = subprocess.run(
                [bane_script, "--version"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                return True, None
    except (subprocess.SubprocessError, OSError):
        pass

    # Method 3: Check if module can be imported (for programmatic use)
    try:
        import AegeanTools  # noqa: F401 - checking availability

        return True, None
    except ImportError:
        pass

    return False, (
        "BANE not found. Install via: "
        "pip install git+https://github.com/PaulHancock/Aegean.git "
        "or pip install AegeanTools"
    )


def _extract_psf_from_header(header: fits.Header) -> tuple[float, float, float]:
    """Extract PSF parameters from FITS header.

    Parameters
    ----------
    header : fits.Header
        FITS header

    """
    # FITS headers typically have BMAJ/BMIN in degrees
    bmaj_deg = header.get("BMAJ")
    bmin_deg = header.get("BMIN")
    bpa_deg = header.get("BPA", 0.0)

    if bmaj_deg is None or bmin_deg is None:
        raise KeyError("BMAJ or BMIN not found in FITS header")

    # Convert to arcseconds
    bmaj_arcsec = float(bmaj_deg) * 3600.0
    bmin_arcsec = float(bmin_deg) * 3600.0

    return bmaj_arcsec, bmin_arcsec, float(bpa_deg)


def _create_aegean_input_table(
    ra_deg: float,
    dec_deg: float,
    bmaj_arcsec: float,
    bmin_arcsec: float,
    bpa_deg: float,
    output_path: str,
) -> None:
    """Create Aegean input table with source position and PSF.

    Follows WABIFAT pattern: creates a FITS table with source position,
    dummy peak flux, and PSF parameters.

    Parameters
    ----------
    ra_deg :
        Right ascension (degrees)
    dec_deg :
        Declination (degrees)
    bmaj_arcsec :
        PSF major axis (arcsec)
    bmin_arcsec :
        PSF minor axis (arcsec)
    bpa_deg :
        PSF position angle (degrees)
    output_path :
        Path to output FITS table
    """
    cols = fits.ColDefs(
        [
            fits.Column(name="ra", format="D", array=np.array([ra_deg])),
            fits.Column(name="dec", format="D", array=np.array([dec_deg])),
            fits.Column(name="peak_flux", format="E", array=np.array([1.0])),  # Dummy
            fits.Column(name="a", format="E", array=np.array([bmaj_arcsec])),
            fits.Column(name="b", format="E", array=np.array([bmin_arcsec])),
            fits.Column(name="pa", format="E", array=np.array([bpa_deg])),
            fits.Column(name="psf_a", format="E", array=np.array([bmaj_arcsec])),
            fits.Column(name="psf_b", format="E", array=np.array([bmin_arcsec])),
            fits.Column(name="psf_pa", format="E", array=np.array([bpa_deg])),
        ]
    )

    hdu = fits.BinTableHDU.from_columns(cols)
    hdu.writeto(output_path, overwrite=True)


def _get_bane_command() -> list[str]:
    """Get BANE command (tries multiple methods)."""
    import os
    import sys

    # Try Python module first (most reliable)
    try:
        python_exe = sys.executable
        result = subprocess.run(
            [
                python_exe,
                "-m",
                "AegeanTools.CLI.BANE",
                "--version",
            ],
            capture_output=True,
            text=True,
            timeout=2,
        )
        if result.returncode == 0:
            return [python_exe, "-m", "AegeanTools.CLI.BANE"]
    except (subprocess.SubprocessError, OSError):
        pass

    # Try command-line script in ~/.local/bin
    try:
        home = os.path.expanduser("~")
        bane_script = os.path.join(home, ".local", "bin", "BANE")
        if os.path.exists(bane_script):
            result = subprocess.run(
                [bane_script, "--version"],
                capture_output=True,
                text=True,
                timeout=2,
            )
            if result.returncode == 0:
                return [bane_script]
    except (subprocess.SubprocessError, OSError):
        pass

    # Try command-line tool (if in PATH)
    try:
        result = subprocess.run(
            ["BANE", "--version"],
            capture_output=True,
            text=True,
            timeout=2,
        )
        if result.returncode == 0:
            return ["BANE"]
    except (subprocess.SubprocessError, OSError):
        pass

    # Fallback: try Python module anyway
    python_exe = sys.executable
    return [python_exe, "-m", "AegeanTools.CLI.BANE"]


def _run_bane(
    fits_path: str,
    output_dir: Path,
) -> tuple[str, str]:
    """Run BANE to estimate RMS and background.

    Parameters
    ----------
    fits_path : str
        Path to input FITS image
    output_dir : Path
        Directory for BANE output files

    """
    bane_available, error = _check_bane_available()
    if not bane_available:
        raise RuntimeError(f"BANE not available: {error}")

    # BANE creates files with suffixes: _rms.fits and _bkg.fits
    base_name = Path(fits_path).stem
    rms_path = str(output_dir / f"{base_name}_rms.fits")
    bkg_path = str(output_dir / f"{base_name}_bkg.fits")

    # Get BANE command (tries multiple methods)
    bane_cmd_base = _get_bane_command()

    # Run BANE
    cmd = bane_cmd_base + [fits_path]
    result = subprocess.run(
        cmd,
        cwd=str(output_dir),
        capture_output=True,
        text=True,
        timeout=300,  # 5 minute timeout
    )

    if result.returncode != 0:
        raise RuntimeError(f"BANE failed: {result.stderr}\nCommand: {' '.join(cmd)}")

    # Check if output files exist
    if not Path(rms_path).exists():
        raise RuntimeError(f"BANE RMS output not found: {rms_path}")
    if not Path(bkg_path).exists():
        raise RuntimeError(f"BANE background output not found: {bkg_path}")

    return rms_path, bkg_path


def _get_aegean_command() -> list[str]:
    """Get Aegean command (tries multiple methods).

    Prefers command-line script as it's the standard installation method.

    """
    import os

    # Try command-line script in ~/.local/bin (pip install --user)
    try:
        home = os.path.expanduser("~")
        aegean_script = os.path.join(home, ".local", "bin", "aegean")
        if os.path.exists(aegean_script):
            result = subprocess.run(
                [aegean_script, "--version"],
                capture_output=True,
                text=True,
                timeout=2,
            )
            # Exit code 120 is normal for --version
            if result.returncode in (0, 120) or "Aegean" in result.stdout:
                return [aegean_script]
    except (subprocess.SubprocessError, OSError):
        pass

    # Try command-line tool (if in PATH)
    try:
        result = subprocess.run(
            ["aegean", "--version"],
            capture_output=True,
            text=True,
            timeout=2,
        )
        # Exit code 120 is normal for --version
        if result.returncode in (0, 120) or "Aegean" in result.stdout:
            return ["aegean"]
    except (subprocess.SubprocessError, OSError):
        pass

    # Fallback: try script path anyway
    home = os.path.expanduser("~")
    return [os.path.join(home, ".local", "bin", "aegean")]


def _run_aegean(
    image_path: str,
    rms_path: str,
    bkg_path: str,
    input_table_path: str,
    output_table_path: str,
    *,
    prioritized: bool = True,
    negative: bool = False,
) -> None:
    """Run Aegean with forced fitting.

    Parameters
    ----------
    image_path : str
        Path to input FITS image
    rms_path : str
        Path to RMS FITS (from BANE)
    bkg_path : str
        Path to background FITS (from BANE)
    input_table_path : str
        Path to input table with source positions
    output_table_path : str
        Path to output table
    prioritized : bool, optional
        Use --priorized flag (for blended sources)
        (Default value = True)
    negative : bool, optional
        Allow negative detections
        (Default value = False)

    """
    aegean_available, error = _check_aegean_available()
    if not aegean_available:
        raise RuntimeError(f"Aegean not available: {error}")

    # Get Aegean command (tries multiple methods)
    aegean_cmd_base = _get_aegean_command()

    # Build Aegean command following WABIFAT pattern
    cmd = aegean_cmd_base + [
        "--autoload",
        "--priorized",
        "1" if prioritized else "0",
    ]

    if negative:
        cmd.append("--negative")

    cmd.extend(
        [
            "--input",
            input_table_path,
            "--floodclip",
            "-1",  # Disable flood clipping
            "--table",
            output_table_path,
            "--noise",
            rms_path,
            "--background",
            bkg_path,
            image_path,
        ]
    )

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=300,  # 5 minute timeout
    )

    if result.returncode != 0:
        raise RuntimeError(f"Aegean failed: {result.stderr}\nCommand: {' '.join(cmd)}")


def _extract_aegean_results(
    output_table_path: str,
    ra_deg: float,
    dec_deg: float,
) -> AegeanResult:
    """Extract results from Aegean output table.

    Parameters
    ----------
    output_table_path : str
        Path to Aegean output FITS table
    ra_deg : float
        Expected RA (degrees)
    dec_deg : float
        Expected Dec (degrees)

    """
    if not Path(output_table_path).exists():
        return AegeanResult(
            ra_deg=ra_deg,
            dec_deg=dec_deg,
            peak_flux_jy=float("nan"),
            err_peak_flux_jy=float("nan"),
            local_rms_jy=float("nan"),
            success=False,
            error_message=f"Output table not found: {output_table_path}",
        )

    try:
        with fits.open(output_table_path) as hdul:
            # Aegean output is typically in extension 1
            if len(hdul) < 2:
                return AegeanResult(
                    ra_deg=ra_deg,
                    dec_deg=dec_deg,
                    peak_flux_jy=float("nan"),
                    err_peak_flux_jy=float("nan"),
                    local_rms_jy=float("nan"),
                    success=False,
                    error_message="Aegean output table has no data extension",
                )

            data = hdul[1].data  # pylint: disable=no-member

            # Extract first source (should match input position)
            peak_flux = float(data["peak_flux"][0])
            err_peak_flux = float(data["err_peak_flux"][0])
            local_rms = float(data["local_rms"][0])

            # Handle negative detections (WABIFAT pattern)
            if peak_flux < 0:
                local_rms = -local_rms

            # Extract optional parameters
            integrated_flux = None
            err_integrated_flux = None
            a_arcsec = None
            b_arcsec = None
            pa_deg = None

            if "int_flux" in data.names:
                integrated_flux = float(data["int_flux"][0])
            if "err_int_flux" in data.names:
                err_integrated_flux = float(data["err_int_flux"][0])
            if "a" in data.names:
                a_arcsec = float(data["a"][0])
            if "b" in data.names:
                b_arcsec = float(data["b"][0])
            if "pa" in data.names:
                pa_deg = float(data["pa"][0])

            return AegeanResult(
                ra_deg=ra_deg,
                dec_deg=dec_deg,
                peak_flux_jy=peak_flux,
                err_peak_flux_jy=err_peak_flux,
                local_rms_jy=local_rms,
                integrated_flux_jy=integrated_flux,
                err_integrated_flux_jy=err_integrated_flux,
                a_arcsec=a_arcsec,
                b_arcsec=b_arcsec,
                pa_deg=pa_deg,
                success=True,
            )

    except Exception as e:
        return AegeanResult(
            ra_deg=ra_deg,
            dec_deg=dec_deg,
            peak_flux_jy=float("nan"),
            err_peak_flux_jy=float("nan"),
            local_rms_jy=float("nan"),
            success=False,
            error_message=f"Error reading Aegean output: {e}",
        )


def measure_with_aegean(
    fits_path: str,
    ra_deg: float,
    dec_deg: float,
    *,
    use_prioritized: bool = True,
    negative: bool = False,
    cleanup_temp: bool = True,
    temp_dir: str | None = None,
) -> AegeanResult:
    """Measure source using Aegean forced fitting.

        Follows WABIFAT's forced_fitter() approach:
        1. Extract PSF from FITS header
        2. Run BANE for RMS/background estimation
        3. Create input table with source position + PSF
        4. Run Aegean with --priorized flag
        5. Extract peak_flux, err_peak_flux, local_rms

    Parameters
    ----------
    fits_path : str
        Path to input FITS image
    ra_deg : float
        Right ascension (degrees)
    dec_deg : float
        Declination (degrees)
    use_prioritized : bool, optional
        Use --priorized flag (for blended sources)
        (Default value = True)
    negative : bool, optional
        Allow negative detections
        (Default value = False)
    cleanup_temp : bool, optional
        Clean up temporary files after execution
        (Default value = True)
    temp_dir : str or None, optional
        Temporary directory (created if None)
        (Default value = None)

    Returns
    -------
        object
        Result object with peak flux and error

    Examples
    --------
        >>> result = measure_with_aegean(
        ...     'image.pbcor.fits',
        ...     ra_deg=128.725,
        ...     dec_deg=55.573,
        ... )
        >>> print(f"Peak flux: {result.peak_flux_jy:.6f} Jy/beam")
        >>> print(f"Error: {result.err_peak_flux_jy:.6f} Jy/beam")
    """
    fits_path_obj = Path(fits_path)
    if not fits_path_obj.exists():
        return AegeanResult(
            ra_deg=ra_deg,
            dec_deg=dec_deg,
            peak_flux_jy=float("nan"),
            err_peak_flux_jy=float("nan"),
            local_rms_jy=float("nan"),
            success=False,
            error_message=f"FITS file not found: {fits_path}",
        )

    # Extract PSF parameters from header
    try:
        header = fits.getheader(fits_path)
        bmaj_arcsec, bmin_arcsec, bpa_deg = _extract_psf_from_header(header)
    except Exception as e:
        return AegeanResult(
            ra_deg=ra_deg,
            dec_deg=dec_deg,
            peak_flux_jy=float("nan"),
            err_peak_flux_jy=float("nan"),
            local_rms_jy=float("nan"),
            success=False,
            error_message=f"Error extracting PSF from header: {e}",
        )

    # Create temporary directory for intermediate files
    if temp_dir is None:
        from dsa110_contimg.common.utils.temp_manager import get_temp_subdir

        temp_dir_obj = get_temp_subdir("aegean")
    else:
        temp_dir_obj = Path(temp_dir)
        temp_dir_obj.mkdir(parents=True, exist_ok=True)

    try:
        # Step 1: Run BANE for RMS/background estimation
        try:
            rms_path, bkg_path = _run_bane(fits_path, temp_dir_obj)
        except Exception as e:
            return AegeanResult(
                ra_deg=ra_deg,
                dec_deg=dec_deg,
                peak_flux_jy=float("nan"),
                err_peak_flux_jy=float("nan"),
                local_rms_jy=float("nan"),
                success=False,
                error_message=f"BANE failed: {e}",
            )

        # Step 2: Create input table
        input_table_path = str(temp_dir_obj / "aegean_input.fits")
        _create_aegean_input_table(
            ra_deg=ra_deg,
            dec_deg=dec_deg,
            bmaj_arcsec=bmaj_arcsec,
            bmin_arcsec=bmin_arcsec,
            bpa_deg=bpa_deg,
            output_path=input_table_path,
        )

        # Step 3: Run Aegean
        output_table_path = str(temp_dir_obj / "aegean_output.fits")
        try:
            _run_aegean(
                image_path=fits_path,
                rms_path=rms_path,
                bkg_path=bkg_path,
                input_table_path=input_table_path,
                output_table_path=output_table_path,
                prioritized=use_prioritized,
                negative=negative,
            )
        except Exception as e:
            return AegeanResult(
                ra_deg=ra_deg,
                dec_deg=dec_deg,
                peak_flux_jy=float("nan"),
                err_peak_flux_jy=float("nan"),
                local_rms_jy=float("nan"),
                success=False,
                error_message=f"Aegean execution failed: {e}",
            )

        # Step 4: Extract results
        result = _extract_aegean_results(
            output_table_path=output_table_path,
            ra_deg=ra_deg,
            dec_deg=dec_deg,
        )

        return result

    finally:
        # Cleanup temporary files
        if cleanup_temp and temp_dir is None:
            import shutil

            try:
                shutil.rmtree(temp_dir_obj, ignore_errors=True)
            except OSError:
                pass  # Ignore cleanup errors
