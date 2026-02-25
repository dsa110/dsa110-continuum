# CASA import moved to function level to prevent logs in workspace root
# See: docs/dev-notes/analysis/casa_log_handling_investigation.md
import logging
import multiprocessing
import os
import shutil
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from dsa110_contimg.core.calibration.casa_service import CASAService

# Ensure CASAPATH is set before importing CASA modules
from dsa110_contimg.common.utils.casa_init import ensure_casa_path
from dsa110_contimg.common.utils.error_context import format_ms_error_with_suggestions
from dsa110_contimg.common.utils import get_env_path

# Import CASA log environment for proper log redirection
try:
    from dsa110_contimg.common.utils.casa_init import casa_log_environment
except ImportError:
    # Fallback no-op context manager if tempdirs unavailable
    @contextmanager
    def casa_log_environment():
        yield None


ensure_casa_path()


# Ensure headless operation to prevent casaplotserver X server errors
# Set multiple environment variables to prevent CASA from launching plotting servers
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("CASA_NO_X", "1")  # Additional CASA-specific flag
if os.environ.get("DISPLAY"):
    os.environ.pop("DISPLAY", None)


@contextmanager
def suppress_subprocess_stderr():
    """Context manager to suppress stderr from subprocesses (like casaplotserver).

    Redirects stderr at the file descriptor level to suppress casaplotserver errors.
    Note: This only suppresses output to stderr; CASA operations still complete normally.

    """
    devnull_fd = None
    old_stderr = None
    old_stderr_fd = None
    try:
        old_stderr_fd = sys.stderr.fileno()
        # Save original stderr
        old_stderr = os.dup(old_stderr_fd)
        # Open devnull and redirect stderr to it
        devnull_fd = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull_fd, old_stderr_fd)
        yield
    except (AttributeError, OSError):
        # Fallback if fd manipulation fails (e.g., in tests or non-standard environments)
        yield
    finally:
        # Restore original stderr
        if old_stderr is not None and old_stderr_fd is not None:
            try:
                os.dup2(old_stderr, old_stderr_fd)
                os.close(old_stderr)
            except OSError:
                pass
        if devnull_fd is not None:
            try:
                os.close(devnull_fd)
            except OSError:
                pass


class PreflightError(Exception):
    """

    Raises
    ------
    FAIL
        FAST
    tool
        is missing or misconfigured
    to
        proceed and fail cryptically later

    """

    def __init__(self, tool: str, reason: str, suggestions: list[str]):
        self.tool = tool
        self.reason = reason
        self.suggestions = suggestions
        super().__init__(self._format_message())

    def _format_message(self) -> str:
        msg = f"Preflight check failed for {self.tool}: {self.reason}"
        msg += "\n\nRequired action:\n"
        msg += "\n".join(f"  - {s}" for s in self.suggestions)
        return msg


def preflight_check_aoflagger(prefer_docker: bool = False) -> dict[str, str]:
    """Verify AOFlagger is available and working BEFORE starting calibration.

    FAIL-FAST: Raises PreflightError with actionable diagnostics if AOFlagger
    is not available, rather than failing cryptically during flagging.

    Parameters
    ----------
    prefer_docker :
        If True, check Docker AOFlagger first (useful for Ubuntu 18.x)

    Returns
    -------
        Dict with 'method' ('docker' or 'native'), 'version', and 'command' info

    Raises
    ------
    PreflightError
        If AOFlagger is not available with actionable suggestions

    """
    logger = logging.getLogger(__name__)

    docker_cmd = shutil.which("docker")
    native_aoflagger = shutil.which("aoflagger")

    # Try Docker first if preferred and available
    if prefer_docker and docker_cmd:
        try:
            result = subprocess.run(
                [docker_cmd, "run", "--rm", "aoflagger:latest", "aoflagger", "--version"],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode == 0:
                version = result.stdout.strip() or result.stderr.strip()
                logger.info(f"AOFlagger preflight OK: Docker ({version})")
                return {
                    "method": "docker",
                    "version": version,
                    "command": f"{docker_cmd} run --rm aoflagger:latest aoflagger",
                }
        except subprocess.TimeoutExpired:
            logger.warning("Docker AOFlagger check timed out - Docker may be slow")
        except subprocess.SubprocessError as e:
            logger.warning(f"Docker AOFlagger check failed: {e}")

    # Try native AOFlagger
    if native_aoflagger:
        try:
            result = subprocess.run(
                [native_aoflagger, "--version"], capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0:
                version = result.stdout.strip() or result.stderr.strip()
                logger.info(f"AOFlagger preflight OK: native ({version})")
                return {"method": "native", "version": version, "command": native_aoflagger}
        except subprocess.SubprocessError as e:
            logger.warning(f"Native AOFlagger check failed: {e}")

    # AOFlagger not available - FAIL FAST with actionable diagnostics
    suggestions = []
    if docker_cmd:
        suggestions.extend(
            [
                "Build the AOFlagger Docker image: docker build -t aoflagger:latest .",
                "Check Docker image exists: docker images | grep aoflagger",
                "Pull from registry if available: docker pull <registry>/aoflagger:latest",
            ]
        )
    else:
        suggestions.extend(
            ["Install Docker (recommended for Ubuntu 18.x)", "Run: sudo apt-get install docker.io"]
        )

    suggestions.extend(
        [
            "Or install native AOFlagger: sudo apt-get install aoflagger",
            "Verify AOFlagger works: aoflagger --version",
        ]
    )

    raise PreflightError(
        tool="AOFlagger",
        reason="No working AOFlagger found (Docker or native)",
        suggestions=suggestions,
    )


def preflight_check_all(
    require_wsclean: bool = False, check_docker_mounts: bool = True
) -> dict[str, Any]:
    """Run all preflight checks before starting calibration pipeline.

    FAIL-FAST: Raises PreflightError on first missing *required* tool.

    Parameters
    ----------
    require_wsclean :
        If True, fail if wsclean not found.
        If False (default), just warn.
    check_docker_mounts :
        If True, also verify Docker volume mounts work

    Returns
    -------
        Dict of tool name -> preflight result info

    Raises
    ------
    PreflightError
        If any required tool is missing

    """
    logger = logging.getLogger(__name__)
    results = {}

    # Required: AOFlagger and CASA
    results["aoflagger"] = preflight_check_aoflagger()
    results["casa"] = preflight_check_casa()

    # Optional Docker mount check
    if check_docker_mounts and results["aoflagger"].get("method") == "docker":
        try:
            results["docker_mounts"] = preflight_check_aoflagger_docker_mounts()
        except PreflightError as e:
            logger.warning(f"Docker mount check failed: {e.reason}")
            results["docker_mounts"] = {"error": e.reason}

    # wsclean - required for imaging, optional for calibration-only
    try:
        results["wsclean"] = preflight_check_wsclean()
    except PreflightError as e:
        if require_wsclean:
            raise
        else:
            logger.warning(f"wsclean not found (imaging will fail): {e.reason}")
            results["wsclean"] = {"error": e.reason, "available": False}

    # Memory check (warning only, never fails)
    results["memory"] = preflight_check_memory()

    return results


def preflight_check_aoflagger_docker_mounts(
    test_paths: list[str] | None = None,
) -> dict[str, bool]:
    """Verify Docker volume mounts work for AOFlagger.

    FAIL-FAST: Verifies that the Docker container can actually see
    the required paths, not just that AOFlagger runs.

    Parameters
    ----------
    test_paths : Optional[List[str]], optional
        List of paths to verify are accessible in container.

    Returns
    -------
    dict
        Mapping of path to accessibility status (True/False)

    Raises
    ------
    PreflightError
        If critical paths are not accessible
    """
    logger = logging.getLogger(__name__)

    if test_paths is None:
        test_paths = ["/data", "/stage", "/dev/shm/dsa110-contimg"]

    docker_cmd = shutil.which("docker")
    if not docker_cmd:
        raise PreflightError(
            tool="Docker",
            reason="Docker not found",
            suggestions=["Install Docker: sudo apt-get install docker.io"],
        )

    results = {}
    failed_paths = []

    for path in test_paths:
        if not os.path.exists(path):
            results[path] = False
            logger.warning(f"Path {path} does not exist on host")
            continue

        try:
            # Test if Docker can see and list the path
            result = subprocess.run(
                [
                    docker_cmd,
                    "run",
                    "--rm",
                    "-v",
                    f"{path}:{path}:ro",
                    "aoflagger:latest",
                    "ls",
                    "-la",
                    path,
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode == 0:
                results[path] = True
                logger.debug(f"Docker mount OK: {path}")
            else:
                results[path] = False
                failed_paths.append(path)
                logger.warning(f"Docker cannot access {path}: {result.stderr}")
        except subprocess.TimeoutExpired:
            results[path] = False
            failed_paths.append(path)
            logger.warning(f"Docker mount test timed out for {path}")
        except subprocess.SubprocessError as e:
            results[path] = False
            failed_paths.append(path)
            logger.warning(f"Docker mount test failed for {path}: {e}")

    if failed_paths:
        raise PreflightError(
            tool="Docker volume mounts",
            reason=f"AOFlagger container cannot access paths: {failed_paths}",
            suggestions=[
                f"Verify paths exist: ls -la {' '.join(failed_paths)}",
                "Check Docker has permission to mount these paths",
                "Ensure no SELinux/AppArmor restrictions",
                "Try: docker run --rm -v /data:/data aoflagger:latest ls /data",
            ],
        )

    logger.info(f"Docker mount preflight OK: {list(results.keys())}")
    return results


def preflight_check_casa() -> dict[str, str]:
    """Verify CASA (casatasks/casacore) is available.

    FAIL-FAST: Raises PreflightError if CASA cannot be imported.

    Returns
    -------
        Dict with 'casatasks_version' and 'casacore_version'

    Raises
    ------
    PreflightError
        If CASA is not available

    """
    logger = logging.getLogger(__name__)
    result = {}
    errors = []

    # Check casacore
    try:
        import casacore.tables

        result["casacore_version"] = getattr(casacore, "__version__", "unknown")
        logger.debug(f"casacore OK: {result['casacore_version']}")
    except ImportError as e:
        errors.append(f"casacore: {e}")

    # Check CASA availability via service
    try:
        from dsa110_contimg.core.calibration.casa_service import CASAService

        service = CASAService()
        version = service.get_version()
        if version:
            result["casatasks_version"] = version
            logger.debug(f"CASA OK: {version}")
        else:
            errors.append("CASA version could not be determined")
    except ImportError as e:
        errors.append(f"CASA service unavailable: {e}")

    if errors:
        raise PreflightError(
            tool="CASA",
            reason=f"CASA import failed: {'; '.join(errors)}",
            suggestions=[
                "Activate CASA environment: conda activate casa6",
                "Install casatools: pip install casatools casatasks",
                "Install casacore: conda install -c conda-forge python-casacore",
                "Check Python environment has CASA packages",
            ],
        )

    logger.info(f"CASA preflight OK: casatasks={result.get('casatasks_version')}")
    return result


def preflight_check_wsclean() -> dict[str, str]:
    """Verify wsclean is available for imaging.

    FAIL-FAST: Raises PreflightError if wsclean is not found.

    Returns
    -------
        Dict with 'version' and 'path'

    Raises
    ------
    PreflightError
        If wsclean is not available

    """
    logger = logging.getLogger(__name__)

    wsclean_path = shutil.which("wsclean")

    if wsclean_path:
        try:
            result = subprocess.run(
                [wsclean_path, "--version"], capture_output=True, text=True, timeout=10
            )
            # wsclean outputs version to stderr
            version = result.stderr.strip() or result.stdout.strip()
            # Extract version number (first line usually contains it)
            version_line = version.split("\n")[0] if version else "unknown"
            logger.info(f"wsclean preflight OK: {version_line}")
            return {"version": version_line, "path": wsclean_path}
        except subprocess.SubprocessError as e:
            logger.warning(f"wsclean version check failed: {e}")

    # wsclean not found - check for Docker alternative
    docker_cmd = shutil.which("docker")
    if docker_cmd:
        try:
            result = subprocess.run(
                [docker_cmd, "run", "--rm", "wsclean:latest", "wsclean", "--version"],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode == 0:
                version = result.stderr.strip() or result.stdout.strip()
                version_line = version.split("\n")[0] if version else "unknown"
                logger.info(f"wsclean preflight OK (Docker): {version_line}")
                return {
                    "version": version_line,
                    "path": "docker:wsclean:latest",
                    "method": "docker",
                }
        except subprocess.SubprocessError:
            pass

    raise PreflightError(
        tool="wsclean",
        reason="wsclean not found (native or Docker)",
        suggestions=[
            "Install wsclean: sudo apt-get install wsclean",
            "Or use Docker: docker pull wsclean:latest",
            "Build from source: https://gitlab.com/aroffringa/wsclean",
            "Verify installation: wsclean --version",
        ],
    )


def preflight_check_disk_space(
    path: str, required_gb: float = 50.0, warn_gb: float = 100.0
) -> dict[str, float]:
    """Check available disk space before expensive operations.

    FAIL-FAST: Raises PreflightError if disk space is critically low.

    Parameters
    ----------
    path :
        Path to check (will check the filesystem containing this path)
    required_gb :
        Minimum required space in GB (fails if below)
    warn_gb :
        Warning threshold in GB (logs warning if below)

    Returns
    -------
        Dict with 'available_gb', 'total_gb', 'used_percent'

    Raises
    ------
    PreflightError
        If available space < required_gb

    """
    logger = logging.getLogger(__name__)

    try:
        stat = os.statvfs(path)
        available_bytes = stat.f_bavail * stat.f_frsize
        total_bytes = stat.f_blocks * stat.f_frsize
        available_gb = available_bytes / (1024**3)
        total_gb = total_bytes / (1024**3)
        used_percent = 100 * (1 - available_bytes / total_bytes)

        result = {
            "available_gb": round(available_gb, 1),
            "total_gb": round(total_gb, 1),
            "used_percent": round(used_percent, 1),
            "path": path,
        }

        if available_gb < required_gb:
            raise PreflightError(
                tool="Disk space",
                reason=f"Only {available_gb:.1f} GB available on {path} (need {required_gb:.1f} GB)",
                suggestions=[
                    f"Free up space on {path}",
                    "Delete old MS files or intermediate products",
                    "Use a different output directory with more space",
                    f"Current usage: {used_percent:.1f}% of {total_gb:.1f} GB",
                ],
            )

        if available_gb < warn_gb:
            logger.warning(
                f"Low disk space: {available_gb:.1f} GB available on {path} "
                f"(warning threshold: {warn_gb:.1f} GB)"
            )
        else:
            logger.debug(f"Disk space OK: {available_gb:.1f} GB available on {path}")

        return result

    except OSError as e:
        raise PreflightError(
            tool="Disk space",
            reason=f"Cannot check disk space for {path}: {e}",
            suggestions=[
                f"Verify path exists: ls -la {path}",
                "Check filesystem is mounted",
                "Check permissions",
            ],
        )


def preflight_check_output_dir(output_dir: str) -> dict[str, Any]:
    """Verify output directory exists and is writable.

    FAIL-FAST: Raises PreflightError if output cannot be written.

    Parameters
    ----------
    output_dir :
        Path to output directory

    Returns
    -------
        Dict with 'path', 'writable', 'disk_info'

    Raises
    ------
    PreflightError
        If directory is not writable

    """
    logger = logging.getLogger(__name__)
    output_path = Path(output_dir)

    # Create if doesn't exist
    try:
        output_path.mkdir(parents=True, exist_ok=True)
    except (PermissionError, OSError) as e:
        raise PreflightError(
            tool="Output directory",
            reason=f"Cannot create output directory {output_dir}: {e}",
            suggestions=[
                f"Check parent directory permissions: ls -la {output_path.parent}",
                "Create directory manually: mkdir -p {output_dir}",
                "Use a different output directory",
            ],
        )

    # Test write permission
    test_file = output_path / ".write_test"
    try:
        test_file.touch()
        test_file.unlink()
    except (PermissionError, OSError) as e:
        raise PreflightError(
            tool="Output directory",
            reason=f"Output directory {output_dir} is not writable: {e}",
            suggestions=[
                f"Check permissions: ls -la {output_dir}",
                f"Fix permissions: chmod u+w {output_dir}",
                "Use a different output directory",
            ],
        )

    # Also check disk space on output directory
    disk_info = preflight_check_disk_space(output_dir, required_gb=20.0, warn_gb=50.0)

    logger.info(
        f"Output directory preflight OK: {output_dir} ({disk_info['available_gb']} GB available)"
    )
    return {"path": output_dir, "writable": True, "disk_info": disk_info}


def preflight_check_memory(required_gb: float = 8.0, warn_gb: float = 16.0) -> dict[str, float]:
    """Check available system memory.

    Logs warning if memory is low but doesn't fail (OOM will happen at runtime).

    Parameters
    ----------
    required_gb :
        Minimum required memory in GB (logs error if below)
    warn_gb :
        Warning threshold in GB (logs warning if below)

    Returns
    -------
        Dict with 'available_gb', 'total_gb', 'used_percent'

    """
    logger = logging.getLogger(__name__)

    try:
        with open("/proc/meminfo") as f:
            meminfo = {}
            for line in f:
                parts = line.split()
                if len(parts) >= 2:
                    key = parts[0].rstrip(":")
                    value_kb = int(parts[1])
                    meminfo[key] = value_kb

        total_gb = meminfo.get("MemTotal", 0) / (1024**2)
        available_gb = meminfo.get("MemAvailable", meminfo.get("MemFree", 0)) / (1024**2)
        used_percent = 100 * (1 - available_gb / total_gb) if total_gb > 0 else 0

        result = {
            "available_gb": round(available_gb, 1),
            "total_gb": round(total_gb, 1),
            "used_percent": round(used_percent, 1),
        }

        if available_gb < required_gb:
            logger.error(
                f"CRITICAL: Only {available_gb:.1f} GB memory available "
                f"(need {required_gb:.1f} GB). Pipeline may OOM!"
            )
        elif available_gb < warn_gb:
            logger.warning(
                f"Low memory: {available_gb:.1f} GB available (warning threshold: {warn_gb:.1f} GB)"
            )
        else:
            logger.debug(f"Memory OK: {available_gb:.1f} GB available")

        return result

    except (OSError, ValueError, KeyError) as e:
        logger.warning(f"Cannot check memory: {e}")
        return {"available_gb": -1, "total_gb": -1, "used_percent": -1, "error": str(e)}


def preflight_check_strategy_file(strategy_file: str | None) -> bool:
    """Verify AOFlagger strategy file exists if specified.

    FAIL-FAST: Raises PreflightError if strategy file doesn't exist.

    Parameters
    ----------
    strategy_file :
        Path to Lua strategy file, or None for auto-detect
    strategy_file: Optional[str] :


    Returns
    -------
        True if valid (or None for auto-detect)

    Raises
    ------
    PreflightError
        If strategy file specified but doesn't exist

    """
    if strategy_file is None:
        return True

    if not os.path.exists(strategy_file):
        raise PreflightError(
            tool="AOFlagger strategy",
            reason=f"Strategy file not found: {strategy_file}",
            suggestions=[
                f"Check file exists: ls -la {strategy_file}",
                "Use strategy_file=None for AOFlagger auto-detection",
                "Create/download the strategy file",
                "Check path is absolute and correct",
            ],
        )

    logger = logging.getLogger(__name__)
    logger.debug(f"Strategy file OK: {strategy_file}")
    return True


def reset_flags(ms: str) -> None:
    service = CASAService()
    service.flagdata(vis=ms, mode="unflag")


def flag_zeros(ms: str, datacolumn: str = "data") -> None:
    service = CASAService()
    service.flagdata(vis=ms, mode="clip", datacolumn=datacolumn, clipzeros=True)


def flag_autocorrelations(ms: str, datacolumn: str = "data") -> None:
    """Flag autocorrelation baselines (antenna with itself).

    Autocorrelations (ant1 == ant2) contain Tsys information but are not
    useful for interferometric calibration and imaging. Flagging them
    reduces data volume and prevents them from contaminating solutions.

    Parameters
    ----------
    ms :
        Path to Measurement Set
    datacolumn :
        Data column to use (default: "data")
    """
    service = CASAService()
    # CASA flagdata with autocorr=True flags only autocorrelations
    service.flagdata(vis=ms, mode="manual", autocorr=True, datacolumn=datacolumn)


def flag_clip_amplitude(
    ms_path: str,
    threshold_max: float = 0.5,
    threshold_min: float | None = None,
    datacolumn: str = "data",
) -> dict:
    """Flag data with amplitude outside a valid range to remove RFI and bad data.

        Uses CASA's flagdata with mode='clip' to identify and flag data points
        where the amplitude exceeds the maximum threshold or falls below the
        minimum threshold. This is effective for removing:
        - Strong RFI that can corrupt bandpass calibration (high amplitude)
        - Bad/noisy data that won't contribute to solutions (low amplitude)

        For DSA-110 data, typical amplitudes are ~0.05, while RFI can reach
        amplitudes of 100+. A max threshold of 0.5 typically catches 1-2% of data
        that is RFI-contaminated.

    Parameters
    ----------
    ms_path : str
        Path to Measurement Set
    threshold_max : float, optional
        Maximum amplitude threshold; data above this is flagged.
    Default: 0.5 (about 10x typical amplitude, catches strong RFI)
    threshold_min : Optional[float], optional
        Minimum amplitude threshold; data below this is flagged.
    Default: None (no minimum clipping). Set to e.g. 0.001 to remove
        very low amplitude data that may be noise or bad correlator output.
    datacolumn : str, optional
        Data column to check. Default: "data"

    Returns
    -------
        dict
        Dictionary with keys:
        - threshold_max: The max threshold used
        - threshold_min: The min threshold used (or None)
        - flagged_fraction: Fraction of data newly flagged by clipping
        - total_flagged_after: Total flag fraction after clipping
    """
    # Use CASAService for lazy import and environment protection
    service = CASAService()

    logger = logging.getLogger(__name__)

    min_val = threshold_min if threshold_min is not None else 0.0
    logger.info(f"Clipping amplitudes outside [{min_val}, {threshold_max}] in {ms_path}")

    # Get flag fraction before
    stats_before = flag_summary(ms_path)
    frac_before = stats_before.get("total_fraction_flagged", 0.0)

    # Use flagdata with mode='clip' to flag data outside valid range
    # clipminmax=[min, max] with clipoutside=True flags anything outside that range
    service.flagdata(
        vis=ms_path,
        mode="clip",
        datacolumn=datacolumn,
        clipminmax=[min_val, threshold_max],
        clipoutside=True,  # Flag data OUTSIDE the range [min, max]
        action="apply",
    )

    # Get flag fraction after
    stats_after = flag_summary(ms_path)
    frac_after = stats_after.get("total_fraction_flagged", 0.0)
    flagged_fraction = frac_after - frac_before

    logger.info(f"Amplitude clipping complete: {flagged_fraction * 100:.2f}% newly flagged")

    return {
        "threshold_max": threshold_max,
        "threshold_min": threshold_min,
        "flagged_fraction": flagged_fraction,
        "total_flagged_after": frac_after,
    }


def detect_and_flag_bad_polarizations(
    ms_path: str,
    snr_ratio_threshold: float = 5.0,
    min_good_snr: float = 10.0,
    phase_table: str | None = None,
    dry_run: bool = False,
) -> dict:
    """Detect and flag antennas with single-polarization failures.

    Some antennas may have one polarization (XX or YY) that is decorrelated or
    has hardware issues, causing very low SNR for that polarization during
    calibration. This function identifies such antennas using two methods:

    1. **Primary (if phase_table provided)**: Analyze SNR in a pre-bandpass phase
       calibration table. This is the most reliable method since it directly
       measures which polarizations failed during calibration.

    2. **Fallback (MS-only analysis)**: Compute phase coherence from raw visibilities.
       Less reliable but works without a calibration table.

    Parameters
    ----------
    ms_path :
        Path to Measurement Set
    snr_ratio_threshold :
        If one polarization's SNR is this many times
        lower than the other in the phase table, flag it (default: 5.0)
    min_good_snr :
        Minimum SNR for a "good" polarization (default: 10.0)
    phase_table :
        Path to pre-bandpass phase calibration table. If provided,
        analyze this table directly (most reliable). Otherwise, analyze
        the MS data using coherence metrics.
    dry_run :
        If True, only report statistics without flagging (default: False)

    Returns
    -------
    Dictionary with
        - 'bad_polarizations': List of (antenna_id, pol_idx, pol_name) tuples
        - 'antenna_stats': Dict mapping antenna ID to per-pol statistics
        - 'n_antennas_affected': Number of antennas with one bad polarization
        - 'total_flagged_before': Overall flagged fraction before action
        - 'total_flagged_after': Overall flagged fraction after action
        - 'action_taken': Whether any flags were applied
        - 'detection_method': 'phase_table' or 'ms_coherence'

    """
    import logging
    import os

    logger = logging.getLogger(__name__)

    result = {
        "bad_polarizations": [],
        "antenna_stats": {},
        "n_antennas_affected": 0,
        "total_flagged_before": 0.0,
        "total_flagged_after": 0.0,
        "action_taken": False,
        "detection_method": "unknown",
    }

    # Method 1: Analyze phase calibration table if provided
    if phase_table and os.path.exists(phase_table):
        logger.info(f"Analyzing phase table: {phase_table}")
        result["detection_method"] = "phase_table"
        bad_polarizations = _detect_bad_pols_from_caltable(
            phase_table, snr_ratio_threshold, min_good_snr, result
        )
    else:
        # Method 2: Analyze MS coherence (less reliable)
        if phase_table:
            logger.warning(f"Phase table not found: {phase_table}, falling back to MS analysis")
        logger.info("Using MS coherence analysis for polarization detection")
        result["detection_method"] = "ms_coherence"
        bad_polarizations = _detect_bad_pols_from_ms_coherence(
            ms_path, snr_ratio_threshold, min_good_snr, result
        )

    result["bad_polarizations"] = bad_polarizations
    result["n_antennas_affected"] = len(bad_polarizations)

    # Log findings
    if bad_polarizations:
        xx_bad = [a for a, p, n in bad_polarizations if p == 0]
        yy_bad = [a for a, p, n in bad_polarizations if p == 1]
        if xx_bad:
            logger.info(f"Detected {len(xx_bad)} antennas with bad XX polarization: {xx_bad}")
        if yy_bad:
            logger.info(f"Detected {len(yy_bad)} antennas with bad YY polarization: {yy_bad}")
    else:
        logger.info("No single-polarization failures detected")

    # Get initial flag fraction
    stats_before = flag_summary(ms_path)
    result["total_flagged_before"] = stats_before.get("total_fraction_flagged", 0.0)

    # Flag bad polarizations if not dry_run
    if bad_polarizations and not dry_run:
        service = CASAService()
        with suppress_subprocess_stderr():
            flagged_count = 0
            skipped_count = 0
            for ant_id, pol_idx, pol_name in bad_polarizations:
                try:
                    logger.info(f"Flagging antenna {ant_id} polarization {pol_name}")
                    service.flagdata(
                        vis=ms_path,
                        mode="manual",
                        antenna=str(ant_id),
                        correlation=pol_name,
                        action="apply",
                    )
                    flagged_count += 1
                except RuntimeError as e:
                    # MSSelectionNullSelection means no unflagged data for this selection
                    # This can happen if the antenna is already completely flagged
                    if "NullSelection" in str(e) or "zero rows" in str(e):
                        logger.debug(
                            f"Skipping antenna {ant_id} {pol_name} - already flagged or no data"
                        )
                        skipped_count += 1
                    else:
                        raise

            if flagged_count > 0:
                result["action_taken"] = True
            if skipped_count > 0:
                logger.info(f"Skipped {skipped_count} antennas (already flagged)")

    # Get final flag fraction
    if not dry_run and bad_polarizations:
        stats_after = flag_summary(ms_path)
        result["total_flagged_after"] = stats_after.get("total_fraction_flagged", 0.0)
    else:
        result["total_flagged_after"] = result["total_flagged_before"]

    return result


def _detect_bad_pols_from_caltable(
    caltable: str,
    snr_ratio_threshold: float,
    min_good_snr: float,
    result: dict,
) -> list:
    """Detect bad polarizations by analyzing a calibration table.

    Examines the SNR and flag columns in a CASA calibration table (typically
    a G-type phase solution) to find antennas where one polarization has
    significantly lower SNR or is completely flagged.

    Parameters
    ----------
    caltable :
        Path to calibration table
    snr_ratio_threshold :
        Flag if SNR ratio exceeds this (one pol much better)
    min_good_snr :
        Minimum SNR for a "good" polarization
    result :
        Dictionary to store antenna stats

    Returns
    -------
        List of (antenna_id, pol_idx, pol_name) tuples for bad polarizations

    """
    import logging

    import numpy as np
    from casatools import table

    logger = logging.getLogger(__name__)
    bad_polarizations = []

    tb = table()
    try:
        tb.open(caltable)

        ant1 = tb.getcol("ANTENNA1")
        snr = tb.getcol("SNR")  # shape: (npol, nchan, nrow)
        flags = tb.getcol("FLAG")

        # Get unique antennas
        unique_ants = np.unique(ant1)
        antenna_stats = {}

        for ant_id in unique_ants:
            ant_mask = ant1 == ant_id
            if ant_mask.sum() == 0:
                continue

            ant_snr = snr[:, :, ant_mask]
            ant_flags = flags[:, :, ant_mask]

            pol_stats = []
            for pol in range(min(2, ant_snr.shape[0])):
                pol_snr = ant_snr[pol, :, :]
                pol_flags = ant_flags[pol, :, :]

                # Get unflagged SNR values
                unflagged_mask = ~pol_flags
                if unflagged_mask.sum() == 0:
                    pol_stats.append({"mean_snr": 0, "flag_frac": 1.0, "status": "all_flagged"})
                    continue

                mean_snr = float(pol_snr[unflagged_mask].mean())
                flag_frac = float(pol_flags.mean())

                pol_stats.append({"mean_snr": mean_snr, "flag_frac": flag_frac, "status": "ok"})

            antenna_stats[int(ant_id)] = pol_stats

            # Check for single-polarization failure
            if len(pol_stats) >= 2:
                snr0 = pol_stats[0]["mean_snr"]
                snr1 = pol_stats[1]["mean_snr"]
                flag0 = pol_stats[0]["flag_frac"]
                flag1 = pol_stats[1]["flag_frac"]

                # Primary check: One polarization completely flagged, other not
                # This is the clearest signal of single-pol failure - we don't require
                # the good polarization to have high SNR since DSA-110 often has low SNR
                # in the pre-bandpass phase solution
                if flag1 > 0.9 and flag0 < 0.5:
                    # YY is 100% flagged, XX is mostly unflagged
                    bad_polarizations.append((int(ant_id), 1, "YY"))
                    logger.debug(
                        f"Antenna {ant_id}: YY 100% flagged in cal table "
                        f"(XX flag={flag0 * 100:.0f}%, SNR={snr0:.1f})"
                    )
                elif flag0 > 0.9 and flag1 < 0.5:
                    # XX is 100% flagged, YY is mostly unflagged
                    bad_polarizations.append((int(ant_id), 0, "XX"))
                    logger.debug(
                        f"Antenna {ant_id}: XX 100% flagged in cal table "
                        f"(YY flag={flag1 * 100:.0f}%, SNR={snr1:.1f})"
                    )
                # Secondary check: SNR ratio for partially flagged cases
                elif snr0 > 0 and snr1 > 0:
                    ratio = snr0 / snr1
                    if ratio > snr_ratio_threshold and snr1 < min_good_snr:
                        bad_polarizations.append((int(ant_id), 1, "YY"))
                        logger.debug(
                            f"Antenna {ant_id}: YY low SNR={snr1:.1f} vs XX SNR={snr0:.1f}"
                        )
                    elif ratio < 1 / snr_ratio_threshold and snr0 < min_good_snr:
                        bad_polarizations.append((int(ant_id), 0, "XX"))
                        logger.debug(
                            f"Antenna {ant_id}: XX low SNR={snr0:.1f} vs YY SNR={snr1:.1f}"
                        )

        result["antenna_stats"] = antenna_stats

    finally:
        tb.close()

    return bad_polarizations


def _detect_bad_pols_from_ms_coherence(
    ms_path: str,
    snr_ratio_threshold: float,
    min_good_snr: float,
    result: dict,
) -> list:
    """Detect bad polarizations from MS coherence analysis.

    Less reliable than calibration table analysis, but useful when no
    calibration table is available. Uses coherence ratio between polarizations.

    Parameters
    ----------
    ms_path :
        Path to Measurement Set
    snr_ratio_threshold :
        Used for coherence ratio threshold (2.5x default)
    min_good_snr :
        Minimum SNR for a "good" polarization
    result :
        Dictionary to store antenna stats

    Returns
    -------
        List of (antenna_id, pol_idx, pol_name) tuples for bad polarizations

    """
    import logging

    import casacore.tables as casatables
    import numpy as np

    logger = logging.getLogger(__name__)
    bad_polarizations = []

    # Use looser coherence ratio threshold - this detection is less reliable
    coherence_ratio_threshold = 2.0  # One pol 2x less coherent

    try:
        with casatables.table(ms_path, readonly=True) as tb:
            n_rows = tb.nrows()
            if n_rows == 0:
                logger.warning(f"MS {ms_path} has no rows")
                return bad_polarizations

            ant1 = tb.getcol("ANTENNA1")
            ant2 = tb.getcol("ANTENNA2")
            data = tb.getcol("DATA")  # shape: (n_rows, n_chan, n_pol)
            flags = tb.getcol("FLAG")

            # Get number of polarizations
            n_pol = data.shape[2]
            if n_pol < 2:
                logger.warning("MS has only one polarization, skipping bad pol detection")
                return bad_polarizations

            # Get unique antenna IDs (excluding autocorrelations)
            cross_mask = ant1 != ant2
            all_antennas = np.unique(np.concatenate([ant1[cross_mask], ant2[cross_mask]]))

            # Compute per-antenna, per-polarization amplitude and coherence statistics
            antenna_stats = {}

            for ant_id in all_antennas:
                # Get all baselines involving this antenna
                mask = ((ant1 == ant_id) | (ant2 == ant_id)) & cross_mask
                ant_data = data[mask]
                ant_flags = flags[mask]

                if ant_data.size == 0:
                    continue

                # Compute coherence for each polarization
                pol_stats = []
                for pol in range(n_pol):
                    pol_data = ant_data[:, :, pol]
                    pol_flags = ant_flags[:, :, pol]

                    # Use only unflagged data
                    valid_mask = ~pol_flags
                    if valid_mask.sum() == 0:
                        pol_stats.append({"coherence": 0, "n_valid": 0})
                        continue

                    # Compute phase coherence: ratio of vector average to scalar average
                    complex_data = pol_data[valid_mask]
                    vector_avg = np.abs(np.mean(complex_data))
                    scalar_avg = np.mean(np.abs(complex_data))
                    coherence = vector_avg / scalar_avg if scalar_avg > 0 else 0

                    pol_stats.append(
                        {
                            "coherence": float(coherence),
                            "n_valid": int(valid_mask.sum()),
                        }
                    )

                antenna_stats[int(ant_id)] = pol_stats

                # Check for single-polarization coherence issue
                if len(pol_stats) >= 2:
                    coh0 = pol_stats[0].get("coherence", 1.0)
                    coh1 = pol_stats[1].get("coherence", 1.0)

                    if coh0 > 0 and coh1 > 0:
                        coh_ratio = coh0 / coh1
                        if coh_ratio > coherence_ratio_threshold:
                            # Pol 1 (YY) is decorrelated
                            bad_polarizations.append((int(ant_id), 1, "YY"))
                            logger.debug(
                                f"Antenna {ant_id}: YY polarization decorrelated "
                                f"(coherence={coh1:.3f} vs XX coherence={coh0:.3f})"
                            )
                        elif coh_ratio < 1 / coherence_ratio_threshold:
                            # Pol 0 (XX) is decorrelated
                            bad_polarizations.append((int(ant_id), 0, "XX"))
                            logger.debug(
                                f"Antenna {ant_id}: XX polarization decorrelated "
                                f"(coherence={coh0:.3f} vs YY coherence={coh1:.3f})"
                            )

            result["antenna_stats"] = antenna_stats

    except Exception as e:
        logger.error(f"Error in MS coherence analysis: {e}", exc_info=True)
        raise

    return bad_polarizations


def flag_residual_rfi_clip(
    ms: str,
    datacolumn: str = "data",
    sigma: float = 7.0,
    *,
    per_channel: bool = False,
) -> dict[str, Any]:
    """Flag residual RFI via MAD-based amplitude clipping on cross-correlations.

    AOFlagger's SumThreshold algorithm requires extended time–frequency structure
    to detect RFI.  DSA-110 drift-scan observations have only 24 time samples,
    leaving a significant residual (kurtosis >> 3).  This function applies a
    robust sigma-clip on the *unflagged* visibility amplitudes *after* AOFlagger
    has already run, catching the remaining broadband and short-baseline RFI that
    SumThreshold misses.

    Algorithm (per polarisation):
        1. Select cross-correlation rows only (``ANTENNA1 != ANTENNA2``).
        2. Compute amplitudes of unflagged visibilities in the selected data column.
        3. Estimate the centre and scale using the median and MAD
           (``σ_MAD = 1.4826 × median(|x − median(x)|)``).
        4. Flag any visibility whose amplitude exceeds ``median + sigma × σ_MAD``.

    When ``per_channel=False`` (default, recommended), a single threshold is
    computed from *all* unflagged cross-correlation amplitudes in each
    polarisation.  This corresponds to the "global 7σ clip" strategy validated
    on DSA-110 data (kurtosis 535 → 2.8 with only +1.2 % data cost).

    When ``per_channel=True``, the threshold is computed independently for each
    frequency channel, which is more sensitive to narrow-band RFI at the expense
    of a slightly higher flag fraction.

    Parameters
    ----------
    ms :
        Path to Measurement Set.
    datacolumn :
        Column to read visibilities from (default ``"data"``).
    sigma :
        Clipping threshold in MAD-based σ units (default 7.0).
    per_channel :
        If *True*, compute a per-channel threshold instead of a global one.

    Returns
    -------
    dict
        Summary with keys ``"new_flags"``, ``"total_flagged_pct"``,
        ``"pre_clip_flagged_pct"``, and ``"threshold"`` (or ``"thresholds"``).

    Raises
    ------
    FileNotFoundError
        If *ms* does not exist.

    """
    import numpy as np

    logger = logging.getLogger(__name__)

    ms_path = Path(ms)
    if not ms_path.exists():
        raise FileNotFoundError(f"Measurement Set not found: {ms}")

    try:
        import casacore.tables as ct
    except ImportError as exc:
        raise ImportError(
            "casacore.tables is required for flag_residual_rfi_clip"
        ) from exc

    MAD_TO_SIGMA = 1.4826  # MAD → Gaussian-σ conversion factor

    with ct.table(str(ms), readonly=False) as t:
        data = t.getcol(datacolumn.upper())     # (nrow, nchan, npol)
        flags = t.getcol("FLAG")                # (nrow, nchan, npol)
        ant1 = t.getcol("ANTENNA1")
        ant2 = t.getcol("ANTENNA2")

        cross = ant1 != ant2
        _nrow, nchan, npol = data.shape

        cross_data = data[cross]                # (n_cross, nchan, npol)
        cross_flags = flags[cross].copy()       # writable copy
        total_new = 0

        threshold_info: dict[str, Any] = {}

        for pol in range(npol):
            amp = np.abs(cross_data[:, :, pol])          # (n_cross, nchan)
            fl = cross_flags[:, :, pol]                   # (n_cross, nchan)

            if per_channel:
                # Per-channel thresholds
                thresholds = np.empty(nchan, dtype=np.float64)
                for ch in range(nchan):
                    unflagged = amp[:, ch][~fl[:, ch]]
                    if unflagged.size < 10:
                        thresholds[ch] = np.inf
                        continue
                    med = np.median(unflagged)
                    mad_sig = np.median(np.abs(unflagged - med)) * MAD_TO_SIGMA
                    thresholds[ch] = med + sigma * mad_sig

                new = (amp > thresholds[np.newaxis, :]) & ~fl
                threshold_info[f"pol{pol}_thresholds_median"] = float(np.median(thresholds[np.isfinite(thresholds)]))
            else:
                # Global threshold across all channels
                unflagged = amp[~fl]
                if unflagged.size < 10:
                    logger.warning("Pol %d: too few unflagged visibilities for sigma-clip", pol)
                    continue
                med = float(np.median(unflagged))
                mad_sig = float(np.median(np.abs(unflagged - med))) * MAD_TO_SIGMA
                thresh = med + sigma * mad_sig
                threshold_info[f"pol{pol}_threshold"] = thresh
                threshold_info[f"pol{pol}_median"] = med
                threshold_info[f"pol{pol}_mad_sigma"] = mad_sig

                new = (amp > thresh) & ~fl

            n_new = int(new.sum())
            total_new += n_new
            cross_flags[:, :, pol] |= new
            logger.info(
                "Post-AOFlagger %s%.0fσ clip pol %d: %d new flags (%.3f%%)",
                "per-ch " if per_channel else "",
                sigma,
                pol,
                n_new,
                100.0 * n_new / amp.size if amp.size > 0 else 0.0,
            )

        # Write back only the cross-correlation flag rows
        full_flags = flags.copy()
        full_flags[cross] = cross_flags
        t.putcol("FLAG", full_flags)

    pre_pct = 100.0 * flags[cross].sum() / flags[cross].size
    post_pct = 100.0 * full_flags[cross].sum() / full_flags[cross].size

    result = {
        "new_flags": total_new,
        "pre_clip_flagged_pct": round(pre_pct, 4),
        "total_flagged_pct": round(post_pct, 4),
        **threshold_info,
    }
    logger.info(
        "Post-AOFlagger sigma-clip: %d new flags, %.2f%% → %.2f%% cross-corr flagged",
        total_new,
        pre_pct,
        post_pct,
    )
    return result


def flag_rfi(
    ms: str,
    datacolumn: str = "data",
    backend: str = "aoflagger",
    aoflagger_path: str | None = None,
    strategy: str | None = None,
    extend_flags: bool = True,
    clip_residual: bool = True,
    clip_sigma: float = 7.0,
) -> None:
    """Flag RFI using CASA or AOFlagger, with optional post-clip.

    When *backend* is ``"aoflagger"`` and *clip_residual* is *True* (the
    default), a MAD-based amplitude sigma-clip is applied after AOFlagger
    to catch residual RFI that SumThreshold misses on short time-axis data.

    Parameters
    ----------
    ms :
        Path to Measurement Set
    datacolumn :
        Data column to use (default: "data")
    backend :
        Backend to use - "aoflagger" (default) or "casa"
    aoflagger_path :
        Path to aoflagger executable or "docker" (for AOFlagger backend)
    strategy :
        Optional path to custom Lua strategy file (for AOFlagger backend)
    extend_flags :
        If True, extend flags to adjacent channels/times after flagging (default: True)
    clip_residual :
        If True, apply a post-AOFlagger MAD sigma-clip on cross-corr amplitudes
        (default: True).  Only used when *backend* is ``"aoflagger"``.
    clip_sigma :
        Threshold in MAD-σ units for the residual clip (default: 7.0).
    """
    from dsa110_contimg.common.utils.ms_permissions import ensure_ms_writable

    ensure_ms_writable(ms)
    if backend == "aoflagger":
        flag_rfi_aoflagger(
            ms, datacolumn=datacolumn, aoflagger_path=aoflagger_path, strategy=strategy
        )

        # Stage 2: MAD-based residual clip on cross-correlation amplitudes.
        # AOFlagger's SumThreshold needs extended time–frequency structure to
        # detect RFI.  DSA-110 drift-scan data has only 24 time samples, so a
        # post-AOFlagger sigma-clip catches the broadband and short-baseline RFI
        # that SumThreshold misses.  Validated: kurtosis 535 → 2.8, +1.2 % cost.
        if clip_residual:
            try:
                flag_residual_rfi_clip(ms, datacolumn=datacolumn, sigma=clip_sigma)
            except Exception:
                logging.getLogger(__name__).warning(
                    "Post-AOFlagger sigma-clip failed; AOFlagger flags are intact.",
                    exc_info=True,
                )

        # Extend flags after AOFlagger (if enabled)
        # Note: Flag extension may fail when using Docker due to permission issues
        # (AOFlagger writes as root, making subsequent writes fail). This is non-fatal.
        if extend_flags:
            time.sleep(2)  # Allow file locks to clear
            try:
                flag_extend(
                    ms,
                    flagnearfreq=True,
                    flagneartime=True,
                    extendpols=True,
                    datacolumn=datacolumn,
                )
                logger = logging.getLogger(__name__)
                logger.debug("Flag extension completed successfully")
            except (RuntimeError, PermissionError, OSError) as e:
                # If file lock or permission issue, log warning but don't fail
                logger = logging.getLogger(__name__)
                error_str = str(e).lower()
                if any(
                    term in error_str
                    for term in [
                        "cannot be opened",
                        "not writable",
                        "permission denied",
                        "permission",
                    ]
                ):
                    logger.warning(
                        f"Flag extension skipped due to file permission/lock issue (common when using Docker AOFlagger). "
                        f"RFI flags from AOFlagger are still applied. Error: {e}"
                    )
                else:
                    logger.warning(
                        f"Flag extension failed: {e}. RFI flags from AOFlagger are still applied."
                    )
    else:
        # Two-stage RFI flagging using flagdata modes (tfcrop then rflag)
        service = CASAService()
        with suppress_subprocess_stderr():
            service.flagdata(
                vis=ms,
                mode="tfcrop",
                datacolumn=datacolumn,
                timecutoff=4.0,
                freqcutoff=4.0,
                timefit="line",
                freqfit="poly",
                maxnpieces=5,
                winsize=3,
                extendflags=False,
            )

            service.flagdata(
                vis=ms,
                mode="rflag",
                datacolumn=datacolumn,
                timedevscale=4.0,
                freqdevscale=4.0,
                extendflags=False,
            )
        # Extend flags to adjacent channels/times after flagging (if enabled)
        if extend_flags:
            try:
                flag_extend(
                    ms,
                    flagnearfreq=True,
                    flagneartime=True,
                    extendpols=True,
                    datacolumn=datacolumn,
                )
            except RuntimeError as e:
                # If file lock or permission issue, log warning but don't fail
                logger = logging.getLogger(__name__)
                if "cannot be opened" in str(e) or "not writable" in str(e):
                    logger.warning(
                        f"Could not extend flags due to file lock/permission: {e}. Flags from tfcrop+rflag are still applied."
                    )
                else:
                    raise


def _get_default_aoflagger_strategy() -> str | None:
    """Get the default DSA-110 AOFlagger strategy file path.

    Returns
    -------
        Path to dsa110-default.lua if it exists, None otherwise

    """
    # Try multiple possible locations for the strategy file
    possible_paths = [
        get_env_path("CONTIMG_BASE_DIR", default="/data/dsa110-contimg")
        / "config/dsa110-default.lua",
        Path(__file__).parent.parent.parent.parent / "config" / "dsa110-default.lua",
        Path(os.getcwd()) / "config" / "dsa110-default.lua",
    ]

    for strategy_path in possible_paths:
        if strategy_path.exists():
            return str(strategy_path.resolve())

    return None


def flag_rfi_aoflagger(
    ms: str,
    datacolumn: str = "data",
    aoflagger_path: str | None = None,
    strategy: str | None = None,
) -> None:
    """Flag RFI using AOFlagger (faster alternative to CASA tfcrop).

        AOFlagger uses the SumThreshold algorithm which is typically 2-5x faster
        than CASA's tfcrop+rflag combination for large datasets.

        **Note:** On Ubuntu 18.x systems, Docker may be required due to CMake/pybind11
        compatibility issues. The default behavior is to prefer native and fall back to Docker.

    Parameters
    ----------
    ms :
        Path to Measurement Set
    datacolumn :
        Data column to use (default: "data")
    aoflagger_path :
        Path to aoflagger executable, "docker" to force Docker, or None to auto-detect
    strategy :
        Optional path to custom Lua strategy file. If None, AOFlagger auto-detects strategy.
        To force a default strategy globally, set CONTIMG_AOFLAGGER_STRATEGY to a strategy path.

    Raises
    ------
    RuntimeError
        If AOFlagger is not available
    subprocess.CalledProcessError
        If AOFlagger execution fails

    """
    logger = logging.getLogger(__name__)

    # Determine AOFlagger command
    # Default to native and fall back to Docker when needed
    use_docker = False
    if aoflagger_path:
        if aoflagger_path == "docker":
            # Force Docker usage
            docker_cmd = shutil.which("docker")
            if not docker_cmd:
                suggestions = [
                    "Install Docker",
                    "Verify Docker is in PATH",
                    "Check Docker service is running",
                    "Use --aoflagger-path to specify native AOFlagger location",
                ]
                error_msg = format_ms_error_with_suggestions(
                    RuntimeError("Docker not found but --aoflagger-path=docker was specified"),
                    ms,
                    "AOFlagger setup",
                    suggestions,
                )
                raise RuntimeError(error_msg)
            use_docker = True
            # Use current user ID to avoid permission issues
            user_id = os.getuid()
            group_id = os.getgid()
            aoflagger_cmd = [
                docker_cmd,
                "run",
                "--rm",
                "--user",
                f"{user_id}:{group_id}",
                "-v",
                "/dev/shm/dsa110-contimg:/dev/shm/dsa110-contimg",
                "-v",
                "/data:/data",
                "-v",
                "/stage:/stage",
                "aoflagger:latest",
                "aoflagger",
            ]
        else:
            # Explicit path provided - use it directly
            aoflagger_cmd = [aoflagger_path]
            logger.info(f"Using AOFlagger from explicit path: {aoflagger_path}")
    else:
        # Auto-detect: prefer native, fall back to Docker
        native_aoflagger = shutil.which("aoflagger")
        docker_cmd = shutil.which("docker")

        if native_aoflagger:
            aoflagger_cmd = [native_aoflagger]
            logger.info("Using native AOFlagger")
        else:
            if docker_cmd:
                # Verify image exists before committing to Docker
                try:
                    img_check = subprocess.run(
                        [docker_cmd, "images", "-q", "aoflagger:latest"],
                        capture_output=True,
                        text=True,
                        timeout=5,
                    )
                    if img_check.returncode == 0 and img_check.stdout.strip():
                        use_docker = True
                    else:
                        logger.debug("Docker found but 'aoflagger:latest' image not found.")
                except (subprocess.SubprocessError, OSError):
                    logger.debug("Failed to check Docker images.")

            if use_docker:
                # Docker is available and image exists - use it as fallback
                # Use current user ID to avoid permission issues
                user_id = os.getuid()
                group_id = os.getgid()
                aoflagger_cmd = [
                    docker_cmd,
                    "run",
                    "--rm",
                    "--user",
                    f"{user_id}:{group_id}",
                    "-v",
                    "/dev/shm/dsa110-contimg:/dev/shm/dsa110-contimg",
                    "-v",
                    "/data:/data",
                    "-v",
                    "/stage:/stage",
                    "aoflagger:latest",
                    "aoflagger",
                ]
                logger.debug("Using Docker for AOFlagger (native not found)")
            else:
                suggestions = [
                    "Install native AOFlagger and ensure it's in PATH",
                    "Install Docker and build aoflagger:latest image",
                    "Use --aoflagger-path to specify AOFlagger location",
                    "Check AOFlagger installation documentation",
                ]
                error_msg = format_ms_error_with_suggestions(
                    RuntimeError("AOFlagger not found (native or Docker)."),
                    ms,
                    "AOFlagger setup",
                    suggestions,
                )
                raise RuntimeError(error_msg)

    # Build command
    cmd = aoflagger_cmd.copy()

    # Determine strategy to use
    strategy_to_use = strategy or os.environ.get("CONTIMG_AOFLAGGER_STRATEGY")
    if strategy_to_use is None or str(strategy_to_use).strip() == "":
        strategy_to_use = None
    logger.info("AOFlagger strategy: %s", strategy_to_use or "auto-detect")

    # Add strategy if we have one
    if strategy_to_use:
        # When using Docker, ensure the strategy path is accessible inside the container
        if use_docker:
            # Strategy file must be under /data or /stage (mounted volumes)
            strategy_path = Path(strategy_to_use)
            if not str(strategy_path).startswith(("/data", "/stage")):
                # Try to find it under /data
                strategy_name = strategy_path.name
                docker_strategy_path = f"{os.environ.get('CONTIMG_BASE_DIR', '/data/dsa110-contimg')}/config/{strategy_name}"
                if Path(
                    str(get_env_path("CONTIMG_BASE_DIR", default="/data/dsa110-contimg"))
                    + "/config/dsa110-default.lua"
                ).exists():
                    strategy_to_use = docker_strategy_path
                    logger.debug(f"Using Docker-accessible strategy path: {strategy_to_use}")
                else:
                    logger.warning(
                        f"Strategy file {strategy_to_use} may not be accessible in Docker container. "
                        f"Ensure it's under /data or /stage, or mount it explicitly."
                    )
        cmd.extend(["-strategy", strategy_to_use])

    # Add MS path (required - AOFlagger will auto-detect strategy if not specified)
    # Also add parallel processing flag (use all available cores)

    # Respect CPU affinity if set (e.g. in Docker/Kubernetes)
    try:
        # available in Python 3.3+ on Linux
        n_cores = len(os.sched_getaffinity(0))
    except (AttributeError, NotImplementedError, OSError):
        # Fallback to cpu_count (physical cores on host)
        n_cores = multiprocessing.cpu_count()

    cmd.extend(["-j", str(n_cores)])

    # If datacolumn is specified and not "data", tell AOFlagger to use it
    # Note: "data" is the default for AOFlagger so we don't need to specify it
    if datacolumn and datacolumn.lower() != "data":
        cmd.extend(["-column", datacolumn])

    cmd.append(ms)

    # Execute AOFlagger
    logger.info(f"Running AOFlagger: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True, capture_output=False)
        logger.info(":check: AOFlagger RFI flagging complete")
    except subprocess.CalledProcessError as e:
        logger.error(f"AOFlagger failed with exit code {e.returncode}")
        raise
    except FileNotFoundError:
        suggestions = [
            "Check AOFlagger installation",
            "Verify AOFlagger is in PATH",
            "Use --aoflagger-path to specify AOFlagger location",
            "Check Docker image is available (if using Docker)",
        ]
        error_msg = format_ms_error_with_suggestions(
            FileNotFoundError(f"AOFlagger executable not found: {aoflagger_cmd[0]}"),
            ms,
            "AOFlagger execution",
            suggestions,
        )
        logger.error(error_msg)
        raise RuntimeError(error_msg)


def flag_antenna(ms: str, antenna: str, datacolumn: str = "data", pol: str | None = None) -> None:
    antenna_sel = antenna if pol is None else f"{antenna}&{pol}"
    service = CASAService()
    with suppress_subprocess_stderr():
        service.flagdata(vis=ms, mode="manual", antenna=antenna_sel, datacolumn=datacolumn)


def flag_baselines(ms: str, uvrange: str = "2~50m", datacolumn: str = "data") -> None:
    service = CASAService()
    with suppress_subprocess_stderr():
        service.flagdata(vis=ms, mode="manual", uvrange=uvrange, datacolumn=datacolumn)


def flag_manual(
    ms: str,
    antenna: str | None = None,
    scan: str | None = None,
    spw: str | None = None,
    field: str | None = None,
    uvrange: str | None = None,
    timerange: str | None = None,
    correlation: str | None = None,
    datacolumn: str = "data",
) -> None:
    """Manual flagging with selection parameters.

        Flags data matching the specified selection criteria using CASA's
        standard selection syntax. All parameters are optional - specify any
        combination to flag matching data.

    Parameters
    ----------
    ms : str
        Path to Measurement Set
    antenna : Optional[str], optional
        Antenna selection (e.g., '0,1,2' or 'ANT01,ANT02')
    scan : Optional[str], optional
        Scan selection (e.g., '1~5' or '1,3,5')
    spw : Optional[str], optional
        Spectral window selection (e.g., '0:10~20')
    field : Optional[str], optional
        Field selection (field IDs or names)
    uvrange : Optional[str], optional
        UV range selection (e.g., '>100m' or '10~50m')
    timerange : Optional[str], optional
        Time range selection (e.g., '2025/01/01/10:00:00~10:05:00')
    correlation : Optional[str], optional
        Correlation product selection (e.g., 'RR,LL')
    datacolumn : str, optional
        Data column to use (default: 'data')

    Note: At least one selection parameter must be provided.

    """
    kwargs = {"vis": ms, "mode": "manual", "datacolumn": datacolumn}
    if antenna:
        kwargs["antenna"] = antenna
    if scan:
        kwargs["scan"] = scan
    if spw:
        kwargs["spw"] = spw
    if field:
        kwargs["field"] = field
    if uvrange:
        kwargs["uvrange"] = uvrange
    if timerange:
        kwargs["timerange"] = timerange
    if correlation:
        kwargs["correlation"] = correlation

    if len([k for k in [antenna, scan, spw, field, uvrange, timerange, correlation] if k]) == 0:
        suggestions = [
            "Provide at least one selection parameter (antenna, time, baseline, etc.)",
            "Check manual flagging command syntax",
            "Review flagging documentation for parameter requirements",
        ]
        error_msg = format_ms_error_with_suggestions(
            ValueError("At least one selection parameter must be provided for manual flagging"),
            ms,
            "manual flagging",
            suggestions,
        )
        raise ValueError(error_msg)

    service = CASAService()
    with suppress_subprocess_stderr():
        service.flagdata(**kwargs)


def flag_shadow(ms: str, tolerance: float = 0.0) -> None:
    """Flag geometrically shadowed baselines.

    Flags data where one antenna physically blocks the line of sight
    between another antenna and the source. This is particularly important
    for low-elevation observations and compact array configurations.

    Parameters
    ----------
    ms :
        Path to Measurement Set
    tolerance :
        Shadowing tolerance in degrees (default: 0.0)
    """
    service = CASAService()
    with suppress_subprocess_stderr():
        service.flagdata(vis=ms, mode="shadow", tolerance=tolerance)


def flag_quack(
    ms: str,
    quackinterval: float = 2.0,
    quackmode: str = "beg",
    datacolumn: str = "data",
) -> None:
    """Flag beginning/end of scans to remove antenna settling transients.

    After slewing to a new source, antennas require time to stabilize
    thermally and mechanically. This function flags the specified duration
    from the beginning or end of each scan.

    Parameters
    ----------
    ms :
        Path to Measurement Set
    quackinterval :
        Duration in seconds to flag (default: 2.0)
    quackmode :
        'beg' (beginning), 'end', 'tail', or 'endb' (default: 'beg')
    datacolumn :
        Data column to use (default: 'data')
    """
    service = CASAService()
    with suppress_subprocess_stderr():
        service.flagdata(
            vis=ms,
            mode="quack",
            datacolumn=datacolumn,
            quackinterval=quackinterval,
            quackmode=quackmode,
        )


def flag_elevation(
    ms: str,
    lowerlimit: float | None = None,
    upperlimit: float | None = None,
    datacolumn: str = "data",
) -> None:
    """Flag observations below/above specified elevation limits.

    Low-elevation observations suffer from increased atmospheric opacity,
    phase instability, and reduced sensitivity. High-elevation observations
    may have other issues. This function flags data outside specified limits.

    Parameters
    ----------
    ms :
        Path to Measurement Set
    lowerlimit :
        Minimum elevation in degrees (flag data below this)
    upperlimit :
        Maximum elevation in degrees (flag data above this)
    datacolumn :
        Data column to use (default: 'data')
    """
    kwargs = {"vis": ms, "mode": "elevation", "datacolumn": datacolumn}
    if lowerlimit is not None:
        kwargs["lowerlimit"] = lowerlimit
    if upperlimit is not None:
        kwargs["upperlimit"] = upperlimit
    service = CASAService()
    with suppress_subprocess_stderr():
        service.flagdata(**kwargs)


def flag_clip(
    ms: str,
    clipminmax: list[float],
    clipoutside: bool = True,
    correlation: str = "ABS_ALL",
    datacolumn: str = "data",
    channelavg: bool = False,
    timeavg: bool = False,
    chanbin: int | None = None,
    timebin: str | None = None,
) -> None:
    """Flag data outside specified amplitude thresholds.

    Flags visibility amplitudes that fall outside acceptable ranges.
    Useful for identifying extreme outliers, strong RFI, or systematic problems.

    Parameters
    ----------
    ms :
        Path to Measurement Set
    clipminmax :
        [min, max] amplitude range in Jy
    clipoutside :
        If True, flag outside range; if False, flag inside range
    correlation :
        Correlation product ('ABS_ALL', 'RR', 'LL', etc.)
    datacolumn :
        Data column to use (default: 'data')
    channelavg :
        Average channels before clipping
    timeavg :
        Average time before clipping
    chanbin :
        Channel binning factor
    timebin :
        Time binning (e.g., '30s')
    """
    kwargs = {
        "vis": ms,
        "mode": "clip",
        "datacolumn": datacolumn,
        "clipminmax": clipminmax,
        "clipoutside": clipoutside,
        "correlation": correlation,
    }
    if channelavg or chanbin:
        kwargs["channelavg"] = channelavg
        if chanbin:
            kwargs["chanbin"] = chanbin
    if timeavg or timebin:
        kwargs["timeavg"] = timeavg
        if timebin:
            kwargs["timebin"] = timebin
    service = CASAService()
    with suppress_subprocess_stderr():
        service.flagdata(**kwargs)


def flag_extend(
    ms: str,
    growtime: float = 0.0,
    growfreq: float = 0.0,
    growaround: bool = False,
    flagneartime: bool = False,
    flagnearfreq: bool = False,
    extendpols: bool = True,
    datacolumn: str = "data",
) -> None:
    """Extend existing flags to neighboring data points.

    RFI often affects neighboring channels, times, or correlations through
    hardware responses, cross-talk, or physical proximity. This function
    grows flagged regions appropriately.

    Parameters
    ----------
    ms :
        Path to Measurement Set
    growtime :
        Fraction of time already flagged to flag entire time slot (0-1)
    growfreq :
        Fraction of frequency already flagged to flag entire channel (0-1)
    growaround :
        Flag points if most neighbors are flagged
    flagneartime :
        Flag points immediately before/after flagged regions
    flagnearfreq :
        Flag points immediately adjacent to flagged channels
    extendpols :
        Extend flags across polarization products
    datacolumn :
        Data column to use (default: 'data')
    """
    # Try using CASA flagdata first
    try:
        service = CASAService()
        with suppress_subprocess_stderr():
            service.flagdata(
                vis=ms,
                mode="extend",
                datacolumn=datacolumn,
                growtime=growtime,
                growfreq=growfreq,
                growaround=growaround,
                flagneartime=flagneartime,
                flagnearfreq=flagnearfreq,
                extendpols=extendpols,
                flagbackup=False,
            )
    except RuntimeError as e:
        # If CASA fails due to file lock, try direct casacore approach for simple extension
        if ("cannot be opened" in str(e) or "not writable" in str(e)) and (
            flagneartime or flagnearfreq
        ):
            logger = logging.getLogger(__name__)
            logger.debug("CASA flagdata failed, trying direct casacore flag extension")
            try:
                _extend_flags_direct(
                    ms,
                    flagneartime=flagneartime,
                    flagnearfreq=flagnearfreq,
                    extendpols=extendpols,
                )
            except Exception as e2:
                logger.warning(f"Direct flag extension also failed: {e2}. Flag extension skipped.")
                raise RuntimeError(f"Flag extension failed: {e}") from e
        else:
            raise


def _extend_flags_direct(
    ms: str,
    flagneartime: bool = False,
    flagnearfreq: bool = False,
    extendpols: bool = True,
) -> None:
    """Extend flags directly using casacore.tables (fallback when CASA flagdata fails).

    This is a simpler implementation that only handles adjacent channel/time extension.
    For more complex extension (growaround, growtime, etc.), use CASA flagdata.

    Parameters
    ----------
    """
    try:
        import casacore.tables as casatables
        import numpy as np

        table = casatables.table

        with table(ms, readonly=False, ack=False) as tb:
            flags = tb.getcol("FLAG")

            if flags.size == 0:
                return

            # Create extended flags
            extended_flags = flags.copy()

            # Extend in frequency direction (adjacent channels)
            if flagnearfreq:
                # Shape: (nrows, nchans, npols)
                nrows, nchans, npols = flags.shape
                for row in range(nrows):
                    for pol in range(npols):
                        row_flags = flags[row, :, pol]
                        # Flag channels adjacent to flagged channels
                        flagged_chans = np.where(row_flags)[0]
                        for chan in flagged_chans:
                            if chan > 0:
                                extended_flags[row, chan - 1, pol] = True
                            if chan < nchans - 1:
                                extended_flags[row, chan + 1, pol] = True

            # Extend in time direction (adjacent time samples)
            if flagneartime:
                # Flag time samples adjacent to flagged samples
                nrows, nchans, npols = flags.shape
                for row in range(nrows):
                    if np.any(flags[row]):
                        # Flag adjacent rows (time samples)
                        if row > 0:
                            extended_flags[row - 1] = extended_flags[row - 1] | flags[row]
                        if row < nrows - 1:
                            extended_flags[row + 1] = extended_flags[row + 1] | flags[row]

            # Extend across polarizations
            if extendpols:
                # If any pol is flagged, flag all pols
                nrows, nchans, npols = flags.shape
                for row in range(nrows):
                    for chan in range(nchans):
                        if np.any(flags[row, chan]):
                            extended_flags[row, chan, :] = True

            # Write extended flags back
            tb.putcol("FLAG", extended_flags)
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.debug(f"Direct flag extension failed: {e}")
        raise


def analyze_channel_flagging_stats(ms_path: str, threshold: float = 0.5) -> dict[int, list[int]]:
    """Analyze flagging statistics per channel across all SPWs.

    After RFI flagging, this function identifies channels that have high flagging
    rates and should be flagged entirely before calibration. This is more precise
    than SPW-level flagging since SPWs are arbitrary subdivisions for data processing.

    Parameters
    ----------
    ms_path : str
        Path to Measurement Set
    threshold : float, optional
        Fraction of flagged data to consider channel problematic (default is 0.5)

    Returns
    -------
    dict
        Dictionary mapping SPW ID to list of problematic channel indices.

    Examples
    --------
    >>> problematic = analyze_channel_flagging_stats('data.ms', threshold=0.5)
    >>> # Returns: {1: [5, 10, 15, 20], 12: [3, 7, 11]}
    """
    import casacore.tables as casatables
    import numpy as np

    table = casatables.table

    logger = logging.getLogger(__name__)
    problematic_channels = {}

    try:
        with table(ms_path, readonly=True) as tb:
            flags = tb.getcol("FLAG")  # Shape: (nrows, nchannels, npol)
            data_desc_id = tb.getcol("DATA_DESC_ID")

            # Get SPW mapping from DATA_DESCRIPTION table
            with table(f"{ms_path}::DATA_DESCRIPTION", readonly=True) as dd:
                spw_ids = dd.getcol("SPECTRAL_WINDOW_ID")

            # Get unique SPWs present in data
            unique_ddids = np.unique(data_desc_id)
            unique_spws = np.unique([spw_ids[ddid] for ddid in unique_ddids])

            logger.debug(f"Analyzing channel flagging for {len(unique_spws)} SPW(s)")

            for spw in unique_spws:
                # Get rows for this SPW
                spw_mask = np.array([spw_ids[ddid] == spw for ddid in data_desc_id])
                spw_flags = flags[spw_mask]

                if len(spw_flags) == 0:
                    continue

                # Calculate flagging fraction per channel
                # flags shape: (nrows, nchannels, npol)
                # Average across rows and polarizations
                channel_flagging = np.mean(spw_flags, axis=(0, 2))

                # Find channels above threshold
                problematic = np.where(channel_flagging > threshold)[0].tolist()

                if problematic:
                    problematic_channels[int(spw)] = problematic
                    logger.debug(
                        f"SPW {spw}: {len(problematic)}/{len(channel_flagging)} channels "
                        f"above {threshold * 100:.1f}% flagging threshold"
                    )

    except Exception as e:
        logger.warning(f"Failed to analyze channel flagging statistics: {e}")
        logger.warning("Skipping channel-level flagging analysis")

    return problematic_channels


def flag_problematic_channels(
    ms_path: str, problematic_channels: dict[int, list[int]], datacolumn: str = "data"
) -> None:
    """Flag problematic channels using CASA flagdata.

    Parameters
    ----------
    ms_path :
        Path to Measurement Set
    problematic_channels :
        Dict mapping SPW ID -> list of channel indices
    datacolumn :
        Data column to flag (default: "data")

    Raises
    ------
    RuntimeError
        If flagdata fails

    """
    logger = logging.getLogger(__name__)

    if not problematic_channels:
        logger.debug("No problematic channels to flag")
        return

    # Build SPW selection string for CASA flagdata
    # Format: "spw:chan1,chan2,chan3;spw:chan1,chan2"
    spw_selections = []
    total_channels = 0

    for spw, channels in sorted(problematic_channels.items()):
        # Sort channels for cleaner output
        channels_sorted = sorted(channels)
        chan_str = ",".join(map(str, channels_sorted))
        spw_selections.append(f"{spw}:{chan_str}")
        total_channels += len(channels_sorted)
        logger.info(
            f"  SPW {spw}: {len(channels_sorted)} problematic channels "
            f"({channels_sorted[:5]}{'...' if len(channels_sorted) > 5 else ''})"
        )

    spw_sel = ";".join(spw_selections)

    logger.info(
        f"Flagging {total_channels} problematic channel(s) across "
        f"{len(problematic_channels)} SPW(s) before calibration"
    )

    try:
        service = CASAService()
        with suppress_subprocess_stderr():
            service.flagdata(
                vis=ms_path,
                spw=spw_sel,
                mode="manual",
                datacolumn=datacolumn,
                flagbackup=False,
            )
        logger.info(f":check: Flagged {total_channels} problematic channel(s) before calibration")
    except Exception as e:
        logger.error(f"Failed to flag problematic channels: {e}")
        raise RuntimeError(f"Channel flagging failed: {e}") from e


def flag_summary(
    ms: str,
    spw: str = "",
    field: str = "",
    antenna: str = "",
    uvrange: str = "",
    correlation: str = "",
    timerange: str = "",
    reason: str = "",
) -> dict:
    """Report flagging statistics without flagging data.

    Provides comprehensive statistics about existing flags, including
    total flagged fraction, breakdowns by antenna, spectral window,
    polarization, and other dimensions. Useful for understanding data quality
    and identifying problematic subsets.

    Parameters
    ----------
    ms :
        Path to Measurement Set
    spw :
        Spectral window selection
    field :
        Field selection
    antenna :
        Antenna selection
    uvrange :
        UV range selection
    correlation :
        Correlation product selection
    timerange :
        Time range selection
    reason :
        Flag reason to query

    Returns
    -------
        Dictionary with flagging statistics

    """
    kwargs = {"vis": ms, "mode": "summary", "display": "report"}
    if spw:
        kwargs["spw"] = spw
    if field:
        kwargs["field"] = field
    if antenna:
        kwargs["antenna"] = antenna
    if uvrange:
        kwargs["uvrange"] = uvrange
    if correlation:
        kwargs["correlation"] = correlation
    if timerange:
        kwargs["timerange"] = timerange
    if reason:
        kwargs["reason"] = reason

    # Skip calling flagdata in summary mode - it triggers casaplotserver which hangs
    # Instead, directly read flags from the MS using casacore.tables
    # This is faster and avoids subprocess issues
    # with suppress_subprocess_stderr():
    #     flagdata(**kwargs)

    # Parse summary statistics directly from MS (faster and avoids casaplotserver)
    try:
        import casacore.tables as casatables
        import numpy as np

        table = casatables.table

        stats = {}
        with table(ms, readonly=True) as tb:
            n_rows = tb.nrows()
            if n_rows > 0:
                flags = tb.getcol("FLAG")
                total_points = flags.size
                flagged_points = np.sum(flags)
                stats["total_fraction_flagged"] = (
                    float(flagged_points / total_points) if total_points > 0 else 0.0
                )
                stats["n_rows"] = int(n_rows)

        return stats
    except (OSError, RuntimeError, KeyError):
        return {}


def detect_and_flag_dead_antennas(
    ms: str,
    threshold: float = 0.95,
    dry_run: bool = False,
) -> dict:
    """Detect antennas with excessive flagging and optionally flag them completely.

    Scans the Measurement Set for per-antenna flagging statistics. Antennas with
    flagged data above the threshold are considered "dead" and can be flagged
    completely to prevent calibration errors (e.g., CASA getcell::TIME errors
    when trying to solve for antennas with no usable data).

    This function should be called AFTER initial flagging (zeros, autocorrelations)
    but BEFORE calibration (bandpass, gaincal).

    Parameters
    ----------
    ms :
        Path to Measurement Set
    threshold :
        Fraction of flagged data above which an antenna is considered
        dead (default: 0.95 = 95% flagged)
    dry_run :
        If True, only report statistics without flagging (default: False)

    Returns
    -------
    Dictionary with
        - 'dead_antennas': List of antenna IDs flagged as dead
        - 'partial_antennas': List of antenna IDs with >50% but <=threshold flagging
        - 'antenna_stats': Dict mapping antenna ID to flagged fraction
        - 'total_flagged_before': Overall flagged fraction before action
        - 'total_flagged_after': Overall flagged fraction after action (same if dry_run)
        - 'n_dead': Number of dead antennas detected
        - 'n_partial': Number of partially bad antennas
        - 'action_taken': Whether antennas were flagged (False if dry_run)

    """
    import logging

    import casacore.tables as casatables
    import numpy as np

    logger = logging.getLogger(__name__)

    result = {
        "dead_antennas": [],
        "partial_antennas": [],
        "antenna_stats": {},
        "total_flagged_before": 0.0,
        "total_flagged_after": 0.0,
        "n_dead": 0,
        "n_partial": 0,
        "action_taken": False,
    }

    try:
        # Read antenna info and compute per-antenna flagging statistics
        with casatables.table(ms, readonly=True) as tb:
            n_rows = tb.nrows()
            if n_rows == 0:
                logger.warning(f"MS {ms} has no rows")
                return result

            # Get columns needed for per-antenna stats
            ant1 = tb.getcol("ANTENNA1")
            ant2 = tb.getcol("ANTENNA2")
            flags = tb.getcol("FLAG")  # shape: (n_rows, n_chan, n_pol)

            # Compute overall flag fraction
            total_points = flags.size
            flagged_points = np.sum(flags)
            result["total_flagged_before"] = float(flagged_points / total_points)

            # Get unique antenna IDs
            all_antennas = np.unique(np.concatenate([ant1, ant2]))

            # Compute per-antenna flagging
            # An antenna participates in a row if it's ant1 or ant2
            antenna_stats = {}
            for ant_id in all_antennas:
                mask = (ant1 == ant_id) | (ant2 == ant_id)
                ant_flags = flags[mask]
                if ant_flags.size > 0:
                    frac = float(np.sum(ant_flags) / ant_flags.size)
                    antenna_stats[int(ant_id)] = frac

            result["antenna_stats"] = antenna_stats

            # Classify antennas
            dead_antennas = []
            partial_antennas = []
            for ant_id, frac in antenna_stats.items():
                if frac >= threshold:
                    dead_antennas.append(ant_id)
                elif frac >= 0.5:
                    partial_antennas.append(ant_id)

            result["dead_antennas"] = sorted(dead_antennas)
            result["partial_antennas"] = sorted(partial_antennas)
            result["n_dead"] = len(dead_antennas)
            result["n_partial"] = len(partial_antennas)

        # Log findings
        if dead_antennas:
            logger.info(
                f"Detected {len(dead_antennas)} dead antennas (>{threshold * 100:.0f}% flagged): "
                f"{sorted(dead_antennas)}"
            )
        if partial_antennas:
            logger.info(
                f"Detected {len(partial_antennas)} partially bad antennas (50-{threshold * 100:.0f}% flagged): "
                f"{sorted(partial_antennas)}"
            )

        # Flag dead antennas if not dry_run
        if dead_antennas and not dry_run:
            antenna_sel = ",".join(str(a) for a in dead_antennas)
            logger.info(f"Flagging dead antennas: {antenna_sel}")
            flag_antenna(ms, antenna_sel)
            result["action_taken"] = True

            # Recompute total flagging after action
            with casatables.table(ms, readonly=True) as tb:
                flags = tb.getcol("FLAG")
                result["total_flagged_after"] = float(np.sum(flags) / flags.size)
        else:
            result["total_flagged_after"] = result["total_flagged_before"]

        # Save results to JSON file alongside the MS for later retrieval
        import json
        from datetime import datetime
        from pathlib import Path

        ms_path = Path(ms)
        report_path = ms_path.parent / f"{ms_path.stem}_antenna_health.json"

        report = {
            "ms_path": str(ms_path.absolute()),
            "ms_name": ms_path.name,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "threshold": threshold,
            "dry_run": dry_run,
            "dead_antennas": result["dead_antennas"],
            "partial_antennas": result["partial_antennas"],
            "n_dead": result["n_dead"],
            "n_partial": result["n_partial"],
            "total_flagged_before": result["total_flagged_before"],
            "total_flagged_after": result["total_flagged_after"],
            "action_taken": result["action_taken"],
            "antenna_stats": result["antenna_stats"],
        }

        try:
            with open(report_path, "w") as f:
                json.dump(report, f, indent=2)
            result["report_path"] = str(report_path)
            logger.info(f"Antenna health report saved to: {report_path}")
        except OSError as e:
            logger.warning(f"Could not save antenna health report: {e}")
            result["report_path"] = None

        return result

    except (OSError, RuntimeError, KeyError) as e:
        logger.error(f"Error detecting dead antennas in {ms}: {e}")
        return result
