"""
Core business logic for FITS file viewing.

Provides:
- Configuration management for FITS viewers (JS9, CARTA, Aladin)
- FITS metadata extraction and validation
- Viewer URL generation
- HTML button generation for viewer integration

Designed for integration with the DSA-110 continuum imaging pipeline
report generation and API endpoints.
"""

from __future__ import annotations

import hashlib
import html
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

from astropy.io import fits
from astropy.wcs import WCS
from dsa110_contimg.common.utils import get_env_path

# Logging
logger = logging.getLogger(__name__)

# File size limits
DEFAULT_MAX_FILE_SIZE_GB = 5.0
MIN_VIEWABLE_DIMENSIONS = 2  # Minimum 2D array required

# URL/Path defaults (single-port design: frontend served by API on 8000)
DEFAULT_JS9_PORT = 8000
DEFAULT_CARTA_PORT = 9002

# Viewer types
VIEWER_JS9 = "js9"
VIEWER_CARTA = "carta"
VIEWER_ALADIN = "aladin"  # Future support

# Color mapping
DEFAULT_COLORMAP = "inferno"
AVAILABLE_COLORMAPS = ["inferno", "viridis", "plasma", "cividis", "gray"]

# WCS axis labels
WCS_AXIS_LABELS = {
    "RA": "Right Ascension",
    "DEC": "Declination",
    "FREQ": "Frequency",
    "STOKES": "Stokes Parameter",
}


class FITSViewerException(Exception):
    """Base exception for FITS viewer errors."""


class FITSFileError(FITSViewerException):
    """File-related errors."""


class FITSParsingError(FITSViewerException):
    """FITS format parsing errors."""


@dataclass
class FITSViewerConfig:
    """Configuration for FITS viewer."""

    # Viewer enablement
    js9_enabled: bool = True
    carta_enabled: bool = False
    aladin_enabled: bool = False
    default_viewer: str = VIEWER_JS9

    # Size and caching
    preview_size: int = 512  # pixels
    max_file_size_gb: float = DEFAULT_MAX_FILE_SIZE_GB
    cache_dir: str | None = None
    enable_symlinks: bool = True

    # URL configuration
    server_base_url: str = "http://localhost:8000"
    js9_instance_url: str | None = None
    carta_instance_url: str | None = None
    aladin_instance_url: str | None = None

    # FITS parsing
    colormap: str = DEFAULT_COLORMAP
    timeout_seconds: int = 5

    # Security
    safe_mode: bool = True
    safe_directories: list[str] = field(default_factory=list)

    # Logging
    logger_name: str = "dsa110.visualization.fits_viewer"

    def __post_init__(self):
        """Validate and normalize configuration."""
        # Validate colormap
        if self.colormap not in AVAILABLE_COLORMAPS:
            logger.warning(
                f"Colormap '{self.colormap}' not in available maps, using '{DEFAULT_COLORMAP}'"
            )
            self.colormap = DEFAULT_COLORMAP

        # Set up cache directory
        if self.cache_dir is None:
            self.cache_dir = self._get_default_cache_dir()

        # Ensure cache directory exists
        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)

        # Validate directory is writable
        try:
            test_file = Path(self.cache_dir) / ".write_test"
            test_file.touch()
            test_file.unlink()
        except OSError as e:
            raise ValueError(f"Cache directory not writable: {self.cache_dir}") from e

        # Validate at least one viewer is enabled
        if not (self.js9_enabled or self.carta_enabled or self.aladin_enabled):
            logger.warning("No viewers enabled; enabling JS9 by default")
            self.js9_enabled = True

        # Validate default viewer is enabled
        if self.default_viewer == VIEWER_JS9 and not self.js9_enabled:
            self.default_viewer = (
                VIEWER_CARTA
                if self.carta_enabled
                else VIEWER_ALADIN
                if self.aladin_enabled
                else VIEWER_JS9
            )

        # Validate safe directories
        if self.safe_mode and not self.safe_directories:
            logger.warning(
                "Safe mode enabled with no safe directories; file access will be restricted"
            )

    @staticmethod
    def _get_default_cache_dir() -> str:
        """Get default cache directory path.

        Priority:
        1. Environment variable FITS_VIEWER_CACHE
        2. Project temp directory + dsa110_fits_viewer
        3. Current working directory + .fits_viewer_cache

        """
        if cache_env := os.environ.get("FITS_VIEWER_CACHE"):
            return cache_env

        from dsa110_contimg.common.utils.temp_manager import get_temp_subdir

        cache_dir = get_temp_subdir("fits_viewer_cache")
        return str(cache_dir)

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)

    def validate(self) -> tuple[bool, str]:
        """Validate configuration completeness."""
        # Check at least one viewer enabled
        if not (self.js9_enabled or self.carta_enabled or self.aladin_enabled):
            return False, "No viewers enabled"

        # Check cache directory
        try:
            Path(self.cache_dir).resolve()
        except (OSError, ValueError) as e:
            return False, f"Invalid cache directory: {e}"

        # Check URLs are valid
        if self.js9_instance_url and not self.js9_instance_url.startswith(("http://", "https://")):
            return False, "JS9 instance URL must be HTTP(S)"

        return True, ""

    @classmethod
    def from_env(cls) -> FITSViewerConfig:
        """Create configuration from environment variables.

            Environment variables:
        FITS_VIEWER_JS9_ENABLED: Enable JS9 viewer (true/false)
        FITS_VIEWER_CARTA_ENABLED: Enable CARTA viewer (true/false)
        FITS_VIEWER_CACHE: Cache directory path
        FITS_VIEWER_MAX_FILE_SIZE_GB: Maximum file size (float)
        FITS_VIEWER_SAFE_MODE: Enable safe mode (true/false)
        FITS_VIEWER_SAFE_DIRECTORIES: Comma-separated list of directories
        FITS_VIEWER_SERVER_BASE_URL: Base URL for viewer links
        FITS_VIEWER_JS9_URL: JS9 instance URL
        FITS_VIEWER_CARTA_URL: CARTA instance URL

        Returns
        -------
            FITSViewerConfig
            Configuration instance created from environment variables
        """

        def parse_bool(val: str) -> bool:
            return val.lower() in ("true", "1", "yes", "on")

        from dsa110_contimg.common.utils import get_env_list

        safe_dirs = get_env_list("FITS_VIEWER_SAFE_DIRECTORIES")

        return cls(
            js9_enabled=parse_bool(os.environ.get("FITS_VIEWER_JS9_ENABLED", "true")),
            carta_enabled=parse_bool(os.environ.get("FITS_VIEWER_CARTA_ENABLED", "false")),
            cache_dir=os.environ.get("FITS_VIEWER_CACHE"),
            max_file_size_gb=float(
                os.environ.get("FITS_VIEWER_MAX_FILE_SIZE_GB", DEFAULT_MAX_FILE_SIZE_GB)
            ),
            safe_mode=parse_bool(os.environ.get("FITS_VIEWER_SAFE_MODE", "true")),
            safe_directories=safe_dirs,
            server_base_url=os.environ.get("FITS_VIEWER_SERVER_BASE_URL", "http://localhost:8000"),
            js9_instance_url=os.environ.get("FITS_VIEWER_JS9_URL"),
            carta_instance_url=os.environ.get("FITS_VIEWER_CARTA_URL"),
        )

    @staticmethod
    def get_placeholder_path() -> Path | None:
        """Get the path to the placeholder FITS image for initial viewer display.

        Returns
        -------
        Optional[Path]
            Path to placeholder FITS file, or None if not found
        """
        base_dir = get_env_path("CONTIMG_BASE_DIR", default="/data/dsa110-contimg")
        placeholder_path = base_dir / "state" / "assets" / "placeholder_demo.fits"

        if placeholder_path.exists():
            return placeholder_path

        return None


@dataclass
class FITSViewerMetadata:
    """Extracted display-oriented metadata from a FITS file.

    This is the visualization-layer metadata model, optimized for
    rendering tooltips, axis summaries, and validity checks.

    For the full per-keyword header extraction used by the API, see
    ``dsa110_contimg.interfaces.api.services.fits_service.FITSMetadata``.
    """

    filename: str = ""
    filepath: str = ""
    shape: tuple[int, ...] = field(default_factory=tuple)
    naxis: int = 0
    axes: list[str] = field(default_factory=list)
    cunit: list[str] = field(default_factory=list)
    cdelt: list[float] = field(default_factory=list)
    resolution: str = "Unknown"
    date_obs: str = ""
    object: str = ""
    telescop: str = ""
    instrume: str = ""
    wcs_available: bool = False
    has_data: bool = False
    file_size_mb: float = 0.0
    is_valid: bool = False
    error_message: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "filename": self.filename,
            "filepath": self.filepath,
            "shape": list(self.shape),
            "naxis": self.naxis,
            "axes": self.axes,
            "cunit": self.cunit,
            "cdelt": self.cdelt,
            "resolution": self.resolution,
            "date_obs": self.date_obs,
            "object": self.object,
            "telescop": self.telescop,
            "instrume": self.instrume,
            "wcs_available": self.wcs_available,
            "has_data": self.has_data,
            "file_size_mb": self.file_size_mb,
            "is_valid": self.is_valid,
        }

    def format_shape(self) -> str:
        """Format shape as readable string."""
        if not self.shape:
            return "No data"
        return " × ".join(str(s) for s in self.shape)

    def format_axes(self) -> str:
        """Format axes as readable string."""
        if not self.axes:
            return "Unknown"
        return ", ".join(self.axes)


def validate_fits_file(fits_path: str) -> tuple[bool, str]:
    """Quick validation of FITS file.

    Parameters
    ----------
    fits_path : str
        Path to FITS file

    Returns
    -------
        bool
        True if FITS file is valid
    """
    path = Path(fits_path)

    # Check file exists
    if not path.exists():
        return False, f"File not found: {fits_path}"

    # Check file is readable
    if not path.is_file():
        return False, f"Not a file: {fits_path}"

    # Check extension
    valid_extensions = {".fits", ".fit", ".fts", ".fits.gz", ".fits.fz"}
    if path.suffix.lower() not in valid_extensions and not str(path).lower().endswith(
        (".fits.gz", ".fits.fz")
    ):
        return False, f"Invalid extension: {path.suffix}"

    # Try to open as FITS
    try:
        with fits.open(fits_path) as hdul:
            if len(hdul) == 0:
                return False, "No HDUs in FITS file"
            # Check for data in primary HDU
            if hdul[0].data is None and len(hdul) == 1:
                return False, "No data in FITS file"
    except OSError as e:
        return False, f"Cannot open FITS file: {e}"
    except (TypeError, ValueError) as e:
        return False, f"FITS parsing error: {e}"
    except Exception as e:
        # Catch-all for astropy.io.fits sometimes raising other exceptions
        return False, f"Unexpected FITS error: {e}"

    return True, ""


def get_file_size_mb(fits_path: str) -> float:
    """Get FITS file size in megabytes.

    Parameters
    ----------
    fits_path : str
        Path to FITS file

    Returns
    -------
        float
        File size in megabytes
    """
    try:
        return Path(fits_path).stat().st_size / (1024 * 1024)
    except OSError:
        return 0.0


def format_resolution_degrees(degrees: float, tolerance: float = 1e-6) -> str:
    """Convert degree value to readable resolution.

    Parameters
    ----------
    degrees : float
        Resolution in degrees
    tolerance : float
        Tolerance for zero detection (default is 1e-6)

    Returns
    -------
        str
        Formatted resolution string
    """
    if abs(degrees) < tolerance:
        return "0"

    abs_degrees = abs(degrees)

    # Convert to appropriate unit
    if abs_degrees >= 1.0:
        return f"{abs_degrees:.2f} deg"
    elif abs_degrees >= 1 / 60:  # >= 1 arcmin
        arcmin = abs_degrees * 60
        return f"{arcmin:.2f} arcmin"
    else:
        arcsec = abs_degrees * 3600
        return f"{arcsec:.2f} arcsec"


def get_axis_label(header: fits.Header, axis_num: int) -> str:
    """Get human-readable label for axis.

    Parameters
    ----------
    header : fits.Header
        FITS header
    axis_num : int
        1-based axis number

    Returns
    -------
        str
        Human-readable axis label
    """
    ctype_key = f"CTYPE{axis_num}"
    ctype = header.get(ctype_key, "")

    if not ctype:
        return f"Axis {axis_num}"

    # Extract main type (e.g., "RA---SIN" -> "RA")
    main_type = ctype.split("-")[0].strip()

    # Look up in WCS_AXIS_LABELS
    return WCS_AXIS_LABELS.get(main_type.upper(), ctype)


class FITSViewerManager:
    """Manage FITS file viewing operations.

        Provides methods for:
        - Validating FITS files for viewing
        - Extracting metadata from FITS headers
        - Generating viewer URLs (JS9, CARTA, Aladin)
        - Creating HTML viewer buttons

        Example
    -------
        >>> config = FITSViewerConfig(js9_enabled=True)
        >>> manager = FITSViewerManager(config)
        >>> is_viewable, reason = manager.is_viewable_fits("/data/image.fits")
        >>> if is_viewable:
        ...     buttons_html = manager.get_viewer_buttons("/data/image.fits")
    """

    def __init__(self, config: FITSViewerConfig, app_config: dict | None = None) -> None:
        """Initialize with configuration.

        Parameters
        ----------
        config : FITSViewerConfig
            FITSViewerConfig instance
        app_config : dict, optional
            Optional application configuration dictionary
        """
        self.config = config
        self.app_config = app_config or {}
        self.logger = logging.getLogger(config.logger_name)

    def _validate_and_resolve_path(self, fits_path: str) -> Path:
        """Validate and resolve FITS file path.

            Prevents directory traversal attacks when safe mode is enabled.

        Parameters
        ----------
        fits_path : str
            Path to FITS file

        Returns
        -------
            str
            Resolved and validated FITS file path
        """
        resolved_path = Path(fits_path).resolve()

        if self.config.safe_mode and self.config.safe_directories:
            safe_paths = [Path(d).resolve() for d in self.config.safe_directories]
            is_safe = any(str(resolved_path).startswith(str(safe_path)) for safe_path in safe_paths)
            if not is_safe:
                raise ValueError(f"Path not in safe directories: {resolved_path}")

        return resolved_path

    def is_viewable_fits(
        self, fits_path: str, max_size_gb: float | None = None
    ) -> tuple[bool, str]:
        """Check if FITS file is viewable.

            Validates:
            - File exists and is readable
            - Valid FITS format
            - File size within limits
            - HDU[0] contains 2D+ data array

        Parameters
        ----------
        fits_path : str
            Path to FITS file
        max_size_gb : float, optional
            Maximum file size in GB (overrides config, default is None)

        Returns
        -------
            tuple
            (bool, str) indicating if viewable and reason if not
        """
        try:
            # Validate path security
            resolved_path = self._validate_and_resolve_path(fits_path)
            fits_path = str(resolved_path)
        except ValueError as e:
            return False, str(e)

        # Basic validation
        is_valid, error = validate_fits_file(fits_path)
        if not is_valid:
            return False, error

        # Check file size
        max_size = max_size_gb or self.config.max_file_size_gb
        size_gb = get_file_size_mb(fits_path) / 1024
        if size_gb > max_size:
            return False, f"File size ({size_gb:.2f} GB) exceeds limit ({max_size} GB)"

        # Check for 2D+ data
        try:
            with fits.open(fits_path) as hdul:
                # Find first HDU with data
                data_hdu = None
                for hdu in hdul:
                    if hdu.data is not None:
                        data_hdu = hdu
                        break

                if data_hdu is None:
                    return False, "No data found in FITS file"

                # Check dimensions
                if data_hdu.data.ndim < MIN_VIEWABLE_DIMENSIONS:
                    return (
                        False,
                        f"Data must be at least 2D (got {data_hdu.data.ndim}D)",
                    )

        except (OSError, TypeError, ValueError) as e:
            return False, f"Error reading FITS data: {e}"

        return True, ""

    def parse_fits_header(self, fits_path: str) -> FITSViewerMetadata:
        """Extract metadata from FITS file header.

        Parameters
        ----------
        fits_path : str
            Path to FITS file

        Returns
        -------
            dict
            Extracted metadata from FITS header
        """
        metadata = FITSViewerMetadata(
            filename=Path(fits_path).name,
            filepath=str(fits_path),
            file_size_mb=get_file_size_mb(fits_path),
        )

        try:
            # Validate path
            resolved_path = self._validate_and_resolve_path(fits_path)
            fits_path = str(resolved_path)
        except ValueError as exc:
            metadata.error_message = str(exc)
            return metadata

        try:
            with fits.open(fits_path) as hdul:
                # Find first HDU with data
                data_hdu = None
                for idx, hdu in enumerate(hdul):
                    if hdu.data is not None:
                        data_hdu = hdu
                        break

                if data_hdu is None:
                    metadata.error_message = "No data found in FITS file"
                    return metadata

                header = data_hdu.header
                metadata.has_data = True
                metadata.is_valid = True

                # Shape and axes
                if data_hdu.data is not None:
                    metadata.shape = tuple(data_hdu.data.shape)
                    metadata.naxis = data_hdu.data.ndim

                # Extract axis information
                naxis = header.get("NAXIS", 0)
                for i in range(1, naxis + 1):
                    ctype = header.get(f"CTYPE{i}", f"AXIS{i}")
                    cunit = header.get(f"CUNIT{i}", "")
                    cdelt = header.get(f"CDELT{i}", 0.0)

                    metadata.axes.append(ctype)
                    metadata.cunit.append(cunit)
                    metadata.cdelt.append(cdelt)

                # Resolution string
                if len(metadata.cdelt) >= 2:
                    res1 = format_resolution_degrees(abs(metadata.cdelt[0]))
                    res2 = format_resolution_degrees(abs(metadata.cdelt[1]))
                    if res1 == res2:
                        metadata.resolution = res1
                    else:
                        metadata.resolution = f"{res1} × {res2}"

                # Standard keywords
                metadata.date_obs = header.get("DATE-OBS", "")
                metadata.object = header.get("OBJECT", "")
                metadata.telescop = header.get("TELESCOP", "")
                metadata.instrume = header.get("INSTRUME", "")

                # Check WCS availability
                try:
                    wcs = WCS(header)
                    metadata.wcs_available = wcs.has_celestial
                except (ValueError, TypeError, AttributeError):
                    metadata.wcs_available = False

        except (OSError, TypeError, ValueError) as exc:
            metadata.error_message = f"Error parsing FITS header: {exc}"
            self.logger.warning("Failed to parse FITS header for %s: %s", fits_path, exc)

        return metadata

    def generate_js9_url(self, fits_path: str, colormap: str | None = None, **kwargs: Any) -> str:
        """Generate URL for JS9 viewer.

        Parameters
        ----------
        fits_path : str
            Path to FITS file
        colormap : str, optional
            Colormap to use (overrides config, default is None)
            **kwargs :
            Additional URL parameters

        Returns
        -------
            str
            Generated JS9 viewer URL
        """
        if not self.config.js9_enabled:
            return ""

        is_viewable, _ = self.is_viewable_fits(fits_path)
        if not is_viewable:
            return ""

        # Build base URL
        if self.config.js9_instance_url:
            base_url = self.config.js9_instance_url
        else:
            base_url = f"{self.config.server_base_url}/js9/viewer.html"

        # Build query parameters
        params = {
            "fits": fits_path,
            "colormap": colormap or self.config.colormap,
        }
        params.update(kwargs)

        return f"{base_url}?{urlencode(params)}"

    def generate_carta_url(self, fits_path: str, **kwargs: Any) -> str:
        """Generate URL for CARTA viewer.

        Parameters
        ----------
        fits_path : str
            Path to FITS file
            **kwargs :
            Additional URL parameters

        Returns
        -------
            str
            Generated CARTA viewer URL
        """
        if not self.config.carta_enabled:
            return ""

        is_viewable, _ = self.is_viewable_fits(fits_path)
        if not is_viewable:
            return ""

        # Build base URL
        if self.config.carta_instance_url:
            base_url = self.config.carta_instance_url
        else:
            base_url = f"{self.config.server_base_url}/carta"

        # Build query parameters
        params = {
            "file": fits_path,
        }
        params.update(kwargs)

        return f"{base_url}?{urlencode(params)}"

    def generate_aladin_url(self, fits_path: str, **kwargs: Any) -> str:
        """Generate URL for Aladin viewer.

        Parameters
        ----------
        fits_path : str
            Path to FITS file
            **kwargs : Any
            Additional URL parameters

        """
        if not self.config.aladin_enabled:
            return ""

        is_viewable, _ = self.is_viewable_fits(fits_path)
        if not is_viewable:
            return ""

        # Build base URL (Aladin Lite)
        if self.config.aladin_instance_url:
            base_url = self.config.aladin_instance_url
        else:
            base_url = f"{self.config.server_base_url}/aladin"

        # Build query parameters
        params = {
            "file": fits_path,
        }
        params.update(kwargs)

        return f"{base_url}?{urlencode(params)}"

    def get_viewer_urls(self, fits_path: str) -> dict[str, str]:
        """Get all available viewer URLs for a FITS file.

        Parameters
        ----------
        fits_path : str
            Path to FITS file

        """
        urls = {}

        if self.config.js9_enabled:
            js9_url = self.generate_js9_url(fits_path)
            if js9_url:
                urls["js9"] = js9_url

        if self.config.carta_enabled:
            carta_url = self.generate_carta_url(fits_path)
            if carta_url:
                urls["carta"] = carta_url

        if self.config.aladin_enabled:
            aladin_url = self.generate_aladin_url(fits_path)
            if aladin_url:
                urls["aladin"] = aladin_url

        return urls

    def get_viewer_buttons(
        self,
        fits_path: str,
        context: dict | None = None,
        include_inline: bool = False,
    ) -> str:
        """Generate HTML buttons for viewing FITS file.

        Parameters
        ----------
        fits_path : str
            Path to FITS file
        context : Optional[Dict], optional
            Optional context dictionary (e.g., report_id), by default None
        include_inline : bool, optional
            Include inline viewer (requires JS9), by default False

        """
        # Import template functions here to avoid circular imports
        from .fits_viewer_templates import render_inline_js9_viewer, render_viewer_button_group

        is_viewable, reason = self.is_viewable_fits(fits_path)
        if not is_viewable:
            return f'<span class="fits-viewer-unavailable">Viewer unavailable: {html.escape(reason)}</span>'

        metadata = self.parse_fits_header(fits_path)
        context = context or {}

        buttons_html = render_viewer_button_group(fits_path, metadata, self, context)
        if include_inline and self.config.js9_enabled:
            viewer_id = hashlib.sha256(fits_path.encode()).hexdigest()[:10]
            inline_html = render_inline_js9_viewer(
                fits_path,
                metadata,
                viewer_id=viewer_id,
                width=self.config.preview_size,
                height=self.config.preview_size,
            )
            return f"{buttons_html}\n{inline_html}"

        return buttons_html

    def create_viewer_cache_symlink(self, fits_path: str, prefix: str = "") -> str | None:
        """Create symlink in cache directory for web serving.

            This allows the web server to serve FITS files from a known
            location without exposing the full file path.

        Parameters
        ----------
        fits_path : str
            Path to original FITS file
        prefix : str, optional
            Optional prefix for symlink name, by default ""

        """
        if not self.config.enable_symlinks:
            return None

        try:
            resolved_path = self._validate_and_resolve_path(fits_path)
        except ValueError:
            return None

        if not resolved_path.exists():
            return None

        # Create unique symlink name
        hash_suffix = hashlib.sha256(str(resolved_path).encode()).hexdigest()[:8]
        symlink_name = f"{prefix}{resolved_path.stem}_{hash_suffix}{resolved_path.suffix}"
        symlink_path = Path(self.config.cache_dir) / symlink_name

        try:
            # Remove existing symlink if present
            if symlink_path.exists() or symlink_path.is_symlink():
                symlink_path.unlink()

            # Create new symlink
            symlink_path.symlink_to(resolved_path)

            return symlink_name
        except OSError as exc:
            self.logger.warning("Failed to create symlink for %s: %s", fits_path, exc)
            return None

    def cleanup_cache(self, max_age_hours: int = 24) -> int:
        """Clean up old cache entries.

        Parameters
        ----------
        max_age_hours :
            Maximum age of cache entries in hours
        max_age_hours : int :
            (Default value = 24)
        max_age_hours : int :
            (Default value = 24)
        """
        cache_dir = Path(self.config.cache_dir)
        if not cache_dir.exists():
            return 0

        max_age_seconds = max_age_hours * 3600
        current_time = time.time()
        removed = 0

        for entry in cache_dir.iterdir():
            try:
                if entry.is_symlink():
                    # Check symlink modification time
                    mtime = entry.lstat().st_mtime
                    if current_time - mtime > max_age_seconds:
                        entry.unlink()
                        removed += 1
            except OSError:
                pass

        return removed
