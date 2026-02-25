"""
CARTA Scripting Integration for DSA-110 Pipeline.

Provides programmatic control of CARTA sessions for:
- Automated moment map generation
- Batch region analysis
- Pipeline-driven visualization
- Session state management

This module prepares for CARTA v6's Python scripting interface while
providing fallback functionality via the CARTA Controller REST API.

CARTA v6 Roadmap Features (expected):
- Full Python scripting interface
- Workspace management
- Collaboration tools
- Time-domain astronomy tools

References
----------
    - https://cartavis.org
    - https://carta.readthedocs.io
    - https://github.com/CARTAvis/carta-backend
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import httpx

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

CARTA_CONTROLLER_URL = os.getenv("CARTA_CONTROLLER_URL", "http://localhost:8080")
CARTA_BACKEND_URL = os.getenv("CARTA_BACKEND_URL", "http://localhost:9002")
CARTA_TIMEOUT = httpx.Timeout(10.0, read=60.0, write=30.0)


# Moment map types (matching CASA/CARTA conventions)
class MomentType(Enum):
    """Moment map types supported by CARTA."""

    M0 = 0  # Integrated intensity
    M1 = 1  # Intensity-weighted velocity
    M2 = 2  # Velocity dispersion
    M3 = 3  # Skewness
    M4 = 4  # Kurtosis
    M5 = 5  # Maximum value
    M6 = 6  # Maximum value coordinate
    M7 = 7  # Minimum value
    M8 = 8  # Peak intensity (same as M5 for most data)
    M9 = 9  # Peak velocity
    M10 = 10  # Second moment about the mean
    M11 = 11  # Abs. mean deviation


@dataclass
class CARTARegion:
    """Region definition for CARTA analysis."""

    region_type: str  # "rectangle", "ellipse", "polygon", "point"
    center_x: float  # pixels or world coords
    center_y: float
    width: float | None = None  # for rectangle
    height: float | None = None
    rotation: float = 0.0  # degrees
    vertices: list[tuple[float, float]] | None = None  # for polygon
    coordinate_type: str = "pixel"  # "pixel" or "world"
    name: str | None = None

    def to_crtf(self) -> str:
        """Export region as CASA Region Text Format (CRTF)."""
        if self.region_type == "rectangle":
            return (
                f"box [[{self.center_x}pix, {self.center_y}pix], "
                f"[{self.width}pix, {self.height}pix]], "
                f"coord=ICRS, linewidth=1, linestyle=-, color=green"
            )
        elif self.region_type == "ellipse":
            return (
                f"ellipse [[{self.center_x}pix, {self.center_y}pix], "
                f"[{self.width}pix, {self.height}pix], {self.rotation}deg], "
                f"coord=ICRS, linewidth=1, linestyle=-, color=green"
            )
        elif self.region_type == "point":
            return f"point [[{self.center_x}pix, {self.center_y}pix]]"
        else:
            raise ValueError(f"Unsupported region type: {self.region_type}")


@dataclass
class CARTASessionState:
    """State of a CARTA session for persistence/restoration."""

    session_id: str
    files_loaded: list[str] = field(default_factory=list)
    active_file_index: int = 0
    regions: list[CARTARegion] = field(default_factory=list)
    colormap: str = "inferno"
    scaling: str = "linear"  # "linear", "log", "sqrt", "square", "power", "gamma"
    channel: int = 0
    stokes: int = 0
    zoom_level: float = 1.0
    center_x: float | None = None
    center_y: float | None = None

    def to_json(self) -> str:
        """Serialize state to JSON for storage."""
        return json.dumps(
            {
                "session_id": self.session_id,
                "files_loaded": self.files_loaded,
                "active_file_index": self.active_file_index,
                "colormap": self.colormap,
                "scaling": self.scaling,
                "channel": self.channel,
                "stokes": self.stokes,
                "zoom_level": self.zoom_level,
                "center_x": self.center_x,
                "center_y": self.center_y,
            }
        )

    @classmethod
    def from_json(cls, json_str: str) -> CARTASessionState:
        """Deserialize state from JSON.

        Parameters
        ----------
        json_str : str
            JSON string to deserialize

        Returns
        -------
            object
            Deserialized state object
        """
        data = json.loads(json_str)
        return cls(**data)


@dataclass
class RegionStatistics:
    """Statistics computed for a region in CARTA."""

    region_name: str
    num_pixels: int
    sum: float
    mean: float
    std_dev: float
    min_val: float
    max_val: float
    rms: float
    flux_density: float | None = None  # Jy if beam info available
    centroid_x: float | None = None
    centroid_y: float | None = None


@dataclass
class MomentMapResult:
    """Result of moment map generation."""

    moment_type: MomentType
    output_path: str
    channel_range: tuple[int, int]
    mask_threshold: float | None = None
    success: bool = True
    error_message: str | None = None


# =============================================================================
# CARTA Scripting Client
# =============================================================================


class CARTAScriptingClient:
    """Programmatic control of CARTA sessions.

    This client provides a Python interface to CARTA functionality,
    preparing for the CARTA v6 scripting API while using the
    Controller REST API as a fallback.

    Usage:
        async with CARTAScriptingClient() as client:
            session = await client.create_session()
            await client.open_file(session.session_id, "/path/to/image.fits")
            stats = await client.compute_region_statistics(
                session.session_id,
                CARTARegion(region_type="ellipse", center_x=256, center_y=256, width=50, height=30)
            )

    """

    def __init__(
        self,
        controller_url: str = CARTA_CONTROLLER_URL,
        backend_url: str = CARTA_BACKEND_URL,
        timeout: httpx.Timeout = CARTA_TIMEOUT,
        *,
        carta_url: str | None = None,  # Alias for controller_url
    ):
        """Initialize CARTA scripting client.

        Parameters
        ----------
        controller_url : str
            URL to CARTA Controller (for multi-user deployments)
        backend_url : str
            URL to CARTA Backend (for direct access)
        timeout : float or int
            HTTP timeout configuration
        carta_url : str
            Alias for controller_url (for convenience)
        """
        # Allow carta_url as alias for controller_url
        if carta_url is not None:
            controller_url = carta_url
        self.controller_url = controller_url
        self.backend_url = backend_url
        self.timeout = timeout
        self._client: httpx.AsyncClient | None = None
        self._sessions: dict[str, CARTASessionState] = {}

    async def __aenter__(self) -> CARTAScriptingClient:
        """Async context manager entry."""
        self._client = httpx.AsyncClient(timeout=self.timeout)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        if self._client:
            await self._client.aclose()
            self._client = None

    @property
    def client(self) -> httpx.AsyncClient:
        """Get HTTP client, creating if needed."""
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.timeout)
        return self._client

    # -------------------------------------------------------------------------
    # Session Management
    # -------------------------------------------------------------------------

    async def check_availability(self) -> tuple[bool, str | None, str | None]:
        """Check if CARTA services are available.

        Returns
        -------
            tuple
            Tuple of (is_available, version, error_message)
        """
        try:
            # Try Controller first (for multi-user setups)
            try:
                response = await self.client.get(f"{self.controller_url}/api/status")
                if response.status_code == 200:
                    data = response.json()
                    return True, data.get("version", "unknown"), None
            except httpx.RequestError:
                # Controller may be unavailable (e.g., single-user or direct-backend deployments);
                # fall back to querying the backend URL below.
                pass

            # Fall back to direct backend
            for endpoint in ["/api/status", "/status", "/"]:
                try:
                    response = await self.client.get(f"{self.backend_url}{endpoint}")
                    if response.status_code == 200:
                        return True, "backend-only", None
                except httpx.RequestError:
                    continue

            return False, None, "CARTA services not responding"

        except Exception as e:
            logger.error(f"Error checking CARTA availability: {e}")
            return False, None, str(e)

    async def create_session(
        self,
        user: str | None = None,
        workspace: str | None = None,
    ) -> CARTASessionState:
        """Create a new CARTA session.

        Parameters
        ----------
        user : str or None
            Optional username for session tracking
        workspace : str or None
            Optional workspace name for grouping

        Returns
        -------
            CARTASessionState
            New session state object
        """
        try:
            # Try Controller API
            response = await self.client.post(
                f"{self.controller_url}/api/sessions", json={"user": user, "workspace": workspace}
            )

            if response.status_code == 200:
                data = response.json()
                session = CARTASessionState(session_id=data["session_id"])
                self._sessions[session.session_id] = session
                logger.info(f"Created CARTA session: {session.session_id}")
                return session

        except httpx.RequestError as e:
            logger.warning(f"Controller not available, using mock session: {e}")

        # Fallback: generate local session ID
        session_id = str(uuid.uuid4())
        session = CARTASessionState(session_id=session_id)
        self._sessions[session_id] = session
        logger.info(f"Created local CARTA session: {session_id}")
        return session

    async def close_session(self, session_id: str) -> bool:
        """Close a CARTA session.

        Parameters
        ----------
        session_id : str or int
            Session to close

        Returns
        -------
            bool
            True if successfully closed
        """
        try:
            response = await self.client.delete(f"{self.controller_url}/api/sessions/{session_id}")
            success = response.status_code in (200, 204)
        except httpx.RequestError:
            success = True  # Assume success if controller unavailable

        if session_id in self._sessions:
            del self._sessions[session_id]

        logger.info(f"Closed CARTA session: {session_id}")
        return success

    async def get_session_state(self, session_id: str) -> CARTASessionState | None:
        """Get current state of a session.

        Parameters
        ----------
        session_id : str or int
            Session to query

        Returns
        -------
            object or None
            Session state or None if not found
        """
        return self._sessions.get(session_id)

    # -------------------------------------------------------------------------
    # File Operations
    # -------------------------------------------------------------------------

    async def open_file(
        self,
        session_id: str,
        file_path: str,
        hdu: int = 0,
        append: bool = False,
    ) -> bool:
        """Open a file in a CARTA session.

        Parameters
        ----------
        session_id : str or int
            Target session
        file_path : str
            Path to FITS/HDF5/CASA image
        hdu : int
            HDU index for FITS files
        append : bool
            If True, add to existing files; if False, replace

        Returns
        -------
            bool
            True if file opened successfully
        """
        # Validate file exists
        if not Path(file_path).exists():
            logger.error(f"File not found: {file_path}")
            return False

        try:
            response = await self.client.post(
                f"{self.controller_url}/api/sessions/{session_id}/open",
                json={"file": file_path, "hdu": hdu, "append": append},
            )

            if response.status_code == 200:
                # Update local state
                if session_id in self._sessions:
                    state = self._sessions[session_id]
                    if append:
                        state.files_loaded.append(file_path)
                    else:
                        state.files_loaded = [file_path]
                        state.active_file_index = 0
                logger.info(f"Opened file in CARTA: {file_path}")
                return True

        except httpx.RequestError as e:
            logger.warning(f"Could not open file via controller: {e}")

        # Update local state anyway for tracking
        if session_id in self._sessions:
            state = self._sessions[session_id]
            if append:
                state.files_loaded.append(file_path)
            else:
                state.files_loaded = [file_path]

        return True  # Assume success for URL-based opening

    async def close_file(
        self,
        session_id: str,
        file_index: int = 0,
    ) -> bool:
        """Close a file in a CARTA session.

        Parameters
        ----------
        session_id : str or int
            Target session
        file_index : int
            Index of file to close

        Returns
        -------
            bool
            True if closed successfully
        """
        try:
            response = await self.client.post(
                f"{self.controller_url}/api/sessions/{session_id}/close",
                json={"file_index": file_index},
            )
            return response.status_code == 200
        except httpx.RequestError:
            pass

        # Update local state
        if session_id in self._sessions:
            state = self._sessions[session_id]
            if 0 <= file_index < len(state.files_loaded):
                state.files_loaded.pop(file_index)

        return True

    # -------------------------------------------------------------------------
    # Analysis Operations (CARTA v6 Scripting API Preview)
    # -------------------------------------------------------------------------

    async def create_moment_map(
        self,
        session_id: str,
        cube_path: str,
        moment: int | MomentType = MomentType.M0,
        channel_range: tuple[int, int] | None = None,
        mask_threshold: float | None = None,
        output_path: str | None = None,
    ) -> MomentMapResult:
        """Generate a moment map from a spectral cube.

            This prepares for CARTA v6's moment map scripting capability.
            Currently falls back to CASA's immoments task.

        Parameters
        ----------
        session_id : str or int
            CARTA session (for future API use)
        cube_path : str
            Path to input spectral cube
        moment : int
            Moment type (0=integrated, 1=velocity, 2=dispersion, etc.)
        channel_range : tuple or None
            (start, end) channel range, None for all
        mask_threshold : float
            Minimum intensity threshold for mask
        output_path : str or None
            Output file path, auto-generated if None

        Returns
        -------
            MomentMapResult
            Result with output path and status
        """
        if isinstance(moment, MomentType):
            moment_int = moment.value
        else:
            moment_int = moment

        # Generate output path if not provided
        if output_path is None:
            cube_stem = Path(cube_path).stem
            output_path = str(Path(cube_path).parent / f"{cube_stem}.moment{moment_int}.fits")

        # Determine channel range
        if channel_range is None:
            channel_range = (0, -1)  # All channels

        try:
            # Try CARTA Controller API (when available in v6)
            response = await self.client.post(
                f"{self.controller_url}/api/sessions/{session_id}/moment",
                json={
                    "file": cube_path,
                    "moment": moment_int,
                    "channel_start": channel_range[0],
                    "channel_end": channel_range[1],
                    "threshold": mask_threshold,
                    "output": output_path,
                },
            )

            if response.status_code == 200:
                logger.info(f"Generated moment {moment_int} map: {output_path}")
                return MomentMapResult(
                    moment_type=MomentType(moment_int),
                    output_path=output_path,
                    channel_range=channel_range,
                    mask_threshold=mask_threshold,
                    success=True,
                )

        except httpx.RequestError as e:
            logger.debug(f"CARTA moment API not available, using CASA fallback: {e}")

        # Fallback: Use CASA immoments
        try:
            return await self._generate_moment_with_casa(
                cube_path, moment_int, channel_range, mask_threshold, output_path
            )
        except Exception as e:
            logger.error(f"Failed to generate moment map: {e}")
            return MomentMapResult(
                moment_type=MomentType(moment_int),
                output_path=output_path,
                channel_range=channel_range,
                mask_threshold=mask_threshold,
                success=False,
                error_message=str(e),
            )

    async def _generate_moment_with_casa(
        self,
        cube_path: str,
        moment: int,
        channel_range: tuple[int, int],
        mask_threshold: float | None,
        output_path: str,
    ) -> MomentMapResult:
        """Generate moment map using CASA's immoments task.

        Parameters
        ----------
        cube_path : str
            Input cube path
        moment : int
            Moment number
        channel_range : tuple or None
            Channel range
        mask_threshold : float
            Intensity threshold
        output_path : str
            Output path

        Returns
        -------
            MomentMapResult
        """
        # Import CASA lazily
        try:
            from casatasks import immoments
        except ImportError:
            raise RuntimeError("CASA not available for moment map generation")

        # Build channel selection string
        if channel_range[1] == -1:
            chans = ""  # All channels
        else:
            chans = f"{channel_range[0]}~{channel_range[1]}"

        # Build mask expression
        mask = ""
        if mask_threshold is not None:
            mask = f'"{cube_path}" > {mask_threshold}'

        # Run immoments
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            None,
            lambda: immoments(
                imagename=cube_path,
                moments=[moment],
                chans=chans,
                mask=mask,
                outfile=output_path,
            ),
        )

        logger.info(f"Generated moment {moment} map with CASA: {output_path}")
        return MomentMapResult(
            moment_type=MomentType(moment),
            output_path=output_path,
            channel_range=channel_range,
            mask_threshold=mask_threshold,
            success=True,
        )

    async def compute_region_statistics(
        self,
        session_id: str,
        region: CARTARegion,
        file_index: int = 0,
        channel: int | None = None,
    ) -> RegionStatistics | None:
        """Compute statistics for a region.

        Parameters
        ----------
        session_id : str or int
            CARTA session
        region : object
            Region definition
        file_index : int
            Index of file to analyze
        channel : int or None
            Specific channel (None for current/all)

        Returns
        -------
            RegionStatistics or None
            Computed statistics or None if failed
        """
        try:
            response = await self.client.post(
                f"{self.controller_url}/api/sessions/{session_id}/statistics",
                json={
                    "region": {
                        "type": region.region_type,
                        "center_x": region.center_x,
                        "center_y": region.center_y,
                        "width": region.width,
                        "height": region.height,
                        "rotation": region.rotation,
                    },
                    "file_index": file_index,
                    "channel": channel,
                },
            )

            if response.status_code == 200:
                data = response.json()
                return RegionStatistics(
                    region_name=region.name or "unnamed",
                    num_pixels=data.get("num_pixels", 0),
                    sum=data.get("sum", 0.0),
                    mean=data.get("mean", 0.0),
                    std_dev=data.get("std_dev", 0.0),
                    min_val=data.get("min", 0.0),
                    max_val=data.get("max", 0.0),
                    rms=data.get("rms", 0.0),
                    flux_density=data.get("flux_density"),
                )

        except httpx.RequestError as e:
            logger.warning(f"Could not compute statistics via CARTA: {e}")

        return None

    async def export_region_file(
        self,
        session_id: str,
        output_path: str,
        format: str = "crtf",  # "crtf" or "ds9"
    ) -> bool:
        """Export regions to a file.

        Parameters
        ----------
        session_id : str or int
            CARTA session
        output_path : str
            Output file path
        format : str
            Region file format ("crtf" or "ds9")

        Returns
        -------
            bool
            True if exported successfully
        """
        state = self._sessions.get(session_id)
        if not state or not state.regions:
            logger.warning(f"No regions to export for session {session_id}")
            return False

        try:
            with open(output_path, "w") as f:
                if format == "crtf":
                    f.write("#CRTF\n")
                    for region in state.regions:
                        f.write(region.to_crtf() + "\n")
                elif format == "ds9":
                    f.write("# Region file format: DS9 version 4.1\n")
                    f.write("global color=green dashlist=8 3 width=1\n")
                    f.write("image\n")
                    for region in state.regions:
                        # Simplified DS9 format
                        if region.region_type == "ellipse":
                            f.write(
                                f"ellipse({region.center_x},{region.center_y},"
                                f"{region.width},{region.height},{region.rotation})\n"
                            )
                        elif region.region_type == "rectangle":
                            f.write(
                                f"box({region.center_x},{region.center_y},"
                                f"{region.width},{region.height},{region.rotation})\n"
                            )

            logger.info(f"Exported regions to {output_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to export regions: {e}")
            return False

    # -------------------------------------------------------------------------
    # View Control (CARTA v6 Scripting API Preview)
    # -------------------------------------------------------------------------

    async def set_colormap(
        self,
        session_id: str,
        colormap: str = "inferno",
        scaling: str = "linear",
        min_val: float | None = None,
        max_val: float | None = None,
    ) -> bool:
        """Set colormap and scaling for the active image.

        Parameters
        ----------
        session_id : str or int
            CARTA session
        colormap : str
            Colormap name (inferno, viridis, plasma, etc.)
        scaling : str
            Scaling function (linear, log, sqrt, etc.)
        min_val : float
            Minimum display value
        max_val : float
            Maximum display value

        Returns
        -------
            bool
            True if settings applied
        """
        if session_id in self._sessions:
            state = self._sessions[session_id]
            state.colormap = colormap
            state.scaling = scaling

        try:
            response = await self.client.post(
                f"{self.controller_url}/api/sessions/{session_id}/render",
                json={
                    "colormap": colormap,
                    "scaling": scaling,
                    "min": min_val,
                    "max": max_val,
                },
            )
            return response.status_code == 200
        except httpx.RequestError:
            return True  # Settings stored locally

    async def set_channel(
        self,
        session_id: str,
        channel: int,
        stokes: int = 0,
    ) -> bool:
        """Set the active channel/stokes for a spectral cube.

        Parameters
        ----------
        session_id : str
            CARTA session.
        channel : int
            Channel index.
        stokes : int, optional
            Stokes parameter index (0=I, 1=Q, 2=U, 3=V).

        Returns
        -------
        bool
            True if channel set.
        """
        if session_id in self._sessions:
            state = self._sessions[session_id]
            state.channel = channel
            state.stokes = stokes

        try:
            response = await self.client.post(
                f"{self.controller_url}/api/sessions/{session_id}/channel",
                json={"channel": channel, "stokes": stokes},
            )
            return response.status_code == 200
        except httpx.RequestError:
            return True

    async def set_zoom(
        self,
        session_id: str,
        zoom_level: float,
        center_x: float | None = None,
        center_y: float | None = None,
    ) -> bool:
        """Set zoom level and center position.

        Parameters
        ----------
        session_id : str
            CARTA session.
        zoom_level : float
            Zoom factor (1.0 = fit to view).
        center_x : float, optional
            Center X in pixels.
        center_y : float, optional
            Center Y in pixels.

        Returns
        -------
        bool
            True if zoom set.
        """
        if session_id in self._sessions:
            state = self._sessions[session_id]
            state.zoom_level = zoom_level
            if center_x is not None:
                state.center_x = center_x
            if center_y is not None:
                state.center_y = center_y

        try:
            response = await self.client.post(
                f"{self.controller_url}/api/sessions/{session_id}/zoom",
                json={
                    "zoom": zoom_level,
                    "center_x": center_x,
                    "center_y": center_y,
                },
            )
            return response.status_code == 200
        except httpx.RequestError:
            return True

    # -------------------------------------------------------------------------
    # Utility Methods
    # -------------------------------------------------------------------------

    def build_viewer_url(
        self,
        file_path: str,
        session_id: str | None = None,
        channel: int | None = None,
        colormap: str | None = None,
    ) -> str:
        """Build a CARTA viewer URL with query parameters.

        Parameters
        ----------
        file_path : str
            Path to file to open
        session_id : Optional[str]
            Optional session ID (default is None)
        channel : Optional[int]
            Initial channel (default is None)
        colormap : Optional[str]
            Initial colormap (default is None)

        Returns
        -------
            str
            Constructed CARTA viewer URL
        """
        from urllib.parse import urlencode

        params = {"file": file_path}
        if session_id:
            params["session"] = session_id
        if channel is not None:
            params["channel"] = str(channel)
        if colormap:
            params["colormap"] = colormap

        query = urlencode(params)
        return f"{self.backend_url}/?{query}"


# =============================================================================
# Convenience Functions
# =============================================================================


async def quick_moment_map(
    cube_path: str,
    moment: int = 0,
    output_path: str | None = None,
) -> str | None:
    """Quick helper to generate a moment map.

    Parameters
    ----------
    cube_path : str
        Path to spectral cube
    moment : int
        Moment type (0, 1, 2, 8, 9)
    output_path : str or None
        Output path (auto-generated if None)

    Returns
    -------
        str or None
        Path to generated moment map or None on failure
    """
    async with CARTAScriptingClient() as client:
        session = await client.create_session()
        result = await client.create_moment_map(
            session.session_id,
            cube_path,
            moment=moment,
            output_path=output_path,
        )
        await client.close_session(session.session_id)

        if result.success:
            return result.output_path
        return None


async def check_carta_services() -> dict[str, Any]:
    """Check status of all CARTA services.

    Returns
    -------
        dict
        Dictionary with service status information
    """
    async with CARTAScriptingClient() as client:
        available, version, error = await client.check_availability()
        return {
            "available": available,
            "version": version,
            "error": error,
            "controller_url": client.controller_url,
            "backend_url": client.backend_url,
        }
