"""Mermaid diagram rendering utilities.

Provides a unified interface for rendering Mermaid diagrams to SVG/PNG using:
1. Local Node.js renderer (if available) - supports custom icons/fonts
2. Mermaid.ink API (fallback) - no dependencies required

"""

import base64
import logging
import os
import subprocess
import sys
import urllib.request
import urllib.parse
from pathlib import Path

logger = logging.getLogger(__name__)


class MermaidRenderer:
    """Renderer for Mermaid diagrams.

    Troubleshooting:
        If local rendering fails on Linux (Puppeteer/Chrome launch errors), install dependencies:

        sudo apt-get update && sudo apt-get install -y \\
            ca-certificates fonts-liberation libasound2 libatk-bridge2.0-0 libatk1.0-0 libc6 \\
            libcairo2 libcups2 libdbus-1-3 libexpat1 libfontconfig1 libgbm1 libgcc1 \\
            libglib2.0-0 libgtk-3-0 libnspr4 libnss3 libpango-1.0-0 libpangocairo-1.0-0 \\
            libstdc++6 libx11-6 libx11-xcb1 libxcb1 libxcomposite1 libxcursor1 libxdamage1 \\
            libxext6 libxfixes3 libxi6 libxrandr2 libxrender1 libxss1 libxtst6 lsb-release \\
            wget xdg-utils
    """

    def __init__(self, node_script_path: str | Path | None = None):
        """Initialize renderer.

        Parameters
        ----------
        node_script_path :
            Path to the local rendering script (render_mermaid_local.js).
            If None, attempts to auto-detect it.
        """
        self.node_script_path = self._resolve_script_path(node_script_path)

    def _resolve_script_path(self, provided_path: str | Path | None) -> Path | None:
        """Resolve path to local rendering script."""
        if provided_path:
            path = Path(provided_path)
            if path.exists():
                return path
            logger.warning("Provided renderer script not found: %s", path)

        # Common locations to check
        candidates = [
            # Environment variable
            os.environ.get("DSA110_MERMAID_RENDERER"),
            # Relative to this file (in source tree)
            Path(__file__).parents[5] / "scripts" / "validation" / "render_mermaid_local.js",
            # Relative to CWD
            Path("scripts/validation/render_mermaid_local.js"),
            # Absolute default
            Path("/data/dsa110-contimg/scripts/validation/render_mermaid_local.js"),
        ]

        for candidate in candidates:
            if candidate:
                path = Path(candidate)
                if path.exists():
                    logger.debug("Found local Mermaid renderer: %s", path)
                    return path

        return None

    def render(self, mermaid_code: str, output_path: str | Path) -> bool:
        """Render Mermaid code to a file.

        Parameters
        ----------
        mermaid_code : str
            The Mermaid diagram definition.
        output_path : str | Path
            Path to save the rendered image (SVG).

        Returns
        -------
        bool
            True if rendering succeeded, False otherwise.
        """
        output_path = Path(output_path)
        
        # Always save the source .mmd file for reference/debugging
        mmd_path = output_path.with_suffix(".mmd")
        try:
            with open(mmd_path, "w", encoding="utf-8") as f:
                f.write(mermaid_code)
        except IOError as e:
            logger.error("Failed to save Mermaid source file %s: %s", mmd_path, e)
            # Continue trying to render even if source save fails

        # Try local renderer first
        if self.node_script_path:
            if self._render_local(mmd_path, output_path):
                return True
            logger.warning("Local rendering failed, falling back to mermaid.ink")
        else:
            logger.info("Local renderer not found, using mermaid.ink")

        # Fallback to remote API
        return self._render_remote(mermaid_code, output_path)

    def _render_local(self, input_path: Path, output_path: Path) -> bool:
        """Render using local Node.js script."""
        if not self.node_script_path:
            return False

        try:
            logger.info("Rendering with local script: %s", self.node_script_path)
            result = subprocess.run(
                ["node", str(self.node_script_path), str(input_path), str(output_path)],
                capture_output=True,
                text=True,
                timeout=60,
            )

            if result.returncode == 0:
                logger.info("Successfully rendered to %s", output_path)
                return True
            
            logger.warning("Local renderer error: %s", result.stderr)
            return False

        except (subprocess.SubprocessError, OSError) as e:
            logger.warning("Failed to execute local renderer: %s", e)
            return False

    def _render_remote(self, mermaid_code: str, output_path: Path) -> bool:
        """Render using mermaid.ink API."""
        logger.info("Rendering via mermaid.ink API...")
        
        mermaid_bytes = mermaid_code.encode("utf-8")
        encoded = base64.urlsafe_b64encode(mermaid_bytes).decode("utf-8").rstrip("=")

        # Check for URL length limits (approx 8KB is safe for most browsers/proxies, 
        # though mermaid.ink might handle more)
        if len(encoded) > 8000:
            logger.error("Diagram too large for mermaid.ink API (%d chars)", len(encoded))
            return False

        url = f"https://mermaid.ink/svg/{encoded}"

        try:
            req = urllib.request.Request(url)
            req.add_header("User-Agent", "dsa110-contimg/1.0")

            with urllib.request.urlopen(req, timeout=30) as response:
                svg_content = response.read().decode("utf-8")

            with open(output_path, "w", encoding="utf-8") as f:
                f.write(svg_content)

            logger.info("Successfully rendered to %s", output_path)
            return True

        except Exception as e:
            logger.error("Remote rendering failed: %s", e)
            return False
