"""
Plot context for automatic format selection.

Defines contexts where different plot formats are optimal, enabling
smart defaults while maintaining backward compatibility.
"""

import logging
import time
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)


class PlotContext(str, Enum):
    """Context in which a plot is being generated.

    This determines the optimal output format:
    - INTERACTIVE contexts → Vega-Lite JSON (web dashboards, APIs)
    - STATIC contexts → PNG/PDF (reports, publications, archives)

    """

    # Interactive contexts (→ Vega-Lite)
    WEB_API = "web_api"  # Backend API serving frontend
    DASHBOARD = "dashboard"  # Web dashboard display
    INTERACTIVE = "interactive"  # Explicit interactive request

    # Static contexts (→ PNG/PDF)
    REPORT = "report"  # HTML/PDF QA reports
    ARCHIVE = "archive"  # Long-term storage
    PUBLICATION = "publication"  # Papers, presentations
    BATCH = "batch"  # Batch processing
    EMAIL = "email"  # Email notifications
    SLACK = "slack"  # Slack attachments

    # Legacy/Default
    DEFAULT = "default"  # Legacy behavior (PNG)


def should_generate_interactive(
    context: PlotContext | None = None,
    interactive: bool | None = None,
) -> bool:
    """Determine whether to generate interactive (Vega-Lite) or static (PNG) plot.

    Priority order:
    1. Explicit `interactive` parameter (if provided)
    2. Context-based heuristic
    3. Default to static (PNG) for backward compatibility

    Parameters
    ----------
    context :
        Usage context for the plot
    interactive :
        Explicit format override (True=Vega, False=PNG)
    context : Optional[PlotContext] :
        (Default value = None)
    interactive : Optional[bool] :
        (Default value = None)
    context : Optional[PlotContext] :
        (Default value = None)
    interactive : Optional[bool] :
        (Default value = None)
    context: Optional[PlotContext] :
         (Default value = None)
    interactive: Optional[bool] :
         (Default value = None)

    Returns
    -------
    >>> # Explicit override always wins
        >>> should_generate_interactive(PlotContext.REPORT, interactive=True)
        True
        >>> should_generate_interactive(PlotContext.WEB_API, interactive=False)
        False

        >>> # Context-based smart defaults
        >>> should_generate_interactive(PlotContext.WEB_API)
        True
        >>> should_generate_interactive(PlotContext.DASHBOARD)
        True
        >>> should_generate_interactive(PlotContext.REPORT)
        False
        >>> should_generate_interactive(PlotContext.ARCHIVE)
        False

        >>> # Legacy behavior (PNG) when no context specified
        >>> should_generate_interactive()
        False
    """
    # Explicit override takes precedence
    if interactive is not None:
        return interactive

    # Context-based heuristics
    if context is None:
        return False  # Default to static for backward compatibility

    # Interactive contexts
    if context in (PlotContext.WEB_API, PlotContext.DASHBOARD, PlotContext.INTERACTIVE):
        return True

    # Static contexts
    return False


def get_file_extension(
    context: PlotContext | None = None,
    interactive: bool | None = None,
) -> str:
    """Get appropriate file extension based on context/format.

    Parameters
    ----------
    context :
        Usage context
    interactive :
        Explicit format override
    context : Optional[PlotContext] :
        (Default value = None)
    interactive : Optional[bool] :
        (Default value = None)
    context : Optional[PlotContext] :
        (Default value = None)
    interactive : Optional[bool] :
        (Default value = None)
    context: Optional[PlotContext] :
         (Default value = None)
    interactive: Optional[bool] :
         (Default value = None)

    Returns
    -------
    >>> get_file_extension(PlotContext.WEB_API)
        '.vega.json'
        >>> get_file_extension(PlotContext.REPORT)
        '.png'
        >>> get_file_extension(interactive=True)
        '.vega.json'
    """
    if should_generate_interactive(context, interactive):
        return ".vega.json"
    return ".png"


def detect_context_from_path(output_path: str | Path) -> PlotContext | None:
    """Automatically detect plot context from output path.

        This enables automatic format selection without explicit context parameter.

        Path patterns:
        - /dashboard/*, /web/*, /api/* → WEB_API
        - /reports/*, /qa/* → REPORT
        - /archive/*, /storage/* → ARCHIVE
        - /batch/*, /stage/* → BATCH
        - *.vega.json → INTERACTIVE (explicit extension)

    Parameters
    ----------
    output_path : Union[str, Path]
        Path where plot will be saved

    Examples
    --------
        >>> detect_context_from_path("/dashboard/plots/rfi.png")
        PlotContext.WEB_API
        >>> detect_context_from_path("/stage/dsa110-contimg/debug/plot.png")
        PlotContext.BATCH
        >>> detect_context_from_path("/reports/qa_report.png")
        PlotContext.REPORT
        >>> detect_context_from_path("output.vega.json")
        PlotContext.INTERACTIVE
    """
    path_str = str(output_path).lower()
    path_parts = Path(path_str).parts

    # Check for explicit Vega-Lite extension
    if path_str.endswith(".vega.json"):
        return PlotContext.INTERACTIVE

    # Check path components for context hints
    for part in path_parts:
        # Web/API contexts
        if part in ("dashboard", "web", "api", "ui", "frontend"):
            return PlotContext.WEB_API

        # Report contexts
        if part in ("reports", "qa", "quality", "validation"):
            return PlotContext.REPORT

        # Archive contexts
        if part in ("archive", "storage", "backup", "historical"):
            return PlotContext.ARCHIVE

        # Batch contexts
        if part in ("batch", "tmp", "temp", "scratch", "processing", "stage"):
            return PlotContext.BATCH

        # Publication contexts
        if part in ("publication", "paper", "manuscript", "figures"):
            return PlotContext.PUBLICATION

    return None


class PerformanceLogger:
    """Performance logger for plot generation tracking.

    Logs format selection decisions, file sizes, and generation time
    for production monitoring and optimization.

    """

    def __init__(
        self,
        plot_type: str,
        output_path: str | Path,
        context: PlotContext | None = None,
        interactive: bool | None = None,
    ):
        """Initialize performance logger.

        Parameters
        ----------
        plot_type : str
            Type of plot (e.g., "rfi_spectrum", "psf_correlation")
        output_path : str
            Output file path
        context : Any
            Plot context
        interactive : Any
            Interactive override
        """
        self.plot_type = plot_type
        self.output_path = Path(output_path)
        self.context = context
        self.interactive = interactive
        self.start_time = time.time()
        self.detected_context: PlotContext | None = None

    def log_format_selection(self, format_type: str, reason: str) -> None:
        """Log format selection decision.

        Parameters
        ----------
        format_type : str
            "vega-lite" or "png"
        reason : str
            Reason for selection (e.g., "explicit_override", "context_heuristic")

        """
        logger.info(
            f"Plot format selection: {self.plot_type} → {format_type} "
            f"(reason={reason}, context={self.context}, "
            f"interactive={self.interactive})"
        )

    def log_completion(self, actual_output_path: Path | None = None) -> None:
        """Log plot generation completion with metrics.

        Parameters
        ----------
        actual_output_path :
            Actual output path (may differ due to extension change)
        actual_output_path : Optional[Path] :
            (Default value = None)
        actual_output_path : Optional[Path] :
            (Default value = None)
        actual_output_path: Optional[Path] :
             (Default value = None)

        """
        elapsed = time.time() - self.start_time

        # Use actual path if provided, otherwise use original
        final_path = actual_output_path or self.output_path

        # Get file size if exists
        size_kb = 0.0
        if final_path.exists():
            size_kb = final_path.stat().st_size / 1024

        format_type = "vega-lite" if str(final_path).endswith(".vega.json") else "png"

        logger.info(
            f"Plot generation complete: {self.plot_type} → {final_path.name} "
            f"(format={format_type}, size={size_kb:.1f}KB, time={elapsed:.2f}s, "
            f"context={self.context})"
        )

    def __enter__(self):
        """Context manager entry."""
        # Auto-detect context if not provided
        if self.context is None and self.interactive is None:
            self.detected_context = detect_context_from_path(self.output_path)
            if self.detected_context:
                logger.debug(
                    f"Auto-detected context: {self.detected_context} from path: {self.output_path}"
                )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if exc_type is None:
            self.log_completion()
        else:
            logger.warning(
                f"Plot generation failed: {self.plot_type} "
                f"(error={exc_type.__name__}, context={self.context})"
            )
        return False


__all__ = [
    "PlotContext",
    "should_generate_interactive",
    "get_file_extension",
    "detect_context_from_path",
    "PerformanceLogger",
]
