"""
HTML and CSS template generation for FITS viewer buttons and inline viewers.

Provides:
- Viewer button group rendering
- Inline JS9 viewer embedding
- CSS styling for viewer elements

All HTML is properly escaped to prevent XSS attacks.
"""

from __future__ import annotations

import html
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .fits_viewer import FITSViewerMetadata, FITSViewerManager


FITS_VIEWER_CSS = """
<style>
/* FITS Viewer Button Styles */
.fits-viewer-buttons {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin: 10px 0;
    align-items: center;
}

.fits-viewer-buttons .btn {
    display: inline-flex;
    align-items: center;
    padding: 6px 12px;
    font-size: 13px;
    font-weight: 500;
    color: #333;
    background-color: #f8f9fa;
    border: 1px solid #ddd;
    border-radius: 4px;
    cursor: pointer;
    text-decoration: none;
    transition: all 0.15s ease-in-out;
}

.fits-viewer-buttons .btn:hover {
    background-color: #e9ecef;
    border-color: #adb5bd;
}

.fits-viewer-buttons .btn:active {
    background-color: #dee2e6;
}

.fits-viewer-buttons .btn-js9 {
    border-left: 3px solid #007bff;
}

.fits-viewer-buttons .btn-js9:hover {
    background-color: #e7f1ff;
    border-color: #007bff;
}

.fits-viewer-buttons .btn-carta {
    border-left: 3px solid #dc3545;
}

.fits-viewer-buttons .btn-carta:hover {
    background-color: #ffedef;
    border-color: #dc3545;
}

.fits-viewer-buttons .btn-aladin {
    border-left: 3px solid #28a745;
}

.fits-viewer-buttons .btn-aladin:hover {
    background-color: #e9f7ec;
    border-color: #28a745;
}

.fits-viewer-buttons .btn-icon {
    margin-right: 6px;
    font-size: 14px;
}

.fits-viewer-buttons .metadata-info {
    font-size: 12px;
    color: #6c757d;
    margin-left: 10px;
}

.fits-viewer-unavailable {
    color: #6c757d;
    font-style: italic;
    font-size: 13px;
}

/* Inline viewer container */
.js9-inline-viewer {
    border: 1px solid #ddd;
    border-radius: 4px;
    margin: 10px 0;
    padding: 5px;
    background-color: #1a1a1a;
}

.js9-inline-viewer .js9-menubar {
    background-color: #2d2d2d;
    border-bottom: 1px solid #444;
}

.js9-inline-viewer .js9-canvas-container {
    background-color: #1a1a1a;
}

/* FITS image block in reports */
.fits-image-block {
    margin: 20px 0;
    padding: 15px;
    background-color: #f8f9fa;
    border-radius: 8px;
    border: 1px solid #e9ecef;
}

.fits-image-block img {
    max-width: 100%;
    height: auto;
    border-radius: 4px;
}

.fits-image-block .fits-viewer-buttons {
    margin-top: 10px;
    padding-top: 10px;
    border-top: 1px solid #e9ecef;
}

/* Tooltip styles for metadata */
.fits-tooltip {
    position: relative;
    display: inline-block;
}

.fits-tooltip .fits-tooltip-text {
    visibility: hidden;
    width: 250px;
    background-color: #333;
    color: #fff;
    text-align: left;
    border-radius: 6px;
    padding: 10px;
    position: absolute;
    z-index: 1000;
    bottom: 125%;
    left: 50%;
    margin-left: -125px;
    opacity: 0;
    transition: opacity 0.3s;
    font-size: 12px;
    line-height: 1.4;
}

.fits-tooltip .fits-tooltip-text::after {
    content: "";
    position: absolute;
    top: 100%;
    left: 50%;
    margin-left: -5px;
    border-width: 5px;
    border-style: solid;
    border-color: #333 transparent transparent transparent;
}

.fits-tooltip:hover .fits-tooltip-text {
    visibility: visible;
    opacity: 1;
}

/* Dark mode support */
@media (prefers-color-scheme: dark) {
    .fits-image-block {
        background-color: #2d2d2d;
        border-color: #444;
    }

    .fits-viewer-buttons .btn {
        background-color: #3d3d3d;
        border-color: #555;
        color: #e0e0e0;
    }

    .fits-viewer-buttons .btn:hover {
        background-color: #4d4d4d;
    }

    .fits-viewer-unavailable {
        color: #999;
    }
}
</style>
"""


def get_css_styles() -> str:
    """Get CSS styling for viewer buttons and elements.

    Returns
    -------
        CSS string for FITS viewer components

    """
    return FITS_VIEWER_CSS


def _render_viewer_icon(viewer_type: str) -> str:
    """Get icon HTML for viewer type.

    Parameters
    ----------
    viewer_type :
        Type of viewer (js9, carta, aladin)

    Returns
    -------
        HTML span with icon

    """
    icons = {
        "js9": "",  # telescope
        "carta": "",  # framed picture
        "aladin": "",  # galaxy
    }
    icon = icons.get(viewer_type, "")
    return f'<span class="btn-icon">{icon}</span>'


def render_viewer_button(
    viewer_type: str,
    url: str,
    label: str = "",
    title: str = "",
    target: str = "_blank",
) -> str:
    """Render a single viewer button.

    Parameters
    ----------
    viewer_type :
        Type of viewer (js9, carta, aladin)
    url :
        URL to open
    label :
        Button label text
    title :
        Tooltip text
    target :
        Link target (_blank, _self, etc.)

    Returns
    -------
        HTML button string

    """
    if not label:
        label = viewer_type.upper()

    icon = _render_viewer_icon(viewer_type)
    escaped_url = html.escape(url)
    escaped_label = html.escape(label)
    escaped_title = html.escape(title)

    return f"""<a href="{escaped_url}"
        class="btn btn-{viewer_type}"
        target="{target}"
        rel="noopener noreferrer"
        title="{escaped_title}">
        {icon}{escaped_label}
    </a>"""


def render_metadata_tooltip(metadata: FITSViewerMetadata) -> str:
    """Render metadata as tooltip content.

    Parameters
    ----------
    metadata :
        FITSViewerMetadata instance

    Returns
    -------
        HTML tooltip string

    """
    info_parts = []

    if metadata.shape:
        info_parts.append(f"<strong>Dimensions:</strong> {metadata.format_shape()}")

    if metadata.resolution and metadata.resolution != "Unknown":
        info_parts.append(f"<strong>Resolution:</strong> {html.escape(metadata.resolution)}")

    if metadata.axes:
        info_parts.append(f"<strong>Axes:</strong> {html.escape(metadata.format_axes())}")

    if metadata.object:
        info_parts.append(f"<strong>Object:</strong> {html.escape(metadata.object)}")

    if metadata.telescop:
        info_parts.append(f"<strong>Telescope:</strong> {html.escape(metadata.telescop)}")

    if metadata.date_obs:
        info_parts.append(f"<strong>Date:</strong> {html.escape(metadata.date_obs)}")

    info_parts.append(f"<strong>Size:</strong> {metadata.file_size_mb:.1f} MB")

    return "<br>".join(info_parts)


def render_viewer_button_group(
    fits_path: str,
    metadata: FITSViewerMetadata,
    manager: FITSViewerManager,
    context: dict[str, Any],
) -> str:
    """Render complete button group for all enabled viewers.

    Parameters
    ----------
    fits_path :
        Path to FITS file
    metadata :
        Extracted FITSViewerMetadata
    manager :
        FITSViewerManager instance
    context :
        Context dictionary (e.g., report_id, session_id)

    Returns
    -------
        HTML string with all viewer buttons

    """
    buttons = []

    # Generate JS9 button
    if manager.config.js9_enabled:
        js9_url = manager.generate_js9_url(fits_path)
        if js9_url:
            buttons.append(
                render_viewer_button(
                    "js9",
                    js9_url,
                    label="Open in JS9",
                    title="Open FITS file in JS9 viewer (interactive)",
                )
            )

    # Generate CARTA button
    if manager.config.carta_enabled:
        carta_url = manager.generate_carta_url(fits_path)
        if carta_url:
            buttons.append(
                render_viewer_button(
                    "carta",
                    carta_url,
                    label="Open in CARTA",
                    title="Open FITS file in CARTA viewer (advanced)",
                )
            )

    # Generate Aladin button
    if manager.config.aladin_enabled:
        aladin_url = manager.generate_aladin_url(fits_path)
        if aladin_url:
            buttons.append(
                render_viewer_button(
                    "aladin",
                    aladin_url,
                    label="Open in Aladin",
                    title="Open FITS file in Aladin Lite viewer",
                )
            )

    if not buttons:
        return '<span class="fits-viewer-unavailable">No viewers available</span>'

    # Build metadata info with tooltip
    tooltip_content = render_metadata_tooltip(metadata)
    metadata_info = f"""
    <span class="fits-tooltip metadata-info">
        ℹ {html.escape(metadata.filename)} ({metadata.format_shape()})
        <span class="fits-tooltip-text">{tooltip_content}</span>
    </span>
    """

    buttons_html = "\n".join(buttons)

    data_attrs = ""
    for key in ("report_id", "session_id", "obs_id"):
        if key in context and context[key] is not None:
            data_key = key.replace("_", "-")
            data_attrs += f' data-{data_key}="{html.escape(str(context[key]))}"'

    return f"""
    <div class="fits-viewer-buttons"{data_attrs}>
        {buttons_html}
        {metadata_info}
    </div>
    """


def render_inline_js9_viewer(
    fits_path: str,
    metadata: FITSViewerMetadata,
    viewer_id: str,
    width: int = 512,
    height: int = 512,
) -> str:
    """Render inline JS9 viewer HTML.

    This creates an embedded JS9 viewer that can be included directly
    in an HTML page. Requires JS9 JavaScript library to be loaded.

    Parameters
    ----------
    fits_path :
        Path to FITS file
    metadata :
        FITSViewerMetadata instance
    viewer_id :
        Unique ID for this viewer instance
    width :
        Canvas width in pixels
    height :
        Canvas height in pixels

    Returns
    -------
        HTML string for inline JS9 viewer

    """
    escaped_path = html.escape(fits_path)
    escaped_id = html.escape(viewer_id)
    escaped_filename = html.escape(metadata.filename)

    return f"""
    <div class="js9-inline-viewer" id="js9-container-{escaped_id}">
        <div class="JS9Menubar" id="JS9Menubar-{escaped_id}"></div>
        <div class="JS9" id="JS9-{escaped_id}" data-width="{width}" data-height="{height}"></div>
        <div class="JS9Colorbar" id="JS9Colorbar-{escaped_id}"></div>
        <script type="text/javascript">
            // Load FITS file when JS9 is ready
            document.addEventListener('DOMContentLoaded', function() {{
                if (typeof JS9 !== 'undefined') {{
                    JS9.Load("{escaped_path}", {{
                        display: "JS9-{escaped_id}",
                        onload: function() {{
                            console.log("Loaded {escaped_filename}");
                        }}
                    }});
                }} else {{
                    console.warn("JS9 not loaded - cannot display inline viewer");
                    document.getElementById("js9-container-{escaped_id}").innerHTML =
                        '<p class="fits-viewer-unavailable">JS9 library not loaded</p>';
                }}
            }});
        </script>
    </div>
    """


def render_js9_script_includes(js9_base_url: str = "/js9") -> str:
    """Render JS9 script and CSS includes.

    Parameters
    ----------
    js9_base_url :
        Base URL path to JS9 installation

    Returns
    -------
        HTML string with script and link tags

    """
    escaped_url = html.escape(js9_base_url)

    return f"""
    <!-- JS9 Dependencies -->
    <link type="text/css" rel="stylesheet" href="{escaped_url}/js9support.css">
    <link type="text/css" rel="stylesheet" href="{escaped_url}/js9.css">
    <script type="text/javascript" src="{escaped_url}/js9prefs.js"></script>
    <script type="text/javascript" src="{escaped_url}/js9support.min.js"></script>
    <script type="text/javascript" src="{escaped_url}/js9.min.js"></script>
    <script type="text/javascript" src="{escaped_url}/js9plugins.js"></script>
    """


def render_download_button(fits_path: str, filename: str = "") -> str:
    """Render a download button for the FITS file.

    Parameters
    ----------
    fits_path :
        Path/URL to FITS file
    filename :
        Suggested filename for download

    Returns
    -------
        HTML string with download button

    """
    escaped_path = html.escape(fits_path)
    if not filename:
        filename = fits_path.split("/")[-1]
    escaped_filename = html.escape(filename)

    return f"""
    <a href="{escaped_path}"
       class="btn btn-download"
       download="{escaped_filename}"
       title="Download FITS file">
        <span class="btn-icon">⬇</span>Download FITS
    </a>
    """


def render_fits_image_block(
    png_html: str,
    viewer_buttons_html: str,
    description: str = "",
) -> str:
    """Render a complete FITS image block with PNG preview and viewer buttons.

    Parameters
    ----------
    png_html :
        HTML for PNG image display
    viewer_buttons_html :
        HTML for viewer buttons
    description :
        Optional description text

    Returns
    -------
        HTML string for complete image block

    """
    description_html = (
        f'<p class="description">{html.escape(description)}</p>' if description else ""
    )

    return f"""
    <div class="fits-image-block">
        {png_html}
        {viewer_buttons_html}
        {description_html}
    </div>
    """
