"""HTML page generation for FITS viewer interface using templates."""

from __future__ import annotations

import html

from dsa110_contimg.common.utils.templates import render_template

from .fits_viewer_templates import get_css_styles


def generate_fits_viewer_page(
    title: str = "FITS Viewer",
    server_base_url: str = "http://localhost:8000",
    api_base: str = "/api/v1/fits",
) -> str:
    """Generate a complete HTML page for FITS file viewing.

    Parameters
    ----------
    title :
        Page title
    server_base_url :
        Base URL for viewer links
    api_base :
        Base path for FITS API endpoints

    Returns
    -------
        Complete HTML page as string

    """
    return render_template(
        "fits_viewer_page.html",
        escaped_title=html.escape(title),
        escaped_api=html.escape(api_base),
        escaped_base_url=html.escape(server_base_url),
        css_styles=get_css_styles(),
    )


def generate_fits_viewer_index() -> str:
    """Generate index page listing available viewers.

    Returns
    -------
        Complete HTML index page

    """
    return render_template("fits_viewer_index.html")
