"""
Visualization module for DSA-110 continuum imaging pipeline.

Provides standardized figure generation for:
- FITS images (cutouts, mosaics, quicklook PNGs)
- Calibration diagnostics (bandpass, gains, delays)
- Source analysis (lightcurves, spectra, validation reports)
- UV coverage and visibility plots
- Image comparison and difference maps
- Per-antenna diagnostics
- PSF/beam analysis

Adapted from:
- dsa110-calib/dsacalib/plotting.py (Dana Simard)
- VAST/vastfast/plot.py (Yuanming Wang)
- ASKAP-continuum-validation/report.py (Jordan Collier)
- radiopadre/fitsfile.py
- eht-imaging/comp_plots.py (image comparison patterns)

Usage:
    from dsa110_contimg.core.visualization import (
        plot_fits_image,
        plot_cutout,
        save_quicklook_png,
        FigureConfig,
    )

    # Quick PNG from FITS
    save_quicklook_png("image.fits", "image.png")

    # Publication-quality cutout
    plot_cutout("image.fits", ra=180.0, dec=45.0, radius_arcmin=5.0,
                output="cutout.pdf", config=FigureConfig(style="publication"))

    # UV coverage plot
    from dsa110_contimg.core.visualization import plot_uv_coverage
    plot_uv_coverage(u_lambda, v_lambda, output="uv_coverage.png")

    # Image comparison
    from dsa110_contimg.core.visualization import plot_image_comparison
    plot_image_comparison(image1, image2, output="comparison.png")

    # Generate diagram for a directory
    from dsa110_contimg.core.visualization import generate_structure_diagram
    generate_structure_diagram("/path/to/source", "output_diagram.svg")
"""

from dsa110_contimg.core.visualization.antenna_correlation import (
    AntennaGainData,
    CorrelationStatistics,
    compute_gain_correlation_matrix,
    extract_gains_from_caltable,
    generate_correlation_diagnostic_report,
    identify_correlated_groups,
    plot_correlation_network,
    plot_correlation_summary,
    plot_gain_correlation_matrix,
    plot_temporal_correlation_evolution,
)
from dsa110_contimg.core.visualization.antenna_plots import (
    compute_antenna_statistics_from_ms,
    plot_antenna_flagging_summary,
    plot_antenna_gain_spectrum,
    plot_antenna_gain_time_series,
    plot_antenna_statistics_grid,
)
from dsa110_contimg.core.visualization.beam_plots import (
    fit_2d_gaussian,
    plot_beam_comparison,
    plot_primary_beam_pattern,
    plot_psf_2d,
    plot_psf_radial_profile,
    plot_sidelobe_analysis,
)
from dsa110_contimg.core.visualization.calibration_plots import (
    plot_bandpass,
    plot_delays,
    plot_dterm_scatter,
    plot_dynamic_spectrum,
    plot_flagging_diagnostics,
    plot_gain_comparison,
    plot_gain_snr,
    plot_gains,
)
from dsa110_contimg.core.visualization.calibration_stability_plots import (
    plot_calibration_stability,
)

# CARTA integration and HDF5 IDIA conversion
from dsa110_contimg.core.visualization.carta_scripting import (
    CARTARegion,
    CARTAScriptingClient,
    CARTASessionState,
    MomentMapResult,
    MomentType,
    RegionStatistics,
)
from dsa110_contimg.core.visualization.closure_phase_plots import (
    compute_closure_phases,
    extract_closure_phases_from_ms,
    plot_closure_phase_antenna_contribution,
    plot_closure_phase_histogram,
    plot_closure_phase_per_triangle,
    plot_closure_phase_vs_time,
)
from dsa110_contimg.core.visualization.config import FigureConfig, PlotStyle
from dsa110_contimg.core.visualization.convergence_plots import (
    ConvergenceData,
    TimeFreqConvergenceData,
    compute_convergence_quality_score,
    compute_time_freq_convergence,
    extract_convergence_from_selfcal_result,
    plot_antenna_solution_quality,
    plot_chi_squared_improvement,
    plot_clean_convergence,
    plot_convergence_comparison,
    plot_per_antenna_convergence_heatmap,
    plot_selfcal_convergence,
    plot_time_freq_convergence_animation,
    plot_time_freq_convergence_heatmap,
    plot_time_freq_difference_heatmap,
)
from dsa110_contimg.core.visualization.elevation_plots import (
    compute_azimuth,
    compute_elevation,
    compute_parallactic_angle,
    extract_geometry_from_hdf5,
    extract_geometry_from_ms,
    plot_azel_track,
    plot_elevation_histogram,
    plot_elevation_vs_time,
    plot_hour_angle_coverage,
    plot_observation_summary,
    plot_parallactic_angle_vs_time,
)
from dsa110_contimg.core.visualization.fits_plots import (
    plot_cutout,
    plot_fits_image,
    plot_mosaic_overview,
    save_quicklook_png,
)
from dsa110_contimg.core.visualization.fits_viewer import (
    AVAILABLE_COLORMAPS,
    DEFAULT_COLORMAP,
    VIEWER_ALADIN,
    VIEWER_CARTA,
    VIEWER_JS9,
    FITSFileError,
    FITSViewerMetadata,
    FITSParsingError,
    FITSViewerConfig,
    FITSViewerException,
    FITSViewerManager,
    format_resolution_degrees,
    get_axis_label,
    get_file_size_mb,
    validate_fits_file,
)
from dsa110_contimg.core.visualization.fits_viewer_templates import (
    get_css_styles,
    render_download_button,
    render_fits_image_block,
    render_inline_js9_viewer,
    render_js9_script_includes,
    render_metadata_tooltip,
    render_viewer_button,
    render_viewer_button_group,
)
from dsa110_contimg.core.visualization.hdf5_idia import (
    ConversionResult,
    check_idia_format,
    convert_fits_to_idia_hdf5,
    find_fits2idia,
)
from dsa110_contimg.core.visualization.hdf5_idia import (
    batch_convert as batch_convert_to_idia,
)
from dsa110_contimg.core.visualization.image_comparison import (
    compare_fits_images,
    compute_comparison_metrics,
    plot_image_comparison,
    plot_pixel_scatter,
    plot_residual_map,
)
from dsa110_contimg.core.visualization.mosaic_plots import (
    plot_coverage_map,
    plot_mosaic_footprints,
    plot_tile_grid,
)
from dsa110_contimg.core.visualization.photometry_plots import (
    plot_aperture_photometry,
    plot_catalog_comparison,
    plot_field_sources,
    plot_photometry_summary,
    plot_snr_map,
)
from dsa110_contimg.core.visualization.plot_context import (
    PerformanceLogger,
    PlotContext,
    detect_context_from_path,
    get_file_extension,
    should_generate_interactive,
)
from dsa110_contimg.core.visualization.qa_plots import (
    plot_dynamic_range_map,
    plot_psf_correlation,
    plot_residual_histogram,
)
from dsa110_contimg.core.visualization.report import (
    ReportMetadata,
    ReportSection,
    create_diagnostic_report,
    generate_html_report,
    generate_pdf_report,
)
from dsa110_contimg.core.visualization.residual_diagnostics import (
    ResidualData,
    ResidualStatistics,
    compute_residual_statistics,
    extract_residuals_from_ms,
    generate_residual_diagnostic_report,
    plot_residual_amplitude_vs_baseline,
    plot_residual_complex_scatter,
    plot_residual_per_antenna,
    plot_residual_phase_vs_time,
)
from dsa110_contimg.core.visualization.residual_diagnostics import (
    plot_residual_histogram as plot_visibility_residual_histogram,
)
from dsa110_contimg.core.visualization.rfi_plots import (
    plot_rfi_spectrum,
    plot_rfi_waterfall,
)
from dsa110_contimg.core.visualization.source_plots import (
    plot_lightcurve,
    plot_monitoring_lightcurve,
    plot_source_comparison,
    plot_spectrum,
)
from dsa110_contimg.core.visualization.lightcurve_data import get_photometry_data
from dsa110_contimg.core.visualization.spectral_plots import (
    compute_spectral_index,
    compute_spectral_index_from_fits,
    plot_multi_frequency_mosaic,
    plot_sed,
    plot_spectral_index_error_map,
    plot_spectral_index_histogram,
    plot_spectral_index_map,
)
from dsa110_contimg.core.visualization.tsys_plots import (
    detect_tsys_anomalies,
    extract_tsys_from_ms,
    plot_tsys_elevation,
    plot_tsys_heatmap,
    plot_tsys_histogram,
    plot_tsys_summary,
    plot_tsys_time_series,
    plot_tsys_vs_elevation,
)
from dsa110_contimg.core.visualization.uv_plots import (
    extract_uv_from_ms,
    plot_baseline_distribution,
    plot_uv_coverage,
    plot_uv_density,
    plot_visibility_amplitude_vs_time,
    plot_visibility_amplitude_vs_uvdist,
    plot_visibility_phase_vs_time,
)
from dsa110_contimg.core.visualization.vega_specs import (
    create_residual_histogram_spec,
    create_rfi_spectrum_spec,
    create_rfi_waterfall_spec,
    create_scatter_spec,
    save_vega_spec,
)
from dsa110_contimg.core.visualization.structure import generate_structure_diagram

__all__ = [
    # Config
    "FigureConfig",
    "PlotStyle",
    # Context
    "PlotContext",
    "should_generate_interactive",
    "get_file_extension",
    "detect_context_from_path",
    "PerformanceLogger",
    # FITS
    "plot_fits_image",
    "plot_cutout",
    "save_quicklook_png",
    "plot_mosaic_overview",
    # Calibration
    "plot_bandpass",
    "plot_gains",
    "plot_delays",
    "plot_dynamic_spectrum",
    "plot_flagging_diagnostics",
    "plot_gain_snr",
    "plot_dterm_scatter",
    "plot_gain_comparison",
    "plot_calibration_stability",
    # Sources
    "plot_lightcurve",
    "plot_monitoring_lightcurve",
    "plot_spectrum",
    "plot_source_comparison",
    "get_photometry_data",
    # Mosaics
    "plot_tile_grid",
    "plot_mosaic_footprints",
    "plot_coverage_map",
    # RFI
    "plot_rfi_spectrum",
    "plot_rfi_waterfall",
    # QA
    "plot_psf_correlation",
    "plot_residual_histogram",
    "plot_dynamic_range_map",
    # Vega-Lite specs
    "create_rfi_spectrum_spec",
    "create_rfi_waterfall_spec",
    "create_residual_histogram_spec",
    "create_scatter_spec",
    "save_vega_spec",
    # Reports
    "ReportSection",
    "ReportMetadata",
    "generate_html_report",
    "generate_pdf_report",
    "create_diagnostic_report",
    # UV/Visibility plots
    "plot_uv_coverage",
    "plot_uv_density",
    "plot_baseline_distribution",
    "plot_visibility_amplitude_vs_uvdist",
    "plot_visibility_phase_vs_time",
    "plot_visibility_amplitude_vs_time",
    "extract_uv_from_ms",
    # Image comparison
    "plot_image_comparison",
    "plot_pixel_scatter",
    "plot_residual_map",
    "compare_fits_images",
    "compute_comparison_metrics",
    # Antenna diagnostics
    "plot_antenna_flagging_summary",
    "plot_antenna_gain_time_series",
    "plot_antenna_gain_spectrum",
    "plot_antenna_statistics_grid",
    "compute_antenna_statistics_from_ms",
    # PSF/Beam analysis
    "plot_psf_radial_profile",
    "plot_psf_2d",
    "plot_beam_comparison",
    "plot_sidelobe_analysis",
    "plot_primary_beam_pattern",
    "fit_2d_gaussian",
    # Closure phases
    "compute_closure_phases",
    "plot_closure_phase_histogram",
    "plot_closure_phase_vs_time",
    "plot_closure_phase_per_triangle",
    "plot_closure_phase_antenna_contribution",
    "extract_closure_phases_from_ms",
    # Spectral index / SED
    "compute_spectral_index",
    "plot_spectral_index_map",
    "plot_spectral_index_error_map",
    "plot_sed",
    "plot_multi_frequency_mosaic",
    "plot_spectral_index_histogram",
    "compute_spectral_index_from_fits",
    # Convergence / optimization
    "ConvergenceData",
    "TimeFreqConvergenceData",
    "extract_convergence_from_selfcal_result",
    "plot_selfcal_convergence",
    "plot_antenna_solution_quality",
    "plot_chi_squared_improvement",
    "plot_clean_convergence",
    "plot_convergence_comparison",
    "compute_time_freq_convergence",
    "plot_time_freq_convergence_heatmap",
    "plot_time_freq_convergence_animation",
    "plot_time_freq_difference_heatmap",
    "plot_per_antenna_convergence_heatmap",
    "compute_convergence_quality_score",
    # System temperature
    "extract_tsys_from_ms",
    "plot_tsys_time_series",
    "plot_tsys_summary",
    "plot_tsys_heatmap",
    "plot_tsys_histogram",
    "plot_tsys_vs_elevation",
    "plot_tsys_elevation",  # Alias for plot_tsys_vs_elevation
    "detect_tsys_anomalies",
    # Elevation / parallactic angle
    "compute_parallactic_angle",
    "compute_elevation",
    "compute_azimuth",
    "plot_elevation_vs_time",
    "plot_parallactic_angle_vs_time",
    "plot_azel_track",
    "plot_hour_angle_coverage",
    "plot_elevation_histogram",
    "plot_observation_summary",
    "extract_geometry_from_ms",
    "extract_geometry_from_hdf5",
    # Visibility Residual Diagnostics
    "ResidualData",
    "ResidualStatistics",
    "extract_residuals_from_ms",
    "compute_residual_statistics",
    "plot_residual_amplitude_vs_baseline",
    "plot_residual_phase_vs_time",
    "plot_visibility_residual_histogram",
    "plot_residual_complex_scatter",
    "plot_residual_per_antenna",
    "generate_residual_diagnostic_report",
    # Antenna Gain Correlation
    "AntennaGainData",
    "CorrelationStatistics",
    "extract_gains_from_caltable",
    "compute_gain_correlation_matrix",
    "plot_gain_correlation_matrix",
    "plot_correlation_network",
    "plot_temporal_correlation_evolution",
    "plot_correlation_summary",
    "identify_correlated_groups",
    "generate_correlation_diagnostic_report",
    # Photometry plots
    "plot_aperture_photometry",
    "plot_snr_map",
    "plot_catalog_comparison",
    "plot_field_sources",
    "plot_photometry_summary",
    # FITS Viewer integration
    "FITSViewerConfig",
    "FITSViewerManager",
    "FITSViewerMetadata",
    "FITSViewerException",
    "FITSFileError",
    "FITSParsingError",
    "VIEWER_JS9",
    "VIEWER_CARTA",
    "VIEWER_ALADIN",
    "DEFAULT_COLORMAP",
    "AVAILABLE_COLORMAPS",
    "validate_fits_file",
    "get_file_size_mb",
    "format_resolution_degrees",
    "get_axis_label",
    "render_viewer_button_group",
    "render_inline_js9_viewer",
    "render_viewer_button",
    "render_metadata_tooltip",
    "render_download_button",
    "render_fits_image_block",
    "render_js9_script_includes",
    "get_css_styles",
    # CARTA scripting and HDF5 IDIA conversion
    "CARTAScriptingClient",
    "CARTASessionState",
    "CARTARegion",
    "MomentType",
    "RegionStatistics",
    "MomentMapResult",
    "convert_fits_to_idia_hdf5",
    "batch_convert_to_idia",
    "check_idia_format",
    "find_fits2idia",
    "ConversionResult",
    # Mermaid
    "MermaidRenderer",
    "generate_structure_diagram",
]
