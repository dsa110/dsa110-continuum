"""Catalog utilities (master catalog build, crossmatches, per-strip databases)."""

from dsa110_contimg.core.catalog.crossmatch import (
    calc_de_ruiter,
    calc_de_ruiter_beamwidth,
    calculate_flux_scale,
    calculate_positional_offsets,
    cross_match_dataframes,
    cross_match_sources,
    identify_duplicate_catalog_sources,
    multi_catalog_match,
    search_around_sky,
)

try:
    from dsa110_contimg.core.catalog.external import (
        gaia_search,
        ned_search,
        query_all_catalogs,
        simbad_search,
    )
except ImportError:
    # astroquery not available
    simbad_search = None
    ned_search = None
    gaia_search = None
    query_all_catalogs = None

__all__ = [
    "calc_de_ruiter",
    "calc_de_ruiter_beamwidth",
    "cross_match_sources",
    "cross_match_dataframes",
    "calculate_positional_offsets",
    "calculate_flux_scale",
    "search_around_sky",
    "multi_catalog_match",
    "identify_duplicate_catalog_sources",
]

from .build_atnf_pulsars import build_atnf_pulsar_db
from .builders import (
    CATALOG_COVERAGE_LIMITS,
    # ATNF
    atnf_full_db_exists,
    auto_build_missing_catalog_databases,
    build_atnf_full_db,
    build_atnf_strip_db,
    build_atnf_strip_from_full,
    # FIRST
    build_first_full_db,
    build_first_strip_db,
    build_first_strip_from_full,
    # NVSS
    build_nvss_full_db,
    build_nvss_strip_db,
    build_nvss_strip_from_full,
    # RAX
    build_rax_full_db,
    build_rax_strip_db,
    build_rax_strip_from_full,
    # VLASS
    build_vlass_full_db,
    build_vlass_strip_db,
    build_vlass_strip_from_full,
    check_and_regenerate_nvss_strips,
    # Utilities
    check_catalog_database_exists,
    check_missing_catalog_databases,
    first_full_db_exists,
    get_atnf_full_db_path,
    get_first_full_db_path,
    get_nvss_full_db_path,
    # RAX
    get_rax_full_db_path,
    # VLASS
    get_vlass_full_db_path,
    nvss_full_db_exists,
    rax_full_db_exists,
    regenerate_nvss_strip_db,
    vlass_full_db_exists,
)
from .query import query_sources, resolve_catalog_path

__all__ += [
    # Query
    "query_sources",
    "resolve_catalog_path",
    # NVSS
    "build_nvss_strip_db",
    "build_nvss_full_db",
    "build_nvss_strip_from_full",
    "regenerate_nvss_strip_db",
    "check_and_regenerate_nvss_strips",
    "nvss_full_db_exists",
    "get_nvss_full_db_path",
    # FIRST
    "build_first_strip_db",
    "build_first_full_db",
    "build_first_strip_from_full",
    "first_full_db_exists",
    "get_first_full_db_path",
    # RAX
    "build_rax_strip_db",
    "build_rax_full_db",
    "build_rax_strip_from_full",
    "rax_full_db_exists",
    "get_rax_full_db_path",
    # VLASS
    "build_vlass_strip_db",
    "build_vlass_full_db",
    "build_vlass_strip_from_full",
    "vlass_full_db_exists",
    "get_vlass_full_db_path",
    # ATNF
    "build_atnf_strip_db",
    "build_atnf_full_db",
    "build_atnf_strip_from_full",
    "build_atnf_pulsar_db",
    "atnf_full_db_exists",
    "get_atnf_full_db_path",
    # Utilities
    "auto_build_missing_catalog_databases",
    "check_missing_catalog_databases",
    "check_catalog_database_exists",
    "CATALOG_COVERAGE_LIMITS",
    # External
    "simbad_search",
    "ned_search",
    "gaia_search",
    "query_all_catalogs",
]
