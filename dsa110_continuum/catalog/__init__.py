"""Catalog utilities (crossmatches, per-strip SQLite databases, cone search)."""

from .crossmatch import (
    calc_de_ruiter,
    calc_de_ruiter_beamwidth,
    calculate_flux_scale,
    cross_match_dataframes,
    cross_match_sources,
    multi_catalog_match,
)

from .builders import (
    CATALOG_COVERAGE_LIMITS,
    build_first_strip_from_full,
    build_nvss_strip_from_full,
    build_rax_strip_from_full,
    build_vlass_strip_from_full,
    check_catalog_database_exists,
    check_missing_catalog_databases,
    get_atnf_full_db_path,
    get_first_full_db_path,
    get_nvss_full_db_path,
    get_rax_full_db_path,
    get_vlass_full_db_path,
)

from .query import cone_search, query_sources, resolve_catalog_path

from .coverage import (
    get_available_catalogs,
    get_catalog_coverage,
    is_position_in_catalog,
    recommend_catalogs,
    validate_catalog_choice,
)

__all__ = [
    # crossmatch
    "calc_de_ruiter",
    "calc_de_ruiter_beamwidth",
    "cross_match_sources",
    "cross_match_dataframes",
    "calculate_flux_scale",
    "multi_catalog_match",
    # builders
    "CATALOG_COVERAGE_LIMITS",
    "build_nvss_strip_from_full",
    "build_first_strip_from_full",
    "build_rax_strip_from_full",
    "build_vlass_strip_from_full",
    "check_catalog_database_exists",
    "check_missing_catalog_databases",
    "get_nvss_full_db_path",
    "get_first_full_db_path",
    "get_rax_full_db_path",
    "get_vlass_full_db_path",
    "get_atnf_full_db_path",
    # query
    "cone_search",
    "query_sources",
    "resolve_catalog_path",
    # coverage
    "get_catalog_coverage",
    "get_available_catalogs",
    "is_position_in_catalog",
    "recommend_catalogs",
    "validate_catalog_choice",
]
