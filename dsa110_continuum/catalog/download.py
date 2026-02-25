"""Download catalogs from external sources (Vizier, etc.)."""

import logging
import os
from pathlib import Path
from dsa110_contimg.common.utils import get_env_path

try:
    from astroquery.vizier import Vizier
    ASTROQUERY_AVAILABLE = True
except ImportError:
    ASTROQUERY_AVAILABLE = False

logger = logging.getLogger(__name__)

# Cache directory (respects CONTIMG_BASE_DIR)
DEFAULT_CACHE_DIR = get_env_path("CONTIMG_BASE_DIR", default="/data/dsa110-contimg") / ".cache" / "catalogs"


def download_first(cache_dir: Path | str = DEFAULT_CACHE_DIR) -> Path | None:
    """Download FIRST catalog from Vizier.
    
    Parameters
    ----------
    cache_dir : Path | str
        Directory to save the downloaded catalog.
        
    Returns
    -------
    Path | None
        Path to the downloaded CSV file, or None if download failed.
    """
    if not ASTROQUERY_AVAILABLE:
        logger.error("astroquery not installed, cannot download FIRST catalog")
        return None

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    output_path = cache_dir / "first_catalog_from_vizier.csv"
    
    if output_path.exists():
        logger.info(f"FIRST catalog already exists at {output_path}")
        return output_path

    logger.info("Downloading FIRST catalog from Vizier (VIII/92)...")
    logger.info("This may take several minutes due to the large catalog size...")

    # FIRST catalog from Vizier (VIII/92)
    vizier = Vizier(columns=["**"], row_limit=-1)
    vizier.ROW_LIMIT = -1  # Get all rows

    try:
        # Query FIRST catalog (VIII/92)
        result = vizier.query_constraints(catalog="VIII/92")

        if len(result) == 0:
            logger.error("Error: No results from Vizier for FIRST catalog")
            return None

        # Get the first table (should be the only one)
        table = result[0]
        df = table.to_pandas()

        logger.info(f"Downloaded {len(df)} FIRST sources")

        # Save to cache
        df.to_csv(output_path, index=False)
        logger.info(f"Saved FIRST catalog to: {output_path}")
        
        return output_path

    except Exception as e:
        logger.error(f"Error downloading FIRST catalog: {e}")
        return None


def download_vlass(cache_dir: Path | str = DEFAULT_CACHE_DIR) -> Path | None:
    """Download VLASS catalog from Vizier.
    
    Parameters
    ----------
    cache_dir : Path | str
        Directory to save the downloaded catalog.
        
    Returns
    -------
    Path | None
        Path to the downloaded CSV file, or None if download failed.
    """
    if not ASTROQUERY_AVAILABLE:
        logger.error("astroquery not installed, cannot download VLASS catalog")
        return None

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    output_path = cache_dir / "vlass_catalog_from_vizier.csv"
    
    if output_path.exists():
        logger.info(f"VLASS catalog already exists at {output_path}")
        return output_path

    logger.info("Downloading VLASS catalog from Vizier (J/ApJS/255/30)...")
    logger.info("This may take several minutes due to the large catalog size...")

    # VLASS catalog from Vizier (J/ApJS/255/30)
    vizier = Vizier(columns=["**"], row_limit=-1)
    vizier.ROW_LIMIT = -1  # Get all rows

    try:
        # Query VLASS catalog (J/ApJS/255/30)
        result = vizier.query_constraints(catalog="J/ApJS/255/30")

        if len(result) == 0:
            logger.error("Error: No results from Vizier for VLASS catalog")
            return None

        # Get the first table (should be the only one)
        table = result[0]
        df = table.to_pandas()

        logger.info(f"Downloaded {len(df)} VLASS sources")

        # Save to cache
        df.to_csv(output_path, index=False)
        logger.info(f"Saved VLASS catalog to: {output_path}")
        
        return output_path

    except Exception as e:
        logger.error(f"Error downloading VLASS catalog: {e}")
        return None


def download_racs(cache_dir: Path | str = DEFAULT_CACHE_DIR) -> Path | None:
    """Download RACS catalog from Vizier.
    
    Parameters
    ----------
    cache_dir : Path | str
        Directory to save the downloaded catalog.
        
    Returns
    -------
    Path | None
        Path to the downloaded CSV file, or None if download failed.
    """
    if not ASTROQUERY_AVAILABLE:
        logger.error("astroquery not installed, cannot download RACS catalog")
        return None

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    output_path = cache_dir / "racs_catalog_from_vizier.csv"
    
    if output_path.exists():
        logger.info(f"RACS catalog already exists at {output_path}")
        return output_path

    logger.info("Downloading RACS catalog from Vizier (J/other/PASA/38.58)...")
    logger.info("This may take several minutes due to the large catalog size...")

    # RACS catalog from Vizier (J/other/PASA/38.58)
    vizier = Vizier(columns=["**"], row_limit=-1)
    vizier.ROW_LIMIT = -1  # Get all rows
    vizier.TIMEOUT = 1200  # 20 minutes timeout

    try:
        # Query RACS catalog (J/other/PASA/38.58)
        # We target the main table (Table 0)
        result = vizier.query_constraints(catalog="J/other/PASA/38.58")

        if len(result) == 0:
            logger.error("Error: No results from Vizier for RACS catalog")
            return None

        # Get the first table (should be the main catalog)
        table = result[0]
        df = table.to_pandas()

        logger.info(f"Downloaded {len(df)} RACS sources")

        # Save to cache
        df.to_csv(output_path, index=False)
        logger.info(f"Saved RACS catalog to: {output_path}")
        
        return output_path

    except Exception as e:
        logger.error(f"Error downloading RACS catalog: {e}")
        return None
