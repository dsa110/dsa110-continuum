#!/usr/bin/env python
"""
Mosaic pipeline CLI entry point.

Usage:
    # Run on-demand mosaic
    python -m dsa110_contimg.core.mosaic on-demand --name custom_mosaic \
        --start 1700000000 --end 1700086400

    # With specific tier
    python -m dsa110_contimg.core.mosaic on-demand --name my_mosaic \
        --start 1700000000 --end 1700086400 --tier science

    # Dry run (show what would be processed without executing)
    python -m dsa110_contimg.core.mosaic on-demand --name test --start 1700000000 \
        --end 1700086400 --dry-run

    # Check pipeline status
    python -m dsa110_contimg.core.mosaic status
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from dsa110_contimg.common.utils import get_env_path

# Configure logging before imports
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("mosaic.cli")


def get_config():
    """Get mosaic pipeline configuration from environment.

    Returns
    -------
        MosaicPipelineConfig
        Configuration with paths from environment variables
    """
    from dsa110_contimg.core.mosaic.pipeline import MosaicPipelineConfig
    from dsa110_contimg.common.utils.paths import resolve_paths

    # Database path from environment
    db_path = os.environ.get(
        "PIPELINE_DB",
        os.environ.get("PIPELINE_DB", "/data/dsa110-contimg/state/db/pipeline.sqlite3"),
    )

    # Mosaic output directory
    base_dir = str(get_env_path("CONTIMG_BASE_DIR", default="/data/dsa110-contimg"))
    staging_dir = str(resolve_paths().staging_dir)
    mosaic_dir = os.environ.get(
        "CONTIMG_MOSAICS_DIR",
        os.environ.get("SCHED_MOSAIC_OUTPUT_DIR", str(Path(staging_dir) / "mosaics")),
    )

    return MosaicPipelineConfig(
        database_path=Path(db_path),
        mosaic_dir=Path(mosaic_dir),
    )


def cmd_on_demand(args: argparse.Namespace) -> int:
    """Run on-demand mosaic pipeline."""
    from dsa110_contimg.core.mosaic.pipeline import run_on_demand_mosaic

    config = get_config()

    logger.info("=" * 60)
    logger.info("On-Demand Mosaic Pipeline")
    logger.info("=" * 60)
    logger.info(f"Name: {args.name}")
    logger.info(
        f"Time range: {datetime.fromtimestamp(args.start, tz=UTC)} "
        f"to {datetime.fromtimestamp(args.end, tz=UTC)}"
    )
    logger.info(f"Tier: {args.tier or 'auto'}")
    logger.info(f"Database: {config.database_path}")
    logger.info(f"Output: {config.mosaic_dir}")
    logger.info("=" * 60)

    if args.dry_run:
        logger.info("DRY RUN - No execution")
        return 0

    # Ensure output directory exists
    config.mosaic_dir.mkdir(parents=True, exist_ok=True)

    try:
        result = run_on_demand_mosaic(
            config=config,
            name=args.name,
            start_time=args.start,
            end_time=args.end,
            tier=args.tier,
        )

        if result.success:
            logger.info(" On-demand mosaic completed successfully")
            logger.info(f"  Mosaic ID: {result.mosaic_id}")
            logger.info(f"  Path: {result.mosaic_path}")
            logger.info(f"  QA Status: {result.qa_status}")
            return 0
        else:
            logger.error(f" On-demand mosaic failed: {result.message}")
            for error in result.errors:
                logger.error(f"  - {error}")
            return 1

    except Exception:
        logger.exception("On-demand mosaic pipeline failed with exception")
        return 1


def cmd_status(args: argparse.Namespace) -> int:
    """Show mosaic pipeline status."""
    config = get_config()

    logger.info("=" * 60)
    logger.info("Mosaic Pipeline Status")
    logger.info("=" * 60)
    logger.info(f"Database: {config.database_path}")
    logger.info(f"Output: {config.mosaic_dir}")

    # Check database
    if not config.database_path.exists():
        logger.warning(f"Database not found: {config.database_path}")
    else:
        logger.info(f"Database exists: {config.database_path}")

    # Check output directory
    if not config.mosaic_dir.exists():
        logger.warning(f"Output directory not found: {config.mosaic_dir}")
    else:
        # Count mosaics
        mosaic_count = len(list(config.mosaic_dir.glob("*.fits")))
        logger.info(f"Output directory exists: {config.mosaic_dir}")
        logger.info(f"FITS files: {mosaic_count}")

    # Check recent executions from database
    try:
        import sqlite3

        conn = sqlite3.connect(config.database_path)
        cursor = conn.cursor()

        # Check for mosaic_groups table
        cursor.execute("""
            SELECT name FROM sqlite_master
            WHERE type='table' AND name='mosaic_groups'
        """)
        if cursor.fetchone():
            cursor.execute("""
                SELECT COUNT(*),
                       SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END),
                       SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END)
                FROM mosaic_groups
            """)
            total, completed, failed = cursor.fetchone()
            logger.info(f"Total mosaic groups: {total or 0}")
            logger.info(f"  Completed: {completed or 0}")
            logger.info(f"  Failed: {failed or 0}")
        else:
            logger.info("No mosaic_groups table found (mosaics not yet run)")

        conn.close()
    except Exception as e:
        logger.warning(f"Could not query database: {e}")

    return 0


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="DSA-110 Mosaic Pipeline CLI (On-Demand Only)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # On-demand command
    on_demand_parser = subparsers.add_parser(
        "on-demand",
        help="Run on-demand mosaic",
    )
    on_demand_parser.add_argument(
        "--name",
        required=True,
        help="Unique mosaic name",
    )
    on_demand_parser.add_argument(
        "--start",
        type=int,
        required=True,
        help="Start time (Unix timestamp)",
    )
    on_demand_parser.add_argument(
        "--end",
        type=int,
        required=True,
        help="End time (Unix timestamp)",
    )
    on_demand_parser.add_argument(
        "--tier",
        choices=["realtime", "science", "deep"],
        help="Mosaic tier (default: auto-select based on time range)",
    )
    on_demand_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without executing",
    )
    on_demand_parser.set_defaults(func=cmd_on_demand)

    # Status command
    status_parser = subparsers.add_parser(
        "status",
        help="Show mosaic pipeline status",
    )
    status_parser.set_defaults(func=cmd_status)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
