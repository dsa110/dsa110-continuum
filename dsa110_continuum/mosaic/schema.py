"""
Database schema for mosaic tables.

Three simple tables:
- mosaic_plans: Planning metadata
- mosaics: Product records
- mosaic_qa: Quality assessment results
"""

from __future__ import annotations

import sqlite3

# SQL for creating mosaic tables
# Note: These are now managed via Alembic migrations (revision 006)
# Kept here for reference or test setup
MOSAIC_TABLES: dict[str, str] = {
    "mosaic_plans": """
        CREATE TABLE IF NOT EXISTS mosaic_plans (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL,
            tier TEXT NOT NULL CHECK(tier IN ('quicklook', 'science', 'deep')),

            -- Time range
            start_time INTEGER NOT NULL,
            end_time INTEGER NOT NULL,

            -- Image selection
            image_ids TEXT NOT NULL,  -- JSON array
            n_images INTEGER NOT NULL,

            -- Coverage statistics
            ra_min_deg REAL,
            ra_max_deg REAL,
            dec_min_deg REAL,
            dec_max_deg REAL,

            -- Metadata
            created_at INTEGER NOT NULL,
            status TEXT DEFAULT 'pending'
                CHECK(status IN ('pending', 'building', 'completed', 'failed'))
        )
    """,
    "mosaics": """
        CREATE TABLE IF NOT EXISTS mosaics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            plan_id INTEGER NOT NULL REFERENCES mosaic_plans(id),

            -- File location
            path TEXT UNIQUE NOT NULL,

            -- Product metadata
            tier TEXT NOT NULL,
            n_images INTEGER NOT NULL,
            median_rms_jy REAL,
            effective_noise_jy REAL,  -- Propagated noise from inverse-variance weighting
            coverage_sq_deg REAL,

            -- Quality assessment
            qa_status TEXT CHECK(qa_status IN ('PASS', 'WARN', 'FAIL')),
            qa_details TEXT,  -- JSON

            -- Timestamps
            created_at INTEGER NOT NULL
        )
    """,
    "mosaic_qa": """
        CREATE TABLE IF NOT EXISTS mosaic_qa (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            mosaic_id INTEGER NOT NULL REFERENCES mosaics(id),

            -- Astrometric quality
            astrometry_rms_arcsec REAL,
            n_reference_stars INTEGER,

            -- Photometric quality
            median_noise_jy REAL,
            dynamic_range REAL,

            -- Artifacts
            has_artifacts INTEGER,
            artifact_score REAL,

            -- Overall
            passed INTEGER NOT NULL,
            warnings TEXT,  -- JSON array

            created_at INTEGER NOT NULL
        )
    """,
}

MOSAIC_INDEXES: list[str] = [
    "CREATE INDEX IF NOT EXISTS idx_mosaic_plans_tier ON mosaic_plans(tier)",
    "CREATE INDEX IF NOT EXISTS idx_mosaic_plans_status ON mosaic_plans(status)",
    "CREATE INDEX IF NOT EXISTS idx_mosaic_plans_created ON mosaic_plans(created_at)",
    "CREATE INDEX IF NOT EXISTS idx_mosaics_tier ON mosaics(tier)",
    "CREATE INDEX IF NOT EXISTS idx_mosaics_created ON mosaics(created_at)",
    "CREATE INDEX IF NOT EXISTS idx_mosaics_plan ON mosaics(plan_id)",
    "CREATE INDEX IF NOT EXISTS idx_mosaic_qa_mosaic ON mosaic_qa(mosaic_id)",
]


def ensure_mosaic_tables(conn: sqlite3.Connection) -> None:
    """Create mosaic tables if they don't exist.

    DEPRECATED: Tables are now managed by Alembic migrations.
    This function is kept for test environment setup.

    Parameters
    ----------
    conn :
        SQLite connection
    conn: sqlite3.Connection :


    """
    # Check if table exists to avoid redundant execution
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='mosaic_plans'")
    if cursor.fetchone():
        return

    for table_name, create_sql in MOSAIC_TABLES.items():
        cursor.execute(create_sql)

    for index_sql in MOSAIC_INDEXES:
        cursor.execute(index_sql)

    conn.commit()


def get_mosaic_schema_sql() -> str:
    """Get complete mosaic schema as a single SQL string."""
    statements = list(MOSAIC_TABLES.values()) + MOSAIC_INDEXES
    return ";\n".join(statements) + ";"
