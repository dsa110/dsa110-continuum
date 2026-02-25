import logging
import re
import sqlite3
from pathlib import Path

import pandas as pd

# Self-contained configuration to avoid package import issues
BAND_MAPPING = {
    "90cm": ("P", 0.33e9),
    "20cm": ("L", 1.4e9),
    "6cm": ("C", 5.0e9),
    "3.7cm": ("X", 8.4e9),
    "2cm": ("U", 15.0e9),
    "1.3cm": ("K", 22.0e9),
    "0.7cm": ("Q", 43.0e9),
}
# Default output location for the database
DEFAULT_OUTPUT_DB = Path("/data/dsa110-contimg/state/catalogs/vla_calibrators.sqlite3")


def _parse_ra_dec(ra_str: str, dec_str: str) -> tuple[float, float]:
    """Parse VLA calibrator RA/Dec strings to degrees."""
    ra_str = ra_str.strip()
    dec_str = dec_str.strip().rstrip('"').rstrip("'")

    ra_match = re.match(r"(\d+)h(\d+)m([\d.]+)s", ra_str)
    if ra_match:
        h, m, s = ra_match.groups()
        ra_deg = (float(h) + float(m) / 60 + float(s) / 3600) * 15
    else:
        raise ValueError(f"Cannot parse RA: {ra_str}")

    dec_match = re.match(r"([+-]?\d+)d(\d+)'([\d.]+)", dec_str)
    if dec_match:
        d, m, s = dec_match.groups()
        sign = -1 if d.startswith("-") else 1
        dec_deg = sign * (abs(float(d)) + float(m) / 60 + float(s) / 3600)
    else:
        raise ValueError(f"Cannot parse Dec: {dec_str}")
    return ra_deg, dec_deg


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_from_csv(csv_path: Path, db_path: Path):
    logger.info(f"Loading {csv_path} into {db_path}")
    df = pd.read_csv(csv_path)

    db_path.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(db_path) as conn:
        # Create schema
        conn.execute("""
            CREATE TABLE IF NOT EXISTS calibrators (
                name TEXT PRIMARY KEY,
                ra_deg REAL NOT NULL,
                dec_deg REAL NOT NULL,
                position_code TEXT,
                alt_name TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS fluxes (
                name TEXT NOT NULL,
                band TEXT NOT NULL,
                band_code TEXT,
                flux_jy REAL NOT NULL,
                freq_hz REAL,
                quality_codes TEXT,
                PRIMARY KEY(name, band),
                FOREIGN KEY(name) REFERENCES calibrators(name) ON DELETE CASCADE
            )
        """)
        conn.execute("DELETE FROM fluxes")
        conn.execute("DELETE FROM calibrators")

        calibrators = {}  # name -> (ra, dec, pc, alt)
        flux_entries = []

        for _, row in df.iterrows():
            name = row["J2000_NAME"]
            ra_str = row["RA_J2000"]
            dec_str = row["DEC_J2000"]
            pc = row["PC_J2000"]
            alt = row["ALT_NAME"]
            if pd.isna(alt):
                alt = None

            if name not in calibrators:
                try:
                    ra_deg, dec_deg = _parse_ra_dec(ra_str, dec_str)
                    calibrators[name] = (ra_deg, dec_deg, pc, alt)
                except Exception as e:
                    logger.warning(f"Error parsing position for {name}: {e}")
                    continue

            band = row["BAND"]
            band_code = row["BAND_CODE"]
            flux_jy = row["FLUX_JY"]
            if pd.isna(flux_jy):
                flux_jy = 0.0

            freq_hz = BAND_MAPPING.get(band, (None, None))[1]

            # Codes A, B, C, D
            codes = "".join(
                [str(row[c]) if not pd.isna(row[c]) else "?" for c in ["A", "B", "C", "D"]]
            )

            flux_entries.append((name, band, band_code, flux_jy, freq_hz, codes))

        logger.info(f"Inserting {len(calibrators)} calibrators...")
        for name, (ra, dec, pc, alt) in calibrators.items():
            conn.execute(
                "INSERT INTO calibrators (name, ra_deg, dec_deg, position_code, alt_name) VALUES (?, ?, ?, ?, ?)",
                (name, ra, dec, pc, alt),
            )

        logger.info(f"Inserting {len(flux_entries)} flux entries...")
        conn.executemany(
            "INSERT OR REPLACE INTO fluxes (name, band, band_code, flux_jy, freq_hz, quality_codes) VALUES (?, ?, ?, ?, ?, ?)",
            flux_entries,
        )

        # Create View
        conn.execute("DROP VIEW IF EXISTS vla_20cm")
        conn.execute("""
            CREATE VIEW vla_20cm AS
            SELECT c.name, c.ra_deg, c.dec_deg, c.position_code, c.alt_name,
                   f.flux_jy, f.quality_codes
            FROM calibrators c
            JOIN fluxes f ON c.name = f.name
            WHERE f.band = '20cm'
        """)

        conn.commit()
    logger.info("Done!")


if __name__ == "__main__":
    csv_path = Path("/data/dsa110-contimg/misc/backup2-dsa110-contimg/vla_calibrators_parsed.csv")
    db_path = DEFAULT_OUTPUT_DB
    load_from_csv(csv_path, db_path)
