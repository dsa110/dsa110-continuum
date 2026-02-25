#!/usr/bin/env python
"""Generate a pyuvsim-compatible antenna layout from antpos_local data."""

import argparse
import csv
import math
import sys
from pathlib import Path

import astropy.units as u

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dsa110_contimg.common.utils.antpos_local import get_itrf
from dsa110_contimg.common.utils.constants import DSA110_ALT, DSA110_LAT, DSA110_LON


def main() -> None:
    parser = argparse.ArgumentParser(description="Write ENU antenna layout CSV")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("simulation/pyuvsim/antennas.csv"),
        help="Destination CSV path",
    )
    args = parser.parse_args()

    df = get_itrf(latlon_center=(DSA110_LAT * u.rad, DSA110_LON * u.rad, DSA110_ALT * u.m))
    indices = df.index.to_numpy()
    dx = df["dx_m"].to_numpy()
    dy = df["dy_m"].to_numpy()
    dz = df["dz_m"].to_numpy()

    sin_lat = math.sin(DSA110_LAT)
    cos_lat = math.cos(DSA110_LAT)
    sin_lon = math.sin(DSA110_LON)
    cos_lon = math.cos(DSA110_LON)

    east = -sin_lon * dx + cos_lon * dy
    north = (-sin_lat * cos_lon * dx) + (-sin_lat * sin_lon * dy) + (cos_lat * dz)
    up = cos_lat * cos_lon * dx + cos_lat * sin_lon * dy + sin_lat * dz

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["antenna_name", "antenna_number", "east_m", "north_m", "up_m"])
        for idx, east_m, north_m, up_m in zip(indices, east, north, up):
            writer.writerow(
                [
                    f"DSA{int(idx):03d}",
                    int(idx),
                    float(round(east_m, 6)),
                    float(round(north_m, 6)),
                    float(round(up_m, 6)),
                ]
            )

    print(f"Wrote antenna layout to {args.output}")


if __name__ == "__main__":
    main()
