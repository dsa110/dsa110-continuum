from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

from dsa110_continuum.mosaic.production import (
    build_epoch_coadd_products,
    weight_map_is_valid,
    write_weight_map,
)


OUTPUT_DIR = Path("/data/dsa110-continuum/outputs/green-refactor-issue-96/pr1-validation")
TILE_DIR = Path("/stage/dsa110-contimg/images/mosaic_2026-01-25")
BASELINE = Path(
    "/data/dsa110-continuum/outputs/mosaic-visual-qa-2026-07-03/"
    "2026-01-25T2200_mosaic_sault.fits"
)


def robust_rms(values: np.ndarray) -> float:
    median = np.median(values)
    return float(1.4826 * np.median(np.abs(values - median)))


def main() -> None:
    started = time.time()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(OUTPUT_DIR / "rebuild.log", mode="w"),
            logging.StreamHandler(),
        ],
    )
    tiles = sorted(str(path) for path in TILE_DIR.glob("2026-01-25T22*-image-pb.fits"))
    if len(tiles) != 11:
        raise RuntimeError(f"expected 11 reference tiles, found {len(tiles)}")

    result = build_epoch_coadd_products(tiles)
    mosaic_path = OUTPUT_DIR / "2026-01-25T2200_mosaic.fits"
    fits.PrimaryHDU(
        data=result.mosaic.astype(np.float32),
        header=result.wcs.to_header(),
    ).writeto(mosaic_path, overwrite=True)
    weight_path = write_weight_map(result.weight, result.wcs, mosaic_path)

    rebuilt = fits.getdata(mosaic_path, memmap=True).squeeze()
    baseline = fits.getdata(BASELINE, memmap=True).squeeze()
    weight = fits.getdata(weight_path, memmap=True).squeeze()
    rebuilt_wcs = WCS(fits.getheader(mosaic_path)).celestial
    baseline_wcs = WCS(fits.getheader(BASELINE)).celestial

    if rebuilt.shape != baseline.shape:
        raise RuntimeError(f"shape changed: {baseline.shape} -> {rebuilt.shape}")

    rebuilt_science = np.isfinite(rebuilt)
    baseline_science = np.isfinite(baseline)
    common = rebuilt_science & baseline_science
    same = (rebuilt == baseline) | (~rebuilt_science & ~baseline_science)
    differences = np.abs(rebuilt[common].astype(np.float64) - baseline[common].astype(np.float64))

    block_rows: list[dict[str, float]] = []
    block_size = 256
    for y0 in range(0, rebuilt.shape[0] - block_size + 1, block_size):
        for x0 in range(0, rebuilt.shape[1] - block_size + 1, block_size):
            image_block = rebuilt[y0 : y0 + block_size, x0 : x0 + block_size]
            weight_block = weight[y0 : y0 + block_size, x0 : x0 + block_size]
            valid = np.isfinite(image_block) & (weight_block > 0)
            if valid.mean() < 0.8:
                continue
            measured = robust_rms(image_block[valid].astype(np.float64))
            predicted = float(np.median(1.0 / np.sqrt(weight_block[valid])))
            if measured > 0 and predicted > 0:
                block_rows.append(
                    {
                        "y": float(y0),
                        "x": float(x0),
                        "measured_rms_jy": measured,
                        "predicted_rms_jy": predicted,
                        "ratio": measured / predicted,
                    }
                )

    measured_values = np.array([row["measured_rms_jy"] for row in block_rows])
    predicted_values = np.array([row["predicted_rms_jy"] for row in block_rows])
    noise_correlation = (
        float(np.corrcoef(measured_values, predicted_values)[0, 1])
        if len(block_rows) >= 2
        else None
    )

    evidence = {
        "baseline": str(BASELINE),
        "mosaic": str(mosaic_path),
        "weight": str(weight_path),
        "tiles": tiles,
        "tile_count": len(tiles),
        "shape": list(rebuilt.shape),
        "wcs_equal_to_baseline": bool(rebuilt_wcs.wcs.compare(baseline_wcs.wcs)),
        "weight_product_valid": weight_map_is_valid(weight_path, mosaic_path),
        "weight_finite": bool(np.all(np.isfinite(weight))),
        "weight_nonnegative": bool(np.all(weight >= 0)),
        "weight_positive_on_science": bool(np.all(weight[rebuilt_science] > 0)),
        "weight_zero_outside_science": bool(np.all(weight[~rebuilt_science] == 0)),
        "baseline_finite_pixels": int(baseline_science.sum()),
        "rebuilt_finite_pixels": int(rebuilt_science.sum()),
        "common_finite_pixels": int(common.sum()),
        "support_pixels_removed": int((baseline_science & ~rebuilt_science).sum()),
        "support_pixels_added": int((rebuilt_science & ~baseline_science).sum()),
        "identical_pixels_including_blank": int(same.sum()),
        "changed_common_pixels": int(np.count_nonzero(differences)),
        "max_abs_pixel_difference_jy": float(np.max(differences)) if differences.size else None,
        "rms_pixel_difference_jy": (
            float(np.sqrt(np.mean(differences**2))) if differences.size else None
        ),
        "noise_block_count": len(block_rows),
        "median_measured_to_predicted_noise_ratio": (
            float(np.median(measured_values / predicted_values)) if len(block_rows) else None
        ),
        "measured_predicted_noise_correlation": noise_correlation,
        "noise_blocks": block_rows,
        "elapsed_seconds": time.time() - started,
    }
    (OUTPUT_DIR / "evidence.json").write_text(json.dumps(evidence, indent=2) + "\n")
    print(json.dumps({key: value for key, value in evidence.items() if key != "noise_blocks"}, indent=2))


if __name__ == "__main__":
    main()
