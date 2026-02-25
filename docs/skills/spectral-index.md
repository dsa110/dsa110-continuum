# Spectral Index Skill

Expert guidance for spectral index computation in the DSA-110 pipeline.

## Overview

The spectral index stage computes spectral indices (α where S ∝ ν^α) from multi-frequency images. This provides insights into source physics (thermal vs non-thermal) and can aid in source classification.

## Key Modules

| Module | Purpose |
|--------|---------|
| `workflow/pipeline/stages/spectral_index.py` | `SpectralIndexStage` pipeline stage |
| `core/imaging/spectral_analysis.py` | Spectral fitting algorithms |
| `core/visualization/sed_plots.py` | SED visualization |

## Spectral Index Definition

Radio flux density relates to frequency as:

```
S(ν) = S₀ × (ν/ν₀)^α

Where:
- S = flux density
- ν = frequency  
- α = spectral index
- S₀ = flux at reference frequency ν₀
```

### Typical Values

| Source Type | α | Interpretation |
|-------------|---|----------------|
| Optically thin synchrotron | -0.7 to -0.8 | Normal AGN, SNR |
| Flat spectrum | -0.5 to +0.5 | Compact cores |
| Steep spectrum | < -1.0 | Aged electrons |
| Thermal (free-free) | -0.1 | HII regions |
| Inverted | > 0 | Self-absorbed synchrotron |

## Pipeline Stage Usage

```python
from dsa110_contimg.workflow.pipeline.stages import SpectralIndexStage

stage = SpectralIndexStage(config)
context = PipelineContext(
    config=config,
    outputs={
        "multi_frequency_images": [
            {"path": "/data/img_1280MHz.fits", "freq_hz": 1.28e9},
            {"path": "/data/img_1400MHz.fits", "freq_hz": 1.40e9},
            {"path": "/data/img_1520MHz.fits", "freq_hz": 1.52e9},
        ]
    }
)

result = stage.execute(context)
spectral_results = result.outputs["spectral_index_results"]
```

## DSA-110 Frequency Coverage

DSA-110 observes 1.28-1.53 GHz across 16 subbands:

| Subband | Center Freq (MHz) | Use for α |
|---------|-------------------|-----------|
| sb00 | 1280 | Yes (low end) |
| sb07 | 1400 | Yes (mid) |
| sb15 | 1530 | Yes (high end) |

**Bandwidth limitation**: 250 MHz span limits spectral index precision to Δα ≈ 0.3.

## Spectral Index Map

Pixel-by-pixel spectral index computation:

```python
from dsa110_contimg.core.imaging.spectral_analysis import (
    compute_spectral_index_map,
    SpectralIndexConfig,
)

config = SpectralIndexConfig(
    method="linear_regression",  # or "weighted_fit"
    min_snr=5.0,                 # Minimum S/N per pixel
    mask_threshold_sigma=3.0,    # Mask below this threshold
)

alpha_map, alpha_error_map = compute_spectral_index_map(
    image_paths=["/data/img_1280.fits", "/data/img_1400.fits", "/data/img_1520.fits"],
    frequencies_hz=[1.28e9, 1.40e9, 1.52e9],
    config=config,
)

# Save as FITS
from astropy.io import fits
hdu = fits.PrimaryHDU(alpha_map, header=wcs_header)
hdu.writeto("/data/spectral_index_map.fits")
```

## Source SED Extraction

Extract spectral energy distribution for catalog sources:

```python
from dsa110_contimg.core.imaging.spectral_analysis import extract_source_seds

seds = extract_source_seds(
    image_paths=image_paths,
    frequencies_hz=frequencies_hz,
    source_catalog=sources_df,
    aperture_radius_arcsec=15.0,
)
# Returns: DataFrame with source_id, freq_hz, flux_jy, flux_err_jy

# Fit spectral index per source
from dsa110_contimg.core.imaging.spectral_analysis import fit_spectral_index

for source_id, source_sed in seds.groupby("source_id"):
    alpha, alpha_err = fit_spectral_index(
        freq_hz=source_sed["freq_hz"].values,
        flux_jy=source_sed["flux_jy"].values,
        flux_err_jy=source_sed["flux_err_jy"].values,
    )
    print(f"{source_id}: α = {alpha:.2f} ± {alpha_err:.2f}")
```

## Result Structure

```python
# spectral_index_results dict
{
    "success": True,
    "alpha_map_path": "/data/spectral_index_map.fits",
    "alpha_error_map_path": "/data/spectral_index_error.fits",
    "source_seds": DataFrame,  # Per-source SEDs
    "statistics": {
        "median_alpha": -0.72,
        "std_alpha": 0.25,
        "n_pixels_fitted": 15000,
        "n_sources_fitted": 150,
    },
}
```

## Configuration

```python
from dsa110_contimg.common.unified_config import settings

# Spectral index settings
settings.spectral_index.enabled = True
settings.spectral_index.min_frequencies = 2
settings.spectral_index.min_snr = 5.0
settings.spectral_index.generate_plots = True
```

## Output Directory

```bash
output_dir/spectral_analysis/
├── spectral_index_map.fits     # α map
├── spectral_index_error.fits   # α uncertainty map
├── source_seds.csv             # Per-source SEDs
├── spectral_index_histogram.png
└── bright_source_seds/         # SED plots for bright sources
    ├── NVSS_J123456+420312_sed.png
    └── ...
```

## Visualization

```python
from dsa110_contimg.core.visualization.sed_plots import (
    plot_spectral_index_map,
    plot_source_sed,
)

# Spectral index map with colorbar
fig = plot_spectral_index_map(
    alpha_map=alpha_map,
    wcs=wcs,
    vmin=-2.0,
    vmax=1.0,
    cmap="coolwarm",
)
fig.savefig("/data/alpha_map.png")

# Individual source SED
fig = plot_source_sed(
    freq_hz=source_sed["freq_hz"],
    flux_jy=source_sed["flux_jy"],
    flux_err_jy=source_sed["flux_err_jy"],
    alpha=alpha,
    source_name="NVSS J123456+420312",
)
fig.savefig("/data/sed_example.png")
```

## CLI Commands

```bash
# Compute spectral index map
dsa110 spectral-index compute \
    /data/img_1280.fits /data/img_1400.fits /data/img_1520.fits \
    --output /data/alpha_map.fits

# Extract source SEDs
dsa110 spectral-index seds \
    --catalog /data/sources.csv \
    --images /data/img_*.fits \
    --output /data/seds.csv
```

## Quality Thresholds

| Metric | Good | Warning | Notes |
|--------|------|---------|-------|
| α uncertainty | <0.3 | 0.3-0.5 | DSA-110 bandwidth limited |
| S/N per band | >5 | 3-5 | Minimum for reliable fit |
| χ² fit | <3 | 3-10 | Curvature may indicate issues |

## Common Issues

| Problem | Cause | Solution |
|---------|-------|----------|
| Large α errors | Narrow bandwidth | Use external catalogs |
| Systematic α offset | Flux calibration | Cross-check with NVSS |
| α varies with position | Primary beam | Apply PB correction first |
| Many masked pixels | Low S/N | Increase integration time |

## ESE Science Relevance

Spectral index can help characterize ESE candidates:

- **Flat/inverted α during event**: Indicates plasma lens or synchrotron self-absorption
- **Steeper α**: Could indicate spectral aging or scattering
- **α unchanged**: Suggests achromatic flux change (geometric)

## Multi-Epoch Spectral Index

Track α over time for variable sources:

```python
# Query spectral indices from database
SELECT source_id, mjd, alpha, alpha_err
FROM spectral_indices
WHERE source_id = 'NVSS_J123456+420312'
ORDER BY mjd;

# Monitor for spectral evolution
```

## Related Resources

- Imaging skill: `.agent/skills/imaging/SKILL.md`
- Photometry skill: `.agent/skills/photometry/SKILL.md`
- Crossmatch skill: `.agent/skills/crossmatch/SKILL.md`
