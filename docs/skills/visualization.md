# Visualization Skill

Expert guidance for CARTA visualization in the DSA-110 pipeline.

## Overview

The visualization stage integrates CARTA (Cube Analysis and Rendering Tool for Astronomy) into the pipeline for quick-look visualization, moment map generation, and HDF5 IDIA format conversion for faster viewing.

## Key Modules

| Module | Purpose |
|--------|---------|
| `workflow/pipeline/stages/visualization.py` | `CARTAVisualizationStage` |
| `core/visualization/carta_integration.py` | CARTA API wrapper |
| `core/visualization/fits_to_idia.py` | HDF5 IDIA conversion |
| `core/visualization/moment_maps.py` | Moment map generation |
| `core/visualization/plot_manager.py` | General plot management |

## CARTA Configuration

```python
from dsa110_contimg.workflow.pipeline.stages.visualization import CARTAVisualizationConfig

config = CARTAVisualizationConfig(
    # CARTA server
    carta_url="http://localhost:9002",
    carta_timeout_seconds=30.0,
    
    # HDF5 conversion
    convert_to_hdf5=True,
    hdf5_cache_dir="/data/hdf5_cache",
    compute_statistics=True,
    
    # Moment maps
    generate_moments=True,
    moment_types=[0, 1, 2],  # Integrated, velocity, dispersion
    moment_threshold_sigma=3.0,
    
    # Visualization
    auto_open_viewer=False,
    default_colormap="viridis",
    
    # Output
    save_previews=True,
    preview_size=(800, 800),
)
```

## Pipeline Stage Usage

```python
from dsa110_contimg.workflow.pipeline.stages.visualization import CARTAVisualizationStage

stage = CARTAVisualizationStage(config)
context = PipelineContext(
    config=pipeline_config,
    outputs={
        "image_path": "/data/image.fits",
        "image_paths": ["/data/img1.fits", "/data/img2.fits"],  # Optional
    }
)

result = stage.execute(context)
products = result.outputs["visualization_products"]
```

## Result Products

```python
# VisualizationStageResult structure
{
    "success": True,
    "products": [
        VisualizationProduct(
            product_type="hdf5",
            input_path="/data/image.fits",
            output_path="/data/hdf5_cache/image.hdf5",
            viewer_url="http://localhost:9002/carta/?file=image.hdf5",
        ),
        VisualizationProduct(
            product_type="moment",
            input_path="/data/cube.fits",
            output_path="/data/moments/cube_mom0.fits",
        ),
        VisualizationProduct(
            product_type="preview",
            input_path="/data/image.fits",
            output_path="/data/previews/image.png",
        ),
    ],
    "errors": [],
    "warnings": [],
    "timing": {"total_seconds": 15.3},
}
```

## HDF5 IDIA Conversion

CARTA performs best with HDF5 IDIA format (pre-computed statistics):

```python
from dsa110_contimg.core.visualization.fits_to_idia import convert_to_idia_hdf5

output_path = convert_to_idia_hdf5(
    fits_path=Path("/data/image.fits"),
    output_dir=Path("/data/hdf5_cache"),
    compute_stats=True,  # Pre-compute statistics
    overwrite=False,
)
```

### Performance Comparison

| Operation | FITS | HDF5 IDIA |
|-----------|------|-----------|
| Initial load | 5-10s | <1s |
| Zoom/pan | 0.5s | <0.1s |
| Statistics | 2-5s | Instant |
| Memory | High | Efficient |

## Moment Maps

For spectral cubes, generate moment maps:

```python
from dsa110_contimg.core.visualization.moment_maps import generate_moment_maps

moments = generate_moment_maps(
    cube_path=Path("/data/cube.fits"),
    output_dir=Path("/data/moments"),
    moment_types=[0, 1, 2],
    threshold_sigma=3.0,
    velocity_range_km_s=(-100, 100),  # Optional
)
# Returns: {
#     0: "/data/moments/cube_mom0.fits",  # Integrated intensity
#     1: "/data/moments/cube_mom1.fits",  # Velocity field
#     2: "/data/moments/cube_mom2.fits",  # Velocity dispersion
# }
```

### Moment Type Reference

| Moment | Name | Description |
|--------|------|-------------|
| 0 | Integrated intensity | Sum over velocity/frequency |
| 1 | Velocity field | Intensity-weighted mean velocity |
| 2 | Velocity dispersion | Intensity-weighted velocity σ |
| 8 | Peak intensity | Maximum along spectral axis |
| 9 | Peak velocity | Velocity at peak intensity |

## PNG Preview Generation

```python
from dsa110_contimg.core.visualization.plot_manager import PlotManager

manager = PlotManager(output_dir=Path("/data/previews"))

preview_path = manager.create_preview(
    fits_path=Path("/data/image.fits"),
    size=(800, 800),
    colormap="viridis",
    scale="asinh",
    stretch_percent=(0.5, 99.5),
    overlay_sources=sources_df,  # Optional catalog overlay
)
```

## CARTA API Integration

```python
from dsa110_contimg.core.visualization.carta_integration import CARTAClient

async with CARTAClient(url="http://localhost:9002") as client:
    # Load image
    session = await client.open_file("/data/image.hdf5")
    
    # Set colormap and scaling
    await session.set_colormap("viridis")
    await session.set_scaling("log", min_val=0.001, max_val=1.0)
    
    # Get rendered image
    png_data = await session.get_raster_image(width=800, height=800)
    
    # Export PNG
    with open("/data/preview.png", "wb") as f:
        f.write(png_data)
```

## Configuration

```python
from dsa110_contimg.common.unified_config import settings

# Visualization settings
settings.visualization.carta_url = "http://localhost:9002"
settings.visualization.convert_to_hdf5 = True
settings.visualization.hdf5_cache_dir = "/data/hdf5_cache"
settings.visualization.generate_moments = True
settings.visualization.save_previews = True
```

## CARTA Server Setup

```bash
# Start CARTA server (Docker)
docker run -d \
    --name carta-server \
    -p 9002:9002 \
    -v /data:/data:ro \
    cartavis/carta:latest

# Or via systemd service
sudo systemctl start carta-server
```

## Output Directory Structure

```bash
visualization/
├── hdf5_cache/          # HDF5 IDIA files
│   ├── image1.hdf5
│   └── image2.hdf5
├── moments/             # Moment maps
│   ├── cube_mom0.fits
│   ├── cube_mom1.fits
│   └── cube_mom2.fits
├── previews/            # PNG previews
│   ├── image1.png
│   └── image2.png
└── comparisons/         # Multi-image comparisons
    └── before_after.png
```

## CLI Commands

```bash
# Generate preview
dsa110 viz preview /data/image.fits --output /data/preview.png

# Convert to HDF5 IDIA
dsa110 viz convert /data/image.fits --format idia

# Generate moment maps
dsa110 viz moments /data/cube.fits --types 0,1,2

# Open in CARTA
dsa110 viz open /data/image.fits
```

## Multi-Image Comparison

For before/after or multi-epoch comparison:

```python
from dsa110_contimg.core.visualization.plot_manager import create_comparison_plot

fig = create_comparison_plot(
    images=["/data/epoch1.fits", "/data/epoch2.fits", "/data/epoch3.fits"],
    labels=["Epoch 1", "Epoch 2", "Epoch 3"],
    colormap="viridis",
    same_scale=True,  # Use same intensity scale for all
)
fig.savefig("/data/comparison.png")
```

## Common Issues

| Problem | Cause | Solution |
|---------|-------|----------|
| CARTA timeout | Large file | Use HDF5 IDIA format |
| Black preview | Wrong scaling | Adjust stretch_percent |
| Missing moments | Not a cube | Check NAXIS3 > 1 |
| Connection refused | CARTA not running | Start CARTA server |

## Performance Tips

1. **Pre-convert to HDF5 IDIA** for interactive viewing
2. **Use async API** for batch processing
3. **Cache previews** to avoid regeneration
4. **Set preview_size** appropriately (larger = slower)

## Related Resources

- Imaging skill: `.agent/skills/imaging/SKILL.md`
- Validation skill: `.agent/skills/validation/SKILL.md`
- Pre-imaging QA skill: `.agent/skills/pre-imaging-qa/SKILL.md`
