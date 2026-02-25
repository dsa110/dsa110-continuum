# GPU Acceleration Skill

Expert guidance for configuring and using GPU acceleration in the DSA-110 imaging pipeline.

> **Pipeline Context**: For pipeline-level GPU decisions (when to use GPU, gridder selection, performance trade-offs), see `.agent/skills/pipeline-advisor/GPU_CONTEXT.md`

## h17 (lxd110h17) System Stack

### Software Versions (Production)

| Component | Version | Location | Notes |
|-----------|---------|----------|-------|
| **WSClean** | 3.6 (master, 2025-02-07) | /opt/wsclean | Built from master for EveryBeam 0.8 support |
| **EveryBeam** | 0.8.0 | /opt/everybeam | DSA-110 Airy beam support (added v0.7.2) |
| **IDG** | 0.8.1 | /opt/wsclean/lib | GPU gridder, rebuilt outside conda |
| **Casacore** | 3.7.1 | conda (casa6) | Required for EveryBeam 0.8.0 |
| **CUDA** | 11.1 / 11.4 (driver) | /usr/local/cuda-11.1 | Turing architecture (sm_75) |
| **UCX** | system | /usr/local/ucx | Signal handling fix for IDG |

### Hardware Specifications

| Property | Value |
|----------|-------|
| **GPUs** | 2× NVIDIA GeForce RTX 2080 Ti |
| **Memory per GPU** | 11 GB GDDR6 |
| **Total GPU Memory** | 22 GB |
| **Compute Capability** | 7.5 (Turing) |
| **Driver Version** | 470.256.02 |
| **GPU 0 Bus ID** | 00000000:3B:00.0 |
| **GPU 1 Bus ID** | 00000000:AF:00.0 |
| **CPU** | Intel Xeon Silver 4210 @ 2.20GHz |
| **System RAM** | 94 GB |

---

## WSClean Wrapper

The production wrapper at `/usr/local/bin/wsclean` handles:

1. **UCX signal fix** - Critical for IDG GPU mode
2. **Library paths** - EveryBeam, IDG, CUDA
3. **Beam caching** - Auto-configured for DSA-110

### Wrapper Contents

```bash
#!/bin/bash
# WSClean wrapper with EveryBeam 0.8.0 + IDG GPU + beam caching
# Updated: 2026-02-03

# UCX signal handling - CRITICAL for IDG GPU
export UCX_ERROR_SIGNALS=""
export UCX_LOG_LEVEL=error

# Force system UCX (not conda)
export LD_PRELOAD="/usr/local/ucx/lib/libucp.so.0:..."

# Library paths
export LD_LIBRARY_PATH="/opt/wsclean/lib:/opt/everybeam/lib:/usr/local/cuda-11.1/lib64:..."

# Beam caching defaults (DSA-110 static beam)
export WSCLEAN_TEMP_DIR="${WSCLEAN_TEMP_DIR:-/dev/shm/wsclean-cache}"

# Auto-inject beam caching args unless user specifies
# -temp-dir /dev/shm/wsclean-cache
# -beam-aterm-update 1800  (conservative default; any value ≥ obs duration works for static beams)

exec /opt/wsclean/bin/wsclean $BEAM_CACHE_ARGS "$@"
```

### Verification

```bash
# Check wrapper features
wsclean --version
# WSClean version 3.6 (2025-02-07)
# EveryBeam is available.
# IDG is available.
# WGridder is available.

# Check EveryBeam libraries linked
ldd /opt/wsclean/bin/wsclean | grep everybeam
# libeverybeam.so
# libeverybeam-hamaker.so
# libeverybeam-oskar.so
# ...
```

---

## EveryBeam DSA-110 Support

### DSA-110 Beam Model

EveryBeam 0.7.2+ includes **DSA-110 Airy beam pattern**:

```cpp
// EveryBeam/telescope/dsa110.h
class Dsa110 final : public Telescope {
    // Airy disk beam pattern implementation
    // Auto-detected from MS TELESCOPE_NAME
};
```

### Beam Correction Modes

| Flag | Purpose | When to Use |
|------|---------|-------------|
| `-apply-primary-beam` | Post-imaging beam correction | Quick imaging, single pointings |
| `-grid-with-beam` | A-projection (direction-dependent) | **Mosaics, wide-field science** |
| `-reuse-primary-beam` | Load cached beam images | Repeated runs on same field |

### Example: Science Imaging with Beam Correction

```bash
# 4k science imaging with direction-dependent beam
wsclean -name science_4k \
    -size 4096 4096 -scale 3asec \
    -niter 100000 -auto-threshold 3 \
    -mgain 0.85 -weight briggs 0.5 \
    -grid-with-beam \
    -use-wgridder \
    observation.ms

# GPU + beam correction (IDG)
wsclean -name science_4k_gpu \
    -size 4096 4096 -scale 3asec \
    -niter 100000 -auto-threshold 3 \
    -grid-with-beam \
    -use-idg -idg-mode gpu \
    observation.ms
```

---

## Beam Caching Configuration

### Default Settings (Auto-Injected)

| Setting | Value | Purpose |
|---------|-------|---------|
| `-temp-dir` | `/dev/shm/wsclean-cache/` | RAM disk for fast I/O |
| `-beam-aterm-update` | 1800 sec | DSA-110 beam is static |

### Cache Directory

```bash
# Created at boot via tmpfiles.d
/dev/shm/wsclean-cache/  # RAM disk, 1777 permissions

# Config file
/etc/tmpfiles.d/wsclean-cache.conf
# d /dev/shm/wsclean-cache 1777 root root -
```

### Override Defaults

```bash
# Force more frequent beam recalculation
wsclean -beam-aterm-update 300 ...

# Use different temp directory
wsclean -temp-dir /stage/tmp ...

# Reuse beam from previous run
wsclean -reuse-primary-beam ...
```

### What Gets Cached

| Cache Type | Location | Size | Purpose |
|------------|----------|------|---------|
| ATerm beam values | temp-dir | ~100 MB | Grid-time beam |
| Average beam matrices | temp-dir | ~50 MB | Major cycle state |
| Primary beam images | output-dir | ~100 MB/image | Post-imaging correction |

---

## IDG GPU Configuration

### Modes

| Mode | Description | Memory | Recommended |
|------|-------------|--------|-------------|
| `gpu` | Pure GPU | High (~10 GB for 4k) | **Default** - fails on GPU error |
| `hybrid` | GPU + CPU | Medium (~6 GB) | Only when accepting CPU fallback |
| `cpu` | CPU only | Low | No GPU, want IDG algorithm |

### Why `gpu` Mode is Default

- **Explicit failure**: Job fails if GPU fails = visible problems
- **No silent degradation**: `hybrid` silently falls back to CPU
- **Predictable performance**: Always GPU speed or error

### Memory Guidelines (RTX 2080 Ti, 11 GB)

| Image Size | IDG Mode | Memory Usage | Status |
|------------|----------|--------------|--------|
| 2048×2048 | gpu | ~4 GB | ✅ Safe |
| 4096×4096 | gpu | ~10 GB | ✅ Works |
| 4096×4096 | hybrid | ~6 GB | ✅ Safe |
| 8192×8192 | hybrid | ~12 GB | ⚠️ May OOM |
| 8192×8192 | wgridder | ~3 GB | ✅ CPU fallback |

---

## Performance Benchmarks (h17)

### IDG GPU vs WGridder (Validated)

| Image Size | IDG GPU | WGridder | Speedup | Notes |
|------------|---------|----------|---------|-------|
| 512×512 | 8m53s | ~2.6s gridding | — | Beam overhead dominates |
| 2048×2048 | 8m47s | ~4 min | ~2× | Beam: 33×15s ≈ 8 min |
| 4096×4096 | ~10 min | ~8 min | ~1.3× | Beam overhead amortized |

**Key insight**: ~8 minute overhead is beam computation (33 beams × ~15s each), not gridding. IDG GPU advantage grows with more cleaning iterations.

### Beam Caching Impact

| Scenario | First Run | Cached Run | Savings |
|----------|-----------|------------|---------|
| Self-cal iteration 1 | 10 min | — | Baseline |
| Self-cal iteration 2 | — | 4 min | 60% faster |
| Re-imaging same field | 10 min | 2 min | 80% faster |

---

## GPU Access Methods

### Direct (Native)

```bash
# Default - use wrapper
wsclean -use-idg ...

# Environment override
export CUDA_VISIBLE_DEVICES=0  # GPU 0 only
```

### Docker

```bash
# Both GPUs
docker run --rm --gpus all dsa110-contimg:gpu wsclean ...

# Specific GPU
docker run --rm --gpus '"device=0"' dsa110-contimg:gpu wsclean ...
```

---

## Python API

### GPU Detection

```python
from dsa110_contimg.common.utils.gpu_utils import (
    is_gpu_available,
    get_gpu_count,
    get_gpu_config,
    build_wsclean_gpu_args,
)

if is_gpu_available():
    config = get_gpu_config()
    print(f"GPUs: {get_gpu_count()}")
    print(f"Gridder: {config.effective_gridder}")

# Build WSClean args
args = build_wsclean_gpu_args()
# Returns: ["-gridder", "idg", "-idg-mode", "gpu"]
```

### Imaging Parameters

```python
from dsa110_contimg.core.imaging.params import ImagingParams

# GPU with beam correction
params = ImagingParams(
    imagename="output",
    gridder="idg",
    grid_with_beam=True,  # EveryBeam direction-dependent
)

# CPU with beam correction
params = ImagingParams(
    imagename="output",
    gridder="wgridder",
    apply_primary_beam=True,  # Post-imaging correction
)
```

### GPU Visibility Prediction

GPU-accelerated degridding for model visibility prediction:

```python
from dsa110_contimg.core.imaging.gpu_predict import (
    predict_model_from_catalog,
    PredictConfig,
    SourceModel,
)

# Predict from catalog sources
result = predict_model_from_catalog(
    ms_path="/stage/observation.ms",
    catalog_sources=[
        {"ra_deg": 202.78, "dec_deg": 30.51, "flux_mjy": 15000, "name": "3C286"},
        {"ra_deg": 203.15, "dec_deg": 30.22, "flux_mjy": 850, "name": "src2"},
    ],
    phase_center_ra=203.0,
    phase_center_dec=30.4,
    min_flux_jy=0.005,  # 5 mJy threshold
    max_sources=50,
    write_model=True,  # Write to MODEL_DATA column
)

if result.success:
    print(f"Predicted {result.n_sources} sources in {result.processing_time_s:.3f}s")
else:
    print(f"Failed: {result.error}")
```

**Performance**: ~30-60× faster than WSClean `-predict`

| Visibilities | GPU Predict | WSClean Predict |
|--------------|-------------|-----------------|
| 100k | ~20ms | ~1s |
| 1M | ~160ms | ~5s |
| 10M | ~1.5s | ~50s |

**Configuration options**:

```python
config = PredictConfig(
    image_size=512,          # Grid size for FFT
    cell_size_arcsec=12.0,   # Pixel scale
    w_planes=32,             # W-projection planes
    max_gpu_gb=8.0,          # GPU memory limit
    ref_freq_hz=1.4e9,       # Reference frequency
)
```

**Spectral index handling**: Automatically looks up spectral indices from master_sources.sqlite3 or VLA calibrators. Sources without spectral index are skipped with warning.

---

## Troubleshooting

### IDG/CUDA Crash

**Symptom**: Signal error or CUDA mapping failure

**Fix**: Ensure wrapper sets `UCX_ERROR_SIGNALS=""`

```bash
# Verify wrapper
grep UCX_ERROR_SIGNALS /usr/local/bin/wsclean
```

### GPU OOM

**Symptom**: `CUDA out of memory`

**Fixes**:
1. Reduce image size
2. Use `-idg-mode hybrid` (accepts CPU fallback)
3. Use `-use-wgridder` (CPU only)

### EveryBeam Not Detected

**Symptom**: `Unknown telescope: DSA-110`

**Fix**: Verify EveryBeam installed and linked

```bash
ldd /opt/wsclean/bin/wsclean | grep everybeam
# Should show libeverybeam.so
```

### Beam Cache Issues

**Symptom**: Slow repeated runs

**Fix**: Check temp-dir is on fast storage

```bash
# Should be RAM disk
df -h /dev/shm/wsclean-cache/
```

---

## Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `PIPELINE_GPU_ENABLED` | auto | Override GPU detection |
| `PIPELINE_GPU_DEVICES` | all | GPU indices (0,1) |
| `PIPELINE_GPU_GRIDDER` | idg | WSClean gridder |
| `PIPELINE_GPU_IDG_MODE` | gpu | IDG execution mode |
| `CUDA_VISIBLE_DEVICES` | — | Limit visible GPUs |
| `UCX_ERROR_SIGNALS` | "" | **CRITICAL** for IDG |
| `WSCLEAN_TEMP_DIR` | /dev/shm/wsclean-cache | Beam cache location |

---

## Key Files

| File | Purpose |
|------|---------|
| `/usr/local/bin/wsclean` | Production wrapper |
| `/opt/wsclean/bin/wsclean` | WSClean binary |
| `/opt/everybeam/lib/` | EveryBeam libraries |
| `/opt/everybeam/include/EveryBeam/telescope/dsa110.h` | DSA-110 beam header |
| `/dev/shm/wsclean-cache/` | Beam cache directory |
| `/etc/tmpfiles.d/wsclean-cache.conf` | Cache dir boot config |
| `docs/reference/IDG_GPU_SETUP.md` | Detailed build notes |

---

## Quick Reference

```bash
# Standard GPU imaging with beam correction
wsclean -name output \
    -size 4096 4096 -scale 6asec \
    -niter 10000 -auto-mask 5 -auto-threshold 1 \
    -grid-with-beam \
    -use-idg -idg-mode gpu \
    observation.ms

# Reuse beam for self-cal iterations
wsclean -name selfcal_iter2 \
    -reuse-primary-beam \
    -grid-with-beam \
    -use-idg -idg-mode gpu \
    observation.ms

# CPU-only with beam correction
wsclean -name output_cpu \
    -apply-primary-beam \
    -use-wgridder \
    observation.ms

# Check GPU status
nvidia-smi
watch -n 1 nvidia-smi  # Live monitoring
```
