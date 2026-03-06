# Reference: Mosaicking

Source: /data/dsa110-contimg/backend/src/dsa110_contimg/core/mosaic/
Files: builder.py, wsclean_mosaic.py, jobs_wsclean.py

## Two-tier architecture

| Tier | Method | When used | Timeout |
|---|---|---|---|
| QUICKLOOK | Image-domain linear mosaicking on deconvolved FITS | < 1 h | 5 min |
| SCIENCE | Visibility-domain joint WSClean deconvolution | 1-48 h | 30 min |
| DEEP | Visibility-domain joint WSClean deconvolution | > 48 h | 120 min |

## QUICKLOOK tier (build_mosaic, builder.py)

Image-domain only -- no MS access required. Fast.

Regridding: nearest-neighbor only (scipy.ndimage.map_coordinates, order=0).
  Comment: "This is the WSClean-style approach." No sinc/bilinear interpolation.
Max output grid: 8192 x 8192 px; larger grids rescaled with pixel scale adjustment.
Primary beam model: Airy disk, dish diameter 4.7 m (builder.py line 251).
  Formula: (2 * J1(x) / x)^2
Weight combination: final_weight = rms_weight * pb_weight^2 (PB^2 = weighting variance).
PB floor: pb_cutoff = 0.1 (< 10% response clipped).
Output: mosaic FITS + {output}.weights.fits (BUNIT = "1/Jy^2").
RMS estimate: astropy.stats.mad_std (robust).

Note: alignment_order parameter is deprecated and unused (builder.py line 64).
No astrometric cross-matching or source-based alignment is performed.

## SCIENCE/DEEP tier (build_wsclean_mosaic, wsclean_mosaic.py)

Visibility-domain joint deconvolution. Scientifically correct for wide-field mosaics.

### Validated WSCleanMosaicConfig defaults (lines 66-80)

  -use-idg
  -idg-mode cpu              # Default: "cpu" -- MUST be changed to "gpu" on GPU nodes
  -grid-with-beam
  -size 4096 4096
  -scale 1asec
  -niter 50000
  -mgain 0.6
  -auto-threshold 3.0        # sigma
  -parallel-deconvolution 2000
  -local-rms                 # enabled (local_rms=True)

Output files: {prefix}-image.fits and {prefix}-image-pb.fits.
Falls back to {prefix}-MFS-image.fits if neither exists.

### Steps

1. Copy all MS to /dev/shm/mosaic/{output_name}_mosaic/ (RAM disk for speed).
2. Optional calibration: copy designated transit MS, run chgcentre to calibrator
   position, solve cal tables, apply to all mosaic copies.
3. Compute mean meridian: arithmetic mean np.mean(ra_values) of all MS FIELD RAs.
4. Run chgcentre on all mosaic copies to mean meridian RA/Dec.
5. Run WSClean joint deconvolution on all copies.

### KNOWN BUG: mean-RA wrap-around near RA=0/360

wsclean_mosaic.py uses np.mean(ra_values) (arithmetic mean) for the mosaic phase
centre. If any observation spans the RA=0/360 wrap boundary, this produces a wildly
incorrect mean. The per-MS phaseshift in runner.py correctly uses a circular mean --
this fix was NOT applied to the mosaic path.

Impact: any mosaic spanning RA=0 produces an incorrect phase centre, leading to
smeared or absent sources in the combined image. Fix by substituting a circular mean.

### EveryBeam for mosaics

EVERYBEAM_PATH env var, default /opt/everybeam.
LD_LIBRARY_PATH is patched to include {EVERYBEAM_PATH}/lib before WSClean.
-grid-with-beam (A-projection, direction-dependent) is correct for mosaics.
This differs from single-field imaging which uses -apply-primary-beam.

### Default idg_mode is "cpu" -- must change for GPU nodes

WSCleanMosaicConfig default idg_mode="cpu" (comment: "No GPU available").
Must be set to "gpu" explicitly when running on GPU-equipped nodes.
