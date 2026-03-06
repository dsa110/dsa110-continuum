# Reference: VAST Pipeline Cross-Reference

Source: /data/radio-pipelines/askap-vast/
Components analysed: vast-pipeline, vast-post-processing, forced_phot

VAST is an ASKAP continuum variability survey. DSA-110 shares its variability
science goals but differs in instrument, source finder, and calibration approach.
This document captures what DSA-110 should adopt or adapt from VAST.

---

## Variability metrics (adopt verbatim)

Source: vast-pipeline/vast_pipeline/pipeline/pairs.py, utils.py

All three metrics cite Mooley et al. (2016), ApJ 818, 105 -- the same reference
used in dsa110-contimg.

Vs (2-epoch t-statistic):
  Vs = (Sa - Sb) / hypot(sigma_a, sigma_b)
  Computed for flux_peak and flux_int independently: vs_peak, vs_int
  All pairwise combinations across epochs (not just consecutive):
    itertools.combinations(measurement_ids, 2), ordered chronologically

m (2-epoch modulation index):
  m = 2 * (Sa - Sb) / (Sa + Sb)
  Produces m_peak and m_int

eta (weighted variance, get_eta_metric in utils.py):
  eta = (N/(N-1)) * (mean(w*S^2) - mean(w*S)^2/mean(w))
  where w = 1/sigma^2
  Returns 0 if N=1.

v (coefficient of variation):
  v = std(fluxes) / mean(fluxes)   (unweighted)

Per-source aggregate (finalise.py calculate_measurement_pair_aggregate_metrics):
  Among all pairs with |Vs| >= 4.3, find the pair with maximum |m|.
  Store: vs_abs_significant_max_{peak/int}, m_abs_significant_max_{peak/int}

VAST validated thresholds (settings.py lines 313-343):
  min |Vs| for pair to enter aggregate: 4.3 (empirical, ASKAP noise)
  new_source_min_sigma: 5.0
  monitor_min_sigma (forced extraction): 3.0

NOTE: The Vs=4.3 threshold is ASKAP/VAST-specific. It must be revalidated for
DSA-110's noise characteristics and beam size.

---

## Forced photometry (adopt with minor changes)

Source: forced_phot/forced_phot/forced_phot.py, class ForcedPhot

The VAST forced_phot library is directly reusable with DSA-110 FITS images
if the images have standard BMAJ/BMIN/BPA headers and are accompanied by
background and noise maps.

Aperture model: 2D Gaussian matched to restoring beam (BMAJ/BMIN/BPA from FITS).
Cutout size: nbeam=3 -> square cutout of side 3*BMAJ pixels.

Matched-filter formula (_convolution, lines 193-212, numba JIT):
  flux     = sum(d * K/n^2) / sum(K^2/n^2)
  flux_err = sum(K/n) / sum(K/n^2)
  chisq    = sum(((d - K*flux)/n)^2)
  where d = image - background, K = 2D Gaussian kernel, n = noise map

Numba JIT (@njit(cache=True)) for ~3-5x speedup.

Background: loaded from a pre-computed background map (ASKAPSoft meanMap).
VAST does not recompute the background internally.
DSA-110 will need to provide a background map from BANE or similar.

Cluster handling: sources within 1.5*BMAJ of each other (KDTree) are fit
simultaneously with a multi-component deblending model.

Upper limits: NOT implemented in VAST. Sources below min_sigma=3.0 are
pre-filtered and simply not extracted. There is no upper-limit column.
DSA-110 should implement explicit 3-sigma upper limits for non-detections.

---

## Condon (1997) errors (adopt with floor adjustment)

Source: vast-pipeline/vast_pipeline/image/utils.py, calc_condon_flux_errors

Alpha exponents (hardcoded, from TraP):
  alpha_maj1=2.5, alpha_min1=0.5  (for RA position error)
  alpha_maj2=0.5, alpha_min2=2.5  (for Dec position error)
  alpha_maj3=1.5, alpha_min3=1.5  (for amplitude error)

Peak flux uncertainty:
  With frac_flux_cal_error=0 and clean_bias=0 (defaults):
    sigma_peak = sqrt(2) * Sp / rho3
  (pure thermal noise term)

Floor values (ASKAP-calibrated -- revalidate for DSA-110):
  FLUX_DEFAULT_MIN_ERROR = 0.001 mJy
  POS_DEFAULT_MIN_ERROR  = 0.01 arcsec

---

## Flux scale correction (adopt approach)

Source: vast-post-processing/corrections.py

VAST cross-matches against RACS Epoch 0 as flux/astrometry reference.
DSA-110 should use NVSS or FIRST instead.

Method:
1. Cross-match Selavy catalog against RACS at 10 arcsec radius
2. Select: point sources, SNR > 20, isolated (nearest source > 1 arcmin),
   within 6.67 deg of field center, 5-sigma clip on flux ratio
3. Compute astrometric offset: median RA + Dec offset (MADFM uncertainty)
4. Compute flux scale: Huber robust regression of S_VAST vs S_RACS
     S_VAST = gradient * S_RACS + offset
5. Apply: multiplicative correction = 1/gradient, additive = -offset
6. Error propagation: sigma_corrected = sqrt(sigma_scale^2 * S^2 + scale^2 * sigma^2 + sigma_offset^2)

The Huber regression (not OLS) is important -- it is robust to outliers
from variable sources, which would corrupt a naive mean flux ratio.

---

## Source association (consider de Ruiter)

Source: vast-pipeline/vast_pipeline/pipeline/association.py

Three methods: basic (nearest-neighbor at fixed arcsec), advanced (all within radius),
deruiter (de Ruiter radius, scale-free by positional errors).

De Ruiter radius formula:
  dr = sqrt(
    cos^2((dec1+dec2)/2) * (ra1-ra2)^2 / (sigma_ra1^2 + sigma_ra2^2)
    + (dec1-dec2)^2 / (sigma_dec1^2 + sigma_dec2^2)
  )

Default parameters (settings.py):
  association_radius: 10.0 arcsec (basic/advanced)
  association_de_ruiter_radius: 5.68 (dimensionless)
  association_beamwidth_limit: 1.5 * BMAJ (de Ruiter search radius multiplier)
  association_epoch_duplicate_radius: 2.5 arcsec (within-epoch deduplication)

NOTE: 10 arcsec basic radius is calibrated for ASKAP (~25 arcsec beam). DSA-110
has a ~5-15 arcsec beam; the association radius should scale accordingly.
The de Ruiter approach is more physically motivated for multi-epoch matching.

Weighted average position update after each epoch:
  weight_ew = sum(1/uncertainty_ew^2)
  wm_ra = sum(interim_ew) / weight_ew
  Only non-forced measurements contribute to position average.

---

## What VAST does NOT provide for DSA-110

  ESE detection module: not present anywhere in the VAST codebase.
  AOFlagger / flagging strategy: VAST uses ASKAPSoft C++ (not accessible here).
  Self-calibration: VAST delegates entirely to ASKAPSoft (yandasoft C++).
  Formal upper limits: no upper-limit column exists in the VAST output schema.
  Source finder integration: VAST is hardcoded to Selavy; DSA-110 uses PyBDSF/Aegean.

---

## VAST pipeline architecture (reference only)

Backend: Django ORM + PostgreSQL + Q3C spherical index extension
Intermediate storage: Apache Parquet (measurements, sources, associations,
                      relations, measurement_pairs)
Parallelism: Dask process scheduler across sky-region groups and images
Web UI: Django REST + React

The Parquet-based intermediate storage and Dask parallelism are directly
adoptable by DSA-110. The Django/PostgreSQL backend is VAST-specific and
heavier than DSA-110 currently needs (DSA-110 uses SQLite).
