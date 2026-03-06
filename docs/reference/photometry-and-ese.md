# Reference: Photometry and ESE Detection

Source: /data/dsa110-contimg/backend/src/dsa110_contimg/core/photometry/
Files: forced.py, normalize.py, ese_detection.py, scoring.py, variability.py,
       thresholds.py, aegean_fitting.py

---

## Forced photometry method: Condon 1997 matched-filter

measure_forced_peak() (forced.py line 530). Default parameters:
  box_size_pix = 5
  annulus_pix  = (30, 50)
  nbeam        = 3.0          (cutout side = 3 * BMAJ pixels)
  use_weighted_convolution = True

When BMAJ/BMIN/BPA are available from FITS header, uses a 2D Gaussian kernel
(PSF-matched) via _weighted_convolution.

Formula (Condon 1997):
  flux     = sum(d * K/sigma^2) / sum(K^2/sigma^2)
  flux_err = sum(sigma * K/sigma^2) / sum(K/sigma^2)
  chi_sq   = sum(((d - K*flux)/sigma)^2)

where d = image - background, K = 2D Gaussian kernel, sigma = noise map.

Numba acceleration (@njit(cache=True)) when available, ~2-5x speedup.
GPU path: via get_array_module() if settings.gpu.prefer_gpu is set.

Background subtraction: from noise map if provided, else annulus-based RMS.

NOTE: This is NOT a traditional fixed-aperture design (the legacy dsa110-contimg-legacy
used a 6" aperture with 9"-15" annulus). The current reference implementation uses
Condon 1997 matched-filter convolution with the PSF.

---

## Differential photometry normalization (normalize.py)

Reference source selection from master_sources.sqlite3:
  resolved_flag=0 AND confusion_flag=0 AND snr_nvss >= 50.0
  within fov_radius_deg = 1.5
  max_sources = 20
  minimum valid references = 3 (else correction not applied)

Baseline: median of first n_baseline_epochs=10 epochs, MAD-based RMS (1.4826 * MAD).
Per-epoch correction factor: median ratio of current to baseline across ensemble.
Sigma-clipping: max_deviation_sigma=3.0 to reject variable references.
Normalized flux = raw_flux / correction_factor.

---

## Source finding: BANE + Aegean (aegean_fitting.py)

BANE is called as a subprocess with no custom parameters (BANE defaults).
  _run_bane(fits_path): BANE {fits_path} with 300-second timeout
  Outputs: {stem}_rms.fits and {stem}_bkg.fits

Aegean runs in --priorized (forced fitting) mode:
  PSF from FITS header (BMAJ/BMIN/BPA)
  Input: prior source catalog
  Output columns: ra_deg, dec_deg, peak_flux_jy, err_peak_flux_jy, local_rms_jy,
                  integrated_flux_jy, err_integrated_flux_jy, a_arcsec, b_arcsec,
                  pa_deg, success, error_message

Status: NOT YET PORTED to dsa110-continuum. Stated as a future task in CLAUDE.md.
The aegean_fitting.py module is the reference to port.

---

## ESE detection algorithm (ese_detection.py)

detect_ese_candidates() line 31:

1. Query monitoring_sources table:
     SELECT ... WHERE sigma_deviation >= min_sigma
   Default min_sigma = 5.0 (CONSERVATIVE preset, see thresholds.py)
   Order: sigma_deviation DESC

2. Fields from table: source_id, ra_deg, dec_deg, nvss_flux_jy, mean_flux_jy,
   std_flux_jy, chi_squared, sigma_deviation, eta, n_detections, last_detected_at

3. For each qualifying source:
   - If ese_candidates table has status='active' for source_id: update significance
   - Else: insert with flag_type='auto', status='active'

4. Optional composite scoring (use_composite_scoring=False by default):
   calls calculate_composite_score() from scoring.py

Epoch alignment: monitoring_sources stores pre-aggregated statistics
(mean_flux_jy, std_flux_jy, sigma_deviation). MJD conversion:
  float(row["last_detected_at"]) / 86400.0 + 40587.0

---

## Variability metrics (variability.py, cited Mooley et al. 2016)

All three primary metrics cite Mooley et al. (2016), ApJ 818, 105.

eta metric (calculate_eta_metric, line 16):
  eta = (N / (N-1)) * (mean(w*f^2) - mean(w*f)^2 / mean(w))
  where w = 1/sigma^2
  Returns 0.0 if N <= 1 or fewer than 2 valid measurements.

Vs metric (calculate_vs_metric, line 86):
  Vs = (flux_a - flux_b) / hypot(sigma_a, sigma_b)
  Uses np.hypot (numerically stable).

m metric (calculate_m_metric, line 131):
  m = 2 * (flux_a - flux_b) / (flux_a + flux_b)

sigma_deviation (calculate_sigma_deviation, line 200) -- primary ESE trigger:
  sigma_deviation = max(|max_flux - mean| / std, |min_flux - mean| / std)
  where std uses ddof=1 (sample standard deviation)

---

## Validated variability thresholds (thresholds.py)

| Preset     | min_sigma | min_chi2_nu | min_eta |
|------------|-----------|-------------|---------|
| CONSERVATIVE | 5.0     | 4.0         | 3.0     |
| MODERATE     | 3.5     | 2.5         | 2.0     |
| SENSITIVE    | 2.5     | 1.5         | 1.0     |

Primary production threshold: min_sigma = 5.0 (CONSERVATIVE).
These values are adopted from VAST Tools literature, not empirically calibrated
for DSA-110 data distributions. No false-positive rate estimate is documented.
Revalidation against DSA-110 data is required before declaring thresholds final.

---

## Composite ESE scoring system (scoring.py)

DEFAULT_WEIGHTS:
  sigma_deviation: 0.5
  chi2_nu: 0.3
  eta_metric: 0.2

CONFIDENCE_THRESHOLDS:
  high:   7.0
  medium: 4.0
  low:    0.0

Normalization: sigma_deviation and chi2_nu normalized over [0,10];
eta_metric over [0,5]. Composite score = weighted sum, scaled to [0,10].

Enabled by use_composite_scoring=True (default is False).

---

## Catalog infrastructure (catalog/query.py, catalog/builders.py)

Supported catalogs: nvss, first, rax (=racs), vlass, master, atnf

Storage: SQLite at /data/dsa110-contimg/state/catalogs/
  Per-dec strip: {type}_dec{dec:+.1f}.sqlite3 (6-degree fuzzy match)
  Full-sky fallback: {type}_full.sqlite3

Query: SQL box pre-filter, then exact astropy great-circle separation.
All queries return pd.DataFrame with ra_deg, dec_deg, flux_mjy at minimum.

Master catalog columns (pre-built cross-match):
  snr_nvss, s_nvss, s_vlass, s_first, s_rax, alpha, resolved_flag,
  confusion_flag, has_nvss, has_vlass, has_first, has_rax

Catalog coverage limits:
  nvss:  dec -40 to +90
  first: dec -40 to +90
  rax:   dec -90 to +49.9
  vlass: dec -40 to +90
  atnf:  all-sky

Sky model seeding (skymodels.py):
  make_unified_wsclean_list(center_ra, center_dec, radius_deg, min_mjy=2.0, freq_ghz=1.4)
  Source priority: FIRST > RACS > NVSS
