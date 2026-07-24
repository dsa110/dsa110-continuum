"""Why does the pre-bandpass phase solve null specific antenna-pols?

On the phaseshifted cal MS (3C454.3 at phase centre, 12.5 Jy — phase should be
~flat for a healthy antenna), for each antenna-pol on its baseline to the
refant, measure two coherence numbers:

  freq_coh: |mean over channels of unit visibilities| at fixed time
            -> low = phase winds across the band (uncorrected DELAY;
               combine_spw averaging would destructively cancel)
  time_coh: per-channel |mean over time of unit visibilities|, median over
            channels -> low = phase unstable in TIME (60s solint decorrelates)

Healthy antenna: both ~1.  Delay-broken pol: freq_coh ~0, time_coh ~1.
Time-unstable pol: time_coh ~0.  Dead pol: amplitude ~0.
Also estimates the implied delay from the FFT peak of the band spectrum.
"""
import numpy as np, sys, json
sys.path.insert(0, "/data/dsa110-continuum")
from dsa110_continuum.adapters.casa_tables import table

MS = "/data/dsa110-continuum/outputs/cal-yield-2026-07-10/refant_103/cal_staging/cal_cal.ms"
REF = 102  # name '103'
FULL_NULL = [18, 47, 70, 90, 92, 98, 100, 108]
POL_NULL = [(48, 0), (72, 0), (88, 1), (93, 1), (96, 1)]
CONTROLS = [1, 5, 30, 73, 95]

with table(MS + "/SPECTRAL_WINDOW", readonly=True, ack=False) as t:
    freqs = np.array(t.getcol("CHAN_FREQ"))  # (nspw, nchan)
nspw, nchan = freqs.shape
order = np.argsort(freqs[:, 0])
fgrid = freqs[order].ravel()

with table(MS, readonly=True, ack=False) as t:
    a1 = np.array(t.getcol("ANTENNA1")); a2 = np.array(t.getcol("ANTENNA2"))
    ddid = np.array(t.getcol("DATA_DESC_ID")); times = np.array(t.getcol("TIME"))
    fl_all = None
    targets = sorted(set(FULL_NULL + [p[0] for p in POL_NULL] + CONTROLS))
    results = {}
    for a in targets:
        lo, hi = (a, REF) if a < REF else (REF, a)
        sel = np.where((a1 == lo) & (a2 == hi))[0]
        if len(sel) == 0:
            results[a] = None
            continue
        # build (ntime, nspw, npol, nchan) cube for this baseline
        rows_t = times[sel]; rows_s = ddid[sel]
        ut = np.unique(rows_t)
        cube = np.zeros((len(ut), nspw, 2, nchan), complex)
        have = np.zeros((len(ut), nspw), bool)
        for r, tt, ss in zip(sel, rows_t, rows_s):
            d = t.getcol("DATA", startrow=int(r), nrow=1)[0]  # (2, nchan) rows-first
            if d.shape[0] != 2:
                d = d.T
            it = np.searchsorted(ut, tt)
            cube[it, ss] = d
            have[it, ss] = True
        res = {}
        for p in range(2):
            V = cube[:, order, p, :].reshape(len(ut), -1)   # time x (spw*chan), band-ordered
            amp = np.abs(V)
            med_amp = float(np.median(amp[amp > 0])) if (amp > 0).any() else 0.0
            good = amp > 0
            U = np.where(good, V / np.maximum(amp, 1e-12), 0)
            # freq coherence per time, then median
            nfg = good.sum(axis=1)
            fc = np.abs(U.sum(axis=1)) / np.maximum(nfg, 1)
            freq_coh = float(np.median(fc[nfg > 100])) if (nfg > 100).any() else float("nan")
            # time coherence per channel, then median
            ntg = good.sum(axis=0)
            tc = np.abs(U.sum(axis=0)) / np.maximum(ntg, 1)
            time_coh = float(np.median(tc[ntg > 5])) if (ntg > 5).any() else float("nan")
            # delay estimate: FFT of time-averaged band spectrum
            spec = U.mean(axis=0)
            n = len(spec)
            pad = 8
            ft = np.fft.fft(spec * np.hanning(n), n * pad)
            df = float(np.median(np.diff(fgrid)))
            lags = np.fft.fftfreq(n * pad, d=df)
            k = int(np.argmax(np.abs(ft)))
            delay_ns = float(lags[k] * 1e9)
            peak_frac = float(np.max(np.abs(ft)) / max(np.sum(np.abs(spec) > 0), 1))
            res[p] = dict(med_amp=med_amp, freq_coh=freq_coh, time_coh=time_coh,
                          delay_ns=delay_ns, delay_peak=peak_frac)
        results[a] = res

def tag(a, p):
    if a in FULL_NULL:
        return "FULLNULL"
    if (a, p) in POL_NULL:
        return "POLNULL"
    return "ok"

print(f"{'ant':>4} {'pol':>3} {'class':>9} {'amp':>8} {'freq_coh':>8} {'time_coh':>8} {'delay_ns':>9} {'peak':>6}")
for a in targets:
    r = results.get(a)
    if r is None:
        print(f"{a:>4}   -  no baseline to refant")
        continue
    for p in range(2):
        e = r[p]
        print(f"{a:>4} {p:>3} {tag(a,p):>9} {e['med_amp']:8.4f} {e['freq_coh']:8.3f} {e['time_coh']:8.3f} {e['delay_ns']:9.1f} {e['delay_peak']:6.3f}")

with open("/data/dsa110-continuum/outputs/cal-yield-2026-07-10/prebp_null_diagnostics.json", "w") as f:
    json.dump({str(k): v for k, v in results.items()}, f, indent=1)
print("saved prebp_null_diagnostics.json")
