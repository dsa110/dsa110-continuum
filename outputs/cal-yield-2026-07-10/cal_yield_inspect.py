"""Calibration-yield inspection (read-only) for the 2026-01-25 3C454.3 solve.

Answers, without re-solving:
  1. Per-antenna flag fraction in the production .b and .g tables.
  2. Base cal-MS FLAG fraction (pre/post RFI state).
  3. Raw per-antenna median |DATA| (antenna health, unflagged rows).
  4. Health of refant candidates 103 (current), 104, 105.
  5. Per-antenna count of baselines longer than 1 klambda (the bandpass solve
     uses uvrange '>1klambda' with minblperant=4) — tests whether the ~11
     "good but flagged" antennas are baseline-starved rather than bad.
"""
import json
import numpy as np
import sys

sys.path.insert(0, "/data/dsa110-continuum")
from dsa110_continuum.adapters.casa_tables import table

MS = "/stage/dsa110-contimg/ms/2026-01-25T22:26:05.ms"
BTAB = "/stage/dsa110-contimg/ms/2026-01-25T22:26:05_0~23.b"
GTAB = "/stage/dsa110-contimg/ms/2026-01-25T22:26:05_0~23.g"
SUSPECTS_FULL = [72, 81, 88, 90, 92, 93, 98]
SUSPECTS_PART = [47, 48, 70, 100]
REFANTS = [103, 104, 105]

out = {}

with table(MS + "/ANTENNA", readonly=True, ack=False) as t:
    names = list(t.getcol("NAME"))
    positions = np.array(t.getcol("POSITION"))
nant = len(names)
# CASA antenna *names* like 'pad103'/'103' vs indices: map name->index
name_to_idx = {n: i for i, n in enumerate(names)}
print("n antennas in MS:", nant)
print("first names:", names[:5], "... last:", names[-3:])

# --- 1. caltable per-antenna flag fraction ---
def caltable_flags(path):
    with table(path, readonly=True, ack=False) as t:
        ant = np.array(t.getcol("ANTENNA1"))
        fl = np.array(t.getcol("FLAG"))
    frac = {}
    for a in np.unique(ant):
        sel = ant == a
        frac[int(a)] = float(np.mean(fl[sel]))
    return frac

bfrac = caltable_flags(BTAB)
gfrac = caltable_flags(GTAB)
fully_flagged_b = sorted([a for a, f in bfrac.items() if f >= 0.999])
partial_b = sorted([a for a, f in bfrac.items() if 0.30 <= f < 0.999])
print("\nB table fully flagged antennas (index):", fully_flagged_b)
print("B table partially (>=30%) flagged:", [(a, round(bfrac[a], 2)) for a in partial_b])
out["b_fully_flagged"] = fully_flagged_b
out["b_partial"] = {a: bfrac[a] for a in partial_b}

# --- 2/3. MS flags + raw amplitude per antenna ---
with table(MS, readonly=True, ack=False) as t:
    nrow = t.nrows()
    a1 = np.array(t.getcol("ANTENNA1"))
    a2 = np.array(t.getcol("ANTENNA2"))
    uvw = np.array(t.getcol("UVW"))
    # Sample amplitude on a row subset for speed: every 8th row
    step = 8
    idx = np.arange(0, nrow, step)
    amp_sum = np.zeros(nant)
    amp_cnt = np.zeros(nant)
    fl_sum = 0.0
    fl_cnt = 0.0
    chunk = 200_000
    for start in range(0, len(idx), chunk):
        rows = idx[start:start + chunk]
        # contiguous getcol on the strided subset is awkward; read blockwise
        r0, r1 = rows[0], rows[-1] + 1
        data = t.getcol("DATA", startrow=r0, nrow=r1 - r0)
        flag = t.getcol("FLAG", startrow=r0, nrow=r1 - r0)
        sub = rows - r0
        d = np.abs(np.asarray(data)[sub])
        f = np.asarray(flag)[sub]
        fl_sum += f.sum(); fl_cnt += f.size
        m = np.where(f, np.nan, d)
        rowamp = np.nanmean(m, axis=tuple(range(1, m.ndim)))
        ra1 = a1[rows]; ra2 = a2[rows]
        ok = np.isfinite(rowamp)
        np.add.at(amp_sum, ra1[ok], rowamp[ok])
        np.add.at(amp_cnt, ra1[ok], 1)
        np.add.at(amp_sum, ra2[ok], rowamp[ok])
        np.add.at(amp_cnt, ra2[ok], 1)

ms_flag_frac = fl_sum / max(fl_cnt, 1)
print(f"\nMS FLAG fraction (sampled): {ms_flag_frac:.4f}")
out["ms_flag_fraction_sampled"] = ms_flag_frac

antamp = amp_sum / np.maximum(amp_cnt, 1)
med = np.median(antamp[antamp > 0])
rel = antamp / med
print("\nraw relative amplitude (|DATA|/median), suspects and refants:")
for a in SUSPECTS_FULL + SUSPECTS_PART + REFANTS:
    if a < nant:
        print(f"  ant {a} ({names[a]}): {rel[a]:.2f}  bflag={bfrac.get(a, float('nan')):.2f}")
out["relative_amplitude"] = {int(a): float(rel[a]) for a in range(nant)}

# --- 5. long-baseline counts (uvrange > 1 klambda at 1.405 GHz) ---
# freq: mid-band 1.405 GHz -> lambda 0.2134 m; 1 klambda = 213.4 m
lam = 0.2134
cut_m = 1000 * lam
# unique baselines from the row set
bl = {}
for i in range(0, nrow, 977):  # sample rows to enumerate baselines
    pass
# use antenna POSITIONS (ITRF) for baseline length instead — exact and cheap
from itertools import combinations
blen = np.zeros((nant, nant))
for i, j in combinations(range(nant), 2):
    d = np.linalg.norm(positions[i] - positions[j])
    blen[i, j] = blen[j, i] = d

# Antennas actually present in the data (any unflagged amp)
present = set(np.unique(np.concatenate([a1, a2])).tolist())
# healthy = present and raw amplitude > 0.3x median (excludes dead 18,108 etc.)
healthy = [a for a in range(nant) if a in present and rel[a] > 0.3]
print(f"\nantennas present in MS rows: {len(present)}; healthy (rel amp > 0.3): {len(healthy)}")

print("\nlong-baseline (>213.4 m) counts to healthy partners:")
report = {}
for a in sorted(set(SUSPECTS_FULL + SUSPECTS_PART + REFANTS + healthy)):
    if a >= nant or a not in present:
        continue
    nlong = sum(1 for b in healthy if b != a and blen[a, b] > cut_m)
    ntot = sum(1 for b in healthy if b != a)
    report[a] = (nlong, ntot)
sorted_by_nlong = sorted(report.items(), key=lambda kv: kv[1][0])
print("  lowest 20 by long-baseline count:")
for a, (nl, nt) in sorted_by_nlong[:20]:
    tag = "SUSPECT" if a in SUSPECTS_FULL + SUSPECTS_PART else ("REFANT" if a in REFANTS else "")
    print(f"   ant {a:4d} ({names[a]}): {nl:3d}/{nt} long baselines  bflag={bfrac.get(a, float('nan')):.2f} {tag}")
print("  suspects/refants explicitly:")
for a in SUSPECTS_FULL + SUSPECTS_PART + REFANTS:
    if a in report:
        nl, nt = report[a]
        print(f"   ant {a:4d} ({names[a]}): {nl:3d}/{nt} long baselines  bflag={bfrac.get(a, float('nan')):.2f}")
out["long_baseline_counts"] = {int(a): [int(v[0]), int(v[1])] for a, v in report.items()}

with open("/data/dsa110-continuum/outputs/cal-yield-2026-07-10/inspection.json", "w") as f:
    json.dump(out, f, indent=1)
print("\nsaved /data/dsa110-continuum/outputs/cal-yield-2026-07-10/inspection.json")
