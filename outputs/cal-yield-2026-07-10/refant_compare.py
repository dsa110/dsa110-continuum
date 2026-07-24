"""Compare per-antenna-pol bandpass flags across refant 103/104/105 solves."""
import glob
import json
import numpy as np
import sys

sys.path.insert(0, "/data/dsa110-continuum")
from dsa110_continuum.adapters.casa_tables import table

EXP = "/data/dsa110-continuum/outputs/cal-yield-2026-07-10"
SUSPECTS = [72, 81, 88, 90, 92, 93, 96, 98, 47, 48, 70, 100]
PROD_BTAB = "/stage/dsa110-contimg/ms/2026-01-25T22:26:05_0~23.b"


def perpol_flags(btab):
    with table(btab, readonly=True, ack=False) as t:
        ant = np.array(t.getcol("ANTENNA1"))
        fl = np.array(t.getcol("FLAG"))  # (rows, 2 pol, 48 chan)
        snr = np.array(t.getcol("SNR"))
    out = {}
    for a in np.unique(ant):
        sel = ant == a
        f = fl[sel]
        s = snr[sel]
        pol = [float(f[:, p, :].mean()) for p in range(f.shape[1])]
        med_snr = float(np.median(s[~f])) if (~f).any() else float("nan")
        out[int(a)] = {"pol": pol, "overall": float(f.mean()), "med_snr": med_snr}
    return out


results = {"production_refant103": perpol_flags(PROD_BTAB)}
for r in (103, 104, 105):
    hits = sorted(glob.glob(f"{EXP}/refant_{r}/solve*.b"))
    if not hits:
        print(f"refant {r}: no .b table found ({EXP}/refant_{r}/solve*.b)")
        continue
    results[f"refant{r}"] = perpol_flags(hits[0])
    results[f"refant{r}_table"] = hits[0]

keys = [k for k in results if k.startswith("refant") and not k.endswith("_table")]
print(f"{'ant':>5} | " + " | ".join(f"{k:^22}" for k in ["prod(103)"] + keys))
all_ants = sorted(results["production_refant103"].keys())
interesting = []
for a in all_ants:
    row = []
    changed = False
    prod = results["production_refant103"].get(a)
    for k in ["production_refant103"] + keys:
        e = results[k].get(a)
        if e is None:
            row.append("absent")
            continue
        row.append(f"{e['pol'][0]:.2f}/{e['pol'][1]:.2f} snr{e['med_snr']:.0f}")
        if k != "production_refant103" and prod and abs(e["overall"] - prod["overall"]) > 0.2:
            changed = True
    if a in SUSPECTS or changed:
        interesting.append((a, row, changed))

for a, row, changed in interesting:
    tag = "*" if a in SUSPECTS else " "
    chg = "CHANGED" if changed else ""
    print(f"{tag}{a:>4} | " + " | ".join(f"{c:^22}" for c in row) + f" {chg}")

# summary: total yield per solve
print("\nyield summary (mean unflagged fraction over all solutions):")
for k in ["production_refant103"] + keys:
    vals = [1 - e["overall"] for e in results[k].values()]
    nfull = sum(1 for e in results[k].values() if e["overall"] >= 0.999)
    print(f"  {k}: mean unflagged {np.mean(vals):.3f}; fully-flagged antennas {nfull}")

with open(f"{EXP}/refant_comparison.json", "w") as f:
    json.dump(results, f, indent=1)
print(f"saved {EXP}/refant_comparison.json")
