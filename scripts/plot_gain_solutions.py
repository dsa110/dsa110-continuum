"""Step 3 diagnostic: per-antenna gain calibration solutions.

Workflow:
  1. Generate a clean calibrator UVH5 (no noise, 10 Jy at phase centre).
  2. Corrupt it with known per-antenna complex gains (5 % amp, 10° phase scatter).
  3. Run the Jacobi solver (same code path as SimulatedPipeline._calibrate).
  4. Compare recovered gains to injected truth.
  5. Plot per-antenna amplitude and phase: truth (circles) vs recovered (crosses).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dsa110_continuum.simulation.harness import SimulationHarness
from dsa110_continuum.simulation.gain_corruption import corrupt_uvh5
from dsa110_continuum.simulation.plot_style import apply_pipeline_style

OUT_DIR = Path("/home/user/workspace/dsa110-continuum/pipeline_outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Parameters ────────────────────────────────────────────────────────────────
N_ANT        = 96
N_INT        = 24
FLUX_JY      = 10.0
AMP_SCATTER  = 0.05      # 5 % fractional amplitude error
PHASE_SCATTER_DEG = 10.0 # 10° phase error
GAIN_SEED    = 17
CAL_SEED     = 42

# ── 1. Generate clean calibrator ──────────────────────────────────────────────
print("Generating clean calibrator UVH5…")
h = SimulationHarness(
    n_antennas=N_ANT,
    n_integrations=N_INT,
    noise_jy=0.0,
    seed=CAL_SEED,
    use_real_positions=True,
)
cal_clean = h.generate_calibrator_subband(
    output_dir=OUT_DIR / "step3",
    flux_jy=FLUX_JY,
    subband_index=0,
)
print(f"  Clean calibrator → {cal_clean}")

# ── 2. Corrupt with known gains ───────────────────────────────────────────────
print("Corrupting with known per-antenna gains…")
cal_corrupted = corrupt_uvh5(
    cal_clean,
    amp_scatter=AMP_SCATTER,
    phase_scatter_deg=PHASE_SCATTER_DEG,
    seed=GAIN_SEED,
    output_path=OUT_DIR / "step3" / "sim_cal_sb00_corrupted.uvh5",
)
print(f"  Corrupted calibrator → {cal_corrupted}")

# Reconstruct the true gains from the same RNG state
rng_truth = np.random.default_rng(GAIN_SEED)
import pyuvdata
uv_tmp = pyuvdata.UVData(); uv_tmp.read(str(cal_clean))
ant_nums_truth = np.unique(np.concatenate([uv_tmp.ant_1_array, uv_tmp.ant_2_array]))
n_ant_truth = len(ant_nums_truth)
amp_errors_truth  = 1.0 + rng_truth.normal(0.0, AMP_SCATTER, size=n_ant_truth)
phase_errors_truth = rng_truth.normal(0.0, np.radians(PHASE_SCATTER_DEG), size=n_ant_truth)
gains_truth_raw    = amp_errors_truth * np.exp(1j * phase_errors_truth)
# corrupt_uvh5 normalises to unit mean amplitude
gains_truth = gains_truth_raw / np.abs(gains_truth_raw).mean()

true_amp   = np.abs(gains_truth)          # (n_ant,)
true_phase = np.degrees(np.angle(gains_truth))  # (n_ant,)

# ── 3. Jacobi gain solve (inline, identical to pipeline._calibrate) ───────────
print("Running Jacobi gain solve…")
uv_cal = pyuvdata.UVData(); uv_cal.read(str(cal_corrupted))

ant_nums = np.unique(np.concatenate([uv_cal.ant_1_array, uv_cal.ant_2_array]))
n_ant    = len(ant_nums)
n_freq   = uv_cal.Nfreqs
ant_idx  = {int(a): i for i, a in enumerate(ant_nums)}
model_amp = FLUX_JY / 2.0

gains = np.ones((n_ant, n_freq), dtype=complex)
for _iter in range(2):
    numerator   = np.zeros_like(gains)
    denominator = np.zeros((n_ant, n_freq), dtype=float)
    for row in range(uv_cal.Nblts):
        i_ant = int(uv_cal.ant_1_array[row])
        j_ant = int(uv_cal.ant_2_array[row])
        if i_ant == j_ant:
            continue
        i = ant_idx[i_ant]
        j = ant_idx[j_ant]
        # Harness stores V_ij = G_i * conj(G_j) * model directly; no conjugation.
        vis = uv_cal.data_array[row, :, 0]
        numerator[i]   += vis * gains[j] / model_amp
        denominator[i] += np.abs(gains[j]) ** 2
        numerator[j]   += np.conj(vis) * gains[i] / model_amp
        denominator[j] += np.abs(gains[i]) ** 2
    gains = numerator / np.maximum(denominator, 1e-12)

# Channel-average the solutions for display (Jacobi gives per-channel gains)
rec_amp   = np.abs(gains).mean(axis=1)       # (n_ant,)
rec_phase = np.degrees(np.angle(gains.mean(axis=1)))  # (n_ant,) -- avg over freq

ant_indices = np.arange(n_ant)

# ── 4. Residuals ──────────────────────────────────────────────────────────────
amp_residual   = rec_amp   - true_amp
phase_residual = rec_phase - true_phase
# Wrap phase residual to [-180, 180]
phase_residual = (phase_residual + 180) % 360 - 180

print(f"\n=== Gain recovery summary ({n_ant} antennas) ===")
print(f"  Amp   residual  rms: {amp_residual.std():.4f}  (mean: {amp_residual.mean():+.4f})")
print(f"  Phase residual  rms: {phase_residual.std():.2f}°  (mean: {phase_residual.mean():+.2f}°)")
print(f"  True amp range:      {true_amp.min():.4f} – {true_amp.max():.4f}")
print(f"  Rec  amp range:      {rec_amp.min():.4f} – {rec_amp.max():.4f}")

# ── 5. Plot ───────────────────────────────────────────────────────────────────
apply_pipeline_style()

fig, axes = plt.subplots(2, 2, figsize=(14, 8))

# ── A: Amplitude per antenna ──────────────────────────────────────────────────
ax = axes[0, 0]
ax.plot(ant_indices, true_amp,  "o", ms=4, color="C0", label="Truth",     zorder=3)
ax.plot(ant_indices, rec_amp,   "x", ms=5, color="C1", label="Recovered", zorder=4, lw=1.5)
ax.axhline(1.0, color="0.5", lw=0.8, linestyle="--", label="Unity gain")
ax.set_xlabel("Antenna index")
ax.set_ylabel("|G| (amplitude)")
ax.set_title("Per-antenna amplitude gain")
ax.legend(fontsize=9)

# ── B: Phase per antenna ──────────────────────────────────────────────────────
ax = axes[0, 1]
ax.plot(ant_indices, true_phase,  "o", ms=4, color="C0", label="Truth")
ax.plot(ant_indices, rec_phase,   "x", ms=5, color="C1", label="Recovered", lw=1.5)
ax.axhline(0.0, color="0.5", lw=0.8, linestyle="--", label="Zero phase")
ax.set_xlabel("Antenna index")
ax.set_ylabel("Phase (degrees)")
ax.set_title("Per-antenna phase gain")
ax.legend(fontsize=9)

# ── C: Amplitude residual ─────────────────────────────────────────────────────
ax = axes[1, 0]
mean_amp_res = amp_residual.mean()
rms_amp = amp_residual.std()   # scatter around mean (excl. global offset)
ax.bar(ant_indices, amp_residual, color="C2", alpha=0.7, width=0.8)
ax.axhline(0.0,            color="k",   lw=0.8)
ax.axhline(mean_amp_res,   color="blue",lw=1.0, linestyle="-",
           label=f"Mean offset = {mean_amp_res:+.4f}\n(global scale degeneracy)")
ax.axhline(mean_amp_res + rms_amp, color="red", lw=0.8, linestyle="--",
           label=f"Mean ± scatter rms={rms_amp:.4f}")
ax.axhline(mean_amp_res - rms_amp, color="red", lw=0.8, linestyle="--")
ax.set_xlabel("Antenna index")
ax.set_ylabel("Residual |G|")
ax.set_title("Amplitude residual (recovered − truth)")
ax.legend(fontsize=8)

# ── D: Phase residual ─────────────────────────────────────────────────────────
ax = axes[1, 1]
# Remove global phase offset (reference antenna degeneracy) before computing rms
mean_ph_res  = phase_residual.mean()
ph_res_deref = phase_residual - mean_ph_res   # demeaned
rms_ph       = ph_res_deref.std()
rms_ph_raw   = phase_residual.std()           # rms including global offset
ax.bar(ant_indices, phase_residual, color="C3", alpha=0.7, width=0.8)
ax.axhline(0.0,           color="k",   lw=0.8)
ax.axhline(mean_ph_res,   color="blue",lw=1.0, linestyle="-",
           label=f"Mean offset = {mean_ph_res:+.2f}°\n(ref-antenna degeneracy)")
ax.axhline(mean_ph_res + rms_ph, color="red", lw=0.8, linestyle="--",
           label=f"Mean ± scatter rms={rms_ph:.2f}°")
ax.axhline(mean_ph_res - rms_ph, color="red", lw=0.8, linestyle="--")
ax.set_xlabel("Antenna index")
ax.set_ylabel("Residual phase (°)")
ax.set_title("Phase residual (recovered − truth)")
ax.legend(fontsize=8)

fig.suptitle(
    f"Jacobi gain calibration  |  {N_ANT} antennas · "
    f"amp scatter {AMP_SCATTER*100:.0f}% · phase scatter {PHASE_SCATTER_DEG:.0f}°\n"
    f"Amp: mean offset {mean_amp_res:+.4f} (global scale), scatter rms={rms_amp:.4f}  |  "
    f"Phase: mean offset {mean_ph_res:+.2f}° (ref-ant), scatter rms={rms_ph:.2f}°",
    fontsize=10,
)
fig.tight_layout()

out_fig = OUT_DIR / "step3_gain_solutions.png"
fig.savefig(out_fig, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\n  Saved figure → {out_fig}")
