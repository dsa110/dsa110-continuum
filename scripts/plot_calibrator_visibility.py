"""Step 2 diagnostic: calibrator visibility plots.

Generates a calibrator UVH5 (single bright point source at phase centre,
no noise) and produces:
  1. Amplitude vs UV-distance (baseline length in wavelengths)
  2. UV-coverage (u vs v) coloured by amplitude
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Repo on path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dsa110_continuum.simulation.harness import SimulationHarness
from dsa110_continuum.simulation.plot_style import apply_pipeline_style

OUT_DIR = Path("/home/user/workspace/dsa110-continuum/pipeline_outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── 1. Generate calibrator UVH5 ───────────────────────────────────────────────
print("Generating calibrator UVH5 (96 antennas, 24 integrations, 10 Jy, no noise)…")

h = SimulationHarness(
    n_antennas=96,
    n_integrations=24,
    noise_jy=0.0,
    seed=42,
    use_real_positions=True,
)

cal_path = h.generate_calibrator_subband(
    output_dir=OUT_DIR,
    flux_jy=10.0,
    subband_index=0,
)
print(f"  Written → {cal_path}")

# ── 2. Load ───────────────────────────────────────────────────────────────────
uv = h.load_subband(cal_path)

# Cross-correlations only (exclude auto-correlations)
cross_mask = uv.ant_1_array != uv.ant_2_array
data   = uv.data_array[cross_mask]          # (n_cross_blts, n_freq, n_pol)
uvw    = uv.uvw_array[cross_mask]           # (n_cross_blts, 3)

# Central frequency for λ conversion
freq_hz     = float(uv.freq_array.mean())
lambda_m    = 3e8 / freq_hz

# UV-distance in wavelengths (channel-averaged amplitude for each blt)
uv_dist_lam = np.sqrt(uvw[:, 0]**2 + uvw[:, 1]**2) / lambda_m
amp_xx      = np.abs(data[:, :, 0]).mean(axis=1)   # XX pol, avg over freq
amp_yy      = np.abs(data[:, :, 1]).mean(axis=1)   # YY pol

u_lam = uvw[:, 0] / lambda_m
v_lam = uvw[:, 1] / lambda_m

# ── 3. Plot ───────────────────────────────────────────────────────────────────
apply_pipeline_style()

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# ── Panel A: Amplitude vs UV-distance ────────────────────────────────────────
ax = axes[0]
ax.scatter(uv_dist_lam, amp_xx, s=1, alpha=0.3, color="C0", label="XX", rasterized=True)
ax.scatter(uv_dist_lam, amp_yy, s=1, alpha=0.3, color="C1", label="YY", rasterized=True)

# Expected value: 10 Jy / 2 = 5 Jy (XX = YY = I/2 convention)
expected = 10.0 / 2.0
ax.axhline(expected, color="red", linewidth=1.0, linestyle="--",
           label=f"Expected {expected:.0f} Jy")

ax.set_xlabel(r"UV-distance ($\lambda$)")
ax.set_ylabel("Visibility amplitude (Jy)")
ax.set_title("Calibrator: amplitude vs baseline length")
ax.legend(markerscale=8, fontsize=9)

# ── Panel B: UV-coverage ─────────────────────────────────────────────────────
ax = axes[1]
sc = ax.scatter(u_lam, v_lam, c=amp_xx, s=1, alpha=0.5,
                cmap="viridis", vmin=0, rasterized=True)
# Mirror conjugate baselines
ax.scatter(-u_lam, -v_lam, c=amp_xx, s=1, alpha=0.5,
           cmap="viridis", vmin=0, rasterized=True)

cbar = fig.colorbar(sc, ax=ax, pad=0.02)
cbar.set_label("XX amplitude (Jy)")
ax.set_xlabel(r"$u$ ($\lambda$)")
ax.set_ylabel(r"$v$ ($\lambda$)")
ax.set_aspect("equal")
ax.set_title("UV-coverage (calibrator, all times)")

fig.suptitle(
    f"DSA-110 calibrator observation  |  96 antennas · 24 integrations · SB0 "
    f"({freq_hz/1e6:.1f} MHz)",
    fontsize=11,
)
fig.tight_layout()

out_fig = OUT_DIR / "step2_calibrator_visibility.png"
fig.savefig(out_fig, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved figure → {out_fig}")

# ── 4. Quick sanity report ────────────────────────────────────────────────────
print("\n=== Calibrator visibility sanity ===")
print(f"  Baselines:          {int(cross_mask.sum()) // uv.Ntimes}")
print(f"  Time integrations:  {uv.Ntimes}")
print(f"  Freq channels:      {uv.Nfreqs}")
print(f"  Central freq:       {freq_hz/1e6:.2f} MHz")
print(f"  UV-dist range:      {uv_dist_lam.min():.0f} – {uv_dist_lam.max():.0f}  λ")
print(f"  XX amp mean ± std:  {amp_xx.mean():.4f} ± {amp_xx.std():.4f} Jy")
print(f"  YY amp mean ± std:  {amp_yy.mean():.4f} ± {amp_yy.std():.4f} Jy")
print(f"  Expected:           {expected:.4f} Jy")
print(f"  Max deviation:      {abs(np.concatenate([amp_xx, amp_yy]) - expected).max()*1000:.2f} mJy")
