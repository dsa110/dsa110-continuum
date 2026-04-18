"""
Regenerate the DSA-110 PSF analysis figure from saved numpy arrays.
Uses SciencePlots science + notebook style.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Ellipse
import scienceplots  # noqa

# ── load saved data ──────────────────────────────────────────────────────────
uv_u    = np.load('/tmp/uv_u.npy')          # metres
uv_v    = np.load('/tmp/uv_v.npy')
psf     = np.load('/tmp/psf.npy')           # 2048×2048
pix_arcsec = float(np.load('/tmp/pix_arcsec.npy'))
uv_grid = np.load('/tmp/uv_grid.npy')       # gridded uv plane (weights)

FREQ_HZ = 1405e6
LAMBDA  = 3e8 / FREQ_HZ   # ~0.2135 m

# convert UV to kλ
uv_u_kl = uv_u / LAMBDA / 1e3
uv_v_kl = uv_v / LAMBDA / 1e3

# ── PSF metrics ──────────────────────────────────────────────────────────────
N = psf.shape[0]
cx = N // 2

# HPBW along central rows/cols (EW = u-direction = axis-1, NS = v-direction = axis-0)
def hpbw_pixels(profile):
    peak = profile.max()
    half = peak / 2.0
    above = np.where(profile >= half)[0]
    return (above[-1] - above[0] + 1) if len(above) else 0

ew_profile = psf[cx, :]
ns_profile = psf[:, cx]

hpbw_ew_px = hpbw_pixels(ew_profile)
hpbw_ns_px = hpbw_pixels(ns_profile)
hpbw_ew_arcsec = hpbw_ew_px * pix_arcsec
hpbw_ns_arcsec = hpbw_ns_px * pix_arcsec

# Peak sidelobe (exclude 3×HPBW central zone)
mask_half = max(hpbw_ew_px, hpbw_ns_px)
excl = max(mask_half * 3, 30)
psf_sidelobe = psf.copy()
psf_sidelobe[cx - excl:cx + excl, cx - excl:cx + excl] = 0
peak_sidelobe = psf_sidelobe.max()

# UV fill fraction
uv_fill = (uv_grid > 0).sum() / uv_grid.size

print(f"HPBW EW: {hpbw_ew_arcsec:.1f} arcsec")
print(f"HPBW NS: {hpbw_ns_arcsec:.1f} arcsec")
print(f"Peak sidelobe: {peak_sidelobe:.3f}")
print(f"UV fill: {uv_fill*100:.2f}%")
print(f"UV extent: u=±{np.abs(uv_u_kl).max():.1f} kλ  v=±{np.abs(uv_v_kl).max():.1f} kλ")

# ── figure ───────────────────────────────────────────────────────────────────
with plt.style.context(['science', 'notebook']):
    fig = plt.figure(figsize=(14, 10))
    fig.suptitle('DSA-110 Synthesized Beam Analysis\n117 Antennas · 1405 MHz · Dec = +16°',
                 fontsize=13, y=0.98)

    gs = gridspec.GridSpec(2, 3, figure=fig,
                           hspace=0.42, wspace=0.45,
                           left=0.07, right=0.97, top=0.91, bottom=0.09)

    # ── panel 1: UV coverage ─────────────────────────────────────────────
    ax_uv = fig.add_subplot(gs[0, 0])
    ax_uv.scatter(uv_u_kl,  uv_v_kl,  s=0.3, alpha=0.15, color='C0', rasterized=True)
    ax_uv.scatter(-uv_u_kl, -uv_v_kl, s=0.3, alpha=0.15, color='C0', rasterized=True)
    ax_uv.set_xlabel('u  [kλ]')
    ax_uv.set_ylabel('v  [kλ]')
    ax_uv.set_title('UV Coverage')
    ax_uv.set_aspect('equal')
    ax_uv.text(0.03, 0.97, f'Fill: {uv_fill*100:.1f}%',
               transform=ax_uv.transAxes, ha='left', va='top', fontsize=8,
               bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, pad=2))

    # ── panel 2: PSF image (wide, full beam+sidelobes) ───────────────────
    ax_psf = fig.add_subplot(gs[0, 1])
    zoom = 300   # pixels each side of centre — ~53 arcmin field
    sub = psf[cx - zoom:cx + zoom, cx - zoom:cx + zoom]
    ext_arcmin = zoom * pix_arcsec / 60.0
    im = ax_psf.imshow(sub, origin='lower', cmap='RdBu_r',
                       vmin=-0.3, vmax=1.0,
                       extent=[-ext_arcmin, ext_arcmin, -ext_arcmin, ext_arcmin])
    plt.colorbar(im, ax=ax_psf, label='Normalized response', pad=0.02)
    ax_psf.set_xlabel('ΔRA  [arcmin]')
    ax_psf.set_ylabel('ΔDec  [arcmin]')
    ax_psf.set_title('PSF (wide view)')
    # HPBW ellipse
    ell = Ellipse((0, 0),
                  width=hpbw_ew_arcsec / 60.0,
                  height=hpbw_ns_arcsec / 60.0,
                  edgecolor='yellow', facecolor='none', lw=1.5, linestyle='--',
                  label=f'HPBW {hpbw_ew_arcsec:.0f}"×{hpbw_ns_arcsec:.0f}"')
    ax_psf.add_patch(ell)
    leg = ax_psf.legend(loc='upper right', fontsize=7)
    leg.get_frame().set_facecolor('0.15')
    leg.get_frame().set_edgecolor('none')
    for text in leg.get_texts():
        text.set_color('white')

    # ── panel 3: PSF image (zoom on main beam) ───────────────────────────
    ax_zoom = fig.add_subplot(gs[0, 2])
    z2 = max(hpbw_ew_px, hpbw_ns_px) * 4
    sub2 = psf[cx - z2:cx + z2, cx - z2:cx + z2]
    ext2 = z2 * pix_arcsec / 60.0
    im2 = ax_zoom.imshow(sub2, origin='lower', cmap='plasma',
                         vmin=0, vmax=1.0,
                         extent=[-ext2, ext2, -ext2, ext2])
    plt.colorbar(im2, ax=ax_zoom, label='Normalized response', pad=0.02)
    ax_zoom.set_xlabel('ΔRA  [arcmin]')
    ax_zoom.set_ylabel('ΔDec  [arcmin]')
    ax_zoom.set_title('Main Beam (zoom)')
    ell2 = Ellipse((0, 0),
                   width=hpbw_ew_arcsec / 60.0,
                   height=hpbw_ns_arcsec / 60.0,
                   edgecolor='cyan', facecolor='none', lw=1.5, linestyle='--')
    ax_zoom.add_patch(ell2)

    # ── panel 4: EW slice ────────────────────────────────────────────────
    ax_ew = fig.add_subplot(gs[1, 0])
    ew_x = (np.arange(N) - cx) * pix_arcsec / 60.0
    ax_ew.plot(ew_x, ew_profile, lw=1.0)
    ax_ew.axhline(0.5, color='gray', lw=0.8, linestyle=':')
    ax_ew.axhline(0,   color='gray', lw=0.5, linestyle='-')
    ax_ew.set_xlim(-ext_arcmin, ext_arcmin)
    ax_ew.set_xlabel('ΔRA  [arcmin]')
    ax_ew.set_ylabel('Normalized response')
    ax_ew.set_title(f'E–W Slice  (HPBW = {hpbw_ew_arcsec:.0f}")')

    # ── panel 5: NS slice ────────────────────────────────────────────────
    ax_ns = fig.add_subplot(gs[1, 1])
    ns_x = (np.arange(N) - cx) * pix_arcsec / 60.0
    ax_ns.plot(ns_x, ns_profile, lw=1.0)
    ax_ns.axhline(0.5, color='gray', lw=0.8, linestyle=':')
    ax_ns.axhline(0,   color='gray', lw=0.5, linestyle='-')
    ax_ns.set_xlim(-ext_arcmin, ext_arcmin)
    ax_ns.set_xlabel('ΔDec  [arcmin]')
    ax_ns.set_ylabel('Normalized response')
    ax_ns.set_title(f'N–S Slice  (HPBW = {hpbw_ns_arcsec:.0f}")')

    # ── panel 6: metrics summary ─────────────────────────────────────────
    ax_txt = fig.add_subplot(gs[1, 2])
    ax_txt.axis('off')

    uv_u_max = np.abs(uv_u_kl).max()
    uv_v_max = np.abs(uv_v_kl).max()

    lines = [
        ('Array geometry', ''),
        ('  Total stations', '117'),
        ('  Active stations', '96'),
        ('  T-core EW span', '~397 m'),
        ('  T-core NS span', '~432 m'),
        ('  Max EW baseline', '~1769 m (outriggers)'),
        ('  Max NS baseline', '~2220 m (outriggers)'),
        ('', ''),
        ('Beam parameters', ''),
        ('  T-core HPBW (EW)', f'{397/LAMBDA/1e3*2.0:.0f}" (est. {hpbw_ew_arcsec:.0f}" measured)'),
        ('  Full-array HPBW (EW)', f'{hpbw_ew_arcsec:.0f}"'),
        ('  Full-array HPBW (NS)', f'{hpbw_ns_arcsec:.0f}"'),
        ('  Peak sidelobe', f'{peak_sidelobe:.3f}'),
        ('', ''),
        ('UV coverage', ''),
        ('  u range', f'±{uv_u_max:.1f} kλ'),
        ('  v range', f'±{uv_v_max:.1f} kλ'),
        ('  Fill fraction', f'{uv_fill*100:.2f}%'),
        ('  Frequency', '1405 MHz'),
        ('  Declination', '+16.15°'),
    ]

    y = 0.97
    dy = 0.048
    for label, val in lines:
        if label == '' and val == '':
            y -= dy * 0.3
            continue
        if val == '':
            ax_txt.text(0.01, y, label, transform=ax_txt.transAxes,
                        fontsize=8, fontweight='bold', va='top')
        else:
            ax_txt.text(0.01, y, label, transform=ax_txt.transAxes,
                        fontsize=7.5, va='top')
            ax_txt.text(0.62, y, val, transform=ax_txt.transAxes,
                        fontsize=7.5, va='top')
        y -= dy

    ax_txt.set_title('Summary', fontsize=10)
    # add a subtle bounding box
    for spine in ax_txt.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.5)
        spine.set_color('gray')

    out = '/home/user/workspace/pipeline_outputs/dsa110_psf_analysis.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved → {out}")
