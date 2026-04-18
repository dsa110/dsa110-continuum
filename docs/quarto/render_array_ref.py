"""
Hand-render array-reference.qmd → _site/array-reference.html
by cloning the boilerplate from lightcurves-and-variability.html
and substituting the content + sidebar entry.
"""
from pathlib import Path
import re, shutil

SITE = Path("_site")
SRC  = SITE / "lightcurves-and-variability.html"
DST  = SITE / "array-reference.html"

# Copy figures into _site/images/
images_dst = SITE / "images"
images_dst.mkdir(exist_ok=True)
for fn in ["dsa110_antenna_layout.png", "dsa110_psf_analysis.png"]:
    src_img = Path("images") / fn
    if src_img.exists():
        shutil.copy(src_img, images_dst / fn)
        print(f"  copied {fn}")

html = SRC.read_text()

# ── 1. title tag ─────────────────────────────────────────────────────────────
html = html.replace(
    "<title>Light Curves and Variability – DSA-110 Continuum Pipeline</title>",
    "<title>Array Reference – DSA-110 Continuum Pipeline</title>",
)

# ── 2. breadcrumb ────────────────────────────────────────────────────────────
html = html.replace(
    '<ol class="breadcrumb"><li class="breadcrumb-item">'
    '<a href="./lightcurves-and-variability.html">Light Curves and Variability</a></li></ol>',
    '<ol class="breadcrumb"><li class="breadcrumb-item">'
    '<a href="./array-reference.html">Array Reference</a></li></ol>',
)

# ── 3. sidebar — add Array Reference entry after Pipeline Overview,
#                update active class ─────────────────────────────────────────
# Remove active from lightcurves sidebar item
html = html.replace(
    '<a href="./lightcurves-and-variability.html" class="sidebar-item-text sidebar-link active">',
    '<a href="./lightcurves-and-variability.html" class="sidebar-item-text sidebar-link">',
)
# Remove active from pipeline-overview (it might already be inactive, just be safe)
# Insert new sidebar item after pipeline-overview item, mark it active
NEW_SIDEBAR_ITEM = """        <li class="sidebar-item">
  <div class="sidebar-item-container"> 
  <a href="./pipeline-overview.html" class="sidebar-item-text sidebar-link">
 <span class="menu-text">Pipeline Overview</span></a>
  </div>
</li>
        <li class="sidebar-item">
  <div class="sidebar-item-container"> 
  <a href="./array-reference.html" class="sidebar-item-text sidebar-link active">
 <span class="menu-text">Array Reference</span></a>
  </div>
</li>"""

OLD_PIPELINE_ITEM = """        <li class="sidebar-item">
  <div class="sidebar-item-container"> 
  <a href="./pipeline-overview.html" class="sidebar-item-text sidebar-link">
 <span class="menu-text">Pipeline Overview</span></a>
  </div>
</li>"""

html = html.replace(OLD_PIPELINE_ITEM, NEW_SIDEBAR_ITEM, 1)

# ── 4. TOC (right-hand margin sidebar) ───────────────────────────────────────
OLD_TOC = html[html.find('<nav id="TOC"'):html.find('</nav>', html.find('<nav id="TOC"')) + 6]
NEW_TOC = """<nav id="TOC" role="doc-toc" class="toc-active">
    <h2 id="toc-title">On this page</h2>
   
  <ul>
  <li><a href="#array-geometry" id="toc-array-geometry" class="nav-link active" data-scroll-target="#array-geometry">Array geometry</a>
  <ul class="collapse">
  <li><a href="#station-layout" id="toc-station-layout" class="nav-link" data-scroll-target="#station-layout">Station layout</a></li>
  <li><a href="#active-antenna-count" id="toc-active-antenna-count" class="nav-link" data-scroll-target="#active-antenna-count">Active antenna count</a></li>
  <li><a href="#key-baseline-lengths" id="toc-key-baseline-lengths" class="nav-link" data-scroll-target="#key-baseline-lengths">Key baseline lengths</a></li>
  </ul></li>
  <li><a href="#synthesized-beam" id="toc-synthesized-beam" class="nav-link" data-scroll-target="#synthesized-beam">Synthesized beam</a>
  <ul class="collapse">
  <li><a href="#beam-parameters" id="toc-beam-parameters" class="nav-link" data-scroll-target="#beam-parameters">Beam parameters</a></li>
  <li><a href="#implications-for-imaging" id="toc-implications-for-imaging" class="nav-link" data-scroll-target="#implications-for-imaging">Implications for imaging</a></li>
  </ul></li>
  <li><a href="#simulation-parameters" id="toc-simulation-parameters" class="nav-link" data-scroll-target="#simulation-parameters">Simulation parameters</a>
  <ul class="collapse">
  <li><a href="#timing" id="toc-timing" class="nav-link" data-scroll-target="#timing">Timing</a></li>
  <li><a href="#spectral-setup" id="toc-spectral-setup" class="nav-link" data-scroll-target="#spectral-setup">Spectral setup</a></li>
  </ul></li>
  <li><a href="#regenerating-figures" id="toc-regenerating-figures" class="nav-link" data-scroll-target="#regenerating-figures">Regenerating figures</a></li>
  </ul>
</nav>"""

html = html.replace(OLD_TOC, NEW_TOC)

# ── 5. main content ───────────────────────────────────────────────────────────
MAIN_START = '<main class="content" id="quarto-document-content">'
MAIN_END   = '</main> <!-- /main -->'

new_main = '''<main class="content" id="quarto-document-content">

<header id="title-block-header" class="quarto-title-block default">
<div class="quarto-title">
<h1 class="title">Array Reference</h1>
</div>
<div class="quarto-title-meta"></div>
</header>

<blockquote class="blockquote">
<p><strong>Status</strong>: Verified against repo commit <code>6ad8a00</code> (2026-04-17)<br>
<strong>Ground truth file</strong>: <a href="https://github.com/dsa110/dsa110-continuum/blob/main/docs/GROUND_TRUTH.md"><code>docs/GROUND_TRUTH.md</code></a> — read this before writing simulation or pipeline code<br>
<strong>See also</strong>: <a href="./pipeline-overview.html">Pipeline Overview</a></p>
</blockquote>

<p>This page documents the DSA-110 array geometry and synthesized beam properties as
they are actually encoded in the simulation pipeline. All parameters are sourced
directly from the codebase — not from memory or published instrument descriptions.</p>

<hr>

<section id="array-geometry" class="level2">
<h2 class="anchored" data-anchor-id="array-geometry">Array geometry</h2>
<p>DSA-110 is a <strong>T-shaped transit array</strong> at OVRO (37.2339° N, −118.2825° E, 1222 m).
It consists of a dense E-W core arm, a N-S arm forming the T-junction, and a set
of more sparsely distributed outrigger antennas at larger baselines.</p>

<div class="callout callout-style-default callout-important callout-titled">
<div class="callout-header d-flex align-content-center">
<div class="callout-icon-container"><i class="callout-icon"></i></div>
<div class="callout-title-container flex-fill">Simulation harness configuration</div>
</div>
<div class="callout-body-container callout-body">
<p>Always use <code>n_antennas=117</code> when constructing a <code>SimulationHarness</code>. The default
of <code>n_antennas=8</code> is for unit tests only. Using <code>n_antennas=96</code> takes the first
96 rows of <code>antennas.csv</code> (51 EW + 45 NS + <strong>0 outriggers</strong>) — geometrically
incorrect and the cause of image artifacts in earlier pipeline runs.</p>
<div class="sourceCode"><pre class="sourceCode python code-with-copy"><code class="sourceCode python">harness <span class="op">=</span> SimulationHarness(n_antennas<span class="op">=</span><span class="dv">117</span>, n_integrations<span class="op">=</span><span class="dv">24</span>)</code><button title="Copy to Clipboard" class="code-copy-button"><i class="bi"></i></button></pre></div>
</div>
</div>

<section id="station-layout" class="level3">
<h3 class="anchored" data-anchor-id="station-layout">Station layout</h3>
<p>The position file used by the simulation is
<code>dsa110_continuum/simulation/pyuvsim/antennas.csv</code> (117 rows, projected ECEF
columns <code>east_m</code>, <code>north_m</code>, <code>up_m</code>). The figure below shows the resulting
local-ENU positions after the geodetic → ECEF → local-ENU transform applied by
<code>load_geodetic_enu()</code>.</p>
<div class="quarto-figure quarto-figure-center">
<figure class="figure">
<p><img src="images/dsa110_antenna_layout.png" class="img-fluid figure-img" style="width:90%"></p>
<figcaption>DSA-110 ENU layout — 117 stations. T-core (EW + NS arms) shown in blue;
outriggers (DSA103–117) shown in orange. Baselines from the T-junction extend to
≈1.8 km E-W and ≈2.2 km N-S with the outriggers included.</figcaption>
</figure>
</div>
</section>

<section id="active-antenna-count" class="level3">
<h3 class="anchored" data-anchor-id="active-antenna-count">Active antenna count</h3>
<p><strong>96 of the 117 allocated station slots are active</strong> (Connor et al. 2025,
arXiv:2510.18136):</p>
<table class="caption-top table">
<thead>
<tr><th>Arm</th><th>Active</th><th>Slot pool</th><th>Notes</th></tr>
</thead>
<tbody>
<tr><td>E-W core</td><td>51</td><td>DSA001–DSA051</td><td>All 51 slots built and active</td></tr>
<tr><td>N-S arm</td><td>35</td><td>DSA052–DSA102</td><td>51 slots allocated; 35 active</td></tr>
<tr><td>Outriggers</td><td>14</td><td>DSA103–DSA117</td><td>15 slots; DSA-117 likely inactive (no elevation in CSV)</td></tr>
<tr><td><strong>Total</strong></td><td><strong>96</strong></td><td>—</td><td>—</td></tr>
</tbody>
</table>
<div class="callout callout-style-default callout-note callout-titled">
<div class="callout-header d-flex align-content-center">
<div class="callout-icon-container"><i class="callout-icon"></i></div>
<div class="callout-title-container flex-fill">Note</div>
</div>
<div class="callout-body-container callout-body">
<p>The exact list of which 96 station numbers are active is <strong>not machine-readable
in this repository</strong>. The authoritative source is the HDF5/MS antenna metadata
from real data on <code>h17</code>, accessed via <code>pyuvdata</code>'s <code>telescope.antenna_numbers</code>
field. For simulation purposes, using all 117 positions is correct — the 21
inactive slots produce zero-weighted baselines that do not degrade image quality.</p>
</div>
</div>
</section>

<section id="key-baseline-lengths" class="level3">
<h3 class="anchored" data-anchor-id="key-baseline-lengths">Key baseline lengths</h3>
<table class="caption-top table">
<thead><tr><th>Dimension</th><th>Extent</th></tr></thead>
<tbody>
<tr><td>E-W arm span (core)</td><td>~397 m (DSA-001 to DSA-051)</td></tr>
<tr><td>N-S arm span (core)</td><td>~432 m (DSA-052 to DSA-102)</td></tr>
<tr><td>Max E-W baseline (with outriggers)</td><td>~1769 m</td></tr>
<tr><td>Max N-S baseline (with outriggers)</td><td>~2220 m</td></tr>
</tbody>
</table>
</section>
</section>

<hr>

<section id="synthesized-beam" class="level2">
<h2 class="anchored" data-anchor-id="synthesized-beam">Synthesized beam</h2>
<p>The figure below shows the full synthesized beam analysis computed for the
117-antenna array observing at Dec +16.15° (OVRO transit strip) at 1405 MHz
(band centre). All 117 station positions from <code>antennas.csv</code> are used.</p>
<div class="quarto-figure quarto-figure-center">
<figure class="figure">
<p><img src="images/dsa110_psf_analysis.png" class="img-fluid figure-img" style="width:100%"></p>
<figcaption>DSA-110 PSF analysis — 117 antennas, 1405 MHz, Dec +16.15°. Top row: UV
coverage (fill 0.90%), PSF wide view with HPBW ellipse, and main-beam zoom.
Bottom row: E-W and N-S slices through the beam centre, and parameter summary.</figcaption>
</figure>
</div>

<section id="beam-parameters" class="level3">
<h3 class="anchored" data-anchor-id="beam-parameters">Beam parameters</h3>
<table class="caption-top table">
<thead><tr><th>Parameter</th><th>Value</th><th>Notes</th></tr></thead>
<tbody>
<tr><td>HPBW E-W</td><td><strong>75"</strong></td><td>Set by ~1769 m outrigger baselines</td></tr>
<tr><td>HPBW N-S</td><td><strong>332"</strong></td><td>Set by ~2220 m outrigger baselines; 4.4× elongated</td></tr>
<tr><td>Beam axis ratio (N-S / E-W)</td><td>4.4</td><td>Reflects asymmetric UV coverage</td></tr>
<tr><td>Peak sidelobe level</td><td><strong>0.415</strong></td><td>High — consequence of 0.90% UV fill</td></tr>
<tr><td>UV fill fraction</td><td><strong>0.90%</strong></td><td>Sparse array; high sidelobes are real, not a bug</td></tr>
<tr><td>u range</td><td>±48.9 kλ</td><td>EW outrigger baselines</td></tr>
<tr><td>v range</td><td>±10.9 kλ</td><td>NS arm + outrigger baselines</td></tr>
</tbody>
</table>
<div class="callout callout-style-default callout-note callout-titled">
<div class="callout-header d-flex align-content-center">
<div class="callout-icon-container"><i class="callout-icon"></i></div>
<div class="callout-title-container flex-fill">Why the beam is elongated N-S</div>
</div>
<div class="callout-body-container callout-body">
<p>The EW arm is longer than the NS arm and the outriggers extend further EW than NS.
Longer baselines → finer resolution → narrower beam in that direction. The T-core
alone (without outriggers) would give a beam of roughly ~86"×343" — the outriggers
tighten both axes but preserve the asymmetry.</p>
</div>
</div>
</section>

<section id="implications-for-imaging" class="level3">
<h3 class="anchored" data-anchor-id="implications-for-imaging">Implications for imaging</h3>
<p>The high peak sidelobe (0.415) is a direct consequence of the sparse UV fill.
CLEAN must run enough major cycles to drive sidelobes below the noise — WSClean
<code>-niter 50000</code> with an appropriate <code>-mgain</code> (0.7–0.8) is the validated setting
(see <code>docs/reference/imaging.md</code>). The NS-elongated beam means sources near the
phase centre will show NS PSF tails in the dirty image; this is expected and not
a calibration artifact.</p>
</section>
</section>

<hr>

<section id="simulation-parameters" class="level2">
<h2 class="anchored" data-anchor-id="simulation-parameters">Simulation parameters</h2>

<section id="timing" class="level3">
<h3 class="anchored" data-anchor-id="timing">Timing</h3>
<table class="caption-top table">
<thead><tr><th>Parameter</th><th>Value</th><th>Source</th></tr></thead>
<tbody>
<tr><td>Integration time</td><td>12.884902 s</td><td><code>dsa110_measured_parameters.yaml</code> → <code>temporal.integration_time_sec</code></td></tr>
<tr><td>Fields per tile</td><td>24</td><td>One drift-scan tile = 24 integrations</td></tr>
<tr><td>Tile duration</td><td>~309.2 s</td><td>24 × 12.885 s</td></tr>
<tr><td>Declination (fixed)</td><td>+16.15°</td><td>OVRO transit strip</td></tr>
</tbody>
</table>
</section>

<section id="spectral-setup" class="level3">
<h3 class="anchored" data-anchor-id="spectral-setup">Spectral setup</h3>
<table class="caption-top table">
<thead><tr><th>Parameter</th><th>Value</th><th>Source</th></tr></thead>
<tbody>
<tr><td>Subbands</td><td>16</td><td><code>harness.py</code></td></tr>
<tr><td>Channels per subband</td><td>48</td><td><code>harness.py</code></td></tr>
<tr><td>Total channels</td><td>768</td><td>16 × 48</td></tr>
<tr><td>Channel width</td><td><strong>244.140625 kHz</strong></td><td>250 MHz / 1024 correlator channels</td></tr>
<tr><td>Frequency range</td><td>1311–1499 MHz</td><td><code>harness.subband_freqs()</code></td></tr>
<tr><td>Bandwidth</td><td>~187.3 MHz</td><td>768 × 244.14 kHz</td></tr>
</tbody>
</table>
<div class="callout callout-style-default callout-warning callout-titled">
<div class="callout-header d-flex align-content-center">
<div class="callout-icon-container"><i class="callout-icon"></i></div>
<div class="callout-title-container flex-fill">Warning</div>
</div>
<div class="callout-body-container callout-body">
<p><code>dsa110_measured_parameters.yaml</code> previously contained a stale entry
<code>frequency_setup.channel_width.value: 325.520833 kHz</code>. This was wrong — it
reflected a different correlator mode. The correct value is
<code>spectral.channel_width_hz: 244140.625</code>. The stale entry was fixed in commit
<code>6ad8a00</code>.</p>
</div>
</div>
</section>
</section>

<hr>

<section id="regenerating-figures" class="level2">
<h2 class="anchored" data-anchor-id="regenerating-figures">Regenerating figures</h2>
<div class="sourceCode"><pre class="sourceCode bash code-with-copy"><code class="sourceCode bash"><span class="co"># Antenna layout (requires antennas.csv via harness)</span>
<span class="ex">python</span> scripts/plot_tile_image.py <span class="at">--layout-only</span>

<span class="co"># PSF analysis (saves numpy arrays to /tmp/, then renders 6-panel figure)</span>
<span class="ex">python</span> scripts/make_psf_analysis.py</code><button title="Copy to Clipboard" class="code-copy-button"><i class="bi"></i></button></pre></div>
<p>Rendered figures are stored at <code>docs/images/</code> and should be updated whenever the
antenna list or simulation frequency changes.</p>
</section>

</main> <!-- /main -->'''

start_idx = html.find(MAIN_START)
end_idx   = html.find(MAIN_END) + len(MAIN_END)
html = html[:start_idx] + new_main + html[end_idx:]

DST.write_text(html)
print(f"Written → {DST}  ({len(html):,} bytes)")
