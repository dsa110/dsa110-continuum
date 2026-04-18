"""
Patch all existing _site/*.html pages to add 'Array Reference' to their sidebars.
Also update index.html pages table.
"""
from pathlib import Path

SITE = Path("_site")

NEW_ITEM = """        <li class="sidebar-item">
  <div class="sidebar-item-container"> 
  <a href="./array-reference.html" class="sidebar-item-text sidebar-link">
 <span class="menu-text">Array Reference</span></a>
  </div>
</li>"""

OLD_EPOCH_ITEM = """        <li class="sidebar-item">
  <div class="sidebar-item-container"> 
  <a href="./epoch-qa.html" class="sidebar-item-text sidebar-link"""

INSERT_BEFORE = '        <li class="sidebar-item">\n  <div class="sidebar-item-container"> \n  <a href="./epoch-qa.html"'

for html_file in sorted(SITE.glob("*.html")):
    if html_file.name == "array-reference.html":
        continue  # already correct
    html = html_file.read_text()
    if "./array-reference.html" in html:
        print(f"  skip (already patched): {html_file.name}")
        continue
    if INSERT_BEFORE not in html:
        print(f"  WARN - anchor not found: {html_file.name}")
        continue
    html = html.replace(INSERT_BEFORE, NEW_ITEM + "\n" + INSERT_BEFORE, 1)
    html_file.write_text(html)
    print(f"  patched sidebar: {html_file.name}")

# Also update index.html pages table
idx = SITE / "index.html"
idx_html = idx.read_text()
if "array-reference.html" not in idx_html:
    OLD_ROW = ('<td><a href="./pipeline-overview.html">Pipeline Overview</a></td>\n'
               '<td>Stages, entry points, calibration, imaging, mosaicking, known silent-failure risks</td>')
    NEW_ROW = (OLD_ROW + '\n</tr>\n<tr class="odd">\n'
               '<td><a href="./array-reference.html">Array Reference</a></td>\n'
               '<td>117-station ENU layout, synthesized beam at Dec +16°, simulation parameters, harness configuration</td>')
    if OLD_ROW in idx_html:
        idx_html = idx_html.replace(OLD_ROW, NEW_ROW, 1)
        idx.write_text(idx_html)
        print("  patched index.html pages table")
    else:
        print("  WARN: index pages table anchor not found")
else:
    print("  skip index.html (already patched)")

print("Done.")
