/**
 * Load manifest.json (sibling to this page) and render charts.
 */
async function loadManifest() {
  const res = await fetch("manifest.json", { cache: "no-store" });
  if (!res.ok) {
    throw new Error(`manifest.json: ${res.status}`);
  }
  return res.json();
}

function formatBytes(n) {
  if (n === 0 || n == null) return "0 B";
  const units = ["B", "KiB", "MiB", "GiB", "TiB", "PiB"];
  let v = n;
  let i = 0;
  while (v >= 1024 && i < units.length - 1) {
    v /= 1024;
    i += 1;
  }
  return `${v.toFixed(i === 0 ? 0 : 2)} ${units[i]}`;
}

function dayIterator(startStr, endStr) {
  const out = [];
  let d = new Date(startStr + "T12:00:00Z");
  const end = new Date(endStr + "T12:00:00Z");
  while (d <= end) {
    out.push(d.toISOString().slice(0, 10));
    d = new Date(d.getTime() + 86400000);
  }
  return out;
}

function fillKpis(m) {
  document.getElementById("kpi-files").textContent = String(m.totals.file_count);
  document.getElementById("kpi-bytes").textContent =
    m.options.no_stat ? "(no-stat)" : formatBytes(m.totals.total_bytes);
  document.getElementById("kpi-root").textContent = m.source_root;
  document.getElementById("kpi-generated").textContent = m.generated_at;
  const fr = m.freshness;
  document.getElementById("kpi-span").textContent =
    fr.earliest_filename_timestamp_utc && fr.latest_filename_timestamp_utc
      ? `${fr.earliest_filename_timestamp_utc} → ${fr.latest_filename_timestamp_utc}`
      : "—";
  document.getElementById("kpi-mtime").textContent =
    fr.earliest_mtime_utc && fr.latest_mtime_utc
      ? `${fr.earliest_mtime_utc} → ${fr.latest_mtime_utc}`
      : m.options.no_stat ? "(no-stat)" : "—";
  document.getElementById("schema-ver").textContent = `v${m.schema_version}`;
  document.getElementById("opt-no-stat").textContent = m.options.no_stat
    ? "Scan used --no-stat (counts only)"
    : "Full stat() for sizes and mtime";
  let hdf5Line = "—";
  if (m.options && m.options.hdf5_metadata === false) {
    hdf5Line = "HDF5 header reads disabled (--no-hdf5-metadata)";
  } else if (m.pointing) {
    hdf5Line = "Phase-center Dec (and optional timeseries) from HDF5 headers";
  } else if (m.schema_version < 2) {
    hdf5Line = "Schema v1: regenerate with current scanner for Dec metadata";
  } else {
    hdf5Line = "No Dec block in this manifest";
  }
  document.getElementById("opt-hdf5").textContent = hdf5Line;

  if (m.pointing) {
    document.getElementById("kpi-dec-wrap").style.display = "block";
    document.getElementById("kpi-dec-files-wrap").style.display = "block";
    const a = m.pointing.dec_deg_min;
    const b = m.pointing.dec_deg_max;
    document.getElementById("kpi-dec").textContent =
      a != null && b != null
        ? `${a.toFixed(3)} → ${b.toFixed(3)} · ${m.pointing.unique_strip_count} distinct (rounded)`
        : "—";
    document.getElementById("kpi-dec-files").textContent = `${m.pointing.files_with_dec} with Dec · ${m.pointing.files_dec_missing} missing`;
  } else {
    document.getElementById("kpi-dec-wrap").style.display = "none";
    document.getElementById("kpi-dec-files-wrap").style.display = "none";
  }
}

function renderDailyAndCumulative(m) {
  const labels = m.by_day.map((r) => r.date);
  const counts = m.by_day.map((r) => r.count);
  let cum = 0;
  const cumulative = counts.map((c) => {
    cum += c;
    return cum;
  });

  const chartOpts = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: { display: false },
    },
    scales: {
      x: {
        ticks: { maxTicksLimit: 12, color: "#8b949e" },
        grid: { color: "#30363d" },
      },
      y: {
        ticks: { color: "#8b949e" },
        grid: { color: "#30363d" },
      },
    },
  };

  const dailyCtx = document.getElementById("chart-daily");
  new Chart(dailyCtx, {
    type: "bar",
    data: {
      labels,
      datasets: [
        {
          label: "Files",
          data: counts,
          backgroundColor: "rgba(88, 166, 255, 0.55)",
          borderColor: "rgba(88, 166, 255, 1)",
          borderWidth: 0,
        },
      ],
    },
    options: chartOpts,
  });

  const cumCtx = document.getElementById("chart-cum");
  new Chart(cumCtx, {
    type: "line",
    data: {
      labels,
      datasets: [
        {
          label: "Cumulative",
          data: cumulative,
          borderColor: "rgba(63, 185, 80, 1)",
          backgroundColor: "rgba(63, 185, 80, 0.1)",
          fill: true,
          tension: 0.1,
          pointRadius: 0,
        },
      ],
    },
    options: chartOpts,
  });
}

function renderDecByDay(m) {
  const panel = document.getElementById("panel-dec-by-day");
  const has =
    m.pointing && m.by_day && m.by_day.some((r) => r.dec_deg_min != null && r.dec_deg_max != null);
  if (!has) {
    panel.style.display = "none";
    return;
  }
  panel.style.display = "block";
  const labels = m.by_day.map((r) => r.date);
  const mins = m.by_day.map((r) => (r.dec_deg_min != null ? r.dec_deg_min : null));
  const maxs = m.by_day.map((r) => (r.dec_deg_max != null ? r.dec_deg_max : null));
  const decCtx = document.getElementById("chart-dec");
  const chartOpts = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: { display: true, labels: { color: "#8b949e" } },
    },
    scales: {
      x: {
        ticks: { maxTicksLimit: 12, color: "#8b949e" },
        grid: { color: "#30363d" },
      },
      y: {
        title: { display: true, text: "Dec (°)", color: "#8b949e" },
        ticks: { color: "#8b949e" },
        grid: { color: "#30363d" },
      },
    },
  };
  new Chart(decCtx, {
    type: "line",
    data: {
      labels,
      datasets: [
        {
          label: "Dec min",
          data: mins,
          borderColor: "rgba(240, 136, 62, 1)",
          backgroundColor: "rgba(240, 136, 62, 0.15)",
          fill: true,
          tension: 0.15,
          pointRadius: 0,
        },
        {
          label: "Dec max",
          data: maxs,
          borderColor: "rgba(163, 113, 247, 1)",
          backgroundColor: "rgba(163, 113, 247, 0.1)",
          fill: true,
          tension: 0.15,
          pointRadius: 0,
        },
      ],
    },
    options: chartOpts,
  });
}

/**
 * @param {object} m manifest
 * @param {Array<{ filename: string, t_mid_utc: string|null, ra_deg: number|null, dec_deg: number|null }>|null} rows
 */
function renderPointingTimeseries(m, rows) {
  const panel = document.getElementById("panel-pointing-series");
  const note = document.getElementById("ts-trunc");
  if (!rows || !Array.isArray(rows) || rows.length === 0) {
    panel.style.display = "none";
    return;
  }
  panel.style.display = "block";
  if (m.pointing_timeseries && m.pointing_timeseries.truncated) {
    note.style.display = "block";
    note.textContent = "Rows capped; scan omitted some files (see manifest pointing_timeseries).";
  } else {
    note.style.display = "none";
  }
  const labels = rows.map((r) => (r.t_mid_utc || r.filename || "—").slice(0, 19));
  const ra = rows.map((r) => (r.ra_deg != null ? r.ra_deg : null));
  const dec = rows.map((r) => (r.dec_deg != null ? r.dec_deg : null));
  const pctx = document.getElementById("chart-pointing");
  new Chart(pctx, {
    type: "line",
    data: {
      labels,
      datasets: [
        {
          label: "RA (°)",
          data: ra,
          borderColor: "rgba(88, 166, 255, 1)",
          yAxisID: "y",
          pointRadius: 0,
          tension: 0.1,
        },
        {
          label: "Dec (°)",
          data: dec,
          borderColor: "rgba(240, 136, 62, 1)",
          yAxisID: "y1",
          pointRadius: 0,
          tension: 0.1,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: { mode: "index", intersect: false },
      plugins: {
        legend: { labels: { color: "#8b949e" } },
      },
      scales: {
        x: {
          ticks: { maxTicksLimit: 16, color: "#8b949e", maxRotation: 60 },
          grid: { color: "#30363d" },
        },
        y: {
          type: "linear",
          position: "left",
          title: { display: true, text: "RA (°)", color: "#8b949e" },
          ticks: { color: "#8b949e" },
          grid: { color: "#30363d" },
        },
        y1: {
          type: "linear",
          position: "right",
          title: { display: true, text: "Dec (°)", color: "#8b949e" },
          ticks: { color: "#8b949e" },
          grid: { drawOnChartArea: false },
        },
      },
    },
  });
}

async function loadTimeseriesIfPresent(m) {
  if (!m.pointing_timeseries || !m.pointing_timeseries.file) {
    return null;
  }
  const res = await fetch(m.pointing_timeseries.file, { cache: "no-store" });
  if (!res.ok) {
    return null;
  }
  return res.json();
}

function renderHeatmap(m) {
  const byDay = new Map(m.by_day.map((r) => [r.date, r.count]));
  if (m.by_day.length === 0) {
    document.getElementById("heatmap").textContent = "No data.";
    document.getElementById("heatmap-axis").textContent = "";
    document.getElementById("heatmap-axis-wrap").style.display = "none";
    return;
  }
  const first = m.by_day[0].date;
  const last = m.by_day[m.by_day.length - 1].date;
  const days = dayIterator(first, last);
  const counts = days.map((d) => byDay.get(d) || 0);
  const maxC = Math.max(1, ...counts);

  const heat = document.getElementById("heatmap");
  const axisWrap = document.getElementById("heatmap-axis-wrap");
  const axis = document.getElementById("heatmap-axis");
  heat.innerHTML = "";
  axis.innerHTML = "";
  axisWrap.style.display = "block";
  let lastLabeledYear = null;
  days.forEach((d, i) => {
    const c = counts[i];
    const el = document.createElement("div");
    el.className = "heatmap-cell";
    const t = c / maxC;
    const alpha = 0.15 + t * 0.85;
    el.style.background = `rgba(88, 166, 255, ${alpha})`;
    el.title = `${d}: ${c} files`;
    heat.appendChild(el);

    const slot = document.createElement("div");
    slot.className = "heatmap-axis-slot";
    const isFirst = i === 0;
    const isLast = i === days.length - 1;
    const isMonthStart = d.endsWith("-01");
    if (isFirst || isLast || isMonthStart) {
      slot.classList.add("tick");
      const dt = new Date(d + "T12:00:00Z");
      const month = dt.toLocaleString("en-US", { month: "short", timeZone: "UTC" });
      const year = dt.getUTCFullYear();
      const showYear = year !== lastLabeledYear;
      const label = document.createElement("span");
      label.className = "heatmap-axis-label";
      label.textContent = showYear ? `${month} ${year}` : month;
      slot.appendChild(label);
      lastLabeledYear = year;
    }
    axis.appendChild(slot);
  });

  // Keep heatmap cells and date axis horizontally aligned while scrolling.
  let syncing = false;
  const syncToAxis = () => {
    if (syncing) return;
    syncing = true;
    axisWrap.scrollLeft = heat.scrollLeft;
    syncing = false;
  };
  const syncToHeat = () => {
    if (syncing) return;
    syncing = true;
    heat.scrollLeft = axisWrap.scrollLeft;
    syncing = false;
  };
  heat.onscroll = syncToAxis;
  axisWrap.onscroll = syncToHeat;

  const leg = document.getElementById("heatmap-legend");
  leg.textContent = `${first} → ${last} · max ${maxC} files/day`;
}

function renderGaps(m) {
  const ol = document.getElementById("gaps-list");
  ol.innerHTML = "";
  const sorted = [...m.gaps].sort((a, b) => b.days - a.days);
  const top = sorted.slice(0, 10);
  if (top.length === 0) {
    const li = document.createElement("li");
    li.textContent = "No gaps in the covered date range (or no files).";
    ol.appendChild(li);
    return;
  }
  top.forEach((g) => {
    const li = document.createElement("li");
    li.textContent = `${g.start} → ${g.end} (${g.days} days)`;
    ol.appendChild(li);
  });
}

function renderRecent(m) {
  const tbody = document.querySelector("#table-recent tbody");
  tbody.innerHTML = "";
  const rows = [...m.by_day].reverse().slice(0, 7);
  rows.forEach((r) => {
    const tr = document.createElement("tr");
    tr.innerHTML = `<td>${r.date}</td><td>${r.count}</td><td>${
      m.options.no_stat ? "—" : formatBytes(r.bytes)
    }</td>`;
    tbody.appendChild(tr);
  });
}

function renderBeams(m) {
  const tbody = document.querySelector("#table-beam tbody");
  tbody.innerHTML = "";
  m.by_beam.forEach((b) => {
    const tr = document.createElement("tr");
    tr.innerHTML = `<td>sb${String(b.beam).padStart(2, "0")}</td><td>${b.count}</td><td>${
      m.options.no_stat ? "—" : formatBytes(b.bytes)
    }</td>`;
    tbody.appendChild(tr);
  });
}

function main() {
  loadManifest()
    .then(async (m) => {
      fillKpis(m);
      renderDailyAndCumulative(m);
      renderDecByDay(m);
      let ts = null;
      try {
        ts = await loadTimeseriesIfPresent(m);
      } catch (e) {
        ts = null;
      }
      renderPointingTimeseries(m, ts);
      renderHeatmap(m);
      renderGaps(m);
      renderRecent(m);
      renderBeams(m);
    })
    .catch((err) => {
      document.body.innerHTML = `<pre style="padding:1rem;color:#f85149;">Failed to load manifest: ${err}</pre>`;
    });
}

main();
