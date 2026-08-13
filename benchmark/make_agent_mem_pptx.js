// Memory-occupancy timeline as an EDITABLE PowerPoint deck: three native line charts
// (prefill HBM / decode HBM / host DRAM), one series per arm, from the sys_mem_breakdown
// CSVs the sweep already writes.
//
// DOWNSAMPLED to --points samples per series. The CSVs hold ~2400 rows each at a 2s
// interval; a native chart carries its own worksheet, so three charts x three arms x
// 2400 points would embed ~20k cells and make the deck slow to open for detail no one
// can see at slide size. Sampling is uniform over the row index, and the peak of each
// series is reported in the speaker notes from the FULL data so the number quoted is
// never a sampling artefact.
//
// 사용:
//   node benchmark/make_agent_mem_pptx.js results/agent_trace/fig_agent_mem.pptx \
//        results/agent_trace [--c 4] [--layout a]

const pptxgen = require("pptxgenjs");
const fs = require("fs");
const path = require("path");

const args = process.argv.slice(2);
const pos = args.filter((a) => !a.startsWith("--"));
const outPath = pos[0] || "results/agent_trace/fig_agent_mem.pptx";
const dataDir = pos[1] || "results/agent_trace";
const flag = (n, d) => {
  const i = args.indexOf(`--${n}`);
  return i >= 0 && args[i + 1] ? args[i + 1] : d;
};
const C = flag("c", "4");
const LAYOUT = flag("layout", "a");
const NPOINTS = Number(flag("points", "150"));

const ARMS = [
  { key: "recompute", label: "Recompute", color: "8A9199" },
  { key: "hicache", label: "SGLang", color: "0072B2" },
  { key: "park_host", label: "Ours (host DRAM)", color: "E69F00" },
  { key: "park", label: "Ours", color: "009E73" },
];
// Which GPUs are prefill is a property of PD_LAYOUT, not of column order: layout 'b'
// interleaves roles (P0=0, D0=1, P1=2, D1=3), and reading the wrong mapping would swap
// two charts and invert the conclusion.
const LAYOUTS = { a: { prefill: [0, 1], decode: [2, 3] },
                  b: { prefill: [0, 2], decode: [1, 3] } };
const gpus = LAYOUTS[LAYOUT];
if (!gpus) { console.error(`[fatal] unknown --layout ${LAYOUT}`); process.exit(1); }

const INK = "1A1A1A", MUTED = "595959", GRID = "D9D9D9", FONT = "Times New Roman";

function loadCsv(file) {
  const lines = fs.readFileSync(file, "utf8").trim().split("\n");
  const head = lines[0].split(",");
  const idx = (n) => head.indexOf(n);
  const out = { t: [], prefill: [], decode: [], host: [] };
  for (const ln of lines.slice(1)) {
    const c = ln.split(",");
    const num = (n) => Number(c[idx(n)]);
    const t = num("elapsed_s");
    if (!Number.isFinite(t)) continue;
    out.t.push(t);
    out.prefill.push(gpus.prefill.reduce((s, g) => s + num(`gpu${g}_hbm_mb`), 0) / 1024);
    out.decode.push(gpus.decode.reduce((s, g) => s + num(`gpu${g}_hbm_mb`), 0) / 1024);
    // anonpages, not file_cache: page cache is reclaimable, so counting it would charge
    // a disk tier as pressure it can hand back on demand.
    out.host.push(num("anonpages_mb") / 1024);
  }
  return out;
}

const series = [];
for (const a of ARMS) {
  const f = path.join(dataDir, `mem_${a.key}_c${C}.csv`);
  if (fs.existsSync(f)) series.push({ ...a, data: loadCsv(f) });
}
if (!series.length) {
  console.error(`[fatal] no mem_*_c${C}.csv in ${dataDir}`);
  process.exit(1);
}

// One shared category axis: sample every series onto the SAME time grid, taken from the
// longest run. A per-series grid would silently pair each y with the wrong x, the same
// trap the scatter guards in the other decks exist for.
const longest = series.reduce((m, s) => (s.data.t.length > m.data.t.length ? s : m));
const step = Math.max(1, Math.floor(longest.data.t.length / NPOINTS));
const gridIdx = [];
for (let i = 0; i < longest.data.t.length; i += step) gridIdx.push(i);
const labels = gridIdx.map((i) => Math.round(longest.data.t[i]).toString());
const sampleOnto = (s, field) =>
  gridIdx.map((i) => (i < s.data[field].length ? Number(s.data[field][i].toFixed(2)) : null));

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE";
pres.author = "SGLang KV placement experiment";
pres.title = "Agent-trace replay: memory occupancy over time";

const s1 = pres.addSlide();
s1.addText("The same reusable KV, held in idle GPU memory instead of CPU memory", {
  x: 0.45, y: 0.22, w: 12.4, h: 0.45,
  fontFace: FONT, fontSize: 22, bold: true, color: INK, margin: 0,
});
s1.addText(
  `Codex / SWE-bench Pro traces · 100 sessions · 8-15 turns · context 12k->48k tokens · `
  + `C=${C} · Llama-3.1-8B · SGLang 2P2D · 4x RTX A6000 (49GB each, two GPUs summed per role)`,
  { x: 0.45, y: 0.68, w: 12.4, h: 0.3,
    fontFace: FONT, fontSize: 11, color: MUTED, italic: true, margin: 0 }
);

const common = {
  showLegend: false,
  lineSize: 2, lineDataSymbol: "none",
  catAxisLabelColor: MUTED, valAxisLabelColor: MUTED,
  catAxisLabelFontFace: FONT, valAxisLabelFontFace: FONT,
  catAxisLabelFontSize: 9, valAxisLabelFontSize: 10,
  catAxisTitleFontFace: FONT, valAxisTitleFontFace: FONT,
  catAxisTitleFontSize: 11, valAxisTitleFontSize: 11,
  catAxisLabelFrequency: Math.ceil(labels.length / 6),
  axisLineColor: GRID, valGridLine: { color: GRID, size: 0.75 },
  valAxisMinVal: 0,
  showValAxisTitle: true, showCatAxisTitle: true,
  catAxisTitle: "time (s)",
  showTitle: true, titleFontFace: FONT, titleFontSize: 13, titleColor: INK,
  chartColors: series.map((s) => s.color),
};

const PANELS = [
  { field: "prefill", title: "(a)  Prefill GPUs", axis: "Prefill HBM (GB)", max: 98 },
  { field: "decode", title: "(b)  Decode GPUs", axis: "Decode HBM (GB)", max: 98 },
  { field: "host", title: "(c)  CPU memory (anon)", axis: "Host DRAM (GB)", max: null },
];
const X0 = 0.5, W = 4.0, GAP = 0.25, Y = 1.35, H = 4.3;
PANELS.forEach((p, i) => {
  s1.addChart(
    pres.ChartType.line,
    series.map((s) => ({ name: s.label, labels, values: sampleOnto(s, p.field) })),
    { ...common, x: X0 + i * (W + GAP), y: Y, w: W, h: H,
      title: p.title, valAxisTitle: p.axis,
      ...(p.max ? { valAxisMaxVal: p.max } : {}) }
  );
});

let lx = 2.6;
for (const s of series) {
  s1.addShape(pres.ShapeType.line, {
    x: lx, y: 5.94, w: 0.42, h: 0, line: { color: s.color, width: 2.25 },
  });
  s1.addText(s.label, {
    x: lx + 0.5, y: 5.76, w: 2.0, h: 0.4,
    fontFace: FONT, fontSize: 12, color: INK, margin: 0, valign: "middle",
  });
  lx += 2.5;
}
s1.addText(
  "Read (a) and (c) together: on their own, more prefill HBM could be a leak and less "
  + "host DRAM could be a smaller cache. Moving in opposite directions, they say the "
  + "bytes changed location. (b) is flat because this run parks only onto each prefill's "
  + "own GPU (PARK_PEER=0) and never reaches a decode one.",
  { x: 0.45, y: 6.35, w: 12.4, h: 0.8,
    fontFace: FONT, fontSize: 11, color: MUTED, margin: 0 }
);
s1.addNotes(
  `Peaks from the FULL CSVs (not the ${NPOINTS}-point sampling used to draw the charts):\n`
  + series.map((s) => `  ${s.label}: prefill ${Math.max(...s.data.prefill).toFixed(1)} GB, `
      + `decode ${Math.max(...s.data.decode).toFixed(1)} GB, `
      + `host ${Math.max(...s.data.host).toFixed(1)} GB`).join("\n")
  + `\n\nGPU roles from PD_LAYOUT='${LAYOUT}': prefill = gpu${gpus.prefill.join("+gpu")}, `
  + `decode = gpu${gpus.decode.join("+gpu")}.\n`
  + "Charts are native: right-click > Edit Data, or the Chart Design / Format tabs."
);

// Rendered figures on their own slides -- the matplotlib versions are the camera-ready
// ones, and the merged single-axes view is the more compact way to show the same thing.
for (const [name, title] of [
  ["fig_agent_mem.png", "Full figure, as rendered (plot_agent_mem_timeline.py)"],
  ["fig_agent_mem_merged.png", "Single-axes version (--merge): colour = location, style = system"],
]) {
  const p = path.join(dataDir, name);
  if (!fs.existsSync(p)) continue;
  const s = pres.addSlide();
  s.addText(title, {
    x: 0.45, y: 0.22, w: 12.4, h: 0.4,
    fontFace: FONT, fontSize: 18, bold: true, color: INK, margin: 0,
  });
  const wide = name.includes("merged");
  s.addImage(wide
    ? { path: p, x: 3.3, y: 0.95, w: 6.7, h: 6.7 * (3.13 / 5.43) }
    : { path: p, x: 0.9, y: 1.4, w: 11.5, h: 11.5 * (2.5 / 9.5) });
  s.addText(`Source: ${p}`, {
    x: 0.45, y: 6.9, w: 12.4, h: 0.3,
    fontFace: FONT, fontSize: 10, color: MUTED, italic: true, margin: 0,
  });
}

pres.writeFile({ fileName: outPath }).then(() => console.log("[saved] " + outPath));
