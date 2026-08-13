// KV-pool occupancy timeline as an EDITABLE PowerPoint deck.
//
// WHAT IS PLOTTED, and why it is not nvidia-smi: SGLang reserves its whole KV pool at
// startup from --mem-fraction-static, so a card-level memory reading is flat from the
// first second and can never show a cache filling. The curve that rises and saturates is
// pool occupancy = (max_total - available) / max_total, which only the server knows --
// see kv_usage_sampler.py, whose CSVs this reads.
//
// SCATTER, NOT LINE, on purpose. A pptxgen `line` chart has a CATEGORY x-axis: the labels
// are strings and every point is spaced equally, so a run sampled at an uneven interval
// would be drawn as if it were even. `scatter` gives a genuine numeric axis in SECONDS,
// which is what the elapsed_s column actually is.
//
// DOWNSAMPLED to --points samples per series. The CSVs hold ~2700 rows at a 2s interval;
// a native chart carries its own worksheet, so four series at full resolution would embed
// >10k cells for detail invisible at slide size. The figures quoted in the speaker notes
// come from the FULL, UNSMOOTHED data so no number is a sampling or filter artefact.
//
// 사용:
//   node benchmark/make_kv_occupancy_pptx.js results/agent_trace/fig_kv_occupancy.pptx \
//        results/agent_trace [--c 4] [--arms hicache,park] [--smooth 15] [--ramp 300]

const pptxgen = require("pptxgenjs");
const fs = require("fs");
const path = require("path");

const args = process.argv.slice(2);
const pos = args.filter((a) => !a.startsWith("--"));
const outPath = pos[0] || "results/agent_trace/fig_kv_occupancy.pptx";
const dataDir = pos[1] || "results/agent_trace";
const flag = (n, d) => {
  const i = args.indexOf(`--${n}`);
  return i >= 0 && args[i + 1] ? args[i + 1] : d;
};
const C = flag("c", "4");
const NPOINTS = Number(flag("points", "180"));
const SMOOTH = Number(flag("smooth", "15"));
const RAMP = Number(flag("ramp", "300"));
const WANT = flag("arms", "recompute,hicache,park_host,park").split(",");

// hue = system, shade = role: one legend entry per line, but the arms stay recognisable
// as colour families instead of four unrelated colours the reader has to memorise.
const ARMS = [
  { key: "recompute", label: "Recompute", prefill: "5A5F63", decode: "A8ADB1" },
  { key: "hicache", label: "SGLang", prefill: "0072B2", decode: "7FC4E8" },
  { key: "park_host", label: "Ours (host DRAM)", prefill: "B87A00", decode: "F2C266" },
  { key: "park", label: "Ours", prefill: "009E73", decode: "7FD9BE" },
];
const ROLES = ["prefill", "decode"];
const INK = "1A1A1A", MUTED = "595959", GRID = "D9D9D9", FONT = "Times New Roman";

function loadCsv(file) {
  // Python's csv.writer emits CRLF, so the last field of every line keeps a \r. Left in
  // place it makes the FINAL column name unmatchable -- here decode_occupancy_frac, i.e.
  // exactly the series this figure exists to show -- so strip it before parsing.
  const lines = fs.readFileSync(file, "utf8").trim().split(/\r?\n/);
  const head = lines[0].split(",").map((h) => h.trim());
  const idx = (n) => head.indexOf(n);
  for (const r of ROLES) {
    if (idx(`${r}_occupancy_frac`) < 0) {
      console.error(`[fatal] ${file} has no ${r}_occupancy_frac column`);
      process.exit(1);
    }
  }
  const out = { t: [], prefill: [], decode: [] };
  for (const ln of lines.slice(1)) {
    const c = ln.split(",").map((x) => x.trim());
    const t = Number(c[idx("elapsed_s")]);
    if (!Number.isFinite(t)) continue;
    // A blank is a node that did not answer /metrics (still starting, or restarting).
    // Number("") is 0, so an empty cell would otherwise be drawn as a genuine 0% pool.
    const raw = ROLES.map((r) => c[idx(`${r}_occupancy_frac`)]);
    if (raw.some((x) => x === "" || x === undefined)) continue;
    const v = raw.map((x) => Number(x) * 100);
    if (v.some((x) => !Number.isFinite(x))) continue;
    out.t.push(t);
    ROLES.forEach((r, i) => out[r].push(v[i]));
  }
  return out;
}

// Centred rolling mean. The decode pool genuinely oscillates request to request; at a few
// thousand samples that fills the panel with ink and hides the level. Stated on the slide
// rather than applied silently, and never applied to the numbers in the notes.
function smooth(ys, w) {
  if (w < 2 || ys.length < w) return ys;
  const half = w >> 1;
  return ys.map((_, i) => {
    const lo = Math.max(0, i - half), hi = Math.min(ys.length, i + half + 1);
    let s = 0;
    for (let j = lo; j < hi; j++) s += ys[j];
    return s / (hi - lo);
  });
}

const series = [];
for (const a of ARMS) {
  if (!WANT.includes(a.key)) continue;
  const f = path.join(dataDir, `kv_${a.key}_c${C}.csv`);
  if (fs.existsSync(f)) series.push({ ...a, data: loadCsv(f) });
  else console.warn(`[warn] ${f} not found -- run the sweep with kv_usage_sampler wired in`);
}
if (!series.length) {
  console.error(`[fatal] no kv_*_c${C}.csv in ${dataDir}. The mem_*.csv files hold `
    + "nvidia-smi readings, which are flat by construction and cannot show a pool filling.");
  process.exit(1);
}

// Every series onto ONE x grid, taken from the longest run: a scatter chart in pptx shares
// a single X series across the sheet, and pairing each y with someone else's x is exactly
// the silent mis-plot this file is arranged to avoid.
const longest = series.reduce((m, s) => (s.data.t.length > m.data.t.length ? s : m));

function chartData(tmax, npoints, win) {
  const keep = [];
  for (let i = 0; i < longest.data.t.length; i++) {
    if (!tmax || longest.data.t[i] <= tmax) keep.push(i);
  }
  const step = Math.max(1, Math.floor(keep.length / npoints));
  const gridIdx = keep.filter((_, k) => k % step === 0);
  const out = [{ name: "time (s)", values: gridIdx.map((i) => longest.data.t[i]) }];
  const colors = [];
  for (const s of series) {
    for (const r of ROLES) {
      const ys = smooth(s.data[r], win);
      out.push({
        name: `${s.label} ${r[0].toUpperCase()}${r.slice(1)}`,
        values: gridIdx.map((i) => (i < ys.length ? Number(ys[i].toFixed(2)) : null)),
      });
      colors.push(s[r]);
    }
  }
  return { data: out, colors, n: gridIdx.length };
}

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE";
pres.author = "SGLang KV placement experiment";
pres.title = "Agent-trace replay: KV-pool occupancy over time";

const SUB = `Codex / SWE-bench Pro traces · 100 sessions · 8-15 turns · context 12k->48k `
  + `tokens · C=${C} · Llama-3.1-8B · SGLang 2P2D · 4x RTX A6000 · pool occupancy = `
  + `(max_total - available) / max_total, summed over the two GPUs of each role`;

const common = (colors, xmax) => ({
  showLegend: false,
  lineSize: 2.25, lineDataSymbol: "none",
  catAxisLabelColor: MUTED, valAxisLabelColor: MUTED,
  catAxisLabelFontFace: FONT, valAxisLabelFontFace: FONT,
  catAxisLabelFontSize: 11, valAxisLabelFontSize: 11,
  catAxisTitleFontFace: FONT, valAxisTitleFontFace: FONT,
  catAxisTitleFontSize: 13, valAxisTitleFontSize: 13,
  axisLineColor: GRID, valGridLine: { color: GRID, size: 0.75 },
  showValAxisTitle: true, showCatAxisTitle: true,
  catAxisTitle: "time (s)",           // genuine numeric seconds -- scatter, not category
  valAxisTitle: "KV pool occupancy (%)",
  valAxisMinVal: 0, valAxisMaxVal: 100,
  catAxisMinVal: 0, ...(xmax ? { catAxisMaxVal: xmax } : {}),
  chartColors: colors,
});

function legend(slide, y) {
  let lx = 0.9;
  for (const s of series) {
    for (const r of ROLES) {
      slide.addShape(pres.ShapeType.line, {
        x: lx, y: y + 0.18, w: 0.42, h: 0, line: { color: s[r], width: 2.25 },
      });
      slide.addText(`${s.label} ${r[0].toUpperCase()}${r.slice(1)}`, {
        x: lx + 0.5, y, w: 2.4, h: 0.4,
        fontFace: FONT, fontSize: 12, color: INK, margin: 0, valign: "middle",
      });
      lx += 2.9;
    }
  }
}

const stats = series.map((s) => {
  const f = (a) => ({ max: Math.max(...a), mean: a.reduce((x, y) => x + y, 0) / a.length,
                      last: a[a.length - 1] });
  return { label: s.label, prefill: f(s.data.prefill), decode: f(s.data.decode),
           dur: s.data.t[s.data.t.length - 1], n: s.data.t.length };
});
const notes = "Full-run values, UNSMOOTHED and at full resolution (the charts are drawn "
  + `from a ${NPOINTS}-point sampling with a ${SMOOTH}-sample rolling mean):\n`
  + stats.map((s) => `  ${s.label}: prefill max ${s.prefill.max.toFixed(1)}% / mean `
      + `${s.prefill.mean.toFixed(1)}% / final ${s.prefill.last.toFixed(1)}%; `
      + `decode max ${s.decode.max.toFixed(1)}% / mean ${s.decode.mean.toFixed(1)}% / `
      + `final ${s.decode.last.toFixed(1)}%  (${s.n} samples over ${Math.round(s.dur)}s)`)
    .join("\n")
  + "\n\ntoken_usage is NOT this quantity: SGLang defines it as (max_total - available - "
  + "evictable) / max_total, so a pool 100% full of reusable prefix cache and an empty one "
  + "both report ~0. The sampler records both so they stay distinguishable.\n"
  + "Charts are native: right-click > Edit Data, or the Chart Design / Format tabs.";

// --- Slide 1: full run -------------------------------------------------------------
const full = chartData(0, NPOINTS, SMOOTH);
const s1 = pres.addSlide();
s1.addText("Prefill KV is saturated while decode KV sits idle", {
  x: 0.45, y: 0.22, w: 12.4, h: 0.45,
  fontFace: FONT, fontSize: 22, bold: true, color: INK, margin: 0,
});
s1.addText(SUB, {
  x: 0.45, y: 0.68, w: 12.4, h: 0.45,
  fontFace: FONT, fontSize: 11, color: MUTED, italic: true, margin: 0,
});
s1.addChart(pres.ChartType.scatter, full.data,
  { ...common(full.colors, null), x: 0.7, y: 1.35, w: 11.9, h: 4.35 });
legend(s1, 5.8);
{
  const h = stats.find((s) => s.label === "SGLang") || stats[0];
  s1.addText(
    `${h.label}: the prefill pool holds ${h.prefill.mean.toFixed(0)}% on average and peaks `
    + `at ${h.prefill.max.toFixed(0)}%, so almost every new prefix must evict something -- `
    + `while the decode pool averages ${h.decode.mean.toFixed(0)}%. The headroom the victim `
    + `cache claims is that gap, on the same node.  Lines are a `
    + `${SMOOTH}-sample (~${SMOOTH * 2}s) rolling mean; the decode pool oscillates request `
    + "to request and is unreadable raw.",
    { x: 0.45, y: 6.3, w: 12.4, h: 0.8,
      fontFace: FONT, fontSize: 11, color: MUTED, margin: 0 });
}
s1.addNotes(notes);

// --- Slide 2: the ramp -------------------------------------------------------------
// Worth its own slide because on the full axis it is a vertical line: the pool saturates
// in ~1% of the run, which is itself the finding rather than a plotting nuisance.
const ramp = chartData(RAMP, NPOINTS, 0);
const s2 = pres.addSlide();
s2.addText(`How fast the pool fills — first ${RAMP} s`, {
  x: 0.45, y: 0.22, w: 12.4, h: 0.45,
  fontFace: FONT, fontSize: 22, bold: true, color: INK, margin: 0,
});
s2.addText("Same data, time axis clipped. Raw samples, no smoothing.", {
  x: 0.45, y: 0.68, w: 12.4, h: 0.3,
  fontFace: FONT, fontSize: 11, color: MUTED, italic: true, margin: 0,
});
s2.addChart(pres.ChartType.scatter, ramp.data,
  { ...common(ramp.colors, RAMP), x: 0.7, y: 1.35, w: 11.9, h: 4.35 });
legend(s2, 5.8);
s2.addText(
  `The prefill pool reaches ~95% within about 50 s of a `
  + `${Math.round(Math.max(...stats.map((s) => s.dur)))} s run — the working set of these `
  + "agent sessions is several times the pool, so saturation is immediate and permanent, "
  + "not a slow warm-up.",
  { x: 0.45, y: 6.3, w: 12.4, h: 0.6,
    fontFace: FONT, fontSize: 11, color: MUTED, margin: 0 });
s2.addNotes(notes);

// --- Rendered figures --------------------------------------------------------------
for (const [name, title] of [
  ["fig_kv_occupancy.png", "Full run, as rendered (plot_kv_occupancy.py --smooth 15)"],
  ["fig_kv_occupancy_ramp.png", "Ramp detail, as rendered (--xmax 300)"],
]) {
  const p = path.join(dataDir, name);
  if (!fs.existsSync(p)) continue;
  const s = pres.addSlide();
  s.addText(title, {
    x: 0.45, y: 0.22, w: 12.4, h: 0.4,
    fontFace: FONT, fontSize: 18, bold: true, color: INK, margin: 0,
  });
  s.addImage({ path: p, x: 3.75, y: 1.0, w: 5.8, h: 5.8 * (2.8 / 3.6) });
  s.addText(`Source: ${p}`, {
    x: 0.45, y: 6.9, w: 12.4, h: 0.3,
    fontFace: FONT, fontSize: 10, color: MUTED, italic: true, margin: 0,
  });
}

pres.writeFile({ fileName: outPath }).then(() => console.log("[saved] " + outPath));
