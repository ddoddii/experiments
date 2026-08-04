// results/intro/fig_ttft_ctx_sweep as an EDITABLE PowerPoint slide: two NATIVE line charts.
//
// LINE, not scatter -- and that is the opposite call to make_qps_sweep_pptx.js, for a
// reason worth keeping written down. A pptxgenjs line chart lays its x values out as
// CATEGORIES, evenly spaced whatever their magnitude. The QPS sweep plots 2,4,8,...,64 on
// a LINEAR axis, so even spacing would have stretched 2->4 to the width of 48->64 and
// flattened the knee; it needed scatter. This sweep plots 1k..32k on a LOG2 axis
// (ttft_ctx_sweep.render: set_xscale("log", base=2)), and the lengths are exact powers of
// two, so log2 spacing IS even spacing. Categories reproduce the matplotlib axis exactly,
// and a line chart keeps the axis labels as clean "1 2 4 8 16 32".
//
// The throughput panel is only meaningful because every request emitted exactly
// gen_tokens output tokens (ignore_eos). Without that the wall time carries a random
// decode length and the curve measures output length rather than prefill, so the JSON is
// checked for it before anything is drawn.
//
// 사용:
//   node benchmark/make_ctx_sweep_pptx.js results/intro/fig_ttft_ctx_sweep.pptx \
//        results/intro/fig_ttft_ctx_sweep.json

const pptxgen = require("pptxgenjs");
const fs = require("fs");
const { execFileSync } = require("child_process");

const outPath = process.argv[2] || "results/intro/fig_ttft_ctx_sweep.pptx";
const dataPath = process.argv[3] || "results/intro/fig_ttft_ctx_sweep.json";
const d = JSON.parse(fs.readFileSync(dataPath, "utf8"));

const G = d.gen_tokens;
const hasTput = G > 1 && Array.isArray(d.cached_tps) && d.cached_tps.every((x) => x);

// The output length has to have been held constant, or panel (b) is not a measurement.
// This produced a visibly wrong figure once -- both arms dipping at the same lengths --
// so it is checked rather than trusted.
if (hasTput) {
  const odd = [];
  for (const [L, r] of Object.entries(d.raw || {})) {
    for (const k of ["cached_n_out", "recompute_n_out"]) {
      if (!r[k]) { odd.push(`${L}:${k} not recorded`); continue; }
      const bad = [...new Set(r[k].filter((n) => n !== G))];
      if (bad.length) odd.push(`${L}:${k}=${bad}`);
    }
  }
  if (odd.length) {
    console.error(`[fatal] output length was not constant (${odd.join(", ")}), so the\n` +
                  `        throughput panel measures how many tokens each random prompt\n` +
                  `        happened to emit, not the prefill difference. Re-run the sweep\n` +
                  `        with the current ttft_ctx_sweep.py (it sets ignore_eos).`);
    process.exit(2);
  }
}

// paperstyle, minus the '#': pptxgenjs corrupts the file on '#' or 8-digit hex.
const C_CACHED = "009E73";
const C_RECOMP = "B4423C";
const INK = "1A1A1A";
const MUTED = "595959";
const GRID = "D9D9D9";
const FONT = "Times New Roman";

const cats = d.lengths.map((L) => (L % 1000 === 0 ? `${L / 1000}` : `${(L / 1000).toFixed(1)}`));

const PANELS = [
  { title: "(a)  Time to first token", axis: "mean TTFT (ms)",
    cached: d.cached_ms, recompute: d.recompute_ms, x: 0.35 },
  { title: `(b)  End-to-end throughput (${G} output tokens)`, axis: "throughput (tok/s)",
    cached: d.cached_tps, recompute: d.recompute_tps, x: 5.0 },
].filter((p) => p.cached && p.recompute && p.cached.every((v) => v != null));

const pptx = new pptxgen();
pptx.layout = "LAYOUT_16x9";
const slide = pptx.addSlide();

const worst = d.recompute_ms[d.recompute_ms.length - 1] / d.cached_ms[d.cached_ms.length - 1];
slide.addText(
  [{ text: "Recomputing an evicted prefix costs more the longer the context",
     options: { bold: true } }],
  { x: 0.4, y: 0.22, w: 9.2, h: 0.4, fontSize: 20, fontFace: FONT, color: INK });
slide.addText(
  `At ${d.lengths[d.lengths.length - 1] / 1000}k tokens a cold prefill takes ` +
  `${worst.toFixed(0)}x the TTFT of a cached prefix hit` +
  (hasTput
    ? `, and end-to-end throughput is ${(d.cached_tps[d.cached_tps.length - 1] /
        d.recompute_tps[d.recompute_tps.length - 1]).toFixed(1)}x lower.`
    : ".") +
  ` Output length is fixed at ${G} tokens, so decode time is constant and only prefill differs.`,
  { x: 0.4, y: 0.66, w: 9.2, h: 0.45, fontSize: 12, fontFace: FONT, color: MUTED });

for (const p of PANELS) {
  slide.addChart(
    pptx.ChartType.line,
    [{ name: "recompute", labels: cats, values: p.recompute.map((v) => Number(v.toFixed(1))) },
     { name: "cached", labels: cats, values: p.cached.map((v) => Number(v.toFixed(1))) }],
    {
      x: p.x, y: 1.25, w: PANELS.length > 1 ? 4.5 : 8.0, h: 3.8,
      chartColors: [C_RECOMP, C_CACHED],
      fill: "FFFFFF",
      title: p.title,
      showTitle: true,
      titleFontFace: FONT,
      titleFontSize: 13,
      titleColor: INK,
      showLegend: true,
      legendPos: "t",
      legendFontFace: FONT,
      legendFontSize: 11,
      lineSize: 2.5,
      lineDataSymbol: "circle",
      lineDataSymbolSize: 6,
      valAxisTitle: p.axis,
      showValAxisTitle: true,
      valAxisMinVal: 0,
      valAxisLabelFontFace: FONT,
      valAxisLabelFontSize: 10,
      valAxisTitleFontFace: FONT,
      valAxisTitleFontSize: 11,
      catAxisTitle: "context length (k tokens)",
      showCatAxisTitle: true,
      catAxisLabelFontFace: FONT,
      catAxisLabelFontSize: 10,
      catAxisTitleFontFace: FONT,
      catAxisTitleFontSize: 11,
      valGridLine: { color: GRID, size: 0.5 },
      catGridLine: { style: "none" },
    });
}

slide.addText(
  `${d.reps} reps per point · fixed ${G}-token output (ignore_eos) · ` +
  `x axis is log2: the lengths are powers of two, so even category spacing is the ` +
  `log2 axis of the PDF`,
  { x: 0.4, y: 5.15, w: 9.2, h: 0.3, fontSize: 9, fontFace: FONT, color: MUTED });

function verify(path) {
  const fail = (m) => { console.error(`[fatal] ${m}`); process.exit(3); };
  const names = execFileSync("unzip", ["-Z1", path], { encoding: "utf8" })
    .split("\n").filter((n) => n.startsWith("ppt/charts/chart"));
  if (names.length !== PANELS.length) {
    fail(`expected ${PANELS.length} charts, found ${names.length}`);
  }
  for (const n of names) {
    const xml = execFileSync("unzip", ["-p", path, n],
                             { encoding: "utf8", maxBuffer: 64 * 1024 * 1024 });
    if (!xml.includes("<c:lineChart>")) fail(`${n}: not a lineChart`);
    const series = [...xml.matchAll(/<c:tx>[\s\S]*?<c:v>([^<]*)<\/c:v>/g)].map((m) => m[1]);
    if (series.length !== 2) fail(`${n}: expected 2 series, found ${series.length}`);
    const pts = (xml.match(/<c:pt idx="/g) || []).length;
    // categories + both series
    if (pts < d.lengths.length * 3) {
      fail(`${n}: only ${pts} points for ${d.lengths.length} lengths x 2 series + labels`);
    }
  }
  console.log(`  [verified] ${names.length} lineChart(s), 2 series each, ` +
              `${d.lengths.length} categories`);
}

pptx.writeFile({ fileName: outPath }).then(() => {
  console.log(`[saved] ${outPath}`);
  console.log(`  panels: ${PANELS.map((p) => p.title).join(" | ")}`);
  verify(outPath);
});
