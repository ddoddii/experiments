// fig_exp2_stack_*_ours as an EDITABLE PowerPoint slide: one NATIVE stacked bar chart.
//
// STACKED, AND THIS ONE REALLY IS -- the opposite of the occupancy overlay next door.
//   Here the three series are disjoint parts of ONE pool (own cache + parked + unused =
//   the GPU's KV pool), so they must add up. pptxgenjs always writes <c:grouping> for a
//   bar chart but defaults it to 'clustered', which would draw three side-by-side bars
//   and destroy the figure, so barGrouping must be set and then verified in the emitted
//   XML. (In make_intro_occupancy_pptx.js the check runs the other way: those two series
//   are fractions of two DIFFERENT pools and stacking them would invent a quantity.)
//
// The bar heights also carry a second fact: the pool total differs per GPU, because a
// park pool is carved from the same HBM the serving pool would have taken. Do not
// normalise the bars to 100% -- that would hide it.
//
// 사용:
//   python benchmark/plot_exp2_gpu_stack.py --dirs results/exp2/big_c32_r1 --only ours \
//       --out results/exp2/fig_exp2_stack_big_ours \
//       --export-json results/exp2/fig_exp2_stack_big_ours.json
//   node benchmark/make_exp2_stack_pptx.js results/exp2/fig_exp2_stack_big_ours.pptx \
//       results/exp2/fig_exp2_stack_big_ours.json

const pptxgen = require("pptxgenjs");
const fs = require("fs");
const { execFileSync } = require("child_process");

const outPath = process.argv[2] || "results/exp2/fig_exp2_stack_big_ours.pptx";
const dataPath = process.argv[3] || "results/exp2/fig_exp2_stack_big_ours.json";
const d = JSON.parse(fs.readFileSync(dataPath, "utf8"));

const arm = d.only === "local" ? "local" : "ours";
const rows = d.arms[arm];
if (!rows || !rows.length) {
  console.error(`[fatal] no bars for arm '${arm}' in ${dataPath}`);
  process.exit(2);
}

// paperstyle, minus the '#': pptxgenjs corrupts the file on '#' or 8-digit hex.
const C_OWN = "BFC4C9";
const C_PARK = "009E73";
const C_UNUSED = "FFFFFF";
const INK = "1A1A1A";
const MUTED = "595959";
const FONT = "Times New Roman";

const labels = rows.map((r) => {
  const role = r.role === "P" ? "Prefill" : "Decode";
  if (r.members) return `${r.label}\n(mean of ${r.members.length} GPUs)`;
  return `${r.label.replace(/^([PD])(\d)$/, "GPU$2")}\n(${role})`;
});
const SERIES = [
  { name: "own cache", key: "own", color: C_OWN },
  { name: arm === "ours" ? "parked on an idle GPU" : "parked (onto its own GPU)",
    key: "parked", color: C_PARK },
  { name: "unused", key: "unused", color: C_UNUSED },
];

const total = rows.reduce((a, r) => a + r.own + r.parked + r.unused, 0);
const onIdle = rows
  .filter((r) => r.role === "D")
  .reduce((a, r) => a + r.parked * (r.members ? r.members.length : 1), 0);
console.log(`  arm=${arm}  bars=${rows.length}  n_runs=${d.n_runs}`);
rows.forEach((r) => console.log(
  `    ${r.label.padEnd(8)} own ${r.own.toFixed(2)} + parked ${r.parked.toFixed(2)}` +
  ` + unused ${r.unused.toFixed(2)} = ${(r.own + r.parked + r.unused).toFixed(2)} GB`));

const pptx = new pptxgen();
pptx.layout = "LAYOUT_16x9";
const slide = pptx.addSlide();

slide.addText(
  [{ text: "Reusable KV lands on the GPUs whose pool was empty", options: { bold: true } }],
  { x: 0.45, y: 0.28, w: 9.1, h: 0.4, fontSize: 20, fontFace: FONT, color: INK });
slide.addText(
  `${onIdle.toFixed(1)} GB of reusable KV sits on the idle decode GPUs, in pool space ` +
  `their own serving work never used. Bar height is each GPU's KV pool; a park pool is ` +
  `carved from the same HBM the serving pool would have taken.`,
  { x: 0.45, y: 0.72, w: 9.1, h: 0.45, fontSize: 12, fontFace: FONT, color: MUTED });

slide.addChart(
  pptx.ChartType.bar,
  SERIES.map((s) => ({ name: s.name, labels: labels,
                       values: rows.map((r) => Number(r[s.key].toFixed(3))) })),
  {
    x: 0.6, y: 1.3, w: 8.8, h: 3.75,
    barDir: "col",
    // REQUIRED. Without it pptxgenjs writes grouping="clustered" and the three parts of
    // one pool are drawn side by side instead of stacked. Asserted after the build.
    barGrouping: "stacked",
    barGapWidthPct: 60,
    chartColors: SERIES.map((s) => s.color),
    chartColorsOpacity: 100,
    fill: "FFFFFF",
    showLegend: true,
    legendPos: "t",
    legendFontFace: FONT,
    legendFontSize: 12,
    valAxisTitle: "KV pool (GB)",
    showValAxisTitle: true,
    valAxisMinVal: 0,
    valAxisLabelFontFace: FONT,
    valAxisLabelFontSize: 11,
    valAxisTitleFontFace: FONT,
    valAxisTitleFontSize: 12,
    catAxisLabelFontFace: FONT,
    catAxisLabelFontSize: 11,
    valGridLine: { color: "D9D9D9", size: 0.5 },
    catGridLine: { style: "none" },
    dataBorder: { pt: 0.5, color: INK },
    showValue: false,
  });

slide.addText(
  `${arm === "ours" ? "GPU-first placement" : "local-only parking"} · ` +
  `${d.n_runs === 1 ? "single run (n=1)" : `mean of ${d.n_runs} runs`} · ` +
  `${d.dirs.join(", ")} · park-fetch hit rate ${d.hit[arm].toFixed(1)}%`,
  { x: 0.45, y: 5.12, w: 9.1, h: 0.3, fontSize: 9, fontFace: FONT, color: MUTED });

function verify(path) {
  const xml = execFileSync("unzip", ["-p", path, "ppt/charts/chart1.xml"],
                           { encoding: "utf8", maxBuffer: 64 * 1024 * 1024 });
  const fail = (m) => { console.error(`[fatal] ${m}`); process.exit(3); };
  if (!xml.includes("<c:barChart>")) fail("no <c:barChart> in the emitted chart");
  const g = /<c:grouping val="([a-zA-Z]+)"\/>/.exec(xml);
  if (!g) fail("no <c:grouping> element -- cannot confirm the bars are stacked");
  if (g[1] !== "stacked") {
    fail(`chart grouping is '${g[1]}', not 'stacked'. own/parked/unused are disjoint ` +
         `parts of ONE pool; drawn side by side the bars no longer sum to the pool and ` +
         `the empty space the parked KV moved into disappears from the figure.`);
  }
  const names = [...xml.matchAll(/<c:tx>[\s\S]*?<c:v>([^<]*)<\/c:v>/g)].map((m) => m[1]);
  if (names.length !== SERIES.length) {
    fail(`expected ${SERIES.length} series, found ${names.length}: ${names}`);
  }
  console.log(`  [verified] barChart, grouping=${g[1]}, series [${names.join(", ")}]`);
}

pptx.writeFile({ fileName: outPath }).then(() => {
  console.log(`[saved] ${outPath}`);
  verify(outPath);
});
