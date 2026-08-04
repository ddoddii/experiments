// results/models/fig_peak_memory as an EDITABLE PowerPoint slide: one NATIVE stacked bar.
//
// STACKED and asserted, same as make_exp2_stack_pptx.js: prefill HBM / decode HBM / CPU
// DRAM are disjoint parts of one provisioned total and must add up. pptxgenjs always
// writes <c:grouping> for a bar chart but defaults it to 'clustered', which would draw
// three side-by-side bars and destroy the point, with no visual tell.
//
// THE SUBTITLE IS DERIVED, NOT WRITTEN. The sign of the total flips across these models
// -- Ours is lower on the 8B and higher on both larger ones -- so any fixed "Ours uses
// less memory" claim would be false on four of the six bars in this very chart. The
// wording is computed from the data every time it is built.
//
// 사용:
//   python benchmark/plot_peak_memory.py --out results/models/fig_peak_memory \
//       --export-json results/models/fig_peak_memory.json
//   node benchmark/make_peak_memory_pptx.js results/models/fig_peak_memory.pptx \
//        results/models/fig_peak_memory.json

const pptxgen = require("pptxgenjs");
const fs = require("fs");
const { execFileSync } = require("child_process");

const outPath = process.argv[2] || "results/models/fig_peak_memory.pptx";
const dataPath = process.argv[3] || "results/models/fig_peak_memory.json";
const d = JSON.parse(fs.readFileSync(dataPath, "utf8"));

const C_P = "BFC4C9";
const C_D = "009E73";
const C_DRAM = "FFFFFF";
const INK = "1A1A1A";
const MUTED = "595959";
const FONT = "Times New Roman";

const ARMS = Object.keys(d.arm_labels);          // ["hicache", "park"]
const cats = [];
const rows = [];
for (const m of d.models) {
  for (const a of ARMS) {
    cats.push(`${d.arm_labels[a]}\n${m.label}`);
    rows.push({ model: m.label, arm: a, ...m.arms[a] });
  }
}
const SERIES = [
  { name: "Prefill HBM", key: "prefill", color: C_P },
  { name: "Decode HBM", key: "decode", color: C_D },
  { name: "CPU DRAM", key: "dram", color: C_DRAM },
];

const tot = (r) => r.prefill + r.decode + r.dram;
const deltas = d.models.map((m) => ({
  label: m.label,
  d: tot(m.arms.park) - tot(m.arms.hicache),
  dram: m.arms.park.dram - m.arms.hicache.dram,
  hbm: (m.arms.park.prefill + m.arms.park.decode)
       - (m.arms.hicache.prefill + m.arms.hicache.decode),
  disk: m.arms.hicache.disk,
}));
const lower = deltas.filter((x) => x.d < 0).map((x) => x.label);
const verdict =
  lower.length === deltas.length ? `lower on all ${deltas.length} models`
  : lower.length === 0 ? `higher on all ${deltas.length} models`
  : `lower only on ${lower.join(", ")}`;

console.log(`  ${rows.length} bars from ${d.models.length} models x ${ARMS.length} arms`);
for (const x of deltas) {
  console.log(`    ${x.label.padEnd(14)} total ${x.d >= 0 ? "+" : ""}${x.d.toFixed(1)} GB` +
              `  (DRAM ${x.dram.toFixed(1)}, HBM +${x.hbm.toFixed(1)}, ` +
              `hicache disk ${x.disk.toFixed(0)} GB)`);
}

const pptx = new pptxgen();
pptx.layout = "LAYOUT_16x9";
const slide = pptx.addSlide();

slide.addText(
  [{ text: "GPU-first placement trades host memory for GPU memory",
     options: { bold: true } }],
  { x: 0.45, y: 0.24, w: 9.1, h: 0.4, fontSize: 20, fontFace: FONT, color: INK });
slide.addText(
  `Returning hicache's host tier saves ` +
  `${Math.abs(Math.max(...deltas.map((x) => x.dram))).toFixed(1)}–` +
  `${Math.abs(Math.min(...deltas.map((x) => x.dram))).toFixed(1)} GB of DRAM, ` +
  `paid for in prefill HBM (+${Math.min(...deltas.map((x) => x.hbm)).toFixed(1)} to ` +
  `+${Math.max(...deltas.map((x) => x.hbm)).toFixed(1)} GB). Total provisioned memory is ` +
  `${verdict}: the trade is favourable only while HBM headroom is large.`,
  { x: 0.45, y: 0.68, w: 9.1, h: 0.5, fontSize: 12, fontFace: FONT, color: MUTED });

slide.addChart(
  pptx.ChartType.bar,
  SERIES.map((s) => ({ name: s.name, labels: cats,
                       values: rows.map((r) => Number(r[s.key].toFixed(2))) })),
  {
    x: 0.7, y: 1.35, w: 8.6, h: 3.7,
    barDir: "col",
    barGrouping: "stacked",     // REQUIRED; asserted below
    barGapWidthPct: 55,
    chartColors: SERIES.map((s) => s.color),
    fill: "FFFFFF",
    showLegend: true,
    legendPos: "t",
    legendFontFace: FONT,
    legendFontSize: 12,
    valAxisTitle: "Peak memory usage (GB)",
    showValAxisTitle: true,
    valAxisMinVal: 0,
    valAxisLabelFontFace: FONT,
    valAxisLabelFontSize: 11,
    valAxisTitleFontFace: FONT,
    valAxisTitleFontSize: 12,
    catAxisLabelFontFace: FONT,
    catAxisLabelFontSize: 10,
    valGridLine: { color: "D9D9D9", size: 0.5 },
    catGridLine: { style: "none" },
    dataBorder: { pt: 0.5, color: INK },
  });

slide.addText(
  `2P2D · prefill HBM = GPU${d.prefill_gpus.join("+")}, decode HBM = ` +
  `GPU${d.decode_gpus.join("+")}, CPU DRAM = ${d.dram_column} (PSS: shared pages ` +
  `charged once) · hicache also writes ` +
  `${deltas.map((x) => x.disk.toFixed(0)).join("/")} GB to its L3 tier on disk, ` +
  `which Ours does not and which is not in these bars`,
  { x: 0.45, y: 5.12, w: 9.1, h: 0.4, fontSize: 9, fontFace: FONT, color: MUTED });

function verify(path) {
  const fail = (m) => { console.error(`[fatal] ${m}`); process.exit(3); };
  const xml = execFileSync("unzip", ["-p", path, "ppt/charts/chart1.xml"],
                           { encoding: "utf8", maxBuffer: 64 * 1024 * 1024 });
  if (!xml.includes("<c:barChart>")) fail("no <c:barChart> in the emitted chart");
  const g = /<c:grouping val="([a-zA-Z]+)"\/>/.exec(xml);
  if (!g) fail("no <c:grouping> element -- cannot confirm the bars are stacked");
  if (g[1] !== "stacked") {
    fail(`chart grouping is '${g[1]}', not 'stacked'. Prefill HBM, decode HBM and CPU ` +
         `DRAM are disjoint parts of one provisioned total; side by side they no longer ` +
         `sum to it and the figure stops being about total memory.`);
  }
  const names = [...xml.matchAll(/<c:tx>[\s\S]*?<c:v>([^<]*)<\/c:v>/g)].map((m) => m[1]);
  if (names.length !== SERIES.length) {
    fail(`expected ${SERIES.length} series, found ${names.length}: ${names}`);
  }
  // Every model must contribute both arms, or a pair is silently half-drawn.
  if (rows.length !== d.models.length * ARMS.length) {
    fail(`expected ${d.models.length * ARMS.length} bars, built ${rows.length}`);
  }
  console.log(`  [verified] barChart, grouping=${g[1]}, series [${names.join(", ")}], ` +
              `${cats.length} bars`);
}

pptx.writeFile({ fileName: outPath }).then(() => {
  console.log(`[saved] ${outPath}`);
  verify(outPath);
});
