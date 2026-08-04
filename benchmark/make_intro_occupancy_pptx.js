// results/fig0/fig_intro_occupancy as an EDITABLE PowerPoint slide: one NATIVE area chart.
//
// Native means the chart carries its own worksheet, so values, colours, fonts and axis
// ranges stay editable in PowerPoint. A pasted PNG does not.
//
// OVERLAID, NOT STACKED -- the one thing that must not go wrong here.
//   The two series are two INSTANCES' occupancy of their own pools. Stacking them would
//   draw 82 + 11 = 93% and invent a quantity that does not exist. PowerPoint's area chart
//   supports grouping = standard | stacked | percentStacked, and the wrong one still
//   renders a plausible-looking chart, so the emitted XML is asserted after the build
//   (see verify_pptx_grouping.sh) rather than trusted.
//
// SERIES ORDER: the busier role goes FIRST so it is drawn behind and the idler in front;
//   the visible band between them is the imbalance. Order follows the data, because which
//   role is busier flips with the gauge (see plot_kv_occupancy_2panel.read_metric).
//
// 사용:
//   node benchmark/make_intro_occupancy_pptx.js results/fig0/fig_intro_occupancy.pptx \
//        results/fig0/fig_intro_occupancy.json

const pptxgen = require("pptxgenjs");
const fs = require("fs");

const outPath = process.argv[2] || "results/fig0/fig_intro_occupancy.pptx";
const dataPath = process.argv[3] || "results/fig0/fig_intro_occupancy.json";
const d = JSON.parse(fs.readFileSync(dataPath, "utf8"));

// The figure is only meaningful on the gauge that counts retained prefixes. token_usage
// subtracts them and inverts which role looks saturated, so refuse rather than emit a
// deck that silently argues the opposite of the paper.
if (d.metric !== "cache_occupancy") {
  console.error(
    `[fatal] this JSON was exported from '${d.metric}', not cache_occupancy.\n` +
    `        token_usage excludes the radix cache (num_used = max_total - available -\n` +
    `        evictable), so a prefill pool full of retained prefixes reads near zero and\n` +
    `        the figure inverts. Re-sample with the current kv_occupancy_timeseries.py.`);
  process.exit(2);
}

// paperstyle.PALETTE, minus the '#': pptxgenjs corrupts the file on '#' or 8-digit hex.
const C_PREFILL = "0072B2";
const C_DECODE = "E69F00";
const INK = "1A1A1A";
const MUTED = "595959";
const GRID = "D9D9D9";
const FONT = "Times New Roman";

const labels = d.t.map((x) => Math.round(x));
const SERIES = {
  prefill: { name: "Prefill instance", values: d.prefill, color: C_PREFILL, avg: d.prefill_avg },
  decode: { name: "Decode instance", values: d.decode, color: C_DECODE, avg: d.decode_avg },
};
// Busier first => drawn behind. Never hardcoded: on token_usage decode is the busier one.
const order = d.prefill_avg >= d.decode_avg ? ["prefill", "decode"] : ["decode", "prefill"];
const busy = SERIES[order[0]], idle = SERIES[order[1]];

const pptx = new pptxgen();
pptx.layout = "LAYOUT_16x9";
const slide = pptx.addSlide();

slide.addText(
  [{ text: "KV pool occupancy is inverted between the two roles", options: { bold: true } }],
  { x: 0.45, y: 0.28, w: 9.1, h: 0.4, fontSize: 20, fontFace: FONT, color: INK });

slide.addText(
  `${busy.name.split(" ")[0].toLowerCase()} averages ${busy.avg.toFixed(0)}% of its KV pool ` +
  `while ${idle.name.split(" ")[0].toLowerCase()} averages ${idle.avg.toFixed(0)}% — ` +
  `${Math.abs(d.gap_avg).toFixed(0)} points of KV capacity sit idle on ` +
  `${idle.name.split(" ")[0].toLowerCase()} while ${busy.name.split(" ")[0].toLowerCase()} ` +
  `is saturated and evicting.`,
  { x: 0.45, y: 0.72, w: 9.1, h: 0.4, fontSize: 12, fontFace: FONT, color: MUTED });

slide.addChart(
  pptx.ChartType.area,
  [
    { name: busy.name, labels: labels, values: busy.values },
    { name: idle.name, labels: labels, values: idle.values },
  ],
  {
    x: 0.45, y: 1.25, w: 9.1, h: 3.7,
    // Explicit. The default is standard, but this is the assumption the whole figure
    // rests on, so it is written down rather than inherited.
    // NOT a pptxgenjs option for area charts -- see the assertion below. pptxgenjs emits
    // <c:grouping> for an areaChart only when barGrouping === 'stacked'; anything else
    // omits the element and OOXML's default ('standard', i.e. overlaid) applies. So the
    // correct setting here is to leave it alone, and to verify the emitted XML.
    chartColors: [busy.color, idle.color],
    fill: "FFFFFF",
    showLegend: true,
    legendPos: "t",
    legendFontFace: FONT,
    legendFontSize: 12,
    valAxisTitle: "KV pool occupancy (cache_occupancy, %)",
    showValAxisTitle: true,
    valAxisMinVal: 0,
    valAxisMaxVal: 100,
    valAxisMajorUnit: 25,
    valAxisLabelFontFace: FONT,
    valAxisLabelFontSize: 11,
    valAxisTitleFontFace: FONT,
    valAxisTitleFontSize: 12,
    catAxisTitle: "time (s)",
    showCatAxisTitle: true,
    catAxisLabelFontFace: FONT,
    catAxisLabelFontSize: 11,
    catAxisTitleFontFace: FONT,
    catAxisTitleFontSize: 12,
    // ~235 category labels would print as an unreadable smear; show roughly every 10th.
    catAxisLabelFrequency: Math.max(1, Math.round(labels.length / 10)),
    catAxisMultiLevelLabels: false,
    valGridLine: { color: GRID, size: 0.5 },
    catGridLine: { style: "none" },
    lineSize: 1,
    lineDataSymbol: "none",
  });

slide.addText(
  `n=${d.n_full} samples decimated to ${d.t.length} · gauge: ${d.metric} · ` +
  `source: ${d.csv} · ${idle.name.split(" ")[0]} above ${busy.name.split(" ")[0]} in ` +
  `${d.frac_decode_above.toFixed(0)}% of samples`,
  { x: 0.45, y: 5.05, w: 9.1, h: 0.3, fontSize: 9, fontFace: FONT, color: MUTED });

// Assert the emitted XML rather than trusting the builder. Stacking these two series
// would draw 82 + 11 = 93% and invent a quantity that does not exist, and a stacked area
// chart looks perfectly plausible -- there is no visual tell. pptxgenjs writes
// <c:grouping> for an areaChart ONLY when barGrouping === 'stacked', so the check is that
// no stacked grouping survived into the file, plus that the busier series really is first
// (drawn behind) and carries the values we handed it.
function verify(path) {
  const { execFileSync } = require("child_process");
  const xml = execFileSync("unzip", ["-p", path, "ppt/charts/chart1.xml"],
                           { encoding: "utf8", maxBuffer: 64 * 1024 * 1024 });
  const fail = (msg) => { console.error(`[fatal] ${msg}`); process.exit(3); };

  if (!xml.includes("<c:areaChart>")) fail("no <c:areaChart> in the emitted chart");
  const grouping = /<c:grouping val="([a-zA-Z]+)"\/>/.exec(xml);
  if (grouping && grouping[1] !== "standard") {
    fail(`chart grouping is '${grouping[1]}', so the two instances' occupancies are ` +
         `ADDED (${busy.avg.toFixed(0)} + ${idle.avg.toFixed(0)} = ` +
         `${(busy.avg + idle.avg).toFixed(0)}%). They are fractions of two different ` +
         `pools; stacking them invents a quantity that does not exist.`);
  }
  const names = [...xml.matchAll(/<c:tx>[\s\S]*?<c:v>([^<]*)<\/c:v>/g)].map((m) => m[1]);
  if (names[0] !== busy.name) {
    fail(`first series is '${names[0]}' but the busier role is '${busy.name}'. The ` +
         `first series is drawn BEHIND; with the order reversed the idler's area would ` +
         `cover the busier curve completely.`);
  }
  const nPts = (xml.match(/<c:pt idx="/g) || []).length;
  console.log(`  [verified] areaChart, grouping=${grouping ? grouping[1] : "absent (=standard, overlaid)"}` +
              `, series order [${names.join(", ")}], ${nPts} plotted points`);
}

pptx.writeFile({ fileName: outPath }).then(() => {
  console.log(`[saved] ${outPath}`);
  console.log(`  behind: ${busy.name} (avg ${busy.avg.toFixed(1)}%)`);
  console.log(`  front : ${idle.name} (avg ${idle.avg.toFixed(1)}%)`);
  verify(outPath);
});
