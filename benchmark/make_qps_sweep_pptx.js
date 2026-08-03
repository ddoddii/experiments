// results/intro/fig_qps_sweep as an EDITABLE PowerPoint slide: three NATIVE scatter charts.
//
// Native means each chart carries its own worksheet, so values, colours, fonts and axis
// ranges stay editable in PowerPoint. A pasted PNG does not, and "let me tweak it in the
// deck" is the whole reason this exists.
//
// SCATTER, not line. The QPS points are 2,4,8,12,16,24,32,48,64 -- unevenly spaced. A
// pptxgenjs line chart treats x values as CATEGORIES and lays them out at equal
// intervals, which would stretch 2->4 to the same width as 48->64 and flatten the knee
// that the figure exists to show. Scatter keeps the axis numeric, matching the PDF.
//
// 사용:
//   node benchmark/make_qps_sweep_pptx.js results/intro/fig_qps_sweep.pptx \
//        results/intro/fig_qps_sweep.json

const pptxgen = require("pptxgenjs");
const fs = require("fs");

const outPath = process.argv[2] || "fig_qps_sweep.pptx";
const dataPath = process.argv[3] || "results/intro/fig_qps_sweep.json";
const raw = JSON.parse(fs.readFileSync(dataPath, "utf8"));
const arms = raw.arms || raw;

// paperstyle.PALETTE, minus the '#': pptxgenjs corrupts the file on '#' or 8-digit hex.
const COLORS = { cached: "009E73", recompute: "B4423C" };
const INK = "1A1A1A";
const MUTED = "595959";
const GRID = "D9D9D9";
const FONT = "Times New Roman";

const names = Object.keys(arms);
const xs = arms[names[0]].map((r) => r.qps);
for (const n of names) {
  const other = arms[n].map((r) => r.qps);
  if (other.join() !== xs.join()) {
    console.error(`[fatal] ${n} has QPS points ${other} but ${names[0]} has ${xs}. A ` +
                  `scatter chart shares ONE x series, so mismatched rates would silently ` +
                  `pair each y with the wrong x.`);
    process.exit(2);
  }
}

const PANELS = [
  { key: "ttft_ms", title: "(a)  TTFT vs QPS", axis: "mean TTFT (ms)" },
  { key: "tpot_ms", title: "(b)  TPOT vs QPS", axis: "mean TPOT (ms/token)" },
  { key: "throughput_tok_s", title: "(c)  Throughput vs QPS", axis: "total throughput (tok/s)" },
];

// Requests that never completed. Past the point where these appear, a mean over the
// SURVIVORS is a biased statistic -- the slowest requests are the ones missing -- and the
// TTFT curve flattens because it has hit the client timeout, not because the server
// stopped degrading. Worth stating on the slide rather than leaving as a flat tail that
// reads like the system recovered.
const failNote = names
  .map((n) => {
    const bad = arms[n].filter((r) => (r.n_fail || 0) > 0);
    if (!bad.length) return null;
    const worst = bad[bad.length - 1];
    const pct = (100 * worst.n_fail) / (worst.n_ok + worst.n_fail);
    return `${n}: unfinished requests appear from ${bad[0].qps} QPS ` +
           `(${pct.toFixed(0)}% by ${worst.qps} QPS)`;
  })
  .filter(Boolean)
  .join(" · ");

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.3 x 7.5 in -- set BEFORE addSlide
pres.author = "SGLang KV placement experiment";
pres.title = "Prefix reuse under increasing offered load";

const slide = pres.addSlide();

slide.addText("Prefix reuse holds latency down until the server saturates", {
  x: 0.45, y: 0.22, w: 12.4, h: 0.45,
  fontFace: FONT, fontSize: 24, bold: true, color: INK, margin: 0,
});
slide.addText(
  `context ${raw.ctx_len ?? "?"} tokens · ${raw.max_new_tokens ?? "?"} new tokens · ` +
  `${raw.duration ?? "?"} s per point · SGLang 2P2D · 4× RTX A6000`,
  { x: 0.45, y: 0.68, w: 12.4, h: 0.3,
    fontFace: FONT, fontSize: 11, color: MUTED, italic: true, margin: 0 }
);

const common = {
  showLegend: false,
  lineSize: 2,
  lineDataSymbolSize: 6,
  catAxisLabelColor: MUTED,
  valAxisLabelColor: MUTED,
  catAxisLabelFontFace: FONT,
  valAxisLabelFontFace: FONT,
  catAxisLabelFontSize: 10,
  valAxisLabelFontSize: 10,
  catAxisTitleFontFace: FONT,
  valAxisTitleFontFace: FONT,
  catAxisTitleFontSize: 11,
  valAxisTitleFontSize: 11,
  axisLineColor: GRID,
  valGridLine: { color: GRID, size: 0.75 },
  catGridLine: { color: GRID, size: 0.75 },
  valAxisMinVal: 0,
  catAxisMinVal: 0,
  showValAxisTitle: true,
  showCatAxisTitle: true,
  catAxisTitle: "QPS (requested rate)",
  showTitle: true,
  titleFontFace: FONT,
  titleFontSize: 13,
  titleColor: INK,
  chartColors: names.map((n) => COLORS[n] || "555555"),
};

const X0 = 0.7, W = 3.85, GAP = 0.25, Y = 1.15, H = 4.3;

PANELS.forEach((p, i) => {
  // pptxgenjs scatter: entry 0 is the shared X series, the rest are Y series.
  const data = [{ name: "QPS", values: xs }].concat(
    names.map((n) => ({ name: n, values: arms[n].map((r) => r[p.key] ?? null) }))
  );
  slide.addChart(pres.ChartType.scatter, data, {
    ...common,
    x: X0 + i * (W + GAP), y: Y, w: W, h: H,
    title: p.title,
    valAxisTitle: p.axis,
  });
});

let lx = 4.9;
names.forEach((n) => {
  slide.addShape(pres.ShapeType.line, {
    x: lx, y: 5.78, w: 0.42, h: 0,
    line: { color: COLORS[n] || "555555", width: 2.25 },
  });
  slide.addText(n, {
    x: lx + 0.5, y: 5.6, w: 1.4, h: 0.4,
    fontFace: FONT, fontSize: 12, color: INK, margin: 0, valign: "middle",
  });
  lx += 1.9;
});

slide.addText(
  (failNote ? `Note — ${failNote}. ` : "") +
  "Beyond those rates panel (a) is bounded by the client timeout rather than by the " +
  "server, and every mean is taken over the requests that finished, so the flat tails " +
  "understate how far behind the system actually is.",
  { x: 0.45, y: 6.2, w: 12.4, h: 0.8,
    fontFace: FONT, fontSize: 11, color: MUTED, margin: 0 }
);

slide.addNotes(
  "All three are native PowerPoint charts: right-click > Edit Data for values, or the " +
  "Chart Design / Format tabs for colours and axes.\n" +
  "Scatter rather than line, because the QPS points are unevenly spaced (2,4,8,...,64) " +
  "and a line chart would place them at equal intervals, flattening the knee.\n" +
  "The TTFT tail on recompute sits at the client timeout, not at a measured latency, " +
  "and by 64 QPS most of its requests never finished -- see the caption.\n" +
  `Source: ${dataPath}`
);

pres.writeFile({ fileName: outPath }).then(() => console.log("[saved] " + outPath));
