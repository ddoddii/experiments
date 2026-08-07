// results/qps/<tag>/qps.json as an EDITABLE PowerPoint slide -- native scatter charts,
// same idea as make_qps_sweep_pptx.js but against collect_qps.py's schema (arms keyed
// recompute/hicache/park, fields rate/throughput_tok_s/ttft_p50_s/..., not the older
// qps_sweep.py schema that script was built for -- field names and units differ, so
// this is a separate generator rather than an extension of that one).
//
// Native means each chart carries its own worksheet: values, colours, fonts and axis
// ranges stay editable in PowerPoint, unlike a pasted PNG.
//
// ONLY (a) Throughput-vs-rate and (b) TTFT-vs-rate are done as native charts, because
// both share ONE x series (the offered rate list is the same for every arm -- checked
// below, same guard as make_qps_sweep_pptx.js). plot_qps.py's panel (c), TTFT vs
// DELIVERED throughput, does NOT share an x series across arms (each arm delivers a
// different throughput at the same offered rate) -- pptxgenjs's scatter chart data model
// (OptsChartData) has one shared `values` x-array feeding every series, with no
// per-series x, so that panel can't be a native chart here. It ships as the existing
// rendered PNG (fig_qps.png, already produced by plot_qps.py) on a second slide instead
// of being faked as something it isn't.
//
// 사용:
//   node benchmark/make_qps_pptx.js results/qps/qps_llama8b_bfcl/qps.pptx \
//        results/qps/qps_llama8b_bfcl/qps.json
//   node benchmark/make_qps_pptx.js out.pptx qps.json --ttft p95 --title "..." --fig path/to/fig_qps.png

const pptxgen = require("pptxgenjs");
const fs = require("fs");
const path = require("path");

const args = process.argv.slice(2);
const pos = args.filter((a) => !a.startsWith("--"));
const outPath = pos[0] || "fig_qps.pptx";
const dataPath = pos[1] || "results/qps/fig_qps/qps.json";

function flagArg(name, fallback) {
  const i = args.indexOf(`--${name}`);
  return i >= 0 && args[i + 1] ? args[i + 1] : fallback;
}
const ttftKey = `ttft_${flagArg("ttft", "p50")}_s`;
const dataDir = path.dirname(dataPath);
const figPath = flagArg("fig", path.join(dataDir, "fig_qps.png"));
const title = flagArg(
  "title",
  "Prefix reuse holds latency down until the server saturates"
);

const raw = JSON.parse(fs.readFileSync(dataPath, "utf8"));
const arms = raw.arms || raw;

// paperstyle.PALETTE, minus '#' (pptxgenjs corrupts the file on '#' or 8-digit hex).
const COLORS = { recompute: "B4423C", hicache: "0072B2", park: "009E73" };
const LABELS = { recompute: "Recompute", hicache: "SGLang", park: "Ours" };
const INK = "1A1A1A";
const MUTED = "595959";
const GRID = "D9D9D9";
const FONT = "Times New Roman";

const CANON = ["recompute", "hicache", "park"];
const names = CANON.filter((n) => arms[n]).concat(
  Object.keys(arms).filter((n) => !CANON.includes(n))
);
if (!names.length) {
  console.error(`[fatal] no arms in ${dataPath}`);
  process.exit(1);
}

const xs = arms[names[0]].map((r) => r.rate);
for (const n of names) {
  const other = arms[n].map((r) => r.rate);
  if (other.join() !== xs.join()) {
    console.error(
      `[fatal] ${n} has offered rates ${other} but ${names[0]} has ${xs}. A scatter ` +
        `chart shares ONE x series, so mismatched rates would silently pair each y with ` +
        `the wrong x. Re-run collect_qps.py against a sweep where every arm used the ` +
        `same RATES list.`
    );
    process.exit(2);
  }
}

// Points past saturation (server didn't drain the offered load inside the window) are
// real measurements, not something to hide -- but a mean/percentile there is queueing
// delay, not converged latency, and worth stating rather than leaving as an unexplained
// kink in the curve.
const satNote = names
  .map((n) => {
    const bad = arms[n].filter((r) => r.past_saturation);
    if (!bad.length) return null;
    return `${n}: saturated from rate ${bad[0].rate} (${bad
      .map((r) => `${r.unfinished}/${r.launched} unfinished @ ${r.rate}`)
      .join(", ")})`;
  })
  .filter(Boolean)
  .join(" · ");
const wrapNote = names
  .map((n) => {
    const bad = arms[n].filter((r) => r.wrapped);
    if (!bad.length) return null;
    return `${n}: corpus wrapped at rate ${bad.map((r) => r.rate).join(", ")}`;
  })
  .filter(Boolean)
  .join(" · ");

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.3 x 7.5in -- set BEFORE addSlide
pres.author = "SGLang KV placement experiment";
pres.title = title;

// ---------------------------------------------------------------- slide 1: native charts
const s1 = pres.addSlide();
s1.addText(title, {
  x: 0.45, y: 0.22, w: 12.4, h: 0.45,
  fontFace: FONT, fontSize: 24, bold: true, color: INK, margin: 0,
});
s1.addText(
  `open-loop offered rate ${xs.join(", ")} sessions/s · ${
    raw.workload || path.basename(dataDir)
  }`,
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
  catAxisTitle: "offered rate (sessions/s)",
  showTitle: true,
  titleFontFace: FONT,
  titleFontSize: 13,
  titleColor: INK,
  chartColors: names.map((n) => COLORS[n] || "555555"),
};

const PANELS = [
  { key: "throughput_tok_s", title: "(a)  Throughput vs offered rate", axis: "delivered tok/s" },
  { key: ttftKey, title: `(b)  Latency vs offered rate`, axis: `${flagArg("ttft", "p50")} TTFT (s)` },
];

const X0 = 1.9, W = 4.6, GAP = 0.5, Y = 1.3, H = 4.5;
PANELS.forEach((p, i) => {
  const data = [{ name: "rate", values: xs }].concat(
    names.map((n) => ({ name: LABELS[n] || n, values: arms[n].map((r) => r[p.key] ?? null) }))
  );
  s1.addChart(pres.ChartType.scatter, data, {
    ...common,
    x: X0 + i * (W + GAP), y: Y, w: W, h: H,
    title: p.title,
    valAxisTitle: p.axis,
  });
});

let lx = 5.0;
names.forEach((n) => {
  s1.addShape(pres.ShapeType.line, {
    x: lx, y: 5.95, w: 0.42, h: 0,
    line: { color: COLORS[n] || "555555", width: 2.25 },
  });
  s1.addText(LABELS[n] || n, {
    x: lx + 0.5, y: 5.77, w: 1.4, h: 0.4,
    fontFace: FONT, fontSize: 12, color: INK, margin: 0, valign: "middle",
  });
  lx += 1.9;
});

const noteLines = [satNote && `Saturated — ${satNote}.`, wrapNote && `Wrapped — ${wrapNote}.`]
  .filter(Boolean)
  .join(" ");
s1.addText(
  (noteLines ? noteLines + " " : "") +
    "Panel (c) of the source figure (latency vs. DELIVERED throughput) isn't shown here " +
    "as a native chart -- each arm delivers a different throughput at the same offered " +
    "rate, so it has no single shared x-axis. See the next slide for the full 3-panel " +
    "figure as rendered.",
  { x: 0.45, y: 6.35, w: 12.4, h: 0.85,
    fontFace: FONT, fontSize: 11, color: MUTED, margin: 0 }
);

s1.addNotes(
  "Both charts are native PowerPoint charts: right-click > Edit Data for values, or the " +
    "Chart Design / Format tabs for colours and axes.\n" +
    `TTFT panel uses ${flagArg("ttft", "p50")} (pass --ttft p95 to regenerate with the tail instead).\n` +
    "Points past saturation are real measurements but their latency is a still-growing " +
    "queue, not a converged number -- see the caption.\n" +
    `Source: ${dataPath}`
);

// ---------------------------------------------------------------- slide 2: full rendered figure
if (fs.existsSync(figPath)) {
  const s2 = pres.addSlide();
  s2.addText("Full figure, as rendered (plot_qps.py)", {
    x: 0.45, y: 0.22, w: 12.4, h: 0.4,
    fontFace: FONT, fontSize: 18, bold: true, color: INK, margin: 0,
  });
  s2.addImage({ path: figPath, x: 0.4, y: 0.85, w: 12.5, h: 12.5 * (2.1 / 7.16) });
  s2.addText(`Source: ${figPath}`, {
    x: 0.45, y: 6.9, w: 12.4, h: 0.3,
    fontFace: FONT, fontSize: 10, color: MUTED, italic: true, margin: 0,
  });
} else {
  console.error(`[warn] ${figPath} not found -- skipping the full-figure slide. Pass ` +
                `--fig <path> if plot_qps.py wrote it somewhere else.`);
}

pres.writeFile({ fileName: outPath }).then(() => console.log("[saved] " + outPath));
