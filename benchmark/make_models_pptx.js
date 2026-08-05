// The cross-model figure as an EDITABLE PowerPoint slide: three NATIVE bar charts.
//
// Native means each chart carries its own worksheet, so numbers, colours, fonts and axis
// ranges are all editable in PowerPoint. A PNG placed on a slide is not -- and "I want to
// tweak it in PPT" is the whole reason this file exists.
//
// Reads the same table.json files as plot_models.py, and derives the same quantities the
// same way, so the slide and the camera-ready PDF cannot drift apart:
//   - TTFT is NORMALISED to Recompute per model. Absolute TTFT is dominated by model size
//     rather than by the mechanism, so an absolute axis invites reading across models,
//     which says nothing. A dashed line series pins the 1.0 baseline.
//   - Host DRAM's ratio label is against SGLang, not the best arm: Recompute has no host
//     tier at all, so it wins that metric by definition and the label would read ~1.0x as
//     if the proposal saved nothing.
//
// 사용:
//   node benchmark/make_models_pptx.js out.pptx Llama-3.1-8B=results/models/llama8b/table.json ...

const pptxgen = require("pptxgenjs");
const fs = require("fs");

const outPath = process.argv[2] || "fig_models.pptx";
const specs = process.argv.slice(3);
if (!specs.length) {
  console.error("usage: node make_models_pptx.js <out.pptx> Label=table.json [...]");
  process.exit(2);
}

// Recompute is the do-nothing control, so it is neutral grey: it is the reference the
// other two are read against, not a competing design. Blue/green match the PDF figure.
// No leading '#': pptxgenjs corrupts the file on '#' or 8-digit hex.
const ARMS = [
  { key: "recompute", label: "Recompute", color: "BFC4C9" },
  { key: "hicache", label: "SGLang", color: "0072B2" },
  { key: "park", label: "Ours", color: "009E73" },
];
const INK = "1A1A1A";
const MUTED = "595959";
const GRID = "D9D9D9";
const FONT = "Times New Roman";

const models = specs.map((s) => {
  const i = s.indexOf("=");
  const label = i < 0 ? s : s.slice(0, i);
  const path = i < 0 ? s : s.slice(i + 1);
  let byArm = {};
  try {
    for (const r of JSON.parse(fs.readFileSync(path, "utf8"))) byArm[r.arm] = r;
  } catch (e) {
    console.error(`[warn] ${path}: ${e.message} -- ${label} will be blank`);
  }
  return { label, byArm };
});
const names = models.map((m) => m.label);

const num = (m, arm, key) => {
  const v = (m.byArm[arm] || {})[key];
  return typeof v === "number" ? v : null;
};

// key, axis title, lower-is-better, ratio denominator arm, normalise-to arm
const PANELS = [
  { key: "cache_hit_rate_pct", title: "(a)  Cache hit rate", axis: "Cache Hit Rate (%)",
    lower: false, max: 100, unit: 25 },
  { key: "ttft_p50_s", title: "(b)  Time to first token", axis: "Normalized TTFT",
    lower: true, normTo: "recompute", max: 1.25, unit: 0.25 },
  // Throughput is deliberately NOT a panel here. Closed-loop it is flat by
  // construction, and the open-loop sweep showed the server saturates on decode, which
  // KV placement does not touch (peak 1105/1249/1135 tok/s across arms). Neither
  // version separates the arms, so neither earns a quarter of the figure.
  { key: "peak_anonpages_gb", title: "(c)  Host memory", axis: "Host DRAM (GB)",
    lower: true, ratioVs: "hicache" },
];

function seriesFor(p) {
  return ARMS.map((a) => ({
    name: a.label,
    labels: names,
    values: models.map((m) => {
      const v = num(m, a.key, p.key);
      if (v === null) return 0;
      if (!p.normTo) return Math.round(v * 100) / 100;
      const d = num(m, p.normTo, p.key);
      return d ? Math.round((v / d) * 1000) / 1000 : 0;
    }),
  }));
}

// Improvement of Ours over the better baseline, per model -- the same rule the PDF uses.
function ratios(p) {
  return models.map((m) => {
    const ours = num(m, "park", p.key);
    if (!ours) return null;
    let base;
    if (p.ratioVs) base = num(m, p.ratioVs, p.key);
    else {
      const bs = ARMS.filter((a) => a.key !== "park")
        .map((a) => num(m, a.key, p.key)).filter((v) => v !== null);
      if (!bs.length) return null;
      base = p.lower ? Math.min(...bs) : Math.max(...bs);
    }
    if (!base) return null;
    return p.lower ? base / ours : ours / base;
  });
}

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE";           // 13.3 x 7.5 in -- set BEFORE addSlide
pres.author = "SGLang KV placement experiment";
pres.title = "GPU-first unified KV placement across models";

const slide = pres.addSlide();

slide.addText("GPU-first KV placement holds across model families", {
  x: 0.45, y: 0.22, w: 12.4, h: 0.45,
  fontFace: FONT, fontSize: 24, bold: true, color: INK, margin: 0,
});
slide.addText(
  "ShareGPT multi-turn · SGLang 2P2D · 4× RTX A6000 · " +
  "labels are Ours vs. the stronger baseline (host DRAM vs. SGLang, which is the only " +
  "baseline with a host tier)",
  { x: 0.45, y: 0.68, w: 12.4, h: 0.3,
    fontFace: FONT, fontSize: 11, color: MUTED, italic: true, margin: 0 }
);

// Shared frame styling. Quiet axes, horizontal-only grid, bars touching inside a group
// (barOverlapPct 0) with wide separation between groups -- one visual unit per model.
const common = {
  barDir: "col",
  barGrouping: "clustered",
  barOverlapPct: 0,
  barGapWidthPct: 90,
  showLegend: false,
  chartColors: ARMS.map((a) => a.color),
  catAxisLabelColor: MUTED,
  valAxisLabelColor: MUTED,
  catAxisLabelFontFace: FONT,
  valAxisLabelFontFace: FONT,
  catAxisLabelFontSize: 10,
  valAxisLabelFontSize: 10,
  catAxisTitleFontFace: FONT,
  valAxisTitleFontFace: FONT,
  valAxisTitleFontSize: 11,
  axisLineColor: GRID,
  valGridLine: { color: GRID, size: 0.75 },
  catGridLine: { style: "none" },
  valAxisMinVal: 0,
  showValAxisTitle: true,
  showTitle: true,
  titleFontFace: FONT,
  titleFontSize: 13,
  titleColor: INK,
};

const X0 = 0.7, W = 3.85, GAP = 0.25, Y = 1.15, H = 4.3;

PANELS.forEach((p, i) => {
  const x = X0 + i * (W + GAP);
  const opts = {
    ...common,
    x, y: Y, w: W, h: H,
    title: p.title,
    valAxisTitle: p.axis,
  };
  if (p.max !== undefined) { opts.valAxisMaxVal = p.max; opts.valAxisMajorUnit = p.unit; }

  if (p.normTo) {
    // A combo chart so the 1.0 baseline is a real, editable series rather than a shape
    // that detaches the moment someone resizes the chart.
    const flat = [{ name: "Baseline (Recompute = 1.0)", labels: names,
                    values: names.map(() => 1) }];
    slide.addChart(
      [{ type: pres.ChartType.bar, data: seriesFor(p), options: { barGrouping: "clustered" } },
       { type: pres.ChartType.line, data: flat,
         options: { chartColors: [INK], lineSize: 1.5, lineDash: "dash",
                    lineDataSymbol: "none", secondaryValAxis: false } }],
      { ...opts, chartColors: ARMS.map((a) => a.color).concat([INK]) }
    );
  } else {
    slide.addChart(pres.ChartType.bar, seriesFor(p), opts);
  }

  // Ratio labels as text: a native chart cannot annotate a GROUP, only a bar, and the
  // claim belongs to the model, not to any one arm.
  ratios(p).forEach((r, mi) => {
    if (r === null) return;
    const colW = W / names.length;
    slide.addText(r.toFixed(2) + "×", {
      x: x + 0.28 + mi * (colW * 0.92), y: Y + 0.42, w: colW * 0.9, h: 0.28,
      fontFace: FONT, fontSize: 12, bold: true, color: INK,
      align: "center", valign: "middle", margin: 0,
    });
  });
});

// One legend for all three panels, as text: it serves every chart and stays put when any
// of them is resized.
let lx = 4.6;
ARMS.forEach((a) => {
  slide.addShape(pres.ShapeType.rect, {
    x: lx, y: 5.72, w: 0.3, h: 0.16, fill: { color: a.color },
    line: { color: INK, width: 0.75 },
  });
  slide.addText(a.label, {
    x: lx + 0.38, y: 5.6, w: 1.3, h: 0.4,
    fontFace: FONT, fontSize: 12, color: INK, margin: 0, valign: "middle",
  });
  lx += 1.55;
});

slide.addText(
  "Hit rate and host DRAM improve on all three models. Normalized TTFT is the " +
  "exception on Llama-2-13B (0.95×): its " +
  "4096-position limit caps conversations at four turns, so the re-prefill a hit avoids " +
  "is small enough that the fetch does not pay for itself — the mechanism still works " +
  "there (1.66× hit rate), it just has less to win.",
  { x: 0.45, y: 6.25, w: 12.4, h: 0.8,
    fontFace: FONT, fontSize: 11, color: MUTED, margin: 0 }
);

slide.addNotes(
  "All four are native PowerPoint charts: right-click > Edit Data to change values, or " +
  "the Chart Design / Format tabs for colours and axes.\n" +
  "TTFT and throughput are normalised to Recompute within each model, because their " +
  "absolute values are set mostly by model size and would invite comparison across " +
  "models. The dashed line is Recompute.\n" +
  "Panel (c) is FLAT ON PURPOSE: the benchmark is closed-loop, so all three arms are " +
  "handed the same conversations at the same concurrency and finish in the same wall " +
  "time. Throughput = fixed work / same wall, identical by construction. Use qps_sweep " +
  "(open-loop, server-saturating) if the throughput claim needs to separate the arms.\n" +
  "Recompute's hit rate is 0 by construction -- it has no cache -- not missing data.\n" +
  "Source: results/models/{llama8b,llama13b,qwen14b}/table.json"
);

pres.writeFile({ fileName: outPath }).then(() => console.log("[saved] " + outPath));
