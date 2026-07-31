// Exp 1 figure as an EDITABLE PowerPoint slide: two NATIVE charts, not pictures.
//
// Native means each chart carries its own worksheet, so the numbers, colours, fonts and
// axis ranges are all editable in PowerPoint. A PDF or PNG placed on a slide is not --
// and "I want to edit it in PPT" is the whole reason this file exists.
//
// Both panels are line charts with category x-axes rather than scatter: a category axis
// is what PowerPoint's chart editor handles most predictably, and neither x variable
// needs true numeric spacing (turns are 0..8, minutes are evenly sampled).
//
// The memory series is downsampled to one point per minute. At the sampler's 1 Hz that
// would be ~2700 rows per arm, which makes the embedded worksheet unusable by hand; the
// curves are flat, so a per-minute mean loses nothing visible.
//
// 사용:
//   python3 -c "...write exp1_chartdata.json..."   (see make_exp1_pptx.sh)
//   node benchmark/make_exp1_pptx.js <chartdata.json> <out.pptx>

const pptxgen = require("pptxgenjs");
const fs = require("fs");

const dataPath = process.argv[2];
const outPath = process.argv[3] || "fig_exp1.pptx";
const D = JSON.parse(fs.readFileSync(dataPath, "utf8"));

// Same hue order as the PDF figure, so the two never disagree. Validated as a
// categorical trio (worst adjacent CVD dE 18.0 protan / 18.7 normal). No leading '#':
// pptxgenjs corrupts the file on '#' or 8-digit hex.
const ARMS = [
  { key: "radix",   label: "GPU-only (radix)",     color: "E69F00" },
  { key: "hicache", label: "HiCache (host DRAM)",  color: "0072B2" },
  { key: "park",    label: "GPU-first (ours)",     color: "009E73" },
];
const INK = "1A1A1A";
const MUTED = "595959";
const GRID = "D9D9D9";
// Times New Roman: matches the paper, and it ships with Office rather than being
// substituted at unpredictable widths.
const FONT = "Times New Roman";

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE";           // 13.3 x 7.5 in -- set BEFORE addSlide
pres.author = "SGLang KV placement experiment";
pres.title = "Exp 1 - GPU-first unified KV placement";

const slide = pres.addSlide();

slide.addText("GPU-first KV placement: same latency budget, half the host DRAM", {
  x: 0.5, y: 0.28, w: 12.3, h: 0.5,
  fontFace: FONT, fontSize: 24, bold: true, color: INK, margin: 0,
});
slide.addText(
  "ShareGPT multi-turn · SGLang 2P2D · Llama-3.1-8B · C=8 · 60k-token prefill pool · 1440 turns per arm · no failures",
  { x: 0.5, y: 0.78, w: 12.3, h: 0.3,
    fontFace: FONT, fontSize: 11, color: MUTED, italic: true, margin: 0 }
);

// Shared frame styling. Quiet axes and a horizontal-only grid keep the marks dominant.
const common = {
  showLegend: false,
  chartColors: ARMS.map((a) => a.color),
  lineSize: 2.5,
  lineSmooth: false,
  catAxisLabelColor: MUTED,
  valAxisLabelColor: MUTED,
  catAxisLabelFontFace: FONT,
  valAxisLabelFontFace: FONT,
  catAxisLabelFontSize: 11,
  valAxisLabelFontSize: 11,
  catAxisTitleFontFace: FONT,
  valAxisTitleFontFace: FONT,
  catAxisTitleFontSize: 12,
  valAxisTitleFontSize: 12,
  axisLineColor: GRID,
  valGridLine: { color: GRID, size: 0.75 },
  catGridLine: { style: "none" },
  showTitle: true,
  titleFontFace: FONT,
  titleFontSize: 14,
  titleColor: INK,
};

// ---------------------------------------------------------------- (a) latency
// x is the TURN INDEX, not wall time: the mechanism can only act once a conversation
// has a prefix to reuse, so the turn number is the amount of reusable history. Turn 0
// is the control -- no arm can reuse or fetch there.
const turns = Object.keys(D.ttft.park).map(Number).sort((a, b) => a - b);
const ttftData = ARMS.map((a) => ({
  name: a.label,
  labels: turns.map(String),
  values: turns.map((t) => D.ttft[a.key][t] ?? null),
}));

slide.addChart(pres.ChartType.line, ttftData, {
  ...common,
  x: 0.5, y: 1.3, w: 6.1, h: 4.4,
  title: "(a)  Median TTFT by turn",
  catAxisTitle: "turn index",
  showCatAxisTitle: true,
  valAxisTitle: "median TTFT (s)",
  showValAxisTitle: true,
  valAxisMinVal: 0,
  valAxisMaxVal: 0.55,
  valAxisMajorUnit: 0.1,
  lineDataSymbol: "circle",
  lineDataSymbolSize: 7,
});

// ---------------------------------------------------------------- (b) host memory
// x is WALL TIME because the claim is about a RESERVATION: HiCache's host pool is
// allocated at server start and never shrinks, which only shows as a line that begins
// high at t=0. Markers off -- 46 points per arm would read as noise.
const mins = Object.keys(D.mem.park).map(Number).sort((a, b) => a - b);
const memData = ARMS.map((a) => ({
  name: a.label,
  labels: mins.map(String),
  values: mins.map((m) => D.mem[a.key][m] ?? null),
}));

slide.addChart(pres.ChartType.line, memData, {
  ...common,
  x: 6.85, y: 1.3, w: 5.95, h: 4.4,
  title: "(b)  Host DRAM (AnonPages) over the run",
  catAxisTitle: "wall time (min)",
  showCatAxisTitle: true,
  valAxisTitle: "host DRAM, AnonPages (GB)",
  showValAxisTitle: true,
  valAxisMinVal: 0,
  valAxisMaxVal: 40,
  valAxisMajorUnit: 5,
  lineDataSymbol: "none",
  catAxisLabelFrequency: 5,        // 46 minute labels would collide
});

// Legend as text rather than the chart's own: one legend serves both panels, and it
// stays put when the charts are resized. Colour-coded text plus the panel titles means
// identity never rests on colour alone.
let lx = 0.5;
ARMS.forEach((a) => {
  slide.addShape(pres.ShapeType.rect, {
    x: lx, y: 6.03, w: 0.32, h: 0.08, fill: { color: a.color }, line: { width: 0 },
  });
  slide.addText(a.label, {
    x: lx + 0.4, y: 5.88, w: 2.3, h: 0.35,
    fontFace: FONT, fontSize: 12, color: INK, margin: 0, valign: "middle",
  });
  lx += 2.85;
});

// The two numbers the figure exists to deliver, as editable text.
slide.addText(
  "Turn 0 is a control: no arm can reuse or fetch, so its gap is fixed overhead. " +
  "GPU-first is faster at every later turn (−0.06 to −0.08 s).",
  { x: 0.5, y: 6.42, w: 6.1, h: 0.55,
    fontFace: FONT, fontSize: 11, color: MUTED, margin: 0 }
);
slide.addText(
  "HiCache commits its host pool at startup and never releases it: −17.9 GB AnonPages, " +
  "traded for +9.3 GB of otherwise-idle GPU HBM.",
  { x: 6.85, y: 6.42, w: 5.95, h: 0.55,
    fontFace: FONT, fontSize: 11, color: MUTED, margin: 0 }
);

slide.addNotes(
  "Both charts are native PowerPoint charts: right-click > Edit Data to change values, " +
  "or use the Chart Design / Format tabs for colours and axes.\n" +
  "Source: results/exp1/sharegpt_p60000_c8_m1024. Memory downsampled to 1 point/min " +
  "from a 1 Hz sampler. TTFT is the median per turn; turns with fewer than 30 samples " +
  "(turn 9, n=24) are omitted."
);

pres.writeFile({ fileName: outPath }).then(() => console.log("[saved] " + outPath));
