// results/ttft_ctx/*.json as an EDITABLE PowerPoint deck.
//
// Same approach as make_qps_pptx.js: panels whose arms share ONE x series become NATIVE
// charts (values/colours/axes stay editable in PowerPoint), and anything that doesn't
// ships as the rendered PNG instead of being faked.
//
//   slide 1  (a) TTFT vs context length + (b) speedup over recompute -- native scatter,
//            both share the context-length x series (guarded below).
//   slide 2  (d) the 64k best-rep vs median bars -- native bar chart. This panel carries
//            the caveat, so it gets its own slide rather than being a footnote.
//   slide 3  the full rendered 4-panel figure.
//
// SCATTER, not line, for slide 1: the context lengths are 4096..65536, i.e. unevenly
// spaced in absolute terms. A pptxgenjs line chart treats x as CATEGORIES and lays them
// out at equal intervals, which is actually what we want here (log2 spacing) -- but
// scatter keeps the axis numeric and matches the rendered figure, so the two versions of
// the same panel can be put side by side without one of them lying about the spacing.
//
// 사용:
//   node benchmark/make_ttft_ctx_pptx.js results/ttft_ctx/fig_ttft_ctx.pptx results/ttft_ctx

const pptxgen = require("pptxgenjs");
const fs = require("fs");
const path = require("path");

const args = process.argv.slice(2);
const pos = args.filter((a) => !a.startsWith("--"));
const outPath = pos[0] || "results/ttft_ctx/fig_ttft_ctx.pptx";
const dataDir = pos[1] || "results/ttft_ctx";

function flagArg(name, fallback) {
  const i = args.indexOf(`--${name}`);
  return i >= 0 && args[i + 1] ? args[i + 1] : fallback;
}
const figPath = flagArg("fig", path.join(dataDir, "fig_ttft_ctx_panels.png"));
const detailLen = parseInt(flagArg("detail-len", "65536"), 10);
const title = flagArg(
  "title",
  "GPU-resident KV keeps prefix reuse working where host offload gives up"
);

// paperstyle.PALETTE, minus '#' (pptxgenjs corrupts the file on '#' or 8-digit hex).
const ARMS = [
  { key: "recompute", label: "Recompute", color: "B4423C" },
  { key: "hicache", label: "SGLang", color: "0072B2" },
  { key: "park_host", label: "Ours (host DRAM)", color: "E69F00" },
  { key: "park", label: "Ours", color: "009E73" },
];
const INK = "1A1A1A";
const MUTED = "595959";
const GRID = "D9D9D9";
const FONT = "Times New Roman";

const data = {};
for (const a of ARMS) {
  const p = path.join(dataDir, `${a.key}.json`);
  if (!fs.existsSync(p)) continue;
  const rows = JSON.parse(fs.readFileSync(p, "utf8")).rows;
  data[a.key] = new Map(rows.map((r) => [r.context_len, r]));
}
const present = ARMS.filter((a) => data[a.key]);
if (!present.length) {
  console.error(`[fatal] no <arm>.json found in ${dataDir}`);
  process.exit(1);
}

const xs = [...new Set(present.flatMap((a) => [...data[a.key].keys()]))].sort(
  (p, q) => p - q
);
// A scatter chart shares ONE x series across every y series, so an arm measured at a
// different set of context lengths would have its values silently paired with the wrong
// x. Same guard as make_qps_pptx.js.
for (const a of present) {
  const own = [...data[a.key].keys()].sort((p, q) => p - q);
  if (own.join() !== xs.join()) {
    console.error(
      `[fatal] ${a.key} has context lengths ${own} but the union is ${xs}. A scatter ` +
        `chart shares one x series, so mismatched lengths would pair each y with the ` +
        `wrong x. Re-run that arm over the same CONTEXT_LENS list.`
    );
    process.exit(2);
  }
}

const fmtTok = (n) => (n % 1024 === 0 ? `${n / 1024}k` : `${(n / 1024).toFixed(1)}k`);

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.3 x 7.5in -- set BEFORE addSlide
pres.author = "SGLang KV placement experiment";
pres.title = title;

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
  showValAxisTitle: true,
  showCatAxisTitle: true,
  catAxisTitle: "context length (tokens)",
  showTitle: true,
  titleFontFace: FONT,
  titleFontSize: 13,
  titleColor: INK,
};

function legendRow(slide, arms, y) {
  let lx = 1.6;
  for (const a of arms) {
    slide.addShape(pres.ShapeType.line, {
      x: lx, y: y + 0.18, w: 0.42, h: 0,
      line: { color: a.color, width: 2.25 },
    });
    slide.addText(a.label, {
      x: lx + 0.5, y, w: 2.2, h: 0.4,
      fontFace: FONT, fontSize: 12, color: INK, margin: 0, valign: "middle",
    });
    lx += 2.6;
  }
}

// ------------------------------------------------------- slide 1: TTFT + speedup
const s1 = pres.addSlide();
s1.addText(title, {
  x: 0.45, y: 0.22, w: 12.4, h: 0.45,
  fontFace: FONT, fontSize: 24, bold: true, color: INK, margin: 0,
});
s1.addText(
  `Llama-3.1-8B · SGLang 2P2D · 4x RTX A6000 · median of 3 reps · context ${xs
    .map(fmtTok)
    .join(", ")}`,
  { x: 0.45, y: 0.68, w: 12.4, h: 0.3,
    fontFace: FONT, fontSize: 11, color: MUTED, italic: true, margin: 0 }
);

const X0 = 1.9, W = 4.6, GAP = 0.5, Y = 1.3, H = 4.4;
const ttftData = [{ name: "context", values: xs }].concat(
  present.map((a) => ({
    name: a.label,
    values: xs.map((L) => data[a.key].get(L)?.ttft_median_s ?? null),
  }))
);
s1.addChart(pres.ChartType.scatter, ttftData, {
  ...common,
  x: X0, y: Y, w: W, h: H,
  title: "(a)  Time to first token",
  valAxisTitle: "TTFT (median, s)",
  chartColors: present.map((a) => a.color),
});

const base = data["recompute"];
const speedArms = present.filter((a) => a.key !== "recompute");
const speedData = [{ name: "context", values: xs }].concat(
  speedArms.map((a) => ({
    name: a.label,
    values: xs.map((L) => {
      const b = base?.get(L)?.ttft_median_s;
      const v = data[a.key].get(L)?.ttft_median_s;
      return b && v ? Number((b / v).toFixed(3)) : null;
    }),
  }))
);
s1.addChart(pres.ChartType.scatter, speedData, {
  ...common,
  x: X0 + W + GAP, y: Y, w: W, h: H,
  title: "(b)  Speedup over recompute  (1.0 = no better than no cache)",
  valAxisTitle: "speedup",
  chartColors: speedArms.map((a) => a.color),
});
legendRow(s1, present, 5.85);

s1.addNotes(
  "Both charts are native PowerPoint charts: right-click > Edit Data for values, or " +
    "the Chart Design / Format tabs for colours and axes.\n" +
    "In (b), 1.0 means the arm did no better than recomputing the prefix from scratch.\n" +
    "The medians at 64k hide a bimodal per-rep distribution -- see the next slide.\n" +
    `Source: ${dataDir}/<arm>.json`
);

// ------------------------------------------------------- slide 2: the 64k caveat
const detail = present.filter((a) => data[a.key].get(detailLen));
if (detail.length) {
  const s2 = pres.addSlide();
  s2.addText(`At ${fmtTok(detailLen)} the median hides how OFTEN the restore happened`, {
    x: 0.45, y: 0.22, w: 12.4, h: 0.45,
    fontFace: FONT, fontSize: 22, bold: true, color: INK, margin: 0,
  });
  s2.addText(
    "Every arm except Ours is bimodal here: its best repetition restores the prefix in " +
      "~5s, its median falls through to a full re-prefill. So the median gap is mostly a " +
      "difference in how RELIABLY the restore happens, not in how fast the transfer is.",
    { x: 0.45, y: 0.68, w: 12.4, h: 0.5,
      fontFace: FONT, fontSize: 12, color: MUTED, italic: true, margin: 0 }
  );
  s2.addChart(
    pres.ChartType.bar,
    [
      { name: "best rep", labels: detail.map((a) => a.label),
        values: detail.map((a) => data[a.key].get(detailLen).ttft_min_s) },
      { name: "median", labels: detail.map((a) => a.label),
        values: detail.map((a) => data[a.key].get(detailLen).ttft_median_s) },
    ],
    {
      ...common,
      x: 2.6, y: 1.4, w: 8.1, h: 4.6,
      barDir: "col", barGrouping: "clustered",
      showLegend: true, legendPos: "t", legendFontFace: FONT, legendFontSize: 11,
      title: `(d)  ${fmtTok(detailLen)}: best rep vs. median`,
      valAxisTitle: "TTFT (s)",
      catAxisTitle: "",
      showCatAxisTitle: false,
      chartColors: ["9DC3E6", "1F4E79"],
    }
  );
  s2.addNotes(
    "Bimodality at 64k: hicache best 5.11s / median 19.41s, Ours(host) best 5.49s / " +
      "median 24.03s, Ours best 4.60s / median 4.60s (all three reps restored).\n" +
      "Ours(host) is the FORCE_HOST ablation -- same index, eviction, reuse ranking and " +
      "fetch code as Ours, park pool still allocated so the arms are memory-matched, " +
      "only the storage medium moves to pinned host DRAM over PCIe.\n" +
      "n=3 reps: the hit-rate difference (3/3 vs 1/3) is suggestive, not yet tight."
  );
}

// ------------------------------------------------------- slide 3: rendered figure
if (fs.existsSync(figPath)) {
  const s3 = pres.addSlide();
  s3.addText("Full figure, as rendered (plot_ttft_ctx_panels.py)", {
    x: 0.45, y: 0.22, w: 12.4, h: 0.4,
    fontFace: FONT, fontSize: 18, bold: true, color: INK, margin: 0,
  });
  s3.addImage({ path: figPath, x: 1.0, y: 0.85, w: 11.3, h: 11.3 * (5.0 / 7.16) });
  s3.addText(`Source: ${figPath}`, {
    x: 0.45, y: 6.95, w: 12.4, h: 0.3,
    fontFace: FONT, fontSize: 10, color: MUTED, italic: true, margin: 0,
  });
} else {
  console.error(`[warn] ${figPath} not found -- skipping the rendered-figure slide. ` +
                `Pass --fig <path> if it was written elsewhere.`);
}

pres.writeFile({ fileName: outPath }).then(() => console.log("[saved] " + outPath));
