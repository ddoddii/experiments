// results/agent_trace/bench_*.json as an EDITABLE PowerPoint deck.
//
// Native clustered-bar charts (values/colours/axes stay editable in PowerPoint), one
// per metric, normalised to the recompute bar exactly as plot_agent_trace.py does.
// Bars share one category axis (concurrency), so unlike the qps/ttft decks nothing here
// needs to fall back to a pasted image for the charts themselves -- the rendered figure
// still ships on its own slide for the exact camera-ready look.
//
// 사용:
//   node benchmark/make_agent_trace_pptx.js results/agent_trace/fig_agent_trace.pptx \
//        results/agent_trace

const pptxgen = require("pptxgenjs");
const fs = require("fs");
const path = require("path");

const args = process.argv.slice(2);
const pos = args.filter((a) => !a.startsWith("--"));
const outPath = pos[0] || "results/agent_trace/fig_agent_trace.pptx";
const dataDir = pos[1] || "results/agent_trace";
const flag = (n, d) => {
  const i = args.indexOf(`--${n}`);
  return i >= 0 && args[i + 1] ? args[i + 1] : d;
};
const ttftKey = flag("ttft-stat", "ttft_p50_s");
const figPath = flag("fig", path.join(dataDir, "fig_agent_trace.png"));

const ARMS = [
  { key: "recompute", label: "Recompute", color: "B8BCC0" },
  { key: "hicache", label: "SGLang", color: "0072B2" },
  { key: "park_host", label: "Ours (host DRAM)", color: "E69F00" },
  { key: "park", label: "Ours", color: "009E73" },
];
const INK = "1A1A1A", MUTED = "595959", GRID = "D9D9D9", FONT = "Times New Roman";

const data = {};
for (const f of fs.readdirSync(dataDir)) {
  const m = /^bench_(.+)_c(\d+)\.json$/.exec(f);
  if (!m) continue;
  try {
    const s = JSON.parse(fs.readFileSync(path.join(dataDir, f), "utf8")).summary;
    (data[m[1]] ||= {})[Number(m[2])] = s;
  } catch (e) {
    console.error(`[warn] ${f}: ${e.message}`);
  }
}
if (!data.recompute) {
  console.error("[fatal] no recompute arm -- it is the normalisation baseline, so every "
                + "bar would be relative to nothing.");
  process.exit(1);
}
// Only concurrencies with the baseline: a group present for some arms but not recompute
// cannot be normalised, and an empty labelled group reads as a measured zero.
const cs = Object.keys(data.recompute).map(Number).sort((a, b) => a - b);
const present = ARMS.filter((a) => data[a.key] && cs.some((c) => data[a.key][c]));

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE";
pres.author = "SGLang KV placement experiment";
pres.title = "Agent-trace replay: TTFT and TPOT";

const s1 = pres.addSlide();
s1.addText("Prefix reuse cuts time-to-first-token; decode is untouched", {
  x: 0.45, y: 0.22, w: 12.4, h: 0.45,
  fontFace: FONT, fontSize: 24, bold: true, color: INK, margin: 0,
});
s1.addText(
  "Codex / SWE-bench Pro traces · 100 sessions · 8-15 turns · context 12k->48k tokens · "
  + "Llama-3.1-8B · SGLang 2P2D · 4x RTX A6000 · normalised to Recompute",
  { x: 0.45, y: 0.68, w: 12.4, h: 0.3,
    fontFace: FONT, fontSize: 11, color: MUTED, italic: true, margin: 0 }
);

const common = {
  barDir: "col", barGrouping: "clustered",
  showLegend: false,
  catAxisLabelColor: MUTED, valAxisLabelColor: MUTED,
  catAxisLabelFontFace: FONT, valAxisLabelFontFace: FONT,
  catAxisLabelFontSize: 11, valAxisLabelFontSize: 10,
  valAxisTitleFontFace: FONT, valAxisTitleFontSize: 11,
  axisLineColor: GRID, valGridLine: { color: GRID, size: 0.75 },
  valAxisMinVal: 0, valAxisMaxVal: 1.25,
  showValAxisTitle: true, showTitle: true,
  titleFontFace: FONT, titleFontSize: 13, titleColor: INK,
  chartColors: present.map((a) => a.color),
};

const PANELS = [
  { key: ttftKey, title: "(a)  Time to first token", axis: "Normalized TTFT" },
  { key: "avg_tpot_s", title: "(b)  Time per output token", axis: "Normalized TPOT" },
];
const X0 = 1.9, W = 4.6, GAP = 0.5, Y = 1.35, H = 4.3;
PANELS.forEach((p, i) => {
  const series = present.map((a) => ({
    name: a.label,
    labels: cs.map((c) => `C=${c}`),
    values: cs.map((c) => {
      const base = data.recompute[c]?.[p.key], v = data[a.key][c]?.[p.key];
      return base && v ? Number((v / base).toFixed(4)) : null;
    }),
  }));
  s1.addChart(pres.ChartType.bar, series, {
    ...common, x: X0 + i * (W + GAP), y: Y, w: W, h: H,
    title: p.title, valAxisTitle: p.axis,
  });
});

let lx = 3.2;
for (const a of present) {
  s1.addShape(pres.ShapeType.rect, {
    x: lx, y: 5.85, w: 0.3, h: 0.18,
    fill: { color: a.color }, line: { color: "000000", width: 0.5 },
  });
  s1.addText(a.label, {
    x: lx + 0.4, y: 5.72, w: 2.0, h: 0.4,
    fontFace: FONT, fontSize: 12, color: INK, margin: 0, valign: "middle",
  });
  lx += 2.4;
}
s1.addText(
  "1.0 = no better than recomputing the prefix from scratch. TPOT sits at ~1.0 because "
  + "prefix reuse is a PREFILL optimisation -- it changes what must be computed before "
  + "the first token and does essentially nothing for the per-token decode rate. That "
  + "panel is the check that the TTFT win comes from where the mechanism acts.",
  { x: 0.45, y: 6.3, w: 12.4, h: 0.8,
    fontFace: FONT, fontSize: 11, color: MUTED, margin: 0 }
);
s1.addNotes(
  `Absolute values at C=4 (${ttftKey} / avg_tpot_s / ttft_p95_s):\n`
  + present.map((a) => {
      const s = data[a.key][cs[0]];
      return s ? `  ${a.label}: ${s[ttftKey]}s / ${s.avg_tpot_s}s / ${s.ttft_p95_s}s` : "";
    }).filter(Boolean).join("\n")
  + "\n\nThe TTFT panel uses the median. Ours is 0.41x recompute at p50 but 0.66x on the "
  + "per-turn mean, because a park miss falls back to a full re-prefill and lands in the "
  + "tail -- at C=4 its p95 (19.5s) is worse than SGLang's (17.6s). Both are real; "
  + "reporting only p50 would overstate the effect.\n"
  + "Charts are native: right-click > Edit Data, or the Chart Design / Format tabs."
);

if (fs.existsSync(figPath)) {
  const s2 = pres.addSlide();
  s2.addText("Full figure, as rendered (plot_agent_trace.py)", {
    x: 0.45, y: 0.22, w: 12.4, h: 0.4,
    fontFace: FONT, fontSize: 18, bold: true, color: INK, margin: 0,
  });
  s2.addImage({ path: figPath, x: 0.9, y: 1.2, w: 11.5, h: 11.5 * (2.5 / 7.16) });
  s2.addText(`Source: ${figPath}`, {
    x: 0.45, y: 6.6, w: 12.4, h: 0.3,
    fontFace: FONT, fontSize: 10, color: MUTED, italic: true, margin: 0,
  });
}

pres.writeFile({ fileName: outPath }).then(() => console.log("[saved] " + outPath));
