#!/usr/bin/env python3
"""
M3: did placement actually follow headroom? (Exp 2)

Reads the JSONL that _select_pool writes when SGLANG_KV_PARK_DECISION_LOG is set: one
record per park, naming the GPU chosen AND every candidate's live serving usage, park
headroom and link class at that instant.

The rejected candidates are the whole point. "The KV went where there was room" is a
claim about them: a run where every park lands on an idle GPU proves nothing if every
candidate was idle. What has to be shown is that the chosen GPU was idler than the ones
passed over, at the moment of choosing.

Three things are reported:

  selection accuracy   fraction of parks that landed on the idlest fast-link candidate.
                       This is close to a restatement of the selector's code, so it is
                       here as a sanity check on the telemetry, not as the result.

  usage gap            mean usage of the chosen GPU minus mean usage of the rejected
                       ones. Negative means placement went toward the idle side, and
                       the magnitude says how much idler. THIS is the claim, in one
                       number, and it is a property of the workload as much as of the
                       policy -- if the candidates are all equally loaded it is 0 no
                       matter how correct the code is.

  null models          the same log replayed under random / round-robin / always-local
                       choice. They answer "would anything have done as well?" without
                       spending a GPU-hour per alternative. If the real policy's usage
                       gap is not clearly below random's, the pressure signal is not
                       doing work.

사용:
  python benchmark/collect_decisions.py --glob 'results/exp2/pd_*/decisions_park_pd*.jsonl'
  python benchmark/collect_decisions.py --dir results/exp2/pd_layoutb_p60000_c16 \
      --arms park_pd park_pd_blind --out .../decisions.json
"""
import argparse
import glob as globmod
import json
import os
import random
import statistics as st


def load(paths):
    recs = []
    for p in paths:
        with open(p) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    recs.append(json.loads(line))
                except json.JSONDecodeError:
                    continue     # a line torn by SIGKILL; the rest of the file is fine
    recs.sort(key=lambda r: r.get("t", 0))
    return recs


def _use(c):
    """A candidate's usage for scoring. None means the GPU published no telemetry, which
    the selector treats as idle; keep that convention here so the null models and the
    real policy are scored on the same footing."""
    u = c.get("use")
    return 0.0 if u is None else u


def analyse(recs, seed=0):
    if not recs:
        return {"error": "no decision records"}
    rng = random.Random(seed)
    n = len(recs)
    hit = 0                       # chose the idlest fast-link candidate
    chosen_u, rejected_u = [], []
    per_target, cross = {}, 0
    no_telem = 0
    rr_state = {}
    null = {k: [] for k in ("random", "round_robin", "always_local", "policy")}

    for r in recs:
        cand = r["cand"]
        chosen = str(r["chosen"])
        src = str(r.get("src_gpu"))
        fast = [g for g, c in cand.items() if not c.get("slow")] or list(cand)

        best = min(fast, key=lambda g: (_use(cand[g]), -cand[g].get("head", 0)))
        if chosen == best:
            hit += 1
        chosen_u.append(_use(cand[chosen]))
        rej = [_use(c) for g, c in cand.items() if g != chosen]
        if rej:
            rejected_u.append(st.mean(rej))
        per_target[chosen] = per_target.get(chosen, 0) + 1
        if chosen != src:
            cross += 1
        if cand[chosen].get("use") is None:
            no_telem += 1

        # Null models, scored by the same quantity as the policy: the usage of the GPU
        # they would have picked. Each sees the identical candidate set and instant.
        null["policy"].append(_use(cand[chosen]))
        null["random"].append(_use(cand[rng.choice(list(cand))]))
        keys = sorted(cand)
        i = rr_state.get(src, 0)
        null["round_robin"].append(_use(cand[keys[i % len(keys)]]))
        rr_state[src] = i + 1
        null["always_local"].append(_use(cand[src]) if src in cand else float("nan"))

    gap = (st.mean(chosen_u) - st.mean(rejected_u)) if rejected_u else None
    out = {
        "parks": n,
        "selection_accuracy": round(hit / n, 4),
        "chosen_use_mean": round(st.mean(chosen_u), 4),
        "rejected_use_mean": round(st.mean(rejected_u), 4) if rejected_u else None,
        "usage_gap": round(gap, 4) if gap is not None else None,
        "cross_gpu_frac": round(cross / n, 4),
        "target_share": {k: round(v / n, 4) for k, v in sorted(per_target.items())},
        "chosen_had_no_telemetry_frac": round(no_telem / n, 4),
        "null_models_mean_use": {
            k: (round(st.mean([x for x in v if x == x]), 4) if any(x == x for x in v)
                else None)
            for k, v in null.items()
        },
    }
    return out


def report(tag, a):
    if "error" in a:
        print(f"\n### {tag}: {a['error']}")
        return
    print(f"\n### {tag}   ({a['parks']} parks)")
    print(f"  selection accuracy (idlest fast-link candidate) : {a['selection_accuracy']*100:.1f}%")
    print(f"  usage of chosen GPU  : {a['chosen_use_mean']:.3f}")
    print(f"  usage of rejected    : {a['rejected_use_mean']}")
    print(f"  USAGE GAP            : {a['usage_gap']}   (negative = placed toward idle)")
    print(f"  cross-GPU parks      : {a['cross_gpu_frac']*100:.1f}%")
    print(f"  target share         : {a['target_share']}")
    if a["chosen_had_no_telemetry_frac"] > 0.01:
        print(f"  [warn] {a['chosen_had_no_telemetry_frac']*100:.0f}% of parks chose a GPU "
              f"with NO telemetry, which the selector scores as idle 0.0. Those "
              f"placements were not evidence-driven; check the usage publishers.")
    print(f"  null models (mean usage of the GPU each would pick, lower is better):")
    for k, v in a["null_models_mean_use"].items():
        mark = "  <- actual" if k == "policy" else ""
        print(f"     {k:14s} {v}{mark}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", help="glob of decision JSONL files (one analysis)")
    ap.add_argument("--dir", help="directory holding decisions_<arm>.*.jsonl")
    ap.add_argument("--arms", nargs="*", default=[])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out")
    args = ap.parse_args()

    jobs = []
    if args.glob:
        jobs.append((os.path.basename(args.glob), sorted(globmod.glob(args.glob))))
    if args.dir:
        arms = args.arms or sorted({
            os.path.basename(p).split(".")[0][len("decisions_"):]
            for p in globmod.glob(os.path.join(args.dir, "decisions_*.jsonl"))
        })
        for arm in arms:
            jobs.append((arm, sorted(globmod.glob(
                os.path.join(args.dir, f"decisions_{arm}.*.jsonl")))))
    if not jobs:
        ap.error("give --glob or --dir")

    out = {}
    for tag, paths in jobs:
        if not paths:
            print(f"[skip] no decision log for {tag} "
                  f"(was SGLANG_KV_PARK_DECISION_LOG set for that arm?)")
            continue
        out[tag] = analyse(load(paths), args.seed)
        report(tag, out[tag])

    if args.out:
        with open(args.out, "w") as fh:
            json.dump(out, fh, indent=2)
        print(f"\n[collect] -> {args.out}")


if __name__ == "__main__":
    main()
