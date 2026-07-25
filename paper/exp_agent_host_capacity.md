# Experiment: host-side KV offloading shrinks the agent stack's host capacity

Backs the last Motivation bullet — *"host side resource를 kv offloading storage로 사용하는
것은 agent-serving stack이 사용할 host capacity를 감소시킴"* — with measured numbers.

## 1. What already exists (no new run needed)

From the write-policy memory sweep (`results/mem/bd_*.csv`, sampled once per second
during steady 2P2D serving on server17, MemTotal ≈ 126 GB):

| config | MemAvailable during serving (mean) | host KV pool (non-recl.) | L3 page cache (recl.) | L3 on disk |
|---|---|---|---|---|
| `write_through` | 44.3 GB | 61.1 GB | 25.3 GB | 108.7 GB |
| `write_through_selective` | 43.7 GB | 61.5 GB | 24.2 GB | 108.6 GB |
| `write_back` | 44.5 GB | 61.1 GB | 0.2 GB | 0.0 GB |
| `L2_only` (no file backend) | 44.6 GB | 61.0 GB | 0.0 GB | 0.0 GB |
| **`park` (KV victim cache)** | **105.5 GB** | **0.0 GB** | **0.0 GB** | 0.0 GB |

Two facts to state in the Motivation:

1. **HiCache leaves the rest of the node 44 GB of 126 GB (35%); the victim cache leaves
   105 GB (84%) — 2.4× more host DRAM for everything that is not the LLM.**
2. **This is a property of `--hicache-ratio`, not of the write policy or the storage
   backend.** `write_back` and L2-only were measured specifically to preempt the
   reviewer question, and both still hold the same ~61 GB. The pool is pre-allocated
   and forced larger than device memory, so no policy knob reduces it.

The weakness of stopping here: 44 GB is still a large absolute number, so a reviewer can
answer "so what — 44 GB is plenty." Section 2 converts capacity into agent-side work.

## 2. New measurement: agent capacity and tool-execution latency

`benchmark/agent_host_pressure.py` co-locates a synthetic **agent-side sidecar** with the
serving stack, because the agent stack is what actually competes for that DRAM:
orchestration state per session, request scheduling, and above all **tool execution**
(code interpreter, dataframe/JSON processing, retrieval).

One *agent worker* = one concurrent tool-execution slot. Each iteration ("tool call")
allocates `--tool-mem-mb` (default 1 GB) of anonymous memory, touches every page, and
runs one bandwidth-bound pass (fill + sum + partial sort) — so it consumes exactly the
two resources the L2 pool holds: host DRAM capacity and host memory bandwidth.

### Metrics

| metric | mode | what it proves |
|---|---|---|
| **K_max** — admitted agent concurrency: largest number of concurrent tool executions sustainable before MemAvailable hits the floor | `ramp` | the capacity claim in agent units: "HiCache admits N concurrent tools, the victim cache admits 2.4N" |
| **tool-call p50/p99 latency vs K** | `ramp` | degradation *below* the cliff — reclaim + bandwidth contention, not just capacity |
| **tool-call p50/p99 at a fixed K all configs sustain** | `fixed` | isolates contention from capacity (same K, same offered load, different host pressure) |
| **RAG probe: p99 read latency + MAJOR fault count** | `rag` | answers "the L3 tier is only reclaimable page cache, so it's free" — reclaim evicts the *agent's* file working set, turning page-cache hits into disk reads |
| MemAvailable / AnonPages / file cache throughout | `sys_mem_breakdown.py` | ties the agent-side result back to where the DRAM went |

The predicted headline, from the MemAvailable numbers with 1 GB tools and an 8 GB floor:
**K_max ≈ 35 for every HiCache config vs ≈ 95 for the victim cache.** If the ramps come
out near that, the Motivation sentence becomes: *"under HiCache the same node admits only
35 concurrent tool executions; reclaiming idle HBM instead admits 95 — 2.7×, with
identical prefix-hit TTFT."*

### Safety on a shared node

server17 is shared, so the harness is written to stop **before** the kernel OOM killer:
every worker sets `oom_score_adj=1000` (the sidecar is the kernel's first victim, never
the SGLang server or another tenant), the ramp aborts as soon as MemAvailable falls
below `--floor-gb` (default 8 GB), and workers free their buffer between iterations so an
abort releases memory at once. Do not set `--floor-gb` below ~6. The RAG probe is
read-only and points at an existing model-weight shard — it writes nothing, since the
root filesystem is at 99%.

## 3. Runbook

Four configs, one invocation each. The server must already be up in the labelled config;
`scripts/run_agent_pressure.sh` starts the memory sampler, puts steady LLM load on the
router, runs the read-only RAG probe, then ramps.

```bash
# A) HiCache, write_through  (what all earlier experiments used)
HICACHE_WRITE_POLICY=write_through ./scripts/sglang/start_2P_2D.sh
LABEL=write_through ./scripts/run_agent_pressure.sh
./scripts/stop.sh

# B) HiCache, write_back  (reviewer Q: does the policy matter?)
HICACHE_WRITE_POLICY=write_back ./scripts/sglang/start_2P_2D.sh
LABEL=write_back ./scripts/run_agent_pressure.sh
./scripts/stop.sh

# C) HiCache, L2 only, file backend off  (reviewer Q: is it just page cache?)
HICACHE_STORAGE_BACKEND=none HICACHE_WRITE_POLICY=write_back ./scripts/sglang/start_2P_2D.sh
LABEL=L2_only ./scripts/run_agent_pressure.sh
./scripts/stop.sh

# D) KV victim cache, host-RAM-free
PARK_NO_HICACHE=1 IDLE_KV_PARKING=1 ./scripts/sglang/start_2P_2D.sh
LABEL=park ./scripts/run_agent_pressure.sh
./scripts/stop.sh

# figure
python benchmark/plot_agent_pressure.py --ramp \
  write_through=results/agent/ramp_write_through.json \
  write_back=results/agent/ramp_write_back.json \
  L2_only=results/agent/ramp_L2_only.json \
  park=results/agent/ramp_park.json \
  --out results/agent/fig_agent_host_capacity.png
```

Optional third arm for the same figure: `--mode fixed --workers 30` per config (a K every
config sustains) to report the pure contention slowdown at equal offered agent load.

Total: ~15 min per config, ~1 h for all four.

## 4. Threats to validity to state in the paper

- The sidecar is synthetic. It is calibrated by working-set size and bandwidth profile,
  not by replaying a real orchestrator; the claim is about capacity, not about any
  specific framework's constant factor.
- K_max depends on `--tool-mem-mb`; report it, and report the ratio (which is
  insensitive to it) alongside the absolute value.
- MemAvailable is a kernel estimate. It is reported together with AnonPages and the
  file-cache split so the non-reclaimable component is auditable.
- Node RAM (126 GB) is small relative to production agent-serving hosts. Since the L2
  pool is required to exceed device memory, the pressure scales with GPU count rather
  than disappearing on larger hosts — but the specific ratio is node-dependent.
