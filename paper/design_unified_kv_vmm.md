# Unified KV Memory Management — how to implement it

Design note for the "CPU DRAM + distributed GPU HBM as one logical KV memory space"
direction. Bottom line up front:

> **Use the CUDA VMM API (`cuMemCreate`/`cuMemMap`/`cuMemSetAccess`), not CUDA Unified
> Memory (`cudaMallocManaged`).** VMM gives exactly the abstraction the paper claims —
> one virtual address space, explicit per-slab physical residency — while UVM gives an
> *implicit* policy that cannot express the one placement decision the paper is about.

## 1. Why not CUDA Unified Memory

The current draft says "CUDA Unified Memory의 공통 주소 공간과 migration mechanism을
활용". Stated that way it is a liability with any reviewer who knows UVM:

1. **UVM cannot spill to a peer GPU.** Under device oversubscription the driver evicts
   pages to *host* memory. `cudaMemAdvise(SetPreferredLocation, peer)` sets a static
   preference; there is no interface for "evict this range to whichever peer currently
   has headroom". That decision — the core of this paper — is precisely what UVM keeps
   inside the driver.
2. **Fault-driven migration inside attention is unacceptable.** KV pages would fault at
   page granularity during a running attention kernel, serialising the GPU behind
   driver page-fault handling. The prefix-cache-hit path must be a pointer read, not a
   fault.
3. **No control over *when*.** The paper's contribution is deciding preferred location
   *and migration time* explicitly. UVM's access-counter heuristics own both.
4. It also does not compose with how SGLang allocates KV: one big preallocated
   per-layer tensor from PyTorch's caching allocator.

So UVM belongs in the paper as the *contrast*, and it is a stronger sentence than
"we use UVM":

> We provide the abstraction of a unified KV memory space — a single address space
> spanning local HBM, peer HBM and host DRAM — but with *explicit* residency and
> migration control, because the implicit policy of CUDA Unified Memory cannot express
> spilling to a transiently-idle peer GPU.

## 2. Why VMM is the right primitive

The VMM API separates **virtual address** from **physical backing**:

| step | call | meaning |
|---|---|---|
| reserve the space | `cuMemAddressReserve` | one contiguous VA range = "the unified KV space" of this rank |
| create physical memory | `cuMemCreate(prop)` | `prop.location` = `DEVICE(local)`, `DEVICE(peer)`, or `HOST`/`HOST_NUMA` |
| place it | `cuMemMap(va + off, size, 0, handle, 0)` | bind physical memory at a chosen address |
| grant access | `cuMemSetAccess(va+off, size, [descs])` | which devices may read/write it |
| change residency | `cuMemUnmap` + `cuMemMap` a different handle | **same VA, new physical location** |
| hand to another process | `cuMemExportToShareableHandle` (POSIX fd) / `cuMemImportFromShareableHandle` | the 2P2D multi-process case |

Three consequences that matter for this paper — **one of which the measurement in §3
kills, so read that before designing around it**:

1. **The pointer never changes.** A slab can move from local HBM to peer HBM to host
   DRAM while attention kernels, the radix index, and the KV index tensors keep the
   same addresses. Compare with the current park implementation, where a peer pool is a
   *separate tensor* and a hit requires an explicit gather-copy back into the local
   pool (`_p2p_gather` / `_gather_copy_from_peer` in `idle_kv_parking.py`). **This
   survives measurement and is the mechanism claim to make.**
2. ~~**Peer residency can be zero-copy.**~~ `cuMemSetAccess` does make a peer-backed
   slab directly readable, but at **27 GB/s vs 333 GB/s local on this machine (8%)** —
   see §3. Reading KV in place from a peer during attention would multiply TPOT by ~12×
   at 8k context. **Do not put peer-resident KV on the attention hot path.** The peer
   tier is a *restore source*, not an execution-time residency.
3. **Host is just another location.** `CU_MEM_LOCATION_TYPE_HOST_NUMA` (CUDA ≥ 12.2)
   would make the overflow tier a location in the same space rather than a separate
   subsystem. On server17 the first attempt returned `CUDA_ERROR_INVALID_VALUE`; the
   probe now tries several (location, handle-type) combinations, since requesting
   fd-exportable handles for a *host* location is itself a likely cause. If none works,
   the unified space covers local+peer HBM only and host overflow falls back to
   `cuMemHostAlloc` — say that explicitly rather than overclaiming.

Precedent that this composes with PyTorch: PyTorch's own
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` allocator is built on these calls.
Read `c10/cuda/CUDACachingAllocator.cpp` (`ExpandableSegment`) — it is the closest
working model for what you need to write.

## 3. Measured on server17 (4× RTX A6000)

`benchmark/vmm_probe.py --pairs --slab-mb 32`, servers down:

| quantity | measured | reading |
|---|---|---|
| allocation granularity | **2.00 MiB** (min = recommended) | KV slab must be a multiple of 2 MiB |
| P2P reachability | **all 12 ordered pairs = 1** | every GPU can map every other GPU's memory |
| local HBM read | **333.2 GB/s** | baseline |
| **peer HBM read (0←1)** | **27.2 GB/s = 8% of local** | **PCIe speed, not NVLink** (A6000 NVLink would be ~50 GB/s) |
| host-backed VMM slab | `CUDA_ERROR_INVALID_VALUE` | needs the retry matrix now in the probe |
| residency change @ fixed VA, 32 MiB | **p50 87.4 µs, p99 98.0 µs** | control-plane cost per slab |
| write/read through a remapped VA | **OK** | the mechanism works |

### 3.1 What this settles

**Zero-copy peer-resident attention is dead.** Llama-3.1-8B carries
32 × 8 × 128 × 2 × 2 B = **128 KiB of KV per token**, so an 8k-token context is 0.98 GiB
and a decode step reads all of it: **3.1 ms from local HBM vs 38.6 ms from a peer.** No
policy makes that acceptable on the hot path.

**But restoring from a peer beats recomputing by 30–43×**, using the measured 27.2 GB/s
against the measured recompute costs in `results/intro/fig_ttft_ctx_sweep.json`:

| ctx | KV size | local restore | **peer restore** | recompute (measured) | **speedup** |
|---|---|---|---|---|---|
| 1 000 | 0.12 GiB | 0.4 ms | **4.8 ms** | 161 ms | **33×** |
| 4 000 | 0.49 GiB | 1.6 ms | **19.3 ms** | 592 ms | **31×** |
| 8 000 | 0.98 GiB | 3.1 ms | **38.6 ms** | 1 203 ms | **31×** |
| 16 000 | 1.95 GiB | 6.3 ms | **77.1 ms** | 2 706 ms | **35×** |
| 32 000 | 3.91 GiB | 12.6 ms | **154.2 ms** | 6 681 ms | **43×** |

This is the sentence the paper wants: **peer GPU HBM has host-DRAM-class transfer
bandwidth (both are PCIe-bound at ~25–27 GB/s) and therefore the same ability to avoid a
recompute, while costing zero host DRAM.** That reframes the contribution away from
"faster than HiCache" (which §7-3 of `research.md` already showed you cannot win) and
onto "same benefit, different resource" — which the 61 GB / 105 GB MemAvailable result
already quantifies.

**Slab size is now a first-order design parameter.** At 87 µs per remap, a 1 GiB
migration costs 32 remaps ≈ 2.8 ms in 32 MiB slabs (7% of the 38.6 ms transfer) but
512 remaps ≈ **44 ms** in 2 MiB slabs — *more than the transfer itself*. Run
`--remap-slabs 2 8 32 128` (now in the probe) to get the real curve; design for the
largest slab the session-keyed allocator can tolerate.

### 3.2 Still to measure (probe updated, ~5 min)

```bash
./scripts/stop.sh
python benchmark/vmm_probe.py --topo --pairs --all-pairs-bw \
    --remap-slabs 2 8 32 128 --slab-mb 32
```

1. **Is there an NVLink bridge at all?** 27.2 GB/s says pair 0-1 went over PCIe.
   `--topo` prints `nvidia-smi topo -m` / `nvlink -s`; `--all-pairs-bw` measures every
   ordered pair. If some pair *is* bridged it will stand out, and the placement policy
   must prefer it — a bridged peer restore would be ~2× faster again.
2. **Does any host location work?** The probe now tries HOST / HOST_NUMA × exportable,
   which distinguishes "driver too old" from "fd export not allowed for host memory".
3. **Remap latency vs slab size** — §3.1.

## 4. Staged implementation plan

Ordered so each stage is independently useful and independently measurable. Stage 0–1
alone is enough evidence for a CAL-length paper; stages 2–3 are the full system.

### Stage 0 — probe (above). ~1 hour.

### Stage 1 — `UnifiedKVSpace`: a VMM-backed slab allocator, standalone. ~1 week.
A small Python class over `cuda.bindings.driver` plus a ~100-line pybind helper to wrap
a mapped VA as a `torch.Tensor` (PyTorch has no public "tensor from raw device pointer"
path; use `torch.utils.dlpack` with a tiny capsule, or `cupy.cuda.UnownedMemory` +
`torch.as_tensor` if cupy is acceptable).

```
class UnifiedKVSpace:
    reserve(n_slabs, slab_bytes)            # one cuMemAddressReserve
    slab_tensor(i) -> torch.Tensor          # stable view; never re-created
    residency(i) -> {LOCAL, PEER(g), HOST}  # current physical location
    place(i, location)                      # cuMemUnmap + cuMemCreate/Map + SetAccess
    export(i) -> fd  /  import_(fd)         # cross-process (2P2D)
```
Unit test it exactly like `shared_park_index.py` was tested: write a pattern, move the
slab through LOCAL → PEER → HOST → LOCAL, verify the bytes and that the tensor's
`data_ptr()` is unchanged. **That invariant is the paper's central mechanism claim.**

### Stage 2 — back SGLang's KV pool with it. ~2–3 weeks, the real cost.
`MHATokenToKVPool` in `python/sglang/srt/mem_cache/memory_pool.py` allocates
`k_buffer[layer]` / `v_buffer[layer]`. Replace that allocation with slab views from
`UnifiedKVSpace`, keeping shapes and dtypes identical so the attention backends and the
allocator (`mem_cache/allocator.py`) are untouched. Constraints:

- **Slab must align to a token boundary.** slab_bytes must be a whole number of
  `bytes_per_token_per_layer`, and a multiple of the Q1 granularity. Your existing
  `SGLANG_KV_PARK_SLAB_TOKENS` design already assumes fixed slabs — reuse the shape.
- **Remap only at a quiescent point.** `cuMemUnmap` is unsafe while a kernel may touch
  the range. Do placement changes at the scheduler step boundary, in the same place the
  current code parks at turn end. Never inside a forward pass.
- **Layer count multiplies the slab count.** 32 layers × K/V = 64 slabs per token range;
  keep a slab *group* as the unit of placement so one migration is one policy decision,
  not 64.

### Stage 3 — the placement policy (the actual research contribution). ~1–2 weeks.
Reuse what already works: the per-GPU pressure telemetry and `/dev/shm` shared index
from `shared_park_index.py`, which already lets one node see every GPU's occupancy.

```
on evict(slab_group, session):
    if peer with headroom:   place(PEER(argmax headroom, prefer NVLink-bridged))
    elif host headroom:      place(HOST)          # overflow only
    else:                    drop (recompute later)

on next turn touches session:
    place(LOCAL)             # ALWAYS migrate in; never execute on peer-resident KV
                             # (27 GB/s vs 333 GB/s -- see §3.1)
                             # issue during the tool-call window, which you already detect
```
Note the difference from the earlier sketch: there is **no "read in place if bandwidth is
adequate" branch**, because §3.1 measured that it never is. The policy question is only
*where does an evicted slab go* and *when is it migrated back*.

### Stage 4 — evaluation. Reuse existing harnesses.
`benchmark/ttft_ctx_sweep.py` (TTFT vs context), `qps_sweep.py` (load), and
`sys_mem_breakdown.py` + `agent_host_pressure.py` (the host-capacity claim). Arms:
HiCache (write_back and L2-only), unified-KV, recompute.

## 5. Hazards specific to this machine

- **Peer bandwidth is PCIe-class, and reachability is not the constraint.** The probe
  measured all 12 ordered pairs reachable, and 0←1 at 27.2 GB/s — i.e. PCIe, not the
  ~50 GB/s an A6000 NVLink bridge would give. So the earlier "NVLink pairs only" worry
  was wrong about *reachability* and right about *bandwidth*: every peer is usable, and
  none of them is fast enough to execute on. Run `--all-pairs-bw` to check whether any
  pair is actually bridged before writing a topology-aware policy.
- **VRAM accounting.** SGLang sizes its pool from `--mem-fraction-static`. A
  VMM-backed pool that can grow into peer HBM breaks the assumption that a GPU's KV
  pool is bounded by its own memory; add explicit per-GPU caps or you will OOM a peer
  that is serving.
- **`cuMemUnmap` + PyTorch caching allocator.** Do not hand VMM memory to PyTorch's
  allocator; own it. Mixing the two is where this kind of work usually breaks.
- **Multi-process handles.** 4 separate server processes → POSIX-fd handles over a unix
  socket, or reuse the existing `/dev/shm` rendezvous pattern to pass fds via
  `SCM_RIGHTS`. `cudaIpcMemHandle` (what the code uses now) does **not** work for VMM
  allocations; it is a different handle type.
- **Host locations may not be available.** The first probe run got
  `CUDA_ERROR_INVALID_VALUE` for a host-located `cuMemCreate`. Most likely cause: the
  probe asked for fd-exportable handles, which host memory need not support; second
  possibility is a driver below CUDA 12.2. The probe now tries HOST / HOST_NUMA ×
  {exportable, not} and reports which combination works. If none does, the overflow tier
  uses `cuMemHostAlloc` plus a separate host pool (what HiCache already does), the
  unified space covers local+peer HBM only, and the paper must say so — it weakens the
  "one space" phrasing but not the result, since §3.1 shows peer HBM is the tier that
  matters and host is only overflow.

## 6. Scope warning, stated plainly

Stages 2–3 are a substantial systems effort (a month-plus at this level of care), and
you already have a *working, measured* mechanism (CUDA IPC + explicit P2P gather, with
results in `results/kv_ts/`, `results/mem/`, `results/intro/`). Two honest options:

- **(a) Reframe what exists.** The current park mechanism already implements "reusable
  KV evicted from one GPU lands in a transiently-idle peer GPU's HBM, host RAM 0". You
  can present it as unified management with an explicit residency table, and cite VMM
  as the mechanism that removes the remaining copy. Lowest risk; the measurements are
  already in hand.
- **(b) Build stage 1 + 2 and make the zero-copy peer-resident read the contribution.**
  Higher risk, considerably stronger paper, because "same virtual address, three
  possible physical locations, policy chooses" is a mechanism claim nobody in this space
  has shipped for KV cache.

If the CAL deadline is near, do (a) now and Stage 1 in parallel — the probe plus the
`data_ptr()`-invariant unit test is a defensible "we validated the mechanism"
paragraph even before full integration.

**§3's measurements shift the balance toward (a).** The one thing VMM would have bought
that the existing IPC mechanism cannot — executing attention directly on peer-resident
KV — is ruled out at 27 GB/s. What remains is that VMM makes the restore a remap plus a
copy into an already-correctly-addressed slab instead of a gather-copy into a separate
pool: real, but an engineering improvement on a mechanism you already have working and
measured, not a new capability. The 30–43× restore-vs-recompute table and the
61 GB → 0 host-DRAM result are the paper's results either way, and both are already in
hand.
