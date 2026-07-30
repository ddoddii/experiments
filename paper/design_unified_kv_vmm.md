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

Three consequences that matter for this paper:

1. **The pointer never changes.** A slab can move from local HBM to peer HBM to host
   DRAM while attention kernels, the radix index, and the KV index tensors keep the
   same addresses. Compare with the current park implementation, where a peer pool is a
   *separate tensor* and a hit requires an explicit gather-copy back into the local
   pool (`_p2p_gather` / `_gather_copy_from_peer` in `idle_kv_parking.py`).
2. **Peer residency can be zero-copy.** `cuMemSetAccess` for the local device on a
   peer-backed slab makes it directly readable over NVLink. "Migration" becomes
   "remap", and the copy disappears — *if* the pair has a P2P path (see §5).
3. **Host is just another location.** `CU_MEM_LOCATION_TYPE_HOST_NUMA` (CUDA ≥ 12.2)
   makes the overflow tier a location in the same space rather than a separate
   subsystem, which is literally the "residency, not tier" reframing.

Precedent that this composes with PyTorch: PyTorch's own
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` allocator is built on these calls.
Read `c10/cuda/CUDACachingAllocator.cpp` (`ExpandableSegment`) — it is the closest
working model for what you need to write.

## 3. Run the feasibility probe FIRST

`benchmark/vmm_probe.py` answers the four questions that decide whether any of this is
worth building, in ~2 minutes and with zero SGLang changes:

```bash
conda activate sglang
./scripts/stop.sh                       # so the probe sees free GPUs
python benchmark/vmm_probe.py --pairs --slab-mb 32
```

| probe | why it gates the design |
|---|---|
| **Q1** allocation granularity | the KV slab size must be a multiple of it (expect 2 MiB) |
| **Q2** remap latency at a fixed VA | the per-migration control cost. Must be ≪ the prefill it avoids (~1200 ms at 8k tokens from `results/intro/fig_ttft_ctx_sweep.json`) |
| **Q3** peer-backed read bandwidth vs local | decides zero-copy-remote vs copy-then-use. Also prints the `cuDeviceCanAccessPeer` matrix |
| **Q4** host-backed read bandwidth | the cost of the overflow tier, and whether `HOST_NUMA` works on this driver |

**Do not skip this.** If Q3 shows no P2P path outside the NVLink pair, or Q2 shows
remap costing milliseconds per slab, the design changes shape — better to learn that
before touching `memory_pool.py`.

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
    if peer with headroom and P2P path:  place(PEER(argmax headroom))
    elif host headroom:                  place(HOST_NUMA(local socket))
    else:                                drop (recompute later)
on next turn touches session:
    if residency == PEER and peer bandwidth is adequate:  read in place (no migration)
    else:                                                  place(LOCAL)   # prefetch
```
The prefetch is issued during the tool-call window — you already have that trigger.

### Stage 4 — evaluation. Reuse existing harnesses.
`benchmark/ttft_ctx_sweep.py` (TTFT vs context), `qps_sweep.py` (load), and
`sys_mem_breakdown.py` + `agent_host_pressure.py` (the host-capacity claim). Arms:
HiCache (write_back and L2-only), unified-KV, recompute.

## 5. Hazards specific to this machine

- **A6000 P2P is pairwise.** NVLink bridges 0-1 and 2-3; across pairs it is PCIe P2P at
  best. `research.md` already notes the NVLink pair. So "place on the peer with the most
  headroom" must be **topology-aware**: prefer a bridged peer, and treat a non-bridged
  peer as roughly host-class bandwidth. Q3 in the probe measures this — do not assume.
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
- **Driver/CUDA version.** `HOST_NUMA` needs CUDA ≥ 12.2. Check on server17 before
  designing the overflow tier around it; the fallback is `cuMemHostAlloc` plus a
  separate host pool (i.e. what HiCache already does), which weakens the "one space"
  claim but does not break the paper.

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
