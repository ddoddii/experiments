# KV-aware Unified Memory — 설계 문서

## 0. 목표 문장 (이 표현을 그대로 쓴다)

> **전체 KV 데이터 크기는 유지하면서, 유휴 GPU HBM을 우선 활용하여 CPU DRAM commitment를 줄인다.**

"총 메모리 사용량 감소"가 아니다. KV 총량은 같고, **그 KV가 어디에 물리적으로 상주하느냐**를 바꿔서
**CPU DRAM commitment**를 줄인다.

기여 문장:

> **We propose a KV-aware unified-memory interface that exposes serving semantics unavailable
> to the existing CUDA Unified Memory API.**

**CUDA Unified Memory를 구현하거나 개선한 것이 아니다.** KV-cache semantics를 반영한
**unified-memory placement policy와 runtime interface**를 제안한다. 이 구분을 Abstract·Intro·
Contribution 모두에 명시한다 — "새 CUDA API를 제안한다"고 쓰면 구현하지 않은 것으로 공격받는다
(CUDA에는 이미 managed allocation, preferred-location advice, prefetch API가 있다).

---

## 1. unified memory 프레이밍이 정당한 이유 (측정)

하나의 주소 공간에서 동일한 접근 경로로 측정 (4× RTX A6000, driver CUDA 13.0, NV4 pair (0,1)·(2,3)):

| residency | 대역폭 | local 대비 | 측정 경로 |
|---|---|---|---|
| local HBM | **331.9 GB/s** | 100% | VMM local |
| NVLink peer HBM | **27.2–52.8 GB/s** | 8–16% | VMM peer / P2P |
| **CPU DRAM (pinned)** | **H2D 26.3 / D2H 26.4 GB/s** | 8% | **실제 구현 경로** ★ |
| CPU DRAM (pageable) | H2D 20.6 / D2H 14.1 GB/s | 4–6% | pinned 필수임을 보임 |
| non-NVLink peer HBM | **3.3 GB/s** | 1% | VMM peer |

CPU DRAM 수치는 `benchmark/pinned_host_probe.py`로 **실제 구현이 쓸 pinned host memory 경로**에서
측정했다(H2D 26.3, D2H 26.4 GB/s, 128 MiB–2 GiB에서 안정). `vmm_probe`의 VMM host location 값
26.1 GB/s와 일치하므로 두 측정이 서로를 검증한다. **pageable은 D2H가 14.1 GB/s로 1.9× 나쁘므로
pinned가 필수다.**

**NVLink peer HBM과 CPU DRAM이 같은 대역폭 구간에 있다** (27.2 vs 26.3 = **3.4% 차**). 여기에 layer-wise
transfer, prefill 연산과의 overlap, tool-call window 중 prefetch가 더해지면 end-to-end TTFT 차이는
사라진다.

→ **HBM 아래에는 대역폭 계층이 존재하지 않는다.** "local HBM(빠름) vs 그 밖의 전부(≈PCIe)" 2층뿐이다.
peer GPU HBM과 CPU DRAM을 **분리된 고정 tier로 볼 근거가 없고**, 하나의 placement 대상 집합
(= unified memory)으로 보는 것이 맞다.

**동시에 이 집합은 메모리 계층 직관대로 정렬되지 않는다.** `--all-pairs-bw` 실측 (행 = 읽는 GPU,
열 = 데이터 보유 GPU, GB/s):

|  | 0 | 1 | 2 | 3 |
|---|---|---|---|---|
| **0** | — | 27.2 | **3.3** | **3.3** |
| **1** | **52.8** | — | **3.3** | **3.3** |
| **2** | **3.3** | 26.3 | — | 27.1 |
| **3** | **3.3** | **3.3** | **52.6** | — |

**non-NVLink peer(3.3 GB/s)는 CPU DRAM(24–26)보다 7–8× 느리다.** *"GPU가 항상 CPU보다 빠르다"*는
고정 tier 가정이 **틀렸다.** 올바른 순위는 노드 토폴로지에 의존하며 런타임이 측정해서 알아야 한다.

---

## 2. CUDA Unified Memory의 한계

`cudaMallocManaged()`는 하나의 managed allocation을 만들고 CPU와 GPU가 공통 주소로 접근하게 한다.
실제 page residency는 **접근에 따른 page fault, eviction, prefetch, driver policy**로 바뀐다.
애플리케이션은 `cudaMemAdvise`와 `cudaMemPrefetchAsync`로 힌트를 줄 수 있으나, migration과 eviction의
주체는 CUDA driver다.

### 2.1 UM이 모르는 것 (semantic gap)

- 이 memory range가 **어느 세션의 KV**인지
- **다음 turn에서 재사용될 가능성**이 높은지
- 현재 **어느 GPU가 prefill 또는 decode로 포화**됐는지
- **어떤 peer GPU가 순간적으로 유휴**한지
- **host DRAM을 agent orchestration에 남겨야** 하는지
- 이 KV를 버리면 **얼마나 긴 re-prefill**이 필요한지 (측정: 8k에서 1 203 ms)

UM은 **접근 패턴과 page fault 중심**으로 residency를 관리하고, 우리 시스템은 **KV reuse 가치와
serving pressure 중심**으로 placement를 결정한다. 이것이 semantic gap이다.

### 2.2 메커니즘 차원의 비효율

1. **oversubscription eviction이 host로만 간다.** device 메모리가 부족하면 driver는 페이지를
   **host로** 내린다. "지금 유휴한 peer GPU로 spill"을 표현하는 인터페이스가 없다.
   `cudaMemAdvise(SetPreferredLocation, peer)`는 정적 선호일 뿐 "그때그때 여유 있는 GPU"를 말할 수
   없다. → **우리가 줄이려는 바로 그 CPU DRAM commitment를 UM은 구조적으로 늘린다.**
2. **fault-driven migration이 KV 접근 패턴과 맞지 않는다.** KV 복원은 *알려진 시점*(tool-call
   window)에 *알려진 크기*(세션 prefix 전체)를 **한 번에 벌크 전송**하는 것이 최적이다. UM은 prefill
   중 수천 번의 page fault로 같은 일을 하고, fault handling 뒤로 GPU가 직렬화된다.
3. **attention hot path에서 fault는 수용 불가.** Llama-3.1-8B은 토큰당 128 KiB KV이므로 8k context는
   0.98 GiB이고 decode step마다 전부 읽는다(local 3.2 ms vs peer 38.6 ms). 이 경로에 driver fault를
   넣을 여지가 없다.
4. **thrashing.** CPU/GPU가 번갈아 접근하면 페이지가 ping-pong migration한다. 세션 KV는 turn 경계에서
   접근 주체가 바뀌므로 정확히 이 패턴이다.
5. **★ UM은 process 경계를 넘지 못한다.** PD disaggregation 배포는 **별개 프로세스 4개**(2P+2D)다.
   managed allocation은 한 프로세스의 주소 공간에 속하므로, **한 프로세스의 유휴 GPU 메모리를 다른
   프로세스의 KV 저장에 쓰는 것은 UM으로 표현조차 불가능하다.** 우리가 필요한 것이 정확히 그것이다.
6. **토폴로지 무지.** §1의 3.3 GB/s vs 26 GB/s 차이를 driver policy는 placement에 반영하지 않는다.
   "GPU면 좋다"는 가정으로 non-NVLink peer에 두면 CPU DRAM보다 7–8× 느려진다.
7. **버릴 수 없다.** UM은 managed 데이터의 정합성을 보장해야 하므로 페이지를 폐기할 수 없고, 그래서
   반드시 host에 보존한다. **KV는 버려도 된다**(recompute가 fallback) — UM이 표현하지 못하는,
   우리에게 가장 중요한 semantics다.

---

## 3. 제안: KV-aware placement policy

### 3.1 Unified KV allocation — 하나의 논리적 KV object

각 reusable KV는 GPU와 CPU에 별도로 복제되는 객체가 아니라 **하나의 논리적 KV object**다.

```
KV object
  identifier         : prefix hash / session ID
  size               : N bytes            (= n_tokens x 128 KiB, Llama-3.1-8B)
  current location   : GPU2
  preferred location : GPU2
  fallback location  : CPU
  reuse value        : 재사용 확률 x 회피되는 re-prefill 시간
```

**중요**: 같은 KV를 GPU와 CPU에 **동시에 반드시 복제하지 않는다.** 한 시점에 유효한 copy는 기본적으로
한 location에만 존재한다. (UM의 `cudaMemAdviseSetReadMostly`가 복제를 만드는 것과 대비)

### 3.2 GPU-first placement

KV가 serving cache에서 축출될 때 runtime이 GPU별 pressure를 확인한다:

```
GPU0 pressure: 91%
GPU1 pressure: 88%
GPU2 pressure: 36%     <- 충분히 비어 있음
GPU3 pressure: 69%

preferred location = GPU2
fallback location  = CPU DRAM
```

**Reusable KV placement priority**

| 순위 | location | 조건 | 측정 대역폭 |
|---|---|---|---|
| 1 | **현재 여유가 가장 큰 peer GPU HBM** | headroom ≥ size, 링크 대역폭 최대 | 27–53 GB/s |
| 2 | **다른 가용 GPU HBM** | headroom ≥ size | 3.3–27 GB/s |
| 3 | **CPU DRAM overflow** | GPU에 자리 없음 | 24–26 GB/s |
| 4 | **eviction / recomputation** | 어디에도 자리 없음 | re-prefill 161–6 681 ms |

**★ 순위 1–2는 링크 대역폭으로 정렬한다.** §1에서 non-NVLink peer(3.3)가 CPU DRAM(26)보다 느리므로
"GPU면 무조건 우선"이 아니라, **측정된 대역폭이 CPU DRAM보다 나은 GPU만** 순위 1–2에 든다. 링크 대역폭
표는 기동 시 1회 측정해 캐시한다(`benchmark/vmm_probe.py --all-pairs-bw` 로직 이식). 하드코딩 금지.

CUDA UM API에 비유하면 `cudaMemAdviseSetPreferredLocation` + `cudaMemPrefetchAsync`가 비슷한 역할이나,
**preferred location은 강제적 placement 계약이 아니라 driver hint**다. 제안 인터페이스는 그보다 높은
수준에서 **KV semantics와 location priority를 명시**한다.

### 3.3 GPU pressure 발생 시 재평가 (demotion)

파킹된 GPU가 serving 요청으로 차기 시작하면 그 GPU의 reusable KV를 재평가한다:

```
GPU2 pressure 36% → 87%
```

**reuse value가 낮은 KV부터**:

| 동작 | 조건 |
|---|---|
| `GPU2 → GPU3` | 여유 있는 다른 GPU가 있고, 그 링크가 CPU DRAM보다 빠름 |
| `GPU2 → CPU` | 여유 GPU 없음 → CPU DRAM으로 demote |
| `GPU2 → EVICT` | 재계산이 더 싸거나 CPU DRAM도 포화 |

**serving이 항상 이긴다.** parked KV는 victim cache이므로 언제든 버릴 수 있고 정합성 fallback은
recompute다. 이 규칙이 "빌린 메모리"를 안전하게 만든다.

### 3.4 다음 turn의 재사용 (prefetch)

```
peer GPU HBM → target GPU HBM      (27–53 GB/s → 8k에서 19–39 ms)
CPU DRAM     → target GPU HBM      (24–26 GB/s → 8k에서 40 ms)
```

prefix hash로 현재 residency를 찾고, fetch 후에는 **기존 radix-cache hit과 동일하게** 사용한다.
tool-call window 중에 발행하므로 TTFT에 노출되지 않는다.

**어느 location에서 오든 recompute 대비 30–43× 빠르다**(§5). 그래서 순위 1–3 사이의 선택은 성능
문제가 아니라 **CPU DRAM commitment 문제**다 — 이것이 목표 문장의 근거다.

---

## 4. Runtime interface (제안의 실체)

**새 CUDA API가 아니다.** serving runtime이 KV allocator에게 노출하는 인터페이스다:

```python
# 논리 KV object 등록 (물리 위치 미정)
kv = ukv.declare(identifier=prefix_hash, size=n_bytes,
                 reuse_value=est_reprefill_ms)

# 배치: 정책이 location을 고른다 (§3.2). hint가 아니라 계약.
ukv.place(kv, priority=[IDLE_PEER_GPU, ANY_GPU, CPU_DRAM, EVICT])

# serving 압박 신호 -> 정책이 demote/evict 결정 (§3.3)
ukv.report_pressure(gpu=2, usage=0.87)

# 다음 turn: target GPU로 벌크 prefetch (§3.4)
ukv.fetch(identifier=prefix_hash, target_gpu=0)

# 조회
ukv.residency(prefix_hash)      # -> GPU2 | CPU | EVICTED
```

UM API와의 대응·차이:

| | CUDA Unified Memory | 제안 인터페이스 |
|---|---|---|
| 배치 표현 | `cudaMemAdvise(PreferredLocation)` = **driver hint** | `place(priority=[...])` = **계약**, 실패 시 다음 순위 |
| 이동 트리거 | page fault / access counter | **serving pressure + KV reuse value** |
| eviction 목적지 | **host 고정** | 유휴 peer GPU → CPU → drop |
| 전송 단위 | page (4 KiB–2 MiB) | **세션 KV object 전체** (벌크) |
| 프로세스 경계 | 불가 | **가능** (PD 노드 간 shared index) |
| 데이터 폐기 | 불가 (정합성 필수) | **가능** (victim cache, recompute fallback) |
| 토폴로지 인지 | 없음 | 기동 시 측정한 링크 대역폭 |

마지막 두 행이 핵심이다. UM은 managed 데이터를 버릴 수 없어 반드시 host에 보존한다. **우리는 버릴 수
있으므로 host를 마지막 수단으로 미룰 수 있다.**

---

## 5. 측정된 근거 (전부 확보됨)

| 주장 | 측정값 | 출처 |
|---|---|---|
| HBM 아래 대역폭 계층 없음 | NVLink peer 27.2 vs CPU DRAM 26.1 GB/s (4% 차) | `benchmark/vmm_probe.py` |
| 고정 tier 가정이 틀림 | non-NVLink peer 3.3 GB/s < CPU DRAM 26 GB/s | `--all-pairs-bw` |
| 어디서 가져와도 recompute 압도 | **30–43×** (1k–32k) | `fig_ttft_ctx_sweep.json` + 대역폭 |
| 유휴 GPU HBM 실재 | decode 75% vs prefill 45%, prefill pool **55% 상시 미사용** | `fig1_occupancy_overlay_col` |
| 줄이려는 CPU DRAM commitment | HiCache **61 GB**, write policy·backend 무관 | `results/mem/bd_*.csv` |
| 그 DRAM이 agent stack과 경쟁 | MemAvailable 105 GB → 44 GB | 동일 |
| TTFT 동률 | park 0.605 s vs hicache 0.614 s (turn 1) | `results/perturn` |

---

## 6. 평가 계획

**주 지표 = CPU DRAM commitment** (총 메모리가 아니다).

| 주장 | 지표 | 도구 | arms |
|---|---|---|---|
| **CPU DRAM commitment 감소** | host anon(non-reclaimable) peak/mean, MemAvailable | `sys_mem_breakdown.py` | hicache(write_back) / **KV-aware UM** / recompute |
| KV 총량 유지 | 캐시된 KV 바이트, hit rate 동률 | park survival + hit-rate 카운터 | 동일 |
| TTFT 동률 | per-turn TTFT, QPS sweep | `sglang_BFCL_v3_multi_turn_concurrent.py`, `qps_sweep.py` | 동일 |
| **placement가 실제로 GPU-first** | location별 파킹 바이트 비율(GPU vs CPU) 시계열 | 신규 카운터 | — |
| agent stack 용량 회복 | K_max, tool p99 | `agent_host_pressure.py` | 동일 |

**신규 카운터 하나만 추가하면 된다**: 파킹된 바이트를 location별 집계
(`parked_bytes{location=gpu2}`, `{location=cpu}`). 이걸로 "GPU-first가 실제로 작동했고 CPU DRAM
commitment가 그만큼 줄었다"를 직접 보인다.

---

## 7. 구현 범위

### 7.0 ★ CPU DRAM overflow를 어떻게 할당하는가 (측정으로 확정)

`benchmark/pinned_host_probe.py` 결과:

| 항목 | 측정값 |
|---|---|
| pinned 할당 (cold) | **~480–640 ms/GiB** — 750 MiB 세션이면 **~360 ms** |
| pinned 할당 (warm, 같은 크기 재요청) | **~0.0 ms** — PyTorch caching host allocator |
| `del` 후 RSS | **반환 안 됨** (+1024 MB 유지) |
| **`torch._C._host_emptyCache()`** | **존재하고 동작함** → RSS +1024 MB → **+0 MB** |
| raw `cudaHostAlloc` / `cudaFreeHost` | alloc 479–620 ms/GiB, free 130–400 ms, **OS 반환됨** |
| 전송 (750 MiB, H2D 26.3 GB/s) | 28.6 ms |

**해석**: cold pin(360 ms)은 전송(28.6 ms)의 **12배**라서 park마다 새로 pin하면 못 쓴다. 그러나
warm은 0 ms이고, `_host_emptyCache()`로 **실제 반환도 가능**하다.

**→ 결정: (a') caching allocator를 통한 on-demand.**

1. host park는 `torch.empty(n, pin_memory=True)`로 받는다 (warm이면 ~0 ms).
2. **★ host park 크기를 몇 개 버킷으로 양자화한다** (예: 256 MiB 배수 또는 2의 거듭제곱).
   세션 크기가 매번 다르면 매번 cold pin(480 ms/GiB)이 되어 (a')가 무너진다. **이 요구사항은
   측정에서 직접 도출된 설계 제약이다.**
3. 파킹된 host 집합이 줄어들 때(evict/demote 누적) **`torch._C._host_emptyCache()`를 명시적으로
   호출**해 RSS를 OS에 반환한다 → **committed host DRAM이 cached KV를 추적한다.**
   - 주의: 이 API는 **프로세스 전역**이라 SGLang의 다른 pinned 버퍼(weight loading 등)까지 비운다.
     매 eviction마다 부르지 말고 **히스테리시스**를 둔다(미사용 cached pinned가 임계 초과 시에만).
   - 이 private API가 없는 torch 빌드를 위해 raw `cudaHostAlloc` 경로를 fallback으로 둔다.
4. **pinned 필수** — pageable은 D2H 14.1 GB/s로 1.9× 느리다.

이 설계로 host commitment는 **RSS로 측정**되고, 그 값이 실제 캐시된 host KV를 따라간다.

### 7.1 구현 항목

**한다 (기존 자산 재사용, 1–2주):**
- `_ParkPool` 선택 로직을 §3.2의 **대역폭-정렬 우선순위**로 교체 (현재는 pressure+headroom만 봄)
- **CPU DRAM overflow 경로 추가 (순위 3)** — 지금은 GPU 아니면 drop이다. 이게 없으면 "GPU-first"를
  주장할 수 없다(비교 대상이 없으므로). **가장 중요한 신규 구현.** 할당 방식은 §7.0의 (a').
- host park 크기 **버킷 양자화** + `_host_emptyCache()` 히스테리시스 (§7.0-2,3)
- 기동 시 링크 대역폭 측정 + 캐시 (`--all-pairs-bw` 로직 이식)
- §3.3 demotion: pressure 신호에 반응해 GPU→GPU / GPU→CPU / drop
- location별 카운터 (§6), shared index에 `location` 필드

**안 한다 (논문에 명시):**
- CUDA Unified Memory 구현·수정, 새 CUDA driver API
- CUDA VMM(`cuMemCreate`/`cuMemMap`) 기반 재작성 — §8
- attention 커널 수정 (peer/CPU 상주 KV로 직접 attention하지 않음, §2.2-3)

---

## 8. VMM은 왜 안 쓰는가 (질문받을 것)

C0 probe로 CUDA VMM 경로를 실측했고(`benchmark/vmm_probe.py`), **범위 대비 이득이 없다**고 판단했다:

- VMM의 고유 능력은 "같은 VA에 물리 위치만 바꿔 kernel 포인터 유지"인데, §2.2-3에서 peer/CPU 상주
  KV로 직접 attention하지 않기로 했으므로 필요가 없다. fetch는 어차피 벌크 복사다.
- 제어 비용이 **mapping 개수에 비례**한다 (같은 128 MiB에서 handle 1개 **90 µs** vs 2 MiB 페이지
  64개 **5 295 µs**, 59×). 큰 handle로 회피 가능하지만 그러면 PyTorch가 소유한 KV pool과 별도
  할당자를 운영해야 한다.
- **단 한 가지는 VMM만 가능하다**: 빌려준 GPU에 물리 페이지를 *실제로* 반환하는 것. 정적 park pool은
  논리적 무효화만 가능하다 → §9의 정직성 항목.

probe 수치는 **UM/VMM 대안을 실측으로 배제했다는 근거**로 Discussion 한 문단에 남긴다. 이게 "왜 더
낮은 레벨로 안 갔나"에 대한 답이 된다.

---

## 9. 정직하게 다뤄야 할 것

- **park pool은 여전히 정적 VRAM 예약이다** (`_ParkPool`이 `torch.zeros`, 26 GB @ 200k tokens).
  목표 문장이 "CPU DRAM commitment 감소"이므로 주장은 성립하지만, 리뷰어는 *"host에서 VRAM으로 옮긴
  것 아닌가"*를 묻는다. **답: host와 device commitment를 둘 다 보고한다.**
  - CPU DRAM: agent orchestration·scheduling·retrieval·tool 실행과 **경쟁하는** 자원
    (측정: MemAvailable 105 → 44 GB)
  - GPU HBM: Fig.1에서 **55%가 상시 미사용**인, 어차피 버려지는 용량
  - **같은 바이트가 아니다.** 이 비대칭이 논거이고, 표로 정직하게 제시하면 방어된다.
- **CPU DRAM을 0으로 만들지 않는다.** 순위 3(overflow)이 설계에 포함되므로 부하가 높으면 CPU를 쓴다.
  주장은 "**줄인다**"이며 "없앤다"가 아니다 — 목표 문장이 이미 정확하다.
- **정책이 이기지 못하는 구간이 있다.** 모든 GPU가 동시에 포화면 GPU-first가 할 일이 없고 HiCache와
  같아진다. 이 구간을 평가에 포함해 보고한다(negative result가 설계 정당화에 쓰인다).
- **process 경계**: §2.2-5는 UM의 한계이면서 우리 구현의 요구사항이다. 기존 CUDA IPC + `/dev/shm`
  shared index가 이미 이 문제를 풀고 있으므로 systems 기여로 명시한다.
- **CPU DRAM 대역폭은 실제 경로로 재측정 완료** — pinned H2D 26.3 / D2H 26.4 GB/s
  (`pinned_host_probe.py`). VMM host location 값 26.1 GB/s와 일치하므로 논문 표에 26.3을 쓰고
  각주로 두 경로가 일치함을 밝힌다.
- **host commitment 지표는 RSS다.** `_host_emptyCache()`가 제어하는 것이 RSS이므로, host 측
  commitment는 `sys_mem_breakdown.py`의 `proc_rss` / 전역 `AnonPages`로 보고한다. `VmLck`/`Mlocked`는
  **0으로 나온다** — CUDA driver가 `mlock()` 외 경로로 pin하기 때문이다. 이걸 모르고 `Mlocked`를
  보면 "pinned 메모리가 없다"고 오판하게 되므로 논문·스크립트 모두에서 RSS를 쓴다.
