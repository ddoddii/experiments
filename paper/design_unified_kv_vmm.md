# Unified KV Memory — 설계 문서 (Design C: park 영역만 CUDA VMM으로)

## 0. 목표 한 줄

> KV offload 공간을 **미리 예약된 host pool**이 아니라, **그 순간 자리가 있는 곳(유휴 GPU HBM 우선,
> host는 overflow)에서 물리 페이지를 커밋하는 하나의 논리 공간**으로 관리해서, 동일 TTFT에서
> **시스템 전체(host+device) 예약 메모리를 줄인다.**

전제(측정 완료): peer GPU 읽기 **27.2 GB/s** ≈ host PCIe **~25 GB/s**. 같은 PCIe 대역폭이고,
layer-wise transfer와 tool-call window로 가려지므로 **CPU에서 가져오나 유휴 GPU에서 가져오나
TTFT는 같다.** 따라서 KV가 어디 있느냐는 성능 문제가 아니라 **자원 예약 문제**다.

---

## 1. 문제: 정적 예약(static reservation)

`torch.zeros(N, ...)`는 VA와 PA를 **동시에** 커밋한다. 그래서 두 시스템 모두 같은 병을 갖는다.

| | 예약 위치 | 예약량 | 캐시 점유율과 무관하게 잡히는가 |
|---|---|---|---|
| SGLang HiCache | host DRAM (`--hicache-ratio`) | **61 GB** (측정) | **예** — 5% 찼든 95% 찼든 |
| 현재 park 구현 | GPU VRAM (`_ParkPool`, `idle_kv_parking.py:187`) | **26 GB** (`PARK_POOL_TOKENS=200k`) | **예** |
| 목표 | — | **0** (사용량만 커밋) | 아니오 |

현재 구현은 예약 문제를 **host DRAM에서 VRAM으로 옮긴 것**이다. 총량은 61→26 GB로 줄었지만 0이
아니고, 무엇보다 **"유휴 GPU를 빌린다"가 아니라 "GPU에서 26 GB를 영구히 뗀다"**가 되어 있다.

---

## 2. 왜 CUDA Unified Memory(UVM)가 아닌가

Intro 초안의 "CUDA Unified Memory의 공통 주소 공간과 migration mechanism을 활용" 표현은 그대로
쓰면 리뷰어에게 걸린다.

1. **UVM은 peer GPU로 spill할 수 없다.** device oversubscription 시 driver는 **host로** evict한다.
   `cudaMemAdvise(SetPreferredLocation, peer)`는 정적 선호일 뿐, "지금 여유 있는 peer로 보내라"를
   표현하는 API가 없다. **이 논문의 핵심 결정을 driver가 쥐고 있다.**
2. **attention 커널 안의 page fault는 수용 불가.** KV 페이지가 커널 실행 중 fault하면 GPU가
   driver fault handling 뒤에 직렬화된다.
3. migration **시점**도 access-counter heuristic이 소유한다.

그러므로 UVM은 **대조군**으로 쓴다. 이게 더 강한 문장이다:

> 우리는 unified KV memory space라는 **추상**은 제공하되 residency와 migration을 **명시적으로**
> 제어한다. CUDA Unified Memory의 암묵적 정책은 "일시적으로 유휴한 peer GPU로 spill"을 표현할 수
> 없기 때문이다.

---

## 3. server17 측정 결과 (`benchmark/vmm_probe.py`)

4× RTX A6000, driver 확인, 서버 정지 상태:

4× RTX A6000, **driver CUDA 13.0**, 서버 정지 상태. NVLink: **GPU0-GPU1 = NV4, GPU2-GPU3 = NV4**
(링크당 14.06 GB/s × 4 = 56.25 GB/s/방향), pair 간은 NODE/PHB(PCIe).

| 항목 | 측정값 | 해석 |
|---|---|---|
| allocation granularity | **2.00 MiB** (min = recommended) | 물리 handle은 2 MiB의 배수 |
| P2P 도달성 | 12개 순서쌍 전부 1 | **대역폭과 무관** — 아래 참조 |
| local HBM read | **331–332 GB/s** | 기준선 |
| **NVLink peer read** | **27.2 / 52.8 GB/s** (방향 비대칭) | NV4 pair 내부 |
| **non-NVLink peer read** | **3.3 GB/s** | pair 간 PCIe — **host보다 느리다** |
| **host DRAM read** (`CU_MEM_LOCATION_TYPE_HOST`) | **23.7–26.1 GB/s** | **동작 확인** |
| residency 변경, **handle 1개** | **p50 87–90 µs, 크기 무관** (2/8/32/128 MiB) | 호출당 비용 |
| residency 변경, **handle N개** | **페이지당 ~80 µs, 배치해도 안 줄어듦** | §6 — 설계 결정적 |

### 3.0 ★ 결과 1 — HBM 아래에 대역폭 계층이 없다

**같은 주소 공간, 같은 접근 경로**로 측정:

| residency | 대역폭 | local 대비 |
|---|---|---|
| local HBM | **331.9 GB/s** | 100% |
| NVLink peer HBM | **27.2–52.8 GB/s** | 8–16% |
| **host DRAM** | **23.7–26.1 GB/s** | 7–8% |
| non-NVLink peer HBM | **3.3 GB/s** | 1% |

**NVLink peer와 host DRAM이 같은 구간에 있다**(27.2 vs 26.1 — 4% 차; NVLink 유리 방향이면 2×).
*"CPU에서 가져오나 유휴 GPU에서 가져오나 같다"*는 논문 전제의 직접 증거이고, **CPU를 HBM 아래의
별도 tier로 둘 대역폭 근거가 없다**는 뜻이다 — **local HBM(빠름) vs 그 밖의 전부(≈PCIe)** 2층뿐이다.
기존 시스템이 host DRAM을 별도 tier로 예약하는 것은 대역폭 근거 없이 **자원만 예약하는 구조**다.

→ Intro에 이 표를 넣는다. 전제가 가정이 아니라 측정이 된다.

### 3.0b ★ 결과 2 — 위치 순위가 메모리 계층 직관과 다르다

`--all-pairs-bw` (행 = 읽는 GPU, 열 = 데이터 보유 GPU, GB/s):

|  | 0 | 1 | 2 | 3 |
|---|---|---|---|---|
| **0** | — | 27.2 | **3.3** | **3.3** |
| **1** | **52.8** | — | **3.3** | **3.3** |
| **2** | **3.3** | 26.3 | — | 27.1 |
| **3** | **3.3** | **3.3** | **52.6** | — |

- NV4 pair 내부: **27–53 GB/s** (방향 비대칭 2× — remote read 방향 효과로 추정, 확정에는
  write 방향/`cuMemcpyPeerAsync` 추가 측정 필요)
- **pair 간: 3.3 GB/s** — host DRAM(24–26 GB/s)보다 **7–8× 느리다**

**따라서 "GPU 먼저"는 틀린 정책이다.** 올바른 순위:

> **NVLink peer (27–53) > host DRAM (24–26) ≫ non-NVLink peer (3.3) > recompute**

(non-NVLink peer도 8k에서 297 ms로 recompute 1203 ms보다는 4× 낫지만, host를 쓰는 게 낫다.)

이건 논문에 유리한 발견이다: **location들이 메모리 계층이 시사하는 순서로 정렬되지 않기 때문에
"명시적 residency 정책"이 필요하다.** 고정 tier 구조로는 이 순위를 표현조차 할 수 없다.

### 3.1 이 측정이 확정한 두 가지

**(a) peer-resident KV로 attention을 직접 돌리면 안 된다.** Llama-3.1-8B은 토큰당
32층 × 8 kv-head × 128 dim × K/V × 2 B = **128 KiB**. 8k context = 0.98 GiB이고 decode step마다
전부 읽는다 → **local 3.1 ms vs peer 38.6 ms.** peer는 **restore 소스**이고 실행 시 residency가
아니다. (정책에서 "대역폭 충분하면 제자리 읽기" 분기는 삭제)

**(b) 그런데 peer restore는 recompute를 30–43× 이긴다.** 측정 27.2 GB/s × 측정 recompute
(`results/intro/fig_ttft_ctx_sweep.json`):

| ctx | KV 크기 | local restore | **peer restore** | host restore | recompute (측정) | **speedup** |
|---|---|---|---|---|---|---|
| 1 000 | 0.12 GiB | 0.4 ms | **4.8 ms** | 5.0 ms | 161 ms | **33×** |
| 4 000 | 0.49 GiB | 1.6 ms | **19.3 ms** | 20.1 ms | 592 ms | **31×** |
| 8 000 | 0.98 GiB | 3.2 ms | **38.6 ms** | 40.2 ms | 1 203 ms | **31×** |
| 16 000 | 1.95 GiB | 6.3 ms | **77.2 ms** | 80.3 ms | 2 706 ms | **35×** |
| 32 000 | 3.91 GiB | 12.7 ms | **154.5 ms** | 160.7 ms | 6 681 ms | **43×** |

→ **peer GPU HBM은 host DRAM과 동급(4% 차)이므로 recompute 회피 능력도 동급이고, host DRAM은
0을 쓴다.** 기여를 "HiCache보다 빠르다"(§7-3에서 이미 못 이긴다고 나온 축)가 아니라
**"동일 이득, 다른 자원"**으로 정의한다.

---

## 4. VA/PA 분리가 주는 이득 4가지

### ① "예약하지 않는다"를 가능하게 하는 유일한 메커니즘
- **VA = 광고하는 논리 용량** (공짜, 물리 HBM보다 커도 됨)
- **PA = 실제 커밋** (실제 캐시된 KV만큼)

GPU마다 "local HBM + 모든 peer + host"를 합친 크기의 VA를 예약하고, 물리 페이지는 그때그때 자리
있는 곳에서 커밋한다. **unified space가 비유가 아니라 문자 그대로**가 되고, residency는 각 페이지의
속성이 된다.

### ② 가장 강한 논거 — 물리적으로 돌려줄 수 있어야 빌려주기가 진짜다
정적 텐서로는 "압박 시 회수"가 **불가능**하다. 논리적으로 무효화해도 VRAM은 그대로 점유되어 빌려준
GPU는 메모리를 실제로 돌려받지 못한다. `cuMemUnmap` + `cuMemRelease`는 물리 페이지를 그 GPU의 free
pool로 **실제로 반환**한다.

> **Lending is only real if you can physically give it back.**

### ③ 새 지표 — commitment vs cached (커밋 효율)
이 분야 논문은 항상 pool 크기만 보고한다. VA/PA를 분리하면 이 표를 낼 수 있다:

| | 논리 용량 | **물리 커밋** | 실제 캐시된 KV | 커밋 효율 |
|---|---|---|---|---|
| HiCache | 61 GB host | **61 GB (항상)** | 예: 12 GB | **20%** |
| park (정적) | 26 GB VRAM | **26 GB (항상)** | 12 GB | 46% |
| **제안** | 100 GB+ | **12 GB** | 12 GB | **100%** |

`benchmark/sys_mem_breakdown.py`로 이미 측정 가능하다(GPU HBM + host anon + page cache 분리 기록).

### ④ CPU가 tier가 아니라 location이 된다
`CU_MEM_LOCATION_TYPE_HOST`로 같은 VA를 host DRAM으로 backing할 수 있다. 그러면 *"CPU와 GPU를 같은
tier로 관리"*가 유추가 아니라 구현이다 — **하나의 주소 공간, 페이지마다 location 필드가 device-N
또는 host.** 이게 thesis statement 그 자체다. (단 §9의 host location 확인 선행)

---

## 5. Design C — park/overflow 영역만 VMM으로

serving KV pool은 **그대로 둔다**(PyTorch 텐서, 현재와 동일). park 공간만 VA 예약 + on-demand 물리
커밋으로 바꾼다.

```
# 초기화 (한 번)
park_va[gpu] = cuMemAddressReserve(아주 큰 크기)      # 물리 0, 공짜

# 파킹: 축출 위험 prefix를 어딘가에 물리 커밋
def park(session, n_tokens):
    loc  = choose_location(n_tokens)     # §5.1 순위표
    size = round_up(n_tokens * 128KiB, 2MiB)
    h    = cuMemCreate(size, prop(loc))  # ★ handle 1개 (§6.3) -- 페이지로 쪼개지 말 것
    cuMemMap(park_va + off, size, 0, h, 0)
    cuMemSetAccess(park_va + off, size, descs)        # 합계 ~90 us
    p2p_copy_in(...)                     # 기존 IPC/P2P 경로 재사용
    shared_index.put(hash, (loc, off, size))          # location 기록

# 다음 turn: 항상 target GPU로 가져온다 (제자리 attention 금지, §3.1a)
def fetch(session):  copy_to_local(...)  # tool-call window 중 prefetch

# 성장: handle은 리사이즈 불가 -> turn마다 뒤에 이어 붙인다 (§6.3)
def grow(session, extra_tokens):  # handle 하나 더, ~90 us
    ...

# 압박 시 회수: 물리 페이지를 실제로 반환
def reclaim(gpu, need_bytes):
    for slab in lru_parked(gpu):
        cuMemUnmap(slab.va, slab.size); cuMemRelease(slab.h)  # -> free pool 복귀
        shared_index.invalidate(slab)
        if freed >= need_bytes: break
```

### 5.1 `choose_location()` — 측정된 대역폭 순위로 (§3.0b)

**"GPU 먼저"가 아니다.** non-NVLink peer(3.3 GB/s)는 host DRAM(24–26)보다 7–8× 느리다.

| 순위 | location | 대역폭 | 조건 |
|---|---|---|---|
| 1 | **NVLink-bridged peer** with headroom | 27–53 GB/s | `nvidia-smi topo`의 NV# 쌍 |
| 2 | **host DRAM** | 24–26 GB/s | 프로세스 로컬만 (§11: fd export 불가) |
| 3 | non-NVLink peer with headroom | 3.3 GB/s | host도 꽉 찼을 때만 |
| 4 | **drop** → 다음 turn recompute | — | victim cache 정합성 fallback |

링크 대역폭 표는 **기동 시 1회 측정**해서 캐시한다(`vmm_probe`의 `--all-pairs-bw` 로직 재사용).
하드코딩하지 말 것 — 노드마다 토폴로지가 다르다.

**Design C가 옳은 이유**
- ✅ 26 GB 정적 예약 제거 (①③)
- ✅ 압박 시 물리적 반환 (②)
- ✅ host가 같은 VA의 location 하나 (④)
- ✅ **`MHATokenToKVPool`·attention 커널 손 안 댐** — park 데이터는 어차피 복사해서 쓰고,
  §3.1(a) 때문에 제자리 attention은 애초에 금지
- ✅ 기존 자산 전부 재사용: shared index(`shared_park_index.py`), per-GPU telemetry, session slab
  allocator, P2P copy 경로, /dev/shm rendezvous

**규모**: 2–3주. (serving pool까지 VMM으로 바꾸는 안은 한 달+이고, 필요 없다.)

---

## 6. ★ 결과 3 — 배치는 통하지 않는다. **park slab당 물리 handle 1개**를 써야 한다

### 6.1 handle 1개짜리 slab: 크기와 무관하게 ~90 µs

| slab (handle 1개) | p50 (µs) | remaps/GiB | **µs/GiB** | 전송(38.6 ms/GiB) 대비 |
|---|---|---|---|---|
| 2 MiB | 88.7 | 512 | 45 424 | **118%** ✗ |
| 8 MiB | 89.2 | 128 | 11 419 | 30% |
| 32 MiB | 89.6 | 32 | 2 868 | 7% |
| 128 MiB | 90.4 | 8 | **724** | **1.9%** ✓ |

### 6.2 배치 실측 — 효과가 **없다**

2 MiB 페이지 N장을 매핑하고 `cuMemSetAccess`/`cuMemUnmap`을 range 단위로 **1회씩만** 호출:

| region | pages | total µs | map | **access** | unmap | **µs/GiB** |
|---|---|---|---|---|---|---|
| 2 MiB | 1 | 995 | 4 | 956 | 35 | 509 500 |
| 16 MiB | 8 | 664 | 12 | 420 | 231 | 42 490 |
| 128 MiB | 64 | 5 295 | 69 | 3 473 | 1 749 | 42 361 |
| 768 MiB | 384 | 32 054 | 410 | 21 137 | 10 479 | 42 739 |
| 1 024 MiB | 512 | 42 240 | 512 | 27 875 | 13 905 | 42 240 |

**µs/GiB가 ~42 000으로 평평하다 — 배치해도 전혀 줄지 않는다.** `cuMemSetAccess`와 `cuMemUnmap`은
range를 받지만 **드라이버가 range 안의 mapping을 하나하나 순회**한다:
setAccess 3 473/64 = **54 µs/페이지**, unmap 1 749/64 = **27 µs/페이지** (vAttention Table 3의
38 + 34 µs와 일치).

**결정적 비교 — 같은 128 MiB 범위:**

| 물리 backing | 제어 비용 | 배수 |
|---|---|---|
| **handle 1개 (128 MiB)** | **90 µs** | 1× |
| handle 64개 (2 MiB × 64) | 5 295 µs | **59×** |

→ 앞서 권고한 (B) 배치는 **폐기한다.** 비싼 것은 호출 횟수가 아니라 **mapping 개수**다.

### 6.3 그래서 설계: **park slab = 물리 handle 1개**

granularity는 2 MiB가 최소값일 뿐이고, `cuMemCreate`는 **그 배수의 아무 크기나** 만들 수 있다.
그러므로 세션의 KV 크기에 맞춰 **큰 handle 하나**를 만든다:

| 항목 | 값 |
|---|---|
| 6000-token 세션 KV | 750 MiB → **handle 1개** (2 MiB로 라운딩, 낭비 ≤2 MiB) |
| 제어 비용 (map+setAccess+unmap) | **~90 µs** |
| 전송 비용 (27 GB/s) | 28 ms |
| **제어 오버헤드** | **0.3%** ✓ |
| 회수(unmap+release) | ~90 µs, 물리 페이지 즉시 반환 |

**세션 성장(in-place grow)**: handle은 크기 변경이 안 되므로, turn마다 **새 handle을 VA 뒤에 이어
붙인다.** 5턴 대화 = handle 5개 = 제어 450 µs. 여전히 무시 가능하다.

**내부 파편화도 문제가 아니다** — 고정 slab 크기가 아니라 세션 실제 크기로 handle을 만들기 때문에
낭비는 handle당 최대 2 MiB(0.3%)다. 이는 (A) 고정 128 MiB slab(최대 127 MiB 낭비)보다도 좋다.

### 6.4 2 MB granularity는 우리에게 부담이 아니다

vAttention이 64 KB 페이지용 CUDA 확장까지 만든 이유는 *짧은 요청*의 내부 파편화였다. 우리는 수천
토큰짜리 대화 prefix를 통째로 파킹하므로 2 MB(= 레이어당 1024 토큰)면 충분하다.
**vAttention의 최대 구현 부담을 우리는 지지 않는다.**

---

## 7. vAttention과의 차별점 (논문에 반드시 명시)

| | vAttention (ASPLOS'25) | 본 연구 |
|---|---|---|
| VMM 사용 목적 | **한 GPU 안에서** paged-attention 커널 회피 + over-allocation 제거 | **여러 GPU + host를 걸쳐** 탄력적 차용 용량 풀 |
| 절약 대상 | 요청 내부 파편화 | **정적 예약된 offload tier 전체** |
| location 다양성 | local device only | device-N / host, 정책이 명시 선택 |
| 회수 | 요청 종료 시 deferred reclamation | **다른 GPU의 serving 압박에 의한 회수** |
| granularity 부담 | 64 KB 확장 필요(짧은 요청) | 2 MB로 충분(긴 prefix) |

---

## 8. 구현 단계

| 단계 | 내용 | 규모 |
|---|---|---|
| **C0** | probe 보강 실행(§9) — host location / NVLink / remap-vs-slab 확정 | 10분 |
| **C1** | `VmmParkSpace` 독립 클래스 + 단위 테스트: 패턴 write → LOCAL→PEER→HOST→LOCAL 이동 → 바이트 검증 + **커밋된 물리량이 사용량에 비례**함을 확인 | 1주 |
| **C2** | `_ParkPool`의 `torch.zeros` 제거 → `VmmParkSpace` 사용. 배치 setAccess. shared index에 location 필드 추가 | 1주 |
| **C3** | `choose_location()` + `reclaim()` 정책. peer headroom telemetry는 기존 것 재사용 | 3–5일 |
| **C4** | 평가: 기존 harness 전부 재사용 (§10) | 3일 |

**C1의 단위 테스트가 논문의 메커니즘 주장 그 자체다** — "논리 용량은 크고 물리 커밋은 사용량만".

---

## 9. C0 진행 상황

**확정됨:**
- ✅ **host location 동작** — `CU_MEM_LOCATION_TYPE_HOST` + non-exportable handle로 성공,
  **26.1 GB/s**. 첫 실행의 `INVALID_VALUE`는 driver 버전이 아니라 **host location에도 POSIX fd
  exportable handle을 요청한 것**이 원인이었다. → §4의 ④가 성립한다: **CPU는 같은 주소 공간의
  location이다.** 논문에서 "one address space spanning GPU HBM and host DRAM"을 그대로 주장 가능.
  - 단, host 페이지는 fd export가 안 되므로 **multi-process에서 host 슬랩을 공유할 수 없다.**
    2P2D에서 host overflow는 **각 프로세스가 자기 것만** 갖고, 공유가 필요한 건 peer HBM으로
    한정한다(§11에 반영).
- ✅ **peer 27.2 vs host 26.1 GB/s (4% 차)** — §3.0, 논문 전제의 직접 증거.
- ✅ **remap 비용은 호출당 (87 µs, slab 크기 무관)** — §6.

- ✅ **NVLink NV4가 (0,1)·(2,3)에 존재** — pair 내부 27–53 GB/s, pair 간 3.3 GB/s.
  `choose_location()`은 **반드시 topology-aware**여야 한다(§5.1). driver CUDA 13.0.
- ✅ **배치는 통하지 않는다** — µs/GiB가 42 000으로 평평. 비용은 **mapping 개수**에 비례.
  → **park slab당 물리 handle 1개** (§6.3). 같은 128 MiB에서 90 µs vs 5 295 µs (59×).

**C0 종료. 남은 미결(선택):**
1. **peer read 방향 비대칭 2×** (1←0: 52.8 vs 0←1: 27.2, 같은 NV4 쌍). remote read 방향 효과로
   추정되나 확정 안 됨. `cuMemcpyPeerAsync` 또는 write 방향 측정으로 확인 가능. **설계에는 영향
   없음**(둘 다 host보다 빠름) — 정책이 쌍별 실측값을 캐시하므로 자동 반영된다.
2. `HOST_NUMA` 동작 여부 — server17은 단일 NUMA(`nvidia-smi topo`의 NUMA Affinity 전부 0)이므로
   **불필요**.
3. host 대역폭이 실행 간 23.7–26.1 GB/s로 흔들린다(공유 서버 노이즈). 논문에는 범위로 보고.

---

## 10. 평가 계획 (기존 harness 재사용)

| 주장 | 측정 도구 | arms |
|---|---|---|
| TTFT 동률 | `benchmark/sglang_BFCL_v3_multi_turn_concurrent.py`, `qps_sweep.py` | hicache(write_back) / unified-KV / recompute |
| **총 예약 메모리 감소 + 커밋 효율** | `sys_mem_breakdown.py` (host anon/page cache/GPU HBM) + 신규 "committed vs cached" 카운터 | hicache / park(정적) / **unified-KV** |
| agent stack 용량 회복 | `agent_host_pressure.py` (K_max, tool p99) | 동일 |
| restore vs recompute | `ttft_ctx_sweep.py` + §3.1b 표 | — |
| 회수가 실제로 동작 | 신규: 빌려준 GPU에 부하 인가 → `nvidia-smi` free VRAM이 실제로 회복되는지 | park(정적) vs unified-KV |

마지막 행이 ②를 증명하는 실험이다. **정적 park pool은 이 테스트에서 반드시 실패한다** — VRAM이
돌아오지 않으므로. 이게 negative control이다.

---

## 11. 위험 요소

- **peer 대역폭이 PCIe급**이고 도달성은 문제가 아니다(12쌍 전부 가능). 어떤 peer도 실행용으로는
  부족하다 — restore 전용으로만 쓴다.
- **`cuMemUnmap`은 커널이 그 범위를 건드릴 수 있는 동안 안전하지 않다.** scheduler step 경계에서만
  residency를 바꾼다(현재 코드가 turn 종료 시 파킹하는 지점과 같다). forward pass 안에서는 절대 금지.
- **PyTorch caching allocator에 VMM 메모리를 넘기지 말 것.** 직접 소유한다.
- **multi-process handle**: 현재 코드의 `cudaIpcMemHandle`은 VMM 할당에 **통하지 않는다**(다른 handle
  type). POSIX fd + unix socket `SCM_RIGHTS`, 또는 기존 `/dev/shm` rendezvous로 fd를 전달해야 한다.
- **host 페이지는 fd export가 안 된다** (§9 측정: exportable을 요구하면 `INVALID_VALUE`). 따라서
  2P2D에서 **host overflow 슬랩은 프로세스 간 공유 불가** — 각 노드가 자기 host overflow만 갖고,
  cross-node로 공유하는 것은 peer HBM 슬랩으로 한정한다. 설계상 문제는 아니다(host는 애초에 마지막
  수단이고 §3.1b 기준 중요한 tier가 아님)지만, shared index가 location별로 공유 가능/불가를 구분해야
  한다.
- **VRAM 회계**: peer HBM으로 자라는 park 공간은 "GPU의 KV pool은 자기 메모리에 bounded"라는 가정을
  깬다. per-GPU 상한과 §5의 `reclaim()`이 없으면 서빙 중인 peer를 OOM시킨다.
