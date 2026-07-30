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

| 항목 | 측정값 | 해석 |
|---|---|---|
| allocation granularity | **2.00 MiB** (min = recommended) | KV slab은 2 MiB의 배수 |
| P2P 도달성 | **12개 순서쌍 전부 1** | 모든 GPU가 서로 매핑 가능 |
| local HBM read | **333.2 GB/s** | 기준선 |
| **peer HBM read (0←1)** | **27.2 GB/s = local의 8%** | **PCIe 속도** (NVLink 브리지면 ~50 GB/s) |
| host location VMM 할당 | `CUDA_ERROR_INVALID_VALUE` | 재시도 매트릭스 필요(§9) |
| 같은 VA에서 residency 변경 (32 MiB) | **p50 87.4 µs** | 배치하면 크게 낮아짐(§6) |
| remap된 VA로 write/read | **OK** | 메커니즘 동작 확인 |

### 3.1 이 측정이 확정한 두 가지

**(a) peer-resident KV로 attention을 직접 돌리면 안 된다.** Llama-3.1-8B은 토큰당
32층 × 8 kv-head × 128 dim × K/V × 2 B = **128 KiB**. 8k context = 0.98 GiB이고 decode step마다
전부 읽는다 → **local 3.1 ms vs peer 38.6 ms.** peer는 **restore 소스**이고 실행 시 residency가
아니다. (정책에서 "대역폭 충분하면 제자리 읽기" 분기는 삭제)

**(b) 그런데 peer restore는 recompute를 30–43× 이긴다.** 측정 27.2 GB/s × 측정 recompute
(`results/intro/fig_ttft_ctx_sweep.json`):

| ctx | KV 크기 | local restore | **peer restore** | recompute (측정) | **speedup** |
|---|---|---|---|---|---|
| 1 000 | 0.12 GiB | 0.4 ms | **4.8 ms** | 161 ms | **33×** |
| 4 000 | 0.49 GiB | 1.6 ms | **19.3 ms** | 592 ms | **31×** |
| 8 000 | 0.98 GiB | 3.1 ms | **38.6 ms** | 1 203 ms | **31×** |
| 16 000 | 1.95 GiB | 6.3 ms | **77.1 ms** | 2 706 ms | **35×** |
| 32 000 | 3.91 GiB | 12.6 ms | **154.2 ms** | 6 681 ms | **43×** |

→ **peer GPU HBM은 host DRAM과 동급 전송 대역폭(둘 다 PCIe-bound)이므로 recompute 회피 능력도
동급이고, host DRAM은 0을 쓴다.** 기여를 "HiCache보다 빠르다"(§7-3에서 이미 못 이긴다고 나온 축)가
아니라 **"동일 이득, 다른 자원"**으로 정의한다.

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
    loc   = choose_location()          # argmax headroom: local -> peers -> host
    pages = [cuMemCreate(2MiB, prop(loc)) for _ in range(n_pages)]
    for p in pages: cuMemMap(park_va + off, 2MiB, 0, p, 0)   # 2 us each
    for r in layer_ranges: cuMemSetAccess(r, descs)          # range 단위로 배치 (§6)
    p2p_copy_in(...)                   # 기존 IPC 경로 재사용
    shared_index.put(hash, (gpu, off, n))

# 다음 turn: 항상 target GPU로 가져온다 (제자리 attention 금지, §3.1a)
def fetch(session):  copy_to_local(...)   # tool-call window 중 prefetch

# 압박 시 회수: 물리 페이지를 실제로 반환
def reclaim(gpu, need_bytes):
    for slab in lru_parked(gpu):
        cuMemUnmap(range); cuMemRelease(pages)   # -> 그 GPU free pool로 복귀
        shared_index.invalidate(slab)
        if freed >= need_bytes: break
```

**배치 정책** `choose_location()`:
1. 여유 있는 **peer GPU** (headroom 최대, NVLink 브리지 우선 — §9로 확인)
2. 어느 GPU에도 자리 없으면 **host** (overflow, 기본값 아님)
3. 그것도 없으면 **drop** (victim cache 의미론: recompute가 정합성 fallback)

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

## 6. remap 비용 — 반드시 배치해야 한다

vAttention Table 3 (2 MB 기준): `cuMemMap` **2 µs**, `cuMemSetAccess` **38 µs**,
`cuMemUnmap` **34 µs**. 핵심은 **비싼 `cuMemSetAccess`가 페이지당이 아니라 range당**이라는 것.

6000-token 세션 slab (= 750 MiB = 2 MiB 페이지 384장, 레이어×K/V 64개 range):

| 방식 | 계산 | 비용 |
|---|---|---|
| 페이지마다 map+setAccess | 384 × 74 µs | **28 ms** ✗ |
| **384개 map 후 range별 setAccess 1회** | 384×2 µs + 64×38 µs | **3.2 ms** ✓ |

같은 slab의 전송이 750 MiB / 27 GB/s = **28 ms**이므로, 배치하면 제어 오버헤드가 **11%**로
떨어진다. (probe의 87 µs/32 MiB는 배치하지 않은 단일 slab 값이다.)

**2 MB granularity는 우리에게 문제가 아니다.** vAttention이 64 KB 페이지용 CUDA 확장까지 만든 이유는
*짧은 요청*의 내부 파편화였다. 우리는 수천 토큰짜리 대화 prefix를 통째로 파킹하므로 2 MB(= 레이어당
1024 토큰)면 충분하다. **vAttention의 최대 구현 부담을 우리는 지지 않는다.**

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

## 9. C0에서 확정해야 할 것 (10분)

```bash
./scripts/stop.sh
python benchmark/vmm_probe.py --topo --pairs --all-pairs-bw \
    --remap-slabs 2 8 32 128 --slab-mb 32
```

1. **host location이 되는가?** 첫 실행은 `INVALID_VALUE`였다. 가장 유력한 원인은 probe가 host
   location에도 **POSIX fd exportable handle을 요청**한 것이고, 두 번째 가능성은 driver < CUDA 12.2.
   probe가 이제 HOST / HOST_NUMA × {exportable, 아님} 조합을 순차 시도하고 어느 게 통하는지 보고한다.
   - 전부 실패하면: overflow tier는 `cuMemHostAlloc` + 별도 host pool(=HiCache 방식)로 두고,
     unified space는 **local+peer HBM만** 덮는다고 논문에 명시한다. ④는 약해지지만 결과는 안 바뀐다
     (§3.1b가 보여주듯 중요한 tier는 peer HBM이고 host는 overflow일 뿐).
2. **NVLink 브리지가 실제로 있는 쌍이 있는가?** 27.2 GB/s는 PCIe다. `--all-pairs-bw`에서 어떤 쌍이
   튀어나오면 `choose_location()`이 topology-aware여야 하고, restore가 다시 ~2× 빨라진다.
3. **remap latency vs slab 크기** — §6의 배치 계산을 실측으로 뒷받침.

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
- **VRAM 회계**: peer HBM으로 자라는 park 공간은 "GPU의 KV pool은 자기 메모리에 bounded"라는 가정을
  깬다. per-GPU 상한과 §5의 `reclaim()`이 없으면 서빙 중인 peer를 OOM시킨다.
