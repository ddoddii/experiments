# CAL Outline — KV Victim Cache: Reclaiming Transiently-Idle GPU HBM for Agentic LLM Serving

**Target**: IEEE Computer Architecture Letters (CAL). Short letter (~4 pp double-column).
**Status**: outline draft (2026-07). 숫자는 2P2D·BFCL·pool 60k 기준; park 동률은 REPS=3 재확인 대기(★).

---

## 0. One-line thesis

Agent/RAG 서버에서 host CPU·DRAM은 희소한데 hicache류는 KV를 host로 민다. PD disaggregation은
GPU HBM을 자주 놀린다. 우리는 **축출 위험 KV를 일시적 유휴 GPU HBM에 담는 "KV Victim Cache"**를
제안 — **동일 latency에서 host RAM을 대폭 절약**(122 GB → 51 GB)한다.

## 1. Introduction (motivation + contributions)

- **문제**: multi-turn agent 서빙에서 tool-call 유휴 구간에 세션 prefix가 GPU radix에서 축출 →
  다음 turn 전체 재-prefill → TTFT 폭증. 완화책 hicache는 KV를 **host DRAM/SSD로 offload**.
- **간과된 비용**: agent/RAG 개인화 서버에서 host 자원(CPU, DRAM)은 orchestration·검색(vector
  DB/rerank)·tool 실행·프로필이 이미 경쟁 중. **KV를 host로 미는 것은 agent 스택 자원을 잠식**한다.
- **간과된 기회**: PD disaggregation은 비대칭이라 **decode 포화(91–94%) 시에도 prefill HBM은
  83–90% idle** (§우리 측정). P↔D/peer-GPU 링크는 빠름.
- **아이디어**: 축출 KV를 **일시적 유휴 GPU HBM에 담는 victim cache**. GPU가 놀면 GPU에 두고,
  host는 진짜 포화일 때만 쓰는 cold overflow로 격하.

### 1.1 핵심 아이디어 (Intro 본문용 불렛)

- **관찰 1 — 대역폭 계층이 존재하지 않는다 (핵심 측정).** 하나의 CUDA VMM 주소 공간에서 동일한
  접근 경로로 측정하면: **local HBM 331 GB/s, peer GPU HBM 27.2 GB/s, host DRAM 26.1 GB/s.**
  peer와 host는 둘 다 PCIe-bound라 **4% 안쪽에서 동일**하다. 여기에 layer-wise transfer·prefill
  overlap·tool-call window prefetch가 더해져 end-to-end TTFT 차이는 사라진다.
  → **"local HBM(빠름) vs 그 밖의 전부(PCIe, 동일)" 2층뿐이다.** 기존 시스템이 host DRAM을 GPU HBM
  아래의 *별도 tier*로 두는 것은 **대역폭 근거 없이 자원만 예약하는 구조**이며, KV를 CPU DRAM에
  두느냐 유휴 GPU HBM에 두느냐는 성능 문제가 아니라 **자원 예약 문제**다. (Fig. X로 이 3-행 표를 제시)
- **관찰 2 — 기존 시스템의 비용은 "정적 예약"에 있다.** SGLang HiCache는 `--hicache-ratio`로 host
  KV pool을 **미리 확보**하며(측정 61 GB, device memory보다 크도록 강제), 캐시가 5% 찼든 95% 찼든
  같은 양을 점유한다. write policy(write_through/selective/**write_back**)나 storage backend를
  끄는 것으로도 줄지 않는다. 이 DRAM은 agent orchestration·scheduling·retrieval·tool execution이
  써야 할 자원이다(측정: MemAvailable 105 GB → 44 GB).
- **관찰 3 — 유휴 용량은 이미 값을 치른 자원이다.** 2P2D에서 decode가 평균 75% 점유일 때 prefill은
  45%로, KV pool의 **55%가 상시 미사용**이다(Fig. 1). 이 공간은 추가 비용이 0인데 버려진다.
- **제안 — Unified KV Memory Management.** CPU DRAM과 분산 GPU HBM을 고정 tier가 아니라 **하나의
  논리 KV 공간**으로 관리한다. 핵심은 **가상 주소(VA)와 물리 페이지(PA)의 분리**(CUDA VMM):
  - **VA = 광고하는 논리 용량** — local HBM + 모든 peer + host를 합친 크기로 예약하며 물리 비용 0.
  - **PA = 실제 커밋** — 실제로 캐시된 KV만큼만, **그 순간 자리가 있는 location**에서 커밋한다
    (유휴 peer GPU HBM 우선 → 자리 없으면 host DRAM overflow → 그것도 없으면 drop/recompute).
  - **residency가 각 페이지의 속성**이 되어, "어느 고정 tier에 저장할 것인가"가
    **"시스템 전체 가용 메모리에서 이 KV의 physical residency를 어디에 둘 것인가"**로 재정의된다.
  - `CU_MEM_LOCATION_TYPE_HOST`가 **동작함을 확인**(26.1 GB/s) — **CPU가 별도 tier가 아니라 같은
    주소 공간의 한 location**이 된다. "CPU와 GPU를 같은 층위로 관리"가 비유가 아니라 구현이다.
- **왜 CUDA Unified Memory(UVM)가 아닌가.** UVM은 oversubscription 시 **host로만** evict하며,
  "지금 유휴한 peer GPU로 spill"을 표현하는 인터페이스가 없다. migration 시점도 access-counter
  heuristic이 소유하고, attention 커널 내 page fault는 수용 불가다. 우리는 unified space라는
  **추상은 제공하되 residency·migration을 명시적으로 제어**한다.
- **왜 정적 예약이 아니라 VA/PA 분리여야 하는가 (핵심 논거).** 유휴 GPU를 위한 park pool을 정적
  텐서로 잡으면 예약 문제를 host DRAM에서 VRAM으로 **옮기는 것에 그친다**(현 구현: 26 GB VRAM 상시
  점유). 더 결정적으로, 정적 텐서는 논리적 무효화만 가능해 **빌려준 GPU가 메모리를 실제로 돌려받지
  못한다.** `cuMemUnmap`/`cuMemRelease`만이 물리 페이지를 그 GPU의 free pool로 반환한다 —
  **돌려줄 수 있어야 빌리는 것이다(lending is only real if you can physically give it back).**
- **결과로 주장하는 것.**
  1. **동일 TTFT** — peer restore는 recompute를 30–43× 이기고(1k–32k 측정), host caching과 동률.
  2. **총 예약 메모리 감소** — 동일 hit rate에서 host 61 GB → 0, 추가 VRAM 예약 0(사용량만 커밋).
  3. **커밋 효율(commitment vs cached)** 이라는 새 지표 — HiCache는 캐시가 20% 찼을 때도 61 GB를
     점유(효율 20%)하지만, 제안 방식은 캐시된 만큼만 커밋한다(≈100%).
  4. **agent stack 용량 회복** — 같은 노드에서 동시 tool execution 수가 회복됨.
- **vAttention(ASPLOS'25)과의 차별.** vAttention은 **한 GPU 안에서** paged-attention 커널을 피하고
  요청 내부 파편화를 없애기 위해 VMM을 쓴다. 우리는 **여러 GPU와 host를 걸쳐** 탄력적 차용 용량
  풀을 만들기 위해 쓴다: location 선택이 정책이 되고, 회수 트리거가 *다른* GPU의 serving 압박이며,
  대상이 긴 대화 prefix라 2 MB granularity로 충분해 vAttention이 필요했던 64 KB CUDA 확장이 불필요하다.
  (설계 상세: `paper/design_unified_kv_vmm.md`)
- **Contributions**:
  1. **Characterization** — KV-tier-movement의 3-벽(recompute:transfer 20–40:1 / decode-bound
     saturation / host DRAM ≫ GPU HBM 용량): 왜 "latency로 host caching을 못 이기고 capacity가 knob"인지 정량화.
  2. **KV Victim Cache** — 유휴 GPU HBM을 축출-KV 보조 캐시로 재정식화(Jouppi victim cache).
     opportunistic(전용 아님) · pressure-aware(그 순간 노는 GPU) · cross-node(shared index) ·
     session-keyed slab(단편화 없는 재사용).
  3. **Evaluation** — 동일 성능 @ **host RAM −71 GB**; naive coexistence가 dominated임을 보이는
     negative result가 host-free replacement 설계를 정당화; agent 시나리오(BFCL) 실측.

## 2. Background & Motivation

- **2.1 PD disaggregation & prefix caching**: prefill/decode 분리, radix/hicache 계층.
- **2.2 The idle-GPU opportunity**: PD 비대칭 측정 — decode 포화 순간 prefill HBM idle. **Fig 0**
  `results/kv_ts/fig_imbalance`: 포화 GPU와 여유 GPU가 **49.6%의 시간 동안 공존**(2P2D, BFCL).
- **2.3 CPU/host scarcity in agent serving** (동기 핵심): agent/RAG 부하가 host를 점유.
  hicache의 **파일 티어가 page cache를 94 GB까지** 채움(§우리 측정) → host 압박.
- **Fig 1 (motivation)**: `fig_mem` — hicache host RAM 55→122 GB 우상향 vs park 51 GB 평평.

## 3. Characterization — The Three Walls (왜 latency로는 못 이기는가)

- **Wall 1 — recompute : transfer ≈ 20–40:1**: prefill 재계산(~0.1–0.23 ms/tok) ≫ KV 로드
  (128 KB/tok ÷ 대역폭 ≈ 0.005 ms/tok). prefix 길이 무관. → 움직여 봐야 재계산 대비 싸지만
  host caching도 같은 이득을 봄. movement가 caching을 latency로 이기지 못하는 이유.
- **Wall 2 — decode-bound saturation**: throughput ÷ TPOT ≈ 동시 decode 수 → decode가 병목,
  prefill 재사용이 throughput 상한을 못 올림.
- **Wall 3 — host DRAM ≫ GPU HBM 용량**: victim 풀은 GPU HBM이라 host tier보다 작다 →
  **capacity가 knob**(hot working set이 풀에 들어가면 이김, 아니면 진다).
- **결론**: victim cache는 latency로 hicache를 "이기는" 게 아니라 **동률**이며, 차별점은
  **KV를 어디에 두느냐(host vs 유휴 GPU)**다. → §4·§5로 연결.

## 4. KV Victim Cache — Design

- **4.1 Victim-cache 정식화** (Jouppi ISCA'90 매핑): L1=GPU 서빙 풀, victim=축출 세션 KV,
  victim cache=유휴 GPU HBM 풀, 재참조 시 GPU→GPU swap-back.
- **4.2 Idle detection (pressure-aware)**: 각 노드가 live KV 점유율을 0.5 s마다 /dev/shm에 발행;
  decode가 가장 한가한 prefill을 선택(§`_select_target_prefill`), prefill이 헤드룸 최대 GPU 선택.
- **4.3 Park (write)**: request 완료 직전, D가 완성 KV(프롬프트+생성)를 유휴 P로 CUDA IPC GPU→GPU
  복사; 두 경계(프롬프트/전체) 해시 등록 + shared index 미러.
- **4.4 Fetch (read)**: 다음 turn prefill 전, 로컬 → shared index로 peer 풀까지 롤링 해시 매칭 →
  IPC 복사 → radix에 삽입(일반 prefix-hit으로 위장). 이미 있으면 조기 반환(중복 회피).
- **4.5 Session-keyed slab**: prefix-supersession으로 같은 대화 = 같은 고정 slab 재사용(제자리 성장),
  full slab만 LRU evict. 단편화 없음, survival ∝ slab 수. (free-list는 단편화로 실패 → slab이 해결.)
- **Fig 2 (design)**: 3-스텝 파이프라인(detect → park → fetch) + victim-cache 계층 다이어그램.

## 5. Evaluation

- **Setup**: server17, A6000×4, Llama-3.1-8B, SGLang, 2P2D, BFCL v3 multi-turn(agent workload),
  C=16, tool-delay 3 s, pool 60k(=10 slabs, host-free).
- **5.1 동일 성능 @ host-RAM-free** (핵심 결과):

  | arm | GPU HBM | host RAM(used+cache) | median TTFT | tput |
  |---|---|---|---|---|
  | hicache | 81 GB | **122 GB** | 3.51 s | ~86 tok/s |
  | KV victim (host-free) | 98 GB | **49 GB (−73)** | 3.52 s | ~85 tok/s |

  → 유휴 GPU HBM +17 GB로 host RAM −73 GB. **TTFT 동률 REPS=3로 확정** (per-rep hicache
  [3.61,3.77,3.88] vs park [3.67,3.90,3.70], pooled median 3.51 vs 3.52 s — 0.01 s 차).
  - **Fig 3**: `fig_perturn` — TTFT / effective throughput vs 증가하는 context length.
  - **Fig 1 재사용**: 메모리 타임라인(host 절약).
- **5.2 Negative result — naive coexistence는 dominated**: hicache(write_through_selective) 위에
  victim을 얹으면 GPU 98 GB + host 123 GB(둘 다 비쌈, host 안 줄어듦). eager offload가 원인.
  → **victim은 host 티어를 대체해야 한다**(공존 아님). 이 negative가 host-free 설계를 정당화.
- **5.3 Capacity is the knob**: survival ∝ pool; session-keyed slab이 같은 survival을 더 작은
  GPU로(120k→96k, ~20%↓). (Wall 3의 실증.)

## 6. Related Work (delta)

| 시스템 | KV tier | idle-GPU 재활용 | pressure-aware | cross-node | 동기 |
|---|---|---|---|---|---|
| Mooncake | DRAM/SSD 전용 공유풀 | ✗ | ✗ | ✓ | throughput |
| LMCache | GPU-local/CPU/disk | ✗ | ✗ | △ | reuse |
| DistServe/DOPD | 없음(인스턴스 분배) | ✗ | ✗ | ✓ | SLO |
| hicache(SGLang) | host DRAM/SSD | ✗ | ✗ | ✗ | prefix reuse |
| **본 연구** | **일시적 유휴 peer-GPU-HBM** | **✓** | **✓** | **✓** | **host/CPU 절약** |

Delta: 전용 공유풀이 아니라 **transiently-idle peer-GPU-HBM을 victim으로 기회주의 재활용**,
동기가 **agent 서버의 host/CPU 희소성**.

## 7. Conclusion

KV Victim Cache는 PD disaggregation의 유휴 GPU HBM을 축출-KV 보조 캐시로 재활용해, **성능 손해
없이 host RAM(특히 hicache의 94 GB 파일 캐시)을 대폭 절약**한다. agent/RAG 서버에서 host 자원이
희소해지는 추세에 부합하는 KV 배치 방향을 제시한다.

## 8. 남은 실험 체크리스트 (제출 전)

- [x] ★ park vs hicache **REPS=3** — TTFT 동률 확정 (pooled median 3.51 vs 3.52 s).
- [ ] (선택) write-back hicache + victim — smart coexistence로 host offload 감소 검증.
- [ ] (선택) CPU-contention arm — host 배경부하 하에서 hicache 저하 / victim 무영향 시연(§2.3 실증).
- [x] Fig 정리: Fig0 fig_imbalance, Fig1 fig_mem, Fig2 fig_design, Fig3 fig_perturn (Times New Roman, PDF).

## Key figures (현재 산출물)
- `results/fig_combined.{pdf,png}` — **cross-workload 종합** (BFCL+ShareGPT × TTFT/throughput/host RAM;
  범례 SGLang vs KV Victim Cache) — 성능 동률 + host RAM win이 두 워크로드에서 성립
- `results/kv_ts/fig_imbalance.{pdf,png}` — Fig 0 (intro: idle-GPU opportunity 49.6%)
- `results/perturn/fig_mem.{pdf,png}` — Fig 1 (host RAM 절약; 동기)
- `results/perturn/fig_perturn.{pdf,png}` — Fig 3 (per-turn TTFT/throughput vs context)
- `results/perturn/fig_design.{pdf,png}` — Fig 2 (victim hierarchy + detect/park/fetch)

## References (핵심)
- N. P. Jouppi, "Improving direct-mapped cache performance …," ISCA 1990 (victim cache 앵커)
- Mooncake (FAST'25), LMCache, DistServe (OSDI'24), DOPD, SGLang HiCache
- InferCept (ICML'24), AttentionStore/CachedAttention (ATC'24) — tool/think-time KV
