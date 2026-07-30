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

**목표 문장**: 전체 KV 데이터 크기는 유지하면서, **유휴 GPU HBM을 우선 활용하여 CPU DRAM
commitment를 줄인다.** (총 메모리 사용량 감소가 아니다.)

**기여 문장**: *We propose a KV-aware unified-memory interface that exposes serving semantics
unavailable to the existing CUDA Unified Memory API.* — CUDA Unified Memory를 구현하거나 개선한
것이 아니라, **KV-cache semantics를 반영한 unified-memory placement policy와 runtime interface**를
제안한다.

- **관찰 1 — HBM 아래에 대역폭 계층이 존재하지 않는다.** 하나의 주소 공간에서 동일한 접근 경로로
  측정하면 (4× A6000, NV4 pair): **local HBM 331 GB/s, NVLink peer HBM 27–53 GB/s,
  CPU DRAM 24–26 GB/s.** NVLink peer와 CPU DRAM은 같은 구간이다(27.2 vs 26.1 = **4% 차**). 여기에
  layer-wise transfer·prefill overlap·tool-call window prefetch가 더해져 end-to-end TTFT 차이는
  사라진다. → **"local HBM(빠름) vs 그 밖의 전부(≈PCIe)" 2층뿐이므로**, peer GPU HBM과 CPU DRAM을
  분리된 고정 tier로 볼 근거가 없다. 하나의 placement 대상 집합(= unified memory)으로 보는 것이 맞다.
  (Fig. X로 이 표 제시)
- **관찰 1b — 그런데 이 집합은 메모리 계층 직관대로 정렬되지 않는다.** 같은 노드에서 **non-NVLink
  peer는 3.3 GB/s로 CPU DRAM보다 7–8× 느리다.** 올바른 순위는
  *NVLink peer(27–53) > CPU DRAM(24–26) ≫ non-NVLink peer(3.3) > recompute*이며, **"GPU가 항상 CPU보다
  빠르다"는 고정 tier 가정은 틀렸다.** → placement는 고정 계층이 아니라 **런타임이 측정한 대역폭에
  따라 결정**되어야 한다.
- **관찰 2 — 기존 시스템은 CPU DRAM을 정적으로 예약한다.** SGLang HiCache는 `--hicache-ratio`로 host
  KV pool을 **미리 확보**하며(측정 **61 GB**, device memory보다 크도록 강제), 캐시가 5% 찼든 95% 찼든
  같은 양을 점유한다. write policy(write_through/selective/**write_back**)나 storage backend를 끄는
  것으로도 줄지 않는다. 이 DRAM은 agent orchestration·scheduling·retrieval·tool execution이 써야 할
  자원이다 (측정: MemAvailable **105 GB → 44 GB**).
- **관찰 3 — 유휴 GPU HBM은 이미 값을 치른 자원이다.** 2P2D에서 decode가 평균 75% 점유일 때 prefill은
  45%로, KV pool의 **55%가 상시 미사용**이다(Fig. 1). 추가 비용이 0인데 버려진다.
- **CUDA Unified Memory가 이 문제를 풀 수 없는 이유 (semantic gap).** `cudaMallocManaged()`는 공통
  주소 공간을 주지만 residency는 **page fault·eviction·prefetch·driver policy**로 결정되고,
  `cudaMemAdvise`/`cudaMemPrefetchAsync`는 **hint**일 뿐이다. UM은 다음을 **모른다**: 이 range가 어느
  세션의 KV인지 / 다음 turn에 재사용될지 / 지금 어느 GPU가 포화인지 / 어떤 peer가 순간적으로 유휴한지
  / host DRAM을 agent stack에 남겨야 하는지 / 버리면 얼마나 긴 re-prefill이 필요한지(8k에서 1 203 ms).
  메커니즘 차원에서도: ① oversubscription eviction이 **host로만** 가서 우리가 줄이려는 CPU DRAM
  commitment를 오히려 **늘린다**, ② fault 단위 migration은 "알려진 시점에 세션 KV 전체를 벌크 전송"과
  맞지 않는다, ③ **UM은 process 경계를 넘지 못하므로**, PD 배포의 별개 프로세스 4개에서 한 프로세스의
  유휴 GPU 메모리를 다른 프로세스의 KV에 쓰는 것을 **표현조차 못 한다**, ④ 토폴로지(관찰 1b)를
  placement에 반영하지 않는다, ⑤ managed 데이터를 **버릴 수 없어** 반드시 host에 보존한다.
  → UM은 **접근 패턴 중심**, 우리는 **KV reuse 가치와 serving pressure 중심**이다.
- **제안 — KV-aware unified-memory placement policy.** 각 reusable KV를 GPU/CPU에 따로 복제되는
  객체가 아니라 **하나의 논리적 KV object**로 관리한다 (identifier = prefix hash/session ID, size,
  current/preferred/fallback location, reuse value). **한 시점에 유효한 copy는 한 location에만** 둔다.
  축출 시 GPU별 pressure를 보고 다음 우선순위로 배치한다:
  1. **현재 여유가 가장 큰 peer GPU HBM** (링크 대역폭이 CPU DRAM보다 나은 GPU만)
  2. **다른 가용 GPU HBM**
  3. **CPU DRAM overflow**
  4. **eviction / recomputation**

  파킹된 GPU가 serving으로 차오르면 reuse value가 낮은 KV부터 재평가해 *다른 여유 GPU로 이동* /
  *CPU DRAM으로 demote* / *폐기*한다 — **serving이 항상 이기고**, 정합성 fallback은 recompute다.
  다음 turn에 필요해지면 prefix hash로 residency를 찾아 **target GPU로 벌크 prefetch**하고, 이후는
  기존 radix-cache hit과 동일하게 쓴다.
- **UM API와의 차이 (표로 제시).** 배치가 driver **hint**가 아니라 **계약**(실패 시 다음 순위) /
  이동 트리거가 fault가 아니라 **serving pressure + reuse value** / eviction 목적지가 host 고정이
  아니라 **유휴 peer GPU → CPU → drop** / 전송 단위가 page가 아니라 **세션 KV object 전체** /
  **프로세스 경계를 넘고** / **데이터를 버릴 수 있다**(victim cache) / **토폴로지를 인지한다**.
  마지막 두 개가 핵심이다 — UM은 버릴 수 없어 host에 보존해야 하지만, **우리는 버릴 수 있으므로
  host를 마지막 수단으로 미룰 수 있다.**
- **결과로 주장하는 것.**
  1. **동일 TTFT** — 어느 location에서 가져와도 recompute 대비 **30–43×**(1k–32k 측정), host
     caching과 동률(park 0.605 s vs hicache 0.614 s).
  2. **CPU DRAM commitment 감소** — 동일 hit rate·동일 KV 총량에서 host 예약이 61 GB → 대폭 감소.
  3. **placement가 실제로 GPU-first로 동작** — location별 파킹 바이트 비율 시계열로 직접 제시.
  4. **agent stack 용량 회복** — 같은 노드의 동시 tool execution 수(K_max)와 tool p99.
- **정직하게 함께 보고할 것.** park pool은 현재 정적 VRAM 예약이므로 **host와 device commitment를
  둘 다 보고**한다. 두 바이트는 같지 않다 — CPU DRAM은 agent stack과 **경쟁하는** 자원이고(105→44 GB),
  GPU HBM은 **55%가 상시 미사용**인 어차피 버려지는 용량이다. 이 비대칭이 논거다. 또한 CPU DRAM을
  0으로 만드는 것이 아니라(순위 3이 설계에 포함) **줄이는** 것이며, 모든 GPU가 동시 포화인 구간에서는
  HiCache와 같아진다는 점도 평가에 포함한다.
  (설계 상세: `paper/design_kv_aware_unified_memory.md`)

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
