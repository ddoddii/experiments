# Research: Tool-call-aware Multi-tier KV Cache Placement for Agentic PD Disaggregation

> **연구 주제**: P-D disaggregation 환경에서 multi-turn agent의 tool call duration을 예측해
> KV cache를 **4-tier storage (D-HBM / P-HBM / CPU DRAM / SSD)** 중 어디에 둘지 결정하고,
> tool call 종료 전에 prefetch해 복구 latency를 숨기는 **placement policy** 설계.

---

## 1. Problem Statement

PD disaggregation에서 prefill 노드가 생성한 KV cache는 decode 노드로 전송된 이후에도 **multi-turn 대화가 지속되는 동안 어딘가에 보관**되어야 한다. 현재 시스템들은 다음 두 극단 사이에 위치한다:

| 전략 | 예시 | 문제점 |
|------|------|--------|
| **항상 GPU에 유지** (LRU) | vllm-ppd | Tool call 중에도 GPU KV memory 점유 → 다른 요청의 처리 용량 감소 |
| **즉시 SSD에 기록** (write_through) | SGLang hicache | 고동시성에서 SSD I/O 경합 → cache가 오히려 latency 증가 |

**핵심 관찰**: agentic 워크로드에서 tool call duration은 수ms~수십초로 **워크로드에 따라 수백 배 차이**가 난다. 이 정보를 활용하면:

- **짧은 tool call** (수백ms): 클라이언트가 금방 돌아옴 → KV를 빠른 tier(GPU)에 유지가 합리적
- **긴 tool call** (수십초): 그 시간 동안 GPU 점유는 낭비 → 저렴한 tier로 이동하거나 evict 후 re-prefill

### 1.1 Tier Hierarchy (제안)

```
Tier 1: D 노드 HBM     — decode가 일어나는 곳. 가장 비싼 자원 (active batch와 경쟁)
Tier 2: P 노드 HBM     — 다음 turn의 (append-)prefill이 일어나는 곳.
                          여기 두면 resume 시 prefix 전송 비용 0. 단 prefill 용량과 경쟁
Tier 3: CPU DRAM       — PCIe ~20 GB/s. 용량 큼 (125 GB)
Tier 4: SSD            — NVMe ~3-7 GB/s. 사실상 무한 용량
EVICT : 폐기 후 re-prefill — 복구 비용 = cold TTFT (context 길이에 비례)
```

Policy: tool call duration `d`를 예측 → `d`가 짧으면 상위 tier 유지, 길수록 하위 tier로 내림
→ tool 종료 예상 시점 `T−δ`에 **prefetch**를 시작해 복구 latency를 `d` 안에 숨김.

---

## 2. Background & Motivation (기존 실험 결과)

### 2.1 KV 재사용의 TTFT 이득 — placement의 가치

BFCL v3 multi-turn 벤치마크(200 items, avg 3.7 turns/item)에서 측정한 per-turn TTFT:

| Config | Turn 0 (cold) | Turn 1 (cache hit) | 이득 |
|--------|-------------|-------------------|------|
| vLLM TP=4 C=1 | 0.145 s | 0.078 s | **−46%** |
| vllm-ppd C=1 | 0.967 s | 0.321 s | **−67%** |
| vllm-ppd C=8 | 1.712 s | 0.669 s | **−61%** |
| SGLang 2P2D C=8 | 1.366 s | 0.802 s | **−41%** |

모든 시스템에서 turn 1부터 KV cache hit로 TTFT 40–67% 감소. **KV retention은 first-order performance lever임**.  
→ "어디에 두느냐"는 이 이득을 얼마나 보존하느냐를 결정한다.

### 2.2 고동시성에서 cache 이득 소멸 — 원인 미확인 이상 현상

SGLang hicache는 `write_through` 정책으로 모든 KV를 즉시 `/tmp/hicache/` (SSD)에 기록한다. C=12 실험에서 이상 현상 관측:

```
SGLang 2P2D C=12, 첫 번째 성공 아이템:
  turn 0 TTFT:  1.413 s  (cold, 정상)
  turn 1 TTFT:  7.652 s  ← cache hit임에도 turn 0보다 5.4× 느림
```

**⚠️ 이 관측의 한계**: 67개 성공 아이템 중 첫 번째 아이템의 단일 데이터 포인트. 현재 가설로는 다음 세 가지가 동등하게 가능하다:

| 가설 | 설명 |
|------|------|
| SSD I/O 경합 | 12개 요청이 동시에 `/tmp/hicache/` read/write → 디스크 대기 | 
| Mooncake KV 전송 실패 → re-prefill | turn1의 KV transfer 실패로 full re-prefill fallback (C=12 에러율 66.5%) |
| Queue 대기 | P/D 노드 포화로 turn1 요청이 큐에서 장시간 대기 |

어느 가설이 맞는지 확인하려면 해당 시점의 `iostat`, Mooncake 에러 로그, SGLang hicache 내부 read latency 측정이 필요하다 (→ §5.D 참조).

**관찰 자체의 함의**: turn1이 turn0보다 느린 상황은, cache hit 여부와 무관하게 **고동시성에서 hicache의 이득이 사라질 수 있음**을 시사한다. 원인이 SSD I/O이든 KV transfer 실패이든, 이 패턴이 재현된다면 placement policy의 motivation이 된다.

### 2.3 Tool call 중 GPU KV는 idle — resource waste 정량화

vllm-ppd C=8 실험에서 측정한 D 노드 KV cache 점유율:

| Node | Max KV | Mean KV |
|------|--------|---------|
| D1 (GPU2) | **17.2%** | 5.9% |
| D2 (GPU3) | **15.5%** | 5.4% |

평균 3.7 turns/item, 각 turn 사이에 tool call이 실행된다. 이 시간 동안 KV는 D 노드 GPU에 남아 있으나 **실제 decode 연산은 진행되지 않는다** — 순수 idle 점유.  
Tool call duration이 길수록 이 낭비는 선형으로 증가한다.  
→ 측정은 아직 "전체 평균"이며 idle 구간만 분리한 측정이 필요하다 (§5.C 참조).

**⚠️ Motivation 관점의 약점**: max 17%는 memory pressure가 없다는 뜻이기도 하다. 이 상태에선
Tier 1 유지가 항상 최적이므로, 평가는 반드시 KV pool이 포화되는 영역(동시성 ↑ 또는
`--mem-fraction` 축소로 인위적 pressure)에서 수행해야 한다 (§3.4 리스크 참조).

### 2.4 Buffer 용량이 동시성 상한을 결정 — KV in-flight 병목

P 노드 KV send buffer (1/2/4 GB) × concurrency (8/12/16) 9-cell sweep 결과:

| buf \ C | C=8 성공률 | C=12 성공률 | C=16 성공률 |
|---------|-----------|------------|------------|
| 1 GB | 50.5% | 47.5% | 47.5% |
| 2 GB | 58% | 51% | 51% |
| 4 GB | **76%** | 55% | 55% |

두 가지 관찰:
1. **버퍼 증가 → C=8 성공률 비례 향상**: P→D 전송 중 KV in-flight 용량이 병목
2. **C=12 = C=16**: buffer 확장만으로는 한계 — C≥12에서는 **요청별 KV 크기(context 길이)**가 지배적 병목으로 전환

→ Buffer 크기 조정은 blunt instrument. **요청별 KV 크기와 inter-turn gap을 함께 고려하는 placement policy**가 필요하다.

### 2.5 Re-prefill 비용은 turn 깊이에 비례 — context 성장 데이터

BFCL v3에서 관측한 context 성장 패턴 (vLLM TP=4 측정):

| Turn | 평균 context 길이 | 증가 |
|------|-----------------|------|
| 0 | ~1,553 chars | — |
| 1 | ~1,727 chars | +11% |
| 2 | ~1,885 chars | +9% |
| 3 | ~2,178 chars | +16% |
| 4 | ~2,369 chars | +9% |
| 5 | ~2,680 chars | +13% |

Turn이 깊어질수록 re-prefill 비용이 증가한다. **초반 turn의 KV eviction은 상대적으로 저렴하고, 후반 turn의 eviction은 비싸다.**  
→ Placement policy는 현재 turn depth를 feature로 사용할 수 있다.

### 2.6 Prefill vs Decode 비용 비대칭 — break-even 분석의 근거

| Framework | TPOT (decode) | Re-prefill TTFT (turn 0 cold) |
|-----------|--------------|------------------------------|
| vLLM TP=4 | 0.022 s/tok | 0.145 s |
| vllm-ppd | 0.065 s/tok | 0.967 s |
| SGLang 2P2D | 0.023 s/tok | 0.814 s |

Prefill은 context token을 병렬 처리하므로 context 2× = TTFT ≈ 1.5× (sub-linear).  
Decode는 autoregressive라 출력 길이에 선형.  
→ 짧은 context의 re-prefill은 상대적으로 저렴하다. **Placement policy의 eviction threshold는 context 길이와 re-prefill 비용 곡선에서 결정되어야 한다.**

---

## 3. Related Work & Novelty Positioning (2026-06 조사)

### 3.1 경쟁 연구 비교

"tool call 동안 KV를 치워둔다"는 코어 아이디어는 2024–2025년에 이미 선점되었다.
**살아남는 novelty는 P-D disaggregation 교차점**이다.

| 시스템 | Tool-call aware | Duration 예측 | Multi-tier | PD disagg | Prefetch |
|---|---|---|---|---|---|
| InferCept (ICML'24) | ✅ | △ (비용 추정) | GPU/CPU/discard | ❌ | ❌ |
| AttentionStore (ATC'24) | ❌ (사람 think time) | ❌ | HBM/DRAM/SSD | ❌ | ✅ (스케줄러 기반) |
| [Continuum/CacheTTL](https://arxiv.org/abs/2511.02230) (2025.11) | ✅ | ✅ (per-tool empirical CDF) | ❌ (GPU pin or evict) | ❌ | ❌ |
| [TokenCake](https://arxiv.org/abs/2510.18586) (2025.10) | ✅ | ✅ (실행시간 추정) | GPU/CPU 2-tier | ❌ | ✅ (predictive upload) |
| Mooncake | ❌ | ❌ | DRAM/SSD pool | ✅ | △ |
| [KVFlow](https://arxiv.org/pdf/2507.07400) (2025.7) | △ (workflow-aware) | △ (steps-to-execution) | GPU/CPU | ❌ | ✅ |
| [AMPD](https://arxiv.org/abs/2602.14516) (2026.2) | △ (multi-round) | ❌ | ❌ | ✅ | ❌ |
| [PPD "Not All Prefills Are Equal"](https://arxiv.org/abs/2603.13358) (2026.3) | ❌ | ❌ | ❌ | ✅ (D에서 append-prefill) | ❌ |
| **본 연구** | ✅ | ✅ | **4-tier (P-HBM 포함)** | ✅ | ✅ |

주요 경쟁자 요약:
- **Continuum/CacheTTL**: tool call duration을 per-tool empirical CDF로 예측, KV를 GPU에
  TTL과 함께 pin, TTL 만료 시 evict. SWE-Bench/BFCL/OpenHands + Llama-3.1 8B/70B 평가,
  JCT 8× 개선. **pin-or-evict 이진 선택, 단일 노드.** → duration 예측기의 feasibility는
  입증됨 (예측기는 차용하고, 기여는 placement에 집중할 것).
- **TokenCake**: function call 중 KV를 CPU로 offload + 실행시간 추정 기반 predictive
  upload(=prefetch). E2E latency −47%. **GPU/CPU 2-tier, PD 미고려.**
- **PPD (Not All Prefills Are Equal)**: multi-turn PD에서 turn 2+를 D 노드에서
  append-prefill하는 dynamic routing. Turn 2+ TTFT −68%. **tool-call/duration 비인지.**

### 3.2 PD 환경 고유 문제 — 본 연구의 delta

리뷰어의 예상 질문 *"Continuum + AttentionStore + Mooncake 조합 아닌가?"* 에 대한 방어는
PD에서만 생기는 다음 세 가지 고유 문제를 논문의 중심에 두는 것이다:

1. **KV가 물리적으로 분산되어 있다.** Turn N의 prompt KV는 P 노드가 생성, 생성 토큰의 KV는
   D 노드에만 존재. Turn N+1의 prefill은 P에서 일어나는데 P는 직전 turn 생성 토큰의 KV를
   가진 적이 없다 — 단일 노드 시스템에는 없는 placement 문제.
2. **P 노드 HBM이 tier로서 독특하다.** 다음 turn의 (append-)prefill이 P에서 일어나므로
   tool call 동안 KV를 P-HBM에 두면 resume 시 prefix 전송 비용이 0. 반면 P-HBM은 prefill
   처리 용량과 경쟁. PPD 논문은 정반대("KV가 있는 D에서 append-prefill")를 주장 —
   **"다음 turn의 KV가 P에 있어야 하나 D에 있어야 하나"는 tool duration과 부하의 함수**라는
   것이 본 연구의 고유 질문.
3. **Prefetch 목적지가 2개다.** 단일 노드에선 prefetch = CPU→GPU 복원이지만, PD에선
   "P로 가져와 append-prefill 후 D로 전송" vs "D로 직접 가져와 D에서 prefill" 두 경로가
   있고, Mooncake 전송 대역폭 경합(§2.4에서 관측한 병목)이 결정에 들어간다.

### 3.3 타당성 정량 검증 (수치)

Llama-3.1-8B (GQA, 32 layers × 8 KV heads × 128 head_dim, BF16) 기준 KV = **128 KB/token**
(Qwen3-14B: 40 layers → 160 KB/token). 8k tokens context ≈ 1 GB.

A6000 시스템(PCIe, NVLink 없음)에서 1 GB KV (8k tokens) 기준 추정:

| 동작 | 추정 비용 | 근거 |
|------|----------|------|
| Tier 2/3 이동 (P-HBM/DRAM, PCIe ~20 GB/s 실효) | ~50 ms | PCIe 4.0 x16 |
| Tier 4 복원 (NVMe ~3–5 GB/s) | ~200–300 ms | NVMe 실효 |
| Re-prefill (EVICT) | 0.8–1.4 s | §2.6 실측 cold TTFT |

→ **모든 tier가 re-prefill보다 싸므로 "치워두기"는 수치상 성립**하고, tier 간 격차
(50 ms vs 300 ms vs 1.4 s)가 충분히 커서 duration-aware 선택이 의미를 가진다.
정확한 값은 §5.A break-even 맵 실험으로 실측한다.

### 3.4 리스크 (솔직한 약점)

| 리스크 | 내용 | 대응 |
|--------|------|------|
| Memory pressure 부재 | §2.3에서 D 노드 KV max 17% — pressure 없으면 Tier 1 유지가 항상 최적 | 동시성 ↑ 또는 `--mem-fraction` 축소로 포화 영역에서 평가. "idle KV가 active 요청을 queueing시킴"을 직접 시연 |
| SSD tier 정당화 | RAM 125 GB 시스템에서 8B 모델 KV는 거의 전부 DRAM에 흡수 | Tier ablation (§5.I)으로 각 tier의 한계 기여 분리. SSD는 DRAM pressure 시나리오(동시 세션 수백 개 / DRAM 캐시 용량 제한)에서만 주장 |
| BFCL mock tool | tool이 즉시 반환 → synthetic delay 주입만으로는 약함 | 실제 trace 기반 분포(SWE-bench agent 로그, 실 API latency) 주입. Continuum과 동일 워크로드로 비교 가능성 확보 |
| Duration 예측 novelty 소진 | Continuum이 이미 empirical CDF 예측 입증 | 예측기는 차용, 기여는 multi-tier placement + PD 고유 문제로 |
| 단일 8B 모델 | 모델 스케일 axis 약함 (70B는 하드웨어상 불가) | long-context 변형(turn 누적 + 긴 system prompt)으로 KV pressure axis 보강 |
| 구현 공수 | D 노드 decode 중 KV를 P-HBM/DRAM/SSD로 내보내고 가져오는 경로가 SGLang에 없음 (HiCache는 P 노드 radix cache 계층화) | Oracle(§5.E)까지는 캐시 flush/유지 조작으로 시뮬레이션 → 가설 확정 후 시스템 구현 |

---

## 4. Research Design

### 4.1 연구 질문

> **RQ1**: Tool call duration에 따라 KV placement를 달리했을 때 TTFT와 GPU memory utilization이 동시에 개선되는가?

> **RQ2**: 어떤 features (tool type, turn depth, context length, concurrency)가 최적 placement를 결정하는 데 충분한가?

> **RQ3**: Oracle placement policy 대비 rule-based policy의 performance gap은 얼마인가?

> **RQ4** (PD 고유): 다음 turn의 prefix KV는 P 노드에 있어야 하는가 D 노드에 있어야 하는가?
> 그 답은 tool duration · context 길이 · 시스템 부하에 따라 어떻게 바뀌는가?

### 4.2 Placement Policy 설계 공간

```
입력 features:
  f1. predicted_tool_duration (per-tool empirical CDF — Continuum 방식 차용)
  f2. turn_depth (현재 conversation의 turn 번호)
  f3. context_length (현재 KV의 token 수)
  f4. current_gpu_pressure (P/D 노드 KV utilization)
  f5. transfer_congestion (Mooncake in-flight KV 양)

출력:
  placement ∈ {D_HBM, P_HBM, CPU_DRAM, SSD, EVICT}
  prefetch_at = tool 종료 예상 시점 − recovery_latency(tier, context_length) × margin

단순 rule-based policy (baseline):
  deepest tier t such that recovery_latency(t, L) ≤ predicted_duration × α
  (prefetch로 복구 latency를 duration 안에 숨길 수 있는 가장 깊은 tier 선택
   → 가장 비싼 메모리를 0의 가시적 TTFT 비용으로 해방)
  단, TTFT_cold(L) < TTFT_ssd(L) 이면 SSD 대신 EVICT
```

```
KV tier별 복구 비용 (실측 — §5.A breakeven map 스크립트):
  recovery(D_HBM)   = 0                  (유지)
  recovery(P_HBM)   ≈ append-prefill만   (prefix 전송 0; 시스템 구현 필요)
  recovery(CPU_DRAM)= PCIe 전송 + (P or D로의 경로 선택)
  recovery(SSD)     = SSD read + PCIe 전송
  recovery(EVICT)   = cold TTFT(L)       (context 길이에 비례)
```

Break-even threshold θ는 `recovery_latency(tier, L) = predicted_duration` 곡면에서 결정됨.

### 4.3 Tool Call Duration 분류 (BFCL 기반)

BFCL v3의 `involved_classes` 필드로 tool type별 latency class를 사전 분류할 수 있다:

| Tool | 예상 실측 latency | Class |
|------|----------------|-------|
| MathAPI | < 10 ms | fast |
| VehicleControlAPI | < 50 ms | fast |
| GorillaFileSystem | 50–500 ms | medium |
| MessageAPI | 100 ms–2 s | medium |
| TicketAPI | 1–5 s | slow |
| TradingBot | 1–10 s | slow |
| TravelAPI (DB 조회) | 2–15 s | slow |
| TwitterAPI | 1–5 s | slow |

현재 BFCL mock 환경에서는 모든 tool이 즉시 반환 → synthetic delay 주입으로 시뮬레이션하되,
**실제 trace 기반 분포(SWE-bench agent 로그 등)를 우선**한다 (§3.4 리스크 참조).
Continuum과 동일 워크로드(BFCL + SWE-bench류)를 쓰면 비교 가능성도 확보된다.

---

## 5. 실험 계획

### A. Break-even 맵: (tool duration × context length) → 최적 tier [최우선]

**목적**: 논문의 **Figure 1**. "최적 tier가 duration과 context 길이에 따라 실제로 바뀐다"를
한 장으로 증명. 기존 계획 A(re-prefill 비용 사다리) + B(inter-turn delay 주입)를 통합.

```
실험 설계:
  context 길이 L ∈ {~0.7k, 2k, 5k, 10k, 20k tokens} sweep:
    각 L에서 측정: TTFT_cold(L), TTFT_gpu(L), TTFT_dram(L), TTFT_ssd(L)
    (기존 sglang_kv_tier_latency.py의 4-phase 방법론을 길이축으로 확장)

  분석:
    recovery_penalty(tier, L) = first_miss(tier, L) − TTFT_gpu(L)
    duration 격자 d ∈ {0.1 … 60 s}에 대해:
      optimal_tier(d, L) = deepest tier with penalty ≤ d  (prefetch 모델)
    → 2D 맵 (with-prefetch vs without-prefetch 두 장 — prefetch의 가치 직접 시연)

결과물:
  break-even 맵 (heatmap) + tier별 bandwidth 실측치 → §4.2 비용 모델의 원료
```

**구현** (작성 완료):
```bash
bash scripts/sglang/start_1P_1D_breakeven.sh          # 1) 서버 시작 (1P+1D, hicache, router 8001)
SERVER_URL=http://127.0.0.1:8001 \
  python benchmark/sglang_kv_breakeven_map.py          # 2) 벤치마크 (상세는 docstring 참조)
```

**측정 가능 범위 주의**: 현 SGLang 구조에서 실측 가능한 사다리는 P-HBM(radix L1) / DRAM(L2)
/ SSD(L3) / cold = **Tier 2/3/4 + EVICT**다. Tier 1(D-HBM 유지)은 D 노드 append-prefill
메커니즘(PPD 방식)이 필요해 시스템 구현 전에는 직접 측정 불가 — L1(GPU radix hit)이
"HBM hit"의 하한 근사치 역할을 한다.

---

### B. Tier 간 KV 이동 마이크로벤치 (경합 곡선 포함)

**목적**: D-HBM↔P-HBM (Mooncake 경유), HBM↔DRAM, DRAM↔SSD의 bandwidth/latency를
KV 크기별 + **동시 전송 수별**로 측정 → 경합이 placement 결정을 바꾸는 지점 식별.

```
실험 설계:
  KV 크기 {0.5k, 2k, 8k, 32k tokens} × 동시 전송 {1, 2, 4, 8}
  Mooncake 경유 P→D 전송은 기존 buffer sweep 인프라 재사용
```

---

### C. GPU KV Idle 점유 분리 측정 + queueing 피해 정량화

**목적**: tool call 실행 중 GPU KV가 실제로 얼마나 낭비되는지 + **pin된 idle KV 때문에
다른 active 요청의 queueing delay/TTFT가 얼마나 증가하는지** 정량화 (motivation 섹션 근거).

```
실험 설계:
  TOOL_DELAY=10s 주입 + 배경 부하(C=8~16) 동시 인가:
    t=0s:   turn N 응답 직후 KV usage snapshot (= active KV)
    t=5s:   tool call 중간 KV usage snapshot (= idle KV)
    t=10+ε: turn N+1 요청 후 KV usage snapshot
  배경 부하 요청들의 TTFT/queueing delay를 idle KV 점유율과 상관 분석

  Grafana KV panel polling 1s + Prometheus pushgateway에 turn timestamp 전송

측정값:
  KV_idle(delay) = KV_usage during tool call / KV_peak
  ΔTTFT_background = f(KV_idle)  ← "idle 점유가 남을 해친다"의 직접 증거
```

---

### D. Tier별 KV 복구 Latency 직접 측정 (SGLang) — §2.2 이상 현상 규명 포함

**목적**: SSD→GPU KV 복구 latency를 TTFT 분해로 측정 + C=12 turn1 5.4× 느림의 원인
(SSD 경합 vs Mooncake 실패 vs queueing) 판별.

```
실험 설계:
  동일 prefix 연속 요청으로 차분 분리 (기존 설계 유지):
    SSD_read_latency ≈ TTFT_ssd - TTFT_gpu
    GPU_hit_benefit  ≈ TTFT_cold - TTFT_gpu
  + C=12 재현 시 iostat, Mooncake 에러 로그, hicache_stats.hit_count 동시 수집
```

---

### E. Oracle Placement Policy 상한 측정 [go/no-go 게이트]

**목적**: 완벽한 사전 지식이 있는 oracle이 달성하는 성능 = 개선 여지 상한선.
**상한이 작으면 여기서 멈추고 방향을 수정한다 — 시스템 구현 전 필수 게이트.**

```
Oracle 정의:
  각 요청의 실제 tool call duration을 미리 알고 있음
  §5.A break-even 맵에서 (d, L) → tier 직독

시뮬레이션:
  BFCL의 involved_classes로 oracle duration 할당 (또는 실 trace 분포)
  캐시 flush/유지 수동 조작으로 tier 상태 재현 (시스템 구현 불필요)

  측정: oracle policy vs LRU (현재) vs write_through (현재)
  지표: avg TTFT, P95 TTFT, GPU KV utilization, eviction count
```

---

### F. Tool Call Duration 예측기 + mispredict 비용

**목적**: policy의 실용성 근거. Continuum이 per-tool empirical CDF 예측의 feasibility를
이미 입증했으므로, 본 연구는 **예측 오차가 placement 성능에 미치는 영향(sensitivity)**에 집중.

```
접근법:
  per-tool empirical CDF 예측기 (Continuum 방식 차용) + tool type rule 비교

  Mispredict 비용의 비대칭성 측정:
    과소예측 (낮은 tier에 못 넣음) → 메모리 낭비 Y MB·s
    과대예측 (너무 깊은 tier)      → prefetch miss로 TTFT +X ms
  인위적 오차 주입 → sensitivity curve
  → 비대칭이 확인되면 보수적 tier 선택 policy의 근거
```

---

### G. Prefetch ablation (PD 고유)

**목적**: prefetch 유무 + **prefetch 경로 선택**(P로 가져와 append-prefill vs D로 직접)의
효과 분리 — §3.2의 고유 기여 3번을 보여주는 실험.

```
실험 설계:
  tool 종료 예상 시점 T−δ에 prefetch 시작, δ ∈ {0, 0.5×, 1×, 2×} × recovery_latency sweep
  경로 A: tier → P-HBM → append-prefill → Mooncake → D
  경로 B: tier → D-HBM → D에서 prefill (PPD 방식)
  부하별 (C=1 / C=8 / C=16) 두 경로의 TTFT 비교 → RQ4의 답
```

---

### H. End-to-end 비교

**목적**: 최종 평가. baseline 최소 셋:

```
baselines:
  (a) D-HBM 유지 + LRU         (vLLM/SGLang 기본)
  (b) write-through hicache    (현 SGLang)
  (c) InferCept식 무조건 CPU swap
  (d) Continuum식 TTL pin (pin-or-evict)
  (e) Oracle (상한)
  vs 제안 policy

sweep:
  부하 C ∈ {4, 8, 12, 16+}  (memory pressure 영역 필수)
  duration 분포 {short-heavy, long-heavy, bimodal}

지표:
  avg/P95 TTFT, TPOT, goodput (SLO attainment), 최대 유지 가능 동시성, GPU KV utilization
```

---

### I. Tier ablation

**목적**: "4-tier가 정말 다 필요한가"에 대한 답 — 없으면 over-engineering으로 보임.

```
2-tier (D-HBM + DRAM) vs 3-tier (+P-HBM) vs 4-tier (+SSD)
→ 각 tier의 한계 기여 분리:
  "P-HBM tier 추가 시 +X%, SSD는 DRAM pressure 시나리오에서만 +Y%"
SSD tier는 DRAM 캐시 용량 제한 실험으로 pressure 시나리오 구성
```

---

## 6. 실험 우선순위 및 의존 관계

```
[A] Break-even 맵 ─────────────► [E] Oracle 상한 (go/no-go 게이트)
     │                                  │
[B] Tier 마이크로벤치 (경합)            ▼
     │                          (개선 여지 충분?) ──No──► 방향 수정
[C] Idle KV + queueing 피해             │Yes
     │                                  ▼
[D] §2.2 이상 현상 규명          시스템 구현 (D 노드 KV export 경로)
                                        │
                                        ▼
                          [F] Predictor  [G] Prefetch  [I] Tier ablation
                                        │
                                        ▼
                                [H] End-to-end 비교
```

**단기 (1~2주)**: A (+B) → break-even 맵 = Figure 1, placement policy의 필요성 수치 증명  
**중기 (2~4주)**: C + D → motivation 완성, E → oracle 상한으로 go/no-go 결정  
**장기**: 시스템 구현 → F + G + I → H 최종 평가

---

## 7. 예상 Contribution

| Contribution | 설명 |
|-------------|------|
| **측정 연구** | PD disaggregation에서 agentic multi-turn 워크로드의 KV tier별 latency 사다리 + break-even 맵 최초 측정 |
| **분석** | Tool call duration × context 길이가 최적 tier를 결정하는 break-even 분석; "prefix KV는 P에 둘 것인가 D에 둘 것인가"(RQ4)에 대한 부하 의존적 답 |
| **Policy 설계** | Duration 예측 기반 4-tier placement + prefetch 경로 선택 policy (PD 고유) |
| **평가** | Oracle 대비 policy의 TTFT/GPU utilization tradeoff + tier ablation으로 각 tier의 한계 기여 정량화 |

**포지셔닝 주의**: 논문의 주장은 "4-tier storage 확장"이 아니라 **"PD disaggregation에서
tool-call-aware KV placement — 어느 노드, 어느 tier에, 언제까지"**로 잡는다.
4-tier는 결과이지 기여가 아니다. Continuum은 가장 좋은 발판: "Continuum은 pin-or-evict
이진 선택이고 단일 노드다; PD에서는 중간 tier들과 P 노드라는 선택지가 생기고,
전송 경합이 결정을 바꾼다"가 자연스러운 스토리.

---

## 8. 관련 결과 파일

| 파일 | 관련 근거 |
|------|---------|
| `results/vllm-ppd/bfcl_multiturn_results_vllm_ppd_2p2d_c8.json` | §2.3 KV idle 점유 (D node 17.2% max) |
| `results/sglang_hicache/bfcl_multiturn_results_sglang_2p2d_c*.json` | §2.2 SSD 경합, per-turn TTFT |
| `results/vllm/bfcl_multiturn_results_vllm_tp4.json` | §2.1 Turn별 TTFT drop (−46%) |
| `results/vllm-ppd/bfcl_multiturn_results_vllm_ppd_buf*_c*.json` | §2.4 Buffer sweep, context 길이 병목 |
| `benchmark/sglang_kv_tier_latency.py` | §5.A 방법론 원형 (단일 길이 4-phase 측정) |
| `scripts/sglang/start_1P_1D_breakeven.sh` | §5.A 서버 시작 (1P+1D + hicache + router 8001) |
| `benchmark/sglang_kv_breakeven_map.py` | §5.A break-even 맵 측정 + 맵 생성 (신규) |
| `results/sglang_hicache/{model}/kv_breakeven_map.json` | §5.A 결과 (실행 후 생성) |

## 9. References

- [Continuum/CacheTTL: Multi-Turn LLM Agent Scheduling with KV Cache Time-to-Live](https://arxiv.org/abs/2511.02230) (2025.11)
- [TokenCake: A KV-Cache-centric Serving Framework for LLM-based Multi-Agent Applications](https://arxiv.org/abs/2510.18586) (2025.10)
- [Not All Prefills Are Equal: PPD Disaggregation for Multi-turn LLM Serving](https://arxiv.org/abs/2603.13358) (2026.3)
- [AMPD: Efficient Multi-round LLM Inference over Disaggregated Serving](https://arxiv.org/abs/2602.14516) (2026.2)
- [KVFlow: Efficient Prefix Caching for LLM-Based Multi-Agent Workflows](https://arxiv.org/pdf/2507.07400) (2025.7)
- [AgentServeSim: A Hardware-aware Simulator for Multi-Turn LLM Agent Serving](https://arxiv.org/html/2606.09613) (2026.6) — 대규모 시나리오 보강용 시뮬레이터
- [Conveyor: Efficient Tool-aware LLM Serving with Tool Partial Execution](https://arxiv.org/pdf/2406.00059) (2024)
- [PrefillShare: A Shared Prefill Module for KV Reuse in Multi-LLM Disaggregated Serving](https://arxiv.org/abs/2602.12029) (2026.2)
- InferCept (ICML 2024), AttentionStore/CachedAttention (USENIX ATC 2024), Mooncake (FAST 2025)
