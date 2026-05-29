# Research: KV Cache Placement Policy for Agentic PD Disaggregation

> **연구 주제**: Single-model P-D disaggregation 환경에서 tool call duration을 관찰해  
> KV cache를 어느 tier에 배치할 것인지 결정하는 **placement policy** 설계.

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

이 직관을 실제로 구현하는 **demand-aware KV placement policy**가 연구 대상이다.

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

어느 가설이 맞는지 확인하려면 해당 시점의 `iostat`, Mooncake 에러 로그, SGLang hicache 내부 read latency 측정이 필요하다 (→ §4.D 참조).

**관찰 자체의 함의**: turn1이 turn0보다 느린 상황은, cache hit 여부와 무관하게 **고동시성에서 hicache의 이득이 사라질 수 있음**을 시사한다. 원인이 SSD I/O이든 KV transfer 실패이든, 이 패턴이 재현된다면 placement policy의 motivation이 된다.

### 2.3 Tool call 중 GPU KV는 idle — resource waste 정량화

vllm-ppd C=8 실험에서 측정한 D 노드 KV cache 점유율:

| Node | Max KV | Mean KV |
|------|--------|---------|
| D1 (GPU2) | **17.2%** | 5.9% |
| D2 (GPU3) | **15.5%** | 5.4% |

평균 3.7 turns/item, 각 turn 사이에 tool call이 실행된다. 이 시간 동안 KV는 D 노드 GPU에 남아 있으나 **실제 decode 연산은 진행되지 않는다** — 순수 idle 점유.  
Tool call duration이 길수록 이 낭비는 선형으로 증가한다.  
→ 측정은 아직 "전체 평균"이며 idle 구간만 분리한 측정이 필요하다 (§4.C 참조).

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

## 3. Research Design

### 3.1 연구 질문

> **RQ1**: Tool call duration에 따라 KV placement를 달리했을 때 TTFT와 GPU memory utilization이 동시에 개선되는가?

> **RQ2**: 어떤 features (tool type, turn depth, context length, concurrency)가 최적 placement를 결정하는 데 충분한가?

> **RQ3**: Oracle placement policy 대비 rule-based policy의 performance gap은 얼마인가?

### 3.2 Placement Policy 설계 공간

```
입력 features:
  f1. observed_tool_duration (현재/과거 같은 tool type의 실측치)
  f2. turn_depth (현재 conversation의 turn 번호)
  f3. context_length (현재 KV의 token 수)
  f4. current_gpu_pressure (GPU KV utilization)

출력:
  placement ∈ {GPU_DRAM, CPU_DRAM, SSD, EVICT}

단순 rule-based policy (baseline):
  if predicted_return_time < θ_fast:
    → GPU_DRAM (keep)
  elif predicted_return_time < θ_slow:
    → CPU_DRAM or SSD
  else:
    → EVICT (re-prefill on next turn)
```

```
KV tier별 예상 TTFT 이득 (측정 필요 — §4.A):
  GPU_DRAM hit:  turn0_TTFT × α_gpu   (α_gpu ≈ 0.35–0.55, 기존 측정)
  SSD hit:       turn0_TTFT × α_ssd   (α_ssd 미측정)
  EVICT (re-prefill): turn0_TTFT       (baseline, context 길이에 비례)
```

Break-even threshold θ는 `TTFT_benefit(tier) = TTFT_cost(evict)` 에서 결정됨.

### 3.3 Tool Call Duration 분류 (BFCL 기반)

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

현재 BFCL mock 환경에서는 모든 tool이 즉시 반환 → 실험 B에서 synthetic delay를 주입해 시뮬레이션.

---

## 4. 추가 실험 계획

### A. Re-prefill 비용 사다리 측정 [최우선]

**목적**: context 길이별 "GPU hit", "SSD hit", "full re-prefill" TTFT를 측정해 placement의 break-even threshold 계산.

```
실험 설계:
  조건 1 — GPU hit (warm):
    → 정상 실행, KV가 GPU에 남아 있는 상태에서 다음 turn 요청

  조건 2 — SSD hit (SGLang hicache):
    → turn 처리 후 L1/L2 flush, L3(SSD)만 남긴 상태에서 다음 turn 요청
    → SGLang: /tmp/hicache/ 유지, GPU DRAM 캐시만 삭제

  조건 3 — Cold re-prefill:
    → /tmp/hicache/ 포함 전체 캐시 삭제 후 동일 turn 재요청

  Context 길이 축: turn 0 (~1.5k chars), turn 2 (~1.9k), turn 4 (~2.4k)

측정값:
  TTFT_gpu(L), TTFT_ssd(L), TTFT_cold(L) at each context length L

결과물:
  break-even curve: TTFT_ssd(L) vs TTFT_cold(L)
  → "SSD hit가 re-prefill보다 빠른 context 길이 범위" 결정
```

**구현**: `benchmark/sglang_BFCL_v3_tiered_ttft.py` 신규 작성, `--cache-mode {gpu,ssd,cold}` 파라미터.

---

### B. Inter-turn Delay 주입 실험 [최우선]

**목적**: synthetic tool call delay가 각 tier에서의 TTFT에 미치는 영향 측정 → "언제 evict해도 되는가"의 threshold 결정.

```
실험 설계:
  benchmark에 TOOL_DELAY env var 추가:
    time.sleep(TOOL_DELAY) between turns

  TOOL_DELAY ∈ {0, 0.5, 1, 2, 5, 10, 30} 초
  각 delay에서: TTFT_turn1 측정 (GPU hit vs SSD hit vs cold)

  기대 결과:
    delay=0s:  GPU hit TTFT (base)
    delay=Xs:  LRU eviction 시작 → GPU hit TTFT 상승, cold와 수렴
    delay=30s: GPU LRU eviction 거의 확실 → GPU hit = cold

  측정값:
    TTFT_turn1(delay) per tier
    → break-even delay: TTFT_gpu(delay*) ≈ TTFT_cold(0)
       → delay* 이상이면 KV를 ssd/evict해도 손해 없음
```

**구현**: `benchmark/sglang_BFCL_v3_multi_turn_base.py`에 `TOOL_DELAY` 파라미터 추가 (5줄 변경).

---

### C. GPU KV Idle 점유 분리 측정

**목적**: tool call 실행 중 GPU KV가 실제로 얼마나 낭비되는지 정량화.

```
실험 설계:
  TOOL_DELAY=10s 주입 후:
    t=0s:   turn N 응답 직후 KV usage snapshot (= active KV)
    t=5s:   tool call 중간 KV usage snapshot (= idle KV)
    t=10s:  tool call 직전 KV usage snapshot
    t=10+ε: turn N+1 요청 후 KV usage snapshot

  현재 Grafana KV panel의 polling interval을 1s로 낮추고
  benchmark에서 Prometheus pushgateway에 turn timestamp 전송

측정값:
  KV_idle(delay) = KV_usage during tool call / KV_peak
  → "delay=10s이면 GPU KV의 X%가 idle 점유"
```

---

### D. Tier별 KV 복구 Latency 직접 측정 (SGLang)

**목적**: SSD→GPU KV 복구 latency를 TTFT 분해로 측정. 현재는 간접 측정만 가능.

```
실험 설계:
  동일 prefix를 연속 요청:
    요청 1: cold (TTFT = prefill time + KV write)
    요청 2: SSD hit (TTFT = SSD read + KV load + minimal prefill)
    요청 3: GPU hit (TTFT = GPU read + minimal prefill)

  차분으로 각 tier 기여 분리:
    SSD_read_latency ≈ TTFT_ssd - TTFT_gpu
    GPU_hit_benefit  ≈ TTFT_cold - TTFT_gpu

  SGLang /get_server_info의 hicache_stats.hit_count 활용해 hit 여부 확인
```

---

### E. Oracle Placement Policy 상한 측정

**목적**: 완벽한 사전 지식이 있는 oracle이 달성하는 성능 = 개선 여지 상한선.

```
Oracle 정의:
  각 요청의 실제 tool call duration을 미리 알고 있음
  duration < θ_fast  → KV를 GPU에 유지
  θ_fast ≤ d < θ_slow → CPU DRAM으로 이동
  d ≥ θ_slow         → SSD 또는 evict

시뮬레이션:
  실험 B의 결과에서 break-even threshold θ를 결정한 후,
  BFCL의 involved_classes로 oracle duration 할당:
    MathAPI, VehicleControl → fast (< θ_fast)
    TicketAPI, TravelAPI     → slow (> θ_slow)

  측정: oracle policy vs LRU (현재) vs write_through (현재)
  지표: avg TTFT, P95 TTFT, GPU KV utilization, eviction count
```

---

### F. Tool Call Duration 예측 가능성 검증

**목적**: policy의 실용성 근거 — tool type이 duration을 예측하는가?

```
접근법:
  BFCL의 involved_classes (8종)에 실제 API latency 분포 매핑
  → 각 tool class의 p50/p95 duration 측정 또는 문헌 조사

  예측 정확도 평가:
    coarse (fast/slow 2-class) 분류 정확도
    → 실제 agentic 서비스에서 tool type은 사전에 알 수 있음
       (API endpoint, function signature에서 추론 가능)
```

---

## 5. 실험 우선순위 및 의존 관계

```
[A] Re-prefill 비용 사다리 ──► [E] Oracle policy
         │
         ▼
[B] Inter-turn delay 주입 ──► (A + B → break-even threshold)
         │
         ▼
[C] GPU KV idle 측정       ──► (resource waste 정량화)
         │
         ▼
[D] SSD read latency       ──► (tier 비용 모델 완성)
```

**단기 (1~2주)**: A + B → break-even threshold 계산, placement policy의 필요성 수치로 증명  
**중기 (2~4주)**: C + D → 비용 모델 완성, E → oracle 상한 측정  
**장기**: F + policy 구현 및 A/B 실험으로 검증

---

## 6. 예상 Contribution

| Contribution | 설명 |
|-------------|------|
| **측정 연구** | Agentic multi-turn 워크로드에서 KV tier별 latency 사다리 최초 측정 |
| **분석** | Tool call duration이 KV placement의 optimal tier를 결정하는 break-even 분석 |
| **Policy 설계** | Tool type + turn depth + GPU pressure를 feature로 하는 rule-based placement policy |
| **평가** | Oracle policy 대비 rule-based policy의 TTFT / GPU utilization tradeoff 정량화 |

---

## 7. 관련 결과 파일

| 파일 | 관련 근거 |
|------|---------|
| `results/vllm-ppd/bfcl_multiturn_results_vllm_ppd_2p2d_c8.json` | §2.3 KV idle 점유 (D node 17.2% max) |
| `results/sglang_hicache/bfcl_multiturn_results_sglang_2p2d_c*.json` | §2.2 SSD 경합, per-turn TTFT |
| `results/vllm/bfcl_multiturn_results_vllm_tp4.json` | §2.1 Turn별 TTFT drop (−46%) |
| `results/vllm-ppd/bfcl_multiturn_results_vllm_ppd_buf*_c*.json` | §2.4 Buffer sweep, context 길이 병목 |
