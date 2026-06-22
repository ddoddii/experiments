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

### 2.7 Break-even 맵 1차 측정 (2026-06-11) — 현 시스템의 중간 tier는 무가치

§5.A 스크립트로 1P1D + hicache(file backend, /tmp = ext4 실디스크)에서 측정한
tier ladder (Qwen3-14B, 단위 ms):

| tokens | cold | L1 GPU | L2 first-miss | L3 first-miss | L1 이득 | L2 − cold | L3 − cold |
|---|---|---|---|---|---|---|---|
| 666 | 434 | 179 | 427 | 457 | **−59%** | −6 | **+24** |
| 2,000 | 1,169 | 329 | 1,158 | 1,287 | **−72%** | −11 | **+118** |
| 5,000 | 2,566 | 706 | 2,548 | 2,952 | **−72%** | −18 | **+385** |
| 10,000 | 5,105 | 1,345 | 5,104 | 5,972 | **−74%** | −1 | **+867** |
| 20,000 | 11,596 | 2,662 | 11,583 | 13,505 | **−77%** | −13 | **+1,910** |

**관찰 3가지:**

1. **L2 ≈ cold가 전 길이에서 0.1–1.5% 이내 일치** (10k: 5104.1 vs 5105.1, 1 ms 차).
   DRAM hit이라면 PCIe ~20 GB/s 기준 5k tokens(781 MB)에서 ~750 ms가 나와야 하는데
   2,548 ms → **L2 측정은 사실상 full re-prefill**.
2. **L3는 cold보다 항상 느리고 초과분이 KV 크기에 비례** (+0.23 → +0.61 ms/MB).
   유효 처리율 ~0.3–0.4 GB/s — re-prefill의 KV 생산 속도(~2,400 tok/s × 160 KB
   ≈ 0.4 GB/s)보다도 느림.
3. 방법론은 건강함: L1 분산 < 3%, L2/L3 재요청 median이 L1 수준으로 복귀
   (eviction → re-hit 사이클 정상 동작).

**결과**: break-even 맵이 "GPU 아니면 EVICT"로 퇴화 — RAM/SSD가 optimal인 칸이 0개.
**현 SGLang hicache 메커니즘 하에서는 Continuum의 pin-or-evict가 사실상 최적**임을
실측으로 확인한 셈.

**원인 가설 2개** (→ §5.A 카운터 수집으로 판별):

| 가설 | 근거 | 판별법 |
|------|------|--------|
| (a) hit 미발생 — 재계산 fallback | L2 = cold 정확 일치. 첫 iteration(666 tok)은 DRAM에 anchor가 분명 있어야 하는 상황(용량 충분)인데도 cold → 용량이 아닌 **hit 경로 문제**. 용의자: `hicache-write-policy` 기본값 selective(host에 안 써짐), `storage-prefetch-policy` best_effort(로드 포기 후 재계산) | first-miss 전후 `/server_info` 캐시 카운터 차분 — 카운터 무변화면 (a) |
| (b) hit은 났지만 load가 병적으로 느림 | file backend의 페이지(64 tok) 단위 sync read면 0.3 GB/s대 가능 | hit 카운터 증가 + TTFT ≈ cold면 (b) |

**기회 갭 정량화 (논문 framing)**: 측정된 유효 복구 대역폭 0.35–0.42 GB/s vs PCIe 실효
~20 GB/s = **약 50×**. 하드웨어 속도라면 5k tokens의 L2 패널티는 1,842 ms → ~40 ms,
L3는 ~260 ms가 되어 **맵의 중간 영역(d = 0.1–5 s) 전체가 RAM/SSD로 채워진다.**
→ "측정된 맵(GPU/EVT뿐)" vs "하드웨어 한계 기준 잠재 맵(RAM/SSD 지배)" 두 장을 나란히
놓는 것이 Figure 1의 강력한 형태.

**현 SGLang PD의 D 노드 KV 처리 (메커니즘 정리)**:
- hicache는 **P 노드 radix(prefix) cache의 확장**이라 P에만 의미가 있다. decode 모드
  서버는 prefix matching을 하지 않으므로 hicache가 동작하지 않는다.
- D 노드 HBM이 부족해지면 **offload가 아니라 retraction/preemption**(KV 폐기 후
  re-queue)이 일어난다. PD에서는 retract된 요청의 re-prefill이 P에서 다시 필요
  → 고동시성에서 관측한 KVTransferError/AbortReq(§2.4 성공률 저하)와 부합.
- Turn 사이의 "session KV"는 사실상 P 노드 radix+hicache에 산다 (다음 turn prefill이
  P에서 일어나므로). D 노드의 turn별 KV는 요청 종료와 함께 재사용 불가.
- → 제안하는 Tier 1(D-HBM) 관리와 D→하위 tier offload는 **현재 존재하지 않는 경로**이며
  시스템 기여 지점 (§3.4 구현 공수 행 참조).
  **(2차 측정 후 업데이트: sglang 0.5.9에 `--disaggregation-decode-enable-offload-kvcache`
  플래그 존재 확인 — §2.9 참조)**

### 2.8 Break-even 맵 2차 측정 (2026-06-11) — PD-특이성 기각, 1.2 GB/s 전송 세금 발견

명시적 정책(`write_through` + `wait_complete`)으로 1p1d 재실행 + 단일 서버(비-PD) 대조
실험 결과 (sglang 0.5.9, 결과: `results/sglang_hicache/Qwen3-14B/{1p1d,1server}_kv_breakeven_map.json`):

**(1) PD-특이성 기각.** 단일 서버에서도 L2 ≈ cold (+2~+41 ms, 0.2–1% 이내),
L3 = cold + Δ (Δ ∝ KV 크기, 환산 1.8–3.1 GB/s = NVMe 읽기 속도 모양) 패턴이 동일 재현.
→ hicache L2/L3 복원 실패는 PD 때문이 아니라 **hicache 메커니즘 자체의 문제**.
가설이 (b') **"storage 읽기는 일어나지만 결과가 compute를 대체하지 못함"**으로 좁혀짐
(읽고 + 어차피 재계산). 역설: 재계산의 KV 생산 속도는 0.33–0.5 GB/s라,
2–3 GB/s SSD 로드가 compute를 실제로 대체하기만 하면 **4–6× 빨라질 하드웨어가 낭비되는 중**.

**(2) 황금 측정치 — Mooncake P→D 전송 세금 ≈ 1.2 GB/s.** L1(GPU radix hit) 비교:

| tokens | L1 단일 서버 | L1 1p1d | 차이 | 환산 대역폭 |
|---|---|---|---|---|
| 666 | 52 ms | 165 ms | +113 ms | 0.92 GB/s |
| 2,000 | 55 ms | 327 ms | +272 ms | 1.15 GB/s |
| 5,000 | 62 ms | 698 ms | +636 ms | 1.23 GB/s |
| 10,000 | 75 ms | 1,383 ms | +1,308 ms | 1.19 GB/s |
| 20,000 | 96 ms | 2,735 ms | +2,639 ms | 1.18 GB/s |

단일 서버 L1은 52–96 ms로 평평(진짜 GPU hit), PD의 L1은 context에 비례 —
**PD에서는 완벽한 prefix hit조차 매 turn 전체 KV를 P→D로 전송**하기 때문.
20k tokens에서 "hit" 비용의 96%가 전송(28×). cold의 PD-단일 차이는 이보다 작음
→ prefill과 전송이 chunk 단위로 겹쳐지지만, hit일 때는 겹칠 compute가 없어 전송이 그대로 노출.

**논문 framing**: Tier 1(tool call 동안 D-HBM 유지)이 정확히 이 세금을 0으로 만든다.
단일 서버 L1(52–96 ms) = Tier-1 hit의 proxy, 1p1d L1 = 현 PD의 최선.
→ **Tier 1 retention의 가치 = turn당 최대 2.6 s (20k tok), 이미 실측 완료.**
부수 발견: SSD 읽기(2–3 GB/s) > Mooncake 경로(1.2 GB/s) → "D가 storage에서 직접 읽는"
prefetch 경로(RQ4)가 P 경유보다 나을 수 있음을 데이터가 뒷받침.

**(3) `wait_complete`는 PD에서 재앙적.** 1p1d L2/L3 first-miss가 24–126 s
비단조·간헐 스톨 (666 tok L3 = 24 s, 5k L2 = 43 s, 5k L3 = 126 s; 20k는 정상 복귀).
1차(기본 best_effort)에서는 L2 ≈ cold로 깔끔했으므로 PD + wait_complete 조합이
storage prefetch 대기/타임아웃 스톨 유발. → **PD에서는 best_effort 고정** (스크립트 반영).

**(4) 카운터 진단 실패 (한계).** `/server_info` 캐시 카운터 차분이 COLD에서조차 전무
→ 이 버전은 HTTP로 hicache 카운터를 노출하지 않음 (`sglang_hicache_exporter.py`가
VmRSS/du로 우회 추정하는 것과 같은 이유). 결정적 판별은 서버 로그 grep으로 수행:
`grep -iE "hicache|prefetch|storage" logs/sglang_single/server.log`

**(5) 전송 세금의 정체 규명 (2026-06-12).** 1.2 GB/s는 하드웨어 한계가 아니라
**전송 스택의 산물**임을 확인:

- 서버 로그: `No RDMA devices found ... Found 0 HCAs → TcpTransport: listen on ...`
  — RDMA NIC 부재로 Mooncake가 **TCP loopback fallback**. 경로:
  GPU(P) → host memcpy → TCP socket → host memcpy → GPU(D). NVLink/PCIe P2P 미사용.
- `nvidia-smi topo -m`: NVLink 쌍은 **GPU0↔GPU1, GPU2↔GPU3 (NV4)**, 그 외는
  NODE(PCIe host bridge 경유). 기존 구성(P=GPU0, D=GPU2)은 NVLink 쌍을 가로지름 —
  물리적으로도 NVLink 경로가 아니었음. (2P2D 구성 P={0,1}/D={2,3}은 가능한 P→D
  조합 4개 전부 NVLink 밖.)

→ **같은 머신에서 전송 대역폭 3-포인트 sweep이 가능**:

| 구성 | 경로 | 대역폭 | pen.T2 @20k 예상 |
|---|---|---|---|
| mooncake (TCP fallback) | GPU→host→TCP→host→GPU | **1.2 GB/s (실측)** | 2,466 ms (실측) |
| nixl + P=0, D=2 | PCIe P2P (NODE) | ~12–20 GB/s | ~160–260 ms |
| nixl + P=0, D=1 | **NVLink NV4** | ~40–50 GB/s | ~60–80 ms |

→ 대역폭 파라미터별 break-even 맵을 전부 실측으로 확보 가능 (robustness 섹션 재료).
스크립트 노브: `P_GPU` / `D_GPU` / `TRANSFER_BACKEND`, 결과 구분: `RUN_TAG`.
**함의**: 전송 세금은 배포 의존적이지만, 어떤 스택에서도 context 비례 + 부하 시
경합(§2.4)이므로 sub-second tool call 영역에서 T1(D-HBM 유지)의 가치는 유지된다.
단, T1 영역의 경계(= pen.T2)는 대역폭에 따라 좌우로 이동 — 논문에서는 bandwidth를
파라미터로 제시할 것.

### 2.9 설계 결정: 3-tier 축소 (D-HBM / P-HBM / CPU DRAM) + EVICT

SSD(Tier 4)를 평가 범위에서 제외하고 **3-tier + EVICT**로 축소한다. 근거:

| 근거 | 측정치 |
|------|--------|
| SSD는 현재 재계산보다 느림 | L3 − cold = +39 ~ +1,766 ms (양쪽 구성 모두, §2.7–2.8) |
| SSD는 고쳐져도 DRAM 대비 ~10× 느림 | SSD 2–3 GB/s vs PCIe DRAM ~20 GB/s |
| DRAM 용량이 이 규모에선 충분 | RAM 125 GB → KV용 ~80 GB = 512k tokens = 10k-token 세션 50개 동시 파킹 |
| Novelty가 상위 3개 tier에 집중 | D-HBM(1.2 GB/s 세금 절약 — 실측), P-HBM(prefix 전송 0, RQ4), DRAM(용량). SSD는 AttentionStore가 이미 한 가장 덜 새로운 tier |

- **Formulation은 N-tier 일반**으로 쓰고 평가만 3-tier — SSD는 discussion에서
  "DRAM pressure가 큰 대규모 배포의 자연스러운 확장"으로 한 단락.
- **EVICT는 유지** — 짧은 context에서 재계산이 여전히 경쟁력 있음(666 tok에 213 ms).
  맵은 {T1: D-HBM, T2: P-HBM, T3: DRAM, EVICT} 4-way 선택.
- 실용 이득: 고장난 hicache file backend가 연구의 blocking dependency에서 빠짐
  (§3.4 "SSD tier 정당화" 리스크 해소).

**3-tier 사다리 현황** (절반은 이미 측정됨):

| Tier | Resume 비용 | 출처 |
|------|------------|------|
| T1: D-HBM 유지 | ~52–96 ms (context 무관) | 단일 서버 L1이 proxy (§2.8) |
| T2: P-HBM (radix) | 165 ms → 2,735 ms (∝ L, 1.2 GB/s 전송 포함) | 1p1d L1 실측 |
| T3: CPU DRAM | T2 + DRAM 로드 (~수십 ms @20 GB/s, **L2 수리 후 측정 필요**) | 미확보 |
| EVICT | 427 ms → 11.4 s (∝ L) | cold 실측 |

**메커니즘 발견 — D→storage offload가 이미 존재**: sglang 0.5.9의
[`--disaggregation-decode-enable-offload-kvcache`](https://github.com/sgl-project/sglang/issues/11016)
플래그는 decode 노드가 생성한 KV를 비동기로 hicache storage에 기록해 **P가 multi-turn에서
재사용**하게 한다 = D→하위 tier 경로의 출발점이 mainline에 있음. 단 multi-turn 부하에서
CUDA error 보고([#11016](https://github.com/sgl-project/sglang/issues/11016)) — 안정성 검증
필요. 동작하면 Tier-1 관리의 구현 공수가 "scheduler 수정"에서 "기존 경로 + 정책"으로 급감.

**D-offload 검증 메모 (2026-06-11, 1차 시도)**: decode 서버에 hicache storage 설정 없이
플래그만 켜면 즉사 — `DecodeKVCacheOffloadManager`가 `HiCacheController`에
`server_args.hicache_storage_backend`를 넘기는데, `ack_backup_queue`는 storage backend가
설정된 경우(`_start_storage_threads`)에만 생성되므로 `check_offload_progress()`에서
`AttributeError: 'HiCacheController' object has no attribute 'ack_backup_queue'`.
→ **decode 서버에도 `--hicache-storage-backend` + `--hicache-ratio` 필요**
([#11016](https://github.com/sgl-project/sglang/issues/11016) 보고자도 동일 구성:
`--hicache-ratio 1.2 --hicache-storage-backend mooncake`). 단일 호스트에서는 file 백엔드로
P와 `/tmp/hicache`를 공유하면 P가 D의 offload KV를 읽을 수 있음 (멀티노드는 mooncake store).
스크립트 반영 완료 (`D_OFFLOAD_KVCACHE=1`이 decode에 storage 플래그 자동 부착).
#11016의 본 크래시(CUDA illegal memory access, `transfer_kv_all_layer_lf_pf` 커널)는
이후 부하 단계에서 재현 여부 관찰 — 재현 시 `D_HICACHE_MEM_LAYOUT`으로 layout을 바꿔
다른 전송 커널 경로 시도.

**다음 단계 (§5.A 후속과 통합):**
```
1. D-offload 검증: D_OFFLOAD_KVCACHE=1 bash scripts/sglang/start_1P_1D_breakeven.sh
   → BFCL multi-turn + tool delay로 동작/안정성 확인 (#11016 재현 여부)
   → 동작 확인 포인트: d1.log에 storage backend 생성 로그, decode 중 /tmp/hicache
     파일 수 증가, turn N+1에서 P가 D의 생성 토큰 KV를 재사용하는지 (TTFT 비교)
2. L2(DRAM) hit 살리기 — 단일 서버에서:
   HICACHE_RATIO=3.0 HICACHE_MEM_LAYOUT=page_first bash scripts/sglang/start_single_hicache.sh
   (대안: HICACHE_MEM_LAYOUT=page_first_direct HICACHE_IO_BACKEND=direct)
   → 로그 grep으로 host write/load 이벤트 확인, T3 진짜 비용 확보
3. L2가 살아나면 3-tier 맵 재생성:
   SKIP_L3=1 T1_REF_JSON=results/sglang_hicache/Qwen3-14B/1server_kv_breakeven_map.json \
     SERVER_URL=http://127.0.0.1:8001 python benchmark/sglang_kv_breakeven_map.py
   → kv_breakeven_map_3tier.{json,png} (T1 proxy / T2 / T3 / EVT)
```

### 2.10 3-tier Break-even 맵 1차 (2026-06-11) — 가설 1차 입증

§2.9의 3-tier 모드로 PD(router 8001) 측정 + 단일 서버 T1 proxy 결합 결과
(`results/sglang_hicache/Qwen3-14B/kv_breakeven_map_3tier.{json,png}`):

| tokens | T1 (D-HBM proxy) | T2 (P-HBM+전송) | T3 (DRAM) | cold | pen.T2 | pen.EVT |
|---|---|---|---|---|---|---|
| 666 | 52 ms | 166 ms | 399 ms | 385 ms | 114 ms | 332 ms |
| 2,000 | 55 ms | 326 ms | 1,137 ms | 1,130 ms | 271 ms | 1,075 ms |
| 5,000 | 62 ms | 672 ms | 2,454 ms | 2,453 ms | 610 ms | 2,391 ms |
| 10,000 | 75 ms | 1,325 ms | 4,879 ms | 4,882 ms | 1,250 ms | 4,807 ms |
| 20,000 | 96 ms | 2,562 ms | 11,179 ms | 11,135 ms | 2,466 ms | 11,040 ms |

**보정 맵** (아래 아티팩트 수정 후):

```
tokens \ d(s) │  0.1  0.2  0.5    1    2    5   10   30   60
          666 │   T1   T2  EVT  EVT  EVT  EVT  EVT  EVT  EVT
        2,000 │   T1   T1   T2   T2  EVT  EVT  EVT  EVT  EVT
        5,000 │   T1   T1   T1   T2   T2  EVT  EVT  EVT  EVT
       10,000 │   T1   T1   T1   T1   T2  EVT  EVT  EVT  EVT
       20,000 │   T1   T1   T1   T1   T1   T2   T2  EVT  EVT
```

**해석 — 연구 가설("duration × context 길이가 최적 placement를 결정")의 첫 실측 증거:**

1. **T1 영역 (d < pen.T2)**: tool call이 Mooncake 전송(0.11–2.5 s)보다 짧으면 D-HBM 유지 외
   대안 없음. context가 클수록 영역 확장 (20k는 d=2s까지). BFCL fast/medium 클래스
   tool call 대부분이 이 영역.
2. **T2 밴드 (pen.T2 ≤ d < pen.EVT)**: 전송은 숨겨지지만 재계산은 못 숨기는 구간 —
   **P-HBM tier가 optimal인 영역이 실존함을 입증.**
3. **Mooncake 세금 3번째 재현**: pen.T2 환산 0.91–1.28 GB/s (3회 독립 실행 일관 — 재현성 확보).
4. **T3(DRAM)는 여전히 사망**: T3 ≈ cold 0.1% 이내 (P 노드 ratio=1.2/layer_first 그대로).
5. **T1은 proxy 측정** (단일 서버 L1) — 실제 D-HBM retention 메커니즘이 아닌 하한 근사임을
   논문에 명시할 것.

**아티팩트 보정**: 원본 맵의 10k row RAM 칸 4개는 pen.T3(4803.6) vs pen.EVT(4807.2)가
**3.6 ms 동률**이라 EVICT 게이트가 코인플립한 것. fetch ≈ recompute 동률(50 ms 마진 내)이면
EVICT로 판정하도록 수정 (`EVT_TIE_MARGIN_MS`, 기본 50 ms).

**L2(DRAM) 수리 후 예측 맵** — pen.T3′ = pen.T2 + KV/20 GB/s (PCIe) 가정:

```
tokens \ d(s) │  0.1  0.2  0.5    1    2    5   10   30   60
          666 │   T1  RAM  RAM  RAM  RAM  RAM  RAM  RAM  RAM
        2,000 │   T1   T1  RAM  RAM  RAM  RAM  RAM  RAM  RAM
        5,000 │   T1   T1   T1  RAM  RAM  RAM  RAM  RAM  RAM
       10,000 │   T1   T1   T1   T1  RAM  RAM  RAM  RAM  RAM
       20,000 │   T1   T1   T1   T1   T1  RAM  RAM  RAM  RAM
```

(예: 20k에서 pen.T3′ = 2,466 + 153 ≈ 2,619 ms vs pen.EVT 11,040 ms → EVT/T2 영역 대부분이
RAM으로 전환.) **"현재 맵(EVT 지배) vs 수리 후 맵(RAM 지배)" before/after 쌍 = Figure 1의
가장 강력한 형태.** 이 예측이 §2.9 step 2(L2 살리기)의 검증 목표를 정량 정의:
실측 pen.T3가 pen.T2 + ~50–150 ms에 근접하는가.

**측정 방법론 주의 (시행착오 기록)**: PD 측정은 반드시 router(8001)로 보낼 것.
P 노드(30000)에 직접 보내면 standalone으로 처리되어 단일 서버와 동일한 수치가 나옴
(전송 세금 미포함) — 1회 잘못 측정 후 재실행으로 확인.

### 2.11 NVLink P2P 전송 실측 (2026-06-12) — 전송 세금 23× 감소, 맵 구조 전환

§2.8(5)의 3-포인트 sweep 중 NVLink 포인트 실측: **NIXL 백엔드 + P=GPU0, D=GPU1 (NV4)**.
결과: `results/sglang_hicache/Qwen3-14B/nixl_nvlink_kv_breakeven_map*.{json,png}`

| tokens | T1 | T2 (NVLink) | T2 (TCP, §2.10) | pen.T2 NVLink | pen.T2 TCP | 환산 BW |
|---|---|---|---|---|---|---|
| 666 | 52 ms | 90 ms | 166 ms | **38 ms** | 114 ms | 2.7 GB/s |
| 2,000 | 55 ms | 94 ms | 326 ms | **39 ms** | 271 ms | 7.9 GB/s |
| 5,000 | 62 ms | 112 ms | 672 ms | **50 ms** | 610 ms | 15.4 GB/s |
| 10,000 | 75 ms | 137 ms | 1,325 ms | **62 ms** | 1,250 ms | 24.7 GB/s |
| 20,000 | 96 ms | 200 ms | 2,562 ms | **105 ms** | 2,466 ms | **29.1 GB/s** |

**관찰:**

1. **NVLink P2P 동작 확인**: 대형 전송에서 유효 29.1 GB/s — PCIe 4.0 x16 실효(~25 GB/s)를
   초과하므로 NVLink 경로가 실제 사용됨이 증명됨 (NV4 단방향 피크 ~56 GB/s의 52%).
2. **전송 비용 구조 = 고정 오버헤드 ~38 ms + 크기/BW**: 소형 전송은 latency 지배
   (666 tok에 2.7 GB/s), 대형은 bandwidth 지배. pen.T2가 20k에서 2,466 → 105 ms (**23×**).
3. **맵 구조 전환 (예측대로)**: T1 영역이 사실상 소멸 (20k @ d=0.1 한 칸), T2가 d=0.1부터
   전 길이 지배. NVLink same-node에서는 **P-HBM 유지가 거의 무료** — 0.1 s tool call에도
   전송이 숨겨짐.
4. **DRAM tier는 여전히 사망**: T3 ≈ cold (2k–20k에서 ±0.5% 이내). 666 row의 RAM 칸은
   cold 단일 샘플 변동(402 vs l2 332 — l2가 재계산 수준) 아티팩트로 판정.
5. T2의 절대값(90–200 ms)은 T1 proxy(52–96 ms)의 1.4–2.1× — 무부하 기준으로는 근접.

**함의 (정직한 업데이트):**

- 전송 세금이 배포 의존적임이 **양 끝점 실측**으로 확정 (TCP 1.2 ↔ NVLink 29 GB/s).
  → T1(D-HBM 유지)의 가치는 deployment의 함수: TCP/no-RDMA/혼잡 환경에서 크고,
  NVLink same-node 무부하에서는 작다. **placement policy가 transfer bandwidth를
  입력 feature로 가져야 한다**는 적응형 policy 논거가 오히려 강화됨 (§4.2 f5와 합류).
- 남은 T1 논거: ① 고정 ~38 ms 오버헤드 + ② 부하 시 전송 경합 (§2.4 buffer 병목 —
  NIXL에서 동시성 sweep으로 검증 필요) + ③ 멀티노드/비-NVLink 배포.
- 다음: (a) PCIe 포인트 (D_GPU=2, RUN_TAG=nixl_pcie)로 3-포인트 완성,
  (b) NIXL + C=4~16 동시성에서 pen.T2 경합 곡선, (c) L2 살리기 (§2.9 step 2 — 여전히
  RAM 영역의 gating item).

### 2.12 P/D HBM 점유 시계열 (2026-06-15) — "P node idle HBM" 전제 1차 검증

NVLink 1P1D + `--enable-metrics`로 합성 multi-turn agent(8 turn, tool 10s, C=1)를
구동하며 P/D KV 점유를 0.5s 간격 동시 폴링. 도구: `benchmark/pd_hbm_occupancy.py`,
결과: `results/sglang_hicache/Qwen3-14B/nvlink_c1_v2_pd_hbm_occupancy.{csv,png}`.

**측정 셋업 교훈 (시행착오)**:
- SGLang은 `enable_metrics=False`(기본)면 `/metrics`가 비어 live `num_used_tokens`/
  `token_usage`를 HTTP로 못 읽음 → 정적 capacity만 나옴. **start 스크립트에 `--enable-metrics`
  필수** (반영 완료). 1차 실행은 이게 빠져 로그 기반 거친 데이터였음(P distinct 5개).
- metric 이름: `sglang:num_used_tokens`, `sglang:token_usage` (`pending_prealloc_token_usage`,
  `swa_token_usage`와 혼동 주의 → 파서 앵커링).

**관측 (C=1)**:

| | capacity | peak used | 사용률 |
|---|---|---|---|
| P node | 14.0 GB (85,172 tok) | 1.29 GB | **9.2%** |
| D node | 14.0 GB | 1.50 GB | 10.8% |

1. **P HBM은 8 turn 끝까지 pool의 9.2%만 사용 → 90%+ idle.** "P node idle HBM을 Tier2로"
   전제가 C=1에서 수치로 성립.
2. **P 점유가 turn마다 증가**(0.2→0.39→0.93→1.11→1.29 GB)하는 것은 버그 아님 —
   `write_through` hicache + radix cache가 누적 prefix KV를 P에 보존하기 때문(62s에서
   radix LRU eviction으로 0으로 떨어졌다 재성장). 증가해도 여전히 <10%.
3. **P→D 전송은 incremental**: turn마다 +967 tok(~0.15 GB)만 전송(전체 context 재전송 아님).
   D는 누적 성장(0.24→1.5 GB). → **P·D 양쪽이 세션 prefix를 보존하고 새 토큰만 이동.**
   (이전 §2.8/v1의 "D가 turn마다 0으로 비운다" 해석은 노이즈 로그의 stale 아티팩트 — 정정.)
4. **P가 이미 prefill prefix를 보유** → "Tier2(P-HBM)가 부분적으로 무료"; novel 전송 대상은
   D의 **생성 토큰 KV**(P가 가진 적 없는 부분). 마이그레이션 볼륨을 줄이는 설계 포인트.

**측정 한계 (정직)**: P·D **gauge는 engine이 idle하면 forward pass가 없어 갱신을 멈춤**
(마지막 값 유지). 따라서 tool-idle 구간의 평탄 부분은 "실제 보존"과 "gauge freeze"가 섞임 —
gauge만으로 "tool call 중 free 여부"를 깨끗이 못 읽음. incremental 전송(+967) 증거상 보존이
우세하나, 엄밀히는 메모리 풀 allocator 직접 probe 필요. (C>1에서는 세션 tool window가
겹쳐 항상 누군가 active → freeze 완화되는 부수 효과.)

**다음 (진행 중): Concurrency sweep C=4/8/16** — correlated-demand 리스크 검증.
`CONCURRENCY` 노브로 C개 세션 병렬 구동, 각 C의 P peak 점유% 비교.
- 가설 H1(전제 성립): D가 prefill+decode로 바쁜 고동시성에서도 P는 prefill 버스트만
  처리하므로 P 점유가 낮게 유지(음의/무 상관) → Tier2 가용.
- 가설 H2(전제 약화): C↑로 P의 prefill 부하가 누적돼 P도 차오름(양의 상관) → "D 압박 =
  P도 압박"이면 Tier2가 필요한 순간에 비어있지 않음. 이 경우 "tool-call 세션은 active
  prefill이 아니므로 위상차가 있다"는 더 정교한 논거 필요.
```
for C in 4 8 16; do
  DRIVE=1 CONCURRENCY=$C NUM_TURNS=8 TOOL_DELAY_SEC=10 OUT_TAG=nvlink_c$C \
  D_LOG=logs/sglang_1p1d/d1.log P_LOG=logs/sglang_1p1d/p1.log \
    python benchmark/pd_hbm_occupancy.py
done
```

**Sweep 결과 (2026-06-15) — H1 확인: 전제 성립.**
`results/sglang_hicache/Qwen3-14B/nvlink_c{4,8,16}_pd_hbm_occupancy.{csv,png}`

| C | D running | D peak 점유 | **D peak 순간 P** | P peak (별도 시점) | corr(P,D) |
|---|---|---|---|---|---|
| 4 | 4 (큐 0) | 46% | **9.9%** | 11.5% @t=68s | +0.14 |
| 8 | 8 (큐 0) | **91%** | **9.9%** | 11.5% @t=89s | +0.44 |
| 16 | 16 (큐 0) | **94%** | **17.4%** | 23.2% @t=223s | +0.20 |

전 구간 세션 100% 완료(C=16: 128/128 turn), D running이 4→8→16으로 큐잉 없이 붙음
(측정 아티팩트 없음).

**결정적 관찰**: D가 포화되는 순간(C=8: 91%, C=16: 94% — KV를 내보내고 싶은 바로 그 순간)
**P는 9.9~17.4%만 차 있어 HBM의 83~90%가 비어 있음.** Tier2가 필요한 시점에 정확히 가용.

1. **D는 C=8에서 이미 포화(91%)**: multi-turn 세션이 KV를 누적 보존 → modest 동시성에서
   memory-pressure 영역 도달 = motivation 실재.
2. **P는 sub-linear 증가**(11.5→11.5→23.2%), D보다 항상 4× 낮음. P가 차는 이유는 동시
   세션 수만큼 radix가 서로 다른 prefix를 보존하기 때문.
3. **P·D peak이 다른 시점**(C=8: P peak일 때 D 18%, D peak일 때 P 9.9%) — prefill 버스트와
   decode 누적의 위상차. 약한 양의 상관(+0.14~0.44)은 전체 누적 추세 탓이지 peak이 겹치는 게 아님.

**정직한 단서 (발표/리뷰 대비)**:
- C=16에서 P가 23%로 오른 것은 mild correlated demand 신호. 극단 부하(C=32+)/세션 수백 개에선
  P radix도 차오를 수 있음. **반론**: P radix는 prefill 캐시라 공격적 evict 가능(KV가 이미 D로
  전송됐거나 재계산 가능) → P-HBM을 Tier2로 비우는 게 D-HBM 비우기보다 훨씬 쌈.
- **"D를 늘리거나 P:D 비율 재조정하면?"**: 이 imbalance는 워크로드 의존적(decode-heavy
  multi-turn이라 D-bound). 정적 재배치는 turn-by-turn tool-call 패턴을 못 따라감. NVLink로
  붙은 P idle HBM(이미 있고 공짜)을 tool-call-aware하게 쓰는 게 정적 프로비저닝이 못 하는 것
  = 본 연구 포지셔닝.

**판정**: 전제 성립. "D 포화 압박 시 P-HBM 83~90% idle"이 C=4/8/16 전부 실측,
D는 modest 동시성에서 이미 포화. P 23% vs D 94%(C=16)의 격차 = tiering이 활용할 headroom.

**다음 단계**:
```
(a) 한계 탐색 — C=32 (이상): P radix가 언제 차는지. P 사용률이 D에 근접하는 동시성이
    Tier2 전략의 손익분기. correlated-demand가 깨지는 지점 식별.
    DRIVE=1 CONCURRENCY=32 NUM_TURNS=8 TOOL_DELAY_SEC=10 OUT_TAG=nvlink_c32 \
      python benchmark/pd_hbm_occupancy.py
    (주의: C=16에서 D 94% 포화 → C=32는 D retraction/abort 발생 예상. 세션 완료율,
     D_queue, abort를 함께 봐야 P 거동 해석 가능. mem_fraction 조정으로 D pool을 키워
     "P만 차는" 영역을 분리하는 것도 방법.)

(b) P radix evict 비용 측정 — Tier2(P-HBM)를 비우는 실제 비용.
    P-HBM을 Tier2로 쓰려면 D의 KV를 받기 위해 P가 자기 radix를 evict해야 할 수 있음.
    그 비용 = (i) evict된 prefix를 다음 turn에 재계산하는 re-prefill TTFT 증가
            (ii) host(DRAM)로 write-back하는 비용 (write_through면 이미 host에 있어 0)
    측정: P radix를 강제 evict(flood)한 뒤 같은 세션 다음 turn의 TTFT vs evict 안 한 경우.
    → "P-HBM Tier2 비우기 비용"을 §2.10 break-even 비용 모델에 입력으로 추가.
    핵심 가설: write_through라 evict는 거의 무료(host에 복제본 존재) → P-HBM은 D-HBM보다
    훨씬 싼 양보 자원임을 정량화.
```

### 2.13 SGLang 메모리 압박 처리 메커니즘 — 본 연구 mechanism이 들어갈 자리

서버 인자 덤프 + 로그 + upstream 이슈로 확인한 SGLang decode의 압박 처리 **3단계**
(`radix_eviction_policy='lru'`, `total_retractions` 추적 확인):

1. **LRU radix eviction** [기본 동작]: ref_count=0인 캐시 KV(= turn 사이/tool-call 중인
   idle 세션 KV)를 LRU로 버림. 1차 방어선.
2. **Retraction (preemption)**: 부족하면 *실행 중* 요청의 KV를 회수→큐로 되돌림.
   abort 아님, **re-prefill로 재개**. `#retracted-req`/`total_retractions`로 추적.
   경로: `release_req → offload_kv_cache → get_cpu_copy` (회수 KV를 CPU로 offload).
3. **Abort**: 특정 실패만 — PD `KVTransferError: Aborted by AbortReq` (KV 전송 후 decode
   OOM, prealloc 큐 정체). 고동시성 PD 알려진 버그: [#6857](https://github.com/sgl-project/sglang/issues/6857),
   [#9266](https://github.com/sgl-project/sglang/issues/9266), [#10111](https://github.com/sgl-project/sglang/issues/10111).

**실측 정합**: §2.12 C=16은 D 94%에도 `#retracted-req=0`, `D_queue=0`, 128/128 완료
(tool window가 KV를 풀어줘 retraction 없이 버팀). C=32는 retraction부터 시작, 더 밀면 위 버그.
→ C=32 한계 실험은 `mem_fraction`↑로 D 포화를 미뤄 "P만 차는 영역"을 abort와 분리해야 함.

**본 연구 mechanism의 통합 지점 (핵심)**: LRU evict의 결과는 "버림→다음 turn re-prefill".
본 연구는 LRU를 *대체하지 않고*, **LRU가 고른 demote 대상의 목적지를 바꾼다**:

| | 기존 SGLang | 본 연구 |
|---|---|---|
| WHAT을 demote | LRU가 idle 세션 KV 선택 | 동일 — LRU를 trigger로 재사용 |
| WHERE로 | 버림 → re-prefill (cold TTFT) | **P-HBM (Tier2, NVLink)** → migrate (pen.T2) |

→ contribution = "evict-to-recompute"를 **"LRU-triggered migration to Tier2"**로 전환.
그리고 retraction의 `offload_kv_cache → get_cpu_copy` hook(= `--disaggregation-decode-enable-
offload-kvcache`가 쓰는 경로)의 **목적지를 CPU 대신 P-HBM으로 리다이렉트**하면 됨 —
밑바닥 구현이 아니라 destination 교체 (§3.4 "구현 공수 완화"의 구체화).

### 2.14 주장 보강용 추가 실험 설계 (2026-06-15)

§2.12가 "P idle 가용"을 보였다면, 아래는 **마이그레이션이 실제로 이득인가**를 직접 증명.

**E1. Tier2 migration의 end-to-end 이득 [최우선 — 핵심 주장의 직접 증거]**
- 목적: "tool-call 중 idle KV를 P-HBM에 두면 LRU evict(=re-prefill) 대비 TTFT 이득" 정량화.
- 설계: D 포화(C=8~16)에서 multi-turn 진행. 두 조건 비교:
  - baseline: 압박 시 LRU evict → 다음 turn TTFT (= cold re-prefill, §2.10 cold)
  - 제안: idle 세션 KV를 P-HBM 유지 → 다음 turn TTFT (= pen.T2, §2.11 NVLink ~38–105ms)
- 메커니즘 미구현 단계에서는 **emulation**: P radix에 prefix를 유지(evict 막기) vs 강제 evict
  후 같은 turn 재요청. TTFT 차이 = migration이 숨기는 비용.
- 산출: turn별 ΔTTFT(evict − migrate) × context 길이 → "절감 곡선". §2.10/2.11 데이터로
  이미 부분 예측됨(evict pen.EVT vs migrate pen.T2). E1은 실 워크로드에서 직접 측정.
- 구현: `benchmark/e1_emulation.py` (EVICT=tool window에 `/flush_cache`, RETAIN=idle,
  migrate=RETAIN+pen.T2 합성). 서버: NVLink 1P1D(`start_1P_1D_breakeven.sh`).

**E1 실측 결과 (2026-06-15)** — NVLink, 8 turn, tool 6s, evict=flush, pen.T2=38+3.35/1k tok.
결과: `results/sglang_hicache/Qwen3-14B/nvlink_e1_migration_benefit.{json,png}` (ms):

| turn | ctx tok | EVICT(=cold) | RETAIN(=warm) | migrate*(=RETAIN+penT2) | benefit |
|---|---|---|---|---|---|
| 0 | 1,548 | 627 | 670 | 713 | −87 |
| 1 | 2,971 | 1,138 | 726 | 774 | 364 |
| 2 | 4,394 | 1,645 | 880 | 932 | 713 |
| 3 | 5,817 | 1,931 | 1,004 | 1,062 | 869 |
| 4 | 7,240 | 2,470 | 1,142 | 1,205 | 1,265 |
| 5 | 8,663 | 2,936 | 1,338 | 1,405 | 1,531 |
| 6 | 10,086 | 3,362 | 1,412 | 1,484 | 1,878 |
| 7 | 11,509 | 3,988 | 1,600 | 1,677 | **2,311** |

**해석**:
1. **turn0은 양쪽 cold로 동일**(627 vs 713, migrate가 살짝 비싼 건 pen.T2 오버레이 탓 —
   첫 turn은 보존할 KV가 없어 이득 없음이 정상). **turn1부터 격차가 열리고 context 누적에
   따라 단조 증가** → migration 이득은 대화가 길어질수록 커짐(핵심 주장 직접 입증).
2. **마지막 turn(11.5k tok): EVICT 3,988ms → migrate 1,677ms = 2.4× 단축, 절감 2,311ms.**
   turn1+ 평균 이득 1,276ms (2.0× 단축).
3. **이득이 EVICT−RETAIN(re-prefill 회피)에서 오고 pen.T2(평균 60ms)는 미미** — NVLink 전송
   비용은 re-prefill 비용 대비 무시할 수준. §2.11 "전송 거의 무료"가 E2E로 확인됨.
4. **주의(정직)**: 이번 측정은 C=1(단일 세션)이라 EVICT를 `/flush_cache`로 강제했다.
   re-prefill 비용(EVICT)이 §2.10 cold 사다리보다 완만한 건(11.5k에서 11s가 아닌 4s) —
   §2.10은 단발 15k-char anchor, E1은 누적 multi-turn이라 chunked-prefill/부분 prefix
   차이일 수 있음. 절대값보다 **추세(context↑ → 이득↑)와 2.4× 비율**이 결론.
   다음: C=8~16 부하에서 자연 LRU eviction으로 재현(flush 대신) → 실 압박 하의 이득.

**E1b 실측: 자연 LRU eviction (flush 없이) — 4개 operating point (2026-06-15~18)**
도구: `benchmark/e1b_natural_eviction.py`. C개 세션 병렬 구동, tool window에 idle →
자연 압박으로 LRU evict 유발, cached_tokens로 turn별 HIT(warm)/MISS(cold) 분류.
결과: `results/sglang_hicache/Qwen3-14B/*_e1b_natural_eviction.{json,png}`.

| run | C | pool(mem_frac) | tool delay | 자연 MISS율 | 결과 |
|---|---|---|---|---|---|
| nvlink_c8 | 8 | 기본(0.861) | 6s | **0%** | pool이 다 담음 — eviction 없음 |
| nvlink_c16 | 16 | 기본 | 6s | **28%** | eviction ✓ 단 **thrashing**(MISS 22–35s, HIT p95 22s) |
| memfrac06_c8 | 8 | 0.6 | 6s | — | **decode OOM hang** (#6857) — KV pool ~1GB |
| memfrac075_td20 | 8 | 0.75 | 20s | **27%** | eviction ✓ hang 없음, 단 **큐잉 지배** |
| clean_c4_td30 | 4 | 0.70 | 30s | **0%** | pool 여유 — eviction 없음 (그래도 p95 12.5s 큐잉) |

**핵심 결론 (정직):**
1. **자연 eviction은 재현됨**: C=16(28%)·C=8/0.75(27%)에서 flush 없이 idle 세션 KV가
   LRU로 축출 → "압박 시 idle KV가 버려진다"는 전제가 자연 발생으로 입증. aggregate로
   MISS가 HIT보다 2.1×(C=8/0.75)~3.7×(C=16) 느림.
2. **그러나 migration 이득의 깨끗한 격리는 이 하드웨어에서 어렵다**: A6000+14B는 KV pool이
   ~14GB뿐이라 (가중치가 0.6 차지), eviction을 유발하는 압박 영역이 항상
   **thrashing(C=16) / hang(0.6) / 큐잉 지배(0.75) / pool 여유(0.70)** 중 하나로 빠진다.
   eviction은 일어나되 TTFT가 re-prefill에 지배되는 "깨끗한 창"이 매우 좁음.
   → **깨끗한 migration 이득 수치는 controlled E1(C=1, flush)의 2.4×가 결론**, E1b는
   "자연 발생 + 압박 영역의 시스템 거동(thrashing/hang)"을 보이는 역할.
3. **hang(#6857) 자체가 motivation**: SGLang PD는 이 워크로드가 만든 메모리 압박을
   graceful하게 못 풀고 decode OOM hang. 본 연구의 migration(idle KV를 P-HBM으로 빼
   D-OOM 자체를 방지)이 이걸 해결 → §2.13 "압박 처리 실패"의 실증.

**방법론 교훈 (재현용)**:
- `mem_fraction_static`은 (가중치+KV) 총 비율 — 14B/49GB는 floor ~0.6 (그 아래 서버 즉사).
  KV pool ≈ (frac×49−30)GB. 자연-eviction 영역 0.68~0.72, 0.6은 hang.
- 고동시성에선 TTFT가 큐잉 지배 → cold/warm 분리가 흐려짐(turn 6–7에서 HIT가 MISS보다
  느린 역전 관측). warm_ref는 turn-matched HIT median으로(선형 fit은 절편 음수로 왜곡).
- 깨끗한 격리는 더 큰 KV pool(H100) 또는 더 작은 모델에서 가능할 것 — 향후 과제.

**E2. P radix evict 비용 측정 (= §2.12 다음단계 b 구체화) [정직성 방어]**
- 목적: P-HBM을 Tier2로 비우는 실제 비용 → "P 양보가 D 양보보다 싸다" 정량화.
- 설계: P radix를 flood로 강제 evict한 뒤 같은 세션 다음 turn TTFT vs evict 안 한 경우.
  write_through면 host에 복제본 존재 → evict 후 host hit으로 복구(≈0) 기대.
- 산출: P evict 비용 ≈ 0 확인 시, §2.12 correlated-demand 단서(C=16 P 23%)가 무력화됨
  ("P가 좀 차도 거의 공짜로 비울 수 있다").
- 구현: `benchmark/e2_concession_cost.py`. anchor 하나를 context별 3 상태 측정 —
  warm(GPU L1) / P_restore(GPU flood로 evict, write_through host 복제본 유지 → host→GPU
  복원) / cold(fresh anchor = full re-prefill). D양보=cold−warm, P양보=P_restore−warm.

**E2 1차 결과 (2026-06-18) — ❌ 측정 버그로 철회.**
`revival_e2_concession_cost.{json,png}`의 "P_restore ≈ warm, P 양보 59× 쌈, DRAM tier
부활"은 **거짓**으로 판명. 원인: `P_restore`를 flood 후 *3 요청의 median*으로 측정했는데,
첫 요청이 anchor를 GPU로 되돌리므로 2·3번째가 warm hit → `median([first-miss, warm, warm])
= warm`. host 복원이 작동하든 깨졌든 항상 ≈warm을 반환하는 측정 버그. 코드 수정:
`restore_first_miss()`가 매 반복 re-flood 후 *첫 요청(first-miss)*만 측정 (breakeven L2와 동일).

**교차 검증 — breakeven 맵이 정답 (§2.10 재측정 2026-06-18, revival 설정 확인됨):**
`revival_kv_breakeven_map.json` (page_first+ratio3.0, 서버 args로 설정 확인):

| tok | cold | L1 | L2 first-miss | L2/cold |
|---|---|---|---|---|
| 666 | 386 | 169 | 380 | 0.98 |
| 5,000 | 2,215 | 693 | 2,225 | 1.00 |
| 10,000 | 4,406 | 1,374 | 4,540 | 1.03 |

→ **L2 first-miss ≈ cold (L2/cold ≈ 1.0)**. page_first+ratio3.0에서도 **host(DRAM) 복원은
여전히 깨져 있음** — 재측정 break-even 맵에도 RAM 칸 0개(GPU/EVT 그대로). 즉:
- **§2.9 step 2(L2 revival) 미해결** — page_first+ratio3.0로도 안 살아남.
- **§2.10 "수리 후 RAM 지배 맵"은 여전히 예측(미실현)**. DRAM tier는 현 SGLang(0.5.9)
  hicache 복원 경로에서 동작 안 함이 재확인됨.
- §2.8의 "L2 ≈ cold" 결론이 옳았음 (E2 1차의 반박은 무효).

**E2 재측정 필요 (코드 수정 후)**: first-miss 기준으로 P_restore를 다시 재면 ≈cold로 나올
것으로 예상 → recovery-cost 비대칭(host 복원이 싸다)은 **성립 안 함**. 따라서 §2.12
correlated-demand 단서는 recovery-cost가 아니라 **opportunity-cost 비대칭**으로 닫아야 함:
P-HBM은 83–90% idle(§2.12)이라 P 양보의 기회비용 ≈ 0; P 양보는 active decode 방해 안 함
(future prefix-miss일 뿐), D 양보는 active 세션 재계산 + hang 유발(§2.14 memfrac06) —
이 비대칭은 hicache 복원 작동 여부와 무관하게 robust.

**측정 교훈**: "evict 후 복원" 비용은 반드시 **first-miss**로 측정 (median은 첫 요청이
캐시를 데워서 복원 비용을 씻어냄). breakeven 맵 방식이 정답.

**E3. Migration vs Retraction 직접 대결 [reorder 대신 migration의 핵심]**
- 목적: 차별점 [D]("preemption 없이 migration으로 진행")의 직접 증거.
- 설계: D를 포화시켜 retraction 유발(C=24~32, mem_fraction 낮춰 강제). 측정:
  - baseline: SGLang 기본 → `#retracted-req`, 재개 지연, 영향받은 요청 TTFT/goodput
  - 제안(emul): 포화 직전 idle 세션 KV를 P-HBM으로 선제 migration → retraction 회피율
- 산출: "migration이 retraction을 N% 제거 → P99 TTFT/goodput 개선". reorder/preempt 대비
  migration이 1차 해결책임을 시연.

**E4. Tool-duration × pressure 2D 운영점 [정책의 필요성]**
- 목적: §2.10 break-even 맵을 *실제 부하*와 결합 → "언제 migrate가 evict보다 이득인가".
- 설계: (tool_delay ∈ {0.5,2,10,30s}) × (concurrency ∈ {4,8,16}) 격자에서 E1 반복.
- 산출: migrate 우세 / evict 우세 영역 맵. 짧은 tool+저부하=keep, 긴 tool+고부하=migrate
  등 정책 결정 경계. duration 예측이 필요한 이유를 운영점으로 제시.

**E5. 전송 대역폭 파라미터화 (TCP/PCIe/NVLink) [일반성]**
- §2.8(5)/2.11의 3-포인트(1.2/~20/29 GB/s)에 E1을 얹어 "migration 이득이 interconnect에
  따라 어떻게 변하나" → NVLink 없는 배포에서도 이득 영역이 존재함을 보이거나, 한계 명시.

**우선순위**: E1(핵심 증거) → E2(정직성) → E3(차별점 D) → E4(정책) → E5(일반성).
E1·E2는 §2.10/2.11/2.12 기존 데이터+emulation으로 시스템 구현 전에 가능 → go/no-go 게이트.

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
| ~~SSD tier 정당화~~ | **해소 (§2.9)**: SSD를 평가 범위에서 제외하고 3-tier로 축소. Formulation만 N-tier 일반 유지 | — |
| BFCL mock tool | tool이 즉시 반환 → synthetic delay 주입만으로는 약함 | 실제 trace 기반 분포(SWE-bench agent 로그, 실 API latency) 주입. Continuum과 동일 워크로드로 비교 가능성 확보 |
| Duration 예측 novelty 소진 | Continuum이 이미 empirical CDF 예측 입증 | 예측기는 차용, 기여는 multi-tier placement + PD 고유 문제로 |
| 단일 8B 모델 | 모델 스케일 axis 약함 (70B는 하드웨어상 불가) | long-context 변형(turn 누적 + 긴 system prompt)으로 KV pressure axis 보강 |
| 구현 공수 | D 노드 decode 중 KV를 하위 tier로 내보내는 경로가 SGLang에 없음 (HiCache는 P 노드 radix cache 계층화) | **완화 (§2.9)**: sglang 0.5.9의 `--disaggregation-decode-enable-offload-kvcache`가 D→storage 경로 제공 (안정성 검증 필요, #11016). Oracle(§5.E)까지는 캐시 flush/유지 조작으로 시뮬레이션 |

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
  placement ∈ {D_HBM, P_HBM, CPU_DRAM, EVICT}   # §2.9 결정: 평가는 3-tier + EVICT
                                                 # (formulation은 N-tier 일반, SSD는 discussion)
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

**측정 경과**:
- 1차 (§2.7): 중간 tier(RAM/SSD)가 optimal인 칸 0개. L2 ≈ cold, L3 > cold.
- 2차 A-1/A-2 완료 (§2.8): PD-특이성 기각 (단일 서버도 동일 패턴), Mooncake P→D
  전송 세금 1.2 GB/s 실측, wait_complete는 PD에서 24–126 s 스톨, 카운터 진단은
  server_info 미노출로 실패 (로그 grep으로 대체).
- 설계 결정 (§2.9): 3-tier 축소. **후속 단계는 §2.9 하단 1–3 참조**
  (D-offload 플래그 검증 → L2 살리기 → 3-tier 맵 재생성).

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
| `scripts/sglang/start_1P_1D_breakeven.sh` | §5.A 서버 시작 (1P+1D + hicache + router 8001, `D_OFFLOAD_KVCACHE` 노브) |
| `scripts/sglang/start_single_hicache.sh` | §2.8 대조군 / §2.9 L2 살리기 (ratio/layout/io/prefetch 노브) |
| `benchmark/sglang_kv_breakeven_map.py` | §5.A break-even 맵 측정 (`SKIP_L3`, `T1_REF_JSON` 3-tier 모드) |
| `benchmark/pd_hbm_occupancy.py` | §2.12 P/D HBM 점유 동시 폴링 + P→D 전송 감지 (`CONCURRENCY` sweep) |
| `benchmark/e1_emulation.py` | §2.14 E1 migration 이득 emulation (EVICT/RETAIN/migrate, `REPLOT_JSON`) |
| `results/sglang_hicache/Qwen3-14B/nvlink_e1_migration_benefit.{json,png}` | §2.14 E1 실측 (2.4× 단축) |
| `benchmark/e1b_natural_eviction.py` | §2.14 E1b 자연 LRU eviction (cached_tokens HIT/MISS 분류, `CONCURRENCY`/`MEM_FRACTION`) |
| `results/sglang_hicache/Qwen3-14B/*_e1b_natural_eviction.{json,png}` | §2.14 E1b 4 operating point (nvlink_c8/c16, memfrac075_td20, clean_c4_td30) |
| `benchmark/e2_concession_cost.py` | §2.14 E2 P vs D 양보 비용 비대칭 (warm/P_restore/cold, `RUN_TAG`) |
| `results/sglang_hicache/Qwen3-14B/revival_e2_concession_cost.{json,png}` | §2.14 E2 실측 (page_first+ratio3.0: P양보 59× 쌈, DRAM tier 부활) |
| `results/sglang_hicache/Qwen3-14B/nvlink_c1_v2_pd_hbm_occupancy.{csv,png}` | §2.12 C=1 점유 시계열 |
| `results/sglang_hicache/Qwen3-14B/nvlink_c{4,8,16}_pd_hbm_occupancy.{csv,png}` | §2.12 concurrency sweep (correlated-demand) |
| `results/sglang_hicache/Qwen3-14B/1p1d_kv_breakeven_map.json` | §2.8 2차 PD 측정 (wait_complete) |
| `results/sglang_hicache/Qwen3-14B/1server_kv_breakeven_map.json` | §2.8 단일 서버 대조군 = T1 proxy 소스 |

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
- [SGLang HiCache Best Practices](https://docs.sglang.ai/advanced_features/hicache_best_practices.html) — mem_layout/io_backend/prefetch 정책 가이드
- [SGLang HiCache 블로그](https://www.lmsys.org/blog/2025-09-10-sglang-hicache/), [Mooncake × HiCache 설계](https://kvcache-ai.github.io/Mooncake/design/hicache-design.html)
- [sglang #11016](https://github.com/sgl-project/sglang/issues/11016) — `disaggregation-decode-enable-offload-kvcache` CUDA error 보고 (안정성 주의)
