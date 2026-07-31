# Evaluation — KV-aware Unified Memory

측정 대상 주장은 §0의 목표 문장 하나다:

> 전체 KV 데이터 크기는 유지하면서, 유휴 GPU HBM을 우선 활용하여 **CPU DRAM commitment를 줄인다.**

따라서 **주 지표는 CPU DRAM commitment**이고, 나머지 지표는 모두 "그 대가로 무엇을 잃지
않았는가"를 보이는 보조 지표다. 총 메모리 사용량은 지표가 아니다.

## 공통 setup

server17, RTX A6000 × 4 (49 GB), Llama-3.1-8B-Instruct, SGLang 2P2D (P: GPU0/1, D: GPU2/3),
Mooncake transfer backend, sglang_router. Workload = BFCL v3 multi-turn (200 items, agentic
multi-turn + tool calls). KV geometry: 32 layer × 8 kv-head × 128 dim × K/V × 2 B =
**128 KiB/token**.

Arms (서버 config, 라벨은 결과 파일명과 동일):

| 라벨 | 구성 |
|---|---|
| `wt` | HiCache write_through (host L2 + file L3) |
| `wts` | HiCache write_through_selective |
| `wb` | HiCache write_back |
| `l2` | HiCache host L2 only (file backend 없음) |
| `park` | **제안 방식** — HiCache 미사용, 유휴 GPU HBM 우선 배치 |

---

## A. 확보된 결과 (재실행 불필요)

### A1 ★ CPU DRAM commitment — 주 결과

`benchmark/sys_mem_breakdown.py`, `results/mem/bd_*.csv` (215 samples/arm, 1 s 간격).

| arm | AnonPages mean / peak (GB) | MemAvailable mean / min (GB) | page cache (GB) | L3 file peak (GB) |
|---|---|---|---|---|
| `wt` | 76.1 / 76.2 | 44.3 / 43.1 | 37.7 | 108.7 |
| `wts` | 76.3 / 76.6 | 43.7 / 42.6 | 33.6 | 108.6 |
| `wb` | 76.0 / 76.2 | 44.5 / 44.0 | 17.0 | 0.0 |
| `l2` | 75.9 / 76.1 | 44.6 / 44.2 | 16.5 | 0.0 |
| **`park`** | **14.9 / 15.1** | **105.5 / 104.6** | 16.7 | 0.0 |

**AnonPages −61.2 GB, MemAvailable +61.2 GB.**

두 가지가 이 결과를 강하게 만든다:

1. **write policy·storage backend와 무관하게 76 GB로 동일하다** (wt/wts/wb/l2 편차 0.4 GB).
   HiCache의 host commitment는 *정책이 아니라 정적 예약*이다 — `--hicache-ratio`로 device보다
   큰 host pool을 부팅 시 잡아버리므로, 어떤 write policy를 골라도 줄어들지 않는다. 즉
   **기존 시스템 안에서 튜닝으로 해결할 수 없다**는 것이 직접 측정으로 보인다.
2. `park`의 page cache는 16.7 GB로 `l2`(16.5)와 같다 — 절약이 page cache를 희생해서 나온 게
   아니다. 줄어든 것은 **회수 불가능한 anonymous 예약**뿐이다.

인용 문장: *HiCache commits 61 GB of host DRAM regardless of write policy or storage backend;
GPU-first placement returns all of it, raising MemAvailable from 44.3 GB to 105.5 GB.*

### A2 성능 동률 (TTFT / throughput)

`results/perturn/` (REPS=3). pooled median TTFT **hicache 3.51 s vs park 3.52 s** (0.01 s 차).
per-rep: hicache [3.61, 3.77, 3.88] vs park [3.67, 3.90, 3.70]. throughput ~86 vs ~85 tok/s.
→ Fig `fig_perturn`, `fig_ttft_byturn`.

cross-workload 확인: `results/fig_combined.{pdf,png}` — BFCL + ShareGPT 두 워크로드에서
동률 + host RAM win 동시 성립.

### A3 unified 프레이밍의 근거 — HBM 아래에 대역폭 계층이 없다

`benchmark/vmm_probe.py`, `benchmark/pinned_host_probe.py` (동일 주소공간·동일 접근 경로):

| residency | 대역폭 | local 대비 |
|---|---|---|
| local HBM | 331.9 GB/s | 100% |
| NVLink peer HBM | 27.2–52.8 GB/s | 8–16% |
| CPU DRAM (pinned) | H2D 26.3 / D2H 26.4 GB/s | 8% |
| CPU DRAM (pageable) | H2D 20.6 / D2H 14.1 | 4–6% |
| **non-NVLink peer HBM** | **3.3 GB/s** | **1%** |

두 개의 인용 가능한 사실:
- NVLink peer HBM ≈ CPU DRAM (27.2 vs 26.3 = **3.4% 차**) → 둘을 다른 tier로 나눌 근거가 없다.
- **non-NVLink peer(3.3) < CPU DRAM(26.3)** → "GPU가 host보다 가깝다"는 고정 tier 순서가
  실제로 **틀린다**. 배치는 측정된 대역폭 기반 정책이어야 한다.

### A4 유휴 GPU HBM이 실재한다

`results/kv_ts/fig1_occupancy_overlay_col.{pdf,png}`. decode KV pool 평균 **74.6%** vs prefill
**44.6%** (30 pt gap), 샘플의 98%에서 decode > prefill. **prefill pool의 55%가 상시 미사용.**

### A5 recompute는 대안이 아니다

`results/intro/fig_ttft_ctx_sweep.json`: re-prefill 161 / 293 / 592 / 1203 / 2706 / 6681 ms
@ 1k / 2k / 4k / 8k / 16k / 32k. 어느 위치에서 가져와도 recompute보다 **30–43× 빠르다**.

### A6 pinned host 할당 특성 (설계 정당화)

`benchmark/pinned_host_probe.py` + `benchmark/test_host_overflow.py` (27/27):
cold pin **488 ms/GiB**, warm **0.0 ms**, RSS가 flush 시 실제로 반환됨(+2048 → +256 MB).
→ host tier는 *캐시된 만큼만 commit*하고 정적 예약을 하지 않는다.

---

## B. 미확보 — 이번 프레이밍에 새로 필요한 것 두 개

기존 결과는 "host를 안 쓴다"까지만 보인다. 새 프레이밍의 두 문장이 아직 증거 없다.

### B1 ★ agent stack 용량 회복 (K_max) — **먼저 할 것**

`results/agent/`에 `ramp_write_through.json` 하나만 있다 (**K_max = 73** of 125.6 GB total,
tool WS 1 GB/worker). **`park` arm이 없어서 회복량을 말할 수 없다.**

이게 Motivation의 정확한 역명제다: motivation은 "host offloading이 agent stack 용량을 깎는다",
결과는 "우리가 그걸 돌려준다". MemAvailable 44.3 → 105.5 GB이므로 K_max는 73 → ~95 근처가
나와야 하고, **안 나오면 A1의 절약이 agent stack에 실제로는 쓸 수 없는 절약**이라는 뜻이라
반드시 확인해야 한다.

```bash
# 서버: park arm (host overflow는 끈다 -- B1은 host tier와 무관하고, 안정성 리스크를 배제)
./scripts/sglang/stop.sh
PARK_NO_HICACHE=1 IDLE_KV_PARKING=1 SGLANG_KV_PARK_HOST_OVERFLOW=0 \
  ./scripts/sglang/start_2P_2D.sh
LABEL=park TOOL_MEM_MB=1024 MAX_WORKERS=110 ./scripts/run_agent_pressure.sh

# 비교군 보강(선택, wb는 file backend 없이 76GB를 쓰는 가장 공정한 baseline)
./scripts/sglang/stop.sh
HICACHE_WRITE_POLICY=write_back ./scripts/sglang/start_2P_2D.sh
LABEL=write_back ./scripts/run_agent_pressure.sh

python benchmark/plot_agent_pressure.py --ramp \
  write_through=results/agent/ramp_write_through.json \
  write_back=results/agent/ramp_write_back.json \
  park=results/agent/ramp_park.json \
  --out results/agent/fig_agent_host_capacity.png
```

주의: `FLOOR_GB=8` 가드는 유지한다(공유 서버, 다른 사용자 작업 중). `MAX_WORKERS=110`이면
park arm에서 ceiling에 먼저 닿을 수 있으므로 K_max가 110이면 `MAX_WORKERS=120`으로 재실행.

### B2 GPU-first placement 시계열 — **B1 다음**

`results/agent/parked_park.csv` + `fig_park_location`은 **사용할 수 없다.** 그 run에서 파킹된
세션은 5개(GPU slab 4 + host block 1)뿐이고 `host_evicted=0`, `host_flushes=0`, t≈99 s 이후
tail이 평평한데 그 구간은 **prefill이 이미 KV transfer 실패로 서비스를 멈춘 뒤**다. 정책이
작동한 그림이 아니다.

필요 조건 — 이게 충족돼야 그림에 의미가 있다:
- `TOOL_DELAY_SEC > 0` (예: 3). 이전 run은 0이어서 **파킹의 전제인 tool-call idle window가
  아예 없었다.**
- `CONCURRENCY ≥ 16`
- park pool을 세션 수보다 작게 (~8–16 slab) → slab 부족 → reuse-value eviction과 host
  overflow가 실제로 발동
- `SGLANG_KV_PARK_HOST_OVERFLOW=1` (host share가 0이면 "overflow tail이 작다"를 못 보임)
- 벤치마크 종료 시 sampler를 반드시 kill (stale publisher가 GPU share를 부풀림 — sampler는
  `--stale-s`로 방어하지만 사후 tail은 그대로 남는다)

**선행 조건**: c4 run이 죽은 원인이 `SGLANG_KV_PARK_HOST_OVERFLOW=1`인지 확정해야 한다.
`_park_to_host`의 cold pin이 488 ms/GiB 동기 블로킹이므로 스케줄러 스레드를 수백 ms 멈춰
Mooncake session이 dead로 보일 수 있고, **첫 host 파킹 시점(t≈99 s)과 장애 시작(t≈80 s)이
겹친다.** B1은 `HOST_OVERFLOW=0`이라 이 문제와 무관하므로 먼저 돌린다.

---

## C. 선택 — 리뷰어가 물을 것

| 질문 | 실험 | 상태 |
|---|---|---|
| KV 총량이 정말 유지되나? | 캐시된 KV 바이트 + prefix hit rate를 arm별 동률로 제시 | 카운터 있음, 집계 미작성 |
| reuse-value eviction이 timestamp LRU보다 나은가? | `SGLANG_KV_PARK_REUSE_AWARE=0/1` A/B, hit rate + TTFT | 코드·테스트 완료(23/23), run 없음 |
| host 배경부하가 있으면? | CPU contention arm — hicache 저하 / park 무영향 | 미실행 |
| non-NVLink peer로 파킹하면? | `SGLANG_KV_PARK_BW_AWARE=0/1` A/B (3.3 GB/s 회피 효과) | 코드 완료, run 없음 |

C는 CAL 4페이지에 다 안 들어간다. **B1 → B2 순으로만 확보하면 논문은 성립한다.**

---

## D. 그림 배치 (CAL 4p)

| Fig | 파일 | 역할 |
|---|---|---|
| 1 | `results/kv_ts/fig1_occupancy_overlay_col.pdf` | A4 — 유휴 HBM이 실재 (1 column) |
| 2 | `results/perturn/fig_design.pdf` | 설계 |
| 3 | `results/mem/fig_mem_breakdown.pdf` | **A1 — 주 결과** |
| 4 | `results/agent/fig_agent_host_capacity.pdf` | **B1 — 회복된 용량** |
| (표) | A3 대역폭 표 | unified 프레이밍 근거 |

A2(동률)는 그림 대신 본문 한 문장 + 표 한 줄로 압축한다 (3.51 vs 3.52 s).
`fig_combined`, `fig_perturn`은 지면이 남으면 넣는다.

---

## E. ShareGPT 확인 (2026-07-31) — BFCL의 TTFT 결과는 placement가 아니었다

### E1 BFCL exp1의 +562 ms는 park 고유 비용이 아니다

`results/exp1/p60000_c16_d3` (BFCL, C=16, 688–698 turn, **infra 실패 0**):

```
 turn  ctx tok      radix          hicache            park
    0      305      1.88s      2.09s (+0.21)    2.44s (+0.56)
  >=3      544      1.63s      1.85s (+0.21)    2.05s (+0.42)
```

turn 0에는 재사용할 prefix도 fetch도 없으므로 **+0.56 s는 메커니즘이 아니다.**
그리고 median context 408 토큰 → full re-prefill **66 ms**가 재사용의 이론적 상한인데,
오버헤드가 그 8.5배다. **BFCL은 이 컨텍스트 길이에서 재사용의 TTFT 이득을 보여줄 수 없다.**

기존 ShareGPT 런(C=1)과 대조하면 원인이 드러난다:

| | context | TTFT |
|---|---|---|
| BFCL turn 0 (C=16) | 305 tok | **1.88 s** |
| ShareGPT turn 1 (C=1) | 582 tok | **0.148 s** |

**컨텍스트가 2배인데 TTFT가 1/12이다.** BFCL의 TTFT는 연산이 아니라 **C=16의 큐잉이 지배**하며,
큐잉은 요청당 상수 오버헤드 차이를 증폭시킨다. ShareGPT에서 park의 turn-0 오버헤드는
**+0.04 s**로, BFCL의 +0.56 s는 재현되지 않는다.

### E2 ShareGPT(C=1)에서는 park가 hicache를 이기고, 격차가 컨텍스트와 함께 커진다

`results/perturn_sharegpt_*` (hicache·park 각 3 rep, recompute 1 rep, **C=1**):

| turn | ctx tok | prize | recompute | hicache | park |
|---|---|---|---|---|---|
| 0 | 60 | 10 ms | 0.061 (−0.04) | 0.102 | 0.141 (+0.04) |
| 1 | 582 | 94 ms | 0.148 (−0.05) | 0.201 | **0.150 (−0.05)** |
| 2 | 1102 | 175 ms | 0.246 (−0.04) | 0.287 | **0.203 (−0.08)** |
| 3 | 1600 | 240 ms | 0.346 (−0.01) | 0.357 | **0.258 (−0.10)** |
| ≥4 | 2163 | 317 ms | 0.461 (+0.03) | 0.434 | **0.341 (−0.09)** |

median TTFT: park **0.221 s** vs hicache 0.264 s (**−16%**). turn 1 이후 모든 turn에서 park가
빠르고, **격차가 컨텍스트와 함께 커진다** — 메커니즘이 예측하는 그대로다.

### E3 정직하게 같이 적어야 할 두 가지

1. **park의 tail이 제일 나쁘다**: p95 park **2.112 s** vs hicache 1.597 vs recompute 0.521.
   C=1이므로 큐잉이 아니라 **fetch 경로가 간헐적으로 블로킹**하는 것이다.
2. **recompute(캐시 없음)가 median에서 park와 대등**하다 (0.213 vs 0.221) **그리고 tail이 가장 좋다.**
   2k 토큰에서는 그냥 다시 계산하는 게 충분히 싸다. 이 컨텍스트 길이는 아직 결정적이지 않다.

### E4 그래서 다음 런의 조건

기존 ShareGPT 기본값(median 840 tok, prize 135 ms)으로는 부족하다. 컨텍스트를 키우는 노브만
올린다 — `MAX_TOKENS 512→1024`, `MAX_TURNS 6→10`, `MIN_TURNS 3→6`,
`MAX_PROMPT_CHARS 6000→16000`. turn 5에서 ~6k(prize ~900 ms), turn 9에서 ~11k(~1.6 s)가 목표다.

그리고 **C=1이 아니라 C=8**로 돌린다. C=1은 경합이 없어 실제 서빙 조건이 아니고, arm당 3시간이
걸린다. C=16은 큐잉이 전부를 덮는다. C=8이면 8×11k=88k가 60k 풀을 압박하므로 축출이 실제로
일어나면서 큐잉이 지배하지는 않는다.

```bash
./scripts/sglang/run_exp1_sharegpt.sh
```

실행 후 **먼저** `ttft_by_turn.txt`에서 median context가 4k를 넘었는지 확인한다. 안 넘었으면
TTFT 비교는 placement에 대해 아무것도 말하지 않으므로 `MAX_TOKENS`를 더 올려 재실행한다.
