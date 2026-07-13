# Phase 2 — Predictive Tool-Gap KV Offload/Prefetch (host tier)

> Phase 1(idle-KV-parking, spare-GPU)에서 얻은 결론으로 아이디어를 재정의한다.
> 새 티어(spare GPU)가 아니라 **tool-call gap의 예측 가능성**을 무기로,
> 기존 host-DRAM 티어(hicache) 위에서 **offload/prefetch를 능동·정시(timed)로 스케줄**한다.

## 1. 왜 재정의하나 (Phase 1 교훈 3줄)

1. **decode-bound면 KV 재배치는 throughput을 못 올린다** (§7-6). reuse는 prefill 일감만 줄이는데,
   고동시성에선 병목이 decode 노드다. → **win-condition = prefill/메모리 병목**.
2. **이 토폴로지의 spare GPU2는 P에서 PCIe** = host DRAM과 대역폭 동일. spare GPU는 host보다 나을 게
   없다 (NVLink-close spare가 아닌 한). → **티어는 host DRAM이 정답**, spare GPU는 버린다.
3. hicache의 offload/prefetch는 **reactive·best-effort** — 3s gap 멀티턴에서 prefetch가 제때 못 옴
   (decode-offload의 L3_prefetched 3,064뿐, §7-2). → **틈새 = 정시 prefetch**.

## 2. 한 줄 논지 (thesis)

> **Tool-call gap은 예측 가능한 유휴 구간이다. 이걸 이용해 (a) 일시정지 대화의 KV를 scarce GPU에서
> host로 능동 축출하고, (b) 다음 턴 도착 시각에 맞춰 host→GPU로 정시 prefetch하면, hicache의 reactive
> best-effort가 놓치는 전송 지연을 숨기고 재-prefill을 없앤다 — prefill/메모리가 병목인 구간에서 TTFT win.**

## 3. 메커니즘

대화(session) 하나의 한 턴 사이클:

```
turn N 종료(=tool call 발행)  ──gap(T초)──►  turn N+1 도착
        │                                          │
        ├─ (a) offload-on-pause                     ├─ prefix가 GPU에 이미 상주 → prefill이 즉시 hit
        │     session S의 prefix KV를 GPU→host      │     (전송 지연 0, 재-prefill 0)
        │     로 능동 축출, GPU 슬롯 확보            │
        │                                          │
        └─ (b) timed prefetch  ── gap 끝 직전 ──────┘
              host→GPU 복원을 T에 맞춰 시작,
              turn N+1 도착 전에 완료
```

- **예측 신호(prototype)**: client가 요청에 힌트를 실어 보낸다.
  - `session_id`: 같은 대화를 잇는 키 (prefix 추적용).
  - `next_turn_eta_ms`: 이번 턴이 tool-call 턴이면 "약 T ms 후 재개" (벤치는 TOOL_DELAY를 알므로 정답 제공).
  - eta=0/미지정이면 tool 턴 아님 → 스케줄 안 함.
- **server 동작**:
  - turn 종료 시 `next_turn_eta_ms`가 있으면 → 해당 prefix를 host로 offload 마킹 + `(session_id → prefix hash, host handle, resume_deadline = now+T)` 등록.
  - 백그라운드 스케줄러가 `resume_deadline - prefetch_lead`(전송시간 추정) 시점에 host→GPU prefetch 시작.
  - turn N+1 도착 → prefix가 이미 GPU radix에 있음 → 자연 hit.
- **server-side 예측(후속, 선택)**: 힌트 없이 session별 gap 이력으로 T 추정. 프로토타입 이후.

## 4. hicache와의 차별점 (무엇을 이겨야 하나)

| 축 | hicache (incumbent) | Phase 2 (predictive) |
|---|---|---|
| offload 시점 | LRU, 반응형 (압박 시) | **pause 신호에 능동, 타겟** |
| prefetch 시점 | best-effort, 요청 도착 후/근처 | **gap 중 정시 — 도착 전 완료** |
| prefetch 적중 | 자주 놓침 (L3 3,064) | **gap 예측으로 100% 상주 목표** |
| 티어 | host DRAM | host DRAM (동일 — 티어가 아니라 스케줄이 기여) |

**핵심 측정 = prefetch on-time rate**: 요청 도착 시 prefix가 이미 GPU에 있었나. hicache는 낮고 우리는 높아야 함.

## 5. Win-condition & 벤치마크 설계

reuse→TTFT 전환은 **prefill/메모리가 병목일 때만** 일어난다(§7-6). 따라서 시험대를 그렇게 만든다:

- **긴 공유 prefix** (session당 큰 문서/시스템프롬프트) → prefix가 크고 재-prefill이 비쌈.
- **짧은 출력** (`max_tokens` ~16–32) → decode 저렴 → **prefill이 지배**.
- **높은 동시성 + pool 압박** (`--max-total-tokens` 작게) → prefix 축출 발생 → 재-prefill or offload 필요.
- **tool-gap** (turn 사이 delay) → offload/prefetch 기회 + 예측 신호.
- (보강) **1P2D** 배치로 prefill을 먼저 포화시키는 옵션.

측정 지표:
- **avg TTFT** (주지표; 재-prefill 회피 + 전송 은닉의 합).
- **prefetch on-time rate** (도착 시 GPU 상주 비율) — 차별점 직접 증거.
- **reuse_ratio / re-prefill tokens avoided**.
- GPU KV 여유(동시성 여력), host RAM.

arm:
- `radix` — GPU-only. 압박 시 축출 → 매턴 재-prefill (바닥).
- `hicache` — reactive host offload/prefetch (incumbent).
- `predictive` — (a)+(b) 정시 스케줄 (제안).

## 6. 슬라이스 계획

- **S0 (testbed)**: prefill-bound + gap 벤치(`sglang_longctx_multi_turn_concurrent.py`) + `radix`/`hicache`
  2-arm 측정. **가설 검증**: 이 구간에서 hicache prefetch가 실제로 늦나(on-time rate 낮나)? prefill이
  병목 맞나(TTFT가 재-prefill에 지배되나)? ← 여기서 premise부터 확인.
- **S1 (signal)**: 요청에 `session_id` + `next_turn_eta_ms` 힌트 전달 경로(벤치→router→server). 서버는
  로깅만(동작 변화 없음) — 신호가 스케줄러까지 도달하는지 확인.
- **S2 (timed prefetch)**: 힌트 기반으로 gap 중 host→GPU prefetch 트리거. on-time rate ↑, TTFT ↓ 측정.
- **S3 (proactive offload)**: pause 신호에 능동 축출로 동시성 여력 확보 → 같은 GPU로 더 많은 세션.

## 7. 열린 질문 (구현 중 결정)

- prefetch_lead(전송시간) 추정: prefix 길이 × 측정 대역폭. 정적 vs 적응.
- 정시 prefetch를 어디서 돌리나: 별도 스케줄러 스레드 vs SGLang hicache prefetch 큐 재사용.
- 신호 전달 통로: OpenAI API `extra_body` / 커스텀 헤더 / router passthrough.
- 압박이 심하면 prefetch가 도착 전 다시 축출될 수 있음(§7-6-b의 파킹실패와 동형) → pin/우선순위 필요.

## 8. S0 결과 (prefill-bound testbed, pool 40k / C16 / delay3 / n128)

premise 1·2는 강하게 성립, 3은 **예상과 반대**로 나와 피벗의 명분을 좁혔다.

| arm | TTFT | TPOT | reuse | tput | wall |
|---|---|---|---|---|---|
| radix | 4.75s | 0.008s | 0.019 | 52.5 | 233.8s |
| hicache | 2.08s | 0.022s | **0.748** | 76.9 | 159.8s |

- **(1) prefill-bound 확실**: 출력 24tok, TPOT~0.01s → TTFT가 응답시간 ~96%. radix reuse 0.019 = pool
  40k에 C16 working set(~58k) 안 들어가 매턴 재-prefill 스래싱.
- **(2) hicache ≫ radix**: TTFT −56%, tput +46%. 압박 하 offload가 재-prefill을 회피.
- **(3) 그런데 hicache reuse 0.748 = 4턴 구조의 이론 최적(turn0은 문서 첫등장이라 원리적 cold, 3/4=0.75).**
  → **hicache는 reuse를 놓치지 않는다.** "reactive prefetch가 늦어 miss"라는 원래 명분은 이 워크로드에서
  성립 안 함.

**턴별 TTFT (turns 1-3)**: radix ~4.5s(스래싱) vs hicache ~1.8s, hicache floor(min) ~0.35s.
hicache 1.8s 분해:
1. **host→GPU KV 로드** (predictive가 gap에 숨길 유일한 부분): KV 128KB/token × 3.5k tok = 448MB,
   PCIe 25GB/s → **≈18ms. 무시가능** (memory-bound).
2. **재-prefill 절약분 ~2.7s/turn**: 이미 hicache가 캐싱으로 회수. predictive 몫 아님.
3. **queue-wait**: 나머지(floor 0.35↔median 1.8 변동). C16 압박+turn0 cold가 큐 막음. **KV 배치로 못 줄임.**

**S0 결론**: hicache는 recompute-회피(큰 이득)를 이미 먹고, 남은 TTFT는 queue-wait(배치로 불가)+로드
18ms(무시). **predictive의 addressable slice(load-hiding)는 3.5k prefix에선 미미.** 단 load-hiding은
prefix 길이에 비례(128k tok ≈ 640ms) → **긴 컨텍스트에서만 niche 가능**. Phase 1 spare-GPU와 동형 패턴:
hicache가 강한 incumbent라 KV-movement 추가 이득이 memory-bound라 항상 작다.

**게이트(S0.5)**: `PREFIX_WORDS` 16k~32k로 키워 hicache TTFT에 로드 성분(수백 ms)이 드러나는지 확인.
드러나면 long-context niche로 S1~S2 진행, 아니면 피벗 종료.

---
_연계: Phase 1 결론 `results/phase1_report/report.md` §7-4~7-6. 이 문서는 Phase 2 설계/계획._
