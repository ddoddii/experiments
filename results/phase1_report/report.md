# Idle KV Parking — Phase 1 결과 보고

**주제**: PD(prefill–decode) disaggregation의 multi-turn agent 워크로드에서, tool-call 유휴시간에
decode 노드의 세션 KV를 **유휴 GPU에 park 했다가 다음 turn에 fetch**해, 메모리 압박으로 축출된
prefix를 재계산 대신 되찾을 수 있는가?

> **결론(TL;DR)**: **된다.** fetch-on-hit + evict-to-room + async까지 구현하니, 강압박(pool 40k)에서
> park이 축출된 prefix를 되찾아 **reuse 0.38→0.74**로 복구하고 **TTFT를 radix 대비 −26%** 줄였다.
> 이는 성숙한 host-DRAM hicache(−24%)와 **대등**하다(park ≈ hicache, 차이 ~1–3%). 즉 **유휴 GPU를
> host DRAM 대신 L2 복구 tier로 써도 동등하게 작동**한다. 단, 복구 tier는 **압박이 있을 때만** 이득이고
> 압박이 없으면(pool≥60k) park·hicache 모두 순수 오버헤드라 radix보다 느리다(crossover).

---

## 1. 배경과 가설

- **문제**: 압박 하에서 P 노드의 prefix(radix) 캐시가 대화 중간에 축출되면, 다음 turn이 그 prefix를
  **재계산(re-prefill)**해 TTFT가 오른다.
- **아이디어**: D→P의 유휴 GPU(GPU2)로 세션 KV를 park 했다가, 다음 turn prefill 직전에 fetch해
  prefix-hit으로 재사용 → 재계산 회피.
- **가설**: "PCIe로 fetch해도 재계산보다 싸다" — 그리고 "유휴 GPU L2가 host-DRAM L2(hicache)와
  같은 양상이어야 한다."

> **radix는 재계산 baseline이 아니다.** radix도 RadixAttention prefix cache라 GPU 잔존 prefix는
> hit한다(reuse 0.38~0.74). 재계산하는 건 hit 못한 토큰뿐: ①매 turn 새 토큰(~26% floor, 불가피) +
> ②압박으로 축출된 prefix. park/hicache가 되찾는 대상은 **②뿐**이고, ②는 강압박에서만 존재한다.

## 2. 설정과 구현

- 하드웨어: server17, A6000×4 (NVLink 쌍 (0-1),(2-3)), Llama-3.1-8B, SGLang 0.5.9 + Mooncake.
- 구성: **1P1D** (P=GPU0, D=GPU1) + **전용 park 풀 GPU2**(200k tok, PCIe from P). BFCL v3 200 items,
  동시성 8, tool-call 유휴 3s. P GPU 풀 크기(=압박)를 40k→120k 스윕.
- 구현(SGLang, 슬라이스 2a~4b): D→P **CUDA IPC + P2P** 채널, turn 종료 시 **프롬프트 prefix park**,
  다음 요청 prefill 직전 `maybe_fetch()`가 최장 parked prefix를 GPU2→GPU0 복사 + radix insert.
  - **동작에 필수였던 두 가지** (아래 §5 참조): **evict-to-room**(자리 없으면 LRU로 콜드 축출 후 복원,
    hicache와 동일) + **async fetch**(복사를 default stream에 올리고 host-sync 제거).

## 3. 결과

### Figure 1 — 압박 스윕: park ≈ hicache, 복구 tier는 압박에서만 이득

![sweep](./catch22_pressure.png)

- **상단(reuse)**: **park(주황)이 hicache(녹색)와 완전히 겹친다** — 둘 다 압박(40k)에서 축출된
  prefix를 되찾아 reuse 0.74 유지. radix(파랑)만 40k에서 0.38로 떨어짐. pool≥60k에선 축출이 없어
  셋 다 0.74로 수렴.
- **하단(TTFT) — crossover**: 40k에선 park·hicache(≈1.35s)가 radix(1.83s)보다 **훨씬 낮다(이득)**.
  pool≥60k에선 반대로 radix가 가장 낮고 park·hicache는 위(오버헤드). **복구 tier는 압박이 있을 때만
  이득**이다.

### Figure 2 — 강압박(40k) 결정 케이스

![win](./pressure_win_40k.png)

radix 1.83s(reuse 0.38) → hicache 1.38s(**−24%**, reuse 0.74) → **park 1.35s(−26%, reuse 0.74)**.
park이 host-DRAM hicache와 대등하게(오히려 근소하게 낮게) 축출된 prefix를 되찾는다.

### 전체 수치 (BFCL, C=8, tool-delay 3s, 200 items)

| pool | radix TTFT / reuse | hicache TTFT / reuse | park TTFT / reuse | park vs radix | park vs hicache |
|---|---|---|---|---|---|
| 40,000 (강압박) | 1.83s / 0.38 | 1.38s / 0.74 | **1.35s / 0.74** | **−26%** | −2.5% |
| 60,000 | 1.20s / 0.74 | 1.34s / 0.74 | 1.36s / 0.74 | +13% | +1.2% |
| 80,000 | 1.22s / 0.74 | 1.32s / 0.74 | 1.35s / 0.74 | +11% | +2.9% |
| 120,000 | 1.12s / 0.74 | 1.27s / 0.75 | 1.31s / 0.74 | +17% | +3.4% |

## 4. 해석

- **park은 동작하며 hicache와 대등하다.** 두 tier가 모든 pool에서 겹쳐 움직인다(차이 ~1–3.5%).
  강압박에선 park이 근소하게 빠르고(GPU-P2P fetch vs host-DRAM load), 무압박에선 근소하게 느리다
  (park은 완료된 요청을 매번 GPU2로 복사하는 상시 오버헤드). → **유휴 GPU를 host DRAM 대신 L2로
  써도 동등**하다는 end-to-end 증명.
- **복구 tier의 경제학은 압박-조건부다.** 축출이 있어야(pool 40k) 되찾을 게 있어 이득이 나고,
  없으면(pool≥60k) park·hicache 모두 순수 오버헤드로 radix보다 느리다. → 실전에선 **압박 감지 시에만
  tier를 켜는 적응형**이 맞다.
- **park의 *우위*(단순 대등이 아니라)가 나려면**: (a) GPU2가 P와 **NVLink-paired**(전송 2×), 또는
  (b) **host DRAM이 부족/경합**해 유휴 GPU 용량이 실제로 필요한 상황. 이 단일 노드·PCIe 셋업에선 동률.

## 5. 정정 이력 (정직성)

초기엔 park이 무이득으로 보였고 "catch-22(압박 땐 복구할 자리가 없다)라 근본적으로 안 된다"고
결론냈으나, **그 결론은 틀렸다.** 두 가지가 파킹을 가리고 있었다:
1. **evict-to-room 부재** — fetch가 P GPU 자리를 못 잡으면(DIAG `nospace 32%`) 포기했다. hicache는
   콜드 엔트리를 LRU-evict해 자리를 만든다. 같은 로직을 park에 추가하니 nospace가 사라졌다.
2. **stale-IPC rendezvous 버그** — prefill이 이전 run이 남긴 `/dev/shm`의 죽은 decode IPC 핸들을
   열어 CUDA "invalid resource handle" → 파킹 setup이 조용히 죽어 reuse가 radix로 되돌아갔다.
   fresh-ts 검증 + 시작 시 rendezvous 정리로 해결.

둘을 고치자 park이 설계대로(hicache처럼) 동작했다. **교훈: "안 된다"는 종종 미구현/버그이지 근본
한계가 아니다.**

## 6. 결론

- **spare-GPU KV parking은 강압박 multi-turn PD에서 축출된 prefix를 되찾아 radix 대비 TTFT −26%**,
  성숙한 host-DRAM hicache와 **대등**하다 — end-to-end 검증 완료.
- **복구 tier는 압박-조건부 이득**이다(무압박선 오버헤드). park의 host 대비 *우위*는 NVLink-paired
  유휴 GPU 또는 host-DRAM 부족 환경에서 실현된다.
- 파이프라인(2a IPC/P2P → 3 park → 4a 유휴풀 → 4b fetch+evict-to-room+async)은 재사용 자산.

---

### 부록 — 재현

```bash
cd ~/experiments
POOLS="40000 60000 80000 120000" ./scripts/sglang/run_head_to_head_pool_sweep.sh
python benchmark/plot_phase1_catch22.py     # 그림 2장 재생성
```
강압박 park DIAG(성공 예): `opened peer KV pool ... MATCH 52.8 GB/s` → `GPU2-park` → `GPU2-fetch:
pulled 5355 tok (P had 87 of 5442) in 50.6ms`(첫 fetch) → 다음 fetch 3.1ms(async). nospace≈0.
상세: `phase1.md`, `research.md` §2.16, SGLang `docs/developer_guide/idle_kv_parking_design.md` §9–§12.

---

## 7. 확장 — 생성-KV 파킹은 워크로드 의존적 (BFCL vs ShareGPT)

park를 prompt prefix뿐 아니라 **decode-생성 KV(assistant 응답)까지** 저장하도록 확장하고
(`SGLANG_KV_PARK_GEN`), 다음 turn이 그 응답 KV를 재계산 대신 fetch하는지 측정했다. 무압박
(pool 120k)에서 재면 축출 효과가 없어 **생성-KV 기여만 isolate**된다 (GEN=0 = prefix만).

**7-1. 워크로드 의존성 (BFCL vs ShareGPT)** — 생성-KV 재사용은 assistant 응답이 길고 verbatim일 때만 큼:

| 워크로드 | park GEN=0 reuse | park GEN=1 reuse | Δreuse |
|---|---|---|---|
| **BFCL** (tool-call, 짧은 응답) | 0.744 | 0.750 | **+0.6pp** (미미) |
| **ShareGPT** (chat, 긴 응답) | 0.618 | **0.932** | **+31.4pp** (큼) |

BFCL은 응답이 짧은 tool-call + 재직렬화 토큰 불일치로 기여 미미. ShareGPT는 assistant 응답이
길고 평문 verbatim으로 다음 turn에 재등장 → decode 생성 KV를 다음 prefill이 통째로 재사용.

**7-2. park(GEN=1) vs SGLang decode-offload — 3-arm (ShareGPT, pool 120k, 200 items)**:

| arm | reuse | recompute tok | L3 prefetched | **host RAM** | TTFT |
|---|---|---|---|---|---|
| radix | 0.618 | 344k | 0 | 22.4 GB | 0.229s |
| **park (GEN=1)** | **0.932** | 62k | 0 | **22.9 GB** (≈0 추가) | 0.236s |
| decode_offload | 0.622 | 343k | 3,064 | **105.9 GB** (+83 GB) | 0.245s |

- **park만 생성-KV 재사용을 실제로 잡는다 (0.93). decode_offload는 못 잡는다 (0.62 = radix).**
  decode_offload는 생성 KV를 host DRAM→disk(file, L3)로 offload하지만 **async best-effort prefetch가
  3s tool-delay 멀티턴 루프에서 제때 못 읽어옴** (L3_prefetched 3,064 토큰뿐) → 재계산. park는 GPU2
  상주 + 동기 fetch라 확실히 hit.
- **host RAM**: park ~0 추가 vs decode_offload **+83 GB**. park는 이득 0.93을 host RAM 없이, decode_
  offload는 83GB 태우고 이득 0.
- **차별점 확정**: park은 decode-offload가 의도한 "생성-KV 재사용"을 *실제로* 달성하고, **host RAM은 0**.

**7-3. 정직한 한계 — reuse 이득은 저부하에선 latent (TTFT로 안 나타남)**: park reuse 0.93 vs radix
0.62 차이가 **TTFT/throughput으로 안 나타난다** (park TTFT +3~4% vs radix, wash). pool 120k·40k 둘 다
동일 — **ShareGPT는 대화가 짧아(세션 ~4.5k, C=8 working set ~36k) pool 40k에서도 축출이 없다**;
pool로는 ShareGPT에 압박을 못 만든다. 그리고 C=8 + 3s tool-delay면 시스템이 저부하라, park가 아낀
prefill 연산(recompute 62k vs 344k = **82% 적음**)이 병목이 아니라 TTFT로 전환 안 됨. → park의 reuse
이득은 **"latent capacity"**(같은 GPU로 더 많이 서빙 가능)이지 저부하 latency 이득이 아니다. 실제
TTFT/throughput 이득을 보려면 **고동시성 + tool-delay=0로 prefill을 saturate**해야 한다(다음 실험).
(decode_offload 기본 설정의 저조는 prefetch policy 튜닝 여지 있으나 host RAM 소모는 불변.)

**7-4. Saturation 실험 (C=32, tool-delay=0, ShareGPT pool 120k) — reuse 0.92가 TTFT로 전환 *안* 됨**:
7-3에서 "고동시성이면 park의 reuse 이득이 TTFT로 전환될 것"이라 예측했으나, **실측은 반대였다.**

| arm | reuse | TTFT | Δ TTFT vs radix | overall tput | host RAM |
|---|---|---|---|---|---|
| radix | 0.617 | 0.349s | — | 881 tok/s | 12.9 GB |
| **park (GEN=1)** | **0.915** | **0.679s** | **+94% (악화)** | **841 tok/s (−4.6%)** | 13.2 GB |
| decode_offload | 0.620 | — | — | — | 96.4 GB |

park DIAG(C=32): `FETCH: hits=667(evict-to-room=116) tok=265,775 avg=3.4ms | miss=220 already=46 nospace=0 (of 933)`.

- **reuse는 여전히 0.92로 성공 (fetch 동작함)**. 그런데 **TTFT는 오히려 2배로 악화**, throughput은 소폭 하락.
- **원인 (avg=3.4ms와의 화해)**: `avg=3.4ms`는 **GPU2→GPU0 async copy 시간만** 잰다. 이 copy는
  `forward_stream.wait_stream`로 critical path 밖이라 문제 아님. 진짜 비용은 **매 prefill(933건)마다
  scheduler 단일 main-thread에서 동기로 도는** ①`_match_park_prefix`의 prefix 해시 탐색(파킹 길이별
  `hash(tuple(token_ids[:L]))`), ②alloc + **evict-to-room 116회(LRU 트리 워크)**, ③fetch 블록 radix
  insert(265k 토큰)다. 이건 3.4ms에 안 잡힌다.
- **핵심 해석**: 저부하에선 GPU-prefill-FLOPs도 scheduler-CPU도 병목이 아니라 park가 wash(7-3).
  **C=32/delay=0로 saturate하면 병목이 GPU-FLOPs → scheduler-CPU-throughput으로 이동**한다. park는
  **GPU-bound 비용(recompute)을 CPU-bound 비용(match/evict/insert)으로 바꾼다** — 그런데 그 CPU 작업이
  이미 포화된 바로 그 스레드에 얹혀서, 아낀 prefill FLOPs보다 더 비싸진다. → reuse 0.92는 진짜지만
  **serving-perf로는 순손실**.
- **결론 (정직)**: park의 생성-KV 확장(GEN=1)은 **① reuse 지표 승리(0.92 vs 0.62) + ② host RAM 0
  (decode_offload 83GB 대비)**의 이점은 확실하나, **저부하에선 latency 무변(wash), 고부하에선 오히려
  악화**다. 즉 현재 구현에서 생성-KV 재사용은 *serving latency/throughput 이득이 아니다.* park의 **견고한
  이득은 여전히 §1–§6의 "압박 하 prefix 복구"**(BFCL 40k, park −26% vs radix, hicache와 동률)에 있다.
- **개선 여지 (미검증)**: 위 순손실은 fetch 오케스트레이션(match/evict/insert)이 scheduler main-thread에
  직렬화되기 때문. 이를 **별도 스레드/더 싼 매칭(해시 캐시)** 으로 옮기면 고부하에서도 reuse가 TTFT로
  전환될 여지가 있다. 단, 그 전엔 park의 생성-KV 확장을 "성능 최적화"가 아니라 **"capacity/RAM-효율
  최적화"**로 규정하는 것이 데이터에 부합한다.
