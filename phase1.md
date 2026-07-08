# Phase 1 실험 로그 — Idle KV Parking (D→P-HBM)

> **한 줄 결론**: multi-turn agentic PD disaggregation에서 tool-call 유휴시간에 decode(D) 노드의
> 세션 KV를 prefill(P) 노드 HBM에 보존(park)하는 아이디어를 SGLang에 구현·측정했다.
> **이 2×A6000(쌍별 NVLink) 하드웨어에서는 기존 host-DRAM hicache를 이기지 못했고**, 그 이유가
> 대역폭이 아니라 **fetch-on-hit 통합의 부재**임을 메커니즘까지 규명했다.
>
> 상위 연구 맥락은 [`research.md`](./research.md) §2.16, SGLang 측 설계·코드 상세는
> 브랜치 `claude/youthful-knuth-det52g`의 `docs/developer_guide/idle_kv_parking_design.md` §9–§10.

---

## 1. 목표와 가설

**아이디어**: PD disaggregation에서 D 노드는 turn 종료 시 KV를 즉시 free하고, P 노드는 LRU로
radix prefix를 evict한다. 압박(pressure) 하에서 세션 prefix가 대화 중간에 축출되면 이후 모든
turn이 전체 history를 재계산(re-prefill)해 TTFT가 폭증한다. → **tool-call 유휴시간 동안 축출
위험에 처한 세션 KV를 D→P 노드 HBM으로 보존(park)했다가, 다음 turn에 prefix-hit으로 재사용**하자.

**핵심 전제 (검증 대상)**:
1. **전송이 싸다** — P↔D가 NVLink로 연결 → D→P KV 전송이 PCIe·host-DRAM보다 빠르다.
2. **P에 여유가 있다** — D가 포화되는 순간에도 P HBM은 idle(§2.12: 83–90%).
3. **보존이 재계산을 이긴다** — parked KV를 다음 turn에 fetch하면 re-prefill을 피한다.

---

## 2. 실험 환경

| 항목 | 내용 |
|---|---|
| 서버 | server17, NVIDIA RTX A6000 × 4 (49GB each), RAM 125GB |
| 모델 | Llama-3.1-8B-Instruct |
| SGLang | 0.5.9 호환 (브랜치 `claude/youthful-knuth-det52g`), KV 전송 Mooncake/NIXL |
| 구성 | **1P1D**: Prefill=GPU0(30000), Decode=GPU1(30001), Router=8000 |
| 벤치마크 | BFCL v3 multi-turn base 200 items, `benchmark/sglang_BFCL_multi_turn_concurrent.py` |
| 압박 knob | `PREFILL_MAX_TOTAL_TOKENS`(P 풀 축소 → radix eviction 강제), `CONCURRENCY`, `TOOL_DELAY`(tool-call 유휴 모사) |
| NVLink 토폴로지 | `nvidia-smi topo -m`: NVLink 쌍은 **(0-1),(2-3)만**. GPU0↔GPU2 = PCIe(`NODE`) |

> **CUDA IPC 제약**: 각 프로세스가 상대 GPU를 "볼 수" 있어야 함 → `CUDA_VISIBLE_DEVICES` 격리
> 대신 두 GPU를 모두 보이게 하고 `--base-gpu-id`로 배치. `start_1P_1D.sh`의 `IDLE_KV_PARKING=1`
> 모드가 처리.

---

## 3. 구현물 (SGLang 수정)

| 파일 | 수정 내용 |
|---|---|
| `server_args.py` | `--disaggregation-enable-idle-kv-parking` 플래그 + PD-mode 검증 |
| `disaggregation/idle_kv_parking.py` (신규) | 핵심 모듈: D→P **CUDA IPC 역방향 채널**, ZMQ 제어 채널, P2P gather-copy, radix insert, 전용 GPU park 풀(4a), DIAG 계측 |
| `managers/scheduler.py` | `maybe_init_idle_kv_parking()` — P/D 양쪽에 매니저 구성 |
| `managers/scheduler_output_processor_mixin.py` | turn 종료(`req.finished()`) 시 decode가 `park(req)` 호출 |
| `disaggregation/prefill.py` | prefill event loop가 `poll_incoming()`으로 park 수신 |

**데이터 흐름**: D가 turn 종료 시 `token_ids + kv_indices`를 ZMQ로 P에 통보 → P가 IPC로 열린
D의 KV 버퍼에서 P2P gather-copy → P radix(Design A) 또는 전용 GPU 풀(4a)에 insert.

---

## 4. 슬라이스별 실험 로그

### 2a — 전송 채널 de-risk (NVLink cross-process P2P)
IPC 핸들 교환 + KV P2P read를 마이크로벤치로 먼저 검증.

| tokens | size | NVLink 쌍 P2P | host-DRAM 왕복(PCIe) | 비고 |
|---|---|---|---|---|
| 512 | 67MB | **52.3 GB/s** | 26.2 GB/s (park) | speedup 4× |
| 4096 | 537MB | **52.8 GB/s** | 26.4 GB/s | cross-proc IPC checksum MATCH |

→ **전제 ①(전송이 싸다) 검증** — 단 이는 NVLink **쌍(GPU0-1)** 한정. (`results/nvlink_microbench.json`,
`results/nvlink_xproc_microbench.json`)

### 2b — gather-copy 정확성
P2P로 읽은 KV를 P의 목적 슬롯에 인덱스 복사. cross-device 에러(`.to(local_dev)` 누락), race
(live 풀 슬롯이 warmup에 덮임 → dedicated race-free 테스트 버퍼로 해결) 수정 후 **checksum MATCH**.

### 3a–3d — 채널·park·수신 파이프라인
ZMQ PUSH/PULL 채널, `park(req)`(decode), `poll_incoming`(prefill, `peer_ready` gating),
`_receive_park`(예외 시 dst free). insert 200/200 정확. 메모리 릭·NoneType 크래시 수정.

### Design A 결과 (P GPU radix park) — **1P1D에서 실패**
파이프라인 정확성은 완비됐으나 **실이득 0**:

| (pool 40000, C=8, delay 3s) | reuse_ratio | avg TTFT | success |
|---|---|---|---|
| 파킹 OFF (A) | 0.392 | 1.848s | 200/200 |
| 파킹 ON (B) | 0.375 | 1.838s | 200/200 |

차이 = 노이즈. DIAG 계측으로 원인 확정:
```
recv=246 processed=30 | copy=30 avg-P-had=0.95 | survival=0%
```
1. **alloc 실패 88%**: 압박 상태에서 P KV 풀이 꽉 차 park 슬롯 할당 불가 (216/246 드롭).
2. **survival 0%**: 복사된 소수도 hit 전에 즉시 evict.
3. 무압박 시엔 `avg-P-had≈0.99` → P가 이미 prefix 보유 → 파킹은 생성 토큰(~5%)만 추가 = 무의미.

→ **근본 원인: "유휴 P GPU" 전제가 압박과 모순.** P가 evict할 만큼 압박받는 순간(파킹이 필요한
그때) P GPU엔 여유가 없다. Design A는 병목과 **동일 자원(P GPU)**을 노려 구조적으로 무효.
(전제 ②가 1P1D 압박 셋업에서 깨짐 — §2.12의 P-idle은 *자연 동시성*에서 측정된 반면, parking
유발엔 P 풀을 인위 축소해 eviction을 강제했다.)

### Slice 4a — 전용 유휴 GPU2 park 풀
Design A의 P-GPU 병목을 우회하려 별도 GPU2(200k tok = 26GB 풀)에 park. 두 실패모드 모두 제거:

| | Design A (P GPU) | 4a (GPU2 전용 풀) |
|---|---|---|
| 드롭(alloc-fail) | 88% (recv246→proc30) | **0%** (recv260→proc260) |
| survival | 0% | **100% (32/32)** |

→ **전제 ②(용량) 검증 완료.** 그러나 GPU0(P)↔GPU2(park) = **PCIe** (NVLink 쌍은 0-1,2-3뿐)
→ park↔fetch 경로가 host-DRAM과 동일 속도. NVLink 이점(전제 ①)과 유휴 용량(전제 ②)이
**한 경로에서 결합되지 않는다.**

---

## 5. Head-to-head — 결정 실험

남은 질문("park가 host-DRAM hicache 대비 실제로 이득이 있는가")을 **3개 arm 동일 세션
back-to-back**(cross-run drift 제거)으로 못박음. `scripts/sglang/run_head_to_head.sh`.

**Phase 0 baseline (같은 압박, 참고)** — host-DRAM(L2)이 evict된 prefix를 담아 되가져와 reuse를
**2.9× 끌어올린다** = 이겨야 할 incumbent:

| mode | reuse_ratio | recompute tok | avg TTFT |
|---|---|---|---|
| radix (GPU only) | 0.257 | 2.85M | 2.40s |
| **hicache_host** | **0.743** | 0.98M | **1.45s** |

**Slice 4b — fetch-on-hit 추가**: 4a는 GPU2에 저장만 했다(reuse=radix, 무이득). "PCIe라도
recompute보다 빠르다"는 가설을 실측하려고 **읽기 경로**를 구현했다: prefill이 요청을 큐에 넣기
직전(`scheduler._add_request_to_queue`), `maybe_fetch(req)`가 요청의 최장 parked prefix를 찾아
GPU2→GPU0로 복사 + radix insert → 다음 `match_prefix`가 prefix-hit. (초기 버그: `origin_input_ids +
output_ids`를 key로 저장 → tool-call turn에서 재렌더링과 어긋나 거의 miss. **fix: 프롬프트
`origin_input_ids`만 park** — 다음 turn의 token-exact prefix.)

**Head-to-head + 압박 완화 스윕 (C=8, delay 3s, 200 items)**:

| pool | radix TTFT / reuse | hicache TTFT / reuse | park(4b) TTFT / reuse | park vs radix |
|---|---|---|---|---|
| 40000 (강압박) | 1.83 / **0.39** | 1.38 / **0.74** | 1.83 / 0.45 | +0.2% |
| 60000 | 1.27 / **0.74** | 1.38 / 0.74 | 1.19 / 0.74 | −6.3%\* |
| 80000 | 1.17 / **0.74** | 1.38 / 0.75 | 1.19 / 0.74 | +1.0% |
| 120000 | 1.15 / **0.74** | 1.27 / 0.75 | 1.15 / 0.74 | +0.4% |

\* pool-60000 −6.3%는 radix-60000 TTFT outlier에 의한 착시(reuse 동일 → fetch 무의미). 노이즈.

**pool 40000 DIAG**: `FETCH: hits=26 miss=206 already=278 nospace=237 (of 747), avg=235.8ms`.
fetch는 실제로 동작(reuse 0.39→0.45, +217k 토큰 fetch)하나 성공률 3.5%. **nospace 32%**: fetch한
KV는 attention이 읽으려면 **압박받는 P GPU 풀에 다시 넣어야** 하는데 pool 40000이 꽉 차 실패.

---

## 6. 종합 결론 — catch-22 (idle KV parking은 단일 노드에서 실익 없음)

전송(2a)·용량(4a)·저장(3)·**fetch(4b)** 파이프라인을 정확성까지 완비했으나, park+fetch는 **어떤
operating point에서도 radix(recompute)를 유의미하게 이기지 못했다.** 원인은 구현 디테일이 아니라
구조적 **catch-22**:

1. **파킹이 가치 있는 구간 = 강압박(pool 40000)뿐** — 여기서만 radix가 evict해 reuse 0.39로
   떨어지고 재계산이 발생. **그런데 이 구간은 nospace 32%** — fetch한 KV가 병목 P GPU에 자리를 못 잡음.
2. **restore 자리가 있는 구간 = pool ≥60000** — 그런데 여기선 radix가 evict를 안 해 reuse가 이미
   0.74 = hicache와 동일 → **되찾을 게 없음.** 세 arm 모두 reuse 0.74로 수렴.
3. → **"restore가 가치 있다"(evict)와 "restore가 자리 있다"(P 여유)가 결코 공존하지 않는다.**

**근본 이유**: 저장은 유휴 GPU로 offload되지만, **fetch한 KV는 attention이 읽으려면 병목 P GPU를
점유해야 한다.** 이 restore-병목은 **transfer 속도(NVLink)와 무관 = 토폴로지 독립적.** hicache가
강압박에서 이기는 건(0.74) 같은 제약을 **evict-to-room + async**로 관리하기 때문이며, park을 그렇게
엔지니어링해도 이 토폴로지선 GPU2→GPU0=PCIe(host DRAM 동속)·26GB<125GB라 **hicache 재현이 상한.**
(부가: pool ≥60000에선 evict가 없어 hicache조차 순수 오버헤드로 radix보다 +8~10% 느림.)

| 전제 | 검증 결과 |
|---|---|
| ① 전송 싸다 (NVLink) | ✅ P↔D(GPU0-1) 52 GB/s — 단 NVLink **쌍**에만 |
| ② P에 여유 있다 | ⚠️ 저장은 유휴 GPU2로 해결(survival 100%), 그러나 **restore는 병목 P GPU 점유(nospace 32%)** |
| ③ fetch가 재계산 이긴다 | ⚠️ reuse 레벨은 참(0.39→0.45)이나 **순 TTFT 무승부** — catch-22로 회수량 제한 |

**아이디어가 실익을 내려면**: (1) attention이 remote KV를 직접 읽는 **disaggregated/remote
attention**(restore가 P GPU를 점유 안 해도 됨), 또는 (2) host DRAM이 유일 로컬 tier이고 원격 GPU
합산 용량이 host를 압도하는 **진짜 multi-node.** 단일 노드·단일 GPU-tier로는 hicache가 이미 상한.
파이프라인 코드(2a~4b: IPC/P2P/park/fetch)는 그런 환경의 재사용 자산으로 남긴다.

이 발견은 `research.md` §2.11/§2.12 motivation의 **정직한 경계조건**이다.

---

## 7. 재현 방법

```bash
cd ~/experiments
# 3-arm(radix/hicache/park) 자동 순회: start → /metrics before → BFCL → after → delta → stop
PREFILL_MAX_TOTAL_TOKENS=40000 CONCURRENCY=8 TOOL_DELAY=3 \
  ./scripts/sglang/run_head_to_head.sh
# 결과 표 + 판정:
#   results/head_to_head/h2h_p40000_c8_d3/head_to_head_summary.json
```

개별 arm 수동 실행:
```bash
# park arm (radix base + 전용 유휴 GPU2 풀)
CACHE_MODE=radix IDLE_KV_PARKING=1 PARK_GPU=2 PARK_POOL_TOKENS=200000 \
  PREFILL_MAX_TOTAL_TOKENS=40000 ./scripts/sglang/start_1P_1D.sh
```

---

## 8. 산출물

| 파일 | 내용 |
|---|---|
| `scripts/sglang/run_head_to_head.sh` | 3-arm head-to-head 러너 |
| `benchmark/head_to_head_analyze.py` | head-to-head 표 + `park.reuse≈radix` 판정 |
| `benchmark/nvlink_cross_process_p2p_microbench.py`, `nvlink_kv_transfer_microbench.py` | 슬라이스 2a NVLink/IPC 마이크로벤치 |
| `results/head_to_head/h2h_p40000_c8_d3/` | head-to-head 실측 + reuse delta |
| `results/bfcl_multiturn_results_{A_nopark,B_park}.json` | Design A 파킹 OFF/ON 대조 |
| `results/nvlink_microbench.json`, `results/nvlink_xproc_microbench.json` | NVLink 52 vs PCIe 26 GB/s, IPC 검증 |
| SGLang `python/sglang/srt/disaggregation/idle_kv_parking.py` | 파킹 핵심 모듈 (브랜치 `youthful-knuth-det52g`) |
| SGLang `docs/developer_guide/idle_kv_parking_design.md` | 설계·구현·Phase1 결론 §9–§10 |
