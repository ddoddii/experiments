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

**Head-to-head (pool 40000, C=8, delay 3s, 200 items)**:

| arm | TTFT | throughput | reuse_ratio | vs radix | vs hicache |
|---|---|---|---|---|---|
| radix | 1.767s | 64.2 tok/s | **0.399** | — | |
| **hicache** ✅ | **1.333s** | 66.7 tok/s | **0.744** | −24.6% | 승자 |
| park (4a) | 1.856s | 62.4 tok/s | **0.390** | **+5.0%** | **+39.2% 느림** |

**해석 (예측 그대로)**:
- `park.reuse(0.390) ≈ radix.reuse(0.399)` ≠ hicache(0.744) → park은 KV를 GPU2에 **저장만**
  하고 prefill이 읽는 **fetch-on-hit이 없어 prefix-hit을 못 만든다.**
- park은 radix보다도 **+5% 느림** — D→GPU2 복사가 읽는 쪽 없는 **순손실**.
- **병목은 대역폭이 아니라 fetch 통합.** 통합된 fetch 경로를 가진 host-DRAM hicache가 명확히 승.

---

## 6. 종합 결론

**"유휴 자원으로 KV parking"의 두 축을 각각 검증했으나, 이 2×A6000(쌍별 NVLink) 토폴로지에선
한 경로에서 결합되지 않는다:**

| 전제 | 검증 결과 |
|---|---|
| ① 전송 싸다 (NVLink) | ✅ P↔D(GPU0-1) 52 GB/s — 단 NVLink **쌍**에만 |
| ② P에 여유 있다 | ⚠️ Design A는 압박 시 P 포화(alloc 88% 실패). 전용 GPU2로 용량은 100% survival 검증 |
| ③ 보존이 재계산 이긴다 | ❌ fetch-on-hit 미통합 → park.reuse ≈ radix, hicache에 −24.6% 뒤짐 |

- NVLink는 P-D에만, 여유 GPU는 PCIe로만 접근 → 전송 이점과 용량 이점이 만나지 않음.
- fetch(P GPU2→GPU0)를 붙여도 PCIe라 host-DRAM hicache 동률이 상한이고, 용량도 26GB < 125GB로 열위.
- **아이디어가 실익을 내려면**: all-to-all NVLink(NVSwitch/DGX), **여유 GPU가 P와 NVLink-paired인
  배치**, 혹은 **진짜 multi-node**(aggregate GPU memory ≫ single host).
- 파이프라인 코드(IPC / P2P gather-copy / radix insert, 슬라이스 2a~4a)는 그런 환경의 재사용 자산.

이 발견은 `research.md` §2.11/§2.12 motivation의 **정직한 경계조건**이기도 하다 — NVLink 이점과
P-idle 용량이 **동시에 성립하는 환경**을 전제로 한다.

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
